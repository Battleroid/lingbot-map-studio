"""HTTP surface the remote worker uses to talk to the studio.

Every endpoint mirrors one method on `JobSource` + `HttpJobSource`. The
pair form a closed loop: whatever the local runner does by poking
`store` / `bus` / the shared volume, a remote runner does by hitting
these endpoints.

All endpoints require a per-job HMAC token minted by the dispatcher
when the job was launched. The token carries:
  * `job_id` — every endpoint's operations resolve against this
  * `execution_target` — informational (e.g. "runpod")
  * `scopes` — set of operations this token may perform

The token's `jid` is the only source of truth for which job the
caller is operating on — URL parameters never carry `{job_id}` for
authorised operations. That way a leaked token can only touch its
own job, not be retargeted by spraying the URL.

`/api/worker/checkpoints/*` is the one exception: those are
job-agnostic (model weights cached in `/models/checkpoints/...`), so
it only checks scope + that the token is still valid.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from app.cloud import storage as storage_mod
from app.cloud import tokens as tokens_mod
from app.config import settings
from app.jobs import store
from app.jobs.events import bus
from app.jobs.schema import Artifact, JobEvent, JobStatus, parse_job_config

log = logging.getLogger(__name__)

# Chunk size for streaming artifact uploads out of the request body and
# upload/checkpoint downloads back out. 1 MiB is a sweet spot: big enough
# that a 300 MB splat.ply upload finishes in ~300 iterations, small
# enough that a cancel during a chunk loses at most 1 MB of progress.
CHUNK_SIZE = 1 << 20


router = APIRouter(prefix="/api/worker", tags=["broker"])


# --- auth dependency ----------------------------------------------------


def _extract_bearer(authorization: Optional[str]) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="missing Authorization header")
    parts = authorization.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=401, detail="Authorization must be a Bearer token"
        )
    return parts[1].strip()


def require_scope(scope: str):
    """FastAPI dependency that validates the token and returns its payload.

    The handler gets the decoded `TokenPayload` so it can read the
    `job_id` without re-parsing; the scope check happens before the
    handler runs.
    """

    def _dep(authorization: Optional[str] = Header(None)) -> tokens_mod.TokenPayload:
        raw = _extract_bearer(authorization)
        try:
            return tokens_mod.verify(
                raw,
                key=settings.cloud_broker_hmac_key,
                required_scope=scope,
            )
        except tokens_mod.TokenError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    return _dep


# --- request / response schemas ----------------------------------------


class ClaimRequest(BaseModel):
    worker_class: str
    worker_id: str


class ClaimResponse(BaseModel):
    job_id: str
    config: dict[str, Any]
    uploads: list[str] = Field(default_factory=list)
    # Storage transport the remote worker must use when uploading
    # artifacts. `broker` = PUT /api/worker/artifacts/{name}; `minio` =
    # fetch a pre-signed PUT URL first via POST /artifacts/presign.
    # Picked by the studio on claim so a compromised pod can't opt
    # itself into a different upload path.
    storage_kind: str = "broker"


class PresignRequest(BaseModel):
    name: str


class PresignResponse(BaseModel):
    # Opaque transport-specific blob: the worker reads `mode` and
    # branches. See `cloud.storage.ArtifactStorage.presign_put`.
    upload: dict[str, str]


class CommitRequest(BaseModel):
    name: str


class CommitResponse(BaseModel):
    name: str
    size_bytes: int


class HeartbeatRequest(BaseModel):
    worker_id: str


class TerminalRequest(BaseModel):
    """Atomic finalize after the processor returns (or after the runner's
    error handler has built its message). Either `status="ready"` with
    artifacts, or a terminal failure status with `error`."""

    status: JobStatus
    error: Optional[str] = None
    artifacts: list[Artifact] = Field(default_factory=list)
    worker_id: Optional[str] = None
    release: bool = True


class SetStatusRequest(BaseModel):
    status: JobStatus


class SetErrorRequest(BaseModel):
    """Record an error message without advancing status or closing the bus.

    The runner's error path does `set_error` + `set_status(failed|cancelled)`
    + `publish_event(...)` in order. If `set_error` rode `/terminal`, the
    bus would close before the trailing event landed. Kept separate for
    the same reason `/artifacts_manifest` is.
    """

    error: str


class SetFramesTotalRequest(BaseModel):
    frames_total: int


class SetArtifactsRequest(BaseModel):
    """Manifest-only artifact update.

    Bytes land via `PUT /artifacts/{name}` ahead of this call. This
    endpoint records the list on the job row without advancing status,
    closing the event stream, or releasing the claim — those steps
    stay on `/status` / `/terminal`. Kept separate from `/terminal`
    so mid-run manifest updates (e.g. a processor that reports its
    final artifact list before its terminal event) don't trigger
    `bus.close` and cut live WS tailers off early.
    """

    artifacts: list[Artifact] = Field(default_factory=list)


# --- endpoints ---------------------------------------------------------


@router.post("/claim", response_model=ClaimResponse)
async def claim(
    body: ClaimRequest,
    token: tokens_mod.TokenPayload = Depends(require_scope("claim")),
) -> ClaimResponse | Response:
    """Try to claim the token's bound job.

    Unlike the local `store.claim_next_job` (which scans the whole queue),
    the broker path is deliberately narrow: the dispatcher minted this
    token *for* one job, so if that job isn't claimable we return empty
    rather than handing over someone else's work.
    """
    job = await store.get_job(token.job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status != "queued":
        # Already picked up by another worker (or retried), or finished.
        return Response(status_code=204)

    # Re-run the atomic claim against exactly this row. `claim_next_job`'s
    # filter (status=queued + worker_class + claimed_by IS NULL) still
    # guarantees no double-claim; we just skip the queue-scanning part.
    claim = await store.claim_next_job(body.worker_class, worker_id=body.worker_id)
    if claim is None or claim[0] != token.job_id:
        # The row was for a different worker_class, or someone else beat
        # us to it. If we accidentally claimed a different job, release
        # it immediately so it stays available.
        if claim is not None and claim[0] != token.job_id:
            await store.release_job(claim[0], worker_id=body.worker_id)
        return Response(status_code=204)

    _job_id, config, upload_names = claim
    return ClaimResponse(
        job_id=_job_id,
        config=json.loads(config.model_dump_json()),
        uploads=upload_names,
        storage_kind=storage_mod.active_storage().kind,
    )


@router.get("/uploads/{name}")
async def fetch_upload(
    name: str,
    token: tokens_mod.TokenPayload = Depends(require_scope("uploads")),
) -> StreamingResponse:
    """Stream a previously-uploaded source clip down to the remote worker."""
    # Forbid path-traversal — the token binds us to a job, but the
    # filename field is attacker-controlled if the token leaks.
    if "/" in name or ".." in name or name.startswith("."):
        raise HTTPException(status_code=400, detail="invalid upload name")
    path = settings.job_uploads(token.job_id) / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="upload not found")
    return StreamingResponse(
        _file_iter(path),
        media_type="application/octet-stream",
        headers={"Content-Length": str(path.stat().st_size)},
    )


@router.get("/checkpoints/{processor_id}/{filename}")
async def fetch_checkpoint(
    processor_id: str,
    filename: str,
    _token: tokens_mod.TokenPayload = Depends(require_scope("checkpoints")),
) -> StreamingResponse:
    """Proxy a file from the studio's `/models/checkpoints/` cache.

    The studio has already run `ensure_checkpoint` locally (or is about
    to) so the file is on disk; this just copies bytes through. Remote
    workers cache the result on the rented pod so they only fetch once
    per pod lifetime.
    """
    for part in (processor_id, filename):
        if "/" in part or ".." in part or part.startswith("."):
            raise HTTPException(status_code=400, detail="invalid checkpoint path")
    path = settings.models_dir / "checkpoints" / processor_id / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="checkpoint not found")
    return StreamingResponse(
        _file_iter(path),
        media_type="application/octet-stream",
        headers={"Content-Length": str(path.stat().st_size)},
    )


@router.post("/events")
async def publish_events(
    events: list[JobEvent],
    token: tokens_mod.TokenPayload = Depends(require_scope("events")),
) -> dict[str, Any]:
    """Append a batch of events to the job's events.jsonl.

    The remote worker is expected to stamp `job_id` on each event
    before sending, but we override it with the token's jid to stop a
    leaked token from posting events into someone else's stream.
    """
    count = 0
    for ev in events:
        ev.job_id = token.job_id
        await bus.publish(ev)
        count += 1
    return {"published": count}


@router.put("/artifacts/{name}")
async def put_artifact(
    name: str,
    request: Request,
    content_length: Optional[int] = Header(None),
    token: tokens_mod.TokenPayload = Depends(require_scope("artifacts")),
) -> dict[str, Any]:
    """Stream an artifact file into the job's artifacts dir.

    Writes to a sibling `.part` file and renames on successful close so
    the frontend's cache-busting loader never sees a half-written file.
    Partial snapshots (`partial_NNN.ply`, `partial_splat_NNN.ply`) use
    the same endpoint; the rename makes each snapshot atomically visible.
    """
    if "/" in name or ".." in name or name.startswith("."):
        raise HTTPException(status_code=400, detail="invalid artifact name")

    artifacts_dir = settings.job_artifacts(token.job_id)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    final_path = artifacts_dir / name
    part_path = artifacts_dir / (name + ".part")

    written = 0
    with part_path.open("wb") as fh:
        async for chunk in request.stream():
            if not chunk:
                continue
            fh.write(chunk)
            written += len(chunk)
    part_path.replace(final_path)

    if content_length is not None and content_length != written:
        log.warning(
            "artifact %s truncated: got %d bytes, expected %d",
            name,
            written,
            content_length,
        )
    return {"name": name, "size_bytes": written}


@router.post("/artifacts/presign", response_model=PresignResponse)
async def presign_artifact(
    body: PresignRequest,
    token: tokens_mod.TokenPayload = Depends(require_scope("artifacts")),
) -> PresignResponse:
    """Vend an upload target for an artifact.

    For `broker` storage the returned `upload` dict just tells the
    worker to PUT to the usual `/api/worker/artifacts/{name}` path;
    for `minio` it's a pre-signed PUT URL against the configured bucket.
    The worker uploads per the returned shape then calls `/commit`.

    Kept in the same `artifacts` scope as `PUT /artifacts/{name}` so
    swapping the storage backend doesn't rotate token shapes.
    """
    if "/" in body.name or ".." in body.name or body.name.startswith("."):
        raise HTTPException(status_code=400, detail="invalid artifact name")
    upload = await storage_mod.active_storage().presign_put(token.job_id, body.name)
    return PresignResponse(upload=upload)


@router.post("/artifacts/commit", response_model=CommitResponse)
async def commit_artifact(
    body: CommitRequest,
    token: tokens_mod.TokenPayload = Depends(require_scope("artifacts")),
) -> CommitResponse:
    """Finalize an uploaded artifact.

    For `broker` storage this is a pass-through that just verifies the
    file landed (the PUT handler already wrote it). For `minio` it
    pulls the uploaded object into the local artifacts dir so the
    viewer can read it, then deletes the remote copy.
    """
    if "/" in body.name or ".." in body.name or body.name.startswith("."):
        raise HTTPException(status_code=400, detail="invalid artifact name")
    try:
        size = await storage_mod.active_storage().commit(token.job_id, body.name)
    except storage_mod.StorageError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return CommitResponse(name=body.name, size_bytes=size)


@router.get("/cancel")
async def get_cancel(
    token: tokens_mod.TokenPayload = Depends(require_scope("cancel")),
) -> dict[str, Any]:
    requested = await store.is_cancel_requested(token.job_id)
    return {"cancel_requested": requested}


@router.post("/heartbeat")
async def heartbeat(
    body: HeartbeatRequest,
    token: tokens_mod.TokenPayload = Depends(require_scope("heartbeat")),
) -> dict[str, Any]:
    await store.heartbeat(token.job_id, worker_id=body.worker_id)
    return {"ok": True}


@router.post("/status")
async def set_status(
    body: SetStatusRequest,
    token: tokens_mod.TokenPayload = Depends(require_scope("terminal")),
) -> dict[str, Any]:
    await store.update_job(token.job_id, status=body.status)
    return {"ok": True}


@router.post("/frames_total")
async def set_frames_total(
    body: SetFramesTotalRequest,
    token: tokens_mod.TokenPayload = Depends(require_scope("terminal")),
) -> dict[str, Any]:
    await store.update_job(token.job_id, frames_total=body.frames_total)
    return {"ok": True}


@router.post("/artifacts_manifest")
async def set_artifacts_manifest(
    body: SetArtifactsRequest,
    token: tokens_mod.TokenPayload = Depends(require_scope("terminal")),
) -> dict[str, Any]:
    await store.update_job(token.job_id, artifacts=body.artifacts)
    return {"ok": True, "count": len(body.artifacts)}


@router.post("/error")
async def set_error(
    body: SetErrorRequest,
    token: tokens_mod.TokenPayload = Depends(require_scope("terminal")),
) -> dict[str, Any]:
    await store.update_job(token.job_id, error=body.error)
    return {"ok": True}


@router.post("/close_events")
async def close_events(
    token: tokens_mod.TokenPayload = Depends(require_scope("terminal")),
) -> dict[str, Any]:
    """Write the `events.done` breadcrumb so WS tailers exit.

    The runner's `finally` block always closes events after the last
    event lands. `/terminal` closes too, but the runner does not
    always hit `/terminal` — on the error path it uses
    `/status` + `/error` + a trailing event. Having a dedicated close
    keeps the WS-close behaviour symmetric between local and remote
    runs without forcing a terminal-shaped call.
    """
    await bus.close(token.job_id)
    return {"ok": True}


@router.post("/terminal")
async def terminal(
    body: TerminalRequest,
    token: tokens_mod.TokenPayload = Depends(require_scope("terminal")),
) -> dict[str, Any]:
    """Atomic finalize: optionally set artifacts, set status, set error,
    release the claim, close the event stream.

    Kept as one endpoint so the remote worker's `finally` block does a
    single HTTP call instead of four; cuts the happy-path latency on
    slow uplinks.
    """
    updates: dict[str, Any] = {"status": body.status}
    if body.error is not None:
        updates["error"] = body.error
    if body.artifacts:
        updates["artifacts"] = body.artifacts
    await store.update_job(token.job_id, **updates)

    if body.release and body.worker_id:
        try:
            await store.release_job(token.job_id, worker_id=body.worker_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("release failed for %s: %s", token.job_id, exc)

    await bus.close(token.job_id)
    return {"ok": True}


# --- helpers -----------------------------------------------------------


def _file_iter(path: Path):
    """Iterator that streams a file as binary chunks.

    FastAPI's StreamingResponse prefers a sync iterator when the
    underlying I/O is synchronous; a 1 MiB chunk keeps disk reads
    coalesced without blocking the event loop for noticeable stretches.
    """

    def _inner():
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(CHUNK_SIZE)
                if not chunk:
                    return
                yield chunk

    return _inner()
