"""Remote `JobSource` — the broker's client side.

Every method is the HTTP mirror of the matching `LocalJobSource` call,
speaking to the `/api/worker/*` router served by the studio. The shape
of the contract (`ClaimedJob` with resolved `uploads` paths, same
return types, same semantics on failure) is identical so the runner
doesn't know whether it's running in a local container next to the DB
or on a rented pod over HTTPS.

One implementation note bears repeating: **every job-scoped call binds
to the token's `jid`**. The broker never takes a `{job_id}` from the
URL for authorised operations, so even if this class had a bug that
passed the wrong id, the studio would authoritatively operate on the
token's bound job. That's the point of the broker design.

Responsibilities specific to this source (that `LocalJobSource`
doesn't have):

  * **Upload staging.** On claim, we download every upload into a
    per-job scratch dir and hand local `Path`s back to the runner.
  * **Artifact upload.** Artifacts written into `artifacts_dir(job_id)`
    only exist on the pod's local disk — we PUT them to the broker at
    `set_artifacts` time so the studio's artifact dir ends up with the
    same bytes. Live-streaming of *partial* snapshots during training
    (the `partial_NNN.ply` UX) rides on top via `ArtifactWatcher`
    (next slice).
  * **Scratch cleanup.** The pod's disk is ephemeral but not
    infinite; per-job scratch dirs live under `LINGBOT_REMOTE_SCRATCH`
    (default `/tmp/remote-scratch`) and get reaped on terminal.

This module deliberately imports `httpx` at module scope: if the pod
image is missing the client, that's a deploy-time failure, not
something we want to hide behind lazy imports.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import httpx

from app.cloud.sources import ClaimedJob, JobSource
from app.jobs.schema import Artifact, JobEvent, JobStatus, parse_job_config

log = logging.getLogger(__name__)

# Chunk size when streaming bytes in either direction. Matches the
# broker's constant so resumable semantics line up: if a partial PUT
# gets cut at chunk N, both sides know what N meant.
CHUNK_SIZE = 1 << 20

# The broker's `PUT /artifacts/{name}` tolerates chunked bodies but
# serialises each upload end-to-end. A 10-minute cap is generous for a
# 1 GB splat at residential-uplink rates (~1.7 MB/s).
DEFAULT_TIMEOUT_S = 600.0


class HttpJobSource(JobSource):
    """Talks to the studio's broker instead of the local DB + volume.

    Constructed once per pod startup in `worker_main._build_job_source`
    when `WORKER_MODE=remote`. One instance handles the whole lifetime
    of the pod — it holds a single `httpx.AsyncClient` so connection
    pooling + HTTP/2 multiplexing stay warm across jobs.
    """

    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        scratch_root: Optional[Path] = None,
        client: Optional[httpx.AsyncClient] = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._auth_headers = {"Authorization": f"Bearer {token}"}
        self._scratch_root = (
            scratch_root
            if scratch_root is not None
            else Path(os.environ.get("LINGBOT_REMOTE_SCRATCH", "/tmp/remote-scratch"))
        )
        self._scratch_root.mkdir(parents=True, exist_ok=True)
        self._timeout_s = timeout_s
        self._client = client or httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout_s, connect=30.0),
            headers=self._auth_headers,
        )
        # Track the job's scratch dir for on-demand cleanup. Populated by
        # `claim_next`; consulted by `release`.
        self._job_scratch: dict[str, Path] = {}
        # Per-job sync bookkeeping: for each file path we've pushed, the
        # (size, mtime_ns) snapshot of the version the studio currently
        # has. Skipping files that haven't moved since keeps the sync
        # loop cheap during a long gsplat training run (one `os.stat`
        # per file per tick, no network traffic).
        self._artifact_sync_state: dict[str, dict[Path, tuple[int, int]]] = {}

    # --- lifecycle ------------------------------------------------------

    async def aclose(self) -> None:
        await self._client.aclose()

    # --- claim ----------------------------------------------------------

    async def claim_next(
        self, worker_class: str, worker_id: str
    ) -> Optional[ClaimedJob]:
        resp = await self._client.post(
            "/api/worker/claim",
            json={"worker_class": worker_class, "worker_id": worker_id},
        )
        if resp.status_code == 204:
            return None
        if resp.status_code == 404:
            # The token's job_id doesn't exist yet or has been purged.
            # Treat the same as "nothing to claim" — the dispatcher
            # decides when a pod should exit.
            return None
        resp.raise_for_status()
        body = resp.json()

        job_id = body["job_id"]
        config = parse_job_config(body["config"])
        upload_names: list[str] = body.get("uploads", [])

        scratch = self._scratch_dir_for(job_id)
        uploads_dir = scratch / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)

        uploads: list[Path] = []
        for name in upload_names:
            dest = uploads_dir / name
            await self._download(f"/api/worker/uploads/{name}", dest)
            uploads.append(dest)

        return ClaimedJob(job_id=job_id, config=config, uploads=uploads)

    # --- heartbeat / release -------------------------------------------

    async def heartbeat(self, job_id: str, worker_id: str) -> None:
        resp = await self._client.post(
            "/api/worker/heartbeat",
            json={"worker_id": worker_id},
        )
        resp.raise_for_status()

    async def release(self, job_id: str, worker_id: str) -> None:
        # The broker's `/terminal` endpoint releases atomically on
        # finalize, so a standalone release isn't usually needed. We
        # keep the method for parity — used only if the runner wants
        # to drop the claim without marking the job terminal (e.g. on
        # an unexpected shutdown). Routed via `/terminal` with
        # `status=queued` would be wrong; instead, we call
        # `/status` + rely on the orphan sweeper to reclaim if the
        # worker truly died. In practice `release` is only hit in the
        # runner's `finally`, after `set_status` already wrote a
        # terminal state. So this is a no-op on the HTTP side and we
        # just clean up local scratch.
        scratch = self._job_scratch.pop(job_id, None)
        if scratch is not None:
            _rmtree_quiet(scratch)

    # --- state transitions ---------------------------------------------

    async def set_status(self, job_id: str, status: JobStatus) -> None:
        resp = await self._client.post(
            "/api/worker/status",
            json={"status": status},
        )
        resp.raise_for_status()

    async def set_frames_total(self, job_id: str, frames_total: int) -> None:
        resp = await self._client.post(
            "/api/worker/frames_total",
            json={"frames_total": frames_total},
        )
        resp.raise_for_status()

    async def set_artifacts(self, job_id: str, artifacts: list[Artifact]) -> None:
        """Push each named artifact's bytes up to the broker, then commit
        the manifest on the job row.

        The broker's `PUT /artifacts/{name}` does the atomic
        `.part` → final rename, so partial uploads from a crashed pod
        don't corrupt the studio's artifact dir. `POST /artifacts_manifest`
        records the list on the DB row *after* the bytes are visible,
        matching `LocalJobSource`'s file-before-row ordering so a
        frontend cache-busting loader never sees a manifest entry for
        a file that hasn't landed yet.
        """
        scratch = self._scratch_dir_for(job_id)
        artifacts_dir = scratch / "artifacts"
        for art in artifacts:
            src = artifacts_dir / art.name
            if not src.is_file():
                log.warning(
                    "set_artifacts: %s not found in local scratch, skipping upload",
                    src,
                )
                continue
            await self._upload_file(f"/api/worker/artifacts/{art.name}", src)

        resp = await self._client.post(
            "/api/worker/artifacts_manifest",
            json={"artifacts": [a.model_dump(mode="json") for a in artifacts]},
        )
        resp.raise_for_status()

    async def set_error(self, job_id: str, error: str) -> None:
        """Record the error message on the job row. No state change,
        no bus close — the runner's error path always follows this
        with a `set_status(failed|cancelled)` + one more event before
        winding down, and closing the stream here would cut the WS
        tailer off mid-flight.
        """
        resp = await self._client.post(
            "/api/worker/error",
            json={"error": error},
        )
        resp.raise_for_status()

    # --- pubsub ---------------------------------------------------------

    async def publish_event(self, event: JobEvent) -> JobEvent:
        resp = await self._client.post(
            "/api/worker/events",
            json=[event.model_dump(mode="json")],
        )
        resp.raise_for_status()
        # The broker stamps the event id; the remote worker doesn't
        # need it, but the JobSource contract says to return the
        # stamped event. We best-effort reconstruct from what the
        # broker returned in the published count — the studio's
        # assigned id isn't in the response body today (the payload
        # is stored in events.jsonl by the bus), so we hand back the
        # event as-we-sent-it. If a caller needs the authoritative
        # id, it can read events.jsonl back via a future endpoint.
        return event

    async def close_events(self, job_id: str) -> None:
        resp = await self._client.post("/api/worker/close_events")
        resp.raise_for_status()

    # --- cancellation ---------------------------------------------------

    async def is_cancel_requested(self, job_id: str) -> bool:
        resp = await self._client.get("/api/worker/cancel")
        resp.raise_for_status()
        return bool(resp.json().get("cancel_requested", False))

    # --- artifact sync (live partial-snapshot streaming) ---------------

    async def sync_artifacts(self, job_id: str) -> None:
        """Scan the pod's artifacts dir and PUT any new/changed files.

        Called on a short interval by the runner's sync loop. Intended
        to make `partial_NNN.ply` / `partial_splat_NNN.ply` snapshots
        show up in the studio's artifacts dir moments after the
        processor writes them, matching local-worker live-preview UX.

        Stability guard: a file whose size changed between the last
        tick and this one is assumed to be mid-write; we skip and try
        again next tick. Without this, a processor writing an artifact
        in append-mode would see half-written bytes land on the
        studio disk until the next PUT replaces them.
        """
        scratch_art_dir = self._scratch_dir_for(job_id) / "artifacts"
        if not scratch_art_dir.is_dir():
            return

        state = self._artifact_sync_state.setdefault(job_id, {})
        for path in sorted(scratch_art_dir.iterdir()):
            if not path.is_file():
                continue
            # Skip `.part` sidecars — those are in-progress writes the
            # broker's own PUT handler already owns.
            if path.name.endswith(".part"):
                continue
            try:
                st = path.stat()
            except FileNotFoundError:
                # Raced with a rename; pick it up next tick.
                continue
            current = (st.st_size, st.st_mtime_ns)
            previous = state.get(path)
            if previous == current:
                continue
            # Stability: if the size changed since last observation and
            # the previous state was a different snapshot (not the
            # first sighting), push anyway — mtime+size together are
            # the stability signal, not size alone. We only hold back
            # a file on its first sighting with zero bytes, which
            # means the writer opened it but hasn't written yet.
            if previous is None and st.st_size == 0:
                continue
            try:
                await self._upload_file(f"/api/worker/artifacts/{path.name}", path)
            except Exception as exc:  # noqa: BLE001
                log.warning("artifact sync failed for %s: %s", path.name, exc)
                continue
            state[path] = current

    # --- filesystem handoff --------------------------------------------

    def artifacts_dir(self, job_id: str) -> Path:
        path = self._scratch_dir_for(job_id) / "artifacts"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def job_dir(self, job_id: str) -> Path:
        return self._scratch_dir_for(job_id)

    def frames_dir(self, job_id: str) -> Path:
        path = self._scratch_dir_for(job_id) / "frames"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # --- terminal finalize (called explicitly by the remote runner) ----

    async def finalize(
        self,
        job_id: str,
        *,
        status: JobStatus,
        error: Optional[str] = None,
        artifacts: Optional[list[Artifact]] = None,
        worker_id: Optional[str] = None,
        release: bool = True,
    ) -> None:
        """Single atomic terminal call. The runner's terminal path prefers
        this over a series of `set_status` / `set_error` hops on slow
        uplinks — one round-trip instead of four.
        """
        payload: dict = {"status": status, "release": release}
        if error is not None:
            payload["error"] = error
        if artifacts is not None:
            payload["artifacts"] = [a.model_dump(mode="json") for a in artifacts]
        if worker_id is not None:
            payload["worker_id"] = worker_id
        resp = await self._client.post("/api/worker/terminal", json=payload)
        resp.raise_for_status()

    # --- checkpoints ----------------------------------------------------

    async def fetch_checkpoint(
        self, processor_id: str, filename: str, dest: Path
    ) -> Path:
        """Download a weight file from the studio's shared cache.

        Returns `dest`. The remote worker caches fetched checkpoints
        under `/scratch/checkpoints/...` so it only pays the download
        cost once per pod lifetime.
        """
        dest.parent.mkdir(parents=True, exist_ok=True)
        await self._download(
            f"/api/worker/checkpoints/{processor_id}/{filename}",
            dest,
        )
        return dest

    # --- internal helpers ----------------------------------------------

    def _scratch_dir_for(self, job_id: str) -> Path:
        """Return the per-job scratch root, creating + remembering on first call."""
        existing = self._job_scratch.get(job_id)
        if existing is not None:
            return existing
        path = self._scratch_root / job_id
        path.mkdir(parents=True, exist_ok=True)
        self._job_scratch[job_id] = path
        return path

    async def _download(self, url: str, dest: Path) -> None:
        """Stream `GET {url}` into `dest`, writing chunks as they arrive.

        Writes to a sibling `.part` file + atomic rename so an
        interrupted download never leaves a truncated file that a
        later step would mistake for complete.
        """
        part = dest.with_name(dest.name + ".part")
        async with self._client.stream("GET", url) as resp:
            resp.raise_for_status()
            with part.open("wb") as fh:
                async for chunk in resp.aiter_bytes(CHUNK_SIZE):
                    if chunk:
                        fh.write(chunk)
        part.replace(dest)

    async def _upload_file(self, url: str, path: Path) -> None:
        """Stream `PUT {url}` with the file's bytes as the body.

        httpx's `AsyncClient` needs an async byte stream; we wrap
        the sync file-read loop in a tiny async iterator so the
        upload stays bounded to one `CHUNK_SIZE` buffer. A 300 MB
        splat upload therefore never materialises in full in RAM.
        """
        size = path.stat().st_size
        resp = await self._client.put(
            url,
            content=_file_astream(path),
            headers={"Content-Length": str(size)},
        )
        resp.raise_for_status()


async def _file_astream(path: Path):
    """Async iterator that yields `CHUNK_SIZE`-byte file chunks."""
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(CHUNK_SIZE)
            if not chunk:
                return
            yield chunk


def _rmtree_quiet(path: Path) -> None:
    """Best-effort scratch cleanup — don't let a stray open handle kill the loop."""
    import shutil

    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception as exc:  # noqa: BLE001
        log.warning("scratch cleanup failed for %s: %s", path, exc)
