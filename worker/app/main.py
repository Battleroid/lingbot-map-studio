from __future__ import annotations

import asyncio
import json
import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import (
    Body,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.config import settings
from app.jobs import drafts, runner, store
from app.jobs.events import bus
from app.jobs.schema import (
    Artifact,
    Job,
    JobConfig,
    JobEvent,
    MeshEditRequest,
    ReexportRequest,
)
from app.mesh.ops import apply_op, mesh_summary
from app.pipeline.export import reexport
from app.pipeline.inference import load_cached_predictions
from app.pipeline.probe import probe_video, suggest_config
from app.utils.logging import configure_logging
from app.utils.paths import new_job_id, safe_filename

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings.ensure_dirs()
    await store.init_store()
    yield


app = FastAPI(title="lingbot-map studio worker", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_MIME = {
    ".glb": "model/gltf-binary",
    ".ply": "application/octet-stream",
    ".obj": "text/plain",
    ".npz": "application/octet-stream",
    ".json": "application/json",
}


@app.get("/api/health")
async def health() -> dict[str, Any]:
    info: dict[str, Any] = {"ok": True}
    try:
        import torch

        info["cuda"] = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:  # noqa: BLE001
        info["cuda"] = {"error": str(exc)}
    return info


@app.get("/api/jobs")
async def list_jobs() -> list[dict[str, Any]]:
    rows = await store.list_jobs()
    return [r.model_dump(mode="json") for r in rows]


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    job = await store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job.model_dump(mode="json")


@app.post("/api/drafts", status_code=201)
async def create_draft(
    videos: list[UploadFile] = File(...),
) -> dict[str, Any]:
    """Upload video(s), probe metadata via ffprobe, return inferred config.

    The uploaded files are staged under /data/drafts/<id>/uploads. Call
    POST /api/jobs with draft_id to launch inference; no re-upload required.
    """
    if not videos:
        raise HTTPException(status_code=422, detail="at least one video required")

    drafts.sweep_expired()
    draft_id = new_job_id()
    uploads_dir = drafts.draft_uploads(draft_id)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for f in videos:
        name = safe_filename(f.filename or "upload.mp4")
        target = uploads_dir / name
        with target.open("wb") as out:
            while chunk := await f.read(1024 * 1024):
                out.write(chunk)
        saved.append(target)

    probes: list[dict[str, Any]] = []
    for path in saved:
        try:
            p = await probe_video(path)
        except Exception as exc:  # noqa: BLE001
            p = {"error": str(exc)}
        p["name"] = path.name
        probes.append(p)

    suggested = suggest_config(probes)
    rec = drafts.save_draft(draft_id, saved, probes, suggested)
    return rec


@app.get("/api/drafts/{draft_id}")
async def get_draft(draft_id: str) -> dict[str, Any]:
    rec = drafts.load_draft(draft_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="draft not found")
    return rec


@app.delete("/api/drafts/{draft_id}")
async def drop_draft(draft_id: str) -> dict[str, bool]:
    ok = drafts.delete_draft(draft_id)
    if not ok:
        raise HTTPException(status_code=404, detail="draft not found")
    return {"deleted": True}


@app.post("/api/jobs", status_code=201)
async def create_job(
    videos: list[UploadFile] | None = File(None),
    config: str | None = Form(None),
    draft_id: str | None = Form(None),
) -> dict[str, str]:
    try:
        if not config:
            raise HTTPException(status_code=422, detail="config is required")
        config_obj = JobConfig.model_validate_json(config)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"invalid config: {exc}") from exc

    job_id = new_job_id()
    uploads_dir = settings.job_uploads(job_id)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    saved_names: list[str] = []

    if draft_id:
        rec = drafts.load_draft(draft_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="draft not found")
        sources = drafts.draft_video_paths(draft_id)
        if not sources:
            raise HTTPException(status_code=409, detail="draft has no uploaded files")
        import shutil as _shutil
        for src in sources:
            target = uploads_dir / src.name
            _shutil.move(str(src), str(target))
            saved_paths.append(target)
            saved_names.append(src.name)
        drafts.delete_draft(draft_id)
    elif videos:
        for f in videos:
            name = safe_filename(f.filename or "upload.mp4")
            target = uploads_dir / name
            with target.open("wb") as out:
                while chunk := await f.read(1024 * 1024):
                    out.write(chunk)
            saved_paths.append(target)
            saved_names.append(name)
    else:
        raise HTTPException(status_code=422, detail="videos or draft_id required")

    job = Job(id=job_id, status="queued", config=config_obj, uploads=saved_names)
    await store.create_job(job)
    asyncio.create_task(runner.run_job(job_id, saved_paths, config_obj))
    return {"id": job_id}


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str) -> dict[str, Any]:
    job = await store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    # Terminal statuses only — do not delete a running job.
    if job.status in {"queued", "ingest", "inference", "export"}:
        raise HTTPException(status_code=409, detail="job is running")
    await store.delete_job(job_id)
    job_dir = settings.job_dir(job_id)
    if job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)
    return {"deleted": True}


@app.get("/api/jobs/{job_id}/artifacts/{name}")
async def get_artifact(job_id: str, name: str):
    safe = safe_filename(name)
    path = settings.job_artifacts(job_id) / safe
    if not path.exists():
        raise HTTPException(status_code=404, detail="artifact not found")
    mime = _MIME.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(path, media_type=mime, filename=safe)


@app.get("/api/jobs/{job_id}/manifest")
async def get_manifest(job_id: str) -> dict[str, Any]:
    job = await store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    art_dir = settings.job_artifacts(job_id)
    artifacts: list[dict[str, Any]] = []
    if art_dir.exists():
        for p in sorted(art_dir.iterdir()):
            artifacts.append(
                {
                    "name": p.name,
                    "size": p.stat().st_size,
                    "suffix": p.suffix.lstrip("."),
                }
            )
    latest_mesh = None
    for a in reversed(artifacts):
        if a["suffix"] == "glb":
            latest_mesh = a["name"]
            break
    return {
        "id": job.id,
        "status": job.status,
        "config": job.config.model_dump(),
        "artifacts": artifacts,
        "latest_mesh": latest_mesh,
        "frames_total": job.frames_total,
        "error": job.error,
    }


@app.post("/api/jobs/{job_id}/reexport")
async def reexport_job(job_id: str, body: ReexportRequest) -> dict[str, Any]:
    job = await store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    try:
        predictions = load_cached_predictions(job_id, settings.data_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    path = await reexport(
        job_id=job_id,
        artifacts_dir=settings.job_artifacts(job_id),
        predictions=predictions,
        config=job.config,
        fmt=body.format,
        conf_percentile=body.conf_percentile,
        show_cam=body.show_cam,
        mask_sky=body.mask_sky,
        mask_black_bg=body.mask_black_bg,
        mask_white_bg=body.mask_white_bg,
        publish=bus.publish,
    )
    art = Artifact(name=path.name, kind=body.format, size_bytes=path.stat().st_size)
    new_artifacts = list(job.artifacts) + [art]
    await store.update_job(job_id, artifacts=new_artifacts)
    return {"name": path.name, "format": body.format, "size": art.size_bytes}


@app.post("/api/jobs/{job_id}/mesh/edit")
async def mesh_edit(job_id: str, body: MeshEditRequest) -> dict[str, Any]:
    job = await store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    artifacts_dir = settings.job_artifacts(job_id)
    try:
        path, rev = await asyncio.to_thread(
            apply_op,
            artifacts_dir,
            body.op,
            body.params or {},
            body.face_indices,
            body.source_revision,
        )
    except Exception as exc:  # noqa: BLE001
        await bus.publish(
            JobEvent(
                job_id=job_id,
                stage="mesh",
                level="error",
                message=f"mesh op {body.op} failed: {exc}",
            )
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    summary = await asyncio.to_thread(mesh_summary, path)
    art = Artifact(
        name=path.name, kind="glb", revision=rev, size_bytes=path.stat().st_size
    )
    new_artifacts = list(job.artifacts) + [art]
    await store.update_job(job_id, artifacts=new_artifacts)
    await bus.publish(
        JobEvent(
            job_id=job_id,
            stage="mesh",
            message=f"{body.op} -> rev {rev}",
            data={"name": path.name, "revision": rev, **summary},
        )
    )
    return {"name": path.name, "revision": rev, **summary}


@app.websocket("/api/jobs/{job_id}/stream")
async def job_stream(ws: WebSocket, job_id: str) -> None:
    await ws.accept()
    try:
        async for event in bus.subscribe(job_id):
            await ws.send_text(event.model_dump_json())
    except WebSocketDisconnect:
        return
    except Exception as exc:  # noqa: BLE001
        log.warning("ws %s closed: %s", job_id, exc)
    finally:
        try:
            await ws.close()
        except Exception:
            pass


@app.exception_handler(Exception)
async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled error")
    return JSONResponse(status_code=500, content={"detail": str(exc)})
