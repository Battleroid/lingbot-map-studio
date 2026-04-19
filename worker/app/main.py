from __future__ import annotations

import asyncio
import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import (
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
from app.jobs import cancel as cancel_mod
from app.jobs import drafts, store
from app.jobs.events import bus
from app.jobs.schema import (
    Artifact,
    Job,
    JobEvent,
    LingbotConfig,
    MeshEditRequest,
    ReexportRequest,
    parse_job_config,
)
from app.processors import worker_class_for
from app.mesh.ops import apply_op, mesh_summary
from app.pipeline.export import reexport
from app.pipeline.inference import load_cached_predictions
from app.pipeline.preview import (
    apply_fisheye,
    extract_frame,
    render_fpv_preview,
    render_osd_preview,
)
from app.pipeline.probe import probe_video, suggest_config
from app.utils.logging import configure_logging
from app.utils.paths import new_job_id, safe_filename

log = logging.getLogger(__name__)


async def _sweep_orphaned_jobs() -> None:
    """Reap jobs whose worker process died and never released the claim.

    Phase 2: the API no longer runs the inference itself, so 'orphaned'
    means 'a worker crashed holding a claim'. `store.sweep_stale_claims`
    checks `claimed_at` against the same threshold the worker uses for
    its heartbeat, so jobs still actively running by a live worker are
    left alone.

    We call this once on API startup and then let the workers' background
    sweep loop keep the DB clean afterwards.
    """
    reaped = await store.sweep_stale_claims(stale_after_s=60.0)
    if reaped:
        log.info("reaped %d orphaned job(s) on startup", reaped)


async def _periodic_sweep_loop(stop: asyncio.Event) -> None:
    """API-side copy of the stale-claim sweep.

    The workers already run this, but the API container has a different
    restart cadence and may notice a crashed worker first. Cheap — one
    SELECT + ~0 updates per pass unless something actually crashed.
    """
    while not stop.is_set():
        try:
            reaped = await store.sweep_stale_claims(stale_after_s=60.0)
            if reaped:
                log.info("api stale-claim sweep reaped %d job(s)", reaped)
        except Exception as exc:  # noqa: BLE001
            log.warning("api stale-claim sweep failed: %s", exc)
        try:
            await asyncio.wait_for(stop.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            continue


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings.ensure_dirs()
    await store.init_store()
    await _sweep_orphaned_jobs()

    # The API no longer runs CUDA itself, so no vram cap here; each worker
    # container applies its own in `worker_main.py`. Spawn the periodic
    # sweep so crashed workers' jobs get reaped even when this API process
    # is the first to notice.
    sweep_stop = asyncio.Event()
    sweep_task = asyncio.create_task(_periodic_sweep_loop(sweep_stop))
    try:
        yield
    finally:
        sweep_stop.set()
        try:
            await asyncio.wait_for(sweep_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            pass


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


@app.get("/api/drafts/{draft_id}/preview/fisheye")
async def preview_fisheye(
    draft_id: str,
    in_fov: float = 165.0,
    out_fov: float = 90.0,
    side: str = "after",
) -> FileResponse:
    """Return a single PNG: an extracted source frame, optionally unwrapped."""
    rec = drafts.load_draft(draft_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="draft not found")
    sources = drafts.draft_video_paths(draft_id)
    if not sources:
        raise HTTPException(status_code=409, detail="draft has no uploaded files")
    src_video = sources[0]
    duration = rec.get("probes", [{}])[0].get("duration_s") if rec.get("probes") else None
    ts = min(max(0.5, (duration or 0.0) * 0.1), max(0.5, (duration or 1.0) - 0.25))

    preview_dir = drafts.draft_dir(draft_id) / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    before_png = preview_dir / "before.png"
    if not before_png.exists():
        await extract_frame(src_video, before_png, timestamp=ts)

    if side == "before":
        return FileResponse(
            before_png,
            media_type="image/png",
            filename="before.png",
            headers={"Cache-Control": "public, max-age=300"},
        )

    after_png = preview_dir / f"after_in{int(in_fov)}_out{int(out_fov)}.png"
    if not after_png.exists():
        await apply_fisheye(before_png, after_png, in_fov=in_fov, out_fov=out_fov)
    return FileResponse(
        after_png,
        media_type="image/png",
        filename=after_png.name,
        headers={"Cache-Control": "public, max-age=300"},
    )


async def _maybe_fisheye_prewarp(
    src_video, preview_dir, in_fov: float, out_fov: float
):
    import asyncio as _asyncio

    warped = preview_dir / f"warped_fe{int(in_fov)}x{int(out_fov)}.mp4"
    if warped.exists():
        return warped
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src_video),
        "-vf",
        f"v360=input=fisheye:output=flat:ih_fov={in_fov}:iv_fov={in_fov}:d_fov={out_fov}",
        "-preset",
        "ultrafast",
        str(warped),
    ]
    proc = await _asyncio.create_subprocess_exec(
        *cmd,
        stdout=_asyncio.subprocess.PIPE,
        stderr=_asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f"fisheye pre-warp failed: {stderr.decode(errors='replace')}",
        )
    return warped


@app.get("/api/drafts/{draft_id}/preview/osd")
async def preview_osd(
    draft_id: str,
    samples: int = 30,
    std_threshold: float = 5.0,
    dilate: int = 2,
    detect_text: bool = True,
    edge_persist_frac: float = 0.75,
    fisheye: bool = False,
    in_fov: float = 165.0,
    out_fov: float = 90.0,
) -> FileResponse:
    """Return a PNG: the first frame with the computed OSD mask overlaid in red."""
    rec = drafts.load_draft(draft_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="draft not found")
    sources = drafts.draft_video_paths(draft_id)
    if not sources:
        raise HTTPException(status_code=409, detail="draft has no uploaded files")
    src_video = sources[0]
    duration = rec.get("probes", [{}])[0].get("duration_s") if rec.get("probes") else None
    fps_hint = rec.get("probes", [{}])[0].get("fps") if rec.get("probes") else None

    preview_dir = drafts.draft_dir(draft_id) / "preview"
    samples_dir = preview_dir / "osd_samples"
    preview_dir.mkdir(parents=True, exist_ok=True)

    work_video = src_video
    txt_key = "t1" if detect_text else "t0"
    key = (
        f"osd_s{samples}_v{std_threshold:g}_d{dilate}_{txt_key}"
        f"_e{edge_persist_frac:g}"
    )
    if fisheye:
        key = f"{key}_fe{int(in_fov)}x{int(out_fov)}"
    out_png = preview_dir / f"{key}.png"
    if out_png.exists():
        return FileResponse(
            out_png,
            media_type="image/png",
            filename=out_png.name,
            headers={"Cache-Control": "public, max-age=300"},
        )

    if fisheye:
        work_video = await _maybe_fisheye_prewarp(src_video, preview_dir, in_fov, out_fov)

    try:
        result = await render_osd_preview(
            video=work_video,
            work_dir=samples_dir,
            out_png=out_png,
            samples=samples,
            std_threshold=std_threshold,
            dilate=dilate,
            duration_s=duration,
            fps_hint=fps_hint,
            detect_text=detect_text,
            edge_persist_frac=edge_persist_frac,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return FileResponse(
        out_png,
        media_type="image/png",
        filename=out_png.name,
        headers={
            "Cache-Control": "public, max-age=300",
            "X-Mask-Coverage": str(result.get("coverage", 0)),
            "X-Mask-Samples": str(result.get("samples", 0)),
        },
    )


_FPV_PREVIEW_STAGES = {"color_norm", "deblur", "analog_cleanup", "rs_correction"}


@app.get("/api/drafts/{draft_id}/preview/fpv")
async def preview_fpv(
    draft_id: str,
    stage: str,
    shear: float | None = None,
    analog_cleanup: bool = False,
    deflicker: bool = False,
) -> FileResponse:
    """Return a PNG showing the effect of one Phase-3 FPV stage on a sampled frame.

    `stage` must be one of `color_norm`, `deblur`, `analog_cleanup`,
    `rs_correction`. `shear` overrides the rolling-shutter estimate.
    `analog_cleanup`/`deflicker` toggle individual filters for the
    `analog_cleanup` stage (they mirror the config flags so the preview
    matches what ingest will actually run).
    """
    if stage not in _FPV_PREVIEW_STAGES:
        raise HTTPException(status_code=422, detail=f"unknown stage: {stage}")

    rec = drafts.load_draft(draft_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="draft not found")
    sources = drafts.draft_video_paths(draft_id)
    if not sources:
        raise HTTPException(status_code=409, detail="draft has no uploaded files")
    src_video = sources[0]
    duration = rec.get("probes", [{}])[0].get("duration_s") if rec.get("probes") else None
    ts = min(max(0.5, (duration or 0.0) * 0.1), max(0.5, (duration or 1.0) - 0.25))

    preview_dir = drafts.draft_dir(draft_id) / "preview"
    work_dir = preview_dir / "fpv"
    preview_dir.mkdir(parents=True, exist_ok=True)

    if stage == "analog_cleanup":
        # Cache bucket differentiates which filters are composed for the preview.
        filters: list[str] = []
        if analog_cleanup:
            filters.append(
                "atadenoise=0a=0.02:0b=0.04:1a=0.02:1b=0.04:2a=0.02:2b=0.04"
            )
        if deflicker:
            filters.append("deflicker=mode=pm:size=5")
        key_bits = []
        if analog_cleanup:
            key_bits.append("ata")
        if deflicker:
            key_bits.append("dfl")
        bucket = "_".join(key_bits) or "none"
        out_png = preview_dir / f"fpv_analog_{bucket}.png"
        params = {"filters": filters}
    elif stage == "rs_correction":
        shear_tag = "auto" if shear is None else f"{shear:+.3f}"
        out_png = preview_dir / f"fpv_rs_{shear_tag}.png"
        params = {"shear_override": shear}
    else:
        out_png = preview_dir / f"fpv_{stage}.png"
        params = {}

    if not out_png.exists():
        try:
            await render_fpv_preview(
                video=src_video,
                out_png=out_png,
                stage=stage,
                timestamp=ts,
                params=params,
                work_dir=work_dir,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return FileResponse(
        out_png,
        media_type="image/png",
        filename=out_png.name,
        headers={"Cache-Control": "public, max-age=300"},
    )


@app.post("/api/jobs", status_code=201)
async def create_job(
    videos: list[UploadFile] | None = File(None),
    config: str | None = Form(None),
    draft_id: str | None = Form(None),
) -> dict[str, str]:
    try:
        if not config:
            raise HTTPException(status_code=422, detail="config is required")
        config_obj = parse_job_config(config)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"invalid config: {exc}") from exc

    # Worker-class routing is derived from the processor id. The API
    # stamps it on the row; the matching worker container claims it.
    worker_class = worker_class_for(config_obj)

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
    await store.create_job(job, worker_class=worker_class)
    await bus.publish(
        JobEvent(
            job_id=job_id,
            stage="queue",
            message=f"enqueued for worker-{worker_class}",
            data={"worker_class": worker_class, "processor": config_obj.processor},
        )
    )
    return {"id": job_id}


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str, force: bool = False) -> dict[str, Any]:
    """Delete a job and all its artifacts.

    Cross-container authoritative signal: the DB row's `claimed_by` column.
    If a worker currently owns the claim we refuse unless `force=true`.
    An unclaimed row in a running status (e.g. the worker died without
    cleaning up) is safe to delete — we mark it failed first for the
    audit log, then drop the files.
    """
    job = await store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    if job.status in store.RUNNING_STATUSES and not force:
        async with store.session() as s:
            row = await s.get(store.JobRow, job_id)
            claimed_by = row.claimed_by if row else None
        if claimed_by:
            raise HTTPException(
                status_code=409,
                detail="job is running — stop it first or pass force=true",
            )
        await store.update_job(
            job_id,
            status="failed",
            error="deleted while orphaned (no active worker claim)",
        )

    await store.delete_job(job_id)
    job_dir = settings.job_dir(job_id)
    if job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)
    return {"deleted": True}


@app.post("/api/jobs/{job_id}/stop")
async def stop_job(job_id: str, force: bool = False) -> dict[str, Any]:
    """Request cancellation of a running job.

    Graceful (default): flips a shared cancel flag. The inference hook +
    ingest/export checkpoints watch for it and raise JobCancelled on next
    check, so the job unwinds cleanly and lands in status=cancelled.

    Force (`?force=true`): immediately marks the job row as cancelled in the
    DB and drops the cancel token. Use this when the worker is genuinely hung
    inside a CUDA call and the graceful path never returns — the underlying
    thread may keep running in the background but the UI state is consistent
    and the job becomes deletable. The orphan sweep on next worker restart
    tidies up the real task.
    """
    job = await store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status in {"ready", "failed", "cancelled"}:
        raise HTTPException(status_code=409, detail="job already finished")

    # Cross-process cancel: flip the DB flag. The worker's cancel-poller
    # mirrors it onto the in-flight CancelToken on its next tick, and the
    # pipeline hooks raise JobCancelled at the next checkpoint.
    await cancel_mod.request_cancel(job_id, "stopped by user")

    if force:
        await store.update_job(
            job_id,
            status="cancelled",
            error="force-stopped by user (GPU task may still be running in the background — a worker restart will fully clean it up)",
        )
        await bus.publish(
            JobEvent(
                job_id=job_id,
                stage="system",
                level="warn",
                message="force-stopped — marked cancelled regardless of task state",
            )
        )
        await bus.close(job_id)
        return {"cancelled": True, "forced": True}

    await bus.publish(
        JobEvent(
            job_id=job_id,
            stage="system",
            level="warn",
            message="stop requested — unwinding current stage...",
        )
    )
    return {"cancelled": True, "forced": False}


@app.post("/api/jobs/{job_id}/restart", status_code=201)
async def restart_job(job_id: str) -> dict[str, str]:
    """Clone this job's config + uploads into a fresh job and start it.

    Keeps the original job + artifacts untouched. Useful for retrying after a
    cancellation, a failure, or just to re-run with a tweaked config (though
    for config changes the `/api/drafts` flow is better since it lets you
    re-probe metadata)."""
    old = await store.get_job(job_id)
    if old is None:
        raise HTTPException(status_code=404, detail="job not found")

    new_id = new_job_id()
    new_uploads_dir = settings.job_uploads(new_id)
    new_uploads_dir.mkdir(parents=True, exist_ok=True)
    old_uploads_dir = settings.job_uploads(job_id)

    saved_paths: list[Path] = []
    saved_names: list[str] = []
    for name in old.uploads:
        src = old_uploads_dir / name
        if not src.exists():
            raise HTTPException(
                status_code=409,
                detail=f"source upload {name} missing — can't restart",
            )
        dst = new_uploads_dir / name
        shutil.copy2(src, dst)
        saved_paths.append(dst)
        saved_names.append(name)

    from app.jobs.schema import Job as JobModel

    new_job_obj = JobModel(
        id=new_id,
        status="queued",
        config=old.config,
        uploads=saved_names,
    )
    worker_class = worker_class_for(old.config)
    await store.create_job(new_job_obj, worker_class=worker_class)
    await bus.publish(
        JobEvent(
            job_id=new_id,
            stage="queue",
            message=f"restarted from {job_id}; enqueued for worker-{worker_class}",
            data={
                "worker_class": worker_class,
                "processor": old.config.processor,
                "source_job_id": job_id,
            },
        )
    )
    return {"id": new_id}


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
    # Reexport is lingbot-specific — it reloads cached model predictions and
    # re-runs the lingbot GLB export with new thresholds. Other modes have
    # their own reexport paths (added in later phases).
    if not isinstance(job.config, LingbotConfig):
        raise HTTPException(
            status_code=409,
            detail=(
                f"reexport not supported for processor={job.config.processor!r}; "
                "this endpoint only handles lingbot jobs"
            ),
        )
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
