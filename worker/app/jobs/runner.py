from __future__ import annotations

import asyncio
import logging
import traceback
from pathlib import Path

from app.config import settings
from app.jobs import store
from app.jobs.events import bus
from app.jobs.schema import Artifact, JobConfig, JobEvent
from app.pipeline.checkpoints import ensure_checkpoint
from app.pipeline.export import export_reconstruction
from app.pipeline.ingest import concat_videos_to_frames
from app.pipeline.inference import run_inference
from app.pipeline.watchdog import VramLimitExceeded, VramWatchState, run_vram_watchdog

log = logging.getLogger(__name__)


async def _publish(event: JobEvent) -> JobEvent:
    return await bus.publish(event)


async def run_job(job_id: str, uploads: list[Path], config: JobConfig) -> None:
    job_dir = settings.job_dir(job_id)
    frames_dir = settings.job_frames(job_id)
    artifacts_dir = settings.job_artifacts(job_id)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    try:
        await _publish(JobEvent(job_id=job_id, stage="queue", message="job starting"))

        # 1. ingest
        await store.update_job(job_id, status="ingest")
        frames_total = await concat_videos_to_frames(
            job_id=job_id,
            sources=uploads,
            dest=frames_dir,
            config=config,
            publish=_publish,
        )
        await store.update_job(job_id, frames_total=frames_total)

        # 2. checkpoint
        ckpt = await ensure_checkpoint(config.model_id, job_id, _publish)

        # 3. inference — spin up a VRAM watchdog alongside the GPU call.
        await store.update_job(job_id, status="inference")
        soft_limit = config.vram_soft_limit_gb or settings.vram_default_soft_limit_gb
        vram_state = VramWatchState(soft_limit_gb=float(soft_limit))
        watchdog_task = asyncio.create_task(
            run_vram_watchdog(job_id, vram_state, _publish)
        )
        try:
            predictions = await run_inference(
                job_id=job_id,
                frames_dir=frames_dir,
                ckpt_path=ckpt,
                config=config,
                publish=_publish,
                vram_state=vram_state,
            )
        finally:
            vram_state.stop()
            try:
                await asyncio.wait_for(watchdog_task, timeout=5.0)
            except asyncio.TimeoutError:
                watchdog_task.cancel()
        await _publish(
            JobEvent(
                job_id=job_id,
                stage="inference",
                message=(
                    f"vram peak {vram_state.peak_gb:.2f} GB "
                    f"(soft limit {vram_state.soft_limit_gb:.1f} GB)"
                ),
                data={
                    "vram_peak_gb": round(vram_state.peak_gb, 3),
                    "vram_soft_limit_gb": vram_state.soft_limit_gb,
                },
            )
        )

        # 4. export
        await store.update_job(job_id, status="export")
        artifacts = await export_reconstruction(
            job_id=job_id,
            frames_dir=frames_dir,
            artifacts_dir=artifacts_dir,
            predictions=predictions,
            config=config,
            publish=_publish,
        )

        art_list = []
        for name, path in artifacts.items():
            art_list.append(
                Artifact(
                    name=path.name,
                    kind=name,  # type: ignore[arg-type]
                    size_bytes=path.stat().st_size,
                )
            )
        # Also surface the cached predictions npz so the UI can see it.
        npz = artifacts_dir / "predictions.npz"
        if npz.exists():
            art_list.append(
                Artifact(
                    name=npz.name,
                    kind="npz",
                    size_bytes=npz.stat().st_size,
                )
            )

        await store.update_job(job_id, status="ready", artifacts=art_list)
        await _publish(
            JobEvent(
                job_id=job_id,
                stage="system",
                message="job ready",
                data={"artifacts": [a.name for a in art_list]},
            )
        )
    except VramLimitExceeded as exc:
        log.warning("job %s aborted by vram watchdog: %s", job_id, exc)
        await store.update_job(
            job_id,
            status="failed",
            error=(
                f"vram watchdog aborted the job: {exc}\n\n"
                "raise vram_soft_limit_gb or drop fps / keyframe_interval / "
                "num_scale_frames and try again."
            ),
        )
        await _publish(
            JobEvent(
                job_id=job_id,
                stage="system",
                level="error",
                message=f"aborted: {exc}",
            )
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("job %s failed", job_id)
        tb = traceback.format_exc()
        await store.update_job(job_id, status="failed", error=f"{exc}\n\n{tb}")
        await _publish(
            JobEvent(
                job_id=job_id,
                stage="system",
                level="error",
                message=f"job failed: {exc}",
                data={"traceback": tb},
            )
        )
    finally:
        await bus.close(job_id)
