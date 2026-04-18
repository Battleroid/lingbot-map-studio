from __future__ import annotations

import asyncio
import logging
import traceback
from pathlib import Path

from app.config import settings
from app.jobs import cancel as cancel_mod
from app.jobs import store
from app.jobs.cancel import JobCancelled
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
    cancel_token = cancel_mod.get_token(job_id)

    def _check_cancel() -> None:
        if cancel_token.cancelled:
            raise JobCancelled(cancel_token.reason)

    try:
        await _publish(JobEvent(job_id=job_id, stage="queue", message="job starting"))
        _check_cancel()

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
        _check_cancel()

        # 2. checkpoint
        ckpt = await ensure_checkpoint(config.model_id, job_id, _publish)
        _check_cancel()

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
                cancel_token=cancel_token,
            )
        finally:
            vram_state.stop()
            try:
                await asyncio.wait_for(watchdog_task, timeout=5.0)
            except asyncio.TimeoutError:
                watchdog_task.cancel()
        _check_cancel()
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
    except JobCancelled as exc:
        log.info("job %s cancelled: %s", job_id, exc)
        await store.update_job(
            job_id,
            status="cancelled",
            error=f"cancelled: {exc}",
        )
        await _publish(
            JobEvent(
                job_id=job_id,
                stage="system",
                level="warn",
                message=f"cancelled: {exc}",
            )
        )
    except VramLimitExceeded as exc:
        log.warning("job %s aborted by vram watchdog: %s", job_id, exc)
        await store.update_job(
            job_id,
            status="failed",
            error=(
                f"vram watchdog aborted the job: {exc}\n\n"
                "try one of these:\n"
                "  · apply the 'low-mem' preset\n"
                "  · set mode=windowed (smaller window_size)\n"
                "  · drop num_scale_frames to 2\n"
                "  · drop fps to ~10\n"
                "  · lower image_size to 384"
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
        # Intercept CUDA OOM specifically so the user sees actionable advice
        # instead of a raw 60-line stack trace.
        is_cuda_oom = (
            type(exc).__name__ == "OutOfMemoryError"
            or ("CUDA out of memory" in str(exc))
        )
        if is_cuda_oom:
            log.warning("job %s failed with CUDA OOM", job_id)
            await store.update_job(
                job_id,
                status="failed",
                error=(
                    "CUDA out of memory during inference.\n\n"
                    "the sequence is too long for streaming mode at current "
                    "settings. try in this order:\n"
                    "  1. apply the 'low-mem' preset\n"
                    "  2. or set mode=windowed, window_size=32, overlap_size=8 manually\n"
                    "  3. or drop fps to 10 to reduce total frame count\n"
                    "  4. or lower num_scale_frames to 2 and kv_cache_sliding_window to 16\n\n"
                    f"raw error: {exc}"
                ),
            )
            await _publish(
                JobEvent(
                    job_id=job_id,
                    stage="system",
                    level="error",
                    message="CUDA OOM — see job error for suggested fixes",
                )
            )
        else:
            # Any other exception: log and mark failed with the full traceback
            # so the UI shows it instead of silently leaving the job in
            # whatever status it was when it crashed.
            log.exception("job %s failed", job_id)
            tb = traceback.format_exc()
            await store.update_job(
                job_id,
                status="failed",
                error=f"{type(exc).__name__}: {exc}\n\n{tb}",
            )
            await _publish(
                JobEvent(
                    job_id=job_id,
                    stage="system",
                    level="error",
                    message=f"job failed: {exc}",
                    data={"traceback": tb},
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
        cancel_mod.drop_token(job_id)
        await bus.close(job_id)
