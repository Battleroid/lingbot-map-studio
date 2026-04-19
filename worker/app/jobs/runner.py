from __future__ import annotations

import logging
import traceback
from pathlib import Path

from app.config import settings
from app.jobs import cancel as cancel_mod
from app.jobs import store
from app.jobs.cancel import JobCancelled
from app.jobs.events import bus
from app.jobs.schema import AnyJobConfig, JobEvent, JobStatus
from app.pipeline.watchdog import VramLimitExceeded
from app.processors import resolve
from app.processors.base import JobContext

log = logging.getLogger(__name__)


async def _publish(event: JobEvent) -> JobEvent:
    return await bus.publish(event)


async def run_job(job_id: str, uploads: list[Path], config: AnyJobConfig) -> None:
    """Run a single job to completion by dispatching to the processor matching
    `config.processor`.

    The runner owns DB state transitions, cancellation bookkeeping, and the
    error-handling fan-out (OOM / cancelled / watchdog / general failure).
    Processors are pure: they publish events, do work, and return the
    artifacts they produced.
    """

    job_dir = settings.job_dir(job_id)
    frames_dir = settings.job_frames(job_id)
    artifacts_dir = settings.job_artifacts(job_id)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    cancel_token = cancel_mod.get_token(job_id)

    async def _set_status(status: JobStatus) -> None:
        await store.update_job(job_id, status=status)

    async def _set_frames_total(frames_total: int) -> None:
        await store.update_job(job_id, frames_total=frames_total)

    ctx = JobContext(
        job_id=job_id,
        uploads=uploads,
        config=config,
        job_dir=job_dir,
        frames_dir=frames_dir,
        artifacts_dir=artifacts_dir,
        cancel=cancel_token,
        publish=_publish,
        set_status=_set_status,
        set_frames_total=_set_frames_total,
    )

    try:
        processor = resolve(config)
        result = await processor.run(ctx)

        await store.update_job(job_id, status="ready", artifacts=result.artifacts)
        await _publish(
            JobEvent(
                job_id=job_id,
                stage="system",
                message="job ready",
                data={
                    "artifacts": [a.name for a in result.artifacts],
                    **result.extras,
                },
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
    finally:
        cancel_mod.drop_token(job_id)
        await bus.close(job_id)
