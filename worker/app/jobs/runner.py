from __future__ import annotations

import asyncio
import logging
import traceback
from pathlib import Path

from app.cloud import JobSource, LocalJobSource
from app.jobs import cancel as cancel_mod
from app.jobs import store
from app.jobs.cancel import CancelToken, JobCancelled
from app.jobs.schema import AnyJobConfig, JobEvent, JobStatus
from app.pipeline.watchdog import VramLimitExceeded
from app.processors import resolve
from app.processors.base import JobContext

log = logging.getLogger(__name__)


# How often each running worker bumps `claimed_at` on its job row. The stale-
# claim sweep uses ~3× this to decide a worker has vanished.
HEARTBEAT_INTERVAL_S = 10.0

# How often the runner polls the cancel flag through the job source. Lower
# than `HEARTBEAT_INTERVAL_S` so cancel feels snappy even over HTTP.
CANCEL_POLL_INTERVAL_S = 0.5


async def _heartbeat_loop(
    source: JobSource, job_id: str, worker_id: str, interval_s: float
) -> None:
    try:
        while True:
            try:
                await source.heartbeat(job_id, worker_id)
            except Exception as exc:  # noqa: BLE001
                log.warning("heartbeat failed for %s: %s", job_id, exc)
            await asyncio.sleep(interval_s)
    except asyncio.CancelledError:
        return


async def _cancel_watch_loop(
    source: JobSource,
    job_id: str,
    token: CancelToken,
    poll_interval_s: float,
) -> None:
    """Mirror the studio's cancel flag onto `token.cancelled`.

    Replaces `cancel_mod.watch_cancel_flag` for the source-aware runner so
    the same poll works for both local sqlite reads and HTTP long-polls.
    """
    try:
        while not token.cancelled:
            try:
                if await source.is_cancel_requested(job_id):
                    token.cancel("stopped by user")
                    return
            except Exception as exc:  # noqa: BLE001
                log.warning("cancel-poller read failed for %s: %s", job_id, exc)
            await asyncio.sleep(poll_interval_s)
    except asyncio.CancelledError:
        return


async def run_job(
    job_id: str,
    uploads: list[Path],
    config: AnyJobConfig,
    *,
    worker_id: str | None = None,
    source: JobSource | None = None,
) -> None:
    """Run a single job to completion by dispatching to the processor matching
    `config.processor`.

    The runner owns state transitions, cancellation bookkeeping, and the
    error-handling fan-out (OOM / cancelled / watchdog / general failure).
    Processors are pure: they publish events, do work, and return the
    artifacts they produced. All "talk to the studio" calls go through the
    `JobSource` so the same runner executes both locally (LocalJobSource) and
    against a remote broker (HttpJobSource, later slice).
    """

    if source is None:
        source = LocalJobSource()

    job_dir = source.job_dir(job_id)
    frames_dir = source.frames_dir(job_id)
    artifacts_dir = source.artifacts_dir(job_id)

    if worker_id is None:
        worker_id = store.worker_identity()

    # Fresh in-memory token for this run. The in-process `_tokens` dict is
    # still populated so code paths that have an existing handle (e.g. older
    # tests) keep working, but the authoritative signal is the studio's
    # cancel flag polled via the source.
    cancel_token = cancel_mod.get_token(job_id)
    cancel_token.cancelled = False
    cancel_token.reason = ""

    async def _publish(event: JobEvent) -> JobEvent:
        return await source.publish_event(event)

    async def _set_status(status: JobStatus) -> None:
        await source.set_status(job_id, status)

    async def _set_frames_total(frames_total: int) -> None:
        await source.set_frames_total(job_id, frames_total)

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

    # Background helpers: mirror the studio's cancel flag onto the token and
    # keep the claim fresh so the stale-sweep doesn't reap us mid-run.
    cancel_watcher = asyncio.create_task(
        _cancel_watch_loop(source, job_id, cancel_token, CANCEL_POLL_INTERVAL_S)
    )
    heartbeat_task = asyncio.create_task(
        _heartbeat_loop(source, job_id, worker_id, HEARTBEAT_INTERVAL_S)
    )

    try:
        processor = resolve(config)
        result = await processor.run(ctx)

        await source.set_artifacts(job_id, result.artifacts)
        await source.set_status(job_id, "ready")
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
        await source.set_error(job_id, f"cancelled: {exc}")
        await source.set_status(job_id, "cancelled")
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
        await source.set_error(
            job_id,
            (
                f"vram watchdog aborted the job: {exc}\n\n"
                "try one of these:\n"
                "  · apply the 'low-mem' preset\n"
                "  · set mode=windowed (smaller window_size)\n"
                "  · drop num_scale_frames to 2\n"
                "  · drop fps to ~10\n"
                "  · lower image_size to 384"
            ),
        )
        await source.set_status(job_id, "failed")
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
            await source.set_error(
                job_id,
                (
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
            await source.set_status(job_id, "failed")
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
            await source.set_error(job_id, f"{type(exc).__name__}: {exc}\n\n{tb}")
            await source.set_status(job_id, "failed")
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
        # Stop background helpers before releasing the claim so a trailing
        # heartbeat doesn't resurrect the claimed_at timestamp.
        for task in (cancel_watcher, heartbeat_task):
            if not task.done():
                task.cancel()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                pass

        cancel_mod.drop_token(job_id)
        try:
            await source.release(job_id, worker_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("failed to release claim for %s: %s", job_id, exc)
        await source.close_events(job_id)


def pick_cancel_token(job_id: str) -> CancelToken:
    """Helper for tests / legacy call sites that want the in-process token.

    Prefer `store.request_cancel(job_id)` from anywhere except a thread that
    is already running this job's pipeline.
    """
    return cancel_mod.get_token(job_id)
