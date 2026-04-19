"""Worker entry point.

Long-running process that claims queued jobs for one worker class and runs
them to completion. Three instances (`worker-lingbot`, `worker-slam`,
`worker-gs`) share the same sqlite DB + data volume but only pick up jobs
whose `worker_class` matches their env.

Loop is deliberately tiny: claim → run → release → sleep. No Celery/Redis;
we rely on SQLite's WAL + BEGIN IMMEDIATE for atomic claim.

Graceful shutdown (SIGTERM from docker): finish the current job's
try/except block (which writes terminal state), then exit. The in-flight
job's claim is released by the runner's `finally`.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal

from app.cloud import JobSource, LocalJobSource
from app.config import settings
from app.jobs import runner, store
from app.processors import ids_for_worker_class, load_processor
from app.utils.logging import configure_logging

log = logging.getLogger(__name__)


# How often the claim loop wakes up when the queue is empty. Low enough to
# feel snappy, high enough that idle workers aren't hammering sqlite.
IDLE_POLL_INTERVAL_S = float(os.environ.get("WORKER_IDLE_POLL_S", "1.5"))

# How long before a silent claim is considered orphaned. ≈3× heartbeat so a
# missed interval doesn't trip the sweeper.
STALE_CLAIM_THRESHOLD_S = float(os.environ.get("WORKER_STALE_THRESHOLD_S", "60.0"))

# How often this worker runs the stale-claim sweep. All three workers sweep;
# the operation is idempotent, so racing writes are fine.
SWEEP_INTERVAL_S = float(os.environ.get("WORKER_SWEEP_INTERVAL_S", "30.0"))


def _apply_vram_cap() -> None:
    """Same fractional CUDA cap the API used to apply pre-Phase-2."""
    try:
        import torch

        if not torch.cuda.is_available():
            return
        frac = max(0.1, min(1.0, settings.vram_limit_fraction))
        torch.cuda.set_per_process_memory_fraction(frac, 0)
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        log.info(
            "cuda memory cap: %.2f × %.1f GB = %.1f GB",
            frac,
            total,
            frac * total,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("failed to apply vram cap: %s", exc)


def _build_job_source() -> JobSource:
    """Pick a transport from env. Default is the shared-volume local source.

    `WORKER_MODE=remote` constructs an `HttpJobSource` from the broker
    URL + per-job token that the dispatcher injected when launching
    this pod. Mis-configured values fail fast so a misconfigured pod
    doesn't silently process nothing (or worse, claim a random job).
    """
    mode = os.environ.get("WORKER_MODE", "local").lower()
    if mode == "local":
        return LocalJobSource()
    if mode == "remote":
        broker_url = os.environ.get("STUDIO_BROKER_URL")
        token = os.environ.get("STUDIO_JOB_TOKEN")
        if not broker_url:
            raise RuntimeError(
                "WORKER_MODE=remote requires STUDIO_BROKER_URL in env"
            )
        if not token:
            raise RuntimeError(
                "WORKER_MODE=remote requires STUDIO_JOB_TOKEN in env"
            )
        from app.cloud import HttpJobSource

        return HttpJobSource(base_url=broker_url, token=token)
    raise RuntimeError(f"unknown WORKER_MODE={mode!r}")


async def _preload_processors(worker_class: str) -> None:
    """Eagerly import every processor this worker is allowed to run.

    Cheap to fail loudly at startup (wrong container for this worker_class)
    than to fail mid-run with an ImportError on the first job.
    """
    ids = ids_for_worker_class(worker_class)
    if not ids:
        raise RuntimeError(
            f"worker_class={worker_class!r} has no processors in WORKER_CLASSES — check spelling"
        )
    for pid in ids:
        cls = load_processor(pid)
        log.info("preloaded processor %s → %s", pid, cls.__name__)


async def _sweep_loop(stop: asyncio.Event) -> None:
    """Reap stale claims periodically. Idempotent across workers."""
    while not stop.is_set():
        try:
            reaped = await store.sweep_stale_claims(STALE_CLAIM_THRESHOLD_S)
            if reaped:
                log.info("stale-claim sweep reaped %d job(s)", reaped)
        except Exception as exc:  # noqa: BLE001
            log.warning("stale-claim sweep failed: %s", exc)
        try:
            await asyncio.wait_for(stop.wait(), timeout=SWEEP_INTERVAL_S)
        except asyncio.TimeoutError:
            continue


async def _run_forever(worker_class: str) -> None:
    worker_id = store.worker_identity()
    source = _build_job_source()
    log.info(
        "worker %s starting as class=%s (source=%s)",
        worker_id,
        worker_class,
        type(source).__name__,
    )

    await store.init_store()
    await _preload_processors(worker_class)
    _apply_vram_cap()

    stop = asyncio.Event()

    def _on_signal(signum: int) -> None:
        log.info("received signal %d — finishing current job then exiting", signum)
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _on_signal, sig)
        except NotImplementedError:
            # Windows / restricted envs — fall back to default behaviour.
            pass

    sweeper = asyncio.create_task(_sweep_loop(stop))

    try:
        while not stop.is_set():
            try:
                claim = await source.claim_next(worker_class, worker_id=worker_id)
            except Exception as exc:  # noqa: BLE001
                log.warning("claim_next failed: %s", exc)
                await asyncio.sleep(IDLE_POLL_INTERVAL_S)
                continue

            if claim is None:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=IDLE_POLL_INTERVAL_S)
                except asyncio.TimeoutError:
                    pass
                continue

            log.info(
                "claimed %s (processor=%s, %d upload(s))",
                claim.job_id,
                claim.config.processor,
                len(claim.uploads),
            )
            try:
                await runner.run_job(
                    claim.job_id,
                    claim.uploads,
                    claim.config,
                    worker_id=worker_id,
                    source=source,
                )
            except Exception:
                # runner.run_job already finalises the row on exception — we
                # just log and continue to the next job.
                log.exception("unhandled error running %s", claim.job_id)
    finally:
        sweeper.cancel()
        try:
            await sweeper
        except (asyncio.CancelledError, Exception):
            pass
        log.info("worker %s exiting", worker_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="lingbot-map-studio worker")
    parser.add_argument(
        "--worker-class",
        default=os.environ.get("WORKER_CLASS"),
        help="which processor family to run (lingbot|slam|gs). Defaults to $WORKER_CLASS.",
    )
    args = parser.parse_args()
    if not args.worker_class:
        raise SystemExit("--worker-class or WORKER_CLASS env var is required")

    configure_logging()
    asyncio.run(_run_forever(args.worker_class))


if __name__ == "__main__":
    main()
