"""Cross-process cancellation.

Phase 2 moves the runner into a separate container from the API, so the
in-memory `_tokens` dict no longer works: the API can't see the worker's
token, and vice versa. The new transport is a boolean column on the job
row (`cancel_requested`).

The synchronous call sites in inference hot paths cannot become async —
they're invoked from inside CUDA forwards on worker threads. We therefore
keep a `CancelToken` object that exposes a plain `cancelled` attribute
that hot paths poll, and have the runner spin up a background async
poller that mirrors the DB column onto that attribute every N ms. Net
effect: the worker's GPU code doesn't change, and the API can still cancel
a job from another container by flipping one bit in sqlite.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class CancelToken:
    """Shared flag between the job runner and the inference / ingest hooks.

    Runner creates a token when the job starts. Stop endpoint sets
    `cancel_requested` on the DB row; the runner's background poller then
    sets `cancelled=True` on this token. The inference forward hook (and
    ingest / export checkpoints) raises JobCancelled when it sees the flag
    flipped, so the thread unwinds cleanly instead of being abruptly
    killed mid-CUDA call.
    """

    cancelled: bool = False
    reason: str = ""

    def cancel(self, reason: str = "stopped by user") -> None:
        self.cancelled = True
        self.reason = reason


class JobCancelled(RuntimeError):
    """Raised from a hook or checkpoint when the cancel token is set."""


# Kept for in-process callers (tests, single-container dev setups). Phase 2
# workers use `watch_cancel_flag` instead, which consumes the DB column.
_tokens: dict[str, CancelToken] = {}
_lock = threading.Lock()


def get_token(job_id: str) -> CancelToken:
    with _lock:
        return _tokens.setdefault(job_id, CancelToken())


def drop_token(job_id: str) -> None:
    with _lock:
        _tokens.pop(job_id, None)


async def request_cancel(job_id: str, reason: str = "stopped by user") -> bool:
    """Ask the worker owning this job to stop.

    Writes `cancel_requested=True` on the job row (cross-process transport)
    and, if this process happens to be running the job too (single-container
    dev mode), also flips the local token so hot paths pick it up without
    waiting for the poller.
    """
    from app.jobs import store

    with _lock:
        tok = _tokens.get(job_id)
    if tok is not None:
        tok.cancel(reason)

    ok = await store.request_cancel(job_id)
    if ok:
        log.info("cancel requested for %s: %s", job_id, reason)
    return ok


async def watch_cancel_flag(
    job_id: str, token: CancelToken, poll_interval_s: float = 0.5
) -> None:
    """Mirror the DB cancel flag onto `token.cancelled` until the job ends.

    Run as a background task in the worker. Cheap — one sqlite read per
    interval. Exits when the token flips (the runner will cancel the task
    after the processor returns anyway).
    """
    from app.jobs import store

    try:
        while not token.cancelled:
            try:
                if await store.is_cancel_requested(job_id):
                    token.cancel("stopped by user")
                    return
            except Exception as exc:  # noqa: BLE001
                # A transient DB glitch shouldn't kill the job — just log and
                # retry next tick.
                log.warning("cancel-poller read failed for %s: %s", job_id, exc)
            await asyncio.sleep(poll_interval_s)
    except asyncio.CancelledError:
        return


# Legacy sync helper, kept so in-process tests can still force a cancel without
# touching the DB. The API's HTTP handler should use `request_cancel()`.
def cancel(job_id: str, reason: str = "stopped by user") -> bool:
    with _lock:
        tok = _tokens.get(job_id)
    if tok is None:
        return False
    tok.cancel(reason)
    log.info("cancel requested (in-process only) for %s: %s", job_id, reason)
    return True
