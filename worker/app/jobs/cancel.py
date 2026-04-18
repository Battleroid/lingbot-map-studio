from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class CancelToken:
    """Shared flag between the job runner and the inference / ingest hooks.

    Runner creates a token when the job starts. Stop endpoint sets `cancelled`.
    The inference forward hook (and ingest / export checkpoints) raises
    JobCancelled when it sees the flag flipped, so the thread unwinds cleanly
    instead of being abruptly killed mid-CUDA call.
    """

    cancelled: bool = False
    reason: str = ""

    def cancel(self, reason: str = "stopped by user") -> None:
        self.cancelled = True
        self.reason = reason


class JobCancelled(RuntimeError):
    """Raised from a hook or checkpoint when the cancel token is set."""


_tokens: dict[str, CancelToken] = {}
_lock = threading.Lock()


def get_token(job_id: str) -> CancelToken:
    with _lock:
        return _tokens.setdefault(job_id, CancelToken())


def drop_token(job_id: str) -> None:
    with _lock:
        _tokens.pop(job_id, None)


def cancel(job_id: str, reason: str = "stopped by user") -> bool:
    """Return True if the token existed (job was running) and we flipped it."""
    with _lock:
        tok = _tokens.get(job_id)
    if tok is None:
        return False
    tok.cancel(reason)
    log.info("cancel requested for %s: %s", job_id, reason)
    return True
