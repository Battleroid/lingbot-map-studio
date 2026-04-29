"""Live-capture wrapper around the existing SLAM sessions.

The capture WebSocket handler in `app.cloud.capture_session.CaptureSession`
needs a `SlamSession`-shaped object (start / step / finalize) to drive
in real time. Each backend (MASt3R-SLAM, DROID-SLAM, DPVO, the
simulated tracker) already exposes that contract via
`worker/app/processors/slam/base.py:SlamSession` — no special live
variant needed.

This module is a thin selector: given a backend id, return the right
session class. The auto-select logic in each backend's
`select_session_cls()` already picks the CUDA path when available and
the simulated path otherwise; we reuse it here so the capture flow
gets the same visibility (Phase 0 warn events) as a batch SLAM job
when it falls back."""

from __future__ import annotations

from typing import Optional

from app.processors.slam.base import SlamSession


def resolve_live_session(backend_id: str) -> SlamSession:
    """Pick the right SlamSession subclass for `backend_id` and
    instantiate it. Falls back to the simulated session if the
    backend id is unknown (rather than raising) so a misconfigured
    capture request gracefully degrades."""
    cls = _resolve_cls(backend_id)
    return cls()


def _resolve_cls(backend_id: str) -> type[SlamSession]:
    if backend_id == "mast3r_slam":
        from app.processors.slam.mast3r_slam import select_session_cls
        return select_session_cls()
    if backend_id == "droid_slam":
        from app.processors.slam.droid_slam import select_session_cls
        return select_session_cls()
    if backend_id == "dpvo":
        from app.processors.slam.dpvo import select_session_cls
        return select_session_cls()
    # Unknown backend → simulated. Captures still produce a poseable
    # result, just without real reconstruction quality.
    from app.processors.slam.tracker import SimulatedSlamSession
    return SimulatedSlamSession


SUPPORTED_BACKENDS = ("mast3r_slam", "droid_slam", "dpvo")


def is_supported(backend_id: Optional[str]) -> bool:
    return backend_id in SUPPORTED_BACKENDS
