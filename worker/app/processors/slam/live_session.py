"""Live-capture wrapper around the existing SLAM sessions.

The capture WebSocket handler in `app.cloud.capture_session.CaptureSession`
needs a `SlamSession`-shaped object (start / step / finalize) to drive
in real time. Each backend (MASt3R-SLAM, DROID-SLAM, DPVO, MonoGS)
already exposes that contract via
`worker/app/processors/slam/base.py:SlamSession`.

The production resolvers (`select_session_cls()` per backend) raise
typed `*UnavailableError` exceptions when their real CUDA stack is
missing — that's what enforces "no simulated splat output anywhere
in production". The api container running the live capture has no
GPU and can't satisfy any of those resolvers, but it still needs
*some* SlamSession to drive the live points-overlay / pose feedback
the user sees on screen during a scan.

This module catches each backend's Unavailable error and substitutes
the corresponding simulated class — the live preview is allowed to
be approximate because:
  - it's never persisted as a final result (the captured frames are
    queued for re-processing on the GPU worker after stop),
  - and the user explicitly asked for *some* real-time visual cue,
    which the simulated tracker's pose + sparse points provide.

So: production reconstruction == strict real backend or fail loud.
Live preview == best-effort, falls through to simulated."""

from __future__ import annotations

from typing import Optional

from app.processors.slam.base import SlamSession


def resolve_live_session(backend_id: str) -> SlamSession:
    """Pick the right SlamSession subclass for `backend_id` and
    instantiate it. Real backend if importable in this process,
    simulated stand-in if not. Unknown backend → simulated."""
    cls = _resolve_cls(backend_id)
    return cls()


def _resolve_cls(backend_id: str) -> type[SlamSession]:
    if backend_id == "mast3r_slam":
        from app.processors.slam.mast3r_slam import (
            Mast3rSlamUnavailableError,
            _Mast3rSlamSession,
            select_session_cls,
        )

        try:
            return select_session_cls()
        except Mast3rSlamUnavailableError:
            return _Mast3rSlamSession
    if backend_id == "droid_slam":
        from app.processors.slam.droid_slam import (
            DroidSlamUnavailableError,
            _DroidSlamSession,
            select_session_cls,
        )

        try:
            return select_session_cls()
        except DroidSlamUnavailableError:
            return _DroidSlamSession
    if backend_id == "dpvo":
        from app.processors.slam.dpvo import (
            DpvoUnavailableError,
            _DpvoSession,
            select_session_cls,
        )

        try:
            return select_session_cls()
        except DpvoUnavailableError:
            return _DpvoSession
    if backend_id == "monogs":
        from app.processors.gsplat.monogs import (
            MonogsSessionUnavailableError,
            _MonogsSession,
            select_session_cls,
        )

        try:
            return select_session_cls()
        except MonogsSessionUnavailableError:
            return _MonogsSession
    # Unknown backend → simulated. Captures still produce a poseable
    # result for the live preview; the post-stop job will fail at
    # claim time anyway because the queued config has no matching
    # processor.
    from app.processors.slam.tracker import SimulatedSlamSession
    return SimulatedSlamSession


SUPPORTED_BACKENDS = ("mast3r_slam", "droid_slam", "dpvo", "monogs")


def is_supported(backend_id: Optional[str]) -> bool:
    return backend_id in SUPPORTED_BACKENDS
