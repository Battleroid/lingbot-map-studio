"""DPVO (Deep Patch Visual Odometry) backend.

Lightweight patch-based VO. Trajectory-only — DPVO emits a very sparse
cloud that's often visually noisier than the camera path on its own.
Pair with a follow-on gsplat job for a dense scene.

Auto-selects the real CUDA tracker when the upstream package + CUDA
are available; falls back to the simulated session otherwise (Phase 0
emits a warn event so the user sees it in the log pane).
"""

from __future__ import annotations

import logging
from typing import ClassVar

from app.processors.slam.base import SlamProcessor, SlamSession
from app.processors.slam.tracker import SimulatedSlamSession

log = logging.getLogger(__name__)


class _DpvoSession(SimulatedSlamSession):
    """Simulated trajectory-only flavour. The processor's point-cloud
    export is skipped for trajectory_only sessions, matching DPVO's
    real output shape."""

    backend_id: ClassVar[str] = "dpvo"
    trajectory_only: ClassVar[bool] = True
    corners_per_keyframe: ClassVar[int] = 48
    keyframe_every: ClassVar[int] = 3


class DpvoUnavailableError(RuntimeError):
    """Raised when the real DPVO CUDA tracker can't be loaded. Same
    strict-no-fallback policy as the other backends."""


def select_session_cls() -> type[SlamSession]:
    """Resolve the real DPVO CUDA session. Raises
    `DpvoUnavailableError` with install instructions when any
    prerequisite is missing."""
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:
        raise DpvoUnavailableError(
            "dpvo: torch is not installed in this worker. The real "
            "CUDA tracker requires the worker-slam image."
        ) from exc
    if not torch.cuda.is_available():
        raise DpvoUnavailableError(
            "dpvo: torch.cuda.is_available() is False. The tracker "
            "needs an NVIDIA GPU + nvidia-container-toolkit "
            "passthrough."
        )
    try:
        import dpvo.dpvo  # noqa: F401, PLC0415
        from app.processors.slam.dpvo_cuda import DpvoCudaSession  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        raise DpvoUnavailableError(
            "dpvo: the upstream DPVO source isn't importable in this "
            f"worker ({type(exc).__name__}: {exc}). Check the dpvo "
            "install + C++ extension build in worker/Dockerfile.slam."
        ) from exc
    return DpvoCudaSession


class DpvoProcessor(SlamProcessor):
    """DPVO. Lightweight patch-based VO; trajectory-only.

    Auto-selects the real CUDA tracker when available; falls back to
    the simulated session with a warn event when it isn't."""

    id: ClassVar[str] = "dpvo"  # type: ignore[assignment]
    display_name: ClassVar[str] = "DPVO"

    def _make_session(self, ctx) -> SlamSession:  # type: ignore[override]
        cls = select_session_cls()
        return cls()
