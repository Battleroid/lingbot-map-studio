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


def select_session_cls() -> type[SlamSession]:
    """Auto-select the CUDA session when its dependencies are all
    importable; fall back to the simulated session otherwise."""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return _DpvoSession
    if not torch.cuda.is_available():
        return _DpvoSession
    try:
        # The upstream package is `dpvo`; the tracker class lives at
        # `dpvo.dpvo.DPVO`. Probe the import the CUDA session uses so a
        # missing C++ extension surfaces here, not on the first frame.
        import dpvo.dpvo  # noqa: F401, PLC0415
        from app.processors.slam.dpvo_cuda import DpvoCudaSession  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        log.info(
            "dpvo: real CUDA tracker not importable (%s); using simulated",
            exc,
        )
        return _DpvoSession
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
