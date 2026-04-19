"""DPVO (Deep Patch Visual Odometry) backend.

Lightweight patch-based VO. Trajectory-only by default — DPVO emits a very
sparse cloud that's often visually noisier than just showing the camera
path. Run Poisson meshing off; pair with a follow-on gsplat job for a
dense scene.
"""

from __future__ import annotations

import logging
from typing import ClassVar

from app.processors.slam.base import SlamProcessor, SlamSession
from app.processors.slam.tracker import SimulatedSlamSession

log = logging.getLogger(__name__)


class _DpvoSession(SimulatedSlamSession):
    """Trajectory-only flavour. The processor's point-cloud export is
    skipped for trajectory_only sessions, matching DPVO's real output
    shape."""

    backend_id: ClassVar[str] = "dpvo"
    trajectory_only: ClassVar[bool] = True
    # Sparse cloud per keyframe even if we flip trajectory_only off.
    corners_per_keyframe: ClassVar[int] = 48
    keyframe_every: ClassVar[int] = 3


class DpvoProcessor(SlamProcessor):
    """DPVO. Lightweight patch-based VO; trajectory-only by default."""

    id: ClassVar[str] = "dpvo"  # type: ignore[assignment]
    display_name: ClassVar[str] = "DPVO"

    def _make_session(self, ctx) -> SlamSession:  # type: ignore[override]
        log.info("dpvo: using simulated tracker (upstream not installed)")
        return _DpvoSession()
