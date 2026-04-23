"""DROID-SLAM backend.

When the upstream CUDA build is available (inside `worker-slam`) this module
imports the real tracker. Until then it falls back to a DROID-flavoured
simulated session so the registry resolves and jobs run end-to-end.

Subclasses keep the concrete `SlamProcessor` tiny: all the orchestration
lives in `base.SlamProcessor`; a backend only has to plug in the session.
"""

from __future__ import annotations

import logging
from typing import ClassVar

from app.processors.slam.base import SlamProcessor, SlamSession
from app.processors.slam.tracker import SimulatedSlamSession

log = logging.getLogger(__name__)


class _DroidSlamSession(SimulatedSlamSession):
    """Placeholder that mimics DROID's dense, keyframe-heavy trajectory.

    Real DROID-SLAM buffers many keyframes and runs a global bundle
    adjustment at the end. Until the upstream CUDA ops are wired in we
    fake the shape by keeping every tracked frame and generating denser
    per-keyframe clouds.
    """

    backend_id: ClassVar[str] = "droid_slam"
    # DROID keeps dense coverage; emit more points per keyframe.
    corners_per_keyframe: ClassVar[int] = 256
    keyframe_every: ClassVar[int] = 1
    depth_noise: ClassVar[float] = 0.15


class DroidSlamProcessor(SlamProcessor):
    """DROID-SLAM. Dense flow-based, VRAM-heavy, best global consistency.

    Pinned to the `slam` worker class. The real torch extension and the
    upstream DROID checkpoint are resolved via
    `pipeline.checkpoints.fetch_checkpoint(processor_id="droid_slam", ...)`.
    """

    id: ClassVar[str] = "droid_slam"  # type: ignore[assignment]
    display_name: ClassVar[str] = "DROID-SLAM"

    def _make_session(self, ctx) -> SlamSession:  # type: ignore[override]
        # Once `droid_slam` is installed in worker-slam, swap this with a
        # real session that wraps `droid_slam.DroidSlam(...)`. The
        # SlamSession interface (start/step/finalize) is a direct match
        # for DROID's per-frame API, so the wrapper is thin.
        log.info("droid_slam: using simulated tracker (upstream not installed)")
        return _DroidSlamSession()
