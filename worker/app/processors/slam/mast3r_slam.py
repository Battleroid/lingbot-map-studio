"""MASt3R-SLAM backend.

Calibration-free — our default recommendation for analog FPV footage where
fx/fy are unknown. Until the upstream CUDA build lands we stand in with a
simulated session tuned to MASt3R's sparser-but-cleaner keyframe cadence.
"""

from __future__ import annotations

import logging
from typing import ClassVar

from app.processors.slam.base import SlamProcessor, SlamSession
from app.processors.slam.tracker import SimulatedSlamSession

log = logging.getLogger(__name__)


class _Mast3rSlamSession(SimulatedSlamSession):
    """MASt3R flavour: sparser keyframes, cleaner cloud per keyframe."""

    backend_id: ClassVar[str] = "mast3r_slam"
    corners_per_keyframe: ClassVar[int] = 160
    # MASt3R-SLAM picks keyframes sparingly; emulate by taking every 2nd
    # tracked frame as a keyframe. The processor's score-gate still runs
    # first.
    keyframe_every: ClassVar[int] = 2
    depth_noise: ClassVar[float] = 0.1


class Mast3rSlamProcessor(SlamProcessor):
    """MASt3R-SLAM. Calibration-free; best default for analog FPV."""

    id: ClassVar[str] = "mast3r_slam"  # type: ignore[assignment]
    display_name: ClassVar[str] = "MASt3R-SLAM"

    def _make_session(self, ctx) -> SlamSession:  # type: ignore[override]
        log.info("mast3r_slam: using simulated tracker (upstream not installed)")
        return _Mast3rSlamSession()
