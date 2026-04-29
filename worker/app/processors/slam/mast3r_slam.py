"""MASt3R-SLAM backend.

Calibration-free — our default recommendation for footage where the
camera intrinsics aren't known (most phone / drone clips). Auto-selects
the real CUDA tracker when the upstream package + CUDA are available;
falls back to the simulated session otherwise (Phase 0 emits a warn
event so the user sees it in the log pane).
"""

from __future__ import annotations

import logging
from typing import ClassVar

from app.processors.slam.base import SlamProcessor, SlamSession
from app.processors.slam.tracker import SimulatedSlamSession

log = logging.getLogger(__name__)


class _Mast3rSlamSession(SimulatedSlamSession):
    """Simulated MASt3R flavour: sparser keyframes, cleaner cloud per
    keyframe. Used as the fallback when the real CUDA tracker isn't
    available."""

    backend_id: ClassVar[str] = "mast3r_slam"
    corners_per_keyframe: ClassVar[int] = 160
    # MASt3R-SLAM picks keyframes sparingly; emulate by taking every 2nd
    # tracked frame as a keyframe. The processor's score-gate still runs
    # first.
    keyframe_every: ClassVar[int] = 2
    depth_noise: ClassVar[float] = 0.1


def select_session_cls() -> type[SlamSession]:
    """Auto-select the CUDA session when its dependencies are all
    importable; fall back to the simulated session otherwise.

    Three failure modes roll into the same fallback path:
      * `torch` not installed (CPU dev box).
      * `torch.cuda.is_available()` False (worker-slam without GPU).
      * `mast3r_slam` package not installed (image hasn't shipped Phase 2
        yet, or the source build failed at image build time).

    Phase 0's warn event fires in the SlamProcessor when the returned
    session is a SimulatedSlamSession instance, so the user always
    sees in-UI signal that they're watching a placeholder.
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return _Mast3rSlamSession
    if not torch.cuda.is_available():
        return _Mast3rSlamSession
    try:
        # Probe the import surface used by the CUDA session. If any of
        # these fail (missing wheel, source build failed, CUDA mismatch),
        # the wrapper module's own internal import of
        # `mast3r_slam.tracker` would have raised the same error — better
        # to detect at session-pick time than at first frame.
        import mast3r_slam.tracker  # noqa: F401, PLC0415
        from app.processors.slam.mast3r_cuda import Mast3rCudaSession  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        log.info(
            "mast3r_slam: real CUDA tracker not importable (%s); using simulated",
            exc,
        )
        return _Mast3rSlamSession
    return Mast3rCudaSession


class Mast3rSlamProcessor(SlamProcessor):
    """MASt3R-SLAM. Calibration-free; the safe default for footage with
    unknown intrinsics. Auto-selects the real CUDA tracker when
    available; falls back to the simulated session with a warn event
    when it isn't."""

    id: ClassVar[str] = "mast3r_slam"  # type: ignore[assignment]
    display_name: ClassVar[str] = "MASt3R-SLAM"

    def _make_session(self, ctx) -> SlamSession:  # type: ignore[override]
        cls = select_session_cls()
        return cls()
