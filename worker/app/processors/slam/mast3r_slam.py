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


class Mast3rSlamUnavailableError(RuntimeError):
    """Raised by `select_session_cls()` when the real MASt3R-SLAM CUDA
    tracker can't be loaded. Mirrors the gsplat / MonoGS strict
    pattern: we no longer silently fall back to `_Mast3rSlamSession`,
    since the simulated tracker produces synthetic geometry that the
    user has called out as "useless". The live-capture wrapper in
    `live_session.py` catches this and uses the simulated class for
    the live preview only — production reconstruction always errors
    out loud rather than shipping placeholder output as if it were
    real."""


def select_session_cls() -> type[SlamSession]:
    """Resolve the real MASt3R-SLAM CUDA session. Raises
    `Mast3rSlamUnavailableError` with install instructions when any
    prerequisite is missing. No simulated fallback in production.
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:
        raise Mast3rSlamUnavailableError(
            "mast3r_slam: torch is not installed in this worker. The "
            "real CUDA tracker requires the worker-slam image (built "
            "from worker/Dockerfile.slam)."
        ) from exc
    if not torch.cuda.is_available():
        raise Mast3rSlamUnavailableError(
            "mast3r_slam: torch.cuda.is_available() is False. The "
            "tracker needs an NVIDIA GPU + nvidia-container-toolkit "
            "passthrough; check the worker-slam container's "
            "deploy.resources.reservations.devices in docker-compose.yml."
        )
    try:
        import mast3r_slam.tracker  # noqa: F401, PLC0415
        from app.processors.slam.mast3r_cuda import Mast3rCudaSession  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        raise Mast3rSlamUnavailableError(
            "mast3r_slam: the upstream MASt3R-SLAM source isn't "
            f"importable in this worker ({type(exc).__name__}: {exc}). "
            "Check that worker/Dockerfile.slam clones MASt3R-SLAM into "
            "the image and that its CUDA submodule build succeeded."
        ) from exc
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
