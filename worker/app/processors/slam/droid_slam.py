"""DROID-SLAM backend.

Dense flow-based tracker with global bundle adjustment. Auto-selects
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


class _DroidSlamSession(SimulatedSlamSession):
    """Simulated DROID flavour: dense per-frame keyframes, fatter cloud
    per keyframe. Used as the fallback when the real CUDA tracker isn't
    available."""

    backend_id: ClassVar[str] = "droid_slam"
    corners_per_keyframe: ClassVar[int] = 256
    keyframe_every: ClassVar[int] = 1
    depth_noise: ClassVar[float] = 0.15


def select_session_cls() -> type[SlamSession]:
    """Auto-select the CUDA session when its dependencies are all
    importable; fall back to the simulated session otherwise."""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return _DroidSlamSession
    if not torch.cuda.is_available():
        return _DroidSlamSession
    try:
        # Probe the import surface used by the CUDA session. The package
        # ships custom CUDA extensions under `droid_backends`; if that
        # native module is missing the import will raise.
        import droid_slam  # noqa: F401, PLC0415
        from app.processors.slam.droid_cuda import DroidCudaSession  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        log.info(
            "droid_slam: real CUDA tracker not importable (%s); using simulated",
            exc,
        )
        return _DroidSlamSession
    return DroidCudaSession


class DroidSlamProcessor(SlamProcessor):
    """DROID-SLAM. Dense flow-based, VRAM-heavy, best global consistency.

    Pinned to the `slam` worker class. Auto-selects the real CUDA tracker
    when available; falls back to the simulated session with a warn
    event when it isn't."""

    id: ClassVar[str] = "droid_slam"  # type: ignore[assignment]
    display_name: ClassVar[str] = "DROID-SLAM"

    def _make_session(self, ctx) -> SlamSession:  # type: ignore[override]
        cls = select_session_cls()
        return cls()
