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


class DroidSlamUnavailableError(RuntimeError):
    """Raised when the real DROID-SLAM CUDA tracker can't be loaded.
    Same strict-no-fallback policy as the other backends — no
    silent simulated output in production."""


def select_session_cls() -> type[SlamSession]:
    """Resolve the real DROID-SLAM CUDA session. Raises
    `DroidSlamUnavailableError` with install instructions when any
    prerequisite is missing."""
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:
        raise DroidSlamUnavailableError(
            "droid_slam: torch is not installed in this worker. The "
            "real CUDA tracker requires the worker-slam image."
        ) from exc
    if not torch.cuda.is_available():
        raise DroidSlamUnavailableError(
            "droid_slam: torch.cuda.is_available() is False. The "
            "tracker needs an NVIDIA GPU + nvidia-container-toolkit "
            "passthrough."
        )
    try:
        import droid_slam  # noqa: F401, PLC0415
        from app.processors.slam.droid_cuda import DroidCudaSession  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        raise DroidSlamUnavailableError(
            "droid_slam: the upstream DROID-SLAM source isn't "
            f"importable in this worker ({type(exc).__name__}: {exc}). "
            "Check the droid-slam install + CUDA-extension build in "
            "worker/Dockerfile.slam."
        ) from exc
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
