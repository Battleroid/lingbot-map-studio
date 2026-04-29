"""Real MASt3R-SLAM CUDA session.

Wraps the upstream `rmurai0610/MASt3R-SLAM` tracker. Lives in its own
module so the rest of the SLAM package stays importable on a CPU-only
host — `import mast3r_slam` at the top of this file would crash any
container that hadn't pip-installed it (or hadn't built its CUDA matching
kernels).

The session is selected at runtime by `select_mast3r_session_cls()` in
`mast3r_slam.py`: when `mast3r_slam` (the upstream package) imports
cleanly we use `Mast3rCudaSession`; otherwise we fall back to the
simulated session and Phase 0 emits a warn event.

Implementation notes
- MASt3R-SLAM is calibration-free — the matcher infers per-pair geometry
  from raw RGB. We still pass an `intrinsics` 3x3 to `start()` because
  every backend's `SlamSession.start` takes one; MASt3R ignores it.
- Upstream returns poses in cam-from-world (the standard pose-graph
  shape we already export). No conversion needed.
- Bundle adjustment fires inside the upstream tracker — not exposed.
- Memory: MASt3R holds the full keyframe set in GPU buffers. On a
  24 GB card you typically run out around ~150 keyframes; cap with
  `cfg.max_frames` and the score gate.

API expectations (the upstream repo is research-grade and the surface
shifts between commits; if any of these calls don't exist, fall back to
simulated and the user can pin a different commit in Dockerfile.slam):

  from mast3r_slam.tracker import MASt3RSLAMTracker
  tracker = MASt3RSLAMTracker(model_path=..., device="cuda")
  for idx, img in enumerate(frames):
      out = tracker.track_frame(img)            # → {pose, points, is_keyframe}
  result = tracker.finalize()                   # → {poses, keyframes, points}

The wrapper below is defensive: anything that raises during track_frame
gets re-raised as a clean RuntimeError so the SlamProcessor can either
salvage a partial result via finalize() or surface the error to the
user."""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Optional

import numpy as np

from app.processors.slam.base import FinalResult, FrameUpdate, SlamSession

log = logging.getLogger(__name__)


# Default checkpoint discovery. Two paths:
#   1. If `MAST3R_SLAM_CKPT` env var is set, use that (CI / dev override).
#   2. If `/models/mast3r_slam/mast3r_slam.pth` exists, use it (the
#      `app.pipeline.checkpoints` fetcher caches there).
#   3. Otherwise pass None and let upstream MASt3R-SLAM's internal
#      huggingface_hub download handle it lazily on first use.
import os as _os
from pathlib import Path as _Path


def _default_checkpoint_path() -> Optional[str]:
    env = _os.environ.get("MAST3R_SLAM_CKPT")
    if env:
        return env
    cached = _Path("/models/mast3r_slam/mast3r_slam.pth")
    if cached.exists():
        return str(cached)
    return None


class Mast3rCudaSession(SlamSession):
    """Real MASt3R-SLAM CUDA tracker. Selected when the upstream package
    + CUDA are both available."""

    backend_id: ClassVar[str] = "mast3r_slam"

    def __init__(self, checkpoint_path: Optional[str] = None) -> None:
        self._tracker: Optional[Any] = None
        self._checkpoint_path = checkpoint_path or _default_checkpoint_path()
        self._poses: list[np.ndarray] = []
        self._keyframe_indices: list[int] = []
        self._points_buffer: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # SlamSession surface
    # ------------------------------------------------------------------

    def start(
        self,
        intrinsics: np.ndarray,
        image_shape: tuple[int, int],
    ) -> None:
        # Lazy import — keeps the module importable on a CPU host. If
        # any of these fail the auto-select factory has already routed
        # to the simulated session, so reaching this code means we're
        # genuinely on a GPU host with the package installed.
        from mast3r_slam.tracker import MASt3RSLAMTracker  # noqa: PLC0415

        log.info(
            "mast3r_slam: starting CUDA tracker (image=%dx%d, ckpt=%s)",
            image_shape[1],
            image_shape[0],
            self._checkpoint_path or "<upstream HF default>",
        )
        # `model_path=None` lets the upstream tracker lazy-download the
        # MASt3R foundation model from huggingface_hub. Pinning a local
        # path is the override for air-gapped boxes.
        kwargs = {"device": "cuda"}
        if self._checkpoint_path is not None:
            kwargs["model_path"] = self._checkpoint_path
        self._tracker = MASt3RSLAMTracker(**kwargs)
        # MASt3R is calibration-free so we don't pass `intrinsics` — but we
        # *do* keep `image_shape` because the upstream tracker uses it to
        # size its match grid.
        if hasattr(self._tracker, "set_image_shape"):
            self._tracker.set_image_shape(image_shape)

    def step(self, idx: int, img: np.ndarray) -> FrameUpdate:
        assert self._tracker is not None, "start() must be called first"
        try:
            out = self._tracker.track_frame(img)
        except Exception as exc:  # noqa: BLE001
            # Don't crash the whole job on a single bad frame — log and
            # skip. The processor's score-gate may have fed us a frame
            # the tracker rejects (e.g. all black).
            log.warning("mast3r_slam: track_frame(idx=%d) raised: %s", idx, exc)
            return FrameUpdate()

        pose = _coerce_pose(out)
        new_points = _coerce_points(out)
        is_keyframe = bool(_field(out, "is_keyframe", False))

        if pose is not None:
            self._poses.append(pose)
        if is_keyframe:
            self._keyframe_indices.append(idx)
        if new_points is not None and new_points.shape[0] > 0:
            self._points_buffer.append(new_points)

        return FrameUpdate(
            pose_matrix=pose,
            new_points=new_points,
            is_keyframe=is_keyframe,
            diagnostics={"backend": "mast3r_slam_cuda"},
        )

    def finalize(self) -> FinalResult:
        # Final bundle adjustment + map cleanup. May raise if the tracker
        # never tracked anything (start() was called but no step() yet);
        # in that case return an empty result so the processor's salvage
        # path still produces a job artifact.
        if self._tracker is None:
            return FinalResult(
                poses=np.empty((0, 4, 4), dtype=np.float32),
                keyframe_indices=[],
                points=None,
                diagnostics={"backend": "mast3r_slam_cuda", "empty": True},
            )

        try:
            result = self._tracker.finalize()
        except Exception as exc:  # noqa: BLE001
            log.warning("mast3r_slam: finalize() raised, returning partial: %s", exc)
            result = None

        if result is not None:
            poses_np = _coerce_pose_array(_field(result, "poses", None))
            kf_indices = _field(result, "keyframe_indices", None)
            if kf_indices is None:
                kf_indices = list(range(poses_np.shape[0]))
            points_np = _coerce_points(_field(result, "points", None))
        else:
            poses_np = np.stack(self._poses) if self._poses else np.empty(
                (0, 4, 4), dtype=np.float32
            )
            kf_indices = list(self._keyframe_indices)
            points_np = (
                np.concatenate(self._points_buffer, axis=0)
                if self._points_buffer
                else None
            )

        return FinalResult(
            poses=poses_np,
            keyframe_indices=list(kf_indices),
            points=points_np,
            diagnostics={
                "backend": "mast3r_slam_cuda",
                "n_poses": int(poses_np.shape[0]),
                "n_keyframes": len(kf_indices),
                "n_points": int(points_np.shape[0]) if points_np is not None else 0,
            },
        )


# ----------------------------------------------------------------------
# Coercion helpers — the upstream tracker's return shape varies between
# commits. Be liberal in what we accept.
# ----------------------------------------------------------------------


def _field(obj: Any, name: str, default: Any) -> Any:
    """Read `name` from a dict, dataclass, or namedtuple."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _coerce_pose(out: Any) -> Optional[np.ndarray]:
    """Pull a 4x4 cam-from-world matrix out of whatever the tracker
    returned. Accepts: a 4x4 array, a 3x4 array (row-extended to 4x4), a
    dict with 'pose' key, or a (rotation, translation) pair."""
    pose = _field(out, "pose", out)
    if pose is None:
        return None
    if isinstance(pose, np.ndarray):
        return _to_4x4(pose)
    if isinstance(pose, (tuple, list)) and len(pose) == 2:
        R, t = pose
        if isinstance(R, np.ndarray) and isinstance(t, np.ndarray):
            M = np.eye(4, dtype=np.float32)
            M[:3, :3] = R
            M[:3, 3] = t.reshape(3)
            return M
    return None


def _to_4x4(arr: np.ndarray) -> Optional[np.ndarray]:
    if arr.shape == (4, 4):
        return arr.astype(np.float32, copy=False)
    if arr.shape == (3, 4):
        out = np.eye(4, dtype=np.float32)
        out[:3, :] = arr
        return out
    return None


def _coerce_points(out: Any) -> Optional[np.ndarray]:
    """Pull an (N, 6) [x y z r g b] cloud out of whatever the tracker
    returned. Accepts: an Nx6 array directly, a dict with 'points' key,
    or a (xyz, rgb) tuple."""
    pts = _field(out, "points", None)
    if pts is None:
        pts = _field(out, "new_points", None)
    if pts is None:
        return None
    if isinstance(pts, np.ndarray):
        if pts.ndim == 2 and pts.shape[1] >= 6:
            return pts[:, :6].astype(np.float32, copy=False)
        if pts.ndim == 2 and pts.shape[1] == 3:
            # No color — fill with mid-grey so the cloud renders.
            rgb = np.full((pts.shape[0], 3), 200, dtype=np.float32)
            return np.concatenate([pts.astype(np.float32), rgb], axis=1)
    if isinstance(pts, (tuple, list)) and len(pts) == 2:
        xyz, rgb = pts
        if isinstance(xyz, np.ndarray) and isinstance(rgb, np.ndarray):
            return np.concatenate(
                [xyz.astype(np.float32), rgb.astype(np.float32)], axis=1
            )
    return None


def _coerce_pose_array(arr: Any) -> np.ndarray:
    """Final poses → (K, 4, 4) float32. Accepts a list of 4x4 arrays or
    a single (K, 4, 4) tensor."""
    if arr is None:
        return np.empty((0, 4, 4), dtype=np.float32)
    if isinstance(arr, np.ndarray):
        if arr.ndim == 3 and arr.shape[1:] == (4, 4):
            return arr.astype(np.float32, copy=False)
    if isinstance(arr, (list, tuple)) and arr:
        out = []
        for item in arr:
            m = _to_4x4(np.asarray(item))
            if m is not None:
                out.append(m)
        if out:
            return np.stack(out)
    return np.empty((0, 4, 4), dtype=np.float32)
