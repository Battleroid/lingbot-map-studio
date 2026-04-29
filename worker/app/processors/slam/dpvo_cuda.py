"""Real DPVO CUDA session.

Wraps the upstream `princeton-vl/DPVO` patch-based deep VO tracker.
Lives in its own module so the SLAM package stays importable on a
CPU-only host.

The session is selected at runtime by `select_session_cls()` in
`dpvo.py`: when `dpvo` (the upstream package) imports cleanly we use
`DpvoCudaSession`; otherwise we fall back to the simulated session and
Phase 0 emits a warn event.

Implementation notes
- DPVO is trajectory-oriented. Its sparse patch-based output is a
  scattered cloud of low confidence — often visually noisier than
  camera-path-only rendering. We expose `points` from the buffer for
  completeness but the SlamSession's `trajectory_only=True` (still
  inherited from `_DpvoSession`) is the right user default; users who
  want dense geometry should pair DPVO + gsplat training instead.
- Like DROID, DPVO needs `(fx, fy, cx, cy)` as a 4-vector per frame.
  Pull from the K matrix at start() time and reuse for every step.
- The upstream API is in `dpvo.dpvo.DPVO`. Per-frame: `__call__(idx,
  image_tensor, intrinsics_tensor)`; per-trajectory pull: `terminate()`
  returns `(N, 7)` rows in `[tx, ty, tz, qx, qy, qz, qw]` shape (same
  schema as DROID; we reuse the conversion helper).
- Buffer size cap is much smaller than DROID — the patch model lives
  in ~2-4 GB of VRAM. Fits on 8 GB cards.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, ClassVar, Optional

import numpy as np

from app.processors.slam.base import FinalResult, FrameUpdate, SlamSession

log = logging.getLogger(__name__)


def _default_checkpoint_path() -> Optional[str]:
    env = os.environ.get("DPVO_CKPT")
    if env:
        return env
    cached = Path("/models/dpvo/dpvo.pth")
    if cached.exists():
        return str(cached)
    return None


def _quat_xyzw_translate_to_matrix(pose_7: np.ndarray) -> np.ndarray:
    """`[tx, ty, tz, qx, qy, qz, qw]` row → 4x4 cam-from-world. Same
    convention as DROID; reuse the math here so we don't add a
    cross-module dep just for one helper."""
    t = pose_7[:3]
    qx, qy, qz, qw = pose_7[3:7]
    norm = (qw * qw + qx * qx + qy * qy + qz * qz) ** 0.5 or 1.0
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    R = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R
    M[:3, 3] = t.astype(np.float32)
    return M


def _bgr_to_torch(img_bgr: np.ndarray):
    """BGR uint8 (H, W, 3) → torch float (3, H, W) in [0, 255] on cuda.
    DPVO's reference loader keeps inputs in 0-255 range — no /255
    normalization."""
    import torch  # noqa: PLC0415

    rgb = img_bgr[..., ::-1].copy()
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float()
    return tensor.to("cuda")


class DpvoCudaSession(SlamSession):
    """Real DPVO CUDA tracker. Selected when the upstream package +
    CUDA are both available."""

    backend_id: ClassVar[str] = "dpvo"
    # DPVO produces sparse patches; the SlamProcessor's cloud export is
    # skipped for trajectory_only sessions, matching the user expectation
    # that DPVO is a "camera path only" backend.
    trajectory_only: ClassVar[bool] = True

    def __init__(self, checkpoint_path: Optional[str] = None) -> None:
        self._checkpoint_path = checkpoint_path or _default_checkpoint_path()
        self._tracker: Optional[Any] = None
        self._k4_tensor: Optional[Any] = None
        self._image_shape: Optional[tuple[int, int]] = None
        self._n_frames = 0
        self._keyframe_indices: list[int] = []

    # ------------------------------------------------------------------
    # SlamSession surface
    # ------------------------------------------------------------------

    def start(
        self,
        intrinsics: np.ndarray,
        image_shape: tuple[int, int],
    ) -> None:
        # Lazy-import so the module is safe to import on any host.
        from dpvo.dpvo import DPVO  # noqa: PLC0415
        from dpvo.config import cfg as _dpvo_cfg  # noqa: PLC0415
        import torch  # noqa: PLC0415

        h, w = image_shape
        self._image_shape = (h, w)
        # DPVO's reference loader takes a 4-vec [fx, fy, cx, cy] cuda
        # tensor that's reused across calls.
        k4 = np.array(
            [intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]],
            dtype=np.float32,
        )
        self._k4_tensor = torch.from_numpy(k4).to("cuda")

        log.info(
            "dpvo: starting CUDA tracker (image=%dx%d, ckpt=%s)",
            w, h, self._checkpoint_path or "<unset>",
        )
        # DPVO's constructor signature is `(cfg, network, ht, wd, viz=False)`.
        # `cfg` is a YACS-style config; we accept the upstream defaults.
        self._tracker = DPVO(
            _dpvo_cfg, self._checkpoint_path, ht=h, wd=w, viz=False
        )

    def step(self, idx: int, img: np.ndarray) -> FrameUpdate:
        assert self._tracker is not None and self._k4_tensor is not None
        try:
            frame = _bgr_to_torch(img)
            # Upstream's __call__ shape is `(tstamp, image, intrinsics)`.
            self._tracker(idx, frame, self._k4_tensor)
            self._n_frames += 1
        except Exception as exc:  # noqa: BLE001
            log.warning("dpvo: step(idx=%d) raised: %s", idx, exc)
            return FrameUpdate()

        # DPVO doesn't expose per-frame keyframe markers; stride
        # heuristically so the partial-snapshot cadence still emits
        # something. Real keyframe selection lives in the patch graph
        # and is reflected in the final terminate() trajectory length.
        is_keyframe = (idx % 6 == 0)
        if is_keyframe:
            self._keyframe_indices.append(idx)
        return FrameUpdate(
            pose_matrix=None,  # DPVO doesn't expose per-frame pre-BA pose
            new_points=None,   # trajectory_only — skip cloud
            is_keyframe=is_keyframe,
            diagnostics={"backend": "dpvo_cuda"},
        )

    def finalize(self) -> FinalResult:
        if self._tracker is None:
            return FinalResult(
                poses=np.empty((0, 4, 4), dtype=np.float32),
                keyframe_indices=[],
                points=None,
                diagnostics={"backend": "dpvo_cuda", "empty": True},
            )

        try:
            traj = self._tracker.terminate()
        except Exception as exc:  # noqa: BLE001
            log.warning("dpvo: terminate() raised: %s", exc)
            traj = None

        # Convert (N, 7) [tx ty tz qx qy qz qw] rows → (N, 4, 4).
        if (
            isinstance(traj, np.ndarray)
            and traj.ndim == 2
            and traj.shape[1] >= 7
        ):
            poses = np.stack(
                [_quat_xyzw_translate_to_matrix(row) for row in traj]
            )
        else:
            poses = np.empty((0, 4, 4), dtype=np.float32)

        return FinalResult(
            poses=poses,
            keyframe_indices=list(self._keyframe_indices),
            points=None,  # trajectory_only — DPVO doesn't ship a dense cloud
            diagnostics={
                "backend": "dpvo_cuda",
                "n_poses": int(poses.shape[0]),
                "n_keyframes": len(self._keyframe_indices),
                "n_tracked_frames": self._n_frames,
            },
        )
