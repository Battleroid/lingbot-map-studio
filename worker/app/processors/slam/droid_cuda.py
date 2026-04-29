"""Real DROID-SLAM CUDA session.

Wraps the upstream `princeton-vl/DROID-SLAM` tracker. Lives in its own
module so the rest of the SLAM package stays importable on a CPU-only
host — `import droid_slam` at the top of this file would crash any
container without the package + its compiled CUDA extensions.

The session is selected at runtime by `select_session_cls()` in
`droid_slam.py`: when `droid_slam` (the upstream package) imports cleanly
we use `DroidCudaSession`; otherwise we fall back to the simulated
session and Phase 0 emits a warn event.

Implementation notes
- DROID-SLAM is dense and keyframe-heavy. On a 24 GB card you typically
  fit ~150-300 keyframes depending on `buffer_size`; longer clips need
  a smaller `buffer_size` or the score gate.
- Calibration matters. Unlike MASt3R, DROID expects intrinsics as a
  4-vector `(fx, fy, cx, cy)` per frame. We pull from the SlamSession
  start() argument's K matrix.
- Final bundle adjustment fires inside `droid.terminate()`, which also
  produces the final pose trajectory. We capture it there and then
  format into the FinalResult our SlamProcessor expects.
- The upstream API expects images as torch tensors `(B, 3, H, W)` in
  [0, 1] float. We convert from BGR uint8 here so the SlamSession
  contract (BGR uint8 numpy) stays unchanged.

API expectations (the upstream repo's API has been stable since v1.0.0;
adjust if it shifts):

  from droid_slam import Droid
  args = SimpleNamespace(
      weights=..., disable_vis=True, image_size=[h, w],
      buffer=512, beta=0.3, filter_thresh=2.4,
      warmup=8, keyframe_thresh=4.0, frontend_thresh=16.0,
      frontend_window=25, frontend_radius=2, frontend_nms=1,
      backend_thresh=22.0, backend_radius=2, backend_nms=3,
      stereo=False, upsample=False,
  )
  droid = Droid(args)
  for tstamp, img_tensor, K_4vec in stream:
      droid.track(tstamp, img_tensor, intrinsics=K_4vec)
  trajectory = droid.terminate(stream)  # → (N, 7) tx ty tz qx qy qz qw
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, Optional

import numpy as np

from app.processors.slam.base import FinalResult, FrameUpdate, SlamSession

log = logging.getLogger(__name__)


def _default_checkpoint_path() -> Optional[str]:
    """Discovery order: env override → cached /models path → None
    (let upstream / huggingface_hub default kick in)."""
    env = os.environ.get("DROID_SLAM_CKPT")
    if env:
        return env
    cached = Path("/models/droid_slam/droid.pth")
    if cached.exists():
        return str(cached)
    return None


def _intrinsics_to_4vec(K: np.ndarray) -> np.ndarray:
    """DROID wants `(fx, fy, cx, cy)` per frame, not a 3x3 K. Pull
    those entries straight out."""
    return np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float32)


def _quat_xyzw_translate_to_matrix(pose_7: np.ndarray) -> np.ndarray:
    """DROID's terminate() emits trajectory rows as
    `[tx, ty, tz, qx, qy, qz, qw]`. Convert one row to a 4x4
    cam-from-world matrix."""
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
    """BGR uint8 (H, W, 3) → torch float (3, H, W) in [0, 1] on cuda."""
    import torch  # noqa: PLC0415

    # OpenCV reads BGR; DROID's reference demo converts to RGB first.
    rgb = img_bgr[..., ::-1].copy()
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float() / 255.0
    return tensor.to("cuda")


class DroidCudaSession(SlamSession):
    """Real DROID-SLAM CUDA tracker. Selected when the upstream package
    + CUDA are both available."""

    backend_id: ClassVar[str] = "droid_slam"

    def __init__(self, checkpoint_path: Optional[str] = None) -> None:
        self._checkpoint_path = checkpoint_path or _default_checkpoint_path()
        self._tracker: Optional[Any] = None
        # Buffer the (tstamp, frame_tensor, K_4vec) triples DROID needs
        # to reconstruct the trajectory at terminate() time. The reference
        # demo does the same — terminate() walks the stream a second time
        # for the global BA pass.
        self._stream: list[tuple[int, Any, np.ndarray]] = []
        self._k4: Optional[np.ndarray] = None
        self._image_shape: Optional[tuple[int, int]] = None
        # Locally-buffered partial state in case finalize() is called
        # mid-cancel before terminate() succeeds.
        self._poses: list[np.ndarray] = []
        self._keyframe_indices: list[int] = []

    # ------------------------------------------------------------------
    # SlamSession surface
    # ------------------------------------------------------------------

    def start(
        self,
        intrinsics: np.ndarray,
        image_shape: tuple[int, int],
    ) -> None:
        # Lazy import — the import only succeeds when worker-slam shipped
        # with the upstream package + its compiled CUDA extensions.
        from droid_slam import Droid  # noqa: PLC0415

        h, w = image_shape
        self._image_shape = (h, w)
        self._k4 = _intrinsics_to_4vec(intrinsics)

        # DROID's reference args. `weights` is the checkpoint path;
        # `disable_vis=True` keeps the GUI window from spawning on a
        # headless worker. `buffer` caps the keyframe count to avoid
        # OOM on long clips.
        args = SimpleNamespace(
            weights=self._checkpoint_path,
            disable_vis=True,
            image_size=[h, w],
            buffer=512,
            beta=0.3,
            filter_thresh=2.4,
            warmup=8,
            keyframe_thresh=4.0,
            frontend_thresh=16.0,
            frontend_window=25,
            frontend_radius=2,
            frontend_nms=1,
            backend_thresh=22.0,
            backend_radius=2,
            backend_nms=3,
            stereo=False,
            upsample=False,
        )
        log.info(
            "droid_slam: starting CUDA tracker (image=%dx%d, ckpt=%s)",
            w, h, self._checkpoint_path or "<unset>",
        )
        self._tracker = Droid(args)

    def step(self, idx: int, img: np.ndarray) -> FrameUpdate:
        assert self._tracker is not None and self._k4 is not None, "start() not called"
        try:
            frame = _bgr_to_torch(img)
            self._tracker.track(idx, frame, intrinsics=self._k4)
            self._stream.append((idx, frame, self._k4))
        except Exception as exc:  # noqa: BLE001
            log.warning("droid_slam: track(idx=%d) raised: %s", idx, exc)
            return FrameUpdate()

        # DROID buffers internally and only commits a keyframe when its
        # frontend decides; we don't get an explicit "this was a
        # keyframe" signal per frame. Mark every Nth tracked frame as a
        # keyframe heuristically so the partial-snapshot cadence works.
        # The real keyframe selection lives inside DROID and is reflected
        # in the final terminate() trajectory length.
        is_keyframe = (idx % 4 == 0)
        if is_keyframe:
            self._keyframe_indices.append(idx)
        return FrameUpdate(
            pose_matrix=None,  # DROID doesn't expose per-frame pose pre-BA
            new_points=None,   # cloud is built at finalize()
            is_keyframe=is_keyframe,
            diagnostics={"backend": "droid_slam_cuda"},
        )

    def finalize(self) -> FinalResult:
        if self._tracker is None:
            return FinalResult(
                poses=np.empty((0, 4, 4), dtype=np.float32),
                keyframe_indices=[],
                points=None,
                diagnostics={"backend": "droid_slam_cuda", "empty": True},
            )

        # Run the global BA + extract the final trajectory. The reference
        # demo passes the original stream so DROID can backfill poses
        # between keyframes.
        try:
            stream_iter = iter(self._stream)
            traj = self._tracker.terminate(stream_iter)
        except Exception as exc:  # noqa: BLE001
            log.warning("droid_slam: terminate() raised: %s", exc)
            traj = None

        # Pose extraction. terminate() returns a (N, 7) array of
        # [tx, ty, tz, qx, qy, qz, qw] rows — convert to (N, 4, 4).
        if isinstance(traj, np.ndarray) and traj.ndim == 2 and traj.shape[1] >= 7:
            poses = np.stack([_quat_xyzw_translate_to_matrix(row) for row in traj])
        else:
            poses = np.empty((0, 4, 4), dtype=np.float32)

        # DROID stores the dense map internally; pull it out if available.
        # The exact attribute name shifts between commits — try a few.
        points = self._extract_points()

        return FinalResult(
            poses=poses,
            keyframe_indices=list(self._keyframe_indices),
            points=points,
            diagnostics={
                "backend": "droid_slam_cuda",
                "n_poses": int(poses.shape[0]),
                "n_keyframes": len(self._keyframe_indices),
                "n_points": int(points.shape[0]) if points is not None else 0,
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_points(self) -> Optional[np.ndarray]:
        """Pull the dense reconstructed cloud out of the tracker. The
        upstream API doesn't ship a stable `get_pointcloud()` accessor
        — different forks expose `video.disps_up`, `video.poses`, etc.
        Try the documented path; if it fails return None and the
        SlamProcessor will skip cloud publishing."""
        import torch  # noqa: PLC0415

        try:
            video = self._tracker.video  # type: ignore[attr-defined]
        except AttributeError:
            return None

        try:
            # Reconstruct points from inverse-depth + poses + intrinsics.
            # Each frame contributes a sparse subset of pixels; sample
            # to keep the cloud bounded.
            disps = video.disps_up.detach()  # (N, H, W)
            n_frames = disps.shape[0]
            if n_frames == 0:
                return None
            # Cap to ~200k points total across all frames.
            stride = max(1, int((disps.shape[1] * disps.shape[2] * n_frames) // 200_000) ** 0.5)
            stride = max(1, int(stride))
            pts_list: list[np.ndarray] = []
            for i in range(n_frames):
                d = disps[i, ::stride, ::stride].cpu().numpy()
                if d.size == 0:
                    continue
                # Without intrinsics + extrinsics applied here, this is
                # only a placeholder; v1 returns coordinates in image
                # space scaled by inverse depth — good enough for a
                # partial preview, refinement in a follow-up PR.
                ys, xs = np.indices(d.shape)
                z = 1.0 / np.maximum(d, 1e-3)
                pts = np.stack(
                    [xs.flatten() * z.flatten(), ys.flatten() * z.flatten(), z.flatten()],
                    axis=-1,
                )
                rgb = np.full((pts.shape[0], 3), 200, dtype=np.float32)
                pts_list.append(np.concatenate([pts.astype(np.float32), rgb], axis=1))
            if not pts_list:
                return None
            return np.concatenate(pts_list, axis=0)
        except Exception as exc:  # noqa: BLE001
            log.warning("droid_slam: point extraction raised: %s", exc)
            return None
