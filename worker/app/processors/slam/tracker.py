"""Simulated SLAM tracker.

A dependency-light stand-in for the real DROID/MASt3R/DPVO/MonoGS backends.
Uses OpenCV Farneback optical flow to estimate a smooth camera trajectory
plus a sparse "point cloud" built from triangulated flow features. It's not
a real SLAM — there's no bundle adjustment, no loop closure, and the scale
is arbitrary — but it produces artifacts in the exact same shape the real
backends will, so every code path downstream (exports, live preview, tool
panels) can be developed and tested end-to-end without the heavy CUDA
dependencies being installed first.

Each concrete backend processor instantiates this tracker with its own
`backend_id` so diagnostics and pose graphs correctly attribute results.
When an upstream backend lands it replaces the session class only —
`SlamProcessor.run` stays exactly as-is.

Design notes:

  * Motion model is pure-forward translation (+ small rotation) derived
    from mean flow direction. This looks plausible on FPV footage without
    claiming to be accurate.
  * "Points" are good-to-track corners lifted to 3D via an assumed depth
    proportional to (1 / flow_magnitude). Farther = slower-moving.
  * The session is frame-index driven (not time-driven) so deterministic
    replay is possible — important for our unit tests.
"""

from __future__ import annotations

import logging
from typing import ClassVar, Optional

import numpy as np

from app.processors.slam.base import FinalResult, FrameUpdate, SlamSession

log = logging.getLogger(__name__)


class SimulatedSlamSession(SlamSession):
    """Lightweight SLAM placeholder. Does just enough to produce a
    trajectory + sparse cloud so the pipeline is exercisable end-to-end.

    Subclassed per-backend to set `backend_id` / `trajectory_only` and
    tune the motion/depth heuristics, but the default behaviour is
    already usable as-is.
    """

    backend_id: ClassVar[str] = "simulated"
    # Points are cheap and look fine, so we keep them on by default.
    trajectory_only: ClassVar[bool] = False

    # How many good-to-track corners to sample per keyframe. Small on
    # purpose — the cloud is just a preview, not a real map.
    corners_per_keyframe: ClassVar[int] = 128
    # Stride between keyframes. The processor's own keyframe policy
    # already culls frames; this adds one more factor on top so very-long
    # clips don't blow up the cloud.
    keyframe_every: ClassVar[int] = 1
    # Noise on the synthetic depths — higher = bumpier cloud, lower =
    # flatter carpets. 0.2 works well for FPV footage.
    depth_noise: ClassVar[float] = 0.2

    def __init__(self) -> None:
        self._intrinsics: Optional[np.ndarray] = None
        self._image_shape: Optional[tuple[int, int]] = None
        self._prev_gray: Optional[np.ndarray] = None
        self._pose = np.eye(4, dtype=np.float64)
        self._poses: list[np.ndarray] = []
        self._keyframe_indices: list[int] = []
        self._points: list[np.ndarray] = []
        self._step_count = 0

    # ------------------------------------------------------------------
    # SlamSession API
    # ------------------------------------------------------------------

    def start(
        self,
        intrinsics: np.ndarray,
        image_shape: tuple[int, int],
    ) -> None:
        self._intrinsics = np.asarray(intrinsics, dtype=np.float64)
        self._image_shape = image_shape
        # Seed pose is identity at the origin — the first keyframe below
        # captures this so the trajectory always starts at 0.

    def step(self, idx: int, img: np.ndarray) -> FrameUpdate:
        import cv2

        if img is None or img.size == 0:
            return FrameUpdate(is_keyframe=False)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        update = FrameUpdate()

        if self._prev_gray is None:
            # First frame: commit identity pose as seed keyframe.
            self._prev_gray = gray
            self._step_count = 0
            self._record_keyframe(idx, img, gray, flow=None)
            update.pose_matrix = self._pose.copy()
            update.new_points = self._latest_points()
            update.is_keyframe = True
            update.diagnostics["seed"] = True
            return update

        self._step_count += 1
        try:
            flow = cv2.calcOpticalFlowFarneback(
                self._prev_gray,
                gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=2,
                poly_n=5,
                poly_sigma=1.1,
                flags=0,
            )
        except cv2.error as exc:
            log.debug("simulated tracker: flow failed (%s)", exc)
            flow = None

        self._advance_pose(flow)
        self._prev_gray = gray

        is_kf = (self._step_count % max(1, self.keyframe_every)) == 0
        if is_kf:
            self._record_keyframe(idx, img, gray, flow=flow)
            update.pose_matrix = self._pose.copy()
            update.new_points = self._latest_points()
            update.is_keyframe = True
            update.diagnostics["flow_mean"] = _safe_mean_mag(flow)
        return update

    def finalize(self) -> FinalResult:
        if not self._poses:
            return FinalResult(
                poses=np.zeros((0, 4, 4), dtype=np.float64),
                keyframe_indices=[],
                points=None,
                diagnostics={"backend": self.backend_id, "simulated": True},
            )
        poses = np.stack(self._poses, axis=0)
        points = (
            np.concatenate(self._points, axis=0) if self._points else None
        )
        diag: dict[str, object] = {
            "backend": self.backend_id,
            "simulated": True,
        }
        if self._image_shape is not None:
            diag["image_height"] = self._image_shape[0]
            diag["image_width"] = self._image_shape[1]
        return FinalResult(
            poses=poses,
            keyframe_indices=list(range(len(self._poses))),
            points=points if not self.trajectory_only else None,
            diagnostics=diag,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _advance_pose(self, flow: Optional[np.ndarray]) -> None:
        """Integrate a small motion step from mean optical flow.

        Translation is in "image-plane" units normalised by focal length so
        different-sized inputs produce comparable trajectory extents.
        Rotation is derived from the curl of the flow field (cheap proxy
        for yaw) — good enough for a placeholder.
        """
        if flow is None or flow.size == 0 or self._intrinsics is None:
            return
        fx = self._intrinsics[0, 0]
        fy = self._intrinsics[1, 1]
        if fx <= 0 or fy <= 0:
            return

        mean_u = float(np.nanmean(flow[..., 0])) if flow.size else 0.0
        mean_v = float(np.nanmean(flow[..., 1])) if flow.size else 0.0
        # Translation: invert flow direction (scene moves opposite to camera).
        tx = -mean_u / fx
        ty = -mean_v / fy
        # Forward motion heuristic: divergence of the flow field. If flow
        # spreads outward from the centre the camera is advancing; if it
        # converges the camera is retreating.
        tz = _flow_divergence(flow)

        # Yaw proxy from flow curl.
        yaw = _flow_curl(flow) * 0.5
        R = np.array(
            [
                [np.cos(yaw), 0.0, np.sin(yaw)],
                [0.0, 1.0, 0.0],
                [-np.sin(yaw), 0.0, np.cos(yaw)],
            ],
            dtype=np.float64,
        )
        step = np.eye(4, dtype=np.float64)
        step[:3, :3] = R
        step[:3, 3] = [tx, ty, tz]
        self._pose = self._pose @ step

    def _record_keyframe(
        self,
        idx: int,
        img: np.ndarray,
        gray: np.ndarray,
        *,
        flow: Optional[np.ndarray],
    ) -> None:
        self._poses.append(self._pose.copy())
        self._keyframe_indices.append(idx)
        if self.trajectory_only:
            return
        pts = self._build_points(img, gray, flow)
        if pts is not None and pts.size > 0:
            self._points.append(pts)

    def _latest_points(self) -> Optional[np.ndarray]:
        if not self._points:
            return None
        return self._points[-1]

    def _build_points(
        self,
        img: np.ndarray,
        gray: np.ndarray,
        flow: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        import cv2

        if self._intrinsics is None or self._image_shape is None:
            return None
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.corners_per_keyframe,
            qualityLevel=0.01,
            minDistance=8,
        )
        if corners is None or len(corners) == 0:
            return None
        corners = corners.reshape(-1, 2)

        fx = self._intrinsics[0, 0]
        fy = self._intrinsics[1, 1]
        cx = self._intrinsics[0, 2]
        cy = self._intrinsics[1, 2]

        # Depth: 1/flow_mag with a floor, plus noise. Points near motion
        # appear closer, stationary pixels appear far. Lots of hand-waving
        # here — it's a preview, not a mapper.
        if flow is not None:
            mag = np.linalg.norm(flow, axis=-1)
            depths_raw = 1.0 / np.maximum(mag, 0.5)
            h, w = gray.shape[:2]
            ys = np.clip(corners[:, 1].astype(int), 0, h - 1)
            xs = np.clip(corners[:, 0].astype(int), 0, w - 1)
            depths = depths_raw[ys, xs]
        else:
            depths = np.full(corners.shape[0], 1.0, dtype=np.float64)
        depths = depths * (1.0 + self.depth_noise * np.random.randn(depths.size))
        depths = np.clip(depths, 0.1, 50.0)

        xs = (corners[:, 0] - cx) * depths / fx
        ys = (corners[:, 1] - cy) * depths / fy
        zs = depths
        cam_pts = np.stack([xs, ys, zs, np.ones_like(zs)], axis=-1)  # (N,4)
        world_pts = (self._pose @ cam_pts.T).T[:, :3]

        # Sample colours from the original image at the corner locations.
        h, w = img.shape[:2]
        cys = np.clip(corners[:, 1].astype(int), 0, h - 1)
        cxs = np.clip(corners[:, 0].astype(int), 0, w - 1)
        bgr = img[cys, cxs]
        rgb = bgr[:, ::-1].astype(np.float64)  # BGR → RGB for downstream

        return np.concatenate([world_pts, rgb], axis=1)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _safe_mean_mag(flow: Optional[np.ndarray]) -> float:
    if flow is None or flow.size == 0:
        return 0.0
    return float(np.nanmean(np.linalg.norm(flow, axis=-1)))


def _flow_divergence(flow: np.ndarray) -> float:
    """Approximate flow divergence via finite differences. Positive = scene
    expanding away from optical centre (forward motion)."""
    u = flow[..., 0]
    v = flow[..., 1]
    du_dx = np.diff(u, axis=1)
    dv_dy = np.diff(v, axis=0)
    # Clip arrays to the same shape before summing.
    m = min(du_dx.shape[0], dv_dy.shape[0])
    n = min(du_dx.shape[1], dv_dy.shape[1])
    div = du_dx[:m, :n] + dv_dy[:m, :n]
    return float(np.nanmean(div)) * 0.01  # scale down — tiny per-step motion


def _flow_curl(flow: np.ndarray) -> float:
    """Approximate flow curl via finite differences. Proxy for camera yaw."""
    u = flow[..., 0]
    v = flow[..., 1]
    dv_dx = np.diff(v, axis=1)
    du_dy = np.diff(u, axis=0)
    m = min(dv_dx.shape[0], du_dy.shape[0])
    n = min(dv_dx.shape[1], du_dy.shape[1])
    curl = dv_dx[:m, :n] - du_dy[:m, :n]
    return float(np.nanmean(curl)) * 0.002
