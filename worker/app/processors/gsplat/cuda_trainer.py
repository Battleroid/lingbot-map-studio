"""Real CUDA gsplat trainer.

Wraps `gsplat.rasterization.rasterization` for forward render + standard
3DGS training loop. Lives in its own module so the rest of the gsplat
package (the simulated trainer, the processor, the IO helpers) stays
importable on a CPU-only host — `import gsplat` at the top of this file
would crash any container that hadn't pip-installed it.

The trainer is selected at runtime by `select_trainer_cls()` in
`trainer.py`: when `gsplat` imports cleanly we use `GsplatCudaTrainer`;
otherwise we fall back to `SimulatedSplatTrainer`. Phase 0 emits a
loud warn event on the simulated path so the user knows.

Implementation notes (so the next person can debug this on the GPU box
without re-reading the gsplat docs):

- All trainable parameters are torch tensors on cuda. Means/quats live
  in linear space; scales are log-space (apply `exp()` before render);
  opacities are logit-space (apply `sigmoid()` before render). This
  matches the original 3DGS / gsplat reference trainer.
- One Adam optimizer per parameter group with the LRs from the 3DGS
  paper (means scales with scene extent, others fixed).
- Loss: L1 + 0.2 * (1 - SSIM). SSIM is computed inline with a small
  separable Gaussian kernel — saves a torchmetrics dep.
- Densification + pruning is intentionally simple: every
  `densify_interval`, clone the top 5% gaussians by gradient magnitude;
  every `prune_interval`, drop gaussians whose sigmoid(opacity) is
  below `prune_opacity`. The reference trainer's full split/clone
  heuristic is more involved; this is "good enough for v1, adjust
  later".
- The `state` property snapshots the GPU tensors back to numpy in the
  shape `TrainerState` expects so `splat_export.write_splat_ply` works
  unchanged. Snapshotting is on the hot path (called every
  `preview_every_iters`), so we cache the last conversion and only
  rebuild when training has advanced.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np

from app.jobs.schema import GsplatConfig
from app.processors.gsplat import io as splat_io
from app.processors.gsplat.trainer import (
    IterationLog,
    SplatTrainer,
    TrainerState,
)

log = logging.getLogger(__name__)


# Paper-standard learning rates. `means_lr_init` is multiplied by scene
# extent at prepare() time so it scales with reconstruction size.
_MEANS_LR = 1.6e-4
_QUATS_LR = 1.0e-3
_SCALES_LR = 5.0e-3
_OPACITIES_LR = 5.0e-2
_COLORS_LR = 2.5e-3


def _quat_from_xyzw(q_xyzw: list[float]) -> tuple[float, float, float, float]:
    """Pose graphs / camera paths use x,y,z,w order; gsplat wants w,x,y,z."""
    if len(q_xyzw) != 4:
        return (1.0, 0.0, 0.0, 0.0)
    x, y, z, w = q_xyzw
    return (w, x, y, z)


def _viewmat_from_pose(t: list[float], q_xyzw: list[float]) -> np.ndarray:
    """Build a 4x4 world-to-camera matrix from a (translation, quaternion)
    pair stored in the standard pose_graph.json schema."""
    import torch  # noqa: PLC0415 — lazy

    if len(q_xyzw) != 4 or len(t) != 3:
        return np.eye(4, dtype=np.float32)

    x, y, z, w = q_xyzw
    # Quaternion → 3x3 rotation. World-to-cam: poses store cam-from-world
    # already in our schema (`processors/slam/export.py:write_pose_graph`),
    # so no additional inversion is needed.
    norm = (w * w + x * x + y * y + z * z) ** 0.5 or 1.0
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R
    M[:3, 3] = np.asarray(t, dtype=np.float32)
    # Suppress unused warning — torch is imported at top of cuda paths.
    _ = torch
    return M


def _intrinsics_from_camera(
    cam: dict, frame_w: int, frame_h: int
) -> np.ndarray:
    """Per-camera intrinsics as a 3x3. Falls back to a reasonable default
    (60° HFOV) when the source job didn't record intrinsics."""
    intr = cam.get("intrinsics") if isinstance(cam, dict) else None
    if isinstance(intr, dict):
        fx = float(intr.get("fx") or intr.get("f") or frame_w * 0.866)
        fy = float(intr.get("fy") or fx)
        cx = float(intr.get("cx") or frame_w / 2)
        cy = float(intr.get("cy") or frame_h / 2)
    else:
        fx = frame_w * 0.866  # 60° HFOV ≈ tan(30°) inverse
        fy = fx
        cx = frame_w / 2
        cy = frame_h / 2
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _gaussian_kernel(window: int, sigma: float):
    """1D gaussian kernel for SSIM. Lazy torch import."""
    import torch  # noqa: PLC0415

    coords = torch.arange(window, dtype=torch.float32) - window // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    return (g / g.sum()).view(1, 1, 1, -1)


def _ssim(a, b, window: int = 11, sigma: float = 1.5):
    """Single-scale SSIM in [0, 1]. Used for the (1 - SSIM) term in the
    standard 3DGS loss. `a` and `b` are (3, H, W) float tensors in [0, 1]."""
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    if a.dim() == 3:
        a = a.unsqueeze(0)
        b = b.unsqueeze(0)
    c = a.shape[1]
    k1 = _gaussian_kernel(window, sigma).to(a.device, dtype=a.dtype)
    k2 = k1.transpose(-1, -2)
    pad = window // 2

    def _conv(x):
        x = F.conv2d(x, k1.expand(c, 1, 1, window), padding=(0, pad), groups=c)
        x = F.conv2d(x, k2.expand(c, 1, window, 1), padding=(pad, 0), groups=c)
        return x

    mu_a = _conv(a)
    mu_b = _conv(b)
    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b
    sigma_a_sq = _conv(a * a) - mu_a_sq
    sigma_b_sq = _conv(b * b) - mu_b_sq
    sigma_ab = _conv(a * b) - mu_ab
    c1 = 0.01**2
    c2 = 0.03**2
    num = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    den = (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)
    return (num / den).mean()


class GsplatCudaTrainer(SplatTrainer):
    """Real GPU-backed gsplat trainer. Selected automatically when the
    `gsplat` package imports cleanly (see `select_trainer_cls`).

    Holds means / quats / scales / opacities / colors as torch.cuda
    nn.Parameters; per-iteration: pick a random keyframe, render, compute
    L1 + (1-SSIM) loss, backward, optimizer step, periodic densify/prune.
    """

    def __init__(self) -> None:
        self._cfg: Optional[GsplatConfig] = None
        # All tensors below allocated in prepare(). torch is lazy-imported
        # to keep this module import-safe on a CPU-only host.
        self._means = None
        self._quats = None
        self._scales = None
        self._opacities = None
        self._colors = None
        self._optimizers: list = []
        # Camera + frame data — loaded from disk in prepare().
        self._viewmats = None  # (B, 4, 4)
        self._Ks = None  # (B, 3, 3)
        self._frames: dict[int, "object"] = {}  # cam_idx → cached HxWx3 tensor
        self._frame_paths: list[Path] = []
        self._cam_to_frame: list[int] = []  # cam_idx → frame_paths index
        self._H = 0
        self._W = 0
        self._cached_state: Optional[TrainerState] = None
        self._dirty = True
        self._device = None
        self._scene_extent = 1.0

    # ------------------------------------------------------------------
    # SplatTrainer surface
    # ------------------------------------------------------------------

    @property
    def state(self) -> TrainerState:
        if self._cached_state is None or self._dirty:
            self._cached_state = self._snapshot()
            self._dirty = False
        return self._cached_state

    def prepare(self, inputs: splat_io.GsplatInputs, cfg: GsplatConfig) -> None:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            raise RuntimeError(
                "GsplatCudaTrainer requires CUDA. "
                "Either run on a GPU host or fall back to SimulatedSplatTrainer."
            )
        self._cfg = cfg
        self._device = torch.device("cuda")

        # 1. Init points → means + colors. Random fallback if the source
        #    job didn't emit a cloud.
        if cfg.init_from == "point_cloud" and inputs.init_points is not None:
            pts = splat_io.load_init_points(inputs.init_points)
            means_np = pts[:, :3].astype(np.float32)
            colors_np = np.clip(pts[:, 3:6] / 255.0, 0.0, 1.0).astype(np.float32)
        else:
            n = cfg.random_init_count
            rng = np.random.default_rng(42)
            u = rng.normal(size=(n, 3)).astype(np.float32)
            u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-9
            means_np = (u * rng.uniform(0.5, 2.0, size=(n, 1))).astype(np.float32)
            colors_np = rng.uniform(0, 1, size=(n, 3)).astype(np.float32)

        n = means_np.shape[0]
        # Scene extent ≈ diagonal of the init-point bounding box; scales
        # the means LR so a 100-unit scene doesn't move at the same step
        # size as a 1-unit scene.
        bbox = means_np.max(axis=0) - means_np.min(axis=0)
        self._scene_extent = float(np.linalg.norm(bbox)) or 1.0

        means = torch.tensor(means_np, device=self._device, requires_grad=True)
        colors = torch.tensor(colors_np, device=self._device, requires_grad=True)
        # Initial scale: a small fraction of the median nearest-neighbor
        # distance is a common heuristic; use a fixed log-space default
        # for simplicity (matches what SimulatedSplatTrainer uses).
        scales = torch.full(
            (n, 3),
            math.log(0.02 * self._scene_extent),
            device=self._device,
            requires_grad=True,
        )
        # Identity quaternion (w=1, x=y=z=0)
        quats = torch.zeros((n, 4), device=self._device, requires_grad=True)
        with torch.no_grad():
            quats[:, 0] = 1.0
        # Logit(0.3) so initial sigmoid → 0.3 — same default as simulated.
        opacities = torch.full(
            (n,),
            float(np.log(0.3 / 0.7)),
            device=self._device,
            requires_grad=True,
        )

        self._means = torch.nn.Parameter(means)
        self._quats = torch.nn.Parameter(quats)
        self._scales = torch.nn.Parameter(scales)
        self._opacities = torch.nn.Parameter(opacities)
        self._colors = torch.nn.Parameter(colors)
        self._build_optimizers()

        # 2. Cameras + frames.
        if inputs.cameras_path is None:
            raise RuntimeError(
                "GsplatCudaTrainer needs a pose_graph.json or camera_path.json "
                "from the source job; none found."
            )
        cameras = splat_io.load_cameras(inputs.cameras_path)
        # Both extensions: .png from ffmpeg-extracted ingest, .jpg
        # from the live-capture path that streams phone frames
        # straight to disk. The two never coexist in one job dir.
        self._frame_paths = sorted(inputs.frames_dir.glob("*.png")) + sorted(
            inputs.frames_dir.glob("*.jpg")
        )
        if not self._frame_paths:
            raise RuntimeError(
                f"GsplatCudaTrainer: no frames in {inputs.frames_dir}"
            )

        # Probe the first frame for H, W (assume all frames are the same size).
        sample_h, sample_w = self._probe_frame_shape(self._frame_paths[0])
        self._H, self._W = sample_h, sample_w

        viewmats = []
        Ks = []
        cam_to_frame = []
        for i, cam in enumerate(cameras):
            t = cam.get("t", [0.0, 0.0, 0.0])
            q = cam.get("q", [0.0, 0.0, 0.0, 1.0])
            viewmats.append(_viewmat_from_pose(t, q))
            Ks.append(_intrinsics_from_camera(cam, sample_w, sample_h))
            # `source_frame` indexes the original clip; if missing fall back
            # to the camera index (common for camera_path.json files).
            src = cam.get("source_frame", i)
            cam_to_frame.append(min(int(src), len(self._frame_paths) - 1))

        self._viewmats = torch.tensor(np.stack(viewmats), device=self._device)
        self._Ks = torch.tensor(np.stack(Ks), device=self._device)
        self._cam_to_frame = cam_to_frame
        self._dirty = True

    def step(self, iter_idx: int) -> IterationLog:
        import torch  # noqa: PLC0415
        from gsplat.rasterization import rasterization  # noqa: PLC0415

        assert self._cfg is not None and self._device is not None

        # Pick a random keyframe for this iteration.
        cam_idx = int(np.random.randint(0, self._viewmats.shape[0]))
        viewmat = self._viewmats[cam_idx : cam_idx + 1]  # (1, 4, 4)
        K = self._Ks[cam_idx : cam_idx + 1]  # (1, 3, 3)
        gt = self._load_frame(self._cam_to_frame[cam_idx])  # (3, H, W)

        # Forward render. gsplat consumes scales in linear space and
        # opacities in [0, 1] — apply exp / sigmoid here so the trainable
        # parameters stay in their natural log / logit space.
        renders, _alphas, _info = rasterization(
            means=self._means,
            quats=self._quats,
            scales=torch.exp(self._scales),
            opacities=torch.sigmoid(self._opacities),
            colors=self._colors,
            viewmats=viewmat,
            Ks=K,
            width=self._W,
            height=self._H,
            sh_degree=None,  # raw RGB; SH is a follow-up
            near_plane=0.01,
            far_plane=1.0e10,
        )
        rendered = renders[0].permute(2, 0, 1).clamp(0.0, 1.0)  # (3, H, W)

        # Standard 3DGS loss: 0.8 * L1 + 0.2 * (1 - SSIM).
        l1 = torch.abs(rendered - gt).mean()
        ssim = _ssim(rendered, gt)
        loss = 0.8 * l1 + 0.2 * (1.0 - ssim)

        for opt in self._optimizers:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        for opt in self._optimizers:
            opt.step()

        # PSNR for the log: 10 * log10(1 / mse) in [0, 1] range.
        with torch.no_grad():
            mse = torch.mean((rendered - gt) ** 2).item()
            psnr = 10.0 * math.log10(1.0 / max(mse, 1e-10))

        cfg = self._cfg
        if iter_idx > 0 and iter_idx % cfg.densify_interval == 0:
            self._densify()
        if iter_idx > 0 and iter_idx % cfg.prune_interval == 0:
            self._prune(cfg.prune_opacity)

        self._dirty = True
        return IterationLog(
            iter=iter_idx,
            n_gaussians=int(self._means.shape[0]),
            psnr=psnr,
            loss=float(loss.item()),
        )

    # ------------------------------------------------------------------
    # Densification + pruning (simple v1 heuristics)
    # ------------------------------------------------------------------

    def _densify(self) -> None:
        """Clone the top 5% of gaussians by means-gradient magnitude.
        Real 3DGS additionally splits very-large gaussians; v1 keeps just
        the clone path until we have a reproducer for the split case."""
        import torch  # noqa: PLC0415

        if self._means.grad is None:
            return
        with torch.no_grad():
            grad_mag = self._means.grad.norm(dim=-1)
            n = grad_mag.shape[0]
            k = max(1, n // 20)
            _, top_idx = torch.topk(grad_mag, k=k)
            self._clone_indices(top_idx)

    def _clone_indices(self, idx) -> None:
        import torch  # noqa: PLC0415

        with torch.no_grad():
            new_means = self._means[idx] + 0.001 * torch.randn_like(self._means[idx])
            new_quats = self._quats[idx].clone()
            new_scales = self._scales[idx].clone()
            new_opacities = self._opacities[idx].clone()
            new_colors = self._colors[idx].clone()
            self._replace_params(
                torch.cat([self._means, new_means]),
                torch.cat([self._quats, new_quats]),
                torch.cat([self._scales, new_scales]),
                torch.cat([self._opacities, new_opacities]),
                torch.cat([self._colors, new_colors]),
            )

    def _prune(self, prune_opacity: float) -> None:
        import torch  # noqa: PLC0415

        with torch.no_grad():
            keep = torch.sigmoid(self._opacities) > prune_opacity
            if keep.all():
                return
            self._replace_params(
                self._means[keep],
                self._quats[keep],
                self._scales[keep],
                self._opacities[keep],
                self._colors[keep],
            )

    def _replace_params(self, means, quats, scales, opacities, colors) -> None:
        """Rebuild the nn.Parameters + optimizers around new tensors.
        Adam state is reset for densified rows — acceptable for v1; the
        reference trainer carries momentum forward but it adds bookkeeping."""
        import torch  # noqa: PLC0415

        self._means = torch.nn.Parameter(means.detach().requires_grad_(True))
        self._quats = torch.nn.Parameter(quats.detach().requires_grad_(True))
        self._scales = torch.nn.Parameter(scales.detach().requires_grad_(True))
        self._opacities = torch.nn.Parameter(
            opacities.detach().requires_grad_(True)
        )
        self._colors = torch.nn.Parameter(colors.detach().requires_grad_(True))
        self._build_optimizers()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_optimizers(self) -> None:
        import torch  # noqa: PLC0415

        means_lr = _MEANS_LR * self._scene_extent
        self._optimizers = [
            torch.optim.Adam([self._means], lr=means_lr),
            torch.optim.Adam([self._quats], lr=_QUATS_LR),
            torch.optim.Adam([self._scales], lr=_SCALES_LR),
            torch.optim.Adam([self._opacities], lr=_OPACITIES_LR),
            torch.optim.Adam([self._colors], lr=_COLORS_LR),
        ]

    def _probe_frame_shape(self, path: Path) -> tuple[int, int]:
        import cv2  # noqa: PLC0415

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"failed to read frame {path}")
        h, w = img.shape[:2]
        return h, w

    def _load_frame(self, frame_idx: int):
        """Cached PNG → (3, H, W) GPU tensor in [0, 1]. Frames stay on
        CPU until first hit, then cached on GPU. Cap memory by reading
        directly into the device."""
        import cv2  # noqa: PLC0415
        import torch  # noqa: PLC0415

        if frame_idx in self._frames:
            return self._frames[frame_idx]
        path = self._frame_paths[frame_idx]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(self._device)
        self._frames[frame_idx] = tensor
        return tensor

    def _snapshot(self) -> TrainerState:
        """Convert GPU tensors back to numpy in the shape `TrainerState`
        expects. Called every preview interval, so cheap matters; we
        `.detach().cpu().numpy()` once and cache until step() dirties it."""
        return TrainerState(
            means=self._means.detach().cpu().numpy().astype(np.float64),
            colors=self._colors.detach().cpu().numpy().astype(np.float64),
            opacities=self._opacities.detach().cpu().numpy().astype(np.float64),
            scales=self._scales.detach().cpu().numpy().astype(np.float64),
            rotations=self._quats.detach().cpu().numpy().astype(np.float64),
        )
