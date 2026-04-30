"""Batch MonoGS wrapper.

Upstream `muskie82/MonoGS` doesn't expose a streaming `process_frame` API
— its only entrypoint is `slam.SLAM(config)`, a research-grade
multi-process driver that reads frames from a Dataset on disk and runs
frontend tracking + backend Gaussian mapping until the dataset is
exhausted. We bridge that mismatch here:

  1. Build a TUM-shaped on-disk workspace from `frames_dir/`. Upstream
     ships dataset adapters for TUM / Replica / EuRoC / RealSense; TUM
     is the simplest one to fake (filename lists + identity poses).
  2. Generate a YAML config that points `Dataset.dataset_path` at the
     workspace and overrides Calibration with the capture's intrinsics.
  3. Subprocess-run `python /opt/monogs/slam.py --config <yaml>`,
     streaming stdout line-by-line into the job event log so the user
     sees real progress.
  4. Read back the splat PLY from `<save_dir>/point_cloud/final/
     point_cloud.ply` and the trajectory from `<save_dir>/traj.txt`
     (when MonoGS writes one).

This module is import-safe on a CPU-only host: the upstream module
imports + the subprocess invocation only happen inside the public
function, after the caller has confirmed it's running on the worker-gs
container with CUDA + MonoGS installed."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import numpy as np
import yaml

from app.jobs.schema import JobEvent

log = logging.getLogger(__name__)


# Path to upstream MonoGS source tree inside the container. Set by both
# Dockerfile.gs and Dockerfile.api via PYTHONPATH; we also need the
# directory itself for the `slam.py` invocation. Override via env for
# tests that pin a fixture.
MONOGS_ROOT = Path(os.environ.get("MONOGS_ROOT", "/opt/monogs"))


class MonogsBatchUnavailableError(RuntimeError):
    """Raised when the batch wrapper can't find upstream MonoGS or
    can't construct a working config. Bubbled up by the processor and
    surfaced to the user as a level=error event."""


@dataclass
class MonogsBatchResult:
    splat_ply: Path
    trajectory: Optional[np.ndarray]      # (N, 4, 4) cam-from-world, may be None
    keyframe_indices: list[int]
    save_dir: Path                        # MonoGS workspace; kept for diagnostics


# ----------------------------------------------------------------------
# TUM workspace builder
# ----------------------------------------------------------------------

# The TUMParser auto-skips lines starting with `#`, so a header is
# helpful for users inspecting the workspace by hand but optional for
# the parser itself.

_RGB_HEADER = (
    "# color images\n"
    "# file: 'lingbot capture'\n"
    "# timestamp filename\n"
)
_DEPTH_HEADER = (
    "# depth maps (placeholder — monocular config, never read)\n"
    "# file: 'lingbot capture'\n"
    "# timestamp filename\n"
)
_GT_HEADER = (
    "# ground truth trajectory (identity placeholder for monocular SLAM)\n"
    "# file: 'lingbot capture'\n"
    "# timestamp tx ty tz qx qy qz qw\n"
)


def _build_tum_workspace(
    workspace: Path,
    frame_paths: list[Path],
    fps: float = 10.0,
) -> int:
    """Materialise a minimal TUM-shaped dataset in `workspace/` driven
    by `frame_paths`. Returns the number of frames installed.

    Layout produced::

        <workspace>/
            rgb/<idx:06d>.png       (symlinks where possible, copy fallback)
            depth/<idx:06d>.png     (single 1×1 placeholder; never read)
            rgb.txt
            depth.txt
            groundtruth.txt

    The TUMParser associates frames across rgb / depth / pose by
    closest timestamp (max_dt = 0.08 s). We use the same monotonic
    timestamps for all three lists so every entry is matched 1:1.
    """
    rgb_dir = workspace / "rgb"
    depth_dir = workspace / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    # Single placeholder depth file shared by every frame entry.
    # Monocular mode never reads it — the parser only stores the path.
    placeholder = depth_dir / "placeholder.png"
    if not placeholder.exists():
        # Smallest valid PNG (1×1 black). The dataset never reads this
        # in monocular mode but the path needs to exist as a string.
        _write_min_png(placeholder)

    rgb_lines: list[str] = [_RGB_HEADER]
    depth_lines: list[str] = [_DEPTH_HEADER]
    gt_lines: list[str] = [_GT_HEADER]

    n_installed = 0
    dt = 1.0 / max(fps, 1.0)
    for i, src in enumerate(frame_paths):
        if not src.exists():
            log.warning("monogs_batch: missing frame %s, skipping", src)
            continue
        rel_rgb = f"rgb/{i:06d}{src.suffix.lower() or '.png'}"
        rel_depth = "depth/placeholder.png"
        dst = workspace / rel_rgb
        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            try:
                os.symlink(src.resolve(), dst)
            except OSError:
                # Filesystem doesn't support symlinks (some volume mounts).
                shutil.copy2(src, dst)
        except Exception as exc:  # noqa: BLE001
            log.warning("monogs_batch: copy %s -> %s failed: %s", src, dst, exc)
            continue

        ts = i * dt
        rgb_lines.append(f"{ts:.6f} {rel_rgb}\n")
        depth_lines.append(f"{ts:.6f} {rel_depth}\n")
        # Identity pose: tx ty tz = 0; qx qy qz qw = 0 0 0 1.
        gt_lines.append(f"{ts:.6f} 0 0 0 0 0 0 1\n")
        n_installed += 1

    (workspace / "rgb.txt").write_text("".join(rgb_lines), encoding="utf-8")
    (workspace / "depth.txt").write_text("".join(depth_lines), encoding="utf-8")
    (workspace / "groundtruth.txt").write_text("".join(gt_lines), encoding="utf-8")
    return n_installed


def _write_min_png(path: Path) -> None:
    """1×1 grayscale PNG (8 bytes signature + IHDR + IDAT + IEND)."""
    # Pre-baked minimal black PNG. Bytewise built so we don't pull
    # PIL/cv2 just for this; the file content is fully deterministic.
    blob = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x00\x00\x00\x00"
        b"\x3a\x7e\x9b\x55"
        b"\x00\x00\x00\nIDAT"
        b"\x78\x9c\x62\x00\x00\x00\x00\x05\x00\x01"
        b"\x0d\x0a\x2d\xb4"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    path.write_bytes(blob)


# ----------------------------------------------------------------------
# Config builder
# ----------------------------------------------------------------------


def _build_config(
    intrinsics: np.ndarray,
    image_shape: tuple[int, int],
    dataset_path: Path,
    save_dir: Path,
) -> dict[str, Any]:
    """Build the MonoGS YAML config dict for a monocular TUM-shaped run.

    Mirrors the `inherit_from: configs/mono/tum/base_config.yaml`
    layout but inlines every key so we don't need the upstream config
    files on the path. Calibration is taken from the streaming
    intrinsics + the first frame's shape.
    """
    h, w = image_shape
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])

    return {
        "Results": {
            "save_results": True,
            "save_dir": str(save_dir),
            "save_trj": True,
            "save_trj_kf_intv": 10,
            "use_gui": False,
            "eval_rendering": False,
            "use_wandb": False,
        },
        "Dataset": {
            "type": "tum",
            "sensor_type": "monocular",
            "dataset_path": str(dataset_path),
            "pcd_downsample": 64,
            "pcd_downsample_init": 32,
            "adaptive_pointsize": True,
            "point_size": 0.01,
            "Calibration": {
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "k1": 0.0,
                "k2": 0.0,
                "p1": 0.0,
                "p2": 0.0,
                "k3": 0.0,
                "width": int(w),
                "height": int(h),
                "distorted": False,
                # Note: no `depth_scale` — that flips MonocularDataset
                # to has_depth=False and the runtime never reads our
                # placeholder depth files.
            },
        },
        "Training": {
            "init_itr_num": 1050,
            "init_gaussian_update": 100,
            "init_gaussian_reset": 500,
            "init_gaussian_th": 0.005,
            "init_gaussian_extent": 30,
            "tracking_itr_num": 100,
            "mapping_itr_num": 150,
            "gaussian_update_every": 150,
            "gaussian_update_offset": 50,
            "gaussian_th": 0.7,
            "gaussian_extent": 1.0,
            "gaussian_reset": 2001,
            "size_threshold": 20,
            "kf_interval": 5,
            "window_size": 8,
            "pose_window": 3,
            "edge_threshold": 1.1,
            "rgb_boundary_threshold": 0.01,
            "kf_translation": 0.08,
            "kf_min_translation": 0.05,
            "kf_overlap": 0.9,
            "kf_cutoff": 0.3,
            "prune_mode": "slam",
            "single_thread": False,
            "spherical_harmonics": False,
            "lr": {
                "cam_rot_delta": 0.003,
                "cam_trans_delta": 0.001,
            },
        },
        "opt_params": {
            "iterations": 30000,
            "position_lr_init": 0.0016,
            "position_lr_final": 0.0000016,
            "position_lr_delay_mult": 0.01,
            "position_lr_max_steps": 30000,
            "feature_lr": 0.0025,
            "opacity_lr": 0.05,
            "scaling_lr": 0.001,
            "rotation_lr": 0.001,
            "percent_dense": 0.01,
            "lambda_dssim": 0.2,
            "densification_interval": 100,
            "opacity_reset_interval": 3000,
            "densify_from_iter": 500,
            "densify_until_iter": 15000,
            "densify_grad_threshold": 0.0002,
        },
        "model_params": {
            "sh_degree": 0,
            "source_path": str(dataset_path),
            "model_path": "",
            "resolution": -1,
            "white_background": False,
            "data_device": "cuda",
        },
        "pipeline_params": {
            "convert_SHs_python": False,
            "compute_cov3D_python": False,
        },
    }


# ----------------------------------------------------------------------
# Subprocess driver
# ----------------------------------------------------------------------


PublishFn = Callable[[JobEvent], Awaitable[Any]]


async def run_monogs_batch(
    *,
    job_id: str,
    frame_paths: list[Path],
    intrinsics: np.ndarray,
    image_shape: tuple[int, int],
    workspace_root: Path,
    publish: PublishFn,
    fps: float = 10.0,
    monogs_root: Path = MONOGS_ROOT,
) -> MonogsBatchResult:
    """Run upstream MonoGS as a subprocess on `frame_paths`.

    Args:
        job_id: surfaced in event payloads.
        frame_paths: ordered list of RGB frames on disk.
        intrinsics: 3×3 camera matrix from the SlamProcessor.
        image_shape: (h, w) of the frames.
        workspace_root: scratch directory; this function writes
            `<root>/dataset/` and `<root>/output/` underneath it. Caller
            owns lifecycle (typically `<artifacts_dir>/.monogs_workspace`).
        publish: async event publisher; receives stdout lines + status.
        fps: capture frame rate, used to space the synthesised TUM
            timestamps.
        monogs_root: path to the upstream source clone.

    Raises:
        MonogsBatchUnavailableError: if the source tree is missing or
            the subprocess fails. The processor catches this and
            surfaces it as a level=error event with install hints.
    """
    if not (monogs_root / "slam.py").exists():
        raise MonogsBatchUnavailableError(
            f"monogs_batch: upstream slam.py not found at {monogs_root}/slam.py. "
            "The Dockerfile should `git clone --recursive https://github.com/"
            "muskie82/MonoGS.git /opt/monogs` and add it to PYTHONPATH; see "
            "worker/Dockerfile.gs."
        )
    if not frame_paths:
        raise MonogsBatchUnavailableError(
            "monogs_batch: no frames to reconstruct."
        )

    workspace_root.mkdir(parents=True, exist_ok=True)
    dataset_dir = workspace_root / "dataset"
    save_dir = workspace_root / "output"
    config_path = workspace_root / "config.yaml"
    save_dir.mkdir(parents=True, exist_ok=True)

    n_installed = await asyncio.to_thread(
        _build_tum_workspace, dataset_dir, frame_paths, fps
    )
    if n_installed < 5:
        raise MonogsBatchUnavailableError(
            f"monogs_batch: too few frames to reconstruct ({n_installed}). "
            "MonoGS needs at least a handful of overlapping views to "
            "initialise its monocular tracker."
        )
    config = _build_config(intrinsics, image_shape, dataset_dir, save_dir)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    await publish(
        JobEvent(
            job_id=job_id,
            stage="slam",
            message=(
                f"monogs: starting batch reconstruction "
                f"(frames={n_installed}, image={image_shape[1]}×{image_shape[0]})"
            ),
            data={
                "frames": n_installed,
                "config": str(config_path),
                "save_dir": str(save_dir),
            },
        )
    )

    env = os.environ.copy()
    py_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{monogs_root}{os.pathsep}{py_path}" if py_path else str(monogs_root)
    )
    # Force unbuffered output so progress lines flush in real time.
    env.setdefault("PYTHONUNBUFFERED", "1")

    cmd = [
        "python",
        "-u",
        str(monogs_root / "slam.py"),
        "--config",
        str(config_path),
    ]
    log.info("monogs_batch: launching %s", " ".join(cmd))

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(monogs_root),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    assert proc.stdout is not None
    line_count = 0
    async for raw in proc.stdout:
        line_count += 1
        try:
            line = raw.decode("utf-8", errors="replace").rstrip()
        except Exception:  # noqa: BLE001
            continue
        if not line:
            continue
        # Map upstream's `Log("…", tag="…")` lines to JobEvents. Most
        # lines are informational; surface them at info level. Errors
        # and warnings get bumped to warn so they show up in the log
        # pane without flooding it.
        level = "info"
        lower = line.lower()
        if "error" in lower or "traceback" in lower:
            level = "error"
        elif "warn" in lower:
            level = "warn"
        await publish(
            JobEvent(
                job_id=job_id,
                stage="slam",
                level=level,
                message=line[:512],
                data={"source": "monogs.stdout"},
            )
        )

    rc = await proc.wait()
    if rc != 0:
        raise MonogsBatchUnavailableError(
            f"monogs_batch: slam.py exited with code {rc} "
            f"after {line_count} stdout line(s). Check the job log above; "
            "common causes are missing CUDA / out-of-memory / corrupt frames."
        )

    splat_ply = _locate_splat_ply(save_dir)
    if splat_ply is None:
        raise MonogsBatchUnavailableError(
            f"monogs_batch: slam.py exited cleanly but no point_cloud.ply "
            f"was written under {save_dir}. Check the job log for upstream "
            "errors that may have been suppressed."
        )

    trajectory = _load_trajectory(save_dir)
    keyframe_indices = list(range(min(n_installed, trajectory.shape[0]) if trajectory is not None else n_installed))

    return MonogsBatchResult(
        splat_ply=splat_ply,
        trajectory=trajectory,
        keyframe_indices=keyframe_indices,
        save_dir=save_dir,
    )


def _locate_splat_ply(save_dir: Path) -> Optional[Path]:
    """Find MonoGS's final splat PLY. Upstream writes it to
    `<save_dir>/point_cloud/final/point_cloud.ply` via
    `save_gaussians(..., final=True)`. Falls back to whichever
    `point_cloud.ply` is newest under `point_cloud/` if the layout
    drifts in a future MonoGS release."""
    final = save_dir / "point_cloud" / "final" / "point_cloud.ply"
    if final.exists():
        return final
    candidates = sorted(
        save_dir.rglob("point_cloud.ply"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _load_trajectory(save_dir: Path) -> Optional[np.ndarray]:
    """Read the keyframe trajectory MonoGS dumps to disk during eval.

    Upstream writes a `traj.txt` (TUM format: `idx tx ty tz qx qy qz qw`)
    when `Results.save_trj` is True. We parse it into an (N, 4, 4)
    cam-from-world array so the SlamProcessor's downstream export
    pipeline can render the camera path. Falls back to None on any
    missing-file or shape mismatch — the splat itself remains the
    primary output."""
    candidates = list(save_dir.rglob("traj.txt")) + list(
        save_dir.rglob("trajectory.txt")
    )
    if not candidates:
        return None
    try:
        rows = np.loadtxt(candidates[0])
    except Exception as exc:  # noqa: BLE001
        log.warning("monogs_batch: traj read failed (%s): %s", candidates[0], exc)
        return None
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.shape[1] < 8:
        return None
    poses = np.empty((rows.shape[0], 4, 4), dtype=np.float32)
    for i, row in enumerate(rows):
        tx, ty, tz, qx, qy, qz, qw = row[1:8]
        norm = float(qw * qw + qx * qx + qy * qy + qz * qz) ** 0.5 or 1.0
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
        M[:3, 3] = (tx, ty, tz)
        poses[i] = M
    return poses
