"""Splat-specific exporters.

The 3D-GS PLY format is just a PLY with extra per-vertex properties
(scale_0..2, rot_0..3, opacity, SH coefficients). Viewers like Spark and
mkkellogg/GaussianSplats3D read the standard property names directly, so
we stick to that convention here and skip inventing a fourth variant.

Written as free functions so both the simulated trainer (Phase 5 today)
and the real gsplat trainer (when it lands) call the same writer — no
subclass needed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


# Standard 3DGS property layout (Inria/Kerbl+2023):
#   float x,y,z                     (position)
#   float nx,ny,nz                  (unused; kept for viewer compat)
#   float f_dc_0,f_dc_1,f_dc_2      (DC SH coeffs — effectively RGB)
#   float opacity                   (logit; sigmoid at render)
#   float scale_0,scale_1,scale_2   (log-space per-axis scale)
#   float rot_0,rot_1,rot_2,rot_3   (quaternion, wxyz)
#
# SH degree>0 adds f_rest_<K> columns; we skip those for SH degree 0 / the
# simulated trainer. Spark treats missing f_rest_* as zero.


def write_splat_ply(
    out: Path,
    *,
    means: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
) -> None:
    """Write an SH-degree-0 3DGS PLY.

    Shapes:
      means     (N, 3) float
      colors    (N, 3) float in [0, 1]
      opacities (N,)   float (raw; caller is responsible for converting
                             to logit space if their trainer expects it)
      scales    (N, 3) float (log-space)
      rotations (N, 4) float (quaternion wxyz, unit-normalised)
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    n = means.shape[0]
    if n == 0:
        out.write_bytes(_header(0).encode("ascii"))
        return

    dtype = np.dtype(
        [
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
            ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"),
            ("opacity", "<f4"),
            ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
            ("rot_0", "<f4"), ("rot_1", "<f4"),
            ("rot_2", "<f4"), ("rot_3", "<f4"),
        ]
    )
    buf = np.empty(n, dtype=dtype)
    buf["x"] = means[:, 0].astype(np.float32)
    buf["y"] = means[:, 1].astype(np.float32)
    buf["z"] = means[:, 2].astype(np.float32)
    # Normals are left at zero; viewers don't use them for splat render.
    buf["nx"] = 0.0
    buf["ny"] = 0.0
    buf["nz"] = 0.0
    # DC SH = (color - 0.5) / SH_C0 where SH_C0 = 0.28209479177387814.
    # Most viewers apply the inverse at render time.
    SH_C0 = 0.28209479177387814
    rgb = np.clip(colors, 0.0, 1.0)
    buf["f_dc_0"] = ((rgb[:, 0] - 0.5) / SH_C0).astype(np.float32)
    buf["f_dc_1"] = ((rgb[:, 1] - 0.5) / SH_C0).astype(np.float32)
    buf["f_dc_2"] = ((rgb[:, 2] - 0.5) / SH_C0).astype(np.float32)
    buf["opacity"] = opacities.astype(np.float32)
    buf["scale_0"] = scales[:, 0].astype(np.float32)
    buf["scale_1"] = scales[:, 1].astype(np.float32)
    buf["scale_2"] = scales[:, 2].astype(np.float32)
    buf["rot_0"] = rotations[:, 0].astype(np.float32)
    buf["rot_1"] = rotations[:, 1].astype(np.float32)
    buf["rot_2"] = rotations[:, 2].astype(np.float32)
    buf["rot_3"] = rotations[:, 3].astype(np.float32)

    with out.open("wb") as f:
        f.write(_header(n).encode("ascii"))
        f.write(buf.tobytes())


def write_cameras_json(out: Path, cameras: list[dict], *, backend: str) -> None:
    """Persist the resolved camera list the trainer consumed.

    Matters for the splat viewer's "fly through training cameras" mode and
    for any re-training run — callers can tweak the cameras and restart
    without going all the way back to the source SLAM job.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"backend": backend, "cameras": cameras})
    )


def append_training_log(
    out: Path,
    row: dict,
) -> None:
    """Append a training-log line. Opened `a` every call — cheap, and
    means a mid-run crash leaves a well-formed file behind."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.write(json.dumps(row) + "\n")


def write_sogs_placeholder(
    out: Path,
    *,
    splat_ply_path: Path,
    iterations: int,
    n_gaussians: int,
) -> Optional[Path]:
    """Placeholder for a future SOGS (compressed splat) exporter.

    The real implementation invokes `splat-transform` / the nerfstudio
    SOGS encoder. Until that dep lands in `worker-gs` we emit a sidecar
    JSON that describes the source PLY so the UI can at least show a
    download link; the splat viewer keeps rendering from the PLY.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    # Sidecar-as-JSON: real SOGS is a tarball of bin+meta, but the UI
    # download handler just streams whatever file is here. Leaving the
    # format stub-like keeps it obvious this isn't a real SOGS yet.
    out.write_text(
        json.dumps(
            {
                "format": "sogs_placeholder",
                "source": splat_ply_path.name,
                "iterations": iterations,
                "n_gaussians": n_gaussians,
                "note": "replace with real SOGS output once the encoder ships",
            }
        )
    )
    return out


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _header(n: int) -> str:
    return (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property float f_dc_0\n"
        "property float f_dc_1\n"
        "property float f_dc_2\n"
        "property float opacity\n"
        "property float scale_0\n"
        "property float scale_1\n"
        "property float scale_2\n"
        "property float rot_0\n"
        "property float rot_1\n"
        "property float rot_2\n"
        "property float rot_3\n"
        "end_header\n"
    )
