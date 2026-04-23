"""Source-job → gsplat-input conversion.

A gsplat job reads its camera poses + init point cloud + frames from a
previously-completed SLAM or Lingbot job. This module centralises the
"where's that file on disk" lookups so the trainer and the API endpoint
agree on the expected layout.

Nothing here touches the DB — callers pass in a loaded `Job` record. That
keeps this module importable from the worker container (which only has
read access to `/data`) and the API container (which owns the DB).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from app.config import settings
from app.jobs.schema import AnyJobConfig, Job

log = logging.getLogger(__name__)


@dataclass
class GsplatInputs:
    """Resolved paths the trainer needs to kick off.

    `init_points` is optional when the gsplat config requested random init,
    and `cameras_path` is optional for source jobs that didn't emit a pose
    graph (lingbot paths emit `camera_path.json` but not `pose_graph.json`).
    """

    frames_dir: Path
    init_points: Optional[Path]
    cameras_path: Optional[Path]
    source_processor: str
    source_config: AnyJobConfig


class GsplatInputsError(ValueError):
    """Raised when the source job isn't a valid gsplat input."""


def resolve_inputs(source_job: Job) -> GsplatInputs:
    """Return the file paths a gsplat job will read from a source job.

    Raises `GsplatInputsError` if the source is unusable (still running,
    missing pose graph, etc). The API calls this before enqueueing so the
    user gets a clear error rather than a mid-training failure.
    """
    if source_job.status != "ready":
        raise GsplatInputsError(
            f"source job {source_job.id} is {source_job.status}, "
            "wait for it to reach ready before training"
        )

    frames_dir = settings.job_frames(source_job.id)
    if not frames_dir.exists() or not any(frames_dir.glob("*.png")):
        raise GsplatInputsError(
            f"source job {source_job.id} has no extracted frames — "
            "reingest or pick a different source"
        )

    artifacts_dir = settings.job_artifacts(source_job.id)

    init_points = _pick_first(
        artifacts_dir / "reconstruction.ply",
        artifacts_dir / "pointcloud.ply",
    )
    cameras_path = _pick_first(
        artifacts_dir / "pose_graph.json",
        artifacts_dir / "camera_path.json",
    )

    return GsplatInputs(
        frames_dir=frames_dir,
        init_points=init_points,
        cameras_path=cameras_path,
        source_processor=source_job.config.processor,
        source_config=source_job.config,
    )


def load_init_points(path: Path, max_points: int = 500_000) -> np.ndarray:
    """Read an (N, 6) xyz+rgb array from a PLY. Subsamples if huge.

    Hand-rolled parser that matches the exporter in
    `processors/slam/export.py::write_ply` and Lingbot's trimesh output.
    Falls back to trimesh for non-matching layouts — imported lazily so
    the worker-gs container doesn't need it on the fast path.
    """
    raw = path.read_bytes()
    header, _, body = raw.partition(b"end_header\n")
    header_str = header.decode("ascii", errors="replace")

    if "binary_little_endian" in header_str and "property uchar red" in header_str:
        # Fast path: our own writer format. Parse vertex count and unpack.
        n = _parse_vertex_count(header_str)
        if n <= 0:
            return np.zeros((0, 6), dtype=np.float64)
        dtype = np.dtype(
            [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
             ("r", "u1"), ("g", "u1"), ("b", "u1")]
        )
        arr = np.frombuffer(body, dtype=dtype, count=n)
        xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=-1).astype(np.float64)
        rgb = np.stack([arr["r"], arr["g"], arr["b"]], axis=-1).astype(np.float64)
        out = np.concatenate([xyz, rgb], axis=1)
    else:
        # Fallback: let trimesh figure it out. Slow but robust.
        import trimesh  # noqa: PLC0415

        mesh = trimesh.load(str(path), force="mesh")
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        if hasattr(mesh, "visual") and getattr(mesh.visual, "vertex_colors", None) is not None:
            colors = np.asarray(mesh.visual.vertex_colors, dtype=np.float64)[:, :3]
        else:
            colors = np.full((verts.shape[0], 3), 200, dtype=np.float64)
        out = np.concatenate([verts, colors], axis=1)

    if out.shape[0] > max_points:
        step = int(out.shape[0] // max_points) + 1
        out = out[::step]
    return out


def load_cameras(path: Path) -> list[dict]:
    """Read a pose_graph.json or camera_path.json into a list of
    `{t, q, intrinsics?, source_frame?}` dicts.

    Both schemas converge on the same per-camera dict shape here so the
    trainer can iterate uniformly. Missing intrinsics get defaulted from
    the frame size at training time.
    """
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "keyframes" in data:
        intr = data.get("intrinsics")
        rows: list[dict] = []
        for kf in data["keyframes"]:
            row = {
                "t": kf.get("t", [0.0, 0.0, 0.0]),
                "q": kf.get("q", [0.0, 0.0, 0.0, 1.0]),
                "source_frame": kf.get("source_frame", kf.get("keyframe")),
            }
            if intr is not None:
                row["intrinsics"] = intr
            rows.append(row)
        return rows
    if isinstance(data, dict) and "poses" in data:
        return [
            {"t": p.get("t", [0, 0, 0]), "q": p.get("q", [0, 0, 0, 1])}
            for p in data["poses"]
        ]
    if isinstance(data, list):
        return list(data)
    raise GsplatInputsError(f"unrecognised camera schema in {path}")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _pick_first(*candidates: Path) -> Optional[Path]:
    for c in candidates:
        if c.exists():
            return c
    return None


def _parse_vertex_count(header: str) -> int:
    for line in header.splitlines():
        if line.startswith("element vertex"):
            parts = line.split()
            if len(parts) >= 3:
                try:
                    return int(parts[2])
                except ValueError:
                    return 0
    return 0
