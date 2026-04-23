"""SLAM-specific exporters.

All SLAM backends share the same artifact shape so the viewer + reexport
paths stay simple:

  * `reconstruction.ply` — dense-ish cloud, same format as the lingbot PLY so
    the existing `PointCloud` viewer layer renders it without any changes.
  * `pose_graph.json` — trajectory + intrinsics in a schema the frontend's
    `CameraPath` component can consume.
  * `keyframes.jsonl` — per-keyframe metadata (frame index → pose). Useful
    for Phase-5 gsplat training (treats each row as a camera) and for
    visualisation overlays.
  * `camera_path.json` — live preview trajectory. Overwritten every snapshot.
  * `partial_NNNN.ply` — live preview partial cloud; same format as the
    final `reconstruction.ply`.

`write_all` is the end-of-run entry point called by `SlamProcessor._export`.
`write_ply` / `write_camera_path` are also called mid-run to publish partial
snapshots, so they have to be cheap and fully synchronous.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Optional

import numpy as np

from app.jobs.schema import Artifact

log = logging.getLogger(__name__)

# Cap cloud size so the browser renders it. Mirrors the lingbot PLY cap
# (see `pipeline/export.py::_export_ply_pointcloud`). If a backend produces
# more points we subsample evenly — SLAM clouds are noisy enough that
# dropping every Nth point is fine visually.
_MAX_CLOUD_POINTS = 2_000_000


def write_ply(out: Path, points: np.ndarray) -> None:
    """Write an (N, 6) xyz+rgb float array as a binary_little_endian PLY.

    Hand-rolled rather than pulling in trimesh because:
      * the partial-snapshot path runs mid-inference and we want the import
        cost to be zero (trimesh pulls in a lot on first use);
      * the format here is exactly the three floats + three uchars the
        frontend's PLYLoader expects, so we avoid accidental drift from
        trimesh's exporter bumping the schema.

    Points outside finite range are dropped — SLAM trackers sometimes emit
    NaN points on rejected frames, and those break the loader.
    """
    if points is None or points.size == 0:
        # Write an empty-but-valid PLY so downstream code doesn't have to
        # special-case a missing file.
        _write_empty_ply(out)
        return

    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(
            f"write_ply expected an (N, 6) or (N, 3) array, got shape {arr.shape}"
        )

    xyz = arr[:, :3].astype(np.float32, copy=False)
    finite = np.all(np.isfinite(xyz), axis=1)
    xyz = xyz[finite]
    if arr.shape[1] >= 6:
        rgb = arr[finite, 3:6]
    else:
        rgb = np.full((xyz.shape[0], 3), 200, dtype=np.float64)
    rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8, copy=False)

    if xyz.shape[0] > _MAX_CLOUD_POINTS:
        step = int(xyz.shape[0] // _MAX_CLOUD_POINTS) + 1
        xyz = xyz[::step]
        rgb = rgb[::step]

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        f.write(_ply_header(xyz.shape[0]).encode("ascii"))
        # Interleave xyz floats + rgb uchars per vertex. Using struct.pack
        # per-row would be slow; instead we build a structured buffer.
        dtype = np.dtype(
            [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
             ("r", "u1"), ("g", "u1"), ("b", "u1")]
        )
        buf = np.empty(xyz.shape[0], dtype=dtype)
        buf["x"] = xyz[:, 0]
        buf["y"] = xyz[:, 1]
        buf["z"] = xyz[:, 2]
        buf["r"] = rgb[:, 0]
        buf["g"] = rgb[:, 1]
        buf["b"] = rgb[:, 2]
        f.write(buf.tobytes())


def _write_empty_ply(out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(_ply_header(0).encode("ascii"))


def _ply_header(n: int) -> str:
    return (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )


def write_camera_path(out: Path, poses: list[np.ndarray]) -> None:
    """Serialise a list of 4x4 world-from-camera matrices for the viewer.

    Schema matches what the frontend's `CameraPath` component expects: a
    flat list of `{t: [x,y,z], q: [x,y,z,w]}` entries. Quaternions are
    derived from the rotation block via the standard SVD-stable formula.
    """
    entries = []
    for pose in poses:
        if pose is None:
            continue
        m = np.asarray(pose)
        if m.shape != (4, 4):
            continue
        if not np.all(np.isfinite(m)):
            continue
        t = m[:3, 3].astype(float).tolist()
        q = _rot_to_quat(m[:3, :3]).tolist()
        entries.append({"t": t, "q": q})

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"poses": entries}))


def write_pose_graph(
    out: Path,
    *,
    poses: np.ndarray,
    keyframe_indices: list[int],
    selected_indices: list[int],
    intrinsics: np.ndarray,
    backend_id: str,
) -> None:
    """End-of-run pose graph.

    Richer than camera_path.json: includes per-keyframe intrinsics, the
    source frame index each keyframe came from, and the backend that
    produced it. Consumed by Phase 5 (gsplat training) and Phase 6's
    pose-graph export tool.
    """
    kf_idx_set = set(keyframe_indices)
    rows = []
    # `poses` is indexed by keyframe slot; line it up with source frames
    # via `selected_indices[keyframe_indices[i]]`.
    for i, _kf in enumerate(keyframe_indices):
        if i >= poses.shape[0]:
            break
        pose = poses[i]
        if not np.all(np.isfinite(pose)):
            continue
        source_frame = (
            selected_indices[_kf] if 0 <= _kf < len(selected_indices) else _kf
        )
        rows.append(
            {
                "keyframe": i,
                "source_frame": source_frame,
                "t": pose[:3, 3].astype(float).tolist(),
                "q": _rot_to_quat(pose[:3, :3]).tolist(),
            }
        )

    K = np.asarray(intrinsics, dtype=float)
    payload = {
        "backend": backend_id,
        "intrinsics": {
            "fx": float(K[0, 0]) if K.size else 0.0,
            "fy": float(K[1, 1]) if K.size else 0.0,
            "cx": float(K[0, 2]) if K.size else 0.0,
            "cy": float(K[1, 2]) if K.size else 0.0,
        },
        "keyframes": rows,
        "n_selected_frames": len(selected_indices),
        "n_keyframes": len(rows),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))


def write_keyframes_jsonl(
    out: Path,
    *,
    poses: np.ndarray,
    keyframe_indices: list[int],
    selected_indices: list[int],
    intrinsics: np.ndarray,
) -> None:
    """One line per keyframe. Separate from pose_graph.json because gsplat
    training wants a streamable format — walking a 30k-keyframe JSON blob
    in node is painful, and the training loop can line-scan this."""
    K = np.asarray(intrinsics, dtype=float)
    intr = {
        "fx": float(K[0, 0]) if K.size else 0.0,
        "fy": float(K[1, 1]) if K.size else 0.0,
        "cx": float(K[0, 2]) if K.size else 0.0,
        "cy": float(K[1, 2]) if K.size else 0.0,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for i, _kf in enumerate(keyframe_indices):
            if i >= poses.shape[0]:
                break
            pose = poses[i]
            if not np.all(np.isfinite(pose)):
                continue
            source_frame = (
                selected_indices[_kf] if 0 <= _kf < len(selected_indices) else _kf
            )
            row = {
                "keyframe": i,
                "source_frame": source_frame,
                "t": pose[:3, 3].astype(float).tolist(),
                "q": _rot_to_quat(pose[:3, :3]).tolist(),
                "intrinsics": intr,
            }
            f.write(json.dumps(row) + "\n")


def write_all(
    artifacts_dir: Path,
    *,
    poses: np.ndarray,
    keyframe_indices: list[int],
    selected_indices: list[int],
    points: Optional[np.ndarray],
    intrinsics: np.ndarray,
    backend_id: str,
    splat_ply: Optional[Path] = None,
    trajectory_only: bool = False,
) -> list[Artifact]:
    """Write every end-of-run artifact and return Artifact records.

    Order matters only for the event stream — the runner publishes a single
    "wrote N artifacts" line after this returns. Returned artifacts are the
    same shape the runner persists to the DB via the Artifact schema.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Artifact] = []

    # Pose graph + keyframes always emitted, even when trajectory_only is
    # true. Frontend uses pose_graph.json to render the camera path.
    pose_graph = artifacts_dir / "pose_graph.json"
    write_pose_graph(
        pose_graph,
        poses=poses,
        keyframe_indices=keyframe_indices,
        selected_indices=selected_indices,
        intrinsics=intrinsics,
        backend_id=backend_id,
    )
    outputs.append(
        Artifact(
            name=pose_graph.name,
            kind="pose_graph_json",
            size_bytes=pose_graph.stat().st_size,
        )
    )

    keyframes_jsonl = artifacts_dir / "keyframes.jsonl"
    write_keyframes_jsonl(
        keyframes_jsonl,
        poses=poses,
        keyframe_indices=keyframe_indices,
        selected_indices=selected_indices,
        intrinsics=intrinsics,
    )
    outputs.append(
        Artifact(
            name=keyframes_jsonl.name,
            kind="keyframes_jsonl",
            size_bytes=keyframes_jsonl.stat().st_size,
        )
    )

    # Also keep the live-preview camera_path.json as a final artifact — the
    # viewer already knows how to load it, and tools want a stable filename.
    camera_path = artifacts_dir / "camera_path.json"
    # Rebuild from `poses` directly in case the tracker never emitted a
    # partial snapshot (very short clips). Pass only valid 4x4 matrices.
    pose_list = [poses[i] for i in range(poses.shape[0])] if poses.size else []
    write_camera_path(camera_path, pose_list)
    outputs.append(
        Artifact(
            name=camera_path.name,
            kind="json",
            size_bytes=camera_path.stat().st_size,
        )
    )

    # Point cloud — skipped for trajectory-only backends.
    if not trajectory_only and points is not None and points.size > 0:
        ply = artifacts_dir / "reconstruction.ply"
        write_ply(ply, points)
        outputs.append(
            Artifact(
                name=ply.name,
                kind="ply",
                size_bytes=ply.stat().st_size,
            )
        )

    # MonoGS / Photo-SLAM path: copy the upstream splat file into the
    # artifacts dir so the frontend can load it with a stable name.
    if splat_ply is not None and splat_ply.exists():
        dest = artifacts_dir / "splat.ply"
        if splat_ply.resolve() != dest.resolve():
            shutil.copy2(splat_ply, dest)
        outputs.append(
            Artifact(
                name=dest.name,
                kind="splat_ply",
                size_bytes=dest.stat().st_size,
            )
        )

    return outputs


# ----------------------------------------------------------------------
# Quaternion helper
# ----------------------------------------------------------------------


def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    """SVD-stable rotation-matrix → quaternion (xyzw order).

    Stays tiny on purpose — scipy.spatial.transform would pull in scipy
    just for this, and we already depend on numpy.
    """
    R = np.asarray(R, dtype=float)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=float)
