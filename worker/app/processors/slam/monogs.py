"""MonoGS / Photo-SLAM backend.

Emits a 3D Gaussian Splat scene incrementally as it tracks. Short-circuits
Phase 5: a user who picks MonoGS gets a splat directly out of SLAM and
doesn't need a follow-on gsplat training job.

Until upstream MonoGS is installed in `worker-slam` we fall back to the
simulated session and synthesise a tiny splat PLY from its point cloud so
the downstream splat tools have something to exercise.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar, Optional

import numpy as np

from app.processors.slam.base import FinalResult, SlamProcessor, SlamSession
from app.processors.slam.tracker import SimulatedSlamSession

log = logging.getLogger(__name__)


class _MonogsSession(SimulatedSlamSession):
    """MonoGS-shaped placeholder.

    Real MonoGS maintains a live 3D-GS scene and refines it across
    keyframes. Here we piggyback on the simulated tracker's point cloud
    and turn it into a bare-minimum splat PLY at finalize time — just
    enough for the splat viewer + tool panel to light up.
    """

    backend_id: ClassVar[str] = "monogs"
    corners_per_keyframe: ClassVar[int] = 200
    keyframe_every: ClassVar[int] = 1
    depth_noise: ClassVar[float] = 0.12

    # The processor reads `splat_ply_path` off the FinalResult and copies
    # the file into the artifacts dir. We write into the session's own
    # temp spot; the processor passes its `ctx.artifacts_dir` only via
    # `_make_session`, which this session persists in `_artifact_dir`.
    _artifact_dir: Optional[Path] = None

    def set_artifact_dir(self, path: Path) -> None:
        self._artifact_dir = path

    def finalize(self) -> FinalResult:
        result = super().finalize()
        splat_path = self._maybe_write_splat_ply(result)
        if splat_path is not None:
            result = FinalResult(
                poses=result.poses,
                keyframe_indices=result.keyframe_indices,
                points=result.points,
                splat_ply_path=splat_path,
                diagnostics={**result.diagnostics, "splat_source": "monogs_sim"},
            )
        return result

    def _maybe_write_splat_ply(self, result: FinalResult) -> Optional[Path]:
        if self._artifact_dir is None or result.points is None:
            return None
        if result.points.size == 0:
            return None
        # Minimal 3D-GS PLY. Real MonoGS writes covariance + SH
        # coefficients; here we emit just positions + colours tagged in
        # the GS-compatible property layout so the viewer's splat loader
        # doesn't reject the file. Phase 6's splat renderer treats
        # missing per-Gaussian fields as defaults.
        out = self._artifact_dir / "splat_monogs.ply"
        _write_minimal_splat_ply(out, result.points)
        return out


class MonogsProcessor(SlamProcessor):
    """MonoGS (Photo-SLAM variant). Produces a splat alongside the usual
    trajectory + cloud."""

    id: ClassVar[str] = "monogs"  # type: ignore[assignment]
    display_name: ClassVar[str] = "MonoGS"
    supported_artifacts = frozenset(
        {"ply", "json", "glb", "pose_graph_json", "keyframes_jsonl", "splat_ply"}
    )

    def _make_session(self, ctx) -> SlamSession:  # type: ignore[override]
        log.info("monogs: using simulated tracker (upstream not installed)")
        session = _MonogsSession()
        session.set_artifact_dir(ctx.artifacts_dir)
        return session


# ----------------------------------------------------------------------
# Minimal splat PLY writer (placeholder).
# ----------------------------------------------------------------------


def _write_minimal_splat_ply(out: Path, points: np.ndarray) -> None:
    """Write a bare-bones splat-flavoured PLY. Enough for the splat viewer
    to load a placeholder scene; will be replaced with a real MonoGS
    exporter once upstream is wired in."""
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return
    xyz = arr[:, :3].astype(np.float32)
    if arr.shape[1] >= 6:
        rgb = np.clip(arr[:, 3:6], 0, 255).astype(np.uint8)
    else:
        rgb = np.full((xyz.shape[0], 3), 200, dtype=np.uint8)
    # Fixed default scale/opacity so the viewer draws visible splats.
    scale = np.full((xyz.shape[0], 3), -2.3, dtype=np.float32)  # log-scale
    opacity = np.full((xyz.shape[0],), 0.8, dtype=np.float32)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {xyz.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float scale_0\n"
        "property float scale_1\n"
        "property float scale_2\n"
        "property float opacity\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    dtype = np.dtype(
        [
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("s0", "<f4"), ("s1", "<f4"), ("s2", "<f4"),
            ("opacity", "<f4"),
            ("r", "u1"), ("g", "u1"), ("b", "u1"),
        ]
    )
    buf = np.empty(xyz.shape[0], dtype=dtype)
    buf["x"] = xyz[:, 0]
    buf["y"] = xyz[:, 1]
    buf["z"] = xyz[:, 2]
    buf["s0"] = scale[:, 0]
    buf["s1"] = scale[:, 1]
    buf["s2"] = scale[:, 2]
    buf["opacity"] = opacity
    buf["r"] = rgb[:, 0]
    buf["g"] = rgb[:, 1]
    buf["b"] = rgb[:, 2]

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(buf.tobytes())
