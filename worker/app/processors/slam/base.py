"""Shared SLAM processor framework.

Every SLAM backend (DROID-SLAM, MASt3R-SLAM, DPVO, MonoGS) subclasses
`SlamProcessor` and plugs in a `SlamSession` — a small per-backend object
that implements `step(idx, img) -> FrameUpdate` and `finalize() -> FinalResult`.
`SlamProcessor.run(ctx)` owns all the I/O: ingest, cancel checks, keyframe
gating, live preview publishing, final export, optional Poisson meshing.

Why it looks like this:

  * Keeps the backend implementations tight — a new SLAM backend only
    writes a tracker, not a fresh end-to-end runner.
  * Centralises the preview publishing so the frontend sees the same
    stream shape (`camera_path.json` + `partial_NNN.ply`) regardless of
    which backend the user picked.
  * Lets the simulated tracker stand in for backends whose upstream
    CUDA integration hasn't landed yet — users still get a working
    pipeline end-to-end, marked as "simulated" in event data so the UI
    can badge it.
"""

from __future__ import annotations

import abc
import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Optional

import numpy as np

from app.jobs.schema import Artifact, JobEvent, _SlamConfigBase
from app.processors.base import JobContext, Processor, ProcessorResult

log = logging.getLogger(__name__)


@dataclass
class FrameUpdate:
    """Per-frame output from a `SlamSession.step` call.

    `pose_matrix` is a 4x4 world-from-camera matrix. `new_points` is a
    new-points-only array (N, 6) — [x, y, z, r, g, b] — so the processor
    can stream incremental clouds without recomputing the whole scene.
    Either field may be None when the tracker declines to update that
    frame (e.g. rejected as a low-quality frame).
    """

    pose_matrix: Optional[np.ndarray] = None
    new_points: Optional[np.ndarray] = None
    is_keyframe: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class FinalResult:
    """End-of-session artifacts returned by `SlamSession.finalize`.

    `poses` is the full keyframe-indexed trajectory (K, 4, 4).
    `points` is the dense point cloud (M, 6). `splat_ply_path` is set by
    splat-producing backends (MonoGS) so the processor copies the file
    into the artifacts dir instead of rewriting the cloud.
    """

    poses: np.ndarray
    keyframe_indices: list[int]
    points: Optional[np.ndarray] = None
    splat_ply_path: Optional[Path] = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


class SlamSession(abc.ABC):
    """Per-backend tracking state. One instance per job.

    The processor creates the session, iterates over `(idx, img)` pairs,
    and calls `finalize()` at the end. Each method is synchronous — the
    processor runs them in a worker thread to keep the asyncio loop free.
    """

    # Identifier emitted in diagnostics so consumers know which backend
    # produced an artifact. Set by subclasses.
    backend_id: ClassVar[str] = "unknown"
    # True when the tracker cannot provide real 3D points (trajectory-only
    # backends like raw DPVO). The processor skips cloud publishing for
    # these and the UI falls back to camera-path-only rendering.
    trajectory_only: ClassVar[bool] = False

    @abc.abstractmethod
    def start(
        self,
        intrinsics: np.ndarray,
        image_shape: tuple[int, int],
    ) -> None:
        """Called once before the first `step`. `intrinsics` is a 3x3 K."""

    @abc.abstractmethod
    def step(self, idx: int, img: np.ndarray) -> FrameUpdate:
        """Track one frame. `img` is a BGR uint8 HxWx3 array."""

    @abc.abstractmethod
    def finalize(self) -> FinalResult:
        """Flush any pending optimisation (bundle adjustment, map clean-up)
        and return the final trajectory + cloud.

        Implementations must still be callable after a mid-run cancel so
        the processor can salvage a partial result — in that case return
        whatever keyframes have been committed so far.
        """


class SlamProcessor(Processor):
    """Base class for every SLAM backend.

    Subclasses:
      * set `id` / `worker_class` class attrs (Phase 1 contract).
      * implement `_make_session(ctx)` to return their SlamSession.

    Everything else — ingest, intrinsics assembly, cancel checks, live
    preview publishing, exports, Poisson meshing — is shared.
    """

    kind: ClassVar[str] = "slam"  # type: ignore[assignment]
    worker_class: ClassVar[str] = "slam"  # type: ignore[assignment]
    supported_artifacts = frozenset(
        {"ply", "json", "glb", "pose_graph_json", "keyframes_jsonl", "splat_ply"}
    )

    # Human-friendly name for log lines and the UI's processor label.
    display_name: ClassVar[str] = "SLAM"

    @abc.abstractmethod
    def _make_session(self, ctx: JobContext) -> SlamSession:
        """Return the backend-specific tracker. Subclass hook."""

    async def run(self, ctx: JobContext) -> ProcessorResult:
        cfg = ctx.config
        if not isinstance(cfg, _SlamConfigBase):
            raise TypeError(
                f"{self.__class__.__name__} expected a SLAM config, got "
                f"{type(cfg).__name__}"
            )

        await ctx.publish(
            JobEvent(
                job_id=ctx.job_id,
                stage="queue",
                message=f"{self.display_name} starting",
                data={"backend": self.id, "uploads": [p.name for p in ctx.uploads]},
            )
        )
        ctx.check_cancel()

        frames_total = await self._ingest(ctx)
        ctx.check_cancel()

        kept_indices = await self._gate_frames(ctx, frames_total)

        await ctx.set_status("slam")
        session = self._make_session(ctx)
        result = await self._track(ctx, session, kept_indices)
        ctx.check_cancel()

        await ctx.set_status("export")
        artifacts = await self._export(ctx, session, result, kept_indices)

        if cfg.run_poisson_mesh and result.points is not None:
            mesh_art = await self._run_poisson(ctx, result)
            if mesh_art is not None:
                artifacts.append(mesh_art)

        return ProcessorResult(
            artifacts=artifacts,
            extras={
                "backend": self.id,
                "keyframes": len(result.keyframe_indices),
                "trajectory_only": session.trajectory_only,
                **result.diagnostics,
            },
        )

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------

    async def _ingest(self, ctx: JobContext) -> int:
        """Re-use the shared ingest pipeline. Same FPV preproc stages as
        lingbot so SLAM jobs get the full Phase-3 clean-up for free."""
        from app.pipeline.ingest import concat_videos_to_frames

        await ctx.set_status("ingest")
        frames_total = await concat_videos_to_frames(
            job_id=ctx.job_id,
            sources=ctx.uploads,
            dest=ctx.frames_dir,
            config=ctx.config,  # type: ignore[arg-type]
            publish=ctx.publish,
        )
        await ctx.set_frames_total(frames_total)
        return frames_total

    async def _gate_frames(self, ctx: JobContext, frames_total: int) -> list[int]:
        """Apply the keyframe policy + the `max_frames` cap.

        Returns the list of frame indices the tracker should see. For
        `keyframe_policy="score_gated"`, reads `frame_scores.jsonl` if the
        keyframe scorer emitted one; otherwise falls back to a plain
        stride loop.
        """
        cfg: _SlamConfigBase = ctx.config  # type: ignore[assignment]
        stride = max(1, int(cfg.stride))

        if cfg.keyframe_policy == "score_gated":
            scores_path = ctx.frames_dir.parent / "frame_scores.jsonl"
            if scores_path.exists():
                return _read_score_gated_indices(
                    scores_path,
                    quantile=cfg.score_gate_quantile,
                    max_frames=cfg.max_frames,
                    stride=stride,
                )

        indices = list(range(0, frames_total, stride))
        if cfg.max_frames is not None:
            indices = indices[: cfg.max_frames]
        await ctx.publish(
            JobEvent(
                job_id=ctx.job_id,
                stage="slam",
                message=(
                    f"keyframe policy={cfg.keyframe_policy}: "
                    f"{len(indices)}/{frames_total} frames selected"
                ),
                data={"selected": len(indices), "total": frames_total},
            )
        )
        return indices

    async def _track(
        self,
        ctx: JobContext,
        session: SlamSession,
        indices: list[int],
    ) -> FinalResult:
        """Drive the tracker over the selected frames. Publishes live
        preview artifacts every N keyframes per config."""
        import cv2

        cfg: _SlamConfigBase = ctx.config  # type: ignore[assignment]
        frame_paths = _resolve_frame_paths(ctx.frames_dir, indices)
        if not frame_paths:
            raise RuntimeError("slam: no frames to track")

        # Read the first frame to seed image_shape + intrinsics.
        first = cv2.imread(str(frame_paths[0]))
        if first is None:
            raise RuntimeError(f"slam: could not read seed frame {frame_paths[0]}")
        h, w = first.shape[:2]
        intrinsics = _build_intrinsics(cfg, w=w, h=h)

        await asyncio.to_thread(session.start, intrinsics, (h, w))

        keyframe_poses: list[np.ndarray] = []
        accumulated_points: list[np.ndarray] = []
        kf_count = 0
        total = len(frame_paths)

        for n, (idx, path) in enumerate(zip(indices, frame_paths)):
            ctx.check_cancel()
            img = cv2.imread(str(path))
            if img is None:
                continue
            update: FrameUpdate = await asyncio.to_thread(session.step, idx, img)
            if update.is_keyframe and update.pose_matrix is not None:
                keyframe_poses.append(update.pose_matrix)
                kf_count += 1
                if update.new_points is not None and update.new_points.size:
                    accumulated_points.append(update.new_points)

                if (
                    cfg.partial_snapshot_every > 0
                    and kf_count % cfg.partial_snapshot_every == 0
                ):
                    await self._publish_partial(
                        ctx,
                        kf_idx=kf_count,
                        poses=keyframe_poses,
                        points_chunks=accumulated_points,
                        trajectory_only=session.trajectory_only,
                    )

            if n % max(1, total // 20) == 0:
                await ctx.publish(
                    JobEvent(
                        job_id=ctx.job_id,
                        stage="slam",
                        message=f"tracked {n + 1}/{total} frames ({kf_count} keyframes)",
                        progress=(n + 1) / total,
                        data={"keyframes": kf_count, **update.diagnostics},
                    )
                )

        ctx.check_cancel()
        await ctx.publish(
            JobEvent(
                job_id=ctx.job_id,
                stage="slam",
                message=f"finalising trajectory ({kf_count} keyframes)",
            )
        )
        return await asyncio.to_thread(session.finalize)

    async def _export(
        self,
        ctx: JobContext,
        session: SlamSession,
        result: FinalResult,
        indices: list[int],
    ) -> list[Artifact]:
        from app.processors.slam import export as slam_export

        ctx.artifacts_dir.mkdir(parents=True, exist_ok=True)
        intrinsics = _build_intrinsics(
            ctx.config,  # type: ignore[arg-type]
            w=result.diagnostics.get("image_width", 0),
            h=result.diagnostics.get("image_height", 0),
        )

        artifacts = await asyncio.to_thread(
            slam_export.write_all,
            ctx.artifacts_dir,
            poses=result.poses,
            keyframe_indices=result.keyframe_indices,
            selected_indices=indices,
            points=result.points,
            intrinsics=intrinsics,
            backend_id=self.id,
            splat_ply=result.splat_ply_path,
            trajectory_only=session.trajectory_only,
        )
        await ctx.publish(
            JobEvent(
                job_id=ctx.job_id,
                stage="export",
                message=(
                    f"wrote {len(artifacts)} artifact(s): "
                    + ", ".join(a.name for a in artifacts)
                ),
                progress=1.0,
            )
        )
        return artifacts

    async def _run_poisson(
        self, ctx: JobContext, result: FinalResult
    ) -> Optional[Artifact]:
        """Optional Poisson surface reconstruction over the SLAM cloud.

        Reuses `mesh.ops.surface_recon` — keeps a single meshing
        implementation across lingbot/SLAM.
        """
        try:
            from app.mesh.ops import surface_recon
        except ImportError as exc:
            log.warning("slam: mesh.ops unavailable, skipping Poisson (%s)", exc)
            return None

        cfg: _SlamConfigBase = ctx.config  # type: ignore[assignment]
        ply_path = ctx.artifacts_dir / "reconstruction.ply"
        if not ply_path.exists():
            return None
        out_glb = ctx.artifacts_dir / "slam_surface.glb"

        await ctx.set_status("meshing")
        await ctx.publish(
            JobEvent(
                job_id=ctx.job_id,
                stage="meshing",
                message=f"poisson surface recon (depth={cfg.poisson_depth})",
            )
        )
        try:
            await asyncio.to_thread(
                surface_recon, ply_path, out_glb, depth=cfg.poisson_depth
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("slam poisson failed")
            await ctx.publish(
                JobEvent(
                    job_id=ctx.job_id,
                    stage="meshing",
                    level="error",
                    message=f"poisson failed: {exc}",
                )
            )
            return None

        return Artifact(
            name=out_glb.name,
            kind="glb",
            size_bytes=out_glb.stat().st_size,
        )

    async def _publish_partial(
        self,
        ctx: JobContext,
        *,
        kf_idx: int,
        poses: list[np.ndarray],
        points_chunks: list[np.ndarray],
        trajectory_only: bool,
    ) -> None:
        """Emit a partial PLY + camera_path.json so the viewer animates
        the reconstruction growing. Idempotent — overwrites the same
        filenames per snapshot index."""
        from app.processors.slam import export as slam_export

        partial_dir = ctx.artifacts_dir
        partial_dir.mkdir(parents=True, exist_ok=True)
        name = f"partial_{kf_idx:04d}.ply"
        path = partial_dir / name
        camera_path = partial_dir / "camera_path.json"

        def _write() -> None:
            if not trajectory_only and points_chunks:
                cloud = np.concatenate(points_chunks, axis=0)
                slam_export.write_ply(path, cloud)
            slam_export.write_camera_path(camera_path, poses)

        await asyncio.to_thread(_write)
        await ctx.publish(
            JobEvent(
                job_id=ctx.job_id,
                stage="artifact",
                message=f"partial snapshot at keyframe {kf_idx}",
                data={"partial": name, "keyframes": kf_idx},
            )
        )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _resolve_frame_paths(frames_dir: Path, indices: list[int]) -> list[Path]:
    """Map a list of ingest-frame indices to the corresponding PNG paths."""
    all_paths = sorted(frames_dir.glob("*.png"))
    out: list[Path] = []
    for i in indices:
        if 0 <= i < len(all_paths):
            out.append(all_paths[i])
    return out


def _build_intrinsics(cfg: _SlamConfigBase, w: int, h: int) -> np.ndarray:
    """Compose a 3x3 K matrix from config; falls back to a 60° auto-FOV."""
    if cfg.calibration == "manual" and cfg.fx and cfg.fy and cfg.cx and cfg.cy:
        fx, fy, cx, cy = cfg.fx, cfg.fy, cfg.cx, cfg.cy
    else:
        fx = fy = float(max(w, 1)) / (2.0 * np.tan(np.deg2rad(60.0) / 2.0))
        cx = w / 2.0 if w else 0.0
        cy = h / 2.0 if h else 0.0
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _read_score_gated_indices(
    scores_path: Path,
    *,
    quantile: float,
    max_frames: Optional[int],
    stride: int,
) -> list[int]:
    """Read `frame_scores.jsonl` and keep frames at or above the quantile."""
    rows: list[dict] = []
    for line in scores_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not rows:
        return []
    qualities = sorted(r.get("quality", 0.0) for r in rows)
    cutoff_idx = int(max(0, min(len(qualities) - 1, quantile * (len(qualities) - 1))))
    cutoff = qualities[cutoff_idx]
    kept = [r["index"] for r in rows if r.get("quality", 0.0) >= cutoff]
    if stride > 1:
        kept = kept[::stride]
    if max_frames is not None:
        kept = kept[:max_frames]
    return kept
