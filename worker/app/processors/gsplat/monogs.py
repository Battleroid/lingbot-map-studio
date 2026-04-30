"""MonoGS / Photo-SLAM backend.

Emits a 3D Gaussian Splat scene end-to-end from uploaded footage. Lives
under the Gaussian Splat tab as an alternate backend to
`gsplat.trainer` — same output artifact (`splat.ply`), different input
shape: MonoGS starts from raw frames, gsplat trains off a completed
SLAM/Lingbot source job.

Runs in the `worker-gs` container. The processor invokes upstream MonoGS
as a subprocess via `monogs_batch.run_monogs_batch`: builds a TUM-shaped
workspace from the extracted frames, generates a YAML config, runs
`slam.py`, and copies the resulting splat PLY into the job's artifacts.

Why batch and not streaming: upstream `muskie82/MonoGS` doesn't expose a
per-frame `process_frame` API. Its only entrypoint is
`slam.SLAM(config)`, a multi-process driver that reads a Dataset off
disk and runs frontend tracking + backend mapping until the dataset is
exhausted. A streaming wrapper would require a custom Dataset feeding
upstream's frontend Process pair via mp.Queue (see `monogs_streaming.py`
for that path; it's used by the live capture preview). The batch
wrapper is the simpler shape for the post-stop reconstruction job.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import ClassVar, Optional

import numpy as np

from app.jobs.schema import JobEvent
from app.processors.slam.base import (
    FinalResult,
    SlamProcessor,
    SlamSession,
    _build_intrinsics,
    _resolve_frame_paths,
)
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


class _NoopSession(SlamSession):
    """Stub session for `MonogsProcessor`.

    `MonogsProcessor` overrides `_track` to drive upstream MonoGS as a
    subprocess (see `monogs_batch.run_monogs_batch`); it never calls
    `start` / `step` / `finalize` on the session. The base class's
    `run()` method still constructs a session via `_make_session`
    before passing it down, and `_export` reads `session.trajectory_only`
    off it, so the stub has to satisfy those two contracts."""

    backend_id: ClassVar[str] = "monogs"

    def start(self, intrinsics, image_shape) -> None:  # noqa: D401, ARG002
        return

    def step(self, idx, img):  # noqa: D401, ARG002
        from app.processors.slam.base import FrameUpdate
        return FrameUpdate()

    def finalize(self) -> FinalResult:  # noqa: D401
        return FinalResult(
            poses=np.empty((0, 4, 4), dtype=np.float32),
            keyframe_indices=[],
            points=None,
            diagnostics={"backend": "monogs", "noop": True},
        )


class MonogsProcessor(SlamProcessor):
    """MonoGS (Photo-SLAM variant). Produces a splat alongside the usual
    trajectory + cloud.

    Bypasses the streaming SlamSession surface that DROID / DPVO /
    MASt3R-SLAM use because upstream MonoGS isn't streaming-shaped.
    `_track` runs upstream as a subprocess via the batch wrapper and
    builds a `FinalResult` from the resulting splat PLY + trajectory."""

    id: ClassVar[str] = "monogs"  # type: ignore[assignment]
    display_name: ClassVar[str] = "MonoGS"
    supported_artifacts = frozenset(
        {"ply", "json", "glb", "pose_graph_json", "keyframes_jsonl", "splat_ply"}
    )

    def _make_session(self, ctx) -> SlamSession:  # type: ignore[override]
        return _NoopSession()

    async def _track(  # type: ignore[override]
        self,
        ctx,
        session: SlamSession,
        indices: list[int],
    ) -> FinalResult:
        # Lazy import — `monogs_batch` pulls in `yaml` which lives only
        # in the worker-gs / api images, and this file is otherwise
        # importable from any worker container.
        from app.processors.gsplat import monogs_batch  # noqa: PLC0415
        import cv2  # noqa: PLC0415

        cfg = ctx.config
        frame_paths = _resolve_frame_paths(ctx.frames_dir, indices)
        if not frame_paths:
            raise RuntimeError("monogs: no frames to reconstruct")

        first = cv2.imread(str(frame_paths[0]))
        if first is None:
            raise RuntimeError(
                f"monogs: could not read seed frame {frame_paths[0]}"
            )
        h, w = first.shape[:2]
        intrinsics = _build_intrinsics(cfg, w=w, h=h)

        workspace = ctx.artifacts_dir / ".monogs_workspace"
        try:
            result = await monogs_batch.run_monogs_batch(
                job_id=ctx.job_id,
                frame_paths=frame_paths,
                intrinsics=intrinsics,
                image_shape=(h, w),
                workspace_root=workspace,
                publish=ctx.publish,
                fps=float(getattr(cfg, "fps", 10.0) or 10.0),
            )
        except monogs_batch.MonogsBatchUnavailableError as exc:
            await ctx.publish(
                JobEvent(
                    job_id=ctx.job_id,
                    stage="system",
                    level="error",
                    message=str(exc),
                    data={"missing_dep": "monogs"},
                )
            )
            raise

        # Trajectory may be missing on very short clips. Fall back to a
        # single identity pose so the export pipeline can still write a
        # pose_graph.json (the splat itself is the primary artifact).
        if result.trajectory is None or result.trajectory.shape[0] == 0:
            poses = np.eye(4, dtype=np.float32)[None]
            keyframe_indices = [0]
        else:
            poses = result.trajectory
            keyframe_indices = result.keyframe_indices or list(
                range(poses.shape[0])
            )

        return FinalResult(
            poses=poses,
            keyframe_indices=keyframe_indices,
            points=None,  # MonoGS's primary output is the splat PLY.
            splat_ply_path=result.splat_ply,
            diagnostics={
                "backend": "monogs",
                "splat_source": "monogs_cuda",
                "n_poses": int(poses.shape[0]),
                "image_width": int(w),
                "image_height": int(h),
            },
        )


class MonogsSessionUnavailableError(RuntimeError):
    """Raised by `select_session_cls()` when the real MonoGS CUDA
    session can't be loaded. Mirrors `GsplatTrainerUnavailableError`:
    we no longer silently fall back to the placeholder, since the
    placeholder produces synthetic geometry that doesn't represent
    the scanned scene.

    `_MonogsSession` is still available for tests (which instantiate
    it directly). Production code paths that go through the resolver
    get the real CUDA session or a clean error."""


def select_session_cls() -> type[SlamSession]:
    """Resolve the real MonoGS CUDA session. Raises
    `MonogsSessionUnavailableError` with install instructions when
    any prerequisite is missing. No simulated fallback — gsplat
    output that "looks real but isn't" is the bug we're fixing.
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:
        raise MonogsSessionUnavailableError(
            "monogs: torch is not installed in this worker. The real "
            "MonoGS tracker requires the worker-gs image (built from "
            "worker/Dockerfile.gs)."
        ) from exc
    if not torch.cuda.is_available():
        raise MonogsSessionUnavailableError(
            "monogs: torch.cuda.is_available() is False. The MonoGS "
            "tracker needs an NVIDIA GPU + nvidia-container-toolkit "
            "passthrough; check the worker-gs container's "
            "deploy.resources.reservations.devices in docker-compose.yml."
        )
    # MonoGS isn't a pip-installable package — the Dockerfile clones
    # the source into /opt/monogs and adds it to PYTHONPATH. Probe a
    # couple of likely top-level imports.
    import importlib  # noqa: PLC0415

    try:
        try:
            importlib.import_module("gaussian_splatting")
        except ImportError:
            # Fall back to the legacy probe — some forks namespace
            # everything under `monogs`.
            importlib.import_module("monogs")
    except Exception as exc:  # noqa: BLE001
        raise MonogsSessionUnavailableError(
            "monogs: the upstream MonoGS source isn't importable in "
            f"this worker ({type(exc).__name__}: {exc}). The Dockerfile "
            "should `git clone` it into /opt/monogs and add that path "
            "to PYTHONPATH; see worker/Dockerfile.gs."
        ) from exc

    # Run the deep probe NOW (not at first frame) so the resolver
    # surfaces "no streaming driver in upstream" as an availability
    # error here, instead of crashing inside `MonogsCudaSession.start`
    # mid-job with a `TypeError: GaussianModel.__init__() missing
    # 1 required positional argument: 'sh_degree'` (which is what
    # happens when the speculative probe falls through to the splat
    # data class). The resolver returns a class with one of
    # process_frame / track / step or raises ImportError with an
    # explanation of why streaming MonoGS isn't actually wired up.
    from app.processors.gsplat.monogs_cuda import (  # noqa: PLC0415
        MonogsCudaSession,
        _resolve_mapper_cls,
    )
    try:
        _resolve_mapper_cls()
    except Exception as exc:  # noqa: BLE001
        raise MonogsSessionUnavailableError(str(exc)) from exc
    return MonogsCudaSession


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
