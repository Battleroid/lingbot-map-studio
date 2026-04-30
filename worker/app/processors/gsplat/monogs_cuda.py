"""Real MonoGS / Photo-SLAM CUDA session.

Wraps the upstream `muskie82/MonoGS` tracker. Lives in its own module so
the rest of the gsplat package stays importable on a CPU-only host —
`import monogs` at the top of this file would crash any container
without the package + its compiled CUDA extensions.

The session is selected at runtime by `select_monogs_session_cls()` in
`monogs.py`: when `monogs` (the upstream package) imports cleanly we
use `MonogsCudaSession`; otherwise we fall back to the simulated
session and Phase 0 emits a warn event.

Implementation notes
- MonoGS is the splat-emitting SLAM. Unlike the other SLAM backends
  (DROID, MASt3R, DPVO), it produces a complete Gaussian-Splat scene
  alongside the trajectory + cloud — that's why it's surfaced under the
  Gaussian-Splat tab in the UI.
- Upstream MonoGS is research-grade with a multi-process architecture
  (frontend tracking + backend mapping). The exact single-process API
  surface shifts between commits. The wrapper below probes a couple of
  likely entry points and gracefully reports if it can't find one — in
  that case auto-select falls back to the simulated session.
- Output: the splat PLY lives at `<artifacts_dir>/splat_monogs.ply`.
  The processor reads `splat_ply_path` off the FinalResult and copies
  the file into the job's artifacts directory.
- CUDA registration: MonoGS bundles its own 3DGS rasterizer (a fork of
  graphdeco-inria/gaussian-splatting). It can coexist with the `gsplat`
  PyPI package (Phase 1) inside worker-gs because each registers under
  its own Python module namespace. If you see import-order issues at
  runtime, import MonoGS *after* gsplat in the trainer factory.

API expectations:

  from monogs.gaussian_splatting.scene import GaussianMapper
  mapper = GaussianMapper(ckpt=..., device="cuda")
  for idx, img in enumerate(frames):
      mapper.process_frame(idx, img, intrinsics=K)
  result = mapper.finalize()  # → {poses, splat_ply_path, points}

We keep `start()` / `step()` / `finalize()` the same shape as the other
SLAM backends so the SlamProcessor's lifecycle is unchanged.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, ClassVar, Optional

import numpy as np

from app.processors.slam.base import FinalResult, FrameUpdate, SlamSession

log = logging.getLogger(__name__)


def _default_checkpoint_path() -> Optional[str]:
    env = os.environ.get("MONOGS_CKPT")
    if env:
        return env
    cached = Path("/models/monogs/monogs.pth")
    if cached.exists():
        return str(cached)
    return None


def _quat_xyzw_translate_to_matrix(pose_7: np.ndarray) -> np.ndarray:
    """`[tx, ty, tz, qx, qy, qz, qw]` row → 4x4 cam-from-world. Same
    helper as the DROID / DPVO sessions; inlined here to keep this
    module self-contained."""
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


_REQUIRED_DRIVER_METHODS = ("process_frame", "track", "step")


def _resolve_mapper_cls():
    """Probe the upstream module for a streaming SLAM driver class.

    The upstream `muskie82/MonoGS` repository does NOT actually expose a
    per-frame streaming API. Its top-level entrypoint is `slam.SLAM`,
    which expects a config dict + a Dataset and runs frontend / backend
    as separate `multiprocessing.Process` instances driven internally —
    there's no `process_frame(idx, img, K)` method to call from a queue
    loop, and no public way to feed frames in one at a time without
    standing up the dataset + queue plumbing ourselves.

    This probe used to fall through to
    `gaussian_splatting.scene.gaussian_model.GaussianModel`, which is
    the splat data structure (it requires `sh_degree` at __init__ and
    has no SLAM driver methods) — so `start()` would crash with
    `TypeError: GaussianModel.__init__() missing 1 required positional
    argument: 'sh_degree'`. We now reject anything that doesn't
    actually look like a streaming driver, and raise a clear error
    instead — `select_session_cls()` then surfaces it as
    `MonogsSessionUnavailableError` and the strict-no-simulated path
    fires (live preview falls back to simulated; post-stop reconstruct
    fails loud).

    A real adapter would need to either:
      (a) build a frame-queue Dataset + drive the upstream
          frontend/backend process pair from our async loop, or
      (b) generate a TUM-shaped on-disk dataset from the captured
          frames + subprocess-run upstream `slam.py --config <yaml>`.

    Neither shortcut is in the tree yet; this resolver's job is to
    fail loudly until one lands rather than crashing mid-frame."""
    candidates = [
        ("gaussian_splatting.scene", "GaussianMapper"),
        ("monogs.gaussian_splatting.scene", "GaussianMapper"),
        ("monogs.scene", "GaussianMapper"),
        ("monogs.tracker", "MonogsTracker"),
    ]
    rejected: list[str] = []
    for module_path, attr in candidates:
        try:
            mod = __import__(module_path, fromlist=[attr])
            cls = getattr(mod, attr)
        except Exception:  # noqa: BLE001
            continue
        if not any(hasattr(cls, m) for m in _REQUIRED_DRIVER_METHODS):
            rejected.append(f"{module_path}.{attr} (no streaming driver methods)")
            continue
        return cls
    raise ImportError(
        "MonoGS streaming SLAM is not implemented in this build. Upstream "
        "muskie82/MonoGS exposes only a multi-process batch entrypoint "
        "(`slam.SLAM(config)`) — there is no `process_frame` / `step` API "
        "to drive from a frame queue. Pick a streaming SLAM backend "
        "(mast3r_slam / droid_slam / dpvo) for the capture pass; gsplat "
        "training can run as a follow-up job from the resulting frames. "
        f"(probed candidates: rejected={rejected})"
    )


def _field(obj: Any, name: str, default: Any) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


class MonogsCudaSession(SlamSession):
    """Real MonoGS CUDA tracker. Selected when the upstream package +
    CUDA are both available."""

    backend_id: ClassVar[str] = "monogs"

    def __init__(
        self,
        artifact_dir: Optional[Path] = None,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        self._tracker: Optional[Any] = None
        self._artifact_dir = artifact_dir
        self._checkpoint_path = checkpoint_path or _default_checkpoint_path()
        self._image_shape: Optional[tuple[int, int]] = None
        self._intrinsics: Optional[np.ndarray] = None
        self._n_frames = 0
        self._keyframe_indices: list[int] = []

    def set_artifact_dir(self, path: Path) -> None:
        """Mirror of `_MonogsSession.set_artifact_dir` so the processor
        can wire the output splat PLY destination after construction."""
        self._artifact_dir = path

    # ------------------------------------------------------------------
    # SlamSession surface
    # ------------------------------------------------------------------

    def start(
        self,
        intrinsics: np.ndarray,
        image_shape: tuple[int, int],
    ) -> None:
        cls = _resolve_mapper_cls()
        self._image_shape = image_shape
        self._intrinsics = intrinsics.astype(np.float32, copy=False)
        log.info(
            "monogs: starting CUDA tracker (image=%dx%d, ckpt=%s)",
            image_shape[1],
            image_shape[0],
            self._checkpoint_path or "<upstream HF default>",
        )
        kwargs: dict[str, Any] = {"device": "cuda"}
        if self._checkpoint_path is not None:
            kwargs["ckpt"] = self._checkpoint_path
        try:
            self._tracker = cls(**kwargs)
        except TypeError:
            # Some upstream variants take positional args.
            self._tracker = cls(self._checkpoint_path) if self._checkpoint_path else cls()

    def step(self, idx: int, img: np.ndarray) -> FrameUpdate:
        assert self._tracker is not None and self._intrinsics is not None
        try:
            # Upstream's per-frame method is typically `process_frame`
            # but some forks expose `track` or `__call__`. Try in
            # order; the first that works on this fork wins.
            if hasattr(self._tracker, "process_frame"):
                self._tracker.process_frame(idx, img, intrinsics=self._intrinsics)
            elif hasattr(self._tracker, "track"):
                self._tracker.track(idx, img, self._intrinsics)
            else:
                self._tracker(idx, img, self._intrinsics)
            self._n_frames += 1
        except Exception as exc:  # noqa: BLE001
            log.warning("monogs: step(idx=%d) raised: %s", idx, exc)
            return FrameUpdate()

        # MonoGS's keyframe selection is internal; mark every Nth frame
        # heuristically so the SlamProcessor's partial-snapshot cadence
        # has something to fire on.
        is_keyframe = (idx % 4 == 0)
        if is_keyframe:
            self._keyframe_indices.append(idx)
        return FrameUpdate(
            pose_matrix=None,  # surfaced at finalize()
            new_points=None,   # cloud built at finalize()
            is_keyframe=is_keyframe,
            diagnostics={"backend": "monogs_cuda"},
        )

    def finalize(self) -> FinalResult:
        if self._tracker is None:
            return FinalResult(
                poses=np.empty((0, 4, 4), dtype=np.float32),
                keyframe_indices=[],
                points=None,
                diagnostics={"backend": "monogs_cuda", "empty": True},
            )

        try:
            result = self._tracker.finalize()
        except Exception as exc:  # noqa: BLE001
            log.warning("monogs: finalize() raised: %s", exc)
            result = None

        # Trajectory.
        traj = _field(result, "poses", None) if result is not None else None
        if isinstance(traj, np.ndarray) and traj.ndim == 2 and traj.shape[1] >= 7:
            poses = np.stack(
                [_quat_xyzw_translate_to_matrix(row) for row in traj]
            )
        elif isinstance(traj, np.ndarray) and traj.ndim == 3 and traj.shape[1:] == (4, 4):
            poses = traj.astype(np.float32, copy=False)
        else:
            poses = np.empty((0, 4, 4), dtype=np.float32)

        # Cloud (sparse — MonoGS's main output is the splat).
        points = _field(result, "points", None) if result is not None else None
        if isinstance(points, np.ndarray) and points.ndim == 2 and points.shape[1] >= 6:
            points = points[:, :6].astype(np.float32, copy=False)
        else:
            points = None

        # Splat PLY. MonoGS may emit the file directly (preferred) or
        # return a state object we have to serialise ourselves.
        splat_path = _field(result, "splat_ply_path", None) if result is not None else None
        if splat_path is None and self._artifact_dir is not None:
            splat_path = self._maybe_export_splat(result)
        if splat_path is not None:
            splat_path = Path(splat_path)

        return FinalResult(
            poses=poses,
            keyframe_indices=list(self._keyframe_indices),
            points=points,
            splat_ply_path=splat_path,
            diagnostics={
                "backend": "monogs_cuda",
                "n_poses": int(poses.shape[0]),
                "n_keyframes": len(self._keyframe_indices),
                "n_tracked_frames": self._n_frames,
                "splat_source": "monogs_cuda",
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _maybe_export_splat(self, result: Any) -> Optional[Path]:
        """If the upstream finalize() didn't produce a PLY file but did
        return a serialisable scene object, write it ourselves. Best
        effort — covers the common case where MonoGS keeps the scene in
        memory and exposes a `save_ply(path)` method on the mapper."""
        if self._artifact_dir is None:
            return None
        out = self._artifact_dir / "splat_monogs.ply"
        try:
            if hasattr(self._tracker, "save_ply"):
                self._tracker.save_ply(str(out))
                if out.exists():
                    return out
            scene = _field(result, "scene", None)
            if scene is not None and hasattr(scene, "save_ply"):
                scene.save_ply(str(out))
                if out.exists():
                    return out
        except Exception as exc:  # noqa: BLE001
            log.warning("monogs: splat export raised: %s", exc)
        return None
