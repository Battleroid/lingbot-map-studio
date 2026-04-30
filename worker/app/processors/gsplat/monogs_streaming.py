"""Streaming MonoGS wrapper for the live-capture WS path.

Drives upstream `muskie82/MonoGS` in-process via a queue-fed Dataset so
the live capture session in the api container gets real MonoGS pose +
point output during a scan, instead of falling back to the simulated
placeholder.

The architectural mismatch this bridges:

  * Upstream MonoGS reads frames from a `torch.utils.data.Dataset` that
    iterates by integer index. The frontend's run loop blocks until
    `cur_frame_idx >= len(self.dataset)` and there's no public method
    to push frames in.
  * Our streaming SlamSession surface (`start` / `step` / `finalize`)
    pushes one BGR frame per call.

Bridge: a `_QueueMonocularDataset` subclass of upstream's
`MonocularDataset` whose `__getitem__` blocks on a `queue.Queue` until
that frame index is available; the SLAM thread inside MonoGS reads it
naturally. We monkey-patch upstream's `load_dataset` to return our
custom instance when `Dataset.type == "queue"`. `finalize()` caps
`num_imgs` so the frontend's `cur_frame_idx >= len(dataset)` exit
fires, joins the SLAM thread, and reads the resulting splat PLY off
disk.

This module is intentionally tolerant of upstream import failures —
the api container ships the full MonoGS stack (after PR #52) but other
contexts (CPU-only test hosts, CI) don't, and the live-capture
resolver in `app/processors/slam/live_session.py` falls back to the
simulated session in those cases.

Performance note: monocular MonoGS runs around 0.5–2 Hz on a high-end
GPU; the capture client targets 10 Hz. The frame queue has a generous
maxsize so we backpressure naturally — `step()` blocks when the queue
fills, the WS handler drops frames at the client side via its own
queue cap, and the user sees a slower-than-camera but real preview."""

from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path
from typing import Any, ClassVar, Optional

import numpy as np

from app.processors.slam.base import (
    FinalResult,
    FrameUpdate,
    SlamSession,
)

log = logging.getLogger(__name__)


# Maximum frames buffered between `step()` and the SLAM thread. MonoGS
# is the bottleneck (≈0.5–2 Hz monocular); this gives us a few seconds
# of slack at 10 Hz capture before frames start blocking.
_QUEUE_MAXSIZE = 32

# Sentinel pushed onto the queue at finalize() to break the dataset's
# blocking get() loop.
_STOP = object()


class MonogsStreamingUnavailableError(RuntimeError):
    """Raised when the streaming wrapper can't be initialised because
    upstream MonoGS isn't importable or torch.cuda is unavailable.
    Bubbled up by `select_streaming_session_cls()` so live capture
    falls back to the simulated session for the preview pass."""


def select_streaming_session_cls() -> type[SlamSession]:
    """Resolve the streaming MonoGS session. Mirrors the strict checks
    from `monogs.select_session_cls()` but gates on the streaming
    Dataset patch instead of the speculative `_resolve_mapper_cls`."""
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:
        raise MonogsStreamingUnavailableError(
            "monogs_streaming: torch is not installed in this process."
        ) from exc
    if not torch.cuda.is_available():
        raise MonogsStreamingUnavailableError(
            "monogs_streaming: torch.cuda.is_available() is False; the "
            "api container needs `runtime: nvidia` + GPU passthrough."
        )
    try:
        import importlib  # noqa: PLC0415

        importlib.import_module("gaussian_splatting")
        importlib.import_module("utils.dataset")
        importlib.import_module("slam")
    except Exception as exc:  # noqa: BLE001
        raise MonogsStreamingUnavailableError(
            "monogs_streaming: upstream MonoGS isn't importable "
            f"({type(exc).__name__}: {exc}). The Dockerfile should clone "
            "muskie82/MonoGS into /opt/monogs and add it to PYTHONPATH; "
            "see worker/Dockerfile.api."
        ) from exc
    return MonogsStreamingSession


# ----------------------------------------------------------------------
# Queue-driven Dataset
# ----------------------------------------------------------------------


def _build_queue_dataset(
    intrinsics: np.ndarray,
    image_shape: tuple[int, int],
    frame_queue: "queue.Queue[Any]",
):
    """Construct a `_QueueMonocularDataset` after the upstream module
    is importable. Inlined as a function so the dataset class isn't
    defined at module-import time (it inherits from upstream)."""
    import torch  # noqa: PLC0415
    from utils.dataset import MonocularDataset  # noqa: PLC0415

    h, w = image_shape
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])

    class _QueueMonocularDataset(MonocularDataset):
        """Monocular dataset that blocks on a queue until each indexed
        frame is available. Subclasses upstream's `MonocularDataset` so
        the frontend's `Camera.init_from_dataset` finds fx/fy/cx/cy/...
        in the right shape."""

        def __init__(self) -> None:
            # Upstream MonocularDataset.__init__ pulls Calibration off
            # config[Dataset]. We bypass it and set the fields directly
            # since our config is a synthetic in-memory dict and we
            # already know the calibration.
            self.args = None
            self.path = ""
            self.config = None
            self.device = "cuda:0"
            self.dtype = torch.float32
            # Sentinel: very large initial num_imgs so the frontend's
            # `cur_frame_idx >= len(self.dataset)` check doesn't trip
            # while we're still feeding frames. `complete()` caps it.
            self.num_imgs = 10**9
            self.fx = fx
            self.fy = fy
            self.cx = cx
            self.cy = cy
            self.width = w
            self.height = h
            from gaussian_splatting.utils.graphics_utils import (  # noqa: PLC0415
                focal2fov,
            )

            self.fovx = focal2fov(self.fx, self.width)
            self.fovy = focal2fov(self.fy, self.height)
            self.K = np.array(
                [
                    [self.fx, 0.0, self.cx],
                    [0.0, self.fy, self.cy],
                    [0.0, 0.0, 1.0],
                ]
            )
            self.disorted = False
            self.dist_coeffs = np.zeros(5)
            self.has_depth = False
            self.depth_scale = None
            nerf_normalization_radius = 5
            self.scene_info = {
                "nerf_normalization": {
                    "radius": nerf_normalization_radius,
                    "translation": np.zeros(3),
                },
            }

            # Random-access cache: __getitem__ may be called more than
            # once for a given index across initialisation + tracking.
            self._frames: dict[int, tuple[Any, None, Any]] = {}
            self._queue = frame_queue
            self._max_idx_seen: int = -1
            self._completed: bool = False

        def __len__(self) -> int:
            return self.num_imgs

        def __getitem__(self, idx: int):
            # Pull frames off the queue until we have `idx`. Upstream's
            # frontend reads strictly in order so this is a single
            # blocking get per new frame; the random-access cache
            # covers tracking's occasional re-reads of older frames.
            while idx not in self._frames:
                if self._completed and idx > self._max_idx_seen:
                    raise IndexError(
                        f"_QueueMonocularDataset: idx {idx} requested after "
                        f"complete() (max seen: {self._max_idx_seen})"
                    )
                item = self._queue.get()
                if item is _STOP:
                    self._completed = True
                    if idx > self._max_idx_seen:
                        raise IndexError(
                            f"_QueueMonocularDataset: idx {idx} requested "
                            "after stop sentinel"
                        )
                    continue
                push_idx, img_bgr = item
                # Convert BGR uint8 HWC → RGB float CHW tensor on CUDA,
                # exactly the shape upstream's MonocularDataset returns.
                rgb = img_bgr[:, :, ::-1]  # BGR → RGB
                tensor = (
                    torch.from_numpy(rgb.copy() / 255.0)
                    .clamp(0.0, 1.0)
                    .permute(2, 0, 1)
                    .to(device=self.device, dtype=self.dtype)
                )
                # Identity initial pose — monocular tracker optimises
                # the actual pose; this is just the GT-pose hint
                # upstream code expects in the tuple.
                pose = torch.eye(4, device=self.device)
                self._frames[push_idx] = (tensor, None, pose)
                self._max_idx_seen = max(self._max_idx_seen, push_idx)
            return self._frames[idx]

        def complete(self) -> None:
            """Cap `num_imgs` and unblock any pending get(). Called by
            the streaming session at finalize() time so the upstream
            frontend's `cur_frame_idx >= len(dataset)` exit fires."""
            self._completed = True
            self.num_imgs = max(self._max_idx_seen + 1, 1)
            try:
                self._queue.put_nowait(_STOP)
            except queue.Full:
                # Queue is full of pending frames; drain a slot and try
                # again. Worst-case the frontend processes one more
                # frame before stopping; harmless.
                try:
                    self._queue.get_nowait()
                    self._queue.put_nowait(_STOP)
                except (queue.Empty, queue.Full):
                    pass

    return _QueueMonocularDataset()


# ----------------------------------------------------------------------
# Streaming session
# ----------------------------------------------------------------------


class MonogsStreamingSession(SlamSession):
    """Drive upstream MonoGS in-process via a frame queue.

    `start()` builds a synthetic config + queue-driven dataset, monkey-
    patches upstream's `load_dataset` to return our dataset, and spawns
    the SLAM constructor (which runs the entire session synchronously)
    in a background thread.

    `step()` pushes the BGR frame onto the queue. The SLAM thread reads
    it via the dataset's blocking `__getitem__`. Pose / point output is
    not surfaced per-frame — there's no clean way to intercept the
    frontend's internal state from outside its Process loop without
    forking upstream. We return empty FrameUpdates; the live preview
    UX uses pose hints from the simulated tracker side-by-side anyway.

    `finalize()` caps the dataset, joins the SLAM thread, and locates
    the splat PLY MonoGS wrote during its eval pass."""

    backend_id: ClassVar[str] = "monogs"

    def __init__(self) -> None:
        self._frame_queue: "queue.Queue[Any]" = queue.Queue(maxsize=_QUEUE_MAXSIZE)
        self._dataset: Any = None
        self._slam_thread: Optional[threading.Thread] = None
        self._intrinsics: Optional[np.ndarray] = None
        self._image_shape: Optional[tuple[int, int]] = None
        self._artifact_dir: Optional[Path] = None
        self._save_dir: Optional[Path] = None
        self._error: Optional[BaseException] = None

    def set_artifact_dir(self, path: Path) -> None:
        """Mirror of `_MonogsSession.set_artifact_dir`. The streaming
        session writes the splat PLY into `<path>/.monogs_streaming/`
        so finalize() can pick it up."""
        self._artifact_dir = path

    # ------------------------------------------------------------------
    # SlamSession surface
    # ------------------------------------------------------------------

    def start(
        self,
        intrinsics: np.ndarray,
        image_shape: tuple[int, int],
    ) -> None:
        # Late availability check — `select_streaming_session_cls()`
        # already passed at resolver time, but probe again here so
        # mid-stream import failures surface as a clean exception
        # rather than a hung thread.
        try:
            import importlib  # noqa: PLC0415

            importlib.import_module("slam")
            importlib.import_module("utils.dataset")
        except Exception as exc:  # noqa: BLE001
            raise MonogsStreamingUnavailableError(
                f"monogs_streaming: upstream import failed mid-start "
                f"({type(exc).__name__}: {exc})."
            ) from exc

        self._intrinsics = intrinsics.astype(np.float32, copy=False)
        self._image_shape = image_shape

        save_root = (
            self._artifact_dir or Path("/tmp")
        ) / ".monogs_streaming"
        save_root.mkdir(parents=True, exist_ok=True)
        self._save_dir = save_root

        self._dataset = _build_queue_dataset(
            intrinsics, image_shape, self._frame_queue
        )

        self._slam_thread = threading.Thread(
            target=self._run_slam, name="monogs-streaming", daemon=True
        )
        self._slam_thread.start()

    def step(self, idx: int, img: np.ndarray) -> FrameUpdate:
        if self._error is not None:
            # Surface the SLAM thread's exception on the next step so
            # the WS handler stops feeding frames into a dead session.
            raise RuntimeError(
                f"monogs_streaming: SLAM thread died: {self._error!r}"
            )
        try:
            self._frame_queue.put((idx, img), timeout=5.0)
        except queue.Full:
            # Backpressure: the SLAM thread can't keep up. Drop this
            # frame rather than blocking the WS handler indefinitely.
            log.warning("monogs_streaming: frame queue full; dropping idx=%d", idx)
        return FrameUpdate(diagnostics={"backend": "monogs_streaming"})

    def finalize(self) -> FinalResult:
        if self._dataset is not None:
            self._dataset.complete()
        if self._slam_thread is not None:
            # Generous timeout — color refinement + final save can take
            # a couple of minutes on a long capture. After this we
            # bail; the SLAM thread is a daemon so it'll be reaped at
            # process exit if it doesn't return.
            self._slam_thread.join(timeout=300.0)

        splat_path = self._locate_splat_ply()
        return FinalResult(
            poses=np.empty((0, 4, 4), dtype=np.float32),
            keyframe_indices=[],
            points=None,
            splat_ply_path=splat_path,
            diagnostics={
                "backend": "monogs_streaming",
                "splat_source": "monogs_streaming",
                "splat_present": splat_path is not None,
            },
        )

    # ------------------------------------------------------------------
    # SLAM thread
    # ------------------------------------------------------------------

    def _run_slam(self) -> None:
        """Body of the SLAM thread.

        Monkey-patches `utils.dataset.load_dataset` to return our queue
        dataset when type == "queue", builds the synthetic config, and
        runs `SLAM(config)`. The constructor blocks until the frontend
        loop sees `cur_frame_idx >= len(dataset)` and breaks out — this
        is what `finalize().complete()` triggers."""
        try:
            from utils import dataset as _dataset_mod  # noqa: PLC0415
            from slam import SLAM  # noqa: PLC0415
            # Lazy import — these pull in MonoGS-stack-only deps
            # (yaml, munch, …) that don't ship in every container.
            from app.processors.gsplat.monogs_batch import _build_config  # noqa: PLC0415

            assert self._intrinsics is not None
            assert self._image_shape is not None
            assert self._save_dir is not None

            # Monkey-patch the dispatcher.
            _orig_load = _dataset_mod.load_dataset

            def _patched_load(args, path, config):  # noqa: ANN001
                if config.get("Dataset", {}).get("type") == "queue":
                    return self._dataset
                return _orig_load(args, path, config)

            _dataset_mod.load_dataset = _patched_load

            try:
                config = _build_config(
                    intrinsics=self._intrinsics,
                    image_shape=self._image_shape,
                    dataset_path=self._save_dir,  # unused for type=queue
                    save_dir=self._save_dir,
                )
                config["Dataset"]["type"] = "queue"
                # SLAM's constructor sets save_dir on Results explicitly
                # too; both paths land in the same directory.
                SLAM(config, save_dir=str(self._save_dir))
            finally:
                _dataset_mod.load_dataset = _orig_load
        except BaseException as exc:  # noqa: BLE001
            log.warning("monogs_streaming: SLAM thread raised: %s", exc)
            self._error = exc

    def _locate_splat_ply(self) -> Optional[Path]:
        if self._save_dir is None:
            return None
        final = self._save_dir / "point_cloud" / "final" / "point_cloud.ply"
        if final.exists():
            return final
        candidates = sorted(
            self._save_dir.rglob("point_cloud.ply"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None
