"""Live camera-capture session.

A capture session is a streaming counterpart to the existing batch
SLAM job: instead of "upload a file → SLAM all frames at once", the
client opens a WebSocket and pushes JPEG-encoded frames as the user
scans, receiving per-frame poses + new sparse points back over the
same socket.

Lifecycle:

    POST /api/capture/start  → CaptureManager.create(...)
                                ↳ allocates session_id
                                ↳ spawns a CaptureSession in the
                                  background that owns a SlamSession +
                                  a frame queue.
    WS /api/capture/{id}     → frames in (binary), pose+points out
                                (JSON). The websocket binds to the
                                session's queue + emit channel.
    POST /api/capture/{id}/stop → CaptureSession.finalize() runs the
                                  SLAM finalize + writes pose_graph +
                                  reconstruction.ply to disk + creates
                                  a regular Job row. The user is
                                  redirected to /jobs/{id} where the
                                  output looks identical to a batch
                                  SLAM run.

Why this lives in `app/cloud/` rather than `app/jobs/`:
the DB-backed Job lifecycle assumes "claim → run → finalize" inside a
worker container. A capture session is interactive — the WS handler
in the api container drives it directly, with the GPU-bound SLAM
session living on the same process for v1. (Multi-worker dispatch is
a v2 concern; doc note on line 0 of the plan.)
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import numpy as np

from app.config import settings

log = logging.getLogger(__name__)


# Bound the session lifetime so a forgotten browser tab doesn't pin a
# GPU forever. Refreshed every time the client sends a frame.
_IDLE_TIMEOUT_S = 60.0



@dataclass
class FramePacket:
    """One client → server frame, decoded into a BGR uint8 image."""
    idx: int
    img_bgr: np.ndarray
    received_at: float


@dataclass
class EmitMessage:
    """One server → client message. Serialised to JSON by the WS
    handler; using a dataclass instead of raw dict so the producer
    side has type-checked field names."""
    type: str  # "pose" | "points" | "stats" | "error" | "ready"
    data: dict = field(default_factory=dict)


class CaptureSession:
    """Owns one streaming capture: backend SLAM session + frame queue
    + emit channel. Driven by the WS handler in api/main.py — the WS
    pushes onto `frame_queue` and reads from `emit_queue`.

    Frame processing runs in a background asyncio task spawned by
    `start()`. The task pulls frames off the queue, runs them through
    `slam_session.step(idx, img)`, and pushes pose + new-points
    messages onto `emit_queue` for the WS to forward.
    """

    def __init__(self, session_id: str, backend: str) -> None:
        self.id = session_id
        self.backend = backend
        self.frame_queue: asyncio.Queue[FramePacket] = asyncio.Queue(maxsize=4)
        self.emit_queue: asyncio.Queue[EmitMessage] = asyncio.Queue()
        self._slam_session = None  # filled in start()
        self._task: Optional[asyncio.Task] = None
        self._stopped = False
        self._frame_count = 0
        self._frames_persisted = 0
        self._dropped = 0
        self._last_activity = time.monotonic()
        # Buffered for finalize().
        self._poses: list[np.ndarray] = []
        self._points_buffer: list[np.ndarray] = []
        self._intrinsics: Optional[np.ndarray] = None
        self._image_shape: Optional[tuple[int, int]] = None
        # Pre-allocate the future Job's id + on-disk layout so we can
        # stream the JPEG payloads straight to `<job_dir>/frames/`
        # while the user is scanning. The capture's stop step then
        # only needs to flip the Job row from non-existent to
        # `queued` — no copy/move is required because the bytes
        # already live in the canonical place.
        #
        # Why this matters: the simulated tracker that runs in the
        # api process is a placeholder, not a real reconstruction.
        # The real backends (mast3r-slam, monogs, etc.) live in the
        # worker-slam / worker-gs containers. Persisting the raw
        # frames lets the regular claim loop pick the captured job
        # up after stop, run the real GPU backend on it, and produce
        # output that actually represents what the user scanned.
        self.job_id: str = "cap" + uuid.uuid4().hex[:10]
        self.job_dir: Path = settings.job_dir(self.job_id)
        self.frames_dir: Path = self.job_dir / "frames"
        self.artifacts_dir: Path = self.job_dir / "artifacts"

    async def start(self) -> None:
        """Pick the SLAM backend + spawn the frame-processing task."""
        from app.processors.slam.live_session import resolve_live_session

        self._slam_session = resolve_live_session(self.backend)
        # Pre-create the on-disk layout. `frames/` will be filled
        # frame-by-frame as the WS receives them; the worker that
        # eventually claims this job reads from there.
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        await self._emit("ready", {"backend": self.backend})
        self._task = asyncio.create_task(self._run())

    async def push_frame(
        self,
        idx: int,
        img_bgr: np.ndarray,
        raw_bytes: Optional[bytes] = None,
    ) -> None:
        """Called by the WS handler for each binary message received.
        Drops the frame if the queue is full — the SLAM step rate is
        the bottleneck and queueing more would just stale the cloud.

        `raw_bytes` is the original JPEG payload from the phone; if
        present, it gets persisted to `frames_dir` under a session-
        local monotonic counter. Persisting these frames is what
        lets the queued post-stop job actually get reconstructed by
        the real GPU backend in worker-slam / worker-gs (the
        simulated tracker that runs in this api process produces
        placeholder output that doesn't represent the scene)."""
        if self._stopped:
            return
        self._last_activity = time.monotonic()
        if raw_bytes is not None:
            # File numbering is session-local and monotonic so reconnects
            # don't overwrite earlier frames (the WS handler's frame_idx
            # restarts at 0 on each new socket). Write straight through —
            # JPEG is the wire format, no need to re-encode.
            counter = self._frames_persisted
            self._frames_persisted += 1
            target = self.frames_dir / f"{counter:06d}.jpg"
            try:
                target.write_bytes(raw_bytes)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "capture_session %s: failed to persist frame %d: %s",
                    self.id,
                    counter,
                    exc,
                )
                self._frames_persisted -= 1
        try:
            self.frame_queue.put_nowait(
                FramePacket(idx=idx, img_bgr=img_bgr, received_at=time.monotonic())
            )
        except asyncio.QueueFull:
            self._dropped += 1

    async def stop(self) -> "FinalizeResult":
        """Drain the queue, finalise the SLAM session, and write the
        result artifacts to disk so the session can be promoted into a
        regular Job row."""
        self._stopped = True
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=2.0)
            except asyncio.TimeoutError:
                self._task.cancel()
        return await self._finalize_to_disk()

    async def _run(self) -> None:
        """Frame-processing loop. One frame at a time — the SLAM step
        is sequential anyway."""
        from app.processors.slam.base import SlamSession
        slam: SlamSession = self._slam_session  # type: ignore[assignment]

        try:
            while not self._stopped:
                try:
                    pkt = await asyncio.wait_for(
                        self.frame_queue.get(), timeout=0.5
                    )
                except asyncio.TimeoutError:
                    if time.monotonic() - self._last_activity > _IDLE_TIMEOUT_S:
                        await self._emit("error", {"message": "session idle timeout"})
                        return
                    continue

                if self._image_shape is None:
                    h, w = pkt.img_bgr.shape[:2]
                    self._image_shape = (h, w)
                    # Calibration-free default — most browsers don't
                    # surface focal length. We pass a 60° HFOV K
                    # matrix; SLAM backends like MASt3R-SLAM ignore
                    # it, others use it as a soft prior.
                    fx = w * 0.866
                    self._intrinsics = np.array(
                        [[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]],
                        dtype=np.float32,
                    )
                    await asyncio.to_thread(slam.start, self._intrinsics, self._image_shape)

                update = await asyncio.to_thread(slam.step, pkt.idx, pkt.img_bgr)
                self._frame_count += 1

                # Surface the first decoded frame loudly so the client
                # can flip from "starting…" to "capturing" immediately.
                # Without this the user has to wait for either the
                # first SLAM keyframe (pose/points emit) or the 10-
                # frame stats throttle to fire before any UI feedback
                # appears, which on a slow phone-to-laptop network
                # looks like the page is broken.
                if self._frame_count == 1:
                    log.info(
                        "capture_session %s: first frame %dx%d",
                        self.id,
                        pkt.img_bgr.shape[1],
                        pkt.img_bgr.shape[0],
                    )
                    await self._emit_stats()

                if update.pose_matrix is not None:
                    self._poses.append(update.pose_matrix)
                    await self._emit_pose(update.pose_matrix, pkt.idx)
                if update.new_points is not None and update.new_points.size > 0:
                    capped = update.new_points[:200]  # bandwidth cap
                    self._points_buffer.append(update.new_points)
                    await self._emit_points(capped, pkt.idx)

                # First five frames get a stats event apiece so the
                # user sees activity right away; after that throttle
                # to one per ten frames (≈1 s at 10 Hz capture) so we
                # don't redraw the chip on every step.
                if self._frame_count <= 5 or self._frame_count % 10 == 0:
                    await self._emit_stats()
        except Exception as exc:  # noqa: BLE001
            log.warning("capture_session %s loop raised: %s", self.id, exc)
            await self._emit("error", {"message": str(exc)})

    async def _emit(self, msg_type: str, data: dict) -> None:
        await self.emit_queue.put(EmitMessage(type=msg_type, data=data))

    async def _emit_pose(self, pose_matrix: np.ndarray, idx: int) -> None:
        # Pose is a 4x4 cam-from-world; emit the translation + a
        # quaternion in xyzw so the client's renderer can reuse the
        # CameraPath conventions. Inlined matrix-to-quat math (no
        # scipy dep — capture_session.py is imported by the api
        # container which doesn't ship scipy).
        try:
            t = pose_matrix[:3, 3].tolist()
            q = _matrix_to_quat_xyzw(pose_matrix[:3, :3])
        except Exception:  # noqa: BLE001
            return
        await self._emit("pose", {"frame": idx, "t": t, "q": q})

    async def _emit_points(self, pts: np.ndarray, idx: int) -> None:
        # Points are (N, 6) — xyz + rgb. Round to mm + uint8 colour
        # to keep the JSON payload tight.
        rows = []
        for row in pts:
            x, y, z = float(row[0]), float(row[1]), float(row[2])
            r = int(max(0, min(255, row[3] if len(row) > 3 else 200)))
            g = int(max(0, min(255, row[4] if len(row) > 4 else 200)))
            b = int(max(0, min(255, row[5] if len(row) > 5 else 200)))
            rows.append([round(x, 4), round(y, 4), round(z, 4), r, g, b])
        await self._emit("points", {"frame": idx, "new": rows})

    async def _emit_stats(self) -> None:
        await self._emit(
            "stats",
            {
                "frames": self._frame_count,
                "queued": self.frame_queue.qsize(),
                "dropped": self._dropped,
            },
        )

    async def _finalize_to_disk(self) -> "FinalizeResult":
        """Promote the captured frames into a regular `queued` Job and
        let the worker tier run real SLAM/MonoGS on them.

        The previous implementation called `simulated_session.finalize()`
        and wrote its synthetic poses + corner-feature point cloud into
        the job's artifacts as if it were a real reconstruction, then
        marked the Job `ready`. Result: the job page showed a splat
        cloud that bore no resemblance to what the user had scanned
        (poses drifted, points sat at fake depths). And gsplat-from-
        source bounced the job with `409 ... no extracted frames` since
        no images had been persisted.

        The fix: persist the JPEG payloads to `frames_dir` during the
        WS session (already wired in `push_frame`), then on stop create
        a Job in `queued` state with no uploads. The standard worker
        claim loop picks it up, runs the real backend (mast3r-slam,
        droid-slam, monogs, …) on the saved frames, and writes
        artifacts that actually represent the scene. The simulated
        in-process tracker still emits live preview events during the
        scan but its outputs are now discarded at stop time.

        Edge case: if zero frames made it through (decode failure, WS
        bounced before any data, etc.) we surface a fail rather than
        creating a job that's guaranteed to error with "no frames to
        track" the moment a worker claims it."""
        if self._frames_persisted == 0:
            return FinalizeResult(
                job_id=None,
                ok=False,
                error=(
                    "no frames captured — check the WS connection / camera "
                    "permission and try again"
                ),
            )

        from app.jobs import store
        from app.jobs.schema import (
            DpvoConfig,
            DroidSlamConfig,
            Job,
            JobEvent,
            Mast3rSlamConfig,
            MonogsConfig,
        )
        from app.jobs.events import bus
        from app.processors import worker_class_for

        await store.init_store()

        # Map the dropdown's backend choice onto the matching job
        # config class. Each one routes to the right worker + the
        # right processor when the claim loop picks it up.
        cfg_cls: dict[str, type] = {
            "mast3r_slam": Mast3rSlamConfig,
            "droid_slam": DroidSlamConfig,
            "dpvo": DpvoConfig,
            "monogs": MonogsConfig,
        }
        cfg = cfg_cls.get(self.backend, Mast3rSlamConfig)()
        worker_class = worker_class_for(cfg)
        job = Job(
            id=self.job_id,
            status="queued",
            config=cfg,
            uploads=[],
            artifacts=[],
            frames_total=self._frames_persisted,
        )
        await store.create_job(job, worker_class=worker_class)

        # Best-effort: mark a marker file so the worker's ingest step
        # can tell "frames already extracted (capture path)" apart from
        # "frames missing (data corruption)". The
        # SlamProcessor._ingest short-circuit reads this.
        marker = self.frames_dir / ".captured"
        try:
            marker.write_text(
                f"captured at {time.time()} via backend={self.backend}\n",
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001
            pass

        await bus.publish(
            JobEvent(
                job_id=self.job_id,
                stage="system",
                level="info",
                message=(
                    f"capture session {self.id} → job {self.job_id} "
                    f"({self._frames_persisted} frames queued for "
                    f"reconstruction via {cfg.processor})"
                ),
                data={
                    "backend": self.backend,
                    "captured": True,
                    "frames_total": self._frames_persisted,
                    "processor": cfg.processor,
                    "worker_class": worker_class,
                },
            )
        )
        # Don't close the bus — the worker's events still need to flow
        # through it once the job claim happens.
        return FinalizeResult(job_id=self.job_id, ok=True)


def _matrix_to_quat_xyzw(R: np.ndarray) -> list[float]:
    """3x3 rotation matrix → quaternion in [x, y, z, w] order.

    Standard Shepperd / max-trace switch — picks the most stable of
    four formulations based on which component is largest. Avoids
    pulling scipy into the api container; the only consumers of this
    are the WS pose emit and a unit test."""
    m = np.asarray(R, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return [float(x), float(y), float(z), float(w)]


@dataclass
class FinalizeResult:
    job_id: Optional[str]
    ok: bool
    error: Optional[str] = None


class CaptureManager:
    """Process-wide registry of active capture sessions. Single-user
    in v1; the api process mounts one of these and the WS handler +
    REST endpoints look sessions up by id.

    Multi-process dispatch (a real worker pool, broker integration) is
    out of scope for v1 and tracked in the plan's risk section."""

    def __init__(self) -> None:
        self._sessions: dict[str, CaptureSession] = {}

    async def create(self, backend: str = "mast3r_slam") -> CaptureSession:
        sid = "cs" + uuid.uuid4().hex[:10]
        session = CaptureSession(session_id=sid, backend=backend)
        self._sessions[sid] = session
        await session.start()
        return session

    def get(self, session_id: str) -> Optional[CaptureSession]:
        return self._sessions.get(session_id)

    async def stop(self, session_id: str) -> Optional[FinalizeResult]:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return None
        return await session.stop()


# Process-wide singleton. Imported by api/main.py.
manager = CaptureManager()
