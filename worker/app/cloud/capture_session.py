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
        self._dropped = 0
        self._last_activity = time.monotonic()
        # Buffered for finalize().
        self._poses: list[np.ndarray] = []
        self._points_buffer: list[np.ndarray] = []
        self._intrinsics: Optional[np.ndarray] = None
        self._image_shape: Optional[tuple[int, int]] = None

    async def start(self) -> None:
        """Pick the SLAM backend + spawn the frame-processing task."""
        from app.processors.slam.live_session import resolve_live_session

        self._slam_session = resolve_live_session(self.backend)
        await self._emit("ready", {"backend": self.backend})
        self._task = asyncio.create_task(self._run())

    async def push_frame(self, idx: int, img_bgr: np.ndarray) -> None:
        """Called by the WS handler for each binary message received.
        Drops the frame if the queue is full — the SLAM step rate is
        the bottleneck and queueing more would just stale the cloud."""
        if self._stopped:
            return
        self._last_activity = time.monotonic()
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

                if update.pose_matrix is not None:
                    self._poses.append(update.pose_matrix)
                    await self._emit_pose(update.pose_matrix, pkt.idx)
                if update.new_points is not None and update.new_points.size > 0:
                    capped = update.new_points[:200]  # bandwidth cap
                    self._points_buffer.append(update.new_points)
                    await self._emit_points(capped, pkt.idx)

                # Throttle stats events to once per second so the chip
                # in the UI doesn't redraw on every frame.
                if self._frame_count % 10 == 0:
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
        """Run the SLAM session's finalize + persist a regular Job."""
        if self._slam_session is None:
            return FinalizeResult(job_id=None, ok=False, error="never started")

        try:
            result = await asyncio.to_thread(self._slam_session.finalize)
        except Exception as exc:  # noqa: BLE001
            log.warning("capture_session %s finalize raised: %s", self.id, exc)
            return FinalizeResult(job_id=None, ok=False, error=str(exc))

        # Promote into a regular Job so the user can hop to the job
        # page + chain gsplat training off the captured run. Reuses
        # the existing artifact write helpers from slam.export.
        from app.jobs import store
        from app.jobs.schema import (
            Artifact,
            Job,
            JobEvent,
            Mast3rSlamConfig,
        )
        from app.jobs.events import bus
        from app.processors.slam import export as slam_export

        await store.init_store()
        job_id = "cap" + uuid.uuid4().hex[:10]
        job_dir = settings.job_dir(job_id)
        artifacts_dir = job_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        intrinsics = (
            self._intrinsics
            if self._intrinsics is not None
            else np.eye(3, dtype=np.float32)
        )
        slam_export.write_pose_graph(
            artifacts_dir / "pose_graph.json",
            poses=result.poses,
            keyframe_indices=result.keyframe_indices,
            selected_indices=result.keyframe_indices,
            intrinsics=intrinsics,
            backend_id=self.backend,
        )
        slam_export.write_camera_path(
            artifacts_dir / "camera_path.json", list(result.poses)
        )
        if result.points is not None and result.points.size > 0:
            slam_export.write_ply(
                artifacts_dir / "reconstruction.ply", result.points
            )

        artifacts: list[Artifact] = []
        for name in ("pose_graph.json", "camera_path.json", "reconstruction.ply"):
            p = artifacts_dir / name
            if p.exists():
                artifacts.append(Artifact(name=name, kind="ply" if name.endswith(".ply") else "json"))

        # The captured config is shaped like a MASt3R-SLAM run —
        # picking it as the default lets the JobList / viewer code
        # treat the row identically to a batch SLAM run. The actual
        # backend used at capture time is recorded in extras.
        cfg = Mast3rSlamConfig()
        job = Job(
            id=job_id,
            status="ready",
            config=cfg,
            uploads=[],
            artifacts=artifacts,
            frames_total=self._frame_count,
        )
        await store.create_job(job, worker_class="slam")
        await store.update_job(job_id, status="ready")

        await bus.publish(
            JobEvent(
                job_id=job_id,
                stage="system",
                level="info",
                message=f"capture session {self.id} → job {job_id} ({self._frame_count} frames)",
                data={"backend": self.backend, "captured": True},
            )
        )
        await bus.close(job_id)
        return FinalizeResult(job_id=job_id, ok=True)


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
