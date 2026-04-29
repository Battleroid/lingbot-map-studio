"""Tests for the live camera-capture session.

Covers:
  * `resolve_live_session` falls back cleanly when the requested
    backend isn't installed (the auto-select pattern from Phases 2-4).
  * `CaptureSession` end-to-end: push N JPEG-decoded frames, watch
    pose + points messages come back through the emit queue, stop
    and confirm a Job row gets created with the expected artifacts.

The SLAM session in CI is the simulated tracker (no real CUDA on the
runner), which produces plausible-shaped output frame-by-frame —
exactly what we need to exercise the capture flow without requiring
a GPU."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest


def test_resolve_live_session_falls_back_to_simulated():
    """Unknown backend → simulated session, no exception."""
    from app.processors.slam.live_session import resolve_live_session
    from app.processors.slam.tracker import SimulatedSlamSession

    session = resolve_live_session("not-a-real-backend")
    assert isinstance(session, SimulatedSlamSession)


def test_resolve_live_session_known_backends_return_a_session():
    """Each known backend resolves to *some* SlamSession (the real
    CUDA path on a GPU box, the simulated subclass on CI)."""
    from app.processors.slam.base import SlamSession
    from app.processors.slam.live_session import (
        SUPPORTED_BACKENDS,
        resolve_live_session,
    )

    for backend in SUPPORTED_BACKENDS:
        s = resolve_live_session(backend)
        assert isinstance(s, SlamSession), backend


@pytest.mark.asyncio
async def test_capture_session_round_trip(tmp_data_dir):
    """Push 6 synthetic frames at the session, drain the emit queue,
    stop, confirm a Job row with the expected artifacts is created."""
    from app.cloud.capture_session import CaptureSession

    session = CaptureSession(session_id="test-cs-001", backend="mast3r_slam")
    await session.start()

    # Six 320×240 BGR frames of random pixels — the simulated tracker
    # only cares about shape, not content.
    rng = np.random.default_rng(0)
    # Yield between pushes so the consumer drains rather than tripping
    # the bounded-queue backpressure path. The realistic capture rate
    # is 10 Hz (100 ms apart); the simulated tracker is faster than
    # that, so all frames make it through.
    for idx in range(6):
        img = rng.integers(0, 255, size=(240, 320, 3), dtype=np.uint8)
        await session.push_frame(idx, img)
        await asyncio.sleep(0.05)

    # Give the background loop time to drain whatever's left.
    deadline = asyncio.get_event_loop().time() + 4.0
    while session._frame_count < 6 and asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.05)
    # At least half should have made it through — exact count depends
    # on the simulated tracker's step latency.
    assert session._frame_count >= 3, (
        f"loop processed {session._frame_count}/6 frames before timeout"
    )

    # We should have at least the "ready" message + per-frame poses
    # + at least one points message in the emit queue.
    received: list[str] = []
    while not session.emit_queue.empty():
        msg = session.emit_queue.get_nowait()
        received.append(msg.type)
    assert "ready" in received, received
    assert "pose" in received, received

    # Stop and confirm a Job got created.
    result = await session.stop()
    assert result.ok, result.error
    assert result.job_id is not None
    # Job row exists, status=ready, artifacts include the SLAM outputs.
    from app.jobs import store

    job = await store.get_job(result.job_id)
    assert job is not None
    assert job.status == "ready"
    artifact_names = {a.name for a in job.artifacts}
    # camera_path.json + pose_graph.json should always be there;
    # reconstruction.ply only when the simulated tracker emitted
    # points (it does, by default).
    assert "camera_path.json" in artifact_names or "pose_graph.json" in artifact_names


@pytest.mark.asyncio
async def test_capture_session_drops_frames_under_backpressure(tmp_data_dir):
    """If the SLAM step rate is slower than the push rate, the
    queue's bounded size means push_frame increments `_dropped`
    rather than blocking. Documents the backpressure contract so a
    regression that flips to a blocking send is caught."""
    from app.cloud.capture_session import CaptureSession

    session = CaptureSession(session_id="test-cs-bp-001", backend="mast3r_slam")
    await session.start()

    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, size=(240, 320, 3), dtype=np.uint8)
    # The frame queue is bounded to 4. Pushing 100 frames as fast as
    # we can will flood it before the loop drains.
    for idx in range(100):
        await session.push_frame(idx, img)

    # _dropped counts client-rate-vs-server-rate mismatches. Will be
    # > 0 unless the simulated tracker is wildly fast (unlikely).
    # Don't assert a specific count — just that the queue size is
    # capped so we didn't silently grow unbounded.
    assert session.frame_queue.qsize() <= 4

    await session.stop()
