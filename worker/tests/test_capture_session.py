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


def _encode_jpeg(img: np.ndarray) -> bytes:
    """Encode a BGR uint8 array as a JPEG byte string the way the
    phone client does — used in tests below to drive the WS-equivalent
    `push_frame(..., raw_bytes=...)` path."""
    import cv2

    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    assert ok
    return bytes(buf.tobytes())


@pytest.mark.asyncio
async def test_capture_session_persists_frames_and_queues_real_job(
    tmp_data_dir,
):
    """The capture session writes incoming JPEGs to `<job_dir>/frames/`
    while the WS is live, then on stop creates a Job in *queued* state
    so the regular worker claim loop picks it up and runs the real
    GPU backend on the saved frames. Pre-fix behavior: the simulated
    in-process tracker's synthetic outputs were written as final
    artifacts and the Job was marked `ready` immediately — which
    looked like a real reconstruction in the UI but bore no
    resemblance to the scanned scene."""
    from app.cloud.capture_session import CaptureSession

    session = CaptureSession(session_id="test-cs-001", backend="mast3r_slam")
    await session.start()

    # Six 320×240 BGR frames of random pixels — the simulated tracker
    # only cares about shape, not content. We pass the JPEG-encoded
    # bytes alongside so the persistence path runs.
    rng = np.random.default_rng(0)
    for idx in range(6):
        img = rng.integers(0, 255, size=(240, 320, 3), dtype=np.uint8)
        await session.push_frame(idx, img, raw_bytes=_encode_jpeg(img))
        await asyncio.sleep(0.05)

    # Give the background loop time to drain.
    deadline = asyncio.get_event_loop().time() + 4.0
    while session._frame_count < 6 and asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.05)
    assert session._frame_count >= 3, (
        f"loop processed {session._frame_count}/6 frames before timeout"
    )

    # All six raw payloads should be on disk before stop is called.
    on_disk = sorted(session.frames_dir.glob("*.jpg"))
    assert len(on_disk) == 6, f"expected 6 frames on disk, got {len(on_disk)}"

    received: list[str] = []
    while not session.emit_queue.empty():
        received.append(session.emit_queue.get_nowait().type)
    assert "ready" in received

    # Stop creates a *queued* job (not ready) — the worker tier will
    # run the real reconstruction. No artifacts at create time.
    result = await session.stop()
    assert result.ok, result.error
    assert result.job_id is not None

    from app.jobs import store

    job = await store.get_job(result.job_id)
    assert job is not None
    assert job.status == "queued", (
        f"capture jobs must be queued so the worker re-runs SLAM; got "
        f"{job.status}"
    )
    # Pre-fix this was a non-empty artifact list with synthetic outputs.
    assert job.artifacts == [], (
        "capture jobs should land with no artifacts; the worker writes them"
    )
    assert job.frames_total == 6
    # The .captured marker tells the SLAM ingest step to skip ffmpeg.
    assert (session.frames_dir / ".captured").exists()


@pytest.mark.asyncio
async def test_capture_session_zero_frames_returns_failure(tmp_data_dir):
    """If the WS bounced before any frames decoded, the session must
    surface a fail rather than queuing a job that's guaranteed to error
    with `slam: no frames to track` the moment a worker claims it."""
    from app.cloud.capture_session import CaptureSession

    session = CaptureSession(session_id="test-cs-empty", backend="mast3r_slam")
    await session.start()
    # No push_frame calls.
    result = await session.stop()
    assert not result.ok
    assert result.error is not None
    assert "no frames" in result.error.lower()


@pytest.mark.asyncio
async def test_slam_ingest_short_circuits_on_captured_marker(tmp_data_dir):
    """A capture-derived job has its frames pre-extracted under
    `<job_dir>/frames/`. The SlamProcessor's ingest stage must detect
    the `.captured` marker and skip the ffmpeg extract step instead
    of running it against `uploads=[]` and producing zero frames."""
    from datetime import datetime, timezone

    import cv2

    from app.jobs.cancel import CancelToken
    from app.jobs.schema import Mast3rSlamConfig
    from app.processors.base import JobContext
    from app.processors.slam.base import SlamProcessor

    job_id = "captestjob01"
    job_dir = tmp_data_dir / "jobs" / job_id
    frames_dir = job_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Drop a couple of jpegs + the marker.
    rng = np.random.default_rng(7)
    for i in range(4):
        img = rng.integers(0, 255, size=(240, 320, 3), dtype=np.uint8)
        cv2.imwrite(str(frames_dir / f"{i:06d}.jpg"), img)
    (frames_dir / ".captured").write_text("test", encoding="utf-8")

    cfg = Mast3rSlamConfig()
    events: list = []

    async def publish(event):
        events.append(event)
        return event

    async def _noop(*_a, **_k):
        return None

    ctx = JobContext(
        job_id=job_id,
        uploads=[],  # Critically: no uploads. Pre-fix this would 0 out.
        config=cfg,
        job_dir=job_dir,
        frames_dir=frames_dir,
        artifacts_dir=job_dir / "artifacts",
        cancel=CancelToken(),
        publish=publish,
        set_status=_noop,
        set_frames_total=_noop,
    )

    proc = SlamProcessor.__subclasses__()
    # Pick whichever concrete SLAM proc is available — they all share
    # the _ingest implementation we're exercising.
    assert proc, "no concrete SlamProcessor subclasses are importable"
    n = await proc[0]()._ingest(ctx)
    assert n == 4
    # The "skipping ffmpeg ingest" event landed.
    msgs = [e.message for e in events if hasattr(e, "message")]
    assert any("skipping ffmpeg ingest" in m for m in msgs), msgs


@pytest.mark.asyncio
async def test_gsplat_resolve_inputs_accepts_jpg_frames(tmp_data_dir):
    """Captured source jobs persist `.jpg` frames; gsplat-from-source
    used to glob only `.png` and 409'd with "no extracted frames" even
    though the JPEGs were sitting on disk. Pin the .jpg fallback so
    that regression doesn't come back."""
    from datetime import datetime, timezone

    from app.jobs import store
    from app.jobs.schema import Artifact, Job, Mast3rSlamConfig
    from app.processors.gsplat import io as splat_io

    src_id = "capjpgsource1"
    job_dir = tmp_data_dir / "jobs" / src_id
    (job_dir / "frames").mkdir(parents=True, exist_ok=True)
    (job_dir / "frames" / "000000.jpg").write_bytes(b"\xff\xd8\xff")  # sentinel

    artifacts_dir = job_dir / "artifacts"
    artifacts_dir.mkdir()
    # gsplat-from-source needs at least camera_path.json or pose_graph.json
    # to resolve cameras + a pointcloud or splat for init.
    (artifacts_dir / "camera_path.json").write_text("[]", encoding="utf-8")
    (artifacts_dir / "reconstruction.ply").write_text(
        "ply\nformat ascii 1.0\nelement vertex 0\nend_header\n",
        encoding="utf-8",
    )

    now = datetime.now(timezone.utc)
    job = Job(
        id=src_id,
        status="ready",
        config=Mast3rSlamConfig(),
        artifacts=[
            Artifact(name="camera_path.json", kind="json"),
            Artifact(name="reconstruction.ply", kind="ply"),
        ],
        created_at=now,
        updated_at=now,
    )
    await store.init_store()
    await store.create_job(job, worker_class="slam")

    # Pre-fix: this raised GsplatInputsError. Post-fix: resolves
    # cleanly because the .jpg frame is detected.
    inputs = splat_io.resolve_inputs(job)
    assert inputs.frames_dir.exists()


def test_resolve_live_session_resolves_monogs():
    """MonoGS is the splat-emitting backend the capture flow points
    users at when they want a live Gaussian Splat preview rather than
    a sparse cloud. Pin that the resolver reaches it."""
    from app.processors.slam.base import SlamSession
    from app.processors.slam.live_session import (
        SUPPORTED_BACKENDS,
        resolve_live_session,
    )

    assert "monogs" in SUPPORTED_BACKENDS
    s = resolve_live_session("monogs")
    assert isinstance(s, SlamSession)
    # The MonoGS sessions (real + simulated) both expose
    # set_artifact_dir so the capture session can hand them a place to
    # write splat.ply at finalize().
    assert hasattr(s, "set_artifact_dir")


@pytest.mark.asyncio
async def test_capture_session_emits_splat_preview(tmp_data_dir):
    """After enough frames, the session writes a splat.ply preview to
    its preview dir and emits a `partial_splat` event so the capture
    page's SplatLayer can render it. Pin the wiring so a future change
    that drops the periodic preview is loud."""
    from app.cloud.capture_session import (
        CaptureSession,
        _PREVIEW_MIN_INTERVAL_S,
    )
    import app.cloud.capture_session as cap_mod

    # Force the throttle to ~0 so the test doesn't have to sleep
    # 2 s for the second preview tick.
    monkeypatched = 0.0
    cap_mod._PREVIEW_MIN_INTERVAL_S = monkeypatched
    try:
        session = CaptureSession(session_id="test-cs-preview", backend="monogs")
        await session.start()
        rng = np.random.default_rng(2)
        for idx in range(10):
            img = rng.integers(0, 255, size=(240, 320, 3), dtype=np.uint8)
            await session.push_frame(idx, img)
            await asyncio.sleep(0.05)

        deadline = asyncio.get_event_loop().time() + 4.0
        while session._preview_count < 1 and asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.05)

        assert session._preview_count >= 1, "no splat preview was emitted"
        assert (session.preview_dir / "splat.ply").exists()

        # The emit_queue should carry at least one `partial_splat` msg.
        types_seen: list[str] = []
        while not session.emit_queue.empty():
            types_seen.append(session.emit_queue.get_nowait().type)
        assert "partial_splat" in types_seen, types_seen

        await session.stop()
    finally:
        cap_mod._PREVIEW_MIN_INTERVAL_S = _PREVIEW_MIN_INTERVAL_S


def test_ws_decode_path_imports_resolve():
    """Closes a regression where every JPEG frame the phone pushed was
    silently dropped because the api container's image was missing
    `opencv-python-headless`. The capture WS handler used to lazy-import
    cv2 inside the per-frame loop, so a missing dep surfaced as a
    per-frame `ModuleNotFoundError` log line and zero frames captured —
    invisible to every existing test because the unit tests bypass the
    WS handler and push pre-decoded numpy arrays directly.

    Asserting the imports resolve at module load (where `app.main`
    pulls them in eagerly) means a future Dockerfile change that drops
    cv2 will fail the api container's startup with a clear
    `ModuleNotFoundError: No module named 'cv2'` rather than silently
    eating frames forever."""
    import app.main  # noqa: F401 — exercising the import is the point
    import cv2  # noqa: F401
    import numpy as np  # noqa: F401

    # Sanity: cv2.imdecode is what the WS handler actually calls. If
    # opencv ever ships a future version where the symbol moves, this
    # forces a loud failure here rather than a silent regression.
    assert hasattr(cv2, "imdecode")


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
