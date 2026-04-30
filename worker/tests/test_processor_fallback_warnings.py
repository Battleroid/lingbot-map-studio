"""Tests for the warn-level events processors emit when they fall back to
simulated/CPU paths. These are user-facing signals — they appear in the log
pane on the job page and tell the user "you're watching a placeholder, not
a real run". Easy to break; worth pinning."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_slam_processor_raises_when_real_tracker_unavailable(
    tmp_data_dir, synthetic_frames
):
    """The simulated-fallback warn path was removed in the strict-
    no-simulated cleanup. `Mast3rSlamProcessor._make_session` now
    raises `Mast3rSlamUnavailableError` when the real CUDA stack is
    missing — the runner's outer except marks the job failed with
    the install-instruction message instead of silently running the
    placeholder and emitting a warn."""
    from app.jobs.cancel import CancelToken
    from app.jobs.schema import Mast3rSlamConfig
    from app.processors.base import JobContext
    from app.processors.slam.mast3r_slam import (
        Mast3rSlamProcessor,
        Mast3rSlamUnavailableError,
    )

    job_id = "warn-slam-001"
    job_dir = tmp_data_dir / "jobs" / job_id
    frames_dir = job_dir / "frames"
    artifacts_dir = job_dir / "artifacts"
    frames_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(synthetic_frames.glob("*.png")):
        (frames_dir / p.name).write_bytes(p.read_bytes())

    cfg = Mast3rSlamConfig(
        keyframe_policy="translation",
        max_frames=None,
        stride=1,
        partial_snapshot_every=4,
        run_poisson_mesh=False,
    )
    events: list = []

    async def publish(event):
        events.append(event)
        return event

    async def noop(*_a, **_k):
        return None

    ctx = JobContext(
        job_id=job_id,
        uploads=[],
        config=cfg,
        job_dir=job_dir,
        frames_dir=frames_dir,
        artifacts_dir=artifacts_dir,
        cancel=CancelToken(),
        publish=publish,
        set_status=noop,
        set_frames_total=noop,
    )

    processor = Mast3rSlamProcessor()

    async def _fake_ingest(_ctx):
        count = len(sorted(frames_dir.glob("*.png")))
        await ctx.set_frames_total(count)
        return count

    processor._ingest = _fake_ingest  # type: ignore[assignment]

    with pytest.raises(Mast3rSlamUnavailableError):
        await processor.run(ctx)

    # No simulated-tracker warn event should be emitted anymore.
    sim_warns = [
        e
        for e in events
        if e.level == "warn"
        and e.stage == "system"
        and "simulated" in e.message.lower()
    ]
    assert not sim_warns, (
        "no simulated-tracker warn should fire — production must "
        f"either run real or fail loud. got: {[e.message[:60] for e in sim_warns]}"
    )


@pytest.mark.asyncio
async def test_gsplat_emits_error_event_when_real_trainer_unavailable(
    tmp_data_dir,
):
    """The simulated-fallback path was removed in the "no fake gsplat
    output" fix. When the real CUDA trainer can't be loaded,
    `GsplatProcessor.run` now emits a level=error system event with
    install instructions and re-raises so the runner marks the job
    failed — instead of silently running the placeholder and shipping
    synthetic PSNR / loss numbers as if they were real."""
    from app.jobs import store
    from app.jobs.cancel import CancelToken
    from app.jobs.schema import (
        GsplatConfig,
        Job,
        Mast3rSlamConfig,
        Artifact,
    )
    from app.processors.base import JobContext
    from app.processors.gsplat.trainer import GsplatProcessor

    # Need a "ready" source job for resolve_inputs() to accept the run.
    await store.init_store()
    source_id = "warn-gs-src-001"
    source_dir = tmp_data_dir / "jobs" / source_id
    src_artifacts = source_dir / "artifacts"
    src_frames = source_dir / "frames"
    src_artifacts.mkdir(parents=True, exist_ok=True)
    src_frames.mkdir(parents=True, exist_ok=True)
    # Minimal frames + cameras so resolve_inputs() can succeed.
    for i in range(3):
        (src_frames / f"frame_{i:06d}.png").write_bytes(
            b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
        )
    (src_artifacts / "camera_path.json").write_text(
        json.dumps({"fps": 10, "poses": [{"position": [0, 0, 0], "quaternion": [0, 0, 0, 1]}]})
    )
    # Empty PLY just to satisfy the init_points lookup.
    (src_artifacts / "reconstruction.ply").write_text(
        "ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
    )

    src_cfg = Mast3rSlamConfig(keyframe_policy="translation")
    src_job = Job(
        id=source_id,
        status="ready",
        config=src_cfg,
        uploads=[],
        artifacts=[
            Artifact(name="camera_path.json", kind="json"),
            Artifact(name="reconstruction.ply", kind="ply"),
        ],
        frames_total=3,
    )
    await store.create_job(src_job, worker_class="slam")
    await store.update_job(source_id, status="ready")

    job_id = "warn-gs-001"
    job_dir = tmp_data_dir / "jobs" / job_id
    frames_dir = job_dir / "frames"
    artifacts_dir = job_dir / "artifacts"
    frames_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    cfg = GsplatConfig(
        source_job_id=source_id,
        iterations=2,
        preview_every_iters=10,
    )
    events: list = []

    async def publish(event):
        events.append(event)
        return event

    async def noop(*_a, **_k):
        return None

    ctx = JobContext(
        job_id=job_id,
        uploads=[],
        config=cfg,
        job_dir=job_dir,
        frames_dir=frames_dir,
        artifacts_dir=artifacts_dir,
        cancel=CancelToken(),
        publish=publish,
        set_status=noop,
        set_frames_total=noop,
    )

    from app.processors.gsplat.trainer import GsplatTrainerUnavailableError

    processor = GsplatProcessor()
    with pytest.raises(GsplatTrainerUnavailableError):
        await processor.run(ctx)

    err_events = [
        e
        for e in events
        if e.level == "error"
        and e.stage == "system"
        and "gsplat" in e.message.lower()
    ]
    assert err_events, (
        "expected a level=error system event with install instructions; "
        f"got {[(e.level, e.stage, e.message[:60]) for e in events]}"
    )
    assert err_events[0].data.get("missing_dep") == "gsplat"
    assert (
        "torch is not installed" in err_events[0].message
        or "gsplat" in err_events[0].message.lower()
    )


@pytest.mark.asyncio
async def test_lingbot_emits_warn_event_when_cuda_unavailable(monkeypatch, tmp_data_dir):
    """When `torch.cuda.is_available()` is False (worker container without
    GPU passthrough), `run_inference` should emit a loud warn event before
    starting the inference thread. Catches the silent CPU-fallback case."""
    import sys
    from types import SimpleNamespace

    from app.jobs.schema import JobEvent, LingbotConfig
    from app.pipeline import inference

    # Stub `torch` so we don't actually need it installed. The function
    # only consults `torch.cuda.is_available()` before the thread spawn,
    # and it raises before the thread runs (no frames in the dir).
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    job_id = "warn-lb-001"
    frames_dir = tmp_data_dir / "jobs" / job_id / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    # Drop a fake png so _list_frames returns at least one entry — enough
    # to get past the "no frames" guard before our warn fires.
    (frames_dir / "frame_000001.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)

    events: list = []

    async def publish(event: JobEvent):
        events.append(event)
        return event

    cfg = LingbotConfig()
    # The thread spawn after our warn will fail (no real model, no
    # checkpoint), but that's OK — we only assert the warn fires first.
    with pytest.raises(Exception):
        await inference.run_inference(
            job_id=job_id,
            frames_dir=frames_dir,
            ckpt_path=tmp_data_dir / "missing.pth",
            config=cfg,
            publish=publish,
        )

    cpu_warns = [
        e
        for e in events
        if e.level == "warn"
        and e.stage == "system"
        and "cuda" in e.message.lower()
        and "cpu" in e.message.lower()
    ]
    assert cpu_warns, (
        "expected a level=warn event about CUDA fallback; got "
        f"{[(e.level, e.stage, e.message[:60]) for e in events]}"
    )
    assert cpu_warns[0].data.get("cpu_fallback") is True
