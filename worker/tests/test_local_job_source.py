"""Parity check for `LocalJobSource` vs the underlying `store`/`bus`.

`LocalJobSource` is a thin delegation layer in front of today's SQLite +
events.jsonl transport. If it accidentally reshapes return values or
drops parameters, the runner refactor (next slice) would silently change
behaviour — so we pin the surface here.

No HTTP, no dispatcher: we only exercise the local impl. The HTTP source
gets its own parity suite in a later slice.

Run: `pytest worker/tests/test_local_job_source.py -q`.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest


async def _seed_queued_lingbot(
    job_id: str = "srcparity01",
    *,
    worker_class: str = "lingbot",
    with_upload: bool = True,
) -> None:
    """Put a job on the queue the way the API endpoint would."""
    from app.config import settings
    from app.jobs import store
    from app.jobs.schema import Job, LingbotConfig

    await store.init_store()

    cfg = LingbotConfig(model_id="lingbot-map", fps=10.0, image_size=384)
    uploads_dir = settings.job_uploads(job_id)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    upload_names: list[str] = []
    if with_upload:
        clip = uploads_dir / "clip.mp4"
        clip.write_bytes(b"\x00\x00\x00\x20ftypisom")  # dummy mp4 header
        upload_names.append(clip.name)

    now = datetime.now(timezone.utc)
    job = Job(
        id=job_id,
        status="queued",
        config=cfg,
        uploads=upload_names,
        artifacts=[],
        created_at=now,
        updated_at=now,
    )
    await store.create_job(job, worker_class=worker_class)


@pytest.mark.asyncio
async def test_claim_next_returns_job_with_resolved_uploads(tmp_data_dir: Path):
    from app.cloud import LocalJobSource
    from app.jobs.schema import LingbotConfig

    await _seed_queued_lingbot(job_id="srcparity01")

    src = LocalJobSource()
    claim = await src.claim_next("lingbot", worker_id="test-worker")
    assert claim is not None
    assert claim.job_id == "srcparity01"
    assert isinstance(claim.config, LingbotConfig)
    assert len(claim.uploads) == 1
    # The path resolves onto the shared data volume, as the runner expects.
    assert claim.uploads[0].name == "clip.mp4"
    assert claim.uploads[0].parent == tmp_data_dir / "jobs" / "srcparity01" / "uploads"


@pytest.mark.asyncio
async def test_claim_next_is_none_on_empty_queue(tmp_data_dir: Path):
    from app.cloud import LocalJobSource
    from app.jobs import store

    await store.init_store()
    src = LocalJobSource()
    assert await src.claim_next("lingbot", worker_id="test-worker") is None


@pytest.mark.asyncio
async def test_claim_next_filters_by_worker_class(tmp_data_dir: Path):
    """A gs-class worker must not swallow a lingbot-class job."""
    from app.cloud import LocalJobSource

    await _seed_queued_lingbot(job_id="srcparity02", worker_class="lingbot")
    src = LocalJobSource()
    assert await src.claim_next("gs", worker_id="wrong-class") is None
    claim = await src.claim_next("lingbot", worker_id="right-class")
    assert claim is not None and claim.job_id == "srcparity02"


@pytest.mark.asyncio
async def test_state_transitions_and_artifacts(tmp_data_dir: Path):
    from app.cloud import LocalJobSource
    from app.jobs import store
    from app.jobs.schema import Artifact

    await _seed_queued_lingbot(job_id="srcparity03")
    src = LocalJobSource()

    await src.set_status("srcparity03", "inference")
    await src.set_frames_total("srcparity03", 128)
    artifacts = [Artifact(name="mesh.glb", kind="glb", size_bytes=42)]
    await src.set_artifacts("srcparity03", artifacts)

    row = await store.get_job("srcparity03")
    assert row is not None
    assert row.status == "inference"
    assert row.frames_total == 128
    assert [a.name for a in row.artifacts] == ["mesh.glb"]

    await src.set_error("srcparity03", "something broke")
    row = await store.get_job("srcparity03")
    assert row is not None and row.error == "something broke"


@pytest.mark.asyncio
async def test_events_roundtrip_and_close(tmp_data_dir: Path):
    from app.cloud import LocalJobSource
    from app.jobs.schema import JobEvent

    await _seed_queued_lingbot(job_id="srcparity04")
    src = LocalJobSource()

    ev = JobEvent(job_id="srcparity04", stage="ingest", message="frame 1")
    stamped = await src.publish_event(ev)
    assert stamped.id >= 1  # bus.publish fills in a monotonic id

    events_path = tmp_data_dir / "jobs" / "srcparity04" / "events.jsonl"
    assert events_path.exists()
    rows = [json.loads(ln) for ln in events_path.read_text().splitlines() if ln.strip()]
    assert rows[-1]["message"] == "frame 1"

    await src.close_events("srcparity04")
    assert (tmp_data_dir / "jobs" / "srcparity04" / "events.done").exists()


@pytest.mark.asyncio
async def test_cancel_flag_visible_through_source(tmp_data_dir: Path):
    from app.cloud import LocalJobSource
    from app.jobs import cancel as cancel_mod

    await _seed_queued_lingbot(job_id="srcparity05")
    src = LocalJobSource()

    assert await src.is_cancel_requested("srcparity05") is False
    await cancel_mod.request_cancel("srcparity05", "from test")
    assert await src.is_cancel_requested("srcparity05") is True


@pytest.mark.asyncio
async def test_heartbeat_and_release_preserve_claim_contract(tmp_data_dir: Path):
    from app.cloud import LocalJobSource
    from app.jobs import store

    await _seed_queued_lingbot(job_id="srcparity06")
    src = LocalJobSource()

    claim = await src.claim_next("lingbot", worker_id="wrk-a")
    assert claim is not None

    # Heartbeat shouldn't change visible status; it only keeps the orphan
    # sweep off our back. The row should still be claimed by us.
    await src.heartbeat("srcparity06", worker_id="wrk-a")
    row = await store.get_job("srcparity06")
    assert row is not None and row.status in {"queued"}

    await src.release("srcparity06", worker_id="wrk-a")
    # After release, another worker can claim the same job.
    again = await src.claim_next("lingbot", worker_id="wrk-b")
    assert again is not None and again.job_id == "srcparity06"


def test_path_helpers_route_to_shared_data_volume(tmp_data_dir: Path):
    from app.cloud import LocalJobSource

    src = LocalJobSource()
    assert src.job_dir("srcparity07") == tmp_data_dir / "jobs" / "srcparity07"
    # `artifacts_dir` creates the directory as a side effect so processors
    # can start writing immediately.
    art = src.artifacts_dir("srcparity07")
    assert art == tmp_data_dir / "jobs" / "srcparity07" / "artifacts"
    assert art.exists()
    assert src.frames_dir("srcparity07") == tmp_data_dir / "jobs" / "srcparity07" / "frames"
