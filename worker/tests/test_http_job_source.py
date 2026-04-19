"""Contract pin for `HttpJobSource`.

Parity test: drive `HttpJobSource` through the exact same operations
`test_local_job_source.py` exercises, against the real broker router
served via FastAPI's `TestClient` as an ASGI transport. A passing suite
here means a remote worker running the unchanged `runner.run_job` will
produce the same DB rows + events.jsonl + artifact files as a local
worker.

The one wrinkle vs the local suite: the remote source stages uploads
into a per-job scratch dir (instead of pointing at the shared volume),
and artifacts are PUT to the broker at `set_artifacts` time. We
explicitly verify both of those extra hops.

Run: `pytest worker/tests/test_http_job_source.py -q`.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient


# --- fixtures ---------------------------------------------------------


def _mint_token(job_id: str, *, scopes: list[str] | None = None) -> str:
    from app.cloud import tokens
    from app.config import settings

    return tokens.mint(
        job_id=job_id,
        execution_target="fake",
        scopes=scopes if scopes is not None else list(tokens.SCOPES),
        ttl_s=300,
        key=settings.cloud_broker_hmac_key,
    )


async def _seed_queued_lingbot(job_id: str, *, upload_name: str | None = "clip.mp4") -> None:
    from app.config import settings
    from app.jobs import store
    from app.jobs.schema import Job, LingbotConfig

    await store.init_store()
    uploads_dir = settings.job_uploads(job_id)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    upload_names: list[str] = []
    if upload_name:
        (uploads_dir / upload_name).write_bytes(b"HELLOREMOTE")
        upload_names.append(upload_name)
    now = datetime.now(timezone.utc)
    await store.create_job(
        Job(
            id=job_id,
            status="queued",
            config=LingbotConfig(model_id="lingbot-map", fps=10.0),
            uploads=upload_names,
            artifacts=[],
            created_at=now,
            updated_at=now,
        ),
        worker_class="lingbot",
    )


@pytest.fixture
def asgi_client(tmp_data_dir: Path):
    """A TestClient so the broker's lifespan runs, plus a ready
    `HttpJobSource` wired to the same ASGI app via httpx's ASGI
    transport. No real network: the request goes straight into
    FastAPI's routing.
    """
    from app.cloud.http_source import HttpJobSource
    from app.main import app

    with TestClient(app):
        yield app


def _make_source(app, job_id: str, scratch_root: Path) -> "HttpJobSource":
    from app.cloud.http_source import HttpJobSource

    token = _mint_token(job_id)
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(
        transport=transport,
        base_url="http://studio.test",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30.0,
    )
    src = HttpJobSource(
        base_url="http://studio.test",
        token=token,
        scratch_root=scratch_root,
        client=client,
    )
    return src


# --- claim + uploads ---------------------------------------------------


async def test_claim_next_downloads_uploads_to_scratch(
    asgi_client, tmp_path: Path, tmp_data_dir: Path
):
    from app.jobs.schema import LingbotConfig

    await _seed_queued_lingbot("httpsrc01")
    scratch = tmp_path / "pod-scratch"
    src = _make_source(asgi_client, "httpsrc01", scratch)

    try:
        claim = await src.claim_next("lingbot", worker_id="pod-1")
        assert claim is not None
        assert claim.job_id == "httpsrc01"
        assert isinstance(claim.config, LingbotConfig)
        assert len(claim.uploads) == 1
        # Upload path is on the pod's scratch disk, not the shared volume.
        assert claim.uploads[0].parent == scratch / "httpsrc01" / "uploads"
        assert claim.uploads[0].read_bytes() == b"HELLOREMOTE"
    finally:
        await src.aclose()


async def test_claim_next_returns_none_on_empty_queue(
    asgi_client, tmp_path: Path, tmp_data_dir: Path
):
    from app.config import settings
    from app.jobs import store
    from app.jobs.schema import Job, LingbotConfig

    # Seed but immediately consume via a parallel "already running" state.
    await store.init_store()
    settings.job_uploads("httpsrc02").mkdir(parents=True, exist_ok=True)
    await store.create_job(
        Job(
            id="httpsrc02",
            status="inference",  # not queued anymore
            config=LingbotConfig(model_id="lingbot-map"),
            uploads=[],
        ),
        worker_class="lingbot",
    )

    src = _make_source(asgi_client, "httpsrc02", tmp_path / "pod-scratch")
    try:
        assert await src.claim_next("lingbot", worker_id="pod-2") is None
    finally:
        await src.aclose()


# --- state transitions + events ---------------------------------------


async def test_state_transitions_roundtrip_through_broker(
    asgi_client, tmp_path: Path, tmp_data_dir: Path
):
    from app.jobs import store

    await _seed_queued_lingbot("httpsrc03")
    src = _make_source(asgi_client, "httpsrc03", tmp_path / "pod-scratch")
    try:
        await src.set_status("httpsrc03", "inference")
        await src.set_frames_total("httpsrc03", 256)
        await src.set_error("httpsrc03", "boom but we keep going")

        row = await store.get_job("httpsrc03")
        assert row is not None
        assert row.status == "inference"
        assert row.frames_total == 256
        assert row.error == "boom but we keep going"
    finally:
        await src.aclose()


async def test_events_append_through_broker(
    asgi_client, tmp_path: Path, tmp_data_dir: Path
):
    from app.jobs.schema import JobEvent

    await _seed_queued_lingbot("httpsrc04")
    src = _make_source(asgi_client, "httpsrc04", tmp_path / "pod-scratch")
    try:
        await src.publish_event(
            JobEvent(job_id="httpsrc04", stage="inference", message="over the wire")
        )

        events_path = tmp_data_dir / "jobs" / "httpsrc04" / "events.jsonl"
        rows = [json.loads(ln) for ln in events_path.read_text().splitlines() if ln.strip()]
        assert any(r["message"] == "over the wire" for r in rows)
    finally:
        await src.aclose()


async def test_close_events_marks_stream_done(
    asgi_client, tmp_path: Path, tmp_data_dir: Path
):
    await _seed_queued_lingbot("httpsrc05")
    src = _make_source(asgi_client, "httpsrc05", tmp_path / "pod-scratch")
    try:
        await src.close_events("httpsrc05")
        assert (tmp_data_dir / "jobs" / "httpsrc05" / "events.done").exists()
    finally:
        await src.aclose()


# --- artifacts round-trip --------------------------------------------


async def test_set_artifacts_uploads_bytes_and_commits_manifest(
    asgi_client, tmp_path: Path, tmp_data_dir: Path
):
    from app.config import settings
    from app.jobs import store
    from app.jobs.schema import Artifact

    await _seed_queued_lingbot("httpsrc06")
    src = _make_source(asgi_client, "httpsrc06", tmp_path / "pod-scratch")
    try:
        # Simulate the processor writing an artifact to the pod's local dir.
        local_art_dir = src.artifacts_dir("httpsrc06")
        (local_art_dir / "mesh.glb").write_bytes(b"FAKE_GLB_BYTES")

        await src.set_artifacts(
            "httpsrc06",
            [Artifact(name="mesh.glb", kind="glb", size_bytes=14)],
        )

        # Bytes landed in the studio's artifacts dir (via atomic rename).
        final = settings.job_artifacts("httpsrc06") / "mesh.glb"
        assert final.read_bytes() == b"FAKE_GLB_BYTES"
        # Manifest recorded on the DB row.
        row = await store.get_job("httpsrc06")
        assert row is not None
        assert [a.name for a in row.artifacts] == ["mesh.glb"]
    finally:
        await src.aclose()


async def test_artifacts_dir_is_on_scratch_disk(
    asgi_client, tmp_path: Path, tmp_data_dir: Path
):
    scratch = tmp_path / "pod-scratch"
    src = _make_source(asgi_client, "httpsrc07", scratch)
    try:
        art_dir = src.artifacts_dir("httpsrc07")
        assert art_dir == scratch / "httpsrc07" / "artifacts"
        assert art_dir.exists()
        # The shared studio volume must NOT have this dir filled in by us.
        # (The broker writes to the studio's artifact dir during upload,
        # but the source's local artifacts_dir is the pod's scratch.)
        assert art_dir != tmp_data_dir / "jobs" / "httpsrc07" / "artifacts"
    finally:
        await src.aclose()


# --- cancel + heartbeat -----------------------------------------------


async def test_cancel_long_poll_reflects_studio_flag(
    asgi_client, tmp_path: Path, tmp_data_dir: Path
):
    from app.jobs import cancel as cancel_mod

    await _seed_queued_lingbot("httpsrc08")
    src = _make_source(asgi_client, "httpsrc08", tmp_path / "pod-scratch")
    try:
        assert await src.is_cancel_requested("httpsrc08") is False
        await cancel_mod.request_cancel("httpsrc08", "flipped by a test")
        assert await src.is_cancel_requested("httpsrc08") is True
    finally:
        await src.aclose()


async def test_heartbeat_reaches_broker(asgi_client, tmp_path: Path, tmp_data_dir: Path):
    await _seed_queued_lingbot("httpsrc09")
    src = _make_source(asgi_client, "httpsrc09", tmp_path / "pod-scratch")
    try:
        # Claim first so the heartbeat endpoint has a claimed_at row to bump.
        from app.jobs import store

        await store.claim_next_job("lingbot", worker_id="pod-hb")
        await src.heartbeat("httpsrc09", worker_id="pod-hb")
    finally:
        await src.aclose()


# --- terminal helper + scratch cleanup --------------------------------


async def test_finalize_posts_terminal_and_releases(
    asgi_client, tmp_path: Path, tmp_data_dir: Path
):
    from app.jobs import store

    await _seed_queued_lingbot("httpsrc10")
    await store.claim_next_job("lingbot", worker_id="pod-fin")

    src = _make_source(asgi_client, "httpsrc10", tmp_path / "pod-scratch")
    try:
        await src.finalize(
            "httpsrc10",
            status="ready",
            artifacts=None,
            worker_id="pod-fin",
        )
        row = await store.get_job("httpsrc10")
        assert row is not None and row.status == "ready"
        async with store.session() as s:
            db_row = await s.get(store.JobRow, "httpsrc10")
            assert db_row is not None and db_row.claimed_by is None
    finally:
        await src.aclose()


async def test_release_cleans_up_scratch(
    asgi_client, tmp_path: Path, tmp_data_dir: Path
):
    scratch = tmp_path / "pod-scratch"
    src = _make_source(asgi_client, "httpsrc11", scratch)
    try:
        # Trigger scratch creation and drop a dummy file.
        art_dir = src.artifacts_dir("httpsrc11")
        (art_dir / "junk").write_bytes(b"x")
        assert (scratch / "httpsrc11").exists()

        await src.release("httpsrc11", worker_id="pod-rel")
        # Scratch dir is gone after release.
        assert not (scratch / "httpsrc11").exists()
    finally:
        await src.aclose()


# --- fetch_checkpoint ------------------------------------------------


async def test_fetch_checkpoint_streams_to_scratch(
    asgi_client, tmp_path: Path, tmp_data_dir: Path
):
    from app.config import settings

    await _seed_queued_lingbot("httpsrc12")
    ckpt_dir = settings.models_dir / "checkpoints" / "mast3r_slam"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "weights.pt").write_bytes(b"BIG_WEIGHTS_PAYLOAD" * 32)

    src = _make_source(asgi_client, "httpsrc12", tmp_path / "pod-scratch")
    dest = tmp_path / "local-ckpt-cache" / "weights.pt"
    try:
        got = await src.fetch_checkpoint("mast3r_slam", "weights.pt", dest)
        assert got == dest
        assert dest.read_bytes() == b"BIG_WEIGHTS_PAYLOAD" * 32
    finally:
        await src.aclose()
