"""Pin the `HttpJobSource.sync_artifacts` watcher semantics.

`sync_artifacts` is what makes live-preview UX parity work for remote
runs: a processor writes `partial_007.ply` into its local scratch
artifacts dir, and within one poll tick the same file appears in the
studio's shared artifacts dir with matching bytes. These tests drive
the watcher against the real broker router via httpx's ASGI transport
so the property we actually care about — "bytes land on the studio's
disk" — is verified end to end.

Run: `pytest worker/tests/test_artifact_sync.py -q`.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient


def _mint_token(job_id: str) -> str:
    from app.cloud import tokens
    from app.config import settings

    return tokens.mint(
        job_id=job_id,
        execution_target="fake",
        scopes=list(tokens.SCOPES),
        ttl_s=300,
        key=settings.cloud_broker_hmac_key,
    )


async def _seed_queued(job_id: str) -> None:
    from app.config import settings
    from app.jobs import store
    from app.jobs.schema import Job, LingbotConfig

    await store.init_store()
    settings.job_uploads(job_id).mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    await store.create_job(
        Job(
            id=job_id,
            status="queued",
            config=LingbotConfig(model_id="lingbot-map"),
            uploads=[],
            artifacts=[],
            created_at=now,
            updated_at=now,
        ),
        worker_class="lingbot",
    )


def _make_source(app, job_id: str, scratch_root: Path):
    from app.cloud.http_source import HttpJobSource

    token = _mint_token(job_id)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://studio.test",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30.0,
    )
    return HttpJobSource(
        base_url="http://studio.test",
        token=token,
        scratch_root=scratch_root,
        client=client,
    )


@pytest.fixture
def asgi_app(tmp_data_dir: Path):
    from app.main import app

    with TestClient(app):
        yield app


async def test_sync_uploads_new_files_to_studio(
    asgi_app, tmp_path: Path, tmp_data_dir: Path
):
    from app.config import settings

    await _seed_queued("syncpartial01")
    src = _make_source(asgi_app, "syncpartial01", tmp_path / "scratch")
    try:
        local_dir = src.artifacts_dir("syncpartial01")
        (local_dir / "partial_001.ply").write_bytes(b"PARTIAL_A")

        await src.sync_artifacts("syncpartial01")

        studio_dir = settings.job_artifacts("syncpartial01")
        assert (studio_dir / "partial_001.ply").read_bytes() == b"PARTIAL_A"
    finally:
        await src.aclose()


async def test_sync_detects_file_updates(
    asgi_app, tmp_path: Path, tmp_data_dir: Path
):
    """A file that grows between ticks should be re-uploaded the next tick.

    Covers the gsplat training case: `partial_splat_NNN.ply` gets
    overwritten in place as training iterations advance.
    """
    from app.config import settings

    await _seed_queued("syncpartial02")
    src = _make_source(asgi_app, "syncpartial02", tmp_path / "scratch")
    try:
        local_dir = src.artifacts_dir("syncpartial02")
        target = local_dir / "partial_splat.ply"
        target.write_bytes(b"v1" * 8)
        await src.sync_artifacts("syncpartial02")
        studio = settings.job_artifacts("syncpartial02") / "partial_splat.ply"
        assert studio.read_bytes() == b"v1" * 8

        # Grow the file (new training iteration) and advance mtime so
        # the sync tick picks it up. Without the explicit bump, mtime
        # may equal the previous tick on fast filesystems.
        target.write_bytes(b"v22222" * 32)
        past = target.stat().st_mtime_ns + 1_000_000
        os.utime(target, ns=(past, past))

        await src.sync_artifacts("syncpartial02")
        assert studio.read_bytes() == b"v22222" * 32
    finally:
        await src.aclose()


async def test_sync_skips_unchanged_files(
    asgi_app, tmp_path: Path, tmp_data_dir: Path
):
    """Calling sync twice back-to-back must not re-upload the same bytes.

    This matters for long gsplat runs where the sync loop runs every
    second but most files haven't changed — we need the common case to
    be an `os.stat` + skip, not an O(N*M) full re-upload.
    """
    from app.config import settings

    await _seed_queued("syncpartial03")
    src = _make_source(asgi_app, "syncpartial03", tmp_path / "scratch")
    try:
        local = src.artifacts_dir("syncpartial03")
        (local / "steady.ply").write_bytes(b"Z" * 64)
        await src.sync_artifacts("syncpartial03")

        # Corrupt the studio-side copy to prove the second sync is a no-op.
        studio = settings.job_artifacts("syncpartial03") / "steady.ply"
        studio.write_bytes(b"TAMPERED")

        await src.sync_artifacts("syncpartial03")
        # Still tampered — the watcher saw no change locally and didn't re-push.
        assert studio.read_bytes() == b"TAMPERED"
    finally:
        await src.aclose()


async def test_sync_ignores_part_sidecars(
    asgi_app, tmp_path: Path, tmp_data_dir: Path
):
    """Broker writes through a `.part` rename; the watcher must not pick
    up `.part` sidecars on its own side. A race where the watcher grabs
    a half-written `.part` would land truncated bytes on the studio.
    """
    from app.config import settings

    await _seed_queued("syncpartial04")
    src = _make_source(asgi_app, "syncpartial04", tmp_path / "scratch")
    try:
        local = src.artifacts_dir("syncpartial04")
        (local / "stream.ply.part").write_bytes(b"DONT-SEND-ME-YET")

        await src.sync_artifacts("syncpartial04")

        studio = settings.job_artifacts("syncpartial04")
        # No file named *.part on the studio side, and no `stream.ply`.
        # Either the dir doesn't exist yet or it's empty — both are proof
        # nothing was uploaded.
        assert not studio.exists() or not list(studio.iterdir())
    finally:
        await src.aclose()


async def test_sync_skips_empty_files(
    asgi_app, tmp_path: Path, tmp_data_dir: Path
):
    """A zero-byte file on first sight is assumed to be a writer that
    just opened the handle. Waiting one tick avoids uploading a blank
    artifact that would then get overwritten seconds later."""
    from app.config import settings

    await _seed_queued("syncpartial05")
    src = _make_source(asgi_app, "syncpartial05", tmp_path / "scratch")
    try:
        local = src.artifacts_dir("syncpartial05")
        empty = local / "not-yet.ply"
        empty.write_bytes(b"")

        await src.sync_artifacts("syncpartial05")
        assert not (settings.job_artifacts("syncpartial05") / "not-yet.ply").exists()

        # Writer finishes; second tick pushes it.
        empty.write_bytes(b"NOW_IT_HAS_DATA")
        await src.sync_artifacts("syncpartial05")
        assert (settings.job_artifacts("syncpartial05") / "not-yet.ply").read_bytes() == b"NOW_IT_HAS_DATA"
    finally:
        await src.aclose()


async def test_local_sync_is_noop(tmp_data_dir: Path):
    """`LocalJobSource.sync_artifacts` must do nothing — the shared volume
    IS the studio's artifacts dir. A non-no-op would double-write."""
    from app.cloud import LocalJobSource

    src = LocalJobSource()
    # Nothing seeded; the method should just return without error.
    await src.sync_artifacts("nonexistent_job")
