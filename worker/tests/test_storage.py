"""Exercise the storage abstraction end-to-end.

The broker backend is easy: its `commit()` just asserts the file exists
on the local artifacts dir (where the PUT handler already wrote it).
The MinIO backend is trickier because we don't have boto3 installed —
we inject a fake S3 client via the `_s3_client_factory` hook and verify
the flow: presign returns a URL, commit downloads into the job
artifacts dir, delete_object runs on success.

These tests avoid any network I/O; the commit path reads from a
stub that simulates S3's `download_fileobj` by writing canned bytes
into the destination file handle.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest

from app.cloud import storage as storage_mod
from app.config import settings


class _FakeS3Client:
    """Stand-in for `boto3.client('s3')`. Records calls + serves canned
    object bodies. `objects` maps `(bucket, key)` to the bytes it would
    return on download_fileobj."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.deleted: list[tuple[str, str]] = []
        self.presigned: list[tuple[str, dict[str, Any], int]] = []

    def generate_presigned_url(
        self,
        ClientMethod: str,
        Params: dict[str, Any] | None = None,
        ExpiresIn: int = 0,
    ) -> str:
        self.presigned.append((ClientMethod, dict(Params or {}), ExpiresIn))
        bucket = (Params or {}).get("Bucket", "")
        key = (Params or {}).get("Key", "")
        return f"https://fake-minio.test/{bucket}/{key}?sig=fake"

    def download_fileobj(self, Bucket: str, Key: str, Fileobj: io.IOBase) -> None:
        body = self.objects.get((Bucket, Key))
        if body is None:
            raise Exception(f"NoSuchKey: {Bucket}/{Key}")
        Fileobj.write(body)

    def delete_object(self, Bucket: str, Key: str) -> dict[str, Any]:
        self.deleted.append((Bucket, Key))
        return {"DeleteMarker": True}


@pytest.fixture(autouse=True)
def _reset_active_storage():
    # Each test starts with no cached backend so settings flips are
    # honoured on the next active_storage() call.
    storage_mod.set_active_storage(None)
    yield
    storage_mod.set_active_storage(None)


@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    """Point `settings.data_dir` at a temp dir so commit() writes into a
    scratch area, not the dev studio's real artifacts tree."""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    return tmp_path


# --- BrokerStorage --------------------------------------------------------


async def test_broker_storage_presign_points_at_broker():
    s = storage_mod.BrokerStorage()
    out = await s.presign_put("jobbroker", "splat.ply")
    assert out == {"mode": "broker", "path": "/api/worker/artifacts/splat.ply"}


async def test_broker_storage_commit_returns_size(tmp_data_dir):
    s = storage_mod.BrokerStorage()
    art_dir = tmp_data_dir / "jobs" / "jobbroker" / "artifacts"
    art_dir.mkdir(parents=True)
    (art_dir / "mesh.glb").write_bytes(b"\x00" * 1234)

    size = await s.commit("jobbroker", "mesh.glb")
    assert size == 1234


async def test_broker_storage_commit_raises_when_missing(tmp_data_dir):
    s = storage_mod.BrokerStorage()
    with pytest.raises(storage_mod.StorageError) as excinfo:
        await s.commit("jobbroker", "absent.ply")
    assert "not found" in str(excinfo.value)


# --- MinioStorage ---------------------------------------------------------


async def test_minio_presign_returns_put_url():
    fake = _FakeS3Client()
    s = storage_mod.MinioStorage(
        endpoint_url="http://minio:9000",
        access_key="key",
        secret_key="secret",
        bucket="bkt",
        region="us-east-1",
        presign_ttl_s=600,
    )
    s._s3_client_factory = lambda: fake

    out = await s.presign_put("jobminio", "splat.ply")
    assert out["mode"] == "minio"
    assert out["bucket"] == "bkt"
    assert out["key"] == "jobminio/splat.ply"
    assert out["url"] == "https://fake-minio.test/bkt/jobminio/splat.ply?sig=fake"
    assert fake.presigned[0] == (
        "put_object",
        {"Bucket": "bkt", "Key": "jobminio/splat.ply"},
        600,
    )


async def test_minio_commit_downloads_and_cleans_transit(tmp_data_dir):
    fake = _FakeS3Client()
    payload = b"splat-bytes" * 100
    fake.objects[("bkt", "jobminio/splat.ply")] = payload
    s = storage_mod.MinioStorage(bucket="bkt", access_key="k", secret_key="s")
    s._s3_client_factory = lambda: fake

    size = await s.commit("jobminio", "splat.ply")

    assert size == len(payload)
    dest = tmp_data_dir / "jobs" / "jobminio" / "artifacts" / "splat.ply"
    assert dest.is_file()
    assert dest.read_bytes() == payload
    # Partial sidecar gone after atomic rename.
    assert not (dest.parent / "splat.ply.part").exists()
    # Transit copy deleted.
    assert fake.deleted == [("bkt", "jobminio/splat.ply")]


async def test_minio_commit_cleans_part_on_download_failure(tmp_data_dir):
    fake = _FakeS3Client()
    # No object inserted → download_fileobj raises "NoSuchKey".
    s = storage_mod.MinioStorage(bucket="bkt", access_key="k", secret_key="s")
    s._s3_client_factory = lambda: fake

    with pytest.raises(storage_mod.StorageError):
        await s.commit("jobminio", "missing.ply")
    # No dangling .part sidecar after a failed fetch.
    part = tmp_data_dir / "jobs" / "jobminio" / "artifacts" / "missing.ply.part"
    assert not part.exists()


# --- active_storage selector ---------------------------------------------


async def test_active_storage_defaults_to_broker(monkeypatch):
    monkeypatch.setattr(settings, "cloud_storage", "broker")
    assert isinstance(storage_mod.active_storage(), storage_mod.BrokerStorage)


async def test_active_storage_selects_minio(monkeypatch):
    monkeypatch.setattr(settings, "cloud_storage", "minio")
    assert isinstance(storage_mod.active_storage(), storage_mod.MinioStorage)


async def test_active_storage_raises_on_unknown_kind(monkeypatch):
    monkeypatch.setattr(settings, "cloud_storage", "gopher")
    with pytest.raises(storage_mod.StorageError):
        storage_mod.active_storage()
