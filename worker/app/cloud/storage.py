"""Pluggable artifact storage transport for remote workers.

Two backends ship behind a common `ArtifactStorage` ABC:

- `BrokerStorage` — default. Remote worker PUTs artifact bytes directly
  to `/api/worker/artifacts/{name}` on the broker, which writes them
  into the studio's local job-artifacts dir (same path local jobs use).
  Pros: zero dependencies, works through any NAT, single auth flow.
  Cons: artifact bytes transit the API process; chunked so memory is
  fine, but a very large splat on a slow uplink can hold a TCP connection
  for many minutes.

- `MinioStorage` — optional, selected via `settings.cloud_storage="minio"`.
  The broker vends a pre-signed PUT URL pointing at an S3-compatible
  bucket the user hosts (MinIO on the same host, or any real S3). The
  remote worker uploads directly to MinIO, then pings `/api/worker/
  artifacts/commit` so the broker fetches the object into the local
  artifacts dir. The viewer keeps reading from the local path, so the
  frontend's cache-busting loader and partial-snapshot UX stay
  unchanged.

The selection happens broker-side: on claim, the broker tells the
worker which mode to use. That way the studio op decides storage
policy, not the pod — a leaked credential can't force a different
upload path.

Partial-snapshot UX: both backends write bytes into the local
`artifacts/{name}.part` file atomically (rename after close), so the
viewer's cache-busting loader behaves the same regardless of transport.

Third-party SDK: the MinIO backend uses plain httpx for upload (we
already have httpx) and `aioboto3` or `boto3` for signing URLs if
available. Lazy import keeps `import` side-effects zero if the user
doesn't enable MinIO.
"""

from __future__ import annotations

import abc
import logging
from pathlib import Path
from typing import AsyncIterator, Callable, Optional

from app.config import settings

log = logging.getLogger(__name__)


# Chunk size for streaming reads/writes on artifact commit. 1 MiB
# matches the broker's chunk size so a single-direction copy doesn't
# realign on the way through.
_CHUNK_SIZE = 1 << 20


class StorageError(RuntimeError):
    """Raised for misconfiguration or a fatal transport failure. Kept as
    a single exception so callers don't juggle MinIO-vs-broker-vs-HTTP
    exception types."""


# ---------------------------------------------------------------------------
# Storage ABC + backends
# ---------------------------------------------------------------------------


class ArtifactStorage(abc.ABC):
    """Write + fetch artifacts on behalf of the studio side of the broker.

    The remote worker never touches this ABC directly — it gets either
    an upload URL (MinIO) or uses the broker's PUT endpoint. This class
    models the studio-side view: given an artifact name and bytes,
    land them on the local artifacts path where the viewer expects
    them.
    """

    @property
    @abc.abstractmethod
    def kind(self) -> str:
        """Stable id surfaced in the claim response (`broker` or `minio`).
        The remote worker branches on this to pick an upload strategy.
        """

    @abc.abstractmethod
    async def presign_put(self, job_id: str, name: str) -> dict[str, str]:
        """Return a dict the remote worker posts to. For broker-storage
        this is `{"mode": "broker", "path": ".../artifacts/<name>"}`; for
        MinIO it's `{"mode": "minio", "url": "...pre-signed PUT URL..."}`.

        Either way the remote worker uploads the bytes per the returned
        shape and then calls `commit()` with the key.
        """

    @abc.abstractmethod
    async def commit(self, job_id: str, name: str) -> int:
        """Finalize an upload. For broker the bytes are already on disk
        (the PUT endpoint landed them). For MinIO, this fetches the
        uploaded object into the local artifacts dir so the viewer can
        load it. Returns the final size in bytes."""


# ---------------------------------------------------------------------------
# Broker-storage backend
# ---------------------------------------------------------------------------


class BrokerStorage(ArtifactStorage):
    """No-op storage: the broker's PUT endpoint already wrote the file.

    `presign_put` tells the worker to upload via the broker's `PUT
    /api/worker/artifacts/{name}`, and `commit` is a pass-through that
    just verifies the file exists on disk. Kept as its own class rather
    than an `if transport == "broker"` branch in the dispatcher so the
    MinIO path is a drop-in replacement.
    """

    @property
    def kind(self) -> str:
        return "broker"

    async def presign_put(self, job_id: str, name: str) -> dict[str, str]:
        return {
            "mode": "broker",
            "path": f"/api/worker/artifacts/{name}",
        }

    async def commit(self, job_id: str, name: str) -> int:
        path = settings.job_artifacts(job_id) / name
        if not path.is_file():
            raise StorageError(
                f"broker-storage commit: artifact {name} not found at {path}"
            )
        return path.stat().st_size


# ---------------------------------------------------------------------------
# MinIO (S3-compatible) storage backend
# ---------------------------------------------------------------------------


class MinioStorage(ArtifactStorage):
    """Pre-signed PUT to an S3-compatible bucket.

    On `presign_put`, we hand the remote worker a short-TTL PUT URL
    pointing at `s3://<bucket>/<job_id>/<name>`. The worker uploads
    directly to MinIO. On `commit`, we GET the object down into the
    local artifacts dir so the viewer can read it, then delete the
    remote copy (the bucket is a transit buffer, not authoritative
    storage — keeping the artifact local simplifies per-job cleanup
    and reuses all existing cache-busting logic).

    SDK is loaded lazily; if `boto3` isn't installed `MinioStorage()`
    constructs fine but the first call raises a clear `StorageError`.
    """

    def __init__(
        self,
        *,
        endpoint_url: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        bucket: Optional[str] = None,
        region: Optional[str] = None,
        presign_ttl_s: Optional[int] = None,
    ) -> None:
        self.endpoint_url = endpoint_url or settings.minio_endpoint_url
        self.access_key = access_key if access_key is not None else settings.minio_access_key
        self.secret_key = secret_key if secret_key is not None else settings.minio_secret_key
        self.bucket = bucket or settings.minio_bucket
        self.region = region or settings.minio_region
        self.presign_ttl_s = presign_ttl_s or settings.minio_presign_ttl_s
        # Test hook: monkeypatch to a stub that generates canned URLs and
        # fake-fetches into the local artifacts dir. See
        # `worker/tests/test_minio_storage.py`.
        self._s3_client_factory: Optional[Callable[[], object]] = None
        self._http_client_factory: Optional[Callable[[], object]] = None

    @property
    def kind(self) -> str:
        return "minio"

    def _key(self, job_id: str, name: str) -> str:
        # One prefix per job so the sweeper can wipe a whole job's
        # transit state with a single `delete_objects` call.
        return f"{job_id}/{name}"

    def _s3_client(self) -> object:
        if self._s3_client_factory is not None:
            return self._s3_client_factory()
        try:
            import boto3  # noqa: WPS433
        except ImportError as exc:
            raise StorageError(
                "MinioStorage requires boto3 (install the cloud-storage extra)"
            ) from exc
        if not self.access_key or not self.secret_key:
            raise StorageError("MinioStorage: access/secret key not configured")
        return boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region,
        )

    async def presign_put(self, job_id: str, name: str) -> dict[str, str]:
        client = self._s3_client()
        key = self._key(job_id, name)
        url = client.generate_presigned_url(  # type: ignore[attr-defined]
            "put_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=self.presign_ttl_s,
        )
        return {
            "mode": "minio",
            "url": url,
            "bucket": self.bucket,
            "key": key,
            "expires_in_s": str(self.presign_ttl_s),
        }

    async def commit(self, job_id: str, name: str) -> int:
        """Copy the uploaded object into the local artifacts dir.

        We use a `.part` + rename pattern so the viewer's cache-busting
        loader never observes a half-written file, matching the local
        artifact flow. After a successful copy, the remote object is
        deleted to keep transit storage from accumulating.
        """
        client = self._s3_client()
        key = self._key(job_id, name)
        dest_dir = settings.job_artifacts(job_id)
        dest_dir.mkdir(parents=True, exist_ok=True)
        final_path = dest_dir / name
        part_path = dest_dir / (name + ".part")

        # boto3's `download_fileobj` is sync + threadsafe; running it
        # inline in the event loop is acceptable at artifact sizes we
        # expect (<1GB) — the dispatcher isn't latency-sensitive on
        # finalize. If this becomes a hot path, move to `asyncio.to_thread`.
        try:
            with part_path.open("wb") as fh:
                client.download_fileobj(self.bucket, key, fh)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            # Clean up a partial file so the next attempt starts fresh.
            try:
                part_path.unlink()
            except FileNotFoundError:
                pass
            raise StorageError(
                f"minio-storage commit: failed to fetch {key}: {exc}"
            ) from exc
        part_path.replace(final_path)

        # Best-effort delete of the transit copy. If this fails we log
        # and move on — the object is still usable but the bucket will
        # accumulate; ops can rely on a lifecycle rule as a backstop.
        try:
            client.delete_object(Bucket=self.bucket, Key=key)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            log.warning("minio delete_object %s failed: %s", key, exc)

        return final_path.stat().st_size


# ---------------------------------------------------------------------------
# Module-level selector
# ---------------------------------------------------------------------------


_ACTIVE_STORAGE: Optional[ArtifactStorage] = None


def active_storage() -> ArtifactStorage:
    """Return the process-wide `ArtifactStorage` matching
    `settings.cloud_storage`. Cached on first call so multiple callers
    share one backing client.

    Tests that want to swap in a stub can call `set_active_storage(...)`
    (below) — there's no settings reload hook because storage
    configuration is a restart-time concern in production.
    """
    global _ACTIVE_STORAGE
    if _ACTIVE_STORAGE is None:
        kind = settings.cloud_storage.lower()
        if kind == "broker":
            _ACTIVE_STORAGE = BrokerStorage()
        elif kind == "minio":
            _ACTIVE_STORAGE = MinioStorage()
        else:
            raise StorageError(
                f"unknown cloud_storage={settings.cloud_storage!r}; "
                "expected 'broker' or 'minio'"
            )
    return _ACTIVE_STORAGE


def set_active_storage(storage: Optional[ArtifactStorage]) -> None:
    """Override the cached storage (test-only). Pass `None` to reset."""
    global _ACTIVE_STORAGE
    _ACTIVE_STORAGE = storage


async def stream_file(path: Path) -> AsyncIterator[bytes]:
    """Async generator yielding `_CHUNK_SIZE` chunks of `path`.

    Shared helper so both backends commit/fetch with the same chunking
    profile. Small enough that a cancel loses ≤ 1 MiB; big enough that a
    multi-GB splat doesn't spin the loop.
    """

    def _read_all() -> list[bytes]:
        # Read into a list under threads — streaming from sync file
        # without blocking the loop would require aiofiles, which we
        # don't pull in for this narrow helper.
        with path.open("rb") as fh:
            out: list[bytes] = []
            while True:
                chunk = fh.read(_CHUNK_SIZE)
                if not chunk:
                    return out
                out.append(chunk)

    import asyncio  # noqa: WPS433 -- only used inside the helper

    chunks = await asyncio.to_thread(_read_all)
    for chunk in chunks:
        yield chunk


__all__ = [
    "ArtifactStorage",
    "BrokerStorage",
    "MinioStorage",
    "StorageError",
    "active_storage",
    "set_active_storage",
    "stream_file",
]
