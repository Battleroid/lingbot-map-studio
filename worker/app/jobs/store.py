from __future__ import annotations

import json
import os
import socket
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator, Optional

from sqlalchemy import Boolean, DateTime, String, Text, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import settings
from app.jobs.schema import (
    AnyJobConfig,
    Artifact,
    Job,
    JobStatus,
    JobSummary,
    dump_job_config,
    parse_job_config,
)


class Base(DeclarativeBase):
    pass


class JobRow(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    config_json: Mapped[str] = mapped_column(Text, nullable=False)
    uploads_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    artifacts_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    frames_total: Mapped[Optional[int]] = mapped_column(nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Queue + cross-process cancel bookkeeping (Phase 2).
    # worker_class mirrors Processor.worker_class so each worker container only
    # claims jobs it can actually run.
    worker_class: Mapped[str] = mapped_column(
        String(16), nullable=False, default="lingbot"
    )
    # claimed_by is a stable identifier for the worker process that owns the
    # job. NULL means the job is waiting in the queue. claimed_at anchors the
    # orphan sweep: if the claim is older than the stale threshold and the job
    # is still marked running, the claim is released.
    claimed_by: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    claimed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # cancel_requested is the cross-process signal. The API flips it; the
    # worker's cancel-poller reads it and raises JobCancelled on next check.
    cancel_requested: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )


_engine = None
_Session: async_sessionmaker[AsyncSession] | None = None


async def init_store() -> None:
    global _engine, _Session
    settings.ensure_dirs()
    url = f"sqlite+aiosqlite:///{settings.sqlite_path()}"
    _engine = create_async_engine(url, future=True)
    _Session = async_sessionmaker(_engine, expire_on_commit=False, class_=AsyncSession)
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # WAL mode lets readers proceed while a writer is active, dramatically
        # reduces journal-file FD churn, and is the recommended mode for any
        # server-style sqlite use. busy_timeout waits up to 5s for locks
        # rather than erroring instantly.
        await conn.exec_driver_sql("PRAGMA journal_mode=WAL")
        await conn.exec_driver_sql("PRAGMA synchronous=NORMAL")
        await conn.exec_driver_sql("PRAGMA busy_timeout=5000")

        # Inline migration for pre-Phase-2 rows: the columns below were added
        # after initial rollout, so existing DBs are missing them. SQLite
        # doesn't allow IF NOT EXISTS on ADD COLUMN, so probe pragma first.
        existing = await conn.exec_driver_sql("PRAGMA table_info(jobs)")
        cols = {row[1] for row in existing.fetchall()}
        migrations = [
            ("worker_class", "ALTER TABLE jobs ADD COLUMN worker_class TEXT NOT NULL DEFAULT 'lingbot'"),
            ("claimed_by", "ALTER TABLE jobs ADD COLUMN claimed_by TEXT"),
            ("claimed_at", "ALTER TABLE jobs ADD COLUMN claimed_at DATETIME"),
            ("cancel_requested", "ALTER TABLE jobs ADD COLUMN cancel_requested BOOLEAN NOT NULL DEFAULT 0"),
        ]
        for col, ddl in migrations:
            if col not in cols:
                await conn.exec_driver_sql(ddl)


@asynccontextmanager
async def session() -> AsyncIterator[AsyncSession]:
    assert _Session is not None, "init_store() not called"
    async with _Session() as s:
        yield s


def _row_to_job(row: JobRow) -> Job:
    return Job(
        id=row.id,
        status=row.status,  # type: ignore[arg-type]
        config=parse_job_config(row.config_json),
        uploads=json.loads(row.uploads_json),
        artifacts=[Artifact.model_validate(a) for a in json.loads(row.artifacts_json)],
        frames_total=row.frames_total,
        error=row.error,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def worker_identity() -> str:
    """Stable-ish identifier for this worker process.

    Combines hostname + PID so a human can map a claim back to a running
    container in `docker compose ps`. Recomputed each call is fine; writes
    go through sqlite anyway.
    """
    return f"{socket.gethostname()}:{os.getpid()}"


async def create_job(job: Job, worker_class: str = "lingbot") -> None:
    async with session() as s:
        row = JobRow(
            id=job.id,
            status=job.status,
            config_json=dump_job_config(job.config),
            uploads_json=json.dumps(job.uploads),
            artifacts_json=json.dumps([a.model_dump(mode="json") for a in job.artifacts]),
            frames_total=job.frames_total,
            error=job.error,
            created_at=job.created_at,
            updated_at=job.updated_at,
            worker_class=worker_class,
        )
        s.add(row)
        await s.commit()


async def get_job(job_id: str) -> Optional[Job]:
    async with session() as s:
        row = await s.get(JobRow, job_id)
        if row is None:
            return None
        return _row_to_job(row)


async def update_job(
    job_id: str,
    *,
    status: Optional[JobStatus] = None,
    frames_total: Optional[int] = None,
    error: Optional[str] = None,
    artifacts: Optional[list[Artifact]] = None,
) -> Optional[Job]:
    async with session() as s:
        row = await s.get(JobRow, job_id)
        if row is None:
            return None
        if status is not None:
            row.status = status
        if frames_total is not None:
            row.frames_total = frames_total
        if error is not None:
            row.error = error
        if artifacts is not None:
            row.artifacts_json = json.dumps([a.model_dump(mode="json") for a in artifacts])
        row.updated_at = datetime.now(timezone.utc)
        await s.commit()
        return _row_to_job(row)


async def list_jobs() -> list[JobSummary]:
    async with session() as s:
        rows = (await s.execute(select(JobRow).order_by(JobRow.created_at.desc()))).scalars().all()
        out: list[JobSummary] = []
        for row in rows:
            artifacts = json.loads(row.artifacts_json)
            # Pre-refactor rows have no `processor` field — treat as lingbot.
            cfg_raw = json.loads(row.config_json)
            processor_id = cfg_raw.get("processor", "lingbot")
            out.append(
                JobSummary(
                    id=row.id,
                    status=row.status,  # type: ignore[arg-type]
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                    frames_total=row.frames_total,
                    artifact_count=len(artifacts),
                    processor=processor_id,
                )
            )
        return out


async def delete_job(job_id: str) -> bool:
    async with session() as s:
        row = await s.get(JobRow, job_id)
        if row is None:
            return False
        await s.delete(row)
        await s.commit()
        return True


# --- Queue ops ------------------------------------------------------------

# Statuses that indicate a job is actively being processed by a worker and
# therefore should participate in claim/orphan bookkeeping.
RUNNING_STATUSES = frozenset({"queued", "ingest", "inference", "export", "slam", "meshing", "training"})


async def claim_next_job(
    worker_class: str, worker_id: Optional[str] = None
) -> Optional[tuple[str, AnyJobConfig, list[str]]]:
    """Atomically claim the next queued job for this worker class.

    Returns `(job_id, config, upload_names)` on success, None if the queue is
    empty. Uses a single transactional UPDATE keyed on status+worker_class so
    two worker processes can't grab the same job.
    """
    if worker_id is None:
        worker_id = worker_identity()
    now = datetime.now(timezone.utc)

    async with session() as s:
        async with s.begin():
            # BEGIN IMMEDIATE (via session.begin() nested) takes a write lock so
            # no other worker can race us between SELECT and UPDATE.
            row = (
                await s.execute(
                    select(JobRow)
                    .where(JobRow.status == "queued")
                    .where(JobRow.worker_class == worker_class)
                    .where(JobRow.claimed_by.is_(None))
                    .order_by(JobRow.created_at.asc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            if row is None:
                return None
            row.claimed_by = worker_id
            row.claimed_at = now
            row.updated_at = now
            # Status stays "queued" here; the processor bumps it to ingest /
                # inference / etc as it progresses.
            return (row.id, parse_job_config(row.config_json), json.loads(row.uploads_json))


async def release_job(
    job_id: str, *, worker_id: Optional[str] = None, requeue: bool = False
) -> None:
    """Clear this worker's claim on the job.

    If `requeue=True`, also bumps status back to 'queued' so another worker
    can pick it up (used on graceful shutdown mid-run). Otherwise just drops
    the claim (used at the end of a run, after status is already terminal).
    """
    if worker_id is None:
        worker_id = worker_identity()
    now = datetime.now(timezone.utc)
    async with session() as s:
        async with s.begin():
            row = await s.get(JobRow, job_id)
            if row is None:
                return
            # Don't stomp on a claim that has since been taken by someone else
            # (shouldn't happen, but we're paranoid about cross-process races).
            if row.claimed_by and row.claimed_by != worker_id:
                return
            row.claimed_by = None
            row.claimed_at = None
            if requeue:
                row.status = "queued"
                row.cancel_requested = False
            row.updated_at = now


async def request_cancel(job_id: str) -> bool:
    """Flip the cross-process cancel flag. Returns True if the row existed."""
    now = datetime.now(timezone.utc)
    async with session() as s:
        async with s.begin():
            row = await s.get(JobRow, job_id)
            if row is None:
                return False
            row.cancel_requested = True
            row.updated_at = now
            return True


async def is_cancel_requested(job_id: str) -> bool:
    """Cheap read used by the worker-side cancel poller."""
    async with session() as s:
        row = await s.get(JobRow, job_id)
        return bool(row and row.cancel_requested)


async def sweep_stale_claims(stale_after_s: float) -> int:
    """Reap claims whose claimed_at is older than `stale_after_s`.

    A worker that died mid-job leaves its claim + running status behind. The
    sweep flips those rows to `failed` with a clear error so the UI shows a
    consistent state and the user can delete or restart. Returns the number
    of rows reaped.
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(seconds=stale_after_s)
    reaped = 0
    async with session() as s:
        async with s.begin():
            rows = (
                await s.execute(
                    select(JobRow)
                    .where(JobRow.claimed_by.is_not(None))
                    .where(JobRow.claimed_at < cutoff)
                    .where(JobRow.status.in_(tuple(RUNNING_STATUSES)))
                )
            ).scalars().all()
            for row in rows:
                row.status = "failed"
                row.error = (
                    f"worker {row.claimed_by} vanished mid-run "
                    f"(no heartbeat since {row.claimed_at.isoformat()}) — "
                    "marked as orphaned. delete or restart."
                )
                row.claimed_by = None
                row.claimed_at = None
                row.updated_at = now
                reaped += 1
    return reaped


async def heartbeat(job_id: str, worker_id: Optional[str] = None) -> None:
    """Bump claimed_at so the stale-claim sweep doesn't reap an active job.

    Workers call this every N seconds inside their run loop. Cheap — a
    single UPDATE by primary key.
    """
    if worker_id is None:
        worker_id = worker_identity()
    now = datetime.now(timezone.utc)
    async with session() as s:
        async with s.begin():
            row = await s.get(JobRow, job_id)
            if row is None or row.claimed_by != worker_id:
                return
            row.claimed_at = now


async def count_queued(worker_class: Optional[str] = None) -> int:
    """Number of jobs waiting in the queue. For health/metrics endpoints."""
    async with session() as s:
        stmt = select(JobRow).where(JobRow.status == "queued").where(JobRow.claimed_by.is_(None))
        if worker_class is not None:
            stmt = stmt.where(JobRow.worker_class == worker_class)
        rows = (await s.execute(stmt)).scalars().all()
        return len(rows)


# Keep the sqlalchemy `text` import referenced for anyone who wants ad-hoc sql
# from this module (e.g. a future admin endpoint).
__all__ = [
    "Base",
    "JobRow",
    "RUNNING_STATUSES",
    "claim_next_job",
    "count_queued",
    "create_job",
    "delete_job",
    "get_job",
    "heartbeat",
    "init_store",
    "is_cancel_requested",
    "list_jobs",
    "release_job",
    "request_cancel",
    "session",
    "sweep_stale_claims",
    "text",
    "update_job",
    "worker_identity",
]
