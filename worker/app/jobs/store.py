from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

from sqlalchemy import JSON, DateTime, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import settings
from app.jobs.schema import Artifact, Job, JobConfig, JobStatus, JobSummary


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


@asynccontextmanager
async def session() -> AsyncIterator[AsyncSession]:
    assert _Session is not None, "init_store() not called"
    async with _Session() as s:
        yield s


def _row_to_job(row: JobRow) -> Job:
    return Job(
        id=row.id,
        status=row.status,  # type: ignore[arg-type]
        config=JobConfig.model_validate_json(row.config_json),
        uploads=json.loads(row.uploads_json),
        artifacts=[Artifact.model_validate(a) for a in json.loads(row.artifacts_json)],
        frames_total=row.frames_total,
        error=row.error,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


async def create_job(job: Job) -> None:
    async with session() as s:
        row = JobRow(
            id=job.id,
            status=job.status,
            config_json=job.config.model_dump_json(),
            uploads_json=json.dumps(job.uploads),
            artifacts_json=json.dumps([a.model_dump(mode="json") for a in job.artifacts]),
            frames_total=job.frames_total,
            error=job.error,
            created_at=job.created_at,
            updated_at=job.updated_at,
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
            out.append(
                JobSummary(
                    id=row.id,
                    status=row.status,  # type: ignore[arg-type]
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                    frames_total=row.frames_total,
                    artifact_count=len(artifacts),
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
