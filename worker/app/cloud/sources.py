"""`JobSource` — the one surface a worker touches to talk to the studio.

The local claim loop today reaches directly into `app.jobs.store` +
`app.jobs.events.bus` + `app.config.settings`. That works because the
worker and the API share a SQLite file and a data volume.

Remote workers don't. To make a rented pod look the same to the runner,
we hide every "talk to the studio" call behind an ABC with two
implementations:

  * `LocalJobSource` — delegates to `store` / `bus` / `settings` exactly
    as today. Preserves existing behaviour byte-for-byte; this lets us
    refactor `worker_main.py` + `runner.py` to consume the ABC without
    changing what happens at runtime.
  * `HttpJobSource` — arrives in a later slice; speaks to the broker
    endpoints that this file's shape is designed against.

Only the surface shared by both implementations lives here. Anything
local-only (the in-process cancel token dict, the BEGIN IMMEDIATE
transaction in `claim_next_job`) stays in `app.jobs.*`.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.jobs.schema import AnyJobConfig, Artifact, JobEvent, JobStatus


@dataclass(frozen=True)
class ClaimedJob:
    """The three things the runner needs to start a job.

    `uploads` is already resolved to concrete local paths. For the HTTP
    source those paths point into a per-job scratch dir the source
    populated from the broker's `GET /uploads/{name}` stream; for the
    local source they point straight at `/data/jobs/{id}/uploads/*`.
    """

    job_id: str
    config: AnyJobConfig
    uploads: list[Path]


class JobSource(abc.ABC):
    """Transport-agnostic contract between the worker and the studio.

    Every method here has exactly one local and one remote implementation.
    Keep it narrow: if a new operation is local-only, put it somewhere
    else. If it's remote-friendly it also belongs on the ABC so the local
    source can stub it trivially.
    """

    # --- job lifecycle ----------------------------------------------------

    @abc.abstractmethod
    async def claim_next(
        self, worker_class: str, worker_id: str
    ) -> Optional[ClaimedJob]:
        """Atomically claim one queued job for this worker class.

        Returns None if the queue is empty; otherwise the returned
        `uploads` list is already materialised on the local filesystem.
        """

    @abc.abstractmethod
    async def heartbeat(self, job_id: str, worker_id: str) -> None:
        """Bump the liveness timestamp so the orphan sweep leaves us alone."""

    @abc.abstractmethod
    async def release(self, job_id: str, worker_id: str) -> None:
        """Drop this worker's claim. Called in the runner's `finally`."""

    # --- state transitions ------------------------------------------------

    @abc.abstractmethod
    async def set_status(self, job_id: str, status: JobStatus) -> None: ...

    @abc.abstractmethod
    async def set_frames_total(self, job_id: str, frames_total: int) -> None: ...

    @abc.abstractmethod
    async def set_artifacts(
        self, job_id: str, artifacts: list[Artifact]
    ) -> None: ...

    @abc.abstractmethod
    async def set_error(self, job_id: str, error: str) -> None: ...

    # --- pubsub -----------------------------------------------------------

    @abc.abstractmethod
    async def publish_event(self, event: JobEvent) -> JobEvent: ...

    @abc.abstractmethod
    async def close_events(self, job_id: str) -> None:
        """Mark the event stream done so WS subscribers exit their tail."""

    # --- cancellation -----------------------------------------------------

    @abc.abstractmethod
    async def is_cancel_requested(self, job_id: str) -> bool: ...

    # --- filesystem handoff ----------------------------------------------

    @abc.abstractmethod
    def artifacts_dir(self, job_id: str) -> Path:
        """Directory the processor writes artifact files into.

        Local: the shared `/data/jobs/{id}/artifacts/` dir. Remote: a
        scratch dir on the rented pod; a sync task (later slice) uploads
        new/changed files to the broker.
        """

    @abc.abstractmethod
    def job_dir(self, job_id: str) -> Path:
        """Root dir for this job's working files (frames, artifacts, logs)."""

    @abc.abstractmethod
    def frames_dir(self, job_id: str) -> Path: ...


class LocalJobSource(JobSource):
    """Current behaviour, wrapped behind the ABC.

    Every method is a thin delegation to the same store/bus/settings call
    `runner.run_job` and `worker_main._run_forever` make today. The only
    transformation is `claim_next`: it reshapes the legacy
    `(id, config, upload_names)` tuple into a `ClaimedJob` whose
    `uploads` are already resolved to `Path`s so the HTTP source's output
    shape is uniform.
    """

    async def claim_next(
        self, worker_class: str, worker_id: str
    ) -> Optional[ClaimedJob]:
        from app.config import settings
        from app.jobs import store

        claim = await store.claim_next_job(worker_class, worker_id=worker_id)
        if claim is None:
            return None
        job_id, config, _upload_names = claim
        uploads_dir = settings.job_uploads(job_id)
        uploads = (
            sorted(uploads_dir.iterdir()) if uploads_dir.exists() else []
        )
        return ClaimedJob(job_id=job_id, config=config, uploads=uploads)

    async def heartbeat(self, job_id: str, worker_id: str) -> None:
        from app.jobs import store

        await store.heartbeat(job_id, worker_id=worker_id)

    async def release(self, job_id: str, worker_id: str) -> None:
        from app.jobs import store

        await store.release_job(job_id, worker_id=worker_id)

    async def set_status(self, job_id: str, status: JobStatus) -> None:
        from app.jobs import store

        await store.update_job(job_id, status=status)

    async def set_frames_total(self, job_id: str, frames_total: int) -> None:
        from app.jobs import store

        await store.update_job(job_id, frames_total=frames_total)

    async def set_artifacts(
        self, job_id: str, artifacts: list[Artifact]
    ) -> None:
        from app.jobs import store

        await store.update_job(job_id, artifacts=artifacts)

    async def set_error(self, job_id: str, error: str) -> None:
        from app.jobs import store

        await store.update_job(job_id, error=error)

    async def publish_event(self, event: JobEvent) -> JobEvent:
        from app.jobs.events import bus

        return await bus.publish(event)

    async def close_events(self, job_id: str) -> None:
        from app.jobs.events import bus

        await bus.close(job_id)

    async def is_cancel_requested(self, job_id: str) -> bool:
        from app.jobs import store

        return await store.is_cancel_requested(job_id)

    def artifacts_dir(self, job_id: str) -> Path:
        from app.config import settings

        path = settings.job_artifacts(job_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def job_dir(self, job_id: str) -> Path:
        from app.config import settings

        return settings.job_dir(job_id)

    def frames_dir(self, job_id: str) -> Path:
        from app.config import settings

        return settings.job_frames(job_id)
