"""Processor abstraction.

Every processing mode (lingbot, SLAM backends, gsplat training) implements
the `Processor` class below. The job runner builds a `JobContext`, resolves
the right processor from the registry based on `config.processor`, and awaits
`processor.run(ctx)`. The processor is responsible for producing artifacts
under `ctx.artifacts_dir` and publishing events via `ctx.publish`.

This file is deliberately dependency-light — it's imported by the API
container to validate registrations and by each worker container at dispatch
time. Heavy imports (torch, pymeshlab, the model libraries) live in the
concrete processor modules so they only load where they can actually run.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

from app.jobs.cancel import CancelToken
from app.jobs.schema import (
    AnyJobConfig,
    Artifact,
    JobEvent,
    JobStatus,
    ProcessorKind,
)


class PublishFn(Protocol):
    def __call__(self, event: JobEvent) -> Awaitable[JobEvent]: ...


class SetStatusFn(Protocol):
    def __call__(self, status: JobStatus) -> Awaitable[None]: ...


class SetFramesTotalFn(Protocol):
    def __call__(self, frames_total: int) -> Awaitable[None]: ...


@dataclass
class JobContext:
    """Everything a processor needs to run one job.

    Intentionally plain data — the runner constructs this, hands it to
    `Processor.run`, and inspects the returned `ProcessorResult` to update
    the DB. Processors never touch sqlite directly.
    """

    job_id: str
    uploads: list[Path]
    config: AnyJobConfig

    job_dir: Path
    frames_dir: Path
    artifacts_dir: Path

    cancel: CancelToken
    publish: PublishFn
    # Runner-provided hooks for state the DB owns. Processors call these at
    # stage boundaries rather than importing `store` directly, so each
    # processor module stays free of the whole DB/API surface.
    set_status: SetStatusFn
    set_frames_total: SetFramesTotalFn

    # Optional cross-processor handles. Populated by the runner when it has
    # the relevant services available — a processor that needs them and
    # finds None should raise a clear error, not silently skip the feature.
    vram_watchdog: Callable[[], Awaitable[None]] | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def check_cancel(self) -> None:
        """Raise JobCancelled if a stop has been requested. Processors call
        this at every natural checkpoint (between ingest/inference/export
        phases, between keyframes inside a loop, etc)."""
        from app.jobs.cancel import JobCancelled

        if self.cancel.cancelled:
            raise JobCancelled(self.cancel.reason)


@dataclass
class ProcessorResult:
    """Returned by `Processor.run` on successful completion.

    The runner writes `artifacts` onto the job row and publishes a final
    system event. `extras` gets merged into that event's `data` payload so
    processors can surface per-mode summary stats (PSNR, trajectory length,
    etc) without needing a bespoke event schema.
    """

    artifacts: list[Artifact] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)


class Processor(abc.ABC):
    """Abstract base class for every processing mode.

    Subclasses declare their identity via class attributes — the registry in
    `app.processors.__init__` reads these to build the dispatch table.
    """

    # Must be one of ProcessorId. Matches the `processor` discriminator on
    # the config subclass this processor accepts.
    id: str
    # Broader grouping used by the UI to pick the right viewer + tool panel.
    kind: ProcessorKind
    # Which worker-class container this processor is allowed to run in
    # (Phase 2). The API uses this to route new jobs.
    worker_class: str = "lingbot"
    # Declared artifact kinds the processor can emit. Used by the frontend
    # manifest endpoint to decide which viewer layers / tools to mount.
    supported_artifacts: frozenset[str] = frozenset()

    @abc.abstractmethod
    async def run(self, ctx: JobContext) -> ProcessorResult:
        """Execute one job end-to-end.

        Implementations are expected to:
          * publish events via `ctx.publish` (stage transitions, progress,
            live snapshots).
          * call `ctx.check_cancel()` at natural checkpoints.
          * write final artifacts into `ctx.artifacts_dir` and return them in
            the result.

        Exceptions propagate to the runner which handles OOM / cancel /
        failure bookkeeping. Processors should raise rather than swallow.
        """
        raise NotImplementedError
