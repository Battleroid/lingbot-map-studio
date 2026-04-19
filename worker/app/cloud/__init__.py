"""Cloud execution package.

Phase R1 introduces the plumbing that lets a worker run against a remote
studio over HTTPS. The shape of the abstraction is deliberately the
mirror image of what the local claim loop + runner already do: claim a
job, publish events, write artifacts, observe cancellation, finalize.

Swap `LocalJobSource` for `HttpJobSource` in `worker_main.py` and the
same `Processor.run(ctx)` code path runs unchanged in a rented pod.

Only `sources.py` lands in this slice; the broker, dispatcher, HTTP
source, storage backends, and provider adapters arrive in later slices.
"""

from __future__ import annotations

from app.cloud.sources import ClaimedJob, JobSource, LocalJobSource

__all__ = ["ClaimedJob", "JobSource", "LocalJobSource"]
