"""Per-job event bus.

Phase 2 splits the worker into its own container, so the in-memory
pub/sub that the API process uses to serve `/api/jobs/{id}/stream` can no
longer see events emitted by the worker. We swap the transport to an
append-only JSONL file per job, stored on the shared data volume:

    /data/jobs/{id}/events.jsonl

Publishers (runners / workers) append one JSON-encoded event per line.
Subscribers (the API's WebSocket handler) tail the file: they read any
existing lines (replay), then poll for new lines. A sidecar sentinel file
`events.done` marks the stream as closed so subscribers exit cleanly
instead of tailing forever after a job finishes.

Trade-offs this deliberately accepts:

  * Polling (≈200 ms) is fine for our volume and avoids a shared Redis.
  * One file per job means an orphaned job's events stay on disk until the
    job is deleted — we want that, for debugging.
  * IDs are assigned from the last line seen + 1, so they're stable within
    one file but can collide across job restarts (we never compare across
    jobs, so that's fine).

If we outgrow this, swap it for Redis streams without changing the call
sites — the `bus.publish` / `bus.subscribe` surface stays the same.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import AsyncIterator

from app.config import settings
from app.jobs.schema import JobEvent

log = logging.getLogger(__name__)

# Poll cadence for the file tail. Low enough that the UI still feels live,
# high enough that an idle API process isn't hammering the kernel. Override
# for tests via EVENTS_POLL_INTERVAL_S.
POLL_INTERVAL_S = float(os.environ.get("EVENTS_POLL_INTERVAL_S", "0.2"))


def _events_path(job_id: str) -> Path:
    return settings.job_dir(job_id) / "events.jsonl"


def _done_path(job_id: str) -> Path:
    return settings.job_dir(job_id) / "events.done"


class EventBus:
    """Per-job pub/sub backed by a JSONL file on the shared volume.

    The surface mirrors the old in-memory bus so existing callers (`runner`,
    `main` WebSocket handler, mesh/reexport endpoints) can stay unchanged.
    """

    def __init__(self) -> None:
        # Global lock serialises filesystem writes inside one process. Across
        # processes we rely on append-only writes with small payloads being
        # atomic up to PIPE_BUF (4 KiB on Linux), which is more than enough
        # for a single JobEvent line.
        self._write_lock = asyncio.Lock()
        self._next_ids: dict[str, int] = {}

    async def publish(self, event: JobEvent) -> JobEvent:
        path = _events_path(event.job_id)
        async with self._write_lock:
            next_id = self._next_ids.get(event.job_id)
            if next_id is None:
                next_id = _scan_last_id(path) + 1
            event.id = next_id
            self._next_ids[event.job_id] = next_id + 1

            path.parent.mkdir(parents=True, exist_ok=True)
            line = event.model_dump_json() + "\n"
            # O_APPEND guarantees the write is atomic for payloads ≤ PIPE_BUF,
            # so concurrent publishers (API + worker) can't interleave bytes.
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
        return event

    async def close(self, job_id: str) -> None:
        """Mark the stream closed so tailing subscribers exit their loop."""
        done = _done_path(job_id)
        done.parent.mkdir(parents=True, exist_ok=True)
        done.touch(exist_ok=True)

    def history(self, job_id: str) -> list[JobEvent]:
        """Return every event recorded for this job so far.

        Cheap for typical runs (events.jsonl tops out in the low megabytes).
        Used by the WS handler on connect so the client gets everything it
        missed while the tab was closed.
        """
        return _read_all(_events_path(job_id))

    async def subscribe(self, job_id: str) -> AsyncIterator[JobEvent]:
        """Replay + tail events for one job.

        Yields every existing event, then every new event appended until
        `events.done` appears. Safe to connect before the job has produced
        anything — the method waits for the file to show up.
        """
        path = _events_path(job_id)
        done = _done_path(job_id)

        offset = 0
        try:
            while True:
                # Replay whatever has been written since our last read.
                events, offset = _read_from(path, offset)
                for ev in events:
                    yield ev

                if done.exists() and offset >= _file_size(path):
                    return

                # Let cancellation propagate while we sleep.
                await asyncio.sleep(POLL_INTERVAL_S)
        except asyncio.CancelledError:
            return


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _scan_last_id(path: Path) -> int:
    """Recover the last event id so publishers can assign monotonically.

    Called once per process-lifetime per job: on first publish we read the
    tail of the file to figure out where to continue. Cheap because we only
    look at the last line.
    """
    import json as _json

    if not path.exists():
        return 0
    try:
        # Read the last 64 KiB — ample for one JSON line. Avoids slurping
        # the whole file for a long-running job.
        size = path.stat().st_size
        with path.open("rb") as fh:
            fh.seek(max(0, size - 65536))
            tail = fh.read().splitlines()
        for raw in reversed(tail):
            if not raw.strip():
                continue
            try:
                payload = _json.loads(raw.decode("utf-8", errors="replace"))
            except _json.JSONDecodeError:
                continue
            return int(payload.get("id") or 0)
    except OSError as exc:
        log.warning("failed to scan last event id for %s: %s", path, exc)
    return 0


def _read_from(path: Path, offset: int) -> tuple[list[JobEvent], int]:
    """Read new JSONL events starting at `offset`.

    Returns the events found and the new byte offset. If the file doesn't
    exist yet, returns `([], offset)` so the caller keeps polling.
    """
    import json as _json

    if not path.exists():
        return [], offset

    events: list[JobEvent] = []
    try:
        with path.open("rb") as fh:
            fh.seek(offset)
            chunk = fh.read()
            new_offset = fh.tell()
        if not chunk:
            return [], new_offset
        # Defer the last partial line (no trailing newline) so we don't
        # publish half of an event and then a phantom duplicate.
        if not chunk.endswith(b"\n"):
            last_nl = chunk.rfind(b"\n")
            if last_nl < 0:
                return [], offset  # nothing complete yet
            usable = chunk[: last_nl + 1]
            new_offset = offset + len(usable)
            chunk = usable

        for raw in chunk.splitlines():
            if not raw.strip():
                continue
            try:
                payload = _json.loads(raw.decode("utf-8", errors="replace"))
            except _json.JSONDecodeError as exc:
                log.warning("skipping corrupt event line in %s: %s", path, exc)
                continue
            try:
                events.append(JobEvent.model_validate(payload))
            except Exception as exc:  # noqa: BLE001
                log.warning("skipping invalid event in %s: %s", path, exc)
                continue
        return events, new_offset
    except OSError as exc:
        log.warning("failed to read events from %s: %s", path, exc)
        return [], offset


def _read_all(path: Path) -> list[JobEvent]:
    events, _ = _read_from(path, 0)
    return events


bus = EventBus()
