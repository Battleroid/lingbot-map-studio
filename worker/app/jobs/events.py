from __future__ import annotations

import asyncio
import itertools
from collections import deque
from typing import AsyncIterator

from app.config import settings
from app.jobs.schema import JobEvent


class EventBus:
    """Per-job pub/sub with replay buffer.

    Publishers call `publish(event)`; subscribers get an async iterator that
    yields every event in the replay buffer on connect, then every new event as
    it is published. When the job finishes, publishers should call `close(job_id)`
    so subscribers can terminate.
    """

    def __init__(self) -> None:
        self._buffers: dict[str, deque[JobEvent]] = {}
        self._subs: dict[str, set[asyncio.Queue[JobEvent | None]]] = {}
        self._closed: set[str] = set()
        self._ids = itertools.count(1)
        self._lock = asyncio.Lock()

    async def publish(self, event: JobEvent) -> JobEvent:
        async with self._lock:
            event.id = next(self._ids)
            buf = self._buffers.setdefault(
                event.job_id, deque(maxlen=settings.event_replay_size)
            )
            buf.append(event)
            for q in self._subs.get(event.job_id, set()):
                q.put_nowait(event)
        return event

    async def close(self, job_id: str) -> None:
        async with self._lock:
            self._closed.add(job_id)
            for q in self._subs.get(job_id, set()):
                q.put_nowait(None)

    def history(self, job_id: str) -> list[JobEvent]:
        buf = self._buffers.get(job_id)
        return list(buf) if buf else []

    async def subscribe(self, job_id: str) -> AsyncIterator[JobEvent]:
        q: asyncio.Queue[JobEvent | None] = asyncio.Queue()
        async with self._lock:
            self._subs.setdefault(job_id, set()).add(q)
            replay = list(self._buffers.get(job_id, []))
            already_closed = job_id in self._closed
        try:
            for ev in replay:
                yield ev
            if already_closed:
                return
            while True:
                item = await q.get()
                if item is None:
                    return
                yield item
        finally:
            async with self._lock:
                subs = self._subs.get(job_id)
                if subs is not None:
                    subs.discard(q)


bus = EventBus()
