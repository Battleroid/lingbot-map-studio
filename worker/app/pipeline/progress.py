from __future__ import annotations

import asyncio
import contextlib
import io
import sys
import threading
from typing import Callable

from app.jobs.schema import JobEvent


class LineStream(io.TextIOBase):
    """Text stream that forwards complete lines to `on_line`."""

    def __init__(self, on_line: Callable[[str], None]) -> None:
        self._buf: list[str] = []
        self._on_line = on_line

    def writable(self) -> bool:  # type: ignore[override]
        return True

    def write(self, s: str) -> int:  # type: ignore[override]
        if not s:
            return 0
        self._buf.append(s)
        joined = "".join(self._buf)
        if "\n" in joined or "\r" in joined:
            parts = joined.replace("\r", "\n").split("\n")
            for line in parts[:-1]:
                if line.strip():
                    self._on_line(line)
            self._buf = [parts[-1]]
        return len(s)

    def flush(self) -> None:  # type: ignore[override]
        tail = "".join(self._buf).strip()
        if tail:
            self._on_line(tail)
        self._buf = []


@contextlib.contextmanager
def capture_stdio(
    job_id: str,
    publish: Callable[[JobEvent], "asyncio.Future | None"],
    stage: str,
    loop: asyncio.AbstractEventLoop,
):
    """Redirect stdout/stderr into the job event bus while the block runs.

    Safe to use around synchronous ML code executed via `asyncio.to_thread`.
    """

    def _forward(level: str):
        def _on_line(line: str) -> None:
            ev = JobEvent(job_id=job_id, stage=stage, level=level, message=line)
            try:
                asyncio.run_coroutine_threadsafe(
                    _publish_async(publish, ev), loop
                )
            except Exception:
                pass

        return _on_line

    out = LineStream(_forward("stdout"))
    err = LineStream(_forward("stderr"))
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            yield
        finally:
            out.flush()
            err.flush()


async def _publish_async(publish, event: JobEvent) -> None:
    res = publish(event)
    if asyncio.iscoroutine(res):
        await res
