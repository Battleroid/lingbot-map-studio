from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Callable

from app.config import settings
from app.jobs.schema import JobEvent

log = logging.getLogger(__name__)

PublishFn = Callable[[JobEvent], "asyncio.Future | None"]


@dataclass
class VramWatchState:
    """Shared state between the watchdog task and the inference hook.

    The hook (called per frame inside `model.forward`) inspects `tripped`
    before proceeding. If the watchdog flipped it, the hook raises and the
    inference thread unwinds cleanly instead of the kernel killing the process.
    """

    soft_limit_gb: float
    tripped: bool = False
    reason: str = ""
    peak_gb: float = 0.0
    samples: int = 0
    _stop: asyncio.Event = field(default_factory=asyncio.Event)

    def trip(self, reason: str) -> None:
        self.tripped = True
        self.reason = reason

    def stop(self) -> None:
        self._stop.set()


class VramLimitExceeded(RuntimeError):
    """Raised from the inference hook when the watchdog has tripped."""


def _read_vram_gb() -> tuple[float, float] | None:
    """Return (allocated_gb, total_gb) for device 0, or None if no CUDA."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        alloc = torch.cuda.memory_allocated(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return alloc, total
    except Exception:  # noqa: BLE001
        return None


async def run_vram_watchdog(
    job_id: str,
    state: VramWatchState,
    publish: PublishFn,
) -> None:
    """Poll GPU memory every N seconds, stream it, trip on soft-limit breach."""
    interval = max(0.5, settings.vram_watchdog_interval_s)
    tripped_announced = False
    while not state._stop.is_set():
        reading = _read_vram_gb()
        if reading is not None:
            alloc, total = reading
            state.samples += 1
            if alloc > state.peak_gb:
                state.peak_gb = alloc
            # Emit a structured event every tick — the UI shows this live.
            ev = JobEvent(
                job_id=job_id,
                stage="inference",
                level="debug",
                message=f"vram {alloc:.2f}/{total:.1f} GB (peak {state.peak_gb:.2f})",
                data={
                    "vram_allocated_gb": round(alloc, 3),
                    "vram_total_gb": round(total, 3),
                    "vram_peak_gb": round(state.peak_gb, 3),
                    "vram_soft_limit_gb": state.soft_limit_gb,
                },
            )
            try:
                res = publish(ev)
                if asyncio.iscoroutine(res):
                    await res
            except Exception:  # noqa: BLE001
                pass

            if alloc > state.soft_limit_gb and not state.tripped:
                reason = (
                    f"VRAM soft limit exceeded: {alloc:.2f} GB > "
                    f"{state.soft_limit_gb:.2f} GB — aborting job"
                )
                state.trip(reason)
                tripped_announced = True
                try:
                    warn = JobEvent(
                        job_id=job_id,
                        stage="inference",
                        level="error",
                        message=reason,
                        data={
                            "vram_allocated_gb": round(alloc, 3),
                            "vram_soft_limit_gb": state.soft_limit_gb,
                        },
                    )
                    res = publish(warn)
                    if asyncio.iscoroutine(res):
                        await res
                except Exception:  # noqa: BLE001
                    pass

        try:
            await asyncio.wait_for(state._stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            continue

    if state.tripped and not tripped_announced:
        log.info("watchdog stopped after trip: %s", state.reason)
