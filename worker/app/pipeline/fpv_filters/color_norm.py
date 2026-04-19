"""Per-frame colour normalisation (grey-world white balance).

Runs after ffmpeg extraction, on the PNG frames in-place. Uses a
grey-world assumption: each channel's mean across a frame should be
roughly equal. Analog/VHS FPV footage is usually strongly green- or
magenta-tinted; a grey-world scale + a light per-frame histogram stretch
recovers natural colour cheaply.

Kept dependency-light (numpy + cv2 only, both already installed in the
worker images). No torch, no GPU.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Callable

from app.jobs.schema import JobEvent

log = logging.getLogger(__name__)

PublishFn = Callable[[JobEvent], "asyncio.Future | None"]


def _process_frame(img):  # type: ignore[no-untyped-def]
    import numpy as np

    arr = img.astype("float32")
    # Grey-world scales: normalise each BGR channel mean to the overall mean.
    means = arr.reshape(-1, 3).mean(axis=0)
    target = means.mean()
    if float(target) <= 1e-3:
        return img  # essentially black frame, skip
    scales = target / np.maximum(means, 1e-3)
    arr *= scales[None, None, :]
    # Light histogram stretch: clip the 1st/99th percentile, rescale to 0..255.
    low, high = np.percentile(arr, (1.0, 99.0))
    if float(high - low) < 1.0:
        return arr.clip(0, 255).astype("uint8")
    arr = (arr - low) * (255.0 / (high - low))
    return arr.clip(0, 255).astype("uint8")


async def apply(
    job_id: str,
    frames_dir: Path,
    publish: PublishFn,
) -> int:
    """Rewrite every PNG under `frames_dir` with grey-world white-balance."""
    import cv2

    frame_paths = sorted(frames_dir.glob("*.png"))
    if not frame_paths:
        return 0

    await _publish(
        publish,
        JobEvent(
            job_id=job_id,
            stage="ingest",
            message=f"color_norm: normalising {len(frame_paths)} frame(s)",
        ),
    )

    def _run_all() -> int:
        touched = 0
        for p in frame_paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            out = _process_frame(img)
            cv2.imwrite(str(p), out)
            touched += 1
        return touched

    touched = await asyncio.to_thread(_run_all)
    await _publish(
        publish,
        JobEvent(
            job_id=job_id,
            stage="ingest",
            message=f"color_norm: done ({touched} frames)",
            progress=1.0,
        ),
    )
    return touched


async def _publish(publish: PublishFn, event: JobEvent) -> None:
    res = publish(event)
    if asyncio.iscoroutine(res):
        await res
