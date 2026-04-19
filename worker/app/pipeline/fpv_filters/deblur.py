"""Motion deblur.

Two options:

  * `"unsharp"` — classical unsharp-mask + gated application. Fast, CPU,
    safe to leave on for mildly-blurred clips.
  * `"nafnet"` — learned deblur with a small NAFNet checkpoint. Phase 3
    ships the plumbing only; the checkpoint fetch + inference hook land
    with a follow-up when the weights have been vetted for licensing.

Gate: per-frame Laplacian variance (proxy for sharpness) is compared to
the clip median. Frames below `deblur_sharpness_gate` * median get the
filter; sharp frames are left alone so we don't blur what's already
crisp. Keeps the median reading honest even when the clip is long.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Callable

from app.jobs.schema import JobEvent, PreprocFields

log = logging.getLogger(__name__)

PublishFn = Callable[[JobEvent], "asyncio.Future | None"]


def _variance_of_laplacian(gray) -> float:  # type: ignore[no-untyped-def]
    import cv2

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _unsharp(img):  # type: ignore[no-untyped-def]
    import cv2

    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=1.4)
    return cv2.addWeighted(img, 1.6, blurred, -0.6, 0)


def _run_unsharp(frames: list[Path], gate_frac: float) -> tuple[int, float]:
    """Score every frame, apply unsharp to those below `gate_frac * median`."""
    import cv2

    scores: list[tuple[Path, float]] = []
    for p in frames:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        scores.append((p, _variance_of_laplacian(img)))
    if not scores:
        return 0, 0.0

    sorted_vars = sorted(s for _, s in scores)
    median = sorted_vars[len(sorted_vars) // 2]
    threshold = median * gate_frac

    touched = 0
    for path, var in scores:
        if var >= threshold:
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        cv2.imwrite(str(path), _unsharp(img))
        touched += 1
    return touched, float(median)


async def apply(
    job_id: str,
    frames_dir: Path,
    cfg: PreprocFields,
    publish: PublishFn,
) -> int:
    mode = cfg.preproc_deblur
    if mode == "none":
        return 0
    frame_paths = sorted(frames_dir.glob("*.png"))
    if not frame_paths:
        return 0

    if mode == "unsharp":
        await _publish(
            publish,
            JobEvent(
                job_id=job_id,
                stage="ingest",
                message=f"deblur: unsharp, gate={cfg.deblur_sharpness_gate:.2f}",
            ),
        )
        touched, median = await asyncio.to_thread(
            _run_unsharp, frame_paths, cfg.deblur_sharpness_gate
        )
        await _publish(
            publish,
            JobEvent(
                job_id=job_id,
                stage="ingest",
                message=(
                    f"deblur: unsharp applied to {touched}/{len(frame_paths)} "
                    f"frame(s) (median Laplacian var={median:.1f})"
                ),
                progress=1.0,
                data={"mode": "unsharp", "touched": touched, "median_var": median},
            ),
        )
        return touched

    if mode == "nafnet":
        # Stub: Phase 3 ships the hook, the learned-model path is wired in a
        # follow-up (needs a checkpoint cache entry + CUDA inference). Fall
        # back to unsharp so the job still progresses.
        log.warning("nafnet deblur not implemented yet, falling back to unsharp")
        return await apply(
            job_id,
            frames_dir,
            cfg.model_copy(update={"preproc_deblur": "unsharp"}),
            publish,
        )

    return 0


async def _publish(publish: PublishFn, event: JobEvent) -> None:
    res = publish(event)
    if asyncio.iscoroutine(res):
        await res
