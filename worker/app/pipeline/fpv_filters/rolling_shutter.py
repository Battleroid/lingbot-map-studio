"""Global rolling-shutter skew correction.

Analog FPV cams (and a lot of CMOS digital ones) have a rolling shutter
whose per-row readout lag shows up as a vertical shear on fast yaws /
rolls. A full per-row correction needs gyro data or a learned model; v1
handles the dominant failure case — a global y-shear — which covers most
of what you actually see on a yawing drone.

Approach:

  1. Estimate a single shear factor `s` (pixels of x-shift per row) for
     the clip. For each sampled frame we compute dense optical flow
     against the next frame, project it to rows, and fit
     `dx_per_row = s * row_index + b`. The median of `s` across samples
     is the clip-wide estimate.
  2. Apply the inverse affine warp to every frame in-place.

Skipped when the estimated shear falls below a small threshold — no
point warping when it'd be a ~0.5 pixel correction.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Callable, Optional

from app.jobs.schema import JobEvent

log = logging.getLogger(__name__)

PublishFn = Callable[[JobEvent], "asyncio.Future | None"]

# If the per-row-pixel shear is less than this, skip the warp entirely —
# applying a sub-pixel affine blurs more than it helps.
MIN_ABS_SHEAR = 0.02
# Sample at most this many frames for the shear estimate to cap cost.
MAX_SAMPLES = 60


def _estimate_shear(frames: list[Path], max_samples: int = MAX_SAMPLES) -> Optional[float]:
    import cv2
    import numpy as np

    if len(frames) < 2:
        return None
    step = max(1, len(frames) // max_samples)
    pairs = list(zip(frames[::step], frames[step::step]))
    if not pairs:
        return None

    shears: list[float] = []
    for a_path, b_path in pairs:
        a = cv2.imread(str(a_path), cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(str(b_path), cv2.IMREAD_GRAYSCALE)
        if a is None or b is None or a.shape != b.shape:
            continue
        flow = cv2.calcOpticalFlowFarneback(
            a, b, None, 0.5, 3, 21, 3, 5, 1.1, 0
        )
        # Mean dx per row.
        dx_per_row = flow[:, :, 0].mean(axis=1)
        # Fit dx = s * row + b, take slope. Robust to noise via median over
        # bootstraps inside the caller.
        rows = np.arange(dx_per_row.shape[0], dtype=np.float32)
        s, _ = np.polyfit(rows, dx_per_row, deg=1)
        shears.append(float(s))
    if not shears:
        return None
    # Median across samples → a robust clip-wide estimate.
    return float(sorted(shears)[len(shears) // 2])


def _apply_shear(frames: list[Path], shear: float) -> int:
    import cv2
    import numpy as np

    touched = 0
    for p in frames:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        # Inverse shear: dx = -shear * row. Build a 2×3 affine matrix.
        m = np.array([[1.0, -shear, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        # We want dx to depend on row, so use a shear-along-y matrix by
        # transposing: the y-component of the output is unchanged, the x is
        # shifted by shear*y. cv2.warpAffine interprets (2,3) as a forward map,
        # so M=[[1,-s,0],[0,1,0]] gives x' = x - s*y, i.e. the inverse warp.
        out = cv2.warpAffine(
            img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        cv2.imwrite(str(p), out)
        touched += 1
    return touched


async def apply(
    job_id: str,
    frames_dir: Path,
    override_shear: Optional[float],
    publish: PublishFn,
) -> int:
    """Estimate + apply a global y-shear correction. Returns frame count."""
    frame_paths = sorted(frames_dir.glob("*.png"))
    if not frame_paths:
        return 0

    if override_shear is not None:
        shear = float(override_shear)
        await _publish(
            publish,
            JobEvent(
                job_id=job_id,
                stage="ingest",
                message=f"rolling_shutter: using override shear {shear:.4f} px/row",
            ),
        )
    else:
        shear = await asyncio.to_thread(_estimate_shear, frame_paths) or 0.0
        await _publish(
            publish,
            JobEvent(
                job_id=job_id,
                stage="ingest",
                message=f"rolling_shutter: estimated shear {shear:.4f} px/row",
                data={"shear_px_per_row": shear},
            ),
        )

    if abs(shear) < MIN_ABS_SHEAR:
        await _publish(
            publish,
            JobEvent(
                job_id=job_id,
                stage="ingest",
                message="rolling_shutter: shear below threshold, skipping warp",
            ),
        )
        return 0

    touched = await asyncio.to_thread(_apply_shear, frame_paths, shear)
    await _publish(
        publish,
        JobEvent(
            job_id=job_id,
            stage="ingest",
            message=f"rolling_shutter: warped {touched} frame(s)",
            progress=1.0,
        ),
    )
    return touched


async def _publish(publish: PublishFn, event: JobEvent) -> None:
    res = publish(event)
    if asyncio.iscoroutine(res):
        await res
