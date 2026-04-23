"""Per-frame quality scoring → frame_scores.jsonl.

Emits a line per frame with:
  * `sharpness`      — variance of Laplacian. Higher = sharper.
  * `motion_px`      — mean optical-flow magnitude against the previous
                       frame. Proxy for inter-frame parallax / movement.
  * `quality`        — sharpness normalised by the clip median, so
                       scale-invariant across clips.

Consumers:

  * SLAM backends with `keyframe_policy="score_gated"` read this file to
    drop low-quality frames before keyframe selection.
  * The UI shows an aggregate sparkline of quality over time in the
    preproc preview.

Idempotent: re-running overwrites the JSONL.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Callable, Optional

from app.jobs.schema import JobEvent

log = logging.getLogger(__name__)

PublishFn = Callable[[JobEvent], "asyncio.Future | None"]


def _score_all(frames: list[Path]) -> list[dict]:
    """Compute sharpness + motion scores in order. Returns one row per frame."""
    import cv2

    rows: list[dict] = []
    prev_gray: Optional["cv2.Mat"] = None  # type: ignore[name-defined]
    for idx, path in enumerate(frames):
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            rows.append(
                {"index": idx, "name": path.name, "sharpness": 0.0, "motion_px": 0.0}
            )
            continue
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        motion_px = 0.0
        if prev_gray is not None and prev_gray.shape == gray.shape:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.5, 2, 15, 2, 5, 1.1, 0
            )
            mag = (flow[..., 0] ** 2 + flow[..., 1] ** 2) ** 0.5
            motion_px = float(mag.mean())
        rows.append(
            {
                "index": idx,
                "name": path.name,
                "sharpness": sharpness,
                "motion_px": motion_px,
            }
        )
        prev_gray = gray
    if not rows:
        return rows
    # Attach per-clip-normalised quality so consumers don't have to re-median.
    sharpness_sorted = sorted(r["sharpness"] for r in rows if r["sharpness"] > 0)
    if sharpness_sorted:
        median = sharpness_sorted[len(sharpness_sorted) // 2]
    else:
        median = 1.0
    for r in rows:
        r["quality"] = r["sharpness"] / max(median, 1e-6)
    return rows


async def write_scores(
    job_id: str,
    frames_dir: Path,
    publish: PublishFn,
) -> Path:
    """Score every frame and write `frame_scores.jsonl` next to the frames dir."""
    frame_paths = sorted(frames_dir.glob("*.png"))
    out = frames_dir.parent / "frame_scores.jsonl"
    if not frame_paths:
        out.write_text("")
        return out

    await _publish(
        publish,
        JobEvent(
            job_id=job_id,
            stage="ingest",
            message=f"keyframe_score: scoring {len(frame_paths)} frame(s)",
        ),
    )
    rows = await asyncio.to_thread(_score_all, frame_paths)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    if rows:
        qualities = [r["quality"] for r in rows]
        avg_q = sum(qualities) / len(qualities)
        min_q = min(qualities)
        await _publish(
            publish,
            JobEvent(
                job_id=job_id,
                stage="ingest",
                message=(
                    f"keyframe_score: {len(rows)} rows, "
                    f"avg quality {avg_q:.2f}, min {min_q:.2f}"
                ),
                progress=1.0,
                data={"rows": len(rows), "avg_quality": avg_q, "min_quality": min_q},
            ),
        )
    return out


async def _publish(publish: PublishFn, event: JobEvent) -> None:
    res = publish(event)
    if asyncio.iscoroutine(res):
        await res
