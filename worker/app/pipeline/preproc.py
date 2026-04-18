from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Callable

import numpy as np

from app.jobs.schema import JobConfig, JobEvent

log = logging.getLogger(__name__)

PublishFn = Callable[[JobEvent], "asyncio.Future | None"]


def build_ingest_filters(config: JobConfig) -> str:
    """Compose the -vf chain for frame extraction.

    Order matters: fisheye unwrap first (geometric), then temporal denoise,
    then deflicker, then fps resample. OSD masking runs after extraction in
    Python since it needs a whole-sequence stddev pass.
    """
    filters: list[str] = []

    if config.preproc_fisheye:
        in_fov = max(60.0, min(180.0, config.fisheye_in_fov))
        out_fov = max(40.0, min(140.0, config.fisheye_out_fov))
        # v360 syntax: input=fisheye:output=flat with in/out FOV controls.
        filters.append(
            f"v360=input=fisheye:output=flat:"
            f"ih_fov={in_fov}:iv_fov={in_fov}:d_fov={out_fov}"
        )

    if config.preproc_denoise:
        # hqdn3d args: luma_spatial:chroma_spatial:luma_tmp:chroma_tmp
        filters.append("hqdn3d=4:3:6:4")
        # deflicker in per-frame max mode, 5-frame window.
        filters.append("deflicker=mode=pm:size=5")

    filters.append(f"fps={config.fps}")
    return ",".join(filters)


def _compute_osd_mask(
    frame_paths: list[Path],
    samples: int,
    std_threshold: float,
    dilate: int,
) -> np.ndarray | None:
    import cv2

    if len(frame_paths) < 3:
        return None

    step = max(1, len(frame_paths) // max(1, samples))
    selected = frame_paths[::step][:samples]
    if len(selected) < 3:
        return None

    first = cv2.imread(str(selected[0]))
    if first is None:
        return None
    h, w = first.shape[:2]
    n = len(selected)

    sum_img = np.zeros((h, w, 3), dtype=np.float64)
    sum_sq = np.zeros((h, w, 3), dtype=np.float64)
    for p in selected:
        img = cv2.imread(str(p))
        if img is None or img.shape[:2] != (h, w):
            continue
        arr = img.astype(np.float64)
        sum_img += arr
        sum_sq += arr * arr
    mean = sum_img / n
    var = np.maximum(sum_sq / n - mean * mean, 0.0)
    std = np.sqrt(var).mean(axis=-1)
    mask = (std < std_threshold).astype(np.uint8) * 255

    if dilate > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=dilate)
    return mask


async def _publish(publish: PublishFn, event: JobEvent) -> None:
    res = publish(event)
    if asyncio.iscoroutine(res):
        await res


async def apply_osd_mask(
    job_id: str,
    frames_dir: Path,
    config: JobConfig,
    publish: PublishFn,
) -> None:
    """Detect static overlay pixels and inpaint them out of every frame.

    Mask is saved to frames_dir.parent / "osd_mask.png" so the UI can surface it.
    """
    import cv2

    frame_paths = sorted(frames_dir.glob("*.png"))
    if not frame_paths:
        return

    await _publish(
        publish,
        JobEvent(
            job_id=job_id,
            stage="ingest",
            message=f"osd: computing mask from {min(config.osd_mask_samples, len(frame_paths))} samples",
        ),
    )
    mask = await asyncio.to_thread(
        _compute_osd_mask,
        frame_paths,
        config.osd_mask_samples,
        config.osd_mask_std_threshold,
        config.osd_mask_dilate,
    )
    if mask is None or not mask.any():
        await _publish(
            publish,
            JobEvent(
                job_id=job_id,
                stage="ingest",
                message="osd: no static overlay detected",
            ),
        )
        return

    coverage = mask.astype(bool).mean() * 100.0
    mask_out = frames_dir.parent / "osd_mask.png"
    cv2.imwrite(str(mask_out), mask)
    await _publish(
        publish,
        JobEvent(
            job_id=job_id,
            stage="ingest",
            message=f"osd: mask {mask.shape[1]}x{mask.shape[0]}, {coverage:.1f}% of frame",
            data={"mask": mask_out.name, "coverage": coverage},
        ),
    )

    def _inpaint_one(path: Path) -> None:
        img = cv2.imread(str(path))
        if img is None:
            return
        out = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        cv2.imwrite(str(path), out)

    total = len(frame_paths)

    def _apply_all() -> None:
        for i, p in enumerate(frame_paths):
            _inpaint_one(p)
            if i and i % max(1, total // 20) == 0:
                pct = i / total
                log.info("osd inpaint %d/%d", i, total)

    await _publish(
        publish,
        JobEvent(
            job_id=job_id,
            stage="ingest",
            message=f"osd: inpainting {total} frames...",
        ),
    )
    await asyncio.to_thread(_apply_all)
    await _publish(
        publish,
        JobEvent(
            job_id=job_id,
            stage="ingest",
            message="osd: inpaint complete",
            progress=1.0,
        ),
    )
