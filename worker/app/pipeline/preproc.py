from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Callable

import numpy as np

from app.jobs.schema import JobEvent, LingbotConfig, SlamConfig

log = logging.getLogger(__name__)

PublishFn = Callable[[JobEvent], "asyncio.Future | None"]

# Any config that carries PreprocFields + an `fps` attr. Both lingbot and
# SLAM configs qualify; gsplat doesn't run ingest so it's excluded.
_IngestConfig = LingbotConfig | SlamConfig


def build_ingest_filters(config: _IngestConfig) -> str:
    """Compose the -vf chain for frame extraction.

    Order (matters):

      1. fisheye unwrap (geometric, changes pixel positions)
      2. analog cleanup (atadenoise) — before hqdn3d so chroma noise is gone
         first, avoiding hqdn3d eating detail trying to suppress it
      3. hqdn3d + paired deflicker (existing `preproc_denoise`)
      4. standalone deflicker (only if `preproc_deflicker` is on without
         `preproc_denoise` — analog_cleanup.ffmpeg_snippet handles that)
      5. fps resample

    OSD masking + all Phase-3 Python stages run after ffmpeg extraction —
    they need whole-sequence state (stddev, optical flow, Laplacian median).
    """
    from app.pipeline.fpv_filters import analog_cleanup

    filters: list[str] = []

    if config.preproc_fisheye:
        in_fov = max(60.0, min(180.0, config.fisheye_in_fov))
        out_fov = max(40.0, min(140.0, config.fisheye_out_fov))
        # v360 syntax: input=fisheye:output=flat with in/out FOV controls.
        filters.append(
            f"v360=input=fisheye:output=flat:"
            f"ih_fov={in_fov}:iv_fov={in_fov}:d_fov={out_fov}"
        )

    filters.extend(analog_cleanup.ffmpeg_snippet(config))

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
    detect_text: bool = True,
    edge_persist_frac: float = 0.75,
) -> np.ndarray | None:
    """Two-signal OSD mask:

    1. **Static pixels** — pixels whose temporal stddev across sampled frames
       is below `std_threshold`. Catches constant parts of overlays
       (label strings like "BAT:", box backgrounds, logos, icons).

    2. **Text-like persistence** (optional, when `detect_text=True`) — for each
       frame we Canny-edge it and dilate the edges by ~5px, then count how
       often each pixel lands inside that dilated edge map across all samples.
       Pixels that are near an edge in ≥ `edge_persist_frac` of frames are
       flagged. This catches CHANGING numeric HUD values ("12.4V" → "12.3V"):
       the specific glyph pixels differ frame-to-frame so signal (1) misses
       them, but the region is always edge-rich so signal (2) catches it.
       Scene edges move with the camera, so a scene pixel is near an edge
       only briefly and falls below the persistence threshold.

    The two masks are unioned, then dilated to grow over anti-aliased edges.
    """
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

    sum_img = np.zeros((h, w, 3), dtype=np.float64)
    sum_sq = np.zeros((h, w, 3), dtype=np.float64)
    edge_hits: np.ndarray | None = (
        np.zeros((h, w), dtype=np.float64) if detect_text else None
    )
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    n = 0
    for p in selected:
        img = cv2.imread(str(p))
        if img is None or img.shape[:2] != (h, w):
            continue
        arr = img.astype(np.float64)
        sum_img += arr
        sum_sq += arr * arr
        if edge_hits is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 80, 160)
            edges_dil = cv2.dilate(edges, edge_kernel, iterations=1)
            edge_hits += (edges_dil > 0).astype(np.float64)
        n += 1
    if n < 3:
        return None

    mean = sum_img / n
    var = np.maximum(sum_sq / n - mean * mean, 0.0)
    std = np.sqrt(var).mean(axis=-1)
    static_mask = (std < std_threshold).astype(np.uint8) * 255

    if edge_hits is not None:
        edge_frac = edge_hits / n
        text_mask = (edge_frac >= edge_persist_frac).astype(np.uint8) * 255
        mask = cv2.bitwise_or(static_mask, text_mask)
    else:
        mask = static_mask

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
    config: _IngestConfig,
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
        config.osd_detect_text,
        config.osd_edge_persist_frac,
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
