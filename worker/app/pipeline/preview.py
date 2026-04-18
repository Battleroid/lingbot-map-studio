from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


async def extract_frame(video: Path, out: Path, timestamp: float = 1.0) -> Path:
    """Extract a single PNG frame at `timestamp` seconds from a video."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        str(timestamp),
        "-i",
        str(video),
        "-frames:v",
        "1",
        str(out),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(f"ffmpeg extract failed: {stderr.decode(errors='replace')}")
    return out


async def apply_fisheye(src_png: Path, out_png: Path, in_fov: float, out_fov: float) -> Path:
    """Apply v360 fisheye→flat to a single PNG."""
    in_fov = max(60.0, min(180.0, in_fov))
    out_fov = max(40.0, min(140.0, out_fov))
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src_png),
        "-vf",
        f"v360=input=fisheye:output=flat:ih_fov={in_fov}:iv_fov={in_fov}:d_fov={out_fov}",
        str(out_png),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg v360 failed: {stderr.decode(errors='replace')}")
    return out_png


async def extract_sample_frames(
    video: Path,
    out_dir: Path,
    count: int,
    duration_s: Optional[float],
    fps_hint: Optional[float],
) -> list[Path]:
    """Extract `count` frames evenly from `video` for OSD mask sampling."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.glob("*.png"):
        p.unlink()
    # Sample rate that gives us ~count frames for the whole duration.
    duration = duration_s or 0
    rate = max(0.5, count / max(duration, 1.0)) if duration else (fps_hint or 1.0)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-vf",
        f"fps={rate}",
        "-frames:v",
        str(count),
        "-start_number",
        "0",
        str(out_dir / "%06d.png"),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg sample failed: {stderr.decode(errors='replace')}")
    return sorted(out_dir.glob("*.png"))


def _compute_osd_mask_sync(
    frames: list[Path],
    std_threshold: float,
    dilate: int,
    detect_text: bool = True,
    edge_persist_frac: float = 0.75,
) -> Optional[np.ndarray]:
    """Same two-signal algorithm as preproc._compute_osd_mask but for previews.

    Signal 1: low temporal stddev = truly static pixels (labels, boxes, logos).
    Signal 2: high temporal edge persistence = text regions, even when the
              digits themselves change frame-to-frame.
    """
    import cv2

    if len(frames) < 3:
        return None
    first = cv2.imread(str(frames[0]))
    if first is None:
        return None
    h, w = first.shape[:2]
    n = 0
    sum_img = np.zeros((h, w, 3), dtype=np.float64)
    sum_sq = np.zeros((h, w, 3), dtype=np.float64)
    edge_hits: Optional[np.ndarray] = (
        np.zeros((h, w), dtype=np.float64) if detect_text else None
    )
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    for p in frames:
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


async def render_osd_preview(
    video: Path,
    work_dir: Path,
    out_png: Path,
    samples: int,
    std_threshold: float,
    dilate: int,
    duration_s: Optional[float],
    fps_hint: Optional[float],
    detect_text: bool = True,
    edge_persist_frac: float = 0.75,
) -> dict:
    """Compute an OSD mask and render an overlay: mask shown as red on the first frame."""
    import cv2

    frames = await extract_sample_frames(
        video=video,
        out_dir=work_dir,
        count=samples,
        duration_s=duration_s,
        fps_hint=fps_hint,
    )
    if len(frames) < 3:
        raise RuntimeError(f"need >=3 sample frames, got {len(frames)}")

    mask = await asyncio.to_thread(
        _compute_osd_mask_sync,
        frames,
        std_threshold,
        dilate,
        detect_text,
        edge_persist_frac,
    )
    if mask is None:
        raise RuntimeError("mask computation returned None")

    first = cv2.imread(str(frames[0]))
    overlay = first.copy()
    # Red tint where mask is set.
    overlay[mask > 0] = (0, 0, 200)
    blended = cv2.addWeighted(first, 0.55, overlay, 0.45, 0)
    # Outline the mask boundary in pure red for visibility.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (0, 0, 255), 1)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), blended)

    coverage = float((mask > 0).mean() * 100.0)
    return {
        "path": str(out_png),
        "coverage": round(coverage, 2),
        "samples": len(frames),
        "width": mask.shape[1],
        "height": mask.shape[0],
    }
