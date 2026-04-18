from __future__ import annotations

import asyncio
import json
import logging
from fractions import Fraction
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


async def probe_video(path: Path) -> dict[str, Any]:
    """Run ffprobe on a video file and return a normalized metadata dict.

    Keys: fps, duration_s, width, height, codec, pix_fmt, bitrate, total_frames,
          container, has_audio, size_bytes. Values may be None if ffprobe can't
          determine them.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        "-count_frames",  # accurate total_frames, more expensive
        str(path),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed on {path.name}: {stderr.decode(errors='replace')}"
        )
    data = json.loads(stdout.decode("utf-8"))

    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt = data.get("format") or {}

    fps = None
    if video:
        r = video.get("r_frame_rate") or video.get("avg_frame_rate") or "0/0"
        try:
            fps = float(Fraction(r)) if r and r != "0/0" else None
        except (ZeroDivisionError, ValueError):
            fps = None

    duration = None
    if fmt.get("duration"):
        try:
            duration = float(fmt["duration"])
        except (TypeError, ValueError):
            duration = None

    total_frames = None
    if video:
        nf = video.get("nb_read_frames") or video.get("nb_frames")
        if nf:
            try:
                total_frames = int(nf)
            except (TypeError, ValueError):
                total_frames = None
        elif fps and duration:
            total_frames = int(round(fps * duration))

    bitrate = None
    if fmt.get("bit_rate"):
        try:
            bitrate = int(fmt["bit_rate"])
        except (TypeError, ValueError):
            bitrate = None

    size = None
    if fmt.get("size"):
        try:
            size = int(fmt["size"])
        except (TypeError, ValueError):
            size = None
    if size is None:
        try:
            size = path.stat().st_size
        except OSError:
            size = None

    return {
        "fps": round(fps, 3) if fps else None,
        "duration_s": round(duration, 3) if duration else None,
        "width": video.get("width") if video else None,
        "height": video.get("height") if video else None,
        "codec": video.get("codec_name") if video else None,
        "pix_fmt": video.get("pix_fmt") if video else None,
        "bitrate": bitrate,
        "total_frames": total_frames,
        "container": fmt.get("format_name"),
        "has_audio": audio is not None,
        "size_bytes": size,
    }


def suggest_config(probes: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a partial JobConfig patch derived from probed metadata.

    Rules:
      - target ~10 fps for reconstruction; cap at source fps.
      - total frame estimate > 2500 → mode=windowed.
      - height <= 720 or bitrate < 8 Mbps → treat as low-fi: mask_sky on,
        conf_percentile 65, smaller num_scale_frames; else high-fi defaults.
    """
    if not probes:
        return {}

    # Pick the best source FPS we see; we'll subsample to target.
    # lingbot-map streams at ~20 FPS on 518px inputs per the paper — cap there.
    src_fps = max(
        (p["fps"] for p in probes if p.get("fps")),
        default=30.0,
    )
    target_fps = min(20.0, src_fps)

    total_duration = sum(p.get("duration_s") or 0.0 for p in probes)
    est_frames_20fps = int(round(target_fps * total_duration))
    # Cap total reconstruction frames at ~500 for streaming mode to stay
    # within ~20 GB VRAM headroom. Longer clips drop the sampling rate.
    if est_frames_20fps > 500 and total_duration > 0:
        target_fps = max(2.0, 500.0 / total_duration)

    est_frames = int(round(target_fps * total_duration))

    max_height = max((p.get("height") or 0) for p in probes)
    max_bitrate = max((p.get("bitrate") or 0) for p in probes)

    low_fi = max_height <= 720 or (max_bitrate and max_bitrate < 8_000_000)

    # Streaming mode's KV cache grows with keyframe count. Switch to windowed
    # for anything beyond ~300 frames so peak memory stays roughly flat.
    mode = "windowed" if est_frames > 300 else "streaming"

    patch: dict[str, Any] = {
        "fps": round(target_fps, 2),
        "mode": mode,
    }
    if low_fi:
        # Low-fi ~ analog FPV drone: assume noise + OSD overlay. Don't auto-enable
        # fisheye — it's destructive on a non-fisheye source, so make it opt-in.
        patch.update(
            mask_sky=True,
            conf_percentile=65,
            keyframe_interval=4,
            num_scale_frames=4,
            camera_num_iterations=2,
            preproc_denoise=True,
            preproc_osd_mask=True,
        )
    else:
        patch.update(
            mask_sky=False,
            conf_percentile=40,
            keyframe_interval=6,
            num_scale_frames=8,
            camera_num_iterations=4,
            preproc_denoise=False,
            preproc_osd_mask=False,
        )
    return patch
