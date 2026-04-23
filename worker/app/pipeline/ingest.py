from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Callable, Iterable

from app.jobs.schema import JobEvent, LingbotConfig, SlamConfig

log = logging.getLogger(__name__)

ProgressFn = Callable[[JobEvent], "asyncio.Future | None"]

# Config types that carry the PreprocFields mixin. Phase 3 stages accept
# either; the runner / processors hand one in based on the active mode.
PreprocCarrier = LingbotConfig | SlamConfig


async def _publish(publish: ProgressFn, event: JobEvent) -> None:
    res = publish(event)
    if asyncio.iscoroutine(res) or isinstance(res, asyncio.Future):
        await res  # type: ignore[func-returns-value]


async def _run_ffmpeg(
    src: Path,
    dst_dir: Path,
    filter_chain: str,
    publish: ProgressFn,
    job_id: str,
    stream_index: int,
    total_streams: int,
) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(dst_dir / "%06d.png")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "info",
        "-y",
        "-i",
        str(src),
        "-vf",
        filter_chain,
        "-vsync",
        "vfr",
        "-start_number",
        "0",
        pattern,
    ]
    await _publish(
        publish,
        JobEvent(
            job_id=job_id,
            stage="ingest",
            message=f"ffmpeg {stream_index + 1}/{total_streams}: {src.name} [{filter_chain}]",
            data={"cmd": " ".join(cmd), "filters": filter_chain},
        ),
    )
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert proc.stdout is not None

    async for raw in proc.stdout:
        line = raw.decode(errors="replace").rstrip()
        if not line:
            continue
        await _publish(
            publish,
            JobEvent(job_id=job_id, stage="ingest", level="stderr", message=line),
        )
    rc = await proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed on {src.name} with code {rc}")

    frames = sorted(dst_dir.glob("*.png"))
    count = len(frames)
    await _publish(
        publish,
        JobEvent(
            job_id=job_id,
            stage="ingest",
            message=f"extracted {count} frames from {src.name}",
            data={"source": src.name, "count": count},
        ),
    )
    return count


async def concat_videos_to_frames(
    job_id: str,
    sources: Iterable[Path],
    dest: Path,
    config: PreprocCarrier,
    publish: ProgressFn,
) -> int:
    """Extract frames from each video into one renumbered folder, then run
    the Phase 3 FPV preproc pipeline on the result.

    Stage order:

      1. Build the ffmpeg -vf chain from the config (fisheye, analog cleanup,
         denoise/deflicker, fps resample).
      2. Extract each source into a temp dir, renumber into `dest/`.
      3. Apply the Python post-extract stages (OSD mask + inpaint, color
         norm, rolling-shutter, deblur, keyframe score). Each stage is
         idempotent and skipped when its flag is off.
    """
    from app.pipeline.fpv_filters import (
        color_norm,
        deblur,
        keyframe_score,
        rolling_shutter,
    )
    from app.pipeline.preproc import apply_osd_mask, build_ingest_filters

    dest.mkdir(parents=True, exist_ok=True)
    sources = list(sources)
    counter = 0
    filter_chain = build_ingest_filters(config)
    for i, src in enumerate(sources):
        tmp = dest.parent / f"_stream_{i}"
        if tmp.exists():
            for p in tmp.iterdir():
                p.unlink()
        else:
            tmp.mkdir(parents=True, exist_ok=True)
        await _run_ffmpeg(src, tmp, filter_chain, publish, job_id, i, len(sources))
        for p in sorted(tmp.iterdir()):
            target = dest / f"{counter:06d}.png"
            p.rename(target)
            counter += 1
        tmp.rmdir()

    await _publish(
        publish,
        JobEvent(
            job_id=job_id,
            stage="ingest",
            message=f"concat complete: {counter} frames in {dest}",
            data={"frames_total": counter},
            progress=1.0,
        ),
    )

    # --- Python post-extract pipeline --------------------------------------

    if config.preproc_osd_mask:
        await apply_osd_mask(job_id, dest, config, publish)

    # Colour normalisation runs before any warping / deblur because those
    # stages assume reasonably normalised luma.
    if config.preproc_color_norm:
        await color_norm.apply(job_id, dest, publish)

    if config.preproc_rs_correction:
        await rolling_shutter.apply(
            job_id,
            dest,
            override_shear=config.rs_shear_px_per_row,
            publish=publish,
        )

    if config.preproc_deblur != "none":
        await deblur.apply(job_id, dest, config, publish)

    # Always write scores when the flag is on (regardless of other stages).
    # Cheap to produce and downstream backends that don't consume it just
    # ignore the file.
    if config.preproc_keyframe_score:
        await keyframe_score.write_scores(job_id, dest, publish)

    return counter
