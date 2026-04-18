from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Callable, Iterable

from app.jobs.schema import JobConfig, JobEvent

log = logging.getLogger(__name__)

ProgressFn = Callable[[JobEvent], "asyncio.Future | None"]


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
    config: JobConfig,
    publish: ProgressFn,
) -> int:
    """Extract frames from each video into one renumbered folder.

    Builds an ffmpeg filter chain from config: optional fisheye unwrap (v360),
    temporal denoise (hqdn3d) + deflicker, then the fps resample. OSD masking
    runs as a separate pass in Python after all frames are extracted.
    """
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

    if config.preproc_osd_mask:
        await apply_osd_mask(job_id, dest, config, publish)

    return counter
