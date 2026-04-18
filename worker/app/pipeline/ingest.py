from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Callable, Iterable

from app.jobs.schema import JobEvent

log = logging.getLogger(__name__)

ProgressFn = Callable[[JobEvent], "asyncio.Future | None"]


async def _publish(publish: ProgressFn, event: JobEvent) -> None:
    res = publish(event)
    if asyncio.iscoroutine(res) or isinstance(res, asyncio.Future):
        await res  # type: ignore[func-returns-value]


async def _run_ffmpeg(
    src: Path,
    dst_dir: Path,
    fps: float,
    publish: ProgressFn,
    job_id: str,
    stream_index: int,
    total_streams: int,
) -> int:
    """Extract frames from one video into dst_dir/%06d.png at the requested fps.

    Returns the number of frames extracted.
    """
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
        f"fps={fps}",
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
            message=f"ffmpeg {stream_index + 1}/{total_streams}: {src.name} @ {fps} fps",
            data={"cmd": " ".join(cmd)},
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
            JobEvent(
                job_id=job_id,
                stage="ingest",
                level="stderr",
                message=line,
            ),
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
    fps: float,
    publish: ProgressFn,
) -> int:
    """Extract frames from each video (in order) into a single renumbered folder.

    Result: dest/000000.png, dest/000001.png, ... contiguous across all inputs.
    """
    dest.mkdir(parents=True, exist_ok=True)
    sources = list(sources)
    counter = 0
    for i, src in enumerate(sources):
        tmp = dest.parent / f"_stream_{i}"
        if tmp.exists():
            for p in tmp.iterdir():
                p.unlink()
        else:
            tmp.mkdir(parents=True, exist_ok=True)
        await _run_ffmpeg(src, tmp, fps, publish, job_id, i, len(sources))
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
    return counter
