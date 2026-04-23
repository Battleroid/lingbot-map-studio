"""End-to-end test of the MASt3R-SLAM processor via the simulated tracker.

Goals:
  * Prove `SlamProcessor.run` glues ingest → track → export together
    without touching GPU code.
  * Prove the artifact shape matches what the frontend expects:
    `pose_graph.json` with a non-empty keyframes list, `keyframes.jsonl`
    with one line per keyframe, `reconstruction.ply` with a non-zero
    vertex count, `camera_path.json` for the live preview.

The simulated tracker in `app.processors.slam.tracker` stands in for the
real CUDA backends — same session contract, just plausibly-wrong numbers.

Run: `pytest worker/tests/test_slam_simulated.py -q`.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest


async def _exercise_slam_pipeline(
    tmp_data_dir: Path, synthetic_frames: Path
) -> dict:
    from app.jobs import store
    from app.jobs.cancel import CancelToken
    from app.jobs.schema import Mast3rSlamConfig
    from app.processors.base import JobContext
    from app.processors.slam.mast3r_slam import Mast3rSlamProcessor

    job_id = "jobtest0001"
    job_dir = tmp_data_dir / "jobs" / job_id
    frames_dir = job_dir / "frames"
    artifacts_dir = job_dir / "artifacts"
    frames_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Copy synthetic frames into the job's frames dir — we skip the real
    # ingest (ffmpeg) by swapping out `_ingest` below.
    for p in sorted(synthetic_frames.glob("*.png")):
        (frames_dir / p.name).write_bytes(p.read_bytes())

    cfg = Mast3rSlamConfig(
        # Default to translation keyframing so we don't need to pre-write
        # frame_scores.jsonl for this smoke.
        keyframe_policy="translation",
        max_frames=None,
        stride=1,
        partial_snapshot_every=4,
        run_poisson_mesh=False,
    )
    events: list = []

    async def publish(event):
        events.append(event)
        return event

    async def set_status(_status):
        return None

    async def set_frames_total(_n):
        return None

    ctx = JobContext(
        job_id=job_id,
        uploads=[],
        config=cfg,
        job_dir=job_dir,
        frames_dir=frames_dir,
        artifacts_dir=artifacts_dir,
        cancel=CancelToken(),
        publish=publish,
        set_status=set_status,
        set_frames_total=set_frames_total,
    )

    processor = Mast3rSlamProcessor()

    # Patch `_ingest` so we don't run ffmpeg on a synthetic PNG sequence;
    # the frames are already on disk, just report the count.
    async def _fake_ingest(_ctx):
        count = len(sorted(frames_dir.glob("*.png")))
        await ctx.set_frames_total(count)
        return count

    processor._ingest = _fake_ingest  # type: ignore[assignment]

    result = await processor.run(ctx)
    return {
        "result": result,
        "events": events,
        "artifacts_dir": artifacts_dir,
    }


@pytest.mark.asyncio
async def test_mast3r_slam_end_to_end_simulated(tmp_data_dir, synthetic_frames):
    outcome = await _exercise_slam_pipeline(tmp_data_dir, synthetic_frames)
    result = outcome["result"]
    artifacts_dir: Path = outcome["artifacts_dir"]

    # The processor declared it's a MASt3R run and reported keyframes.
    assert result.extras["backend"] == "mast3r_slam"
    assert result.extras["keyframes"] > 0

    # Artifact manifest lines up with the on-disk files.
    names = {a.name for a in result.artifacts}
    assert {"pose_graph.json", "keyframes.jsonl", "camera_path.json"}.issubset(names)
    # Non-trajectory-only: reconstruction.ply is produced.
    assert "reconstruction.ply" in names

    pose_graph = json.loads((artifacts_dir / "pose_graph.json").read_text())
    assert pose_graph["backend"] == "mast3r_slam"
    assert pose_graph["n_keyframes"] > 0
    assert len(pose_graph["keyframes"]) == pose_graph["n_keyframes"]
    # Each keyframe row carries a translation + quaternion + a source_frame
    # index (used by gsplat to map back to the source clip).
    for kf in pose_graph["keyframes"]:
        assert len(kf["t"]) == 3
        assert len(kf["q"]) == 4
        assert "source_frame" in kf

    keyframes_jsonl = (artifacts_dir / "keyframes.jsonl").read_text().splitlines()
    assert len(keyframes_jsonl) == pose_graph["n_keyframes"]

    # reconstruction.ply has a non-zero vertex count in the header.
    ply_bytes = (artifacts_dir / "reconstruction.ply").read_bytes()
    header = ply_bytes.split(b"end_header\n", 1)[0].decode("ascii")
    vertex_line = next(
        (ln for ln in header.splitlines() if ln.startswith("element vertex")), ""
    )
    assert vertex_line, "reconstruction.ply missing vertex header"
    n_verts = int(vertex_line.split()[-1])
    assert n_verts > 0


@pytest.mark.asyncio
async def test_slam_respects_cancellation(tmp_data_dir, synthetic_frames):
    """Flipping the cancel token mid-track should unwind cleanly."""
    from app.jobs.cancel import CancelToken, JobCancelled
    from app.jobs.schema import Mast3rSlamConfig
    from app.processors.base import JobContext
    from app.processors.slam.mast3r_slam import Mast3rSlamProcessor

    job_id = "jobcancel001"
    job_dir = tmp_data_dir / "jobs" / job_id
    frames_dir = job_dir / "frames"
    artifacts_dir = job_dir / "artifacts"
    frames_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(synthetic_frames.glob("*.png")):
        (frames_dir / p.name).write_bytes(p.read_bytes())

    cancel = CancelToken()

    async def publish(event):
        # Flip cancel as soon as the SLAM stage starts tracking.
        if getattr(event, "stage", None) == "slam":
            cancel.cancel("test: stopping early")
        return event

    ctx = JobContext(
        job_id=job_id,
        uploads=[],
        config=Mast3rSlamConfig(
            keyframe_policy="translation",
            partial_snapshot_every=0,
            run_poisson_mesh=False,
        ),
        job_dir=job_dir,
        frames_dir=frames_dir,
        artifacts_dir=artifacts_dir,
        cancel=cancel,
        publish=publish,
        set_status=lambda _s: asyncio.sleep(0),
        set_frames_total=lambda _n: asyncio.sleep(0),
    )

    processor = Mast3rSlamProcessor()

    async def _fake_ingest(_ctx):
        return len(sorted(frames_dir.glob("*.png")))

    processor._ingest = _fake_ingest  # type: ignore[assignment]

    with pytest.raises(JobCancelled):
        await processor.run(ctx)
