"""End-to-end test for the gsplat processor chaining off a SLAM job.

Chains:
  1. Create a completed SLAM job row in the store with its artifacts on
     disk (from a previous simulated run).
  2. Run `GsplatProcessor.run` against a `GsplatConfig(source_job_id=…)`.
  3. Assert the training loop produced a non-empty `splat.ply`, a
     `training_log.jsonl` with one row per iteration, a `cameras.json`,
     and a `splat.sogs` placeholder.

The simulated trainer in `app.processors.gsplat.trainer` keeps this
CPU-only. Run: `pytest worker/tests/test_gsplat_from_slam.py -q`.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest


async def _seed_ready_slam_job(
    data_dir: Path, synthetic_frames: Path, job_id: str = "slamsource01"
) -> str:
    """Write a plausible 'ready' SLAM job into the store + artifacts dir."""
    from app.jobs import store
    from app.jobs.schema import Artifact, Job, Mast3rSlamConfig
    from app.processors.slam import export as slam_export

    await store.init_store()

    # Materialise frames + artifacts on disk.
    frames_dir = data_dir / "jobs" / job_id / "frames"
    artifacts_dir = data_dir / "jobs" / job_id / "artifacts"
    frames_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(synthetic_frames.glob("*.png")):
        (frames_dir / p.name).write_bytes(p.read_bytes())

    # Fake trajectory + cloud good enough for gsplat's init reader.
    n_kf = 6
    poses = np.tile(np.eye(4), (n_kf, 1, 1))
    for i in range(n_kf):
        poses[i, :3, 3] = [0.1 * i, 0.0, 0.05 * i]

    rng = np.random.default_rng(0)
    points = np.concatenate(
        [
            rng.uniform(-1.0, 1.0, size=(256, 3)),
            rng.integers(50, 230, size=(256, 3)).astype(float),
        ],
        axis=1,
    )

    slam_export.write_all(
        artifacts_dir,
        poses=poses,
        keyframe_indices=list(range(n_kf)),
        selected_indices=list(range(n_kf)),
        points=points,
        intrinsics=np.array([[80.0, 0, 48.0], [0, 80.0, 32.0], [0, 0, 1]]),
        backend_id="mast3r_slam",
        trajectory_only=False,
    )

    # Persist the job record so `get_job(source_job_id)` finds it.
    cfg = Mast3rSlamConfig()
    artifacts = [
        Artifact(name="reconstruction.ply", kind="ply"),
        Artifact(name="pose_graph.json", kind="pose_graph_json"),
        Artifact(name="camera_path.json", kind="json"),
        Artifact(name="keyframes.jsonl", kind="keyframes_jsonl"),
    ]
    now = datetime.now(timezone.utc)
    job = Job(
        id=job_id,
        status="ready",
        config=cfg,
        artifacts=artifacts,
        created_at=now,
        updated_at=now,
    )
    await store.create_job(job, worker_class="slam")
    # create_job sets status to whatever the Job passed in; no further
    # update needed.
    return job_id


@pytest.mark.asyncio
async def test_gsplat_trains_from_slam_source(
    tmp_data_dir: Path, synthetic_frames: Path
):
    from app.jobs.cancel import CancelToken
    from app.jobs.schema import GsplatConfig
    from app.processors.base import JobContext
    from app.processors.gsplat.trainer import GsplatProcessor

    src_id = await _seed_ready_slam_job(tmp_data_dir, synthetic_frames)

    gs_job_id = "gsplatjob001"
    job_dir = tmp_data_dir / "jobs" / gs_job_id
    artifacts_dir = job_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Keep the loop small — we only need to prove the wiring. The splat
    # state updates deterministically per step, so 8 iters is enough.
    cfg = GsplatConfig(
        source_job_id=src_id,
        iterations=8,
        densify_interval=4,
        prune_interval=4,
        preview_every_iters=4,
        preview_max_gaussians=1_000,
    )

    events: list = []

    async def publish(event):
        events.append(event)
        return event

    ctx = JobContext(
        job_id=gs_job_id,
        uploads=[],
        config=cfg,
        job_dir=job_dir,
        frames_dir=job_dir / "frames",
        artifacts_dir=artifacts_dir,
        cancel=CancelToken(),
        publish=publish,
        set_status=_noop,
        set_frames_total=_noop,
    )

    result = await GsplatProcessor().run(ctx)

    names = {a.name for a in result.artifacts}
    assert {"splat.ply", "cameras.json", "splat.sogs"}.issubset(names)
    assert result.extras["source_job_id"] == src_id
    assert result.extras["final_gaussians"] > 0

    # splat.ply has a valid 3DGS header with every expected property.
    head = (artifacts_dir / "splat.ply").read_bytes().split(b"end_header\n", 1)[0].decode()
    for prop in ("f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "rot_0"):
        assert prop in head, f"splat.ply missing {prop}"
    n_verts_line = next(ln for ln in head.splitlines() if ln.startswith("element vertex"))
    assert int(n_verts_line.split()[-1]) > 0

    # training_log.jsonl is present and has one row per iteration.
    log_path = artifacts_dir / "training_log.jsonl"
    assert log_path.exists()
    rows = [json.loads(ln) for ln in log_path.read_text().splitlines() if ln.strip()]
    assert len(rows) == cfg.iterations
    # Monotonic iteration counter with plausible metrics.
    assert [r["iter"] for r in rows] == list(range(1, cfg.iterations + 1))
    assert rows[-1]["psnr"] > rows[0]["psnr"]

    # At least one partial_splat_*.ply was emitted mid-run.
    partials = sorted(artifacts_dir.glob("partial_splat_*.ply"))
    assert partials, "trainer did not emit any preview snapshots"

    # cameras.json was written with the source backend noted.
    cameras = json.loads((artifacts_dir / "cameras.json").read_text())
    assert cameras["backend"] == "mast3r_slam"
    assert isinstance(cameras["cameras"], list)


@pytest.mark.asyncio
async def test_gsplat_refuses_non_ready_source(tmp_data_dir: Path):
    """Safety: `resolve_inputs` has to reject mid-run sources rather than
    letting training fire on an incomplete artifact set."""
    from datetime import datetime, timezone

    from app.jobs import store
    from app.jobs.schema import Artifact, Job, Mast3rSlamConfig
    from app.processors.gsplat import io as splat_io

    await store.init_store()

    src_id = "unreadyslam1"
    cfg = Mast3rSlamConfig()
    now = datetime.now(timezone.utc)
    await store.create_job(
        Job(
            id=src_id,
            status="slam",  # still running
            config=cfg,
            artifacts=[Artifact(name="x", kind="ply")],
            created_at=now,
            updated_at=now,
        ),
        worker_class="slam",
    )
    job = await store.get_job(src_id)
    assert job is not None
    with pytest.raises(splat_io.GsplatInputsError):
        splat_io.resolve_inputs(job)


async def _noop(*_args, **_kwargs):
    return None
