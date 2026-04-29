"""Pin the partial-snapshot event payload shape so the frontend's
viewer can find live preview URLs.

The frontend (`web/src/app/jobs/[id]/page.tsx`) walks the event stream
filtering on `ev.stage === "artifact"` and `ev.data.kind` —
`partial_ply` for the live point cloud, `partial_splat` for the live
splat. The actual artifact filename comes from `ev.data.name`.

Earlier the workers emitted `data={"partial": name}` with no `kind`
field. The frontend never matched, so partial snapshots never showed
up in the viewer — users only saw the final artifact at job
completion. These tests pin both payloads so a regression doesn't
silently re-break the live preview."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.asyncio
async def test_slam_partial_publish_emits_kind_and_name(tmp_data_dir):
    from app.jobs.cancel import CancelToken
    from app.jobs.schema import Mast3rSlamConfig
    from app.processors.base import JobContext
    from app.processors.slam.mast3r_slam import Mast3rSlamProcessor

    job_id = "slam-payload-001"
    job_dir = tmp_data_dir / "jobs" / job_id
    artifacts_dir = job_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    cfg = Mast3rSlamConfig()
    events: list = []

    async def publish(event):
        events.append(event)
        return event

    async def noop(*_a, **_k):
        return None

    ctx = JobContext(
        job_id=job_id,
        uploads=[],
        config=cfg,
        job_dir=job_dir,
        frames_dir=job_dir / "frames",
        artifacts_dir=artifacts_dir,
        cancel=CancelToken(),
        publish=publish,
        set_status=noop,
        set_frames_total=noop,
    )

    processor = Mast3rSlamProcessor()
    await processor._publish_partial(
        ctx,
        kf_idx=4,
        poses=[np.eye(4, dtype=np.float32)],
        points_chunks=[],
        trajectory_only=True,
    )

    artifact_events = [e for e in events if e.stage == "artifact"]
    assert artifact_events, "expected an artifact event from _publish_partial"
    payload = artifact_events[-1].data

    # Both `name` + `kind` must be set — the frontend's filter is
    # `data.kind === "partial_ply"` && reads `data.name`.
    assert payload.get("kind") == "partial_ply", payload
    assert isinstance(payload.get("name"), str), payload
    assert payload["name"].endswith(".ply"), payload


@pytest.mark.asyncio
async def test_gsplat_partial_publish_emits_kind_and_name(tmp_data_dir):
    from app.jobs.cancel import CancelToken
    from app.jobs.schema import GsplatConfig
    from app.processors.base import JobContext
    from app.processors.gsplat.trainer import (
        GsplatProcessor,
        IterationLog,
        TrainerState,
    )

    job_id = "gsplat-payload-001"
    job_dir = tmp_data_dir / "jobs" / job_id
    artifacts_dir = job_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    state = TrainerState(
        means=np.zeros((10, 3), dtype=np.float64),
        colors=np.zeros((10, 3), dtype=np.float64),
        opacities=np.zeros((10,), dtype=np.float64),
        scales=np.zeros((10, 3), dtype=np.float64),
        rotations=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (10, 1)),
    )
    cfg = GsplatConfig(
        source_job_id="dummy",
        iterations=2,
        preview_max_gaussians=1000,
    )
    last_log = IterationLog(iter=10, n_gaussians=10, psnr=20.0, loss=0.1)

    events: list = []

    async def publish(event):
        events.append(event)
        return event

    async def noop(*_a, **_k):
        return None

    ctx = JobContext(
        job_id=job_id,
        uploads=[],
        config=cfg,
        job_dir=job_dir,
        frames_dir=job_dir / "frames",
        artifacts_dir=artifacts_dir,
        cancel=CancelToken(),
        publish=publish,
        set_status=noop,
        set_frames_total=noop,
    )

    processor = GsplatProcessor()
    await processor._publish_partial(ctx, state, cfg, iter_idx=10, last_log=last_log)

    artifact_events = [e for e in events if e.stage == "artifact"]
    assert artifact_events, "expected an artifact event from _publish_partial"
    payload = artifact_events[-1].data

    # The frontend's `latestPartialSplat` reducer matches on
    # `kind === "partial_splat" || kind === "splat_ply"` and reads
    # `name`. Without these keys the splat layer never picks up the
    # live preview URLs.
    assert payload.get("kind") == "partial_splat", payload
    assert isinstance(payload.get("name"), str), payload
    assert payload["name"].endswith(".ply"), payload
