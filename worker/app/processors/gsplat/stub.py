"""Gaussian-splat training stub.

Phase 2 smoke-test implementation. Walks through a plausible training
progress sequence so the API + worker pipeline can be verified before the
real gsplat trainer lands in Phase 5.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import ClassVar

from app.jobs.schema import Artifact, GsplatConfig, JobEvent
from app.processors.base import JobContext, Processor, ProcessorResult

log = logging.getLogger(__name__)


class GsplatStubProcessor(Processor):
    id: ClassVar[str] = "gsplat"  # type: ignore[assignment]
    kind: ClassVar[str] = "gsplat"  # type: ignore[assignment]
    worker_class: ClassVar[str] = "gs"  # type: ignore[assignment]
    supported_artifacts = frozenset({"splat_ply", "splat_sogs", "json"})

    async def run(self, ctx: JobContext) -> ProcessorResult:
        cfg = ctx.config
        assert isinstance(cfg, GsplatConfig), (
            f"GsplatStubProcessor got non-gsplat config: {type(cfg).__name__}"
        )

        await ctx.publish(
            JobEvent(
                job_id=ctx.job_id,
                stage="queue",
                message="gsplat stub starting",
                data={"source_job_id": cfg.source_job_id, "iterations": cfg.iterations},
            )
        )
        ctx.check_cancel()

        await ctx.set_status("training")
        # Pretend to train over a handful of checkpoints. Real trainer
        # publishes these at `preview_every_iters` iteration intervals.
        checkpoints = 5
        for i in range(1, checkpoints + 1):
            ctx.check_cancel()
            await ctx.publish(
                JobEvent(
                    job_id=ctx.job_id,
                    stage="training",
                    message=f"iter {i * (cfg.iterations // checkpoints)} / {cfg.iterations}",
                    progress=i / checkpoints,
                    data={
                        "iter": i * (cfg.iterations // checkpoints),
                        "n_gaussians": 1000 * i,
                        "psnr": 18.0 + i * 1.5,
                    },
                )
            )
            await asyncio.sleep(0.6)

        ctx.artifacts_dir.mkdir(parents=True, exist_ok=True)
        marker = ctx.artifacts_dir / "training_log.jsonl"
        with marker.open("w") as fh:
            for i in range(1, checkpoints + 1):
                fh.write(
                    json.dumps(
                        {
                            "iter": i * (cfg.iterations // checkpoints),
                            "n_gaussians": 1000 * i,
                            "psnr": 18.0 + i * 1.5,
                        }
                    )
                    + "\n"
                )

        return ProcessorResult(
            artifacts=[
                Artifact(
                    name=marker.name,
                    kind="json",
                    size_bytes=marker.stat().st_size,
                )
            ],
            extras={"stub": True, "source_job_id": cfg.source_job_id},
        )
