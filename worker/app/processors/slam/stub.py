"""SLAM stub processors.

Phase 2 smoke-test implementations. Each one:
  * flips the job status through a plausible stage sequence,
  * publishes periodic progress events so the UI's log/status strip gets
    exercised,
  * honours `ctx.check_cancel` every tick so stop/restart can be tested,
  * emits a trivial `pose_graph.json` artifact so the manifest isn't empty.

Phase 4 replaces each subclass with a real backend. The class layout and
`id` values stay the same so the processor registry doesn't need to change.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import ClassVar

from app.jobs.schema import Artifact, JobEvent, SlamConfig
from app.processors.base import JobContext, Processor, ProcessorResult

log = logging.getLogger(__name__)


class _SlamStub(Processor):
    """Shared behaviour for every SLAM backend stub."""

    kind: ClassVar[str] = "slam"  # type: ignore[assignment]
    worker_class: ClassVar[str] = "slam"  # type: ignore[assignment]
    supported_artifacts = frozenset({"ply", "json", "pose_graph_json", "keyframes_jsonl"})

    # Friendly name used in log lines so `worker-slam` output is readable.
    display_name: ClassVar[str] = "SLAM"

    async def run(self, ctx: JobContext) -> ProcessorResult:
        cfg = ctx.config
        assert isinstance(cfg, SlamConfig), (
            f"{self.__class__.__name__} got non-SLAM config: {type(cfg).__name__}"
        )
        await ctx.publish(
            JobEvent(
                job_id=ctx.job_id,
                stage="queue",
                message=f"{self.display_name} stub starting",
                data={"backend": self.id, "uploads": [p.name for p in ctx.uploads]},
            )
        )
        ctx.check_cancel()

        await ctx.set_status("ingest")
        for pct in (10, 25, 50, 75, 100):
            ctx.check_cancel()
            await ctx.publish(
                JobEvent(
                    job_id=ctx.job_id,
                    stage="ingest",
                    message=f"ingest {pct}%",
                    progress=pct / 100.0,
                )
            )
            await asyncio.sleep(0.5)

        await ctx.set_status("slam")
        for i in range(1, 6):
            ctx.check_cancel()
            await ctx.publish(
                JobEvent(
                    job_id=ctx.job_id,
                    stage="slam",
                    message=f"tracking keyframe {i}/5",
                    progress=i / 5.0,
                    data={"keyframe": i, "inliers": 100 + 20 * i},
                )
            )
            await asyncio.sleep(0.5)

        # Emit a tiny pose-graph JSON so the manifest has something real.
        ctx.artifacts_dir.mkdir(parents=True, exist_ok=True)
        pose_graph = ctx.artifacts_dir / "pose_graph.json"
        pose_graph.write_text(
            json.dumps(
                {
                    "backend": self.id,
                    "poses": [
                        {"frame": i, "t": [0.0, 0.0, float(i) * 0.1], "q": [0.0, 0.0, 0.0, 1.0]}
                        for i in range(5)
                    ],
                    "intrinsics": {"fx": cfg.fx, "fy": cfg.fy, "cx": cfg.cx, "cy": cfg.cy},
                }
            )
        )

        return ProcessorResult(
            artifacts=[
                Artifact(
                    name=pose_graph.name,
                    kind="pose_graph_json",
                    size_bytes=pose_graph.stat().st_size,
                )
            ],
            extras={"stub": True, "backend": self.id},
        )


class DroidSlamStubProcessor(_SlamStub):
    id: ClassVar[str] = "droid_slam"  # type: ignore[assignment]
    display_name: ClassVar[str] = "DROID-SLAM (stub)"


class Mast3rSlamStubProcessor(_SlamStub):
    id: ClassVar[str] = "mast3r_slam"  # type: ignore[assignment]
    display_name: ClassVar[str] = "MASt3R-SLAM (stub)"


class DpvoStubProcessor(_SlamStub):
    id: ClassVar[str] = "dpvo"  # type: ignore[assignment]
    display_name: ClassVar[str] = "DPVO (stub)"


class MonogsStubProcessor(_SlamStub):
    id: ClassVar[str] = "monogs"  # type: ignore[assignment]
    display_name: ClassVar[str] = "MonoGS (stub)"
