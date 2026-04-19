"""Lingbot processor.

Wraps the existing ingest → checkpoint → inference → export pipeline so it
rides on the `Processor` abstraction. Behaviour is unchanged from the
pre-refactor runner — the code was moved here almost verbatim, with the
runner-owned DB calls swapped for `ctx.set_status` / `ctx.set_frames_total`.
"""

from __future__ import annotations

import asyncio
import logging

from app.config import settings
from app.jobs.schema import Artifact, JobEvent, LingbotConfig
from app.pipeline.checkpoints import ensure_checkpoint
from app.pipeline.export import export_reconstruction
from app.pipeline.ingest import concat_videos_to_frames
from app.pipeline.inference import run_inference
from app.pipeline.watchdog import VramWatchState, run_vram_watchdog
from app.processors.base import JobContext, Processor, ProcessorResult

log = logging.getLogger(__name__)


class LingbotProcessor(Processor):
    id = "lingbot"
    kind = "reconstruction"
    worker_class = "lingbot"
    supported_artifacts = frozenset({"glb", "ply", "obj", "npz", "json"})

    async def run(self, ctx: JobContext) -> ProcessorResult:
        cfg = ctx.config
        assert isinstance(cfg, LingbotConfig), (
            f"LingbotProcessor got non-lingbot config: {type(cfg).__name__}"
        )

        await ctx.publish(
            JobEvent(job_id=ctx.job_id, stage="queue", message="job starting")
        )
        ctx.check_cancel()

        # 1. ingest
        await ctx.set_status("ingest")
        frames_total = await concat_videos_to_frames(
            job_id=ctx.job_id,
            sources=ctx.uploads,
            dest=ctx.frames_dir,
            config=cfg,
            publish=ctx.publish,
        )
        await ctx.set_frames_total(frames_total)
        ctx.check_cancel()

        # 2. checkpoint
        ckpt = await ensure_checkpoint(cfg.model_id, ctx.job_id, ctx.publish)
        ctx.check_cancel()

        # 3. inference — spin up a VRAM watchdog alongside the GPU call.
        await ctx.set_status("inference")
        soft_limit = cfg.vram_soft_limit_gb or settings.vram_default_soft_limit_gb
        vram_state = VramWatchState(soft_limit_gb=float(soft_limit))
        watchdog_task = asyncio.create_task(
            run_vram_watchdog(ctx.job_id, vram_state, ctx.publish)
        )
        try:
            predictions = await run_inference(
                job_id=ctx.job_id,
                frames_dir=ctx.frames_dir,
                ckpt_path=ckpt,
                config=cfg,
                publish=ctx.publish,
                vram_state=vram_state,
                cancel_token=ctx.cancel,
            )
        finally:
            vram_state.stop()
            try:
                await asyncio.wait_for(watchdog_task, timeout=5.0)
            except asyncio.TimeoutError:
                watchdog_task.cancel()
        ctx.check_cancel()
        await ctx.publish(
            JobEvent(
                job_id=ctx.job_id,
                stage="inference",
                message=(
                    f"vram peak {vram_state.peak_gb:.2f} GB "
                    f"(soft limit {vram_state.soft_limit_gb:.1f} GB)"
                ),
                data={
                    "vram_peak_gb": round(vram_state.peak_gb, 3),
                    "vram_soft_limit_gb": vram_state.soft_limit_gb,
                },
            )
        )

        # 4. export
        await ctx.set_status("export")
        artifact_paths = await export_reconstruction(
            job_id=ctx.job_id,
            frames_dir=ctx.frames_dir,
            artifacts_dir=ctx.artifacts_dir,
            predictions=predictions,
            config=cfg,
            publish=ctx.publish,
        )

        art_list: list[Artifact] = []
        for name, path in artifact_paths.items():
            art_list.append(
                Artifact(
                    name=path.name,
                    kind=name,  # type: ignore[arg-type]
                    size_bytes=path.stat().st_size,
                )
            )
        npz = ctx.artifacts_dir / "predictions.npz"
        if npz.exists():
            art_list.append(
                Artifact(
                    name=npz.name,
                    kind="npz",
                    size_bytes=npz.stat().st_size,
                )
            )

        return ProcessorResult(
            artifacts=art_list,
            extras={
                "vram_peak_gb": round(vram_state.peak_gb, 3),
            },
        )
