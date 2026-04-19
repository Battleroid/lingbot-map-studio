"""Gaussian-splat training processor.

Structure mirrors `SlamProcessor`: the concrete `GsplatProcessor` owns the
lifecycle (inputs resolution → training loop → exports → optional mesh
bake) and the per-iteration work is delegated to a `SplatTrainer` object.

The default `SplatTrainer` is a simulated trainer that runs entirely on
CPU/numpy. It:

  * loads the source job's sparse cloud as initial gaussians (or a random
    sphere if `init_from="random"`),
  * mutates the gaussian parameters with a deterministic schedule that
    plausibly resembles densification + opacity pruning,
  * emits partial PLY snapshots + `training_log.jsonl` rows at the same
    cadence the real trainer will,
  * leaves a final `splat.ply` / `cameras.json` / `training_log.jsonl`
    artifact set.

When the real gsplat CUDA trainer lands in `worker-gs`, it replaces
`SimulatedSplatTrainer` with a `GsplatCudaTrainer` that wraps
`gsplat.rasterize` — everything else (processor orchestration, exports,
artifact manifest) stays the same.
"""

from __future__ import annotations

import abc
import asyncio
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Optional

import numpy as np

from app.jobs.schema import Artifact, GsplatConfig, JobEvent
from app.processors.base import JobContext, Processor, ProcessorResult
from app.processors.gsplat import export as splat_export
from app.processors.gsplat import io as splat_io

log = logging.getLogger(__name__)


@dataclass
class TrainerState:
    """Mutable gaussian-splat state the trainer grows over time.

    Matches the shape `splat_export.write_splat_ply` consumes, so
    snapshots are a one-shot `write_splat_ply(out, **state.as_kwargs())`.
    """

    means: np.ndarray
    colors: np.ndarray
    opacities: np.ndarray
    scales: np.ndarray
    rotations: np.ndarray

    def as_kwargs(self) -> dict:
        return {
            "means": self.means,
            "colors": self.colors,
            "opacities": self.opacities,
            "scales": self.scales,
            "rotations": self.rotations,
        }

    @property
    def n(self) -> int:
        return int(self.means.shape[0])


@dataclass
class IterationLog:
    iter: int
    n_gaussians: int
    psnr: float
    loss: float
    extras: dict = field(default_factory=dict)


class SplatTrainer(abc.ABC):
    """One per job. Frame + camera loading happens in `prepare`; each
    `step(iter_idx)` advances training by a single iteration's worth of
    work and returns current metrics."""

    @abc.abstractmethod
    def prepare(
        self,
        inputs: splat_io.GsplatInputs,
        cfg: GsplatConfig,
    ) -> None:
        """Load frames, cameras, init points and seed `state`."""

    @abc.abstractmethod
    def step(self, iter_idx: int) -> IterationLog:
        """Advance one iteration. Must update `self.state` in-place."""

    @property
    @abc.abstractmethod
    def state(self) -> TrainerState:
        """Current gaussian-splat parameters."""


class SimulatedSplatTrainer(SplatTrainer):
    """CPU-only stand-in for the real gsplat CUDA trainer.

    Keeps the pipeline exercisable without torch/gsplat installed. The
    "training" is a scripted schedule that adds, nudges, and occasionally
    prunes gaussians, emitting plausible PSNR/loss numbers so the
    frontend's training chart works.
    """

    def __init__(self) -> None:
        self._state: Optional[TrainerState] = None
        self._cfg: Optional[GsplatConfig] = None
        self._rng = np.random.default_rng(42)
        self._base_psnr = 12.0  # arbitrary; grows through training

    @property
    def state(self) -> TrainerState:
        assert self._state is not None, "prepare() must be called first"
        return self._state

    def prepare(
        self,
        inputs: splat_io.GsplatInputs,
        cfg: GsplatConfig,
    ) -> None:
        self._cfg = cfg
        if cfg.init_from == "point_cloud" and inputs.init_points is not None:
            pts = splat_io.load_init_points(inputs.init_points)
            means = pts[:, :3].astype(np.float64)
            # Source cloud colours are 0-255; gsplat uses 0-1.
            colors = np.clip(pts[:, 3:6] / 255.0, 0.0, 1.0).astype(np.float64)
        else:
            # Random unit-sphere init.
            n = cfg.random_init_count
            u = self._rng.normal(size=(n, 3))
            u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-9
            means = u * self._rng.uniform(0.5, 2.0, size=(n, 1))
            colors = self._rng.uniform(0, 1, size=(n, 3))

        n = means.shape[0]
        # Sensible defaults for the auxiliary fields. The real trainer
        # learns these; the simulated trainer just picks plausible seeds.
        scales = np.full((n, 3), math.log(0.02), dtype=np.float64)
        rotations = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
        opacities = np.full((n,), _inv_sigmoid(0.3), dtype=np.float64)

        self._state = TrainerState(
            means=means,
            colors=colors,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
        )

    def step(self, iter_idx: int) -> IterationLog:
        assert self._state is not None and self._cfg is not None
        st = self._state
        cfg = self._cfg

        # Move means a little toward where they'd converge — just jitter
        # with decreasing amplitude. Purely cosmetic.
        lr_scale = max(0.02, 1.0 - iter_idx / max(1, cfg.iterations))
        st.means += self._rng.normal(scale=0.002 * lr_scale, size=st.means.shape)
        # Slowly push opacities up so the splat "saturates" over training.
        st.opacities += 0.001 * lr_scale

        # Densification: every `densify_interval` iters, clone a fraction
        # of the current gaussians with small perturbations.
        if iter_idx > 0 and iter_idx % cfg.densify_interval == 0:
            self._densify()

        # Pruning: every `prune_interval` iters, drop low-opacity gaussians.
        if iter_idx > 0 and iter_idx % cfg.prune_interval == 0:
            self._prune(cfg.prune_opacity)

        # Fake metrics: PSNR climbs from ~12 to ~28 logarithmically.
        psnr = self._base_psnr + 16.0 * (
            math.log1p(iter_idx) / math.log1p(cfg.iterations)
        )
        loss = 0.25 / (1.0 + iter_idx / 500.0)
        return IterationLog(
            iter=iter_idx,
            n_gaussians=st.n,
            psnr=psnr,
            loss=loss,
        )

    def _densify(self) -> None:
        st = self._state
        assert st is not None
        # Pick top 5% by opacity and clone with small offsets.
        probs = _sigmoid(st.opacities)
        if st.n < 16:
            return
        cutoff = np.quantile(probs, 0.95)
        mask = probs >= cutoff
        if not np.any(mask):
            return
        picks = np.where(mask)[0]
        # Cap clone count to keep the simulated cloud reasonable.
        max_clone = max(0, min(len(picks), 200_000 - st.n))
        if max_clone <= 0:
            return
        picks = picks[:max_clone]
        new_means = st.means[picks] + self._rng.normal(scale=0.02, size=(picks.size, 3))
        new_colors = st.colors[picks]
        new_opacities = st.opacities[picks] - 0.1  # split halves opacity
        new_scales = st.scales[picks] - 0.1  # split halves scale
        new_rotations = st.rotations[picks]
        self._state = TrainerState(
            means=np.concatenate([st.means, new_means]),
            colors=np.concatenate([st.colors, new_colors]),
            opacities=np.concatenate([st.opacities, new_opacities]),
            scales=np.concatenate([st.scales, new_scales]),
            rotations=np.concatenate([st.rotations, new_rotations]),
        )

    def _prune(self, prune_opacity: float) -> None:
        st = self._state
        assert st is not None
        if st.n < 16:
            return
        keep = _sigmoid(st.opacities) >= prune_opacity
        if not np.any(keep):
            return
        if keep.all():
            return
        self._state = TrainerState(
            means=st.means[keep],
            colors=st.colors[keep],
            opacities=st.opacities[keep],
            scales=st.scales[keep],
            rotations=st.rotations[keep],
        )


class GsplatProcessor(Processor):
    """3D Gaussian Splat training. Pinned to the `gs` worker class."""

    id: ClassVar[str] = "gsplat"  # type: ignore[assignment]
    kind: ClassVar[str] = "gsplat"  # type: ignore[assignment]
    worker_class: ClassVar[str] = "gs"  # type: ignore[assignment]
    supported_artifacts = frozenset({"splat_ply", "splat_sogs", "json", "glb"})

    display_name: ClassVar[str] = "Gaussian Splat"

    # Subclasses can swap this for the real CUDA trainer once it's wired
    # in. The default keeps the pipeline green on any machine.
    trainer_cls: ClassVar[type[SplatTrainer]] = SimulatedSplatTrainer

    async def run(self, ctx: JobContext) -> ProcessorResult:
        cfg = ctx.config
        if not isinstance(cfg, GsplatConfig):
            raise TypeError(
                f"GsplatProcessor expected a GsplatConfig, got {type(cfg).__name__}"
            )

        await ctx.publish(
            JobEvent(
                job_id=ctx.job_id,
                stage="queue",
                message=f"gsplat starting from source={cfg.source_job_id}",
                data={"source_job_id": cfg.source_job_id, "iterations": cfg.iterations},
            )
        )
        ctx.check_cancel()

        inputs = await self._resolve_inputs(ctx, cfg)
        trainer = self.trainer_cls()
        await asyncio.to_thread(trainer.prepare, inputs, cfg)
        await ctx.publish(
            JobEvent(
                job_id=ctx.job_id,
                stage="training",
                message=(
                    f"initialised {trainer.state.n} gaussians "
                    f"(init_from={cfg.init_from})"
                ),
                data={"n_gaussians": trainer.state.n},
            )
        )

        await ctx.set_status("training")
        log_path = ctx.artifacts_dir / "training_log.jsonl"
        ctx.artifacts_dir.mkdir(parents=True, exist_ok=True)
        # Wipe any stale log from a restart — we re-run from iter 0.
        if log_path.exists():
            log_path.unlink()

        last_log: Optional[IterationLog] = None
        for it in range(1, cfg.iterations + 1):
            ctx.check_cancel()
            last_log = await asyncio.to_thread(trainer.step, it)
            splat_export.append_training_log(
                log_path,
                {
                    "iter": last_log.iter,
                    "n_gaussians": last_log.n_gaussians,
                    "psnr": last_log.psnr,
                    "loss": last_log.loss,
                },
            )

            if (
                cfg.preview_every_iters > 0
                and it % cfg.preview_every_iters == 0
            ):
                await self._publish_partial(ctx, trainer.state, cfg, it, last_log)

            # Heartbeat ~20 events total across the run so the UI status
            # strip stays fresh without flooding.
            if it % max(1, cfg.iterations // 20) == 0:
                await ctx.publish(
                    JobEvent(
                        job_id=ctx.job_id,
                        stage="training",
                        message=(
                            f"iter {it}/{cfg.iterations} · "
                            f"{last_log.n_gaussians} gaussians · "
                            f"psnr={last_log.psnr:.2f}"
                        ),
                        progress=it / cfg.iterations,
                        data={
                            "iter": it,
                            "n_gaussians": last_log.n_gaussians,
                            "psnr": last_log.psnr,
                            "loss": last_log.loss,
                        },
                    )
                )

        ctx.check_cancel()
        await ctx.set_status("export")
        artifacts = await self._finalise(ctx, trainer.state, inputs, cfg, last_log)

        extras: dict = {
            "source_job_id": cfg.source_job_id,
            "final_gaussians": trainer.state.n,
        }
        if last_log is not None:
            extras.update(final_psnr=last_log.psnr, final_iter=last_log.iter)
        if isinstance(trainer, SimulatedSplatTrainer):
            extras["simulated"] = True
        return ProcessorResult(artifacts=artifacts, extras=extras)

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------

    async def _resolve_inputs(
        self, ctx: JobContext, cfg: GsplatConfig
    ) -> splat_io.GsplatInputs:
        """Load the source job, resolve its paths. Split out so a bake-
        from-mesh variant can stub this in the future without duplicating
        the full run()."""
        from app.jobs.store import get_job  # noqa: PLC0415

        src = await get_job(cfg.source_job_id)
        if src is None:
            raise splat_io.GsplatInputsError(
                f"source job {cfg.source_job_id} not found"
            )
        inputs = splat_io.resolve_inputs(src)
        await ctx.publish(
            JobEvent(
                job_id=ctx.job_id,
                stage="ingest",
                message=(
                    f"source={src.id} processor={inputs.source_processor} "
                    f"init={inputs.init_points.name if inputs.init_points else 'random'} "
                    f"cameras={inputs.cameras_path.name if inputs.cameras_path else 'none'}"
                ),
                data={
                    "source_job_id": src.id,
                    "source_processor": inputs.source_processor,
                    "has_init_points": inputs.init_points is not None,
                    "has_cameras": inputs.cameras_path is not None,
                },
            )
        )
        return inputs

    async def _publish_partial(
        self,
        ctx: JobContext,
        state: TrainerState,
        cfg: GsplatConfig,
        iter_idx: int,
        last_log: Optional[IterationLog],
    ) -> None:
        """Write a capped-size partial splat PLY + emit an artifact event."""
        ctx.artifacts_dir.mkdir(parents=True, exist_ok=True)
        snapshot = _maybe_subsample(state, cap=cfg.preview_max_gaussians)
        name = f"partial_splat_{iter_idx:06d}.ply"
        out = ctx.artifacts_dir / name
        await asyncio.to_thread(
            splat_export.write_splat_ply, out, **snapshot.as_kwargs()
        )
        await ctx.publish(
            JobEvent(
                job_id=ctx.job_id,
                stage="artifact",
                message=f"partial splat at iter {iter_idx} ({snapshot.n} gaussians)",
                data={
                    "partial": name,
                    "iter": iter_idx,
                    "n_gaussians": snapshot.n,
                    **(
                        {"psnr": last_log.psnr, "loss": last_log.loss}
                        if last_log
                        else {}
                    ),
                },
            )
        )

    async def _finalise(
        self,
        ctx: JobContext,
        state: TrainerState,
        inputs: splat_io.GsplatInputs,
        cfg: GsplatConfig,
        last_log: Optional[IterationLog],
    ) -> list[Artifact]:
        artifacts: list[Artifact] = []

        splat = ctx.artifacts_dir / "splat.ply"
        await asyncio.to_thread(
            splat_export.write_splat_ply, splat, **state.as_kwargs()
        )
        artifacts.append(
            Artifact(
                name=splat.name,
                kind="splat_ply",
                size_bytes=splat.stat().st_size,
            )
        )

        cameras_out = ctx.artifacts_dir / "cameras.json"
        cameras = []
        if inputs.cameras_path is not None:
            try:
                cameras = splat_io.load_cameras(inputs.cameras_path)
            except Exception as exc:  # noqa: BLE001
                log.warning("gsplat: failed to replay cameras (%s)", exc)
        await asyncio.to_thread(
            splat_export.write_cameras_json,
            cameras_out,
            cameras,
            backend=inputs.source_processor,
        )
        artifacts.append(
            Artifact(
                name=cameras_out.name,
                kind="json",
                size_bytes=cameras_out.stat().st_size,
            )
        )

        log_path = ctx.artifacts_dir / "training_log.jsonl"
        if log_path.exists():
            artifacts.append(
                Artifact(
                    name=log_path.name,
                    kind="keyframes_jsonl",
                    size_bytes=log_path.stat().st_size,
                )
            )

        sogs = ctx.artifacts_dir / "splat.sogs"
        await asyncio.to_thread(
            splat_export.write_sogs_placeholder,
            sogs,
            splat_ply_path=splat,
            iterations=cfg.iterations,
            n_gaussians=state.n,
        )
        artifacts.append(
            Artifact(
                name=sogs.name,
                kind="splat_sogs",
                size_bytes=sogs.stat().st_size,
            )
        )

        final_msg = f"wrote splat.ply ({state.n} gaussians)"
        if last_log is not None:
            final_msg += f", psnr={last_log.psnr:.2f}"
        await ctx.publish(
            JobEvent(
                job_id=ctx.job_id,
                stage="export",
                message=final_msg,
                progress=1.0,
                data={"n_gaussians": state.n},
            )
        )
        return artifacts


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _inv_sigmoid(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return float(np.log(p / (1.0 - p)))


def _maybe_subsample(state: TrainerState, *, cap: int) -> TrainerState:
    if cap <= 0 or state.n <= cap:
        return state
    step = int(state.n // cap) + 1
    return TrainerState(
        means=state.means[::step],
        colors=state.colors[::step],
        opacities=state.opacities[::step],
        scales=state.scales[::step],
        rotations=state.rotations[::step],
    )
