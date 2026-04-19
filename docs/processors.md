# Processors

A **processor** is a backend that turns an ingested frame sequence (or, for
gsplat, another job's output) into 3D artifacts. All processors ride on the
same `Processor` interface in `worker/app/processors/base.py`, so adding a
new one means filling in the interface and wiring it into the registry.

## The interface

```python
# worker/app/processors/base.py
class Processor(ABC):
    id: ClassVar[str]                       # e.g. "droid_slam"
    kind: ClassVar[Literal["reconstruction", "slam", "gsplat"]]
    worker_class: ClassVar[str]             # "lingbot" | "slam" | "gs"
    supported_artifacts: ClassVar[set[str]]

    async def run(self, ctx: JobContext) -> None: ...
```

`JobContext` bundles: frames dir, output dir, cancel token, event publisher,
the parsed config, checkpoint cache dir, and the VRAM watchdog hook. Every
processor drives its own loop and calls `ctx.publish(stage, level, message,
progress, data)` for live events.

## Registry + routing

Two tables in `worker/app/processors/__init__.py`:

1. `WORKER_CLASSES: dict[str, str]` — maps processor id → container name
   (`lingbot`, `slam`, `gs`). Stays import-free so the API process can read
   it without pulling torch.
2. `_MODULE_PATHS: dict[str, (module, class_name)]` — used by
   `load_processor(id)` to import the concrete class on demand. The SLAM
   processors, for example, live in `app.processors.slam.*` and only get
   imported inside `worker-slam`, which has the matching CUDA extensions.

Add a new processor by adding one entry to each table and creating the
module. Example for a hypothetical `my_slam`:

```python
# worker/app/processors/__init__.py
WORKER_CLASSES["my_slam"] = "slam"
_MODULE_PATHS["my_slam"] = ("app.processors.slam.my_slam", "MySlamProcessor")
```

## Per-processor config

`worker/app/jobs/schema.py` holds a discriminated-union over all processor
configs, keyed by `processor`:

```python
AnyJobConfig = Annotated[
    Union[LingbotConfig, DroidSlamConfig, Mast3rSlamConfig,
          DpvoConfig, MonogsConfig, GsplatConfig],
    Field(discriminator="processor"),
]
```

A new processor adds a new member to that union. The frontend mirrors the
shape in `web/src/lib/types.ts` and updates `DEFAULT_SLAM_CONFIGS` (or adds
a new `DEFAULT_*_CONFIG`) so the UI has a seed config to show.

## Checkpoint cache

`worker/app/pipeline/checkpoints.py` exposes
`ensure_checkpoint(model_id, job_id, publish, *, processor_id, optional)`
keyed by `(processor_id, model_id)`. Each processor registers its HF
repo + filename in `_REGISTRY`; missing-on-disk checkpoints stream in with
progress events, and `optional=True` is the escape hatch for simulated
backends that tolerate absence.

## Live preview contract

The viewer knows how to display four artifact kinds:

| Kind | Event `data.kind` | Cleanup event | Rendered by |
| --- | --- | --- | --- |
| Partial point cloud | `partial_ply` | `partial_cleanup` | `PointCloud` |
| Final mesh | (manifest only) | — | `MeshLayer` |
| Camera trajectory | (manifest only) | — | `CameraPath` |
| Partial splat | `partial_splat` | `partial_splat_cleanup` | `SplatLayer` |

A processor gets live preview for free by writing incrementally-numbered
`partial_NNNN.ply` (or `partial_splat_NNNN.ply`) files to the job's
artifacts dir and emitting the matching event. The frontend's
`useJobStream` + `useMemo` scan picks up the latest automatically.

## Export artifacts

Each processor is responsible for its final artifacts; the runner only
handles the lifecycle. Standard filenames the UI knows to promote:

- `reconstruction.ply` — final sparse/dense cloud.
- `mesh.glb` / `rev_NNN.glb` — meshes + edit revisions.
- `camera_path.json` — minimal `{fps, poses: [{t, q}]}`.
- `pose_graph.json` — richer per-keyframe with intrinsics (SLAM only).
- `keyframes.jsonl` — per-keyframe metadata.
- `splat.ply` — standard 3DGS PLY (gsplat + MonoGS).
- `splat.sogs` — compressed splat (placeholder sidecar until the real
  encoder ships).
