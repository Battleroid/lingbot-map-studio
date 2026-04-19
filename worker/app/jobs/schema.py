from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

JobStatus = Literal[
    "queued",
    "ingest",
    "inference",
    "export",
    "slam",
    "meshing",
    "training",
    "ready",
    "failed",
    "cancelled",
]
EventLevel = Literal["info", "warn", "error", "stdout", "stderr", "debug"]
EventStage = Literal[
    "queue",
    "ingest",
    "checkpoint",
    "inference",
    "slam",
    "training",
    "meshing",
    "export",
    "mesh",
    "artifact",
    "system",
]

ProcessorId = Literal[
    "lingbot",
    "droid_slam",
    "mast3r_slam",
    "dpvo",
    "monogs",
    "gsplat",
]
SlamBackend = Literal["droid_slam", "mast3r_slam", "dpvo", "monogs"]
ProcessorKind = Literal["reconstruction", "slam", "gsplat"]

# Widened to cover all future modes. Artifact.kind is free-form string in
# practice — the UI keys off suffix — but we enumerate the known kinds so the
# viewer + tool panel code has exhaustive switches to hang off of.
ArtifactKind = Literal[
    "glb",
    "ply",
    "obj",
    "npz",
    "json",
    # SLAM / GS additions (kinds used from Phase 4 / Phase 5 onward).
    "splat_ply",
    "splat_sogs",
    "pose_graph_json",
    "keyframes_jsonl",
]


def _now() -> datetime:
    return datetime.now(timezone.utc)


class LingbotConfig(BaseModel):
    """Config for the existing dense-pointmap model. Unchanged from the
    original JobConfig — a `processor` discriminator is added so it can
    participate in the AnyJobConfig union."""

    model_config = ConfigDict(protected_namespaces=())

    processor: Literal["lingbot"] = "lingbot"

    model_id: str = "lingbot-map"
    mode: Literal["streaming", "windowed"] = "streaming"
    window_size: int = 64
    overlap_size: int = 16
    image_size: int = 518
    patch_size: int = 14
    fps: float = 10.0
    first_k: Optional[int] = None
    stride: int = 1
    mask_sky: bool = True
    # Percentile passed to predictions_to_glb's conf_thres (0-100). Higher =
    # stricter filter (fewer, more-confident points). lingbot-map's default is 50.
    conf_percentile: float = 50.0
    keyframe_interval: int = 6
    # Frames used in the initial batched scale-anchor pass. Lower = lower
    # peak VRAM for the first forward call; 4 is a safe default on 20-24 GB.
    num_scale_frames: int = 4
    camera_num_iterations: int = 4
    max_frame_num: int = 1024
    # KV cache is trimmed once keyframe count exceeds this. Lower values
    # trade a bit of global context for much flatter memory over long runs.
    kv_cache_sliding_window: int = 32
    enable_3d_rope: bool = True
    use_sdpa: bool = True
    offload_to_cpu: bool = True
    show_cam: bool = True
    mask_black_bg: bool = False
    mask_white_bg: bool = False

    # --- Preprocessing (applied during ffmpeg ingest / post-extract) ---
    # Fisheye → rectilinear unwrap. Most FPV cams are ~155-170° in_fov; 90° out
    # keeps the useful centre and trims the rim where distortion was worst.
    preproc_fisheye: bool = False
    fisheye_in_fov: float = 165.0
    fisheye_out_fov: float = 90.0
    # Temporal denoise + deflicker — kills analog static and brightness flicker.
    preproc_denoise: bool = False
    # Detect static pixels across frames and inpaint them out (OSD/telemetry overlays).
    preproc_osd_mask: bool = False
    osd_mask_samples: int = 60
    osd_mask_std_threshold: float = 5.0
    osd_mask_dilate: int = 2
    # Second detector: flag pixels that live near an edge in >N% of sampled
    # frames. Catches changing OSD digits that the stddev detector misses.
    osd_detect_text: bool = True
    osd_edge_persist_frac: float = 0.75

    # Per-job VRAM soft limit in GB. If allocated GPU memory crosses this during
    # inference the watchdog aborts the job. None = use worker-wide default.
    vram_soft_limit_gb: Optional[float] = None

    # Live reconstruction snapshots — every N processed frames the inference
    # hook writes a partial PLY so the viewer can show the point cloud growing.
    # 0 disables. Lower = smoother updates, more disk/CPU churn.
    partial_snapshot_every: int = 60


class SlamConfig(BaseModel):
    """Stub for SLAM-mode jobs. Implemented in Phase 4 — this shape is the
    minimum the discriminated-union router + UI need to dispatch to the right
    backend. Per-backend tunables are added alongside each processor."""

    model_config = ConfigDict(protected_namespaces=())

    processor: SlamBackend
    # Most SLAM backends publish a ply + pose graph; the shared fields below
    # cover the common ingest/keyframe/calibration knobs. Per-backend configs
    # extend this in Phase 4.
    max_frames: Optional[int] = None
    downscale: float = 1.0
    stride: int = 1
    fps: float = 10.0
    calibration: Literal["auto", "manual"] = "auto"
    fx: Optional[float] = None
    fy: Optional[float] = None
    cx: Optional[float] = None
    cy: Optional[float] = None
    keyframe_policy: Literal["score_gated", "translation", "hybrid"] = "score_gated"

    # Shared preproc flags (a subset of lingbot's; Phase 3 introduces a richer
    # PreprocConfig shared across all modes).
    preproc_fisheye: bool = False
    fisheye_in_fov: float = 165.0
    fisheye_out_fov: float = 90.0
    preproc_denoise: bool = False
    preproc_osd_mask: bool = False

    vram_soft_limit_gb: Optional[float] = None
    partial_snapshot_every: int = 60


class GsplatConfig(BaseModel):
    """Stub for the GS training mode. Consumes a completed SLAM (or lingbot)
    job's output. Implemented in Phase 5."""

    model_config = ConfigDict(protected_namespaces=())

    processor: Literal["gsplat"] = "gsplat"
    source_job_id: str
    iterations: int = 30_000
    sh_degree: int = 3
    densify_interval: int = 500
    prune_opacity: float = 0.005
    init_from: Literal["point_cloud", "random"] = "point_cloud"
    preview_every_iters: int = 1000
    vram_soft_limit_gb: Optional[float] = None


# Discriminated union. Pydantic v2 picks the right class based on the
# "processor" field. SlamConfig covers four literal values of the discriminator
# which Pydantic folds into the union correctly.
AnyJobConfig = Annotated[
    Union[LingbotConfig, SlamConfig, GsplatConfig],
    Field(discriminator="processor"),
]

# Back-compat alias so modules that only know the lingbot shape keep working.
# New code should reference AnyJobConfig (at boundaries) or the specific
# LingbotConfig/SlamConfig/GsplatConfig (inside a processor).
JobConfig = LingbotConfig

_ANY_JOB_CONFIG_ADAPTER: TypeAdapter[AnyJobConfig] = TypeAdapter(AnyJobConfig)


def parse_job_config(raw: Union[str, bytes, dict[str, Any]]) -> AnyJobConfig:
    """Parse a raw job config payload into the discriminated union.

    Rows created before this refactor have no `processor` field — treat them
    as lingbot so existing jobs keep loading cleanly.
    """
    if isinstance(raw, (str, bytes)):
        data = json.loads(raw)
    else:
        data = dict(raw)
    if "processor" not in data:
        data["processor"] = "lingbot"
    return _ANY_JOB_CONFIG_ADAPTER.validate_python(data)


def dump_job_config(cfg: AnyJobConfig) -> str:
    """JSON-encode a config regardless of which branch of the union it is."""
    return _ANY_JOB_CONFIG_ADAPTER.dump_json(cfg).decode()


def processor_kind(cfg: AnyJobConfig) -> ProcessorKind:
    """Group the specific processor id into its broader kind for UI wiring."""
    pid = cfg.processor
    if pid == "lingbot":
        return "reconstruction"
    if pid == "gsplat":
        return "gsplat"
    return "slam"


class Artifact(BaseModel):
    name: str
    kind: ArtifactKind
    revision: int = 0
    size_bytes: int = 0
    created_at: datetime = Field(default_factory=_now)


class JobEvent(BaseModel):
    id: int = 0
    job_id: str
    stage: EventStage
    level: EventLevel = "info"
    message: str = ""
    progress: Optional[float] = None  # 0.0..1.0
    data: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_now)


class Job(BaseModel):
    id: str
    status: JobStatus = "queued"
    config: AnyJobConfig
    uploads: list[str] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
    frames_total: Optional[int] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class JobSummary(BaseModel):
    id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    frames_total: Optional[int] = None
    artifact_count: int = 0
    processor: ProcessorId = "lingbot"


class MeshEditRequest(BaseModel):
    op: Literal[
        "cull",
        "fill_holes",
        "decimate",
        "smooth",
        "remove_small",
        "surface_recon",
    ]
    params: dict[str, Any] = Field(default_factory=dict)
    face_indices: Optional[list[int]] = None
    source_revision: Optional[int] = None  # default = latest


class ReexportRequest(BaseModel):
    format: Literal["glb", "ply", "obj"] = "glb"
    conf_percentile: Optional[float] = None
    mask_sky: Optional[bool] = None
    show_cam: Optional[bool] = None
    mask_black_bg: Optional[bool] = None
    mask_white_bg: Optional[bool] = None
