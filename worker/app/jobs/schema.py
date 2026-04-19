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


class PreprocFields(BaseModel):
    """FPV-oriented preprocessing knobs shared between lingbot + SLAM.

    Kept flat (no nested model) so legacy rows keep deserialising: Pydantic
    fills in defaults for fields a stored JSON blob doesn't know about. The
    mixin is inlined into each concrete config below rather than nested, so
    `cfg.preproc_<stage>` continues to read the way existing ingest code
    expects.
    """

    model_config = ConfigDict(protected_namespaces=())

    # --- Geometric / optical pre-passes (ffmpeg) ---
    preproc_fisheye: bool = False
    fisheye_in_fov: float = 165.0
    fisheye_out_fov: float = 90.0

    # --- Analog noise / flicker cleanup (ffmpeg) ---
    # `preproc_denoise` is the basic hqdn3d+deflicker pair (already present
    # pre-Phase-3). `preproc_analog_cleanup` adds temporal `atadenoise` tuned
    # for VHS/analog static — heavier, only use for genuinely noisy sources.
    preproc_denoise: bool = False
    preproc_analog_cleanup: bool = False
    preproc_deflicker: bool = False  # standalone deflicker without the denoise pair

    # --- OSD/HUD mask (Python post-extract) ---
    preproc_osd_mask: bool = False
    osd_mask_samples: int = 60
    osd_mask_std_threshold: float = 5.0
    osd_mask_dilate: int = 2
    osd_detect_text: bool = True
    osd_edge_persist_frac: float = 0.75

    # --- Colour normalisation (Python post-extract) ---
    # Grey-world white-balance + gamma-consistent per-frame histogram
    # stretch. Cheap and very effective on VHS-era chroma fringing.
    preproc_color_norm: bool = False

    # --- Rolling-shutter correction (Python post-extract) ---
    # v1 only handles dominant-skew case: estimates a global shear from
    # optical flow between rows, applies an inverse y-shear per frame.
    # Full per-row RS remains out of scope.
    preproc_rs_correction: bool = False
    # Optional explicit shear override; None = estimate from data.
    rs_shear_px_per_row: Optional[float] = None

    # --- Motion deblur (Python post-extract) ---
    # "none" → off. "unsharp" → classical unsharp-mask + gating (fast, CPU).
    # "nafnet" → learned single-image deblurring; Phase 3 ships the hook but
    # the checkpoint fetcher lands with its own PR to avoid surprise VRAM use.
    preproc_deblur: Literal["none", "unsharp", "nafnet"] = "none"
    # Per-frame sharpness gate: only apply deblur to frames whose Laplacian
    # variance falls below this fraction of the clip median. 1.0 = always,
    # 0.5 = only the blurriest half, etc.
    deblur_sharpness_gate: float = 0.6

    # --- Keyframe scoring (Python post-extract) ---
    # Produces `frame_scores.jsonl` next to the frames dir. Consumed by SLAM
    # backends with a keyframe_policy="score_gated" and by the UI's live
    # preview toolbar. No cost to emitting scores even when unused.
    preproc_keyframe_score: bool = False
    keyframe_min_sharpness_frac: float = 0.0  # drop frames below this quantile
    keyframe_min_motion_px: float = 0.0  # drop frames with near-zero flow


class LingbotConfig(PreprocFields):
    """Config for the existing dense-pointmap model. Unchanged from the
    original JobConfig — a `processor` discriminator is added so it can
    participate in the AnyJobConfig union."""

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

    # Preprocessing fields (fisheye, denoise, OSD, analog cleanup, color
    # normalisation, rolling shutter, deblur, keyframe scoring) come from
    # the `PreprocFields` mixin above.

    # Per-job VRAM soft limit in GB. If allocated GPU memory crosses this during
    # inference the watchdog aborts the job. None = use worker-wide default.
    vram_soft_limit_gb: Optional[float] = None

    # Live reconstruction snapshots — every N processed frames the inference
    # hook writes a partial PLY so the viewer can show the point cloud growing.
    # 0 disables. Lower = smoother updates, more disk/CPU churn.
    partial_snapshot_every: int = 60


class _SlamConfigBase(PreprocFields):
    """Shared SLAM tunables. Not used directly — one of the per-backend
    subclasses below is the actual union member with its own discriminator.

    Kept as a base class (rather than inlining fields into each subclass) so
    the list of common knobs has exactly one definition. The per-backend
    subclasses only carry the fields that materially differ between
    implementations (buffer sizes, learning-rate caps, splat-specific params).
    """

    # Common to every backend.
    model_id: str = "default"
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
    # Per-backend keyframe throttle; base default is 6 frames (matches
    # lingbot's keyframe cadence). Backends with their own heuristic
    # (MASt3R, MonoGS) override in config.
    keyframe_interval: int = 6
    # If score_gated: keep frames whose quality is above this percentile
    # of the clip median. 0.5 = reject the blurriest half.
    score_gate_quantile: float = 0.5
    # Live preview: emit partial_splat/partial PLY + camera_path.json every
    # N processed keyframes. 0 disables live updates.
    partial_snapshot_every: int = 5
    # Optional end-of-job Poisson meshing. Cheap wrapper over existing
    # mesh.ops.surface_recon. Disabled for MonoGS (it emits a splat, not
    # points) and DPVO (trajectory-only is often cleaner as-is).
    run_poisson_mesh: bool = False
    poisson_depth: int = 8

    vram_soft_limit_gb: Optional[float] = None


class DroidSlamConfig(_SlamConfigBase):
    """DROID-SLAM. Dense optical-flow based; VRAM-heavy but globally
    consistent on long indoor/outdoor sequences. Calibration should be
    provided for best results — `auto` estimates from FOV but wobbles."""

    processor: Literal["droid_slam"] = "droid_slam"
    # Upstream caps tracked keyframes; we expose the buffer cap so users
    # with 16 GB cards can drop it from the 512 default.
    buffer_size: int = 512
    # Every Nth frame becomes a DROID keyframe; DROID's own policy also
    # adds extra keyframes when optical-flow magnitude is high.
    keyframe_interval: int = 4
    # Global bundle-adjustment iterations after the forward pass.
    global_ba_iters: int = 25


class Mast3rSlamConfig(_SlamConfigBase):
    """MASt3R-SLAM. Calibration-free — the best default for analog FPV
    footage where fx/fy aren't reliably known. Uses a score-gated keyframe
    policy by default to drop blurred frames before tracking."""

    processor: Literal["mast3r_slam"] = "mast3r_slam"
    # MASt3R's own matcher threshold; lower = more matches per pair, more
    # VRAM. 0.1 is the upstream default.
    match_threshold: float = 0.1
    # Frames per tracking window. Smaller windows are cheaper but miss
    # loop closures on flythrough sequences.
    window_size: int = 16


class DpvoConfig(_SlamConfigBase):
    """DPVO (Deep Patch VO). Lightweight patch-based VO; best for long
    clips or low-VRAM machines. Produces a sparse cloud — enable
    `run_poisson_mesh=False` and feed into a follow-on GS job for dense
    reconstruction."""

    processor: Literal["dpvo"] = "dpvo"
    # Upstream default = 96 patches/frame. Lower = cheaper, less stable.
    patch_per_frame: int = 96
    # Max removed keyframes kept in the sparse cloud — higher = denser
    # trajectory, more memory.
    buffer_keyframes: int = 2048


class MonogsConfig(_SlamConfigBase):
    """MonoGS (Photo-SLAM variant). Gaussian-splat SLAM — emits a splat
    scene incrementally. Short-circuits Phase 5: if a user picks MonoGS
    they get a splat out of SLAM directly; the downstream gsplat training
    job becomes optional."""

    processor: Literal["monogs"] = "monogs"
    # Refinement iterations after each keyframe is added to the splat.
    refine_iters: int = 50
    # Opacity prune threshold for the final scene.
    prune_opacity: float = 0.005
    # MonoGS doesn't produce a point cloud suitable for Poisson meshing.
    run_poisson_mesh: bool = False


# Keep the old name as a type alias so existing code (ingest.py, preproc.py)
# that typed against SlamConfig continues to work. New code should reference
# the specific per-backend class or _SlamConfigBase.
SlamConfig = (
    DroidSlamConfig | Mast3rSlamConfig | DpvoConfig | MonogsConfig
)


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
# "processor" field. Each SLAM backend gets its own member so per-backend
# fields aren't hidden inside a shared shape.
AnyJobConfig = Annotated[
    Union[
        LingbotConfig,
        DroidSlamConfig,
        Mast3rSlamConfig,
        DpvoConfig,
        MonogsConfig,
        GsplatConfig,
    ],
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
