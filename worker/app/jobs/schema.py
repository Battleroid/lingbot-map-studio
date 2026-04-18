from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

JobStatus = Literal["queued", "ingest", "inference", "export", "ready", "failed"]
EventLevel = Literal["info", "warn", "error", "stdout", "stderr", "debug"]
EventStage = Literal[
    "queue",
    "ingest",
    "checkpoint",
    "inference",
    "export",
    "mesh",
    "artifact",
    "system",
]


def _now() -> datetime:
    return datetime.now(timezone.utc)


class JobConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

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
    # Matches gct_stream constructor kv_cache_scale_frames (default 8).
    num_scale_frames: int = 8
    camera_num_iterations: int = 4
    max_frame_num: int = 1024
    kv_cache_sliding_window: int = 64
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


class Artifact(BaseModel):
    name: str
    kind: Literal["glb", "ply", "obj", "npz", "json"]
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
    config: JobConfig
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


class MeshEditRequest(BaseModel):
    op: Literal["cull", "fill_holes", "decimate", "smooth", "remove_small"]
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
