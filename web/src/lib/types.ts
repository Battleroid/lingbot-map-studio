export type JobStatus =
  | "queued"
  | "ingest"
  | "inference"
  | "export"
  | "slam"
  | "meshing"
  | "training"
  | "ready"
  | "failed"
  | "cancelled";

export type EventLevel =
  | "info"
  | "warn"
  | "error"
  | "stdout"
  | "stderr"
  | "debug";

export type EventStage =
  | "queue"
  | "ingest"
  | "checkpoint"
  | "inference"
  | "slam"
  | "training"
  | "meshing"
  | "export"
  | "mesh"
  | "artifact"
  | "system";

export type InferenceMode = "streaming" | "windowed";

export type ProcessorId =
  | "lingbot"
  | "droid_slam"
  | "mast3r_slam"
  | "dpvo"
  | "monogs"
  | "gsplat";

export type SlamBackend = "droid_slam" | "mast3r_slam" | "dpvo" | "monogs";

export type ProcessorKind = "reconstruction" | "slam" | "gsplat";

export interface LingbotConfig {
  processor: "lingbot";

  model_id: string;
  mode: InferenceMode;
  window_size: number;
  overlap_size: number;
  image_size: number;
  patch_size: number;
  fps: number;
  first_k: number | null;
  stride: number;
  mask_sky: boolean;
  conf_percentile: number;
  keyframe_interval: number;
  num_scale_frames: number;
  camera_num_iterations: number;
  max_frame_num: number;
  kv_cache_sliding_window: number;
  enable_3d_rope: boolean;
  use_sdpa: boolean;
  offload_to_cpu: boolean;
  show_cam: boolean;
  mask_black_bg: boolean;
  mask_white_bg: boolean;

  // Preprocessing
  preproc_fisheye: boolean;
  fisheye_in_fov: number;
  fisheye_out_fov: number;
  preproc_denoise: boolean;
  preproc_osd_mask: boolean;
  osd_mask_samples: number;
  osd_mask_std_threshold: number;
  osd_mask_dilate: number;
  osd_detect_text: boolean;
  osd_edge_persist_frac: number;

  // Guardrails
  vram_soft_limit_gb: number | null;
  partial_snapshot_every?: number;
}

// SLAM config stub — Phase 4 extends with per-backend tunables.
export interface SlamConfig {
  processor: SlamBackend;
  max_frames: number | null;
  downscale: number;
  stride: number;
  fps: number;
  calibration: "auto" | "manual";
  fx: number | null;
  fy: number | null;
  cx: number | null;
  cy: number | null;
  keyframe_policy: "score_gated" | "translation" | "hybrid";

  preproc_fisheye: boolean;
  fisheye_in_fov: number;
  fisheye_out_fov: number;
  preproc_denoise: boolean;
  preproc_osd_mask: boolean;

  vram_soft_limit_gb: number | null;
  partial_snapshot_every: number;
}

// Gaussian-splat training config stub — Phase 5 extends.
export interface GsplatConfig {
  processor: "gsplat";
  source_job_id: string;
  iterations: number;
  sh_degree: number;
  densify_interval: number;
  prune_opacity: number;
  init_from: "point_cloud" | "random";
  preview_every_iters: number;
  vram_soft_limit_gb: number | null;
}

export type AnyJobConfig = LingbotConfig | SlamConfig | GsplatConfig;

// Back-compat alias: most existing UI code only knows the lingbot shape.
// Phase 6 migrates those call sites to be mode-aware.
export type JobConfig = LingbotConfig;

export interface JobSummary {
  id: string;
  status: JobStatus;
  created_at: string;
  updated_at: string;
  frames_total: number | null;
  artifact_count: number;
  processor: ProcessorId;
}

// Known artifact kinds. `suffix`-based routing in the viewer uses this
// enum to decide which layer/tool to mount for each artifact.
export type ArtifactKind =
  | "glb"
  | "ply"
  | "obj"
  | "npz"
  | "json"
  | "splat_ply"
  | "splat_sogs"
  | "pose_graph_json"
  | "keyframes_jsonl";

export interface Artifact {
  name: string;
  kind: ArtifactKind;
  revision: number;
  size_bytes: number;
  created_at: string;
}

export interface Job {
  id: string;
  status: JobStatus;
  config: AnyJobConfig;
  uploads: string[];
  artifacts: Artifact[];
  frames_total: number | null;
  error: string | null;
  created_at: string;
  updated_at: string;
}

export interface ManifestArtifact {
  name: string;
  size: number;
  suffix: string;
}

export interface JobManifest {
  id: string;
  status: JobStatus;
  config: AnyJobConfig;
  artifacts: ManifestArtifact[];
  latest_mesh: string | null;
  frames_total: number | null;
  error: string | null;
}

export interface JobEvent {
  id: number;
  job_id: string;
  stage: EventStage;
  level: EventLevel;
  message: string;
  progress: number | null;
  data: Record<string, unknown>;
  created_at: string;
}

export type MeshOp =
  | "cull"
  | "fill_holes"
  | "decimate"
  | "smooth"
  | "remove_small"
  | "surface_recon";

export interface MeshEditRequest {
  op: MeshOp;
  params?: Record<string, unknown>;
  face_indices?: number[];
  source_revision?: number;
}

export interface ReexportRequest {
  format: "glb" | "ply" | "obj";
  conf_percentile?: number;
  mask_sky?: boolean;
  show_cam?: boolean;
  mask_black_bg?: boolean;
  mask_white_bg?: boolean;
}

export function isLingbotConfig(c: AnyJobConfig): c is LingbotConfig {
  return c.processor === "lingbot";
}

export function isSlamConfig(c: AnyJobConfig): c is SlamConfig {
  return (
    c.processor === "droid_slam" ||
    c.processor === "mast3r_slam" ||
    c.processor === "dpvo" ||
    c.processor === "monogs"
  );
}

export function isGsplatConfig(c: AnyJobConfig): c is GsplatConfig {
  return c.processor === "gsplat";
}

export function processorKind(c: AnyJobConfig): ProcessorKind {
  if (isLingbotConfig(c)) return "reconstruction";
  if (isGsplatConfig(c)) return "gsplat";
  return "slam";
}

export const DEFAULT_CONFIG: LingbotConfig = {
  processor: "lingbot",
  model_id: "lingbot-map",
  mode: "streaming",
  window_size: 64,
  overlap_size: 16,
  image_size: 518,
  patch_size: 14,
  fps: 10,
  first_k: null,
  stride: 1,
  mask_sky: true,
  conf_percentile: 50,
  keyframe_interval: 6,
  num_scale_frames: 4,
  camera_num_iterations: 4,
  max_frame_num: 1024,
  kv_cache_sliding_window: 32,
  enable_3d_rope: true,
  use_sdpa: true,
  offload_to_cpu: true,
  show_cam: true,
  mask_black_bg: false,
  mask_white_bg: false,
  preproc_fisheye: false,
  fisheye_in_fov: 165,
  fisheye_out_fov: 90,
  preproc_denoise: false,
  preproc_osd_mask: false,
  osd_mask_samples: 60,
  osd_mask_std_threshold: 5,
  osd_mask_dilate: 2,
  osd_detect_text: true,
  osd_edge_persist_frac: 0.75,
  vram_soft_limit_gb: null,
};

export const PRESETS: Record<string, Partial<LingbotConfig>> = {
  "low-mem": {
    // Aggressive VRAM reduction — for longer clips or smaller cards.
    // NOTE: image_size stays at 518 because the pretrained checkpoint's
    // positional embeddings are fixed to that grid (518/14 = 37×37 tokens).
    // Memory savings come from windowed mode + small window + dropped fps.
    mode: "windowed",
    window_size: 32,
    overlap_size: 8,
    image_size: 518,
    fps: 8,
    num_scale_frames: 2,
    keyframe_interval: 6,
    kv_cache_sliding_window: 16,
    camera_num_iterations: 2,
    offload_to_cpu: true,
    use_sdpa: true,
    mask_sky: true,
    conf_percentile: 65,
  },
  "fpv drone": {
    mask_sky: true,
    conf_percentile: 70,
    keyframe_interval: 4,
    num_scale_frames: 4,
    camera_num_iterations: 2,
    mode: "streaming",
    preproc_denoise: true,
    preproc_fisheye: true,
    fisheye_in_fov: 165,
    fisheye_out_fov: 90,
    preproc_osd_mask: true,
  },
  "low-fi": {
    mask_sky: true,
    conf_percentile: 65,
    keyframe_interval: 4,
    num_scale_frames: 4,
    camera_num_iterations: 2,
    mode: "streaming",
    preproc_denoise: true,
    preproc_fisheye: false,
    preproc_osd_mask: true,
  },
  "high-fi": {
    mask_sky: false,
    conf_percentile: 35,
    keyframe_interval: 6,
    num_scale_frames: 6,
    camera_num_iterations: 4,
    mode: "streaming",
    preproc_denoise: false,
    preproc_fisheye: false,
    preproc_osd_mask: false,
  },
};
