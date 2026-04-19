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

// FPV-oriented preprocessing knobs shared between lingbot + SLAM. Mirrors
// the `PreprocFields` mixin in worker/app/jobs/schema.py.
export interface PreprocFields {
  preproc_fisheye: boolean;
  fisheye_in_fov: number;
  fisheye_out_fov: number;
  preproc_denoise: boolean;
  preproc_analog_cleanup: boolean;
  preproc_deflicker: boolean;
  preproc_osd_mask: boolean;
  osd_mask_samples: number;
  osd_mask_std_threshold: number;
  osd_mask_dilate: number;
  osd_detect_text: boolean;
  osd_edge_persist_frac: number;
  preproc_color_norm: boolean;
  preproc_rs_correction: boolean;
  rs_shear_px_per_row: number | null;
  preproc_deblur: "none" | "unsharp" | "nafnet";
  deblur_sharpness_gate: number;
  preproc_keyframe_score: boolean;
  keyframe_min_sharpness_frac: number;
  keyframe_min_motion_px: number;
}

export interface LingbotConfig extends PreprocFields {
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

  // Guardrails
  vram_soft_limit_gb: number | null;
  partial_snapshot_every?: number;
}

// Shared SLAM tunables. Phase 4's per-backend configs extend this; a plain
// `SlamConfig` alias is exported below as the union of the four concrete
// shapes so mode-aware UI code can narrow on the discriminator.
export interface SlamConfigBase extends PreprocFields {
  model_id: string;
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
  keyframe_interval: number;
  score_gate_quantile: number;
  partial_snapshot_every: number;
  run_poisson_mesh: boolean;
  poisson_depth: number;
  vram_soft_limit_gb: number | null;
}

export interface DroidSlamConfig extends SlamConfigBase {
  processor: "droid_slam";
  buffer_size: number;
  global_ba_iters: number;
}

export interface Mast3rSlamConfig extends SlamConfigBase {
  processor: "mast3r_slam";
  match_threshold: number;
  window_size: number;
}

export interface DpvoConfig extends SlamConfigBase {
  processor: "dpvo";
  patch_per_frame: number;
  buffer_keyframes: number;
}

export interface MonogsConfig extends SlamConfigBase {
  processor: "monogs";
  refine_iters: number;
  prune_opacity: number;
}

export type SlamConfig =
  | DroidSlamConfig
  | Mast3rSlamConfig
  | DpvoConfig
  | MonogsConfig;

// Gaussian-splat training config. Mirrors worker/app/jobs/schema.py.
export interface GsplatConfig {
  processor: "gsplat";
  source_job_id: string;
  iterations: number;
  sh_degree: number;
  densify_interval: number;
  prune_interval: number;
  prune_opacity: number;
  init_from: "point_cloud" | "random";
  random_init_count: number;
  initial_resolution: number;
  upsample_at_iter: number;
  preview_every_iters: number;
  preview_max_gaussians: number;
  bake_mesh_after: boolean;
  bake_mesh_depth: number;
  vram_soft_limit_gb: number | null;
}

export const DEFAULT_GSPLAT_CONFIG: Omit<GsplatConfig, "source_job_id"> = {
  processor: "gsplat",
  iterations: 30_000,
  sh_degree: 3,
  densify_interval: 500,
  prune_interval: 200,
  prune_opacity: 0.005,
  init_from: "point_cloud",
  random_init_count: 100_000,
  initial_resolution: 0.5,
  upsample_at_iter: 5_000,
  preview_every_iters: 1_000,
  preview_max_gaussians: 500_000,
  bake_mesh_after: false,
  bake_mesh_depth: 10,
  vram_soft_limit_gb: null,
};

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
  preproc_analog_cleanup: false,
  preproc_deflicker: false,
  preproc_osd_mask: false,
  osd_mask_samples: 60,
  osd_mask_std_threshold: 5,
  osd_mask_dilate: 2,
  osd_detect_text: true,
  osd_edge_persist_frac: 0.75,
  preproc_color_norm: false,
  preproc_rs_correction: false,
  rs_shear_px_per_row: null,
  preproc_deblur: "none",
  deblur_sharpness_gate: 0.6,
  preproc_keyframe_score: false,
  keyframe_min_sharpness_frac: 0,
  keyframe_min_motion_px: 0,
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

// Shared SLAM defaults. Each per-backend default spreads these, then adds
// its own backend-specific tunables. Mirrors the defaults in
// worker/app/jobs/schema.py (_SlamConfigBase) one-for-one.
const _slamPreprocDefaults: PreprocFields = {
  preproc_fisheye: false,
  fisheye_in_fov: 165,
  fisheye_out_fov: 90,
  preproc_denoise: true,
  preproc_analog_cleanup: true,
  preproc_deflicker: true,
  preproc_osd_mask: true,
  osd_mask_samples: 60,
  osd_mask_std_threshold: 5,
  osd_mask_dilate: 2,
  osd_detect_text: true,
  osd_edge_persist_frac: 0.75,
  preproc_color_norm: true,
  preproc_rs_correction: true,
  rs_shear_px_per_row: null,
  preproc_deblur: "unsharp",
  deblur_sharpness_gate: 0.6,
  preproc_keyframe_score: true,
  keyframe_min_sharpness_frac: 0,
  keyframe_min_motion_px: 0,
};

const _slamBaseDefaults: SlamConfigBase = {
  ..._slamPreprocDefaults,
  model_id: "default",
  max_frames: null,
  downscale: 1,
  stride: 1,
  fps: 10,
  calibration: "auto",
  fx: null,
  fy: null,
  cx: null,
  cy: null,
  keyframe_policy: "score_gated",
  keyframe_interval: 6,
  score_gate_quantile: 0.5,
  partial_snapshot_every: 5,
  run_poisson_mesh: false,
  poisson_depth: 8,
  vram_soft_limit_gb: null,
};

export const DEFAULT_SLAM_CONFIGS: {
  droid_slam: DroidSlamConfig;
  mast3r_slam: Mast3rSlamConfig;
  dpvo: DpvoConfig;
  monogs: MonogsConfig;
} = {
  droid_slam: {
    ..._slamBaseDefaults,
    processor: "droid_slam",
    keyframe_interval: 4,
    buffer_size: 512,
    global_ba_iters: 25,
  },
  mast3r_slam: {
    ..._slamBaseDefaults,
    processor: "mast3r_slam",
    match_threshold: 0.1,
    window_size: 16,
  },
  dpvo: {
    ..._slamBaseDefaults,
    processor: "dpvo",
    patch_per_frame: 96,
    buffer_keyframes: 2048,
  },
  monogs: {
    ..._slamBaseDefaults,
    processor: "monogs",
    refine_iters: 50,
    prune_opacity: 0.005,
    run_poisson_mesh: false,
  },
};

// FPV preprocessing presets. Applied on top of the currently-selected base
// preset — these only touch the `preproc_*` fields so they compose.
export const PREPROC_PRESETS: Record<string, Partial<PreprocFields>> = {
  none: {
    preproc_denoise: false,
    preproc_analog_cleanup: false,
    preproc_deflicker: false,
    preproc_osd_mask: false,
    preproc_color_norm: false,
    preproc_rs_correction: false,
    preproc_deblur: "none",
    preproc_keyframe_score: false,
  },
  "analog fpv (default)": {
    // Good default for a low-bitrate/DVR/analog-receiver capture. Keeps
    // the cheap stages on and unsharp deblur active, skips the heavy
    // atadenoise (use "aggressive" for that).
    preproc_denoise: true,
    preproc_deflicker: true,
    preproc_osd_mask: true,
    preproc_color_norm: true,
    preproc_rs_correction: true,
    preproc_deblur: "unsharp",
    preproc_keyframe_score: true,
  },
  aggressive: {
    // Everything on. Slow; reserve for visibly rough clips where the
    // default preset still leaves artefacts.
    preproc_denoise: true,
    preproc_analog_cleanup: true,
    preproc_deflicker: true,
    preproc_osd_mask: true,
    preproc_color_norm: true,
    preproc_rs_correction: true,
    preproc_deblur: "unsharp",
    preproc_keyframe_score: true,
  },
};
