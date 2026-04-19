"use client";

import { Tip } from "@/components/Tip";
import { type JobConfig, PREPROC_PRESETS, PRESETS } from "@/lib/types";

interface Props {
  config: JobConfig;
  onChange: (patch: Partial<JobConfig>) => void;
  readOnly?: boolean;
  compact?: boolean;
  title?: string;
}

const TIPS: Record<string, string> = {
  image_size:
    "Input resolution the model operates at. ⚠️ Keep this at 518 — the pretrained DINOv2 position embeddings are fixed to a 37×37 token grid (518/14). Changing this value will crash with a state_dict size-mismatch. Use windowed mode + smaller window_size for VRAM relief instead.",
  kv_cache_sliding_window:
    "Number of keyframes kept in the attention KV cache. Once this many keyframes accumulate, the oldest are evicted. Lower = lower steady-state VRAM but less long-range context. 16-32 is safe; 64+ can OOM on longer clips.",
  model_id:
    "Which lingbot-map checkpoint to use.\n• lingbot-map: balanced, ~4.6 GB (default).\n• long: tuned for long sequences.\n• stage1: can be loaded into a VGGT base.",
  mode:
    "Inference loop:\n• streaming: one KV cache, best for ≤~3000 frames.\n• windowed: processes in overlapping windows, scales to 10k+ frames.",
  window_size:
    "Windowed mode: number of keyframes per window (including scale frames). Larger = more global context, more VRAM.",
  overlap_size:
    "Windowed mode: frames shared between consecutive windows so their point clouds align.",
  fps:
    "Frames-per-second sampled from the source video for reconstruction. Lower = fewer frames = faster, less detailed. Auto-suggested from the video's native fps (capped at 10).",
  conf_percentile:
    "Percentile cutoff on per-point confidence when exporting the point cloud.\n• Higher (70-90) = strict, fewer but cleaner points — good for noisy drone.\n• Lower (20-40) = permissive, more points, more noise.",
  keyframe_interval:
    "Every Nth frame updates the KV cache as a keyframe; frames between are predicted without extending the cache. Larger = less memory, slightly worse quality.",
  num_scale_frames:
    "Frames used in the first batched pass to anchor scale. 2-4 for short/cheap, 8 for best quality.",
  camera_num_iterations:
    "Refinement iterations inside the camera head. 2 = fast, 4 = more accurate pose.",
  mask_sky:
    "Run a tiny sky-segmentation ONNX pass and zero the confidence for sky pixels before export. Essential for outdoor/drone footage; auto-downloads skyseg.onnx on first use.",
  use_sdpa:
    "Use PyTorch's scaled_dot_product_attention instead of FlashInfer's paged KV cache. Leave on unless you built with flashinfer-python.",
  offload_to_cpu:
    "Move per-frame outputs to CPU as they're produced, saving GPU memory for longer sequences. Small speed cost.",
  show_cam:
    "Include camera frustum glyphs in the exported GLB. Useful for debugging alignment; turn off for pure geometry.",
  mask_black_bg:
    "Drop points whose source-image color is pure black (often synthetic padding).",
  mask_white_bg:
    "Drop points whose source-image color is pure white (often overexposed sky or greenscreens).",
  fill_preset: "Apply a preset tuned for low-fidelity drone or higher-fidelity source material.",
  preproc_fisheye:
    "Unwrap fisheye-lens footage to rectilinear before reconstruction. Most FPV micro cams (2.1-2.5 mm) are 150-170° — leaving them distorted gives the model false geometry. The unwrap crops the rim where distortion was worst.",
  fisheye_in_fov:
    "Source lens horizontal FOV in degrees. Measure or look up your cam: Caddx Ratel 2.1 ≈ 165°, RunCam Phoenix 2 ≈ 155°, Foxeer Predator Micro ≈ 160°. Wrong value = skewed geometry.",
  fisheye_out_fov:
    "Target diagonal FOV after unwrap. 90° keeps the sharper centre, 110-120° keeps more peripheral content at the cost of residual edge distortion.",
  preproc_denoise:
    "Temporal denoise (hqdn3d) + deflicker. Reduces analog static, snow, and per-frame luma jitter from analog FPV feeds. Adds a few seconds to ingest; safe to leave on for any noisy source.",
  preproc_osd_mask:
    "Detect pixels that do not change over time (telemetry text, timer, battery, home arrow, station logo) and inpaint them out of every frame before reconstruction. Without this they become false geometry fixed in camera space.",
  osd_mask_samples:
    "How many frames to sample when computing the static-pixel mask. More = better detection but slower mask computation.",
  osd_mask_std_threshold:
    "Per-pixel standard-deviation cutoff (0-255 scale). Pixels below this are treated as static overlay. 5 is a conservative default; raise to 10-15 for aggressive masking, drop for subtle overlays.",
  osd_mask_dilate:
    "Morphological dilation iterations on the mask — grows it outward to catch anti-aliased text edges. 2-3 usually enough.",
  osd_detect_text:
    "Second OSD signal: flag regions that are near an edge in most sampled frames. Catches changing numeric HUD values (e.g. battery voltage ticking down) that the stddev-only detector misses because the digit pixels themselves change.",
  osd_edge_persist_frac:
    "Fraction of frames (0-1) where a pixel must be near an edge to be flagged as text. Higher = stricter (fewer false positives on scene edges), lower = more aggressive. 0.75 is a balanced default.",
  vram_soft_limit_gb:
    "Per-job VRAM soft limit in GB. A background watchdog samples GPU memory every 2s during inference; if allocated VRAM exceeds this, the job is aborted cleanly before the kernel kills the process. Leave blank to use the worker default (22 GB on a 24 GB card). The worker also enforces a hard process-wide cap to keep WSL2 from hanging the host.",
  preproc_analog_cleanup:
    "Heavier temporal denoise (atadenoise) tuned for VHS/analog chroma noise. Expensive — only enable for genuinely noisy captures where the standard hqdn3d pair still leaves dot crawl or chroma fringing. No-op on clean digital clips.",
  preproc_deflicker:
    "Standalone ffmpeg deflicker without the hqdn3d denoise pair. Useful when brightness jitter is the dominant problem and you don't want the spatial blurring that hqdn3d adds. Auto-skipped if `denoise + deflicker` is already on.",
  preproc_color_norm:
    "Per-frame grey-world white-balance + 1/99-percentile histogram stretch. Recovers natural colour on green/magenta-tinted analog feeds. Cheap, CPU-only.",
  preproc_rs_correction:
    "Global y-shear correction for rolling-shutter skew. Estimates a single px-per-row shear from optical flow between sampled frames, applies the inverse affine warp to every frame. Full per-row RS (needs gyro data) is out of scope.",
  rs_shear_px_per_row:
    "Override the estimated shear in pixels per row. Leave blank to let the estimator pick. Negative values tilt the other way. Values smaller than ±0.02 px/row skip the warp entirely.",
  preproc_deblur:
    "Motion-deblur strategy:\n• none — off (default, cheapest).\n• unsharp — classical unsharp-mask gated by per-frame Laplacian variance. Fast, CPU.\n• nafnet — learned single-image deblur (Phase 3 wires the hook; the checkpoint ships in a follow-up, falls back to unsharp for now).",
  deblur_sharpness_gate:
    "Apply the deblur filter only to frames whose Laplacian variance is below this fraction of the clip median. 1.0 = every frame, 0.6 (default) = the blurriest ~60%, 0.3 = only the worst offenders.",
  preproc_keyframe_score:
    "Write `frame_scores.jsonl` with per-frame sharpness and optical-flow magnitude. SLAM backends with `keyframe_policy=score_gated` read this to drop low-quality frames before keyframe selection. Cheap to leave on; downstream backends that don't consume it just ignore the file.",
  preproc_preset:
    "Bundled FPV preprocessing preset. Toggles all preproc_* fields at once; the basic sliders above are unaffected. `analog fpv (default)` enables the cheap stages (deflicker, colour norm, rolling-shutter, unsharp deblur, keyframe scoring). `aggressive` layers atadenoise on top. `none` disables everything.",
};

function NumberRow({
  label,
  tipKey,
  value,
  onChange,
  step = 1,
  min,
  max,
  readOnly,
}: {
  label: string;
  tipKey: string;
  value: number;
  onChange: (v: number) => void;
  step?: number;
  min?: number;
  max?: number;
  readOnly?: boolean;
}) {
  return (
    <label className="stat">
      <Tip text={TIPS[tipKey] ?? ""}>
        <span>{label}</span>
      </Tip>
      <input
        type="number"
        value={value}
        step={step}
        min={min}
        max={max}
        readOnly={readOnly}
        disabled={readOnly}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}

function BoolRow({
  label,
  tipKey,
  value,
  onChange,
  readOnly,
}: {
  label: string;
  tipKey: string;
  value: boolean;
  onChange: (v: boolean) => void;
  readOnly?: boolean;
}) {
  return (
    <label className="stat">
      <Tip text={TIPS[tipKey] ?? ""}>
        <span>{label}</span>
      </Tip>
      <input
        type="checkbox"
        checked={value}
        disabled={readOnly}
        onChange={(e) => onChange(e.target.checked)}
      />
    </label>
  );
}

export function ConfigPanel({ config, onChange, readOnly, compact, title }: Props) {
  return (
    <div className="panel">
      <div className="panel-header">
        <span>{title ?? "config"}</span>
        {readOnly && <span className="meta">locked</span>}
      </div>
      <div className="panel-body" style={{ display: "grid", gap: 6 }}>
        {!readOnly && (
          <div style={{ display: "flex", gap: 4 }}>
            <Tip text={TIPS.fill_preset} showIcon={false}>
              <span className="section-title">presets</span>
            </Tip>
          </div>
        )}
        {!readOnly && (
          <div style={{ display: "flex", gap: 4 }}>
            {Object.entries(PRESETS).map(([name, patch]) => (
              <button
                key={name}
                type="button"
                onClick={() => onChange(patch)}
                style={{ flex: 1 }}
              >
                {name}
              </button>
            ))}
          </div>
        )}

        <label className="stat">
          <Tip text={TIPS.model_id}>
            <span>model</span>
          </Tip>
          <select
            value={config.model_id}
            disabled={readOnly}
            onChange={(e) => onChange({ model_id: e.target.value })}
          >
            <option value="lingbot-map">lingbot-map</option>
            <option value="lingbot-map-long">lingbot-map-long</option>
            <option value="lingbot-map-stage1">lingbot-map-stage1</option>
          </select>
        </label>

        <label className="stat">
          <Tip text={TIPS.mode}>
            <span>mode</span>
          </Tip>
          <select
            value={config.mode}
            disabled={readOnly}
            onChange={(e) =>
              onChange({ mode: e.target.value as JobConfig["mode"] })
            }
          >
            <option value="streaming">streaming</option>
            <option value="windowed">windowed</option>
          </select>
        </label>

        <NumberRow
          label="fps"
          tipKey="fps"
          value={config.fps}
          step={0.5}
          min={0.5}
          max={60}
          readOnly={readOnly}
          onChange={(v) => onChange({ fps: v })}
        />
        <NumberRow
          label="conf %ile"
          tipKey="conf_percentile"
          value={config.conf_percentile}
          step={5}
          min={0}
          max={95}
          readOnly={readOnly}
          onChange={(v) => onChange({ conf_percentile: v })}
        />
        <NumberRow
          label="keyframe interval"
          tipKey="keyframe_interval"
          value={config.keyframe_interval}
          step={1}
          min={1}
          max={30}
          readOnly={readOnly}
          onChange={(v) => onChange({ keyframe_interval: v })}
        />
        <NumberRow
          label="num scale frames"
          tipKey="num_scale_frames"
          value={config.num_scale_frames}
          step={1}
          min={1}
          max={16}
          readOnly={readOnly}
          onChange={(v) => onChange({ num_scale_frames: v })}
        />
        <NumberRow
          label="kv cache window"
          tipKey="kv_cache_sliding_window"
          value={config.kv_cache_sliding_window}
          step={8}
          min={8}
          max={256}
          readOnly={readOnly}
          onChange={(v) => onChange({ kv_cache_sliding_window: v })}
        />
        <NumberRow
          label="image size"
          tipKey="image_size"
          value={config.image_size}
          step={14}
          min={224}
          max={700}
          readOnly={readOnly}
          onChange={(v) => onChange({ image_size: v })}
        />
        <NumberRow
          label="camera iters"
          tipKey="camera_num_iterations"
          value={config.camera_num_iterations}
          step={1}
          min={1}
          max={8}
          readOnly={readOnly}
          onChange={(v) => onChange({ camera_num_iterations: v })}
        />
        <BoolRow
          label="mask sky"
          tipKey="mask_sky"
          value={config.mask_sky}
          readOnly={readOnly}
          onChange={(v) => onChange({ mask_sky: v })}
        />
        <BoolRow
          label="use sdpa"
          tipKey="use_sdpa"
          value={config.use_sdpa}
          readOnly={readOnly}
          onChange={(v) => onChange({ use_sdpa: v })}
        />
        <BoolRow
          label="offload cpu"
          tipKey="offload_to_cpu"
          value={config.offload_to_cpu}
          readOnly={readOnly}
          onChange={(v) => onChange({ offload_to_cpu: v })}
        />
        <BoolRow
          label="show cameras"
          tipKey="show_cam"
          value={config.show_cam}
          readOnly={readOnly}
          onChange={(v) => onChange({ show_cam: v })}
        />
        <BoolRow
          label="mask black bg"
          tipKey="mask_black_bg"
          value={config.mask_black_bg}
          readOnly={readOnly}
          onChange={(v) => onChange({ mask_black_bg: v })}
        />
        <BoolRow
          label="mask white bg"
          tipKey="mask_white_bg"
          value={config.mask_white_bg}
          readOnly={readOnly}
          onChange={(v) => onChange({ mask_white_bg: v })}
        />

        {!compact && config.mode === "windowed" && (
          <>
            <NumberRow
              label="window size"
              tipKey="window_size"
              value={config.window_size}
              step={8}
              min={16}
              max={512}
              readOnly={readOnly}
              onChange={(v) => onChange({ window_size: v })}
            />
            <NumberRow
              label="overlap"
              tipKey="overlap_size"
              value={config.overlap_size}
              step={4}
              min={0}
              max={128}
              readOnly={readOnly}
              onChange={(v) => onChange({ overlap_size: v })}
            />
          </>
        )}

        <div
          style={{
            marginTop: 6,
            paddingTop: 6,
            borderTop: "1px solid var(--rule)",
          }}
        >
          <div className="section-title">preprocessing</div>
        </div>

        <BoolRow
          label="fisheye unwrap"
          tipKey="preproc_fisheye"
          value={config.preproc_fisheye}
          readOnly={readOnly}
          onChange={(v) => onChange({ preproc_fisheye: v })}
        />
        {config.preproc_fisheye && (
          <>
            <NumberRow
              label="fisheye in fov"
              tipKey="fisheye_in_fov"
              value={config.fisheye_in_fov}
              step={5}
              min={60}
              max={180}
              readOnly={readOnly}
              onChange={(v) => onChange({ fisheye_in_fov: v })}
            />
            <NumberRow
              label="fisheye out fov"
              tipKey="fisheye_out_fov"
              value={config.fisheye_out_fov}
              step={5}
              min={40}
              max={140}
              readOnly={readOnly}
              onChange={(v) => onChange({ fisheye_out_fov: v })}
            />
          </>
        )}
        <BoolRow
          label="denoise + deflicker"
          tipKey="preproc_denoise"
          value={config.preproc_denoise}
          readOnly={readOnly}
          onChange={(v) => onChange({ preproc_denoise: v })}
        />
        <BoolRow
          label="mask osd text"
          tipKey="preproc_osd_mask"
          value={config.preproc_osd_mask}
          readOnly={readOnly}
          onChange={(v) => onChange({ preproc_osd_mask: v })}
        />
        {!compact && config.preproc_osd_mask && (
          <>
            <NumberRow
              label="osd · samples"
              tipKey="osd_mask_samples"
              value={config.osd_mask_samples}
              step={10}
              min={10}
              max={400}
              readOnly={readOnly}
              onChange={(v) => onChange({ osd_mask_samples: v })}
            />
            <NumberRow
              label="osd · stddev thr"
              tipKey="osd_mask_std_threshold"
              value={config.osd_mask_std_threshold}
              step={0.5}
              min={0.5}
              max={30}
              readOnly={readOnly}
              onChange={(v) => onChange({ osd_mask_std_threshold: v })}
            />
            <NumberRow
              label="osd · dilate"
              tipKey="osd_mask_dilate"
              value={config.osd_mask_dilate}
              step={1}
              min={0}
              max={10}
              readOnly={readOnly}
              onChange={(v) => onChange({ osd_mask_dilate: v })}
            />
            <BoolRow
              label="osd · detect text"
              tipKey="osd_detect_text"
              value={config.osd_detect_text}
              readOnly={readOnly}
              onChange={(v) => onChange({ osd_detect_text: v })}
            />
            {config.osd_detect_text && (
              <NumberRow
                label="osd · edge persist"
                tipKey="osd_edge_persist_frac"
                value={config.osd_edge_persist_frac}
                step={0.05}
                min={0.3}
                max={0.99}
                readOnly={readOnly}
                onChange={(v) => onChange({ osd_edge_persist_frac: v })}
              />
            )}
          </>
        )}

        <div
          style={{
            marginTop: 6,
            paddingTop: 6,
            borderTop: "1px solid var(--rule)",
          }}
        >
          <div className="section-title">fpv preprocessing</div>
        </div>
        {!readOnly && (
          <div style={{ display: "flex", gap: 4 }}>
            <Tip text={TIPS.preproc_preset} showIcon={false}>
              <span className="section-title" style={{ opacity: 0.8 }}>
                presets
              </span>
            </Tip>
          </div>
        )}
        {!readOnly && (
          <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
            {Object.entries(PREPROC_PRESETS).map(([name, patch]) => (
              <button
                key={name}
                type="button"
                onClick={() => onChange(patch as Partial<JobConfig>)}
                style={{ flex: 1, minWidth: 0 }}
              >
                {name}
              </button>
            ))}
          </div>
        )}
        <BoolRow
          label="analog cleanup (atadenoise)"
          tipKey="preproc_analog_cleanup"
          value={config.preproc_analog_cleanup}
          readOnly={readOnly}
          onChange={(v) => onChange({ preproc_analog_cleanup: v })}
        />
        <BoolRow
          label="standalone deflicker"
          tipKey="preproc_deflicker"
          value={config.preproc_deflicker}
          readOnly={readOnly}
          onChange={(v) => onChange({ preproc_deflicker: v })}
        />
        <BoolRow
          label="colour normalisation"
          tipKey="preproc_color_norm"
          value={config.preproc_color_norm}
          readOnly={readOnly}
          onChange={(v) => onChange({ preproc_color_norm: v })}
        />
        <BoolRow
          label="rolling-shutter correction"
          tipKey="preproc_rs_correction"
          value={config.preproc_rs_correction}
          readOnly={readOnly}
          onChange={(v) => onChange({ preproc_rs_correction: v })}
        />
        {!compact && config.preproc_rs_correction && (
          <label className="stat">
            <Tip text={TIPS.rs_shear_px_per_row}>
              <span>rs · shear px/row</span>
            </Tip>
            <input
              type="number"
              value={config.rs_shear_px_per_row ?? ""}
              step={0.01}
              placeholder="auto"
              readOnly={readOnly}
              disabled={readOnly}
              onChange={(e) =>
                onChange({
                  rs_shear_px_per_row:
                    e.target.value === "" ? null : Number(e.target.value),
                })
              }
            />
          </label>
        )}
        <label className="stat">
          <Tip text={TIPS.preproc_deblur}>
            <span>deblur</span>
          </Tip>
          <select
            value={config.preproc_deblur}
            disabled={readOnly}
            onChange={(e) =>
              onChange({
                preproc_deblur: e.target.value as JobConfig["preproc_deblur"],
              })
            }
          >
            <option value="none">none</option>
            <option value="unsharp">unsharp</option>
            <option value="nafnet">nafnet</option>
          </select>
        </label>
        {!compact && config.preproc_deblur !== "none" && (
          <NumberRow
            label="deblur · sharpness gate"
            tipKey="deblur_sharpness_gate"
            value={config.deblur_sharpness_gate}
            step={0.05}
            min={0.1}
            max={1.5}
            readOnly={readOnly}
            onChange={(v) => onChange({ deblur_sharpness_gate: v })}
          />
        )}
        <BoolRow
          label="keyframe scoring"
          tipKey="preproc_keyframe_score"
          value={config.preproc_keyframe_score}
          readOnly={readOnly}
          onChange={(v) => onChange({ preproc_keyframe_score: v })}
        />

        <div
          style={{
            marginTop: 6,
            paddingTop: 6,
            borderTop: "1px solid var(--rule)",
          }}
        >
          <div className="section-title">guardrails</div>
        </div>
        <label className="stat">
          <span>
            <span className="tip-target" data-tip={TIPS.vram_soft_limit_gb} tabIndex={0}>
              vram soft limit (gb)
              <span className="tip-icon">?</span>
            </span>
          </span>
          <input
            type="number"
            value={config.vram_soft_limit_gb ?? ""}
            step={1}
            min={1}
            max={80}
            placeholder="default"
            readOnly={readOnly}
            disabled={readOnly}
            onChange={(e) =>
              onChange({
                vram_soft_limit_gb:
                  e.target.value === "" ? null : Number(e.target.value),
              })
            }
          />
        </label>
      </div>
    </div>
  );
}
