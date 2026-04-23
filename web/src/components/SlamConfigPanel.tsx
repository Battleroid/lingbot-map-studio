"use client";

import { Tip } from "@/components/Tip";
import {
  type DpvoConfig,
  type DroidSlamConfig,
  type Mast3rSlamConfig,
  type MonogsConfig,
  PREPROC_PRESETS,
  type PreprocFields,
  type SlamConfig,
} from "@/lib/types";

interface Props {
  config: SlamConfig;
  onChange: (patch: Partial<SlamConfig>) => void;
  readOnly?: boolean;
  compact?: boolean;
  title?: string;
}

const TIPS: Record<string, string> = {
  max_frames:
    "Hard cap on the number of input frames handed to the tracker. Blank = run the entire clip. Useful for timing-box tests or long clips where you only care about the first N frames.",
  downscale:
    "Integer scale factor applied to each input frame before tracking. 1 = full resolution, 2 = half (4x less work). Helps heavy backends (DROID-SLAM) fit on smaller cards.",
  stride:
    "Frame skip. 1 = every frame, 2 = every other. Cheap way to drop framerate when the FPS of the capture is much higher than the camera actually moved.",
  fps: "Resample source video to this fps before tracking. SLAM is frame-count-sensitive, not wall-clock sensitive — lower = fewer frames to process.",
  calibration:
    "Camera intrinsics.\n• auto: estimate fx/fy from a 60° horizontal FOV assumption (safe default for MASt3R-SLAM which is calibration-free anyway).\n• manual: provide fx/fy/cx/cy below.",
  keyframe_policy:
    "How new keyframes are selected:\n• score_gated: use the FPV keyframe_score stage as a gate (recommended for analog footage).\n• translation: classic — drop a keyframe when the camera has moved more than a threshold.\n• hybrid: both — translation triggers but also require a min sharpness score.",
  keyframe_interval:
    "Minimum frame gap between keyframes. Tighter = more graph density, slower, better under jittery motion.",
  score_gate_quantile:
    "Only used when keyframe_policy=score_gated. Keep frames whose combined sharpness+motion score is above this quantile. 0.5 drops the blurriest/most-static half.",
  partial_snapshot_every:
    "Every N keyframes, write a partial_XXXX.ply snapshot for the live viewer. Lower = smoother live preview but more disk writes.",
  run_poisson_mesh:
    "After tracking, run Poisson surface reconstruction on the sparse cloud to produce a .glb you can load into the existing mesh tools. Slow for dense clouds; free on sparse ones.",
  poisson_depth:
    "Poisson octree depth. 7 coarse/fast, 9 detailed/slow. 8 is a good middle ground.",
  buffer_size:
    "DROID-SLAM keyframe buffer capacity. Higher = more global context in bundle adjustment, but also the biggest VRAM cost. Pin by card: 24 GB → 512, 16 GB → 256.",
  global_ba_iters:
    "DROID-SLAM global bundle-adjustment iterations at finalization. 10-25 typical. More iters polishes the trajectory at a few seconds' cost.",
  match_threshold:
    "MASt3R-SLAM: minimum correspondence confidence to accept a feature match. Lower = looser (more edges, noisier), higher = stricter.",
  window_size:
    "MASt3R-SLAM: local sliding window length used for two-view matching around the current keyframe.",
  patch_per_frame:
    "DPVO patches per frame. 96 default. More = denser tracks = higher accuracy at higher VRAM/CPU cost.",
  buffer_keyframes:
    "DPVO patch buffer cap. Larger = more history in local BA, but more memory.",
  refine_iters:
    "MonoGS/Photo-SLAM refinement iterations per keyframe. Drives how aggressively the on-the-fly splat is fit.",
  prune_opacity:
    "MonoGS/Photo-SLAM: drop gaussians whose opacity falls below this value during training. Keeps the splat lean.",
  vram_soft_limit_gb:
    "Abort the job cleanly if allocated VRAM crosses this threshold. Leave blank to use the worker default (22 GB on a 24 GB card).",
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
  placeholder,
}: {
  label: string;
  tipKey: string;
  value: number | null;
  onChange: (v: number | null) => void;
  step?: number;
  min?: number;
  max?: number;
  readOnly?: boolean;
  placeholder?: string;
}) {
  return (
    <label className="stat">
      <Tip text={TIPS[tipKey] ?? ""}>
        <span>{label}</span>
      </Tip>
      <input
        type="number"
        value={value ?? ""}
        step={step}
        min={min}
        max={max}
        readOnly={readOnly}
        disabled={readOnly}
        placeholder={placeholder}
        onChange={(e) =>
          onChange(e.target.value === "" ? null : Number(e.target.value))
        }
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

function isDroid(c: SlamConfig): c is DroidSlamConfig {
  return c.processor === "droid_slam";
}
function isMast3r(c: SlamConfig): c is Mast3rSlamConfig {
  return c.processor === "mast3r_slam";
}
function isDpvo(c: SlamConfig): c is DpvoConfig {
  return c.processor === "dpvo";
}
function isMonogs(c: SlamConfig): c is MonogsConfig {
  return c.processor === "monogs";
}

export function SlamConfigPanel({
  config,
  onChange,
  readOnly,
  compact,
  title,
}: Props) {
  return (
    <div className="panel">
      <div className="panel-header">
        <span>{title ?? `config · ${config.processor}`}</span>
        {readOnly && <span className="meta">locked</span>}
      </div>
      <div className="panel-body" style={{ display: "grid", gap: 6 }}>
        <NumberRow
          label="max frames"
          tipKey="max_frames"
          value={config.max_frames}
          step={100}
          min={1}
          placeholder="all"
          readOnly={readOnly}
          onChange={(v) => onChange({ max_frames: v } as Partial<SlamConfig>)}
        />
        <NumberRow
          label="downscale"
          tipKey="downscale"
          value={config.downscale}
          step={1}
          min={1}
          max={8}
          readOnly={readOnly}
          onChange={(v) =>
            onChange({ downscale: (v ?? 1) as number } as Partial<SlamConfig>)
          }
        />
        <NumberRow
          label="stride"
          tipKey="stride"
          value={config.stride}
          step={1}
          min={1}
          max={10}
          readOnly={readOnly}
          onChange={(v) =>
            onChange({ stride: (v ?? 1) as number } as Partial<SlamConfig>)
          }
        />
        <NumberRow
          label="fps"
          tipKey="fps"
          value={config.fps}
          step={0.5}
          min={0.5}
          max={60}
          readOnly={readOnly}
          onChange={(v) =>
            onChange({ fps: (v ?? 10) as number } as Partial<SlamConfig>)
          }
        />

        <label className="stat">
          <Tip text={TIPS.calibration}>
            <span>calibration</span>
          </Tip>
          <select
            value={config.calibration}
            disabled={readOnly}
            onChange={(e) =>
              onChange({
                calibration: e.target.value as SlamConfig["calibration"],
              } as Partial<SlamConfig>)
            }
          >
            <option value="auto">auto</option>
            <option value="manual">manual</option>
          </select>
        </label>
        {config.calibration === "manual" && !compact && (
          <>
            <NumberRow
              label="fx"
              tipKey="calibration"
              value={config.fx}
              step={10}
              min={1}
              placeholder="px"
              readOnly={readOnly}
              onChange={(v) => onChange({ fx: v } as Partial<SlamConfig>)}
            />
            <NumberRow
              label="fy"
              tipKey="calibration"
              value={config.fy}
              step={10}
              min={1}
              placeholder="px"
              readOnly={readOnly}
              onChange={(v) => onChange({ fy: v } as Partial<SlamConfig>)}
            />
            <NumberRow
              label="cx"
              tipKey="calibration"
              value={config.cx}
              step={10}
              min={0}
              placeholder="px"
              readOnly={readOnly}
              onChange={(v) => onChange({ cx: v } as Partial<SlamConfig>)}
            />
            <NumberRow
              label="cy"
              tipKey="calibration"
              value={config.cy}
              step={10}
              min={0}
              placeholder="px"
              readOnly={readOnly}
              onChange={(v) => onChange({ cy: v } as Partial<SlamConfig>)}
            />
          </>
        )}

        <label className="stat">
          <Tip text={TIPS.keyframe_policy}>
            <span>keyframe policy</span>
          </Tip>
          <select
            value={config.keyframe_policy}
            disabled={readOnly}
            onChange={(e) =>
              onChange({
                keyframe_policy:
                  e.target.value as SlamConfig["keyframe_policy"],
              } as Partial<SlamConfig>)
            }
          >
            <option value="score_gated">score_gated</option>
            <option value="translation">translation</option>
            <option value="hybrid">hybrid</option>
          </select>
        </label>
        <NumberRow
          label="keyframe interval"
          tipKey="keyframe_interval"
          value={config.keyframe_interval}
          step={1}
          min={1}
          max={30}
          readOnly={readOnly}
          onChange={(v) =>
            onChange({
              keyframe_interval: (v ?? 6) as number,
            } as Partial<SlamConfig>)
          }
        />
        {config.keyframe_policy !== "translation" && (
          <NumberRow
            label="score gate quantile"
            tipKey="score_gate_quantile"
            value={config.score_gate_quantile}
            step={0.05}
            min={0}
            max={1}
            readOnly={readOnly}
            onChange={(v) =>
              onChange({
                score_gate_quantile: (v ?? 0.5) as number,
              } as Partial<SlamConfig>)
            }
          />
        )}
        <NumberRow
          label="partial snapshot every"
          tipKey="partial_snapshot_every"
          value={config.partial_snapshot_every}
          step={1}
          min={1}
          max={50}
          readOnly={readOnly}
          onChange={(v) =>
            onChange({
              partial_snapshot_every: (v ?? 5) as number,
            } as Partial<SlamConfig>)
          }
        />

        <BoolRow
          label="run poisson mesh"
          tipKey="run_poisson_mesh"
          value={config.run_poisson_mesh}
          readOnly={readOnly}
          onChange={(v) =>
            onChange({ run_poisson_mesh: v } as Partial<SlamConfig>)
          }
        />
        {config.run_poisson_mesh && (
          <NumberRow
            label="poisson depth"
            tipKey="poisson_depth"
            value={config.poisson_depth}
            step={1}
            min={5}
            max={11}
            readOnly={readOnly}
            onChange={(v) =>
              onChange({
                poisson_depth: (v ?? 8) as number,
              } as Partial<SlamConfig>)
            }
          />
        )}

        <div
          style={{
            marginTop: 6,
            paddingTop: 6,
            borderTop: "1px solid var(--rule)",
          }}
        >
          <div className="section-title">backend tuning</div>
        </div>

        {isDroid(config) && (
          <>
            <NumberRow
              label="buffer size"
              tipKey="buffer_size"
              value={config.buffer_size}
              step={32}
              min={32}
              max={2048}
              readOnly={readOnly}
              onChange={(v) =>
                onChange({
                  buffer_size: (v ?? 512) as number,
                } as Partial<DroidSlamConfig>)
              }
            />
            <NumberRow
              label="global BA iters"
              tipKey="global_ba_iters"
              value={config.global_ba_iters}
              step={1}
              min={0}
              max={200}
              readOnly={readOnly}
              onChange={(v) =>
                onChange({
                  global_ba_iters: (v ?? 25) as number,
                } as Partial<DroidSlamConfig>)
              }
            />
          </>
        )}
        {isMast3r(config) && (
          <>
            <NumberRow
              label="match threshold"
              tipKey="match_threshold"
              value={config.match_threshold}
              step={0.01}
              min={0}
              max={1}
              readOnly={readOnly}
              onChange={(v) =>
                onChange({
                  match_threshold: (v ?? 0.1) as number,
                } as Partial<Mast3rSlamConfig>)
              }
            />
            <NumberRow
              label="window size"
              tipKey="window_size"
              value={config.window_size}
              step={1}
              min={2}
              max={128}
              readOnly={readOnly}
              onChange={(v) =>
                onChange({
                  window_size: (v ?? 16) as number,
                } as Partial<Mast3rSlamConfig>)
              }
            />
          </>
        )}
        {isDpvo(config) && (
          <>
            <NumberRow
              label="patches/frame"
              tipKey="patch_per_frame"
              value={config.patch_per_frame}
              step={16}
              min={16}
              max={512}
              readOnly={readOnly}
              onChange={(v) =>
                onChange({
                  patch_per_frame: (v ?? 96) as number,
                } as Partial<DpvoConfig>)
              }
            />
            <NumberRow
              label="buffer keyframes"
              tipKey="buffer_keyframes"
              value={config.buffer_keyframes}
              step={128}
              min={128}
              max={16384}
              readOnly={readOnly}
              onChange={(v) =>
                onChange({
                  buffer_keyframes: (v ?? 2048) as number,
                } as Partial<DpvoConfig>)
              }
            />
          </>
        )}
        {isMonogs(config) && (
          <>
            <NumberRow
              label="refine iters/kf"
              tipKey="refine_iters"
              value={config.refine_iters}
              step={5}
              min={1}
              max={500}
              readOnly={readOnly}
              onChange={(v) =>
                onChange({
                  refine_iters: (v ?? 50) as number,
                } as Partial<MonogsConfig>)
              }
            />
            <NumberRow
              label="prune opacity"
              tipKey="prune_opacity"
              value={config.prune_opacity}
              step={0.001}
              min={0}
              max={0.5}
              readOnly={readOnly}
              onChange={(v) =>
                onChange({
                  prune_opacity: (v ?? 0.005) as number,
                } as Partial<MonogsConfig>)
              }
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
          <div className="section-title">fpv preprocessing</div>
        </div>
        {!readOnly && (
          <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
            {Object.entries(PREPROC_PRESETS).map(([name, patch]) => (
              <button
                key={name}
                type="button"
                onClick={() =>
                  onChange(patch as unknown as Partial<SlamConfig>)
                }
                style={{ flex: 1, minWidth: 0 }}
              >
                {name}
              </button>
            ))}
          </div>
        )}
        <BoolRow
          label="denoise + deflicker"
          tipKey="preproc_denoise"
          value={config.preproc_denoise}
          readOnly={readOnly}
          onChange={(v) =>
            onChange({ preproc_denoise: v } as Partial<SlamConfig>)
          }
        />
        <BoolRow
          label="analog cleanup"
          tipKey="preproc_analog_cleanup"
          value={config.preproc_analog_cleanup}
          readOnly={readOnly}
          onChange={(v) =>
            onChange({
              preproc_analog_cleanup: v,
            } as Partial<SlamConfig>)
          }
        />
        <BoolRow
          label="standalone deflicker"
          tipKey="preproc_deflicker"
          value={config.preproc_deflicker}
          readOnly={readOnly}
          onChange={(v) =>
            onChange({ preproc_deflicker: v } as Partial<SlamConfig>)
          }
        />
        <BoolRow
          label="fisheye unwrap"
          tipKey="preproc_fisheye"
          value={config.preproc_fisheye}
          readOnly={readOnly}
          onChange={(v) =>
            onChange({ preproc_fisheye: v } as Partial<SlamConfig>)
          }
        />
        {config.preproc_fisheye && !compact && (
          <>
            <NumberRow
              label="fisheye in fov"
              tipKey="preproc_fisheye"
              value={config.fisheye_in_fov}
              step={5}
              min={60}
              max={180}
              readOnly={readOnly}
              onChange={(v) =>
                onChange({
                  fisheye_in_fov: (v ?? 165) as number,
                } as Partial<SlamConfig>)
              }
            />
            <NumberRow
              label="fisheye out fov"
              tipKey="preproc_fisheye"
              value={config.fisheye_out_fov}
              step={5}
              min={40}
              max={140}
              readOnly={readOnly}
              onChange={(v) =>
                onChange({
                  fisheye_out_fov: (v ?? 90) as number,
                } as Partial<SlamConfig>)
              }
            />
          </>
        )}
        <BoolRow
          label="mask osd text"
          tipKey="preproc_osd_mask"
          value={config.preproc_osd_mask}
          readOnly={readOnly}
          onChange={(v) =>
            onChange({ preproc_osd_mask: v } as Partial<SlamConfig>)
          }
        />
        <BoolRow
          label="colour normalisation"
          tipKey="preproc_color_norm"
          value={config.preproc_color_norm}
          readOnly={readOnly}
          onChange={(v) =>
            onChange({ preproc_color_norm: v } as Partial<SlamConfig>)
          }
        />
        <BoolRow
          label="rolling-shutter correction"
          tipKey="preproc_rs_correction"
          value={config.preproc_rs_correction}
          readOnly={readOnly}
          onChange={(v) =>
            onChange({
              preproc_rs_correction: v,
            } as Partial<SlamConfig>)
          }
        />
        <label className="stat">
          <Tip text={TIPS.preproc_deblur ?? ""}>
            <span>deblur</span>
          </Tip>
          <select
            value={config.preproc_deblur}
            disabled={readOnly}
            onChange={(e) =>
              onChange({
                preproc_deblur:
                  e.target.value as PreprocFields["preproc_deblur"],
              } as Partial<SlamConfig>)
            }
          >
            <option value="none">none</option>
            <option value="unsharp">unsharp</option>
            <option value="nafnet">nafnet</option>
          </select>
        </label>
        <BoolRow
          label="keyframe scoring"
          tipKey="preproc_keyframe_score"
          value={config.preproc_keyframe_score}
          readOnly={readOnly}
          onChange={(v) =>
            onChange({
              preproc_keyframe_score: v,
            } as Partial<SlamConfig>)
          }
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
        <NumberRow
          label="vram soft limit (gb)"
          tipKey="vram_soft_limit_gb"
          value={config.vram_soft_limit_gb}
          step={1}
          min={1}
          max={80}
          placeholder="default"
          readOnly={readOnly}
          onChange={(v) =>
            onChange({ vram_soft_limit_gb: v } as Partial<SlamConfig>)
          }
        />
      </div>
    </div>
  );
}
