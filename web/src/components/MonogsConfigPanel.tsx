"use client";

import { Tip } from "@/components/Tip";
import { BoolRow, NumberRow } from "@/components/tracking-rows";
import {
  type MonogsConfig,
  PREPROC_PRESETS,
  type PreprocFields,
} from "@/lib/types";

interface Props {
  config: MonogsConfig;
  onChange: (patch: Partial<MonogsConfig>) => void;
  readOnly?: boolean;
  title?: string;
}

const TIPS: Record<string, string> = {
  max_frames:
    "Hard cap on the number of input frames fed to the MonoGS tracker. Blank = run the entire clip.",
  downscale:
    "Integer scale factor applied to each input frame before tracking. 1 = full resolution, 2 = half (4x less work).",
  stride: "Frame skip. 1 = every frame, 2 = every other.",
  fps: "Resample source video to this fps before tracking.",
  calibration:
    "Camera intrinsics. `auto` estimates fx/fy from a 60° horizontal FOV assumption; `manual` lets you provide fx/fy/cx/cy.",
  keyframe_policy:
    "How new keyframes are chosen. `score_gated` uses the FPV keyframe scoring stage as a gate — recommended for analog footage.",
  keyframe_interval: "Minimum frame gap between keyframes added to the splat.",
  score_gate_quantile:
    "score_gated only: keep frames whose combined sharpness+motion score is above this quantile. 0.5 drops the blurriest/most-static half.",
  partial_snapshot_every:
    "Every N keyframes, emit a partial_splat_XXXX.ply so the splat viewer can render the scene growing. Lower = smoother preview but more disk writes.",
  refine_iters:
    "Refinement iterations per keyframe. Higher = more aggressive fit to each new view, more compute per frame.",
  prune_opacity:
    "Drop gaussians whose opacity falls below this threshold during training. Keeps the final splat lean.",
  vram_soft_limit_gb:
    "Abort the job cleanly if allocated VRAM crosses this threshold. Leave blank to use the worker default.",
};

export function MonogsConfigPanel({
  config,
  onChange,
  readOnly,
  title,
}: Props) {
  return (
    <div className="panel">
      <div className="panel-header">
        <span>{title ?? "config · monogs"}</span>
        {readOnly && <span className="meta">locked</span>}
      </div>
      <div className="panel-body" style={{ display: "grid", gap: 6 }}>
        <div className="mono-small" style={{ opacity: 0.8 }}>
          MonoGS trains a Gaussian splat end-to-end from uploaded footage —
          no separate SLAM pass required. The output is a standard
          `splat.ply` that loads in the splat viewer and tools.
        </div>

        <NumberRow
          label="max frames"
          tip={TIPS.max_frames}
          value={config.max_frames}
          step={100}
          min={1}
          placeholder="all"
          readOnly={readOnly}
          onChange={(v) => onChange({ max_frames: v })}
        />
        <NumberRow
          label="downscale"
          tip={TIPS.downscale}
          value={config.downscale}
          step={1}
          min={1}
          max={8}
          readOnly={readOnly}
          onChange={(v) => onChange({ downscale: (v ?? 1) as number })}
        />
        <NumberRow
          label="stride"
          tip={TIPS.stride}
          value={config.stride}
          step={1}
          min={1}
          max={10}
          readOnly={readOnly}
          onChange={(v) => onChange({ stride: (v ?? 1) as number })}
        />
        <NumberRow
          label="fps"
          tip={TIPS.fps}
          value={config.fps}
          step={0.5}
          min={0.5}
          max={60}
          readOnly={readOnly}
          onChange={(v) => onChange({ fps: (v ?? 10) as number })}
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
                calibration: e.target.value as MonogsConfig["calibration"],
              })
            }
          >
            <option value="auto">auto</option>
            <option value="manual">manual</option>
          </select>
        </label>
        {config.calibration === "manual" && (
          <>
            <NumberRow
              label="fx"
              tip="Focal length x (px)."
              value={config.fx}
              step={10}
              min={1}
              readOnly={readOnly}
              onChange={(v) => onChange({ fx: v })}
            />
            <NumberRow
              label="fy"
              tip="Focal length y (px)."
              value={config.fy}
              step={10}
              min={1}
              readOnly={readOnly}
              onChange={(v) => onChange({ fy: v })}
            />
            <NumberRow
              label="cx"
              tip="Principal point x (px)."
              value={config.cx}
              step={10}
              min={0}
              readOnly={readOnly}
              onChange={(v) => onChange({ cx: v })}
            />
            <NumberRow
              label="cy"
              tip="Principal point y (px)."
              value={config.cy}
              step={10}
              min={0}
              readOnly={readOnly}
              onChange={(v) => onChange({ cy: v })}
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
                  e.target.value as MonogsConfig["keyframe_policy"],
              })
            }
          >
            <option value="score_gated">score_gated</option>
            <option value="translation">translation</option>
            <option value="hybrid">hybrid</option>
          </select>
        </label>
        <NumberRow
          label="keyframe interval"
          tip={TIPS.keyframe_interval}
          value={config.keyframe_interval}
          step={1}
          min={1}
          max={30}
          readOnly={readOnly}
          onChange={(v) => onChange({ keyframe_interval: (v ?? 6) as number })}
        />
        {config.keyframe_policy !== "translation" && (
          <NumberRow
            label="score gate quantile"
            tip={TIPS.score_gate_quantile}
            value={config.score_gate_quantile}
            step={0.05}
            min={0}
            max={1}
            readOnly={readOnly}
            onChange={(v) =>
              onChange({ score_gate_quantile: (v ?? 0.5) as number })
            }
          />
        )}
        <NumberRow
          label="partial snapshot every"
          tip={TIPS.partial_snapshot_every}
          value={config.partial_snapshot_every}
          step={1}
          min={1}
          max={50}
          readOnly={readOnly}
          onChange={(v) =>
            onChange({ partial_snapshot_every: (v ?? 5) as number })
          }
        />

        <div
          style={{
            marginTop: 6,
            paddingTop: 6,
            borderTop: "1px solid var(--rule)",
          }}
        >
          <div className="section-title">splat tuning</div>
        </div>
        <NumberRow
          label="refine iters/kf"
          tip={TIPS.refine_iters}
          value={config.refine_iters}
          step={5}
          min={1}
          max={500}
          readOnly={readOnly}
          onChange={(v) => onChange({ refine_iters: (v ?? 50) as number })}
        />
        <NumberRow
          label="prune opacity"
          tip={TIPS.prune_opacity}
          value={config.prune_opacity}
          step={0.001}
          min={0}
          max={0.5}
          readOnly={readOnly}
          onChange={(v) =>
            onChange({ prune_opacity: (v ?? 0.005) as number })
          }
        />

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
                  onChange(patch as unknown as Partial<MonogsConfig>)
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
          value={config.preproc_denoise}
          readOnly={readOnly}
          onChange={(v) => onChange({ preproc_denoise: v })}
        />
        <BoolRow
          label="analog cleanup"
          value={config.preproc_analog_cleanup}
          readOnly={readOnly}
          onChange={(v) => onChange({ preproc_analog_cleanup: v })}
        />
        <BoolRow
          label="standalone deflicker"
          value={config.preproc_deflicker}
          readOnly={readOnly}
          onChange={(v) => onChange({ preproc_deflicker: v })}
        />
        <BoolRow
          label="fisheye unwrap"
          value={config.preproc_fisheye}
          readOnly={readOnly}
          onChange={(v) => onChange({ preproc_fisheye: v })}
        />
        <BoolRow
          label="mask osd text"
          value={config.preproc_osd_mask}
          readOnly={readOnly}
          onChange={(v) => onChange({ preproc_osd_mask: v })}
        />
        <BoolRow
          label="colour normalisation"
          value={config.preproc_color_norm}
          readOnly={readOnly}
          onChange={(v) => onChange({ preproc_color_norm: v })}
        />
        <BoolRow
          label="rolling-shutter correction"
          value={config.preproc_rs_correction}
          readOnly={readOnly}
          onChange={(v) => onChange({ preproc_rs_correction: v })}
        />
        <label className="stat">
          <Tip text="Optional motion-deblur pass.">
            <span>deblur</span>
          </Tip>
          <select
            value={config.preproc_deblur}
            disabled={readOnly}
            onChange={(e) =>
              onChange({
                preproc_deblur:
                  e.target.value as PreprocFields["preproc_deblur"],
              })
            }
          >
            <option value="none">none</option>
            <option value="unsharp">unsharp</option>
            <option value="nafnet">nafnet</option>
          </select>
        </label>
        <BoolRow
          label="keyframe scoring"
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
        <NumberRow
          label="vram soft limit (gb)"
          tip={TIPS.vram_soft_limit_gb}
          value={config.vram_soft_limit_gb}
          step={1}
          min={1}
          max={80}
          placeholder="default"
          readOnly={readOnly}
          onChange={(v) => onChange({ vram_soft_limit_gb: v })}
        />
      </div>
    </div>
  );
}
