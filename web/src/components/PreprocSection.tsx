"use client";

import { useState } from "react";

import { Tip } from "@/components/Tip";
import {
  deriveFootageProfile,
  FOOTAGE_PROFILE_ORDER,
  FOOTAGE_PROFILES,
  FOOTAGE_PROFILE_PATCHES,
  type FootageProfile,
} from "@/lib/footageProfiles";
import type { PreprocFields } from "@/lib/types";

interface Props {
  /** Current config slice — typically the same object the parent panel
   *  hands its other rows. Only the `preproc_*` fields are read. */
  config: PreprocFields;
  /** Patch sink — same shape the parent panel uses for other rows. The
   *  profile dropdown applies the bundled patch through this; the
   *  Advanced toggles below send single-field patches. */
  onChange: (patch: Partial<PreprocFields>) => void;
  readOnly?: boolean;
  /** When true, hide the rs_shear / deblur_sharpness_gate detail rows
   *  that only make sense in the wider Lingbot panel. SLAM panels run
   *  in compact mode. */
  compact?: boolean;
}

/**
 * Footage-preprocessing section, shared across ConfigPanel /
 * SlamConfigPanel / MonogsConfigPanel. Replaces the old per-panel
 * "fpv preprocessing" block.
 *
 * Primary affordance is the **footage profile** dropdown: hi-def by
 * default, with analog-FPV bundles as one click. Individual toggles
 * live below an "advanced" disclosure that auto-opens when the
 * profile is `custom`.
 */
export function PreprocSection({ config, onChange, readOnly, compact }: Props) {
  const profile = deriveFootageProfile(config);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const showAdvanced = advancedOpen || profile === "custom";

  function applyProfile(next: FootageProfile) {
    if (next === "custom" || next === profile) return;
    onChange(FOOTAGE_PROFILE_PATCHES[next] as Partial<PreprocFields>);
  }

  return (
    <div
      style={{
        marginTop: 6,
        paddingTop: 6,
        borderTop: "1px solid var(--rule)",
        display: "grid",
        gap: 6,
      }}
    >
      <div className="section-title">preprocessing</div>

      <label className="stat">
        <Tip text={TIPS.footage_profile}>
          <span>footage profile</span>
        </Tip>
        <select
          value={profile}
          disabled={readOnly}
          onChange={(e) => applyProfile(e.target.value as FootageProfile)}
        >
          {FOOTAGE_PROFILE_ORDER.map((id) => (
            <option key={id} value={id}>
              {FOOTAGE_PROFILES[id].label}
            </option>
          ))}
          {profile === "custom" && (
            <option value="custom">{FOOTAGE_PROFILES.custom.label}</option>
          )}
        </select>
      </label>

      <div
        className="mono-small"
        style={{ color: "var(--muted)", lineHeight: 1.4 }}
      >
        {FOOTAGE_PROFILES[profile].desc}
      </div>

      {!readOnly && (
        <button
          type="button"
          onClick={() => setAdvancedOpen((v) => !v)}
          aria-expanded={showAdvanced}
          style={{ width: "100%", textAlign: "left" }}
        >
          {showAdvanced ? "▾ advanced (hide)" : "▸ advanced (show toggles)"}
        </button>
      )}

      {showAdvanced && (
        <div style={{ display: "grid", gap: 4 }}>
          <BoolRow
            label="fisheye unwrap"
            tipText={TIPS.preproc_fisheye}
            value={config.preproc_fisheye}
            readOnly={readOnly}
            onChange={(v) => onChange({ preproc_fisheye: v })}
          />
          {config.preproc_fisheye && !compact && (
            <>
              <NumberRow
                label="fisheye · in fov°"
                tipText={TIPS.fisheye_in_fov}
                value={config.fisheye_in_fov}
                step={1}
                min={60}
                max={220}
                readOnly={readOnly}
                onChange={(v) => onChange({ fisheye_in_fov: v })}
              />
              <NumberRow
                label="fisheye · out fov°"
                tipText={TIPS.fisheye_out_fov}
                value={config.fisheye_out_fov}
                step={1}
                min={50}
                max={180}
                readOnly={readOnly}
                onChange={(v) => onChange({ fisheye_out_fov: v })}
              />
            </>
          )}

          <BoolRow
            label="temporal denoise + deflicker"
            tipText={TIPS.preproc_denoise}
            value={config.preproc_denoise}
            readOnly={readOnly}
            onChange={(v) => onChange({ preproc_denoise: v })}
          />
          <BoolRow
            label="atadenoise (heavier)"
            tipText={TIPS.preproc_analog_cleanup}
            value={config.preproc_analog_cleanup}
            readOnly={readOnly}
            onChange={(v) => onChange({ preproc_analog_cleanup: v })}
          />
          <BoolRow
            label="standalone deflicker"
            tipText={TIPS.preproc_deflicker}
            value={config.preproc_deflicker}
            readOnly={readOnly}
            onChange={(v) => onChange({ preproc_deflicker: v })}
          />

          <BoolRow
            label="static-overlay mask + inpaint"
            tipText={TIPS.preproc_osd_mask}
            value={config.preproc_osd_mask}
            readOnly={readOnly}
            onChange={(v) => onChange({ preproc_osd_mask: v })}
          />
          {!compact && config.preproc_osd_mask && (
            <>
              <NumberRow
                label="osd · samples"
                tipText={TIPS.osd_mask_samples}
                value={config.osd_mask_samples}
                step={10}
                min={10}
                max={400}
                readOnly={readOnly}
                onChange={(v) => onChange({ osd_mask_samples: v })}
              />
              <NumberRow
                label="osd · stddev thr"
                tipText={TIPS.osd_mask_std_threshold}
                value={config.osd_mask_std_threshold}
                step={0.5}
                min={0.5}
                max={30}
                readOnly={readOnly}
                onChange={(v) => onChange({ osd_mask_std_threshold: v })}
              />
              <NumberRow
                label="osd · dilate"
                tipText={TIPS.osd_mask_dilate}
                value={config.osd_mask_dilate}
                step={1}
                min={0}
                max={10}
                readOnly={readOnly}
                onChange={(v) => onChange({ osd_mask_dilate: v })}
              />
              <BoolRow
                label="osd · detect text"
                tipText={TIPS.osd_detect_text}
                value={config.osd_detect_text}
                readOnly={readOnly}
                onChange={(v) => onChange({ osd_detect_text: v })}
              />
              {config.osd_detect_text && (
                <NumberRow
                  label="osd · edge persist"
                  tipText={TIPS.osd_edge_persist_frac}
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

          <BoolRow
            label="white-balance + histogram stretch"
            tipText={TIPS.preproc_color_norm}
            value={config.preproc_color_norm}
            readOnly={readOnly}
            onChange={(v) => onChange({ preproc_color_norm: v })}
          />

          <BoolRow
            label="rolling-shutter correction"
            tipText={TIPS.preproc_rs_correction}
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
              <span>motion deblur</span>
            </Tip>
            <select
              value={config.preproc_deblur}
              disabled={readOnly}
              onChange={(e) =>
                onChange({
                  preproc_deblur: e.target
                    .value as PreprocFields["preproc_deblur"],
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
              tipText={TIPS.deblur_sharpness_gate}
              value={config.deblur_sharpness_gate}
              step={0.05}
              min={0.1}
              max={1.5}
              readOnly={readOnly}
              onChange={(v) => onChange({ deblur_sharpness_gate: v })}
            />
          )}

          <BoolRow
            label="keyframe scoring (frame_scores.jsonl)"
            tipText={TIPS.preproc_keyframe_score}
            value={config.preproc_keyframe_score}
            readOnly={readOnly}
            onChange={(v) => onChange({ preproc_keyframe_score: v })}
          />
        </div>
      )}
    </div>
  );
}

/**
 * Source-neutral tip copy. Each entry leads with what the stage *does*;
 * FPV / analog usage is mentioned only where the stage is genuinely
 * footage-type-specific (fisheye unwrap, OSD-overlay masking).
 */
const TIPS: Record<string, string> = {
  footage_profile:
    "Bundles the toggles below into one pick. `hi-def · no preproc` is the default — leaves footage from phones, mirrorless, action cams, and HD drones untouched. The `fpv` profiles enable the cleanup stack tuned for low-bitrate analog captures (DVR / analog receiver footage). Toggle individual stages in `advanced` to override; the picker flips to `custom` when your toggles don't match a named profile.",
  preproc_fisheye:
    "Unwrap fisheye-lens footage to rectilinear before reconstruction. Anything wider than ~120° (action cams in superview, FPV micro cams at 150-170°) gives the model false geometry if left distorted. The unwrap crops the rim where distortion was worst.",
  fisheye_in_fov:
    "Source lens horizontal FOV in degrees. Look up your camera spec: GoPro Wide ≈ 122°, Insta360 X-series ≈ 170°, Caddx Ratel 2.1 ≈ 165°. Wrong value = skewed geometry.",
  fisheye_out_fov:
    "Target diagonal FOV after unwrap. 90° keeps the sharper centre, 110-120° keeps more peripheral content at the cost of residual edge distortion.",
  preproc_denoise:
    "Temporal denoise (hqdn3d) + deflicker. Reduces noise on any low-light, high-ISO, or compressed source — phone clips at night, drone footage at dusk, low-bitrate streams. Adds a few seconds to ingest; safe to leave on for any noisy input.",
  preproc_osd_mask:
    "Detect pixels that don't change over time (telemetry text, timer, station logo, watermark, GoPro battery indicator) and inpaint them out before reconstruction. Without this they become false geometry fixed in camera space. Worth enabling for any footage with a persistent overlay.",
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
  preproc_analog_cleanup:
    "Heavier temporal denoise (atadenoise) tuned for chroma noise and dot crawl. Expensive — only enable for genuinely noisy captures (analog DVR rips, very low-bitrate streams) where the standard denoise still leaves artifacts. No-op on clean digital clips.",
  preproc_deflicker:
    "Standalone ffmpeg deflicker without the hqdn3d denoise pair. Useful when brightness jitter is the dominant problem and you don't want the spatial blurring that hqdn3d adds. Auto-skipped if `denoise + deflicker` is already on.",
  preproc_color_norm:
    "Per-frame grey-world white-balance + 1/99-percentile histogram stretch. Recovers natural colour on any source with a colour cast — tungsten indoor footage, green-tinted analog feeds, magenta-cast drone clips. Cheap, CPU-only.",
  preproc_rs_correction:
    "Global y-shear correction for rolling-shutter skew. Most CMOS cameras (phones, action cams, FPV digital, drones with electronic shutters) read the sensor line-by-line; fast pans show characteristic skew. Estimates a single px-per-row shear from optical flow and applies the inverse warp to every frame. Full per-row RS (needs gyro data) is out of scope.",
  rs_shear_px_per_row:
    "Override the estimated shear in pixels per row. Leave blank to let the estimator pick. Negative values tilt the other way. Values smaller than ±0.02 px/row skip the warp entirely.",
  preproc_deblur:
    "Motion-deblur strategy:\n• none — off (default, cheapest).\n• unsharp — classical unsharp-mask gated by per-frame Laplacian variance. Fast, CPU.\n• nafnet — learned single-image deblur (Phase 3 wires the hook; the checkpoint ships in a follow-up, falls back to unsharp for now).",
  deblur_sharpness_gate:
    "Apply the deblur filter only to frames whose Laplacian variance is below this fraction of the clip median. 1.0 = every frame, 0.6 (default) = the blurriest ~60%, 0.3 = only the worst offenders.",
  preproc_keyframe_score:
    "Write `frame_scores.jsonl` with per-frame sharpness and optical-flow magnitude. SLAM backends with `keyframe_policy=score_gated` read this to drop low-quality frames before keyframe selection. Cheap to leave on; downstream backends that don't consume it just ignore the file.",
};

function BoolRow({
  label,
  tipText,
  value,
  onChange,
  readOnly,
}: {
  label: string;
  tipText: string;
  value: boolean;
  onChange: (v: boolean) => void;
  readOnly?: boolean;
}) {
  return (
    <label className="stat">
      <Tip text={tipText}>
        <span>{label}</span>
      </Tip>
      <input
        type="checkbox"
        checked={value}
        readOnly={readOnly}
        disabled={readOnly}
        onChange={(e) => onChange(e.target.checked)}
      />
    </label>
  );
}

function NumberRow({
  label,
  tipText,
  value,
  onChange,
  step = 1,
  min,
  max,
  readOnly,
}: {
  label: string;
  tipText: string;
  value: number;
  onChange: (v: number) => void;
  step?: number;
  min?: number;
  max?: number;
  readOnly?: boolean;
}) {
  return (
    <label className="stat">
      <Tip text={tipText}>
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
