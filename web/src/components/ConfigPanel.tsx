"use client";

import { type JobConfig, PRESETS } from "@/lib/types";

interface Props {
  config: JobConfig;
  onChange: (patch: Partial<JobConfig>) => void;
  readOnly?: boolean;
  compact?: boolean;
}

function NumberRow({
  label,
  value,
  onChange,
  step = 1,
  min,
  max,
  readOnly,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  step?: number;
  min?: number;
  max?: number;
  readOnly?: boolean;
}) {
  return (
    <label className="stat">
      <span>{label}</span>
      <input
        type="number"
        value={value}
        step={step}
        min={min}
        max={max}
        readOnly={readOnly}
        disabled={readOnly}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ width: 80, textAlign: "right" }}
      />
    </label>
  );
}

function BoolRow({
  label,
  value,
  onChange,
  readOnly,
}: {
  label: string;
  value: boolean;
  onChange: (v: boolean) => void;
  readOnly?: boolean;
}) {
  return (
    <label className="stat">
      <span>{label}</span>
      <input
        type="checkbox"
        checked={value}
        disabled={readOnly}
        onChange={(e) => onChange(e.target.checked)}
      />
    </label>
  );
}

export function ConfigPanel({ config, onChange, readOnly, compact }: Props) {
  return (
    <div className="panel">
      <div className="panel-header">config {readOnly ? "· locked" : ""}</div>
      <div className="panel-body" style={{ display: "grid", gap: 6 }}>
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
          <span>model</span>
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
          <span>mode</span>
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
          value={config.fps}
          step={0.5}
          min={0.5}
          max={60}
          readOnly={readOnly}
          onChange={(v) => onChange({ fps: v })}
        />
        <NumberRow
          label="conf %ile"
          value={config.conf_percentile}
          step={5}
          min={0}
          max={95}
          readOnly={readOnly}
          onChange={(v) => onChange({ conf_percentile: v })}
        />
        <NumberRow
          label="keyframe interval"
          value={config.keyframe_interval}
          step={1}
          min={1}
          max={30}
          readOnly={readOnly}
          onChange={(v) => onChange({ keyframe_interval: v })}
        />
        <NumberRow
          label="num scale frames"
          value={config.num_scale_frames}
          step={1}
          min={1}
          max={16}
          readOnly={readOnly}
          onChange={(v) => onChange({ num_scale_frames: v })}
        />
        <NumberRow
          label="camera iters"
          value={config.camera_num_iterations}
          step={1}
          min={1}
          max={8}
          readOnly={readOnly}
          onChange={(v) => onChange({ camera_num_iterations: v })}
        />
        <BoolRow
          label="mask sky"
          value={config.mask_sky}
          readOnly={readOnly}
          onChange={(v) => onChange({ mask_sky: v })}
        />
        <BoolRow
          label="use sdpa"
          value={config.use_sdpa}
          readOnly={readOnly}
          onChange={(v) => onChange({ use_sdpa: v })}
        />
        <BoolRow
          label="offload cpu"
          value={config.offload_to_cpu}
          readOnly={readOnly}
          onChange={(v) => onChange({ offload_to_cpu: v })}
        />
        <BoolRow
          label="show cameras"
          value={config.show_cam}
          readOnly={readOnly}
          onChange={(v) => onChange({ show_cam: v })}
        />

        {!compact && config.mode === "windowed" && (
          <>
            <NumberRow
              label="window size"
              value={config.window_size}
              step={8}
              min={16}
              max={512}
              readOnly={readOnly}
              onChange={(v) => onChange({ window_size: v })}
            />
            <NumberRow
              label="overlap"
              value={config.overlap_size}
              step={4}
              min={0}
              max={128}
              readOnly={readOnly}
              onChange={(v) => onChange({ overlap_size: v })}
            />
          </>
        )}
      </div>
    </div>
  );
}
