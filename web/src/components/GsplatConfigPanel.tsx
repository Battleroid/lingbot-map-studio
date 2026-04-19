"use client";

import { Tip } from "@/components/Tip";
import type { GsplatConfig } from "@/lib/types";

interface Props {
  config: GsplatConfig;
  onChange: (patch: Partial<GsplatConfig>) => void;
  readOnly?: boolean;
  title?: string;
}

const TIPS: Record<string, string> = {
  iterations:
    "Total training iterations. 30k is the standard 3DGS default — noticeable quality gains slow after ~20k. Drop to 5-10k for quick previews.",
  sh_degree:
    "Max spherical-harmonics degree for view-dependent color. 0 = per-gaussian RGB only (cheapest, roughest look). 3 = full 3DGS default (45 coeffs/gaussian, best specular reproduction).",
  densify_interval:
    "Gaussians are split/cloned every N iterations. 500 is a safe default; lower densifies sooner (more gaussians, more vram).",
  prune_interval:
    "How often low-opacity gaussians are culled.",
  prune_opacity:
    "Opacity floor — gaussians below this get pruned. Raise to keep the splat lean; drop to keep wispy low-opacity detail.",
  init_from:
    "• point_cloud — initialise gaussians from the source job's reconstruction.ply (recommended — faster convergence).\n• random — scatter random gaussians in a unit sphere.",
  random_init_count:
    "Only used with init_from=random. Number of gaussians to start with.",
  initial_resolution:
    "Training resolution as a fraction of the source images. 0.5 = quarter-pixel-count. Upsampled later by upsample_at_iter.",
  upsample_at_iter:
    "Iteration at which training switches to full source resolution. Coarse-to-fine schedule; earlier = slower but smoother early previews.",
  preview_every_iters:
    "Write a partial splat checkpoint (.ply) this often for the live viewer. Lower = smoother live preview; more disk writes.",
  preview_max_gaussians:
    "Cap on gaussian count exposed in live previews. The final splat.ply is unaffected.",
  bake_mesh_after:
    "After training, run a mesh-from-splat bake (2DGS / SuGaR-lite) to produce a baked.glb. Slow; off by default — you can bake on-demand from the splat tools panel instead.",
  bake_mesh_depth:
    "Octree depth for the bake-to-mesh step. 9-10 typical.",
  vram_soft_limit_gb:
    "Abort cleanly if allocated VRAM crosses this threshold. Leave blank to use the worker default.",
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

export function GsplatConfigPanel({
  config,
  onChange,
  readOnly,
  title,
}: Props) {
  return (
    <div className="panel">
      <div className="panel-header">
        <span>{title ?? "config · gaussian splat"}</span>
        {readOnly && <span className="meta">locked</span>}
      </div>
      <div className="panel-body" style={{ display: "grid", gap: 6 }}>
        <div className="mono-small" style={{ opacity: 0.7 }}>
          source · {config.source_job_id || "—"}
        </div>

        <NumberRow
          label="iterations"
          tipKey="iterations"
          value={config.iterations}
          step={1000}
          min={100}
          max={200_000}
          readOnly={readOnly}
          onChange={(v) => onChange({ iterations: (v ?? 30_000) as number })}
        />
        <NumberRow
          label="sh degree"
          tipKey="sh_degree"
          value={config.sh_degree}
          step={1}
          min={0}
          max={3}
          readOnly={readOnly}
          onChange={(v) => onChange({ sh_degree: (v ?? 3) as number })}
        />
        <NumberRow
          label="densify every"
          tipKey="densify_interval"
          value={config.densify_interval}
          step={50}
          min={50}
          max={5000}
          readOnly={readOnly}
          onChange={(v) =>
            onChange({ densify_interval: (v ?? 500) as number })
          }
        />
        <NumberRow
          label="prune every"
          tipKey="prune_interval"
          value={config.prune_interval}
          step={50}
          min={50}
          max={5000}
          readOnly={readOnly}
          onChange={(v) =>
            onChange({ prune_interval: (v ?? 200) as number })
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
            onChange({ prune_opacity: (v ?? 0.005) as number })
          }
        />

        <label className="stat">
          <Tip text={TIPS.init_from}>
            <span>init from</span>
          </Tip>
          <select
            value={config.init_from}
            disabled={readOnly}
            onChange={(e) =>
              onChange({
                init_from: e.target.value as GsplatConfig["init_from"],
              })
            }
          >
            <option value="point_cloud">point_cloud</option>
            <option value="random">random</option>
          </select>
        </label>
        {config.init_from === "random" && (
          <NumberRow
            label="random init count"
            tipKey="random_init_count"
            value={config.random_init_count}
            step={10_000}
            min={1000}
            max={2_000_000}
            readOnly={readOnly}
            onChange={(v) =>
              onChange({ random_init_count: (v ?? 100_000) as number })
            }
          />
        )}

        <NumberRow
          label="initial resolution"
          tipKey="initial_resolution"
          value={config.initial_resolution}
          step={0.05}
          min={0.1}
          max={1}
          readOnly={readOnly}
          onChange={(v) =>
            onChange({ initial_resolution: (v ?? 0.5) as number })
          }
        />
        <NumberRow
          label="upsample at iter"
          tipKey="upsample_at_iter"
          value={config.upsample_at_iter}
          step={500}
          min={0}
          max={200_000}
          readOnly={readOnly}
          onChange={(v) =>
            onChange({ upsample_at_iter: (v ?? 5000) as number })
          }
        />
        <NumberRow
          label="preview every iters"
          tipKey="preview_every_iters"
          value={config.preview_every_iters}
          step={100}
          min={50}
          max={10_000}
          readOnly={readOnly}
          onChange={(v) =>
            onChange({ preview_every_iters: (v ?? 1000) as number })
          }
        />
        <NumberRow
          label="preview max gaussians"
          tipKey="preview_max_gaussians"
          value={config.preview_max_gaussians}
          step={50_000}
          min={10_000}
          max={5_000_000}
          readOnly={readOnly}
          onChange={(v) =>
            onChange({ preview_max_gaussians: (v ?? 500_000) as number })
          }
        />

        <label className="stat">
          <Tip text={TIPS.bake_mesh_after}>
            <span>bake mesh after</span>
          </Tip>
          <input
            type="checkbox"
            checked={config.bake_mesh_after}
            disabled={readOnly}
            onChange={(e) => onChange({ bake_mesh_after: e.target.checked })}
          />
        </label>
        {config.bake_mesh_after && (
          <NumberRow
            label="bake mesh depth"
            tipKey="bake_mesh_depth"
            value={config.bake_mesh_depth}
            step={1}
            min={5}
            max={12}
            readOnly={readOnly}
            onChange={(v) =>
              onChange({ bake_mesh_depth: (v ?? 10) as number })
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
          onChange={(v) => onChange({ vram_soft_limit_gb: v })}
        />
      </div>
    </div>
  );
}
