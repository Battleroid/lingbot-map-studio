"use client";

import type {
  GsplatBackend,
  JobSummary,
  ProcessorId,
  SlamBackend,
} from "@/lib/types";

export type StudioMode = "lingbot" | "slam" | "gsplat";

// Per the v2 design — three modes, each keyed to a phosphor accent so the
// tile's category bar + soft hover wash + active inset stripe all read in
// the same color. No 3D canvas previews: just title + description.
//   lingbot → amber  (warn-leaning, the existing/legacy mode)
//   slam    → green  (the recommended default for most footage)
//   gsplat  → cyan   (info / forward-looking)
type ModeMeta = {
  acc: "amber" | "green" | "cyan";
  label: string;
  desc: string;
};
const MODES: Record<StudioMode, ModeMeta> = {
  lingbot: {
    acc: "amber",
    label: "lingbot",
    desc: "feed-forward reconstruction. point cloud + textured mesh + camera path.",
  },
  slam: {
    acc: "green",
    label: "slam",
    desc: "mast3r, droid, dpvo. frame-by-frame tracking + map.",
  },
  gsplat: {
    acc: "cyan",
    label: "gaussian splat",
    desc: "gaussian splat training. chains off a slam/lingbot job, or runs monogs directly.",
  },
};

type SlamMeta = { acc: "violet" | "magenta" | "cyan"; hint: string };
const SLAM_BACKENDS: Record<SlamBackend, SlamMeta> = {
  mast3r_slam: {
    acc: "violet",
    hint: "calibration-free · robust to unknown intrinsics · the safe default",
  },
  droid_slam: { acc: "magenta", hint: "dense · VRAM-heavy · high fidelity" },
  dpvo: { acc: "cyan", hint: "patch-based · lightweight · long clips" },
};
const SLAM_BACKEND_LABELS: Record<SlamBackend, string> = {
  mast3r_slam: "MASt3R-SLAM",
  droid_slam: "DROID-SLAM",
  dpvo: "DPVO",
};

type GsplatMeta = { acc: "cyan" | "amber"; label: string; hint: string };
const GSPLAT_BACKENDS: Record<GsplatBackend, GsplatMeta> = {
  gsplat: {
    acc: "cyan",
    label: "gsplat · chain from source",
    hint: "train a 3DGS scene off a completed SLAM or Lingbot job (reuses its frames + poses + cloud)",
  },
  monogs: {
    acc: "amber",
    label: "MonoGS · direct from footage",
    hint: "end-to-end splat SLAM — upload footage and get a splat directly, no intermediate SLAM job required",
  },
};

interface Props {
  mode: StudioMode;
  onMode: (m: StudioMode) => void;
  slamBackend: SlamBackend;
  onSlamBackend: (b: SlamBackend) => void;
  gsplatBackend: GsplatBackend;
  onGsplatBackend: (b: GsplatBackend) => void;
  gsplatSourceId: string | null;
  onGsplatSource: (id: string | null) => void;
  sourceJobs: JobSummary[];
  disabled?: boolean;
}

/** Ready jobs whose output can seed a gsplat training run. MonoGS jobs
 * already produce a splat directly, so chaining another gsplat training
 * pass off them is redundant — excluded here. */
export function selectableSplatSources(jobs: JobSummary[]): JobSummary[] {
  const allowed: ProcessorId[] = [
    "lingbot",
    "droid_slam",
    "mast3r_slam",
    "dpvo",
  ];
  return jobs.filter(
    (j) => j.status === "ready" && allowed.includes(j.processor),
  );
}

/** Tile contents — uppercase tracked label with an arrow that animates in
 * when active, plus a lowercase plain-language description. The arrow span
 * is driven from CSS (`.tile.is-active .tile-label .arrow { width: 14px }`)
 * so the text doesn't jump on selection. */
function TileContents({ label, desc }: { label: string; desc: string }) {
  return (
    <>
      <span className="tile-label">
        <span className="arrow">{"> "}</span>
        {label}
      </span>
      <span className="tile-desc">{desc}</span>
    </>
  );
}

export function ModePicker({
  mode,
  onMode,
  slamBackend,
  onSlamBackend,
  gsplatBackend,
  onGsplatBackend,
  gsplatSourceId,
  onGsplatSource,
  sourceJobs,
  disabled,
}: Props) {
  const eligibleSources = selectableSplatSources(sourceJobs);
  const modeKeys: StudioMode[] = ["lingbot", "slam", "gsplat"];
  const slamKeys = Object.keys(SLAM_BACKENDS) as SlamBackend[];
  const gsplatKeys = Object.keys(GSPLAT_BACKENDS) as GsplatBackend[];

  return (
    <div className="panel" data-acc={MODES[mode].acc}>
      <div className="panel-header">
        <span>mode</span>
      </div>
      <div className="panel-body" style={{ display: "grid", gap: 8 }}>
        <div
          className="tile-row"
          style={{ gridTemplateColumns: "repeat(3, 1fr)" }}
        >
          {modeKeys.map((m, i) => {
            const meta = MODES[m];
            const active = mode === m;
            return (
              <button
                key={m}
                type="button"
                className={"tile" + (active ? " is-active" : "")}
                disabled={disabled}
                onClick={() => onMode(m)}
                data-pressed={active ? "true" : "false"}
                data-acc={meta.acc}
                style={{
                  borderRight: i < 2 ? "1px solid var(--rule)" : "none",
                }}
              >
                <TileContents label={meta.label} desc={meta.desc} />
              </button>
            );
          })}
        </div>

        {mode === "slam" && (
          <>
            <div className="section-title">backend</div>
            <div
              className="tile-row"
              style={{ gridTemplateColumns: "repeat(3, 1fr)" }}
            >
              {slamKeys.map((b, i) => {
                const meta = SLAM_BACKENDS[b];
                const active = slamBackend === b;
                return (
                  <button
                    key={b}
                    type="button"
                    className={"tile" + (active ? " is-active" : "")}
                    disabled={disabled}
                    onClick={() => onSlamBackend(b)}
                    data-pressed={active ? "true" : "false"}
                    data-acc={meta.acc}
                    style={{
                      borderRight:
                        i < slamKeys.length - 1
                          ? "1px solid var(--rule)"
                          : "none",
                    }}
                  >
                    <TileContents
                      label={SLAM_BACKEND_LABELS[b]}
                      desc={meta.hint}
                    />
                  </button>
                );
              })}
            </div>
          </>
        )}

        {mode === "gsplat" && (
          <>
            <div className="section-title">backend</div>
            <div
              className="tile-row"
              style={{ gridTemplateColumns: "repeat(2, 1fr)" }}
            >
              {gsplatKeys.map((b, i) => {
                const meta = GSPLAT_BACKENDS[b];
                const active = gsplatBackend === b;
                return (
                  <button
                    key={b}
                    type="button"
                    className={"tile" + (active ? " is-active" : "")}
                    disabled={disabled}
                    onClick={() => onGsplatBackend(b)}
                    data-pressed={active ? "true" : "false"}
                    data-acc={meta.acc}
                    style={{
                      borderRight:
                        i < gsplatKeys.length - 1
                          ? "1px solid var(--rule)"
                          : "none",
                    }}
                  >
                    <TileContents label={meta.label} desc={meta.hint} />
                  </button>
                );
              })}
            </div>

            {gsplatBackend === "gsplat" && (
              <>
                <div className="section-title">source job</div>
                {eligibleSources.length === 0 ? (
                  <div className="mono-small" style={{ opacity: 0.7 }}>
                    no ready slam or lingbot jobs to train from. run a slam
                    job first, then come back — or switch to MonoGS to train
                    directly from uploaded footage.
                  </div>
                ) : (
                  <select
                    value={gsplatSourceId ?? ""}
                    disabled={disabled}
                    onChange={(e) => onGsplatSource(e.target.value || null)}
                  >
                    <option value="">— pick a source —</option>
                    {eligibleSources.map((j) => (
                      <option key={j.id} value={j.id}>
                        {j.id} · {j.processor} · {j.frames_total ?? "?"} frames
                      </option>
                    ))}
                  </select>
                )}
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}
