"use client";

import type {
  GsplatBackend,
  JobSummary,
  ProcessorId,
  SlamBackend,
} from "@/lib/types";

export type StudioMode = "lingbot" | "slam" | "gsplat";

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

const SLAM_BACKEND_LABELS: Record<SlamBackend, string> = {
  mast3r_slam: "MASt3R-SLAM",
  droid_slam: "DROID-SLAM",
  dpvo: "DPVO",
};

const SLAM_BACKEND_HINTS: Record<SlamBackend, string> = {
  mast3r_slam: "calibration-free · best default for FPV",
  droid_slam: "dense · VRAM-heavy · high fidelity",
  dpvo: "patch-based · lightweight · long clips",
};

const GSPLAT_BACKEND_LABELS: Record<GsplatBackend, string> = {
  gsplat: "gsplat · chain from source",
  monogs: "MonoGS · direct from footage",
};

const GSPLAT_BACKEND_HINTS: Record<GsplatBackend, string> = {
  gsplat:
    "train a 3DGS scene off a completed SLAM or Lingbot job (reuses its frames + poses + cloud)",
  monogs:
    "end-to-end splat SLAM — upload footage and get a splat directly, no intermediate SLAM job required",
};

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

  return (
    <div className="panel">
      <div className="panel-header">
        <span>mode</span>
      </div>
      <div className="panel-body" style={{ display: "grid", gap: 8 }}>
        <div style={{ display: "flex", gap: 4 }}>
          {(["lingbot", "slam", "gsplat"] as StudioMode[]).map((m) => (
            <button
              key={m}
              type="button"
              disabled={disabled}
              onClick={() => onMode(m)}
              aria-pressed={mode === m}
              style={{
                flex: 1,
                background:
                  mode === m ? "var(--fg)" : "var(--panel-bg, transparent)",
                color: mode === m ? "var(--bg)" : "var(--fg)",
              }}
            >
              {m === "lingbot"
                ? "lingbot"
                : m === "slam"
                  ? "slam"
                  : "gaussian splat"}
            </button>
          ))}
        </div>

        {mode === "slam" && (
          <>
            <div className="section-title">backend</div>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(3, 1fr)",
                gap: 4,
              }}
            >
              {(Object.keys(SLAM_BACKEND_LABELS) as SlamBackend[]).map((b) => (
                <button
                  key={b}
                  type="button"
                  disabled={disabled}
                  onClick={() => onSlamBackend(b)}
                  aria-pressed={slamBackend === b}
                  style={{
                    background:
                      slamBackend === b
                        ? "var(--fg)"
                        : "var(--panel-bg, transparent)",
                    color: slamBackend === b ? "var(--bg)" : "var(--fg)",
                  }}
                >
                  {SLAM_BACKEND_LABELS[b]}
                </button>
              ))}
            </div>
            <div className="mono-small" style={{ opacity: 0.7 }}>
              {SLAM_BACKEND_HINTS[slamBackend]}
            </div>
          </>
        )}

        {mode === "gsplat" && (
          <>
            <div className="section-title">backend</div>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(2, 1fr)",
                gap: 4,
              }}
            >
              {(Object.keys(GSPLAT_BACKEND_LABELS) as GsplatBackend[]).map(
                (b) => (
                  <button
                    key={b}
                    type="button"
                    disabled={disabled}
                    onClick={() => onGsplatBackend(b)}
                    aria-pressed={gsplatBackend === b}
                    style={{
                      background:
                        gsplatBackend === b
                          ? "var(--fg)"
                          : "var(--panel-bg, transparent)",
                      color: gsplatBackend === b ? "var(--bg)" : "var(--fg)",
                    }}
                  >
                    {GSPLAT_BACKEND_LABELS[b]}
                  </button>
                ),
              )}
            </div>
            <div className="mono-small" style={{ opacity: 0.7 }}>
              {GSPLAT_BACKEND_HINTS[gsplatBackend]}
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
