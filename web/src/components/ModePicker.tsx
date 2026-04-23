"use client";

import { ThreeTile, type TileId } from "@/components/ThreeTile";
import type {
  GsplatBackend,
  JobSummary,
  ProcessorId,
  SlamBackend,
} from "@/lib/types";

export type StudioMode = "lingbot" | "slam" | "gsplat";

// Which three.js tile each mode / backend shows in its picker card.
// Kept here (not in types.ts) because it's a UI-only concern.
const MODE_TILES: Record<StudioMode, TileId> = {
  lingbot: "lingbot",
  slam: "slam",
  gsplat: "gsplat",
};
const MODE_DESCRIPTIONS: Record<StudioMode, string> = {
  lingbot:
    "feed-forward reconstruction. point cloud + textured mesh + camera path.",
  slam: "mast3r, droid, dpvo. frame-by-frame tracking + map.",
  gsplat:
    "gaussian splat training. chains off a slam/lingbot job, or runs monogs directly.",
};
const SLAM_BACKEND_TILES: Record<SlamBackend, TileId> = {
  mast3r_slam: "mast3r",
  droid_slam: "droid",
  dpvo: "dpvo",
};
const GSPLAT_BACKEND_TILES: Record<GsplatBackend, TileId> = {
  gsplat: "gsplat",
  monogs: "monogs",
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
        <div
          className="tile-row"
          style={{ gridTemplateColumns: "repeat(3, 1fr)" }}
        >
          {(["lingbot", "slam", "gsplat"] as StudioMode[]).map((m, i) => {
            const active = mode === m;
            return (
              <button
                key={m}
                type="button"
                className={"tile" + (active ? " is-active" : "")}
                disabled={disabled}
                onClick={() => onMode(m)}
                data-pressed={active ? "true" : "false"}
                style={{
                  borderRight: i < 2 ? "1px solid var(--rule)" : "none",
                }}
              >
                <ThreeTile
                  tile={MODE_TILES[m]}
                  height={60}
                  className="tile-canvas"
                  ariaLabel={`${m} preview`}
                />
                <span className="tile-label">
                  {active ? "> " : ""}
                  {m === "gsplat" ? "gaussian splat" : m}
                </span>
                <span className="tile-desc">{MODE_DESCRIPTIONS[m]}</span>
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
              {(Object.keys(SLAM_BACKEND_LABELS) as SlamBackend[]).map(
                (b, i, arr) => {
                  const active = slamBackend === b;
                  return (
                    <button
                      key={b}
                      type="button"
                      className={"tile" + (active ? " is-active" : "")}
                      disabled={disabled}
                      onClick={() => onSlamBackend(b)}
                      data-pressed={active ? "true" : "false"}
                      style={{
                        borderRight:
                          i < arr.length - 1
                            ? "1px solid var(--rule)"
                            : "none",
                      }}
                    >
                      <ThreeTile
                        tile={SLAM_BACKEND_TILES[b]}
                        height={52}
                        className="tile-canvas"
                        ariaLabel={`${b} preview`}
                      />
                      <span className="tile-label">
                        {active ? "> " : ""}
                        {SLAM_BACKEND_LABELS[b]}
                      </span>
                      <span className="tile-desc">
                        {SLAM_BACKEND_HINTS[b]}
                      </span>
                    </button>
                  );
                },
              )}
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
              {(Object.keys(GSPLAT_BACKEND_LABELS) as GsplatBackend[]).map(
                (b, i, arr) => {
                  const active = gsplatBackend === b;
                  return (
                    <button
                      key={b}
                      type="button"
                      className={"tile" + (active ? " is-active" : "")}
                      disabled={disabled}
                      onClick={() => onGsplatBackend(b)}
                      data-pressed={active ? "true" : "false"}
                      style={{
                        borderRight:
                          i < arr.length - 1
                            ? "1px solid var(--rule)"
                            : "none",
                      }}
                    >
                      <ThreeTile
                        tile={GSPLAT_BACKEND_TILES[b]}
                        height={52}
                        className="tile-canvas"
                        ariaLabel={`${b} preview`}
                      />
                      <span className="tile-label">
                        {active ? "> " : ""}
                        {GSPLAT_BACKEND_LABELS[b]}
                      </span>
                      <span className="tile-desc">
                        {GSPLAT_BACKEND_HINTS[b]}
                      </span>
                    </button>
                  );
                },
              )}
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
