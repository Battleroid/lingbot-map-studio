"use client";

import { useMemo, useState } from "react";

import { Tip } from "@/components/Tip";
import { artifactUrl, reexport } from "@/lib/api";
import { useCollapsible } from "@/lib/useCollapsible";

interface Props {
  jobId: string;
  artifacts: Array<{ name: string; suffix: string; size: number }>;
  latestMesh: string | null;
  onReexport: (name: string) => void;
  /** Show a click-to-collapse arrow on the header (job page side pane). */
  collapsible?: boolean;
}

// Partial-snapshot files (`partial_*.ply`, `partial_splat_*.ply`) pile
// up during long runs — a 30 k-iter gsplat job emits 30+ of them. They
// aren't useful to download (they're internal preview snapshots), so
// hide them from the primary downloads list and surface a single
// summary row instead.
function isPartialSnapshot(name: string): boolean {
  return /^partial_/.test(name);
}

function humanSize(n: number): string {
  if (n < 1024) return `${n} b`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} kb`;
  return `${(n / 1024 / 1024).toFixed(1)} mb`;
}

export function ExportMenu({
  jobId,
  artifacts,
  latestMesh,
  onReexport,
  collapsible,
}: Props) {
  const [format, setFormat] = useState<"glb" | "ply" | "obj">("glb");
  const [percentile, setPercentile] = useState(50);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showPartials, setShowPartials] = useState(false);
  const c = useCollapsible({ enabled: collapsible });

  // Split artifacts: real downloads (final outputs the user wants) vs
  // partials (preview snapshots that pile up during long runs).
  const { mainArtifacts, partialArtifacts } = useMemo(() => {
    const main: typeof artifacts = [];
    const partial: typeof artifacts = [];
    for (const a of artifacts) {
      (isPartialSnapshot(a.name) ? partial : main).push(a);
    }
    return { mainArtifacts: main, partialArtifacts: partial };
  }, [artifacts]);

  async function runReexport() {
    setBusy(true);
    setError(null);
    try {
      const res = await reexport(jobId, { format, conf_percentile: percentile });
      onReexport(res.name);
    } catch (e) {
      setError(String((e as Error).message));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="panel" {...c.panelProps}>
      <div className="panel-header" {...c.headerProps}>
        <span>
          {c.arrow}export
        </span>
      </div>
      <div className="panel-body" style={{ display: "grid", gap: 8 }}>
        <div style={{ display: "flex", gap: 4 }}>
          {(["glb", "ply", "obj"] as const).map((f) => (
            <button
              key={f}
              type="button"
              data-pressed={format === f}
              onClick={() => setFormat(f)}
              style={{ flex: 1 }}
            >
              {f}
            </button>
          ))}
        </div>
        <label className="stat">
          <Tip text="Re-filter the cached predictions at a different percentile cutoff. Higher = stricter filter (fewer, more-confident points).">
            <span>conf %ile</span>
          </Tip>
          <input
            type="number"
            value={percentile}
            min={0}
            max={95}
            step={5}
            onChange={(e) => setPercentile(Number(e.target.value))}
          />
        </label>
        <button type="button" disabled={busy} onClick={runReexport}>
          {busy ? "re-exporting..." : "re-export"}
        </button>
        {error && (
          <div style={{ color: "var(--danger)", fontSize: "var(--fs-xs)" }}>{error}</div>
        )}

        <hr />
        <div className="section-title">downloads</div>
        {/* Cap the height + scroll. Without this the panel grows to
            match the artifact list, which during a 30k-iter gsplat
            job (30+ partial snapshots) takes over the entire side
            column. Partials hide behind a click-to-show toggle so
            they don't pollute the list either. */}
        <div
          style={{
            display: "grid",
            gap: 2,
            maxHeight: 240,
            overflowY: "auto",
            border: "1px solid var(--rule)",
            borderRadius: "var(--r-xs)",
            padding: 4,
          }}
        >
          {mainArtifacts.length === 0 && partialArtifacts.length === 0 && (
            <span className="mono-small">none yet</span>
          )}
          {mainArtifacts.map((a) => (
            <a
              key={a.name}
              href={artifactUrl(jobId, a.name)}
              download={a.name}
              title={a.name}
              style={{
                fontSize: "var(--fs-xs)",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                gap: 8,
                padding: "2px 4px",
                minWidth: 0,
                textDecoration: "none",
                border:
                  latestMesh === a.name
                    ? "1px solid var(--rule)"
                    : "1px solid transparent",
              }}
            >
              <span
                style={{
                  flex: "1 1 auto",
                  minWidth: 0,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}
              >
                {a.name}
              </span>
              <span style={{ color: "var(--muted)", flex: "0 0 auto" }}>
                {humanSize(a.size)}
              </span>
            </a>
          ))}
          {partialArtifacts.length > 0 && (
            <>
              <button
                type="button"
                onClick={() => setShowPartials((v) => !v)}
                style={{
                  fontSize: "var(--fs-xs)",
                  textAlign: "left",
                  padding: "2px 4px",
                }}
              >
                {showPartials ? "▾" : "▸"} {partialArtifacts.length} partial
                snapshot
                {partialArtifacts.length === 1 ? "" : "s"}
              </button>
              {showPartials &&
                partialArtifacts.map((a) => (
                  <a
                    key={a.name}
                    href={artifactUrl(jobId, a.name)}
                    download={a.name}
                    title={a.name}
                    style={{
                      fontSize: "var(--fs-xs)",
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      gap: 8,
                      padding: "2px 4px 2px 16px",
                      minWidth: 0,
                      textDecoration: "none",
                      color: "var(--muted)",
                    }}
                  >
                    <span
                      style={{
                        flex: "1 1 auto",
                        minWidth: 0,
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {a.name}
                    </span>
                    <span style={{ flex: "0 0 auto" }}>
                      {humanSize(a.size)}
                    </span>
                  </a>
                ))}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
