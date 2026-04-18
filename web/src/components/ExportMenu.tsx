"use client";

import { useState } from "react";

import { artifactUrl, reexport } from "@/lib/api";

interface Props {
  jobId: string;
  artifacts: Array<{ name: string; suffix: string; size: number }>;
  latestMesh: string | null;
  onReexport: (name: string) => void;
}

function humanSize(n: number): string {
  if (n < 1024) return `${n} b`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} kb`;
  return `${(n / 1024 / 1024).toFixed(1)} mb`;
}

export function ExportMenu({ jobId, artifacts, latestMesh, onReexport }: Props) {
  const [format, setFormat] = useState<"glb" | "ply" | "obj">("glb");
  const [percentile, setPercentile] = useState(50);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
    <div className="panel">
      <div className="panel-header">export</div>
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
          <span>conf %ile</span>
          <input
            type="number"
            value={percentile}
            min={0}
            max={95}
            step={5}
            onChange={(e) => setPercentile(Number(e.target.value))}
            style={{ width: 70, textAlign: "right" }}
          />
        </label>
        <button type="button" disabled={busy} onClick={runReexport}>
          {busy ? "re-exporting..." : "re-export"}
        </button>
        {error && <div style={{ color: "var(--danger)", fontSize: 11 }}>{error}</div>}

        <hr />
        <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 2 }}>
          downloads
        </div>
        <div style={{ display: "grid", gap: 2 }}>
          {artifacts.length === 0 && (
            <span style={{ color: "var(--muted)", fontSize: 11 }}>
              none yet
            </span>
          )}
          {artifacts.map((a) => (
            <a
              key={a.name}
              href={artifactUrl(jobId, a.name)}
              download={a.name}
              style={{
                fontSize: 11,
                display: "flex",
                justifyContent: "space-between",
                gap: 8,
                padding: "1px 4px",
                textDecoration: "none",
                border: latestMesh === a.name ? "1px solid var(--rule)" : "1px solid transparent",
              }}
            >
              <span style={{ wordBreak: "break-all" }}>{a.name}</span>
              <span style={{ color: "var(--muted)" }}>{humanSize(a.size)}</span>
            </a>
          ))}
        </div>
      </div>
    </div>
  );
}
