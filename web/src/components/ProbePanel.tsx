"use client";

import type { DraftRecord } from "@/lib/api";

function human(n: number | null, unit: string): string {
  if (n === null || n === undefined) return "—";
  return `${n}${unit}`;
}

function humanBytes(n: number | null): string {
  if (n === null || n === undefined) return "—";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} kB`;
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

function humanDuration(s: number | null): string {
  if (s === null || s === undefined) return "—";
  const total = Math.round(s);
  const mm = Math.floor(total / 60);
  const ss = total % 60;
  return `${mm}:${String(ss).padStart(2, "0")}`;
}

function humanBitrate(bps: number | null): string {
  if (!bps) return "—";
  if (bps < 1_000_000) return `${(bps / 1000).toFixed(0)} kb/s`;
  return `${(bps / 1_000_000).toFixed(1)} Mb/s`;
}

export function ProbePanel({ draft }: { draft: DraftRecord }) {
  const totalDuration = draft.probes.reduce(
    (acc, p) => acc + (p.duration_s || 0),
    0,
  );
  const totalFrames = draft.probes.reduce(
    (acc, p) => acc + (p.total_frames || 0),
    0,
  );

  return (
    <div className="panel">
      <div className="panel-header">
        <span>probed video{draft.probes.length === 1 ? "" : "s"}</span>
        <span className="meta">
          {draft.probes.length} file{draft.probes.length === 1 ? "" : "s"} ·{" "}
          {humanDuration(totalDuration)} · {totalFrames.toLocaleString()} frames
        </span>
      </div>
      <div className="panel-body" style={{ padding: 0 }}>
        <table className="dtable">
          <colgroup>
            <col style={{ width: "36px" }} />
            <col />
            <col style={{ width: "80px" }} />
            <col style={{ width: "100px" }} />
            <col style={{ width: "72px" }} />
            <col style={{ width: "60px" }} />
            <col style={{ width: "64px" }} />
            <col style={{ width: "84px" }} />
            <col style={{ width: "84px" }} />
          </colgroup>
          <thead>
            <tr>
              <th className="num">#</th>
              <th>file</th>
              <th className="num">size</th>
              <th>res</th>
              <th>codec</th>
              <th className="num">fps</th>
              <th className="num">dur</th>
              <th className="num">bitrate</th>
              <th className="num">frames</th>
            </tr>
          </thead>
          <tbody>
            {draft.probes.map((p, i) => (
              <tr key={p.name}>
                <td className="num">{i + 1}</td>
                <td className="wrap">{p.name}</td>
                <td className="num">{humanBytes(p.size_bytes)}</td>
                <td>
                  {p.width && p.height ? `${p.width}×${p.height}` : "—"}
                </td>
                <td>{p.codec ?? "—"}</td>
                <td className="num">{human(p.fps, "")}</td>
                <td className="num">{humanDuration(p.duration_s)}</td>
                <td className="num">{humanBitrate(p.bitrate)}</td>
                <td className="num">
                  {p.total_frames?.toLocaleString() ?? "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {draft.probes.some((p) => p.error) && (
          <div
            style={{
              padding: "6px 10px",
              color: "var(--danger)",
              fontSize: "var(--fs-xs)",
              borderTop: "1px solid var(--rule)",
            }}
          >
            {draft.probes
              .filter((p) => p.error)
              .map((p) => `${p.name}: ${p.error}`)
              .join(" · ")}
          </div>
        )}
      </div>
    </div>
  );
}
