"use client";

import Link from "next/link";

import { useJobList } from "@/hooks/useJob";

function relTime(iso: string): string {
  const ms = Date.now() - new Date(iso).getTime();
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s}s ago`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.round(h / 24);
  return `${d}d ago`;
}

export function JobList() {
  const { data, error, isLoading } = useJobList();
  return (
    <div className="panel">
      <div className="panel-header">
        <span>jobs</span>
        {data && <span className="meta">{data.length}</span>}
      </div>
      <div className="panel-body" style={{ padding: 0 }}>
        {isLoading && (
          <div style={{ padding: "var(--pad)", color: "var(--muted)" }}>loading...</div>
        )}
        {error && (
          <div
            style={{
              padding: "var(--pad)",
              color: "var(--danger)",
              fontSize: "var(--fs-sm)",
            }}
          >
            error: {String((error as Error).message)}
          </div>
        )}
        {data && data.length === 0 && (
          <div style={{ padding: "var(--pad)", color: "var(--muted)" }}>
            no jobs yet
          </div>
        )}
        {data && data.length > 0 && (
          <table className="dtable">
            <colgroup>
              <col style={{ width: "130px" }} />
              <col style={{ width: "110px" }} />
              <col style={{ width: "80px" }} />
              <col style={{ width: "90px" }} />
              <col />
              <col style={{ width: "80px" }} />
            </colgroup>
            <thead>
              <tr>
                <th>id</th>
                <th>status</th>
                <th className="num">frames</th>
                <th className="num">artifacts</th>
                <th>created</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {data.map((j) => (
                <tr key={j.id}>
                  <td>{j.id}</td>
                  <td>
                    <span className="chip" data-status={j.status}>
                      {j.status}
                    </span>
                  </td>
                  <td className="num">{j.frames_total ?? "—"}</td>
                  <td className="num">{j.artifact_count}</td>
                  <td className="mono-small">{relTime(j.created_at)}</td>
                  <td>
                    <Link href={`/jobs/${j.id}`}>open</Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
