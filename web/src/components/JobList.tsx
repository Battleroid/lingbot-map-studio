"use client";

import Link from "next/link";
import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

import { useJobList } from "@/hooks/useJob";
import { deleteJob } from "@/lib/api";
import type { JobStatus } from "@/lib/types";

/**
 * Relative time, tolerant of client/server clock drift.
 * Negative deltas (server ahead of browser) clamp to "just now".
 */
function relTime(iso: string): string {
  const then = new Date(iso).getTime();
  if (!isFinite(then)) return "—";
  const ms = Date.now() - then;
  if (ms < 5_000) return "just now";
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s}s ago`;
  const m = Math.round(s / 60);
  if (m === 1) return "1 min ago";
  if (m < 60) return `${m} min ago`;
  const h = Math.round(m / 60);
  if (h === 1) return "1 hr ago";
  if (h < 24) return `${h} hr ago`;
  const d = Math.round(h / 24);
  if (d === 1) return "yesterday";
  if (d < 7) return `${d} days ago`;
  if (d < 30) return `${Math.round(d / 7)} wk ago`;
  // older than ~a month: render the date itself
  const date = new Date(then);
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

/**
 * Local-readable absolute timestamp for hover tooltips.
 * Shown like "2026-04-17 22:15:03 EDT".
 */
function absTime(iso: string): string {
  const t = new Date(iso);
  if (!isFinite(t.getTime())) return iso;
  const pad = (n: number) => String(n).padStart(2, "0");
  const tzMatch = t.toString().match(/\(([^)]+)\)$/);
  const tz = tzMatch ? tzMatch[1] : "";
  return (
    `${t.getFullYear()}-${pad(t.getMonth() + 1)}-${pad(t.getDate())} ` +
    `${pad(t.getHours())}:${pad(t.getMinutes())}:${pad(t.getSeconds())}` +
    (tz ? ` ${tz}` : "") +
    `\n${iso}`
  );
}

/** Per-status accent for row tinting + mini-pb fill color. */
function statusAcc(s: JobStatus): "green" | "amber" | "cyan" | "coral" | "muted" {
  if (s === "ready") return "green";
  if (s === "failed") return "coral";
  if (s === "cancelled") return "muted";
  if (s === "queued") return "cyan";
  return "amber"; // preproc / ingest / inference / meshing / export
}

/** Coarse progress for the table mini-bar. Per-job real progress would
 * require a WS subscription per row — heavy for a list view, deferred to
 * the job detail page. Terminal states pin at 100%, queued at 0%, and
 * in-progress states render the shimmering bar at a deterministic 50%
 * just so the row carries motion. */
function progressFor(s: JobStatus): { pct: number; isStatic: boolean } {
  if (s === "ready" || s === "failed") return { pct: 100, isStatic: true };
  if (s === "cancelled") return { pct: 100, isStatic: true };
  if (s === "queued") return { pct: 0, isStatic: false };
  return { pct: 50, isStatic: false };
}

function StageIndicator({ status }: { status: JobStatus }) {
  return (
    <span className="stage-ind" data-status={status}>
      <span className="glyph" />
      <span>{status}</span>
    </span>
  );
}

export function JobList() {
  const { data, error, isLoading } = useJobList();
  const qc = useQueryClient();
  const [busyId, setBusyId] = useState<string | null>(null);
  const [actionErr, setActionErr] = useState<string | null>(null);

  async function onDelete(id: string) {
    if (
      !window.confirm(
        `Delete job ${id}? All artifacts and uploads will be removed from disk.`,
      )
    ) {
      return;
    }
    setBusyId(id);
    setActionErr(null);
    try {
      await deleteJob(id);
      // Invalidate so the list refreshes without the deleted row.
      await qc.invalidateQueries({ queryKey: ["jobs"] });
    } catch (e) {
      setActionErr(String((e as Error).message));
    } finally {
      setBusyId(null);
    }
  }

  return (
    <div className="panel" data-acc="cyan">
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
          <div
            style={{
              padding: "22px 10px",
              textAlign: "center",
              color: "var(--muted)",
            }}
          >
            no jobs yet
          </div>
        )}
        {data && data.length > 0 && (
          <table className="dtable">
            <colgroup>
              <col style={{ width: "130px" }} />
              <col style={{ width: "90px" }} />
              <col style={{ width: "120px" }} />
              <col />
              <col style={{ width: "70px" }} />
              <col style={{ width: "100px" }} />
              <col style={{ width: "70px" }} />
            </colgroup>
            <thead>
              <tr>
                <th>id</th>
                <th>mode</th>
                <th>stage</th>
                <th>progress</th>
                <th className="num">frames</th>
                <th>created</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {data.map((j) => {
                const terminal =
                  j.status === "ready" ||
                  j.status === "failed" ||
                  j.status === "cancelled";
                const acc = statusAcc(j.status);
                const { pct, isStatic } = progressFor(j.status);
                return (
                  <tr key={j.id} className="job-row" data-status={j.status}>
                    <td>
                      <Link
                        href={`/jobs/${j.id}`}
                        className="row-link"
                        title={j.id}
                      >
                        {j.id}
                      </Link>
                    </td>
                    <td>{j.processor}</td>
                    <td>
                      <StageIndicator status={j.status} />
                    </td>
                    <td>
                      <div className="prog-cell">
                        <span
                          className={"mini-pb" + (isStatic ? " static" : "")}
                          data-acc={acc}
                        >
                          <span
                            className="fill"
                            style={{ width: `${pct}%` }}
                          />
                        </span>
                        <span className="pct">{pct}%</span>
                      </div>
                    </td>
                    <td className="num">{j.frames_total ?? "—"}</td>
                    <td className="mono-small" title={absTime(j.created_at)}>
                      {relTime(j.created_at)}
                    </td>
                    <td>
                      <button
                        type="button"
                        disabled={!terminal || busyId === j.id}
                        onClick={() => onDelete(j.id)}
                        title={
                          terminal
                            ? "Delete job and its artifacts"
                            : "Stop the job first"
                        }
                        style={{ padding: "1px 6px" }}
                      >
                        {busyId === j.id ? "…" : "delete"}
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
        {actionErr && (
          <div
            style={{
              padding: "6px var(--pad)",
              color: "var(--danger)",
              fontSize: "var(--fs-xs)",
            }}
          >
            {actionErr}
          </div>
        )}
      </div>
    </div>
  );
}
