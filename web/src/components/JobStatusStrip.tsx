"use client";

import { useEffect, useState } from "react";

import { type JobStatusDerived, type StageState } from "@/hooks/useJobStatus";
import type { JobStatus } from "@/lib/types";

interface Props {
  derived: JobStatusDerived;
  jobStatus: JobStatus | undefined;
  wsStatus: "connecting" | "open" | "closed";
}

function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !isFinite(seconds))
    return "—";
  if (seconds < 1) return `${Math.round(seconds * 1000)}ms`;
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return s === 0 ? `${m}m` : `${m}m ${s}s`;
}

function StageLine({ stage }: { stage: StageState }) {
  let meta = "—";
  if (stage.state === "active") {
    meta =
      stage.latest_progress !== null
        ? `${Math.round(stage.latest_progress * 100)}%`
        : "…";
  } else if (stage.state === "done" || stage.state === "failed") {
    meta =
      stage.duration_s !== null ? formatDuration(stage.duration_s) : "—";
  } else {
    meta = "pending";
  }
  return (
    <div className="stage-line" data-state={stage.state}>
      <span className="sl-glyph" />
      <span className="sl-name">{stage.name}</span>
      <span className="sl-meta">{meta}</span>
    </div>
  );
}

function VramSparkline({
  samples,
  limit,
  total,
}: {
  samples: JobStatusDerived["vram"]["samples"];
  limit: number | null;
  total: number | null;
}) {
  if (samples.length < 2) {
    return (
      <div
        style={{
          height: 28,
          display: "grid",
          placeItems: "center",
          color: "var(--muted)",
          fontSize: "var(--fs-xs)",
        }}
      >
        awaiting readings…
      </div>
    );
  }
  const max = Math.max(
    total ?? 0,
    limit ?? 0,
    ...samples.map((s) => s.allocated_gb),
  );
  if (!isFinite(max) || max <= 0) return null;
  const n = samples.length;
  const w = 100;
  const h = 100;
  const xStep = w / Math.max(1, n - 1);
  const pts = samples.map((s, i) => {
    const x = i * xStep;
    const y = h - (s.allocated_gb / max) * h;
    return `${x.toFixed(2)},${y.toFixed(2)}`;
  });
  const line = `M ${pts.join(" L ")}`;
  const area = `M 0,${h} L ${pts.join(" L ")} L ${w.toFixed(2)},${h} Z`;
  return (
    <svg
      className="vram-sparkline"
      viewBox={`0 0 ${w} ${h}`}
      preserveAspectRatio="none"
    >
      <path className="area" d={area} />
      <path className="line" d={line} />
    </svg>
  );
}

function VramCell({ derived }: { derived: JobStatusDerived }) {
  const { vram } = derived;
  const total = vram.total_gb ?? 24;
  const cap = vram.limit_gb ?? total;
  const upper = Math.max(total, cap);
  const current = vram.current_gb ?? 0;
  const peak = vram.peak_gb ?? current;

  const usedPct = Math.min(100, (current / upper) * 100);
  const peakPct = Math.min(100, (peak / upper) * 100);
  const limitPct = Math.min(100, (cap / upper) * 100);
  const danger = current > cap * 0.92;

  return (
    <div className="status-cell">
      <div style={{ display: "flex", justifyContent: "space-between" }}>
        <span className="status-label">vram · gpu memory</span>
        <span className="kv-row">
          <span>
            <b>{current.toFixed(2)}</b> / {upper.toFixed(1)} GB
          </span>
        </span>
      </div>
      <div className="vram-bar" data-danger={danger}>
        <div className="used" style={{ width: `${usedPct}%` }} />
        <div className="peak-mark" style={{ left: `${peakPct}%` }} />
        <div className="limit-mark" style={{ left: `${limitPct}%` }} />
      </div>
      <div className="kv-row">
        <span>
          peak <b>{peak.toFixed(2)} GB</b>
        </span>
        <span>
          soft <b>{cap.toFixed(1)} GB</b>
        </span>
        {vram.samples.length > 0 && (
          <span>samples <b>{vram.samples.length}</b></span>
        )}
      </div>
      <VramSparkline
        samples={vram.samples}
        limit={cap}
        total={total}
      />
    </div>
  );
}

function StagesCell({ derived, wsStatus }: { derived: JobStatusDerived; wsStatus: string }) {
  const activeMsg = derived.activeStage
    ? derived.stages.find((s) => s.name === derived.activeStage)?.latest_message ?? ""
    : "";
  return (
    <div className="status-cell">
      <div style={{ display: "flex", justifyContent: "space-between" }}>
        <span className="status-label">pipeline</span>
        <span className="kv-row">
          {derived.elapsed_s !== null && (
            <span>
              elapsed <b>{formatDuration(derived.elapsed_s)}</b>
            </span>
          )}
          <span>
            ws <b>{wsStatus}</b>
          </span>
        </span>
      </div>
      <div className="stage-list">
        {derived.stages.map((s) => (
          <StageLine key={s.name} stage={s} />
        ))}
      </div>
      {activeMsg && (
        <div className="latest-msg" title={activeMsg}>
          {activeMsg}
        </div>
      )}
    </div>
  );
}

function ProgressCell({
  derived,
  jobStatus,
}: {
  derived: JobStatusDerived;
  jobStatus: JobStatus | undefined;
}) {
  // Ticker: animate displayed frame count toward the real one. Purely cosmetic.
  const [displayFrame, setDisplayFrame] = useState(derived.frame.done ?? 0);
  useEffect(() => {
    const target = derived.frame.done ?? 0;
    if (target === displayFrame) return;
    const step = Math.max(1, Math.ceil(Math.abs(target - displayFrame) / 8));
    const id = setInterval(() => {
      setDisplayFrame((prev) => {
        if (prev === target) return prev;
        if (prev < target) return Math.min(target, prev + step);
        return Math.max(target, prev - step);
      });
    }, 30);
    return () => clearInterval(id);
  }, [derived.frame.done, displayFrame]);

  const done = displayFrame;
  const total = derived.frame.total;
  const pct = derived.frame.percent;
  const fps = derived.frame.fps;
  const eta = derived.frame.eta_s;
  const active = derived.activeStage;

  // During ingest/checkpoint we haven't got a frame count yet — use stage
  // progress if present, else show an indeterminate bar.
  const indeterminate =
    active !== null &&
    active !== "inference" &&
    (pct === null || pct === 0);

  const terminal = jobStatus === "ready" || jobStatus === "failed";

  return (
    <div className="status-cell">
      <div style={{ display: "flex", justifyContent: "space-between" }}>
        <span className="status-label">
          {terminal
            ? jobStatus === "ready"
              ? "ready"
              : "failed"
            : active ?? "queued"}
        </span>
        {(pct !== null || terminal) && (
          <span className="kv-row">
            <span>
              <b>
                {Math.round((terminal ? 1 : pct ?? 0) * 100)}%
              </b>
            </span>
          </span>
        )}
      </div>
      <div className="bignum">
        {active === "inference" || derived.frame.total !== null ? (
          <>
            <span>{done.toLocaleString()}</span>
            <span style={{ color: "var(--muted)", fontSize: 16 }}>
              / {total?.toLocaleString() ?? "—"}
            </span>
            <span style={{ color: "var(--muted)", fontSize: 12, marginLeft: 4 }}>
              frames
            </span>
            {!terminal && <span className="cursor" />}
          </>
        ) : (
          <span style={{ fontSize: 16, letterSpacing: "0.06em" }}>
            {active ? active.toUpperCase() : "QUEUED"}
            {!terminal && <span className="cursor" />}
          </span>
        )}
      </div>
      <div className={`pb${indeterminate ? " indeterminate" : ""}`}>
        <div
          className="fill"
          style={{
            width: indeterminate
              ? undefined
              : `${Math.round((terminal ? 1 : pct ?? 0) * 100)}%`,
          }}
        />
      </div>
      <div className="kv-row">
        {fps !== null && fps > 0 && (
          <span>
            <b>{fps.toFixed(2)}</b> fps
          </span>
        )}
        {eta !== null && eta > 0 && (
          <span>
            eta <b>{formatDuration(eta)}</b>
          </span>
        )}
        {derived.elapsed_s !== null && fps === null && !terminal && (
          <span>
            <b>{formatDuration(derived.elapsed_s)}</b> elapsed
          </span>
        )}
      </div>
    </div>
  );
}

export function JobStatusStrip({ derived, jobStatus, wsStatus }: Props) {
  return (
    <div className="status-strip">
      <StagesCell derived={derived} wsStatus={wsStatus} />
      <ProgressCell derived={derived} jobStatus={jobStatus} />
      <VramCell derived={derived} />
    </div>
  );
}
