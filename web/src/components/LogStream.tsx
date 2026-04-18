"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import type { JobEvent } from "@/lib/types";

const FRESH_MS = 1200;

const LEVEL_ORDER: Record<string, number> = {
  error: 0,
  warn: 1,
  info: 2,
  stdout: 3,
  stderr: 4,
  debug: 5,
};

const ALL_LEVELS = ["info", "warn", "error", "stdout", "stderr", "debug"] as const;

export function LogStream({ events }: { events: JobEvent[] }) {
  const [enabled, setEnabled] = useState<Record<string, boolean>>({
    info: true,
    warn: true,
    error: true,
    stdout: true,
    stderr: true,
    debug: false,
  });
  const [autoscroll, setAutoscroll] = useState(true);
  const scrollRef = useRef<HTMLDivElement>(null);
  const mountedAt = useRef(Date.now() + FRESH_MS);

  const filtered = useMemo(
    () => events.filter((e) => enabled[e.level] ?? true),
    [events, enabled],
  );

  // Tick to re-render and drop the "fresh" class off old log lines.
  const [, setFreshTick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setFreshTick((x) => x + 1), 500);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    if (!autoscroll) return;
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [filtered, autoscroll]);

  return (
    <div className="panel" style={{ height: "100%", border: "none" }}>
      <div className="panel-header">
        <span>log</span>
        <span className="meta">
          {filtered.length}/{events.length}
        </span>
      </div>
      <div
        style={{
          padding: "5px 10px",
          borderBottom: "1px solid var(--rule)",
          display: "flex",
          flexWrap: "wrap",
          gap: 4,
          alignItems: "center",
        }}
      >
        {ALL_LEVELS.slice()
          .sort((a, b) => LEVEL_ORDER[a] - LEVEL_ORDER[b])
          .map((lvl) => (
            <button
              key={lvl}
              type="button"
              data-pressed={enabled[lvl]}
              onClick={() => setEnabled({ ...enabled, [lvl]: !enabled[lvl] })}
              style={{ padding: "1px 6px", fontSize: "var(--fs-xs)" }}
            >
              {lvl}
            </button>
          ))}
        <label
          style={{
            display: "flex",
            gap: 4,
            alignItems: "center",
            marginLeft: "auto",
            fontSize: "var(--fs-xs)",
          }}
        >
          <input
            type="checkbox"
            checked={autoscroll}
            onChange={(e) => setAutoscroll(e.target.checked)}
          />
          autoscroll
        </label>
      </div>
      <div ref={scrollRef} className="log-stream">
        {filtered.map((ev) => {
          const evTs = Date.parse(ev.created_at);
          const isFresh =
            evTs > mountedAt.current && Date.now() - evTs < FRESH_MS;
          return (
            <div
              key={ev.id}
              className={`log-line${isFresh ? " fresh" : ""}`}
              data-level={ev.level}
            >
              <span style={{ color: "var(--muted)" }}>
                [{ev.stage}]
                {ev.progress !== null ? ` ${(ev.progress * 100).toFixed(0)}%` : ""}{" "}
              </span>
              {ev.message}
            </div>
          );
        })}
        {filtered.length === 0 && (
          <div style={{ color: "var(--muted)" }}>no events</div>
        )}
      </div>
    </div>
  );
}
