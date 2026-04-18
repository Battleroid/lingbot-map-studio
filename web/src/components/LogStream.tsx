"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import type { JobEvent } from "@/lib/types";

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

  const filtered = useMemo(
    () => events.filter((e) => enabled[e.level] ?? true),
    [events, enabled],
  );

  useEffect(() => {
    if (!autoscroll) return;
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [filtered, autoscroll]);

  return (
    <div
      className="panel"
      style={{ display: "flex", flexDirection: "column", height: "100%" }}
    >
      <div
        className="panel-header"
        style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}
      >
        <span>log · {filtered.length}/{events.length}</span>
        <label style={{ fontSize: 10, display: "flex", gap: 4, alignItems: "center" }}>
          <input
            type="checkbox"
            checked={autoscroll}
            onChange={(e) => setAutoscroll(e.target.checked)}
          />
          autoscroll
        </label>
      </div>
      <div
        style={{
          padding: "6px 10px",
          borderBottom: "1px solid var(--rule)",
          display: "flex",
          flexWrap: "wrap",
          gap: 4,
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
              style={{ fontSize: 10, padding: "1px 6px" }}
            >
              {lvl}
            </button>
          ))}
      </div>
      <div ref={scrollRef} className="log-stream panel-body" style={{ flex: 1 }}>
        {filtered.map((ev) => (
          <div
            key={ev.id}
            className="log-line"
            data-level={ev.level}
            style={{ paddingBottom: 1 }}
          >
            <span style={{ color: "var(--muted)" }}>
              [{ev.stage}] {ev.progress !== null ? `${(ev.progress * 100).toFixed(0)}% ` : ""}
            </span>
            {ev.message}
          </div>
        ))}
        {filtered.length === 0 && (
          <div style={{ color: "var(--muted)" }}>no events</div>
        )}
      </div>
    </div>
  );
}
