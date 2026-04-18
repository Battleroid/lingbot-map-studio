"use client";

import { type RenderMode, useViewerStore } from "@/lib/viewerStore";

const MODES: { value: RenderMode; label: string }[] = [
  { value: "mesh", label: "mesh" },
  { value: "wireframe", label: "wire" },
  { value: "points", label: "points" },
  { value: "points-color", label: "color pts" },
];

export function ViewerControls() {
  const mode = useViewerStore((s) => s.mode);
  const setMode = useViewerStore((s) => s.setMode);
  const pointSize = useViewerStore((s) => s.pointSize);
  const setPointSize = useViewerStore((s) => s.setPointSize);
  const lassoActive = useViewerStore((s) => s.lassoActive);
  const setLassoActive = useViewerStore((s) => s.setLassoActive);
  const selection = useViewerStore((s) => s.selection);
  const clearSelection = useViewerStore((s) => s.clearSelection);

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 6,
        padding: "6px 10px",
        borderBottom: "1px solid var(--rule)",
        flexWrap: "wrap",
      }}
    >
      <div style={{ display: "flex", gap: 2 }}>
        {MODES.map((m) => (
          <button
            key={m.value}
            type="button"
            data-pressed={mode === m.value}
            onClick={() => setMode(m.value)}
          >
            {m.label}
          </button>
        ))}
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 11 }}>
        <span style={{ color: "var(--muted)" }}>pt size</span>
        <input
          type="range"
          min={0.001}
          max={0.05}
          step={0.001}
          value={pointSize}
          onChange={(e) => setPointSize(Number(e.target.value))}
        />
      </div>
      <div style={{ marginLeft: "auto", display: "flex", gap: 6, alignItems: "center" }}>
        <button
          type="button"
          data-pressed={lassoActive}
          onClick={() => setLassoActive(!lassoActive)}
        >
          lasso{lassoActive ? " (on)" : ""}
        </button>
        <span style={{ fontSize: 11, color: "var(--muted)" }}>
          {selection.size} selected
        </span>
        {selection.size > 0 && (
          <button type="button" onClick={clearSelection}>
            clear
          </button>
        )}
      </div>
    </div>
  );
}
