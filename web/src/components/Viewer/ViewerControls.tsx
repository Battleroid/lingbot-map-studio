"use client";

import { Tip } from "@/components/Tip";
import { type RenderMode, useViewerStore } from "@/lib/viewerStore";

const MODES: { value: RenderMode; label: string; tip: string }[] = [
  { value: "mesh", label: "mesh", tip: "Loaded GLB rendered as solid triangles." },
  { value: "wireframe", label: "wire", tip: "Same GLB rendered with wireframe-only materials." },
  {
    value: "points",
    label: "points",
    tip: "PLY point cloud in a single foreground color. Fastest mode.",
  },
  {
    value: "points-color",
    label: "color pts",
    tip: "PLY point cloud with per-vertex colors from the source frames.",
  },
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
        gap: 8,
        padding: "6px 12px",
        borderBottom: "1px solid var(--rule)",
        flexWrap: "wrap",
        background: "var(--bg)",
      }}
    >
      <div style={{ display: "flex", gap: 2 }}>
        {MODES.map((m) => (
          <Tip key={m.value} text={m.tip} showIcon={false}>
            <button
              type="button"
              data-pressed={mode === m.value}
              onClick={() => setMode(m.value)}
            >
              {m.label}
            </button>
          </Tip>
        ))}
      </div>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 6,
          fontSize: "var(--fs-xs)",
        }}
      >
        <Tip text="Size of rendered points when a point-cloud mode is active. Lower = crisper, higher = filling gaps.">
          <span style={{ color: "var(--muted)" }}>pt size</span>
        </Tip>
        <input
          type="range"
          min={0.001}
          max={0.05}
          step={0.001}
          value={pointSize}
          onChange={(e) => setPointSize(Number(e.target.value))}
        />
      </div>
      <div
        style={{
          marginLeft: "auto",
          display: "flex",
          gap: 8,
          alignItems: "center",
        }}
      >
        <Tip text="Hold and drag to draw a polygon. Faces whose centroids land inside are selected for mesh-tool operations.">
          <button
            type="button"
            data-pressed={lassoActive}
            onClick={() => setLassoActive(!lassoActive)}
          >
            lasso{lassoActive ? " (on)" : ""}
          </button>
        </Tip>
        <span className="mono-small">{selection.size} selected</span>
        {selection.size > 0 && (
          <button type="button" onClick={clearSelection}>
            clear
          </button>
        )}
      </div>
    </div>
  );
}
