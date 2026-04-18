"use client";

import { create } from "zustand";

export type RenderMode = "mesh" | "wireframe" | "points" | "points-color";

interface ViewerState {
  mode: RenderMode;
  showFrustums: boolean;
  /** Auto-computed base point size derived from the scene's bounding box. */
  autoPointSize: number;
  /** User slider multiplier relative to autoPointSize (0.1 .. 5.0). */
  pointSizeScale: number;
  /** Final applied size = autoPointSize * pointSizeScale. */
  pointSize: number;
  confPercentile: number;
  selection: Set<number>;
  lassoActive: boolean;
  activeMeshName: string | null;
  refitSignal: number; // incremented to request a camera re-fit
  setMode: (m: RenderMode) => void;
  setShowFrustums: (v: boolean) => void;
  setAutoPointSize: (v: number) => void;
  setPointSizeScale: (v: number) => void;
  setConfPercentile: (v: number) => void;
  setSelection: (s: Set<number>) => void;
  clearSelection: () => void;
  setLassoActive: (v: boolean) => void;
  setActiveMeshName: (name: string | null) => void;
  requestRefit: () => void;
}

export const useViewerStore = create<ViewerState>((set) => ({
  mode: "points-color",
  showFrustums: true,
  autoPointSize: 0.01,
  pointSizeScale: 1,
  pointSize: 0.01,
  confPercentile: 50,
  selection: new Set(),
  lassoActive: false,
  activeMeshName: null,
  refitSignal: 0,
  setMode: (mode) => set({ mode }),
  setShowFrustums: (showFrustums) => set({ showFrustums }),
  setAutoPointSize: (v) =>
    set((s) => ({ autoPointSize: v, pointSize: v * s.pointSizeScale })),
  setPointSizeScale: (v) =>
    set((s) => ({ pointSizeScale: v, pointSize: s.autoPointSize * v })),
  setConfPercentile: (confPercentile) => set({ confPercentile }),
  setSelection: (selection) => set({ selection }),
  clearSelection: () => set({ selection: new Set() }),
  setLassoActive: (lassoActive) => set({ lassoActive }),
  setActiveMeshName: (activeMeshName) => set({ activeMeshName }),
  requestRefit: () => set((s) => ({ refitSignal: s.refitSignal + 1 })),
}));
