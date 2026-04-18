"use client";

import { create } from "zustand";

export type RenderMode = "mesh" | "wireframe" | "points" | "points-color";

interface ViewerState {
  mode: RenderMode;
  showFrustums: boolean;
  pointSize: number;
  pointSizeAuto: boolean; // true = auto-computed from geometry bbox on load
  confPercentile: number;
  selection: Set<number>;
  lassoActive: boolean;
  activeMeshName: string | null;
  setMode: (m: RenderMode) => void;
  setShowFrustums: (v: boolean) => void;
  setPointSize: (v: number) => void;
  setPointSizeFromAuto: (v: number) => void;
  setConfPercentile: (v: number) => void;
  setSelection: (s: Set<number>) => void;
  clearSelection: () => void;
  setLassoActive: (v: boolean) => void;
  setActiveMeshName: (name: string | null) => void;
}

export const useViewerStore = create<ViewerState>((set) => ({
  mode: "points-color",
  showFrustums: true,
  pointSize: 0.012,
  pointSizeAuto: true,
  confPercentile: 50,
  selection: new Set(),
  lassoActive: false,
  activeMeshName: null,
  setMode: (mode) => set({ mode }),
  setShowFrustums: (showFrustums) => set({ showFrustums }),
  // User-driven change via slider: lock auto off so we don't overwrite their choice.
  setPointSize: (pointSize) => set({ pointSize, pointSizeAuto: false }),
  // Auto-computed change from the loader: only applied while auto mode is on.
  setPointSizeFromAuto: (pointSize) => set({ pointSize }),
  setConfPercentile: (confPercentile) => set({ confPercentile }),
  setSelection: (selection) => set({ selection }),
  clearSelection: () => set({ selection: new Set() }),
  setLassoActive: (lassoActive) => set({ lassoActive }),
  setActiveMeshName: (activeMeshName) => set({ activeMeshName }),
}));
