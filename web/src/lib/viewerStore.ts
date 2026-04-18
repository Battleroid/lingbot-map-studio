"use client";

import { create } from "zustand";

export type RenderMode = "mesh" | "wireframe" | "points" | "points-color";
export type CameraMode = "orbit" | "fly";

export interface MeshRevision {
  name: string;
  revision: number;
  op?: string;
}

interface ViewerState {
  mode: RenderMode;
  cameraMode: CameraMode;
  /** Mesh-edit undo stack. Does NOT include the base reconstruction. */
  meshHistory: MeshRevision[];
  /** Pointer into meshHistory. -1 = showing base reconstruction (no edits). */
  meshHistoryIndex: number;
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
  setCameraMode: (m: CameraMode) => void;
  /** Called after a successful mesh-edit response. Branches history if the
   *  user had undone past some edits (classic undo stack behavior). */
  pushRevision: (rev: MeshRevision) => void;
  undo: () => void;
  redo: () => void;
  resetHistory: () => void;
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
  cameraMode: "orbit",
  meshHistory: [],
  meshHistoryIndex: -1,
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
  setCameraMode: (cameraMode) => set({ cameraMode }),
  pushRevision: (rev) =>
    set((s) => {
      // If we're mid-undo (index isn't at the tail), drop everything after
      // the current pointer — new edits replace the "redo" branch.
      const kept = s.meshHistory.slice(0, s.meshHistoryIndex + 1);
      const nextHistory = [...kept, rev];
      return {
        meshHistory: nextHistory,
        meshHistoryIndex: nextHistory.length - 1,
      };
    }),
  undo: () =>
    set((s) => ({
      meshHistoryIndex: Math.max(-1, s.meshHistoryIndex - 1),
    })),
  redo: () =>
    set((s) => ({
      meshHistoryIndex: Math.min(
        s.meshHistory.length - 1,
        s.meshHistoryIndex + 1,
      ),
    })),
  resetHistory: () => set({ meshHistory: [], meshHistoryIndex: -1 }),
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
