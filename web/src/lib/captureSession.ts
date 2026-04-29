"use client";

import { create } from "zustand";

import { captureWsUrl } from "@/lib/api";

/**
 * Live camera-capture client state.
 *
 * Holds the WS lifecycle, a streaming sparse cloud (positions + colors
 * as flat Float32Arrays for direct upload to a BufferGeometry), the
 * pose trajectory the SLAM tracker emits, a voxel coverage map for the
 * Scaniverse-style "filled in / empty" pattern, and stats from the
 * server.
 *
 * Uses a Zustand store rather than React state because the data
 * updates 10+ Hz and we need O(1) appends without re-rendering the
 * entire capture page on every frame. The renderer subscribes to
 * specific slices.
 */

export type CaptureStatus = "idle" | "connecting" | "open" | "closed";

export interface CapturePose {
  frame: number;
  t: [number, number, number];
  q: [number, number, number, number];
}

export interface CaptureStats {
  frames: number;
  queued: number;
  dropped: number;
}

interface CaptureState {
  status: CaptureStatus;
  sessionId: string | null;
  ws: WebSocket | null;

  poses: CapturePose[];
  // Sparse cloud — flat Float32Arrays. Capacity grows by powers of two.
  pointsXyz: Float32Array;
  pointsRgb: Float32Array;
  pointsCount: number;

  // Voxel coverage. Keyed by `floor(x/v),floor(y/v),floor(z/v)`; value
  // is the cumulative count of points landed in that voxel.
  voxelSize: number;
  voxels: Map<string, number>;

  stats: CaptureStats;
  error: string | null;

  // Actions
  open: (sessionId: string) => void;
  close: () => void;
  reset: () => void;
  setVoxelSize: (size: number) => void;
  sendFrame: (blob: Blob) => void;
}

const INITIAL_POINT_CAPACITY = 4096;

const makeXyz = (cap: number) => new Float32Array(cap * 3);
const makeRgb = (cap: number) => new Float32Array(cap * 3);

export const useCaptureStore = create<CaptureState>((set, get) => ({
  status: "idle",
  sessionId: null,
  ws: null,
  poses: [],
  pointsXyz: makeXyz(INITIAL_POINT_CAPACITY),
  pointsRgb: makeRgb(INITIAL_POINT_CAPACITY),
  pointsCount: 0,
  voxelSize: 0.1,
  voxels: new Map(),
  stats: { frames: 0, queued: 0, dropped: 0 },
  error: null,

  open: (sessionId: string) => {
    const ws = new WebSocket(captureWsUrl(sessionId));
    ws.binaryType = "arraybuffer";
    set({ status: "connecting", sessionId, error: null, ws });

    ws.addEventListener("open", () => set({ status: "open" }));
    ws.addEventListener("close", () => set({ status: "closed" }));
    ws.addEventListener("error", () =>
      set({ status: "closed", error: "websocket error" }),
    );
    ws.addEventListener("message", (evt) => {
      try {
        const msg = JSON.parse(evt.data as string) as {
          type: string;
          data: { [k: string]: unknown };
        };
        if (msg.type === "pose") {
          const pose = msg.data as unknown as CapturePose;
          set((s) => ({ poses: [...s.poses, pose] }));
        } else if (msg.type === "points") {
          const rows = (msg.data["new"] as number[][]) || [];
          appendPoints(rows, get, set);
        } else if (msg.type === "stats") {
          set({ stats: msg.data as unknown as CaptureStats });
        } else if (msg.type === "error") {
          set({ error: String(msg.data["message"] ?? "capture error") });
        }
      } catch {
        /* drop malformed */
      }
    });
  },

  close: () => {
    const { ws } = get();
    try {
      ws?.close();
    } catch {
      /* noop */
    }
    set({ ws: null, status: "closed" });
  },

  reset: () => {
    const { ws } = get();
    try {
      ws?.close();
    } catch {
      /* noop */
    }
    set({
      status: "idle",
      sessionId: null,
      ws: null,
      poses: [],
      pointsXyz: makeXyz(INITIAL_POINT_CAPACITY),
      pointsRgb: makeRgb(INITIAL_POINT_CAPACITY),
      pointsCount: 0,
      voxels: new Map(),
      stats: { frames: 0, queued: 0, dropped: 0 },
      error: null,
    });
  },

  setVoxelSize: (size: number) => {
    const v = Math.max(0.01, Math.min(1.0, size));
    // Re-voxelize the existing cloud at the new resolution so the
    // coverage map stays consistent if the user tweaks mid-capture.
    const { pointsXyz, pointsCount } = get();
    const voxels = new Map<string, number>();
    for (let i = 0; i < pointsCount; i++) {
      const x = pointsXyz[i * 3];
      const y = pointsXyz[i * 3 + 1];
      const z = pointsXyz[i * 3 + 2];
      const key = `${Math.floor(x / v)},${Math.floor(y / v)},${Math.floor(z / v)}`;
      voxels.set(key, (voxels.get(key) ?? 0) + 1);
    }
    set({ voxelSize: v, voxels });
  },

  sendFrame: (blob: Blob) => {
    const { ws } = get();
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    // Drop if we're already 256 KB behind — client-side
    // backpressure so the server's SLAM step rate is the bottleneck,
    // not a queue of stale frames.
    if (ws.bufferedAmount > 256 * 1024) return;
    void blob.arrayBuffer().then((buf) => {
      try {
        ws.send(buf);
      } catch {
        /* drop on disconnect */
      }
    });
  },
}));

function appendPoints(
  rows: number[][],
  get: () => CaptureState,
  set: (
    partial:
      | Partial<CaptureState>
      | ((s: CaptureState) => Partial<CaptureState>),
  ) => void,
): void {
  if (rows.length === 0) return;
  const state = get();
  const need = state.pointsCount + rows.length;
  let cap = state.pointsXyz.length / 3;
  let xyz = state.pointsXyz;
  let rgb = state.pointsRgb;
  if (need > cap) {
    while (cap < need) cap *= 2;
    const nextXyz = makeXyz(cap);
    nextXyz.set(xyz);
    xyz = nextXyz;
    const nextRgb = makeRgb(cap);
    nextRgb.set(rgb);
    rgb = nextRgb;
  }
  let off = state.pointsCount;
  const voxels = new Map(state.voxels);
  const v = state.voxelSize;
  for (const row of rows) {
    const x = row[0],
      y = row[1],
      z = row[2];
    xyz[off * 3] = x;
    xyz[off * 3 + 1] = y;
    xyz[off * 3 + 2] = z;
    rgb[off * 3] = (row[3] ?? 200) / 255;
    rgb[off * 3 + 1] = (row[4] ?? 200) / 255;
    rgb[off * 3 + 2] = (row[5] ?? 200) / 255;
    off += 1;
    const key = `${Math.floor(x / v)},${Math.floor(y / v)},${Math.floor(z / v)}`;
    voxels.set(key, (voxels.get(key) ?? 0) + 1);
  }
  set({
    pointsXyz: xyz,
    pointsRgb: rgb,
    pointsCount: off,
    voxels,
  });
}
