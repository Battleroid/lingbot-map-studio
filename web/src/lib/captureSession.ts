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

export interface CaptureLogEntry {
  /** Wall-clock ms since session start; relative timestamps are more
   *  useful than absolute on a phone where the user may not have set
   *  the timezone correctly. */
  t: number;
  level: "info" | "warn" | "error";
  msg: string;
}

/** How many log entries to keep before evicting from the head. Caps
 *  the worst-case memory of an all-day capture session at a few KB. */
const LOG_CAPACITY = 200;

/** Resolution of the view-direction coverage grid. 10° per bucket on
 *  both axes — coarse enough that a typical pan gesture lights up
 *  several cells but fine enough that "I missed that corner" is
 *  visible on the strip. Total cells: 36 az × 18 el = 648. */
export const VIEW_COVERAGE_AZ_BUCKETS = 36;
export const VIEW_COVERAGE_EL_BUCKETS = 18;
const VIEW_COVERAGE_CELLS =
  VIEW_COVERAGE_AZ_BUCKETS * VIEW_COVERAGE_EL_BUCKETS;

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

  // View-direction coverage (Scaniverse-style "have I pointed there
  // yet?" feedback). Two flat Uint16Arrays sized `36 × 18` — one
  // azimuth bucket per 10° (0..360°) crossed with one elevation
  // bucket per 10° (-90..90°). Each cell holds the number of poses
  // that landed in that direction. Stored as flat typed arrays
  // rather than Map<string,number> so the camera-view compass strips
  // can iterate without per-frame allocation.
  viewCoverage: Uint16Array;
  // Index of the current az/el bucket (the cursor on the strips).
  // -1 means "no pose yet", which the UI renders as a hidden cursor.
  viewCoverageCurrentAz: number;
  viewCoverageCurrentEl: number;

  stats: CaptureStats;
  error: string | null;

  // Client-side count of frame-send *attempts*. Incremented on every
  // sendFrame call, regardless of whether the WS actually flushed the
  // bytes. Useful as a "your phone is doing something" signal that's
  // independent of the server-emitted `stats.frames` (which only
  // updates after the server processes a frame). When framesSent
  // climbs but stats.frames stays at zero we know the WS-to-server
  // pipe is the broken bit, not the camera capture.
  framesSent: number;
  // Same idea for client-side drops: we increment when sendFrame
  // bails because the WS buffer is too full, so the user can see if
  // their connection is the bottleneck.
  framesDroppedClient: number;

  // Local view of the camera element's readiness. Pushed in by
  // `CameraStream` on each frame-grab tick. Surfaced on the chip so
  // "phone not sending" can be diagnosed as either "video element
  // never reached HAVE_CURRENT_DATA" (videoReady=false) or "WS won't
  // accept the frame" (sent=0 with videoReady=true).
  videoReady: boolean;
  videoSize: [number, number] | null;

  // Append-only client-side log of capture-flow events. Bounded so a
  // long session doesn't pin unbounded memory on a phone. The capture
  // page exposes a "show log / copy" affordance that ships these
  // entries to the clipboard so a stuck-capture report from a mobile
  // user is shareable without needing to ssh into the studio.
  log: CaptureLogEntry[];

  // Reconnect bookkeeping. `reconnectAttempt` is 0 while connected /
  // idle and >0 while a retry is scheduled or in flight (used in the
  // UI to show "reconnecting…" copy + a counter). `reconnectTimer`
  // holds the pending setTimeout id so user-initiated stop can cancel
  // a pending retry. `userInitiatedClose` distinguishes "I tapped
  // stop" (don't reconnect) from "the network blipped" (do reconnect).
  reconnectAttempt: number;
  reconnectTimer: ReturnType<typeof setTimeout> | null;
  userInitiatedClose: boolean;

  // Actions
  open: (sessionId: string) => void;
  close: () => void;
  reset: () => void;
  setVoxelSize: (size: number) => void;
  sendFrame: (blob: Blob) => void;
  setVideoState: (ready: boolean, size: [number, number] | null) => void;
  pushLog: (level: CaptureLogEntry["level"], msg: string) => void;
  clearLog: () => void;
}

const INITIAL_POINT_CAPACITY = 4096;

// Reconnect backoff schedule. Doubles each attempt: 1, 2, 4, 8, 16 s
// for a total of ~31 s of retries before we give up. That window
// matches the server's 60 s idle timeout (capture_session.py:54), so
// a session staying alive across a reconnect is the common case rather
// than the exception.
const RECONNECT_DELAYS_MS = [1_000, 2_000, 4_000, 8_000, 16_000];

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
  viewCoverage: new Uint16Array(VIEW_COVERAGE_CELLS),
  viewCoverageCurrentAz: -1,
  viewCoverageCurrentEl: -1,
  stats: { frames: 0, queued: 0, dropped: 0 },
  error: null,
  framesSent: 0,
  framesDroppedClient: 0,
  videoReady: false,
  videoSize: null,
  log: [],
  reconnectAttempt: 0,
  reconnectTimer: null,
  userInitiatedClose: false,

  open: (sessionId: string) => {
    // Fresh connect attempt. Clear any retry bookkeeping from a prior
    // session that ended without an explicit reset.
    const { reconnectTimer } = get();
    if (reconnectTimer !== null) clearTimeout(reconnectTimer);
    sessionT0 = Date.now();
    set({
      sessionId,
      error: null,
      reconnectAttempt: 0,
      reconnectTimer: null,
      userInitiatedClose: false,
      log: [],
    });
    get().pushLog("info", `open session ${sessionId}`);
    connectWs(sessionId, set, get);
  },

  close: () => {
    const { ws, reconnectTimer } = get();
    // Mark this as user-initiated *before* closing the socket so the
    // close handler doesn't kick off a reconnect.
    set({ userInitiatedClose: true });
    if (reconnectTimer !== null) clearTimeout(reconnectTimer);
    try {
      ws?.close();
    } catch {
      /* noop */
    }
    set({ ws: null, status: "closed", reconnectTimer: null });
  },

  reset: () => {
    const { ws, reconnectTimer } = get();
    set({ userInitiatedClose: true });
    if (reconnectTimer !== null) clearTimeout(reconnectTimer);
    try {
      ws?.close();
    } catch {
      /* noop */
    }
    // Restart the relative-timestamp clock so a second capture in
    // the same tab doesn't show "minutes since the page loaded" on
    // its first log entry.
    sessionT0 = Date.now();
    set({
      status: "idle",
      sessionId: null,
      ws: null,
      poses: [],
      pointsXyz: makeXyz(INITIAL_POINT_CAPACITY),
      pointsRgb: makeRgb(INITIAL_POINT_CAPACITY),
      pointsCount: 0,
      voxels: new Map(),
      viewCoverage: new Uint16Array(VIEW_COVERAGE_CELLS),
      viewCoverageCurrentAz: -1,
      viewCoverageCurrentEl: -1,
      stats: { frames: 0, queued: 0, dropped: 0 },
      error: null,
      framesSent: 0,
      framesDroppedClient: 0,
      videoReady: false,
      videoSize: null,
      log: [],
      reconnectAttempt: 0,
      reconnectTimer: null,
      userInitiatedClose: false,
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
    if (ws.bufferedAmount > 256 * 1024) {
      set((s) => ({ framesDroppedClient: s.framesDroppedClient + 1 }));
      return;
    }
    void blob.arrayBuffer().then((buf) => {
      try {
        ws.send(buf);
        set((s) => ({ framesSent: s.framesSent + 1 }));
        // Log the first frame we successfully push so the timeline
        // has a clear "client started sending" marker.
        const cur = get();
        if (cur.framesSent === 1) {
          cur.pushLog(
            "info",
            `first frame sent (${(buf as ArrayBuffer).byteLength} bytes)`,
          );
        }
      } catch (exc) {
        get().pushLog(
          "warn",
          `ws.send threw: ${(exc as Error).message ?? exc}`,
        );
      }
    });
  },

  setVideoState: (ready: boolean, size: [number, number] | null) => {
    // Only write through when something actually changed — guards
    // against a per-tick `setState` no-op storm from the camera
    // grab loop.
    const cur = get();
    const sizeChanged =
      (cur.videoSize === null) !== (size === null) ||
      (cur.videoSize !== null &&
        size !== null &&
        (cur.videoSize[0] !== size[0] || cur.videoSize[1] !== size[1]));
    if (cur.videoReady === ready && !sizeChanged) return;
    set({ videoReady: ready, videoSize: size });
    if (sizeChanged && size !== null) {
      get().pushLog("info", `video ready ${size[0]}x${size[1]}`);
    }
  },

  pushLog: (level, msg) => {
    set((s) => {
      const entry: CaptureLogEntry = {
        t: Date.now() - sessionT0,
        level,
        msg,
      };
      const next = s.log.concat(entry);
      // Bounded ring — drop the oldest if we'd overflow the cap.
      if (next.length > LOG_CAPACITY) {
        next.splice(0, next.length - LOG_CAPACITY);
      }
      return { log: next };
    });
  },

  clearLog: () => set({ log: [] }),
}));

// Session-relative t=0. Set on module load so the first log entry
// has a sensible offset; reset() also resets this so a second
// capture in the same tab starts back at 0.
let sessionT0 = Date.now();

/** Build a fresh WebSocket for `sessionId` and wire its handlers to
 *  the store. Called from `open()` (first connect) and from the retry
 *  timer (subsequent attempts) — both paths share this code so the
 *  message-decoding logic only lives in one place. */
function connectWs(
  sessionId: string,
  set: (
    partial:
      | Partial<CaptureState>
      | ((s: CaptureState) => Partial<CaptureState>),
  ) => void,
  get: () => CaptureState,
): void {
  const ws = new WebSocket(captureWsUrl(sessionId));
  ws.binaryType = "arraybuffer";
  set({ status: "connecting", ws });

  ws.addEventListener("open", () => {
    // First-attempt or reconnect — either way we're live again, so
    // surface "open" + clear the retry counter so the UI stops
    // showing "reconnecting (n/N)".
    set({ status: "open", reconnectAttempt: 0, error: null });
    get().pushLog("info", "ws open");
  });

  // We schedule the retry from `close` rather than `error` because a
  // remote drop fires `close` after `error`, and reacting to `error`
  // alone would race with the "is this user-initiated?" check below.
  ws.addEventListener("error", () => {
    get().pushLog("warn", "ws error event");
  });

  ws.addEventListener("close", (evt) => {
    const state = get();
    // User pressed Stop / navigated away — honour that.
    if (state.userInitiatedClose) {
      set({ ws: null, status: "closed" });
      get().pushLog(
        "info",
        `ws closed by user (code=${evt.code} reason=${evt.reason || "-"})`,
      );
      return;
    }
    // Out of retries → surface the failure and stop trying.
    if (state.reconnectAttempt >= RECONNECT_DELAYS_MS.length) {
      set({
        ws: null,
        status: "closed",
        error: `lost connection after ${RECONNECT_DELAYS_MS.length} retries`,
      });
      get().pushLog(
        "error",
        `ws gave up after ${RECONNECT_DELAYS_MS.length} retries (code=${evt.code})`,
      );
      return;
    }
    // Schedule the next attempt.
    const delay = RECONNECT_DELAYS_MS[state.reconnectAttempt];
    const timer = setTimeout(() => {
      // The user may have stopped during the wait — bail out cleanly.
      const cur = get();
      if (cur.userInitiatedClose) return;
      connectWs(sessionId, set, get);
    }, delay);
    set({
      ws: null,
      // Status stays "connecting" for the duration of the backoff so
      // the UI can show "reconnecting (n/5) …" without a status flap.
      status: "connecting",
      reconnectAttempt: state.reconnectAttempt + 1,
      reconnectTimer: timer,
    });
    get().pushLog(
      "warn",
      `ws closed (code=${evt.code}); retrying in ${delay}ms (attempt ${state.reconnectAttempt + 1}/${RECONNECT_DELAYS_MS.length})`,
    );
  });

  ws.addEventListener("message", (evt) => {
    try {
      const msg = JSON.parse(evt.data as string) as {
        type: string;
        data: { [k: string]: unknown };
      };
      if (msg.type === "pose") {
        const pose = msg.data as unknown as CapturePose;
        set((s) => ({ poses: [...s.poses, pose] }));
        markViewCoverageFromPose(pose, get, set);
      } else if (msg.type === "points") {
        const rows = (msg.data["new"] as number[][]) || [];
        appendPoints(rows, get, set);
      } else if (msg.type === "stats") {
        const next = msg.data as unknown as CaptureStats;
        const prev = get().stats;
        set({ stats: next });
        // Log the first stats event explicitly — that's the moment
        // the server confirms it has decoded a frame, the single
        // most-useful signal in a stuck-capture report.
        if (prev.frames === 0 && next.frames > 0) {
          get().pushLog(
            "info",
            `first server-decoded frame (server processed=${next.frames})`,
          );
        }
      } else if (msg.type === "ready") {
        get().pushLog(
          "info",
          `server ready (backend=${msg.data["backend"] ?? "?"})`,
        );
      } else if (msg.type === "error") {
        const m = String(msg.data["message"] ?? "capture error");
        set({ error: m });
        get().pushLog("error", `server: ${m}`);
      }
    } catch (exc) {
      get().pushLog("warn", `malformed ws message: ${(exc as Error).message}`);
    }
  });
}

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
  // Coverage buckets are now driven by *observed points*, not by
  // poses (where the camera looked). Stripes peel back only when the
  // SLAM tracker actually produced geometry in that direction —
  // panning at a feature-poor wall and finding nothing leaves the
  // stripes intact, which is the Scaniverse "real-space capture"
  // cue the user asked for.
  const cov = new Uint16Array(state.viewCoverage);
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
    // Project the point onto the unit sphere centred at the world
    // origin (≈ first pose's position; trajectory hasn't drifted
    // far this early in a scan). az = atan2(x, -z) so 0° = forward,
    // matching the shader's pixel-to-direction math; el = asin(y).
    const len = Math.hypot(x, y, z) || 1;
    const nx = x / len,
      ny = y / len,
      nz = z / len;
    const azDeg = (Math.atan2(nx, -nz) * 180) / Math.PI;
    const elDeg = (Math.asin(Math.max(-1, Math.min(1, ny))) * 180) / Math.PI;
    const az = ((azDeg % 360) + 360) % 360;
    const azIdx = Math.min(
      VIEW_COVERAGE_AZ_BUCKETS - 1,
      Math.floor((az / 360) * VIEW_COVERAGE_AZ_BUCKETS),
    );
    const elIdx = Math.min(
      VIEW_COVERAGE_EL_BUCKETS - 1,
      Math.max(
        0,
        Math.floor(((elDeg + 90) / 180) * VIEW_COVERAGE_EL_BUCKETS),
      ),
    );
    const flat = elIdx * VIEW_COVERAGE_AZ_BUCKETS + azIdx;
    if (cov[flat] < 0xffff) cov[flat] = cov[flat] + 1;
  }
  set({
    pointsXyz: xyz,
    pointsRgb: rgb,
    pointsCount: off,
    voxels,
    viewCoverage: cov,
  });
}

/** Number of reconnect attempts the store will make before giving up.
 *  Exposed for the UI to render `reconnecting (n/N)…`. */
export const MAX_RECONNECT_ATTEMPTS = RECONNECT_DELAYS_MS.length;

/** Track the current camera-look direction as az/el bucket indices
 *  so the (now diagnostic) cursor on the strip overlays still
 *  follows the user's pan. Coverage *count* bumping has moved to
 *  `appendPoints` — a bucket only flips to "covered" when the SLAM
 *  tracker actually produced geometry there, not just because the
 *  camera looked. The user wants stripes that reflect "was anything
 *  observed in this direction?", not "did the camera point here?". */
function markViewCoverageFromPose(
  pose: CapturePose,
  _get: () => CaptureState,
  set: (
    partial:
      | Partial<CaptureState>
      | ((s: CaptureState) => Partial<CaptureState>),
  ) => void,
): void {
  const [qx, qy, qz, qw] = pose.q;
  const fx = -2 * (qx * qz + qw * qy);
  const fy = -2 * (qy * qz - qw * qx);
  const fz = -(1 - 2 * (qx * qx + qy * qy));
  const len = Math.hypot(fx, fy, fz) || 1;
  const nx = fx / len;
  const ny = fy / len;
  const nz = fz / len;
  const azDeg = (Math.atan2(nx, -nz) * 180) / Math.PI;
  const elDeg = (Math.asin(Math.max(-1, Math.min(1, ny))) * 180) / Math.PI;
  const az = ((azDeg % 360) + 360) % 360;
  const azIdx = Math.min(
    VIEW_COVERAGE_AZ_BUCKETS - 1,
    Math.floor((az / 360) * VIEW_COVERAGE_AZ_BUCKETS),
  );
  const elIdx = Math.min(
    VIEW_COVERAGE_EL_BUCKETS - 1,
    Math.max(
      0,
      Math.floor(((elDeg + 90) / 180) * VIEW_COVERAGE_EL_BUCKETS),
    ),
  );
  set({
    viewCoverageCurrentAz: azIdx,
    viewCoverageCurrentEl: elIdx,
  });
}
