"use client";

import { Canvas } from "@react-three/fiber";
import { Bounds, OrbitControls } from "@react-three/drei";
import { useRouter, useSearchParams } from "next/navigation";
import {
  Suspense,
  useEffect,
  useMemo,
  useState,
  type CSSProperties,
} from "react";
import * as THREE from "three";

import { CameraOverlayAR } from "@/components/capture/CameraOverlayAR";
import { CameraStream } from "@/components/capture/CameraStream";
import {
  CoverageVoxels,
  useCoverageSummary,
} from "@/components/capture/CoverageVoxels";
import { FollowPoseCamera } from "@/components/capture/FollowPoseCamera";
import {
  startCaptureSession,
  stopCaptureSession,
  type CaptureBackend,
} from "@/lib/api";
import { useCaptureStore } from "@/lib/captureSession";

/** Type-guard for the `?backend=` query param. Anything outside the
 *  dropdown's supported set falls back to the default; we don't want
 *  a stale link `/capture?backend=foo` to crash the page or push an
 *  invalid value into the API. */
function isCaptureBackend(s: string | null | undefined): s is CaptureBackend {
  return (
    s === "mast3r_slam" || s === "droid_slam" || s === "dpvo" || s === "monogs"
  );
}

const CAPTURE_BUTTON_STYLE: CSSProperties = {
  padding: "12px 24px",
  fontSize: "var(--fs-md)",
  minHeight: 48,
  background: "#fff",
  color: "#000",
  border: "1px solid #fff",
  borderRadius: 6,
  fontWeight: 600,
};

/**
 * Live camera capture page.
 *
 * Two responsive layouts switched by viewport width:
 *   - narrow / portrait: <video> fullscreen background + PiP <Canvas>
 *     in the top-left + control bar at the bottom.
 *   - wide / landscape: <video> on the left half + <Canvas> on the
 *     right half, controls layered onto the video.
 *
 * The page is a third entry point alongside the home-page upload
 * tile + the gsplat-from-source tile. The existing batch-upload
 * pipeline is unchanged; this just adds a streaming alternative.
 */
/** Public default export. Wraps the real component in `<Suspense>`
 *  so `useSearchParams()` is allowed under Next 14's static prerender
 *  rules — without the boundary the build bails with
 *  `useSearchParams() should be wrapped in a suspense boundary`. */
export default function CapturePage() {
  return (
    <Suspense fallback={null}>
      <CapturePageInner />
    </Suspense>
  );
}

function CapturePageInner() {
  const router = useRouter();
  // Pre-select the backend the user picked on the home page so they
  // don't have to re-pick it here. Validates against the supported
  // dropdown values; an unrecognised query falls back to the default.
  const searchParams = useSearchParams();
  const backendFromQuery = searchParams?.get("backend");
  const initialBackend: CaptureBackend = isCaptureBackend(backendFromQuery)
    ? backendFromQuery
    : "mast3r_slam";
  const [backend, setBackend] = useState<CaptureBackend>(initialBackend);
  const [deviceId, setDeviceId] = useState<string | undefined>(undefined);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [busy, setBusy] = useState<"start" | "stop" | null>(null);
  const [pageError, setPageError] = useState<string | null>(null);

  const open = useCaptureStore((s) => s.open);
  const reset = useCaptureStore((s) => s.reset);
  const status = useCaptureStore((s) => s.status);
  const sessionId = useCaptureStore((s) => s.sessionId);
  const stats = useCaptureStore((s) => s.stats);
  const pointsCount = useCaptureStore((s) => s.pointsCount);
  const sessionError = useCaptureStore((s) => s.error);
  const reconnectAttempt = useCaptureStore((s) => s.reconnectAttempt);
  const framesSent = useCaptureStore((s) => s.framesSent);
  const framesDroppedClient = useCaptureStore((s) => s.framesDroppedClient);
  const videoReady = useCaptureStore((s) => s.videoReady);
  const videoSize = useCaptureStore((s) => s.videoSize);
  const log = useCaptureStore((s) => s.log);

  // The log overlay is collapsed by default — most users never need
  // it. When open, the user can read or copy the timeline (plus a
  // header summary of the chip stats) to share when reporting a
  // stuck capture.
  const [showLog, setShowLog] = useState(false);
  const [copyState, setCopyState] = useState<"idle" | "ok" | "err">("idle");

  // PiP view mode: `follow` snaps the canvas camera to the latest
  // SLAM pose so the user sees their phone's orientation reflected
  // in the splat space; `orbit` is free OrbitControls navigation
  // (the previous default). Toggle is a tiny button on the PiP.
  const [pipMode, setPipMode] = useState<"follow" | "orbit">("follow");

  // "open" or actively retrying — both are flavours of an in-progress
  // session, so treat them as `capturing` for the purpose of which
  // button (Start vs Stop) is rendered. Without this a transient
  // network blip would flip the UI to "Start" mid-scan, which would
  // erroneously kick off a new session.
  const capturing = status === "open" || reconnectAttempt > 0;
  const summary = useCoverageSummary();

  // Detect insecure context client-side. `window.isSecureContext` is the
  // exact gate `getUserMedia` checks: localhost + HTTPS pass, plain
  // http://<lan-ip> doesn't. Without this check the user lands on a
  // black page that silently does nothing when they tap start (the
  // camera permission prompt never appears because `mediaDevices` is
  // undefined off-secure-context).
  //
  // `null` until mounted so SSR matches and we don't flash the wrong
  // screen during hydration.
  const [insecure, setInsecure] = useState<boolean | null>(null);
  useEffect(() => {
    // Defer past the current render tick so React doesn't treat this
    // as a synchronous setState-in-effect (cascading-renders lint).
    let cancelled = false;
    void Promise.resolve().then(() => {
      if (cancelled) return;
      const noSecureContext = !window.isSecureContext;
      const noMediaDevices = !navigator.mediaDevices?.getUserMedia;
      setInsecure(noSecureContext || noMediaDevices);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  // Probe the user's available cameras once on mount. enumerateDevices
  // returns deviceIds without labels until the user has granted at
  // least one getUserMedia permission, so this list is shorter on
  // first paint and fuller after Start was clicked once.
  useEffect(() => {
    let cancelled = false;
    async function probe() {
      if (!navigator.mediaDevices?.enumerateDevices) return;
      try {
        const all = await navigator.mediaDevices.enumerateDevices();
        if (cancelled) return;
        setDevices(all.filter((d) => d.kind === "videoinput"));
      } catch {
        /* ignore — capture flow still works with the default device */
      }
    }
    void probe();
    return () => {
      cancelled = true;
    };
  }, [status]);

  // Reset the store when leaving the page so a back-then-return
  // doesn't surface stale state.
  useEffect(() => () => reset(), [reset]);

  // Early-return *after* all hooks so React's hook-order rule is
  // satisfied across the insecure / secure transitions. (`null` keeps
  // the page-shell rendering during the first hydration tick — same as
  // the secure case — so there's no flash.)
  if (insecure === true) return <InsecureContextNotice />;

  async function start() {
    setBusy("start");
    setPageError(null);
    try {
      const res = await startCaptureSession(backend);
      open(res.session_id);
    } catch (e) {
      setPageError(String((e as Error).message));
    } finally {
      setBusy(null);
    }
  }

  async function stop() {
    if (!sessionId) return;
    setBusy("stop");
    try {
      const res = await stopCaptureSession(sessionId);
      reset();
      router.push(`/jobs/${res.job_id}`);
    } catch (e) {
      setPageError(String((e as Error).message));
      setBusy(null);
    }
  }

  return (
    <div
      className="capture-shell"
      style={{
        position: "relative",
        width: "100%",
        // 100dvh tracks the *currently visible* viewport. Mobile
        // browsers expose less than 100vh while the URL bar / bottom
        // chrome is showing, so a 100vh shell with bottom-anchored
        // controls would push the start button off-screen on first
        // load. dvh resolves to the dynamic value and swaps cleanly
        // when the user scrolls and the chrome retracts.
        height: "100dvh",
        // Honour the home-indicator inset on iOS so the controls
        // anchor above it instead of under it.
        paddingBottom: "env(safe-area-inset-bottom)",
        background: "#000",
        color: "#fff",
        overflow: "hidden",
      }}
    >
      {/* Background camera feed. Fills the viewport on portrait;
          shrinks to the left half on landscape via a CSS media
          query in capture.module-style inline class below. */}
      <div className="capture-video-pane">
        <CameraStream
          capturing={capturing}
          fps={10}
          deviceId={deviceId}
        />
        {/* Scaniverse-style 3D coverage overlay. A transparent
         *  three.js Canvas mounted over the video element with its
         *  camera locked to the live SLAM pose. Captured cells render
         *  with a world-space diagonal-stripe shader; an immediately-
         *  adjacent shell of uncovered cells renders translucent
         *  red. As the user pans into the red regions the SLAM
         *  tracker fills the bins and they flip to stripes — same
         *  panning-feedback loop as Scaniverse. Replaces the prior
         *  compass-strip approximation, which surfaced direction
         *  coverage as 1D bars and didn't carry the spatial cue. */}
        {capturing && <CameraOverlayAR />}
      </div>

      {/* Coverage canvas. PiP on portrait (top-left, 35% × 35%);
          right-half pane on landscape. */}
      <div className="capture-canvas-pane">
        <Canvas
          camera={{ position: [2, 2, 2], fov: 45, near: 0.001, far: 1000 }}
          style={{ background: "#0a0a0a" }}
        >
          <ambientLight intensity={0.7} />
          <directionalLight position={[3, 5, 2]} intensity={0.6} />
          {pipMode === "follow" ? (
            // Follow mode skips Bounds (which would auto-fit the
            // view to scene contents and override our pose-driven
            // camera every render). FollowPoseCamera writes
            // position+quaternion straight from the latest pose.
            <>
              <CoverageVoxels />
              <CapturePointCloud />
              <CaptureFrustum />
              <FollowPoseCamera />
            </>
          ) : (
            <Bounds margin={1.4} observe>
              <CoverageVoxels />
              <CapturePointCloud />
              <CaptureFrustum />
            </Bounds>
          )}
          <axesHelper args={[0.3]} />
          {pipMode === "orbit" && (
            <OrbitControls makeDefault enableDamping dampingFactor={0.08} />
          )}
        </Canvas>
        {/* PiP-corner toggle. Stays small + dark so it doesn't hide
         *  the canvas content; tap target is 32px which is fine for
         *  a non-primary control. */}
        <button
          type="button"
          onClick={() =>
            setPipMode((m) => (m === "follow" ? "orbit" : "follow"))
          }
          style={{
            position: "absolute",
            top: 4,
            right: 4,
            padding: "2px 6px",
            background: "rgba(10, 10, 10, 0.78)",
            color: "#fff",
            border: "1px solid rgba(255, 255, 255, 0.4)",
            borderRadius: 3,
            fontSize: "10px",
            letterSpacing: "0.04em",
            minHeight: 24,
          }}
          aria-pressed={pipMode === "follow"}
          title={
            pipMode === "follow"
              ? "PiP follows your phone's orientation; tap to orbit freely"
              : "PiP is free-orbit; tap to follow the live pose"
          }
        >
          {pipMode === "follow" ? "follow ●" : "orbit ○"}
        </button>
      </div>

      {/* Stats + controls overlay.
       *  Two frame counters intentionally: `frames sent` is how many
       *  the phone has *attempted* to push to the server (incremented
       *  client-side in captureSession.sendFrame), `processed` is how
       *  many the SLAM tracker has *consumed* (server-side stats). If
       *  the first one grows but the second stays at 0 the WS-to-
       *  server pipe is broken; if neither grows the camera capture
       *  itself is the issue. The split is what makes a stuck capture
       *  diagnosable from the phone instead of needing to ssh into
       *  the studio. */}
      <div
        style={{
          position: "absolute",
          // Anchored just above the start/stop button rather than the
          // top-left, where the PiP coverage canvas (35vw × 35vw) was
          // landing on top of it on phones. This keeps the chip in
          // the user's eyeline as they pan the camera + leaves the
          // PiP unobstructed for the splat preview.
          bottom:
            "max(88px, calc(env(safe-area-inset-bottom) + 88px))",
          left: 12,
          right: 12,
          padding: "6px 10px",
          background: "rgba(10, 10, 10, 0.78)",
          fontSize: "var(--fs-xs)",
          letterSpacing: "0.04em",
          borderRadius: "var(--r-xs)",
          textAlign: "center",
        }}
      >
        <div>
          ●&nbsp;{pointsCount.toLocaleString()} pts &middot;{" "}
          {Math.round(summary.ratio * 100)}% covered
        </div>
        {/* Three-tier diagnostic: video → ws → frames. A stuck capture
         *  has *one* of these stuck, so showing them all together
         *  makes the failure mode obvious from the phone:
         *    video: ✗ → camera permission denied / not playing
         *    ws: closed → caddy / cert / network issue
         *    sent 0 with ws open → client backpressure dropping frames
         *    sent N processed 0 → server can't decode
         *    sent N processed M → working, just slow */}
        <div style={{ opacity: 0.85 }}>
          video:{" "}
          {videoReady && videoSize
            ? `${videoSize[0]}×${videoSize[1]}`
            : videoReady
              ? "ready"
              : "✗"}{" "}
          · ws: {status}
          {reconnectAttempt > 0 ? ` (retry ${reconnectAttempt})` : ""} · sent{" "}
          {framesSent}
          {framesDroppedClient > 0 && ` (${framesDroppedClient} dropped)`} ·
          processed {stats.frames}
          {stats.dropped > 0 && ` (${stats.dropped} dropped)`}
        </div>
      </div>

      {/* The reconnecting state is visible inline on the diagnostic
       *  chip below as `ws: connecting · (retry N)`, which is enough
       *  signal in the new layout. The dedicated yellow badge that
       *  used to sit at the top-left got obscured by the PiP coverage
       *  canvas anyway. */}

      <div
        style={{
          position: "absolute",
          top: 12,
          right: 12,
          display: "flex",
          gap: 6,
          flexDirection: "column",
          alignItems: "flex-end",
        }}
      >
        <select
          disabled={capturing}
          value={backend}
          onChange={(e) => setBackend(e.target.value as CaptureBackend)}
          style={{ background: "#0a0a0a", color: "#fff" }}
        >
          <option value="mast3r_slam">mast3r-slam (default)</option>
          <option value="droid_slam">droid-slam</option>
          <option value="dpvo">dpvo</option>
          {/* MonoGS: the captured frames are queued for real CUDA
              MonoGS reconstruction in worker-gs after stop. The live
              session in this api process is just a pose-tracker for
              the AR coverage overlay; the splat itself is generated
              post-stop on the GPU worker. */}
          <option value="monogs">monogs (gaussian splat)</option>
        </select>
        {devices.length > 1 && (
          <select
            disabled={capturing}
            value={deviceId ?? ""}
            onChange={(e) => setDeviceId(e.target.value || undefined)}
            style={{ background: "#0a0a0a", color: "#fff", maxWidth: 220 }}
          >
            <option value="">auto · rear camera</option>
            {devices.map((d) => (
              <option key={d.deviceId} value={d.deviceId}>
                {d.label || `camera ${d.deviceId.slice(0, 8)}`}
              </option>
            ))}
          </select>
        )}
        {/* Log toggle. Sits next to the dropdowns so it's reachable
         *  whether the user is mid-capture or just inspecting. The
         *  badge shows the current entry count so a stuck session
         *  reports something helpful even before the user opens it. */}
        <button
          type="button"
          onClick={() => setShowLog((v) => !v)}
          style={{
            background: "#0a0a0a",
            color: "#fff",
            border: "1px solid rgba(255, 255, 255, 0.4)",
            borderRadius: "var(--r-xs)",
            padding: "4px 8px",
            fontSize: "var(--fs-xs)",
            minHeight: 32,
          }}
        >
          {showLog ? "hide log" : `log (${log.length})`}
        </button>
      </div>

      <div
        style={{
          position: "absolute",
          // Anchor above the iOS home indicator + a 24px breathing
          // gap. `max()` keeps a sensible minimum on phones with no
          // inset (Android, desktop), where env(...) resolves to 0.
          bottom: "max(24px, calc(env(safe-area-inset-bottom) + 24px))",
          left: 0,
          right: 0,
          display: "flex",
          justifyContent: "center",
          gap: 12,
          // Bump tap target on mobile — the previous "8px 20px" button
          // was a 32px-tall pill, well below Apple's 44pt / Material's
          // 48dp accessibility guidance for primary actions.
          padding: "0 16px",
        }}
      >
        {/* Explicit background + color on the buttons. The page shell
         *  inherits color: white from the dark surround, and mobile
         *  browsers default <button> background to a near-white, which
         *  was rendering the start-capture button as white-on-white
         *  (text invisible) on Android Chrome. Pinning a high-contrast
         *  black-on-white CTA fixes the readability and side-steps any
         *  global button styles in globals.css that don't account for
         *  this inverted shell. */}
        {!capturing ? (
          <button
            type="button"
            onClick={start}
            disabled={busy !== null}
            style={CAPTURE_BUTTON_STYLE}
          >
            {busy === "start" ? "starting…" : "start capture"}
          </button>
        ) : (
          <button
            type="button"
            onClick={stop}
            disabled={busy !== null}
            style={CAPTURE_BUTTON_STYLE}
          >
            {busy === "stop" ? "finalizing…" : "stop + open job"}
          </button>
        )}
      </div>

      {(pageError || sessionError) && (
        <div
          style={{
            position: "absolute",
            // Stacks above the diagnostic chip (which now sits at 88px)
            // so they don't overlap when both are visible.
            bottom:
              "max(140px, calc(env(safe-area-inset-bottom) + 140px))",
            left: 12,
            right: 12,
            padding: "6px 10px",
            background: "rgba(232, 89, 58, 0.85)",
            color: "#fff",
            fontSize: "var(--fs-xs)",
            borderRadius: "var(--r-xs)",
          }}
        >
          {pageError || sessionError}
        </div>
      )}

      {showLog && (
        <CaptureLogPanel
          log={log}
          onClose={() => setShowLog(false)}
          copyState={copyState}
          onCopy={async () => {
            const header = formatLogHeader({
              videoReady,
              videoSize,
              status,
              reconnectAttempt,
              framesSent,
              framesDroppedClient,
              framesProcessed: stats.frames,
              framesDroppedServer: stats.dropped,
              pointsCount,
            });
            const body = log
              .map(
                (e) =>
                  `[${(e.t / 1000).toFixed(2)}s] ${e.level.padEnd(5)} ${e.msg}`,
              )
              .join("\n");
            const text = `${header}\n\n${body}`;
            try {
              if (navigator.clipboard?.writeText) {
                await navigator.clipboard.writeText(text);
              } else {
                // Fallback for browsers without async clipboard (older
                // mobile Safari versions). The textarea-select-execCommand
                // trick still works there.
                const ta = document.createElement("textarea");
                ta.value = text;
                ta.style.position = "fixed";
                ta.style.opacity = "0";
                document.body.appendChild(ta);
                ta.select();
                document.execCommand("copy");
                document.body.removeChild(ta);
              }
              setCopyState("ok");
            } catch {
              setCopyState("err");
            }
            setTimeout(() => setCopyState("idle"), 1500);
          }}
        />
      )}

      {/* Layout rules. Inline so the page stays self-contained — no
          new globals.css edits for a feature that lives at one URL. */}
      <style jsx>{`
        .capture-video-pane {
          position: absolute;
          inset: 0;
          z-index: 0;
        }
        .capture-canvas-pane {
          position: absolute;
          z-index: 1;
          /* portrait default */
          top: 12px;
          left: 12px;
          width: 35vw;
          height: 35vw;
          max-width: 240px;
          max-height: 240px;
          border: 1px solid rgba(255, 255, 255, 0.4);
          border-radius: var(--r-xs);
          overflow: hidden;
          box-shadow: 0 0 0 2px rgba(10, 10, 10, 0.4);
        }
        @media (min-width: 900px) {
          /* landscape: video left half, canvas right half */
          .capture-video-pane {
            inset: 0 50% 0 0;
          }
          .capture-canvas-pane {
            top: 0;
            left: 50%;
            width: 50%;
            height: 100dvh;
            max-width: none;
            max-height: none;
            border: none;
            border-left: 1px solid rgba(255, 255, 255, 0.4);
            border-radius: 0;
            box-shadow: none;
          }
        }
      `}</style>
    </div>
  );
}

/** Slide-up overlay that exposes the client-side capture log. The
 *  dedicated panel exists so a phone user can copy the timeline +
 *  paste it into a chat when reporting a stuck capture — without it
 *  the chip alone is too narrow to surface much beyond the running
 *  counters. */
function CaptureLogPanel({
  log,
  onClose,
  onCopy,
  copyState,
}: {
  log: { t: number; level: "info" | "warn" | "error"; msg: string }[];
  onClose: () => void;
  onCopy: () => void;
  copyState: "idle" | "ok" | "err";
}) {
  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        zIndex: 10,
        display: "flex",
        flexDirection: "column",
        background: "rgba(10, 10, 10, 0.94)",
        color: "#fff",
        padding: 16,
      }}
      role="dialog"
      aria-label="capture log"
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 8,
          marginBottom: 12,
        }}
      >
        <div style={{ fontSize: "var(--fs-md)", fontWeight: 600 }}>
          capture log
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button
            type="button"
            onClick={onCopy}
            style={{
              padding: "8px 14px",
              minHeight: 40,
              background:
                copyState === "ok"
                  ? "#3a7"
                  : copyState === "err"
                    ? "#a33"
                    : "#fff",
              color: copyState === "idle" ? "#000" : "#fff",
              border: "1px solid #fff",
              borderRadius: 4,
              fontWeight: 600,
              fontSize: "var(--fs-sm)",
            }}
          >
            {copyState === "ok"
              ? "copied"
              : copyState === "err"
                ? "copy failed"
                : "copy"}
          </button>
          <button
            type="button"
            onClick={onClose}
            style={{
              padding: "8px 14px",
              minHeight: 40,
              background: "transparent",
              color: "#fff",
              border: "1px solid rgba(255, 255, 255, 0.5)",
              borderRadius: 4,
              fontSize: "var(--fs-sm)",
            }}
          >
            close
          </button>
        </div>
      </div>
      <div
        style={{
          flex: 1,
          overflowY: "auto",
          fontFamily: "var(--font-mono, ui-monospace, monospace)",
          fontSize: "var(--fs-xs)",
          lineHeight: 1.4,
          background: "rgba(0, 0, 0, 0.5)",
          padding: 8,
          borderRadius: 4,
          // Keep the text readable on narrow screens — wrap rather
          // than horizontal-scroll, which is awkward on mobile.
          whiteSpace: "pre-wrap",
          wordBreak: "break-word",
        }}
      >
        {log.length === 0 ? (
          <div style={{ opacity: 0.6 }}>
            (no log entries yet — start a capture to populate)
          </div>
        ) : (
          log.map((e, i) => (
            <div
              key={i}
              style={{
                color:
                  e.level === "error"
                    ? "#f88"
                    : e.level === "warn"
                      ? "#fc8"
                      : "#cdf",
              }}
            >
              [{(e.t / 1000).toFixed(2)}s] {e.level.padEnd(5)} {e.msg}
            </div>
          ))
        )}
      </div>
    </div>
  );
}

/** Build the human-readable header that gets prepended to the
 *  copied log so the recipient sees the chip stats in context. */
function formatLogHeader(state: {
  videoReady: boolean;
  videoSize: [number, number] | null;
  status: string;
  reconnectAttempt: number;
  framesSent: number;
  framesDroppedClient: number;
  framesProcessed: number;
  framesDroppedServer: number;
  pointsCount: number;
}): string {
  const lines = [
    `# vid3d capture log @ ${new Date().toISOString()}`,
    `ua: ${typeof navigator !== "undefined" ? navigator.userAgent : "?"}`,
    `video: ${state.videoReady && state.videoSize ? `${state.videoSize[0]}x${state.videoSize[1]}` : state.videoReady ? "ready" : "not ready"}`,
    `ws: ${state.status}${state.reconnectAttempt > 0 ? ` (retry ${state.reconnectAttempt})` : ""}`,
    `frames sent=${state.framesSent} dropped(client)=${state.framesDroppedClient}`,
    `frames processed=${state.framesProcessed} dropped(server)=${state.framesDroppedServer}`,
    `points=${state.pointsCount}`,
  ];
  return lines.join("\n");
}

/** Streams the live sparse cloud as a `<points>` geometry. Reads the
 *  store's flat Float32Arrays + only re-uploads the GPU buffer when
 *  the count grows. */
function CapturePointCloud() {
  const xyz = useCaptureStore((s) => s.pointsXyz);
  const rgb = useCaptureStore((s) => s.pointsRgb);
  const count = useCaptureStore((s) => s.pointsCount);

  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(xyz.slice(0, count * 3), 3));
    g.setAttribute("color", new THREE.BufferAttribute(rgb.slice(0, count * 3), 3));
    return g;
  }, [xyz, rgb, count]);

  if (count === 0) return null;
  return (
    <points geometry={geometry}>
      <pointsMaterial
        size={0.015}
        vertexColors
        sizeAttenuation
      />
    </points>
  );
}

/** Friendly full-page notice shown when the page is loaded outside a
 *  secure context (plain http on a non-localhost origin). Without this,
 *  the user taps Start, the camera prompt never appears (because
 *  `navigator.mediaDevices` is undefined off secure-context), and they
 *  have no clue why — the silent-fail mode is the worst possible UX
 *  for a feature that *only* works behind HTTPS.
 *
 *  The notice constructs the suggested HTTPS URL from the current
 *  origin (drops the port — Caddy listens on the standard 443) and
 *  links to the README's `make up-https` section for the cert install
 *  recipe. Same anchor used in the home-page CTA so the docs flow is
 *  consistent. */
function InsecureContextNotice() {
  // One state object so we only fire one setState on mount — keeps
  // the cascading-render lint rule happy and matches the on-mount
  // semantics (these values never change after first render).
  const [loc, setLoc] = useState<{ host: string; httpsUrl: string } | null>(
    null,
  );

  useEffect(() => {
    // Defer past the current render tick — same reason as the parent's
    // insecure detection: avoids the synchronous-setState-in-effect lint.
    let cancelled = false;
    void Promise.resolve().then(() => {
      if (cancelled) return;
      const { hostname, pathname, search } = window.location;
      // Standard 443 — Caddy publishes there. Drop any custom dev port
      // because the production HTTPS path doesn't carry one.
      setLoc({
        host: hostname,
        httpsUrl: `https://${hostname}${pathname}${search}`,
      });
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const host = loc?.host ?? null;
  const suggestedUrl = loc?.httpsUrl ?? null;

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "grid",
        placeItems: "center",
        padding: 24,
        background: "#0a0a0a",
        color: "#fff",
        fontSize: "var(--fs-md)",
        lineHeight: 1.5,
      }}
    >
      <div
        style={{
          maxWidth: 560,
          padding: 24,
          border: "1px solid rgba(255, 255, 255, 0.2)",
          borderRadius: "var(--r-sm)",
          background: "rgba(255, 255, 255, 0.03)",
        }}
      >
        <h1
          style={{
            margin: 0,
            marginBottom: 12,
            fontSize: "var(--fs-lg)",
            letterSpacing: "0.02em",
          }}
        >
          camera capture needs HTTPS
        </h1>
        <p style={{ margin: "0 0 12px" }}>
          Mobile browsers refuse{" "}
          <code style={{ background: "rgba(255,255,255,0.08)", padding: "0 4px" }}>
            getUserMedia
          </code>{" "}
          on plain http unless the page is on{" "}
          <code style={{ background: "rgba(255,255,255,0.08)", padding: "0 4px" }}>
            localhost
          </code>
          . The studio supports an opt-in HTTPS path with a one-shot
          setup; the recipe below gets you scanning in a couple minutes.
        </p>
        <ol style={{ margin: "12px 0", paddingLeft: 20 }}>
          <li style={{ marginBottom: 8 }}>
            On the studio host, stop the current stack and run{" "}
            <code style={{ background: "rgba(255,255,255,0.08)", padding: "0 4px" }}>
              make up-https
            </code>
            . It auto-installs mkcert, trusts a local root CA, generates
            a cert pair covering this host, and starts Caddy on 443.
          </li>
          <li style={{ marginBottom: 8 }}>
            From this phone, visit{" "}
            {host ? (
              <code style={{ background: "rgba(255,255,255,0.08)", padding: "0 4px" }}>
                http://{host}/mkcert-rootCA.pem
              </code>
            ) : (
              "http://<host>/mkcert-rootCA.pem"
            )}{" "}
            and install the file (Settings → Security → Install
            certificate → CA on Android; General → VPN & Device
            Management + Certificate Trust Settings on iOS).
          </li>
          <li style={{ marginBottom: 8 }}>
            Reload this page over HTTPS:{" "}
            {suggestedUrl ? (
              <a
                href={suggestedUrl}
                style={{ color: "#ffd166", textDecoration: "underline" }}
              >
                {suggestedUrl}
              </a>
            ) : (
              <span>https://&lt;host&gt;/capture</span>
            )}
          </li>
        </ol>
        <p
          style={{
            margin: "16px 0 0",
            fontSize: "var(--fs-sm)",
            opacity: 0.7,
          }}
        >
          Full walkthrough:{" "}
          <a
            href="https://github.com/Battleroid/lingbot-map-studio#scanning-from-a-phone--make-up-https"
            target="_blank"
            rel="noreferrer noopener"
            style={{ color: "#ffd166" }}
          >
            README → Scanning from a phone
          </a>
          . If you already have HTTPS set up but still landed here,{" "}
          <code
            style={{ background: "rgba(255,255,255,0.08)", padding: "0 4px" }}
          >
            navigator.mediaDevices
          </code>{" "}
          may be unavailable in this browser — try a recent Chrome /
          Safari / Firefox.
        </p>
      </div>
    </div>
  );
}

/** Renders the latest pose as a small frustum so the user can read
 *  "where am I in this 3D space?" while panning. */
function CaptureFrustum() {
  const poses = useCaptureStore((s) => s.poses);
  const latest = poses.length > 0 ? poses[poses.length - 1] : null;
  if (!latest) return null;
  const [x, y, z] = latest.t;
  return (
    <mesh position={[x, y, z]}>
      <sphereGeometry args={[0.05, 12, 8]} />
      <meshBasicMaterial color="#d8348a" />
    </mesh>
  );
}
