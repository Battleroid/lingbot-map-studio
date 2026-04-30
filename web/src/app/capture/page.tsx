"use client";

import { Canvas } from "@react-three/fiber";
import { Bounds, OrbitControls } from "@react-three/drei";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState, type CSSProperties } from "react";
import * as THREE from "three";

import { CameraStream } from "@/components/capture/CameraStream";
import {
  CoverageVoxels,
  useCoverageSummary,
} from "@/components/capture/CoverageVoxels";
import { SplatLayer } from "@/components/Viewer/SplatLayer";
import {
  startCaptureSession,
  stopCaptureSession,
  type CaptureBackend,
} from "@/lib/api";
import {
  MAX_RECONNECT_ATTEMPTS,
  useCaptureStore,
} from "@/lib/captureSession";

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
export default function CapturePage() {
  const router = useRouter();
  const [backend, setBackend] = useState<CaptureBackend>("mast3r_slam");
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
  const latestSplatPreview = useCaptureStore((s) => s.latestSplatPreview);
  const videoReady = useCaptureStore((s) => s.videoReady);
  const videoSize = useCaptureStore((s) => s.videoSize);

  // "open" or actively retrying — both are flavours of an in-progress
  // session, so treat them as `capturing` for the purpose of which
  // button (Start vs Stop) is rendered. Without this a transient
  // network blip would flip the UI to "Start" mid-scan, which would
  // erroneously kick off a new session.
  const capturing = status === "open" || reconnectAttempt > 0;
  const reconnecting = reconnectAttempt > 0 && status !== "open";
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
          <Bounds margin={1.4} observe>
            <CoverageVoxels />
            {/* Splat preview when the server has written one;
                otherwise fall back to the raw point cloud so the
                user always sees *something* growing. SplatLayer
                renders the same xyz positions as gaussians, so once
                the first splat snapshot arrives (~2 s in) we hide
                the points to avoid double-render. */}
            {latestSplatPreview ? (
              <SplatLayer url={latestSplatPreview} />
            ) : (
              <CapturePointCloud />
            )}
            <CaptureFrustum />
          </Bounds>
          <axesHelper args={[0.3]} />
          <OrbitControls makeDefault enableDamping dampingFactor={0.08} />
        </Canvas>
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
          top: 12,
          left: 12,
          padding: "6px 10px",
          background: "rgba(10, 10, 10, 0.65)",
          fontSize: "var(--fs-xs)",
          letterSpacing: "0.04em",
          borderRadius: "var(--r-xs)",
          maxWidth: "calc(100vw - 24px)",
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

      {reconnecting && (
        <div
          style={{
            position: "absolute",
            top: 44,
            left: 12,
            padding: "6px 10px",
            background: "rgba(232, 180, 58, 0.85)",
            color: "#222",
            fontSize: "var(--fs-xs)",
            borderRadius: "var(--r-xs)",
          }}
          role="status"
        >
          reconnecting… (attempt {reconnectAttempt}/{MAX_RECONNECT_ATTEMPTS})
        </div>
      )}

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
          {/* MonoGS: produces a real Gaussian Splat as it tracks. The
              capture session feeds it the same frame stream + writes
              a live splat preview to disk, which the canvas above
              renders via SplatLayer. */}
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
            // Sit just above the controls, scaled the same way so
            // both move together when the safe-area inset is non-zero.
            bottom: "max(88px, calc(env(safe-area-inset-bottom) + 88px))",
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
