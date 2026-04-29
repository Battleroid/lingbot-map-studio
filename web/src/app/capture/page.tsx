"use client";

import { Canvas } from "@react-three/fiber";
import { Bounds, OrbitControls } from "@react-three/drei";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import * as THREE from "three";

import { CameraStream } from "@/components/capture/CameraStream";
import {
  CoverageVoxels,
  useCoverageSummary,
} from "@/components/capture/CoverageVoxels";
import {
  startCaptureSession,
  stopCaptureSession,
  type CaptureBackend,
} from "@/lib/api";
import { useCaptureStore } from "@/lib/captureSession";

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

  const capturing = status === "open";
  const summary = useCoverageSummary();

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
        height: "100vh",
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
            <CapturePointCloud />
            <CaptureFrustum />
          </Bounds>
          <axesHelper args={[0.3]} />
          <OrbitControls makeDefault enableDamping dampingFactor={0.08} />
        </Canvas>
      </div>

      {/* Stats + controls overlay. */}
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
        }}
      >
        ●&nbsp;{pointsCount.toLocaleString()} pts &middot;{" "}
        {Math.round(summary.ratio * 100)}% covered &middot;{" "}
        {stats.frames} frames {stats.dropped > 0 && `(${stats.dropped} dropped)`}
      </div>

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
          bottom: 24,
          left: 0,
          right: 0,
          display: "flex",
          justifyContent: "center",
          gap: 12,
        }}
      >
        {!capturing ? (
          <button
            type="button"
            onClick={start}
            disabled={busy !== null}
            style={{ padding: "8px 20px", fontSize: "var(--fs-md)" }}
          >
            {busy === "start" ? "starting…" : "start capture"}
          </button>
        ) : (
          <button
            type="button"
            onClick={stop}
            disabled={busy !== null}
            style={{ padding: "8px 20px", fontSize: "var(--fs-md)" }}
          >
            {busy === "stop" ? "finalizing…" : "stop + open job"}
          </button>
        )}
      </div>

      {(pageError || sessionError) && (
        <div
          style={{
            position: "absolute",
            bottom: 80,
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
            height: 100vh;
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
