"use client";

import { Canvas } from "@react-three/fiber";
import {
  Bounds,
  FlyControls,
  Grid,
  OrbitControls,
  useBounds,
} from "@react-three/drei";
import { Suspense, useEffect, useMemo, useRef } from "react";
import type { FlyControls as FlyControlsImpl } from "three-stdlib";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";

import { CameraPath, type CameraPose } from "./CameraPath";
import { LassoSelect } from "./LassoSelect";
import { MeshLayer } from "./MeshLayer";
import { PointCloud } from "./PointCloud";
import { useViewerStore } from "@/lib/viewerStore";

interface Props {
  glbUrl: string | null;
  plyUrl: string | null;
  cameraPath?: { fps: number; poses: CameraPose[] } | null;
}

function AxesAndGrid() {
  return (
    <>
      {/* fadeDistance is relative to the camera. Large here because
          reconstructed scenes can span hundreds of units and a tight value
          makes the grid invisible once the camera is far from origin. */}
      <Grid
        args={[40, 40]}
        cellColor="#cccccc"
        sectionColor="#000000"
        sectionThickness={1}
        fadeDistance={2000}
        fadeStrength={2}
        infiniteGrid
      />
      <axesHelper args={[1]} />
    </>
  );
}

/**
 * Re-fits the camera when the user clicks "recenter". Per-mount initial fit
 * is handled inside PointCloud / MeshLayer — those only fire once per mount,
 * so partial updates during live inference don't steal the user's orbit.
 *
 * Also retriggers when cameraMode changes, so switching to fly mode lands on
 * a known-good frame instead of leaving the user stranded far from the scene.
 */
function RefitController() {
  const bounds = useBounds();
  const signal = useViewerStore((s) => s.refitSignal);
  const cameraMode = useViewerStore((s) => s.cameraMode);
  useEffect(() => {
    const id = requestAnimationFrame(() => {
      try {
        // Refresh picks up any new geometry; fit positions the camera.
        // We no longer call clip() — it tightens near/far too aggressively,
        // which frustum-culls everything as soon as fly mode walks away.
        bounds.refresh().fit();
      } catch {
        /* Bounds not ready yet */
      }
    });
    return () => cancelAnimationFrame(id);
  }, [bounds, signal, cameraMode]);
  return null;
}

// Base speed is picked so traversing the entire scene takes ~5 s at a
// continuous key hold, scaled by the user's fly-speed multiplier.
const BASE_SPEED_PER_DIAG = 0.2;
const SHIFT_SLOW_FACTOR = 0.12;

/**
 * Drives the FlyControls' `movementSpeed` from the scene's bounding-box
 * diagonal and listens for Shift to temporarily slow to a crawl.
 *
 * Without scene-aware scaling, a fixed movement speed is useless because
 * the reconstructions span anything from 2 units across to 200+.
 */
function FlySpeedModifier({
  flyRef,
}: {
  flyRef: React.RefObject<FlyControlsImpl | null>;
}) {
  const sceneDiagonal = useViewerStore((s) => s.sceneDiagonal);
  const pathDiagonal = useViewerStore((s) => s.pathDiagonal);
  const flySpeedMult = useViewerStore((s) => s.flySpeedMult);
  const shiftHeldRef = useRef(false);

  // Use the LARGER of point-cloud and camera-path extents — FPV-style
  // reconstructions often have a camera flight wider than the visible
  // points, and scaling speed only to the tight point cluster makes
  // traversing the flight path painful.
  const effectiveDiag = Math.max(sceneDiagonal, pathDiagonal);
  const baseSpeed = Math.max(0.05, effectiveDiag * BASE_SPEED_PER_DIAG * flySpeedMult);

  // Apply whenever base speed changes (scene diagonal updated, multiplier
  // changed, or controls remounted on refit).
  useEffect(() => {
    const fc = flyRef.current;
    if (!fc) return;
    fc.movementSpeed = shiftHeldRef.current
      ? baseSpeed * SHIFT_SLOW_FACTOR
      : baseSpeed;
  }, [baseSpeed, flyRef]);

  useEffect(() => {
    const apply = () => {
      const fc = flyRef.current;
      if (!fc) return;
      fc.movementSpeed = shiftHeldRef.current
        ? baseSpeed * SHIFT_SLOW_FACTOR
        : baseSpeed;
    };
    const onDown = (e: KeyboardEvent) => {
      if (e.key !== "Shift") return;
      shiftHeldRef.current = true;
      apply();
    };
    const onUp = (e: KeyboardEvent) => {
      if (e.key !== "Shift") return;
      shiftHeldRef.current = false;
      apply();
    };
    window.addEventListener("keydown", onDown);
    window.addEventListener("keyup", onUp);
    return () => {
      window.removeEventListener("keydown", onDown);
      window.removeEventListener("keyup", onUp);
      shiftHeldRef.current = false;
      const fc = flyRef.current;
      if (fc) fc.movementSpeed = baseSpeed;
    };
  }, [baseSpeed, flyRef]);
  return null;
}

export function ViewerCanvas({ glbUrl, plyUrl, cameraPath }: Props) {
  const mode = useViewerStore((s) => s.mode);
  const cameraMode = useViewerStore((s) => s.cameraMode);
  const refitSignal = useViewerStore((s) => s.refitSignal);
  const lassoActive = useViewerStore((s) => s.lassoActive);
  const controls = useRef<OrbitControlsImpl>(null);
  const flyRef = useRef<FlyControlsImpl>(null);

  const showPoints = mode === "points" || mode === "points-color";
  const showMesh = mode === "mesh" || mode === "wireframe";

  const cameraProps = useMemo(
    () => ({
      position: [2.5, 2.5, 2.5] as [number, number, number],
      fov: 45,
      near: 0.001,
      far: 10000,
    }),
    [],
  );

  return (
    <div style={{ position: "relative", width: "100%", height: "100%" }}>
      <Canvas camera={cameraProps}>
        <color attach="background" args={["#ffffff"]} />
        <ambientLight intensity={0.9} />
        <directionalLight position={[5, 8, 3]} intensity={0.9} />
        <Suspense fallback={null}>
          <AxesAndGrid />
          {/* Bounds without `fit` = provide the useBounds context only; never
              fit on mount (which would fit on infinite grid before geometry
              arrives, causing the camera to zoom to ∞). Initial fit is
              triggered by PointCloud/MeshLayer once their geometry loads.
              `clip` is deliberately off — it auto-tightens camera near/far
              around the scene, which causes fly mode to frustum-cull
              everything the moment the camera walks outside those bounds. */}
          <Bounds observe margin={1.3}>
            <RefitController />
            {showMesh && glbUrl && (
              <MeshLayer url={glbUrl} wireframe={mode === "wireframe"} />
            )}
            {showPoints && plyUrl && (
              <PointCloud url={plyUrl} color={mode === "points-color"} />
            )}
            {/* Camera path lives INSIDE Bounds so the initial fit includes
                both the reconstructed points and the camera trajectory —
                otherwise tight-clustered points hide a wide-flight path. */}
            {cameraPath && cameraPath.poses.length > 1 && (
              <CameraPath
                poses={cameraPath.poses}
                recordedFps={cameraPath.fps || 10}
              />
            )}
          </Bounds>
          {showMesh && glbUrl && lassoActive && <LassoSelect />}
        </Suspense>
        {cameraMode === "orbit" ? (
          <OrbitControls
            ref={controls}
            enabled={!lassoActive}
            makeDefault
            enableDamping
            dampingFactor={0.08}
          />
        ) : (
          // Key on refitSignal so a recenter click fully remounts FlyControls
          // and it picks up the camera's freshly-set pitch/yaw/position
          // instead of overwriting them from its stale internal state.
          <>
            <FlyControls
              key={`fly-${refitSignal}`}
              ref={flyRef}
              makeDefault
              movementSpeed={1}
              rollSpeed={0.6}
              dragToLook
            />
            <FlySpeedModifier flyRef={flyRef} />
          </>
        )}
      </Canvas>
      {cameraMode === "fly" && (
        <div
          style={{
            position: "absolute",
            bottom: 10,
            left: 10,
            padding: "4px 8px",
            background: "var(--fg)",
            color: "var(--bg)",
            fontSize: "var(--fs-xs)",
            letterSpacing: "0.06em",
            pointerEvents: "none",
          }}
        >
          fly · WASD move · drag to look · R/F up/down · Q/E roll · hold SHIFT to crawl
        </div>
      )}
      {!glbUrl && !plyUrl && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            display: "grid",
            placeItems: "center",
            color: "var(--muted)",
            fontSize: 12,
            letterSpacing: "0.08em",
            textTransform: "uppercase",
            pointerEvents: "none",
          }}
        >
          waiting for reconstruction…
        </div>
      )}
    </div>
  );
}
