"use client";

import { Canvas } from "@react-three/fiber";
import { Bounds, Grid, OrbitControls, useBounds } from "@react-three/drei";
import { Suspense, useEffect, useMemo, useRef } from "react";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";

import { LassoSelect } from "./LassoSelect";
import { MeshLayer } from "./MeshLayer";
import { PointCloud } from "./PointCloud";
import { useViewerStore } from "@/lib/viewerStore";

interface Props {
  glbUrl: string | null;
  plyUrl: string | null;
}

function AxesAndGrid() {
  return (
    <>
      <Grid
        args={[40, 40]}
        cellColor="#cccccc"
        sectionColor="#000000"
        sectionThickness={1}
        fadeDistance={40}
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
 */
function RefitController() {
  const bounds = useBounds();
  const signal = useViewerStore((s) => s.refitSignal);
  useEffect(() => {
    if (signal === 0) return; // skip initial render; user hasn't requested refit
    const id = requestAnimationFrame(() => {
      try {
        bounds.refresh().clip().fit();
      } catch {
        /* Bounds not ready */
      }
    });
    return () => cancelAnimationFrame(id);
  }, [bounds, signal]);
  return null;
}

export function ViewerCanvas({ glbUrl, plyUrl }: Props) {
  const mode = useViewerStore((s) => s.mode);
  const lassoActive = useViewerStore((s) => s.lassoActive);
  const controls = useRef<OrbitControlsImpl>(null);

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
              triggered by PointCloud/MeshLayer once their geometry loads. */}
          <Bounds clip observe margin={1.3}>
            <RefitController />
            {showMesh && glbUrl && (
              <MeshLayer url={glbUrl} wireframe={mode === "wireframe"} />
            )}
            {showPoints && plyUrl && (
              <PointCloud url={plyUrl} color={mode === "points-color"} />
            )}
          </Bounds>
          {showMesh && glbUrl && lassoActive && <LassoSelect />}
        </Suspense>
        <OrbitControls
          ref={controls}
          enabled={!lassoActive}
          makeDefault
          enableDamping
          dampingFactor={0.08}
        />
      </Canvas>
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
