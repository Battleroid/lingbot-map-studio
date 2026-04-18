"use client";

import { Canvas } from "@react-three/fiber";
import { Bounds, Grid, OrbitControls } from "@react-three/drei";
import { Suspense, useMemo, useRef } from "react";
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

export function ViewerCanvas({ glbUrl, plyUrl }: Props) {
  const mode = useViewerStore((s) => s.mode);
  const lassoActive = useViewerStore((s) => s.lassoActive);
  const controls = useRef<OrbitControlsImpl>(null);

  const showPoints = mode === "points" || mode === "points-color";
  const showMesh = mode === "mesh" || mode === "wireframe";

  // Bounds keys include the active URL so that when a new partial PLY lands,
  // Bounds remounts and refits the camera to the new geometry.
  const boundsKey = `${showMesh ? glbUrl ?? "" : ""}|${showPoints ? plyUrl ?? "" : ""}|${mode}`;

  const cameraProps = useMemo(
    () => ({ position: [2.5, 2.5, 2.5] as [number, number, number], fov: 45, near: 0.01, far: 5000 }),
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
          <Bounds key={boundsKey} fit clip observe margin={1.3}>
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
