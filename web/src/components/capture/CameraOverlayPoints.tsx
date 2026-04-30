"use client";

import { Canvas } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";

import { useCaptureStore } from "@/lib/captureSession";

import { FollowPoseCamera } from "./FollowPoseCamera";

/**
 * Real-space points overlay on the live camera feed.
 *
 * Replaces the earlier candy-stripe coverage mask, which used a
 * 36×18 az/el bucket grid that filled in within seconds (the
 * simulated SLAM tracker emits feature corners scattered widely
 * across the camera frame, and a single point claims a whole 10°
 * cone) — so the stripes "disappeared and were never seen again".
 *
 * Visual: the camera feed is the underlay. On top, a transparent
 * three.js Canvas mounts a FollowPoseCamera that snaps to the live
 * SLAM pose. The Canvas renders the SLAM point cloud as bright
 * dots in actual world space. As the user pans, the dots stay
 * stuck to the world locations the tracker has observed; areas
 * with no dots haven't been scanned yet.
 *
 * This is the simpler half of the user's "either gaussian points
 * in real space or a working overlay" ask. It doesn't need a
 * coverage projection / shader / depth assumption — it's just the
 * existing tracked points rendered through the live camera basis.
 */
export function CameraOverlayPoints() {
  return (
    <Canvas
      gl={{ alpha: true, premultipliedAlpha: false, antialias: false }}
      camera={{
        position: [0, 0, 0],
        // 60° HFOV → ~46° VFOV at 4:3, matching the simulated SLAM
        // tracker's intrinsic assumption (`fx = w * 0.866`). Same
        // FOV as the previous mask so the overlay stays internally
        // consistent with the SLAM coordinate frame.
        fov: 46,
        near: 0.01,
        far: 100,
      }}
      style={{
        position: "absolute",
        inset: 0,
        background: "transparent",
        pointerEvents: "none",
      }}
    >
      <FollowPoseCamera />
      <CapturedPointCloudPoints />
    </Canvas>
  );
}

/** Renders the existing pointsXyz / pointsRgb buffers as a single
 *  `<points>` instance. Re-uploads the GPU buffer when the count
 *  grows; reuses the same BufferAttribute references in between to
 *  avoid GC churn. Points are rendered larger + slightly opaque so
 *  they read as a clear "captured here" cue against the moving
 *  camera feed underneath. */
function CapturedPointCloudPoints() {
  const xyz = useCaptureStore((s) => s.pointsXyz);
  const rgb = useCaptureStore((s) => s.pointsRgb);
  const count = useCaptureStore((s) => s.pointsCount);
  const geometryRef = useRef<THREE.BufferGeometry | null>(null);

  // Build the geometry once; mutate it in place when the point set
  // grows. Allocating a fresh BufferGeometry per render would churn
  // the GPU upload + tank phone fps once we have a few thousand
  // points. The BufferAttribute size matches the underlying typed
  // array, so we just bump `drawRange` to expose more points.
  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(xyz, 3));
    g.setAttribute("color", new THREE.BufferAttribute(rgb, 3));
    g.setDrawRange(0, count);
    geometryRef.current = g;
    return g;
    // We intentionally rebuild the geometry only when xyz / rgb
    // identity changes (i.e. the store grew the underlying typed
    // array — power-of-two reallocation in `appendPoints`). Mid-
    // capture growth without realloc is handled by the effect below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [xyz, rgb]);

  useEffect(() => {
    const g = geometryRef.current;
    if (!g) return;
    g.setDrawRange(0, count);
    const pos = g.getAttribute("position") as THREE.BufferAttribute;
    const col = g.getAttribute("color") as THREE.BufferAttribute;
    pos.needsUpdate = true;
    col.needsUpdate = true;
  }, [count]);

  if (count === 0) return null;
  return (
    <points geometry={geometry}>
      <pointsMaterial
        size={0.025}
        vertexColors
        sizeAttenuation
        transparent
        opacity={0.9}
        depthWrite={false}
      />
    </points>
  );
}
