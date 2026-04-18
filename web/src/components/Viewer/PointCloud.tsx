"use client";

import { useBounds } from "@react-three/drei";
import { useLoader } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";

import { useViewerStore } from "@/lib/viewerStore";

interface Props {
  url: string;
  color: boolean;
}

export function PointCloud({ url, color }: Props) {
  const geometry = useLoader(PLYLoader, url) as THREE.BufferGeometry;
  const pointSize = useViewerStore((s) => s.pointSize);
  const setAutoPointSize = useViewerStore((s) => s.setAutoPointSize);
  const setSceneDiagonal = useViewerStore((s) => s.setSceneDiagonal);
  const setSceneScale = useViewerStore((s) => s.setSceneScale);
  const bounds = useBounds();
  const ptsRef = useRef<THREE.Points>(null);
  const hasFittedRef = useRef(false);

  // Compute point-size, scene diagonal, AND scene scale from the geometry.
  // scene_scale = ||P95 - P5|| of point positions — matches lingbot-map's
  // upstream scene-scale logic (resistant to outlier points that would
  // otherwise dominate min/max bbox). Used to size camera frustums + path.
  useEffect(() => {
    if (!geometry.boundingBox) geometry.computeBoundingBox();
    const bb = geometry.boundingBox;
    if (!bb) return;
    const size = new THREE.Vector3();
    bb.getSize(size);
    const diag = size.length();
    if (!Number.isFinite(diag) || diag <= 0) return;
    setSceneDiagonal(diag);
    setAutoPointSize(Math.max(0.0005, Math.min(2, diag * 0.0015)));

    // Percentile-based scale. Sort per-axis, pick P5/P95, diagonal of the
    // extent vector. Float32Array.sort is a typed sort — fast for 1-2M pts.
    const pos = geometry.getAttribute("position");
    if (!pos) return;
    const n = pos.count;
    if (n < 100) {
      setSceneScale(diag);
      return;
    }
    const xs = new Float32Array(n);
    const ys = new Float32Array(n);
    const zs = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      xs[i] = pos.getX(i);
      ys[i] = pos.getY(i);
      zs[i] = pos.getZ(i);
    }
    xs.sort();
    ys.sort();
    zs.sort();
    const p5 = Math.floor(n * 0.05);
    const p95 = Math.floor(n * 0.95);
    const dx = xs[p95] - xs[p5];
    const dy = ys[p95] - ys[p5];
    const dz = zs[p95] - zs[p5];
    const scale = Math.max(0.1, Math.sqrt(dx * dx + dy * dy + dz * dz));
    setSceneScale(scale);
  }, [geometry, setAutoPointSize, setSceneDiagonal, setSceneScale]);

  // Fit the camera ONCE per mount. Subsequent URL changes (incoming partial
  // snapshots) must NOT refit — otherwise it yanks the camera while the user
  // is orbiting. Manual recenter goes through a separate refitSignal path.
  useEffect(() => {
    if (hasFittedRef.current) return;
    hasFittedRef.current = true;
    const id = requestAnimationFrame(() => {
      try {
        // No .clip() — tightening near/far breaks fly mode (see Canvas.tsx).
        bounds.refresh().fit();
      } catch {
        hasFittedRef.current = false;
      }
    });
    return () => cancelAnimationFrame(id);
  }, [bounds, geometry]);

  const material = useMemo(() => {
    if (color && geometry.getAttribute("color")) {
      return new THREE.PointsMaterial({
        size: pointSize,
        vertexColors: true,
        sizeAttenuation: true,
      });
    }
    return new THREE.PointsMaterial({
      size: pointSize,
      color: "#000000",
      sizeAttenuation: true,
    });
  }, [color, geometry, pointSize]);

  return <points ref={ptsRef} geometry={geometry} material={material} />;
}
