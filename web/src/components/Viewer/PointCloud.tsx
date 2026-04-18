"use client";

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
  const pointSizeAuto = useViewerStore((s) => s.pointSizeAuto);
  const setPointSizeFromAuto = useViewerStore((s) => s.setPointSizeFromAuto);
  const ptsRef = useRef<THREE.Points>(null);

  // Compute a sensible default point size from the geometry's bounding box
  // so the cloud is visible regardless of scene scale (depth maps can emit
  // scenes that span 1e-2 to 1e3 units).
  useEffect(() => {
    if (!pointSizeAuto) return;
    if (!geometry.boundingBox) geometry.computeBoundingBox();
    const bb = geometry.boundingBox;
    if (!bb) return;
    const size = new THREE.Vector3();
    bb.getSize(size);
    const diag = size.length();
    if (!Number.isFinite(diag) || diag <= 0) return;
    // ~0.15% of scene diagonal → visible without crushing small features.
    const target = Math.max(0.0005, Math.min(1.5, diag * 0.0015));
    // Only touch the store if the current size is meaningfully different so
    // we don't bounce on every re-render.
    if (Math.abs(target - pointSize) / Math.max(target, 1e-6) > 0.1) {
      setPointSizeFromAuto(target);
    }
  }, [geometry, pointSizeAuto, pointSize, setPointSizeFromAuto]);

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
