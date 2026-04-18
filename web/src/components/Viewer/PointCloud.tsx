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
  const bounds = useBounds();
  const ptsRef = useRef<THREE.Points>(null);
  const hasFittedRef = useRef(false);

  // Compute the auto base point size from the scene bbox. Re-computes on URL
  // change (e.g. new partial) so the slider multiplier stays meaningful.
  useEffect(() => {
    if (!geometry.boundingBox) geometry.computeBoundingBox();
    const bb = geometry.boundingBox;
    if (!bb) return;
    const size = new THREE.Vector3();
    bb.getSize(size);
    const diag = size.length();
    if (!Number.isFinite(diag) || diag <= 0) return;
    // ~0.15% of the scene diagonal → visible from any distance.
    setAutoPointSize(Math.max(0.0005, Math.min(2, diag * 0.0015)));
  }, [geometry, setAutoPointSize]);

  // Fit the camera ONCE per mount. Subsequent URL changes (incoming partial
  // snapshots) must NOT refit — otherwise it yanks the camera while the user
  // is orbiting. Manual recenter goes through a separate refitSignal path.
  useEffect(() => {
    if (hasFittedRef.current) return;
    hasFittedRef.current = true;
    const id = requestAnimationFrame(() => {
      try {
        bounds.refresh().clip().fit();
      } catch {
        /* Bounds not ready yet; rerun on next mount. */
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
