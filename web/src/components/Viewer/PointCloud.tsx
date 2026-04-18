"use client";

import { useLoader } from "@react-three/fiber";
import { useMemo } from "react";
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

  return <points geometry={geometry} material={material} />;
}
