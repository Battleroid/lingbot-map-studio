"use client";

import { useBounds, useGLTF } from "@react-three/drei";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";

interface Props {
  url: string;
  wireframe: boolean;
}

export function MeshLayer({ url, wireframe }: Props) {
  const gltf = useGLTF(url, true);
  const scene = useMemo(() => gltf.scene.clone(true), [gltf.scene]);
  const bounds = useBounds();
  const hasFittedRef = useRef(false);

  // Fit once per mount only; partial updates must not disturb the user's
  // orbit. Manual recenter uses the refitSignal path (Canvas.tsx).
  useEffect(() => {
    if (hasFittedRef.current) return;
    hasFittedRef.current = true;
    const id = requestAnimationFrame(() => {
      try {
        bounds.refresh().clip().fit();
      } catch {
        hasFittedRef.current = false;
      }
    });
    return () => cancelAnimationFrame(id);
  }, [bounds, scene]);

  useEffect(() => {
    scene.traverse((obj) => {
      if ((obj as THREE.Mesh).isMesh) {
        const mesh = obj as THREE.Mesh;
        const mat = mesh.material as THREE.Material | THREE.Material[];
        const apply = (m: THREE.Material) => {
          if ("wireframe" in m) {
            (m as THREE.MeshStandardMaterial).wireframe = wireframe;
          }
          m.side = THREE.DoubleSide;
          if (wireframe && "color" in m) {
            (m as THREE.MeshStandardMaterial).color = new THREE.Color("#000000");
          }
        };
        if (Array.isArray(mat)) mat.forEach(apply);
        else apply(mat);
      }
      if ((obj as THREE.Points).isPoints) {
        const pts = obj as THREE.Points;
        pts.visible = !wireframe;
      }
    });
  }, [scene, wireframe]);

  return <primitive object={scene} />;
}

useGLTF.preload = () => undefined;
