"use client";

import { useGLTF } from "@react-three/drei";
import { useEffect, useMemo } from "react";
import * as THREE from "three";

interface Props {
  url: string;
  wireframe: boolean;
}

export function MeshLayer({ url, wireframe }: Props) {
  const gltf = useGLTF(url, true);
  const scene = useMemo(() => gltf.scene.clone(true), [gltf.scene]);

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
