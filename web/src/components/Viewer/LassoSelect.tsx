"use client";

import { useThree } from "@react-three/fiber";
import { useEffect, useRef } from "react";
import * as THREE from "three";

import { useViewerStore } from "@/lib/viewerStore";

function pointInPolygon(x: number, y: number, poly: Array<[number, number]>): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i][0];
    const yi = poly[i][1];
    const xj = poly[j][0];
    const yj = poly[j][1];
    const intersect =
      yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi + 1e-12) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

export function LassoSelect() {
  const { gl, scene, camera } = useThree();
  const drawing = useRef(false);
  const points = useRef<Array<[number, number]>>([]);
  const overlay = useRef<HTMLCanvasElement | null>(null);
  const setSelection = useViewerStore((s) => s.setSelection);

  useEffect(() => {
    const canvas = gl.domElement;
    const parent = canvas.parentElement;
    if (!parent) return;

    // Full-bleed overlay for the lasso outline.
    const ov = document.createElement("canvas");
    ov.style.position = "absolute";
    ov.style.inset = "0";
    ov.style.pointerEvents = "none";
    ov.width = canvas.clientWidth;
    ov.height = canvas.clientHeight;
    parent.appendChild(ov);
    overlay.current = ov;
    const ctx = ov.getContext("2d");

    const redraw = () => {
      if (!ctx) return;
      ctx.clearRect(0, 0, ov.width, ov.height);
      if (points.current.length < 2) return;
      ctx.strokeStyle = "#000000";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(points.current[0][0], points.current[0][1]);
      for (const [x, y] of points.current.slice(1)) ctx.lineTo(x, y);
      ctx.stroke();
    };

    const onDown = (e: PointerEvent) => {
      drawing.current = true;
      const rect = canvas.getBoundingClientRect();
      points.current = [[e.clientX - rect.left, e.clientY - rect.top]];
      redraw();
    };
    const onMove = (e: PointerEvent) => {
      if (!drawing.current) return;
      const rect = canvas.getBoundingClientRect();
      points.current.push([e.clientX - rect.left, e.clientY - rect.top]);
      redraw();
    };
    const onUp = () => {
      if (!drawing.current) return;
      drawing.current = false;
      if (points.current.length < 3) {
        if (ctx) ctx.clearRect(0, 0, ov.width, ov.height);
        return;
      }

      // Collect triangle centroids from all meshes in the scene, project to
      // screen space, test polygon containment.
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      const selected = new Set<number>();
      let globalFaceOffset = 0;

      scene.traverse((obj) => {
        const mesh = obj as THREE.Mesh;
        if (!mesh.isMesh || !mesh.geometry) return;
        const geom = mesh.geometry as THREE.BufferGeometry;
        const pos = geom.getAttribute("position") as THREE.BufferAttribute | undefined;
        if (!pos) return;
        const index = geom.index;
        const faceCount = index ? Math.floor(index.count / 3) : Math.floor(pos.count / 3);
        const world = new THREE.Vector3();
        const a = new THREE.Vector3();
        const b = new THREE.Vector3();
        const c = new THREE.Vector3();
        for (let f = 0; f < faceCount; f++) {
          let ia: number, ib: number, ic: number;
          if (index) {
            ia = index.getX(f * 3);
            ib = index.getX(f * 3 + 1);
            ic = index.getX(f * 3 + 2);
          } else {
            ia = f * 3;
            ib = f * 3 + 1;
            ic = f * 3 + 2;
          }
          a.fromBufferAttribute(pos, ia);
          b.fromBufferAttribute(pos, ib);
          c.fromBufferAttribute(pos, ic);
          world
            .set((a.x + b.x + c.x) / 3, (a.y + b.y + c.y) / 3, (a.z + b.z + c.z) / 3)
            .applyMatrix4(mesh.matrixWorld);
          world.project(camera);
          const sx = (world.x * 0.5 + 0.5) * width;
          const sy = (-world.y * 0.5 + 0.5) * height;
          if (world.z > 1 || world.z < -1) continue;
          if (pointInPolygon(sx, sy, points.current)) {
            selected.add(globalFaceOffset + f);
          }
        }
        globalFaceOffset += faceCount;
      });

      setSelection(selected);
      if (ctx) ctx.clearRect(0, 0, ov.width, ov.height);
      points.current = [];
    };

    canvas.addEventListener("pointerdown", onDown);
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => {
      canvas.removeEventListener("pointerdown", onDown);
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      ov.remove();
    };
  }, [gl, scene, camera, setSelection]);

  return null;
}
