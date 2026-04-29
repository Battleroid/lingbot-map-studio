"use client";

import { Instance, Instances } from "@react-three/drei";
import { useMemo } from "react";
import * as THREE from "three";

import { useCaptureStore } from "@/lib/captureSession";

/**
 * Scaniverse-style coverage rendering. Each tracked sparse point is
 * voxelized into a `voxelSize` (default 10 cm) world-space bin by
 * the captureSession store. This component reads the resulting map
 * and emits one instanced cube per non-empty bin:
 *
 *   - count >= FULL_COUNT  → fully opaque, scene-tinted.
 *   - 0 < count < FULL     → translucent, opacity ∝ count.
 *
 * The instanced render keeps the GPU cost flat at ~one draw call
 * regardless of voxel count, which matters once a room-scale scan
 * accumulates 5000+ filled bins.
 *
 * The "wireframe placeholder for empty voxels along the current
 * frustum" idea from the plan is *not* implemented in v1 —
 * generating those voxels per-frame would need the camera
 * intrinsics + frustum culling math to land first. v1 settles for
 * "fill in" feedback (covered area shows up); empty area is just
 * empty.
 */

const FULL_COUNT = 8;
const MAX_INSTANCES = 5000;

export function CoverageVoxels() {
  const voxels = useCaptureStore((s) => s.voxels);
  const voxelSize = useCaptureStore((s) => s.voxelSize);

  const instances = useMemo(() => {
    if (voxels.size === 0) return [];
    const out: { pos: [number, number, number]; opacity: number }[] = [];
    for (const [key, count] of voxels) {
      const [ix, iy, iz] = key.split(",").map(Number);
      if (!Number.isFinite(ix)) continue;
      const pos: [number, number, number] = [
        (ix + 0.5) * voxelSize,
        (iy + 0.5) * voxelSize,
        (iz + 0.5) * voxelSize,
      ];
      const opacity = Math.max(
        0.15,
        Math.min(1.0, count / FULL_COUNT),
      );
      out.push({ pos, opacity });
      if (out.length >= MAX_INSTANCES) break;
    }
    return out;
  }, [voxels, voxelSize]);

  if (instances.length === 0) return null;

  return (
    <Instances limit={MAX_INSTANCES} range={instances.length}>
      <boxGeometry args={[voxelSize * 0.9, voxelSize * 0.9, voxelSize * 0.9]} />
      <meshBasicMaterial color="#00b86b" transparent depthWrite={false} />
      {instances.map((v, i) => (
        <Instance
          key={i}
          position={v.pos}
          // drei's <Instance> accepts an opacity-like alpha channel
          // via color; we encode opacity by darkening the green so
          // unfilled bins read as faded. (Per-instance opacity needs
          // a custom shader; keeping this simple for v1.)
          color={
            new THREE.Color("#00b86b").multiplyScalar(
              0.3 + v.opacity * 0.7,
            )
          }
        />
      ))}
    </Instances>
  );
}

/** Coverage stats consumed by the page chip: filled-voxel count +
 *  estimated coverage ratio over the trajectory bounding box. */
export function useCoverageSummary(): {
  filled: number;
  ratio: number;
} {
  const voxels = useCaptureStore((s) => s.voxels);
  const voxelSize = useCaptureStore((s) => s.voxelSize);
  const poses = useCaptureStore((s) => s.poses);

  return useMemo(() => {
    const filled = voxels.size;
    if (poses.length < 2) {
      return { filled, ratio: 0 };
    }
    // Rough bounding box of the trajectory + a 50% inflation for
    // the volume the user is plausibly trying to scan.
    let minX = Infinity,
      minY = Infinity,
      minZ = Infinity;
    let maxX = -Infinity,
      maxY = -Infinity,
      maxZ = -Infinity;
    for (const p of poses) {
      const [x, y, z] = p.t;
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (z < minZ) minZ = z;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
      if (z > maxZ) maxZ = z;
    }
    const sx = (maxX - minX) * 1.5;
    const sy = (maxY - minY) * 1.5;
    const sz = (maxZ - minZ) * 1.5;
    if (sx <= 0 || sy <= 0 || sz <= 0) return { filled, ratio: 0 };
    const expected =
      (sx / voxelSize) * (sy / voxelSize) * (sz / voxelSize);
    if (expected <= 0) return { filled, ratio: 0 };
    return { filled, ratio: Math.min(1, filled / expected) };
  }, [voxels, voxelSize, poses]);
}
