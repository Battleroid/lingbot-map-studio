"use client";

import { Canvas } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";

import { useCaptureStore } from "@/lib/captureSession";

import { FollowPoseCamera } from "./FollowPoseCamera";

/**
 * Scaniverse-style coverage overlay rendered on top of the live
 * camera feed.
 *
 * The full-screen video element underneath shows the phone camera.
 * This component overlays a transparent three.js Canvas with its
 * camera locked to the live SLAM pose; the only thing it draws is
 * the voxel coverage grid:
 *
 *   - **Covered cells** (any tracked sparse point landed in the
 *     bin) → a diagonal black/white stripe pattern, rendered in
 *     world space so the stripes appear stuck to the surfaces the
 *     SLAM tracker has observed.
 *   - **Adjacent-shell cells** (one neighbour of a covered cell,
 *     not itself covered) → translucent red. As the user pans into
 *     a red region the SLAM tracker fills the bin and it flips to
 *     the diagonal pattern — same loop as Scaniverse's
 *     "captured / uncaptured" UX.
 *
 * Cells beyond the immediate shell aren't rendered, which keeps the
 * draw count bounded at ~7× the captured-bin count regardless of
 * how big the expected scene volume gets. Without that cap a room-
 * scale scan would try to render hundreds of thousands of empty
 * cells and crater the phone's GPU.
 *
 * Alignment caveat: we use a 60° HFOV approximation matching the
 * simulated tracker's intrinsic assumption (`fx = w * 0.866`) — not
 * real lens calibration. The overlay will be self-consistent with
 * the SLAM coordinate system but slightly misaligned with the
 * actual camera image (typically a few degrees of yaw/pitch for a
 * phone wider than 60° HFOV). Doing pixel-aligned AR needs a
 * calibration step on first use; that's a separate effort.
 */

// Per-layer hard cap — protects the phone GPU from a runaway scan.
// At 0.1 m voxels covering a 4 × 4 × 2 m room, a fully-covered scene
// is ~16 K bins; the shell adds ~3× more. 50K per layer keeps the
// instanced draw call comfortably inside what mobile browsers can
// chew through at 30+ fps.
const MAX_INSTANCES_PER_LAYER = 50_000;

const NEIGHBOURS_6: ReadonlyArray<readonly [number, number, number]> = [
  [1, 0, 0],
  [-1, 0, 0],
  [0, 1, 0],
  [0, -1, 0],
  [0, 0, 1],
  [0, 0, -1],
];

/** Top-level AR overlay. Mount inside the video pane as a sibling
 *  of the `<video>` element. The Canvas is alpha + non-interactive
 *  so the camera feed shows through and pinch / orbit gestures
 *  still hit the phone's native UI. */
export function CameraOverlayAR() {
  return (
    <Canvas
      gl={{ alpha: true, premultipliedAlpha: false, antialias: false }}
      camera={{
        position: [0, 0, 0],
        // 60° HFOV → roughly 46° VFOV at a 4:3 aspect; r3f's `fov`
        // is vertical. Matches the simulated SLAM tracker's
        // assumption so the overlay is internally consistent with
        // the SLAM coordinate frame.
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
      <CoverageARScene />
    </Canvas>
  );
}

function CoverageARScene() {
  const voxels = useCaptureStore((s) => s.voxels);
  const voxelSize = useCaptureStore((s) => s.voxelSize);

  const { covered, shell } = useMemo(() => {
    if (voxels.size === 0) {
      return { covered: new Float32Array(0), shell: new Float32Array(0) };
    }
    const coveredKeys: string[] = [];
    for (const [k, c] of voxels) {
      if (c > 0) coveredKeys.push(k);
    }
    // Build a Set of adjacent-but-not-covered cells. Bound the iter
    // count so an absurdly large covered set doesn't pin the main
    // thread — drop the tail rather than hang the UI.
    const shellSet = new Set<string>();
    const limit = Math.min(coveredKeys.length, MAX_INSTANCES_PER_LAYER);
    for (let i = 0; i < limit; i++) {
      const key = coveredKeys[i];
      const parts = key.split(",");
      const x = Number(parts[0]);
      const y = Number(parts[1]);
      const z = Number(parts[2]);
      if (!Number.isFinite(x)) continue;
      for (const [dx, dy, dz] of NEIGHBOURS_6) {
        const nKey = `${x + dx},${y + dy},${z + dz}`;
        if ((voxels.get(nKey) ?? 0) === 0) shellSet.add(nKey);
        if (shellSet.size >= MAX_INSTANCES_PER_LAYER) break;
      }
      if (shellSet.size >= MAX_INSTANCES_PER_LAYER) break;
    }

    const coveredOut = new Float32Array(limit * 3);
    for (let i = 0; i < limit; i++) {
      const parts = coveredKeys[i].split(",");
      const x = Number(parts[0]);
      const y = Number(parts[1]);
      const z = Number(parts[2]);
      coveredOut[i * 3] = (x + 0.5) * voxelSize;
      coveredOut[i * 3 + 1] = (y + 0.5) * voxelSize;
      coveredOut[i * 3 + 2] = (z + 0.5) * voxelSize;
    }
    const shellOut = new Float32Array(shellSet.size * 3);
    let si = 0;
    for (const k of shellSet) {
      const parts = k.split(",");
      const x = Number(parts[0]);
      const y = Number(parts[1]);
      const z = Number(parts[2]);
      shellOut[si] = (x + 0.5) * voxelSize;
      shellOut[si + 1] = (y + 0.5) * voxelSize;
      shellOut[si + 2] = (z + 0.5) * voxelSize;
      si += 3;
    }
    return { covered: coveredOut, shell: shellOut };
  }, [voxels, voxelSize]);

  return (
    <>
      <CoveredCells positions={covered} size={voxelSize} />
      <UncoveredShell positions={shell} size={voxelSize} />
    </>
  );
}

/** Instanced cubes for the covered set, rendered with a world-space
 *  diagonal-stripe shader so the pattern feels "stuck" to the
 *  surfaces the SLAM tracker has observed. */
function CoveredCells({
  positions,
  size,
}: {
  positions: Float32Array;
  size: number;
}) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const count = positions.length / 3;

  // Shader is hoisted via useMemo so it's not rebuilt every render.
  // World-space stripes: pick the dominant face-normal axis, then
  // stripe in the perpendicular world plane. 30 stripes per metre
  // works well at the 0.1m default voxel size — ~3 stripes per cube
  // face, readable without flickering when the user pans.
  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        transparent: true,
        depthWrite: false,
        side: THREE.DoubleSide,
        vertexShader: STRIPE_VERTEX,
        fragmentShader: STRIPE_FRAGMENT,
      }),
    [],
  );

  useEffect(
    () => () => {
      material.dispose();
    },
    [material],
  );

  useEffect(() => {
    const m = meshRef.current;
    if (!m) return;
    const tmp = new THREE.Object3D();
    for (let i = 0; i < count; i++) {
      tmp.position.set(
        positions[i * 3],
        positions[i * 3 + 1],
        positions[i * 3 + 2],
      );
      tmp.updateMatrix();
      m.setMatrixAt(i, tmp.matrix);
    }
    m.count = count;
    m.instanceMatrix.needsUpdate = true;
  }, [positions, count]);

  if (count === 0) return null;
  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, material, MAX_INSTANCES_PER_LAYER]}
      frustumCulled={false}
    >
      <boxGeometry args={[size * 0.95, size * 0.95, size * 0.95]} />
    </instancedMesh>
  );
}

/** Instanced cubes for the adjacent uncovered shell — the "you
 *  haven't pointed there yet" cue. Translucent red, no shader work
 *  needed beyond a basic transparent material. */
function UncoveredShell({
  positions,
  size,
}: {
  positions: Float32Array;
  size: number;
}) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const count = positions.length / 3;

  const material = useMemo(
    () =>
      new THREE.MeshBasicMaterial({
        color: 0xe8593a,
        transparent: true,
        opacity: 0.42,
        depthWrite: false,
      }),
    [],
  );

  useEffect(
    () => () => {
      material.dispose();
    },
    [material],
  );

  useEffect(() => {
    const m = meshRef.current;
    if (!m) return;
    const tmp = new THREE.Object3D();
    for (let i = 0; i < count; i++) {
      tmp.position.set(
        positions[i * 3],
        positions[i * 3 + 1],
        positions[i * 3 + 2],
      );
      tmp.updateMatrix();
      m.setMatrixAt(i, tmp.matrix);
    }
    m.count = count;
    m.instanceMatrix.needsUpdate = true;
  }, [positions, count]);

  if (count === 0) return null;
  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, material, MAX_INSTANCES_PER_LAYER]}
      frustumCulled={false}
    >
      <boxGeometry args={[size * 0.95, size * 0.95, size * 0.95]} />
    </instancedMesh>
  );
}

// ── Diagonal-stripe shader ──────────────────────────────────────
// World-space stripes so the pattern reads as "applied to the
// surface" rather than "scrolling across the screen" — that's
// what makes the captured-vs-uncaptured cue match Scaniverse's
// visual language.

const STRIPE_VERTEX = /* glsl */ `
varying vec3 vWorldPos;
varying vec3 vNormal;

void main() {
  vec4 wp = modelMatrix * instanceMatrix * vec4(position, 1.0);
  vWorldPos = wp.xyz;
  vec4 wn = modelMatrix * instanceMatrix * vec4(normal, 0.0);
  vNormal = normalize(wn.xyz);
  gl_Position = projectionMatrix * viewMatrix * wp;
}
`;

const STRIPE_FRAGMENT = /* glsl */ `
precision mediump float;
varying vec3 vWorldPos;
varying vec3 vNormal;

void main() {
  // Pick the world plane parallel to this face (i.e. the two axes
  // *not* aligned with the dominant normal direction) and stripe
  // diagonally in that plane. This makes the stripe orientation
  // consistent across all six cube faces.
  vec3 absN = abs(vNormal);
  vec2 uv;
  if (absN.x > absN.y && absN.x > absN.z) {
    uv = vWorldPos.yz;
  } else if (absN.y > absN.z) {
    uv = vWorldPos.xz;
  } else {
    uv = vWorldPos.xy;
  }
  float t = mod((uv.x + uv.y) * 30.0, 1.0);
  float stripe = step(0.5, t);
  vec3 col = mix(vec3(0.06), vec3(0.92), stripe);
  gl_FragColor = vec4(col, 0.78);
}
`;
