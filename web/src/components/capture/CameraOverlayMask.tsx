"use client";

import { useEffect, useMemo, useRef } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import * as THREE from "three";

import { useCaptureStore } from "@/lib/captureSession";

/**
 * Scaniverse-style coverage mask, screen-space.
 *
 * Visual: the camera feed is the underlay. On top, an alpha layer
 * paints a red+white candy-stripe pattern over screen regions whose
 * view direction the SLAM tracker hasn't observed yet. As the user
 * pans into striped regions the corresponding az/el bucket flips to
 * "covered" and the stripes peel back, revealing the camera feed.
 *
 * Why screen-space + view-direction (not world-space + projected
 * voxels): without per-pixel depth (no LIDAR, no stereo) we can't
 * project screen pixels onto real surfaces. The visually-equivalent
 * cheap approximation is "have I pointed at this direction yet?",
 * which collapses with real-space coverage at a single typical
 * scanning distance from the subject. Looks like Scaniverse without
 * needing depth.
 *
 * Pipeline:
 *   1. The capture session already maintains a `viewCoverage`
 *      Uint16Array (36 az × 18 el) that the WS handler bumps each
 *      time a pose lands in a bucket. We upload it to the GPU as a
 *      36×18 R8 DataTexture, refreshed when the array reference
 *      changes.
 *   2. A fullscreen ortho quad's fragment shader maps each screen
 *      UV → camera-relative ray → world ray (rotated by the live
 *      pose quaternion, with the same OpenCV→OpenGL flip
 *      `FollowPoseCamera` applies) → az/el → texture lookup.
 *   3. Covered → transparent. Uncovered → candy-stripe with a slight
 *      darken of the camera feed underneath.
 */
export function CameraOverlayMask() {
  return (
    <Canvas
      // Ortho + a single fullscreen quad — no scene depth, no
      // perspective. The fragment shader does all the work.
      orthographic
      camera={{ position: [0, 0, 1], near: 0, far: 2, zoom: 1 }}
      gl={{ alpha: true, premultipliedAlpha: false, antialias: false }}
      style={{
        position: "absolute",
        inset: 0,
        background: "transparent",
        pointerEvents: "none",
      }}
    >
      <CoverageMaskQuad />
    </Canvas>
  );
}

function CoverageMaskQuad() {
  const coverage = useCaptureStore((s) => s.viewCoverage);
  const poses = useCaptureStore((s) => s.poses);
  const size = useThree((s) => s.size);

  // Build the 36×18 R8 texture once + reuse the same instance. We
  // rewrite the underlying Uint8Array in-place when coverage updates,
  // rather than allocating a fresh DataTexture, to keep the GPU
  // upload cheap on phones.
  const coverageTex = useMemo(() => {
    const data = new Uint8Array(36 * 18);
    const tex = new THREE.DataTexture(
      data,
      36,
      18,
      THREE.RedFormat,
      THREE.UnsignedByteType,
    );
    tex.minFilter = THREE.NearestFilter;
    tex.magFilter = THREE.NearestFilter;
    tex.wrapS = THREE.RepeatWrapping; // azimuth wraps 0/360
    tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.needsUpdate = true;
    return tex;
  }, []);

  // Rewrite the texture pixels when the store's coverage array
  // changes. Coverage is Uint16; clamp to 1 = "covered" (any non-zero
  // count counts as observed) before writing to the R8 texture.
  useEffect(() => {
    const dst = coverageTex.image.data as Uint8Array;
    for (let i = 0; i < 36 * 18 && i < coverage.length; i++) {
      dst[i] = coverage[i] > 0 ? 255 : 0;
    }
    coverageTex.needsUpdate = true;
  }, [coverage, coverageTex]);

  // Material with the latest pose + screen size + coverage texture
  // wired in as uniforms. We bump the uniform values per render
  // rather than rebuilding the material so the shader compiles once.
  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        transparent: true,
        depthWrite: false,
        depthTest: false,
        uniforms: {
          uCoverage: { value: coverageTex },
          // FOV: matches the simulated SLAM tracker's intrinsic
          // assumption (60° HFOV → fx = w * 0.866). We pass the
          // tangent of the half-angles so the shader can build
          // pixel rays without trig.
          uFovTan: { value: new THREE.Vector2(Math.tan((60 * Math.PI) / 360), 0) },
          uPoseQuat: { value: new THREE.Vector4(0, 0, 0, 1) },
          uHasPose: { value: 0 },
        },
        vertexShader: VERTEX,
        fragmentShader: FRAGMENT,
      }),
    [coverageTex],
  );

  // Push live updates: pose + aspect-driven vertical FOV.
  const matRef = useRef(material);
  matRef.current = material;

  useEffect(() => {
    // VFOV from the canvas aspect ratio. With a 60° HFOV a 9:16
    // portrait phone gives ~92° VFOV; landscape webcam gives ~36°.
    // (tan(VFOV/2) = tan(HFOV/2) * (h / w))
    const tanH = Math.tan((60 * Math.PI) / 360);
    const tanV = tanH * (size.height / Math.max(1, size.width));
    material.uniforms.uFovTan.value.set(tanH, tanV);
  }, [size.width, size.height, material]);

  useEffect(() => {
    if (poses.length === 0) {
      material.uniforms.uHasPose.value = 0;
      return;
    }
    const latest = poses[poses.length - 1];
    const [qx, qy, qz, qw] = latest.q;
    // Apply the OpenCV→OpenGL camera-local 180°-X flip so the ray
    // we computed in OpenGL convention maps to the same world
    // direction the SLAM camera was looking. Same flip
    // `FollowPoseCamera` applies. We do it here on the CPU
    // (one quat multiply per pose update) rather than in the
    // shader to keep the per-pixel work minimal.
    const q = new THREE.Quaternion(qx, qy, qz, qw).multiply(
      new THREE.Quaternion(1, 0, 0, 0),
    );
    material.uniforms.uPoseQuat.value.set(q.x, q.y, q.z, q.w);
    material.uniforms.uHasPose.value = 1;
  }, [poses, material]);

  return (
    <mesh material={material}>
      {/* PlaneGeometry exactly fills the ortho frustum (-1..1 on
       *  both axes by default for r3f's ortho). Each vertex carries
       *  a (0..1, 0..1) UV the fragment shader reads as screen
       *  position. */}
      <planeGeometry args={[2, 2]} />
    </mesh>
  );
}

const VERTEX = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

// World-direction lookup against a 36×18 R8 coverage texture, with
// candy-stripe + slight darken-under for uncovered pixels. ~30 alu
// ops per pixel; trivial on a phone GPU.
const FRAGMENT = /* glsl */ `
precision mediump float;

uniform sampler2D uCoverage;
uniform vec2 uFovTan;       // (tan(HFOV/2), tan(VFOV/2))
uniform vec4 uPoseQuat;     // world-from-camera, OpenGL frame
uniform float uHasPose;     // 0 until first pose; 1 after
varying vec2 vUv;

const float PI = 3.14159265359;
const float TAU = 6.28318530718;

vec3 applyQuat(vec4 q, vec3 v) {
  // v + 2 * q.xyz × (q.xyz × v + q.w * v)
  vec3 t = 2.0 * cross(q.xyz, v);
  return v + q.w * t + cross(q.xyz, t);
}

void main() {
  // Before the first pose lands, paint the whole screen with the
  // stripe pattern so the user has the same "you haven't started
  // yet" visual cue Scaniverse shows.
  bool covered = false;
  if (uHasPose > 0.5) {
    // Camera-local ray. UV (0,0) = bottom-left, (1,1) = top-right
    // (three.js convention). Map to (-1..1) on each axis, scale by
    // the FOV tangents, look forward (-Z in OpenGL).
    vec3 rayCam = normalize(vec3(
      (vUv.x - 0.5) * 2.0 * uFovTan.x,
      (vUv.y - 0.5) * 2.0 * uFovTan.y,
      -1.0
    ));
    vec3 rayWorld = applyQuat(uPoseQuat, rayCam);
    // Convert to az [0, 2π) and el [-π/2, π/2].
    float az = atan(rayWorld.x, -rayWorld.z);
    if (az < 0.0) az += TAU;
    float el = asin(clamp(rayWorld.y, -1.0, 1.0));
    vec2 uvCov = vec2(az / TAU, (el + 0.5 * PI) / PI);
    covered = texture2D(uCoverage, uvCov).r > 0.5;
  }

  if (covered) {
    discard;
  }

  // Candy-stripe in screen space. ~6 stripes across, 45° angle.
  // Slight darken-under via a 0.55 alpha so the camera feed shows
  // through with reduced saturation — same "this isn't ready yet"
  // affordance Scaniverse uses.
  float stripe = step(0.5, mod((vUv.x + vUv.y) * 6.0, 1.0));
  vec3 col = mix(vec3(0.92, 0.18, 0.27), vec3(0.97, 0.97, 0.97), stripe);
  gl_FragColor = vec4(col, 0.55);
}
`;
