"use client";

import { useEffect, useMemo } from "react";
import { useThree } from "@react-three/fiber";
import * as THREE from "three";

import { useCaptureStore } from "@/lib/captureSession";

/**
 * Drives the parent Canvas's camera from the latest SLAM pose so the
 * scene is rendered from the user's *current* phone viewpoint, not a
 * fixed orbit.
 *
 * Used in two places:
 *   1. The capture page's PiP canvas, where it gives the user a "this
 *      is the splat from where I'm pointing right now" preview.
 *   2. The full-screen AR overlay on the camera view, where pose-
 *      aligned rendering is the entire point — the diagonal-stripe
 *      and translucent-red voxel cubes need to project from the
 *      phone's camera basis onto the live video feed for the
 *      "captured / uncaptured" Scaniverse-style cue to make sense.
 *
 * Coordinate convention. The SLAM tracker emits poses in **OpenCV**
 * convention: camera-local +X right, +Y *down*, +Z *forward*. Three.js
 * uses **OpenGL** convention: +X right, +Y up, *-Z* forward. A
 * verbatim quaternion copy from one to the other makes three.js look
 * 180° away from the scene — points placed in front of the OpenCV
 * camera (at world coords corresponding to the camera's +Z) end up
 * behind the OpenGL camera and aren't rendered. The fix is a single
 * π-around-camera-local-X post-multiply, which flips Y (down ↔ up)
 * and Z (forward ↔ backward) together.
 *
 * The camera writes happen in a `useEffect` on the latest pose
 * (~10 Hz) rather than `useFrame` (~60 Hz). useFrame would burn GPU
 * for no benefit during the SLAM gaps; the effect-on-pose-change
 * pattern is enough.
 *
 * Skips the first paint when no poses have arrived yet — leaves
 * whatever default camera the Canvas was mounted with so the canvas
 * isn't an empty/black hole during warmup.
 */
export function FollowPoseCamera() {
  const poses = useCaptureStore((s) => s.poses);
  const camera = useThree((s) => s.camera);

  // π-around-X. Allocated once and reused so we're not GC-churning
  // a Quaternion per pose update. xyzw form: x=1, y=0, z=0, w=0
  // encodes a 180° rotation around the camera-local X axis.
  const opencvToOpengl = useMemo(
    () => new THREE.Quaternion(1, 0, 0, 0),
    [],
  );
  const tmp = useMemo(() => new THREE.Quaternion(), []);

  useEffect(() => {
    if (poses.length === 0) return;
    const latest = poses[poses.length - 1];
    const [tx, ty, tz] = latest.t;
    const [qx, qy, qz, qw] = latest.q;
    camera.position.set(tx, ty, tz);
    // Apply the OpenCV→OpenGL flip on top of the SLAM quaternion. The
    // post-multiply order is "first rotate 180° around the camera's
    // own X axis, then apply the SLAM orientation" — net effect is
    // the camera ends up looking the same world direction the
    // OpenCV camera was looking, just with the OpenGL frame.
    tmp.set(qx, qy, qz, qw).multiply(opencvToOpengl);
    camera.quaternion.copy(tmp);
    camera.updateMatrixWorld();
  }, [poses, camera, opencvToOpengl, tmp]);

  return null;
}
