"use client";

import { useEffect } from "react";
import { useThree } from "@react-three/fiber";

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

  useEffect(() => {
    if (poses.length === 0) return;
    const latest = poses[poses.length - 1];
    const [tx, ty, tz] = latest.t;
    const [qx, qy, qz, qw] = latest.q;
    camera.position.set(tx, ty, tz);
    camera.quaternion.set(qx, qy, qz, qw);
    camera.updateMatrixWorld();
  }, [poses, camera]);

  return null;
}
