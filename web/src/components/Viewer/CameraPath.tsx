"use client";

import { Line } from "@react-three/drei";
import { useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo } from "react";
import * as THREE from "three";

import { useViewerStore } from "@/lib/viewerStore";

export interface CameraPose {
  position: [number, number, number];
  quaternion: [number, number, number, number];
}

interface Props {
  poses: CameraPose[];
  recordedFps: number;
}

/**
 * lingbot-map stores camera rotations in COLMAP/OpenCV convention
 * (+X right, +Y down, +Z forward). three.js uses the OpenGL convention
 * (+X right, +Y up, −Z forward). Post-multiplying by a 180° rotation
 * around the camera's local X flips Y and Z → correct orientation.
 */
const COLMAP_TO_THREE = new THREE.Quaternion().setFromAxisAngle(
  new THREE.Vector3(1, 0, 0),
  Math.PI,
);

/**
 * Renders the recorded camera trajectory as a polyline + a small marker at
 * the current playback position, and when playback is active drives the
 * default camera through the poses using the recorded FPS (modulated by a
 * user speed multiplier).
 */
export function CameraPath({ poses, recordedFps }: Props) {
  const showCameraPath = useViewerStore((s) => s.showCameraPath);
  const playing = useViewerStore((s) => s.playing);
  const playbackFrame = useViewerStore((s) => s.playbackFrame);
  const playbackSpeed = useViewerStore((s) => s.playbackSpeed);
  const setPlaybackFrame = useViewerStore((s) => s.setPlaybackFrame);
  const setPlaying = useViewerStore((s) => s.setPlaying);
  const camera = useThree((s) => s.camera);

  const positions = useMemo(
    () =>
      poses.map(
        (p) => new THREE.Vector3(p.position[0], p.position[1], p.position[2]),
      ),
    [poses],
  );

  // Scene-relative marker size so the arrow is visible at any scale.
  const markerSize = useMemo(() => {
    if (positions.length < 2) return 0.05;
    const box = new THREE.Box3().setFromPoints(positions);
    const size = new THREE.Vector3();
    box.getSize(size);
    return Math.max(0.02, size.length() * 0.015);
  }, [positions]);

  useFrame((_state, delta) => {
    if (!playing || poses.length < 2) return;
    const fps = Math.max(1, recordedFps) * Math.max(0.1, playbackSpeed);
    const next = playbackFrame + delta * fps;
    if (next >= poses.length - 1) {
      setPlaybackFrame(poses.length - 1);
      setPlaying(false);
    } else {
      setPlaybackFrame(next);
    }
  });

  // Interpolated current pose → camera + marker.
  const clamped = Math.max(0, Math.min(poses.length - 1, playbackFrame));
  const lo = Math.floor(clamped);
  const hi = Math.min(poses.length - 1, lo + 1);
  const t = clamped - lo;
  const a = poses[lo];
  const b = poses[hi];

  const currentPos = useMemo(() => {
    if (!a || !b) return new THREE.Vector3();
    return new THREE.Vector3(
      a.position[0] + (b.position[0] - a.position[0]) * t,
      a.position[1] + (b.position[1] - a.position[1]) * t,
      a.position[2] + (b.position[2] - a.position[2]) * t,
    );
  }, [a, b, t]);

  const currentQuat = useMemo(() => {
    if (!a || !b) return new THREE.Quaternion();
    const qa = new THREE.Quaternion(
      a.quaternion[0],
      a.quaternion[1],
      a.quaternion[2],
      a.quaternion[3],
    );
    const qb = new THREE.Quaternion(
      b.quaternion[0],
      b.quaternion[1],
      b.quaternion[2],
      b.quaternion[3],
    );
    // slerp then apply the COLMAP → three.js axis flip.
    return qa.clone().slerp(qb, t).multiply(COLMAP_TO_THREE);
  }, [a, b, t]);

  useEffect(() => {
    if (!playing || !a || !b) return;
    camera.position.copy(currentPos);
    camera.quaternion.copy(currentQuat);
  }, [playing, camera, currentPos, currentQuat, a, b]);

  if (!showCameraPath || positions.length < 2) return null;
  return (
    <group>
      <Line
        points={positions}
        color="#000000"
        lineWidth={2}
        dashed={false}
        transparent
        opacity={0.65}
      />
      {/* Current-pose marker: a cone that points along the camera's forward
          axis. three.js cone default axis is +Y; the camera's forward is -Z,
          so we rotate the cone to align its tip with the camera forward. */}
      <group position={currentPos} quaternion={currentQuat}>
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <coneGeometry args={[markerSize * 0.4, markerSize, 12]} />
          <meshBasicMaterial color="#000000" transparent opacity={0.85} />
        </mesh>
      </group>
    </group>
  );
}
