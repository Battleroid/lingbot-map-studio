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
 * Renders the recorded camera trajectory as a polyline and, when playback
 * is active, drives the R3F default camera through the poses using the
 * recorded FPS (modulated by a user speed multiplier).
 */
export function CameraPath({ poses, recordedFps }: Props) {
  const showCameraPath = useViewerStore((s) => s.showCameraPath);
  const playing = useViewerStore((s) => s.playing);
  const playbackFrame = useViewerStore((s) => s.playbackFrame);
  const playbackSpeed = useViewerStore((s) => s.playbackSpeed);
  const setPlaybackFrame = useViewerStore((s) => s.setPlaybackFrame);
  const setPlaying = useViewerStore((s) => s.setPlaying);
  const camera = useThree((s) => s.camera);

  // Accumulator across frames so we can do sub-frame interpolation on the
  // frame index without rounding errors from multiplying delta * speed * fps.
  const positions = useMemo(
    () =>
      poses.map(
        (p) => new THREE.Vector3(p.position[0], p.position[1], p.position[2]),
      ),
    [poses],
  );

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

  // Apply the current playback pose to the camera. Linear interp between
  // integer frames keeps motion smooth when playbackSpeed is low.
  useEffect(() => {
    if (!poses.length) return;
    const clamped = Math.max(0, Math.min(poses.length - 1, playbackFrame));
    const lo = Math.floor(clamped);
    const hi = Math.min(poses.length - 1, lo + 1);
    const t = clamped - lo;
    if (!playing && !Number.isFinite(clamped)) return;
    const a = poses[lo];
    const b = poses[hi];
    if (!a || !b) return;
    camera.position.set(
      a.position[0] + (b.position[0] - a.position[0]) * t,
      a.position[1] + (b.position[1] - a.position[1]) * t,
      a.position[2] + (b.position[2] - a.position[2]) * t,
    );
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
    camera.quaternion.copy(qa).slerp(qb, t);
  }, [poses, playbackFrame, playing, camera]);

  if (!showCameraPath || positions.length < 2) return null;
  return (
    <Line
      points={positions}
      color="#000000"
      lineWidth={1}
      dashed={false}
      transparent
      opacity={0.6}
    />
  );
}
