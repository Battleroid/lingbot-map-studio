"use client";

import { Instance, Instances, Line } from "@react-three/drei";
import { useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";

import { useViewerStore } from "@/lib/viewerStore";

export interface CameraPose {
  position: [number, number, number];
  quaternion: [number, number, number, number];
  /** Vertical FOV in degrees, derived from the frame's intrinsic + height. */
  fov_y_deg?: number;
}

interface Props {
  poses: CameraPose[];
  recordedFps: number;
}

/**
 * COLMAP/OpenCV → three.js rotation: 180° around the camera's local X axis
 * flips +Y (down) to -Y (up) and +Z (forward) to -Z (three.js forward).
 * Applied post-multiplicatively so the camera looks the right direction
 * during playback.
 */
const COLMAP_TO_THREE = new THREE.Quaternion().setFromAxisAngle(
  new THREE.Vector3(1, 0, 0),
  Math.PI,
);

// Matches upstream lingbot-map's frustum sizing from glb_export.py:
//   cam_width  = scene_scale * 0.05
//   cam_height = scene_scale * 0.1
// Relative to the 5-95 percentile extent of the filtered point cloud so
// frustums are always visible but never swamp the scene.
const FRUSTUM_WIDTH_FRACTION = 0.05;
const FRUSTUM_HEIGHT_FRACTION = 0.1;

export function CameraPath({ poses, recordedFps }: Props) {
  const showCameraPath = useViewerStore((s) => s.showCameraPath);
  const playing = useViewerStore((s) => s.playing);
  const playbackFrame = useViewerStore((s) => s.playbackFrame);
  const playbackSpeed = useViewerStore((s) => s.playbackSpeed);
  const setPlaybackFrame = useViewerStore((s) => s.setPlaybackFrame);
  const setPlaying = useViewerStore((s) => s.setPlaying);
  const setPathDiagonal = useViewerStore((s) => s.setPathDiagonal);
  const sceneScale = useViewerStore((s) => s.sceneScale);
  const camera = useThree((s) => s.camera);

  const positions = useMemo(
    () =>
      poses.map(
        (p) => new THREE.Vector3(p.position[0], p.position[1], p.position[2]),
      ),
    [poses],
  );

  const quaternions = useMemo(
    () =>
      poses.map((p) => {
        const q = new THREE.Quaternion(
          p.quaternion[0],
          p.quaternion[1],
          p.quaternion[2],
          p.quaternion[3],
        );
        // Same convention flip as the camera during playback — makes the
        // frustum visually point where the camera was actually looking.
        return q.multiply(COLMAP_TO_THREE.clone());
      }),
    [poses],
  );

  // Publish path diagonal for fly-speed scaling.
  const pathDiag = useMemo(() => {
    if (positions.length < 2) return 0;
    const box = new THREE.Box3().setFromPoints(positions);
    const size = new THREE.Vector3();
    box.getSize(size);
    const d = size.length();
    return Number.isFinite(d) && d > 0 ? d : 0;
  }, [positions]);
  useEffect(() => {
    setPathDiagonal(pathDiag);
  }, [pathDiag, setPathDiagonal]);

  // Cone frustum dimensions — absolute world units derived from scene_scale.
  const camWidth = Math.max(0.001, sceneScale * FRUSTUM_WIDTH_FRACTION);
  const camHeight = Math.max(0.001, sceneScale * FRUSTUM_HEIGHT_FRACTION);

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

  // Interpolated current pose (used to drive the playback camera).
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
    return qa.clone().slerp(qb, t).multiply(COLMAP_TO_THREE);
  }, [a, b, t]);

  const savedFovRef = useRef<number | null>(null);

  useEffect(() => {
    if (!playing || !a || !b) return;
    camera.position.copy(currentPos);
    camera.quaternion.copy(currentQuat);

    if (!("fov" in camera)) return;
    const persp = camera as unknown as {
      fov: number;
      updateProjectionMatrix: () => void;
    };
    if (savedFovRef.current === null) savedFovRef.current = persp.fov;
    const fov = a.fov_y_deg;
    if (typeof fov === "number" && Number.isFinite(fov) && fov > 5 && fov < 170) {
      if (Math.abs(persp.fov - fov) > 0.1) {
        persp.fov = fov;
        persp.updateProjectionMatrix();
      }
    }
  }, [playing, camera, currentPos, currentQuat, a, b]);

  useEffect(() => {
    return () => {
      if (savedFovRef.current === null) return;
      if (!("fov" in camera)) return;
      const persp = camera as unknown as {
        fov: number;
        updateProjectionMatrix: () => void;
      };
      persp.fov = savedFovRef.current;
      persp.updateProjectionMatrix();
      savedFovRef.current = null;
    };
  }, [camera]);

  useEffect(() => {
    if (playing) return;
    if (savedFovRef.current === null) return;
    if (!("fov" in camera)) return;
    const persp = camera as unknown as {
      fov: number;
      updateProjectionMatrix: () => void;
    };
    persp.fov = savedFovRef.current;
    persp.updateProjectionMatrix();
    savedFovRef.current = null;
  }, [playing, camera]);

  if (!showCameraPath || positions.length < 2) return null;

  const currentIdx = Math.round(clamped);

  return (
    <group>
      {/* Path polyline connecting the camera centers in order. */}
      <Line
        points={positions}
        color="#000000"
        lineWidth={2}
        dashed={false}
        transparent
        opacity={0.5}
      />

      {/* A small frustum cone at every recorded pose. Instanced for perf —
          250+ cones render as a single draw call. */}
      <Instances limit={positions.length} range={positions.length}>
        {/* three.js cone default axis is +Y, apex at top. Rotate geometry
            baking (rotation-in-args) so +Z is the "forward" axis, matching
            the camera quaternion conventions. Because we premultiplied
            COLMAP_TO_THREE above, the cone's apex ends up at the camera
            center and its base in the view direction. */}
        <coneGeometry args={[camWidth, camHeight, 4, 1, true]} />
        <meshBasicMaterial
          color="#000000"
          wireframe
          transparent
          opacity={0.55}
        />
        {positions.map((pos, i) => (
          <PoseInstance
            key={i}
            position={pos}
            quaternion={quaternions[i]}
            coneHeight={camHeight}
            isCurrent={i === currentIdx}
          />
        ))}
      </Instances>

      {/* Solid "current playback pose" marker so the playhead stands out
          from the wireframe-gray frustums. */}
      <group position={currentPos} quaternion={currentQuat}>
        <mesh position={[0, 0, -camHeight / 2]} rotation={[Math.PI / 2, 0, 0]}>
          <coneGeometry args={[camWidth * 1.1, camHeight * 1.1, 4]} />
          <meshBasicMaterial color="#000000" transparent opacity={0.95} />
        </mesh>
      </group>
    </group>
  );
}

/** One instanced cone at a pose. `quaternion` already includes the
 *  COLMAP→three.js flip. We also nudge the cone so its apex sits at the
 *  camera origin and the base points forward (toward what the camera
 *  was looking at). */
function PoseInstance({
  position,
  quaternion,
  coneHeight,
  isCurrent,
}: {
  position: THREE.Vector3;
  quaternion: THREE.Quaternion;
  coneHeight: number;
  isCurrent: boolean;
}) {
  // drei <Instance> accepts position, rotation, scale, color. We pre-rotate
  // the cone so its +Y axis aligns with the camera's -Z (forward), then
  // offset it so the apex sits on the camera origin.
  const finalQuat = useMemo(() => {
    // Cone geometry: apex +Y, base -Y. We want apex at origin, base
    // along camera forward (camera local -Z after the flip). Rotate the
    // cone 90° about X so +Y becomes -Z.
    const alignCone = new THREE.Quaternion().setFromAxisAngle(
      new THREE.Vector3(1, 0, 0),
      -Math.PI / 2,
    );
    return quaternion.clone().multiply(alignCone);
  }, [quaternion]);

  const offset = useMemo(() => {
    // Move cone so apex lands at `position` instead of center.
    const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(quaternion);
    return forward.multiplyScalar(coneHeight / 2).add(position);
  }, [position, quaternion, coneHeight]);

  return (
    <Instance
      position={offset}
      quaternion={finalQuat}
      color={isCurrent ? "#000000" : "#555555"}
    />
  );
}
