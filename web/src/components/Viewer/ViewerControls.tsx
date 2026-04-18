"use client";

import { Tip } from "@/components/Tip";
import { type RenderMode, useViewerStore } from "@/lib/viewerStore";

const MODES: { value: RenderMode; label: string; tip: string }[] = [
  { value: "mesh", label: "mesh", tip: "Loaded GLB rendered as solid triangles." },
  { value: "wireframe", label: "wire", tip: "Same GLB rendered with wireframe-only materials." },
  {
    value: "points",
    label: "points",
    tip: "PLY point cloud in a single foreground color. Fastest mode.",
  },
  {
    value: "points-color",
    label: "color pts",
    tip: "PLY point cloud with per-vertex colors from the source frames.",
  },
];

interface Props {
  /** Total poses in the recorded camera path. 0 disables playback UI. */
  pathPoseCount?: number;
}

export function ViewerControls({ pathPoseCount = 0 }: Props) {
  const mode = useViewerStore((s) => s.mode);
  const setMode = useViewerStore((s) => s.setMode);
  const cameraMode = useViewerStore((s) => s.cameraMode);
  const setCameraMode = useViewerStore((s) => s.setCameraMode);
  const pointSizeScale = useViewerStore((s) => s.pointSizeScale);
  const setPointSizeScale = useViewerStore((s) => s.setPointSizeScale);
  const lassoActive = useViewerStore((s) => s.lassoActive);
  const setLassoActive = useViewerStore((s) => s.setLassoActive);
  const selection = useViewerStore((s) => s.selection);
  const clearSelection = useViewerStore((s) => s.clearSelection);
  const requestRefit = useViewerStore((s) => s.requestRefit);
  const meshHistory = useViewerStore((s) => s.meshHistory);
  const meshHistoryIndex = useViewerStore((s) => s.meshHistoryIndex);
  const undo = useViewerStore((s) => s.undo);
  const redo = useViewerStore((s) => s.redo);
  const showCameraPath = useViewerStore((s) => s.showCameraPath);
  const setShowCameraPath = useViewerStore((s) => s.setShowCameraPath);
  const playing = useViewerStore((s) => s.playing);
  const setPlaying = useViewerStore((s) => s.setPlaying);
  const playbackFrame = useViewerStore((s) => s.playbackFrame);
  const setPlaybackFrame = useViewerStore((s) => s.setPlaybackFrame);
  const playbackSpeed = useViewerStore((s) => s.playbackSpeed);
  const setPlaybackSpeed = useViewerStore((s) => s.setPlaybackSpeed);
  const flySpeedMult = useViewerStore((s) => s.flySpeedMult);
  const setFlySpeedMult = useViewerStore((s) => s.setFlySpeedMult);
  const sceneDiagonal = useViewerStore((s) => s.sceneDiagonal);
  const pathDiagonal = useViewerStore((s) => s.pathDiagonal);

  // Ratio of camera-path span to point-cloud span. >>1 means cameras are
  // spread much wider than the visible scene — a classic monocular scale
  // issue worth surfacing so the user isn't confused by apparent scale.
  const scaleRatio =
    sceneDiagonal > 0 && pathDiagonal > 0 ? pathDiagonal / sceneDiagonal : 0;

  const canUndo = meshHistoryIndex >= 0;
  const canRedo = meshHistoryIndex < meshHistory.length - 1;

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        padding: "6px 12px",
        borderBottom: "1px solid var(--rule)",
        flexWrap: "wrap",
        background: "var(--bg)",
      }}
    >
      <div style={{ display: "flex", gap: 2 }}>
        {MODES.map((m) => (
          <Tip key={m.value} text={m.tip} showIcon={false}>
            <button
              type="button"
              data-pressed={mode === m.value}
              onClick={() => setMode(m.value)}
            >
              {m.label}
            </button>
          </Tip>
        ))}
      </div>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 6,
          fontSize: "var(--fs-xs)",
        }}
      >
        <Tip text="Multiplier on the auto-computed point size (derived from the scene's bounding box). 0.25× = crisper, 4× = filling gaps. The absolute size rescales automatically when you open a scene with a different extent.">
          <span style={{ color: "var(--muted)" }}>pt ×</span>
        </Tip>
        <input
          type="range"
          min={0.25}
          max={4}
          step={0.05}
          value={pointSizeScale}
          onChange={(e) => setPointSizeScale(Number(e.target.value))}
          style={{ width: 120 }}
        />
        <span
          className="mono-small"
          style={{ minWidth: 38, textAlign: "right" }}
        >
          {pointSizeScale.toFixed(2)}×
        </span>
      </div>
      <div
        style={{
          marginLeft: "auto",
          display: "flex",
          gap: 8,
          alignItems: "center",
        }}
      >
        <Tip
          text={
            canUndo
              ? `Undo back to ${meshHistoryIndex === 0 ? "base reconstruction" : `rev ${meshHistory[meshHistoryIndex - 1]?.revision}`}.`
              : "Nothing to undo — you're on the base reconstruction."
          }
          showIcon={false}
        >
          <button
            type="button"
            onClick={undo}
            disabled={!canUndo}
            title="Undo last mesh edit"
          >
            ↶ undo
          </button>
        </Tip>
        <Tip
          text={
            canRedo
              ? `Redo forward to rev ${meshHistory[meshHistoryIndex + 1]?.revision}.`
              : "Nothing to redo — you're at the latest edit."
          }
          showIcon={false}
        >
          <button
            type="button"
            onClick={redo}
            disabled={!canRedo}
            title="Redo next mesh edit"
          >
            ↷ redo
          </button>
        </Tip>
        <Tip
          text="Orbit: click-drag rotates around a pivot; scroll zooms. Fly: WASD translates, mouse-drag looks around, Q/E roll, hold Shift to crawl."
          showIcon={false}
        >
          <button
            type="button"
            data-pressed={cameraMode === "fly"}
            onClick={() =>
              setCameraMode(cameraMode === "orbit" ? "fly" : "orbit")
            }
          >
            {cameraMode === "fly" ? "fly" : "orbit"}
          </button>
        </Tip>
        {scaleRatio >= 5 && (
          <Tip
            text={`Camera-path span (${pathDiagonal.toFixed(1)}) is ${scaleRatio.toFixed(1)}× the point-cloud span (${sceneDiagonal.toFixed(1)}). Monocular reconstruction can put cameras wider than visible geometry — the scene will look small at the viewer's default zoom. Use recenter to frame everything.`}
            showIcon={false}
          >
            <span
              className="mono-small"
              style={{
                color: "var(--danger)",
                padding: "1px 6px",
                border: "1px solid var(--danger)",
              }}
            >
              scale ratio {scaleRatio.toFixed(1)}×
            </span>
          </Tip>
        )}
        {cameraMode === "fly" && (
          <Tip
            text="Fly-mode speed multiplier. 1× traverses the scene in ~5 s at full key hold. Shift while holding any movement key temporarily slows to ~12% for precise positioning."
            showIcon={false}
          >
            <span
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 4,
                fontSize: "var(--fs-xs)",
                color: "var(--muted)",
              }}
            >
              speed
              <input
                type="range"
                min={0.1}
                max={5}
                step={0.1}
                value={flySpeedMult}
                onChange={(e) => setFlySpeedMult(Number(e.target.value))}
                style={{ width: 80 }}
              />
              <span
                className="mono-small"
                style={{ minWidth: 32, textAlign: "right" }}
              >
                {flySpeedMult.toFixed(1)}×
              </span>
            </span>
          </Tip>
        )}
        <Tip
          text="Reframe the camera on the current geometry. Use when the reconstruction drifts out of view or after lassoing / editing."
          showIcon={false}
        >
          <button type="button" onClick={requestRefit}>
            recenter
          </button>
        </Tip>
        {pathPoseCount > 1 && (
          <>
            <span
              style={{
                width: 1,
                height: 16,
                background: "var(--rule)",
                margin: "0 4px",
              }}
            />
            <Tip
              text="Show/hide the recorded camera trajectory drawn as a line through the scene."
              showIcon={false}
            >
              <button
                type="button"
                data-pressed={showCameraPath}
                onClick={() => setShowCameraPath(!showCameraPath)}
              >
                path
              </button>
            </Tip>
            <Tip
              text={
                playing
                  ? "Pause playback."
                  : "Fly the camera through the recorded trajectory at the capture fps. Scroll the slider to scrub."
              }
              showIcon={false}
            >
              <button
                type="button"
                data-pressed={playing}
                onClick={() => setPlaying(!playing)}
              >
                {playing ? "⏸ pause" : "▶ play"}
              </button>
            </Tip>
            <input
              type="range"
              min={0}
              max={pathPoseCount - 1}
              step={1}
              value={Math.round(playbackFrame)}
              onChange={(e) => {
                setPlaying(false);
                setPlaybackFrame(Number(e.target.value));
              }}
              style={{ width: 120 }}
              title="Scrub camera position along the path"
            />
            <span
              className="mono-small"
              style={{ minWidth: 64, textAlign: "right" }}
              title="Current pose / total poses"
            >
              {Math.round(playbackFrame) + 1}/{pathPoseCount}
            </span>
            <Tip
              text="Playback speed multiplier against the recorded capture fps."
              showIcon={false}
            >
              <select
                value={playbackSpeed}
                onChange={(e) => setPlaybackSpeed(Number(e.target.value))}
                style={{ width: 60 }}
              >
                <option value={0.25}>0.25×</option>
                <option value={0.5}>0.5×</option>
                <option value={1}>1×</option>
                <option value={2}>2×</option>
                <option value={4}>4×</option>
              </select>
            </Tip>
          </>
        )}
        <Tip text="Hold and drag to draw a polygon. Faces whose centroids land inside are selected for mesh-tool operations.">
          <button
            type="button"
            data-pressed={lassoActive}
            onClick={() => setLassoActive(!lassoActive)}
          >
            lasso{lassoActive ? " (on)" : ""}
          </button>
        </Tip>
        <span className="mono-small">{selection.size} selected</span>
        {selection.size > 0 && (
          <button type="button" onClick={clearSelection}>
            clear
          </button>
        )}
      </div>
    </div>
  );
}
