"use client";

import { useState } from "react";

import { meshEdit } from "@/lib/api";
import type { MeshOp } from "@/lib/types";
import { useViewerStore } from "@/lib/viewerStore";

interface Props {
  jobId: string;
  onRevision: (name: string) => void;
}

export function MeshTools({ jobId, onRevision }: Props) {
  const selection = useViewerStore((s) => s.selection);
  const clearSelection = useViewerStore((s) => s.clearSelection);
  const [busy, setBusy] = useState<MeshOp | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [decimateRatio, setDecimateRatio] = useState(0.5);
  const [smoothIters, setSmoothIters] = useState(3);
  const [holeSize, setHoleSize] = useState(30);
  const [smallPct, setSmallPct] = useState(5);

  async function run(op: MeshOp, params: Record<string, unknown>, faces?: number[]) {
    setBusy(op);
    setError(null);
    try {
      const res = await meshEdit(jobId, {
        op,
        params,
        face_indices: faces,
      });
      onRevision(res.name);
      clearSelection();
    } catch (e) {
      setError(String((e as Error).message));
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="panel">
      <div className="panel-header">mesh tools</div>
      <div className="panel-body" style={{ display: "grid", gap: 8 }}>
        <button
          type="button"
          disabled={busy !== null || selection.size === 0}
          onClick={() => run("cull", {}, Array.from(selection))}
        >
          {busy === "cull" ? "culling..." : `cull (${selection.size})`}
        </button>

        <label className="stat">
          <span>fill holes (max size)</span>
          <input
            type="number"
            value={holeSize}
            min={5}
            max={500}
            onChange={(e) => setHoleSize(Number(e.target.value))}
            style={{ width: 70, textAlign: "right" }}
          />
        </label>
        <button
          type="button"
          disabled={busy !== null}
          onClick={() => run("fill_holes", { max_hole_size: holeSize })}
        >
          {busy === "fill_holes" ? "filling..." : "fill holes"}
        </button>

        <label className="stat">
          <span>decimate ratio</span>
          <input
            type="number"
            step={0.05}
            min={0.05}
            max={0.95}
            value={decimateRatio}
            onChange={(e) => setDecimateRatio(Number(e.target.value))}
            style={{ width: 70, textAlign: "right" }}
          />
        </label>
        <button
          type="button"
          disabled={busy !== null}
          onClick={() => run("decimate", { ratio: decimateRatio })}
        >
          {busy === "decimate" ? "decimating..." : "decimate"}
        </button>

        <label className="stat">
          <span>smooth iters</span>
          <input
            type="number"
            min={1}
            max={20}
            value={smoothIters}
            onChange={(e) => setSmoothIters(Number(e.target.value))}
            style={{ width: 70, textAlign: "right" }}
          />
        </label>
        <button
          type="button"
          disabled={busy !== null}
          onClick={() => run("smooth", { iters: smoothIters })}
        >
          {busy === "smooth" ? "smoothing..." : "smooth"}
        </button>

        <label className="stat">
          <span>remove small (% faces)</span>
          <input
            type="number"
            min={0.1}
            max={50}
            step={0.5}
            value={smallPct}
            onChange={(e) => setSmallPct(Number(e.target.value))}
            style={{ width: 70, textAlign: "right" }}
          />
        </label>
        <button
          type="button"
          disabled={busy !== null}
          onClick={() => run("remove_small", { min_diag_perc: smallPct })}
        >
          {busy === "remove_small" ? "removing..." : "remove small"}
        </button>

        {error && <div style={{ color: "var(--danger)", fontSize: 11 }}>{error}</div>}
      </div>
    </div>
  );
}
