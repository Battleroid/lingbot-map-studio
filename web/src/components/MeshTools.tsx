"use client";

import { useState } from "react";

import { Tip } from "@/components/Tip";
import { meshEdit } from "@/lib/api";
import type { MeshOp } from "@/lib/types";
import { useViewerStore } from "@/lib/viewerStore";

interface Props {
  jobId: string;
  onRevision: (name: string) => void;
}

const TIPS: Record<MeshOp, string> = {
  cull: "Delete the currently lasso-selected faces. Use to remove obvious sky, water, or reflection artifacts.",
  fill_holes:
    "Close triangle holes whose boundary is ≤ N edges. Too-large values can cap open surfaces with undesirable webs.",
  decimate:
    "Reduce triangle count using quadric edge collapse. Ratio 0.5 = halve faces. Preserves boundaries and normals.",
  smooth:
    "Laplacian smoothing, N iterations. Use to soften noisy point-map reconstructions; too many iterations shrinks detail.",
  remove_small:
    "Delete disconnected components smaller than N% of the mesh. Good for ripping out floating fragments.",
};

/** Extract the numeric revision from `rev_003.glb` → 3. */
function parseRev(name: string): number {
  const m = /^rev_0*(\d+)\.glb$/.exec(name);
  return m ? Number(m[1]) : 0;
}

export function MeshTools({ jobId, onRevision }: Props) {
  const selection = useViewerStore((s) => s.selection);
  const clearSelection = useViewerStore((s) => s.clearSelection);
  const meshHistory = useViewerStore((s) => s.meshHistory);
  const meshHistoryIndex = useViewerStore((s) => s.meshHistoryIndex);
  const pushRevision = useViewerStore((s) => s.pushRevision);
  const [busy, setBusy] = useState<MeshOp | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [decimateRatio, setDecimateRatio] = useState(0.5);
  const [smoothIters, setSmoothIters] = useState(3);
  const [holeSize, setHoleSize] = useState(30);
  const [smallPct, setSmallPct] = useState(5);

  // If the user has undone past some revisions, branch from the currently
  // visible revision (server supports this via `source_revision`).
  const currentRev =
    meshHistoryIndex >= 0 ? meshHistory[meshHistoryIndex] : null;

  async function run(op: MeshOp, params: Record<string, unknown>, faces?: number[]) {
    setBusy(op);
    setError(null);
    try {
      const res = await meshEdit(jobId, {
        op,
        params,
        face_indices: faces,
        source_revision: currentRev?.revision,
      });
      pushRevision({ name: res.name, revision: parseRev(res.name), op });
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
      <div className="panel-header">
        <span>mesh tools</span>
      </div>
      <div className="panel-body" style={{ display: "grid", gap: 8 }}>
        <Tip text={TIPS.cull} showIcon={false}>
          <button
            type="button"
            disabled={busy !== null || selection.size === 0}
            onClick={() => run("cull", {}, Array.from(selection))}
            style={{ width: "100%" }}
          >
            {busy === "cull" ? "culling..." : `cull (${selection.size})`}
          </button>
        </Tip>

        <label className="stat">
          <Tip text={TIPS.fill_holes}>
            <span>fill holes (max size)</span>
          </Tip>
          <input
            type="number"
            value={holeSize}
            min={5}
            max={500}
            onChange={(e) => setHoleSize(Number(e.target.value))}
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
          <Tip text={TIPS.decimate}>
            <span>decimate ratio</span>
          </Tip>
          <input
            type="number"
            step={0.05}
            min={0.05}
            max={0.95}
            value={decimateRatio}
            onChange={(e) => setDecimateRatio(Number(e.target.value))}
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
          <Tip text={TIPS.smooth}>
            <span>smooth iters</span>
          </Tip>
          <input
            type="number"
            min={1}
            max={20}
            value={smoothIters}
            onChange={(e) => setSmoothIters(Number(e.target.value))}
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
          <Tip text={TIPS.remove_small}>
            <span>remove small (% faces)</span>
          </Tip>
          <input
            type="number"
            min={0.1}
            max={50}
            step={0.5}
            value={smallPct}
            onChange={(e) => setSmallPct(Number(e.target.value))}
          />
        </label>
        <button
          type="button"
          disabled={busy !== null}
          onClick={() => run("remove_small", { min_diag_perc: smallPct })}
        >
          {busy === "remove_small" ? "removing..." : "remove small"}
        </button>

        {error && (
          <div style={{ color: "var(--danger)", fontSize: "var(--fs-xs)" }}>{error}</div>
        )}
      </div>
    </div>
  );
}
