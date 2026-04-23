"use client";

import { useState } from "react";

import { Tip } from "@/components/Tip";
import {
  createGsplatJobFromSource,
  meshEdit,
  artifactUrl,
} from "@/lib/api";
import type { JobManifest } from "@/lib/types";
import { useViewerStore } from "@/lib/viewerStore";
import { useRouter } from "next/navigation";

interface Props {
  jobId: string;
  manifest?: JobManifest | null;
  onRevision: (name: string) => void;
}

const TIPS = {
  poisson:
    "Run screened-Poisson surface reconstruction on the SLAM cloud to produce a baked .glb. Promotes the job into mesh-tools territory — cull/decimate/smooth/etc. become available. Slow on dense clouds; free on sparse DPVO output.",
  train_gs:
    "Kick off a gsplat training job seeded from this SLAM output. Frames + poses + reconstruction.ply are reused; no upload needed. Lands on the gs worker container.",
  pose_graph:
    "Download the pose_graph.json for this run — per-keyframe poses + intrinsics, suitable for feeding a separate gsplat training run or an external tool.",
};

/** Extract the numeric revision from `rev_003.glb` → 3. */
function parseRev(name: string): number {
  const m = /^rev_0*(\d+)\.glb$/.exec(name);
  return m ? Number(m[1]) : 0;
}

export function SlamToolsPanel({ jobId, manifest, onRevision }: Props) {
  const router = useRouter();
  const pushRevision = useViewerStore((s) => s.pushRevision);
  const [busy, setBusy] = useState<"poisson" | "train_gs" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [poissonDepth, setPoissonDepth] = useState(8);

  const hasPoseGraph = !!manifest?.artifacts.some(
    (a) => a.name === "pose_graph.json",
  );
  const canTrainGs = manifest?.status === "ready";
  const canMesh = manifest?.status === "ready";

  async function runPoisson() {
    setBusy("poisson");
    setError(null);
    try {
      const res = await meshEdit(jobId, {
        op: "surface_recon",
        params: { depth: poissonDepth },
      });
      pushRevision({
        name: res.name,
        revision: parseRev(res.name),
        op: "surface_recon",
      });
      onRevision(res.name);
    } catch (e) {
      setError(String((e as Error).message));
    } finally {
      setBusy(null);
    }
  }

  async function trainGsplat() {
    setBusy("train_gs");
    setError(null);
    try {
      const { id: newId } = await createGsplatJobFromSource(jobId);
      router.push(`/jobs/${newId}`);
    } catch (e) {
      setError(String((e as Error).message));
      setBusy(null);
    }
  }

  return (
    <div className="panel">
      <div className="panel-header">
        <span>slam tools</span>
      </div>
      <div className="panel-body" style={{ display: "grid", gap: 8 }}>
        <div className="section-title">surface reconstruction</div>
        <label className="stat">
          <Tip text={TIPS.poisson}>
            <span>poisson depth</span>
          </Tip>
          <input
            type="number"
            min={5}
            max={11}
            step={1}
            value={poissonDepth}
            onChange={(e) => setPoissonDepth(Number(e.target.value))}
          />
        </label>
        <Tip text={TIPS.poisson} showIcon={false}>
          <button
            type="button"
            disabled={busy !== null || !canMesh}
            onClick={runPoisson}
            style={{ width: "100%" }}
          >
            {busy === "poisson"
              ? "meshing..."
              : "mesh from point cloud (poisson)"}
          </button>
        </Tip>

        <div
          style={{
            marginTop: 6,
            paddingTop: 6,
            borderTop: "1px solid var(--rule)",
          }}
        >
          <div className="section-title">downstream</div>
        </div>
        <Tip text={TIPS.train_gs} showIcon={false}>
          <button
            type="button"
            disabled={busy !== null || !canTrainGs}
            onClick={trainGsplat}
            style={{ width: "100%" }}
          >
            {busy === "train_gs"
              ? "starting..."
              : "train gaussian splat from this"}
          </button>
        </Tip>

        {hasPoseGraph && (
          <Tip text={TIPS.pose_graph} showIcon={false}>
            <a
              href={artifactUrl(jobId, "pose_graph.json")}
              download
              className="button-like"
              style={{
                display: "block",
                textAlign: "center",
                padding: "4px 8px",
                border: "1px solid var(--rule)",
                textDecoration: "none",
                color: "var(--fg)",
              }}
            >
              download pose_graph.json
            </a>
          </Tip>
        )}

        {error && (
          <div
            style={{ color: "var(--danger)", fontSize: "var(--fs-xs)" }}
          >
            {error}
          </div>
        )}
      </div>
    </div>
  );
}
