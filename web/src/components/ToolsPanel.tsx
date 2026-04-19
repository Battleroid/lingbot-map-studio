"use client";

import { MeshToolsPanel } from "@/components/tools/MeshToolsPanel";
import { SlamToolsPanel } from "@/components/tools/SlamToolsPanel";
import { SplatToolsPanel } from "@/components/tools/SplatToolsPanel";
import { useJobManifest } from "@/hooks/useJob";
import {
  type AnyJobConfig,
  isGsplatConfig,
  isLingbotConfig,
  isSlamConfig,
} from "@/lib/types";

interface Props {
  jobId: string;
  config: AnyJobConfig | null;
  onRevision: (name: string) => void;
}

/**
 * Dispatch to the right per-mode tool panel based on the job's processor.
 *
 * - Lingbot: mesh tools (cull / fill_holes / decimate / smooth / etc).
 * - SLAM: SLAM tools (mesh-from-cloud promotes to mesh tools; train-gs
 *         chain; pose-graph download). If the job already has a baked mesh,
 *         both panels are shown so mesh edits are available too.
 * - Gsplat: splat tools (layer toggles, viewer-side opacity prune,
 *           downloads).
 */
export function ToolsPanel({ jobId, config, onRevision }: Props) {
  const { data: manifest } = useJobManifest(jobId);

  if (!config) {
    return null;
  }

  if (isLingbotConfig(config)) {
    return <MeshToolsPanel jobId={jobId} onRevision={onRevision} />;
  }

  if (isSlamConfig(config)) {
    const hasMesh =
      !!manifest?.latest_mesh ||
      !!manifest?.artifacts.some((a) => a.suffix === "glb");
    return (
      <>
        <SlamToolsPanel
          jobId={jobId}
          manifest={manifest}
          onRevision={onRevision}
        />
        {hasMesh && <MeshToolsPanel jobId={jobId} onRevision={onRevision} />}
      </>
    );
  }

  if (isGsplatConfig(config)) {
    return <SplatToolsPanel jobId={jobId} manifest={manifest} onRevision={onRevision} />;
  }

  return null;
}
