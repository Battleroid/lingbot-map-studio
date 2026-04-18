"use client";

import Link from "next/link";
import { use, useMemo, useState } from "react";

import { ConfigPanel } from "@/components/ConfigPanel";
import { ExportMenu } from "@/components/ExportMenu";
import { JobStatusStrip } from "@/components/JobStatusStrip";
import { LogStream } from "@/components/LogStream";
import { MeshTools } from "@/components/MeshTools";
import { ViewerCanvas } from "@/components/Viewer/Canvas";
import { ViewerControls } from "@/components/Viewer/ViewerControls";
import { useJobManifest } from "@/hooks/useJob";
import { useJobStatus } from "@/hooks/useJobStatus";
import { useJobStream } from "@/hooks/useJobStream";
import { artifactUrl } from "@/lib/api";

interface Props {
  params: Promise<{ id: string }>;
}

export default function JobPage({ params }: Props) {
  const { id } = use(params);
  const { data: manifest } = useJobManifest(id);
  const { events, status } = useJobStream(id);
  const derived = useJobStatus(events, manifest?.status);
  const [meshOverride, setMeshOverride] = useState<string | null>(null);

  const activeMeshName = meshOverride || manifest?.latest_mesh || null;

  // Find the latest partial PLY emitted during inference (live preview) so we
  // can swap the viewer's point cloud URL as the reconstruction grows.
  const latestPartialPly = useMemo(() => {
    let latest: string | null = null;
    for (const ev of events) {
      if (ev.stage === "artifact" && ev.data?.["kind"] === "partial_ply") {
        const name = ev.data?.["name"];
        if (typeof name === "string") latest = name;
      }
    }
    return latest;
  }, [events]);

  const finalPlyName = useMemo(() => {
    if (!manifest) return null;
    const ply = manifest.artifacts.find(
      (a) => a.suffix === "ply" && a.name.startsWith("reconstruction"),
    );
    return ply?.name || null;
  }, [manifest]);

  // Prefer the final PLY once the job is ready; otherwise show the latest
  // partial snapshot during inference.
  const plyName = finalPlyName || latestPartialPly || null;

  const glbUrl = activeMeshName ? artifactUrl(id, activeMeshName) : null;
  const plyUrl = plyName ? artifactUrl(id, plyName) : null;

  return (
    <div className="job-shell">
      <header className="job-header">
        <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
          <Link href="/" className="page-title" style={{ textDecoration: "none" }}>
            · studio
          </Link>
          <span className="mono-small">job {id}</span>
          <span className="chip" data-status={manifest?.status ?? "queued"}>
            {manifest?.status ?? "queued"}
          </span>
        </div>
      </header>

      <aside className="job-side">
        {manifest && (
          <ConfigPanel
            config={manifest.config}
            onChange={() => undefined}
            readOnly
            compact
          />
        )}
        <ExportMenu
          jobId={id}
          artifacts={manifest?.artifacts ?? []}
          latestMesh={activeMeshName}
          onReexport={(name) =>
            setMeshOverride(name.endsWith(".glb") ? name : activeMeshName)
          }
        />
        <MeshTools jobId={id} onRevision={setMeshOverride} />
        {manifest?.error && (
          <div className="panel">
            <div className="panel-header">
              <span>error</span>
            </div>
            <div
              className="panel-body"
              style={{
                fontSize: "var(--fs-xs)",
                whiteSpace: "pre-wrap",
                color: "var(--danger)",
              }}
            >
              {manifest.error}
            </div>
          </div>
        )}
      </aside>

      <section className="job-center">
        <JobStatusStrip
          derived={derived}
          jobStatus={manifest?.status}
          wsStatus={status}
        />
        <ViewerControls />
        <div style={{ flex: 1, position: "relative", minHeight: 0 }}>
          <ViewerCanvas glbUrl={glbUrl} plyUrl={plyUrl} />
          {plyName && plyName.startsWith("partial_") && (
            <div
              style={{
                position: "absolute",
                top: 10,
                left: 10,
                padding: "3px 8px",
                background: "var(--fg)",
                color: "var(--bg)",
                fontSize: "var(--fs-xs)",
                textTransform: "uppercase",
                letterSpacing: "0.08em",
                border: "1px solid var(--fg)",
                pointerEvents: "none",
              }}
            >
              live · {plyName.replace(/^partial_0*/, "").replace(".ply", "")} frames
            </div>
          )}
        </div>
      </section>

      <aside className="job-side right">
        <LogStream events={events} />
      </aside>
    </div>
  );
}
