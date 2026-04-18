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

  const plyName = useMemo(() => {
    if (!manifest) return null;
    const ply = manifest.artifacts.find((a) => a.suffix === "ply");
    return ply?.name || null;
  }, [manifest]);

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
        </div>
      </section>

      <aside className="job-side right">
        <LogStream events={events} />
      </aside>
    </div>
  );
}
