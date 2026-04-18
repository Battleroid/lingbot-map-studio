"use client";

import Link from "next/link";
import { use, useMemo, useState } from "react";

import { ConfigPanel } from "@/components/ConfigPanel";
import { ExportMenu } from "@/components/ExportMenu";
import { LogStream } from "@/components/LogStream";
import { MeshTools } from "@/components/MeshTools";
import { ViewerCanvas } from "@/components/Viewer/Canvas";
import { ViewerControls } from "@/components/Viewer/ViewerControls";
import { useJobManifest } from "@/hooks/useJob";
import { useJobStream } from "@/hooks/useJobStream";
import { artifactUrl } from "@/lib/api";

interface Props {
  params: Promise<{ id: string }>;
}

export default function JobPage({ params }: Props) {
  const { id } = use(params);
  const { data: manifest } = useJobManifest(id);
  const { events, status, latestProgress, latestStage } = useJobStream(id);
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
    <main
      style={{
        display: "grid",
        gridTemplateColumns: "300px 1fr 340px",
        gridTemplateRows: "auto 1fr",
        height: "100vh",
        overflow: "hidden",
      }}
    >
      <header
        style={{
          gridColumn: "1 / -1",
          borderBottom: "1px solid var(--rule)",
          padding: "8px 16px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <div style={{ display: "flex", gap: 12, alignItems: "baseline" }}>
          <Link href="/" style={{ textTransform: "uppercase", letterSpacing: "0.08em", fontSize: 12 }}>
            · studio
          </Link>
          <div style={{ fontSize: 12 }}>job {id}</div>
          <span className="chip" data-status={manifest?.status ?? "queued"}>
            {manifest?.status ?? "queued"}
          </span>
          <span className="chip" data-status={status === "open" ? "ready" : "failed"}>
            ws {status}
          </span>
        </div>
        <div style={{ flex: 1, maxWidth: 240, marginLeft: 16 }}>
          <div className="progress-bar">
            <span
              style={{
                width: `${((latestProgress ?? (manifest?.status === "ready" ? 1 : 0)) * 100).toFixed(0)}%`,
              }}
            />
          </div>
          <div style={{ fontSize: 10, color: "var(--muted)", marginTop: 2 }}>
            {latestStage ?? "—"}
          </div>
        </div>
      </header>

      <aside
        style={{
          borderRight: "1px solid var(--rule)",
          overflowY: "auto",
          padding: 10,
          display: "grid",
          gap: 10,
          alignContent: "start",
        }}
      >
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
          onReexport={(name) => setMeshOverride(name.endsWith(".glb") ? name : activeMeshName)}
        />
        <MeshTools jobId={id} onRevision={setMeshOverride} />
        {manifest?.error && (
          <div className="panel">
            <div className="panel-header">error</div>
            <div
              className="panel-body"
              style={{ fontSize: 11, whiteSpace: "pre-wrap", color: "var(--danger)" }}
            >
              {manifest.error}
            </div>
          </div>
        )}
      </aside>

      <section style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
        <ViewerControls />
        <div style={{ flex: 1, position: "relative" }}>
          <ViewerCanvas glbUrl={glbUrl} plyUrl={plyUrl} />
        </div>
      </section>

      <aside
        style={{
          borderLeft: "1px solid var(--rule)",
          overflow: "hidden",
          display: "flex",
          flexDirection: "column",
        }}
      >
        <LogStream events={events} />
      </aside>
    </main>
  );
}
