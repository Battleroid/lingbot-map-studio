"use client";

import Link from "next/link";
import { use, useEffect, useMemo, useState } from "react";

import { ConfigPanel } from "@/components/ConfigPanel";
import { ExportMenu } from "@/components/ExportMenu";
import { JobStatusStrip } from "@/components/JobStatusStrip";
import { LogStream } from "@/components/LogStream";
import { MeshTools } from "@/components/MeshTools";
import { ViewerCanvas } from "@/components/Viewer/Canvas";
import { ViewerControls } from "@/components/Viewer/ViewerControls";
import { ViewerErrorBoundary } from "@/components/Viewer/ViewerErrorBoundary";
import { useCameraPath } from "@/hooks/useCameraPath";
import { useJobManifest } from "@/hooks/useJob";
import { useJobStatus } from "@/hooks/useJobStatus";
import { useJobStream } from "@/hooks/useJobStream";
import { artifactUrl, restartJob, stopJob } from "@/lib/api";
import { isLingbotConfig } from "@/lib/types";
import { useViewerStore } from "@/lib/viewerStore";

interface Props {
  params: Promise<{ id: string }>;
}

import { useRouter } from "next/navigation";

export default function JobPage({ params }: Props) {
  const { id } = use(params);
  const router = useRouter();
  const { data: manifest } = useJobManifest(id);
  const { events, status } = useJobStream(id);
  const derived = useJobStatus(events, manifest?.status);
  const [meshOverride, setMeshOverride] = useState<string | null>(null);
  const [busy, setBusy] = useState<"stop" | "restart" | null>(null);

  // Mesh edit history — reset when navigating between jobs so the undo stack
  // doesn't bleed across.
  const meshHistory = useViewerStore((s) => s.meshHistory);
  const meshHistoryIndex = useViewerStore((s) => s.meshHistoryIndex);
  const resetMeshHistory = useViewerStore((s) => s.resetHistory);
  useEffect(() => {
    resetMeshHistory();
  }, [id, resetMeshHistory]);
  const [stopRequestedAt, setStopRequestedAt] = useState<number | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  const jobStatus = manifest?.status;
  const isRunning =
    jobStatus === "queued" ||
    jobStatus === "ingest" ||
    jobStatus === "inference" ||
    jobStatus === "export";
  const isTerminal =
    jobStatus === "ready" || jobStatus === "failed" || jobStatus === "cancelled";

  async function onStop() {
    setBusy("stop");
    setActionError(null);
    // If the first graceful stop was >3s ago and the job is still running,
    // a second click escalates to force-stop.
    const shouldForce =
      stopRequestedAt !== null && Date.now() - stopRequestedAt > 3000;
    try {
      await stopJob(id, shouldForce);
      if (!shouldForce) setStopRequestedAt(Date.now());
    } catch (e) {
      setActionError(String((e as Error).message));
    } finally {
      setBusy(null);
    }
  }

  async function onRestart() {
    setBusy("restart");
    setActionError(null);
    try {
      const { id: newId } = await restartJob(id);
      router.push(`/jobs/${newId}`);
    } catch (e) {
      setActionError(String((e as Error).message));
      setBusy(null);
    }
  }

  // Active mesh priority:
  //   1. explicit override (e.g. Export Menu's re-export picks a specific GLB)
  //   2. undo/redo pointer position (history-driven edits)
  //   3. most recent GLB on disk (manifest.latest_mesh — the server's view)
  const historyMesh =
    meshHistoryIndex >= 0 ? meshHistory[meshHistoryIndex]?.name : null;
  const activeMeshName =
    meshOverride || historyMesh || manifest?.latest_mesh || null;

  // Find the latest partial PLY emitted during inference (live preview) so we
  // can swap the viewer's point cloud URL as the reconstruction grows. A
  // partial_cleanup event from the worker clears this so we don't try to
  // fetch a 404'd URL after export deletes the snapshots.
  const latestPartialPly = useMemo(() => {
    let latest: string | null = null;
    for (const ev of events) {
      if (ev.stage !== "artifact") continue;
      const kind = ev.data?.["kind"];
      if (kind === "partial_ply") {
        const name = ev.data?.["name"];
        if (typeof name === "string") latest = name;
      } else if (kind === "partial_cleanup") {
        latest = null;
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

  // Camera path is only produced on successful export — only fetch when the
  // manifest actually lists it (avoids spurious 404s during inference).
  const cameraPathAvailable = Boolean(
    manifest?.artifacts?.some((a) => a.name === "camera_path.json"),
  );
  const { data: cameraPath } = useCameraPath(id, cameraPathAvailable);

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
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          {actionError && (
            <span className="mono-small" style={{ color: "var(--danger)" }}>
              {actionError}
            </span>
          )}
          {isRunning && (
            <button
              type="button"
              disabled={busy !== null}
              onClick={onStop}
              title={
                stopRequestedAt === null
                  ? "Request graceful cancellation at the next checkpoint"
                  : "Force-abandon — marks cancelled even if the GPU call is hung"
              }
            >
              {busy === "stop"
                ? "stopping…"
                : stopRequestedAt === null
                  ? "stop"
                  : "force stop"}
            </button>
          )}
          {isTerminal && manifest && (
            <button
              type="button"
              disabled={busy !== null}
              onClick={onRestart}
            >
              {busy === "restart" ? "restarting…" : "restart"}
            </button>
          )}
        </div>
      </header>

      <aside className="job-side">
        {manifest && isLingbotConfig(manifest.config) && (
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
        <ViewerControls pathPoseCount={cameraPath?.poses.length ?? 0} />
        <div style={{ flex: 1, position: "relative", minHeight: 0 }}>
          <ViewerErrorBoundary resetKey={`${glbUrl}|${plyUrl}`}>
            <ViewerCanvas
              glbUrl={glbUrl}
              plyUrl={plyUrl}
              cameraPath={cameraPath}
            />
          </ViewerErrorBoundary>
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
