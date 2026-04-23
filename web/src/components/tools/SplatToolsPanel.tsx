"use client";

import { useState } from "react";

import { Tip } from "@/components/Tip";
import { artifactUrl } from "@/lib/api";
import type { JobManifest } from "@/lib/types";
import { useViewerStore } from "@/lib/viewerStore";

interface Props {
  jobId: string;
  manifest?: JobManifest | null;
  onRevision: (name: string) => void;
}

const TIPS = {
  opacity:
    "Hide gaussians below this sigmoid-opacity at render time. Doesn't touch the underlying splat.ply — pure viewer filter. The server-side prune (coming next) produces a new revision with the gaussians actually removed.",
  layers:
    "Toggle individual render layers. Useful for comparing the raw splat against a baked mesh or for hiding the camera-path line while navigating.",
  download_splat:
    "Download the trained splat.ply in standard 3DGS format. Loads directly into Spark, nerfstudio viewer, mkkellogg/GaussianSplats3D, and Luma's viewer.",
  download_sogs:
    "Download the SOGS sidecar (compressed splat placeholder until the real encoder ships).",
};

export function SplatToolsPanel({ jobId, manifest }: Props) {
  const [busy, setBusy] = useState(false);
  const [_error, setError] = useState<string | null>(null);
  void busy;
  void setBusy;

  const threshold = useViewerStore((s) => s.splatOpacityThreshold);
  const setThreshold = useViewerStore((s) => s.setSplatOpacityThreshold);
  const renderLayers = useViewerStore((s) => s.renderLayers);
  const toggleLayer = useViewerStore((s) => s.toggleRenderLayer);

  const splatPlyName = manifest?.artifacts.find(
    (a) =>
      a.suffix === "ply" &&
      (a.name === "splat.ply" || a.name.startsWith("splat_")),
  )?.name;
  const sogsName = manifest?.artifacts.find((a) => a.name === "splat.sogs")
    ?.name;

  // Silence the unused-setter warning at compile time while reserving the
  // state slot for the real server-side prune / crop actions below.
  void setError;

  return (
    <div className="panel">
      <div className="panel-header">
        <span>splat tools</span>
      </div>
      <div className="panel-body" style={{ display: "grid", gap: 8 }}>
        <div className="section-title">layers</div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
          {(["splat", "points", "mesh", "camera_path"] as const).map((l) => {
            const on = renderLayers.has(l);
            return (
              <Tip key={l} text={TIPS.layers} showIcon={false}>
                <button
                  type="button"
                  onClick={() => toggleLayer(l)}
                  aria-pressed={on}
                  style={{
                    flex: 1,
                    minWidth: 0,
                    background: on ? "var(--fg)" : "var(--panel-bg, transparent)",
                    color: on ? "var(--bg)" : "var(--fg)",
                  }}
                >
                  {l.replace("_", " ")}
                </button>
              </Tip>
            );
          })}
        </div>

        <div
          style={{
            marginTop: 6,
            paddingTop: 6,
            borderTop: "1px solid var(--rule)",
          }}
        >
          <div className="section-title">viewer prune</div>
        </div>
        <label className="stat">
          <Tip text={TIPS.opacity}>
            <span>opacity ≥</span>
          </Tip>
          <input
            type="number"
            min={0}
            max={1}
            step={0.01}
            value={threshold}
            onChange={(e) => setThreshold(Number(e.target.value))}
          />
        </label>
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={threshold}
          onChange={(e) => setThreshold(Number(e.target.value))}
        />

        <div
          style={{
            marginTop: 6,
            paddingTop: 6,
            borderTop: "1px solid var(--rule)",
          }}
        >
          <div className="section-title">downloads</div>
        </div>
        {splatPlyName && (
          <Tip text={TIPS.download_splat} showIcon={false}>
            <a
              href={artifactUrl(jobId, splatPlyName)}
              download
              style={{
                display: "block",
                textAlign: "center",
                padding: "4px 8px",
                border: "1px solid var(--rule)",
                textDecoration: "none",
                color: "var(--fg)",
              }}
            >
              splat.ply
            </a>
          </Tip>
        )}
        {sogsName && (
          <Tip text={TIPS.download_sogs} showIcon={false}>
            <a
              href={artifactUrl(jobId, sogsName)}
              download
              style={{
                display: "block",
                textAlign: "center",
                padding: "4px 8px",
                border: "1px solid var(--rule)",
                textDecoration: "none",
                color: "var(--fg)",
              }}
            >
              splat.sogs
            </a>
          </Tip>
        )}
        {!splatPlyName && (
          <div className="mono-small" style={{ opacity: 0.7 }}>
            splat artifacts appear after training completes.
          </div>
        )}

        <div className="mono-small" style={{ opacity: 0.7 }}>
          server-side crop / prune / bake-to-mesh land in a follow-up — those
          need the worker-gs backend wired. the viewer-side prune above is
          always available.
        </div>
      </div>
    </div>
  );
}
