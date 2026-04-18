"use client";

import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";

import { ConfigPanel } from "@/components/ConfigPanel";
import { JobList } from "@/components/JobList";
import { PreprocPreview } from "@/components/PreprocPreview";
import { ProbePanel } from "@/components/ProbePanel";
import { UploadDropzone } from "@/components/UploadDropzone";
import {
  createDraft,
  deleteDraft,
  startJobFromDraft,
  type DraftRecord,
} from "@/lib/api";
import { DEFAULT_CONFIG, type JobConfig } from "@/lib/types";

export default function Home() {
  const router = useRouter();
  const [files, setFiles] = useState<File[]>([]);
  const [uploadPct, setUploadPct] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [draft, setDraft] = useState<DraftRecord | null>(null);
  const [config, setConfig] = useState<JobConfig>(DEFAULT_CONFIG);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canProbe = files.length > 0 && !uploading && !draft;

  const effectiveConfig = useMemo(() => {
    if (!draft) return config;
    return { ...DEFAULT_CONFIG, ...draft.suggested_config, ...config };
  }, [draft, config]);

  async function runProbe() {
    setError(null);
    setUploading(true);
    setUploadPct(0);
    try {
      const d = await createDraft(files, setUploadPct);
      setDraft(d);
      setConfig({ ...DEFAULT_CONFIG, ...d.suggested_config });
    } catch (e) {
      setError(String((e as Error).message));
    } finally {
      setUploading(false);
    }
  }

  async function reset() {
    if (draft) {
      try {
        await deleteDraft(draft.id);
      } catch {
        /* ignore */
      }
    }
    setDraft(null);
    setFiles([]);
    setConfig(DEFAULT_CONFIG);
    setUploadPct(0);
    setError(null);
  }

  async function start() {
    if (!draft) return;
    setSubmitting(true);
    setError(null);
    try {
      const { id } = await startJobFromDraft(draft.id, effectiveConfig);
      router.push(`/jobs/${id}`);
    } catch (e) {
      setError(String((e as Error).message));
      setSubmitting(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <h1 className="page-title">lingbot-map studio</h1>
        <span className="mono-small">feed-forward 3d reconstruction · local gpu</span>
      </header>
      <main className="app-main">
        <section className="home-grid">
          <div className="panel">
            <div className="panel-header">
              <span>{draft ? "2 · review + start" : "1 · upload + probe"}</span>
              {draft && <span className="meta">draft {draft.id}</span>}
            </div>
            <div className="panel-body" style={{ display: "grid", gap: 12 }}>
              {!draft && (
                <>
                  <UploadDropzone
                    files={files}
                    onChange={setFiles}
                    disabled={uploading}
                  />
                  {uploading && (
                    <div>
                      <div className="progress-bar">
                        <span style={{ width: `${(uploadPct * 100).toFixed(0)}%` }} />
                      </div>
                      <div className="mono-small" style={{ marginTop: 4 }}>
                        uploading · {(uploadPct * 100).toFixed(0)}%
                      </div>
                    </div>
                  )}
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      gap: 8,
                    }}
                  >
                    <span className="mono-small">
                      {files.length} file{files.length === 1 ? "" : "s"} · concatenated in upload order
                    </span>
                    <button type="button" onClick={runProbe} disabled={!canProbe}>
                      {uploading ? "probing..." : "probe videos"}
                    </button>
                  </div>
                </>
              )}

              {draft && (
                <>
                  <ProbePanel draft={draft} />
                  <PreprocPreview draftId={draft.id} config={effectiveConfig} />
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      gap: 8,
                    }}
                  >
                    <span className="mono-small">
                      config auto-populated from probe — tweak on the right, then start
                    </span>
                    <div style={{ display: "flex", gap: 6 }}>
                      <button type="button" onClick={reset} disabled={submitting}>
                        discard
                      </button>
                      <button type="button" onClick={start} disabled={submitting}>
                        {submitting ? "starting..." : "start reconstruction"}
                      </button>
                    </div>
                  </div>
                </>
              )}

              {error && (
                <div style={{ color: "var(--danger)", fontSize: "var(--fs-sm)" }}>
                  {error}
                </div>
              )}
            </div>
          </div>

          <ConfigPanel
            config={effectiveConfig}
            onChange={(patch) => setConfig({ ...effectiveConfig, ...patch })}
            readOnly={!draft}
            title="config"
          />
        </section>

        <JobList />
      </main>
    </div>
  );
}
