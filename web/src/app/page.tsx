"use client";

import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";

import { ConfigPanel } from "@/components/ConfigPanel";
import { ExecutionPanel } from "@/components/ExecutionPanel";
import { GsplatConfigPanel } from "@/components/GsplatConfigPanel";
import { JobList } from "@/components/JobList";
import { ModePicker, type StudioMode } from "@/components/ModePicker";
import { PreprocPreview } from "@/components/PreprocPreview";
import { ProbePanel } from "@/components/ProbePanel";
import { SlamConfigPanel } from "@/components/SlamConfigPanel";
import { UploadDropzone } from "@/components/UploadDropzone";
import {
  createDraft,
  createGsplatJobFromSource,
  deleteDraft,
  startJobFromDraft,
  type DraftRecord,
} from "@/lib/api";
import { useJobList } from "@/hooks/useJob";
import {
  DEFAULT_CONFIG,
  DEFAULT_EXECUTION_FIELDS,
  DEFAULT_GSPLAT_CONFIG,
  DEFAULT_SLAM_CONFIGS,
  type AnyJobConfig,
  type ExecutionFields,
  type GsplatConfig,
  type LingbotConfig,
  type SlamBackend,
  type SlamConfig,
} from "@/lib/types";

export default function Home() {
  const router = useRouter();
  const { data: allJobs = [] } = useJobList();

  const [mode, setMode] = useState<StudioMode>("lingbot");
  const [slamBackend, setSlamBackend] = useState<SlamBackend>("mast3r_slam");
  const [gsplatSourceId, setGsplatSourceId] = useState<string | null>(null);

  const [files, setFiles] = useState<File[]>([]);
  const [uploadPct, setUploadPct] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [draft, setDraft] = useState<DraftRecord | null>(null);
  const [lingbotConfig, setLingbotConfig] =
    useState<LingbotConfig>(DEFAULT_CONFIG);
  const [slamConfigs, setSlamConfigs] = useState<{
    [K in SlamBackend]: SlamConfig;
  }>(DEFAULT_SLAM_CONFIGS);
  const [gsplatConfig, setGsplatConfig] = useState<
    Omit<GsplatConfig, "source_job_id">
  >(DEFAULT_GSPLAT_CONFIGS_DEFAULT());
  // Execution target (local vs cloud provider) + instance spec. Shared
  // across modes — a user toggling from lingbot → slam keeps their GPU
  // class / spot / cost-cap choices intact.
  const [executionFields, setExecutionFields] = useState<ExecutionFields>(
    DEFAULT_EXECUTION_FIELDS,
  );
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const requiresUpload = mode === "lingbot" || mode === "slam";
  const canProbe =
    requiresUpload && files.length > 0 && !uploading && !draft;

  const currentSlamConfig = slamConfigs[slamBackend];

  const effectiveLingbotConfig = useMemo<LingbotConfig>(() => {
    if (!draft) return lingbotConfig;
    return { ...DEFAULT_CONFIG, ...draft.suggested_config, ...lingbotConfig };
  }, [draft, lingbotConfig]);

  const effectiveConfig: AnyJobConfig =
    mode === "lingbot"
      ? effectiveLingbotConfig
      : mode === "slam"
        ? currentSlamConfig
        : {
            ...gsplatConfig,
            processor: "gsplat",
            source_job_id: gsplatSourceId ?? "",
          };

  async function runProbe() {
    setError(null);
    setUploading(true);
    setUploadPct(0);
    try {
      const d = await createDraft(files, setUploadPct);
      setDraft(d);
      if (mode === "lingbot") {
        setLingbotConfig({ ...DEFAULT_CONFIG, ...d.suggested_config });
      }
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
    setLingbotConfig(DEFAULT_CONFIG);
    setUploadPct(0);
    setError(null);
  }

  function changeMode(next: StudioMode) {
    if (next === mode) return;
    // Reset any in-flight upload state; modes have different prereqs.
    if (draft) {
      deleteDraft(draft.id).catch(() => undefined);
    }
    setMode(next);
    setDraft(null);
    setFiles([]);
    setLingbotConfig(DEFAULT_CONFIG);
    setUploadPct(0);
    setError(null);
  }

  async function start() {
    setSubmitting(true);
    setError(null);
    try {
      if (mode === "lingbot" || mode === "slam") {
        if (!draft) {
          throw new Error("probe a clip first");
        }
        const baseCfg: AnyJobConfig =
          mode === "lingbot" ? effectiveLingbotConfig : currentSlamConfig;
        const cfg = { ...baseCfg, ...executionFields } as AnyJobConfig;
        const { id } = await startJobFromDraft(draft.id, cfg);
        router.push(`/jobs/${id}`);
        return;
      }
      // gsplat: no upload, chain off a source job.
      if (!gsplatSourceId) {
        throw new Error("pick a source job");
      }
      const { id } = await createGsplatJobFromSource(gsplatSourceId, {
        ...gsplatConfig,
        ...executionFields,
      });
      router.push(`/jobs/${id}`);
    } catch (e) {
      setError(String((e as Error).message));
      setSubmitting(false);
    }
  }

  const startDisabled =
    submitting ||
    (requiresUpload && !draft) ||
    (mode === "gsplat" && !gsplatSourceId);

  const startLabel =
    mode === "lingbot"
      ? "start reconstruction"
      : mode === "slam"
        ? `start ${slamBackend.replace("_", " ")}`
        : "train gaussian splat";

  return (
    <div className="app-shell">
      <header className="app-header">
        <h1 className="page-title">lingbot-map studio</h1>
        <span className="mono-small">
          feed-forward 3d reconstruction · slam · gaussian splat · local gpu
        </span>
      </header>
      <main className="app-main">
        <section className="home-grid">
          <div style={{ display: "grid", gap: 12 }}>
            <ModePicker
              mode={mode}
              onMode={changeMode}
              slamBackend={slamBackend}
              onSlamBackend={setSlamBackend}
              gsplatSourceId={gsplatSourceId}
              onGsplatSource={setGsplatSourceId}
              sourceJobs={allJobs}
              disabled={submitting || uploading}
            />

            <div className="panel">
              <div className="panel-header">
                <span>
                  {mode === "gsplat"
                    ? "1 · pick source · 2 · review + start"
                    : draft
                      ? "2 · review + start"
                      : "1 · upload + probe"}
                </span>
                {draft && <span className="meta">draft {draft.id}</span>}
              </div>
              <div
                className="panel-body"
                style={{ display: "grid", gap: 12 }}
              >
                {requiresUpload && !draft && (
                  <>
                    <UploadDropzone
                      files={files}
                      onChange={setFiles}
                      disabled={uploading}
                    />
                    {uploading && (
                      <div>
                        <div className="progress-bar">
                          <span
                            style={{
                              width: `${(uploadPct * 100).toFixed(0)}%`,
                            }}
                          />
                        </div>
                        <div
                          className="mono-small"
                          style={{ marginTop: 4 }}
                        >
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
                        {files.length} file{files.length === 1 ? "" : "s"} ·
                        concatenated in upload order
                      </span>
                      <button
                        type="button"
                        onClick={runProbe}
                        disabled={!canProbe}
                      >
                        {uploading ? "probing..." : "probe videos"}
                      </button>
                    </div>
                  </>
                )}

                {requiresUpload && draft && (
                  <>
                    <ProbePanel draft={draft} />
                    {mode === "lingbot" && (
                      <PreprocPreview
                        draftId={draft.id}
                        config={effectiveLingbotConfig}
                      />
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
                        config auto-populated from probe — tweak on the right,
                        then start
                      </span>
                      <div style={{ display: "flex", gap: 6 }}>
                        <button
                          type="button"
                          onClick={reset}
                          disabled={submitting}
                        >
                          discard
                        </button>
                        <button
                          type="button"
                          onClick={start}
                          disabled={startDisabled}
                        >
                          {submitting ? "starting..." : startLabel}
                        </button>
                      </div>
                    </div>
                  </>
                )}

                {mode === "gsplat" && (
                  <>
                    <div className="mono-small" style={{ opacity: 0.8 }}>
                      gsplat training runs on the gs worker container and
                      reuses frames + poses + initial cloud from the source
                      job picked above.
                    </div>
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "flex-end",
                        gap: 6,
                      }}
                    >
                      <button
                        type="button"
                        onClick={start}
                        disabled={startDisabled}
                      >
                        {submitting ? "starting..." : startLabel}
                      </button>
                    </div>
                  </>
                )}

                {error && (
                  <div
                    style={{
                      color: "var(--danger)",
                      fontSize: "var(--fs-sm)",
                    }}
                  >
                    {error}
                  </div>
                )}
              </div>
            </div>
          </div>

          <div style={{ display: "grid", gap: 12 }}>
            <ExecutionPanel
              value={executionFields}
              onChange={(patch) =>
                setExecutionFields((prev) => ({ ...prev, ...patch }))
              }
              readOnly={submitting}
            />
            {mode === "lingbot" && (
              <ConfigPanel
                config={effectiveLingbotConfig}
                onChange={(patch) =>
                  setLingbotConfig({ ...effectiveLingbotConfig, ...patch })
                }
                readOnly={!draft}
                title="config · lingbot"
              />
            )}
          {mode === "slam" && (
            <SlamConfigPanel
              config={currentSlamConfig}
              onChange={(patch) =>
                setSlamConfigs((prev) => ({
                  ...prev,
                  [slamBackend]: {
                    ...prev[slamBackend],
                    ...patch,
                  } as SlamConfig,
                }))
              }
              readOnly={!draft}
              title={`config · ${slamBackend.replace("_", " ")}`}
            />
          )}
          {mode === "gsplat" && (
            <GsplatConfigPanel
              config={
                {
                  ...gsplatConfig,
                  processor: "gsplat",
                  source_job_id: gsplatSourceId ?? "",
                } as GsplatConfig
              }
              onChange={(patch) => {
                // Strip the read-only fields the panel includes for display.
                const {
                  processor: _p,
                  source_job_id: _s,
                  ...rest
                } = patch as Partial<GsplatConfig>;
                void _p;
                void _s;
                setGsplatConfig((prev) => ({ ...prev, ...rest }));
              }}
              readOnly={!gsplatSourceId}
              title="config · gaussian splat"
            />
          )}
          </div>
        </section>

        <JobList />
      </main>
    </div>
  );
}

function DEFAULT_GSPLAT_CONFIGS_DEFAULT(): Omit<
  GsplatConfig,
  "source_job_id"
> {
  // Clone the exported defaults so per-user tweaks don't mutate the import.
  return { ...DEFAULT_GSPLAT_CONFIG };
}
