"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

import { ConfigPanel } from "@/components/ConfigPanel";
import { JobList } from "@/components/JobList";
import { UploadDropzone } from "@/components/UploadDropzone";
import { createJob } from "@/lib/api";
import { DEFAULT_CONFIG, type JobConfig } from "@/lib/types";

export default function Home() {
  const router = useRouter();
  const [files, setFiles] = useState<File[]>([]);
  const [config, setConfig] = useState<JobConfig>(DEFAULT_CONFIG);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit() {
    setError(null);
    if (!files.length) {
      setError("add at least one video");
      return;
    }
    setSubmitting(true);
    try {
      const { id } = await createJob(files, config);
      router.push(`/jobs/${id}`);
    } catch (e) {
      setError(String((e as Error).message));
      setSubmitting(false);
    }
  }

  return (
    <main
      style={{
        maxWidth: 1000,
        margin: "0 auto",
        padding: "20px 24px 60px",
        display: "grid",
        gap: 16,
      }}
    >
      <header
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          borderBottom: "1px solid var(--rule)",
          paddingBottom: 10,
        }}
      >
        <h1 style={{ fontSize: 16, letterSpacing: "0.08em" }}>
          LINGBOT-MAP STUDIO
        </h1>
        <div style={{ fontSize: 11, color: "var(--muted)" }}>
          feed-forward 3d reconstruction · local gpu
        </div>
      </header>

      <section style={{ display: "grid", gap: 16, gridTemplateColumns: "1fr 320px" }}>
        <div className="panel">
          <div className="panel-header">new job</div>
          <div className="panel-body" style={{ display: "grid", gap: 12 }}>
            <UploadDropzone files={files} onChange={setFiles} disabled={submitting} />
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div style={{ fontSize: 11, color: "var(--muted)" }}>
                {files.length} video{files.length === 1 ? "" : "s"} · frames concatenated in order
              </div>
              <button type="button" onClick={submit} disabled={submitting || !files.length}>
                {submitting ? "uploading..." : "start reconstruction"}
              </button>
            </div>
            {error && <div style={{ color: "var(--danger)", fontSize: 12 }}>{error}</div>}
          </div>
        </div>
        <ConfigPanel config={config} onChange={(p) => setConfig({ ...config, ...p })} />
      </section>

      <JobList />
    </main>
  );
}
