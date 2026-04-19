import type {
  AnyJobConfig,
  Job,
  JobConfig,
  JobManifest,
  JobSummary,
  MeshEditRequest,
  ReexportRequest,
} from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}${body ? `: ${body}` : ""}`);
  }
  return res.json() as Promise<T>;
}

export function listJobs(): Promise<JobSummary[]> {
  return fetchJson("/api/jobs");
}

export function getJob(id: string): Promise<Job> {
  return fetchJson(`/api/jobs/${id}`);
}

export function getManifest(id: string): Promise<JobManifest> {
  return fetchJson(`/api/jobs/${id}/manifest`);
}

export interface ProbeResult {
  name: string;
  fps: number | null;
  duration_s: number | null;
  width: number | null;
  height: number | null;
  codec: string | null;
  pix_fmt: string | null;
  bitrate: number | null;
  total_frames: number | null;
  container: string | null;
  has_audio: boolean;
  size_bytes: number | null;
  error?: string;
}

export interface DraftRecord {
  id: string;
  created_at: number;
  uploads: string[];
  probes: ProbeResult[];
  suggested_config: Partial<JobConfig>;
}

export async function createDraft(
  videos: File[],
  onProgress?: (pct: number) => void,
): Promise<DraftRecord> {
  const fd = new FormData();
  for (const f of videos) fd.append("videos", f, f.name);
  return new Promise<DraftRecord>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}/api/drafts`);
    xhr.responseType = "text";
    if (onProgress) {
      xhr.upload.addEventListener("progress", (e) => {
        if (e.lengthComputable) onProgress(e.loaded / e.total);
      });
    }
    xhr.addEventListener("load", () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as DraftRecord);
        } catch (err) {
          reject(err);
        }
      } else {
        reject(new Error(`draft upload failed: ${xhr.status} ${xhr.responseText}`));
      }
    });
    xhr.addEventListener("error", () => reject(new Error("draft upload network error")));
    xhr.send(fd);
  });
}

export function deleteDraft(id: string): Promise<{ deleted: true }> {
  return fetchJson(`/api/drafts/${id}`, { method: "DELETE" });
}

export async function startJobFromDraft(
  draftId: string,
  config: AnyJobConfig,
): Promise<{ id: string }> {
  const fd = new FormData();
  fd.append("draft_id", draftId);
  fd.append("config", JSON.stringify(config));
  const res = await fetch(`${API_BASE}/api/jobs`, { method: "POST", body: fd });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`start failed: ${res.status} ${body}`);
  }
  return res.json();
}

export function deleteJob(id: string): Promise<{ deleted: true }> {
  return fetchJson(`/api/jobs/${id}`, { method: "DELETE" });
}

/** Kick off a gsplat training job seeded from a completed SLAM/Lingbot job. */
export function createGsplatJobFromSource(
  sourceJobId: string,
  overrides?: Record<string, unknown>,
): Promise<{ id: string }> {
  return fetchJson(`/api/jobs/gsplat-from/${sourceJobId}`, {
    method: "POST",
    body: JSON.stringify(overrides ?? {}),
  });
}

export function stopJob(
  id: string,
  force = false,
): Promise<{ cancelled: true; forced: boolean }> {
  const qs = force ? "?force=true" : "";
  return fetchJson(`/api/jobs/${id}/stop${qs}`, { method: "POST" });
}

export function restartJob(id: string): Promise<{ id: string }> {
  return fetchJson(`/api/jobs/${id}/restart`, { method: "POST" });
}

export function reexport(
  id: string,
  body: ReexportRequest,
): Promise<{ name: string; format: string; size: number }> {
  return fetchJson(`/api/jobs/${id}/reexport`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function meshEdit(
  id: string,
  body: MeshEditRequest,
): Promise<{ name: string; revision: number; vertices: number; faces: number }> {
  return fetchJson(`/api/jobs/${id}/mesh/edit`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function artifactUrl(jobId: string, name: string): string {
  return `${API_BASE}/api/jobs/${jobId}/artifacts/${encodeURIComponent(name)}`;
}

export function fisheyePreviewUrl(
  draftId: string,
  opts: { side: "before" | "after"; in_fov?: number; out_fov?: number; v?: number },
): string {
  const qs = new URLSearchParams({
    side: opts.side,
    ...(opts.in_fov !== undefined ? { in_fov: String(opts.in_fov) } : {}),
    ...(opts.out_fov !== undefined ? { out_fov: String(opts.out_fov) } : {}),
    ...(opts.v !== undefined ? { _v: String(opts.v) } : {}),
  });
  return `${API_BASE}/api/drafts/${draftId}/preview/fisheye?${qs}`;
}

export function osdPreviewUrl(
  draftId: string,
  opts: {
    samples?: number;
    std_threshold?: number;
    dilate?: number;
    detect_text?: boolean;
    edge_persist_frac?: number;
    fisheye?: boolean;
    in_fov?: number;
    out_fov?: number;
    v?: number;
  },
): string {
  const qs = new URLSearchParams();
  if (opts.samples !== undefined) qs.set("samples", String(opts.samples));
  if (opts.std_threshold !== undefined)
    qs.set("std_threshold", String(opts.std_threshold));
  if (opts.dilate !== undefined) qs.set("dilate", String(opts.dilate));
  if (opts.detect_text !== undefined)
    qs.set("detect_text", String(opts.detect_text));
  if (opts.edge_persist_frac !== undefined)
    qs.set("edge_persist_frac", String(opts.edge_persist_frac));
  if (opts.fisheye) qs.set("fisheye", "true");
  if (opts.in_fov !== undefined) qs.set("in_fov", String(opts.in_fov));
  if (opts.out_fov !== undefined) qs.set("out_fov", String(opts.out_fov));
  if (opts.v !== undefined) qs.set("_v", String(opts.v));
  return `${API_BASE}/api/drafts/${draftId}/preview/osd?${qs}`;
}

export type FpvPreviewStage =
  | "color_norm"
  | "deblur"
  | "analog_cleanup"
  | "rs_correction";

export function fpvPreviewUrl(
  draftId: string,
  opts: {
    stage: FpvPreviewStage;
    shear?: number | null;
    analog_cleanup?: boolean;
    deflicker?: boolean;
    v?: number;
  },
): string {
  const qs = new URLSearchParams({ stage: opts.stage });
  if (opts.shear !== undefined && opts.shear !== null)
    qs.set("shear", String(opts.shear));
  if (opts.analog_cleanup) qs.set("analog_cleanup", "true");
  if (opts.deflicker) qs.set("deflicker", "true");
  if (opts.v !== undefined) qs.set("_v", String(opts.v));
  return `${API_BASE}/api/drafts/${draftId}/preview/fpv?${qs}`;
}

export function jobStreamUrl(jobId: string): string {
  const u = new URL(API_BASE);
  const proto = u.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${u.host}/api/jobs/${jobId}/stream`;
}
