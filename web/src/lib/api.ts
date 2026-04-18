import type {
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

export async function createJob(
  videos: File[],
  config: JobConfig,
): Promise<{ id: string }> {
  const fd = new FormData();
  for (const f of videos) fd.append("videos", f, f.name);
  fd.append("config", JSON.stringify(config));
  const res = await fetch(`${API_BASE}/api/jobs`, { method: "POST", body: fd });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`upload failed: ${res.status} ${body}`);
  }
  return res.json();
}

export function deleteJob(id: string): Promise<{ deleted: true }> {
  return fetchJson(`/api/jobs/${id}`, { method: "DELETE" });
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

export function jobStreamUrl(jobId: string): string {
  const u = new URL(API_BASE);
  const proto = u.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${u.host}/api/jobs/${jobId}/stream`;
}
