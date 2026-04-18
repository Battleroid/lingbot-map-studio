"use client";

import { useQuery } from "@tanstack/react-query";

import { getManifest, listJobs } from "@/lib/api";
import type { JobManifest, JobSummary } from "@/lib/types";

export function useJobList() {
  return useQuery<JobSummary[]>({
    queryKey: ["jobs"],
    queryFn: listJobs,
    refetchInterval: 3000,
  });
}

export function useJobManifest(id: string, enabled = true) {
  return useQuery<JobManifest>({
    queryKey: ["job", id, "manifest"],
    queryFn: () => getManifest(id),
    enabled,
    refetchInterval: (query) => {
      const data = query.state.data;
      if (!data) return 1500;
      return data.status === "ready" || data.status === "failed" ? 10_000 : 1500;
    },
  });
}
