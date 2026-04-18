"use client";

import { useQuery } from "@tanstack/react-query";

import { artifactUrl } from "@/lib/api";
import type { CameraPose } from "@/components/Viewer/CameraPath";

export interface CameraPathFile {
  fps: number;
  poses: CameraPose[];
}

export function useCameraPath(
  jobId: string,
  available: boolean,
): {
  data: CameraPathFile | null;
  isLoading: boolean;
} {
  const q = useQuery<CameraPathFile>({
    queryKey: ["camera-path", jobId],
    enabled: available,
    staleTime: Infinity,
    queryFn: async () => {
      const res = await fetch(artifactUrl(jobId, "camera_path.json"));
      if (!res.ok) throw new Error(`camera_path.json: ${res.status}`);
      return res.json();
    },
  });
  return { data: q.data ?? null, isLoading: q.isLoading };
}
