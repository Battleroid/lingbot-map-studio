"use client";

import { useEffect, useRef, useState } from "react";

import { jobStreamUrl } from "@/lib/api";
import type { JobEvent } from "@/lib/types";
import { JobStreamClient } from "@/lib/ws";

export interface JobStreamState {
  events: JobEvent[];
  status: "connecting" | "open" | "closed";
  latestProgress: number | null;
  latestStage: string | null;
}

const MAX_EVENTS = 2000;

export function useJobStream(jobId: string): JobStreamState {
  const [events, setEvents] = useState<JobEvent[]>([]);
  const [status, setStatus] =
    useState<"connecting" | "open" | "closed">("connecting");
  const seen = useRef<Set<number>>(new Set());

  useEffect(() => {
    const client = new JobStreamClient(jobStreamUrl(jobId));
    const offEv = client.onEvent((ev) => {
      if (seen.current.has(ev.id)) return;
      seen.current.add(ev.id);
      setEvents((prev) => {
        const next = [...prev, ev];
        if (next.length > MAX_EVENTS) next.splice(0, next.length - MAX_EVENTS);
        return next;
      });
    });
    const offStatus = client.onStatus(setStatus);
    client.connect();
    return () => {
      offEv();
      offStatus();
      client.close();
    };
  }, [jobId]);

  let latestProgress: number | null = null;
  let latestStage: string | null = null;
  for (let i = events.length - 1; i >= 0; i--) {
    const ev = events[i];
    if (ev.progress !== null && latestProgress === null) latestProgress = ev.progress;
    if (latestStage === null) latestStage = ev.stage;
    if (latestProgress !== null && latestStage !== null) break;
  }

  return { events, status, latestProgress, latestStage };
}
