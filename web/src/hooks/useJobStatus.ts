"use client";

import { useMemo } from "react";

import type { JobEvent, JobStatus } from "@/lib/types";

export type StageName =
  | "queue"
  | "ingest"
  | "checkpoint"
  | "inference"
  | "export";

export interface StageState {
  name: StageName;
  started_at: number | null; // ms timestamp
  ended_at: number | null;
  duration_s: number | null;
  latest_progress: number | null; // 0..1, the last progress emitted while in this stage
  state: "pending" | "active" | "done" | "failed";
  latest_message: string | null;
}

export interface VramSample {
  t: number; // event timestamp, ms
  allocated_gb: number;
}

export interface VramStats {
  current_gb: number | null;
  peak_gb: number | null;
  limit_gb: number | null;
  total_gb: number | null;
  samples: VramSample[];
}

export interface FrameStats {
  done: number | null;
  total: number | null;
  percent: number | null;
  fps: number | null;
  eta_s: number | null;
}

export interface JobStatusDerived {
  stages: StageState[];
  activeStage: StageName | null;
  frame: FrameStats;
  vram: VramStats;
  elapsed_s: number | null;
}

const STAGE_ORDER: StageName[] = [
  "queue",
  "ingest",
  "checkpoint",
  "inference",
  "export",
];

const STAGE_SET = new Set<string>(STAGE_ORDER);

function parseTs(iso: string): number {
  const t = Date.parse(iso);
  return Number.isNaN(t) ? Date.now() : t;
}

export function useJobStatus(
  events: JobEvent[],
  jobStatus: JobStatus | undefined,
): JobStatusDerived {
  return useMemo(() => {
    // seed stages
    const byName = new Map<StageName, StageState>();
    for (const name of STAGE_ORDER) {
      byName.set(name, {
        name,
        started_at: null,
        ended_at: null,
        duration_s: null,
        latest_progress: null,
        state: "pending",
        latest_message: null,
      });
    }

    let firstTs: number | null = null;
    let lastStage: StageName | null = null;
    let frameDone: number | null = null;
    let frameTotal: number | null = null;
    let frameFirstTs: number | null = null;
    let frameLastTs: number | null = null;

    const vramSamples: VramSample[] = [];
    let vramCurrent: number | null = null;
    let vramPeak: number | null = null;
    let vramLimit: number | null = null;
    let vramTotal: number | null = null;

    for (const ev of events) {
      if (!STAGE_SET.has(ev.stage)) continue;
      const t = parseTs(ev.created_at);
      if (firstTs === null) firstTs = t;
      const name = ev.stage as StageName;
      const s = byName.get(name)!;

      if (s.started_at === null) s.started_at = t;
      s.latest_message = ev.message || s.latest_message;
      if (typeof ev.progress === "number") s.latest_progress = ev.progress;

      // Infer stage transitions: when a new stage appears, previous stages in
      // the pipeline order are considered done.
      if (lastStage !== null && lastStage !== name) {
        const prevIdx = STAGE_ORDER.indexOf(lastStage);
        const curIdx = STAGE_ORDER.indexOf(name);
        if (prevIdx >= 0 && curIdx > prevIdx) {
          const prev = byName.get(lastStage)!;
          if (prev.ended_at === null) {
            prev.ended_at = t;
            if (prev.started_at !== null) {
              prev.duration_s = (t - prev.started_at) / 1000;
            }
            prev.state = "done";
            if (prev.latest_progress === null) prev.latest_progress = 1;
          }
        }
      }
      lastStage = name;

      // Inference frame counter
      if (name === "inference" && ev.data) {
        const done = ev.data["done"];
        const total = ev.data["total"];
        if (typeof done === "number" && typeof total === "number" && total > 0) {
          frameDone = done;
          frameTotal = total;
          if (frameFirstTs === null) frameFirstTs = t;
          frameLastTs = t;
        }
      }

      // VRAM readings from the watchdog
      const alloc = ev.data ? (ev.data["vram_allocated_gb"] as number | undefined) : undefined;
      const peak = ev.data ? (ev.data["vram_peak_gb"] as number | undefined) : undefined;
      const total = ev.data ? (ev.data["vram_total_gb"] as number | undefined) : undefined;
      const limit = ev.data ? (ev.data["vram_soft_limit_gb"] as number | undefined) : undefined;
      if (typeof alloc === "number") {
        vramSamples.push({ t, allocated_gb: alloc });
        vramCurrent = alloc;
      }
      if (typeof peak === "number") vramPeak = peak;
      if (typeof total === "number") vramTotal = total;
      if (typeof limit === "number") vramLimit = limit;
    }

    // Decide active stage + finalize terminal states.
    const terminal = jobStatus === "ready" || jobStatus === "failed";
    let activeStage: StageName | null = null;
    if (!terminal && lastStage !== null) {
      activeStage = lastStage;
      const s = byName.get(lastStage)!;
      if (s.state === "pending") s.state = "active";
    }

    if (terminal) {
      // All started stages are done; unstarted stages stay pending; on failure
      // the last-seen stage flips to failed.
      for (const name of STAGE_ORDER) {
        const s = byName.get(name)!;
        if (s.started_at !== null) {
          if (s.ended_at === null) {
            s.ended_at = Date.now();
            s.duration_s =
              s.started_at !== null
                ? (s.ended_at - s.started_at) / 1000
                : null;
          }
          s.state = "done";
          if (s.latest_progress === null) s.latest_progress = 1;
        }
      }
      if (jobStatus === "failed" && lastStage !== null) {
        byName.get(lastStage)!.state = "failed";
      }
    } else if (lastStage !== null) {
      // Active stage hasn't ended yet.
      const s = byName.get(lastStage)!;
      s.state = "active";
    }

    // Frame stats — if we haven't seen frame events yet but we're mid-inference,
    // fall back to the stage's latest_progress.
    let percent: number | null = null;
    if (frameDone !== null && frameTotal !== null && frameTotal > 0) {
      percent = frameDone / frameTotal;
    } else {
      const s = byName.get("inference")!;
      if (s.latest_progress !== null) percent = s.latest_progress;
    }
    let fps: number | null = null;
    let eta: number | null = null;
    if (
      frameDone !== null &&
      frameTotal !== null &&
      frameFirstTs !== null &&
      frameLastTs !== null &&
      frameLastTs > frameFirstTs &&
      frameDone > 0
    ) {
      const seconds = (frameLastTs - frameFirstTs) / 1000;
      fps = frameDone / seconds;
      if (fps > 0 && frameTotal > frameDone) {
        eta = (frameTotal - frameDone) / fps;
      }
    }

    const stages = STAGE_ORDER.map((n) => byName.get(n)!);
    const elapsed_s =
      firstTs !== null ? (Date.now() - firstTs) / 1000 : null;

    return {
      stages,
      activeStage,
      frame: {
        done: frameDone,
        total: frameTotal,
        percent,
        fps,
        eta_s: eta,
      },
      vram: {
        current_gb: vramCurrent,
        peak_gb: vramPeak,
        limit_gb: vramLimit,
        total_gb: vramTotal,
        samples: vramSamples.slice(-120), // keep last ~4 min at 2s interval
      },
      elapsed_s,
    };
  }, [events, jobStatus]);
}
