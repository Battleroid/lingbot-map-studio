"use client";

import { useEffect, useState } from "react";

import { estimateCost } from "@/lib/api";
import type { ExecutionTarget, InstanceSpec } from "@/lib/types";

interface Props {
  target: ExecutionTarget;
  spec: InstanceSpec;
  // Defaults to the server-side default (15 min) if omitted.
  expectedDurationS?: number;
  // Per-job ceiling; surfaced red if the estimate exceeds it.
  costCapCents?: number | null;
}

function formatCents(c: number | null | undefined): string {
  if (c === null || c === undefined) return "—";
  if (c < 100) return `${c}¢`;
  return `$${(c / 100).toFixed(2)}`;
}

function formatDuration(s: number | null | undefined): string {
  if (s === null || s === undefined || !isFinite(s)) return "—";
  if (s < 60) return `${Math.round(s)}s`;
  if (s < 3600) return `${Math.round(s / 60)}m`;
  return `${(s / 3600).toFixed(1)}h`;
}

// Debounce spec changes so the user dragging sliders doesn't hammer
// the provider's pricing API (or our local estimate cache).
function useDebounced<T>(value: T, ms: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const id = setTimeout(() => setDebounced(value), ms);
    return () => clearTimeout(id);
  }, [value, ms]);
  return debounced;
}

export function CostPreview({
  target,
  spec,
  expectedDurationS,
  costCapCents,
}: Props) {
  // Debounce on a serialised spec blob so deep-equal changes settle.
  const key = JSON.stringify({ target, spec, expectedDurationS });
  const debouncedKey = useDebounced(key, 300);

  const [cents, setCents] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function run() {
      if (target === "local") {
        setCents(0);
        return;
      }
      setLoading(true);
      setError(null);
      try {
        const res = await estimateCost({
          execution_target: target,
          instance_spec: spec,
          expected_duration_s: expectedDurationS,
        });
        if (!cancelled) setCents(res.cents);
      } catch (e) {
        if (!cancelled) {
          setError(String((e as Error).message));
          setCents(null);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    void run();
    return () => {
      cancelled = true;
    };
    // debouncedKey encodes target/spec/duration already.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [debouncedKey]);

  const overCap =
    costCapCents !== undefined &&
    costCapCents !== null &&
    cents !== null &&
    cents > costCapCents;

  return (
    <div
      className="stat"
      style={{
        display: "grid",
        gap: 4,
        borderTop: "1px solid var(--rule)",
        paddingTop: 6,
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between" }}>
        <span>estimated cost</span>
        <span
          style={{
            color: overCap ? "var(--danger)" : undefined,
            fontWeight: overCap ? 600 : undefined,
          }}
        >
          {loading ? "…" : error ? "n/a" : formatCents(cents)}
        </span>
      </div>
      <div
        className="mono-small"
        style={{
          display: "flex",
          justifyContent: "space-between",
          color: "var(--muted)",
        }}
      >
        <span>
          for ~{formatDuration(expectedDurationS ?? 15 * 60)} ·{" "}
          {spec.spot ? "spot" : "on-demand"}
        </span>
        {overCap && <span>exceeds cap {formatCents(costCapCents ?? 0)}</span>}
      </div>
      {error && (
        <div
          className="mono-small"
          style={{ color: "var(--danger)" }}
          title={error}
        >
          estimate failed: {error.slice(0, 80)}
        </div>
      )}
    </div>
  );
}
