"use client";

import { useEffect, useMemo, useState } from "react";

import { CloudCredentialsDialog } from "@/components/CloudCredentialsDialog";
import { CostPreview } from "@/components/CostPreview";
import { Tip } from "@/components/Tip";
import { listCloudProviders } from "@/lib/api";
import {
  DEFAULT_INSTANCE_SPEC,
  GPU_CLASS_OPTIONS,
  type ExecutionFields,
  type ExecutionTarget,
  type InstanceSpec,
} from "@/lib/types";

interface Props {
  value: ExecutionFields;
  onChange: (patch: Partial<ExecutionFields>) => void;
  // When true the panel renders as read-only (job already submitted).
  readOnly?: boolean;
  // Hint the panel about how long the job is expected to run so the
  // cost preview can compute a sensible estimate. Optional; defaults
  // to 15 minutes server-side.
  expectedDurationS?: number;
}

const TARGET_LABELS: Record<ExecutionTarget, string> = {
  local: "local (no cloud)",
  fake: "fake (CI/demo)",
  runpod: "runpod",
  "runpod-serverless": "runpod · serverless",
  vast: "vast.ai",
  lambda_labs: "lambda labs",
  "paperspace-core": "paperspace · core",
  "paperspace-gradient": "paperspace · gradient",
  "aws-ec2": "aws · ec2",
  "gcp-gce": "gcp · compute engine",
  "azure-vm": "azure · virtual machine",
};

// Providers that require browser-pasted credentials when the studio
// env doesn't carry them. The ExecutionPanel opens a dialog for these
// if the user picks one without the corresponding creds already live.
const CREDENTIAL_PROVIDERS = new Set<ExecutionTarget>([
  "runpod",
  "runpod-serverless",
  "vast",
  "lambda_labs",
  "paperspace-core",
  "paperspace-gradient",
  "aws-ec2",
  "gcp-gce",
  "azure-vm",
]);

const TIP_TEXT: Partial<Record<keyof ExecutionFields | string, string>> = {
  execution_target:
    "Where this job runs. `local` uses the in-process worker on this machine — no cloud cost, no spin-up wait. Every other option rents a GPU from the selected provider; the studio dispatches a pod, streams live previews back through the broker, and terminates the pod on finalize.",
  gpu_class:
    "Desired GPU class. The provider will fall back to a compatible class if the exact SKU is unavailable (RunPod in particular silently upgrades RTX 3090 → 4090 on busy days).",
  spot:
    "Prefer a spot / preemptible instance. Much cheaper (30–80% off) but the provider may evict mid-run — the orphan sweeper will mark the job failed and offer a retry button.",
  region: "Provider region. Leave blank to let the provider pick the cheapest option.",
  cost_cap_cents:
    "Per-job hard cap on cloud spend in cents. The dispatcher refuses to launch if the cost estimate exceeds this; a watchdog aborts in-flight if elapsed × hourly rate crosses the cap. Falls back to the studio default when empty.",
};

function centsInput(cents: number | null): string {
  return cents === null ? "" : String(cents);
}

export function ExecutionPanel({
  value,
  onChange,
  readOnly,
  expectedDurationS,
}: Props) {
  const [availableTargets, setAvailableTargets] = useState<ExecutionTarget[]>([
    "local",
  ]);
  const [sessionTargets, setSessionTargets] = useState<string[]>([]);
  const [defaultCostCapCents, setDefaultCostCapCents] = useState<number>(5000);
  const [credDialogFor, setCredDialogFor] = useState<ExecutionTarget | null>(
    null,
  );

  async function refreshProviders() {
    try {
      const res = await listCloudProviders();
      setAvailableTargets(res.targets);
      setSessionTargets(res.session_targets);
      setDefaultCostCapCents(res.cost_cap_cents_default);
    } catch {
      /* Provider discovery is best-effort; fall back to the defaults. */
    }
  }

  useEffect(() => {
    void refreshProviders();
  }, []);

  // Targets registered from env *or* pasted into this browser session.
  const usableTargets = useMemo<ExecutionTarget[]>(() => {
    const merged = new Set<ExecutionTarget>(availableTargets);
    for (const t of sessionTargets) {
      merged.add(t as ExecutionTarget);
    }
    return Array.from(merged);
  }, [availableTargets, sessionTargets]);

  const target = value.execution_target;
  const isRemote = target !== "local";
  const spec: InstanceSpec = value.instance_spec ?? DEFAULT_INSTANCE_SPEC;

  function setSpec(patch: Partial<InstanceSpec>): void {
    onChange({ instance_spec: { ...spec, ...patch } });
  }

  function changeTarget(next: ExecutionTarget): void {
    // Switching to a provider that needs creds but doesn't have them
    // yet (not registered from env, not in session) opens the paste
    // dialog. The picker still flips immediately so the config reflects
    // intent even before the key is pasted.
    const registered = availableTargets.includes(next);
    const inSession = sessionTargets.includes(next as string);
    const needsCreds =
      next !== "local" &&
      next !== "fake" &&
      CREDENTIAL_PROVIDERS.has(next) &&
      !registered &&
      !inSession;
    onChange({
      execution_target: next,
      instance_spec: next === "local" ? null : value.instance_spec ?? DEFAULT_INSTANCE_SPEC,
    });
    if (needsCreds) setCredDialogFor(next);
  }

  async function onCredentialsSaved(): Promise<void> {
    setCredDialogFor(null);
    await refreshProviders();
  }

  return (
    <div className="panel">
      <div className="panel-header">
        <span>execution</span>
        {readOnly && <span className="meta">locked</span>}
      </div>
      <div className="panel-body" style={{ display: "grid", gap: 6 }}>
        <label className="stat">
          <Tip text={TIP_TEXT.execution_target ?? ""}>
            <span>target</span>
          </Tip>
          <select
            value={target}
            disabled={readOnly}
            onChange={(e) => changeTarget(e.target.value as ExecutionTarget)}
          >
            {(
              ["local", ...usableTargets.filter((t) => t !== "local")] as ExecutionTarget[]
            ).map((t) => (
              <option key={t} value={t}>
                {TARGET_LABELS[t] ?? t}
              </option>
            ))}
            {/* Unregistered providers the user might want to paste creds for. */}
            {(Object.keys(TARGET_LABELS) as ExecutionTarget[])
              .filter((t) => !usableTargets.includes(t) && t !== "local")
              .map((t) => (
                <option key={t} value={t}>
                  {TARGET_LABELS[t]} · paste key to enable
                </option>
              ))}
          </select>
        </label>

        {isRemote && (
          <>
            <label className="stat">
              <Tip text={TIP_TEXT.gpu_class ?? ""}>
                <span>gpu</span>
              </Tip>
              <select
                value={spec.gpu_class}
                disabled={readOnly}
                onChange={(e) => setSpec({ gpu_class: e.target.value })}
              >
                {GPU_CLASS_OPTIONS.map((g) => (
                  <option key={g.id} value={g.id} title={g.hint}>
                    {g.label}
                  </option>
                ))}
              </select>
            </label>

            <label className="stat">
              <Tip text={TIP_TEXT.region ?? ""}>
                <span>region</span>
              </Tip>
              <input
                type="text"
                value={spec.region ?? ""}
                placeholder="auto"
                disabled={readOnly}
                onChange={(e) =>
                  setSpec({ region: e.target.value === "" ? null : e.target.value })
                }
              />
            </label>

            <label className="stat">
              <Tip text={TIP_TEXT.spot ?? ""}>
                <span>spot / preemptible</span>
              </Tip>
              <input
                type="checkbox"
                checked={spec.spot}
                disabled={readOnly}
                onChange={(e) => setSpec({ spot: e.target.checked })}
              />
            </label>

            <label className="stat">
              <Tip text={TIP_TEXT.cost_cap_cents ?? ""}>
                <span>
                  cost cap (¢) · studio default{" "}
                  <span style={{ color: "var(--muted)" }}>
                    {defaultCostCapCents}
                  </span>
                </span>
              </Tip>
              <input
                type="number"
                value={centsInput(value.cost_cap_cents)}
                min={0}
                step={25}
                placeholder="default"
                disabled={readOnly}
                onChange={(e) =>
                  onChange({
                    cost_cap_cents:
                      e.target.value === "" ? null : Number(e.target.value),
                  })
                }
              />
            </label>

            {!readOnly && (
              <div style={{ display: "flex", justifyContent: "flex-end" }}>
                <button
                  type="button"
                  onClick={() => setCredDialogFor(target)}
                >
                  paste credentials for {TARGET_LABELS[target] ?? target}
                </button>
              </div>
            )}

            <CostPreview
              target={target}
              spec={spec}
              expectedDurationS={expectedDurationS}
              costCapCents={value.cost_cap_cents ?? defaultCostCapCents}
            />
          </>
        )}

        {credDialogFor && (
          <CloudCredentialsDialog
            provider={credDialogFor}
            onClose={() => setCredDialogFor(null)}
            onSaved={onCredentialsSaved}
          />
        )}
      </div>
    </div>
  );
}
