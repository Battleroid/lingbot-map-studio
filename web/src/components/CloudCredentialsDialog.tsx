"use client";

import { useMemo, useState } from "react";

import { setSessionCredentials } from "@/lib/api";
import type { ExecutionTarget } from "@/lib/types";

interface Props {
  provider: ExecutionTarget;
  onClose: () => void;
  onSaved: () => void;
}

// Per-provider field schemas. Adding a new provider means listing the
// env keys its adapter actually reads; the backend stores whatever
// we post under the matching key. We deliberately keep this list
// short — anything more elaborate (service-account JSON blobs, for
// example) is documented on the adapter itself.
const PROVIDER_FIELDS: Record<string, { key: string; label: string; type: "text" | "password" | "textarea"; placeholder?: string }[]> = {
  runpod: [
    { key: "api_key", label: "RunPod API key", type: "password", placeholder: "rpa_..." },
  ],
  "runpod-serverless": [
    { key: "api_key", label: "RunPod API key", type: "password", placeholder: "rpa_..." },
    { key: "endpoint_id", label: "Serverless endpoint id", type: "text", placeholder: "abc123" },
  ],
  vast: [
    { key: "api_key", label: "Vast.ai API key", type: "password" },
  ],
  lambda_labs: [
    { key: "api_key", label: "Lambda Labs API key", type: "password" },
  ],
  "paperspace-core": [
    { key: "api_key", label: "Paperspace API key", type: "password" },
  ],
  "paperspace-gradient": [
    { key: "api_key", label: "Paperspace API key", type: "password" },
  ],
  "aws-ec2": [
    { key: "access_key_id", label: "AWS access key id", type: "text", placeholder: "AKIA..." },
    { key: "secret_access_key", label: "AWS secret access key", type: "password" },
    { key: "region", label: "Default region", type: "text", placeholder: "us-east-1" },
  ],
  "gcp-gce": [
    {
      key: "service_account_json",
      label: "Service-account JSON",
      type: "textarea",
      placeholder: '{"type":"service_account", "project_id":"...", ...}',
    },
    { key: "region", label: "Default region", type: "text", placeholder: "us-central1" },
  ],
  "azure-vm": [
    { key: "subscription_id", label: "Subscription id", type: "text" },
    { key: "resource_group", label: "Resource group", type: "text" },
    { key: "nic_id", label: "Network-interface id", type: "text", placeholder: "/subscriptions/.../networkInterfaces/..." },
    { key: "image_id", label: "VM image id", type: "text", placeholder: "/subscriptions/.../images/..." },
  ],
};

export function CloudCredentialsDialog({ provider, onClose, onSaved }: Props) {
  const fields = useMemo(() => PROVIDER_FIELDS[provider] ?? [], [provider]);
  const [values, setValues] = useState<Record<string, string>>(() =>
    Object.fromEntries(fields.map((f) => [f.key, ""])),
  );
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function save() {
    setSaving(true);
    setError(null);
    try {
      const nonEmpty = Object.fromEntries(
        Object.entries(values).filter(([, v]) => v.trim() !== ""),
      );
      if (Object.keys(nonEmpty).length === 0) {
        throw new Error("enter at least one credential field");
      }
      await setSessionCredentials(provider, nonEmpty);
      onSaved();
    } catch (e) {
      setError(String((e as Error).message));
      setSaving(false);
    }
  }

  if (fields.length === 0) {
    return (
      <div className="modal-backdrop" onClick={onClose}>
        <div
          className="panel"
          style={{ maxWidth: 420, margin: "auto", marginTop: 120 }}
          onClick={(e) => e.stopPropagation()}
        >
          <div className="panel-header">
            <span>credentials for {provider}</span>
          </div>
          <div className="panel-body" style={{ display: "grid", gap: 8 }}>
            <span className="mono-small">
              no credential schema wired up for this provider. paste values
              via the studio env or extend PROVIDER_FIELDS.
            </span>
            <button type="button" onClick={onClose}>
              close
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      className="modal-backdrop"
      onClick={onClose}
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.45)",
        display: "grid",
        placeItems: "center",
        zIndex: 100,
      }}
    >
      <div
        className="panel"
        style={{ minWidth: 380, maxWidth: 520 }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="panel-header">
          <span>credentials · {provider}</span>
          <button
            type="button"
            onClick={onClose}
            style={{ background: "transparent", border: "none" }}
          >
            ×
          </button>
        </div>
        <div className="panel-body" style={{ display: "grid", gap: 8 }}>
          <div className="mono-small" style={{ opacity: 0.8 }}>
            pasted here = stored in memory only, scoped to this browser
            tab. closes with the tab, never written to disk.
          </div>
          {fields.map((f) => (
            <label key={f.key} className="stat" style={{ display: "grid", gap: 4 }}>
              <span>{f.label}</span>
              {f.type === "textarea" ? (
                <textarea
                  rows={6}
                  value={values[f.key] ?? ""}
                  placeholder={f.placeholder}
                  onChange={(e) =>
                    setValues((v) => ({ ...v, [f.key]: e.target.value }))
                  }
                />
              ) : (
                <input
                  type={f.type}
                  value={values[f.key] ?? ""}
                  placeholder={f.placeholder}
                  autoComplete="off"
                  spellCheck={false}
                  onChange={(e) =>
                    setValues((v) => ({ ...v, [f.key]: e.target.value }))
                  }
                />
              )}
            </label>
          ))}
          {error && (
            <div style={{ color: "var(--danger)", fontSize: "var(--fs-sm)" }}>
              {error}
            </div>
          )}
          <div
            style={{
              display: "flex",
              justifyContent: "flex-end",
              gap: 6,
            }}
          >
            <button type="button" onClick={onClose} disabled={saving}>
              cancel
            </button>
            <button type="button" onClick={save} disabled={saving}>
              {saving ? "saving..." : "save for this session"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
