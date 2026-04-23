"use client";

import { Tip } from "@/components/Tip";

// Shared row helpers for tracking-config panels (SLAM backends + MonoGS).
// Kept in its own module so SlamConfigPanel.tsx and MonogsConfigPanel.tsx
// don't duplicate 30 lines of generic number/bool inputs.

export function NumberRow({
  label,
  tip,
  value,
  onChange,
  step = 1,
  min,
  max,
  readOnly,
  placeholder,
}: {
  label: string;
  tip?: string;
  value: number | null;
  onChange: (v: number | null) => void;
  step?: number;
  min?: number;
  max?: number;
  readOnly?: boolean;
  placeholder?: string;
}) {
  return (
    <label className="stat">
      <Tip text={tip ?? ""}>
        <span>{label}</span>
      </Tip>
      <input
        type="number"
        value={value ?? ""}
        step={step}
        min={min}
        max={max}
        readOnly={readOnly}
        disabled={readOnly}
        placeholder={placeholder}
        onChange={(e) =>
          onChange(e.target.value === "" ? null : Number(e.target.value))
        }
      />
    </label>
  );
}

export function BoolRow({
  label,
  tip,
  value,
  onChange,
  readOnly,
}: {
  label: string;
  tip?: string;
  value: boolean;
  onChange: (v: boolean) => void;
  readOnly?: boolean;
}) {
  return (
    <label className="stat">
      <Tip text={tip ?? ""}>
        <span>{label}</span>
      </Tip>
      <input
        type="checkbox"
        checked={value}
        disabled={readOnly}
        onChange={(e) => onChange(e.target.checked)}
      />
    </label>
  );
}
