"use client";

import { useEffect, useMemo, useState } from "react";

import { CompareSlider } from "@/components/CompareSlider";
import { fisheyePreviewUrl, osdPreviewUrl } from "@/lib/api";
import type { JobConfig } from "@/lib/types";

interface Props {
  draftId: string;
  config: JobConfig;
}

function useDebounced<T>(value: T, ms = 400): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return v;
}

function PreviewTile({
  src,
  label,
  alt,
}: {
  src: string;
  label: string;
  alt: string;
}) {
  const [loaded, setLoaded] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    setLoaded(false);
    setErr(null);
  }, [src]);
  return (
    <div
      style={{
        border: "1px solid var(--rule)",
        background: "#0b0b0b",
        display: "flex",
        flexDirection: "column",
        minHeight: 0,
      }}
    >
      <div
        style={{
          background: "var(--fg)",
          color: "var(--bg)",
          padding: "3px 8px",
          fontSize: "var(--fs-xs)",
          textTransform: "uppercase",
          letterSpacing: "0.08em",
          display: "flex",
          justifyContent: "space-between",
        }}
      >
        <span>{label}</span>
        {!loaded && !err && <span style={{ opacity: 0.7 }}>rendering…</span>}
        {err && <span style={{ color: "var(--bg)", opacity: 0.9 }}>error</span>}
      </div>
      <div
        style={{
          position: "relative",
          aspectRatio: "16 / 9",
          background: "#0b0b0b",
          overflow: "hidden",
        }}
      >
        {!err && (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={src}
            alt={alt}
            onLoad={() => setLoaded(true)}
            onError={() => setErr("load failed")}
            style={{
              width: "100%",
              height: "100%",
              objectFit: "contain",
              display: "block",
              opacity: loaded ? 1 : 0,
              transition: "opacity 120ms linear",
            }}
          />
        )}
        {err && (
          <div
            style={{
              position: "absolute",
              inset: 0,
              display: "grid",
              placeItems: "center",
              color: "#ff8080",
              fontSize: "var(--fs-xs)",
              padding: "0 12px",
              textAlign: "center",
            }}
          >
            {err}
          </div>
        )}
      </div>
    </div>
  );
}

export function PreprocPreview({ draftId, config }: Props) {
  const inFov = useDebounced(config.fisheye_in_fov, 500);
  const outFov = useDebounced(config.fisheye_out_fov, 500);
  const samples = useDebounced(config.osd_mask_samples, 500);
  const stdT = useDebounced(config.osd_mask_std_threshold, 500);
  const dilate = useDebounced(config.osd_mask_dilate, 500);
  const detectText = useDebounced(config.osd_detect_text, 500);
  const edgeFrac = useDebounced(config.osd_edge_persist_frac, 500);
  const useFisheye = config.preproc_fisheye;
  const useOsd = config.preproc_osd_mask;

  const beforeUrl = useMemo(
    () => fisheyePreviewUrl(draftId, { side: "before" }),
    [draftId],
  );
  const afterUrl = useMemo(
    () =>
      fisheyePreviewUrl(draftId, {
        side: "after",
        in_fov: inFov,
        out_fov: outFov,
      }),
    [draftId, inFov, outFov],
  );
  const osdUrl = useMemo(
    () =>
      osdPreviewUrl(draftId, {
        samples,
        std_threshold: stdT,
        dilate,
        detect_text: detectText,
        edge_persist_frac: edgeFrac,
        fisheye: useFisheye,
        in_fov: inFov,
        out_fov: outFov,
      }),
    [
      draftId,
      samples,
      stdT,
      dilate,
      detectText,
      edgeFrac,
      useFisheye,
      inFov,
      outFov,
    ],
  );

  if (!useFisheye && !useOsd) {
    return (
      <div className="panel">
        <div className="panel-header">
          <span>preview</span>
        </div>
        <div
          className="panel-body"
          style={{ color: "var(--muted)", fontSize: "var(--fs-xs)" }}
        >
          enable fisheye or osd masking to see a live preview of the first frame
          with those operations applied.
        </div>
      </div>
    );
  }

  const fisheyeBlock = useFisheye && (
    <section
      style={{ display: "grid", gap: 6 }}
      aria-label="fisheye preview"
    >
      <div
        className="section-title"
        style={{ display: "flex", justifyContent: "space-between" }}
      >
        <span>fisheye · drag to compare</span>
        <span style={{ color: "var(--muted)" }}>
          {inFov}° → {outFov}°
        </span>
      </div>
      <CompareSlider
        leftSrc={beforeUrl}
        rightSrc={afterUrl}
        leftLabel="before"
        rightLabel="after"
        alt="fisheye unwrap"
      />
    </section>
  );

  const osdBlock = useOsd && (
    <section style={{ display: "grid", gap: 6 }} aria-label="osd preview">
      <div
        className="section-title"
        style={{ display: "flex", justifyContent: "space-between" }}
      >
        <span>osd mask · detected regions in red</span>
        <span style={{ color: "var(--muted)" }}>
          {detectText ? `edge ${edgeFrac} + ` : ""}
          std {stdT} · dilate {dilate}
        </span>
      </div>
      <PreviewTile
        src={osdUrl}
        alt="OSD mask overlay"
        label={`osd · ${samples} samples${useFisheye ? " · post-fisheye" : ""}`}
      />
    </section>
  );

  return (
    <div className="panel">
      <div className="panel-header">
        <span>preview</span>
        <span className="meta">first frame · live</span>
      </div>
      <div
        className="panel-body"
        style={{ display: "grid", gap: 14 }}
      >
        {fisheyeBlock}
        {osdBlock}
      </div>
    </div>
  );
}
