"use client";

import { useEffect, useMemo, useState } from "react";

import { CompareSlider } from "@/components/CompareSlider";
import { fisheyePreviewUrl, fpvPreviewUrl, osdPreviewUrl } from "@/lib/api";
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
  const useColorNorm = config.preproc_color_norm;
  const useRs = config.preproc_rs_correction;
  const rsShear = useDebounced(config.rs_shear_px_per_row, 500);
  const deblurMode = config.preproc_deblur;
  const useDeblur = deblurMode !== "none";
  const useAnalog = config.preproc_analog_cleanup;
  const useDeflicker = config.preproc_deflicker;
  const showAnalogFfmpeg = useAnalog || useDeflicker;

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

  const colorNormUrl = useMemo(
    () => fpvPreviewUrl(draftId, { stage: "color_norm" }),
    [draftId],
  );
  const deblurUrl = useMemo(
    () => fpvPreviewUrl(draftId, { stage: "deblur" }),
    [draftId],
  );
  const rsUrl = useMemo(
    () => fpvPreviewUrl(draftId, { stage: "rs_correction", shear: rsShear }),
    [draftId, rsShear],
  );
  const analogUrl = useMemo(
    () =>
      fpvPreviewUrl(draftId, {
        stage: "analog_cleanup",
        analog_cleanup: useAnalog,
        deflicker: useDeflicker,
      }),
    [draftId, useAnalog, useDeflicker],
  );

  const hasPreview =
    useFisheye ||
    useOsd ||
    useColorNorm ||
    useRs ||
    useDeblur ||
    showAnalogFfmpeg;
  if (!hasPreview) {
    return (
      <div className="panel">
        <div className="panel-header">
          <span>preview</span>
        </div>
        <div
          className="panel-body"
          style={{ color: "var(--muted)", fontSize: "var(--fs-xs)" }}
        >
          enable any preprocessing stage to see a live preview of the first frame
          with that operation applied.
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

  const colorNormBlock = useColorNorm && (
    <section
      style={{ display: "grid", gap: 6 }}
      aria-label="colour normalisation preview"
    >
      <div className="section-title">
        <span>colour norm · drag to compare</span>
      </div>
      <CompareSlider
        leftSrc={beforeUrl}
        rightSrc={colorNormUrl}
        leftLabel="before"
        rightLabel="after"
        alt="colour normalisation"
      />
    </section>
  );

  const rsBlock = useRs && (
    <section
      style={{ display: "grid", gap: 6 }}
      aria-label="rolling-shutter preview"
    >
      <div
        className="section-title"
        style={{ display: "flex", justifyContent: "space-between" }}
      >
        <span>rolling-shutter · drag to compare</span>
        <span style={{ color: "var(--muted)" }}>
          {rsShear == null ? "shear · auto" : `shear · ${rsShear} px/row`}
        </span>
      </div>
      <CompareSlider
        leftSrc={beforeUrl}
        rightSrc={rsUrl}
        leftLabel="before"
        rightLabel="after"
        alt="rolling-shutter correction"
      />
    </section>
  );

  const deblurBlock = useDeblur && (
    <section style={{ display: "grid", gap: 6 }} aria-label="deblur preview">
      <div
        className="section-title"
        style={{ display: "flex", justifyContent: "space-between" }}
      >
        <span>deblur · drag to compare</span>
        <span style={{ color: "var(--muted)" }}>{deblurMode}</span>
      </div>
      <CompareSlider
        leftSrc={beforeUrl}
        rightSrc={deblurUrl}
        leftLabel="before"
        rightLabel="after"
        alt="motion deblur"
      />
    </section>
  );

  const analogBlock = showAnalogFfmpeg && (
    <section
      style={{ display: "grid", gap: 6 }}
      aria-label="analog cleanup preview"
    >
      <div
        className="section-title"
        style={{ display: "flex", justifyContent: "space-between" }}
      >
        <span>analog ffmpeg · drag to compare</span>
        <span style={{ color: "var(--muted)" }}>
          {[useAnalog && "atadenoise", useDeflicker && "deflicker"]
            .filter(Boolean)
            .join(" + ")}
        </span>
      </div>
      <CompareSlider
        leftSrc={beforeUrl}
        rightSrc={analogUrl}
        leftLabel="before"
        rightLabel="after"
        alt="analog ffmpeg cleanup"
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
        {analogBlock}
        {colorNormBlock}
        {rsBlock}
        {deblurBlock}
      </div>
    </div>
  );
}
