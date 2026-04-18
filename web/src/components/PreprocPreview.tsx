"use client";

import { useEffect, useMemo, useState } from "react";

import { fisheyePreviewUrl, osdPreviewUrl } from "@/lib/api";
import type { JobConfig } from "@/lib/types";

interface Props {
  draftId: string;
  config: JobConfig;
}

/**
 * Debounce a changing value. Used to avoid firing a preview rerender on every
 * keystroke while dragging a slider.
 */
function useDebounced<T>(value: T, ms = 400): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return v;
}

function LoadingImg({
  src,
  alt,
  label,
}: {
  src: string;
  alt: string;
  label: string;
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
        background: "var(--bg)",
        display: "flex",
        flexDirection: "column",
        minHeight: 0,
      }}
    >
      <div
        style={{
          background: "var(--fg)",
          color: "var(--bg)",
          padding: "2px 8px",
          fontSize: "var(--fs-xs)",
          textTransform: "uppercase",
          letterSpacing: "0.08em",
          display: "flex",
          justifyContent: "space-between",
        }}
      >
        <span>{label}</span>
        {!loaded && !err && <span style={{ opacity: 0.7 }}>rendering...</span>}
        {err && <span style={{ color: "var(--bg)", opacity: 0.9 }}>error</span>}
      </div>
      <div
        style={{
          position: "relative",
          aspectRatio: "16 / 9",
          background: "var(--soft)",
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
              color: "var(--danger)",
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
  // Debounce the knobs so dragging a slider doesn't hammer the backend.
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
    [draftId, samples, stdT, dilate, detectText, edgeFrac, useFisheye, inFov, outFov],
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
          enable fisheye or osd masking on the right to see a live preview of the first
          frame with those operations applied.
        </div>
      </div>
    );
  }

  return (
    <div className="panel">
      <div className="panel-header">
        <span>preview</span>
        <span className="meta">first frame · live</span>
      </div>
      <div
        className="panel-body"
        style={{
          display: "grid",
          gap: 10,
          gridTemplateColumns: useFisheye
            ? useOsd
              ? "repeat(3, minmax(0, 1fr))"
              : "repeat(2, minmax(0, 1fr))"
            : "minmax(0, 1fr)",
        }}
      >
        {useFisheye && (
          <>
            <LoadingImg src={beforeUrl} alt="source frame" label="before · raw" />
            <LoadingImg
              src={afterUrl}
              alt="fisheye unwrapped"
              label={`after · ${inFov}°→${outFov}°`}
            />
          </>
        )}
        {useOsd && (
          <LoadingImg
            src={osdUrl}
            alt="OSD mask overlay"
            label={`osd mask · ${detectText ? `edge ${edgeFrac} + ` : ""}std ${stdT} · dilate ${dilate}`}
          />
        )}
      </div>
    </div>
  );
}
