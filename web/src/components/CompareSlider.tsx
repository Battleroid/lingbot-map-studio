"use client";

import { useEffect, useRef, useState } from "react";

interface Props {
  leftSrc: string;
  rightSrc: string;
  leftLabel: string;
  rightLabel: string;
  alt: string;
}

/**
 * Before/after compare slider. Stacks both images at the same position and
 * uses clip-path on the right layer to reveal whatever fraction the user
 * dragged the divider to. Pointer events live on the container so the drag
 * works anywhere, not just on the handle itself.
 */
export function CompareSlider({
  leftSrc,
  rightSrc,
  leftLabel,
  rightLabel,
  alt,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [pct, setPct] = useState(50);
  const [dragging, setDragging] = useState(false);
  const [leftLoaded, setLeftLoaded] = useState(false);
  const [rightLoaded, setRightLoaded] = useState(false);
  const [leftErr, setLeftErr] = useState<string | null>(null);
  const [rightErr, setRightErr] = useState<string | null>(null);

  useEffect(() => {
    setLeftLoaded(false);
    setLeftErr(null);
  }, [leftSrc]);
  useEffect(() => {
    setRightLoaded(false);
    setRightErr(null);
  }, [rightSrc]);

  useEffect(() => {
    if (!dragging) return;
    function move(e: PointerEvent) {
      const el = containerRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const x = Math.min(Math.max(0, e.clientX - rect.left), rect.width);
      setPct((x / rect.width) * 100);
    }
    function up() {
      setDragging(false);
    }
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    return () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
  }, [dragging]);

  const bothLoaded = leftLoaded && rightLoaded;
  const anyErr = leftErr || rightErr;

  return (
    <div
      ref={containerRef}
      className="compare"
      onPointerDown={(e) => {
        const rect = containerRef.current?.getBoundingClientRect();
        if (!rect) return;
        const x = Math.min(Math.max(0, e.clientX - rect.left), rect.width);
        setPct((x / rect.width) * 100);
        setDragging(true);
      }}
    >
      {!anyErr && (
        <>
          {/* eslint-disable @next/next/no-img-element */}
          <img
            src={leftSrc}
            alt={`${alt} · ${leftLabel}`}
            className="compare-img"
            onLoad={() => setLeftLoaded(true)}
            onError={() => setLeftErr("load failed")}
            draggable={false}
          />
          <img
            src={rightSrc}
            alt={`${alt} · ${rightLabel}`}
            className="compare-img compare-right"
            style={{ clipPath: `inset(0 0 0 ${pct}%)` }}
            onLoad={() => setRightLoaded(true)}
            onError={() => setRightErr("load failed")}
            draggable={false}
          />
          {/* eslint-enable @next/next/no-img-element */}
          <div
            className="compare-divider"
            style={{ left: `${pct}%` }}
            aria-hidden
          >
            <span className="compare-handle">‖</span>
          </div>
          <span className="compare-label left">{leftLabel}</span>
          <span className="compare-label right">{rightLabel}</span>
        </>
      )}
      {!bothLoaded && !anyErr && (
        <div className="compare-status">rendering…</div>
      )}
      {anyErr && <div className="compare-status err">load failed</div>}
    </div>
  );
}
