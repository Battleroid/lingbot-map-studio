"use client";

import { useEffect, useRef, type CSSProperties, type ReactElement } from "react";
import * as THREE from "three";

/**
 * ThreeTile — renders one of the registered monochrome three.js tiles
 * into an inline canvas. Ported from the v2 design-bundle's three-tiles.js
 * IIFE, with the single-shared-renderer architecture kept intact.
 *
 * Design constraints:
 *  - All tiles share ONE WebGLRenderer (browsers cap ~16 contexts/page).
 *    Each tile's on-page <canvas> is a 2D canvas we blit into each frame.
 *  - 24fps cap — the animations are accents, not the main viewer.
 *  - Offscreen tiles pause via IntersectionObserver.
 *  - Monochrome: black geometry on transparent background so a white /
 *    soft / soft-2 parent always reads correctly.
 */

export type TileId =
  | "lingbot"
  | "slam"
  | "gsplat"
  | "mast3r"
  | "droid"
  | "dpvo"
  | "monogs"
  | "stage_ingest"
  | "stage_preproc"
  | "stage_inference"
  | "stage_meshing"
  | "stage_export"
  | "axis_gizmo";

interface Props {
  tile: TileId;
  height?: number | string;
  className?: string;
  style?: CSSProperties;
  ariaLabel?: string;
}

type TileInstance = {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  step: (t: number) => void;
  dispose: () => void;
};

type Record = {
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  inst: TileInstance;
  visible: boolean;
  io: IntersectionObserver;
};

const FPS = 24;
const FRAME_MS = 1000 / FPS;
const TAU = Math.PI * 2;

let sharedCanvas: HTMLCanvasElement | null = null;
let renderer: THREE.WebGLRenderer | null = null;
const records = new Set<Record>();
let rafStarted = false;
let lastFrame = 0;
let epoch = 0;

function pr(): number {
  return typeof window === "undefined"
    ? 1
    : Math.min(window.devicePixelRatio || 1, 2);
}

function getRenderer(): THREE.WebGLRenderer {
  if (renderer) return renderer;
  sharedCanvas = document.createElement("canvas");
  sharedCanvas.width = 256;
  sharedCanvas.height = 256;
  renderer = new THREE.WebGLRenderer({
    canvas: sharedCanvas,
    antialias: true,
    alpha: true,
    preserveDrawingBuffer: true,
  });
  renderer.setPixelRatio(pr());
  renderer.setClearColor(0x000000, 0);
  return renderer;
}

function sizeCanvas(c: HTMLCanvasElement): { w: number; h: number } {
  const r = c.getBoundingClientRect();
  const w = Math.max(1, Math.round(r.width));
  const h = Math.max(1, Math.round(r.height));
  const wp = Math.round(w * pr());
  const hp = Math.round(h * pr());
  if (c.width !== wp || c.height !== hp) {
    c.width = wp;
    c.height = hp;
  }
  return { w: wp, h: hp };
}

function loop(now: number): void {
  requestAnimationFrame(loop);
  if (now - lastFrame < FRAME_MS) return;
  lastFrame = now;
  if (records.size === 0) return;
  const t = (now - epoch) / 1000;
  const R = getRenderer();
  const shared = sharedCanvas!;
  for (const rec of records) {
    if (!rec.visible) continue;
    const { w, h } = sizeCanvas(rec.canvas);
    if (shared.width !== w || shared.height !== h) {
      shared.width = w;
      shared.height = h;
      R.setSize(w, h, false);
    }
    R.setSize(w, h, false);
    rec.inst.camera.aspect = w / h;
    rec.inst.camera.updateProjectionMatrix();
    rec.inst.step(t);
    R.render(rec.inst.scene, rec.inst.camera);
    rec.ctx.clearRect(0, 0, w, h);
    rec.ctx.drawImage(shared, 0, 0, w, h);
  }
}

function ensureLoop(): void {
  if (rafStarted) return;
  rafStarted = true;
  epoch = performance.now();
  requestAnimationFrame(loop);
}

// Tile registry — populated below by ./three-tile-factories.
// Each factory builds its own scene + camera + step function; the
// returned `dispose` is called when the tile unmounts.
import { FACTORIES } from "./three-tile-factories";

export function ThreeTile({
  tile,
  height = 60,
  className,
  style,
  ariaLabel,
}: Props): ReactElement {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const factory = FACTORIES[tile];
    if (!factory) {
      console.warn("[ThreeTile] unknown tile:", tile);
      return;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const inst = factory();
    const rec: Record = {
      canvas,
      ctx,
      inst,
      visible: true,
      io: new IntersectionObserver(
        (es) => {
          for (const e of es) rec.visible = e.isIntersecting;
        },
        { threshold: 0.01 },
      ),
    };
    rec.io.observe(canvas);
    records.add(rec);
    ensureLoop();
    return () => {
      rec.io.disconnect();
      records.delete(rec);
      try {
        inst.dispose();
      } catch {
        /* ignore */
      }
    };
  }, [tile]);
  return (
    <canvas
      ref={ref}
      className={className}
      aria-label={ariaLabel}
      role={ariaLabel ? "img" : undefined}
      style={{ width: "100%", display: "block", height, ...style }}
    />
  );
}

// Re-export so callers can use a type guard etc.
export { TAU };
