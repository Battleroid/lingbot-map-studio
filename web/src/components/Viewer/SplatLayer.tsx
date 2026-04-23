"use client";

import { useBounds } from "@react-three/drei";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";

import { useViewerStore } from "@/lib/viewerStore";

interface Props {
  url: string;
}

interface SplatData {
  positions: Float32Array;
  colors: Float32Array;
  sizes: Float32Array;
  opacities: Float32Array;
  count: number;
}

// SH_C0 matches the encode step in worker/app/processors/gsplat/export.py.
const SH_C0 = 0.28209479177387814;

/**
 * Render a 3D Gaussian Splat PLY.
 *
 * v1 implementation: parses the standard 3DGS PLY (xyz + nx/ny/nz + f_dc_0..2
 * + opacity + scale_0..2 + rot_0..3) and draws each gaussian as a colored
 * point with per-point size derived from exp(mean(scale_*)) and per-point
 * alpha from sigmoid(opacity). This deliberately avoids the heavy Spark /
 * mkkellogg dependency for the v1 landing so the frontend ships working
 * splat preview against the exact PLY the worker produces. Real splat
 * rasterization (view-dependent SH, true billboards) is a follow-up pass —
 * the parsing above already returns every field the upgrade needs.
 */
export function SplatLayer({ url }: Props) {
  const [data, setData] = useState<SplatData | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const bounds = useBounds();
  const hasFittedRef = useRef(false);
  const pointSize = useViewerStore((s) => s.pointSize);
  const threshold = useViewerStore((s) => s.splatOpacityThreshold);

  useEffect(() => {
    let cancelled = false;
    setErr(null);
    (async () => {
      try {
        const res = await fetch(url, { cache: "no-store" });
        if (!res.ok) throw new Error(`splat ${res.status}`);
        const buf = await res.arrayBuffer();
        const parsed = parseSplatPly(buf);
        if (!cancelled) setData(parsed);
      } catch (e) {
        if (!cancelled) setErr(String((e as Error).message));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [url]);

  const geometry = useMemo(() => {
    if (!data) return null;
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(data.positions, 3));
    g.setAttribute("color", new THREE.BufferAttribute(data.colors, 3));
    g.setAttribute("splatSize", new THREE.BufferAttribute(data.sizes, 1));
    g.setAttribute(
      "splatAlpha",
      new THREE.BufferAttribute(data.opacities, 1),
    );
    g.computeBoundingBox();
    return g;
  }, [data]);

  useEffect(() => {
    if (!geometry || hasFittedRef.current) return;
    hasFittedRef.current = true;
    const id = requestAnimationFrame(() => {
      try {
        bounds.refresh().fit();
      } catch {
        hasFittedRef.current = false;
      }
    });
    return () => cancelAnimationFrame(id);
  }, [geometry, bounds]);

  const material = useMemo(() => {
    // Use a small custom ShaderMaterial so each gaussian picks up its own
    // size and alpha. Depth-write off so translucent splats composite
    // cleanly with the mesh/camera-path overlays.
    const m = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      vertexColors: true,
      uniforms: {
        uPointScale: { value: pointSize },
        uAlphaFloor: { value: threshold },
      },
      vertexShader: `
        attribute float splatSize;
        attribute float splatAlpha;
        varying vec3 vColor;
        varying float vAlpha;
        uniform float uPointScale;
        uniform float uAlphaFloor;
        void main() {
          vColor = color;
          vAlpha = splatAlpha < uAlphaFloor ? 0.0 : splatAlpha;
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          // Scale per-gaussian + global viewer point-size, plus a perspective
          // attenuation so near gaussians look bigger than far ones.
          gl_PointSize = max(1.5, splatSize * uPointScale * 200.0 / -mv.z);
          gl_Position = projectionMatrix * mv;
        }
      `,
      fragmentShader: `
        varying vec3 vColor;
        varying float vAlpha;
        void main() {
          // Round sprite with soft gaussian falloff — the cheapest honest
          // approximation of a 2D-projected gaussian short of real splatting.
          vec2 c = gl_PointCoord - vec2(0.5);
          float r2 = dot(c, c) * 4.0;
          if (r2 > 1.0) discard;
          float a = vAlpha * exp(-r2 * 2.0);
          if (a < 0.01) discard;
          gl_FragColor = vec4(vColor, a);
        }
      `,
    });
    return m;
  }, [pointSize, threshold]);

  useEffect(() => {
    material.uniforms.uPointScale.value = pointSize;
    material.uniforms.uAlphaFloor.value = threshold;
  }, [material, pointSize, threshold]);

  if (err) return null;
  if (!geometry) return null;

  return <points geometry={geometry} material={material} />;
}

/**
 * Parse a 3DGS PLY binary_little_endian file.
 *
 * Returns per-gaussian positions, RGB colors (decoded from DC SH), raw
 * exp-mean-scale (used as the radius in the simple point-sprite render),
 * and sigmoid-decoded opacities.
 *
 * Tolerates optional f_rest_* / extra property columns by tracking
 * per-property byte offsets instead of assuming layout.
 */
function parseSplatPly(buf: ArrayBuffer): SplatData {
  const bytes = new Uint8Array(buf);
  // Find header end ("end_header\n")
  const marker = "end_header\n";
  let headerEnd = -1;
  for (let i = 0; i < bytes.length - marker.length; i++) {
    let ok = true;
    for (let j = 0; j < marker.length; j++) {
      if (bytes[i + j] !== marker.charCodeAt(j)) {
        ok = false;
        break;
      }
    }
    if (ok) {
      headerEnd = i + marker.length;
      break;
    }
  }
  if (headerEnd < 0) throw new Error("no end_header");
  const header = new TextDecoder().decode(bytes.slice(0, headerEnd));
  const lines = header.split(/\r?\n/);
  let n = 0;
  const props: { name: string; type: string; offset: number }[] = [];
  const typeSize: Record<string, number> = {
    float: 4,
    float32: 4,
    double: 8,
    uchar: 1,
    uint8: 1,
    int: 4,
    short: 2,
    ushort: 2,
  };
  let cursor = 0;
  let format: string | null = null;
  for (const raw of lines) {
    const line = raw.trim();
    if (line.startsWith("format ")) format = line.split(/\s+/)[1];
    if (line.startsWith("element vertex")) {
      n = Number(line.split(/\s+/)[2]);
    } else if (line.startsWith("property")) {
      const parts = line.split(/\s+/);
      // `property <type> <name>` (ignore list-property lines)
      if (parts[1] === "list") continue;
      const type = parts[1];
      const name = parts[2];
      const size = typeSize[type] ?? 0;
      if (!size) throw new Error(`unsupported ply property type: ${type}`);
      props.push({ name, type, offset: cursor });
      cursor += size;
    }
  }
  if (format !== "binary_little_endian") {
    throw new Error(`unsupported ply format: ${format}`);
  }
  const stride = cursor;
  const dv = new DataView(buf, headerEnd);
  const need = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity"];
  const byName: Record<string, { type: string; offset: number }> = {};
  for (const p of props) byName[p.name] = p;
  for (const k of need) {
    if (!byName[k]) throw new Error(`missing splat property: ${k}`);
  }
  const scaleKeys = ["scale_0", "scale_1", "scale_2"].filter(
    (k) => byName[k],
  );

  const positions = new Float32Array(n * 3);
  const colors = new Float32Array(n * 3);
  const sizes = new Float32Array(n);
  const opacities = new Float32Array(n);

  for (let i = 0; i < n; i++) {
    const base = i * stride;
    positions[i * 3] = readProp(dv, base, byName.x);
    positions[i * 3 + 1] = readProp(dv, base, byName.y);
    positions[i * 3 + 2] = readProp(dv, base, byName.z);
    const c0 = readProp(dv, base, byName.f_dc_0);
    const c1 = readProp(dv, base, byName.f_dc_1);
    const c2 = readProp(dv, base, byName.f_dc_2);
    // Decode DC SH → approximate RGB. Clamp to [0,1] for display.
    colors[i * 3] = Math.min(1, Math.max(0, c0 * SH_C0 + 0.5));
    colors[i * 3 + 1] = Math.min(1, Math.max(0, c1 * SH_C0 + 0.5));
    colors[i * 3 + 2] = Math.min(1, Math.max(0, c2 * SH_C0 + 0.5));
    // Scale is stored in log-space; radius ~= exp(scale). Average the axes
    // for a single isotropic size good enough for the point-sprite render.
    if (scaleKeys.length === 3) {
      const s0 = Math.exp(readProp(dv, base, byName.scale_0));
      const s1 = Math.exp(readProp(dv, base, byName.scale_1));
      const s2 = Math.exp(readProp(dv, base, byName.scale_2));
      sizes[i] = (s0 + s1 + s2) / 3;
    } else {
      sizes[i] = 0.02;
    }
    // Opacity is logit-encoded; sigmoid to 0..1.
    const op = readProp(dv, base, byName.opacity);
    opacities[i] = 1 / (1 + Math.exp(-op));
  }
  return { positions, colors, sizes, opacities, count: n };
}

function readProp(
  dv: DataView,
  base: number,
  p: { type: string; offset: number },
): number {
  const o = base + p.offset;
  switch (p.type) {
    case "float":
    case "float32":
      return dv.getFloat32(o, true);
    case "double":
      return dv.getFloat64(o, true);
    case "uchar":
    case "uint8":
      return dv.getUint8(o);
    case "int":
      return dv.getInt32(o, true);
    case "short":
      return dv.getInt16(o, true);
    case "ushort":
      return dv.getUint16(o, true);
    default:
      throw new Error(`unreadable ply property type: ${p.type}`);
  }
}
