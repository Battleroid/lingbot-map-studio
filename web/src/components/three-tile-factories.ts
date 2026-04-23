import * as THREE from "three";

import type { TileId } from "./ThreeTile";

const TAU = Math.PI * 2;

export type TileInstance = {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  step: (t: number) => void;
  dispose: () => void;
};

type SceneCtx = {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  disposables: Array<{ dispose: () => void }>;
};

function makeScene(opts: { fov?: number; z?: number } = {}): SceneCtx {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(opts.fov ?? 40, 1, 0.1, 100);
  camera.position.set(0, 0, opts.z ?? 3.2);
  camera.lookAt(0, 0, 0);
  return { scene, camera, disposables: [] };
}

function finalize(
  ctx: SceneCtx,
  step: (t: number) => void,
): TileInstance {
  return {
    scene: ctx.scene,
    camera: ctx.camera,
    step,
    dispose: () => {
      for (const d of ctx.disposables) {
        try {
          d.dispose();
        } catch {
          /* best-effort */
        }
      }
    },
  };
}

function pts(positions: number[], mat: THREE.PointsMaterial): THREE.Points {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  return new THREE.Points(g, mat);
}

function line(
  points: THREE.Vector3[],
  mat: THREE.LineBasicMaterial,
): THREE.Line {
  const g = new THREE.BufferGeometry().setFromPoints(points);
  return new THREE.Line(g, mat);
}

function blackLineMat(opacity = 0.85): THREE.LineBasicMaterial {
  return new THREE.LineBasicMaterial({
    color: 0x000000,
    transparent: true,
    opacity,
  });
}

function blackPtsMat(size = 0.03): THREE.PointsMaterial {
  return new THREE.PointsMaterial({
    color: 0x000000,
    size,
    sizeAttenuation: true,
  });
}

function grayPtsMat(size = 0.03): THREE.PointsMaterial {
  return new THREE.PointsMaterial({
    color: 0x000000,
    size,
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.4,
  });
}

function seedCloud(n: number, r = 1.1, seed = 1): number[] {
  let s = seed;
  const rand = (): number => {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280;
  };
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    const u = rand();
    const v = rand();
    const th = u * TAU;
    const ph = Math.acos(2 * v - 1);
    const rr = r * Math.pow(rand(), 0.6);
    out.push(
      rr * Math.sin(ph) * Math.cos(th),
      rr * Math.sin(ph) * Math.sin(th),
      rr * Math.cos(ph),
    );
  }
  return out;
}

function frustum(mat: THREE.LineBasicMaterial): THREE.LineSegments {
  const s = 0.35;
  const d = 0.5;
  const p = [
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(-s, -s, -d),
    new THREE.Vector3(s, -s, -d),
    new THREE.Vector3(s, s, -d),
    new THREE.Vector3(-s, s, -d),
  ];
  const g = new THREE.BufferGeometry().setFromPoints([
    p[0], p[1], p[0], p[2], p[0], p[3], p[0], p[4],
    p[1], p[2], p[2], p[3], p[3], p[4], p[4], p[1],
  ]);
  return new THREE.LineSegments(g, mat);
}

// Placeholder; real factories added via split edits in subsequent commits.
export const FACTORIES: Record<TileId, () => TileInstance> = {
  lingbot: lingbotTile,
  slam: slamTile,
  gsplat: gsplatTile,
  mast3r: mast3rTile,
  droid: droidTile,
  dpvo: dpvoTile,
  monogs: monogsTile,
  stage_ingest: stageIngestTile,
  stage_preproc: stagePreprocTile,
  stage_inference: stageInferenceTile,
  stage_meshing: stageMeshingTile,
  stage_export: stageExportTile,
  axis_gizmo: axisGizmoTile,
};

// ═══════════ MODE TILES ═══════════

function lingbotTile(): TileInstance {
  const ctx = makeScene({ z: 2.6 });
  const cloudMat = blackPtsMat(0.02);
  const ghostMat = grayPtsMat(0.018);
  const cloud = pts(seedCloud(420, 0.95, 7), cloudMat);
  const ghost = pts(seedCloud(300, 1.1, 11), ghostMat);
  ctx.scene.add(cloud, ghost);
  ctx.disposables.push(
    cloud.geometry,
    ghost.geometry,
    cloudMat,
    ghostMat,
  );
  return finalize(ctx, (t) => {
    cloud.rotation.y = t * 0.12;
    cloud.rotation.x = Math.sin(t * 0.08) * 0.15;
    ghost.rotation.y = -t * 0.05;
  });
}

function slamTile(): TileInstance {
  const ctx = makeScene({ z: 3.0 });
  const path = new THREE.CatmullRomCurve3(
    [
      new THREE.Vector3(-1.1, -0.3, 0.4),
      new THREE.Vector3(-0.4, 0.5, -0.3),
      new THREE.Vector3(0.6, 0.3, 0.4),
      new THREE.Vector3(1.0, -0.4, -0.2),
      new THREE.Vector3(0.2, -0.6, 0.5),
      new THREE.Vector3(-0.7, -0.2, 0.6),
    ],
    true,
    "catmullrom",
    0.5,
  );
  const polyMat = blackLineMat();
  const poly = line(path.getSpacedPoints(140), polyMat);
  const kfMat = new THREE.PointsMaterial({ color: 0x000000, size: 0.06 });
  const kfPositions = path
    .getSpacedPoints(18)
    .flatMap((v) => [v.x, v.y, v.z]);
  const kf = pts(kfPositions, kfMat);
  const lmMat = grayPtsMat(0.02);
  const lm = pts(seedCloud(90, 1.3, 23), lmMat);
  const fruMat = blackLineMat();
  const fru = frustum(fruMat);
  ctx.scene.add(poly, kf, lm, fru);
  ctx.disposables.push(
    poly.geometry,
    kf.geometry,
    lm.geometry,
    fru.geometry,
    polyMat,
    kfMat,
    lmMat,
    fruMat,
  );
  return finalize(ctx, (t) => {
    const u = (t * 0.08) % 1;
    const p = path.getPointAt(u);
    const p2 = path.getPointAt((u + 0.01) % 1);
    fru.position.copy(p);
    fru.lookAt(p2);
    poly.rotation.y = t * 0.03;
    kf.rotation.y = t * 0.03;
    lm.rotation.y = t * 0.03;
  });
}

function gsplatTile(): TileInstance {
  const ctx = makeScene({ z: 2.8 });
  const group = new THREE.Group();
  ctx.scene.add(group);
  const geo = new THREE.SphereGeometry(0.08, 8, 6);
  const mat = new THREE.MeshBasicMaterial({
    color: 0x000000,
    transparent: true,
    opacity: 0.22,
    wireframe: true,
  });
  const N = 36;
  let s = 9;
  const rnd = (): number => {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280;
  };
  type Splat = THREE.Mesh & {
    userData: { phase: number; base: THREE.Vector3 };
  };
  const splats: Splat[] = [];
  for (let i = 0; i < N; i++) {
    const m = new THREE.Mesh(geo, mat) as unknown as Splat;
    m.position.set(
      (rnd() - 0.5) * 2.0,
      (rnd() - 0.5) * 1.8,
      (rnd() - 0.5) * 1.6,
    );
    m.userData.phase = rnd() * TAU;
    m.userData.base = new THREE.Vector3(
      0.4 + rnd() * 1.6,
      0.4 + rnd() * 1.0,
      0.3 + rnd() * 1.2,
    );
    m.scale.copy(m.userData.base);
    group.add(m);
    splats.push(m);
  }
  ctx.disposables.push(geo, mat);
  return finalize(ctx, (t) => {
    group.rotation.y = t * 0.18;
    group.rotation.x = Math.sin(t * 0.15) * 0.2;
    for (const m of splats) {
      const k = 0.85 + 0.3 * Math.sin(t * 1.4 + m.userData.phase);
      m.scale.copy(m.userData.base).multiplyScalar(k);
    }
  });
}

// Placeholders until the next edits. These let the file type-check
// while the remaining 10 factories are appended in follow-up edits.
// ═══════════ SLAM BACKEND TILES ═══════════

function mast3rTile(): TileInstance {
  // Two camera frustums triangulating a shared dense point cloud.
  // Correspondence lines flicker between the left frustum and points
  // in the cloud to evoke dense matching.
  const ctx = makeScene({ z: 3.0 });
  const f1Mat = blackLineMat();
  const f2Mat = blackLineMat();
  const f1 = frustum(f1Mat);
  const f2 = frustum(f2Mat);
  f1.position.set(-0.9, 0, 0.4);
  f1.lookAt(0.2, 0, -0.3);
  f2.position.set(0.9, 0, 0.4);
  f2.lookAt(-0.2, 0, -0.3);
  const densePositions = seedCloud(260, 0.55, 3).map((v, i) =>
    i % 3 === 2 ? v - 0.2 : v,
  );
  const denseMat = blackPtsMat(0.018);
  const dense = pts(densePositions, denseMat);
  const corrMat = new THREE.LineBasicMaterial({
    color: 0x000000,
    transparent: true,
    opacity: 0.2,
  });
  let corr = new THREE.LineSegments(new THREE.BufferGeometry(), corrMat);
  ctx.scene.add(f1, f2, dense, corr);
  ctx.disposables.push(
    f1.geometry,
    f2.geometry,
    dense.geometry,
    f1Mat,
    f2Mat,
    denseMat,
    corrMat,
  );
  return finalize(ctx, (t) => {
    dense.rotation.y = t * 0.15;
    const n = 8;
    const arr = new Float32Array(n * 6);
    for (let i = 0; i < n; i++) {
      const ph = t * 2 + i * 0.7;
      const x = Math.cos(ph) * 0.4;
      const y = Math.sin(ph * 1.3) * 0.3;
      const z = -0.2 + Math.sin(ph * 0.7) * 0.2;
      arr[i * 6 + 0] = -0.9;
      arr[i * 6 + 1] = 0;
      arr[i * 6 + 2] = 0.4;
      arr[i * 6 + 3] = x;
      arr[i * 6 + 4] = y;
      arr[i * 6 + 5] = z;
    }
    corr.geometry.dispose();
    ctx.scene.remove(corr);
    corr = new THREE.LineSegments(new THREE.BufferGeometry(), corrMat);
    corr.geometry.setAttribute("position", new THREE.BufferAttribute(arr, 3));
    ctx.scene.add(corr);
  });
}
function droidTile(): TileInstance {
  // 8×8 dense optical-flow field; arrows sweep in a sinusoidal pattern
  // to evoke DROID-SLAM's dense flow estimator.
  const ctx = makeScene({ z: 2.2, fov: 50 });
  const group = new THREE.Group();
  ctx.scene.add(group);
  const G = 8;
  const lineMat = blackLineMat();
  const arrows: Array<{
    ln: THREE.Line;
    gx: number;
    gy: number;
    geom: THREE.BufferGeometry;
  }> = [];
  for (let y = 0; y < G; y++) {
    for (let x = 0; x < G; x++) {
      const gx = (x / (G - 1) - 0.5) * 1.8;
      const gy = (y / (G - 1) - 0.5) * 1.4;
      const g = new THREE.BufferGeometry();
      g.setAttribute(
        "position",
        new THREE.Float32BufferAttribute([gx, gy, 0, gx, gy, 0], 3),
      );
      const ln = new THREE.Line(g, lineMat);
      group.add(ln);
      arrows.push({ ln, gx, gy, geom: g });
    }
  }
  ctx.disposables.push(lineMat, ...arrows.map((a) => a.geom));
  return finalize(ctx, (t) => {
    for (const a of arrows) {
      const ang = Math.sin(t * 0.8 + a.gx * 2 + a.gy * 1.3) * 1.2;
      const mag = 0.15 + 0.08 * Math.sin(t * 1.5 + a.gx + a.gy);
      const dx = Math.cos(ang) * mag;
      const dy = Math.sin(ang) * mag;
      const attr = a.ln.geometry.getAttribute(
        "position",
      ) as THREE.BufferAttribute;
      const pos = attr.array as Float32Array;
      pos[3] = a.gx + dx;
      pos[4] = a.gy + dy;
      pos[5] = 0;
      attr.needsUpdate = true;
    }
  });
}
function dpvoTile(): TileInstance {
  // Sparse patch squares drifting over a thin point field — DPVO's
  // patch-based VO idea.
  const ctx = makeScene({ z: 2.6 });
  const cloudMat = grayPtsMat(0.018);
  const cloud = pts(seedCloud(80, 1.0, 17), cloudMat);
  ctx.scene.add(cloud);
  const pMat = new THREE.LineBasicMaterial({ color: 0x000000 });
  const patches: Array<THREE.LineSegments & { userData: { phase: number } }> =
    [];
  for (let i = 0; i < 10; i++) {
    const s = 0.08;
    const geom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-s, -s, 0),
      new THREE.Vector3(s, -s, 0),
      new THREE.Vector3(s, -s, 0),
      new THREE.Vector3(s, s, 0),
      new THREE.Vector3(s, s, 0),
      new THREE.Vector3(-s, s, 0),
      new THREE.Vector3(-s, s, 0),
      new THREE.Vector3(-s, -s, 0),
    ]);
    const m = new THREE.LineSegments(geom, pMat) as unknown as
      THREE.LineSegments & { userData: { phase: number } };
    m.userData = { phase: i * 0.6 };
    patches.push(m);
    ctx.scene.add(m);
    ctx.disposables.push(geom);
  }
  ctx.disposables.push(cloud.geometry, cloudMat, pMat);
  return finalize(ctx, (t) => {
    cloud.rotation.y = t * 0.1;
    for (let i = 0; i < patches.length; i++) {
      const p = patches[i];
      const ph = t * 0.6 + p.userData.phase;
      p.position.x = Math.cos(ph) * (0.7 + (i % 3) * 0.15);
      p.position.y = Math.sin(ph * 1.3) * 0.5;
      p.position.z = Math.sin(ph * 0.7) * 0.3;
      p.rotation.z = ph * 0.2;
    }
  });
}
function monogsTile(): TileInstance {
  // Splats blooming outward from a single frustum — the "direct splat
  // SLAM" shape. Gaussians slide out from the camera position and scale
  // up as they land.
  const ctx = makeScene({ z: 2.8 });
  const fruMat = blackLineMat();
  const fru = frustum(fruMat);
  fru.position.set(-1.0, 0, 0.5);
  fru.lookAt(0.3, 0, -0.3);
  ctx.scene.add(fru);
  const group = new THREE.Group();
  ctx.scene.add(group);
  const geo = new THREE.SphereGeometry(0.07, 8, 6);
  const mat = new THREE.MeshBasicMaterial({
    color: 0x000000,
    wireframe: true,
    transparent: true,
    opacity: 0.3,
  });
  const N = 22;
  let s = 13;
  const rnd = (): number => {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280;
  };
  type Splat = THREE.Mesh & {
    userData: { tx: number; ty: number; tz: number; delay: number };
  };
  const splats: Splat[] = [];
  for (let i = 0; i < N; i++) {
    const tx = 0.2 + (rnd() - 0.5) * 0.9;
    const ty = (rnd() - 0.5) * 0.9;
    const tz = -0.3 + (rnd() - 0.5) * 0.6;
    const m = new THREE.Mesh(geo, mat) as unknown as Splat;
    m.userData = { tx, ty, tz, delay: rnd() };
    group.add(m);
    splats.push(m);
  }
  ctx.disposables.push(fru.geometry, fruMat, geo, mat);
  return finalize(ctx, (t) => {
    for (const m of splats) {
      const k = (Math.sin(t * 0.5 + m.userData.delay * 2) + 1) / 2;
      m.position.set(
        -1.0 + (m.userData.tx + 1.0) * k,
        0 + m.userData.ty * k,
        0.5 + (m.userData.tz - 0.5) * k,
      );
      m.scale.setScalar(0.6 + 0.5 * k);
    }
    group.rotation.y = t * 0.08;
  });
}
// ═══════════ STAGE GLYPHS ═══════════
// Rendered inline at ~14px on the active row of the stage list. The
// animation is part of the "this stage is live" signal.

function stageIngestTile(): TileInstance {
  const ctx = makeScene({ z: 2.4, fov: 50 });
  const geo = new THREE.TorusGeometry(0.6, 0.08, 6, 20);
  const mat = new THREE.MeshBasicMaterial({ color: 0x000000, wireframe: true });
  const ring = new THREE.Mesh(geo, mat);
  ctx.scene.add(ring);
  ctx.disposables.push(geo, mat);
  return finalize(ctx, (t) => {
    ring.rotation.z = t * 2.0;
    ring.rotation.x = 0.4;
  });
}

function stagePreprocTile(): TileInstance {
  const ctx = makeScene({ z: 2.0, fov: 50 });
  const grp = new THREE.Group();
  ctx.scene.add(grp);
  const gmat = new THREE.LineBasicMaterial({
    color: 0x000000,
    transparent: true,
    opacity: 0.3,
  });
  const G = 6;
  for (let i = 0; i <= G; i++) {
    const u = i / G - 0.5;
    const h = line(
      [new THREE.Vector3(-0.6, u * 1.2, 0), new THREE.Vector3(0.6, u * 1.2, 0)],
      gmat,
    );
    const v = line(
      [new THREE.Vector3(u * 1.2, -0.6, 0), new THREE.Vector3(u * 1.2, 0.6, 0)],
      gmat,
    );
    grp.add(h, v);
    ctx.disposables.push(h.geometry, v.geometry);
  }
  const scanMat = new THREE.LineBasicMaterial({ color: 0x000000 });
  const scan = line(
    [new THREE.Vector3(-0.6, 0, 0.01), new THREE.Vector3(0.6, 0, 0.01)],
    scanMat,
  );
  ctx.scene.add(scan);
  ctx.disposables.push(gmat, scanMat, scan.geometry);
  return finalize(ctx, (t) => {
    scan.position.y = Math.sin(t * 2.2) * 0.5;
    grp.rotation.y = Math.sin(t * 0.4) * 0.2;
  });
}

function stageInferenceTile(): TileInstance {
  const ctx = makeScene({ z: 2.4 });
  const cloudMat = blackPtsMat(0.03);
  const cloud = pts(seedCloud(40, 0.9, 5), cloudMat);
  const fruMat = blackLineMat();
  const fru = frustum(fruMat);
  ctx.scene.add(cloud, fru);
  ctx.disposables.push(cloud.geometry, fru.geometry, cloudMat, fruMat);
  return finalize(ctx, (t) => {
    const u = (t * 0.3) % 1;
    fru.position.set(
      Math.cos(u * TAU) * 0.8,
      Math.sin(u * TAU * 1.2) * 0.4,
      Math.sin(u * TAU) * 0.8,
    );
    fru.lookAt(0, 0, 0);
    cloud.rotation.y = t * 0.15;
  });
}

function stageMeshingTile(): TileInstance {
  const ctx = makeScene({ z: 2.4 });
  const geo = new THREE.IcosahedronGeometry(0.7, 1);
  const mat = new THREE.MeshBasicMaterial({ color: 0x000000, wireframe: true });
  const m = new THREE.Mesh(geo, mat);
  ctx.scene.add(m);
  ctx.disposables.push(geo, mat);
  return finalize(ctx, (t) => {
    m.rotation.y = t * 0.5;
    m.rotation.x = t * 0.25;
    m.scale.setScalar(0.85 + 0.15 * Math.sin(t * 1.5));
  });
}

function stageExportTile(): TileInstance {
  const ctx = makeScene({ z: 2.4 });
  const geo = new THREE.BoxGeometry(0.9, 0.9, 0.9);
  const mat = new THREE.MeshBasicMaterial({ color: 0x000000, wireframe: true });
  const m = new THREE.Mesh(geo, mat);
  ctx.scene.add(m);
  ctx.disposables.push(geo, mat);
  return finalize(ctx, (t) => {
    m.rotation.y = t * 0.6;
    m.rotation.x = 0.3 + Math.sin(t) * 0.2;
    m.scale.setScalar(1.0 - 0.15 * ((Math.sin(t * 1.2) + 1) / 2));
  });
}

// Viewer empty-state — an axis gizmo + rotating gray cloud shown before
// the first artifact has been rendered.
function axisGizmoTile(): TileInstance {
  const ctx = makeScene({ z: 3.0 });
  const grp = new THREE.Group();
  ctx.scene.add(grp);
  const mat = new THREE.LineBasicMaterial({ color: 0x000000 });
  const x = line([new THREE.Vector3(0, 0, 0), new THREE.Vector3(1, 0, 0)], mat);
  const y = line([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 1, 0)], mat);
  const z = line([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, 1)], mat);
  grp.add(x, y, z);
  const cloudMat = grayPtsMat(0.016);
  const cloud = pts(seedCloud(250, 1.2, 31), cloudMat);
  ctx.scene.add(cloud);
  ctx.disposables.push(
    x.geometry,
    y.geometry,
    z.geometry,
    cloud.geometry,
    mat,
    cloudMat,
  );
  return finalize(ctx, (t) => {
    grp.rotation.y = t * 0.3;
    grp.rotation.x = Math.sin(t * 0.4) * 0.2;
    cloud.rotation.y = t * 0.08;
  });
}
