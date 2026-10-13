// Geometry helpers shared by the viewer components.
import * as THREE from "three";
import type { Agent, CameraMeta, Entity, Scene, Vec3 } from "../lib/types";
import { EVIDENCE_COLORS } from "../lib/types";
import { agentPositionAt } from "../lib/store";

export function mat4(rows: number[][]): THREE.Matrix4 {
  const m = new THREE.Matrix4();
  m.set(
    rows[0][0], rows[0][1], rows[0][2], rows[0][3],
    rows[1][0], rows[1][1], rows[1][2], rows[1][3],
    rows[2][0], rows[2][1], rows[2][2], rows[2][3],
    0, 0, 0, 1,
  );
  return m;
}

// METRICA cameras use the computer-vision convention (x right, y down, z forward).
// three.js cameras look along -z with y up: flip the y and z axes.
const CV_TO_GL = new THREE.Matrix4().makeScale(1, -1, -1);

export function glCameraMatrix(rows: number[][]): THREE.Matrix4 {
  return mat4(rows).multiply(CV_TO_GL);
}

export function cameraCenter(rows: number[][]): THREE.Vector3 {
  return new THREE.Vector3(rows[0][3], rows[1][3], rows[2][3]);
}

/** Pose of a camera at case time t (moving cameras: nearest registered/localized pose). */
export function cameraPoseAt(c: CameraMeta, t: number): { rows: number[][]; frame: number | null; source: string | null; exact: boolean } {
  if (!c.poses || !c.poses.t.length) return { rows: c.world_from_cam, frame: null, source: null, exact: true };
  const ts = c.poses.t;
  let lo = 0, hi = ts.length - 1;
  if (t <= ts[0]) hi = 0;
  else if (t >= ts[hi]) lo = hi;
  else {
    while (hi - lo > 1) {
      const m = (lo + hi) >> 1;
      if (ts[m] <= t) lo = m; else hi = m;
    }
  }
  const k = Math.abs(ts[lo] - t) <= Math.abs(ts[hi] - t) ? lo : hi;
  return { rows: c.poses.world_from_cam[k], frame: c.poses.frames[k], source: c.poses.source[k], exact: Math.abs(ts[k] - t) < 0.08 };
}

export interface Bounds {
  center: THREE.Vector3;
  radius: number;
  min: THREE.Vector3;
  max: THREE.Vector3;
}

export function sceneBounds(scene: Scene): Bounds {
  const box = new THREE.Box3();
  for (const l of scene.layers) {
    if (l.stats?.bbox_min && l.stats?.bbox_max) {
      box.expandByPoint(new THREE.Vector3(...(l.stats.bbox_min as Vec3)));
      box.expandByPoint(new THREE.Vector3(...(l.stats.bbox_max as Vec3)));
    }
  }
  for (const c of scene.cameras) box.expandByPoint(cameraCenter(c.world_from_cam));
  for (const e of scene.entities) {
    const b = entityBox(e);
    if (b && e.category !== "ground") box.union(b);
  }
  if (box.isEmpty()) box.set(new THREE.Vector3(-10, 0, -10), new THREE.Vector3(10, 3, 10));
  // robustness: very distant depth (sky/horizon) inflates the box; clamp to agents/entities when available
  const center = box.getCenter(new THREE.Vector3());
  const radius = Math.max(1, box.getSize(new THREE.Vector3()).length() / 2);
  return { center, radius, min: box.min.clone(), max: box.max.clone() };
}

export function entityBox(e: Entity): THREE.Box3 | null {
  const g = e.geometry;
  const b = new THREE.Box3();
  switch (g.type) {
    case "plane_polygon": {
      const o = new THREE.Vector3(...(g.origin as Vec3));
      const u = new THREE.Vector3(...(g.u_axis as Vec3));
      const v = new THREE.Vector3(...(g.v_axis as Vec3));
      for (const poly of g.polygons_uv as number[][][]) for (const [pu, pv] of poly) b.expandByPoint(o.clone().addScaledVector(u, pu).addScaledVector(v, pv));
      return b;
    }
    case "cylinder": {
      const [x, y, z] = g.base as Vec3;
      const r = Math.max(g.radius_bottom, g.radius_top);
      return b.set(new THREE.Vector3(x - r, y, z - r), new THREE.Vector3(x + r, y + g.height, z + r));
    }
    case "vehicle_box": {
      const [x, , z] = g.center as Vec3;
      const r = Math.hypot(g.length, g.width) / 2;
      return b.set(new THREE.Vector3(x - r, 0, z - r), new THREE.Vector3(x + r, g.height, z + r));
    }
    case "box": {
      const [x, y, z] = g.center as Vec3;
      const [L, H, W] = g.size as number[];
      const r = Math.hypot(L, W) / 2;
      return b.set(new THREE.Vector3(x - r, y - H / 2, z - r), new THREE.Vector3(x + r, y + H / 2, z + r));
    }
    case "bbox":
      return b.set(new THREE.Vector3(...(g.min as Vec3)), new THREE.Vector3(...(g.max as Vec3)));
    case "ground_plane": {
      for (const poly of g.observed_polygons_xz as number[][][]) for (const [x, z] of poly) b.expandByPoint(new THREE.Vector3(x, 0, z));
      return b.isEmpty() ? null : b;
    }
  }
  return null;
}

export function agentFocus(a: Agent, t: number): THREE.Vector3 {
  const p = agentPositionAt(a, t, 1e9);
  if (p) return new THREE.Vector3(...p.p);
  const s = a.samples[Math.floor(a.samples.length / 2)];
  return new THREE.Vector3(...s.p);
}

export function evidenceColor(cls: string | undefined): string {
  return EVIDENCE_COLORS[cls ?? "UNKNOWN"] ?? EVIDENCE_COLORS.UNKNOWN;
}

/** Shape in the XZ ground plane (for ShapeGeometry rotated by -90° about X). */
export function shapeXZ(poly: number[][]): THREE.Shape {
  return new THREE.Shape(poly.map(([x, z]) => new THREE.Vector2(x, -z)));
}

/** Triangulated planar polygon embedded in 3D (origin + u*U + v*V), with UVs mapped over [u0,u1]x[v0,v1]. */
export function planePolygonGeometry(
  polys: number[][][], origin: Vec3, U: Vec3, V: Vec3, N: Vec3, offset: number,
  uvRange?: { u: [number, number]; v: [number, number] },
): THREE.BufferGeometry {
  const pos: number[] = [];
  const uv: number[] = [];
  const idx: number[] = [];
  for (const poly of polys) {
    if (poly.length < 3) continue;
    const contour = poly.map(([a, b]) => new THREE.Vector2(a, b));
    const tris = THREE.ShapeUtils.triangulateShape(contour, []);
    const base = pos.length / 3;
    for (const [a, b] of poly) {
      pos.push(
        origin[0] + a * U[0] + b * V[0] + offset * N[0],
        origin[1] + a * U[1] + b * V[1] + offset * N[1],
        origin[2] + a * U[2] + b * V[2] + offset * N[2],
      );
      if (uvRange) uv.push((a - uvRange.u[0]) / Math.max(1e-9, uvRange.u[1] - uvRange.u[0]), (b - uvRange.v[0]) / Math.max(1e-9, uvRange.v[1] - uvRange.v[0]));
      else uv.push(0, 0);
    }
    for (const t of tris) idx.push(base + t[0], base + t[1], base + t[2]);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
  g.setIndex(idx);
  g.computeVertexNormals();
  return g;
}

export function planePoint(origin: Vec3, U: Vec3, V: Vec3, N: Vec3, a: number, b: number, off = 0): Vec3 {
  return [
    origin[0] + a * U[0] + b * V[0] + off * N[0],
    origin[1] + a * U[1] + b * V[1] + off * N[1],
    origin[2] + a * U[2] + b * V[2] + off * N[2],
  ];
}

/** Diagonal hatch texture used to mark inferred (not observed) surfaces. */
let hatch: THREE.CanvasTexture | null = null;
export function hatchTexture(): THREE.CanvasTexture {
  if (hatch) return hatch;
  const c = document.createElement("canvas");
  c.width = c.height = 32;
  const g = c.getContext("2d")!;
  g.fillStyle = "rgba(220,80,80,0.22)";
  g.fillRect(0, 0, 32, 32);
  g.strokeStyle = "rgba(255,110,110,0.9)";
  g.lineWidth = 3;
  for (let k = -32; k <= 64; k += 12) {
    g.beginPath();
    g.moveTo(k, 0);
    g.lineTo(k + 32, 32);
    g.stroke();
  }
  hatch = new THREE.CanvasTexture(c);
  hatch.wrapS = hatch.wrapT = THREE.RepeatWrapping;
  hatch.colorSpace = THREE.SRGBColorSpace;
  return hatch;
}

export function isEntityVisible(e: Entity, cat: Record<string, boolean>, ev: Record<string, boolean>): boolean {
  if (cat[e.category] === false) return false;
  if (e.evidence?.class && ev[e.evidence.class] === false) return false;
  return true;
}
