// Semantic 3D objects: facades, poles, vehicles, static objects, clusters and the ground.
// Every object is drawn so that its evidence class stays visible: observed parts solid,
// prior/fitted parts in their evidence colour, inferred parts hatched red.
import { useMemo } from "react";
import * as THREE from "three";
import { Edges, Html, Line } from "@react-three/drei";
import type { ThreeEvent } from "@react-three/fiber";
import { sceneFileUrl } from "../lib/api";
import { useStore } from "../lib/store";
import type { Entity, Vec3 } from "../lib/types";
import { useTexture } from "./Layers";
import { evidenceColor, hatchTexture, isEntityVisible, planePoint, planePolygonGeometry, shapeXZ } from "./util";

type PickFn = (e: ThreeEvent<MouseEvent>, ent: Entity) => void;

export default function Entities({ onPick }: { onPick: PickFn }) {
  const scene = useStore((s) => s.scene)!;
  const cat = useStore((s) => s.categoryVisible);
  const ev = useStore((s) => s.evidenceVisible);
  const show = useStore((s) => s.show);
  if (!show.entities) return null;
  return (
    <group>
      {scene.entities.filter((e) => isEntityVisible(e, cat, ev)).map((e) => <EntityNode key={e.id} e={e} onPick={onPick} />)}
    </group>
  );
}

function EntityNode({ e, onPick }: { e: Entity; onPick: PickFn }) {
  const sel = useStore((s) => s.selection);
  const selected = sel?.kind === "entity" && sel.id === e.id;
  const click = (ev: ThreeEvent<MouseEvent>) => { ev.stopPropagation(); onPick(ev, e); };
  switch (e.geometry.type) {
    case "plane_polygon": return <Facade e={e} selected={selected} onClick={click} />;
    case "cylinder": return <Pole e={e} selected={selected} onClick={click} />;
    case "vehicle_box": return <Vehicle e={e} selected={selected} onClick={click} />;
    case "box": return <BoxObj e={e} selected={selected} onClick={click} />;
    case "bbox": return <Cluster e={e} selected={selected} onClick={click} />;
    case "ground_plane": return <Ground e={e} selected={selected} />;
  }
  return null;
}

interface NodeProps { e: Entity; selected: boolean; onClick?: (ev: ThreeEvent<MouseEvent>) => void }

function Label({ e, position }: { e: Entity; position: Vec3 }) {
  const show = useStore((s) => s.show.labels);
  const sel = useStore((s) => s.selection);
  if (!show && !(sel?.kind === "entity" && sel.id === e.id)) return null;
  return (
    <Html position={position} center zIndexRange={[4, 0]} style={{ pointerEvents: "none" }}>
      <div style={{ fontSize: 10.5, whiteSpace: "nowrap", background: "rgba(10,14,19,.75)", padding: "1px 5px", borderRadius: 4, border: `1px solid ${evidenceColor(e.evidence?.class)}`, color: "#e6ebf2" }}>
        {e.display ?? e.category} <span style={{ opacity: 0.6 }}>{e.id.split("-").slice(-2).join("-")}</span>
      </div>
    </Html>
  );
}

const SEL = "#4cc2ff";

function Facade({ e, selected, onClick }: NodeProps) {
  const g = e.geometry;
  const caseId = useStore((s) => s.caseId)!;
  const mode = useStore((s) => s.colorMode);
  const show = useStore((s) => s.show);
  const wire = useStore((s) => s.wireframe);
  const texUrl = e.texture ? sceneFileUrl(caseId, e.texture.file) : null;
  const tex = useTexture(mode === "photo" ? texUrl : null);
  const N = g.normal as Vec3;
  const geom = useMemo(() => planePolygonGeometry(g.polygons_uv, g.origin, g.u_axis, g.v_axis, N, 0.01,
    e.texture ? { u: e.texture.u_range, v: e.texture.v_range } : undefined), [e]);
  const patches = useMemo(() => (e.inferred_patches ?? []).map((p) =>
    planePolygonGeometry([p.polygon_uv], g.origin, g.u_axis, g.v_axis, N, 0.02, { u: [0, 0.5], v: [0, 0.5] })), [e]);
  const outline = useMemo(() => (g.polygons_uv as number[][][]).map((poly) => [...poly, poly[0]].map(([a, b]) => planePoint(g.origin, g.u_axis, g.v_axis, N, a, b, 0.015))), [e]);
  const top = useMemo(() => {
    const all = (g.polygons_uv as number[][][]).flat();
    const umid = (Math.min(...all.map((p) => p[0])) + Math.max(...all.map((p) => p[0]))) / 2;
    return planePoint(g.origin, g.u_axis, g.v_axis, N, umid, Math.max(...all.map((p) => p[1])) + 0.4, 0.05);
  }, [e]);
  const col = mode === "evidence" ? evidenceColor(e.evidence?.class) : "#c9ada7";
  return (
    <group>
      <mesh geometry={geom} onClick={onClick} renderOrder={1}>
        {tex ? <meshBasicMaterial map={tex} transparent alphaTest={0.05} side={THREE.DoubleSide} wireframe={wire} />
          : <meshLambertMaterial color={col} transparent opacity={tex ? 1 : 0.55} side={THREE.DoubleSide} wireframe={wire} />}
      </mesh>
      {outline.map((pts, i) => <Line key={i} points={pts} color={selected ? SEL : evidenceColor(e.evidence?.class)} lineWidth={selected ? 3 : 1.5} />)}
      {show.inferred && patches.map((pg, i) => (
        <mesh key={i} geometry={pg} renderOrder={2} onClick={onClick}>
          <meshBasicMaterial map={hatchTexture()} transparent depthWrite={false} side={THREE.DoubleSide} />
        </mesh>
      ))}
      {show.openings && (e.openings ?? []).map((o, i) => (
        <Line key={i} color="#ffb347" lineWidth={1.5} dashed dashSize={0.15} gapSize={0.1}
          points={[[o.u0, o.v0], [o.u1, o.v0], [o.u1, o.v1], [o.u0, o.v1], [o.u0, o.v0]].map(([a, b]) => planePoint(g.origin, g.u_axis, g.v_axis, N, a, b, 0.03))} />
      ))}
      {show.inferred && (e.relations ?? []).filter((r) => r.type === "meets_at_corner" && r.corner_xz).map((r, i) => (
        <Line key={`c${i}`} color="#dc5050" dashed dashSize={0.2} gapSize={0.15} lineWidth={1}
          points={[[r.corner_xz[0], 0, r.corner_xz[1]], [r.corner_xz[0], Math.max(2, top[1] - 0.4), r.corner_xz[1]]]} />
      ))}
      <Label e={e} position={top} />
    </group>
  );
}

function Pole({ e, selected, onClick }: NodeProps) {
  const g = e.geometry;
  const mode = useStore((s) => s.colorMode);
  const show = useStore((s) => s.show);
  const wire = useStore((s) => s.wireframe);
  const [x, y, z] = g.base as Vec3;
  const [h0, h1] = g.observed_range as [number, number];
  const H = Math.max(1e-3, h1 - h0);
  const rAt = (h: number) => g.radius_bottom + (g.radius_top - g.radius_bottom) * (h / Math.max(g.height, 1e-6));
  const col = mode === "evidence" ? evidenceColor(e.evidence?.class) : "#f2e94e";
  const lowerBound = e.dimensions?.height?.basis === "lower bound";
  return (
    <group>
      <mesh position={[x, y + h0 + H / 2, z]} onClick={onClick}>
        <cylinderGeometry args={[Math.max(rAt(h1), 0.01), Math.max(rAt(h0), 0.01), H, 20]} />
        <meshLambertMaterial color={selected ? SEL : col} wireframe={wire} />
      </mesh>
      {show.inferred && h0 > 0.01 && (
        <mesh position={[x, y + h0 / 2, z]} onClick={onClick}>
          <cylinderGeometry args={[Math.max(rAt(h0), 0.01), Math.max(g.radius_bottom, 0.01), h0, 20, 1, true]} />
          <meshBasicMaterial map={hatchTexture()} transparent side={THREE.DoubleSide} depthWrite={false} />
        </mesh>
      )}
      {lowerBound && (
        <Line points={[[x, y + h1, z], [x, y + h1 + Math.max(0.5, 0.15 * g.height), z]]} color="#ffb347" dashed dashSize={0.1} gapSize={0.08} lineWidth={1.5} />
      )}
      <Label e={e} position={[x, y + g.height + 0.4, z]} />
    </group>
  );
}

/** Generic vehicle template (lower body + cabin) scaled to the estimated dimensions — not a scan. */
function Vehicle({ e, selected, onClick }: NodeProps) {
  const g = e.geometry;
  const mode = useStore((s) => s.colorMode);
  const scene = useStore((s) => s.scene)!;
  const wire = useStore((s) => s.wireframe);
  const [x, , z] = g.center as Vec3;
  const L = g.length, W = g.width, H = g.height;
  const col = mode === "evidence" ? evidenceColor(e.evidence?.class) : scene.categories[e.category]?.color ?? "#2e86ab";
  const tall = g.template === "truck" || g.template === "bus";
  const body = tall ? 0.78 : 0.55;
  return (
    <group position={[x, 0, z]} rotation={[0, -g.yaw_rad, 0]}>
      <mesh position={[0, (H * body) / 2 + 0.02 * H, 0]} onClick={onClick}>
        <boxGeometry args={[L, H * body, W]} />
        <meshLambertMaterial color={col} transparent opacity={0.72} wireframe={wire} />
        <Edges color={selected ? SEL : "#e6ebf2"} />
      </mesh>
      <mesh position={[tall ? L * 0.36 : -L * 0.05, H * body + (H * (1 - body)) / 2, 0]} onClick={onClick}>
        <boxGeometry args={[tall ? L * 0.24 : L * 0.52, H * (1 - body), W * 0.92]} />
        <meshLambertMaterial color={col} transparent opacity={0.5} wireframe={wire} />
        <Edges color={selected ? SEL : "#cfd6df"} />
      </mesh>
      <Html position={[0, H + 0.5, 0]} center zIndexRange={[4, 0]} style={{ pointerEvents: "none" }}>
        <LabelBox e={e} />
      </Html>
    </group>
  );
}

function LabelBox({ e }: { e: Entity }) {
  const show = useStore((s) => s.show.labels);
  if (!show) return null;
  return (
    <div style={{ fontSize: 10.5, whiteSpace: "nowrap", background: "rgba(10,14,19,.75)", padding: "1px 5px", borderRadius: 4, border: `1px solid ${evidenceColor(e.evidence?.class)}`, color: "#e6ebf2" }}>
      {e.display ?? e.category} <span style={{ opacity: 0.6 }}>{e.id.split("-").slice(-2).join("-")}</span>
    </div>
  );
}

function BoxObj({ e, selected, onClick }: NodeProps) {
  const g = e.geometry;
  const mode = useStore((s) => s.colorMode);
  const scene = useStore((s) => s.scene)!;
  const wire = useStore((s) => s.wireframe);
  const [x, y, z] = g.center as Vec3;
  const [L, H, W] = g.size as number[];
  const col = mode === "evidence" ? evidenceColor(e.evidence?.class) : scene.categories[e.category]?.color ?? "#9aa0a6";
  return (
    <group position={[x, y, z]} rotation={[0, -g.yaw_rad, 0]}>
      <mesh onClick={onClick}>
        <boxGeometry args={[L, H, W]} />
        <meshLambertMaterial color={col} transparent opacity={0.6} wireframe={wire} />
        <Edges color={selected ? SEL : "#e6ebf2"} />
      </mesh>
      <Html position={[0, H / 2 + 0.4, 0]} center zIndexRange={[4, 0]} style={{ pointerEvents: "none" }}><LabelBox e={e} /></Html>
    </group>
  );
}

function Cluster({ e, selected, onClick }: NodeProps) {
  const g = e.geometry;
  const mn = g.min as Vec3, mx = g.max as Vec3;
  const c: Vec3 = [(mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2];
  const s: Vec3 = [mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]];
  const col = e.category === "vegetation" ? "#3a7d44" : "#9aa0a6";
  return (
    <group position={c}>
      <mesh onClick={onClick}>
        <boxGeometry args={s} />
        <meshBasicMaterial color={col} transparent opacity={selected ? 0.25 : 0.06} depthWrite={false} />
        <Edges color={selected ? SEL : col} />
      </mesh>
      <Html position={[0, s[1] / 2 + 0.3, 0]} center zIndexRange={[4, 0]} style={{ pointerEvents: "none" }}><LabelBox e={e} /></Html>
    </group>
  );
}

const REGION_COLORS: Record<string, string> = { road: "#9aa3b5", sidewalk: "#d9c7ae", terrain: "#9dbf9e" };

function Ground({ e, selected }: NodeProps) {
  const g = e.geometry;
  const show = useStore((s) => s.show);
  const inferred = useMemo(() => (g.inferred_polygons_xz as number[][][]).filter((p) => p.length >= 3).map((p) => new THREE.ShapeGeometry(shapeXZ(p))), [e]);
  const loops = (polys: number[][][], y: number) => polys.filter((p) => p.length >= 2).map((p) => [...p, p[0]].map(([x, z]) => [x, y, z] as Vec3));
  const observed = useMemo(() => loops(g.observed_polygons_xz, 0.01), [e]);
  const regions = useMemo(() => Object.entries(e.regions ?? {}).map(([k, r]: [string, any]) => ({ k, lines: loops(r.polygons_xz, 0.015) })), [e]);
  return (
    <group>
      {observed.map((pts, i) => <Line key={i} points={pts} color={selected ? SEL : "#5a82dc"} lineWidth={1} transparent opacity={0.8} />)}
      {regions.map((r) => r.lines.map((pts, i) => <Line key={r.k + i} points={pts} color={REGION_COLORS[r.k] ?? "#888"} lineWidth={1} dashed dashSize={0.3} gapSize={0.2} />))}
      {show.inferred && inferred.map((geo, i) => (
        <mesh key={i} geometry={geo} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.008, 0]} renderOrder={2}>
          <meshBasicMaterial map={hatchTexture()} transparent depthWrite={false} side={THREE.DoubleSide} />
        </mesh>
      ))}
    </group>
  );
}
