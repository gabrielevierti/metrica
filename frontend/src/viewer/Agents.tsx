// Tracked people/vehicles: trajectories (solid where observed, dashed across observation gaps),
// time-synchronised body proxies, 1-sigma position uncertainty rings and labels.
import { useMemo, type ReactNode } from "react";
import * as THREE from "three";
import { Html, Line } from "@react-three/drei";
import { agentPositionAt, useStore } from "../lib/store";
import type { Agent, Vec3 } from "../lib/types";

const SEL = "#4cc2ff";

function splitRuns(a: Agent): { observed: Vec3[][]; gaps: [Vec3, Vec3][] } {
  const s = a.samples;
  if (s.length < 2) return { observed: s.length ? [[s[0].p, s[0].p]] : [], gaps: [] };
  const dts = s.slice(1).map((x, i) => x.t - s[i].t).sort((p, q) => p - q);
  const med = dts[Math.floor(dts.length / 2)] || 0.1;
  const thr = Math.max(0.6, 4 * med);
  const runs: { t0: number; t1: number }[] = [];
  const gaps: [Vec3, Vec3][] = [];
  let start = s[0].t;
  for (let i = 1; i < s.length; i++) {
    if (s[i].t - s[i - 1].t > thr) {
      runs.push({ t0: start, t1: s[i - 1].t });
      gaps.push([s[i - 1].p, s[i].p]);
      start = s[i].t;
    }
  }
  runs.push({ t0: start, t1: s[s.length - 1].t });
  const src = a.smoothed && a.smoothed.length > 1 ? a.smoothed : s;
  const observed = runs.map((r) => src.filter((x) => x.t >= r.t0 - 1e-6 && x.t <= r.t1 + 1e-6).map((x) => [x.p[0], x.p[1] + 0.03, x.p[2]] as Vec3)).filter((r) => r.length >= 2);
  return { observed, gaps: gaps.map(([p, q]) => [[p[0], p[1] + 0.03, p[2]], [q[0], q[1] + 0.03, q[2]]]) };
}

export default function Agents({ onSelect }: { onSelect: (id: string) => void }) {
  const scene = useStore((s) => s.scene)!;
  const show = useStore((s) => s.show);
  const cat = useStore((s) => s.categoryVisible);
  const visible = useMemo(() => scene.agents.filter((a) => cat[a.category] !== false), [scene, cat]);
  if (!show.agents && !show.trajectories) return null;
  return (
    <group>
      {show.trajectories && visible.map((a) => <Trajectory key={a.id} a={a} />)}
      {show.agents && <Markers agents={visible} onSelect={onSelect} />}
    </group>
  );
}

function Trajectory({ a }: { a: Agent }) {
  const scene = useStore((s) => s.scene)!;
  const sel = useStore((s) => s.selection);
  const selected = sel?.kind === "agent" && sel.id === a.id;
  const runs = useMemo(() => splitRuns(a), [a]);
  const color = scene.categories[a.category]?.color ?? "#e4572e";
  return (
    <group>
      {runs.observed.map((pts, i) => <Line key={i} points={pts} color={selected ? SEL : color} lineWidth={selected ? 3 : 1.2} transparent opacity={selected ? 1 : 0.75} />)}
      {runs.gaps.map((pts, i) => <Line key={`g${i}`} points={pts} color={selected ? SEL : color} lineWidth={1} dashed dashSize={0.25} gapSize={0.25} transparent opacity={0.6} />)}
    </group>
  );
}

function heading(a: Agent, t: number): number | null {
  const src = a.smoothed && a.smoothed.length > 1 ? a.smoothed : a.samples;
  const p0 = agentPositionAt({ samples: a.samples, smoothed: src as any }, t - 0.4, 1e9);
  const p1 = agentPositionAt({ samples: a.samples, smoothed: src as any }, t + 0.4, 1e9);
  if (!p0 || !p1) return null;
  const dx = p1.p[0] - p0.p[0], dz = p1.p[2] - p0.p[2];
  if (Math.hypot(dx, dz) < 0.15) return null;
  return Math.atan2(dz, dx);
}

function Markers({ agents, onSelect }: { agents: Agent[]; onSelect: (id: string) => void }) {
  const t = useStore((s) => s.time);
  const show = useStore((s) => s.show);
  const scene = useStore((s) => s.scene)!;
  const sel = useStore((s) => s.selection);
  const items = agents.map((a) => ({ a, at: agentPositionAt(a, t) })).filter((x) => x.at);
  return (
    <group>
      {items.map(({ a, at }) => {
        const selected = sel?.kind === "agent" && sel.id === a.id;
        const color = selected ? SEL : scene.categories[a.category]?.color ?? "#e4572e";
        const rep = a.representation ?? { type: "marker" };
        const op = at!.interpolated ? 0.35 : 0.9;
        const click = (e: any) => { e.stopPropagation(); onSelect(a.id); };
        let body: ReactNode;
        let top = 1;
        if (rep.type === "mannequin") {
          const h = rep.height || 1.7;
          top = h;
          body = (
            <group onClick={click}>
              <mesh position={[0, h * 0.42, 0]}>
                <capsuleGeometry args={[h * 0.13, h * 0.56, 4, 10]} />
                <meshLambertMaterial color={color} transparent opacity={op} />
              </mesh>
              <mesh position={[0, h * 0.91, 0]}>
                <sphereGeometry args={[h * 0.075, 12, 10]} />
                <meshLambertMaterial color={color} transparent opacity={op} />
              </mesh>
            </group>
          );
        } else if (rep.type === "vehicle") {
          const yaw = heading(a, t) ?? 0;
          top = rep.height;
          body = (
            <group rotation={[0, -yaw, 0]} onClick={click}>
              <mesh position={[0, rep.height * 0.3, 0]}>
                <boxGeometry args={[rep.length, rep.height * 0.6, rep.width]} />
                <meshLambertMaterial color={color} transparent opacity={op * 0.8} />
              </mesh>
              <mesh position={[-rep.length * 0.05, rep.height * 0.8, 0]}>
                <boxGeometry args={[rep.length * 0.5, rep.height * 0.4, rep.width * 0.9]} />
                <meshLambertMaterial color={color} transparent opacity={op * 0.6} />
              </mesh>
            </group>
          );
        } else {
          body = (
            <mesh position={[0, 0.5, 0]} onClick={click}>
              <sphereGeometry args={[0.3, 12, 10]} />
              <meshLambertMaterial color={color} transparent opacity={op} />
            </mesh>
          );
        }
        return (
          <group key={a.id} position={at!.p}>
            {body}
            {show.uncertainty && at!.sigma != null && at!.sigma > 0 && (
              <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.02, 0]}>
                <ringGeometry args={[Math.max(0.01, at!.sigma * 0.92), at!.sigma, 40]} />
                <meshBasicMaterial color={color} transparent opacity={0.8} side={THREE.DoubleSide} depthWrite={false} />
              </mesh>
            )}
            {show.labels && (
              <Html position={[0, top + 0.35, 0]} center zIndexRange={[4, 0]} style={{ pointerEvents: "none" }}>
                <div style={{ fontSize: 10.5, whiteSpace: "nowrap", background: "rgba(10,14,19,.75)", padding: "1px 5px", borderRadius: 4, color: "#e6ebf2", border: `1px solid ${color}` }}>
                  {a.id}{at!.interpolated ? " (between observations)" : ""}
                </div>
              </Html>
            )}
          </group>
        );
      })}
    </group>
  );
}
