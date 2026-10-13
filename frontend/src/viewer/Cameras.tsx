// Camera frustums (static cameras) and camera paths + current pose (moving cameras).
import { useMemo } from "react";
import { Html, Line } from "@react-three/drei";
import * as THREE from "three";
import { useStore } from "../lib/store";
import type { CameraMeta, Vec3 } from "../lib/types";
import { cameraPoseAt, mat4 } from "./util";

const SEL = "#4cc2ff";

function frustumLines(c: CameraMeta, rows: number[][], depth: number): Vec3[][] {
  const { fx, fy, cx, cy, width, height } = c.intrinsics;
  const M = mat4(rows);
  const corner = (u: number, v: number) => new THREE.Vector3(((u - cx) / fx) * depth, ((v - cy) / fy) * depth, depth).applyMatrix4(M);
  const C = new THREE.Vector3().setFromMatrixPosition(M);
  const k = [corner(0, 0), corner(width, 0), corner(width, height), corner(0, height)];
  const up = corner(width / 2, -0.25 * height); // "up" tick shows image top
  const v = (p: THREE.Vector3) => p.toArray() as Vec3;
  return [
    [v(C), v(k[0])], [v(C), v(k[1])], [v(C), v(k[2])], [v(C), v(k[3])],
    [v(k[0]), v(k[1]), v(k[2]), v(k[3]), v(k[0])],
    [v(k[0]), v(up), v(k[1])],
  ];
}

export default function Cameras({ size, onSelect }: { size: number; onSelect: (id: string) => void }) {
  const scene = useStore((s) => s.scene)!;
  const show = useStore((s) => s.show.cameras);
  if (!show) return null;
  return <group>{scene.cameras.map((c) => <Cam key={c.id} c={c} size={size} onSelect={onSelect} />)}</group>;
}

function Cam({ c, size, onSelect }: { c: CameraMeta; size: number; onSelect: (id: string) => void }) {
  const t = useStore((s) => s.time);
  const sel = useStore((s) => s.selection);
  const labels = useStore((s) => s.show.labels);
  const selected = sel?.kind === "camera" && sel.id === c.id;
  const posed = !!c.poses?.t.length; // moving (SfM) or rotating (pan/tilt) cameras
  const pose = posed ? cameraPoseAt(c, t) : { rows: c.world_from_cam, frame: null, source: null, exact: true };
  const lines = useMemo(() => frustumLines(c, pose.rows, size), [c, pose.rows, size]);
  const path = useMemo(() => (c.poses && c.kind === "moving" ? c.poses.world_from_cam.map((m) => [m[0][3], m[1][3], m[2][3]] as Vec3) : null), [c]);
  const keyframes = useMemo(() => (c.poses ? c.poses.world_from_cam.filter((_, i) => c.poses!.source[i] === "keyframe").map((m) => [m[0][3], m[1][3], m[2][3]] as Vec3) : []), [c]);
  const color = selected ? SEL : c.color;
  const C: Vec3 = [pose.rows[0][3], pose.rows[1][3], pose.rows[2][3]];
  return (
    <group>
      {lines.map((pts, i) => <Line key={i} points={pts} color={color} lineWidth={selected ? 2.5 : 1.5} transparent opacity={pose.exact ? 1 : 0.5} />)}
      <mesh position={C} onClick={(e) => { e.stopPropagation(); onSelect(c.id); }}>
        <sphereGeometry args={[size * 0.12, 12, 10]} />
        <meshBasicMaterial color={color} />
      </mesh>
      {path && path.length > 1 && <Line points={path} color={c.color} lineWidth={1} transparent opacity={0.6} />}
      {keyframes.map((p, i) => (
        <mesh key={i} position={p}>
          <sphereGeometry args={[size * 0.04, 6, 5]} />
          <meshBasicMaterial color={c.color} />
        </mesh>
      ))}
      {labels && (
        <Html position={[C[0], C[1] + size * 0.5, C[2]]} center zIndexRange={[4, 0]} style={{ pointerEvents: "none" }}>
          <div style={{ fontSize: 10.5, whiteSpace: "nowrap", background: "rgba(10,14,19,.75)", padding: "1px 5px", borderRadius: 4, color: "#e6ebf2", border: `1px solid ${c.color}` }}>
            {c.id}{posed && pose.frame != null ? ` · f${pose.frame}${pose.source && pose.source !== "keyframe" ? ` (${pose.source})` : ""}` : ""}
          </div>
        </Html>
      )}
    </group>
  );
}
