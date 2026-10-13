// Measurement overlay (inside the Canvas): picked points, in-progress segments and saved measurements.
import { useEffect, useState } from "react";
import { Html, Line } from "@react-three/drei";
import { api } from "../lib/api";
import { useStore } from "../lib/store";
import type { Vec3 } from "../lib/types";

export default function MeasureOverlay({ size }: { size: number }) {
  const caseId = useStore((s) => s.caseId)!;
  const picked = useStore((s) => s.picked);
  const version = useStore((s) => s.measurementsVersion);
  const scene = useStore((s) => s.scene);
  const [saved, setSaved] = useState<any[]>([]);
  useEffect(() => {
    api.measurements(caseId).then((l) => setSaved(l.filter((m) => m.active !== false && m.points?.length))).catch(() => setSaved([]));
  }, [caseId, version, scene]);
  const r = size * 0.05;
  return (
    <group>
      {picked.map((p, i) => (
        <mesh key={i} position={p.p}>
          <sphereGeometry args={[r, 12, 10]} />
          <meshBasicMaterial color="#ffb347" depthTest={false} />
        </mesh>
      ))}
      {picked.length >= 2 && <Line points={picked.map((p) => p.p)} color="#ffb347" lineWidth={2} depthTest={false} />}
      {saved.map((m) => {
        const pts = m.points as Vec3[];
        const mid: Vec3 = pts.length === 3 ? pts[1] : [
          pts.reduce((s, p) => s + p[0], 0) / pts.length,
          pts.reduce((s, p) => s + p[1], 0) / pts.length,
          pts.reduce((s, p) => s + p[2], 0) / pts.length,
        ];
        const line: Vec3[] = m.kind === "height_above_ground" && pts.length === 1 ? [[pts[0][0], 0, pts[0][2]], pts[0]] : pts;
        return (
          <group key={m.id}>
            {line.length >= 2 && <Line points={line} color="#ffd166" lineWidth={2} dashed={m.kind === "height_above_ground"} dashSize={0.1} gapSize={0.06} />}
            {pts.map((p, i) => (
              <mesh key={i} position={p}>
                <sphereGeometry args={[r * 0.7, 10, 8]} />
                <meshBasicMaterial color="#ffd166" />
              </mesh>
            ))}
            <Html position={[mid[0], mid[1] + r * 3, mid[2]]} center zIndexRange={[5, 0]} style={{ pointerEvents: "none" }}>
              <div style={{ fontSize: 11, whiteSpace: "nowrap", background: "rgba(36,28,8,.9)", border: "1px solid #ffd166", padding: "1px 6px", borderRadius: 4, color: "#fff3cf" }}>
                <b>{m.id}</b> {m.display}
              </div>
            </Html>
          </group>
        );
      })}
    </group>
  );
}
