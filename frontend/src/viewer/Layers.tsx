import { useEffect, useMemo, useState } from "react";
import * as THREE from "three";
import type { ThreeEvent } from "@react-three/fiber";
import { fetchBuffer, sceneFileUrl } from "../lib/api";
import { useStore } from "../lib/store";
import { EVIDENCE, EVIDENCE_COLORS, type LayerMeta } from "../lib/types";

const SEMANTIC_COLORS: Record<string, string> = {
  unknown: "#9aa0a6", road: "#6c6f7d", sidewalk: "#c8b9a6", terrain: "#9dbf9e", building: "#c9ada7", wall: "#a39594",
  fence: "#9a8c98", pole: "#f2e94e", traffic_light: "#ffbe0b", traffic_sign: "#fb5607", vegetation: "#3a7d44",
  sky: "#87ceeb", person: "#e4572e", car: "#2e86ab", truck: "#1b4965", bus: "#5fa8d3", train: "#62b6cb",
  motorcycle: "#f3a712", bicycle: "#f3a712", ground: "#8d8d8d",
};

interface Loaded {
  pos: Float32Array;
  idx?: Uint32Array;
  col?: Uint8Array;
  evi?: Uint8Array;
  conf?: Uint8Array;
  lab?: Uint8Array;
}

const cache = new Map<string, Promise<Loaded>>();

function load(caseId: string, l: LayerMeta): Promise<Loaded> {
  const key = caseId + l.id + JSON.stringify(l.stats?.bbox_min ?? "");
  if (!cache.has(key)) {
    const f = l.files;
    const get = (k: string) => (f[k] ? fetchBuffer(sceneFileUrl(caseId, f[k])) : Promise.resolve(null));
    cache.set(key, Promise.all([get("positions"), get("indices"), get("colors"), get("evidence"), get("confidence"), get("labels")]).then(([p, i, c, e, cf, lb]) => ({
      pos: new Float32Array(p!), idx: i ? new Uint32Array(i) : undefined, col: c ? new Uint8Array(c) : undefined,
      evi: e ? new Uint8Array(e) : undefined, conf: cf ? new Uint8Array(cf) : undefined, lab: lb ? new Uint8Array(lb) : undefined,
    })));
  }
  return cache.get(key)!;
}

function colorsFor(d: Loaded, mode: string, labels: string[]): Uint8Array {
  const n = d.pos.length / 3;
  const out = new Uint8Array(n * 3);
  if (mode === "photo" && d.col) return d.col;
  const tmp = new THREE.Color();
  const palette = (hex: string) => { tmp.set(hex); return [tmp.r * 255, tmp.g * 255, tmp.b * 255]; };
  if (mode === "evidence" && d.evi) {
    const pal = EVIDENCE.map((e) => palette(EVIDENCE_COLORS[e]));
    for (let i = 0; i < n; i++) { const c = pal[d.evi[i]] ?? pal[5]; out[3 * i] = c[0]; out[3 * i + 1] = c[1]; out[3 * i + 2] = c[2]; }
  } else if (mode === "semantic" && d.lab) {
    const pal = labels.map((l) => palette(SEMANTIC_COLORS[l] ?? "#999"));
    for (let i = 0; i < n; i++) { const c = pal[d.lab[i]] ?? pal[0]; out[3 * i] = c[0]; out[3 * i + 1] = c[1]; out[3 * i + 2] = c[2]; }
  } else if (mode === "confidence" && d.conf) {
    for (let i = 0; i < n; i++) {
      const v = d.conf[i] / 255; // red (low) -> yellow -> green (high)
      out[3 * i] = Math.round(255 * Math.min(1, 2 * (1 - v))); out[3 * i + 1] = Math.round(255 * Math.min(1, 2 * v)); out[3 * i + 2] = 60;
    }
  } else if (d.col) return d.col;
  else out.fill(170);
  return out;
}

export function MeshLayer({ layer, onPick }: { layer: LayerMeta; onPick: (e: ThreeEvent<MouseEvent>, evidence: string | null) => void }) {
  const caseId = useStore((s) => s.caseId)!;
  const mode = useStore((s) => s.colorMode);
  const wire = useStore((s) => s.wireframe);
  const evVis = useStore((s) => s.evidenceVisible);
  const labels = useStore((s) => s.scene!.scene_labels);
  const [data, setData] = useState<Loaded | null>(null);
  useEffect(() => { load(caseId, layer).then(setData).catch((e) => console.error(e)); }, [caseId, layer.id]);
  const geom = useMemo(() => {
    if (!data) return null;
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(data.pos, 3));
    if (data.idx) {
      let idx = data.idx;
      const hidden = EVIDENCE.map((c) => evVis[c] === false);
      if (data.evi && hidden.some(Boolean)) {
        const keep: number[] = [];
        for (let f = 0; f < idx.length; f += 3) {
          if (!hidden[data.evi[idx[f]]] && !hidden[data.evi[idx[f + 1]]] && !hidden[data.evi[idx[f + 2]]]) keep.push(idx[f], idx[f + 1], idx[f + 2]);
        }
        idx = new Uint32Array(keep);
      }
      g.setIndex(new THREE.BufferAttribute(idx, 1));
      g.computeVertexNormals();
    }
    g.computeBoundingSphere();
    g.computeBoundingBox();
    return g;
  }, [data, evVis]);
  useEffect(() => {
    if (!geom || !data) return;
    geom.setAttribute("color", new THREE.BufferAttribute(colorsFor(data, mode, labels), 3, true));
  }, [geom, data, mode, labels]);
  useEffect(() => () => geom?.dispose(), [geom]);
  if (!geom || !data) return null;
  if (layer.kind === "points") {
    return (
      <points geometry={geom} onClick={(e) => { e.stopPropagation(); onPick(e, data.evi && e.index != null ? EVIDENCE[data.evi[e.index]] : null); }}>
        <pointsMaterial size={0.06} sizeAttenuation vertexColors />
      </points>
    );
  }
  return (
    <mesh geometry={geom} onClick={(e) => {
      e.stopPropagation();
      const f = e.face;
      onPick(e, f && data.evi ? EVIDENCE[data.evi[f.a]] : null);
    }}>
      {mode === "photo" ? <meshBasicMaterial vertexColors side={THREE.DoubleSide} wireframe={wire} /> : <meshLambertMaterial vertexColors side={THREE.DoubleSide} wireframe={wire} />}
    </mesh>
  );
}

const texCache = new Map<string, THREE.Texture>();
export function useTexture(url: string | null): THREE.Texture | null {
  const [t, setT] = useState<THREE.Texture | null>(url ? texCache.get(url) ?? null : null);
  useEffect(() => {
    if (!url) return;
    if (texCache.has(url)) { setT(texCache.get(url)!); return; }
    new THREE.TextureLoader().load(url, (tex) => {
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.anisotropy = 4;
      texCache.set(url, tex);
      setT(tex);
    });
  }, [url]);
  return t;
}

export function CoverageLayer({ layer }: { layer: LayerMeta }) {
  const caseId = useStore((s) => s.caseId)!;
  const tex = useTexture(sceneFileUrl(caseId, layer.files.texture));
  const geom = useMemo(() => {
    const c = layer.corners_xz!;
    const g = new THREE.BufferGeometry();
    // corners: (x0,z0), (x1,z0), (x1,z1), (x0,z1); image row 0 = z0
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array([c[0][0], 0.02, c[0][1], c[1][0], 0.02, c[1][1], c[2][0], 0.02, c[2][1], c[3][0], 0.02, c[3][1]]), 3));
    g.setAttribute("uv", new THREE.BufferAttribute(new Float32Array([0, 1, 1, 1, 1, 0, 0, 0]), 2));
    g.setIndex([0, 2, 1, 0, 3, 2]);
    return g;
  }, [layer]);
  if (!tex) return null;
  return (
    <mesh geometry={geom} renderOrder={2}>
      <meshBasicMaterial map={tex} transparent depthWrite={false} side={THREE.DoubleSide} />
    </mesh>
  );
}
