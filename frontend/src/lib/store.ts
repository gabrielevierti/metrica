// Application state (zustand). The case clock (`time`) drives the 3D scene, videos and timeline.
import { create } from "zustand";
import type { CaseDetail, Scene, Selection, Vec3 } from "./types";

export type Tool = "navigate" | "measure" | "height" | "angle" | "calibrate" | "inspect";
export type ColorMode = "photo" | "evidence" | "semantic" | "confidence";
export type NavMode = "orbit" | "fly";

export interface PickedPoint {
  p: Vec3;
  evidence: string | null;
  source: string;
}

interface State {
  caseId: string | null;
  detail: CaseDetail | null;
  scene: Scene | null;
  time: number;
  playing: boolean;
  rate: number;
  selection: Selection;
  hover: string | null;
  tool: Tool;
  colorMode: ColorMode;
  navMode: NavMode;
  wireframe: boolean;
  layerVisible: Record<string, boolean>;
  categoryVisible: Record<string, boolean>;
  evidenceVisible: Record<string, boolean>;
  show: { entities: boolean; agents: boolean; trajectories: boolean; cameras: boolean; grid: boolean; inferred: boolean; uncertainty: boolean; openings: boolean; labels: boolean };
  picked: PickedPoint[];
  measurementsVersion: number;
  viewRequest: { kind: "top" | "overview" | "camera" | "focus"; target?: string; nonce: number } | null;
  frameViewer: { video: string; frame: number; bbox?: number[]; title?: string } | null;
  toast: { text: string; kind: "info" | "error" | "ok" } | null;
  setCase: (id: string | null) => void;
  setDetail: (d: CaseDetail | null) => void;
  setScene: (s: Scene | null) => void;
  setTime: (t: number) => void;
  setPlaying: (p: boolean) => void;
  setRate: (r: number) => void;
  select: (s: Selection) => void;
  setHover: (id: string | null) => void;
  setTool: (t: Tool) => void;
  setColorMode: (m: ColorMode) => void;
  setNavMode: (m: NavMode) => void;
  toggle: (k: keyof State["show"]) => void;
  setLayerVisible: (id: string, v: boolean) => void;
  setCategoryVisible: (c: string, v: boolean) => void;
  setEvidenceVisible: (c: string, v: boolean) => void;
  setWireframe: (v: boolean) => void;
  pick: (p: PickedPoint) => void;
  clearPicked: () => void;
  bumpMeasurements: () => void;
  requestView: (kind: "top" | "overview" | "camera" | "focus", target?: string) => void;
  openFrame: (v: State["frameViewer"]) => void;
  notify: (text: string, kind?: "info" | "error" | "ok") => void;
}

export const useStore = create<State>((set, get) => ({
  caseId: null,
  detail: null,
  scene: null,
  time: 0,
  playing: false,
  rate: 1,
  selection: null,
  hover: null,
  tool: "navigate",
  colorMode: "photo",
  navMode: "orbit",
  wireframe: false,
  layerVisible: {},
  categoryVisible: {},
  evidenceVisible: {},
  show: { entities: true, agents: true, trajectories: true, cameras: true, grid: true, inferred: true, uncertainty: true, openings: true, labels: true },
  picked: [],
  measurementsVersion: 0,
  viewRequest: null,
  frameViewer: null,
  toast: null,
  setCase: (id) => set({ caseId: id, scene: null, detail: null, selection: null, picked: [], time: 0, playing: false }),
  setDetail: (d) => set({ detail: d }),
  setScene: (s) => {
    const lv: Record<string, boolean> = {};
    s?.layers.forEach((l) => (lv[l.id] = get().layerVisible[l.id] ?? l.default_visible));
    set({ scene: s, layerVisible: lv, time: s ? Math.max(get().time, s.time.start) : 0 });
  },
  setTime: (t) => {
    const s = get().scene;
    if (s) t = Math.min(Math.max(t, s.time.start), s.time.end);
    set({ time: t });
  },
  setPlaying: (p) => set({ playing: p }),
  setRate: (r) => set({ rate: r }),
  select: (s) => set({ selection: s }),
  setHover: (id) => set({ hover: id }),
  setTool: (t) => set({ tool: t, picked: [] }),
  setColorMode: (m) => set({ colorMode: m }),
  setNavMode: (m) => set({ navMode: m }),
  toggle: (k) => set({ show: { ...get().show, [k]: !get().show[k] } }),
  setLayerVisible: (id, v) => set({ layerVisible: { ...get().layerVisible, [id]: v } }),
  setCategoryVisible: (c, v) => set({ categoryVisible: { ...get().categoryVisible, [c]: v } }),
  setEvidenceVisible: (c, v) => set({ evidenceVisible: { ...get().evidenceVisible, [c]: v } }),
  setWireframe: (v) => set({ wireframe: v }),
  pick: (p) => set({ picked: [...get().picked, p] }),
  clearPicked: () => set({ picked: [] }),
  bumpMeasurements: () => set({ measurementsVersion: get().measurementsVersion + 1 }),
  requestView: (kind, target) => set({ viewRequest: { kind, target, nonce: Math.random() } }),
  openFrame: (v) => set({ frameViewer: v }),
  notify: (text, kind = "info") => {
    set({ toast: { text, kind } });
    setTimeout(() => {
      if (get().toast?.text === text) set({ toast: null });
    }, 5000);
  },
}));

// helpers -----------------------------------------------------------------------------
export function fmt(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return v.toFixed(digits);
}

export function fmtTime(t: number): string {
  const sign = t < 0 ? "-" : "";
  t = Math.abs(t);
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  return `${sign}${m}:${s.toFixed(2).padStart(5, "0")}`;
}

export function agentPositionAt(a: { smoothed?: { t: number; p: Vec3 }[]; samples: { t: number; p: Vec3; s: number | null }[] }, t: number, maxGap = 1.0): { p: Vec3; interpolated: boolean; sigma: number | null } | null {
  const src = a.smoothed && a.smoothed.length ? a.smoothed : a.samples;
  if (!src.length || t < src[0].t - 0.05 || t > src[src.length - 1].t + 0.05) return null;
  let lo = 0, hi = src.length - 1;
  while (hi - lo > 1) {
    const m = (lo + hi) >> 1;
    if (src[m].t <= t) lo = m; else hi = m;
  }
  const A = src[lo], B = src[hi];
  if (B.t - A.t > maxGap && t > A.t + 0.05 && t < B.t - 0.05) return null; // inside an observation gap
  const w = B.t > A.t ? Math.min(1, Math.max(0, (t - A.t) / (B.t - A.t))) : 0;
  const p: Vec3 = [A.p[0] + (B.p[0] - A.p[0]) * w, A.p[1] + (B.p[1] - A.p[1]) * w, A.p[2] + (B.p[2] - A.p[2]) * w];
  const near = a.samples.length ? a.samples[Math.min(a.samples.length - 1, Math.max(0, lo))] : null;
  return { p, interpolated: Math.min(t - A.t, B.t - t) > 0.15, sigma: near?.s ?? null };
}
