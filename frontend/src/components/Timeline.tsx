import { useEffect, useMemo, useRef, useState } from "react";
import { fmtTime, useStore } from "../lib/store";

const EVENT_COLORS: Record<string, string> = {
  appearance: "#3ccf7a", disappearance: "#ff6b6b", proximity: "#ffb347", path_crossing: "#b98cff",
  stationary_period: "#4cc2ff", face_observed: "#f15bb5", text_candidate: "#fee440", track_relink: "#f5b041",
  observation_gap: "#5d6577",
};
const LANE_H = 13;
const TOP = 34;

export default function Timeline() {
  const scene = useStore((s) => s.scene)!;
  const time = useStore((s) => s.time);
  const playing = useStore((s) => s.playing);
  const rate = useStore((s) => s.rate);
  const sel = useStore((s) => s.selection);
  const { setTime, setPlaying, setRate, select } = useStore.getState();
  const canvas = useRef<HTMLCanvasElement>(null);
  const wrap = useRef<HTMLDivElement>(null);
  const [view, setView] = useState<[number, number]>([scene.time.start, Math.max(scene.time.end, scene.time.start + 1)]);
  const [scroll, setScroll] = useState(0);
  const [hoverEv, setHoverEv] = useState<string | null>(null);
  useEffect(() => setView([scene.time.start, Math.max(scene.time.end, scene.time.start + 1)]), [scene]);
  const agents = useMemo(() => scene.agents.filter((a) => a.samples.length), [scene]);
  const events = useMemo(() => scene.events.filter((e) => e.type !== "observation_gap"), [scene]);

  useEffect(() => {
    const c = canvas.current, w = wrap.current;
    if (!c || !w) return;
    const dpr = window.devicePixelRatio || 1;
    const W = w.clientWidth, H = w.clientHeight;
    c.width = W * dpr; c.height = H * dpr; c.style.width = W + "px"; c.style.height = H + "px";
    const g = c.getContext("2d")!;
    g.scale(dpr, dpr);
    g.fillStyle = "#131a22"; g.fillRect(0, 0, W, H);
    const L = 90;
    const [t0, t1] = view;
    const x = (t: number) => L + ((t - t0) / (t1 - t0)) * (W - L - 8);
    // ruler
    const span = t1 - t0;
    const step = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600].find((s) => span / s < (W - L) / 70) ?? 600;
    g.fillStyle = "#8b97a8"; g.font = "10px Inter, sans-serif"; g.strokeStyle = "#243040";
    for (let t = Math.ceil(t0 / step) * step; t <= t1; t += step) {
      const X = x(t);
      g.beginPath(); g.moveTo(X, 12); g.lineTo(X, H); g.stroke();
      g.fillText(fmtTime(t), X + 2, 10);
    }
    // events lane
    g.fillStyle = "#8b97a8"; g.fillText("events", 6, 26);
    for (const e of events) {
      const X = x(e.t);
      if (X < L || X > W) continue;
      g.fillStyle = EVENT_COLORS[e.type] ?? "#ccc";
      g.beginPath(); g.moveTo(X, 17); g.lineTo(X + 4, 27); g.lineTo(X - 4, 27); g.closePath(); g.fill();
      if (sel?.kind === "event" && sel.id === e.id) { g.strokeStyle = "#fff"; g.stroke(); }
    }
    // track lanes
    const visible = Math.floor((H - TOP) / LANE_H);
    agents.slice(scroll, scroll + visible).forEach((a, k) => {
      const y = TOP + k * LANE_H;
      const selected = sel?.kind === "agent" && sel.id === a.id;
      g.fillStyle = selected ? "#cfefff" : "#8b97a8";
      g.fillText(a.id, 6, y + 9);
      const col = scene.categories[a.category]?.color ?? "#aaa";
      const ts = a.samples.map((s) => s.t);
      let s0 = ts[0];
      for (let i = 1; i <= ts.length; i++) {
        if (i === ts.length || ts[i] - ts[i - 1] > 1.0) {
          g.fillStyle = col; g.globalAlpha = selected ? 1 : 0.75;
          g.fillRect(x(s0), y + 2, Math.max(1.5, x(ts[i - 1]) - x(s0)), LANE_H - 4);
          if (i < ts.length) { // gap: dashed connector (not observed)
            g.globalAlpha = 0.6; g.strokeStyle = col; g.setLineDash([2, 3]);
            g.beginPath(); g.moveTo(x(ts[i - 1]), y + LANE_H / 2); g.lineTo(x(ts[i]), y + LANE_H / 2); g.stroke(); g.setLineDash([]);
            s0 = ts[i];
          }
        }
      }
      g.globalAlpha = 1;
    });
    if (agents.length > visible) { g.fillStyle = "#5d6577"; g.fillText(`${scroll + 1}-${Math.min(agents.length, scroll + visible)} of ${agents.length} (scroll)`, W - 140, H - 3); }
    // playhead
    const X = x(time);
    g.strokeStyle = "#ffb347"; g.lineWidth = 1.5; g.beginPath(); g.moveTo(X, 0); g.lineTo(X, H); g.stroke(); g.lineWidth = 1;
  }, [scene, time, view, scroll, sel, agents, events]);

  const toTime = (clientX: number) => {
    const r = wrap.current!.getBoundingClientRect();
    const L = 90;
    return view[0] + ((clientX - r.left - L) / (r.width - L - 8)) * (view[1] - view[0]);
  };
  const onDown = (e: React.MouseEvent) => {
    const r = wrap.current!.getBoundingClientRect();
    const y = e.clientY - r.top;
    const t = toTime(e.clientX);
    if (y < 30) {
      const near = events.filter((ev) => Math.abs(toTime(e.clientX) - ev.t) < (view[1] - view[0]) * 0.006);
      if (near.length) { select({ kind: "event", id: near[0].id }); setTime(near[0].t); return; }
    } else {
      const k = Math.floor((y - TOP) / LANE_H) + scroll;
      if (agents[k] && e.clientX - r.left < 90) { select({ kind: "agent", id: agents[k].id }); return; }
    }
    setTime(t);
    const move = (ev: MouseEvent) => setTime(toTime(ev.clientX));
    const up = () => { window.removeEventListener("mousemove", move); window.removeEventListener("mouseup", up); };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  };
  const onWheel = (e: React.WheelEvent) => {
    if (e.shiftKey || Math.abs(e.deltaY) < Math.abs(e.deltaX)) {
      setScroll(Math.max(0, Math.min(agents.length - 1, scroll + Math.sign(e.deltaY || e.deltaX))));
      return;
    }
    const tc = toTime(e.clientX);
    const f = e.deltaY > 0 ? 1.25 : 0.8;
    const a = Math.max(scene.time.start - 1, tc - (tc - view[0]) * f);
    const b = Math.min(scene.time.end + 1, tc + (view[1] - tc) * f);
    if (b - a > 0.2) setView([a, b]);
  };
  return (
    <>
      <div className="row" style={{ padding: "4px 8px", borderBottom: "1px solid var(--line)" }}>
        <button className="icon" onClick={() => setTime(scene.time.start)}>⏮</button>
        <button className="icon" onClick={() => setTime(time - 0.1)}>◀</button>
        <button className="icon primary" style={{ width: 54 }} onClick={() => setPlaying(!playing)}>{playing ? "❚❚" : "▶"}</button>
        <button className="icon" onClick={() => setTime(time + 0.1)}>▶</button>
        <select value={rate} onChange={(e) => setRate(parseFloat(e.target.value))}>{[0.1, 0.25, 0.5, 1, 2, 4].map((r) => <option key={r} value={r}>{r}×</option>)}</select>
        <span className="mono" style={{ fontSize: 13 }}>{fmtTime(time)}</span>
        <span className="muted small">case time (video time + offset)</span>
        <span className="grow" />
        <span className="muted small">wheel: zoom · shift+wheel: scroll tracks</span>
        <button className="icon" onClick={() => setView([scene.time.start, scene.time.end])}>fit</button>
      </div>
      <div ref={wrap} style={{ flex: 1, position: "relative", cursor: "col-resize" }} onMouseDown={onDown} onWheel={onWheel}>
        <canvas ref={canvas} />
      </div>
    </>
  );
}
