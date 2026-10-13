import { useEffect, useRef, useState } from "react";
import { fileUrl } from "../lib/api";
import { useStore } from "../lib/store";

// Synchronised review video(s). The case clock is the master: video time = case time - offset.
export default function VideoDock() {
  const scene = useStore((s) => s.scene)!;
  const caseId = useStore((s) => s.caseId)!;
  const [active, setActive] = useState(scene.videos[0]?.id);
  const [big, setBig] = useState(false);
  const [hidden, setHidden] = useState(false);
  const [boxes, setBoxes] = useState(true);
  const video = useRef<HTMLVideoElement>(null);
  const overlay = useRef<HTMLCanvasElement>(null);
  const time = useStore((s) => s.time);
  const playing = useStore((s) => s.playing);
  const rate = useStore((s) => s.rate);
  const sel = useStore((s) => s.selection);
  const v = scene.videos.find((x) => x.id === active) ?? scene.videos[0];

  useEffect(() => {
    const el = video.current;
    if (!el || !v) return;
    const vt = time - v.offset;
    const inRange = vt >= 0 && vt <= (v.duration || 1e9);
    if (!inRange) { el.pause(); return; }
    el.playbackRate = rate;
    if (playing) {
      if (el.paused) el.play().catch(() => {});
      if (Math.abs(el.currentTime - vt) > 0.25) el.currentTime = vt;
    } else {
      if (!el.paused) el.pause();
      if (Math.abs(el.currentTime - vt) > 0.02) el.currentTime = vt;
    }
  }, [time, playing, rate, v?.id]);

  // detection boxes of tracks at the displayed frame
  useEffect(() => {
    const c = overlay.current, el = video.current;
    if (!c || !el || !v) return;
    const draw = () => {
      const W = el.clientWidth, H = el.clientHeight;
      c.width = W; c.height = H;
      const g = c.getContext("2d")!;
      g.clearRect(0, 0, W, H);
      if (!boxes) return;
      const t = useStore.getState().time;
      const sx = W / v.width, sy = H / v.height;
      for (const a of scene.agents) {
        if (a.video !== v.id || t < a.first_t - 0.2 || t > a.last_t + 0.2) continue;
        let best = null as any, bd = 1e9;
        for (const s of a.samples) { const d = Math.abs(s.t - t); if (d < bd) { bd = d; best = s; } }
        if (!best || bd > 0.6 / Math.max(1, v.fps / 10)) continue;
        const bb = best.b as number[] | undefined;
        const selected = sel?.kind === "agent" && sel.id === a.id;
        const col = scene.categories[a.category]?.color ?? "#fff";
        if (bb) {
          g.strokeStyle = col; g.lineWidth = selected ? 3 : 1.5;
          g.strokeRect(bb[0] * sx, bb[1] * sy, (bb[2] - bb[0]) * sx, (bb[3] - bb[1]) * sy);
        }
        g.fillStyle = selected ? "#ffb347" : col; g.font = "11px Inter, sans-serif";
                if (bb) g.fillText(a.id, bb[0] * sx, bb[1] * sy - 3);
      }
    };
    draw();
    const id = setInterval(draw, 120);
    return () => clearInterval(id);
  }, [v?.id, boxes, sel, scene]);

  if (!v) return null;
  if (hidden) return <button style={{ position: "absolute", top: 10, right: 10, zIndex: 6 }} onClick={() => setHidden(false)}>Show video</button>;
  const vt = time - v.offset;
  return (
    <div className={`videodock ${big ? "big" : ""}`}>
      <div className="vhead">
        {scene.videos.map((x) => <button key={x.id} className={x.id === v.id ? "active icon" : "icon"} onClick={() => setActive(x.id)}>{x.id}</button>)}
        <span className="muted small grow" style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{v.name}</span>
        <button className={boxes ? "active icon" : "icon"} onClick={() => setBoxes(!boxes)} title="Detection boxes of the tracks (observed samples only)">boxes</button>
        <button className="icon" onClick={() => setBig(!big)}>{big ? "–" : "+"}</button>
        <button className="icon" onClick={() => setHidden(true)}>✕</button>
      </div>
      <div className="videobox">
        <video ref={video} src={fileUrl(caseId, v.proxy)} muted playsInline preload="auto" />
        <canvas ref={overlay} />
        {(vt < 0 || vt > (v.duration || 1e9)) && <div className="overlay" style={{ position: "absolute" }}><span className="muted">outside this recording ({vt < 0 ? "before start" : "after end"})</span></div>}
      </div>
      <div className="row small" style={{ padding: "3px 8px" }}>
        <span className="mono">video t {vt.toFixed(3)} s · ~frame {Math.max(0, Math.round(vt * v.fps))}</span>
        <span className="grow" />
        <span className="muted">review proxy (H.264); evidence frames via "frame" buttons</span>
      </div>
    </div>
  );
}
