import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { useStore } from "../lib/store";
import type { CaseDetail } from "../lib/types";

const ADVANCED: [string, string, string][] = [
  ["analysis_fps", "number", "Frames per second analysed by the detector/tracker"],
  ["max_keyframes", "number", "Maximum keyframes per moving video (SfM)"],
  ["hfov_deg", "number", "Known horizontal field of view in degrees (leave empty if unknown)"],
  ["person_height_m", "number", "Assumed mean stature of adults (scale prior)"],
  ["detector", "text", "Detector model key (yolox_s, yolox_m, yolo11n*)"],
  ["use_semantic_segmentation", "bool", "Dense scene semantics (roads, buildings, poles, vegetation)"],
  ["use_instance_masks", "bool", "Instance masks (EfficientSAM) for subjects and objects"],
  ["use_faces", "bool", "Face detection (no recognition)"],
  ["use_ocr", "bool", "OCR candidates on vehicles"],
  ["infer_hidden_surfaces", "bool", "Add clearly labelled inferred surfaces (hidden ground/walls)"],
  ["process_all_frames", "bool", "Analyse every frame (slow)"],
  ["joint_sfm", "bool", "Reconstruct moving videos jointly"],
];

export default function AnalyzeDialog({ detail, onClose, onStarted }: { detail: CaseDetail; onClose: () => void; onStarted: () => void }) {
  const [sys, setSys] = useState<any>(null);
  const [profile, setProfile] = useState("standard");
  const [over, setOver] = useState<Record<string, any>>({});
  const [modes, setModes] = useState<Record<string, string>>({});
  const [force, setForce] = useState(false);
  const notify = useStore((s) => s.notify);
  useEffect(() => { api.system().then(setSys); }, []);
  const base = sys?.profiles?.[profile] ?? {};
  const start = async () => {
    const overrides: Record<string, any> = {};
    for (const [k, v] of Object.entries(over)) if (v !== "" && v !== undefined) overrides[k] = v;
    const cm = Object.fromEntries(Object.entries(modes).filter(([, v]) => v && v !== "auto"));
    if (Object.keys(cm).length) overrides.capture_mode_overrides = cm;
    try {
      const r = await api.analyze(detail.id, profile, overrides, force);
      if (r.missing_models?.length) notify(`Missing models: ${r.missing_models.join(", ")} — affected stages will run in a documented degraded mode.`, "error");
      onStarted();
    } catch (e: any) { notify(e.message, "error"); }
  };
  return (
    <div className="overlay">
      <div className="modal wide">
        <div className="row"><h3 className="grow">Analyse {detail.videos.length} video(s)</h3><button className="icon" onClick={onClose}>✕</button></div>
        <div className="muted small" style={{ margin: "6px 0 12px" }}>The default settings run the whole pipeline automatically. Results of each stage are cached: re-running only recomputes what changed.</div>
        <table className="list">
          <thead><tr><th>Video</th><th>Format</th><th>Detected capture</th><th>Treat as</th></tr></thead>
          <tbody>
            {detail.videos.map((v) => (
              <tr key={v.id}>
                <td><b>{v.id}</b> <span className="mono">{v.original_name}</span><div className="muted small mono">{v.sha256_intake.slice(0, 24)}…</div></td>
                <td className="small">{v.ingest ? `${v.ingest.frame_size.width}×${v.ingest.frame_size.height} · ${v.ingest.timing.fps_measured.toFixed(2)} fps · ${v.ingest.timing.duration_s?.toFixed(1)} s` : <span className="muted">probed at ingest</span>}</td>
                <td className="small">{v.ingest ? `${v.ingest.mode.mode} (${v.ingest.mode.confidence.toFixed(2)})` : "auto"}</td>
                <td><select value={modes[v.id] ?? "auto"} onChange={(e) => setModes({ ...modes, [v.id]: e.target.value })}><option value="auto">auto</option><option value="static">static camera</option><option value="rotating">panning / tilting camera</option><option value="moving">moving camera</option></select></td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="row" style={{ margin: "14px 0" }}>
          <b>Profile</b>
          {["fast", "standard", "quality"].map((p) => <button key={p} className={profile === p ? "active" : ""} onClick={() => setProfile(p)}>{p}</button>)}
          <span className="muted small">{profile === "fast" ? "quick look: fewer frames/keyframes, no semantics or masks" : profile === "quality" ? "denser sampling, larger detector, more masks — slower" : "balanced default"}</span>
        </div>
        {sys && !sys.hardware.accelerated && <div className="callout small">No GPU acceleration detected: neural models run on the CPU. A 1-minute clip typically takes several minutes in the standard profile.</div>}
        <details style={{ marginTop: 10 }}>
          <summary>Advanced settings</summary>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "6px 18px", marginTop: 8 }}>
            {ADVANCED.map(([k, type, help]) => (
              <label key={k} className="row small" title={help}>
                <span className="grow">{k}</span>
                {type === "bool" ? (
                  <input type="checkbox" checked={over[k] ?? base[k] ?? false} onChange={(e) => setOver({ ...over, [k]: e.target.checked })} />
                ) : (
                  <input style={{ width: 110 }} placeholder={String(base[k] ?? "")} value={over[k] ?? ""} onChange={(e) => setOver({ ...over, [k]: type === "number" ? (e.target.value === "" ? "" : parseFloat(e.target.value)) : e.target.value })} />
                )}
              </label>
            ))}
            <label className="row small"><span className="grow">recompute every stage</span><input type="checkbox" checked={force} onChange={(e) => setForce(e.target.checked)} /></label>
          </div>
          <div className="muted small" style={{ marginTop: 6 }}>* Ultralytics YOLO11 is AGPL-3.0 licensed and must be downloaded explicitly (opt-in).</div>
        </details>
        <div className="row" style={{ marginTop: 16 }}>
          <span className="grow" />
          <button onClick={onClose}>Cancel</button>
          <button className="primary" onClick={start}>Start analysis</button>
        </div>
      </div>
    </div>
  );
}
