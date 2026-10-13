import { useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import { useStore } from "../lib/store";
import type { CaseSummary } from "../lib/types";
import { openCase } from "../App";

export default function Home() {
  const [cases, setCases] = useState<CaseSummary[]>([]);
  const [sys, setSys] = useState<any>(null);
  const [models, setModels] = useState<any>(null);
  const [title, setTitle] = useState("");
  const [ref, setRef] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [paths, setPaths] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [over, setOver] = useState(false);
  const [analyst, setAnalyst] = useState(localStorage.getItem("metrica.analyst") || "");
  const notify = useStore((s) => s.notify);
  const inputRef = useRef<HTMLInputElement>(null);

  const refresh = () => {
    api.cases().then(setCases).catch((e) => notify(e.message, "error"));
    api.system().then(setSys).catch(() => {});
    api.models().then(setModels).catch(() => {});
  };
  useEffect(refresh, []);
  useEffect(() => {
    if (!models?.download?.running) return;
    const t = setInterval(() => api.models().then(setModels), 1500);
    return () => clearInterval(t);
  }, [models?.download?.running]);

  const create = async () => {
    if (!title.trim()) return notify("Give the case a title", "error");
    if (!files.length && !paths.trim()) return notify("Add at least one video (drop files or enter a local path)", "error");
    try {
      setBusy("Creating case…");
      const c = await api.createCase(title.trim(), ref.trim());
      for (const [i, f] of files.entries()) {
        await api.upload(c.id, f, (p) => setBusy(`Importing ${f.name} (${i + 1}/${files.length}) ${(p * 100).toFixed(0)}% — hashing and copying, originals are not modified`));
      }
      for (const p of paths.split("\n").map((x) => x.trim()).filter(Boolean)) {
        setBusy(`Importing ${p}…`);
        await api.importPath(c.id, p, true);
      }
      openCase(c.id);
    } catch (e: any) {
      notify(e.message, "error");
    } finally {
      setBusy(null);
    }
  };

  const missing = sys?.models?.missing_default ?? [];
  return (
    <div className="home">
      <header>
        <span className="brand">METRICA</span>
        <span className="muted">Evidence-oriented video-to-3D reconstruction · local-first</span>
        <span className="grow" />
        <label className="row small">Analyst
          <input style={{ width: 160 }} value={analyst} placeholder="name / ID" onChange={(e) => { setAnalyst(e.target.value); localStorage.setItem("metrica.analyst", e.target.value); }} />
        </label>
        <span className="muted small">v{sys?.version}</span>
      </header>
      <main>
        <div className="col">
          <div className="card col">
            <h3>New case</h3>
            <div className="row">
              <input className="grow" placeholder="Case title (e.g. Via Roma, 6 October)" value={title} onChange={(e) => setTitle(e.target.value)} />
              <input style={{ width: 180 }} placeholder="Case reference (optional)" value={ref} onChange={(e) => setRef(e.target.value)} />
            </div>
            <div
              className={`dropzone ${over ? "over" : ""}`}
              onDragOver={(e) => { e.preventDefault(); setOver(true); }}
              onDragLeave={() => setOver(false)}
              onDrop={(e) => { e.preventDefault(); setOver(false); setFiles([...files, ...Array.from(e.dataTransfer.files)]); }}
              onClick={() => inputRef.current?.click()}
            >
              <div style={{ fontSize: 15, marginBottom: 4 }}>Drop one or more video files here, or click to choose</div>
              <div className="small">CCTV, phone, body-worn, dashcam, drone… Each file is hashed (SHA-256) and copied read-only into the case.</div>
              <input ref={inputRef} type="file" multiple accept="video/*,.mkv,.avi,.mov,.mp4,.ts,.mts,.m4v" style={{ display: "none" }}
                onChange={(e) => setFiles([...files, ...Array.from(e.target.files || [])])} />
            </div>
            {files.length > 0 && (
              <div className="col small">
                {files.map((f, i) => (
                  <div key={i} className="row">
                    <span className="mono grow">{f.name}</span>
                    <span className="muted">{(f.size / 1e6).toFixed(1)} MB</span>
                    <button className="icon" onClick={() => setFiles(files.filter((_, j) => j !== i))}>✕</button>
                  </div>
                ))}
              </div>
            )}
            <details>
              <summary className="muted small">Large files: import by local path instead of uploading</summary>
              <textarea rows={2} style={{ width: "100%", marginTop: 6 }} placeholder="/path/to/video.mp4 (one per line)" value={paths} onChange={(e) => setPaths(e.target.value)} />
            </details>
            <div className="row">
              <button className="primary" disabled={!!busy} onClick={create}>Create case and import</button>
              {busy && <span className="muted small">{busy}</span>}
            </div>
          </div>
          <div className="card col">
            <div className="row"><h3>Cases</h3><span className="grow" /><span className="muted small">{sys?.workspace}</span></div>
            {cases.length === 0 && <div className="empty">No cases yet.</div>}
            <table className="list">
              <tbody>
                {cases.map((c) => (
                  <tr key={c.id} onClick={() => openCase(c.id)}>
                    <td><b>{c.title}</b><div className="muted small">{c.reference}</div></td>
                    <td className="muted small">{c.created?.slice(0, 16).replace("T", " ")}</td>
                    <td className="small">{c.videos} video{c.videos === 1 ? "" : "s"}</td>
                    <td><span className={`badge ${c.status === "completed" ? "ok" : c.status.includes("fail") ? "failed" : c.status === "running" ? "running" : "pending"}`}>{c.status}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <div className="col">
          <div className="card col">
            <h3>Models</h3>
            {missing.length > 0 ? (
              <div className="callout">Missing default models: {missing.join(", ")}. Analysis stages that need them will run in a documented degraded mode.</div>
            ) : (
              <div className="callout info">All default models are present and verified.</div>
            )}
            <table className="list">
              <tbody>
                {models?.models?.map((m: any) => {
                  const d = models.download.items?.[m.key];
                  return (
                    <tr key={m.key} title={m.notes || m.about}>
                      <td><b className="small">{m.key}</b><div className="muted small">{m.task}</div></td>
                      <td className="small">{m.license}{m.restricted && <div className="badge hyp">opt-in</div>}</td>
                      <td className="small">{d?.state === "downloading" ? `${((d.done / (d.total || 1)) * 100).toFixed(0)}%` : d?.state === "failed" ? <span style={{ color: "var(--bad)" }}>failed</span> : m.state === "ready" ? <span style={{ color: "var(--ok)" }}>ready</span> : m.state}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <div className="row wrap">
              <button disabled={models?.download?.running} onClick={() => api.downloadModels({}).then(() => api.models().then(setModels)).catch((e) => notify(e.message, "error"))}>Download default models</button>
              <button disabled={models?.download?.running} onClick={() => api.downloadModels({ include_optional: true }).then(() => api.models().then(setModels))}>+ optional</button>
            </div>
            <div className="muted small">Weights are fetched from their official sources (GitHub releases, OpenCV / ONNX model zoos), verified by SHA-256 and stored in {models?.directory}. Downloading is the only network access METRICA needs; analysis runs offline.</div>
          </div>
          <div className="card col small">
            <h3>This computer</h3>
            {sys && (
              <div className="kv">
                <span>CPU threads</span><span>{sys.hardware.cpu_count}</span>
                <span>Memory</span><span>{sys.hardware.memory_gb ?? "?"} GB</span>
                <span>Acceleration</span><span>{sys.hardware.accelerator ?? "none (CPU)"}</span>
                <span>Providers</span><span className="mono">{sys.hardware.onnx_providers.join(", ")}</span>
              </div>
            )}
            {sys?.hardware?.notes && <div className="muted">{sys.hardware.notes}</div>}
            <div className="muted">{sys?.bind_note}</div>
          </div>
        </div>
      </main>
    </div>
  );
}
