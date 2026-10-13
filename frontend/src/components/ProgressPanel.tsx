import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { useStore } from "../lib/store";
import type { CaseDetail } from "../lib/types";

export default function ProgressPanel({ detail }: { detail: CaseDetail }) {
  const [status, setStatus] = useState<any>(detail.status);
  const [collapsed, setCollapsed] = useState(false);
  const notify = useStore((s) => s.notify);
  const scene = useStore((s) => s.scene);
  const job = detail.job;
  useEffect(() => {
    if (!job) { setStatus(detail.status); return; }
    const poll = () => api.job(job.id).then((r) => setStatus(r.status)).catch(() => {});
    poll();
    const t = setInterval(poll, 800);
    return () => clearInterval(t);
  }, [job?.id, detail.status]);
  if (!status?.stages) return null;
  if (!job && scene) return null;
  if (collapsed) return <button style={{ position: "absolute", bottom: 10, right: 10, zIndex: 7 }} onClick={() => setCollapsed(false)}>Progress {(status.progress * 100).toFixed(0)}%</button>;
  const eta = (() => {
    if (!job || !status.started || status.progress < 0.03) return null;
    const el = (Date.now() - Date.parse(status.started)) / 1000;
    return Math.max(0, el / status.progress - el);
  })();
  return (
    <div className={scene ? "" : "overlay"} style={scene ? { position: "absolute", left: "50%", top: 60, transform: "translateX(-50%)", zIndex: 7 } : {}}>
      <div className="modal" style={{ width: 560 }}>
        <div className="row">
          <h3 className="grow">{job ? "Analysis in progress" : `Last analysis: ${status.state}`}</h3>
          {job && <span className="muted small">{eta != null ? `~${Math.ceil(eta / 60)} min left (rough)` : ""}</span>}
          <button className="icon" onClick={() => setCollapsed(true)}>–</button>
        </div>
        <div className="progress" style={{ margin: "10px 0" }}><div style={{ width: `${(status.progress ?? 0) * 100}%` }} /></div>
        <div className="small muted" style={{ minHeight: 18 }}>{status.message}</div>
        <div style={{ marginTop: 8 }}>
          {status.stages.map((s: any) => (
            <div key={s.name} className="stage">
              <span className="small">{s.title}{s.reused ? <span className="muted"> (reused)</span> : null}</span>
              <span className={`badge ${s.status}`}>{s.status === "running" && status.stage === s.name ? `${((status.stage_progress ?? 0) * 100).toFixed(0)}%` : s.status}</span>
              {s.error && <span className="small" style={{ gridColumn: "1 / -1", color: "var(--bad)" }}>{String(s.error).split("\n")[0]}</span>}
              {s.warnings > 0 && <span className="small muted" style={{ gridColumn: "1 / -1" }}>{s.warnings} warning(s)</span>}
            </div>
          ))}
        </div>
        {job && (
          <div className="row" style={{ marginTop: 10 }}>
            <span className="muted small grow">Each stage saves its results: a cancelled or interrupted analysis resumes from the last completed stage.</span>
            <button className="danger" disabled={job.status === "cancel_requested"} onClick={() => api.cancel(job.id).then(() => notify("Cancelling after the current step…")).catch((e) => notify(e.message, "error"))}>Cancel</button>
          </div>
        )}
      </div>
    </div>
  );
}
