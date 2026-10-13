import { useEffect, useState } from "react";
import { api, fileUrl } from "../lib/api";
import { useStore } from "../lib/store";

export default function ReportDialog({ caseId, onClose }: { caseId: string; onClose: () => void }) {
  const [list, setList] = useState<any[]>([]);
  const [analyst, setAnalyst] = useState(localStorage.getItem("metrica.analyst") || "");
  const [purpose, setPurpose] = useState("");
  const [task, setTask] = useState<string | null>(null);
  const notify = useStore((s) => s.notify);
  const load = () => api.reports(caseId).then(setList).catch(() => {});
  useEffect(() => { load(); }, [caseId]);
  useEffect(() => {
    if (!task) return;
    const t = setInterval(async () => {
      const r = await api.reportTask(task);
      if (r.state !== "running") {
        setTask(null);
        if (r.state === "done") { notify("Report generated", "ok"); load(); } else notify(`Report failed: ${r.error}`, "error");
      }
    }, 1000);
    return () => clearInterval(t);
  }, [task]);
  return (
    <div className="overlay">
      <div className="modal">
        <div className="row"><h3 className="grow">Reports</h3><button className="icon" onClick={onClose}>✕</button></div>
        <div className="col" style={{ marginTop: 10 }}>
          <div className="row"><input className="grow" placeholder="Analyst name / ID" value={analyst} onChange={(e) => setAnalyst(e.target.value)} /><input className="grow" placeholder="Purpose (optional)" value={purpose} onChange={(e) => setPurpose(e.target.value)} /></div>
          <div className="muted small">The report includes the input inventory with hashes, methods, calibration, object inventory, people/face and vehicle observations, the event timeline, measurements, analyst verifications, snapshots, limitations and all parameters. Generation is recorded in the audit log.</div>
          <div className="row"><button className="primary" disabled={!!task} onClick={() => api.report(caseId, analyst, purpose).then((r) => setTask(r.task)).catch((e) => notify(e.message, "error"))}>{task ? "Generating…" : "Generate new report"}</button></div>
        </div>
        <div className="sep" />
        {list.length === 0 && <div className="muted">No reports yet.</div>}
        {list.map((r) => (
          <div key={r.report_id} className="row" style={{ padding: "6px 0", borderBottom: "1px solid var(--line)" }}>
            <span className="mono grow">{r.report_id}</span>
            <span className="muted small">{r.generated}</span>
            <a href={fileUrl(caseId, r.files.html)} target="_blank">HTML</a>
            {r.files.pdf && <a href={fileUrl(caseId, r.files.pdf)} target="_blank">PDF</a>}
            <a href={fileUrl(caseId, r.files.zip)} download>ZIP</a>
          </div>
        ))}
      </div>
    </div>
  );
}
