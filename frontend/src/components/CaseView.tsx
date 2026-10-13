import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import { useStore } from "../lib/store";
import { goHome } from "../App";
import LeftPanel from "./LeftPanel";
import Inspector from "./Inspector";
import Timeline from "./Timeline";
import VideoDock from "./VideoDock";
import ProgressPanel from "./ProgressPanel";
import AnalyzeDialog from "./AnalyzeDialog";
import ReportDialog from "./ReportDialog";
import FrameViewer from "./FrameViewer";
import Viewport from "../viewer/Viewport";

export default function CaseView({ caseId }: { caseId: string }) {
  const detail = useStore((s) => s.detail);
  const scene = useStore((s) => s.scene);
  const setDetail = useStore((s) => s.setDetail);
  const setScene = useStore((s) => s.setScene);
  const notify = useStore((s) => s.notify);
  const [analyzeOpen, setAnalyzeOpen] = useState(false);
  const [reportOpen, setReportOpen] = useState(false);
  const lastJobState = useRef<string | null>(null);

  const loadScene = useCallback(() => {
    api.scene(caseId).then(setScene).catch(() => setScene(null));
  }, [caseId, setScene]);

  const refresh = useCallback(async () => {
    try {
      const d = await api.case(caseId);
      setDetail(d);
      const state = d.job ? d.job.status : d.status?.state;
      if (lastJobState.current && ["running", "queued", "cancel_requested"].includes(lastJobState.current) && !d.job) {
        // job just ended
        if (d.has_scene) loadScene();
        notify(`Analysis ${d.status?.state ?? "finished"}`, d.status?.state?.startsWith("completed") ? "ok" : "error");
      }
      lastJobState.current = d.job ? d.job.status : null;
      return state;
    } catch (e: any) {
      notify(e.message, "error");
    }
  }, [caseId, setDetail, loadScene, notify]);

  useEffect(() => {
    refresh().then(() => {
      const d = useStore.getState().detail;
      if (d?.has_scene) loadScene();
      if (d && d.videos.length && !d.has_scene && !d.job) setAnalyzeOpen(true);
    });
  }, [caseId]);

  useEffect(() => {
    const running = !!detail?.job;
    const t = setInterval(() => refresh(), running ? 1200 : 6000);
    return () => clearInterval(t);
  }, [detail?.job, refresh]);

  // play clock
  useEffect(() => {
    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      const st = useStore.getState();
      if (st.playing && st.scene) {
        const t = st.time + ((now - last) / 1000) * st.rate;
        if (t >= st.scene.time.end) {
          st.setTime(st.scene.time.end);
          st.setPlaying(false);
        } else st.setTime(t);
      }
      last = now;
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  // keyboard shortcuts
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).tagName.match(/INPUT|TEXTAREA|SELECT/)) return;
      const st = useStore.getState();
      if (e.code === "Space") {
        e.preventDefault();
        if ((e.target as HTMLElement).tagName === "BUTTON") (e.target as HTMLElement).blur(); // no double toggle
        st.setPlaying(!st.playing);
      }
      else if (e.key === "ArrowRight") st.setTime(st.time + (e.shiftKey ? 1 : 0.1));
      else if (e.key === "ArrowLeft") st.setTime(st.time - (e.shiftKey ? 1 : 0.1));
      else if (e.key === "m") st.setTool(st.tool === "measure" ? "navigate" : "measure");
      else if (e.key === "h") st.setTool(st.tool === "height" ? "navigate" : "height");
      else if (e.key === "k") st.setTool(st.tool === "calibrate" ? "navigate" : "calibrate");
      else if (e.key === "Escape") { st.setTool("navigate"); st.select(null); }
      else if (e.key === "t") st.requestView("top");
      else if (e.key === "o") st.requestView("overview");
      else if (e.key === "f" && st.selection) st.requestView("focus");
      else if (e.key === "1") st.setColorMode("photo");
      else if (e.key === "2") st.setColorMode("evidence");
      else if (e.key === "3") st.setColorMode("semantic");
      else if (e.key === "4") st.setColorMode("confidence");
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const cs = scene?.coordinate_system;
  const job = detail?.job;
  return (
    <div className="caseview">
      <div className="topbar">
        <button className="icon" onClick={goHome} title="All cases">◀</button>
        <span className="brand" style={{ fontSize: 12 }}>METRICA</span>
        <b style={{ fontSize: 14 }}>{detail?.title ?? "…"}</b>
        {detail?.reference && <span className="muted">{detail.reference}</span>}
        {cs && (
          <span className={`badge ${cs.status}`} title={cs.status_text + (cs.scale.scale_sigma_rel ? ` (1σ ${(cs.scale.scale_sigma_rel * 100).toFixed(1)}%)` : "")}>
            {cs.status.replace("_", " ")} · {cs.unit}
            {cs.scale.scale_sigma_rel ? ` ±${(cs.scale.scale_sigma_rel * 100).toFixed(0)}%` : ""}
          </span>
        )}
        <span className="grow" />
        {job && <span className="badge running">analysis {job.status}</span>}
        <button onClick={() => setAnalyzeOpen(true)} disabled={!!job || !detail?.videos.length}>Analyze…</button>
        <button disabled={!scene} onClick={() => setReportOpen(true)}>Report…</button>
      </div>
      <LeftPanel onRefresh={refresh} onReloadScene={loadScene} />
      <div className="viewwrap">
        {scene ? <Viewport /> : <div className="empty" style={{ paddingTop: 120 }}>{detail?.videos.length ? "No reconstruction yet — run the analysis." : "Add videos to this case to begin."}</div>}
        {scene && <VideoDock />}
        {(job || (detail && !scene && detail.status?.state && detail.status.state !== "new")) && <ProgressPanel detail={detail!} />}
      </div>
      <Inspector />
      <div className="timeline">{scene ? <Timeline /> : <div className="empty">Timeline appears after analysis.</div>}</div>
      {analyzeOpen && detail && <AnalyzeDialog detail={detail} onClose={() => setAnalyzeOpen(false)} onStarted={() => { setAnalyzeOpen(false); refresh(); }} />}
      {reportOpen && <ReportDialog caseId={caseId} onClose={() => setReportOpen(false)} />}
      <FrameViewer caseId={caseId} />
    </div>
  );
}
