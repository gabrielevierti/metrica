import { useEffect, useMemo, useRef, useState } from "react";
import { api, fileUrl } from "../lib/api";
import { fmt, fmtTime, useStore } from "../lib/store";
import { EVIDENCE, EVIDENCE_COLORS, EVIDENCE_SHORT } from "../lib/types";

type Tab = "inputs" | "layers" | "objects" | "tracks" | "events" | "measure" | "case";

export default function LeftPanel({ onRefresh, onReloadScene }: { onRefresh: () => void; onReloadScene: () => void }) {
  const [tab, setTab] = useState<Tab>("layers");
  const scene = useStore((s) => s.scene);
  useEffect(() => { if (!scene) setTab("inputs"); }, [scene]);
  const tabs: [Tab, string][] = [["inputs", "Inputs"], ["layers", "Layers"], ["objects", "Objects"], ["tracks", "Tracks"], ["events", "Events"], ["measure", "Measure"], ["case", "Case"]];
  return (
    <div className="left">
      <div className="tabs">
        {tabs.map(([k, l]) => (
          <button key={k} className={tab === k ? "active" : ""} onClick={() => setTab(k)} disabled={!scene && !["inputs", "case"].includes(k)}>{l}</button>
        ))}
      </div>
      <div className="panelbody">
        {tab === "inputs" && <Inputs onRefresh={onRefresh} />}
        {tab === "layers" && scene && <Layers />}
        {tab === "objects" && scene && <Objects />}
        {tab === "tracks" && scene && <Tracks />}
        {tab === "events" && scene && <Events />}
        {tab === "measure" && scene && <Measure />}
        {tab === "case" && <CaseTab onReloadScene={onReloadScene} />}
      </div>
    </div>
  );
}

function Inputs({ onRefresh }: { onRefresh: () => void }) {
  const detail = useStore((s) => s.detail);
  const scene = useStore((s) => s.scene);
  const notify = useStore((s) => s.notify);
  const [busy, setBusy] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [offsets, setOffsets] = useState<Record<string, string>>({});
  if (!detail) return null;
  const upload = async (files: FileList | null) => {
    if (!files) return;
    try {
      for (const f of Array.from(files)) await api.upload(detail.id, f, (p) => setBusy(`${f.name} ${(p * 100).toFixed(0)}%`));
      onRefresh();
    } catch (e: any) { notify(e.message, "error"); } finally { setBusy(null); }
  };
  return (
    <div className="col">
      {detail.videos.map((v) => {
        const sv = scene?.videos.find((x) => x.id === v.id);
        const ing = v.ingest;
        return (
          <div key={v.id} className="card col" style={{ padding: 10 }}>
            <div className="row"><b>{v.id}</b><span className="mono grow" style={{ overflow: "hidden", textOverflow: "ellipsis" }}>{v.original_name}</span></div>
            {sv && <img src={fileUrl(detail.id, sv.thumbnail)} style={{ width: "100%", borderRadius: 4 }} />}
            <div className="kv">
              <span>SHA-256</span><span className="mono" style={{ wordBreak: "break-all" }}>{v.sha256_intake}</span>
              <span>Size</span><span>{(v.size_bytes / 1e6).toFixed(1)} MB {v.copied ? "· copied read-only" : "· referenced in place"}</span>
              {ing && <>
                <span>Format</span><span>{ing.probe.container} / {ing.probe.video.codec} · {ing.frame_size.width}×{ing.frame_size.height}</span>
                <span>Timing</span><span>{fmt(ing.timing.fps_measured, 2)} fps · {ing.timing.frames} frames · {fmt(ing.timing.duration_s, 1)} s{ing.timing.vfr ? " · VFR" : ""}</span>
                <span>Capture</span><span>{ing.mode.mode} ({fmt(ing.mode.confidence, 2)}){ing.mode.override ? " · manual" : ""}</span>
                <span>Quality</span><span>{fmt((ing.quality.usable_fraction ?? 0) * 100, 0)}% usable{ing.quality.flags?.length ? " · " + ing.quality.flags.join("; ") : ""}</span>
                <span>Decode</span><span>{ing.decode.decode_errors} errors · rotation {ing.decode.rotation_applied_deg}°</span>
                {ing.probe.creation_time && <><span>Created (tag)</span><span>{ing.probe.creation_time} <span className="muted">(device clock, unverified)</span></span></>}
              </>}
            </div>
            {sv && scene!.videos.length > 1 && (
              <div className="row small">
                <span className="muted">Offset</span>
                <span className="mono">{sv.offset.toFixed(3)} s</span>
                <span className="muted grow">{sv.sync?.method}</span>
                <input style={{ width: 70 }} placeholder="s" value={offsets[v.id] ?? ""} onChange={(e) => setOffsets({ ...offsets, [v.id]: e.target.value })} />
                <button className="icon" onClick={() => {
                  const val = parseFloat(offsets[v.id]);
                  if (Number.isNaN(val)) return;
                  api.sync(detail.id, v.id, val).then(() => { notify("Offset saved; re-running affected stages", "ok"); onRefresh(); }).catch((e) => notify(e.message, "error"));
                }}>Set</button>
              </div>
            )}
          </div>
        );
      })}
      <button disabled={!!detail.job} onClick={() => inputRef.current?.click()}>+ Add video</button>
      <input ref={inputRef} type="file" multiple accept="video/*" style={{ display: "none" }} onChange={(e) => upload(e.target.files)} />
      {busy && <div className="muted small">Importing {busy}</div>}
    </div>
  );
}

function Layers() {
  const scene = useStore((s) => s.scene)!;
  const st = useStore();
  const cats = useMemo(() => Array.from(new Set(scene.entities.map((e) => e.category).concat(scene.agents.map((a) => a.category)))), [scene]);
  return (
    <div className="col">
      <h4>Colour</h4>
      <div className="row wrap">
        {(["photo", "evidence", "semantic", "confidence"] as const).map((m, i) => (
          <button key={m} className={st.colorMode === m ? "active" : ""} onClick={() => st.setColorMode(m)} title={`key ${i + 1}`}>{m}</button>
        ))}
        <button className={st.wireframe ? "active" : ""} onClick={() => st.setWireframe(!st.wireframe)}>wireframe</button>
      </div>
      <h4>Reconstruction layers</h4>
      {scene.layers.map((l) => (
        <label key={l.id} className="row small" title={l.method}>
          <input type="checkbox" checked={!!st.layerVisible[l.id]} onChange={(e) => st.setLayerVisible(l.id, e.target.checked)} />
          <span className="grow">{l.unit} · {l.name ?? l.kind} <span className="muted">({l.kind})</span></span>
          <span className="muted">{l.stats.vertices ? `${(l.stats.vertices / 1000).toFixed(0)}k` : ""}</span>
        </label>
      ))}
      <h4>Evidence classes</h4>
      {EVIDENCE.map((c) => (
        <label key={c} className="row small" title={scene.evidence_classes[c]}>
          <input type="checkbox" checked={st.evidenceVisible[c] !== false} onChange={(e) => st.setEvidenceVisible(c, e.target.checked)} />
          <span className="swatch" style={{ background: EVIDENCE_COLORS[c] }} />
          <span>{EVIDENCE_SHORT[c]}</span>
        </label>
      ))}
      <h4>Show</h4>
      <div className="row wrap">
        {(Object.keys(st.show) as (keyof typeof st.show)[]).map((k) => (
          <button key={k} className={st.show[k] ? "active" : ""} onClick={() => st.toggle(k)}>{k}</button>
        ))}
      </div>
      <h4>Categories</h4>
      {cats.map((c) => (
        <label key={c} className="row small">
          <input type="checkbox" checked={st.categoryVisible[c] !== false} onChange={(e) => st.setCategoryVisible(c, e.target.checked)} />
          <span className="swatch" style={{ background: scene.categories[c]?.color ?? "#999" }} />
          <span>{scene.categories[c]?.display ?? c}</span>
        </label>
      ))}
    </div>
  );
}

function Objects() {
  const scene = useStore((s) => s.scene)!;
  const sel = useStore((s) => s.selection);
  const select = useStore((s) => s.select);
  const requestView = useStore((s) => s.requestView);
  return (
    <table className="list">
      <thead><tr><th>Object</th><th>Evidence</th></tr></thead>
      <tbody>
        {scene.entities.map((e) => (
          <tr key={e.id} className={sel?.kind === "entity" && sel.id === e.id ? "sel" : ""} onClick={() => select({ kind: "entity", id: e.id })} onDoubleClick={() => requestView("focus")}>
            <td><span className="swatch" style={{ background: scene.categories[e.category]?.color }} /> {e.display ?? e.category}<div className="muted mono">{e.id}</div></td>
            <td className="small"><span className="swatch" style={{ background: EVIDENCE_COLORS[e.evidence?.class ?? "UNKNOWN"] }} /> {EVIDENCE_SHORT[e.evidence?.class ?? "UNKNOWN"]}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Tracks() {
  const scene = useStore((s) => s.scene)!;
  const sel = useStore((s) => s.selection);
  const select = useStore((s) => s.select);
  const setTime = useStore((s) => s.setTime);
  const [q, setQ] = useState("");
  const list = scene.agents.filter((a) => !q || a.id.toLowerCase().includes(q.toLowerCase()) || a.category.includes(q.toLowerCase()));
  return (
    <div className="col">
      <input placeholder="Filter (id or category)" value={q} onChange={(e) => setQ(e.target.value)} />
      <table className="list">
        <thead><tr><th>Track</th><th>Observed</th><th></th></tr></thead>
        <tbody>
          {list.map((a) => (
            <tr key={a.id} className={sel?.kind === "agent" && sel.id === a.id ? "sel" : ""} onClick={() => { select({ kind: "agent", id: a.id }); setTime(a.first_t); }}>
              <td><span className="swatch" style={{ background: scene.categories[a.category]?.color }} /> <b>{a.id}</b><div className="muted small">{a.category}{a.stationary ? " · stationary" : ""}</div></td>
              <td className="small">{fmtTime(a.first_t)}–{fmtTime(a.last_t)}<div className="muted">{a.observed_count} obs · {a.gaps.length} gaps</div></td>
              <td className="small">{a.faces.length ? `${a.faces.length} faces` : ""}{a.links.length ? <div className="badge hyp">re-linked</div> : null}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Events() {
  const scene = useStore((s) => s.scene)!;
  const select = useStore((s) => s.select);
  const setTime = useStore((s) => s.setTime);
  const sel = useStore((s) => s.selection);
  const types = useMemo(() => Array.from(new Set(scene.events.map((e) => e.type))), [scene]);
  const [hidden, setHidden] = useState<Record<string, boolean>>({ observation_gap: true });
  const list = scene.events.filter((e) => !hidden[e.type]);
  return (
    <div className="col">
      <div className="row wrap">{types.map((t) => <button key={t} className={hidden[t] ? "" : "active"} onClick={() => setHidden({ ...hidden, [t]: !hidden[t] })}>{t.replace("_", " ")}</button>)}</div>
      <div className="muted small">{list.length} of {scene.events.length} events · all machine-generated observations</div>
      <table className="list">
        <tbody>
          {list.map((e) => (
            <tr key={e.id} className={sel?.kind === "event" && sel.id === e.id ? "sel" : ""} onClick={() => { select({ kind: "event", id: e.id }); setTime(e.t); }}>
              <td className="mono small">{fmtTime(e.t)}</td>
              <td className="small">{e.description}{(e.hypothesis || e.requires_verification) && <span className="badge hyp" style={{ marginLeft: 4 }}>verify</span>}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Measure() {
  const caseId = useStore((s) => s.caseId)!;
  const st = useStore();
  const scene = st.scene!;
  const [list, setList] = useState<any[]>([]);
  const [refs, setRefs] = useState<any[]>([]);
  useEffect(() => {
    api.measurements(caseId).then(setList).catch(() => {});
    api.annotations(caseId, "calibration").then((c) => setRefs(c?.references ?? [])).catch(() => {});
  }, [caseId, st.measurementsVersion, scene]);
  const cs = scene.coordinate_system;
  return (
    <div className="col">
      <div className="callout info">Click points on the reconstruction. Every value carries its unit, calibration status and uncertainty. Keys: <b>M</b> distance, <b>H</b> height, <b>K</b> calibrate, <b>Esc</b> stop.</div>
      <div className="row wrap">
        {(["measure", "height", "angle", "calibrate"] as const).map((t) => (
          <button key={t} className={st.tool === t ? "active" : ""} onClick={() => st.setTool(st.tool === t ? "navigate" : t)}>{t === "measure" ? "distance" : t}</button>
        ))}
      </div>
      {cs.status !== "CALIBRATED_METRIC" && <div className="callout">Scale status {cs.status}: use <b>calibrate</b> on a known real distance (e.g. a parking bay, a door width) to obtain calibrated metres.</div>}
      <h4>Measurements</h4>
      {list.filter((m) => m.active !== false).length === 0 && <div className="muted small">None yet.</div>}
      {list.filter((m) => m.active !== false).map((m) => (
        <div key={m.id} className="card" style={{ padding: 8 }}>
          <div className="row"><b className="mono">{m.id}</b><span className="muted small grow">{m.kind}{m.label ? ` · ${m.label}` : ""}</span>
            <button className="icon danger" title="Withdraw (kept in the audit log)" onClick={() => api.withdraw(caseId, m.id).then(st.bumpMeasurements)}>✕</button></div>
          <div>{m.display}</div>
          {m.stale && <div className="callout small">{m.stale}</div>}
          <div className="muted small">{m.actor} · {m.created?.slice(0, 19).replace("T", " ")}</div>
        </div>
      ))}
      <h4>Scale references</h4>
      {refs.length === 0 && <div className="muted small">No analyst references.</div>}
      {refs.map((r) => (
        <div key={r.id} className="row small">
          <span className="mono">{r.id}</span><span className="grow">{r.label || "reference"}: {r.true_length_m} m ± {r.sigma_m}</span>
          {r.active ? <button className="icon danger" onClick={() => api.calibOff(caseId, r.id).then(() => st.notify("Reference deactivated; recomputing", "ok"))}>off</button> : <span className="muted">inactive</span>}
        </div>
      ))}
    </div>
  );
}

function CaseTab({ onReloadScene }: { onReloadScene: () => void }) {
  const detail = useStore((s) => s.detail);
  const scene = useStore((s) => s.scene);
  const notify = useStore((s) => s.notify);
  const [audit, setAudit] = useState<any>(null);
  const [verify, setVerify] = useState<any>(null);
  useEffect(() => { if (detail) api.audit(detail.id).then(setAudit).catch(() => {}); }, [detail?.id]);
  if (!detail) return null;
  return (
    <div className="col">
      <div className="kv">
        <span>Case ID</span><span className="mono">{detail.id}</span>
        <span>Folder</span><span className="mono" style={{ wordBreak: "break-all" }}>{detail.path}</span>
        <span>Created</span><span>{detail.created}</span>
      </div>
      <h4>Pipeline stages</h4>
      {detail.stages.map((s) => (
        <div key={s.name} className="stage">
          <span className="small">{s.title}</span><span className={`badge ${s.status}`}>{s.status}</span>
          {(s.warnings.length > 0 || s.fallbacks.length > 0) && <span className="muted small" style={{ gridColumn: "1 / -1" }}>{[...s.warnings.slice(0, 3), ...s.fallbacks.map((f: any) => `Fallback: ${f.component} → ${f.used}`)].join(" · ")}</span>}
        </div>
      ))}
      <h4>Integrity</h4>
      <button onClick={() => api.verifyEvidence(detail.id).then(setVerify).catch((e) => notify(e.message, "error"))}>Re-verify evidence hashes</button>
      {verify && verify.evidence.map((r: any) => (
        <div key={r.id} className="small">{r.id}: <b style={{ color: r.ok ? "var(--ok)" : "var(--bad)" }}>{r.ok ? "matches intake hash" : "MISMATCH"}</b></div>
      ))}
      {audit && <div className="small">Audit log: {audit.chain.entries} entries · chain <b style={{ color: audit.chain.valid ? "var(--ok)" : "var(--bad)" }}>{audit.chain.valid ? "valid" : "BROKEN"}</b><div className="mono muted" style={{ wordBreak: "break-all" }}>head {audit.chain.head}</div></div>}
      <div className="muted small">A matching hash proves the file is unchanged since intake; it does not by itself establish how, when or by whom the recording was made.</div>
      {audit && (
        <details>
          <summary className="small">Audit entries</summary>
          {audit.entries.slice().reverse().map((e: any) => (
            <div key={e.seq} className="small" style={{ borderBottom: "1px solid var(--line)", padding: "3px 0" }}>
              <span className="mono muted">#{e.seq} {e.ts.slice(0, 19).replace("T", " ")}</span> <b>{e.action}</b> <span className="muted">{e.actor}{e.target ? ` → ${e.target}` : ""}</span>
            </div>
          ))}
        </details>
      )}
      {scene?.grouping && scene.videos.length > 1 && (
        <>
          <h4>Video grouping (suggestion)</h4>
          <div className="small">Groups: {scene.grouping.groups.map((g: string[]) => `[${g.join(", ")}]`).join(" ")}</div>
          {scene.grouping.pairs.map((p: any) => (
            <div key={p.a + p.b} className="small" style={{ borderBottom: "1px solid var(--line)", padding: "3px 0" }}>
              <b>{p.a} / {p.b}</b>: {p.suggestion}
              {p.strong_evidence.length > 0 && <div>{p.strong_evidence.join("; ")}</div>}
              {p.weak_context.length > 0 && <div className="muted">context (not used): {p.weak_context.join("; ")}</div>}
            </div>
          ))}
          <div className="muted small">{scene.grouping.note}</div>
        </>
      )}
      <Notes caseId={detail.id} />
      {scene && scene.warnings.length > 0 && (
        <>
          <h4>Warnings ({scene.warnings.length})</h4>
          {scene.warnings.map((w, i) => <div key={i} className="callout small">{w.source}: {w.message}</div>)}
        </>
      )}
      <button onClick={onReloadScene}>Reload scene</button>
    </div>
  );
}

function Notes({ caseId }: { caseId: string }) {
  const notify = useStore((s) => s.notify);
  const [list, setList] = useState<any[]>([]);
  const [text, setText] = useState("");
  const load = () => api.annotations(caseId, "notes").then((l) => setList(Array.isArray(l) ? l : [])).catch(() => {});
  useEffect(() => { load(); }, [caseId]);
  const add = () => {
    const sel = useStore.getState().selection;
    api.note(caseId, text.trim(), sel ? sel.id : undefined)
      .then(() => { setText(""); load(); notify("Note saved (audit log)", "ok"); })
      .catch((e) => notify(e.message, "error"));
  };
  return (
    <>
      <h4>Analyst notes</h4>
      <textarea rows={3} placeholder="Note (attached to the current selection, if any)" value={text} onChange={(e) => setText(e.target.value)} />
      <button disabled={!text.trim()} onClick={add}>Add note</button>
      {list.slice().reverse().map((n) => (
        <div key={n.id} className="small" style={{ borderBottom: "1px solid var(--line)", padding: "3px 0" }}>
          <span className="mono muted">{n.id} · {n.actor} · {String(n.created).slice(0, 16).replace("T", " ")}{n.target ? ` · ${n.target}` : ""}</span>
          <div>{n.text}</div>
        </div>
      ))}
    </>
  );
}
