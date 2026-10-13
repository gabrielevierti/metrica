import { useState } from "react";
import { api, fileUrl, sceneFileUrl } from "../lib/api";
import { fmt, fmtTime, useStore } from "../lib/store";
import { EVIDENCE_COLORS, EVIDENCE_SHORT, type Agent, type Entity, type SceneEvent } from "../lib/types";

export default function Inspector() {
  const scene = useStore((s) => s.scene);
  const sel = useStore((s) => s.selection);
  return (
    <div className="right">
      <div className="tabs"><button className="active">Inspector</button></div>
      <div className="panelbody">
        {!scene && <div className="muted">Nothing to inspect yet.</div>}
        {scene && !sel && <SceneInfo />}
        {scene && sel?.kind === "entity" && <EntityView e={scene.entities.find((x) => x.id === sel.id)!} />}
        {scene && sel?.kind === "agent" && <AgentView a={scene.agents.find((x) => x.id === sel.id)!} />}
        {scene && sel?.kind === "event" && <EventView e={scene.events.find((x) => x.id === sel.id)!} />}
        {scene && sel?.kind === "camera" && <CameraView id={sel.id} />}
      </div>
    </div>
  );
}

function SceneInfo() {
  const scene = useStore((s) => s.scene)!;
  const cs = scene.coordinate_system;
  return (
    <div className="col">
      <h3>Scene</h3>
      <div className="kv">
        <span>Scale</span><span><span className={`badge ${cs.status}`}>{cs.status}</span></span>
        <span>Meaning</span><span>{cs.status_text}</span>
        <span>Method</span><span>{cs.scale.method}</span>
        <span>Uncertainty</span><span>{cs.scale.scale_sigma_rel ? `±${(cs.scale.scale_sigma_rel * 100).toFixed(1)}% (1σ)` : "not quantified"}</span>
        <span>Unit</span><span>{cs.unit}</span>
        <span>Objects</span><span>{scene.entities.length}</span>
        <span>Tracks</span><span>{scene.agents.length}</span>
        <span>Events</span><span>{scene.events.length}</span>
        <span>Cameras</span><span>{scene.cameras.length}</span>
      </div>
      <h4>Assumptions</h4>
      <ul className="small" style={{ paddingLeft: 16, margin: 0 }}>{cs.scale.assumptions.map((a, i) => <li key={i}>{a}</li>)}</ul>
      <h4>Evidence classes</h4>
      {Object.entries(scene.evidence_classes).map(([k, v]) => (
        <div key={k} className="small row" style={{ alignItems: "flex-start" }}><span className="swatch" style={{ background: EVIDENCE_COLORS[k], marginTop: 3 }} /><span><b>{EVIDENCE_SHORT[k]}</b> — {v}</span></div>
      ))}
      <div className="muted small">Select an object, track, event or camera to inspect its evidence and provenance.</div>
    </div>
  );
}

function Verify({ target }: { target: string }) {
  const caseId = useStore((s) => s.caseId)!;
  const notify = useStore((s) => s.notify);
  const [note, setNote] = useState("");
  const send = (verdict: string) => api.verify(caseId, target, verdict, note).then(() => { notify(`${target}: ${verdict} (recorded in audit log)`, "ok"); setNote(""); }).catch((e) => notify(e.message, "error"));
  return (
    <div className="col card" style={{ padding: 8 }}>
      <div className="small muted">Analyst review (recorded with your name in the audit log)</div>
      <textarea rows={2} placeholder="Note / basis of the verification" value={note} onChange={(e) => setNote(e.target.value)} />
      <div className="row">
        <button onClick={() => send("verified")}>Verified</button>
        <button onClick={() => send("uncertain")}>Uncertain</button>
        <button className="danger" onClick={() => send("rejected")}>Rejected</button>
      </div>
    </div>
  );
}

function Dims({ dims }: { dims?: Record<string, any> }) {
  if (!dims) return null;
  return (
    <table className="list"><tbody>
      {Object.entries(dims).map(([k, d]) => (
        <tr key={k} title={d.note}>
          <td className="muted">{k.replace(/_/g, " ")}</td>
          <td>{typeof d.value === "number" ? <>{fmt(d.value)}{d.sigma != null ? ` ± ${fmt(d.sigma)}` : ""} {d.unit ?? ""}</> : "—"}</td>
          <td><span className={`badge ${d.basis === "measured" ? "verified" : d.basis === "prior" ? "hyp" : ""}`}>{d.basis}</span></td>
        </tr>
      ))}
    </tbody></table>
  );
}

function EntityView({ e }: { e: Entity }) {
  const caseId = useStore((s) => s.caseId)!;
  const openFrame = useStore((s) => s.openFrame);
  const requestView = useStore((s) => s.requestView);
  const select = useStore((s) => s.select);
  if (!e) return null;
  const ev = e.evidence?.class ?? "UNKNOWN";
  return (
    <div className="col">
      <div className="row"><h3 className="grow">{e.display ?? e.category}</h3><button className="icon" onClick={() => requestView("focus")}>focus</button></div>
      <div className="mono muted">{e.id}</div>
      <div className="row wrap">
        <span className="badge" style={{ borderColor: EVIDENCE_COLORS[ev], color: EVIDENCE_COLORS[ev] }}>{EVIDENCE_SHORT[ev]}</span>
        {e.evidence?.confidence != null && <span className="chip">confidence {fmt(e.evidence.confidence)}</span>}
        {e.evidence?.views && <span className="chip">{e.evidence.views.length} view(s)</span>}
        <span className="badge machine">machine-generated</span>
      </div>
      {e.shape_note && <div className="callout">{e.shape_note}</div>}
      {e.note && <div className="callout info">{e.note}</div>}
      <h4>Dimensions</h4>
      <Dims dims={e.dimensions} />
      {e.texture && <img src={sceneFileUrl(caseId, e.texture.file)} style={{ width: "100%", background: "repeating-conic-gradient(#222 0 25%, #333 0 50%) 50% / 16px 16px", borderRadius: 4 }} title="Rectified observed texture (transparent = not observed)" />}
      {e.openings && e.openings.length > 0 && <div className="small">{e.openings.length} candidate opening(s) (window/door) — appearance-based, not verified.</div>}
      {(e.inferred_patches?.length || e.inferred_parts?.length) ? (
        <div className="callout small">Inferred parts: {[...(e.inferred_patches ?? []).map((p: any) => p.reason), ...(e.inferred_parts ?? []).map((p: any) => `${p.part}: ${p.reason}`)].join(" · ")}</div>
      ) : null}
      {e.flags && e.flags.length > 0 && <div className="callout small">{e.flags.join(" · ")}</div>}
      {e.regions && Object.keys(e.regions).length > 0 && (
        <div className="small">{Object.entries(e.regions).map(([k, r]: any) => <div key={k}>{k}: {fmt(r.area, 1)} m² ({r.use})</div>)}</div>
      )}
      <h4>How this was reconstructed</h4>
      <div className="small">{e.provenance?.method}</div>
      <div className="muted small mono">{e.provenance?.algorithm}</div>
      {e.relations && e.relations.length > 0 && (
        <>
          <h4>Relations</h4>
          {e.relations.map((r: any, i: number) => <div key={i} className="small">{r.type.replace(/_/g, " ")} <a href="#" onClick={(ev) => { ev.preventDefault(); select({ kind: "entity", id: r.with }); }}>{r.with}</a>{r.distance != null ? ` (${fmt(r.distance)} ${r.unit})` : ""}{r.note ? <span className="muted"> — {r.note}</span> : null}</div>)}
        </>
      )}
      {e.tracks && e.tracks.length > 0 && <div className="small">Linked tracks: {e.tracks.map((t) => <a key={t} href="#" onClick={(ev) => { ev.preventDefault(); select({ kind: "agent", id: t }); }}>{t} </a>)}</div>}
      {e.source_observations && e.source_observations.length > 0 && (
        <>
          <h4>Source observations</h4>
          {e.source_observations.map((o: any, i: number) => (
            <div key={i} className="row small">
              <span className="mono grow">{o.video} {o.frame != null ? `frame ${o.frame}` : "(background plate)"}{o.detector_label ? ` · ${o.detector_label} ${fmt(o.detector_conf)}` : ""}</span>
              {o.frame != null && <button className="icon" onClick={() => openFrame({ video: o.video, frame: o.frame, bbox: o.bbox_px, title: e.id })}>frame</button>}
            </div>
          ))}
        </>
      )}
      <Verify target={e.id} />
    </div>
  );
}

function AgentView({ a }: { a: Agent }) {
  const caseId = useStore((s) => s.caseId)!;
  const scene = useStore((s) => s.scene)!;
  const openFrame = useStore((s) => s.openFrame);
  const setTime = useStore((s) => s.setTime);
  const requestView = useStore((s) => s.requestView);
  if (!a) return null;
  const unit = scene.coordinate_system.unit;
  const rel = ["RELATIVE", "ARBITRARY"].includes(scene.coordinate_system.status);
  const off = scene.time.offsets[a.video] ?? 0;
  return (
    <div className="col">
      <div className="row"><h3 className="grow">{a.id}</h3><button className="icon" onClick={() => requestView("focus")}>focus</button></div>
      <div className="row wrap"><span className="chip">{a.category}</span><span className="chip">{a.video}</span><span className="badge machine">machine-generated</span>{a.stationary && <span className="chip">stationary</span>}</div>
      <div className="kv">
        <span>Observed</span><span><a href="#" onClick={(e) => { e.preventDefault(); setTime(a.first_t); }}>{fmtTime(a.first_t)}</a> – <a href="#" onClick={(e) => { e.preventDefault(); setTime(a.last_t); }}>{fmtTime(a.last_t)}</a></span>
        <span>Observations</span><span>{a.observed_count}</span>
        {a.summary && <>
          <span>Observed path</span><span>{fmt(a.summary.path_length_observed, 1)} {unit}</span>
          <span>Displacement</span><span>{fmt(a.summary.displacement, 1)} {unit}</span>
          {a.summary.speed_median != null ? <>
            <span>Speed (median)</span><span>{fmt(a.summary.speed_median)} {unit}/s{rel ? " (relative)" : ""}</span>
            <span>Speed (max)</span><span>{fmt(a.summary.speed_max)} {unit}/s{a.flags.some((f: any) => f.type === "implausible_speed") ? " — implausible, see flags" : ""}</span>
          </> : <><span>Speed</span><span className="muted">{(a as any).speed_note ? "not reported (scale not metric)" : "—"}</span></>}
          <span>Position σ</span><span>{fmt(a.summary.position_sigma_median)} {unit}</span>
        </>}
        {a.stature && <><span>Stature (est.)</span><span>{fmt(a.stature.median)} {unit} (IQR {fmt(a.stature.iqr)}, n={a.stature.n})</span></>}
      </div>
      {a.summary?.path_note && <div className="muted small">{a.summary.path_note}</div>}
      {a.representation?.note && <div className="callout small">{a.representation.note}</div>}
      {a.color && <div className="small row"><span className="swatch" style={{ background: a.color.hex ?? `rgb(${a.color.rgb})` }} /> Apparent colour <b>{a.color.name}</b> — <span className={a.color.reliable ? "" : "muted"}>{a.color.reliable ? "consistent across crops" : "not reliable"}</span></div>}
      {a.ocr_candidates && a.ocr_candidates.length > 0 && (
        <>
          <h4>Text candidates (verify)</h4>
          {a.ocr_candidates.map((o: any) => <div key={o.text} className="row small"><span className="mono grow">{o.text}</span><span>{fmt(o.confidence)}</span><span className="badge hyp">verify</span></div>)}
        </>
      )}
      {a.gaps.length > 0 && (
        <details><summary className="small">{a.gaps.length} observation gap(s)</summary>
          {a.gaps.map((g: any, i: number) => <div key={i} className="small">{fmtTime(g.t0)}–{fmtTime(g.t1)}: {g.reason}</div>)}
        </details>
      )}
      {a.links.length > 0 && <div className="callout small">Re-linked fragments (hypothesis): {a.links.map((l: any) => `${l.from_fragment}→${l.to_fragment} (gap ${l.gap_s}s, score ${l.score})`).join("; ")}</div>}
      {a.flags.length > 0 && (
        <details><summary className="small">{a.flags.length} tracking flag(s)</summary>
          {a.flags.map((f: any, i: number) => <div key={i} className="small">{f.t != null ? `${fmtTime(f.t + off)} — ` : ""}{f.note}</div>)}
        </details>
      )}
      {a.crops.length > 0 && (
        <>
          <h4>Best observations</h4>
          <div className="crops">
            {a.crops.map((c: any) => (
              <figure key={c.file} onClick={() => openFrame({ video: a.video, frame: c.frame, bbox: c.bbox, title: a.id })}>
                <img src={fileUrl(caseId, c.file)} />
                <figcaption>f{c.frame} · {fmtTime(c.t + off)}{c.mask ? " · mask" : ""}</figcaption>
              </figure>
            ))}
          </div>
        </>
      )}
      {a.faces.length > 0 && (
        <>
          <h4>Face observations</h4>
          <div className="callout small">Unmodified original pixels. Detection only — METRICA does not identify people.</div>
          <div className="crops">
            {a.faces.map((f: any) => (
              <figure key={f.file} title={`detection ${fmt(f.detection_score)}, sharpness ${fmt(f.sharpness, 0)}, occlusion ${f.occlusion}`}>
                <a href={fileUrl(caseId, f.file)} download><img src={fileUrl(caseId, f.file)} style={{ height: 90 }} /></a>
                <figcaption onClick={() => openFrame({ video: a.video, frame: f.frame, bbox: f.bbox, title: `${a.id} face` })}>f{f.frame} · {fmtTime(f.t + off)}<br />{Math.round(f.face_height_px)} px · {f.pose_class}</figcaption>
              </figure>
            ))}
          </div>
        </>
      )}
      <Verify target={a.id} />
    </div>
  );
}

function EventView({ e }: { e: SceneEvent }) {
  const select = useStore((s) => s.select);
  const openFrame = useStore((s) => s.openFrame);
  if (!e) return null;
  return (
    <div className="col">
      <h3>{e.type.replace(/_/g, " ")}</h3>
      <div className="mono muted">{e.id}</div>
      <div>{e.description}</div>
      <div className="row wrap"><span className="badge machine">machine-generated</span>{(e.hypothesis || e.requires_verification) && <span className="badge hyp">hypothesis — verify</span>}<span className="chip">confidence {fmt(e.confidence)}</span></div>
      <div className="kv">
        <span>Case time</span><span>{fmtTime(e.t)}{e.t_end ? ` – ${fmtTime(e.t_end)}` : ""}</span>
        <span>Source</span><span>{e.video}{e.frame != null ? ` frame ${e.frame}` : ""}</span>
        <span>Tracks</span><span>{e.tracks?.map((t) => <a key={t} href="#" onClick={(ev) => { ev.preventDefault(); select({ kind: "agent", id: t }); }}>{t} </a>)}</span>
      </div>
      {e.frame != null && e.video && <button onClick={() => openFrame({ video: e.video!, frame: e.frame!, title: e.id })}>Open source frame</button>}
      <Verify target={e.id} />
    </div>
  );
}

function CameraView({ id }: { id: string }) {
  const scene = useStore((s) => s.scene)!;
  const requestView = useStore((s) => s.requestView);
  const c = scene.cameras.find((x) => x.id === id);
  if (!c) return null;
  return (
    <div className="col">
      <h3>{c.id}</h3>
      <div className="kv">
        <span>Video</span><span>{c.video}</span>
        <span>Kind</span><span>{c.kind} ({c.mode})</span>
        <span>Resolution</span><span>{c.intrinsics.width}×{c.intrinsics.height}</span>
        <span>Field of view</span><span>{fmt(c.intrinsics.hfov_deg, 1)}° horizontal</span>
        <span>Focal (px)</span><span>{fmt(c.intrinsics.fx, 1)}</span>
        {c.poses && <><span>Poses</span><span>{c.poses.frames.length} ({["keyframe", "localized", "interpolated"].map((k) => `${c.poses!.source.filter((s) => s === k).length} ${k}`).join(", ")})</span></>}
      </div>
      <button onClick={() => requestView("camera", c.id)}>View from this camera</button>
    </div>
  );
}
