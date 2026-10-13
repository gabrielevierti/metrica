// The 3D viewer: reconstruction layers, semantic objects, tracked agents, cameras and measurement tools.
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { Canvas, useFrame, useThree, type ThreeEvent } from "@react-three/fiber";
import { Bvh, GizmoHelper, GizmoViewport, Grid, OrbitControls } from "@react-three/drei";
import { api } from "../lib/api";
import { fmt, fmtTime, useStore, type Tool } from "../lib/store";
import { EVIDENCE, EVIDENCE_COLORS, EVIDENCE_SHORT, type Entity, type Scene, type Vec3 } from "../lib/types";
import { CoverageLayer, MeshLayer } from "./Layers";
import Entities from "./Entities";
import Agents from "./Agents";
import Cameras from "./Cameras";
import MeasureOverlay from "./Measure";
import { agentFocus, cameraCenter, cameraPoseAt, entityBox, glCameraMatrix, sceneBounds } from "./util";

const TOOL_POINTS: Partial<Record<Tool, number>> = { measure: 2, height: 1, angle: 3, calibrate: 2, inspect: 1 };
const TOOL_HINT: Partial<Record<Tool, string>> = {
  measure: "Click two points on the reconstruction or on objects.",
  height: "Click one point: height above the ground plane.",
  angle: "Click three points; the middle one is the vertex.",
  calibrate: "Click both ends of an object whose real length you know.",
  inspect: "Click a point to read its coordinates, evidence class and uncertainty.",
};

/** Points of interest used to frame default views (robust to far-away depth such as sky/horizon). */
function interestBox(scene: Scene): THREE.Box3 {
  // where the action is: subject positions (weighted), object centres and cameras; 5th-95th percentile
  const pts: THREE.Vector3[] = [];
  for (const e of scene.entities) {
    if (e.category === "ground") continue;
    const eb = entityBox(e);
    if (eb) pts.push(eb.getCenter(new THREE.Vector3()));
  }
  for (const a of scene.agents) {
    const step = Math.max(1, Math.floor(a.samples.length / 30));
    for (let i = 0; i < a.samples.length; i += step) pts.push(new THREE.Vector3(...a.samples[i].p));
  }
  for (const c of scene.cameras) pts.push(cameraCenter(c.world_from_cam));
  if (pts.length < 3) {
    const sb = sceneBounds(scene);
    return new THREE.Box3(sb.min, sb.max);
  }
  const q = (vals: number[], f: number) => { const s = [...vals].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.max(0, Math.round(f * (s.length - 1))))]; };
  const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y), zs = pts.map((p) => p.z);
  const b = new THREE.Box3(new THREE.Vector3(q(xs, 0.05), Math.min(0, q(ys, 0.05)), q(zs, 0.05)), new THREE.Vector3(q(xs, 0.95), Math.max(2, q(ys, 0.95)), q(zs, 0.95)));
  for (const c of scene.cameras) b.expandByPoint(cameraCenter(c.world_from_cam));
  return b;
}

export default function Viewport() {
  const scene = useStore((s) => s.scene)!;
  const tool = useStore((s) => s.tool);
  const navMode = useStore((s) => s.navMode);
  const showGrid = useStore((s) => s.show.grid);
  const layerVisible = useStore((s) => s.layerVisible);
  const picked = useStore((s) => s.picked);
  const st = useStore.getState;
  const glRef = useRef<THREE.WebGLRenderer | null>(null);
  const bounds = useMemo(() => sceneBounds(scene), [scene]);
  const ibox = useMemo(() => interestBox(scene), [scene]);
  const isize = Math.max(2, ibox.getSize(new THREE.Vector3()).length());
  const markerSize = Math.min(1.2, Math.max(0.15, isize / 60));
  const [measureKind, setMeasureKind] = useState<"distance" | "horizontal_distance" | "height_difference">("distance");
  const [result, setResult] = useState<any>(null);
  const [calib, setCalib] = useState<{ p1: Vec3; p2: Vec3 } | null>(null);
  const [snap, setSnap] = useState<Blob | null>(null);

  // --- picking ---------------------------------------------------------------------------
  const pickPoint = (e: ThreeEvent<MouseEvent>, evidence: string | null, source: string) => {
    if (e.delta > 3) return; // a drag, not a click
    const s = st();
    if (s.tool === "navigate") return;
    const p = e.point.toArray().map((v) => Math.round(v * 1e4) / 1e4) as Vec3;
    s.pick({ p, evidence, source });
  };
  const onLayerPick = (e: ThreeEvent<MouseEvent>, evidence: string | null) => pickPoint(e, evidence, "surface");
  const onEntityPick = (e: ThreeEvent<MouseEvent>, ent: Entity) => {
    if (st().tool !== "navigate") pickPoint(e, ent.evidence?.class ?? null, ent.id);
    else if (e.delta <= 3) st().select({ kind: "entity", id: ent.id });
  };

  // --- tool completion -------------------------------------------------------------------
  useEffect(() => {
    const need = TOOL_POINTS[tool];
    if (!need || picked.length < need) return;
    const s = st();
    const caseId = s.caseId!;
    const pts = picked.slice(0, need);
    if (tool === "calibrate") {
      setCalib({ p1: pts[0].p, p2: pts[1].p });
      s.clearPicked();
      return;
    }
    const kind = tool === "measure" ? measureKind : tool === "height" ? "height_above_ground" : tool === "angle" ? "angle" : "point";
    api.measure(caseId, { kind, points: pts.map((x) => x.p), evidence: pts.map((x) => x.evidence), save: tool !== "inspect" })
      .then((r) => { setResult({ ...r, picked: pts }); if (tool !== "inspect") s.bumpMeasurements(); })
      .catch((e) => s.notify(e.message, "error"))
      .finally(() => s.clearPicked());
  }, [picked, tool, measureKind]);
  useEffect(() => setResult(null), [tool]);

  const snapshot = () => {
    const c = glRef.current?.domElement;
    if (!c) return;
    c.toBlob((b) => b && setSnap(b), "image/png");
  };

  return (
    <div style={{ position: "absolute", inset: 0 }}>
      <Canvas
        gl={{ preserveDrawingBuffer: true, antialias: true }}
        dpr={[1, 2]}
        camera={{ fov: 50, near: Math.max(0.01, bounds.radius * 2e-4), far: bounds.radius * 40, position: [0, 10, 10] }}
        raycaster={{ params: { Points: { threshold: markerSize * 0.25 }, Line: { threshold: markerSize * 0.2 } } as any }}
        onCreated={({ gl }) => { glRef.current = gl; }}
        onPointerMissed={(e) => { if (st().tool === "navigate" && (e as MouseEvent).button === 0) st().select(null); }}
        style={{ cursor: tool === "navigate" ? "grab" : "crosshair" }}
      >
        <color attach="background" args={["#0a0e13"]} />
        <hemisphereLight args={["#dfe9f5", "#3a3328", 0.9]} />
        <directionalLight position={[30, 60, 20]} intensity={1.1} />
        <Bvh firstHitOnly>
          {scene.layers.filter((l) => layerVisible[l.id] && l.kind !== "coverage").map((l) => <MeshLayer key={l.id} layer={l} onPick={onLayerPick} />)}
          <Entities onPick={onEntityPick} />
        </Bvh>
        {scene.layers.filter((l) => layerVisible[l.id] && l.kind === "coverage").map((l) => <CoverageLayer key={l.id} layer={l} />)}
        <Agents onSelect={(id) => st().select({ kind: "agent", id })} />
        <Cameras size={markerSize * 2} onSelect={(id) => st().select({ kind: "camera", id })} />
        <MeasureOverlay size={markerSize} />
        {showGrid && (
          <Grid position={[0, -0.002, 0]} args={[10, 10]} cellSize={1} sectionSize={10} infiniteGrid fadeDistance={Math.max(60, isize * 2)}
            cellColor="#1e2a36" sectionColor="#2f4256" cellThickness={0.6} sectionThickness={1} />
        )}
        {navMode === "orbit" ? <Orbit /> : <Fly speed={Math.max(2, isize / 12)} />}
        <ViewController ibox={ibox} />
        <GizmoHelper alignment="bottom-right" margin={[70, 120]}>
          <GizmoViewport axisColors={["#e4572e", "#3ccf7a", "#4cc2ff"]} labelColor="#0a0e13" />
        </GizmoHelper>
      </Canvas>
      <Toolbar onSnapshot={snapshot} />
      <Legend />
      <StatusBar />
      {(tool !== "navigate" || result) && (
        <div className="hint">
          {tool !== "navigate" && (
            <div className="row">
              <b>{tool === "measure" ? "Distance" : tool[0].toUpperCase() + tool.slice(1)}</b>
              <span>{TOOL_HINT[tool]}</span>
              {tool === "measure" && (
                <select value={measureKind} onChange={(e) => setMeasureKind(e.target.value as any)}>
                  <option value="distance">3D distance</option>
                  <option value="horizontal_distance">horizontal distance</option>
                  <option value="height_difference">height difference</option>
                </select>
              )}
              <span className="muted">{picked.length}/{TOOL_POINTS[tool]}</span>
              <button className="icon" onClick={() => st().setTool("navigate")}>done</button>
            </div>
          )}
          {result && (
            <div style={{ marginTop: 4 }}>
              {result.kind === "point" ? (
                <span className="mono">({result.details?.coordinates?.map((v: number) => fmt(v, 3)).join(", ")}) {result.unit} · ±{fmt(result.sigma, 3)} · evidence {EVIDENCE_SHORT[result.picked?.[0]?.evidence] ?? "n/a"}</span>
              ) : (
                <span><b>{result.id ? `${result.id}: ` : ""}</b>{result.display}</span>
              )}
              {result.status && result.status !== "CALIBRATED_METRIC" && <span className="muted small"> · scale {result.status}</span>}
              <button className="icon" style={{ marginLeft: 8 }} onClick={() => setResult(null)}>✕</button>
            </div>
          )}
        </div>
      )}
      {calib && <CalibrateDialog p1={calib.p1} p2={calib.p2} unit={scene.coordinate_system.unit} onClose={() => setCalib(null)} />}
      {snap && <SnapshotDialog blob={snap} onClose={() => setSnap(null)} />}
    </div>
  );
}

// ------------------------------------------------------------------------------- navigation
function Orbit() {
  const camera = useThree((s) => s.camera);
  const target = useMemo(() => {
    const dir = camera.getWorldDirection(new THREE.Vector3());
    return camera.position.clone().addScaledVector(dir, 10);
  }, []);
  return <OrbitControls makeDefault target={target} enableDamping dampingFactor={0.12} screenSpacePanning maxPolarAngle={Math.PI * 0.98} />;
}

/** First-person navigation: drag to look, WASD to move, Q/E down/up, Shift faster. */
function Fly({ speed }: { speed: number }) {
  const { camera, gl } = useThree();
  const keys = useRef<Record<string, boolean>>({});
  useEffect(() => {
    const el = gl.domElement;
    const euler = new THREE.Euler(0, 0, 0, "YXZ");
    let drag = false, lx = 0, ly = 0;
    const editable = (t: EventTarget | null) => !!(t as HTMLElement)?.tagName?.match(/INPUT|TEXTAREA|SELECT/);
    const kd = (e: KeyboardEvent) => { if (!editable(e.target)) keys.current[e.code] = true; };
    const ku = (e: KeyboardEvent) => { keys.current[e.code] = false; };
    const pd = (e: PointerEvent) => { drag = true; lx = e.clientX; ly = e.clientY; };
    const pm = (e: PointerEvent) => {
      if (!drag) return;
      euler.setFromQuaternion(camera.quaternion);
      euler.y -= (e.clientX - lx) * 0.0035;
      euler.x = Math.max(-1.5, Math.min(1.5, euler.x - (e.clientY - ly) * 0.0035));
      euler.z = 0;
      camera.quaternion.setFromEuler(euler);
      lx = e.clientX; ly = e.clientY;
    };
    const pu = () => { drag = false; };
    window.addEventListener("keydown", kd);
    window.addEventListener("keyup", ku);
    el.addEventListener("pointerdown", pd);
    window.addEventListener("pointermove", pm);
    window.addEventListener("pointerup", pu);
    return () => {
      window.removeEventListener("keydown", kd);
      window.removeEventListener("keyup", ku);
      el.removeEventListener("pointerdown", pd);
      window.removeEventListener("pointermove", pm);
      window.removeEventListener("pointerup", pu);
    };
  }, [camera, gl]);
  useFrame((_, dt) => {
    const k = keys.current;
    const v = speed * (k.ShiftLeft || k.ShiftRight ? 4 : 1) * Math.min(dt, 0.1);
    const fwd = camera.getWorldDirection(new THREE.Vector3());
    const right = new THREE.Vector3().crossVectors(fwd, camera.up).normalize();
    if (k.KeyW) camera.position.addScaledVector(fwd, v);
    if (k.KeyS) camera.position.addScaledVector(fwd, -v);
    if (k.KeyD) camera.position.addScaledVector(right, v);
    if (k.KeyA) camera.position.addScaledVector(right, -v);
    if (k.KeyE) camera.position.y += v;
    if (k.KeyQ) camera.position.y -= v;
  });
  return null;
}

/** Executes view requests (top / overview / camera / focus) with a short animated transition. */
function ViewController({ ibox }: { ibox: THREE.Box3 }) {
  const { camera, controls } = useThree() as any;
  const req = useStore((s) => s.viewRequest);
  const scene = useStore((s) => s.scene)!;
  const anim = useRef<{ p0: THREE.Vector3; p1: THREE.Vector3; t0: THREE.Vector3; t1: THREE.Vector3; fov: number; fov0: number; k: number } | null>(null);
  const did = useRef(false);

  const go = (pos: THREE.Vector3, target: THREE.Vector3, fov = 50, instant = false) => {
    const cam = camera as THREE.PerspectiveCamera;
    const t0 = controls?.target ? controls.target.clone() : cam.position.clone().add(cam.getWorldDirection(new THREE.Vector3()));
    if (instant) {
      cam.position.copy(pos);
      cam.fov = fov;
      cam.updateProjectionMatrix();
      cam.lookAt(target);
      if (controls?.target) { controls.target.copy(target); controls.update(); }
      return;
    }
    anim.current = { p0: cam.position.clone(), p1: pos, t0, t1: target, fov, fov0: cam.fov, k: 0 };
  };

  const overview = (instant = false) => {
    const c = ibox.getCenter(new THREE.Vector3());
    const r = Math.max(3, ibox.getSize(new THREE.Vector3()).length() / 2);
    const cam0 = scene.cameras[0];
    let dir = new THREE.Vector3(0.3, 0.6, 1).normalize();
    if (cam0) {
      const C = cameraCenter(cam0.world_from_cam);
      const d = C.clone().sub(c);
      d.y = 0;
      if (d.length() > 1e-3) dir = d.normalize().multiplyScalar(0.8).add(new THREE.Vector3(0, 0.6, 0)).normalize();
    }
    go(c.clone().addScaledVector(dir, r * 1.9), c, 50, instant);
  };

  useEffect(() => {
    if (!did.current && controls) {
      did.current = true;
      overview(true);
    }
  }, [controls]);

  useEffect(() => {
    if (!req) return;
    const st = useStore.getState();
    const cam = camera as THREE.PerspectiveCamera;
    if (req.kind === "overview") overview();
    else if (req.kind === "top") {
      const c = ibox.getCenter(new THREE.Vector3());
      const s = ibox.getSize(new THREE.Vector3());
      const h = Math.max(s.x, s.z, 4) * 0.6 / Math.tan((25 * Math.PI) / 180);
      go(new THREE.Vector3(c.x, Math.max(c.y, 0) + h, c.z + h * 0.002), new THREE.Vector3(c.x, 0, c.z));
    } else if (req.kind === "camera") {
      const c = scene.cameras.find((x) => x.id === req.target || x.video === req.target) ?? scene.cameras[0];
      if (!c) return;
      const pose = cameraPoseAt(c, st.time);
      const M = glCameraMatrix(pose.rows);
      const pos = new THREE.Vector3().setFromMatrixPosition(M);
      const fwd = new THREE.Vector3(0, 0, -1).transformDirection(M);
      const vfov = (2 * Math.atan(c.intrinsics.height / 2 / c.intrinsics.fy) * 180) / Math.PI;
      go(pos, pos.clone().addScaledVector(fwd, 5), vfov);
    } else if (req.kind === "focus") {
      const sel = st.selection;
      let c: THREE.Vector3 | null = null;
      let r = 3;
      if (sel?.kind === "entity") {
        const e = scene.entities.find((x) => x.id === sel.id);
        const b = e && entityBox(e);
        if (b) { c = b.getCenter(new THREE.Vector3()); r = Math.max(1.5, b.getSize(new THREE.Vector3()).length() / 2); }
      } else if (sel?.kind === "agent") {
        const a = scene.agents.find((x) => x.id === sel.id);
        if (a) { c = agentFocus(a, st.time); r = 3; }
      } else if (sel?.kind === "camera") {
        const cm = scene.cameras.find((x) => x.id === sel.id);
        if (cm) c = cameraCenter(cameraPoseAt(cm, st.time).rows);
      } else if (sel?.kind === "event") {
        const ev = scene.events.find((x) => x.id === sel.id);
        if (ev?.position) c = new THREE.Vector3(...ev.position);
      }
      if (!c) return;
      const dir = cam.position.clone().sub(controls?.target ?? c);
      if (dir.length() < 1e-6) dir.set(0, 1, 1);
      dir.normalize();
      if (dir.y < 0.25) { dir.y = 0.25; dir.normalize(); }
      go(c.clone().addScaledVector(dir, r * 2.6), c);
    }
  }, [req?.nonce]);

  useFrame((_, dt) => {
    const a = anim.current;
    if (!a) return;
    const cam = camera as THREE.PerspectiveCamera;
    a.k = Math.min(1, a.k + dt / 0.55);
    const e = a.k < 0.5 ? 2 * a.k * a.k : 1 - Math.pow(-2 * a.k + 2, 2) / 2;
    cam.position.lerpVectors(a.p0, a.p1, e);
    const t = new THREE.Vector3().lerpVectors(a.t0, a.t1, e);
    cam.fov = a.fov0 + (a.fov - a.fov0) * e;
    cam.updateProjectionMatrix();
    if (controls?.target) { controls.target.copy(t); controls.update(); } else cam.lookAt(t);
    if (a.k >= 1) anim.current = null;
  });
  return null;
}

// ------------------------------------------------------------------------------- overlays
function Toolbar({ onSnapshot }: { onSnapshot: () => void }) {
  const st = useStore();
  const tools: [Tool, string, string][] = [
    ["navigate", "✋", "Navigate / select (Esc)"], ["measure", "📏", "Measure distance (M)"], ["height", "⇕", "Height above ground (H)"],
    ["angle", "∠", "Angle"], ["inspect", "⌖", "Inspect a point"], ["calibrate", "⚖", "Calibrate scale with a known length (K)"],
  ];
  return (
    <div className="toolbar">
      <div className="grp">
        {tools.map(([t, icon, title]) => <button key={t} className={`icon ${st.tool === t ? "active" : ""}`} title={title} onClick={() => st.setTool(t)}>{icon}</button>)}
      </div>
      <div className="grp">
        {(["photo", "evidence", "semantic", "confidence"] as const).map((m, i) => (
          <button key={m} className={`icon ${st.colorMode === m ? "active" : ""}`} title={`Colour by ${m} (${i + 1})`} onClick={() => st.setColorMode(m)}>{m}</button>
        ))}
      </div>
      <div className="grp">
        <button className="icon" title="Overview (O)" onClick={() => st.requestView("overview")}>overview</button>
        <button className="icon" title="Top view (T)" onClick={() => st.requestView("top")}>top</button>
        {st.scene!.cameras.map((c) => <button key={c.id} className="icon" title={`View from ${c.id}`} onClick={() => st.requestView("camera", c.id)}>{c.video}</button>)}
      </div>
      <div className="grp">
        <button className={`icon ${st.navMode === "orbit" ? "active" : ""}`} title="Orbit / pan / zoom" onClick={() => st.setNavMode("orbit")}>orbit</button>
        <button className={`icon ${st.navMode === "fly" ? "active" : ""}`} title="First person: drag to look, WASD move, Q/E down/up, Shift faster" onClick={() => st.setNavMode("fly")}>walk</button>
        <button className={`icon ${st.wireframe ? "active" : ""}`} title="Wireframe" onClick={() => st.setWireframe(!st.wireframe)}>wire</button>
        <button className="icon" title="Save a snapshot of the view to the case (included in reports)" onClick={onSnapshot}>📷</button>
      </div>
    </div>
  );
}

function Legend() {
  const mode = useStore((s) => s.colorMode);
  const scene = useStore((s) => s.scene)!;
  const [open, setOpen] = useState(true);
  if (!open) return <button className="legend" style={{ padding: "4px 8px" }} onClick={() => setOpen(true)}>legend</button>;
  const present = new Set<string>();
  scene.layers.forEach((l) => Object.keys(l.stats?.evidence_vertices ?? {}).forEach((k) => present.add(k)));
  scene.entities.forEach((e) => e.evidence?.class && present.add(e.evidence.class));
  return (
    <div className="legend" onClick={() => setOpen(false)} title="click to hide">
      {mode === "evidence" && EVIDENCE.filter((c) => present.has(c) || c === "INFERRED").map((c) => (
        <div key={c} className="row"><span className="swatch" style={{ background: EVIDENCE_COLORS[c] }} />{EVIDENCE_SHORT[c]}</div>
      ))}
      {mode === "semantic" && <div className="muted">Surface labels from dense semantic segmentation (Cityscapes classes; indicative only).</div>}
      {mode === "confidence" && (
        <div>
          <div style={{ height: 8, width: 180, background: "linear-gradient(90deg,#ff3c3c,#ffff3c,#3cff3c)", borderRadius: 3 }} />
          <div className="row small muted"><span className="grow">low</span><span>surface confidence</span><span className="grow" style={{ textAlign: "right" }}>high</span></div>
        </div>
      )}
      {mode === "photo" && <div className="muted">Colours sampled from the source frames.</div>}
      <div className="sep" />
      <div className="row small"><span className="swatch" style={{ background: "repeating-linear-gradient(45deg,#dc5050 0 3px,#3a1a1a 3px 6px)" }} />inferred (not observed)</div>
      <div className="row small"><span style={{ width: 16, borderTop: "2px dashed #aaa" }} />track gap (not observed)</div>
      <div className="row small"><span style={{ width: 16, borderTop: "2px dashed #ffb347" }} />candidate opening / height lower bound</div>
      <div className="row small"><span className="swatch" style={{ background: "#5a82dc", height: 2 }} />observed ground extent</div>
    </div>
  );
}

function StatusBar() {
  const scene = useStore((s) => s.scene)!;
  const t = useStore((s) => s.time);
  const cs = scene.coordinate_system;
  return (
    <div className="statusbar">
      <span className="mono">{fmtTime(t)}</span> · {cs.status.replace("_", " ")} ({cs.unit}{cs.scale.scale_sigma_rel ? ` ±${(cs.scale.scale_sigma_rel * 100).toFixed(0)}%` : ""}) · y up, ground y=0
    </div>
  );
}

function CalibrateDialog({ p1, p2, unit, onClose }: { p1: Vec3; p2: Vec3; unit: string; onClose: () => void }) {
  const caseId = useStore((s) => s.caseId)!;
  const notify = useStore((s) => s.notify);
  const [len, setLen] = useState("");
  const [sigma, setSigma] = useState("0.02");
  const [label, setLabel] = useState("");
  const shown = Math.hypot(p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]);
  const submit = () => {
    const L = parseFloat(len);
    if (!(L > 0)) return notify("Enter the true length in metres", "error");
    api.calibrate(caseId, { p1, p2, true_length_m: L, sigma_m: parseFloat(sigma) || 0, label })
      .then(() => { notify("Scale reference saved; the scene is being recomputed in calibrated metres", "ok"); onClose(); })
      .catch((e) => notify(e.message, "error"));
  };
  return (
    <div className="overlay">
      <div className="modal">
        <h3>Calibrate scale with a known length</h3>
        <div className="muted small" style={{ margin: "6px 0 10px" }}>
          The two points are {fmt(shown, 3)} {unit} apart in the current reconstruction. Enter the real-world length of this segment
          (for example a measured parking bay, door width, or kerb). The reference is stored as an analyst annotation, logged in the audit
          trail, and the scale status becomes CALIBRATED_METRIC. All values are recomputed; the original analysis is kept.
        </div>
        <div className="col">
          <label className="row"><span style={{ width: 140 }}>True length (m)</span><input autoFocus value={len} onChange={(e) => setLen(e.target.value)} /></label>
          <label className="row"><span style={{ width: 140 }}>Uncertainty 1σ (m)</span><input value={sigma} onChange={(e) => setSigma(e.target.value)} /></label>
          <label className="row"><span style={{ width: 140 }}>What was measured</span><input className="grow" placeholder="e.g. parking bay width, surveyed" value={label} onChange={(e) => setLabel(e.target.value)} /></label>
        </div>
        <div className="row" style={{ marginTop: 14 }}><span className="grow" /><button onClick={onClose}>Cancel</button><button className="primary" onClick={submit}>Apply calibration</button></div>
      </div>
    </div>
  );
}

function SnapshotDialog({ blob, onClose }: { blob: Blob; onClose: () => void }) {
  const caseId = useStore((s) => s.caseId)!;
  const notify = useStore((s) => s.notify);
  const [caption, setCaption] = useState("");
  const url = useMemo(() => URL.createObjectURL(blob), [blob]);
  useEffect(() => () => URL.revokeObjectURL(url), [url]);
  return (
    <div className="overlay">
      <div className="modal">
        <h3>Save snapshot</h3>
        <img src={url} style={{ width: "100%", margin: "10px 0", border: "1px solid var(--line)" }} />
        <input className="grow" style={{ width: "100%" }} placeholder="Caption (what this view shows)" value={caption} onChange={(e) => setCaption(e.target.value)} />
        <div className="row" style={{ marginTop: 12 }}>
          <a href={url} download="metrica_snapshot.png">download</a>
          <span className="grow" />
          <button onClick={onClose}>Cancel</button>
          <button className="primary" onClick={() => api.snapshot(caseId, blob, caption).then(() => { notify("Snapshot saved to the case", "ok"); onClose(); }).catch((e) => notify(e.message, "error"))}>Save to case</button>
        </div>
      </div>
    </div>
  );
}
