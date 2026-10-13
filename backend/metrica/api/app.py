"""Local HTTP API and static frontend server.

The server binds to 127.0.0.1 by default: footage, crops and case data never leave the
machine unless the user explicitly binds to another interface or exports files.
"""
from __future__ import annotations

import os
import re
import shutil
import tempfile
import threading
import time
import uuid
import webbrowser
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import __version__
from ..analysis import measurements as meas
from ..config import AnalysisConfig, profile_defaults
from ..evidence.audit import verify_chain
from ..evidence.case import Case, default_workspace, list_cases
from ..hardware import hardware_summary
from ..log import get_logger
from ..models import registry
from ..pipeline.base import frame_fingerprint
from ..pipeline.runner import stage_statuses, stage_titles
from .jobs import JobStore

log = get_logger("api")


def frontend_dir() -> Path:
    env = os.environ.get("METRICA_FRONTEND")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for cand in (here.parent / "static", here.parents[3] / "frontend" / "dist"):
        if (cand / "index.html").exists():
            return cand
    return here.parents[3] / "frontend" / "dist"


# ------------------------------------------------------------------------------ request models
class NewCase(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    reference: str = ""


class AnalyzeReq(BaseModel):
    profile: str = "standard"
    overrides: dict[str, Any] = {}
    force: bool = False


class PathImport(BaseModel):
    path: str
    copy_into_case: bool = True


class MeasureReq(BaseModel):
    kind: str
    points: list[list[float]] | None = None
    evidence: list[str | None] | None = None
    targets: list[str] | None = None
    t: float | None = None
    t0: float | None = None
    t1: float | None = None
    label: str = ""
    save: bool = True


class CalibrationReq(BaseModel):
    p1: list[float]
    p2: list[float]
    true_length_m: float = Field(gt=0)
    sigma_m: float = Field(ge=0, default=0.02)
    label: str = ""


class SyncReq(BaseModel):
    video: str
    offset_s: float


class AlignReq(BaseModel):
    unit: str
    pairs: list[dict[str, list[float]]]


class VerifyReq(BaseModel):
    target: str
    verdict: str = Field(pattern="^(verified|rejected|uncertain)$")
    note: str = ""


class NoteReq(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    target: str | None = None


class ReportReq(BaseModel):
    analyst: str = ""
    purpose: str = ""
    pdf: bool = True


class DownloadReq(BaseModel):
    keys: list[str] = []
    include_optional: bool = False
    accept_restricted: bool = False


def create_app(workspace: Path | None = None) -> FastAPI:
    ws = Path(workspace or default_workspace()).expanduser()
    ws.mkdir(parents=True, exist_ok=True)
    jobs = JobStore(ws / ".metrica" / "jobs.db")
    app = FastAPI(title="METRICA", version=__version__)
    app.add_middleware(GZipMiddleware, minimum_size=4096)
    app.state.workspace = ws
    app.state.jobs = jobs
    downloads: dict[str, Any] = {"running": False, "items": {}, "log": []}
    report_tasks: dict[str, dict[str, Any]] = {}
    actor_default = os.environ.get("METRICA_ANALYST", "analyst")

    def actor_of(req: Request) -> str:
        return req.headers.get("X-Metrica-Analyst") or actor_default

    def case_by_id(cid: str) -> Case:
        if not re.fullmatch(r"[0-9a-f]{6,32}", cid):
            raise HTTPException(400, "invalid case id")
        for c in list_cases(ws):
            if c["id"] == cid:
                return Case(Path(c["path"]))
        raise HTTPException(404, "case not found")

    # ------------------------------------------------------------------ system
    @app.get("/api/system")
    def system() -> dict[str, Any]:
        st = registry.all_status()
        return {"version": __version__, "workspace": str(ws), "hardware": hardware_summary(),
                "models": {"ready": sum(s["state"] == "ready" for s in st), "total": len(st),
                           "missing_default": [s["key"] for s in st if s["default"] and s["state"] != "ready"]},
                "frontend": (frontend_dir() / "index.html").exists(), "stages": stage_titles(),
                "profiles": {p: profile_defaults(p).to_dict() for p in ("fast", "standard", "quality")},
                "bind_note": "Local-first: the server only listens on this computer unless configured otherwise."}

    @app.get("/api/models")
    def models() -> dict[str, Any]:
        return {"models": registry.all_status(), "directory": str(registry.models_dir()), "download": downloads}

    @app.post("/api/models/download")
    def models_download(req: DownloadReq) -> dict[str, Any]:
        if downloads["running"]:
            return downloads
        keys = req.keys or [k for k, m in registry.MODELS.items()
                            if (m.default or req.include_optional) and (not m.restricted or req.accept_restricted)]
        for k in keys:
            if registry.MODELS[k].restricted and not req.accept_restricted:
                raise HTTPException(400, f"{k} requires accept_restricted (license {registry.MODELS[k].license})")

        def run():
            downloads["running"] = True
            for k in keys:
                downloads["items"][k] = {"state": "downloading", "done": 0, "total": 0}
                try:
                    registry.download(k, accept_restricted=req.accept_restricted,
                                      progress=lambda key, d, t: downloads["items"][key].update(done=d, total=t))
                    downloads["items"][k]["state"] = "ready"
                except Exception as e:  # noqa: BLE001
                    downloads["items"][k] = {"state": "failed", "error": str(e)}
            downloads["running"] = False

        threading.Thread(target=run, daemon=True).start()
        return {"started": keys}

    # ------------------------------------------------------------------ cases
    @app.get("/api/cases")
    def cases() -> list[dict[str, Any]]:
        return list_cases(ws)

    @app.post("/api/cases")
    def create_case(req: NewCase, request: Request) -> dict[str, Any]:
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", req.title).strip("-").lower()[:40] or "case"
        root = ws / f"{slug}-{uuid.uuid4().hex[:6]}"
        case = Case.create(root, req.title, req.reference)
        case.audit.append("case.opened_in_app", actor=actor_of(request))
        return case_detail(case.id)

    @app.get("/api/cases/{cid}")
    def case_detail(cid: str) -> dict[str, Any]:
        case = case_by_id(cid)
        meta = case.meta
        vids = []
        for v in meta["videos"]:
            ing = case.read_json(f"derived/videos/{v['id']}/ingest.json")
            vids.append({**{k: v[k] for k in ("id", "original_name", "size_bytes", "sha256_intake", "intake_time",
                                              "copied")},
                         "ingest": None if not ing else {
                             "probe": ing["probe"], "timing": ing["timing"], "mode": ing["mode"],
                             "quality": ing["quality"]["summary"], "frame_size": ing["frame_size"],
                             "decode": ing["decode"], "keyframes": len(ing["keyframes"])}})
        status = case.read_json("derived/status.json", {"state": "new"})
        reports = sorted((p.parent.name for p in (case.root / "derived" / "reports").glob("*/report.json")),
                         reverse=True) if (case.root / "derived" / "reports").exists() else []
        return {"id": meta["id"], "title": meta["title"], "reference": meta.get("reference", ""),
                "created": meta["created"], "path": str(case.root), "videos": vids, "status": status,
                "stages": stage_statuses(case), "job": jobs.active_for(meta["id"]),
                "has_scene": (case.root / "derived" / "scene" / "scene.json").exists(), "reports": reports}

    @app.post("/api/cases/{cid}/videos")
    async def upload_video(cid: str, request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
        case = case_by_id(cid)
        if jobs.active_for(cid):
            raise HTTPException(409, "an analysis is running for this case")
        name = Path(file.filename or "video").name
        tmpdir = Path(tempfile.mkdtemp(dir=case.root / "evidence"))
        tmp = tmpdir / name
        try:
            with open(tmp, "wb") as f:
                while True:
                    chunk = await file.read(8 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            try:
                from ..ingest.probe import probe

                probe(tmp)
            except Exception as e:  # noqa: BLE001
                raise HTTPException(400, f"Not a readable video file: {e}") from e
            try:
                rec = case.add_video(tmp, copy=True, actor=actor_of(request))
            except ValueError as e:
                raise HTTPException(409, str(e)) from e
            rec["original_path"] = f"(uploaded through the application as '{name}')"
            case.update_video(rec["id"], original_path=rec["original_path"])
            return {"video": rec}
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @app.post("/api/cases/{cid}/videos/path")
    def import_path(cid: str, req: PathImport, request: Request) -> dict[str, Any]:
        case = case_by_id(cid)
        p = Path(req.path).expanduser()
        if not p.is_file():
            raise HTTPException(404, f"file not found: {p}")
        try:
            from ..ingest.probe import probe

            probe(p)
            rec = case.add_video(p, copy=req.copy_into_case, actor=actor_of(request))
        except ValueError as e:
            raise HTTPException(409, str(e)) from e
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"Not a readable video file: {e}") from e
        return {"video": rec}

    @app.get("/api/cases/{cid}/videos/{vid}/probe")
    def probe_video(cid: str, vid: str) -> dict[str, Any]:
        from ..ingest.probe import probe

        case = case_by_id(cid)
        return probe(case.video_path(vid))

    @app.post("/api/cases/{cid}/analyze")
    def analyze(cid: str, req: AnalyzeReq, request: Request) -> dict[str, Any]:
        case = case_by_id(cid)
        if not case.videos():
            raise HTTPException(400, "add at least one video first")
        if jobs.active_for(cid):
            raise HTTPException(409, "an analysis is already queued or running for this case")
        try:
            cfg = AnalysisConfig.from_dict({**req.overrides, "profile": req.profile})
        except (ValueError, TypeError) as e:
            raise HTTPException(400, str(e)) from e
        missing = [k for k in ("depth_anything_v2_s", cfg.detector) if not registry.available(k)]
        job = jobs.submit(cid, case.root, cfg.to_dict(), req.force, actor=actor_of(request))
        return {"job": job, "missing_models": missing}

    @app.get("/api/jobs/{jid}")
    def job(jid: str) -> dict[str, Any]:
        j = jobs.get(jid)
        if not j:
            raise HTTPException(404, "job not found")
        st = {}
        try:
            st = Case(Path(j["case_path"])).read_json("derived/status.json", {})
        except FileNotFoundError:
            pass
        return {"job": j, "status": st}

    @app.post("/api/jobs/{jid}/cancel")
    def cancel(jid: str, request: Request) -> dict[str, Any]:
        j = jobs.cancel(jid)
        if not j:
            raise HTTPException(404, "job not found")
        Case(Path(j["case_path"])).audit.append("analysis.cancel_requested", actor=actor_of(request), target=jid)
        return {"job": j}

    @app.get("/api/cases/{cid}/jobs")
    def case_jobs(cid: str) -> list[dict[str, Any]]:
        return jobs.list(cid)

    # ------------------------------------------------------------------ scene & files
    @app.get("/api/cases/{cid}/scene")
    def scene(cid: str) -> Response:
        case = case_by_id(cid)
        p = case.root / "derived" / "scene" / "scene.json"
        if not p.exists():
            raise HTTPException(404, "no scene yet: run the analysis")
        return FileResponse(p, media_type="application/json")

    @app.get("/api/cases/{cid}/files/{rel:path}")
    def files(cid: str, rel: str) -> Response:
        case = case_by_id(cid)
        base = (case.root / "derived").resolve()
        p = (base / rel).resolve()
        if base not in p.parents and p != base:
            raise HTTPException(403, "outside the case's derived data")
        if not p.is_file():
            raise HTTPException(404, "file not found")
        return FileResponse(p)

    @app.get("/api/cases/{cid}/frame/{vid}/{frame}")
    def source_frame(cid: str, vid: str, frame: int) -> Response:
        """The exact decoded source frame (from the evidence file), cached as a PNG."""
        import cv2

        from ..ingest.decode import read_frames_at

        case = case_by_id(cid)
        cdir = case.stage_dir("frames", vid)
        p = cdir / f"{frame:07d}.png"
        if not p.exists():
            imgs = read_frames_at(case.video_path(vid), [frame])
            if frame not in imgs:
                raise HTTPException(404, "frame not found")
            cv2.imwrite(str(p), imgs[frame])
        return FileResponse(p, media_type="image/png")

    # ------------------------------------------------------------------ analyst actions
    def _scene_json(case: Case) -> dict[str, Any]:
        s = case.read_json("derived/scene/scene.json")
        if s is None:
            raise HTTPException(404, "no scene yet")
        return s

    @app.post("/api/cases/{cid}/measurements")
    def measure(cid: str, req: MeasureReq, request: Request) -> dict[str, Any]:
        case = case_by_id(cid)
        sc = _scene_json(case)
        try:
            r = meas.compute(sc, req.kind, req.points, req.evidence, req.targets, req.t, req.t0, req.t1)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        if not req.save:
            return r
        k = sc["coordinate_system"]["global_scale_factor"] or 1.0
        base = [[v / k for v in p] for p in (req.points or [])]
        rec = meas.record(r, req.points, req.targets, actor_of(request), req.label, base)
        rec["frame_fingerprint"] = frame_fingerprint(case)
        lst = case.annotations("measurements")
        lst.append(rec)
        case.save_annotations("measurements", lst, actor_of(request), "annotation.measurement_added",
                              {"id": rec["id"], "kind": rec["kind"], "display": rec["display"]})
        return rec

    @app.get("/api/cases/{cid}/measurements")
    def list_measurements(cid: str) -> list[dict[str, Any]]:
        case = case_by_id(cid)
        return meas.current_measurements(case.annotations("measurements"), case.read_json("derived/scene/scene.json"),
                                         frame_fingerprint(case))

    @app.post("/api/cases/{cid}/measurements/{mid}/withdraw")
    def withdraw_measurement(cid: str, mid: str, request: Request) -> dict[str, Any]:
        case = case_by_id(cid)
        lst = case.annotations("measurements")
        for m in lst:
            if m["id"] == mid:
                m["active"] = False
                m["withdrawn"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        case.save_annotations("measurements", lst, actor_of(request), "annotation.measurement_withdrawn", {"id": mid})
        return {"ok": True}

    @app.post("/api/cases/{cid}/calibration")
    def calibrate(cid: str, req: CalibrationReq, request: Request) -> dict[str, Any]:
        case = case_by_id(cid)
        sc = _scene_json(case)
        k = sc["coordinate_system"]["global_scale_factor"] or 1.0
        cal = case.annotations("calibration") or {}
        refs = cal.get("references", [])
        ref = {"id": "C" + uuid.uuid4().hex[:6].upper(), "p1": [v / k for v in req.p1], "p2": [v / k for v in req.p2],
               "true_length_m": req.true_length_m, "sigma_m": req.sigma_m, "label": req.label,
               "displayed_length_before": float(np.linalg.norm(np.subtract(req.p2, req.p1))),
               "actor": actor_of(request), "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "active": True,
               "frame_fingerprint": frame_fingerprint(case)}
        refs.append(ref)
        case.save_annotations("calibration", {"references": refs}, actor_of(request), "annotation.calibration_added",
                              {k2: ref[k2] for k2 in ("id", "true_length_m", "sigma_m", "label",
                                                      "displayed_length_before")})
        job = jobs.submit(cid, case.root, case.read_json("derived/config.json") or AnalysisConfig().to_dict(),
                          actor=actor_of(request), kind="recalibration")
        return {"reference": ref, "job": job}

    @app.post("/api/cases/{cid}/calibration/{ref_id}/deactivate")
    def calib_off(cid: str, ref_id: str, request: Request) -> dict[str, Any]:
        case = case_by_id(cid)
        cal = case.annotations("calibration") or {}
        for r in cal.get("references", []):
            if r["id"] == ref_id:
                r["active"] = False
        case.save_annotations("calibration", cal, actor_of(request), "annotation.calibration_deactivated", {"id": ref_id})
        job = jobs.submit(cid, case.root, case.read_json("derived/config.json") or AnalysisConfig().to_dict(),
                          actor=actor_of(request), kind="recalibration")
        return {"job": job}

    @app.post("/api/cases/{cid}/sync")
    def set_sync(cid: str, req: SyncReq, request: Request) -> dict[str, Any]:
        case = case_by_id(cid)
        s = case.annotations("sync") or {}
        s[req.video] = {"offset_s": req.offset_s, "actor": actor_of(request), "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        case.save_annotations("sync", s, actor_of(request), "annotation.sync_offset_set", {"video": req.video,
                                                                                           "offset_s": req.offset_s})
        job = jobs.submit(cid, case.root, case.read_json("derived/config.json") or AnalysisConfig().to_dict(),
                          actor=actor_of(request), kind="resync")
        return {"job": job}

    @app.post("/api/cases/{cid}/alignment")
    def set_alignment(cid: str, req: AlignReq, request: Request) -> dict[str, Any]:
        case = case_by_id(cid)
        if len(req.pairs) < 2:
            raise HTTPException(400, "at least two point pairs are needed")
        a = case.annotations("alignment") or {}
        a[req.unit] = req.pairs
        case.save_annotations("alignment", a, actor_of(request), "annotation.alignment_points_set",
                              {"unit": req.unit, "pairs": len(req.pairs)})
        job = jobs.submit(cid, case.root, case.read_json("derived/config.json") or AnalysisConfig().to_dict(),
                          actor=actor_of(request), kind="realign")
        return {"job": job}

    @app.post("/api/cases/{cid}/verifications")
    def verify_item(cid: str, req: VerifyReq, request: Request) -> dict[str, Any]:
        case = case_by_id(cid)
        lst = case.annotations("verifications")
        rec = {"id": "V" + uuid.uuid4().hex[:6].upper(), "target": req.target, "verdict": req.verdict, "note": req.note,
               "actor": actor_of(request), "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        lst.append(rec)
        case.save_annotations("verifications", lst, actor_of(request), "annotation.verification", rec)
        return rec

    @app.get("/api/cases/{cid}/annotations/{kind}")
    def get_annotations(cid: str, kind: str) -> Any:
        case = case_by_id(cid)
        try:
            return case.annotations(kind)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    @app.post("/api/cases/{cid}/notes")
    def add_note(cid: str, req: NoteReq, request: Request) -> dict[str, Any]:
        case = case_by_id(cid)
        lst = case.annotations("notes")
        rec = {"id": "N" + uuid.uuid4().hex[:6].upper(), "text": req.text, "target": req.target,
               "actor": actor_of(request), "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        lst.append(rec)
        case.save_annotations("notes", lst, actor_of(request), "annotation.note_added", {"id": rec["id"]})
        return rec

    @app.post("/api/cases/{cid}/snapshots")
    async def snapshot(cid: str, request: Request, file: UploadFile = File(...), caption: str = Form("")) -> dict[str, Any]:
        case = case_by_id(cid)
        sid = "S" + uuid.uuid4().hex[:6].upper()
        d = case.root / "annotations" / "snapshots"
        d.mkdir(parents=True, exist_ok=True)
        data = await file.read()
        if not data.startswith(b"\x89PNG"):
            raise HTTPException(400, "snapshot must be a PNG image")
        (d / f"{sid}.png").write_bytes(data)
        lst = case.annotations("snapshots")
        rec = {"id": sid, "file": f"annotations/snapshots/{sid}.png", "caption": caption, "actor": actor_of(request),
               "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        lst.append(rec)
        case.save_annotations("snapshots", lst, actor_of(request), "annotation.snapshot_added", {"id": sid})
        return rec

    # ------------------------------------------------------------------ reports, audit, integrity
    @app.post("/api/cases/{cid}/reports")
    def make_report(cid: str, req: ReportReq, request: Request) -> dict[str, Any]:
        case = case_by_id(cid)
        _scene_json(case)
        tid = uuid.uuid4().hex[:8]
        report_tasks[tid] = {"state": "running"}

        def run():
            try:
                from ..reporting.exports import generate_report

                rec = generate_report(case, {"analyst": req.analyst, "purpose": req.purpose}, actor=actor_of(request),
                                      pdf=req.pdf)
                report_tasks[tid] = {"state": "done", "report": rec}
            except Exception as e:  # noqa: BLE001
                log.exception("report failed")
                report_tasks[tid] = {"state": "failed", "error": str(e)}

        threading.Thread(target=run, daemon=True).start()
        return {"task": tid}

    @app.get("/api/report-tasks/{tid}")
    def report_task(tid: str) -> dict[str, Any]:
        if tid not in report_tasks:
            raise HTTPException(404, "unknown task")
        return report_tasks[tid]

    @app.get("/api/cases/{cid}/reports")
    def reports(cid: str) -> list[dict[str, Any]]:
        case = case_by_id(cid)
        base = case.root / "derived" / "reports"
        out = []
        if base.exists():
            for p in sorted(base.glob("*/report.json"), reverse=True):
                r = case.read_json(str(p.relative_to(case.root)))
                r["files"] = {k: (v.replace("derived/", "", 1) if isinstance(v, str) else
                                  [x.replace("derived/", "", 1) for x in v]) for k, v in r["files"].items()}
                out.append(r)
        return out

    @app.get("/api/cases/{cid}/audit")
    def audit(cid: str) -> dict[str, Any]:
        case = case_by_id(cid)
        e = case.audit.entries()
        return {"entries": e, "chain": verify_chain(e)}

    @app.get("/api/cases/{cid}/verify")
    def verify(cid: str, request: Request) -> dict[str, Any]:
        case = case_by_id(cid)
        res = case.verify_originals()
        case.audit.append("evidence.verified_on_request", actor=actor_of(request),
                          details={r["id"]: r["ok"] for r in res})
        return {"evidence": res, "chain": verify_chain(case.audit.entries())}

    @app.get("/api/cases/{cid}/manifest")
    def manifest(cid: str) -> Response:
        case = case_by_id(cid)
        p = case.root / "manifest.json"
        if not p.exists():
            raise HTTPException(404, "no manifest yet")
        return FileResponse(p, media_type="application/json")

    @app.exception_handler(FileNotFoundError)
    async def fnf(_, exc):  # noqa: ANN001
        return JSONResponse({"detail": str(exc)}, status_code=404)

    fd = frontend_dir()
    if (fd / "index.html").exists():
        app.mount("/", StaticFiles(directory=str(fd), html=True), name="frontend")
    else:
        @app.get("/")
        def no_frontend() -> Response:
            return Response("<h1>METRICA</h1><p>The frontend build was not found. Run <code>npm --prefix frontend "
                            "install && npm --prefix frontend run build</code>. The API is available under /api.</p>",
                            media_type="text/html")
    return app


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True, workspace: str | None = None) -> None:
    import uvicorn

    app = create_app(Path(workspace) if workspace else None)
    url = f"http://{host}:{port}/"
    log.info(f"METRICA {__version__} serving {url} (workspace {app.state.workspace})")
    if host not in ("127.0.0.1", "localhost", "::1"):
        log.warning("Listening on a non-local interface: case data becomes reachable from the network.")
    if open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
