"""Assemble the report data model from a processed case (single source for HTML, PDF, exports)."""
from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import cv2

from .. import __version__
from ..analysis.measurements import current_measurements
from ..evidence.audit import verify_chain
from ..evidence.case import Case
from ..evidence.manifest import INTEGRITY_STATEMENT
from ..geometry.calibration import STATUS_TEXT
from ..pipeline.base import frame_fingerprint
from ..pipeline.runner import stage_statuses
from ..reconstruction.layers import EVIDENCE_TEXT
from . import figures

METHOD_TEXT = {
    "single_view": (
        "Static camera, single view. A background plate was formed as the per-pixel temporal median of frames sampled "
        "across the recording, removing moving subjects. Camera orientation and height were estimated by single-view "
        "metrology: the vertical vanishing point of long vertical edges and the foot/head positions of standing people "
        "(assumed mean stature). Monocular depth (Depth Anything V2) was aligned to the calibrated ground plane by a "
        "robust affine fit in inverse depth, ground pixels were snapped to the plane, vertical walls were fitted as "
        "planes, and the remaining surfaces were regularised per superpixel. Everything visible comes from one "
        "viewpoint: depth along the viewing direction is the least constrained quantity."),
    "sfm": (
        "Moving camera, multi-view. Keyframes were selected adaptively for parallax and sharpness. COLMAP structure "
        "from motion estimated camera poses, intrinsics (refined from an explicit field-of-view prior) and a sparse "
        "set of triangulated points. The reconstruction was aligned to gravity using the ground plane fitted to "
        "points labelled as walking surface. Monocular depth for each keyframe was aligned to the points that "
        "keyframe observed; pixels contradicted by neighbouring keyframes were discarded; the rest were fused in a "
        "truncated signed distance field and meshed. Each vertex records how many keyframes supported it."),
}
OBJECT_TEXT = (
    "Semantic objects are reconstructed with category-specific methods: facades as vertical planes whose polygons "
    "cover only the observed extent (with candidate openings and separately marked inferred patches); poles as "
    "gravity-aligned cylinders from their image footprint, base contact and apparent width, merged across views; "
    "parked vehicles and static objects from instance masks lifted to 3D, with unobserved vehicle dimensions completed "
    "by class priors and labelled as such; the ground as a plane whose observed and hidden parts come from the "
    "cameras' visibility. Moving people and vehicles are represented by their 3D trajectories and generic proxies "
    "(mannequin / vehicle template), never as scans of the actual individual or vehicle.")


def _img(src: Path, dst_dir: Path, name: str, max_side: int | None = None) -> str | None:
    if not src.exists():
        return None
    dst = dst_dir / name
    if max_side:
        im = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if im is None:
            return None
        s = max_side / max(im.shape[:2])
        if s < 1:
            im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst), im)
    else:
        shutil.copyfile(src, dst)
    return f"assets/{name}"


def fmt_dim(d: dict[str, Any]) -> str:
    if not isinstance(d.get("value"), (int, float)):
        return "-"
    s = f"{d['value']:.2f}"
    if d.get("sigma") is not None:
        s += f" ± {d['sigma']:.2f}"
    s += f" {d.get('unit', '')}".rstrip()
    if d.get("basis"):
        s += f" ({d['basis']})"
    return s


def build(case: Case, options: dict[str, Any] | None = None) -> tuple[dict[str, Any], Path]:
    options = options or {}
    scene = case.read_json("derived/scene/scene.json")
    if scene is None:
        raise RuntimeError("The case has no scene yet: run the analysis first.")
    rid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    rdir = case.stage_dir("reports", rid)
    adir = rdir / "assets"
    adir.mkdir(parents=True, exist_ok=True)
    sdir = case.root / "derived" / "scene"
    manifest = case.read_json("manifest.json", {})
    ann_meas = current_measurements(case.annotations("measurements"), scene, frame_fingerprint(case),
                                    include_withdrawn=False)
    ann_ver = case.annotations("verifications")
    ann_notes = case.annotations("notes")
    ann_snap = case.annotations("snapshots")
    chain = verify_chain(case.audit.entries())
    unit = scene["coordinate_system"]["unit"]
    # ---- figures
    figs = {}
    figures.plan_view(scene, adir / "plan_view.png")
    figs["plan_view"] = "assets/plan_view.png"
    if figures.perspective_views(scene, sdir, adir / "view_color.png", adir / "view_evidence.png"):
        figs["view_color"] = "assets/view_color.png"
        figs["view_evidence"] = "assets/view_evidence.png"
    figures.timeline_chart(scene, adir / "timeline.png")
    if (adir / "timeline.png").exists():
        figs["timeline"] = "assets/timeline.png"
    snapshots = []
    for s in ann_snap:
        p = case.root / s["file"]
        rel = _img(p, adir, f"snapshot_{s['id']}.png")
        if rel:
            snapshots.append({**s, "image": rel})
    # ---- inputs
    inputs = []
    for v in scene["videos"]:
        ing = case.read_json(f"derived/videos/{v['id']}/ingest.json")
        figures.quality_chart(ing, adir / f"quality_{v['id']}.png")
        rec = case.video(v["id"])
        thumb = _img(case.root / "derived" / v["thumbnail"], adir, f"thumb_{v['id']}.jpg")
        inputs.append({
            "id": v["id"], "name": v["name"], "sha256": rec["sha256_intake"], "size_bytes": rec["size_bytes"],
            "intake_time": rec["intake_time"], "copied": rec.get("copied"),
            "container": ing["probe"]["container"], "codec": ing["probe"]["video"]["codec"],
            "resolution": f"{v['width']} x {v['height']}", "fps": round(v["fps"], 3), "frames": v["frames"],
            "duration_s": round(v["duration"] or 0, 2), "vfr": v["vfr"], "audio": ing["probe"]["has_audio"],
            "rotation": ing["decode"]["rotation_applied_deg"], "decode_errors": ing["decode"]["decode_errors"],
            "creation_time": ing["probe"].get("creation_time"), "mode": ing["mode"]["mode"],
            "mode_confidence": round(ing["mode"]["confidence"], 2), "mode_notes": ing["mode"]["notes"],
            "quality": ing["quality"]["summary"], "quality_chart": f"assets/quality_{v['id']}.png", "thumbnail": thumb,
            "timing": {k: ing["timing"][k] for k in ("fps_measured", "vfr", "gap_count", "duplicate_frames")
                       if k in ing["timing"]},
            "offset": v["offset"], "sync": v.get("sync")})
    # ---- units / reconstruction quality
    units = case.read_json("derived/geometry/units.json")["units"]
    unit_rows = []
    for u in units:
        recon = case.read_json(f"derived/reconstruction/{u['id']}/reconstruction.json") or {}
        geo = case.read_json(f"derived/geometry/{u['videos'][0]}.json") or {}
        unit_rows.append({"id": u["id"], "kind": u["kind"], "videos": u["videos"], "scale": u["scale"],
                          "method": METHOD_TEXT.get(u["kind"], ""), "calibration": geo.get("calibration"),
                          "sfm": u.get("sfm"), "gravity": u.get("gravity"), "fusion": recon.get("fusion"),
                          "coverage": recon.get("coverage"), "report": recon.get("report"),
                          "layers": [{"name": l["name"], "kind": l["kind"], **l["stats"]} for l in recon.get("layers", [])],
                          "alignment": scene["alignment"].get(u["id"])})
    # ---- objects
    objects = []
    for e in scene["entities"]:
        objects.append({"id": e["id"], "category": e["category"], "display": e.get("display", e["category"]),
                        "type": e["geometry"]["type"], "evidence": e.get("evidence", {}).get("class"),
                        "confidence": e.get("evidence", {}).get("confidence"),
                        "views": len(e.get("evidence", {}).get("views", []) or []),
                        "dimensions": {k: fmt_dim(d) for k, d in e.get("dimensions", {}).items()},
                        "flags": e.get("flags", []), "method": e.get("provenance", {}).get("method"),
                        "openings": len(e.get("openings", [])), "inferred": len(e.get("inferred_patches", []))
                        + len(e.get("inferred_parts", [])), "tracks": e.get("tracks", [])})
    # ---- people, faces, vehicles
    people, vehicles = [], []
    for a in scene["agents"]:
        base = {"id": a["id"], "video": a["video"], "category": a["category"], "t0": a["first_t"], "t1": a["last_t"],
                "observations": a["observed_count"], "gaps": len(a.get("gaps", [])), "links": len(a.get("links", [])),
                "flags": a.get("flags", []), "summary": a.get("summary", {}), "stationary": a.get("stationary")}
        crops = []
        for c in a.get("crops", [])[:2]:
            rel = _img(case.root / "derived" / c["file"], adir, f"{a['id']}_body_{c['frame']}.jpg", 260)
            if rel:
                crops.append({"image": rel, "frame": c["frame"], "t": c["t"] + scene["time"]["offsets"].get(a["video"], 0)})
        base["crops"] = crops
        if a["group"] == "person":
            faces = []
            for f in a.get("faces", [])[:3]:
                rel = _img(case.root / "derived" / f["file"], adir, f"{a['id']}_face_{f['frame']}.png")
                if rel:
                    faces.append({"image": rel, "frame": f["frame"],
                                  "t": f["t"] + scene["time"]["offsets"].get(a["video"], 0),
                                  "height_px": round(f["face_height_px"], 1), "pose": f["pose_class"],
                                  "score": round(f["detection_score"], 2), "sharpness": round(f["sharpness"], 1),
                                  "bbox": [round(x, 1) for x in f["bbox"]]})
            base["faces"] = faces
            base["stature"] = a.get("stature")
            people.append(base)
        elif a["group"] in ("vehicle", "two_wheeler"):
            base["color"] = a.get("color")
            base["ocr"] = a.get("ocr_candidates", [])
            vehicles.append(base)
    # ---- stages / models
    stages = stage_statuses(case)
    models = manifest.get("models", [])
    # ---- warnings and limitations
    warns = scene.get("warnings", [])
    limitations = [
        "Monocular depth is estimated by a neural network; it is aligned to geometric evidence but remains an estimate.",
        "Single-view geometry is least constrained along the viewing direction.",
        "Object dimensions marked 'prior' are class-typical values, not measurements of the actual object.",
        "Inferred geometry (hatched/red) shows plausible continuations of observed surfaces, not observations.",
        "Trajectory positions use the bottom of detection boxes as ground contact; partial occlusion biases them.",
        "Face crops are original pixels without enhancement; METRICA performs no recognition or identification.",
    ]
    if scene["coordinate_system"]["status"] in ("RELATIVE", "ARBITRARY"):
        limitations.insert(0, "The scale is NOT metric: distances and speeds are in scene units and must not be read "
                              "as real-world values. Calibrate with a known distance to obtain metres.")
    data = {
        "report_id": rid, "generated": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "software_version": __version__,
        "case": scene["case"], "analyst": options.get("analyst", ""), "purpose": options.get("purpose", ""),
        "coordinate_system": scene["coordinate_system"], "status_text": STATUS_TEXT,
        "time": scene["time"], "inputs": inputs, "units": unit_rows, "objects": objects, "people": people,
        "vehicles": vehicles, "events": scene["events"], "conflicts": scene["conflicts"],
        "grouping": scene.get("grouping"),
        "measurements": ann_meas, "verifications": ann_ver, "notes": ann_notes, "snapshots": snapshots,
        "stages": stages, "models": models, "config": case.read_json("derived/config.json", {}), "warnings": warns,
        "limitations": limitations, "figures": figs, "evidence_classes": EVIDENCE_TEXT, "object_method": OBJECT_TEXT,
        "integrity": {"statement": INTEGRITY_STATEMENT, "audit_entries": chain["entries"],
                      "audit_chain_valid": chain["valid"], "audit_head": chain["head"],
                      "manifest_generated": manifest.get("generated"),
                      "outputs": len(manifest.get("outputs", []))},
        "unit": unit,
    }
    data["summary"] = executive_summary(data, scene)
    return data, rdir


def executive_summary(d: dict[str, Any], scene: dict[str, Any]) -> list[str]:
    cs = d["coordinate_system"]
    n_people = len(d["people"])
    n_veh = len(d["vehicles"])
    dur = sum(i["duration_s"] for i in d["inputs"])
    modes = ", ".join(f"{i['id']}: {i['mode']}" for i in d["inputs"])
    cats: dict[str, int] = {}
    for o in d["objects"]:
        cats[o["display"]] = cats.get(o["display"], 0) + 1
    objs = ", ".join(f"{v} x {k}" for k, v in sorted(cats.items(), key=lambda x: -x[1]))
    lines = [
        f"{len(d['inputs'])} video file(s), {dur:.1f} s in total ({modes}), were analysed locally by METRICA "
        f"{d['software_version']}. The original files were hashed (SHA-256) at intake and were not modified.",
        f"Coordinate system: {cs['status']} - {cs['status_text']}"
        + (f" Relative scale uncertainty (1 sigma): {cs['scale']['scale_sigma_rel'] * 100:.1f}%."
           if cs["scale"].get("scale_sigma_rel") else ""),
        f"The reconstructed scene contains: {objs or 'no structured objects'}.",
        f"Tracking produced {n_people} person track(s) and {n_veh} vehicle track(s); "
        f"{sum(len(p.get('faces', [])) for p in d['people'])} face observation(s) were retained for review "
        "(detection only, no identification).",
        f"The event timeline lists {len(d['events'])} machine-generated observation(s); "
        f"{sum(1 for v in d['verifications'] if v.get('verdict') == 'verified')} item(s) were verified by an analyst.",
    ]
    if d["conflicts"]:
        nc = sum(1 for c in d["conflicts"] if c.get("conflict"))
        lines.append(f"Multi-camera comparison found {nc} conflict(s) between recordings.")
    if d["warnings"]:
        lines.append(f"{len(d['warnings'])} warning(s) were recorded during processing (see Limitations and Warnings).")
    return lines
