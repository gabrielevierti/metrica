"""Machine-readable exports (JSON and CSV) and the report generation entry point."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

from ..evidence.case import Case
from ..evidence.hashing import sha256_file


def _csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


def write_exports(case: Case, data: dict[str, Any], edir: Path) -> list[str]:
    edir.mkdir(parents=True, exist_ok=True)
    scene = case.read_json("derived/scene/scene.json")
    files = []
    unit = scene["coordinate_system"]["unit"]
    status = scene["coordinate_system"]["status"]
    _csv(edir / "objects.csv", [{"id": e["id"], "category": e["category"], "geometry_type": e["geometry"]["type"],
                                 "evidence": e.get("evidence", {}).get("class"),
                                 "confidence": e.get("evidence", {}).get("confidence"),
                                 "dimensions": e.get("dimensions", {}), "unit": unit, "scale_status": status,
                                 "geometry": e["geometry"]} for e in scene["entities"]],
         ["id", "category", "geometry_type", "evidence", "confidence", "dimensions", "unit", "scale_status", "geometry"])
    rows = []
    for a in scene["agents"]:
        for s in a["samples"]:
            rows.append({"track": a["id"], "category": a["category"], "video": a["video"], "frame": s["f"],
                         "case_time_s": s["t"], "x": s["p"][0], "y": s["p"][1], "z": s["p"][2],
                         "sigma": s["s"], "detector_conf": s["conf"], "flags": s["flags"], "pose_source": s["pose"],
                         "unit": unit, "scale_status": status, "kind": "observed"})
    _csv(edir / "trajectories_observed.csv", rows, ["track", "category", "video", "frame", "case_time_s", "x", "y", "z",
                                                    "sigma", "detector_conf", "flags", "pose_source", "unit",
                                                    "scale_status", "kind"])
    rows = [{"track": a["id"], "case_time_s": s["t"], "speed": s["v"], "speed_sigma": s["s"], "unit": f"{unit}/s",
             "scale_status": status} for a in scene["agents"] for s in a.get("speed", [])]
    _csv(edir / "speeds.csv", rows, ["track", "case_time_s", "speed", "speed_sigma", "unit", "scale_status"])
    _csv(edir / "events.csv", scene["events"], ["id", "t", "t_end", "type", "description", "video", "frame", "tracks",
                                                "confidence", "machine_generated", "status"])
    rows = [{"track": a["id"], "video": a["video"], "frame": f["frame"], "case_time_s": f["t"] + scene["time"]["offsets"]
             .get(a["video"], 0), "bbox": f["bbox"], "face_height_px": f["face_height_px"], "pose_class":
             f["pose_class"], "detection_score": f["detection_score"], "sharpness": f["sharpness"],
             "crop_file": f["file"]} for a in scene["agents"] for f in a.get("faces", [])]
    _csv(edir / "face_observations.csv", rows, ["track", "video", "frame", "case_time_s", "bbox", "face_height_px",
                                                "pose_class", "detection_score", "sharpness", "crop_file"])
    _csv(edir / "measurements.csv", data["measurements"], ["id", "kind", "value", "unit", "sigma", "status", "display",
                                                           "points", "targets", "actor", "created"])
    (edir / "report_data.json").write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    shutil.copyfile(case.root / "derived" / "scene" / "scene.json", edir / "scene.json")
    if (case.root / "manifest.json").exists():
        shutil.copyfile(case.root / "manifest.json", edir / "manifest.json")
    for p in sorted(edir.iterdir()):
        files.append(p.name)
    return files


def generate_report(case: Case, options: dict[str, Any] | None = None, actor: str = "system",
                    pdf: bool = True) -> dict[str, Any]:
    from .builder import build
    from .html import render_html
    from .pdf import render_pdf

    data, rdir = build(case, options)
    html = render_html(data, rdir / "report.html")
    files = {"html": str(html.relative_to(case.root))}
    if pdf:
        p = render_pdf(data, rdir, rdir / "report.pdf")
        files["pdf"] = str(p.relative_to(case.root))
    exports = write_exports(case, data, rdir / "exports")
    files["exports"] = [f"{rdir.relative_to(case.root)}/exports/{e}" for e in exports]
    # zip for download
    zpath = shutil.make_archive(str(rdir), "zip", root_dir=rdir)
    files["zip"] = str(Path(zpath).relative_to(case.root))
    hashes = {k: sha256_file(case.root / v) for k, v in files.items() if isinstance(v, str)}
    rec = {"report_id": data["report_id"], "generated": data["generated"], "files": files, "sha256": hashes,
           "options": options or {}}
    case.write_json(f"derived/reports/{data['report_id']}/report.json", rec)
    case.audit.append("report.generated", actor=actor, target=data["report_id"],
                      details={"sha256": hashes, "options": options or {}})
    return rec
