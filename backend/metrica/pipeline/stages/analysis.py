"""Stage 6 - investigative analysis: vehicle attributes, OCR candidates, face observations, timeline."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ...analysis.timeline import crossing_events, proximity_events, stationary_events, track_events
from ...analysis.vehicles import aggregate_colors, aggregate_ocr, dominant_color
from ...geometry.calibration import resolve_global_scale
from ...log import get_logger
from ...models.ocr import SceneTextOCR
from ...models.registry import ModelUnavailable
from ..base import Context, Stage, StageResult

log = get_logger("analysis")


class AnalysisStage(Stage):
    name = "analysis"
    title = "Investigative analysis and event timeline"
    depends = ("tracking", "trajectories", "alignment")
    config_keys = ("use_ocr",)
    version = "2"
    annotation_keys = ("calibration", "sync", "alignment")

    def run(self, ctx: Context) -> StageResult:
        res = StageResult()
        cfg = ctx.config
        ocr = None
        if cfg.use_ocr:
            try:
                ocr = SceneTextOCR()
                res.models += ["ppocr_det_en", "crnn_en"]
            except ModelUnavailable as e:
                res.fallback("OCR", "disabled", str(e))
        align = ctx.case.read_json("derived/alignment/alignment.json") or {}
        offsets = align.get("offsets", {})
        units = ctx.case.read_json("derived/geometry/units.json")["units"]
        unit_of = {v: u["id"] for u in units for v in u["videos"]}
        transforms = {v: np.asarray(align.get("units", {}).get(unit_of.get(v), {}).get("transform", np.eye(4)))
                      for v in ctx.video_ids()}
        ref_unit = next(u for u in units if u["id"] == unit_of[ctx.video_ids()[0]])
        k, scale = resolve_global_scale(ref_unit["scale"], (ctx.case.annotations("calibration") or {}).get("references"))
        unit = scale["unit"]
        for v, M in transforms.items():  # k * (M p) == M' (k p) with the translation scaled by k
            M = M.copy()
            M[:3, 3] *= k
            transforms[v] = M
        vehicles: dict[str, Any] = {}
        faces: dict[str, Any] = {}
        events: list[dict[str, Any]] = []
        all_tracks: list[dict[str, Any]] = []
        vid_of: dict[str, str] = {}
        for n, vid in enumerate(ctx.video_ids()):
            ctx.check_cancel()
            ctx.progress(n / max(1, len(ctx.video_ids())), f"{vid}: vehicles, OCR, faces")
            t2d = ctx.case.read_json(f"derived/videos/{vid}/tracks2d.json", {"tracks": []})
            traj = ctx.case.read_json(f"derived/trajectories/{vid}.json", {"tracks": []})
            ing = ctx.case.read_json(f"derived/videos/{vid}/ingest.json")
            w, h = ing["frame_size"]["width"], ing["frame_size"]["height"]
            vdir = ctx.vdir(vid)
            off = float(offsets.get(vid, 0.0))
            by2d = {t["id"]: t for t in t2d["tracks"]}
            for tr in traj["tracks"]:
                t = by2d.get(tr["id"], {})
                tr = _scaled(tr, k)
                all_tracks.append(tr)
                vid_of[tr["id"]] = vid
                events += track_events(tr, vid, off, w, h)
                events += stationary_events(tr, vid, off, unit)
                for l in tr.get("links", []):
                    events.append({"type": "track_relink", "t": tr["first_t"] + off, "video": vid, "tracks": [tr["id"]],
                                   "description": f"Fragments {l['from_fragment']} and {l['to_fragment']} were joined "
                                                  f"into {tr['id']} (machine hypothesis: {l['method']}).",
                                   "confidence": float(max(0.2, l.get("score", 0.5))), "hypothesis": True})
                if t.get("group") in ("vehicle", "two_wheeler"):
                    cols, reads = [], []
                    for c in t.get("crops", []):
                        img = cv2.imread(str(vdir / c["file"]))
                        if img is None:
                            continue
                        mask = None
                        if c.get("mask"):
                            mm = cv2.imread(str(vdir / c["mask"]["file"]), cv2.IMREAD_GRAYSCALE)
                            if mm is not None and mm.shape == img.shape[:2]:
                                mask = mm > 0
                        cols.append(dominant_color(img, mask))
                        if ocr is not None and img.shape[1] >= 90:
                            for r in ocr.read(img):
                                reads.append({**r, "frame": c["frame"], "crop": c["file"]})
                    vehicles[tr["id"]] = {"video": vid, "category": tr["category"], "color": aggregate_colors(cols),
                                          "ocr_candidates": aggregate_ocr(reads), "crops": len(t.get("crops", [])),
                                          "stationary": tr.get("stationary"), "static_entity": tr.get("static_entity")}
                    for oc in vehicles[tr["id"]]["ocr_candidates"]:
                        events.append({"type": "text_candidate", "t": t["t0"] + off, "video": vid,
                                       "frame": oc["frames"][0], "tracks": [tr["id"]],
                                       "description": f"Possible text '{oc['text']}' read on {tr['id']} "
                                                      f"(candidate, confidence {oc['confidence']:.2f}; verify).",
                                       "confidence": oc["confidence"], "requires_verification": True})
                if t.get("faces"):
                    best = max(t["faces"], key=lambda f: f["score"])
                    faces[tr["id"]] = {"video": vid, "observations": t["faces"],
                                       "best_face_height_px": best["face_height_px"],
                                       "best_pose": best["pose_class"]}
                    events.append({"type": "face_observed", "t": best["t"] + off, "video": vid, "frame": best["frame"],
                                   "tracks": [tr["id"]],
                                   "description": f"Face of {tr['id']} visible ({best['pose_class']}, "
                                                  f"{best['face_height_px']:.0f} px tall).",
                                   "confidence": float(best["detection_score"])})
        ctx.progress(0.8, "proximity and path crossings")
        thr = 1.5  # metres, or scene units in RELATIVE scenes (documented)
        events += proximity_events(all_tracks, vid_of, offsets, transforms, unit, max_dist=thr)
        events += crossing_events(all_tracks, vid_of, offsets, transforms)
        events.sort(key=lambda e: (e["t"], e["type"]))
        for i, e in enumerate(events, start=1):
            e["id"] = f"E{i:04d}"
            e["machine_generated"] = True
            e["status"] = "unreviewed"
        doc = {"scale": scale, "global_scale_factor": k, "vehicles": vehicles, "faces": faces, "events": events,
               "stats": {"events": len(events), "vehicles": len(vehicles), "people_with_faces": len(faces),
                         "face_observations": sum(len(f["observations"]) for f in faces.values()),
                         "ocr_candidates": sum(len(v["ocr_candidates"]) for v in vehicles.values())},
               "policy": "Observations only. No identity attribution, no inference of intent, sensitive attributes or "
                         "wrongdoing. OCR and re-identification links are hypotheses requiring human verification."}
        p = ctx.case.write_json("derived/analysis/analysis.json", doc)
        res.outputs.append(p)
        res.metrics = doc["stats"]
        return res


def _scaled(tr: dict[str, Any], k: float) -> dict[str, Any]:
    """Apply the global (analyst) scale factor to a trajectory's spatial quantities."""
    if k == 1.0:
        return tr
    tr = dict(tr)
    tr["samples"] = [{**s, "p": [v * k for v in s["p"]], "s": s["s"] * k if s["s"] is not None else None}
                     for s in tr["samples"]]
    if tr.get("smoothed"):
        tr["smoothed"] = [{**s, "p": [v * k for v in s["p"]]} for s in tr["smoothed"]]
        tr["speed"] = [{**s, "v": s["v"] * k, "s": s["s"] * k} for s in tr["speed"]]
        sm = dict(tr["summary"])
        for key in ("path_length_observed", "displacement", "speed_median", "speed_max", "position_sigma_median"):
            if sm.get(key) is not None:
                sm[key] *= k
        tr["summary"] = sm
    return tr
