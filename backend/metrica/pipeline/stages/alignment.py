"""Stage 8 - multi-video grouping, synchronisation, spatial alignment and conflict detection."""
from __future__ import annotations

from typing import Any

import numpy as np

from ...log import get_logger
from ...multiview.alignment import (
    find_conflicts,
    landmark_correspondences,
    ransac_similarity_4dof,
    similarity_2d,
    to_4x4,
    track_alignment,
)
from ...multiview.sync import audio_offset, visual_offset
from ...reconstruction.views import load_views
from ..base import Context, Stage, StageResult

log = get_logger("alignment")


class AlignmentStage(Stage):
    name = "alignment"
    title = "Multi-video synchronisation and alignment"
    depends = ("trajectories", "objects", "reconstruction", "ingest")
    config_keys = ("audio_sync", "visual_sync", "manual_offsets", "manual_align_points")
    version = "4"
    annotation_keys = ("sync", "alignment")

    def run(self, ctx: Context) -> StageResult:
        res = StageResult()
        cfg = ctx.config
        vids = ctx.video_ids()
        ref = vids[0]
        units = ctx.case.read_json("derived/geometry/units.json")["units"]
        unit_of = {v: u["id"] for u in units for v in u["videos"]}
        doc: dict[str, Any] = {"reference_video": ref, "reference_unit": unit_of.get(ref), "sync": {}, "units": {},
                               "grouping": None, "conflicts": []}
        ann_sync = ctx.case.annotations("sync") or {}
        ann_align = ctx.case.annotations("alignment") or {}
        ing = {v: ctx.case.read_json(f"derived/videos/{v}/ingest.json") for v in vids}
        trajs = {v: (ctx.case.read_json(f"derived/trajectories/{v}.json") or {}).get("tracks", []) for v in vids}
        # ---------------------------------------------------------------- synchronisation
        offsets = {ref: 0.0}
        doc["sync"][ref] = {"offset_s": 0.0, "method": "reference video", "confidence": 1.0}
        for v in vids[1:]:
            ctx.check_cancel()
            cands = []
            manual = ann_sync.get(v, {}).get("offset_s") if isinstance(ann_sync.get(v), dict) else None
            if manual is None and v in cfg.manual_offsets:
                manual = cfg.manual_offsets[v]
            if manual is not None:
                offsets[v] = float(manual)
                doc["sync"][v] = {"offset_s": float(manual), "method": "manual (analyst)", "confidence": 1.0}
                continue
            if cfg.audio_sync and ing[ref]["probe"]["has_audio"] and ing[v]["probe"]["has_audio"]:
                a = audio_offset(ctx.case.video_path(ref), ctx.case.video_path(v))
                if a:
                    cands.append(a)
            if cfg.visual_sync:
                vo = visual_offset(ing[ref]["quality"]["samples"], ing[v]["quality"]["samples"])
                if vo:
                    cands.append(vo)
            best = max(cands, key=lambda c: c["confidence"]) if cands else None
            ct = (ing[ref]["probe"].get("creation_time"), ing[v]["probe"].get("creation_time"))
            if best and best["confidence"] >= 0.3:
                offsets[v] = best["offset_s"]
                doc["sync"][v] = {**best, "candidates": cands, "creation_times": ct}
            else:
                offsets[v] = 0.0
                doc["sync"][v] = {"offset_s": 0.0, "method": "none (unsynchronised)", "confidence": 0.0,
                                  "candidates": cands, "creation_times": ct,
                                  "warning": "No reliable synchronisation evidence; timelines are shown unaligned. "
                                             "Set the offset manually in the timeline."}
                if len(vids) > 1:
                    res.warn(f"{v}: could not be synchronised automatically.")
        # ---------------------------------------------------------------- spatial alignment
        ref_unit = unit_of.get(ref)
        transforms: dict[str, np.ndarray] = {}
        views_cache: dict[str, list] = {}

        def views_of(uid):
            if uid not in views_cache:
                u = next(x for x in units if x["id"] == uid)
                recon = ctx.case.read_json(f"derived/reconstruction/{uid}/reconstruction.json")
                views_cache[uid] = load_views(ctx.case, u, recon) if recon and recon.get("layers") else []
            return views_cache[uid]

        side_offset = 0.0
        for u in units:
            uid = u["id"]
            if uid == ref_unit:
                doc["units"][uid] = {"transform": np.eye(4).tolist(), "method": "reference unit", "status": "reference"}
                continue
            ctx.check_cancel()
            rec: dict[str, Any] | None = None
            pts = ann_align.get(uid) or next((cfg.manual_align_points[v] for v in u["videos"]
                                              if v in cfg.manual_align_points), None)
            if pts and len(pts) >= 2:
                src = np.array([p["src"] for p in pts], float)
                dst = np.array([p["dst"] for p in pts], float)
                s, R, t, rms = similarity_2d(src, dst, allow_scale=len(pts) >= 3)
                rec = {"transform": to_4x4(s, R, t).tolist(), "method": "analyst point pairs", "pairs": len(pts),
                       "rms": rms, "scale": s, "status": "aligned"}
            if rec is None and views_of(uid) and views_of(ref_unit):
                A3, B3, st = landmark_correspondences(views_of(ref_unit), views_of(uid))
                r = ransac_similarity_4dof(B3, A3, thr=0.6)
                # a 4-DOF model fitted to a minority of matches is usually a coincidence: require a
                # clear majority-like consensus, enough support and a plausible relative scale
                if r and not (r["inliers"] >= 15 and r["inliers"] >= 0.4 * r["matches"] and r["rms"] < 0.8
                              and 0.25 < r["scale"] < 4.0):
                    res.warn(f"{uid}: shared-landmark alignment rejected ({r['inliers']} of {r['matches']} "
                             f"matches consistent, rms {r['rms']:.2f}); add analyst point pairs to align it.")
                    doc.setdefault("rejected_alignments", {})[uid] = {k: v for k, v in r.items() if k != "M"}
                    r = None
                if r:
                    rec = {"transform": r["M"].tolist(), "method": "shared visual landmarks (SIFT + depth, 4-DOF RANSAC)",
                           "inliers": r["inliers"], "matches": r["matches"], "rms": r["rms"], "scale": r["scale"],
                           "yaw_deg": r["yaw_deg"], "status": "aligned", "image_pairs": st}
            if rec is None:
                tr_ref = trajs.get(ref, [])
                tr_u = [t for v in u["videos"] for t in trajs.get(v, [])]
                off_u = offsets.get(u["videos"][0], 0.0)
                r = track_alignment(tr_ref, tr_u, 0.0, off_u)
                if r and r["support"] >= 1.0:
                    rec = {"transform": r["M"].tolist(), "method": "shared tracks (same subject seen by both cameras; "
                           "hypothesis)", "rms": r["rms"], "scale": r["scale"], "hypotheses": r["hypotheses"],
                           "pair": r["pair"], "status": "aligned"}
            if rec is None:
                side_offset += 40.0
                M = np.eye(4)
                M[0, 3] = side_offset
                rec = {"transform": M.tolist(), "method": "none", "status": "unaligned",
                       "warning": "No shared evidence found: this unit is placed beside the reference for viewing only. "
                                  "Positions across units are NOT comparable. Add point pairs to align it."}
                res.warn(f"{uid}: could not be aligned with the reference unit.")
            doc["units"][uid] = rec
        for uid, rec in doc["units"].items():
            transforms_u = np.asarray(rec["transform"])
            for v in next(x for x in units if x["id"] == uid)["videos"]:
                transforms[v] = transforms_u
        # ---------------------------------------------------------------- conflicts
        aligned = {v: transforms[v] for v in vids if doc["units"][unit_of[v]]["status"] in ("reference", "aligned")}
        if len(aligned) > 1:
            doc["conflicts"] = find_conflicts({v: trajs[v] for v in aligned}, aligned, offsets)
            n_conf = sum(1 for c in doc["conflicts"] if c["conflict"])
            if n_conf:
                res.warn(f"{n_conf} cross-camera conflict(s): the same subject appears at different positions.")
        doc["offsets"] = offsets
        doc["grouping"] = _grouping(ctx, vids, unit_of, doc, ing)
        res.metrics = {"videos": len(vids), "units": len(units),
                       "aligned_units": sum(1 for r in doc["units"].values() if r["status"] != "unaligned"),
                       "synchronised": sum(1 for v in vids[1:] if doc["sync"][v]["confidence"] > 0)}
        p = ctx.case.write_json("derived/alignment/alignment.json", doc)
        res.outputs.append(p)
        return res


def _thumb_similarity(ctx: Context, a: str, b: str) -> float | None:
    """Colour-histogram correlation of the two videos' thumbnails (weak context only)."""
    import cv2
    ims = []
    for v in (a, b):
        p = ctx.case.root / "derived" / "videos" / v / "thumb.jpg"
        im = cv2.imread(str(p)) if p.exists() else None
        if im is None:
            return None
        h = cv2.calcHist([cv2.cvtColor(im, cv2.COLOR_BGR2HSV)], [0, 1], None, [24, 16], [0, 180, 0, 256])
        ims.append(cv2.normalize(h, None).flatten())
    return float(cv2.compareHist(ims[0], ims[1], cv2.HISTCMP_CORREL))


def _grouping(ctx: Context, vids: list[str], unit_of: dict[str, str], doc: dict[str, Any],
              ing: dict[str, Any]) -> dict[str, Any]:
    """Suggest which videos record the same scene. Evidence per pair, union-find on strong evidence.

    Strong evidence: reconstructed jointly (shared SfM unit), spatially aligned through shared visual
    landmarks or analyst point pairs, or synchronised with confidence >= 0.5. Weak context (shown, never
    used to group): container creation times (device clocks are unreliable) and thumbnail colour similarity.
    """
    parent = {v: v for v in vids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = []
    for i, a in enumerate(vids):
        for b in vids[i + 1:]:
            strong, weak = [], []
            if unit_of.get(a) == unit_of.get(b):
                strong.append("reconstructed jointly (shared structure-from-motion features)")
            else:
                ua, ub = doc["units"].get(unit_of.get(a), {}), doc["units"].get(unit_of.get(b), {})
                for u in (ua, ub):
                    m = u.get("method", "")
                    if u.get("status") == "aligned" and ("landmark" in m or "analyst" in m):
                        strong.append(f"spatially aligned: {m}")
                    elif u.get("status") == "aligned" and "tracks" in m:
                        weak.append(f"aligned only through shared-track hypothesis: {m}")
            ref = vids[0]
            if a == ref and doc["sync"].get(b, {}).get("confidence", 0) >= 0.5:
                strong.append(f"synchronised ({doc['sync'][b]['method']}, confidence {doc['sync'][b]['confidence']:.2f})")
            ca, cb = ing[a]["probe"].get("creation_time"), ing[b]["probe"].get("creation_time")
            if ca and cb:
                weak.append(f"container creation times {ca} / {cb} (device clocks are not verified)")
            sim = _thumb_similarity(ctx, a, b)
            if sim is not None:
                weak.append(f"thumbnail colour similarity {sim:.2f}")
            if strong:
                parent[find(a)] = find(b)
            pairs.append({"a": a, "b": b, "strong_evidence": strong, "weak_context": weak,
                          "suggestion": "same scene" if strong else "undetermined"})
    groups: dict[str, list[str]] = {}
    for v in vids:
        groups.setdefault(find(v), []).append(v)
    return {"groups": sorted(groups.values(), key=lambda g: vids.index(g[0])), "pairs": pairs,
            "note": "Machine suggestion. Videos are grouped only on strong geometric or synchronisation evidence; "
                    "the analyst should confirm that the recordings show the same incident."}
