"""Stage D/5 - detection, multi-object tracking, best crops, face detection, instance masks."""
from __future__ import annotations

import gzip
import heapq
import json
from itertools import count
from typing import Any

import cv2
import numpy as np

from ...ingest.decode import FrameSource
from ...log import get_logger
from ...models.detection import make_detector
from ...models.faces import YuNetFaces, face_pose_class
from ...models.registry import ModelUnavailable, available
from ...models.segmentation import EfficientSAMPrompt, GrabCutMask
from ...semantics.labels import TRACKED_CATEGORIES
from ...tracking.appearance import signature
from ...tracking.linking import link_fragments, renumber
from ...tracking.tracker import Tracker
from ..base import Context, Stage, StageResult

log = get_logger("tracking")
_tie = count()


def _sharpness(img: np.ndarray) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if min(g.shape) < 8:
        return 0.0
    g = cv2.resize(g, (64, int(64 * g.shape[0] / g.shape[1])) if g.shape[1] > 64 else (g.shape[1], g.shape[0]))
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def _crop(frame: np.ndarray, box: list[float], pad: float) -> tuple[np.ndarray, list[int]]:
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    X1, Y1 = int(max(0, x1 - pad * bw)), int(max(0, y1 - pad * bh))
    X2, Y2 = int(min(W, x2 + pad * bw)), int(min(H, y2 + pad * bh))
    return frame[Y1:Y2, X1:X2].copy(), [X1, Y1, X2, Y2]


class _Keeper:
    """Top-k items per key by score (min-heap)."""

    def __init__(self, k: int):
        self.k = k
        self.h: dict[Any, list] = {}

    def offer(self, key, score: float, item: dict) -> None:
        heap = self.h.setdefault(key, [])
        entry = (score, next(_tie), item)
        if len(heap) < self.k:
            heapq.heappush(heap, entry)
        elif score > heap[0][0]:
            heapq.heapreplace(heap, entry)

    def best(self, keys, k: int) -> list[dict]:
        items = [e for key in keys for e in self.h.get(key, [])]
        items.sort(key=lambda e: -e[0])
        return [e[2] | {"score": e[0]} for e in items[:k]]


def _reid_embeddings(raw: list[dict], body: Any, res: StageResult) -> dict[int, np.ndarray] | None:
    """Optional appearance embeddings (Youtu ReID) of each person fragment's best crop.

    Used only as an additional similarity cue when re-linking fragments of the same video; the
    embeddings are never stored or compared with anything outside this video."""
    if not available("youtu_reid"):
        return None
    try:
        from ...models.reid import YoutuReID

        reid = YoutuReID()
        ids, crops = [], []
        for tr in raw:
            if tr["group"] != "person":
                continue
            best = body.best([tr["raw_id"]], 1)
            if best and best[0].get("img") is not None and best[0]["img"].size:
                ids.append(tr["raw_id"])
                crops.append(best[0]["img"])
        if not crops:
            return None
        vecs = np.concatenate([reid.embed(crops[k:k + 16]) for k in range(0, len(crops), 16)])
        res.models.append("youtu_reid")
        return dict(zip(ids, vecs))
    except Exception as e:  # noqa: BLE001
        res.fallback("track re-linking", "colour-histogram appearance only",
                     f"Re-identification embeddings unavailable ({type(e).__name__}).")
        return None


class DetectionTrackingStage(Stage):
    name = "tracking"
    title = "Detection and multi-object tracking"
    depends = ("ingest",)
    config_keys = ("analysis_fps", "process_all_frames", "detector", "detector_conf", "detector_nms_iou",
                   "track_max_gap_s", "reid_link_max_gap_s", "min_track_hits", "use_faces",
                   "use_instance_masks", "masks_per_track", "max_mask_calls")
    version = "4"

    def run(self, ctx: Context) -> StageResult:
        res = StageResult()
        cfg = ctx.config
        try:
            detector = make_detector(cfg.detector, cfg.detector_conf, cfg.detector_nms_iou)
            res.models.append(cfg.detector)
        except ModelUnavailable as e:
            res.fallback("detection", "none", f"{e}. No people/vehicles will be tracked; the scene geometry is "
                                              "still reconstructed.")
            for vid in ctx.video_ids():
                p = ctx.case.write_json(f"derived/videos/{vid}/tracks2d.json",
                                        {"video_id": vid, "tracks": [], "stride": 1, "detector": None})
                res.outputs.append(p)
            return res
        faces = None
        if cfg.use_faces:
            try:
                faces = YuNetFaces()
                res.models.append("yunet")
            except ModelUnavailable as e:
                res.fallback("face detection", "disabled", str(e))
        vids = ctx.video_ids()
        for k, vid in enumerate(vids):
            def sub(f, m, k=k, vid=vid):
                ctx.progress((k + f) / len(vids), f"{vid}: {m}")
            self._run_video(ctx, vid, detector, faces, res, sub)
        return res

    # ------------------------------------------------------------------
    def _run_video(self, ctx: Context, vid: str, detector, faces, res: StageResult, progress) -> None:
        cfg = ctx.config
        ing = ctx.case.read_json(f"derived/videos/{vid}/ingest.json")
        fps = ing["timing"]["fps_measured"] or 25.0
        n_frames = ing["timing"]["frames"]
        stride = 1 if cfg.process_all_frames else max(1, int(round(fps / cfg.analysis_fps)))
        bg = cv2.imread(str(ctx.vdir(vid) / "background.jpg")) if ing.get("background") else None
        max_gap_steps = int(np.ceil(cfg.track_max_gap_s * fps / stride))
        tracker = Tracker(max_gap_steps=max_gap_steps, min_hits=cfg.min_track_hits)
        body = _Keeper(8)
        face_keep = _Keeper(8)
        det_log: list[dict[str, Any]] = []
        det_id = 0
        src = FrameSource(ctx.case.video_path(vid))
        steps = 0
        for i, t, frame in src.iter_bgr(every=stride):
            if steps % 25 == 0:
                ctx.check_cancel()
                progress(0.85 * i / max(n_frames, 1), f"frame {i}/{n_frames}")
            steps += 1
            dets = detector(frame)
            H, W = frame.shape[:2]
            bgm = bg if (bg is not None and bg.shape == frame.shape) else None
            for d in dets:
                d["det_id"] = det_id
                det_id += 1
                if d["category"] in TRACKED_CATEGORIES:
                    d["feat"] = signature(frame, d["bbox"], background=bgm)
            tracked = [d for d in dets if d["category"] in TRACKED_CATEGORIES]
            tracker.update(i, t, tracked)
            for d in tracked:
                tid = d.get("track_id")
                if tid is None:
                    continue
                x1, y1, x2, y2 = d["bbox"]
                bh = y2 - y1
                truncated = x1 < 2 or y1 < 2 or x2 > W - 2 or y2 > H - 2
                crop, region = _crop(frame, d["bbox"], 0.12)
                sharp = _sharpness(crop)
                score = d["conf"] * min(1.0, bh / 220.0) * min(1.0, sharp / 150.0 + 0.2) * (0.6 if truncated else 1.0)
                body.offer(tid, score, {"frame": i, "t": t, "bbox": d["bbox"], "region": region, "img": crop,
                                        "conf": d["conf"], "sharpness": sharp, "truncated": truncated,
                                        "category": d["category"]})
                n_faces = len(face_keep.h.get(tid, []))
                if faces is not None and d["category"] == "person" and bh >= 40 and \
                        (n_faces < 3 or steps % 3 == 0):
                    hw = (x2 - x1)
                    head = (x1 - 0.15 * hw, y1 - 0.05 * bh, x2 + 0.15 * hw, y1 + 0.38 * bh)
                    for f in faces.detect(frame, head):
                        fx1, fy1, fx2, fy2 = f["bbox"]
                        fh = fy2 - fy1
                        if fh < 6:
                            continue
                        fcrop, fregion = _crop(frame, f["bbox"], 0.35)
                        fsharp = _sharpness(frame[int(max(0, fy1)):int(fy2), int(max(0, fx1)):int(fx2)])
                        pose, ratio = face_pose_class(f["landmarks"])
                        fscore = f["score"] * min(1.0, fh / 60.0) * min(1.0, fsharp / 120.0 + 0.25) * \
                            (1.0 if pose == "frontal" else 0.75 if pose == "partially turned" else 0.5)
                        face_keep.offer(tid, fscore, {
                            "frame": i, "t": t, "bbox": f["bbox"], "landmarks": f["landmarks"],
                            "detection_score": f["score"], "region": fregion, "img": fcrop,
                            "face_height_px": fh, "sharpness": fsharp, "pose_class": pose,
                            "yaw_ratio": ratio, "visibility": float(min(1.0, fh / max(0.13 * bh, 1))),
                            "upsample_for_detection": f["upsample"], "person_bbox": d["bbox"]})
            det_log.append({"frame": i, "t": round(t, 4),
                            "dets": [{"b": [round(v, 1) for v in d["bbox"]], "c": round(d["conf"], 3),
                                      "l": d["label"], "k": d["category"], "tr": d.get("track_id"),
                                      "id": d["det_id"]} for d in dets]})

        progress(0.86, "linking track fragments")
        raw = tracker.finish(gap_frames=stride)
        linked = link_fragments(raw, cfg.reid_link_max_gap_s, embed=_reid_embeddings(raw, body, res))
        tracks = renumber(linked, vid)
        raw_to_final = {}
        for tr in tracks:
            for rid in tr.get("fragments", [tr["raw_id"]]):
                raw_to_final[rid] = tr["id"]
        for fr in det_log:
            for d in fr["dets"]:
                if d["tr"] is not None:
                    d["tr"] = raw_to_final.get(d["tr"])  # unconfirmed tentative tracks -> None
        with gzip.open(ctx.vdir(vid) / "detections.json.gz", "wt", encoding="utf-8") as f:
            json.dump({"video_id": vid, "stride": stride, "detector": cfg.detector, "frames": det_log}, f)
        res.outputs.append(ctx.vdir(vid) / "detections.json.gz")

        # ---- crops, faces, masks
        cdir = ctx.vdir(vid, "crops")
        seg = None
        mask_budget = cfg.max_mask_calls if cfg.use_instance_masks else 0
        if mask_budget:
            if available("efficientsam_ti"):
                seg = EfficientSAMPrompt()
                res.models.append("efficientsam_ti")
            else:
                seg = GrabCutMask()
                res.fallback("instance masks", "GrabCut (classical colour model)",
                             "EfficientSAM not downloaded; masks are coarser and marked as classical.")
        mask_calls = 0
        order = sorted(tracks, key=lambda tr: -len(tr["observations"]))
        for n, tr in enumerate(order):
            if n % 10 == 0:
                ctx.check_cancel()
                progress(0.88 + 0.1 * n / max(len(order), 1), f"crops and masks ({n}/{len(order)})")
            keys = tr.get("fragments", [tr["raw_id"]])
            crops = []
            for j, c in enumerate(body.best(keys, 4)):
                name = f"{tr['id']}_body_{c['frame']:07d}.jpg"
                cv2.imwrite(str(cdir / name), c["img"], [cv2.IMWRITE_JPEG_QUALITY, 95])
                rec = {k: c[k] for k in ("frame", "t", "bbox", "region", "conf", "sharpness", "truncated",
                                         "category", "score")}
                rec["file"] = f"crops/{name}"
                if seg is not None and j < cfg.masks_per_track and mask_calls < mask_budget:
                    X1, Y1 = c["region"][:2]
                    lb = [c["bbox"][0] - X1, c["bbox"][1] - Y1, c["bbox"][2] - X1, c["bbox"][3] - Y1]
                    m, ms = seg.mask_for_box(c["img"], lb)
                    mask_calls += 1
                    if m.any():
                        mname = f"{tr['id']}_mask_{c['frame']:07d}.png"
                        cv2.imwrite(str(cdir / mname), (m * 255).astype(np.uint8))
                        rec["mask"] = {"file": f"crops/{mname}", "method": seg.key, "score": round(ms, 3),
                                       "frame_region": c["region"]}
                crops.append(rec)
            tr["crops"] = crops
            fl = []
            if tr["category"] == "person":
                for c in face_keep.best(keys, 5):
                    name = f"{tr['id']}_face_{c['frame']:07d}.png"
                    cv2.imwrite(str(cdir / name), c["img"])  # PNG: lossless copy of the original pixels
                    rec = {k: c[k] for k in c if k != "img"}
                    rec["file"] = f"crops/{name}"
                    rec["occlusion"] = "not assessed"
                    fl.append(rec)
            tr["faces"] = fl
            tr.pop("appearance", None)
        doc = {"video_id": vid, "stride": stride, "fps": fps, "detector": cfg.detector,
               "analysis_frames": steps, "tracks": tracks}
        p = ctx.case.write_json(f"derived/videos/{vid}/tracks2d.json", doc)
        res.outputs.append(p)
        n_p = sum(1 for t in tracks if t["group"] == "person")
        n_v = sum(1 for t in tracks if t["group"] in ("vehicle", "two_wheeler"))
        n_links = sum(len(t["links"]) for t in tracks)
        res.metrics[vid] = {"analysis_frames": steps, "stride": stride, "tracks": len(tracks), "people": n_p,
                            "vehicles": n_v, "relinks": n_links,
                            "face_observations": sum(len(t["faces"]) for t in tracks), "mask_calls": mask_calls}
        log.info(f"{vid}: {steps} frames analysed (stride {stride}), {len(tracks)} tracks "
                 f"({n_p} people, {n_v} vehicles), {n_links} re-links, {mask_calls} masks")
