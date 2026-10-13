"""Multi-object tracker (ByteTrack-style two-stage association with Kalman prediction).

* Stage 1 matches confident detections to tracks on IoU (+ appearance when available).
* Stage 2 matches remaining low-confidence detections to still-unmatched tracks, which
  keeps tracks alive through partial occlusion without starting spurious new tracks.
* Association is restricted to compatible class groups (a car never continues a person).
* Every interruption, ambiguous crossing and appearance jump is *recorded* on the track
  (``gaps`` and ``flags``) instead of being hidden.

Missing observations are never filled in here; interpolation, if any, happens later and
is labelled as such.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from .appearance import distance as app_distance
from .kalman import BoxKalman

GROUP_OF = {"person": "person", "car": "vehicle", "truck": "vehicle", "bus": "vehicle", "train": "vehicle",
            "motorcycle": "two_wheeler", "bicycle": "two_wheeler", "animal": "animal"}


def group_of(category: str) -> str:
    return GROUP_OF.get(category, category)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


class _Track:
    def __init__(self, tid: int, det: dict, frame: int, t: float):
        self.id = tid
        self.group = group_of(det["category"])
        self.kf = BoxKalman(np.asarray(det["bbox"], float))
        self.obs: list[dict] = []
        self.votes: dict[str, float] = {}
        self.feat: np.ndarray | None = None
        self.last_frame = frame
        self.last_step = 0
        self.flags: list[dict] = []
        self.add(det, frame, t)

    def add(self, det: dict, frame: int, t: float, step: int | None = None) -> None:
        if self.obs:
            self.kf.update(np.asarray(det["bbox"], float))
        f = det.get("feat")
        if f is not None and np.sum(f) > 0:
            if self.feat is not None:
                d = app_distance(self.feat, f)
                if d > 0.5 and len(self.obs) > 3:
                    self.flags.append({"type": "appearance_change", "frame": frame, "t": t,
                                       "distance": round(d, 3),
                                       "note": "Large change in colour appearance: possible identity switch "
                                               "or lighting change."})
            self.feat = f.copy() if self.feat is None else (0.8 * self.feat + 0.2 * f)
        self.votes[det["category"]] = self.votes.get(det["category"], 0.0) + det["conf"]
        o = {k: det[k] for k in ("bbox", "conf", "label", "category") if k in det}
        o["frame"], o["t"] = frame, t
        if "det_id" in det:
            o["det_id"] = det["det_id"]
        self.obs.append(o)
        self.last_frame = frame
        if step is not None:
            self.last_step = step

    @property
    def category(self) -> str:
        return max(self.votes, key=self.votes.get)


class Tracker:
    def __init__(self, max_gap_steps: int, min_hits: int = 3, high_thr: float = 0.5, low_thr: float = 0.1,
                 iou_gate: float = 0.15, app_weight: float = 0.3):
        self.max_gap = max(1, int(max_gap_steps))
        self.min_hits = min_hits
        self.high, self.low = high_thr, low_thr
        self.iou_gate = iou_gate
        self.app_weight = app_weight
        self.active: list[_Track] = []
        self.finished: list[_Track] = []
        self.next_id = 1
        self.step = 0
        self.interactions: list[dict] = []

    def _cost(self, tracks: list[_Track], dets: list[dict], pred: np.ndarray) -> np.ndarray:
        boxes = np.array([d["bbox"] for d in dets], float).reshape(-1, 4)
        cost = 1.0 - iou_matrix(pred, boxes)
        # centre-distance fallback for small, fast boxes whose IoU drops to zero between samples
        pc = (pred[:, :2] + pred[:, 2:]) / 2
        dc = (boxes[:, :2] + boxes[:, 2:]) / 2
        ph = np.maximum(pred[:, 3] - pred[:, 1], 1)
        dist = np.linalg.norm(pc[:, None] - dc[None], axis=2) / ph[:, None]
        cost = np.minimum(cost, np.where(dist < 0.5, 0.55 + dist / 2, 1.0))
        for i, t in enumerate(tracks):
            for j, d in enumerate(dets):
                if group_of(d["category"]) != t.group:
                    cost[i, j] = 1.0
                elif cost[i, j] < 1.0 and t.feat is not None and d.get("feat") is not None:
                    a = app_distance(t.feat, d["feat"])
                    cost[i, j] = 1.0 if a > 0.6 else (1 - self.app_weight) * cost[i, j] + self.app_weight * a
        return cost

    def _match(self, tracks, dets, pred, gate):
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))
        cost = self._cost(tracks, dets, pred)
        r, c = linear_sum_assignment(cost)
        matches, ut, ud = [], set(range(len(tracks))), set(range(len(dets)))
        for i, j in zip(r, c):
            if cost[i, j] <= gate:
                matches.append((i, j))
                ut.discard(i)
                ud.discard(j)
        return matches, sorted(ut), sorted(ud)

    def update(self, frame: int, t: float, dets: list[dict]) -> None:
        self.step += 1
        for tr in self.active:
            tr.kf.predict(1.0)  # one step per analysed frame; missed steps keep predicting
        high = [d for d in dets if d["conf"] >= self.high]
        low = [d for d in dets if self.low <= d["conf"] < self.high]
        tracks = list(self.active)
        pred = np.array([tr.kf.box for tr in tracks]).reshape(-1, 4)
        m1, ut, ud = self._match(tracks, high, pred, 1.0 - self.iou_gate)
        for i, j in m1:
            tracks[i].add(high[j], frame, t, self.step)
            high[j]["track_id"] = tracks[i].id
        rest = [tracks[i] for i in ut]
        pred2 = np.array([tr.kf.box for tr in rest]).reshape(-1, 4)
        m2, ut2, _ = self._match(rest, low, pred2, 0.5)
        for i, j in m2:
            rest[i].add(low[j], frame, t, self.step)
            low[j]["track_id"] = rest[i].id
        for j in ud:  # new tracks only from confident detections
            d = high[j]
            tr = _Track(self.next_id, d, frame, t)
            tr.last_step = self.step
            d["track_id"] = tr.id
            self.active.append(tr)
            self.next_id += 1
        # record close interactions (where identity swaps are most likely)
        confirmed = [tr for tr in self.active if tr.last_frame == frame]
        if len(confirmed) > 1:
            bx = np.array([tr.kf.box for tr in confirmed])
            ov = iou_matrix(bx, bx)
            ii, jj = np.where(np.triu(ov, 1) > 0.3)
            for a, b in zip(ii, jj):
                if confirmed[a].group == confirmed[b].group:
                    self.interactions.append({"frame": frame, "t": t, "tracks": [confirmed[a].id, confirmed[b].id],
                                              "iou": float(ov[a, b])})
        alive = []
        for tr in self.active:
            (alive if self.step - tr.last_step <= self.max_gap else self.finished).append(tr)
        self.active = alive

    def finish(self, frame_times: dict[int, float] | None = None, gap_frames: int = 1) -> list[dict[str, Any]]:
        out = []
        for tr in self.finished + self.active:
            if len(tr.obs) < self.min_hits:
                continue
            out.append(self._export(tr, gap_frames))
        out.sort(key=lambda d: d["first_frame"])
        # interactions -> flags on both tracks
        by_id = {d["raw_id"]: d for d in out}
        for it in self.interactions:
            for a in it["tracks"]:
                if a in by_id:
                    others = [b for b in it["tracks"] if b != a]
                    fl = by_id[a]["flags"]
                    if not fl or fl[-1].get("type") != "close_interaction" or it["frame"] - fl[-1]["frame_end"] > 2 * gap_frames:
                        fl.append({"type": "close_interaction", "frame": it["frame"], "frame_end": it["frame"],
                                   "t": it["t"], "with_raw": others,
                                   "note": "Overlapping with a same-class subject: identity may be ambiguous here."})
                    else:
                        fl[-1]["frame_end"] = it["frame"]
        return out

    @staticmethod
    def _export(tr: _Track, gap_frames: int) -> dict[str, Any]:
        frames = [o["frame"] for o in tr.obs]
        gaps = []
        for a, b in zip(tr.obs[:-1], tr.obs[1:]):
            if b["frame"] - a["frame"] > 1.5 * gap_frames:
                gaps.append({"after_frame": a["frame"], "before_frame": b["frame"], "t0": a["t"], "t1": b["t"],
                             "duration_s": round(b["t"] - a["t"], 3), "reason": "not detected (occlusion, "
                             "missed detection or out of view)"})
        return {"raw_id": tr.id, "category": tr.category, "group": tr.group, "observations": tr.obs,
                "first_frame": frames[0], "last_frame": frames[-1], "t0": tr.obs[0]["t"], "t1": tr.obs[-1]["t"],
                "gaps": gaps, "flags": tr.flags, "links": [],
                "appearance": tr.feat.tolist() if tr.feat is not None else None}
