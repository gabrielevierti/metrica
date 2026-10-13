"""Chronological, neutral, source-linked event timeline.

Events describe *observations* ("first observed", "came within 1.2 m of") and never
intent or wrongdoing. Each event links to its video, frame(s) and tracks, carries a
machine-generated flag and a confidence, and can be verified or rejected by an analyst.
"""
from __future__ import annotations

from typing import Any

import numpy as np

GROUP_NOUN = {"person": "Person", "vehicle": "Vehicle", "two_wheeler": "Two-wheeler", "animal": "Animal"}


def _noun(tr: dict[str, Any]) -> str:
    return GROUP_NOUN.get(tr["group"], tr["category"].replace("_", " ").capitalize())


def _fmt(d: float, unit: str) -> str:
    return f"{d:.1f} {unit}"


def track_events(tr: dict[str, Any], vid: str, offset: float, width: int, height: int) -> list[dict[str, Any]]:
    ev = []
    obs = tr["samples"]
    noun = _noun(tr)
    first = tr["first_t"]
    last = tr["last_t"]
    ev.append({"type": "appearance", "t": first + offset, "video": vid, "frame": obs[0]["f"] if obs else None,
               "tracks": [tr["id"]], "description": f"{noun} {tr['id']} first observed.",
               "position": obs[0]["p"] if obs and obs[0]["s"] is not None else None, "confidence": 0.9})
    ev.append({"type": "disappearance", "t": last + offset, "video": vid, "frame": obs[-1]["f"] if obs else None,
               "tracks": [tr["id"]], "description": f"{noun} {tr['id']} last observed.",
               "position": obs[-1]["p"] if obs and obs[-1]["s"] is not None else None, "confidence": 0.9})
    for g in tr.get("gaps", []):
        relinked = "re-linked" in g["reason"]
        ev.append({"type": "observation_gap", "t": g["t0"] + offset, "t_end": g["t1"] + offset, "video": vid,
                   "frame": g["after_frame"], "frame_end": g["before_frame"], "tracks": [tr["id"]],
                   "description": f"{noun} {tr['id']} not observed for {g['duration_s']:.1f} s ({g['reason']}).",
                   "confidence": 0.6 if relinked else 0.9, "hypothesis": relinked})
    return ev


def stationary_events(tr: dict[str, Any], vid: str, offset: float, unit: str, min_s: float = 2.0,
                      thr: float = 0.3) -> list[dict[str, Any]]:
    sp = tr.get("speed", [])
    if not sp or tr.get("stationary"):
        return []
    ts = np.array([s["t"] for s in sp])
    v = np.array([s["v"] for s in sp])
    slow = v < thr
    out = []
    i = 0
    while i < len(ts):
        if slow[i]:
            j = i
            while j + 1 < len(ts) and slow[j + 1] and ts[j + 1] - ts[j] < 1.0:
                j += 1
            if ts[j] - ts[i] >= min_s:
                out.append({"type": "stationary_period", "t": ts[i] + offset, "t_end": ts[j] + offset, "video": vid,
                            "tracks": [tr["id"]],
                            "description": f"{_noun(tr)} {tr['id']} remained nearly stationary for "
                                           f"{ts[j] - ts[i]:.1f} s (speed below {thr} {unit}/s).",
                            "confidence": 0.7})
            i = j + 1
        else:
            i += 1
    return out


def proximity_events(tracks: list[dict[str, Any]], vid_of: dict[str, str], offsets: dict[str, float],
                     transforms: dict[str, np.ndarray], unit: str, max_dist: float = 1.5,
                     min_duration: float = 0.4) -> list[dict[str, Any]]:
    """Pairs of subjects that came within max_dist of each other at the same time."""
    def series(tr):
        vid = vid_of[tr["id"]]
        t = np.array([s["t"] for s in tr["smoothed"]]) + offsets.get(vid, 0.0)
        p = np.array([s["p"] for s in tr["smoothed"]])
        M = transforms.get(vid, np.eye(4))
        return t, p @ M[:3, :3].T + M[:3, 3]

    out = []
    trs = [t for t in tracks if t.get("smoothed") and t["group"] in ("person", "vehicle", "two_wheeler")]
    ser = {t["id"]: series(t) for t in trs}
    for i in range(len(trs)):
        for j in range(i + 1, len(trs)):
            a, b = trs[i], trs[j]
            if a["group"] == b["group"] == "vehicle":
                continue
            if vid_of[a["id"]] != vid_of[b["id"]] and a["group"] == b["group"]:
                continue  # same-class pairs across cameras are probably the same subject (see conflicts)
            ta, pa = ser[a["id"]]
            tb, pb = ser[b["id"]]
            t0, t1 = max(ta[0], tb[0]), min(ta[-1], tb[-1])
            if t1 - t0 < min_duration:
                continue
            tt = np.arange(t0, t1, 0.1)
            da = np.stack([np.interp(tt, ta, pa[:, k]) for k in (0, 2)], 1)
            db = np.stack([np.interp(tt, tb, pb[:, k]) for k in (0, 2)], 1)
            d = np.linalg.norm(da - db, axis=1)
            close = d < max_dist
            if close.sum() * 0.1 < min_duration:
                continue
            k = int(np.argmin(d))
            sa = a["summary"].get("position_sigma_median") or 0.0
            sb = b["summary"].get("position_sigma_median") or 0.0
            out.append({"type": "proximity", "t": float(tt[close][0]), "t_end": float(tt[close][-1]),
                        "video": vid_of[a["id"]], "tracks": [a["id"], b["id"]],
                        "min_distance": float(d[k]), "min_distance_sigma": float(np.hypot(sa, sb)),
                        "t_min": float(tt[k]),
                        "description": f"{_noun(a)} {a['id']} and {_noun(b)} {b['id']} came within "
                                       f"{_fmt(d[k], unit)} (±{np.hypot(sa, sb):.1f}) of each other.",
                        "confidence": float(np.clip(1 - np.hypot(sa, sb) / max(max_dist, 1e-6), 0.2, 0.9))})
    return out


def crossing_events(tracks: list[dict[str, Any]], vid_of: dict[str, str], offsets: dict[str, float],
                    transforms: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """Paths that intersect in the ground plane (possibly at different times)."""
    def seg_series(tr):
        vid = vid_of[tr["id"]]
        M = transforms.get(vid, np.eye(4))
        P = np.array([s["p"] for s in tr["smoothed"]]) @ M[:3, :3].T + M[:3, 3]
        t = np.array([s["t"] for s in tr["smoothed"]]) + offsets.get(vid, 0.0)
        return P[::3, [0, 2]], t[::3]

    out = []
    trs = [t for t in tracks if t.get("smoothed") and not t.get("stationary") and len(t["smoothed"]) > 6]
    ss = {t["id"]: seg_series(t) for t in trs}
    for i in range(len(trs)):
        for j in range(i + 1, len(trs)):
            Pa, ta = ss[trs[i]["id"]]
            Pb, tb = ss[trs[j]["id"]]
            hit = _first_intersection(Pa, Pb)
            if hit is None:
                continue
            ia, ib, pt = hit
            out.append({"type": "path_crossing", "t": float(min(ta[ia], tb[ib])), "video": vid_of[trs[i]["id"]],
                        "tracks": [trs[i]["id"], trs[j]["id"]], "point_xz": pt.round(3).tolist(),
                        "times": [float(ta[ia]), float(tb[ib])],
                        "description": f"Paths of {trs[i]['id']} and {trs[j]['id']} cross "
                                       f"({abs(ta[ia] - tb[ib]):.1f} s apart).",
                        "confidence": 0.6})
    return out


def _first_intersection(A: np.ndarray, B: np.ndarray):
    for i in range(len(A) - 1):
        p, r = A[i], A[i + 1] - A[i]
        for j in range(len(B) - 1):
            q, s = B[j], B[j + 1] - B[j]
            den = r[0] * s[1] - r[1] * s[0]
            if abs(den) < 1e-12:
                continue
            qp = q - p
            t = (qp[0] * s[1] - qp[1] * s[0]) / den
            u = (qp[0] * r[1] - qp[1] * r[0]) / den
            if 0 <= t <= 1 and 0 <= u <= 1:
                return i, j, p + t * r
    return None
