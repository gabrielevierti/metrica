"""Camera-motion analysis and adaptive keyframe selection.

For consecutive sampled frames, background features are tracked with pyramidal
Lucas-Kanade (forward-backward checked). Two robust models are fitted:

* a homography H (explains pure rotation / zoom, or a planar scene), and
* a fundamental matrix F (explains general motion with parallax).

Per pair the camera is classified as *still* (median displacement ~0; moving objects
do not affect the median), *rotating* (H explains the motion about as well as F), or
*translating* (F explains clearly more correspondences than H: there is parallax, which
is what multi-view reconstruction needs).
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

LK = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def pair_motion(g0: np.ndarray, g1: np.ndarray) -> dict[str, Any]:
    h, w = g0.shape
    diag = float(np.hypot(w, h))
    p0 = cv2.goodFeaturesToTrack(g0, maxCorners=500, qualityLevel=0.01, minDistance=7, blockSize=7)
    if p0 is None or len(p0) < 20:
        return {"state": "textureless", "disp_frac": float("nan"), "tracked": 0}
    p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, **LK)
    pb, stb, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, **LK)
    fb = np.linalg.norm((pb - p0).reshape(-1, 2), axis=1)
    good = (st.ravel() == 1) & (stb.ravel() == 1) & (fb < 1.0)
    a, b = p0.reshape(-1, 2)[good], p1.reshape(-1, 2)[good]
    if len(a) < 20:
        return {"state": "lost", "disp_frac": float("nan"), "tracked": int(len(a))}
    d = np.linalg.norm(b - a, axis=1)
    med = float(np.median(d))
    res: dict[str, Any] = {"tracked": int(len(a)), "disp_px": med, "disp_frac": med / diag}
    if med < 0.0025 * diag:
        res["state"] = "still"
        return res
    H, mh = cv2.findHomography(a, b, cv2.RANSAC, 1.5)
    F, mf = cv2.findFundamentalMat(a, b, cv2.FM_RANSAC, 1.0, 0.999)
    nh = int(mh.sum()) if mh is not None else 0
    nf = int(mf.sum()) if mf is not None else 0
    res["h_inliers"], res["f_inliers"] = nh, nf
    if H is not None and nh > 0:
        proj = cv2.perspectiveTransform(a[mh.ravel() == 1].reshape(-1, 1, 2), H).reshape(-1, 2)
        res["h_residual_px"] = float(np.median(np.linalg.norm(proj - b[mh.ravel() == 1], axis=1)))
    # GRIC-like decision: F has more freedom, so require a clear margin before calling it parallax
    res["state"] = "translating" if nf > 1.15 * nh + 8 else "rotating"
    return res


def classify(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    states = [p["state"] for p in pairs]
    n = max(1, len(states))
    frac = {s: states.count(s) / n for s in ("still", "rotating", "translating", "textureless", "lost")}
    if frac["still"] >= 0.85:
        mode, conf = "static", frac["still"]
    elif frac["translating"] >= 0.25:
        mode, conf = "moving", min(1.0, frac["translating"] * 1.5)
    elif frac["rotating"] + frac["translating"] > 0.3:
        mode, conf = "rotating", frac["rotating"]
    else:
        mode, conf = "static", frac["still"] + frac["textureless"] * 0.5
    disp = np.array([p.get("disp_frac", np.nan) for p in pairs], float)
    shake = float(np.nanmedian(disp)) if np.isfinite(disp).any() else 0.0
    notes = []
    if mode == "static" and shake > 0.0008:
        notes.append("Small residual image motion: camera may be mounted on a vibrating support.")
    if frac["textureless"] + frac["lost"] > 0.3:
        notes.append("Many frames lack trackable texture (darkness, blur or uniform surfaces).")
    if mode == "rotating":
        notes.append("Camera rotates (pan/tilt/zoom) without moving: no parallax for triangulation; "
                     "single-view geometry is used.")
    return {"mode": mode, "confidence": float(conf), "fractions": frac, "median_disp_frac": shake,
            "pairs": len(pairs), "notes": notes}


def segments(pairs_t: list[tuple[float, str]], min_len_s: float = 2.0) -> list[dict[str, Any]]:
    """Merge per-pair states into time segments; short runs are absorbed by their longer neighbour."""
    runs: list[dict[str, Any]] = []
    for t, s in pairs_t:
        s = "static" if s in ("still", "textureless", "lost") else ("moving" if s == "translating" else s)
        if runs and runs[-1]["state"] == s:
            runs[-1]["t1"] = t
            runs[-1]["n"] += 1
        else:
            runs.append({"state": s, "t0": t, "t1": t, "n": 1})
    changed = True
    while changed and len(runs) > 1:
        changed = False
        k = min(range(len(runs)), key=lambda i: runs[i]["t1"] - runs[i]["t0"])
        if runs[k]["t1"] - runs[k]["t0"] >= min_len_s:
            break
        nb = [j for j in (k - 1, k + 1) if 0 <= j < len(runs)]
        j = max(nb, key=lambda i: runs[i]["t1"] - runs[i]["t0"])
        a, b = sorted((j, k))
        keep = runs[j]
        merged = {"state": keep["state"], "t0": runs[a]["t0"], "t1": runs[b]["t1"], "n": runs[a]["n"] + runs[b]["n"]}
        runs[a:b + 1] = [merged]
        # re-merge identical neighbours
        i = 0
        while i < len(runs) - 1:
            if runs[i]["state"] == runs[i + 1]["state"]:
                runs[i:i + 2] = [{"state": runs[i]["state"], "t0": runs[i]["t0"], "t1": runs[i + 1]["t1"],
                                  "n": runs[i]["n"] + runs[i + 1]["n"]}]
            else:
                i += 1
        changed = True
    return [{"state": r["state"], "t0": r["t0"], "t1": r["t1"]} for r in runs]


def select_keyframes(samples: list[dict[str, Any]], pairs: list[dict[str, Any]], flow_frac: float,
                     max_keyframes: int, min_sharpness: float) -> list[int]:
    """Adaptive keyframes for SfM from sampled frames.

    samples[i] = {"i": frame_index, "sharpness": ..., "usable": bool}; pairs[i] is the motion
    between samples[i] and samples[i+1]. A keyframe is emitted once accumulated median motion
    exceeds ``flow_frac`` of the diagonal, choosing the sharpest usable frame of the window.
    """
    if not samples:
        return []
    keys = [0]
    acc = 0.0
    window: list[int] = []
    for j, p in enumerate(pairs, start=1):
        d = p.get("disp_frac", 0.0)
        acc += d if np.isfinite(d) else 0.0
        if samples[j].get("usable", True) and samples[j]["sharpness"] >= min_sharpness:
            window.append(j)
        if acc >= flow_frac and window:
            best = max(window[-3:], key=lambda k: samples[k]["sharpness"])
            if best != keys[-1]:
                keys.append(best)
            acc = 0.0
            window = []
    if keys[-1] != len(samples) - 1 and samples[-1].get("usable", True):
        keys.append(len(samples) - 1)
    if len(keys) > max_keyframes:
        sel = np.linspace(0, len(keys) - 1, max_keyframes).round().astype(int)
        keys = [keys[k] for k in sorted(set(sel.tolist()))]
    return [samples[k]["i"] for k in keys]
