"""Spatial alignment of reconstruction units into one case world frame, and conflict checks.

Every unit is already gravity-aligned (Y up, ground y = 0), so a unit-to-world transform has
4 degrees of freedom (yaw, ground translation) plus a scale when the units' scales differ.

Evidence, strongest first:
1. joint SfM (videos registered in one reconstruction share a frame by construction);
2. analyst point pairs (>= 2 pairs; scale estimated with >= 3);
3. shared visual landmarks: SIFT matches between views of the two units, lifted to 3D with
   each unit's aligned depth, then a robust 4-DOF similarity (RANSAC);
4. shared tracks: the same subject seen by both cameras at the same (synchronised) time.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..reconstruction.views import View


def similarity_2d(src: np.ndarray, dst: np.ndarray, allow_scale: bool = True):
    """Umeyama in the ground plane: dst ~ s R src + t (2D). Returns (s, R2, t2, rms)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[1, 1] = -1
    R = U @ D @ Vt
    var_s = (xs ** 2).sum() / len(src)
    s = float(np.trace(np.diag(S) @ D) / var_s) if allow_scale and var_s > 1e-12 else 1.0
    t = mu_d - s * R @ mu_s
    rms = float(np.sqrt(((dst - (s * (src @ R.T) + t)) ** 2).sum(1).mean()))
    return s, R, t, rms


def to_4x4(s: float, R2: np.ndarray, t2: np.ndarray, ty: float = 0.0) -> np.ndarray:
    """Embed a ground-plane similarity acting on (x, z) into a Y-up 4x4 transform."""
    M = np.eye(4)
    M[0, 0], M[0, 2] = R2[0, 0] * s, R2[0, 1] * s
    M[2, 0], M[2, 2] = R2[1, 0] * s, R2[1, 1] * s
    M[1, 1] = s
    M[0, 3], M[1, 3], M[2, 3] = t2[0], ty, t2[1]
    return M


def ransac_similarity_4dof(src: np.ndarray, dst: np.ndarray, thr: float, iters: int = 800,
                           allow_scale: bool = True, seed: int = 0) -> dict[str, Any] | None:
    """Robust (yaw, translation, scale) from 3D-3D correspondences in gravity-aligned frames."""
    if len(src) < 6:
        return None
    rng = np.random.default_rng(seed)
    best = None
    s2, d2 = src[:, [0, 2]], dst[:, [0, 2]]
    for _ in range(iters):
        i, j = rng.choice(len(src), 2, replace=False)
        if np.linalg.norm(s2[i] - s2[j]) < 1e-3:
            continue
        s, R, t, _ = similarity_2d(s2[[i, j]], d2[[i, j]], allow_scale)
        if not 0.2 < s < 5:
            continue
        pred = s * (s2 @ R.T) + t
        ey = np.abs(s * src[:, 1] - dst[:, 1])
        err = np.hypot(np.linalg.norm(pred - d2, axis=1), ey)
        inl = err < thr
        if best is None or inl.sum() > best[0].sum():
            best = (inl, s, R, t)
    if best is None or best[0].sum() < 6:
        return None
    inl = best[0]
    s, R, t, rms = similarity_2d(s2[inl], d2[inl], allow_scale)
    return {"M": to_4x4(s, R, t), "scale": s, "yaw_deg": float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))),
            "inliers": int(inl.sum()), "matches": int(len(src)), "rms": rms}


def landmark_correspondences(views_a: list[View], views_b: list[View], max_pairs: int = 12,
                             ratio: float = 0.75) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """3D-3D correspondences between two units from SIFT matches on their views."""
    sift = cv2.SIFT_create(nfeatures=3000)
    feats = {}

    def get(v: View):
        if v.key not in feats:
            g = cv2.cvtColor(v.image, cv2.COLOR_BGR2GRAY)
            feats[v.key] = sift.detectAndCompute(g, None)
        return feats[v.key]

    pairs = [(a, b) for a in views_a[:: max(1, len(views_a) // 4)] for b in views_b[:: max(1, len(views_b) // 4)]]
    pairs = pairs[:max_pairs]
    A3, B3 = [], []
    stats = {"pairs": len(pairs), "verified_matches": 0}
    bf = cv2.BFMatcher(cv2.NORM_L2)
    for va, vb in pairs:
        ka, da = get(va)
        kb, db = get(vb)
        if da is None or db is None or len(ka) < 20 or len(kb) < 20:
            continue
        m = [x[0] for x in bf.knnMatch(da, db, k=2) if len(x) == 2 and x[0].distance < ratio * x[1].distance]
        if len(m) < 15:
            continue
        pa = np.float32([ka[x.queryIdx].pt for x in m])
        pb = np.float32([kb[x.trainIdx].pt for x in m])
        F, mask = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 2.0, 0.999)
        if F is None:
            continue
        good = mask.ravel() == 1
        pa, pb = pa[good], pb[good]
        stats["verified_matches"] += int(good.sum())
        for (ua, va_), (ub, vb_) in zip(pa, pb):
            za = va.z[int(va_), int(ua)]
            zb = vb.z[int(vb_), int(ub)]
            if not (np.isfinite(za) and np.isfinite(zb)):
                continue
            A3.append(va.cam.backproject(np.array([[ua, va_]]), np.array([za]))[0])
            B3.append(vb.cam.backproject(np.array([[ub, vb_]]), np.array([zb]))[0])
    return np.array(A3).reshape(-1, 3), np.array(B3).reshape(-1, 3), stats


def track_alignment(ref: list[dict[str, Any]], other: list[dict[str, Any]], off_ref: float, off_other: float,
                    min_overlap_s: float = 2.0) -> dict[str, Any] | None:
    """Hypothesise 'track A (ref) == track B (other)', fit a similarity, keep the best-supported."""
    def sample(tr, times, off):
        t = np.array([s["t"] for s in tr["smoothed"]]) + off
        p = np.array([s["p"] for s in tr["smoothed"]])
        return np.stack([np.interp(times, t, p[:, 0]), np.interp(times, t, p[:, 2])], 1)

    cands = []
    for A in ref:
        for B in other:
            if A["group"] != B["group"] or not A.get("smoothed") or not B.get("smoothed"):
                continue
            t0 = max(A["smoothed"][0]["t"] + off_ref, B["smoothed"][0]["t"] + off_other)
            t1 = min(A["smoothed"][-1]["t"] + off_ref, B["smoothed"][-1]["t"] + off_other)
            if t1 - t0 < min_overlap_s:
                continue
            times = np.linspace(t0, t1, 20)
            pa, pb = sample(A, times, off_ref), sample(B, times, off_other)
            if np.linalg.norm(pa[-1] - pa[0]) < 1.0:
                continue  # needs movement to constrain rotation
            s, R, t, rms = similarity_2d(pb, pa)
            if 0.6 < s < 1.6:
                cands.append((s, R, t, rms, A["id"], B["id"]))
    if not cands:
        return None
    best, best_sup = None, -1.0
    for c in cands:
        sup = 0.0
        for c2 in cands:
            ang = np.arctan2(c[1][1, 0], c[1][0, 0]) - np.arctan2(c2[1][1, 0], c2[1][0, 0])
            if abs((ang + np.pi) % (2 * np.pi) - np.pi) < np.radians(10) and np.linalg.norm(c[2] - c2[2]) < 2.0:
                sup += 1.0 / (1.0 + c2[3])
        if sup > best_sup:
            best, best_sup = c, sup
    s, R, t, rms, a, b = best
    return {"M": to_4x4(s, R, t), "scale": s, "rms": rms, "hypotheses": len(cands), "support": best_sup,
            "pair": [a, b], "yaw_deg": float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))}


def find_conflicts(tracks_by_video: dict[str, list[dict[str, Any]]], transforms: dict[str, np.ndarray],
                   offsets: dict[str, float], max_dist: float = 2.5) -> list[dict[str, Any]]:
    """Compare the same subject as seen by two cameras after alignment and synchronisation."""
    from scipy.optimize import linear_sum_assignment

    def world(tr, vid, times):
        t = np.array([s["t"] for s in tr["smoothed"]]) + offsets.get(vid, 0.0)
        p = np.array([s["p"] for s in tr["smoothed"]])
        M = transforms[vid]
        P = p @ M[:3, :3].T + M[:3, 3]
        return np.stack([np.interp(times, t, P[:, 0]), np.interp(times, t, P[:, 2])], 1)

    out = []
    vids = [v for v in tracks_by_video if v in transforms]
    for i in range(len(vids)):
        for j in range(i + 1, len(vids)):
            A = [t for t in tracks_by_video[vids[i]] if t.get("smoothed")]
            B = [t for t in tracks_by_video[vids[j]] if t.get("smoothed")]
            if not A or not B:
                continue
            cost = np.full((len(A), len(B)), 1e6)
            info = {}
            for a, ta in enumerate(A):
                for b, tb in enumerate(B):
                    t0 = max(ta["smoothed"][0]["t"] + offsets.get(vids[i], 0), tb["smoothed"][0]["t"] + offsets.get(vids[j], 0))
                    t1 = min(ta["smoothed"][-1]["t"] + offsets.get(vids[i], 0), tb["smoothed"][-1]["t"] + offsets.get(vids[j], 0))
                    if t1 - t0 < 1.0:
                        continue
                    times = np.linspace(t0, t1, 15)
                    d = np.linalg.norm(world(ta, vids[i], times) - world(tb, vids[j], times), axis=1)
                    cost[a, b] = float(np.median(d))
                    info[(a, b)] = (t0, t1, float(np.max(d)))
            r, c = linear_sum_assignment(cost)
            for a, b in zip(r, c):
                if cost[a, b] >= 1e6 or cost[a, b] > 3 * max_dist:
                    continue
                t0, t1, dmax = info[(a, b)]
                ta, tb = A[a], B[b]
                rec = {"videos": [vids[i], vids[j]], "tracks": [ta["id"], tb["id"]], "t0": t0, "t1": t1,
                       "median_distance": cost[a, b], "max_distance": dmax,
                       "categories": [ta["category"], tb["category"]]}
                if ta["group"] != tb["group"]:
                    rec["conflict"] = "same position and time but different object classes"
                elif cost[a, b] > max_dist:
                    rec["conflict"] = "likely the same subject, but positions disagree beyond tolerance"
                else:
                    rec["conflict"] = None
                out.append(rec)
    return out
