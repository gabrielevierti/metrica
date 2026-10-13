"""Buildings and facades: vertical planes with their *observed* extent, connected corners,
candidate openings, occlusion-explained inferred patches, and rectified textures.

Rules: a facade polygon covers only where surface points were observed (small gaps closed).
Gaps that a camera could not see because something stood in front are added as separate
INFERRED patches; gaps the camera *did* see through (openings, recesses, sky) stay open.
Unseen sides of buildings are never generated.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..semantics.labels import STRUCTURE_IDS
from .primitives import extent_polygons, plane_frame, vertical_planes
from .views import View

TEX_PX_PER_UNIT = 40.0
TEX_MAX = 2048


def bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear colour sampling at float pixel positions (no size limits, unlike cv2.remap)."""
    h, w = img.shape[:2]
    x = np.clip(x - 0.5, 0, w - 1.001)
    y = np.clip(y - 0.5, 0, h - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    I = img.astype(np.float32)
    c = (I[y0, x0] * (1 - fx) * (1 - fy) + I[y0, x0 + 1] * fx * (1 - fy) + I[y0 + 1, x0] * (1 - fx) * fy
         + I[y0 + 1, x0 + 1] * fx * fy)
    return np.clip(c, 0, 255).astype(np.uint8)


def facade_points(views: list[View], stride: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Candidate structure points (unit frame) and the index of the view each came from."""
    pts, src = [], []
    for k, v in enumerate(views):
        if v.plane_id is not None:      # single view: planes found during reconstruction
            mask = v.plane_id >= 1
        elif v.labels is not None:
            mask = np.isin(v.labels, list(STRUCTURE_IDS))
        else:
            continue
        if v.support is not None:
            mask &= v.support > 0       # multi-view consistent pixels only
        p, _ = v.points(mask, stride)
        p = p[p[:, 1] > 0.3]
        pts.append(p)
        src.append(np.full(len(p), k))
    if not pts:
        return np.zeros((0, 3)), np.zeros(0, int)
    return np.concatenate(pts), np.concatenate(src)


def _merge_coplanar(planes: list[dict[str, Any]], pts: np.ndarray, thr: float) -> list[dict[str, Any]]:
    """Merge planes that describe the same wall (parallel, close, overlapping along the wall)."""
    changed = True
    while changed:
        changed = False
        for i in range(len(planes)):
            for j in range(i + 1, len(planes)):
                a, b = planes[i], planes[j]
                na, nb = a["normal_xz"], b["normal_xz"]
                if abs(na @ nb) < np.cos(np.radians(8)):
                    continue
                sgn = 1.0 if na @ nb > 0 else -1.0
                if abs(a["offset"] - sgn * b["offset"]) > 3 * thr + 0.25:
                    continue
                u = np.array([na[1], -na[0]])
                ua = pts[a["inliers"]][:, [0, 2]] @ u
                ub = pts[b["inliers"]][:, [0, 2]] @ u
                if min(ua.max(), ub.max()) - max(ua.min(), ub.min()) < -1.0:
                    continue
                idx = np.concatenate([a["inliers"], b["inliers"]])
                xz = pts[idx][:, [0, 2]]
                c0 = xz.mean(0)
                n = np.linalg.svd(xz - c0, full_matrices=False)[2][-1]
                res = np.abs((xz - c0) @ n)
                planes[i] = {"normal_xz": n, "offset": float(n @ c0), "inliers": idx,
                             "rms": float(np.sqrt((res ** 2).mean()))}
                planes.pop(j)
                changed = True
                break
            if changed:
                break
    return planes


def reconstruct_facades(views: list[View], unit_id: str, unit_scale: float = 1.0,
                        noise: float = 0.0) -> list[dict[str, Any]]:
    pts, src = facade_points(views)
    if len(pts) < 300:
        return []
    thr = max(0.08 * unit_scale, 1.2 * noise)
    planes = vertical_planes(pts, max_planes=12, thr=thr, min_inliers=max(250, len(pts) // 200),
                             min_length=1.0 * unit_scale)
    planes = _merge_coplanar(planes, pts, thr)
    out = []
    for k, pl in enumerate(planes):
        P = pts[pl["inliers"]]
        origin, u3, v3, n3 = plane_frame(pl["normal_xz"], pl["offset"])
        # orient the normal towards the cameras that observed it
        cams = np.array([views[i].cam.center for i in np.unique(src[pl["inliers"]])])
        if ((cams - origin) @ n3).mean() < 0:
            n3, u3 = -n3, -u3  # same plane, only the orientation flips
        uv = np.stack([(P - origin) @ u3, P[:, 1]], 1)
        cell = 0.2 * unit_scale
        polys, info = extent_polygons(uv, cell, close_cells=2, min_area_cells=10, simplify=1.0)
        if not polys:
            continue
        u0, u1 = float(uv[:, 0].min()), float(uv[:, 0].max())
        vtop = float(np.percentile(uv[:, 1], 99))
        views_used = sorted({views[i].key for i in src[pl["inliers"]]})
        ent = {
            "id": f"{unit_id}-facade-{k + 1}",
            "category": "facade",
            "geometry": {"type": "plane_polygon", "origin": origin.tolist(), "u_axis": u3.tolist(),
                         "v_axis": v3.tolist(), "normal": n3.tolist(),
                         "polygons_uv": [p.round(3).tolist() for p in polys], "main_polygon": 0},
            "dimensions": {
                "observed_width": {"value": u1 - u0, "basis": "measured", "note": "observed extent along the wall"},
                "observed_height": {"value": vtop, "basis": "measured",
                                    "note": "highest observed point; the building may be taller"}},
            "fit": {"inlier_points": int(len(P)), "rms": pl["rms"], "occupied_cells": info["occupied_cells"],
                    "cell": info["cell"]},
            "evidence": {"class": "OBSERVED_MULTI_VIEW" if len(views_used) > 1 else "OBSERVED_SINGLE_VIEW",
                         "views": views_used, "confidence": float(np.clip(1.0 - pl["rms"] / (2 * thr), 0.2, 0.95))},
            "provenance": {"method": "sequential RANSAC for vertical planes on structure points; extent from an "
                                     "occupancy grid of observed points", "algorithm": "metrica.facades v1"},
            "_grid": info, "_uv_min": uv.min(0), "_plane": (origin, u3, v3, n3),
        }
        ent["openings"] = []
        ent["inferred_patches"] = []
        out.append(ent)
    _connect_corners(out, unit_scale)
    return out


def _connect_corners(facades: list[dict[str, Any]], unit_scale: float) -> None:
    for i in range(len(facades)):
        for j in range(i + 1, len(facades)):
            a, b = facades[i], facades[j]
            oa, ua, _, na = a["_plane"]
            ob, ub, _, nb = b["_plane"]
            if abs(na @ nb) > 0.95:
                continue  # parallel walls do not meet
            # intersection line of two vertical planes: a vertical line through point p (in XZ)
            A = np.array([[na[0], na[2]], [nb[0], nb[2]]])
            rhs = np.array([na @ oa, nb @ ob])
            try:
                pxz = np.linalg.solve(A, rhs)
            except np.linalg.LinAlgError:
                continue
            p = np.array([pxz[0], 0.0, pxz[1]])
            da = _distance_to_extent(a, p)
            db = _distance_to_extent(b, p)
            if da < 1.5 * unit_scale and db < 1.5 * unit_scale:
                ang = float(np.degrees(np.arccos(np.clip(abs(na @ nb), 0, 1))))
                rel = {"type": "meets_at_corner", "with": b["id"], "corner_xz": pxz.round(3).tolist(),
                       "angle_deg": round(ang, 1), "gap_to_observed": [round(da, 2), round(db, 2)],
                       "note": "corner position inferred from the intersection of the two fitted planes"}
                a.setdefault("relations", []).append(rel)
                b.setdefault("relations", []).append({**rel, "with": a["id"]})


def _distance_to_extent(f: dict[str, Any], p: np.ndarray) -> float:
    o, u, _, _ = f["_plane"]
    t = (p - o) @ u
    us = [pt[0] for poly in f["geometry"]["polygons_uv"] for pt in poly]
    return float(max(0.0, min(us) - t, t - max(us)))


def openings_and_inference(f: dict[str, Any], views: list[View], unit_scale: float) -> None:
    """Candidate openings (appearance: dark rectangles on the rectified texture) and inferred
    patches (holes inside the facade hull that are occluded in every view)."""
    info = f["_grid"]
    grid_closed, raw = info["grid"], info["raw"]
    cell = info["cell"]
    lo = np.asarray(info["origin"])
    o, u3, v3, n3 = f["_plane"]
    hull = np.zeros_like(grid_closed)
    cnts, _ = cv2.findContours(grid_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cv2.drawContours(hull, [cv2.convexHull(max(cnts, key=cv2.contourArea))], -1, 1, -1)
    # restrict the hull below the highest observed row (never invent upper floors)
    rows_obs = np.nonzero(raw.any(1))[0]
    if len(rows_obs):
        hull[rows_obs.max() + 1:] = 0
    gaps = (hull > 0) & (grid_closed == 0)
    occluded = np.zeros_like(gaps)
    gi, gj = np.nonzero(gaps)
    if len(gi):
        uvc = lo + (np.stack([gj, gi], 1) + 0.5) * cell
        P = o + uvc[:, :1] * u3 + uvc[:, 1:2] * v3
        votes = np.zeros(len(P), int)
        seen = np.zeros(len(P), int)
        for v in views:
            uv, zc = v.cam.project(P)
            h, w = v.z.shape
            ui, vi = uv[:, 0].astype(int), uv[:, 1].astype(int)
            ins = (zc > 0) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
            facing = (v.cam.center - P) @ n3 > 0
            ins &= facing
            dz = np.full(len(P), np.nan)
            dz[ins] = v.z[vi[ins], ui[ins]]
            nearer = ins & np.isfinite(dz) & (dz < zc * 0.92)
            through = ins & (~np.isfinite(dz) | (dz > zc * 1.05))
            votes += nearer
            seen += through
        occ = (votes > 0) & (seen == 0)
        occluded[gi[occ], gj[occ]] = 1
        occluded = cv2.morphologyEx(occluded.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)).astype(bool)
        cnts, _ = cv2.findContours(occluded.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) < 4:
                continue
            c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2).astype(float)
            if len(c) >= 3:
                f["inferred_patches"].append({"polygon_uv": (lo + (c + 0.5) * cell).round(3).tolist(),
                                              "evidence": "INFERRED",
                                              "reason": "hidden behind a nearer object in every view; wall "
                                                        "assumed to continue (plane extension)"})
    f["fit"]["inferred_area"] = float(occluded.sum() * cell * cell)
    f["fit"]["unexplained_gap_area"] = float((gaps & ~occluded).sum() * cell * cell)


def facade_texture(f: dict[str, Any], views: list[View], out_path, unit_scale: float) -> dict[str, Any] | None:
    """Rectified, occlusion-aware texture of the facade's observed extent (RGBA PNG)."""
    o, u3, v3, n3 = f["_plane"]
    polys = [np.asarray(p) for p in f["geometry"]["polygons_uv"]]
    allp = np.concatenate(polys)
    umin, vmin = allp.min(0)
    umax, vmax = allp.max(0)
    px = TEX_PX_PER_UNIT / max(unit_scale, 1e-6)
    W = int(min(TEX_MAX, max(8, (umax - umin) * px)))
    H = int(min(TEX_MAX, max(8, (vmax - vmin) * px)))
    gu, gv = np.meshgrid(np.linspace(umin, umax, W), np.linspace(vmax, vmin, H))
    P = o + gu.reshape(-1, 1) * u3 + gv.reshape(-1, 1) * v3
    best_col = np.zeros((H * W, 3), np.uint8)
    best_q = np.zeros(H * W, np.float32)
    for v in views:
        uv, zc = v.cam.project(P)
        h, w = v.z.shape
        ins = (zc > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
        ins &= (v.cam.center - P) @ n3 > 0
        if not ins.any():
            continue
        ui, vi = uv[ins, 0].astype(int), uv[ins, 1].astype(int)
        dz = v.z[vi, ui]
        vis = ~np.isfinite(dz) | (np.abs(dz - zc[ins]) < 0.06 * zc[ins] + 0.05 * unit_scale)
        idx = np.flatnonzero(ins)[vis]
        view_dir = (v.cam.center - P[idx])
        dist = np.linalg.norm(view_dir, axis=1)
        q = (view_dir @ n3) / np.maximum(dist, 1e-9) / (1 + dist / (20 * unit_scale))
        better = q > best_q[idx]
        sel = idx[better]
        best_col[sel] = bilinear(v.image, uv[sel, 0], uv[sel, 1])
        best_q[sel] = q[better]
    alpha = (best_q > 0).astype(np.uint8) * 255
    mask = np.zeros((H, W), np.uint8)
    for p in polys:
        q = np.stack([(p[:, 0] - umin) / max(umax - umin, 1e-9) * (W - 1),
                      (vmax - p[:, 1]) / max(vmax - vmin, 1e-9) * (H - 1)], 1).astype(np.int32)
        cv2.fillPoly(mask, [q], 255)
    alpha = np.minimum(alpha.reshape(H, W), mask)
    if alpha.sum() == 0:
        return None
    rgba = np.dstack([best_col.reshape(H, W, 3), alpha])
    cv2.imwrite(str(out_path), rgba)
    # candidate openings from appearance: dark, rectangular blobs on the observed texture
    gray = cv2.cvtColor(best_col.reshape(H, W, 3), cv2.COLOR_BGR2GRAY)
    valid = alpha > 0
    if valid.sum() > 100:
        med = np.median(gray[valid])
        dark = (gray < 0.62 * med) & valid
        dark = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(dark)
        for k in range(1, n):
            x, y, ww, hh, area = stats[k]
            wu, hu = ww / px, hh / px
            if not (0.35 * unit_scale <= wu <= 4.0 * unit_scale and 0.4 * unit_scale <= hu <= 4.0 * unit_scale):
                continue
            if area / max(ww * hh, 1) < 0.7:
                continue
            f["openings"].append({"u0": float(umin + x / px), "u1": float(umin + (x + ww) / px),
                                  "v0": float(vmax - (y + hh) / px), "v1": float(vmax - y / px),
                                  "kind": "candidate opening (window/door)",
                                  "basis": "appearance: dark rectangular region on the rectified facade texture",
                                  "confidence": round(float(area / max(ww * hh, 1)) * 0.6, 2)})
    return {"file": out_path.name, "u_range": [float(umin), float(umax)], "v_range": [float(vmin), float(vmax)],
            "size_px": [W, H], "observed_fraction": float((alpha > 0).sum() / max(1, (mask > 0).sum()))}
