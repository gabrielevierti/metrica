"""Geometric primitive fitting used by the semantic object reconstructors.

All fits are robust (RANSAC + least-squares refinement) and return the residuals and
inlier support so each object can report how well its primitive explains the evidence.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from scipy import ndimage


def vertical_planes(points: np.ndarray, max_planes: int = 8, thr: float = 0.08, min_inliers: int = 200,
                    min_length: float = 1.0, seed: int = 0) -> list[dict[str, Any]]:
    """Sequential RANSAC for vertical planes (normal perpendicular to +Y) in an Nx3 cloud."""
    rng = np.random.default_rng(seed)
    remaining = np.arange(len(points))
    planes = []
    xz = points[:, [0, 2]]
    for _ in range(max_planes):
        if len(remaining) < min_inliers:
            break
        sub = remaining if len(remaining) <= 30000 else rng.choice(remaining, 30000, replace=False)
        best, best_n = None, 0
        for _ in range(400):
            i, j = rng.choice(sub, 2, replace=False)
            d = xz[j] - xz[i]
            L = np.linalg.norm(d)
            if L < 0.3:
                continue
            n = np.array([-d[1], d[0]]) / L
            inl = np.abs((xz[sub] - xz[i]) @ n) < thr
            if inl.sum() > best_n:
                best, best_n = (n, float(n @ xz[i])), int(inl.sum())
        if best is None or best_n < 0.3 * min_inliers:
            break
        n, off = best
        inl_all = remaining[np.abs(xz[remaining] @ n - off) < thr]
        c0 = xz[inl_all].mean(0)
        n = np.linalg.svd(xz[inl_all] - c0, full_matrices=False)[2][-1]
        off = float(n @ c0)
        inl_all = remaining[np.abs(xz[remaining] @ n - off) < thr]
        # keep the largest contiguous run along the plane (two separate walls on one line are split)
        u_axis = np.array([n[1], -n[0]])
        u = xz[inl_all] @ u_axis
        keep = _largest_run(u, gap=1.5)
        inl_all = inl_all[keep]
        if len(inl_all) < min_inliers or np.ptp(u[keep]) < min_length:
            remaining = np.setdiff1d(remaining, inl_all)
            continue
        res = np.abs(xz[inl_all] @ n - off)
        planes.append({"normal_xz": n, "offset": off, "inliers": inl_all, "rms": float(np.sqrt((res ** 2).mean()))})
        remaining = np.setdiff1d(remaining, inl_all)
    return planes


def _largest_run(u: np.ndarray, gap: float) -> np.ndarray:
    order = np.argsort(u)
    us = u[order]
    breaks = np.where(np.diff(us) > gap)[0]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks + 1, len(us)]
    k = int(np.argmax(ends - starts))
    mask = np.zeros(len(u), bool)
    mask[order[starts[k]:ends[k]]] = True
    return mask


def plane_frame(normal_xz: np.ndarray, offset: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(origin, u_axis, v_axis, normal) of a vertical plane; v is world up."""
    n3 = np.array([normal_xz[0], 0.0, normal_xz[1]])
    u3 = np.array([normal_xz[1], 0.0, -normal_xz[0]])
    origin = n3 * offset
    return origin, u3, np.array([0.0, 1.0, 0.0]), n3


def extent_polygons(uv: np.ndarray, cell: float, close_cells: int = 2, min_area_cells: int = 6,
                    simplify: float = 1.0) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Observed-extent polygons of 2D points (plane coordinates) via an occupancy grid.

    Small gaps (<= close_cells) are closed morphologically; larger holes are kept as holes:
    they are regions without evidence (occlusion or openings), not filled in.
    """
    if len(uv) < 3:
        return [], {}
    lo = uv.min(0) - cell
    hi = uv.max(0) + cell
    n = np.ceil((hi - lo) / cell).astype(int) + 1
    if n[0] * n[1] > 4_000_000:
        f = np.sqrt(n[0] * n[1] / 4_000_000)
        cell *= f
        n = np.ceil((hi - lo) / cell).astype(int) + 1
    occ = np.zeros((n[1], n[0]), np.uint8)
    ij = np.floor((uv - lo) / cell).astype(int)
    occ[ij[:, 1], ij[:, 0]] = 1
    raw = occ.copy()
    if close_cells:
        k = np.ones((2 * close_cells + 1, 2 * close_cells + 1), np.uint8)
        occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, k)
    contours, hier = cv2.findContours(occ, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < min_area_cells:
            continue
        c = cv2.approxPolyDP(c, simplify, True).reshape(-1, 2).astype(float)
        if len(c) < 3:
            continue
        polys.append(lo + (c + 0.5) * cell)
    info = {"cell": cell, "origin": lo.tolist(), "occupied_cells": int(raw.sum()),
            "filled_cells_after_closing": int(occ.sum()), "grid": occ, "raw": raw}
    return polys, info


def fit_circle(xz: np.ndarray, iters: int = 200, thr: float = 0.03, seed: int = 0) -> dict[str, Any] | None:
    """Robust circle fit (Kasa algebraic fit inside RANSAC). Works on partial arcs."""
    if len(xz) < 6:
        return None
    rng = np.random.default_rng(seed)

    def kasa(p):
        A = np.c_[2 * p[:, 0], 2 * p[:, 1], np.ones(len(p))]
        b = (p ** 2).sum(1)
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        cx, cz = sol[0], sol[1]
        r = np.sqrt(max(sol[2] + cx * cx + cz * cz, 1e-12))
        return np.array([cx, cz]), r

    best = None
    for _ in range(iters):
        idx = rng.choice(len(xz), 3, replace=False)
        try:
            c, r = kasa(xz[idx])
        except np.linalg.LinAlgError:
            continue
        if not np.isfinite(r) or r > 2.0:
            continue
        inl = np.abs(np.linalg.norm(xz - c, axis=1) - r) < thr
        if best is None or inl.sum() > best[2].sum():
            best = (c, r, inl)
    if best is None or best[2].sum() < 6:
        return None
    c, r = kasa(xz[best[2]])
    res = np.abs(np.linalg.norm(xz[best[2]] - c, axis=1) - r)
    ang = np.arctan2(xz[best[2], 1] - c[1], xz[best[2], 0] - c[0])
    arc = _arc_coverage(ang)
    return {"center": c, "radius": float(r), "inliers": int(best[2].sum()), "rms": float(np.sqrt((res ** 2).mean())),
            "arc_coverage_deg": arc}


def _arc_coverage(ang: np.ndarray) -> float:
    a = np.sort(np.mod(ang, 2 * np.pi))
    gaps = np.diff(np.r_[a, a[0] + 2 * np.pi])
    return float(np.degrees(2 * np.pi - gaps.max()))


def min_area_rect(xz: np.ndarray) -> dict[str, Any]:
    """Minimum-area rectangle; yaw is the direction of the LONG side (angle in the x-z plane).

    The orientation is taken from the rectangle's corner points rather than from OpenCV's
    angle output, whose convention changed between OpenCV versions."""
    rect = cv2.minAreaRect(xz.astype(np.float32))
    (cx, cz), _, _ = rect
    pts = cv2.boxPoints(rect).astype(np.float64)
    e1, e2 = pts[1] - pts[0], pts[2] - pts[1]
    l1, l2 = float(np.linalg.norm(e1)), float(np.linalg.norm(e2))
    long_edge, length, width = (e1, l1, l2) if l1 >= l2 else (e2, l2, l1)
    yaw = float(np.arctan2(long_edge[1], long_edge[0]))
    yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2  # heading of a footprint is ambiguous by 180 degrees
    return {"center": [float(cx), float(cz)], "length": length, "width": width, "yaw_rad": float(yaw)}


def cluster_grid(points: np.ndarray, cell: float, min_points: int = 20) -> list[np.ndarray]:
    """Connected-component clustering on an XZ occupancy grid (fast, deterministic)."""
    if len(points) == 0:
        return []
    xz = points[:, [0, 2]]
    lo = xz.min(0)
    ij = np.floor((xz - lo) / cell).astype(int)
    shape = ij.max(0) + 1
    if shape[0] * shape[1] > 25_000_000:
        return [np.arange(len(points))]
    grid = np.zeros((shape[1], shape[0]), bool)
    grid[ij[:, 1], ij[:, 0]] = True
    lab, n = ndimage.label(grid, structure=np.ones((3, 3)))
    pl = lab[ij[:, 1], ij[:, 0]]
    out = []
    for k in range(1, n + 1):
        idx = np.flatnonzero(pl == k)
        if len(idx) >= min_points:
            out.append(idx)
    return out
