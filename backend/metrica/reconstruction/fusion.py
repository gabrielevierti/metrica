"""Multi-view dense reconstruction from SfM poses + aligned monocular depth.

1. **Alignment** - each keyframe's relative inverse depth is mapped to the SfM frame with a
   robust affine fit against the 3D points *observed in that keyframe* (COLMAP tracks).
2. **Consistency** - each pixel's 3D point is reprojected into neighbouring keyframes:
   *consistent* (their depth agrees), *not visible* (outside or occluded there) or
   *conflicting* (a neighbour sees through it). Conflicting pixels are discarded.
3. **Fusion** - a narrow-band truncated signed distance field (TSDF) accumulates the
   surviving depth, weighted by consistency, on a dense voxel grid.
4. **Meshing** - marching cubes on voxels with data only (no surface is invented in
   unobserved space). Each vertex records how many keyframes supported it: >= 2 is
   OBSERVED_MULTI_VIEW, 1 is OBSERVED_SINGLE_VIEW.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage

from ..geometry.camera import Camera
from ..geometry.single_view import fit_inverse_depth_affine, metric_depth_from_affine
from ..log import get_logger

log = get_logger("fusion")


@dataclass
class KeyframeDepth:
    key: str
    cam: Camera
    z: np.ndarray            # metric (unit-frame) camera depth, NaN = invalid
    rgb: np.ndarray          # HxWx3 uint8 (RGB)
    fit: dict[str, Any]
    labels: np.ndarray | None = None
    support: np.ndarray | None = None   # consistent neighbour count per pixel


def align_depth(inv: np.ndarray, cam: Camera, points: np.ndarray, max_depth: float,
                extra_uv: np.ndarray | None = None, extra_z: np.ndarray | None = None,
                extra_weight: float = 0.5) -> tuple[np.ndarray, dict] | None:
    """Affine inverse-depth alignment against SfM points observed in this view, optionally
    with extra (pixel, depth) constraints such as ground-plane depths."""
    h, w = inv.shape
    uvs_l, z_l, w_l = [], [], []
    if len(points):
        uv, zc = cam.project(points)
        sx, sy = w / cam.intr.width, h / cam.intr.height
        uvs = np.stack([uv[:, 0] * sx, uv[:, 1] * sy], 1)
        ok = (zc > 0) & (uvs[:, 0] >= 0) & (uvs[:, 0] < w) & (uvs[:, 1] >= 0) & (uvs[:, 1] < h)
        uvs_l.append(uvs[ok])
        z_l.append(zc[ok])
        w_l.append(np.ones(ok.sum()))
    if extra_uv is not None and len(extra_uv):
        uvs_l.append(np.asarray(extra_uv, float))
        z_l.append(np.asarray(extra_z, float))
        w_l.append(np.full(len(extra_uv), extra_weight))
    if not uvs_l or sum(len(z) for z in z_l) < 25:
        return None
    uvs, zc, wts = np.concatenate(uvs_l), np.concatenate(z_l), np.concatenate(w_l)
    fit = fit_inverse_depth_affine(inv, uvs, zc, wts)
    if fit is None:
        return None
    md = min(max_depth, 3.0 * float(np.percentile(zc, 95)))
    fit["depth_range_ratio"] = float(np.percentile(zc, 95) / max(np.percentile(zc, 5), 1e-9))
    return metric_depth_from_affine(inv, fit["a"], fit["b"], md), {**fit, "max_depth": md}


def ground_constraints(cam: Camera, labels: np.ndarray, ground_ids: set[int], step: int = 6,
                       max_n: int = 3000) -> tuple[np.ndarray, np.ndarray]:
    """(pixel, depth) pairs for semantic ground pixels, from the ray / ground-plane (y=0) intersection."""
    h, w = labels.shape
    vs, us = np.mgrid[0:h:step, 0:w:step]
    m = np.isin(labels[vs, us], list(ground_ids))
    if not m.any():
        return np.zeros((0, 2)), np.zeros(0)
    uv = np.stack([us[m] + 0.5, vs[m] + 0.5], 1).astype(float)
    sx, sy = cam.intr.width / w, cam.intr.height / h
    P, ok = cam.ground_intersect(uv * [sx, sy])
    zc = cam.pose.to_cam(np.where(ok[:, None], P, 0))[:, 2]
    keep = ok & (zc > 0)
    uv, zc = uv[keep], zc[keep]
    if len(uv) > max_n:
        sel = np.random.default_rng(0).choice(len(uv), max_n, replace=False)
        uv, zc = uv[sel], zc[sel]
    return uv, zc


def projected_constraints(target: Camera, shape: tuple[int, int], sources: list[KeyframeDepth],
                          stride: int = 4, max_n: int = 6000) -> tuple[np.ndarray, np.ndarray]:
    """Depth constraints for a view from z-buffered projection of already aligned keyframes."""
    h, w = shape
    zbuf = np.full(h * w, np.inf)
    for kf in sources:
        hh, ww = kf.z.shape
        vs, us = np.mgrid[0:hh:stride, 0:ww:stride]
        zz = kf.z[vs, us]
        ok = np.isfinite(zz) & ((kf.support[vs, us] > 0) if kf.support is not None else True)
        if not ok.any():
            continue
        uv = np.stack([(us[ok] + 0.5) * kf.cam.intr.width / ww, (vs[ok] + 0.5) * kf.cam.intr.height / hh], 1)
        X = kf.cam.backproject(uv, zz[ok])
        uvt, zt = target.project(X)
        ui = (uvt[:, 0] * w / target.intr.width).astype(int)
        vi = (uvt[:, 1] * h / target.intr.height).astype(int)
        inb = (zt > 0) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        np.minimum.at(zbuf, vi[inb] * w + ui[inb], zt[inb])
    idx = np.flatnonzero(np.isfinite(zbuf))
    if len(idx) > max_n:
        idx = np.random.default_rng(0).choice(idx, max_n, replace=False)
    uv = np.stack([idx % w + 0.5, idx // w + 0.5], 1).astype(float)
    return uv, zbuf[idx]


def consistency(kfs: list[KeyframeDepth], neighbours: int = 2, rel: float = 0.06, stride: int = 2) -> dict[str, int]:
    stats = {"consistent": 0, "not_visible": 0, "conflict": 0}
    n = len(kfs)
    for i, kf in enumerate(kfs):
        h, w = kf.z.shape
        sup = np.zeros((h, w), np.int8)
        conflict = np.zeros((h, w), bool)
        nb = [j for j in sorted(range(n), key=lambda j: abs(j - i)) if j != i][:neighbours]
        vs, us = np.mgrid[0:h:stride, 0:w:stride]
        zz = kf.z[vs, us]
        valid = np.isfinite(zz)
        uv = np.stack([(us + 0.5) * kf.cam.intr.width / w, (vs + 0.5) * kf.cam.intr.height / h], -1)[valid]
        X = kf.cam.backproject(uv, zz[valid])
        for j in nb:
            o = kfs[j]
            uvj, zj = o.cam.project(X)
            hj, wj = o.z.shape
            sx, sy = wj / o.cam.intr.width, hj / o.cam.intr.height
            ui, vi = (uvj[:, 0] * sx).astype(int), (uvj[:, 1] * sy).astype(int)
            inside = (zj > 0) & (ui >= 0) & (ui < wj) & (vi >= 0) & (vi < hj)
            dz = np.full(len(X), np.nan)
            dz[inside] = o.z[vi[inside], ui[inside]]
            agree = inside & np.isfinite(dz) & (np.abs(dz - zj) < rel * zj)
            sees_through = inside & np.isfinite(dz) & (dz > zj * (1 + 2.5 * rel))
            a = np.zeros_like(valid, np.int8)
            a[valid] = agree
            c = np.zeros_like(valid)
            c[valid] = sees_through
            sup[vs, us] += a
            conflict[vs, us] |= c & ~(a > 0)
        # pixels contradicted by a neighbour and supported by none are dropped
        drop = conflict & (sup == 0)
        kf.z[drop] = np.nan
        kf.support = sup
        stats["consistent"] += int((sup[vs, us] > 0).sum())
        stats["conflict"] += int(drop[vs, us].sum())
        stats["not_visible"] += int(((sup[vs, us] == 0) & ~drop[vs, us] & np.isfinite(kf.z[vs, us])).sum())
    return stats


def tsdf_fuse(kfs: list[KeyframeDepth], bounds: tuple[np.ndarray, np.ndarray], voxel: float,
              max_voxels: int, stride: int = 2, trunc_vox: float = 3.0) -> dict[str, Any]:
    lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    dims = np.ceil((hi - lo) / voxel).astype(int) + 1
    while np.prod(dims) > max_voxels:
        voxel *= 1.15
        dims = np.ceil((hi - lo) / voxel).astype(int) + 1
    nvox = int(np.prod(dims))
    W = np.zeros(nvox, np.float32)
    D = np.zeros(nvox, np.float32)
    RGB = np.zeros((nvox, 3), np.float32)
    NV = np.zeros(nvox, np.uint16)
    trunc = trunc_vox * voxel
    offsets = np.linspace(-trunc, trunc, int(2 * np.ceil(trunc_vox)) + 1)
    for kf in kfs:
        h, w = kf.z.shape
        vs, us = np.mgrid[0:h:stride, 0:w:stride]
        zz = kf.z[vs, us]
        valid = np.isfinite(zz)
        if not valid.any():
            continue
        sup = kf.support[vs, us][valid].astype(np.float32) if kf.support is not None else np.zeros(valid.sum())
        wpx = np.where(sup > 0, 1.0 + sup, 0.35)
        col = kf.rgb[vs, us][valid].astype(np.float32)
        uv = np.stack([(us + 0.5) * kf.cam.intr.width / w, (vs + 0.5) * kf.cam.intr.height / h], -1)[valid]
        rays = kf.cam.rays_cam(uv)  # z = 1
        z0 = zz[valid]
        touched = []
        for off in offsets:
            zs = z0 + off
            Xc = rays * zs[:, None]
            Xw = kf.cam.pose.to_world(Xc)
            ijk = np.floor((Xw - lo) / voxel).astype(np.int64)
            inside = np.all((ijk >= 0) & (ijk < dims), axis=1) & (zs > 0)
            lin = np.ravel_multi_index(ijk[inside].T, dims)
            sdf = np.clip(-off / trunc, -1.0, 1.0)
            ww = wpx[inside] * (1.0 if off <= 0 else 0.8)
            W += np.bincount(lin, ww, nvox).astype(np.float32)
            D += np.bincount(lin, ww * sdf, nvox).astype(np.float32)
            for c in range(3):
                RGB[:, c] += np.bincount(lin, ww * col[inside, c], nvox).astype(np.float32)
            if abs(off) <= voxel:
                touched.append(lin)
        if touched:
            u = np.unique(np.concatenate(touched))
            NV[u] += 1
    return {"W": W.reshape(dims), "D": D.reshape(dims), "RGB": RGB.reshape(*dims, 3), "NV": NV.reshape(dims),
            "origin": lo, "voxel": voxel, "dims": dims.tolist()}


def extract_mesh(vol: dict[str, Any], min_weight: float = 0.6) -> dict[str, np.ndarray] | None:
    from skimage.measure import marching_cubes

    W, D = vol["W"], vol["D"]
    has = W > min_weight
    if has.sum() < 50:
        return None
    tsdf = np.where(has, D / np.maximum(W, 1e-6), 1.0).astype(np.float32)
    mask = ndimage.minimum_filter(has, size=2)
    try:
        verts, faces, _, _ = marching_cubes(tsdf, level=0.0, mask=mask, allow_degenerate=False)
    except (ValueError, RuntimeError):
        return None
    if len(faces) == 0:
        return None
    ijk = np.clip(np.round(verts).astype(int), 0, np.array(W.shape) - 1)
    w = W[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    rgb = vol["RGB"][ijk[:, 0], ijk[:, 1], ijk[:, 2]] / np.maximum(w, 1e-6)[:, None]
    nv = vol["NV"][ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    pos = vol["origin"] + verts * vol["voxel"]
    # marching cubes winding: flip so normals face the observed (positive) side
    faces = faces[:, ::-1]
    return {"pos": pos.astype(np.float32), "idx": faces.astype(np.uint32), "col": np.clip(rgb, 0, 255).astype(np.uint8),
            "support": nv.astype(np.int32), "weight": w.astype(np.float32)}


def label_vertices(pos: np.ndarray, kfs: list[KeyframeDepth], n_labels: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Majority scene label per vertex from semantic keyframes where the vertex is visible."""
    votes = np.zeros((len(pos), n_labels), np.float32)
    for kf in kfs:
        if kf.labels is None:
            continue
        uv, zc = kf.cam.project(pos)
        h, w = kf.labels.shape
        sx, sy = w / kf.cam.intr.width, h / kf.cam.intr.height
        ui, vi = (uv[:, 0] * sx).astype(int), (uv[:, 1] * sy).astype(int)
        ok = (zc > 0) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        dz = np.full(len(pos), np.nan)
        zh, zw = kf.z.shape
        ui2 = np.clip((uv[:, 0] * zw / kf.cam.intr.width).astype(int), 0, zw - 1)
        vi2 = np.clip((uv[:, 1] * zh / kf.cam.intr.height).astype(int), 0, zh - 1)
        dz[ok] = kf.z[vi2[ok], ui2[ok]]
        vis = ok & (~np.isfinite(dz) | (np.abs(dz - zc) < 0.08 * np.abs(zc) + 0.05))
        lab = kf.labels[vi[vis], ui[vis]]
        np.add.at(votes, (np.flatnonzero(vis), lab.astype(int)), 1.0)
    best = votes.argmax(1).astype(np.uint8)
    tot = votes.sum(1)
    conf = np.where(tot > 0, votes.max(1) / np.maximum(tot, 1), 0)
    best[tot == 0] = 0
    return best, conf
