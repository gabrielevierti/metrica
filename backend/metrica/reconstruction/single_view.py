"""Single-view surface reconstruction for a calibrated static camera.

1. Relative depth is converted to metric depth by a robust affine fit (in inverse depth)
   against the calibrated ground plane at pixels known to be walking surface: the foot
   points of tracked people (temporal evidence) and pixels labelled road/sidewalk/terrain.
2. Ground pixels are snapped onto the plane (removes network ripples).
3. Vertical planes (walls/facades) are found by RANSAC in the ground-plane projection of
   building/wall pixels, kept only as large connected regions seen reasonably head-on, and
   snapped flat.
4. Remaining "free" surfaces are smoothed per superpixel with a robust local plane, which
   stops depth ramps at object borders from turning into stretched sheets.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..geometry.camera import Camera
from ..geometry.single_view import fit_inverse_depth_affine, metric_depth_from_affine
from ..log import get_logger
from ..semantics.labels import GROUND_IDS, SCENE_ID, STRUCTURE_IDS

log = get_logger("reconstruction")


def _ground_depth(cam: Camera, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    vs, us = np.mgrid[0:h, 0:w]
    uv = np.stack([us + 0.5, vs + 0.5], -1).reshape(-1, 2).astype(float)
    P, ok = cam.ground_intersect(uv)
    zc = cam.pose.to_cam(np.where(ok[:, None], P, 0))[:, 2]
    zg = np.where(ok, zc, np.nan).reshape(h, w)
    return zg, ok.reshape(h, w)


def superpixels(bgr: np.ndarray, size: int) -> np.ndarray:
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (3, 3), 0), cv2.COLOR_BGR2LAB)
    slic = cv2.ximgproc.createSuperpixelSLIC(lab, algorithm=cv2.ximgproc.SLICO, region_size=size)
    slic.iterate(6)
    slic.enforceLabelConnectivity(max(4, size * size // 4))
    return slic.getLabels().astype(np.int32)


def single_view_surface(bgr: np.ndarray, inv: np.ndarray, cam: Camera, walk_uv: np.ndarray,
                        sem: np.ndarray | None, max_planes: int = 8, smooth: bool = True) -> dict[str, Any]:
    h, w = inv.shape
    zg, gok = _ground_depth(cam, h, w)
    # ---- constraints for the affine fit: people's foot points (weight 3) + walking-surface pixels
    walk_uv = np.asarray(walk_uv, float).reshape(-1, 2)
    rng = np.random.default_rng(0)

    def sample(mask: np.ndarray, step: int, n: int) -> np.ndarray:
        vs, us = np.nonzero(mask[::step, ::step])
        if not len(vs):
            return np.zeros((0, 2))
        sel = rng.choice(len(vs), min(n, len(vs)), replace=False)
        return np.stack([us[sel] * step, vs[sel] * step], 1).astype(float)

    def fit_with(extra: np.ndarray):
        uv = np.concatenate([walk_uv, extra])
        uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)]
        zt = zg[uv[:, 1].astype(int), uv[:, 0].astype(int)]
        wts = np.r_[np.full(len(walk_uv), 3.0), np.ones(len(extra))][: len(uv)]
        return fit_inverse_depth_affine(inv, uv, zt, wts)

    sem_ground = (np.isin(sem, list(GROUND_IDS)) & gok) if sem is not None else None
    use_sem = sem is not None and sem_ground.sum() >= 0.05 * max(1, gok.sum())
    if use_sem:
        seed = sample(sem_ground, 3, 4000)
        seed_kind = "semantic walking surface"
    else:  # lower image part below the horizon, where most pixels are ground (RANSAC rejects the rest)
        seed = sample(gok & (np.arange(h)[:, None] > h * 0.55), 4, 4000)
        seed_kind = "lower image below the horizon"
    fit = fit_with(seed)
    if fit is None:
        raise RuntimeError("Could not align monocular depth to the ground plane.")
    # second pass: every pixel that agrees with the calibrated ground plane extends the fit to the
    # whole depth range of the visible ground (not only where the first seed was)
    z1 = metric_depth_from_affine(inv, fit["a"], fit["b"], 1e6)
    geo_ground = gok & np.isfinite(z1) & (np.abs(z1 - zg) < 0.06 * zg)
    semantics_note = None
    if geo_ground.sum() > 200:
        fit2 = fit_with(sample(geo_ground, 3, 5000))
        if fit2 is not None and fit2["rel_residual"] <= 1.2 * fit["rel_residual"] + 0.01:
            fit = {**fit2, "pass": 2, "seed": seed_kind}
            z1 = metric_depth_from_affine(inv, fit["a"], fit["b"], 1e6)
            geo_ground = gok & np.isfinite(z1) & (np.abs(z1 - zg) < 0.06 * zg)
    # semantic plausibility: walking surface labelled as building/wall means the segmentation model
    # does not fit this footage (domain mismatch); fall back to geometry-based labels
    if sem is not None and geo_ground.sum() > 500:
        wrong = np.isin(sem, list(STRUCTURE_IDS)) & geo_ground
        if wrong.sum() > 0.4 * geo_ground.sum():
            semantics_note = (f"{wrong.sum() / geo_ground.sum():.0%} of the pixels that lie on the calibrated ground "
                              "plane were labelled as buildings or walls: dense semantics ignored for this view")
            log.warning("Single-view surface: " + semantics_note)
            sem = None
    walk_d = zg[np.clip(walk_uv[:, 1].astype(int), 0, h - 1), np.clip(walk_uv[:, 0].astype(int), 0, w - 1)] \
        if len(walk_uv) else np.array([])
    walk_d = walk_d[np.isfinite(walk_d)]
    ground_d = zg[geo_ground]
    far = np.percentile(ground_d, 98) if len(ground_d) > 100 else 0.0
    max_depth = max(40.0, 2.5 * float(np.percentile(walk_d, 98)) if len(walk_d) > 10 else 0.0, 2.0 * float(far))
    if not len(walk_d) and not far:
        max_depth = 80.0
    z = metric_depth_from_affine(inv, fit["a"], fit["b"], max_depth)
    labels = sem.copy() if sem is not None else np.zeros((h, w), np.uint8)
    if sem is not None:
        z[sem == SCENE_ID["sky"]] = np.nan
    # ---- ground snapping
    ground = gok & np.isfinite(z) & (np.abs(z - zg) < 0.08 * zg)
    if sem is not None:
        ground &= ~np.isin(sem, list(STRUCTURE_IDS | {SCENE_ID["vegetation"], SCENE_ID["pole"], SCENE_ID["sky"]}))
    ground = cv2.morphologyEx(ground.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    z = np.where(ground, zg, z)
    plane_id = np.full((h, w), -1, np.int32)
    plane_id[ground] = 0
    if sem is None:
        labels[ground] = SCENE_ID["ground"]
    # ---- vertical planes
    planes = _vertical_planes(z, ground, labels if sem is not None else None, cam, plane_id, max_planes)
    # ---- superpixel smoothing of free surfaces
    if smooth:
        size = max(10, int(round(min(h, w) / 45)))
        sp = superpixels(bgr, size)
        z = _smooth_free(z, sp, plane_id, cam)
    # ---- geometric labels where semantics are missing
    if sem is None:
        for k, _pl in enumerate(planes, start=1):
            labels[plane_id == k] = SCENE_ID["wall"]
    report = {"affine_fit": fit, "max_depth_m": max_depth, "ground_fraction": float(ground.mean()),
              "vertical_planes": len(planes), "walk_points": int(len(walk_uv)), "fit_seed": seed_kind,
              "semantics_ignored": semantics_note}
    return {"z": z, "labels": labels, "plane_id": plane_id, "planes": planes, "ground_mask": ground,
            "report": report, "max_depth": max_depth}


def _vertical_planes(z, ground, sem, cam: Camera, plane_id, max_planes):
    h, w = z.shape
    vs, us = np.mgrid[0:h, 0:w]
    uv = np.stack([us + 0.5, vs + 0.5], -1).reshape(-1, 2).astype(float)
    Pw = cam.backproject(uv, np.nan_to_num(z, nan=1.0).reshape(-1)).reshape(h, w, 3)
    free = np.isfinite(z) & ~ground & (Pw[..., 1] > 0.2)
    if sem is not None:
        cand = free & np.isin(sem, list(STRUCTURE_IDS))
        if cand.sum() < 2000:
            cand = free & ~np.isin(sem, [SCENE_ID[k] for k in ("person", "car", "truck", "bus", "vegetation", "pole",
                                                                "sky", "bicycle", "motorcycle")])
    else:
        cand = free
    min_px = max(1500, int(0.004 * h * w))
    rng = np.random.default_rng(0)
    C = cam.center
    planes = []
    for _ in range(max_planes):
        idx = np.flatnonzero(cand.ravel())
        if len(idx) < min_px:
            break
        sub = idx[rng.choice(len(idx), min(len(idx), 20000), replace=False)]
        xz = Pw.reshape(-1, 3)[sub][:, [0, 2]]
        dep = z.ravel()[sub]
        best, best_n = None, 0
        for _ in range(300):
            i, j = rng.choice(len(sub), 2, replace=False)
            dvec = xz[j] - xz[i]
            nrm = np.linalg.norm(dvec)
            if nrm < 0.5:
                continue
            n = np.array([-dvec[1], dvec[0]]) / nrm
            inl = np.abs((xz - xz[i]) @ n) < 0.03 * dep + 0.08
            if inl.sum() > best_n:
                best, best_n = (n, float(n @ xz[i])), int(inl.sum())
        if best is None or best_n < 0.05 * len(sub):
            break
        n, off = best
        allxz = Pw[..., [0, 2]]
        inl = cand & (np.abs(allxz @ n - off) < 0.03 * z + 0.08)
        pts = allxz[inl]
        c0 = pts.mean(0)
        n = np.linalg.svd(pts - c0, full_matrices=False)[2][-1]
        off = float(n @ c0)
        inl = cand & (np.abs(allxz @ n - off) < 0.035 * z + 0.08)
        ncomp, comp, stats, _ = cv2.connectedComponentsWithStats(inl.astype(np.uint8), connectivity=8)
        keep = np.zeros_like(inl)
        for ci in range(1, ncomp):
            if stats[ci, cv2.CC_STAT_AREA] >= min_px:
                keep |= comp == ci
        if not keep.any():
            cand &= ~inl
            continue
        rays = cam.rays_world(uv.reshape(h, w, 2)[keep])
        facing = float(np.median(np.abs(rays[:, [0, 2]] @ n) / np.maximum(np.linalg.norm(rays[:, [0, 2]], axis=1), 1e-9)))
        if facing < 0.25:  # seen almost edge-on: not trustworthy as a plane
            cand &= ~keep
            continue
        rw = cam.rays_world(uv.reshape(h, w, 2)[keep], normalize=False)
        denom = rw[:, [0, 2]] @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (off - C[[0, 2]] @ n) / denom
        zc = s  # camera rays have z_cam = 1, so the ray parameter is the camera depth
        good = np.isfinite(zc) & (zc > 0)
        zk = z[keep]
        zk[good] = zc[good]
        z[keep] = zk
        k = len(planes) + 1
        plane_id[keep] = k
        if n @ (C[[0, 2]] - c0) < 0:  # normal points towards the camera
            n, off = -n, -off
        label = None
        if sem is not None:
            vals, cnt = np.unique(sem[keep], return_counts=True)
            label = int(vals[np.argmax(cnt)])
        planes.append({"id": k, "normal_xz": n.tolist(), "offset": off, "pixels": int(keep.sum()),
                       "facing": facing, "label": label})
        cand &= ~keep
    return planes


def _smooth_free(z, sp, plane_id, cam: Camera, max_rel_res: float = 0.04):
    """Replace depth inside each free superpixel by a robust local plane fit (in camera space)."""
    h, w = z.shape
    free = (plane_id < 0) & np.isfinite(z)
    if not free.any():
        return z
    vs, us = np.mgrid[0:h, 0:w]
    xy = cam.intr.normalized(np.stack([us + 0.5, vs + 0.5], -1).reshape(-1, 2).astype(float)).reshape(h, w, 2)
    out = z.copy()
    ids = np.unique(sp[free])
    lab = sp.ravel()
    order = np.argsort(lab, kind="stable")
    bounds = np.searchsorted(lab[order], ids)
    ends = np.r_[bounds[1:], len(order)]
    fz, fx, fy, ff = z.ravel(), xy[..., 0].ravel(), xy[..., 1].ravel(), free.ravel()
    for _i, (a, b) in enumerate(zip(bounds, ends)):
        pix = order[a:b]
        pix = pix[ff[pix]]
        if len(pix) < 12:
            continue
        Z = fz[pix]
        X = np.c_[fx[pix] * Z, fy[pix] * Z, Z]
        c0 = np.median(X, 0)
        try:
            nrm = np.linalg.svd(X - c0, full_matrices=False)[2][-1]
        except np.linalg.LinAlgError:
            continue
        res = np.abs((X - c0) @ nrm)
        if np.median(res / Z) > max_rel_res:
            continue  # not planar (vegetation, people): keep the network's depth
        d = nrm @ c0
        denom = nrm[0] * fx[pix] + nrm[1] * fy[pix] + nrm[2]
        with np.errstate(divide="ignore", invalid="ignore"):
            zn = d / denom
        good = np.isfinite(zn) & (zn > 0) & (np.abs(zn - Z) < 0.15 * Z)
        out.ravel()[pix[good]] = zn[good]
    return out
