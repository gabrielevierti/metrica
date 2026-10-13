"""Quantitative evaluation of a processed case against synthetic ground truth.

The reconstruction lives in its own (gravity-aligned, possibly unscaled) frame. It is mapped
to the ground-truth world with a similarity transform estimated *only* from the camera poses,
so the geometry itself is never used to fit the alignment it is evaluated with. The rotation
is the chordal mean of the per-keyframe rotation offsets (camera centres alone leave the roll
about a nearly straight walking path unconstrained); scale and translation then come from the
camera centres. For static cameras (one centre) the known camera height/orientation are used.

Metrics
-------
* camera trajectory: absolute trajectory error (ATE, RMSE) after Sim3 alignment
* scale: recovered scale / true scale (only meaningful when the case claims metric scale)
* surface: distance from reconstructed surface vertices to the nearest GT surface, measured
  along camera rays (accuracy) and the fraction of GT-visible surface reconstructed (completeness)
* poles: detection recall/precision (match radius 0.5 m), position / height / radius errors
* facades: plane normal angle error and plane offset error for matched GT building faces
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..geometry.camera import Camera, Intrinsics, Pose
from ..reconstruction.layers import read_layer
from ..synthetic.scene import Box, Cylinder, SyntheticScene, render


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / ((xs ** 2).sum() / len(src))) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def pose_similarity(est: np.ndarray, ref: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Sim3 (s, R, t) mapping estimated world_from_cam poses (N,4,4) onto reference poses."""
    A = sum(r[:3, :3] @ e[:3, :3].T for e, r in zip(est, ref))
    U, _, Vt = np.linalg.svd(A)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    ce, cr = est[:, :3, 3], ref[:, :3, 3]
    xe = (R @ (ce - ce.mean(0)).T).T
    xr = cr - cr.mean(0)
    s = float((xe * xr).sum() / max((xe ** 2).sum(), 1e-12))
    t = cr.mean(0) - s * R @ ce.mean(0)
    return s, R, t


def _scene_from_gt(gt: dict[str, Any]) -> SyntheticScene:
    sc = gt["scene"]
    s = SyntheticScene(sc["name"])
    s.boxes = [Box(**b) for b in sc["boxes"]]
    s.cylinders = [Cylinder(**c) for c in sc["cylinders"]]
    s.movers = []
    return s


def evaluate_case(case_root: Path, gt_path: Path, video_id: str = "v1") -> dict[str, Any]:
    case_root = Path(case_root)
    gt = json.loads(Path(gt_path).read_text())
    geo = json.loads((case_root / "derived" / "geometry" / f"{video_id}.json").read_text())
    out: dict[str, Any] = {"case": str(case_root), "ground_truth": str(gt_path), "video": video_id,
                           "claimed_scale": geo["scale"]["status"]}
    gt_poses = {p["frame"]: np.asarray(p["world_from_cam"]) for p in gt["poses"]}
    if geo["mode"] == "sfm":
        frames = geo["poses"]["frames"]
        src = geo["poses"]["source"]
        M = np.asarray(geo["poses"]["world_from_cam"])
        kf = [k for k, s in enumerate(src) if s == "keyframe"]
        est = M[kf, :3, 3]
        ref = np.array([gt_poses[frames[k]][:3, 3] for k in kf])
        s, R, t = pose_similarity(M[kf], np.array([gt_poses[frames[k]] for k in kf]))
        aligned = (s * (R @ est.T)).T + t
        err = np.linalg.norm(aligned - ref, axis=1)
        s_u, R_u, t_u = umeyama(est, ref)
        err_u = np.linalg.norm((s_u * (R_u @ est.T)).T + t_u - ref, axis=1)
        out["trajectory"] = {"keyframes": len(kf), "ate_rmse_m": float(np.sqrt((err ** 2).mean())),
                             "ate_rmse_m_umeyama_centres_only": float(np.sqrt((err_u ** 2).mean())),
                             "alignment": "rotation from camera orientations, scale/translation from centres",
                             "ate_max_m": float(err.max()),
                             "path_length_m": float(np.linalg.norm(np.diff(ref, axis=0), axis=1).sum())}
        all_est = M[:, :3, 3]
        all_ref = np.array([gt_poses[f][:3, 3] for f in frames])
        e2 = np.linalg.norm((s * (R @ all_est.T)).T + t - all_ref, axis=1)
        out["trajectory"]["all_frames_rmse_m"] = float(np.sqrt((e2 ** 2).mean()))
        out["trajectory"]["by_source"] = {k: float(np.sqrt((e2[[i for i, x in enumerate(src) if x == k]] ** 2).mean()))
                                          for k in set(src)}
    elif geo.get("poses"):  # rotating camera: fixed centre, per-frame orientation
        frames = geo["poses"]["frames"]
        M = np.asarray(geo["poses"]["world_from_cam"])
        Gs = np.array([gt_poses[f] for f in frames])
        A = sum(g[:3, :3] @ m[:3, :3].T for m, g in zip(M, Gs))
        U, _, Vt = np.linalg.svd(A)
        D = np.eye(3)
        D[2, 2] = np.sign(np.linalg.det(U @ Vt))
        R = U @ D @ Vt
        errs = [float(np.degrees(np.arccos(np.clip((np.trace((R @ m[:3, :3]).T @ g[:3, :3]) - 1) / 2, -1, 1))))
                for m, g in zip(M, Gs)]
        T = np.asarray(geo["static_pose"]["world_from_cam"])
        s = float(Gs[0][1, 3] / max(T[1, 3], 1e-9))
        t = Gs[0][:3, 3] - s * R @ T[:3, 3]
        src = geo["poses"]["source"]
        out["rotation"] = {"frames_with_pose": len(frames), "frames_total": len(gt_poses),
                           "orientation_error_deg_median": float(np.median(errs)),
                           "orientation_error_deg_max": float(np.max(errs)),
                           "by_source": {k: float(np.median([e for e, x in zip(errs, src) if x == k]))
                                         for k in set(src)},
                           "note": "per-frame orientation after one common rotation (estimated from all posed frames)"}
        out["camera_height_estimated"] = float(T[1, 3])
        out["camera_height_true"] = float(Gs[0][1, 3])
    else:
        T = np.asarray(geo["static_pose"]["world_from_cam"])
        G = gt_poses[0]
        # rotation that maps the estimated camera orientation onto the true one; translation of the centre
        R = G[:3, :3] @ T[:3, :3].T
        s = 1.0
        t = G[:3, 3] - R @ T[:3, 3]
        # a single static camera defines the world's heading: only the tilt (pitch and roll, i.e. the
        # direction of gravity in the camera frame) is observable and evaluated
        up_e = T[:3, :3].T @ np.array([0.0, 1.0, 0.0])
        up_g = G[:3, :3].T @ np.array([0.0, 1.0, 0.0])
        out["tilt_error_deg"] = float(np.degrees(np.arccos(np.clip(up_e @ up_g, -1, 1))))
        out["camera_height_estimated"] = float(T[1, 3])
        out["camera_height_true"] = float(G[1, 3])
        s = float(G[1, 3] / max(T[1, 3], 1e-9))
        t = G[:3, 3] - s * R @ T[:3, 3]
    out["scale_true_over_estimated"] = float(s)
    out["scale_error_rel"] = float(abs(1.0 / s - 1.0)) if out["claimed_scale"] != "ARBITRARY" else None

    def to_gt(X):
        return (s * (R @ np.asarray(X, float).T)).T + t

    scene = _scene_from_gt(gt)
    intr = Intrinsics.from_json(gt["intrinsics"])
    # ---- surface accuracy along GT camera rays
    unit = geo["unit_id"]
    rdir = case_root / "derived" / "reconstruction" / unit
    recon = json.loads((rdir / "reconstruction.json").read_text())
    surf = next((l for l in recon["layers"] if l["name"] == "surface"), None)
    if surf is not None:
        L = read_layer(rdir, surf)
        V = to_gt(L["pos"])
        sel_frames = sorted(gt_poses)[:: max(1, len(gt_poses) // 6)]
        errs, comp = [], []
        for f in sel_frames:
            P = gt_poses[f]
            cam = Camera(intr, Pose(P[:3, :3], P[:3, 3]))
            _, depth, _ = render(scene, intr, P[:3, :3], P[:3, 3], 1e9)
            uv, z = cam.project(V)
            ui, vi = uv[:, 0].astype(int), uv[:, 1].astype(int)
            ok = (z > 0.1) & (ui >= 0) & (ui < intr.width) & (vi >= 0) & (vi < intr.height)
            dz = depth[vi[ok], ui[ok]]
            vis = (dz > 0) & (z[ok] < dz * 1.25)  # vertices in front-ish of the true visible surface
            errs.append(np.abs(z[ok][vis] - dz[vis]) / dz[vis])
            # completeness: fraction of GT pixels (stride 8) that receive a vertex within 5%
            hit = np.zeros(depth.shape, bool)
            close = np.abs(z[ok] - dz) < 0.05 * dz
            hit[vi[ok][close], ui[ok][close]] = True
            import cv2
            hit = cv2.dilate(hit.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            valid = depth > 0
            comp.append(float(hit[valid].mean()))
        e = np.concatenate(errs) if errs else np.array([])
        out["surface"] = {"vertices": int(len(V)), "rel_depth_error_median": float(np.median(e)) if len(e) else None,
                          "rel_depth_error_p90": float(np.percentile(e, 90)) if len(e) else None,
                          "within_5pct": float((e < 0.05).mean()) if len(e) else None,
                          "completeness_5pct": float(np.mean(comp)) if comp else None,
                          "note": "relative depth error along true camera rays, after Sim3 alignment"}
    # ---- poles
    objs = json.loads((case_root / "derived" / "objects" / unit / "objects.json").read_text())["entities"]
    poles = [e for e in objs if e["category"] == "pole"]
    gt_poles = [c for c in scene.cylinders if c.kind == "pole" or c.name.startswith(("pole", "bollard"))]
    matches, used = [], set()
    for c in gt_poles:
        best, bd = None, 0.5
        for k, p in enumerate(poles):
            if k in used:
                continue
            b = to_gt(np.asarray(p["geometry"]["base"])[None])[0]
            d = float(np.hypot(b[0] - c.base[0], b[2] - c.base[2]))
            if d < bd:
                best, bd = k, d
        if best is not None:
            used.add(best)
            g = poles[best]["geometry"]
            matches.append({"gt": c.name, "entity": poles[best]["id"], "position_error_m": bd,
                            "height_est_m": g["height"] * s, "height_true_m": c.height,
                            "radius_est_m": (g["radius_bottom"] + g["radius_top"]) / 2 * s, "radius_true_m": c.radius})
    unmatched = []
    for k, p in enumerate(poles):
        if k in used:
            continue
        b = to_gt(np.asarray(p["geometry"]["base"])[None])[0]
        d = [(float(np.hypot(b[0] - c.base[0], b[2] - c.base[2])), c.name) for c in gt_poles]
        dn, name = min(d) if d else (None, None)
        unmatched.append({"entity": p["id"], "base_gt_frame": b.round(3).tolist(), "nearest_gt": name,
                          "distance_m": dn, "height_est_m": p["geometry"]["height"] * s})
    out["poles"] = {"gt": len(gt_poles), "detected": len(poles), "matched": len(matches),
                    "recall": len(matches) / max(1, len(gt_poles)),
                    "precision": len(matches) / max(1, len(poles)), "match_radius_m": 0.5, "matches": matches,
                    "unmatched_detections": unmatched}
    # ---- facades vs GT building faces
    faces = []
    for b in scene.boxes:
        if b.kind != "building":
            continue
        cx, _, cz = b.center
        sx, _, sz = b.size
        for n, off_pt in (([1, 0], [cx + sx / 2, cz]), ([-1, 0], [cx - sx / 2, cz]),
                          ([0, 1], [cx, cz + sz / 2]), ([0, -1], [cx, cz - sz / 2])):
            faces.append((b.name, np.array(n, float), np.array(off_pt)))
    fac = [e for e in objs if e["category"] == "facade"]
    fm = []
    for e in fac:
        g = e["geometry"]
        n3 = R @ np.asarray(g["normal"])
        n2 = np.array([n3[0], n3[2]]) / max(np.hypot(n3[0], n3[2]), 1e-9)
        o = to_gt(np.asarray(g["origin"])[None])[0]
        poly = np.asarray(g["polygons_uv"][0])
        cu = poly[:, 0].mean()
        centre = o + s * (R @ np.asarray(g["u_axis"])) * cu
        best = None
        for name, n, p in faces:
            ang = np.degrees(np.arccos(np.clip(abs(n2 @ n), -1, 1)))
            dist = abs((centre[[0, 2]] - p) @ n)
            if ang < 20 and dist < 2.0 and (best is None or dist < best[2]):
                best = (name, ang, dist)
        fm.append({"entity": e["id"], "matched_face": best[0] if best else None,
                   "normal_error_deg": best[1] if best else None, "offset_error_m": best[2] if best else None})
    matched = [m for m in fm if m["matched_face"]]
    out["facades"] = {"detected": len(fac), "matched": len(matched),
                      "normal_error_deg_median": float(np.median([m["normal_error_deg"] for m in matched])) if matched else None,
                      "offset_error_m_median": float(np.median([m["offset_error_m"] for m in matched])) if matched else None,
                      "details": fm}
    return out
