"""Gravity and ground alignment of an SfM reconstruction, and scale estimation.

SfM returns an arbitrary similarity frame. METRICA brings it to a *local world* with
Y up and the ground plane at y = 0:

1. A first 'up' estimate is the mean image-up direction of the cameras (hand-held and
   vehicle cameras are mostly upright).
2. The ground plane is fitted (RANSAC) to sparse points - those labelled road/sidewalk/
   terrain by semantics when available - requiring a normal close to the first estimate
   and the cameras on its upper side.
3. Without a plausible ground plane, the camera up direction is kept and the ground is
   placed below the cameras at an assumed camera height (flagged, RELATIVE scale).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .camera import Pose


def similarity(R: np.ndarray, t: np.ndarray, s: float = 1.0) -> np.ndarray:
    S = np.eye(4)
    S[:3, :3] = s * R
    S[:3, 3] = t
    return S


def apply_similarity(S: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.asarray(X, float) @ S[:3, :3].T + S[:3, 3]


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation mapping unit vector a onto unit vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        if c > 0:
            return np.eye(3)
        axis = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        v = np.cross(a, axis)
        v /= np.linalg.norm(v)
        return 2 * np.outer(v, v) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def fit_ground(points: np.ndarray, up0: np.ndarray, centers: np.ndarray, max_angle_deg: float = 25.0,
               seed: int = 0) -> dict[str, Any] | None:
    if len(points) < 30:
        return None
    rng = np.random.default_rng(seed)
    ext = np.linalg.norm(np.percentile(points, 95, 0) - np.percentile(points, 5, 0))
    thr = max(ext * 0.004, 1e-6)
    cos_max = np.cos(np.radians(max_angle_deg))
    best = None
    for _ in range(1500):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n /= nn
        if n @ up0 < 0:
            n = -n
        if n @ up0 < cos_max:
            continue
        side = (centers - a) @ n
        if (side > 0).mean() < 0.9:
            continue
        d = np.abs((points - a) @ n)
        inl = d < thr
        # prefer the lowest supported plane: ties broken by height below cameras
        score = inl.sum()
        if best is None or score > best[0]:
            best = (score, n, a, inl)
    if best is None or best[0] < max(20, 0.03 * len(points)):
        return None
    _, n, a, inl = best
    Q = points[inl]
    c0 = Q.mean(0)
    n2 = np.linalg.svd(Q - c0)[2][-1]
    if n2 @ n < 0:
        n2 = -n2
    rms = float(np.sqrt(np.mean(((Q - c0) @ n2) ** 2)))
    return {"normal": n2, "point": c0, "inliers": int(inl.sum()), "inlier_ratio": float(inl.mean()),
            "rms": rms, "threshold": thr, "angle_to_camera_up_deg": float(np.degrees(np.arccos(np.clip(n2 @ up0, -1, 1))))}


def gravity_align(points: np.ndarray, ground_mask: np.ndarray | None, poses: list[Pose],
                  nominal_camera_height: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Similarity (rotation+translation, scale 1) from SfM frame to local Y-up frame."""
    centers = np.array([p.C for p in poses])
    ups = np.array([-p.R_wc[:, 1] for p in poses])  # image 'up' (-y cam) in world
    up0 = ups.mean(0)
    up0 /= np.linalg.norm(up0)
    report: dict[str, Any] = {"camera_up_spread_deg": float(np.degrees(np.arccos(np.clip(ups @ up0, -1, 1))).mean())}
    g = None
    if ground_mask is not None and ground_mask.sum() >= 30:
        g = fit_ground(points[ground_mask], up0, centers)
        if g:
            report["ground_source"] = "semantic ground points"
    if g is None:
        # lower part of the cloud only: below the cameras along up0
        h = (points - centers.mean(0)) @ up0
        low = h < np.percentile(h, 40)
        g = fit_ground(points[low], up0, centers)
        if g:
            report["ground_source"] = "lowest dominant plane"
    if g is not None:
        up = g["normal"]
        ground_pt = g["point"]
        report.update({"ground_found": True, "ground_inliers": g["inliers"], "ground_rms": g["rms"],
                       "ground_angle_to_camera_up_deg": g["angle_to_camera_up_deg"]})
    else:
        up = up0
        # ground placed below the median camera at the nominal height (in SfM units: unknown!)
        cam_h = np.median((centers - centers.mean(0)) @ up)
        span = np.linalg.norm(np.ptp(centers, 0)) + 1e-9
        ground_pt = centers.mean(0) + (cam_h - 0.3 * span) * up
        report.update({"ground_found": False,
                       "note": "No ground plane found in the sparse points; ground placed below the cameras."})
    R = rotation_between(up, np.array([0.0, 1.0, 0.0]))
    # heading: first camera looks towards -Z
    fwd = R @ poses[0].R_wc[:, 2]
    yaw = np.arctan2(fwd[0], -fwd[2])
    cy, sy = np.cos(yaw), np.sin(yaw)
    Ry = np.array([[cy, 0, -sy], [0, 1, 0], [sy, 0, cy]])
    R = Ry @ R
    origin = R @ poses[0].C
    origin_ground_y = (R @ ground_pt)[1]
    t = -np.array([origin[0], origin_ground_y, origin[2]])
    S = similarity(R, t, 1.0)
    heights = np.array([apply_similarity(S, p.C[None])[0, 1] for p in poses])
    report["camera_heights_su"] = {"median": float(np.median(heights)), "min": float(heights.min()),
                                   "max": float(heights.max())}
    if not report.get("ground_found"):
        # rescale so the median camera sits at the nominal height (assumption)
        s = nominal_camera_height / max(float(np.median(heights)), 1e-9)
        S = similarity(np.eye(3), np.zeros(3), s) @ S
        report["assumed_camera_height"] = nominal_camera_height
    return S, report


def person_heights(observations: list[dict[str, Any]], poses: dict[int, Pose], cam_intr, min_box_h: float = 40):
    """Heights (local units) of standing people: foot ray on y=0, head on the vertical through it."""
    from .camera import Camera

    out: dict[str, list[float]] = {}
    for o in observations:
        pose = poses.get(o["frame"])
        if pose is None:
            continue
        x1, y1, x2, y2 = o["bbox"]
        if y2 - y1 < min_box_h or not 1.8 <= (y2 - y1) / max(x2 - x1, 1) <= 5:
            continue
        if x1 < 3 or y1 < 3 or x2 > cam_intr.width - 3 or y2 > cam_intr.height - 3:
            continue
        cam = Camera(cam_intr, pose)
        u = (x1 + x2) / 2
        F, ok = cam.ground_intersect(np.array([[u, y2]]))
        if not ok[0]:
            continue
        F = F[0]
        d = cam.rays_world(np.array([[u, y1]]))[0]
        C = pose.C
        # closest point on the vertical line through F to the head ray
        a = np.array([0.0, 1.0, 0.0])
        w0 = C - F
        b = a @ d
        denom = 1 - b * b
        if denom < 1e-6:
            continue
        s_line = (a @ w0 - b * (d @ w0)) / denom
        if s_line <= 0:
            continue
        out.setdefault(o["track_id"], []).append(float(s_line))
    return out
