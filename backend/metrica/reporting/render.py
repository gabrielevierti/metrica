"""Offscreen rendering for reports (pure numpy, no GPU / OpenGL needed).

Meshes are rendered by z-buffered splatting of densely sampled surface points (vertices
plus barycentric samples of each triangle), which is robust and dependency-free. It is a
documentation aid: the interactive viewer is the primary visual interface.
"""
from __future__ import annotations

import numpy as np

from ..geometry.camera import Camera, Intrinsics, Pose


def look_at(eye: np.ndarray, target: np.ndarray, up=np.array([0.0, 1.0, 0.0])) -> Pose:
    f = target - eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-9:
        r = np.array([1.0, 0, 0])
    r /= np.linalg.norm(r)
    d = np.cross(f, r)  # camera y (down) = forward x right
    R_wc = np.stack([r, d, f], 1)
    return Pose(R_wc, np.asarray(eye, float))


def sample_mesh(pos: np.ndarray, idx: np.ndarray, col: np.ndarray, per_face: int = 3, seed: int = 0):
    if len(idx) == 0:
        return pos, col
    rng = np.random.default_rng(seed)
    A, B, C = pos[idx[:, 0]], pos[idx[:, 1]], pos[idx[:, 2]]
    cA, cB, cC = col[idx[:, 0]].astype(float), col[idx[:, 1]].astype(float), col[idx[:, 2]].astype(float)
    pts, cols = [pos], [col.astype(float)]
    for _ in range(per_face):
        u = rng.random((len(idx), 1))
        v = rng.random((len(idx), 1))
        flip = (u + v) > 1
        u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
        pts.append(A + u * (B - A) + v * (C - A))
        cols.append(cA + u * (cB - cA) + v * (cC - cA))
    return np.concatenate(pts), np.concatenate(cols)


def splat(points: np.ndarray, colors: np.ndarray, cam: Camera, radius: int = 1,
          background=(250, 250, 252)) -> np.ndarray:
    w, h = cam.intr.width, cam.intr.height
    uv, z = cam.project(points)
    ok = (z > 0.05) & np.isfinite(uv).all(1)
    uv, z, c = uv[ok], z[ok], np.asarray(colors)[ok]
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = background
    zbuf = np.full(h * w, np.inf)
    owner = np.full(h * w, -1, np.int64)
    order = np.argsort(-z)  # far first, near overwrite
    uv, z, c = uv[order], z[order], c[order]
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            ui = np.round(uv[:, 0]).astype(int) + dx
            vi = np.round(uv[:, 1]).astype(int) + dy
            inb = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
            lin = vi[inb] * w + ui[inb]
            zz = z[inb]
            ids = np.flatnonzero(inb)
            better = zz < zbuf[lin]
            # stable "nearest wins": process ordered far->near, later writes win
            zbuf[lin[better]] = zz[better]
            owner[lin[better]] = ids[better]
    has = owner >= 0
    img.reshape(-1, 3)[has] = np.clip(c[owner[has]], 0, 255).astype(np.uint8)
    return img


def overview_camera(bmin: np.ndarray, bmax: np.ndarray, width: int = 1280, height: int = 800,
                    elevation_deg: float = 35.0, azimuth_deg: float = 30.0, hfov: float = 50.0) -> Camera:
    c = (np.asarray(bmin) + np.asarray(bmax)) / 2
    ext = float(np.linalg.norm(np.asarray(bmax) - np.asarray(bmin)))
    dist = ext * 0.6 / np.tan(np.radians(hfov) / 2)
    el, az = np.radians(elevation_deg), np.radians(azimuth_deg)
    eye = c + dist * np.array([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)])
    return Camera(Intrinsics.from_hfov(width, height, hfov), look_at(eye, c))


def top_camera(bmin: np.ndarray, bmax: np.ndarray, width: int = 1200, height: int = 1200, hfov: float = 40.0) -> Camera:
    c = (np.asarray(bmin) + np.asarray(bmax)) / 2
    ext = float(max(bmax[0] - bmin[0], bmax[2] - bmin[2]))
    dist = ext * 0.6 / np.tan(np.radians(hfov) / 2) + bmax[1]
    eye = np.array([c[0], dist, c[2] + 1e-3])
    return Camera(Intrinsics.from_hfov(width, height, hfov), look_at(eye, np.array([c[0], 0.0, c[2]]),
                                                                       up=np.array([0.0, 0.0, -1.0])))
