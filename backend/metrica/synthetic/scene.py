"""Procedural synthetic scenes with exact ground truth (for tests, demos and evaluation).

A small vectorised ray caster renders analytic primitives - a textured ground plane,
box buildings with window patterns, vertical cylinders (poles) and moving boxes/cylinders
(vehicle- and person-like proxies) - from known cameras. Because every quantity is known
(camera intrinsics and poses, primitive dimensions, object trajectories, metric scale),
reconstruction, scale recovery, primitive fitting and tracking geometry can be evaluated
quantitatively.

Limits (by design): the moving "people" and "vehicles" are geometric proxies; a COCO
detector is not expected to recognise them. Detection/tracking quality must be evaluated
on real, consented footage (see docs/EVALUATION.md).

World frame: Y up, metres, ground y = 0. Cameras use the OpenCV convention.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np

from ..geometry.camera import Intrinsics, look_rotation


# ------------------------------------------------------------------------------ textures
def _hash(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    h = (ix.astype(np.int64) * 374761393 + iy.astype(np.int64) * 668265263 + seed * 2147483647) & 0xFFFFFFFF
    h = (h ^ (h >> 13)) * 1274126177 & 0xFFFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFF).astype(np.float32) / 65535.0


def value_noise(u: np.ndarray, v: np.ndarray, scale: float, seed: int) -> np.ndarray:
    x, y = u * scale, v * scale
    ix, iy = np.floor(x), np.floor(y)
    fx, fy = x - ix, y - iy
    fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    a = _hash(ix, iy, seed)
    b = _hash(ix + 1, iy, seed)
    c = _hash(ix, iy + 1, seed)
    d = _hash(ix + 1, iy + 1, seed)
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def fbm(u, v, base: float, seed: int, octaves: int = 4) -> np.ndarray:
    out = np.zeros_like(u, dtype=np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        out += amp * value_noise(u, v, base * (2 ** o), seed + o * 17)
        tot += amp
        amp *= 0.5
    return out / tot


# ------------------------------------------------------------------------------ primitives
@dataclass
class Box:
    name: str
    center: list[float]   # x, y (bottom), z
    size: list[float]     # sx, sy, sz
    color: list[float]
    kind: str = "building"  # building | vehicle | object
    yaw_deg: float = 0.0
    seed: int = 1


@dataclass
class Cylinder:
    name: str
    base: list[float]   # x, y, z of bottom centre
    radius: float
    height: float
    color: list[float]
    kind: str = "pole"
    seed: int = 2


@dataclass
class Mover:
    name: str
    kind: str                     # person | vehicle
    waypoints: list[list[float]]  # [t, x, z]
    size: list[float]             # person: [radius, height]; vehicle: [length, height, width]
    color: list[float]

    def position(self, t: float) -> tuple[np.ndarray, float] | None:
        w = np.asarray(self.waypoints, float)
        if t < w[0, 0] or t > w[-1, 0]:
            return None
        x = np.interp(t, w[:, 0], w[:, 1])
        z = np.interp(t, w[:, 0], w[:, 2])
        k = int(np.clip(np.searchsorted(w[:, 0], t) - 1, 0, len(w) - 2))
        heading = float(np.arctan2(w[k + 1, 1] - w[k, 1], w[k + 1, 2] - w[k, 2]))
        return np.array([x, 0.0, z]), heading


@dataclass
class SyntheticScene:
    name: str
    boxes: list[Box] = field(default_factory=list)
    cylinders: list[Cylinder] = field(default_factory=list)
    movers: list[Mover] = field(default_factory=list)
    sun: list[float] = field(default_factory=lambda: [0.4, 0.8, 0.35])

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "boxes": [asdict(b) for b in self.boxes],
                "cylinders": [asdict(c) for c in self.cylinders], "movers": [asdict(m) for m in self.movers]}


def _ray_box(o, d, center, size, yaw):
    """Slab intersection with a yaw-rotated box. Returns (t, normal_world)."""
    c, s = np.cos(-yaw), np.sin(-yaw)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    lo_c = np.array([center[0], center[1] + size[1] / 2, center[2]])
    ol = (o - lo_c) @ R.T
    dl = d @ R.T
    half = np.array(size) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / dl
        t1 = (-half - ol) * inv
        t2 = (half - ol) * inv
    tmin = np.nanmax(np.minimum(t1, t2), axis=1)
    tmax = np.nanmin(np.maximum(t1, t2), axis=1)
    hit = (tmax >= np.maximum(tmin, 1e-4))
    t = np.where(hit, np.where(tmin > 1e-4, tmin, np.inf), np.inf)
    axis = np.argmax(np.minimum(t1, t2), axis=1)
    nl = np.zeros_like(dl)
    nl[np.arange(len(dl)), axis] = -np.sign(dl[np.arange(len(dl)), axis])
    return t, nl @ R


def _ray_cyl(o, d, base, r, h):
    ox, oz = o[:, 0] - base[0], o[:, 2] - base[2]
    dx, dz = d[:, 0], d[:, 2]
    a = dx * dx + dz * dz
    b = 2 * (ox * dx + oz * dz)
    c = ox * ox + oz * oz - r * r
    disc = b * b - 4 * a * c
    t = np.full(len(o), np.inf)
    ok = (disc >= 0) & (a > 1e-12)
    sq = np.sqrt(np.where(ok, disc, 0))
    t0 = (-b - sq) / np.where(ok, 2 * a, 1)
    y = o[:, 1] + t0 * d[:, 1]
    hit = ok & (t0 > 1e-4) & (y >= base[1]) & (y <= base[1] + h)
    t[hit] = t0[hit]
    nrm = np.stack([ox + t * dx, np.zeros_like(ox), oz + t * dz], 1) / r
    # top cap
    with np.errstate(divide="ignore", invalid="ignore"):
        tc = (base[1] + h - o[:, 1]) / d[:, 1]
    xc, zc = ox + tc * dx, oz + tc * dz
    cap = (tc > 1e-4) & (xc * xc + zc * zc <= r * r) & (tc < t)
    t[cap] = tc[cap]
    nrm[cap] = [0, 1, 0]
    return t, nrm


def render(scene: SyntheticScene, intr: Intrinsics, R_wc: np.ndarray, C: np.ndarray, t_sec: float,
           ss: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render RGB (uint8, BGR), depth (camera z, float32) and an instance-id map (int16)."""
    W, H = intr.width * ss, intr.height * ss
    us, vs = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    x = (us / ss - intr.cx) / intr.fx
    y = (vs / ss - intr.cy) / intr.fy
    dc = np.stack([x, y, np.ones_like(x)], -1).reshape(-1, 3)
    d = dc @ R_wc.T
    dn = d / np.linalg.norm(d, axis=1, keepdims=True)
    o = np.broadcast_to(C, dn.shape).astype(np.float64)
    n = len(dn)
    best_t = np.full(n, np.inf)
    nrm = np.zeros((n, 3))
    obj = np.full(n, -1, np.int32)
    color = np.zeros((n, 3), np.float32)
    # ground
    with np.errstate(divide="ignore", invalid="ignore"):
        tg = -o[:, 1] / dn[:, 1]
    g = (tg > 1e-4) & (dn[:, 1] < 0)
    best_t[g] = tg[g]
    nrm[g] = [0, 1, 0]
    obj[g] = 0
    prims: list[tuple[str, Any]] = []
    for b in scene.boxes:
        prims.append(("box", b))
    for c in scene.cylinders:
        prims.append(("cyl", c))
    for m in scene.movers:
        st = m.position(t_sec)
        if st is None:
            continue
        p, heading = st
        if m.kind == "person":
            prims.append(("cyl", Cylinder(m.name, [p[0], 0.0, p[2]], m.size[0], m.size[1], m.color, "person", 7)))
        else:
            L, Hh, Wd = m.size
            prims.append(("box", Box(m.name, [p[0], 0.0, p[2]], [Wd, Hh, L], m.color, "vehicle",
                                     float(np.degrees(heading)), 9)))
    for k, (kind, pr) in enumerate(prims, start=1):
        if kind == "box":
            t, nn = _ray_box(o, dn, pr.center, pr.size, np.radians(pr.yaw_deg))
        else:
            t, nn = _ray_cyl(o, dn, pr.base, pr.radius, pr.height)
        closer = t < best_t
        best_t[closer] = t[closer]
        nrm[closer] = nn[closer]
        obj[closer] = k
    P = o + dn * np.where(np.isfinite(best_t), best_t, 0)[:, None]
    # ---- textures
    gm = obj == 0
    if gm.any():
        u, v = P[gm, 0], P[gm, 2]
        base = 0.30 + 0.20 * fbm(u, v, 0.9, 3) + 0.10 * fbm(u, v, 7.0, 5)
        speck = 0.16 * (_hash(np.floor(u * 30), np.floor(v * 30), 77) - 0.5) \
            + 0.10 * (_hash(np.floor(u * 9), np.floor(v * 9), 78) - 0.5)
        stains = -0.12 * (fbm(u, v, 2.3, 91) > 0.68)
        tiles = ((np.floor(u / 1.5) + np.floor(v / 1.5)) % 2) * 0.05
        joints = -0.10 * ((np.abs(u % 1.5) < 0.03) | (np.abs(v % 1.5) < 0.03))
        lines = ((np.abs(((u + 0.0) % 7.0) - 3.5) < 0.08) & (np.abs(v % 6) < 3)) * 0.45
        g0 = base + speck + stains + tiles + joints + lines
        color[gm] = np.stack([g0, g0 * 0.98, g0 * 0.93], 1)
    for k, (kind, pr) in enumerate(prims, start=1):
        m = obj == k
        if not m.any():
            continue
        base = np.array(pr.color, np.float32)
        Pm = P[m]
        if kind == "box" and pr.kind == "building":
            nn = nrm[m]
            u = np.where(np.abs(nn[:, 0]) > 0.5, Pm[:, 2], Pm[:, 0])
            v = Pm[:, 1]
            row = np.floor(v * 4.0)
            col = np.floor(u * 2.0 + 0.5 * (row % 2))
            tint = 0.75 + 0.45 * _hash(row, col, pr.seed)
            mortar = ((v * 4.0) % 1.0 < 0.10) | ((u * 2.0 + 0.5 * (row % 2)) % 1.0 < 0.06)
            noise = 0.85 + 0.25 * fbm(u, v, 3.0, pr.seed) + 0.12 * (_hash(np.floor(u * 25), np.floor(v * 25), pr.seed + 5) - 0.5)
            col_ = base[None] * (tint * noise)[:, None]
            col_[mortar] = np.array([0.78, 0.76, 0.72]) * noise[mortar, None]
            wx, wy = (u % 3.0), (v % 3.2)
            frame_ = ((wx > 0.9) & (wx < 2.3) & (wy > 0.9) & (wy < 2.5) & (v > 0.8))
            win = ((wx > 1.0) & (wx < 2.2) & (wy > 1.0) & (wy < 2.4) & (v > 0.8))
            col_[frame_] = np.array([0.92, 0.92, 0.88])
            wtint = 0.6 + 0.8 * _hash(np.floor(u[win] / 3.0), np.floor(v[win] / 3.2), pr.seed + 9)
            col_[win] = np.array([0.16, 0.20, 0.27]) * wtint[:, None] * (0.85 + 0.3 * fbm(u[win], v[win], 5.0, 11))[:, None]
            col_[np.abs(nn[:, 1]) > 0.5] = base * 0.6
            color[m] = col_
        elif kind == "cyl":
            ang = np.arctan2(Pm[:, 2] - pr.base[2], Pm[:, 0] - pr.base[0])
            stripes = ((Pm[:, 1] * 3.0) % 1.0 < 0.15) * 0.25
            noise = 0.75 + 0.3 * fbm(ang * 3, Pm[:, 1], 2.0, pr.seed) \
                + 0.2 * (_hash(np.floor(ang * 8), np.floor(Pm[:, 1] * 12), pr.seed) - 0.5)
            color[m] = base[None] * noise[:, None] + stripes[:, None]
        else:
            uu, vv = Pm[:, 0] + Pm[:, 1], Pm[:, 2] + Pm[:, 1]
            noise = 0.8 + 0.25 * fbm(uu, vv, 4.0, pr.seed) + 0.25 * (_hash(np.floor(uu * 14), np.floor(vv * 14), pr.seed) - 0.5)
            color[m] = base[None] * noise[:, None]
    sun = np.array(scene.sun) / np.linalg.norm(scene.sun)
    lam = 0.55 + 0.45 * np.clip(nrm @ sun, 0, 1)
    color *= lam[:, None]
    sky = obj < 0
    color[sky] = np.stack([0.75 + 0.2 * (-dn[sky, 1]), 0.82 + 0.1 * (-dn[sky, 1]), 0.95 * np.ones(sky.sum())], 1)
    img = np.clip(color, 0, 1).reshape(H, W, 3)
    tt = np.where(np.isfinite(best_t), best_t, 0.0)
    zc = (tt[:, None] * dn) @ R_wc
    depth = np.where(np.isfinite(best_t), zc[:, 2], 0).reshape(H, W).astype(np.float32)
    inst = obj.reshape(H, W).astype(np.int16)
    if ss > 1:
        img = img.reshape(intr.height, ss, intr.width, ss, 3).mean((1, 3))
        depth = depth[::ss, ::ss]
        inst = inst[::ss, ::ss]
    bgr = (img[..., ::-1] * 255).astype(np.uint8)
    return bgr, depth, inst


# ------------------------------------------------------------------------------ canned scenes
def street_scene() -> SyntheticScene:
    s = SyntheticScene("street")
    s.boxes += [
        Box("building_A", [-12.0, 0.0, -24.0], [16.0, 11.0, 8.0], [0.72, 0.52, 0.42], seed=11),
        Box("building_B", [8.0, 0.0, -26.0], [14.0, 15.0, 9.0], [0.62, 0.64, 0.68], seed=21),
        Box("building_C", [22.0, 0.0, -12.0], [8.0, 9.0, 18.0], [0.80, 0.74, 0.58], seed=31),
        Box("kiosk", [3.5, 0.0, -11.0], [2.0, 2.6, 2.0], [0.30, 0.45, 0.35], kind="object", seed=41),
        Box("parked_car", [-6.5, 0.0, -9.0], [1.8, 1.5, 4.4], [0.65, 0.12, 0.10], kind="vehicle", seed=51),
        Box("planter_1", [-3.5, 0.0, -4.0], [1.2, 0.8, 1.2], [0.55, 0.45, 0.35], kind="object", seed=71),
        Box("planter_2", [4.5, 0.0, -5.0], [1.2, 0.8, 1.2], [0.50, 0.42, 0.33], kind="object", seed=72),
        Box("bus_shelter", [-9.5, 0.0, -14.5], [3.5, 2.5, 1.4], [0.40, 0.42, 0.50], kind="object", seed=73),
    ]
    s.cylinders += [
        Cylinder("pole_1", [-2.5, 0.0, -8.0], 0.10, 6.0, [0.55, 0.55, 0.58], seed=61),
        Cylinder("pole_2", [6.0, 0.0, -16.0], 0.12, 7.5, [0.45, 0.45, 0.48], seed=62),
        Cylinder("bollard", [1.0, 0.0, -6.0], 0.09, 0.9, [0.85, 0.75, 0.2], seed=63),
    ]
    s.movers += [
        Mover("walker_1", "person", [[0.0, -9.0, -14.0], [6.0, -1.0, -12.0], [12.0, 6.0, -9.0]], [0.24, 1.76],
              [0.15, 0.25, 0.55]),
        Mover("walker_2", "person", [[2.0, 9.0, -7.0], [10.0, 0.5, -13.0]], [0.22, 1.64], [0.6, 0.2, 0.2]),
        Mover("car_1", "vehicle", [[0.0, 18.0, -18.5], [9.0, -18.0, -18.5]], [4.5, 1.45, 1.85],
              [0.15, 0.30, 0.65]),
    ]
    return s


def static_camera(width: int = 640, height: int = 360) -> tuple[Intrinsics, np.ndarray, np.ndarray]:
    intr = Intrinsics.from_hfov(width, height, 70.0, source="synthetic")
    R = look_rotation(np.radians(22.0), np.radians(1.5), yaw=np.radians(4.0))
    return intr, R, np.array([0.5, 4.2, 2.0])


def walking_path(n: int, width: int = 640, height: int = 360) -> tuple[Intrinsics, list[np.ndarray], list[np.ndarray]]:
    """A hand-held camera walking ~12 m sideways and forward, gently panning and bobbing."""
    intr = Intrinsics.from_hfov(width, height, 66.0, source="synthetic")
    Rs, Cs = [], []
    for k in range(n):
        a = k / max(1, n - 1)
        C = np.array([-7.0 + 13.0 * a, 1.55 + 0.03 * np.sin(a * 18), 1.5 - 4.0 * a])
        yaw = np.radians(18.0 - 34.0 * a)
        pitch = np.radians(6.0 + 2.0 * np.sin(a * 7))
        roll = np.radians(1.5 * np.sin(a * 11))
        Rs.append(look_rotation(pitch, roll, yaw))
        Cs.append(C)
    return intr, Rs, Cs


def write_video(path: Path, frames: list[np.ndarray], fps: float = 15.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as out:
        st = out.add_stream("libx264" if "libx264" in av.codecs_available else "mpeg4", rate=Fraction(fps).limit_denominator(1000))
        h, w = frames[0].shape[:2]
        st.width, st.height, st.pix_fmt = w, h, "yuv420p"
        st.options = {"crf": "17", "preset": "medium"} if st.codec_context.name == "libx264" else {}
        for f in frames:
            for p in st.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="bgr24")):
                out.mux(p)
        for p in st.encode():
            out.mux(p)


def generate(out_dir: Path, kind: str = "walk", seconds: float = 6.0, fps: float = 12.0,
             width: int = 640, height: int = 360, ss: int | None = None) -> dict[str, Any]:
    """Render a synthetic video plus ground truth JSON. kind: 'walk' (moving), 'static' or 'pan'
    (fixed centre, panning).

    ``ss`` is the supersampling factor (anti-aliasing). Static scenes default to 2: single-view
    calibration relies on straight edges, and real cameras do not produce aliased staircases."""
    if ss is None:
        ss = 2 if kind in ("static", "pan") else 1
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = street_scene()
    n = int(round(seconds * fps))
    if kind == "static":
        intr, R, C = static_camera(width, height)
        Rs, Cs = [R] * n, [C] * n
    elif kind == "pan":  # fixed centre, the camera pans across the scene (PTZ-like)
        intr, _, C = static_camera(width, height)
        Rs = [look_rotation(np.radians(22.0), np.radians(1.5), yaw=np.radians(4.0 + 32.0 * (k / max(1, n - 1) - 0.5)))
              for k in range(n)]
        Cs = [C] * n
    else:
        intr, Rs, Cs = walking_path(n, width, height)
    frames, poses, movers = [], [], []
    for k in range(n):
        t = k / fps
        img, _, _ = render(scene, intr, Rs[k], Cs[k], t, ss=ss)
        frames.append(img)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = Rs[k], Cs[k]
        poses.append({"frame": k, "t": t, "world_from_cam": T.tolist()})
        for m in scene.movers:
            st = m.position(t)
            if st is not None:
                movers.append({"frame": k, "t": t, "name": m.name, "kind": m.kind,
                               "position": st[0].tolist(), "heading_rad": st[1]})
    vpath = out_dir / f"synthetic_{kind}.mp4"
    write_video(vpath, frames, fps)
    gt = {"schema": "metrica-synthetic-gt/1", "kind": kind, "fps": fps, "frames": n,
          "intrinsics": intr.to_json(), "poses": poses, "scene": scene.to_json(), "movers": movers,
          "world": "Y up, metres, ground y=0, OpenCV camera convention",
          "license": "Generated procedurally by METRICA; public domain (CC0)."}
    gpath = out_dir / f"synthetic_{kind}_gt.json"
    gpath.write_text(json.dumps(gt, indent=1))
    return {"video": str(vpath), "ground_truth": str(gpath), "frames": n}
