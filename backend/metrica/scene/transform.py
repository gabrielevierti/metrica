"""Applying a similarity (unit frame -> case world, including the analyst scale factor) to
every kind of scene element. Dimensions and uncertainties scale with the similarity."""
from __future__ import annotations

import copy
from typing import Any

import numpy as np


class Similarity:
    def __init__(self, M: np.ndarray, k: float = 1.0):
        M = np.asarray(M, float)
        s = float(np.cbrt(abs(np.linalg.det(M[:3, :3]))))
        self.R = M[:3, :3] / s
        self.s = s * k
        self.t = M[:3, 3] * k
        self.yaw = float(np.arctan2(self.R[0, 2], self.R[0, 0]))  # rotation about +Y

    @property
    def matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.s * self.R
        T[:3, 3] = self.t
        return T

    def point(self, p) -> np.ndarray:
        return self.s * (np.asarray(p, float) @ self.R.T) + self.t

    def vec(self, v) -> np.ndarray:
        return np.asarray(v, float) @ self.R.T

    def xz(self, pts) -> np.ndarray:
        P = np.asarray(pts, float)
        P3 = np.c_[P[:, 0], np.zeros(len(P)), P[:, 1]]
        Q = self.point(P3)
        return Q[:, [0, 2]]


def _scale_dims(dims: dict[str, Any], s: float, unit: str) -> dict[str, Any]:
    out = {}
    for k, d in dims.items():
        d = dict(d)
        if isinstance(d.get("value"), (int, float)) and k not in ("taper_ratio",):
            area = isinstance(d.get("unit"), str) and d["unit"].endswith("^2")
            f = s * s if area else s
            d["value"] = d["value"] * f
            if d.get("sigma") is not None:
                d["sigma"] = d["sigma"] * f
            d["unit"] = f"{unit}^2" if area else unit
        out[k] = d
    return out


def transform_entity(e: dict[str, Any], S: Similarity, unit: str) -> dict[str, Any]:
    e = copy.deepcopy(e)
    g = e["geometry"]
    t = g["type"]
    if t == "plane_polygon":
        g["origin"] = S.point(g["origin"]).tolist()
        g["u_axis"] = S.vec(g["u_axis"]).tolist()
        g["v_axis"] = S.vec(g["v_axis"]).tolist()
        g["normal"] = S.vec(g["normal"]).tolist()
        g["polygons_uv"] = [(np.asarray(p) * S.s).round(4).tolist() for p in g["polygons_uv"]]
        for o in e.get("openings", []):
            for k in ("u0", "u1", "v0", "v1"):
                o[k] *= S.s
        for p in e.get("inferred_patches", []):
            p["polygon_uv"] = (np.asarray(p["polygon_uv"]) * S.s).round(4).tolist()
        if e.get("texture"):
            e["texture"]["u_range"] = [v * S.s for v in e["texture"]["u_range"]]
            e["texture"]["v_range"] = [v * S.s for v in e["texture"]["v_range"]]
        for rel in e.get("relations", []):
            if "corner_xz" in rel:
                rel["corner_xz"] = S.xz([rel["corner_xz"]])[0].round(4).tolist()
    elif t == "cylinder":
        g["base"] = S.point(g["base"]).tolist()
        for k in ("height", "radius_bottom", "radius_top"):
            g[k] *= S.s
        g["observed_range"] = [v * S.s for v in g["observed_range"]]
        for ip in e.get("inferred_parts", []):
            ip["range"] = [v * S.s for v in ip["range"]]
    elif t in ("vehicle_box", "box"):
        g["center"] = S.point(g["center"]).tolist()
        g["yaw_rad"] = g["yaw_rad"] - S.yaw
        if t == "vehicle_box":
            for k in ("length", "width", "height"):
                g[k] *= S.s
        else:
            g["size"] = [v * S.s for v in g["size"]]
    elif t == "bbox":
        corners = np.array([[x, y, z] for x in (g["min"][0], g["max"][0]) for y in (g["min"][1], g["max"][1])
                            for z in (g["min"][2], g["max"][2])])
        C = S.point(corners)
        g["min"], g["max"] = C.min(0).round(4).tolist(), C.max(0).round(4).tolist()
    elif t == "ground_plane":
        g["observed_polygons_xz"] = [S.xz(p).round(4).tolist() for p in g["observed_polygons_xz"]]
        g["inferred_polygons_xz"] = [S.xz(p).round(4).tolist() for p in g["inferred_polygons_xz"]]
        for r in e.get("regions", {}).values():
            r["polygons_xz"] = [S.xz(p).round(4).tolist() for p in r["polygons_xz"]]
            r["area"] *= S.s * S.s
    if "dimensions" in e:
        e["dimensions"] = _scale_dims(e["dimensions"], S.s, unit)
    if e.get("evidence", {}).get("position_spread") is not None:
        e["evidence"]["position_spread"] *= S.s
    return e


def transform_agent(tr: dict[str, Any], S: Similarity, offset: float) -> dict[str, Any]:
    tr = copy.deepcopy(tr)
    for s in tr["samples"]:
        s["p"] = S.point(s["p"]).round(4).tolist()
        s["s"] = s["s"] * S.s if s["s"] is not None else None
        s["t"] = round(s["t"] + offset, 4)
    for s in tr.get("smoothed", []):
        s["p"] = S.point(s["p"]).round(4).tolist()
        s["t"] = round(s["t"] + offset, 4)
    for s in tr.get("speed", []):
        s["t"] = round(s["t"] + offset, 4)
        s["v"] *= S.s
        s["s"] *= S.s
    if tr.get("summary"):
        for k in ("path_length_observed", "displacement", "speed_median", "speed_max", "position_sigma_median"):
            if tr["summary"].get(k) is not None:
                tr["summary"][k] *= S.s
    if tr.get("stature"):
        tr["stature"] = {**tr["stature"], "median": tr["stature"]["median"] * S.s, "iqr": tr["stature"]["iqr"] * S.s}
    tr["first_t"] += offset
    tr["last_t"] += offset
    for g in tr.get("gaps", []):
        g["t0"] += offset
        g["t1"] += offset
    return tr
