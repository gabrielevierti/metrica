"""Interoperable exports of the scene: GLB (meshes and primitives) and PLY (point layers)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import trimesh


def ear_clip(poly: np.ndarray) -> np.ndarray:
    """Triangulate a simple polygon (Nx2) by ear clipping. Returns (M,3) index triples."""
    pts = [tuple(p) for p in poly]
    n = len(pts)
    if n < 3:
        return np.zeros((0, 3), int)
    area = sum(pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1] for i in range(n))
    idx = list(range(n)) if area > 0 else list(range(n))[::-1]
    tris = []

    def inside(p, a, b, c):
        def s(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
        d1, d2, d3 = s(p, a, b), s(p, b, c), s(p, c, a)
        return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))

    guard = 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        found = False
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = pts[i0], pts[i1], pts[i2]
            if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) <= 0:
                continue
            if any(inside(pts[j], a, b, c) for j in idx if j not in (i0, i1, i2)):
                continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            found = True
            break
        if not found:
            break
    if len(idx) == 3:
        tris.append(tuple(idx))
    return np.array(tris, int).reshape(-1, 3)


def entity_meshes(e: dict[str, Any]) -> list[tuple[str, trimesh.Trimesh]]:
    g = e["geometry"]
    out = []
    t = g["type"]
    if t == "plane_polygon":
        o, u, v = np.asarray(g["origin"]), np.asarray(g["u_axis"]), np.asarray(g["v_axis"])
        for k, poly in enumerate(g["polygons_uv"]):
            P = np.asarray(poly)
            tri = ear_clip(P)
            if len(tri):
                V = o + P[:, :1] * u + P[:, 1:2] * v
                out.append((f"{e['id']}-{k}", trimesh.Trimesh(V, tri, process=False)))
    elif t == "cylinder":
        h = g["height"]
        m = trimesh.creation.cone(radius=g["radius_bottom"], height=h) if g["radius_top"] < 1e-4 else \
            trimesh.creation.cylinder(radius=(g["radius_bottom"] + g["radius_top"]) / 2, height=h, sections=24)
        m.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
        m.apply_translation(np.asarray(g["base"]) + [0, h / 2, 0])
        out.append((e["id"], m))
    elif t in ("vehicle_box", "box"):
        if t == "vehicle_box":
            L, W, H = g["length"], g["width"], g["height"]
            body = trimesh.creation.box([L, 0.55 * H, W])
            body.apply_translation([0, 0.275 * H, 0])
            cabin = trimesh.creation.box([0.55 * L, 0.45 * H, 0.92 * W])
            cabin.apply_translation([-0.05 * L, 0.775 * H, 0])
            m = trimesh.util.concatenate([body, cabin])
        else:
            m = trimesh.creation.box(g["size"])
            m.apply_translation([0, 0, 0])
        R = trimesh.transformations.rotation_matrix(-g["yaw_rad"], [0, 1, 0])
        m.apply_transform(R)
        c = np.asarray(g["center"])
        m.apply_translation([c[0], c[1] if t == "box" else 0.0, c[2]])
        out.append((e["id"], m))
    return out


def export_glb(path: Path, mesh_layers: list[dict[str, np.ndarray]], entities: list[dict[str, Any]]) -> dict[str, Any]:
    scene = trimesh.Scene()
    for L in mesh_layers:
        if "idx" not in L or len(L["idx"]) == 0:
            continue
        col = np.c_[L["col"], np.full(len(L["col"]), 255, np.uint8)] if "col" in L else None
        m = trimesh.Trimesh(L["pos"].astype(np.float64), L["idx"].astype(np.int64), vertex_colors=col, process=False)
        scene.add_geometry(m, node_name=L["name"], geom_name=L["name"])
    n_prim = 0
    for e in entities:
        for name, m in entity_meshes(e):
            scene.add_geometry(m, node_name=name, geom_name=name)
            n_prim += 1
    scene.export(str(path))
    return {"file": path.name, "primitives": n_prim, "layers": len(mesh_layers),
            "note": "Y-up, units as in scene.json coordinate_system; primitives without evidence colouring."}


def export_ply(path: Path, pos: np.ndarray, col: np.ndarray | None) -> None:
    pc = trimesh.PointCloud(pos.astype(np.float64), colors=col)
    pc.export(str(path))
