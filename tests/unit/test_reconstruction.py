"""Layer IO, primitive fitting and coverage helpers."""
from __future__ import annotations

import numpy as np

from metrica.reconstruction.layers import EVI, EVIDENCE, compact, read_layer, write_layer
from metrica.reconstruction.primitives import (
    cluster_grid,
    extent_polygons,
    fit_circle,
    min_area_rect,
    plane_frame,
    vertical_planes,
)


def test_layer_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(100, 3)).astype(np.float32)
    col = rng.integers(0, 255, (100, 3), dtype=np.uint8)
    idx = rng.integers(0, 100, (40, 3)).astype(np.uint32)
    evi = rng.integers(0, len(EVIDENCE), 100).astype(np.uint8)
    meta = write_layer(tmp_path, "L", pos, col, idx, evi, np.full(100, 200, np.uint8), np.zeros(100, np.uint8))
    assert meta["stats"]["vertices"] == 100 and meta["stats"]["faces"] == 40
    assert sum(meta["stats"]["evidence_vertices"].values()) == 100
    back = read_layer(tmp_path, meta)
    assert np.array_equal(back["pos"], pos) and np.array_equal(back["idx"], idx) and np.array_equal(back["evi"], evi)
    p2, i2, e2 = compact(pos, idx[:5].astype(np.int64), evi)
    assert len(p2) == len(np.unique(idx[:5])) and i2.max() < len(p2)
    assert EVI["INFERRED"] == 4


def _wall(n, origin, direction, length, height, rng, noise=0.01):
    u = rng.uniform(0, length, n)
    v = rng.uniform(0, height, n)
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    nrm = np.array([-d[1], d[0]])
    xz = np.asarray(origin) + u[:, None] * d + rng.normal(0, noise, n)[:, None] * nrm
    return np.c_[xz[:, 0], v, xz[:, 1]]


def test_vertical_planes_finds_two_walls():
    rng = np.random.default_rng(1)
    P = np.vstack([_wall(3000, (0, -10), (1, 0), 12, 6, rng), _wall(2000, (12, -10), (0, -1), 8, 6, rng),
                   rng.uniform([-5, 0, -25], [20, 6, 0], (500, 3))])
    planes = vertical_planes(P, thr=0.05, min_inliers=300)
    assert len(planes) >= 2
    normals = sorted([abs(p["normal_xz"][0]) for p in planes[:2]])
    assert normals[0] < 0.05 and normals[1] > 0.99  # one wall along X, one along Z
    o, u, v, n = plane_frame(planes[0]["normal_xz"], planes[0]["offset"])
    assert np.allclose(v, [0, 1, 0]) and abs(u @ n) < 1e-9


def test_extent_polygons_keep_unobserved_holes():
    rng = np.random.default_rng(2)
    uv = rng.uniform([0, 0], [10, 5], (20000, 2))
    hole = (uv[:, 0] > 4) & (uv[:, 0] < 6) & (uv[:, 1] > 1) & (uv[:, 1] < 3)
    polys, info = extent_polygons(uv[~hole], cell=0.2, close_cells=1)
    assert info["occupied_cells"] > 0
    # outer boundary plus a hole contour: the unobserved region is not filled in
    assert len(polys) >= 2


def test_fit_circle_on_partial_arc():
    rng = np.random.default_rng(3)
    ang = rng.uniform(0, np.pi * 0.8, 200)  # only the visible side of a pole
    xz = np.c_[2 + 0.12 * np.cos(ang), -7 + 0.12 * np.sin(ang)] + rng.normal(0, 0.003, (200, 2))
    r = fit_circle(xz, thr=0.01)
    assert r is not None and abs(r["radius"] - 0.12) < 0.02 and np.allclose(r["center"], [2, -7], atol=0.02)
    assert 100 < r["arc_coverage_deg"] < 180


def test_min_area_rect_and_clusters():
    rng = np.random.default_rng(4)
    yaw = 0.4
    loc = rng.uniform([-2.2, -0.9], [2.2, 0.9], (500, 2))
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    xz = loc @ R.T + [5, -12]
    r = min_area_rect(xz)
    assert abs(r["length"] - 4.4) < 0.15 and abs(r["width"] - 1.8) < 0.15
    ax = np.array([np.cos(r["yaw_rad"]), np.sin(r["yaw_rad"])])
    assert abs(abs(ax @ R[:, 0]) - 1) < 0.01  # length axis recovered (up to 180 degrees)
    P = np.vstack([np.c_[xz[:, 0], np.ones(500), xz[:, 1]], rng.normal([20, 1, 0], 0.3, (300, 3))])
    cl = cluster_grid(P, 0.5, min_points=50)
    assert sorted(len(c) for c in cl) == [300, 500]
