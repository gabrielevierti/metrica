"""Triangulation of a depth map on its pixel grid, without the classic artefacts.

Removed faces:
* across depth discontinuities (unless all corners lie on one fitted plane);
* free-form faces seen almost edge-on (the long "streaks" a single view cannot know);
* faces far longer in 3D than the pixels they cover (sheets pulled across occlusion edges);
* tiny disconnected fragments.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from ..geometry.camera import Camera


def grid_mesh(z: np.ndarray, cam: Camera, stride: int = 2, plane_id: np.ndarray | None = None,
              edge_ratio: float = 1.06, min_view_cos: float = 0.17, stretch: float = 4.0,
              min_component_faces: int = 40):
    """Returns (vertices_world (N,3), faces (M,3), vertex_pixel (N,2) int, vertex_view_cos (N,))."""
    h, w = z.shape
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    gh, gw = vs.shape
    zz = z[vs, us]
    valid = np.isfinite(zz) & (zz > 0)
    uv = np.stack([us + 0.5, vs + 0.5], -1).reshape(-1, 2).astype(float)
    P = cam.backproject(uv, np.where(valid, zz, 1.0).reshape(-1)).astype(np.float64)
    idx = np.arange(gh * gw).reshape(gh, gw)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    za, zb, zc, zd = zz[:-1, :-1], zz[:-1, 1:], zz[1:, :-1], zz[1:, 1:]
    va = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        zmax = np.fmax(np.fmax(za, zb), np.fmax(zc, zd))
        zmin = np.fmin(np.fmin(za, zb), np.fmin(zc, zd))
        smooth = zmax / zmin < edge_ratio
    same = np.zeros_like(va)
    if plane_id is not None:
        pg = plane_id[vs, us]
        pa = pg[:-1, :-1]
        same = (pa >= 0) & (pa == pg[:-1, 1:]) & (pa == pg[1:, :-1]) & (pa == pg[1:, 1:])
    keep = va & (smooth | same)
    tris = np.concatenate([np.stack([a, c, b], -1)[keep], np.stack([b, c, d], -1)[keep]])
    planar = np.concatenate([same[keep], same[keep]])
    if len(tris) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros((0, 2), int), np.zeros(0)
    A, B, C = P[tris[:, 0]], P[tris[:, 1]], P[tris[:, 2]]
    nrm = np.cross(B - A, C - A)
    view = (A + B + C) / 3 - cam.center
    with np.errstate(invalid="ignore", divide="ignore"):
        cosv = np.abs((nrm * view).sum(1)) / (np.linalg.norm(nrm, axis=1) * np.linalg.norm(view, axis=1))
    edge = np.maximum.reduce([np.linalg.norm(B - A, axis=1), np.linalg.norm(C - B, axis=1),
                              np.linalg.norm(A - C, axis=1)])
    dist = np.linalg.norm(view, axis=1)
    footprint = dist * stride * 1.5 / cam.intr.fx
    ok = planar | ((cosv > min_view_cos) & (edge < stretch * footprint))
    tris, cosv = tris[ok], cosv[ok]
    if len(tris) and min_component_faces:
        n = gh * gw
        e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]]])
        g = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
        _, lab = connected_components(g, directed=False)
        fl = lab[tris[:, 0]]
        cnt = np.bincount(fl)
        okc = cnt[fl] >= min_component_faces
        tris, cosv = tris[okc], cosv[okc]
    vcos = np.zeros(gh * gw)
    for k in range(3):
        np.maximum.at(vcos, tris[:, k], np.nan_to_num(cosv))
    pix = np.stack([us, vs], -1).reshape(-1, 2)
    return P, tris, pix, vcos
