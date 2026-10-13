"""Ground coverage: which parts of the ground plane any camera actually saw.

Cells are classified per camera as *seen* (the ground is the visible surface there),
*occluded* (inside the view but something nearer hides it) or *outside* the field of view.
The union over cameras gives the coverage map shown in the viewer: everything that is not
'seen' is explicitly unknown ground.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..geometry.camera import Camera

SEEN, OCCLUDED, OUTSIDE = 2, 1, 0


def coverage_map(views: list[tuple[Camera, np.ndarray | None]], extent: tuple[float, float, float, float],
                 cell: float, max_range: float) -> dict[str, Any]:
    """views: (camera, metric depth map or None). extent = (xmin, xmax, zmin, zmax) in world."""
    xmin, xmax, zmin, zmax = extent
    nx = max(1, int(np.ceil((xmax - xmin) / cell)))
    nz = max(1, int(np.ceil((zmax - zmin) / cell)))
    if nx * nz > 4_000_000:  # keep the texture reasonable
        s = np.sqrt(nx * nz / 4_000_000)
        cell *= s
        nx, nz = int(np.ceil((xmax - xmin) / cell)), int(np.ceil((zmax - zmin) / cell))
    xs = xmin + (np.arange(nx) + 0.5) * cell
    zs = zmin + (np.arange(nz) + 0.5) * cell
    X, Z = np.meshgrid(xs, zs)
    P = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1)
    state = np.zeros(len(P), np.uint8)
    nviews = np.zeros(len(P), np.uint16)
    for cam, depth in views:
        uv, zc = cam.project(P)
        w, h = cam.intr.width, cam.intr.height
        rng = np.linalg.norm(P - cam.center, axis=1)
        inside = (zc > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h) & (rng < max_range)
        seen = inside.copy()
        if depth is not None:
            dh, dw = depth.shape
            sx, sy = dw / w, dh / h
            ui = np.clip((uv[inside, 0] * sx).astype(int), 0, dw - 1)
            vi = np.clip((uv[inside, 1] * sy).astype(int), 0, dh - 1)
            dz = depth[vi, ui]
            vis = ~np.isfinite(dz) | (dz >= zc[inside] * 0.93)
            tmp = np.zeros(inside.sum(), bool)
            tmp[:] = vis
            seen[inside] = tmp
        state = np.maximum(state, np.where(seen, SEEN, np.where(inside, OCCLUDED, OUTSIDE)).astype(np.uint8))
        nviews += seen
    grid = state.reshape(nz, nx)
    return {"grid": grid, "views": nviews.reshape(nz, nx), "extent": [xmin, xmin + nx * cell, zmin, zmin + nz * cell],
            "cell": cell, "seen_fraction": float((grid == SEEN).mean()),
            "occluded_fraction": float((grid == OCCLUDED).mean())}
