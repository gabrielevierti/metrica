"""Evidence classes and the binary layer format shared by the pipeline and the viewer.

A *layer* is a mesh or point set stored as flat little-endian buffers next to a JSON
description, so the browser can load it with zero parsing:

    <name>.pos.f32   N x 3 positions
    <name>.col.u8    N x 3 colours (sRGB)
    <name>.idx.u32   M x 3 triangle indices (meshes only)
    <name>.evi.u8    N evidence class ids (see EVIDENCE)
    <name>.conf.u8   N confidence 0..255
    <name>.lab.u8    N scene label ids (semantics.labels.SCENE_LABELS)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# Evidence classes: the core vocabulary for "how do we know this geometry?"
EVIDENCE = ["OBSERVED_MULTI_VIEW", "OBSERVED_SINGLE_VIEW", "FITTED_PRIMITIVE", "CLASS_PRIOR", "INFERRED", "UNKNOWN"]
EVI = {k: i for i, k in enumerate(EVIDENCE)}
EVIDENCE_TEXT = {
    "OBSERVED_MULTI_VIEW": "Surface seen consistently from two or more viewpoints.",
    "OBSERVED_SINGLE_VIEW": "Surface seen from one viewpoint; depth from a single image (monocular estimate "
                            "aligned to geometric evidence).",
    "FITTED_PRIMITIVE": "Geometric model (plane, cylinder, box) fitted to observed points; the observed extent is "
                        "recorded separately from any extrapolation.",
    "CLASS_PRIOR": "Shape or dimensions taken from typical values for the object class, positioned by observations. "
                   "Not a scan of the actual object.",
    "INFERRED": "Plausible completion of an unobserved region (e.g. ground continuing behind an object). Not observed.",
    "UNKNOWN": "No evidence: not reconstructed.",
}


def write_layer(out_dir: Path, name: str, pos: np.ndarray, col: np.ndarray | None = None,
                idx: np.ndarray | None = None, evi: np.ndarray | None = None, conf: np.ndarray | None = None,
                lab: np.ndarray | None = None) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(pos)
    files = {}
    np.ascontiguousarray(pos, np.float32).tofile(out_dir / f"{name}.pos.f32")
    files["positions"] = f"{name}.pos.f32"
    arrs = {"colors": (col, np.uint8, 3, ".col.u8"), "evidence": (evi, np.uint8, 1, ".evi.u8"),
            "confidence": (conf, np.uint8, 1, ".conf.u8"), "labels": (lab, np.uint8, 1, ".lab.u8")}
    for key, (a, dt, k, ext) in arrs.items():
        if a is None:
            continue
        a = np.ascontiguousarray(a, dt).reshape(n, k) if k > 1 else np.ascontiguousarray(a, dt).reshape(n)
        a.tofile(out_dir / f"{name}{ext}")
        files[key] = f"{name}{ext}"
    if idx is not None:
        np.ascontiguousarray(idx, np.uint32).tofile(out_dir / f"{name}.idx.u32")
        files["indices"] = f"{name}.idx.u32"
    stats: dict[str, Any] = {"vertices": int(n), "faces": int(len(idx)) if idx is not None else 0}
    if evi is not None and n:
        counts = np.bincount(np.asarray(evi, np.uint8), minlength=len(EVIDENCE))
        stats["evidence_vertices"] = {EVIDENCE[i]: int(c) for i, c in enumerate(counts) if c}
    if n:
        stats["bbox_min"] = np.asarray(pos).min(0).round(4).tolist()
        stats["bbox_max"] = np.asarray(pos).max(0).round(4).tolist()
    return {"name": name, "files": files, "stats": stats}


def read_layer(out_dir: Path, meta: dict[str, Any]) -> dict[str, np.ndarray]:
    f = meta["files"]
    pos = np.fromfile(out_dir / f["positions"], np.float32).reshape(-1, 3)
    n = len(pos)
    out = {"pos": pos}
    if "colors" in f:
        out["col"] = np.fromfile(out_dir / f["colors"], np.uint8).reshape(n, 3)
    if "indices" in f:
        out["idx"] = np.fromfile(out_dir / f["indices"], np.uint32).reshape(-1, 3)
    for key, ext in (("evidence", "evi"), ("confidence", "conf"), ("labels", "lab")):
        if key in f:
            out[ext] = np.fromfile(out_dir / f[key], np.uint8)
    return out


def compact(pos: np.ndarray, idx: np.ndarray, *attrs: np.ndarray):
    """Drop unreferenced vertices."""
    used = np.zeros(len(pos), bool)
    used[idx.ravel()] = True
    remap = -np.ones(len(pos), np.int64)
    remap[used] = np.arange(used.sum())
    return (pos[used], remap[idx].astype(np.uint32), *[a[used] if a is not None else None for a in attrs])
