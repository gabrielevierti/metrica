"""Uniform access to the calibrated views of a reconstruction unit.

A *view* is an image with a camera in the unit frame, its metric depth map and (optionally)
dense scene labels: the background plate of a static camera, or the aligned keyframes of
an SfM unit. Object reconstructors operate on views so the same code serves both cases.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..evidence.case import Case
from ..geometry.camera import Camera, Intrinsics, Pose


@dataclass
class View:
    key: str
    video: str
    frame: int | None
    cam: Camera
    image: np.ndarray            # BGR, same size as z
    z: np.ndarray                # metric camera depth (NaN invalid)
    labels: np.ndarray | None    # scene label ids or None
    plane_id: np.ndarray | None = None
    support: np.ndarray | None = None

    def points(self, mask: np.ndarray, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """3D points (unit frame) of masked pixels with valid depth, and their pixel coords."""
        m = mask.copy()
        if stride > 1:
            sub = np.zeros_like(m)
            sub[::stride, ::stride] = True
            m &= sub
        m &= np.isfinite(self.z)
        vs, us = np.nonzero(m)
        if len(vs) == 0:
            return np.zeros((0, 3)), np.zeros((0, 2), int)
        uv = np.stack([us + 0.5, vs + 0.5], 1).astype(float)
        return self.cam.backproject(uv, self.z[vs, us]), np.stack([us, vs], 1)


def load_views(case: Case, unit: dict[str, Any], recon: dict[str, Any], semantic_only: bool = False) -> list[View]:
    root = case.root
    views: list[View] = []
    rdir = root / "derived" / "reconstruction" / unit["id"]
    if unit["kind"] == "single_view":
        vid = unit["videos"][0]
        geo = case.read_json(f"derived/geometry/{vid}.json")
        d = np.load(rdir / recon["depth_maps"][vid])
        img = cv2.imread(str(root / geo["reference_image"]))
        intr = Intrinsics.from_json(geo["intrinsics"])
        if intr.width != img.shape[1]:
            intr = intr.scaled(img.shape[1] / intr.width)
        cam = Camera(intr, Pose.from_json(geo["static_pose"]))
        labels = d["labels"]
        has_sem = bool(case.read_json(f"derived/videos/{vid}/depth_index.json", {}).get("semantic_model"))
        views.append(View(f"{vid}_bg", vid, None, cam, img, d["z"].astype(np.float32),
                          labels if has_sem else None, d["plane_id"].astype(np.int32)))
        return views
    for rel in recon.get("keyframe_depths", []):
        p = rdir / rel
        d = np.load(p)
        key = Path(rel).stem
        vid, frame = key.rsplit("_", 1)
        K = d["K"]
        intr = Intrinsics(float(K[0]), float(K[1]), float(K[2]), float(K[3]), int(K[4]), int(K[5]), float(K[6]), "sfm")
        cam = Camera(intr, Pose(d["world_from_cam"][:3, :3], d["world_from_cam"][:3, 3]))
        labels = d["labels"] if d["labels"].ndim == 2 else None
        if semantic_only and labels is None:
            continue
        img = cv2.imread(str(root / "derived" / "videos" / vid / "keyframes" / f"{int(frame):07d}.jpg"))
        if img.shape[1] != intr.width:
            img = cv2.resize(img, (intr.width, intr.height), interpolation=cv2.INTER_AREA)
        views.append(View(key, vid, int(frame), cam, img, d["z"].astype(np.float32), labels, None, d["support"]))
    return views
