"""Monocular depth estimation adapters.

Depth Anything V2 predicts *relative inverse depth*: it is correct only up to an unknown
scale and shift (affine-invariant). METRICA never uses it as metric depth directly: it is
aligned per image to geometric evidence (SfM points or a calibrated ground plane) by the
reconstruction stage, and the alignment residuals are recorded as uncertainty.
"""
from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

from .runtime import session

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DepthEstimator(Protocol):
    key: str

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        """Relative inverse depth (larger = nearer), float32, same size as the input."""


class DepthAnythingV2:
    key = "depth_anything_v2_s"

    def __init__(self, short_side: int = 518):
        self.sess = session(self.key)
        self.short_side = int(short_side)
        self.input_name = self.sess.get_inputs()[0].name

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        s = self.short_side / min(h, w)
        nh = max(14, int(round(h * s / 14)) * 14)
        nw = max(14, int(round(w * s / 14)) * 14)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
        x = ((x - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None].astype(np.float32)
        d = self.sess.run(None, {self.input_name: x})[0][0]
        d = cv2.resize(d.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        return np.maximum(d, 1e-6)
