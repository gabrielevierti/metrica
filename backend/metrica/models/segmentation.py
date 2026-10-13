"""Segmentation adapters.

* :class:`EfficientSAMPrompt` - instance mask for a detection box (box prompt). The model
  is run on a context crop around the box, which gives small objects more resolution.
* :class:`GrabCutMask` - classical fallback (no model), clearly labelled as such.
* :class:`DUCCityscapes` - dense semantic labels for street scenes (19 Cityscapes classes),
  mapped to METRICA scene labels.
"""
from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

from ..semantics.labels import CITYSCAPES_NAMES, CITYSCAPES_TO_CATEGORY, SCENE_ID
from .runtime import session


class PromptSegmenter(Protocol):
    key: str

    def mask_for_box(self, bgr: np.ndarray, box: list[float]) -> tuple[np.ndarray, float]:
        """Boolean mask (full image size) and a quality score in [0, 1]."""


def _context_crop(shape, box, pad: float = 0.25, min_size: int = 48):
    h, w = shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
    side = max(bw, bh) * (1 + 2 * pad)
    side = max(side, min_size)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    X1, Y1 = int(max(0, np.floor(cx - side / 2))), int(max(0, np.floor(cy - side / 2)))
    X2, Y2 = int(min(w, np.ceil(cx + side / 2))), int(min(h, np.ceil(cy + side / 2)))
    return X1, Y1, X2, Y2


class EfficientSAMPrompt:
    key = "efficientsam_ti"
    SIZE = 1024

    def __init__(self):
        self.sess = session(self.key)

    def mask_for_box(self, bgr: np.ndarray, box: list[float]) -> tuple[np.ndarray, float]:
        H, W = bgr.shape[:2]
        X1, Y1, X2, Y2 = _context_crop(bgr.shape, box)
        crop = bgr[Y1:Y2, X1:X2]
        ch, cw = crop.shape[:2]
        img = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (self.SIZE, self.SIZE)).astype(np.float32) / 255.0
        pts = np.zeros((1, 1, 6, 2), np.float32)
        lab = -np.ones((1, 1, 6), np.float32)
        sx, sy = self.SIZE / cw, self.SIZE / ch
        pts[0, 0, 0] = [(box[0] - X1) * sx, (box[1] - Y1) * sy]
        pts[0, 0, 1] = [(box[2] - X1) * sx, (box[3] - Y1) * sy]
        lab[0, 0, 0], lab[0, 0, 1] = 2, 3  # SAM convention: box top-left / bottom-right
        masks, iou = self.sess.run(None, {"batched_images": img.transpose(2, 0, 1)[None],
                                          "batched_point_coords": pts, "batched_point_labels": lab})
        k = int(np.argmax(iou[0, 0]))
        m = cv2.resize((masks[0, 0, k] >= 0).astype(np.uint8), (cw, ch), interpolation=cv2.INTER_NEAREST)
        full = np.zeros((H, W), bool)
        full[Y1:Y2, X1:X2] = m > 0
        # a box prompt must not leak outside its box by more than a few pixels
        bx1, by1 = int(max(0, box[0] - 3)), int(max(0, box[1] - 3))
        bx2, by2 = int(min(W, box[2] + 3)), int(min(H, box[3] + 3))
        clip = np.zeros_like(full)
        clip[by1:by2, bx1:bx2] = True
        full &= clip
        return full, float(np.clip(iou[0, 0, k], 0, 1))


class GrabCutMask:
    """Classical colour-model segmentation inside a box (no neural network)."""

    key = "grabcut"

    def mask_for_box(self, bgr: np.ndarray, box: list[float]) -> tuple[np.ndarray, float]:
        H, W = bgr.shape[:2]
        X1, Y1, X2, Y2 = _context_crop(bgr.shape, box, pad=0.15)
        crop = bgr[Y1:Y2, X1:X2]
        rect = (int(box[0] - X1), int(box[1] - Y1), int(max(2, box[2] - box[0])), int(max(2, box[3] - box[1])))
        mask = np.zeros(crop.shape[:2], np.uint8)
        try:
            cv2.grabCut(crop, mask, rect, np.zeros((1, 65)), np.zeros((1, 65)), 3, cv2.GC_INIT_WITH_RECT)
        except cv2.error:
            return np.zeros((H, W), bool), 0.0
        fg = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
        full = np.zeros((H, W), bool)
        full[Y1:Y2, X1:X2] = fg
        area = (box[2] - box[0]) * (box[3] - box[1])
        fill = full.sum() / max(area, 1)
        return full, float(np.clip(1.0 - abs(fill - 0.55) * 1.5, 0.05, 0.6))


class DUCCityscapes:
    """ResNet101-DUC-HDC (Cityscapes). Output is rearranged from the DUC sub-pixel layout."""

    key = "duc_cityscapes"
    SIZE = 800

    def __init__(self):
        self.sess = session(self.key)
        self.input_name = self.sess.get_inputs()[0].name
        self._map = np.array([SCENE_ID[CITYSCAPES_TO_CATEGORY[n]] for n in CITYSCAPES_NAMES], np.uint8)

    def __call__(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (scene label ids uint8 HxW, confidence float32 HxW)."""
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = cv2.resize(rgb, (self.SIZE, self.SIZE), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        x = (x - x.reshape(-1, 3).mean(0)).transpose(2, 0, 1)[None].astype(np.float32)
        o = self.sess.run(None, {self.input_name: x})[0][0]
        n = self.SIZE // 8
        o = o.reshape(19, 4, 4, n, n).transpose(0, 3, 1, 4, 2).reshape(19, n * 4, n * 4)
        o = o - o.max(0, keepdims=True)
        p = np.exp(o)
        p /= p.sum(0, keepdims=True)
        lab = p.argmax(0).astype(np.uint8)
        conf = p.max(0).astype(np.float32)
        lab = cv2.resize(lab, (w, h), interpolation=cv2.INTER_NEAREST)
        conf = cv2.resize(conf, (w, h), interpolation=cv2.INTER_LINEAR)
        return self._map[lab], conf
