"""Object detection adapters (YOLOX default, Ultralytics YOLO11 optional).

Each detector returns a list of dicts with a common format::

    {"bbox": [x1, y1, x2, y2], "conf": float, "label": coco_name, "category": metrica_category}
"""
from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

from ..semantics.labels import COCO_NAMES, category_for_coco
from .runtime import session


class Detector(Protocol):
    key: str

    def __call__(self, bgr: np.ndarray) -> list[dict]: ...


def _letterbox(bgr: np.ndarray, size: int, pad_value: int = 114, center: bool = False):
    h, w = bgr.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
    top, left = ((size - nh) // 2, (size - nw) // 2) if center else (0, 0)
    canvas[top:top + nh, left:left + nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas, r, top, left


def _nms_per_class(boxes: np.ndarray, conf: np.ndarray, cls: np.ndarray, score_thr: float, iou: float):
    keep_all = []
    for c in np.unique(cls):
        m = np.where(cls == c)[0]
        xywh = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes[m]]
        idx = cv2.dnn.NMSBoxes(xywh, conf[m].astype(float).tolist(), score_thr, iou)
        keep_all.extend(m[np.array(idx, dtype=int).reshape(-1)].tolist())
    return np.array(sorted(keep_all), dtype=int)


class YOLOX:
    """YOLOX (Apache-2.0). The official ONNX export expects raw BGR in [0, 255], letterboxed
    top-left with value 114, and outputs grid-relative boxes that must be decoded."""

    def __init__(self, key: str = "yolox_s", conf: float = 0.30, iou: float = 0.55, size: int = 640):
        self.key = key
        self.sess = session(key)
        self.conf, self.iou, self.size = conf, iou, size
        self.input_name = self.sess.get_inputs()[0].name
        grids, strides = [], []
        for st in (8, 16, 32):
            n = size // st
            xv, yv = np.meshgrid(np.arange(n), np.arange(n))
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((n * n, 1), st))
        self._grid = np.concatenate(grids).astype(np.float32)
        self._stride = np.concatenate(strides).astype(np.float32)

    def __call__(self, bgr: np.ndarray) -> list[dict]:
        h, w = bgr.shape[:2]
        canvas, r, _, _ = _letterbox(bgr, self.size)
        x = canvas.transpose(2, 0, 1)[None].astype(np.float32)
        out = self.sess.run(None, {self.input_name: x})[0][0]
        xy = (out[:, :2] + self._grid) * self._stride
        wh = np.exp(np.clip(out[:, 2:4], -10, 10)) * self._stride
        scores = out[:, 4:5] * out[:, 5:]
        cls = scores.argmax(1)
        conf = scores[np.arange(len(cls)), cls]
        m = conf >= self.conf
        if not m.any():
            return []
        xy, wh, cls, conf = xy[m], wh[m], cls[m], conf[m]
        boxes = np.concatenate([xy - wh / 2, xy + wh / 2], 1) / r
        boxes = boxes.clip([0, 0, 0, 0], [w, h, w, h])
        keep = _nms_per_class(boxes, conf, cls, self.conf, self.iou)
        dets = []
        for k in keep:
            b = boxes[k]
            if b[2] - b[0] < 2 or b[3] - b[1] < 2:
                continue
            name = COCO_NAMES[int(cls[k])]
            dets.append({"bbox": [float(v) for v in b], "conf": float(conf[k]), "label": name,
                         "category": category_for_coco(name)})
        return dets


class YOLO11:
    """Ultralytics YOLO11 ONNX (AGPL-3.0, opt-in only)."""

    def __init__(self, key: str = "yolo11n", conf: float = 0.30, iou: float = 0.55, size: int = 640):
        self.key = key
        self.sess = session(key)
        self.conf, self.iou, self.size = conf, iou, size
        self.input_name = self.sess.get_inputs()[0].name

    def __call__(self, bgr: np.ndarray) -> list[dict]:
        h, w = bgr.shape[:2]
        canvas, r, top, left = _letterbox(bgr, self.size, center=True)
        x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        out = self.sess.run(None, {self.input_name: x})[0][0].T
        scores = out[:, 4:84]
        cls = scores.argmax(1)
        conf = scores[np.arange(len(cls)), cls]
        m = conf >= self.conf
        if not m.any():
            return []
        cx, cy, bw, bh = out[m, :4].T
        cls, conf = cls[m], conf[m]
        boxes = np.stack([(cx - bw / 2 - left) / r, (cy - bh / 2 - top) / r,
                          (cx + bw / 2 - left) / r, (cy + bh / 2 - top) / r], 1).clip([0, 0, 0, 0], [w, h, w, h])
        keep = _nms_per_class(boxes, conf, cls, self.conf, self.iou)
        return [{"bbox": [float(v) for v in boxes[k]], "conf": float(conf[k]), "label": COCO_NAMES[int(cls[k])],
                 "category": category_for_coco(COCO_NAMES[int(cls[k])])} for k in keep]


def make_detector(key: str, conf: float, iou: float) -> Detector:
    if key.startswith("yolox"):
        return YOLOX(key, conf=conf, iou=iou)
    if key.startswith("yolo11"):
        return YOLO11(key, conf=conf, iou=iou)
    raise ValueError(f"Unknown detector '{key}'")
