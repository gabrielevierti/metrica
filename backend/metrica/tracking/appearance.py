"""Explainable appearance cues: colour histograms of the upper and lower part of a subject.

Used to keep identities apart when boxes cross, and to suggest re-links across occlusions.
Pixels come from the instance mask when available, otherwise from the central part of the
box (to limit background), and - for static cameras - only from pixels that differ from
the background plate.
"""
from __future__ import annotations

import cv2
import numpy as np

H_BINS, S_BINS, V_BINS = 12, 4, 4


def _hist(hsv: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    h = cv2.calcHist([hsv], [0, 1, 2], mask, [H_BINS, S_BINS, V_BINS], [0, 180, 0, 256, 0, 256]).ravel()
    s = h.sum()
    return (h / s).astype(np.float32) if s > 0 else h.astype(np.float32)


def signature(bgr: np.ndarray, box: list[float], background: np.ndarray | None = None,
              mask: np.ndarray | None = None) -> np.ndarray:
    """Concatenated [upper, lower] HSV histograms (normalised), length 2*H*S*V."""
    H, W = bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 - x1 < 4 or y2 - y1 < 8:
        return np.zeros(2 * H_BINS * S_BINS * V_BINS, np.float32)
    crop = bgr[y1:y2, x1:x2]
    m = np.ones(crop.shape[:2], np.uint8) * 255
    if mask is not None:
        m = (mask[y1:y2, x1:x2] > 0).astype(np.uint8) * 255
    else:
        bw = x2 - x1
        m[:, : int(bw * 0.2)] = 0
        m[:, int(bw * 0.8):] = 0
    if background is not None and background.shape == bgr.shape:
        diff = np.abs(crop.astype(np.int16) - background[y1:y2, x1:x2].astype(np.int16)).max(2)
        fg = (diff > 25).astype(np.uint8) * 255
        if (fg & m).sum() > 0.15 * m.sum():
            m = fg & m
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hh = crop.shape[0]
    up_m, lo_m = m.copy(), m.copy()
    up_m[int(hh * 0.55):] = 0
    up_m[: int(hh * 0.12)] = 0  # skip head/hair
    lo_m[: int(hh * 0.55)] = 0
    return np.concatenate([_hist(hsv, up_m), _hist(hsv, lo_m)])


def distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Mean Bhattacharyya distance of the two halves in [0, 1] (0 = identical)."""
    if a is None or b is None or a.sum() == 0 or b.sum() == 0:
        return 0.5
    n = len(a) // 2
    out = []
    for sa, sb in ((a[:n], b[:n]), (a[n:], b[n:])):
        if sa.sum() == 0 or sb.sum() == 0:
            continue
        bc = np.sqrt(sa * sb).sum() / np.sqrt(sa.sum() * sb.sum())
        out.append(np.sqrt(max(0.0, 1.0 - bc)))
    return float(np.mean(out)) if out else 0.5
