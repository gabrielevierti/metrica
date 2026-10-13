"""Vehicle observations: dominant colour (with a reliability test) and OCR candidates.

Colour names are coarse and lighting-dependent; a colour is reported as *reliable* only when
several crops agree and the pixels are neither too dark nor too desaturated. OCR output is a
candidate transcription that always requires human verification.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

COLOR_HEX = {"black": "#1b1b1b", "dark grey": "#4a4a4a", "grey/silver": "#9a9a9a", "white": "#ececec",
             "red": "#c8312b", "orange": "#e07b25", "yellow": "#e3c62f", "green": "#3f8f45", "light blue": "#5aa9df",
             "blue": "#2c4fa3", "purple": "#7b4ca8", "pink": "#e07fb0", "brown": "#6e4a2e", "beige": "#cdb88f"}


def color_name(rgb) -> str:
    px = np.uint8([[list(rgb)[::-1]]])
    h, s, v = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0].astype(float)
    h *= 2.0
    s /= 255.0
    v /= 255.0
    if v < 0.18:
        return "black"
    if s < 0.18:
        if v < 0.40:
            return "dark grey"
        return "grey/silver" if v < 0.78 else "white"
    if 15 <= h < 45 and v < 0.55:
        return "brown"
    if 20 <= h < 50 and s < 0.45 and v > 0.55:
        return "beige"
    if h < 15 or h >= 345:
        return "red" if v > 0.35 else "brown"
    if h < 40:
        return "orange" if v > 0.5 else "brown"
    if h < 70:
        return "yellow"
    if h < 165:
        return "green"
    if h < 200:
        return "light blue"
    if h < 255:
        return "light blue" if v > 0.75 and s < 0.5 else "blue"
    if h < 290:
        return "purple"
    return "pink"


def dominant_color(img: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any] | None:
    h, w = img.shape[:2]
    m = np.ones((h, w), bool) if mask is None else mask.astype(bool)
    if mask is None:  # central region: avoid road and background
        m[:, : int(w * 0.15)] = m[:, int(w * 0.85):] = False
        m[: int(h * 0.15)] = m[int(h * 0.85):] = False
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2] / 255.0
    m &= (v > 0.12) & (v < 0.97)  # exclude windows/shadows and specular highlights
    px = img[m].astype(np.float32)
    if len(px) < 30:
        return None
    lab = cv2.cvtColor(px.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    k = 3 if len(px) > 90 else 1
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, lbl, centers = cv2.kmeans(lab, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(lbl.ravel(), minlength=k)
    j = int(np.argmax(counts))
    bgr = cv2.cvtColor(centers[j].reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_LAB2BGR).reshape(3)
    rgb = bgr[::-1].astype(int).tolist()
    return {"name": color_name(rgb), "rgb": rgb, "fraction": float(counts[j] / counts.sum())}


def aggregate_colors(per_crop: list[dict[str, Any]]) -> dict[str, Any] | None:
    per_crop = [c for c in per_crop if c]
    if not per_crop:
        return None
    names = [c["name"] for c in per_crop]
    best = max(set(names), key=names.count)
    agree = names.count(best) / len(names)
    reliable = len(per_crop) >= 2 and agree >= 0.6 and best not in ("black", "dark grey") or \
        (len(per_crop) >= 3 and agree >= 0.75)
    rgb = np.median([c["rgb"] for c in per_crop if c["name"] == best], 0).astype(int).tolist()
    return {"name": best, "rgb": rgb, "hex": COLOR_HEX.get(best), "agreement": round(agree, 2), "crops": len(per_crop),
            "reliable": bool(reliable),
            "note": ("Consistent across crops." if reliable else
                     "Not reliable: few crops, disagreement between crops, or dark/achromatic pixels where lighting "
                     "dominates the apparent colour.")}


def aggregate_ocr(reads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge identical candidate strings across crops; repeated reads raise confidence."""
    by_text: dict[str, list[dict[str, Any]]] = {}
    for r in reads:
        by_text.setdefault(r["text"], []).append(r)
    out = []
    for text, rs in by_text.items():
        conf = float(1 - np.prod([1 - x["recognition_confidence"] * x["detection_confidence"] for x in rs]))
        out.append({"text": text, "reads": len(rs), "confidence": round(conf, 3),
                    "frames": sorted({x["frame"] for x in rs}), "status": "candidate - requires human verification",
                    "evidence": [{"frame": x["frame"], "crop": x["crop"], "quad_in_crop": x["quad"]} for x in rs]})
    return sorted(out, key=lambda d: -d["confidence"])
