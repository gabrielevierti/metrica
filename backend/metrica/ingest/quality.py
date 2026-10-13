"""Per-frame technical quality metrics (all classical, explainable image statistics)."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

_NOISE_K = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)


def frame_metrics(gray: np.ndarray) -> dict[str, float]:
    """Metrics on a downscaled grayscale frame (~480 px max side)."""
    g = gray.astype(np.float32)
    h, w = g.shape
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    sharp = float(lap.var())
    # Immerkaer (1996) fast noise sigma estimate
    conv = cv2.filter2D(g, -1, _NOISE_K)
    noise = float(np.sqrt(np.pi / 2) * np.abs(conv[1:-1, 1:-1]).sum() / (6.0 * max(1, (w - 2) * (h - 2))))
    return {
        "sharpness": sharp,
        "brightness": float(g.mean()),
        "contrast": float(g.std()),
        "dark_clip": float((g < 8).mean()),
        "bright_clip": float((g > 247).mean()),
        "noise_sigma": noise,
    }


def blockiness(gray_full: np.ndarray) -> float:
    """Compression blocking: mean gradient on 8-px block borders / mean gradient elsewhere
    (1.0 = no visible blocking; >1.3 suggests strong compression artefacts)."""
    g = gray_full.astype(np.float32)
    h, w = g.shape
    cy, cx = h // 2, w // 2
    half = min(256, h // 2, w // 2)
    g = g[cy - half:cy + half, cx - half:cx + half]
    if g.shape[1] < 32:
        return 1.0
    dx = np.abs(np.diff(g, axis=1))
    cols = np.arange(dx.shape[1])
    border = dx[:, (cols % 8) == 7].mean()
    inner = dx[:, (cols % 8) != 7].mean()
    return float(border / max(inner, 1e-6))


def thumb(gray: np.ndarray) -> np.ndarray:
    return cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)


def is_duplicate(prev_thumb: np.ndarray | None, cur_thumb: np.ndarray, thr: float = 0.6) -> bool:
    if prev_thumb is None:
        return False
    return float(np.abs(prev_thumb - cur_thumb).mean()) < thr


def summarize(samples: list[dict[str, Any]], min_sharpness: float) -> dict[str, Any]:
    if not samples:
        return {"frames_assessed": 0}
    arr = {k: np.array([s[k] for s in samples if k in s], float) for k in samples[0] if k not in ("i", "t")}
    usable = np.array([s.get("usable", True) for s in samples])
    dup = np.array([s.get("duplicate", False) for s in samples])
    out: dict[str, Any] = {"frames_assessed": len(samples), "usable_fraction": float(usable.mean()),
                           "duplicate_fraction": float(dup.mean())}
    for k, v in arr.items():
        if v.dtype == bool or len(v) == 0:
            continue
        out[k] = {"median": float(np.median(v)), "p10": float(np.percentile(v, 10)),
                  "p90": float(np.percentile(v, 90))}
    flags = []
    if out.get("sharpness", {}).get("median", 1e9) < min_sharpness * 2:
        flags.append("low sharpness / possible motion blur or heavy compression")
    if out.get("dark_clip", {}).get("median", 0) > 0.3:
        flags.append("underexposed (many near-black pixels)")
    if out.get("bright_clip", {}).get("median", 0) > 0.2:
        flags.append("overexposed (many saturated pixels)")
    if out.get("blockiness", {}).get("median", 1) > 1.3:
        flags.append("visible compression blocking")
    if out["duplicate_fraction"] > 0.2:
        flags.append("many repeated frames (frame duplication or a frozen stream)")
    if out.get("noise_sigma", {}).get("median", 0) > 8:
        flags.append("high sensor noise")
    out["flags"] = flags
    return out
