"""Per-frame orientation of a camera that rotates about a fixed centre (pan / tilt).

For a camera that only rotates, two images are related by the homography
``x_ref ~ K R_ref^T R_i K^-1 x_i`` whatever the scene depth, so each frame's rotation follows
from the homography that maps it onto the reference frame: ``R_i = R_ref · K^-1 H K`` (after
normalisation and projection onto SO(3)).

Frames are registered directly against the reference image when they overlap it; otherwise the
homography is chained through the previous registered frame (flagged ``chained``; errors
accumulate, so chains are limited). Frames whose homography is not a rotation (zoom changes,
camera translation, bad matches) are left without a pose: their observations get no 3D position.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import cv2
import numpy as np

from ..log import get_logger

log = get_logger("geometry")

MAX_SIDE = 960
MAX_CHAIN = 40


class _Features:
    def __init__(self, max_features: int = 3000):
        self.sift = cv2.SIFT_create(nfeatures=max_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    def detect(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, float]:
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
        s = min(1.0, MAX_SIDE / max(g.shape))
        if s < 1.0:
            g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        kp, des = self.sift.detectAndCompute(g, None)
        pts = np.array([k.pt for k in kp], np.float32).reshape(-1, 2) / s
        return pts, des, s

    def homography(self, a: tuple, b: tuple, ratio: float = 0.75) -> tuple[np.ndarray | None, int, float]:
        """Homography mapping image a onto image b, its inlier count and inlier ratio."""
        pa, da, _ = a
        pb, db, _ = b
        if da is None or db is None or len(da) < 20 or len(db) < 20:
            return None, 0, 0.0
        m = self.matcher.knnMatch(da, db, k=2)
        good = [x[0] for x in m if len(x) == 2 and x[0].distance < ratio * x[1].distance]
        if len(good) < 25:
            return None, 0, 0.0
        src = pa[[g.queryIdx for g in good]]
        dst = pb[[g.trainIdx for g in good]]
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0, maxIters=3000, confidence=0.999)
        if H is None:
            return None, 0, 0.0
        n = int(mask.sum())
        return H, n, n / len(good)


def rotation_from_homography(H: np.ndarray, K: np.ndarray) -> tuple[np.ndarray | None, float]:
    """Closest rotation to K^-1 H K and how far the normalised matrix is from orthonormal."""
    M = np.linalg.inv(K) @ H @ K
    d = np.linalg.det(M)
    if d <= 0:
        return None, float("inf")
    M = M / np.cbrt(d)
    U, S, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        return None, float("inf")
    return R, float(np.max(np.abs(S - 1.0)))


def register_homographies(frames: Iterable[tuple[int, np.ndarray]], reference: np.ndarray, min_inliers: int = 40,
                          cancelled: Callable[[], bool] | None = None) -> dict[str, Any]:
    """Homography of each frame onto the reference image (direct, or chained through the previous
    registered frame). Returns {"frames": {f: {"H", "source", "inliers", "hops"}}, "rejected": {f: reason}}."""
    feat = _Features()
    ref = feat.detect(reference)
    out: dict[int, dict[str, Any]] = {}
    rejected: dict[int, str] = {}
    prev = None          # (frame, features, H_to_ref, hops)
    for f, img in frames:
        if cancelled and cancelled():
            break
        cur = feat.detect(img)
        H, n, ratio = feat.homography(cur, ref)
        src, hops = "direct", 0
        if H is None or n < min_inliers or ratio < 0.3:
            H = None
            if prev is not None and prev[3] < MAX_CHAIN:
                Hp, n, ratio = feat.homography(cur, prev[1])
                if Hp is not None and n >= min_inliers and ratio >= 0.3:
                    H, src, hops = prev[2] @ Hp, "chained", prev[3] + 1
        if H is None:
            rejected[f] = "not enough matches to the reference or the previous frame"
            prev = None
            continue
        out[f] = {"H": H, "source": src, "inliers": n, "hops": hops}
        prev = (f, cur, H, hops)
    return {"frames": out, "rejected": rejected}


def rotations_from_homographies(reg: dict[str, Any], K: np.ndarray,
                                max_non_rotation: float = 0.06) -> dict[str, Any]:
    """Relative rotations (frame -> reference) for registered frames whose homography is a rotation."""
    out: dict[int, dict[str, Any]] = {}
    rejected = dict(reg["rejected"])
    for f, r in reg["frames"].items():
        R, dev = rotation_from_homography(r["H"], K)
        if R is None or dev > max_non_rotation:
            rejected[f] = (f"image motion is not a pure rotation (deviation {dev:.2f}): zoom change, camera "
                           "translation or a mismatched frame")
            continue
        out[f] = {"R_rel": R, "source": r["source"], "inliers": r["inliers"], "hops": r["hops"], "deviation": dev}
    stats = {"registered": len(out), "direct": sum(1 for r in out.values() if r["source"] == "direct"),
             "chained": sum(1 for r in out.values() if r["source"] == "chained"), "without_pose": len(rejected)}
    log.info(f"Rotating camera: {stats['registered']} frames oriented ({stats['direct']} direct, "
             f"{stats['chained']} chained), {stats['without_pose']} without pose")
    return {"frames": out, "rejected": rejected, "stats": stats}


def map_point(H: np.ndarray, uv: np.ndarray) -> np.ndarray:
    p = np.c_[np.asarray(uv, float).reshape(-1, 2), np.ones(len(np.asarray(uv).reshape(-1, 2)))] @ H.T
    return p[:, :2] / p[:, 2:3]
