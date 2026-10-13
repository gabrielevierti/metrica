"""Face *detection* (YuNet). METRICA never performs face recognition or identity search."""
from __future__ import annotations

import cv2
import numpy as np

from .registry import require


class YuNetFaces:
    key = "yunet"

    def __init__(self, score_threshold: float = 0.6):
        self.path = str(require(self.key))
        self.det = cv2.FaceDetectorYN.create(self.path, "", (320, 320), score_threshold, 0.3, 50)

    def detect(self, bgr: np.ndarray, region: tuple[int, int, int, int] | None = None,
               min_side: int = 160) -> list[dict]:
        """Detect faces, optionally inside region=(x1,y1,x2,y2). Small regions are upsampled
        (bicubic, for detection only - saved crops always use original pixels)."""
        H, W = bgr.shape[:2]
        x1, y1, x2, y2 = region if region is not None else (0, 0, W, H)
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(W, int(x2)), min(H, int(y2))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return []
        crop = bgr[y1:y2, x1:x2]
        s = max(1.0, min_side / max(1, min(crop.shape[:2])))
        s = min(s, 6.0)
        img = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC) if s > 1 else crop
        self.det.setInputSize((img.shape[1], img.shape[0]))
        _, faces = self.det.detect(img)
        out = []
        if faces is None:
            return out
        for f in faces:
            bx, by, bw, bh = f[:4] / s
            lm = (f[4:14].reshape(5, 2) / s) + [x1, y1]
            out.append({"bbox": [float(x1 + bx), float(y1 + by), float(x1 + bx + bw), float(y1 + by + bh)],
                        "landmarks": lm.tolist(), "score": float(f[14]), "upsample": float(s)})
        return out


def face_pose_class(landmarks: list[list[float]]) -> tuple[str, float]:
    """Coarse head orientation from the 5 landmarks (eyes, nose, mouth corners).

    Returns (class, yaw_ratio). yaw_ratio is the horizontal offset of the nose from the eye
    midpoint, normalised by the inter-ocular distance (0 = frontal). Thresholds are coarse:
    'frontal' < 0.15, 'partially turned' < 0.40, otherwise 'profile/strongly turned'.
    """
    lm = np.asarray(landmarks, float)
    re, le, nose = lm[0], lm[1], lm[2]
    iod = np.linalg.norm(le - re)
    if iod < 1e-6:
        return "undetermined", float("nan")
    mid = (re + le) / 2
    axis = (le - re) / iod
    ratio = float(abs((nose - mid) @ axis) / iod)
    if ratio < 0.15:
        return "frontal", ratio
    if ratio < 0.40:
        return "partially turned", ratio
    return "strongly turned", ratio
