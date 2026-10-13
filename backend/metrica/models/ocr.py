"""Scene-text OCR (PP-OCRv3 detection + CRNN recognition).

Output is always a *candidate* transcription with a confidence; reports present it as
requiring human verification. Characters: digits and latin letters (case-insensitive).
"""
from __future__ import annotations

import cv2
import numpy as np

from .registry import require
from .runtime import session

CHARSET = "0123456789abcdefghijklmnopqrstuvwxyz"


class SceneTextOCR:
    key = "ppocr_det_en+crnn_en"

    def __init__(self):
        self.det = cv2.dnn_TextDetectionModel_DB(str(require("ppocr_det_en")))
        self.det.setBinaryThreshold(0.3)
        self.det.setPolygonThreshold(0.5)
        self.det.setMaxCandidates(50)
        self.det.setUnclipRatio(2.0)
        self.det.setInputParams(1.0 / 255.0, (736, 736), (122.67891434, 116.66876762, 104.00698793), False)
        self.rec = session("crnn_en")
        self.rec_input = self.rec.get_inputs()[0].name
        self._target = np.array([[0, 31], [0, 0], [99, 0], [99, 31]], np.float32)

    def _recognize(self, bgr: np.ndarray, quad: np.ndarray) -> tuple[str, float]:
        M = cv2.getPerspectiveTransform(quad.astype(np.float32), self._target)
        crop = cv2.warpPerspective(bgr, M, (100, 32))
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        x = ((g - 127.5) / 127.5)[None, None]
        logits = self.rec.run(None, {self.rec_input: x})[0][:, 0, :]
        logits = logits - logits.max(1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(1, keepdims=True)
        idx = p.argmax(1)
        chars, probs, prev = [], [], 0
        for t, k in enumerate(idx):
            if k != 0 and k != prev:
                chars.append(CHARSET[k - 1])
                probs.append(float(p[t, k]))
            prev = k
        if not chars:
            return "", 0.0
        return "".join(chars).upper(), float(np.exp(np.mean(np.log(np.maximum(probs, 1e-6)))))

    def read(self, bgr: np.ndarray, min_upsample_height: int = 220) -> list[dict]:
        """OCR on a crop. Small crops are upsampled for *detection*; recognition reads the same
        upsampled crop (no learned super-resolution is ever applied)."""
        h = bgr.shape[0]
        s = min(4.0, max(1.0, min_upsample_height / max(h, 1)))
        img = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC) if s > 1 else bgr
        quads, det_conf = self.det.detect(img)
        out = []
        for q, c in zip(quads, det_conf):
            q = np.asarray(q, np.float32)
            text, rc = self._recognize(img, q)
            if len(text) < 2:
                continue
            out.append({"text": text, "recognition_confidence": rc, "detection_confidence": float(c),
                        "quad": (q / s).tolist()})
        return out
