"""Appearance embeddings for re-linking fragmented tracks of the *same video*.

This is a similarity cue for tracking, not identity recognition: embeddings are never
compared against any external gallery or database.
"""
from __future__ import annotations

import cv2
import numpy as np

from .runtime import session

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class YoutuReID:
    key = "youtu_reid"

    def __init__(self):
        self.sess = session(self.key)
        self.input_name = self.sess.get_inputs()[0].name

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, 768), np.float32)
        x = np.stack([((cv2.resize(c, (128, 256))[:, :, ::-1].astype(np.float32) / 255.0 - MEAN) / STD)
                      .transpose(2, 0, 1) for c in crops]).astype(np.float32)
        f = self.sess.run(None, {self.input_name: x})[0].reshape(len(crops), -1)
        return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)
