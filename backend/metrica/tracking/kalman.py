"""Constant-velocity Kalman filter on image boxes (cx, cy, aspect, height), as in SORT/ByteTrack."""
from __future__ import annotations

import numpy as np


def xyxy_to_xyah(b: np.ndarray) -> np.ndarray:
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w / max(h, 1e-6), h], float)


def xyah_to_xyxy(x: np.ndarray) -> np.ndarray:
    cx, cy, a, h = x[:4]
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], float)


class BoxKalman:
    std_pos = 1.0 / 20
    std_vel = 1.0 / 160

    def __init__(self, box_xyxy: np.ndarray):
        m = xyxy_to_xyah(box_xyxy)
        self.x = np.r_[m, np.zeros(4)]
        h = m[3]
        std = [2 * self.std_pos * h, 2 * self.std_pos * h, 1e-2, 2 * self.std_pos * h,
               10 * self.std_vel * h, 10 * self.std_vel * h, 1e-5, 10 * self.std_vel * h]
        self.P = np.diag(np.square(std))
        self.F = np.eye(8)
        self.H = np.eye(4, 8)

    def predict(self, dt: float = 1.0) -> np.ndarray:
        F = self.F.copy()
        F[:4, 4:] = np.eye(4) * dt
        h = self.x[3]
        q = [self.std_pos * h, self.std_pos * h, 1e-2, self.std_pos * h,
             self.std_vel * h, self.std_vel * h, 1e-5, self.std_vel * h]
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + np.diag(np.square(q)) * max(dt, 1.0)
        self.x[3] = max(self.x[3], 1.0)
        self.x[2] = max(self.x[2], 0.05)
        return xyah_to_xyxy(self.x)

    def update(self, box_xyxy: np.ndarray) -> None:
        z = xyxy_to_xyah(box_xyxy)
        h = self.x[3]
        r = [self.std_pos * h, self.std_pos * h, 1e-1, self.std_pos * h]
        S = self.H @ self.P @ self.H.T + np.diag(np.square(r))
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(8) - K @ self.H) @ self.P

    @property
    def box(self) -> np.ndarray:
        return xyah_to_xyxy(self.x)
