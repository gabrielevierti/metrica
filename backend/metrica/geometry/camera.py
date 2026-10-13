"""Pinhole camera model and poses.

Conventions
-----------
* Camera frame: OpenCV (x right, y down, z forward).
* World frame: right-handed, **Y up**, metric when calibrated. When a ground plane is
  established it is ``y = 0``. (three.js uses the same world convention.)
* ``R_wc`` maps camera-frame vectors to world-frame vectors; ``C`` is the camera centre
  in world coordinates. ``X_w = R_wc @ X_c + C``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    k1: float = 0.0  # radial distortion (SIMPLE_RADIAL), normalised coordinates
    source: str = "assumed"  # assumed | estimated_single_view | sfm | user

    @classmethod
    def from_hfov(cls, width: int, height: int, hfov_deg: float, source: str = "assumed") -> Intrinsics:
        fx = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
        return cls(float(fx), float(fx), width / 2.0, height / 2.0, int(width), int(height), 0.0, source)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    @property
    def hfov_deg(self) -> float:
        return float(np.degrees(2 * np.arctan(self.width / 2 / self.fx)))

    def normalized(self, uv: np.ndarray) -> np.ndarray:
        """Undistorted normalised image coordinates (x, y) for pixel coordinates."""
        uv = np.asarray(uv, np.float64)
        x = (uv[..., 0] - self.cx) / self.fx
        y = (uv[..., 1] - self.cy) / self.fy
        if self.k1:
            xd, yd = x.copy(), y.copy()
            for _ in range(8):  # fixed-point inversion of r' = r (1 + k1 r^2)
                r2 = x * x + y * y
                f = 1 + self.k1 * r2
                x, y = xd / f, yd / f
        return np.stack([x, y], -1)

    def project_normalized(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, np.float64)
        x, y = xy[..., 0], xy[..., 1]
        if self.k1:
            f = 1 + self.k1 * (x * x + y * y)
            x, y = x * f, y * f
        return np.stack([x * self.fx + self.cx, y * self.fy + self.cy], -1)

    def scaled(self, s: float) -> Intrinsics:
        return Intrinsics(self.fx * s, self.fy * s, self.cx * s, self.cy * s, int(round(self.width * s)),
                          int(round(self.height * s)), self.k1, self.source)

    def to_json(self) -> dict[str, Any]:
        return {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy, "width": self.width,
                "height": self.height, "k1": self.k1, "hfov_deg": self.hfov_deg, "source": self.source}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Intrinsics:
        return cls(d["fx"], d["fy"], d["cx"], d["cy"], d["width"], d["height"], d.get("k1", 0.0),
                   d.get("source", "assumed"))


@dataclass
class Pose:
    R_wc: np.ndarray
    C: np.ndarray

    @classmethod
    def identity(cls) -> Pose:
        return cls(np.eye(3), np.zeros(3))

    @classmethod
    def from_cam_from_world(cls, R_cw: np.ndarray, t_cw: np.ndarray) -> Pose:
        R_wc = np.asarray(R_cw).T
        return cls(R_wc, -R_wc @ np.asarray(t_cw, float))

    @property
    def R_cw(self) -> np.ndarray:
        return self.R_wc.T

    @property
    def t_cw(self) -> np.ndarray:
        return -self.R_wc.T @ self.C

    def to_cam(self, Xw: np.ndarray) -> np.ndarray:
        return (np.asarray(Xw, float) - self.C) @ self.R_wc

    def to_world(self, Xc: np.ndarray) -> np.ndarray:
        return np.asarray(Xc, float) @ self.R_wc.T + self.C

    def matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.R_wc
        T[:3, 3] = self.C
        return T

    def to_json(self) -> dict[str, Any]:
        return {"world_from_cam": self.matrix().tolist()}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Pose:
        T = np.asarray(d["world_from_cam"], float)
        return cls(T[:3, :3], T[:3, 3])

    def transformed(self, S: np.ndarray) -> Pose:
        """Apply a similarity S (4x4, world' = S @ world) to this pose."""
        s = np.cbrt(np.linalg.det(S[:3, :3]))
        R = S[:3, :3] / s
        return Pose(R @ self.R_wc, S[:3, :3] @ self.C + S[:3, 3])


@dataclass
class Camera:
    intr: Intrinsics
    pose: Pose
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def center(self) -> np.ndarray:
        return self.pose.C

    def project(self, Xw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pixel coordinates and camera-frame depth (z) of world points."""
        Xc = self.pose.to_cam(Xw)
        z = Xc[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            xy = Xc[..., :2] / z[..., None]
        return self.intr.project_normalized(xy), z

    def rays_cam(self, uv: np.ndarray) -> np.ndarray:
        """Camera-frame rays with z = 1."""
        xy = self.intr.normalized(uv)
        return np.concatenate([xy, np.ones(xy.shape[:-1] + (1,))], -1)

    def rays_world(self, uv: np.ndarray, normalize: bool = True) -> np.ndarray:
        d = self.rays_cam(uv) @ self.pose.R_wc.T
        if normalize:
            d = d / np.linalg.norm(d, axis=-1, keepdims=True)
        return d

    def backproject(self, uv: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """World points for pixels with camera-frame depth z."""
        Xc = self.rays_cam(uv) * np.asarray(depth)[..., None]
        return self.pose.to_world(Xc)

    def ground_intersect(self, uv: np.ndarray, ground_y: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Intersect pixel rays with the horizontal plane y = ground_y. Returns (points, valid)."""
        d = self.rays_world(uv, normalize=False)
        C = self.pose.C
        dy = d[..., 1]
        valid = dy < -1e-9
        s = np.where(valid, (ground_y - C[1]) / np.where(valid, dy, -1.0), np.nan)
        valid &= s > 0
        P = C + d * s[..., None]
        return P, valid

    def to_json(self) -> dict[str, Any]:
        return {"intrinsics": self.intr.to_json(), **self.pose.to_json(), **self.meta}


def look_rotation(pitch: float, roll: float, yaw: float = 0.0) -> np.ndarray:
    """R_wc for a camera with given pitch (down positive), roll and yaw (radians).

    yaw = 0 looks towards world -Z. Built from the world 'up' vector expressed in camera
    coordinates, so the same parameterisation is used everywhere in single-view geometry.
    """
    up_c = np.array([0.0, -np.cos(pitch), -np.sin(pitch)])
    c, s = np.cos(roll), np.sin(roll)
    up_c = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ up_c
    fwd = np.array([0.0, 0.0, 1.0])
    fwd_h = fwd - (fwd @ up_c) * up_c
    fwd_h /= np.linalg.norm(fwd_h)
    Y = up_c
    Z = -fwd_h
    X = np.cross(Y, Z)
    R_wc = np.stack([X, Y, Z], 0)  # rows: world axes in camera coords -> maps cam to world
    if yaw:
        cy, sy = np.cos(yaw), np.sin(yaw)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        R_wc = Ry @ R_wc
    return R_wc


def up_in_camera(R_wc: np.ndarray) -> np.ndarray:
    return R_wc.T @ np.array([0.0, 1.0, 0.0])


def slerp_rotation(R0: np.ndarray, R1: np.ndarray, a: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation, Slerp

    s = Slerp([0, 1], Rotation.from_matrix(np.stack([R0, R1])))
    return s([a]).as_matrix()[0]
