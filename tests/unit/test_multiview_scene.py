"""Synchronisation by cross-correlation, ground-plane alignment, scene transforms."""
from __future__ import annotations

import numpy as np
import pytest

from metrica.multiview.alignment import ransac_similarity_4dof, similarity_2d, to_4x4
from metrica.multiview.sync import xcorr_offset
from metrica.scene.transform import Similarity, transform_entity


def test_xcorr_recovers_known_offset():
    rng = np.random.default_rng(0)
    x = rng.normal(size=3000)
    rate = 100.0
    shift = 237
    a, b = x[:2500], x[shift:shift + 2500]
    lag, peak, ratio = xcorr_offset(a, b, rate)
    assert lag == pytest.approx(shift / rate, abs=1 / rate) and peak > 0.5 and ratio > 2


def test_similarity_2d_and_ransac_with_outliers():
    rng = np.random.default_rng(1)
    src = rng.uniform(-10, 10, (60, 2))
    th, s, t = 0.7, 1.3, np.array([4.0, -2.0])
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    dst = s * src @ R.T + t
    s_, R_, t_, rms = similarity_2d(src, dst)
    assert s_ == pytest.approx(s) and np.allclose(R_, R) and np.allclose(t_, t) and rms < 1e-9
    src3 = np.c_[src[:, 0], rng.uniform(0, 3, 60), src[:, 1]]
    dst3 = np.c_[dst[:, 0], s * src3[:, 1], dst[:, 1]]
    dst3[:15] += rng.uniform(-20, 20, (15, 3))  # 25% outliers
    r = ransac_similarity_4dof(src3, dst3, thr=0.2)
    assert r is not None and r["inliers"] >= 45 and r["scale"] == pytest.approx(s, rel=1e-3)
    M = to_4x4(s, R, t)
    p = np.array([2.0, 1.0, 3.0, 1.0])
    q = M @ p
    assert np.allclose(q[[0, 2]], s * R @ p[[0, 2]] + t) and q[1] == pytest.approx(s * 1.0)


def test_transform_entity_scales_dimensions_and_rotates_yaw():
    th = np.radians(30)
    M = np.eye(4)
    M[0, 0], M[0, 2], M[2, 0], M[2, 2] = np.cos(th), np.sin(th), -np.sin(th), np.cos(th)
    M[:3, 3] = [1, 0, 2]
    S = Similarity(M, k=2.0)
    assert S.s == pytest.approx(2.0) and S.yaw == pytest.approx(th)
    pole = {"id": "p", "category": "pole",
            "geometry": {"type": "cylinder", "base": [1, 0, 0], "axis": [0, 1, 0], "height": 5.0, "radius_bottom": 0.1,
                         "radius_top": 0.08, "observed_range": [0.5, 5.0]},
            "dimensions": {"height": {"value": 5.0, "sigma": 0.2, "unit": "m"}, "taper_ratio": {"value": 0.8}},
            "inferred_parts": [{"part": "base", "range": [0, 0.5]}], "evidence": {"class": "FITTED_PRIMITIVE"}}
    out = transform_entity(pole, S, "m")
    assert out["geometry"]["height"] == pytest.approx(10.0) and out["dimensions"]["height"]["sigma"] == pytest.approx(0.4)
    assert out["dimensions"]["taper_ratio"]["value"] == 0.8
    assert out["inferred_parts"][0]["range"] == [0, 1.0]
    assert pole["geometry"]["height"] == 5.0  # input untouched
    car = {"id": "c", "category": "car", "geometry": {"type": "vehicle_box", "center": [0, 0, 0], "yaw_rad": 0.5,
                                                      "length": 4.4, "width": 1.8, "height": 1.5}}
    oc = transform_entity(car, S, "m")
    assert oc["geometry"]["yaw_rad"] == pytest.approx(0.5 - th) and oc["geometry"]["length"] == pytest.approx(8.8)
