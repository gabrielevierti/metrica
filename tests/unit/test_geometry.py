"""Camera model, single-view metrology on synthetic ground truth, scale/calibration records."""
from __future__ import annotations

import numpy as np
import pytest

from metrica.geometry.calibration import known_distance_scale, person_height_scale_sigma, resolve_global_scale
from metrica.geometry.camera import Camera, Intrinsics, Pose, look_rotation
from metrica.geometry.gravity import rotation_between
from metrica.geometry.single_view import calibrate_people, calibrate_vertical_vp, fit_inverse_depth_affine


def _cam(pitch_deg=20.0, roll_deg=0.0, h=4.0, hfov=60.0, w=960, hh=540):
    intr = Intrinsics.from_hfov(w, hh, hfov)
    return Camera(intr, Pose(look_rotation(np.radians(pitch_deg), np.radians(roll_deg)), np.array([0.0, h, 0.0])))


def test_look_rotation_is_orthonormal_and_points_forward_down():
    R = look_rotation(np.radians(25), np.radians(3))
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(R), 1)
    fwd_world = R @ np.array([0, 0, 1.0])
    assert fwd_world[1] < 0 and fwd_world[2] < 0  # looks down and towards -Z


def test_project_backproject_and_ground_intersect_roundtrip():
    cam = _cam()
    rng = np.random.default_rng(0)
    P = np.c_[rng.uniform(-5, 5, 50), np.zeros(50), rng.uniform(-30, -6, 50)]
    uv, z = cam.project(P)
    assert (z > 0).all()
    G, ok = cam.ground_intersect(uv)
    assert ok.all() and np.allclose(G, P, atol=1e-6)
    B = cam.backproject(uv, z)
    assert np.allclose(B, P, atol=1e-6)


def test_radial_distortion_inverse():
    intr = Intrinsics(800, 800, 480, 270, 960, 540, k1=-0.12)
    xy = np.array([[0.3, -0.2], [-0.5, 0.25], [0.0, 0.0]])
    assert np.allclose(intr.normalized(intr.project_normalized(xy)), xy, atol=1e-5)


def test_pose_similarity_transform():
    p = Pose(look_rotation(0.3, 0.0), np.array([1.0, 2.0, 3.0]))
    S = np.eye(4)
    S[:3, :3] *= 2.0
    S[:3, 3] = [5, 0, 0]
    q = p.transformed(S)
    assert np.allclose(q.C, [7, 4, 6]) and np.allclose(q.R_wc, p.R_wc)


def _people_samples(cam, n_people=8, per_person=12, height=1.70, noise_px=0.8, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_people):
        x0, z0 = rng.uniform(-6, 6), rng.uniform(-28, -9)
        hp = height + rng.normal(0, 0.06)
        for k in range(per_person):
            foot = np.array([x0 + 0.3 * k, 0.0, z0 + 0.2 * k])
            uf, _ = cam.project(foot[None])
            uh, _ = cam.project((foot + [0, hp, 0])[None])
            rows.append((uf[0, 0] + rng.normal(0, noise_px), uf[0, 1] + rng.normal(0, noise_px),
                         uh[0, 0] + rng.normal(0, noise_px), uh[0, 1] + rng.normal(0, noise_px), pid))
    return np.array(rows)


def test_people_as_ruler_recovers_height_and_pitch_with_known_fov():
    gt = _cam(pitch_deg=18.0, h=4.5, hfov=60.0)
    s = _people_samples(gt)
    cam, rep = calibrate_people(s, 960, 540, hfov=60.0, person_height=1.70)
    assert abs(rep["camera_height_m"] - 4.5) / 4.5 < 0.06
    assert abs(rep["pitch_deg"] - 18.0) < 1.5
    assert rep["median_reprojection_px"] < 5


def test_people_calibration_reports_focal_identifiability():
    gt = _cam(pitch_deg=25.0, h=5.0, hfov=55.0)
    s = _people_samples(gt, n_people=10)
    cam, rep = calibrate_people(s, 960, 540, hfov=None, person_height=1.70)
    assert rep["focal_estimated"] and "focal_profile" in rep and rep["scale_sigma_rel_focal"] >= 0
    # needs at least 2 people and 20 samples
    assert calibrate_people(s[:10], 960, 540, hfov=60.0, person_height=1.7) is None


def test_vertical_vp_gives_pitch_and_roll():
    cam = _cam(pitch_deg=22.0, roll_deg=4.0)
    up_c = cam.pose.R_wc.T @ np.array([0, 1.0, 0])
    vp = cam.intr.K @ up_c
    pitch, roll = calibrate_vertical_vp(vp, 960, 540, 60.0)
    assert abs(np.degrees(pitch) - 22.0) < 0.5 and abs(np.degrees(roll) - 4.0) < 0.5


def test_inverse_depth_affine_fit_is_robust():
    rng = np.random.default_rng(3)
    h, w = 60, 80
    z_true = rng.uniform(3, 40, (h, w))
    inv = 2.0 / z_true + 0.01 + rng.normal(0, 1e-4, (h, w))  # relative inverse depth = a/z + b
    uv = np.c_[rng.integers(0, w, 300), rng.integers(0, h, 300)].astype(float)
    zm = z_true[uv[:, 1].astype(int), uv[:, 0].astype(int)].copy()
    zm[:30] *= 3.0  # outliers
    r = fit_inverse_depth_affine(inv, uv, zm)
    assert r is not None and r["inlier_ratio"] > 0.8 and r["rel_residual"] < 0.05


def test_scale_uncertainty_models():
    assert person_height_scale_sigma(1) > person_height_scale_sigma(10) > person_height_scale_sigma(1000)
    # systematic terms do not average away
    assert person_height_scale_sigma(10 ** 6) > 0.03
    k, sig = known_distance_scale([(2.0, 5.0, 0.01)])
    assert k == pytest.approx(2.5) and 0.009 < sig < 0.012
    base = {"status": "ESTIMATED_METRIC", "unit": "m", "method": "people", "scale_sigma_rel": 0.2, "gravity_aligned": True}
    k0, rec0 = resolve_global_scale(base, [])
    assert k0 == 1.0 and rec0["status"] == "ESTIMATED_METRIC"
    refs = [{"id": "C1", "p1": [0, 0, 0], "p2": [4, 0, 0], "true_length_m": 5.0, "sigma_m": 0.02, "active": True},
            {"id": "C2", "p1": [0, 0, 0], "p2": [0, 0, 2], "true_length_m": 2.5, "sigma_m": 0.02, "active": False}]
    k1, rec1 = resolve_global_scale(base, refs)
    assert k1 == pytest.approx(1.25) and rec1["status"] == "CALIBRATED_METRIC" and len(rec1["references"]) == 1


def test_rotation_between():
    a, b = np.array([0, 0, 1.0]), np.array([0, 1.0, 0])
    R = rotation_between(a, b)
    assert np.allclose(R @ a, b, atol=1e-9)


def _wireframe_image(cam, boxes, w=960, h=540):
    import cv2
    img = np.full((h, w, 3), 90, np.uint8)
    for (cx, cz, sx, sz, hh) in boxes:
        xs, zs = (cx - sx / 2, cx + sx / 2), (cz - sz / 2, cz + sz / 2)
        corners = [(x, y, z) for x in xs for y in (0.0, hh) for z in zs]
        edges = []
        for i, a in enumerate(corners):
            for b in corners[i + 1:]:
                if sum(abs(np.subtract(a, b)) > 1e-9) == 1:
                    edges.append((a, b))
        # windows: horizontal lines on the facades (more horizontal evidence)
        for y in np.arange(1.0, hh, 1.5):
            edges.append(((xs[0], y, zs[1]), (xs[1], y, zs[1])))
            edges.append(((xs[1], y, zs[0]), (xs[1], y, zs[1])))
        for a, b in edges:
            P = np.array([a, b], float)
            uv, z = cam.project(P)
            if (z > 0.5).all():
                cv2.line(img, tuple(np.round(uv[0]).astype(int)), tuple(np.round(uv[1]).astype(int)), (230, 230, 230), 2,
                         cv2.LINE_AA)
    return img


def test_focal_from_orthogonal_vanishing_points():
    from metrica.geometry.single_view import manhattan_focal, vertical_segments

    intr = Intrinsics.from_hfov(960, 540, 55.0)
    R = look_rotation(np.radians(14.0), np.radians(2.0), yaw=np.radians(25.0))
    cam = Camera(intr, Pose(R, np.array([0.0, 6.0, 0.0])))
    boxes = [(-6, -30, 8, 6, 9), (8, -26, 6, 10, 12), (0, -45, 14, 8, 15), (-14, -20, 5, 5, 7)]
    img = _wireframe_image(cam, boxes)
    segs, vp = vertical_segments(img)
    assert vp is not None and len(segs) >= 6
    r = manhattan_focal(img, vp, segs)
    assert r is not None and r["f"] is not None
    assert abs(r["f"] - intr.fx) / intr.fx < 0.05, (r["f"], intr.fx)
    assert r["sigma_rel"] < 0.1


def test_focal_not_claimed_without_structure():
    from metrica.geometry.single_view import manhattan_focal

    rng = np.random.default_rng(0)
    img = (rng.random((540, 960, 3)) * 255).astype(np.uint8)
    assert manhattan_focal(img, np.array([480.0, 5000.0, 1.0])) is None
