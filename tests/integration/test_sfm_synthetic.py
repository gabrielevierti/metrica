"""Structure from motion on rendered frames with known camera poses (no neural models needed; slow).

Measures the camera trajectory error after a similarity alignment (scale is unobservable from
images alone) and the recovered focal length. Thresholds are deliberately loose; the measured
values are printed so they can be recorded in docs/EVALUATION.md.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

pytestmark = pytest.mark.slow
pycolmap = pytest.importorskip("pycolmap")


def _umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / ((xs ** 2).sum() / len(src))
    return s, R, mu_d - s * R @ mu_s


def test_sfm_recovers_walking_trajectory(tmp_path):
    from metrica.geometry.sfm import run_sfm
    from metrica.synthetic.scene import render, street_scene, walking_path

    n = 16
    intr, Rs, Cs = walking_path(n, 640, 360)
    scene = street_scene()
    items = []
    for k in range(n):
        img, _, _ = render(scene, intr, Rs[k], Cs[k], k / 3.0)
        p = tmp_path / "kf" / f"{k:04d}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        items.append((k, p))
    res = run_sfm(tmp_path / "sfm", {"v1": items}, max_features=4096, overlap=8)
    assert res.ok, res.warnings
    frames = sorted(f for (v, f) in res.poses)
    assert len(frames) >= 0.8 * n
    est = np.array([res.poses[("v1", f)].C for f in frames])
    gt = np.array([Cs[f] for f in frames])
    s, R, t = _umeyama(est, gt)
    aligned = s * est @ R.T + t
    ate = float(np.sqrt(((aligned - gt) ** 2).sum(1).mean()))
    path_len = float(np.linalg.norm(np.diff(gt, axis=0), axis=1).sum())
    f_est = res.intrinsics["v1"].fx
    print(f"SfM synthetic: registered {len(frames)}/{n}, ATE {ate * 1000:.1f} mm over {path_len:.1f} m path, "
          f"focal {f_est:.1f} px (true {intr.fx:.1f})")
    assert ate < 0.02 * path_len
    assert abs(f_est - intr.fx) / intr.fx < 0.08
