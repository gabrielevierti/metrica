"""Single-view calibration of a camera that observes a ground plane.

Used for static cameras (no parallax) and as a fallback for degenerate moving footage.
The same idea as Sopralluogo, rewritten and extended:

1. **Vertical vanishing point** - long near-vertical edges (poles, door frames, building
   corners) are parallel to gravity, so in the image they meet at one point. That point
   fixes pitch and roll (and constrains focal length).
2. **People as a ruler** (single-view metrology; Criminisi et al. 2000, Lv et al. 2006) -
   each fully visible standing person gives a (feet, head) pixel pair. With an assumed mean
   stature H, the pitch, roll, camera height (and optionally focal length) are those that
   make all people stand upright on one ground plane. This yields *metric* scale whose
   uncertainty is dominated by the stature prior (see calibration.person_height_scale_sigma).
3. **Ground plane from depth** - fallback without people: fit the floor in the back-projected
   relative depth. Scale is then unknown (RELATIVE).
4. **Affine inverse-depth alignment** - relative depth (Depth Anything) is mapped to metric
   depth with a robust fit against the calibrated ground plane.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares

from ..log import get_logger
from .camera import Camera, Intrinsics, Pose, look_rotation

log = get_logger("geometry")


# ------------------------------------------------------------------------------ verticals
def _segment_vp_angle(segs: np.ndarray, vp: np.ndarray) -> np.ndarray:
    mid = (segs[:, :2] + segs[:, 2:]) / 2
    sd = segs[:, 2:] - segs[:, :2]
    if abs(vp[2]) < 1e-9 * np.linalg.norm(vp):
        dd = np.tile(vp[:2], (len(segs), 1))
    else:
        dd = vp[:2] / vp[2] - mid
    cross = sd[:, 0] * dd[:, 1] - sd[:, 1] * dd[:, 0]
    sin = np.abs(cross) / (np.linalg.norm(sd, axis=1) * np.linalg.norm(dd, axis=1) + 1e-12)
    return np.arcsin(np.clip(sin, 0, 1))


def line_segments(bgr: np.ndarray, min_len: float = 25.0) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    try:
        lsd = cv2.createLineSegmentDetector()
        segs = lsd.detect(gray)[0]
    except cv2.error:  # some OpenCV builds lack LSD: fall back to the fast line detector
        fld = cv2.ximgproc.createFastLineDetector()
        segs = fld.detect(gray)
    if segs is None:
        return np.zeros((0, 4))
    segs = segs.reshape(-1, 4).astype(np.float64)
    length = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    return segs[length >= min_len]


def vertical_segments(bgr: np.ndarray, min_len: float | None = None, max_tilt_deg: float = 20.0,
                      seed: int = 0) -> tuple[np.ndarray, np.ndarray | None]:
    """Near-vertical segments consistent with one vanishing point. Returns (segments, vp).
    The minimum segment length scales with the image height (6 %, 20-80 px)."""
    if min_len is None:
        min_len = float(np.clip(0.06 * bgr.shape[0], 20, 80))
    segs = line_segments(bgr, min_len)
    if len(segs) == 0:
        return np.zeros((0, 4)), None
    d = segs[:, 2:] - segs[:, :2]
    ang = np.degrees(np.arctan2(np.abs(d[:, 0]), np.abs(d[:, 1])))
    segs = segs[ang < max_tilt_deg]
    if len(segs) < 6:
        return np.zeros((0, 4)), None
    p1 = np.c_[segs[:, :2], np.ones(len(segs))]
    p2 = np.c_[segs[:, 2:], np.ones(len(segs))]
    lines = np.cross(p1, p2)
    rng = np.random.default_rng(seed)
    best = None
    w = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    for _ in range(500):
        i, j = rng.choice(len(segs), 2, replace=False)
        vp = np.cross(lines[i], lines[j])
        if np.linalg.norm(vp) < 1e-12:
            continue
        inl = _segment_vp_angle(segs, vp) < np.radians(1.0)
        if best is None or w[inl].sum() > w[best].sum():
            best = inl
    if best is None or best.sum() < 6:
        return np.zeros((0, 4)), None
    # least-squares VP over inliers (smallest singular vector of stacked lines)
    L = lines[best] / np.linalg.norm(lines[best][:, :2], axis=1, keepdims=True)
    vp = np.linalg.svd(L * w[best][:, None])[2][-1]
    return segs[best], vp


# ------------------------------------------------------------------------------ focal from orthogonal VPs
def _lines(segs: np.ndarray) -> np.ndarray:
    p1 = np.c_[segs[:, :2], np.ones(len(segs))]
    p2 = np.c_[segs[:, 2:], np.ones(len(segs))]
    L = np.cross(p1, p2)
    return L / np.linalg.norm(L[:, :2], axis=1, keepdims=True)


def _fit_vp(L: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.linalg.svd(L * w[:, None])[2][-1]


def _focal_sq(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float | None:
    """f^2 = -(a - c).(b - c) for the vanishing points of two orthogonal 3D directions."""
    if abs(a[2]) < 1e-12 or abs(b[2]) < 1e-12:
        return None
    pa, pb = a[:2] / a[2], b[:2] / b[2]
    f2 = -float((pa - c) @ (pb - c))
    return f2 if f2 > 0 else None


def manhattan_focal(bgr: np.ndarray, vp_vertical: np.ndarray | None, verticals: np.ndarray | None = None,
                    min_len: float | None = None, seed: int = 0) -> dict[str, Any] | None:
    """Focal length from the vertical vanishing point and horizontal vanishing points of building
    edges, assuming those edges are horizontal and perpendicular to gravity (built environment).

    For two orthogonal 3D directions with finite vanishing points a and b and principal point c,
    f^2 = -(a - c).(b - c) (Caprile & Torre 1990). Up to two horizontal directions are searched; the
    uncertainty is the bootstrap spread over line segments. Returns None when the image does not
    contain enough consistent structure (no claim is made then). When the vertical segments are
    given, the bootstrap also resamples them, so the vertical vanishing point's uncertainty is
    included."""
    if vp_vertical is None or abs(vp_vertical[2]) < 1e-12:
        return None
    h, w = bgr.shape[:2]
    c = np.array([w / 2.0, h / 2.0])
    if min_len is None:
        min_len = float(np.clip(0.05 * h, 18, 60))
    segs = line_segments(bgr, min_len)
    if len(segs) < 20:
        return None
    segs = segs[_segment_vp_angle(segs, vp_vertical) > np.radians(8.0)]  # not vertical
    if len(segs) < 15:
        return None
    L = _lines(segs)
    wts = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    Lv = wv_ = None
    if verticals is not None and len(verticals) >= 6:
        Lv = _lines(verticals)
        wv_ = np.hypot(verticals[:, 2] - verticals[:, 0], verticals[:, 3] - verticals[:, 1])
    rng = np.random.default_rng(seed)
    remaining = np.arange(len(segs))
    found = []
    diag = float(np.hypot(w, h))
    for _ in range(2):
        if len(remaining) < 10:
            break
        best = None
        for _ in range(800):
            i, j = rng.choice(remaining, 2, replace=False)
            vp = np.cross(L[i], L[j])
            if np.linalg.norm(vp) < 1e-12:
                continue
            inl = remaining[_segment_vp_angle(segs[remaining], vp) < np.radians(1.0)]
            if best is None or wts[inl].sum() > wts[best].sum():
                best = inl
        if best is None or len(best) < 10:
            break
        vp = _fit_vp(L[best], wts[best])
        mids = (segs[best, :2] + segs[best, 2:]) / 2
        spread = float(np.linalg.norm(np.ptp(mids, axis=0)) / diag)
        found.append({"vp": vp, "idx": best, "segments": int(len(best)), "spread": spread})
        remaining = np.setdiff1d(remaining, best)
    estimates = []
    for k, hv in enumerate(found):
        if hv["spread"] < 0.15:  # all lines on one small object: not reliable
            continue
        f2 = _focal_sq(hv["vp"], vp_vertical, c)
        if f2 is None:
            continue
        f = float(np.sqrt(f2))
        hfov = float(np.degrees(2 * np.arctan(w / 2 / f)))
        if not 15.0 <= hfov <= 120.0:
            continue
        # bootstrap over the horizontal-direction segments
        boots = []
        idx = hv["idx"]
        for _ in range(60):
            sel = rng.choice(idx, len(idx), replace=True)
            vv = vp_vertical
            if Lv is not None:
                sv = rng.choice(len(Lv), len(Lv), replace=True)
                vv = _fit_vp(Lv[sv], wv_[sv])
            f2b = _focal_sq(_fit_vp(L[sel], wts[sel]), vv, c)
            if f2b is not None:
                boots.append(np.sqrt(f2b))
        if len(boots) < 40:
            continue
        estimates.append({"direction": k + 1, "f": f, "hfov_deg": hfov, "sigma_f": float(np.std(boots)),
                          "segments": hv["segments"], "spread": hv["spread"]})
    if not estimates:
        return None
    if len(found) == 2:  # the two horizontal directions are themselves orthogonal (rectangular buildings)
        f2 = _focal_sq(found[0]["vp"], found[1]["vp"], c)
        cross_check = float(np.sqrt(f2)) if f2 else None
    else:
        cross_check = None
    fs = np.array([e["f"] for e in estimates])
    sig = np.array([max(e["sigma_f"], 0.01 * e["f"]) for e in estimates])
    wv = 1 / sig ** 2
    f = float((fs * wv).sum() / wv.sum())
    sigma = float(np.sqrt(1 / wv.sum()))
    if len(fs) > 1:
        sigma = float(np.hypot(sigma, np.std(fs)))
    if len(fs) > 1 and np.ptp(fs) > 0.25 * f:
        return {"f": None, "rejected": "horizontal directions disagree", "estimates": estimates}
    return {"f": f, "sigma_f": sigma, "hfov_deg": float(np.degrees(2 * np.arctan(w / 2 / f))),
            "sigma_rel": sigma / f, "estimates": estimates, "cross_check_f_horizontal_pair": cross_check,
            "method": "orthogonal vanishing points (vertical + horizontal building edges)"}


# ------------------------------------------------------------------------------ people samples
def person_samples(observations: list[dict[str, Any]], width: int, height: int,
                   margin: int = 3) -> np.ndarray:
    """(foot_u, foot_v, head_u, head_v, track_id) for detections that look like whole standing people."""
    rows = []
    tid_index: dict[Any, int] = {}
    for d in observations:
        if d.get("category") != "person":
            continue
        x1, y1, x2, y2 = d["bbox"]
        bw, bh = x2 - x1, y2 - y1
        if d.get("conf", 0) < 0.5 or bh < 28:
            continue
        if x1 < margin or y1 < margin or x2 > width - margin or y2 > height - margin:
            continue
        if not 1.8 <= bh / max(bw, 1) <= 5.0:  # standing; rejects sitting/crouching/occluded
            continue
        u = (x1 + x2) / 2
        tid = tid_index.setdefault(d.get("track_id", -1), len(tid_index))
        rows.append((u, y2, u, y1, tid))
    return np.array(rows, dtype=np.float64).reshape(-1, 5)


def _camera(width, height, pitch, roll, h, hfov) -> Camera:
    intr = Intrinsics.from_hfov(width, height, hfov, source="estimated_single_view")
    return Camera(intr, Pose(look_rotation(pitch, roll), np.array([0.0, h, 0.0])))


def _vp_of_up(cam: Camera) -> np.ndarray:
    up_c = cam.pose.R_wc.T @ np.array([0.0, 1.0, 0.0])
    K = cam.intr.K
    return K @ up_c


def vp_strength(vp: np.ndarray | None, width: int, height: int) -> float:
    """Distance of the vertical vanishing point from the image centre, in image diagonals
    (inf when the vertical lines are parallel in the image)."""
    if vp is None or abs(vp[2]) < 1e-12:
        return float("inf")
    p = vp[:2] / vp[2]
    return float(np.hypot(p[0] - width / 2, p[1] - height / 2) / np.hypot(width, height))


def calibrate_people(samples: np.ndarray, width: int, height: int, hfov: float | None,
                     person_height: float, verticals: np.ndarray | None = None,
                     hfov_profile: list[float] | None = None) -> tuple[Camera, dict] | None:
    n_people = len(np.unique(samples[:, 4])) if len(samples) else 0
    if len(samples) < 20 or n_people < 2:
        return None
    fu, fv, hu, hv = samples[:, 0], samples[:, 1], samples[:, 2], samples[:, 3]
    fit_f = hfov is None
    feet = np.stack([fu, fv], 1)

    def build(p):
        return _camera(width, height, p[0], p[1], p[2], p[3] if fit_f else hfov)

    nv = len(verticals) if verticals is not None else 0

    def residuals(p):
        cam = build(p)
        F, ok = cam.ground_intersect(feet)
        Hd = F + np.array([0.0, person_height, 0.0])
        uv, z = cam.project(Hd)
        r = np.concatenate([uv[:, 0] - hu, uv[:, 1] - hv])
        bad = ~ok | (z <= 0)
        r[np.concatenate([bad, bad])] = 1e3
        r[~np.isfinite(r)] = 1e3
        if nv:
            vp = _vp_of_up(cam)
            half = np.linalg.norm(verticals[:, 2:] - verticals[:, :2], axis=1) / 2
            w_v = np.sqrt(len(r) / nv)  # a few dozen precise edges weigh as much as many noisy boxes
            r = np.concatenate([r, np.sin(_segment_vp_angle(verticals, vp)) * half * w_v])
        return r

    best = None
    for pitch0 in np.radians([5, 15, 28, 45, 65]):
        for h0 in (1.6, 3.0, 6.0, 12.0):
            for fov0 in ([45.0, 65.0, 90.0] if fit_f else [hfov]):
                x0 = [pitch0, 0.0, h0] + ([fov0] if fit_f else [])
                lo = [np.radians(-10), np.radians(-20), 0.3] + ([10.0] if fit_f else [])
                hi = [np.radians(89), np.radians(20), 120.0] + ([130.0] if fit_f else [])
                try:
                    r = least_squares(residuals, x0, bounds=(lo, hi), loss="soft_l1", f_scale=4.0, max_nfev=400)
                except ValueError:
                    continue
                if best is None or r.cost < best.cost:
                    best = r
    if best is None:
        return None
    cam = build(best.x)
    res = residuals(best.x)
    n = len(samples)
    err = np.hypot(res[:n], res[n:2 * n])
    inl = err < 8
    try:
        J = best.jac
        sigma2 = float(np.median(err[inl]) ** 2) if inl.any() else float(np.median(err) ** 2)
        cov = np.linalg.pinv(J.T @ J) * sigma2
        std = np.sqrt(np.clip(np.diag(cov), 0, None))
    except Exception:  # pragma: no cover
        std = np.full(len(best.x), np.nan)
    vert_err = None
    if nv:
        vp = _vp_of_up(cam)
        half = np.linalg.norm(verticals[:, 2:] - verticals[:, :2], axis=1) / 2
        vert_err = float(np.median(np.sin(_segment_vp_angle(verticals, vp)) * half))
    box_h = np.maximum(fv - hv, 1.0)
    rel_err = err / box_h
    report = {
        "median_relative_error": float(np.median(rel_err)),
        "inlier_ratio_relative": float((rel_err < 0.12).mean()),
        "method": "single-view metrology: people as a ruler + vertical vanishing point",
        "assumed_person_height_m": person_height,
        "samples": int(n), "people": int(n_people),
        "inlier_ratio": float(inl.mean()),
        "median_reprojection_px": float(np.median(err)),
        "pitch_deg": float(np.degrees(best.x[0])), "roll_deg": float(np.degrees(best.x[1])),
        "camera_height_m": float(best.x[2]), "hfov_deg": cam.intr.hfov_deg,
        "std_pitch_deg": float(np.degrees(std[0])), "std_roll_deg": float(np.degrees(std[1])),
        "std_camera_height_m_pixel_noise_only": float(std[2]),
        "focal_estimated": fit_f, "std_hfov_deg": float(std[3]) if fit_f else None,
        "vertical_lines": int(nv), "vertical_lines_median_px": vert_err,
    }
    # ---- focal-length identifiability: profile the fit over fixed fields of view
    if fit_f:
        best_err = float(np.median(err))
        prof = []
        grid = sorted({round(float(best.x[3]), 2), 12.0, 18.0, 25.0, 35.0, 50.0, 65.0, 85.0, 110.0})
        for hf in grid:

            def r_fixed(p, hf=hf):
                return residuals(np.r_[p, hf])
            try:
                rr = least_squares(r_fixed, best.x[:3], bounds=([np.radians(-10), np.radians(-20), 0.3],
                                                                [np.radians(89), np.radians(20), 120.0]),
                                   loss="soft_l1", f_scale=4.0, max_nfev=200)
            except ValueError:
                continue
            rv = r_fixed(rr.x)
            e = float(np.median(np.hypot(rv[:n], rv[n:2 * n])))
            prof.append({"hfov_deg": hf, "camera_height_m": float(rr.x[2]), "pitch_deg": float(np.degrees(rr.x[0])),
                         "median_error_px": e, "plausible": bool(e <= 1.25 * best_err + 0.5)})
        plaus = [p for p in prof if p["plausible"]]
        hs = [p["camera_height_m"] for p in plaus] or [float(best.x[2])]
        report["focal_profile"] = prof
        report["hfov_plausible_range_deg"] = [min(p["hfov_deg"] for p in plaus), max(p["hfov_deg"] for p in plaus)] \
            if plaus else None
        report["scale_sigma_rel_focal"] = float((max(hs) - min(hs)) / (2 * max(float(best.x[2]), 1e-6)))
        report["focal_at_search_bound"] = bool(best.x[3] <= 10.5 or best.x[3] >= 129.5)
        if report["focal_at_search_bound"]:
            report["warning"] = "Field of view converged to a search bound: focal length is not identifiable."
    elif hfov_profile:
        # focal assumed: how much would the metric result change across a plausible lens range?
        prof = []
        for hf in hfov_profile:
            def r_fixed(p, hf=hf):
                return residuals_fixed(p, hf)

            def residuals_fixed(p, hf):
                cam_ = _camera(width, height, p[0], p[1], p[2], hf)
                F, ok = cam_.ground_intersect(feet)
                uv, z = cam_.project(F + np.array([0.0, person_height, 0.0]))
                r = np.concatenate([uv[:, 0] - hu, uv[:, 1] - hv])
                bad = ~ok | (z <= 0)
                r[np.concatenate([bad, bad])] = 1e3
                r[~np.isfinite(r)] = 1e3
                return r
            try:
                rr = least_squares(r_fixed, best.x[:3], bounds=([np.radians(-10), np.radians(-20), 0.3],
                                                                [np.radians(89), np.radians(20), 120.0]),
                                   loss="soft_l1", f_scale=4.0, max_nfev=200)
            except ValueError:
                continue
            rv = r_fixed(rr.x)
            prof.append({"hfov_deg": hf, "camera_height_m": float(rr.x[2]), "pitch_deg": float(np.degrees(rr.x[0])),
                         "median_error_px": float(np.median(np.hypot(rv[:n], rv[n:2 * n])))})
        hs = [p["camera_height_m"] for p in prof] + [float(best.x[2])]
        report["focal_profile"] = prof
        report["scale_sigma_rel_focal"] = float((max(hs) - min(hs)) / (2 * max(float(best.x[2]), 1e-6)))
    log.info(f"Single-view calibration: pitch {report['pitch_deg']:.1f} deg, roll {report['roll_deg']:.1f} deg, "
             f"camera height {report['camera_height_m']:.2f} m, HFOV {report['hfov_deg']:.1f} deg, "
             f"median error {report['median_reprojection_px']:.1f} px over {n} observations of {n_people} people")
    return cam, report


def calibrate_vertical_vp(vp: np.ndarray, width: int, height: int, hfov: float) -> tuple[float, float] | None:
    """Pitch and roll from the vertical vanishing point with a given focal length."""
    K = Intrinsics.from_hfov(width, height, hfov).K
    up = np.linalg.solve(K, vp)
    up /= np.linalg.norm(up)
    if up[1] > 0:  # world up points towards image top (negative y in OpenCV camera frame)
        up = -up
    # up_c = Rz(roll) @ [0, -cos(pitch), -sin(pitch)]
    pitch = float(np.arcsin(np.clip(-up[2], -1, 1)))
    roll = float(np.arctan2(up[0], -up[1]))
    return pitch, roll


def ground_plane_from_depth(inv_depth: np.ndarray, intr: Intrinsics, mask: np.ndarray | None = None,
                            seed: int = 0) -> tuple[np.ndarray, float, float] | None:
    """Fit a plane to back-projected (relative) depth in ``mask`` (default: lower 40%).

    Returns (normal_cam (pointing up), distance (relative units), inlier ratio)."""
    h, w = inv_depth.shape
    if mask is None:
        mask = np.zeros((h, w), bool)
        mask[int(h * 0.6):] = True
    vs, us = np.nonzero(mask[::4, ::4])
    vs, us = vs * 4, us * 4
    if len(vs) < 50:
        return None
    d = inv_depth[vs, us]
    ok = d > np.percentile(inv_depth, 5) + 1e-9
    z = 1.0 / d[ok]
    xy = intr.normalized(np.stack([us[ok], vs[ok]], 1).astype(float))
    P = np.c_[xy * z[:, None], z]
    rng = np.random.default_rng(seed)
    best_inl = None
    for _ in range(300):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        nrm = np.cross(b - a, c - a)
        if np.linalg.norm(nrm) < 1e-12:
            continue
        nrm /= np.linalg.norm(nrm)
        inl = np.abs((P - a) @ nrm) < 0.02 * np.median(z)
        if best_inl is None or inl.sum() > best_inl.sum():
            best_inl = inl
    if best_inl is None or best_inl.sum() < 30:
        return None
    Q = P[best_inl]
    c0 = Q.mean(0)
    nrm = np.linalg.svd(Q - c0)[2][-1]
    if nrm[1] > 0:  # OpenCV y points down: 'up' has negative y
        nrm = -nrm
    return nrm, float(abs(c0 @ nrm)), float(best_inl.mean())


def pitch_roll_from_up(up_c: np.ndarray) -> tuple[float, float]:
    up = up_c / np.linalg.norm(up_c)
    return float(np.arcsin(np.clip(-up[2], -1, 1))), float(np.arctan2(up[0], -up[1]))


def fit_inverse_depth_affine(inv_depth: np.ndarray, uv: np.ndarray, z_metric: np.ndarray,
                             weights: np.ndarray | None = None, seed: int = 0) -> dict[str, Any] | None:
    """Robustly find a, b with inv_depth ~= a / z + b at pixels uv with known metric depth z.

    Returns {"a", "b", "inlier_ratio", "rel_residual"} or None. Metric depth is then
    z = a / (inv_depth - b).
    """
    h, w = inv_depth.shape
    uv = np.asarray(uv, float)
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h) & np.isfinite(z_metric) & (z_metric > 0)
    if ok.sum() < 8:
        return None
    x = 1.0 / z_metric[ok]
    y = inv_depth[uv[ok, 1].astype(int), uv[ok, 0].astype(int)]
    wts = weights[ok] if weights is not None else np.ones_like(x)
    rng = np.random.default_rng(seed)
    best, best_score = None, -1.0
    span = np.ptp(y) + 1e-9
    for _ in range(600):
        i, j = rng.choice(len(x), 2, replace=False)
        if abs(x[i] - x[j]) < 1e-12:
            continue
        a = (y[i] - y[j]) / (x[i] - x[j])
        if a <= 0:
            continue
        b = y[i] - a * x[i]
        inl = np.abs(y - (a * x + b)) < 0.04 * span
        score = wts[inl].sum()
        if score > best_score:
            best, best_score = inl, score
    if best is None or best.sum() < 6:
        return None
    A = np.stack([x[best], np.ones(best.sum())], 1) * np.sqrt(wts[best])[:, None]
    a, b = np.linalg.lstsq(A, y[best] * np.sqrt(wts[best]), rcond=None)[0]
    if a <= 0:
        return None
    z_fit = a / np.maximum(y[best] - b, 1e-9)
    rel = np.abs(z_fit - 1 / x[best]) * x[best]
    return {"a": float(a), "b": float(b), "inlier_ratio": float(best.mean()), "rel_residual": float(np.median(rel)),
            "points": int(len(x))}


def metric_depth_from_affine(inv_depth: np.ndarray, a: float, b: float, max_depth: float) -> np.ndarray:
    denom = inv_depth - b
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(denom > 1e-6 * max(abs(a), 1e-9), a / denom, np.nan)
    z[(z <= 0.05) | (z > max_depth)] = np.nan
    return z.astype(np.float32)
