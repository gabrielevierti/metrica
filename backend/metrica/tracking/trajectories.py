"""3D trajectories from 2D tracks and camera geometry.

Positions come from the ground-contact point of each observation (bottom centre of the
box) intersected with the ground plane in the unit frame. Every sample keeps its source
frame, its 1-sigma position uncertainty and flags. Observed samples and smoothed /
interpolated values are stored separately; nothing interpolated is ever presented as an
observation.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import savgol_filter

from ..geometry.camera import Camera
from ..reconstruction.objects3d import VEHICLE_PRIORS

UP = np.array([0.0, 1.0, 0.0])


# speeds above which a track's kinematics are flagged (m/s; scene units in RELATIVE scenes)
PLAUSIBLE_SPEED = {"person": 4.5, "two_wheeler": 25.0, "vehicle": 60.0}


def foot_uncertainty(cam: Camera, u: float, v: float, box_h: float) -> float:
    """1-sigma ground position error from foot-row uncertainty (2 px + 3% of box height)."""
    e = 2.0 + 0.03 * box_h
    a, oka = cam.ground_intersect(np.array([[u, v - e]]))
    b, okb = cam.ground_intersect(np.array([[u, v + e]]))
    if not (oka[0] and okb[0]):
        return float("nan")
    return float(np.linalg.norm(a[0] - b[0]) / 2)


def stature(cam: Camera, foot: np.ndarray, v_top: float) -> float | None:
    """Height of the vertical segment at `foot` whose top projects to image row v_top."""
    lo, hi = 0.0, 4.0
    def row(hh):
        uv, z = cam.project((foot + hh * UP)[None])
        return uv[0, 1] if z[0] > 0 else np.nan
    r_lo, r_hi = row(lo), row(hi)
    if not (np.isfinite(r_lo) and np.isfinite(r_hi)) or not (r_hi <= v_top <= r_lo):
        return None
    for _ in range(30):
        mid = (lo + hi) / 2
        if row(mid) > v_top:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def build_track(track: dict[str, Any], cam_for_frame, width: int, height: int, scale_rel_sigma: float | None,
                pose_source: dict[int, str], person_height: float = 1.70) -> dict[str, Any]:
    cat = track["category"]
    is_vehicle = cat in VEHICLE_PRIORS
    samples, statures = [], []
    for o in track["observations"]:
        cam = cam_for_frame(o["frame"])
        x1, y1, x2, y2 = o["bbox"]
        flags = []
        if cam is None:
            continue
        if y2 >= height - 2:
            flags.append("ground contact outside frame")
        if x1 <= 1 or x2 >= width - 1:
            flags.append("box truncated at image side")
        u, v = (x1 + x2) / 2, y2
        P, ok = cam.ground_intersect(np.array([[u, v]]))
        if not ok[0]:
            continue
        P = P[0]
        sig = foot_uncertainty(cam, u, v, y2 - y1)
        src = pose_source.get(o["frame"], "static")
        if src == "interpolated":
            sig = float(np.hypot(sig, 0.03 * np.linalg.norm(P - cam.center)))
            flags.append("camera pose interpolated")
        if is_vehicle:
            # box bottom marks the nearest edge; move towards the vehicle centre along the view direction
            pri = VEHICLE_PRIORS[cat]
            d = P - cam.center
            d[1] = 0
            d /= max(np.linalg.norm(d), 1e-9)
            P = P + d * 0.5 * pri["width"][0]
            sig = float(np.hypot(sig, 0.25 * pri["length"][0]))
        if "ground contact outside frame" in flags:
            sig = float("nan")
        samples.append({"f": o["frame"], "t": round(o["t"], 4), "p": [round(float(P[0]), 4), 0.0,
                        round(float(P[2]), 4)], "s": None if not np.isfinite(sig) else round(sig, 4),
                        "conf": round(o["conf"], 3), "flags": flags, "pose": src,
                        "b": [round(float(x1), 1), round(float(y1), 1), round(float(x2), 1), round(float(y2), 1)]})
        if cat == "person" and not flags and (y2 - y1) > 30:
            h = stature(cam, P, y1)
            if h:
                samples[-1]["_h"] = h
                if 1.8 <= (y2 - y1) / max(x2 - x1, 1) <= 5:
                    statures.append(h)
    if cat == "person":
        # a box much shorter (or taller) than the subject's stature means hidden feet, a cut head or two
        # people merged: the foot point is then not on the ground and the position is unreliable
        ref = float(np.median(statures)) if len(statures) >= 5 else person_height
        for smp in samples:
            h = smp.pop("_h", None)
            if h is not None and not (0.72 * ref <= h <= 1.35 * ref):
                smp["flags"].append("box inconsistent with the subject's stature (feet hidden or boxes merged): "
                                    "position unreliable")
                smp["s"] = None
    good = [s for s in samples if s["s"] is not None]
    out = {"id": track["id"], "category": cat, "group": track["group"], "samples": samples,
           "observed_count": len(samples), "first_t": track["t0"], "last_t": track["t1"],
           "gaps": track["gaps"], "flags": track["flags"], "links": track.get("links", [])}
    if statures:
        out["stature"] = {"median": float(np.median(statures)), "iqr": float(np.subtract(*np.percentile(statures, [75, 25]))),
                          "n": len(statures)}
    if len(good) >= 2:
        out.update(_kinematics(good, scale_rel_sigma))
        vmax = (out.get("summary") or {}).get("speed_max")
        limit = PLAUSIBLE_SPEED.get(track["group"])
        if vmax is not None and limit is not None and vmax > limit:
            t_max = max(out["speed"], key=lambda x: x["v"])["t"] if out.get("speed") else out["first_t"]
            out["flags"] = list(out["flags"]) + [{
                "type": "implausible_speed", "t": t_max, "speed_max": round(vmax, 2), "limit": limit,
                "note": f"Speeds above {limit} m/s are implausible for this class: positions are probably affected "
                        "by occlusion, a detection error or an identity switch. Do not rely on these speeds."}]
    return out


def _segments(ts: np.ndarray, max_gap: float) -> list[np.ndarray]:
    breaks = np.where(np.diff(ts) > max_gap)[0]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks + 1, len(ts)]
    return [np.arange(a, b) for a, b in zip(starts, ends)]


def _kinematics(good: list[dict[str, Any]], scale_rel_sigma: float | None, max_gap: float = 1.5) -> dict[str, Any]:
    ts = np.array([s["t"] for s in good])
    P = np.array([s["p"] for s in good], float)
    sig = np.array([s["s"] for s in good], float)
    smooth = P.copy()
    speed = np.full(len(ts), np.nan)
    speed_sigma = np.full(len(ts), np.nan)
    for seg in _segments(ts, max_gap):
        if len(seg) >= 5:
            dt = np.median(np.diff(ts[seg]))
            win = int(round(0.8 / max(dt, 1e-3))) | 1
            win = max(5, min(win, len(seg) if len(seg) % 2 else len(seg) - 1))
            if win >= 5 and win <= len(seg):
                for k in (0, 2):
                    smooth[seg, k] = savgol_filter(P[seg, k], win, 2)
        # speed: least-squares slope over a +-0.6 s window of smoothed positions
        for _j, i in enumerate(seg):
            m = seg[np.abs(ts[seg] - ts[i]) <= 0.6]
            if len(m) < 3 or np.ptp(ts[m]) < 0.4:
                continue
            A = np.c_[ts[m] - ts[i], np.ones(len(m))]
            vx = np.linalg.lstsq(A, smooth[m, 0], rcond=None)[0][0]
            vz = np.linalg.lstsq(A, smooth[m, 2], rcond=None)[0][0]
            speed[i] = float(np.hypot(vx, vz))
            # positional noise over the window, plus the scale uncertainty
            s_pos = float(np.median(sig[m])) * np.sqrt(12.0 / len(m)) / max(np.ptp(ts[m]), 1e-3)
            speed_sigma[i] = float(np.hypot(s_pos, speed[i] * (scale_rel_sigma or 0.0)))
    steps = np.linalg.norm(np.diff(smooth[:, [0, 2]], axis=0), axis=1)
    within = np.diff(ts) <= max_gap
    length_obs = float(steps[within].sum())
    finite = np.isfinite(speed)
    return {
        "smoothed": [{"t": round(float(t), 4), "p": [round(float(p[0]), 4), 0.0, round(float(p[2]), 4)]}
                     for t, p in zip(ts, smooth)],
        "speed": [{"t": round(float(t), 4), "v": round(float(v), 4), "s": round(float(e), 4)}
                  for t, v, e in zip(ts[finite], speed[finite], speed_sigma[finite])],
        "summary": {"path_length_observed": length_obs,
                    "path_note": "sum of smoothed displacements between consecutive observations; gaps longer than "
                                 f"{max_gap} s are not bridged",
                    "displacement": float(np.linalg.norm(smooth[-1, [0, 2]] - smooth[0, [0, 2]])),
                    "speed_median": float(np.median(speed[finite])) if finite.any() else None,
                    "speed_max": float(np.max(speed[finite])) if finite.any() else None,
                    "position_sigma_median": float(np.median(sig))},
    }
