"""Measurement engine: values, units, calibration status and uncertainty for every measurement.

Point uncertainty depends on the evidence class of the surface the point was picked on and on
its distance from the nearest camera (depth error grows with range). The scale uncertainty of
the coordinate system is added in quadrature for lengths. Measurements in RELATIVE or
ARBITRARY frames are always labelled as not being real-world distances.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

import numpy as np

# relative depth uncertainty (1 sigma, fraction of camera distance) per evidence class
POINT_REL_SIGMA = {"OBSERVED_MULTI_VIEW": 0.02, "OBSERVED_SINGLE_VIEW": 0.06, "FITTED_PRIMITIVE": 0.03,
                   "CLASS_PRIOR": 0.10, "INFERRED": 0.15, "UNKNOWN": 0.25}
KINDS = ("distance", "horizontal_distance", "height_difference", "height_above_ground", "angle", "path_length",
         "track_distance_at_time", "track_speed_at_time", "point")


def _cam_centers(scene: dict[str, Any]) -> np.ndarray:
    cs = []
    for c in scene["cameras"]:
        if c["kind"] == "moving":
            cs += [np.asarray(m)[:3, 3] for m in c["poses"]["world_from_cam"][:: max(1, len(c["poses"]["frames"]) // 50)]]
        else:
            cs.append(np.asarray(c["world_from_cam"])[:3, 3])
    return np.array(cs) if cs else np.zeros((1, 3))


def point_sigma(p: np.ndarray, evidence: str | None, centers: np.ndarray) -> float:
    d = float(np.min(np.linalg.norm(centers - p, axis=1)))
    return POINT_REL_SIGMA.get(evidence or "UNKNOWN", 0.1) * d


def _display(value: float, sigma: float | None, unit: str, status: str, kind: str) -> str:
    if kind == "angle":
        s = f"{value:.1f}°" + (f" ± {sigma:.1f}°" if sigma else "")
        return s
    s = f"{value:.2f}" + (f" ± {sigma:.2f}" if sigma is not None else "") + f" {unit}"
    if status in ("RELATIVE", "ARBITRARY"):
        s += " (scene units - not a real-world measurement)"
    else:
        s += f" ({status})"
    return s


def compute(scene: dict[str, Any], kind: str, points: list[list[float]] | None = None,
            evidence: list[str | None] | None = None, targets: list[str] | None = None,
            t: float | None = None, t0: float | None = None, t1: float | None = None) -> dict[str, Any]:
    if kind not in KINDS:
        raise ValueError(f"unknown measurement kind '{kind}'")
    cs = scene["coordinate_system"]
    unit, status = cs["unit"], cs["status"]
    sig_scale = cs["scale"].get("scale_sigma_rel") or 0.0
    centers = _cam_centers(scene)
    P = [np.asarray(p, float) for p in (points or [])]
    ev = list(evidence or [None] * len(P))
    sig = [point_sigma(p, e, centers) for p, e in zip(P, ev)]
    details: dict[str, Any] = {}
    if kind in ("distance", "horizontal_distance", "height_difference"):
        if len(P) != 2:
            raise ValueError("needs exactly two points")
        d = P[1] - P[0]
        if kind == "horizontal_distance":
            d = d[[0, 2]]
        elif kind == "height_difference":
            d = d[[1]]
        val = float(np.linalg.norm(d))
        sigma = float(np.hypot(np.hypot(*sig), val * sig_scale))
        out_unit = unit
    elif kind == "height_above_ground":
        if len(P) != 1:
            raise ValueError("needs one point")
        val = float(P[0][1])
        sigma = float(np.hypot(sig[0], abs(val) * sig_scale))
        out_unit = unit
    elif kind == "point":
        if len(P) != 1:
            raise ValueError("needs one point")
        val = 0.0
        sigma = sig[0]
        out_unit = unit
        details["coordinates"] = P[0].round(4).tolist()
    elif kind == "angle":
        if len(P) != 3:
            raise ValueError("needs three points (vertex in the middle)")
        a, b = P[0] - P[1], P[2] - P[1]
        cosv = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
        val = float(np.degrees(np.arccos(np.clip(cosv, -1, 1))))
        # angular uncertainty from point uncertainty (small-angle approximation)
        sigma = float(np.degrees(np.hypot(sig[0] / max(np.linalg.norm(a), 1e-9), sig[2] / max(np.linalg.norm(b), 1e-9))))
        out_unit = "deg"
    else:
        agents = {a["id"]: a for a in scene["agents"]}
        tr = [agents[x] for x in (targets or []) if x in agents]
        if kind == "path_length":
            if len(tr) != 1:
                raise ValueError("needs one track")
            sm = tr[0].get("smoothed") or []
            pts = [(s["t"], np.asarray(s["p"])) for s in sm if (t0 is None or s["t"] >= t0) and (t1 is None or s["t"] <= t1)]
            if len(pts) < 2:
                raise ValueError("not enough observed samples in the interval")
            steps = [np.linalg.norm((b[1] - a[1])[[0, 2]]) for a, b in zip(pts[:-1], pts[1:]) if b[0] - a[0] <= 1.5]
            val = float(np.sum(steps))
            ps = tr[0]["summary"].get("position_sigma_median") or 0.0
            sigma = float(np.hypot(np.sqrt(2) * ps, val * sig_scale))
            out_unit = unit
            details = {"interval": [pts[0][0], pts[-1][0]], "observed_samples": len(pts),
                       "note": "gaps longer than 1.5 s are not bridged"}
        elif kind == "track_distance_at_time":
            if len(tr) != 2 or t is None:
                raise ValueError("needs two tracks and a time")
            pp, ss = [], []
            for a in tr:
                sm = a.get("smoothed") or []
                ts = np.array([s["t"] for s in sm])
                if not len(ts) or t < ts[0] or t > ts[-1]:
                    raise ValueError(f"{a['id']} is not observed at t={t:.2f} s")
                if float(np.min(np.abs(ts - t))) > 0.75:
                    raise ValueError(f"{a['id']} is inside an observation gap at t={t:.2f} s; "
                                     "positions are not interpolated across gaps")
                P_ = np.array([s["p"] for s in sm])
                pp.append(np.array([np.interp(t, ts, P_[:, k]) for k in range(3)]))
                ss.append(a["summary"].get("position_sigma_median") or 0.0)
            val = float(np.linalg.norm((pp[1] - pp[0])[[0, 2]]))
            sigma = float(np.hypot(np.hypot(*ss), val * sig_scale))
            out_unit = unit
            details = {"t": t, "positions": [p.round(3).tolist() for p in pp]}
        else:  # track_speed_at_time
            if len(tr) != 1 or t is None:
                raise ValueError("needs one track and a time")
            if status not in ("ESTIMATED_METRIC", "CALIBRATED_METRIC"):
                raise ValueError(f"speeds need a metric scale; the scene is {status}")
            sp = tr[0].get("speed") or []
            if not sp:
                raise ValueError("no speed estimate for this track")
            k = int(np.argmin([abs(s["t"] - t) for s in sp]))
            if abs(sp[k]["t"] - t) > 0.5:
                raise ValueError("no speed estimate near this time")
            val, sigma = float(sp[k]["v"]), float(sp[k]["s"])
            out_unit = f"{unit}/s"
            details = {"t": sp[k]["t"]}
    return {"kind": kind, "value": val, "sigma": sigma, "unit": out_unit, "status": status,
            "display": _display(val, sigma, out_unit, status, kind), "details": details,
            "uncertainty_model": "point sigma = class-dependent fraction of camera distance; lengths add the "
                                 "scale uncertainty in quadrature",
            "point_evidence": ev}


def record(result: dict[str, Any], points, targets, actor: str, label: str = "", base_points=None) -> dict[str, Any]:
    return {"id": "M" + uuid.uuid4().hex[:6].upper(), **result, "points": points, "base_points": base_points,
            "targets": targets or [], "label": label, "actor": actor, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "active": True}


TRACK_KINDS = ("path_length", "track_distance_at_time", "track_speed_at_time")


def current_measurements(annotations: list[dict[str, Any]], scene: dict[str, Any] | None, frame_fp: str | None,
                         include_withdrawn: bool = True) -> list[dict[str, Any]]:
    """Analyst measurements re-evaluated in the current calibration (values change when the scale is
    calibrated), flagged when they were placed on an earlier reconstruction."""
    out = []
    for m in annotations:
        if not include_withdrawn and m.get("active", True) is False:
            continue
        if frame_fp and m.get("frame_fingerprint") and m["frame_fingerprint"] != frame_fp:
            m = {**m, "stale": "placed on an earlier reconstruction: the points may no longer lie on the same "
                               "features - re-check or re-measure"}
        if scene and m.get("base_points") and m.get("active", True) and m["kind"] not in TRACK_KINDS:
            k = scene["coordinate_system"]["global_scale_factor"] or 1.0
            pts = [[v * k for v in p] for p in m["base_points"]]
            try:
                cur = compute(scene, m["kind"], pts, m.get("point_evidence"), m.get("targets"))
                m = {**m, **{x: cur[x] for x in ("value", "sigma", "unit", "status", "display")}, "points": pts,
                     "value_at_creation": m.get("display")}
            except ValueError:
                pass
        out.append(m)
    return out
