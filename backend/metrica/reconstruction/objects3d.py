"""Semantic 3D objects: poles, static objects and parked vehicles, vegetation, unclassified
structures and the ground.

Each reconstructor produces per-view *observations*; observations of the same object from
different views are merged, and the spread between them becomes part of the uncertainty.
Dimensions always carry a ``basis``: ``measured`` (from observed pixels/points),
``estimated`` (single-view metrology) or ``prior`` (class-typical value used to complete an
unobserved part).
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..semantics.labels import CATEGORIES, GROUND_IDS, SCENE_ID
from .primitives import cluster_grid, min_area_rect
from .views import View

UP = np.array([0.0, 1.0, 0.0])

# Class priors for vehicle dimensions (metres): mean, SD. Sources: typical European/US fleet
# ranges; used only to complete dimensions that were not observed.
VEHICLE_PRIORS = {
    "car": {"length": (4.4, 0.45), "width": (1.80, 0.10), "height": (1.50, 0.15)},
    "truck": {"length": (6.5, 2.0), "width": (2.20, 0.25), "height": (2.70, 0.60)},
    "bus": {"length": (11.5, 1.5), "width": (2.55, 0.10), "height": (3.20, 0.30)},
    "motorcycle": {"length": (2.1, 0.2), "width": (0.80, 0.15), "height": (1.30, 0.15)},
    "bicycle": {"length": (1.75, 0.10), "width": (0.60, 0.10), "height": (1.10, 0.10)},
    "train": {"length": (20.0, 5.0), "width": (2.9, 0.2), "height": (3.8, 0.4)},
}
STATIC_DETECT = {"traffic_light", "traffic_sign", "fire_hydrant", "parking_meter", "bench", "potted_plant", "chair",
                 "couch", "table", "bed", "tv", "suitcase", "bag", "bottle"}


# ------------------------------------------------------------------------------ helpers
def _row_of_height(view: View, base: np.ndarray, H: float) -> float:
    uv, z = view.cam.project((base + H * UP)[None])
    return float(uv[0, 1]) if z[0] > 0 else np.nan


def _height_for_row(view: View, base: np.ndarray, row: float, hmax: float) -> float | None:
    """Height on the vertical through `base` whose projection is at image row `row`."""
    lo, hi = 0.0, hmax
    r_lo, r_hi = _row_of_height(view, base, lo), _row_of_height(view, base, hi)
    # a camera pitched down cannot see points high above a nearby base (behind the image plane):
    # shrink the search range to the heights that project in front of the camera
    for _ in range(30):
        if np.isfinite(r_hi) or hi < 1e-3:
            break
        hi *= 0.7
        r_hi = _row_of_height(view, base, hi)
    if not (np.isfinite(r_lo) and np.isfinite(r_hi)) or not (min(r_lo, r_hi) <= row <= max(r_lo, r_hi)):
        return None
    for _ in range(40):
        mid = (lo + hi) / 2
        r = _row_of_height(view, base, mid)
        if (r - row) * (r_lo - row) > 0:
            lo, r_lo = mid, r
        else:
            hi = mid
    return (lo + hi) / 2


def _ground_below(view: View, u: float, v: float) -> bool:
    h, w = view.z.shape
    ui = int(np.clip(u, 0, w - 1))
    rows = range(int(v) + 1, min(h, int(v) + 6))
    for r in rows:
        if view.plane_id is not None and view.plane_id[r, ui] == 0:
            return True
        if view.labels is not None and view.labels[r, ui] in GROUND_IDS:
            return True
        if np.isfinite(view.z[r, ui]):
            P = view.cam.backproject(np.array([[ui + 0.5, r + 0.5]]), np.array([view.z[r, ui]]))[0]
            if abs(P[1]) < 0.15 * max(1.0, abs(view.z[r, ui]) * 0.02):
                return True
    return False


# ------------------------------------------------------------------------------ poles
def semantic_pole_mask(view: View) -> np.ndarray | None:
    if view.labels is None:
        return None
    return view.labels == SCENE_ID["pole"]


def depth_strip_mask(view: View, half_widths=(5, 9, 14), ratio: float = 0.85) -> np.ndarray:
    """Thin vertical structures standing in front of their surroundings (geometry-only cue):
    pixels clearly nearer than the depth on *both* sides, bordered by strong vertical edges."""
    z = np.nan_to_num(view.z, nan=1e6).astype(np.float32)
    strip = np.zeros(z.shape, bool)
    for k in half_widths:
        left = np.roll(z, k, axis=1)
        right = np.roll(z, -k, axis=1)
        strip |= (z < ratio * left) & (z < ratio * right)
    strip[:, :max(half_widths)] = False
    strip[:, -max(half_widths):] = False
    gray = cv2.cvtColor(view.image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    vert_edge = (gx > 2.0 * gy + 10).astype(np.uint8)
    near_edge = cv2.dilate(vert_edge, np.ones((1, 7), np.uint8)) > 0
    strip &= near_edge | cv2.dilate(strip.astype(np.uint8), np.ones((1, 3), np.uint8)).astype(bool)
    strip &= np.isfinite(view.z)
    return strip


def _extend_down(view: View, rows: np.ndarray, centers: np.ndarray, widths: np.ndarray, y0: int) -> tuple[float, float]:
    """Follow a thin vertical structure below where its depth stops standing out (near its base the
    pole and the ground behind it are at almost the same depth). The axis is first re-centred on the
    pole's two image borders (monocular depth is blurred across thin objects), then followed down
    while both borders remain visible as vertical edges. Returns (bottom row, axis u at the bottom)."""
    gray = cv2.cvtColor(view.image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    h, w = gray.shape
    rr = y0 + rows.astype(float)
    k = np.polyfit(rr, centers, 1) if len(rr) >= 5 else np.array([0.0, float(centers[-1])])
    wmed = max(2.0, float(np.median(widths)))
    hw = wmed / 2
    # ---- re-centre on the two strongest border edges in the lowest rows of the strip
    r1 = int(rr.max())
    r0 = max(int(rr.min()), r1 - 25)
    uc0 = float(np.polyval(k, (r0 + r1) / 2))
    lo, hi = int(max(0, uc0 - 1.5 * wmed - 3)), int(min(w, uc0 + 1.5 * wmed + 4))
    if hi - lo > 4:
        prof = gx[r0:r1 + 1, lo:hi].mean(0)
        best = None
        for i in range(len(prof)):
            for j in range(i + 2, len(prof)):
                if 0.5 * wmed <= j - i <= 1.8 * wmed + 2:
                    sc = min(prof[i], prof[j])
                    if best is None or sc > best[0]:
                        best = (sc, i, j)
        band = gx[r0:r1 + 1, max(0, lo - 20):min(w, hi + 20)]
        if best is not None and best[0] > 1.5 * float(np.median(band)):
            k = np.array([k[0], k[1] + (lo + (best[1] + best[2]) / 2 + 0.5) - uc0])
            hw = (best[2] - best[1]) / 2
    # ---- follow both borders downwards (3-row averaged edge strength)
    r = r1 + 1
    last, misses = float(r1), 0
    while r < h - 1 and misses <= 3:
        uc = float(np.polyval(k, r))
        lcol, rcol = int(round(uc - hw)), int(round(uc + hw))
        if lcol - 3 < 0 or rcol + 3 >= w:
            break
        win = gx[r - 1:r + 2]
        local = float(np.median(win[:, max(0, lcol - 20):min(w, rcol + 20)]))
        thr = max(15.0, 1.4 * local)
        left = win[:, lcol - 2:lcol + 3].mean(0).max()
        right = win[:, rcol - 2:rcol + 3].mean(0).max()
        if left > thr and right > thr:
            last, misses = float(r), 0
        else:
            misses += 1
        r += 1
    return last + 1.0, float(np.polyval(k, last))


def _reject(debug: list | None, reason: str) -> None:
    if debug is not None:
        debug.append(reason)


def pole_observations(view: View, unit_scale: float = 1.0, mask: np.ndarray | None = None,
                      source: str = "semantic", debug: list | None = None) -> list[dict[str, Any]]:
    if mask is None:
        mask = semantic_pole_mask(view)
        if mask is None:
            return []
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8))  # join columns
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 1), np.uint8))                   # join rows
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 1), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    fx = view.cam.intr.fx
    out = []
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if h < 20 or area < 40 or h / max(w, 1) < 3.5:
            _reject(debug, "shape (size/aspect)")
            continue
        comp = lab[y:y + h, x:x + w] == k
        rows = np.nonzero(comp.any(1))[0]
        centers = np.array([x + np.nonzero(comp[r])[0].mean() + 0.5 for r in rows])
        widths = np.array([comp[r].sum() for r in rows], float)
        vb, vt = y + rows.max() + 1.0, y + rows.min()
        ub = centers[-1]
        if source == "depth":
            vb, ub = _extend_down(view, rows, centers, widths, y)
        Himg = view.z.shape[0]
        top_cut = y <= 1
        # geometry-only candidates: followed down to visible ground (evidence that the base is visible);
        # the axis position itself comes from the pole's own depth, which is more reliable than the
        # exact contact row (low contrast at the foot of a pole)
        contact = False
        if source == "depth" and _ground_below(view, ub, vb - 1):
            Bg, okg = view.cam.ground_intersect(np.array([[ub, vb]]))
            cm = np.zeros(view.z.shape, bool)
            cm[y:y + h, x:x + w] = comp
            pts, _ = view.points(cm)
            if okg[0] and len(pts) >= 10:
                zc_pts = float(np.median(view.cam.pose.to_cam(pts)[:, 2]))
                r0 = float(np.median(widths)) * zc_pts / fx / 2
                axis = np.median(pts[:, [0, 2]], 0)
                vd = axis - view.cam.center[[0, 2]]
                vd /= max(np.linalg.norm(vd), 1e-9)
                axis = axis + vd * r0
                Bd = np.array([axis[0], 0.0, axis[1]])
                rng_d = float(np.linalg.norm(Bd[[0, 2]] - view.cam.center[[0, 2]]))
                rng_c = float(np.linalg.norm(Bg[0][[0, 2]] - view.cam.center[[0, 2]]))
                if abs(rng_d - rng_c) < 0.25 * rng_c:  # contact row and depth agree: base visible
                    B = Bd
                    base_obs = contact = True
        if not contact:
            # axis position from the pole's own 3D points (robust to where the mask ends near the ground)
            cm = np.zeros(view.z.shape, bool)
            cm[y:y + h, x:x + w] = comp
            pts, _ = view.points(cm)
            if len(pts) >= 10:
                axis = np.median(pts[:, [0, 2]], 0)
                vd = axis - view.cam.center[[0, 2]]
                vd /= max(np.linalg.norm(vd), 1e-9)
                r0 = float(np.median(widths)) * float(np.median(view.cam.pose.to_cam(pts)[:, 2])) / fx / 2
                axis = axis + vd * r0  # visible surface is in front of the axis by one radius
                B = np.array([axis[0], 0.0, axis[1]])
                hb_try = _height_for_row(view, B, vb, 60.0 * unit_scale)
                base_obs = (hb_try is not None and hb_try < 0.3 * unit_scale) or (vb >= Himg - 1 and False)
                base_obs = base_obs and _ground_below(view, ub, vb - 1) if source == "depth" else \
                    (base_obs or _ground_below(view, ub, vb - 1))
            else:
                base_obs = _ground_below(view, ub, vb - 1)
                if not base_obs:
                    _reject(debug, "no base: not on visible ground")
                    continue
                Bg, ok = view.cam.ground_intersect(np.array([[ub, vb]]))
                if not ok[0]:
                    _reject(debug, "base ray misses ground")
                    continue
                B = Bg[0]
        hmax = 60.0 * unit_scale
        H = _height_for_row(view, B, vt, hmax)
        if H is not None and not top_cut:
            # extend upwards along the projected axis while the observed depth stays on the pole
            step = 0.05 * unit_scale
            hh = H
            misses = 0
            while hh < hmax:
                q = B + (hh + step) * UP
                uv, zc = view.cam.project(q[None])
                ui, vi = int(uv[0, 0]), int(uv[0, 1])
                if zc[0] <= 0 or vi < 0:
                    top_cut = vi < 0
                    break
                if not (0 <= ui < view.z.shape[1]):
                    break
                zz = view.z[vi, max(0, ui - 1):ui + 2]
                ok = np.isfinite(zz) & (np.abs(zz - zc[0]) < 0.12 * zc[0])
                if ok.any():
                    hh += step
                    misses = 0
                else:
                    misses += 1
                    hh += step
                    if misses > 3:
                        hh -= 4 * step
                        break
            if hh > H:
                H = hh
                vt = float(view.cam.project((B + H * UP)[None])[0][0, 1])
        if H is None or H < 0.4 * unit_scale:
            _reject(debug, "too short (<0.4) or height unknown")
            continue
        hb = 0.0 if base_obs else (_height_for_row(view, B, vb, hmax) or 0.0)
        # radius from pixel widths along the central part of the pole
        sel = np.arange(int(0.2 * len(rows)), max(int(0.8 * len(rows)), int(0.2 * len(rows)) + 1))
        hs, rs = [], []
        for i in sel:
            row = y + rows[i] + 0.5
            hh = _height_for_row(view, B, row, hmax)
            if hh is None:
                continue
            zc = view.cam.pose.to_cam((B + hh * UP)[None])[0, 2]
            hs.append(hh)
            rs.append(widths[i] * zc / fx / 2)
        if len(rs) < 3:
            _reject(debug, "too few width samples")
            continue
        hs, rs = np.array(hs), np.array(rs)
        r_med = float(np.median(rs))
        if len(hs) > 5 and np.ptp(hs) > 0.5 * unit_scale:
            kfit = np.polyfit(hs, rs, 1)
            r_bot, r_top = float(np.polyval(kfit, hb)), float(np.polyval(kfit, H))
            if not (0 < r_top <= r_bot * 1.05):  # only accept a plausible narrowing taper
                r_bot = r_top = r_med
        else:
            r_bot = r_top = r_med
        zc_mid = float(view.cam.pose.to_cam((B + 0.5 * (hb + H) * UP)[None])[0, 2])
        # depth consistency with the network's estimate (catches poles attached to walls etc.)
        zs = view.z[y:y + h, x:x + w][comp]
        zs = zs[np.isfinite(zs)]
        depth_rel = float(abs(np.median(zs) - zc_mid) / zc_mid) if len(zs) else None
        if source == "depth" and (depth_rel is None or depth_rel > 0.2 or not base_obs
                                  or np.median(widths) < 3 or r_med < 0.02 * unit_scale):
            # geometry-only candidates must agree with the depth they were found in, stand on visible ground
            # and be wide enough to measure; otherwise they are edges of walls or window frames
            _reject(debug, "depth candidate rejected (depth mismatch, base not visible, or too thin)")
            continue
        out.append({"view": view.key, "base": B, "height": float(H), "observed_bottom": float(hb), "source": source,
                    "top_cut": bool(top_cut),
                    "radius_bottom": max(r_bot, 0.005), "radius_top": max(r_top, 0.005), "radius_median": r_med,
                    "base_observed": bool(base_obs), "bbox_px": [int(x), int(y), int(x + w), int(y + h)],
                    "sigma_radius": float(0.5 * zc_mid / fx + 0.15 * r_med),
                    "sigma_height": float(abs(_height_for_row(view, B, vt + 2, hmax) or H) - H) + 0.02 * H,
                    "depth_consistency": depth_rel, "frame": view.frame, "video": view.video})
    return out


def dedupe_per_view(obs: list[dict[str, Any]], radius: float) -> list[dict[str, Any]]:
    """Two detectors may find the same pole in the same view: keep one (semantic preferred)."""
    out: list[dict[str, Any]] = []
    for o in sorted(obs, key=lambda d: (d.get("source") != "semantic", -d["height"])):
        if any(x["view"] == o["view"] and np.linalg.norm(np.asarray(x["base"])[[0, 2]] - np.asarray(o["base"])[[0, 2]])
               < radius for x in out):
            continue
        out.append(o)
    return out


def merge(obs: list[dict[str, Any]], radius: float) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for o in sorted(obs, key=lambda d: -d["height"] if "height" in d else 0):
        p = np.asarray(o["base"])[[0, 2]]
        for g in groups:
            q = np.median([np.asarray(x["base"])[[0, 2]] for x in g], 0)
            if np.linalg.norm(p - q) < radius and all(x["view"] != o["view"] for x in g):
                g.append(o)
                break
        else:
            groups.append([o])
    return groups


def pole_entities(obs: list[dict[str, Any]], unit_id: str, unit: str, unit_scale: float,
                  n_views_total: int) -> list[dict[str, Any]]:
    ents = []
    obs = dedupe_per_view(obs, 0.6 * unit_scale)
    for k, g in enumerate(merge(obs, 0.8 * unit_scale)):
        if n_views_total > 2 and len(g) < 2:
            continue  # multi-view unit: require confirmation by a second view
        if n_views_total <= 2 and all(o.get("source") == "depth" for o in g) and not any(o["base_observed"] for o in g):
            continue  # single view: a geometry-only candidate needs a visible base on the ground
        base = np.median([o["base"] for o in g], 0)
        full = [o["height"] for o in g if not o.get("top_cut")]
        top_seen = bool(full)
        H = float(np.median(full)) if full else float(max(o["height"] for o in g))
        rb = float(np.median([o["radius_bottom"] for o in g]))
        rt = float(np.median([o["radius_top"] for o in g]))
        hb = float(min(o["observed_bottom"] for o in g))
        spread_pos = float(np.median([np.linalg.norm(np.asarray(o["base"]) - base) for o in g])) if len(g) > 1 else None
        spread_h = float(np.std([o["height"] for o in g], ddof=1)) if len(g) > 1 else None
        sig_h = float(np.hypot(np.median([o["sigma_height"] for o in g]), spread_h or 0.0))
        sig_r = float(np.hypot(np.median([o["sigma_radius"] for o in g]), np.std([o["radius_median"] for o in g])))
        base_seen = any(o["base_observed"] for o in g)
        plausible = 0.015 * unit_scale <= rb <= 0.8 * unit_scale and 0.5 * unit_scale <= H <= 45 * unit_scale
        ents.append({
            "id": f"{unit_id}-pole-{k + 1}", "category": "pole",
            "geometry": {"type": "cylinder", "base": base.round(4).tolist(), "axis": [0, 1, 0], "height": round(H, 4),
                         "radius_bottom": round(rb, 4), "radius_top": round(rt, 4),
                         "observed_range": [round(hb, 3), round(H, 3)]},
            "dimensions": {
                "height": {"value": H, "sigma": sig_h, "unit": unit, "basis": "estimated" if top_seen else "lower bound",
                           "note": "from the image rows of base and top along the gravity direction" + (
                               "" if top_seen else "; the top is outside every frame, so this is a minimum")},
                "diameter": {"value": 2 * float(np.median([o["radius_median"] for o in g])), "sigma": 2 * sig_r,
                             "unit": unit, "basis": "estimated", "note": "from apparent width and distance"},
                "taper_ratio": {"value": round(rt / max(rb, 1e-6), 3), "basis": "estimated"}},
            "evidence": {"class": "FITTED_PRIMITIVE", "views": sorted({o["view"] for o in g}),
                         "observations": len(g), "confidence": float(np.clip(0.35 + 0.15 * len(g), 0.35, 0.9)),
                         "base_observed": base_seen, "position_spread": spread_pos},
            "inferred_parts": ([] if base_seen else [{"part": "base", "range": [0.0, round(hb, 3)],
                                "reason": "base hidden in every view; pole assumed to reach the ground"}]),
            "flags": ([] if plausible else ["implausible dimensions for a pole: check the semantic mask"])
            + (["base position inferred from depth (base not visible)"] if not base_seen else []),
            "provenance": {"method": "pole mask (" + "+".join(sorted({o.get("source", "semantic") for o in g})) +
                                     ") -> base on calibrated ground (or depth) -> vertical axis by gravity -> height "
                                     "from top row, radius from apparent width; merged across views",
                           "algorithm": "metrica.poles v1"},
            "source_observations": [{"view": o["view"], "video": o["video"], "frame": o["frame"],
                                     "bbox_px": o["bbox_px"]} for o in g],
        })
    return ents


# ------------------------------------------------------------------------------ static objects / vehicles
def _mask_for(view: View, det: dict[str, Any], seg) -> tuple[np.ndarray, str]:
    if seg is not None:
        try:
            m, _ = seg.mask_for_box(view.image, det["bbox"])
            if m.sum() > 20:
                return m, seg.key
        except Exception:  # noqa: BLE001 - fall back to the box core
            pass
    m = np.zeros(view.z.shape, bool)
    x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
    bw, bh = x2 - x1, y2 - y1
    m[max(0, y1 + bh // 6):y2 - bh // 10, max(0, x1 + bw // 6):x2 - bw // 6] = True
    return m, "box core"


def object_observation(view: View, det: dict[str, Any], seg, unit_scale: float) -> dict[str, Any] | None:
    mask, mask_method = _mask_for(view, det, seg)
    P, _ = view.points(mask, stride=1)
    if len(P) < 15:
        return None
    zc = view.cam.pose.to_cam(P)[:, 2]
    med = np.median(zc)
    P = P[np.abs(zc - med) < 0.25 * med]
    if len(P) < 15:
        return None
    rect = min_area_rect(P[:, [0, 2]])
    y0, y1 = float(np.percentile(P[:, 1], 2)), float(np.percentile(P[:, 1], 98))
    on_ground = y0 < 0.3 * unit_scale
    x1, _, x2, y2 = det["bbox"]
    gp, ok = view.cam.ground_intersect(np.array([[(x1 + x2) / 2, y2]]))
    return {"ground_point": gp[0] if ok[0] else None,"view": view.key, "video": view.video, "frame": view.frame, "category": det["category"],
            "label": det["label"], "conf": det["conf"], "bbox_px": [round(v, 1) for v in det["bbox"]],
            "base": np.array([rect["center"][0], 0.0, rect["center"][1]]), "rect": rect, "y_range": [y0, y1],
            "on_ground": on_ground, "points": len(P), "mask_method": mask_method,
            "camera_center": view.cam.center}


def _dedupe_labels(lst: list[dict[str, Any]], radius: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for o in sorted(lst, key=lambda d: -d["conf"]):
        if any(x["view"] == o["view"] and np.linalg.norm(x["base"][[0, 2]] - o["base"][[0, 2]]) < radius for x in out):
            continue
        out.append(o)
    return out


def complete_vehicle(o: dict[str, Any], unit_scale: float, metric: bool) -> dict[str, Any]:
    pri = VEHICLE_PRIORS.get(o["category"], VEHICLE_PRIORS["car"])
    rect = o["rect"]
    meas = {"length": rect["length"], "width": rect["width"], "height": o["y_range"][1] - min(0.0, o["y_range"][0])}
    implausible = any(meas[d] > (pri[d][0] + 3 * pri[d][1]) * unit_scale for d in ("length", "width", "height"))
    if implausible and o.get("ground_point") is not None:
        # depth too noisy to measure the footprint: ground contact point + class-typical dimensions
        gp = np.asarray(o["ground_point"])
        away = gp[[0, 2]] - np.asarray(o["camera_center"])[[0, 2]]
        away /= max(np.linalg.norm(away), 1e-9)
        c = gp[[0, 2]] + away * pri["width"][0] * unit_scale / 2
        yaw = float(np.arctan2(away[1], away[0]) + np.pi / 2)
        return {"center_xz": c, "yaw_rad": yaw, "dims": {d: pri[d][0] * unit_scale for d in pri},
                "basis": {d: "prior" for d in pri}, "measured": meas,
                "note": "Footprint from depth was implausible (depth noise at this distance); position from the "
                        "ground contact point, dimensions from class priors."}
    dims, basis = {}, {}
    for d in ("length", "width", "height"):
        mu = pri[d][0] * unit_scale  # unit_scale converts metres to the unit frame
        if meas[d] >= 0.85 * mu:
            dims[d], basis[d] = meas[d], "measured"
        else:
            dims[d], basis[d] = mu, "prior"  # unobserved part completed with the class-typical value
    # shift the footprint centre away from the camera for the completed dimensions
    c = np.array(rect["center"], float)
    yaw = rect["yaw_rad"]
    ax_l = np.array([np.cos(yaw), np.sin(yaw)])
    ax_w = np.array([-np.sin(yaw), np.cos(yaw)])
    away = c - np.asarray(o["camera_center"])[[0, 2]]
    for d, ax in (("length", ax_l), ("width", ax_w)):
        if basis[d] == "prior":
            s = np.sign(away @ ax) or 1.0
            c = c + ax * s * (dims[d] - meas[d]) / 2
    return {"center_xz": c, "yaw_rad": yaw, "dims": dims, "basis": basis, "measured": meas}


def object_entities(obs: list[dict[str, Any]], unit_id: str, unit: str, unit_scale: float, metric: bool,
                    n_views_total: int) -> list[dict[str, Any]]:
    ents = []
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for o in obs:
        key = "vehicle" if o["category"] in ("car", "truck", "bus", "train") else o["category"]
        by_cat.setdefault(key, []).append(o)
    counter: dict[str, int] = {}
    for key, lst in by_cat.items():
        rad = (2.5 if key == "vehicle" or key in VEHICLE_PRIORS else 0.8) * unit_scale
        # same object detected twice in one view under different labels (e.g. car + truck): keep the best
        lst = _dedupe_labels(lst, rad)
        for g in merge(lst, rad):
            if n_views_total > 2 and len(g) < 2:
                continue  # a single sighting in a moving sequence may be a moving object: not a static entity
            votes: dict[str, float] = {}
            for o in g:
                votes[o["category"]] = votes.get(o["category"], 0.0) + o["conf"]
            cat = max(votes, key=votes.get)
            is_vehicle = cat in VEHICLE_PRIORS
            counter[cat] = counter.get(cat, 0) + 1
            eid = f"{unit_id}-{cat}-{counter[cat]}"
            conf = float(np.clip(np.mean([o["conf"] for o in g]) * (0.7 + 0.1 * len(g)), 0.1, 0.95))
            src = [{"view": o["view"], "video": o["video"], "frame": o["frame"], "bbox_px": o["bbox_px"],
                    "detector_label": o["label"], "detector_conf": round(o["conf"], 3), "mask": o["mask_method"]}
                   for o in g]
            if is_vehicle:
                comp = [complete_vehicle(o, unit_scale, metric) for o in g]
                cxz = np.median([c["center_xz"] for c in comp], 0)
                yaw = float(np.angle(np.mean(np.exp(2j * np.array([c["yaw_rad"] for c in comp])))) / 2)
                dims = {d: float(np.median([c["dims"][d] for c in comp])) for d in ("length", "width", "height")}
                basis = {d: ("measured" if any(c["basis"][d] == "measured" for c in comp) else "prior")
                         for d in dims}
                pri = VEHICLE_PRIORS[cat]
                ents.append({
                    "id": eid, "category": cat,
                    "geometry": {"type": "vehicle_box", "center": [float(cxz[0]), 0.0, float(cxz[1])],
                                 "yaw_rad": yaw, "length": dims["length"], "width": dims["width"],
                                 "height": dims["height"], "template": "car" if cat == "car" else cat},
                    "dimensions": {d: {"value": dims[d], "unit": unit, "basis": basis[d],
                                       "sigma": (pri[d][1] * unit_scale if basis[d] == "prior" else 0.08 * dims[d])}
                                   for d in dims},
                    "evidence": {"class": "CLASS_PRIOR" if "prior" in basis.values() else "FITTED_PRIMITIVE",
                                 "views": sorted({o["view"] for o in g}), "observations": len(g), "confidence": conf},
                    "shape_note": "Displayed with a generic vehicle template scaled to the estimated dimensions; it "
                                  "is not a scan of the actual vehicle.",
                    "flags": ["heading ambiguous by 180 degrees for a stationary vehicle"],
                    "provenance": {"method": "detection -> instance mask -> 3D points from aligned depth -> minimum-area "
                                             "footprint; unobserved dimensions completed with class priors",
                                   "algorithm": "metrica.vehicles v1"},
                    "source_observations": src, "stationary": True})
            else:
                rects = [o["rect"] for o in g]
                cxz = np.median([r["center"] for r in rects], 0)
                L = float(np.median([r["length"] for r in rects]))
                W = float(np.median([r["width"] for r in rects]))
                y0 = float(np.median([o["y_range"][0] for o in g]))
                y1 = float(np.median([o["y_range"][1] for o in g]))
                ents.append({
                    "id": eid, "category": cat,
                    "geometry": {"type": "box", "center": [float(cxz[0]), (y0 + y1) / 2, float(cxz[1])],
                                 "size": [L, max(y1 - y0, 0.05 * unit_scale), W],
                                 "yaw_rad": float(np.median([r["yaw_rad"] for r in rects]))},
                    "dimensions": {"visible_width": {"value": L, "unit": unit, "basis": "measured"},
                                   "visible_depth": {"value": W, "unit": unit, "basis": "measured",
                                                     "note": "only the observed side(s); likely an underestimate"},
                                   "height_top": {"value": y1, "unit": unit, "basis": "measured"}},
                    "evidence": {"class": "FITTED_PRIMITIVE", "views": sorted({o["view"] for o in g}),
                                 "observations": len(g), "confidence": conf, "on_ground": any(o["on_ground"] for o in g)},
                    "provenance": {"method": "detection -> mask -> 3D points -> oriented bounding box of the observed "
                                             "points", "algorithm": "metrica.static_objects v1"},
                    "source_observations": src})
    return ents


# ------------------------------------------------------------------------------ surface clusters
def surface_cluster_entities(pos: np.ndarray, lab: np.ndarray, taken: np.ndarray, unit_id: str, unit: str,
                             unit_scale: float) -> list[dict[str, Any]]:
    ents = []
    for cat, ids, min_pts in (("vegetation", [SCENE_ID["vegetation"]], 60), ("unknown", [SCENE_ID["unknown"]], 120)):
        m = np.isin(lab, ids) & ~taken & (pos[:, 1] > 0.3 * unit_scale)
        idx = np.flatnonzero(m)
        if len(idx) < min_pts:
            continue
        for k, c in enumerate(cluster_grid(pos[idx], 0.5 * unit_scale, min_points=min_pts)):
            P = pos[idx[c]]
            lo, hi = np.percentile(P, 2, 0), np.percentile(P, 98, 0)
            if np.max(hi - lo) < 0.4 * unit_scale:
                continue
            ent = {"id": f"{unit_id}-{cat}-{k + 1}", "category": cat,
                   "geometry": {"type": "bbox", "min": lo.round(3).tolist(), "max": hi.round(3).tolist(),
                                "surface_vertices": int(len(P))},
                   "dimensions": {"height_top": {"value": float(hi[1]), "unit": unit, "basis": "measured"},
                                  "horizontal_extent": {"value": float(max(hi[0] - lo[0], hi[2] - lo[2])),
                                                        "unit": unit, "basis": "measured"}},
                   "evidence": {"class": "OBSERVED_SINGLE_VIEW", "confidence": 0.5},
                   "provenance": {"method": "connected cluster of labelled surface vertices", "algorithm":
                                  "metrica.surface_clusters v1"}}
            if cat == "unknown":
                ent["note"] = "Observed geometry that could not be classified; kept as surface only."
            ents.append(ent)
    return ents


# ------------------------------------------------------------------------------ ground
def ground_entity(unit_id: str, coverage: dict[str, Any], cov_grid: np.ndarray, views: list[View],
                  unit: str) -> dict[str, Any]:
    ext = coverage["extent"]
    cell = coverage["cell"]
    nz, nx = cov_grid.shape

    def polys(mask: np.ndarray, min_cells: int = 8) -> list[list[list[float]]]:
        out = []
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) < min_cells:
                continue
            c = cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2).astype(float)
            if len(c) >= 3:
                out.append([[ext[0] + (p[0] + 0.5) * cell, ext[2] + (p[1] + 0.5) * cell] for p in c])
        return out

    seen = cov_grid == 2
    occl = cov_grid == 1
    near = cv2.dilate(seen.astype(np.uint8), np.ones((int(max(3, 3.0 / cell)),) * 2, np.uint8)) > 0
    inferred = occl & near
    regions = {}
    # semantic ground classes projected onto the ground grid
    cls_votes = np.zeros((nz, nx, 4), np.int32)
    names = ["road", "sidewalk", "terrain", "ground"]
    for v in views:
        if v.labels is None:
            continue
        h, w = v.labels.shape
        vs, us = np.mgrid[0:h:4, 0:w:4]
        lab = v.labels[vs, us]
        gm = np.isin(lab, [SCENE_ID[n] for n in names])
        if not gm.any():
            continue
        P, ok = v.cam.ground_intersect(np.stack([us[gm] + 0.5, vs[gm] + 0.5], 1).astype(float))
        gx = ((P[ok, 0] - ext[0]) / cell).astype(int)
        gz = ((P[ok, 2] - ext[2]) / cell).astype(int)
        inb = (gx >= 0) & (gx < nx) & (gz >= 0) & (gz < nz)
        lut = np.full(256, -1, int)
        for k, n in enumerate(names):
            lut[SCENE_ID[n]] = k
        li = lut[lab[gm][ok][inb]]
        if len(li):
            np.add.at(cls_votes, (gz[inb], gx[inb], li), 1)
    if cls_votes.sum():
        best = cls_votes.argmax(2)
        has = cls_votes.sum(2) > 0
        for k, n in enumerate(names[:3]):
            m = has & (best == k) & seen
            if m.sum() >= 8:
                regions[n] = {"polygons_xz": polys(m), "area": float(m.sum() * cell * cell),
                              "use": {"road": "drivable (semantic)", "sidewalk": "walkable (semantic)",
                                      "terrain": "unpaved (semantic)"}[n]}
    return {"id": f"{unit_id}-ground", "category": "ground",
            "geometry": {"type": "ground_plane", "y": 0.0, "observed_polygons_xz": polys(seen),
                         "inferred_polygons_xz": polys(inferred, 4)},
            "regions": regions,
            "dimensions": {"observed_area": {"value": float(seen.sum() * cell * cell), "unit": f"{unit}^2",
                                             "basis": "measured"},
                           "inferred_area": {"value": float(inferred.sum() * cell * cell), "unit": f"{unit}^2",
                                             "basis": "inferred"}},
            "evidence": {"class": "FITTED_PRIMITIVE", "confidence": 0.8,
                         "note": "Plane fitted to walking-surface evidence; inferred patches are hidden ground next "
                                 "to observed ground, assumed to continue at the same level."},
            "provenance": {"method": "calibrated/fitted ground plane; extent from the per-camera visibility "
                                     "(coverage) map", "algorithm": "metrica.ground v1"}}


def category_display(cat: str) -> str:
    return CATEGORIES.get(cat, CATEGORIES["object"]).display
