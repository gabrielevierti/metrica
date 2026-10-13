"""Stage B - camera calibration, poses and scale for every video.

Static/rotating cameras: single-view calibration (vertical vanishing point + people as a
ruler; fallback: ground plane in relative depth). Moving cameras: COLMAP SfM, gravity/ground
alignment, localisation of analysis frames, scale from people statures or an assumed
camera height. Output: ``derived/geometry/<vid>.json`` and ``derived/geometry/units.json``.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ...geometry.calibration import ScaleRecord, person_height_scale_sigma
from ...geometry.camera import Camera, Intrinsics, Pose, look_rotation
from ...geometry.gravity import apply_similarity, gravity_align, person_heights, similarity
from ...geometry.rotation import map_point, register_homographies, rotations_from_homographies
from ...geometry.sfm import Localizer, interpolate_pose, run_sfm
from ...geometry.single_view import (
    calibrate_people,
    calibrate_vertical_vp,
    ground_plane_from_depth,
    manhattan_focal,
    person_samples,
    pitch_roll_from_up,
    vertical_segments,
    vp_strength,
)
from ...ingest.decode import FrameSource
from ...log import get_logger
from ...semantics.labels import GROUND_IDS
from ..base import Context, Stage, StageResult

log = get_logger("geometry")
ASSUMED_STATIC_CAMERA_HEIGHT = 3.0
ASSUMED_HFOV = 60.0


def _tracks(ctx: Context, vid: str) -> list[dict[str, Any]]:
    d = ctx.case.read_json(f"derived/videos/{vid}/tracks2d.json", {"tracks": []})
    return d["tracks"]


def _person_obs(tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for t in tracks:
        if t["category"] != "person":
            continue
        for o in t["observations"]:
            out.append({**o, "track_id": t["id"]})
    return out


def _depth_index(ctx: Context, vid: str) -> dict[str, Any]:
    return ctx.case.read_json(f"derived/videos/{vid}/depth_index.json", {"frames": []})


class GeometryStage(Stage):
    name = "geometry"
    title = "Camera calibration, poses and scale"
    depends = ("ingest", "tracking", "depth")
    config_keys = ("hfov_deg", "person_height_m", "person_height_sigma_m", "handheld_camera_height_m",
                   "sfm_max_features", "sfm_sequential_overlap", "localize_all_frames", "joint_sfm")
    version = "6"

    def run(self, ctx: Context) -> StageResult:
        res = StageResult()
        units: list[dict[str, Any]] = []
        moving, static = [], []
        for vid in ctx.video_ids():
            ing = ctx.case.read_json(f"derived/videos/{vid}/ingest.json")
            (moving if ing["mode"]["mode"] == "moving" else static).append(vid)
        n = len(moving) + len(static)
        done = 0
        for vid in static:
            ctx.progress(done / n, f"{vid}: single-view calibration")
            units.append(self._static(ctx, vid, res))
            done += 1
        if moving:
            groups = [moving] if ctx.config.joint_sfm else [[v] for v in moving]
            for g in groups:
                def sub(f, m, done=done, g=g):
                    ctx.progress((done + f * len(g)) / n, m)
                units.extend(self._moving(ctx, g, res, sub))
                done += len(g)
        p = ctx.case.write_json("derived/geometry/units.json", {"units": units})
        res.outputs.append(p)
        res.metrics["units"] = [{"id": u["id"], "kind": u["kind"], "videos": u["videos"],
                                 "scale": u["scale"]["status"]} for u in units]
        return res

    # ------------------------------------------------------------------ static
    def _static(self, ctx: Context, vid: str, res: StageResult) -> dict[str, Any]:
        cfg = ctx.config
        vdir = ctx.vdir(vid)
        ing = ctx.case.read_json(f"derived/videos/{vid}/ingest.json")
        img_path = vdir / (ing["background"] or ing["keyframes"][0]["file"])
        bg = cv2.imread(str(img_path))
        h, w = bg.shape[:2]
        tracks = _tracks(ctx, vid)
        reg = None
        obs = _person_obs(tracks)
        if ing["mode"]["mode"] == "rotating":
            reg = self._register_rotating(ctx, vid, bg, tracks)
            obs = _obs_to_reference(obs, reg)
        samples = person_samples(obs, w, h)
        verticals, vp = vertical_segments(bg)
        warnings: list[str] = []
        strength = vp_strength(vp, w, h)
        mf = None
        if not cfg.hfov_deg and vp is not None and len(verticals) >= 6:
            try:
                mf = manhattan_focal(bg, vp, verticals)
            except Exception as e:  # noqa: BLE001 - optional evidence
                log.warning(f"{vid}: focal from vanishing points failed: {e}")
        mf_ok = bool(mf and mf.get("f") and mf["sigma_rel"] < 0.12)
        focal_assumed = False

        def hfov_of(f: float) -> float:
            return float(np.degrees(2 * np.arctan(w / 2 / f)))

        if cfg.hfov_deg:
            hfov_fit, focal_policy, prof = cfg.hfov_deg, "configured field of view", None
        elif mf_ok:
            hfov_fit = mf["hfov_deg"]
            focal_policy = (f"estimated from orthogonal vanishing points: horizontal field of view "
                            f"{mf['hfov_deg']:.1f} deg (1 sigma {mf['sigma_rel']:.1%} of the focal length)")
            prof = [hfov_of(mf["f"] * (1 - mf["sigma_rel"])), hfov_of(mf["f"] * (1 + mf["sigma_rel"]))]
        elif vp is not None and len(verticals) >= 8 and strength < 4.0:
            hfov_fit, focal_policy, prof = None, "estimated (strong vertical vanishing point)", None
        else:
            focal_assumed = True
            hfov_fit, focal_policy, prof = ASSUMED_HFOV, (f"assumed {ASSUMED_HFOV:.0f} deg (vertical lines do not "
                                                          "constrain the focal length)"), [40.0, 50.0, 70.0, 85.0]
        calib = calibrate_people(samples, w, h, hfov_fit, cfg.person_height_m, verticals, hfov_profile=prof)
        if calib is not None:
            calib[1]["focal_policy"] = focal_policy
            calib[1]["vertical_vp_distance_diagonals"] = strength if np.isfinite(strength) else None
        orient_only = None
        if calib is not None:
            cam, rep = calib
            if rep["median_relative_error"] > 0.12 or rep["inlier_ratio_relative"] < 0.5:
                warnings.append(f"{vid}: people-based metric calibration rejected (median error "
                                f"{rep['median_relative_error']:.0%} of person height, "
                                f"{rep['inlier_ratio_relative']:.0%} consistent observations).")
                if rep["median_relative_error"] < 0.25:
                    orient_only = calib  # horizon/tilt still informative, scale not trusted
                calib = None
        if calib is not None:
            cam, rep = calib
            sigma = person_height_scale_sigma(rep["people"], cfg.person_height_sigma_m, cfg.person_height_m)
            if rep.get("scale_sigma_rel_focal"):
                sigma = float(np.hypot(sigma, rep["scale_sigma_rel_focal"]))
            if rep.get("focal_at_search_bound") or focal_assumed:
                warnings.append(f"{vid}: field of view not identifiable from the footage ({focal_policy}); scale "
                                "uncertainty includes the focal-length ambiguity. Supply the camera's field of view or "
                                "a known distance to tighten it.")
            scale = ScaleRecord("ESTIMATED_METRIC", "m", "statures of observed people (single-view metrology)",
                                sigma, True,
                                assumptions=[f"Mean stature of observed standing adults = {cfg.person_height_m:.2f} m "
                                             f"(population SD {cfg.person_height_sigma_m:.2f} m).",
                                             "People counted were standing upright and fully visible.",
                                             "The ground is locally planar where people walked."]
                                + [f"Focal length: {focal_policy}."],
                                details={"people": rep["people"], "samples": rep["samples"],
                                         "sigma_components": {"stature_prior": person_height_scale_sigma(
                                             rep["people"], cfg.person_height_sigma_m, cfg.person_height_m),
                                             "focal_ambiguity": rep.get("scale_sigma_rel_focal")}})
            method = "single_view_people"
        else:
            hfov = cfg.hfov_deg or (mf["hfov_deg"] if mf_ok else ASSUMED_HFOV)
            src = "configured" if cfg.hfov_deg else ("orthogonal vanishing points" if mf_ok else "assumed")
            intr = Intrinsics.from_hfov(w, h, hfov, source={"configured": "user", "assumed": "assumed"}.get(
                src, "estimated_vanishing_points"))
            rep = {"hfov_deg": hfov, "hfov_source": src}
            pr = None
            cam_h = ASSUMED_STATIC_CAMERA_HEIGHT
            if orient_only is not None:
                oc, orep = orient_only
                pr = (np.radians(orep["pitch_deg"]), np.radians(orep["roll_deg"]))
                cam_h = float(orep["camera_height_m"])
                intr = oc.intr
                rep["orientation_source"] = ("people-based fit (orientation and camera height only; model fit too "
                                             "poor for a metric scale)")
                rep["people_fit"] = {k: orep[k] for k in ("median_reprojection_px", "median_relative_error",
                                                          "people", "samples")}
            if pr is None and vp is not None and len(verticals) >= 8 and strength < 4.0:
                pr = calibrate_vertical_vp(vp, w, h, hfov)
                rep["orientation_source"] = f"vertical vanishing point ({len(verticals)} line segments)"
            if pr is None:
                di = _depth_index(ctx, vid)
                rec = next((f for f in di["frames"] if f["depth"]), None)
                if rec is not None:
                    inv = np.load(vdir / rec["depth"])["inv_depth"].astype(np.float32)
                    mask = None
                    if rec.get("semantic"):
                        lab = cv2.imread(str(vdir / rec["semantic"]), cv2.IMREAD_UNCHANGED)
                        mask = np.isin(lab, list(GROUND_IDS))
                        if mask.mean() < 0.05:
                            mask = None
                    g = ground_plane_from_depth(inv, intr, mask)
                    if g is not None:
                        pr = pitch_roll_from_up(g[0])
                        rep["orientation_source"] = "ground plane in relative depth"
                        rep["ground_inlier_ratio"] = g[2]
            if pr is None:
                pr = (np.radians(20.0), 0.0)
                rep["orientation_source"] = "assumed (20 deg downward tilt) - no usable evidence"
                warnings.append(f"{vid}: camera orientation could not be estimated; a default tilt is assumed.")
            cam = Camera(intr, Pose(look_rotation(pr[0], pr[1]), np.array([0.0, cam_h, 0.0])))
            rep.update({"method": "single view without metric reference", "pitch_deg": float(np.degrees(pr[0])),
                        "roll_deg": float(np.degrees(pr[1])), "camera_height_su": cam_h})
            basis = ("camera height from a people-based fit that is not accurate enough for metric use"
                     if orient_only is not None else f"assumed camera mounting height {cam_h:.1f} su")
            scale = ScaleRecord("RELATIVE", "su", basis, None, True,
                                assumptions=[f"Camera {cam_h:.2f} scene units above the ground "
                                             "(1 su = 1 m only if this is true).",
                                             f"Horizontal field of view {intr.hfov_deg:.0f} deg ({src})."])
            method = "single_view_relative"
            res.fallback(f"{vid} scale", "RELATIVE scale (scene units)",
                         "No reliable metric reference in the footage. Distances are in scene units; calibrate with "
                         "a known distance in the viewer to obtain metres.")
        rep["vertical_vanishing_point"] = vp.tolist() if vp is not None else None
        rep["focal_from_vanishing_points"] = mf  # evidence record, also when it was not used
        poses = rotation = None
        if reg is not None:
            poses, rotation = _rotation_poses(ctx, vid, reg, cam)
            if rotation["stats"]["registered"] == 0:
                warnings.append(f"{vid}: no frame of the rotating camera could be oriented; subjects are placed "
                                "only in the representative view.")
            elif rotation["stats"]["without_pose"]:
                warnings.append(f"{vid}: {rotation['stats']['without_pose']} analysed frame(s) of the rotating camera "
                                "have no orientation (no overlap, zoom change or camera movement); observations in "
                                "those frames have no 3D position.")
        for wmsg in warnings:
            res.warn(wmsg)
        unit_id = f"u_{vid}"
        doc = {"video_id": vid, "unit_id": unit_id, "mode": method, "capture": ing["mode"]["mode"],
               "intrinsics": cam.intr.to_json(), "static_pose": cam.pose.to_json(), "calibration": rep,
               "scale": scale.to_json(), "ground": {"normal": [0, 1, 0], "y": 0.0, "method": "calibrated plane"},
               "reference_image": str(img_path.relative_to(ctx.case.root)), "warnings": warnings}
        if poses is not None:
            doc["poses"] = poses
            doc["rotation"] = rotation
        p = ctx.case.write_json(f"derived/geometry/{vid}.json", doc)
        res.outputs.append(p)
        return {"id": unit_id, "kind": "single_view", "videos": [vid], "scale": scale.to_json(),
                "points": None}

    def _register_rotating(self, ctx: Context, vid: str, ref: np.ndarray,
                           tracks: list[dict[str, Any]]) -> dict[str, Any]:
        """Homographies of the analysed frames onto the representative frame (rotating cameras)."""
        frames = sorted({o["frame"] for t in tracks for o in t["observations"]})
        times = np.load(ctx.vdir(vid) / "frame_times.npy")
        # plus a regular sampling (2 per second) so the camera's orientation is known along the video
        step = max(1, int(round(len(times) / max(times[-1] - times[0], 1e-3) / 2))) if len(times) > 1 else 1
        frames = sorted(set(frames) | set(range(0, len(times), step)))
        if len(frames) > 1500:
            frames = [frames[i] for i in np.linspace(0, len(frames) - 1, 1500).round().astype(int)]
        src = FrameSource(ctx.case.video_path(vid))
        it = ((i, img) for i, _, img in src.iter_bgr(want=set(frames), stop_after=max(frames)))
        return register_homographies(it, ref, cancelled=ctx._cancelled)

    # ------------------------------------------------------------------ moving
    def _moving(self, ctx: Context, vids: list[str], res: StageResult, progress) -> list[dict[str, Any]]:
        cfg = ctx.config
        unit_id = "u_" + "_".join(vids)
        groups = {}
        for vid in vids:
            ing = ctx.case.read_json(f"derived/videos/{vid}/ingest.json")
            groups[vid] = [(k["i"], ctx.vdir(vid) / k["file"]) for k in ing["keyframes"]]
        work = ctx.case.stage_dir("geometry", "sfm", unit_id)
        sfm = run_sfm(work, groups, cfg.sfm_max_features, cfg.sfm_sequential_overlap, cfg.hfov_deg,
                      cancelled=lambda: ctx._cancelled(), progress=lambda f, m: progress(0.5 * f, m))
        for wmsg in sfm.warnings:
            res.warn(wmsg)
        ctx.check_cancel()
        if not sfm.ok or len(sfm.poses) < 3:
            res.fallback(f"SfM for {', '.join(vids)}", "single-view geometry on one keyframe",
                         "Structure from motion failed; moving-camera footage is reduced to a single view. "
                         "Object positions over time are not reconstructed in 3D for these videos.")
            out = []
            for vid in vids:
                out.append(self._static(ctx, vid, res))
            return out
        res.models.append("colmap-sfm")
        # ---- semantic ground labels for sparse points
        ground_mask = self._ground_point_mask(ctx, sfm)
        poses_all = list(sfm.poses.values())
        S, grav = gravity_align(sfm.points, ground_mask, poses_all, cfg.handheld_camera_height_m)
        # ---- localise analysis frames
        progress(0.55, "localising analysis frames")
        frame_poses: dict[str, dict[int, tuple[Pose, str, dict]]] = {}
        for vid in vids:
            frame_poses[vid] = self._localize(ctx, vid, sfm, work, res, lambda f, m: progress(0.55 + 0.35 * f / len(vids), m))
        # ---- apply gravity similarity
        pts = apply_similarity(S, sfm.points)
        for vid in vids:
            frame_poses[vid] = {f: (p.transformed(S), src, info) for f, (p, src, info) in frame_poses[vid].items()}
        # ---- scale
        progress(0.92, "estimating scale")
        heights: dict[str, list[float]] = {}
        for vid in vids:
            pd = {f: p for f, (p, src, _) in frame_poses[vid].items()}
            obs = _person_obs(_tracks(ctx, vid))
            for tid, hs in person_heights(obs, pd, sfm.intrinsics[vid]).items():
                heights[tid] = hs
        track_med = [float(np.median(v)) for v in heights.values() if len(v) >= 5]
        if len(track_med) >= 3:
            med = float(np.median(track_med))
            s = cfg.person_height_m / med
            spread = float(np.std(np.array(track_med) / med, ddof=1) / np.sqrt(len(track_med)))
            sigma = float(np.hypot(person_height_scale_sigma(len(track_med), cfg.person_height_sigma_m,
                                                             cfg.person_height_m), spread))
            scale = ScaleRecord("ESTIMATED_METRIC", "m", "statures of observed people in localised frames", sigma,
                                bool(grav.get("ground_found")),
                                assumptions=[f"Mean stature of observed standing adults = {cfg.person_height_m:.2f} m.",
                                             "Ground plane fitted to sparse points is the walking surface."],
                                details={"people": len(track_med), "median_height_su": med})
        else:
            cam_h = grav["camera_heights_su"]["median"] if grav.get("ground_found") else cfg.handheld_camera_height_m
            s = cfg.handheld_camera_height_m / max(cam_h, 1e-9)
            scale = ScaleRecord("RELATIVE", "su", "assumed hand-held camera height", None,
                                bool(grav.get("ground_found")),
                                assumptions=[f"Median camera height above ground = {cfg.handheld_camera_height_m:.2f} su "
                                             "(1 su = 1 m only if true)."],
                                details={"people_with_heights": len(track_med)})
            res.fallback(f"scale for {', '.join(vids)}", "assumed camera height (RELATIVE)",
                         "Not enough fully visible standing people to estimate metric scale.")
        if not grav.get("ground_found"):
            res.warn(f"{unit_id}: no ground plane found in sparse points; vertical alignment relies on camera "
                     "orientation only.")
        Ss = similarity(np.eye(3), np.zeros(3), s)
        pts = apply_similarity(Ss, pts)
        np.savez_compressed(work / "sparse_points.npz", xyz=pts.astype(np.float32), rgb=sfm.colors,
                            error=sfm.errors.astype(np.float32), track_len=sfm.track_len.astype(np.int16),
                            ids=sfm.point_ids, ground=ground_mask if ground_mask is not None else np.zeros(len(pts), bool),
                            **{f"obs_{v}_{f}": o for (v, f), o in sfm.observations.items()})
        S_total = Ss @ S
        out_units = []
        for vid in vids:
            fp = {f: (p.transformed(Ss), src, info) for f, (p, src, info) in frame_poses[vid].items()}
            frames = sorted(fp)
            intr = sfm.intrinsics.get(vid)
            if intr is None:
                res.warn(f"{vid}: not registered in the reconstruction; no poses.")
                continue
            doc = {
                "video_id": vid, "unit_id": unit_id, "mode": "sfm", "capture": "moving",
                "intrinsics": intr.to_json(),
                "poses": {"frames": frames,
                          "world_from_cam": [fp[f][0].matrix().round(6).tolist() for f in frames],
                          "source": [fp[f][1] for f in frames],
                          "inliers": [fp[f][2].get("inliers") for f in frames]},
                "pose_sources": {k: sum(1 for f in frames if fp[f][1] == k) for k in ("keyframe", "localized",
                                                                                        "interpolated")},
                "sfm": sfm.stats, "gravity": grav, "scale": scale.to_json(),
                "sfm_to_local": S_total.tolist(),
                "ground": {"normal": [0, 1, 0], "y": 0.0, "method": grav.get("ground_source", "assumed")},
                "warnings": sfm.warnings,
            }
            p = ctx.case.write_json(f"derived/geometry/{vid}.json", doc)
            res.outputs.append(p)
        res.outputs.append(work / "sparse_points.npz")
        res.metrics[unit_id] = {"registered": sfm.stats["registered"], "points": sfm.stats["points3D"],
                                "reprojection_px": round(sfm.stats["mean_reprojection_error_px"], 3),
                                "scale": scale.status}
        out_units.append({"id": unit_id, "kind": "sfm", "videos": vids, "scale": scale.to_json(),
                          "points": str((work / "sparse_points.npz").relative_to(ctx.case.root)),
                          "sfm": sfm.stats, "gravity": grav})
        return out_units

    def _ground_point_mask(self, ctx: Context, sfm) -> np.ndarray | None:
        votes = np.zeros(len(sfm.points))
        total = np.zeros(len(sfm.points))
        any_sem = False
        for (vid, frame), pose in sfm.poses.items():
            di = _depth_index(ctx, vid)
            rec = next((f for f in di["frames"] if f["i"] == frame and f.get("semantic")), None)
            if rec is None:
                continue
            lab = cv2.imread(str(ctx.vdir(vid) / rec["semantic"]), cv2.IMREAD_UNCHANGED)
            any_sem = True
            cam = Camera(sfm.intrinsics[vid], pose)
            uv, z = cam.project(sfm.points)
            h, w = lab.shape
            ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
            l = lab[uv[ok, 1].astype(int), uv[ok, 0].astype(int)]
            votes[ok] += np.isin(l, list(GROUND_IDS))
            total[ok] += 1
        if not any_sem:
            return None
        return (total > 0) & (votes / np.maximum(total, 1) > 0.5)

    def _localize(self, ctx: Context, vid: str, sfm, work, res: StageResult, progress):
        cfg = ctx.config
        known = {f: p for (v, f), p in sfm.poses.items() if v == vid}
        out: dict[int, tuple[Pose, str, dict]] = {f: (p, "keyframe", {}) for f, p in known.items()}
        tr = ctx.case.read_json(f"derived/videos/{vid}/tracks2d.json", {"tracks": []})
        needed = sorted({o["frame"] for t in tr["tracks"] for o in t["observations"]} - set(known))
        if not known:
            return out
        to_localize: list[int] = []
        if cfg.localize_all_frames and needed:
            step = max(1, int(np.ceil(len(needed) / 600)))
            to_localize = needed[::step]
        if to_localize:
            loc = Localizer(sfm.model_dir, work / "database.db", sfm, vid, cfg.sfm_max_features)
            want = set(to_localize)
            src = FrameSource(ctx.case.video_path(vid))
            ok_n = 0
            for k, (i, _t, img) in enumerate(src.iter_bgr(want=want, stop_after=max(want))):
                if k % 20 == 0:
                    ctx.check_cancel()
                    progress(k / len(want), f"{vid}: localising frame {k + 1}/{len(want)}")
                g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                r = loc.localize(i, g)
                if r is not None:
                    pose, info = r
                    # sanity: must be close to interpolation between neighbouring keyframes
                    ip = interpolate_pose(i, known)
                    if ip is not None:
                        span = np.linalg.norm(np.ptp(np.array([p.C for p in known.values()]), 0)) + 1e-9
                        if np.linalg.norm(ip.C - pose.C) > 0.25 * span:
                            continue
                    out[i] = (pose, "localized", info)
                    ok_n += 1
            res.metrics.setdefault("localization", {})[vid] = {"attempted": len(want), "localized": ok_n}
            if ok_n < 0.5 * len(want):
                res.warn(f"{vid}: only {ok_n}/{len(want)} frames localised by 2D-3D matching; others interpolated.")
        anchors = {f: p for f, (p, src, _) in out.items()}
        for f in needed:
            if f in out:
                continue
            ip = interpolate_pose(f, anchors)
            if ip is not None:
                out[f] = (ip, "interpolated", {})
        return out


def _obs_to_reference(obs: list[dict[str, Any]], reg: dict[str, Any]) -> list[dict[str, Any]]:
    """Person observations of a rotating camera mapped into the representative frame (exact for a pure
    rotation, whatever the depth); observations in unregistered frames are dropped."""
    out = []
    for o in obs:
        r = reg["frames"].get(o["frame"])
        if r is None:
            continue
        x1, y1, x2, y2 = o["bbox"]
        u = (x1 + x2) / 2
        (fu, fv), (hu, hv) = map_point(r["H"], np.array([[u, y2], [u, y1]]))
        bw = x2 - x1
        out.append({**o, "bbox": [fu - bw / 2, hv, fu + bw / 2, fv]})
    return out


def _rotation_poses(ctx: Context, vid: str, reg: dict[str, Any], cam: Camera) -> tuple[dict[str, Any], dict[str, Any]]:
    rot = rotations_from_homographies(reg, cam.intr.K)
    times = np.load(ctx.vdir(vid) / "frame_times.npy")
    known = {f: Pose(cam.pose.R_wc @ r["R_rel"], cam.pose.C.copy()) for f, r in rot["frames"].items()}
    source = {f: f"rotation_{r['source']}" for f, r in rot["frames"].items()}
    # short gaps between oriented frames: interpolated (flagged), never extrapolated
    for f in sorted(rot["rejected"]):
        prev = max((k for k in known if k < f and source[k] != "interpolated"), default=None)
        nxt = min((k for k in known if k > f and source[k] != "interpolated"), default=None)
        if prev is not None and nxt is not None and times[nxt] - times[prev] <= 1.0:
            p = interpolate_pose(f, {prev: known[prev], nxt: known[nxt]})
            if p is not None:
                known[f] = p
                source[f] = "interpolated"
    frames = sorted(known)
    poses = {"frames": frames, "world_from_cam": [known[f].matrix().round(6).tolist() for f in frames],
             "source": [source[f] for f in frames]}
    rotation = {"stats": {**rot["stats"], "interpolated": sum(1 for f in frames if source[f] == "interpolated")},
                "method": "frame-to-reference homographies (SIFT, RANSAC) decomposed as pure rotations "
                          "K^-1 H K; chained through neighbouring frames when a frame does not overlap the reference",
                "rejected_examples": {str(k): v for k, v in list(rot["rejected"].items())[:20]}}
    return poses, rotation
