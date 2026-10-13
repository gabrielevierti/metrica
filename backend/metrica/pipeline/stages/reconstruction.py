"""Stage C (part 2) - observed geometry: dense surfaces, sparse points, ground coverage."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ...geometry.camera import Camera, Intrinsics, Pose
from ...log import get_logger
from ...reconstruction.coverage import coverage_map
from ...reconstruction.fusion import (
    KeyframeDepth,
    align_depth,
    consistency,
    extract_mesh,
    ground_constraints,
    label_vertices,
    projected_constraints,
    tsdf_fuse,
)
from ...reconstruction.grid_mesh import grid_mesh
from ...reconstruction.layers import EVI, compact, write_layer
from ...reconstruction.single_view import single_view_surface
from ...semantics.labels import GROUND_IDS
from ..base import Context, Stage, StageResult

log = get_logger("reconstruction")
FUSION_MAX_WIDTH = 960


def save_coverage_png(cov: dict[str, Any], path) -> None:
    g = cov["grid"]
    rgba = np.zeros(g.shape + (4,), np.uint8)
    rgba[g == 2] = (80, 200, 120, 90)     # seen
    rgba[g == 1] = (240, 170, 60, 110)    # inside view, occluded
    rgba[g == 0] = (120, 120, 140, 0)     # never in view
    cv2.imwrite(str(path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))


def _cam(geo: dict[str, Any], pose_json: dict[str, Any] | None = None) -> Camera:
    intr = Intrinsics.from_json(geo["intrinsics"])
    return Camera(intr, Pose.from_json(pose_json or geo["static_pose"]))


class ReconstructionStage(Stage):
    name = "reconstruction"
    title = "Dense surface reconstruction"
    depends = ("geometry", "depth", "tracking")
    config_keys = ("tsdf_voxel_frac", "tsdf_max_voxels", "depth_consistency_rel")
    version = "3"

    def run(self, ctx: Context) -> StageResult:
        res = StageResult()
        units = ctx.case.read_json("derived/geometry/units.json")["units"]
        for k, u in enumerate(units):
            ctx.check_cancel()
            ctx.progress(k / len(units), f"unit {u['id']}")
            try:
                if u["kind"] == "single_view":
                    info = self._single(ctx, u, res)
                else:
                    info = self._sfm(ctx, u, res, lambda f, m, k=k: ctx.progress((k + f) / len(units), m))
            except Exception as e:  # one failed unit must not hide the others
                res.fallback(f"reconstruction of {u['id']}", "none", f"{type(e).__name__}: {e}")
                info = {"unit": u["id"], "kind": u["kind"], "layers": [], "error": str(e)}
            p = ctx.case.write_json(f"derived/reconstruction/{u['id']}/reconstruction.json", info)
            res.outputs.append(p)
            res.metrics[u["id"]] = {l["name"]: l["stats"].get("vertices") for l in info.get("layers", [])}
        return res

    # ------------------------------------------------------------------ single view
    def _single(self, ctx: Context, unit: dict[str, Any], res: StageResult) -> dict[str, Any]:
        vid = unit["videos"][0]
        geo = ctx.case.read_json(f"derived/geometry/{vid}.json")
        di = ctx.case.read_json(f"derived/videos/{vid}/depth_index.json")
        rec = next((f for f in di["frames"] if f["depth"]), None)
        if rec is None:
            raise RuntimeError("no depth map available")
        vdir = ctx.vdir(vid)
        img = cv2.imread(str(vdir / rec["image"]))
        inv = np.load(vdir / rec["depth"])["inv_depth"].astype(np.float32)
        sem = cv2.imread(str(vdir / rec["semantic"]), cv2.IMREAD_UNCHANGED) if rec.get("semantic") else None
        cam = _cam(geo)
        if (cam.intr.width, cam.intr.height) != (img.shape[1], img.shape[0]):
            cam = Camera(cam.intr.scaled(img.shape[1] / cam.intr.width), cam.pose)
        tr = ctx.case.read_json(f"derived/videos/{vid}/tracks2d.json", {"tracks": []})["tracks"]
        walk = np.array([[(o["bbox"][0] + o["bbox"][2]) / 2, o["bbox"][3]] for t in tr if t["category"] == "person"
                         for o in t["observations"] if o["conf"] > 0.5], float).reshape(-1, 2)
        surf = single_view_surface(img, inv, cam, walk, sem, smooth=ctx.config.profile != "fast")
        if surf["report"].get("semantics_ignored"):
            res.fallback(f"{vid} scene semantics", "geometry-based ground/wall labels",
                         "The segmentation model's labels contradict the calibrated geometry ("
                         + surf["report"]["semantics_ignored"] + ").")
        z = surf["z"]
        h, w = z.shape
        stride = max(2, int(np.ceil(np.sqrt(h * w / 250_000))))
        P, tris, pix, vcos = grid_mesh(z, cam, stride=stride, plane_id=surf["plane_id"])
        out = ctx.case.stage_dir("reconstruction", unit["id"])
        layers = []
        if len(tris):
            col = img[pix[:, 1], pix[:, 0]][:, ::-1]
            lab = surf["labels"][pix[:, 1], pix[:, 0]]
            zpix = z[pix[:, 1], pix[:, 0]]
            planar = surf["plane_id"][pix[:, 1], pix[:, 0]] >= 0
            cosq = np.where(planar, 1.0, np.clip(vcos / 0.5, 0, 1))
            conf = (0.3 + 0.7 * cosq) * (1 - 0.6 * np.clip(np.nan_to_num(zpix) / surf["max_depth"], 0, 1))
            evi = np.full(len(P), EVI["OBSERVED_SINGLE_VIEW"], np.uint8)
            pos, idx, col, lab, conf, evi = compact(P, tris, col, lab, conf, evi)
            layers.append({**write_layer(out, "surface", pos, col, idx, evi, (conf * 255).astype(np.uint8), lab),
                           "kind": "mesh", "evidence_summary": "OBSERVED_SINGLE_VIEW",
                           "method": "monocular depth aligned to calibrated ground plane; planar regions snapped"})
        # supplementary point layer
        vs, us = np.mgrid[0:h:4, 0:w:4]
        zz = z[vs, us]
        ok = np.isfinite(zz)
        uv = np.stack([us[ok] + 0.5, vs[ok] + 0.5], 1).astype(float)
        pts = cam.backproject(uv, zz[ok])
        layers.append({**write_layer(out, "points", pts, img[vs[ok], us[ok]][:, ::-1], None,
                                     np.full(len(pts), EVI["OBSERVED_SINGLE_VIEW"], np.uint8),
                                     np.full(len(pts), 160, np.uint8), surf["labels"][vs[ok], us[ok]]),
                       "kind": "points", "method": "back-projected aligned monocular depth"})
        # coverage of the ground
        gm = surf["ground_mask"]
        gv, gu = np.nonzero(gm[::4, ::4])
        if len(gv):
            gp = cam.backproject(np.stack([gu * 4 + 0.5, gv * 4 + 0.5], 1).astype(float), z[gv * 4, gu * 4])
            x0, x1 = np.percentile(gp[:, 0], [1, 99])
            z0, z1 = np.percentile(gp[:, 2], [1, 99])
        else:
            x0, x1, z0, z1 = -20, 20, -40, 0
        pad = 0.15 * max(x1 - x0, z1 - z0)
        ext = (float(x0 - pad), float(x1 + pad), float(z0 - pad), float(z1 + pad))
        cov = coverage_map([(cam, z)], ext, cell=max(0.1, (ext[1] - ext[0]) / 400), max_range=surf["max_depth"])
        save_coverage_png(cov, out / "coverage.png")
        np.save(out / "coverage_grid.npy", cov["grid"])
        np.savez_compressed(out / f"depth_{vid}.npz", z=z.astype(np.float16), labels=surf["labels"],
                            plane_id=surf["plane_id"].astype(np.int16))
        res.outputs += [out / "coverage.png", out / f"depth_{vid}.npz"]
        for l in layers:
            res.outputs += [out / f for f in l["files"].values()]
        log.info(f"{unit['id']}: single-view surface with {layers[0]['stats']['vertices'] if layers else 0} vertices, "
                 f"{len(surf['planes'])} vertical planes, ground coverage {cov['seen_fraction']:.0%}")
        return {"unit": unit["id"], "kind": "single_view", "videos": [vid], "layers": layers,
                "planes": surf["planes"], "report": surf["report"],
                "coverage": {"image": "coverage.png", "extent": cov["extent"], "cell": cov["cell"],
                             "seen_fraction": cov["seen_fraction"], "occluded_fraction": cov["occluded_fraction"]},
                "depth_maps": {vid: f"depth_{vid}.npz"}, "reference_size": [w, h]}

    # ------------------------------------------------------------------ multi view
    def _sfm(self, ctx: Context, unit: dict[str, Any], res: StageResult, progress) -> dict[str, Any]:
        cfg = ctx.config
        sp = np.load(ctx.case.root / unit["points"])
        xyz = sp["xyz"].astype(float)
        out = ctx.case.stage_dir("reconstruction", unit["id"])
        kdir = ctx.case.stage_dir("reconstruction", unit["id"], "kf_depth")
        kfs: list[KeyframeDepth] = []
        pending: list[tuple] = []
        fits = []
        ground_ok = bool(unit.get("gravity", {}).get("ground_found"))
        for vid in unit["videos"]:
            geo = ctx.case.read_json(f"derived/geometry/{vid}.json")
            if geo is None or geo.get("mode") != "sfm":
                continue
            di = ctx.case.read_json(f"derived/videos/{vid}/depth_index.json")
            frames = geo["poses"]["frames"]
            src = dict(zip(frames, geo["poses"]["source"]))
            mats = dict(zip(frames, geo["poses"]["world_from_cam"]))
            intr = Intrinsics.from_json(geo["intrinsics"])
            for rec in di["frames"]:
                if rec["i"] is None or rec["depth"] is None or src.get(rec["i"]) != "keyframe":
                    continue
                key = f"obs_{vid}_{rec['i']}"
                if key not in sp.files:
                    continue
                ctx.check_cancel()
                vdir = ctx.vdir(vid)
                img = cv2.imread(str(vdir / rec["image"]))
                inv = np.load(vdir / rec["depth"])["inv_depth"].astype(np.float32)
                s = min(1.0, FUSION_MAX_WIDTH / img.shape[1])
                if s < 1:
                    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                    inv = cv2.resize(inv, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
                cam = Camera(intr.scaled(img.shape[1] / intr.width), Pose.from_json({"world_from_cam": mats[rec["i"]]}))
                labels = None
                if rec.get("semantic"):
                    labels = cv2.imread(str(vdir / rec["semantic"]), cv2.IMREAD_UNCHANGED)
                    labels = cv2.resize(labels, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                euv, ez = (ground_constraints(cam, labels, GROUND_IDS) if labels is not None and ground_ok
                           else (None, None))
                al = align_depth(inv, cam, xyz[sp[key]], max_depth=1e6, extra_uv=euv, extra_z=ez)
                item = (f"{vid}_{rec['i']}", cam, inv, img, labels, xyz[sp[key]])
                if al is None or al[1]["rel_residual"] > 0.15 or al[1]["depth_range_ratio"] < 1.25:
                    pending.append(item)
                    continue
                z, fit = al
                fit["method"] = "SfM points" + (" + semantic ground plane" if euv is not None and len(euv) else "")
                kfs.append(KeyframeDepth(item[0], cam, z, img[:, :, ::-1].copy(), fit, labels))
                fits.append(fit["rel_residual"])
        # second pass: align difficult keyframes to the depth of already aligned neighbours
        if kfs and pending:
            consistency(kfs, rel=cfg.depth_consistency_rel)
            for key_, cam, inv, img, labels, pts in pending:
                order = sorted(kfs, key=lambda k: np.linalg.norm(k.cam.center - cam.center))[:4]
                puv, pz = projected_constraints(cam, inv.shape, order)
                al = align_depth(inv, cam, pts, max_depth=1e6, extra_uv=puv, extra_z=pz, extra_weight=0.7)
                if al is None or al[1]["rel_residual"] > 0.15:
                    res.warn(f"{key_}: monocular depth could not be aligned reliably; keyframe not fused.")
                    continue
                z, fit = al
                fit["method"] = "SfM points + depth projected from aligned neighbouring keyframes"
                kfs.append(KeyframeDepth(key_, cam, z, img[:, :, ::-1].copy(), fit, labels))
                fits.append(fit["rel_residual"])
            kfs.sort(key=lambda k: (k.key.split("_")[0], int(k.key.split("_")[1])))
        if len(kfs) < 2:
            raise RuntimeError("fewer than two keyframes with aligned depth")
        progress(0.3, f"multi-view consistency over {len(kfs)} keyframes")
        cons = consistency(kfs, rel=cfg.depth_consistency_rel)
        for kf in kfs:
            np.savez_compressed(kdir / f"{kf.key}.npz", z=kf.z.astype(np.float16), support=kf.support,
                                labels=kf.labels if kf.labels is not None else np.zeros(1, np.uint8),
                                K=np.array([kf.cam.intr.fx, kf.cam.intr.fy, kf.cam.intr.cx, kf.cam.intr.cy,
                                            kf.cam.intr.width, kf.cam.intr.height, kf.cam.intr.k1]),
                                world_from_cam=kf.cam.pose.matrix())
        centers = np.array([kf.cam.center for kf in kfs])
        lo = np.minimum(np.percentile(xyz, 2, 0), centers.min(0))
        hi = np.maximum(np.percentile(xyz, 98, 0), centers.max(0))
        pad = 0.1 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        lo[1] = min(lo[1], -0.5)
        horiz = max(hi[0] - lo[0], hi[2] - lo[2])
        voxel = horiz * cfg.tsdf_voxel_frac
        progress(0.45, "TSDF fusion")
        vol = tsdf_fuse(kfs, (lo, hi), voxel, cfg.tsdf_max_voxels)
        progress(0.75, "extracting surface")
        mesh = extract_mesh(vol)
        layers = []
        if mesh is not None:
            lab, lconf = label_vertices(mesh["pos"], kfs)
            evi = np.where(mesh["support"] >= 2, EVI["OBSERVED_MULTI_VIEW"], EVI["OBSERVED_SINGLE_VIEW"]).astype(np.uint8)
            conf = np.clip(0.25 + 0.75 * np.minimum(mesh["weight"] / 6.0, 1.0), 0, 1)
            layers.append({**write_layer(out, "surface", mesh["pos"], mesh["col"], mesh["idx"], evi,
                                         (conf * 255).astype(np.uint8), lab),
                           "kind": "mesh", "method": "TSDF fusion of monocular depth aligned to SfM, multi-view checked",
                           "voxel_size": vol["voxel"]})
        # sparse SfM points (always triangulated from >= 2 views)
        err = sp["error"]
        tl = sp["track_len"]
        pconf = np.clip(1.0 - err / 3.0, 0.1, 1.0) * np.clip(tl / 5.0, 0.3, 1.0)
        layers.append({**write_layer(out, "sparse", xyz, sp["rgb"], None,
                                     np.full(len(xyz), EVI["OBSERVED_MULTI_VIEW"], np.uint8),
                                     (pconf * 255).astype(np.uint8), None),
                       "kind": "points", "method": "COLMAP triangulated points"})
        # ground coverage
        x0, x1 = lo[0], hi[0]
        z0, z1 = lo[2], hi[2]
        cov = coverage_map([(kf.cam, kf.z) for kf in kfs], (float(x0), float(x1), float(z0), float(z1)),
                           cell=max(0.05, horiz / 400), max_range=float(np.linalg.norm(hi - lo)))
        save_coverage_png(cov, out / "coverage.png")
        np.save(out / "coverage_grid.npy", cov["grid"])
        res.outputs.append(out / "coverage.png")
        for l in layers:
            res.outputs += [out / f for f in l["files"].values()]
        log.info(f"{unit['id']}: fused {len(kfs)} keyframes, voxel {vol['voxel']:.3f}, "
                 f"{layers[0]['stats']['vertices'] if mesh is not None else 0} surface vertices")
        return {"unit": unit["id"], "kind": "sfm", "videos": unit["videos"], "layers": layers,
                "fusion": {"keyframes": len(kfs), "voxel": vol["voxel"], "dims": vol["dims"],
                           "alignment_median_residual": float(np.median(fits)), "consistency": cons},
                "coverage": {"image": "coverage.png", "extent": cov["extent"], "cell": cov["cell"],
                             "seen_fraction": cov["seen_fraction"], "occluded_fraction": cov["occluded_fraction"]},
                "keyframe_depths": sorted(f"kf_depth/{kf.key}.npz" for kf in kfs)}
