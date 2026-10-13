"""Stage E/F - semantic 3D objects, structured geometry and explicitly inferred regions."""
from __future__ import annotations

from typing import Any

import numpy as np

from ...log import get_logger
from ...models.detection import make_detector
from ...models.registry import ModelUnavailable, available
from ...models.segmentation import EfficientSAMPrompt, GrabCutMask
from ...reconstruction.facades import facade_texture, openings_and_inference, reconstruct_facades
from ...reconstruction.layers import read_layer
from ...reconstruction.objects3d import (
    STATIC_DETECT,
    VEHICLE_PRIORS,
    depth_strip_mask,
    ground_entity,
    object_entities,
    object_observation,
    pole_entities,
    pole_observations,
    surface_cluster_entities,
)
from ...reconstruction.views import load_views
from ..base import Context, Stage, StageResult

log = get_logger("objects")
MAX_DETECTION_VIEWS = 12
MAX_MASKS_PER_UNIT = 24


def _strip_private(e: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in e.items() if not k.startswith("_")}


def _taken_mask(pos: np.ndarray, ents: list[dict[str, Any]], scale: float) -> np.ndarray:
    taken = np.zeros(len(pos), bool)
    for e in ents:
        g = e["geometry"]
        if g["type"] == "plane_polygon":
            n = np.asarray(g["normal"])
            o = np.asarray(g["origin"])
            u = np.asarray(g["u_axis"])
            d = np.abs((pos - o) @ n)
            t = (pos - o) @ u
            us = [p[0] for poly in g["polygons_uv"] for p in poly]
            taken |= (d < 0.35 * scale) & (t > min(us) - 0.3) & (t < max(us) + 0.3)
        elif g["type"] == "cylinder":
            b = np.asarray(g["base"])
            r = max(g["radius_bottom"], g["radius_top"]) + 0.3 * scale
            taken |= (np.hypot(pos[:, 0] - b[0], pos[:, 2] - b[2]) < r) & (pos[:, 1] < g["height"] + 0.3)
        elif g["type"] in ("vehicle_box", "box"):
            c = np.asarray(g["center"])
            L, W = (g["length"], g["width"]) if g["type"] == "vehicle_box" else (g["size"][0], g["size"][2])
            yaw = g["yaw_rad"]
            dx, dz = pos[:, 0] - c[0], pos[:, 2] - c[2]
            a = dx * np.cos(yaw) + dz * np.sin(yaw)
            b2 = -dx * np.sin(yaw) + dz * np.cos(yaw)
            taken |= (np.abs(a) < L / 2 + 0.3 * scale) & (np.abs(b2) < W / 2 + 0.3 * scale)
    return taken


class ObjectsStage(Stage):
    name = "objects"
    title = "Semantic 3D objects and structured geometry"
    depends = ("reconstruction", "geometry", "tracking", "depth")
    config_keys = ("use_instance_masks", "detector", "infer_hidden_surfaces", "detector_conf")
    version = "6"

    def run(self, ctx: Context) -> StageResult:
        res = StageResult()
        cfg = ctx.config
        units = ctx.case.read_json("derived/geometry/units.json")["units"]
        try:
            detector = make_detector(cfg.detector, max(0.4, cfg.detector_conf), cfg.detector_nms_iou)
            res.models.append(cfg.detector)
        except ModelUnavailable as e:
            detector = None
            res.fallback("static object detection", "none", str(e))
        seg = None
        if cfg.use_instance_masks:
            if available("efficientsam_ti"):
                seg = EfficientSAMPrompt()
                res.models.append("efficientsam_ti")
            else:
                seg = GrabCutMask()
        for k, unit in enumerate(units):
            ctx.check_cancel()
            ctx.progress(k / len(units), f"unit {unit['id']}")
            recon = ctx.case.read_json(f"derived/reconstruction/{unit['id']}/reconstruction.json")
            if not recon or not recon.get("layers"):
                res.warn(f"{unit['id']}: no reconstruction available; objects skipped.")
                continue
            ents = self._unit(ctx, unit, recon, detector, seg, res,
                              lambda f, m, k=k: ctx.progress((k + f) / len(units), m))
            p = ctx.case.write_json(f"derived/objects/{unit['id']}/objects.json",
                                    {"unit": unit["id"], "entities": ents})
            res.outputs.append(p)
            cats: dict[str, int] = {}
            for e in ents:
                cats[e["category"]] = cats.get(e["category"], 0) + 1
            res.metrics[unit["id"]] = cats
        return res

    def _unit(self, ctx: Context, unit, recon, detector, seg, res: StageResult, progress) -> list[dict[str, Any]]:
        cfg = ctx.config
        scale = unit["scale"]
        metric = scale["status"] in ("ESTIMATED_METRIC", "CALIBRATED_METRIC")
        unit_str = scale["unit"]
        views = load_views(ctx.case, unit, recon)
        if not views:
            return []
        odir = ctx.case.stage_dir("objects", unit["id"])
        ents: list[dict[str, Any]] = []
        # ---- ground
        rdir = ctx.case.root / "derived" / "reconstruction" / unit["id"]
        if (rdir / "coverage_grid.npy").exists():
            g = ground_entity(unit["id"], recon["coverage"], np.load(rdir / "coverage_grid.npy"), views, unit_str)
            if not cfg.infer_hidden_surfaces:
                g["geometry"]["inferred_polygons_xz"] = []
            ents.append(g)
        # ---- facades
        progress(0.1, "facades")
        facades = reconstruct_facades(views, unit["id"], 1.0, noise=float(recon.get("fusion", {}).get("voxel", 0.0)))
        for i, f in enumerate(facades):
            ctx.check_cancel()
            if cfg.infer_hidden_surfaces:
                openings_and_inference(f, views, 1.0)
            tex = facade_texture(f, views, odir / f"facade_{i + 1}.png", 1.0)
            if tex:
                f["texture"] = tex
                res.outputs.append(odir / tex["file"])
            for d in f["dimensions"].values():
                d["unit"] = unit_str
        ents += [_strip_private(f) for f in facades]
        # ---- poles
        progress(0.35, "poles")
        sem_views = [v for v in views if v.labels is not None]
        pobs = [o for v in sem_views for o in pole_observations(v)]
        for v in views:  # geometry-only cue, independent of the semantic model
            pobs += pole_observations(v, 1.0, mask=depth_strip_mask(v), source="depth")
        ents += pole_entities(pobs, unit["id"], unit_str, 1.0, len(views))
        if not sem_views:
            res.warn(f"{unit['id']}: no semantic labels; poles come from geometry only, vegetation and road regions "
                     "are not distinguished.")
        # ---- detected static objects and parked vehicles
        if detector is not None:
            progress(0.5, "static objects and parked vehicles")
            dviews = views if len(views) <= MAX_DETECTION_VIEWS else \
                [views[i] for i in np.linspace(0, len(views) - 1, MAX_DETECTION_VIEWS).round().astype(int)]
            obs = []
            masks_used = 0
            for v in dviews:
                ctx.check_cancel()
                for d in detector(v.image):
                    if d["category"] not in STATIC_DETECT and d["category"] not in VEHICLE_PRIORS:
                        continue
                    use_seg = seg if masks_used < MAX_MASKS_PER_UNIT else None
                    o = object_observation(v, d, use_seg, 1.0)
                    if use_seg is not None:
                        masks_used += 1
                    if o is not None:
                        obs.append(o)
            ents += object_entities(obs, unit["id"], unit_str, 1.0, metric, len(views))
        # ---- vegetation and unclassified structures from the observed surface
        progress(0.85, "vegetation and unclassified structures")
        surf = next((l for l in recon["layers"] if l["name"] == "surface"), None)
        if surf is not None and "labels" in surf["files"]:
            L = read_layer(rdir, surf)
            taken = _taken_mask(L["pos"], ents, 1.0)
            ents += surface_cluster_entities(L["pos"], L["lab"], taken, unit["id"], unit_str, 1.0)
        for e in ents:
            e["unit"] = unit["id"]
            if e["category"] not in ("ground", "facade"):
                e.setdefault("relations", []).append({"type": "stands_on", "with": f"{unit['id']}-ground"}) \
                    if e["category"] in ("pole",) or e["category"] in VEHICLE_PRIORS else None
        log.info(f"{unit['id']}: {len(ents)} entities ({len(facades)} facades, "
                 f"{sum(1 for e in ents if e['category'] == 'pole')} poles)")
        return ents
