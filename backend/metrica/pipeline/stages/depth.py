"""Stage C (part 1) - relative monocular depth and dense scene semantics on reconstruction frames.

Depth here is *relative* (affine-invariant). Metric conversion happens in the reconstruction
stage, after camera geometry is known, against geometric evidence.
"""
from __future__ import annotations

import cv2
import numpy as np

from ...log import get_logger
from ...models.depth import DepthAnythingV2
from ...models.registry import ModelUnavailable, available
from ...models.segmentation import DUCCityscapes
from ..base import Context, Stage, StageResult

log = get_logger("depth")
MAX_SEMANTIC_FRAMES = 12


class DepthSemanticsStage(Stage):
    name = "depth"
    title = "Monocular depth and scene semantics"
    depends = ("ingest",)
    config_keys = ("depth_input_short_side", "use_semantic_segmentation")
    version = "2"

    def run(self, ctx: Context) -> StageResult:
        res = StageResult()
        cfg = ctx.config
        try:
            depth = DepthAnythingV2(cfg.depth_input_short_side)
            res.models.append(depth.key)
        except ModelUnavailable as e:
            depth = None
            res.fallback("depth estimation", "none", f"{e}. Dense surfaces cannot be built; only sparse "
                                                     "geometry (SfM points, fitted planes) will be available.")
        sem = None
        if cfg.use_semantic_segmentation:
            if available("duc_cityscapes"):
                sem = DUCCityscapes()
                res.models.append(sem.key)
            else:
                res.fallback("scene semantics", "geometry-only labels",
                             "Semantic model not downloaded: ground/walls come from geometry, poles from line "
                             "evidence; vegetation/sky are not distinguished.")
        else:
            res.warn("Scene semantic segmentation disabled by profile: geometry-only labels.")
        vids = ctx.video_ids()
        for k, vid in enumerate(vids):
            ing = ctx.case.read_json(f"derived/videos/{vid}/ingest.json")
            vdir = ctx.vdir(vid)
            ddir, sdir = ctx.vdir(vid, "depth"), ctx.vdir(vid, "semantic")
            items = []
            if ing["mode"]["mode"] != "moving" and ing.get("background"):
                items.append(("background", None, None, vdir / "background.jpg"))
            else:
                for kf in ing["keyframes"]:
                    items.append((f"{kf['i']:07d}", kf["i"], kf["t"], vdir / kf["file"]))
            sem_set = set()
            if sem is not None and items:
                sel = np.linspace(0, len(items) - 1, min(MAX_SEMANTIC_FRAMES, len(items))).round().astype(int)
                sem_set = {items[j][0] for j in sel}
            index = []
            for j, (name, i, t, path) in enumerate(items):
                ctx.check_cancel()
                ctx.progress((k + j / max(1, len(items))) / len(vids), f"{vid}: frame {j + 1}/{len(items)}")
                img = cv2.imread(str(path))
                rec = {"name": name, "i": i, "t": t, "image": str(path.relative_to(vdir)), "depth": None,
                       "semantic": None}
                if depth is not None:
                    d = depth(img)
                    np.savez_compressed(ddir / f"{name}.npz", inv_depth=d.astype(np.float16))
                    rec["depth"] = f"depth/{name}.npz"
                if name in sem_set:
                    lab, conf = sem(img)
                    cv2.imwrite(str(sdir / f"{name}.png"), lab)
                    cv2.imwrite(str(sdir / f"{name}_conf.png"), (conf * 255).astype(np.uint8))
                    rec["semantic"] = f"semantic/{name}.png"
                    rec["semantic_conf"] = f"semantic/{name}_conf.png"
                index.append(rec)
            p = ctx.case.write_json(f"derived/videos/{vid}/depth_index.json",
                                    {"video_id": vid, "frames": index,
                                     "depth_model": depth.key if depth else None,
                                     "semantic_model": sem.key if sem else None})
            res.outputs.append(p)
            res.metrics[vid] = {"depth_frames": sum(1 for r in index if r["depth"]),
                                "semantic_frames": sum(1 for r in index if r["semantic"])}
        return res
