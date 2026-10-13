"""Stage 9 - assemble the versioned scene graph (metrica-scene/1) in the case world frame."""
from __future__ import annotations

import shutil
import time
from typing import Any

import numpy as np

from ... import SCENE_SCHEMA, __version__
from ...geometry.calibration import STATUS_TEXT, resolve_global_scale
from ...geometry.camera import Pose
from ...log import get_logger
from ...reconstruction.layers import EVIDENCE_TEXT, read_layer, write_layer
from ...scene.export import export_glb, export_ply
from ...scene.transform import Similarity, transform_agent, transform_entity
from ...semantics.labels import CATEGORIES, SCENE_LABELS
from ..base import Context, Stage, StageResult, frame_fingerprint, load_record

log = get_logger("scene")
CAMERA_COLORS = ["#ff6b35", "#00a6fb", "#9b5de5", "#00bb7f", "#f15bb5", "#fee440"]


def _relations(entities: list[dict[str, Any]], unit: str, near: float = 2.0) -> None:
    pos = {}
    for e in entities:
        g = e["geometry"]
        if g["type"] == "cylinder":
            pos[e["id"]] = np.asarray(g["base"])
        elif g["type"] in ("vehicle_box", "box"):
            pos[e["id"]] = np.asarray(g["center"])
    ids = list(pos)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            d = float(np.hypot(*(pos[ids[i]] - pos[ids[j]])[[0, 2]]))
            if d < near:
                for a, b in ((ids[i], ids[j]), (ids[j], ids[i])):
                    ea = next(e for e in entities if e["id"] == a)
                    ea.setdefault("relations", []).append({"type": "near", "with": b, "distance": round(d, 3),
                                                           "unit": unit})


class SceneStage(Stage):
    name = "scene"
    title = "Scene graph assembly"
    depends = ("objects", "trajectories", "alignment", "analysis", "reconstruction")
    config_keys = ()
    version = "6"
    annotation_keys = ("calibration", "sync", "alignment")

    def run(self, ctx: Context) -> StageResult:
        res = StageResult()
        case = ctx.case
        sdir = case.stage_dir("scene")
        ldir = case.stage_dir("scene", "layers")
        units = case.read_json("derived/geometry/units.json")["units"]
        align = case.read_json("derived/alignment/alignment.json") or {"units": {}, "offsets": {}, "sync": {}}
        analysis = case.read_json("derived/analysis/analysis.json") or {"events": [], "vehicles": {}, "faces": {}}
        vids = ctx.video_ids()
        unit_of = {v: u["id"] for u in units for v in u["videos"]}
        ref_unit = next(u for u in units if u["id"] == unit_of[vids[0]])
        refs = (case.annotations("calibration") or {}).get("references") or []
        k, scale = resolve_global_scale(ref_unit["scale"], refs)
        unit = scale["unit"]
        offsets = align.get("offsets", {})
        sims = {u["id"]: Similarity(np.asarray(align["units"].get(u["id"], {}).get("transform", np.eye(4))), k)
                for u in units}
        warnings: list[dict[str, Any]] = []
        fp_now = frame_fingerprint(case)
        for r in refs:
            if r.get("active", True) and r.get("frame_fingerprint") and r["frame_fingerprint"] != fp_now:
                warnings.append({"source": "calibration", "message": f"Scale reference {r['id']} was placed on an "
                                 "earlier reconstruction; its points may no longer lie on the same features. "
                                 "Verify it or place it again."})
        for u in units:
            st = align["units"].get(u["id"], {})
            if st.get("status") == "unaligned":
                warnings.append({"source": "alignment", "message": st.get("warning")})
            if u["scale"]["status"] != scale["status"] and u["id"] != ref_unit["id"]:
                warnings.append({"source": "scale", "message": f"Unit {u['id']} has scale status {u['scale']['status']} "
                                                               f"while the reference is {scale['status']}."})
        # ---------------------------------------------------------------- layers
        layers, glb_layers = [], []
        for u in units:
            ctx.check_cancel()
            recon = case.read_json(f"derived/reconstruction/{u['id']}/reconstruction.json")
            if not recon:
                continue
            rdir = case.root / "derived" / "reconstruction" / u["id"]
            S = sims[u["id"]]
            for meta in recon.get("layers", []):
                L = read_layer(rdir, meta)
                pos = S.point(L["pos"]).astype(np.float32)
                name = f"{u['id']}_{meta['name']}"
                out = write_layer(ldir, name, pos, L.get("col"), L.get("idx"), L.get("evi"), L.get("conf"), L.get("lab"))
                layers.append({"id": name, "unit": u["id"], "kind": meta["kind"], "name": meta["name"],
                               "files": {k2: f"layers/{v}" for k2, v in out["files"].items()},
                               "stats": out["stats"], "method": meta.get("method"),
                               "default_visible": meta["kind"] == "mesh"})
                if meta["kind"] == "mesh":
                    glb_layers.append({"name": name, "pos": pos, "idx": L["idx"], "col": L.get("col")})
                elif meta["kind"] == "points" and len(pos):
                    export_ply(sdir / f"{name}.ply", pos, L.get("col"))
            cov = recon.get("coverage")
            if cov and (rdir / cov["image"]).exists():
                x0, x1, z0, z1 = cov["extent"]
                corners = S.xz(np.array([[x0, z0], [x1, z0], [x1, z1], [x0, z1]]))
                dst = ldir / f"{u['id']}_coverage.png"
                shutil.copyfile(rdir / cov["image"], dst)
                layers.append({"id": f"{u['id']}_coverage", "unit": u["id"], "kind": "coverage",
                               "files": {"texture": f"layers/{dst.name}"}, "corners_xz": corners.round(4).tolist(),
                               "stats": {"seen_fraction": cov["seen_fraction"],
                                         "occluded_fraction": cov["occluded_fraction"]},
                               "legend": {"green": "ground seen by a camera", "orange": "in view but hidden",
                                          "transparent": "never in any field of view (unknown)"},
                               "default_visible": False})
        # ---------------------------------------------------------------- entities
        entities = []
        for u in units:
            objs = case.read_json(f"derived/objects/{u['id']}/objects.json", {"entities": []})["entities"]
            odir = case.root / "derived" / "objects" / u["id"]
            for e in objs:
                te = transform_entity(e, sims[u["id"]], unit)
                if te.get("texture"):
                    src = odir / te["texture"]["file"]
                    if src.exists():
                        dst = ldir / f"{u['id']}_{te['texture']['file']}"
                        shutil.copyfile(src, dst)
                        te["texture"]["file"] = f"layers/{dst.name}"
                te["display"] = ("Ground plane" if te["geometry"]["type"] == "ground_plane"
                                 else CATEGORIES.get(te["category"], CATEGORIES["object"]).display)
                entities.append(te)
        _relations(entities, unit)
        # ---------------------------------------------------------------- cameras, videos, agents
        cameras, videos, agents = [], [], []
        for n, vid in enumerate(vids):
            geo = case.read_json(f"derived/geometry/{vid}.json")
            ing = case.read_json(f"derived/videos/{vid}/ingest.json")
            v = case.video(vid)
            S = sims[unit_of[vid]]
            off = float(offsets.get(vid, 0.0))
            videos.append({"id": vid, "name": v["original_name"], "sha256": v["sha256_intake"],
                           "proxy": f"videos/{vid}/proxy.mp4", "thumbnail": f"videos/{vid}/thumb.jpg",
                           "fps": ing["timing"]["fps_measured"], "frames": ing["timing"]["frames"],
                           "duration": ing["timing"].get("duration_s"), "width": ing["frame_size"]["width"],
                           "height": ing["frame_size"]["height"], "offset": off, "mode": ing["mode"]["mode"],
                           "unit": unit_of[vid], "sync": align.get("sync", {}).get(vid),
                           "vfr": ing["timing"]["vfr"], "quality_flags": ing["quality"]["summary"].get("flags", [])})
            if geo is None:
                continue
            cam = {"id": f"cam-{vid}", "video": vid, "intrinsics": geo["intrinsics"], "color": CAMERA_COLORS[n % 6],
                   "mode": geo["mode"]}
            Sm = S.matrix
            if geo["mode"] == "sfm":
                frames = geo["poses"]["frames"]
                times = np.load(case.root / "derived" / "videos" / vid / "frame_times.npy")
                mats = [Pose.from_json({"world_from_cam": m}).transformed(Sm).matrix().round(5).tolist()
                        for m in geo["poses"]["world_from_cam"]]
                cam["kind"] = "moving"
                cam["poses"] = {"frames": frames, "t": [round(float(times[f]) + off, 4) for f in frames],
                                "world_from_cam": mats, "source": geo["poses"]["source"]}
                ki = [i for i, s in enumerate(geo["poses"]["source"]) if s == "keyframe"]
                cam["world_from_cam"] = mats[ki[0] if ki else 0]
            else:
                cam["kind"] = "static"
                cam["world_from_cam"] = Pose.from_json(geo["static_pose"]).transformed(Sm).matrix().round(6).tolist()
                if geo.get("poses"):  # rotating camera: fixed centre, per-frame orientation
                    frames = geo["poses"]["frames"]
                    times = np.load(case.root / "derived" / "videos" / vid / "frame_times.npy")
                    cam["kind"] = "rotating"
                    cam["poses"] = {"frames": frames, "t": [round(float(times[f]) + off, 4) for f in frames],
                                    "world_from_cam": [Pose.from_json({"world_from_cam": m}).transformed(Sm).matrix()
                                                       .round(5).tolist() for m in geo["poses"]["world_from_cam"]],
                                    "source": geo["poses"]["source"]}
            cameras.append(cam)
            traj = case.read_json(f"derived/trajectories/{vid}.json", {"tracks": []})
            t2d = {t["id"]: t for t in case.read_json(f"derived/videos/{vid}/tracks2d.json", {"tracks": []})["tracks"]}
            for tr in traj["tracks"]:
                a = transform_agent(tr, S, off)
                a["video"] = vid
                src = t2d.get(tr["id"], {})
                a["crops"] = [{**c, "file": f"videos/{vid}/{c['file']}",
                               **({"mask": {**c["mask"], "file": f"videos/{vid}/{c['mask']['file']}"}}
                                  if c.get("mask") else {})} for c in src.get("crops", [])]
                a["faces"] = [{**f, "file": f"videos/{vid}/{f['file']}"} for f in src.get("faces", [])]
                veh = analysis.get("vehicles", {}).get(tr["id"])
                if veh:
                    a["color"] = veh["color"]
                    a["ocr_candidates"] = veh["ocr_candidates"]
                if a["group"] == "person":
                    h = (a.get("stature") or {}).get("median")
                    a["representation"] = {"type": "mannequin", "height": h or 1.70,
                                           "basis": "estimated stature" if h else "class prior (1.70)",
                                           "note": "Generic body proxy; not a scan of the individual."}
                elif a["group"] in ("vehicle", "two_wheeler"):
                    from ...reconstruction.objects3d import VEHICLE_PRIORS
                    pri = VEHICLE_PRIORS.get(a["category"], VEHICLE_PRIORS["car"])
                    a["representation"] = {"type": "vehicle", "length": pri["length"][0], "width": pri["width"][0],
                                           "height": pri["height"][0], "basis": "class prior dimensions",
                                           "note": "Generic template; dimensions are class-typical values."}
                else:
                    a["representation"] = {"type": "marker"}
                agents.append(a)
        # speeds are only meaningful in a metric frame: withheld otherwise (they reappear when the scene is
        # recomputed after a calibration with a known distance)
        if scale["status"] not in ("ESTIMATED_METRIC", "CALIBRATED_METRIC"):
            for a in agents:
                if a.get("speed") or (a.get("summary") or {}).get("speed_median") is not None:
                    a["speed"] = []
                    if a.get("summary"):
                        a["summary"]["speed_median"] = a["summary"]["speed_max"] = None
                    a["speed_note"] = (f"Speed not reported: the scale is {scale['status']} (no metric calibration). "
                                       "Calibrate with a known distance to obtain speeds.")
                a["flags"] = [f for f in a.get("flags", []) if f.get("type") != "implausible_speed"]
        # reverse links: static entity <- stationary tracks
        for a in agents:
            if a.get("static_entity"):
                for e in entities:
                    if e["id"] == a["static_entity"]:
                        e.setdefault("tracks", []).append(a["id"])
        # ---------------------------------------------------------------- warnings from stage records
        from ..runner import all_stages
        for st in all_stages():
            rec = load_record(case, st.name)
            if not rec:
                continue
            for w in rec.get("warnings", []):
                warnings.append({"source": st.name, "message": w})
            for f in rec.get("fallbacks", []):
                warnings.append({"source": st.name, "message": f"Fallback in {f['component']}: {f['used']}. "
                                                               f"{f['implication']}"})
        t_all = [a["first_t"] for a in agents] + [v["offset"] for v in videos]
        t_end = [a["last_t"] for a in agents] + [v["offset"] + (v["duration"] or 0) for v in videos]
        scene = {
            "schema": SCENE_SCHEMA, "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "software_version": __version__,
            "case": {k2: case.meta.get(k2) for k2 in ("id", "title", "reference")},
            "coordinate_system": {"up": "+Y", "handedness": "right", "ground": "y = 0", "unit": unit,
                                  "scale": scale, "status": scale["status"], "status_text": STATUS_TEXT[scale["status"]],
                                  "global_scale_factor": k, "reference_video": vids[0],
                                  "origin": "below the reference camera at its first registered position"},
            "time": {"reference_video": vids[0], "offsets": offsets, "start": float(min(t_all) if t_all else 0.0),
                     "end": float(max(t_end) if t_end else 0.0),
                     "note": "case time = video time + offset; video time 0 = first decoded frame"},
            "videos": videos, "cameras": cameras, "layers": layers, "entities": entities, "agents": agents,
            "events": analysis.get("events", []), "alignment": align.get("units", {}),
            "conflicts": align.get("conflicts", []), "grouping": align.get("grouping"), "warnings": warnings,
            "evidence_classes": EVIDENCE_TEXT,
            "categories": {k2: {"display": c.display, "color": c.color, "group": c.group} for k2, c in CATEGORIES.items()},
            "scene_labels": SCENE_LABELS,
            "stats": {"entities": len(entities), "agents": len(agents), "events": len(analysis.get("events", [])),
                      "layers": len(layers), "cameras": len(cameras)},
        }
        p = case.write_json("derived/scene/scene.json", scene)
        res.outputs.append(p)
        try:
            glb = export_glb(sdir / "scene.glb", glb_layers, entities)
            res.outputs.append(sdir / "scene.glb")
            res.metrics["glb"] = glb
        except Exception as e:  # noqa: BLE001 - export is a convenience
            res.warn(f"GLB export failed: {e}")
        res.metrics.update(scene["stats"])
        return res
