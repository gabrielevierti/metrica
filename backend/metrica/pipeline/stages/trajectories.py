"""Stage 5 - 3D trajectories of tracked people and vehicles in each unit frame."""
from __future__ import annotations

from typing import Any

import numpy as np

from ...geometry.camera import Camera, Intrinsics, Pose
from ...log import get_logger
from ...tracking.trajectories import build_track
from ..base import Context, Stage, StageResult

log = get_logger("trajectories")


class TrajectoryStage(Stage):
    name = "trajectories"
    title = "3D trajectories and kinematics"
    depends = ("tracking", "geometry", "objects")
    config_keys = ("person_height_m",)
    version = "4"

    def run(self, ctx: Context) -> StageResult:
        res = StageResult()
        for vid in ctx.video_ids():
            geo = ctx.case.read_json(f"derived/geometry/{vid}.json")
            tr = ctx.case.read_json(f"derived/videos/{vid}/tracks2d.json", {"tracks": []})
            ing = ctx.case.read_json(f"derived/videos/{vid}/ingest.json")
            w, h = ing["frame_size"]["width"], ing["frame_size"]["height"]
            if geo is None:
                res.warn(f"{vid}: no camera geometry; trajectories unavailable.")
                continue
            intr = Intrinsics.from_json(geo["intrinsics"])
            if intr.width != w:
                intr = intr.scaled(w / intr.width)
            pose_source: dict[int, str] = {}
            if geo.get("poses"):  # moving (SfM) and rotating cameras: per-frame poses, none -> no 3D
                frames = geo["poses"]["frames"]
                mats = {f: np.asarray(m) for f, m in zip(frames, geo["poses"]["world_from_cam"])}
                pose_source = dict(zip(frames, geo["poses"]["source"]))

                def cam_for(f, mats=mats, intr=intr):
                    m = mats.get(f)
                    return None if m is None else Camera(intr, Pose(m[:3, :3], m[:3, 3]))
            else:
                static = Camera(intr, Pose.from_json(geo["static_pose"]))

                def cam_for(f, static=static):
                    return static
            sigma = geo["scale"].get("scale_sigma_rel")
            tracks = [build_track(t, cam_for, w, h, sigma, pose_source, ctx.config.person_height_m)
                      for t in tr["tracks"]]
            if geo["scale"]["status"] in ("RELATIVE", "ARBITRARY"):  # plausibility limits are in m/s
                for t in tracks:
                    t["flags"] = [f for f in t["flags"] if f.get("type") != "implausible_speed"]
            missing = sum(1 for t in tracks if t["observed_count"] == 0)
            if missing:
                res.warn(f"{vid}: {missing} track(s) have no 3D position (no camera pose for their frames).")
            self._link_static(ctx, vid, geo["unit_id"], tracks)
            p = ctx.case.write_json(f"derived/trajectories/{vid}.json",
                                    {"video_id": vid, "unit": geo["unit_id"], "scale": geo["scale"], "tracks": tracks})
            res.outputs.append(p)
            res.metrics[vid] = {"tracks": len(tracks), "with_3d": len(tracks) - missing}
        return res

    @staticmethod
    def _link_static(ctx: Context, vid: str, unit: str, tracks: list[dict[str, Any]]) -> None:
        objs = ctx.case.read_json(f"derived/objects/{unit}/objects.json", {"entities": []})["entities"]
        vehicles = [e for e in objs if e["geometry"]["type"] == "vehicle_box"]
        for t in tracks:
            if not t.get("smoothed"):
                continue
            P = np.array([s["p"] for s in t["smoothed"]])
            n = max(1, len(P) // 10)
            disp = float(np.linalg.norm(np.median(P[-n:], 0) - np.median(P[:n], 0)))
            sm = t["summary"]
            noise = 3 * (sm.get("position_sigma_median") or 0.0)
            t["stationary"] = bool(disp < max(1.5, noise) and (sm.get("speed_median") or 0) < 0.35)
            if t["stationary"]:
                sm["path_note"] += "; subject essentially stationary: path length reflects measurement noise"
            if t["group"] != "vehicle" or not t["stationary"]:
                continue
            c = np.median(P, 0)
            for e in vehicles:
                g = e["geometry"]["center"]
                if np.hypot(c[0] - g[0], c[2] - g[2]) < 4.0:
                    t["static_entity"] = e["id"]  # reverse link is added when the scene is assembled
