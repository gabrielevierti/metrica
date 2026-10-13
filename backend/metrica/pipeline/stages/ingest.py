"""Stage A - video ingestion, technical quality assessment, motion classification, keyframes."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ...ingest import motion as mot
from ...ingest import quality as q
from ...ingest.decode import FrameSource, read_frames_at
from ...ingest.probe import probe, timing_analysis
from ...ingest.proxy import make_proxy
from ...log import get_logger
from ..base import STATUS_FAILED, Context, Stage, StageResult

log = get_logger("ingest")
BG_SAMPLES = 48
BG_MAX_SIDE = 1920
QUALITY_SIDE = 320
MOTION_SIDE = 480


class IngestStage(Stage):
    name = "ingest"
    title = "Video ingestion and quality assessment"
    config_keys = ("min_sharpness", "keyframe_flow_frac", "max_keyframes", "capture_mode",
                   "capture_mode_overrides")
    version = "4"

    def run(self, ctx: Context) -> StageResult:
        res = StageResult()
        vids = ctx.video_ids()
        # evidence check first: a modified original must stop the analysis
        for r in ctx.case.verify_originals():
            if not r["ok"]:
                res.status = STATUS_FAILED
                res.warn(f"Evidence file {r['id']} failed verification: {r.get('reason', 'hash mismatch')}. "
                         "Processing stopped.")
                return res
        ctx.case.audit.append("evidence.verified", details={"videos": len(vids)})
        for k, vid in enumerate(vids):
            def sub(f, m, k=k, vid=vid):
                ctx.progress((k + f) / len(vids), f"{vid}: {m}")
            info = self._ingest_video(ctx, vid, sub, res)
            res.metrics[vid] = {"mode": info["mode"]["mode"], "frames": info["timing"]["frames"],
                                "keyframes": len(info["keyframes"]), "usable_fraction":
                                info["quality"]["summary"].get("usable_fraction")}
        return res

    # ------------------------------------------------------------------
    def _ingest_video(self, ctx: Context, vid: str, progress, res: StageResult) -> dict[str, Any]:
        cfg = ctx.config
        path = ctx.case.video_path(vid)
        out = ctx.vdir(vid)
        meta = probe(path)
        declared_fps = meta["video"]["average_rate"]
        est_frames = meta["video"]["frames_declared"] or int((meta["duration_s"] or 0) * (declared_fps or 25))
        bg_targets = set(np.linspace(0, max(est_frames - 1, 0), BG_SAMPLES).round().astype(int).tolist()) \
            if est_frames > 0 else set()
        motion_every = max(1, int(round((declared_fps or 25) / 5.0)))

        src = FrameSource(path)
        times: list[float] = []
        samples: list[dict[str, Any]] = []
        motion_samples: list[dict[str, Any]] = []
        pairs: list[dict[str, Any]] = []
        prev_thumb = None
        prev_motion_gray = None
        prev_sample_thumb = None
        bg_frames: list[np.ndarray] = []
        dup_count = 0
        progress(0.0, "reading frames")
        for i, t, fr, _key in src.raw_frames():
            ctx.check_cancel() if i % 200 == 0 else None
            times.append(t)
            g = src.to_gray(fr, QUALITY_SIDE)
            th = q.thumb(g)
            dup = q.is_duplicate(prev_thumb, th)
            dup_count += dup
            prev_thumb = th
            if i % motion_every == 0:
                m = q.frame_metrics(g)
                m.update({"i": i, "t": round(t, 4), "duplicate": bool(dup)})
                # scene activity (for visual synchronisation): mean change since the previous sample
                m["activity"] = float(np.abs(th - prev_sample_thumb).mean()) if prev_sample_thumb is not None else 0.0
                prev_sample_thumb = th
                m["usable"] = bool(m["sharpness"] >= cfg.min_sharpness and m["dark_clip"] < 0.6
                                   and m["bright_clip"] < 0.6)
                if len(samples) % 5 == 0:
                    full = src.to_bgr(fr)
                    m["blockiness"] = q.blockiness(cv2.cvtColor(full, cv2.COLOR_BGR2GRAY))
                samples.append(m)
                mg = src.to_gray(fr, MOTION_SIDE)
                if prev_motion_gray is not None and prev_motion_gray.shape == mg.shape:
                    pm = mot.pair_motion(prev_motion_gray, mg)
                    pm["t"] = round(t, 3)
                    pairs.append(pm)
                prev_motion_gray = mg
                motion_samples.append({"i": i, "sharpness": m["sharpness"] * (MOTION_SIDE / QUALITY_SIDE) ** 2,
                                       "usable": m["usable"]})
            if i in bg_targets:
                bg_frames.append(src.to_bgr(fr, BG_MAX_SIDE))
            if i % 300 == 0 and est_frames:
                progress(0.6 * min(1.0, i / est_frames), f"frame {i}/{est_frames}")
        stats = src.stats
        times_arr = np.array(times)
        np.save(out / "frame_times.npy", times_arr)
        timing = timing_analysis(times_arr, declared_fps)
        timing["duplicate_frames"] = int(dup_count)
        if stats.decode_errors:
            res.warn(f"{vid}: {stats.decode_errors} corrupted packet(s) skipped during decoding.")
        if timing["vfr"]:
            res.warn(f"{vid}: variable frame rate detected; per-frame timestamps from the container are used.")
        if timing["gap_count"]:
            res.warn(f"{vid}: {timing['gap_count']} timestamp gap(s) longer than 2.5 frame intervals "
                     "(dropped frames or recording interruptions).")

        mode = mot.classify(pairs)
        mode["auto_mode"] = mode["mode"]
        override = cfg.capture_mode_overrides.get(vid) or (cfg.capture_mode if cfg.capture_mode != "auto" else None)
        if override:
            mode["mode"] = override
            mode["override"] = override
            mode["notes"].append(f"Capture mode set manually to '{override}'.")
        mode["segments"] = mot.segments([(p["t"], p["state"]) for p in pairs])
        quality_summary = q.summarize(samples, cfg.min_sharpness)

        # ---- keyframes (moving) or background plate (static / rotating)
        keyframe_idx: list[int] = []
        bg_path = None
        if mode["mode"] == "moving":
            keyframe_idx = mot.select_keyframes(motion_samples, pairs, cfg.keyframe_flow_frac, cfg.max_keyframes,
                                                cfg.min_sharpness * (MOTION_SIDE / QUALITY_SIDE) ** 2)
        else:
            if bg_frames:
                stack = np.stack(bg_frames[: BG_SAMPLES])
                bg = np.median(stack, axis=0).astype(np.uint8) if mode["mode"] == "static" else None
                if bg is None:  # rotating: a median across pans is meaningless; use the sharpest frame
                    sharp = [cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var() for f in bg_frames]
                    bg = bg_frames[int(np.argmax(sharp))]
                    res.warn(f"{vid}: rotating (pan/tilt) camera - the surface is reconstructed from one "
                             "representative frame; per-frame orientation is estimated by registering frames to it.")
                bg_path = out / "background.jpg"
                cv2.imwrite(str(bg_path), bg, [cv2.IMWRITE_JPEG_QUALITY, 95])
                res.outputs.append(bg_path)
            # a few sharp frames for depth/semantics cross-checks
            good = sorted([s for s in samples if s["usable"]], key=lambda s: -s["sharpness"])[:3]
            keyframe_idx = sorted(s["i"] for s in good)
        del bg_frames
        progress(0.65, "extracting keyframes")
        kdir = ctx.vdir(vid, "keyframes")
        frames = read_frames_at(path, keyframe_idx)
        keyframes = []
        for i in sorted(frames):
            p = kdir / f"{i:07d}.jpg"
            cv2.imwrite(str(p), frames[i], [cv2.IMWRITE_JPEG_QUALITY, 95])
            keyframes.append({"i": int(i), "t": float(times_arr[i]), "file": f"keyframes/{p.name}"})
        if frames:
            first = frames[min(frames)]
            thumb = cv2.resize(first, (320, int(320 * first.shape[0] / first.shape[1])))
            cv2.imwrite(str(out / "thumb.jpg"), thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])

        progress(0.75, "encoding review proxy")
        proxy = make_proxy(path, out / "proxy.mp4")
        res.outputs.append(out / "proxy.mp4")

        h, w = (frames[min(frames)].shape[:2]) if frames else (meta["video"]["height"], meta["video"]["width"])
        info = {
            "video_id": vid, "probe": meta, "timing": timing,
            "decode": {"frames": stats.frames, "decode_errors": stats.decode_errors,
                       "nonmonotonic_pts": stats.nonmonotonic_pts, "missing_pts": stats.missing_pts,
                       "rotation_applied_deg": stats.rotation, "t0_pts_s": stats.t0_pts_s},
            "frame_size": {"width": int(w), "height": int(h)},
            "quality": {"summary": quality_summary, "samples": samples},
            "mode": mode, "motion_pairs": pairs,
            "keyframes": keyframes, "background": "background.jpg" if bg_path else None,
            "proxy": proxy,
        }
        p = ctx.case.write_json(f"derived/videos/{vid}/ingest.json", info)
        res.outputs.append(p)
        log.info(f"{vid}: {timing['frames']} frames, {timing['fps_measured']:.2f} fps, mode {mode['mode']} "
                 f"(confidence {mode['confidence']:.2f}), {len(keyframes)} keyframes")
        return info
