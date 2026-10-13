"""Container and stream metadata (PyAV), with explicit flags for unusual properties."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import av
import numpy as np


def probe(path: Path | str) -> dict[str, Any]:
    info: dict[str, Any] = {"path_name": Path(path).name}
    with av.open(str(path)) as c:
        info["container"] = c.format.name
        info["container_long"] = c.format.long_name
        info["duration_s"] = float(c.duration / av.time_base) if c.duration else None
        info["bit_rate"] = c.bit_rate
        info["tags"] = {k: v for k, v in (c.metadata or {}).items()}
        if not c.streams.video:
            raise ValueError("No video stream found")
        vs = c.streams.video[0]
        cc = vs.codec_context
        info["video"] = {
            "codec": cc.name,
            "profile": getattr(cc, "profile", None),
            "width": cc.width,
            "height": cc.height,
            "pix_fmt": cc.pix_fmt,
            "average_rate": float(vs.average_rate) if vs.average_rate else None,
            "base_rate": float(vs.base_rate) if getattr(vs, "base_rate", None) else None,
            "time_base": str(vs.time_base),
            "frames_declared": int(vs.frames or 0),
            "start_time_s": float(vs.start_time * vs.time_base) if vs.start_time is not None else None,
            "tags": dict(vs.metadata or {}),
        }
        info["audio"] = [{"codec": a.codec_context.name, "sample_rate": a.codec_context.sample_rate,
                          "channels": a.codec_context.channels if hasattr(a.codec_context, "channels")
                          else getattr(a.codec_context.layout, "nb_channels", None)}
                         for a in c.streams.audio]
        info["has_audio"] = bool(c.streams.audio)
        info["creation_time"] = info["tags"].get("creation_time") or info["video"]["tags"].get("creation_time")
        enc = " ".join(str(v) for v in list(info["tags"].values()) + list(info["video"]["tags"].values())).lower()
        info["encoder_hints"] = [h for h in ("dji", "gopro", "apple", "android", "axis", "hikvision", "dahua")
                                 if h in enc or h in Path(path).name.lower()]
    return info


def timing_analysis(times: np.ndarray, declared_fps: float | None) -> dict[str, Any]:
    """Frame-interval statistics from the decoded timestamps (detects VFR and drops)."""
    if len(times) < 3:
        return {"frames": int(len(times)), "vfr": False, "fps_measured": declared_fps}
    dt = np.diff(times)
    med = float(np.median(dt))
    fps = 1.0 / med if med > 0 else (declared_fps or 0)
    rel = np.abs(dt - med) / max(med, 1e-9)
    irregular = float((rel > 0.25).mean())
    gaps = np.where(dt > 2.5 * med)[0]
    return {
        "frames": int(len(times)),
        "duration_s": float(times[-1] - times[0] + med),
        "fps_measured": fps,
        "fps_declared": declared_fps,
        "interval_median_s": med,
        "interval_std_s": float(dt.std()),
        "irregular_interval_fraction": irregular,
        "vfr": irregular > 0.02,
        "gaps": [{"after_frame": int(i), "t": float(times[i]), "gap_s": float(dt[i])} for i in gaps[:200]],
        "gap_count": int(len(gaps)),
    }
