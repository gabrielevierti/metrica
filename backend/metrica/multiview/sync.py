"""Temporal synchronisation of several videos of the same event.

Offsets are *seconds to add to a video's canonical time to obtain case time* (the
reference video has offset 0). Evidence sources, each with a confidence:

* audio  - cross-correlation of onset-strength envelopes (shared sounds: horns, voices,
           impacts). Strong when both recordings contain the same distinct sounds.
* visual - cross-correlation of scene-activity signals. Weak evidence: only meaningful
           when both cameras see the same moving objects at the same time.
* manual - analyst-entered offset (always overrides; recorded in the audit log).

Container timestamps (creation_time) are reported as context only: device clocks are
frequently wrong and are never used silently.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import av
import numpy as np

ENV_RATE = 100  # envelope samples per second


def audio_envelope(path: Path, max_s: float = 900.0, sr: int = 8000) -> np.ndarray | None:
    try:
        with av.open(str(path)) as c:
            if not c.streams.audio:
                return None
            st = c.streams.audio[0]
            res = av.AudioResampler(format="s16", layout="mono", rate=sr)
            chunks = []
            total = 0
            for fr in c.decode(st):
                for r in res.resample(fr):
                    a = r.to_ndarray().reshape(-1)
                    chunks.append(a)
                    total += len(a)
                if total > max_s * sr:
                    break
    except av.error.FFmpegError:
        return None
    if not chunks:
        return None
    x = np.concatenate(chunks).astype(np.float32)
    hop = sr // ENV_RATE
    n = len(x) // hop
    if n < ENV_RATE:
        return None
    env = np.log(np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(1) + 1e-6))
    onset = np.clip(np.diff(env, prepend=env[0]), 0, None)
    return (onset - onset.mean()) / (onset.std() + 1e-9)


def xcorr_offset(a: np.ndarray, b: np.ndarray, rate: float, max_lag_s: float | None = None,
                 exclude_s: float = 0.5) -> tuple[float, float, float] | None:
    """Lag (s) to add to b's time to align with a, peak value, and a peak-to-second-peak ratio."""
    if a is None or b is None or len(a) < 10 or len(b) < 10:
        return None
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    xc = np.fft.irfft(np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n)), n)
    xc = np.concatenate([xc[-(len(b) - 1):], xc[: len(a)]])
    lags = np.arange(-(len(b) - 1), len(a))
    if max_lag_s is not None:
        m = np.abs(lags) <= max_lag_s * rate
        xc, lags = xc[m], lags[m]
    xc = xc / (np.sqrt((a ** 2).sum() * (b ** 2).sum()) + 1e-9)
    k = int(np.argmax(xc))
    mask = np.abs(lags - lags[k]) > exclude_s * rate
    second = float(np.max(xc[mask])) if mask.any() else 0.0
    ratio = float(xc[k] / max(second, 1e-6)) if second > 0 else float("inf")
    return float(lags[k] / rate), float(xc[k]), ratio


def audio_offset(ref: Path, other: Path) -> dict[str, Any] | None:
    a, b = audio_envelope(ref), audio_envelope(other)
    r = xcorr_offset(a, b, ENV_RATE)
    if r is None:
        return None
    lag, peak, ratio = r
    conf = float(np.clip((ratio - 1.0) / 1.5, 0, 1))  # ratio 2.5+ -> confident
    return {"method": "audio onset cross-correlation", "offset_s": lag, "peak": peak, "peak_ratio": ratio,
            "confidence": conf, "resolution_s": 1.0 / ENV_RATE}


def activity_series(samples: list[dict[str, Any]], rate: float = 10.0) -> tuple[np.ndarray, float] | None:
    ts = np.array([s["t"] for s in samples if "activity" in s])
    act = np.array([s["activity"] for s in samples if "activity" in s])
    if len(ts) < 20:
        return None
    grid = np.arange(ts[0], ts[-1], 1.0 / rate)
    x = np.interp(grid, ts, act)
    x = np.diff(x, prepend=x[0])  # changes in activity are what is shared
    return (x - x.mean()) / (x.std() + 1e-9), float(ts[0])


def visual_offset(ref_samples: list[dict[str, Any]], other_samples: list[dict[str, Any]]) -> dict[str, Any] | None:
    ra, rb = activity_series(ref_samples), activity_series(other_samples)
    if ra is None or rb is None:
        return None
    r = xcorr_offset(ra[0], rb[0], 10.0)
    if r is None:
        return None
    lag, peak, ratio = r
    lag += ra[1] - rb[1]
    conf = float(np.clip((ratio - 1.2) / 2.0, 0, 1) * np.clip(peak / 0.3, 0, 1))
    return {"method": "scene-activity cross-correlation (weak evidence)", "offset_s": lag, "peak": peak,
            "peak_ratio": ratio, "confidence": conf, "resolution_s": 0.1}
