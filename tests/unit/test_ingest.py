"""Decoding, probing, timing, quality metrics and camera-motion classification."""
from __future__ import annotations

import cv2
import numpy as np
from conftest import write_test_video

from metrica.ingest.decode import FrameSource, read_frames_at, rotation_code
from metrica.ingest.motion import classify, pair_motion, segments, select_keyframes
from metrica.ingest.probe import probe, timing_analysis
from metrica.ingest.quality import frame_metrics, is_duplicate, thumb


def test_probe_and_decode_all_frames(tiny_video):
    info = probe(tiny_video)
    assert info["video"]["width"] == 160 and info["video"]["height"] == 96
    src = FrameSource(tiny_video)
    frames = list(src.iter_bgr())
    assert len(frames) == 24
    idx, t, img = frames[5][:3]
    assert idx == 5 and abs(t - 5 / 12) < 1e-3 and img.shape == (96, 160, 3)
    got = read_frames_at(tiny_video, [3, 10])
    assert set(got) == {3, 10}


def test_rotation_codes():
    assert rotation_code(0) is None and rotation_code(None) is None
    assert rotation_code(90) is not None and rotation_code(-90) is not None
    assert rotation_code(180) == cv2.ROTATE_180


def test_timing_detects_vfr_and_gaps():
    cfr = np.arange(100) / 25.0
    r = timing_analysis(cfr, 25.0)
    assert not r["vfr"] and abs(r["fps_measured"] - 25.0) < 0.1
    rng = np.random.default_rng(0)
    vfr = np.cumsum(rng.choice([1 / 30, 1 / 15, 1 / 24], 200))
    assert timing_analysis(vfr, 30.0)["vfr"]
    gap = np.r_[np.arange(50) / 25.0, 5.0 + np.arange(50) / 25.0]
    assert timing_analysis(gap, 25.0)["gaps"]


def test_quality_metrics_rank_blur():
    rng = np.random.default_rng(1)
    g = (rng.random((240, 320)) * 255).astype(np.uint8)
    g = cv2.resize(cv2.resize(g, (80, 60)), (320, 240), interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(g, (0, 0), 4)
    assert frame_metrics(g)["sharpness"] > 3 * frame_metrics(blur)["sharpness"]
    assert is_duplicate(thumb(g), thumb(g.copy()))
    assert not is_duplicate(thumb(g), thumb(255 - g))


def _texture(seed=0, w=320, h=240):
    rng = np.random.default_rng(seed)
    g = (rng.random((h // 4, w // 4)) * 255).astype(np.uint8)
    return cv2.GaussianBlur(cv2.resize(g, (w, h), interpolation=cv2.INTER_NEAREST), (3, 3), 0)


def test_motion_still_vs_moving():
    g = _texture()
    assert pair_motion(g, g.copy())["state"] == "still"
    # synthetic parallax: two depth layers moving by different amounts
    near, far = _texture(1, 640, 240), _texture(2, 640, 240)
    def view(dx):
        img = far[:, 40 + dx // 3: 40 + dx // 3 + 320].copy()
        img[120:, :] = near[120:, 40 + dx: 40 + dx + 320]
        return img
    m = pair_motion(view(0), view(24))
    assert m["state"] in ("translating", "rotating") and m["disp_frac"] > 0.0025
    c = classify([{"state": "still", "disp_frac": 0.0}] * 20)
    assert c["mode"] == "static"
    c = classify([{"state": "translating", "disp_frac": 0.02}] * 10 + [{"state": "still", "disp_frac": 0}] * 5)
    assert c["mode"] == "moving"


def test_segments_absorb_short_runs():
    pairs = [(t * 0.5, "still") for t in range(10)] + [(5.0, "translating")] + [(5.5 + t * 0.5, "still") for t in range(10)]
    seg = segments(pairs, min_len_s=2.0)
    assert len(seg) == 1 and seg[0]["state"] == "static"


def test_keyframes_follow_motion_and_respect_limit():
    samples = [{"i": i * 3, "sharpness": 50.0, "usable": True} for i in range(40)]
    pairs = [{"disp_frac": 0.01} for _ in range(39)]
    k = select_keyframes(samples, pairs, flow_frac=0.035, max_keyframes=100, min_sharpness=10)
    assert k[0] == 0 and k[-1] == 117 and 8 <= len(k) <= 14
    k2 = select_keyframes(samples, pairs, flow_frac=0.035, max_keyframes=5, min_sharpness=10)
    assert len(k2) <= 5


def test_corrupt_tail_is_tolerated(tmp_path):
    p = write_test_video(tmp_path / "c.mp4", n=36)
    data = bytearray(p.read_bytes())
    # damage bytes in the second half of the file (inside packet data)
    for k in range(len(data) // 2, len(data) // 2 + 400, 7):
        data[k] ^= 0xFF
    bad = tmp_path / "damaged.mp4"
    bad.write_bytes(bytes(data))
    src = FrameSource(bad)
    n = sum(1 for _ in src.iter_bgr())
    assert n > 0  # decoding continues past damaged packets; problems are counted, not fatal
    assert src.stats.frames == n
