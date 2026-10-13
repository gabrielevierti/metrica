"""Browser-playable review copies ("proxies") on the canonical clock.

The proxy is H.264 (+AAC audio when present) in MP4 with a short GOP so the viewer can
seek frame-accurately. Its timestamps are the canonical video times (first decoded frame
= 0 s), preserving variable frame timing. The original file is only read.
"""
from __future__ import annotations

import heapq
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np

from ..log import get_logger
from .decode import FrameSource

log = get_logger("ingest")
TB = Fraction(1, 90000)


def _encode_video(src: Path, dst: Path, max_side: int) -> tuple[dict[str, Any], float]:
    codec = "libx264" if "libx264" in av.codecs_available else ("h264" if "h264" in av.codecs_available else None)
    if codec is None:
        raise RuntimeError("No H.264 encoder available in PyAV")
    fs = FrameSource(src, max_side=max_side)
    out = av.open(str(dst), mode="w")
    vstream = None
    n = 0
    try:
        for _, t, fr, _ in fs.raw_frames():
            img = fs.to_bgr(fr)
            if vstream is None:
                h, w = img.shape[:2]
                vstream = out.add_stream(codec, rate=30)
                vstream.width, vstream.height = w // 2 * 2, h // 2 * 2
                vstream.pix_fmt = "yuv420p"
                vstream.time_base = TB
                vstream.codec_context.time_base = TB
                vstream.options = {"preset": "veryfast", "crf": "20", "g": "12", "bf": "0"}
            img = np.ascontiguousarray(img[: vstream.height, : vstream.width])
            vf = av.VideoFrame.from_ndarray(img, format="bgr24")
            vf.pts = int(round(t / TB))
            vf.time_base = TB
            for pkt in vstream.encode(vf):
                out.mux(pkt)
            n += 1
        if vstream is not None:
            for pkt in vstream.encode():
                out.mux(pkt)
    finally:
        out.close()
    return ({"codec": codec, "frames": n, "width": vstream.width if vstream else None,
             "height": vstream.height if vstream else None}, fs.stats.t0_pts_s)


def _audio_packets(src: Path, astream, t0: float) -> Iterator:
    with av.open(str(src)) as c:
        ain = c.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=48000)
        tb_in = float(ain.time_base)
        next_pts = None
        for fr in c.decode(ain):
            if fr.pts is None:
                continue
            if next_pts is None:
                next_pts = int(round((fr.pts * tb_in - t0) * 48000))
            for rf in resampler.resample(fr):
                rf.pts = next_pts
                rf.time_base = Fraction(1, 48000)
                next_pts += rf.samples
                if rf.pts < 0:
                    continue
                yield from astream.encode(rf)
        yield from astream.encode()


def make_proxy(src: Path, dst: Path, max_side: int = 1920, with_audio: bool = True) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    vtmp = dst.with_name(dst.stem + ".video.part.mp4")
    info, t0 = _encode_video(src, vtmp, max_side)
    info["audio"] = False
    has_audio = False
    if with_audio and "aac" in av.codecs_available:
        try:
            with av.open(str(src)) as c:
                has_audio = bool(c.streams.audio)
        except av.error.FFmpegError:
            has_audio = False
    if not has_audio:
        _remux(vtmp, dst)
        vtmp.unlink(missing_ok=True)
    else:
        try:
            _mux_with_audio(vtmp, src, dst, t0)
            info["audio"] = True
        except Exception as e:  # audio problems must not lose the video proxy
            log.warning(f"Audio not included in proxy for {src.name}: {e}")
            _remux(vtmp, dst)
        finally:
            vtmp.unlink(missing_ok=True)
    info["path"] = dst.name
    info["clock"] = "canonical (first decoded video frame = 0 s)"
    return info


def _remux(vsrc: Path, dst: Path) -> None:
    with av.open(str(vsrc)) as inp, av.open(str(dst), "w", options={"movflags": "+faststart"}) as out:
        vin = inp.streams.video[0]
        vout = out.add_stream_from_template(vin)
        for pkt in inp.demux(vin):
            if pkt.dts is None:
                continue
            pkt.stream = vout
            out.mux(pkt)


def _mux_with_audio(vsrc: Path, asrc: Path, dst: Path, t0: float) -> None:
    with av.open(str(vsrc)) as inp, av.open(str(dst), "w", options={"movflags": "+faststart"}) as out:
        vin = inp.streams.video[0]
        vout = out.add_stream_from_template(vin)
        aout = out.add_stream("aac", rate=48000)
        aout.layout = "stereo"

        def vpk():
            for p in inp.demux(vin):
                if p.dts is not None:
                    yield p

        def key(p, tb):
            return float(p.dts * tb) if p.dts is not None else 0.0

        vt = float(vin.time_base)
        heap: list = []
        vgen, agen = vpk(), _audio_packets(asrc, aout, t0)
        for tag, gen in (("v", vgen), ("a", agen)):
            p = next(gen, None)
            if p is not None:
                heapq.heappush(heap, (key(p, vt if tag == "v" else 1 / 48000), tag, id(p), p, gen))
        while heap:
            _, tag, _, p, gen = heapq.heappop(heap)
            if tag == "v":
                p.stream = vout
            out.mux(p)
            nxt = next(gen, None)
            if nxt is not None:
                tbv = vt if tag == "v" else float(nxt.time_base) if nxt.time_base else 1 / 48000
                heapq.heappush(heap, (key(nxt, tbv), tag, id(nxt), nxt, gen))
