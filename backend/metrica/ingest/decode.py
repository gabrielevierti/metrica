"""Frame access with exact presentation timestamps (PyAV / FFmpeg libraries).

Design rules
------------
* Frame numbers are positions in *presentation order of successfully decoded frames*.
  Their timestamps come from the container (pts x time_base), so variable-frame-rate
  video is handled correctly: METRICA never assumes ``t = index / fps``.
* Canonical video time ``t`` is measured from the first decoded video frame
  (``t = pts_seconds - t0``). Proxies are encoded on the same clock.
* Corrupted packets are skipped and counted, never fatal.
* Display-rotation metadata (phones) is applied, so frames are upright.
* Frames are streamed; a long video is never loaded into memory.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import av
import cv2
import numpy as np

from ..log import get_logger

log = get_logger("ingest")


def rotation_code(rotation_deg: float | int | None) -> int | None:
    """OpenCV rotate code that makes a frame upright, from PyAV's display rotation (CCW deg)."""
    if not rotation_deg:
        return None
    theta = int(round(-float(rotation_deg))) % 360
    if 45 <= theta < 135:
        return cv2.ROTATE_90_CLOCKWISE
    if 135 <= theta < 225:
        return cv2.ROTATE_180
    if 225 <= theta < 315:
        return cv2.ROTATE_90_COUNTERCLOCKWISE
    return None


@dataclass
class DecodeStats:
    frames: int = 0
    decode_errors: int = 0
    nonmonotonic_pts: int = 0
    missing_pts: int = 0
    rotation: float = 0.0
    t0_pts_s: float = 0.0


class FrameSource:
    """Sequential decoder yielding frames in presentation order with canonical timestamps."""

    def __init__(self, path: Path | str, max_side: int | None = None):
        self.path = str(path)
        self.max_side = max_side
        self.stats = DecodeStats()
        self._rot: int | None = None
        with av.open(self.path) as c:
            if not c.streams.video:
                raise ValueError(f"No video stream in {self.path}")

    # ------------------------------------------------------------------ conversions
    def to_bgr(self, frame: av.VideoFrame, max_side: int | None = None) -> np.ndarray:
        img = frame.to_ndarray(format="bgr24")
        if self._rot is not None:
            img = cv2.rotate(img, self._rot)
        ms = max_side or self.max_side
        if ms and max(img.shape[:2]) > ms:
            s = ms / max(img.shape[:2])
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        return img

    def to_gray(self, frame: av.VideoFrame, max_side: int) -> np.ndarray:
        w, h = frame.width, frame.height
        s = min(1.0, max_side / max(w, h))
        small = frame.reformat(width=max(2, int(w * s)) // 2 * 2, height=max(2, int(h * s)) // 2 * 2,
                               format="gray")
        img = small.to_ndarray()
        if self._rot is not None:
            img = cv2.rotate(img, self._rot)
        return img

    # ------------------------------------------------------------------ iteration
    def raw_frames(self) -> Iterator[tuple[int, float, av.VideoFrame, bool]]:
        """Yield (index, t, av_frame, is_keyframe). The av_frame is valid only during the step."""
        self.stats = DecodeStats()
        with av.open(self.path) as c:
            vs = c.streams.video[0]
            vs.thread_type = "AUTO"
            tb = float(vs.time_base) if vs.time_base else 1.0 / 25.0
            fps_guess = float(vs.average_rate) if vs.average_rate else 25.0
            idx, last, t0 = 0, None, None
            for packet in c.demux(vs):
                try:
                    decoded = packet.decode()
                except (av.error.InvalidDataError, av.error.ValueError, OSError) as e:
                    self.stats.decode_errors += 1
                    if self.stats.decode_errors <= 5:
                        log.warning(f"Decode error skipped in {Path(self.path).name}: {e}")
                    continue
                for fr in decoded:
                    if fr.pts is None:
                        self.stats.missing_pts += 1
                        ts = (last + 1.0 / fps_guess) if last is not None else 0.0
                    else:
                        ts = fr.pts * tb
                    if last is not None and ts <= last:
                        self.stats.nonmonotonic_pts += 1
                        ts = last + 1e-4
                    last = ts
                    if t0 is None:
                        t0 = ts
                        self.stats.t0_pts_s = ts
                        self.stats.rotation = float(getattr(fr, "rotation", 0) or 0)
                        self._rot = rotation_code(self.stats.rotation)
                    yield idx, ts - t0, fr, bool(getattr(fr, "key_frame", False))
                    idx += 1
            self.stats.frames = idx

    def iter_bgr(self, want: set[int] | None = None, every: int | None = None,
                 stop_after: int | None = None) -> Iterator[tuple[int, float, np.ndarray]]:
        for i, t, fr, _ in self.raw_frames():
            if (want is None and every is None) or (want is not None and i in want) or \
                    (every is not None and i % every == 0):
                yield i, t, self.to_bgr(fr)
            if stop_after is not None and i >= stop_after:
                break


def read_frames_at(path: Path | str, indices: list[int] | set[int],
                   max_side: int | None = None) -> dict[int, np.ndarray]:
    """Decode specific frames (canonical indices) in one sequential pass."""
    want = {int(i) for i in indices}
    if not want:
        return {}
    src = FrameSource(path, max_side=max_side)
    return {i: img for i, _, img in src.iter_bgr(want=want, stop_after=max(want))}
