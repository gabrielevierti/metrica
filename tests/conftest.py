"""Shared fixtures. Test data is generated on the fly (no recordings are stored in the repository)."""
from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "backend") not in sys.path:  # allow running without an editable install
    sys.path.insert(0, str(ROOT / "backend"))


def write_test_video(path: Path, n: int = 24, w: int = 160, h: int = 96, fps: float = 12.0,
                     shift_px: float = 0.0, seed: int = 0, rotate: int | None = None) -> Path:
    """Small H.264/MPEG-4 clip of a textured pattern (optionally translating)."""
    import av

    rng = np.random.default_rng(seed)
    base = (rng.random((h * 2, w * 2, 3)) * 255).astype(np.uint8)
    base = np.kron(base[::4, ::4], np.ones((4, 4, 1), np.uint8))[: h * 2, : w * 2]
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as out:
        codec = "libx264" if "libx264" in av.codecs_available else "mpeg4"
        st = out.add_stream(codec, rate=Fraction(fps).limit_denominator(1000))
        st.width, st.height, st.pix_fmt = w, h, "yuv420p"
        if rotate is not None:
            try:
                st.metadata["rotate"] = str(rotate)
            except Exception:  # noqa: BLE001
                pass
        for i in range(n):
            x0 = int(round(i * shift_px)) % w
            frame = np.ascontiguousarray(base[: h, x0: x0 + w])
            for p in st.encode(av.VideoFrame.from_ndarray(frame, format="bgr24")):
                out.mux(p)
        for p in st.encode():
            out.mux(p)
    return path


@pytest.fixture
def tiny_video(tmp_path: Path) -> Path:
    return write_test_video(tmp_path / "src" / "tiny.mp4")


@pytest.fixture
def scene_stub() -> dict:
    """Minimal scene graph for measurement tests (one static camera at 5 m height)."""
    def scene(status: str = "ESTIMATED_METRIC", unit: str = "m", sigma: float | None = 0.05):
        return {
            "coordinate_system": {"unit": unit, "status": status,
                                  "scale": {"scale_sigma_rel": sigma}, "global_scale_factor": 1.0},
            "cameras": [{"kind": "static", "world_from_cam": [[1, 0, 0, 0], [0, 1, 0, 5], [0, 0, 1, 0], [0, 0, 0, 1]]}],
            "agents": [],
        }
    return scene


def models_available(*keys: str) -> bool:
    try:
        from metrica.models import registry
    except Exception:  # noqa: BLE001
        return False
    return all(registry.available(k) for k in keys)
