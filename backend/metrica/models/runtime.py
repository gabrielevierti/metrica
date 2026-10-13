"""ONNX Runtime sessions with hardware-aware provider selection and a CPU fallback."""
from __future__ import annotations

import os
import threading

from ..hardware import onnx_providers
from ..log import get_logger
from .registry import require

log = get_logger("models")
_sessions: dict[str, object] = {}
_lock = threading.Lock()
_used: dict[str, str] = {}


def session(key: str):
    """Shared InferenceSession for a registered model (raises ModelUnavailable)."""
    import onnxruntime as ort

    with _lock:
        if key in _sessions:
            return _sessions[key]
        path = require(key)
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        threads = int(os.environ.get("METRICA_THREADS", "0"))
        if threads > 0:
            opts.intra_op_num_threads = threads
        providers = onnx_providers() or ["CPUExecutionProvider"]
        try:
            sess = ort.InferenceSession(str(path), sess_options=opts, providers=providers)
        except Exception as e:  # accelerator can reject some graphs: fall back explicitly and say so
            if providers != ["CPUExecutionProvider"]:
                log.warning(f"{key}: accelerated providers {providers} failed ({e}); falling back to CPU")
                sess = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
            else:
                raise
        _used[key] = sess.get_providers()[0]
        _sessions[key] = sess
        return sess


def providers_used() -> dict[str, str]:
    return dict(_used)


def clear() -> None:
    with _lock:
        _sessions.clear()
