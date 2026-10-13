"""Hardware detection: CPU, memory, and ONNX Runtime execution providers."""
from __future__ import annotations

import os
import platform
from functools import lru_cache
from typing import Any

# Preferred order. CPU is always last and always available.
_PREFERRED = ["CUDAExecutionProvider", "CoreMLExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]


def total_memory_gb() -> float | None:
    try:
        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError):
        pass
    if platform.system() == "Windows":  # pragma: no cover - platform specific
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return st.ullTotalPhys / 1e9
    return None


@lru_cache(maxsize=1)
def onnx_providers() -> list[str]:
    try:
        import onnxruntime as ort

        avail = ort.get_available_providers()
    except Exception:  # pragma: no cover
        return []
    disabled = {p.strip() for p in os.environ.get("METRICA_DISABLE_PROVIDERS", "").split(",") if p.strip()}
    # accepts full names (CUDAExecutionProvider) or short ones (CUDA, CoreML, Dml)
    return [p for p in _PREFERRED if p in avail and p not in disabled
            and p.replace("ExecutionProvider", "") not in disabled]


def hardware_summary() -> dict[str, Any]:
    mach = platform.machine().lower()
    providers = onnx_providers()
    accel = [p for p in providers if p != "CPUExecutionProvider"]
    return {
        "os": platform.system(),
        "machine": mach,
        "apple_silicon": platform.system() == "Darwin" and mach in ("arm64", "aarch64"),
        "cpu_count": os.cpu_count(),
        "memory_gb": round(total_memory_gb() or 0, 1) or None,
        "onnx_providers": providers,
        "accelerated": bool(accel),
        "accelerator": accel[0] if accel else None,
        "notes": ("No GPU execution provider found: neural models run on the CPU. Depth, segmentation and "
                  "detection will be slower but produce the same results.") if not accel else "",
    }
