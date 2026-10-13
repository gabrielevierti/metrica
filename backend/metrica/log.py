"""Structured logging.

Every pipeline message is written twice: a human-readable line on the console and a
JSON line in the case's ``logs/pipeline.jsonl``. The JSON log is part of the audit trail:
it records warnings, fallbacks and failures exactly as they happened.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

_LOGGER_NAME = "metrica"


class JsonLineHandler(logging.Handler):
    def __init__(self, path: Path):
        super().__init__()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    def emit(self, record: logging.LogRecord) -> None:
        entry: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created))
            + f".{int(record.msecs):03d}",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "data", None)
        if extra:
            entry["data"] = extra
        try:
            self._fh.write(json.dumps(entry, default=str) + "\n")
            self._fh.flush()
        except Exception:  # pragma: no cover - logging must never crash the pipeline
            pass

    def close(self) -> None:
        try:
            self._fh.close()
        finally:
            super().close()


def get_logger(name: str | None = None) -> logging.Logger:
    base = logging.getLogger(_LOGGER_NAME)
    if not base.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
        base.addHandler(h)
        base.setLevel(logging.INFO)
        base.propagate = False
    return base if name is None else base.getChild(name)


def attach_case_log(path: Path) -> JsonLineHandler:
    handler = JsonLineHandler(path)
    get_logger().addHandler(handler)
    return handler


def detach_handler(handler: logging.Handler) -> None:
    get_logger().removeHandler(handler)
    handler.close()
