"""Tamper-evident audit log of analyst actions and system events.

Each entry stores the SHA-256 of the previous entry, so deleting, reordering or editing
an entry breaks the chain and is detected by :func:`verify_chain`. This makes the log
tamper-*evident*, not tamper-*proof*: someone with write access could rewrite the whole
chain. Exporting the final chain hash (it is printed in reports) to an external record
closes that gap.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from .hashing import sha256_bytes

GENESIS = "0" * 64
_lock = threading.Lock()


def _canonical(entry: dict[str, Any]) -> bytes:
    return json.dumps(entry, sort_keys=True, separators=(",", ":"), default=str).encode()


class AuditLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def last_hash(self) -> str:
        e = self.entries()
        return e[-1]["hash"] if e else GENESIS

    def append(self, action: str, actor: str = "system", target: str | None = None,
               details: dict[str, Any] | None = None) -> dict[str, Any]:
        with _lock:
            existing = self.entries()
            prev = existing[-1]["hash"] if existing else GENESIS
            entry = {
                "seq": len(existing) + 1,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "actor": actor,
                "action": action,
                "target": target,
                "details": details or {},
                "prev_hash": prev,
            }
            entry["hash"] = sha256_bytes(_canonical(entry))
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
            return entry


def verify_chain(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Return {'valid': bool, 'entries': n, 'first_broken_seq': int|None, 'head': hash}."""
    prev = GENESIS
    for i, e in enumerate(entries):
        body = {k: v for k, v in e.items() if k != "hash"}
        if e.get("prev_hash") != prev or e.get("seq") != i + 1:
            return {"valid": False, "entries": len(entries), "first_broken_seq": i + 1, "head": None}
        if sha256_bytes(_canonical(body)) != e.get("hash"):
            return {"valid": False, "entries": len(entries), "first_broken_seq": i + 1, "head": None}
        prev = e["hash"]
    return {"valid": True, "entries": len(entries), "first_broken_seq": None, "head": prev}
