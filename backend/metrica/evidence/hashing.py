"""File hashing utilities (SHA-256, streamed: never loads a whole video into memory)."""
from __future__ import annotations

import hashlib
from pathlib import Path

CHUNK = 1 << 20


def sha256_file(path: Path | str, chunk: int = CHUNK) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_file(path: Path | str, expected: str) -> bool:
    return sha256_file(path) == expected.lower()
