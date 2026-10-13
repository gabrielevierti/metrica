"""Case folders: the unit of work, storage and evidence handling.

Layout (all paths relative to the case root)::

    case.json                 case identity, title, video inventory with intake hashes
    evidence/originals/       byte-exact copies of the imported files (read-only)
    derived/                  everything METRICA computes (re-creatable from evidence/)
    annotations/              analyst input: measurements, notes, verifications, overrides
    audit/audit.jsonl         hash-chained log of analyst actions and system events
    logs/pipeline.jsonl       structured processing log
    manifest.json             inventory of inputs and outputs with SHA-256, versions, parameters

Originals are never modified. They are hashed *before* copying, the copy is re-hashed
and compared, and the copy is made read-only. When ``copy_originals`` is off, the
original is referenced in place and re-verified each time processing starts.
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import time
import uuid
from pathlib import Path
from typing import Any

from .. import CASE_SCHEMA, __version__
from .audit import AuditLog
from .hashing import sha256_file

ANNOTATION_KINDS = ("measurements", "notes", "verifications", "calibration", "sync", "alignment",
                    "face_review", "snapshots")


def default_workspace() -> Path:
    return Path(os.environ.get("METRICA_HOME", Path.home() / "METRICA")).expanduser()


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, default=str)
    os.replace(tmp, path)


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class Case:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        if not (self.root / "case.json").exists():
            raise FileNotFoundError(f"Not a METRICA case: {self.root}")
        self.audit = AuditLog(self.root / "audit" / "audit.jsonl")

    # ------------------------------------------------------------------ creation / io
    @classmethod
    def create(cls, root: Path, title: str, case_ref: str | None = None) -> Case:
        root = Path(root).resolve()
        if (root / "case.json").exists():
            raise FileExistsError(f"A case already exists at {root}")
        for d in ("evidence/originals", "derived", "annotations", "audit", "logs"):
            (root / d).mkdir(parents=True, exist_ok=True)
        meta = {
            "schema": CASE_SCHEMA,
            "id": uuid.uuid4().hex[:12],
            "reference": case_ref or "",
            "title": title,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "software_version": __version__,
            "videos": [],
        }
        _write_json(root / "case.json", meta)
        case = cls(root)
        case.audit.append("case.created", details={"title": title, "reference": case_ref or ""})
        return case

    @property
    def meta(self) -> dict[str, Any]:
        return _read_json(self.root / "case.json")

    def save_meta(self, meta: dict[str, Any]) -> None:
        _write_json(self.root / "case.json", meta)

    @property
    def id(self) -> str:
        return self.meta["id"]

    @property
    def derived(self) -> Path:
        return self.root / "derived"

    def stage_dir(self, *parts: str) -> Path:
        p = self.derived.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ------------------------------------------------------------------ evidence intake
    def add_video(self, src: Path, copy: bool = True, actor: str = "analyst") -> dict[str, Any]:
        src = Path(src)
        if not src.is_file():
            raise FileNotFoundError(src)
        meta = self.meta
        vid = f"v{len(meta['videos']) + 1}"
        intake_hash = sha256_file(src)
        for v in meta["videos"]:
            if v["sha256_intake"] == intake_hash:
                raise ValueError(f"This file is already in the case as {v['id']} ({v['original_name']}).")
        record: dict[str, Any] = {
            "id": vid,
            "original_name": src.name,
            "original_path": str(src.resolve()),
            "size_bytes": src.stat().st_size,
            "sha256_intake": intake_hash,
            "intake_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "copied": bool(copy),
            "source_mtime": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(src.stat().st_mtime)),
        }
        if copy:
            dst_dir = self.root / "evidence" / "originals"
            dst = dst_dir / f"{vid}__{src.name}"
            shutil.copyfile(src, dst)
            stored_hash = sha256_file(dst)
            if stored_hash != intake_hash:
                dst.unlink(missing_ok=True)
                raise OSError(f"Copy verification failed for {src.name}: hash mismatch after copy.")
            os.chmod(dst, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            record["stored_path"] = str(dst.relative_to(self.root))
            record["sha256_stored"] = stored_hash
        meta["videos"].append(record)
        self.save_meta(meta)
        self.audit.append("evidence.video_added", actor=actor, target=vid,
                          details={k: record[k] for k in ("original_name", "sha256_intake", "size_bytes", "copied")})
        return record

    def videos(self) -> list[dict[str, Any]]:
        return self.meta["videos"]

    def video(self, vid: str) -> dict[str, Any]:
        for v in self.videos():
            if v["id"] == vid:
                return v
        raise KeyError(vid)

    def video_path(self, vid: str) -> Path:
        v = self.video(vid)
        if v.get("copied"):
            return self.root / v["stored_path"]
        return Path(v["original_path"])

    def update_video(self, vid: str, **fields: Any) -> None:
        meta = self.meta
        for v in meta["videos"]:
            if v["id"] == vid:
                v.update(fields)
        self.save_meta(meta)

    def verify_originals(self) -> list[dict[str, Any]]:
        """Re-hash every evidence file and compare with the intake hash."""
        results = []
        for v in self.videos():
            p = self.video_path(v["id"])
            if not p.exists():
                results.append({"id": v["id"], "ok": False, "reason": "file missing", "path": str(p)})
                continue
            h = sha256_file(p)
            results.append({"id": v["id"], "ok": h == v["sha256_intake"], "sha256": h,
                            "expected": v["sha256_intake"], "path": str(p)})
        return results

    # ------------------------------------------------------------------ annotations
    def annotations(self, kind: str) -> Any:
        if kind not in ANNOTATION_KINDS:
            raise ValueError(f"Unknown annotation kind {kind}")
        default: Any = {} if kind in ("calibration", "sync", "alignment") else []
        return _read_json(self.root / "annotations" / f"{kind}.json", default)

    def save_annotations(self, kind: str, data: Any, actor: str, action: str,
                         details: dict[str, Any] | None = None) -> None:
        if kind not in ANNOTATION_KINDS:
            raise ValueError(f"Unknown annotation kind {kind}")
        _write_json(self.root / "annotations" / f"{kind}.json", data)
        self.audit.append(action, actor=actor, target=kind, details=details)

    # ------------------------------------------------------------------ helpers
    def read_json(self, rel: str, default: Any = None) -> Any:
        return _read_json(self.root / rel, default)

    def write_json(self, rel: str, data: Any) -> Path:
        p = self.root / rel
        _write_json(p, data)
        return p


def list_cases(workspace: Path) -> list[dict[str, Any]]:
    out = []
    if not workspace.exists():
        return out
    for d in sorted(workspace.iterdir()):
        cj = d / "case.json"
        if cj.exists():
            try:
                m = _read_json(cj)
                status = _read_json(d / "derived" / "status.json", {})
                out.append({"path": str(d), "id": m["id"], "title": m["title"], "created": m["created"],
                            "reference": m.get("reference", ""), "videos": len(m["videos"]),
                            "status": status.get("state", "new")})
            except Exception:  # corrupted case.json: list it so the user can see it
                out.append({"path": str(d), "id": d.name, "title": d.name, "created": "", "videos": 0,
                            "status": "unreadable"})
    return out
