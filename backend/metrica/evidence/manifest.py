"""Case manifest: what went in, what came out, how, and with which software."""
from __future__ import annotations

import importlib.metadata
import platform
import time
from pathlib import Path
from typing import Any

from .. import __version__
from .audit import verify_chain
from .case import Case
from .hashing import sha256_file

INTEGRITY_STATEMENT = (
    "SHA-256 values prove that a file's contents are bit-for-bit identical to the content that "
    "was hashed. They do not establish how, when, by whom or with which device a recording was "
    "made, whether it was edited before intake, or the chain of custody before intake: a matching "
    "hash is file-integrity verification, not proof of authenticity. Those questions require "
    "separate provenance documentation."
)

VOLATILE = {"derived/status.json"}

_PACKAGES = ["numpy", "scipy", "opencv-contrib-python-headless", "onnxruntime", "onnxruntime-gpu", "pycolmap",
             "av", "scikit-image", "trimesh", "fastapi", "reportlab", "matplotlib", "pydantic"]


def software_environment() -> dict[str, Any]:
    pkgs = {}
    for p in _PACKAGES:
        try:
            pkgs[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            continue
    from ..hardware import hardware_summary

    return {"name": "metrica", "version": __version__, "python": platform.python_version(),
            "platform": platform.platform(), "packages": pkgs, "hardware": hardware_summary()}


def build_manifest(case: Case, config: dict[str, Any] | None, models: list[dict[str, Any]],
                   stages: list[dict[str, Any]], verify_inputs: bool = True) -> dict[str, Any]:
    root = case.root
    inputs = []
    verification = {r["id"]: r for r in case.verify_originals()} if verify_inputs else {}
    for v in case.videos():
        rec = {"id": v["id"], "original_name": v["original_name"], "size_bytes": v["size_bytes"],
               "sha256_intake": v["sha256_intake"], "copied_into_case": v.get("copied", False),
               "stored_path": v.get("stored_path"), "intake_time": v["intake_time"]}
        if v["id"] in verification:
            rec["verified_at_manifest_time"] = bool(verification[v["id"]]["ok"])
        inputs.append(rec)

    outputs = []
    for sub in ("derived", "annotations"):
        base = root / sub
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            rel = str(p.relative_to(root)).replace("\\", "/")
            if rel in VOLATILE:  # live progress file, rewritten after the manifest
                continue
            if p.is_file() and not p.name.endswith(".tmp") and p.suffix != ".db-journal":
                outputs.append({"path": rel, "size_bytes": p.stat().st_size,
                                "sha256": sha256_file(p)})
    chain = verify_chain(case.audit.entries())
    return {
        "schema": "metrica-manifest/1",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "case": {"id": case.meta["id"], "title": case.meta["title"], "reference": case.meta.get("reference", "")},
        "software": software_environment(),
        "config": config,
        "models": models,
        "inputs": inputs,
        "stages": stages,
        "outputs": outputs,
        "audit": {"entries": chain["entries"], "chain_valid": chain["valid"], "head_hash": chain["head"]},
        "integrity_statement": INTEGRITY_STATEMENT,
    }


def write_manifest(case: Case, manifest: dict[str, Any]) -> tuple[Path, str]:
    path = case.write_json("manifest.json", manifest)
    return path, sha256_file(path)
