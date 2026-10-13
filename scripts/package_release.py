#!/usr/bin/env python3
"""Build the METRICA release archive: metrica-<version>.zip with a top-level ``metrica/`` folder.

* rebuilds the frontend when npm is available (``--no-frontend-build`` to skip; dist must exist);
* copies the repository without caches, virtual environments, node_modules, models or footage;
* writes RELEASE_SHA256SUMS.txt (SHA-256 of every file in the archive);
* verifies the archive afterwards (CRC test, hashes, required files).

Usage: python scripts/package_release.py [--out DIR] [--no-frontend-build]
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", ".ruff_cache",
                ".mypy_cache", "build", ".idea", ".vscode"}
EXCLUDE_GLOBS = ["*.pyc", "*.pyo", "*.egg-info", "*.onnx", "*.mp4", "*.mov", "*.avi", "*.mkv", "*.mts", "*.webm",
                 ".DS_Store", "Thumbs.db", "*.zip", ".env", "*.key", "*.pem"]
REQUIRED = ["README.md", "INSTALL.md", "LICENSE", "NOTICE", "CONTRIBUTING.md", "CHANGELOG.md", "pyproject.toml",
            "requirements-lock.txt", "docs/USER_GUIDE.md", "docs/ARCHITECTURE.md", "docs/RECONSTRUCTION_METHODS.md",
            "docs/EVALUATION.md", "docs/PRIVACY_AND_EVIDENCE.md", "docs/MODEL_LICENSES.md",
            "backend/metrica/__init__.py", "frontend/dist/index.html", "scripts/install.sh", "scripts/install.ps1"]


def version() -> str:
    m = re.search(r'__version__\s*=\s*"([^"]+)"', (ROOT / "backend/metrica/__init__.py").read_text())
    return m.group(1) if m else "0.0.0"


def excluded(rel: Path) -> bool:
    if any(part in EXCLUDE_DIRS for part in rel.parts[:-1]) or rel.name in EXCLUDE_DIRS:
        # frontend/dist is a release artefact even though "build"-like folders are excluded
        return True
    if any(part.endswith(".egg-info") for part in rel.parts):
        return True
    return any(fnmatch.fnmatch(rel.name, g) for g in EXCLUDE_GLOBS)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT.parent))
    ap.add_argument("--no-frontend-build", action="store_true")
    a = ap.parse_args()
    if not a.no_frontend_build and shutil.which("npm"):
        print("building frontend ...")
        subprocess.run(["npm", "ci"], cwd=ROOT / "frontend", check=True)
        subprocess.run(["npm", "run", "build"], cwd=ROOT / "frontend", check=True)
    if not (ROOT / "frontend/dist/index.html").exists():
        print("frontend/dist is missing: build the frontend first", file=sys.stderr)
        return 1
    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and not excluded(p.relative_to(ROOT)))
    missing = [r for r in REQUIRED if not (ROOT / r).exists()]
    if missing:
        print(f"missing required files: {missing}", file=sys.stderr)
        return 1
    ver = version()
    out = Path(a.out).resolve() / f"metrica-{ver}.zip"
    # the built web app is also placed inside the Python package (backend/metrica/api/static), so that
    # non-editable installs (pip install ., wheels) serve it too
    entries = [(p, p.relative_to(ROOT).as_posix()) for p in files]
    dist = ROOT / "frontend" / "dist"
    entries += [(p, "backend/metrica/api/static/" + p.relative_to(dist).as_posix())
                for p in sorted(dist.rglob("*")) if p.is_file()]
    sums = [(sha256(p), r) for p, r in entries]
    sums_text = "".join(f"{h}  metrica/{r}\n" for h, r in sums)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p, r in entries:
            info = zipfile.ZipInfo.from_file(p, f"metrica/{r}")
            info.compress_type = zipfile.ZIP_DEFLATED
            with open(p, "rb") as f:
                z.writestr(info, f.read())
        si = zipfile.ZipInfo("metrica/RELEASE_SHA256SUMS.txt", date_time=time.localtime()[:6])
        si.compress_type = zipfile.ZIP_DEFLATED
        si.external_attr = 0o644 << 16
        z.writestr(si, sums_text)
    # ---- verification
    with zipfile.ZipFile(out) as z:
        bad = z.testzip()
        if bad:
            print(f"CRC error in {bad}", file=sys.stderr)
            return 1
        names = set(z.namelist())
        for h, r in sums:
            if hashlib.sha256(z.read(f"metrica/{r}")).hexdigest() != h:
                print(f"hash mismatch for {r}", file=sys.stderr)
                return 1
        for r in REQUIRED:
            if f"metrica/{r}" not in names:
                print(f"required file missing from archive: {r}", file=sys.stderr)
                return 1
        forbidden = [n for n in names if any(fnmatch.fnmatch(Path(n).name, g) for g in EXCLUDE_GLOBS)]
        if forbidden:
            print(f"forbidden files in archive: {forbidden[:5]}", file=sys.stderr)
            return 1
    size = out.stat().st_size
    print(f"{out}  ({len(entries) + 1} files, {size / 1e6:.2f} MB)\nSHA-256 {sha256(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
