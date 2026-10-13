"""Pipeline primitives: stage contract, execution context, stage records.

A stage reads the outputs of earlier stages from the case folder and writes its own
outputs there. Each run leaves a *stage record* (``derived/stages/<name>.json``) with its
status, timing, parameter fingerprint, output hashes, warnings and fallbacks. A record
whose fingerprint still matches and whose outputs are intact lets the stage be skipped
on resume.
"""
from __future__ import annotations

import hashlib
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import AnalysisConfig
from ..evidence.case import Case
from ..evidence.hashing import sha256_bytes, sha256_file
from ..log import get_logger

log = get_logger("pipeline")

STATUS_OK = "ok"                # completed as designed
STATUS_DEGRADED = "degraded"    # completed with a documented fallback or partial result
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"      # not applicable (e.g. multi-video stage with one video)
STATUS_CANCELLED = "cancelled"


class Cancelled(Exception):
    pass


@dataclass
class StageResult:
    status: str = STATUS_OK
    outputs: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    fallbacks: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    models: list[str] = field(default_factory=list)

    def warn(self, msg: str) -> None:
        log.warning(msg)
        self.warnings.append(msg)

    def fallback(self, what: str, used: str, implication: str) -> None:
        log.warning(f"Fallback in {what}: using {used}. {implication}")
        self.fallbacks.append({"component": what, "used": used, "implication": implication})
        if self.status == STATUS_OK:
            self.status = STATUS_DEGRADED


class Context:
    def __init__(self, case: Case, config: AnalysisConfig,
                 progress: Callable[[str, float, str], None] | None = None,
                 cancelled: Callable[[], bool] | None = None):
        self.case = case
        self.config = config
        self._progress = progress or (lambda s, f, m: None)
        self._cancelled = cancelled or (lambda: False)
        self.stage_name = ""
        self.cache: dict[str, Any] = {}

    def progress(self, frac: float, msg: str = "") -> None:
        self._progress(self.stage_name, max(0.0, min(1.0, float(frac))), msg)

    def check_cancel(self) -> None:
        if self._cancelled():
            raise Cancelled()

    def video_ids(self) -> list[str]:
        return [v["id"] for v in self.case.videos()]

    def vdir(self, vid: str, *parts: str) -> Path:
        return self.case.stage_dir("videos", vid, *parts)


class Stage:
    name: str = ""
    title: str = ""
    depends: tuple[str, ...] = ()
    config_keys: tuple[str, ...] = ()
    version: str = "1"  # bump when the algorithm changes so old results are not reused
    annotation_keys: tuple[str, ...] = ()  # analyst annotations whose change invalidates this stage

    def applicable(self, ctx: Context) -> bool:
        return True

    def run(self, ctx: Context) -> StageResult:  # pragma: no cover - abstract
        raise NotImplementedError

    # ------------------------------------------------------------------ records
    def fingerprint(self, ctx: Context, upstream: dict[str, dict[str, Any]]) -> str:
        parts = [self.name, self.version, ctx.config.fingerprint(list(self.config_keys)),
                 ",".join(f"{v['id']}:{v['sha256_intake']}" for v in ctx.case.videos())]
        for d in self.depends:
            parts.append(upstream.get(d, {}).get("fingerprint", "none"))
        # analyst overrides that change this stage's results
        for k in self.annotation_keys:
            ann = ctx.case.annotations(k)
            if ann:
                parts.append(f"{k}:{sha256_bytes(repr(ann).encode())[:12]}")
        return sha256_bytes("|".join(parts).encode())[:16]


def record_path(case: Case, name: str) -> Path:
    return case.root / "derived" / "stages" / f"{name}.json"


def load_record(case: Case, name: str) -> dict[str, Any] | None:
    return case.read_json(f"derived/stages/{name}.json")


def frame_fingerprint(case: Case) -> str:
    """Identity of the reconstruction frame (geometry + reconstruction + alignment results). Analyst
    points (measurements, scale references) are only meaningful in the frame they were placed in."""
    parts = [(load_record(case, n) or {}).get("fingerprint", "none") for n in ("geometry", "reconstruction",
                                                                              "alignment")]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def record_valid(case: Case, rec: dict[str, Any] | None, fingerprint: str) -> bool:
    if not rec or rec.get("status") not in (STATUS_OK, STATUS_DEGRADED, STATUS_SKIPPED):
        return False
    if rec.get("fingerprint") != fingerprint:
        return False
    for o in rec.get("outputs", []):
        p = case.root / o["path"]
        if not p.exists() or p.stat().st_size != o.get("size_bytes", -1):
            return False
    return True


def write_record(case: Case, stage: Stage, fingerprint: str, result: StageResult | None, started: float,
                 status: str | None = None, error: str | None = None) -> dict[str, Any]:
    outs = []
    if result:
        for p in result.outputs:
            p = Path(p)
            if p.exists() and p.is_file():
                outs.append({"path": str(p.relative_to(case.root)), "size_bytes": p.stat().st_size,
                             "sha256": sha256_file(p)})
    rec = {
        "name": stage.name, "title": stage.title, "version": stage.version,
        "status": status or (result.status if result else STATUS_FAILED),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started)),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "duration_s": round(time.time() - started, 2),
        "fingerprint": fingerprint,
        "outputs": outs,
        "warnings": result.warnings if result else [],
        "fallbacks": result.fallbacks if result else [],
        "metrics": result.metrics if result else {},
        "models": sorted(set(result.models)) if result else [],
        "error": error,
    }
    case.write_json(f"derived/stages/{stage.name}.json", rec)
    return rec


def format_exc() -> str:
    return traceback.format_exc(limit=8)
