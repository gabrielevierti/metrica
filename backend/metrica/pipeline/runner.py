"""Pipeline orchestration: ordered stages, resume, cancellation, status file, manifest."""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from ..config import AnalysisConfig
from ..evidence.case import Case
from ..evidence.manifest import build_manifest, write_manifest
from ..log import attach_case_log, detach_handler, get_logger
from ..models.registry import model_record
from .base import (
    STATUS_CANCELLED,
    STATUS_DEGRADED,
    STATUS_FAILED,
    STATUS_OK,
    STATUS_SKIPPED,
    Cancelled,
    Context,
    Stage,
    format_exc,
    load_record,
    record_valid,
    write_record,
)

log = get_logger("pipeline")

# Stages whose failure makes everything after them meaningless.
CRITICAL = {"ingest"}


def all_stages() -> list[Stage]:
    from .stages.alignment import AlignmentStage
    from .stages.analysis import AnalysisStage
    from .stages.depth import DepthSemanticsStage
    from .stages.geometry import GeometryStage
    from .stages.ingest import IngestStage
    from .stages.objects import ObjectsStage
    from .stages.reconstruction import ReconstructionStage
    from .stages.report import ReportStage
    from .stages.scene import SceneStage
    from .stages.tracking import DetectionTrackingStage
    from .stages.trajectories import TrajectoryStage

    return [IngestStage(), DetectionTrackingStage(), DepthSemanticsStage(), GeometryStage(),
            ReconstructionStage(), ObjectsStage(), TrajectoryStage(), AlignmentStage(), AnalysisStage(),
            SceneStage(), ReportStage()]


def stage_titles() -> list[dict[str, str]]:
    return [{"name": s.name, "title": s.title} for s in all_stages()]


class StatusWriter:
    def __init__(self, case: Case, stages: list[Stage]):
        self.case = case
        self.stages = stages
        self.state: dict[str, Any] = {"state": "running", "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                      "stages": [{"name": s.name, "title": s.title, "status": "pending"}
                                                 for s in stages],
                                      "progress": 0.0, "message": "", "warnings": 0}
        self._last = 0.0
        self.flush(force=True)

    def stage_status(self, name: str, status: str, **extra: Any) -> None:
        for s in self.state["stages"]:
            if s["name"] == name:
                s["status"] = status
                s.update(extra)
        self.flush(force=True)

    def progress(self, stage: str, frac: float, msg: str) -> None:
        idx = next((k for k, s in enumerate(self.stages) if s.name == stage), 0)
        self.state["stage"] = stage
        self.state["stage_progress"] = round(frac, 4)
        self.state["progress"] = round((idx + frac) / max(1, len(self.stages)), 4)
        self.state["message"] = msg
        self.flush()

    def flush(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last < 0.5:
            return
        self._last = now
        self.state["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.case.write_json("derived/status.json", self.state)


def run_pipeline(case: Case, config: AnalysisConfig, progress: Callable[[str, float, str], None] | None = None,
                 cancelled: Callable[[], bool] | None = None, force: bool | set[str] = False,
                 only: list[str] | None = None, actor: str = "system") -> dict[str, Any]:
    stages = all_stages()
    if only:
        stages = [s for s in stages if s.name in only]
    handler = attach_case_log(case.root / "logs" / "pipeline.jsonl")
    status = StatusWriter(case, stages)

    def prog(stage: str, frac: float, msg: str) -> None:
        status.progress(stage, frac, msg)
        if progress:
            progress(stage, frac, msg)

    ctx = Context(case, config, prog, cancelled)
    # evidence first, on every run (also when every stage would be reused): a changed original stops it
    bad = [r for r in case.verify_originals() if not r["ok"]]
    if bad:
        status.state.update({"state": "failed", "message": "evidence verification failed: " + ", ".join(
            f"{r['id']} ({r.get('reason', 'hash mismatch')})" for r in bad)})
        status.flush(force=True)
        case.audit.append("analysis.refused", actor=actor, details={"reason": "evidence verification failed",
                                                                    "videos": [r["id"] for r in bad]})
        detach_handler(handler)
        return status.state
    case.write_json("derived/config.json", config.to_dict())
    case.audit.append("analysis.started", actor=actor,
                      details={"profile": config.profile, "config_fingerprint": config.fingerprint()})
    records: dict[str, dict[str, Any]] = {}
    for s in all_stages():  # upstream records for fingerprints, even if not re-run
        r = load_record(case, s.name)
        if r:
            records[s.name] = r
    outcome = "completed"
    failed: set[str] = set()
    t_start = time.time()
    try:
        for stage in stages:
            ctx.stage_name = stage.name
            fp = stage.fingerprint(ctx, records)
            forced = force is True or (isinstance(force, set) and stage.name in force)
            if any(d in failed for d in stage.depends):
                rec = write_record(case, stage, fp, None, time.time(), status=STATUS_FAILED,
                                   error=f"upstream stage failed: {sorted(set(stage.depends) & failed)}")
                records[stage.name] = rec
                failed.add(stage.name)
                status.stage_status(stage.name, STATUS_FAILED, error=rec["error"])
                continue
            if not forced and record_valid(case, records.get(stage.name), fp):
                log.info(f"[{stage.name}] up to date - reusing previous results")
                status.stage_status(stage.name, records[stage.name]["status"], reused=True)
                continue
            if not stage.applicable(ctx):
                rec = write_record(case, stage, fp, None, time.time(), status=STATUS_SKIPPED)
                records[stage.name] = rec
                status.stage_status(stage.name, STATUS_SKIPPED)
                continue
            log.info(f"[{stage.name}] {stage.title}")
            status.stage_status(stage.name, "running")
            prog(stage.name, 0.0, "starting")
            started = time.time()
            try:
                result = stage.run(ctx)
                rec = write_record(case, stage, fp, result, started)
            except Cancelled:
                write_record(case, stage, fp, None, started, status=STATUS_CANCELLED, error="cancelled by user")
                status.stage_status(stage.name, STATUS_CANCELLED)
                raise
            except Exception as e:  # stage failure: record it, continue with independent stages
                log.error(f"[{stage.name}] failed: {e}")
                rec = write_record(case, stage, fp, None, started, status=STATUS_FAILED,
                                   error=f"{type(e).__name__}: {e}\n{format_exc()}")
            records[stage.name] = rec
            if rec["status"] == STATUS_FAILED:
                failed.add(stage.name)
                if stage.name in CRITICAL:
                    status.stage_status(stage.name, STATUS_FAILED, error=rec["error"])
                    outcome = "failed"
                    break
            status.state["warnings"] += len(rec.get("warnings", []))
            status.stage_status(stage.name, rec["status"], duration_s=rec["duration_s"],
                                warnings=len(rec.get("warnings", [])), error=rec.get("error"))
    except Cancelled:
        outcome = "cancelled"
    finally:
        try:
            all_recs = [records[s.name] for s in all_stages() if s.name in records]
            used_models = sorted({m for r in all_recs for m in r.get("models", [])})
            models = []
            for m in used_models:
                for key in m.split("+"):
                    try:
                        models.append(model_record(key.replace("ppocr_det_en", "ppocr_det_en")))
                    except KeyError:
                        models.append({"key": key})
            manifest = build_manifest(case, config.to_dict(), models, all_recs)
            _, mhash = write_manifest(case, manifest)
        except Exception as e:  # pragma: no cover
            log.error(f"Manifest could not be written: {e}")
            mhash = None
        if outcome == "completed" and failed:
            outcome = "completed_with_errors"
        status.state["state"] = outcome
        status.state["progress"] = 1.0 if outcome.startswith("completed") else status.state["progress"]
        status.state["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        status.state["duration_s"] = round(time.time() - t_start, 1)
        status.state["manifest_sha256"] = mhash
        status.flush(force=True)
        case.audit.append("analysis.finished", actor=actor,
                          details={"outcome": outcome, "failed_stages": sorted(failed), "manifest_sha256": mhash})
        detach_handler(handler)
    return status.state


def stage_statuses(case: Case) -> list[dict[str, Any]]:
    out = []
    for s in all_stages():
        r = load_record(case, s.name) or {}
        out.append({"name": s.name, "title": s.title, "status": r.get("status", "pending"),
                    "duration_s": r.get("duration_s"), "warnings": r.get("warnings", []),
                    "fallbacks": r.get("fallbacks", []), "error": r.get("error"), "metrics": r.get("metrics", {})})
    return out


__all__ = ["run_pipeline", "all_stages", "stage_statuses", "STATUS_OK", "STATUS_DEGRADED"]
