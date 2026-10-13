"""Worker process: runs one queued job (`python -m metrica.pipeline.worker --job ID --db PATH`)."""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
import traceback
from pathlib import Path

from ..config import AnalysisConfig
from ..evidence.case import Case
from ..log import get_logger
from .runner import run_pipeline

log = get_logger("worker")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--db", required=True)
    a = ap.parse_args()
    db = Path(a.db)

    def q(sql, args=()):
        with sqlite3.connect(db, timeout=10) as c:
            c.row_factory = sqlite3.Row
            return c.execute(sql, args).fetchone()

    def upd(**f):
        with sqlite3.connect(db, timeout=10) as c:
            c.execute(f"UPDATE jobs SET {', '.join(k + '=?' for k in f)} WHERE id=?", (*f.values(), a.job))

    job = q("SELECT * FROM jobs WHERE id=?", (a.job,))
    if job is None:
        return 2
    last = [0.0, False]

    def cancelled() -> bool:
        now = time.time()
        if now - last[0] > 1.0:
            r = q("SELECT status FROM jobs WHERE id=?", (a.job,))
            last[1] = bool(r and r["status"] == "cancel_requested")
            last[0] = now
        return last[1]

    try:
        case = Case(Path(job["case_path"]))
        cfg = AnalysisConfig.from_dict(json.loads(job["config"] or "{}"))
        state = run_pipeline(case, cfg, cancelled=cancelled, force=bool(job["force"]), actor=job["actor"] or "analyst")
        status = {"completed": "completed", "completed_with_errors": "completed_with_errors",
                  "cancelled": "cancelled"}.get(state["state"], "failed")
        upd(status=status, finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        return 0
    except Exception as e:  # noqa: BLE001
        log.error(traceback.format_exc())
        upd(status="failed", finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"), error=f"{type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
