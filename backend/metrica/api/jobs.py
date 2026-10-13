"""Persistent job queue (SQLite) with subprocess workers.

Each analysis runs in its own Python process (`python -m metrica.pipeline.worker`), so a
crash or memory exhaustion never takes down the application, and cancellation can always
be enforced. Jobs survive application restarts (a job found 'running' with a dead worker
is marked 'interrupted' and can be resumed: completed stages are reused).
"""
from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

SCHEMA = """CREATE TABLE IF NOT EXISTS jobs (
 id TEXT PRIMARY KEY, case_id TEXT, case_path TEXT, kind TEXT, status TEXT, config TEXT, force INTEGER,
 created TEXT, started TEXT, finished TEXT, pid INTEGER, error TEXT, actor TEXT)"""
ACTIVE = ("queued", "running", "cancel_requested")


class JobStore:
    def __init__(self, db: Path):
        self.db = Path(db)
        self.db.parent.mkdir(parents=True, exist_ok=True)
        with self._c() as c:
            c.execute(SCHEMA)
        self._lock = threading.Lock()
        self._recover()
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _c(self):
        c = sqlite3.connect(self.db, timeout=10)
        c.row_factory = sqlite3.Row
        return c

    def _recover(self) -> None:
        with self._c() as c:
            for r in c.execute("SELECT * FROM jobs WHERE status IN ('running','cancel_requested')").fetchall():
                if not _alive(r["pid"]):
                    c.execute("UPDATE jobs SET status='interrupted', finished=? WHERE id=?", (_now(), r["id"]))

    def submit(self, case_id: str, case_path: Path, config: dict[str, Any], force: bool = False,
               kind: str = "analysis", actor: str = "analyst") -> dict[str, Any]:
        jid = uuid.uuid4().hex[:10]
        with self._c() as c:
            c.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (jid, case_id, str(case_path), kind, "queued", json.dumps(config), int(force), _now(), None, None,
                       None, None, actor))
        self._kick()
        return self.get(jid)

    def get(self, jid: str) -> dict[str, Any] | None:
        with self._c() as c:
            r = c.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
        return dict(r) if r else None

    def list(self, case_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        with self._c() as c:
            if case_id:
                rows = c.execute("SELECT * FROM jobs WHERE case_id=? ORDER BY created DESC LIMIT ?", (case_id, limit))
            else:
                rows = c.execute("SELECT * FROM jobs ORDER BY created DESC LIMIT ?", (limit,))
            return [dict(r) for r in rows.fetchall()]

    def active_for(self, case_id: str) -> dict[str, Any] | None:
        with self._c() as c:
            r = c.execute(f"SELECT * FROM jobs WHERE case_id=? AND status IN {ACTIVE} ORDER BY created DESC",
                          (case_id,)).fetchone()
        return dict(r) if r else None

    def cancel(self, jid: str) -> dict[str, Any] | None:
        j = self.get(jid)
        if not j:
            return None
        with self._c() as c:
            if j["status"] == "queued":
                c.execute("UPDATE jobs SET status='cancelled', finished=? WHERE id=?", (_now(), jid))
            elif j["status"] == "running":
                c.execute("UPDATE jobs SET status='cancel_requested' WHERE id=?", (jid,))
        threading.Timer(45.0, self._force_kill, args=(jid,)).start()
        return self.get(jid)

    def set(self, jid: str, **fields: Any) -> None:
        keys = ", ".join(f"{k}=?" for k in fields)
        with self._c() as c:
            c.execute(f"UPDATE jobs SET {keys} WHERE id=?", (*fields.values(), jid))

    def _force_kill(self, jid: str) -> None:
        j = self.get(jid)
        if j and j["status"] == "cancel_requested" and _alive(j["pid"]):
            try:
                os.kill(j["pid"], signal.SIGTERM)
            except OSError:
                pass
            self.set(jid, status="cancelled", finished=_now(), error="worker terminated after cancel timeout")

    def _kick(self) -> None:
        with self._lock:
            with self._c() as c:
                running = c.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('running','cancel_requested')").fetchone()[0]
                if running:
                    return
                nxt = c.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created LIMIT 1").fetchone()
                if not nxt:
                    return
                c.execute("UPDATE jobs SET status='running', started=? WHERE id=?", (_now(), nxt["id"]))
            cmd = [sys.executable, "-m", "metrica.pipeline.worker", "--job", nxt["id"], "--db", str(self.db)]
            log = Path(nxt["case_path"]) / "logs" / f"worker_{nxt['id']}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0  # type: ignore[attr-defined]
            p = subprocess.Popen(cmd, stdout=open(log, "a"), stderr=subprocess.STDOUT, creationflags=flags)
            self.set(nxt["id"], pid=p.pid)

    def _loop(self) -> None:
        while not self._stop:
            time.sleep(1.0)
            try:
                with self._c() as c:
                    rows = c.execute("SELECT * FROM jobs WHERE status IN ('running','cancel_requested')").fetchall()
                for r in rows:
                    if r["pid"] and not _alive(r["pid"]):
                        # worker ended without writing a final state (crash, OOM, kill)
                        j = self.get(r["id"])
                        if j and j["status"] in ("running", "cancel_requested"):
                            self.set(r["id"], status="failed" if j["status"] == "running" else "cancelled",
                                     finished=_now(), error=j["error"] or "worker process ended unexpectedly")
                self._kick()
            except sqlite3.Error:
                continue


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    if os.name == "nt":  # pragma: no cover
        import ctypes

        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not h:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        return code.value == 259
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:  # reap zombies of our own children
        wpid, _ = os.waitpid(pid, os.WNOHANG)
        return wpid == 0
    except ChildProcessError:
        return True


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
