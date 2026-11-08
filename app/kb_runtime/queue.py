from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import time
from contextlib import contextmanager

# Same cadence as the native systemd units; interval, initial delay, script arguments.
TASKS = {
    "scan": (60, 30, ["kb-pipeline-scan.sh"]),
    "worker": (300, 60, ["kb-pipeline-worker-once.sh"]),
    "graph": (7200, 600, ["kb-graph-rebuild-check.sh"]),
    "gc": (86400, 1800, ["kb-cleanup.sh", "parse-assets-gc"]),
    "cache": (604800, 3600, ["kb-cleanup.sh", "weekly"]),
    "logs": (2592000, 7200, ["kb-cleanup.sh", "monthly"]),
}
UNITS = dict(
    zip(
        (
            "carrel-scan.service",
            "carrel-worker.service",
            "carrel-graph-rebuild.service",
            "carrel-qdrant-gc.service",
            "carrel-cache-weekly.service",
            "carrel-logs-monthly.service",
        ),
        TASKS,
    )
)


def state_dir() -> Path:
    return (
        Path(
            os.environ.get(
                "KB_RUNTIME_DIR",
                str(Path(os.environ.get("KB_LOCAL_BASE_DIR", "/data")) / "runtime"),
            )
        )
        / "state"
    )


@contextmanager
def connect():
    folder = state_dir()
    folder.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(folder / "automation.db", timeout=15)
    con.row_factory = sqlite3.Row
    con.executescript("""
        CREATE TABLE IF NOT EXISTS requests (
          id INTEGER PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL,
          status TEXT NOT NULL DEFAULT 'queued', created REAL NOT NULL,
          started REAL, finished REAL, result INTEGER, pid INTEGER);
        CREATE INDEX IF NOT EXISTS requests_status ON requests(status, kind);
        CREATE TABLE IF NOT EXISTS schedules (
          kind TEXT PRIMARY KEY, next_due REAL NOT NULL, last_started REAL, last_finished REAL, result INTEGER);
    """)
    try:
        with con:
            yield con
    finally:
        con.close()


def enqueue(kind: str, payload: dict | None = None) -> int:
    if kind not in TASKS and kind != "graph-build":
        raise ValueError("Unknown background task")
    payload = payload or {}
    if kind == "graph-build":
        import re

        if not re.fullmatch(r"[\w.-]+", str(payload.get("kb_id", ""))) or set(
            payload
        ) - {"kb_id", "graph_version", "append", "force"}:
            raise ValueError("Invalid graph task")
        if payload.get("graph_version") is not None and not isinstance(
            payload["graph_version"], str
        ):
            raise ValueError("Invalid graph version")
        if any(
            key in payload and not isinstance(payload[key], bool)
            for key in ("append", "force")
        ):
            raise ValueError("Invalid graph task flags")
    elif payload:
        raise ValueError("Periodic task takes no payload")
    raw = json.dumps(payload, sort_keys=True)
    with connect() as con:
        con.execute("BEGIN IMMEDIATE")
        if kind == "graph-build":
            row = con.execute(
                "SELECT id FROM requests WHERE kind='graph-build' AND status IN ('queued','running')"
            ).fetchone()
            if row:
                raise ValueError("A graph build is already running")
        row = con.execute(
            "SELECT id FROM requests WHERE kind=? AND payload=? AND status='queued'",
            (kind, raw),
        ).fetchone()
        if row:
            return int(row[0])
        return int(
            con.execute(
                "INSERT INTO requests(kind,payload,created) VALUES(?,?,?)",
                (kind, raw, time.time()),
            ).lastrowid
        )


def active_graph(kb_id=None):
    with connect() as con:
        rows = list(
            con.execute(
                "SELECT * FROM requests WHERE kind='graph-build' AND status IN ('queued','running')"
            )
        )
    return next(
        (
            dict(r)
            for r in rows
            if kb_id is None or json.loads(r["payload"])["kb_id"] == kb_id
        ),
        None,
    )


def owns_graph_build(request, build) -> bool:
    """A child can publish its terminal build state before dispatch cleanup finishes."""
    return bool(
        request and build is not None and request["status"] == "running"
        and request.get("pid") and request["pid"] == build["worker_pid"]
        and int(build["started_at"] or 0) >= int(request.get("started") or request["created"])
    )


def cancel_graph(kb_id):
    """Cancel dispatch as well as the brief interval before the build writes its state row."""
    import signal

    with connect() as con:
        con.execute("BEGIN IMMEDIATE")
        rows = list(
            con.execute(
                "SELECT * FROM requests WHERE kind='graph-build' AND status IN ('queued','running')"
            )
        )
        rows = [r for r in rows if json.loads(r["payload"])["kb_id"] == kb_id]
        for row in rows:
            if row["pid"]:
                try:
                    cmd = Path(f"/proc/{row['pid']}/cmdline").read_bytes().split(b"\0")
                    if b"kb_pipeline" in cmd and b"graph" in cmd:
                        os.killpg(row["pid"], signal.SIGTERM)
                except (FileNotFoundError, ProcessLookupError):
                    pass
            con.execute(
                "UPDATE requests SET status='cancelled',finished=? WHERE id=?",
                (time.time(), row["id"]),
            )
    return bool(rows)


def timer_rows() -> list[dict]:
    try:
        with connect() as con:
            schedules = {
                r["kind"]: dict(r) for r in con.execute("SELECT * FROM schedules")
            }
            running = {
                r[0]
                for r in con.execute("SELECT kind FROM requests WHERE status='running'")
            }
        heartbeat = json.loads((state_dir() / "runtime.json").read_text())
        alive = time.time() - heartbeat["at"] < 20 and not heartbeat.get("stopping")
    except (OSError, ValueError, KeyError, sqlite3.Error):
        return []

    def fmt(value):
        return time.strftime("%m-%d %H:%M", time.localtime(value)) if value else None

    return [
        {
            "key": k,
            "unit": "container:" + k,
            "label": k,
            "timer_active": alive,
            "running": k in running,
            "ok": None if v.get("result") is None else v["result"] == 0,
            "last": fmt(v.get("last_started")),
            "next": fmt(v.get("next_due")),
            "defers": 0,
            "state": "running" if k in running else "waiting",
            "result": v.get("result"),
        }
        for k, v in schedules.items()
        if k in TASKS
    ]
