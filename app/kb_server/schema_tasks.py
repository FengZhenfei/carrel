"""One persisted label-extraction task per KB, executed by the console process.

No queue service: SQLite owns status and single-flight claims, ChatClient owns
step caching. Browser disconnects do not stop the worker. A server restart is
reported as interrupted; clicking retry reuses successful model responses.
"""
from __future__ import annotations

import os
import socket
import threading
import time
import uuid
from typing import Any

from kb_pipeline import db, discovery
from kb_pipeline.graph import schema_flow as flow
from kb_pipeline.graph.llm import LLMInterrupted
from kb_pipeline.graph.sampling_budget import sampling_budget

_INSTANCE = uuid.uuid4().hex
_active: dict[str, threading.Event] = {}


def summary(con: Any, kb_id: str) -> dict[str, Any] | None:
    task = db.get_app_config(con, flow.SUGGEST_TASK_PREFIX + kb_id)
    if not isinstance(task, dict):
        return None
    out = {k: task.get(k) for k in ("task_id", "status", "created_at", "finished_at", "error")}
    out["sampled"] = task.get("sampled")
    mark = db.get_app_config(con, flow.SUGGEST_MARK_PREFIX + kb_id) or {}
    if task.get("status") == "running" and (mark.get("token") != task.get("task_id")
            or time.time() - float(mark.get("started_at") or 0) > flow.SUGGEST_MARK_FRESH_SECONDS):
        out["status"] = "interrupted"
    # Completion is committed before the worker releases its progress claim.
    if task.get("status") == "running" and mark.get("token") == task.get("task_id"):
        out.update(phase=mark.get("phase"), sampled=mark.get("sampled"))
    elif task.get("result"):
        out["sampled"] = task["result"].get("sampled")
    return out


def _finish(settings: Any, kb_id: str, token: str, status: str, error: str = "") -> None:
    with db.connect(settings.state_db) as con:
        con.execute("BEGIN IMMEDIATE")
        task = db.get_app_config(con, flow.SUGGEST_TASK_PREFIX + kb_id)
        if not isinstance(task, dict) or task.get("task_id") != token or task.get("status") != "running":
            return  # Cancelled/deleted/replaced tasks cannot come back.
        mark = db.get_app_config(con, flow.SUGGEST_MARK_PREFIX + kb_id) or {}
        if mark.get("token") == token:
            task["sampled"] = mark.get("sampled")
        task.update(status=status, error=error[:500], finished_at=int(time.time()))
        db.set_app_config(con, flow.SUGGEST_TASK_PREFIX + kb_id, task)
        con.execute("DELETE FROM app_config WHERE key=? AND json_extract(value,'$.token')=?",
                    (flow.SUGGEST_MARK_PREFIX + kb_id, token))


def start(settings: Any, source: Any, spec: Any) -> dict[str, Any]:
    sampling_budget(spec)  # Reject missing capacity before creating a task.
    kb_id = source.kb_id
    token = flow.claim_suggest(settings, kb_id, flow.ORIGIN_MANUAL)
    if token is None:
        with db.connect(settings.state_db) as con:
            existing = summary(con, kb_id)
            mark = db.get_app_config(con, flow.SUGGEST_MARK_PREFIX + kb_id) or {}
        if existing and existing["status"] == "running" and existing["task_id"] == mark.get("token"):
            return existing
        raise ValueError(flow.SUGGEST_BUSY_MESSAGE)
    stop = threading.Event()
    task = {"task_id": token, "status": "running", "created_at": int(time.time()),
            "worker_pid": os.getpid(), "worker_host": socket.gethostname(), "worker_instance": _INSTANCE}
    try:
        with db.connect(settings.state_db) as con:
            con.execute("BEGIN IMMEDIATE")
            mark = db.get_app_config(con, flow.SUGGEST_MARK_PREFIX + kb_id) or {}
            if mark.get("token") != token:
                raise ValueError(flow.SUGGEST_SUPERSEDED_MESSAGE)
            row = con.execute("SELECT status FROM kb_sources WHERE kb_id=?", (kb_id,)).fetchone()
            config = discovery.get_config(con, kb_id)
            enabled = discovery.build_source(settings.mirror_root, source.source_root, config,
                                             kb_id=kb_id, collection=source.collection).graph_enabled
            if row is None or row["status"] != "active" or not enabled:
                raise ValueError("The knowledge base or graph is disabled")
            db.set_app_config(con, flow.SUGGEST_TASK_PREFIX + kb_id, task)
        _active[token] = stop

        def run() -> None:
            try:
                flow.suggest_schema_version(settings, source, origin=flow.ORIGIN_MANUAL, adopt=False,
                                            spec=spec, stop=stop, claimed_token=token)
            except LLMInterrupted:
                _finish(settings, kb_id, token, "cancelled")
            except Exception as exc:
                _finish(settings, kb_id, token, "failed", f"{type(exc).__name__}: {exc}")
            finally:
                _active.pop(token, None)
                flow.release_suggest(settings, kb_id, token)

        threading.Thread(target=run, name=f"schema-task-{kb_id}", daemon=True).start()
    except Exception:
        stop.set()
        _active.pop(token, None)
        _finish(settings, kb_id, token, "failed", "Could not start label extraction")
        flow.release_suggest(settings, kb_id, token)
        raise
    return {k: task[k] for k in ("task_id", "status", "created_at")}


def cancel(settings: Any, kb_id: str, token: str) -> dict[str, Any]:
    _finish(settings, kb_id, token, "cancelled")
    event = _active.get(token)
    if event is not None:
        event.set()
    with db.connect(settings.state_db) as con:
        return summary(con, kb_id) or {"status": "idle"}


def status(settings: Any, kb_id: str) -> dict[str, Any]:
    from .service import schema_versions_view

    with db.connect(settings.state_db) as con:
        config = discovery.get_config(con, kb_id)
        out = summary(con, kb_id) or {"status": "idle"}
        task = db.get_app_config(con, flow.SUGGEST_TASK_PREFIX + kb_id) or {}
    if out["status"] == "completed":
        result = dict(task.get("result") or {})
        version_id = result.get("version_id")
        if not any(v.get("id") == version_id for v in config.get("graph_schema_versions") or []):
            return {**out, "status": "superseded"}
        result["schema"] = schema_versions_view({**discovery.DEFAULTS, **config})
        result["saved"] = config.get("graph_schema_active") == version_id
        result["pending"] = not task.get("acknowledged") and not result["saved"]
        out["result"] = result
    return out


def recover_interrupted(settings: Any) -> None:
    with db.connect(settings.state_db) as con:
        tasks = [(r["key"], db.get_app_config(con, r["key"])) for r in con.execute(
            "SELECT key FROM app_config WHERE key LIKE ?", (flow.SUGGEST_TASK_PREFIX + "%",))]
    for key, task in tasks:
        if not isinstance(task, dict) or task.get("status") != "running" or task.get("worker_host") != socket.gethostname():
            continue
        if task.get("worker_instance") == _INSTANCE:
            continue
        pid = int(task.get("worker_pid") or 0)
        if pid > 0 and pid != os.getpid():
            try:
                os.kill(pid, 0)
                continue  # Do not interfere with another live console process.
            except ProcessLookupError:
                pass
            except PermissionError:
                continue
        _finish(settings, key[len(flow.SUGGEST_TASK_PREFIX):], task["task_id"], "interrupted")
