from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import signal
import socket
import sqlite3
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, Iterable

from .config import Settings
from . import db
from .utils import looks_like_kb_process
from .graph.lock import GraphBuildLock, build_lock_held, build_lock_path, clear_lock_leftovers


class PartialDeleteError(RuntimeError):
    """A delete left some storage side uncleaned. Carries the completed parts and the failure reasons so
    the console can tell the user honestly -- the registry row is kept as delete_failed and the next GC
    round retries (without waiting out the retention period)."""

    def __init__(self, kb_id: str, entry: dict[str, Any], errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.kb_id = kb_id
        self.entry = entry
        self.errors = errors


ACTIVE_CLEANUP_LOGS = {
    "cache-weekly.stdout.log",
    "cache-weekly.stderr.log",
    "logs-monthly.stdout.log",
    "logs-monthly.stderr.log",
}


def ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def human_size(num: int) -> str:
    value = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{int(value)}B" if unit == "B" else f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}TB"


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def status(settings: Settings) -> dict[str, object]:
    busy, reasons = service_busy(settings)
    history = settings.runtime_dir / "history"
    return {
        "idle": not busy,
        "busy_reasons": reasons,
        "sizes": {
            "cache_dir": human_size(dir_size(settings.cache_dir)),
            "parse_assets": human_size(dir_size(settings.cache_dir / "parse")),
            "logs": human_size(dir_size(settings.log_dir)),
            "history_cache_rotations": human_size(dir_size(history / "cache_rotations")),
            "history_log_rotations": human_size(dir_size(history / "log_rotations")),
        },
        "retention": {
            "cache_rotations": int(os.getenv("KB_CACHE_ROTATION_KEEP", "2")),
            "log_months": int(os.getenv("KB_LOG_ROTATION_KEEP_MONTHS", "3")),
            "qdrant_inactive_days": settings.qdrant_inactive_retention_days,
        },
    }


def weekly_cache_cleanup(settings: Settings, *, dry_run: bool = False) -> dict[str, object]:
    busy, reasons = service_busy(settings)
    if busy:
        return {"skipped": True, "reason": "service busy", "busy_reasons": reasons}

    rotation_root = settings.runtime_dir / "history" / "cache_rotations" / ts()
    moved: dict[str, int] = {
        # Under cache_dir the pipeline itself only writes parse and vlm-cache, each reclaimed its own way
        # (vlm-cache is the cross-version caption / vector cache -- rotating it away would send every image
        # through the models again); anything else that shows up here is rotated away, keeping the latest few.
        str(settings.cache_dir): move_dir_contents(settings.cache_dir, rotation_root / "kb-pipeline", dry_run=dry_run, exclude_names={"parse", "vlm-cache"}),
    }

    ds_store = remove_named_files([settings.runtime_dir, settings.log_dir], ".DS_Store", dry_run=dry_run)
    keep = int(os.getenv("KB_CACHE_ROTATION_KEEP", "2"))
    pruned = prune_old_dirs(settings.runtime_dir / "history" / "cache_rotations", keep=keep, dry_run=dry_run)
    vlm_cache = prune_vlm_cache(settings.cache_dir, dry_run=dry_run)
    return {
        "skipped": False,
        "dry_run": dry_run,
        "rotation_root": str(rotation_root),
        "moved_entries": moved,
        "removed_ds_store": ds_store,
        "pruned_rotations": [path.name for path in pruned],
        "vlm_cache_pruned": vlm_cache["removed"],
        "vlm_cache": vlm_cache,
    }


def monthly_log_cleanup(settings: Settings, *, dry_run: bool = False) -> dict[str, object]:
    busy, reasons = service_busy(settings)
    if busy:
        return {"skipped": True, "reason": "service busy", "busy_reasons": reasons}
    month_root = settings.runtime_dir / "history" / "log_rotations" / time.strftime("%Y%m")
    rotated = rotate_logs_in_dir(settings.log_dir, month_root / "logs", dry_run=dry_run)
    keep = int(os.getenv("KB_LOG_ROTATION_KEEP_MONTHS", "3"))
    pruned = prune_old_dirs(settings.runtime_dir / "history" / "log_rotations", keep=keep, dry_run=dry_run)
    return {
        "skipped": False,
        "dry_run": dry_run,
        "month_root": str(month_root),
        "rotated_logs": rotated,
        "pruned_months": [path.name for path in pruned],
    }


def _all_known_collections(settings: Settings) -> set[str]:
    """Collections of the active knowledge bases plus the deactivated ones still in the registry. When the
    point-level GC only walked active ones, closed KBs were never reclaimed during the whole retention
    period."""
    collections = {source.collection for source in settings.sources.values()}
    try:
        with db.connect(settings.state_db) as con:
            rows = con.execute("SELECT collection FROM kb_sources").fetchall()
        collections.update(str(row["collection"]) for row in rows if row["collection"])
    except sqlite3.Error:
        pass
    return collections


def qdrant_inactive_gc(settings: Settings, *, retention_days: int, dry_run: bool = False) -> dict[str, object]:
    if retention_days < 1:
        raise ValueError("retention_days must be >= 1")
    busy, reasons = service_busy(settings)
    if busy:
        return {"skipped": True, "reason": "service busy", "busy_reasons": reasons}

    from .vector.qdrant import client as qdrant_client
    from .vector.qdrant import delete_inactive_points_older_than

    cutoff_ts = int(time.time()) - retention_days * 24 * 3600
    q = qdrant_client(settings.qdrant_url, settings.qdrant_api_key)
    collections = sorted({source.collection for source in settings.sources.values()})
    deleted: dict[str, int] = {}
    errors: list[str] = []
    for collection in collections:
        try:
            deleted[collection] = delete_inactive_points_older_than(q, collection, cutoff_ts, dry_run=dry_run)
        except Exception as exc:
            deleted[collection] = 0
            errors.append(f"{collection}: {exc!r}")
    return {
        "skipped": False,
        "dry_run": dry_run,
        "retention_days": retention_days,
        "cutoff_ts": cutoff_ts,
        "collections": deleted,
        "total_points": sum(deleted.values()),
        "errors": errors,
    }


def superseded_unsuccessful_versions(settings: Settings) -> set[str]:
    """For the timer cleanup: version numbers left by paused / failed builds of each KB that a newer
    successful version has already superseded (the most recently paused one is kept for resume)."""
    out: set[str] = set()
    try:
        with db.connect(settings.state_db) as con:
            for kb_id in settings.sources:
                out |= db.unsuccessful_graph_versions(con, kb_id, superseded_only=True)
    except sqlite3.OperationalError:
        return set()
    return out


def qdrant_graph_collection_gc(
    settings: Settings,
    *,
    retention_days: int,
    dry_run: bool = False,
    delete_unparseable: bool = False,
) -> dict[str, object]:
    if retention_days < 1:
        raise ValueError("retention_days must be >= 1")
    busy, reasons = service_busy(settings)
    if busy:
        return {"skipped": True, "reason": "service busy", "busy_reasons": reasons}

    from .vector.qdrant import client as qdrant_client
    from .vector.qdrant import delete_old_graph_collections

    q = qdrant_client(settings.qdrant_url, settings.qdrant_api_key)
    source_collections = sorted({source.collection for source in settings.sources.values()})
    result = delete_old_graph_collections(
        q,
        source_collections=source_collections,
        retention_days=retention_days,
        dry_run=dry_run,
        delete_unparseable=delete_unparseable,
        discard_versions=superseded_unsuccessful_versions(settings),
    )
    result["skipped"] = False
    return result


def neo4j_graph_gc(
    settings: Settings,
    *,
    retention_days: int,
    dry_run: bool = False,
    delete_unparseable: bool = False,
) -> dict[str, object]:
    if retention_days < 1:
        raise ValueError("retention_days must be >= 1")
    busy, reasons = service_busy(settings)
    if busy:
        return {"skipped": True, "reason": "service busy", "busy_reasons": reasons}

    from .graph.neo4j_import import delete_old_neo4j_graph_versions

    result = delete_old_neo4j_graph_versions(
        settings,
        sources=settings.sources.values(),
        retention_days=retention_days,
        dry_run=dry_run,
        delete_unparseable=delete_unparseable,
        discard_versions=superseded_unsuccessful_versions(settings),
    )
    result["skipped"] = False
    return result


def graph_gc(settings: Settings, *, keep_latest: int | None = None, dry_run: bool = False) -> dict[str, object]:
    """Nightly safety net: clean every base's superseded graph versions (Qdrant collections, Neo4j versions,
    workspace directories, build records) with the same "active plus N - 1" rule as the end of a build. The
    end-of-build GC only runs when that base builds, so idle bases and bases whose GC failed rely on this one.
    The newest paused / failed version is kept for a resume, outside the keep window. The whole round is
    skipped while a build runs (the build lock cannot be taken)."""
    from .graph.build import gc_graph_versions
    from .graph.lock import GraphBuildLock
    from .vector.qdrant import client as qdrant_client
    from .vector.qdrant import graph_alias_targets, parse_graph_collection_name

    keep = max(1, int(keep_latest if keep_latest is not None else getattr(settings, "graph_gc_keep_versions", 2) or 2))
    out: dict[str, object] = {"skipped": False, "dry_run": dry_run, "keep_latest": keep, "sources": {}, "errors": {}}
    lock = GraphBuildLock(settings)
    try:
        lock.acquire()
    except RuntimeError:
        return {"skipped": True, "reason": "graph build running", "yielded_to": "graph_build",
                "dry_run": dry_run, "keep_latest": keep}
    try:
        q = qdrant_client(settings.qdrant_url, settings.qdrant_api_key)
        for key, source in sorted(settings.sources.items(), key=lambda kv: kv[1].kb_id):
            if not getattr(source, "graph_enabled", False):
                continue
            versions: set[str] = set()
            for target in graph_alias_targets(q, source.collection).values():
                parsed = parse_graph_collection_name(str(target)) if target else None
                if parsed:
                    versions.add(str(parsed["graph_version"]))
            active = next(iter(versions)) if len(versions) == 1 else ""
            # The newest paused / failed version is kept for a resume: not deleted and not counted against the
            # "latest N versions"; older half-finished versions are deleted outright
            try:
                with db.connect(settings.state_db) as con:
                    resumable = db.resumable_graph_versions(con, source.kb_id) - {active}
                    discard = db.unsuccessful_graph_versions(con, source.kb_id) - resumable - {active}
            except sqlite3.OperationalError:
                resumable, discard = set(), set()
            gc = gc_graph_versions(settings, source, q=q, graph_version=active, keep_latest=keep, discard=discard,
                                   protect=resumable, grace_seconds=0, dry_run=dry_run)
            out["sources"][source.kb_id] = {"source": key, "active_graph_version": active or None, **gc["result"]}
            if gc["errors"]:
                out["errors"][source.kb_id] = gc["errors"]
    finally:
        lock.release()
    return out


# Only kill processes whose command line carries this marker. Graph builds and parse workers are both
# `python -m kb_pipeline …`, and pids get recycled by the system -- for a stale graph_builds row that
# sat there for days, the pid may long belong to someone else's process (worst case, this host's own web
# service). See _process_matches.
_KB_PROCESS_MARKER = "kb_pipeline"


def _process_matches(pid: int, marker: str) -> bool:
    """Whether this pid is one of our own processes. Returns False when /proc cannot be read (non-Linux,
    insufficient permission) -- when unsure, do not kill; better to let the caller report "could not
    terminate"."""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False
    cmdline = raw.replace(b"\x00", b" ").decode("utf-8", "replace")
    return marker in cmdline or looks_like_kb_process(cmdline)      # `kb …` starts were missed (Codex review N10)


def _terminate_pid(pid: int, *, timeout: float = 20.0, marker: str = _KB_PROCESS_MARKER) -> str:
    """SIGTERM, then give it timeout seconds to finish its own wrap-up (a graph build process writes its
    terminal state, a worker releases its lock); SIGKILL if still alive. The return value is only for logs
    and error messages; callers determine the real outcome by polling afterwards."""
    if pid <= 0:
        return "skipped:bad-pid"
    if pid == os.getpid():
        # The recorded pid equals the caller itself = the record is bogus (or fabricated by a test).
        return "skipped:self"
    if not _process_matches(pid, marker):
        # The process is gone, or the pid has been reused by someone else. Neither should be touched: the
        # former leaves nothing to do, and killing the latter by mistake would take down an unrelated process.
        # They mean different things to the caller though: the former is "confirmed dead", the latter is
        # "identity mismatch, cannot confirm" (Codex review N10).
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return "already-gone"
        except PermissionError:
            return "not-ours"
        return "not-ours"
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return "already-gone"
    except PermissionError:
        return "denied"
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(0.3)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return "terminated"
        except PermissionError:
            return "denied"
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
    time.sleep(0.5)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "killed"
    return "still-alive"


def _terminate_lease(locked_by: str, *, timeout: float = 20.0) -> str:
    """jobs.locked_by is the "hostname:pid" written by worker.worker_id(). Only local processes are killed;
    a worker on another host is left to the lease timeout as the fallback -- a hard kill from here is
    neither possible nor appropriate."""
    host, _, pid_text = str(locked_by or "").rpartition(":")
    if not host:
        return f"skipped:bad-lease:{locked_by!r}"
    if host != socket.gethostname():
        return f"skipped:other-host:{host}"
    try:
        pid = int(pid_text)
    except ValueError:
        return f"skipped:bad-lease:{locked_by!r}"
    return _terminate_pid(pid, timeout=timeout)


def stop_kb_parse_jobs(
    settings: Settings,
    *,
    kb_id: str,
    reason: str,
    kill: bool = False,
    timeout: float = 25.0,
) -> dict[str, Any]:
    """Stop the parse queue of a knowledge base.

    kill=False (close knowledge base): queued jobs are voided outright, running ones only get the cancel
    flag, and this returns immediately -- the parse job exits on its own at the next phase boundary, so
    the console request is not blocked. Data already stored is never touched.

    kill=True (delete knowledge base): on top of that, SIGTERM the worker processes holding the locks and
    **wait until they are really dead**. Waiting is mandatory: if a worker is still alive after the
    collection is dropped, ensure_collection recreates it, leaving a zombie collection with no registry
    row."""
    with db.connect(settings.state_db) as con:
        counts = db.request_kb_job_cancel(con, kb_id, reason)
        leases = db.running_job_leases(con, kb_id)
    entry: dict[str, Any] = dict(counts)
    entry["running"] = len(leases)
    if not leases:
        entry["stopped"] = True
        return entry
    if not kill:
        entry["stopped"] = False   # cooperative: the console shows "stopping"
        return entry
    entry["terminated"] = [_terminate_lease(locked_by, timeout=timeout) for _, locked_by in leases]
    deadline = time.time() + 10.0
    while True:
        with db.connect(settings.state_db) as con:
            remaining = db.running_job_leases(con, kb_id)
        if not remaining or time.time() >= deadline:
            break
        time.sleep(0.5)
    # "Confirmed dead" and "lease still present" are judged separately (Codex review N10): a process that is
    # terminated / killed / never existed simply had no time to release its lease and counts as stopped;
    # only identity mismatch (not-ours) or unkillable (still-alive / denied) counts as not stopped, and the
    # delete-KB entry point refuses to go on dropping the collection based on that
    dead = all(str(r) in ("terminated", "killed", "already-gone", "skipped:self") for r in entry["terminated"])
    entry["stopped"] = (not remaining) or dead
    with db.connect(settings.state_db) as con:
        for job_id, _ in leases:
            db.cancel_job(con, job_id, reason)
        con.commit()
    return entry


_STOP_GRACE_SECONDS = 15.0        # time to wait after SIGTERM for the process to write its terminal state


def stop_graph_build_now(
    settings: Settings,
    *,
    kb_id: str,
    reason: str,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Terminate the graph build process running for this knowledge base, touching no artifacts and neither
    the LLM cache nor the units already extracted.

    "Close knowledge graph" relies on it for checkpoint pausing: extraction results are stored per unit
    and responses are cached by content, both keyed by KB (not by version), so the next build hits the
    finished ones directly and continues from where it was interrupted."""
    hostname = socket.gethostname()
    with db.connect(settings.state_db) as con:
        db.reconcile_stale_graph_builds(con)
        row = con.execute(
            "SELECT graph_build_id, status, worker_host, worker_pid FROM graph_builds "
            "WHERE kb_id=? ORDER BY started_at DESC LIMIT 1",
            (kb_id,),
        ).fetchone()
    if row is None or str(row["status"]) != "running":
        return {"running": False, "stopped": True}
    build_id = str(row["graph_build_id"])
    host = str(row["worker_host"] or "")
    pid = int(row["worker_pid"] or 0)
    entry: dict[str, Any] = {"running": True, "graph_build_id": build_id}
    if host and host != hostname:
        raise ValueError(f"The graph build runs on another host ({host}) and cannot be terminated from here")
    term = _terminate_pid(pid, timeout=timeout)
    entry["terminated"] = term
    # Only when the process is confirmed gone (exited normally, killed, never existed, bogus record) may the
    # terminal state be written as a fallback. not-ours / denied / still-alive are no proof of death:
    # setting the database to cancelled and then taking "not running" as a successful stop would let
    # "delete graph" delete the version that is still being written (final review F01).
    confirmed_dead = term in ("terminated", "killed", "already-gone", "skipped:self", "skipped:bad-pid")
    # After SIGTERM the process writes its own record as cancelled; one taken down by SIGKILL never gets the
    # chance, so this is the fallback -- otherwise the running row would block the next build forever
    deadline = time.time() + (_STOP_GRACE_SECONDS if term in ("terminated", "killed") else 0.0)
    while True:
        with db.connect(settings.state_db) as con:
            cur = con.execute(
                "SELECT status FROM graph_builds WHERE graph_build_id=?", (build_id,)
            ).fetchone()
        status = str(cur["status"]) if cur is not None else "gone"
        if status != "running" or time.time() >= deadline:
            break
        time.sleep(0.5)
    if status == "running" and confirmed_dead:
        with db.connect(settings.state_db) as con:
            db.finish_graph_build(con, build_id, status="cancelled", error=reason)
            con.commit()
        status = "cancelled"
    entry["status"] = status
    entry["stopped"] = status != "running"
    if not entry["stopped"]:
        entry["error"] = f"The graph build process did not confirm stopping ({term}); the record stays running"
    # The lock is an flock, released by the kernel as soon as the process dies; this only clears the pid /
    # started_at it had no time to remove (display only)
    if clear_lock_leftovers(build_lock_path(settings)):
        entry["lock_reclaimed"] = True
    return entry


def _drop_graph_data(settings: Settings, con, *, kb_id: str, collection: str, errors: list[str]) -> dict[str, Any]:
    """Complete teardown of the graph-side data: the graph collections and aliases, the Neo4j projection,
    the graph build workspace and LLM cache, the per-unit extraction results, the graph_builds records,
    and the extraction constraints the LLM derived (type table / language / predicates / parent types).
    Shared by KB deletion (including GC) and the console's "delete knowledge graph", with strictly the
    same cleanup scope; the main KB data is not part of it.

    Why the type table and language count as "graph-side data": they are not hand-written configuration
    but derived results that the console's "extract / re-extract labels" induced from **this version of
    the corpus**. Once the graph is gone, the basis of that induction is gone too; keeping them would only
    make the next build inherit an old constraint of unknown origin. Clearing them falls back to the
    global defaults. The sample size and the entity label model are set by hand and are not cleaned up.
    """
    from . import discovery
    from .vector.qdrant import client as qdrant_client
    from .vector.qdrant import graph_collection_short_name

    entry: dict[str, Any] = {}
    q = qdrant_client(settings.qdrant_url, settings.qdrant_api_key)
    try:
        entry["graph_qdrant"] = _drop_graph_artifacts(q, collection)
    except Exception as exc:
        errors.append(f"{kb_id}: graph artefact drop failed: {exc!r}")
    try:
        entry["neo4j"] = _drop_neo4j_projection(settings, kb_id)
    except Exception as exc:
        errors.append(f"{kb_id}: neo4j projection drop failed: {exc!r}")
    try:
        short = graph_collection_short_name(collection)
        removed: list[str] = []
        work = settings.graph_work_dir / "work" / short
        if work.exists():
            shutil.rmtree(work, ignore_errors=True)
            removed.append(str(work))
        for suffix in ("", "-wal", "-shm"):
            cache = settings.graph_work_dir / "cache" / f"{short}.sqlite{suffix}"
            if cache.exists():
                cache.unlink(missing_ok=True)
                if not suffix:
                    removed.append(str(cache))
        if removed:
            entry["graph_workspace_removed"] = removed
    except Exception as exc:
        errors.append(f"{kb_id}: graph workspace cleanup failed: {exc!r}")
    try:
        for table in ("graph_build_chunks", "graph_build_phases", "graph_units"):
            con.execute(
                f"DELETE FROM {table} WHERE graph_build_id IN "
                "(SELECT graph_build_id FROM graph_builds WHERE kb_id=?)", (kb_id,))
        cur = con.execute("DELETE FROM graph_builds WHERE kb_id=?", (kb_id,))
        entry["graph_builds_deleted"] = int(cur.rowcount or 0)
        entry["graph_extractions_deleted"] = db.delete_graph_extractions(con, kb_id)
        entry["graph_facts_deleted"] = db.delete_graph_facts(con, kb_id)
    except Exception as exc:
        errors.append(f"{kb_id}: graph build records cleanup failed: {exc!r}")
    try:
        discovery.set_config(con, kb_id, {
            "graph_entity_types": None, "graph_language": None,
            "graph_predicates": None, "graph_parent_types": None,
            "graph_type_definitions": None, "graph_examples": None, "graph_profile": None,
            # Same for the historical versions: they were all induced from this version of the corpus, and
            # once the graph is gone their basis is gone. The capability questions are a hand-maintained
            # acceptance set and stay.
            "graph_schema_versions": None, "graph_schema_active": None,
        })
        entry["graph_schema_cleared"] = True
    except KeyError:
        # The registry row is already gone (should the hard-delete path ever drop the row first). There is
        # no config to clear, which is not a failure -- recording an error would flip delete_graph_now's
        # cleared verdict to false and leave graph_enabled uncleared.
        entry["graph_schema_cleared"] = False
    except Exception as exc:
        errors.append(f"{kb_id}: graph schema cleanup failed: {exc!r}")
    return entry


def delete_graph_now(settings: Settings, *, kb_id: str) -> dict[str, Any]:
    """The console's "delete knowledge graph": only graph-side data is deleted, the main KB and parse
    results are untouched; graph_enabled is cleared afterwards (the console toggle goes grey).

    A build in progress is no longer a reason to refuse -- the process is terminated first, then
    everything including the LLM cache is wiped. "Delete" means "all progress is void", deliberately
    distinct from "close" (stop but keep the cache for resume)."""
    from . import discovery

    errors: list[str] = []
    stopped = stop_graph_build_now(settings, kb_id=kb_id, reason="Build terminated by “Delete knowledge graph”")
    if not stopped.get("stopped"):
        raise ValueError("The graph build process could not be terminated; deletion abandoned rather than left half done. Try again later")
    # Hold the graph build lock from the confirmed stop until deletion finishes (final review F01): failing
    # to get the lock means a build is still alive (in its preparation phase, or another one just started),
    # so do not delete; once held, no build started during the deletion can get in either
    guard = GraphBuildLock(settings)
    try:
        guard.acquire()
    except RuntimeError as exc:
        raise ValueError("A graph build still holds the build lock; deletion abandoned rather than left half done. Try again later") from exc
    try:
        with db.connect(settings.state_db) as con:
            discovery.init_schema(con)
            # Running rows whose process is dead are marked failed first, otherwise one crash would make the
            # graph undeletable forever
            db.reconcile_stale_graph_builds(con)
            row = con.execute("SELECT * FROM kb_sources WHERE kb_id=?", (kb_id,)).fetchone()
            if row is None:
                raise KeyError(kb_id)
            entry = _drop_graph_data(settings, con, kb_id=kb_id, collection=str(row["collection"]), errors=errors)
            # If any external delete failed, keep the toggle: otherwise the alias would still point at the old
            # collection and the Neo4j nodes would remain, while the UI believes the graph no longer exists
            # and there is no entry point left to clean them up. The build records are gone by now, and a base
            # with the toggle on and no successful build is taken for never built by the next maintenance
            # round and rebuilt in full -- the very graph the user asked to delete (2026-09-29 audit). So the
            # pause is recorded as well and automatic maintenance yields; it is cleared when a retried delete
            # succeeds or the user starts a build.
            cleared = not errors
            if cleared:
                discovery.set_config(con, kb_id, {"graph_enabled": None, "graph_paused": None})
            else:
                discovery.set_config(con, kb_id, {"graph_paused": True})
            con.commit()
    finally:
        guard.release()
    entry.update({"kb_id": kb_id, "errors": errors, "graph_enabled_cleared": cleared,
                  "build_stopped": stopped})
    return entry


def _hard_delete_kb(settings: Settings, con, *, kb_id: str, collection: str, errors: list[str]) -> dict[str, Any]:
    """Drop every trace of one KB: Qdrant collection, graph data (via
    _drop_graph_data), OpenSearch index, SQLite state, parse cache, and
    finally the kb_sources row. Shared by the retention GC and the console's
    immediate delete -- both paths must clean up identically."""
    from . import discovery, search_fts
    from .vector.qdrant import client as qdrant_client
    from .vector.qdrant import delete_collection as delete_qdrant_collection

    entry: dict[str, Any] = {}
    purge_failed = False
    q = qdrant_client(settings.qdrant_url, settings.qdrant_api_key)
    try:
        entry["qdrant"] = delete_qdrant_collection(q, collection)
    except Exception as exc:
        errors.append(f"{kb_id}: qdrant drop failed: {exc!r}")
    # Graph artefacts would otherwise leak forever: the graph GCs only
    # iterate currently-present directories and always skip aliased
    # collections, so a vanished graph-enabled KB matched neither.
    entry.update(_drop_graph_data(settings, con, kb_id=kb_id, collection=collection, errors=errors))
    try:
        entry["opensearch"] = search_fts.delete_collection(settings.opensearch_url, collection=collection)
    except Exception as exc:
        errors.append(f"{kb_id}: opensearch drop failed: {exc!r}")
    try:
        entry["sqlite"] = db.purge_kb_state(con, kb_id)
        # KB ids are never reused, so these two would be rows nobody ever cleans up: the "extracting" mark of
        # label extraction and the verdict recorded by the scheduled check
        from .graph.schema_flow import SUGGEST_MARK_PREFIX

        con.execute("DELETE FROM app_config WHERE key = ?", (SUGGEST_MARK_PREFIX + kb_id,))
        try:
            con.execute("DELETE FROM graph_checks WHERE kb_id = ?", (kb_id,))
        except sqlite3.OperationalError:
            pass      # no table yet: the scheduled check never ran on this state database
    except Exception as exc:
        purge_failed = True
        errors.append(f"{kb_id}: sqlite purge failed: {exc!r}")
    cache_dir = settings.cache_dir / "parse" / kb_id
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
        entry["cache_dir_removed"] = str(cache_dir)
    # The registry row is the only clue for finding these external resources again: the collection name,
    # the graph short name and the Neo4j kb_id exist only in this row. If any side is not fully deleted,
    # keep it and mark it delete_failed so the next GC round retries; once forgotten, the leftover
    # collection / projection / index can never be discovered by any path (GC only walks the registry)
    # and would occupy disk and memory forever.
    if purge_failed or errors:
        entry["forgotten"] = False
        entry["delete_incomplete"] = True
        try:
            con.execute(
                "UPDATE kb_sources SET status='inactive', inactive_reason='delete_failed', "
                "inactive_at=COALESCE(inactive_at, ?) WHERE kb_id=?",
                (int(time.time()), kb_id),
            )
        except sqlite3.Error as exc:  # if even the status cannot be written, all we can do is report it
            errors.append(f"{kb_id}: could not mark delete_failed: {exc!r}")
    else:
        discovery.forget(con, kb_id)
        entry["forgotten"] = True
    return entry


def delete_kb_now(settings: Settings, *, kb_id: str) -> dict[str, Any]:
    """Console's immediate hard delete: identical drops to the retention GC,
    for one explicitly chosen KB, without waiting out the retention window.

    This KB's own running parse / graph build is no longer a reason to refuse -- "delete" means
    terminating the jobs along with it. Only two things really refuse: jobs that cannot be killed (better
    not to delete than to delete half), and a graph build lock that belongs to **another** KB (that build
    has nothing to do with this KB and must not be killed)."""
    from . import discovery

    with db.connect(settings.state_db) as con:
        discovery.init_schema(con)
        row = con.execute("SELECT * FROM kb_sources WHERE kb_id=?", (kb_id,)).fetchone()
    if row is None:
        raise KeyError(kb_id)
    # The busy check is per KB: a global gate would mean that any KB parsing, a scan lock existing, or even
    # a non-empty MinerU queue refuses deleting another KB; with the scan running every minute that is
    # almost the normal state. Delete means "terminate the running jobs along with it": first stop this
    # KB's graph build, then this KB's parse jobs, and only touch external storage once both are confirmed
    # dead.
    graph_stopped = stop_graph_build_now(settings, kb_id=kb_id, reason="Build terminated by “Delete knowledge base”")
    if not graph_stopped.get("stopped"):
        raise ValueError("The graph build process could not be terminated; deletion abandoned rather than left half done. Try again later")
    # Hold the graph build lock from the moment the build is confirmed stopped until the deletion is done, as
    # "Delete knowledge graph" does: deleting takes minutes, and without the lock the scheduled check could
    # start a build for the same KB meanwhile, leaving a build record without a registry row, an empty work
    # dir or graph database nodes. Not getting the lock = someone else holds it, and that build is not ours
    # to kill.
    guard = GraphBuildLock(settings)
    try:
        guard.acquire()
    except RuntimeError as exc:
        from .graph.schema_flow import suggest_in_progress

        if suggest_in_progress(settings, kb_id) is not None:
            # A build takes the lock first, then extracts labels, and only then writes its build record:
            # during those minutes there is no process for the code above to stop
            raise ValueError(
                "This knowledge base is extracting labels (the preparation before a build), which cannot be stopped; delete after it finishes, usually a few minutes"
            ) from exc
        raise ValueError("Another knowledge base is building its graph or being deleted; delete after it finishes") from exc
    try:
        jobs_stopped = stop_kb_parse_jobs(
            settings, kb_id=kb_id, reason="cancelled because the knowledge base is being deleted", kill=True
        )
        if not jobs_stopped.get("stopped"):
            raise ValueError(
                "Parse jobs could not be terminated; deletion abandoned rather than leaving orphaned collections. Try again later"
                f" (terminated={jobs_stopped.get('terminated')})"
            )
        errors: list[str] = []
        with db.connect(settings.state_db) as con:
            # Mark the registry row as being deleted and void the queued / retry jobs, commit immediately, and
            # only then touch external storage. Deletion spans several network calls to Qdrant / Neo4j /
            # OpenSearch and takes minutes for a large KB, during which SQLite holds no write lock:
            # - with the registry row still active, the scan keeps registering files, the scheduled check keeps
            #   starting builds for it and the check before a build is published lets it through; and if the
            #   process dies midway, nothing shows that the KB was half deleted;
            # - a worker starting in that window would claim this KB's queued jobs and, after the collection is
            #   dropped, recreate it via ensure_collection, leaving an ownerless zombie collection.
            now = int(time.time())
            con.execute(
                "UPDATE kb_sources SET status='inactive', inactive_reason='deleting', "
                "inactive_at=COALESCE(inactive_at, ?) WHERE kb_id=?", (now, kb_id))
            cancelled = con.execute(
                "UPDATE jobs SET status='cancelled', error=?, finished_at=?, updated_at=?, "
                "locked_by=NULL, locked_until=NULL WHERE kb_id=? AND status IN ('queued','retry')",
                ("cancelled because the knowledge base is being deleted", now, now, kb_id),
            ).rowcount
            con.commit()
            if cancelled:
                print(f"[delete] cancelled {cancelled} pending job(s) before dropping {kb_id}", flush=True)
            entry = _hard_delete_kb(settings, con, kb_id=kb_id, collection=str(row["collection"]), errors=errors)
            entry["cancelled_jobs"] = int(cancelled or 0) + int(jobs_stopped.get("cancelled") or 0)
            entry["stopped_jobs"] = jobs_stopped
            entry["stopped_graph_build"] = graph_stopped
            con.commit()
    finally:
        guard.release()
    entry.update({"kb_id": kb_id, "errors": errors})
    if errors:
        # The console used to report "completely deleted" on any HTTP 200. An incomplete delete must be
        # said out loud, otherwise the user has no way to know that Qdrant / Neo4j still hold leftovers.
        raise PartialDeleteError(kb_id, entry, errors)
    return entry


def kb_sources_gc(settings: Settings, *, retention_days: int, dry_run: bool = False) -> dict[str, object]:
    """Drop knowledge bases whose top-level directory has been gone longer
    than the retention window: Qdrant collection, graph artefacts, OpenSearch
    index, graph workspace, SQLite state, parse cache, and the kb_sources
    row (all via _hard_delete_kb). Mirrors per-point inactive GC one level
    up. Never touches a KB whose directory is present again."""
    from . import discovery

    busy, reasons = service_busy(settings)
    if busy:
        # The one GC that hard-deletes whole collections gets the same gate as
        # its siblings; racing a live worker could otherwise drop a collection
        # under an in-flight job.
        return {"skipped": True, "reason": "service busy", "busy_reasons": reasons}

    cutoff_ts = int(time.time()) - retention_days * 24 * 3600
    dropped: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    errors: list[str] = []
    with db.connect(settings.state_db) as con:
        discovery.init_schema(con)
        rows = discovery.inactive_older_than(con, cutoff_ts)
        present = set(discovery.discover_directories(settings.mirror_root))
        for row in rows:
            reason = str(row["inactive_reason"] or "")
            # "Skip while the directory is still there" only holds for the "directory vanished" kind of
            # deactivation (the mirror-era model). A KB closed from the console keeps its directory intact,
            # so applying that rule would mean it never expires -- the 7-day automatic deletion promised by
            # the dialog would be an empty promise. Reopening clears inactive_at, so anything reaching this
            # point has definitely not been opened for a whole retention period.
            # A permanent deletion that did not finish (delete_failed, or deleting left behind when the
            # process died midway) is not protected by "directory still there" either: the console promised
            # "retried by the next maintenance round", and a KB whose derived data was cleared while its
            # source directory remains matches the skip condition exactly, so applying the rule would mean it
            # is never retried. Reaching this point means no deletion is in progress: the busy check above
            # looks at the graph build lock, which a deletion holds from start to end
            if reason not in ("unenrolled", *discovery.DELETE_PENDING_REASONS) and str(row["source_root"]) in present:
                # The directory is back. Note that the scan does **not** revive it automatically --
                # touch_seen deliberately only refreshes last_seen and never flips the status; revival can
                # only be an explicit reopen in the console (enroll() goes through reactivated, files are
                # not re-parsed). So skipping does not mean "wait for the scan to handle it" but "no hard
                # delete until a human deals with it": it can stay in this state indefinitely, and the
                # console shows no countdown for it accordingly.
                continue
            kb_id = str(row["kb_id"]); collection = str(row["collection"])
            entry: dict[str, Any] = {"kb_id": kb_id, "collection": collection,
                                     "source_root": row["source_root"], "inactive_at": row["inactive_at"]}
            pending = con.execute(
                "SELECT COUNT(*) AS c FROM jobs WHERE kb_id = ? AND status IN ('queued', 'retry', 'running')",
                (kb_id,),
            ).fetchone()
            if int(pending["c"] if pending else 0):
                skipped.append({**entry, "reason": "jobs still pending", "pending_jobs": int(pending["c"])})
                continue
            if dry_run:
                entry["dry_run"] = True
                dropped.append(entry)
                continue
            entry.update(_hard_delete_kb(settings, con, kb_id=kb_id, collection=collection, errors=errors))
            dropped.append(entry)
        con.commit()
    return {
        "retention_days": retention_days,
        "dry_run": dry_run,
        "dropped": dropped,
        "skipped": skipped,
        "errors": errors,
    }


def _drop_graph_artifacts(q: Any, collection: str) -> dict[str, Any]:
    from .vector.qdrant import (
        ALL_GRAPH_VECTOR_TYPES,
        delete_collection,
        graph_collection_alias,
        graph_collection_short_name,
        parse_graph_collection_name,
    )
    from qdrant_client.http import models as qdrant_models

    result: dict[str, Any] = {"aliases_deleted": [], "collections_deleted": []}
    aliases = {item.alias_name: item.collection_name for item in q.get_aliases().aliases}
    operations = []
    for graph_type in ALL_GRAPH_VECTOR_TYPES:
        alias = graph_collection_alias(collection, graph_type)
        if alias in aliases:
            operations.append(
                qdrant_models.DeleteAliasOperation(delete_alias=qdrant_models.DeleteAlias(alias_name=alias))
            )
            result["aliases_deleted"].append(alias)
    if operations:
        q.update_collection_aliases(operations)
    short = graph_collection_short_name(collection)
    for item in q.get_collections().collections:
        parsed = parse_graph_collection_name(item.name)
        if parsed and parsed["source_short"] == short:
            result["collections_deleted"].append(delete_collection(q, item.name))
    return result


def _drop_neo4j_projection(settings: Settings, kb_id: str) -> dict[str, Any]:
    if not settings.neo4j_password:
        # The graph build side raises outright under the same condition; the delete side used to skip
        # silently, so a "successfully deleted" KB left all its nodes in Neo4j. Let the caller see it.
        raise RuntimeError("neo4j password not configured; projection left untouched")
    from .graph.neo4j_import import delete_kb_projection, neo4j_driver

    driver = neo4j_driver(settings)
    try:
        return {"deleted_nodes": delete_kb_projection(driver, kb_id)}
    finally:
        driver.close()


def parse_assets_gc(settings: Settings, *, retention_days: int, dry_run: bool = False) -> dict[str, object]:
    if retention_days < 1:
        raise ValueError("retention_days must be >= 1")
    busy, reasons = service_busy(settings)
    if busy:
        return {"skipped": True, "reason": "service busy", "busy_reasons": reasons}

    from .vector.qdrant import client as qdrant_client
    from .vector.qdrant import active_doc_point_count
    from .vector.qdrant import delete_expired_inactive_points
    from .vector.qdrant import delete_malformed_inactive_points
    from .vector.qdrant import expired_inactive_point_ids
    from .vector.qdrant import inactive_doc_versions_older_than

    cutoff_ts = int(time.time()) - retention_days * 24 * 3600
    q = qdrant_client(settings.qdrant_url, settings.qdrant_api_key)
    # settings.sources only holds active rows; the collections of deactivated KBs must keep getting
    # point-level cleanup until they are hard-deleted, otherwise a closed KB reclaims no garbage at all
    # during the whole retention period.
    collections = sorted(_all_known_collections(settings))
    parse_root = settings.cache_dir / "parse"

    qdrant_versions: list[dict[str, Any]] = []
    deleted_files: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    totals = {
        "qdrant_points": 0,
        "qdrant_malformed_points": 0,
        "sqlite_chunks": 0,
        "sqlite_files": 0,
        "sqlite_jobs": 0,
        "sqlite_failures": 0,
        "cache_dirs": 0,
        "cache_bytes": 0,
    }
    errors: list[str] = []

    with db.connect(settings.state_db) as con:
        # Also prune long-finished jobs and failure records (with 4000-character stack traces); the state
        # database used to only ever grow
        totals["job_history_pruned"] = db.prune_job_history(
            con, retention_days=max(retention_days, int(os.getenv("KB_JOB_HISTORY_DAYS", "30"))), dry_run=dry_run)
        for collection in collections:
            try:
                # Points whose payload lost doc_id/content_version can never be
                # selected by the per-version grouping below; sweep them here
                # so nothing inactive outlives the retention window.
                totals["qdrant_malformed_points"] += delete_malformed_inactive_points(
                    q, collection, cutoff_ts, dry_run=dry_run
                )
            except Exception as exc:
                errors.append(f"{collection}: malformed-point sweep failed: {exc!r}")
            try:
                groups = inactive_doc_versions_older_than(q, collection, cutoff_ts)
            except Exception as exc:
                errors.append(f"{collection}: list inactive versions failed: {exc!r}")
                continue
            for group in groups:
                doc_id = str(group["doc_id"])
                parsed = _parse_doc_id(doc_id)
                if parsed is None:
                    skipped.append({"reason": "invalid_doc_id", **group})
                    continue
                kb_id, file_key = parsed
                content_version = str(group["content_version"])
                file_id = db.file_id_for(kb_id, file_key)
                in_use = _version_in_use(db.get_file_by_id(con, file_id), content_version, cutoff_ts)
                if not in_use:
                    # Leave the whole group alone while another live file with the same content exists in the
                    # same KB, the same rule as the deleted-files section below
                    twins = [row for row in db.active_files_by_checksum(con, kb_id, content_version)
                             if str(row["file_id"]) != file_id]
                    if twins:
                        skipped.append({"reason": "active_same_hash", "active_matches": len(twins), **group})
                        continue

                try:
                    candidates = expired_inactive_point_ids(
                        q, collection, doc_id=doc_id, content_version=content_version, cutoff_ts=cutoff_ts)
                except Exception as exc:
                    errors.append(f"{collection}:{doc_id}:{content_version}: list inactive points failed: {exc!r}")
                    continue
                # Points still active in the ledger are not deleted; while the version is in use, the deleted
                # batch (kept for a restore) is not deleted either
                held = {"active", "deleted"} if in_use else {"active"}
                status_of = db.chunk_status_by_point(con, file_id, candidates)
                point_ids = [p for p in candidates if status_of.get(p) not in held]
                kept_points = len(candidates) - len(point_ids)
                # Clear the ledger and commit first, then delete the points: if this is interrupted, the points
                # left are listed again next round; the other way round, rows left behind would have no points to
                # go by and nobody would ever clean them up. Commit per group so that the write lock is not held
                # across the network calls that follow.
                chunk_count = db.delete_inactive_chunks(con, file_id, point_ids, dry_run=dry_run)
                if not dry_run:
                    con.commit()
                totals["sqlite_chunks"] += chunk_count
                try:
                    point_count = len(point_ids) if dry_run else delete_expired_inactive_points(
                        q, collection, point_ids, cutoff_ts)
                except Exception as exc:
                    errors.append(f"{collection}:{doc_id}:{content_version}: qdrant delete failed: {exc!r}")
                    continue
                totals["qdrant_points"] += point_count
                cache_dir = parse_root / kb_id / str(file_key) / _safe_version(content_version)
                # Look at the files row once more before deleting the cache: the scan may have just taken the
                # file back during the steps above
                if in_use or kept_points or _version_in_use(db.get_file_by_id(con, file_id), content_version, cutoff_ts):
                    cache_result = {"path": str(cache_dir), "exists": cache_dir.exists(), "removed": False, "bytes": 0}
                else:
                    cache_result = _remove_tree(cache_dir, dry_run=dry_run)
                    totals["cache_dirs"] += int(cache_result["removed"] or dry_run and cache_result["exists"])
                    totals["cache_bytes"] += int(cache_result["bytes"])
                qdrant_versions.append(
                    {
                        **group,
                        "in_use": in_use,
                        "qdrant_points_deleted": point_count,
                        "qdrant_points_kept": kept_points,
                        "sqlite_chunks_deleted": chunk_count,
                        "cache_dir": cache_result,
                    }
                )

        for row in db.deleted_files_older_than(con, cutoff_ts):
            kb_id = str(row["kb_id"])
            pending_jobs = db.pending_jobs_for_file(con, str(row["file_id"]))
            if pending_jobs:
                skipped.append(
                    {
                        "reason": "deleted_file_has_pending_jobs",
                        "file_id": row["file_id"],
                        "kb_id": kb_id,
                        "jobs": [
                            {"job_id": job["job_id"], "job_type": job["job_type"], "status": job["status"]}
                            for job in pending_jobs
                        ],
                    }
                )
                continue
            checksum = str(row["checksum"] or row["content_version"] or "")
            active_matches = _active_same_hash_count(con, kb_id, checksum)
            if active_matches:
                skipped.append(
                    {
                        "reason": "deleted_file_active_same_hash",
                        "active_matches": active_matches,
                        "file_id": row["file_id"],
                        "kb_id": kb_id,
                        "checksum": checksum,
                    }
                )
                continue
            try:
                active_points = active_doc_point_count(
                    q,
                    str(row["collection"]),
                    kb_id=kb_id,
                    file_key=int(row["file_key"]),
                )
            except Exception as exc:
                errors.append(f"{row['collection']}:{row['file_id']}: active point count failed: {exc!r}")
                continue
            if active_points:
                skipped.append(
                    {
                        "reason": "deleted_file_still_has_active_points",
                        "active_points": active_points,
                        "file_id": row["file_id"],
                        "kb_id": kb_id,
                        "collection": row["collection"],
                    }
                )
                continue
            cache_result = _remove_tree(
                parse_root / kb_id / str(row["file_key"]),
                dry_run=dry_run,
            )
            state_counts = _purge_file_state(con, str(row["file_id"]), dry_run=dry_run)
            deleted_files.append(
                {
                    "file_id": row["file_id"],
                    "kb_id": kb_id,
                    "source_path": row["source_path"],
                    "deleted_at": row["last_seen_at"],
                    "cache_dir": cache_result,
                    "sqlite_deleted": state_counts,
                }
            )
            totals["sqlite_chunks"] += int(state_counts["chunks"])
            totals["sqlite_files"] += int(state_counts["files"])
            totals["sqlite_jobs"] += int(state_counts["jobs"])
            totals["sqlite_failures"] += int(state_counts["failures"])
            totals["cache_dirs"] += int(cache_result["removed"] or dry_run and cache_result["exists"])
            totals["cache_bytes"] += int(cache_result["bytes"])


    return {
        "skipped": False,
        "dry_run": dry_run,
        "retention_days": retention_days,
        "cutoff_ts": cutoff_ts,
        "totals": totals,
        "qdrant_versions": qdrant_versions,
        "deleted_files": deleted_files,
        "skipped_items": skipped,
        "errors": errors,
    }


def qdrant_backfill_inactive_at(
    settings: Settings,
    *,
    inactive_at_ts: int,
    dry_run: bool = False,
) -> dict[str, object]:
    busy, reasons = service_busy(settings)
    if busy:
        return {"skipped": True, "reason": "service busy", "busy_reasons": reasons}

    from .vector.qdrant import backfill_inactive_at
    from .vector.qdrant import client as qdrant_client

    q = qdrant_client(settings.qdrant_url, settings.qdrant_api_key)
    collections = sorted({source.collection for source in settings.sources.values()})
    backfilled: dict[str, int] = {}
    errors: list[str] = []
    for collection in collections:
        try:
            backfilled[collection] = backfill_inactive_at(q, collection, inactive_at_ts, dry_run=dry_run)
        except Exception as exc:
            backfilled[collection] = 0
            errors.append(f"{collection}: {exc!r}")
    return {
        "skipped": False,
        "dry_run": dry_run,
        "inactive_at_ts": inactive_at_ts,
        "collections": backfilled,
        "total_points": sum(backfilled.values()),
        "errors": errors,
    }


# The two locks that are each held for a few seconds every minute: the scan, and the push into the mirror
_SHORT_LOCKS = frozenset({"kb_scan", "mirror_sync"})


def _mirror_lock_is_orphan(lock: Path) -> bool:
    """The mirror sync lock is created on this machine over ssh by the machine that pushes the mirror: a killed
    push or a power cut leaves it behind, and only that machine's next push reclaims it. Older than
    KB_MIRROR_LOCK_STALE_SECONDS it counts as an orphan and not as busy (it is ignored, not removed), the same
    rule and the same variable as the scan script -- otherwise, while the pushing machine is away, a lock that
    nobody holds would keep blocking the whole maintenance chain."""
    try:
        age = time.time() - lock.stat().st_mtime
    except OSError:
        return False
    return age >= int(os.getenv("KB_MIRROR_LOCK_STALE_SECONDS", "3600"))


def _busy_state(settings: Settings) -> tuple[list[str], list[str]]:
    """(locks held, every reason for being busy). No reasons means not busy."""
    held: list[str] = []
    reasons: list[str] = []
    for name in ("kb_worker", "kb_scan", "mirror_sync", "graph_build"):
        lock = settings.runtime_dir / "state" / f"{name}.lock.d"
        if lock.exists():
            if name == "graph_build" and not build_lock_held(lock):
                continue                      # the graph build lock is judged by flock, not by the directory
            if name == "mirror_sync" and _mirror_lock_is_orphan(lock):
                print(f"[maintenance] ignoring orphaned mirror sync lock: {lock}", flush=True)
                continue
            held.append(name)
            reasons.append(f"lock exists: {lock}")

    if _pgrep(str(settings.runtime_dir.parent / "app" / ".venv" / "bin" / "python") + r".*kb_pipeline.*worker"):
        reasons.append("kb_pipeline worker process running")

    if settings.state_db.exists():
        try:
            with db.connect(settings.state_db) as con:
                rows = con.execute(
                    # Only running jobs with a still-valid lease count as busy. A row left behind by a
                    # SIGKILLed worker is not taken over until the next claim, and meanwhile (a parse lease
                    # can be 12-18h) every GC / rotation / rebuild check / hard delete would be blocked by
                    # one dead record.
                    "SELECT job_id, kb_id, job_type FROM jobs "
                    "WHERE status = 'running' AND COALESCE(locked_until, 0) >= ? "
                    "ORDER BY started_at ASC LIMIT 10",
                    (int(time.time()),),
                ).fetchall()
                reasons.extend(f"running job: {row['job_id']} {row['kb_id']} {row['job_type']}" for row in rows)
                stale = con.execute(
                    "SELECT COUNT(*) FROM jobs WHERE status = 'running' AND COALESCE(locked_until, 0) < ?",
                    (int(time.time()),),
                ).fetchone()[0]
                if int(stale):
                    print(f"[maintenance] ignoring {int(stale)} running job(s) with an expired lease", flush=True)
        except sqlite3.Error as exc:
            reasons.append(f"state db unavailable: {exc!r}")

    mineru_busy = _mineru_busy(settings.mineru_url)
    if mineru_busy:
        reasons.append(mineru_busy)
    return held, reasons


def service_busy(settings: Settings) -> tuple[bool, list[str]]:
    _, reasons = _busy_state(settings)
    return bool(reasons), reasons


def wait_out_short_locks(settings: Settings, *, timeout: float = 60.0, poll: float = 2.0) -> None:
    """When only the scan lock / mirror sync lock stand in the way, wait for them to be released and then
    return, for at most timeout seconds; when not busy, or busy for any other reason, return at once. When the
    timer fires, a maintenance task often starts at the same moment as the once-a-minute scan, and these two
    locks are held for a few seconds only: without waiting, yielding once to a lock held for seconds costs 15
    minutes, and a few such collisions in a row are even recorded as consecutive yields."""
    deadline = time.time() + timeout
    while True:
        held, reasons = _busy_state(settings)
        short_only = bool(reasons) and len(reasons) == len(held) and set(held) <= _SHORT_LOCKS
        if not short_only or time.time() >= deadline:
            return
        time.sleep(poll)


def move_dir_contents(src: Path, dst: Path, *, dry_run: bool, exclude_names: set[str] | None = None) -> int:
    if not src.exists():
        return 0
    excluded = exclude_names or set()
    entries = [item for item in src.iterdir() if item.name != ".DS_Store" and item.name not in excluded]
    if not entries:
        return 0
    if dry_run:
        return len(entries)
    dst.mkdir(parents=True, exist_ok=True)
    moved = 0
    for item in entries:
        target = dst / item.name
        if target.exists():
            target = dst / f"{item.name}.{ts()}"
        shutil.move(str(item), str(target))
        moved += 1
    return moved


def _parse_doc_id(doc_id: str) -> tuple[str, int] | None:
    kb_id, sep, raw_id = doc_id.rpartition(":")
    if not sep or not kb_id:
        return None
    try:
        return kb_id, int(raw_id)
    except ValueError:
        return None


def _safe_version(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)[:120]


def _active_same_hash_count(con: sqlite3.Connection, kb_id: str, checksum: str) -> int:
    if not checksum:
        return 0
    return len(db.active_files_by_checksum(con, kb_id, checksum))


def _version_in_use(file_row: sqlite3.Row | None, content_version: str, cutoff_ts: int) -> bool:
    """Whether the parse cache and restore ledger of this content version are still used, judged by the files
    row alone: it is the file's current content version, or the indexed version still being served (the two
    differ while the content has changed and the new version is not parsed yet). A deleted file may still be
    restored as it was within the retention period, so it counts as in use too; only past the retention
    period, or once the files row is gone, is the version unused."""
    if file_row is None:
        return False
    if content_version not in {str(file_row["content_version"] or ""), str(file_row["indexed_version"] or "")}:
        return False
    if str(file_row["status"]) != "deleted":
        return True
    return int(file_row["last_seen_at"] or 0) >= cutoff_ts


def _purge_file_state(con: sqlite3.Connection, file_id: str, *, dry_run: bool) -> dict[str, int]:
    if dry_run:
        return db.file_state_counts(con, file_id)
    return db.purge_file_state(con, file_id)


def _remove_tree(path: Path, *, dry_run: bool) -> dict[str, object]:
    if not path.exists():
        return {"path": str(path), "exists": False, "removed": False, "bytes": 0}
    size = dir_size(path)
    if not dry_run:
        shutil.rmtree(path, ignore_errors=True)
    return {"path": str(path), "exists": True, "removed": not dry_run, "bytes": size}


def backup_state(settings: Settings, *, base_dir: Path | None = None, keep: int | None = None,
                 dry_run: bool = False) -> dict[str, Any]:
    """Keep a copy of what cannot be regenerated: the state database (SQLite online backup, a consistent snapshot
    even while the pipeline runs; library configuration, the model registry and the extraction caches live in
    it), the pipeline env file, every deployment/*/.env and the question sets and gold standards under
    runtime/eval. Vectors, indexes, graphs and the parse cache are recomputed from the mirror and the caches in
    the state database, so they are left out. Written to <base>/backups/state/<stamp>/ (directory 700, files
    600); only the newest KB_BACKUP_KEEP copies (default 7) are kept. Needs no service, so the nightly round
    runs it before waiting for Qdrant."""
    from .config import BASE_DIR

    base = Path(base_dir) if base_dir else BASE_DIR
    keep = int(os.getenv("KB_BACKUP_KEEP", "7")) if keep is None else int(keep)
    root = base / "backups" / "state"
    target = root / ts()
    state_db = Path(settings.state_db)
    copies: list[tuple[str, Path]] = []
    env_file = Path(settings.env_file)
    if env_file.is_file():
        copies.append((env_file.name, env_file))
    for env in sorted((base / "deployment").glob("*/.env")):
        copies.append((f"deployment-{env.parent.name}.env", env))
    eval_dir = settings.runtime_dir / "eval"
    for item in sorted(eval_dir.iterdir()) if eval_dir.is_dir() else []:
        if item.is_file():
            copies.append((f"eval/{item.name}", item))
    out: dict[str, Any] = {
        "dry_run": dry_run, "backup_dir": str(target), "keep": keep,
        "state_db": str(state_db) if state_db.is_file() else None,
        "files": [name for name, _ in copies], "bytes": 0, "pruned": [],
    }
    if dry_run:
        out["pruned"] = [path.name for path in prune_old_dirs(root, keep=keep - 1, dry_run=True)]
        return out
    root.mkdir(parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    target.mkdir(mode=0o700)
    if state_db.is_file():
        dest = target / state_db.name
        src = sqlite3.connect(state_db)
        try:
            dst = sqlite3.connect(dest)
            try:
                src.backup(dst)
            finally:
                dst.close()
        finally:
            src.close()
        os.chmod(dest, 0o600)
    for name, path in copies:
        dest = target / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        os.chmod(dest, 0o600)
    for sub in target.rglob("*"):
        if sub.is_dir():
            os.chmod(sub, 0o700)
    out["bytes"] = sum(item.stat().st_size for item in target.rglob("*") if item.is_file())
    out["pruned"] = [path.name for path in prune_old_dirs(root, keep=keep, dry_run=False)]
    return out


def prune_old_dirs(parent: Path, *, keep: int, dry_run: bool) -> list[Path]:
    if keep < 0 or not parent.exists():
        return []
    dirs = sorted([item for item in parent.iterdir() if item.is_dir()], key=lambda path: path.name, reverse=True)
    old = dirs[keep:]
    if not dry_run:
        for path in old:
            shutil.rmtree(path, ignore_errors=True)
    return old


def remove_named_files(paths: Iterable[Path], name: str, *, dry_run: bool) -> int:
    found: list[Path] = []
    for path in paths:
        if path.exists():
            found.extend(path.rglob(name))
    if not dry_run:
        for item in found:
            try:
                item.unlink()
            except OSError:
                pass
    return len(found)


def rotate_logs_in_dir(src: Path, dst: Path, *, dry_run: bool) -> int:
    if not src.exists():
        return 0
    count = 0
    for path in src.rglob("*"):
        if not path.is_file() or path.name.endswith(".gz") or path.name in ACTIVE_CLEANUP_LOGS:
            continue
        try:
            if path.stat().st_size == 0:
                continue
        except OSError:
            continue
        rel = path.relative_to(src)
        target = (dst / rel).with_name(rel.name + ".gz")
        if target.exists():
            target = target.with_name(target.name + f".{ts()}")
        count += 1
        if not dry_run:
            gzip_file_and_truncate(path, target)
    return count


def gzip_file_and_truncate(src: Path, dst: Path) -> None:
    """Archive the bytes present when rotation started, then cut exactly that
    prefix: lines appended while gzip ran are shifted to the front instead of
    being discarded by a blanket truncate-to-zero."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    size = src.stat().st_size
    with src.open("rb") as input_file, gzip.open(dst, "wb") as output_file:
        remaining = size
        while remaining > 0:
            chunk = input_file.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            output_file.write(chunk)
            remaining -= len(chunk)
    with src.open("r+b") as handle:
        handle.seek(size)
        tail = handle.read()
        handle.seek(0)
        handle.write(tail)
        handle.truncate(len(tail))


# Entries written only just now are kept for a while: the console's chunk preview writes this cache too, and
# it is outside what service_busy checks
VLM_CACHE_GRACE_SECONDS = 24 * 3600


def prune_vlm_cache(cache_dir: Path, *, dry_run: bool = False) -> dict[str, Any]:
    """The caches of image descriptions, visual vectors and table checks live under vlm-cache keyed by image
    hash, shared across versions and documents; file names start with the image's sha256. Whether an entry
    stays follows the references: while the parse cache still holds the image, the entry stays and never
    expires however old it is (expiring it would send every image through the models again at the next
    re-parse); once the parse cache is reclaimed (the file or the KB deleted, an old version replaced), the
    image's transcription and vectors go with it. When the parse cache directory is missing, or an image in it
    cannot be read, nothing is deleted this round: the references are not fully counted, so there is no
    telling."""
    from .vision.images import IMAGE_SUFFIXES, file_hash

    root = cache_dir / "vlm-cache"
    parse_root = cache_dir / "parse"
    out: dict[str, Any] = {"entries": 0, "referenced": 0, "removed": 0, "unreadable_images": 0}
    if not root.is_dir() or not parse_root.is_dir():
        return out
    referenced: set[str] = set()
    for path in parse_root.rglob("*"):
        if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
            continue
        try:
            referenced.add(file_hash(path))
        except OSError:
            out["unreadable_images"] += 1
    cutoff = time.time() - VLM_CACHE_GRACE_SECONDS
    stale: list[Path] = []
    for path in root.rglob("*"):
        image_hash = path.name[:64]
        if not path.is_file() or not re.fullmatch(r"[0-9a-f]{64}", image_hash):
            continue
        out["entries"] += 1
        if image_hash in referenced:
            out["referenced"] += 1
            continue
        try:
            if path.stat().st_mtime < cutoff:
                stale.append(path)
        except OSError:
            continue
    if out["unreadable_images"]:
        return out
    out["removed"] = len(stale)
    if not dry_run:
        for path in stale:
            path.unlink(missing_ok=True)
    return out


def _pgrep(pattern: str) -> bool:
    try:
        output = subprocess.check_output(["ps", "axo", "pid=,command="], text=True, stderr=subprocess.DEVNULL)
    except subprocess.SubprocessError:
        return False
    current_pid = os.getpid()
    for raw in output.splitlines():
        line = raw.strip()
        if not line:
            continue
        pid_raw, _, command = line.partition(" ")
        try:
            pid = int(pid_raw)
        except ValueError:
            continue
        if pid == current_pid:
            continue
        if re.search(pattern, command):
            return True
    return False


def _mineru_busy(url: str) -> str | None:
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url.rstrip("/") + "/health", timeout=2) as response:
            raw = response.read().decode("utf-8")
        data = json.loads(raw)
    except Exception:
        return None
    queued = int(data.get("queued_tasks") or 0)
    processing = int(data.get("processing_tasks") or 0)
    if queued or processing:
        return f"mineru busy queued={queued} processing={processing}"
    return None
