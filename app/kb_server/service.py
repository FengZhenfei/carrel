"""Bridge between the web API and kb_pipeline. Stateless: settings are
re-loaded per call so enrollment changes take effect without restarts."""

from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

from kb_pipeline import db, discovery, search_fts
from kb_pipeline.config import (
    Settings, load_settings, parse_interval_days, parse_new_chunk_count, parse_ratio,
)
from kb_pipeline.localfs import scanner
from kb_pipeline.limits import (
    normalize_parent_types,
    normalize_predicates,
    CURRENT_SCHEMA_VERSION_ID,
    LEGACY_SCHEMA_VERSION_ID,
    chunk_limits,
    normalize_entity_types,
    push_schema_version,
    validate_chunk_config,
    validate_graph_schema_config,
    validate_graph_tune_config,
    validate_graph_unit_config,
)
from kb_pipeline.pipeline.scheduler import (
    requeue_kb_files,
    requeue_single_file,
    schedule_deletes_for_source,
)
from kb_pipeline.graph.lock import build_lock_held, build_lock_path
from kb_pipeline.vector.qdrant import client as qdrant_client
from kb_pipeline.vector.qdrant import ensure_collection

GRAPH_LLM_STEPS = ("extract", "summarize")
# Model slot used by "Extract labels now / again". It sits in graph_llm alongside the three graph build
# steps but is **not** a prerequisite for building -- a graph can be built without it, you just cannot
# click "Extract labels now / again".
GRAPH_TUNE_STEP = "tune"
_last_kick: dict[str, float] = {}


def settings() -> Settings:
    return load_settings()


# ── service control ─────────────────────────────────────────────────────

def _env_file_keys(env_file: str | Path) -> set[str]:
    """Variable names defined in the env file (same parsing rules as utils.load_env_file)."""
    keys: set[str] = set()
    try:
        lines = Path(env_file).read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return keys
    for raw in lines:
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            keys.add(line.split("=", 1)[0].strip())
    return keys


def _child_env(env_file: str, base_dir: Path) -> dict[str, str]:
    """Environment for the child processes the web process spawns (graph builds, fallback scripts): drop the
    variables the web process snapshotted from the env file at startup so the child re-reads the current file
    itself; every other variable (PATH, XDG_RUNTIME_DIR, the DBUS address etc., which systemd-run needs) is
    inherited as usual.
    Seen on the real box 2026-09-12: the file said KB_GRAPH_LLM_CONCURRENCY=16 when web started and was later
    changed to 128, yet every graph build child spawned by web still ran with 16 -- load_env_file uses
    setdefault, so a value already present in the child's environment is never overridden by the file;
    editing the file changed nothing, only a web restart would have applied it."""
    keys = _env_file_keys(env_file)
    env = {k: v for k, v in os.environ.items() if k not in keys}
    env["KB_ENV_FILE"] = str(env_file)
    env["KB_LOCAL_BASE_DIR"] = str(base_dir)
    return env


def kick(unit: str, fallback_script: str | None = None, *, min_interval: float = 10.0) -> str:
    """Start a oneshot systemd unit now (instead of waiting for its timer).
    Falls back to launching the underlying script directly when systemd is
    unavailable. Rate-limited so the poller can call it opportunistically."""
    now = time.time()
    if now - _last_kick.get(unit, 0.0) < min_interval:
        return "recently-kicked"
    _last_kick[unit] = now
    try:
        proc = subprocess.run(
            ["systemctl", "--user", "start", "--no-block", unit],
            capture_output=True, text=True, timeout=10,
        )
        if proc.returncode == 0:
            return "systemd"
    except Exception:
        pass
    if fallback_script:
        cfg = settings()
        base_dir = cfg.state_db.parent.parent.parent  # <base>/runtime/state/db
        script = base_dir / "scripts" / fallback_script
        if script.exists():
            log_dir = cfg.log_dir
            log_dir.mkdir(parents=True, exist_ok=True)
            with (log_dir / f"web-kick-{fallback_script}.log").open("ab") as log:
                subprocess.Popen(
                    ["/bin/bash", str(script)],
                    stdout=log, stderr=log, start_new_session=True,
                    env=_child_env(str(cfg.env_file), base_dir),
                )
            return "script"
    return "unavailable"


def kick_scan() -> str:
    cfg = settings()
    flag = cfg.runtime_dir / "state" / "mirror_changed.flag"
    try:
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.touch()  # the scan script skips without the mirror flag
    except OSError:
        pass
    return kick("carrel-scan.service", "kb-pipeline-scan.sh")


def kick_worker() -> str:
    return kick("carrel-worker.service", "kb-pipeline-worker-once.sh")


# ── overview / progress ─────────────────────────────────────────────────

_dir_count_cache: dict[str, tuple[float, int]] = {}
_rebuild_check_cache: dict[str, tuple[float, dict[str, Any]]] = {}
REBUILD_CHECK_TTL = 60.0


def rebuild_check_cached(cfg, kb_id: str, *, now: float | None = None) -> dict[str, Any] | None:
    """Current progress against the auto-rebuild conditions (days since the last full rebuild / threshold, new
    chunks / threshold): the very check the timer runs every two hours (evaluate_rebuild), drawn by the
    console as one line under the policy row. One evaluation compares the chunk ledger of the whole knowledge
    base while polling runs every 2.5 s, so the result is cached per knowledge base for 60 seconds;
    unenrolled knowledge bases and those without the graph turned on are skipped (2026-09-23: the user asked
    to show only the progress)."""
    sources = getattr(cfg, "sources", None) or {}
    source = sources.get(kb_id)
    if source is None or not getattr(source, "graph_enabled", False):
        return None
    ts = time.time() if now is None else float(now)
    hit = _rebuild_check_cache.get(kb_id)
    if hit is not None and ts - hit[0] < REBUILD_CHECK_TTL:
        return hit[1]
    from kb_pipeline.graph.build import evaluate_rebuild

    try:
        result = evaluate_rebuild(cfg, source_key=kb_id, source=source)
    except Exception as exc:
        result = {"source": kb_id, "due": False, "reason": "check_failed", "error": repr(exc)}
    keep = {k: result.get(k) for k in ("due", "reason", "skipped_reason", "operator", "appends_since_full",
                                       "baseline_graph_version", "conditions") if k in result}
    _rebuild_check_cache[kb_id] = (ts, keep)
    return keep


def _dir_file_count(mirror_root: Path, name: str, *, ttl: float = 30.0) -> int | None:
    """Pipeline-eligible file count of one top-level directory, walked from
    the filesystem so unenrolled dirs get a number too (the state DB only
    knows enrolled KBs). Same skip rules and extension whitelist as the
    scanner, so the count matches files_total once the KB is enrolled.
    Cached briefly -- the console polls every few seconds."""
    now = time.time()
    hit = _dir_count_cache.get(name)
    if hit and now - hit[0] < ttl:
        return hit[1]
    count = 0
    try:
        for path in (mirror_root / name).rglob("*"):
            try:
                if not path.is_file() or scanner.should_skip(path):
                    continue
            except OSError:
                continue
            if path.suffix.lower() in scanner.SUPPORTED_EXTS or path.name in scanner.SUPPORTED_FILENAMES:
                count += 1
    except OSError:
        return hit[1] if hit else None
    _dir_count_cache[name] = (now, count)
    return count


def _graph_status(con, kb_id: str, dir_name: str, config: dict[str, Any]) -> str:
    """Graph build status, using the same vocabulary as the parse status dot: disabled (graph build not
    turned on) / pending (turned on, not built yet) / running (build in progress) / ok (built) /
    failed (last build failed) / stopped (interrupted, cache still there)."""
    effective = dict(discovery.DEFAULTS)
    effective.update(discovery.DIR_DEFAULTS.get(dir_name, {}))
    effective.update(config)
    if not effective.get("graph_enabled"):
        return "disabled"
    row = con.execute(
        "SELECT status FROM graph_builds WHERE kb_id=? AND status != 'rolled_back' ORDER BY started_at DESC LIMIT 1",
        (kb_id,),
    ).fetchone()
    if row is None:
        return "pending"
    status = str(row["status"])
    if status == "done":
        return "ok"
    if status == "failed":
        return "failed"
    if status == "cancelled":
        # A build interrupted by "Turn off knowledge graph" or by an operator stopping the service. The
        # artifacts are incomplete but the LLM cache is kept, so the next build hits the cache and resumes
        # from where it stopped -- so this is not a failure, it is "to be continued".
        return "stopped"
    return "running"


def _graph_artifacts_exist(con, kb_id: str) -> bool:
    """Whether this knowledge base still has graph artifacts (build records). graph_enabled only expresses
    the user's intent; whether artifacts exist is a separate matter -- keeping the two apart is what stops
    the delete entry point from being locked by the switch state."""
    row = con.execute("SELECT 1 FROM graph_builds WHERE kb_id=? LIMIT 1", (kb_id,)).fetchone()
    return row is not None


def _paused_cache_reuse(con, kb_id: str, row) -> dict[str, Any]:
    """Whether the cache of the paused build can still be resumed now.

    "Resume build" is a promise about cost: LLM calls that already completed come back instantly. But the
    cache key is the hash of a single call's input_args, which includes both model and messages -- switch
    the build model, extract labels again or change the chunking parameters and the old cache is missed
    entirely. Clicking "Resume build" would then really be a full re-run of several hours, while the button
    says "Resume".

    So two things are compared: the config fingerprint recorded when the build started, and the corpus
    fingerprint recorded when it finished. If either does not match, the console switches the button back
    to "Build now / again" and says which one changed.
    """
    from kb_pipeline.graph.build import graph_cache_fingerprint, source_snapshot_hash

    cfg = settings()
    try:
        kb_row = con.execute(
            "SELECT source_root, collection FROM kb_sources WHERE kb_id=?", (kb_id,)).fetchone()
        source = discovery.build_source(
            cfg.mirror_root, str(kb_row["source_root"]), discovery.get_config(con, kb_id),
            kb_id=kb_id, collection=str(kb_row["collection"]))
        now_fp = graph_cache_fingerprint(cfg, source)
    except Exception:
        # No model selected, knowledge base deleted, config unreadable -- none of these can promise "Resume"
        return {"cache_reusable": False, "cache_stale": "Config unreadable, cannot tell whether the cache can be reused"}

    stale: list[str] = []
    stored_fp = str(row["cache_fingerprint"] or "")
    if not stored_fp:
        # This column was added later. Old records have no fingerprint, so the config at the time is
        # unknown and nothing is promised.
        stale.append("this build predates the fingerprint record")
    elif stored_fp != now_fp:
        stale.append("labels / predicates / model / unit settings changed")
    stored_corpus = str(row["source_content_hash"] or "")
    if stored_corpus and source_snapshot_hash(
            db.active_chunk_refs(con, source.collection)) != stored_corpus:
        stale.append("corpus changed")
    return {"cache_reusable": not stale, "cache_stale": " · ".join(stale)}


def _graph_build_info(con, kb_id: str, collection: str = "") -> dict[str, Any] | None:
    """Details of the most recent graph build (phase check-ins / timestamps / error), for the
    "Graph build status" panel."""
    row = con.execute(
        "SELECT graph_build_id, graph_version, status, stage, started_at, finished_at, error, "
        "cache_fingerprint, source_content_hash, build_kind FROM graph_builds "
        "WHERE kb_id=? AND status != 'rolled_back' ORDER BY started_at DESC LIMIT 1",
        (kb_id,),
    ).fetchone()
    if row is None:
        return None
    info = dict(row)
    # How many appends have landed since the last full-build version: the status card explains
    # "this graph = full build + N appends"
    full = con.execute(
        "SELECT COALESCE(finished_at, started_at) AS ts FROM graph_builds WHERE kb_id=? AND status='done' "
        "AND build_kind='full' ORDER BY finished_at DESC, started_at DESC LIMIT 1", (kb_id,)).fetchone()
    info["appends_since_full"] = db.count_graph_builds_since(
        con, kb_id, kind="append", since_ts=int(full["ts"] or 0)) if full is not None else 0
    # Only count cache entries and completed phases once the build has stopped (paused or failed): it tells
    # the user how much "Resume" gets for free. Polling runs every 2.5 s and should not walk a directory of
    # thousands of files for a number that is only shown while stopped.
    if collection and str(info.get("status")) in ("cancelled", "failed"):
        from kb_pipeline.graph.build import GRAPH_PHASE_LABELS

        labels = dict(GRAPH_PHASE_LABELS)
        info["cache_entries"] = graph_cache_entries(settings(), collection)
        info["extraction_entries"] = db.graph_extraction_count(con, kb_id)
        info.update(_paused_cache_reuse(con, kb_id, row))
        info["phases_done"] = [labels.get(p, p) for p in db.graph_phases_done(con, str(row["graph_build_id"]))
                               if p in labels]
    # Since the build history table was removed (2026-09-06), size and resolution numbers live on the status
    # card: take the manifest of the most recent successful version
    done_row = con.execute(
        "SELECT * FROM graph_builds WHERE kb_id=? AND status='done' ORDER BY finished_at DESC, started_at DESC LIMIT 1",
        (kb_id,)).fetchone()
    info["active_graph_version"] = None
    if done_row is not None:
        try:
            summary = _graph_build_summary(dict(done_row), [], {})
            info["counts"] = summary.get("counts")
            info["resolution"] = summary.get("resolution")
            info["counts_at"] = done_row["finished_at"] or done_row["started_at"]
            # Units / documents whose extraction failed: at <=5% the build still completes, but it has to be
            # visible on the status card (health check R3)
            extract = ((summary.get("stats") or {}).get("extract") or {})
            info["extract_failed_units"] = extract.get("failed_units")
            info["extract_failed_documents"] = extract.get("failed_documents") or []
            facts_stats = ((summary.get("stats") or {}).get("facts") or {})
            info["facts_partial_units"] = int(facts_stats.get("partial_units_total") if facts_stats.get("partial_units_total") is not None
                                              else len(facts_stats.get("partial_units") or []))
            info["evidence_conflict_facts"] = facts_stats.get("evidence_conflict_facts")
        except Exception:
            info["counts"] = None
        # The graph preview draws this version (the most recently completed one that is not a trial build),
        # kept apart from the new version number of a build in progress: the frontend uses it to decide
        # "does the preview need re-fetching", so a running build no longer redraws on every poll
        # (health check B3)
        active = db.latest_successful_graph_build(con, str(done_row["source_key"]))
        info["active_graph_version"] = str(active["graph_version"]) if active is not None else None
    # Verdict of the most recent scheduled check (health check D7) and the actual per-phase durations of the
    # last completed build -- the progress bar weights (health check D8)
    info["last_check"] = db.latest_graph_check(con, kb_id)
    info["stage_weights"] = {}
    if done_row is not None:
        from kb_pipeline.graph.build import stage_weights_from_phases

        from kb_pipeline.graph.build import merge_stage_weights

        # Take the per-phase maximum across the last few completed versions: an append or a cache-hit
        # extraction takes only seconds, and weighting from that one alone would flatten the progress bar of
        # the next from-scratch extraction
        recent = con.execute(
            "SELECT graph_build_id, started_at FROM graph_builds WHERE kb_id=? AND status='done' "
            "ORDER BY finished_at DESC, started_at DESC LIMIT 5", (kb_id,)).fetchall()
        info["stage_weights"] = merge_stage_weights([
            stage_weights_from_phases(int(r["started_at"] or 0), db.graph_phase_timestamps(con, str(r["graph_build_id"])))
            for r in recent])
    # The fingerprints are of no use to the frontend; keep them out of the 2.5-second polling response
    info.pop("cache_fingerprint", None)
    info.pop("source_content_hash", None)
    # A pause is persistent and the panel has to say so -- otherwise "paused" and "auto-rebuild may resume
    # it any moment" look exactly the same in the UI.
    info["paused"] = bool(discovery.get_config(con, kb_id).get("graph_paused"))
    return info


def _schema_suggest_info(con, kb_id: str) -> dict[str, Any] | None:
    """The "extraction in progress" mark schema_flow writes to app_config during label extraction; a stale
    one (left behind by a killed process) does not count."""
    from kb_pipeline.graph.schema_flow import SUGGEST_MARK_FRESH_SECONDS, SUGGEST_MARK_PREFIX

    mark = db.get_app_config(con, SUGGEST_MARK_PREFIX + kb_id)
    if not isinstance(mark, dict):
        return None
    started = int(mark.get("started_at") or 0)
    if time.time() - started > SUGGEST_MARK_FRESH_SECONDS:
        return None
    return {"origin": mark.get("origin"), "started_at": started, "seconds": max(0, int(time.time() - started))}


def overview() -> dict[str, Any]:
    cfg = settings()
    dirs = discovery.discover_directories(cfg.mirror_root)
    # The schema was created once in init_state() during lifespan; the 2.5-second poll need not re-run
    # CREATE IF NOT EXISTS plus twenty-odd trial ALTER TABLEs every time (health check D3)
    kbs: list[dict[str, Any]] = []
    parsing_active = False
    with db.connect(cfg.state_db) as con:
        # When a graph build child died but left a running row behind, mark it failed here -- otherwise the
        # console shows "building" forever and build now / delete graph are refused forever.
        recovered = db.reconcile_stale_graph_builds(con)
        if recovered:
            print(f"[web] recovered stale graph builds: {','.join(recovered)}", flush=True)
        rows = {str(r["source_root"]): r for r in discovery.known_sources(con)}
        for name in dirs:
            row = rows.pop(name, None)
            # Only unenrolled directories need a filesystem walk for their count; enrolled knowledge bases use
            # files_total from the DB (a more accurate figure too), sparing the rglob+stat over a large
            # directory every 30 seconds.
            entry: dict[str, Any] = {"dir": name}
            if row is None or str(row["status"]) != "active":
                entry["dir_files"] = _dir_file_count(cfg.mirror_root, name)
            if row is None:
                # ids are allocated at enrollment; an unenrolled dir has none.
                # dir_defaults prefill the console's draft config form.
                entry["state"] = "unenrolled"
                entry["dir_defaults"] = discovery.DIR_DEFAULTS.get(name, {})
                kbs.append(entry)
                continue
            entry["kb_id"] = str(row["kb_id"])
            entry["collection"] = str(row["collection"])
            if str(row["status"]) == "active":
                entry["state"] = "active"
                stats = _kb_stats(con, entry["kb_id"])
                entry.update(stats)
                if stats["files_total"] == 0:
                    # Just enabled and the scan has not enrolled the files yet: the sidebar shows the count
                    # from disk for now (marked "waiting for the scan to enroll"), otherwise the moment of
                    # enabling jumps from "N files" to "0 files" and back once the scan finishes (2026-09-08)
                    entry["dir_files"] = _dir_file_count(cfg.mirror_root, name)
                if stats["jobs_active"] > 0:
                    parsing_active = True
                    # Phase weights for the parse progress bar: the actual per-phase durations (seconds) of
                    # this knowledge base's most recent parses
                    entry["parse_stage_weights"] = _parse_stage_weights_cached(con, entry["kb_id"])
                try:
                    kb_config = discovery.get_config(con, entry["kb_id"])
                except KeyError:
                    kb_config = {}
                # The whole config (including the long vlm_prompt text) is no longer returned with every
                # 2.5-second poll: the frontend only reads config via /kbs/{id}/config, so here it was pure
                # bandwidth waste.
                entry["graph_status"] = _graph_status(con, entry["kb_id"], name, kb_config)
                entry["graph_build"] = _graph_build_info(
                    con, entry["kb_id"], str(row["collection"]))
                entry["rebuild_check"] = rebuild_check_cached(cfg, entry["kb_id"])
                # Once the switch is off graph_status becomes disabled, but artifacts may still exist (turned
                # off mid-build, or the last delete failed). The frontend uses this to decide whether
                # "Delete knowledge graph" is clickable, so an existing graph never becomes an orphan that
                # cannot be cleaned up.
                entry["graph_artifacts"] = _graph_artifacts_exist(con, entry["kb_id"])
                # Whether a label extraction is running for this knowledge base (started from the console or
                # automatically before a build): the console shows the "Extract labels" button as extracting,
                # stays truthful across switching KBs / tabs / reloads, and re-fetches the config once it is
                # done so the new version appears
                entry["schema_suggest"] = _schema_suggest_info(con, entry["kb_id"])
            else:
                entry["state"] = "inactive"
                try:
                    inactive_cfg = json.loads(row["config_json"] or "{}")
                except Exception:
                    inactive_cfg = {}
                entry["graph_status"] = _graph_status(con, entry["kb_id"], name, inactive_cfg)
                entry["graph_build"] = _graph_build_info(
                    con, entry["kb_id"], str(row["collection"]))
                entry["graph_artifacts"] = _graph_artifacts_exist(con, entry["kb_id"])
                entry["inactive_reason"] = row["inactive_reason"] if "inactive_reason" in row.keys() else None
                inactive_at = int(row["inactive_at"] or 0)
                retention = cfg.qdrant_inactive_retention_days * 86400
                # Reaching this branch means the directory **is on disk** (the outer loop walks existing
                # directories). kb_sources_gc exempts knowledge bases deactivated because their directory
                # vanished: once the directory is back it is skipped and never hard-deleted. This code used
                # to emit a countdown anyway, so it reached 0 and stayed there -- "permanently deleted in
                # 0 days" was shown forever while nothing would ever be deleted. Unenrolling from the console
                # (unenrolled) gets no such exemption, so that countdown is real.
                gc_exempt = str(entry["inactive_reason"] or "") != "unenrolled"
                entry["gc_exempt"] = gc_exempt
                entry["gc_in_seconds"] = (
                    None if gc_exempt or not inactive_at
                    else max(0, inactive_at + retention - int(time.time()))
                )
            kbs.append(entry)
        # rows whose directory vanished entirely (still registered)
        for name, row in rows.items():
            still_active = str(row["status"]) == "active"
            linked = still_active and discovery.directory_admitted(cfg.mirror_root, name) == (False, "linked")
            entry = {
                "dir": name, "kb_id": str(row["kb_id"]), "collection": str(row["collection"]),
                "state": ("directory_linked" if linked else "directory_missing") if still_active else "inactive",
                "inactive_reason": row["inactive_reason"] if "inactive_reason" in row.keys() else None,
            }
            if not still_active:
                # Directory really gone + deactivated = the retention period is genuinely running, and GC will
                # hard-delete the collection, graph and index together when it expires. This is exactly the
                # branch the countdown used to miss: it was shown on the branch where the directory still
                # exists and nothing is ever deleted -- the two were reversed.
                inactive_at = int(row["inactive_at"] or 0)
                entry["gc_exempt"] = False
                entry["gc_in_seconds"] = (
                    max(0, inactive_at + cfg.qdrant_inactive_retention_days * 86400 - int(time.time()))
                    if inactive_at else None
                )
            kbs.append(entry)
        active_jobs = con.execute(
            "SELECT j.job_id, j.kb_id, j.status, j.stage, j.updated_at, j.started_at, f.rel_path, f.filename "
            "FROM jobs j LEFT JOIN files f ON f.file_id = j.file_id "
            "WHERE j.job_type = 'parse' AND j.status IN ('running', 'queued', 'retry') "
            "ORDER BY CASE j.status WHEN 'running' THEN 0 WHEN 'queued' THEN 1 ELSE 2 END, j.updated_at DESC LIMIT 30"
        ).fetchall()
        jobs_out: list[dict[str, Any]] = []
        now_ts = int(time.time())
        for r in active_jobs:
            item = dict(r)
            if item.get("status") == "running":
                # Seconds elapsed in the current phase, computed server-side for the frontend: the progress bar
                # interpolates within the phase from it, immune to clock skew between the two machines
                item["stage_elapsed"] = stage_elapsed(con, str(item["job_id"]), now_ts)
            jobs_out.append(item)
    if parsing_active:
        # opportunistic: make sure a worker run is actually consuming the queue
        kick_worker()
    return {
        "kbs": kbs,
        "parsing_active": parsing_active,
        "active_jobs": jobs_out,
        "parse_enabled": cfg.parse_enabled,
        "retention_days": cfg.qdrant_inactive_retention_days,
    }


# "Occupying the pipeline" = running, or a queued / retry job whose claim time has arrived. A retry job
# still in its backoff wait (next_attempt_at in the future) cannot be claimed by the worker at all and must
# not count as busy: otherwise one failed file makes the console pretend it is parsing, kicks the worker
# for nothing every 10 seconds and blocks graph operations for up to an hour (the backoff cap).
# The criterion itself lives in db.CLAIMABLE_JOB_SQL -- the automatic graph rebuild (cli) uses the same one;
# two separate copies would drift sooner or later: manual builds allowed while automatic ones yield, or
# the other way round.
_CLAIMABLE_SQL = db.CLAIMABLE_JOB_SQL
_WAITING_SQL = "(status IN ('queued','retry') AND next_attempt_at > ?)"


def _stage_head(stage: str | None) -> str:
    return str(stage or "").split("(", 1)[0].strip()


def parse_stage_weights(con, kb_id: str, *, limit: int = 8) -> dict[str, float]:
    """Median actual duration (seconds) of each phase over this knowledge base's most recent parses,
    aggregated by phase head (the part before the parenthesis), reading only jobs / job_events. The parse
    progress bar uses it to size each phase's span and interpolates within a phase by elapsed time: the
    MinerU step reports no sub-progress, and with fixed weights the bar sat motionless at 9% for over a
    hundred seconds and then jumped to 70% (2026-09-08). Lives in the console layer, does not touch
    pipeline code."""
    rows = con.execute(
        """
        SELECT e.job_id, e.ts, e.text, j.finished_at
        FROM job_events e
        JOIN (SELECT job_id, finished_at FROM jobs WHERE kb_id = ? AND job_type = 'parse' AND status = 'done'
              ORDER BY finished_at DESC LIMIT ?) j ON j.job_id = e.job_id
        WHERE e.kind = 'stage'
        ORDER BY e.job_id, e.ts, e.rowid
        """,
        (kb_id, limit),
    ).fetchall()
    by_job: dict[str, list] = {}
    for r in rows:
        by_job.setdefault(str(r["job_id"]), []).append(r)
    samples: dict[str, list[float]] = {}
    for events in by_job.values():
        for i, e in enumerate(events):
            head = _stage_head(e["text"])
            if not head or head == "Done":
                continue
            end = int(events[i + 1]["ts"]) if i + 1 < len(events) else int(e["finished_at"] or e["ts"])
            samples.setdefault(head, []).append(float(max(0, end - int(e["ts"]))))
    return {head: float(statistics.median(v)) for head, v in samples.items() if v}


def stage_elapsed(con, job_id: str, now: int | None = None) -> int | None:
    """Seconds a running job has spent in its current phase (measured from the latest phase event); None when
    there is no phase event."""
    row = con.execute("SELECT MAX(ts) AS ts FROM job_events WHERE job_id = ? AND kind = 'stage'", (job_id,)).fetchone()
    if row is None or row["ts"] is None:
        return None
    return max(0, int(now if now is not None else time.time()) - int(row["ts"]))


_stage_weight_cache: dict[str, tuple[float, dict[str, float]]] = {}


def _parse_stage_weights_cached(con, kb_id: str, *, ttl: float = 30.0) -> dict[str, float]:
    """parse_stage_weights with a 30-second cache: polling runs every 2.5 s, while the weights can only change
    every few minutes."""
    now = time.time()
    hit = _stage_weight_cache.get(kb_id)
    if hit and now - hit[0] < ttl:
        return dict(hit[1])
    weights = parse_stage_weights(con, kb_id)
    _stage_weight_cache[kb_id] = (now, weights)
    return dict(weights)


def _kb_stats(con, kb_id: str) -> dict[str, int]:
    def one(sql: str, *params) -> int:
        row = con.execute(sql, params).fetchone()
        return int(row[0] if row else 0)

    total = one("SELECT COUNT(*) FROM files WHERE kb_id=? AND status != 'deleted'", kb_id)
    indexed = one(
        "SELECT COUNT(*) FROM files WHERE kb_id=? AND status != 'deleted' AND COALESCE(indexed_version,'') = content_version",
        kb_id,
    )
    failed = one(
        "SELECT COUNT(*) FROM files f WHERE f.kb_id=? AND f.status != 'deleted' "
        "AND COALESCE(f.indexed_version,'') != f.content_version "
        "AND EXISTS (SELECT 1 FROM jobs j WHERE j.file_id=f.file_id AND j.job_type='parse' AND j.status='failed') "
        "AND NOT EXISTS (SELECT 1 FROM jobs j2 WHERE j2.file_id=f.file_id AND j2.job_type='parse' AND j2.status IN ('queued','retry','running'))",
        kb_id,
    )
    now_ts = int(time.time())
    jobs_active = one(
        f"SELECT COUNT(*) FROM jobs WHERE kb_id=? AND job_type='parse' AND {_CLAIMABLE_SQL}",
        kb_id, now_ts,
    )
    jobs_waiting = one(
        f"SELECT COUNT(*) FROM jobs WHERE kb_id=? AND job_type='parse' AND {_WAITING_SQL}",
        kb_id, now_ts,
    )
    # A full re-parse overwrites in place: files stay indexed throughout. The progress bar has to subtract
    # the "indexed but being re-parsed" files from the completed count, otherwise a re-parse shows 100%
    # from the very start.
    reparsing = one(
        "SELECT COUNT(*) FROM files f WHERE f.kb_id=? AND f.status != 'deleted' "
        "AND COALESCE(f.indexed_version,'') = f.content_version "
        "AND EXISTS (SELECT 1 FROM jobs j WHERE j.file_id=f.file_id AND j.job_type='parse' "
        "AND j.status IN ('queued','retry','running'))",
        kb_id,
    )
    pending = max(0, total - indexed - failed)
    return {"files_total": total, "files_indexed": indexed, "files_failed": failed,
            "files_pending": pending, "jobs_active": jobs_active,
            "jobs_waiting": jobs_waiting, "files_reparsing": reparsing}


def kb_files(kb_id: str) -> list[dict[str, Any]]:
    """File table: besides the status dot it also gives chunk count, indexed time and parser profile -- the
    console sorts and filters on these, so nobody has to open files one by one to learn how many chunks a
    file produced or which parser version indexed it."""
    cfg = settings()
    with db.connect(cfg.state_db) as con:
        rows = con.execute(
            "SELECT file_id, rel_path, filename, size, mtime, status, content_version, indexed_version, "
            "indexed_parser_profile, last_seen_at, chunk_diag_json FROM files "
            "WHERE kb_id=? AND status != 'deleted' ORDER BY rel_path", (kb_id,)
        ).fetchall()
        jobs = con.execute(
            "SELECT job_id, file_id, status, stage, error, updated_at FROM jobs "
            "WHERE kb_id=? AND job_type='parse' ORDER BY created_at", (kb_id,)
        ).fetchall()
        chunk_counts = {str(r["file_id"]): int(r["n"]) for r in con.execute(
            "SELECT c.file_id, COUNT(*) AS n FROM chunks c JOIN files f ON f.file_id = c.file_id "
            "WHERE f.kb_id=? AND c.status='active' GROUP BY c.file_id", (kb_id,))}
        indexed_at = {str(r["file_id"]): int(r["t"]) for r in con.execute(
            "SELECT file_id, MAX(finished_at) AS t FROM jobs WHERE kb_id=? AND job_type='parse' "
            "AND status='done' AND finished_at IS NOT NULL GROUP BY file_id", (kb_id,))}
    latest: dict[str, Any] = {}
    for j in jobs:
        latest[str(j["file_id"])] = j  # created_at ascending: last wins
    out = []
    for r in rows:
        file_id = str(r["file_id"])
        job = latest.get(file_id)
        job_status = str(job["status"]) if job else None
        if str(r["content_version"]) == str(r["indexed_version"] or ""):
            dot = "green"
        elif job_status in {"queued", "retry", "running"}:
            dot = "yellow"
        elif job_status == "failed":
            dot = "red"
        else:
            dot = "yellow"
        out.append({
            "file_id": file_id,
            "rel_path": str(r["rel_path"]),
            "size": int(r["size"]),
            "mtime": int(r["mtime"] or 0),
            "dot": dot,
            "job_status": job_status,
            "job_id": (str(job["job_id"]) if job else None),
            "stage": (str(job["stage"]) if job and job["stage"] else None),
            "error": (str(job["error"])[:400] if job and job["error"] and job_status == "failed" else None),
            "chunk_diag": _chunk_diag_brief(r["chunk_diag_json"]) if dot == "green" else None,
            "chunks": chunk_counts.get(file_id) if dot == "green" else None,
            "indexed_profile": (str(r["indexed_parser_profile"]) if dot == "green" and r["indexed_parser_profile"] else None),
            "indexed_at": indexed_at.get(file_id) if dot == "green" else None,
        })
    return out


_JOB_LIST_FILTERS = {
    "all": "",
    "active": " AND j.status IN ('running','queued','retry')",
    "failed": " AND j.status IN ('failed','retry')",
    "done": " AND j.status IN ('done','cancelled')",
}


def kb_jobs(kb_id: str, status: str = "all", limit: int = 200) -> dict[str, Any]:
    """One knowledge base's job list plus its own queue depth, for the "Jobs" tab. Running jobs come first,
    then failed / backing off (the ones a human needs to look at), then queued, and finally the history in
    reverse chronological order."""
    if status not in _JOB_LIST_FILTERS:
        raise ValueError("status must be one of all / active / failed / done")
    limit = max(1, min(int(limit), 1000))
    cfg = settings()
    with db.connect(cfg.state_db) as con:
        rows = con.execute(
            "SELECT j.job_id, j.file_id, j.job_type, j.status, j.stage, j.error, j.retry_count, "
            "j.next_attempt_at, j.started_at, j.finished_at, j.created_at, j.updated_at, "
            "j.cancel_requested, f.rel_path FROM jobs j LEFT JOIN files f ON f.file_id = j.file_id "
            f"WHERE j.kb_id=? {_JOB_LIST_FILTERS[status]} "
            "ORDER BY CASE j.status WHEN 'running' THEN 0 WHEN 'failed' THEN 1 WHEN 'retry' THEN 2 "
            "WHEN 'queued' THEN 3 ELSE 4 END, j.updated_at DESC LIMIT ?",
            (kb_id, limit),
        ).fetchall()
        depth = {r["status"]: int(r["n"]) for r in con.execute(
            "SELECT status, COUNT(*) AS n FROM jobs WHERE kb_id=? AND job_type='parse' "
            "AND status IN ('running','queued','retry') GROUP BY status", (kb_id,))}
    now = int(time.time())
    items = []
    for r in rows:
        st = str(r["status"])
        items.append({
            "job_id": str(r["job_id"]), "file_id": r["file_id"], "job_type": str(r["job_type"]),
            "status": st, "stage": r["stage"],
            "error": (str(r["error"])[:300] if r["error"] and st in ("failed", "retry") else None),
            "retry_count": int(r["retry_count"] or 0),
            "retry_in": max(0, int(r["next_attempt_at"] or 0) - now) if st == "retry" else None,
            "started_at": r["started_at"], "finished_at": r["finished_at"],
            "created_at": r["created_at"], "updated_at": r["updated_at"],
            "cancel_requested": bool(r["cancel_requested"]),
            "rel_path": r["rel_path"],
        })
    return {"jobs": items, "queue": {k: depth.get(k, 0) for k in ("running", "queued", "retry")}}


def _col(row, name: str, default=None):
    """Works on sqlite3.Row and dict alike; a missing column (old records, hand-made rows in tests) yields
    the default."""
    try:
        return row[name] if name in row.keys() else default
    except (AttributeError, TypeError):
        return default


def _graph_build_summary(row, phases: list[dict[str, Any]], labels: dict[str, str]) -> dict[str, Any]:
    """Readable summary of one graph build record: kind, size, resolution effect, completed phases. All
    numbers come from the manifest (the one persisted when the build finished); a step that never ran has
    none, and the UI leaves it blank rather than guessing."""
    try:
        manifest = json.loads(row["manifest_json"] or "{}")
    except (TypeError, ValueError):
        manifest = {}
    build_kind = str(manifest.get("build_kind") or _col(row, "build_kind") or "full")
    # How much an incremental append reused: vectors summed per collection (manifest.enrich), the resolution
    # replay is in resolution
    reuse = None
    collections = ((manifest.get("enrich") or {}).get("collections") or {}) if isinstance(manifest.get("enrich"), dict) else {}
    if any("reused" in (c or {}) for c in collections.values()):
        reuse = {"vectors_reused": sum(int((c or {}).get("reused") or 0) for c in collections.values()),
                 "vectors_embedded": sum(int((c or {}).get("embedded") or 0) for c in collections.values())}
    # The summary-tree (raptor) mode was removed on 2026-09-05; old records remain, and only a few numbers
    # are read from their old manifests for the history table
    mode = "raptor" if (manifest.get("mode") == "raptor" or "raptor" in manifest) else "entity_graph"
    counts: dict[str, Any] | None = None
    resolution = None
    if mode == "raptor":
        stats = manifest.get("raptor") or {}
        if stats:
            counts = {k: stats.get(k) for k in ("nodes", "documents", "llm_calls", "cache_hits", "degraded")}
    else:
        graph = manifest.get("graph") or {}
        expected = (manifest.get("neo4j_import") or {}).get("expected_counts") or {}
        if graph or expected:
            counts = {
                "entities": graph.get("entities", expected.get("entities")),
                "relations": graph.get("relations", expected.get("relations")),
                "text_units": graph.get("units", expected.get("text_units")),
                "documents": (manifest.get("input") or {}).get("documents", expected.get("documents")),
                "mentions": graph.get("mentions"),
            }
        res = graph.get("resolution") if isinstance(graph, dict) else None
        if isinstance(res, dict):
            resolution = res
        merge_stats = graph.get("merge") if isinstance(graph, dict) else None
        if isinstance(merge_stats, dict) and counts is not None:
            counts["type_violations"] = merge_stats.get("type_violations")
            counts["schema_drift_types"] = merge_stats.get("schema_drift_types")
            counts["schema_drift_predicates"] = merge_stats.get("schema_drift_predicates")
            counts["negated_dropped"] = merge_stats.get("negated_dropped")
            counts["orphan_dropped"] = merge_stats.get("orphan_dropped")
            # Noise-reduction stats (2026-09-04): boilerplate / listing units, boilerplate-only relations,
            # value folding, reference entities, demoted types
            for key in ("boilerplate_units", "listing_units", "boilerplate_relations", "boilerplate_entities",
                        "value_entities_dropped", "value_relations_folded", "reference_entities", "demoted_types"):
                counts[key] = merge_stats.get(key)
            counts["demoted_type_names"] = merge_stats.get("demoted_type_names") or []
            counts["noisy_type_names"] = merge_stats.get("noisy_type_names") or []
            counts["scoped_entities"] = merge_stats.get("scoped_entities")
            counts["derived_variant_edges"] = merge_stats.get("derived_variant_edges")
        facts_block = manifest.get("facts") or {}
        if counts is not None and facts_block and not facts_block.get("skipped"):
            counts["specs"] = facts_block.get("facts")
            counts["spec_units"] = facts_block.get("units")
            counts["specs_linked"] = facts_block.get("subjects_linked")
    input_docs = (manifest.get("input") or {}).get("documents")
    # The three stat blocks for the build record details: corpus figures (chunks per unit, average tokens,
    # gleaning rounds), the extraction phase (units / cache / calls / failures / parse quality) and the
    # summary phase. All picked verbatim from the manifest; a step that never ran is None and the UI leaves
    # it blank.
    input_block = manifest.get("input") or {}
    extract_block = manifest.get("extract") or {}
    graph_block = manifest.get("graph") or {}
    stats = {
        "input": {**{k: input_block.get(k) for k in
                     ("units", "unit_chunks", "avg_unit_chunks", "avg_unit_tokens", "max_unit_tokens", "max_gleanings",
                      "units_by_kind")}}
        if input_block else None,
        "extract": {
            **{k: extract_block.get(k) for k in ("units", "extracted", "cached", "failed_units", "failed_documents", "entities_total")},
            "llm": extract_block.get("llm"),
            "parse": extract_block.get("parse"),
        } if extract_block else None,
        "summaries": graph_block.get("summaries") if isinstance(graph_block, dict) else None,
        "llm": graph_block.get("llm") if isinstance(graph_block, dict) else None,
        "facts": manifest.get("facts") if isinstance(manifest.get("facts"), dict) else None,
        "predicate_health": (graph_block.get("merge") or {}).get("predicate_health") if isinstance(graph_block, dict) else None,
        "reuse": reuse,
    }
    return {
        "graph_build_id": str(row["graph_build_id"]),
        "graph_version": str(row["graph_version"]),
        "status": str(row["status"]),
        "stage": row["stage"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "mode": mode,
        "build_kind": build_kind,
        "base_version": manifest.get("base_version"),
        "delta": manifest.get("delta") if isinstance(manifest.get("delta"), dict) else None,
        "input_rows": int(input_docs if isinstance(input_docs, int) else (row["input_rows"] or 0)),
        "active_chunk_count": int(row["active_chunk_count"] or 0),
        "counts": counts,
        "resolution": resolution,
        "stats": stats,
        "phases": [{"phase": p["phase"], "label": labels.get(p["phase"], p["phase"]), "done_at": p["done_at"]}
                   for p in phases],
        "resumed_phases": [labels.get(p, p) for p in manifest.get("resumed_phases") or []],
        "steps": list(manifest.get("steps") or []),
        "error": (str(row["error"])[:600] if row["error"] else None),
    }


def graph_builds(kb_id: str, limit: int = 30) -> dict[str, Any]:
    """One knowledge base's graph build records (newest first), for the history table on the "Graph" tab."""
    from kb_pipeline.graph.build import GRAPH_PHASE_LABELS

    labels = dict(GRAPH_PHASE_LABELS)
    limit = max(1, min(int(limit), 200))
    cfg = settings()
    with db.connect(cfg.state_db) as con:
        rows = con.execute(
            "SELECT graph_build_id, graph_version, status, stage, started_at, finished_at, input_rows, "
            "active_chunk_count, output_dir, manifest_json, error, build_kind FROM graph_builds "
            "WHERE kb_id=? ORDER BY started_at DESC LIMIT ?", (kb_id, limit)).fetchall()
        phases: dict[str, list[dict[str, Any]]] = {}
        if rows:
            marks = ",".join("?" for _ in rows)
            for p in con.execute(
                    f"SELECT graph_build_id, phase, done_at FROM graph_build_phases "
                    f"WHERE graph_build_id IN ({marks}) ORDER BY done_at, rowid",
                    [str(r["graph_build_id"]) for r in rows]):
                phases.setdefault(str(p["graph_build_id"]), []).append(
                    {"phase": str(p["phase"]), "done_at": int(p["done_at"])})
    return {"builds": [_graph_build_summary(r, phases.get(str(r["graph_build_id"]), []), labels) for r in rows]}


def _chunk_diag_brief(raw) -> dict[str, Any] | None:
    """The file list only needs the verdict: chunk count, mean length, whether acceptance passed and, if
    not, why."""
    if not raw:
        return None
    try:
        diag = json.loads(raw)
    except (TypeError, ValueError):
        return None
    stats = diag.get("stats") or {}
    return {
        "ok": bool(diag.get("ok")),
        "chunks": int(stats.get("chunks") or 0),
        "tokens_mean": stats.get("tokens_mean"),
        "reasons": [str(r.get("message") or r.get("key")) for r in diag.get("reasons") or []],
    }


PREVIEW_CHUNK_LIMIT = 400


def chunk_preview(kb_id: str, file_id: str, max_tokens: int | None = None,
                  overlap_tokens: int | None = None) -> dict[str, Any]:
    """Re-chunk an already parsed file with the current (or the form's still unsaved) chunking parameters and
    return the diagnostics plus every chunk's content. Does not touch the vector store: MinerU results and
    VLM descriptions both come from the parse cache, so only files that have already been indexed can be
    previewed -- otherwise the document would really be sent to the GPU for parsing."""
    import dataclasses

    from kb_pipeline.chunking.chunker import blocks_to_chunks
    from kb_pipeline.chunking.diagnose import chunk_diagnostics
    from kb_pipeline.parsers.common import parser_profile_for_path
    from kb_pipeline.parsers.errors import NonRetryableParseError
    from kb_pipeline.pipeline.parse_job import _parse_blocks, parse_cache_dir, verify_source_file

    cfg = settings()
    source = next((s for s in cfg.sources.values() if s.kb_id == kb_id), None)
    if source is None:
        raise KeyError(kb_id)
    with db.connect(cfg.state_db) as con:
        row = db.get_file_by_id(con, file_id)
    if row is None or str(row["kb_id"]) != kb_id:
        raise KeyError(file_id)
    if str(row["status"]) == "deleted":
        raise ValueError("This file has been removed from the source")
    path = Path(str(row["physical_path"]))
    if not path.exists():
        raise ValueError(f"Source file is not on disk: {path}")
    try:
        # The same boundary check as the worker: the directory was not swapped for a link and the file's real
        # location is still inside the enrolled directory (the preview reads the source file as well)
        verify_source_file(cfg, source, path)
    except NonRetryableParseError as exc:
        raise ValueError(str(exc)) from exc

    if max_tokens is not None or overlap_tokens is not None:
        mt = int(max_tokens if max_tokens is not None else source.max_tokens)
        ov = int(overlap_tokens if overlap_tokens is not None else source.overlap_tokens)
        if mt < 16:
            raise ValueError("max_tokens must be at least 16")
        if ov < 0 or ov >= mt:
            raise ValueError("overlap_tokens must be between 0 and max_tokens")
        source = dataclasses.replace(source, max_tokens=mt, overlap_tokens=ov)

    content_version = str(row["content_version"])
    cache_dir = parse_cache_dir(cfg, kb_id, int(row["file_key"]), content_version)
    suffix = path.suffix.lower()
    cached = {
        ".pdf": cache_dir / "mineru" / "result.json",
        ".docx": cache_dir / "mineru" / "docx-result.json",
        ".pptx": cache_dir / "pptx_structure_input" / "structure-safe.json",
    }.get(suffix)
    if cached is not None and not cached.exists():
        raise ValueError("This file has not finished parsing; the preview needs the parse cache, try again once it is indexed")

    parser_profile = parser_profile_for_path(path)
    blocks = _parse_blocks(cfg, source, path, cache_dir, parser_profile, row)
    chunks = blocks_to_chunks(
        kb_id=kb_id,
        file_key=int(row["file_key"]),
        content_version=content_version,
        parser_profile=parser_profile,
        blocks=blocks,
        max_tokens=source.max_tokens,
        overlap_tokens=source.overlap_tokens,
    )
    diag = chunk_diagnostics(chunks, max_tokens=source.max_tokens, blocks=blocks)
    items = []
    for c in chunks[:PREVIEW_CHUNK_LIMIT]:
        b = c.block
        items.append({
            "chunk_index": c.chunk_index,
            "tokens": c.token_count,
            "block_id": b.block_id,
            "block_type": b.block_type,
            "page_idx": b.page_idx,
            "page_end": b.metadata.get("page_end", b.page_idx),
            "section_path": [str(x) for x in (b.metadata.get("section_path") or [])],
            "table_flags": b.metadata.get("table_flags") or None, "table_repair": b.metadata.get("table_repair") or None,
            "text": c.text,
        })
    return {
        "file": {"file_id": file_id, "rel_path": str(row["rel_path"]), "parser_profile": parser_profile},
        "max_tokens": source.max_tokens,
        "overlap_tokens": source.overlap_tokens,
        "diagnostics": diag,
        "chunks": items,
        "truncated": len(chunks) > PREVIEW_CHUNK_LIMIT,
    }


# ── enrollment ──────────────────────────────────────────────────────────

def enroll(dir_name: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Enroll a directory, optionally with the config the user drafted in the
    console. The draft is validated BEFORE enrollment (a bad draft leaves
    nothing enrolled) and written before the scan kick, so the very first
    parse already runs with the user's strategy."""
    cfg = settings()
    db.init_db(cfg.state_db)
    config = strip_retired_config(config) if config else config
    with db.connect(cfg.state_db) as con:
        if config:
            limits = chunk_limits(cfg.embedding_base_url)
            effective = dict(discovery.DEFAULTS)
            effective.update(discovery.DIR_DEFAULTS.get(dir_name, {}))
            effective.update({k: v for k, v in config.items() if v is not None})
            errors = _validate_config(con, effective, normalize_graph_updates(config), limits)
            if errors:
                raise ValueError("; ".join(errors))
        source, outcome = discovery.enroll(con, cfg.mirror_root, dir_name)
        if config:
            # Write the draft when re-enabling a deactivated knowledge base too: the console's semantics are
            # "save = persist". It used to be written only on first enrollment, so a user editing the config
            # of a deactivated KB and clicking save saw a success message, yet a reload brought the old
            # values back. Saving only updates the strategy; it has no side effect on indexed content.
            discovery.set_config(con, source.kb_id, config)
    # External resources are created outside the transaction: by the time one fails the knowledge base is
    # already active. Rather than raising a bare 500 (the frontend would show only Internal Server Error
    # while polling shows it enabled), return normally with warnings -- the main collection is lazily
    # created by the first parse job, so it self-heals.
    warnings: list[str] = []
    created: Any = None
    q = qdrant_client(cfg.qdrant_url, cfg.qdrant_api_key)
    try:
        created = ensure_collection(q, source.collection, cfg.vector_layout)
    except Exception as exc:
        warnings.append(f"Vector collection creation failed ({exc.__class__.__name__}); it is recreated automatically at the first parse")
    try:
        search_fts.ensure_indices(cfg.opensearch_url, [source.collection])
    except Exception as exc:
        warnings.append(f"Keyword index creation failed ({exc.__class__.__name__}); it is retried when parse results are written")
    kick_scan()
    kick_worker()
    return {"kb_id": source.kb_id, "collection": source.collection, "outcome": outcome,
            "qdrant": created, "warnings": warnings}


def delete_kb(kb_id: str) -> dict[str, Any]:
    """Permanent deletion: immediately clears the vector store, the graph stores (Qdrant graph collection +
    Neo4j projection), the graph build cache, the keyword index, the parse cache and all state; it goes
    through the same _hard_delete_kb as the retention-expiry GC, so the cleanup scope is strictly
    identical."""
    from kb_pipeline.maintenance import PartialDeleteError, delete_kb_now

    try:
        return delete_kb_now(settings(), kb_id=kb_id)
    except PartialDeleteError as exc:
        # Some storage layers were not deleted: the registry row is kept as delete_failed for the next GC
        # round to retry; report that truthfully to the console instead of letting it show "permanently
        # deleted".
        raise ValueError(
            "Deletion incomplete, these parts failed (the knowledge base is kept, retried at the next maintenance run): "
            + "; ".join(exc.errors)
        )


def adopt_kb(kb_id: str, dir_name: str) -> dict[str, Any]:
    """Console action "this directory is knowledge base X renamed": adopt the id and all data, only pointing
    the knowledge base at the new directory; then kick one scan so files with unchanged content refresh
    their path through metadata_update instead of being re-parsed. The similarity is returned as well, so a
    wrong match is visible."""
    cfg = settings()
    name = str(dir_name or "").strip()
    if not name:
        raise ValueError("Missing directory name")
    with db.connect(cfg.state_db) as con:
        report = discovery.directory_match_report(con, cfg.mirror_root, kb_id, name)
        source = discovery.adopt_directory(con, cfg.mirror_root, kb_id, name)
        con.commit()
    print(f"[web] kb adopted: {discovery.kb_label(kb_id, name)} matched {report['matched']}/{report['total']}", flush=True)
    kick_scan()
    kick_worker()
    return {"kb_id": source.kb_id, "dir": name, "matched": report["matched"], "total": report["total"],
            "verified": report["verified"], "looks_like_rename": discovery.looks_like_rename(report)}


def unenroll_info(kb_id: str) -> dict[str, Any]:
    cfg = settings()
    with db.connect(cfg.state_db) as con:
        row = con.execute("SELECT * FROM kb_sources WHERE kb_id=?", (kb_id,)).fetchone()
        if row is None:
            raise KeyError(kb_id)
        files = con.execute(
            "SELECT COUNT(*) FROM files WHERE kb_id=? AND status != 'deleted'", (kb_id,)
        ).fetchone()[0]
    return {
        "kb_id": kb_id,
        "dir": str(row["source_root"]),
        "files": int(files),
        "retention_days": cfg.qdrant_inactive_retention_days,
    }


def unenroll(kb_id: str) -> dict[str, Any]:
    cfg = settings()
    with db.connect(cfg.state_db) as con:
        row = con.execute("SELECT * FROM kb_sources WHERE kb_id=?", (kb_id,)).fetchone()
        if row is None:
            raise KeyError(kb_id)
        if str(row["status"]) != "active":
            # Idempotent: calling this again on an already closed knowledge base used to schedule another
            # round of delete jobs and create another ingest_run (the points are already inactive, pure
            # busywork)
            return {"kb_id": kb_id, "files_delete_queued": 0, "delete_jobs": 0,
                    "already_inactive": True,
                    "retention_days": cfg.qdrant_inactive_retention_days}
        discovery.mark_inactive(con, kb_id, reason="unenrolled")
        run_id = db.begin_run(con, "web-unenroll", note=f"unenroll {row['source_root']}")
        con.commit()
    # Closing = stopping work, not deleting data: queued jobs are voided, and the running one exits on its
    # own at the next phase boundary once it sees the cancel flag. Indexed content merely enters the
    # deactivated retention period and revives as-is on re-enabling; files that were not fully parsed are
    # re-queued by the next scan -- i.e. parsing continues from where it stopped.
    from kb_pipeline.maintenance import stop_graph_build_now, stop_kb_parse_jobs

    stopped = stop_kb_parse_jobs(cfg, kb_id=kb_id, reason="cancelled because the knowledge base was closed")
    # Closing the knowledge base also stops its graph build (final review F02): cooperative SIGTERM, cache
    # and artifacts kept for the retention period, resumed after re-enabling; a build that cannot be stopped
    # is only recorded in the result and does not block the close itself (the build re-checks that the KB
    # is still enabled before publishing)
    try:
        graph_stopped = stop_graph_build_now(cfg, kb_id=kb_id, reason="cancelled because the knowledge base was closed")
    except Exception as exc:
        graph_stopped = {"stopped": False, "error": repr(exc)}
    with db.connect(cfg.state_db) as con:
        deleted, jobs = schedule_deletes_for_source(
            con,
            ingest_run_id=run_id,
            kb_id=kb_id,
            collection=str(row["collection"]),
            seen_file_keys=set(),
        )
        db.finish_run(con, run_id, added_count=0, updated_count=0, moved_count=0,
                      deleted_count=deleted, failed_count=0)
    kick_worker()
    return {"kb_id": kb_id, "files_delete_queued": deleted, "delete_jobs": jobs,
            "stopped_jobs": stopped, "stopped_graph": graph_stopped,
            "retention_days": cfg.qdrant_inactive_retention_days}


# ── per-KB config ───────────────────────────────────────────────────────

def _dir_defaults(con, kb_id: str) -> dict[str, Any]:
    """The directory-name-keyed presets this KB inherits at runtime
    (build_source layers them between DEFAULTS and config_json)."""
    row = con.execute("SELECT source_root FROM kb_sources WHERE kb_id=?", (kb_id,)).fetchone()
    if row is None:
        return {}
    return discovery.DIR_DEFAULTS.get(str(row["source_root"]), {})


def _validate_rebuild_policy(effective: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        parse_interval_days(effective.get("graph_rebuild_interval"))
    except ValueError:
        errors.append("Invalid auto-rebuild interval: use 7d / 2w / 1m or a plain number of days; empty means no time condition")
    try:
        parse_ratio(effective.get("graph_rebuild_new_chunk_pct"))
    except ValueError:
        errors.append("Invalid auto-rebuild new-content percentage: use 20% or 0.2; empty means no delta condition")
    try:
        parse_new_chunk_count(effective.get("graph_rebuild_new_chunk_count"))
    except ValueError:
        errors.append("Invalid auto-rebuild new chunk count: it must be an integer of at least 1; empty means no delta condition")
    # The two measures are two expressions of the same condition; with both in effect the operator's meaning
    # becomes unclear (under "or", whichever is met first? under "and", both required?). The console
    # dropdown is already an either/or choice; this blocks direct API calls that bypass the UI.
    if (effective.get("graph_rebuild_new_chunk_pct")
            and effective.get("graph_rebuild_new_chunk_count")):
        errors.append("The auto-rebuild new-content condition must be either a percentage or a chunk count, not both")
    op = str(effective.get("graph_rebuild_operator") or "or").strip().lower()
    if op not in {"or", "and"}:
        errors.append("The auto-rebuild condition operator must be or / and")
    return errors


def normalize_graph_updates(updates: dict[str, Any]) -> dict[str, Any]:
    """Normalize the graph fields about to be written into config_json (in place).

    Only one rule so far: a new-content percentage above 100% is clamped to 100%. "Rebuild once it grew by
    150%" and "rebuild once it doubled" are exactly equivalent in the check (the denominator is the number
    of still-alive baseline chunks); keeping the larger number only makes the threshold look stricter than it
    is. Clamping here rather than only on the read side means the console shows 100% after a reload --
    otherwise the UI would say 150 while the behaviour follows 100, and the two would not match.
    """
    raw = updates.get("graph_rebuild_new_chunk_pct")
    if raw in (None, ""):
        return updates
    try:
        ratio = parse_ratio(raw)
    except ValueError:
        return updates          # an invalid format is left for _validate_config to report, not our job here
    if ratio is not None and ratio > 1:
        updates["graph_rebuild_new_chunk_pct"] = "100%"
    return updates


def _validate_config(con, effective: dict[str, Any], updates: dict[str, Any], limits: dict[str, Any]) -> list[str]:
    """Shared by update_kb_config and enroll-with-config: chunk sizes against
    live limits, rebuild policy formats, graph_llm references."""
    errors = validate_chunk_config(effective.get("max_tokens"), effective.get("overlap_tokens"), limits)
    errors.extend(validate_graph_unit_config(
        effective.get("graph_unit_chunks"), effective.get("graph_max_gleanings")))
    errors.extend(validate_graph_schema_config(
        effective.get("graph_predicates"), effective.get("graph_parent_types")))
    errors.extend(_validate_rebuild_policy(effective))
    errors.extend(validate_graph_tune_config(
        effective.get("graph_entity_types"), effective.get("graph_tune_sample_size")))
    graph_llm = updates.get("graph_llm")
    if isinstance(graph_llm, dict):
        for step, name in graph_llm.items():
            if step not in GRAPH_LLM_STEPS and step != GRAPH_TUNE_STEP:
                errors.append(f"Unknown graph step: {step}")
            elif name and db.get_llm(con, str(name)) is None:
                errors.append(f"Graph step {step} references model {name!r}, which is not in the registry")
    return errors


def schema_versions_view(effective: dict[str, Any]) -> dict[str, Any]:
    """Label version view for the console: the version list plus the currently selected id.

    Version management was added later. Labels extracted before it exist only as graph_entity_types, with
    no record of "when, with which model, at what sample size" -- such knowledge bases get a fallback entry
    whose id is CURRENT_SCHEMA_VERSION_ID, with empty metadata honestly shown as unknown. It exists only in
    this view and is never persisted; on save it is treated as "keep things as they are" rather than a real
    selection.
    """
    versions = [dict(v) for v in (effective.get("graph_schema_versions") or [])]
    active = str(effective.get("graph_schema_active") or "")
    known = {str(v.get("id")) for v in versions}
    types = list(effective.get("graph_entity_types") or [])
    if active not in known:
        active = ""
    if not active and types:
        versions.insert(0, {
            "id": CURRENT_SCHEMA_VERSION_ID,
            "created_at": None, "model": None, "sample_size": None,
            "language": effective.get("graph_language"),
            "entity_types": types,
            "legacy": True,
        })
        active = CURRENT_SCHEMA_VERSION_ID
    return {"versions": versions, "active": active}


def _ring_with_legacy(stored: dict[str, Any]) -> list[dict[str, Any]]:
    """See kb_pipeline.graph.schema_flow.ring_with_legacy (moved there 2026-09-08: the pipeline itself also
    has to add versions to the ring)."""
    from kb_pipeline.graph.schema_flow import ring_with_legacy

    return ring_with_legacy(stored)


def _apply_schema_version(current: dict[str, Any], updates: dict[str, Any]) -> None:
    """See kb_pipeline.limits.apply_schema_version (moved there 2026-09-08: the pipeline itself also has to
    put a version into effect)."""
    from kb_pipeline.limits import apply_schema_version

    apply_schema_version(current, updates)


def get_kb_config(kb_id: str) -> dict[str, Any]:
    cfg = settings()
    with db.connect(cfg.state_db) as con:
        config = discovery.get_config(con, kb_id)
        merged = dict(discovery.DEFAULTS)
        merged.update(_dir_defaults(con, kb_id))
        merged.update(config)
    return {
        "config": config,
        "effective": merged,
        "schema": schema_versions_view(merged),
        "limits": chunk_limits(cfg.embedding_base_url),
    }


def delete_graph_schema_version(kb_id: str, version_id: str) -> dict[str, Any]:
    """Delete one entry from the version ring.

    This goes through the server instead of letting the console write graph_schema_versions directly, for
    the same reason extraction adds to the ring server-side: if clients could write it, "when, with which
    model and at what sample size this version was extracted" becomes something anyone can make up, while
    the entire value of rolling back to a version rests on that record being trustworthy.

    The only thing blocked is **the version currently in effect**. The effective graph_entity_types /
    graph_language were copied from it and the built graph was built with it; deleting it would make the
    dropdown silently fall back to the newest version, so the UI would show A's labels while the build used
    B's -- and since both are just lists of words, nobody would notice the mismatch. push_schema_version
    skips it too when rotating out old versions: one rule, two exits.
    """
    cfg = settings()
    wanted = str(version_id or "").strip()
    if not wanted:
        raise ValueError("Missing label version id")
    if wanted == CURRENT_SCHEMA_VERSION_ID:
        # This is the read-only entry schema_versions_view makes up on the fly; it is never persisted, so
        # there is nothing to delete.
        raise ValueError("This entry is only a read-only view of the labels in effect, not a saved version")
    with db.connect(cfg.state_db) as con:
        row = con.execute("SELECT kb_id FROM kb_sources WHERE kb_id=?", (kb_id,)).fetchone()
        if row is None:
            raise KeyError(kb_id)
        current = discovery.get_config(con, kb_id)
        versions = [dict(v) for v in (current.get("graph_schema_versions") or [])]
        if not any(str(v.get("id")) == wanted for v in versions):
            raise ValueError("The label version does not exist or has been rotated out")
        if wanted == str(current.get("graph_schema_active") or ""):
            raise ValueError("The version in effect cannot be deleted: the graph was built with it and it could never be selected again")
        kept = [v for v in versions if str(v.get("id")) != wanted]
        # An empty list has to be written as None: set_config uses None to mean "remove this key", whereas []
        # would leave an empty array in config_json, which is not the same state as "never extracted".
        discovery.set_config(con, kb_id, {"graph_schema_versions": kept or None})
        con.commit()
        merged = dict(discovery.DEFAULTS)
        merged.update(_dir_defaults(con, kb_id))
        merged.update(discovery.get_config(con, kb_id))
    return {"deleted": wanted, "schema": schema_versions_view(merged)}


# Config keys of the old pipeline. Forms from before the console redesign still send
# them; the server drops them silently instead of raising, and old values already in config_json are
# ignored by build_source.
RETIRED_CONFIG_KEYS = {
    "graph_chunk_size", "graph_chunk_overlap", "graph_encoding_model",
    "graph_max_cluster_size", "graph_use_lcc", "graph_unit_tokens",
    "graph_mode",          # 2026-09-05: summary-tree (raptor) mode removed, only the entity graph remains
}
RETIRED_LLM_STEPS = {"community"}


def strip_retired_config(updates: dict[str, Any]) -> dict[str, Any]:
    cleaned = {k: v for k, v in (updates or {}).items() if k not in RETIRED_CONFIG_KEYS}
    llm = cleaned.get("graph_llm")
    if isinstance(llm, dict):
        cleaned["graph_llm"] = {k: v for k, v in llm.items() if k not in RETIRED_LLM_STEPS}
    return cleaned


def update_kb_config(kb_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    cfg = settings()
    limits = chunk_limits(cfg.embedding_base_url)
    updates = strip_retired_config(updates)
    with db.connect(cfg.state_db) as con:
        current = discovery.get_config(con, kb_id)
        normalize_graph_updates(updates)
        # "Select a label version and save" = copy that version's labels / language into the effective
        # values. The effective values are still just graph_entity_types / graph_language, so the build path
        # needs no change at all.
        _apply_schema_version(current, updates)
        # Mirror the post-save runtime view: a None update clears the key, so
        # validation and warnings run against what build_source will see.
        merged_cfg = dict(current)
        for key, value in updates.items():
            if value is None:
                merged_cfg.pop(key, None)
            else:
                merged_cfg[key] = value
        effective = dict(discovery.DEFAULTS)
        effective.update(_dir_defaults(con, kb_id))
        effective.update(merged_cfg)
        errors = _validate_config(con, effective, updates, limits)
        if errors:
            raise ValueError("; ".join(errors))
        # Changing the graph switch / models during a build causes a state mismatch: the build finishes as
        # usual and leaves artifacts, but graph_status has already become disabled, "Delete knowledge graph"
        # greys out with it, and the artifacts have no entry point left for cleanup. Parse-related fields are
        # unaffected and can be changed as usual.
        graph_keys = {"graph_enabled", "graph_llm"}
        stop_build = False
        if graph_keys & set(updates):
            db.reconcile_stale_graph_builds(con)
            latest = con.execute(
                "SELECT status FROM graph_builds WHERE kb_id=? ORDER BY started_at DESC LIMIT 1", (kb_id,)
            ).fetchone()
            if latest is not None and str(latest["status"]) == "running":
                # Turning the switch off = an explicit request to stop work, so allow it and terminate the
                # build along the way; changing a model is merely a config change, and switching mid-way would
                # build the two halves with different models, so that is still refused.
                if updates.get("graph_enabled") is False:
                    stop_build = True
                else:
                    raise ValueError("A graph build is running; step models cannot be changed now. Wait for it to finish or turn the knowledge graph off first")
        # "Turn on knowledge graph" is an explicit intent that overrides the earlier pause. Turning it off
        # needs no clearing: while off, evaluate_rebuild already returns at the graph_enabled gate.
        if updates.get("graph_enabled") is True and current.get("graph_paused"):
            updates = dict(updates)
            updates["graph_paused"] = None
        saved = discovery.set_config(con, kb_id, updates)
    if stop_build:
        # Only the process is stopped; artifacts and the LLM cache are all kept: a build after re-enabling
        # hits the cache and resumes.
        from kb_pipeline.maintenance import stop_graph_build_now

        stop_graph_build_now(cfg, kb_id=kb_id, reason="Build stopped by “Turn off knowledge graph”")
    # This used to generate "no model selected / no rebuild condition" reminders; removed at the user's
    # request -- a missing model is caught by the "Build now / again" validation and the failed build
    # status, no more nagging.
    return {"config": saved, "warnings": []}


def parse_now(kb_id: str) -> dict[str, Any]:
    """The "Parse now" action: an empty knowledge base = initial full load, existing content = process the
    incremental files; in essence it kicks one scan + worker round right away, exactly the same path as the
    automatic scan every minute."""
    cfg = settings()
    with db.connect(cfg.state_db) as con:
        row = con.execute("SELECT status FROM kb_sources WHERE kb_id=?", (kb_id,)).fetchone()
        if row is None:
            raise KeyError(kb_id)
        if str(row["status"]) != "active":
            raise ValueError("The knowledge base is not enabled")
    scan_kick, worker_kick = kick_scan(), kick_worker()
    if "unavailable" in (scan_kick, worker_kick):
        # When both systemd and the script fallback failed, this used to return kicked=true anyway and the
        # user believed parsing had started
        raise RuntimeError(f"Could not trigger the scan / worker (scan={scan_kick}, worker={worker_kick}); check the background services")
    return {"kicked": True, "scan": scan_kick, "worker": worker_kick}


def delete_graph(kb_id: str) -> dict[str, Any]:
    """The "Delete knowledge graph" action: deletes only the graph data, leaves the knowledge base alone; the
    switch goes back to grey afterwards."""
    from kb_pipeline.maintenance import delete_graph_now

    result = delete_graph_now(settings(), kb_id=kb_id)
    if result.get("errors"):
        raise ValueError(
            "Graph deletion incomplete, these parts failed (the switch stays on so it can be retried): "
            + "; ".join(str(e) for e in result["errors"])
        )
    return result


def _spawn_graph_build(cfg: Settings, kb_id: str, *, graph_version: str | None = None,
                       append: bool = False) -> dict[str, Any]:
    base_dir = cfg.state_db.parent.parent.parent
    py = base_dir / "app" / ".venv" / "bin" / "python"
    env_file = os.environ.get("KB_ENV_FILE") or str(base_dir / "config" / "knowledge-base.env")
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = cfg.log_dir / f"web-graph-build-{kb_id}.log"
    verb = "append" if append else "build"
    build_cmd = [str(py), "-m", "kb_pipeline", "--env-file", env_file, "graph", verb, "--source", kb_id]
    if graph_version and not append:
        # Resume last time's version: phases already completed are skipped per graph_build_phases
        build_cmd += ["--graph-version", graph_version, "--allow-existing-graph-version"]
    # A graph build easily runs for hours. A child spawned by a plain Popen stays in carrel-web's cgroup,
    # and with systemd's default KillMode=control-group every web restart (deploy, config change) would
    # take the running build down with it. So first try systemd-run to start it in a separate transient
    # scope detached from web's lifecycle; fall back to Popen when systemd-run is unavailable.
    # Fallback wall-clock cap. The circuit breaker covers "the provider is down as a whole", not "everything
    # works but it is just slow"; without this line a console-started build has RuntimeMaxUSec=infinity and
    # a genuinely stuck one never ends. The value must exceed the slowest normal build (kb_004 measured at
    # about 23 hours); default 48 hours. On timeout systemd sends SIGTERM, which takes the existing signal
    # wind-down: write cancelled, keep the cache.
    max_hours = os.environ.get("KB_GRAPH_BUILD_MAX_HOURS", "48")
    scope = ["systemd-run", "--user", "--scope", "--collect",
             f"--unit=kb-graph-build-{kb_id}-{int(time.time())}", "--quiet",
             f"--property=RuntimeMaxSec={int(float(max_hours) * 3600)}"]
    with log_path.open("ab") as log:
        try:
            proc = subprocess.Popen(
                scope + build_cmd,
                cwd=str(base_dir), stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                env=_child_env(env_file, base_dir),
            )
            detached = "systemd-scope"
        except (FileNotFoundError, OSError):
            proc = subprocess.Popen(
                build_cmd,
                cwd=str(base_dir), stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                env=_child_env(env_file, base_dir),
            )
            detached = "popen"
    return {"started": True, "pid": proc.pid, "log": str(log_path), "detached": detached,
            "resumed": graph_version, "append": append}


def pause_graph_build(kb_id: str) -> dict[str, Any]:
    """The "Pause build" action: terminate the process, touch no artifacts, and leave graph_enabled alone.

    Shares the mechanism of "Turn off knowledge graph" (stop_graph_build_now); the only difference is the
    switch: turning off means "no more automatic rebuilds from now on", pausing means "stop for now, carry
    on in a while".

    Why stopping and starting over is still worthwhile: extraction results are stored per unit in
    graph_extractions and LLM responses are cached by call content in the per-KB SQLite (neither keyed by
    version), so on the next build every unit and call that already completed comes back instantly. The
    condition is not to change models, labels, predicates, output language or unit parameters while paused:
    all of those go into the fingerprint, one change voids the whole cache, and the resume becomes a re-run.
    """
    from kb_pipeline.maintenance import stop_graph_build_now

    cfg = settings()
    with db.connect(cfg.state_db) as con:
        db.reconcile_stale_graph_builds(con)
        row = con.execute("SELECT * FROM kb_sources WHERE kb_id=?", (kb_id,)).fetchone()
        if row is None:
            raise KeyError(kb_id)
        latest = con.execute(
            "SELECT status FROM graph_builds WHERE kb_id=? AND status != 'rolled_back' ORDER BY started_at DESC LIMIT 1",
            (kb_id,),
        ).fetchone()
    if latest is None or str(latest["status"]) != "running":
        raise ValueError("No graph build is running")
    result = stop_graph_build_now(cfg, kb_id=kb_id, reason="Build stopped by “Pause build”")
    if not result.get("stopped"):
        raise ValueError("The graph build process could not be terminated; try again later")
    # Stopping the process is not enough: a pause leaves cancelled rather than a successful build, and
    # evaluate_rebuild only accepts successful builds as the baseline -- so for a KB that never built
    # successfully, due stays true after a pause and the 00:00 run that night resumes it. Persist the intent
    # so the auto-rebuild side honours it first.
    with db.connect(cfg.state_db) as con:
        discovery.set_config(con, kb_id, {"graph_paused": True})
    result["cache_entries"] = graph_cache_entries(cfg, str(row["collection"]))
    return result


def graph_cache_entries(cfg, collection: str) -> int:
    """Number of cached LLM responses this knowledge base has accumulated -- a direct measure of how much a
    resume gets for free. Only computed once the build has stopped (see _graph_build_info)."""
    from kb_pipeline.graph.build import graph_cache_entries as _entries

    return _entries(cfg, collection)


def _graph_action_gate(con, cfg: Settings, kb_id: str):
    """Preconditions shared by "Build now / again" and "Append new content": the KB is enabled, the directory
    exists, this KB is not being parsed, the graph is on, the models are all set and no build is running.
    Returns (kb_sources row, config, most recent build record)."""
    db.reconcile_stale_graph_builds(con)   # a dead record must not block rebuilding forever
    row = con.execute("SELECT * FROM kb_sources WHERE kb_id=?", (kb_id,)).fetchone()
    if row is None:
        raise KeyError(kb_id)
    if str(row["status"]) != "active":
        raise ValueError("The knowledge base is not enabled, so no graph can be built")
    if not (cfg.mirror_root / str(row["source_root"])).is_dir():
        # When the directory vanished but the scan has not yet marked the KB deactivated, the build child
        # exits with "unknown source", and that error only shows up in the background log; the console sees
        # nothing.
        raise ValueError("The knowledge base directory does not exist right now, so no graph can be built")
    if db.kb_parse_busy(con, kb_id):
        raise ValueError("This knowledge base is being parsed; graph operations are unavailable until it finishes.")
    config = discovery.get_config(con, kb_id)
    effective = dict(discovery.DEFAULTS)
    effective.update(_dir_defaults(con, kb_id))
    effective.update(config)
    if not effective.get("graph_enabled"):
        raise ValueError("The knowledge graph is not turned on")
    from kb_pipeline.graph.build import default_graph_llm

    chosen = config.get("graph_llm") or {}
    missing = [s for s in GRAPH_LLM_STEPS if not str(chosen.get(s) or "").strip()]
    if missing and default_graph_llm(con) is None:
        # Unselected slots follow default_graph_llm (the default name, or the only model in the registry);
        # only block when even that fallback is missing
        raise ValueError(f"No model selected for graph steps: {', '.join(missing)}")
    latest = con.execute(
        "SELECT * FROM graph_builds WHERE kb_id=? ORDER BY started_at DESC LIMIT 1", (kb_id,)
    ).fetchone()
    if latest is not None and str(latest["status"]) == "running":
        raise ValueError("A graph build is already running")
    # The build lock is global: while another KB is building, the child fails to take the lock before writing
    # its build record and exits quietly, yet the console has already answered "started"
    # (Codex 2026-09-13 F04). Check the lock here first and say so plainly when it is held.
    try:
        lock_dir = build_lock_path(cfg)
    except Exception:
        lock_dir = None
    if lock_dir is not None and build_lock_held(lock_dir):
        raise ValueError("Another knowledge base is building its graph (only one build runs at a time); try again when it finishes")
    return row, config, latest


APPEND_SKIP = {
    "source_unchanged": "No documents were added, changed or removed relative to the current graph; nothing to append",
    "no_successful_build": "No graph has been built yet; run a full build first",
    "config_changed_needs_full_rebuild": "Model / labels / predicates / unit settings changed; the previous extraction and merge cannot be reused, run a full rebuild",
    "llm_not_configured": "No model selected for the graph build steps",
    "paused_by_operator": "The build is paused; use “Resume build” first",
    "no_active_content": "This knowledge base has no active chunks yet; finish parsing first",
}


def trigger_graph_append(kb_id: str) -> dict[str, Any]:
    """The "Append new content" action: same preconditions as "Build now / again", plus a completed graph,
    an unchanged config and document-level corpus changes relative to the current version; when they hold,
    an incremental append (graph append) runs as a background process."""
    from kb_pipeline.graph.build import evaluate_append

    cfg = settings()
    with db.connect(cfg.state_db) as con:
        row, config, _latest = _graph_action_gate(con, cfg, kb_id)
    source = discovery.build_source(
        cfg.mirror_root, str(row["source_root"]), config, kb_id=kb_id, collection=str(row["collection"]))
    decision = evaluate_append(cfg, source_key=kb_id, source=source, ignore_auto_flag=True)
    if not decision.get("due"):
        reason = str(decision.get("reason") or "")
        raise ValueError(APPEND_SKIP.get(reason, f"Cannot append right now ({reason})"))
    spawned = _spawn_graph_build(cfg, kb_id, append=True)
    return {**spawned, "base_version": decision.get("base_version"), "delta": decision.get("delta")}


def trigger_graph_build(kb_id: str) -> dict[str, Any]:
    """The "Save and build now" action: after validation, start one graph build for this knowledge base as a
    background process; progress checks in through graph_builds.stage and is visible live on the
    "Graph build status" panel."""
    cfg = settings()
    with db.connect(cfg.state_db) as con:
        _row, _config, latest = _graph_action_gate(con, cfg, kb_id)
        # Last build stopped midway (paused / failed), config and corpus unchanged, and some phases really
        # completed: resume the same version and skip the completed phases. Otherwise start a new version
        # from scratch.
        resume_version = None
        if latest is not None and str(latest["status"]) in ("cancelled", "failed"):
            if (_paused_cache_reuse(con, kb_id, latest)["cache_reusable"]
                    and db.graph_phases_done(con, str(latest["graph_build_id"]))):
                resume_version = str(latest["graph_version"])
        # "Resume build / Rebuild now" is an explicit intent to resume, so clear the pause mark along the way
        # -- otherwise once this run finishes, the auto-rebuild side still treats the KB as paused.
        if discovery.get_config(con, kb_id).get("graph_paused"):
            discovery.set_config(con, kb_id, {"graph_paused": None})
    return _spawn_graph_build(cfg, kb_id, graph_version=resume_version)


# Wall-clock cap for the four LLM calls (domain / language / persona / type table). This is a synchronous
# endpoint -- the console needs the result to fill the form, so it is better to let a slow model hit the cap
# and get a "switch to a faster model" hint than to leave the browser waiting forever. Measured: MiniMax M3
# finishes a full prompt-tune (6+N calls) in 50 s, and there are only 4 calls here.
_SUGGEST_TIMEOUT_SECONDS = 420


def graph_corpus_stats(kb_id: str) -> dict[str, Any]:
    """Size of this knowledge base's graph corpus: active documents, active chunks, and the estimated number of
    units at the current unit size. The hint under "Sample size" in the console needs it -- only when the
    total chunk count is known does sampling N chunks read as a proportion. All from the state database,
    Qdrant is not touched."""
    cfg = settings()
    with db.connect(cfg.state_db) as con:
        row = con.execute("SELECT * FROM kb_sources WHERE kb_id=?", (kb_id,)).fetchone()
        if row is None:
            raise KeyError(kb_id)
        config = discovery.get_config(con, kb_id)
        stats = con.execute(
            """
            SELECT COUNT(*) AS chunks, COUNT(DISTINCT c.file_id) AS documents
            FROM chunks c WHERE c.collection = ? AND c.status = 'active'
            """,
            (str(row["collection"]),),
        ).fetchone()
    source = discovery.build_source(
        cfg.mirror_root, str(row["source_root"]), config,
        kb_id=kb_id, collection=str(row["collection"]),
    )
    chunks = int(stats["chunks"] or 0)
    documents = int(stats["documents"] or 0)
    per_unit = max(1, int(source.graph_unit_chunks))
    return {
        "kb_id": kb_id, "documents": documents, "chunks": chunks,
        "units_estimate": (chunks + per_unit - 1) // per_unit if chunks else 0,
        "unit_chunks": int(source.graph_unit_chunks),
    }


# Wall-clock cap for the seven LLM calls (domain / language / persona / type table / parent types /
# predicates / competency questions). This is a synchronous endpoint -- the console needs the result to
# fill the form, so it is better to let a slow model hit the cap and get a "switch to a faster model" hint
# than to leave the browser waiting forever. A flash-class model finishes the whole set in about 1-2 minutes.
_SUGGEST_TIMEOUT_SECONDS = 600


def suggest_graph_schema(kb_id: str) -> dict[str, Any]:
    """The "Extract labels now / again" action: sample chunks from this knowledge base and have the LLM infer
    the domain, output language, entity labels, parent types, relation predicates, type definitions, scene
    profile and corpus examples (plan 4.10).

    Only returns the suggestion and adds it to the version ring; it does **not** take effect automatically --
    the result is filled back into the console form for you to review and edit, and takes effect only when
    "Save config" is clicked. When a current version exists, the revision is based on it and the endpoint
    ledger (schema_flow.suggest_schema_version).
    """
    from kb_pipeline.graph import build as graph_build
    from kb_pipeline.graph.schema_flow import ORIGIN_MANUAL, SUGGEST_BUSY_MESSAGE, suggest_in_progress, suggest_schema_version

    cfg = settings()
    with db.connect(cfg.state_db) as con:
        row = con.execute("SELECT * FROM kb_sources WHERE kb_id=?", (kb_id,)).fetchone()
        if row is None:
            raise KeyError(kb_id)
        if str(row["status"]) != "active":
            raise ValueError("The knowledge base is not enabled")
        config = discovery.get_config(con, kb_id)
        effective = dict(discovery.DEFAULTS)
        effective.update(_dir_defaults(con, kb_id))
        effective.update(config)
        if not effective.get("graph_enabled"):
            raise ValueError("The knowledge graph is not turned on")
        # While parsing is in progress chunks are still arriving; sampling now sees half a corpus and the
        # inferred type table would be skewed. Same gate as the graph build.
        if db.kb_parse_busy(con, kb_id):
            raise ValueError("This knowledge base is being parsed; label extraction is unavailable until it finishes.")
    # The automatic pre-build extraction (or another window) is extracting right now: another run would only
    # duplicate it, and the version it produces takes effect directly once it finishes
    if suggest_in_progress(cfg, kb_id) is not None:
        raise ValueError(SUGGEST_BUSY_MESSAGE)
    with db.connect(cfg.state_db) as con:
        ledger = db.active_chunk_refs(con, str(row["collection"]))
    if not ledger:
        raise ValueError("This knowledge base has no active chunks yet; finish parsing before extracting labels")

    source = discovery.build_source(
        cfg.mirror_root, str(row["source_root"]), config,
        kb_id=kb_id, collection=str(row["collection"]),
    )
    try:
        spec = graph_build.tune_llm_spec(cfg, source)
    except RuntimeError as exc:
        # "No model selected" is a config problem the user can fix, not a service fault: _wrap only turns
        # ValueError into a 422 with the original text for the console, while a RuntimeError becomes an
        # uninformative 500.
        raise ValueError(str(exc)) from exc
    started = time.time()
    try:
        result = suggest_schema_version(cfg, source, origin=ORIGIN_MANUAL, adopt=False, use_prior=True, spec=spec)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Extraction failed ({exc.__class__.__name__}): {str(exc)[:300]}") from exc
    if time.time() - started > _SUGGEST_TIMEOUT_SECONDS:
        print(f"[web] suggest_graph_schema took {time.time() - started:.0f}s for {kb_id}", flush=True)
    with db.connect(cfg.state_db) as con:
        merged = dict(discovery.DEFAULTS)
        merged.update(_dir_defaults(con, kb_id))
        merged.update(discovery.get_config(con, kb_id))
    result["schema"] = schema_versions_view(merged)
    # active in the view is still the old version; tell the frontend to point the dropdown at the version
    # just extracted
    result["schema"]["active"] = result["version_id"]
    return result


# An extraction preview runs a single unit: 1 + gleaning rounds calls, a dozen or so seconds on a
# flash-class model. This is a synchronous endpoint; the cap is set to the single-call timeout so the
# browser never waits forever.
_PREVIEW_TIMEOUT_SECONDS = 300


def _graph_source_for_units(kb_id: str):
    """Preconditions shared by the extraction preview and the unit list: the KB is enabled, the graph is on
    and there are active chunks. Returns (cfg, source, config, ledger)."""
    cfg = settings()
    with db.connect(cfg.state_db) as con:
        row = con.execute("SELECT * FROM kb_sources WHERE kb_id=?", (kb_id,)).fetchone()
        if row is None:
            raise KeyError(kb_id)
        if str(row["status"]) != "active":
            raise ValueError("The knowledge base is not enabled")
        config = discovery.get_config(con, kb_id)
        effective = dict(discovery.DEFAULTS)
        effective.update(_dir_defaults(con, kb_id))
        effective.update(config)
        if not effective.get("graph_enabled"):
            raise ValueError("The knowledge graph is not turned on")
        ledger = db.active_chunk_refs(con, str(row["collection"]))
    if not ledger:
        raise ValueError("This knowledge base has no active chunks yet; finish parsing first")
    source = discovery.build_source(
        cfg.mirror_root, str(row["source_root"]), config,
        kb_id=kb_id, collection=str(row["collection"]),
    )
    return cfg, source, config, ledger


def _build_graph_units(cfg, source, ledger, q=None):
    from kb_pipeline.graph.units import build_units, fetch_chunk_payloads

    q = q or qdrant_client(cfg.qdrant_url, cfg.qdrant_api_key)
    payloads = fetch_chunk_payloads(q, source.collection, ledger)
    units = build_units(payloads, kb_id=source.kb_id, unit_chunks=source.graph_unit_chunks)
    if not units:
        raise ValueError("The active chunks have no text, nothing to extract from")
    return units


def file_chunks(kb_id: str, file_id: str, limit: int = PREVIEW_CHUNK_LIMIT) -> dict[str, Any]:
    """The chunks an indexed file currently has in the store (by chunk_index), together with the chunking
    diagnostics recorded at indexing time. No re-chunking: the chunk ledger is in the state database and
    text plus payload are fetched from Qdrant by point; to see the effect of new rules / parameters use
    "Re-parse"."""
    cfg = settings()
    source = next((s for s in cfg.sources.values() if s.kb_id == kb_id), None)
    with db.connect(cfg.state_db) as con:
        row = db.get_file_by_id(con, file_id)
        if row is None or str(row["kb_id"]) != kb_id:
            raise KeyError(file_id)
        refs = con.execute(
            "SELECT chunk_uid, chunk_index, point_id FROM chunks WHERE file_id = ? AND status = 'active' ORDER BY chunk_index",
            (file_id,)).fetchall()
    keys = row.keys()
    diag = json.loads(row["chunk_diag_json"]) if ("chunk_diag_json" in keys and row["chunk_diag_json"]) else None
    total = len(refs)
    refs = refs[:limit]
    items: list[dict[str, Any]] = []
    if refs:
        q = qdrant_client(cfg.qdrant_url, cfg.qdrant_api_key)
        collection = str(row["collection"])
        payloads: dict[str, dict[str, Any]] = {}
        ids = [str(r["point_id"]) for r in refs]
        for i in range(0, len(ids), 200):
            for p in q.retrieve(collection_name=collection, ids=ids[i:i + 200], with_payload=True, with_vectors=False):
                payloads[str(p.id)] = dict(p.payload or {})
        for r in refs:
            pl = payloads.get(str(r["point_id"]), {})
            items.append({
                "chunk_index": int(r["chunk_index"]), "chunk_uid": str(r["chunk_uid"]),
                "tokens": int(pl.get("token_count") or 0),
                "block_id": str(pl.get("block_id") or ""), "block_type": str(pl.get("block_type") or ""),
                "page_idx": pl.get("page_idx"), "page_end": pl.get("page_end", pl.get("page_idx")),
                "section_path": [str(x) for x in (pl.get("section_path") or [])],
                "title": pl.get("title"), "caption": pl.get("caption"),
                "embedding_context": pl.get("embedding_context"),
                "table_flags": pl.get("table_flags") or None, "table_repair": pl.get("table_repair") or None,
                "text": str(pl.get("text") or ""),
            })
    profile = str(row["parser_profile"]) if ("parser_profile" in keys and row["parser_profile"]) else None
    return {
        "file": {"file_id": file_id, "rel_path": str(row["rel_path"]), "parser_profile": profile, "chunks_total": total},
        "max_tokens": int(source.max_tokens) if source is not None else None,
        "overlap_tokens": int(source.overlap_tokens) if source is not None else None,
        "diagnostics": diag,
        "chunks": items,
        "truncated": total > limit,
    }


# ── graph preview: a subset of the current version's graph.json entities / relations for the console to draw ──
_GRAPH_JSON_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def _load_graph_json(path: Path) -> dict[str, Any]:
    """graph.json is two or three MB; keep one copy cached by path + mtime (a new version means a new path,
    and the old copy is dropped)."""
    key = str(path)
    mtime = path.stat().st_mtime
    hit = _GRAPH_JSON_CACHE.get(key)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    data = json.loads(path.read_text(encoding="utf-8"))
    _GRAPH_JSON_CACHE.clear()
    _GRAPH_JSON_CACHE[key] = (mtime, data)
    return data


def _entity_upper(e: dict[str, Any]) -> str:
    return str(e.get("upper") or e.get("parent_type") or e.get("type") or "")


def graph_preview_pick(graph: dict[str, Any], *, limit: int = 60, q: str = "", upper: str = "", key: str = "") -> dict[str, Any]:
    """Pick the part of one graph version to draw: by default the top `limit` entities by degree (relation
    count) + frequency, boilerplate entities excluded; limit <= 0 means no cap ("whole graph": every entity
    and relation of the version, boilerplate included, drawn faded by the frontend). With key, that entity
    is the focus; with q, the best-matching entity is the focus, and the focus plus its one-hop neighbours
    are taken; with upper, only entities of that upper ontology class are picked. Only edges whose both
    ends were picked are kept."""
    from collections import Counter

    entities = [e for e in (graph.get("entities") or []) if e.get("key")]
    relations = [r for r in (graph.get("relations") or []) if r.get("source_key") and r.get("target_key")]
    degree: Counter = Counter()
    for r in relations:
        degree[str(r["source_key"])] += 1
        degree[str(r["target_key"])] += 1
    by_key = {str(e["key"]): e for e in entities}
    upper_counts = Counter(_entity_upper(e) for e in entities if not e.get("boilerplate"))
    type_counts = Counter(str(e.get("type") or "") for e in entities if not e.get("boilerplate"))

    def score(e: dict[str, Any]) -> tuple[int, int]:
        return (degree.get(str(e["key"]), 0), int(e.get("frequency") or 0))

    focus = by_key.get(str(key or "").strip()) if key else None
    ql = str(q or "").strip().lower()
    if focus is None and ql:
        cands = [e for e in entities if ql in str(e.get("title") or "").lower()
                 or any(ql in str(a).lower() for a in (e.get("aliases") or []))]
        if cands:
            focus = max(cands, key=score)
    if focus is not None:
        fk = str(focus["key"])
        neigh: set[str] = set()
        for r in relations:
            if str(r["source_key"]) == fk:
                neigh.add(str(r["target_key"]))
            elif str(r["target_key"]) == fk:
                neigh.add(str(r["source_key"]))
        pool = sorted((by_key[k] for k in neigh if k in by_key and k != fk), key=score, reverse=True)
        chosen = [focus] + (pool if limit <= 0 else pool[: max(0, limit - 1)])
    else:
        pool = [e for e in entities if (limit <= 0 or not e.get("boilerplate")) and (not upper or _entity_upper(e) == upper)]
        ranked = sorted(pool, key=score, reverse=True)
        chosen = ranked if limit <= 0 else ranked[:limit]
    keys = {str(e["key"]) for e in chosen}
    edges = [r for r in relations if str(r["source_key"]) in keys and str(r["target_key"]) in keys]
    # Document-scoped entities (part / property / process are scoped to their document) carry a short label
    # of the document they came from: date / version / file name. Three annual reports each have an
    # "Overview"; unlabelled, they are three identical dots on the graph.
    docs_meta = graph.get("documents") or {}
    docs_meta = docs_meta if isinstance(docs_meta, dict) else {}

    def doc_label(e: dict[str, Any]) -> str | None:
        scope = str(e.get("scope") or "")
        if not scope:
            return None
        meta = docs_meta.get(scope) or {}
        rel = str(meta.get("rel_path") or "")
        return (str(meta.get("date") or "") or str(meta.get("version") or "")
                or (rel.rsplit("/", 1)[-1].rsplit(".", 1)[0] if rel else scope.rsplit(":", 1)[-1][-6:]))

    nodes = [{
        "key": str(e["key"]), "title": str(e.get("title") or ""), "type": str(e.get("type") or ""),
        "upper": _entity_upper(e), "degree": degree.get(str(e["key"]), 0),
        "frequency": int(e.get("frequency") or 0), "docs": len(e.get("doc_ids") or []),
        "description": str(e.get("description") or "")[:240], "boilerplate": bool(e.get("boilerplate")),
        "scope": str(e.get("scope") or "") or None, "doc": doc_label(e),
    } for e in chosen]
    return {
        "nodes": nodes,
        "edges": [{"source": str(r["source_key"]), "target": str(r["target_key"]), "predicate": str(r.get("predicate") or ""),
                   "strength": r.get("strength_sum"), "description": str(r.get("description") or "")[:160]} for r in edges],
        "totals": {"entities": len(entities), "relations": len(relations),
                   "boilerplate": sum(1 for e in entities if e.get("boilerplate"))},
        "upper_counts": dict(upper_counts.most_common()),
        "type_counts": dict(type_counts.most_common(40)),
        "focus": str(focus["key"]) if focus is not None else None,
    }


def graph_merges(kb_id: str, limit: int = 2000) -> dict[str, Any]:
    """Entity resolution log of the current version: every merged pair (merged into whom, source, evidence
    category) and every rejected pair (reason), read from resolution_log / resolution_rejected in the
    graph.json build artifact, for the "Merges" drawer of the graph preview."""
    from kb_pipeline.graph.build import graph_paths

    cfg = settings()
    found = next(((k, s) for k, s in cfg.sources.items() if s.kb_id == kb_id), None)
    if found is None:
        raise KeyError(kb_id)
    source_key, source = found
    with db.connect(cfg.state_db) as con:
        latest = db.latest_successful_graph_build(con, source_key)
    if latest is None:
        return {"version": None, "merges": [], "rejected": [], "stats": {}}
    version = str(latest["graph_version"])
    graph_file = graph_paths(cfg, source, version).graph_file
    if not graph_file.exists():
        raise ValueError(f"The graph artifacts of the current version {version} are not on disk ({graph_file.name}); rebuild the graph and try again")
    graph = _load_graph_json(graph_file)
    limit = max(1, min(int(limit or 2000), 5000))
    merges = list(graph.get("resolution_log") or [])[:limit]
    rejected = list(graph.get("resolution_rejected") or [])[:limit]
    res = dict(((graph.get("stats") or {}).get("resolution") or {}))
    stats = {k: res.get(k) for k in ("candidates", "auto_pairs", "embedding_candidates", "yes", "vetoed", "rechecked", "recheck_dropped",
                                     "entities_before", "entities_after", "yes_by_category") if k in res}
    return {"version": version, "merges": merges, "rejected": rejected, "stats": stats,
            "totals": {"merges": len(graph.get("resolution_log") or []), "rejected": len(graph.get("resolution_rejected") or [])}}


def graph_preview(kb_id: str, limit: int = 60, q: str = "", upper: str = "", key: str = "") -> dict[str, Any]:
    """Graph preview of the current version: read from the graph.json build artifact, Neo4j is not touched.
    Returns empty when no graph has been built. limit is clamped to 10 ~ 3000 (2026-09-12, user decision:
    presets 100 ~ 3000, the console no longer has a "whole graph" button); passing 0 still returns the whole
    version, the interface is kept."""
    from kb_pipeline.graph.build import graph_paths

    cfg = settings()
    found = next(((k, s) for k, s in cfg.sources.items() if s.kb_id == kb_id), None)
    if found is None:
        raise KeyError(kb_id)
    source_key, source = found
    with db.connect(cfg.state_db) as con:
        latest = db.latest_successful_graph_build(con, source_key)
    if latest is None:
        return {"version": None, "nodes": [], "edges": [], "totals": {"entities": 0, "relations": 0},
                "upper_counts": {}, "type_counts": {}, "focus": None}
    version = str(latest["graph_version"])
    graph_file = graph_paths(cfg, source, version).graph_file
    if not graph_file.exists():
        raise ValueError(f"The graph artifacts of the current version {version} are not on disk ({graph_file.name}); rebuild the graph and try again")
    limit = int(limit if limit is not None else 60)
    limit = 0 if limit <= 0 else max(10, min(limit, 3000))
    out = graph_preview_pick(_load_graph_json(graph_file), limit=limit, q=q, upper=upper, key=key)
    out["version"] = version
    out["build_kind"] = str(latest["build_kind"] or "full") if "build_kind" in latest.keys() else "full"
    return out


def reparse_kb(kb_id: str, reason: str = "web console reparse") -> dict[str, Any]:
    cfg = settings()
    with db.connect(cfg.state_db) as con:
        row = con.execute("SELECT status FROM kb_sources WHERE kb_id=?", (kb_id,)).fetchone()
        if row is None:
            raise KeyError(kb_id)          # used to return 200 {"requeued": 0} for a non-existent KB
        if str(row["status"]) != "active":
            raise ValueError("The knowledge base is not enabled, so it cannot be re-parsed")
        run_id = db.begin_run(con, "web-reparse", note=f"{kb_id}: {reason}")
        count = requeue_kb_files(con, ingest_run_id=run_id, kb_id=kb_id, reason=reason)
        db.finish_run(con, run_id, added_count=0, updated_count=count, moved_count=0,
                      deleted_count=0, failed_count=0)
    kick_worker()
    return {"requeued": count}


# ── job details / failure panel ─────────────────────────────────

_JOB_FIELDS = ("job_id", "kb_id", "file_id", "job_type", "status", "stage", "parser_profile", "retry_count",
               "next_attempt_at", "started_at", "finished_at", "created_at", "updated_at", "error",
               "cancel_requested", "locked_by")


def job_detail(job_id: str) -> dict[str, Any]:
    """Metadata + timeline (phase switches, milestones, errors) of one parse job, for the console dialog."""
    cfg = settings()
    with db.connect(cfg.state_db) as con:
        row = con.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        job = {k: row[k] for k in _JOB_FIELDS if k in row.keys()}
        file_row = db.get_file_by_id(con, str(row["file_id"])) if row["file_id"] else None
        events = db.job_events(con, job_id)
    file_info = None
    if file_row is not None:
        file_info = {k: file_row[k] for k in ("file_id", "rel_path", "filename", "size", "mime_type",
                                              "content_version", "indexed_version", "status")}
    synthesized = False
    if not events:
        # The timeline table was added later; earlier jobs (and those whose events were pruned) have no event
        # rows. Synthesize an outline from the job's own timestamps and say so explicitly -- it does not pose
        # as a real record.
        events = _synthesize_job_events(job)
        synthesized = bool(events)
    return {"job": job, "file": file_info, "events": events, "synthesized": synthesized}


def _synthesize_job_events(job: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def add(ts, kind: str, text: str) -> None:
        if ts:
            out.append({"ts": int(ts), "kind": kind, "text": text, "synthesized": True})

    add(job.get("created_at"), "info", "Queued")
    add(job.get("started_at"), "stage", str(job.get("stage") or "Started"))
    status = str(job.get("status") or "")
    if status == "done":
        add(job.get("finished_at"), "done", "Done")
    elif status == "failed":
        add(job.get("finished_at") or job.get("updated_at"), "error", str(job.get("error") or "Failed")[:300])
    elif status == "cancelled":
        add(job.get("finished_at") or job.get("updated_at"), "cancelled", "Cancelled")
    elif status == "retry":
        add(job.get("updated_at"), "retry", str(job.get("error") or "Waiting to retry")[:300])
    out.sort(key=lambda e: e["ts"])
    return out


def cancel_job(job_id: str) -> dict[str, Any]:
    cfg = settings()
    with db.connect(cfg.state_db) as con:
        outcome = db.request_job_cancel(con, job_id, "Cancelled from the console")
    return {"job_id": job_id, "outcome": outcome}


def failed_jobs(limit: int = 100) -> dict[str, Any]:
    """Failed / backing-off parse jobs plus the queue depth, for the failure section of the "Parse status"
    panel."""
    cfg = settings()
    with db.connect(cfg.state_db) as con:
        rows = con.execute(
            "SELECT j.job_id, j.kb_id, j.status, j.stage, j.error, j.retry_count, j.next_attempt_at, "
            "j.updated_at, f.rel_path, f.file_id FROM jobs j LEFT JOIN files f ON f.file_id = j.file_id "
            "WHERE j.job_type='parse' AND j.status IN ('failed','retry') "
            "ORDER BY CASE j.status WHEN 'failed' THEN 0 ELSE 1 END, j.updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        depth = {r["status"]: int(r["n"]) for r in con.execute(
            "SELECT status, COUNT(*) AS n FROM jobs WHERE job_type='parse' "
            "AND status IN ('running','queued','retry') GROUP BY status")}
    now = int(time.time())
    items = []
    for r in rows:
        items.append({
            "job_id": str(r["job_id"]), "kb_id": str(r["kb_id"]), "status": str(r["status"]),
            "stage": r["stage"], "error": (str(r["error"])[:300] if r["error"] else None),
            "retry_count": int(r["retry_count"] or 0),
            "retry_in": max(0, int(r["next_attempt_at"] or 0) - now) if str(r["status"]) == "retry" else None,
            "rel_path": r["rel_path"], "file_id": r["file_id"], "updated_at": r["updated_at"],
        })
    return {"jobs": items, "queue": {k: depth.get(k, 0) for k in ("running", "queued", "retry")}}


def retry_file(file_id: str) -> dict[str, Any]:
    cfg = settings()
    with db.connect(cfg.state_db) as con:
        row = db.get_file_by_id(con, file_id)
        if row is None:
            raise KeyError(file_id)
        if str(row["status"]) == "deleted":
            raise ValueError("This file has been removed from the source and cannot be re-parsed")
        run_id = db.begin_run(con, "web-retry", note=str(row["source_path"]))
        job_id = requeue_single_file(con, ingest_run_id=run_id, file_row=row)
        db.finish_run(con, run_id, added_count=0, updated_count=1 if job_id else 0,
                      moved_count=0, deleted_count=0, failed_count=0)
    kick_worker()
    return {"job_id": job_id, "already_active": job_id is None}


# ── LLM registry ────────────────────────────────────────────────────────

def _mask(row) -> dict[str, Any]:
    return {
        "name": str(row["name"]),
        "base_url": str(row["base_url"]),
        "model_id": str(row["model_id"]),
        "has_api_key": bool(str(row["api_key"] or "")),
        "protocol": str(row["protocol"] if "protocol" in row.keys() else "openai") or "openai",
    }


def _llm_usage(con) -> dict[str, list[dict[str, Any]]]:
    """Model name -> which steps of which knowledge bases reference it.

    For **information** only; it no longer blocks deletion: deleting and being in use are decoupled (see
    remove_llm). The console uses it in the confirmation dialog to spell out "which fields go empty after
    deleting".
    """
    from kb_pipeline.graph.build import GRAPH_STEP_LABELS

    usage: dict[str, list[dict[str, Any]]] = {}
    for row in discovery.known_sources(con):
        try:
            kb_llm = (json.loads(row["config_json"] or "{}").get("graph_llm") or {})
        except Exception:
            continue
        by_model: dict[str, list[str]] = {}
        for step, name in kb_llm.items():
            if name:
                by_model.setdefault(str(name), []).append(
                    GRAPH_STEP_LABELS.get(step, step))
        for name, labels in by_model.items():
            usage.setdefault(name, []).append({
                "kb_id": str(row["kb_id"]),
                "source_root": str(row["source_root"]),
                "steps": labels,
            })
    return usage


def list_llms() -> list[dict[str, Any]]:
    cfg = settings()
    db.init_db(cfg.state_db)
    with db.connect(cfg.state_db) as con:
        usage = _llm_usage(con)
        out = []
        for r in db.list_llms(con):
            item = _mask(r)
            item["used_by"] = usage.get(str(item.get("name")), [])
            out.append(item)
        return out


def _endpoint_id(url: str) -> tuple[str, str, int | None]:
    """(scheme, host, port): used to tell "is it still the same service"; the path does not count."""
    try:
        parts = urlsplit(url.strip())
        return (parts.scheme.lower(), (parts.hostname or "").lower(), parts.port)
    except ValueError:
        return ("", url.strip(), None)


def _probe_llm(base_url: str, api_key: str, model_id: str, protocol: str = "openai") -> None:
    """Connectivity test before saving: ask the model to reply with OK only; a reply containing ok
    (case-insensitive) passes; any network / protocol / content problem raises ValueError and the save is
    refused."""
    base = base_url.rstrip("/")
    headers = {"Content-Type": "application/json"}
    body: dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Connectivity test: reply with OK only"}],
        # reasoning models burn their thinking budget first; too small and the body comes back empty
        "max_tokens": 512,
        "temperature": 0,
    }
    if protocol == "anthropic":
        # Same joining rule as litellm on the graph build side: append /v1/messages when base lacks it
        url = base if base.endswith("/v1/messages") else base + "/v1/messages"
        headers["anthropic-version"] = "2023-06-01"
        if api_key:
            headers["x-api-key"] = api_key
    else:
        url = base + "/chat/completions"
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
    try:
        # No redirects: following one to another host with the key attached hands the key to that host
        resp = requests.post(url, headers=headers, json=body, timeout=30, allow_redirects=False)
    except requests.RequestException as exc:
        raise ValueError(f"Connectivity test failed: cannot reach {url} ({exc.__class__.__name__})")
    if 300 <= resp.status_code < 400:
        raise ValueError(f"Connectivity test failed: the service redirected ({resp.status_code} → {str(resp.headers.get('location') or '')[:120]}); enter the final address directly")
    if resp.status_code != 200:
        raise ValueError(f"Connectivity test failed: HTTP {resp.status_code} {resp.text[:120]}")
    try:
        if protocol == "anthropic":
            blocks = resp.json()["content"]
            content = "".join(str(b.get("text") or "") for b in blocks if b.get("type") == "text")
        else:
            content = str(resp.json()["choices"][0]["message"]["content"] or "")
    except Exception:
        raise ValueError("Connectivity test failed: the response is not a valid chat format for the selected protocol")
    if not re.search(r"\bok\b", content, re.IGNORECASE):
        raise ValueError(f"Connectivity test did not pass: the model replied “{content.strip()[:60]}”, which does not contain OK")


def save_llm(payload: dict[str, Any]) -> dict[str, Any]:
    name = str(payload.get("name") or "").strip()
    base_url = str(payload.get("base_url") or "").strip()
    model_id = str(payload.get("model_id") or "").strip()
    protocol = str(payload.get("protocol") or "openai").strip().lower()
    if not name or not base_url or not model_id:
        raise ValueError("name, base_url and model_id are all required")
    # The name goes into a URL path (DELETE /llms/{name}); a name containing / can be saved but never deleted
    if not re.fullmatch(r"[\w.\- ]{1,64}", name, re.UNICODE):
        raise ValueError("Model names may only contain letters, CJK characters, digits, spaces, dots, underscores and hyphens (64 characters at most)")
    if protocol not in ("openai", "anthropic"):
        raise ValueError("protocol must be 'openai' or 'anthropic'")
    cfg = settings()
    with db.connect(cfg.state_db) as con:
        existing = db.get_llm(con, name)
        api_key = str(payload.get("api_key") or "")
        # Leaving the key empty while editing = keep the stored key; it is cleared only when clear_api_key is
        # passed explicitly (endpoints that need no key, like a local vLLM, used to have no way to remove a
        # key entered by mistake). Keeping it is limited to the same endpoint: if the address's scheme / host /
        # port or the protocol changed, the old key must not follow to the new address and has to be
        # re-entered.
        if (api_key == "" and existing is not None and str(existing["api_key"] or "") and not payload.get("clear_api_key")
                and (_endpoint_id(str(existing["base_url"] or "")) != _endpoint_id(base_url)
                     or str(existing["protocol"] or "openai").lower() != protocol)):
            raise ValueError("The endpoint or protocol changed; enter the API key again, or tick \"clear key\" for endpoints that need none")
        if payload.get("clear_api_key"):
            db.upsert_llm(con, name=name, base_url=base_url, api_key="",
                          model_id=model_id, protocol=protocol)
            con.execute("UPDATE llm_registry SET api_key='' WHERE name=?", (name,))
            existing = db.get_llm(con, name)
            api_key = ""
        probe_key = api_key or (str(existing["api_key"] or "") if existing is not None else "")
        _probe_llm(base_url, probe_key, model_id, protocol)
        db.upsert_llm(
            con,
            name=name,
            base_url=base_url,
            api_key=api_key,
            model_id=model_id,
            protocol=protocol,
        )
        return _mask(db.get_llm(con, name))


def remove_llm(name: str) -> dict[str, Any]:
    """Delete one model registration. **No longer refused because "a knowledge base is using it".**

    It used to be a hard block: as soon as the name appeared in any KB's graph_llm it could not be deleted.
    But being referenced is not being in use -- once the knowledge graph is turned off, the whole
    graph-fields block (the three model slots and the label extraction button) is disabled; none of those
    names can run then, yet they still locked the model registry.

    Now decoupled: the deletion just happens, and every KB referencing the model gets that slot cleared
    (instead of keeping a dangling name pointing at a deleted model -- which would make the build report
    "not in the registry" when the truth is "you have not selected one yet"). The console lists the
    affected KBs and slots in the confirmation dialog first, so this is not silent.

    Returns the cleared references so the console can report them truthfully.
    """
    cfg = settings()
    with db.connect(cfg.state_db) as con:
        affected = _llm_usage(con).get(name, [])
        for row in discovery.known_sources(con):
            try:
                kb_llm = dict(json.loads(row["config_json"] or "{}").get("graph_llm") or {})
            except Exception:
                continue
            pruned = {k: v for k, v in kb_llm.items() if v != name}
            if pruned != kb_llm:
                discovery.set_config(con, str(row["kb_id"]), {"graph_llm": pruned or None})
        if not db.delete_llm(con, name):
            raise KeyError(name)
        con.commit()
    return {"deleted": name, "cleared": affected}


def init_state() -> None:
    db.init_db(settings().state_db)


# ── service restarts ────────────────────────────────────────────────────

# Console row -> docker containers of the compose stack (deployment/compose).
# The compose restart policies own the lifecycle; the button just bounces the
# container(s).
SERVICE_CONTAINERS: dict[str, list[str]] = {
    # The "Database services" row covers three stores: vector store, keyword index, graph store
    "database": ["carrel-qdrant", "carrel-opensearch", "carrel-neo4j"],
    "mineru": ["carrel-mineru"],
    "embedding": ["carrel-embedding"],
    "visual_embedding": ["carrel-vl-embedding"],
    "vlm": ["carrel-vlm"],
    # The "Reranker service" row covers two rerank models: text rerank + cross-modal rerank. The parse /
    # graph build pipelines do not use them (they only serve retrieval); they are managed here so that
    # "Stop all" really stops everything on the GPU.
    "reranker": ["carrel-reranker", "carrel-vl-reranker"],
}

# The database and parse rows are always shown (the pipeline cannot run without them); the four model rows
# are shown only when listed in KB_CONSOLE_SERVICES -- their addresses may well be public endpoints, with no
# container on this machine to probe or restart.
ALWAYS_SHOWN_SERVICES: tuple[str, ...] = ("database", "mineru")

# Services the pipeline does not depend on: restarting one on its own interrupts no parse / graph build job,
# so it skips the busy check (restart all / stop all are still checked, since they take the stores down too).
_PIPELINE_INDEPENDENT: frozenset[str] = frozenset({"reranker"})


def managed_services() -> tuple[str, ...]:
    """Service rows managed by the console (KB_CONSOLE_SERVICES): only these get their model service probed,
    a restart button and a place in restart all / stop all. When the settings object lacks the field (old
    test stubs), the default two rows apply."""
    from kb_pipeline.config import CONSOLE_SERVICES_DEFAULT

    keys = getattr(settings(), "console_services", None)
    if keys is None:
        keys = CONSOLE_SERVICES_DEFAULT
    return tuple(key for key in keys if key in SERVICE_CONTAINERS)


def _refuse_if_busy(action: str, force: bool) -> None:
    """Restarting / stopping Qdrant, OpenSearch or the VLM makes a running parse fail (it retries) or throws
    away a graph build's progress (the whole round restarts). Without force, refuse while the system is
    busy; the frontend shows the reason and asks the user to confirm a second time."""
    if force:
        return
    from kb_pipeline.maintenance import service_busy

    busy, reasons = service_busy(settings())
    if busy:
        raise ValueError(
            f"The system is busy; a {action} would interrupt running work: " + "; ".join(reasons[:3])
            + f". Click again to force the {action}."
        )


def _docker_control(command: str, containers: list[str]) -> None:
    """Send docker restart / stop to a group of containers. Fire and forget: docker restart blocks until the
    stop grace period ends, and the health probe is the honest "it is back" signal. Output goes to a log
    instead of DEVNULL: a renamed / missing container used to fail completely silently, with the console
    only showing "command sent"."""
    log_dir = settings().log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "web-docker-restart.log").open("ab") as log:
        for name in containers:
            subprocess.Popen(
                ["docker", command, name],
                stdout=log, stderr=log, start_new_session=True,
            )


def _all_service_containers() -> list[str]:
    """Targets of restart all / stop all: the managed rows' containers, deduplicated in row order."""
    out: list[str] = []
    for key in managed_services():
        for name in SERVICE_CONTAINERS[key]:
            if name not in out:
                out.append(name)
    return out


def restart_service(key: str, force: bool = False) -> dict[str, Any]:
    containers = SERVICE_CONTAINERS.get(key)
    if not containers or key not in managed_services():
        raise KeyError(key)
    if key not in _PIPELINE_INDEPENDENT:
        _refuse_if_busy("restart", force)
    _docker_control("restart", containers)
    return {"restarting": containers}


def restart_all_services(force: bool = False) -> dict[str, Any]:
    """The "Restart all" action: every system service container listed in the console. docker restart brings
    a stopped container straight up."""
    containers = _all_service_containers()
    _refuse_if_busy("restart", force)
    _docker_control("restart", containers)
    return {"restarting": containers}


def stop_all_services(force: bool = False) -> dict[str, Any]:
    """The "Stop all" action: stop every system service container. The parse queue yields because Qdrant is
    unreachable (the queue is kept); use "Restart all" to bring everything back."""
    containers = _all_service_containers()
    _refuse_if_busy("shutdown", force)
    _docker_control("stop", containers)
    return {"stopping": containers}


# ── health ──────────────────────────────────────────────────────────────

_health_cache: dict[str, Any] = {"at": 0.0, "value": None}


# The six timer jobs (unit names without suffix) and each one's yield-count file name under
# runtime/state/maintenance (see scripts/lib/kb-maint-defer.sh: graph-rebuild, cleanup-<subcommand>).
# Scan and parse have no yield count.
_TIMER_UNITS: tuple[tuple[str, str, str, str | None], ...] = (
    ("scan", "Mirror scan", "carrel-scan", None),
    ("worker", "Parse queue", "carrel-worker", None),
    ("graph", "Graph maintenance check", "carrel-graph-rebuild", "graph-rebuild"),
    ("gc", "Inactive point / parse asset GC", "carrel-qdrant-gc", "cleanup-parse-assets-gc"),
    ("cache", "Weekly cache cleanup", "carrel-cache-weekly", "cleanup-weekly"),
    ("logs", "Monthly log rotation", "carrel-logs-monthly", "cleanup-monthly"),
)
_SYSTEMD_TS_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):\d{2}")


def parse_systemctl_show(output: str) -> dict[str, dict[str, str]]:
    """Split the output of `systemctl show -p A,B unit1 unit2` into dicts keyed by Id; empty output / systemd
    not installed -> {}."""
    blocks: dict[str, dict[str, str]] = {}
    for chunk in str(output or "").strip().split("\n\n"):
        kv = dict(line.split("=", 1) for line in chunk.splitlines() if "=" in line)
        if kv.get("Id"):
            blocks[kv["Id"]] = kv
    return blocks


def _systemctl_show(units: list[str], props: list[str]) -> dict[str, dict[str, str]]:
    try:
        out = subprocess.run(["systemctl", "--user", "show", f"--property={','.join(props)}", *units],
                             capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return {}
    return parse_systemctl_show(out)


def _systemctl_list_timers() -> dict[str, int]:
    """`systemctl --user list-timers --all --output=json` -> {unit: next trigger in wall-clock microseconds}.
    Relative-time timers (OnActiveSec / OnUnitActiveSec) have no NextElapseUSecRealtime in `systemctl show`,
    only a monotonic-clock value; list-timers converts it to a wall-clock instant for us (since the timers
    were switched to relative time on 2026-09-09). Returns {} when unavailable."""
    try:
        out = subprocess.run(["systemctl", "--user", "list-timers", "--all", "--output=json"],
                             capture_output=True, text=True, timeout=5).stdout
        rows = json.loads(out or "[]")
    except Exception:
        return {}
    result: dict[str, int] = {}
    for row in rows if isinstance(rows, list) else []:
        unit, nxt = str(row.get("unit") or ""), row.get("next")
        if unit and isinstance(nxt, (int, float)) and nxt > 0:
            result[unit] = int(nxt)
    return result


def merge_timer_next(timers: dict[str, dict[str, str]], next_usec: dict[str, int]) -> dict[str, dict[str, str]]:
    """For timers whose show output lacks NextElapseUSecRealtime, fill it from the list-timers wall-clock
    microseconds (formatted like a systemd timestamp, parsed uniformly later)."""
    for unit, tm in timers.items():
        if tm.get("NextElapseUSecRealtime"):
            continue
        usec = next_usec.get(unit)
        if usec:
            tm["NextElapseUSecRealtime"] = time.strftime("%a %Y-%m-%d %H:%M:%S %Z", time.localtime(usec / 1_000_000))
    return timers


def _short_systemd_ts(value: str | None) -> str | None:
    m = _SYSTEMD_TS_RE.search(str(value or ""))
    return f"{m.group(2)}-{m.group(3)} {m.group(4)}:{m.group(5)}" if m else None


def _maintenance_defers(folder: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        for f in folder.glob("*.defers"):
            try:
                out[f.stem] = int(f.read_text(encoding="utf-8").strip() or "0")
            except (ValueError, OSError):
                out[f.stem] = 0
    except OSError:
        pass
    return out


def timer_health(services: dict[str, dict[str, str]], timers: dict[str, dict[str, str]],
                 defers: dict[str, int]) -> list[dict[str, Any]]:
    """Status rows of the timer jobs (health check R6): last result, whether running, next trigger, how many
    consecutive rounds yielded. ok=False only when the last run failed; a unit systemd cannot find -> None
    (unknown, not broken)."""
    rows: list[dict[str, Any]] = []
    for key, label, unit, defer_name in _TIMER_UNITS:
        svc = services.get(f"{unit}.service") or {}
        tm = timers.get(f"{unit}.timer") or {}
        state = svc.get("ActiveState")
        result = svc.get("Result")
        running = state in ("active", "activating")
        failed = bool(svc) and (state == "failed" or (result not in (None, "", "success") and not running))
        rows.append({
            "key": key, "label": label, "unit": unit,
            "ok": (not failed) if svc else None,
            "running": running, "state": state, "result": result,
            "timer_active": (tm.get("ActiveState") == "active") if tm else None,
            "last": _short_systemd_ts(tm.get("LastTriggerUSec")),
            "next": _short_systemd_ts(tm.get("NextElapseUSecRealtime")),
            "defers": int(defers.get(defer_name or "", 0) or 0),
        })
    return rows


def _probe_neo4j(uri: str, user: str, password: str | None) -> bool:
    """Graph store health probe: open one connection over bolt (HTTP 7474 is not necessarily exposed).
    Unreachable and authentication failure both count as unhealthy."""
    if not uri:
        return False
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(uri, auth=(user, password or ""), connection_timeout=4)
        try:
            driver.verify_connectivity()
            return True
        finally:
            driver.close()
    except Exception:
        return False


def health() -> dict[str, Any]:
    cfg = settings()
    # With the panel open this is polled every 5 seconds, while probing 5-6 services serially with a 4-second
    # timeout each means 20 seconds per call when several are down at once -- requests pile up and
    # out-of-order responses make the status flip back and forth. Probe concurrently and cache for 3 seconds.
    now = time.time()
    if _health_cache["value"] is not None and now - float(_health_cache["at"]) < 3.0:
        return dict(_health_cache["value"])

    def probe(url: str) -> bool:
        try:
            return requests.get(url, timeout=4).status_code == 200
        except Exception:
            return False

    def vllm(base: str) -> bool:
        root = base.rstrip("/")
        if root.endswith("/v1"):
            root = root[: -len("/v1")]
        return probe(f"{root}/health")

    neo4j_uri = str(getattr(cfg, "neo4j_uri", "") or "")
    managed = managed_services()
    checks = {
        "qdrant": lambda: probe(f"{cfg.qdrant_url.rstrip('/')}/healthz"),
        "opensearch": lambda: probe(f"{cfg.opensearch_url.rstrip('/')}/_cluster/health"),
        "neo4j": lambda: _probe_neo4j(neo4j_uri, getattr(cfg, "neo4j_user", "neo4j"), getattr(cfg, "neo4j_password", None)),
        "mineru": lambda: probe(f"{cfg.mineru_url.rstrip('/')}/health"),
    }
    # Only probe the managed model service rows: an unmanaged address may be a public endpoint, and probing
    # vLLM's /health there would just stay red.
    if "embedding" in managed:
        checks["embedding"] = lambda: vllm(cfg.embedding_base_url)
    if "vlm" in managed:
        checks["vlm"] = lambda: vllm(cfg.vlm_base_url)
    if "visual_embedding" in managed and cfg.visual_embedding_enabled:
        checks["visual_embedding"] = lambda: vllm(cfg.visual_embedding_base_url)
    # Rerank: text / cross-modal are probed separately and the frontend merges them into one "Reranker
    # service" row; an empty address = not deployed, give None so the row is hidden
    reranker_url = str(getattr(cfg, "reranker_base_url", "") or "")
    visual_reranker_url = str(getattr(cfg, "visual_reranker_base_url", "") or "")
    if "reranker" in managed and reranker_url:
        checks["reranker"] = lambda: vllm(reranker_url)
    if "reranker" in managed and visual_reranker_url:
        checks["visual_reranker"] = lambda: vllm(visual_reranker_url)

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=len(checks)) as pool:
        futures = {name: pool.submit(fn) for name, fn in checks.items()}
        result: dict[str, Any] = {name: future.result() for name, future in futures.items()}
    for name in ("embedding", "vlm", "visual_embedding", "reranker", "visual_reranker"):
        result.setdefault(name, None)
    if not neo4j_uri:
        result["neo4j"] = None          # no graph store: "Database services" checks only Qdrant / OpenSearch
    result["managed"] = list(managed)
    # Parse backend: chosen by the container itself from the hardware (MINERU_BACKEND=auto) or fixed in the
    # config; shown on the parse service row
    from kb_pipeline.parsers.mineru_backend import resolve_backend

    result["mineru_backend"], result["mineru_backend_source"] = resolve_backend(getattr(cfg, "runtime_dir", None))
    result["parse_enabled"] = cfg.parse_enabled
    units = [u for _, _, u, _ in _TIMER_UNITS]
    result["timers"] = timer_health(
        _systemctl_show([f"{u}.service" for u in units], ["Id", "ActiveState", "SubState", "Result"]),
        merge_timer_next(_systemctl_show([f"{u}.timer" for u in units], ["Id", "ActiveState", "NextElapseUSecRealtime", "LastTriggerUSec"]),
                         _systemctl_list_timers()),
        _maintenance_defers(Path(cfg.state_db).parent / "maintenance"),
    )
    _health_cache["at"] = time.time()
    _health_cache["value"] = result
    return dict(result)
