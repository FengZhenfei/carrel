from __future__ import annotations

import hashlib
import json
import os
import socket
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

from .models import SourceFile
from .models import UnifiedChunk
from .utils import stable_json_hash


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS files (
  file_id TEXT PRIMARY KEY,
  kb_id TEXT NOT NULL,
  collection TEXT NOT NULL,
  source_root TEXT NOT NULL,
  source_type TEXT NOT NULL,
  file_key INTEGER NOT NULL,
  source_path TEXT NOT NULL,
  rel_path TEXT NOT NULL,
  filename TEXT NOT NULL,
  dir TEXT NOT NULL,
  physical_path TEXT NOT NULL,
  mime_type TEXT NOT NULL,
  size INTEGER NOT NULL,
  mtime INTEGER NOT NULL,
  checksum TEXT,
  content_version TEXT NOT NULL,
  metadata_fingerprint TEXT NOT NULL,
  first_seen_at INTEGER NOT NULL,
  last_seen_at INTEGER NOT NULL,
  indexed_version TEXT,
  indexed_parser_profile TEXT,
  status TEXT NOT NULL DEFAULT 'seen',
  UNIQUE(kb_id, file_key)
);

CREATE INDEX IF NOT EXISTS idx_files_kb ON files(kb_id);
CREATE INDEX IF NOT EXISTS idx_files_nc ON files(file_key);
CREATE INDEX IF NOT EXISTS idx_files_status ON files(status);
CREATE INDEX IF NOT EXISTS idx_files_checksum ON files(kb_id, checksum);

CREATE TABLE IF NOT EXISTS ingest_runs (
  ingest_run_id TEXT PRIMARY KEY,
  started_at INTEGER NOT NULL,
  finished_at INTEGER,
  trigger_type TEXT NOT NULL,
  status TEXT NOT NULL,
  added_count INTEGER NOT NULL DEFAULT 0,
  updated_count INTEGER NOT NULL DEFAULT 0,
  moved_count INTEGER NOT NULL DEFAULT 0,
  deleted_count INTEGER NOT NULL DEFAULT 0,
  failed_count INTEGER NOT NULL DEFAULT 0,
  note TEXT
);

CREATE TABLE IF NOT EXISTS jobs (
  job_id TEXT PRIMARY KEY,
  ingest_run_id TEXT,
  file_id TEXT,
  kb_id TEXT NOT NULL,
  collection TEXT NOT NULL,
  file_key INTEGER,
  job_type TEXT NOT NULL,
  parser_profile TEXT,
  status TEXT NOT NULL,
  priority INTEGER NOT NULL DEFAULT 100,
  locked_by TEXT,
  locked_until INTEGER,
  retry_count INTEGER NOT NULL DEFAULT 0,
  next_attempt_at INTEGER NOT NULL DEFAULT 0,
  started_at INTEGER,
  finished_at INTEGER,
  error TEXT,
  payload_json TEXT NOT NULL DEFAULT '{}',
  dedupe_key TEXT,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, next_attempt_at, priority);
CREATE INDEX IF NOT EXISTS idx_jobs_file ON jobs(file_id);

CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_dedupe_active
ON jobs(dedupe_key)
WHERE dedupe_key IS NOT NULL AND status IN ('queued', 'retry', 'running');

CREATE TABLE IF NOT EXISTS llm_registry (
  name TEXT PRIMARY KEY,
  base_url TEXT NOT NULL,
  api_key TEXT NOT NULL DEFAULT '',
  model_id TEXT NOT NULL,
  notes TEXT NOT NULL DEFAULT '',
  builtin INTEGER NOT NULL DEFAULT 0,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS app_config (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
  chunk_uid TEXT PRIMARY KEY,
  file_id TEXT NOT NULL,
  content_version TEXT NOT NULL,
  chunk_index INTEGER NOT NULL,
  point_id TEXT NOT NULL,
  collection TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at INTEGER NOT NULL
);


-- Timeline of a parse job: stage changes, milestones, errors. The console's job detail
-- drawer draws the timeline from it; rows of finished jobs go with prune_job_history.
CREATE TABLE IF NOT EXISTS job_events (
  job_id TEXT NOT NULL,
  ts INTEGER NOT NULL,
  kind TEXT NOT NULL,
  text TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_job_events_job ON job_events(job_id, ts);

CREATE TABLE IF NOT EXISTS failures (
  failure_id TEXT PRIMARY KEY,
  file_id TEXT,
  job_id TEXT,
  stage TEXT NOT NULL,
  error_type TEXT NOT NULL,
  error_message TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  resolved_at INTEGER
);

CREATE TABLE IF NOT EXISTS graph_builds (
  graph_build_id TEXT PRIMARY KEY,
  source_key TEXT NOT NULL,
  kb_id TEXT NOT NULL,
  source_collection TEXT NOT NULL,
  graph_version TEXT NOT NULL,
  status TEXT NOT NULL,
  started_at INTEGER NOT NULL,
  finished_at INTEGER,
  input_rows INTEGER NOT NULL DEFAULT 0,
  active_chunk_count INTEGER NOT NULL DEFAULT 0,
  active_doc_count INTEGER NOT NULL DEFAULT 0,
  source_content_hash TEXT,
  output_dir TEXT,
  manifest_json TEXT NOT NULL DEFAULT '{}',
  error TEXT
);

CREATE INDEX IF NOT EXISTS idx_graph_builds_source
ON graph_builds(source_key, status, finished_at);

CREATE UNIQUE INDEX IF NOT EXISTS idx_graph_builds_version
ON graph_builds(source_collection, graph_version);

CREATE TABLE IF NOT EXISTS graph_build_chunks (
  graph_build_id TEXT NOT NULL,
  point_id TEXT NOT NULL,
  chunk_uid TEXT NOT NULL,
  doc_id TEXT NOT NULL,
  content_version TEXT NOT NULL,
  PRIMARY KEY(graph_build_id, point_id)
);

CREATE INDEX IF NOT EXISTS idx_graph_build_chunks_build
ON graph_build_chunks(graph_build_id);

-- Phase markers of a graph build: one row per completed phase. When the same version is
-- resumed with unchanged config and corpus, recorded phases are skipped (see _run_phase in graph/build.py).
CREATE TABLE IF NOT EXISTS graph_build_phases (
  graph_build_id TEXT NOT NULL,
  phase TEXT NOT NULL,
  done_at INTEGER NOT NULL,
  PRIMARY KEY(graph_build_id, phase)
);

-- Latest verdict of the graph maintenance check (check-rebuild, every 2 hours) per knowledge base:
-- the status card explains why nothing was built, or what was.
CREATE TABLE IF NOT EXISTS graph_checks (
  kb_id TEXT PRIMARY KEY,
  checked_at INTEGER NOT NULL,
  decision_json TEXT NOT NULL
);

-- Extraction units: how one build split the corpus (the text lives in the work dir's units.jsonl; only sizes here).
CREATE TABLE IF NOT EXISTS graph_units (
  graph_build_id TEXT NOT NULL,
  unit_id TEXT NOT NULL,
  doc_id TEXT NOT NULL,
  section TEXT,
  n_tokens INTEGER NOT NULL DEFAULT 0,
  chunk_count INTEGER NOT NULL DEFAULT 0,
  unit_order INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY(graph_build_id, unit_id)
);

-- Extraction result per unit, keyed by (kb, unit, fingerprint): a unit whose key already exists
-- costs no LLM call on the next build. fingerprint = model + type table + predicates + language +
-- gleaning rounds + prompt hash (see graph/extract.py).
CREATE TABLE IF NOT EXISTS graph_extractions (
  kb_id TEXT NOT NULL,
  unit_id TEXT NOT NULL,
  fingerprint TEXT NOT NULL,
  entities_json TEXT NOT NULL,
  relations_json TEXT NOT NULL,
  model TEXT NOT NULL,
  calls INTEGER NOT NULL DEFAULT 0,
  stats_json TEXT NOT NULL DEFAULT '{}',
  created_at INTEGER NOT NULL,
  PRIMARY KEY(kb_id, unit_id, fingerprint)
);

CREATE INDEX IF NOT EXISTS idx_graph_extractions_kb_fp
ON graph_extractions(kb_id, fingerprint);

-- Qualified facts of table / list units (structured extraction), cached by (kb, unit, fingerprint) like graph_extractions.
CREATE TABLE IF NOT EXISTS graph_facts (
  kb_id TEXT NOT NULL,
  unit_id TEXT NOT NULL,
  fingerprint TEXT NOT NULL,
  facts_json TEXT NOT NULL,
  model TEXT NOT NULL,
  calls INTEGER NOT NULL DEFAULT 0,
  stats_json TEXT NOT NULL DEFAULT '{}',
  created_at INTEGER NOT NULL,
  PRIMARY KEY(kb_id, unit_id, fingerprint)
);

CREATE INDEX IF NOT EXISTS idx_graph_facts_kb_fp
ON graph_facts(kb_id, fingerprint);
"""


class ClosingConnection(sqlite3.Connection):
    def __exit__(self, exc_type, exc, tb):
        # super().__exit__ commits; if that raises on a lock timeout, the original code never reached
        # close(), so the connection lingered until GC -- holding a read snapshot the whole time, which
        # kept the WAL from ever being checkpointed.
        try:
            return super().__exit__(exc_type, exc, tb)
        finally:
            self.close()


def connect(path: str | Path) -> sqlite3.Connection:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path, factory=ClosingConnection)
    _chmod_private(db_path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=30000")
    return con


def init_db(path: str | Path) -> None:
    from .discovery import init_schema as init_kb_sources_schema

    with connect(path) as con:
        con.executescript(SCHEMA)
        migrate_schema(con)
        init_kb_sources_schema(con)
    _chmod_private(Path(path))


def _chmod_private(path: Path) -> None:
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        try:
            candidate.chmod(0o600)
        except FileNotFoundError:
            pass


def now_ts() -> int:
    return int(time.time())


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def file_id_for(kb_id: str, file_key: int) -> str:
    return f"{kb_id}:{file_key}"


def metadata_fingerprint(file: SourceFile) -> str:
    return stable_json_hash(
        {
            "source_path": file.source_path,
            "rel_path": file.rel_path,
            "filename": file.filename,
            "dir": file.dir,
        }
    )


def begin_run(con: sqlite3.Connection, trigger_type: str, note: str | None = None) -> str:
    run_id = new_id("run")
    con.execute(
        """
        INSERT INTO ingest_runs(ingest_run_id, started_at, trigger_type, status, note)
        VALUES (?, ?, ?, 'running', ?)
        """,
        (run_id, now_ts(), trigger_type, note),
    )
    return run_id


def finish_run(con: sqlite3.Connection, run_id: str, status: str = "done", **counts: int) -> None:
    fields = ["finished_at = ?", "status = ?"]
    values: list[Any] = [now_ts(), status]
    for key, value in counts.items():
        fields.append(f"{key} = ?")
        values.append(value)
    values.append(run_id)
    con.execute(f"UPDATE ingest_runs SET {', '.join(fields)} WHERE ingest_run_id = ?", values)


def begin_graph_build(
    con: sqlite3.Connection,
    *,
    source_key: str,
    kb_id: str,
    source_collection: str,
    graph_version: str,
    allow_existing: bool = False,
    cache_fingerprint: str = "",
    build_kind: str = "full",
) -> str:
    existing = con.execute(
        """
        SELECT graph_build_id
        FROM graph_builds
        WHERE source_collection = ? AND graph_version = ?
        LIMIT 1
        """,
        (source_collection, graph_version),
    ).fetchone()
    if existing is not None:
        if not allow_existing:
            raise RuntimeError(
                "graph build record already exists for "
                f"{source_collection}:{graph_version}; rerun with "
                "--allow-existing-graph-version to resume this version, "
                "or choose a fresh --graph-version"
            )
        build_id = str(existing["graph_build_id"])
        # Resume: only reset the fields that belong to "this run". The corpus fingerprint, the chunk
        # ledger and the phase markers are assets left by the previous run; whether they can be reused
        # is decided by _resumable_phases in build.py.
        con.execute(
            """
            UPDATE graph_builds
            SET source_key = ?,
                kb_id = ?,
                status = 'running',
                started_at = ?,
                finished_at = NULL,
                manifest_json = '{}',
                error = NULL,
                cache_fingerprint = ?,
                build_kind = ?,
                worker_host = ?,
                worker_pid = ?,
                heartbeat_at = ?
            WHERE graph_build_id = ?
            """,
            (source_key, kb_id, now_ts(), cache_fingerprint or None, build_kind or "full",
             socket.gethostname(), os.getpid(), now_ts(), build_id),
        )
        return build_id

    build_id = new_id("graph")
    con.execute(
        """
        INSERT INTO graph_builds(
          graph_build_id, source_key, kb_id, source_collection, graph_version,
          status, started_at, worker_host, worker_pid, heartbeat_at, cache_fingerprint, build_kind
        )
        VALUES (?, ?, ?, ?, ?, 'running', ?, ?, ?, ?, ?, ?)
        """,
        (build_id, source_key, kb_id, source_collection, graph_version, now_ts(),
         socket.gethostname(), os.getpid(), now_ts(), cache_fingerprint or None, build_kind or "full"),
    )
    return build_id


def reconcile_stale_graph_builds(con: sqlite3.Connection, *, stale_after_seconds: int = 3600) -> list[str]:
    """Re-mark graph build records that are still 'running' although their process is dead as failed.

    When the graph build subprocess is SIGKILLed, taken down together with systemd, or loses power,
    nobody writes a terminal state: the console shows "building" forever, "build now" and "delete
    graph" are refused forever, and the midnight rebuild yields forever. Parse jobs heal themselves
    through leases; this gives graph builds the same ability: on the same host, liveness is judged by
    the pid; on another host (or for historical rows without a pid) by heartbeat timeout. Returns the
    ids of the reclaimed records."""
    hostname = socket.gethostname()
    now = now_ts()
    recovered: list[str] = []
    try:
        rows = con.execute(
            "SELECT graph_build_id, worker_host, worker_pid, heartbeat_at, started_at "
            "FROM graph_builds WHERE status = 'running'"
        ).fetchall()
    except sqlite3.OperationalError:
        return recovered  # migration has not run yet
    for row in rows:
        host = str(row["worker_host"] or "")
        pid = int(row["worker_pid"] or 0)
        last_seen = int(row["heartbeat_at"] or row["started_at"] or 0)
        if host == hostname and pid > 0:
            alive = True
            try:
                os.kill(pid, 0)
            except (OSError, ValueError):
                alive = False
            if alive:
                continue
            reason = f"Graph build process no longer exists (pid={pid}); marked failed by the reaper"
        else:
            if now - last_seen < max(60, stale_after_seconds):
                continue
            reason = f"Graph build heartbeat timed out ({now - last_seen}s); marked failed"
        con.execute(
            "UPDATE graph_builds SET status='failed', finished_at=?, error=? "
            "WHERE graph_build_id=? AND status='running'",
            (now, reason, str(row["graph_build_id"])),
        )
        recovered.append(str(row["graph_build_id"]))
    if recovered:
        con.commit()
    return recovered


def touch_graph_build(con: sqlite3.Connection, graph_build_id: str) -> None:
    """Graph build heartbeat: called together with set_graph_build_stage, commits immediately."""
    try:
        con.execute(
            "UPDATE graph_builds SET heartbeat_at = ? WHERE graph_build_id = ?",
            (now_ts(), graph_build_id),
        )
        con.commit()
    except sqlite3.OperationalError:
        pass


def finish_graph_build(
    con: sqlite3.Connection,
    graph_build_id: str,
    *,
    status: str,
    input_rows: int = 0,
    active_chunk_count: int = 0,
    active_doc_count: int = 0,
    source_content_hash: str | None = None,
    output_dir: str | None = None,
    manifest: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    con.execute(
        """
        UPDATE graph_builds
        SET status = ?,
            finished_at = ?,
            input_rows = ?,
            active_chunk_count = ?,
            active_doc_count = ?,
            source_content_hash = ?,
            output_dir = ?,
            manifest_json = ?,
            error = ?
        WHERE graph_build_id = ?
        """,
        (
            status,
            now_ts(),
            int(input_rows),
            int(active_chunk_count),
            int(active_doc_count),
            source_content_hash,
            output_dir,
            json.dumps(manifest or {}, ensure_ascii=False, sort_keys=True),
            error,
            graph_build_id,
        ),
    )


def update_graph_build_manifest(con: sqlite3.Connection, graph_build_id: str, manifest: dict[str, Any] | None) -> None:
    """Manifest only: results that exist only after the build was recorded as done (version GC, cache pruning)
    are added to it; the status and the finish time stay as they are."""
    con.execute("UPDATE graph_builds SET manifest_json = ? WHERE graph_build_id = ?",
                (json.dumps(manifest or {}, ensure_ascii=False, sort_keys=True), graph_build_id))


def replace_graph_build_chunks(
    con: sqlite3.Connection,
    graph_build_id: str,
    chunks: Iterable[dict[str, Any]],
) -> None:
    con.execute("DELETE FROM graph_build_chunks WHERE graph_build_id = ?", (graph_build_id,))
    con.executemany(
        """
        INSERT OR REPLACE INTO graph_build_chunks(
          graph_build_id, point_id, chunk_uid, doc_id, content_version,
          block_id, block_start, block_end, text_sha
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            (
                graph_build_id,
                str(chunk.get("point_id") or ""),
                str(chunk.get("chunk_uid") or ""),
                str(chunk.get("doc_id") or ""),
                str(chunk.get("content_version") or ""),
                chunk.get("block_id"),
                chunk.get("block_start"),
                chunk.get("block_end"),
                (str(chunk.get("text_sha")) if chunk.get("text_sha") else None),
            )
            for chunk in chunks
            if chunk.get("point_id")
        ),
    )


def graph_build_by_version(con: sqlite3.Connection, source_collection: str, graph_version: str) -> sqlite3.Row | None:
    return con.execute(
        "SELECT * FROM graph_builds WHERE source_collection = ? AND graph_version = ? LIMIT 1",
        (source_collection, graph_version),
    ).fetchone()


def record_graph_build_input(
    con: sqlite3.Connection,
    graph_build_id: str,
    *,
    input_rows: int,
    chunks: list[dict[str, Any]],
    source_content_hash: str,
) -> None:
    """Record the corpus size and fingerprint as soon as the corpus is prepared, without waiting for the
    end: a build stopped midway still has to know which corpus it was built from, and the resume check
    (did the corpus change?) relies on exactly this column."""
    con.execute(
        """
        UPDATE graph_builds
        SET input_rows = ?, active_chunk_count = ?, active_doc_count = ?, source_content_hash = ?
        WHERE graph_build_id = ?
        """,
        (int(input_rows), len(chunks), len({c["doc_id"] for c in chunks if c.get("doc_id")}),
         source_content_hash, graph_build_id),
    )


def graph_build_chunk_refs(con: sqlite3.Connection, graph_build_id: str) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in con.execute(
            "SELECT point_id, chunk_uid, doc_id, content_version, block_id, block_start, block_end "
            "FROM graph_build_chunks WHERE graph_build_id = ? ORDER BY doc_id, chunk_uid",
            (graph_build_id,),
        )
    ]


def graph_phases_done(con: sqlite3.Connection, graph_build_id: str) -> list[str]:
    return [
        str(row["phase"])
        for row in con.execute(
            "SELECT phase FROM graph_build_phases WHERE graph_build_id = ? ORDER BY done_at, rowid",
            (graph_build_id,),
        )
    ]


def mark_graph_phase_done(con: sqlite3.Connection, graph_build_id: str, phase: str) -> None:
    con.execute(
        "INSERT OR REPLACE INTO graph_build_phases(graph_build_id, phase, done_at) VALUES (?, ?, ?)",
        (graph_build_id, phase, now_ts()),
    )
    con.commit()


def clear_graph_phases(con: sqlite3.Connection, graph_build_id: str) -> None:
    con.execute("DELETE FROM graph_build_phases WHERE graph_build_id = ?", (graph_build_id,))


def graph_phase_timestamps(con: sqlite3.Connection, graph_build_id: str) -> list[tuple[str, int]]:
    """Completion time of each phase, in completion order: the progress bar needs it to weight phases by
    the actual durations of the last successful build."""
    return [(str(r["phase"]), int(r["done_at"])) for r in con.execute(
        "SELECT phase, done_at FROM graph_build_phases WHERE graph_build_id = ? ORDER BY done_at, rowid",
        (graph_build_id,))]


def record_graph_check(con: sqlite3.Connection, kb_id: str, decision: dict[str, Any]) -> None:
    """Record the latest check-rebuild decision for this knowledge base (health check D7). The table is
    created with SCHEMA; on an old database init_db may not have been re-run by the time this is first
    reached, so it is created here too."""
    con.execute(
        "CREATE TABLE IF NOT EXISTS graph_checks (kb_id TEXT PRIMARY KEY, checked_at INTEGER NOT NULL, "
        "decision_json TEXT NOT NULL)"
    )
    con.execute(
        "INSERT OR REPLACE INTO graph_checks(kb_id, checked_at, decision_json) VALUES (?, ?, ?)",
        (kb_id, now_ts(), json.dumps(decision, ensure_ascii=False)),
    )
    con.commit()


def latest_graph_check(con: sqlite3.Connection, kb_id: str) -> dict[str, Any] | None:
    try:
        row = con.execute("SELECT checked_at, decision_json FROM graph_checks WHERE kb_id = ?", (kb_id,)).fetchone()
    except sqlite3.OperationalError:      # no table yet: web process started before the first check-rebuild
        return None
    if row is None:
        return None
    try:
        decision = json.loads(row["decision_json"] or "{}")
    except (TypeError, ValueError):
        decision = {}
    return {"at": int(row["checked_at"] or 0), **decision}


def latest_successful_graph_build(
    con: sqlite3.Connection,
    source_key: str,
    *,
    kind: str | None = None,
    exclude_samples: bool = True,
) -> sqlite3.Row | None:
    """The most recent successful graph build. kind="full" only considers full-rebuild versions (the
    baseline of the rebuild policy); exclude_samples skips trial builds (those whose manifest carries a
    doc_filter): they never switch the alias and do not represent the current graph."""
    rows = con.execute(
        """
        SELECT *
        FROM graph_builds
        WHERE source_key = ? AND status = 'done'
        ORDER BY finished_at DESC, started_at DESC
        LIMIT 100
        """,
        (source_key,),
    ).fetchall()
    for row in rows:
        if kind and str(row["build_kind"] or "full") != kind:
            continue
        if exclude_samples and _is_sample_build(row):
            continue
        return row
    return None


def _is_sample_build(row: sqlite3.Row) -> bool:
    try:
        manifest = json.loads(row["manifest_json"] or "{}")
    except (TypeError, ValueError):
        return False
    return bool((manifest.get("input") or {}).get("doc_filter"))


def replace_graph_units(con: sqlite3.Connection, graph_build_id: str, units: Iterable[Any]) -> int:
    con.execute("DELETE FROM graph_units WHERE graph_build_id = ?", (graph_build_id,))
    rows = [
        (graph_build_id, str(u.unit_id), str(u.doc_id), " > ".join(u.section_path), int(u.n_tokens),
         len(u.chunk_uids), int(u.order))
        for u in units
    ]
    con.executemany(
        "INSERT OR REPLACE INTO graph_units(graph_build_id, unit_id, doc_id, section, n_tokens, chunk_count, unit_order) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    return len(rows)


def graph_extraction_unit_ids(con: sqlite3.Connection, kb_id: str, fingerprint: str, *,
                              skip_empty_malformed: bool = False) -> set[str]:
    """Units that already have an extraction result under this fingerprint. skip_empty_malformed: rows with
    no entities and no relations whose response contained malformed records do not count -- that is a
    protocol error recorded as a "zero-entity success" (final review F09) and must be re-extracted on the
    next graph build; a legitimate zero-entity result (no malformed records) counts as cached as usual."""
    sql = "SELECT unit_id FROM graph_extractions WHERE kb_id = ? AND fingerprint = ?"
    if skip_empty_malformed:
        sql += (" AND NOT (json_array_length(entities_json) = 0 AND json_array_length(relations_json) = 0"
                " AND COALESCE(json_extract(stats_json, '$.malformed'), 0) > 0)")
    rows = con.execute(sql, (kb_id, fingerprint)).fetchall()
    return {str(row["unit_id"]) for row in rows}


def save_graph_extraction(
    con: sqlite3.Connection,
    *,
    kb_id: str,
    unit_id: str,
    fingerprint: str,
    entities: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    model: str,
    calls: int,
    stats: dict[str, Any] | None = None,
) -> None:
    con.execute(
        """
        INSERT OR REPLACE INTO graph_extractions(
          kb_id, unit_id, fingerprint, entities_json, relations_json, model, calls, stats_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            kb_id, unit_id, fingerprint,
            json.dumps(entities, ensure_ascii=False, separators=(",", ":")),
            json.dumps(relations, ensure_ascii=False, separators=(",", ":")),
            model, int(calls), json.dumps(stats or {}, ensure_ascii=False, separators=(",", ":")), now_ts(),
        ),
    )
    con.commit()


def load_graph_extractions(
    con: sqlite3.Connection, kb_id: str, fingerprint: str, unit_ids: Iterable[str],
) -> dict[str, dict[str, Any]]:
    wanted = set(str(u) for u in unit_ids)
    out: dict[str, dict[str, Any]] = {}
    for row in con.execute(
        "SELECT unit_id, entities_json, relations_json, model, calls, stats_json FROM graph_extractions "
        "WHERE kb_id = ? AND fingerprint = ?",
        (kb_id, fingerprint),
    ):
        unit_id = str(row["unit_id"])
        if unit_id not in wanted:
            continue
        try:
            stats = json.loads(row["stats_json"] or "{}")
        except (TypeError, ValueError):
            stats = {}
        out[unit_id] = {
            "entities": json.loads(row["entities_json"] or "[]"),
            "relations": json.loads(row["relations_json"] or "[]"),
            "model": str(row["model"]), "calls": int(row["calls"] or 0),
            "stats": stats if isinstance(stats, dict) else {},
        }
    return out


def graph_extraction_entity_total(con: sqlite3.Connection, kb_id: str, fingerprint: str, unit_ids: Iterable[str]) -> int:
    """How many entity records this batch of units produced in total (used to detect an empty graph)."""
    total = 0
    for unit_id, row in load_graph_extractions(con, kb_id, fingerprint, unit_ids).items():
        total += len(row.get("entities") or [])
    return total


def graph_fact_unit_ids(con: sqlite3.Connection, kb_id: str, fingerprint: str) -> set[str]:
    rows = con.execute("SELECT unit_id FROM graph_facts WHERE kb_id = ? AND fingerprint = ?", (kb_id, fingerprint)).fetchall()
    return {str(row["unit_id"]) for row in rows}


def save_graph_facts(con: sqlite3.Connection, *, kb_id: str, unit_id: str, fingerprint: str,
                     facts: list[dict[str, Any]], model: str, calls: int, stats: dict[str, Any] | None = None) -> None:
    con.execute(
        "INSERT OR REPLACE INTO graph_facts(kb_id, unit_id, fingerprint, facts_json, model, calls, stats_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (kb_id, unit_id, fingerprint, json.dumps(facts, ensure_ascii=False, separators=(",", ":")), model, int(calls),
         json.dumps(stats or {}, ensure_ascii=False, separators=(",", ":")), now_ts()),
    )
    con.commit()


def load_graph_facts(con: sqlite3.Connection, kb_id: str, fingerprint: str, unit_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    wanted = set(str(u) for u in unit_ids)
    out: dict[str, dict[str, Any]] = {}
    for row in con.execute("SELECT unit_id, facts_json, model, calls, stats_json FROM graph_facts WHERE kb_id = ? AND fingerprint = ?",
                           (kb_id, fingerprint)):
        unit_id = str(row["unit_id"])
        if unit_id not in wanted:
            continue
        try:
            stats = json.loads(row["stats_json"] or "{}")
        except (TypeError, ValueError):
            stats = {}
        out[unit_id] = {"facts": json.loads(row["facts_json"] or "[]"), "model": str(row["model"]),
                        "calls": int(row["calls"] or 0), "stats": stats if isinstance(stats, dict) else {}}
    return out


def graph_fact_count(con: sqlite3.Connection, kb_id: str) -> int:
    row = con.execute("SELECT COUNT(*) AS n FROM graph_facts WHERE kb_id = ?", (kb_id,)).fetchone()
    return int(row["n"] if row else 0)


def delete_graph_facts(con: sqlite3.Connection, kb_id: str) -> int:
    cur = con.execute("DELETE FROM graph_facts WHERE kb_id = ?", (kb_id,))
    return int(cur.rowcount or 0)


def graph_extraction_count(con: sqlite3.Connection, kb_id: str) -> int:
    row = con.execute("SELECT COUNT(*) AS n FROM graph_extractions WHERE kb_id = ?", (kb_id,)).fetchone()
    return int(row["n"] if row else 0)


def delete_graph_extractions(con: sqlite3.Connection, kb_id: str) -> int:
    cur = con.execute("DELETE FROM graph_extractions WHERE kb_id = ?", (kb_id,))
    return int(cur.rowcount or 0)


def prune_graph_builds(con: sqlite3.Connection, kb_id: str, *, keep_versions: Iterable[str] = ()) -> int:
    """Keep only the useful graph build records: the running one, the latest successful one, and every row
    started after the latest full-rebuild version (the rebuild policy counts appends from them, and the
    status card's "N appends since the last full rebuild" relies on them too). Older rows are deleted
    together with their chunk ledger, phase markers and unit table -- once artifacts are only kept from
    the current version onward, hoarding the records makes no sense. Returns the number of deleted
    records. ``keep_versions``: versions whose artifacts are still kept (the active one and those inside the
    keep window); their records stay too, so the status card still has figures after a rollback to them."""
    kept_versions = {str(v) for v in keep_versions if str(v)}
    rows = con.execute(
        "SELECT graph_build_id, graph_version, status, build_kind, started_at, finished_at FROM graph_builds WHERE kb_id = ?",
        (kb_id,)).fetchall()
    ts = lambda r: int(r["finished_at"] or r["started_at"] or 0)
    done = [r for r in rows if str(r["status"]) == "done"]
    latest_done = max(done, key=ts, default=None)
    last_full = max((r for r in done if str(r["build_kind"] or "full") == "full"), key=ts, default=None)
    floor_ts = int(last_full["started_at"] or 0) if last_full is not None else 0
    keep = {str(r["graph_build_id"]) for r in rows
            if str(r["status"]) == "running" or int(r["started_at"] or 0) >= floor_ts
            or str(r["graph_version"] or "") in kept_versions}
    if latest_done is not None:
        keep.add(str(latest_done["graph_build_id"]))
    stale = [str(r["graph_build_id"]) for r in rows if str(r["graph_build_id"]) not in keep]
    for i in range(0, len(stale), 200):
        batch = stale[i:i + 200]
        marks = ",".join("?" * len(batch))
        for table in ("graph_build_chunks", "graph_build_phases", "graph_units"):
            con.execute(f"DELETE FROM {table} WHERE graph_build_id IN ({marks})", batch)
        con.execute(f"DELETE FROM graph_builds WHERE graph_build_id IN ({marks})", batch)
    return len(stale)


def unsuccessful_graph_versions(con: sqlite3.Connection, kb_id: str, *, superseded_only: bool = False) -> set[str]:
    """Version numbers left behind by paused / failed builds (cancelled / failed graph build records; a
    version later resumed to success does not count). When old versions are cleaned up they do not take
    a slot among the "latest N versions" and are deleted outright once a newer successful version exists
    -- otherwise a half-finished version would push out the previous good graph. superseded_only=True
    only counts versions started before the latest successful build: the timer cleanup takes this path
    so the most recently paused version is kept for resume; the cleanup at the end of a build does not
    need it, since that version is about to become the successful one."""
    rows = con.execute("SELECT graph_version, status, started_at FROM graph_builds WHERE kb_id = ?", (kb_id,)).fetchall()
    alive = {str(r["graph_version"]) for r in rows if str(r["status"]) in ("done", "running")}
    floor: int | None = None
    if superseded_only:
        done = [int(r["started_at"] or 0) for r in rows if str(r["status"]) == "done"]
        if not done:
            return set()
        floor = max(done)
    out: set[str] = set()
    for r in rows:
        version = str(r["graph_version"] or "")
        status = str(r["status"])
        if status not in ("cancelled", "failed", "rolled_back") or not version or version in alive:
            continue
        # A rolled-back version was rejected on purpose: never kept for a resume, discarded at the next GC
        if status != "rolled_back" and floor is not None and int(r["started_at"] or 0) >= floor:
            continue
        out.add(version)
    return out


def resumable_graph_versions(con: sqlite3.Connection, kb_id: str) -> set[str]:
    """The version kept for a resume: when the newest build record of the base (by start time) is cancelled or
    failed, or still marked running after its process died, its version id. "Continue build" only resumes that
    newest record (kb_server.service.trigger_graph_build); no entry point resumes an older half-finished
    version. The scheduled GC neither deletes it nor lets it take a slot among the "latest N versions": once it
    took a slot, KEEP=1 deleted the resume artifacts the same night and KEEP=2 pushed out the previous good
    graph (2026-09-29 audit). A rolled-back version does not count."""
    row = con.execute(
        "SELECT graph_version, status FROM graph_builds WHERE kb_id = ? ORDER BY started_at DESC, rowid DESC LIMIT 1",
        (kb_id,)).fetchone()
    if row is None or str(row["status"]) not in ("cancelled", "failed", "running"):
        return set()
    version = str(row["graph_version"] or "")
    return {version} if version else set()


def graph_extraction_flag_counts(con: sqlite3.Connection, kb_id: str, fingerprint: str, *, flags: Iterable[str] = ("truncated", "partial"),
                                 table: str = "graph_extractions") -> dict[str, int]:
    """How many units in the current cached assets carry each flag in stats (truncated / partial > 0) --
    reported separately from the counts of "new calls in this round": a round served entirely from cache
    has truncated_units = 0, yet the truncated units are still in the assets (Codex 2026-09-13 F05)."""
    assert table in ("graph_extractions", "graph_facts")
    out: dict[str, int] = {}
    for flag in flags:
        row = con.execute(
            f"SELECT COUNT(*) FROM {table} WHERE kb_id = ? AND fingerprint = ? "
            f"AND CAST(json_extract(stats_json, '$.{flag}') AS INTEGER) > 0", (kb_id, fingerprint)).fetchone()
        out[str(flag)] = int(row[0] or 0) if row else 0
    return out


def prune_graph_extractions(con: sqlite3.Connection, kb_id: str, keep_unit_ids: Iterable[str]) -> int:
    """Drop extraction cache rows that no longer correspond to any existing unit (deleting a document or
    changing its text changes the unit_id, so old rows only take up space). Rows of existing units are
    kept regardless of fingerprint: switching the configuration and back still hits the cache. Called
    after a full build completes, and only for whole-corpus builds (not trial builds with doc_ids),
    otherwise cache outside the trial scope would be deleted by mistake."""
    keep = {str(u) for u in keep_unit_ids}
    stale = [str(r["unit_id"]) for r in con.execute(
        "SELECT DISTINCT unit_id FROM graph_extractions WHERE kb_id = ?", (kb_id,)) if str(r["unit_id"]) not in keep]
    removed = 0
    for i in range(0, len(stale), 500):
        batch = stale[i:i + 500]
        cur = con.execute(
            f"DELETE FROM graph_extractions WHERE kb_id = ? AND unit_id IN ({','.join('?' * len(batch))})",
            (kb_id, *batch))
        removed += int(cur.rowcount or 0)
    return removed


def graph_build_point_ids(con: sqlite3.Connection, graph_build_id: str) -> set[str]:
    rows = con.execute(
        "SELECT point_id FROM graph_build_chunks WHERE graph_build_id = ?",
        (graph_build_id,),
    ).fetchall()
    return {str(row["point_id"]) for row in rows}


def graph_build_doc_chunks(con: sqlite3.Connection, graph_build_id: str) -> dict[str, set[tuple[str, str, str]]]:
    """The chunk ledger frozen by one graph build, grouped by document: {doc_id: {(point_id,
    content_version, text_sha)}}. Incremental append uses it to compute document-level differences;
    text_sha is the text fingerprint, recorded as an empty string for old ledgers that lack it (treated
    as unknown when comparing)."""
    out: dict[str, set[tuple[str, str, str]]] = {}
    for row in con.execute(
            "SELECT doc_id, point_id, content_version, text_sha FROM graph_build_chunks WHERE graph_build_id = ?",
            (graph_build_id,)):
        out.setdefault(str(row["doc_id"]), set()).add(
            (str(row["point_id"]), str(row["content_version"] or ""), str(row["text_sha"] or "")))
    return out


def count_graph_builds_since(con: sqlite3.Connection, kb_id: str, *, kind: str, since_ts: int) -> int:
    """How many builds of a given kind this knowledge base completed after a point in time (the console
    shows "N appends since the last full rebuild")."""
    row = con.execute(
        "SELECT COUNT(*) FROM graph_builds WHERE kb_id = ? AND status = 'done' AND build_kind = ? "
        "AND COALESCE(finished_at, started_at) > ?",
        (kb_id, kind, int(since_ts)),
    ).fetchone()
    return int(row[0] if row else 0)


def get_file(con: sqlite3.Connection, kb_id: str, file_key: int) -> sqlite3.Row | None:
    return con.execute(
        "SELECT * FROM files WHERE kb_id = ? AND file_key = ?",
        (kb_id, file_key),
    ).fetchone()


def get_file_by_path(con: sqlite3.Connection, kb_id: str, source_path: str) -> sqlite3.Row | None:
    return con.execute(
        """
        SELECT * FROM files
        WHERE kb_id = ? AND source_path = ?
        ORDER BY CASE WHEN status = 'deleted' THEN 1 ELSE 0 END, last_seen_at DESC
        LIMIT 1
        """,
        (kb_id, source_path),
    ).fetchone()


def prune_job_history(con: sqlite3.Connection, *, retention_days: int = 30) -> dict[str, int]:
    """Clean up long-finished jobs and failure records. These rows used to be deleted only when a knowledge
    base or file was deleted: failure records carry a 4000-character stack trace, every full re-parse
    adds N rows, the state database only ever grew, and every console poll runs statistics over this
    table."""
    cutoff = now_ts() - max(1, retention_days) * 86400
    removed: dict[str, int] = {}
    cur = con.execute(
        "DELETE FROM failures WHERE resolved_at IS NOT NULL AND resolved_at < ?", (cutoff,))
    removed["failures"] = int(cur.rowcount or 0)
    cur = con.execute(
        "DELETE FROM jobs WHERE status IN ('done','cancelled') AND COALESCE(finished_at, updated_at) < ?",
        (cutoff,))
    removed["jobs"] = int(cur.rowcount or 0)
    cur = con.execute("DELETE FROM job_events WHERE job_id NOT IN (SELECT job_id FROM jobs)")
    removed["job_events"] = int(cur.rowcount or 0)
    try:
        cur = con.execute("DELETE FROM ingest_runs WHERE COALESCE(finished_at, started_at) < ?", (cutoff,))
        removed["ingest_runs"] = int(cur.rowcount or 0)
    except sqlite3.OperationalError:
        removed["ingest_runs"] = 0
    con.commit()
    return removed


def touch_file_seen(con: sqlite3.Connection, file_id: str) -> None:
    """Only refresh last_seen_at. With the scan running every minute, rewriting the whole row for unchanged
    files is pure write amplification: the WAL grows and the write lock is held longer, and that lock
    spans the entire scan round."""
    con.execute("UPDATE files SET last_seen_at = ? WHERE file_id = ?", (now_ts(), file_id))


def upsert_file(con: sqlite3.Connection, file: SourceFile, status: str = "seen") -> str:
    ts = now_ts()
    fid = file_id_for(file.kb_id, file.file_key)
    values = {
        "file_id": fid,
        "kb_id": file.kb_id,
        "collection": file.collection,
        "source_root": file.source_root,
        "source_type": file.source_type,
        "file_key": file.file_key,
        "source_path": file.source_path,
        "rel_path": file.rel_path,
        "filename": file.filename,
        "dir": file.dir,
        "physical_path": file.physical_path,
        "mime_type": file.mime_type,
        "size": file.size,
        "mtime": file.mtime,
        "checksum": file.checksum,
        "content_version": file.content_version,
        "metadata_fingerprint": metadata_fingerprint(file),
        "last_seen_at": ts,
        "status": status,
    }
    con.execute(
        """
        INSERT INTO files (
          file_id, kb_id, collection, source_root, source_type,
          file_key, source_path, rel_path, filename, dir,
          physical_path, mime_type, size, mtime, checksum, content_version,
          metadata_fingerprint, first_seen_at, last_seen_at, status
        )
        VALUES (
          :file_id, :kb_id, :collection, :source_root, :source_type,
          :file_key, :source_path, :rel_path, :filename, :dir,
          :physical_path, :mime_type, :size, :mtime, :checksum, :content_version,
          :metadata_fingerprint, :last_seen_at, :last_seen_at, :status
        )
        ON CONFLICT(file_id) DO UPDATE SET
          collection = excluded.collection,
          source_root = excluded.source_root,
          source_type = excluded.source_type,
          source_path = excluded.source_path,
          rel_path = excluded.rel_path,
          filename = excluded.filename,
          dir = excluded.dir,
          physical_path = excluded.physical_path,
          mime_type = excluded.mime_type,
          size = excluded.size,
          mtime = excluded.mtime,
          checksum = excluded.checksum,
          content_version = excluded.content_version,
          metadata_fingerprint = excluded.metadata_fingerprint,
          last_seen_at = excluded.last_seen_at,
          status = excluded.status
        """,
        values,
    )
    return fid


def enqueue_job(
    con: sqlite3.Connection,
    *,
    ingest_run_id: str | None,
    file_id: str | None,
    kb_id: str,
    collection: str,
    file_key: int | None,
    job_type: str,
    parser_profile: str | None = None,
    priority: int = 100,
    payload: dict[str, Any] | None = None,
    dedupe_key: str | None = None,
) -> str:
    job_id = new_id("job")
    ts = now_ts()
    try:
        con.execute(
        """
        INSERT INTO jobs (
          job_id, ingest_run_id, file_id, kb_id, collection, file_key,
          job_type, parser_profile, status, priority, next_attempt_at,
          payload_json, dedupe_key, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, 0, ?, ?, ?, ?)
        """,
        (
            job_id,
            ingest_run_id,
            file_id,
            kb_id,
            collection,
            file_key,
            job_type,
            parser_profile,
            priority,
            json.dumps(payload or {}, ensure_ascii=False),
            dedupe_key,
            ts,
            ts,
        ),
        )
    except sqlite3.IntegrityError:
        if not dedupe_key:
            raise
        existing = con.execute(
            """
            SELECT job_id FROM jobs
            WHERE dedupe_key = ? AND status IN ('queued', 'retry', 'running')
            ORDER BY created_at ASC
            LIMIT 1
            """,
            (dedupe_key,),
        ).fetchone()
        if existing:
            # When a high-priority "parse now" job collides with an already queued lower-priority job of the
            # same key, the old job used to be returned as is: the user clicked "now" yet the job kept its
            # original place in the queue.
            con.execute(
                "UPDATE jobs SET priority = MIN(priority, ?), updated_at = ? WHERE job_id = ?",
                (priority, now_ts(), str(existing["job_id"])),
            )
            return str(existing["job_id"])
        raise
    return job_id


def resolve_failures_for_dedupe_key(con: sqlite3.Connection, dedupe_key: str) -> int:
    """Mark the open failure records of every past attempt under the same dedupe key as resolved. When a
    compensation job exhausts its retries the scan re-queues it under a new job_id, so resolving only
    the current job's failure would leave the earliest one hanging in the UI forever."""
    if not dedupe_key:
        return 0
    cur = con.execute(
        "UPDATE failures SET resolved_at = COALESCE(resolved_at, ?) "
        "WHERE job_id IN (SELECT job_id FROM jobs WHERE dedupe_key = ?)",
        (now_ts(), dedupe_key),
    )
    return int(cur.rowcount or 0)


def fts_sync_dedupe_key(file_id: str, content_version: str) -> str:
    """Dedupe key of the keyword-index compensation job (fts_sync): only one per file and version is queued;
    when the scan recovers failed jobs it recomputes the key with the same formula and only re-queues on
    a match (once the file has a new version, that version's parse syncs the index itself)."""
    return f"fts_sync:{file_id}:{content_version}"


def failed_job_for_dedupe_key(con: sqlite3.Connection, dedupe_key: str) -> sqlite3.Row | None:
    return con.execute(
        """
        SELECT * FROM jobs
        WHERE dedupe_key = ? AND status = 'failed'
        ORDER BY finished_at DESC, updated_at DESC, created_at DESC
        LIMIT 1
        """,
        (dedupe_key,),
    ).fetchone()


def active_job_for_dedupe_key(con: sqlite3.Connection, dedupe_key: str) -> sqlite3.Row | None:
    return con.execute(
        """
        SELECT * FROM jobs
        WHERE dedupe_key = ? AND status IN ('queued', 'retry', 'running')
        LIMIT 1
        """,
        (dedupe_key,),
    ).fetchone()


def latest_job_for_dedupe_key(con: sqlite3.Connection, dedupe_key: str) -> sqlite3.Row | None:
    return con.execute(
        """
        SELECT * FROM jobs
        WHERE dedupe_key = ?
        ORDER BY created_at DESC, updated_at DESC, rowid DESC
        LIMIT 1
        """,
        (dedupe_key,),
    ).fetchone()


_JOBS_MIGRATIONS = (
    "ALTER TABLE jobs ADD COLUMN stage TEXT",
    # Cooperative cancellation: set to 1 when a knowledge base is closed / deleted; the parse job exits on
    # its own at the next phase boundary. A flag is used instead of changing status directly because a
    # running row belongs to the worker -- changing status from outside would race with the worker's
    # terminal-state write, and the job would look cancelled while still writing to the vector store.
    "ALTER TABLE jobs ADD COLUMN cancel_requested INTEGER",
    # The block-span table was a product of the old pipeline (sliding-window text-unit provenance);
    # once extraction units come straight from chunks it has no consumer, so the table goes.
    "DROP TABLE IF EXISTS block_spans",
    "ALTER TABLE graph_builds ADD COLUMN stage TEXT",
    # API protocol: openai = /chat/completions compatible; anthropic = /v1/messages
    "ALTER TABLE llm_registry ADD COLUMN protocol TEXT NOT NULL DEFAULT 'openai'",
    # Graph build process identity: after a kill / power loss it tells stale running rows apart for reclaiming
    "ALTER TABLE graph_builds ADD COLUMN worker_host TEXT",
    "ALTER TABLE graph_builds ADD COLUMN worker_pid INTEGER",
    "ALTER TABLE graph_builds ADD COLUMN heartbeat_at INTEGER",
    # Fingerprint, taken at build start, of the configuration items that enter the LLM cache key. The
    # console uses it to decide whether a paused build can still be offered as "resume graph build" --
    # see graph_cache_fingerprint in graph/build.py. Old records are NULL and are never promised a resume.
    "ALTER TABLE graph_builds ADD COLUMN cache_fingerprint TEXT",
    # Hot-path indexes: chunks.file_id used to be unindexed, so every parse job did a full-table scan
    # inside its write transaction (row count includes historical inactive versions); the per-KB job
    # statistics are polled by the console every 2.5 seconds; files (kb_id, source_path) decides whether
    # the first ingest of a new knowledge base is O(N^2).
    "CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_collection_status ON chunks(collection, status)",
    "CREATE INDEX IF NOT EXISTS idx_jobs_kb_type_status ON jobs(kb_id, job_type, status)",
    "CREATE INDEX IF NOT EXISTS idx_jobs_dedupe ON jobs(dedupe_key)",
    "CREATE INDEX IF NOT EXISTS idx_files_kb_path ON files(kb_id, source_path)",
    "CREATE INDEX IF NOT EXISTS idx_failures_job ON failures(job_id)",
    # idx_files_nc (file_key alone) serves no query: every query carries kb_id, which UNIQUE(kb_id,
    # file_key) already covers. The "nc" in the name is also a leftover from the Nextcloud era.
    "DROP INDEX IF EXISTS idx_files_nc",
    # Chunking acceptance verdict (chunking/diagnose.py), written with indexed_version on a successful parse
    "ALTER TABLE files ADD COLUMN chunk_diag_json TEXT",
    # The old pipeline froze block spans into the chunk ledger; the columns stay (SQLite cannot easily drop
    # columns) but the new pipeline no longer writes them.
    "ALTER TABLE graph_build_chunks ADD COLUMN block_id TEXT",
    "ALTER TABLE graph_build_chunks ADD COLUMN block_start INTEGER",
    "ALTER TABLE graph_build_chunks ADD COLUMN block_end INTEGER",
    # Build kind: full = full rebuild; append = incremental append (only extracts new units, replays the
    # resolution decisions, reuses vectors). The rebuild policy only takes full builds as its baseline --
    # otherwise a single append would reset the "new content" counter and a full rebuild would never come.
    "ALTER TABLE graph_builds ADD COLUMN build_kind TEXT NOT NULL DEFAULT 'full'",
    # Chunk text fingerprint (first 16 hex digits of sha256): after a re-parse where the uid stayed the same
    # but the text changed (block ids are stable when the profile is not upgraded), incremental append must
    # still see a "document change". Old rows are NULL and are skipped as "unknown" when comparing, so
    # there are no false positives.
    "ALTER TABLE chunks ADD COLUMN text_sha TEXT",
    "ALTER TABLE graph_build_chunks ADD COLUMN text_sha TEXT",
)


def migrate_schema(con: sqlite3.Connection) -> None:
    for statement in _JOBS_MIGRATIONS:
        try:
            con.execute(statement)
        except sqlite3.OperationalError as exc:
            message = str(exc).lower()
            # Only swallow the "already exists" family; real faults such as lock timeouts must surface,
            # otherwise the migration silently never ran and things fail later, in a harder-to-read way,
            # when the new columns are used.
            if "duplicate column" in message or "already exists" in message:
                continue
            raise


def set_job_stage(con: sqlite3.Connection, job_id: str, stage: str) -> None:
    """Progress checkpoint for the web console. Commits immediately so the
    poller sees it mid-parse; safe because parse jobs accumulate no other
    uncommitted writes until the final replace_chunks/mark_* block.

    A timeline event is only recorded when the "head" of the stage name (the part before the parenthesis)
    changes: VLM progress rewrites "image description (VLM 3/7)" every 1.5 seconds, and recording each
    one would leave a long job with tens of thousands of rows. A progress refresh within the same stage
    rewrites the text of the latest stage event: previously only jobs.stage was updated and the timeline
    row stayed at the first-written "VLM 1/166" forever (found by the user on 2026-09-10); now it shows
    the current progress while running and 166/166 at the end."""
    last = con.execute(
        "SELECT rowid, text FROM job_events WHERE job_id = ? AND kind = 'stage' ORDER BY ts DESC, rowid DESC LIMIT 1",
        (job_id,),
    ).fetchone()
    if last is not None and _stage_head(last["text"]) == _stage_head(stage):
        if str(last["text"]) != stage:
            con.execute("UPDATE job_events SET text = ? WHERE rowid = ?", (stage, last["rowid"]))
    else:
        con.execute("INSERT INTO job_events(job_id, ts, kind, text) VALUES (?, ?, 'stage', ?)",
                    (job_id, now_ts(), stage))
    con.execute("UPDATE jobs SET stage = ?, updated_at = ? WHERE job_id = ?", (stage, now_ts(), job_id))
    con.commit()


def _stage_head(stage: str | None) -> str:
    return str(stage or "").split("(", 1)[0].strip()


def add_job_event(con: sqlite3.Connection, job_id: str, kind: str, text: str) -> None:
    """One timeline event (info / error / retry / done / cancelled). Commits immediately, like
    set_job_stage and for the same reason."""
    con.execute("INSERT INTO job_events(job_id, ts, kind, text) VALUES (?, ?, ?, ?)",
                (job_id, now_ts(), kind, str(text)[:4000]))
    con.commit()


def job_events(con: sqlite3.Connection, job_id: str, *, limit: int = 500) -> list[dict[str, Any]]:
    rows = con.execute(
        "SELECT ts, kind, text FROM job_events WHERE job_id = ? ORDER BY ts, rowid LIMIT ?",
        (job_id, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def set_graph_build_stage(con: sqlite3.Connection, graph_build_id: str, stage: str) -> None:
    """Graph build stage checkpoint, same semantics as set_job_stage: commits immediately so the polling
    console sees it at any time. Also refreshes the heartbeat that reconcile_stale_graph_builds uses to
    judge liveness."""
    try:
        con.execute(
            "UPDATE graph_builds SET stage = ?, heartbeat_at = ? WHERE graph_build_id = ?",
            (stage, now_ts(), graph_build_id),
        )
    except sqlite3.OperationalError:  # old database where the migration has not run yet
        con.execute("UPDATE graph_builds SET stage = ? WHERE graph_build_id = ?", (stage, graph_build_id))
    con.commit()


# ── LLM registry / app config (web console) ─────────────────────────────

def list_llms(con: sqlite3.Connection) -> list[sqlite3.Row]:
    return con.execute("SELECT * FROM llm_registry ORDER BY builtin DESC, name").fetchall()


def get_llm(con: sqlite3.Connection, name: str) -> sqlite3.Row | None:
    return con.execute("SELECT * FROM llm_registry WHERE name = ?", (name,)).fetchone()


def upsert_llm(
    con: sqlite3.Connection,
    *,
    name: str,
    base_url: str,
    api_key: str,
    model_id: str,
    protocol: str = "openai",
    notes: str = "",
    builtin: bool = False,
) -> None:
    ts = now_ts()
    con.execute(
        """
        INSERT INTO llm_registry(name, base_url, api_key, model_id, protocol, notes, builtin, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
          base_url = excluded.base_url,
          api_key = CASE WHEN excluded.api_key != '' THEN excluded.api_key ELSE llm_registry.api_key END,
          model_id = excluded.model_id,
          protocol = excluded.protocol,
          notes = excluded.notes,
          updated_at = excluded.updated_at
        """,
        (name, base_url, api_key, model_id, protocol, notes, 1 if builtin else 0, ts, ts),
    )


def delete_llm(con: sqlite3.Connection, name: str) -> bool:
    row = con.execute("SELECT builtin FROM llm_registry WHERE name = ?", (name,)).fetchone()
    if row is None:
        return False
    if int(row["builtin"]):
        raise ValueError(f"Built-in model “{name}” cannot be deleted")
    con.execute("DELETE FROM llm_registry WHERE name = ?", (name,))
    return True


def get_app_config(con: sqlite3.Connection, key: str, default: Any = None) -> Any:
    row = con.execute("SELECT value FROM app_config WHERE key = ?", (key,)).fetchone()
    if row is None:
        return default
    try:
        return json.loads(row["value"])
    except Exception:
        return default


def set_app_config(con: sqlite3.Connection, key: str, value: Any) -> None:
    con.execute(
        "INSERT INTO app_config(key, value, updated_at) VALUES (?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
        (key, json.dumps(value, ensure_ascii=False), now_ts()),
    )


def metadata_fingerprint_from_row(row: sqlite3.Row) -> str:
    """The fingerprint of a files row, matching metadata_fingerprint(file)."""
    return stable_json_hash(
        {
            "source_path": str(row["source_path"]),
            "rel_path": str(row["rel_path"]),
            "filename": str(row["filename"]),
            "dir": str(row["dir"]),
        }
    )


def claim_job_for_types(
    con: sqlite3.Connection,
    worker_id: str,
    lease_seconds: int = 3600,
    allowed_job_types: set[str] | None = None,
    allowed_kb_ids: set[str] | None = None,
) -> sqlite3.Row | None:
    ts = now_ts()
    until = ts + lease_seconds
    if con.in_transaction:
        con.commit()
    con.execute("BEGIN IMMEDIATE")
    type_clause = ""
    kb_clause = ""
    params: list[Any] = [ts, ts]
    if allowed_job_types is not None:
        if not allowed_job_types:
            con.commit()
            return None
        placeholders = ",".join(["?"] * len(allowed_job_types))
        type_clause = f" AND job_type IN ({placeholders})"
        params.extend(sorted(allowed_job_types))
    if allowed_kb_ids is not None:
        if not allowed_kb_ids:
            con.commit()
            return None
        placeholders = ",".join(["?"] * len(allowed_kb_ids))
        kb_clause = f" AND kb_id IN ({placeholders})"
        params.extend(sorted(allowed_kb_ids))
    row = con.execute(
        f"""
        SELECT * FROM jobs
        WHERE
          (status = 'queued' OR status = 'retry' OR (status = 'running' AND locked_until < ?))
          AND next_attempt_at <= ?
          {type_clause}
          {kb_clause}
        ORDER BY priority ASC, created_at ASC
        LIMIT 1
        """,
        params,
    ).fetchone()
    if not row:
        con.commit()
        return None
    con.execute(
        """
        UPDATE jobs
        SET status = 'running', locked_by = ?, locked_until = ?, started_at = COALESCE(started_at, ?),
            updated_at = ?, stage = NULL
        WHERE job_id = ?
        """,
        (worker_id, until, ts, ts, row["job_id"]),
    )
    con.commit()
    return con.execute("SELECT * FROM jobs WHERE job_id = ?", (row["job_id"],)).fetchone()


def mark_job_done(con: sqlite3.Connection, job_id: str, worker_id: str | None = None) -> None:
    ts = now_ts()
    # Only write the terminal state while the job is still running and (when worker_id is given) the lock is
    # still ours: a job taken over by someone else after the lease expired, or cancelled meanwhile (KB
    # deleted), must not be marked done by this worker that is already out (Codex review N07: the new owner
    # after a takeover is also running, so checking the status alone is not enough).
    cur = con.execute(
        "UPDATE jobs SET status = 'done', error = NULL, finished_at = ?, updated_at = ?, "
        "locked_by = NULL, locked_until = NULL WHERE job_id = ? AND status = 'running' AND (? IS NULL OR locked_by = ?)",
        (ts, ts, job_id, worker_id, worker_id),
    )
    if cur.rowcount:
        con.execute("INSERT INTO job_events(job_id, ts, kind, text) VALUES (?, ?, 'done', 'Done')", (job_id, ts))
    con.execute(
        "UPDATE failures SET resolved_at = COALESCE(resolved_at, ?) WHERE job_id = ?",
        (ts, job_id),
    )


def release_job(con: sqlite3.Connection, job_id: str) -> None:
    """Hand a claimed job straight back to the queue without counting an
    attempt. Only for clean releases (dry-run); an orphan left behind by a dead
    worker must go through `release_crashed_job` so a poison document cannot
    loop forever."""
    con.execute(
        "UPDATE jobs SET status = 'queued', locked_by = NULL, locked_until = NULL, updated_at = ? WHERE job_id = ? AND status = 'running'",
        (now_ts(), job_id),
    )


def release_crashed_job(
    con: sqlite3.Connection,
    job_id: str,
    *,
    max_retries: int,
    retry_delay_seconds: int,
    reason: str = "worker process died before finishing this job",
    expected_owner: str | None = None,
) -> str:
    """Recover a job whose worker vanished (OOM kill, segfault, power loss).
    expected_owner: the lock holder verified just before reclaiming; if the lock changed hands meanwhile
    (another worker took over) nothing is written and 'skipped' is returned (Codex review N07).

    Unlike a normal release this counts as an attempt and applies the usual
    backoff, so a document that reliably kills the interpreter is retried a
    bounded number of times and then parked as failed instead of being re-served
    first on every run and starving the rest of the queue. Returns 'retry' or
    'failed'."""
    ts = now_ts()
    row = con.execute(
        "SELECT retry_count, file_id, cancel_requested, locked_by FROM jobs "
        "WHERE job_id = ? AND status = 'running'", (job_id,)
    ).fetchone()
    if row is None:
        return "skipped"
    if expected_owner is not None and str(row["locked_by"] or "") != str(expected_owner):
        return "skipped"
    if int(row["cancel_requested"] or 0):
        # A worker killed by "close / delete knowledge base" did not crash: this must not count as a retry,
        # let alone go back to the queue, or the next worker round would keep parsing a KB that is already
        # closed.
        cancel_job(con, job_id, "cancelled with the knowledge base")
        con.commit()
        return "cancelled"
    attempts = int(row["retry_count"] or 0) + 1
    give_up = attempts > max(0, max_retries)
    if give_up:
        con.execute(
            """
            UPDATE jobs
            SET status = 'failed', retry_count = ?, error = ?, finished_at = ?, updated_at = ?,
                locked_by = NULL, locked_until = NULL
            WHERE job_id = ? AND status = 'running' AND (? IS NULL OR locked_by = ?)
            """,
            (attempts, f"{reason} (attempts={attempts})"[:4000], ts, ts, job_id, expected_owner, expected_owner),
        )
    else:
        con.execute(
            """
            UPDATE jobs
            SET status = 'retry', retry_count = ?, next_attempt_at = ?, error = ?, updated_at = ?,
                locked_by = NULL, locked_until = NULL
            WHERE job_id = ? AND status = 'running' AND (? IS NULL OR locked_by = ?)
            """,
            (attempts, ts + max(1, retry_delay_seconds), f"{reason} (attempt {attempts})"[:4000], ts, job_id,
             expected_owner, expected_owner),
        )
    add_failure(
        con,
        file_id=str(row["file_id"]) if row["file_id"] is not None else None,
        job_id=job_id,
        stage="worker-crash",
        error_type="WorkerDied",
        error_message=reason,
    )
    con.commit()
    return "failed" if give_up else "retry"


def mark_job_paused(con: sqlite3.Connection, job_id: str, reason: str) -> None:
    ts = now_ts()
    con.execute(
        """
        UPDATE jobs
        SET status = 'paused', error = ?, updated_at = ?, locked_by = NULL, locked_until = NULL
        WHERE job_id = ?
        """,
        (reason[:4000], ts, job_id),
    )


def cancel_job(con: sqlite3.Connection, job_id: str, reason: str) -> None:
    ts = now_ts()
    cur = con.execute(
        """
        UPDATE jobs
        SET status = 'cancelled', error = ?, finished_at = ?, updated_at = ?,
            locked_by = NULL, locked_until = NULL
        WHERE job_id = ? AND status IN ('queued', 'retry', 'running')
        """,
        (reason[:4000], ts, ts, job_id),
    )
    if cur.rowcount:
        con.execute("INSERT INTO job_events(job_id, ts, kind, text) VALUES (?, ?, 'cancelled', ?)",
                    (job_id, ts, reason[:1000]))
    con.execute(
        "UPDATE failures SET resolved_at = COALESCE(resolved_at, ?) WHERE job_id = ?",
        (ts, job_id),
    )


def cancel_job_if_owned(con: sqlite3.Connection, job_id: str, worker_id: str, reason: str) -> bool:
    """Write the terminal state when the worker receives JobCancelled, but only touch jobs that are still
    ours: running with locked_by set to us, or explicitly requested to cancel. Jobs reclaimed as orphans
    into retry by another worker, or taken over, are left alone -- this used to write cancelled
    unconditionally, overwriting a retry that someone else had scheduled, and the file was never parsed
    again (on 2026-09-09, 8 files of the product KB and 1 file of the health KB were stuck on the old
    parser version because of this). Returns whether anything was written."""
    ts = now_ts()
    cur = con.execute(
        """
        UPDATE jobs
        SET status = 'cancelled', error = ?, finished_at = ?, updated_at = ?,
            locked_by = NULL, locked_until = NULL
        WHERE job_id = ?
          AND (COALESCE(cancel_requested, 0) = 1 OR (status = 'running' AND locked_by = ?))
        """,
        (reason[:4000], ts, ts, job_id, worker_id),
    )
    if cur.rowcount:
        con.execute("INSERT INTO job_events(job_id, ts, kind, text) VALUES (?, ?, 'cancelled', ?)",
                    (job_id, ts, reason[:1000]))
        con.execute(
            "UPDATE failures SET resolved_at = COALESCE(resolved_at, ?) WHERE job_id = ?",
            (ts, job_id),
        )
    return bool(cur.rowcount)


def request_kb_job_cancel(con: sqlite3.Connection, kb_id: str, reason: str) -> dict[str, int]:
    """Stop the parse queue of a knowledge base. Queued / retry jobs are voided outright; running jobs only
    get the cancel flag -- they belong to the worker, and parse_job exits on its own once it sees the
    flag at the next phase boundary. Returns {"cancelled": number voided, "signalled": number of running
    jobs asked to stop}."""
    ts = now_ts()
    cancelled = con.execute(
        """
        UPDATE jobs
        SET status = 'cancelled', error = ?, finished_at = ?, updated_at = ?,
            locked_by = NULL, locked_until = NULL
        WHERE kb_id = ? AND status IN ('queued', 'retry')
        """,
        (reason[:4000], ts, ts, kb_id),
    ).rowcount
    signalled = con.execute(
        "UPDATE jobs SET cancel_requested = 1, updated_at = ? "
        "WHERE kb_id = ? AND status = 'running'",
        (ts, kb_id),
    ).rowcount
    con.commit()
    return {"cancelled": int(cancelled or 0), "signalled": int(signalled or 0)}


def request_job_cancel(con: sqlite3.Connection, job_id: str, reason: str) -> str:
    """Cancel a single job, same semantics as request_kb_job_cancel: queued / retry jobs are voided
    outright, running jobs get the flag and the worker exits at the next phase boundary. Returns
    cancelled / signalled / noop."""
    row = con.execute("SELECT status FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    if row is None:
        raise KeyError(job_id)
    status = str(row["status"])
    if status in ("queued", "retry"):
        cancel_job(con, job_id, reason)
        con.commit()
        return "cancelled"
    if status == "running":
        con.execute("UPDATE jobs SET cancel_requested = 1, updated_at = ? WHERE job_id = ?", (now_ts(), job_id))
        con.commit()
        return "signalled"
    return "noop"


def job_cancel_requested(con: sqlite3.Connection, job_id: str, worker_id: str | None = None) -> bool:
    """Whether this worker should stop: cancellation was requested, the job is no longer running, or (when
    worker_id is given) the lock is no longer ours -- after a takeover by another worker the new owner is
    also running, so checking the status alone would let the ousted worker keep writing (Codex review
    N07)."""
    row = con.execute(
        "SELECT cancel_requested, status, locked_by FROM jobs WHERE job_id = ?", (job_id,)
    ).fetchone()
    if row is None:
        # A missing row (KB deletion already reached the SQLite cleanup step) also counts as cancelled: going
        # on would only write orphan data.
        return True
    if worker_id and str(row["locked_by"] or "") != str(worker_id):
        return True
    return bool(int(row["cancel_requested"] or 0)) or str(row["status"]) != "running"


def running_job_leases(con: sqlite3.Connection, kb_id: str) -> list[tuple[str, str]]:
    """Running jobs of this knowledge base that still hold a live lease -> [(job_id, locked_by)].
    locked_by is the "hostname:pid" written by worker.worker_id()."""
    rows = con.execute(
        "SELECT job_id, locked_by FROM jobs WHERE kb_id = ? AND status = 'running' "
        "AND COALESCE(locked_until, 0) >= ?",
        (kb_id, now_ts()),
    ).fetchall()
    return [(str(r["job_id"]), str(r["locked_by"] or "")) for r in rows]


def cancel_pending_jobs_for_file(
    con: sqlite3.Connection,
    file_id: str,
    reason: str,
    *,
    exclude_job_types: set[str] | None = None,
) -> int:
    ts = now_ts()
    params: list[Any] = [reason[:4000], ts, ts, file_id]
    exclude_clause = ""
    if exclude_job_types:
        placeholders = ",".join("?" for _ in exclude_job_types)
        exclude_clause = f" AND job_type NOT IN ({placeholders})"
        params.extend(sorted(exclude_job_types))
    cur = con.execute(
        f"""
        UPDATE jobs
        SET status = 'cancelled', error = ?, finished_at = ?, updated_at = ?,
            locked_by = NULL, locked_until = NULL
        WHERE file_id = ?
          AND status IN ('queued', 'retry')
          {exclude_clause}
        """,
        params,
    )
    con.execute(
        """
        UPDATE failures
        SET resolved_at = COALESCE(resolved_at, ?)
        WHERE file_id = ?
          AND job_id IN (SELECT job_id FROM jobs WHERE file_id = ? AND status = 'cancelled')
        """,
        (ts, file_id, file_id),
    )
    return int(cur.rowcount or 0)


def mark_job_failed(
    con: sqlite3.Connection,
    job_id: str,
    error: str,
    retry: bool = True,
    retry_delay_seconds: int = 300,
    worker_id: str | None = None,
) -> bool:
    """Terminal state for failure / backoff retry. When worker_id is given, only jobs that are still running
    and locked by us are touched: a job taken over or cancelled is not for the ousted worker to change
    (Codex review N07). Returns whether anything was written."""
    ts = now_ts()
    owner_clause = " AND (? IS NULL OR (status = 'running' AND locked_by = ?))"
    if retry:
        next_attempt_at = ts + max(1, retry_delay_seconds)
        cur = con.execute(
            """
            UPDATE jobs
            SET status = 'retry', retry_count = retry_count + 1, next_attempt_at = ?,
                error = ?, updated_at = ?, locked_by = NULL, locked_until = NULL
            WHERE job_id = ?
            """ + owner_clause,
            (next_attempt_at, error[:4000], ts, job_id, worker_id, worker_id),
        )
        if cur.rowcount:
            con.execute("INSERT INTO job_events(job_id, ts, kind, text) VALUES (?, ?, 'retry', ?)",
                        (job_id, ts, f"Retry in {max(1, retry_delay_seconds)}s: {error[:1000]}"))
    else:
        cur = con.execute(
            """
            UPDATE jobs
            SET status = 'failed', error = ?, finished_at = ?, updated_at = ?,
                locked_by = NULL, locked_until = NULL
            WHERE job_id = ?
            """ + owner_clause,
            (error[:4000], ts, ts, job_id, worker_id, worker_id),
        )
        if cur.rowcount:
            con.execute("INSERT INTO job_events(job_id, ts, kind, text) VALUES (?, ?, 'error', ?)",
                        (job_id, ts, error[:4000]))
    return bool(cur.rowcount)


def add_failure(
    con: sqlite3.Connection,
    *,
    file_id: str | None,
    job_id: str | None,
    stage: str,
    error_type: str,
    error_message: str,
) -> None:
    con.execute(
        """
        INSERT INTO failures(failure_id, file_id, job_id, stage, error_type, error_message, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (new_id("failure"), file_id, job_id, stage, error_type, error_message[:4000], now_ts()),
    )


def recent_jobs(con: sqlite3.Connection, limit: int = 20) -> Iterable[sqlite3.Row]:
    return con.execute(
        "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()


def get_file_by_id(con: sqlite3.Connection, file_id: str) -> sqlite3.Row | None:
    return con.execute("SELECT * FROM files WHERE file_id = ?", (file_id,)).fetchone()


def files_for_kb(con: sqlite3.Connection, kb_id: str) -> list[sqlite3.Row]:
    return con.execute(
        "SELECT * FROM files WHERE kb_id = ? AND status != 'deleted'",
        (kb_id,),
    ).fetchall()


def active_files_by_checksum(con: sqlite3.Connection, kb_id: str, checksum: str) -> list[sqlite3.Row]:
    return con.execute(
        """
        SELECT * FROM files
        WHERE kb_id = ? AND checksum = ? AND status != 'deleted'
        ORDER BY last_seen_at DESC
        """,
        (kb_id, checksum),
    ).fetchall()


def deleted_files_by_checksum(con: sqlite3.Connection, kb_id: str, checksum: str) -> list[sqlite3.Row]:
    return con.execute(
        """
        SELECT * FROM files
        WHERE kb_id = ? AND checksum = ? AND status = 'deleted'
        ORDER BY last_seen_at DESC
        """,
        (kb_id, checksum),
    ).fetchall()


def deleted_files_older_than(con: sqlite3.Connection, cutoff_ts: int) -> list[sqlite3.Row]:
    return con.execute(
        """
        SELECT * FROM files
        WHERE status = 'deleted' AND last_seen_at < ?
        ORDER BY last_seen_at ASC
        """,
        (cutoff_ts,),
    ).fetchall()


def mark_file_deleted(con: sqlite3.Connection, file_id: str) -> None:
    con.execute(
        "UPDATE files SET status = 'deleted', last_seen_at = ? WHERE file_id = ?",
        (now_ts(), file_id),
    )


def mark_file_indexed(
    con: sqlite3.Connection,
    file_id: str,
    content_version: str,
    parser_profile: str | None = None,
    *,
    chunk_diag: dict[str, Any] | None = None,
) -> None:
    con.execute(
        """
        UPDATE files
        SET indexed_version = ?,
            indexed_parser_profile = COALESCE(?, indexed_parser_profile),
            chunk_diag_json = ?,
            status = CASE WHEN status = 'deleted' THEN 'deleted' ELSE 'indexed' END,
            last_seen_at = ?
        WHERE file_id = ?
        """,
        (content_version, parser_profile, json.dumps(chunk_diag, ensure_ascii=False) if chunk_diag else None,
         now_ts(), file_id),
    )


def chunk_text_sha(text: str) -> str:
    """Chunk text fingerprint: first 16 hex digits of sha256. Written to the chunks table and the graph build
    ledger; incremental append uses it to detect "uid unchanged, text changed"."""
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()[:16]


def replace_chunks(
    con: sqlite3.Connection,
    *,
    file_id: str,
    collection: str,
    content_version: str,
    chunks: list[UnifiedChunk],
    point_ids: list[str],
) -> None:
    ts = now_ts()
    con.execute("UPDATE chunks SET status = 'inactive' WHERE file_id = ?", (file_id,))
    con.executemany(
        """
        INSERT INTO chunks(chunk_uid, file_id, content_version, chunk_index, point_id, collection, status, created_at, text_sha)
        VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?)
        ON CONFLICT(chunk_uid) DO UPDATE SET
          content_version = excluded.content_version,
          chunk_index = excluded.chunk_index,
          point_id = excluded.point_id,
          collection = excluded.collection,
          status = excluded.status,
          created_at = excluded.created_at,
          text_sha = excluded.text_sha
        """,
        [
            (chunk.chunk_uid, file_id, content_version, chunk.chunk_index, point_id, collection, ts, chunk_text_sha(chunk.text))
            for chunk, point_id in zip(chunks, point_ids, strict=True)
        ],
    )


# The criterion for "this job counts as in flight right now": running, or queued/retry whose backoff has
# elapsed so it will be claimed on the next round. Failed jobs still in backoff do not count -- otherwise a
# file that keeps failing would make the console show "parsing" indefinitely and block graph operations
# for up to an hour (the backoff cap).
CLAIMABLE_JOB_SQL = "(status='running' OR (status IN ('queued','retry') AND next_attempt_at <= ?))"


def kb_parse_busy(con: sqlite3.Connection, kb_id: str) -> int:
    """How many parse jobs of this knowledge base are still in flight.

    The single criterion for "can we build the graph / extract labels", shared by manual triggers and
    the automatic rebuild at 00:00 every day. Only looks at **this knowledge base itself**: building a
    graph while the corpus is still growing yields half a graph, while another KB being parsed has
    nothing to do with this one -- the embedding concurrency budget is partitioned (pipeline 20 + graph
    build 10 + 2 reserved for retrieval = max_num_seqs 32), so there is no need to yield to each other.
    """
    row = con.execute(
        f"SELECT COUNT(*) FROM jobs WHERE kb_id = ? AND job_type = 'parse' AND {CLAIMABLE_JOB_SQL}",
        (kb_id, now_ts()),
    ).fetchone()
    return int(row[0] if row else 0)


def active_chunk_refs(con: sqlite3.Connection, collection: str) -> list[dict[str, str]]:
    """Active chunk references for the graph build ledger.

    This ledger used to require a full scroll over Qdrant, yet the four fields it needs are all in
    SQLite anyway: the chunks table has point_id / chunk_uid / content_version, and doc_id is composed
    from kb_id:file_key in files (the same formula written into the payload). Taking it from here is
    much faster, and the graph build no longer depends on Qdrant being online.
    """
    rows = con.execute(
        """
        SELECT c.point_id AS point_id, c.chunk_uid AS chunk_uid,
               c.content_version AS content_version, c.text_sha AS text_sha,
               f.kb_id AS kb_id, f.file_key AS file_key
        FROM chunks c
        JOIN files f ON f.file_id = c.file_id
        WHERE c.collection = ? AND c.status = 'active'
        """,
        (collection,),
    ).fetchall()
    return [
        {
            "point_id": str(row["point_id"]),
            "chunk_uid": str(row["chunk_uid"]),
            "doc_id": f"{row['kb_id']}:{row['file_key']}",
            "content_version": str(row["content_version"]),
            "text_sha": str(row["text_sha"] or ""),
        }
        for row in rows
        if row["point_id"]
    ]


def mark_chunks_deleted(con: sqlite3.Connection, file_id: str) -> int:
    """Deletion path (file vanished / KB closed): only the chunks that are **active** right now are marked
    deleted. Old chunks of the same content version that were replaced (left over from re-chunking after
    a chunking rule or parameter change) are already inactive and are left alone -- on restore only the
    deleted batch comes back. On 2026-09-06, closing and reopening a KB with the old implementation
    flipped back everything by "file + version", reviving every chunking ever done: the chunk ledger
    grew 6x and the automatic rebuild was triggered by a phantom increment."""
    cur = con.execute(
        "UPDATE chunks SET status = 'deleted' WHERE file_id = ? AND status = 'active'",
        (file_id,),
    )
    return int(cur.rowcount or 0)


def chunk_point_ids_to_restore(con: sqlite3.Connection, file_id: str, content_version: str) -> list[str]:
    """Point ids to flip back to active for a file that returned after deletion: the batch that was active
    then and is marked deleted now."""
    return [str(r["point_id"]) for r in con.execute(
        "SELECT point_id FROM chunks WHERE file_id = ? AND content_version = ? AND status = 'deleted' "
        "ORDER BY chunk_index, chunk_uid",
        (file_id, content_version),
    )]


def mark_chunks_active(con: sqlite3.Connection, file_id: str, content_version: str) -> int:
    """Re-mark the **deleted** chunks of a file's current version as active.

    A file that returns after deletion is restored via metadata_update (reactivate) without re-parsing:
    reactivate_file_metadata flips the Qdrant points back to is_active=True, but on the SQLite side only
    the mark_chunks_inactive half ever existed -- chunks.status stayed inactive forever. That stale state
    used to be harmless (the graph build ledger came from a Qdrant scroll); once the ledger moved to
    SQLite it became load-bearing: the file came back but never entered the graph, and since the ledger
    did not change the rebuild policy could not notice either.

    Only deleted rows are flipped (the batch that was active at the moment of closing / deletion): old
    chunks replaced within the same version and chunks of older versions are inactive and keep waiting
    for GC. Old data deleted before the fix has no deleted state and cannot be flipped back here; the
    worker falls back to a re-parse to realign both sides.
    """
    cur = con.execute(
        "UPDATE chunks SET status = 'active' "
        "WHERE file_id = ? AND content_version = ? AND status = 'deleted'",
        (file_id, content_version),
    )
    return int(cur.rowcount or 0)


def mark_chunks_inactive(con: sqlite3.Connection, file_id: str) -> int:
    cur = con.execute(
        "UPDATE chunks SET status = 'inactive' WHERE file_id = ? AND status != 'inactive'",
        (file_id,),
    )
    return int(cur.rowcount or 0)


def delete_chunks_for_version(con: sqlite3.Connection, file_id: str, content_version: str) -> int:
    row = con.execute(
        "SELECT COUNT(*) AS c FROM chunks WHERE file_id = ? AND content_version = ?",
        (file_id, content_version),
    ).fetchone()
    count = int(row["c"] if row else 0)
    con.execute(
        "DELETE FROM chunks WHERE file_id = ? AND content_version = ?",
        (file_id, content_version),
    )
    return count


# Whole-KB queries go through subselects instead of expanded IN (...) lists:
# a KB re-ingested a few times crosses SQLite's bound-variable limit (32766)
# and the expanded form starts throwing "too many SQL variables".
_KB_FAILURES_WHERE = (
    "file_id IN (SELECT file_id FROM files WHERE kb_id = :kb) "
    "OR job_id IN (SELECT job_id FROM jobs WHERE kb_id = :kb)"
)


def kb_state_counts(con: sqlite3.Connection, kb_id: str) -> dict[str, int]:
    def one(sql: str) -> int:
        row = con.execute(sql, {"kb": kb_id}).fetchone()
        return int(row["c"] if row else 0)

    return {
        "files": one("SELECT COUNT(*) AS c FROM files WHERE kb_id = :kb"),
        "chunks": one("SELECT COUNT(*) AS c FROM chunks WHERE file_id IN (SELECT file_id FROM files WHERE kb_id = :kb)"),
        "jobs": one("SELECT COUNT(*) AS c FROM jobs WHERE kb_id = :kb"),
        "failures": one(f"SELECT COUNT(*) AS c FROM failures WHERE {_KB_FAILURES_WHERE}"),
    }


def purge_kb_state(con: sqlite3.Connection, kb_id: str) -> dict[str, int]:
    counts = kb_state_counts(con, kb_id)
    con.execute(f"DELETE FROM failures WHERE {_KB_FAILURES_WHERE}", {"kb": kb_id})
    con.execute("DELETE FROM chunks WHERE file_id IN (SELECT file_id FROM files WHERE kb_id = ?)", (kb_id,))
    con.execute("DELETE FROM job_events WHERE job_id IN (SELECT job_id FROM jobs WHERE kb_id = ?)", (kb_id,))
    con.execute("DELETE FROM jobs WHERE kb_id = ?", (kb_id,))
    con.execute("DELETE FROM files WHERE kb_id = ?", (kb_id,))
    return counts


def purge_file_state(con: sqlite3.Connection, file_id: str) -> dict[str, int]:
    counts = file_state_counts(con, file_id)
    _delete_failure_refs(con, [file_id], _job_ids_for_file(con, file_id))
    con.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
    con.execute("DELETE FROM job_events WHERE job_id IN (SELECT job_id FROM jobs WHERE file_id = ?)", (file_id,))
    con.execute("DELETE FROM jobs WHERE file_id = ?", (file_id,))
    con.execute("DELETE FROM files WHERE file_id = ?", (file_id,))
    return counts


def file_state_counts(con: sqlite3.Connection, file_id: str) -> dict[str, int]:
    file_row = con.execute("SELECT file_id FROM files WHERE file_id = ?", (file_id,)).fetchone()
    job_ids = _job_ids_for_file(con, file_id)
    file_ids = [file_id] if file_row else []
    return {
        "files": len(file_ids),
        "chunks": _count_in(con, "chunks", "file_id", [file_id]),
        "jobs": len(job_ids),
        "failures": _count_failure_refs(con, [file_id], job_ids),
    }


def pending_jobs_for_file(con: sqlite3.Connection, file_id: str) -> list[sqlite3.Row]:
    return con.execute(
        """
        SELECT job_id, job_type, status
        FROM jobs
        WHERE file_id = ? AND status IN ('queued', 'retry', 'running')
        ORDER BY created_at ASC
        """,
        (file_id,),
    ).fetchall()


def _job_ids_for_file(con: sqlite3.Connection, file_id: str) -> list[str]:
    return [str(row["job_id"]) for row in con.execute("SELECT job_id FROM jobs WHERE file_id = ?", (file_id,)).fetchall()]


def running_jobs_for_kbs(con: sqlite3.Connection, kb_ids: list[str]) -> list[sqlite3.Row]:
    if not kb_ids:
        return []
    placeholders = ",".join("?" for _ in kb_ids)
    return con.execute(
        f"""
        SELECT job_id, kb_id, collection, job_type, status, locked_by, started_at
        FROM jobs
        WHERE kb_id IN ({placeholders}) AND status = 'running'
        ORDER BY started_at ASC
        """,
        kb_ids,
    ).fetchall()


_SQL_IN_BATCH = 500


def _count_in(con: sqlite3.Connection, table: str, column: str, values: list[str]) -> int:
    total = 0
    for start in range(0, len(values), _SQL_IN_BATCH):
        batch = values[start : start + _SQL_IN_BATCH]
        placeholders = ",".join("?" for _ in batch)
        row = con.execute(f"SELECT COUNT(*) AS c FROM {table} WHERE {column} IN ({placeholders})", batch).fetchone()
        total += int(row["c"] if row else 0)
    return total


def _delete_in(con: sqlite3.Connection, table: str, column: str, values: list[str]) -> None:
    for start in range(0, len(values), _SQL_IN_BATCH):
        batch = values[start : start + _SQL_IN_BATCH]
        placeholders = ",".join("?" for _ in batch)
        con.execute(f"DELETE FROM {table} WHERE {column} IN ({placeholders})", batch)


def _count_failure_refs(con: sqlite3.Connection, file_ids: list[str], job_ids: list[str]) -> int:
    clauses: list[str] = []
    params: list[str] = []
    if file_ids:
        clauses.append(f"file_id IN ({','.join('?' for _ in file_ids)})")
        params.extend(file_ids)
    if job_ids:
        clauses.append(f"job_id IN ({','.join('?' for _ in job_ids)})")
        params.extend(job_ids)
    if not clauses:
        return 0
    row = con.execute(f"SELECT COUNT(*) AS c FROM failures WHERE {' OR '.join(clauses)}", params).fetchone()
    return int(row["c"] if row else 0)


def _delete_failure_refs(con: sqlite3.Connection, file_ids: list[str], job_ids: list[str]) -> None:
    clauses: list[str] = []
    params: list[str] = []
    if file_ids:
        clauses.append(f"file_id IN ({','.join('?' for _ in file_ids)})")
        params.extend(file_ids)
    if job_ids:
        clauses.append(f"job_id IN ({','.join('?' for _ in job_ids)})")
        params.extend(job_ids)
    if clauses:
        con.execute(f"DELETE FROM failures WHERE {' OR '.join(clauses)}", params)
