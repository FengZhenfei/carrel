"""Knowledge-base enrollment: which top-level mirror directories are managed.

The mirror root (KB_MIRROR_ROOT) receives *everything* the user's sync tool
puts there, but a directory only enters the pipeline when the user enrolls it
(the web console checkbox).
kb_sources is the authoritative registry: a row exists for every enrolled KB,
`settings.sources` is built from the active rows, and per-KB strategy
(chunk sizing, VLM prompt, graph settings) lives in the row's config_json --
there is no file-based `.kb.json` any more, because the mirror is typically a
one-way sync target (rsync --delete or similar) and files written into it
would be stomped.

The four historical KBs keep their original kb_id / collection through an
explicit alias table (those ids are baked into chunk_uid -> point_id and the
parse-cache layout); anything else gets a stable id derived from the
directory name.

Lifecycle (identical machinery to per-file soft deletion, one level up):

  enroll (checkbox on)   -> kb_sources row (active), Qdrant collection +
                            OpenSearch index ensured, scan starts covering it
  unenroll (checkbox off)-> status = inactive (reason 'unenrolled'), every
                            file rides the delete queue: points flip
                            is_active=false, OpenSearch rows drop
  directory vanishes     -> same, with reason 'directory_missing' (scan)
  re-enroll in retention -> row flips back to active; files walk the normal
                            restore path (same checksum -> reactivation,
                            no re-parse)
  inactive > N days      -> daily GC drops the collection, the index, the
                            graph artefacts and the row; enrolling after
                            that is a full re-parse
"""

from __future__ import annotations

import os

import json
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

from .limits import (
    GRAPH_MAX_GLEANINGS_DEFAULT, GRAPH_TUNE_SAMPLE_DEFAULT, GRAPH_UNIT_CHUNKS_DEFAULT,
    normalize_entity_types, normalize_examples, normalize_parent_types, normalize_predicates,
    normalize_profile, normalize_type_definitions,
)
from .models import GraphRebuildPolicy, KBSource


DEFAULTS = {
    "max_tokens": 400,
    "overlap_tokens": 80,
    "graph_enabled": False,
    "graph_unit_chunks": GRAPH_UNIT_CHUNKS_DEFAULT,
    "graph_max_gleanings": GRAPH_MAX_GLEANINGS_DEFAULT,
    # Empty values for derived data: empty = use the global default type table / no language constraint / only
    # the single predicate related_to.
    "graph_entity_types": (),
    "graph_language": None,
    "graph_predicates": (),
    "graph_parent_types": {},
    "graph_type_definitions": {},
    "graph_examples": "",
    "graph_profile": {},
    # Historical versions of the entity labels (newest first, ≤3) and the id of the version currently in
    # effect. The effective values are still graph_entity_types / graph_language -- the graph build reads
    # only those two; this is only about "being able to roll back".
    "graph_schema_versions": (),
    "graph_schema_active": None,
    "graph_tune_sample_size": 8,
    # The persistent intent left by "pause graph build". It is distinct from graph_enabled: the latter is
    # "whether to build at all from now on", this one is "stop for now". See the comment in
    # graph/build.py::evaluate_rebuild for why they are not merged.
    "graph_paused": False,
    # New / changed documents are automatically appended to the current graph (checked every 30 minutes); a
    # full rebuild still follows the policy conditions below.
    "graph_auto_append": True,
    # A threshold-triggered full rebuild first re-extracts a label version on top of the current version +
    # endpoint ledger and activates it before building (2026-09-08); when off, build with the current version
    "graph_rebuild_resuggest": True,
}
# Convenience presets applied at first enrollment (then stored in config_json
# and fully editable in the console). Keyed by DIRECTORY NAME -- ids are
# per-enrollment sequence numbers and carry no meaning.
DIR_DEFAULTS: dict[str, dict[str, Any]] = {
    "图书馆": {"max_tokens": 800, "overlap_tokens": 120},
    "library": {"max_tokens": 800, "overlap_tokens": 120},
    # graph_enabled is never preset: enabling the knowledge graph must be an explicit user action in the
    # console; only the automatic rebuild policy is prefilled here so it is ready once the user enables it.
    "产品资料": {"graph_rebuild_interval": "1m",
               "graph_rebuild_new_chunk_pct": "20%", "graph_rebuild_operator": "and"},
    "products": {"graph_rebuild_interval": "1m",
                 "graph_rebuild_new_chunk_pct": "20%", "graph_rebuild_operator": "and"},
}

# Keys the web console may write into config_json.
CONFIG_KEYS = {
    "max_tokens", "overlap_tokens", "vlm_prompt", "graph_enabled",
    "graph_unit_chunks", "graph_max_gleanings",
    "graph_rebuild_interval", "graph_rebuild_new_chunk_pct",
    "graph_rebuild_new_chunk_count",   # mutually exclusive with _pct; the console dropdown picks one
    "graph_rebuild_operator",
    "graph_auto_append",
    "graph_rebuild_resuggest",
    "graph_llm",  # {"extract"|"summarize"|"tune": name} -> llm_registry
    # Extraction constraints that the console's "extract / re-extract labels" induces with the LLM (derived
    # data, cleared when the graph is deleted), plus that step's own sample size.
    "graph_entity_types", "graph_language", "graph_tune_sample_size",
    "graph_predicates", "graph_parent_types",
    "graph_type_definitions", "graph_examples", "graph_profile",
    "graph_schema_versions", "graph_schema_active",
    # Server-side only: written by "pause graph build", cleared by "build / rebuild graph now" and "enable
    # knowledge graph". The console cannot write it directly (api._reject_bad_config blocks it) -- it is the
    # trace of an action, not an option to tick casually.
    "graph_paused",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS kb_sources (
  kb_id TEXT PRIMARY KEY,
  collection TEXT NOT NULL UNIQUE,
  source_root TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'active',
  first_seen_at INTEGER NOT NULL,
  last_seen_at INTEGER NOT NULL,
  inactive_at INTEGER,
  config_json TEXT NOT NULL DEFAULT '{}'
);
"""

_MIGRATIONS = (
    "ALTER TABLE kb_sources ADD COLUMN inactive_reason TEXT",
)

_SKIP_DIR_RE = re.compile(r"^[._~]|^\$RECYCLE|^System Volume|^lost\+found$")


def allocate_kb_id(con: sqlite3.Connection) -> str:
    """kb_id == collection == kb_<NNN>, allocated at first enrollment from a
    persistent counter (never reused, so a GC-and-re-enrolled directory gets a
    fresh number). Numbers instead of directory-derived names: identifiers
    stay ASCII for every store (OpenSearch index names in particular), carry
    no semantics, and renaming a directory has no naming side effects."""
    seq = int(con.execute(
        "SELECT COALESCE(MAX(CAST(value AS INTEGER)), 0) FROM app_config WHERE key = 'kb_seq'"
    ).fetchone()[0]) + 1
    con.execute(
        "INSERT INTO app_config(key, value, updated_at) VALUES ('kb_seq', ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
        (str(seq), now()),
    )
    return f"kb_{seq:03d}"


class CollectionMismatch(RuntimeError):
    """The registry row's collection no longer matches its kb_id -- an
    invariant break (tampered row); nothing must ingest into it."""

    def __init__(self, kb_id: str, stored: str, incoming: str) -> None:
        super().__init__(
            f"kb {kb_id!r} is registered with collection {stored!r} but the code derives "
            f"{incoming!r}; refusing to touch it until the registry row is reconciled"
        )
        self.kb_id = kb_id
        self.stored = stored
        self.incoming = incoming




def build_source(
    mirror_root: Path,
    name: str,
    config: dict[str, Any] | None = None,
    *,
    kb_id: str = "kb_000",
    collection: str | None = None,
) -> KBSource:
    from .config import (  # local import: avoid cycle
        bool_value, parse_interval_days, parse_new_chunk_count, parse_ratio,
    )

    collection = collection or kb_id
    cfg: dict[str, Any] = dict(DEFAULTS)
    cfg.update(DIR_DEFAULTS.get(name, {}))
    cfg.update(config or {})
    # Cap the ratio once more: the console already clamps it to 100% on save, but the config can also be
    # changed via the CLI or by writing the database directly, and "rebuild after 150% growth" is no
    # different from "rebuild after doubling" in the verdict -- it only makes the threshold look stricter
    # than it is.
    ratio = parse_ratio(cfg.get("graph_rebuild_new_chunk_pct"))
    policy = GraphRebuildPolicy(
        interval_days=parse_interval_days(cfg.get("graph_rebuild_interval")),
        new_chunk_ratio=None if ratio is None else min(1.0, ratio),
        new_chunk_count=parse_new_chunk_count(cfg.get("graph_rebuild_new_chunk_count")),
        operator=str(cfg.get("graph_rebuild_operator", "or")).strip().lower(),
    )
    prompt = str(cfg.get("vlm_prompt") or "").strip() or None
    return KBSource(
        kb_id=kb_id,
        collection=collection,
        source_root=name,
        source_type="local_mirror",
        physical_base=mirror_root / name,
        max_tokens=int(cfg["max_tokens"]),
        overlap_tokens=int(cfg["overlap_tokens"]),
        vlm_prompt=prompt,
        graph_enabled=bool_value(cfg.get("graph_enabled"), False),
        graph_paused=bool_value(cfg.get("graph_paused"), False),
        graph_auto_append=bool_value(cfg.get("graph_auto_append"), True),
        graph_unit_chunks=int(cfg.get("graph_unit_chunks") or GRAPH_UNIT_CHUNKS_DEFAULT),
        graph_max_gleanings=int(cfg.get("graph_max_gleanings") if cfg.get("graph_max_gleanings") is not None
                                else GRAPH_MAX_GLEANINGS_DEFAULT),
        graph_entity_types=normalize_entity_types(cfg.get("graph_entity_types")),
        graph_language=str(cfg.get("graph_language") or "").strip() or None,
        graph_predicates=normalize_predicates(cfg.get("graph_predicates")),
        graph_parent_types=normalize_parent_types(cfg.get("graph_parent_types")),
        graph_type_definitions=normalize_type_definitions(cfg.get("graph_type_definitions")),
        graph_examples=normalize_examples(cfg.get("graph_examples")),
        graph_profile=normalize_profile(cfg.get("graph_profile")),
        graph_tune_sample_size=int(cfg.get("graph_tune_sample_size") or GRAPH_TUNE_SAMPLE_DEFAULT),
        graph_rebuild_policy=policy,
    )


def linked_dirs_allowed() -> bool:
    """KB_MIRROR_ALLOW_LINKED_DIRS: whether a top-level mirror entry that is a symbolic link may be read."""
    return os.getenv("KB_MIRROR_ALLOW_LINKED_DIRS", "").strip().lower() in {"1", "true", "yes", "on"}


def directory_admitted(mirror_root: Path, name: str) -> tuple[bool, str]:
    """Whether the top-level mirror directory `name` may be read right now: it has to be a directory, and a
    symbolic link only when KB_MIRROR_ALLOW_LINKED_DIRS is on. The same rule applies when a directory is
    discovered, when an enrolled knowledge base is loaded and when the worker opens a file (security review
    F04 follow-up: an enrolled directory swapped for a link must not turn the link's target into the boundary).
    Returns (admitted, reason) with reason "missing" or "linked"."""
    path = mirror_root / name
    if not path.is_dir():
        return False, "missing"
    if path.is_symlink() and not linked_dirs_allowed():
        return False, "linked"
    return True, ""


def discover_directories(mirror_root: Path) -> list[str]:
    """Every top-level directory in the mirror -- enrolled or not. This is the
    web console's candidate list; it creates nothing.

    A top-level entry that is a symlink is ignored unless
    KB_MIRROR_ALLOW_LINKED_DIRS is set: a sync tool copying links, or a stray
    `ln -s`, must not silently turn a directory elsewhere on this host into a
    knowledge base. Mounting a folder on purpose is what that switch is for."""
    if not mirror_root.is_dir():
        return []
    names = []
    for child in sorted(mirror_root.iterdir()):
        if not child.is_dir() or _SKIP_DIR_RE.match(child.name):
            continue
        if not directory_admitted(mirror_root, child.name)[0]:
            continue
        names.append(child.name)
    return names


def enrolled_sources(state_db: Path, mirror_root: Path) -> dict[str, KBSource]:
    """The pipeline's live sources: active kb_sources rows whose directory is
    present and admitted by the link policy. An active row with a missing directory is deliberately
    excluded -- cmd_scan's vanish pass sees it as gone and starts the soft delete. A row whose directory
    has become a symbolic link (and links are not allowed) is excluded too, but stays registered: the
    console shows why, nothing is deleted, and it comes back as soon as the link is replaced by a directory
    or KB_MIRROR_ALLOW_LINKED_DIRS is set."""
    if not Path(state_db).exists():
        return {}
    try:
        con = sqlite3.connect(str(state_db), timeout=5)
    except sqlite3.Error:
        return {}
    try:
        con.row_factory = sqlite3.Row
        try:
            rows = con.execute("SELECT * FROM kb_sources WHERE status = 'active' ORDER BY kb_id").fetchall()
        except sqlite3.OperationalError:
            return {}  # table not created yet
        out: dict[str, KBSource] = {}
        for row in rows:
            admitted, why = directory_admitted(mirror_root, str(row["source_root"]))
            if not admitted:
                if why == "linked":
                    print(f"[discovery] skipping {row['kb_id']} ({row['source_root']}): the directory is a symbolic "
                          "link and KB_MIRROR_ALLOW_LINKED_DIRS is off", file=sys.stderr, flush=True)
                continue
            try:
                src = source_from_row(mirror_root, row)
            except Exception as exc:
                # One bad config_json row (hand-edited, old data, an invalid rebuild interval) used to make
                # load_settings raise as a whole -- web, scan, worker and GC all failed to start, and every
                # console endpoint returned 500. Skip this row with a warning; the other KBs keep working.
                print(f"[discovery] skipping {row['kb_id']} ({row['source_root']}): bad config: {exc!r}",
                      file=sys.stderr, flush=True)
                continue
            out[src.kb_id] = src
        return out
    finally:
        con.close()


# ── state ────────────────────────────────────────────────────────────────

def init_schema(con: sqlite3.Connection) -> None:
    con.executescript(SCHEMA)
    for statement in _MIGRATIONS:
        try:
            con.execute(statement)
        except sqlite3.OperationalError as exc:
            # Only swallow "column / index already exists". A lock, a read-only database or a full disk is not
            # "already migrated"; swallowing those makes later code fail on a table missing the column with a
            # harder-to-read error -- same criterion as db.migrate_schema (health check B7)
            message = str(exc).lower()
            if "duplicate column" in message or "already exists" in message:
                continue
            raise


def now() -> int:
    return int(time.time())


def enroll(con: sqlite3.Connection, mirror_root: Path, name: str) -> tuple[KBSource, str]:
    """Enroll a top-level directory. Returns (source, outcome) where outcome is
    'new' (first enrollment), 'reactivated' (re-enrolled inside the retention
    window; files restore without re-parsing) or 'already' (was active)."""
    if name not in discover_directories(mirror_root):
        raise ValueError(f"directory {name!r} does not exist under the mirror root")
    # Identity is looked up by directory name: a directory re-enrolled inside
    # the retention window gets its old number back (restore path); after GC
    # the row is gone and the directory starts over with a fresh number.
    row = con.execute(
        "SELECT * FROM kb_sources WHERE source_root = ? ORDER BY (status = 'active') DESC LIMIT 1", (name,)
    ).fetchone()
    ts = now()
    if row is None:
        kb_id = allocate_kb_id(con)
        config = dict(DIR_DEFAULTS.get(name, {}))
        con.execute(
            "INSERT INTO kb_sources(kb_id, collection, source_root, status, first_seen_at, last_seen_at, config_json) "
            "VALUES (?, ?, ?, 'active', ?, ?, ?)",
            (kb_id, kb_id, name, ts, ts, json.dumps(config, ensure_ascii=False)),
        )
        return build_source(mirror_root, name, config, kb_id=kb_id), "new"
    kb_id = str(row["kb_id"])
    if str(row["collection"]) != kb_id:
        raise CollectionMismatch(kb_id, str(row["collection"]), kb_id)
    outcome = "already" if str(row["status"]) == "active" else "reactivated"
    con.execute(
        "UPDATE kb_sources SET status='active', last_seen_at=?, inactive_at=NULL, inactive_reason=NULL "
        "WHERE kb_id=?",
        (ts, kb_id),
    )
    return source_from_row(mirror_root, con.execute("SELECT * FROM kb_sources WHERE kb_id=?", (kb_id,)).fetchone()), outcome


def kb_label(kb_id: str, source_root: str | None) -> str:
    """For identifying a KB in logs: kb_004(semiconductors). The id is the key; the directory name is for
    humans."""
    return f"{kb_id}({source_root})" if source_root else str(kb_id)


def directory_match_report(con: sqlite3.Connection, mirror_root: Path, kb_id: str, dir_name: str,
                           *, sample: int = 20) -> dict[str, Any]:
    """Whether an unenrolled directory looks like a renamed KB: look for the files the KB has on record
    (relative path + size) in the new directory, then compare checksums on a sample. Returns matched / total
    / ratio / verified / new_files."""
    from .localfs.scanner import SUPPORTED_EXTS, SUPPORTED_FILENAMES
    from .utils import sha256_file

    rows = con.execute("SELECT rel_path, size, checksum FROM files WHERE kb_id = ?", (kb_id,)).fetchall()
    root = mirror_root / dir_name
    matched: list[tuple[Path, str]] = []
    for row in rows:
        path = root / str(row["rel_path"])
        try:
            if path.is_file() and path.stat().st_size == int(row["size"]):
                matched.append((path, str(row["checksum"] or "")))
        except OSError:
            continue
    verified = 0
    mismatched = 0
    for path, checksum in matched[:sample]:
        if not checksum:
            continue
        try:
            if sha256_file(path) == checksum:
                verified += 1
            else:
                mismatched += 1
        except OSError:
            mismatched += 1
    new_files = 0
    try:
        for path in root.rglob("*"):
            if path.is_file() and (path.suffix.lower() in SUPPORTED_EXTS or path.name in SUPPORTED_FILENAMES):
                new_files += 1
    except OSError:
        pass
    total = len(rows)
    return {"kb_id": kb_id, "dir": dir_name, "matched": len(matched), "total": total,
            "ratio": (len(matched) / total) if total else 0.0, "verified": verified, "mismatched": mismatched,
            "new_files": new_files}


def looks_like_rename(report: dict[str, Any], *, min_ratio: float = 0.9) -> bool:
    """The rename criterion: over ninety percent of the KB's files exist unchanged in the new directory (path +
    size), no sampled checksum mismatches, and the new directory is not a much larger one that merely
    swallowed the old KB whole."""
    if not report.get("total") or report.get("mismatched"):
        return False
    if report["ratio"] < min_ratio:
        return False
    return int(report.get("new_files") or 0) <= max(2 * int(report["total"]), int(report["total"]) + 20)


def adopt_directory(con: sqlite3.Connection, mirror_root: Path, kb_id: str, new_name: str) -> KBSource:
    """Adopt a directory as the renamed home of a registered KB: the KB's id, collection, index, cache and graph
    are all kept; only source_root is pointed at the new directory and the file rows' paths follow (the
    metadata fingerprint keeps its old value: the next scan judges them metadata_changed, and the
    metadata_update job refreshes the path fields in Qdrant / OpenSearch without re-parsing). Only allowed
    for a KB whose directory is no longer on disk (deactivated, or just vanished and not yet marked inactive
    by the scan); the new directory must exist and no other registry row may hold that name."""
    if new_name not in discover_directories(mirror_root):
        raise ValueError(f"directory {new_name!r} does not exist under the mirror root")
    row = con.execute("SELECT * FROM kb_sources WHERE kb_id=?", (kb_id,)).fetchone()
    if row is None:
        raise KeyError(kb_id)
    old_name = str(row["source_root"])
    if old_name == new_name:
        raise ValueError(f"{kb_id} already lives in {new_name!r}")
    if str(row["status"]) == "active" and (mirror_root / old_name).is_dir():
        raise ValueError(f"The directory “{old_name}” of {kb_id} still exists; it cannot also point to “{new_name}”")
    taken = con.execute("SELECT kb_id FROM kb_sources WHERE source_root=? AND kb_id!=?", (new_name, kb_id)).fetchone()
    if taken is not None:
        raise ValueError(f"Directory “{new_name}” is already registered as {taken['kb_id']}")
    ts = now()
    con.execute(
        "UPDATE kb_sources SET source_root=?, status='active', last_seen_at=?, inactive_at=NULL, inactive_reason=NULL "
        "WHERE kb_id=?",
        (new_name, ts, kb_id),
    )
    for f in con.execute("SELECT file_id, rel_path FROM files WHERE kb_id=?", (kb_id,)).fetchall():
        rel_path = str(f["rel_path"])
        con.execute(
            "UPDATE files SET source_root=?, source_path=?, physical_path=? WHERE file_id=?",
            (new_name, f"{new_name}/{rel_path}", str(mirror_root / new_name / rel_path), str(f["file_id"])),
        )
    return source_from_row(mirror_root, con.execute("SELECT * FROM kb_sources WHERE kb_id=?", (kb_id,)).fetchone())


def find_renamed_directories(con: sqlite3.Connection, mirror_root: Path, *, min_ratio: float = 0.9) -> list[dict[str, Any]]:
    """For the scan: KBs deactivated because their directory vanished × unregistered directories on disk; a
    content match means a rename. Each directory is matched to at most one KB and each KB to at most one
    directory (both taking the highest match count)."""
    known = known_sources(con)
    registered = {str(r["source_root"]) for r in known}
    candidates = [r for r in known
                  if str(r["status"]) == "inactive" and str(r["inactive_reason"] or "") == "directory_missing"
                  and not (mirror_root / str(r["source_root"])).is_dir()]
    if not candidates:
        return []
    unenrolled = [d for d in discover_directories(mirror_root) if d not in registered]
    if not unenrolled:
        return []
    reports: list[dict[str, Any]] = []
    for r in candidates:
        for d in unenrolled:
            report = directory_match_report(con, mirror_root, str(r["kb_id"]), d)
            if looks_like_rename(report, min_ratio=min_ratio):
                report["old_dir"] = str(r["source_root"])
                reports.append(report)
    reports.sort(key=lambda x: (-x["matched"], x["kb_id"], x["dir"]))
    used_kb: set[str] = set()
    used_dir: set[str] = set()
    out: list[dict[str, Any]] = []
    for report in reports:
        if report["kb_id"] in used_kb or report["dir"] in used_dir:
            continue
        used_kb.add(report["kb_id"])
        used_dir.add(report["dir"])
        out.append(report)
    return out


def mark_inactive(con: sqlite3.Connection, kb_id: str, *, reason: str = "directory_missing") -> None:
    con.execute(
        "UPDATE kb_sources SET status='inactive', inactive_at=COALESCE(inactive_at, ?), "
        "inactive_reason=COALESCE(inactive_reason, ?) WHERE kb_id=?",
        (now(), reason, kb_id),
    )


def touch_seen(con: sqlite3.Connection, src: KBSource) -> None:
    """Refresh last_seen_at for an enrolled source during a scan. Never flips
    status: re-activation is an explicit enroll() -- a directory merely being
    present must not undo the user's unenroll."""
    row = con.execute("SELECT collection FROM kb_sources WHERE kb_id = ?", (src.kb_id,)).fetchone()
    if row is None:
        return
    if str(row["collection"]) != src.collection:
        raise CollectionMismatch(src.kb_id, str(row["collection"]), src.collection)
    con.execute(
        "UPDATE kb_sources SET last_seen_at=? WHERE kb_id=? AND status='active'",
        (now(), src.kb_id),
    )


def get_config(con: sqlite3.Connection, kb_id: str) -> dict[str, Any]:
    row = con.execute("SELECT config_json FROM kb_sources WHERE kb_id=?", (kb_id,)).fetchone()
    if row is None:
        raise KeyError(kb_id)
    try:
        data = json.loads(row["config_json"] or "{}")
    except Exception:
        data = {}
    return data if isinstance(data, dict) else {}


def set_config(con: sqlite3.Connection, kb_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    """Merge validated updates into config_json. Unknown keys are rejected;
    a None value removes the key (falls back to defaults)."""
    unknown = set(updates) - CONFIG_KEYS
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    config = get_config(con, kb_id)
    for key, value in updates.items():
        if value is None:
            config.pop(key, None)
        else:
            config[key] = value
    con.execute(
        "UPDATE kb_sources SET config_json=? WHERE kb_id=?",
        (json.dumps(config, ensure_ascii=False), kb_id),
    )
    return config


def known_sources(con: sqlite3.Connection) -> list[sqlite3.Row]:
    return con.execute("SELECT * FROM kb_sources ORDER BY kb_id").fetchall()


def inactive_older_than(con: sqlite3.Connection, cutoff_ts: int) -> list[sqlite3.Row]:
    return con.execute(
        "SELECT * FROM kb_sources WHERE status='inactive' AND inactive_at IS NOT NULL AND inactive_at < ? ORDER BY inactive_at",
        (cutoff_ts,),
    ).fetchall()


def forget(con: sqlite3.Connection, kb_id: str) -> None:
    con.execute("DELETE FROM kb_sources WHERE kb_id=?", (kb_id,))


def source_from_row(mirror_root: Path, row: sqlite3.Row) -> KBSource:
    """Rebuild a KBSource from its registry row (also used for KBs whose
    directory is gone, so delete/GC paths can address the collection)."""
    cfg = {}
    try:
        cfg = json.loads(row["config_json"] or "{}")
    except Exception:
        pass
    return build_source(
        mirror_root,
        str(row["source_root"]),
        cfg,
        kb_id=str(row["kb_id"]),
        collection=str(row["collection"]),
    )
