from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3

import requests
import sys
import time
from pathlib import Path

from . import db
from . import discovery, search_fts
from .config import load_settings
from .graph.build import (GraphBuildInterrupted, adopt_current_graph, build_graph, evaluate_append, evaluate_rebuild,
                          llm_ready, rollback_graph_version)
from .graph.llm import LLMInterrupted
from .graph.lock import build_lock_held, build_lock_path, clear_lock_leftovers
from .graph.neo4j_import import delete_neo4j_graph_version, import_graph_to_neo4j, neo4j_status
from .localfs.scanner import list_recent_source_files, list_source_files
from .maintenance import (
    graph_gc,
    kb_sources_gc,
    monthly_log_cleanup,
    neo4j_graph_gc,
    parse_assets_gc,
    qdrant_backfill_inactive_at,
    qdrant_graph_collection_gc,
    qdrant_inactive_gc,
    status as maintenance_status,
    weekly_cache_cleanup,
)
from .parsers.service_clients import mineru_health
from .pipeline.scheduler import ScanStats, requeue_failed_lifecycle_jobs, schedule_deletes_for_source, schedule_file, update_stats
from .pipeline.worker import run_once
from .vector.qdrant import GRAPH_VECTOR_TYPES
from .vector.qdrant import client as qdrant_client
from .vector.qdrant import collection_vector_layout, delete_collection, ensure_collection
from .vector.qdrant import ensure_graph_collection, graph_collection_alias, graph_collection_name


def cmd_config(args: argparse.Namespace) -> int:
    settings = load_settings(args.env_file)
    data = {
        "state_db": str(settings.state_db),
        "opensearch_url": settings.opensearch_url,
        "runtime_dir": str(settings.runtime_dir),
        "cache_dir": str(settings.cache_dir),
        "log_dir": str(settings.log_dir),
        "qdrant_url": settings.qdrant_url,
        "qdrant_api_key": "***" if settings.qdrant_api_key else "",
        "embedding_base_url": settings.embedding_base_url,
        "embedding_model_id": settings.embedding_model_id,
        "embedding_dim": settings.embedding_dim,
        "embedding_batch": settings.embedding_batch,
        "embedding_retry": settings.embedding_retry,
        "embedding_sleep_seconds": settings.embedding_sleep_seconds,
        "embedding_api_key": "***" if settings.embedding_api_key else "",
        "visual_embedding_enabled": settings.visual_embedding_enabled,
        "visual_embedding_base_url": settings.visual_embedding_base_url,
        "visual_embedding_model_id": settings.visual_embedding_model_id,
        "visual_embedding_dim": settings.visual_embedding_dim,
        "visual_embedding_concurrency": settings.visual_embedding_concurrency,
        "image_max_pixels": settings.image_max_pixels,
        "vector_layout": settings.vector_layout.sizes(),
        "vlm_base_url": settings.vlm_base_url,
        "vlm_model_id": settings.vlm_model_id,
        "vlm_api_key": "***" if settings.vlm_api_key else "",
        "vlm_concurrency": settings.vlm_concurrency,
        "vlm_temperature": settings.vlm_temperature,
        "vlm_top_p": settings.vlm_top_p,
        "vlm_max_tokens": settings.vlm_max_tokens,
        "vlm_structured_output": settings.vlm_structured_output,
        "parse_enabled": settings.parse_enabled,
        "min_file_age_seconds": settings.min_file_age_seconds,
        "metadata_job_lease_seconds": settings.metadata_job_lease_seconds,
        "parse_job_lease_seconds": settings.parse_job_lease_seconds,
        "job_max_retries": settings.job_max_retries,
        "job_retry_base_seconds": settings.job_retry_base_seconds,
        "job_retry_max_seconds": settings.job_retry_max_seconds,
        "qdrant_upsert_max_bytes": settings.qdrant_upsert_max_bytes,
        "qdrant_inactive_retention_days": settings.qdrant_inactive_retention_days,
        "graph_work_dir": str(settings.graph_work_dir),
        "graph_llm_concurrency": settings.graph_llm_concurrency,
        "graph_llm_timeout_seconds": settings.graph_llm_timeout_seconds,
        "graph_circuit_fails": settings.graph_circuit_fails,
        "graph_gc_retention_days": settings.graph_gc_retention_days,
        "neo4j_graph_retention_days": settings.neo4j_graph_retention_days,
        "neo4j_uri": settings.neo4j_uri,
        "neo4j_user": settings.neo4j_user,
        "neo4j_password": "***" if settings.neo4j_password else "",
        "graph_neo4j_import_after_build": settings.graph_neo4j_import_after_build,
        "graph_neo4j_import_batch_size": settings.graph_neo4j_import_batch_size,
        "mineru_url": settings.mineru_url,
        "sources": {
            key: {
                "collection": src.collection,
                "source_root": src.source_root,
                "max_tokens": src.max_tokens,
                "overlap_tokens": src.overlap_tokens,
                "graph_enabled": src.graph_enabled,
                "graph_unit_chunks": src.graph_unit_chunks,
                "graph_max_gleanings": src.graph_max_gleanings,
                "graph_rebuild_policy": {
                    "interval_days": src.graph_rebuild_policy.interval_days,
                    "new_chunk_ratio": src.graph_rebuild_policy.new_chunk_ratio,
                    "operator": src.graph_rebuild_policy.operator,
                },
            }
            for key, src in settings.sources.items()
        },
    }
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


def cmd_init_db(args: argparse.Namespace) -> int:
    settings = load_settings(args.env_file)
    db.init_db(settings.state_db)
    print(f"initialized state db: {settings.state_db}")
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    settings = load_settings(args.env_file)
    db.init_db(settings.state_db)
    stats = ScanStats()
    if args.source and args.source not in settings.sources:
        raise ValueError(f"unknown source {args.source!r}; available: {', '.join(settings.sources)}")
    source_keys = [args.source] if args.source else list(settings.sources)

    hash_content = os.getenv("KB_LOCAL_SCAN_HASH", "1").strip().lower() in {"1", "true", "yes", "on"}
    rehash_all = bool(getattr(args, "rehash", False)) or os.getenv(
        "KB_SCAN_REHASH_ALL", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    # Checksum reuse relies on (size, mtime). Writes that preserve mtime (touch -r, archive syncs with
    # --archive) bypass that criterion, so changed content would never be re-parsed. A periodic full
    # recompute covers this blind spot: every 7 days by default, with a marker file recording the time.
    rehash_marker = settings.runtime_dir / "state" / "last_full_rehash"
    rehash_hours = int(os.getenv("KB_SCAN_REHASH_EVERY_HOURS", "168"))
    if hash_content and not rehash_all and rehash_hours > 0 and not args.source:
        try:
            due = (not rehash_marker.exists()
                   or time.time() - rehash_marker.stat().st_mtime > rehash_hours * 3600)
        except OSError:
            due = False
        if due:
            rehash_all = True
            print(f"[scan] periodic full rehash (every {rehash_hours}h): checksum reuse disabled this run")

    # Reuse recorded checksums for files whose size and mtime are unchanged.
    # rsync decides what reaches the mirror on exactly that check, so this adds
    # no blind spot the sync does not already have. --rehash forces a full pass
    # for periodic deep verification.
    known_by_source: dict[str, dict[str, tuple[int, int, str]]] = {}
    if hash_content and not rehash_all:
        with db.connect(settings.state_db) as known_con:
            for key in source_keys:
                kb_id = settings.sources[key].kb_id
                known_by_source[key] = {
                    str(row["rel_path"]): (
                        int(row["size"]),
                        int(row["mtime"]),
                        str(row["checksum"] or ""),
                    )
                    for row in known_con.execute(
                        "SELECT rel_path, size, mtime, checksum FROM files WHERE kb_id = ?",
                        (kb_id,),
                    )
                }
    elif rehash_all:
        print("[scan] rehash requested; checksum reuse disabled for this run")

    scanned_sources = []
    for key in source_keys:
        source = settings.sources[key]
        recent_files = list_recent_source_files(
            source,
            min_age_seconds=settings.min_file_age_seconds,
            limit=5,
        )
        if recent_files:
            stats.recent_pending += len(recent_files)
            print(
                f"[scan] source={discovery.kb_label(source.kb_id, source.source_root)} recent_pending>={len(recent_files)} "
                f"min_age={settings.min_file_age_seconds}s"
            )
            if args.verbose:
                for path in recent_files:
                    print(f"  recent_pending   {path}")
        too_recent_keys: set[int] = set()
        files = list_source_files(
            source,
            limit=args.limit,
            min_age_seconds=settings.min_file_age_seconds,
            hash_content=hash_content,
            known_files=known_by_source.get(key),
            too_recent_keys=too_recent_keys,
        )
        print(f"[scan] source={discovery.kb_label(source.kb_id, source.source_root)} collection={source.collection} files={len(files)}")
        if too_recent_keys:
            print(f"[scan] source={discovery.kb_label(source.kb_id, source.source_root)} too_recent={len(too_recent_keys)} (deferred, counted as present)")
        # Files still inside the settle window are present on disk; counting
        # them as seen keeps delete detection (and move detection) from reading
        # a freshly edited file as gone.
        current_seen_ids = {file.file_key for file in files} | too_recent_keys
        current_checksum_counts: dict[str, int] = {}
        for file in files:
            if file.checksum:
                current_checksum_counts[str(file.checksum)] = current_checksum_counts.get(str(file.checksum), 0) + 1
        scanned_sources.append((key, source, files, current_seen_ids, current_checksum_counts, too_recent_keys))

    with db.connect(settings.state_db) as state_con:
        run_id = "dry-run" if args.dry_run else db.begin_run(state_con, "scan", note="manual scan")

        # ── KB lifecycle: register what is present, surface what vanished ──
        # A KB whose top-level directory disappeared is handled exactly like
        # its files would be: every file goes through the delete queue (worker
        # marks points inactive, drops OpenSearch rows), the KB row flips to
        # inactive, and the daily GC drops the collection after the retention
        # window. Nothing is deleted on sight; a returning directory flips the
        # row back and files reactivate via the normal restore path.
        present_kb_ids = {source.kb_id for _, source, *_ in scanned_sources}
        vanished: list = []
        skipped_kb_ids: set[str] = set()
        if not args.source and args.limit is None:
            # "Everything vanished at once" is indistinguishable from a missing
            # or unmounted mirror root. The file-level delete circuit breaker has been removed (see
            # schedule_deletes_for_source), but tearing down a whole KB is another matter: it drops the
            # Qdrant collection, the graph and the OpenSearch index together, and rebuilding the graph
            # costs real LLM money that a 7-day retention period cannot bring back. So this one stays --
            # it guards the knowledge base lifecycle, not file deletion.
            active_known = [
                row for row in discovery.known_sources(state_con) if str(row["status"]) == "active"
            ]
            mirror_missing = not settings.mirror_root.is_dir()
            all_vanished = bool(active_known) and not present_kb_ids
            if mirror_missing or (all_vanished and not args.force_kb_teardown):
                reason = (
                    f"mirror root missing: {settings.mirror_root}"
                    if mirror_missing
                    else f"all {len(active_known)} known KBs vanished at once (empty mirror?)"
                )
                print(
                    f"[scan] REFUSING KB lifecycle pass: {reason}; "
                    "no KB is marked vanished. Fix the mirror, unenroll from the "
                    "console, or rerun with --force-kb-teardown if this is intentional.",
                    file=sys.stderr,
                    flush=True,
                )
                if not args.dry_run:
                    db.finish_run(
                        state_con,
                        run_id,
                        added_count=0,
                        updated_count=0,
                        moved_count=0,
                        deleted_count=0,
                        failed_count=0,
                    )
                return 2
            if not args.dry_run:
                for _, source, *_ in scanned_sources:
                    try:
                        # Enrollment (row creation, collection/index setup,
                        # reactivation) happens in the web console; the scan
                        # only refreshes last_seen and never flips status --
                        # a directory merely being present must not undo an
                        # unenroll.
                        discovery.touch_seen(state_con, source)
                    except (discovery.CollectionMismatch, sqlite3.IntegrityError) as exc:
                        print(f"[scan] kb skipped: {source.kb_id} ({source.source_root}): {exc}", file=sys.stderr, flush=True)
                        skipped_kb_ids.add(source.kb_id)
                        continue
            for row in discovery.known_sources(state_con):
                if row["kb_id"] in present_kb_ids:
                    continue
                # Present on disk but refused by the link policy (third review, item 1): that is not "vanished".
                # Nothing is scanned, deactivated or queued for deletion; the console shows directory_linked
                # until the link is replaced by a directory or KB_MIRROR_ALLOW_LINKED_DIRS is set.
                if str(row["status"]) == "active" and discovery.directory_admitted(settings.mirror_root, str(row["source_root"])) == (False, "linked"):
                    print(f"[scan] kb refused: {row['kb_id']} ({row['source_root']}) is a symbolic link; left untouched")
                    continue
                try:
                    gone = discovery.source_from_row(settings.mirror_root, row)
                except Exception as exc:
                    # A base whose configuration cannot be read (edited by hand, old data): enrolled_sources
                    # already skips it, and here too only this one base is skipped. A directory that is still
                    # there has not "vanished": it is neither deactivated nor queued for deletion. When the
                    # directory is really gone the source is built with an empty configuration, just to
                    # address the deletes by id
                    if str(row["status"]) == "active" and discovery.directory_admitted(settings.mirror_root, str(row["source_root"]))[0]:
                        print(f"[scan] kb skipped: {row['kb_id']} ({row['source_root']}): bad config: {exc!r}", file=sys.stderr, flush=True)
                        skipped_kb_ids.add(str(row["kb_id"]))
                        continue
                    gone = discovery.build_source(settings.mirror_root, str(row["source_root"]), {},
                                                  kb_id=str(row["kb_id"]), collection=str(row["collection"]))
                if str(row["status"]) == "active":
                    print(f"[scan] kb vanished: {gone.kb_id} ({gone.source_root}); marking inactive, queueing deletes dry_run={args.dry_run}")
                    if not args.dry_run:
                        discovery.mark_inactive(state_con, gone.kb_id)
                vanished.append(gone)
            # Directory rename detection (2026-09-06): when a KB deactivated because its directory vanished
            # and an unenrolled directory with matching content appears on disk, treat it as a rename -- the
            # id, collection, graph and cache are all kept, only the directory changes; the next scan
            # recognizes files by the new path, and unchanged content only gets the path in its payload
            # refreshed, without re-parsing. The very round that marked it inactive above can already
            # recognize it.
            for report in discovery.find_renamed_directories(state_con, settings.mirror_root):
                print(f"[scan] kb renamed: {discovery.kb_label(report['kb_id'], report['old_dir'])} -> {report['dir']} "
                      f"(matched {report['matched']}/{report['total']}, verified {report['verified']}) dry_run={args.dry_run}")
                if not args.dry_run:
                    discovery.adopt_directory(state_con, settings.mirror_root, report["kb_id"], report["dir"])
                    vanished = [g for g in vanished if g.kb_id != report["kb_id"]]    # no delete jobs for it anymore
        # vanished KBs ride the same per-source delete scheduling below with an
        # empty seen-set.
        # The placeholder tuple must match the shape of scanned_sources above (6 fields, the last being
        # too_recent_keys): one field short and the unpacking below crashes -- on 2026-09-06 a closed KB
        # made the scan fail every minute, and new files of the other KBs could not get in either
        scanned_sources = list(scanned_sources) + [
            (f"vanished:{g.kb_id}", g, [], set(), {}, set()) for g in vanished
        ]

        for key, source, files, current_seen_ids, current_checksum_counts, too_recent_keys in scanned_sources:
            if source.kb_id in skipped_kb_ids:
                continue
            # Deferred-but-present files seed the seen set; schedule_file may
            # still re-key a moved file below, which is why this is not simply
            # current_seen_ids.
            seen_ids: set[int] = set(too_recent_keys)
            for file in files:
                change_type, job_id = schedule_file(
                    state_con,
                    ingest_run_id=run_id,
                    file=file,
                    current_seen_file_keys=current_seen_ids,
                    current_checksum_counts=current_checksum_counts,
                    requeue_failed=args.requeue_failed,
                    dry_run=args.dry_run,
                )
                seen_ids.add(file.file_key)
                update_stats(stats, change_type, job_id)
                # Unchanged files are not listed one by one: a round every minute with a line per file made up
                # more than nine tenths of the user journal (2026-09-29 audit); add --list-unchanged to see all
                if args.verbose and (change_type != "unchanged" or getattr(args, "list_unchanged", False)):
                    print(f"  {change_type:16s} {file.source_path}")
            if not args.no_detect_deletes and args.limit is None:
                deleted, delete_jobs = schedule_deletes_for_source(
                    state_con,
                    ingest_run_id=run_id,
                    kb_id=source.kb_id,
                    collection=source.collection,
                    seen_file_keys=seen_ids,
                    # A vanished KB's directory is gone by definition; for a live
                    # scan, re-check the disk before tearing anything down.
                    verify_physical=not key.startswith("vanished:"),
                    dry_run=args.dry_run,
                )
                stats.deleted += deleted
                stats.jobs += delete_jobs
                if deleted:
                    print(f"[scan] source={discovery.kb_label(source.kb_id, source.source_root)} deleted={deleted} delete_jobs={delete_jobs} dry_run={args.dry_run}")
        if not args.source and args.limit is None:
            requeued = requeue_failed_lifecycle_jobs(state_con, ingest_run_id=run_id, dry_run=args.dry_run)
            if requeued:
                stats.jobs += requeued
                print(f"[scan] requeued_failed_lifecycle_jobs={requeued} dry_run={args.dry_run}")
        if not args.dry_run:
            db.finish_run(
                state_con,
                run_id,
                added_count=stats.added,
                updated_count=stats.content_changed + stats.parser_changed + stats.metadata_changed + stats.needs_parse,
                moved_count=stats.metadata_changed,
                deleted_count=stats.deleted,
                failed_count=stats.parse_failed,
            )
    print(
        "scan summary: "
        f"seen={stats.seen} added={stats.added} content_changed={stats.content_changed} parser_changed={stats.parser_changed} "
        f"metadata_changed={stats.metadata_changed} needs_parse={stats.needs_parse} "
        f"parse_failed={stats.parse_failed} deleted={stats.deleted} unchanged={stats.unchanged} jobs={stats.jobs} "
        f"recent_pending={stats.recent_pending} "
        f"dry_run={args.dry_run}"
    )
    if hash_content and rehash_all and not args.dry_run and not args.source:
        try:
            rehash_marker.parent.mkdir(parents=True, exist_ok=True)
            rehash_marker.write_text(str(int(time.time())), encoding="utf-8")
        except OSError:
            pass
    if stats.recent_pending and args.exit_code_on_recent:
        return 75
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    settings = load_settings(args.env_file)
    db.init_db(settings.state_db)
    with db.connect(settings.state_db) as con:
        file_counts = con.execute("SELECT kb_id, status, COUNT(*) AS c FROM files GROUP BY kb_id, status ORDER BY kb_id, status").fetchall()
        job_counts = con.execute("SELECT status, job_type, COUNT(*) AS c FROM jobs GROUP BY status, job_type ORDER BY status, job_type").fetchall()
        unindexed_files = con.execute(
            """
            SELECT file_id, kb_id, source_path, indexed_version, content_version, status
            FROM files
            WHERE status != 'deleted' AND COALESCE(indexed_version, '') != content_version
            ORDER BY last_seen_at DESC
            LIMIT 20
            """
        ).fetchall()
        unindexed_count = con.execute(
            """
            SELECT COUNT(*) AS c
            FROM files
            WHERE status != 'deleted' AND COALESCE(indexed_version, '') != content_version
            """
        ).fetchone()
        print("files:")
        for row in file_counts:
            print(f"  {row['kb_id']} {row['status']} {row['c']}")
        print("jobs:")
        for row in job_counts:
            print(f"  {row['status']} {row['job_type']} {row['c']}")
        total_unindexed = int(unindexed_count["c"] if unindexed_count else 0)
        if total_unindexed:
            print(f"unindexed_current_files: {total_unindexed}")
            for row in unindexed_files:
                print(f"  {row['kb_id']} {row['status']} {row['file_id']} {row['source_path']}")
    return 0


def cmd_worker(args: argparse.Namespace) -> int:
    settings = load_settings(args.env_file)
    if args.source and args.source not in settings.sources:
        raise ValueError(f"unknown source {args.source!r}; available: {', '.join(settings.sources)}")
    db.init_db(settings.state_db)
    with db.connect(settings.state_db) as con:
        if args.once:
            # With --max-jobs>1, run several jobs in-process: starting a new interpreter per job costs
            # 1-3 seconds of startup (plus rebuilding the Qdrant/HTTP client connection pools), which adds
            # up on a large queue. Stop when no job can be claimed, same semantics as the original --once.
            budget = max(1, int(getattr(args, "max_jobs", 1) or 1))
            # The time-budget checkpoint must sit **between jobs**. systemd's TimeoutStartSec is a hard
            # kill from outside, and where it lands is pure luck -- on 2026-08-25 it landed in the middle
            # of one file's MinerU parse, wasting those 12 minutes of compute (the parse cache is only
            # written once the whole document is done; a mid-way kill leaves an empty directory). Here we
            # stop voluntarily, always on a file boundary.
            max_seconds = float(getattr(args, "max_seconds", 0) or 0)
            deadline = time.monotonic() + max_seconds if max_seconds > 0 else None
            done = 0
            while True:
                result = run_once(con, settings=settings, dry_run=args.dry_run, source_key=args.source)
                print(result)
                done += 1
                if result == "no-job" or done >= budget:
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    # Clean exit, leaving the queue to the next timer round. Not an error, so it must not
                    # take the failure path.
                    print(f"time-budget-reached done={done}", flush=True)
                    break
            return 0
        raise SystemExit("continuous worker is intentionally not enabled before first parsing approval")


def cmd_health(args: argparse.Namespace) -> int:
    settings = load_settings(args.env_file)
    checks: dict[str, object] = {}
    checks["env_file"] = str(settings.env_file)
    checks["state_db_parent_exists"] = settings.state_db.parent.exists()
    checks["cache_dir_parent_exists"] = settings.cache_dir.parent.exists()
    checks["qdrant_key_present"] = bool(settings.qdrant_api_key)
    try:
        q = qdrant_client(settings.qdrant_url, settings.qdrant_api_key)
        existing = {item.name for item in q.get_collections().collections}
        collections: dict[str, object] = {}
        expected_layout = settings.vector_layout.sizes()
        for collection in sorted({source.collection for source in settings.sources.values()}):
            if collection not in existing:
                collections[collection] = {
                    "exists": False,
                    "expected_vectors": expected_layout,
                    "ok": False,
                }
                continue
            actual_layout = collection_vector_layout(q, collection)
            collections[collection] = {
                "exists": True,
                "vectors": actual_layout,
                "expected_vectors": expected_layout,
                "ok": actual_layout == expected_layout,
            }
        checks["qdrant"] = {"collections": collections}
    except Exception as exc:
        checks["qdrant"] = {"error": repr(exc)}
    try:
        checks["mineru"] = mineru_health(settings.mineru_url)
    except Exception as exc:
        checks["mineru"] = {"error": repr(exc)}
    endpoints: dict[str, object] = {
        "embedding": _vllm_health(settings.embedding_base_url),
        "vlm": _vllm_health(settings.vlm_base_url),
    }
    if settings.visual_embedding_enabled:
        endpoints["visual_embedding"] = _vllm_health(settings.visual_embedding_base_url)
    checks["endpoints"] = endpoints
    try:
        os_health = requests.get(
            f"{settings.opensearch_url.rstrip('/')}/_cluster/health", timeout=5
        ).json()
        checks["opensearch"] = {
            "status": os_health.get("status"),
            "ok": os_health.get("status") in {"green", "yellow"},
        }
    except Exception as exc:
        checks["opensearch"] = {"error": repr(exc), "ok": False}
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    return 0


def _vllm_health(base_url: str) -> dict[str, object]:
    """vLLM serves /health at the server root, one level above /v1."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    try:
        response = requests.get(f"{root}/health", timeout=5)
        return {"ok": response.status_code == 200, "status_code": response.status_code}
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def cmd_qdrant(args: argparse.Namespace) -> int:
    settings = load_settings(args.env_file)
    q = qdrant_client(settings.qdrant_url, settings.qdrant_api_key)
    for source in settings.sources.values():
        result = ensure_collection(q, source.collection, settings.vector_layout, dry_run=args.dry_run)
        print(result)
    return 0


def cmd_qdrant_graph(args: argparse.Namespace) -> int:
    settings = load_settings(args.env_file)
    q = qdrant_client(settings.qdrant_url, settings.qdrant_api_key)
    graph_types = args.graph_type or list(GRAPH_VECTOR_TYPES)
    source_keys = args.source
    if not source_keys and not args.collection and not args.all:
        source_keys = [key for key, src in settings.sources.items() if src.graph_enabled] or None
    selected = _selected_sources(
        settings,
        source_keys=source_keys,
        collections=args.collection,
        all_sources=args.all,
    )
    if not selected:
        raise ValueError("no sources selected")

    result = []
    for key, source in selected:
        for graph_type in graph_types:
            alias = graph_collection_alias(source.collection, graph_type)
            physical = graph_collection_name(source.collection, graph_type, args.graph_version)
            status = ensure_graph_collection(q, physical, settings.embedding_dim, dry_run=args.dry_run)
            result.append({
                "source": key,
                "kb_id": source.kb_id,
                "source_collection": source.collection,
                "graph_type": graph_type,
                "alias": alias,
                "collection": physical,
                "status": status,
            })
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_fts(args: argparse.Namespace) -> int:
    settings = load_settings(args.env_file)
    requested_collections = getattr(args, "collection", None)
    collections = [requested_collections] if isinstance(requested_collections, str) else _selected_collections(settings, requested_collections)
    if args.fts_command == "init":
        result = {
            "ensured": search_fts.ensure_indices(settings.opensearch_url, collections),
            "status": search_fts.status(settings.opensearch_url, collections),
        }
    elif args.fts_command == "status":
        result = search_fts.status(settings.opensearch_url, collections)
        if args.compare_qdrant:
            q = qdrant_client(settings.qdrant_url, settings.qdrant_api_key)
            result["compare_qdrant"] = search_fts.compare_with_qdrant(settings.opensearch_url, q, collections)
    elif args.fts_command == "rebuild":
        q = qdrant_client(settings.qdrant_url, settings.qdrant_api_key)
        result = search_fts.rebuild_from_qdrant(
            url=settings.opensearch_url,
            qdrant=q,
            collections=collections,
            batch_size=args.batch_size,
        )
        result["compare_qdrant"] = search_fts.compare_with_qdrant(settings.opensearch_url, q, collections)
    elif args.fts_command == "sync-doc":
        q = qdrant_client(settings.qdrant_url, settings.qdrant_api_key)
        result = search_fts.sync_doc_from_qdrant(
            url=settings.opensearch_url,
            qdrant=q,
            collection=args.collection,
            doc_id=args.doc_id,
            batch_size=args.batch_size,
        )
    elif args.fts_command == "search":
        result = {
            "query": args.query,
            "limit": args.limit,
            "results": search_fts.search(
                settings.opensearch_url,
                args.query,
                collections=collections,
                limit=args.limit,
            ),
        }
    else:
        raise ValueError(f"unknown fts command: {args.fts_command}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
def cmd_reset(args: argparse.Namespace) -> int:
    settings = load_settings(args.env_file)
    db.init_db(settings.state_db)
    selected = _selected_sources(settings, source_keys=args.source, collections=args.collection, all_sources=args.all)
    if not selected:
        raise ValueError("no source selected; use --source, --collection, or --all")

    kb_ids = [source.kb_id for _, source in selected]
    with db.connect(settings.state_db) as con:
        running = db.running_jobs_for_kbs(con, kb_ids)
        if running and not args.force:
            print("running jobs detected; use --force only after you are sure no worker is actually active:")
            for row in running:
                print(f"  {row['job_id']} {row['kb_id']} {row['collection']} {row['job_type']} locked_by={row['locked_by']}")
            return 2

        plan = []
        for key, source in selected:
            plan.append(
                {
                    "source": key,
                    "kb_id": source.kb_id,
                    "collection": source.collection,
                    "state_counts": db.kb_state_counts(con, source.kb_id),
                    "cache_dir": str(settings.cache_dir / "parse" / source.kb_id),
                    "purge_fts_collection": not args.keep_collection,
                }
            )
        print(json.dumps({"dry_run": not args.yes, "selected": plan}, ensure_ascii=False, indent=2))
        if not args.yes:
            print("no changes made; rerun with --yes to purge state/cache and reset Qdrant collection(s)")
            return 0

        q = qdrant_client(settings.qdrant_url, settings.qdrant_api_key)
        if not args.keep_collection:
            q.get_collections()
        reset_collections: set[str] = set()
        for key, source in selected:
            # After resetting the main KB the graph corpus is stale and every baseline point_id in
            # graph_builds is invalid: a KB whose rebuild is only interval-triggered would serve graph
            # retrieval inconsistent with the main KB for a whole cycle. Unless --keep-graph, tear down
            # the graph side as well.
            if not getattr(args, "keep_graph", False):
                from .maintenance import _drop_graph_data

                graph_errors: list[str] = []
                dropped = _drop_graph_data(settings, con, kb_id=source.kb_id,
                                           collection=source.collection, errors=graph_errors)
                print(f"graph-dropped:{key}:{json.dumps(dropped, ensure_ascii=False)}")
                for err in graph_errors:
                    print(f"graph-drop-error:{err}", file=sys.stderr)
            counts = db.purge_kb_state(con, source.kb_id)
            cache_dir = settings.cache_dir / "parse" / source.kb_id
            if cache_dir.exists() and not args.keep_cache:
                shutil.rmtree(cache_dir)
            print(f"state-purged:{key}:{source.kb_id}:{counts}")
            reset_collections.add(source.collection)

        if not args.keep_collection:
            for collection in sorted(reset_collections):
                print(delete_collection(q, collection))
                print(search_fts.delete_collection(settings.opensearch_url, collection=collection))
                if not args.no_recreate:
                    print(ensure_collection(q, collection, settings.vector_layout))
    return 0


def cmd_cleanup(args: argparse.Namespace) -> int:
    settings = load_settings(args.env_file)
    if args.cleanup_command == "status":
        result = maintenance_status(settings)
    elif args.cleanup_command == "weekly":
        result = weekly_cache_cleanup(settings, dry_run=args.dry_run)
    elif args.cleanup_command == "monthly":
        result = monthly_log_cleanup(settings, dry_run=args.dry_run)
    elif args.cleanup_command == "qdrant-gc":
        retention_days = args.retention_days if args.retention_days is not None else settings.qdrant_inactive_retention_days
        result = qdrant_inactive_gc(settings, retention_days=retention_days, dry_run=args.dry_run)
    elif args.cleanup_command == "qdrant-graph-gc":
        retention_days = args.retention_days if args.retention_days is not None else settings.graph_gc_retention_days
        result = qdrant_graph_collection_gc(
            settings,
            retention_days=retention_days,
            dry_run=args.dry_run,
            delete_unparseable=args.delete_unparseable,
        )
    elif args.cleanup_command == "neo4j-graph-gc":
        retention_days = args.retention_days if args.retention_days is not None else settings.neo4j_graph_retention_days
        result = neo4j_graph_gc(
            settings,
            retention_days=retention_days,
            dry_run=args.dry_run,
            delete_unparseable=args.delete_unparseable,
        )
    elif args.cleanup_command == "graph-gc":
        result = graph_gc(settings, keep_latest=args.keep_latest, dry_run=args.dry_run)
    elif args.cleanup_command == "parse-assets-gc":
        retention_days = args.retention_days if args.retention_days is not None else settings.qdrant_inactive_retention_days
        result = parse_assets_gc(settings, retention_days=retention_days, dry_run=args.dry_run)
        if result.get("skipped"):
            result["kb_sources_gc"] = {"skipped": True, "reason": "parse-assets-gc skipped"}
        else:
            result["kb_sources_gc"] = kb_sources_gc(settings, retention_days=retention_days, dry_run=args.dry_run)
    elif args.cleanup_command == "qdrant-backfill-inactive-at":
        inactive_at_ts = args.inactive_at_ts
        if inactive_at_ts is None:
            inactive_at_ts = int(time.time()) - max(0, args.days_ago) * 24 * 3600
        result = qdrant_backfill_inactive_at(settings, inactive_at_ts=inactive_at_ts, dry_run=args.dry_run)
    else:
        raise ValueError(f"unknown cleanup command: {args.cleanup_command}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if isinstance(result, dict) and result.get("errors"):
        # This used to exit 0 whether or not errors was empty, so systemd always showed success and an
        # undeletable collection / projection gave no signal at all.
        print(f"[cleanup] finished with {len(result['errors'])} failed step(s)", file=sys.stderr, flush=True)
        return 1
    if isinstance(result, dict) and result.get("skipped") and result.get("yielded_to") == "graph_build":
        # The graph-version safety net yields to a running build: that build's own clean-up covers this round
        print("[cleanup] skipped: a graph build is running (its own clean-up covers this round); exiting 75",
              file=sys.stderr, flush=True)
        return 75
    if isinstance(result, dict) and result.get("skipped"):
        # Being blocked by service_busy is not success. Exit 75 so the systemd unit's Restart=on-failure
        # retries after 15 minutes -- otherwise one collision would mean the whole maintenance round of
        # the day (or week / month) is silently skipped, and Persistent does not make it up either.
        print("[cleanup] skipped because the service was busy; exiting 75 so the unit retries",
              file=sys.stderr, flush=True)
        return 75
    return 0


def _rebuild_blocked(settings, source) -> dict | None:
    """Whether the automatic rebuild should yield. Returns the reason when it should, None otherwise.

    The criterion is exactly the console's "build / rebuild graph now" (db.kb_parse_busy): only whether
    **this knowledge base itself** still has parse jobs in flight -- building while the corpus is still
    growing yields half a graph.

    This used to use the global service_busy: any running job of any KB, any worker process, any of the
    five locks, even a non-empty MinerU queue, all made it yield. So during the hours of loading one big
    KB, the other KBs that had long finished parsing never got a turn, while this check only ran once a
    day. The embedding concurrency budget is partitioned anyway (pipeline 20 + graph build 10 + 2
    reserved for retrieval = max_num_seqs 32), so there is no need to yield to each other.

    The graph build lock is the only global condition kept: it serializes graph builds by design (one at
    a time), which is queuing rather than interference -- after yielding, the unit's retry picks it up.
    """
    with db.connect(settings.state_db) as con:
        pending = db.kb_parse_busy(con, source.kb_id)
    if pending:
        return {"reason": "kb parsing", "kb_id": source.kb_id, "parse_jobs": pending, "retry": True}
    lock = build_lock_path(settings)
    if build_lock_held(lock):                 # flock probe: only a live holder counts (see graph/lock.py)
        return {"reason": "another graph build is running", "lock": str(lock), "retry": True}
    clear_lock_leftovers(lock)                # display-only pid / started_at left by a SIGKILLed predecessor
    return None




def cmd_search(args: argparse.Namespace) -> int:
    """Regression evaluation and question-set generation for the search service (kb_search); the service
    code itself lives in app/kb_search."""
    from kb_search import evalset

    if args.search_command == "eval":
        return evalset.cli_eval(args)
    if args.search_command == "make-set":
        return evalset.cli_make_set(args)
    raise SystemExit(f"unknown search command: {args.search_command}")


def cmd_graph(args: argparse.Namespace) -> int:
    settings = load_settings(args.env_file)
    db.init_db(settings.state_db)

    if args.graph_command == "status":
        with db.connect(settings.state_db) as con:
            rows = con.execute(
                """
                SELECT source_key, kb_id, source_collection, graph_version, status,
                       started_at, finished_at, input_rows, active_chunk_count,
                       active_doc_count, output_dir, error
                FROM graph_builds
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (args.limit,),
            ).fetchall()
        print(json.dumps([dict(row) for row in rows], ensure_ascii=False, indent=2))
        return 0

    selected = _selected_sources(
        settings,
        source_keys=getattr(args, "source", None),
        collections=getattr(args, "collection", None),
        all_sources=getattr(args, "all", False),
    )
    if not selected:
        selected = [(key, source) for key, source in settings.sources.items() if source.graph_enabled]
    if not selected:
        # The scheduled rebuild-policy check idling when "no KB has graph building enabled" is normal, not
        # an error; a mistyped --source/--collection already raised earlier in _selected_sources.
        # Both are diagnostic commands: idling when no KB has the graph enabled is a normal result, not
        # an error
        if args.graph_command in ("check-rebuild", "append"):
            print(json.dumps([], ensure_ascii=False, indent=2))
            return 0
        raise ValueError("no graph-enabled source selected")

    if args.graph_command == "build":
        if len(selected) != 1 and args.graph_version:
            raise ValueError("--graph-version can only be used with a single selected source")
        results = []
        for key, source in selected:
            results.append(
                build_graph(
                    settings,
                    source_key=key,
                    source=source,
                    graph_version=args.graph_version,
                    dry_run=args.dry_run,
                    activate_aliases=not args.no_activate_aliases,
                    allow_existing_graph_version=args.allow_existing_graph_version,
                    run_gc=not args.no_graph_gc,
                )
            )
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    if args.graph_command == "append":
        # Manual "append new content": ignores the auto toggle; --force appends once even without changes
        # (to verify replay and vector reuse)
        results = []
        deferred = False
        for key, source in selected:
            decision = evaluate_append(settings, source_key=key, source=source, ignore_auto_flag=True)
            if decision.get("due") or getattr(args, "force", False):
                blocked = _rebuild_blocked(settings, source)
                if blocked:
                    decision["build_skipped"] = blocked
                    deferred = True
                else:
                    decision["build"] = build_graph(settings, source_key=key, source=source, incremental=True)
            results.append(decision)
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 75 if deferred else 0

    if args.graph_command == "check-rebuild":
        # Each round (every 2 hours), for every graph-enabled KB: when the full-rebuild condition is met,
        # rebuild from scratch; when it is not, the corpus has document-level changes relative to the
        # current version, and auto-append is on -> incremental append. Both yield while this KB is
        # parsing / the graph build lock is held.
        results = []
        had_errors = False
        stop_requested = False
        force_full = bool(getattr(args, "force_full", False))
        if force_full and (not args.execute or len(selected) != 1):
            raise ValueError("--force-full requires --execute and exactly one --source")
        for key, source in selected:
            try:
                if force_full:
                    decision = {"source": key, "kb_id": source.kb_id, "collection": source.collection,
                                "graph_enabled": True, "due": True, "reason": "forced_by_operator"}
                else:
                    decision = evaluate_rebuild(settings, source_key=key, source=source)
                decision["dir"] = source.source_root      # for the logs: directory name next to the KB id

                def run_full(decision=decision, key=key, source=source, resuggest=False):
                    not_ready = llm_ready(settings, source)
                    if not_ready:
                        # No model selected: retrying is pointless, skip quietly (not a yield)
                        decision["build_skipped"] = {"reason": "llm_not_configured", "detail": not_ready}
                        return
                    blocked = None if args.dry_run else _rebuild_blocked(settings, source)
                    if blocked:
                        decision["build_skipped"] = blocked
                        return
                    if resuggest and not args.dry_run:
                        # A threshold-triggered full rebuild first re-extracts a label version on top of the
                        # current version + endpoint ledger and activates it (requested by the user on
                        # 2026-09-08); if extraction fails (model fault), build with the current version
                        # rather than letting the full rebuild get stuck at this step
                        from .graph.schema_flow import resuggest_for_rebuild

                        try:
                            source, info = resuggest_for_rebuild(settings, source)
                        except Exception as exc:
                            info = {"error": repr(exc)}
                            print(f"[graph] {key}: schema resuggest failed, building with the current version: {exc!r}",
                                  file=sys.stderr, flush=True)
                        decision["schema_resuggest"] = info
                    decision["build"] = build_graph(
                        settings, source_key=key, source=source, dry_run=args.dry_run,
                        allow_existing_graph_version=False,
                    )

                if args.execute and decision.get("due"):
                    # The first build (no successful version yet) does not re-extract: that label version is
                    # usually the one the user just extracted; without one, the build fills it in itself
                    run_full(resuggest=decision.get("reason") != "no_successful_build")
                elif args.execute:
                    append = evaluate_append(settings, source_key=key, source=source)
                    decision["append"] = append
                    if append.get("due"):
                        blocked = None if args.dry_run else _rebuild_blocked(settings, source)
                        if blocked:
                            decision["build_skipped"] = blocked
                        else:
                            append["build"] = build_graph(
                                settings, source_key=key, source=source, dry_run=args.dry_run, incremental=True)
                    elif append.get("reason") == "config_changed_needs_full_rebuild" and append.get("changed") \
                            and source.graph_auto_append:
                        # The configuration changed and new documents arrived: the old extraction and
                        # resolution cannot be reused, so a full rebuild is the only option -- otherwise
                        # the new documents would never enter the graph
                        decision["due"] = True
                        decision["reason"] = "config_changed_with_new_content"
                        run_full()
            except Exception as exc:
                had_errors = True
                decision = {
                    "source": key,
                    "kb_id": source.kb_id,
                    "collection": source.collection,
                    "error": repr(exc),
                }
                # The process was asked to stop (pause, base closed, service stopped): record this base and
                # finish, do not go on to build the next one
                stop_requested = isinstance(exc, (GraphBuildInterrupted, LLMInterrupted))
            results.append(decision)
            if args.execute and not args.dry_run:
                # Record one row with this round's decision for this KB; the console status card uses it to say
                # "config changed / appended / yielded" (health check D7)
                from .graph.build import check_kind

                append_info = decision.get("append") or {}
                skipped = decision.get("build_skipped") or append_info.get("build_skipped") or {}
                try:
                    with db.connect(settings.state_db) as con:
                        db.record_graph_check(con, source.kb_id, {
                            "kind": check_kind(decision), "due": bool(decision.get("due")),
                            "reason": decision.get("reason"), "append_reason": append_info.get("reason"),
                            "skipped_reason": skipped.get("reason"), "error": decision.get("error"),
                        })
                except Exception as record_exc:
                    print(f"[graph] record check failed for {key}: {record_exc!r}", file=sys.stderr, flush=True)
            if stop_requested:
                print(f"[graph] stop requested while building {key}; skipping the remaining sources", file=sys.stderr, flush=True)
                break
        print(json.dumps(results, ensure_ascii=False, indent=2))
        if had_errors:
            return 1
        # For KBs that could not start building because this KB is parsing / the graph build lock is held,
        # exit 75 so the unit records a yield (see kb-maint-defer.sh); skips where a retry would not help,
        # such as no model selected, do not count.
        deferred = [d for d in results if isinstance(d, dict) and (d.get("build_skipped") or {}).get("retry")]
        if deferred:
            why = ", ".join(sorted({str(d["build_skipped"].get("reason")) for d in deferred}))
            print(f"[graph] rebuild deferred ({why}); exiting 75 so the unit retries",
                  file=sys.stderr, flush=True)
            return 75
        return 0

    if args.graph_command == "adopt-current":
        if len(selected) != 1 and args.graph_version:
            raise ValueError("--graph-version can only be used with a single selected source")
        results = [
            adopt_current_graph(
                settings,
                source_key=key,
                source=source,
                graph_version=args.graph_version,
            )
            for key, source in selected
        ]
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    if args.graph_command == "rollback":
        if len(selected) != 1:
            raise ValueError("rollback requires exactly one selected source (--source / --collection)")
        (key, source), = selected
        result = rollback_graph_version(settings, source_key=key, source=source, graph_version=args.graph_version,
                                        force=bool(getattr(args, "force", False)))
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0

    if args.graph_command == "neo4j-import":
        if len(selected) != 1 and args.graph_version:
            raise ValueError("--graph-version can only be used with a single selected source")
        results = [
            import_graph_to_neo4j(
                settings,
                source_key=key,
                source=source,
                graph_version=args.graph_version,
                replace=args.replace,
                activate=not args.no_activate,
                dry_run=args.dry_run,
                batch_size=args.batch_size,
                acquire_lock=True,
            )
            for key, source in selected
        ]
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    if args.graph_command == "query":
        from .graph.recall import graph_query

        if len(selected) != 1:
            raise ValueError("query requires exactly one selected source")
        _, source = selected[0]
        result = graph_query(settings, source, args.question, hops=args.hops, chunk_limit=args.limit)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.graph_command == "factcheck":
        from .graph.build import graph_paths
        from .graph.factcheck import factcheck, factcheck_markdown, load_gold

        if len(selected) != 1:
            raise ValueError("factcheck requires exactly one selected source")
        key, source = selected[0]
        version = args.graph_version
        if not version:
            with db.connect(settings.state_db) as con:
                latest = db.latest_successful_graph_build(con, key)
            if latest is None:
                raise ValueError("this knowledge base has no completed graph")
            version = str(latest["graph_version"])
        graph_file = graph_paths(settings, source, version).graph_file
        graph = json.loads(graph_file.read_text(encoding="utf-8"))
        report = factcheck(graph, load_gold(Path(args.gold)))
        report["kb_id"] = source.kb_id
        if args.out:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            out.with_suffix(".md").write_text(factcheck_markdown(report, source.kb_id), encoding="utf-8")
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
        return 0

    if args.graph_command == "neo4j-status":
        if len(selected) != 1 and args.graph_version:
            raise ValueError("--graph-version can only be used with a single selected source")
        results = [
            neo4j_status(
                settings,
                source=source,
                graph_version=args.graph_version,
            )
            for _, source in selected
        ]
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    if args.graph_command == "neo4j-delete":
        if len(selected) != 1 or not args.graph_version:
            raise ValueError("neo4j-delete requires one selected source and --graph-version")
        key, source = selected[0]
        result = delete_neo4j_graph_version(
            settings,
            source=source,
            graph_version=args.graph_version,
            dry_run=args.dry_run,
        )
        result["source"] = key
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    raise ValueError(f"unknown graph command: {args.graph_command}")


def _selected_sources(settings, *, source_keys: list[str] | None, collections: list[str] | None, all_sources: bool):
    if all_sources:
        return list(settings.sources.items())
    selected: list[tuple[str, object]] = []
    for key in source_keys or []:
        if key not in settings.sources:
            raise ValueError(f"unknown source {key!r}; available: {', '.join(settings.sources)}")
        selected.append((key, settings.sources[key]))
    for collection in collections or []:
        matches = [(key, source) for key, source in settings.sources.items() if source.collection == collection]
        if not matches:
            raise ValueError(f"collection {collection!r} is not configured in current sources")
        selected.extend(matches)
    seen: set[str] = set()
    unique = []
    for key, source in selected:
        if source.kb_id in seen:
            continue
        seen.add(source.kb_id)
        unique.append((key, source))
    return unique


def _selected_collections(settings, requested: list[str] | None = None) -> list[str]:
    if requested:
        return list(dict.fromkeys(requested))
    return list(dict.fromkeys(source.collection for source in settings.sources.values()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kb")
    parser.add_argument("--env-file", default=None, help="Path to .env file")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("config").set_defaults(func=cmd_config)
    sub.add_parser("init-db").set_defaults(func=cmd_init_db)

    scan = sub.add_parser("scan")
    scan.add_argument("--dry-run", action="store_true")
    scan.add_argument("--limit", type=int, default=None)
    scan.add_argument("--source", default=None, help="Source key; see `kb config`")
    scan.add_argument("--no-detect-deletes", action="store_true", help="Skip delete detection for this scan")
    scan.add_argument(
        "--force-kb-teardown",
        action="store_true",
        help="Approve the KB lifecycle pass when every enrolled directory vanished at once",
    )
    scan.add_argument(
        "--requeue-failed",
        action="store_true",
        help="Requeue current-version parse jobs even when an earlier terminal failed job exists",
    )
    scan.add_argument(
        "--exit-code-on-recent",
        action="store_true",
        help="Return 75 when supported files are still younger than KB_MIN_FILE_AGE_SECONDS",
    )
    scan.add_argument("--verbose", action="store_true", help="List every file whose state changed")
    scan.add_argument("--list-unchanged", action="store_true", help="With --verbose: list unchanged files as well")
    scan.add_argument(
        "--rehash",
        action="store_true",
        help="recompute every checksum instead of reusing recorded ones for unchanged files",
    )
    scan.set_defaults(func=cmd_scan)

    sub.add_parser("status").set_defaults(func=cmd_status)

    worker = sub.add_parser("worker")
    worker.add_argument("--once", action="store_true")
    worker.add_argument("--max-jobs", type=int, default=1,
                        help="Consume up to N jobs in this process before exiting (default 1)")
    worker.add_argument(
        "--max-seconds", type=float, default=0,
        help="Time budget for this call in seconds. When it runs out the worker exits cleanly before the next job, "
             "so it always stops at a file boundary; 0 = unlimited. kb-pipeline-worker-once.sh passes the remainder of its round.")
    worker.add_argument("--dry-run", action="store_true")
    worker.add_argument("--source", default=None, help="Only claim jobs for one source key")
    worker.set_defaults(func=cmd_worker)

    sub.add_parser("health").set_defaults(func=cmd_health)

    qdrant = sub.add_parser("qdrant")
    qdrant_sub = qdrant.add_subparsers(dest="qdrant_command", required=True)
    ensure = qdrant_sub.add_parser("ensure-collections")
    ensure.add_argument("--dry-run", action="store_true")
    ensure.set_defaults(func=cmd_qdrant)
    ensure_graph = qdrant_sub.add_parser("ensure-graph-collections")
    ensure_graph.add_argument("--graph-version", required=True)
    ensure_graph.add_argument("--source", action="append", default=None, help="Source key; defaults to every graph-enabled source")
    ensure_graph.add_argument("--collection", action="append", default=None, help="Configured source collection; can be repeated")
    ensure_graph.add_argument("--all", action="store_true", help="Create graph collections for all configured sources")
    ensure_graph.add_argument(
        "--graph-type",
        action="append",
        choices=GRAPH_VECTOR_TYPES,
        default=None,
        help="Graph vector type; defaults to every graph vector type",
    )
    ensure_graph.add_argument("--dry-run", action="store_true")
    ensure_graph.set_defaults(func=cmd_qdrant_graph)

    fts = sub.add_parser("fts", help="Manage the OpenSearch keyword index")
    fts_sub = fts.add_subparsers(dest="fts_command", required=True)
    fts_init = fts_sub.add_parser("init", help="Create the OpenSearch indices (one per collection)")
    fts_init.add_argument("--collection", action="append", default=None, help="Accepted for symmetry; init does not filter")
    fts_init.set_defaults(func=cmd_fts)
    fts_status = fts_sub.add_parser("status", help="Show OpenSearch index counts")
    fts_status.add_argument("--collection", action="append", default=None, help="Collection to compare; can be repeated")
    fts_status.add_argument("--compare-qdrant", action="store_true", help="Compare FTS rows with Qdrant active point counts")
    fts_status.set_defaults(func=cmd_fts)
    fts_rebuild = fts_sub.add_parser("rebuild", help="Rebuild OpenSearch indices from Qdrant active points")
    fts_rebuild.add_argument("--collection", action="append", default=None, help="Collection to rebuild; can be repeated")
    fts_rebuild.add_argument("--batch-size", type=int, default=512)
    fts_rebuild.set_defaults(func=cmd_fts)
    fts_sync_doc = fts_sub.add_parser("sync-doc", help="Rebuild one document's OpenSearch rows from Qdrant active points")
    fts_sync_doc.add_argument("--collection", required=True)
    fts_sync_doc.add_argument("--doc-id", required=True)
    fts_sync_doc.add_argument("--batch-size", type=int, default=512)
    fts_sync_doc.set_defaults(func=cmd_fts)
    fts_search = fts_sub.add_parser("search", help="Run an OpenSearch keyword search")
    fts_search.add_argument("query")
    fts_search.add_argument("--collection", action="append", default=None, help="Accepted for symmetry; search returns indexed rows")
    fts_search.add_argument("--limit", type=int, default=10)
    fts_search.set_defaults(func=cmd_fts)

    reset = sub.add_parser("reset", help="Purge selected KB state/cache/FTS and reset its Qdrant collection")
    reset.add_argument("--source", action="append", default=None, help="Source key; can be repeated")
    reset.add_argument("--collection", action="append", default=None, help="Configured collection name; can be repeated")
    reset.add_argument("--all", action="store_true", help="Reset all configured sources")
    reset.add_argument("--yes", action="store_true", help="Actually purge and reset; omitted means dry-run plan only")
    reset.add_argument("--force", action="store_true", help="Ignore running-job rows in SQLite")
    reset.add_argument("--keep-cache", action="store_true", help="Keep parse cache for selected KB(s)")

    reset.add_argument("--keep-graph", action="store_true",

                        help="Leave graph collections / Neo4j / graph workspace in place")
    reset.add_argument(
        "--keep-collection",
        action="store_true",
        help="Keep Qdrant collection(s) and their corresponding FTS rows",
    )
    reset.add_argument("--no-recreate", action="store_true", help="Delete collection(s) without recreating")
    reset.set_defaults(func=cmd_reset)

    cleanup = sub.add_parser("cleanup", help="Cache/log cleanup helpers")
    cleanup_sub = cleanup.add_subparsers(dest="cleanup_command", required=True)
    cleanup_sub.add_parser("status").set_defaults(func=cmd_cleanup)
    weekly = cleanup_sub.add_parser("weekly")
    weekly.add_argument("--dry-run", action="store_true")
    weekly.set_defaults(func=cmd_cleanup)
    monthly = cleanup_sub.add_parser("monthly")
    monthly.add_argument("--dry-run", action="store_true")
    monthly.set_defaults(func=cmd_cleanup)
    qdrant_gc = cleanup_sub.add_parser("qdrant-gc", help="Physically delete inactive Qdrant points older than retention")
    qdrant_gc.add_argument("--retention-days", type=int, default=None)
    qdrant_gc.add_argument("--dry-run", action="store_true")
    qdrant_gc.set_defaults(func=cmd_cleanup)
    qdrant_graph_gc = cleanup_sub.add_parser(
        "qdrant-graph-gc",
        help="Days-only manual tool: delete unaliased graph collections older than retention (the keep-N rule is cleanup graph-gc)",
    )
    qdrant_graph_gc.add_argument("--retention-days", type=int, default=None)
    qdrant_graph_gc.add_argument(
        "--delete-unparseable",
        action="store_true",
        help="Also delete unaliased graph collections whose version has no timestamp",
    )
    qdrant_graph_gc.add_argument("--dry-run", action="store_true")
    qdrant_graph_gc.set_defaults(func=cmd_cleanup)
    neo4j_graph_gc_parser = cleanup_sub.add_parser(
        "neo4j-graph-gc",
        help="Days-only manual tool: delete inactive Neo4j graph versions older than retention (the keep-N rule is cleanup graph-gc)",
    )
    neo4j_graph_gc_parser.add_argument("--retention-days", type=int, default=None)
    neo4j_graph_gc_parser.add_argument(
        "--delete-unparseable",
        action="store_true",
        help="Also delete inactive Neo4j graph versions whose version has no timestamp and no imported_at",
    )
    neo4j_graph_gc_parser.add_argument("--dry-run", action="store_true")
    neo4j_graph_gc_parser.set_defaults(func=cmd_cleanup)
    graph_gc_parser = cleanup_sub.add_parser(
        "graph-gc",
        help="Delete superseded graph versions of every base with the active-plus-N-1 rule (same as the end of a build; skipped while a build runs)",
    )
    graph_gc_parser.add_argument("--keep-latest", type=int, default=None, help="Override GRAPH_GC_KEEP_VERSIONS")
    graph_gc_parser.add_argument("--dry-run", action="store_true")
    graph_gc_parser.set_defaults(func=cmd_cleanup)
    parse_assets_gc_parser = cleanup_sub.add_parser(
        "parse-assets-gc",
        help="Delete expired inactive points, SQLite rows, and parse asset dirs after retention",
    )
    parse_assets_gc_parser.add_argument("--retention-days", type=int, default=None)
    parse_assets_gc_parser.add_argument("--dry-run", action="store_true")
    parse_assets_gc_parser.set_defaults(func=cmd_cleanup)
    qdrant_backfill = cleanup_sub.add_parser(
        "qdrant-backfill-inactive-at",
        help="Add inactive_at to legacy inactive Qdrant points that do not have it",
    )
    qdrant_backfill.add_argument("--inactive-at-ts", type=int, default=None)
    qdrant_backfill.add_argument("--days-ago", type=int, default=0)
    qdrant_backfill.add_argument("--dry-run", action="store_true")
    qdrant_backfill.set_defaults(func=cmd_cleanup)

    search = sub.add_parser("search", help="Search service regression evaluation (eval) and question-set generation (make-set)")
    search_sub = search.add_subparsers(dest="search_command", required=True)
    search_eval = search_sub.add_parser("eval", help="Run a question set: hit@k / MRR / expect / min_docs / negatives, diffed against the previous result")
    search_eval.add_argument("--set", required=True, help="Question-set JSON (runtime/eval/*.json)")
    search_eval.add_argument("--out", default=None, help="Result JSON; an existing file of the same name is used as the previous result to diff against")
    search_eval.add_argument("--kbs", nargs="*", default=None, help="Restrict to these knowledge bases (a question's own kbs take precedence)")
    search_eval.add_argument("--top-k", default=12)
    search_eval.add_argument("--compare", default=None, help="Previous result file to diff against")
    search_eval.add_argument("--auto", action="store_true", help="Ignore the questions' own kbs and let the service route (also tests routing)")
    search_eval.set_defaults(func=cmd_search)
    search_make = search_sub.add_parser("make-set", help="Sample chunks from a base and draft questions with its extraction model (gold = chunk point_id)")
    search_make.add_argument("--kb", required=True)
    search_make.add_argument("--n", default=15)
    search_make.add_argument("--out", required=True)
    search_make.add_argument("--seed", default=7)
    search_make.set_defaults(func=cmd_search)

    graph = sub.add_parser("graph", help="Entity graph: build, Neo4j import, version maintenance and graph recall")
    graph_sub = graph.add_subparsers(dest="graph_command", required=True)

    def add_graph_source_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--source", action="append", default=None, help="Source key; defaults to graph-enabled sources")
        p.add_argument("--collection", action="append", default=None, help="Configured source collection; can be repeated")
        p.add_argument("--all", action="store_true", help="Select all configured sources")

    graph_build = graph_sub.add_parser("build", help="chunks -> extraction units -> entities/relations -> Qdrant vectors -> Neo4j -> alias switch")
    add_graph_source_args(graph_build)
    graph_build.add_argument("--graph-version", default=None)
    graph_build.add_argument("--dry-run", action="store_true", help="Only prepare the corpus and print unit counts; no LLM calls, no writes")
    graph_build.add_argument("--no-activate-aliases", action="store_true")
    graph_build.add_argument("--allow-existing-graph-version", action="store_true")
    graph_build.add_argument("--no-graph-gc", action="store_true")
    graph_build.set_defaults(func=cmd_graph)

    graph_check = graph_sub.add_parser("check-rebuild", help="Scheduled check: full rebuild when the policy is due, otherwise append new/changed documents (--execute to run)")
    graph_check.add_argument("--force-full", action="store_true",
                             help="Ignore the policy and do a full rebuild now the way the timer would (labels first, then build); one --source only, needs --execute")
    add_graph_source_args(graph_check)
    graph_check.add_argument("--execute", action="store_true", help="Run builds for sources whose policy is due")
    graph_check.add_argument("--dry-run", action="store_true", help="When executing, plan/export/ensure only")
    graph_check.set_defaults(func=cmd_graph)

    graph_append = graph_sub.add_parser("append", help="Incremental append: extract only new/changed documents, replay the previous merge decisions, reuse unchanged vectors, switch to the new version")
    add_graph_source_args(graph_append)
    graph_append.add_argument("--force", action="store_true", help="Append even when nothing changed (verifies replay and vector reuse)")
    graph_append.set_defaults(func=cmd_graph)

    graph_adopt = graph_sub.add_parser("adopt-current", help="Record current graph aliases as the rebuild baseline")
    add_graph_source_args(graph_adopt)
    graph_adopt.add_argument("--graph-version", default=None)
    graph_adopt.set_defaults(func=cmd_graph)

    graph_rollback = graph_sub.add_parser("rollback", help="Roll the live graph back to a kept earlier version (switch aliases, activate the Neo4j version, record it); the rejected version is deleted afterwards")
    add_graph_source_args(graph_rollback)
    graph_rollback.add_argument("--graph-version", required=True, help="Version to switch back to (must still be inside the kept N versions)")
    graph_rollback.add_argument("--force", action="store_true",
                                help="Switch even when the target's record is not a finished build (paused / failed) or is "
                                     "missing; collections with a wrong point count are refused all the same")
    graph_rollback.set_defaults(func=cmd_graph)

    graph_neo4j_import = graph_sub.add_parser("neo4j-import", help="Import a completed graph version into Neo4j")
    add_graph_source_args(graph_neo4j_import)
    graph_neo4j_import.add_argument("--graph-version", default=None)
    graph_neo4j_import.add_argument("--replace", action="store_true", help="Delete and rebuild an existing Neo4j projection for this graph version")
    graph_neo4j_import.add_argument("--no-activate", action="store_true", help="Do not set KB.active_graph_version after import")
    graph_neo4j_import.add_argument("--dry-run", action="store_true")
    graph_neo4j_import.add_argument("--batch-size", type=int, default=1000)
    graph_neo4j_import.set_defaults(func=cmd_graph)

    graph_neo4j_status = graph_sub.add_parser("neo4j-status", help="Show Neo4j graph projection status")
    add_graph_source_args(graph_neo4j_status)
    graph_neo4j_status.add_argument("--graph-version", default=None)
    graph_neo4j_status.set_defaults(func=cmd_graph)

    graph_neo4j_delete = graph_sub.add_parser("neo4j-delete", help="Delete one Neo4j graph projection version")
    add_graph_source_args(graph_neo4j_delete)
    graph_neo4j_delete.add_argument("--graph-version", required=True)
    graph_neo4j_delete.add_argument("--dry-run", action="store_true")
    graph_neo4j_delete.set_defaults(func=cmd_graph)

    graph_query = graph_sub.add_parser("query", help="Graph recall prototype: question -> entity/relation seeds -> Neo4j expansion -> chunks")
    add_graph_source_args(graph_query)
    graph_query.add_argument("question")
    graph_query.add_argument("--hops", type=int, default=2)
    graph_query.add_argument("--limit", type=int, default=12)
    graph_query.set_defaults(func=cmd_graph)

    graph_factcheck = graph_sub.add_parser("factcheck", help="Fact-level check: are hand-verified facts/series present in the current graph, with the right subject, as one series")
    add_graph_source_args(graph_factcheck)
    graph_factcheck.add_argument("--gold", required=True, help="Gold JSON (format described at the top of graph/factcheck.py)")
    graph_factcheck.add_argument("--out", default=None, help="Result JSON path (a .md summary table is written next to it)")
    graph_factcheck.add_argument("--graph-version", default=None)
    graph_factcheck.set_defaults(func=cmd_graph)

    graph_status = graph_sub.add_parser("status", help="Show recent graph build records")
    graph_status.add_argument("--limit", type=int, default=20)
    graph_status.set_defaults(func=cmd_graph)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
