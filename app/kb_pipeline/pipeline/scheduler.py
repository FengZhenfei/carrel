from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from .. import db
from ..models import SourceFile
from .detect_changes import detect_change, parser_profile_for


@dataclass
class ScanStats:
    seen: int = 0
    added: int = 0
    content_changed: int = 0
    parser_changed: int = 0
    metadata_changed: int = 0
    needs_parse: int = 0
    parse_failed: int = 0
    deleted: int = 0
    unchanged: int = 0
    jobs: int = 0
    recent_pending: int = 0


def schedule_file(
    con: sqlite3.Connection,
    *,
    ingest_run_id: str,
    file: SourceFile,
    current_seen_file_keys: set[int] | None = None,
    current_checksum_counts: dict[str, int] | None = None,
    requeue_failed: bool = False,
    dry_run: bool = False,
) -> tuple[str, str | None]:
    old = db.get_file(con, file.kb_id, file.file_key)
    if old is None and file.source_type == "local_mirror":
        same_path = db.get_file_by_path(con, file.kb_id, file.source_path)
        if same_path is not None:
            file.file_key = int(same_path["file_key"])
            old = same_path
    if old is None and file.source_type == "local_mirror" and file.checksum:
        moved_from = _same_content_local_file(
            con,
            file,
            current_seen_file_keys=current_seen_file_keys,
            current_checksum_counts=current_checksum_counts,
        )
        if moved_from is None:
            moved_from = _deleted_same_content_local_file(
                con,
                file,
                current_seen_file_keys=current_seen_file_keys,
                current_checksum_counts=current_checksum_counts,
            )
        if moved_from is not None:
            file.file_key = int(moved_from["file_key"])
            old = moved_from
    change = detect_change(old, file)
    file_id = db.file_id_for(file.kb_id, file.file_key)
    parse_profile = parser_profile_for(file) if change.job_type == "parse" else None
    dedupe_key: str | None = None
    if change.job_type == "parse":
        dedupe_key = f"parse:{file_id}:{file.content_version}:{parse_profile}"
        failed_job = db.failed_job_for_dedupe_key(con, dedupe_key)
        # The dedupe key pins content_version AND parser_profile. Steady-state
        # re-detections of a terminally failed combination stay blocked:
        # needs_parse would retry the identical work every scan, and a profile
        # bump (parser_changed) used to re-parse a permanently broken file
        # forever. Event-driven changes stay OPEN on purpose -- restoring a
        # deleted file or reverting content is a human action and grants one
        # fresh attempt per event; if that attempt fails too, the file settles
        # back into the blocked needs_parse state on the next scan.
        if (
            change.change_type in {"needs_parse", "parser_changed"}
            and failed_job is not None
            and not requeue_failed
        ):
            if _failed_parse_blocks_auto_requeue(failed_job, change.change_type):
                if not dry_run:
                    db.upsert_file(con, file, status=_status_after_scan(old, file))
                return "parse_failed", None

    if dry_run:
        return change.change_type, None

    if change.change_type == "unchanged" and old is not None:
        # No change at all: only record "seen this round", do not rewrite the whole row
        db.touch_file_seen(con, str(old["file_id"]))
        return change.change_type, None

    db.upsert_file(con, file, status=_status_after_scan(old, file))
    if change.change_type == "content_changed" and old is not None and not dry_run:
        # If the old version is still queued, it would run the full MinerU+VLM+embedding and then be judged
        # stale and discarded right before writing -- a whole GPU round burned for nothing.
        cancelled = db.cancel_pending_jobs_for_file(
            con, str(old["file_id"]),
            "cancelled because the file changed again before this attempt started",
            exclude_job_types={"delete"},
        )
        if cancelled:
            print(f"[scan] superseded {cancelled} queued job(s) for {file.source_path}", flush=True)
    job_id: str | None = None
    if change.job_type:
        priority = 10 if change.job_type == "metadata_update" else 100
        if change.job_type != "parse":
            dedupe_key = f"{change.job_type}:{file_id}:{db.metadata_fingerprint(file)}"
        payload = {
            "reason": change.reason,
            "source_path": file.source_path,
            "rel_path": file.rel_path,
            "content_version": file.content_version,
            "metadata_fingerprint": db.metadata_fingerprint(file),
        }
        if old is not None and str(old["status"]) == "deleted" and change.job_type == "metadata_update":
            payload["reactivate"] = True
        job_id = db.enqueue_job(
            con,
            ingest_run_id=ingest_run_id,
            file_id=file_id,
            kb_id=file.kb_id,
            collection=file.collection,
            file_key=file.file_key,
            job_type=change.job_type,
            parser_profile=parse_profile,
            priority=priority,
            payload=payload,
            dedupe_key=dedupe_key,
        )
    return change.change_type, job_id


def _same_content_local_file(
    con: sqlite3.Connection,
    file: SourceFile,
    *,
    current_seen_file_keys: set[int] | None = None,
    current_checksum_counts: dict[str, int] | None = None,
):
    if current_checksum_counts is not None and current_checksum_counts.get(str(file.checksum), 0) != 1:
        return None
    matches = [
        row
        for row in db.active_files_by_checksum(con, file.kb_id, str(file.checksum))
        if str(row["source_path"]) != file.source_path
        and int(row["file_key"]) not in (current_seen_file_keys or set())
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def _deleted_same_content_local_file(
    con: sqlite3.Connection,
    file: SourceFile,
    *,
    current_seen_file_keys: set[int] | None = None,
    current_checksum_counts: dict[str, int] | None = None,
):
    if current_checksum_counts is not None and current_checksum_counts.get(str(file.checksum), 0) != 1:
        return None
    matches = [
        row
        for row in db.deleted_files_by_checksum(con, file.kb_id, str(file.checksum))
        if str(row["source_path"]) != file.source_path
        and int(row["file_key"]) not in (current_seen_file_keys or set())
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def update_stats(stats: ScanStats, change_type: str, job_id: str | None) -> None:
    stats.seen += 1
    if change_type == "new":
        stats.added += 1
    elif change_type == "content_changed":
        stats.content_changed += 1
    elif change_type == "parser_changed":
        stats.parser_changed += 1
    elif change_type == "metadata_changed":
        stats.metadata_changed += 1
    elif change_type == "needs_parse":
        stats.needs_parse += 1
    elif change_type == "parse_failed":
        stats.parse_failed += 1
    else:
        stats.unchanged += 1
    if job_id:
        stats.jobs += 1


def requeue_failed_lifecycle_jobs(con: sqlite3.Connection, *, ingest_run_id: str, dry_run: bool = False) -> int:
    """Re-enqueue one-shot lifecycle jobs whose latest attempt failed terminally.

    The scan records the desired state in SQLite immediately and enqueues a
    metadata_update/delete job to apply it to Qdrant/OpenSearch; when that job
    exhausts its retries, no later scan re-detects the change (detect_change
    compares against the already-updated row), so the divergence was permanent:
    a failed reactivate left a live file unsearchable, a failed delete left
    ghost results. Parse jobs are excluded on purpose -- retrying those is the
    (expensive) job of --requeue-failed."""
    rows = con.execute(
        "SELECT DISTINCT dedupe_key FROM jobs "
        "WHERE status = 'failed' AND job_type IN ('metadata_update', 'delete', 'fts_sync') AND dedupe_key IS NOT NULL"
    ).fetchall()
    requeued = 0
    for row in rows:
        key = str(row["dedupe_key"])
        latest = db.latest_job_for_dedupe_key(con, key)
        if latest is None or str(latest["status"]) != "failed":
            continue  # a newer attempt succeeded or is already in flight
        file_id = str(latest["file_id"] or "")
        file_row = db.get_file_by_id(con, file_id) if file_id else None
        if file_row is None:
            continue
        job_type = str(latest["job_type"])
        file_status = str(file_row["status"])
        if job_type == "delete" and file_status != "deleted":
            continue  # the file came back; deletion is no longer wanted
        if job_type == "metadata_update":
            if file_status == "deleted":
                continue  # superseded by a deletion
            current_key = f"metadata_update:{file_id}:{db.metadata_fingerprint_from_row(file_row)}"
            if key != current_key:
                continue  # the row moved on; the normal scan path owns the new key
        if job_type == "fts_sync":
            # Keyword index compensation: only re-queue while the file still exists and the version is unchanged
            # (a new version is synced by its own parse). The compensation job used to borrow the
            # metadata_update type with an fts-retry:… key, which never matched (Codex review F06)
            if file_status == "deleted":
                continue
            if key != db.fts_sync_dedupe_key(file_id, str(file_row["content_version"] or "")):
                continue
        if dry_run:
            requeued += 1
            continue
        try:
            payload = json.loads(latest["payload_json"] or "{}")
        except (TypeError, ValueError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        payload["requeued_from"] = str(latest["job_id"])
        db.enqueue_job(
            con,
            ingest_run_id=ingest_run_id,
            file_id=file_id,
            kb_id=str(latest["kb_id"]),
            collection=str(latest["collection"]),
            file_key=int(latest["file_key"]) if latest["file_key"] is not None else None,
            job_type=job_type,
            parser_profile=None,
            priority=int(latest["priority"] or (10 if job_type == "metadata_update" else 20)),
            payload=payload,
            dedupe_key=key,
        )
        requeued += 1
        print(f"[scan] requeued failed {job_type} job file={file_row['source_path']}", flush=True)
    return requeued


def requeue_single_file(con: sqlite3.Connection, *, ingest_run_id: str, file_row: sqlite3.Row) -> str | None:
    """Web-console retry for one file: enqueue a parse job for the file's
    current version+profile through the normal queue (same dedupe key and
    worker lease as the automatic pipeline, so no conflicting side channel).
    Returns None when an attempt is already queued/running."""
    from pathlib import Path as _Path

    from ..parsers.common import parser_profile_for_path

    profile = parser_profile_for_path(_Path(str(file_row["filename"])))
    dedupe_key = f"parse:{file_row['file_id']}:{file_row['content_version']}:{profile}"
    if db.active_job_for_dedupe_key(con, dedupe_key) is not None:
        return None
    return db.enqueue_job(
        con,
        ingest_run_id=ingest_run_id,
        file_id=str(file_row["file_id"]),
        kb_id=str(file_row["kb_id"]),
        collection=str(file_row["collection"]),
        file_key=int(file_row["file_key"]),
        job_type="parse",
        parser_profile=profile,
        priority=50,
        payload={"reason": "manual retry from web console", "source_path": str(file_row["source_path"])},
        dedupe_key=dedupe_key,
    )


def requeue_kb_files(con: sqlite3.Connection, *, ingest_run_id: str, kb_id: str, reason: str) -> int:
    """Re-enqueue every live file of a KB (web console "apply new prompt").
    Same content_version + profile means identical chunk_uids: the points are
    overwritten in place with fresh captions/embeddings."""
    requeued = 0
    from pathlib import Path as _Path

    from ..parsers.common import parser_profile_for_path

    for row in con.execute(
        "SELECT * FROM files WHERE kb_id = ? AND status != 'deleted' ORDER BY source_path", (kb_id,)
    ).fetchall():
        profile = parser_profile_for_path(_Path(str(row["filename"])))
        dedupe_key = f"parse:{row['file_id']}:{row['content_version']}:{profile}"
        if db.active_job_for_dedupe_key(con, dedupe_key) is not None:
            continue
        db.enqueue_job(
            con,
            ingest_run_id=ingest_run_id,
            file_id=str(row["file_id"]),
            kb_id=kb_id,
            collection=str(row["collection"]),
            file_key=int(row["file_key"]),
            job_type="parse",
            parser_profile=profile,
            priority=120,
            payload={"reason": reason, "source_path": str(row["source_path"])},
            dedupe_key=dedupe_key,
        )
        requeued += 1
    return requeued


# A service-side outage (MinerU/VLM/embedding/OpenSearch down) marks every job in that window as failed;
# after the service recovers the scan does not heal by itself, leaving only per-file retries or a full
# re-parse. Give failed jobs a cooldown period, after which the scan may re-queue them once automatically.
# 0 = disable this behaviour.
FAILED_RETRY_COOLDOWN_SECONDS = int(os.getenv("KB_FAILED_RETRY_COOLDOWN_SECONDS", str(24 * 3600)))


def _failed_parse_blocks_auto_requeue(job_row, change_type: str = "needs_parse") -> bool:
    """Whether this scan should still hold back a file whose last parse failed instead of re-queueing it.
    True = hold it back (the file stays in the failed state)."""
    error_raw = str(job_row["error"] or "")
    error = error_raw.lower()
    if "physical file not found" in error:
        return False   # the file reappeared: re-queue
    if "NonRetryableParseError" in error_raw:
        # Deterministic failure (no parser route, broken format): re-queueing is pointless, so hold it back
        # until the parser / router changes or the user retries by hand. This used to return False (let
        # through), re-enqueueing the same unchanged input on every scan (Codex review F07).
        return change_type != "parser_changed"
    if FAILED_RETRY_COOLDOWN_SECONDS > 0:
        finished = 0
        try:
            finished = int(job_row["finished_at"] or 0)
        except (KeyError, IndexError, TypeError):
            finished = 0
        if finished and int(time.time()) - finished >= FAILED_RETRY_COOLDOWN_SECONDS:
            return False   # cooldown elapsed: let it through so the scan re-queues it once
    return True


def schedule_deletes_for_source(
    con: sqlite3.Connection,
    *,
    ingest_run_id: str,
    kb_id: str,
    collection: str,
    seen_file_keys: set[int],
    verify_physical: bool = False,
    dry_run: bool = False,
) -> tuple[int, int]:
    current_rows = db.files_for_kb(con, kb_id)
    missing_rows = [
        row for row in current_rows if int(row["file_key"]) not in seen_file_keys
    ]
    if verify_physical and missing_rows:
        # Second line of defence for scan-driven deletes: never tear down a file
        # that is still sitting on disk (a scanner skip, a permission blip, or a
        # key that drifted). Unenroll passes verify_physical=False -- there the
        # files legitimately stay put while their points go away.
        kept = []
        for row in missing_rows:
            physical = str(row["physical_path"] or "")
            if physical and Path(physical).exists():
                print(
                    f"[scan] source={kb_id} delete skipped, file still on disk: {row['source_path']}",
                    flush=True,
                )
                continue
            kept.append(row)
        missing_rows = kept
    missing_count = len(missing_rows)
    current_count = len(current_rows)
    missing_ratio = missing_count / current_count if current_count else 0.0
    # This used to be the mass-delete circuit breaker: once missing exceeded a threshold, propagation was
    # refused until someone added --force-mass-delete. It was removed because what it guarded was reversible
    # anyway -- a delete only sets files.status to deleted and the Qdrant points to inactive, and a file put
    # back within QDRANT_INACTIVE_RETENTION_DAYS revives through reactivate without even re-parsing; deletes
    # also take no part in the graph rebuild criterion (only additions count), so no LLM money is wasted.
    # The cost, on the other hand, was real: in 17 days of operation the breaker fired once in total across
    # the Mac and DGX sides, with 0 true positives, yet it stalled the whole sync chain for 4 hours before a
    # human noticed. The criterion survives as one log line -- when troubleshooting, grep "deletes detected"
    # shows how big a slice each round deleted, and from which KB.
    if missing_count:
        print(
            f"[scan] source={kb_id} deletes detected: missing={missing_count} "
            f"current={current_count} ratio={missing_ratio:.1%}",
            flush=True,
        )

    deleted = 0
    jobs = 0
    for row in missing_rows:
        nc_file_id = int(row["file_key"])
        deleted += 1
        if dry_run:
            continue
        db.mark_file_deleted(con, str(row["file_id"]))
        cancelled = db.cancel_pending_jobs_for_file(
            con,
            str(row["file_id"]),
            "cancelled because file disappeared from source scan",
            exclude_job_types={"delete"},
        )
        if cancelled:
            print(
                f"[scan] source={kb_id} cancelled_pending_jobs={cancelled} "
                f"file={row['source_path']}",
                flush=True,
            )
        db.enqueue_job(
            con,
            ingest_run_id=ingest_run_id,
            file_id=str(row["file_id"]),
            kb_id=kb_id,
            collection=collection,
            file_key=nc_file_id,
            job_type="delete",
            priority=20,
            payload={
                "reason": "file disappeared from source scan",
                "previous_path": str(row["source_path"]),
            },
            dedupe_key=f"delete:{row['file_id']}",
        )
        jobs += 1
    return deleted, jobs


def _status_after_scan(old_row, file: SourceFile) -> str:
    if old_row is None or str(old_row["status"]) == "deleted":
        return "seen"
    if str(old_row["indexed_version"] or "") == file.content_version:
        return "indexed"
    return "seen"
