from __future__ import annotations

import json
import os
import socket
import sqlite3
import subprocess
import threading
import time
import traceback

from .. import db
from .. import search_fts
from ..config import Settings
from ..parsers.errors import service_unreachable
from ..utils import looks_like_worker_command
from .parse_job import JobCancelled, process_parse_job
from ..vector.qdrant import client as qdrant_client
from ..vector.qdrant import (
    file_version_point_count, mark_file_inactive, reactivate_file_metadata, update_file_metadata,
)


def worker_id() -> str:
    return f"{socket.gethostname()}:{__import__('os').getpid()}"


def run_once(
    con: sqlite3.Connection,
    *,
    settings: Settings | None = None,
    dry_run: bool = False,
    source_key: str | None = None,
) -> str:
    allowed_job_types = None
    allowed_kb_ids = None
    lease_seconds = 3600
    if settings is not None and source_key:
        allowed_kb_ids = {settings.sources[source_key].kb_id}
    if settings is not None and not settings.parse_enabled:
        allowed_job_types = {"metadata_update", "delete", "fts_sync"}
        lease_seconds = settings.metadata_job_lease_seconds
    elif settings is not None:
        lease_seconds = settings.parse_job_lease_seconds
    current_worker_id = worker_id()
    released = _recover_or_refresh_running_jobs(
        con,
        lease_seconds=lease_seconds,
        max_retries=settings.job_max_retries if settings is not None else 0,
        retry_delay_seconds=settings.job_retry_base_seconds if settings is not None else 300,
    )
    if released:
        print(f"[worker] recovered orphan jobs={','.join(released)}", flush=True)
    job = db.claim_job_for_types(
        con,
        current_worker_id,
        lease_seconds=lease_seconds,
        allowed_job_types=allowed_job_types,
        allowed_kb_ids=allowed_kb_ids,
    )
    if not job:
        return "no-job"
    job_type = str(job["job_type"])
    heartbeat = None
    if settings is not None and not dry_run:
        heartbeat = JobLeaseHeartbeat(
            db_path=settings.state_db,
            job_id=str(job["job_id"]),
            locked_by=current_worker_id,
            lease_seconds=lease_seconds,
        )
        heartbeat.start()
    try:
        if dry_run:
            db.release_job(con, job["job_id"])
            return f"dry-run-claimed:{job['job_id']}:{job_type}:kb={job['kb_id']}"
        if job_type == "metadata_update":
            if settings:
                file_row = db.get_file_by_id(con, str(job["file_id"]))
                if file_row:
                    payload = json.loads(str(job["payload_json"] or "{}"))
                    q = qdrant_client(settings.qdrant_url, settings.qdrant_api_key)
                    if payload.get("reactivate"):
                        # A file that returned after deletion is restored through this path without re-parsing.
                        # Once the Qdrant points are flipped back to active, the SQLite chunk ledger must follow,
                        # otherwise the file is searchable but never enters the graph -- and since the ledger
                        # did not change, the rebuild policy cannot notice either.
                        remaining = file_version_point_count(q, file_row)
                        # Only flip back the points (by id) that were active at the moment of deletion / KB close:
                        # old chunks replaced within the same version stay inactive. Flipping a whole version
                        # would revive every chunking ever done (2026-09-06 close-and-reopen: the chunk ledger
                        # grew 6x and the automatic rebuild was triggered by a phantom increment).
                        restore_ids = db.chunk_point_ids_to_restore(
                            con, str(file_row["file_id"]), str(file_row["content_version"])
                        )
                        if remaining > 0 and restore_ids:
                            reactivate_file_metadata(q, file_row, point_ids=restore_ids)
                            db.mark_chunks_active(
                                con, str(file_row["file_id"]), str(file_row["content_version"])
                            )
                            print(f"[restore] points reactivated={len(restore_ids)} file={file_row['rel_path']}", flush=True)
                        else:
                            # The inactive points were GC'd after the retention period, or the ledger has no
                            # deleted chunks to bring back (old data deleted before the fix): the restore does not
                            # hold, fall back to a re-parse to realign both sides; chunks stay as they are until
                            # then.
                            from .scheduler import requeue_single_file

                            requeued = requeue_single_file(
                                con,
                                ingest_run_id=str(job["ingest_run_id"] or "") or "restore",
                                file_row=file_row,
                            )
                            print(
                                f"[restore] points gone, re-parse queued={requeued} "
                                f"file={file_row['rel_path']}",
                                flush=True,
                            )
                    else:
                        update_file_metadata(q, file_row)
                    _sync_fts_doc(settings, q, file_row)
                    if str(file_row["indexed_version"] or "") == str(file_row["content_version"] or ""):
                        db.mark_file_indexed(con, str(file_row["file_id"]), str(file_row["content_version"]))
            db.mark_job_done(con, job["job_id"], current_worker_id)
            return f"metadata-done:{job['job_id']}"
        if job_type == "fts_sync":
            # Keyword index compensation (queued when OpenSearch could not be written during the parse): rewrite
            # the full-text index from this file's active points in Qdrant. Void it when the file was deleted or
            # changed version -- the new version's parse syncs by itself. A failed write raises as usual and
            # takes the failure / backoff retry path below; the failure record hangs on this job and stays
            # visible until the index is filled in (Codex review F06).
            if settings:
                file_row = db.get_file_by_id(con, str(job["file_id"]))
                payload = json.loads(str(job["payload_json"] or "{}"))
                wanted = str(payload.get("content_version") or "")
                if (file_row and str(file_row["status"]) != "deleted"
                        and (not wanted or wanted == str(file_row["content_version"] or ""))):
                    q = qdrant_client(settings.qdrant_url, settings.qdrant_api_key)
                    _sync_fts_doc(settings, q, file_row)
                    print(f"[fts] compensation synced file={file_row['source_path']}", flush=True)
                else:
                    print(f"[fts] compensation superseded job={job['job_id']}", flush=True)
                # Also resolve the failure records of earlier attempts under this key (the ones the scan
                # re-queued after retries were exhausted)
                db.resolve_failures_for_dedupe_key(con, str(job["dedupe_key"] or ""))
            db.mark_job_done(con, job["job_id"], current_worker_id)
            return f"fts-sync-done:{job['job_id']}"
        if job_type == "delete":
            if settings:
                file_row = db.get_file_by_id(con, str(job["file_id"]))
                if file_row and str(file_row["status"]) == "deleted":
                    q = qdrant_client(settings.qdrant_url, settings.qdrant_api_key)
                    mark_file_inactive(q, file_row)
                    _sync_fts_doc(settings, q, file_row)
                    # Only the batch active right now is marked deleted (only that batch comes back on restore);
                    # old chunks of the same version stay inactive
                    deleted_chunks = db.mark_chunks_deleted(con, str(file_row["file_id"]))
                    if deleted_chunks:
                        print(
                            f"[delete] deleted sqlite chunks={deleted_chunks} file={file_row['source_path']}",
                            flush=True,
                        )
            db.mark_job_done(con, job["job_id"], current_worker_id)
            return f"delete-done:{job['job_id']}"
        if job_type == "parse":
            if settings is None or not settings.parse_enabled:
                db.mark_job_paused(con, job["job_id"], "parse disabled; set KB_PARSE_ENABLED=1 after parser implementation is approved")
                return f"parse-paused:{job['job_id']}"
            file_row = db.get_file_by_id(con, str(job["file_id"]))
            if file_row and str(file_row["status"]) == "deleted":
                db.cancel_job(con, job["job_id"], "parse cancelled because file is deleted")
                return f"parse-cancelled:{job['job_id']}:deleted"
            result = process_parse_job(con, settings, job)
            db.mark_job_done(con, job["job_id"], current_worker_id)
            _rerun_if_asked(con, job)
            return result
        raise RuntimeError(f"unknown job_type={job_type}")
    except JobCancelled as exc:
        # The user closed / deleted this KB: not a failure, no failure record, no backoff retry, straight to
        # the terminal state. But the job may no longer belong to this worker -- reclaimed as an orphan into
        # retry by another worker, or taken over; in that case change nothing and let it come back as the
        # scheduled retry, never overwrite it with cancelled (or the file stays on the old version forever).
        owned = db.cancel_job_if_owned(con, job["job_id"], current_worker_id, str(exc))
        con.commit()
        if not owned:
            print(f"[worker] job={job['job_id']} kb={job['kb_id']} no longer ours; leaving its status alone", flush=True)
            return f"superseded:{job['job_id']}"
        print(f"[worker] cancelled job={job['job_id']} kb={job['kb_id']}", flush=True)
        return f"cancelled:{job['job_id']}"
    except Exception as exc:
        # Cancellation takes precedence over failure retry (Codex review N07): after a KB close the cancel flag
        # is set but the checkpoint has not come yet, and an external call in between raised first -- this used
        # to take "failure -> backoff retry" and put the cancelled job back in the queue; a job no longer ours
        # must not have a failure recorded on someone else's behalf either
        try:
            gone = db.job_cancel_requested(con, job["job_id"], current_worker_id)
        except Exception:
            gone = False
        if gone:
            owned = db.cancel_job_if_owned(con, job["job_id"], current_worker_id, f"cancelled; last error: {exc!r}"[:4000])
            con.commit()
            print(f"[worker] job={job['job_id']} kb={job['kb_id']} cancelled while failing ({exc!r})", flush=True)
            return f"cancelled:{job['job_id']}" if owned else f"superseded:{job['job_id']}"
        if settings is not None and service_unreachable(exc):
            # After a boot the parsing, embedding and image-description services can take over ten minutes to become
            # ready, while the worker starts claiming jobs after two: not reaching them is not a failure of this file
            delay = db.defer_job(
                con, job["job_id"], repr(exc), worker_id=current_worker_id,
                base_delay_seconds=settings.job_retry_base_seconds, max_delay_seconds=settings.job_retry_max_seconds,
                max_deferrals=settings.job_max_retries,
            )
            if delay is not None:
                return f"deferred:{job['job_id']}:delay={delay}s:{exc!r}"
        should_retry = _should_retry(job, settings, exc)
        retry_delay = _retry_delay_seconds(job, settings) if should_retry else 0
        db.add_failure(
            con,
            file_id=job["file_id"],
            job_id=job["job_id"],
            stage="worker",
            error_type=type(exc).__name__,
            error_message=f"{exc}\n{traceback.format_exc()}",
        )
        db.mark_job_failed(con, job["job_id"], repr(exc), retry=should_retry, retry_delay_seconds=retry_delay,
                           worker_id=current_worker_id)
        if should_retry:
            return f"retry:{job['job_id']}:delay={retry_delay}s:{exc!r}"
        return f"failed:{job['job_id']}:{exc!r}"
    finally:
        if heartbeat is not None:
            heartbeat.stop()


def _rerun_if_asked(con: sqlite3.Connection, job: sqlite3.Row) -> None:
    """The console's "Re-parse all" reached this file again while this run was in progress (recorded by
    scheduler._ask_rerun): queue one more run once it is done. That run re-reads the KB configuration when it
    starts, so it uses the changed one."""
    from .scheduler import RERUN_PRIORITY, requeue_single_file

    row = con.execute("SELECT status, payload_json FROM jobs WHERE job_id = ?", (job["job_id"],)).fetchone()
    if row is None or str(row["status"]) != "done":
        return
    try:
        reason = json.loads(str(row["payload_json"] or "{}")).get("rerun")
    except (AttributeError, ValueError):
        reason = None
    file_row = db.get_file_by_id(con, str(job["file_id"])) if reason else None
    if file_row is None or str(file_row["status"]) == "deleted":
        return
    queued = requeue_single_file(
        con, ingest_run_id=str(job["ingest_run_id"] or "") or "rerun", file_row=file_row,
        reason=str(reason), priority=RERUN_PRIORITY,
    )
    print(f"[worker] re-parse requested while job={job['job_id']} was running; queued={queued}", flush=True)


def _should_retry(job: sqlite3.Row, settings: Settings | None, exc: Exception) -> bool:
    from .parse_job import NonRetryableParseError

    if settings is None or settings.job_max_retries <= 0:
        return False
    # Decide by exception type, not by error string: deterministic failures (unsupported format, missing
    # file, over the size limit, vector dimension mismatch) come out the same no matter how often retried.
    if isinstance(exc, NonRetryableParseError):
        return False
    if getattr(exc, "deterministic", False):   # MinerU 4xx: the document itself is broken
        return False
    from ..embedding.client import EmbeddingDimensionError

    if isinstance(exc, EmbeddingDimensionError):
        return False
    message = str(exc).lower()
    if "physical file not found" in message:   # compatibility with the generic exception raised by old paths
        return False
    return int(job["retry_count"] or 0) < settings.job_max_retries


def _retry_delay_seconds(job: sqlite3.Row, settings: Settings | None) -> int:
    if settings is None:
        return 300
    retry_count = max(0, int(job["retry_count"] or 0))
    delay = settings.job_retry_base_seconds * (2 ** retry_count)
    return max(1, min(settings.job_retry_max_seconds, delay))


def _sync_fts_doc(settings: Settings, q, file_row: sqlite3.Row) -> None:
    result = search_fts.sync_doc_from_qdrant(
        url=settings.opensearch_url,
        qdrant=q,
        collection=str(file_row["collection"]),
        doc_id=f"{file_row['kb_id']}:{file_row['file_key']}",
    )
    print(
        f"[fts] sync-doc collection={result['collection']} doc_id={result['doc_id']} "
        f"deleted={result['deleted_rows']} inserted={result['inserted_rows']}",
        flush=True,
    )


def _recover_or_refresh_running_jobs(
    con: sqlite3.Connection,
    *,
    lease_seconds: int,
    max_retries: int = 0,
    retry_delay_seconds: int = 300,
) -> list[str]:
    hostname = socket.gethostname()
    refreshed_until = db.now_ts() + max(1, int(lease_seconds))
    released: list[str] = []
    rows = con.execute(
        "SELECT job_id, locked_by FROM jobs WHERE status = 'running' ORDER BY started_at ASC"
    ).fetchall()
    for row in rows:
        locked_by = str(row["locked_by"] or "")
        locked_host, separator, pid_text = locked_by.rpartition(":")
        if not separator or locked_host != hostname:
            continue
        try:
            pid = int(pid_text)
        except ValueError:
            pid = -1
        if _pid_is_worker(pid):
            con.execute(
                """
                UPDATE jobs
                SET locked_until = ?, updated_at = ?
                WHERE job_id = ? AND status = 'running' AND locked_by = ?
                """,
                (refreshed_until, db.now_ts(), row["job_id"], locked_by),
            )
            continue
        # The owning process is gone: count the attempt and back off, so a
        # document that kills the interpreter cannot be re-served first on
        # every run and starve everything behind it.
        outcome = db.release_crashed_job(
            con,
            str(row["job_id"]),
            max_retries=max_retries,
            retry_delay_seconds=retry_delay_seconds,
            expected_owner=locked_by,           # no-op if the lock changed hands meanwhile (Codex review N07)
        )
        released.append(f"{row['job_id']}:{outcome}")
    return released


def _looks_like_worker_command(command: str) -> bool:
    """Whether the command line from `ps -o command=` is a parse worker; the recognition rule is shared with the
    maintenance module (utils.looks_like_worker_command)."""
    return looks_like_worker_command(command)


def _pid_is_worker(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    try:
        proc = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return proc.returncode == 0 and _looks_like_worker_command(proc.stdout.strip())


class JobLeaseHeartbeat:
    def __init__(
        self,
        *,
        db_path,
        job_id: str,
        locked_by: str,
        lease_seconds: int,
    ) -> None:
        self.db_path = db_path
        self.job_id = job_id
        self.locked_by = locked_by
        self.lease_seconds = max(1, int(lease_seconds))
        self.interval = max(30, min(300, self.lease_seconds // 4))
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name=f"job-lease-{job_id}", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5)

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval):
            now = int(time.time())
            try:
                with db.connect(self.db_path) as con:
                    con.execute(
                        """
                        UPDATE jobs
                        SET locked_until = ?, updated_at = ?
                        WHERE job_id = ? AND status = 'running' AND locked_by = ?
                        """,
                        (now + self.lease_seconds, now, self.job_id, self.locked_by),
                    )
            except sqlite3.Error as exc:
                print(f"[worker] lease heartbeat failed job={self.job_id} error={exc!r}", flush=True)
