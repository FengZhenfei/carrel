"""Scanning, change detection, the job queue, and the worker's branches, cancellation and leases."""
from __future__ import annotations

import errno
import io
import json
import os
import re
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import ExitStack, contextmanager, redirect_stdout

from kb_pipeline import db
from kb_pipeline.localfs.scanner import should_skip
from kb_pipeline.models import KBSource
from kb_pipeline.parsers.common import parser_profile_for_path
from kb_pipeline.pipeline.parse_job import _embed_visual_chunks as _REAL_EMBED_VISUAL_CHUNKS
from kb_pipeline.pipeline.scheduler import schedule_deletes_for_source, schedule_file

from _support import _CodexAudit20260906TestsSupport, _block, _local_file, _repo_file


class ScannerRegressionTests(unittest.TestCase):
    def test_same_kb_move_stays_metadata_only_across_later_scans(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_db = Path(tmp) / "state.db"
            db.init_db(state_db)
            with db.connect(state_db) as con:
                original = _local_file("before.pdf")
                db.upsert_file(con, original, status="indexed")
                db.mark_file_indexed(
                    con,
                    original.doc_id,
                    original.content_version,
                    parser_profile_for_path(Path(original.filename)),
                )

                moved = _local_file("folder/after.pdf")
                generated_moved_id = moved.file_key
                first_change, first_job = schedule_file(
                    con,
                    ingest_run_id="scan-1",
                    file=moved,
                    current_seen_file_keys={generated_moved_id},
                    current_checksum_counts={str(moved.checksum): 1},
                )
                self.assertEqual(first_change, "metadata_changed")
                self.assertIsNotNone(first_job)
                self.assertEqual(moved.file_key, original.file_key)

                later_scan = _local_file("folder/after.pdf")
                second_change, second_job = schedule_file(
                    con,
                    ingest_run_id="scan-2",
                    file=later_scan,
                    current_seen_file_keys={later_scan.file_key},
                    current_checksum_counts={str(later_scan.checksum): 1},
                )
                deleted, delete_jobs = schedule_deletes_for_source(
                    con,
                    ingest_run_id="scan-2",
                    kb_id=later_scan.kb_id,
                    collection=later_scan.collection,
                    seen_file_keys={later_scan.file_key},
                )
                job_types = [str(row["job_type"]) for row in con.execute("SELECT job_type FROM jobs").fetchall()]
                active_files = db.files_for_kb(con, later_scan.kb_id)

            self.assertEqual(second_change, "unchanged")
            self.assertIsNone(second_job)
            self.assertEqual(later_scan.file_key, original.file_key)
            self.assertEqual((deleted, delete_jobs), (0, 0))
            self.assertEqual(job_types, ["metadata_update"])
            self.assertEqual(len(active_files), 1)

    def test_large_delete_proceeds_and_is_recorded(self) -> None:
        """The mass-delete circuit breaker has been removed. What it blocked was reversible anyway (soft delete +
        reactivate within the retention period), and in 17 days live it had 0 true positives yet stalled the whole
        sync chain for 4 hours. Two things are pinned here: a 40% delete is no longer blocked, and the statistics
        line used for troubleshooting is still there (with matching numbers)."""
        with tempfile.TemporaryDirectory() as tmp:
            state_db = Path(tmp) / "state.db"
            db.init_db(state_db)
            with db.connect(state_db) as con:
                files = [_local_file(f"file-{index}.pdf", checksum=f"checksum-{index}") for index in range(10)]
                for file in files:
                    db.upsert_file(con, file, status="indexed")
                seen_ids = {file.file_key for file in files[:6]}

                buf = io.StringIO()
                with redirect_stdout(buf):
                    deleted, jobs = schedule_deletes_for_source(
                        con,
                        ingest_run_id="scan-large-delete",
                        kb_id="project_materials",
                        collection="kb_project",
                        seen_file_keys=seen_ids,
                    )
                self.assertEqual((deleted, jobs), (4, 4))
                self.assertIn("missing=4 current=10 ratio=40.0%", buf.getvalue())
                self.assertEqual(
                    con.execute("SELECT COUNT(*) FROM files WHERE status = 'deleted'").fetchone()[0], 4)
                self.assertEqual(con.execute("SELECT COUNT(*) FROM jobs").fetchone()[0], 4)

    def test_no_deletes_stays_silent(self) -> None:
        """Without deletions that log line must not appear — otherwise the once-a-minute scan turns it into noise,
        and a real incident goes unseen."""
        with tempfile.TemporaryDirectory() as tmp:
            state_db = Path(tmp) / "state.db"
            db.init_db(state_db)
            with db.connect(state_db) as con:
                files = [_local_file(f"keep-{index}.pdf", checksum=f"sum-{index}") for index in range(4)]
                for file in files:
                    db.upsert_file(con, file, status="indexed")
                buf = io.StringIO()
                with redirect_stdout(buf):
                    deleted, jobs = schedule_deletes_for_source(
                        con,
                        ingest_run_id="scan-nothing-gone",
                        kb_id="project_materials",
                        collection="kb_project",
                        seen_file_keys={file.file_key for file in files},
                    )
            self.assertEqual((deleted, jobs), (0, 0))
            self.assertNotIn("deletes detected", buf.getvalue())

    def test_hidden_parent_directories_are_skipped(self) -> None:
        self.assertTrue(should_skip(Path("/tmp/kb/.git/config")))
        self.assertTrue(should_skip(Path("/tmp/kb/folder/.private/file.md")))
        self.assertFalse(should_skip(Path("/tmp/kb/folder/file.md")))

    def test_a_mirror_under_a_hidden_directory_is_still_scanned(self) -> None:
        """Hidden directories are judged by the path inside the KB only. Every part of the absolute path used to
        count: with the mirror root under a location like ~/.local/share, every file was skipped as hidden, the
        scan found 0 files without an error, and the console's directory file count was 0 as well."""
        from kb_pipeline.localfs.scanner import list_recent_source_files, list_source_files
        from kb_server import service

        with tempfile.TemporaryDirectory() as tmp:
            mirror = Path(tmp) / ".local" / "share" / "mirror"
            root = mirror / "库"
            (root / "sub").mkdir(parents=True)
            (root / ".git").mkdir()
            (root / "sub" / ".private").mkdir()
            for rel in ("a.md", "sub/b.md", ".git/config.md", "sub/.private/c.md", ".hidden.md"):
                (root / rel).write_text("x", encoding="utf-8")
            source = KBSource(kb_id="k", collection="kb_k", source_root="库", source_type="local_mirror",
                              max_tokens=400, overlap_tokens=80, physical_base=root)
            self.assertEqual([f.rel_path for f in list_source_files(source, hash_content=False)], ["a.md", "sub/b.md"])
            self.assertEqual(sorted(p.name for p in list_recent_source_files(source, min_age_seconds=3600)), ["a.md", "b.md"])
            service._dir_count_cache.pop("库", None)
            self.assertEqual(service._dir_file_count(mirror, "库"), 2)
            service._dir_count_cache.pop("库", None)

    def test_a_modification_time_in_the_future_does_not_defer_the_file_forever(self) -> None:
        """The settling period is meant for files only just written. A file with a modification time in the future
        used to count as too recent in every round -- not parsed, not deleted, no error -- until that moment
        arrived. One further off than the settling period counts as stable; one off by a few seconds (the clock
        difference between two machines) still waits."""
        from kb_pipeline.localfs.scanner import list_recent_source_files, list_source_files

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "库"; root.mkdir()
            now = time.time()
            for name, mtime in (("settled.md", now - 3600), ("next-year.md", now + 365 * 86400),
                                ("skewed.md", now + 20), ("fresh.md", now - 5)):
                (root / name).write_text("x", encoding="utf-8")
                os.utime(root / name, (mtime, mtime))
            source = KBSource(kb_id="k", collection="kb_k", source_root="库", source_type="local_mirror",
                              max_tokens=400, overlap_tokens=80, physical_base=root)
            too_recent: set[int] = set()
            files = list_source_files(source, min_age_seconds=180, hash_content=False, too_recent_keys=too_recent)
            self.assertEqual([f.filename for f in files], ["next-year.md", "settled.md"])
            self.assertEqual(len(too_recent), 2)
            self.assertEqual(sorted(p.name for p in list_recent_source_files(source, min_age_seconds=180)),
                             ["fresh.md", "skewed.md"])


class ScannerChecksumReuseTests(unittest.TestCase):
    def test_recorded_checksum_reused_and_rehashed_on_change_or_move(self) -> None:
        import os

        from kb_pipeline.localfs.scanner import list_source_files
        from kb_pipeline.utils import sha256_file

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            f = root / "a.md"
            f.write_text("hello world\n", encoding="utf-8")
            st = f.stat()
            source = KBSource(
                kb_id="k", collection="c", source_root="r",
                source_type="local_mirror", max_tokens=400, overlap_tokens=80,
                physical_base=root,
            )

            known = {"a.md": (st.st_size, int(st.st_mtime), "recorded-checksum")}
            [got] = list_source_files(source, known_files=known)
            self.assertEqual(got.checksum, "recorded-checksum")

            # mtime bumped -> quick check fails -> a real hash is computed
            os.utime(f, (st.st_atime, st.st_mtime + 10))
            [got] = list_source_files(source, known_files=known)
            self.assertEqual(got.checksum, sha256_file(f))

            # a moved file arrives under a new rel_path, so a map entry under
            # the old path can never match and the file is freshly hashed.
            # (Regression: a file_key-keyed map matched nothing after a move --
            # move detection keeps the original key -- so moved files were
            # rehashed on every scan forever.)
            g = root / "b.md"
            f.rename(g)
            [got] = list_source_files(source, known_files=known)
            self.assertEqual(got.rel_path, "b.md")
            self.assertEqual(got.checksum, sha256_file(g))

            # once the new path is recorded, reuse resumes
            st2 = g.stat()
            known2 = {"b.md": (st2.st_size, int(st2.st_mtime), "recorded-2")}
            [got] = list_source_files(source, known_files=known2)
            self.assertEqual(got.checksum, "recorded-2")


class VisualRefRelativizationTests(unittest.TestCase):
    def test_visual_ref_is_stored_relative_to_cache_root(self) -> None:
        from kb_pipeline.pipeline.parse_job import _relative_visual_ref

        settings = SimpleNamespace(cache_dir=Path("/data/kb/runtime/parse_cache/kb-pipeline"))
        inside = "/data/kb/runtime/parse_cache/kb-pipeline/parse/k/1/v/mineru/images/x.jpg"
        self.assertEqual(_relative_visual_ref(settings, inside), "parse/k/1/v/mineru/images/x.jpg")
        # anything outside the cache root is left alone rather than corrupted
        outside = "/somewhere/else/x.jpg"
        self.assertEqual(_relative_visual_ref(settings, outside), outside)


class WorkerClaimTests(unittest.TestCase):
    """Claiming and leases previously had zero coverage: a wrong claim lets two workers process the same document
    at once, or lets a backed-off job be claimed early."""

    def _job(self, con, job_id: str, *, status: str = "queued", priority: int = 100,
             next_attempt_at: int = 0, job_type: str = "parse", kb_id: str = "kb_1",
             created_at: int = 1) -> None:
        con.execute(
            "INSERT INTO jobs(job_id, kb_id, collection, job_type, status, priority, "
            "next_attempt_at, created_at, updated_at) VALUES(?, ?, 'c', ?, ?, ?, ?, ?, 1)",
            (job_id, kb_id, job_type, status, priority, next_attempt_at, created_at),
        )

    def test_claim_order_backoff_and_type_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.db"
            db.init_db(path)
            now = int(time.time())
            with db.connect(path) as con:
                self._job(con, "later", priority=100, created_at=200)
                self._job(con, "urgent", priority=50, created_at=300)     # higher priority
                self._job(con, "waiting", priority=10, next_attempt_at=now + 999)  # backing off
                con.commit()

                first = db.claim_job_for_types(con, "w1", lease_seconds=60)
                self.assertEqual(first["job_id"], "urgent")               # taken by priority
                second = db.claim_job_for_types(con, "w2", lease_seconds=60)
                self.assertEqual(second["job_id"], "later")
                # A job whose backoff has not expired cannot be claimed
                self.assertIsNone(db.claim_job_for_types(con, "w3", lease_seconds=60))

    def test_expired_lease_is_taken_over_but_live_one_is_not(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.db"
            db.init_db(path)
            now = int(time.time())
            with db.connect(path) as con:
                self._job(con, "dead", status="running")
                con.execute("UPDATE jobs SET locked_by='otherhost:1', locked_until=? WHERE job_id='dead'",
                            (now - 10,))          # lease expired
                self._job(con, "alive", status="running", created_at=2)
                con.execute("UPDATE jobs SET locked_by='otherhost:2', locked_until=? WHERE job_id='alive'",
                            (now + 600,))         # lease valid
                con.commit()
                claimed = db.claim_job_for_types(con, "w1", lease_seconds=60)
                self.assertEqual(claimed["job_id"], "dead")   # only the expired one is taken over
                self.assertIsNone(db.claim_job_for_types(con, "w2", lease_seconds=60))

    def test_claim_filters_by_type_and_kb(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.db"
            db.init_db(path)
            with db.connect(path) as con:
                self._job(con, "p", job_type="parse", kb_id="kb_1")
                self._job(con, "m", job_type="metadata_update", kb_id="kb_2", created_at=2)
                con.commit()
                # With parsing off only lifecycle jobs are claimed (the behaviour of worker.run_once)
                got = db.claim_job_for_types(con, "w", lease_seconds=60,
                                             allowed_job_types={"metadata_update", "delete"})
                self.assertEqual(got["job_id"], "m")
                db.release_job(con, "m")
                # Restricted to a knowledge base
                got = db.claim_job_for_types(con, "w", lease_seconds=60, allowed_kb_ids={"kb_1"})
                self.assertEqual(got["job_id"], "p")

    def test_terminal_updates_respect_the_lease(self) -> None:
        """An ousted worker (its lease taken over) must not mark the job done any more."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.db"
            db.init_db(path)
            with db.connect(path) as con:
                self._job(con, "j1")
                con.commit()
                db.claim_job_for_types(con, "w1", lease_seconds=60)
                con.execute("UPDATE jobs SET status='cancelled' WHERE job_id='j1'")   # cancelled by a KB deletion in the meantime
                con.commit()
                db.mark_job_done(con, "j1")
                status = con.execute("SELECT status FROM jobs WHERE job_id='j1'").fetchone()[0]
                self.assertEqual(status, "cancelled")     # not overwritten to done


class ScanBoundaryTests(unittest.TestCase):
    """The scan's boundaries had almost no coverage, yet it runs every minute and can delete data."""

    def _settings(self, root: Path, state: Path):
        return SimpleNamespace(
            state_db=state, mirror_root=root, runtime_dir=root / "rt",
            min_file_age_seconds=30, sources={},
        )

    def test_mirror_root_missing_refuses_to_touch_anything(self) -> None:
        """A missing mirror root is indistinguishable from "all KBs vanished at once": the whole round of lifecycle
        processing must be refused, otherwise one failed mount marks every knowledge base deactivated and fills the
        queue with delete jobs."""
        import argparse
        from unittest import mock

        from kb_pipeline import cli, discovery

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "mirror"; root.mkdir()
            (root / "库A").mkdir()
            state = Path(tmp) / "s.db"; db.init_db(state)
            with db.connect(state) as con:
                src, _ = discovery.enroll(con, root, "库A")
                con.commit()
            import shutil as _sh
            _sh.rmtree(root / "库A")         # make the directory vanish first, then load sources in production order
            settings = self._settings(root, state)
            # enrolled_sources excludes KBs whose directory is gone — so this round "found none at all", which is
            # indistinguishable from "the mirror root is not mounted", and the whole lifecycle round must be refused
            settings.sources = discovery.enrolled_sources(state, root)
            self.assertEqual(settings.sources, {})
            args = argparse.Namespace(env_file=None, source=None, limit=None, verbose=False,
                                      dry_run=False, rehash=False, requeue_failed=False,
                                      no_detect_deletes=False, force_kb_teardown=False,
                                      exit_code_on_recent=False)
            with mock.patch.object(cli, "load_settings", return_value=settings):
                code = cli.cmd_scan(args)
            self.assertEqual(code, 2)        # refuses and exits with an error
            with db.connect(state) as con:
                status = con.execute("SELECT status FROM kb_sources WHERE kb_id=?", (src.kb_id,)).fetchone()[0]
            self.assertEqual(status, "active")   # the status was not touched

    def test_scan_leaves_a_refused_linked_kb_alone(self) -> None:
        """Third review, item 1: with two enrolled KBs, one of them swapped for a symbolic link, a full scan must
        not read the refused one as vanished: it stays active, its file rows stay, and no delete job is queued.
        Only the other KB is scanned."""
        import argparse
        import shutil
        from unittest import mock

        from kb_pipeline import cli, discovery

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "mirror"; root.mkdir()
            outside = Path(tmp) / "outside"; outside.mkdir()
            (outside / "synthetic.md").write_text("# outside", encoding="utf-8")
            for name in ("库A", "库B"):
                (root / name).mkdir()
                doc = root / name / "a.md"; doc.write_text(f"# {name}\n\n内容", encoding="utf-8")
                os.utime(doc, (time.time() - 3600, time.time() - 3600))
            state = Path(tmp) / "s.db"; db.init_db(state)
            with db.connect(state) as con:
                a, _ = discovery.enroll(con, root, "库A"); b, _ = discovery.enroll(con, root, "库B"); con.commit()
            args = argparse.Namespace(env_file=None, source=None, limit=None, verbose=False, dry_run=False, rehash=False,
                                      requeue_failed=False, no_detect_deletes=False, force_kb_teardown=False,
                                      exit_code_on_recent=False)
            with patch.dict(os.environ, {"KB_MIRROR_ALLOW_LINKED_DIRS": ""}):
                settings = self._settings(root, state)
                settings.sources = discovery.enrolled_sources(state, root)
                with mock.patch.object(cli, "load_settings", return_value=settings):
                    self.assertEqual(cli.cmd_scan(args), 0)                   # both directories enrolled their file
                with db.connect(state) as con:
                    self.assertEqual(con.execute("SELECT COUNT(*) FROM files WHERE kb_id=?", (a.kb_id,)).fetchone()[0], 1)
                shutil.rmtree(root / "库A"); (root / "库A").symlink_to(outside, target_is_directory=True)
                settings.sources = discovery.enrolled_sources(state, root)
                self.assertEqual(set(settings.sources), {b.kb_id})
                with mock.patch.object(cli, "load_settings", return_value=settings):
                    self.assertEqual(cli.cmd_scan(args), 0)
                with db.connect(state) as con:
                    status = con.execute("SELECT status FROM kb_sources WHERE kb_id=?", (a.kb_id,)).fetchone()[0]
                    files = con.execute("SELECT COUNT(*) FROM files WHERE kb_id=? AND status!='deleted'", (a.kb_id,)).fetchone()[0]
                    deletes = con.execute("SELECT COUNT(*) FROM jobs WHERE job_type='delete'").fetchone()[0]
                    outside_rows = con.execute("SELECT COUNT(*) FROM files WHERE source_path LIKE '%synthetic%'").fetchone()[0]
                self.assertEqual((status, files, deletes, outside_rows), ("active", 1, 0, 0))

    def test_scan_skips_a_kb_with_broken_config_and_lists_only_changes(self) -> None:
        """2026-09-29 audit: one base whose config_json cannot be read made the whole scan raise and exit, and every
        base stopped ingesting; without the exception that base would have been taken for a vanished directory,
        deactivated and queued for deletion. Only that base is skipped now. Also: --verbose lists the files
        whose state changed, not every unchanged one."""
        import argparse
        from unittest import mock

        from kb_pipeline import cli, discovery

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "mirror"; root.mkdir()
            for name in ("base-a", "base-b"):
                (root / name).mkdir()
                doc = root / name / "a.md"; doc.write_text(f"# {name}\n\ncontent", encoding="utf-8")
                os.utime(doc, (time.time() - 3600, time.time() - 3600))
            state = Path(tmp) / "s.db"; db.init_db(state)
            with db.connect(state) as con:
                a, _ = discovery.enroll(con, root, "base-a"); b, _ = discovery.enroll(con, root, "base-b"); con.commit()
            args = argparse.Namespace(env_file=None, source=None, limit=None, verbose=True, dry_run=False, rehash=False,
                                      requeue_failed=False, no_detect_deletes=False, force_kb_teardown=False,
                                      exit_code_on_recent=False)
            settings = self._settings(root, state)
            settings.sources = discovery.enrolled_sources(state, root)
            out = io.StringIO()
            with mock.patch.object(cli, "load_settings", return_value=settings), redirect_stdout(out):
                self.assertEqual(cli.cmd_scan(args), 0)
            self.assertEqual(len([l for l in out.getvalue().splitlines() if l.startswith("  ") and "a.md" in l]), 2)   # two new files
            with db.connect(state) as con:
                for row in con.execute("SELECT file_id, content_version FROM files").fetchall():      # as if both were indexed
                    db.mark_file_indexed(con, str(row["file_id"]), str(row["content_version"]), parser_profile_for_path(Path("a.md")))
                con.execute("UPDATE jobs SET status = 'done', finished_at = ?", (int(time.time()),))
                con.execute("UPDATE kb_sources SET config_json = ? WHERE kb_id = ?",
                            (json.dumps({"graph_rebuild_interval": "weekly"}), a.kb_id))
                with self.assertRaises(Exception):
                    discovery.source_from_row(root, con.execute("SELECT * FROM kb_sources WHERE kb_id = ?", (a.kb_id,)).fetchone())
            settings.sources = discovery.enrolled_sources(state, root)
            self.assertEqual(set(settings.sources), {b.kb_id})                     # the base with the broken config is not a source
            out = io.StringIO()
            with mock.patch.object(cli, "load_settings", return_value=settings), redirect_stdout(out):
                self.assertEqual(cli.cmd_scan(args), 0)                            # the round completes as usual
            self.assertEqual([l for l in out.getvalue().splitlines() if l.startswith("  ") and "a.md" in l], [])   # unchanged files are not listed
            self.assertIn("unchanged=1", out.getvalue())
            with db.connect(state) as con:
                status = con.execute("SELECT status FROM kb_sources WHERE kb_id=?", (a.kb_id,)).fetchone()[0]
                files = con.execute("SELECT COUNT(*) FROM files WHERE kb_id=? AND status!='deleted'", (a.kb_id,)).fetchone()[0]
                deletes = con.execute("SELECT COUNT(*) FROM jobs WHERE job_type='delete'").fetchone()[0]
            self.assertEqual((status, files, deletes), ("active", 1, 0))           # not deactivated, nothing queued for deletion
            args.list_unchanged = True
            out = io.StringIO()
            with mock.patch.object(cli, "load_settings", return_value=settings), redirect_stdout(out):
                self.assertEqual(cli.cmd_scan(args), 0)
            self.assertEqual(len([l for l in out.getvalue().splitlines() if l.startswith("  unchanged") and "a.md" in l]), 1)

    def test_scan_survives_a_closed_kb(self) -> None:
        """2026-09-06: with a KB in the closed state, the deactivated placeholder tuple had one field fewer than the
        scan tuple, so the scan crashed every minute and new files of other KBs could not get in either."""
        import argparse
        from unittest import mock

        from kb_pipeline import cli, discovery

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "mirror"
            root.mkdir()
            (root / "库A").mkdir()
            (root / "库B").mkdir()
            (root / "库A" / "a.md").write_text("# A\n\n内容", encoding="utf-8")
            state = Path(tmp) / "s.db"
            db.init_db(state)
            with db.connect(state) as con:
                discovery.enroll(con, root, "库A")
                closed, _ = discovery.enroll(con, root, "库B")
                discovery.mark_inactive(con, closed.kb_id, reason="unenrolled")
                con.commit()
            settings = self._settings(root, state)
            settings.sources = discovery.enrolled_sources(state, root)
            self.assertEqual(set(settings.sources), {"kb_001"})
            args = argparse.Namespace(env_file=None, source=None, limit=None, verbose=False,
                                      dry_run=False, rehash=False, requeue_failed=False,
                                      no_detect_deletes=False, force_kb_teardown=False,
                                      exit_code_on_recent=False)
            with mock.patch.object(cli, "load_settings", return_value=settings):
                code = cli.cmd_scan(args)
            self.assertEqual(code, 0)
            with db.connect(state) as con:
                status = con.execute("SELECT status FROM kb_sources WHERE kb_id=?", (closed.kb_id,)).fetchone()[0]
            self.assertEqual(status, "inactive")     # the closed KB stays closed, untouched by the scan

    def _scan_args(self, **overrides):
        import argparse

        values = dict(env_file=None, source=None, limit=None, verbose=False, dry_run=False, rehash=False,
                      requeue_failed=False, no_detect_deletes=False, force_kb_teardown=False, exit_code_on_recent=False)
        values.update(overrides)
        return argparse.Namespace(**values)

    def _one_kb(self, tmp: str):
        from kb_pipeline import discovery

        root = Path(tmp) / "mirror"; (root / "库A").mkdir(parents=True)
        doc = root / "库A" / "a.md"; doc.write_text("# A\n\n内容", encoding="utf-8")
        os.utime(doc, (time.time() - 3600, time.time() - 3600))
        state = Path(tmp) / "s.db"; db.init_db(state)
        with db.connect(state) as con:
            src, _ = discovery.enroll(con, root, "库A"); con.commit()
        settings = self._settings(root, state)
        settings.sources = discovery.enrolled_sources(state, root)
        return root, state, settings, src

    def _scan(self, settings, **overrides) -> tuple[int, str]:
        from unittest import mock

        from kb_pipeline import cli

        out = io.StringIO()
        with mock.patch.object(cli, "load_settings", return_value=settings), redirect_stdout(out):
            code = cli.cmd_scan(self._scan_args(**overrides))
        return code, out.getvalue()

    def test_a_scan_that_did_nothing_leaves_no_run_record(self) -> None:
        """The scan runs every minute and in most rounds nothing happens, yet every round used to write an
        ingest_runs row (over 1,200 all-zero records a day)."""
        with tempfile.TemporaryDirectory() as tmp:
            root, state, settings, src = self._one_kb(tmp)

            def runs() -> list[tuple]:
                with db.connect(state) as con:
                    return [tuple(r) for r in con.execute(
                        "SELECT status, added_count, updated_count, deleted_count, note FROM ingest_runs ORDER BY started_at, rowid")]

            self.assertEqual(self._scan(settings)[0], 0)
            self.assertEqual(runs(), [("done", 1, 0, 0, None)])                # a round that queued jobs is recorded as before
            with db.connect(state) as con:
                for row in con.execute("SELECT file_id, content_version FROM files").fetchall():
                    db.mark_file_indexed(con, str(row["file_id"]), str(row["content_version"]), parser_profile_for_path(Path("a.md")))
                con.execute("UPDATE jobs SET status = 'done', finished_at = ?", (int(time.time()),))
            code, out = self._scan(settings)
            self.assertIn("unchanged=1 jobs=0", out)
            self.assertEqual((code, len(runs())), (0, 1))                      # an idle round leaves no record
            (root / "库A" / "a.md").unlink()
            self.assertEqual(self._scan(settings)[0], 0)
            self.assertEqual(runs()[1:], [("done", 0, 0, 1, None)])            # a deletion is an event too
            import shutil
            shutil.rmtree(root)
            self.assertEqual(self._scan(settings)[0], 2)                       # mirror root gone: refused, no record either
            self.assertEqual(len(runs()), 2)

    def test_a_job_waiting_in_backoff_is_not_queued_again_every_minute(self) -> None:
        """While a parse job backed off, every scan round "queued" the same job again and counted it in jobs=N, so
        the wrapper script kicked the worker for nothing every minute."""
        with tempfile.TemporaryDirectory() as tmp:
            _, state, settings, _ = self._one_kb(tmp)
            self.assertIn("added=1", self._scan(settings)[1])
            with db.connect(state) as con:
                con.execute("UPDATE jobs SET status = 'retry', retry_count = 1, next_attempt_at = ?, updated_at = 5",
                            (int(time.time()) + 600,))
            code, out = self._scan(settings)
            self.assertIn("needs_parse=1", out)
            self.assertIn("jobs=0", out)
            with db.connect(state) as con:
                rows = con.execute("SELECT status, updated_at FROM jobs").fetchall()
                runs = con.execute("SELECT COUNT(*) FROM ingest_runs").fetchone()[0]
            self.assertEqual([tuple(r) for r in rows], [("retry", 5)])          # still the same job, untouched
            self.assertEqual(runs, 1)

    def test_a_kb_deleted_or_closed_while_the_scan_lists_files_is_left_alone(self) -> None:
        """The source list is read when the scan starts. A KB permanently deleted during the seconds spent listing
        files used to get its file rows and parse jobs inserted again by the scan, under an id that no longer
        exists, where no cleanup could ever find them; a closed KB got jobs queued again."""
        from unittest import mock

        from kb_pipeline import cli, discovery

        for action in ("delete", "close"):
            with tempfile.TemporaryDirectory() as tmp:
                _, state, settings, src = self._one_kb(tmp)
                real_list = cli.list_source_files

                def list_then_lose_the_kb(source, **kwargs):
                    files = real_list(source, **kwargs)
                    with db.connect(state) as con:
                        if action == "delete":
                            db.purge_kb_state(con, source.kb_id)
                            discovery.forget(con, source.kb_id)
                        else:
                            discovery.mark_inactive(con, source.kb_id, reason="unenrolled")
                    return files

                with mock.patch.object(cli, "list_source_files", list_then_lose_the_kb):
                    code, out = self._scan(settings)
                self.assertEqual(code, 0, action)
                with db.connect(state) as con:
                    counts = tuple(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                                   for table in ("files", "jobs", "ingest_runs"))
                self.assertEqual(counts, (0, 0, 0), action)

    def test_recent_pending_exit_code(self) -> None:
        """Exits with 75 while files are still settling; the script keeps the flag for the next round based on that."""
        import argparse
        from unittest import mock

        from kb_pipeline import cli, discovery

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "mirror"; (root / "库A").mkdir(parents=True)
            (root / "库A" / "新文件.txt").write_text("just written", encoding="utf-8")
            state = Path(tmp) / "s.db"; db.init_db(state)
            with db.connect(state) as con:
                discovery.enroll(con, root, "库A"); con.commit()
            settings = self._settings(root, state)
            settings.sources = discovery.enrolled_sources(state, root)
            args = argparse.Namespace(env_file=None, source=None, limit=None, verbose=False,
                                      dry_run=False, rehash=False, requeue_failed=False,
                                      no_detect_deletes=False, force_kb_teardown=False,
                                      exit_code_on_recent=True)
            with mock.patch.object(cli, "load_settings", return_value=settings):
                self.assertEqual(cli.cmd_scan(args), 75)


class ScannerContractTests(unittest.TestCase):
    def test_sync_temporaries_are_skipped(self) -> None:
        """rsync intermediates must not be indexed: they would be parsed with half their content and then vanish."""
        from kb_pipeline.localfs.scanner import should_skip

        for name in (".doc.txt.AbC123", ".~tmp~/x.pdf", ".rsync-partial/y.pdf",
                     ".git/config", ".DS_Store"):
            self.assertTrue(should_skip(Path("/mirror/库") / name), name)
        for name in ("正常.pdf", "sub/正常.md"):
            self.assertFalse(should_skip(Path("/mirror/库") / name), name)


class WorkerBranchTests(unittest.TestCase):
    """The branch matrix of run_once previously had zero coverage — it decides the final status written for every job."""

    def _settings(self, state: Path):
        return SimpleNamespace(
            state_db=state, parse_enabled=True, parse_job_lease_seconds=600,
            metadata_job_lease_seconds=600, job_max_retries=2, job_retry_base_seconds=300,
            job_retry_max_seconds=3600, qdrant_url="http://q", qdrant_api_key="",
            opensearch_url="http://o", sources={},
        )

    def _file_and_job(self, con, *, status: str = "seen", job_type: str = "parse") -> tuple[str, str]:
        file = _local_file("dir/a.pdf", checksum="c1")
        db.upsert_file(con, file, status=status)
        fid = db.file_id_for(file.kb_id, file.file_key)
        job_id = db.enqueue_job(con, ingest_run_id=None, file_id=fid, kb_id=file.kb_id,
                                collection=file.collection, file_key=file.file_key,
                                job_type=job_type, dedupe_key=f"{job_type}:{fid}")
        return fid, job_id

    def test_parse_of_deleted_file_is_cancelled_not_failed(self) -> None:
        from kb_pipeline.pipeline.worker import run_once

        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"; db.init_db(state)
            with db.connect(state) as con:
                fid, job_id = self._file_and_job(con)
                db.mark_file_deleted(con, fid)      # the file was deleted while queued
                con.commit()
                result = run_once(con, settings=self._settings(state))
                status = con.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,)).fetchone()[0]
            self.assertIn("cancelled", result)
            self.assertEqual(status, "cancelled")   # not failed: this is not an error

    def test_parse_disabled_pauses_instead_of_consuming(self) -> None:
        from kb_pipeline.pipeline.worker import run_once

        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"; db.init_db(state)
            settings = self._settings(state)
            settings.parse_enabled = False
            with db.connect(state) as con:
                _, job_id = self._file_and_job(con)
                con.commit()
                # With parsing off only lifecycle jobs are claimed; a parse job cannot be claimed
                self.assertEqual(run_once(con, settings=settings), "no-job")
                status = con.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,)).fetchone()[0]
            self.assertEqual(status, "queued")      # stays in the queue, not lost

    def test_failure_marks_retry_with_backoff_then_gives_up(self) -> None:
        from unittest import mock

        from kb_pipeline.pipeline import worker as worker_mod

        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"; db.init_db(state)
            settings = self._settings(state)
            with db.connect(state) as con:
                _, job_id = self._file_and_job(con)
                con.commit()
                with mock.patch.object(worker_mod, "process_parse_job",
                                       side_effect=RuntimeError("MinerU 超时")):
                    for expected in ("retry", "retry", "failed"):   # max_retries=2
                        con.execute("UPDATE jobs SET next_attempt_at=0 WHERE job_id=?", (job_id,))
                        con.commit()
                        result = worker_mod.run_once(con, settings=settings)
                        self.assertTrue(result.startswith(expected), f"{result} 应以 {expected} 开头")
                failures = con.execute("SELECT COUNT(*) FROM failures WHERE job_id=?", (job_id,)).fetchone()[0]
            self.assertEqual(failures, 3)          # every failure leaves a trace

    def test_deterministic_failure_does_not_retry(self) -> None:
        from unittest import mock

        from kb_pipeline.pipeline import worker as worker_mod
        from kb_pipeline.pipeline.parse_job import NonRetryableParseError

        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"; db.init_db(state)
            with db.connect(state) as con:
                _, job_id = self._file_and_job(con)
                con.commit()
                with mock.patch.object(worker_mod, "process_parse_job",
                                       side_effect=NonRetryableParseError("格式不支持")):
                    result = worker_mod.run_once(con, settings=self._settings(state))
                status = con.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,)).fetchone()[0]
            self.assertTrue(result.startswith("failed"))   # judged failed outright
            self.assertEqual(status, "failed")             # no 5 idle backoff rounds


class JobCancellationTests(unittest.TestCase):
    """Closing / deleting a knowledge base must be able to stop a running parse. Cancellation goes through a flag
    rather than writing status directly: a running row belongs to the worker, and an outsider racing to write a
    terminal state leaves a job that looks stopped while still writing to the vector store."""

    def _settings(self, state: Path):
        return SimpleNamespace(
            state_db=state, parse_enabled=True, parse_job_lease_seconds=600,
            metadata_job_lease_seconds=600, job_max_retries=2, job_retry_base_seconds=300,
            job_retry_max_seconds=3600, qdrant_url="http://q", qdrant_api_key="",
            opensearch_url="http://o", sources={},
        )

    def _job(self, con, job_id: str, *, kb_id: str = "kb_1", status: str = "queued",
             locked_by: str | None = None, locked_until: int | None = None) -> None:
        con.execute(
            "INSERT INTO jobs(job_id, kb_id, collection, job_type, status, priority, "
            "next_attempt_at, created_at, updated_at, locked_by, locked_until) "
            "VALUES(?, ?, 'c', 'parse', ?, 100, 0, 1, 1, ?, ?)",
            (job_id, kb_id, status, locked_by, locked_until),
        )

    def test_request_cancel_voids_queue_and_flags_the_running_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"; db.init_db(state)
            now = int(time.time())
            with db.connect(state) as con:
                self._job(con, "q", status="queued")
                self._job(con, "r", status="retry")
                self._job(con, "run", status="running", locked_by="h:1", locked_until=now + 600)
                self._job(con, "other", kb_id="kb_2", status="queued")
                self._job(con, "done", status="done")
                con.commit()

                counts = db.request_kb_job_cancel(con, "kb_1", "closed")
                self.assertEqual(counts, {"cancelled": 2, "signalled": 1})
                rows = {r["job_id"]: r for r in con.execute("SELECT * FROM jobs").fetchall()}
                self.assertEqual(rows["q"]["status"], "cancelled")
                self.assertEqual(rows["r"]["status"], "cancelled")
                # running only gets the flag; the status still belongs to the worker
                self.assertEqual(rows["run"]["status"], "running")
                self.assertEqual(int(rows["run"]["cancel_requested"] or 0), 1)
                self.assertEqual(rows["other"]["status"], "queued")   # other KBs are not affected
                self.assertEqual(rows["done"]["status"], "done")      # terminal states are not rolled back

    def test_cancel_requested_is_true_for_flagged_missing_and_finished(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"; db.init_db(state)
            now = int(time.time())
            with db.connect(state) as con:
                self._job(con, "live", status="running", locked_by="h:1", locked_until=now + 600)
                self._job(con, "gone_status", status="cancelled")
                con.commit()
                self.assertFalse(db.job_cancel_requested(con, "live"))
                db.request_kb_job_cancel(con, "kb_1", "closed")
                self.assertTrue(db.job_cancel_requested(con, "live"))
                # The row was purged by a KB deletion: continuing would only write orphan data, so it is treated as cancelled too
                self.assertTrue(db.job_cancel_requested(con, "no-such-job"))
                self.assertTrue(db.job_cancel_requested(con, "gone_status"))

    def test_running_leases_ignore_expired_ones(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"; db.init_db(state)
            now = int(time.time())
            with db.connect(state) as con:
                self._job(con, "live", status="running", locked_by="host:42", locked_until=now + 600)
                self._job(con, "dead", status="running", locked_by="host:43", locked_until=now - 10)
                con.commit()
                self.assertEqual(db.running_job_leases(con, "kb_1"), [("live", "host:42")])

    def test_killed_worker_of_a_cancelled_job_is_not_retried(self) -> None:
        """Deleting a KB SIGTERMs the worker. The recovery flow must recognise this as "cancelled" rather than
        "crashed", otherwise the job goes back to the queue and the next worker round keeps parsing a KB that is
        already closed."""
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"; db.init_db(state)
            now = int(time.time())
            with db.connect(state) as con:
                self._job(con, "run", status="running", locked_by="h:1", locked_until=now + 600)
                con.commit()
                db.request_kb_job_cancel(con, "kb_1", "deleting")
                outcome = db.release_crashed_job(con, "run", max_retries=3, retry_delay_seconds=300)
                row = con.execute("SELECT status FROM jobs WHERE job_id='run'").fetchone()
                failures = con.execute("SELECT COUNT(*) FROM failures WHERE job_id='run'").fetchone()[0]
            self.assertEqual(outcome, "cancelled")
            self.assertEqual(row["status"], "cancelled")
            self.assertEqual(failures, 0)          # a cancellation is not a fault and must leave no failure record

    def test_worker_maps_job_cancelled_to_cancelled_without_failure_row(self) -> None:
        from unittest import mock

        from kb_pipeline.pipeline import worker as worker_mod
        from kb_pipeline.pipeline.parse_job import JobCancelled

        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"; db.init_db(state)
            with db.connect(state) as con:
                file = _local_file("dir/a.pdf", checksum="c1")
                db.upsert_file(con, file, status="seen")
                fid = db.file_id_for(file.kb_id, file.file_key)
                job_id = db.enqueue_job(con, ingest_run_id=None, file_id=fid, kb_id=file.kb_id,
                                        collection=file.collection, file_key=file.file_key,
                                        job_type="parse", dedupe_key=f"parse:{fid}")
                con.commit()
                with mock.patch.object(worker_mod, "process_parse_job",
                                       side_effect=JobCancelled("closed mid-parse")):
                    result = worker_mod.run_once(con, settings=self._settings(state))
                status, retry = con.execute(
                    "SELECT status, retry_count FROM jobs WHERE job_id=?", (job_id,)).fetchone()
                failures = con.execute(
                    "SELECT COUNT(*) FROM failures WHERE job_id=?", (job_id,)).fetchone()[0]
            self.assertTrue(result.startswith("cancelled:"), result)
            self.assertEqual(status, "cancelled")
            self.assertEqual(int(retry or 0), 0)   # no retry counted
            self.assertEqual(failures, 0)          # no fault recorded

    def test_worker_leaves_a_job_alone_once_another_worker_has_recovered_it(self) -> None:
        """2026-09-09: two workers running side by side, A recovers the job B is running as an orphan into retry, and
        B sees the changed status at its next checkpoint and raises JobCancelled — B used to overwrite retry with
        cancelled, leaving the file stuck on the old parse version forever. Now a job it does not own is not
        touched at all."""
        from unittest import mock

        from kb_pipeline.pipeline import worker as worker_mod
        from kb_pipeline.pipeline.parse_job import JobCancelled

        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"; db.init_db(state)
            with db.connect(state) as con:
                file = _local_file("dir/a.pdf", checksum="c1")
                db.upsert_file(con, file, status="seen")
                fid = db.file_id_for(file.kb_id, file.file_key)
                job_id = db.enqueue_job(con, ingest_run_id=None, file_id=fid, kb_id=file.kb_id,
                                        collection=file.collection, file_key=file.file_key,
                                        job_type="parse", dedupe_key=f"parse:{fid}")
                con.commit()

                def recovered_elsewhere(con_, settings, job):
                    # another worker's _recover_or_refresh_running_jobs has already recovered it into retry
                    outcome = db.release_crashed_job(con_, job["job_id"], max_retries=3, retry_delay_seconds=300)
                    assert outcome == "retry"
                    raise JobCancelled(f"job {job['job_id']} cancelled by operator")

                with mock.patch.object(worker_mod, "process_parse_job", side_effect=recovered_elsewhere):
                    result = worker_mod.run_once(con, settings=self._settings(state))
                row = con.execute("SELECT status, retry_count, locked_by FROM jobs WHERE job_id=?", (job_id,)).fetchone()
                events = [r[0] for r in con.execute("SELECT kind FROM job_events WHERE job_id=?", (job_id,))]
            self.assertTrue(result.startswith("superseded:"), result)
            self.assertEqual((row["status"], int(row["retry_count"]), row["locked_by"]), ("retry", 1, None))   # left for the scheduled retry
            self.assertNotIn("cancelled", events)
            # An explicitly requested cancellation (KB closed) still lands on cancelled, even when the status is no longer running
            with db.connect(state) as con:
                con.execute("UPDATE jobs SET cancel_requested = 1 WHERE job_id=?", (job_id,))
                self.assertTrue(db.cancel_job_if_owned(con, job_id, "someone-else", "closed"))
                self.assertEqual(con.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,)).fetchone()[0], "cancelled")

    def test_orphan_check_recognises_both_ways_of_launching_a_worker(self) -> None:
        """systemd's `python -m kb_pipeline … worker --once` and a manual `.venv/bin/kb worker --once` are both live
        workers; only other processes (a reused pid) count as orphans."""
        from unittest import mock

        from kb_pipeline.pipeline import worker as worker_mod

        ok = ["/x/.venv/bin/python -m kb_pipeline --env-file /x/config/knowledge-base.env worker --once --max-jobs 10 --max-seconds 4382",
              "/x/.venv/bin/python /x/.venv/bin/kb worker --once --source kb_002 --max-jobs 130 --max-seconds 0",
              "kb worker --once"]
        bad = ["/x/.venv/bin/python -m http.server", "bash /x/scripts/kb-pipeline-worker-once.sh", "kb scan --source kb_002",
               "/x/.venv/bin/kb graph build --source kb_001", ""]
        for cmd in ok:
            self.assertTrue(worker_mod._looks_like_worker_command(cmd), cmd)
        for cmd in bad:
            self.assertFalse(worker_mod._looks_like_worker_command(cmd), cmd)
        fake = SimpleNamespace(returncode=0, stdout="/x/.venv/bin/python /x/.venv/bin/kb worker --once --source kb_002\n")
        with mock.patch.object(worker_mod.subprocess, "run", return_value=fake):
            self.assertTrue(worker_mod._pid_is_worker(os.getpid()))
        with mock.patch.object(worker_mod.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="python -m http.server\n")):
            self.assertFalse(worker_mod._pid_is_worker(os.getpid()))


@contextmanager
def _parse_harness(tmp: str):
    """Run the real worker and process_parse_job with only the external services (parsers, embedding, Qdrant,
    OpenSearch) stubbed. Provides an enrolled KB, one file on disk and its queued parse job; h.hooks[step] is an
    action run first when that step is reached, h.calls records the steps passed in order, h.args[step] holds the
    arguments that step received last, and h.results[step] replaces that step's return value."""
    from unittest import mock

    from kb_pipeline import discovery
    from kb_pipeline.localfs.scanner import list_source_files
    from kb_pipeline.pipeline import parse_job as pj
    from kb_pipeline.vector.layout import VectorLayout

    base = Path(tmp)
    mirror = base / "mirror"; (mirror / "库A").mkdir(parents=True)
    (mirror / "库A" / "a.md").write_text("# 标题\n\n正文", encoding="utf-8")
    state = base / "s.db"; db.init_db(state)
    with db.connect(state) as con:
        src, _ = discovery.enroll(con, mirror, "库A"); con.commit()
    [file] = list_source_files(discovery.enrolled_sources(state, mirror)[src.kb_id])
    settings = SimpleNamespace(
        state_db=state, mirror_root=mirror, cache_dir=base / "cache", sources={}, max_file_bytes=0,
        vlm_failure_retry_ratio=0.001, embedding_base_url="http://e", embedding_api_key="", embedding_model_id="emb",
        embedding_dim=4, embedding_batch=8, embedding_retry=1, embedding_sleep_seconds=0.0,
        qdrant_url="http://q", qdrant_api_key="", qdrant_upsert_max_bytes=1 << 20, vector_layout=VectorLayout(4, 4),
        opensearch_url="http://o", parse_enabled=True, parse_job_lease_seconds=600, metadata_job_lease_seconds=600,
        job_max_retries=2, job_retry_base_seconds=300, job_retry_max_seconds=3600,
        visual_embedding_enabled=True, visual_embedding_base_url="http://ve", visual_embedding_api_key="",
        visual_embedding_model_id="ve-model", visual_embedding_dim=4, visual_embedding_instruction="x",
        visual_embedding_concurrency=1, visual_embedding_retry=1, visual_embedding_timeout_seconds=5,
        image_max_pixels=1000)
    h = SimpleNamespace(settings=settings, kb_id=src.kb_id, file_id=db.file_id_for(file.kb_id, file.file_key),
                        calls=[], hooks={}, args={}, results={})

    def step(name: str, result=None):
        def run(*args, **kwargs):
            h.calls.append(name)
            h.args[name] = args
            if name in h.hooks:
                h.hooks[name]()
            return h.results.get(name, result)
        return run

    class Embedder:
        def __init__(self, **kwargs) -> None:
            pass

        def embed(self, texts):
            step("embed")()
            return [[0.0] * 4 for _ in texts]

    stubs = {
        "_parse_blocks": step("parse", [_block("b1", "正文内容,足够切成一片。")]),
        "EmbeddingClient": Embedder,
        "_embed_visual_chunks": step("visual", {}),
        "qdrant_client": step("qdrant"),
        "collection_exists": step("collection", True),
        "validate_collection_layout": step("layout"),
        "upsert_chunks": step("upsert"),
        "mark_old_versions_inactive": step("mark_old"),
        "mark_stale_file_points_inactive": step("mark_stale", 0),
    }
    with db.connect(state) as con, ExitStack() as stack:
        _, h.job_id = schedule_file(con, ingest_run_id="r1", file=file,
                                    current_seen_file_keys=set(), current_checksum_counts={})
        con.commit()
        for name, stub in stubs.items():
            stack.enter_context(mock.patch.object(pj, name, stub))
        stack.enter_context(mock.patch.object(
            pj.search_fts, "sync_doc_from_qdrant", step("fts", {"doc_id": "d", "inserted_rows": 1, "deleted_rows": 0})))
        h.con = con
        yield h


class ParseCommitPointTests(unittest.TestCase):
    """2026-09-29 audit: cancellation was wrong at both ends. A cancellation during image description was
    swallowed by the except Exception around the callback, so every image of the document still went to the
    model; conversely, after points had been written to the vector store the job still honoured a cancellation
    without bringing the state database up to date, and a re-parse of the same version left the two out of step
    from then on. Now a cancellation inside the callback stops the images not yet started, and once the vector
    write has begun the job counts as committed and completes the state database."""

    def _job(self, h):
        return h.con.execute("SELECT status, retry_count, cancel_requested FROM jobs WHERE job_id = ?", (h.job_id,)).fetchone()

    def _indexed(self, h) -> tuple[int, bool]:
        active = h.con.execute("SELECT COUNT(*) FROM chunks WHERE file_id = ? AND status = 'active'", (h.file_id,)).fetchone()[0]
        row = db.get_file_by_id(h.con, h.file_id)
        return int(active), str(row["indexed_version"] or "") == str(row["content_version"])

    def test_cancel_inside_the_caption_callback_stops_the_remaining_images(self) -> None:
        from unittest import mock

        from kb_pipeline.pipeline.parse_job import JobCancelled, vlm_progress_callback
        from kb_pipeline.vision import vlm

        captioned: list[str] = []

        def slow_caption(*, image_path, **kwargs):
            captioned.append(image_path.name)
            time.sleep(0.02)
            return dict(vlm.EMPTY_RESULT)

        def cancelled_stage(text: str, *, force: bool = False) -> None:
            raise JobCancelled("cancelled by operator")

        jobs = [(f"b{i}", Path(f"/x/{i}.png"), None, None) for i in range(30)]
        with mock.patch.object(vlm, "caption_image", slow_caption):
            with self.assertRaises(JobCancelled):
                vlm.caption_images_parallel(jobs, base_url="http://v", api_key="k", model_id="m", concurrency=2,
                                            progress_cb=vlm_progress_callback(cancelled_stage))
            self.assertLessEqual(len(captioned), 6)            # the running ones finish, the queued ones are not sent
            settled = len(captioned)
            time.sleep(0.1)
            self.assertEqual(len(captioned), settled)          # no thread goes on describing images after the return
            # an error raised by the progress report itself still does not affect the descriptions
            captioned.clear()
            results = vlm.caption_images_parallel(jobs[:4], base_url="http://v", api_key="k", model_id="m",
                                                  progress_cb=lambda done, total: 1 / 0)
        self.assertEqual((len(results), len(captioned)), (4, 4))

    def test_cancel_before_the_vector_write_stops_the_job(self) -> None:
        from kb_pipeline.pipeline import worker as worker_mod

        with tempfile.TemporaryDirectory() as tmp, _parse_harness(tmp) as h:
            h.hooks["parse"] = lambda: db.request_job_cancel(h.con, h.job_id, "cancelled from the console")
            result = worker_mod.run_once(h.con, settings=h.settings)
            self.assertTrue(result.startswith("cancelled:"), result)
            self.assertNotIn("upsert", h.calls)
            self.assertEqual(self._job(h)["status"], "cancelled")
            self.assertEqual(self._indexed(h), (0, False))

    def test_cancel_after_the_vector_write_started_lets_the_job_finish(self) -> None:
        from kb_pipeline.pipeline import worker as worker_mod

        with tempfile.TemporaryDirectory() as tmp, _parse_harness(tmp) as h:
            h.hooks["upsert"] = lambda: db.request_job_cancel(h.con, h.job_id, "cancelled from the console")
            result = worker_mod.run_once(h.con, settings=h.settings)
            self.assertTrue(result.startswith("parse-done:"), result)
            self.assertEqual(h.calls[-4:], ["upsert", "mark_old", "mark_stale", "fts"])
            self.assertEqual(self._job(h)["status"], "done")
            self.assertEqual(self._indexed(h), (1, True))      # the state database caught up with the vector store

    def test_failure_after_the_vector_write_is_retried_even_if_cancel_was_requested(self) -> None:
        from kb_pipeline.pipeline import worker as worker_mod

        def cancel_then_fail() -> None:
            db.request_job_cancel(h.con, h.job_id, "cancelled from the console")
            raise RuntimeError("qdrant went away")

        with tempfile.TemporaryDirectory() as tmp, _parse_harness(tmp) as h:
            h.hooks["mark_old"] = cancel_then_fail
            result = worker_mod.run_once(h.con, settings=h.settings)
            self.assertTrue(result.startswith("retry:"), result)
            job = self._job(h)
            self.assertEqual((job["status"], int(job["retry_count"]), int(job["cancel_requested"] or 0)), ("retry", 1, 0))
            # the retry is no longer stopped by the old cancellation and finishes the indexing
            del h.hooks["mark_old"]
            h.con.execute("UPDATE jobs SET next_attempt_at = 0 WHERE job_id = ?", (h.job_id,))
            h.con.commit()
            result = worker_mod.run_once(h.con, settings=h.settings)
            self.assertTrue(result.startswith("parse-done:"), result)
            self.assertEqual(self._indexed(h), (1, True))
            # a failure before the write: the cancellation still wins over the retry
        with tempfile.TemporaryDirectory() as tmp, _parse_harness(tmp) as h:
            h.hooks["embed"] = cancel_then_fail
            result = worker_mod.run_once(h.con, settings=h.settings)
            self.assertTrue(result.startswith("cancelled:"), result)


class VisualEmbedRejectionTests(unittest.TestCase):
    """When the visual embedding service rejects a picture for its content, the document is indexed as usual and
    that picture gets no visual vector -- a settled trade-off. But the "rejected" mark used to live only in the
    memory of the parse process: not in the payload, not on the timeline, so nobody could tell later that the
    picture lacks a vector."""

    def test_a_rejected_picture_is_marked_in_the_payload_and_on_the_timeline(self) -> None:
        from unittest import mock

        from kb_pipeline.pipeline import parse_job as pj
        from kb_pipeline.pipeline import worker as worker_mod

        class RejectingClient:
            def __init__(self, **kwargs) -> None:
                pass

            def embed_images(self, jobs):
                return [None for _ in jobs]

        with tempfile.TemporaryDirectory() as tmp, _parse_harness(tmp) as h:
            picture = Path(tmp) / "figure.png"; picture.write_bytes(b"not really a picture")
            h.results["parse"] = [
                _block("img1", "VISUAL SUMMARY: 一张流程图", block_type="image", visual_ref=str(picture),
                       metadata={"vlm_status": "success", "visual_sha256": "ab" * 32}),
            ]
            with mock.patch.object(pj, "_embed_visual_chunks", _REAL_EMBED_VISUAL_CHUNKS), \
                    mock.patch.object(pj, "VisualEmbeddingClient", RejectingClient):
                result = worker_mod.run_once(h.con, settings=h.settings)
            self.assertTrue(result.startswith("parse-done:"), result)
            payloads = {p["block_id"]: p for p in h.args["upsert"][4]}
            self.assertEqual(payloads["img1"]["visual_embed_status"], "content_rejected")
            self.assertNotIn("visual_embedding_model", payloads["img1"])
            self.assertIn("The visual embedding service rejected 1 images; they have no visual vector",
                          [e["text"] for e in db.job_events(h.con, h.job_id)])


class ReparseWhileRunningTests(unittest.TestCase):
    """2026-09-29 audit: "Re-parse all" skipped files that already had a job with the same key. A queued one
    parses with the new configuration, which is fine; a running one uses the configuration from when it started,
    and once it is done neither the content version nor the parser profile has changed, the scan judges the file
    unchanged, and the result of the old configuration stays for good. Now the running job gets a note and the
    worker queues one more run once it is done."""

    def test_a_reparse_request_that_hits_a_running_job_is_replayed_after_it(self) -> None:
        from kb_pipeline import discovery
        from kb_pipeline.pipeline import worker as worker_mod
        from kb_pipeline.pipeline.scheduler import requeue_kb_files

        with tempfile.TemporaryDirectory() as tmp, _parse_harness(tmp) as h:
            # still queued, not started: nothing to replay, it reads the new configuration when it starts
            self.assertEqual(requeue_kb_files(h.con, ingest_run_id="web", kb_id=h.kb_id, reason="web console reparse"), 0)
            requeued: list[int] = []

            def console_changes_config_and_reparses() -> None:
                discovery.set_config(h.con, h.kb_id, {"max_tokens": 256, "overlap_tokens": 32})
                requeued.append(requeue_kb_files(h.con, ingest_run_id="web", kb_id=h.kb_id, reason="web console reparse"))
                h.con.commit()

            h.hooks["embed"] = console_changes_config_and_reparses
            result = worker_mod.run_once(h.con, settings=h.settings)
            self.assertTrue(result.startswith("parse-done:"), result)
            self.assertEqual(h.args["parse"][1].max_tokens, 400)          # this run still uses the configuration it started with
            self.assertEqual(requeued, [1])                               # the console counts it as requeued
            jobs = h.con.execute("SELECT job_id, status, priority, payload_json FROM jobs WHERE file_id = ? "
                                 "ORDER BY created_at, rowid", (h.file_id,)).fetchall()
            self.assertEqual([str(j["status"]) for j in jobs], ["done", "queued"])
            self.assertEqual(json.loads(jobs[1]["payload_json"])["reason"], "web console reparse")
            del h.hooks["embed"]
            result = worker_mod.run_once(h.con, settings=h.settings)
            self.assertTrue(result.startswith("parse-done:"), result)
            self.assertEqual(h.args["parse"][1].max_tokens, 256)          # the replayed run uses the new configuration
            self.assertEqual(worker_mod.run_once(h.con, settings=h.settings), "no-job")   # replayed once only


class ServiceNotReadyTests(unittest.TestCase):
    """2026-09-29 audit: after a boot the parsing, embedding and image-description services need 5-18 minutes
    to become ready while the worker starts claiming jobs after about two; not reaching them counted as an
    ordinary failure against the retries, and every restart cost each queued job one or two retries for nothing.
    Now connection-level errors put the job back in the queue without counting a retry; the deferrals per job
    are capped, so a document that brings a service down every time still comes to a stop."""

    @staticmethod
    def _mineru_down() -> Exception:
        import requests

        from kb_pipeline.parsers.mineru_pdf import MinerUServiceError

        try:
            try:
                raise requests.ConnectionError("('Connection aborted.', RemoteDisconnected('Remote end closed connection'))")
            except requests.ConnectionError as exc:
                raise MinerUServiceError(str(exc), status_code=None) from exc
        except MinerUServiceError as wrapped:
            return wrapped

    def test_only_connection_level_failures_count_as_unreachable(self) -> None:
        import http.client

        import httpx
        import openai
        import requests
        from qdrant_client.http.exceptions import ResponseHandlingException

        from kb_pipeline.parsers.errors import service_unreachable
        from kb_pipeline.parsers.mineru_pdf import MinerUServiceError

        def chained(outer: Exception, inner: Exception, *, explicit: bool = True) -> Exception:
            try:
                try:
                    raise inner
                except type(inner) as exc:
                    if explicit:
                        raise outer from exc
                    raise outer
            except type(outer) as wrapped:
                return wrapped

        request = httpx.Request("POST", "http://127.0.0.1:8101/v1/embeddings")
        down = [
            ConnectionRefusedError(61, "Connection refused"),
            http.client.RemoteDisconnected("Remote end closed connection without response"),
            requests.ConnectionError("refused"),
            requests.ConnectTimeout("connect timed out"),
            self._mineru_down(),
            chained(openai.APIConnectionError(request=request), httpx.ConnectError("refused")),
            chained(openai.APIConnectionError(request=request), httpx.RemoteProtocolError("Server disconnected")),
            chained(ResponseHandlingException("boom"), httpx.ConnectError("refused"), explicit=False),
        ]
        for exc in down:
            self.assertTrue(service_unreachable(exc), repr(exc))
        response = requests.Response(); response.status_code = 503
        up = [
            RuntimeError("VLM description failed for 3/10 images"),
            requests.ReadTimeout("read timed out"),
            requests.HTTPError("503 Server Error", response=response),
            chained(openai.APITimeoutError(request=request), httpx.ReadTimeout("slow")),
            chained(MinerUServiceError("500", status_code=500), requests.HTTPError("500")),
            TimeoutError("timed out"),
        ]
        for exc in up:
            self.assertFalse(service_unreachable(exc), repr(exc))
        loop = RuntimeError("a"); loop.__cause__ = loop                 # a chain pointing at itself must not loop forever
        self.assertFalse(service_unreachable(loop))

    def test_unreachable_service_puts_the_job_back_without_counting_a_retry(self) -> None:
        from unittest import mock

        from kb_pipeline.pipeline import worker as worker_mod

        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"; db.init_db(state)
            settings = WorkerBranchTests._settings(self, state)          # job_max_retries=2
            with db.connect(state) as con:
                _, job_id = WorkerBranchTests._file_and_job(self, con)
                con.commit()
                waits = []
                with mock.patch.object(worker_mod, "process_parse_job", side_effect=self._mineru_down()):
                    for expected in ("deferred", "deferred", "retry", "retry", "failed"):
                        before = int(time.time())
                        result = worker_mod.run_once(con, settings=settings)
                        self.assertTrue(result.startswith(expected + ":"), f"{result} should start with {expected}")
                        row = con.execute("SELECT status, retry_count, next_attempt_at, locked_by, error FROM jobs "
                                          "WHERE job_id = ?", (job_id,)).fetchone()
                        waits.append((int(row["retry_count"]), int(row["next_attempt_at"] or 0) - before))
                        if expected == "deferred":
                            self.assertEqual((row["status"], row["locked_by"]), ("retry", None))
                            self.assertTrue(str(row["error"]).startswith("Service not ready"), row["error"])
                            # not claimable during the backoff; tried again once it is due
                            self.assertEqual(worker_mod.run_once(con, settings=settings), "no-job")
                        con.execute("UPDATE jobs SET next_attempt_at = 0 WHERE job_id = ?", (job_id,))
                        con.commit()
                failures = con.execute("SELECT COUNT(*) FROM failures WHERE job_id = ?", (job_id,)).fetchone()[0]
                kinds = [e["kind"] for e in db.job_events(con, job_id)]
            # The two deferrals count no retry and the wait doubles; once they are used up, retries are counted and
            # backed off as usual, ending in failed
            self.assertEqual([w[0] for w in waits], [0, 0, 1, 2, 2])
            self.assertTrue(300 <= waits[0][1] <= 302 and 600 <= waits[1][1] <= 602, waits)
            self.assertEqual(failures, 3)
            self.assertEqual(kinds, ["retry", "retry", "retry", "retry", "error"])

    def test_a_real_failure_is_still_counted(self) -> None:
        from unittest import mock

        from kb_pipeline.pipeline import worker as worker_mod

        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"; db.init_db(state)
            with db.connect(state) as con:
                _, job_id = WorkerBranchTests._file_and_job(self, con)
                con.commit()
                with mock.patch.object(worker_mod, "process_parse_job", side_effect=RuntimeError("MinerU 500")):
                    result = worker_mod.run_once(con, settings=WorkerBranchTests._settings(self, state))
                self.assertTrue(result.startswith("retry:"), result)
                self.assertEqual(con.execute("SELECT retry_count FROM jobs WHERE job_id = ?", (job_id,)).fetchone()[0], 1)

    def test_an_unreachable_model_stops_the_caption_run(self) -> None:
        from unittest import mock

        import httpx
        import openai

        from kb_pipeline.parsers.errors import service_unreachable
        from kb_pipeline.vision import vlm

        tried: list[str] = []

        def refused(*, image_path, **kwargs):
            tried.append(image_path.name)
            time.sleep(0.02)
            try:
                raise httpx.ConnectError("[Errno 111] Connection refused")
            except httpx.ConnectError as exc:
                raise openai.APIConnectionError(request=httpx.Request("POST", "http://127.0.0.1:8105/v1")) from exc

        jobs = [(f"b{i}", Path(f"/x/{i}.png"), None, None) for i in range(30)]
        with mock.patch.object(vlm, "caption_image", refused), redirect_stdout(io.StringIO()):
            with self.assertRaises(openai.APIConnectionError) as caught:
                vlm.caption_images_parallel(jobs, base_url="http://v", api_key="k", model_id="m", concurrency=2)
        self.assertTrue(service_unreachable(caught.exception))
        self.assertLessEqual(len(tried), 6)                         # the remaining images are no longer tried one by one

    def test_a_visual_embedding_transport_failure_skips_the_queued_images(self) -> None:
        from unittest import mock

        import requests

        from kb_pipeline.embedding.visual import VisualEmbeddingClient

        client = VisualEmbeddingClient(base_url="http://127.0.0.1:8103/v1", api_key="", model_id="m", dim=4, concurrency=2)
        tried: list[str] = []

        def refused(path, *, cache_json=None):
            tried.append(path.name)
            time.sleep(0.02)
            raise requests.ConnectionError("refused")

        with mock.patch.object(client, "embed_image", refused), redirect_stdout(io.StringIO()):
            with self.assertRaises(requests.ConnectionError):
                client.embed_images([(Path(f"/x/{i}.png"), None) for i in range(30)])
        self.assertLessEqual(len(tried), 6)


class WrapperScriptTests(unittest.TestCase):
    """The scan and worker wrapper scripts, really run with a fake python."""

    REPO = Path(__file__).resolve().parents[2]

    def _fake_python(self, tmp: Path, body: str) -> None:
        py = tmp / "app" / ".venv" / "bin" / "python"
        py.parent.mkdir(parents=True, exist_ok=True)
        py.write_text("#!/bin/bash\n" + body, encoding="utf-8")
        py.chmod(0o755)

    def _env(self, tmp: Path, **extra: str) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if not k.startswith("KB_")}
        env.update({
            "KB_LOCAL_BASE_DIR": str(tmp), "KB_ENV_FILE": str(tmp / "absent.env"), "TMPDIR": str(tmp),
            "KB_SCAN_REQUIRE_FLAG": "0",
            "KB_SCAN_KICK_WORKER": "0",          # a test must not kick the real worker of this machine
            "no_proxy": "127.0.0.1,localhost", "NO_PROXY": "127.0.0.1,localhost",
        })
        env.update(extra)
        return env

    @contextmanager
    def _qdrant_is_up(self):
        """The worker script probes Qdrant's /healthz with curl before it starts: serve a local port that only
        answers 200."""
        import http.server
        import threading

        class Healthy(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self.send_response(200); self.send_header("Content-Length", "0"); self.end_headers()

            def log_message(self, *args) -> None:
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Healthy)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}"
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=5)

    def _worker(self, tmp: Path, qdrant_url: str, **extra: str):
        import subprocess

        return subprocess.run(["bash", str(self.REPO / "scripts" / "kb-pipeline-worker-once.sh")],
                              env=self._env(tmp, QDRANT_URL=qdrant_url, **extra), capture_output=True, text=True, timeout=60)

    def test_a_failed_scan_reaches_the_failure_branch_and_keeps_its_exit_code(self) -> None:
        """A failed scan prints no summary line. Under set -euo pipefail the script used to die on the line that
        extracts jobs=: it exited with 1, and both the real exit code and the "KB scan failed" line were lost."""
        import subprocess

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._fake_python(tmp, 'echo "[scan] REFUSING KB lifecycle pass: mirror root missing" >&2\nexit 2\n')
            proc = subprocess.run(["bash", str(self.REPO / "scripts" / "kb-pipeline-scan.sh")],
                                  env=self._env(tmp), capture_output=True, text=True, timeout=60)
            self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
            self.assertIn("REFUSING KB lifecycle pass", proc.stdout)
            self.assertIn("KB scan failed (flag restored)", proc.stdout)
            state = tmp / "runtime" / "state"
            self.assertTrue((state / "mirror_changed.flag").exists())
            self.assertFalse((state / "kb_scan.lock.d").exists())
            self.assertEqual(list(tmp.glob("kb-scan.*")), [])
            # a normal round works as before
            self._fake_python(tmp, 'echo "scan summary: seen=3 added=0 unchanged=3 jobs=0 recent_pending=0 dry_run=False"\nexit 0\n')
            proc = subprocess.run(["bash", str(self.REPO / "scripts" / "kb-pipeline-scan.sh")],
                                  env=self._env(tmp), capture_output=True, text=True, timeout=60)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("KB scan end", proc.stdout)

    def test_the_worker_script_counts_jobs_not_batches(self) -> None:
        """processed used to grow by 1 per python batch (up to 10 jobs each), and the batch that ended in no-job
        did not count at all: the processed figure in the log did not match the jobs actually done, and "at most
        200 jobs per run" really meant two thousand."""
        batch = ('n=0\nwhile [[ $# -gt 0 ]]; do [[ "$1" == "--max-jobs" ]] && n="$2"; shift; done\n'
                 'echo "$n" >> "$KB_LOCAL_BASE_DIR/batches.log"\n')
        with tempfile.TemporaryDirectory() as tmpdir, self._qdrant_is_up() as qdrant:
            tmp = Path(tmpdir)
            self._fake_python(tmp, batch + 'for i in $(seq 1 "$n"); do echo "[parse] start job=job_$i"; '
                                           'echo "parse-done:job_$i:chunks=1"; done\nexit 0\n')
            proc = self._worker(tmp, qdrant, KB_WORKER_MAX_JOBS_PER_RUN="25", KB_WORKER_JOBS_PER_PROCESS="10")
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("KB worker reached per-run cap processed=25", proc.stdout)
            self.assertEqual((tmp / "batches.log").read_text(encoding="utf-8").split(), ["10", "10", "5"])
            # the queue ran empty after two jobs: those two count as well
            self._fake_python(tmp, 'echo "retry:job_1:delay=300s:RuntimeError()"\necho "deferred:job_2:delay=300s:x"\n'
                                   'echo "no-job"\nexit 0\n')
            proc = self._worker(tmp, qdrant)
            self.assertIn("KB worker loop idle processed=2", proc.stdout)
            # the interpreter crashed midway: the jobs already done are still counted
            self._fake_python(tmp, 'echo "parse-done:job_1:chunks=1"\nexit 137\n')
            proc = self._worker(tmp, qdrant)
            self.assertEqual(proc.returncode, 137, proc.stdout + proc.stderr)
            self.assertIn("KB worker giving up after 3 crashed iterations", proc.stdout)
            self.assertIn("crashes=3 processed=3", proc.stdout)
            self.assertFalse((tmp / "runtime" / "state" / "kb_worker.lock.d").exists())
            self.assertEqual(list(tmp.glob("kb-worker.*")), [])

    def test_the_worker_script_prints_while_the_job_is_still_running(self) -> None:
        """Output used to be captured per batch and printed only when the process ended: a parse running for an
        hour or two left nothing in the log meanwhile."""
        import subprocess

        with tempfile.TemporaryDirectory() as tmpdir, self._qdrant_is_up() as qdrant:
            tmp = Path(tmpdir)
            # after the first line, wait until the test lets it go on (at most 10 seconds); the test only does so
            # once it has seen that line
            self._fake_python(tmp, 'echo "parse-done:job_1:chunks=1"\n'
                                   'for i in $(seq 1 100); do [[ -f "$KB_LOCAL_BASE_DIR/go" ]] && break; sleep 0.1; done\n'
                                   '[[ -f "$KB_LOCAL_BASE_DIR/go" ]] && echo "released-by-the-reader" || echo "gave-up-waiting"\n'
                                   'echo "no-job"\nexit 0\n')
            proc = subprocess.Popen(["bash", str(self.REPO / "scripts" / "kb-pipeline-worker-once.sh")],
                                    env=self._env(tmp, QDRANT_URL=qdrant), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                seen = []
                for line in proc.stdout:
                    seen.append(line.strip())
                    if line.startswith("parse-done:job_1"):
                        (tmp / "go").write_text("", encoding="utf-8")
                proc.wait(timeout=30)
            finally:
                proc.kill() if proc.poll() is None else None
                proc.stdout.close()
            self.assertIn("released-by-the-reader", seen)
            self.assertTrue(seen[-1].endswith("KB worker loop end processed=1 ==="), seen)


class WorkerTimeBudgetTests(unittest.TestCase):
    """The worker's per-round time limit must take effect **between two files**.

    2026-08-25: systemd's TimeoutStartSec=7200 killed the worker from outside, right in the middle of one file's
    MinerU parse. The parse cache is only written once the whole document is done — a mid-way kill leaves just an
    empty directory (observed: only 7 of 9 mineru directories had a result.json), and that round's 12 minutes of
    compute were wasted.

    That round parsed 7 files in a row over 108 minutes; the 8th, the one cut off, was only 4.3MB, while a 67.6MB
    file in the same round completed. So the problem has nothing to do with single-file size: **whenever a round's
    total duration exceeds the hard limit, some file is necessarily cut off midway**; which one depends purely on
    queue order.
    """

    def test_budget_is_checked_after_a_job_not_before(self) -> None:
        """The checkpoint comes after run_once — placed before, it would become "no budget, no start" and the last
        slot would always be wasted; more importantly it must sit on a job boundary and never interrupt the job
        itself."""
        source = _repo_file("app/kb_pipeline/cli.py")
        body = source.split("budget = max(1, int(getattr(args", 1)[1].split("return 0", 1)[0]
        run = body.index("result = run_once(")
        check = body.index("time.monotonic() >= deadline")
        self.assertGreater(check, run, "预算检查必须在任务跑完之后")
        self.assertIn("time-budget-reached", body)

    def test_exhausting_the_budget_is_not_a_failure(self) -> None:
        """Exhausting the budget is a normal end of shift and must not take the failure path — otherwise every round
        leaves a bogus failed in the queue and burns that file's retry count."""
        source = _repo_file("app/kb_pipeline/cli.py")
        body = source.split("budget = max(1, int(getattr(args", 1)[1].split("return 0", 1)[0]
        tail = body[body.index("time-budget-reached"):]
        for bad in ("raise", "SystemExit", "return 1"):
            self.assertNotIn(bad, tail, f"预算用完不该 {bad}")

    def test_the_shell_shares_one_budget_across_inner_invocations(self) -> None:
        """The inner python does at most 10 jobs per call and the outer bash loops once per 10. The budget must be
        the remainder **shared across the whole round**; handing each call its own 90 minutes is no limit at all."""
        sh = _repo_file("scripts/kb-pipeline-worker-once.sh")
        self.assertIn("START_TS=", sh)
        self.assertIn("remaining=$(( MAX_SECONDS - ( $(date +%s) - START_TS ) ))", sh)
        self.assertIn('--max-seconds "$remaining"', sh)
        # When python reports the budget exhausted, the outer layer must stop too instead of idling on to MAX_JOBS_PER_RUN
        self.assertIn("time-budget-reached", sh)

    def test_hard_timeout_leaves_room_for_one_more_long_file(self) -> None:
        """The soft limit can only take effect after a job ends, so the hard limit must leave enough headroom to
        "finish one more longest file". Originally there was no soft limit and the hard limit was 2 hours; with the
        two pressed together a file gets cut off midway."""
        unit = _repo_file("deployment/systemd/carrel-worker.service")
        hard = int(re.search(r"TimeoutStartSec=(\d+)", unit).group(1))
        sh = _repo_file("scripts/kb-pipeline-worker-once.sh")
        soft = int(re.search(r'KB_WORKER_MAX_SECONDS:-(\d+)', sh).group(1))
        self.assertGreater(soft, 0)
        self.assertGreaterEqual(hard - soft, 3600,
                                "硬限与软限之间至少留 1 小时,够跑完一个大文件")


class JobTimelineTests(unittest.TestCase):
    """Timeline and failure panel for parse jobs: stage changes / milestones / errors go into job_events and the
    console dialog draws the timeline from them; failed and backing-off jobs have a list, a queue depth, and can be
    cancelled or retried."""

    def _job(self, state: Path) -> str:
        with db.connect(state) as con:
            con.execute(
                "INSERT INTO jobs(job_id, kb_id, collection, job_type, status, created_at, updated_at, retry_count) "
                "VALUES('j1', 'kb_1', 'kb_1', 'parse', 'running', 0, 0, 0)")
            con.commit()
        return "j1"

    def test_stage_events_dedupe_on_the_stage_head(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"; db.init_db(state)
            jid = self._job(state)
            with db.connect(state) as con:
                db.set_job_stage(con, jid, "文档解析")
                for i in range(1, 6):
                    db.set_job_stage(con, jid, f"图片描述(VLM {i}/5)")   # progress refreshes of the same stage are not recorded again
                db.set_job_stage(con, jid, "切块")
                db.add_job_event(con, jid, "info", "切块 52 片 · 验收通过")
                db.mark_job_done(con, jid)
                events = db.job_events(con, jid)
        self.assertEqual([(e["kind"], e["text"].split("(")[0]) for e in events],
                         [("stage", "文档解析"), ("stage", "图片描述"), ("stage", "切块"),
                          ("info", "切块 52 片 · 验收通过"), ("done", "Done")])
        # A progress refresh of the same stage rewrites that row's text: the timeline shows the latest progress, 5/5 at
        # the end rather than the 1/5 first written (2026-09-10)
        self.assertEqual(events[1]["text"], "图片描述(VLM 5/5)")

    def test_vlm_progress_final_tick_bypasses_the_stage_throttle(self) -> None:
        from kb_pipeline.pipeline.parse_job import vlm_progress_callback

        seen: list[tuple[str, bool]] = []

        def stage(text: str, *, force: bool = False) -> None:
            seen.append((text, force))

        cb = vlm_progress_callback(stage)
        cb(1, 3); cb(2, 3); cb(3, 3)
        self.assertEqual(seen, [("Describing images (VLM 1/3)", False), ("Describing images (VLM 2/3)", False), ("Describing images (VLM 3/3)", True)])
        plain: list[str] = []
        cb2 = vlm_progress_callback(lambda text: plain.append(text))      # a caller that does not accept force works as before
        cb2(2, 2)
        self.assertEqual(plain, ["Describing images (VLM 2/2)"])
        self.assertIsNone(vlm_progress_callback(None))

    def test_failure_retry_and_cancel_leave_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"; db.init_db(state)
            jid = self._job(state)
            with db.connect(state) as con:
                db.mark_job_failed(con, jid, "RuntimeError('VLM 挂了')", retry=True, retry_delay_seconds=30)
                self.assertEqual(db.request_job_cancel(con, jid, "控制台取消"), "cancelled")
                kinds = [e["kind"] for e in db.job_events(con, jid)]
                self.assertEqual(kinds, ["retry", "cancelled"])
                self.assertEqual(db.request_job_cancel(con, jid, "再来"), "noop")
                con.execute("UPDATE jobs SET status='running', cancel_requested=NULL WHERE job_id=?", (jid,))
                self.assertEqual(db.request_job_cancel(con, jid, "控制台取消"), "signalled")
                self.assertTrue(db.job_cancel_requested(con, jid))
                with self.assertRaises(KeyError):
                    db.request_job_cancel(con, "nope", "x")
                db.mark_job_failed(con, jid, "boom", retry=False)
                self.assertEqual(db.job_events(con, jid)[-1]["kind"], "error")

    def test_prune_drops_orphan_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"; db.init_db(state)
            jid = self._job(state)
            with db.connect(state) as con:
                db.set_job_stage(con, jid, "文档解析")
                con.execute("UPDATE jobs SET status='done', finished_at=1 WHERE job_id=?", (jid,))
                con.commit()
                removed = db.prune_job_history(con, retention_days=1)
                self.assertEqual((removed["jobs"], removed["job_events"]), (1, 1))
                self.assertEqual(db.job_events(con, jid), [])

    def test_service_detail_and_routes(self) -> None:
        from unittest import mock

        from fastapi.testclient import TestClient

        from kb_server import service
        from kb_server.main import create_app

        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"; db.init_db(state)
            jid = self._job(state)
            with db.connect(state) as con:
                db.set_job_stage(con, jid, "文档解析")
                db.mark_job_failed(con, jid, "MinerU 500", retry=False)
                con.execute(
                    "INSERT INTO jobs(job_id, kb_id, collection, job_type, status, created_at, updated_at, "
                    "retry_count, next_attempt_at) VALUES('j2', 'kb_1', 'kb_1', 'parse', 'retry', 0, 0, 2, ?)",
                    (int(time.time()) + 90,))
                con.execute(
                    "INSERT INTO jobs(job_id, kb_id, collection, job_type, status, created_at, updated_at) "
                    "VALUES('j3', 'kb_1', 'kb_1', 'parse', 'queued', 0, 0)")
                con.commit()
            with mock.patch.object(service, "settings", lambda: SimpleNamespace(state_db=state)):
                detail = service.job_detail(jid)
                self.assertEqual(detail["job"]["status"], "failed")
                self.assertEqual([e["kind"] for e in detail["events"]], ["stage", "error"])
                self.assertEqual(service.cancel_job("j3")["outcome"], "cancelled")
                with self.assertRaises(KeyError):
                    service.job_detail("nope")
                client = TestClient(create_app(), raise_server_exceptions=False)
                self.assertEqual(client.get("/api/jobs/nope").status_code, 404)
                self.assertEqual(client.post("/api/jobs/j2/cancel").json()["outcome"], "cancelled")

    def test_console_shows_the_job_timeline_in_the_drawer(self) -> None:
        """The jobs tab was withdrawn (2026-09-06): the timeline opens in the right-hand drawer by clicking a filename
        in the file table, cancel / retry stay in the drawer, and the drawer's timeline refreshes with the polling."""
        html = _repo_file("app/kb_server/static/index.html")
        js = _repo_file("app/kb_server/static/app.js")
        self.assertNotIn('id="tab-jobs"', html)
        for needle in ("/jobs/${encodeURIComponent(jobId)}", "function openJob(", 'openSide("job"', "synthesized",
                       'state.side.kind === "job") refreshJobDetail()'):
            self.assertIn(needle, js, needle)
        self.assertNotIn("renderJobsList(", js)
        parse = _repo_file("app/kb_pipeline/pipeline/parse_job.py")
        self.assertIn('db.add_job_event(con, str(job["job_id"]), "info", text)', parse)

    def test_job_detail_synthesizes_a_timeline_for_jobs_without_events(self) -> None:
        """The timeline table was added later: earlier jobs have no event rows, and the detail must not be blank.
        A summary is synthesised from the job's own timestamps and marked synthesized — the UI draws it dashed
        rather than passing it off as a record."""
        from unittest import mock

        from kb_server import service

        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"; db.init_db(state)
            with db.connect(state) as con:
                con.execute(
                    "INSERT INTO jobs(job_id, kb_id, collection, job_type, status, stage, created_at, started_at, "
                    "finished_at, updated_at, error) VALUES('old', 'kb_1', 'kb_1', 'parse', 'failed', '切块', "
                    "100, 110, 150, 150, 'boom')")
                con.commit()
            with mock.patch.object(service, "settings", lambda: SimpleNamespace(state_db=state)):
                detail = service.job_detail("old")
        self.assertTrue(detail["synthesized"])
        self.assertEqual([(e["ts"], e["kind"]) for e in detail["events"]],
                         [(100, "info"), (110, "stage"), (150, "error")])
        self.assertTrue(all(e["synthesized"] for e in detail["events"]))
        self.assertEqual(detail["events"][1]["text"], "切块")

    def test_kb_files_carries_chunk_counts_profile_and_index_time(self) -> None:
        """The file table must sort by chunk count and index time and show which parser version indexed the file —
        previously only visible by opening the database."""
        from unittest import mock

        from kb_server import service

        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"; db.init_db(state)
            with db.connect(state) as con:
                for fid, ver, idx in (("fa", "v1", "v1"), ("fb", "v2", None)):
                    con.execute(
                        "INSERT INTO files(file_id, kb_id, collection, source_root, source_type, file_key, "
                        "source_path, rel_path, filename, dir, physical_path, mime_type, size, mtime, "
                        "content_version, metadata_fingerprint, first_seen_at, last_seen_at, indexed_version, "
                        "indexed_parser_profile, status) VALUES(?, 'kb_1', 'kb_1', 'r', 'local', ?, ?, ?, ?, '', "
                        "?, 'application/pdf', 10, 7, ?, 'fp', 1, 1, ?, ?, 'seen')",
                        (fid, hash(fid) & 0xFFFF, f"/r/{fid}.pdf", f"{fid}.pdf", f"{fid}.pdf", f"/r/{fid}.pdf",
                         ver, idx, "pdf v6" if idx else None))
                for i in range(3):
                    con.execute(
                        "INSERT INTO chunks(chunk_uid, file_id, content_version, chunk_index, point_id, collection, "
                        "status, created_at) VALUES(?, 'fa', 'v1', ?, ?, 'kb_1', ?, 1)",
                        (f"c{i}", i, f"p{i}", "active" if i < 2 else "deleted"))
                con.execute(
                    "INSERT INTO jobs(job_id, kb_id, collection, file_id, job_type, status, created_at, updated_at, "
                    "finished_at) VALUES('j1', 'kb_1', 'kb_1', 'fa', 'parse', 'done', 0, 0, 1234)")
                con.commit()
            with mock.patch.object(service, "settings", lambda: SimpleNamespace(state_db=state)):
                rows = {r["file_id"]: r for r in service.kb_files("kb_1")}
        self.assertEqual((rows["fa"]["dot"], rows["fa"]["chunks"], rows["fa"]["indexed_profile"],
                          rows["fa"]["indexed_at"], rows["fa"]["mtime"]), ("green", 2, "pdf v6", 1234, 7))
        # A file not yet indexed reports no chunk count or index time, so the previous version's numbers are not
        # mistaken for this one's
        self.assertEqual((rows["fb"]["dot"], rows["fb"]["chunks"], rows["fb"]["indexed_at"]), ("yellow", None, None))


class WorkerFixRegressionTests(_CodexAudit20260906TestsSupport, unittest.TestCase):
    """Regressions for problems found by past re-reviews, health checks and audits; each case's docstring names the
    source and the symptom at the time."""

    def test_parser_change_of_a_terminally_failed_file_stays_blocked(self) -> None:  # issue 5
        from kb_pipeline.pipeline.scheduler import schedule_file

        file = _local_file("dir/报告.pdf", checksum="c1")
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"; db.init_db(state)
            with db.connect(state) as con:
                change, job_id = schedule_file(con, ingest_run_id="r1", file=file,
                                               current_seen_file_keys=set(), current_checksum_counts={})
                self.assertEqual(change, "new"); self.assertIsNotNone(job_id)
                con.execute("UPDATE jobs SET status='failed', error='MinerU exploded' WHERE job_id = ?", (job_id,))
                con.execute("UPDATE files SET indexed_version = content_version, indexed_parser_profile = 'pdf-old-profile' WHERE kb_id = ?", (file.kb_id,))
                # same content, changed profile -> parser_changed; the failed job
                # under the identical dedupe key must block the re-enqueue
                change2, job2 = schedule_file(con, ingest_run_id="r2", file=file,
                                              current_seen_file_keys=set(), current_checksum_counts={})
                self.assertEqual(change2, "parse_failed"); self.assertIsNone(job2)
                # --requeue-failed still overrides
                change3, job3 = schedule_file(con, ingest_run_id="r3", file=file,
                                              current_seen_file_keys=set(), current_checksum_counts={},
                                              requeue_failed=True)
                self.assertEqual(change3, "parser_changed"); self.assertIsNotNone(job3)

    def test_restored_file_gets_one_fresh_parse_attempt(self) -> None:  # issue 5 relaxation
        from kb_pipeline.pipeline.scheduler import schedule_file

        file = _local_file("dir/恢复的文件.pdf", checksum="c1")
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"; db.init_db(state)
            with db.connect(state) as con:
                change, job_id = schedule_file(con, ingest_run_id="r1", file=file,
                                               current_seen_file_keys=set(), current_checksum_counts={})
                self.assertEqual(change, "new")
                con.execute("UPDATE jobs SET status='failed', error='embedding endpoint down' WHERE job_id = ?", (job_id,))
                # steady state: the failed combination is blocked
                blocked, none_job = schedule_file(con, ingest_run_id="r2", file=file,
                                                  current_seen_file_keys=set(), current_checksum_counts={})
                self.assertEqual(blocked, "parse_failed"); self.assertIsNone(none_job)
                # the user deletes and restores the file: that event grants a retry
                file_id = db.file_id_for(file.kb_id, file.file_key)
                db.mark_file_deleted(con, file_id)
                restored, retry_job = schedule_file(con, ingest_run_id="r3", file=file,
                                                    current_seen_file_keys=set(), current_checksum_counts={})
                self.assertEqual(restored, "new")          # "file restored after delete"
                self.assertIsNotNone(retry_job)            # fresh attempt enqueued
                # the retry fails as well -> next scans are blocked again, no loop
                con.execute("UPDATE jobs SET status='failed' WHERE job_id = ?", (retry_job,))
                again, none_job2 = schedule_file(con, ingest_run_id="r4", file=file,
                                                 current_seen_file_keys=set(), current_checksum_counts={})
                self.assertEqual(again, "parse_failed"); self.assertIsNone(none_job2)

    def test_failed_lifecycle_job_is_requeued_by_the_next_scan(self) -> None:  # issue 5
        from kb_pipeline.pipeline.scheduler import requeue_failed_lifecycle_jobs, schedule_file

        file = _local_file("dir/说明.pdf", checksum="c1")
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"; db.init_db(state)
            with db.connect(state) as con:
                schedule_file(con, ingest_run_id="r1", file=file,
                              current_seen_file_keys=set(), current_checksum_counts={})
                file_id = db.file_id_for(file.kb_id, file.file_key)
                key = f"metadata_update:{file_id}:{db.metadata_fingerprint(file)}"
                db.enqueue_job(con, ingest_run_id="r1", file_id=file_id, kb_id=file.kb_id,
                               collection=file.collection, file_key=file.file_key,
                               job_type="metadata_update", priority=10,
                               payload={"reactivate": True}, dedupe_key=key)
                con.execute("UPDATE jobs SET status='failed' WHERE job_type='metadata_update'")
                self.assertEqual(requeue_failed_lifecycle_jobs(con, ingest_run_id="r2"), 1)
                row = con.execute(
                    "SELECT payload_json FROM jobs WHERE job_type='metadata_update' AND status='queued'"
                ).fetchone()
                self.assertIsNotNone(row)
                requeued_payload = json.loads(row["payload_json"])
                self.assertTrue(requeued_payload.get("reactivate"))  # the flag must survive the requeue
                self.assertIn("requeued_from", requeued_payload)
                # a queued attempt now exists under the key -> no double requeue
                self.assertEqual(requeue_failed_lifecycle_jobs(con, ingest_run_id="r3"), 0)
                # once the newest attempt succeeded, nothing is requeued either
                con.execute("UPDATE jobs SET status='done' WHERE job_type='metadata_update' AND status='queued'")
                self.assertEqual(requeue_failed_lifecycle_jobs(con, ingest_run_id="r4"), 0)

    def test_purge_kb_state_uses_subqueries(self) -> None:  # issue 15
        files = [_local_file(f"dir/f{i}.pdf", checksum=f"c{i}") for i in range(30)]
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"; db.init_db(state)
            with db.connect(state) as con:
                from kb_pipeline.pipeline.scheduler import schedule_file

                for f in files:
                    schedule_file(con, ingest_run_id="r1", file=f,
                                  current_seen_file_keys=set(), current_checksum_counts={})
                counts = db.kb_state_counts(con, files[0].kb_id)
                self.assertEqual(counts["files"], 30)
                self.assertEqual(counts["jobs"], 30)
                purged = db.purge_kb_state(con, files[0].kb_id)
                self.assertEqual(purged["files"], 30)
                after = db.kb_state_counts(con, files[0].kb_id)
                self.assertEqual(after, {"files": 0, "chunks": 0, "jobs": 0, "failures": 0})

    def test_scanner_tolerates_files_vanishing_mid_walk(self) -> None:  # scan robustness
        from kb_pipeline.localfs.scanner import list_source_files

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "库"; root.mkdir()
            keep = root / "keep.txt"; keep.write_text("ok", encoding="utf-8")
            ghost = root / "ghost.txt"; ghost.write_text("gone", encoding="utf-8")
            source = KBSource(kb_id="k", collection="kb_k", source_root="库", source_type="local_mirror",
                              max_tokens=400, overlap_tokens=80, physical_base=root)
            real_stat = Path.stat

            def racy_stat(self, **kwargs):
                if self.name == "ghost.txt":
                    # errno matters: Path.is_file() only swallows OSErrors
                    # carrying ENOENT-class errnos, like a real os.stat does
                    raise FileNotFoundError(errno.ENOENT, "vanished mid-walk", str(self))
                return real_stat(self, **kwargs)

            with patch.object(Path, "stat", racy_stat):
                files = list_source_files(source, min_age_seconds=0, hash_content=False)
            self.assertEqual([f.filename for f in files], ["keep.txt"])

    def test_recently_touched_file_is_deferred_not_deleted(self) -> None:
        """S1: a just-modified file is merely not parsed this round and must never be treated as deleted.
        The scanner reports its file_key into too_recent_keys and the caller counts it as seen."""
        from kb_pipeline.localfs.scanner import list_source_files

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "库"; root.mkdir()
            settled = root / "old.txt"; settled.write_text("old", encoding="utf-8")
            os.utime(settled, (time.time() - 600, time.time() - 600))
            fresh = root / "just-edited.txt"; fresh.write_text("new", encoding="utf-8")
            source = KBSource(kb_id="k", collection="kb_k", source_root="库", source_type="local_mirror",
                              max_tokens=400, overlap_tokens=80, physical_base=root)
            too_recent: set[int] = set()
            files = list_source_files(source, min_age_seconds=30, hash_content=False,
                                      too_recent_keys=too_recent)
            self.assertEqual([f.filename for f in files], ["old.txt"])   # the new file is not parsed this round
            self.assertEqual(len(too_recent), 1)                          # but it is reported as "present"
            # Old callers (not passing too_recent_keys) behave as before
            self.assertEqual(len(list_source_files(source, min_age_seconds=30, hash_content=False)), 1)

    def test_delete_detection_never_removes_a_file_still_on_disk(self) -> None:
        """S1's second line of defence: no delete is queued while physical_path is still on disk."""
        with tempfile.TemporaryDirectory() as tmp:
            present = Path(tmp) / "still-here.pdf"; present.write_text("x", encoding="utf-8")
            file = _local_file("dir/still-here.pdf", checksum="c1")
            file.physical_path = str(present)
            with db.connect(Path(tmp) / "s.db") as con:
                con.executescript(db.SCHEMA); db.migrate_schema(con)
                schedule_file(con, ingest_run_id="r1", file=file,
                              current_seen_file_keys=set(), current_checksum_counts={})
                # the file is absent from this round's seen set (simulating a scanner skip)
                deleted, jobs = schedule_deletes_for_source(
                    con, ingest_run_id="r2", kb_id=file.kb_id, collection=file.collection,
                    seen_file_keys=set(), verify_physical=True)
                self.assertEqual((deleted, jobs), (0, 0))     # still on disk → not deleted
                row = db.get_file_by_id(con, db.file_id_for(file.kb_id, file.file_key))
                self.assertNotEqual(str(row["status"]), "deleted")
                # Deletion proceeds only once the file is really gone
                present.unlink()
                deleted, jobs = schedule_deletes_for_source(
                    con, ingest_run_id="r3", kb_id=file.kb_id, collection=file.collection,
                    seen_file_keys=set(), verify_physical=True)
                self.assertEqual(deleted, 1)

    def test_job_stage_roundtrip(self) -> None:
        file = _local_file("dir/进度.pdf", checksum="c1")
        with tempfile.TemporaryDirectory() as tmp:
            with db.connect(Path(tmp) / "s.db") as con:
                con.executescript(db.SCHEMA)
                db.migrate_schema(con)
                from kb_pipeline.pipeline.scheduler import schedule_file

                _, job_id = schedule_file(con, ingest_run_id="r1", file=file,
                                          current_seen_file_keys=set(), current_checksum_counts={})
                db.set_job_stage(con, job_id, "图片描述(VLM 3/7)")
                row = con.execute("SELECT stage FROM jobs WHERE job_id=?", (job_id,)).fetchone()
                self.assertEqual(row["stage"], "图片描述(VLM 3/7)")

    def test_requeue_single_file_respects_active_jobs(self) -> None:
        from kb_pipeline.pipeline.scheduler import requeue_single_file, schedule_file

        file = _local_file("dir/重试.pdf", checksum="c1")
        with tempfile.TemporaryDirectory() as tmp:
            with db.connect(Path(tmp) / "s.db") as con:
                con.executescript(db.SCHEMA); db.migrate_schema(con)
                _, job_id = schedule_file(con, ingest_run_id="r1", file=file,
                                          current_seen_file_keys=set(), current_checksum_counts={})
                row = db.get_file_by_id(con, db.file_id_for(file.kb_id, file.file_key))
                # an attempt is already queued -> no duplicate
                self.assertIsNone(requeue_single_file(con, ingest_run_id="r2", file_row=row))
                con.execute("UPDATE jobs SET status='failed' WHERE job_id=?", (job_id,))
                new_job = requeue_single_file(con, ingest_run_id="r3", file_row=row)
                self.assertIsNotNone(new_job)   # manual retry goes through the same queue

    def test_expired_lease_running_job_does_not_block_maintenance(self) -> None:
        """C4: a running row left by a killed worker has an expired lease and is only taken over at the next claim;
        in the meantime (a parse lease can reach 18h) it must not block GC / deletion / rebuild."""
        from unittest import mock

        from kb_pipeline import maintenance

        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"; db.init_db(state)
            now = int(time.time())
            with db.connect(state) as con:
                con.execute(
                    "INSERT INTO jobs(job_id, kb_id, collection, job_type, status, priority, "
                    "locked_until, created_at, updated_at) VALUES('dead', 'kb', 'kb', 'parse', 'running', 100, ?, 1, 1)",
                    (now - 60,),   # lease expired
                )
                con.commit()
            stub = SimpleNamespace(state_db=state, runtime_dir=Path(tmp) / "rt", mineru_url="http://m")
            with mock.patch.object(maintenance, "_pgrep", return_value=False), \
                    mock.patch.object(maintenance, "_mineru_busy", return_value=None):
                busy, reasons = maintenance.service_busy(stub)
            self.assertFalse(busy, f"过期租约不该算忙: {reasons}")

            with db.connect(state) as con:      # with the lease still valid it still counts as busy
                con.execute("UPDATE jobs SET locked_until = ? WHERE job_id='dead'", (now + 3600,))
                con.commit()
            with mock.patch.object(maintenance, "_pgrep", return_value=False), \
                    mock.patch.object(maintenance, "_mineru_busy", return_value=None):
                busy, _ = maintenance.service_busy(stub)
            self.assertTrue(busy)

    def test_fts_degrade_branches_assign_fts_result(self) -> None:
        """B1 / B5: the keyword-index degradation branch must assign fts_result, and the empty-document path needs the
        same degradation."""
        src = _repo_file("app/kb_pipeline/pipeline/parse_job.py")
        main = src.split('stage("Keyword indexing")', 1)[1].split('stage("Done", force=True)', 1)[0]
        degraded = main.split("except Exception as exc:", 1)[1]
        self.assertIn('fts_result = {"inserted_rows": 0, "deferred": True}', degraded)
        empty = src.split("def _finish_empty_document", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("try:\n        fts_result = search_fts.sync_doc_from_qdrant(", empty)
        self.assertIn('fts_result = {"inserted_rows": 0, "deferred": True}', empty)
        self.assertIn("_defer_fts_sync(", empty)      # 09-06 evening: the compensation job became its own fts_sync type (F06)

    def test_unsupported_route_is_a_deterministic_failure(self) -> None:
        """B6: a format with no parse route raises NonRetryableParseError, and the worker no longer goes through 5
        backoff rounds."""
        from kb_pipeline.parsers.errors import NonRetryableParseError
        from kb_pipeline.parsers.router import parse_native
        from kb_pipeline.pipeline import parse_job
        from kb_pipeline.pipeline.worker import _should_retry

        self.assertIs(parse_job.NonRetryableParseError, NonRetryableParseError)
        with tempfile.TemporaryDirectory() as tmp:
            odd = Path(tmp) / "x.xyz"
            odd.write_text("?", encoding="utf-8")
            with self.assertRaises(NonRetryableParseError):
                parse_native(odd)
        self.assertFalse(_should_retry({"retry_count": 0}, SimpleNamespace(job_max_retries=5),
                                       NonRetryableParseError("route")))
        self.assertIn('NonRetryableParseError("legacy .doc files are not supported")',
                      _repo_file("app/kb_pipeline/pipeline/parse_job.py"))

    def test_scanner_accepts_every_code_suffix_the_parser_knows(self) -> None:
        """B8: the scanner's extension table ⊇ the code parser's language table."""
        from kb_pipeline.localfs.scanner import SUPPORTED_EXTS
        from kb_pipeline.parsers.code_symbols import LANGUAGE_BY_SUFFIX

        for ext in (".mjs", ".cjs", ".lua", ".cxx", ".hh"):
            self.assertIn(ext, SUPPORTED_EXTS, ext)
        self.assertEqual(set(LANGUAGE_BY_SUFFIX) - SUPPORTED_EXTS, set())

    def test_f07_nonretryable_failures_stay_final_until_the_parser_changes(self) -> None:
        """F07: deterministic failures (NonRetryableParseError) used to be re-enqueued by every scan; now they are
        blocked and only released when the parser / route changes; a file that reappears, and an ordinary failure
        past its cooldown, are released as before."""
        from kb_pipeline.pipeline import scheduler

        def job(error, finished=None):
            return {"error": error, "finished_at": finished if finished is not None else int(time.time())}

        self.assertTrue(scheduler._failed_parse_blocks_auto_requeue(job("NonRetryableParseError: 没有解析路由"), "needs_parse"))
        self.assertTrue(scheduler._failed_parse_blocks_auto_requeue(job("非重试:格式坏了"), "needs_parse"))
        self.assertFalse(scheduler._failed_parse_blocks_auto_requeue(job("NonRetryableParseError: 没有解析路由"), "parser_changed"))
        self.assertFalse(scheduler._failed_parse_blocks_auto_requeue(job("physical file not found"), "needs_parse"))
        self.assertTrue(scheduler._failed_parse_blocks_auto_requeue(job("MinerU timeout"), "needs_parse"))          # blocked within the cooldown
        with patch.object(scheduler, "FAILED_RETRY_COOLDOWN_SECONDS", 10):
            self.assertFalse(scheduler._failed_parse_blocks_auto_requeue(job("MinerU timeout", finished=int(time.time()) - 60), "needs_parse"))
        src = _repo_file("app/kb_pipeline/pipeline/scheduler.py")
        self.assertIn("_failed_parse_blocks_auto_requeue(failed_job, change.change_type)", src)

    def test_f06_fts_compensation_is_its_own_job_type_and_recovers(self) -> None:
        """F06: the keyword-index compensation job used to borrow the metadata_update type with a fts-retry:… key,
        the recovery logic only recognised metadata_update:… keys, so once retries were exhausted it was never
        queued again; its failure record hung under the main parse job and was marked resolved as soon as the main
        job finished. Now: a separate fts_sync job, the failure hangs under it, the scan re-queues by version, and
        the worker really rewrites the index."""
        from unittest import mock

        from kb_pipeline.pipeline import worker as worker_mod
        from kb_pipeline.pipeline.parse_job import _defer_fts_sync
        from kb_pipeline.pipeline.scheduler import requeue_failed_lifecycle_jobs

        settings = SimpleNamespace(
            runtime_dir=None, state_db=None, qdrant_url="http://q", qdrant_api_key="", opensearch_url="http://o",
            parse_enabled=True, parse_job_lease_seconds=600, metadata_job_lease_seconds=600,
            job_max_retries=2, job_retry_base_seconds=300, job_retry_max_seconds=3600, sources={})
        with tempfile.TemporaryDirectory() as tmp:
            settings.runtime_dir = Path(tmp) / "runtime"; settings.state_db = Path(tmp) / "state.db"
            db.init_db(settings.state_db)
            with db.connect(settings.state_db) as con:
                file = _local_file("x/a.pdf", checksum="c1")
                db.upsert_file(con, file, status="indexed")
                fid = db.file_id_for(file.kb_id, file.file_key)
                row = db.get_file_by_id(con, fid)
                version = str(row["content_version"])
                parse_job = {"job_id": "job-parse-1", "ingest_run_id": "r1"}
                # 1. OpenSearch is down during parsing: a compensation job is queued, and the failure record hangs
                #    under it, not under the main parse job
                fts_job = _defer_fts_sync(con, parse_job, row, collection=file.collection, content_version=version,
                                          exc=RuntimeError("opensearch down"))
                job_row = con.execute("SELECT * FROM jobs WHERE job_id=?", (fts_job,)).fetchone()
                self.assertEqual((job_row["job_type"], job_row["dedupe_key"]), ("fts_sync", db.fts_sync_dedupe_key(fid, version)))
                self.assertEqual(json.loads(job_row["payload_json"])["content_version"], version)
                fail = con.execute("SELECT job_id, stage, resolved_at FROM failures WHERE file_id=?", (fid,)).fetchone()
                self.assertEqual((fail["job_id"], fail["stage"], fail["resolved_at"]), (fts_job, "fts-sync", None))
                db.mark_job_done(con, "job-parse-1")           # main parse job done: the compensation's failure record must remain
                self.assertIsNone(con.execute("SELECT resolved_at FROM failures WHERE job_id=?", (fts_job,)).fetchone()[0])
                # 2. The worker runs the compensation: OpenSearch still down → backoff retry, the failure record still
                #    hangs under it
                with mock.patch.object(worker_mod, "qdrant_client"), \
                        mock.patch.object(worker_mod, "_sync_fts_doc", side_effect=RuntimeError("still down")):
                    result = worker_mod.run_once(con, settings=settings)
                self.assertTrue(result.startswith("retry:"), result)
                self.assertEqual(con.execute("SELECT status FROM jobs WHERE job_id=?", (fts_job,)).fetchone()[0], "retry")
                # 3. Retries exhausted into failed: the next scan re-queues it on "file still there, version unchanged";
                #    a changed version / deleted file is not re-queued
                con.execute("UPDATE jobs SET status='failed', next_attempt_at=0 WHERE job_id=?", (fts_job,))
                self.assertEqual(requeue_failed_lifecycle_jobs(con, ingest_run_id="r2"), 1)
                queued = con.execute("SELECT job_id, payload_json FROM jobs WHERE job_type='fts_sync' AND status='queued'").fetchone()
                self.assertIsNotNone(queued)
                self.assertEqual(json.loads(queued["payload_json"])["requeued_from"], fts_job)
                self.assertEqual(requeue_failed_lifecycle_jobs(con, ingest_run_id="r3"), 0)     # already queued, no duplicate
                # 4. OpenSearch is back: the worker rewrites the index from the active points, the job completes and
                #    the failure record is resolved
                with mock.patch.object(worker_mod, "qdrant_client"), \
                        mock.patch.object(worker_mod, "_sync_fts_doc") as synced:
                    result = worker_mod.run_once(con, settings=settings)
                self.assertTrue(result.startswith("fts-sync-done:"), result)
                synced.assert_called_once()
                self.assertEqual(con.execute("SELECT status FROM jobs WHERE job_id=?", (queued["job_id"],)).fetchone()[0], "done")
                # The earliest failure hangs under the first compensation job; the re-queued job's success must resolve it too
                unresolved = con.execute("SELECT COUNT(*) FROM failures WHERE file_id=? AND resolved_at IS NULL", (fid,)).fetchone()[0]
                self.assertEqual(unresolved, 0)
                # 5. An old compensation job whose version has changed: voided, the index is not touched
                stale = db.enqueue_job(con, ingest_run_id=None, file_id=fid, kb_id=file.kb_id, collection=file.collection,
                                       file_key=file.file_key, job_type="fts_sync", priority=60,
                                       payload={"content_version": "old-version"}, dedupe_key=db.fts_sync_dedupe_key(fid, "old-version"))
                with mock.patch.object(worker_mod, "qdrant_client"), \
                        mock.patch.object(worker_mod, "_sync_fts_doc") as synced:
                    result = worker_mod.run_once(con, settings=settings)
                self.assertTrue(result.startswith("fts-sync-done:"), result)
                synced.assert_not_called()
                con.execute("UPDATE jobs SET status='failed' WHERE job_id=?", (stale,))
                self.assertEqual(requeue_failed_lifecycle_jobs(con, ingest_run_id="r4"), 0)     # version mismatch: not re-queued
        worker_src = _repo_file("app/kb_pipeline/pipeline/worker.py")
        self.assertIn('allowed_job_types = {"metadata_update", "delete", "fts_sync"}', worker_src)
        self.assertNotIn('dedupe_key=f"fts-retry:', _repo_file("app/kb_pipeline/pipeline/parse_job.py"))

    def test_job_terminal_writes_require_ownership(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"; db.init_db(state)
            with db.connect(state) as con:
                file = _local_file("dir/a.pdf", checksum="c1")
                db.upsert_file(con, file, status="seen")
                fid = db.file_id_for(file.kb_id, file.file_key)
                job_id = db.enqueue_job(con, ingest_run_id=None, file_id=fid, kb_id=file.kb_id, collection=file.collection,
                                        file_key=file.file_key, job_type="parse", dedupe_key=f"parse:{fid}")
                con.commit()
                job = db.claim_job_for_types(con, "box:1", lease_seconds=60)
                self.assertEqual(job["job_id"], job_id)
                # Another worker cannot write a terminal state or record a failure; checkpoints honour the lock too
                db.mark_job_done(con, job_id, "box:2")
                self.assertFalse(db.mark_job_failed(con, job_id, "x", retry=True, worker_id="box:2"))
                self.assertEqual(tuple(con.execute("SELECT status, retry_count, locked_by FROM jobs WHERE job_id=?", (job_id,)).fetchone()),
                                 ("running", 0, "box:1"))
                self.assertTrue(db.job_cancel_requested(con, job_id, "box:2"))
                self.assertFalse(db.job_cancel_requested(con, job_id, "box:1"))
                # Crash recovery leaves the job alone when the lock has changed hands
                self.assertEqual(db.release_crashed_job(con, job_id, max_retries=2, retry_delay_seconds=1, expected_owner="box:2"), "skipped")
                self.assertEqual(con.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,)).fetchone()[0], "running")
                # The owner's own writes work as usual
                self.assertTrue(db.mark_job_failed(con, job_id, "x", retry=True, worker_id="box:1"))
                self.assertEqual(con.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,)).fetchone()[0], "retry")

    def test_worker_cancels_instead_of_retrying_when_the_flag_is_up(self) -> None:
        from unittest import mock

        from kb_pipeline.pipeline import worker as worker_mod

        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"; db.init_db(state)
            with db.connect(state) as con:
                file = _local_file("dir/a.pdf", checksum="c1")
                db.upsert_file(con, file, status="seen")
                fid = db.file_id_for(file.kb_id, file.file_key)
                job_id = db.enqueue_job(con, ingest_run_id=None, file_id=fid, kb_id=file.kb_id, collection=file.collection,
                                        file_key=file.file_key, job_type="parse", dedupe_key=f"parse:{fid}")
                con.commit()

                def closed_then_boom(con_, settings, job):
                    # KB closed: the cancel flag is up, the checkpoint has not come yet, and an external call raised an
                    # ordinary exception first
                    db.request_kb_job_cancel(con_, job["kb_id"], "closed")
                    raise RuntimeError("collection is gone")

                with mock.patch.object(worker_mod, "process_parse_job", side_effect=closed_then_boom):
                    result = worker_mod.run_once(con, settings=JobCancellationTests._settings(self, state))
                status = con.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,)).fetchone()[0]
                failures = con.execute("SELECT COUNT(*) FROM failures WHERE job_id=?", (job_id,)).fetchone()[0]
            self.assertTrue(result.startswith("cancelled:"), result)
            self.assertEqual((status, failures), ("cancelled", 0))      # not a failure: not re-queued, no fault recorded


class MirrorBoundaryTests(unittest.TestCase):
    """2026-09-28 security review F04: symlinks must not bring files from outside an enabled directory into the
    index. File-level / subdirectory-level symlinks are always judged by their real location; a top-level directory
    that is itself a symlink is not recognised by default, only with KB_MIRROR_ALLOW_LINKED_DIRS=1."""

    def test_symlinks_leaving_the_enrolled_directory_are_skipped(self) -> None:
        from kb_pipeline.localfs.scanner import inside_boundary, list_recent_source_files, list_source_files

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            outside = base / "outside"; outside.mkdir()
            (outside / "secret.md").write_text("outside", encoding="utf-8")
            (outside / "sub").mkdir(); (outside / "sub" / "deep.md").write_text("deep", encoding="utf-8")
            root = base / "mirror" / "kb"; root.mkdir(parents=True)
            (root / "own.md").write_text("mine", encoding="utf-8")
            (root / "link.md").symlink_to(outside / "secret.md")
            (root / "linkdir").symlink_to(outside / "sub", target_is_directory=True)
            (root / "inner").mkdir(); (root / "inner" / "same.md").write_text("mine too", encoding="utf-8")
            (root / "innerlink.md").symlink_to(root / "inner" / "same.md")      # a symlink pointing back inside the directory is fine
            source = KBSource(kb_id="k", collection="kb_k", source_root="kb", source_type="local_mirror",
                              max_tokens=400, overlap_tokens=80, physical_base=root)
            got = sorted(f.rel_path for f in list_source_files(source, hash_content=False))
            self.assertEqual(got, ["inner/same.md", "innerlink.md", "own.md"])
            recent = sorted(p.name for p in list_recent_source_files(source, min_age_seconds=3600))
            self.assertEqual(recent, ["own.md", "same.md", "innerlink.md"] if False else sorted(["own.md", "same.md", "innerlink.md"]))
            self.assertTrue(inside_boundary(root / "own.md", root.resolve()))
            self.assertFalse(inside_boundary(root / "link.md", root.resolve()))
            self.assertFalse(inside_boundary(root / "missing.md", root.resolve()))

    def test_an_enrolled_directory_swapped_for_a_link_is_not_read(self) -> None:
        """Second review R01 (a): the directory was real when it was enrolled; replacing it with a symbolic link
        must not make the link's target the new boundary. The row stays registered (nothing is deleted), the
        policy reports why, and with the switch on it is read again."""
        import shutil

        from kb_pipeline import discovery

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            mirror = base / "mirror"; (mirror / "kb").mkdir(parents=True)
            (mirror / "kb" / "own.md").write_text("mine", encoding="utf-8")
            outside = base / "outside"; outside.mkdir(); (outside / "synthetic.txt").write_text("outside", encoding="utf-8")
            state = base / "state.db"; db.init_db(state)
            with db.connect(state) as con:
                src, _ = discovery.enroll(con, mirror, "kb"); con.commit()
            with patch.dict(os.environ, {"KB_MIRROR_ALLOW_LINKED_DIRS": ""}):
                self.assertIn(src.kb_id, discovery.enrolled_sources(state, mirror))
                shutil.rmtree(mirror / "kb"); (mirror / "kb").symlink_to(outside, target_is_directory=True)
                self.assertEqual(discovery.enrolled_sources(state, mirror), {})
                self.assertEqual(discovery.directory_admitted(mirror, "kb"), (False, "linked"))
                self.assertEqual(discovery.discover_directories(mirror), [])
                with db.connect(state) as con:
                    self.assertEqual(str(discovery.known_sources(con)[0]["status"]), "active")   # registered, not deleted
            with patch.dict(os.environ, {"KB_MIRROR_ALLOW_LINKED_DIRS": "1"}):
                self.assertIn(src.kb_id, discovery.enrolled_sources(state, mirror))

    def test_the_worker_rechecks_the_file_before_reading_it(self) -> None:
        """Second review R01 (b): a file queued as a regular file and replaced by a link before the worker gets
        to it is refused at read time; the same check refuses a root that became a link, and it runs before
        the parser opens the file."""
        import shutil

        from kb_pipeline.parsers.errors import NonRetryableParseError
        from kb_pipeline.pipeline.parse_job import verify_source_file

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            mirror = base / "mirror"; (mirror / "kb").mkdir(parents=True)
            outside = base / "outside"; outside.mkdir(); (outside / "synthetic.txt").write_text("outside", encoding="utf-8")
            f = mirror / "kb" / "doc.md"; f.write_text("mine", encoding="utf-8")
            settings = SimpleNamespace(mirror_root=mirror)
            source = SimpleNamespace(source_root="kb")
            with patch.dict(os.environ, {"KB_MIRROR_ALLOW_LINKED_DIRS": ""}):
                verify_source_file(settings, source, f)                                     # a regular file inside: fine
                f.unlink(); f.symlink_to(outside / "synthetic.txt")
                with self.assertRaisesRegex(NonRetryableParseError, "outside the enrolled directory"):
                    verify_source_file(settings, source, f)
                shutil.rmtree(mirror / "kb"); (mirror / "kb").symlink_to(outside, target_is_directory=True)
                with self.assertRaisesRegex(NonRetryableParseError, "symbolic link"):
                    verify_source_file(settings, source, mirror / "kb" / "synthetic.txt")
                with self.assertRaisesRegex(NonRetryableParseError, "missing"):
                    verify_source_file(settings, SimpleNamespace(source_root="gone"), mirror / "gone" / "x.md")
            with patch.dict(os.environ, {"KB_MIRROR_ALLOW_LINKED_DIRS": "1"}):
                verify_source_file(settings, source, mirror / "kb" / "synthetic.txt")       # explicitly allowed: the target is the boundary
        src = _repo_file("app/kb_pipeline/pipeline/parse_job.py")
        self.assertLess(src.index("verify_source_file(settings, source, path)"), src.index("_parse_blocks(settings, source, path"))

    def test_each_job_reads_the_source_from_the_registry(self) -> None:
        """2026-09-29 audit: one worker process runs several jobs; with only the source snapshot taken at start-up
        a renamed directory is checked against its old name and the job is judged non-retryable, and chunk
        settings changed in the console have no effect. Every job reads the registry by kb_id; a base that is
        not enrolled cancels the job."""
        from kb_pipeline import discovery
        from kb_pipeline.parsers.errors import NonRetryableParseError
        from kb_pipeline.pipeline.parse_job import JobCancelled, _source_for_file, verify_source_file

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            mirror = base / "mirror"; (mirror / "OldName").mkdir(parents=True)
            state = base / "state.db"; db.init_db(state)
            with db.connect(state) as con:
                discovery.init_schema(con)
                con.execute(
                    "INSERT INTO kb_sources(kb_id, source_root, collection, status, first_seen_at, last_seen_at, config_json) "
                    "VALUES('kb_001', 'OldName', 'kb_001', 'active', 1, 1, ?)",
                    (json.dumps({"max_tokens": 400, "overlap_tokens": 80}),))
            stale = KBSource(kb_id="kb_001", collection="kb_001", source_root="OldName", source_type="local_mirror",
                             max_tokens=400, overlap_tokens=80, physical_base=mirror / "OldName")
            settings = SimpleNamespace(mirror_root=mirror, state_db=state, sources={"kb_001": stale})
            # the directory is renamed and the scan has adopted it; the console also changed the chunk size
            (mirror / "OldName").rename(mirror / "NewName")
            doc = mirror / "NewName" / "doc.md"; doc.write_text("x", encoding="utf-8")
            with db.connect(state) as con:
                con.execute("UPDATE kb_sources SET source_root = 'NewName', config_json = ? WHERE kb_id = 'kb_001'",
                            (json.dumps({"max_tokens": 256, "overlap_tokens": 32}),))
            with patch.dict(os.environ, {"KB_MIRROR_ALLOW_LINKED_DIRS": ""}):
                with self.assertRaises(NonRetryableParseError):
                    verify_source_file(settings, stale, doc)                     # what the stale snapshot used to do
                fresh = _source_for_file(settings, "kb_001")
                self.assertEqual((fresh.source_root, fresh.max_tokens, fresh.overlap_tokens), ("NewName", 256, 32))
                verify_source_file(settings, fresh, doc)                         # passes with the registry's current row
            with db.connect(state) as con:
                con.execute("UPDATE kb_sources SET status = 'inactive' WHERE kb_id = 'kb_001'")
            with self.assertRaisesRegex(JobCancelled, "not enrolled or has been disabled"):
                _source_for_file(settings, "kb_001")                             # the snapshot still holds the pre-close source: not used
            settings.sources = {}
            with self.assertRaisesRegex(JobCancelled, "not enrolled or has been disabled"):
                _source_for_file(settings, "kb_001")                             # not a failure: nothing to retry, nothing to block
            # a source outside the mirror (KB_EXTRA_SOURCES_JSON) keeps its own directory as the boundary
            extra_dir = base / "elsewhere"; extra_dir.mkdir()
            inside = extra_dir / "a.md"; inside.write_text("a", encoding="utf-8")
            extra = KBSource(kb_id="kb_900", collection="kb_900", source_root="elsewhere", source_type="local_mirror",
                             max_tokens=400, overlap_tokens=80, physical_base=extra_dir)
            settings.sources = {"kb_900": extra}
            self.assertIs(_source_for_file(settings, "kb_900"), extra)
            with patch.dict(os.environ, {"KB_MIRROR_ALLOW_LINKED_DIRS": ""}):
                verify_source_file(settings, extra, inside)
                with self.assertRaisesRegex(NonRetryableParseError, "outside the enrolled directory"):
                    verify_source_file(settings, extra, doc)
        # the two other readers of mirrored files apply the same boundary
        self.assertIn("return path if inside_boundary(path, base.resolve()) else None", _repo_file("app/kb_pipeline/graph/build.py"))
        self.assertIn("if not inside_boundary(p, (Path(settings.mirror_root) / top).resolve()):", _repo_file("app/kb_search/images.py"))
        self.assertIn("elif not inside_boundary(p, Path(settings.mirror_root).resolve()):", _repo_file("app/kb_search/images.py"))

    def test_graph_file_reads_recheck_the_root_policy_at_read_time(self) -> None:
        """Third review, item 2: the deterministic extractor's file lookup applies the root policy on every
        read, so a source loaded before its directory was swapped for a link reads nothing from the
        link's target; with the switch on the target is the boundary."""
        import shutil

        from kb_pipeline.graph.build import deterministic_extractor

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            mirror = base / "mirror"; (mirror / "kb").mkdir(parents=True)
            (mirror / "kb" / "config.json").write_text('{"own": true}', encoding="utf-8")
            outside = base / "outside"; outside.mkdir(); (outside / "config.json").write_text('{"marker": "outside"}', encoding="utf-8")
            settings = SimpleNamespace(mirror_root=mirror)
            source = SimpleNamespace(source_root="kb", kb_id="k")            # loaded while the directory was real
            with patch.dict(os.environ, {"KB_MIRROR_ALLOW_LINKED_DIRS": ""}):
                ex = deterministic_extractor(settings, source, [])
                self.assertEqual(ex.file_for("config.json"), mirror / "kb" / "config.json")
                self.assertIn("own", ex._text("config.json"))
                shutil.rmtree(mirror / "kb"); (mirror / "kb").symlink_to(outside, target_is_directory=True)
                ex2 = deterministic_extractor(settings, source, [])
                self.assertIsNone(ex2.file_for("config.json"))
                self.assertEqual(ex2._text("config.json"), "")               # not the outside marker
                self.assertIsNone(ex.file_for("config.json"))                # the earlier extractor rechecks too
            with patch.dict(os.environ, {"KB_MIRROR_ALLOW_LINKED_DIRS": "1"}):
                self.assertIn("outside", deterministic_extractor(settings, source, [])._text("config.json"))

    def test_top_level_symlinked_directories_need_an_explicit_switch(self) -> None:
        from kb_pipeline.discovery import discover_directories

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            mirror = base / "mirror"; mirror.mkdir()
            (mirror / "real").mkdir()
            elsewhere = base / "elsewhere"; elsewhere.mkdir()
            (mirror / "linked").symlink_to(elsewhere, target_is_directory=True)
            with patch.dict(os.environ, {"KB_MIRROR_ALLOW_LINKED_DIRS": ""}):
                self.assertEqual(discover_directories(mirror), ["real"])
            with patch.dict(os.environ, {"KB_MIRROR_ALLOW_LINKED_DIRS": "1"}):
                self.assertEqual(discover_directories(mirror), ["linked", "real"])
