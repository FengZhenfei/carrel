"""Maintenance and operations: cleanup jobs, the yield policy, systemd units, scripts."""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from kb_pipeline import db

from _support import _CodexAudit20260906TestsSupport, _CodexFinalTestsSupport, _local_file, _repo_file


def _render_unit(template: Path) -> str:
    """The installed version is the copy that install-systemd.sh produces by replacing __CARREL_HOME__ with the
    repository path; render by the same rule before comparing, otherwise the template always looks "drifted"."""
    home = template.resolve().parents[2]
    return template.read_text(encoding="utf-8").replace("__CARREL_HOME__", str(home))


class MaintenanceCleanupTests(unittest.TestCase):
    """The cleanup actions had only ever been run on the real box, and they delete things from disk."""

    def test_job_history_pruning_keeps_live_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"; db.init_db(state)
            old = int(time.time()) - 60 * 86400
            with db.connect(state) as con:
                for jid, status, finished in (("done-old", "done", old),
                                              ("done-new", "done", int(time.time())),
                                              ("failed-old", "failed", old),
                                              ("queued", "queued", None)):
                    con.execute(
                        "INSERT INTO jobs(job_id, kb_id, collection, job_type, status, priority, "
                        "next_attempt_at, created_at, updated_at, finished_at) "
                        "VALUES(?, 'kb', 'c', 'parse', ?, 100, 0, 1, ?, ?)",
                        (jid, status, finished or int(time.time()), finished))
                con.commit()
                removed = db.prune_job_history(con, retention_days=30)
                left = {r[0] for r in con.execute("SELECT job_id FROM jobs").fetchall()}
            self.assertEqual(removed["jobs"], 1)          # only long-finished rows are removed
            self.assertEqual(left, {"done-new", "failed-old", "queued"})  # failed rows are kept for troubleshooting

    def test_latest_successful_parse_of_a_live_file_outlives_the_retention(self) -> None:
        """The file table's "Indexed at" and the job timeline come from the latest successful parse job: while the
        file exists, that job is not pruned with the job history."""
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"; db.init_db(state)
            old = int(time.time()) - 60 * 86400
            with db.connect(state) as con:
                file = _local_file("a.pdf", checksum="c1")
                db.upsert_file(con, file, status="indexed")
                fid = db.file_id_for(file.kb_id, file.file_key)
                for jid, job_type, status, file_id, finished in (
                        ("parse-older", "parse", "done", fid, old - 500),
                        ("parse-latest", "parse", "done", fid, old),
                        ("parse-cancelled", "parse", "cancelled", fid, old + 100),
                        ("meta-done", "metadata_update", "done", fid, old + 200),
                        ("parse-gone-file", "parse", "done", "project_materials:404", old)):
                    con.execute(
                        "INSERT INTO jobs(job_id, file_id, kb_id, collection, job_type, status, priority, "
                        "next_attempt_at, created_at, updated_at, finished_at) VALUES(?, ?, 'kb', 'c', ?, ?, 100, 0, 1, ?, ?)",
                        (jid, file_id, job_type, status, finished, finished))
                    db.add_job_event(con, jid, "info", "x")
                con.commit()
                planned = db.prune_job_history(con, retention_days=30, dry_run=True)
                removed = db.prune_job_history(con, retention_days=30)
                self.assertEqual((planned["jobs"], planned["job_events"]), (removed["jobs"], removed["job_events"]))
                self.assertEqual(removed["jobs"], 4)
                self.assertEqual([r[0] for r in con.execute("SELECT job_id FROM jobs")], ["parse-latest"])
                self.assertEqual([r[0] for r in con.execute("SELECT DISTINCT job_id FROM job_events")], ["parse-latest"])
                row = con.execute("SELECT MAX(finished_at) FROM jobs WHERE file_id = ? AND job_type = 'parse' AND status = 'done'",
                                  (fid,)).fetchone()
                self.assertEqual(int(row[0]), old)                       # the index time is still there
                db.purge_file_state(con, fid)                            # it goes when the file is purged
                self.assertEqual(con.execute("SELECT COUNT(*) FROM jobs").fetchone()[0], 0)

    def test_weekly_cleanup_never_touches_the_vlm_cache(self) -> None:
        """The VLM description cache is reused across versions: letting the weekly rotation move it away means
        every image gets burned through the GPU again."""
        from unittest import mock

        from kb_pipeline import maintenance

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"; (cache / "vlm-cache").mkdir(parents=True)
            (cache / "vlm-cache" / "a.json").write_text("{}", encoding="utf-8")
            (cache / "parse").mkdir(); (cache / "parse" / "keep.txt").write_text("x", encoding="utf-8")
            (cache / "scratch").mkdir(); (cache / "scratch" / "tmp.bin").write_text("x", encoding="utf-8")
            settings = SimpleNamespace(runtime_dir=root, cache_dir=cache, log_dir=root / "logs",
                                       state_db=root / "s.db", qdrant_inactive_retention_days=7)
            (root / "logs").mkdir()
            with mock.patch.object(maintenance, "service_busy", return_value=(False, [])):
                out = maintenance.weekly_cache_cleanup(settings)
            self.assertTrue((cache / "vlm-cache" / "a.json").exists())   # kept
            self.assertTrue((cache / "parse" / "keep.txt").exists())     # parse assets are kept too
            self.assertFalse((cache / "scratch" / "tmp.bin").exists())   # everything else is rotated away
            self.assertEqual(out["moved_entries"], {str(cache): 1})      # the cache directory is the only thing rotated


class VlmCacheReferenceTests(unittest.TestCase):
    """Entries of the image description / visual vector cache follow the pictures in the parse cache; they do not
    expire by age."""

    def _world(self, tmp: str):
        import hashlib

        cache = Path(tmp) / "cache"
        images = cache / "parse" / "kb_t" / "1" / "sha-a" / "mineru" / "images"
        images.mkdir(parents=True)
        (images / "fig.jpg").write_bytes(b"picture-in-use")
        used = hashlib.sha256(b"picture-in-use").hexdigest()
        gone = hashlib.sha256(b"picture-of-a-deleted-file").hexdigest()
        old = time.time() - 400 * 86400

        def entry(rel: str, *, mtime: float | None = old) -> Path:
            path = cache / "vlm-cache" / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
            if mtime is not None:
                os.utime(path, (mtime, mtime))
            return path

        return cache, used, gone, entry

    def test_entries_follow_the_pictures_in_the_parse_cache(self) -> None:
        from kb_pipeline import maintenance

        with tempfile.TemporaryDirectory() as tmp:
            cache, used, gone, entry = self._world(tmp)
            kept = [entry(f"{used[:2]}/{used}-12345678.caption.json"), entry(f"{used[:2]}/{used}.embed.json"),
                    entry(f"table-repair/{used[:2]}/{used}-abcdef12.json"),
                    entry(f"{gone[:2]}/{gone}-fresh000.caption.json", mtime=None),      # just written: kept for now, unreferenced or not
                    entry("notes.txt")]                                                   # unknown files are not touched
            dropped = [entry(f"{gone[:2]}/{gone}-12345678.caption.json"), entry(f"{gone[:2]}/{gone}.embed.json"),
                       entry(f"table-repair/{gone[:2]}/{gone}-abcdef12.json")]
            planned = maintenance.prune_vlm_cache(cache, dry_run=True)
            self.assertEqual((planned["entries"], planned["referenced"], planned["removed"]), (7, 3, 3))
            self.assertTrue(all(p.exists() for p in kept + dropped))                      # a dry run deletes nothing
            done = maintenance.prune_vlm_cache(cache)
            self.assertEqual(done["removed"], 3)
            self.assertEqual([p.exists() for p in kept], [True] * 5)                      # in use: 400 days old and still kept
            self.assertEqual([p.exists() for p in dropped], [False] * 3)                  # picture gone: transcription and vectors go too

    def test_nothing_is_dropped_when_the_references_cannot_be_counted(self) -> None:
        from unittest import mock

        from kb_pipeline import maintenance
        from kb_pipeline.vision import images

        with tempfile.TemporaryDirectory() as tmp:
            cache, used, gone, entry = self._world(tmp)
            stale = entry(f"{gone[:2]}/{gone}.embed.json")
            with mock.patch.object(images, "file_hash", side_effect=OSError("read failed")):
                out = maintenance.prune_vlm_cache(cache)
            self.assertEqual((out["unreadable_images"], out["removed"]), (1, 0))
            self.assertTrue(stale.exists())
            (cache / "parse").rename(cache / "parse.moved")                       # parse cache directory missing: nothing deleted either
            self.assertEqual(maintenance.prune_vlm_cache(cache)["removed"], 0)
            self.assertTrue(stale.exists())

    def test_weekly_cleanup_reports_what_it_dropped(self) -> None:
        from unittest import mock

        from kb_pipeline import maintenance

        with tempfile.TemporaryDirectory() as tmp:
            cache, used, gone, entry = self._world(tmp)
            live, stale = entry(f"{used[:2]}/{used}.embed.json"), entry(f"{gone[:2]}/{gone}.embed.json")
            root = Path(tmp)
            (root / "logs").mkdir()
            settings = SimpleNamespace(runtime_dir=root, cache_dir=cache, log_dir=root / "logs",
                                       state_db=root / "s.db", qdrant_inactive_retention_days=7)
            with mock.patch.object(maintenance, "service_busy", return_value=(False, [])):
                out = maintenance.weekly_cache_cleanup(settings)
            self.assertEqual(out["vlm_cache_pruned"], 1)
            self.assertEqual((live.exists(), stale.exists()), (True, False))
            self.assertTrue((cache / "parse" / "kb_t" / "1" / "sha-a" / "mineru" / "images" / "fig.jpg").exists())


class _GcWorld:
    """The scene for the reclaim cases: one KB, a real QdrantClient(":memory:"), a state database and a parse cache
    directory. Qdrant is the real client: behaviour such as the with_payload whitelist or deleting points by filter
    cannot be imitated by a fake."""

    KB = "kb_t"

    def __init__(self, tmp: str) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models

        self.root = Path(tmp)
        self.state = self.root / "s.db"
        db.init_db(self.state)
        self.q = QdrantClient(":memory:")
        self.q.create_collection(self.KB, vectors_config={"text": models.VectorParams(size=4, distance=models.Distance.COSINE)})
        self.settings = SimpleNamespace(state_db=self.state, cache_dir=self.root / "cache", qdrant_url="http://q",
                                        qdrant_api_key="", sources={self.KB: SimpleNamespace(kb_id=self.KB, collection=self.KB)})
        self.now = int(time.time())

    def add_file(self, name: str, *, version: str | None, indexed: str | None = None, status: str = "indexed",
                 seen_days_ago: float = 0) -> int:
        """version=None is a file scanned without content hashing: its checksum is empty and its content version is
        "mtime:...:size:...". Returns the file_key."""
        from kb_pipeline.localfs.scanner import stable_int
        from kb_pipeline.models import SourceFile

        file = SourceFile(kb_id=self.KB, collection=self.KB, source_root="库", source_type="local_mirror",
                          file_key=stable_int(f"file:{self.KB}:{name}"), source_path=f"库/{name}", rel_path=name, filename=name,
                          dir="", physical_path=f"/nowhere/{name}", mime_type="application/pdf", size=12, mtime=1,
                          checksum=version)
        with db.connect(self.state) as con:
            db.upsert_file(con, file, status=status)
            con.execute("UPDATE files SET indexed_version = ?, last_seen_at = ? WHERE file_id = ?",
                        (indexed if indexed is not None else file.content_version,
                         self.now - int(seen_days_ago * 86400), db.file_id_for(self.KB, file.file_key)))
        return file.file_key

    def add_batch(self, file_key: int, version: str, tag: str, n: int, *, status: str,
                  inactive_days_ago: float | None = None, visual_first: bool = False, rows: bool = True) -> list[str]:
        """A batch of chunks: Qdrant points plus rows in the chunk ledger. status is the ledger status; with
        inactive_days_ago given, the points are inactive."""
        from qdrant_client.http import models

        from kb_pipeline.pipeline.parse_job import _point_id

        ids: list[str] = []
        points = []
        with db.connect(self.state) as con:
            for i in range(n):
                uid = f"{self.KB}:{file_key}:{version}:{tag}:b:{i}"
                pid = _point_id(uid)
                ids.append(pid)
                payload = {"kb_id": self.KB, "doc_id": f"{self.KB}:{file_key}", "content_version": version,
                           "chunk_uid": uid, "is_active": inactive_days_ago is None}
                if inactive_days_ago is not None:
                    payload["inactive_at"] = self.now - int(inactive_days_ago * 86400)
                if visual_first and i == 0:
                    payload["visual_sha256"] = "ab" * 32
                points.append(models.PointStruct(id=pid, vector={"text": [1.0, 0.0, 0.0, 0.0]}, payload=payload))
                if rows:
                    con.execute(
                        "INSERT INTO chunks(chunk_uid, file_id, content_version, chunk_index, point_id, collection, status, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (uid, db.file_id_for(self.KB, file_key), version, i, pid, self.KB, status, self.now))
        self.q.upsert(self.KB, points=points, wait=True)
        return ids

    def cache(self, file_key: int, version: str) -> Path:
        from kb_pipeline.pipeline.parse_job import parse_cache_dir

        path = parse_cache_dir(self.settings, self.KB, file_key, version)
        (path / "images").mkdir(parents=True)
        (path / "images" / "x.jpg").write_bytes(b"jpg")
        return path

    def run(self, *, dry_run: bool = False) -> dict:
        from unittest import mock

        from kb_pipeline import maintenance

        with mock.patch.object(maintenance, "service_busy", return_value=(False, [])), \
                mock.patch("kb_pipeline.vector.qdrant.client", return_value=self.q):
            return maintenance.parse_assets_gc(self.settings, retention_days=7, dry_run=dry_run)

    def rows(self, file_key: int) -> dict[str, int]:
        with db.connect(self.state) as con:
            return {str(r[0]): int(r[1]) for r in con.execute(
                "SELECT status, COUNT(*) FROM chunks WHERE file_id = ? GROUP BY status", (db.file_id_for(self.KB, file_key),))}

    def has(self, point_ids: list[str]) -> list[bool]:
        found = {str(r.id) for r in self.q.retrieve(self.KB, ids=point_ids, with_payload=False, with_vectors=False)}
        return [p in found for p in point_ids]


class ParseAssetsGcTests(unittest.TestCase):
    """The nightly reclaim deletes points, chunk rows and parse caches. Both sides must hold: the active chunks,
    restore ledger and cache of a version in use stay untouched; old batches left by re-chunking the same version,
    superseded versions and deleted files past the retention period really are reclaimed."""

    def test_same_version_rechunk_leftovers_are_reclaimed_and_the_live_batch_is_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            w = _GcWorld(tmp)
            fk = w.add_file("a.pdf", version="sha-a")
            live = w.add_batch(fk, "sha-a", "p2", 5, status="active")
            old = w.add_batch(fk, "sha-a", "p1", 3, status="inactive", inactive_days_ago=10)
            recent = w.add_batch(fk, "sha-a", "p1b", 2, status="inactive", inactive_days_ago=2)   # still inside the retention period
            cache = w.cache(fk, "sha-a")
            out = w.run()
            self.assertEqual(out["errors"], [])
            self.assertEqual((out["totals"]["qdrant_points"], out["totals"]["sqlite_chunks"], out["totals"]["cache_dirs"]), (3, 3, 0))
            self.assertEqual(w.has(old), [False] * 3)                       # the expired old batch is reclaimed
            self.assertEqual(w.has(live) + w.has(recent), [True] * 7)       # the live batch and the unexpired old one remain
            self.assertEqual(w.rows(fk), {"active": 5, "inactive": 2})
            self.assertTrue((cache / "images" / "x.jpg").exists())          # the parse cache of the live version is untouched
            self.assertTrue(out["qdrant_versions"][0]["in_use"])
            again = w.run()                                                  # another round: nothing to delete, and no nightly skip
            self.assertEqual((again["totals"]["qdrant_points"], again["skipped_items"]), (0, []))

    def test_indexed_version_is_in_use_while_the_new_content_is_not_parsed_yet(self) -> None:
        """The content changed and the new version is stuck in backoff or failed: the old version is still the one
        being served, so its chunk ledger and cache must stay."""
        with tempfile.TemporaryDirectory() as tmp:
            w = _GcWorld(tmp)
            fk = w.add_file("a.pdf", version="sha-new", indexed="sha-old", status="seen")
            live = w.add_batch(fk, "sha-old", "p2", 5, status="active")
            old = w.add_batch(fk, "sha-old", "p1", 3, status="inactive", inactive_days_ago=10)
            cache = w.cache(fk, "sha-old")
            out = w.run()
            self.assertEqual(w.has(live), [True] * 5)
            self.assertEqual(w.has(old), [False] * 3)
            self.assertEqual(w.rows(fk), {"active": 5})
            self.assertTrue(cache.exists())
            self.assertEqual(out["totals"]["cache_dirs"], 0)

    def test_content_hash_switched_off_does_not_expose_the_live_version(self) -> None:
        """KB_LOCAL_SCAN_HASH=0: the checksum is empty and the version is "mtime:...". The verdict does not rely on
        the checksum and comes out the same as with hashing on."""
        with tempfile.TemporaryDirectory() as tmp:
            w = _GcWorld(tmp)
            fk = w.add_file("a.pdf", version=None)
            version = "mtime:1:size:12"
            live = w.add_batch(fk, version, "p2", 4, status="active")
            old = w.add_batch(fk, version, "p1", 2, status="inactive", inactive_days_ago=9)
            cache = w.cache(fk, version)
            w.run()
            self.assertEqual(w.has(live) + w.has(old), [True] * 4 + [False] * 2)
            self.assertEqual(w.rows(fk), {"active": 4})
            self.assertTrue(cache.exists())

    def test_deleted_file_keeps_its_restore_batch_inside_the_undo_window(self) -> None:
        """The file was just deleted (or the KB just closed): the points just went inactive and the ledger says
        deleted. Older leftovers of the same version are reclaimed once expired; the batch kept for a restore and
        the cache stay."""
        with tempfile.TemporaryDirectory() as tmp:
            w = _GcWorld(tmp)
            fk = w.add_file("a.pdf", version="sha-a", status="deleted", seen_days_ago=1)
            restore = w.add_batch(fk, "sha-a", "p2", 5, status="deleted", inactive_days_ago=1)
            old = w.add_batch(fk, "sha-a", "p1", 3, status="inactive", inactive_days_ago=10)
            cache = w.cache(fk, "sha-a")
            out = w.run()
            self.assertEqual(out["deleted_files"], [])
            self.assertEqual(w.has(restore) + w.has(old), [True] * 5 + [False] * 3)
            self.assertEqual(w.rows(fk), {"deleted": 5})
            self.assertTrue(cache.exists())
            with db.connect(w.state) as con:
                self.assertEqual(sorted(db.chunk_point_ids_to_restore(con, db.file_id_for(w.KB, fk), "sha-a")), sorted(restore))

    def test_deleted_file_past_the_undo_window_is_reclaimed_entirely(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            w = _GcWorld(tmp)
            fk = w.add_file("a.pdf", version="sha-a", status="deleted", seen_days_ago=9)
            restore = w.add_batch(fk, "sha-a", "p2", 5, status="deleted", inactive_days_ago=9)
            old = w.add_batch(fk, "sha-a", "p1", 3, status="inactive", inactive_days_ago=20)
            cache = w.cache(fk, "sha-a")
            keep = w.add_file("b.pdf", version="sha-b")
            keep_points = w.add_batch(keep, "sha-b", "p2", 2, status="active")
            out = w.run()
            self.assertEqual(out["errors"], [])
            self.assertEqual(w.has(restore) + w.has(old), [False] * 8)
            self.assertEqual(w.rows(fk), {})
            self.assertFalse(cache.parent.exists())                          # the file's whole parse cache is reclaimed
            self.assertEqual([d["file_id"] for d in out["deleted_files"]], [db.file_id_for(w.KB, fk)])
            with db.connect(w.state) as con:
                self.assertIsNone(db.get_file_by_id(con, db.file_id_for(w.KB, fk)))
            self.assertEqual(w.has(keep_points), [True] * 2)                 # other files are not affected
            self.assertEqual(w.rows(keep), {"active": 2})

    def test_superseded_version_loses_its_points_rows_and_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            w = _GcWorld(tmp)
            fk = w.add_file("a.pdf", version="sha-new")
            live = w.add_batch(fk, "sha-new", "p1", 4, status="active")
            old = w.add_batch(fk, "sha-old", "p1", 3, status="inactive", inactive_days_ago=8)
            old_cache, live_cache = w.cache(fk, "sha-old"), w.cache(fk, "sha-new")
            out = w.run()
            self.assertEqual(w.has(live) + w.has(old), [True] * 4 + [False] * 3)
            self.assertEqual(w.rows(fk), {"active": 4})
            self.assertFalse(old_cache.exists())
            self.assertTrue(live_cache.exists())
            self.assertEqual(out["totals"]["cache_dirs"], 1)
            self.assertFalse(out["qdrant_versions"][0]["in_use"])

    def test_points_the_ledger_still_needs_are_never_deleted(self) -> None:
        """The file came back only after the retention period and the restore job has not run yet: the batch has
        "expired", but the ledger marks it deleted, waiting to become active again. The same goes for points the
        ledger marks active. Neither is deleted, and the cache stays."""
        with tempfile.TemporaryDirectory() as tmp:
            w = _GcWorld(tmp)
            fk = w.add_file("a.pdf", version="sha-a", status="seen")           # the scan has already seen it return
            restore = w.add_batch(fk, "sha-a", "p2", 3, status="deleted", inactive_days_ago=9)
            odd = w.add_batch(fk, "sha-a", "p3", 1, status="active", inactive_days_ago=9)
            old = w.add_batch(fk, "sha-a", "p1", 2, status="inactive", inactive_days_ago=9)
            orphan = w.add_batch(fk, "sha-a", "p0", 1, status="inactive", inactive_days_ago=9, rows=False)   # a point without a ledger row
            cache = w.cache(fk, "sha-a")
            out = w.run()
            self.assertEqual(w.has(restore) + w.has(odd), [True] * 4)
            self.assertEqual(w.has(old) + w.has(orphan), [False] * 3)
            self.assertEqual(w.rows(fk), {"deleted": 3, "active": 1})
            self.assertEqual(out["qdrant_versions"][0]["qdrant_points_kept"], 4)
            self.assertTrue(cache.exists())

    def test_a_leading_image_point_does_not_change_the_verdict(self) -> None:
        """Grouping does not read the image hash: with visual_sha256 on the first point of a group, the live version
        is still recognized."""
        with tempfile.TemporaryDirectory() as tmp:
            w = _GcWorld(tmp)
            fk = w.add_file("a.pdf", version="sha-a")
            live = w.add_batch(fk, "sha-a", "p2", 3, status="active")
            old = w.add_batch(fk, "sha-a", "p1", 2, status="inactive", inactive_days_ago=10, visual_first=True)
            cache = w.cache(fk, "sha-a")
            from kb_pipeline.vector.qdrant import inactive_doc_versions_older_than

            groups = inactive_doc_versions_older_than(w.q, w.KB, w.now - 7 * 86400)
            self.assertEqual([(g["doc_id"], g["content_version"], g["points"]) for g in groups], [(f"{w.KB}:{fk}", "sha-a", 2)])
            self.assertNotIn("sha256", groups[0])
            w.run()
            self.assertEqual(w.has(live) + w.has(old), [True] * 3 + [False] * 2)
            self.assertEqual(w.rows(fk), {"active": 3})
            self.assertTrue(cache.exists())

    def test_version_shared_with_another_live_file_is_left_alone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            w = _GcWorld(tmp)
            gone = w.add_file("a.pdf", version="sha-a", status="deleted", seen_days_ago=9)
            gone_points = w.add_batch(gone, "sha-a", "p1", 2, status="deleted", inactive_days_ago=9)
            cache = w.cache(gone, "sha-a")
            w.add_file("copy.pdf", version="sha-a")                           # another live copy with the same content
            out = w.run()
            self.assertEqual([s["reason"] for s in out["skipped_items"]], ["active_same_hash", "deleted_file_active_same_hash"])
            self.assertEqual(w.has(gone_points), [True] * 2)
            self.assertEqual(w.rows(gone), {"deleted": 2})
            self.assertTrue(cache.exists())

    def test_dry_run_counts_without_deleting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            w = _GcWorld(tmp)
            fk = w.add_file("a.pdf", version="sha-new")
            w.add_batch(fk, "sha-new", "p1", 2, status="active")
            old = w.add_batch(fk, "sha-old", "p1", 3, status="inactive", inactive_days_ago=8)
            old_cache = w.cache(fk, "sha-old")
            planned = w.run(dry_run=True)
            self.assertEqual((planned["totals"]["qdrant_points"], planned["totals"]["sqlite_chunks"], planned["totals"]["cache_dirs"]),
                             (3, 3, 1))
            self.assertEqual(w.has(old), [True] * 3)
            self.assertEqual(w.rows(fk), {"active": 2, "inactive": 3})
            self.assertTrue(old_cache.exists())
            done = w.run()
            self.assertEqual(done["totals"]["qdrant_points"], planned["totals"]["qdrant_points"])

    def test_deleted_file_is_not_purged_while_something_still_holds_it(self) -> None:
        """A deleted file past the retention period that still has queued jobs, or still has active points in the
        vector store: its state and parse cache stay for now."""
        with tempfile.TemporaryDirectory() as tmp:
            w = _GcWorld(tmp)
            queued = w.add_file("queued.pdf", version="sha-q", status="deleted", seen_days_ago=9)
            alive = w.add_file("alive.pdf", version="sha-l", status="deleted", seen_days_ago=9)
            alive_points = w.add_batch(alive, "sha-l", "p1", 2, status="active")          # the delete job has not deactivated the points yet
            caches = [w.cache(queued, "sha-q"), w.cache(alive, "sha-l")]
            with db.connect(w.state) as con:
                db.enqueue_job(con, ingest_run_id=None, file_id=db.file_id_for(w.KB, queued), kb_id=w.KB, collection=w.KB,
                               file_key=queued, job_type="delete", dedupe_key="delete:q")
            out = w.run()
            self.assertEqual(sorted(s["reason"] for s in out["skipped_items"]),
                             ["deleted_file_has_pending_jobs", "deleted_file_still_has_active_points"])
            self.assertEqual(out["deleted_files"], [])
            self.assertTrue(all(c.exists() for c in caches))
            self.assertEqual(w.has(alive_points), [True] * 2)
            with db.connect(w.state) as con:
                self.assertEqual(con.execute("SELECT COUNT(*) FROM files").fetchone()[0], 2)

    def test_points_without_identity_are_swept_once_expired(self) -> None:
        """Inactive points whose payload lost doc_id / content_version fit no group and are swept on their own
        once expired; unexpired and active ones stay."""
        from qdrant_client.http import models

        from kb_pipeline.pipeline.parse_job import _point_id

        with tempfile.TemporaryDirectory() as tmp:
            w = _GcWorld(tmp)
            ids = [_point_id(f"malformed-{i}") for i in range(3)]
            payloads = [{"is_active": False, "inactive_at": w.now - 10 * 86400},
                        {"is_active": False, "inactive_at": w.now - 86400},
                        {"is_active": True}]
            w.q.upsert(w.KB, wait=True, points=[models.PointStruct(id=i, vector={"text": [1.0, 0.0, 0.0, 0.0]}, payload=p)
                                               for i, p in zip(ids, payloads)])
            out = w.run()
            self.assertGreaterEqual(out["totals"]["qdrant_malformed_points"], 1)
            self.assertEqual(w.has(ids), [False, True, True])

    def test_manual_point_gc_only_takes_expired_inactive_points(self) -> None:
        """The manual cleanup qdrant-gc deletes by point only; active points and inactive ones still inside the
        retention period stay."""
        from unittest import mock

        from kb_pipeline import maintenance

        with tempfile.TemporaryDirectory() as tmp:
            w = _GcWorld(tmp)
            fk = w.add_file("a.pdf", version="sha-a")
            live = w.add_batch(fk, "sha-a", "p2", 3, status="active")
            old = w.add_batch(fk, "sha-a", "p1", 2, status="inactive", inactive_days_ago=10)
            recent = w.add_batch(fk, "sha-a", "p1b", 2, status="inactive", inactive_days_ago=2)
            with mock.patch.object(maintenance, "service_busy", return_value=(False, [])), \
                    mock.patch("kb_pipeline.vector.qdrant.client", return_value=w.q):
                planned = maintenance.qdrant_inactive_gc(w.settings, retention_days=7, dry_run=True)
                self.assertEqual(w.has(old), [True] * 2)
                done = maintenance.qdrant_inactive_gc(w.settings, retention_days=7)
            self.assertEqual((planned["total_points"], done["total_points"], done["errors"]), (2, 2, []))
            self.assertEqual(w.has(live) + w.has(recent) + w.has(old), [True] * 5 + [False] * 2)

    def test_point_revived_after_listing_survives_the_delete(self) -> None:
        """Between listing the ids and deleting them, a point was written back by a re-parse (or turned active by a
        restore): deleting by id must not take it along."""
        from kb_pipeline.vector.qdrant import delete_expired_inactive_points, expired_inactive_point_ids

        with tempfile.TemporaryDirectory() as tmp:
            w = _GcWorld(tmp)
            fk = w.add_file("a.pdf", version="sha-a")
            old = w.add_batch(fk, "sha-a", "p1", 3, status="inactive", inactive_days_ago=10)
            cutoff = w.now - 7 * 86400
            listed = expired_inactive_point_ids(w.q, w.KB, doc_id=f"{w.KB}:{fk}", content_version="sha-a", cutoff_ts=cutoff)
            self.assertEqual(sorted(listed), sorted(old))
            w.q.set_payload(w.KB, payload={"is_active": True}, points=[old[0]])
            w.q.delete_payload(w.KB, keys=["inactive_at"], points=[old[0]])
            self.assertEqual(delete_expired_inactive_points(w.q, w.KB, listed, cutoff), 2)
            self.assertEqual(w.has(old), [True, False, False])
            self.assertEqual(delete_expired_inactive_points(w.q, "no-such-collection", listed, cutoff), 0)


class StateBackupTests(unittest.TestCase):
    """The nightly backup takes only what cannot be regenerated (state database, env files, evaluation sets),
    keeps the newest N copies with tightened permissions, and runs before the nightly round waits for Qdrant."""

    def _base(self, tmp: str) -> tuple[Path, SimpleNamespace]:
        base = Path(tmp)
        for rel in ("runtime/state", "runtime/eval", "config", "deployment/compose"):
            (base / rel).mkdir(parents=True)
        state = base / "runtime" / "state" / "kb-pipeline.db"
        db.init_db(state)
        with db.connect(state) as con:
            con.execute("INSERT INTO llm_registry(name, base_url, api_key, model_id, notes, builtin, created_at, updated_at) "
                        "VALUES('m', 'http://x', 'k', 'id', '', 0, 1, 1)")
            con.commit()
        (base / "config" / "knowledge-base.env").write_text("A=1\n", encoding="utf-8")
        (base / "deployment" / "compose" / ".env").write_text("B=2\n", encoding="utf-8")
        (base / "runtime" / "eval" / "gold.json").write_text("{}", encoding="utf-8")
        settings = SimpleNamespace(state_db=state, env_file=base / "config" / "knowledge-base.env", runtime_dir=base / "runtime")
        return base, settings

    def test_backup_copies_state_env_and_eval_and_keeps_n(self) -> None:
        from kb_pipeline import maintenance

        with tempfile.TemporaryDirectory() as tmp:
            base, settings = self._base(tmp)
            stamps = iter(("20251231-000000", "20260101-000000", "20260102-000000", "20260103-000000"))   # the dry run takes one too
            with patch.object(maintenance, "ts", side_effect=lambda: next(stamps)):
                dry = maintenance.backup_state(settings, base_dir=base, keep=2, dry_run=True)
                self.assertFalse((base / "backups").exists())                        # a dry run writes nothing
                self.assertEqual(dry["files"], ["knowledge-base.env", "deployment-compose.env", "eval/gold.json"])
                first = maintenance.backup_state(settings, base_dir=base, keep=2)
                second = maintenance.backup_state(settings, base_dir=base, keep=2)
                third = maintenance.backup_state(settings, base_dir=base, keep=2)
            target = Path(third["backup_dir"])                                        # the first copy is already pruned (keep=2)
            self.assertEqual(sorted(p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()),
                             ["deployment-compose.env", "eval/gold.json", "kb-pipeline.db", "knowledge-base.env"])
            with db.connect(target / "kb-pipeline.db") as con:                       # the online backup opens as a consistent snapshot
                self.assertEqual(tuple(con.execute("SELECT name, api_key FROM llm_registry").fetchone()), ("m", "k"))
            self.assertEqual(target.stat().st_mode & 0o777, 0o700)
            self.assertEqual((target / "eval").stat().st_mode & 0o777, 0o700)                # subdirectories tightened too
            for item in target.rglob("*"):
                if item.is_file():
                    self.assertEqual(item.stat().st_mode & 0o777, 0o600, item.name)
            self.assertEqual((base / "backups" / "state").stat().st_mode & 0o777, 0o700)
            self.assertGreater(first["bytes"], 0)
            self.assertEqual(first["files"], third["files"])
            self.assertEqual((second["pruned"], third["pruned"]), ([], ["20260101-000000"]))     # newest two kept
            self.assertEqual(sorted(p.name for p in (base / "backups" / "state").iterdir()), ["20260102-000000", "20260103-000000"])

    def test_nightly_run_backs_up_before_waiting_for_qdrant(self) -> None:
        """The backup needs no service: the nightly round backs up first and waits for Qdrant afterwards, so a night
        on which Qdrant does not come up still has its backup; a failed backup is not a maintenance failure."""
        script = _repo_file("scripts/kb-cleanup.sh")
        self.assertIn("cleanup backup", script)
        self.assertLess(script.index("state_backup || true"), script.index('maint_wait_qdrant_or_defer "cleanup-$COMMAND"'))
        self.assertIn("KB_BACKUP_KEEP=7", _repo_file("config/knowledge-base.env.example"))


class ScheduledCleanupCoverageTests(unittest.TestCase):
    """The cleanup subcommands the timers actually run must cover everything that grows monotonically.

    This guards against one specific kind of bug: the code hangs off a function that **no timer ever calls**,
    while the unit name looks as if it did. The graph_stale cleanup slipped through a whole round exactly this
    way: purge_stale lived in qdrant_inactive_gc, while the unit named carrel-qdrant-gc actually ran
    parse-assets-gc in its ExecStart. Code, comments and unit name each made sense on their own; put together,
    nobody did the work.
    """

    # subcommand -> handler function in maintenance, mirroring the dispatch in cli.cmd_cleanup.
    # The first test below checks it against the real dispatch in cli.py; if the routing changes, so must this.
    HANDLERS = {
        "status": "maintenance_status",
        "weekly": "weekly_cache_cleanup",
        "monthly": "monthly_log_cleanup",
        "qdrant-gc": "qdrant_inactive_gc",
        "qdrant-graph-gc": "qdrant_graph_collection_gc",
        "neo4j-graph-gc": "neo4j_graph_gc",
        "graph-gc": "graph_gc",
        "parse-assets-gc": "parse_assets_gc",
        "backup": "backup_state",
    }

    def _scheduled_subcommands(self) -> set[str]:
        """Derive from the systemd units in the repository which cleanup subcommands are really scheduled."""
        import re

        units = Path(__file__).resolve().parents[2] / "deployment" / "systemd"
        if not units.is_dir():
            self.skipTest("systemd 单元不在仓库里")
        found: set[str] = set()
        for service in units.glob("*.service"):
            text = service.read_text(encoding="utf-8")
            # only units with a matching .timer count: no timer, no recurring job
            if not service.with_suffix(".timer").exists():
                continue
            for line in text.splitlines():
                if not line.startswith("ExecStart="):
                    continue
                m = re.search(r"kb-cleanup\.sh\s+([a-z0-9-]+)", line)
                if m:
                    found.add(m.group(1))
        return found

    def test_scheduled_subcommands_all_route_somewhere(self) -> None:
        """Every subcommand written in a unit must have a branch in cli.cmd_cleanup, otherwise the timer runs a
        command that argparse rejects immediately, round after round, and a oneshot unit fails very quietly."""
        import inspect

        from kb_pipeline import cli

        dispatch = inspect.getsource(cli.cmd_cleanup)
        scheduled = self._scheduled_subcommands()
        self.assertTrue(scheduled, "没有从单元里解析出任何 cleanup 子命令")
        for sub in sorted(scheduled):
            self.assertIn(f'"{sub}"', dispatch, f"cmd_cleanup 里没有 {sub} 分支")
            self.assertIn(sub, self.HANDLERS, f"本用例的 HANDLERS 表缺 {sub}")

    def test_recurring_growth_is_actually_swept(self) -> None:
        """Things that only ever grow must be swept by a round that **really runs**.

        Each entry corresponds to an incident that happened or nearly happened:
        - prune_job_history: jobs/failures/ingest_runs only grow; failure rows carry 4000-character stack traces
        """
        import inspect

        from kb_pipeline import maintenance

        scheduled = self._scheduled_subcommands()
        swept = "\n".join(
            inspect.getsource(getattr(maintenance, self.HANDLERS[sub]))
            for sub in sorted(scheduled)
            if sub in self.HANDLERS and hasattr(maintenance, self.HANDLERS[sub])
        )
        for needle, what in (
            ("prune_job_history", "jobs / failures / ingest_runs 的历史行"),
        ):
            # assertTrue rather than assertIn: on failure the latter dumps hundreds of lines of function source,
            # burying the information that matters (what nobody sweeps, what the current schedule is).
            self.assertTrue(
                needle in swept,
                f"{what} 没有被任何**已排期**的 cleanup 子命令清理"
                f"({needle} 不在其中任何一个的实现里)。当前排期:{sorted(scheduled)}",
            )

    def test_repo_units_match_what_is_installed(self) -> None:
        """When the repository units drift from the installed ones, the two guards above check against thin air."""
        import shutil
        import subprocess as sp

        if not shutil.which("systemctl"):
            self.skipTest("没有 systemctl")
        units = Path(__file__).resolve().parents[2] / "deployment" / "systemd"
        if not units.is_dir():
            self.skipTest("systemd 单元不在仓库里")
        for service in sorted(units.glob("*.service")):
            proc = sp.run(["systemctl", "--user", "cat", service.name],
                          capture_output=True, text=True, timeout=30)
            if proc.returncode != 0:
                self.skipTest(f"{service.name} 未安装(非部署机)")
            installed = {l for l in proc.stdout.splitlines() if l.startswith("ExecStart=")}
            in_repo = {l for l in _render_unit(service).splitlines() if l.startswith("ExecStart=")}
            self.assertEqual(installed, in_repo, f"{service.name} 的 ExecStart 与仓库不一致")


class MaintenanceDeferPolicyTests(unittest.TestCase):
    """The yield policy for maintenance jobs that run into service_busy. Both extremes have bitten us: a silent
    exit 0 makes whole GC rounds vanish for days with nobody noticing; an exit 75 every time paints normal
    behaviour such as "avoiding the ingestion peak" as failed for long stretches, so real faults stop being
    believed. Now the unit only turns red once the run of consecutive deferrals exceeds the limit."""

    REPO = Path(__file__).resolve().parents[2]

    def _fake_repo(self, tmp: Path, exit_code: int) -> None:
        py = tmp / "app" / ".venv" / "bin" / "python"
        py.parent.mkdir(parents=True, exist_ok=True)
        py.write_text(f"#!/bin/sh\nexit {exit_code}\n", encoding="utf-8")
        py.chmod(0o755)

    def _fake_curl(self, tmp: Path, exit_code: int) -> None:
        """The script probes Qdrant with curl before starting; tests cannot depend on a real instance, so a fake
        curl goes at the front of PATH."""
        curl = tmp / "bin" / "curl"
        curl.parent.mkdir(parents=True, exist_ok=True)
        curl.write_text(f"#!/bin/sh\nexit {exit_code}\n", encoding="utf-8")
        curl.chmod(0o755)

    def _fake_docker(self, tmp: Path, *, running: bool = False) -> None:
        """After a successful run the script goes into the MinerU container to delete intermediate output, and the
        weekly round can prune images; when the tests run on the deployment host, the docker on PATH is the real
        one, so a fake goes in front as well: it only records its arguments and answers inspect as running says."""
        docker = tmp / "bin" / "docker"
        docker.parent.mkdir(parents=True, exist_ok=True)
        docker.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$(dirname "$0")/../docker.calls"\n'
                          f'[ "$1" = inspect ] && echo {"true" if running else "false"}\nexit 0\n', encoding="utf-8")
        docker.chmod(0o755)

    def _run(self, tmp: Path, script: str, *args: str, **extra: str):
        import subprocess

        if not (tmp / "bin" / "curl").exists():
            self._fake_curl(tmp, 0)             # Qdrant is up by default
        if not (tmp / "bin" / "docker").exists():
            self._fake_docker(tmp)
        # The environment is not inherited wholesale: KB_* exported on the host would change what the script deletes
        # and how old it must be; HOME points at the temporary directory, so the weekly ~/.cache cleanup cannot
        # reach the real home directory.
        env = {
            "PATH": str(tmp / "bin") + os.pathsep + os.environ.get("PATH", ""),
            "HOME": str(tmp),
            "KB_LOCAL_BASE_DIR": str(tmp),
            "KB_MAINT_STATE_DIR": str(tmp / "maint"),
            "KB_MAINT_DEFER_LIMIT": "3",
            "KB_CLEANUP_BUSY_ATTEMPTS": "1",   # give up at once; do not sleep 15 minutes in a test
            "KB_GRAPH_BUSY_ATTEMPTS": "1",
            "KB_QDRANT_WAIT_ROUNDS": "1",      # one probe decides; do not wait 5 minutes in a test
            "KB_QDRANT_WAIT_SECONDS": "0",
            **extra,
        }
        return subprocess.run(
            ["bash", str(self.REPO / "scripts" / script), *args],
            env=env, capture_output=True, text=True, timeout=60)

    def test_occasional_deferral_is_quiet_and_a_run_of_them_is_not(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._fake_repo(tmp, 75)            # the CLI says "yield" every time
            codes = [self._run(tmp, "kb-cleanup.sh", "parse-assets-gc").returncode
                     for _ in range(3)]
            self.assertEqual(codes, [0, 0, 75],
                             "前两轮该安静让路,第三轮才让单元变红")

            counter = tmp / "maint" / "cleanup-parse-assets-gc.defers"
            self.assertEqual(counter.read_text(encoding="utf-8").strip(), "3")

            # one successful run resets the count; the next deferral starts quiet again
            self._fake_repo(tmp, 0)
            self.assertEqual(self._run(tmp, "kb-cleanup.sh", "parse-assets-gc").returncode, 0)
            self.assertFalse(counter.exists(), "成功一轮必须清掉连续计数")
            self._fake_repo(tmp, 75)
            self.assertEqual(self._run(tmp, "kb-cleanup.sh", "parse-assets-gc").returncode, 0)

    def test_container_and_host_cleanup_stay_inside_the_sandbox(self) -> None:
        """The container and host cleanup after a successful run only touch the fake docker and the temporary HOME;
        what gets deleted does not depend on the environment of the shell running the tests."""
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmpdir, \
                mock.patch.dict(os.environ, {"KB_MINERU_OUTPUT_KEEP_MINUTES": "1", "KB_MINERU_CONTAINER": "other",
                                             "KB_HOST_HOUSEKEEPING": "1"}):
            tmp = Path(tmpdir)
            self._fake_repo(tmp, 0)
            calls = tmp / "docker.calls"
            out = self._run(tmp, "kb-cleanup.sh", "parse-assets-gc")
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertEqual(calls.read_text(encoding="utf-8").splitlines(), ["inspect -f {{.State.Running}} carrel-mineru"])
            self.assertIn("not running, skipped", out.stdout)
            # container running: inside it, the output directories older than a day are deleted
            calls.unlink()
            self._fake_docker(tmp, running=True)
            self.assertEqual(self._run(tmp, "kb-cleanup.sh", "parse-assets-gc").returncode, 0)
            probe, remove = calls.read_text(encoding="utf-8").splitlines()
            self.assertTrue(probe.startswith("inspect "), probe)
            self.assertTrue(remove.startswith("exec carrel-mineru sh -c find /data/output "), remove)
            self.assertIn("-mmin +1440 ", remove)
            # Weekly cleanup: cleaning outside the project is opt-in here (KB_HOST_HOUSEKEEPING), and the variable
            # exported by the shell running the tests does not reach the script
            calls.unlink()
            cache = tmp / ".cache" / "uv"
            cache.mkdir(parents=True)
            self.assertEqual(self._run(tmp, "kb-cleanup.sh", "weekly").returncode, 0)
            self.assertTrue(cache.exists())
            self.assertFalse(calls.exists())
            # switched on: the cache removed is the one under the temporary HOME, the image cleanup goes to the fake docker
            self.assertEqual(self._run(tmp, "kb-cleanup.sh", "weekly", KB_HOST_HOUSEKEEPING="1").returncode, 0)
            self.assertFalse(cache.exists())
            self.assertEqual([line.split()[0:2] for line in calls.read_text(encoding="utf-8").splitlines()],
                             [["image", "prune"], ["builder", "prune"]])

    def test_real_failures_still_surface_immediately(self) -> None:
        """Yielding is reserved for exit code 75. Any other non-zero exit code must surface unchanged rather than
        being swallowed by the counter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._fake_repo(tmp, 1)
            self.assertEqual(self._run(tmp, "kb-cleanup.sh", "parse-assets-gc").returncode, 1)
            self.assertFalse((tmp / "maint").exists(), "真失败不该记进让路计数")

    def test_graph_rebuild_check_shares_the_same_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._fake_repo(tmp, 75)
            codes = [self._run(tmp, "kb-graph-rebuild-check.sh").returncode for _ in range(3)]
            self.assertEqual(codes, [0, 0, 75])

    def test_both_schedulers_use_the_shared_policy(self) -> None:
        """There must be a single policy. A script with its own exit 75 would make the two sides diverge."""
        for name in ("kb-cleanup.sh", "kb-graph-rebuild-check.sh"):
            text = (self.REPO / "scripts" / name).read_text(encoding="utf-8")
            self.assertIn("lib/kb-maint-defer.sh", text, f"{name} 没有共用让路策略")
            self.assertIn("maint_defer_give_up", text)
            self.assertIn("maint_defer_clear", text)
            self.assertNotIn("exit 75", text, f"{name} 里还留着自己的 exit 75")

    def test_qdrant_not_ready_is_a_deferral_not_a_failure(self) -> None:
        """2026-09-06: when Qdrant could not be reached, the unit's ExecStartPre used to exit 75 directly,
        painting the unit failed every round while the services were deliberately stopped. The wait now lives in
        the script and goes through the same deferral counter: quiet exit, red only once the run exceeds the
        limit; once Qdrant is back the CLI runs as usual. Cleanup subcommands that never touch Qdrant neither
        wait nor count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._fake_repo(tmp, 1)             # the CLI would fail with 1 if it were reached, proving it was not
            self._fake_curl(tmp, 22)            # the health probe keeps failing
            codes = [self._run(tmp, "kb-graph-rebuild-check.sh").returncode for _ in range(3)]
            self.assertEqual(codes, [0, 0, 75])
            self.assertEqual((tmp / "maint" / "graph-rebuild.defers").read_text(encoding="utf-8").strip(), "3")
            codes = [self._run(tmp, "kb-cleanup.sh", "parse-assets-gc").returncode for _ in range(3)]
            self.assertEqual(codes, [0, 0, 75])
            self.assertEqual(self._run(tmp, "kb-cleanup.sh", "weekly").returncode, 1)
            self.assertFalse((tmp / "maint" / "cleanup-weekly.defers").exists())
            # Qdrant is back: straight into the CLI, and one successful run resets the count
            self._fake_curl(tmp, 0)
            self._fake_repo(tmp, 0)
            self.assertEqual(self._run(tmp, "kb-graph-rebuild-check.sh").returncode, 0)
            self.assertFalse((tmp / "maint" / "graph-rebuild.defers").exists())
            self.assertEqual(self._run(tmp, "kb-cleanup.sh", "parse-assets-gc").returncode, 0)
            self.assertFalse((tmp / "maint" / "cleanup-parse-assets-gc.defers").exists())
        for name in ("carrel-graph-rebuild.service", "carrel-qdrant-gc.service"):
            unit = (self.REPO / "deployment" / "systemd" / name).read_text(encoding="utf-8")
            self.assertNotIn("\nExecStartPre=", unit, f"{name} 还在 ExecStartPre 里自己 exit 75")   # mentioning the word in a comment does not count
        lib = (self.REPO / "scripts" / "lib" / "kb-maint-defer.sh").read_text(encoding="utf-8")
        self.assertIn("maint_wait_qdrant_or_defer()", lib)
        self.assertIn('maint_wait_qdrant_or_defer "graph-rebuild"', (self.REPO / "scripts" / "kb-graph-rebuild-check.sh").read_text(encoding="utf-8"))
        self.assertIn('maint_wait_qdrant_or_defer "cleanup-$COMMAND"', (self.REPO / "scripts" / "kb-cleanup.sh").read_text(encoding="utf-8"))


class BusyGateTests(unittest.TestCase):
    """The busy check of maintenance jobs: an orphaned lock must not block forever, a lock held for seconds is not
    worth yielding 15 minutes for, and a "blocked" result one level down has to be passed up."""

    def _settings(self, tmp: str):
        root = Path(tmp)
        (root / "rt" / "state").mkdir(parents=True)
        return SimpleNamespace(state_db=root / "s.db", runtime_dir=root / "rt", mineru_url="http://m")

    def _lock(self, settings, name: str, *, age: float = 0.0) -> Path:
        lock = settings.runtime_dir / "state" / f"{name}.lock.d"
        lock.mkdir()
        if age:
            os.utime(lock, (time.time() - age, time.time() - age))
        return lock

    def _quiet(self):
        from unittest import mock

        from kb_pipeline import maintenance

        return mock.patch.multiple(maintenance, _pgrep=lambda pattern: False, _mineru_busy=lambda url: None)

    def test_orphaned_mirror_lock_is_ignored_like_the_scan_does(self) -> None:
        from kb_pipeline import maintenance

        with tempfile.TemporaryDirectory() as tmp, self._quiet():
            settings = self._settings(tmp)
            lock = self._lock(settings, "mirror_sync", age=30)
            busy, reasons = maintenance.service_busy(settings)
            self.assertTrue(busy)                                        # a push is in progress: yield
            self.assertEqual(reasons, [f"lock exists: {lock}"])
            os.utime(lock, (time.time() - 7200, time.time() - 7200))
            self.assertEqual(maintenance.service_busy(settings), (False, []))   # untouched for two hours: an orphan, not busy
            self.assertTrue(lock.exists())                               # ignored, not removed: reclaiming it is up to the pushing side
            with patch.dict(os.environ, {"KB_MIRROR_LOCK_STALE_SECONDS": "86400"}):
                self.assertTrue(maintenance.service_busy(settings)[0])   # the threshold reads the same variable as the scan script

    def test_locks_that_only_exist_on_the_pushing_machine_are_not_checked(self) -> None:
        from kb_pipeline import maintenance

        with tempfile.TemporaryDirectory() as tmp, self._quiet():
            settings = self._settings(tmp)
            self._lock(settings, "source_prefetch")                      # a step of the machine that pushes the mirror
            self.assertEqual(maintenance.service_busy(settings), (False, []))
            worker = self._lock(settings, "kb_worker", age=7200)          # locks of this machine do not age out
            self.assertEqual(maintenance.service_busy(settings), (True, [f"lock exists: {worker}"]))

    def test_short_locks_are_waited_out(self) -> None:
        from unittest import mock

        from kb_pipeline import maintenance

        with tempfile.TemporaryDirectory() as tmp, self._quiet():
            settings = self._settings(tmp)
            with mock.patch.object(maintenance.time, "sleep", side_effect=AssertionError("not busy, so no waiting")):
                maintenance.wait_out_short_locks(settings, timeout=5, poll=0.01)
            scan = self._lock(settings, "kb_scan")
            polls = []

            def release_after_three(seconds: float) -> None:
                polls.append(seconds)
                if len(polls) == 3:
                    scan.rmdir()                                          # the scan is done and releases its lock

            with mock.patch.object(maintenance.time, "sleep", release_after_three):
                maintenance.wait_out_short_locks(settings, timeout=5, poll=0.01)
            self.assertEqual(len(polls), 3)
            self.assertEqual(maintenance.service_busy(settings), (False, []))

    def test_real_work_is_not_waited_for(self) -> None:
        from unittest import mock

        from kb_pipeline import maintenance

        with tempfile.TemporaryDirectory() as tmp, self._quiet():
            settings = self._settings(tmp)
            self._lock(settings, "kb_scan")
            with mock.patch.object(maintenance.time, "sleep", side_effect=AssertionError("must not wait")):
                self._lock(settings, "kb_worker")                         # parsing is running: return at once, the caller yields
                maintenance.wait_out_short_locks(settings, timeout=5, poll=0.01)
                (settings.runtime_dir / "state" / "kb_worker.lock.d").rmdir()
                with mock.patch.object(maintenance, "_mineru_busy", return_value="mineru busy queued=1 processing=0"):
                    maintenance.wait_out_short_locks(settings, timeout=5, poll=0.01)
            slept = []
            with mock.patch.object(maintenance.time, "sleep", slept.append):
                maintenance.wait_out_short_locks(settings, timeout=0.05, poll=0.01)   # the lock is never released: wait up to the limit
            self.assertTrue(slept)
            self.assertTrue(maintenance.service_busy(settings)[0])

    def _cleanup_exit_code(self, *, outer: dict, nested: dict) -> int:
        import contextlib
        import io
        from unittest import mock

        from kb_pipeline import cli

        args = SimpleNamespace(env_file=None, cleanup_command="parse-assets-gc", retention_days=7, dry_run=False)
        settings = SimpleNamespace(qdrant_inactive_retention_days=7)
        with mock.patch.object(cli, "load_settings", return_value=settings), \
                mock.patch.object(cli, "wait_out_short_locks") as waited, \
                mock.patch.object(cli, "parse_assets_gc", return_value=dict(outer)), \
                mock.patch.object(cli, "kb_sources_gc", return_value=dict(nested)) as nested_gc, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = cli.cmd_cleanup(args)
        self.nested_calls = nested_gc.call_count
        self.waits = waited.call_count
        return code

    def test_a_blocked_nested_step_asks_for_a_retry(self) -> None:
        done = {"skipped": False, "errors": []}
        self.assertEqual(self._cleanup_exit_code(outer=done, nested={"dropped": [], "skipped": [], "errors": []}), 0)
        self.assertEqual((self.nested_calls, self.waits), (1, 2))          # one wait for short locks before each step
        blocked = {"skipped": True, "reason": "service busy", "busy_reasons": ["lock exists: kb_scan.lock.d"]}
        self.assertEqual(self._cleanup_exit_code(outer=done, nested=blocked), 75)
        self.assertEqual(self._cleanup_exit_code(outer=blocked, nested=blocked), 75)
        self.assertEqual(self.nested_calls, 0)                             # when the outer step yields, the nested one does not run
        self.assertEqual(self._cleanup_exit_code(outer={"skipped": False, "errors": ["x"]}, nested=blocked), 1)


class SystemdUnitInstallTests(unittest.TestCase):
    """The installed units are **independent copies** (not symlinks); editing the repo files tells systemd nothing.

    Between intent (git) and effect (systemd) sit two steps a human has to remember, "cp + daemon-reload";
    forgetting them raises no error, the behaviour just differs from what you think it is; that is configuration
    drift. The script turns the sync into one idempotent command, and this test makes forgetting to sync
    visible.
    """

    UNIT_DIR = Path("deployment/systemd")

    def _installed_dir(self) -> Path:
        base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
        return Path(base) / "systemd" / "user"

    def test_installed_units_match_the_repo(self) -> None:
        repo = Path(__file__).resolve().parents[2] / self.UNIT_DIR
        units = sorted(p for p in repo.iterdir() if p.suffix in {".service", ".timer"})
        self.assertTrue(units, "仓库里应当有 unit 文件")
        installed = self._installed_dir()
        if not any((installed / u.name).exists() for u in units):
            self.skipTest(f"这台机器没装这些 unit（{installed}）")
        drifted = []
        for unit in units:
            target = installed / unit.name
            if not target.exists():
                drifted.append(f"{unit.name}: 未安装")
            elif target.read_text(encoding="utf-8") != _render_unit(unit):
                drifted.append(f"{unit.name}: 内容不一致")
        self.assertEqual(drifted, [], "跑 ./scripts/install-systemd.sh 同步:\n" + "\n".join(drifted))

    def test_install_script_is_idempotent_and_never_restarts(self) -> None:
        """Restarting a running worker loses the parse progress of the current file, so the install script only
        reloads and never restarts. A reload only makes systemd re-read the configuration; it does not touch
        the processes."""
        script = _repo_file("scripts/install-systemd.sh")
        self.assertIn("daemon-reload", script)
        self.assertNotIn("systemctl --user restart", script)
        self.assertIn("cmp -s", script)          # no copy when the content is unchanged
        self.assertIn("--check", script)         # there is a read-only drift-check mode


class WorkerScriptAndTimerTests(unittest.TestCase):
    def test_worker_script_tolerates_state_db_locks(self) -> None:
        """R1 / R2: a state database lock timeout is not a crash; wait a while and retry. B11: the graph-rebuild
        unit loads the env file into its environment."""
        script = _repo_file("scripts/kb-pipeline-worker-once.sh")
        self.assertIn("database is locked", script)
        self.assertIn("MAX_LOCK_WAITS", script)
        self.assertIn("lock_waits=0", script)
        unit = _repo_file("deployment/systemd/carrel-graph-rebuild.service")
        self.assertIn("EnvironmentFile=-__CARREL_HOME__/config/knowledge-base.env", unit)

    def test_graph_check_runs_every_two_hours(self) -> None:
        """2026-09-06: the graph maintenance check moved from every 30 minutes to every 2 hours; the deferral
        limit went from 48 rounds to 12 accordingly (still one day)."""
        timer = _repo_file("deployment/systemd/carrel-graph-rebuild.timer")
        self.assertIn("OnUnitActiveSec=2h", timer)
        self.assertNotIn("OnCalendar", timer)                 # 2026-09-09 switched to relative time: no wall clock, no time zone
        self.assertNotIn("07/30", timer)
        self.assertIn('KB_MAINT_DEFER_LIMIT:-12}', _repo_file("scripts/kb-graph-rebuild-check.sh"))
        for rel in ("app/kb_server/static/app.js", "app/kb_server/static/index.html"):
            text = _repo_file(rel)
            self.assertIn("每 2 小时", text, rel)
            self.assertNotIn("每 30 分钟", text, rel)


class DeploymentFileTests(unittest.TestCase):
    """The deployment files cannot be run on a development machine; what can be checked is that they agree with
    each other and that no key is misspelled."""

    UNITS = Path(__file__).resolve().parents[2] / "deployment" / "systemd"
    # systemd only logs a warning for a key it does not know and starts the unit anyway: a misspelling goes unnoticed
    DIRECTIVES = {
        "Unit": {"Description", "After", "StartLimitIntervalSec", "StartLimitBurst"},
        "Service": {"Type", "WorkingDirectory", "Environment", "EnvironmentFile", "ExecStart", "Restart", "RestartSec",
                    "TimeoutStartSec", "OOMScoreAdjust"},
        "Timer": {"OnActiveSec", "OnUnitActiveSec", "AccuracySec"},
        "Install": {"WantedBy"},
    }

    def _unit(self, path: Path) -> dict[str, list[tuple[str, str]]]:
        sections: dict[str, list[tuple[str, str]]] = {}
        current = ""
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                current = line[1:-1]
                sections.setdefault(current, [])
                continue
            key, sep, value = line.partition("=")
            self.assertTrue(sep and current, f"{path.name}: cannot parse the line {raw!r}")
            sections[current].append((key, value))
        return sections

    def test_unit_files_only_use_directives_systemd_knows(self) -> None:
        units = sorted(p for p in self.UNITS.iterdir() if p.suffix in {".service", ".timer"})
        self.assertGreaterEqual(len(units), 14)
        for path in units:
            for section, pairs in self._unit(path).items():
                self.assertIn(section, self.DIRECTIVES, path.name)
                for key, _ in pairs:
                    self.assertIn(key, self.DIRECTIVES[section], f"{path.name} [{section}]")

    def test_always_on_services_outlive_batch_units_under_memory_pressure(self) -> None:
        """The always-on services are set to 100, the lowest a user unit may use, and the batch units to 200: under
        memory pressure the ones killed first are those that retry and can resume."""
        always_on = []
        for path in sorted(self.UNITS.glob("*.service")):
            service = dict(self._unit(path)["Service"])
            if service.get("Restart") == "always":
                always_on.append(path.name)
                self.assertEqual(service.get("OOMScoreAdjust"), "100", path.name)
            else:
                self.assertEqual((service.get("Type"), service.get("OOMScoreAdjust")), ("oneshot", "200"), path.name)
        self.assertEqual(always_on, ["carrel-search.service", "carrel-web.service"])

    @staticmethod
    def _pairs(text: str) -> dict[str, str]:
        rows = (line.split("=", 1) for line in text.splitlines() if "=" in line and not line.lstrip().startswith("#"))
        return {key.strip(): value.strip() for key, value in rows}

    def test_compose_defaults_match_the_example_env(self) -> None:
        """The default in each ${KEY:-default} of the compose file is the value in .env.example, and compose uses
        every key of the example. If each side is edited on its own, a deployment missing a key silently runs on
        another set of parameters."""
        compose = _repo_file("deployment/compose/docker-compose.yml")
        example = self._pairs(_repo_file("deployment/compose/.env.example"))
        defaults = re.findall(r"\$\{(\w+):-([^}]*)\}", compose)
        self.assertGreater(len(defaults), 80)
        self.assertEqual({key: value for key, value in defaults if example.get(key, value) != value}, {})
        # COMPOSE_FILE / COMPOSE_PROFILES are read by docker compose itself, not through ${...}. The README of this
        # compose directory reports no GPU share total to compare against: the parser's share is derived from the
        # GPU at start-up (MINERU_GPU_MEMORY_UTILIZATION is empty in the example)
        self.assertEqual(set(example) - set(re.findall(r"\$\{(\w+)", compose)) - {"COMPOSE_FILE", "COMPOSE_PROFILES"}, set())

    def test_every_key_in_the_pipeline_example_env_is_read(self) -> None:
        """A key left in the example that nothing reads (the removed mass-delete guard once left two) makes whoever
        reads the configuration believe it still has an effect."""
        root = Path(__file__).resolve().parents[2]
        code = "".join(path.read_text(encoding="utf-8")
                       for base in ("app/kb_pipeline", "app/kb_server", "app/kb_search", "scripts")
                       for path in sorted((root / base).rglob("*")) if path.suffix in {".py", ".sh"})
        unread = {key for key in self._pairs(_repo_file("config/knowledge-base.env.example")) if key not in code}
        self.assertEqual(unread, {"NO_PROXY", "no_proxy"})        # these two are read by requests / curl themselves

    def test_every_container_rotates_its_log(self) -> None:
        compose = _repo_file("deployment/compose/docker-compose.yml")
        head, body = compose.split("\nservices:\n", 1)
        common = head.split("x-vllm-common:", 1)[1].split("\nx-", 1)[0]
        self.assertIn('max-size: "50m"', head)
        self.assertIn("logging: *default-logging", common)
        blocks = re.split(r"(?m)^  ([\w.-]+):\n", body.split("\nnetworks:\n", 1)[0])
        services = dict(zip(blocks[1::2], blocks[2::2]))
        self.assertEqual(len(services), 9)
        for name, block in services.items():
            self.assertTrue("logging: *default-logging" in block or "<<: *vllm-common" in block, name)

    def test_search_model_servers_are_batch_invariant(self) -> None:
        """The three model servers a search calls (embedding, reranker, vl-embedding) run in batch-invariant
        mode, so the same query ranks the same way every time; the generative ones (vlm, the parser) do not."""
        compose = _repo_file("deployment/compose/docker-compose.yml")
        head, body = compose.split("\nservices:\n", 1)
        anchor = head.split("x-vllm-deterministic: &vllm-deterministic", 1)[1].split("\nx-", 1)[0]
        self.assertIn("<<: *vllm-environment", anchor)
        self.assertIn('VLLM_BATCH_INVARIANT: "${VLLM_BATCH_INVARIANT:-1}"', anchor)
        blocks = re.split(r"(?m)^  ([\w.-]+):\n", body.split("\nnetworks:\n", 1)[0])
        services = dict(zip(blocks[1::2], blocks[2::2]))
        self.assertEqual({name for name, block in services.items() if "*vllm-deterministic" in block},
                         {"embedding", "reranker", "vl-embedding"})


class DependencyDeclarationTests(unittest.TestCase):
    """Every third-party package the code imports must be declared in pyproject, and every declared one must be
    used. Used but not declared, a venv rebuilt by the documentation silently lacks a feature (YAML configurations
    yield no facts, the PDF missing-text check is skipped entirely); declared but unused only costs installation.
    PyMuPDF is an optional extra here (pdf-images, AGPL): its imports are guarded and the features degrade without
    it, so it is declared there and never among the required dependencies."""

    APP = Path(__file__).resolve().parents[1]
    PACKAGES = ("kb_pipeline", "kb_server", "kb_search")
    # import names that differ from the distribution name
    DISTRIBUTION = {"yaml": "pyyaml", "pil": "pillow", "opensearchpy": "opensearch-py", "fitz": "pymupdf"}
    TRANSITIVE = {"numpy", "pydantic", "httpx"}   # brought in by qdrant-client / fastapi / openai, versions follow theirs
    NOT_IMPORTED = {"tree-sitter"}            # runtime of the grammar package; the code only imports tree_sitter_language_pack
    OPTIONAL = {"pymupdf"}                    # optional extras: guarded imports, the feature degrades without them

    def _imported(self) -> set[str]:
        import ast
        import sys

        names: set[str] = set()
        for package in self.PACKAGES:
            for path in (self.APP / package).rglob("*.py"):
                for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                    if isinstance(node, ast.Import):
                        names.update(alias.name.split(".")[0] for alias in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                        names.add(node.module.split(".")[0])
        names -= set(sys.stdlib_module_names) | set(self.PACKAGES)
        return {self.DISTRIBUTION.get(name.lower(), name.lower().replace("_", "-")) for name in names}

    def _declared(self, *, optional: bool = False) -> set[str]:
        import tomllib

        project = tomllib.loads((self.APP / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        deps = ([dep for extra in project.get("optional-dependencies", {}).values() for dep in extra] if optional
                else project["dependencies"])
        return {re.split(r"[<>=!~;\[ ]", dep, maxsplit=1)[0].lower().replace("_", "-") for dep in deps}

    def test_imports_and_declarations_agree(self) -> None:
        imported, declared, optional = self._imported(), self._declared(), self._declared(optional=True)
        self.assertEqual(imported - declared - optional - self.TRANSITIVE, set(), "used by the code, not declared in pyproject")
        self.assertEqual(declared - imported - self.NOT_IMPORTED, set(), "declared in pyproject, used by no code")
        self.assertEqual(optional, self.OPTIONAL)              # the optional extra is PyMuPDF ...
        self.assertEqual(optional & declared, set())           # ... never a required dependency ...
        self.assertLessEqual(optional, imported)               # ... and really used by the code

    def _pins(self, name: str) -> dict[str, str]:
        pins: dict[str, str] = {}
        for line in (self.APP / name).read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            m = re.fullmatch(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9_.!+-]+)", line.strip())
            self.assertIsNotNone(m, line)
            pins[m.group(1).lower().replace("_", "-")] = m.group(2)
        return pins

    def test_lock_files_pin_every_declared_dependency_inside_its_range(self) -> None:
        """requirements.lock is the exact set the suite was run against: every required dependency is in it with a
        version inside its declared range, pytest too, and nothing but name==version lines. The optional pdf-images
        extra (AGPL) has its own requirements-pdf-images.lock and never appears in the main one, so a plain install
        stays free of it. Every declared range carries an upper bound."""
        import tomllib

        from packaging.requirements import Requirement
        from packaging.version import Version

        project = tomllib.loads((self.APP / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        main, extra = self._pins("requirements.lock"), self._pins("requirements-pdf-images.lock")
        optional = [dep for group in project.get("optional-dependencies", {}).values() for dep in group]
        for dep, pins in [(dep, main) for dep in project["dependencies"]] + [(dep, extra) for dep in optional]:
            req = Requirement(dep)
            name = req.name.lower().replace("_", "-")
            self.assertIn(name, pins, dep)
            self.assertTrue(req.specifier.contains(Version(pins[name]), prereleases=True), f"{dep} is locked at {pins[name]}")
            self.assertTrue(any(op in ("<", "<=", "==", "~=") for op in (spec.operator for spec in req.specifier)), f"{dep} has no upper bound")
        self.assertIn("pytest", main)
        self.assertEqual({Requirement(dep).name.lower() for dep in optional} & set(main), set())   # the AGPL extra stays out of the main lock


class OpsFixRegressionTests(_CodexAudit20260906TestsSupport, _CodexFinalTestsSupport, unittest.TestCase):
    """Regressions for problems found by successive re-reviews, health checks and audits; each test's
    docstring records where it came from and the symptom observed at the time."""

    def test_log_rotation_keeps_lines_appended_during_gzip(self) -> None:  # issue 23
        import gzip as gzip_module

        from kb_pipeline import maintenance

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "kb.log"; dst = Path(tmp) / "kb.log.gz"
            src.write_text("OLD-LINE\n", encoding="utf-8")
            real_open = gzip_module.open
            appended = {"done": False}

            def open_and_append(*args, **kwargs):
                handle = real_open(*args, **kwargs)
                if not appended["done"]:
                    appended["done"] = True
                    with src.open("a", encoding="utf-8") as live:
                        live.write("APPENDED-DURING-ROTATION\n")
                return handle

            with patch.object(maintenance.gzip, "open", open_and_append):
                maintenance.gzip_file_and_truncate(src, dst)
            self.assertEqual(src.read_text(encoding="utf-8"), "APPENDED-DURING-ROTATION\n")
            with gzip_module.open(dst, "rt", encoding="utf-8") as archived:
                self.assertEqual(archived.read(), "OLD-LINE\n")

    def test_unenrolled_kb_is_hard_deleted_when_retention_expires(self) -> None:
        """S3: a knowledge base closed from the console still has its directory, so the old GC skipped it
        forever and the "deleted automatically after 7 days" promised by the dialog never happened. The old
        semantics of directory-missing deactivation must stay unchanged."""
        from unittest import mock

        from kb_pipeline import discovery, maintenance

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); mirror = root / "mirror"
            (mirror / "关掉的库").mkdir(parents=True)
            (mirror / "还在的库").mkdir()
            state = root / "s.db"; db.init_db(state)
            with db.connect(state) as con:
                a, _ = discovery.enroll(con, mirror, "关掉的库")
                b, _ = discovery.enroll(con, mirror, "还在的库")
                discovery.mark_inactive(con, a.kb_id, reason="unenrolled")        # closed from the console
                discovery.mark_inactive(con, b.kb_id, reason="directory_missing")  # the directory went missing once
                con.execute("UPDATE kb_sources SET inactive_at = ?", (int(time.time()) - 30 * 86400,))
                con.commit()
            stub = SimpleNamespace(state_db=state, mirror_root=mirror,
                                   qdrant_url="http://q", qdrant_api_key="",
                                   opensearch_url="http://o", cache_dir=root / "cache",
                                   graph_work_dir=root / "gw")
            with mock.patch.object(maintenance, "service_busy", return_value=(False, [])), \
                    mock.patch.object(maintenance, "_hard_delete_kb",
                                      side_effect=lambda *a, **k: {"forgotten": True}) as hard_delete:
                result = maintenance.kb_sources_gc(stub, retention_days=7)
            dropped = {entry["kb_id"] for entry in result["dropped"]}
            self.assertEqual(dropped, {a.kb_id})            # the closed KB is deleted once retention expires
            self.assertEqual(hard_delete.call_count, 1)
            # a directory-missing deactivation whose directory exists is still skipped (the scan reactivates it)
            self.assertNotIn(b.kb_id, dropped)

    def test_failed_external_delete_keeps_the_registry_row(self) -> None:
        """S4: the registry row is the only lead back to the collection / Neo4j projection. If any side fails
        to delete, the row must stay, marked delete_failed, otherwise nothing can ever find the leftovers."""
        from unittest import mock

        from kb_pipeline import discovery, maintenance

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); mirror = root / "mirror"; (mirror / "库X").mkdir(parents=True)
            state = root / "s.db"; db.init_db(state)
            with db.connect(state) as con:
                src, _ = discovery.enroll(con, mirror, "库X")
                con.commit()
            stub = SimpleNamespace(state_db=state, mirror_root=mirror,
                                   qdrant_url="http://q", qdrant_api_key="",
                                   opensearch_url="http://o", cache_dir=root / "cache",
                                   graph_work_dir=root / "gw", neo4j_password="")
            boom = mock.Mock(side_effect=RuntimeError("qdrant unreachable"))
            with db.connect(state) as con:
                errors: list[str] = []
                with mock.patch("kb_pipeline.vector.qdrant.client", return_value=mock.Mock()), \
                        mock.patch("kb_pipeline.vector.qdrant.delete_collection", boom), \
                        mock.patch.object(maintenance, "_drop_graph_data", return_value={}), \
                        mock.patch("kb_pipeline.search_fts.delete_collection", return_value={}):
                    entry = maintenance._hard_delete_kb(
                        stub, con, kb_id=src.kb_id, collection=src.collection, errors=errors)
                con.commit()
                row = con.execute("SELECT status, inactive_reason FROM kb_sources WHERE kb_id=?",
                                  (src.kb_id,)).fetchone()
            self.assertTrue(errors)                          # the failure is recorded
            self.assertFalse(entry["forgotten"])             # the row was not forgotten
            self.assertIsNotNone(row)                        # the row is still there; the next GC round can retry
            self.assertEqual(str(row["inactive_reason"]), "delete_failed")

    def test_side_notes_readme_and_retention(self) -> None:
        # The schedule table moved from the README into the operations guide (2026-09-28 restructure).
        self.assertNotIn("每 30 分钟", _repo_file("docs/operations.zh-CN.md"))
        self.assertIn("每 2 小时", _repo_file("docs/operations.zh-CN.md"))

    def test_stop_path_recognises_kb_launched_processes_and_separates_dead_from_foreign(self) -> None:
        from unittest import mock

        from kb_pipeline import maintenance
        from kb_pipeline.utils import looks_like_kb_process, looks_like_worker_command

        self.assertTrue(looks_like_kb_process("/x/.venv/bin/python /x/.venv/bin/kb worker --once --source kb_002"))
        self.assertTrue(looks_like_kb_process("/x/.venv/bin/python -m kb_pipeline --env-file e graph build --source kb_001"))
        self.assertTrue(looks_like_kb_process("kb graph build"))
        self.assertFalse(looks_like_kb_process("/usr/bin/python3 -m http.server"))
        self.assertFalse(looks_like_kb_process("bash /x/scripts/kb-pipeline-worker-once.sh"))
        self.assertTrue(looks_like_worker_command("/x/.venv/bin/kb worker --once"))
        self.assertFalse(looks_like_worker_command("/x/.venv/bin/kb scan --source kb_002"))
        with mock.patch.object(maintenance.Path, "read_bytes", return_value=b"/x/.venv/bin/kb\x00worker\x00--once\x00"):
            self.assertTrue(maintenance._process_matches(4242, "kb_pipeline"))
        with mock.patch.object(maintenance.Path, "read_bytes", return_value=b"/usr/bin/python3\x00-m\x00http.server\x00"):
            self.assertFalse(maintenance._process_matches(4242, "kb_pipeline"))
        # alive but not ours: left alone and not counted as stopped; only a missing pid counts as "confirmed dead"
        with mock.patch.object(maintenance, "_process_matches", return_value=False):
            self.assertEqual(maintenance._terminate_pid(os.getppid()), "not-ours")
            self.assertEqual(maintenance._terminate_pid(4194300), "already-gone")
        src = _repo_file("app/kb_pipeline/maintenance.py")
        self.assertIn('entry["stopped"] = (not remaining) or dead', src)

    def test_delete_failed_rows_are_retried_even_if_the_directory_is_still_there(self) -> None:
        from unittest import mock

        from kb_pipeline import discovery, maintenance

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); mirror = root / "mirror"
            (mirror / "删失败的库").mkdir(parents=True)
            state = root / "s.db"; db.init_db(state)
            with db.connect(state) as con:
                a, _ = discovery.enroll(con, mirror, "删失败的库")
                con.execute("UPDATE kb_sources SET status='inactive', inactive_reason='delete_failed', inactive_at=? WHERE kb_id=?",
                            (int(time.time()) - 30 * 86400, a.kb_id))
                con.commit()
            stub = SimpleNamespace(state_db=state, mirror_root=mirror, qdrant_url="http://q", qdrant_api_key="",
                                   opensearch_url="http://o", cache_dir=root / "cache", graph_work_dir=root / "gw")
            with mock.patch.object(maintenance, "service_busy", return_value=(False, [])), \
                    mock.patch.object(maintenance, "_hard_delete_kb", side_effect=lambda *a, **k: {"forgotten": True}) as hard_delete:
                result = maintenance.kb_sources_gc(stub, retention_days=7)
            self.assertEqual(hard_delete.call_count, 1)
            self.assertEqual({e["kb_id"] for e in result["dropped"]}, {a.kb_id})


class JobHistoryDryRunTests(unittest.TestCase):
    """2026-09-29 audit: cleanup parse-assets-gc --dry-run still deleted the jobs, failures and events older than
    30 days. A dry run counts and deletes nothing."""

    def test_dry_run_counts_and_deletes_nothing(self) -> None:
        import tempfile
        import time
        from pathlib import Path

        from kb_pipeline import db

        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"; db.init_db(state)
            old = int(time.time()) - 90 * 86400
            with db.connect(state) as con:
                for job_id, status, when in (("j-old", "done", old), ("j-new", "done", int(time.time())), ("j-run", "running", old)):
                    con.execute(
                        "INSERT INTO jobs(job_id, kb_id, collection, job_type, status, priority, "
                        "next_attempt_at, created_at, updated_at, finished_at) "
                        "VALUES(?, 'kb', 'c', 'parse', ?, 100, 0, 1, ?, ?)",
                        (job_id, status, when, when if status == "done" else None))
                con.commit()
                before = con.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
                planned = db.prune_job_history(con, retention_days=30, dry_run=True)
                self.assertEqual(con.execute("SELECT COUNT(*) FROM jobs").fetchone()[0], before)      # nothing was deleted
                self.assertEqual(planned["jobs"], 1)
                removed = db.prune_job_history(con, retention_days=30)
                self.assertEqual(removed["jobs"], planned["jobs"])                                  # the real run deletes what the dry run counted
                self.assertEqual({r[0] for r in con.execute("SELECT job_id FROM jobs")}, {"j-new", "j-run"})
        src = _repo_file("app/kb_pipeline/maintenance.py")
        self.assertIn('int(os.getenv("KB_JOB_HISTORY_DAYS", "30"))), dry_run=dry_run)', src)


class GraphGcSafetyNetTests(unittest.TestCase):
    """Nightly graph-gc: shares the build lock with builds and skips the whole round while one runs; bases without
    a graph are left alone."""

    def test_graph_gc_skips_while_a_build_holds_the_lock(self) -> None:
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace

        from kb_pipeline import maintenance
        from kb_pipeline.graph.lock import GraphBuildLock

        with tempfile.TemporaryDirectory() as tmp:
            settings = SimpleNamespace(runtime_dir=Path(tmp), state_db=Path(tmp) / "s.db", sources={},
                                       graph_gc_keep_versions=2, qdrant_url="http://127.0.0.1:1", qdrant_api_key=None)
            holder = GraphBuildLock(settings)
            holder.acquire()
            try:
                out = maintenance.graph_gc(settings, dry_run=True)
            finally:
                holder.release()
            self.assertEqual((out["skipped"], out["reason"]), (True, "graph build running"))
            self.assertEqual(out["yielded_to"], "graph_build")           # the wrapper script reports a yield, not a failure
            free = maintenance.graph_gc(settings, dry_run=True)          # no graph-enabled base: nothing to do, lock released
            self.assertEqual((free["skipped"], free["sources"]), (False, {}))
            self.assertEqual(free["keep_latest"], 2)

    def test_graph_gc_keeps_the_resumable_version_outside_the_quota(self) -> None:
        """Nightly GC: the newest paused / failed version is kept for a resume (not deleted, not counted); older
        half-finished versions are deleted outright."""
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest import mock

        from kb_pipeline import db, maintenance
        from kb_pipeline.graph import build as build_mod

        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"; db.init_db(state)
            with db.connect(state) as con:
                for bid, status, started in (("good", "done", 100), ("fail1", "failed", 200), ("paused", "cancelled", 300)):
                    con.execute(
                        "INSERT INTO graph_builds(graph_build_id, source_key, kb_id, source_collection, graph_version, "
                        "status, started_at, build_kind) VALUES(?, 'kb_1', 'kb_1', 'kb_1', ?, ?, ?, 'full')",
                        (bid, "v-" + bid, status, started))
            source = SimpleNamespace(kb_id="kb_1", collection="kb_1", graph_enabled=True)
            settings = SimpleNamespace(runtime_dir=Path(tmp), state_db=state, sources={"kb_1": source},
                                       graph_gc_keep_versions=1, qdrant_url="http://127.0.0.1:1", qdrant_api_key=None)
            seen: dict = {}

            def fake_gc(settings, source, **kwargs):
                seen.update(kwargs)
                return {"result": {}, "steps": [], "errors": {}}

            with mock.patch.object(build_mod, "gc_graph_versions", fake_gc), \
                    mock.patch("kb_pipeline.vector.qdrant.client", lambda *a, **k: object()), \
                    mock.patch("kb_pipeline.vector.qdrant.graph_alias_targets",
                               lambda q, collection: {"entity": "graph_1_entity__v-good"}), \
                    mock.patch("kb_pipeline.vector.qdrant.parse_graph_collection_name",
                               lambda name: {"graph_version": "v-good"}):
                out = maintenance.graph_gc(settings, dry_run=True)
            self.assertFalse(out["skipped"])
            self.assertEqual(seen["graph_version"], "v-good")
            self.assertEqual(seen["protect"], {"v-paused"})
            self.assertEqual(seen["discard"], {"v-fail1"})
            self.assertEqual(seen["keep_latest"], 1)
