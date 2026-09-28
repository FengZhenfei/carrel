"""State database, vector-store payloads and filters, config loading, directory enrollment and renames, and
restoring the projections."""
from __future__ import annotations

import socket
import sqlite3
import stat
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from kb_pipeline import db, search_fts
from kb_pipeline.models import ParsedBlock
from kb_pipeline.pipeline.detect_changes import detect_change
from kb_pipeline.pipeline.worker import _recover_or_refresh_running_jobs

from _support import _block, _local_file, _repo_file


class StateRegressionTests(unittest.TestCase):
    def test_state_db_permissions_are_repaired_on_connect(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.db"
            db.init_db(path)
            path.chmod(0o644)
            with db.connect(path):
                pass
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_keyword_doc_source_composition(self) -> None:
        payload = {
            "chunk_uid": "kb:1:v:p:b:0",
            "kb_id": "work_product",
            "doc_id": "work_product:1",
            "content_version": "v1",
            "chunk_index": "7",
            "source_path": "产品资料/x/a.pdf",
            "rel_path": "x/a.pdf",
            "filename": "a.pdf",
            "dir": "x",
            "title": "架构总览",
            "caption": "图 3.1",
            "section_path": ["3. 系统设计", "3.1 总体架构"],
            "text": "正文内容",
            "visual_summary": "三层架构",
            "visual_text": "接入层\n业务层",
            "visual_keywords": ["架构"],
            "page_idx": None,
        }
        source = search_fts.doc_source(payload)
        self.assertEqual(source["body"], "正文内容")
        self.assertIn("3. 系统设计 / 3.1 总体架构", source["title"])
        self.assertIn("图 3.1", source["title"])
        self.assertIn("接入层", source["visual"])
        self.assertEqual(source["chunk_index"], 7)
        self.assertEqual(source["section_path"], ["3. 系统设计", "3.1 总体架构"])
        # None / empty values must be dropped so "dynamic": "strict" stays happy
        self.assertNotIn("page_idx", source)

    def test_sync_doc_deletes_then_reindexes_through_one_bulk(self) -> None:
        calls: dict[str, object] = {}

        class FakeIndices:
            def exists(self, index):
                return True

            def refresh(self, index):
                calls["refreshed"] = index

        class FakeOS:
            indices = FakeIndices()

            def delete_by_query(self, index, body, params):
                calls["delete_index"] = index
                calls["delete_query"] = body
                calls["delete_params"] = params
                return {"deleted": 3}

        records = [
            SimpleNamespace(id="p-1", payload={"doc_id": "kb:1", "text": "one", "filename": "a.md"}),
            SimpleNamespace(id="p-2", payload={"doc_id": "kb:1", "text": "two", "filename": "a.md"}),
        ]
        with patch.object(search_fts, "client", return_value=FakeOS()), patch.object(
            search_fts, "_collection_exists", return_value=True
        ), patch.object(search_fts, "_scroll_active", return_value=iter(records)), patch.object(
            search_fts, "_bulk_index", side_effect=lambda _c, actions: calls.setdefault("actions", actions) and len(actions)
        ):
            result = search_fts.sync_doc_from_qdrant(
                url="http://fake:9200", qdrant=object(), collection="kb_product", doc_id="kb:1"
            )
        self.assertEqual(result["deleted_rows"], 3)
        self.assertEqual(result["inserted_rows"], 2)
        self.assertEqual(calls["delete_query"], {"query": {"term": {"doc_id": "kb:1"}}})
        self.assertEqual(calls["delete_params"], {"refresh": "false", "conflicts": "proceed"})   # health check D6: refresh once, only after the bulk
        actions = calls["actions"]
        self.assertEqual([a["_id"] for a in actions], ["p-1", "p-2"])
        self.assertTrue(all(a["_index"] == "kb_product" for a in actions))
        self.assertEqual(calls["refreshed"], "kb_product")

    def test_delete_collection_only_touches_target_index(self) -> None:
        deleted: list[str] = []

        class FakeIndices:
            def exists(self, index):
                return index == "kb_product"

            def delete(self, index):
                deleted.append(index)

        class FakeOS:
            indices = FakeIndices()

            def count(self, index):
                return {"count": 5}

        with patch.object(search_fts, "client", return_value=FakeOS()):
            result = search_fts.delete_collection("http://fake:9200", collection="kb_product")
            missing = search_fts.delete_collection("http://fake:9200", collection="kb_project")
        self.assertEqual(result["deleted_rows"], 5)
        self.assertEqual(deleted, ["kb_product"])
        self.assertFalse(missing["exists"])

    def test_dead_local_worker_job_is_retried_with_backoff_then_failed(self) -> None:
        """S5: a running job left behind by a killed worker must count as a retry and back off, otherwise a
        document that crashes the interpreter is re-claimed first forever and starves the whole queue."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.db"
            db.init_db(path)
            dead = f"{socket.gethostname()}:99999999"
            with db.connect(path) as con:
                def put_running(job_id: str, retry_count: int) -> None:
                    con.execute(
                        """
                        INSERT INTO jobs(
                          job_id, kb_id, collection, job_type, status, priority,
                          retry_count, locked_by, locked_until, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, 'running', 100, ?, ?, ?, 1, 1)
                        """,
                        (job_id, "kb", "kb_test", "parse", retry_count, dead, 9999999999),
                    )

                put_running("job-1", 0)
                released = _recover_or_refresh_running_jobs(
                    con, lease_seconds=3600, max_retries=2, retry_delay_seconds=300)
                row = con.execute("SELECT status, locked_by, retry_count, next_attempt_at FROM jobs WHERE job_id='job-1'").fetchone()
                self.assertEqual(released, ["job-1:retry"])
                self.assertEqual(row["status"], "retry")          # no longer re-queued immediately
                self.assertEqual(int(row["retry_count"]), 1)      # one attempt counted
                self.assertGreater(int(row["next_attempt_at"]), int(time.time()))  # backoff in effect
                self.assertIsNone(row["locked_by"])
                # the crash leaves a trace so the console / troubleshooting can see the cause
                self.assertEqual(
                    int(con.execute("SELECT COUNT(*) FROM failures WHERE stage='worker-crash'").fetchone()[0]), 1)

                # past the limit it is no longer re-queued; it stops at failed and the jobs behind it get through
                put_running("job-2", 2)
                released = _recover_or_refresh_running_jobs(
                    con, lease_seconds=3600, max_retries=2, retry_delay_seconds=300)
                row2 = con.execute("SELECT status FROM jobs WHERE job_id='job-2'").fetchone()
                self.assertEqual(released, ["job-2:failed"])
                self.assertEqual(row2["status"], "failed")

    def test_cancelled_job_resolves_its_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.db"
            db.init_db(path)
            with db.connect(path) as con:
                con.execute(
                    """
                    INSERT INTO jobs(job_id, kb_id, collection, job_type, status, created_at, updated_at)
                    VALUES ('job-1', 'kb', 'kb_test', 'parse', 'queued', 1, 1)
                    """
                )
                con.execute(
                    """
                    INSERT INTO failures(failure_id, job_id, stage, error_type, error_message, created_at)
                    VALUES ('failure-1', 'job-1', 'worker', 'Error', 'old', 1)
                    """
                )
                db.cancel_job(con, "job-1", "cancelled")
                resolved = con.execute(
                    "SELECT resolved_at FROM failures WHERE failure_id = 'failure-1'"
                ).fetchone()["resolved_at"]
            self.assertIsNotNone(resolved)


class PayloadVisualRefTests(unittest.TestCase):
    def _payload(self, block_type, visual_ref="/cache/parse/k/1/v/mineru/images/x.jpg"):
        from kb_pipeline.models import ParsedBlock, UnifiedChunk
        from kb_pipeline.pipeline.parse_job import _payload_for_chunk

        settings = SimpleNamespace(cache_dir=Path("/cache"), embedding_model_id="m")
        row = {"kb_id": "k", "file_key": 1, "source_path": "r/a.pdf", "rel_path": "a.pdf",
               "filename": "a.pdf", "dir": "", "content_version": "v", "size": 1, "mtime": 1,
               "mime_type": "application/pdf"}
        block = ParsedBlock(parser="mineru", parser_profile="p", doc_type="pdf",
                            block_type=block_type, text="<table>x</table>", block_id="b",
                            visual_ref=visual_ref)
        chunk = UnifiedChunk(chunk_uid="u", chunk_index=0, text="t", block=block, token_count=1)
        return _payload_for_chunk(settings, row, chunk, 1)

    def test_payload_carries_the_page_range_of_merged_blocks(self) -> None:
        """Codex review F09: a table merged across pages only recorded its first page, so a value on page 12
        pointed to page 11 as its source. page_start / page_end from metadata go into the payload; for a single
        page both equal page_idx; the estimated-reading conflict records are carried along too."""
        from kb_pipeline.models import ParsedBlock, UnifiedChunk
        from kb_pipeline.pipeline.parse_job import _payload_for_chunk

        settings = SimpleNamespace(cache_dir=Path("/cache"), embedding_model_id="m")
        row = {"kb_id": "k", "file_key": 1, "source_path": "r/a.pdf", "rel_path": "a.pdf",
               "filename": "a.pdf", "dir": "", "content_version": "v", "size": 1, "mtime": 1, "mime_type": "application/pdf"}
        merged = ParsedBlock(parser="mineru", parser_profile="p", doc_type="pdf", block_type="table", text="| a |", block_id="t",
                             page_idx=11, metadata={"page_start": 11, "page_end": 12})
        p = _payload_for_chunk(settings, row, UnifiedChunk(chunk_uid="u", chunk_index=0, text="t", block=merged, token_count=1), 1)
        self.assertEqual((p["page_idx"], p["page_start"], p["page_end"]), (11, 11, 12))
        single = ParsedBlock(parser="mineru", parser_profile="p", doc_type="pdf", block_type="text", text="x", block_id="s", page_idx=3,
                             metadata={"visual_value_conflicts": [{"label": "综合风险指数", "text_value": "34", "model_value": "31"}]})
        p2 = _payload_for_chunk(settings, row, UnifiedChunk(chunk_uid="u2", chunk_index=1, text="t", block=single, token_count=1), 2)
        self.assertEqual((p2["page_idx"], p2["page_start"], p2["page_end"]), (3, 3, 3))
        self.assertEqual(p2["visual_value_conflicts"][0]["text_value"], "34")
        svc = _repo_file("app/kb_server/service.py")
        self.assertEqual(svc.count('"page_end": pl.get("page_end", pl.get("page_idx"))'), 1)
        self.assertIn('"page_end": b.metadata.get("page_end", b.page_idx)', svc)
        js = _repo_file("app/kb_server/static/app.js")
        self.assertIn("function pageLabel(c)", js)
        self.assertIn("${pageLabel(c)}", js)

    def test_table_and_equation_crops_stay_out_of_the_payload(self) -> None:
        # MinerU returns a crop for every table/equation; they are provenance
        # only. Regression: they used to land in the payload and flag the chunk
        # as embedding_text_source=visual_ref, which was simply false.
        for bt in ("table", "equation", "text", "title"):
            p = self._payload(bt)
            self.assertNotIn("visual_ref", p, bt)
            self.assertNotIn("embedding_text_source", p, bt)

    def test_real_visual_blocks_keep_visual_ref(self) -> None:
        for bt in ("image", "chart", "slide"):
            p = self._payload(bt)
            self.assertEqual(p["visual_ref"], "parse/k/1/v/mineru/images/x.jpg", bt)
            self.assertEqual(p["embedding_text_source"], "visual_ref", bt)


class KBDiscoveryTests(unittest.TestCase):
    def test_numbered_ids_allocate_in_sequence_and_never_reuse(self) -> None:
        from kb_pipeline import discovery

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("产品资料", "半导体资料", "图书馆"):
                (root / name).mkdir()
            with db.connect(Path(tmp) / "s.db") as con:
                con.executescript(db.SCHEMA); discovery.init_schema(con)
                a, _ = discovery.enroll(con, root, "产品资料")
                b, _ = discovery.enroll(con, root, "半导体资料")
                self.assertEqual((a.kb_id, a.collection), ("kb_001", "kb_001"))
                self.assertEqual((b.kb_id, b.collection), ("kb_002", "kb_002"))
                # re-enrolling an existing row keeps its number
                again, outcome = discovery.enroll(con, root, "产品资料")
                self.assertEqual((again.kb_id, outcome), ("kb_001", "already"))
                # a forgotten (GC'd) directory starts over with a FRESH number
                discovery.forget(con, "kb_002")
                c, outcome = discovery.enroll(con, root, "半导体资料")
                self.assertEqual((c.kb_id, outcome), ("kb_003", "new"))
                # directory-name presets apply at first enrollment
                lib, _ = discovery.enroll(con, root, "图书馆")
                self.assertEqual((lib.kb_id, lib.max_tokens), ("kb_004", 800))
                self.assertEqual(lib.block_merge_tokens, 1600)   # derived: 2x max_tokens
                self.assertEqual(a.block_merge_tokens, 800)
                self.assertFalse(a.graph_enabled)  # enabling the graph is an explicit action; directory presets no longer switch graph build on

    def test_enrollment_gates_the_pipeline(self) -> None:
        from kb_pipeline import discovery

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "产品资料").mkdir(); (root / "新资料").mkdir()
            (root / ".hidden").mkdir(); (root / "~tmp").mkdir()
            (root / "stray.txt").write_text("x", encoding="utf-8")
            state = Path(tmp) / "s.db"
            # every top-level dir is DISCOVERABLE...
            self.assertEqual(discovery.discover_directories(root), ["产品资料", "新资料"])
            # ...but nothing is a pipeline source until enrolled
            self.assertEqual(discovery.enrolled_sources(state, root), {})
            with db.connect(state) as con:
                con.executescript(db.SCHEMA)
                discovery.init_schema(con)
                src, outcome = discovery.enroll(con, root, "产品资料")
                self.assertEqual((src.kb_id, src.collection, outcome), ("kb_001", "kb_001", "new"))
                self.assertEqual(src.max_tokens, 400)
                self.assertTrue(discovery.enroll(con, root, "产品资料")[1] == "already")
                with self.assertRaises(ValueError):
                    discovery.enroll(con, root, "不存在的目录")
            srcs = discovery.enrolled_sources(state, root)
            self.assertEqual(set(srcs), {"kb_001"})  # the un-enrolled directory (新资料) stays out until it is ticked

    def test_kb_lifecycle_rows(self) -> None:
        from kb_pipeline import discovery

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "X").mkdir()
            state = Path(tmp) / "s.db"
            with db.connect(state) as con:
                con.executescript(db.SCHEMA)
                discovery.init_schema(con)
                src, _ = discovery.enroll(con, root, "X")
                # unenroll: inactive with reason; the directory still existing
                # must NOT resurrect it (scan only touches, never flips)
                discovery.mark_inactive(con, src.kb_id, reason="unenrolled")
                discovery.touch_seen(con, src)
                row = discovery.known_sources(con)[0]
                self.assertEqual(row["status"], "inactive")
                self.assertEqual(row["inactive_reason"], "unenrolled")
                self.assertIsNotNone(row["inactive_at"])
                self.assertEqual(discovery.inactive_older_than(con, row["inactive_at"] - 1), [])
                con.commit()
                self.assertEqual(discovery.enrolled_sources(state, root), {})
                # explicit re-enroll inside retention flips back
                src2, outcome = discovery.enroll(con, root, "X")
                self.assertEqual(outcome, "reactivated")
                row = discovery.known_sources(con)[0]
                self.assertEqual(row["status"], "active"); self.assertIsNone(row["inactive_at"])
                # per-KB config round-trip with unknown-key rejection
                discovery.set_config(con, src.kb_id, {"max_tokens": 600, "vlm_prompt": "半导体图片重点识别时序"})
                cfg = discovery.get_config(con, src.kb_id)
                self.assertEqual(cfg["max_tokens"], 600)
                with self.assertRaises(ValueError):
                    discovery.set_config(con, src.kb_id, {"bogus": 1})
                rebuilt = discovery.source_from_row(root, discovery.known_sources(con)[0])
                self.assertEqual((rebuilt.kb_id, rebuilt.max_tokens, rebuilt.vlm_prompt),
                                 (src.kb_id, 600, "半导体图片重点识别时序"))


class VectorLayoutTests(unittest.TestCase):
    def test_layout_declares_text_and_visual_named_vectors(self) -> None:
        from kb_pipeline.vector.layout import TEXT_VECTOR, VISUAL_VECTOR, VectorLayout
        from kb_pipeline.vector.qdrant import vectors_config_for

        layout = VectorLayout(text_size=1024, visual_size=2048)
        self.assertEqual(layout.sizes(), {TEXT_VECTOR: 1024, VISUAL_VECTOR: 2048})
        cfg = vectors_config_for(layout)
        self.assertEqual({k: v.size for k, v in cfg.items()}, {"text": 1024, "visual": 2048})
        self.assertEqual({v.distance.value for v in cfg.values()}, {"Cosine"})

    def test_upsert_sends_named_vectors_and_requires_text(self) -> None:
        from kb_pipeline.vector.qdrant import upsert_chunks

        class FakeQdrant:
            def __init__(self) -> None:
                self.points = []

            def upsert(self, collection_name, points, wait):
                self.points.extend(points)

        q = FakeQdrant()
        upsert_chunks(
            q, "c", ["p1", "p2"],
            [{"text": [0.1] * 4}, {"text": [0.2] * 4, "visual": [0.3] * 8}],
            [{"a": 1}, {"a": 2}],
        )
        self.assertEqual([sorted(p.vector) for p in q.points], [["text"], ["text", "visual"]])
        self.assertEqual(len(q.points[1].vector["visual"]), 8)
        with self.assertRaises(ValueError):
            upsert_chunks(q, "c", ["p3"], [{"visual": [0.3] * 8}], [{}])

    def test_layout_validation_rejects_legacy_unnamed_collection(self) -> None:
        from kb_pipeline.vector.layout import VectorLayout
        from kb_pipeline.vector.qdrant import collection_vector_layout, validate_collection_layout

        class Legacy:
            def get_collection(self, collection_name):
                return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=1024))))

        class Named:
            def get_collection(self, collection_name):
                return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(
                    vectors={"text": SimpleNamespace(size=1024), "visual": SimpleNamespace(size=2048)})))

        layout = VectorLayout(1024, 2048)
        self.assertEqual(collection_vector_layout(Legacy(), "c"), {"": 1024})
        with self.assertRaises(RuntimeError) as ctx:
            validate_collection_layout(Legacy(), "c", layout)
        self.assertIn("unnamed", str(ctx.exception))
        self.assertEqual(validate_collection_layout(Named(), "c", layout), {"text": 1024, "visual": 2048})
        with self.assertRaises(RuntimeError):
            validate_collection_layout(Named(), "c", VectorLayout(1024, 4096))


class ConfigLoadingTests(unittest.TestCase):
    def test_bad_config_row_is_skipped_not_fatal(self) -> None:
        """One bad config_json row used to make load_settings raise, so web/scan/worker/GC all failed to start
        and every console endpoint returned 500."""
        from kb_pipeline import discovery

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "好库").mkdir(); (root / "坏库").mkdir()
            state = root / "s.db"; db.init_db(state)
            with db.connect(state) as con:
                good, _ = discovery.enroll(con, root, "好库")
                bad, _ = discovery.enroll(con, root, "坏库")
                con.execute("UPDATE kb_sources SET config_json=? WHERE kb_id=?",
                            ('{"max_tokens": "not-a-number"}', bad.kb_id))
                con.commit()
            sources = discovery.enrolled_sources(state, root)
            self.assertIn(good.kb_id, sources)     # the good KB works as usual
            self.assertNotIn(bad.kb_id, sources)   # the bad KB is skipped instead of taking everything down


class QdrantFilterTests(unittest.TestCase):
    """The delete filters had zero coverage before: one wrong condition would delete live points. These tests
    assert the structure of the filter itself; no real Qdrant connection is needed."""

    def test_inactive_gc_filter_requires_both_conditions(self) -> None:
        from kb_pipeline.vector.qdrant import inactive_older_than_filter

        f = inactive_older_than_filter(1000)
        conditions = {c.key: c for c in f.must}
        self.assertIn("is_active", conditions)
        self.assertIs(conditions["is_active"].match.value, False)   # only points that are already deactivated
        self.assertEqual(conditions["inactive_at"].range.lt, 1000.0)  # and past the retention period
        self.assertEqual(len(f.must), 2, "少一个条件就会误删活点")

    def test_active_doc_filter_is_scoped_to_one_document(self) -> None:
        from kb_pipeline.vector.qdrant import active_doc_filter

        f = active_doc_filter("kb_007", 42)
        conditions = {c.key: c for c in f.must}
        self.assertEqual(conditions["doc_id"].match.value, "kb_007:42")  # scoped to this one document
        self.assertIs(conditions["is_active"].match.value, True)

    def test_gc_helpers_are_noops_when_collection_is_absent(self) -> None:
        """A missing collection must not raise (GC still iterates over it after the KB was deleted)."""
        from unittest import mock

        from kb_pipeline.vector import qdrant as q_mod

        fake = mock.Mock()
        fake.collection_exists.return_value = False
        self.assertEqual(q_mod.delete_inactive_points_older_than(fake, "gone", 1000), 0)
        fake.count.assert_not_called()
        fake.delete.assert_not_called()


class RestoreProjectionTests(unittest.TestCase):
    """A file that was deleted and then came back takes the restore path (no re-parse); all three projections
    must come back together.

    reactivate_file_metadata only flips the Qdrant points. For a long time SQLite's chunks.status only had the
    mark_chunks_inactive half. That stale state used to be harmless (the graph build ledger came from a Qdrant
    scroll); once the ledger was taken from SQLite instead it became load-bearing: the file came back but could
    not enter the graph, and since the ledger did not change, the rebuild policy noticed nothing either.
    """

    def _settings(self, tmp: Path):
        return SimpleNamespace(
            runtime_dir=tmp / "runtime", state_db=tmp / "state.db",
            qdrant_url="http://q", qdrant_api_key="", opensearch_url="http://o",
            parse_enabled=True, parse_job_lease_seconds=600, metadata_job_lease_seconds=600,
            job_max_retries=2, job_retry_base_seconds=300, job_retry_max_seconds=3600,
            sources={},
        )

    def test_mark_chunks_active_only_touches_the_current_version(self) -> None:
        from kb_pipeline.models import UnifiedChunk

        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "state.db"
            db.init_db(state)
            with db.connect(state) as con:
                file = _local_file("a.pdf", checksum="c1")
                db.upsert_file(con, file, status="indexed")
                fid = db.file_id_for(file.kb_id, file.file_key)
                for version in ("v1", "v2"):
                    db.replace_chunks(
                        con, file_id=fid, collection=file.collection, content_version=version,
                        chunks=[UnifiedChunk(chunk_uid=f"u-{version}", chunk_index=0,
                                             text="x", block=_block("b", "x"))],
                        point_ids=[f"p-{version}"],
                    )
                db.mark_chunks_deleted(con, fid)          # delete path: only the active v2 becomes deleted
                touched = db.mark_chunks_active(con, fid, "v2")
                rows = dict(con.execute(
                    "SELECT chunk_uid, status FROM chunks WHERE file_id=?", (fid,)
                ).fetchall())
            self.assertEqual(touched, 1)
            self.assertEqual(rows["u-v2"], "active")
            self.assertEqual(rows["u-v1"], "inactive")   # the old version keeps waiting for GC

    def test_restore_only_revives_the_batch_that_was_active(self) -> None:
        """2026-09-06: the same content version was re-chunked (chunking rules / parameters changed) and the old
        batch stayed inactive. Closing and reopening the KB may only bring back the batch that was active at the
        time; flipping the whole version would resurrect every chunking ever done and inflate the chunk ledger
        sixfold."""
        from kb_pipeline.models import UnifiedChunk

        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "state.db"
            db.init_db(state)
            with db.connect(state) as con:
                file = _local_file("a.pdf", checksum="c1")
                db.upsert_file(con, file, status="indexed")
                fid = db.file_id_for(file.kb_id, file.file_key)
                for batch in ("old", "new"):        # the same version v1 chunked twice
                    db.replace_chunks(
                        con, file_id=fid, collection=file.collection, content_version="v1",
                        chunks=[UnifiedChunk(chunk_uid=f"{batch}-{i}", chunk_index=i, text="x", block=_block("b", "x"))
                                for i in range(2)],
                        point_ids=[f"p-{batch}-{i}" for i in range(2)],
                    )
                self.assertEqual(db.mark_chunks_deleted(con, fid), 2)          # only the active "new" batch is touched
                self.assertEqual(db.chunk_point_ids_to_restore(con, fid, "v1"), ["p-new-0", "p-new-1"])
                self.assertEqual(db.mark_chunks_active(con, fid, "v1"), 2)
                rows = dict(con.execute("SELECT chunk_uid, status FROM chunks WHERE file_id=?", (fid,)).fetchall())
            self.assertEqual({k: v for k, v in rows.items() if k.startswith("new")}, {"new-0": "active", "new-1": "active"})
            self.assertEqual({k: v for k, v in rows.items() if k.startswith("old")}, {"old-0": "inactive", "old-1": "inactive"})

    def test_reactivate_uses_explicit_point_ids(self) -> None:
        """Likewise on the Qdrant side: flip back the point ids given by SQLite, not a whole batch matched by a
        "file + version" filter."""
        from unittest import mock

        from kb_pipeline.vector import qdrant as qmod

        calls: list = []

        class FakeQ:
            def set_payload(self, **kw):
                calls.append(("set", kw.get("points")))

            def delete_payload(self, **kw):
                calls.append(("del", kw.get("points")))

        row = {"collection": "kb_1", "kb_id": "kb_1", "file_key": 7, "content_version": "v1", "source_path": "x/a.pdf",
               "rel_path": "a.pdf", "filename": "a.pdf", "dir": "x", "mime_type": "application/pdf", "size": 1, "mtime": 1}
        with mock.patch.object(qmod, "collection_exists", return_value=True), \
                mock.patch.object(qmod, "metadata_payload_from_file_row", return_value={"source_path": "x/a.pdf"}):
            qmod.reactivate_file_metadata(FakeQ(), row, point_ids=["p1", "p2"])
            self.assertEqual([c[1] for c in calls], [["p1", "p2"], ["p1", "p2"]])
            calls.clear()
            qmod.reactivate_file_metadata(FakeQ(), row, point_ids=[])
            self.assertEqual(calls, [])                                  # nothing to restore, nothing touched
        worker = _repo_file("app/kb_pipeline/pipeline/worker.py")
        self.assertIn("db.chunk_point_ids_to_restore(", worker)
        self.assertIn("reactivate_file_metadata(q, file_row, point_ids=restore_ids)", worker)
        self.assertIn("db.mark_chunks_deleted(", worker)

    def _reactivate_job(self, con, settings):
        file = _local_file("x/a.pdf", checksum="c1")
        db.upsert_file(con, file, status="indexed")
        fid = db.file_id_for(file.kb_id, file.file_key)
        job_id = db.enqueue_job(
            con, ingest_run_id=None, file_id=fid, kb_id=file.kb_id, collection=file.collection,
            file_key=file.file_key, job_type="metadata_update",
            payload={"reactivate": True}, dedupe_key=f"meta:{fid}",
        )
        return file, fid, job_id

    def test_worker_brings_all_three_projections_back(self) -> None:
        from unittest import mock

        from kb_pipeline.pipeline import worker as worker_mod

        with tempfile.TemporaryDirectory() as tmp:
            settings = self._settings(Path(tmp))
            db.init_db(settings.state_db)
            with db.connect(settings.state_db) as con:
                file, fid, _ = self._reactivate_job(con, settings)
                row = db.get_file_by_id(con, fid)
                from kb_pipeline.models import UnifiedChunk

                db.replace_chunks(
                    con, file_id=fid, collection=file.collection,
                    content_version=str(row["content_version"]),
                    chunks=[UnifiedChunk(chunk_uid="u1", chunk_index=0, text="x", block=_block("b", "x"))],
                    point_ids=["p1"],
                )
                # run a delete: both sides deactivate (SQLite side: deleted; restore revives only this batch)
                db.mark_chunks_deleted(con, fid)
                con.commit()

                with mock.patch.object(worker_mod, "qdrant_client"), \
                        mock.patch.object(worker_mod, "reactivate_file_metadata") as reactivated, \
                        mock.patch.object(worker_mod, "_sync_fts_doc"):
                    result = worker_mod.run_once(con, settings=settings)
                statuses = [r[0] for r in con.execute(
                    "SELECT status FROM chunks WHERE file_id=?", (fid,)
                ).fetchall()]

            self.assertTrue(result.startswith("metadata-done"))
            reactivated.assert_called_once()                       # Qdrant
            self.assertEqual(statuses, ["active"])                 # SQLite

    def test_worker_falls_back_to_reparse_when_the_copy_is_gone(self) -> None:
        """The deactivated points passed the retention period and were GC'd: restoring is impossible, fall back
        to re-parsing.

        The chunks must not be marked active here; that would create an "in the ledger, not in Qdrant" state,
        and the graph build would take the ledger and fetch points that do not exist. Keep them inactive and
        bring everything back together once the re-parse finishes.
        """
        from unittest import mock

        from kb_pipeline.pipeline import worker as worker_mod

        with tempfile.TemporaryDirectory() as tmp:
            settings = self._settings(Path(tmp))
            db.init_db(settings.state_db)
            with db.connect(settings.state_db) as con:
                file, fid, _ = self._reactivate_job(con, settings)
                row = db.get_file_by_id(con, fid)
                from kb_pipeline.models import UnifiedChunk

                db.replace_chunks(
                    con, file_id=fid, collection=file.collection,
                    content_version=str(row["content_version"]),
                    chunks=[UnifiedChunk(chunk_uid="u1", chunk_index=0, text="x", block=_block("b", "x"))],
                    point_ids=["p1"],
                )
                db.mark_chunks_inactive(con, fid)      # the deactivated points never existed / were GC'd
                con.commit()

                with mock.patch.object(worker_mod, "qdrant_client"), \
                        mock.patch.object(worker_mod, "file_version_point_count", return_value=0), \
                        mock.patch.object(worker_mod, "reactivate_file_metadata"), \
                        mock.patch.object(worker_mod, "_sync_fts_doc"):
                    worker_mod.run_once(con, settings=settings)
                statuses = [r[0] for r in con.execute(
                    "SELECT status FROM chunks WHERE file_id=?", (fid,)
                ).fetchall()]
                parse_jobs = con.execute(
                    "SELECT COUNT(*) FROM jobs WHERE file_id=? AND job_type='parse'", (fid,)
                ).fetchone()[0]

            self.assertEqual(statuses, ["inactive"])   # no half-active state is created
            self.assertEqual(parse_jobs, 1)            # falls back to re-parsing


class ChunkTextShaLedgerTests(unittest.TestCase):
    """Chunk text fingerprint: replace_chunks writes it into the chunks table → active_chunk_refs carries it
    out → the graph build ledger freezes it → the next incremental merge uses it to recognise documents whose
    uid is unchanged but whose text changed."""

    def test_text_sha_flows_from_chunks_table_into_the_build_ledger(self) -> None:
        from kb_pipeline.graph.build import document_delta
        from kb_pipeline.models import UnifiedChunk

        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "state.db"
            db.init_db(state)
            with db.connect(state) as con:
                file = _local_file("a.pdf", checksum="c1")
                db.upsert_file(con, file, status="indexed")
                fid = db.file_id_for(file.kb_id, file.file_key)

                def parse(texts):
                    db.replace_chunks(
                        con, file_id=fid, collection=file.collection, content_version="v1",
                        chunks=[UnifiedChunk(chunk_uid=f"u{i}", chunk_index=i, text=t, block=_block(f"b{i}", t))
                                for i, t in enumerate(texts)],
                        point_ids=[f"p{i}" for i in range(len(texts))],
                    )
                    return db.active_chunk_refs(con, file.collection)

                refs = parse(["第一段", "第二段"])
                self.assertEqual([r["text_sha"] for r in refs], [db.chunk_text_sha("第一段"), db.chunk_text_sha("第二段")])
                bid = db.begin_graph_build(con, source_key="k", kb_id=file.kb_id, source_collection=file.collection, graph_version="v1")
                db.replace_graph_build_chunks(con, bid, refs)
                base = db.graph_build_doc_chunks(con, bid)
                doc_id = f"{file.kb_id}:{file.file_key}"
                self.assertEqual(base, {doc_id: {("p0", "v1", db.chunk_text_sha("第一段")), ("p1", "v1", db.chunk_text_sha("第二段"))}})
                # same uid, same content_version, only the second paragraph's text changed
                refs = parse(["第一段", "第二段(改)"])
                self.assertEqual(document_delta(base, refs)["modified_docs"], [doc_id])
                self.assertEqual(document_delta(base, parse(["第一段", "第二段"]))["modified_docs"], [])


class KbRenameTests(unittest.TestCase):
    """Top-level directory rename detection (companion of the 2026-09-06 decision to keep numbered ids): the
    number and all data are kept, only the directory changes, nothing is re-parsed."""

    def _seed(self, root: Path, state: Path):
        from kb_pipeline import discovery
        from kb_pipeline.localfs.scanner import list_source_files
        from kb_pipeline.pipeline.detect_changes import parser_profile_for

        (root / "半导体").mkdir()
        for name, body in (("a.md", "# A\n\nalpha " * 20), ("sub/b.md", "# B\n\nbeta " * 30), ("c.md", "# C\n\ngamma " * 10)):
            p = root / "半导体" / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body, encoding="utf-8")
        db.init_db(state)
        with db.connect(state) as con:
            src, _ = discovery.enroll(con, root, "半导体")
            for f in list_source_files(src, min_age_seconds=0, hash_content=True):
                db.upsert_file(con, f, status="indexed")
                db.mark_file_indexed(con, db.file_id_for(f.kb_id, f.file_key), f.content_version, parser_profile_for(f))
            con.commit()
        return src

    def test_renamed_directory_is_recognised_and_adopted_without_reparse(self) -> None:
        from kb_pipeline import discovery
        from kb_pipeline.localfs.scanner import list_source_files
        from kb_pipeline.pipeline.detect_changes import detect_change

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = root / "s.db"
            src = self._seed(root, state)
            (root / "半导体").rename(root / "半导体资料")
            with db.connect(state) as con:
                discovery.mark_inactive(con, src.kb_id)          # what the scan does after finding the directory gone
                found = discovery.find_renamed_directories(con, root)
                self.assertEqual([(f["kb_id"], f["old_dir"], f["dir"]) for f in found], [(src.kb_id, "半导体", "半导体资料")])
                self.assertEqual((found[0]["matched"], found[0]["total"], found[0]["verified"]), (3, 3, 3))
                adopted = discovery.adopt_directory(con, root, src.kb_id, "半导体资料")
                self.assertEqual((adopted.kb_id, adopted.source_root, adopted.collection), (src.kb_id, "半导体资料", src.collection))
                row = con.execute("SELECT status, source_root, inactive_reason FROM kb_sources WHERE kb_id=?", (src.kb_id,)).fetchone()
                self.assertEqual((row["status"], row["source_root"], row["inactive_reason"]), ("active", "半导体资料", None))
                paths = {r["rel_path"]: (r["source_path"], r["physical_path"])
                         for r in con.execute("SELECT rel_path, source_path, physical_path FROM files WHERE kb_id=?", (src.kb_id,))}
                self.assertEqual(paths["sub/b.md"], ("半导体资料/sub/b.md", str(root / "半导体资料" / "sub" / "b.md")))
                # next scan round: files are recognised under the new path, content unchanged → metadata change only
                # (metadata_update refreshes the path fields in the payload); none of them goes through parsing
                changes = {f.rel_path: (c.change_type, c.job_type) for f in list_source_files(adopted, min_age_seconds=0, hash_content=True)
                           for c in [detect_change(db.get_file_by_id(con, db.file_id_for(f.kb_id, f.file_key)), f)]}
                self.assertEqual(changes, {name: ("metadata_changed", "metadata_update") for name in ("a.md", "sub/b.md", "c.md")})
                # once adopted there are no more candidates
                self.assertEqual(discovery.find_renamed_directories(con, root), [])

    def test_adoption_refuses_ambiguous_cases(self) -> None:
        from kb_pipeline import discovery

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = root / "s.db"
            src = self._seed(root, state)
            (root / "别的资料").mkdir()
            (root / "别的资料" / "x.md").write_text("unrelated", encoding="utf-8")
            with db.connect(state) as con:
                # an active KB whose directory still exists cannot be re-pointed
                with self.assertRaises(ValueError):
                    discovery.adopt_directory(con, root, src.kb_id, "别的资料")
                with self.assertRaises(KeyError):
                    discovery.adopt_directory(con, root, "kb_999", "别的资料")
                # a directory whose content does not match is not a rename
                report = discovery.directory_match_report(con, root, src.kb_id, "别的资料")
                self.assertEqual((report["matched"], report["total"]), (0, 3))
                self.assertFalse(discovery.looks_like_rename(report))
                # a KB the user closed (unenrolled) is not adopted automatically; left to the console, by hand
                (root / "半导体").rename(root / "半导体资料")
                discovery.mark_inactive(con, src.kb_id, reason="unenrolled")
                self.assertEqual(discovery.find_renamed_directories(con, root), [])
                adopted = discovery.adopt_directory(con, root, src.kb_id, "半导体资料")     # manual adoption still works
                self.assertEqual(adopted.source_root, "半导体资料")
                # the new directory is already taken by another registry row
                other, _ = discovery.enroll(con, root, "别的资料")
                with self.assertRaises(ValueError):
                    discovery.adopt_directory(con, root, src.kb_id, "别的资料")
                self.assertTrue(discovery.looks_like_rename({"total": 10, "matched": 9, "ratio": 0.9, "mismatched": 0, "new_files": 12}))
                self.assertFalse(discovery.looks_like_rename({"total": 2, "matched": 2, "ratio": 1.0, "mismatched": 0, "new_files": 500}))

    def test_rename_is_wired_into_scan_console_and_logs(self) -> None:
        cli = _repo_file("app/kb_pipeline/cli.py")
        self.assertIn("discovery.find_renamed_directories(state_con, settings.mirror_root)", cli)
        self.assertIn("discovery.adopt_directory(state_con, settings.mirror_root, report[\"kb_id\"], report[\"dir\"])", cli)
        self.assertIn("discovery.kb_label(source.kb_id, source.source_root)", cli)      # the log shows the directory name next to the id
        self.assertIn('"/kbs/{kb_id}/adopt"', _repo_file("app/kb_server/api.py"))
        self.assertIn("def adopt_kb(", _repo_file("app/kb_server/service.py"))
        html = _repo_file("app/kb_server/static/index.html")
        self.assertIn('id="sel-adopt"', html)
        self.assertIn('id="sel-id"', html)
        js = _repo_file("app/kb_server/static/app.js")
        self.assertIn("function renderAdoptRow", js)
        self.assertIn("/adopt`", js)
        self.assertIn("[graph] build start kb=", _repo_file("app/kb_pipeline/graph/build.py"))


class StateFixRegressionTests(unittest.TestCase):
    """Regressions for problems found by successive re-reviews, health checks and audits; each test's
    docstring records where it came from and the symptom observed at the time."""

    def test_enrollment_invariants(self) -> None:  # issue 6 successor
        from kb_pipeline import discovery

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "目录A").mkdir()
            with db.connect(Path(tmp) / "s.db") as con:
                con.executescript(db.SCHEMA)
                discovery.init_schema(con)
                src, _ = discovery.enroll(con, root, "目录A")
                # a tampered registry row (different collection than the code
                # derives) is refused instead of splitting the KB
                con.execute("UPDATE kb_sources SET collection='kb_other' WHERE kb_id=?", (src.kb_id,))
                with self.assertRaises(discovery.CollectionMismatch):
                    discovery.touch_seen(con, src)
                with self.assertRaises(discovery.CollectionMismatch):
                    discovery.enroll(con, root, "目录A")

    def test_chunk_limits_and_validation(self) -> None:
        from kb_pipeline import limits as limits_module

        class FakeResponse:
            def json(self):
                return {"data": [{"max_model_len": 4096}]}

        with patch.object(limits_module.requests, "get", return_value=FakeResponse()):
            lim = limits_module.chunk_limits("http://x/v1")
        self.assertEqual(lim["max_tokens_cap"], 3276)      # int(4096 * 0.8)
        self.assertTrue(lim["live"])
        self.assertEqual(limits_module.validate_chunk_config(800, 120, lim), [])
        self.assertTrue(limits_module.validate_chunk_config(4000, 80, lim))
        self.assertTrue(limits_module.validate_chunk_config(400, 200, lim))   # overlap >= max/2
        self.assertTrue(limits_module.validate_chunk_config("abc", 80, lim))

        def boom(*a, **k):
            raise OSError("down")

        with patch.object(limits_module.requests, "get", boom):
            offline = limits_module.chunk_limits("http://x/v1")
        self.assertFalse(offline["live"])
        self.assertEqual(offline["max_tokens_cap"], 3276)  # fallback: 80% of 4096

    def test_llm_registry_and_graph_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with db.connect(Path(tmp) / "s.db") as con:
                con.executescript(db.SCHEMA); db.migrate_schema(con)
                db.upsert_llm(con, name="gpu-a-qwen", base_url="http://gpu-a:8000/v1",
                              api_key="k1", model_id="qwen3.7-plus")
                db.upsert_llm(con, name="本机", base_url="http://127.0.0.1:8105/v1",
                              api_key="local", model_id="qwen3-vl-8b-instruct-fp8", builtin=True)
                self.assertEqual(len(db.list_llms(con)), 2)
                # blank api_key on update keeps the stored secret
                db.upsert_llm(con, name="gpu-a-qwen", base_url="http://gpu-a:8000/v1",
                              api_key="", model_id="qwen3.7-plus-v2")
                row = db.get_llm(con, "gpu-a-qwen")
                self.assertEqual((row["api_key"], row["model_id"]), ("k1", "qwen3.7-plus-v2"))
                with self.assertRaises(ValueError):
                    db.delete_llm(con, "本机")      # builtin protected
                self.assertTrue(db.delete_llm(con, "gpu-a-qwen"))
                db.set_app_config(con, "some_setting", {"k": 1})
                self.assertEqual(db.get_app_config(con, "some_setting"), {"k": 1})

    def test_discovery_migrations_only_swallow_duplicates(self) -> None:
        """B7: migrations only swallow "already exists"; locked / read-only / disk full still raise."""
        import sqlite3

        from kb_pipeline import discovery

        class Con:
            def __init__(self, message):
                self.message = message

            def executescript(self, sql):
                pass

            def execute(self, sql, *args):
                raise sqlite3.OperationalError(self.message)

        with self.assertRaises(sqlite3.OperationalError):
            discovery.init_schema(Con("database is locked"))
        discovery.init_schema(Con("duplicate column name: graph_paused"))
        discovery.init_schema(Con("index idx_x already exists"))
