from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from kb_pipeline import db, discovery
from kb_pipeline.embedding.client import EmbeddingDimensionError
from kb_pipeline.models import KBSource, SourceFile
from kb_pipeline.parsers.errors import (
    ConfigParseError,
    FAILURE_CONFIG,
    FAILURE_DETERMINISTIC,
    FAILURE_RETRYABLE,
    FAILURE_VERDICT_KEY,
    NonRetryableParseError,
)
from kb_pipeline.parsers.mineru_pdf import MinerUServiceError
from kb_pipeline.pipeline import parse_job, scheduler, worker
from kb_server import service


def _source_file(
    kb_id: str = "kb_1",
    file_key: int = 1,
    filename: str = "doc.pdf",
    version: str = "v1",
    size: int = 1234,
) -> SourceFile:
    return SourceFile(
        kb_id=kb_id,
        collection="col_1",
        source_root=f"/{kb_id}",
        source_type="local_mirror",
        file_key=file_key,
        source_path=f"/{kb_id}/{filename}",
        rel_path=filename,
        filename=filename,
        dir="",
        mime_type="application/pdf" if filename.endswith(".pdf") else "text/markdown",
        size=size,
        mtime=1700000000,
        checksum=f"sha-{file_key}-{version}",
        physical_path=f"/tmp/{kb_id}/{filename}",
    )


def _settings(state: Path, mirror_root: Path | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        state_db=state,
        mirror_root=mirror_root or state.parent,
        parse_enabled=True,
        parse_job_lease_seconds=600,
        metadata_job_lease_seconds=600,
        job_max_retries=2,
        job_retry_base_seconds=300,
        job_retry_max_seconds=3600,
        qdrant_url="http://q.invalid",
        qdrant_api_key="",
        opensearch_url="http://o.invalid",
        embedding_base_url="http://e.invalid",
        sources={},
    )


def _insert_kb(con, kb_id: str = "kb_1", *, config: dict | None = None) -> None:
    con.execute(
        "INSERT INTO kb_sources (kb_id, collection, source_root, status, first_seen_at, last_seen_at, config_json) "
        "VALUES (?, ?, ?, 'active', 1, 1, ?)",
        (kb_id, f"col_{kb_id}", f"dir_{kb_id}", json.dumps(config or {})),
    )


class Round11SchedulerConsoleTests(unittest.TestCase):
    def _init_db(self, tmp: str) -> Path:
        state = Path(tmp) / "state.db"
        db.init_db(state)
        return state

    def test_b10_paused_kb_queued_parse_jobs_do_not_count_as_busy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = self._init_db(tmp)
            with db.connect(state) as con:
                _insert_kb(con, "kb_1", config={"parse_paused": False})
                f1 = _source_file("kb_1", 1, "a.pdf", "v1")
                f2 = _source_file("kb_1", 2, "b.pdf", "v1")
                _, j1 = scheduler.schedule_file(con, ingest_run_id="r1", file=f1)
                _, j2 = scheduler.schedule_file(con, ingest_run_id="r1", file=f2)
                self.assertIsNotNone(j1)
                self.assertIsNotNone(j2)
                # Active KB: 2 queued claimable jobs -> busy == 2
                self.assertEqual(db.kb_parse_busy(con, "kb_1"), 2)

                # Pause KB: queued jobs cannot be claimed -> busy == 0
                discovery.set_config(con, "kb_1", {"parse_paused": True})
                self.assertEqual(db.kb_parse_busy(con, "kb_1"), 0)

                # If one job is still actively running before yielding, busy == 1
                con.execute("UPDATE jobs SET status='running' WHERE job_id=?", (j1,))
                self.assertEqual(db.kb_parse_busy(con, "kb_1"), 1)

                # Unpause KB: both running + queued count again -> busy == 2
                discovery.set_config(con, "kb_1", {"parse_paused": False})
                self.assertEqual(db.kb_parse_busy(con, "kb_1"), 2)

    def test_b11_restore_after_gc_clears_indexed_version_and_skips_fts_and_mark_indexed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = self._init_db(tmp)
            sf = _source_file("kb_1", 1, "guide.md", "v1")
            fid = db.file_id_for(sf.kb_id, sf.file_key)
            with db.connect(state) as con:
                db.upsert_file(con, sf, status="seen")
                con.execute(
                    "UPDATE files SET indexed_version = ?, indexed_parser_profile = 'md-sections-v3' WHERE file_id = ?",
                    (sf.content_version, fid),
                )
                job_id = db.enqueue_job(
                    con,
                    ingest_run_id="restore-1",
                    file_id=fid,
                    kb_id="kb_1",
                    collection="col_1",
                    file_key=1,
                    job_type="metadata_update",
                    priority=10,
                    payload={"reactivate": True, "reason": "restored"},
                    dedupe_key=f"metadata_update:{fid}:fp1",
                )
                con.commit()

                fts_calls = []

                class _FakeChunks:
                    def version_count(self, _row, *, point_ids=None):
                        return 0  # GC'd in Qdrant

                    def reactivate(self, *_a, **_kw):
                        raise AssertionError("reactivate must not be called when points are gone")

                    def deactivate(self, *_a, **_kw):
                        return None

                class _FakeKeywords:
                    def sync_document(self, **kw):
                        fts_calls.append(kw)
                        return {}

                fake_store = SimpleNamespace(chunks=_FakeChunks(), keywords=_FakeKeywords())
                settings = _settings(state)
                with patch.object(worker, "get_storage", return_value=fake_store):
                    res = worker.run_once(con, settings=settings)
                self.assertEqual(res, f"metadata-done:{job_id}")
                # FTS sync must NOT have been called
                self.assertEqual(fts_calls, [])
                # File must NOT be marked indexed; indexed_version cleared so next scan keeps it unindexed
                row = db.get_file_by_id(con, fid)
                self.assertIsNone(row["indexed_version"])
                self.assertIsNone(row["indexed_parser_profile"])
                self.assertNotEqual(str(row["status"]), "indexed")
                # A re-parse job must have been queued
                parse_jobs = con.execute("SELECT * FROM jobs WHERE file_id=? AND job_type='parse' AND status='queued'", (fid,)).fetchall()
                self.assertEqual(len(parse_jobs), 1)
                # Subsequent scan must NOT flip status back to indexed while re-parse is pending
                change, _ = scheduler.schedule_file(con, ingest_run_id="scan-2", file=sf)
                self.assertEqual(change, "needs_parse")
                self.assertEqual(str(db.get_file_by_id(con, fid)["status"]), "seen")

    def test_e6_chunk_preview_reads_indexed_parser_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = self._init_db(tmp)
            sf = _source_file("kb_1", 1, "spec.pdf", "v1")
            fid = db.file_id_for(sf.kb_id, sf.file_key)
            with db.connect(state) as con:
                db.upsert_file(con, sf, status="indexed")
                db.mark_file_indexed(con, fid, sf.content_version, parser_profile="pdf-mineru-table-vlm-v15+source-v1")
                con.commit()
            cfg = _settings(state)
            with patch.object(service, "settings", return_value=cfg):
                resp = service.file_chunks("kb_1", fid)
            self.assertEqual(resp["file"]["parser_profile"], "pdf-mineru-table-vlm-v15+source-v1")

    def test_e9_kb_job_cancel_resolves_failures_on_cancelled_queued_and_retry_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = self._init_db(tmp)
            with db.connect(state) as con:
                f1 = _source_file("kb_1", 1, "a.pdf", "v1")
                f2 = _source_file("kb_1", 2, "b.pdf", "v1")
                _, j1 = scheduler.schedule_file(con, ingest_run_id="r1", file=f1)
                _, j2 = scheduler.schedule_file(con, ingest_run_id="r1", file=f2)
                con.execute("UPDATE jobs SET status='retry' WHERE job_id=?", (j1,))
                con.execute("UPDATE jobs SET status='running' WHERE job_id=?", (j2,))
                db.add_failure(con, file_id=db.file_id_for("kb_1", 1), job_id=j1, stage="worker",
                               error_type="RuntimeError", error_message="boom1")
                db.add_failure(con, file_id=db.file_id_for("kb_1", 2), job_id=j2, stage="worker",
                               error_type="RuntimeError", error_message="boom2")
                con.commit()

                counts = db.request_kb_job_cancel(con, "kb_1", "kb stopped")
                self.assertEqual(counts, {"cancelled": 1, "signalled": 1})
                f_j1 = con.execute("SELECT resolved_at FROM failures WHERE job_id=?", (j1,)).fetchone()
                f_j2 = con.execute("SELECT resolved_at FROM failures WHERE job_id=?", (j2,)).fetchone()
                self.assertIsNotNone(f_j1["resolved_at"])   # retry job cancelled -> failure resolved
                self.assertIsNone(f_j2["resolved_at"])      # running job only signalled -> resolved when worker exits

    def test_e7_user_cancel_holds_back_scans_until_content_change_or_manual_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = self._init_db(tmp)
            sf = _source_file("kb_1", 1, "big.pdf", "v1")
            fid = db.file_id_for(sf.kb_id, sf.file_key)
            with db.connect(state) as con:
                # 1. Initial scan queues parse job
                ch, jid = scheduler.schedule_file(con, ingest_run_id="scan-1", file=sf)
                self.assertEqual(ch, "new")
                self.assertIsNotNone(jid)
                con.commit()

                # 2. User cancels from console -> cancelled_by="user" written into payload_json
                cfg = _settings(state)
                with patch.object(service, "settings", return_value=cfg):
                    res = service.cancel_job(jid)
                self.assertEqual(res["outcome"], "cancelled")
                payload = json.loads(con.execute("SELECT payload_json FROM jobs WHERE job_id=?", (jid,)).fetchone()[0])
                self.assertEqual(payload.get("cancelled_by"), "user")

                # 3. Subsequent scans hold back (do NOT re-enqueue)
                ch2, jid2 = scheduler.schedule_file(con, ingest_run_id="scan-2", file=sf)
                self.assertEqual((ch2, jid2), ("cancelled", None))
                self.assertEqual(con.execute("SELECT COUNT(*) FROM jobs WHERE file_id=?", (fid,)).fetchone()[0], 1)

                # 4. Manual retry from console enqueues again immediately
                jid3 = scheduler.requeue_single_file(con, ingest_run_id="retry-1", file_row=db.get_file_by_id(con, fid))
                self.assertIsNotNone(jid3)

                # 5. Running job cancelled by user keeps cancelled_by="user" after worker exits
                con.execute("UPDATE jobs SET status='running', locked_by='w1' WHERE job_id=?", (jid3,))
                con.commit()
                self.assertEqual(db.request_job_cancel(con, jid3, "Cancelled from the console", cancelled_by="user"), "signalled")
                self.assertTrue(db.cancel_job_if_owned(con, jid3, "w1", "Cancelled"))
                ch4, jid4 = scheduler.schedule_file(con, ingest_run_id="scan-4", file=sf)
                self.assertEqual((ch4, jid4), ("cancelled", None))

                # 6. Content change overrides holdback and queues fresh parse
                sf_v2 = _source_file("kb_1", 1, "big.pdf", "v2", size=2000)
                ch5, jid5 = scheduler.schedule_file(con, ingest_run_id="scan-5", file=sf_v2)
                self.assertEqual(ch5, "content_changed")
                self.assertIsNotNone(jid5)

                # 7. System cancellation (no cancelled_by="user") is NOT held back on next scan
                db.cancel_job(con, jid5, "kb closed temporarily")
                ch6, jid6 = scheduler.schedule_file(con, ingest_run_id="scan-6", file=sf_v2)
                self.assertEqual(ch6, "needs_parse")
                self.assertIsNotNone(jid6)

    def test_3b_config_errors_separated_from_deterministic_and_auto_requeued_on_save(self) -> None:
        # Classification checks
        self.assertEqual(worker._failure_verdict(ConfigParseError("no vlm")), FAILURE_CONFIG)
        self.assertEqual(worker._failure_verdict(EmbeddingDimensionError("dim mismatch")), FAILURE_CONFIG)
        for code in (401, 403, 404, 409):
            err = MinerUServiceError(f"HTTP {code}", status_code=code)
            self.assertTrue(err.config_error)
            self.assertFalse(err.deterministic)
            self.assertEqual(worker._failure_verdict(err), FAILURE_CONFIG)
        for code in (400, 413, 415, 422):
            err = MinerUServiceError(f"HTTP {code}", status_code=code)
            self.assertTrue(err.deterministic)
            self.assertFalse(err.config_error)
            self.assertEqual(worker._failure_verdict(err), FAILURE_DETERMINISTIC)
        self.assertEqual(worker._failure_verdict(MinerUServiceError("429", status_code=429)), FAILURE_RETRYABLE)
        self.assertEqual(worker._failure_verdict(NonRetryableParseError(".doc unsupported")), FAILURE_DETERMINISTIC)

        # _vlm_endpoint raises ConfigParseError when no model configured
        with tempfile.TemporaryDirectory() as tmp:
            state = self._init_db(tmp)
            cfg = _settings(state, mirror_root=Path(tmp))
            src = SimpleNamespace(kb_id="kb_1", collection="col_1", vlm_model=None)
            with self.assertRaises(ConfigParseError):
                parse_job._vlm_endpoint(cfg, src)

            # Seed 3 files: file1 failed with FAILURE_CONFIG, file2 failed with FAILURE_DETERMINISTIC, file3 user-cancelled
            with db.connect(state) as con:
                _insert_kb(con, "kb_1")
                con.execute(
                    "UPDATE kb_sources SET config_json = ? WHERE kb_id = 'kb_1'",
                    (json.dumps({"vlm_model": "vlm-main"}),),
                )
                f1 = _source_file("kb_1", 1, "cfg_err.pdf", "v1")
                f2 = _source_file("kb_1", 2, "bad_doc.pdf", "v1")
                f3 = _source_file("kb_1", 3, "user_stop.pdf", "v1")
                _, j1 = scheduler.schedule_file(con, ingest_run_id="r1", file=f1)
                _, j2 = scheduler.schedule_file(con, ingest_run_id="r1", file=f2)
                _, j3 = scheduler.schedule_file(con, ingest_run_id="r1", file=f3)
                for jid, verdict in ((j1, FAILURE_CONFIG), (j2, FAILURE_DETERMINISTIC)):
                    p = json.loads(con.execute("SELECT payload_json FROM jobs WHERE job_id=?", (jid,)).fetchone()[0])
                    p[FAILURE_VERDICT_KEY] = verdict
                    con.execute("UPDATE jobs SET status='failed', payload_json=?, finished_at=? WHERE job_id=?",
                                (json.dumps(p), db.now_ts(), jid))
                db.request_job_cancel(con, j3, "Cancelled from the console", cancelled_by="user")
                con.commit()

                # Saving LLM in console auto-requeues ONLY f1 (FAILURE_CONFIG), leaving f2 & f3 untouched
                kicked = []
                with patch.object(service, "settings", return_value=cfg), \
                        patch.object(service, "_probe_llm", return_value=None), \
                        patch.object(service, "kick_worker", side_effect=lambda: kicked.append(1)):
                    service.save_llm({
                        "name": "vlm-main",
                        "base_url": "http://127.0.0.1:8000/v1",
                        "model_id": "qwen-vl",
                        "protocol": "openai",
                        "multimodal": True,
                        "concurrency": 2,
                    })
                self.assertEqual(len(kicked), 1)
                queued_files = {
                    r["file_id"] for r in con.execute("SELECT file_id FROM jobs WHERE status='queued'").fetchall()
                }
                self.assertEqual(queued_files, {db.file_id_for("kb_1", 1)})

                # Fail f1 again with FAILURE_CONFIG, then save KB config -> auto-requeues f1 again
                j1_new = con.execute("SELECT job_id, payload_json FROM jobs WHERE file_id=? AND status='queued'",
                                     (db.file_id_for("kb_1", 1),)).fetchone()
                p = json.loads(j1_new["payload_json"])
                p[FAILURE_VERDICT_KEY] = FAILURE_CONFIG
                con.execute("UPDATE jobs SET status='failed', payload_json=?, finished_at=? WHERE job_id=?",
                            (json.dumps(p), db.now_ts(), j1_new["job_id"]))
                con.commit()

                with patch.object(service, "settings", return_value=cfg), \
                        patch.object(service, "chunk_limits", return_value={"max_tokens_min": 100, "max_tokens_cap": 2000, "effective_max_model_len": 8192}), \
                        patch.object(service, "kick_worker", side_effect=lambda: kicked.append(2)):
                    # Clear vlm_model or change to another registered model triggers KB VLM config requeue
                    db.upsert_llm(con, name="vlm-backup", base_url="http://127.0.0.1:8001/v1", api_key="",
                                  model_id="qwen-vl-2", protocol="openai", max_input_tokens=8192,
                                  multimodal=True, concurrency=2)
                    con.commit()
                    service.update_kb_config("kb_1", {"vlm_model": "vlm-backup"})
                self.assertEqual(len(kicked), 2)
                queued_files2 = {
                    r["file_id"] for r in con.execute("SELECT file_id FROM jobs WHERE status='queued'").fetchall()
                }
                self.assertEqual(queued_files2, {db.file_id_for("kb_1", 1)})

    def test_adversarial_prune_retention_inactive_kb_and_endpoint_save(self) -> None:
        from kb_server import model_endpoints

        with tempfile.TemporaryDirectory() as tmp:
            state = self._init_db(tmp)
            old_ts = db.now_ts() - 40 * 86400
            with db.connect(state) as con:
                _insert_kb(con, "kb_1")
                con.execute(
                    "INSERT INTO kb_sources (kb_id, collection, source_root, status, first_seen_at, last_seen_at, config_json) "
                    "VALUES ('kb_off', 'col_off', 'dir_off', 'inactive', 1, 1, '{}')"
                )
                # 1. User-cancelled live file survives 30-day prune_job_history; system-cancelled does not
                f_live = _source_file("kb_1", 1, "keep_cancel.pdf", "v1")
                f_sys = _source_file("kb_1", 2, "sys_cancel.pdf", "v1")
                _, j_live = scheduler.schedule_file(con, ingest_run_id="r1", file=f_live)
                _, j_sys = scheduler.schedule_file(con, ingest_run_id="r1", file=f_sys)
                db.request_job_cancel(con, j_live, "Cancelled from the console", cancelled_by="user")
                db.cancel_job(con, j_sys, "system cancel")
                con.execute("UPDATE jobs SET finished_at = ?, updated_at = ? WHERE job_id IN (?, ?)",
                            (old_ts, old_ts, j_live, j_sys))
                con.commit()

                db.prune_job_history(con, retention_days=30)
                con.commit()
                remaining = {r[0] for r in con.execute("SELECT job_id FROM jobs").fetchall()}
                self.assertIn(j_live, remaining)
                self.assertNotIn(j_sys, remaining)
                # Next scan STILL holds back f_live after 30-day prune
                self.assertEqual(scheduler.schedule_file(con, ingest_run_id="scan-after-prune", file=f_live), ("cancelled", None))

                # 2. Inactive KB config-failed file is NOT requeued even when kb_id is passed explicitly
                f_off = _source_file("kb_off", 10, "off.pdf", "v1")
                _, j_off = scheduler.schedule_file(con, ingest_run_id="r1", file=f_off)
                p = json.loads(con.execute("SELECT payload_json FROM jobs WHERE job_id=?", (j_off,)).fetchone()[0])
                p[FAILURE_VERDICT_KEY] = FAILURE_CONFIG
                con.execute("UPDATE jobs SET status='failed', payload_json=?, finished_at=? WHERE job_id=?",
                            (json.dumps(p), db.now_ts(), j_off))
                con.commit()
                self.assertEqual(scheduler.requeue_config_failed_files(con, kb_id="kb_off"), 0)

                # 3. model_endpoints.save_endpoint auto-requeues FAILURE_CONFIG on active KB
                f_emb = _source_file("kb_1", 3, "dim.pdf", "v1")
                _, j_emb = scheduler.schedule_file(con, ingest_run_id="r1", file=f_emb)
                p2 = json.loads(con.execute("SELECT payload_json FROM jobs WHERE job_id=?", (j_emb,)).fetchone()[0])
                p2[FAILURE_VERDICT_KEY] = FAILURE_CONFIG
                con.execute("UPDATE jobs SET status='failed', payload_json=?, finished_at=? WHERE job_id=?",
                            (json.dumps(p2), db.now_ts(), j_emb))
                con.commit()

            cfg = _settings(state, mirror_root=Path(tmp))
            cfg.embedding_model_id = "bge"
            cfg.embedding_dim = 1024
            cfg.embedding_batch = 16
            cfg.embedding_concurrency = 4
            cfg.embedding_api_key = ""
            cfg.visual_embedding_enabled = False
            cfg.visual_embedding_base_url = ""
            cfg.visual_embedding_model_id = ""
            cfg.visual_embedding_api_key = ""
            cfg.visual_embedding_dim = 1024
            cfg.visual_embedding_batch = 8
            cfg.visual_embedding_concurrency = 2
            cfg.reranker_base_url = ""
            cfg.reranker_model_id = ""
            cfg.reranker_api_key = ""
            cfg.visual_reranker_base_url = ""
            rev = model_endpoints._revision(model_endpoints.endpoint_values(cfg)["embedding"])
            kicked = []
            with patch.object(model_endpoints, "load_settings", return_value=cfg), \
                    patch.object(model_endpoints, "_probe", side_effect=lambda _r, c: c), \
                    patch.object(model_endpoints, "_check_embedding_change", return_value=None), \
                    patch.object(service, "kick_worker", side_effect=lambda: kicked.append(1)):
                model_endpoints.save_endpoint(
                    "embedding",
                    {"base_url": "http://127.0.0.1:8001/v1", "model_id": "bge", "dim": 1024, "revision": rev},
                )
            self.assertEqual(len(kicked), 1)
            with db.connect(state) as con:
                q = con.execute("SELECT file_id FROM jobs WHERE status='queued' AND file_id=?",
                                (db.file_id_for("kb_1", 3),)).fetchall()
                self.assertEqual(len(q), 1)


if __name__ == "__main__":
    unittest.main()
