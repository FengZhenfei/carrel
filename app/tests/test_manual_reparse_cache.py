"""Manual parsing refreshes service output; automatic retries keep this run's progress."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from kb_pipeline import db, parse_cache
from kb_pipeline.parsers import service_clients, table_repair
from kb_pipeline.pipeline import parse_job, scheduler, worker
from kb_pipeline.vision import vlm
from kb_server import service
from test_scanner_worker import _parse_harness
from _support import _block


@pytest.fixture
def mineru(tmp_path, monkeypatch):
    src = tmp_path / "example.pdf"
    src.write_bytes(b"%PDF synthetic request input")
    result = tmp_path / "mineru" / "result.json"
    posts = []
    def post(*args, **kwargs):
        posts.append(kwargs)
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            "content_list": [{"type": "text", "text": f"revision {len(posts)}"}]})
    session = Mock()
    session.__enter__ = Mock(return_value=SimpleNamespace(post=post))
    session.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(service_clients, "_local_service_session", lambda: session)
    monkeypatch.setenv("MINERU_BACKEND", "vlm-engine")
    monkeypatch.setenv("MINERU_REUSE_CACHE", "1")
    return SimpleNamespace(posts=posts, cache=result, call=lambda **kw: service_clients.call_mineru_sync(
        "http://example.invalid", src, result, result.with_suffix(".md"), **kw))


def test_mineru_refresh_overwrites_one_cache_and_retries_reuse_it(mineru):
    mineru.call()
    mineru.cache.with_suffix(".md").write_text("previous markdown")
    with parse_cache.refresh_scope("manual-1"):
        with pytest.raises(RuntimeError, match="no cached MinerU result"):
            mineru.call(cached_only=True)
        mineru.call()
        mineru.call()
    assert len(mineru.posts) == 2
    assert mineru.cache.with_suffix(".md").read_text() == ""
    assert json.loads(mineru.cache.read_text())["content_list"][0]["text"] == "revision 2"
    assert sorted(p.name for p in mineru.cache.parent.iterdir()) == ["result.json", "result.md"]
    mineru.call()  # ordinary callers accept the latest refreshed output
    assert len(mineru.posts) == 2
    with parse_cache.refresh_scope("manual-2"):
        mineru.call()
    assert len(mineru.posts) == 3


@pytest.fixture
def model(tmp_path, monkeypatch):
    image = tmp_path / "image.png"
    Image.new("RGB", (24, 24), "white").save(image)
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        text = json.dumps({"summary": f"Synthetic chart revision {len(calls)}", "keywords": [],
                           "details": [], "confidence": "high", "rows": [["A", "10"], ["B", "20"]]})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(vlm, "_client_for", lambda *a: client)
    monkeypatch.setattr(table_repair, "_client_for", lambda *a: client)
    return SimpleNamespace(image=image, calls=calls, args=dict(base_url="http://example.invalid", api_key="synthetic", model_id="example-model"))


def test_caption_refresh_reaches_parallel_workers_without_leaking(model, tmp_path):
    cache = tmp_path / "caption.json"
    jobs = [("a", model.image, "Describe the chart", cache), ("b", model.image, "Describe the chart", cache)]
    def call(): return vlm.caption_images_parallel(jobs=jobs, concurrency=2, **model.args)
    call()
    with parse_cache.refresh_scope("manual-1"):
        fresh = call()
        retried = call()
        assert not fresh["a"]["vlm_cache_hit"] and fresh["a"] == fresh["b"]
        assert retried["a"]["vlm_cache_hit"]
    assert len(model.calls) == 2
    assert call()["a"]["vlm_cache_hit"]
    with parse_cache.refresh_scope("manual-2"):
        assert not call()["a"]["vlm_cache_hit"]
    assert len(model.calls) == 3
    assert parse_cache.reusable_in_run({})  # task-local policy was reset


@pytest.mark.parametrize("kind", ["table", "rows"])
def test_table_review_refreshes_both_cache_paths(model, tmp_path, kind):
    cache = tmp_path / "table.json"
    def call():
        if kind == "table":
            return table_repair.transcribe_table(model.image, prompt="Transcribe", cache=cache, options={}, **model.args)
        # Row-key transcription has a different output contract.
        return table_repair.transcribe_rows(model.image, row_keys=["A", "B"], width=2, cache_json=cache, **model.args)
    if kind == "rows":
        def create(**kwargs):
            model.calls.append(kwargs)
            content = json.dumps({"rows": [{"key": "A", "cells": ["A", "10"]}, {"key": "B", "cells": ["B", "20"]}]})
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
        table_repair._client_for(*[model.args[k] for k in ("base_url", "api_key")]).chat.completions.create = create
    assert call()
    with parse_cache.refresh_scope("manual-1"):
        assert call()
        assert call()
    assert len(model.calls) == 2
    with parse_cache.refresh_scope("manual-2"):
        assert call()
    assert len(model.calls) == 3


def test_failed_atomic_write_keeps_previous_cache_and_resets_scope(tmp_path, monkeypatch):
    path = tmp_path / "result.json"
    path.write_text('{"old":true}')
    monkeypatch.setattr(parse_cache.os, "replace", Mock(side_effect=OSError("disk error")))
    with pytest.raises(OSError), parse_cache.refresh_scope("manual"):
        parse_cache.write_cache(path, {"new": True})
    assert json.loads(path.read_text()) == {"old": True}
    assert list(tmp_path.iterdir()) == [path]
    assert parse_cache.reusable_in_run({})


def test_console_button_refreshes_cache_and_retry_keeps_old_index_until_success(tmp_path, monkeypatch, mineru):
    with _parse_harness(str(tmp_path / "pipeline")) as h, monkeypatch.context() as scoped:
        scoped.setattr(service, "settings", lambda: h.settings)
        scoped.setattr(service, "kick_worker", lambda: None)
        def parse(*args, **kwargs):
            data = mineru.call()
            return [_block("body", data["content_list"][0]["text"])]
        scoped.setattr(parse_job, "_parse_blocks", parse)
        assert worker.run_once(h.con, settings=h.settings).startswith("parse-done:")
        h.con.commit()
        old = dict(db.get_file_by_id(h.con, h.file_id))
        old_chunks = [dict(r) for r in h.con.execute("SELECT * FROM chunks WHERE file_id=? AND status='active'", (h.file_id,))]
        reply = service.retry_file(h.file_id)
        job = h.con.execute("SELECT * FROM jobs WHERE job_id=?", (reply["job_id"],)).fetchone()
        assert json.loads(job["payload_json"])["refresh_parse_cache"] is True
        def fail(): raise RuntimeError("temporary embedding error")
        h.hooks["embed"] = fail
        assert worker.run_once(h.con, settings=h.settings).startswith("retry:")
        assert len(mineru.posts) == 2
        assert db.get_file_by_id(h.con, h.file_id)["indexed_version"] == old["indexed_version"]
        assert [dict(r) for r in h.con.execute("SELECT * FROM chunks WHERE file_id=? AND status='active'", (h.file_id,))] == old_chunks
        h.hooks.pop("embed")
        h.con.execute("UPDATE jobs SET next_attempt_at=0 WHERE job_id=?", (reply["job_id"],));h.con.commit()
        assert worker.run_once(h.con, settings=h.settings).startswith("parse-done:")
        assert len(mineru.posts) == 2
        active = h.con.execute("SELECT * FROM chunks WHERE file_id=? AND status='active'", (h.file_id,)).fetchall()
        assert len(active) == 1
        assert active[0]["text_sha"] == db.chunk_text_sha("revision 2")
        assert h.args["upsert"][4][0]["text"] == "revision 2"
        assert h.calls[-4:] == ["upsert", "mark_old", "mark_stale", "fts"]


def test_single_and_whole_kb_requests_upgrade_pending_work_without_duplicates(tmp_path):
    with _parse_harness(str(tmp_path)) as h:
        row = db.get_file_by_id(h.con, h.file_id)
        assert scheduler.requeue_single_file(h.con, ingest_run_id="manual", file_row=row, refresh_cache=True) is None
        job = h.con.execute("SELECT * FROM jobs WHERE job_id=?", (h.job_id,)).fetchone()
        assert json.loads(job["payload_json"])["refresh_parse_cache"] is True
        assert scheduler.requeue_kb_files(h.con, ingest_run_id="all", kb_id=h.kb_id, reason="manual") == 0
        assert h.con.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
        assert worker.run_once(h.con, settings=h.settings).startswith("parse-done:")
        assert scheduler.requeue_kb_files(h.con, ingest_run_id="all", kb_id=h.kb_id, reason="manual") == 1
        queued = h.con.execute("SELECT * FROM jobs WHERE status='queued'").fetchone()
        assert json.loads(queued["payload_json"])["refresh_parse_cache"] is True


def test_whole_kb_request_during_parse_queues_a_fresh_followup(tmp_path):
    with _parse_harness(str(tmp_path)) as h:
        h.hooks["embed"] = lambda: scheduler.requeue_kb_files(
            h.con, ingest_run_id="manual", kb_id=h.kb_id, reason="full re-parse")
        assert worker.run_once(h.con, settings=h.settings).startswith("parse-done:")
        pending = h.con.execute("SELECT * FROM jobs WHERE status='queued'").fetchall()
        assert len(pending) == 1
        assert json.loads(pending[0]["payload_json"])["refresh_parse_cache"] is True
