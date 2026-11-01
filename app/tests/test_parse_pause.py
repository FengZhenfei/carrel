"""Per-KB parsing pause preserves jobs, caches and indexed content across resume."""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from kb_pipeline import db, discovery
from kb_pipeline.parsers.errors import JobPaused
from kb_pipeline.pipeline import parse_job, scheduler, worker
from kb_pipeline.vision import vlm
from kb_server import api, service
from kb_server.main import create_app
from test_scanner_worker import _parse_harness, _rescan
from test_manual_reparse_cache import mineru  # noqa: F401 — shared service/cache fixture
from _support import _block


def pause(h):
    discovery.set_config(h.con, h.kb_id, {"parse_paused": True})
    h.con.commit()


def job(h):
    return h.con.execute("SELECT * FROM jobs WHERE job_id=?", (h.job_id,)).fetchone()


def test_pause_api_survives_reconnect_and_scan_without_duplicate_jobs(tmp_path, monkeypatch):
    with _parse_harness(str(tmp_path)) as h, monkeypatch.context() as scoped:
        scoped.setattr(service, "settings", lambda: h.settings)
        kicks = []
        scoped.setattr(service, "kick_scan", lambda: kicks.append("scan") or "systemd")
        scoped.setattr(service, "kick_worker", lambda: kicks.append("worker") or "systemd")
        client = TestClient(create_app())
        url = f"/api/kbs/{h.kb_id}"
        for _ in range(2):
            assert client.post(url + "/parse_pause").json() == {"paused": True, "pausing": False}
        with db.connect(h.settings.state_db) as fresh:
            assert db.parse_is_paused(fresh, h.kb_id)
            assert db.claim_job_for_types(fresh, "another-worker", lease_seconds=60) is None
        _rescan(h)
        assert h.con.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
        assert job(h)["status"] == "queued"
        assert kicks == []
        assert service.kb_files(h.kb_id)[0]["parse_paused"]
        assert service.job_detail(h.job_id)["job"]["parse_paused"]
        stats = service._kb_stats(h.con, h.kb_id)
        assert (stats["jobs_active"], stats["jobs_waiting"]) == (0, 0)
        reply = client.post(url + "/parse_now").json()
        assert reply["resumed"] and kicks == ["scan", "worker"]
        assert not db.parse_is_paused(h.con, h.kb_id)
        assert worker.run_once(h.con, settings=h.settings).startswith("parse-done:")
        assert job(h)["status"] == "done"
        assert not service.job_detail(h.job_id)["job"]["parse_paused"]


@pytest.mark.parametrize("status", ["queued", "retry", "running"])
def test_pause_blocks_claims_but_not_other_kbs_or_lifecycle_work(tmp_path, status):
    with _parse_harness(str(tmp_path)) as h:
        pause(h)
        h.con.execute("UPDATE jobs SET status=?, locked_until=0 WHERE job_id=?", (status, h.job_id))
        for jid, kb_id, kind in (("other", "kb_other", "parse"), ("metadata", h.kb_id, "metadata_update")):
            h.con.execute("INSERT INTO jobs(job_id,kb_id,collection,job_type,status,created_at,updated_at) "
                          "VALUES(?,?,?,?,'queued',0,0)", (jid, kb_id, kb_id, kind))
        h.con.commit()
        ids = {db.claim_job_for_types(h.con, "worker", lease_seconds=60)["job_id"] for _ in range(2)}
        assert ids == {"other", "metadata"}
        assert db.claim_job_for_types(h.con, "worker", lease_seconds=60) is None
        assert job(h)["status"] == status


@pytest.mark.parametrize("step,raises", [("parse", False), ("embed", False), ("embed", True)])
def test_pause_before_commit_preserves_job_without_failure_or_retry(tmp_path, step, raises):
    with _parse_harness(str(tmp_path)) as h:
        def stop():
            pause(h)
            if raises:
                raise RuntimeError("service restarting")
        h.hooks[step] = stop
        result = worker.run_once(h.con, settings=h.settings)
        assert result == f"parse-paused:{h.job_id}"
        assert (job(h)["status"], job(h)["retry_count"], job(h)["locked_by"], job(h)["error"]) == ("queued", 0, None, None)
        assert h.con.execute("SELECT COUNT(*) FROM failures").fetchone()[0] == 0
        assert "upsert" not in h.calls
        assert worker.run_once(h.con, settings=h.settings) == "no-job"
        h.hooks.clear()
        discovery.set_config(h.con, h.kb_id, {"parse_paused": None})
        h.con.commit()
        assert worker.run_once(h.con, settings=h.settings).startswith("parse-done:")
        assert h.con.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1


@pytest.mark.parametrize("fail_after_commit", [False, True])
def test_pause_during_index_commit_does_not_leave_half_written_state(tmp_path, fail_after_commit):
    with _parse_harness(str(tmp_path)) as h:
        h.hooks["upsert"] = lambda: pause(h)
        if fail_after_commit:
            def fail(): raise RuntimeError("write interrupted")
            h.hooks["mark_old"] = fail
        result = worker.run_once(h.con, settings=h.settings)
        if fail_after_commit:
            assert result.startswith("retry:")
            assert job(h)["retry_count"] == 1
        else:
            assert result.startswith("parse-done:")
            assert h.calls[-4:] == ["upsert", "mark_old", "mark_stale", "fts"]
            assert db.get_file_by_id(h.con, h.file_id)["indexed_version"]
        assert worker.run_once(h.con, settings=h.settings) == "no-job"


@pytest.mark.parametrize("cancel", [False, True])
def test_pause_respects_cancellation_and_lease_takeover(tmp_path, cancel):
    with _parse_harness(str(tmp_path)) as h:
        def stop():
            pause(h)
            if cancel:
                db.request_job_cancel(h.con, h.job_id, "cancelled by operator")
            else:
                h.con.execute("UPDATE jobs SET locked_by='another-worker' WHERE job_id=?", (h.job_id,))
                h.con.commit()
            raise RuntimeError("service stopping")
        h.hooks["embed"] = stop
        result = worker.run_once(h.con, settings=h.settings)
        assert result.startswith("cancelled:" if cancel else "superseded:")
        assert job(h)["status"] == ("cancelled" if cancel else "running")
        if not cancel:
            assert job(h)["locked_by"] == "another-worker"
        assert h.con.execute("SELECT COUNT(*) FROM failures").fetchone()[0] == 0


def test_manual_reparse_pause_keeps_cache_run_and_old_index(tmp_path, monkeypatch, mineru):
    with _parse_harness(str(tmp_path / "pipeline")) as h, monkeypatch.context() as scoped:
        scoped.setattr(service, "settings", lambda: h.settings)
        scoped.setattr(service, "kick_worker", lambda: "systemd")
        scoped.setattr(service, "kick_scan", lambda: "systemd")
        def parse(*args, **kwargs):
            data = mineru.call()
            return [_block("body", data["content_list"][0]["text"])]
        scoped.setattr(parse_job, "_parse_blocks", parse)
        assert worker.run_once(h.con, settings=h.settings).startswith("parse-done:")
        h.con.commit()
        old_chunks = [dict(r) for r in h.con.execute("SELECT * FROM chunks WHERE status='active'")]
        h.job_id = service.retry_file(h.file_id)["job_id"]
        assert json.loads(job(h)["payload_json"])["refresh_parse_cache"]
        h.hooks["embed"] = lambda: pause(h)
        assert worker.run_once(h.con, settings=h.settings) == f"parse-paused:{h.job_id}"
        assert len(mineru.posts) == 2
        assert [dict(r) for r in h.con.execute("SELECT * FROM chunks WHERE status='active'")] == old_chunks
        assert scheduler.requeue_single_file(h.con, ingest_run_id="duplicate", file_row=db.get_file_by_id(h.con, h.file_id)) is None
        h.con.commit()
        h.hooks.clear()
        assert service.parse_now(h.kb_id)["resumed"]
        assert worker.run_once(h.con, settings=h.settings).startswith("parse-done:")
        assert len(mineru.posts) == 2  # the completed MinerU result belongs to the same manual run
        assert h.con.execute("SELECT COUNT(*) FROM chunks WHERE status='active'").fetchone()[0] == 1
        assert h.args["upsert"][4][0]["text"] == "revision 2"


def test_pause_cancels_unsent_visual_requests_and_joins_inflight_calls(monkeypatch):
    called = []
    stop = threading.Event()
    def caption(*, image_path, **kwargs):
        called.append(image_path)
        if len(called) > 1:
            assert stop.wait(5)
        return dict(vlm.EMPTY_RESULT)
    def progress(done, total):
        stop.set()
        raise JobPaused("Parsing paused")
    monkeypatch.setattr(vlm, "caption_image", caption)
    jobs = [(str(i), Path(f"/synthetic/{i}.png"), None, None) for i in range(30)]
    with pytest.raises(JobPaused):
        vlm.caption_images_parallel(jobs, base_url="http://example.invalid", api_key="synthetic",
                                    model_id="model", concurrency=2, progress_cb=progress)
    assert len(called) < len(jobs)


@pytest.mark.parametrize("value", [None, True, False, "true"])
def test_pause_flag_is_an_action_not_an_editable_config(value):
    with pytest.raises(HTTPException) as raised:
        api._reject_bad_config({"parse_paused": value})
    assert raised.value.status_code == 422


def test_pause_route_rejects_unknown_and_inactive_kbs(tmp_path, monkeypatch):
    with _parse_harness(str(tmp_path)) as h, monkeypatch.context() as scoped:
        scoped.setattr(service, "settings", lambda: h.settings)
        client = TestClient(create_app())
        assert client.post("/api/kbs/missing/parse_pause").status_code == 404
        h.con.execute("UPDATE kb_sources SET status='inactive' WHERE kb_id=?", (h.kb_id,))
        h.con.commit()
        assert client.post(f"/api/kbs/{h.kb_id}/parse_pause").status_code == 422
        assert not db.parse_is_paused(h.con, h.kb_id)
