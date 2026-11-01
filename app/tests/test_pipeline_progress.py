"""Synthetic progress-contract cases; no models or external stores are needed."""
import json
from types import SimpleNamespace

from kb_pipeline import db
from kb_pipeline.progress import advance, job_view


def test_embedding_progress_does_not_change_batches_or_order(monkeypatch):
    from kb_pipeline.embedding import client

    calls = []
    def create(**kwargs):
        calls.append(kwargs["input"])
        return SimpleNamespace(data=[SimpleNamespace(embedding=[float(t)]) for t in kwargs["input"]])

    monkeypatch.setattr(client, "_client_for", lambda *_: SimpleNamespace(embeddings=SimpleNamespace(create=create)))
    embedder = client.EmbeddingClient(base_url="http://example.invalid", api_key="", model_id="example",
                                      dim=1, batch_size=2, sleep_seconds=0)
    progress = []
    vectors = embedder.embed(["1", "2", "3"], progress_cb=lambda d, n: progress.append((d, n)))
    assert calls == [["1", "2"], ["3"]]
    assert vectors == [[1.0], [2.0], [3.0]]
    assert progress == [(0, 2), (1, 2), (2, 2)]


def test_graph_stage_changes_bypass_throttle(monkeypatch):
    from kb_pipeline.graph import build

    monkeypatch.setattr(build.time, "time", lambda: 100)
    seen = []
    progress = build._Progress(seen.append)
    progress("Description summaries · entities 0/4")
    progress("Description summaries · entities 1/4")
    progress("Description summaries · relations 0/2")
    progress("Description summaries · relations 2/2", force=True)
    assert seen == ["Description summaries · entities 0/4", "Description summaries · relations 0/2",
                    "Description summaries · relations 2/2"]


def test_counters_keep_stage_start_and_models():
    first = advance(None, "Text embedding (0/4 batches)", 100, models={"embed_text": "example-model"})
    second = advance(first, "Text embedding (2/4 batches)", 120)
    assert second["started_at"] == 100
    assert second["updated_at"] == 120
    assert (second["completed"], second["total"]) == (2, 4)
    assert second["models"] == {"embed_text": "example-model"}
    next_stage = advance(second, "Writing vectors", 121)
    assert next_stage["started_at"] == 121
    assert "completed" not in next_stage
    assert next_stage["models"] == second["models"]


def test_graph_substeps_and_retry_do_not_share_counters():
    first = advance(None, "Description summaries · entities 10/10", 100)
    second = advance(first, "Description summaries · relations 0/7", 101)
    assert second["subphase"] == "relations"
    assert second["started_at"] == 101
    retry = advance(second, "Description summaries (attempt 2 failed, retrying in 10s)", 120)
    assert retry["wait_reason"] == "retry"
    assert retry["next_attempt_at"] == 130
    assert "completed" not in retry
    assert "wait_reason" not in advance(retry, "Description summaries", 130)


def test_waiting_is_reported_from_scheduler_state():
    p = advance(None, "Text embedding (1/4 batches)", 100)
    row = {"status": "retry", "progress_json": json.dumps(p), "next_attempt_at": 150,
           "error": "Service not ready, trying again in 30s (not counted as a retry): unavailable"}
    assert job_view(row, 120)["wait_reason"] == "service"
    row["error"] = "HTTP 429"
    assert job_view(row, 120)["wait_reason"] == "retry"
    row["cancel_requested"] = 1
    assert job_view(row, 120)["wait_reason"] == "stopping"
    row.update(status="running", cancel_requested=0)
    assert "wait_reason" not in job_view(row, 10000)


def test_progress_migration_and_attempt_boundaries(tmp_path):
    path = tmp_path / "state.db"
    db.init_db(path)
    db.init_db(path)
    with db.connect(path) as con:
        jid = db.enqueue_job(con, ingest_run_id=None, file_id=None, kb_id="example", collection="example",
                             file_key=None, job_type="parse", payload={})
        db.claim_job_for_types(con, "worker-example")
        db.set_job_stage(con, jid, "Parsing document")
        db.set_job_stage(con, jid, "Describing images (VLM 0/2)")
        db.set_job_stage(con, jid, "Describing images (VLM 2/2)")
        events = db.job_events(con, jid)
        assert len([e for e in events if e["kind"] == "stage"]) == 2
        row = con.execute("SELECT * FROM jobs WHERE job_id=?", (jid,)).fetchone()
        assert job_view(row, 100)["completed"] == 2
        db.mark_job_failed(con, jid, "temporary", retry=True, retry_delay_seconds=1)
        con.execute("UPDATE jobs SET next_attempt_at=0 WHERE job_id=?", (jid,))
        db.claim_job_for_types(con, "worker-example")
        db.set_job_stage(con, jid, "Describing images (VLM 0/2)")
        stages = [e for e in db.job_events(con, jid) if e["kind"] == "stage"]
        assert len(stages) == 3
        assert stages[-2]["text"].endswith("2/2)")
        assert stages[-1]["text"].endswith("0/2)")
