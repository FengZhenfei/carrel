"""Standard storage contracts exercised with temporary data, never a live KB."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch
import time
import uuid

import pytest
from qdrant_client import QdrantClient

from kb_pipeline import db, discovery, search_fts
from kb_pipeline.graph import build
from kb_pipeline.pipeline import parse_job, worker
from kb_pipeline.storage import get_storage
from kb_pipeline.storage import standard
from kb_pipeline.vector.layout import VectorLayout
from kb_search import channels
from test_scanner_worker import _parse_harness, _rescan


def test_factory_is_lazy_reuses_connections_and_rejects_unknown_backend():
    settings = SimpleNamespace(qdrant_url="http://unused", qdrant_api_key="secret", state_db=None)
    with patch.object(standard, "vector_client", return_value=object()) as create:
        first = get_storage(settings, query_timeout=7)
        create.assert_not_called()
        # All concurrent facets share one connection on their first use.
        with ThreadPoolExecutor(max_workers=8) as pool:
            clients = list(pool.map(lambda _: first._q, range(24)))
        assert all(client is clients[0] for client in clients)
        create.assert_called_once_with(settings.qdrant_url, settings.qdrant_api_key, timeout=7)
        refreshed = get_storage(settings, query_timeout=7, previous=first)
        assert refreshed._q is first._q
        changed = get_storage(SimpleNamespace(qdrant_url="http://other", qdrant_api_key="secret", state_db=None),
                              query_timeout=7, previous=first)
        assert changed._client is None
    with pytest.raises(ValueError, match="not implemented"):
        get_storage(SimpleNamespace(storage_backend="unsupported"))


@pytest.fixture
def local_store():
    q = QdrantClient(":memory:")
    store = standard.StandardStorage(SimpleNamespace(opensearch_url="http://unused"), vector_client=q)
    store.chunks.ensure("kb_synthetic", VectorLayout(4, 4))
    try:
        yield store, q
    finally:
        q.close()


def test_vectors_filters_versions_and_preview_follow_original_contract(local_store):
    store, q = local_store
    ids = [str(uuid.uuid4()) for _ in range(3)]
    payloads = [dict(kb_id="kb_synthetic", file_key=i, doc_id=f"kb_synthetic:{i}", content_version="v1",
                     is_active=True, chunk_uid=f"c{i}", chunk_index=0, block_type="text",
                     rel_path=f"folder/{i}.md", text=f"Synthetic evidence {i}") for i in range(3)]
    vectors = [{"text": [1., 0., 0., 0.], "visual": [0., 1., 0., 0.]} for _ in ids]
    store.chunks.write("kb_synthetic", ids, vectors, payloads, max_bytes=4096)
    # Both semantic and visual candidates preserve scores, payloads and order.
    for visual in (False, True):
        fn = channels.visual_channel if visual else channels.vector_channel
        vec = [0., 1., 0., 0.] if visual else [1., 0., 0., 0.]
        assert store.chunks.vector_candidates("kb_synthetic", vec, limit=10, visual=visual) == fn(
            q, "kb_synthetic", vec, limit=10)
        scoped = store.chunks.vector_candidates("kb_synthetic", vec, limit=10, visual=visual,
                                                hints={"doc_ids": ["kb_synthetic:0", "kb_synthetic:1"]}, excluded=["folder/1.md"])
        assert [row["point_id"] for row in scoped] == [ids[0]]
    store.chunks.retire_previous("kb_synthetic", kb_id="kb_synthetic", file_key=0, active_content_version="v2")
    assert store.chunks.active_count("kb_synthetic") == 2
    assert ids[0] not in store.chunks.payloads("kb_synthetic", ids)
    assert store.chunks.payloads("kb_synthetic", ids, active_only=False)[ids[0]]["text"] == payloads[0]["text"]
    assert store.chunks.metadata("kb_synthetic", [ids[0]])[ids[0]]["active"] is False
    # A filter that used to match the old version cannot surface it after replacement.
    assert store.chunks.vector_candidates("kb_synthetic", [1., 0., 0., 0.], limit=10,
                                          hints={"doc_ids": ["kb_synthetic:0"]}) == []


@pytest.mark.parametrize("filename,text,expected", [
    ("notes.md", "# Operations\n\nThe synthetic ORION valve operates at 12 kPa.\n\n"
                 "## Inspection\n\nInspect the seal before every restart.", ["ORION", "12 kPa", "Inspect the seal"]),
    ("readme.md", "# Example\n\nThe helper returns a calibration code.\n\n"
                  "```python\ndef calibration_code():\n    return 'ORION-27'\n```", ["calibration_code", "ORION-27"]),
    ("measurements.csv", "item,value,unit\nORION,12,kPa\nVEGA,27,kPa\n", ["ORION", "VEGA", "27"]),
    ("guide.txt", "Synthetic calibration guide.\nORION uses 12 kPa.\nVEGA uses 27 kPa.\n", ["ORION", "VEGA"]),
])
def test_parse_retrieve_graph_input_replace_and_empty(tmp_path, filename, text, expected):
    """Real parsers, chunking, SQLite ledger and Qdrant; deterministic embeddings and OS transport."""
    real_parse = parse_job._parse_blocks
    real_sync = search_fts.sync_doc_from_qdrant
    functions = {name: getattr(standard.qdrant, name) for name in (
        "collection_exists", "ensure_collection", "validate_collection_layout", "upsert_chunks",
        "mark_old_versions_inactive", "mark_stale_file_points_inactive")}
    q = QdrantClient(":memory:")
    projection = {}
    class KeywordTransport:
        indices = SimpleNamespace(exists=lambda **kw: True, refresh=lambda **kw: None)

        def delete_by_query(self, *, index, body, **kw):
            doc = body["query"]["term"]["doc_id"]
            doomed = [key for key, value in projection.items() if value["doc_id"] == doc]
            for key in doomed:
                del projection[key]
            return {"deleted": len(doomed)}

    def bulk(_client, actions):
        for action in actions:
            projection[action["_id"]] = action["_source"]
        return len(actions)

    try:
        with _parse_harness(str(tmp_path), filename=filename, text=text) as h, ExitStack() as st:
            h.settings.vlm_temperature = 0.1
            h.settings.vlm_top_p = 0.8
            h.settings.vlm_max_tokens = 64
            h.settings.vlm_structured_output = True
            h.settings.graph_work_dir = tmp_path / "graph"
            st.enter_context(patch.object(parse_job, "_parse_blocks", real_parse))
            st.enter_context(patch.object(standard, "vector_client", return_value=q))
            for name, fn in functions.items():
                st.enter_context(patch.object(standard.qdrant, name, fn))
            st.enter_context(patch.object(search_fts, "sync_doc_from_qdrant", real_sync))
            st.enter_context(patch.object(search_fts, "client", return_value=KeywordTransport()))
            st.enter_context(patch.object(search_fts, "ensure_index"))
            st.enter_context(patch.object(search_fts, "_bulk_index", side_effect=bulk))
            assert worker.run_once(h.con, settings=h.settings).startswith("parse-done:")
            h.con.commit()
            source = discovery.enrolled_sources(h.settings.state_db, h.settings.mirror_root)[h.kb_id]
            store = get_storage(h.settings, vector_client=q)
            refs = db.active_chunk_refs(h.con, source.collection)
            ids = [row["point_id"] for row in refs]
            payloads = store.chunks.payloads(source.collection, ids)
            rendered = "\n".join(row["text"] for row in payloads.values())
            for phrase in expected:
                assert phrase in rendered
                assert phrase in "\n".join(row["body"] for row in projection.values())
            hits = store.chunks.vector_candidates(source.collection, [1., 0., 0., 0.], limit=100)
            assert {row["point_id"] for row in hits} == set(ids)
            paths = build.graph_paths(h.settings, source, "synthetic-v1")
            manifest, frozen, units = build.prepare_graph_input(h.settings, source, paths=paths, storage=store)
            assert manifest["active_chunks"] == len(refs)
            assert {row["point_id"] for row in frozen} == set(ids)
            assert units and set().union(*(set(unit.point_ids) for unit in units)) == set(ids)

            # The next parse replaces active content in all projections; retained history remains readable only explicitly.
            path = h.settings.mirror_root / "库A" / filename
            path.write_text(text.replace("ORION", "SIRIUS"), encoding="utf-8")
            _rescan(h)
            assert worker.run_once(h.con, settings=h.settings).startswith("parse-done:")
            h.con.commit()
            assert all(not row.get("is_active") for row in store.chunks.payloads(source.collection, ids, active_only=False).values())
            assert "ORION" not in "\n".join(row["body"] for row in projection.values())
            assert "SIRIUS" in "\n".join(row["body"] for row in projection.values())
            path.write_text("", encoding="utf-8")
            _rescan(h, "empty")
            assert worker.run_once(h.con, settings=h.settings).startswith("parse-empty:")
            h.con.commit()
            assert store.chunks.active_count(source.collection) == 0
            assert projection == {}
            assert db.active_chunk_refs(h.con, source.collection) == []
            with pytest.raises(build.NoGraphCorpus):
                build.prepare_graph_input(h.settings, source, paths=paths, storage=store)
    finally:
        q.close()


def test_storage_deadline_preserves_hard_stop(local_store):
    store, _ = local_store
    expired = store.with_deadline(time.monotonic() - 1)
    with pytest.raises(TimeoutError):
        expired.chunks.payloads("kb_synthetic", [str(uuid.uuid4())])
