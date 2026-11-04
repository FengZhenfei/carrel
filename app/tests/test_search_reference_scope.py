"""Source and graph versions stay bound across follow-up calls."""
from types import SimpleNamespace
from unittest.mock import Mock, patch
import sqlite3
import uuid

from fastapi import HTTPException
from PIL import Image
import pytest

from kb_search import api, graphwalk, library, service


def test_versions_must_match_both_the_active_document_and_the_graph_input(tmp_path):
    state = tmp_path / "versions.sqlite"
    with sqlite3.connect(state) as con:
        con.executescript("""
        CREATE TABLE files(file_id TEXT, file_key INTEGER, kb_id TEXT, status TEXT);
        CREATE TABLE chunks(file_id TEXT, collection TEXT, content_version TEXT, status TEXT);
        CREATE TABLE graph_builds(graph_build_id TEXT, source_collection TEXT, graph_version TEXT);
        CREATE TABLE graph_build_chunks(graph_build_id TEXT, doc_id TEXT, content_version TEXT);
        INSERT INTO files VALUES ('file-a', 1, 'sample', 'indexed'), ('file-b', 2, 'sample', 'indexed');
        INSERT INTO chunks VALUES ('file-a','c','v2','active'),('file-b','c','v1','active');
        INSERT INTO graph_builds VALUES ('build-old','c','g1'),('build-new','c','g2');
        INSERT INTO graph_build_chunks VALUES ('build-old','sample:1','v1'),('build-new','sample:1','v2');
        """)
    settings = SimpleNamespace(state_db=state)
    source = SimpleNamespace(kb_id="sample", collection="c")
    library.require_document_version(settings, source, "sample:1", "v2", graph_version="g2")
    for version, graph in [("v1", None), ("v2", "g1"), ("v2", "unavailable")]:
        with pytest.raises(HTTPException) as error:
            api._wrap(library.require_document_version, settings, source, "sample:1", version, graph_version=graph)
        assert error.value.status_code == 409

    client = Mock()
    client.indices.exists.return_value = True
    client.msearch.return_value = {"responses": [{"hits": {"hits": [], "total": {"value": 0}}}]}
    library.grep(settings, source, ["term"], doc_ids=["sample:1", "sample:2"],
                 doc_versions={"sample:1": "v2", "sample:2": "v1"}, client=client)
    filters = client.msearch.call_args.kwargs["body"][1]["query"]["bool"]["filter"]
    assert {"bool": {"minimum_should_match": 1, "should": [
        {"bool": {"filter": [{"term": {"doc_id": "sample:1"}}, {"term": {"content_version": "v2"}}]}},
        {"bool": {"filter": [{"term": {"doc_id": "sample:2"}}, {"term": {"content_version": "v1"}}]}}]}} in filters


@pytest.mark.parametrize("call,kwargs", [(graphwalk.neighbors, {"entity": None, "entity_id": "e"}),
                                          (graphwalk.list_facts, {"subject_id": "e"})])
def test_stale_graph_reference_is_rejected_before_resolving_ids(call, kwargs):
    driver = Mock()
    with patch("kb_pipeline.graph.neo4j_import.active_neo4j_graph_version", return_value="new"):
        with pytest.raises(library.StaleReference, match="graph_version_changed"):
            call(SimpleNamespace(), SimpleNamespace(kb_id="sample"), driver=driver,
                 expected_graph_version="old", **kwargs)
    driver.session.assert_not_called()


def test_images_and_crops_apply_the_same_document_scope(tmp_path):
    source = SimpleNamespace(kb_id="sample", collection="c")
    settings = SimpleNamespace(cache_dir=tmp_path, mirror_root=tmp_path,
                               corrections_dir=tmp_path, sources={"sample": source})
    (tmp_path/"sample.yaml").write_text("documents:\n  - path: report.pdf\n    valid_until: 2020-12-31\n")
    Image.new("RGB", (16, 16)).save(tmp_path/"figure.png")
    pid = str(uuid.uuid4())
    q = Mock()
    q.retrieve.return_value = [SimpleNamespace(payload={"is_active": True, "rel_path": "report.pdf", "visual_ref": "figure.png"})]
    with patch.object(service, "runtime", return_value=(settings, None, q)):
        for fn, args in [(service.image, ()), (service.crop, ([0, 0, .5, .5],))]:
            with pytest.raises(KeyError, match="scope"):
                fn("sample", pid, *args)
            assert fn("sample", pid, *args, history=True)["bytes"]
            assert fn("sample", pid, *args, as_of="2020")["bytes"]
            with pytest.raises(KeyError, match="scope"):
                fn("sample", pid, *args, as_of="2022")


@pytest.mark.parametrize("mode", ["off", "discovery"])
def test_text_profile_preserves_graph_off_and_reports_delivered_evidence(monkeypatch, mode):
    from dataclasses import make_dataclass
    from kb_search import readback
    import test_search

    def fixture_settings(**fields):
        if "sources" not in fields:
            return SimpleNamespace(**fields)
        fields.setdefault("visual_reranker_base_url", "")
        return make_dataclass("FixtureSettings", list(fields))(**fields)

    monkeypatch.setattr(test_search, "SimpleNamespace", fixture_settings)
    supplement = {"n": 12, "kb_id": "kb_001", "point_id": "supplement", "role": "graph_followup",
                  "text": "A separately verified original.", "accepted": True}
    with patch.object(readback, "read_originals", return_value=([supplement], {"added": 1})) as probe:
        result, calls = test_search.OrchestrationTests()._run(
            kbs=["kb_001"], settings_over={"graph_mode": mode, "text_discovery": True, "rerank_threshold": 0.1},
            rerank_fn=lambda docs: [0.0001] * len(docs))
    summary = result["retrieval_summary"]
    assert summary["graph"]["mode"] == mode
    if mode == "off":
        assert not calls["graph"] and not probe.called and summary["no_relevant_content"]
    else:
        assert probe.called and summary["primary_evidence_state"] == "diagnostic"
        assert summary["evidence_state"] == "accepted" and not summary["no_relevant_content"]
