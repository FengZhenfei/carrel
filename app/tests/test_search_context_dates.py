"""Document dates in readback are metadata, not a fact-time filter."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from kb_search import docstatus, service


def readback(*, date=None, found="2023-10-12", error=None, as_of=None):
    payload = {"kb_id": "sample", "doc_id": "d", "point_id": "p", "content_version": "v",
               "rel_path": "report.md", "text": "A retrospective account of an earlier event.",
               "chunk_index": 0, "chunk_total": 1, "token_count": 10}
    if date:
        payload["doc_date"] = date
    store = SimpleNamespace(chunks=SimpleNamespace(context=Mock(return_value=[payload])),
                            graph=SimpleNamespace(document_dates=Mock(return_value={("sample", "d"): found} if found else {}, side_effect=error)),
                            catalog=Mock(return_value=[{"kb_id": "sample", "graph_version": "g"}]))
    store.graph.annotate_dates = lambda rows, scope, **kw: docstatus.annotate_dates(
        None, rows, scope, provider=store.graph.document_dates, **kw)
    settings = SimpleNamespace(sources={"sample": SimpleNamespace(collection="c", source_root="documents")})
    ss = SimpleNamespace(catalog_ttl=30, channel_timeout=5)
    with patch.object(service, "runtime", return_value=(settings, ss, store)), patch.object(docstatus, "allowed", return_value=True):
        result = service.context("sample", "d", 0, 0, content_version="v", as_of=as_of)
    assert result["sources"][0]["text"] == payload["text"]
    assert result["sources"][0]["content_version"] == "v"
    assert store.chunks.context.call_args.kwargs["content_version"] == "v"
    return result, store


def test_current_context_fills_date_from_existing_document_metadata():
    result, store = readback()
    assert result["sources"][0]["doc_date"] == "2023-10-12"
    assert store.graph.document_dates.call_args.args[1] == {"sample": "g"}
    assert result["time"]["mode"] == "current"


def test_existing_payload_date_is_preserved_without_an_extra_lookup():
    result, store = readback(date="2022-05-06")
    assert result["sources"][0]["doc_date"] == "2022-05-06"
    store.catalog.assert_not_called()
    store.graph.document_dates.assert_not_called()


@pytest.mark.parametrize("error", [None, RuntimeError("metadata unavailable")])
def test_unknown_or_unavailable_date_does_not_drop_originals(error):
    result, _ = readback(found=None, error=error)
    assert "doc_date" not in result["sources"][0]
    assert result["tokens_total"] == 10
    if error:
        assert result["time"]["degraded"] == "doc_dates: RuntimeError"


def test_asof_keeps_a_later_retrospective_source_and_its_existing_scope():
    result, store = readback(as_of="2023-01-01")
    assert result["sources"][0]["doc_date"] == "2023-10-12"
    assert result["sources"][0]["document_after_cutoff"] is True
    assert result["time"]["mode"] == "as_of"
    assert store.graph.document_dates.call_count == 1
