"""Tier-1 fixes of 2026-11-20 in the search service.

D1: with the text reranker unavailable, one picture passing its threshold made the service return pictures only;
    candidates without a verdict were dropped together with the rejected ones.
D2: stitching a short hit skipped a neighbour that did not fit and took a farther chunk, leaving a hole in a text
    that read as continuous.
D4: a derived row (entity, relation, fact) was dropped from the current scope as soon as one of its source
    documents was archived or superseded, although the others were current.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest import mock

import pytest

import test_dual_rerank as dual            # candidate / run helpers and the env fixture of the dual-ranking tests
import test_search as search_tests         # the orchestration harness (OrchestrationTests._run)
from kb_pipeline.graph import corrections
from kb_search import docstatus, service
from kb_search.evidence import STITCHABLE_BLOCKS, stitch_short_hit

env = dual.env                             # the same fixture, under the name pytest looks up

# ── D1 ────────────────────────────────────────────────────────────────────


def _unscored(*rows):
    for row in rows:
        row.pop("score_rerank")
    return list(rows)


def test_unranked_text_stays_next_to_an_accepted_picture(env, monkeypatch):
    monkeypatch.setattr(service, "rerank", mock.Mock(side_effect=ConnectionError()))
    result = dual.run(env, _unscored(dual.candidate("t1"), dual.candidate("image", image=True), dual.candidate("t2")))
    ordered = result["ordered"]
    assert [c["point_id"] for c in ordered] == ["image", "t1", "t2"]         # the accepted picture first at an equal rank
    assert [c["accepted"] for c in ordered] == [True, None, None]            # unknown is not rejected and is not accepted
    assert [c["ranking"]["branch"] for c in ordered] == ["visual", "unranked", "unranked"]
    assert result["status"] == "skipped"                                     # the text ranking did not happen: say so
    assert result["ranking"]["text"]["status"] == "skipped"
    assert result["error"] == "rerank: ConnectionError"
    assert result["evidence_state"] == "accepted"


def test_a_rejected_picture_is_still_dropped_while_unranked_text_stays(env, monkeypatch):
    monkeypatch.setattr(service, "rerank", mock.Mock(side_effect=ConnectionError()))
    env[2].return_value["scores"] = {"image": 0.8, "weak": 0.01}
    rows = _unscored(dual.candidate("t1"), dual.candidate("image", image=True), dual.candidate("weak", image=True))
    for row in rows:
        row["scores"].pop("score_visual", None)                              # a text question: no visual-recall rescue
    result = dual.run(env, rows)
    assert [c["point_id"] for c in result["ordered"]] == ["image", "t1"]
    assert next(c for c in result["cands"] if c["point_id"] == "weak")["accepted"] is False


def test_without_an_accepted_picture_the_fallback_is_unchanged(env, monkeypatch):
    monkeypatch.setattr(service, "rerank", mock.Mock(side_effect=ConnectionError()))
    env[2].return_value["scores"] = {"image": 0.01}
    rows = _unscored(dual.candidate("t1"), dual.candidate("image", image=True))
    for row in rows:
        row["scores"].pop("score_visual", None)
    result = dual.run(env, rows)
    assert {c["point_id"] for c in result["ordered"]} == {"t1", "image"}     # nothing accepted: everything in recall order
    assert result["status"] == "skipped" and result["evidence_state"] == "unranked"


def test_with_both_rankers_working_the_status_stays_ok(env):
    result = dual.run(env, [dual.candidate("text"), dual.candidate("image", image=True)])
    assert result["status"] == "ok"
    assert all(c["accepted"] is True for c in result["ordered"])


# ── D2 ────────────────────────────────────────────────────────────────────

KIND = sorted(STITCHABLE_BLOCKS)[0]


def _chunk(index: int, text: str) -> dict:
    return {"chunk_index": index, "block_type": KIND, "text": text, "point_id": f"p{index}", "page_idx": 1}


def test_stitching_stops_at_a_neighbour_that_does_not_fit():
    row = _chunk(10, "命中的短句。")
    neighbors = [_chunk(11, "紧邻的下一片，" + "很长" * 200), _chunk(12, "再下一片，较短。"), _chunk(9, "上一片，较短。")]
    assert stitch_short_hit(row, neighbors, min_chars=40, max_chars=120)
    assert [p["chunk_index"] for p in row["stitched"]["pieces"]] == [9, 10]
    assert (row["stitched"]["chunk_from"], row["stitched"]["chunk_to"]) == (9, 10)
    assert "再下一片" not in row["text"]              # chunk 12 lies behind chunk 11, which was left out


def test_a_side_that_is_closed_does_not_stop_the_other_side():
    row = _chunk(10, "命中的短句。")
    neighbors = [_chunk(11, "很长" * 200), _chunk(12, "再下一片。"), _chunk(9, "上一片，较短。"), _chunk(8, "更前一片，较短。")]
    assert stitch_short_hit(row, neighbors, min_chars=400, max_chars=120)
    assert [p["chunk_index"] for p in row["stitched"]["pieces"]] == [8, 9, 10]


def test_stitching_still_alternates_while_neighbours_fit():
    row = _chunk(10, "命中的短句。")
    neighbors = [_chunk(11, "下一片。" * 3), _chunk(12, "再下一片。" * 3), _chunk(9, "上一片。" * 3), _chunk(8, "更前一片。" * 3)]
    assert stitch_short_hit(row, neighbors, min_chars=60, max_chars=400)
    assert [p["chunk_index"] for p in row["stitched"]["pieces"]] == [8, 9, 10, 11, 12]


# ── D4 ────────────────────────────────────────────────────────────────────

ENTITIES = [
    {"title": "浓度", "type": "measurement", "score": 0.7, "hop": 0, "via": "lexical",
     "docs": ["dir/d1.pdf", "dir/d2.pdf"], "point_ids": ["p1", "p4"]},           # p1 is in d1, p4 in d2
    {"title": "液温", "type": "measurement", "score": 0.6, "hop": 0, "via": "lexical",
     "docs": ["dir/d2.pdf"], "point_ids": ["p4"]},
]
SPECS = [
    {"id": "s1", "subject": "李", "property": "浓度", "value": "433", "score": 0.9, "rel_path": "dir/d1.pdf", "point_ids": ["p1", "p4"]},
    {"id": "s2", "subject": "李", "property": "液温", "value": "5.3", "score": 0.8, "rel_path": "dir/d2.pdf", "point_ids": ["p4", "p1"]},
]


def _recheck(rows):
    """The status recheck on its own, with one archived folder in the knowledge base."""
    from kb_pipeline.graph.currency import query_scope

    corr = corrections.parse_corrections({"documents": [{"path": "old/*", "status": "archived"}]})
    assert not corr["errors"]
    source = SimpleNamespace(kb_id="kb1")
    settings = SimpleNamespace(sources={"kb1": source})
    meta = {"c1": {"rel_path": "cur/a.pdf"}, "c2": {"rel_path": "cur/b.pdf"}, "o1": {"rel_path": "old/x.pdf"}}
    rows = [{"kb_id": "kb1", **row} for row in rows]
    with mock.patch.object(docstatus, "corrections_of", return_value=corr):
        service._recheck_source_status(settings, rows, meta, query_scope())
    return rows


def test_status_recheck_keeps_a_row_with_a_confirmed_source_and_trims_the_rest():
    entity, lapsed, unknown, mixed_unknown, untouched, no_points = _recheck([
        {"title": "kept", "point_ids": ["c1", "o1", "c2"], "docs": ["cur/a.pdf", "old/x.pdf", "cur/b.pdf"]},
        {"title": "archived only", "point_ids": ["o1"], "docs": ["old/x.pdf"]},
        {"title": "no metadata at all", "point_ids": ["gone"]},
        {"title": "one confirmed, one without metadata", "point_ids": ["c1", "gone"]},
        {"title": "all current", "point_ids": ["c1", "c2"], "docs": ["cur/a.pdf", "cur/b.pdf"]},
        {"title": "no source points"},
    ])
    assert "_status_excluded" not in entity
    assert (entity["point_ids"], entity["docs"]) == (["c1", "c2"], ["cur/a.pdf", "cur/b.pdf"])
    assert lapsed["_status_excluded"] is True
    assert unknown["_status_excluded"] is True                      # nothing confirmed: excluded, as before
    assert "_status_excluded" not in mixed_unknown and mixed_unknown["point_ids"] == ["c1"]
    assert untouched == {"kb_id": "kb1", "title": "all current", "point_ids": ["c1", "c2"], "docs": ["cur/a.pdf", "cur/b.pdf"]}
    assert no_points == {"kb_id": "kb1", "title": "no source points"}


def test_status_recheck_still_excludes_a_fact_whose_own_document_is_not_allowed():
    own_current, own_archived = _recheck([
        {"id": "s1", "rel_path": "cur/a.pdf", "point_ids": ["c1", "o1"]},
        {"id": "s2", "rel_path": "old/x.pdf", "point_ids": ["o1", "c1"]},
    ])
    assert "_status_excluded" not in own_current and own_current["point_ids"] == ["c1"]
    assert own_archived["_status_excluded"] is True


def test_status_recheck_changes_nothing_in_history_scope_or_without_statuses():
    from kb_pipeline.graph.currency import query_scope

    source = SimpleNamespace(kb_id="kb1")
    settings = SimpleNamespace(sources={"kb1": source})
    meta = {"c1": {"rel_path": "cur/a.pdf"}, "o1": {"rel_path": "old/x.pdf"}}
    archived = corrections.parse_corrections({"documents": [{"path": "old/*", "status": "archived"}]})
    for corr, scope in ((archived, query_scope(history=True)), (corrections.parse_corrections(None), query_scope())):
        rows = [{"kb_id": "kb1", "point_ids": ["c1", "o1", "gone"], "docs": ["cur/a.pdf", "old/x.pdf"]}]
        with mock.patch.object(docstatus, "corrections_of", return_value=corr):
            service._recheck_source_status(settings, rows, meta, scope)
        assert rows == [{"kb_id": "kb1", "point_ids": ["c1", "o1", "gone"], "docs": ["cur/a.pdf", "old/x.pdf"]}]


def _search(*, archived: bool):
    run = lambda: search_tests.OrchestrationTests()._run(kbs=["kb_001"], hoods=[], graph_entities=[dict(e) for e in ENTITIES],
                                                        graph_specs=[dict(s) for s in SPECS])[0]
    if not archived:
        return run()
    corr = corrections.parse_corrections({"documents": [{"path": "dir/d2.pdf", "status": "archived"}]})
    assert not corr["errors"]
    with mock.patch.object(docstatus, "corrections_of", return_value=corr), \
            mock.patch.object(docstatus, "_doc_paths", return_value=["dir/d1.pdf", "dir/d2.pdf", "dir/d8.pdf", "dir/d9.pdf"]):
        return run()


# The two tests below run the whole search() through the orchestration harness of test_search.py. That harness
# hands search() a SimpleNamespace as settings, and search() calls dataclasses.replace on it, which accepts a
# SimpleNamespace only from Python 3.13 on. On 3.12 (the production image) the harness cannot run at all; the
# three tests above cover the same rule there.
needs_harness = pytest.mark.skipif(sys.version_info < (3, 13),
                                   reason="the orchestration harness of test_search.py needs Python 3.13+")


@needs_harness
def test_a_row_keeps_its_current_sources_when_one_source_document_is_archived():
    out = _search(archived=True)
    entities = {e["title"]: e for e in out["entities"]}
    assert set(entities) == {"浓度"}                           # 液温 rests on the archived document only
    assert entities["浓度"]["docs"] == ["dir/d1.pdf"]          # and the kept row no longer cites that document
    assert entities["浓度"]["sources_active"] == "1/1"
    specs = {s["id"]: s for s in out["specs"]}
    assert set(specs) == {"s1"}                                # s2 is an occurrence in the archived document itself
    assert specs["s1"]["point_ids"] == ["p1"]


@needs_harness
def test_without_document_statuses_nothing_is_trimmed():
    out = _search(archived=False)
    entities = {e["title"]: e for e in out["entities"]}
    assert set(entities) == {"浓度", "液温"}
    assert entities["浓度"]["docs"] == ["dir/d1.pdf", "dir/d2.pdf"]
    assert {s["id"]: s["point_ids"] for s in out["specs"]} == {"s1": ["p1", "p4"], "s2": ["p4", "p1"]}
