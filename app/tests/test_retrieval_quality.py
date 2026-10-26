"""Budget, provenance and model-failure contracts for retrieval improvements."""
import copy
import math
from unittest.mock import Mock

import pytest

from kb_pipeline.utils import count_tokens
from kb_search.fusion import admit_candidates, relevance_weight
from kb_search.text import relevant_excerpt, truncate_tokens
from kb_search.evidence import assemble_sources, select_hits
from kb_search import rerank as rank_client


def candidate(pid, score, *, image=False):
    return {"point_id": pid, "rrf": score, "recall_sources": ["visual"] if image else ["text"],
            "payload": {"text": "Example evidence.", "block_type": "image" if image else "text",
                        "visual_ref": "figure.png" if image else None}}


def test_admission_protects_groups_but_spends_remainder_on_recall_quality():
    strong = [candidate(f"a{i}", .05 - i / 10000) for i in range(60)]
    weak = [candidate(f"b{i}", .01) for i in range(60)]
    rows = admit_candidates([strong, weak], limit=64)
    assert len(rows) == 64
    assert {c["point_id"] for c in strong} <= {c["point_id"] for c in rows}
    assert {"b0", "b1"} <= {c["point_id"] for c in rows}
    # Widening reuses these objects and their completed model scores.
    assert next(c for c in rows if c["point_id"] == "a0") is strong[0]


def test_admission_has_one_shared_budget_with_visual_reserve_and_duplicates():
    a = [candidate(f"t{i}", .1) for i in range(80)]
    b = [candidate(f"v{i}", .001, image=True) for i in range(10)]
    rows = admit_candidates([a, a[:3] + b], limit=64, visual_limit=8)
    assert len(rows) == len({c["point_id"] for c in rows}) == 64
    assert sum(bool(c["payload"]["visual_ref"]) for c in rows) >= 8
    assert admit_candidates([a], limit=0) == []
    assert admit_candidates([], limit=64) == []


def test_relevant_excerpt_preserves_contiguous_original_and_condition():
    text = "Introduction and unrelated details.\n" * 80
    text += "Only while the replica is healthy.\nFailover recovery takes forty seconds.\nDo not restart the primary.\n"
    excerpt, start = relevant_excerpt(text, 40, "How long does failover recovery take?")
    assert start > 0 and excerpt == text[start:start + len(excerpt)]
    assert "Only while the replica is healthy" in excerpt
    assert "forty seconds" in excerpt
    assert count_tokens(excerpt) <= 40
    assert relevant_excerpt(text, 40, "")[0] == truncate_tokens(text, 40)


def test_late_unpunctuated_match_and_tiny_unicode_budget():
    text = "背景" * 300 + "目标系统恢复时间为三十秒"
    excerpt, start = relevant_excerpt(text, 30, "目标系统恢复时间")
    assert start > 0 and "目标系统" in excerpt
    for n in range(5):
        assert count_tokens(truncate_tokens("𐐀𐐀𐐀", n)) <= n


def test_assembly_excerpt_does_not_change_source_identity_or_budget():
    row = candidate("source", .1)
    row["payload"].update({"text": "Unrelated introduction.\n" * 200 + "The retention policy keeps records for ninety days.\n",
                           "kb_id": "kb_example", "doc_id": "doc-1", "content_version": "v1", "chunk_index": 7,
                           "page_idx": 4, "section_path": ["Retention"]})
    original = copy.deepcopy(row)
    rows, stats = assemble_sources([row], budget_tokens=40, neighbors=None, question="What is the retention policy?")
    source = rows[0]
    assert "ninety days" in source["text"]
    assert source["doc_id"] == "doc-1" and source["content_version"] == "v1" and source["page_idx"] == 4
    assert source["text_truncated"] and source["excerpt"]["char_start"] > 0
    assert stats["tokens_total"] <= 40 and row == original


@pytest.mark.parametrize("results", [[], [{"index": 0, "relevance_score": .5}],
    [{"index": 0, "relevance_score": .5}, {"index": 0, "relevance_score": .8}],
    [{"index": 0, "relevance_score": math.nan}, {"index": 1, "relevance_score": .8}],
    [{"index": 0}, {"index": 1, "relevance_score": .8}]])
def test_missing_or_invalid_text_scores_are_service_failures(monkeypatch, results):
    response = Mock(ok=True)
    response.json.return_value = {"results": results}
    monkeypatch.setattr(rank_client._session, "post", Mock(return_value=response))
    client = rank_client.Reranker("https://example.invalid/v1", model_id="example-ranker")
    with pytest.raises(ValueError):
        client.score("question", ["first", "second"])


def test_equal_valid_scores_remain_valid_evidence(monkeypatch):
    response = Mock(ok=True)
    response.json.return_value = {"results": [{"index": i, "relevance_score": .8} for i in range(2)]}
    monkeypatch.setattr(rank_client._session, "post", Mock(return_value=response))
    client = rank_client.Reranker("https://example.invalid/v1", model_id="example-ranker")
    assert client.score("question", ["first", "second"]) == [.8, .8]


def test_relevance_margin_uses_each_branch_gate_without_changing_acceptance():
    assert relevance_weight(.9, .1) == relevance_weight(.9, .5) == 1
    assert 0 < relevance_weight(.51, .5) < relevance_weight(.7, .5)
    assert relevance_weight(.5, .5) == 0
    assert relevance_weight(.8, 0) == relevance_weight(1, 1) == 1


def test_shared_subjects_cannot_consume_another_librarys_result_quota():
    rows = [{**candidate(str(i), .1), "kb_id": "library_a" if i < 8 else "library_b",
             "bucket": "shared_subject", "accepted": True, "score_final": 1 - i / 20} for i in range(10)]
    hits, stats = select_hits(rows, 6, buckets={"keys": ["shared_subject"]},
                              kb_keys=["library_a", "library_b"], mmr_lambda=None)
    assert [c["point_id"] for c in hits] == ["0", "1", "2", "3", "8", "9"]
    assert stats["kb_quota_filled"] == {"library_a": 2, "library_b": 2}
    rows[-1]["accepted"] = rows[-2]["accepted"] = False
    hits, stats = select_hits(rows[:8], 6, kb_keys=["library_a", "library_b"], mmr_lambda=None)
    assert len(hits) == 6 and stats["kb_quota_filled"]["library_b"] == 0
    hits, _ = select_hits(rows, 1, kb_keys=["library_a", "library_b"], mmr_lambda=None)
    assert len(hits) == 1
