"""Separate relevance decisions must survive fusion, timeouts and fallbacks."""
import time
from dataclasses import replace
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from kb_search import rerank as ranking_client
from kb_search import service
from kb_search.config import load_search_settings
from kb_search.evidence import source_row


def candidate(pid, *, image=False, score=0.9):
    return {"point_id": pid, "kb_id": "kb_example", "rrf": 0.03,
            "scores": {"score_visual": 0.99} if image else {},
            "recall_sources": ["visual"] if image else ["text"], "score_rerank": score,
            "payload": {"text": "A source passage about an example workflow.", "block_type": "image" if image else "text",
                        "visual_ref": pid + ".png" if image else None, "is_active": True}}


@pytest.fixture
def env(monkeypatch):
    ss = replace(load_search_settings({}), rerank_n=6,
                 visual_rerank_n=2, final_lex_weight=0, rerank_threshold=0.1)
    cfg = SimpleNamespace(reranker_base_url="http://text.invalid/v1", visual_reranker_base_url="http://image.invalid/v1")
    monkeypatch.setattr(service, "_backfill", lambda q, settings, ss, cands, **kw: cands)
    visual = Mock(return_value={"scores": {"image": 0.8}, "errors": [], "requested": 1})
    monkeypatch.setattr(service, "rerank_images", visual)
    return cfg, ss, visual


def run(env, rows, **kwargs):
    cfg, ss, _ = env
    result = service._rank(cfg, ss, None, "example workflow", [rows], until=time.time() + 3, **kwargs)
    service._final_scores(result["ordered"], ss, reranked=result["status"] == "ok",
                          image_query=bool(kwargs.get("image_query")), question="example workflow")
    result["evidence_state"] = service._mark_accepted(result["ordered"], result["status"], result["floor"],
                                                     image_query=bool(kwargs.get("image_query")))
    return result


def test_fusion_uses_ranks_and_an_image_has_only_one_vote(env):
    result = run(env, [candidate("text", score=0.2), candidate("image", image=True, score=0.99)])
    text, image = result["ordered"]
    assert text["score_fusion"] == image["score_fusion"] == 1 / 61
    assert image["ranking"]["branch"] == "visual"
    assert result["evidence_state"] == "accepted"
    row = source_row(1, image, image["payload"])
    assert row["scores"]["score_rerank_visual"] == 0.8
    assert row["ranking"]["branch"] == "visual"


def test_equal_branch_ranks_keep_recall_order_unless_query_contains_pixels(env):
    result = run(env, [candidate("image", image=True), candidate("text")])
    assert [c["point_id"] for c in result["ordered"]] == ["image", "text"]
    result = run(env, [candidate("text"), candidate("image", image=True)], image_query=True)
    assert [c["point_id"] for c in result["ordered"]] == ["image", "text"]


def test_fused_refinements_are_counted_without_applying_them_twice(env):
    row = candidate("image", image=True)
    row["payload"]["visual_confidence"] = "low"
    result = run(env, [row, candidate("text")], hint={"block_types": ["image"]})
    before = [c["score_final"] for c in result["ordered"]]
    counts = service._final_scores(result["ordered"], env[1], reranked=True, block_types=["image"])
    assert counts["visual_low"] == counts["hinted_blocks"] == 1
    assert [c["score_final"] for c in result["ordered"]] == before


def test_low_image_relevance_cannot_be_rescued_by_text_or_vector_score(env):
    env[2].return_value["scores"] = {"image": 0.02}
    result = run(env, [candidate("text"), candidate("image", image=True)], image_query=True, image_bytes=b"query")
    assert [c["point_id"] for c in result["ordered"]] == ["text"]
    assert next(c for c in result["cands"] if c["point_id"] == "image")["accepted"] is False
    assert env[2].call_args.kwargs["image_bytes"] == b"query"


def test_all_unrelated_is_diagnostic_even_when_every_list_has_a_first_place(env):
    env[2].return_value["scores"] = {"image": 0.01}
    result = run(env, [candidate("text", score=0.001), candidate("image", image=True, score=0.001)])
    assert result["status"] == "below_threshold"
    assert result["evidence_state"] == "diagnostic"
    assert all(c["accepted"] is False for c in result["ordered"])


@pytest.mark.parametrize("enabled,url", [(False, "http://image.invalid/v1"), (True, "")])
def test_global_reranking_disabled_or_endpoint_unconfigured_skips_image_service(env, enabled, url):
    cfg, ss, visual = env
    cfg.visual_reranker_base_url = url
    result = run((cfg, replace(ss, rerank_enabled=enabled), visual),
                 [candidate("text"), candidate("image", image=True, score=0.2)])
    visual.assert_not_called()
    assert result["ranking"]["method"] == "text"
    assert [c["point_id"] for c in result["ordered"]] == ["text", "image"]


def test_pure_text_keeps_existing_ranking_without_an_image_request(env):
    result = run(env, [candidate("low", score=0.3), candidate("high", score=0.9)])
    env[2].assert_not_called()
    assert result["ranking"]["visual"]["status"] == "no_candidates"
    assert [c["point_id"] for c in result["ordered"]] == ["high", "low"]


def test_image_failure_retains_text_order_and_reports_degradation(env):
    env[2].side_effect = TimeoutError("model unavailable")
    result = run(env, [candidate("text", score=0.9), candidate("image", image=True, score=0.2)])
    assert [c["point_id"] for c in result["ordered"]] == ["text", "image"]
    assert result["ranking"]["method"] == "text"
    assert result["degraded"] == ["visual_rerank:TimeoutError"]


def test_partial_image_failure_falls_back_only_for_missing_scores(env):
    env[2].return_value = {"scores": {"image": 0.8}, "errors": ["image:FileNotFoundError"], "requested": 2}
    result = run(env, [candidate("image", image=True), candidate("missing", image=True, score=0.7)])
    branches = {c["point_id"]: c["ranking"]["branch"] for c in result["ordered"]}
    assert branches == {"image": "visual", "missing": "text"}
    assert result["ranking"]["visual"]["status"] == "partial"


def test_text_failure_still_returns_verified_image_results(env, monkeypatch):
    monkeypatch.setattr(service, "rerank", Mock(side_effect=ConnectionError()))
    rows = [candidate("text"), candidate("image", image=True)]
    for c in rows:
        c.pop("score_rerank")
    result = run(env, rows)
    assert result["evidence_state"] == "accepted"
    assert [c["point_id"] for c in result["ordered"]] == ["image"]
    assert result["error"] == "rerank: ConnectionError"


def test_reserve_images_within_the_total_candidate_budget(env):
    cfg, ss, visual = env
    rows = [candidate(str(i)) for i in range(10)] + [candidate("image", image=True)]
    result = run((cfg, replace(ss, rerank_n=3), visual), rows)
    assert len(result["cands"]) == 3
    assert "image" in {c["point_id"] for c in result["cands"]}
    assert len(visual.call_args.args[2]) <= ss.visual_rerank_n


def test_explicit_scope_is_applied_before_images_are_read(env, monkeypatch):
    monkeypatch.setattr(service.channels, "hint_allows", lambda hint, payload: payload.get("allowed", False))
    allowed, blocked = candidate("text"), candidate("image", image=True)
    allowed["payload"]["allowed"] = True
    result = run(env, [allowed, blocked], hint={"rel_paths": ["allowed.pdf"]})
    assert [c["point_id"] for c in result["cands"]] == ["text"]
    env[2].assert_not_called()


def test_rankers_run_concurrently(env, monkeypatch):
    text_started, image_started = Event(), Event()

    def text(*args, **kw):
        text_started.set()
        assert image_started.wait(1)
        return [0.9, 0.8]

    def image(*args, **kw):
        image_started.set()
        assert text_started.wait(1)
        return {"scores": {"image": 0.8}, "errors": [], "requested": 1}

    monkeypatch.setattr(service, "rerank", text)
    env[2].side_effect = image
    rows = [candidate("text"), candidate("image", image=True)]
    for c in rows:
        c.pop("score_rerank")
    assert run(env, rows)["evidence_state"] == "accepted"


def test_late_image_job_does_not_mutate_a_completed_response(env):
    cfg, ss, visual = env
    release, finished = Event(), Event()

    def slow(*args, **kw):
        release.wait(2)
        finished.set()
        return {"scores": {"image": 0.01}, "errors": [], "requested": 1}

    visual.side_effect = slow
    try:
        result = run((cfg, replace(ss, visual_rerank_timeout=0.03), visual),
                     [candidate("text"), candidate("image", image=True)])
        assert result["ranking"]["method"] == "text"
        assert result["degraded"] == ["visual_rerank:TimeoutError"]
    finally:
        release.set()
    assert finished.wait(1)
    assert all("score_rerank_visual" not in c for c in result["cands"])


@pytest.mark.parametrize("results", [[], [{"index": 0, "relevance_score": float("nan")}],
                                    [{"index": 0, "relevance_score": None}],
                                    [{"index": 1, "relevance_score": 0.8}],
                                    [{"index": 0, "relevance_score": 0.5}] * 2])
def test_malformed_image_scores_fail_instead_of_appearing_valid(monkeypatch, results):
    response = Mock(ok=True)
    response.json.return_value = {"results": results}
    monkeypatch.setattr(ranking_client._session, "post", Mock(return_value=response))
    with pytest.raises(ValueError):
        ranking_client.Reranker("http://invalid.test/v1", model_id="example").score("question", ["document"], strict=True)


def test_image_payload_uses_pixels_and_preserves_query_and_bounds_paths(tmp_path, monkeypatch):
    Image.new("RGB", (24, 24), "white").save(tmp_path / "image.png")
    cfg = SimpleNamespace(cache_dir=tmp_path, visual_reranker_base_url="http://invalid.test/v1")
    client = Mock()
    client.score.return_value = [0.8]
    monkeypatch.setattr(ranking_client, "Reranker", Mock(return_value=client))
    rows = [candidate("image", image=True), candidate("escape", image=True)]
    rows[1]["payload"]["visual_ref"] = "../outside.png"
    data = (tmp_path / "image.png").read_bytes()
    out = ranking_client.rerank_images(cfg, "Find the workflow", rows, image_bytes=data, until=time.time() + 3)
    assert out["scores"] == {"image": 0.8}
    assert out["errors"] == ["image:KeyError"]
    query, docs = client.score.call_args.args
    assert query["content"][0]["text"] == "Find the workflow"
    assert query["content"][1]["image_url"]["url"].startswith("data:image/")
    assert len(docs) == 1 and docs[0]["content"][1]["image_url"]["url"].startswith("data:image/")


@pytest.mark.parametrize("bucketed", [False, True])
@pytest.mark.parametrize("kb_count", [1, 2])
def test_image_reserve_survives_grouping_without_expanding_the_total_budget(env, bucketed, kb_count):
    cfg, ss, visual = env
    ss = replace(ss, rerank_n=3, visual_rerank_n=1)
    per_kb = {}
    for kb in range(kb_count):
        rows = [candidate(f"text-{kb}-{i}") for i in range(10)] + [candidate(f"image-{kb}", image=True)]
        for i, row in enumerate(rows):
            row["kb_id"], row["bucket"] = str(kb), str(i % 2)
        per_kb[str(kb)] = {"fused": rows}
    buckets = {"mode": "doc", "keys": ["0", "1"]} if bucketed else None
    groups = service._groups(list(per_kb), per_kb, buckets, ss.rerank_n, visual_limit=ss.visual_rerank_n)
    visual.return_value = {"scores": {"image-0": 0.8}, "errors": [], "requested": 1}
    result = service._rank(cfg, ss, None, "workflow", groups, buckets, until=time.time() + 2)
    assert len(result["cands"]) == ss.rerank_n
    assert "image-0" in {c["point_id"] for c in result["cands"]}
    assert len(visual.call_args.args[2]) == ss.visual_rerank_n
    legacy = service._groups(list(per_kb), per_kb, buckets, ss.rerank_n)
    assert all(not row["payload"]["visual_ref"] for group in legacy for row in group)


@pytest.mark.parametrize("visual_score", [0.01, 0.8])
@pytest.mark.parametrize("image_query", [False, True])
def test_image_caption_does_not_set_the_text_branch_floor(env, visual_score, image_query):
    env[2].return_value["scores"] = {"image": visual_score}
    result = run(env, [candidate("text", score=0.08), candidate("image", image=True, score=0.9)],
                 image_query=image_query)
    accepted = {c["point_id"]: c["accepted"] for c in result["cands"]}
    assert accepted == {"text": True, "image": visual_score >= env[1].visual_rerank_threshold}
    assert result["floor"] == pytest.approx(0.07)
    assert result["ranking"]["text"]["status"] == "floor_lowered"
    assert result["evidence_state"] == "accepted"


def test_partial_image_failure_recalculates_the_floor_for_remaining_text(env):
    env[2].return_value = {"scores": {"image": 0.01}, "errors": ["image:FileNotFoundError"], "requested": 2}
    result = run(env, [candidate("text", score=0.08), candidate("missing", image=True, score=0.09),
                       candidate("image", image=True, score=0.9)])
    accepted = {c["point_id"]: c["accepted"] for c in result["cands"]}
    assert accepted == {"text": True, "missing": True, "image": False}
    assert result["ranking"]["visual"]["status"] == "partial"
    assert result["floor"] == pytest.approx(0.07)


def test_search_applies_table_preference_before_dual_fusion(monkeypatch):
    from test_search import OrchestrationTests

    original = service._rank_dual

    def with_table(cfg, ss, question, cands, **kwargs):
        for cand in cands:
            if cand["point_id"] == "p2":
                cand["payload"] = {**cand["payload"], "block_type": "table"}
        return original(cfg, ss, question, cands, **kwargs)

    def scores(documents):
        return [0.9 if "尿酸 433" in doc else 0.8 if "甘油" in doc else 0.1 for doc in documents]

    monkeypatch.setattr(service, "visual_rerank_active", lambda *args: True)
    monkeypatch.setattr(service, "rerank_images", Mock(return_value={"scores": {"p3": 0.8}, "errors": [], "requested": 1}))
    monkeypatch.setattr(service, "_rank_dual", with_table)
    case = OrchestrationTests()
    options = {"kbs": ["kb_001"], "rerank_fn": scores,
               "settings_over": {"final_lex_weight": 0, "mmr_enabled": False, "hint_block_boost": 1.2}}
    plain, _ = case._run(**options)
    preferred, _ = case._run(**options, hints={"block_types": ["table"]})

    def text_order(out):
        return [row["point_id"] for row in out["sources"]
                if row["role"] == "hit" and row["ranking"]["branch"] == "text"]

    assert text_order(plain) == ["p1", "p2"]
    assert text_order(preferred) == ["p2", "p1"]  # 0.8 * 1.2 precedes 0.9.
    assert preferred["retrieval_summary"]["downweighted"]["hinted_blocks"] == 1
    assert preferred["retrieval_summary"]["ranking"]["method"] == "rrf"
