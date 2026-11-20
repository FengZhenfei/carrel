"""Tier-1 fixes of 2026-11-20 in the graph build.

A2: a fact record whose fact_kind / subject_origin was a list or an object crashed normalize_fact with TypeError,
    which failed the whole facts unit; the reply was well-formed and cached, so every retry failed the same way.
A4: a prose reply to the type-induction prompt was split at commas and line breaks into "type names"; it passed
    the format check, was cached, and one fragment over the type-name limit failed the whole induction for good.
"""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from kb_pipeline.graph import schema as graph_schema
from kb_pipeline.graph.facts import FactExtractor, facts_response_ok, normalize_fact
from kb_pipeline.graph.llm import LLMMalformedResponse
from kb_pipeline.ontology import TYPE_NAME_MAX_LEN, normalize_model

from _support import _client, _unit

# ── A2 ────────────────────────────────────────────────────────────────────

GOOD_FACT = {"subject": "电源模块", "property": "输出电压", "value": "5", "unit": "V", "evidence": "输出电压为 5 V"}
UNIT_TEXT = "电源模块的输出电压为 5 V，测试条件为 25 摄氏度。第二路输出电压为 12 V。"


@pytest.mark.parametrize("odd", [["observation"], {"kind": "state"}, [["state"], {"a": 1}]])
def test_unhashable_fact_kind_and_subject_origin_count_as_unrecognised_values(odd):
    fact = normalize_fact({**GOOD_FACT, "fact_kind": odd, "subject_origin": odd})
    assert fact is not None
    assert fact["fact_kind"] == "observation"       # the default an unknown string gets as well
    assert "subject_origin" not in fact


def test_recognised_values_are_kept_and_unknown_strings_still_get_the_default():
    fact = normalize_fact({**GOOD_FACT, "fact_kind": "state", "subject_origin": "context"})
    assert (fact["fact_kind"], fact["subject_origin"]) == ("state", "context")
    fact = normalize_fact({**GOOD_FACT, "fact_kind": "measurement", "subject_origin": "guess"})
    assert fact["fact_kind"] == "observation" and "subject_origin" not in fact
    fact = normalize_fact(dict(GOOD_FACT))
    assert fact["fact_kind"] == "observation" and "subject_origin" not in fact


def test_one_record_with_a_list_valued_field_no_longer_fails_the_unit():
    other = {"subject": "电源模块", "property": "第二路输出电压", "value": "12", "unit": "V", "evidence": "第二路输出电压为 12 V"}
    reply = json.dumps({"facts": [{**GOOD_FACT, "fact_kind": ["observation"], "subject_origin": ["explicit"]}, other]},
                       ensure_ascii=False)
    assert facts_response_ok(reply)                  # well-formed, so the client accepts and caches it
    result = FactExtractor(_client([reply])).extract(replace(_unit("u1"), text=UNIT_TEXT))
    assert [f["property"] for f in result.facts] == ["输出电压", "第二路输出电压"]
    assert result.facts[0]["fact_kind"] == "observation"


# ── A4 ────────────────────────────────────────────────────────────────────

PROSE = ("Based on the documents, the main entity types are product, supplier and contract\n"
         "In addition the corpus repeatedly refers to quality inspection records that should probably be modelled as their own type")


def _types_reply_ok(text: str) -> bool:
    """The check schema.suggest hands to the client for the entity-type call."""
    return bool(graph_schema.parse_entity_types(graph_schema.parse_json_object(text) or text))


def test_prose_is_not_a_type_list():
    assert graph_schema.parse_entity_types(PROSE) == []
    assert not _types_reply_ok(PROSE)
    assert not _types_reply_ok("person, organization, product")      # a bare comma list is not the requested format either


def test_json_objects_arrays_and_bracketed_lists_are_still_read():
    assert graph_schema.parse_entity_types({"entity_types": ["product", " supplier "]}) == ["product", "supplier"]
    assert graph_schema.parse_entity_types('["product", "supplier"]') == ["product", "supplier"]
    assert graph_schema.parse_entity_types("Here you go: [product, supplier]") == ["product", "supplier"]
    assert _types_reply_ok('```json\n{"entity_types": ["product", "supplier"]}\n```')
    truncated = '{"entity_types": ["product", "supplier", "contr'       # cut off by the output limit
    assert graph_schema.parse_entity_types(graph_schema.parse_json_object(truncated) or truncated)[:2] == ["product", "supplier"]


def test_an_overlong_name_is_dropped_alone_and_not_truncated():
    long_name = "quality inspection record that should be its own type " * 2
    assert len(long_name.strip()) > TYPE_NAME_MAX_LEN
    names = graph_schema.parse_entity_types({"entity_types": ["product", long_name, "supplier"]})
    assert names == ["product", "supplier"]
    normalize_model({"types": [{"name": n, "is_a": "entity"} for n in names]})       # the shape suggest() builds; must not raise
    edge = "x" * TYPE_NAME_MAX_LEN
    assert graph_schema.parse_entity_types({"entity_types": [edge, edge + "y"]}) == [edge]
    assert not _types_reply_ok(json.dumps({"entity_types": [long_name]}))            # nothing usable left: ask again


def _suggest_answers(type_replies: list[str]) -> list[str]:
    return ["Synthetic module specifications", "English", "You are an expert.", *type_replies,
            '{"parent_types": {"sample module": "component", "pin": "interface"}}',
            '{"predicates": [{"name": "has_pin", "description": "d", "source_parents": ["component"], "target_parents": ["interface"]}]}']


def test_suggest_asks_again_after_a_prose_reply_and_uses_the_json_answer():
    client = _client(_suggest_answers([PROSE, '{"entity_types": ["sample module", "pin"]}']))
    out = graph_schema.suggest(client, ["sample text"])
    assert [t["name"] for t in out["model"]["types"]] == ["sample module", "pin"]


def test_suggest_fails_with_a_format_error_when_the_model_keeps_answering_in_prose():
    client = _client(_suggest_answers([PROSE, PROSE]))
    with pytest.raises(LLMMalformedResponse):
        graph_schema.suggest(client, ["sample text"])
