"""Synthetic assertion semantics, independent of any deployed knowledge base."""
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from kb_pipeline.graph.quality import (
    flag_status, relation_semantics, relation_qualifier, relation_quality,
    traversable_relation,
)
from kb_pipeline.graph.reconcile import comparison_groups, reconcile_facts
from kb_pipeline.graph.extract import parse_records, consolidate
from kb_pipeline.graph.vectors import relation_payload, relation_embed_text, spec_payload


def fact(**fields):
    return {"id": "f", "subject": "Sample", "subject_key": "sample", "property": "Output",
            "concept_key": "output", "value": "8", "value_num": 8.0, "unit": "V",
            "conditions": {}, "doc_id": "d", "fact_kind": "observation", **fields}


@pytest.mark.parametrize("marker", ["正常", "未见异常", "normal", "within range", "In Range"])
def test_normal_marker_is_not_an_abnormal_observation(marker):
    assert flag_status(fact(flag=marker)) == "normal"


@pytest.mark.parametrize("marker", ["↑", "↓", "H", "L", "偏高", "偏低", "异常", "high", "low", "out of range"])
def test_explicit_abnormal_marker(marker):
    assert flag_status(fact(flag=marker)) == "abnormal"


@pytest.mark.parametrize("marker", ["", "阴性", "阳性", "+", "-", "pending", "复测"])
def test_unknown_marker_is_not_assumed_abnormal(marker):
    assert flag_status(fact(flag=marker)) == "unknown"


def test_bounds_and_marker_disagreement_is_visible():
    assert flag_status(fact(ref_min="1", ref_max="9")) == "normal"
    assert flag_status(fact(ref_min="1", ref_max="7")) == "abnormal"
    assert flag_status(fact(flag="正常", ref_min="1", ref_max="7")) == "uncertain"
    assert flag_status(fact(flag="↑", ref_min="1", ref_max="9")) == "uncertain"
    assert flag_status(fact(flag="↑", quality="ambiguous_source")) == "uncertain"


@pytest.mark.parametrize("changed", [
    {"subject_key":"other"}, {"concept_key":"other"}, {"conditions":{"mode":"fast"}},
    {"unit":""}, {"unit":"mV"}, {"unit":"MW"}, {"fact_kind":"state"},
    {"valid_from":"v2"},
])
def test_incomparable_facts_do_not_share_page_or_series_groups(changed):
    rows=[fact(id="a",valid_from="2024",unit="mW"),fact(**{"id":"b","valid_from":"2025","unit":"mW",**changed})]
    assert len(comparison_groups(rows)) == 2
    reconcile_facts(rows)
    assert all("series_key" not in r for r in rows)


def test_observations_and_explicit_states_keep_history():
    rows=[fact(id=str(year),valid_from=str(year),doc_id=str(year)) for year in (2023,2024,2025)]
    reconcile_facts(rows)
    assert len({r["series_key"] for r in rows}) == 1
    assert all("superseded_by" not in r for r in rows)
    for r in rows:
        r["fact_kind"]="state"
    reconcile_facts(rows)
    assert "superseded_by" not in rows[0]
    assert "superseded_by" not in rows[1]
    assert "superseded_by" not in rows[2]


def test_unresolved_owner_never_drives_a_trend():
    rows=[fact(id="a",valid_from="2024"),fact(id="b",valid_from="2025",subject_origin="unresolved")]
    assert reconcile_facts(rows)["stats"]["evidence_conflict_skipped"] == 1
    assert all("series_key" not in r for r in rows)


@pytest.mark.parametrize("qualifier", [
    {"polarity":"negated"}, {"target_polarity":"negated"}, {"modality":"recommended"},
    {"modality":"planned"}, {"modality":"hypothetical"}, {"conditions":{"mode":"recovery"}},
    {"polarity":"uncertain"},
])
def test_qualified_relations_survive_extraction_merge_and_vector_projection(qualifier):
    base={"polarity":"affirmed","target_polarity":"affirmed","modality":"actual","conditions":{}}
    records=[]
    for sem in (base,{**base,**qualifier}):
        records.append('("relationship"<|>Assembly<|>Module<|>uses<|>The source statement<|>8<|><|>'+json.dumps(sem)+')')
    _, rows, stats=parse_records('##'.join(records))
    assert stats.get("malformed_metadata",0)==0
    _, rows=consolidate([],rows)
    assert len(rows)==2
    payloads=[]
    for row in rows:
        row.update(source_key="assembly",target_key="module",unit_ids=["u"],doc_ids=["d"])
        payloads.append(relation_payload(row,kb_id="kb_x",source_collection="kb_x",graph_version="g",point_ids=["p"]))
    assert payloads[0]["gr_id"] != payloads[1]["gr_id"]
    assert payloads[0]["traversable"] is True
    assert payloads[1]["traversable"] is False
    assert relation_semantics(payloads[1]) == {**base,**qualifier}
    assert payloads[1]["point_ids"] == ["p"]
    assert relation_embed_text(rows[0]) != relation_embed_text(rows[1])


def test_negation_scope_and_serialized_conditions_are_distinct():
    a={"polarity":"negated","predicate":"recommends"}
    b={"target_polarity":"negated","predicate":"recommends"}
    assert relation_qualifier(a)!=relation_qualifier(b)
    assert relation_semantics({"conditions_json":'{"mode":"safe"}'})["conditions"]=={"mode":"safe"}
    assert not traversable_relation({"type_violation":True})
    assert relation_quality({"quality_state":"needs_review"})=="needs_review"


def test_different_fact_kinds_and_case_sensitive_units_do_not_deduplicate():
    from kb_pipeline.graph.recall import collapse_duplicate_facts
    rows=[fact(unit="mW"),fact(unit="MW"),fact(unit="mW",fact_kind="reference")]
    assert len(collapse_duplicate_facts(rows))==3


@pytest.mark.parametrize("a,b", [
    ("适合自动化","不适合自动化"), ("enable routing","disable routing"),
    ("允许访问","不允许访问"), ("压力(高)","压力(低)"),
])
def test_concept_opposites_stay_separate_even_if_model_agrees(a,b):
    from kb_pipeline.graph.concepts import build_concepts
    class Agree:
        def chat(self,prompt,**kwargs):return '\n'.join(f'{i+1}. yes' for i in range(prompt.count('  vs  ')))
    rows=[fact(property=a,concept_key=None),fact(property=b,concept_key=None)]
    build_concepts(rows,embed=lambda texts:[[1.,0.] for _ in texts],client=Agree())
    assert rows[0]["concept_key"]!=rows[1]["concept_key"]


def test_identical_symbols_do_not_prove_property_identity():
    from kb_pipeline.graph.concepts import build_concepts
    rows=[fact(property=p,symbol="MODX",value="A",value_num=None,unit="") for p in ("接口类型","评估结论","建议措施")]
    build_concepts(rows)
    assert len({r["concept_key"] for r in rows})==3


def test_semantic_alias_requires_judgement_and_remains_recoverable():
    from kb_pipeline.graph.concepts import build_concepts
    rows=[fact(property="Supply voltage"),fact(property="Operating voltage")]
    embed=lambda texts:[[1.,0.] for _ in texts]
    build_concepts(rows,embed=embed)
    assert rows[0]["concept_key"]!=rows[1]["concept_key"]
    class Agree:
        def chat(self,*args,**kwargs):return '1. yes'
    build_concepts(rows,embed=embed,client=Agree())
    assert rows[0]["concept_key"]==rows[1]["concept_key"]


def test_qualified_value_nodes_are_not_folded_into_positive_attributes():
    from kb_pipeline.graph.merge import admit_entities
    ents=[{"key":"a","title":"Controller","type":"entity","descriptions":[]},
          {"key":"b","title":"3.3 V","type":"value","descriptions":[]}]
    rels=[{"source_key":"a","target_key":"b","predicate":"requires","polarity":"negated"}]
    es,rs,_=admit_entities(deepcopy(ents),deepcopy(rels))
    assert len(es)==2 and len(rs)==1
    assert not es[0].get("attributes")


def test_fact_quality_and_source_availability_are_separate():
    from kb_search import evidence
    payload=spec_payload(fact(subject_origin="unresolved"),kb_id="kb_x",source_collection="kb_x",graph_version="g")
    assert payload["quality_state"]=="needs_review"
    assert payload["comparable"] is False
    assert "source_valid" not in payload  # determined at request time by checking source points


def test_neo4j_projection_preserves_qualified_assertions(tmp_path):
    from kb_pipeline.graph.neo4j_import import load_bundle
    from kb_pipeline.models import KBSource
    source = KBSource(kb_id="kb_x", collection="kb_x", source_root="samples", source_type="local_mirror", max_tokens=400, overlap_tokens=80)
    edge = {"source_key":"a", "target_key":"b", "source":"Controller", "target":"Module",
            "predicate":"uses", "directed":True, "target_polarity":"negated",
            "modality":"recommended", "conditions":{"mode":"recovery"}, "unit_ids":[]}
    graph = {"entities":[{"key":"a","title":"Controller"},{"key":"b","title":"Module"}],
             "relations":[edge],"specs":[],"pages":[]}
    (tmp_path / "graph.json").write_text(json.dumps(graph))
    (tmp_path / "units.jsonl").write_text("")
    bundle = load_bundle(tmp_path, source=source, graph_version="v1")
    node = bundle["relations"][0]
    relation = bundle["relation_entity_edges"][0]["related_props"]
    for row in (node, relation):
        assert row["target_polarity"] == "negated"
        assert row["modality"] == "recommended"
        assert json.loads(row["conditions_json"]) == {"mode":"recovery"}
        assert row["traversable"] is False
    assert node["id"] == relation["id"]


def test_default_relation_examples_follow_the_qualified_record_contract():
    from kb_pipeline.graph.prompts import DEFAULT_EXAMPLES
    from kb_pipeline.graph.extract import parse_records
    lines = [line for line in DEFAULT_EXAMPLES.splitlines() if line.startswith('(\"relationship\"')]
    assert lines
    for line in lines:
        assert len(line.split('<|>')) == 8
        _, rows, stats = parse_records(line)
        assert len(rows) == 1 and not stats.get('malformed_metadata')
        assert relation_semantics(rows[0])["modality"] == "actual"
