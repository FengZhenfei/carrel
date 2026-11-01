"""Synthetic evidence, identity and gradual topic-change regression cases."""
import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from kb_pipeline.graph.comparison import compare_graphs, digest, save_report, source_snapshot
from kb_pipeline.graph.corrections import apply_entity_corrections, apply_fact_corrections, parse_corrections
from kb_pipeline.graph.extract import ExtractionSchema
from kb_pipeline.graph.governance import context, prepare_record, apply_records
from kb_search.structured_eval import evaluate_structured
from _support import _unit


def entity(key, doc="a", **kwargs):
    return {"key": key, "title": "Shared", "type": "object", "scope": doc, "doc_ids": [doc],
            "unit_ids": ["u" + doc], "aliases": [], **kwargs}


def test_custom_graph_versions_share_workspace_retention_and_keep_records(tmp_path):
    import os
    from kb_pipeline import db
    from kb_pipeline.graph.build import gc_graph_versions
    from kb_pipeline.vector.qdrant import graph_collection_name

    state = tmp_path / "state.db"
    db.init_db(state)
    versions = ["published", "prior-new", "prior-old", "expired"]
    names = [graph_collection_name("kb_001", "entity", v) for v in versions]

    class Q:
        def get_aliases(self):
            return SimpleNamespace(aliases=[SimpleNamespace(collection_name=names[0])])

        def get_collections(self):
            return SimpleNamespace(collections=[SimpleNamespace(name=n) for n in names])

        def delete_collection(self, collection_name):
            names.remove(collection_name)

    with db.connect(state) as con:
        for i, v in enumerate(versions):
            p = tmp_path / "work" / "001" / v
            p.mkdir(parents=True)
            os.utime(p, (400 - i * 100, 400 - i * 100))
            con.execute("INSERT INTO graph_builds(graph_build_id, source_key, kb_id, source_collection, "
                        "graph_version, status, started_at, finished_at, build_kind) "
                        "VALUES(?, 'k', 'kb_001', 'kb_001', ?, 'done', ?, ?, 'full')",
                        (v, v, 400 - i * 100, 400 - i * 100))
        con.commit()
    settings = SimpleNamespace(state_db=state, graph_work_dir=tmp_path, graph_gc_keep_versions=3,
                               graph_gc_retention_days=14, graph_neo4j_import_after_build=False)
    source = SimpleNamespace(kb_id="kb_001", collection="kb_001")
    result = gc_graph_versions(settings, source, q=Q(), graph_version="published")
    assert not result["errors"]
    assert {p.name for p in (tmp_path / "work" / "001").iterdir()} == set(versions[:3])
    assert names == [graph_collection_name("kb_001", "entity", v) for v in versions[:3]]
    with db.connect(state) as con:
        assert {r[0] for r in con.execute("SELECT graph_version FROM graph_builds")} == set(versions[:3])


def test_custom_version_timestamp_does_not_leak_between_libraries():
    from kb_pipeline.vector.qdrant import delete_old_graph_collections, graph_collection_name
    names = [graph_collection_name(k, "entity", "custom") for k in ["kb_001", "kb_002"]]
    q = SimpleNamespace(get_aliases=lambda: SimpleNamespace(aliases=[]),
                        get_collections=lambda: SimpleNamespace(collections=[SimpleNamespace(name=n) for n in names]))
    out = delete_old_graph_collections(q, source_collections=["kb_001", "kb_002"], retention_days=1,
                                      dry_run=True, version_timestamps={("001", "custom"): 1})
    assert [r["collection"] for r in out["deleted"]] == [names[0]]
    assert out["skipped_collections"][0]["collection"] == names[1]
    assert out["skipped_collections"][0]["reason"] == "unparseable_version_timestamp"


def test_comparison_separates_missing_assertion_from_deleted_source():
    old = {"graph_version": "v1", "entities": [entity("one")], "specs": [
        {"subject_key": "one", "property": "rate", "value": "12", "unit": "%", "doc_id": "a", "unit_id": "ua"}]}
    new = {**old, "graph_version": "v2", "specs": []}
    same = compare_graphs(old, new, before_sources={"a": "hash"}, after_sources={"a": "hash"})
    assert same["kinds"]["facts"]["removed_with_unchanged_source"] == 1
    assert same["review_required"]
    changed = compare_graphs(old, new, before_sources={"a": "hash"}, after_sources={"a": "new"})
    assert not changed["review_required"]
    assert not changed["comparable"]["sources"]
    assert old["specs"][0]["value"] == "12"


def test_comparison_keeps_conditions_polarity_and_multiple_values():
    row = {"subject_key": "x", "property": "price", "value": "10", "doc_id": "a", "conditions": {"tier": "A"}}
    before = {"specs": [row, {**row, "value": "11"}]}
    after = {"specs": [{**row, "value": "12"}, {**row, "value": "11"}]}
    report = compare_graphs(before, after)
    assert report["kinds"]["facts"]["changed"] == 1
    assert report["kinds"]["facts"]["before"] == 2
    after["specs"][0]["conditions"] = {"tier": "B"}
    assert compare_graphs(before, after)["kinds"]["facts"]["added"] == 1
    relation = {"source_key": "x", "target_key": "y", "predicate": "uses", "polarity": "affirmed"}
    report = compare_graphs({"relations": [relation]}, {"relations": [{**relation, "polarity": "negated"}]})
    assert report["kinds"]["relations"]["removed"] == report["kinds"]["relations"]["added"] == 1


def test_report_retention_does_not_need_full_graphs(tmp_path):
    for i in range(25):
        save_report(tmp_path, {"after": "v" + str(i), "status": "no_previous_version"})
    assert len(list(tmp_path.glob("*.json"))) == 20
    assert (tmp_path / "v24.json").exists()
    with pytest.raises(ValueError):
        save_report(tmp_path, {"after": "../other"})


def test_source_snapshot_order_is_stable_and_retains_duplicates():
    a, b = replace(_unit("ua", "a"), text="A"), replace(_unit("ub", "a", order=1), text="B")
    assert source_snapshot([a, b]) == source_snapshot([b, a])
    assert source_snapshot([a, b]) != source_snapshot([a, b, b])


@pytest.mark.parametrize("rule", [{"types": {"Shared": "person"}}, {"aliases": {"Shared": ["Alias"]}},
                                  {"reject": ["Shared"]}, {"merge": [["Shared", "Distinct"]]},
                                  {"separate": [["Shared", "Distinct"]]}])
def test_legacy_name_rules_never_pick_one_of_several_scoped_objects(rule):
    rows = [entity("one", "a"), entity("two", "b"), entity("three", "c", title="Distinct")]
    before = copy.deepcopy(rows)
    kept, _, stats, pairs = apply_entity_corrections(rows, [], parse_corrections({"entities": rule}), known_types=["object", "person"])
    assert kept == before and not pairs
    assert stats["ambiguous_names"] == ["Shared"]


def governance_fixture():
    units = [replace(_unit("ua", "a"), text="Shared manages Project A."),
             replace(_unit("ub", "b"), text="Shared manages Project B.")]
    schema = ExtractionSchema(("object", "person"), type_definitions={"object": "A thing", "person": "A person"})
    graph = {"entities": [entity("one", "a"), entity("two", "b")]}
    corr = parse_corrections({"kb": {"identity": "generation-a"}})
    corr["_context"] = context(schema, units)
    payload = {"id": "decision-1", "action": "entity.type", "targets": [{"name": "Shared", "scope": "b"}],
               "value": "person", "submitted_by": "reviewer", "evidence": ["document:b"], "base_graph_version": "v1"}
    return units, schema, graph, corr, payload


def test_scoped_correction_precisely_targets_and_records_outcome():
    units, schema, graph, corr, payload = governance_fixture()
    record = prepare_record(payload, graph, corr, graph_version="v1", schema=schema, units=units)
    corr["records"] = [record]
    entities, _, _, _ = apply_records(graph["entities"], [], [], corr, phase="entity", known_types=schema.entity_types)
    assert [e["type"] for e in entities] == ["object", "person"]
    assert corr["_outcomes"] == {"decision-1": "applied"}


@pytest.mark.parametrize("stale", ["schema", "source", "library", "revoked"])
def test_stale_or_revoked_correction_cannot_change_an_entity(stale):
    units, schema, graph, corr, payload = governance_fixture()
    record = prepare_record(payload, graph, corr, graph_version="v1", schema=schema, units=units)
    corr["records"] = [record]
    if stale == "schema": corr["_context"]["schema_sha"] = "changed"
    if stale == "source": corr["_context"]["source_units"]["ub"] = "changed"
    if stale == "library": corr["identity"] = "generation-b"
    if stale == "revoked": record["state"] = "revoked"
    entities, _, _, _ = apply_records(graph["entities"], [], [], corr, phase="entity", known_types=schema.entity_types)
    assert [e["type"] for e in entities] == ["object", "object"]


def test_fact_rejection_distinguishes_same_names_and_conditions():
    corr = parse_corrections({"facts": {"reject": [{"subject": "Shared", "property": "price"}]}})
    rows = [{"subject": "Shared", "subject_key": "one", "property": "price", "value": "1"},
            {"subject": "Shared", "subject_key": "two", "property": "price", "value": "2"}]
    kept, n = apply_fact_corrections(rows, corr)
    assert kept == rows and n == 0


def test_evaluation_requires_same_fact_and_complete_path():
    item = {"required_facts": [{"subject": "X", "property": "price", "value": "10", "conditions": {"tier": "A"}}],
            "required_documents": ["a.md", "b.pdf"], "required_paths": [[
                {"source": "X", "predicate": "part_of", "target": "Y"},
                {"source": "Y", "predicate": "owned_by", "target": "Z"}]]}
    out = {"specs": [{"subject": "X", "property": "price", "value": "10", "conditions": {"tier": "B"}}],
           "sources": [{"rel_path": "a.md"}], "relationships": [{"source": "X", "type": "part_of", "target": "Y"}]}
    result = evaluate_structured(item, out)
    assert not result["required_facts"] and not result["required_paths"] and not result["required_documents"]
    out["specs"][0]["conditions"]["tier"] = "A"
    out["sources"].append({"rel_path": "b.pdf"})
    out["relationships"].append({"source": "Y", "type": "owned_by", "target": "Z"})
    result = evaluate_structured(item, out)
    assert result["required_facts"] and result["required_paths"] and result["required_documents"]


def test_scoped_rejection_survives_phase_resume():
    from kb_pipeline.graph.governance import save_state, restore_state
    units, schema, graph, corr, payload = governance_fixture()
    payload.update(action="entity.reject")
    corr["records"] = [prepare_record(payload, graph, corr, graph_version="v1", schema=schema, units=units)]
    entities, relations, _, _ = apply_records(graph["entities"], [], [], corr, phase="entity")
    saved = {"entities": entities, "relations": relations}
    save_state(saved, corr)
    resumed = {"identity": corr["identity"], "records": corr["records"], "_context": corr["_context"]}
    restore_state(json.loads(json.dumps(saved)), resumed)
    facts = [{"subject": "Shared", "subject_key": "one", "property": "rate", "value": "1", "doc_id": "a"},
             {"subject": "Shared", "subject_key": "two", "property": "rate", "value": "2", "doc_id": "b"}]
    kept, count = apply_fact_corrections(facts, resumed)
    assert count == 1 and kept == facts[:1]
    assert resumed["_outcomes"] == {"decision-1": "applied"}


def test_old_correction_file_receives_identity_without_changing_rules(tmp_path):
    from kb_pipeline.graph.corrections import ensure_corrections_file, load_corrections, effective_sha
    settings = SimpleNamespace(corrections_dir=tmp_path)
    path = tmp_path / "test.yaml"
    path.write_text('kb:\n  id: test\nentities:\n  aliases:\n    Sample: [Example]\n')
    before = effective_sha(load_corrections(path))
    assert ensure_corrections_file(settings, "test", folder="Example")["identity_added"]
    first = path.read_bytes()
    assert effective_sha(load_corrections(path)) == before
    ensure_corrections_file(settings, "test", folder="Example")
    assert path.read_bytes() == first


def test_structured_evaluation_detects_future_evidence_and_empty_conditions():
    item = {"required_facts": [{"subject": "Example", "property": "price", "value": "10", "conditions": {}}],
            "forbidden_facts": [{"subject": "Example", "observed_at": "2027-01-01"}]}
    out = {"specs": [{"subject": "Example", "property": "price", "value": "10", "conditions": {},
                      "observed_at": "2026-01-01"}]}
    assert evaluate_structured(item, out)["required_facts"]
    assert evaluate_structured(item, out)["forbidden_facts"]
    out["specs"].append({**out["specs"][0], "observed_at": "2027-01-01"})
    assert not evaluate_structured(item, out)["forbidden_facts"]
    out["specs"] = [{**out["specs"][0], "conditions": {"tier": "B"}}]
    assert not evaluate_structured(item, out)["required_facts"]


def test_eval_forwards_time_scope_and_counts_errors_as_structured_failures():
    from kb_search.evalset import evaluate, compare
    calls = []
    def search(q, **kwargs):
        calls.append(kwargs)
        if q == "broken": raise RuntimeError("service unavailable")
        return {"specs": [{"subject": "Example", "value": "10"}]}
    items = [{"q": q, "as_of": "2026-01-01", "history": True,
              "required_facts": [{"subject": "Example", "value": "10"}]} for q in ["normal", "broken"]]
    result = evaluate(items, search)
    assert calls[0]["as_of"] == "2026-01-01" and calls[0]["history"]
    assert result["summary"]["required_facts"] == "1/2"
    assert result["summary"]["errors"] == 1
    assert next(x for x in compare(result, None) if x[0] == "required_facts")[1] == "1/2"


def test_schema_retention_keeps_ten_recent_plus_referenced_predecessors():
    from kb_pipeline.limits import push_schema_version
    out = push_schema_version([{"id": f"v{i}"} for i in range(14, 0, -1)], {"id": "v15"},
                              active_id="v1", protected_ids={"v2"})
    assert [r["id"] for r in out] == [f"v{i}" for i in range(15, 5, -1)] + ["v2", "v1"]


def test_rejected_entity_never_deletes_another_scope_in_the_same_document():
    corr = {"_rejected_entities": [entity("one", "a")]}
    facts = [{"subject": "Shared", "subject_key": key, "doc_id": "a", "property": "rate", "value": "1"}
             for key in ["one", "two", ""]]
    kept, count = apply_fact_corrections(facts, corr)
    assert count == 1 and kept == facts[1:]


def test_missing_historical_model_metadata_is_unknown_not_a_model_change():
    report = compare_graphs({"models": {"extract": "Example"}}, {},
                            after_manifest={"model_identity": {"extract": {"name": "Example", "model_id": "example"}}})
    assert report["comparable"]["models"] is None
    assert "models" in report["unknown_inputs"] and "models" not in report["changed_inputs"]
