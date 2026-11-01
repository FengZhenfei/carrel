"""Generic graph fidelity and gradual schema evolution scenarios."""
from copy import deepcopy
from types import SimpleNamespace
import json

import pytest

from kb_pipeline.graph.units import Unit


def unit(text, uid="u", path="sample.csv", **kw):
    return Unit(unit_id=uid, doc_id=path, rel_path=path, text=text, section_path=[],
                block_ids=[], chunk_uids=[uid], point_ids=[uid], n_tokens=100, **kw)


def test_version_words_and_total_natural_order():
    from kb_pipeline.graph.temporal import find_versions, axis_sort_key
    assert not find_versions("reverse revenue review revisionism versioned")
    assert find_versions("Rev. *L / revision 03 / v2.10 / Version 3") == ["rev *L", "rev 03", "v2.10", "v3"]
    values = ["rev *L", "rev 03", "rev A", "v2.10", "v2.2", "2026-01-01", ""]
    assert sorted(values, key=axis_sort_key)[-1] == ""
    assert axis_sort_key("v2.2") < axis_sort_key("v2.10")


@pytest.mark.parametrize("path,text,expected", [
    ("schema.xml", '<x xmlns="https://example.org/2015/9/8/a"/>', ""),
    ("observations.csv", "date,value\n2025-01-01,12\n2025-02-01,13", ""),
    ("notes.md", "Based on the specification (December 2024).\nLast Updated: 2026-01-07", "2026-01-07"),
    ("notes.md", "Example:\n```python\ndate = '2025-02-01'\n```", ""),
    ("2025-record.pdf", "报告日期：2025年3月2日", "2025-03-02"),
    ("notes.md", "Published: May 12, 2025", "2025-05-12"),
    ("notes.md", "Published: May 12, 2025\nLast Updated: 2026-01-07", "2026-01-07"),
    ("sample.py", "date = '2025-02-01'\nprint(date)", ""),
    ("settings.yaml", "date: 2025-02-01", ""),
    ("sample.py", "# Updated: 2025-03-01\ndate = '2025-02-01'", "2025-03-01"),
    ("article.pdf", "TITLE: Process planning\n原创 编辑甲 示例刊物 2025年4月3日 10:20\nArticle text", "2025-04-03"),
    ("article.md", "Process planning\nBy Example Editor, April 3, 2025\nArticle text", "2025-04-03"),
])
def test_document_dates_have_document_evidence(path, text, expected):
    from kb_pipeline.graph.temporal import document_axis
    assert document_axis(path, text)["date"] == expected


def test_packed_code_keeps_all_symbols_and_external_targets(tmp_path):
    from kb_pipeline.parsers.code_python import python_symbol_blocks
    from kb_pipeline.chunking.chunker import blocks_to_chunks
    from kb_pipeline.graph.deterministic import DeterministicExtractor, UPPER, ALLOWED_ENDS
    text = '''from example_sdk.reader import fetch as load
class Counter:
    def __init__(self):
        self.value = 0
    def read(self):
        return self.value
    def clear(self):
        self.value = 0
    def remote(self):
        return load()
'''
    path = tmp_path / "sample.py"; path.write_text(text)
    blocks = python_symbol_blocks(text)
    chunks = blocks_to_chunks(kb_id="sample", file_key=1, content_version="a", parser_profile="py-symbols-v2",
                              blocks=blocks, max_tokens=400, overlap_tokens=80)
    u = unit("\n".join(c.text for c in chunks), path="sample.py")
    u.block_ids = [c.block.block_id for c in chunks]
    u.source_locations = [{"locator": c.block.metadata.get("source_locator", {})} for c in chunks]
    out = DeterministicExtractor([u], file_for=lambda _: path).extract(u)
    names = {e["name"]: e["type"] for e in out.entities}
    assert {"Counter", "Counter.__init__", "Counter.read", "Counter.clear", "Counter.remote"} <= names.keys()
    assert len([r for r in out.relations if r["predicate"] == "has_method"]) == 4
    calls = [r for r in out.relations if r["predicate"] == "calls"]
    assert any(r["target"] == "example_sdk.reader.fetch" for r in calls)
    assert all(UPPER[names[r["target"]]] in ALLOWED_ENDS["calls"][1] for r in calls)


def test_derived_candidates_are_preserved_but_not_traversed():
    from kb_pipeline.graph.quality import relation_quality, traversable_relation
    row = {"predicate": "variant_of", "derived": True, "evidence": 0}
    assert relation_quality(row) == "needs_review"
    assert not traversable_relation(row)
    assert traversable_relation({"predicate": "variant_of", "evidence": 2})


def test_series_query_uses_explicit_time_scope():
    from kb_search.graphwalk import series_members, time_scope, in_effect
    class Session:
        def run(self, query, **params):
            self.query, self.params = query, params
            return SimpleNamespace(data=lambda: [])
    s = Session(); scope = time_scope(as_of="2025-06-01")
    series_members(s, "kb", "v", ["series"], scope=scope)
    assert in_effect("f", observed="observed", scope=scope) in s.query
    assert s.params["day"] == "2025-06-01"
    series_members(s, "kb", "v", ["series"])
    assert in_effect("f", observed="observed", scope=time_scope()) in s.query


def test_overlapping_cells_keep_distinct_rows_and_all_evidence():
    from kb_pipeline.graph.tables import TableExtractor, deduplicate_cells
    from kb_pipeline.graph.extract import ExtractionSchema
    from kb_pipeline.graph.facts import link_facts
    a = unit("SHEET: records\nROWS: 2-3\nHEADER: name | amount\nA | 7\nB | 8", "u1")
    b = unit("SHEET: records\nROWS: 3-4\nHEADER: name | amount\nB | 8\nB | 8", "u2")
    builder = TableExtractor([a, b], schema=ExtractionSchema(entity_types=("Item",)))
    builder.load_profiles({key: {"row_type": "Item", "name_column": "name"}
                           for entries in builder.unit_tables.values() for key, _ in entries})
    facts = builder.facts(a) + builder.facts(b)
    link_facts(facts, [], units_by_id={u.unit_id: u for u in [a, b]})
    out, merged = deduplicate_cells(facts)
    assert (len(out), merged) == (3, 1)
    shared = next(f for f in out if set(f["point_ids"]) == {"u1", "u2"})
    assert set(shared["evidence_unit_ids"]) == {"u1", "u2"}
    assert len({f["id"] for f in out}) == 3
    from kb_pipeline.graph.tables import parse_table_text
    sparse = parse_table_text("ROWS: 2-6\nHEADER: name | amount\nA | 7\nB | 8")
    assert all("row_number" not in r for r in sparse[0]["rows"])


def test_response_budget_never_discards_already_extracted_facts():
    from kb_pipeline.graph.facts import FactExtractor
    response = json.dumps({"facts": [{"subject": "Sample", "property": f"field{i}", "value": str(i)} for i in range(167)]})
    client = SimpleNamespace(chat=lambda *a, **kw: response)
    out = FactExtractor(client, max_facts=50).extract(unit("data", path="table.pdf"))
    assert len(out.facts) == 167
    assert out.stats["partial"] == out.stats["capped"] == 0


def test_dense_native_tables_partition_with_column_context():
    from kb_pipeline.graph.tables import fact_request_parts
    from kb_pipeline.graph.facts import FactExtractor
    text = "SHEET: readings\nROWS: 2-31\nHEADER: item | pressure | speed\n" + "\n".join(
        f"Item{i} | {100+i} | {200+i}" for i in range(30))
    u = unit(text)
    parts = fact_request_parts(u, 20)
    assert len(parts) == 5
    assert all("pressure:" in p and "speed:" in p for p in parts)
    assert all(f"item: Item{i} | pressure: {100+i} | speed: {200+i}" in "\n".join(parts) for i in range(30))
    assert all(not fact_request_parts(unit(p), 20) for p in parts)
    responses = iter(json.dumps({"facts": [{"subject": f"Item{i}", "property": prop, "value": str(base+i)}
                        for i in range(j*6, j*6+6) for prop,base in [("pressure",100),("speed",200)]]}) for j in range(5))
    result = FactExtractor(SimpleNamespace(chat=lambda *a, **kw: next(responses)), max_facts=20).extract(u)
    assert len(result.facts) == 60
    assert result.calls == 5
    assert result.stats["partial"] == 0


def test_gradual_topics_keep_registry_identity_without_blocking_new_types():
    from kb_pipeline.ontology import normalize_model, dump_linkml, load_linkml, to_linkml, from_linkml, expand_model
    from kb_pipeline.graph.schema_reuse import reuse_supported_model, supported_definitions
    from kb_pipeline.limits import normalize_predicates
    old = normalize_model({"types": [{"name": f"Category{i}", "description": f"Category {i}", "is_a": "entity"}
                                    for i in range(32)], "predicates": []})[0]
    registry = {t["name"]: deepcopy(t) for t in old["types"]}
    for wave in range(4):
        proposed = normalize_model({"types": [{"name": f"Topic{wave}Item{i}", "is_a": "entity"} for i in range(12)],
                                    "predicates": [{"name": f"rel_{wave}_{i}"} for i in range(12)]})[0]
        # New material gradually dominates the sample. Older documents remain.
        support = {"types": list(registry), "predicates": [p["name"] for p in old["predicates"]]}
        result, _ = reuse_supported_model(proposed, {"schema_model": old, "source_support": support})
        assert all(next(t for t in result["types"] if t["name"] == k) == v for k, v in registry.items())
        assert {t["name"] for t in proposed["types"]} <= {t["name"] for t in result["types"]}
        registry = {t["name"]: deepcopy(t) for t in result["types"]}; old = result
    assert len(old["types"]) == 80
    assert len(normalize_predicates(expand_model(old)["graph_predicates"])) == 48
    restored, _ = from_linkml(load_linkml(dump_linkml(to_linkml(old, name="synthetic"))))
    assert restored == old
    assert supported_definitions({"entities": [], "relations": []}, old, set()) == {"types": [], "predicates": []}


def test_endpoint_evolution_requires_grounded_additive_evidence():
    from kb_pipeline.ontology import normalize_model
    from kb_pipeline.graph.schema_reuse import reuse_supported_model
    old = normalize_model({"types": [{"name": "Device", "is_a": "entity"}, {"name": "Component", "is_a": "part"}],
        "predicates": [{"name": "contains", "description": "Contains a constituent", "source_parents": ["Device"],
                        "target_parents": ["Component"]}]})[0]
    proposed = deepcopy(old)
    proposed["types"].append({"name": "Assembly", "is_a": "entity", "description": "A named assembly"})
    proposed["predicates"][0]["target_parents"].append("Assembly")
    prior = {"schema_model": old, "source_support": {"types": ["Device", "Component"], "predicates": ["contains"]}}
    quote = "Device A contains Assembly B as its replaceable constituent."
    evidence = {"contains": [{"source": "Device A", "target": "Assembly B", "source_type": "Device",
                              "target_type": "Assembly", "quote": quote}]}
    for samples in [[], ["An unrelated statement."]]:
        out, guard = reuse_supported_model(proposed, prior, endpoint_evidence=evidence, samples=samples)
        assert out["predicates"][0]["target_parents"] == ["Component"]
        assert not guard["widened"]
    out, guard = reuse_supported_model(proposed, prior, endpoint_evidence=evidence, samples=[quote])
    assert out["predicates"][0]["target_parents"] == ["Component", "Assembly"]
    assert len(guard["widened"]) == 1
    proposed["predicates"][0]["description"] = "Is contained by"
    out, guard = reuse_supported_model(proposed, prior, endpoint_evidence=evidence, samples=[quote])
    assert out["predicates"][0]["target_parents"] == ["Component"]


def test_induction_can_reference_retained_types_omitted_from_new_proposal():
    from kb_pipeline.ontology import normalize_model
    from kb_pipeline.graph.schema import suggest
    old = normalize_model({"types": [{"name": "Device", "is_a": "entity"}, {"name": "Component", "is_a": "part"}],
        "predicates": [{"name": "contains", "description": "Contains a constituent", "source_parents": ["Device"],
                        "target_parents": ["Component"]}]})[0]
    quote = "Device A contains Assembly B as its replaceable constituent."
    predicate = {**old["predicates"][0], "target_parents": ["Component", "Assembly"],
                 "endpoint_evidence": [{"source": "Device A", "target": "Assembly B", "source_type": "Device",
                                        "target_type": "Assembly", "quote": quote}]}
    replies = iter(["equipment", "English", "An expert in source evidence.",
        json.dumps({"entity_types": ["Assembly"]}), json.dumps({"parent_types": {"Assembly": "entity"}}),
        json.dumps({"predicates": [predicate]}), json.dumps({"definitions": {"Assembly": "A named assembly"}}),
        json.dumps({"axis": "none"})])
    out = suggest(SimpleNamespace(chat=lambda *a, **kw: next(replies)), [quote], examples=False,
                  prior={"schema_model": old, "source_support": {"types": ["Device", "Component"], "predicates": ["contains"]}})
    assert {t["name"] for t in out["model"]["types"]} == {"Device", "Component", "Assembly"}
    assert out["model"]["predicates"][0]["target_parents"] == ["Component", "Assembly"]


def test_fact_format_retry_preserves_quoted_source():
    from kb_pipeline.graph.facts import FactExtractor
    from kb_pipeline.graph.llm import ChatClient, LLMSpec
    evidence = 'Set mode to "fast" and path to C:\\data.'
    good = json.dumps({'facts': [{'subject': 'Tool', 'property': 'mode', 'value': 'fast', 'evidence': evidence}]})
    calls = []
    def respond(spec, messages, **kwargs):
        calls.append(messages)
        if len(calls) == 1:
            return '{"facts": [{"subject": "Tool", "evidence": "Set mode to "fast"."}]}'
        assert 'escape embedded double quotes' in messages[-1]['content']
        return good
    client = ChatClient(LLMSpec(name='test', base_url='http://example.invalid', api_key='', model_id='test'), chat=respond)
    result = FactExtractor(client).extract(unit(evidence, path='guide.md'))
    assert len(calls) == 2
    assert result.facts[0]['evidence'] == evidence
    assert not result.stats['partial']
