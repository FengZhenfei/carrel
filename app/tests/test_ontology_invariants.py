"""General fact/identity/source invariants; no enrolled files or KB-specific expectations."""
from __future__ import annotations

import itertools
import json
import tempfile
import unittest
from pathlib import Path

from kb_pipeline.chunking.chunker import blocks_to_chunks
from kb_pipeline.graph.deterministic import DeterministicExtractor
from kb_pipeline.graph.facts import classify_value, comparable_number, link_facts, normalize_fact, normalize_measurements
from kb_pipeline.graph.resolution import auto_merge_pairs, merge_entities
from kb_pipeline.graph.temporal import find_dates, name_date, parse_date
from kb_pipeline.graph.units import ChunkRef, Unit, build_units
from kb_pipeline.parsers.router import parse_native


def entity(key, title, upper="concept", etype="term", docs=None):
    return {"key": key, "title": title, "type": etype, "upper": upper, "scope": "", "doc_ids": docs or ["d"], "frequency": 1}


class OntologyInvariantsTests(unittest.TestCase):
    def test_fact_evidence_agrees_across_graph_vectors_and_api(self):
        from _support import _bundle_dir
        from kb_pipeline.models import KBSource
        from kb_pipeline.graph.neo4j_import import load_bundle
        from kb_pipeline.graph.vectors import spec_payload
        from kb_pipeline.graph.recall import spec_result_row
        from kb_search.service import SPEC_FIELDS, FACT_FIELDS

        source = KBSource(kb_id="kb_003", collection="kb_003", source_root="fixtures", source_type="local_mirror",
                          max_tokens=400, overlap_tokens=80)
        with tempfile.TemporaryDirectory() as tmp:
            out, graph, units, _ = _bundle_dir(self, tmp)
            u = units[0]
            base = dict(subject="Item", property="Attribute", value="complete value", unit_id=u.unit_id, doc_id=u.doc_id)
            graph["specs"] = [{**base, "id": "legacy"},
                              {**base, "id": "field", "evidence_unit_ids": [u.unit_id, u.unit_id],
                               "evidence_point_ids": [u.point_ids[0]], "point_ids": [u.point_ids[0]],
                               "source_locator": {"kind": "lines", "start": 4, "end": 4}, "evidence_precision": "field",
                               "value_json": '"complete value"', "subject_origin": "explicit"},
                              {**base, "id": "document", "evidence_unit_ids": [], "evidence_point_ids": [],
                               "evidence_precision": "document", "point_ids": []}]
            (out/"graph.json").write_text(json.dumps(graph))
            bundle = load_bundle(out, source=source, graph_version="v1")
            by_id = {f["id"]: f for f in bundle["specs"]}
            edges = [e["spec_uid"] for e in bundle["states_edges"]]
            self.assertEqual(edges.count(by_id["legacy"]["uid"]), 1)
            self.assertEqual(edges.count(by_id["field"]["uid"]), 1)
            self.assertNotIn(by_id["document"]["uid"], edges)
            fact = graph["specs"][1]
            row = spec_result_row(spec_payload(fact, kb_id=source.kb_id, source_collection=source.collection, graph_version="v1"))
            for key in ("source_locator", "evidence_precision", "subject_origin", "value_json", "evidence_unit_ids"):
                self.assertEqual(row[key], fact[key])
                self.assertIn(key, SPEC_FIELDS)
                self.assertIn(key, FACT_FIELDS)

    def test_frontmatter_fact_is_not_duplicated_across_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"guide.md"
            path.write_text('---\nname: Example\nusage: ' + 'long description ' * 300 + '\n---\n# Guide\nText.\n')
            chunks = blocks_to_chunks(kb_id="fixture", file_key=1, content_version="v", parser_profile="p",
                                      blocks=parse_native(path), max_tokens=80, overlap_tokens=0)
            refs = [ChunkRef(str(i), c.chunk_uid, "d", "v", i, c.block.block_id, c.block.block_type, [], c.text, c.token_count,
                            rel_path="guide.md") for i, c in enumerate(chunks)]
            units = build_units(refs, kb_id="fixture", unit_chunks=1)
            det = DeterministicExtractor(units, file_for=lambda rel: Path(tmp)/rel)
            facts = [f for u in units for f in det.facts(u)]
            descriptions = [f for f in facts if f["property"] == "usage"]
            self.assertEqual(len(descriptions), 1)
            self.assertEqual(descriptions[0]["value"], ('long description ' * 300).strip())
            self.assertEqual(descriptions[0]["evidence_precision"], "document")
            self.assertEqual(descriptions[0]["evidence_point_ids"], [])

    def test_identifiers_and_unknown_magnitudes_never_lose_suffixes(self):
        for value in ("8123a", "8123a/e", "9C", "128K", "1M", "12XYZ", "1e999"):
            with self.subTest(value=value):
                fact = normalize_fact({"subject": "Item", "property": "Attribute", "value": value})
                self.assertIsNone(fact["value_num"])
                self.assertIsNone(comparable_number(fact))
                self.assertEqual(fact["value"], value)
        self.assertEqual(classify_value(0)["num"], 0)
        self.assertIsNone(classify_value("1e308M", unit="count")["num"])

    def test_measurements_and_explicit_magnitude_context_survive(self):
        for value, unit, expected in [("2.7V", "", 2.7), ("-40°C", "", -40), ("10 mA", "", 10),
                                      ("1M", "bit", 1e6), ("128Ki", "B", 131072), ("128K", "bit", 128000),
                                      ("9 C", "°C", 9), ("$1,234", "", 1234),
                                      ("4.2K", "token", 4200), ("50M", "reads", 50000000), ("98B", "parameters", 98e9),
                                      ("15 minutes", "", 15), ("30 seconds", "", 30),
                                      ("50 bp", "", 50), ("0.25pt", "", 0.25)]:
            fact = normalize_fact({"subject": "Device", "property": "Measurement", "value": value, "unit": unit})
            self.assertEqual(fact["value_num"], expected, (value, unit))
            self.assertEqual(comparable_number(fact), expected)
        self.assertEqual(classify_value("1M to 2M", unit="bit")["hi"], 2e6)
        self.assertEqual(classify_value("-1.1 to -0.25")["hi"], -0.25)
        self.assertIsNone(classify_value("20.03 LTS")["num"])
        self.assertIsNone(classify_value("12L")["num"])
        self.assertIsNone(classify_value("2 foo to 4 bar")["num"])
        self.assertEqual(classify_value("2 foo to 4 bar")["kind"], "text")

    def test_ambiguous_acronym_cannot_merge_directly_or_transitively(self):
        originals = [entity("a", "Signal Transfer (ST)"), entity("b", "Storage Tier (ST)"), entity("s", "ST")]
        for rows in itertools.permutations(originals):
            self.assertEqual(auto_merge_pairs(list(rows)), [])
            for pairs in ([(0, 1), (1, 2)], [(1, 2), (0, 1)], [(0, 2), (0, 1)]):
                merged, _, _ = merge_entities(list(rows), [], pairs, require_compatible_identity=True)
                self.assertEqual(len(merged), 3)

    def test_unambiguous_aliases_and_bilingual_expansions_still_merge(self):
        rows = [entity("a", "Signal Transfer (ST)"), entity("b", "ST"), entity("c", "ST (Signal Transfer,信号传输)")]
        pairs = auto_merge_pairs(rows)
        merged, _, _ = merge_entities(rows, [], pairs, require_compatible_identity=True)
        self.assertEqual(len(merged), 1)
        self.assertEqual(set(merged[0]["member_keys"]), {"a", "b", "c"})
        different = [entity("a", "ST", etype="role"), entity("b", "ST", etype="component")]
        self.assertEqual(auto_merge_pairs(different), [])

    def test_explicit_comparison_owner_wins_over_path_and_coarse_type(self):
        owner = entity("o", "Organization A", "entity", "organization", ["d", "e"])
        unit = Unit("u", "d", "Organization A/report.md", [], [], [], [], 100,
                    "Peer average | Rate | 12%\nOrganization A | Rate | 10%")
        for upper in ("property", "process", "event", "entity", "concept"):
            ents = [owner, entity("p", "Peer average", upper, "metric")]
            for origin in (None, "explicit"):
                fact = normalize_fact({"subject": "Peer average", "property": "Rate", "value": "12%", "subject_origin": origin})
                fact.update(unit_id="u", doc_id="d")
                normalize_measurements([fact], ents, units_by_id={"u": unit}, profile={"subject_types": ["organization"], "axis": "date"},
                                       documents={"d": "Organization A/report.md", "e": "Organization A/older.md"})
                self.assertEqual(fact["subject"], "Peer average", (upper, origin))
        inferred = normalize_fact({"subject": "Rate", "property": "value", "value": "10%", "subject_origin": "inferred"})
        inferred.update(unit_id="u", doc_id="d")
        normalize_measurements([inferred], [owner, entity("p", "Rate", "property", "metric")], units_by_id={"u": unit},
                               profile={"subject_types": ["organization"]}, documents={"d": "Organization A/report.md"})
        self.assertEqual(inferred["subject"], "Rate")  # directories are candidate context, not ownership proof

    def test_invalid_calendar_dates_do_not_become_coarser_valid_dates(self):
        for value in ("2024/06/31", "2023-02-29", "2024-13-01", "2024年2月30日", "20240230", "April 31, 2024", "April 99, 2024", "31 April 2024"):
            self.assertIsNone(parse_date(value), value)
            self.assertEqual(find_dates(value), [], value)
            self.assertIsNone(name_date(value + ".md"), value)
        for value, expected in [("2024-02-29", "2024-02-29"), ("2024/06", "2024-06"), ("2024年", "2024")]:
            self.assertEqual(parse_date(value), expected)
        self.assertEqual(find_dates("wrong 2023-02-29 then 2024-02-29"), ["2024-02-29"])

    def test_large_config_all_fields_full_values_and_actual_evidence(self):
        data = {f"field_{i}": f"value {i}: " + "long instruction. " * 30 for i in range(241)}
        data.update({"empty": "", "null": None, "flag": False, "zero": 0, "a.b": "literal", "a": {"b": "nested"}})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "project/config.json"
            path.parent.mkdir(); path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
            chunks = blocks_to_chunks(kb_id="fixture", file_key=1, content_version="v", parser_profile="p",
                                      blocks=parse_native(path), max_tokens=80, overlap_tokens=0)
            refs = [ChunkRef(str(i), c.chunk_uid, "d", "v", i, c.block.block_id, c.block.block_type, [], c.text, c.token_count,
                             rel_path="project/config.json", source_locator=c.block.metadata.get("source_locator") or {}) for i, c in enumerate(chunks)]
            units = build_units(refs, kb_id="fixture", unit_chunks=3)
            det = DeterministicExtractor(units, file_for=lambda rel: Path(tmp) / rel)
            facts = [f for unit in units for f in det.facts(unit)]
            self.assertEqual(len(facts), 247)
            self.assertEqual(len({f["id"] for f in facts}), len(facts))
            self.assertGreater(len({f["unit_id"] for f in facts}), 1)
            values = {f["property"]: f for f in facts}
            for i in range(241):
                self.assertEqual(values[f"field_{i}"]["value"], data[f"field_{i}"].strip())
            self.assertNotEqual(values['["a.b"]']["value"], values["a.b"]["value"])
            link_facts(facts, [], units_by_id={u.unit_id: u for u in units})
            by_point = {c.point_id: c for c in refs}
            for f in facts:
                self.assertEqual(f["evidence_precision"], "field")
                self.assertTrue(f["point_ids"])
                evidence = "".join(by_point[p].text for p in f["point_ids"])
                raw = json.loads(f["value_json"])
                self.assertIn(json.dumps(raw, ensure_ascii=False), evidence)
            self.assertEqual(facts, [f for unit in units for f in det.facts(unit)])
