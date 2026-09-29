"""Extraction and entity resolution: unit splitting, output parsing, merging, sameness judgement, noise reduction,
ontology."""
from __future__ import annotations

import json
import tempfile
import time
import unittest
from typing import Any
from pathlib import Path
from types import SimpleNamespace

from kb_pipeline import db
from kb_pipeline.graph import prompts, resolution, schema as graph_schema
from kb_pipeline.graph.extract import ExtractionSchema, GraphExtractor, consolidate, entity_key, extraction_fingerprint, normalize_name, parse_records, render_extract_prompt
from kb_pipeline.graph.llm import ChatClient, HTTPStatusError, LLMCache, LLMSpec
from kb_pipeline.graph.merge import attribute_mentions, compute_weights, is_negated, merge_extractions, pagerank
from kb_pipeline.graph.units import ChunkRef, Unit, build_units, dedupe_overlap, read_units, unit_id_for, write_units
from kb_pipeline.models import KBSource

from _support import SPEC, _CodexAudit20260906TestsSupport, _CodexFinalTestsSupport, _bundle_dir, _chunk, _client, _ext, _fake_chat_client, _repo_file, _unit


class UnitBuildingTests(unittest.TestCase):
    def test_overlap_between_consecutive_chunks_is_removed(self) -> None:
        self.assertEqual(dedupe_overlap("A sentence one. B sentence two.", "B sentence two. C sentence three."),
                         "C sentence three.")
        # An overlap that is too short does not count (coincidentally identical characters)
        self.assertEqual(dedupe_overlap("abc", "c def"), "c def")
        self.assertEqual(dedupe_overlap("", "x"), "x")

    def test_units_group_n_consecutive_chunks(self) -> None:
        chunks = [_chunk(i, f"第一节内容{i}。", block=f"b{i}", tokens=300) for i in range(5)]
        by3 = build_units(chunks, kb_id="kb_003", unit_chunks=3)
        self.assertEqual([len(u.chunk_uids) for u in by3], [3, 2])
        self.assertEqual(by3[0].block_ids, ["b0", "b1", "b2"])
        self.assertEqual([u.order for u in by3], [0, 1])
        self.assertEqual([len(u.chunk_uids) for u in build_units(chunks, kb_id="kb_003", unit_chunks=2)], [2, 2, 1])
        # 1 = no merging: one chunk per unit
        single = build_units(chunks, kb_id="kb_003", unit_chunks=1)
        self.assertEqual([len(u.chunk_uids) for u in single], [1, 1, 1, 1, 1])
        self.assertEqual(single[0].text, "第一节内容0。")
        # Content-addressed: the same text + the same positions give the same id; chunk uids carrying
        # content_version are no longer accepted (health check R10)
        self.assertEqual(by3[0].unit_id, unit_id_for("kb_003", "kb_003:1", by3[0].text, positions=[0, 1, 2]))
        self.assertNotEqual(by3[0].unit_id, unit_id_for("kb_003", "kb_003:1", by3[0].text, positions=[3, 4, 5]))
        self.assertNotEqual(by3[0].unit_id, unit_id_for("kb_003", "kb_003:1", by3[0].text, by3[0].chunk_uids))

    def test_section_change_breaks_only_after_half_a_unit(self) -> None:
        """Tiny sections of a few dozen tokens one after another: a section change does not break the current unit
        until it has accumulated at least half a unit, and the Section line lists every section crossed."""
        chunks = [
            _chunk(0, "引脚 A。", section=("引脚", "A"), tokens=80),
            _chunk(1, "引脚 B。", block="b2", section=("引脚", "B"), tokens=80),
            _chunk(2, "引脚 C。", block="b3", section=("引脚", "C"), tokens=80),
            _chunk(3, "时序。", block="b4", section=("时序",), tokens=600),
            _chunk(4, "封装。", block="b5", section=("封装",), tokens=100),
        ]
        units = build_units(chunks, kb_id="kb_003", unit_chunks=3)      # breaks at a section change only after 2 accumulated
        self.assertEqual([len(u.chunk_uids) for u in units], [2, 2, 1])
        self.assertEqual(units[0].section_label, "引脚 > A | 引脚 > B")
        self.assertEqual(units[1].section_label, "引脚 > C | 时序")
        self.assertEqual(units[2].section_label, "封装")
        # Without merging it is naturally one unit per section
        self.assertEqual([len(u.chunk_uids) for u in build_units(chunks, kb_id="kb_003", unit_chunks=1)], [1, 1, 1, 1, 1])

    def test_table_block_stays_whole_when_it_fits_twice_the_budget(self) -> None:
        chunks = [
            _chunk(0, "引言。", tokens=700),
            _chunk(1, "| a | b |\n| 1 | 2 |", block="t1", block_type="table", tokens=500),
            _chunk(2, "| a | b |\n| 3 | 4 |", block="t1", block_type="table", tokens=500),
            _chunk(3, "| a | b |\n| 5 | 6 |", block="t1", block_type="table", tokens=500),
            _chunk(4, "结语。", block="b9", tokens=100),
        ]
        units = build_units(chunks, kb_id="kb_003", unit_chunks=3)
        # The three-chunk table following the intro would exceed 3 chunks, but ≤ 2×3, so the whole block starts its own unit
        self.assertEqual([len(u.chunk_uids) for u in units], [1, 3, 1])
        # Without merging, tables are no exception: each chunk is its own unit
        self.assertEqual([len(u.chunk_uids) for u in build_units(chunks, kb_id="kb_003", unit_chunks=1)], [1, 1, 1, 1, 1])
        self.assertEqual(units[1].block_ids, ["t1"])
        self.assertEqual(units[1].block_types, ["table"])

    def test_consecutive_chunks_of_one_block_are_joined_without_duplicate_text(self) -> None:
        chunks = [
            _chunk(0, "The device has 8 pins. Pin 1 is VDD.", tokens=100),
            _chunk(1, "Pin 1 is VDD. Pin 2 is GND.", tokens=100),
        ]
        units = build_units(chunks, kb_id="kb_003", unit_chunks=3)
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].text.count("Pin 1 is VDD."), 1)
        self.assertIn("Pin 2 is GND.", units[0].text)

    def test_identical_text_in_two_places_yields_two_units(self) -> None:
        """The same table appears twice in a datasheet: two units, two ids, so Neo4j ends up with two TextUnits."""
        chunks = [_chunk(0, "| a | b |", block="t1", block_type="table", section=("A",), tokens=500),
                  _chunk(1, "| a | b |", block="t2", block_type="table", section=("B",), tokens=500)]
        units = build_units(chunks, kb_id="kb_003", unit_chunks=1)
        self.assertEqual(len(units), 2)
        self.assertNotEqual(units[0].unit_id, units[1].unit_id)

    def test_units_round_trip_through_jsonl(self) -> None:
        units = build_units([_chunk(0, "文本。", tokens=10)], kb_id="kb_003")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "units.jsonl"
            self.assertEqual(write_units(path, units), 1)
            back = read_units(path)
        self.assertEqual(back[0].to_json(), units[0].to_json())
        self.assertEqual(back[0].section_label, "第一章")


class ExtractionOutputParserTests(unittest.TestCase):
    SCHEMA = ExtractionSchema(entity_types=("organization", "person"),
                              predicates=({"name": "chairs", "description": "", "source_parents": [], "target_parents": []},))

    def test_complete_marker_glued_to_last_record_does_not_corrupt_fields(self) -> None:
        text = ('("entity"<|>Central Institution<|>ORGANIZATION<|>The bank)\n##\n'
                '("relationship"<|>Martin Smith<|>Central Institution<|>chairs<|>Martin chairs it<|>9)\n<|COMPLETE|>')
        entities, relations, stats = parse_records(text, self.SCHEMA)
        self.assertEqual(entities[0]["type"], "organization")           # normalised to the spelling in the configuration
        self.assertEqual(relations[0]["strength"], 9.0)                  # upstream glued ")\\n<|COMPLETE|>" into the strength, making it 1.0
        self.assertEqual(relations[0]["predicate"], "chairs")
        self.assertEqual(stats["malformed"], 0)

    def test_five_field_relationship_without_predicate_falls_back(self) -> None:
        text = '("relationship"<|>A<|>B<|>A owns B<|>5)'
        _, relations, stats = parse_records(text, self.SCHEMA)
        self.assertEqual((relations[0]["predicate"], relations[0]["strength"]), (prompts.DEFAULT_PREDICATE, 5.0))
        # Predicate not in the table → related_to, and counted as drift
        _, relations, stats = parse_records('("relationship"<|>A<|>B<|>owns<|>A owns B<|>5)', self.SCHEMA)
        self.assertEqual(relations[0]["predicate"], prompts.DEFAULT_PREDICATE)
        self.assertEqual(relations[0]["predicate_raw"], "owns")
        self.assertEqual(stats["unknown_predicates"], 1)

    def test_quotes_parens_and_unknown_types_are_tolerated(self) -> None:
        text = '（entity<|>Zhang San<|>Person<|>desc）\n##\n(entity<|>Acme<|>company<|>x)\n##\ngarbage'
        entities, _, stats = parse_records(text, self.SCHEMA)
        self.assertEqual([e["type"] for e in entities], ["person", "company"])
        self.assertEqual(stats["unknown_types"], 1)
        self.assertEqual(stats["malformed"], 1)

    def test_consolidate_merges_within_a_unit(self) -> None:
        entities = [{"name": "Acme", "type": "organization", "description": "a"},
                    {"name": "ACME", "type": "organization", "description": "b"},
                    {"name": "acme ", "type": "company", "description": "a"}]
        relations = [{"source": "Acme", "target": "Bob", "predicate": "related_to", "description": "x", "strength": 2},
                     {"source": "ACME", "target": "bob", "predicate": "related_to", "description": "y", "strength": 7},
                     {"source": "Bob", "target": "Bob", "predicate": "related_to", "description": "self", "strength": 1}]
        ents, rels = consolidate(entities, relations)
        self.assertEqual(len(ents), 1)
        self.assertEqual((ents[0]["mentions"], ents[0]["descriptions"]), (3, ["a", "b"]))
        self.assertEqual(ents[0]["types"], {"organization": 2, "company": 1})
        self.assertEqual(len(rels), 1)                 # self-loop dropped, same key merged
        self.assertEqual((rels[0]["strength"], rels[0]["descriptions"]), (7.0, ["x", "y"]))

    def test_gleaning_rounds_and_yes_no_gate(self) -> None:
        first = '("entity"<|>A<|>ORGANIZATION<|>first)'
        cont = '("entity"<|>B<|>ORGANIZATION<|>second)<|COMPLETE|>'
        client = _client([first, cont, "y\n", '("entity"<|>C<|>PERSON<|>third)'])
        extractor = GraphExtractor(client, self.SCHEMA, max_gleanings=2)
        unit = Unit(unit_id="u", doc_id="d", rel_path="a.pdf", section_path=["S"], block_ids=[], chunk_uids=[],
                    point_ids=[], n_tokens=500, text="A and B and C")
        result = extractor.extract(unit)
        self.assertEqual(sorted(e["name"] for e in result.entities), ["A", "B", "C"])
        self.assertEqual(result.calls, 4)              # first round + gleaning + Y/N + gleaning
        self.assertEqual(result.stats["gleanings"], 2)
        # N stops it: with two gleaning rounds configured there are only 3 calls
        client = _client([first, cont, " n "])
        result = GraphExtractor(client, self.SCHEMA, max_gleanings=2).extract(unit)
        self.assertEqual(result.calls, 3)
        # Units that are too short get no gleaning: done in a single call
        tiny = Unit(unit_id="t", doc_id="d", rel_path="a.pdf", section_path=["S"], block_ids=[], chunk_uids=[],
                    point_ids=[], n_tokens=120, text="A")
        client = _client([first, cont])
        result = GraphExtractor(client, self.SCHEMA, max_gleanings=2).extract(tiny)
        self.assertEqual((result.calls, result.stats["gleanings"]), (1, 0))

    def test_truncated_output_is_asked_again_with_a_bigger_budget(self) -> None:
        """2026-09-12: extraction budget 4096 → 8192; when the output is truncated (finish_reason=length) ask again
        with double the budget; if still truncated keep the longer one and record truncated. Previously this step
        ignored finish_reason, so lost records left no trace."""
        from kb_pipeline.graph.extract import EXTRACT_MAX_TOKENS

        seen: list[int] = []

        def chat(spec, messages, max_tokens=None, meta=None, **_):
            seen.append(int(max_tokens))
            if int(max_tokens) == EXTRACT_MAX_TOKENS:
                if meta is not None:
                    meta.update(finish_reason="length", truncated=True, cached=False)
                return '("entity"<|>A<|>ORGANIZATION<|>cut'
            if meta is not None:
                meta.update(finish_reason="stop", truncated=False, cached=False)
            return '("entity"<|>A<|>ORGANIZATION<|>full)\n##\n("entity"<|>B<|>ORGANIZATION<|>second)<|COMPLETE|>'
        client = ChatClient(SPEC, chat=chat, cache=LLMCache(None), backoff_base=0.0, backoff_max=0.0)
        extractor = GraphExtractor(client, self.SCHEMA, max_gleanings=1)
        self.assertEqual(extractor.max_tokens, 8192)
        tiny = Unit(unit_id="t", doc_id="d", rel_path="a.pdf", section_path=["S"], block_ids=[], chunk_uids=[],
                    point_ids=[], n_tokens=120, text="A and B")
        result = extractor.extract(tiny)
        self.assertEqual(seen, [8192, 16384])
        self.assertEqual((result.calls, result.stats["truncated"]), (2, 0))
        self.assertEqual(sorted(e["name"] for e in result.entities), ["A", "B"])
        # Still truncated at double the budget: keep the longer one, record truncated

        def chat_always_cut(spec, messages, max_tokens=None, meta=None, **_):
            if meta is not None:
                meta.update(finish_reason="length", truncated=True, cached=False)
            return '("entity"<|>A<|>ORGANIZATION<|>cut)' + ('\n##\n("entity"<|>B<|>ORGANIZATION<|>more' if int(max_tokens) > 8192 else "")
        client = ChatClient(SPEC, chat=chat_always_cut, cache=LLMCache(None), backoff_base=0.0, backoff_max=0.0)
        result = GraphExtractor(client, self.SCHEMA, max_gleanings=1).extract(tiny)
        self.assertEqual((result.calls, result.stats["truncated"]), (2, 1))
        self.assertEqual(sorted(e["name"] for e in result.entities), ["A", "B"])   # the longer one is kept

    def test_prompt_carries_section_predicates_language(self) -> None:
        text = render_extract_prompt("正文", section="第二章 > 2.5", schema=ExtractionSchema(
            entity_types=("pin",), predicates=({"name": "has_pin"},), language="Chinese"))
        self.assertIn("Section: 第二章 > 2.5", text)
        self.assertIn("Predicates: has_pin; related_to", text)      # 09-06 evening: predicates carry descriptions and endpoints, semicolon-separated
        self.assertIn("Write all descriptions in Chinese", text)
        self.assertIn("Entity_types: pin\n", text)

    def test_fingerprint_changes_with_model_types_predicates_language_and_gleanings(self) -> None:
        base = ExtractionSchema(entity_types=("a",), predicates=(), language="English")
        fp = extraction_fingerprint(SPEC, base, max_gleanings=1)
        self.assertNotEqual(fp, extraction_fingerprint(SPEC, base, max_gleanings=0))
        self.assertNotEqual(fp, extraction_fingerprint(SPEC, ExtractionSchema(entity_types=("b",)), max_gleanings=1))
        self.assertNotEqual(fp, extraction_fingerprint(SPEC, ExtractionSchema(entity_types=("a",), predicates=({"name": "p"},)), max_gleanings=1))
        self.assertNotEqual(fp, extraction_fingerprint(SPEC, ExtractionSchema(entity_types=("a",), language="Chinese"), max_gleanings=1))
        other = LLMSpec(name="m", base_url="http://x/v1", api_key="", model_id="model-b")
        self.assertNotEqual(fp, extraction_fingerprint(other, base, max_gleanings=1))
        self.assertEqual(fp, extraction_fingerprint(LLMSpec(name="n", base_url="http://y", api_key="k", model_id="model-a"), base, max_gleanings=1))

    def test_names_keep_case_but_keys_do_not(self) -> None:
        self.assertEqual(normalize_name('  "ZK7C1049GN"  '), "ZK7C1049GN")
        self.assertEqual(entity_key("Ｎorthwind  Tech"), "northwind tech")


class MergeTests(unittest.TestCase):
    def test_entities_merge_by_normalized_name_with_majority_type(self) -> None:
        units = [_unit("u1"), _unit("u2", order=1)]
        ext = {
            "u1": _ext([("Northwind", "organization", "maker"), ("SRAM", "memory device", "chip")],
                       [("Northwind", "SRAM", "produces", "makes it", 8)]),
            "u2": _ext([("NORTHWIND", "company", "German company"), ("SRAM", "memory device", "chip")],
                       [("SRAM", "Northwind", "related_to", "made by", 3), ("Ghost", "SRAM", "related_to", "x", 1)]),
        }
        out = merge_extractions(units, ext, entity_types=("organization", "memory device"),
                                parent_types={"organization": "org"},
                                allowed_ends={"produces": ({"org"}, {"component"})})
        ents = {e["key"]: e for e in out["entities"]}
        self.assertEqual(set(ents), {"northwind", "sram"})
        self.assertEqual(ents["northwind"]["title"], "Northwind")          # a tie on occurrences takes the first spelling
        self.assertEqual(ents["northwind"]["descriptions"], ["maker", "German company"])
        self.assertEqual(ents["northwind"]["frequency"], 2)
        self.assertEqual(ents["northwind"]["parent_type"], "org")
        self.assertEqual(ents["sram"]["descriptions"], ["chip"])          # deduplicated
        rels = {(r["source_key"], r["target_key"], r["predicate"]): r for r in out["relations"]}
        self.assertIn(("northwind", "sram", "produces"), rels)             # controlled predicates keep their direction
        self.assertIn(("northwind", "sram", "related_to"), rels)           # the fallback predicate uses sorted endpoints
        self.assertTrue(rels[("northwind", "sram", "produces")]["type_violation"])   # sram's parent type is not component
        self.assertEqual(out["stats"]["orphan_dropped"], 1)               # Ghost has no entity
        self.assertEqual(out["stats"]["schema_drift_types"], 0)           # company lost to organization
        self.assertEqual(out["stats"]["type_violations"], 1)

    def test_negated_and_self_loop_relations_are_dropped(self) -> None:
        self.assertTrue(is_negated("There is no clear relationship between A and B"))
        self.assertTrue(is_negated("两者无明确关系"))
        self.assertFalse(is_negated("A supplies power to B"))
        units = [_unit("u1")]
        ext = {"u1": _ext([("A", "t", "a"), ("B", "t", "b")],
                          [("A", "B", "related_to", "no clear relationship", 1), ("A", "a", "related_to", "self", 1)])}
        out = merge_extractions(units, ext)
        self.assertEqual(out["relations"], [])
        self.assertEqual((out["stats"]["negated_dropped"], out["stats"]["self_loops_dropped"]), (1, 1))

    def test_weights_npmi_and_pagerank(self) -> None:
        # Since 2026-09-12 single-letter names are document-scoped; the fixture uses two capital letters (still global entities)
        units = [_unit(f"u{i}", order=i) for i in range(4)]
        ext = {
            "u0": _ext([("AA", "t", "a"), ("BB", "t", "b")], [("AA", "BB", "related_to", "ab", 9)]),
            "u1": _ext([("AA", "t", "a"), ("BB", "t", "b")], [("AA", "BB", "related_to", "ab", 9)]),
            "u2": _ext([("AA", "t", "a"), ("CC", "t", "c")], [("AA", "CC", "related_to", "ac", 1)]),
            "u3": _ext([("DD", "t", "d"), ("CC", "t", "c")], [("DD", "CC", "related_to", "dc", 1)]),
        }
        out = merge_extractions(units, ext)
        info = compute_weights(out["entities"], out["relations"], n_units=4)
        ents = {e["key"]: e for e in out["entities"]}
        rels = {(r["source_key"], r["target_key"]): r for r in out["relations"]}
        self.assertEqual(ents["aa"]["degree"], 2)
        self.assertEqual(rels[("aa", "bb")]["cooccur"], 2)
        self.assertEqual(rels[("aa", "bb")]["evidence"], 2)
        self.assertEqual(rels[("aa", "bb")]["combined_degree"], 3)
        # NPMI: p(a)=3/4, p(b)=2/4, p(ab)=2/4 → pmi=log(0.5/0.375)>0, within (0,1] after normalisation
        self.assertGreater(rels[("aa", "bb")]["npmi"], 0)
        self.assertLessEqual(rels[("aa", "bb")]["npmi"], 1.0)
        # a-c co-occur only once and a is very common → the negative NPMI is kept
        self.assertLess(rels[("aa", "cc")]["npmi"], rels[("aa", "bb")]["npmi"])
        for r in out["relations"]:
            self.assertGreaterEqual(r["weight"], 1.0)
            self.assertLessEqual(r["weight"], 10.0)
        self.assertGreater(rels[("aa", "bb")]["weight"], rels[("cc", "dd")]["weight"])   # undirected edges use sorted endpoints
        self.assertAlmostEqual(sum(e["pagerank"] for e in out["entities"]), 1.0, places=5)
        self.assertEqual(ents["aa"]["pagerank_norm"], 1.0)
        self.assertEqual(info["n_units"], 4)

    def test_pagerank_handles_isolated_nodes(self) -> None:
        ranks = pagerank(["a", "b", "c"], [("a", "b", 1.0)])
        self.assertAlmostEqual(sum(ranks.values()), 1.0, places=6)
        self.assertGreater(ranks["a"], ranks["c"])

    def test_mentions_are_attributed_to_the_chunks_that_contain_the_name(self) -> None:
        unit = _unit("u1", points=("p1", "p2"))
        entities = [{"key": "vdd", "title": "VDD", "aliases": ["Vdd pin"], "unit_ids": ["u1"]},
                    {"key": "ghost", "title": "Ghost", "aliases": [], "unit_ids": ["u1"]}]
        rows = attribute_mentions(entities, {"u1": unit}, {"p1": "VDD is 1.8V; the vdd pin", "p2": "nothing here"})
        by = {(r["entity_key"], r["point_id"]): r["count"] for r in rows}
        self.assertEqual(by[("vdd", "p1")], 3)                      # VDD + vdd + "vdd pin" each count as one hit
        self.assertNotIn(("vdd", "p2"), by)
        self.assertEqual((by[("ghost", "p1")], by[("ghost", "p2")]), (0, 0))   # no hit: attached to every chunk, count=0


class ResolutionTests(unittest.TestCase):
    def test_prefilter_rules(self) -> None:
        self.assertFalse(resolution.is_similar("ZK7C4021KV13", "ZK7C4041KV13"))
        self.assertTrue(resolution.is_similar("Northwind", "NORTHWIND"))
        self.assertTrue(resolution.is_similar("nvSRAM", "nv-SRAM"))
        self.assertFalse(resolution.is_similar("SRAM", "Static Random Access Memory"))
        self.assertTrue(resolution.is_similar("非易失性存储器", "非易失存储器"))
        self.assertFalse(resolution.is_similar("电源电压", "输入电压"))
        self.assertFalse(resolution.is_similar("电源管理", "管理电源"))          # ordered n-grams: a bag of characters would misjudge this
        self.assertFalse(resolution.is_similar("", "x"))

    def test_candidates_and_prompt(self) -> None:
        ents = [{"title": "Northwind", "type": "ORGANIZATION", "descriptions": ["German maker. Big."]},
                {"title": "NORTHWIND", "type": "ORGANIZATION", "descriptions": []},
                {"title": "Northwind", "type": "PRODUCT"},
                {"title": "ZK7C1049GN", "type": "PRODUCT"}, {"title": "ZK7C1049GN-10", "type": "PRODUCT"}]
        pairs = resolution.candidate_pairs(ents)
        self.assertIn((0, 1), pairs)
        self.assertNotIn((0, 2), pairs)
        self.assertNotIn((3, 4), pairs)
        prompt = resolution.render_batch([(0, 1)], ents)
        self.assertIn('1. "Northwind" [German maker. Big.] | "NORTHWIND" []', prompt)          # sameness judgement sees the first two sentences of the description
        self.assertIn("Entity type: ORGANIZATION", prompt)
        self.assertEqual(resolution.parse_answers("1: yes\n2: no\n3. YES\n4) 否", 5), [True, False, True, False, False])

    def test_components_merge_whole_and_edges_are_rewritten(self) -> None:
        ents = [
            {"key": "northwind", "title": "NORTHWIND", "type": "ORG", "descriptions": ["maker"], "unit_ids": ["t1"], "frequency": 5, "aliases": []},
            {"key": "northwind technologies", "title": "Northwind Technologies", "type": "ORG", "descriptions": ["semiconductor company"], "unit_ids": ["t2", "t1"], "frequency": 2, "aliases": []},
            {"key": "northwind ag", "title": "Northwind AG", "type": "ORG", "descriptions": ["maker"], "unit_ids": ["t3"], "frequency": 1, "aliases": []},
            {"key": "zk7c1049gn", "title": "ZK7C1049GN", "type": "PRODUCT", "descriptions": ["sram"], "unit_ids": ["t1"], "frequency": 3, "aliases": []},
        ]
        rels = [
            {"source_key": "northwind", "target_key": "zk7c1049gn", "source": "NORTHWIND", "target": "ZK7C1049GN", "predicate": "produces", "directed": True, "descriptions": ["makes"], "strength_sum": 2.0, "evidence": 1, "unit_ids": ["t1"]},
            {"source_key": "northwind technologies", "target_key": "zk7c1049gn", "source": "Northwind Technologies", "target": "ZK7C1049GN", "predicate": "produces", "directed": True, "descriptions": ["manufactures"], "strength_sum": 1.0, "evidence": 1, "unit_ids": ["t2"]},
            {"source_key": "northwind ag", "target_key": "northwind", "source": "Northwind AG", "target": "NORTHWIND", "predicate": "related_to", "directed": False, "descriptions": ["same"], "strength_sum": 1.0, "evidence": 1, "unit_ids": ["t3"]},
        ]
        merged, edges, stats = resolution.merge_entities(ents, rels, [(0, 1), (1, 2)])
        self.assertEqual(stats["entities_after"], 2)
        head = next(e for e in merged if e["type"] == "ORG")
        self.assertEqual(head["title"], "NORTHWIND")
        self.assertEqual(head["frequency"], 8)
        self.assertEqual(head["unit_ids"], ["t1", "t2", "t3"])
        self.assertEqual(head["descriptions"], ["maker", "semiconductor company"])
        self.assertEqual(head["aliases"], ["Northwind Technologies", "Northwind AG"])
        self.assertEqual(len(edges), 1)
        self.assertEqual((edges[0]["source"], edges[0]["target"], edges[0]["strength_sum"], edges[0]["evidence"]),
                         ("NORTHWIND", "ZK7C1049GN", 3.0, 2))
        self.assertEqual(edges[0]["unit_ids"], ["t1", "t2"])
        self.assertEqual(stats["self_loops_dropped"], 1)

    def test_resolve_runs_batches_and_tolerates_a_failed_batch(self) -> None:
        # Northwind / NORTHWIND and Northwind / Northwind Technologies are now merged directly without asking the
        # model (identifier equality, organisation suffix); here we use a pair that does need the model (spelling variant)
        ents = [{"key": "a", "title": "Northwind", "type": "ORG", "descriptions": [], "unit_ids": ["t1"], "frequency": 1, "aliases": []},
                {"key": "b", "title": "Northwindn", "type": "ORG", "descriptions": [], "unit_ids": ["t2"], "frequency": 1, "aliases": []}]
        client = _client(["1: yes"])
        merged, _, stats = resolution.resolve(client, ents, [])
        self.assertEqual((stats["candidates"], stats["yes"], stats["entities_after"]), (1, 1, 1))
        client = _client([HTTPStatusError(400, "bad")], attempts=1)
        merged, _, stats = resolution.resolve(client, ents, [])
        self.assertEqual((stats["failed_batches"], stats["entities_after"]), (1, 2))


class NoiseReductionTests(unittest.TestCase):
    """Graph-build noise reduction (Desktop "Graph-build noise reduction plan 2026-09-04"): unit classification,
    value / reference admission, splitting of combined names, deterministic resolution rules and vector candidates,
    section-path attribution, type health check, landing checks for capability questions, recall filtering."""

    def test_unit_classifier_recognises_toc_history_and_listings(self) -> None:
        from kb_pipeline.graph.units import classify_unit_text, unit_signals

        toc = "Contents Pin Configurations ....5 Pin Definitions ....7 Functional Overview ...... 9 Command Cycles ....9 Read and Write ....11 TAP ....22"
        history = ("CAPTION: Document History Page | Document Title: ZK7C4021KV13 | Rev | ECN | Date | Description |\n"
                   "| *J | 4575129 | 11/20/2014 | PRIT | Updated Functional Description |\n"
                   "| *K | 4741050 | 05/05/2015 | PRIT | Updated Ordering Information |\n"
                   "| *L | 5012377 | 12/01/2015 | PRIT | Updated to new template |")
        ballmap = "\n".join("| " + " | ".join(f"{r}{c}" if (r + c) % 3 else "VDD" for c in range(1, 13)) + " |" for r in range(1, 12))
        ac_table = ("| Parameter | Description | 667 MHz Min | 667 MHz Max | 600 MHz Min | Unit |\n"
                    "| tAS | Address setup to CK rising edge, measured from the address transition | 0.160 | - | 0.180 | ns |\n"
                    "| tCK | Clock cycle time, defined between two rising edges of the same clock | 1.5 | 3.333 | 1.667 | ns |")
        prose = "The CK/CK# clock is associated with the address and control pins: A[24:0], LDA#, LDB#, RWA#, RWB#. The CK/CK# transitions are centered with respect to the address and control signal transitions."
        self.assertEqual(classify_unit_text(toc, ["Contents"]), "boilerplate")
        self.assertEqual(classify_unit_text(history, ["Units of Measure"]), "boilerplate")   # heading is no help; relies on the first line and the dates
        self.assertEqual(classify_unit_text(ballmap, ["Pin Configurations"]), "listing")
        self.assertEqual(classify_unit_text(ac_table, ["Switching Characteristics"]), "body")   # parameter tables are body text
        self.assertEqual(classify_unit_text(prose, ["Functional Overview"]), "body")
        self.assertEqual(classify_unit_text("本文档所含信息如有更改,恕不另行通知。版权所有。", ["法律声明"]), "boilerplate")
        self.assertEqual(classify_unit_text("销售、解决方案和法律信息 全球销售和设计支持 赛普拉斯公司拥有一个由办事处组成的全球性网络。", ["销售、解决方案和法律信息"]), "boilerplate")
        self.assertEqual(classify_unit_text("CAPTION: 测量单位 | 符号 | 测量单位 | | --- | --- | | °C | 摄氏度 | | MHz | 兆赫兹 |", ["封装图"]), "boilerplate")
        sig = unit_signals(ballmap)
        self.assertGreaterEqual(sig["table_ratio"], 0.5)
        self.assertLessEqual(sig["cell_len"], 8)

    def test_units_carry_a_kind_and_the_parser_reads_the_unit_record(self) -> None:
        from kb_pipeline.graph.extract import ExtractionResult, extraction_fingerprint, parse_records

        chunks = [_chunk(0, "Contents Pin Configurations ....5 Pin Definitions ....7 Overview ...... 9 Cycles ....9 TAP ....22 Timing ....30")]
        units = build_units(chunks, kb_id="kb_t", unit_chunks=3)
        self.assertEqual(units[0].kind, "boilerplate")
        rt = Unit.from_json(units[0].to_json())
        self.assertEqual(rt.kind, "boilerplate")
        self.assertEqual(Unit.from_json({"unit_id": "u", "doc_id": "d"}).kind, "body")   # old units.jsonl has no such field
        text = '("unit"<|>boilerplate<|>table of contents)##("entity"<|>TAP<|>logic block<|>test port)<|COMPLETE|>'
        ents, rels, stats = parse_records(text, None)
        self.assertEqual(stats["unit_kind"], "boilerplate")
        self.assertEqual(stats["malformed"], 0)
        self.assertEqual(len(ents), 1)
        _, _, stats2 = parse_records('("unit"<|>garbage<|>x)##("entity"<|>A<|>t<|>d)', None)
        self.assertEqual(stats2["unit_kind"], "body")
        self.assertEqual(ExtractionResult("u", [], [], 1, {"unit_kind": "listing"}).unit_kind, "listing")
        self.assertEqual(ExtractionResult("u", [], [], 1, {}).unit_kind, "body")
        # The prompt changed, so the fingerprint must change (otherwise old extractions would count as hits for the new prompt)
        self.assertIn("unit", prompts.GRAPH_EXTRACTION_PROMPT)
        self.assertNotEqual(extraction_fingerprint(SPEC, ExtractionSchema(("a",)), max_gleanings=1)[:4], "")

    def test_latex_names_are_normalised_to_plain_identifiers(self) -> None:
        from kb_pipeline.graph.extract import normalize_name, strip_latex

        self.assertEqual(strip_latex("$t_{\\text{AS}}$"), "tAS")
        self.assertEqual(strip_latex("$V_{CC}$"), "VCC")
        self.assertEqual(strip_latex("$\\overline{\\mathrm{CE}}$"), "CE#")
        self.assertEqual(strip_latex("$t_{JIT(per)}$"), "tJIT(per)")
        self.assertEqual(strip_latex("t_AS"), "t_AS")                      # a non-LaTeX underscore is left alone
        self.assertEqual(strip_latex("t $_{AA}$"), "tAA")
        self.assertEqual(strip_latex("A $_{0-A19}$"), "A0-A19")
        self.assertEqual(strip_latex("芯片使能($\\overline{\\mathrm{CE}}$)"), "芯片使能(CE#)")
        self.assertEqual(strip_latex("ZK7C4021KV13"), "ZK7C4021KV13")
        self.assertEqual(normalize_name("  $t_{\\text{TF}}$ "), "tTF")
        self.assertEqual(entity_key("$t_{\\text{AS}}$"), entity_key("tAS"))
        # Merging recomputes the key from the name: a stored extraction whose name is LaTeX and whose key came from
        # the old rule still merges with the plain-text form
        units = [_unit("u1"), _unit("u2", order=1)]
        ext = {"u1": {"entities": [{"name": "$t_{\\text{AS}}$", "key": "$t_{\\text{as}}$", "type": "timing parameter", "description": "setup"}],
                      "relations": []},
               "u2": {"entities": [{"name": "tAS", "key": "tas", "type": "timing parameter", "description": "address setup"}],
                      "relations": []}}
        out = merge_extractions(units, ext)
        self.assertEqual([e["title"] for e in out["entities"]], ["tAS"])
        self.assertEqual(out["entities"][0]["frequency"], 2)

    def test_combine_unit_kind_prefers_boilerplate_then_listing(self) -> None:
        from kb_pipeline.graph.merge import combine_unit_kind

        self.assertEqual(combine_unit_kind("body", "boilerplate"), "boilerplate")
        self.assertEqual(combine_unit_kind("boilerplate", None), "boilerplate")
        self.assertEqual(combine_unit_kind("listing", "body"), "listing")
        self.assertEqual(combine_unit_kind("body", "listing"), "body")        # the model's listing does not count: it calls spec tables listings too
        self.assertEqual(combine_unit_kind("body", "body"), "body")
        self.assertEqual(combine_unit_kind(None, "nonsense"), "body")

    def test_value_and_reference_names(self) -> None:
        from kb_pipeline.graph.merge import is_reference_name, is_value_name

        for name in ("1.3V", "667 MHz", "2.7 V至3.6 V", "1024 K × 8", "65nm", "-40°C to 85°C", "4021", "3.333 ns", "0.160"):
            self.assertTrue(is_value_name(name), name)
        for name in ("ZK7C4021KV13", "361-ball FCBGA", "8 Mbit", "Commercial", "tAS", "PE#", "QuantumTrap", "边界扫描寄存器"):
            self.assertFalse(is_value_name(name), name)
        for name in ("Figure 5", "Table 20", "图 5", "表 3.2", "001-79553", "Rev *J", "*J", "Note 15", "附录 A"):
            self.assertTrue(is_reference_name(name), name)
        for name in ("Errata", "TAP Timing Diagram", "ZK7C4021KV13", "SRAM"):
            self.assertFalse(is_reference_name(name), name)

    def _merge_input(self):
        u_body = _unit("u1", points=("p1",))
        u_toc = _unit("u2", points=("p2",))
        u_toc.kind = "boilerplate"
        extractions = {
            "u1": {"entities": [
                        {"name": "ZK7C4021KV13", "type": "device", "description": "a QDR SRAM", "mentions": 2},
                        {"name": "1.3V", "type": "operating condition", "description": "core voltage", "mentions": 1},
                        {"name": "tAS", "type": "timing parameter", "description": "address setup", "mentions": 1},
                        {"name": "Figure 5", "type": "document", "description": "block diagram", "mentions": 1},
                        {"name": "ZK7C4041KV13", "type": "device", "description": "the x36 sibling", "mentions": 1},
                        {"name": "ZK7C4021KV13/ZK7C4041KV13", "type": "device", "description": "the family datasheet", "mentions": 1}],
                   "relations": [
                        {"source": "ZK7C4021KV13", "target": "1.3V", "predicate": "has_voltage_level", "description": "VDD is 1.3V", "strength": 8},
                        {"source": "ZK7C4021KV13", "target": "tAS", "predicate": "has_timing_parameter", "description": "tAS min 0.16 ns", "strength": 7},
                        {"source": "ZK7C4021KV13", "target": "Figure 5", "predicate": "has_document", "description": "shown in figure 5", "strength": 3},
                        {"source": "ZK7C4021KV13/ZK7C4041KV13", "target": "tAS", "predicate": "has_timing_parameter", "description": "family has tAS", "strength": 5}]},
            "u2": {"entities": [{"name": "TAP Registers", "type": "document", "description": "section title", "mentions": 1},
                                {"name": "ZK7C4021KV13", "type": "device", "description": "the part", "mentions": 1}],
                   "relations": [{"source": "ZK7C4021KV13", "target": "TAP Registers", "predicate": "has_document", "description": "toc entry", "strength": 4}]},
        }
        return [u_body, u_toc], extractions

    def test_merge_marks_boilerplate_folds_values_and_splits_combined_names(self) -> None:
        units, extractions = self._merge_input()
        out = merge_extractions(units, extractions, entity_types=["device", "timing parameter", "operating condition", "document"])
        ents = {e["key"]: e for e in out["entities"]}
        rels = {(r["source"], r["predicate"], r["target"]): r for r in out["relations"]}
        st = out["stats"]
        # Values do not become nodes; they fold into the device's attribute descriptions
        self.assertNotIn("1.3v", ents)
        self.assertIn("has_voltage_level = 1.3V", ents["zk7c4021kv13"]["attributes"])
        self.assertTrue(any("has_voltage_level = 1.3V" in d for d in ents["zk7c4021kv13"]["descriptions"]))
        self.assertEqual((st["value_entities_dropped"], st["value_relations_folded"]), (1, 1))
        # References are kept but flagged reference, and relations touching them carry the same flag
        self.assertTrue(ents["figure 5"]["reference"])
        self.assertTrue(rels[("ZK7C4021KV13", "has_document", "Figure 5")]["reference"])
        self.assertFalse(rels[("ZK7C4021KV13", "has_timing_parameter", "tAS")]["reference"])
        # A combined name splits into its two parts: the combined node disappears and each device gets its own
        # has_timing_parameter → tAS
        self.assertNotIn("zk7c4021kv13/zk7c4041kv13", ents)
        self.assertEqual(st["combined_names_split"], 1)
        self.assertIn(("ZK7C4041KV13", "has_timing_parameter", "tAS"), rels)
        self.assertIn("ZK7C4021KV13/ZK7C4041KV13", ents["zk7c4041kv13"]["aliases"])
        merged = rels[("ZK7C4021KV13", "has_timing_parameter", "tAS")]
        self.assertEqual(merged["evidence"], 1)                       # same unit, evidence is not counted twice
        self.assertAlmostEqual(merged["strength_sum"], 12.0)          # 7 + 5 merged into the same edge
        # Boilerplate unit: relations and entities are flagged, strength is discounted
        toc_rel = rels[("ZK7C4021KV13", "has_document", "TAP Registers")]
        self.assertTrue(toc_rel["boilerplate"])
        self.assertEqual(toc_rel["evidence_kind"], "boilerplate")
        self.assertAlmostEqual(toc_rel["strength_sum"], 1.0)         # 4 × 0.25
        self.assertTrue(ents["tap registers"]["boilerplate"])
        self.assertFalse(ents["zk7c4021kv13"]["boilerplate"])         # also appeared in body text
        self.assertEqual(ents["zk7c4021kv13"]["evidence_kind"], "body")
        self.assertEqual((st["boilerplate_units"], st["boilerplate_relations"], st["boilerplate_entities"]), (1, 1, 1))
        self.assertIn("type_health", st)
        # The type health check only reports, it does not demote whole types: in type document, Figure 5 is a
        # reference and TAP Registers is boilerplate-only, but the other entities of that type must not lose seed status
        self.assertEqual(st["demoted_types"], 0)
        self.assertEqual(st["demoted_type_names"], [])

    def test_type_health_demotes_types_made_of_references_and_boilerplate(self) -> None:
        from kb_pipeline.graph.merge import type_health

        ents = [{"key": f"d{i}", "type": "document", "reference": i < 3, "boilerplate": i >= 3} for i in range(5)]
        ents += [{"key": f"t{i}", "type": "timing parameter", "reference": False, "boilerplate": False} for i in range(5)]
        rows = {r["type"]: r for r in type_health(ents, value_counts={"operating condition": 6, "timing parameter": 1})}
        self.assertTrue(rows["document"]["demoted"])
        self.assertFalse(rows["timing parameter"]["demoted"])
        self.assertEqual(rows["operating condition"]["values"], 6)
        self.assertFalse(rows["operating condition"]["demoted"])      # no remaining entities to demote
        # Too few instances: no demotion
        few = type_health([{"key": "x", "type": "t", "reference": True, "boilerplate": False}])
        self.assertFalse(few[0]["demoted"])

    def test_resolution_auto_merges_identifier_variants_and_type_words(self) -> None:
        from kb_pipeline.graph.resolution import (
            auto_merge_pairs, canonical_identifier, embedding_candidates, is_containment, strip_type_words,
        )

        self.assertEqual(canonical_identifier("t_AS"), canonical_identifier("tAS"))
        self.assertEqual(canonical_identifier("t AS"), "tas")
        # Type words are no longer hard-coded: with none passed only organisation suffix words are stripped; words
        # from ontology type names / profile type_words are passed through extra
        self.assertEqual(strip_type_words("tASH parameter"), "tash parameter")
        self.assertEqual(strip_type_words("tASH parameter", ("parameter",)), "tash")
        self.assertEqual(strip_type_words("输出使能信号", ("信号",)), "输出使能")
        self.assertEqual(strip_type_words("信号", ("信号",)), "")                       # the whole name is a type word: nothing left, and callers never merge on an empty string
        self.assertEqual(strip_type_words("parameter", ("parameter",)), "")
        ents = [{"title": "tASH", "type": "timing parameter"}, {"title": "tASH parameter", "type": "timing parameter"},
                {"title": "t_AS", "type": "timing parameter"}, {"title": "tAS", "type": "timing parameter"},
                {"title": "tAS", "type": "pin"}]
        pairs = auto_merge_pairs(ents)
        self.assertIn((0, 1), pairs)
        self.assertIn((2, 3), pairs)
        self.assertFalse(any(4 in p for p in pairs))                  # different types do not merge
        self.assertTrue(is_containment("tASH", "tASH setup time"))
        self.assertFalse(is_containment("SRAM", "Static Random Access Memory"))
        self.assertFalse(is_containment("ZK7C4021KV13", "ZK7C4041KV13 device"))   # the digits differ
        vectors = {"AutoStore": [1.0, 0.0], "自动存储": [0.99, 0.1], "PE#": [0.0, 1.0]}

        def embed(titles):
            return [vectors[t] for t in titles]

        ents2 = [{"title": "AutoStore", "type": "feature"}, {"title": "自动存储", "type": "feature"}, {"title": "PE#", "type": "feature"}]
        self.assertEqual(embedding_candidates(ents2, embed), [(0, 1)])

    def test_resolve_applies_auto_pairs_without_asking_the_model(self) -> None:
        ents = [{"key": "t_as", "title": "t_AS", "type": "timing parameter", "descriptions": [], "unit_ids": ["t1"], "frequency": 1, "aliases": [],
                 "boilerplate": False, "reference": False, "evidence_kind": "body", "attributes": ["min = 0.16 ns"]},
                {"key": "tas", "title": "tAS", "type": "timing parameter", "descriptions": [], "unit_ids": ["t2"], "frequency": 3, "aliases": [],
                 "boilerplate": True, "reference": False, "evidence_kind": "boilerplate", "attributes": []}]
        client = _client([])          # no candidate batches, so the model must not be called
        merged, _, stats = resolution.resolve(client, ents, [])
        self.assertEqual((stats["auto_pairs"], stats["candidates"], stats["batches"], stats["entities_after"]), (1, 0, 0, 1))
        self.assertEqual(merged[0]["title"], "tAS")
        self.assertFalse(merged[0]["boilerplate"])                   # with a body source it is not boilerplate-only
        self.assertEqual(merged[0]["evidence_kind"], "body")
        self.assertEqual(merged[0]["attributes"], ["min = 0.16 ns"])

    def test_mentions_also_match_the_section_path(self) -> None:
        unit = _unit("u1", points=("p1", "p2"))
        entities = [{"key": "tap registers", "title": "TAP Registers", "aliases": [], "unit_ids": ["u1"]}]
        rows = attribute_mentions(entities, {"u1": unit}, {"p1": "During this state, instructions are shifted…", "p2": "Contents … TAP Registers ....22"},
                                  chunk_sections={"p1": "IEEE 1149.1 > TAP Registers", "p2": "Contents"})
        by = {r["point_id"]: r["count"] for r in rows}
        self.assertEqual(by["p1"], 1)          # hit in the section heading
        self.assertEqual(by["p2"], 1)          # hit in the table-of-contents text

    def test_question_identifiers_pick_the_tokens_worth_matching(self) -> None:
        from kb_pipeline.graph.merge import question_identifiers

        self.assertEqual(question_identifiers("ZK7C4021KV13 的输入高电平电压 VIH 在 VCC 为 2.7 V 至 3.6 V 时的最小值是多少？"),
                         ["ZK7C4021KV13", "VIH", "VCC", "2.7", "3.6"])
        self.assertEqual(question_identifiers("IDCODE 指令在 TAP 控制器中执行什么功能？"), ["IDCODE", "TAP"])
        self.assertEqual(question_identifiers("tAS 的最小值"), ["tAS"])

    def test_payloads_bundle_and_seed_filter_carry_the_flags(self) -> None:
        from kb_pipeline.graph.recall import seed_filter
        from kb_pipeline.graph.vectors import entity_payload, relation_payload
        from kb_pipeline.vector.qdrant import GRAPH_PAYLOAD_INDEX_FIELDS

        e = {"key": "a", "title": "A", "type": "t", "boilerplate": True, "reference": False, "evidence_kind": "boilerplate",
             "attributes": ["has_voltage_level = 1.3V"], "unit_ids": ["u1"]}
        pe = entity_payload(e, kb_id="kb", source_collection="kb", graph_version="v1", point_ids=["p1"])
        self.assertEqual((pe["boilerplate"], pe["reference"], pe["evidence_kind"], pe["attributes"]),
                         (True, False, "boilerplate", ["has_voltage_level = 1.3V"]))
        r = {"source_key": "a", "target_key": "b", "predicate": "related_to", "source": "A", "target": "B",
             "reference": True, "evidence_kind": "listing"}
        pr = relation_payload(r, kb_id="kb", source_collection="kb", graph_version="v1", point_ids=[])
        self.assertEqual((pr["boilerplate"], pr["reference"], pr["evidence_kind"]), (False, True, "listing"))
        for field in ("boilerplate", "reference", "evidence_kind"):
            self.assertIn(field, GRAPH_PAYLOAD_INDEX_FIELDS)
        flt = seed_filter()
        self.assertEqual(sorted(c.key for c in flt.must_not), ["boilerplate", "reference"])
        recall_src = Path(__file__).resolve().parents[1].joinpath("kb_pipeline/graph/recall.py").read_text(encoding="utf-8")
        self.assertIn("coalesce(r.boilerplate, false) = false", recall_src)
        self.assertIn("coalesce(c.kind, 'body') <> 'boilerplate'", recall_src)
        self.assertIn("coalesce(tu.kind, 'body') <> 'boilerplate'", recall_src)

    def test_bundle_writes_unit_kind_on_text_units_and_chunks(self) -> None:
        from kb_pipeline.graph.neo4j_import import load_bundle

        source = KBSource(kb_id="kb_003", collection="kb_003", source_root="半导体", source_type="local_mirror",
                          max_tokens=400, overlap_tokens=80)
        with tempfile.TemporaryDirectory() as tmp:
            out, graph, units, entity_id = _bundle_dir(self, tmp)
            graph["unit_kinds"] = {units[0].unit_id: "boilerplate"}
            graph["entities"][0]["boilerplate"] = True
            graph["relations"][0]["evidence_kind"] = "listing"
            (out / "graph.json").write_text(json.dumps(graph), encoding="utf-8")
            bundle = load_bundle(out, source=source, graph_version="v1")
        kinds = {tu["id"]: tu["kind"] for tu in bundle["text_units"]}
        self.assertEqual(kinds[units[0].unit_id], "boilerplate")
        self.assertEqual(kinds[units[1].unit_id], "body")
        chunk_kinds = {c["point_id"]: c["kind"] for c in bundle["qdrant_chunks"]}
        self.assertEqual(chunk_kinds["p1"], "boilerplate")
        self.assertTrue(bundle["entities"][0]["boilerplate"])
        self.assertEqual(bundle["relation_entity_edges"][0]["related_props"]["evidence_kind"], "listing")

    def test_lexical_seeds_find_entities_by_exact_identifier(self) -> None:
        """Dense seeds miss short identifiers like "tAS"; an exact pass over titles / aliases / relation endpoints fills them in."""
        from kb_pipeline.graph.recall import identifier_variants, lexical_seeds, lexical_tokens

        q = "ZK7C4021KV13 的地址建立时间 tAS 在 667 MHz 和 600 MHz 下的最小值分别是多少？"
        self.assertEqual(lexical_tokens(q), ["ZK7C4021KV13", "tAS", "MHz"])   # 667 / 600 are values, not searched
        self.assertIn("TAS", identifier_variants("tAS"))
        self.assertEqual(identifier_variants("t_AS")[:2], ["t_AS", "T_AS"])
        calls: list[tuple[str, Any]] = []
        points = {
            "graph_kb_entity": [SimpleNamespace(id="p1", payload={"gr_id": "e1", "title": "tAS", "aliases": ["t_AS"]}),
                                SimpleNamespace(id="p2", payload={"gr_id": "e2", "title": "ZK7C4021KV13", "aliases": []})],
            "graph_kb_relation": [SimpleNamespace(id="p3", payload={"gr_id": "r1", "source": "ZK7C4021KV13", "target": "tAS", "type": "has_timing_parameter"})],
        }

        class FakeQ:
            def scroll(self, collection_name, scroll_filter, limit, with_payload, with_vectors):
                calls.append((collection_name, scroll_filter))
                wanted = set(scroll_filter.should[0].match.any)
                keys = [c.key for c in scroll_filter.should]
                hits = []
                for p in points[collection_name]:
                    values = set()
                    for key in keys:
                        v = p.payload.get(key)
                        values |= set(v) if isinstance(v, list) else {v}
                    if values & wanted:
                        hits.append(p)
                return hits, None

        ents, rels = lexical_seeds(FakeQ(), "graph_kb_entity", "graph_kb_relation", q)
        self.assertEqual({e["gr_id"]: e["_score"] for e in ents}, {"e1": 0.85, "e2": 0.85})   # exact title hit
        self.assertEqual([(r["gr_id"], r["_score"]) for r in rels], [("r1", 0.8)])
        self.assertEqual(len(calls), 6)                                   # 3 identifiers × (entity + relation)
        flt = calls[0][1]
        self.assertEqual(sorted(c.key for c in flt.should), ["aliases", "title"])
        self.assertEqual(sorted(c.key for c in flt.must_not), ["boilerplate", "reference"])
        self.assertEqual(lexical_seeds(FakeQ(), "a", "b", "这是一个没有记号的问题？"), ([], []))


class OntologyTests(unittest.TestCase):
    """Ontology strengthening (Desktop "Knowledge graph ontology strengthening plan 2026-09-04"): fixed upper
    ontology, document-local scope, qualified facts, derived variant edges, predicate health check, fact seeds in
    recall."""

    def test_upper_ontology_is_fixed_and_legacy_parents_map_onto_it(self) -> None:
        from kb_pipeline.limits import DOCUMENT_SCOPED_PARENTS, UPPER_PARENTS, normalize_upper_parents, upper_parent_of

        self.assertEqual(UPPER_PARENTS, ("entity", "part", "property", "process", "standard", "document"))
        self.assertEqual(upper_parent_of("component"), "entity")
        self.assertEqual(upper_parent_of("parameter"), "property")
        self.assertEqual(upper_parent_of("Interface"), "part")
        self.assertEqual(upper_parent_of("mode"), "process")
        self.assertIsNone(upper_parent_of("something new"))
        self.assertEqual(normalize_upper_parents({"Timing Parameter": "parameter", "pin": "part", "x": "??"}, ["timing parameter", "pin", "x"]),
                         {"timing parameter": "property", "pin": "part"})
        self.assertEqual(DOCUMENT_SCOPED_PARENTS, {"part", "property", "process"})
        # The sampling prompts allow only the six classes
        self.assertIn("{upper_parents}", prompts.PARENT_TYPES_PROMPT)
        self.assertIn("{upper_parents}", prompts.UPPER_MAPPING_PROMPT)
        self.assertEqual(graph_schema.normalize_parent_types({"parent_types": {"pin": "component", "tAS": "parameter"}}, ["pin", "tAS"]),
                         {"pin": "entity", "tAS": "property"})

    def test_scoped_types_get_document_local_keys(self) -> None:
        from kb_pipeline.graph.merge import scoped_key

        u1 = _unit("u1", doc="kb_003:1"); u2 = _unit("u2", doc="kb_003:2", order=1)
        ext = {
            "u1": _ext([("ZK7C4021KV13", "device", "part A"), ("tAS", "timing parameter", "setup A")],
                       [("ZK7C4021KV13", "tAS", "has_timing_parameter", "A has tAS", 7)]),
            "u2": _ext([("ZK7C1049GN", "device", "part B"), ("tAS", "timing parameter", "setup B")],
                       [("ZK7C1049GN", "tAS", "has_timing_parameter", "B has tAS", 7)]),
        }
        out = merge_extractions([u1, u2], ext, entity_types=("device", "timing parameter"),
                                upper_parents={"device": "entity", "timing parameter": "property"})
        ents = {e["key"]: e for e in out["entities"]}
        self.assertEqual(set(ents), {"zk7c4021kv13", "zk7c1049gn", scoped_key("tas", "kb_003:1"), scoped_key("tas", "kb_003:2")})
        self.assertEqual(ents[scoped_key("tas", "kb_003:1")]["scope"], "kb_003:1")
        self.assertEqual(ents[scoped_key("tas", "kb_003:1")]["upper"], "property")
        self.assertEqual(ents["zk7c4021kv13"]["scope"], "")
        rels = {(r["source_key"], r["target_key"]) for r in out["relations"]}
        self.assertIn(("zk7c4021kv13", scoped_key("tas", "kb_003:1")), rels)
        self.assertIn(("zk7c1049gn", scoped_key("tas", "kb_003:2")), rels)
        self.assertEqual(out["stats"]["scoped_entities"], 2)
        # No upper mapping given: old behaviour, global merge
        old = merge_extractions([u1, u2], ext, entity_types=("device", "timing parameter"))
        self.assertIn("tas", {e["key"] for e in old["entities"]})
        # Resolution does not cross scopes: the two tAS share name and type but differ in scope, so no auto-merge
        pairs = resolution.auto_merge_pairs(out["entities"])
        self.assertEqual(pairs, [])
        self.assertEqual(resolution.candidate_pairs(out["entities"]), [])

    def test_two_letter_pin_names_count_as_identifiers(self) -> None:
        from kb_pipeline.graph.merge import question_identifiers

        self.assertEqual(question_identifiers("ZK7C4021KV13 的 AP 引脚在 ×18 数据宽度下覆盖哪些地址引脚"), ["ZK7C4021KV13", "AP", "18"])
        self.assertEqual(question_identifiers("ap 引脚 pin"), [])      # lowercase ordinary words do not count

    def test_variant_edges_are_derived_from_names(self) -> None:
        from kb_pipeline.graph.merge import derive_variant_edges

        ents = [{"key": "zk7c4021kv13", "title": "ZK7C4021KV13", "type": "device", "scope": "", "unit_ids": ["u1"]},
                {"key": "zk7c4021kv13-667fcxc", "title": "ZK7C4021KV13-667FCXC", "type": "device", "scope": "", "unit_ids": ["u2"]},
                {"key": "iphone 15", "title": "iPhone 15", "type": "device", "scope": "", "unit_ids": []},
                {"key": "iphone 15 pro", "title": "iPhone 15 Pro", "type": "device", "scope": "", "unit_ids": []},
                {"key": "kb_003:1::tas", "title": "tAS", "type": "timing parameter", "scope": "kb_003:1", "unit_ids": []},
                {"key": "kb_003:1::tas-x", "title": "tAS-x", "type": "timing parameter", "scope": "kb_003:1", "unit_ids": []},
                {"key": "ab", "title": "AB", "type": "device", "scope": "", "unit_ids": []},
                {"key": "ab-1", "title": "AB-1", "type": "device", "scope": "", "unit_ids": []}]
        derived = derive_variant_edges(ents, [{"source_key": "iphone 15 pro", "target_key": "iphone 15"}])
        pairs = {(d["source_key"], d["target_key"]) for d in derived}
        self.assertEqual(pairs, {("zk7c4021kv13-667fcxc", "zk7c4021kv13")})   # iPhone already has a relation; document-local entities and too-short base names are not derived
        self.assertTrue(derived[0]["derived"])
        self.assertEqual(derived[0]["predicate"], "variant_of")

    def test_predicate_health_reports_violations_and_fanout(self) -> None:
        from kb_pipeline.graph.merge import predicate_health

        ents = [{"key": "a", "parent_type": "component"}, {"key": "b", "parent_type": "document"}, {"key": "c", "parent_type": "component"}]
        rels = [{"source_key": "a", "target_key": "b", "predicate": "has_pin", "type_violation": True},
                {"source_key": "a", "target_key": "c", "predicate": "has_pin", "type_violation": False},
                {"source_key": "c", "target_key": "a", "predicate": "variant_of", "derived": True}]
        rows = {r["predicate"]: r for r in predicate_health(rels, ents)}
        self.assertEqual((rows["has_pin"]["edges"], rows["has_pin"]["violations"], rows["has_pin"]["violation_ratio"]), (2, 1, 0.5))
        self.assertEqual(rows["has_pin"]["top_violating_pairs"], [{"source_parent": "component", "target_parent": "document", "count": 1}])
        self.assertEqual(rows["has_pin"]["fanout_p95"], 2)
        self.assertEqual(rows["variant_of"]["derived"], 1)

    def test_fact_parsing_and_normalisation(self) -> None:
        from kb_pipeline.graph.facts import fact_id, normalize_fact, parse_facts_json, parse_number, spec_text, wants_facts

        raw = '''```json
{"facts": [
  {"subject": "ZK7C4021KV13", "property": "Address setup to CK", "symbol": "tAS", "min": "0.160", "unit": "ns", "conditions": {"speed grade": "667 MHz"}},
  {"subject": "ZK7C4021KV13", "property": "Address setup to CK", "symbol": "tAS", "min": "0.180", "unit": "ns", "conditions": {"speed grade": "600 MHz"}},
  {"subject": "", "property": "x", "value": "1"},
  {"subject": "ZK7C4021KV13", "property": "no value here"},
]}
```'''
        facts, malformed = parse_facts_json(raw)
        self.assertEqual((len(facts), malformed), (4, 0))
        norm = [normalize_fact(f) for f in facts]
        self.assertEqual([n is not None for n in norm], [True, True, False, False])
        self.assertEqual(norm[0]["min_num"], 0.16)
        self.assertEqual(norm[0]["conditions"], {"speed grade": "667 MHz"})
        self.assertNotEqual(fact_id("u1", norm[0]), fact_id("u1", norm[1]))     # different conditions make two facts
        self.assertEqual(spec_text(norm[0]), "ZK7C4021KV13 · Address setup to CK (tAS): min 0.160 ns | speed grade: 667 MHz")
        self.assertEqual(parse_facts_json("no json here"), ([], 1))
        self.assertEqual(parse_number("1,024"), 1024.0)
        self.assertEqual(parse_number("-40°C"), -40.0)
        self.assertIsNone(parse_number("n/a"))
        table_unit = _unit("t1")
        table_unit.block_types = ["table"]
        self.assertTrue(wants_facts(table_unit))
        self.assertFalse(wants_facts(table_unit, "boilerplate"))
        self.assertFalse(wants_facts(_unit("p1")))

    def test_fact_extractor_links_subjects_and_builds_spec_payloads(self) -> None:
        from kb_pipeline.graph.facts import FactExtractor, document_subjects, link_facts
        from kb_pipeline.graph.vectors import spec_payload

        unit = _unit("u1", doc="kb_003:1", points=("p1", "p2"))
        unit.block_types = ["table"]
        client = _client(['{"facts": [{"subject": "ZK7C4021KV13", "property": "Address setup", "symbol": "tAS", "min": "0.160", "unit": "ns", "conditions": {"speed grade": "667 MHz"}}]}'])
        res = FactExtractor(client).extract(unit, document="a.pdf", subjects=["ZK7C4021KV13"])
        self.assertEqual((len(res.facts), res.calls, res.stats["facts"]), (1, 1, 1))
        ents = [{"key": "zk7c4021kv13", "title": "ZK7C4021KV13", "aliases": [], "scope": "", "upper": "entity", "doc_ids": ["kb_003:1"], "frequency": 9},
                {"key": "kb_003:1::tas", "title": "tAS", "aliases": ["t_AS"], "scope": "kb_003:1", "upper": "property", "doc_ids": ["kb_003:1"], "frequency": 2}]
        stats = link_facts(res.facts, ents, units_by_id={"u1": unit})
        self.assertEqual(stats, {"subjects_linked": 1, "properties_linked": 1, "compound_subjects": 0})
        # Compound subject "A, B": both are linked, the primary key takes the first
        both = [{"subject": "ZK7C4021KV13, ZK7C4041KV13", "property": "x", "value": "1", "unit_id": "u1"}]
        ents2 = ents + [{"key": "zk7c4041kv13", "title": "ZK7C4041KV13", "aliases": [], "scope": "", "upper": "entity", "doc_ids": ["kb_003:1"], "frequency": 3}]
        st2 = link_facts(both, ents2, units_by_id={"u1": unit})
        self.assertEqual((both[0]["subject_key"], both[0]["subject_keys"], st2["compound_subjects"]), ("zk7c4021kv13", ["zk7c4021kv13", "zk7c4041kv13"], 1))
        fact = res.facts[0]
        self.assertEqual((fact["subject_key"], fact["property_key"], fact["point_ids"]), ("zk7c4021kv13", "kb_003:1::tas", ["p1", "p2"]))
        self.assertEqual(document_subjects(ents), {"kb_003:1": ["ZK7C4021KV13"]})   # document-local entities are not subjects
        payload = spec_payload(fact, kb_id="kb_003", source_collection="kb_003", graph_version="v1")
        self.assertEqual(payload["graph_type"], "spec")
        self.assertEqual(payload["symbol"], "tAS")
        self.assertEqual(payload["conditions"], {"speed grade": "667 MHz"})
        self.assertIn("0.160", payload["text"])
        self.assertEqual(payload["point_ids"], ["p1", "p2"])

    def test_bundle_carries_specs_and_recall_seeds_from_the_spec_collection(self) -> None:
        from kb_pipeline.graph.neo4j_import import load_bundle
        from kb_pipeline.graph.recall import spec_seeds
        from kb_pipeline.vector.qdrant import GRAPH_COLLECTION_RE, GRAPH_VECTOR_TYPES

        self.assertIn("spec", GRAPH_VECTOR_TYPES)
        self.assertTrue(GRAPH_COLLECTION_RE.match("graph_003_spec__v1"))
        source = KBSource(kb_id="kb_003", collection="kb_003", source_root="半导体", source_type="local_mirror",
                          max_tokens=400, overlap_tokens=80)
        with tempfile.TemporaryDirectory() as tmp:
            out, graph, units, entity_id = _bundle_dir(self, tmp)
            graph["specs"] = [{"id": "f1", "subject": "A, B", "subject_key": "a", "subject_keys": ["a", "b"], "property": "p", "symbol": "", "value": "1", "unit": "V",
                               "conditions": {"mode": "x"}, "unit_id": units[0].unit_id, "doc_id": units[0].doc_id}]
            (out / "graph.json").write_text(json.dumps(graph), encoding="utf-8")
            bundle = load_bundle(out, source=source, graph_version="v1")
        self.assertEqual(bundle["expected_counts"]["specs"], 1)
        from kb_pipeline.graph.neo4j_import import bundle_type_counts
        self.assertEqual(bundle_type_counts({"entities": [1, 2], "relations": [1], "specs": [1, 2, 3]}), {"entity": 2, "relation": 1, "spec": 3, "page": 0})
        self.assertEqual(set(bundle_type_counts({})), set(GRAPH_VECTOR_TYPES))    # preflight compares counts per type, so a new type must have a key
        self.assertEqual((bundle["expected_counts"]["has_spec_edges"], bundle["expected_counts"]["states_edges"], bundle["expected_counts"]["of_property_edges"]), (2, 1, 0))
        self.assertEqual(bundle["specs"][0]["conditions"], '{"mode": "x"}')

        class FakeQ:
            def query_points(self, collection_name, query, limit, with_payload, query_filter=None):
                return SimpleNamespace(points=[SimpleNamespace(id="s1", score=0.7, payload={"gr_id": "f1", "subject": "ZK7C4021KV13", "symbol": "tCK", "property": "cycle", "point_ids": ["p9"]})])

            def scroll(self, collection_name, scroll_filter, limit, with_payload, with_vectors):
                return [SimpleNamespace(id="s2", payload={"gr_id": "f2", "subject": "ZK7C4021KV13", "symbol": "tAS", "property": "setup", "point_ids": ["p1"]})], None

        seeds = spec_seeds(FakeQ(), "graph_003_spec", [0.1], "ZK7C4021KV13 的 tAS 最小值")
        by = {s["gr_id"]: s for s in seeds}
        self.assertEqual(by["f2"]["_via"], "lexical")
        self.assertGreater(by["f2"]["_score"], by["f1"]["_score"])      # a hit on both subject and symbol ranks first
        from kb_pipeline.graph.recall import SPEC_EVIDENCE_LIMIT, spec_evidence_seeds
        weak = [{"gr_id": f"w{i}", "_score": 0.5} for i in range(6)]
        strong = [{"gr_id": "s1", "_score": 0.95}, {"gr_id": "s2", "_score": 0.86}]
        self.assertEqual([x["gr_id"] for x in spec_evidence_seeds(strong + weak)], ["s1", "s2"])   # weakly related facts do not affect chunk ranking
        self.assertEqual(len(spec_evidence_seeds([{"gr_id": str(i), "_score": 0.9} for i in range(10)])), SPEC_EVIDENCE_LIMIT)
        self.assertEqual(spec_seeds(SimpleNamespace(query_points=lambda **kw: (_ for _ in ()).throw(RuntimeError("no collection"))), "x", [0.1], "q"), [])


class DeterministicBuilderTests(unittest.TestCase):
    """Code-repository knowledge base: rule-based extraction routed by file type (code AST / structured markdown /
    config), producing records of the same shape as model extraction; cross-file references carry scope_doc, and
    exact identities are never merged."""

    REPO = {
        "wiki/lib/util.py": "def helper(x):\n    \"\"\"Help.\"\"\"\n    return x\n\n\nclass Base:\n    def ping(self):\n        return 1\n",
        "wiki/cli.py": "import os\nimport requests\nfrom lib.util import helper, Base\n\nLIMIT = 3\n\n\ndef main():\n    \"\"\"Entry.\"\"\"\n    return helper(LIMIT) + Engine().run()\n\n\nclass Engine(Base):\n    def run(self):\n        return self.ping() + os.getpid()\n\n\nif __name__ == \"__main__\":\n    main()\n",
        "wiki/SKILL.md": "---\nname: wiki-skill\ndescription: 编译 wiki。\nversion: 1.2\ntags: a,b\n---\n\n# 用法\n\n运行 `python cli.py build`,配置见 skillhub.json。\n",
        "wiki/skillhub.json": "{\"name\": \"wiki-skill\", \"ns\": \"wps\", \"limits\": {\"max\": 5}}",
        "wiki/requirements.txt": "requests>=2.28\n# comment\npyyaml>=6.0\n",
        "wiki/design.md": "# 设计\n\n普通说明文档。\n",
    }

    def _units(self, root: Path):
        from kb_pipeline.graph.units import ChunkRef, build_units
        from kb_pipeline.parsers.router import parse_native

        refs = []
        doc_ids = {}
        for i, (rel, text) in enumerate(self.REPO.items(), 1):
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            doc_ids[rel] = f"kb_006:{i}"
            blocks = parse_native(path, "")
            refs += [ChunkRef(point_id=f"p{i}-{k}", chunk_uid=f"c{i}-{k}", doc_id=doc_ids[rel], content_version="v", chunk_index=k,
                              block_id=b.block_id, block_type=b.block_type, section_path=list(b.metadata.get("section_path") or []),
                              text=b.text, n_tokens=max(1, len(b.text) // 4), rel_path=rel, filename=rel.rsplit("/", 1)[-1])
                     for k, b in enumerate(blocks)]
        # Units are built for the whole KB at once: order is KB-wide, so a config file's unit order is not 0 (as in the
        # graph build)
        return build_units(refs, kb_id="kb_006", unit_chunks=3), doc_ids

    def test_routes_records_and_cross_file_scope(self) -> None:
        from kb_pipeline.graph.deterministic import DeterministicExtractor, route_for

        self.assertEqual([route_for(p) for p in ("a/x.py", "a/skillhub.json", "a/requirements.txt", "a/SKILL.md", "a/notes.md", "a/x.pdf")],
                         ["code", "config", "config", "structured_md", "llm", "llm"])
        self.assertEqual(route_for("a/notes.md", first_line="---"), "structured_md")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            units, doc_ids = self._units(root)
            det = DeterministicExtractor(units, file_for=lambda rel: root / rel, kb_id="kb_006")
            by_rel = {}
            for u in units:
                by_rel.setdefault(u.rel_path, []).append(u)
            self.assertFalse(det.wants(by_rel["wiki/design.md"][0]))          # ordinary md still goes to the model
            # Code: the units of cli.py
            ents, rels = [], []
            for u in by_rel["wiki/cli.py"]:
                res = det.extract(u)
                self.assertEqual(res.calls, 0)
                ents += res.entities; rels += res.relations
            names = {(e["name"], e["type"]) for e in ents}
            self.assertIn(("wiki/cli.py", "module"), names)
            self.assertIn(("main", "function"), names)
            self.assertIn(("Engine", "class"), names)
            self.assertIn(("Engine.run", "method"), names)
            self.assertIn(("LIMIT", "constant"), names)
            self.assertIn(("requests", "package"), names)
            self.assertNotIn(("os", "package"), names)                          # the standard library does not become an entity
            helper = next(e for e in ents if e["name"] == "helper")
            self.assertEqual(helper["scope_doc"], doc_ids["wiki/lib/util.py"])   # a cross-file reference points to the file that defines it
            self.assertTrue(all(e.get("exact") for e in ents))
            triples = {(r["source"], r["predicate"], r["target"]) for r in rels}
            for t in [("wiki/cli.py", "imports", "wiki/lib/util.py"), ("wiki/cli.py", "imports", "requests"),
                      ("wiki/cli.py", "defines", "main"), ("wiki/cli.py", "defines", "LIMIT"),
                      ("main", "calls", "helper"), ("main", "calls", "Engine"), ("Engine", "has_method", "Engine.run"),
                      ("Engine.run", "calls", "Base.ping"), ("Engine", "inherits", "Base")]:
                self.assertIn(t, triples, t)
            ping = next(e for e in ents if e["name"] == "Base.ping")
            self.assertEqual(ping["scope_doc"], doc_ids["wiki/lib/util.py"])     # an inherited method points to the file that defines it
            self.assertEqual(det.route_counts["code"], len(by_rel["wiki/cli.py"]))
            # Structured markdown: entities + frontmatter facts + reference edges
            md_ents, md_rels, md_facts = [], [], []
            for u in by_rel["wiki/SKILL.md"]:
                res = det.extract(u); md_ents += res.entities; md_rels += res.relations; md_facts += det.facts(u)
            skill = next(e for e in md_ents if e["type"] == "skill")
            self.assertEqual((skill["name"], skill["descriptions"]), ("wiki-skill", ["编译 wiki。"]))
            self.assertEqual({(r["source"], r["predicate"], r["target"]) for r in md_rels},
                             {("wiki-skill", "references", "wiki/cli.py"), ("wiki-skill", "references", "wiki/skillhub.json")})
            self.assertEqual({(f["property"], f["value"]) for f in md_facts}, {("version", "1.2"), ("tags", "a,b")})
            # Config: key-path facts; requirements: package entities and depends_on
            cfg = by_rel["wiki/skillhub.json"][0]
            self.assertEqual({(f["property"], f["value"]) for f in det.facts(cfg)}, {("name", "wiki-skill"), ("ns", "wps"), ("limits.max", "5")})
            req = by_rel["wiki/requirements.txt"][0]
            res = det.extract(req)
            self.assertEqual({(r["source"], r["predicate"], r["target"]) for r in res.relations}, {("wiki", "depends_on", "requests"), ("wiki", "depends_on", "pyyaml")})
            self.assertEqual({(f["symbol"], f["value"]) for f in det.facts(req)}, {("requests", ">=2.28"), ("pyyaml", ">=6.0")})
            self.assertTrue(all(u.kind == "body" for u in by_rel["wiki/cli.py"]))   # code units are not down-weighted as listings

    def test_merge_honours_scope_doc_exact_and_aliases(self) -> None:
        from kb_pipeline.graph.deterministic import TYPES, UPPER
        from kb_pipeline.graph.merge import scoped_key

        u1 = _unit("u1", doc="kb_006:1"); u2 = _unit("u2", doc="kb_006:2", order=1)
        ext = {
            "u1": {"entities": [{"name": "helper", "type": "function", "descriptions": ["def helper(x)"], "mentions": 1, "types": {"function": 1}, "exact": True}],
                   "relations": []},
            "u2": {"entities": [{"name": "main", "type": "function", "descriptions": ["def main()"], "mentions": 1, "types": {"function": 1}, "exact": True, "aliases": ["entry"]},
                               {"name": "helper", "type": "function", "descriptions": [], "mentions": 1, "types": {"function": 1}, "exact": True, "scope_doc": "kb_006:1"}],
                   "relations": [{"source": "main", "target": "helper", "predicate": "calls", "predicate_raw": "calls", "descriptions": ["main calls helper"], "strength": 1}]},
        }
        out = merge_extractions([u1, u2], ext, entity_types=TYPES, parent_types=UPPER, upper_parents=UPPER)
        ents = {e["key"]: e for e in out["entities"]}
        self.assertEqual(set(ents), {scoped_key("helper", "kb_006:1"), scoped_key("main", "kb_006:2")})   # no kb_006:2::helper
        helper = ents[scoped_key("helper", "kb_006:1")]
        self.assertEqual((helper["scope"], helper["exact"], sorted(helper["unit_ids"])), ("kb_006:1", True, ["u1", "u2"]))
        self.assertIn("entry", ents[scoped_key("main", "kb_006:2")]["aliases"])
        self.assertEqual([(r["source_key"], r["target_key"], r["predicate"]) for r in out["relations"]],
                         [(scoped_key("main", "kb_006:2"), scoped_key("helper", "kb_006:1"), "calls")])
        # exact entities never enter the resolution candidates
        rows = [{"key": "a", "title": "Engine", "type": "class", "scope": "d", "exact": True},
                {"key": "b", "title": "engine", "type": "class", "scope": "d", "exact": True}]
        self.assertEqual(resolution.auto_merge_pairs(rows), [])
        self.assertEqual(resolution.candidate_pairs(rows), [])

    def test_code_names_count_as_identifiers_and_modules_have_path_aliases(self) -> None:
        from kb_pipeline.graph.deterministic import module_aliases
        from kb_pipeline.graph.merge import question_identifiers

        self.assertEqual(question_identifiers("team_wiki_search 函数定义在哪个文件里"), ["team_wiki_search"])
        self.assertEqual(question_identifiers("lib/http_client.py 导入了 lib.v7_env 吗"), ["lib/http_client.py", "lib.v7_env"])
        self.assertEqual(question_identifiers("这个函数返回什么"), [])
        self.assertEqual(module_aliases("示例365-wiki/lib/http_client.py"), ["lib.http_client", "lib/http_client.py", "http_client.py"])
        self.assertEqual(module_aliases("示例365-wiki/lib/mcts/__init__.py"), ["lib.mcts", "lib/mcts/__init__.py", "__init__.py"])

    def test_javascript_repo_cross_file_imports(self) -> None:
        try:
            import tree_sitter_language_pack  # noqa: F401
        except ImportError:
            self.skipTest("tree-sitter-language-pack not installed")
        from kb_pipeline.graph.deterministic import DeterministicExtractor, module_aliases, route_for
        from kb_pipeline.graph.units import ChunkRef, build_units
        from kb_pipeline.parsers.router import parse_native

        self.assertEqual([route_for(p) for p in ("a/x.ts", "a/y.go", "a/z.rs", "a/w.sh", "a/s.css", "a/q.sql", "a/n.ini")],
                         ["code", "code", "code", "code", "code_plain", "code_plain", "config"])
        self.assertEqual(module_aliases("app/lib/util.js"), ["lib.util", "lib/util.js", "util.js"])
        repo = {
            "app/lib/util.js": "export function helper(x) { return x; }\nexport class Base {\n  ping() { return 1; }\n}\n",
            "app/index.js": "import express from \"express\";\nimport fs from \"fs\";\nimport { helper, Base } from \"./lib/util.js\";\nexport const LIMIT = 3;\nexport function main() { return helper(LIMIT) + new Engine().run(); }\nclass Engine extends Base {\n  run() { return this.ping() + fs.readFileSync(\"x\"); }\n}\nmain();\n",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            refs, doc_ids = [], {}
            for i, (rel, text) in enumerate(repo.items(), 1):
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
                doc_ids[rel] = f"kb_007:{i}"
                blocks = parse_native(path, "")
                self.assertEqual(blocks[0].parser_profile, "code-symbols-v1")
                refs += [ChunkRef(point_id=f"p{i}-{k}", chunk_uid=f"c{i}-{k}", doc_id=doc_ids[rel], content_version="v", chunk_index=k,
                                  block_id=b.block_id, block_type=b.block_type, section_path=list(b.metadata.get("section_path") or []),
                                  text=b.text, n_tokens=max(1, len(b.text) // 4), rel_path=rel, filename=rel.rsplit("/", 1)[-1])
                         for k, b in enumerate(blocks)]
            units = build_units(refs, kb_id="kb_007", unit_chunks=3)
            det = DeterministicExtractor(units, file_for=lambda rel: root / rel, kb_id="kb_007")
            ents, rels = [], []
            for u in units:
                if u.rel_path == "app/index.js":
                    res = det.extract(u); ents += res.entities; rels += res.relations
            names = {(e["name"], e["type"]) for e in ents}
            for n in [("app/index.js", "module"), ("main", "function"), ("Engine", "class"), ("Engine.run", "method"), ("LIMIT", "constant"), ("express", "package")]:
                self.assertIn(n, names, n)
            self.assertNotIn(("fs", "package"), names)                      # node built-in modules do not count as packages
            helper = next(e for e in ents if e["name"] == "helper")
            self.assertEqual(helper["scope_doc"], doc_ids["app/lib/util.js"])
            triples = {(r["source"], r["predicate"], r["target"]) for r in rels}
            for t in [("app/index.js", "imports", "app/lib/util.js"), ("app/index.js", "imports", "express"), ("main", "calls", "helper"),
                      ("main", "calls", "Engine"), ("Engine", "inherits", "Base"), ("Engine.run", "calls", "Base.ping"), ("app/index.js", "calls", "main")]:
                self.assertIn(t, triples, t)

    def test_python_submodule_imports_resolve_call_edges(self) -> None:
        """Imports of submodules (`from .. import db`, `from .graph import build`) must resolve to module aliases so
        that `db.claim()` and `build.build_graph()` get calls edges (the repository's own worker -> claim edge was
        lost this way); class-method calls on imported and local classes (Store.open(), Local.make()) get edges too."""
        from kb_pipeline.graph.deterministic import DeterministicExtractor
        from kb_pipeline.graph.units import ChunkRef, build_units
        from kb_pipeline.parsers.router import parse_native

        repo = {
            "app/pkg/__init__.py": '"""pkg"""\n',
            "app/pkg/db.py": "def claim(con):\n    return 1\n\n\nclass Store:\n    @classmethod\n    def open(cls):\n        return cls()\n",
            "app/pkg/graph/__init__.py": '"""graph"""\n',
            "app/pkg/graph/build.py": "def build_graph():\n    return 2\n",
            "app/pkg/pipeline/__init__.py": "",
            "app/pkg/pipeline/worker.py": ("from .. import db\nfrom ..graph import build\nfrom ..db import Store\n\n\n"
                                           "class Local:\n    @classmethod\n    def make(cls):\n        return cls()\n\n\n"
                                           "def run(con):\n    job = db.claim(con)\n    build.build_graph()\n    Store.open()\n    Local.make()\n    return job\n"),
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            refs = []
            for i, (rel, text) in enumerate(repo.items(), 1):
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
                refs += [ChunkRef(point_id=f"p{i}-{k}", chunk_uid=f"c{i}-{k}", doc_id=f"kb_x:{i}", content_version="v", chunk_index=k,
                                  block_id=b.block_id, block_type=b.block_type, section_path=list(b.metadata.get("section_path") or []),
                                  text=b.text, n_tokens=max(1, len(b.text) // 4), rel_path=rel, filename=rel.rsplit("/", 1)[-1])
                         for k, b in enumerate(parse_native(path, ""))]
            units = build_units(refs, kb_id="kb_x", unit_chunks=3)
            det = DeterministicExtractor(units, file_for=lambda rel: root / rel, kb_id="kb_x")
            imports = det._import_map("app/pkg/pipeline/worker.py")
            self.assertEqual(imports["db"], ("app/pkg/db.py", None, None))                 # submodule -> module alias
            self.assertEqual(imports["build"], ("app/pkg/graph/build.py", None, None))     # no longer graph/__init__.py
            self.assertEqual(imports["Store"], ("app/pkg/db.py", "Store", None))           # a real symbol import is unchanged
            rels = []
            for u in units:
                if u.rel_path == "app/pkg/pipeline/worker.py":
                    rels += det.extract(u).relations
            triples = {(r["source"], r["predicate"], r["target"]) for r in rels}
            for t in [("run", "calls", "claim"), ("run", "calls", "build_graph"), ("run", "calls", "Store.open"), ("run", "calls", "Local.make"),
                      ("app/pkg/pipeline/worker.py", "imports", "app/pkg/db.py"), ("app/pkg/pipeline/worker.py", "imports", "app/pkg/graph/build.py")]:
                self.assertIn(t, triples, t)
            self.assertNotIn(("app/pkg/pipeline/worker.py", "imports", "app/pkg/graph/__init__.py"), triples)

    def test_external_imports_never_land_on_a_same_named_file(self) -> None:
        """2026-09-29 audit: the submodule fallback resolved ``from qdrant_client.http import models`` to a models.py
        in the base and dropped the edge to the external package. Submodules are only looked for in the package
        directory the import names; when the whole path does not match it is an external package."""
        from kb_pipeline.graph.deterministic import DeterministicExtractor
        from kb_pipeline.graph.units import ChunkRef, build_units
        from kb_pipeline.parsers.router import parse_native

        repo = {
            "app/pkg/__init__.py": "",
            "app/pkg/models.py": "class Row:\n    pass\n",
            "app/pkg/utils.py": "def helper():\n    return 1\n",
            "app/pkg/vector/__init__.py": "",
            "app/pkg/vector/store.py": ("from qdrant_client.http import models\nfrom django.db import models as dj\n"
                                        "from requests import utils\nfrom pkg import models as own\nfrom pkg.vector import missing\n"
                                        "from tools.extra import helper_mod\n\n\n"
                                        "def point():\n    return models.PointStruct(id=1)\n"),
            "tools/extra/helper_mod.py": "def run():\n    return 1\n",          # a package without __init__.py: the whole path has to match
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            refs = []
            for i, (rel, text) in enumerate(repo.items(), 1):
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
                refs += [ChunkRef(point_id=f"p{i}-{k}", chunk_uid=f"c{i}-{k}", doc_id=f"kb_x:{i}", content_version="v", chunk_index=k,
                                  block_id=b.block_id, block_type=b.block_type, section_path=list(b.metadata.get("section_path") or []),
                                  text=b.text, n_tokens=max(1, len(b.text) // 4), rel_path=rel, filename=rel.rsplit("/", 1)[-1])
                         for k, b in enumerate(parse_native(path, ""))]
            units = build_units(refs, kb_id="kb_x", unit_chunks=3)
            det = DeterministicExtractor(units, file_for=lambda rel: root / rel, kb_id="kb_x")
            imports = det._import_map("app/pkg/vector/store.py")
            self.assertEqual(imports["models"], (None, "models", "qdrant_client"))         # an external package, not the base's models.py
            self.assertEqual(imports["dj"], (None, "models", "django"))
            self.assertEqual(imports["utils"], (None, "utils", "requests"))
            self.assertEqual(imports["own"], ("app/pkg/models.py", None, None))            # a package of the base still resolves to its submodule
            self.assertEqual(imports["helper_mod"], ("tools/extra/helper_mod.py", None, None))
            self.assertNotEqual(imports.get("missing", (None,))[0], "app/pkg/models.py")
            rels = []
            for u in units:
                if u.rel_path == "app/pkg/vector/store.py":
                    rels += det.extract(u).relations
            triples = {(r["source"], r["predicate"], r["target"]) for r in rels}
            self.assertIn(("app/pkg/vector/store.py", "imports", "qdrant_client"), triples)
            self.assertIn(("app/pkg/vector/store.py", "imports", "app/pkg/models.py"), triples)   # from `from pkg import models`

    def test_extraction_fingerprint_only_changes_with_deterministic_units(self) -> None:
        from kb_pipeline.graph.build import extraction_fingerprint

        self.assertEqual(extraction_fingerprint("abc", deterministic_units=0), "abc")
        mixed = extraction_fingerprint("abc", deterministic_units=3)
        self.assertNotEqual(mixed, "abc")
        self.assertEqual(mixed, extraction_fingerprint("abc", deterministic_units=1))


class ConceptNeighbourScaleTests(unittest.TestCase):
    def test_numpy_neighbour_search_matches_the_pairwise_definition_and_scales(self) -> None:
        """2026-09-12, product documentation KB with 12,933 concept candidates: pairwise cosine meant 167 million
        computations, about 3 hours, and the graph build stalled at "property concept normalisation". The numpy
        path must give the same result in the same order as the pairwise implementation, and a few thousand
        candidates must finish within seconds."""
        import random
        import time
        from kb_pipeline.graph.concepts import _neighbour_pairs, _neighbour_pairs_python, _unit_family

        rnd = random.Random(7)
        n, d = 400, 32
        cands = [{"label": f"c{i}", "unit": rnd.choice(["", "", "", "%", "个", "CNY"]), "symbols_text": ""} for i in range(n)]
        vectors = [[rnd.gauss(0, 1) for _ in range(d)] for _ in range(n)]
        vectors[5] = [0.0] * d                                    # zero vector: norm counted as 1, no division by zero
        fast = _neighbour_pairs(vectors, cands, 3)
        slow = _neighbour_pairs_python(vectors, cands, [_unit_family(c["unit"]) for c in cands], 3)
        self.assertEqual([(i, j) for i, j, _ in fast], [(i, j) for i, j, _ in slow])
        for (_, _, a), (_, _, b) in zip(fast, slow):
            self.assertAlmostEqual(a, b, places=9)
        # Comparable pool by unit family: candidates with a unit only compare with the same family and unit-less ones
        fams = [_unit_family(c["unit"]) for c in cands]
        for i, j, _ in fast:
            self.assertTrue(fams[i] == "" or fams[j] == "" or fams[i] == fams[j])
        big = [[rnd.gauss(0, 1) for _ in range(64)] for _ in range(3000)]
        bcands = [{"label": f"b{i}", "unit": "", "symbols_text": ""} for i in range(3000)]
        t0 = time.time()
        pairs = _neighbour_pairs(big, bcands, 3)
        self.assertLess(time.time() - t0, 20)                     # pure-Python pairwise takes minutes here
        self.assertEqual(len(pairs), 3000 * 3)
        # Ragged vector lengths (common with test stubs): fall back to the pairwise implementation without an error
        ragged = [[1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0]]
        rc = [{"label": f"r{i}", "unit": "", "symbols_text": ""} for i in range(3)]
        self.assertEqual([(i, j) for i, j, _ in _neighbour_pairs(ragged, rc, 1)], [(0, 1), (1, 0), (2, 1)])


class PromptAndUnitClassificationTests(unittest.TestCase):
    """First batch of the 2026-09-07 plan: tightened prompts (type menu, record caps, Document/Section as context
    only, corpus examples), unit classification gains a conclusion kind plus Chinese report layouts, summary
    thresholds and source attribution, axis / bound fields on facts, scenario profile and the config pipeline."""

    def test_temporal_dates_versions_axis_and_question_window(self) -> None:
        from datetime import date

        from kb_pipeline.graph.temporal import (
            axis_sort_key, document_axis, find_dates, find_versions, in_window, parse_date, question_time_window,
        )

        self.assertEqual(parse_date("现将您2021年03月14日的体检结果"), "2021-03-14")
        self.assertEqual(parse_date("Revised August 24, 2020"), "2020-08-24")
        self.assertEqual(parse_date("2023/09/12 14:27"), "2023-09-12")
        self.assertEqual(parse_date("报告时间 20230912"), "2023-09-12")
        self.assertEqual(parse_date("2022 年 06 月"), "2022-06")
        self.assertEqual(parse_date("2021体检报告.pdf"), "2021")
        self.assertIsNone(parse_date("01/02/03 nothing here"))
        self.assertEqual(find_dates("2021年3月14日 与 2022-06-05,以及 2026"), ["2021-03-14", "2022-06-05", "2026"])
        self.assertEqual(find_versions("Document Number: 001-79553 Rev. *L  Version 2.1 版本 3"), ["rev *L", "v2.1", "v3"])
        health = document_axis("李华/体检报告/2021体检报告.pdf", "尊敬的先生,您好!现将您2021年03月14日的体检结果报告如下")
        self.assertEqual((health["kind"], health["value"], health["date"]), ("date", "2021-03-14", "2021-03-14"))
        sheet = document_axis("northwind-zk7c4021kv13-datasheet-en.pdf", "ZK7C4021KV13 72-Mbit QDR-IV SRAM Document Number: 001-79553 Rev. *L Revised August 24, 2020", kind="version")
        self.assertEqual((sheet["kind"], sheet["value"], sheet["date"]), ("version", "rev *L", "2020-08-24"))
        self.assertEqual(document_axis("notes.md", "nothing dated")["kind"], "none")
        self.assertEqual(document_axis("2022-report.pdf", "2022-06-05 发布", kind="date")["value"], "2022-06-05")   # the filename only has the year; the first page completes it
        self.assertLess(axis_sort_key("2021-03"), axis_sort_key("2022-06-05"))
        self.assertLess(axis_sort_key("rev *J"), axis_sort_key("rev *L"))
        self.assertLess(axis_sort_key("v1.2"), axis_sort_key("v1.10"))
        self.assertLess(axis_sort_key("2024"), axis_sort_key(""))
        today = date(2026, 9, 7)
        self.assertEqual(question_time_window("这两年我的身体状况如何", today=today), {"from": "2025", "to": "2026", "text": "这两年"})
        self.assertEqual(question_time_window("最近三年的血脂变化", today=today)["from"], "2024")
        self.assertEqual(question_time_window("2024到2026年体检有什么变化", today=today)["to"], "2026")
        self.assertEqual(question_time_window("去年的报告", today=today), {"from": "2025", "to": "2025", "text": "去年"})
        self.assertEqual(question_time_window("since 2023 what changed", today=today)["from"], "2023")
        self.assertEqual(question_time_window("2025年10月的心电图", today=today)["from"], "2025")
        self.assertIsNone(question_time_window("tAS 的最小值是多少", today=today))
        win = {"from": "2022", "to": "2023"}
        self.assertTrue(in_window("2022-06-05", win))
        self.assertFalse(in_window("2021-03-14", win))
        self.assertTrue(in_window("rev *L", win))      # non-date axis values are not filtered
        self.assertTrue(in_window("", None))

    def test_unit_classifier_recognises_report_boilerplate_and_conclusions(self) -> None:
        from kb_pipeline.graph.units import classify_unit_text, unit_signals

        toc = ("www.northwind-checkup.test / 目录 / CONTENTS\n1 体检重要异常结果、复查建议及治疗建议 04 异常情况、专家建议与指导、标准治疗方案\n"
               "2 专家建议与指导 05 建议与指导\n3 健康体检结果 07 检查详细结果\n4 口腔检查结果 26 口腔健康整体解决方案/建议\n"
               "5 历年主要异常指标对比 28 历年数据对比及健康预测\n尊敬的李华先生,您好! 感谢您的光临。现将您2023年09月12日的体检结果报告如下。")
        self.assertEqual(classify_unit_text(toc, []), "boilerplate")
        guide = ("报告阅读说明书\n1. 您本次体检报告由健康信息、本次体检主要阳性结果和异常情况、专家指导建议及本次体检结果等部分组成。\n"
                 "2. 健康体检数据只是针对本次体检覆盖的相关器官的相关项目或指标的检查结果,并非能覆盖人体全部器官及全部指标。")
        self.assertEqual(classify_unit_text(guide, []), "boilerplate")
        self.assertEqual(classify_unit_text("尊敬的张三女士,您好!感谢您选择我们的体检服务,现将报告奉上。", []), "boilerplate")
        promo = "想随时随地看报告?扫码下载Northwind体检APP。关注公众号,检前检后全管理。IMAGE: 一个二维码。"
        self.assertEqual(classify_unit_text(promo, []), "boilerplate")
        findings = ("1. 体检重要异常结果、复查建议及治疗建议 / 阳性结果和异常情况 / 【1】甲状腺左叶小囊肿(TI-RADS2类) / 【2】轻度脂肪肝 / "
                    "【3】鼻中隔偏曲 / 【4】低密度脂蛋白胆固醇增高 / 【5】屈光不正 / 2. 专家建议与指导")
        self.assertEqual(classify_unit_text(findings, ["体检重要异常结果"]), "body")                  # the "advice" word is not a generic conclusion cue (every section has it)
        self.assertEqual(classify_unit_text("异常结果汇总\n1. 总胆固醇偏高", ["异常结果汇总"]), "conclusion")   # the "summary" word is
        domain_only = "阳性结果和异常情况 / 【1】甲状腺左叶小囊肿(TI-RADS2类) / 【2】轻度脂肪肝 / 【3】鼻中隔偏曲"
        self.assertEqual(classify_unit_text(domain_only, ["阳性结果和异常情况"]), "body")               # domain words are not in the code
        self.assertEqual(classify_unit_text(domain_only, ["阳性结果和异常情况"], conclusion_headings=["阳性结果和异常情况"]), "conclusion")
        self.assertEqual(classify_unit_text("Key findings: the device meets the 667 MHz timing budget across all corners.", ["Summary"]), "conclusion")
        body = "The CK/CK# clock is associated with the address and control pins. Transitions are centered with respect to the address bus."
        self.assertEqual(classify_unit_text(body, ["Functional Overview"]), "body")
        # Conclusion heading words learned by this KB's profile: recognised even when the generic word list lacks them
        self.assertEqual(classify_unit_text("本次评估结果:风险指数 36,中低风险。", ["评估总览"]), "body")
        self.assertEqual(classify_unit_text("本次评估结果:风险指数 36,中低风险。", ["评估总览"], conclusion_headings=["评估总览"]), "conclusion")
        # Parameter tables and ball maps are judged as before
        ac_table = ("| Parameter | Description | 667 MHz Min | Unit |\n| tAS | Address setup to CK rising edge, measured from the transition | 0.160 | ns |\n"
                    "| tCK | Clock cycle time, defined between two rising edges of the same clock | 1.5 | ns |")
        self.assertEqual(classify_unit_text(ac_table, ["Switching Characteristics"]), "body")
        self.assertGreaterEqual(unit_signals(toc)["toc_lines"], 3)

    def test_units_carry_axis_and_document_label(self) -> None:
        chunks = [_chunk(0, "第一段正文,讲的是器件的概述与特性。" * 3, doc="kb_t:1"), _chunk(1, "第二段正文,讲的是引脚。" * 3, doc="kb_t:1")]
        units = build_units(chunks, kb_id="kb_t", unit_chunks=3, axes={"kb_t:1": "2021-03-14"})
        self.assertEqual(units[0].axis, "2021-03-14")
        self.assertEqual(units[0].document_label, "a.pdf (2021-03-14)")
        rt = Unit.from_json(units[0].to_json())
        self.assertEqual(rt.axis, "2021-03-14")
        self.assertEqual(Unit.from_json({"unit_id": "u", "doc_id": "d"}).document_label, "d")
        # The axis value does not enter unit_id
        self.assertEqual(build_units(chunks, kb_id="kb_t", unit_chunks=3)[0].unit_id, units[0].unit_id)

    def test_extract_prompt_has_type_menu_caps_document_and_examples(self) -> None:
        from kb_pipeline.graph.extract import consolidate, record_caps, type_menu

        schema = ExtractionSchema(entity_types=("pin", "signal"), predicates=({"name": "has_pin", "source_parents": ["entity"], "target_parents": ["part"]},),
                                  language="Chinese", parent_types={"pin": "part", "signal": "part"},
                                  type_definitions={"pin": "a physical package terminal, not the logical signal it carries"})
        self.assertEqual(type_menu(schema), "pin: a physical package terminal, not the logical signal it carries (parent: part); signal (parent: part)")
        text = render_extract_prompt("正文", section="第二章 > 2.5", schema=schema, document="a.pdf (rev *L)", unit_kind="listing")
        self.assertIn("Entity_types: pin: a physical package terminal", text)
        self.assertIn("Document: a.pdf (rev *L)", text)
        self.assertIn("at most 30 records in total and at most 20 entity records", text)
        self.assertIn("do NOT extract entities or relationships from the document name or the section heading", text)
        self.assertIn("Example 1:", text)      # generic examples when the KB has none of its own
        self.assertIn("conclusion:", text)
        self.assertEqual(record_caps("body", has_table=True), (160, 80))
        self.assertEqual(record_caps("boilerplate"), (8, 5))
        self.assertEqual(record_caps("nonsense"), (120, 50))
        custom = ExtractionSchema(entity_types=("pin",), examples="Example 1:\nEntity_types: pin\nText: x\nOutput:\n(\"unit\"<|>body<|>s)\n<|COMPLETE|>")
        text2 = render_extract_prompt("正文", section="s", schema=custom)
        self.assertIn("Text: x", text2)
        self.assertNotIn("Central Institution", text2)
        # Fingerprint: type definitions, examples and caps all enter the fingerprint
        fp = extraction_fingerprint(SPEC, schema, max_gleanings=1)
        self.assertNotEqual(fp, extraction_fingerprint(SPEC, ExtractionSchema(entity_types=("pin", "signal"), predicates=schema.predicates, language="Chinese", parent_types=schema.parent_types), max_gleanings=1))
        self.assertNotEqual(fp, extraction_fingerprint(SPEC, ExtractionSchema(entity_types=("pin", "signal"), predicates=schema.predicates, language="Chinese", parent_types=schema.parent_types, type_definitions=schema.type_definitions, examples="x"), max_gleanings=1))
        # The longer description from a gleaning round wins: a containing one replaces, unrelated ones sit side by side,
        # exact duplicates are not repeated
        ents, _ = consolidate([{"name": "A", "type": "t", "description": "a pin"},
                               {"name": "A", "type": "t", "description": "A pin used for chip enable"},
                               {"name": "A", "type": "t", "description": "a pin"},
                               {"name": "A", "type": "t", "description": "something else"}], [])
        self.assertEqual(ents[0]["descriptions"], ["A pin used for chip enable", "something else"])

    def test_summaries_use_thresholds_json_rows_and_attribution(self) -> None:
        from kb_pipeline.graph.summarize import concatenate, description_rows, needs_summary, summarize_one, summarize_rows

        row = {"descriptions": ["42 岁", "43 岁", "受检者"], "description_sources": [{"source": "2021体检报告.pdf", "when": "2021-03-14"}, {"source": "2022体检报告.pdf", "when": "2022-06-05"}, {}]}
        rows = description_rows(row)
        self.assertEqual(rows[0], {"text": "42 岁", "source": "2021体检报告.pdf", "when": "2021-03-14"})
        self.assertEqual(rows[2], {"text": "受检者"})
        self.assertFalse(needs_summary(rows))                      # 3 short descriptions: concatenate, no model call
        self.assertTrue(needs_summary(rows + [{"text": "x"}]))     # ≥4 rows
        self.assertTrue(needs_summary([{"text": "长" * 900}, {"text": "文" * 900}]))   # over the token limit
        self.assertIn("[2021体检报告.pdf (2021-03-14)] 42 岁", concatenate(rows))
        seen: list[str] = []

        def chat(messages):
            seen.append(messages[-1]["content"])
            return "汇总"
        client = _client(chat)
        self.assertEqual(summarize_one(client, "李华", rows + [{"text": "x"}], language="Chinese"), "汇总")
        self.assertIn('{"text": "42 岁", "source": "2021体检报告.pdf", "when": "2021-03-14"}', seen[0])
        self.assertIn("keep the differences and attribute them", seen[0])
        stats = summarize_rows(client, [dict(row), {"descriptions": ["only one"]}], language="Chinese", name_of=lambda r: "n")
        self.assertEqual((stats["summarized"], stats["concatenated"], stats["skipped"]), (0, 2, 2))

    def test_facts_carry_flag_reference_range_axis_and_bound_distance(self) -> None:
        from kb_pipeline.graph.facts import (
            bound_distance, canonical_unit, fact_id, link_facts, normalize_fact, property_text, spec_text, value_text, wants_facts,
        )

        raw = {"subject": "体重指数", "property": "测量结果", "value": "25.5", "unit": "", "flag": "↑", "ref_min": "18.5", "ref_max": "23.99",
               "valid_from": "2023年09月12日", "period_text": "2026/09", "note": ""}
        f = normalize_fact(raw)
        self.assertEqual((f["flag"], f["ref_min_num"], f["ref_max_num"], f["valid_from"], f["period_text"]), ("↑", 18.5, 23.99, "2023-09-12", "2026/09"))
        self.assertAlmostEqual(bound_distance(f), round((25.5 - 23.99) / (23.99 - 18.5), 4))
        self.assertEqual(bound_distance({"value_num": 20.0, "ref_min_num": 18.5, "ref_max_num": 23.99}), 0.0)
        self.assertIsNone(bound_distance({"value_num": 1.0}))
        self.assertEqual(bound_distance({"value_num": 5.0, "ref_max_num": 4.0}), 1.0)
        self.assertEqual(canonical_unit("×10~9/L"), "10^9/L")
        self.assertEqual(canonical_unit("毫安"), "mA")
        self.assertEqual(canonical_unit("mmol/l"), "mmol/L")
        self.assertEqual(canonical_unit("weird"), "weird")
        # A leading M / m carries physical meaning: MA is no longer folded into mA as a spelling variant (Codex re-review
        # N06); an unrecognised unit is kept as is, and all-lowercase ma still normalises
        self.assertEqual(normalize_fact({"subject": "X", "property": "p", "value": "1", "unit": "MA"})["unit_canonical"], "MA")
        self.assertEqual(normalize_fact({"subject": "X", "property": "p", "value": "1", "unit": "ma"})["unit_canonical"], "mA")
        g = normalize_fact({"subject": "X", "property": "p", "value": "1", "valid_from": "not a date", "period_text": "x"})
        self.assertNotIn("valid_from", g)
        self.assertNotIn("period_text", g)
        a = normalize_fact({"subject": "X", "property": "p", "value": "1", "valid_from": "2024"})
        b = normalize_fact({"subject": "X", "property": "p", "value": "1", "valid_from": "2025"})
        self.assertNotEqual(fact_id("u", a), fact_id("u", b))
        self.assertIn("| when: 2023-09-12 | ref: 18.5~23.99", spec_text(f))
        self.assertIn("25.5 ↑", value_text(f))
        self.assertEqual(property_text({"concept": "bmi", "property": "测量结果", "symbol": "", "subject": "体重指数"}), "bmi · 测量结果 · 体重指数")
        unit = _unit("u1", doc="kb_t:1", points=("p1",))
        unit.axis = "2022-06-05"
        facts = [{"subject": "X", "property": "p", "value": "30", "unit_id": "u1", "ref_min_num": 18.5, "ref_max_num": 24.0, "value_num": 30.0},
                 {"subject": "X", "property": "q", "value": "1", "unit_id": "u1", "valid_from": "2020"}]
        link_facts(facts, [], units_by_id={"u1": unit})
        self.assertEqual((facts[0]["axis"], facts[0]["valid_from"], facts[1]["valid_from"]), ("2022-06-05", "2022-06-05", "2020"))
        self.assertGreater(facts[0]["bound_distance"], 0)
        conclusion = _unit("c1")
        conclusion.kind = "conclusion"
        self.assertTrue(wants_facts(conclusion))
        self.assertTrue(wants_facts(_unit("c2"), "conclusion"))

    def test_merge_records_description_sources_and_conclusion_kind(self) -> None:
        from kb_pipeline.graph.merge import combine_unit_kind

        units = [_unit("u1", doc="kb_t:1"), _unit("u2", doc="kb_t:2")]
        units[0].rel_path, units[0].axis = "2021体检报告.pdf", "2021-03-14"
        units[1].rel_path, units[1].axis = "2022体检报告.pdf", "2022-06-05"
        ex = {"u1": {"entities": [{"name": "李华", "type": "person", "descriptions": ["34 岁"]}], "relations": []},
              "u2": {"entities": [{"name": "李华", "type": "person", "descriptions": ["35 岁"]}], "relations": []}}
        merged = merge_extractions(units, ex, entity_types=("person",))
        e = merged["entities"][0]
        self.assertEqual(e["descriptions"], ["34 岁", "35 岁"])
        self.assertEqual(e["description_sources"], [{"source": "2021体检报告.pdf", "when": "2021-03-14"}, {"source": "2022体检报告.pdf", "when": "2022-06-05"}])
        self.assertEqual(combine_unit_kind("conclusion", "body"), "conclusion")
        self.assertEqual(combine_unit_kind("body", "conclusion"), "conclusion")
        self.assertEqual(combine_unit_kind("conclusion", "boilerplate"), "boilerplate")
        self.assertEqual(combine_unit_kind("listing", "conclusion"), "listing")

    def test_schema_suggests_definitions_profile_and_examples(self) -> None:
        from kb_pipeline.graph.schema import generate_examples

        answers = ["Health reports", "Chinese", "You are an expert.",
                   '{"entity_types": ["person", "indicator"]}',
                   '{"parent_types": {"person": "entity", "indicator": "property"}}',
                   '{"predicates": [{"name": "has_indicator", "description": "d", "source_parents": ["entity"], "target_parents": ["property"]}]}',
                   '{"definitions": {"person": "the examinee a report is about", "indicator": "a measured health metric", "ghost": "x"}}',
                   '{"subject_types": ["person", "nope"], "axis": "date", "conclusion_headings": ["异常结果汇总", "综合建议"], "extension_predicates": ["has_indicator", "bogus"]}']
        client = _client(list(answers))
        out = graph_schema.suggest(client, ["sample text"], examples=False)
        self.assertEqual(out["type_definitions"], {"person": "the examinee a report is about", "indicator": "a measured health metric"})
        self.assertEqual(out["profile"], {"subject_types": ["person"], "axis": "date", "conclusion_headings": ["异常结果汇总", "综合建议"],
                                          "boilerplate_headings": [], "listing_headings": [], "type_words": [], "extension_predicates": ["has_indicator"]})
        self.assertEqual(out["examples"], "")
        self.assertEqual(client.stats["calls"], 8)
        schema = ExtractionSchema(entity_types=("person", "indicator"), predicates=({"name": "has_indicator", "source_parents": ["entity"], "target_parents": ["property"]},),
                                  language="Chinese", parent_types={"person": "entity", "indicator": "property"})
        good = ('("unit"<|>body<|>s)##("entity"<|>李华<|>person<|>受检者)##("entity"<|>体重指数<|>indicator<|>指标)##'
                '("relationship"<|>李华<|>体重指数<|>has_indicator<|>报告给出体重指数<|>8)##'
                '("relationship"<|>体重指数<|>李华<|>has_indicator<|>越界的方向<|>3)<|COMPLETE|>')
        weak = '("entity"<|>A<|>person<|>x)##("entity"<|>B<|>indicator<|>y)##("relationship"<|>A<|>B<|>related_to<|>泛化<|>2)<|COMPLETE|>'
        text, stats = generate_examples(_client([weak, good]), ["短" * 220, "长文本 " * 60], schema)
        # Tried in descending length: the "long text" sample gets weak first (only related_to, below the bar), the "short"
        # one gets good; the reversed record in good violates the endpoints (1/2, not over half) and is removed
        self.assertEqual((stats["kept"], stats["fallback"], stats["violations_removed"], stats["dropped_empty"]), (1, 0, 1, 0))
        self.assertIn("Example 1:", text)
        self.assertIn('("relationship"<|>李华<|>体重指数<|>has_indicator<|>报告给出体重指数<|>8)', text)
        self.assertNotIn("越界的方向", text)
        self.assertNotIn("(parent:", text)                                          # the menu in examples lists names only, definitions are not repeated
        self.assertIn("Predicates: has_indicator", text)
        self.assertEqual(generate_examples(_client([]), ["x"], schema)[0], "")   # no sample long enough: empty

    def test_schema_fields_flow_through_config_source_and_version_apply(self) -> None:
        from kb_pipeline import discovery
        from kb_pipeline.graph.build import extraction_schema_for, scenario_profile
        from kb_pipeline.limits import normalize_examples, normalize_profile, normalize_type_definitions
        from kb_server.service import _apply_schema_version

        self.assertEqual(normalize_type_definitions({"Pin": " a terminal ", "ghost": "x"}, ["pin"]), {"pin": "a terminal"})
        self.assertEqual(normalize_type_definitions('{"pin": "t"}'), {"pin": "t"})
        self.assertEqual(normalize_profile({"axis": "DATE", "subject_types": "person, person", "extension_predicates": ["Has Pin"]}),
                         {"subject_types": ["person"], "axis": "date", "conclusion_headings": [], "boilerplate_headings": [], "listing_headings": [], "type_words": [], "extension_predicates": ["has_pin"]})
        self.assertEqual(normalize_profile({"axis": "weird"}), {})
        self.assertEqual(normalize_examples("  ex  "), "ex")
        for key in ("graph_type_definitions", "graph_examples", "graph_profile"):
            self.assertIn(key, discovery.CONFIG_KEYS)
            self.assertIn(key, discovery.DEFAULTS)
        with tempfile.TemporaryDirectory() as tmp:
            src = discovery.build_source(Path(tmp), "库", {
                "graph_entity_types": ["pin"], "graph_parent_types": {"pin": "part"},
                "graph_type_definitions": {"pin": "a terminal"}, "graph_examples": "Example 1: x",
                "graph_profile": {"subject_types": ["pin"], "axis": "version", "conclusion_headings": ["Summary"]},
            })
        self.assertEqual(src.graph_type_definitions, {"pin": "a terminal"})
        self.assertEqual(scenario_profile(src)["axis"], "version")
        schema = extraction_schema_for(src)
        self.assertEqual((schema.type_definitions, schema.examples, schema.subject_types), ({"pin": "a terminal"}, "Example 1: x", ("pin",)))
        current = {"graph_schema_versions": [{"id": "v9", "entity_types": ["pin"], "language": "Chinese",
                                              "type_definitions": {"pin": "t"}, "examples": "E", "profile": {"axis": "date"}},
                                             {"id": "v8", "entity_types": ["pin"]}]}
        updates = {"graph_schema_active": "v9"}
        _apply_schema_version(current, updates)
        self.assertEqual((updates["graph_type_definitions"], updates["graph_examples"], updates["graph_profile"]),
                         ({"pin": "t"}, "E", {"subject_types": [], "axis": "date", "conclusion_headings": [], "boilerplate_headings": [], "listing_headings": [], "type_words": [], "extension_predicates": []}))
        old = {"graph_schema_active": "v8"}
        _apply_schema_version(current, old)
        self.assertEqual((old["graph_type_definitions"], old["graph_examples"], old["graph_profile"]), (None, None, None))


class ResolutionEvidenceTests(unittest.TestCase):
    """2026-09-08 generality review: merges need evidence. Identifier-like names merge only on canonical identifier
    equality; opposite polarity never merges; a meaningful extra qualifier (subclass) never merges; a model "yes"
    must give an evidence category; every merged pair records its source and category."""

    def test_identifier_like_names_only_merge_on_canonical_equality(self) -> None:
        from kb_pipeline.graph.resolution import candidate_pairs, embedding_candidates, is_identifier_like, is_similar

        for name in ("t_DBE", "VIH", "ZK7C1049GN", "rs0000133", "ADRB2", "PE#", "51-85087"):
            self.assertTrue(is_identifier_like(name), name)
        for name in ("Northwind", "总胆固醇", "data input", "AutoStore", "Market Strategy Committee"):
            self.assertFalse(is_identifier_like(name), name)
        self.assertFalse(is_similar("t_DBE", "t_DOE"))          # identifiers one letter apart are two parameters (kb_001 once merged them)
        self.assertFalse(is_similar("ADRB2", "ADRB3"))
        self.assertFalse(is_similar("ADH2", "ALDH2"))
        ents = [{"title": "t_DBE", "type": "timing parameter"}, {"title": "t_DOE", "type": "timing parameter"},
                {"title": "tDOE", "type": "timing parameter"}]
        self.assertEqual(candidate_pairs(ents), [])            # the similarity route does not admit them
        vectors = {"t_DBE": [1.0, 0.0], "t_DOE": [0.99, 0.1], "tDOE": [0.99, 0.1]}
        self.assertEqual(embedding_candidates(ents, lambda titles: [vectors[t] for t in titles]), [])   # nor does the vector route
        from kb_pipeline.graph.resolution import auto_merge_pairs
        self.assertEqual(auto_merge_pairs(ents), [(1, 2)])     # t_DOE / tDOE: merged only on canonical identifier equality

    def test_polarity_and_qualifiers_block_merges(self) -> None:
        from kb_pipeline.graph.resolution import is_containment, is_similar, polarity_conflict

        self.assertTrue(polarity_conflict("data input", "Data Output"))
        self.assertTrue(polarity_conflict("输入数据", "输出数据"))
        self.assertTrue(polarity_conflict("VIH input high", "VIL input low"))
        self.assertFalse(polarity_conflict("data input", "input data"))
        self.assertFalse(polarity_conflict("Northwind", "Northwind Technologies"))
        self.assertFalse(is_similar("data input", "data output"))
        self.assertFalse(is_similar("output data valid", "input data valid"))
        # A subclass is not the same thing: a meaningful extra word blocks the merge; only type words / organisation
        # suffixes / "identifier + descriptor" count as qualifiers
        self.assertFalse(is_containment("并发症", "糖尿病并发症"))
        self.assertFalse(is_containment("视力减退原因待查", "双眼视力减退原因待查"))
        self.assertFalse(is_containment("Diabetes", "Type 2 Diabetes"))
        from kb_pipeline.graph.resolution import auto_merge_pairs, strip_type_words
        self.assertEqual(strip_type_words("Northwind Technologies AG"), "northwind")            # organisation suffix words: identical once stripped → merged directly
        rows = [{"title": "Northwind", "type": "ORG"}, {"title": "Northwind Technologies", "type": "ORG"},
                {"title": "输出使能", "type": "signal"}, {"title": "输出使能信号", "type": "signal"}]
        self.assertEqual(auto_merge_pairs(rows), [(0, 1)])                                 # Chinese type words come from the profile and only enter candidates, never merge directly
        self.assertTrue(is_containment("tASH", "tASH setup time"))
        self.assertFalse(is_containment("VSS", "I VSS"))                                    # the word before the identifier is a modifier: a current is not a voltage
        self.assertFalse(is_containment("FBGA", "48 FBGA"))
        self.assertFalse(is_containment("LDB#", "LDA#, LDB#"))
        self.assertFalse(is_containment("Northwind", "Northwind Semiconductor Technologies"))  # a meaningful extra word beyond type / suffix: not containment
        self.assertFalse(is_containment("Diabetes", "Diabetes Mellitus Complications"))

    def test_paren_aliases_type_name_words_and_symbol_descriptions(self) -> None:
        from kb_pipeline.graph.resolution import auto_merge_pairs, is_containment, paren_alias, strip_type_words, type_words_of

        self.assertEqual(paren_alias("体重指数(BMI)"), ("体重指数", "BMI"))
        self.assertEqual(paren_alias("低密度脂蛋白胆固醇 (LDL-C)"), ("低密度脂蛋白胆固醇", "LDL-C"))
        self.assertEqual(paren_alias("深色蔬菜(紫甘蓝)"), ("", ""))                       # the parenthesis holds an example, not an alias
        self.assertEqual(paren_alias("脑梗(缺血性脑卒中)"), ("", ""))
        # The reverse "identifier (expansion)" is an alias too: FDE (Forward-Deployed Engineer), AIP (AI Platform); the
        # expansion may contain Chinese characters and may exceed 24 characters
        self.assertEqual(paren_alias("FDE (Forward-Deployed Engineer)"), ("FDE", "Forward-Deployed Engineer"))
        self.assertEqual(paren_alias("FDE(Forward Deployed Engineer,前线部署工程师)"), ("FDE", "Forward Deployed Engineer,前线部署工程师"))
        self.assertEqual(paren_alias("AIP (AI Platform)"), ("AIP", "AI Platform"))      # the capital letters joined = AIP
        self.assertEqual(paren_alias("Skill(x)"), ("", ""))
        self.assertEqual(paren_alias("ΘJA (With Still Air)"), ("", ""))                    # the parenthesis holds a condition, the initials do not match
        self.assertEqual(paren_alias("HSB (STORE only)"), ("", ""))
        self.assertEqual(paren_alias("FDE(前线部署工程师)"), ("", ""))                       # a purely Chinese expansion cannot be checked by initials; left to the model
        fde = [{"title": "FDE", "type": "role"}, {"title": "FDE (Forward-Deployed Engineer)", "type": "role"},
               {"title": "FDE(Forward Deployed Engineer,前线部署工程师)", "type": "role"}, {"title": "Forward-Deployed Engineer", "type": "role"}]
        reasons: dict = {}
        groups = auto_merge_pairs(fde, reasons)
        self.assertEqual({frozenset(p) for p in groups} & {frozenset((0, 1)), frozenset((0, 2)), frozenset((1, 3))},
                         {frozenset((0, 1)), frozenset((0, 2)), frozenset((1, 3))})
        self.assertEqual({reasons[p] for p in groups}, {"paren"})                                   # the resolution log records a paren alias, not identifier equality
        reasons2: dict = {}
        auto_merge_pairs([{"title": "t_AS", "type": "timing parameter"}, {"title": "tAS", "type": "timing parameter"},
                          {"title": "361-ball FCBGA Package", "type": "package"}, {"title": "361-ball FCBGA", "type": "package"}], reasons2)
        self.assertEqual(sorted(reasons2.values()), ["identifier", "type_words"])
        ents = [{"title": "体重指数(BMI)", "type": "health indicator"}, {"title": "体重指数", "type": "health indicator"},
                {"title": "BMI", "type": "health indicator"}, {"title": "深色蔬菜(紫甘蓝)", "type": "food"}, {"title": "深色蔬菜", "type": "food"},
                {"title": "361-ball FCBGA Package", "type": "package"}, {"title": "361-ball FCBGA", "type": "package"},
                {"title": "血常规 test", "type": "laboratory test"}, {"title": "血常规", "type": "laboratory test"}]
        pairs = set(auto_merge_pairs(ents))
        self.assertIn((0, 1), pairs)
        self.assertIn((0, 2), pairs)
        self.assertNotIn((3, 4), pairs)
        self.assertIn((5, 6), pairs)                                                      # the words of the type name package are type words
        self.assertIn((7, 8), pairs)
        self.assertEqual(type_words_of("laboratory test"), {"laboratory", "test"})
        self.assertEqual(strip_type_words("361-ball FCBGA Package", type_words_of("package")), "361-ball fcbga")
        self.assertTrue(is_containment("CYP2C9", "CYP2C9酶"))                             # identifier + Chinese descriptor
        self.assertTrue(is_containment("CYP2C9", "细胞色素CYP2C9"))
        self.assertFalse(is_containment("并发症", "糖尿病并发症"))

    def test_variant_edges_cover_packaging_suffixes_and_paren_forms(self) -> None:
        from kb_pipeline.graph.merge import derive_variant_edges

        def ent(key, title, etype="product"):
            return {"key": key, "title": title, "type": etype, "unit_ids": ["u1"]}
        ents = [ent("a", "ZK14B108L-ZS25XI"), ent("b", "ZK14B108L-ZS25XIT"), ent("c", "ZK7C4021KV13"), ent("d", "ZK7C4021KV13-667FCXC"),
                ent("e", "脑梗", "medical condition"), ent("f", "脑梗(缺血性脑卒中)", "medical condition"), ent("g", "ZK7C4041KV13")]
        edges = {(r["source_key"], r["target_key"]) for r in derive_variant_edges(ents, [])}
        self.assertEqual(edges, {("b", "a"), ("d", "c"), ("f", "e")})                     # 4041 is not a variant of 4021

    def test_judge_answers_carry_a_category_and_merges_are_logged(self) -> None:
        from kb_pipeline.graph import resolution

        self.assertEqual(resolution.parse_answer_categories("1: yes abbreviation\n2: no\n3: yes (translation)\n4: yes\n5: yes bogus", 5),
                         ["abbreviation", "", "translation", "unspecified", "unspecified"])
        prompt = resolution.render_batch([(0, 1)], [{"title": "Northwind", "type": "ORG", "descriptions": ["maker"]},
                                                    {"title": "Northwind Technologies", "type": "ORG", "descriptions": []}])
        self.assertIn("abbreviation", prompt)
        self.assertIn("a specific thing and its general category", prompt)
        self.assertIn("a function and its module, a clause and its contract", prompt)      # the examples span domains
        for dead in ("merely look alike", "opposite direction or polarity"):            # identifiers / polarity are blocked by code, the prompt does not repeat it
            self.assertNotIn(dead, prompt, dead)
        self.assertIn("`<number>: yes <category>`", prompt)
        ents = [{"key": "a", "title": "Northwind", "type": "ORG", "descriptions": ["maker"], "unit_ids": ["t1"], "frequency": 3, "aliases": []},
                {"key": "b", "title": "Northwindn", "type": "ORG", "descriptions": ["semiconductor maker"], "unit_ids": ["t2"], "frequency": 1, "aliases": []},
                {"key": "c", "title": "data input", "type": "signal", "descriptions": ["input"], "unit_ids": ["t3"], "frequency": 1, "aliases": []},
                {"key": "d", "title": "data output", "type": "signal", "descriptions": ["output"], "unit_ids": ["t4"], "frequency": 1, "aliases": []}]
        client = _client(["1: yes alias"])
        merged, _, stats = resolution.resolve(client, ents, [])
        self.assertEqual(stats["candidates"], 1)                     # data input / output never enter the candidates at all
        self.assertEqual(stats["yes_by_category"], {"alias": 1})
        self.assertEqual(stats["_log"], [{"kept": "a", "merged": "b", "kept_title": "Northwind", "merged_title": "Northwindn",
                                          "source": "lexical", "category": "alias"}])
        self.assertEqual({e["key"] for e in merged}, {"a", "c", "d"})
        src = (Path(__file__).resolve().parents[1] / "kb_pipeline" / "graph" / "build.py").read_text(encoding="utf-8")
        self.assertIn('"resolution_log": resolution_log', src)

    def test_embedding_alias_merges_get_a_second_stricter_check(self) -> None:
        from kb_pipeline.graph import resolution

        self.assertEqual(resolution.parse_same("1: same\n2: narrower\n3: different\n4: broader", 5), [True, False, False, False, False])
        self.assertEqual(resolution.parse_verdicts("1: same\n2: narrower\n3: 不同\n4: broader", 5), ["same", "narrower", "different", "broader", ""])
        prompt = resolution.render_hyponym_batch([(0, 1)], [{"title": "尿酸增高", "type": "finding", "descriptions": ["化验所见"]},
                                                              {"title": "高尿酸血症", "type": "finding", "descriptions": ["诊断"]}])
        self.assertIn("a finding and a diagnosis", prompt)
        self.assertIn("a function and its module, a clause and its contract", prompt)      # the examples span domains, not only medicine
        self.assertIn("are never `same`", prompt)                                          # this sentence is the task definition; 88d3e91 removed it once, the paper KB got looser, restored
        self.assertIn("`<number>: same|broader|narrower|different`", prompt)
        ents = [{"key": "a", "title": "尿酸增高", "type": "finding", "descriptions": ["化验所见"], "unit_ids": ["t1"], "frequency": 2, "aliases": []},
                {"key": "b", "title": "高尿酸血症", "type": "finding", "descriptions": ["诊断"], "unit_ids": ["t2"], "frequency": 1, "aliases": []},
                {"key": "c", "title": "中风", "type": "finding", "descriptions": ["急性脑血管病"], "unit_ids": ["t3"], "frequency": 2, "aliases": []},
                {"key": "d", "title": "脑卒中", "type": "finding", "descriptions": ["急性脑血管病"], "unit_ids": ["t4"], "frequency": 1, "aliases": []}]
        vectors = {"尿酸增高": [1.0, 0.0], "高尿酸血症": [0.98, 0.15], "中风": [0.0, 1.0], "脑卒中": [0.1, 0.99]}
        # The first pass says alias for both pairs; the second pass: a (elevated uric acid) is a manifestation of
        # b (hyperuricemia), so narrower; c / d (stroke / cerebral stroke) are same
        client = _client(["1: yes alias\n2: yes alias", "1: narrower\n2: same"])
        merged, _, stats = resolution.resolve(client, ents, [], embed=lambda titles: [vectors[t] for t in titles])
        self.assertEqual((stats["embedding_candidates"], stats["yes"], stats["rechecked"], stats["recheck_dropped"]), (2, 2, 2, 1))
        # Pairs the model calls alias are re-checked regardless of route (profile type words let these two pairs in):
        # s1 / s2 (permission model / permission) judged narrower is not merged; s3 / s4 (AutoStore feature / AutoStore)
        # judged same is merged. All four entities share a type: sameness is asked in one batch, candidates sorted by
        # title put the AutoStore pair first and the permission pair second, and the re-check batch follows that order
        ents2 = [{"key": "s1", "title": "权限模型", "type": "technology", "descriptions": ["平台的权限模型"], "unit_ids": ["t1"], "frequency": 1, "aliases": []},
                 {"key": "s2", "title": "权限", "type": "technology", "descriptions": ["访问权限"], "unit_ids": ["t2"], "frequency": 3, "aliases": []},
                 {"key": "s3", "title": "AutoStore功能", "type": "technology", "descriptions": ["断电自动存储"], "unit_ids": ["t3"], "frequency": 1, "aliases": []},
                 {"key": "s4", "title": "AutoStore", "type": "technology", "descriptions": ["断电自动存储"], "unit_ids": ["t4"], "frequency": 2, "aliases": []}]
        self.assertEqual(resolution.candidate_pairs(ents2), [])
        self.assertEqual(resolution.candidate_pairs(ents2, type_words=["模型", "功能"]), [(3, 2), (1, 0)])
        client2 = _client(["1: yes alias\n2: yes alias", "1: same\n2: narrower"])
        merged2, _, stats2 = resolution.resolve(client2, ents2, [], type_words=["模型", "功能"])
        self.assertEqual((stats2["rechecked"], stats2["recheck_dropped"]), (2, 1))
        self.assertEqual({e["key"] for e in merged2}, {"s1", "s2", "s4"})
        self.assertEqual([r["reason"] for r in stats2["_rejected"]], ["recheck"])
        # Literally similar (edit distance) + alias pairs are re-checked too: output buffers / Output drivers differ by
        # a meaningful word; "identifier + descriptor" (CLK / CLK clock input) and spelling pairs (Northwindn / Northwind)
        # are not re-checked. Same type → asked in one batch, sorted by title
        ents3 = [{"key": "r1", "title": "output buffers", "type": "component", "descriptions": ["buffers"], "unit_ids": ["t1"], "frequency": 1, "aliases": []},
                 {"key": "r2", "title": "Output drivers", "type": "component", "descriptions": ["drivers"], "unit_ids": ["t2"], "frequency": 2, "aliases": []},
                 {"key": "r3", "title": "CLK", "type": "component", "descriptions": ["clock"], "unit_ids": ["t3"], "frequency": 2, "aliases": []},
                 {"key": "r4", "title": "CLK clock input", "type": "component", "descriptions": ["clock input"], "unit_ids": ["t4"], "frequency": 1, "aliases": []},
                 {"key": "r5", "title": "Northwind", "type": "component", "descriptions": ["maker"], "unit_ids": ["t5"], "frequency": 2, "aliases": []},
                 {"key": "r6", "title": "Northwindn", "type": "component", "descriptions": ["maker"], "unit_ids": ["t6"], "frequency": 1, "aliases": []}]
        self.assertEqual(resolution.candidate_pairs(ents3), [(2, 3), (4, 5), (0, 1)])
        client3 = _client(["1: yes alias\n2: yes spelling\n3: yes alias", "1: different"])
        merged3, _, stats3 = resolution.resolve(client3, ents3, [])
        self.assertEqual(stats3["yes"], 3)
        self.assertEqual((stats3["rechecked"], stats3["recheck_dropped"]), (1, 1))
        self.assertEqual({e["key"] for e in merged3}, {"r1", "r2", "r3", "r5"})
        self.assertEqual({e["key"] for e in merged}, {"a", "b", "c"})
        self.assertEqual(stats["_log"], [{"kept": "c", "merged": "d", "kept_title": "中风", "merged_title": "脑卒中", "source": "embedding", "category": "alias"}])
        # Pairs blocked by the second pass leave a trace too (not merged, not in the resolution log), stored as
        # resolution_rejected in the graph-build output
        self.assertEqual(stats["_rejected"], [{"a": "a", "b": "b", "a_title": "尿酸增高", "b_title": "高尿酸血症", "source": "embedding",
                                               "category": "alias", "reason": "recheck", "verdict": "narrower"}])     # a withdrawn pair records the relation the model answered
        self.assertEqual(stats["yes_by_category"], {"alias": 1})
        src = (Path(__file__).resolve().parents[1] / "kb_pipeline" / "graph" / "build.py").read_text(encoding="utf-8")
        self.assertIn('"resolution_rejected": resolution_rejected', src)

    def test_profile_headings_replace_domain_words_in_code(self) -> None:
        from kb_pipeline.graph.units import _BOILER_STRONG_RE, _CONCLUSION_HINT_RE, _LISTING_HINT_RE, classify_unit_text
        from kb_pipeline.limits import normalize_profile

        for word in ("主检医师", "总检", "体检小结", "阳性结果", "worldwide", "sales", "ordering", "ball", "pinout", "引脚"):
            self.assertNotIn(word, _CONCLUSION_HINT_RE.pattern + _BOILER_STRONG_RE.pattern + _LISTING_HINT_RE.pattern, word)
        sales = "Worldwide Sales and Design Support Cypress maintains a worldwide network of offices, solution centers and distributors."
        self.assertEqual(classify_unit_text(sales, ["Worldwide Sales and Design Support"]), "body")
        self.assertEqual(classify_unit_text(sales, ["Worldwide Sales and Design Support"], boilerplate_headings=["Worldwide Sales"]), "boilerplate")
        grid = "| A1 | VDD |\n| A2 | GND |\n| A3 | DQ0 |\n| A4 | DQ1 |\n| B1 | CK |\n| B2 | CK# |\n| B3 | DQ2 |\n| B4 | DQ3 |\n" * 2
        self.assertEqual(classify_unit_text("Ball Map\n" + grid, ["Ball Map"], listing_headings=["Ball Map"]), "listing")
        prof = normalize_profile({"axis": "auto", "boilerplate_headings": ["Worldwide Sales"], "listing_headings": ["Ball Map"]})
        self.assertEqual((prof["boilerplate_headings"], prof["listing_headings"], prof["subject_types"]), (["Worldwide Sales"], ["Ball Map"], []))
        # Type words also come from the profile: the code has no TYPE_WORDS any more; profile type_words takes single
        # words only, at most 10
        from kb_pipeline.graph import resolution
        self.assertFalse(hasattr(resolution, "TYPE_WORDS"))
        self.assertEqual(normalize_profile({"type_words": ["信号", "status register", " 指标 "]})["type_words"], ["信号", "指标"])
        ents = [{"title": "复位信号", "type": "signal"}, {"title": "复位", "type": "signal"},
                {"title": "血红蛋白指标", "type": "indicator"}, {"title": "血红蛋白", "type": "indicator"}]
        # Profile type words do not merge directly (permission model / permission are identical once stripped yet not
        # necessarily one thing); the pair only becomes a candidate for the model to judge from the descriptions
        self.assertEqual(resolution.auto_merge_pairs(ents), [])
        pairs_plain = {frozenset(p) for p in resolution.candidate_pairs(ents)}
        pairs_learned = {frozenset(p) for p in resolution.candidate_pairs(ents, type_words=["信号", "指标"])}
        self.assertNotIn(frozenset((0, 1)), pairs_plain)                                   # reset signal / reset: without the profile word it is not a candidate
        self.assertEqual(pairs_learned, {frozenset((0, 1)), frozenset((2, 3))})
        self.assertTrue(resolution.is_containment("复位", "复位信号", ["信号"]))
        self.assertFalse(resolution.is_containment("复位", "复位信号"))                     # without the profile word the extra "signal" is a meaningful qualifier
        self.assertEqual(resolution.strip_type_words("大模型", ("模型",)), "大模型")          # fewer than two characters would remain after stripping: not stripped
        # Similarity for mixed Chinese / English: an English run counts as one feature, so "reusable Skill / Skill" no
        # longer looks alike; "standardised SaaS / standard SaaS" and "ERP industry / ERP industry" (spacing) still do
        self.assertFalse(resolution.is_similar("可复用Skill", "Skill"))
        self.assertFalse(resolution.is_similar("交付Playbook", "Playbook"))
        self.assertTrue(resolution.is_similar("标准化SaaS", "标准SaaS"))
        self.assertTrue(resolution.is_similar("ERP 行业", "ERP行业"))
        self.assertTrue(resolution.is_similar("非易失性存储器", "非易失存储器"))
        src = (Path(__file__).resolve().parents[1] / "kb_pipeline" / "graph" / "prompts.py").read_text(encoding="utf-8")
        self.assertIn('"type_words": ["<word>", ...]', src)
        for word in ("has_pin, part_of", "tAS, VCC", "Ball Map", "Worldwide Sales", "operates_in_mode"):
            self.assertNotIn(word, src, word)                                             # the prompt examples are not only electronics datasheets


class ExtractionFixRegressionTests(_CodexAudit20260906TestsSupport, _CodexFinalTestsSupport, unittest.TestCase):
    """Regressions for problems found by past re-reviews, health checks and audits; each case's docstring names the
    source and the symptom at the time."""

    def test_short_ascii_entity_names_match_on_word_boundaries(self) -> None:
        """B10: CE is not counted inside CELL; Chinese names still match by substring."""
        from kb_pipeline.graph.merge import attribute_mentions
        from kb_pipeline.graph.units import Unit

        unit = Unit(unit_id="u1", doc_id="d", rel_path="a.pdf", section_path=[], block_ids=[],
                    chunk_uids=["c1", "c2"], point_ids=["p1", "p2"], n_tokens=10, text="")
        entities = [{"key": "ce", "title": "CE", "aliases": ["chip enable"], "unit_ids": ["u1"]},
                    {"key": "华为", "title": "华为", "aliases": [], "unit_ids": ["u1"]}]
        texts = {"p1": "the CELL array; CE# pin goes low; chip enable", "p2": "华为云与华为终端"}
        rows = attribute_mentions(entities, {"u1": unit}, texts)
        by = {(r["entity_key"], r["point_id"]): r["count"] for r in rows}
        self.assertEqual(by[("ce", "p1")], 2)
        self.assertNotIn(("ce", "p2"), by)
        self.assertEqual(by[("华为", "p2")], 2)

    def test_f05_extraction_prompt_shows_predicate_descriptions_and_endpoints(self) -> None:
        from kb_pipeline.graph.extract import ExtractionSchema, render_extract_prompt

        schema = ExtractionSchema(entity_types=("component", "pin"), predicates=(
            {"name": "has_pin", "description": "a component exposes a pin", "source_parents": ["component"], "target_parents": ["interface"]},
            {"name": "conforms_to"},
        ))
        prompt = render_extract_prompt("text", section="s", schema=schema)
        self.assertIn("has_pin: a component exposes a pin (component -> interface); conforms_to; related_to", prompt)

    def test_protocol_failures_in_entity_extraction_are_not_cached_as_empty_success(self) -> None:
        from kb_pipeline.graph import prompts
        from kb_pipeline.graph.extract import ExtractionSchema, GraphExtractor, extraction_response_ok
        from kb_pipeline.graph.llm import LLMMalformedResponse
        from kb_pipeline.graph.units import Unit

        td, rd, cd = prompts.TUPLE_DELIMITER, prompts.RECORD_DELIMITER, prompts.COMPLETION_DELIMITER
        self.assertFalse(extraction_response_ok("I cannot provide the requested format."))
        self.assertTrue(extraction_response_ok(cd))                                            # a legitimate "no entities"
        self.assertTrue(extraction_response_ok(f'("entity"{td}ACME{td}organization{td}a company){rd}{cd}'))
        self.assertTrue(extraction_response_ok(f'("unit"{td}boilerplate){rd}{cd}'))
        self.assertFalse(extraction_response_ok(f'garbage{rd}more garbage{rd}{cd}'))
        unit = Unit(unit_id="u", doc_id="d", rel_path="d.pdf", section_path=["S"], block_ids=["b"], chunk_uids=["c"],
                    point_ids=["p"], n_tokens=10, text="t", order=0)
        client = _fake_chat_client(["nope", "still nope", "nope again"])              # still malformed after two corrections → failure, not cached
        with self.assertRaises(LLMMalformedResponse):
            GraphExtractor(client, ExtractionSchema(entity_types=("organization",)), max_gleanings=0).extract(unit)
        self.assertEqual(client.stats["malformed"], 1)
        ok = _fake_chat_client([f'("entity"{td}ACME{td}organization{td}a company){rd}{cd}'])
        res = GraphExtractor(ok, ExtractionSchema(entity_types=("organization",)), max_gleanings=0).extract(unit)
        self.assertEqual([e["name"] for e in res.entities], ["ACME"])
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"; db.init_db(state)
            with db.connect(state) as con:
                for uid, ents, stats in (("u-bad", [], {"records": 1, "malformed": 1}), ("u-empty", [], {"records": 0, "malformed": 0}),
                                         ("u-ok", [{"name": "A", "type": "t", "description": ""}], {"records": 1, "malformed": 0})):
                    db.save_graph_extraction(con, kb_id="kb", unit_id=uid, fingerprint="fp", entities=ents, relations=[],
                                             model="m", calls=1, stats=stats)
                con.commit()
                self.assertEqual(db.graph_extraction_unit_ids(con, "kb", "fp"), {"u-bad", "u-empty", "u-ok"})
                self.assertEqual(db.graph_extraction_unit_ids(con, "kb", "fp", skip_empty_malformed=True), {"u-empty", "u-ok"})
        self.assertIn("skip_empty_malformed=True", _repo_file("app/kb_pipeline/graph/build.py"))


class WideTableRenderingTests(unittest.TestCase):
    """Wide tables reach the model as "column: value" pairs (spot check: entity descriptions read the wrong column of a
    15-column comparison sheet); chunk text is untouched and only expanded units get a render-versioned unit_id."""

    NATIVE = ("SHEET: 竞品对比表\nROWS: 126-129\nHEADER: 一级模块 | 二级模块 | 功能 | 描述 | 钉钉 | 飞书 | 热聊v4.8.61 | 备注\n"
              "聊天能力 | 会话列表 | 移除会话 |  | 1 | 0 | 1 | \n聊天能力 | 会话列表 | 标签 |  | 1 | 1 | 0 | 计划支持")
    NARROW = "SHEET: s\nROWS: 1-2\nHEADER: 功能 | 钉钉 | 飞书\n标签 | 1 | 0"
    MARKDOWN = "| 功能 | a | b | c | d | e |\n|---|---|---|---|---|---|\n| 标签 | 1 | 0 | 1 |  | 0 |\n\n后面的正文 | 不是表"

    def test_native_and_markdown_wide_rows_become_name_value_pairs(self) -> None:
        from kb_pipeline.graph.tabletext import expand_wide_tables, table_render_tag

        out = expand_wide_tables(self.NATIVE)
        self.assertIn("HEADER: 一级模块 | 二级模块", out)                                          # prefix lines stay as they are
        self.assertIn("一级模块: 聊天能力 | 二级模块: 会话列表 | 功能: 移除会话 | 钉钉: 1 | 飞书: 0 | 热聊v4.8.61: 1", out)
        self.assertIn("功能: 标签 | 钉钉: 1 | 飞书: 1 | 热聊v4.8.61: 0 | 备注: 计划支持", out)
        self.assertNotIn("聊天能力 | 会话列表 | 标签", out)                                          # empty cells skipped, column names keep positions clear
        self.assertIsNone(expand_wide_tables(self.NARROW))                                       # tables with few columns are left alone
        md = expand_wide_tables(self.MARKDOWN)
        self.assertIn("功能: 标签 | a: 1 | b: 0 | c: 1 | e: 0", md)
        self.assertIn("后面的正文 | 不是表", md)                                                    # after a blank line it is no longer a table
        self.assertEqual(table_render_tag(self.NATIVE), table_render_tag(self.MARKDOWN))
        self.assertTrue(table_render_tag(self.NATIVE).startswith("wide-table-v1"))
        self.assertEqual(table_render_tag(self.NARROW), "")

    # 2026-09-29 audit: positions are checked before names are assigned by position
    LEAD_EMPTY = ("SHEET: editions\nROWS: 2-4\nHEADER: | Spec | Team | Pro | Plus | Standard | Flagship\n"
                  "| Editors | - | Web | Web | Web | Web+PC\n"
                  "|  | Private | Public |  | Hybrid\n"
                  "Storage | Quota | 100G | 1T | 1T | 2T | Unlimited")
    SPLIT_PIECES = ("SHEET: survey\nROWS: 7-7\nHEADER: Submitted | ID | Team | Question 1 | Question 2 | Question 3\n"
                    "Submitted: 2026-08-24 08:42:00 | Question 2: less stock on hand, fewer stock-outs\n"
                    "Submitted: 2026-08-24 08:42:00 | F: sales of the last 7/14/30 days")
    DOUBLE_LABELLED = ("SHEET: features\nROWS: 9-9\nHEADER: Module | Level 1 | Level 2 | Level 3 | Description | Public\n"
                       "Module: Module: Contacts | Level 1: Description: use the group in documents | Level 2: Public: yes")
    MD_PIECES = ("| Model | a | b | c | d | e |\n|---|---|---|---|---|---|\n"
                 "| Model: customer model | c: provided by the customer | column 9: note |\n"
                 "| In-house model | 1 | 0 |\n"
                 "| Open model | 1 | 0 | 1 | 0 | 1 |")

    def test_rows_with_an_empty_first_cell_keep_their_columns(self) -> None:
        """A row whose first cell is empty starts with "| " in the text and has lost one empty cell; it is put back
        before names are assigned, otherwise the whole row shifts one column to the left."""
        from kb_pipeline.graph.tabletext import RENDER_TAG, RENDER_TAG_REALIGNED, expand_wide_tables, table_render_tag

        out = expand_wide_tables(self.LEAD_EMPTY)
        self.assertIn("Spec: Editors | Team: - | Pro: Web | Plus: Web | Standard: Web | Flagship: Web+PC", out)
        self.assertIn("Team: Private | Pro: Public | Standard: Hybrid", out)                       # two empty cells in a row
        self.assertIn("A: Storage | Spec: Quota | Team: 100G", out)                                # unnamed first header cell: column letter
        self.assertNotIn("Spec: Private", out)
        self.assertFalse([line for line in out.split("\n") if line.startswith("| Spec:")])          # the first version merged "| " into the first cell
        self.assertIn("HEADER: | Spec | Team", out)                                                # prefix lines untouched
        self.assertEqual(table_render_tag(self.LEAD_EMPTY), RENDER_TAG_REALIGNED)
        self.assertNotEqual(RENDER_TAG, RENDER_TAG_REALIGNED)
        self.assertEqual(table_render_tag(self.NATIVE), RENDER_TAG)                                # same rendering, same version, cache kept

    def test_rows_that_already_carry_column_names_are_left_alone(self) -> None:
        """The continuation pieces of an over-long row are already "column: value" and skip empty cells; pairing them
        by position once more puts every value under the wrong column."""
        from kb_pipeline.graph.tabletext import RENDER_TAG_REALIGNED, expand_wide_tables, table_render_tag

        self.assertIsNone(expand_wide_tables(self.SPLIT_PIECES))                                   # nothing to rewrite: the original text is used
        self.assertEqual(table_render_tag(self.SPLIT_PIECES), "")
        mixed = self.SPLIT_PIECES + "\n2026-08-25 09:00:00 | 17 | Purchasing | x | y | z"
        out = expand_wide_tables(mixed)
        self.assertIn("Submitted: 2026-08-24 08:42:00 | Question 2: less stock on hand, fewer stock-outs\n", out)
        self.assertIn("Submitted: 2026-08-24 08:42:00 | F: sales of the last 7/14/30 days", out)
        self.assertIn("Submitted: 2026-08-25 09:00:00 | ID: 17 | Team: Purchasing | Question 1: x | Question 2: y | Question 3: z", out)
        self.assertNotIn("Submitted: Submitted", out)
        self.assertEqual(table_render_tag(mixed), RENDER_TAG_REALIGNED)
        # chunk text the chunker already paired by position once more: the inner name is the right one
        fixed = expand_wide_tables(self.DOUBLE_LABELLED)
        self.assertIn("Module: Contacts | Description: use the group in documents | Public: yes", fixed)
        self.assertNotIn("Module: Module", fixed)
        self.assertNotIn("Level 2: Public", fixed)
        self.assertEqual(table_render_tag(self.DOUBLE_LABELLED), RENDER_TAG_REALIGNED)
        # an ordinary row whose value happens to start with a column name does not qualify: not every cell is named
        plain = ("SHEET: s\nROWS: 1-1\nHEADER: Module | Level 1 | Level 2 | Level 3 | Description | Public\n"
                 "Module: Contacts | groups | dynamic | scenes | text | yes")
        self.assertIn("Module: Module: Contacts | Level 1: groups", expand_wide_tables(plain))

    def test_markdown_rows_with_unreliable_positions_are_left_alone(self) -> None:
        from kb_pipeline.graph.tabletext import RENDER_TAG_REALIGNED, expand_wide_tables, table_render_tag

        out = expand_wide_tables(self.MD_PIECES)
        self.assertIn("| Model: customer model | c: provided by the customer | column 9: note |", out)   # a continuation piece, untouched
        self.assertIn("| In-house model | 1 | 0 |", out)                 # cell count differs from the header: no names assigned
        self.assertIn("Model: Open model | a: 1 | b: 0 | c: 1 | d: 0 | e: 1", out)
        self.assertNotIn("Model: Model", out)
        self.assertNotIn("Model: In-house model", out)
        self.assertEqual(table_render_tag(self.MD_PIECES), RENDER_TAG_REALIGNED)

    def test_only_expanded_units_get_a_new_unit_id(self) -> None:
        from kb_pipeline.graph.tabletext import table_render_tag
        from kb_pipeline.graph.units import unit_id_for

        plain = unit_id_for("kb", "d", self.NARROW, positions=[0])
        self.assertEqual(unit_id_for("kb", "d", self.NARROW, positions=[0], render_tag=table_render_tag(self.NARROW)), plain)
        wide = unit_id_for("kb", "d", self.NATIVE, positions=[0])
        self.assertNotEqual(unit_id_for("kb", "d", self.NATIVE, positions=[0], render_tag=table_render_tag(self.NATIVE)), wide)
        src = (Path(__file__).resolve().parents[1] / "kb_pipeline" / "graph" / "extract.py").read_text(encoding="utf-8")
        self.assertIn("render_extract_prompt(expand_wide_tables(unit.text) or unit.text", src)   # the extraction prompt uses the expanded text
        src = (Path(__file__).resolve().parents[1] / "kb_pipeline" / "graph" / "units.py").read_text(encoding="utf-8")
        self.assertIn("render_tag=table_render_tag(text)", src)
