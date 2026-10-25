"""Facts spine and view layer: property keys, reconciliation, value normalisation, page compilation, evaluation."""
from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from kb_pipeline.graph import prompts, resolution
from kb_pipeline.graph.extract import ExtractionSchema, parse_records
from kb_pipeline.graph.llm import ChatClient, LLMCache, LLMSpec, cache_key
from kb_pipeline.graph.merge import merge_extractions
from kb_pipeline.graph.units import ChunkRef, Unit, build_units
from kb_pipeline.parsers.common import parser_profile_for_path
from kb_pipeline.vector.qdrant import activate_graph_aliases

from _support import _CodexAudit20260906TestsSupport, _CodexFinalTestsSupport, _chunk, _client, _fake_chat_client, _repo_file, _unit


class RecallScoringTests(unittest.TestCase):
    """Graph recall weights entity / relation scores by idf when aggregating them onto chunks. On the 32 capability
    questions of kb_003, without idf a seed entity such as a device name that appears in hundreds of chunks pushed
    the document history page and the ordering information table to the top of every question, leaving the answer
    chunk outside the top ten."""

    def test_rare_entities_outweigh_hub_entities(self) -> None:
        from kb_pipeline.graph.recall import evidence_weight, mention_weight

        hub = mention_weight(0.9, count=10, df=150, n_chunks=295)      # device name: mentioned ten times in the document history page
        rare = mention_weight(0.6, count=1, df=3, n_chunks=295)        # PE#: appears in only three chunks
        self.assertGreater(rare, hub)
        # Count saturation: ten mentions are worth less than three times one mention
        self.assertLess(mention_weight(0.9, count=10, df=3, n_chunks=295),
                        3 * mention_weight(0.9, count=1, df=3, n_chunks=295))
        # Relation evidence is on the same scale as entity attribution: a relation backed by only three chunks
        # contributes about as much as an equally scored rare entity
        self.assertAlmostEqual(evidence_weight(0.6, df=3, n_chunks=295),
                               mention_weight(0.6, count=0, df=3, n_chunks=295), places=6)
        # Degenerate input does not blow up
        self.assertEqual(mention_weight(0.0, count=0, df=0, n_chunks=0), 0.0)
        self.assertGreater(evidence_weight(1.0, df=0, n_chunks=0), 0.0)

    def test_list_pages_do_not_win_by_naming_everything(self) -> None:
        """The ordering information table mentions fifteen seed entities and fifty names; the answer chunk mentions
        two seed entities and is evidence for one seed relation. The latter must rank first."""
        from kb_pipeline.graph.recall import chunk_score

        listing = chunk_score([3.0] * 15, [], hub=50, mean_hub=13.6)
        answer = chunk_score([3.5, 1.0], [4.1], hub=10, mean_hub=13.6)
        self.assertGreater(answer, listing)
        # Only the strongest three entity attributions count: listing more names does not raise the score
        self.assertEqual(chunk_score([3.0] * 3, [], hub=10, mean_hub=13.6), chunk_score([3.0] * 30, [], hub=10, mean_hub=13.6))
        # The larger the total mention count (the more it looks like a listing page), the lower the same attributions score
        self.assertLess(chunk_score([3.0], [], hub=60, mean_hub=13.6), chunk_score([3.0], [], hub=6, mean_hub=13.6))
        self.assertEqual(chunk_score([], [], hub=0, mean_hub=0.0), 0.0)

    def test_candidates_are_reranked_by_question_text_match(self) -> None:
        """The graph score only shortlists candidates; the final rank depends on the chunk text's similarity to the
        question and on word overlap."""
        from kb_pipeline.graph.recall import lexical_overlap, rerank_score

        q = "ZK7C4021KV13 的地址建立时间 tAS 在 667 MHz 下的最小值是多少？"
        table = "tAS A to CK setup 667 MHz Min 0.160 ns 600 MHz Min 0.180 ns ZK7C4021KV13"
        history = "Document History Page ZK7C4021KV13/ZK7C4041KV13 Updated Functional Description New Template"
        self.assertGreater(lexical_overlap(q, table), lexical_overlap(q, history))
        self.assertEqual(lexical_overlap("", table), 0.0)
        self.assertLessEqual(lexical_overlap(q, q), 3.0)
        # The graph rank is only a weak prior: a 50-place rank gap does not outweigh a 0.1 similarity gap
        self.assertGreater(rerank_score(0.60, 0.0, 51), rerank_score(0.50, 0.0, 1))
        self.assertGreater(rerank_score(0.50, 0.0, 1), rerank_score(0.50, 0.0, 51))
        self.assertGreater(rerank_score(0.50, 1.0, 1), rerank_score(0.50, 0.0, 1))


class FactsSpineTests(unittest.TestCase):
    """Second batch of the 2026-09-07 plan (facts spine): property concept keys, cross-document reconciliation, three
    named vectors for facts, an optional page collection, and on the retrieval side the time window / greedy cover /
    in-network relations / conclusion boost."""

    def test_concept_norm_and_key(self) -> None:
        from kb_pipeline.graph.concepts import concept_key, concept_norm

        self.assertEqual(concept_norm("Supply voltage", "V_CC"), "vcc")
        self.assertEqual(concept_norm("Supply voltage", "$V_{CC}$"), "vcc")
        self.assertEqual(concept_norm("VCC"), "vcc")
        self.assertEqual(concept_norm("总胆固醇测量结果"), "总胆固醇")
        self.assertEqual(concept_norm("总胆固醇 值"), "总胆固醇")
        self.assertEqual(concept_norm("Total Cholesterol"), "totalcholesterol")
        self.assertEqual(concept_norm(""), "")
        self.assertEqual(concept_key("vcc"), concept_key("vcc"))
        self.assertNotEqual(concept_key("vcc"), concept_key("vdd"))

    def test_build_concepts_merges_by_norm_embedding_and_judge(self) -> None:
        from kb_pipeline.graph.concepts import build_concepts, parse_answers

        facts = [
            {"id": "f1", "subject": "S", "property": "Supply voltage", "symbol": "V_CC", "value": "4.1", "unit": "V", "doc_id": "d1"},
            {"id": "f2", "subject": "S", "property": "VCC", "symbol": "", "value": "3.0", "unit": "V", "doc_id": "d2"},
            {"id": "f3", "subject": "S", "property": "Operating voltage", "symbol": "", "value": "4.1", "unit": "V", "doc_id": "d3"},
            {"id": "f4", "subject": "S", "property": "Clock frequency", "symbol": "f_CLK", "value": "100", "unit": "MHz", "doc_id": "d1"},
            {"id": "f5", "subject": "S", "property": "Clock speed", "symbol": "", "value": "133", "unit": "MHz", "doc_id": "d2"},
            {"id": "f6", "subject": "S", "property": "", "symbol": "", "value": "x"},
        ]
        vectors = {"Supply voltage (V_CC)": [1.0, 0.0, 0.0], "Operating voltage": [0.9, 0.1, 0.0],       # cosine ≈ 0.99: auto-merged
                   "Clock frequency (f_CLK)": [0.0, 1.0, 0.0], "Clock speed": [0.0, 0.85, 0.35]}        # cosine ≈ 0.92: ask the model

        def embed(texts):
            return [vectors[t] for t in texts]

        asked: list[str] = []

        class Judge:
            def chat(self, prompt, max_tokens=0):
                asked.append(prompt)
                return "1. yes"

        concepts, stats = build_concepts(facts, embed=embed, client=Judge(), auto_threshold=0.95, ask_threshold=0.82)
        by_key = {f["id"]: f.get("concept_key") for f in facts}
        self.assertEqual(by_key["f1"], by_key["f2"])              # merged directly by normalisation (symbol first)
        self.assertEqual(by_key["f1"], by_key["f3"])              # vector neighbour auto-merged
        self.assertEqual(by_key["f4"], by_key["f5"])              # judged the same by the model
        self.assertIsNone(by_key["f6"])
        self.assertEqual(len(asked), 1)
        self.assertIn("Clock frequency", asked[0])
        self.assertEqual((stats["norms"], stats["concepts"], stats["auto_merged"], stats["judged_yes"]), (4, 2, 1, 1))
        self.assertEqual(stats["cross_doc_concepts"], 2)
        top = concepts[0]
        self.assertEqual(top["facts"], 3)
        self.assertEqual(top["label"], "Supply voltage")
        self.assertEqual(facts[0]["concept"], "Supply voltage")
        self.assertIn("V_CC", top["symbols"])
        self.assertEqual(parse_answers("1. yes\n2: no\n3) 是", 3), [True, False, True])
        # No embeddings / no model: merge by normalisation only, without an error
        for f in facts:
            f.pop("concept_key", None)
        concepts2, stats2 = build_concepts(facts)
        self.assertEqual(stats2["concepts"], 4)
        self.assertEqual({f.get("concept_key") for f in facts[:2]}.__len__(), 1)

    def test_contrast_markers_and_bare_labels_never_auto_merge(self) -> None:
        """Synthetic contrast pairs exercise short-token distinctions and identity tokens.
        Similarity alone must not merge distinct labels; a label with an identity token
        must not merge with a bare generic label."""
        from kb_pipeline.graph.concepts import _contrast, build_concepts, compatible

        for a, b in (("矫正视力(右)", "矫正视力(左)"), ("胃蛋白酶原I(T-12)", "胃蛋白酶原II(T-12)"), ("静脉阻塞 类似", "动脉阻塞 类似"),
                     ("糖网轻度非增 类似", "糖网重度非增 类似"), ("j = j1 + 1/2 的CG系数", "j = j1 - 1/2 的CG系数"), ("v7 2407b", "v7 2407"),
                     ("ALT", "AST")):
            self.assertTrue(_contrast(a, b), (a, b))
        for a, b in (("Supply voltage", "Operating voltage"), ("文件名称", "文件名"), ("在线状态", "在线状态显示"), ("总胆固醇", "总胆固醇"),
                     ("前列腺癌易感风险", "前列腺癌易感风险建议")):
            self.assertFalse(_contrast(a, b), (a, b))
        cand = lambda label, sym="": {"label": label, "symbols_text": sym, "unit": ""}
        self.assertEqual(compatible(cand("矫正视力(右)"), cand("矫正视力(左)")), "contrast")
        # The synthetic coefficient cases differ by more than one token and are blocked by their differing
        # number sequences (j=j1+1/2 versus j=j1,m2=1)
        self.assertEqual(compatible(cand("j = j1 + 1/2, m2 = 1/2 的CG系数"), cand("j = j1, m2 = 1 的CG系数")), "contrast")
        self.assertEqual(compatible(cand("量子力学(第2版)"), cand("量子力学(第3版)")), "contrast")
        self.assertEqual(compatible(cand("F分布 分位数 0.05"), cand("F-分布 分位数 0.05")), "ok")
        self.assertEqual(compatible(cand("基因型"), cand("GENEX rs0000133 基因型")), "identity")
        self.assertEqual(compatible(cand("在线状态"), cand("在线状态显示")), "ok")
        # Full flow: the right / left eye pair at cosine 0.97 is no longer auto-merged; without a model they stay two
        # concepts, with a model it decides
        facts = [{"id": "r", "subject": "S", "property": "矫正视力(右)", "symbol": "", "value": "1.0", "doc_id": "d1"},
                 {"id": "l", "subject": "S", "property": "矫正视力(左)", "symbol": "", "value": "0.8", "doc_id": "d1"}]
        vectors = {"矫正视力(右)": [1.0, 0.0], "矫正视力(左)": [0.97, 0.243]}
        _, stats = build_concepts(facts, embed=lambda texts: [vectors[t] for t in texts], auto_threshold=0.95, ask_threshold=0.82)
        self.assertNotEqual(facts[0]["concept_key"], facts[1]["concept_key"])
        self.assertEqual((stats["auto_merged"], stats["contrast_to_judge"], stats["skipped_llm"]), (0, 1, True))
        asked: list[str] = []

        class Judge:
            def chat(self, prompt, max_tokens=0):
                asked.append(prompt)
                return "1. no"
        for f in facts:
            f.pop("concept_key", None)
        _, stats = build_concepts(facts, embed=lambda texts: [vectors[t] for t in texts], client=Judge(), auto_threshold=0.95, ask_threshold=0.82)
        self.assertEqual((len(asked), stats["judged_yes"]), (1, 0))
        self.assertNotEqual(facts[0]["concept_key"], facts[1]["concept_key"])

    def test_shared_global_subjects_keep_their_facts_under_a_path_subject(self) -> None:
        """Synthetic multi-subject documents: the top-level directory (the vendor name) matched an entity, and every
        fact in the per-product columns of the feature comparison table was re-attributed to the vendor. Generic rule:
        path / series evidence only re-attributes facts filed under the document's own subject entities (scoped ones,
        or ones appearing only in this document); facts under cross-document global entities stay put."""
        from kb_pipeline.graph.facts import normalize_measurements

        ents = [{"key": "vendor", "title": "示例办公", "type": "product", "upper": "entity", "frequency": 50, "doc_ids": ["d1", "d2", "d3", "d4"], "scope": ""},
                {"key": "demo-collab", "title": "示例协作", "type": "product", "upper": "entity", "frequency": 40, "doc_ids": ["d1", "d2"], "scope": ""},
                {"key": "rival", "title": "移动办公", "type": "product", "upper": "entity", "frequency": 30, "doc_ids": ["d1", "d2"], "scope": ""},
                {"key": "d1::定制工作台", "title": "定制工作台", "type": "product_module", "upper": "part", "frequency": 5, "doc_ids": ["d1"], "scope": "d1"},
                {"key": "own", "title": "某某_编号0001", "type": "product", "upper": "entity", "frequency": 3, "doc_ids": ["d4"], "scope": ""}]
        docs = {"d1": "示例办公/竞品对比/功能对比.xlsx", "d2": "示例办公/私网/功能清单.xlsx", "d3": "示例办公/白皮书.pdf", "d4": "示例办公/单主体报告.pdf"}
        units = {"u1": _unit("u1", doc="d1"), "u4": _unit("u4", doc="d4")}
        facts = [{"id": "a", "unit_id": "u1", "doc_id": "d1", "subject": "示例协作", "property": "接单机制", "value": "支持", "value_num": None, "unit": "", "conditions": {}},
                 {"id": "b", "unit_id": "u1", "doc_id": "d1", "subject": "移动办公", "property": "接单机制", "value": "不支持", "value_num": None, "unit": "", "conditions": {}},
                 {"id": "m", "unit_id": "u1", "doc_id": "d1", "subject": "定制工作台", "property": "自定义字段", "value": "支持", "value_num": None, "unit": "", "conditions": {}},
                 {"id": "c", "unit_id": "u4", "doc_id": "d4", "subject": "某某_编号0001", "property": "版本", "value": "v7", "value_num": None, "unit": "", "conditions": {}}]
        stats = normalize_measurements(facts, ents, units_by_id=units, profile={"subject_types": ["product", "product_module"], "axis": "version"}, documents=docs)
        # The comparison table (d1, several subjects and components) is left alone; in the single-subject document (d4)
        # the facts under the ID-style subject go to the path subject
        self.assertEqual([(f["subject"], f.get("subject_raw")) for f in facts],
                         [("示例协作", None), ("移动办公", None), ("定制工作台", None), ("示例办公", "某某_编号0001")])
        self.assertEqual((stats["resubjected_identity"], stats["identity_kept_shared"]), (1, 3))

    def test_entity_name_gate_and_prompt_example_grounding(self) -> None:
        """Formula fragments, SQL statements and whole questions are not entity names.
        Synthetic schema examples must not enter a graph unless the source contains them."""
        from kb_pipeline.graph.extract import entity_name_ok, ground_records, parse_records, prompt_example_names

        for bad in ("x = y + 1", "SELECT * FROM fruits;", "app:", "Corn Prod,", "(hat a dagger)", "fr(q)", "请描述一例最能代表当前问题的演示案例。?",
                    "发现示例数据过期时,目前如何处理?", "langle n | hat q | n prime rangle", "AI 项目方案组 (202...", "", "···"):
            self.assertFalse(entity_name_ok(bad), bad)
        for good in ("肝", "A", "σ", "示例+365", "A/B测试", "CPI(UNME)", "Skill(技能)", "RV5/SV1", "C++", "Johnson & Johnson", "v7.0.2412a"):
            self.assertTrue(entity_name_ok(good), good)
        ents, rels, stats = parse_records('("entity"<|>SELECT 1;<|>ORGANIZATION<|>x)\n##\n("entity"<|>肝<|>ORGANIZATION<|>y)\n##\n'
                                          '("relationship"<|>肝<|>x = 1<|>related_to<|>d<|>3)')
        self.assertEqual(([e["name"] for e in ents], rels, stats["bad_names"]), (["肝"], [], 2))
        schema = ExtractionSchema(entity_types=("product", "feature"),
                                  type_definitions={"product": "A named commercial offering such as 示例 365, 示例协作, or 示例社区; not a module.",
                                                    "feature": "A capability like 邮箱搬家 or 轻打卡."},
                                  examples='("entity"<|>Prometheus<|>tool<|>monitoring)')
        self.assertEqual(prompt_example_names(schema), {"示例365", "示例协作", "示例社区", "邮箱搬家", "轻打卡", "prometheus"})
        entities = [{"name": "示例社区", "type": "product", "description": "示例办公旗下的产品"},
                    {"name": "示例 365", "type": "product", "description": "套件"},
                    {"name": "示例PC客户端(Win&Linux&Mac)", "type": "feature", "description": "客户端"},
                    {"name": "文档中心", "type": "product", "description": "本单元真正讲的东西"}]
        relations = [{"source": "示例社区", "target": "文档中心", "predicate": "related_to"},
                     {"source": "示例 365", "target": "文档中心", "predicate": "has_feature"}]
        context = "示例 365 里的文档中心支持 示例PC客户端（Win&Linux&Mac）。"
        kept, rels2, dropped = ground_records(entities, relations, context=context, example_names=prompt_example_names(schema))
        self.assertEqual(([e["name"] for e in kept], len(rels2), dropped), (["示例 365", "示例PC客户端(Win&Linux&Mac)", "文档中心"], 1, 1))
        kept2, _, dropped2 = ground_records(entities, relations, context="示例社区是示例软件的知识库产品 " + context, example_names=prompt_example_names(schema))
        self.assertEqual((len(kept2), dropped2), (4, 0))                                     # the one that really appears in the text is left alone
        build = (Path(__file__).resolve().parents[1] / "kb_pipeline" / "graph" / "build.py").read_text(encoding="utf-8")
        self.assertIn("ents, rels, n = ground_records(ents, rels, context=context, example_names=example_names)", build)   # cached extractions also pass through it before merging
        self.assertIn('merged["stats"]["ungrounded_dropped"] = ungrounded_dropped', build)

    def test_symbol_names_are_document_scoped_and_year_parens_are_not_aliases(self) -> None:
        """Symbols are scoped to their documents. Two fictional citations sharing a year
        must remain distinct; a parenthesized year is not an alias."""
        from kb_pipeline.graph.merge import merge_extractions, scoped_key, symbol_like_name
        from kb_pipeline.graph.resolution import auto_merge_pairs, paren_alias

        for s in ("A", "r", "σ", "x0", "r1", "σ2"):
            self.assertTrue(symbol_like_name(s), s)
        for g in ("OA", "IM", "WW", "肝", "AI", "3D", "EXO", "π介子"):
            self.assertFalse(symbol_like_name(g), g)
        u1, u2 = _unit("u1", doc="kb_004:1"), _unit("u2", doc="kb_004:2")
        ext = {"u1": {"entities": [{"name": "A", "type": "concept", "description": "总资产"}, {"name": "OA", "type": "system", "description": "办公"}], "relations": []},
               "u2": {"entities": [{"name": "A", "type": "concept", "description": "振幅"}, {"name": "OA", "type": "system", "description": "办公"}], "relations": []}}
        out = merge_extractions([u1, u2], ext, entity_types=("concept", "system"), upper_parents={"concept": "entity", "system": "entity"})
        keys = {e["key"] for e in out["entities"]}
        self.assertEqual(keys, {scoped_key("a", "kb_004:1"), scoped_key("a", "kb_004:2"), "oa"})
        self.assertEqual(paren_alias("AuthorA和AuthorB(1998)"), ("", ""))
        self.assertEqual(paren_alias("体重指数(BMI)"), ("体重指数", "BMI"))
        rows = [{"key": "a", "title": "AuthorA和AuthorB(1998)", "type": "publication"}, {"key": "b", "title": "AuthorC(1998)", "type": "publication"}]
        self.assertEqual(auto_merge_pairs(rows), [])

    def test_prefix_and_initial_alias_candidates(self) -> None:
        """Synthetic vendor prefixes and initials create candidates for model judgment.
        Literal similarity alone cannot recover every shortened name."""
        from kb_pipeline.graph.resolution import _initials, candidate_pairs

        self.assertEqual((_initials("WorkWidget"), _initials("Data Systems Engineer"), _initials("WW"), _initials("示例社区")), ("WW", "DSE", "", ""))
        ents = [{"title": "EXO", "type": "product"}, {"title": "EXO Gadget", "type": "product"}, {"title": "Gadget", "type": "product"},
                {"title": "示例软件", "type": "organization"}, {"title": "示例软件示例社区", "type": "product"}, {"title": "示例社区", "type": "product"},
                {"title": "WorkWidget", "type": "product"}, {"title": "WW", "type": "product"}, {"title": "文档中心", "type": "product"},
                {"title": "Office", "type": "product"}]
        pairs = candidate_pairs(ents)
        self.assertIn((1, 2), pairs)           # EXO Gadget / Gadget
        self.assertIn((4, 5), pairs)           # Example Vendor Knowledge Community / Knowledge Community
        self.assertIn((6, 7), pairs)           # WorkWidget / WW
        self.assertNotIn((0, 9), pairs)        # EXO / Office is not a prefix relation
        # Scoped entities never enter these two candidate classes
        scoped = [{"title": "EXO Gadget", "type": "product", "scope": "d1"}, {"title": "Gadget", "type": "product", "scope": "d1"}]
        self.assertEqual(candidate_pairs(scoped), [])

    def test_declared_alias_in_description_merges_without_the_judge(self) -> None:
        """An explicitly declared alias merges directly. Synthetic descriptions test
        definition-first selection, Chinese parentheses and the longest-text fallback."""
        from kb_pipeline.graph.resolution import _first_sentence, auto_merge_pairs, declared_alias

        self.assertEqual(declared_alias("WW", ["WW（WorkWidget）是示例软件开发的协作工具"]), "WorkWidget")
        self.assertEqual(declared_alias("DSE", ["DSE (Data Systems Engineer) 是一种岗位"]), "Data Systems Engineer")
        self.assertEqual(declared_alias("WW", ["WorkWidget(WW)是…"]), "")                 # the parenthesis does not follow its own name: does not count
        self.assertEqual(declared_alias("AuthorA和AuthorB", ["AuthorA和AuthorB(1998)提出…"]), "")   # a year is not an alias
        self.assertEqual(declared_alias("WW", ["WW(WW)…"]), "")
        # A description opening with "short for X" (or its Chinese equivalents) is a declared alias too.
        self.assertEqual(declared_alias("WW", ["WorkWidget的简称，用于示例日程管理"]), "WorkWidget")
        self.assertEqual(declared_alias("DSE", ["Short for Data Systems Engineer, a role"]), "Data Systems Engineer")
        self.assertEqual(declared_alias("WW", ["一款示例协作工具，是WorkWidget的简称"]), "")     # not at the start: does not count (avoids false catches)
        self.assertEqual(declared_alias("EXO文档中台系统V7", ["对客产品全称/软著全称，对应专业版与增强版"]), "")   # no possessive particle: a column heading, not a self-description
        # Two entities each claiming to be short for the same non-existent name: not merged with each other
        ghosts = [{"key": "a", "title": "系统A V7", "type": "product", "descriptions": ["对客产品的全称，专业版"]},
                  {"key": "b", "title": "系统B V7", "type": "product", "descriptions": ["对客产品的全称，增强版"]}]
        self.assertEqual(auto_merge_pairs(ghosts), [])
        ents = [{"key": "workwidget", "title": "WorkWidget", "type": "product", "descriptions": ["一款用于示例流程的协作工具", "示例软件的协作工具"]},
                {"key": "ww", "title": "WW", "type": "product", "descriptions": ["WW（WorkWidget）是示例软件开发的协作工具，用于整理虚构任务。它也支持示例日程。"]},
                {"key": "ww2", "title": "W.W.", "type": "product", "descriptions": ["WorkWidget的简称"]},
                {"key": "示例软件", "title": "示例软件", "type": "organization", "descriptions": ["示例软件（WW 的厂商）"]}]
        reasons: dict = {}
        self.assertEqual(auto_merge_pairs(ents, reasons), [(0, 1), (0, 2), (1, 2)])      # (1, 2): WW / W.W. are equal identifiers, merged anyway
        self.assertEqual((reasons[(0, 1)], reasons[(0, 2)], reasons[(1, 2)]), ("declared", "declared", "identifier"))
        self.assertTrue(_first_sentence(ents[1]).startswith("WW（WorkWidget）是示例软件"))     # given verbatim (full-width kept), the definition-style one first
        long_row = {"title": "WorkWidget", "descriptions": ["短。", "一款用于示例流程的协作工具，包含任务、日程和列表功能，供团队成员在示例流程中协作。第二句。第三句。"]}
        self.assertTrue(_first_sentence(long_row).startswith("一款用于示例流程的协作工具"))    # no definition-style description: take the longest

    def test_reconcile_marks_series_and_real_conflicts_only(self) -> None:
        from kb_pipeline.graph.reconcile import reconcile_facts, series_of

        def fact(fid, doc, axis, value, *, prop="总胆固醇", unit="mmol/L", unit_id=None, conditions=None):
            return {"id": fid, "subject": "张三", "subject_key": "zhangsan", "property": prop, "symbol": "", "value": value,
                    "value_num": float(value), "unit": unit, "doc_id": doc, "unit_id": unit_id or f"u-{fid}",
                    "valid_from": axis, "conditions": conditions or {}}

        facts = [
            fact("a", "d1", "2024-03-01", "4.6"), fact("b", "d2", "2025-03-01", "5.4"), fact("c", "d3", "2026-03-01", "6.1"),
            fact("x", "d4", "2025-03-01", "5.7"),                                                 # same axis, different value, different document: conflict
            fact("p", "d5", "2025-01-01", "4.1", prop="血糖", unit="mmol/L"), fact("q", "d5", "2025-01-01", "4.7", prop="血糖", unit="mmol/L", unit_id="same"),
            fact("r", "d6", "", "1"), fact("s", "d7", "", "1"),                                    # same value: not a conflict
        ]
        facts[4]["unit_id"] = "same"                                                              # two values in the same unit: not a cross-document conflict
        out = reconcile_facts(facts)
        by = {f["id"]: f for f in facts}
        self.assertEqual((by["a"]["series_index"], by["b"]["series_index"], by["c"]["series_index"]), (0, 1, 2))
        self.assertEqual(by["a"]["series_len"], 3)
        self.assertEqual(by["x"]["series_key"], by["a"]["series_key"])
        self.assertEqual(by["b"]["conflict_group"], by["x"]["conflict_group"])
        self.assertNotIn("conflict_group", by["a"])
        self.assertNotIn("conflict_group", by["p"])
        self.assertNotIn("conflict_group", by["r"])
        self.assertEqual(out["stats"]["series"], 1)
        self.assertEqual(out["stats"]["conflicts"], 1)
        # Two values in different passages of the same document (two heart rates in a check-up) are not a cross-document
        # conflict, only counted
        same_doc = [fact("m", "d9", "2021-03-14", "74", prop="心率", unit="bpm", unit_id="u-m1"), fact("n", "d9", "2021-03-14", "68", prop="心率", unit="bpm", unit_id="u-m2")]
        out2 = reconcile_facts(same_doc)
        self.assertEqual((out2["stats"]["conflicts"], out2["stats"]["same_document_variants"]), (0, 1))
        self.assertNotIn("conflict_group", same_doc[0])
        self.assertEqual(out["conflicts"][0]["axis"], "2025-03-01")
        self.assertEqual(sorted(v["value"] for v in out["conflicts"][0]["values"]), ["5.4", "5.7"])
        series = series_of(facts)
        self.assertEqual([f["id"] for f in series[by["a"]["series_key"]]], ["a", "b", "x", "c"])
        # A unit-less flag fact joins the measurement series that has a unit; only two real units split into two groups
        from kb_pipeline.graph.reconcile import split_by_unit
        rows = [fact("m1", "d1", "2024-01-01", "4.6"), fact("m2", "d2", "2025-01-01", "5.4"),
                {"id": "m3", "subject": "张三", "subject_key": "zhangsan", "property": "总胆固醇", "symbol": "", "value": "增高", "unit": "",
                 "doc_id": "d3", "unit_id": "u-m3", "valid_from": "2026-01-01", "conditions": {}, "flag": "↑"},
                {"id": "m4", "subject": "张三", "subject_key": "zhangsan", "property": "总胆固醇", "symbol": "", "value": "228", "value_num": 228.0,
                 "unit": "mg/dL", "doc_id": "d4", "unit_id": "u-m4", "valid_from": "2023-01-01", "conditions": {}}]
        parts = split_by_unit(rows)
        self.assertEqual([(u, [r["id"] for r in part]) for u, part in parts], [("mmol/L", ["m1", "m2", "m3"]), ("mg/dL", ["m4"])])
        out2 = reconcile_facts(rows)
        self.assertEqual((rows[0]["series_key"], rows[2]["series_key"], rows[2]["series_index"], rows[0]["series_len"]), (rows[1]["series_key"], rows[0]["series_key"], 2, 3))
        self.assertNotIn("series_key", rows[3])
        self.assertEqual(out2["stats"]["conflicts"], 0)
        same_axis = [fact("n1", "d1", "2025-01-01", "5.4"),
                     {"id": "n2", "subject": "张三", "subject_key": "zhangsan", "property": "总胆固醇", "symbol": "", "value": "增高", "unit": "",
                      "doc_id": "d1", "unit_id": "u-n2", "valid_from": "2025-01-01", "conditions": {}, "flag": "↑"}]
        self.assertEqual(reconcile_facts(same_axis)["stats"]["conflicts"], 0)     # on the same axis "5.4 mmol/L" and "elevated" are two spellings of one measurement

    def test_date_and_version_axes_never_share_a_series(self) -> None:
        """In a KB whose profile is version both kinds of axis value coexist: documents without a version number get
        a date axis value, and validity periods the model copies from the text are dates too. Dates used to sort
        before versions, so "2023-10-31 -> v6.0" became the start and end of one series. Now dates and versions are
        sorted separately."""
        from kb_pipeline.graph.reconcile import reconcile_facts, series_of
        from kb_pipeline.graph.temporal import axis_value_kind

        self.assertEqual([axis_value_kind(v) for v in ("2024", "2024-07", "2024-07-26", "v6.0", "rev *J", "v2024.1", "")],
                         ["date", "date", "date", "version", "version", "version", ""])

        def fact(fid, doc, value, *, valid_from="", axis=""):
            row = {"id": fid, "subject": "某平台", "subject_key": "platform", "property": "支撑用户量", "symbol": "", "value": value,
                   "value_num": float(value), "unit": "万", "doc_id": doc, "unit_id": f"u-{fid}", "conditions": {}}
            if valid_from:
                row["valid_from"] = valid_from
            if axis:
                row["axis"] = axis
            return row

        # one date and one version: the two points have no order between them, so they form no series
        pair = [fact("a", "d1", "50", valid_from="2023-10-31", axis="v6.0"), fact("b", "d1", "80", valid_from="v6.0", axis="v6.0")]
        out = reconcile_facts(pair)
        self.assertEqual(out["stats"]["series"], 0)
        self.assertEqual([("series_key" in f) for f in pair], [False, False])
        self.assertEqual(series_of(pair), {})
        # two dates, two versions and one without an axis: one series per kind, each indexed from 0
        rows = [fact("d1", "d1", "50", valid_from="2023-10-31"), fact("v2", "d2", "90", axis="v6.10"), fact("d2", "d3", "60", valid_from="2024-05"),
                fact("v1", "d4", "80", axis="v6.2"), fact("n", "d5", "70")]
        out = reconcile_facts(rows)
        by = {f["id"]: f for f in rows}
        self.assertEqual(out["stats"]["series"], 2)
        self.assertEqual(by["d1"]["series_key"], by["d2"]["series_key"])
        self.assertEqual(by["v1"]["series_key"], by["v2"]["series_key"])
        self.assertNotEqual(by["d1"]["series_key"], by["v1"]["series_key"])
        self.assertEqual([(by[k]["series_index"], by[k]["series_len"]) for k in ("d1", "d2", "v1", "v2")], [(0, 2), (1, 2), (0, 2), (1, 2)])
        self.assertNotIn("series_key", by["n"])
        self.assertEqual(sorted([f["id"] for f in part] for part in series_of(rows).values()), [["d1", "d2"], ["v1", "v2"]])

    def test_spec_vectors_are_three_named_vectors_and_pages_are_optional(self) -> None:
        from kb_pipeline.graph.vectors import page_payload, spec_embed_texts, write_graph_vectors
        from kb_pipeline.vector.qdrant import GRAPH_OPTIONAL_TYPES, GRAPH_VECTOR_TYPES, graph_vector_layout

        self.assertEqual(GRAPH_VECTOR_TYPES, ("entity", "relation", "spec", "page"))
        self.assertEqual(GRAPH_OPTIONAL_TYPES, frozenset({"spec", "page"}))
        self.assertEqual(graph_vector_layout("spec"), ("text", "property", "value"))
        self.assertIsNone(graph_vector_layout("entity"))
        f = {"id": "f1", "subject": "IC", "property": "Supply voltage", "symbol": "VCC", "concept": "Supply voltage", "min": "3.0", "max": "3.6", "unit": "V",
             "conditions": {}, "flag": "", "valid_from": "2024-01-01", "axis": "2024-01-01"}
        texts = spec_embed_texts(f)
        self.assertEqual(set(texts), {"text", "property", "value"})
        self.assertIn("Supply voltage", texts["property"])
        self.assertIn("3.0", texts["value"])
        created: dict[str, object] = {}
        upserts: dict[str, list] = {}

        class FakeQ:
            def get_collections(self):
                return SimpleNamespace(collections=[SimpleNamespace(name=n) for n in created])

            def create_collection(self, collection_name, vectors_config):
                created[collection_name] = vectors_config

            def get_collection(self, collection_name):
                return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=3))))

            def create_payload_index(self, **kw):
                pass

            def upsert(self, collection_name, points, wait):
                upserts.setdefault(collection_name, []).extend(points)

            def scroll(self, collection_name, limit, offset, with_payload, with_vectors):
                return list(upserts.get(collection_name, [])), None

            def delete(self, collection_name, points_selector, wait):
                pass

            def count(self, collection_name, exact):
                return SimpleNamespace(count=len(upserts.get(collection_name, [])))

        embedded: list[str] = []

        def embed(texts):
            embedded.extend(texts)
            return [[0.1, 0.2, 0.3] for _ in texts]

        bundle = {"entities": [], "relations": [], "specs": [f], "mentions": [], "pages": [], "stats": {}}
        summary = write_graph_vectors(FakeQ(), embed, kb_id="kb_003", source_collection="kb_003", graph_version="v1",
                                      bundle=bundle, units_by_id={}, vector_size=3)
        self.assertIsInstance(created["graph_003_spec__v1"], dict)
        self.assertEqual(set(created["graph_003_spec__v1"]), {"text", "property", "value"})
        self.assertNotIn("graph_003_page__v1", created)                        # no pages: no empty collection is created
        self.assertNotIn("page", summary["collections"])
        point = upserts["graph_003_spec__v1"][0]
        self.assertEqual(set(point.vector), {"text", "property", "value"})
        self.assertEqual(point.payload["concept"], "Supply voltage")
        self.assertEqual(point.payload["valid_from"], "2024-01-01")
        self.assertEqual(len(embedded), 3)
        page = {"id": "pg1", "kind": "subject", "title": "IC", "summary": "s", "text": "body", "entity_keys": ["ic"], "doc_ids": ["d1"],
                "point_ids": ["p1"], "spec_ids": ["f1"], "concept_keys": ["c1"], "path": "wiki/subjects/ic.md"}
        pl = page_payload(page, kb_id="kb_003", source_collection="kb_003", graph_version="v1")
        self.assertEqual((pl["graph_type"], pl["kind"], pl["gr_id"]), ("page", "subject", "pg1"))
        bundle["pages"] = [page]
        summary = write_graph_vectors(FakeQ(), embed, kb_id="kb_003", source_collection="kb_003", graph_version="v2",
                                      bundle=bundle, units_by_id={}, vector_size=3)
        self.assertEqual(summary["collections"]["page"]["points"], 1)
        self.assertIn("graph_003_page__v2", created)

    def test_ensure_graph_collection_checks_named_layout_of_existing_collections(self) -> None:
        from kb_pipeline.vector.qdrant import ensure_graph_collection

        class FakeQ:
            def __init__(self, vectors):
                self.vectors = vectors

            def get_collections(self):
                return SimpleNamespace(collections=[SimpleNamespace(name="graph_003_spec__v1")])

            def get_collection(self, collection_name):
                return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=self.vectors)))

            def create_payload_index(self, **kw):
                pass

        with self.assertRaises(RuntimeError):
            ensure_graph_collection(FakeQ(SimpleNamespace(size=4)), "graph_003_spec__v1", 4, layout=("text", "property", "value"))
        named = {n: SimpleNamespace(size=4) for n in ("text", "property", "value")}
        self.assertTrue(ensure_graph_collection(FakeQ(named), "graph_003_spec__v1", 4, layout=("text", "property", "value")).startswith("exists"))

    def test_preflight_and_alias_activation_tolerate_a_missing_optional_collection(self) -> None:
        from kb_pipeline.graph.neo4j_import import bundle_type_counts
        from kb_pipeline.vector.qdrant import activate_graph_aliases

        self.assertEqual(bundle_type_counts({"pages": [1]}), {"entity": 0, "relation": 0, "spec": 0, "page": 1})
        ops: list = []

        class FakeQ:
            def get_aliases(self):
                return SimpleNamespace(aliases=[])

            def collection_exists(self, name):
                return not name.startswith("graph_003_page")

            def update_collection_aliases(self, operations):
                ops.extend(operations)

        out = activate_graph_aliases(FakeQ(), source_collection="kb_003", graph_version="v1")
        self.assertEqual(out["skipped"], ["graph_003_page"])
        self.assertEqual(set(out["targets"]), {"graph_003_entity", "graph_003_relation", "graph_003_spec"})
        self.assertEqual(len(ops), 3)

        class MissingEntity(FakeQ):
            def collection_exists(self, name):
                return not name.startswith("graph_003_entity")

        with self.assertRaises(RuntimeError):
            activate_graph_aliases(MissingEntity(), source_collection="kb_003", graph_version="v1")

    def test_spec_seeds_merge_named_vectors_apply_the_window_and_cover_the_question(self) -> None:
        from kb_pipeline.graph.recall import _query_points, _spec_dense_rows, greedy_cover, spec_seeds

        calls: list[str | None] = []

        class NamedQ:
            def query_points(self, collection_name, query, limit, with_payload, query_filter=None, using=None):
                calls.append(using)
                rows = {
                    "text": [("f1", 0.70), ("f2", 0.60)],
                    "property": [("f2", 0.80), ("f3", 0.50)],
                    "value": [("f2", 0.75)],
                }[using]
                pl = {"f1": {"gr_id": "f1", "subject": "A", "property": "VCC", "valid_from": "2023-05-01"},
                      "f2": {"gr_id": "f2", "subject": "A", "property": "VCC", "valid_from": "2025-05-01"},
                      "f3": {"gr_id": "f3", "subject": "B", "property": "ICC", "valid_from": ""}}
                return SimpleNamespace(points=[SimpleNamespace(id=k, score=s, payload=pl[k]) for k, s in rows])

            def scroll(self, collection_name, scroll_filter, limit, with_payload, with_vectors):
                return [], None

        rows = _spec_dense_rows(NamedQ(), "graph_003_spec", [0.1], 5)
        self.assertEqual(calls, ["text", "property", "value"])
        by = {r["gr_id"]: r for r in rows}
        self.assertAlmostEqual(by["f2"]["_score"], 0.85)                       # top score 0.80 + 0.05 for hits on both the property and the value side
        self.assertAlmostEqual(by["f1"]["_score"], 0.70)
        self.assertEqual(by["f3"]["_via"], "vector")

        class LegacyQ:
            def query_points(self, collection_name, query, limit, with_payload, query_filter=None):
                return SimpleNamespace(points=[SimpleNamespace(id="f9", score=0.5, payload={"gr_id": "f9"})])

        self.assertEqual([r["gr_id"] for r in _spec_dense_rows(LegacyQ(), "c", [0.1], 5)], ["f9"])    # old single-vector graph: falls back to the unnamed vector
        self.assertEqual(_query_points(LegacyQ(), "c", [0.1], 5)[0]["_score"], 0.5)

        seeds = spec_seeds(NamedQ(), "graph_003_spec", [0.1], "A 这两年的 VCC 怎么样", window={"from": "2024", "to": "2026"})
        by = {r["gr_id"]: r for r in seeds}
        self.assertGreater(by["f2"]["_score"], by["f1"]["_score"])
        self.assertLess(by["f1"]["_score"], 0.70 + 0.05 * by["f1"]["_hits"])    # discounted outside the window
        self.assertGreater(by["f3"]["_score"], 0.5)                           # facts without an axis value are not discounted
        covered = greedy_cover([
            {"gr_id": "x", "_score": 0.9, "subject": "A", "property": "VCC"},
            {"gr_id": "y", "_score": 0.8, "subject": "A", "property": "ICC"},
            {"gr_id": "z", "_score": 0.7, "subject": "B", "property": "tCK"},
        ], "A 的 ICC 和 B 的 tCK", limit=3)
        self.assertEqual([c["gr_id"] for c in covered][:2], ["y", "z"])       # cover the identifiers in the question first, then go by score

    def test_page_seeds_and_collections_include_the_view_layer(self) -> None:
        from kb_pipeline.graph.recall import graph_collections_for, page_seeds

        self.assertEqual(graph_collections_for("kb_004")["page"], "graph_004_page")

        class FakeQ:
            def query_points(self, collection_name, query, limit, with_payload, query_filter=None, using=None):
                return SimpleNamespace(points=[SimpleNamespace(id="pg", score=0.7, payload={"gr_id": "pg1", "kind": "timeline", "title": "张三 · 总胆固醇",
                                                                                            "text": "总胆固醇 2024 4.6 2025 5.4", "point_ids": ["p1"]})])

        rows = page_seeds(FakeQ(), "graph_004_page", [0.1], "张三 总胆固醇 变化")
        self.assertEqual(rows[0]["gr_id"], "pg1")
        self.assertGreater(rows[0]["_score"], 0.7)
        self.assertEqual(page_seeds(SimpleNamespace(query_points=lambda **kw: (_ for _ in ()).throw(RuntimeError("missing"))), "x", [0.1], "q"), [])

    def test_recall_source_boosts_conclusions_and_keeps_in_network_relations(self) -> None:
        _repo_file = lambda rel: (Path(__file__).resolve().parents[2] / rel).read_text(encoding="utf-8")
        src = _repo_file("app/kb_pipeline/graph/recall.py")
        self.assertIn("CONCLUSION_BOOST", src)
        self.assertIn("coalesce(c.kind, 'body') AS kind", src)
        self.assertIn("in_network = nid in seed_ids and hop == 1", src)
        self.assertIn("question_time_window(question)", src)
        self.assertIn('"pages": page_rows', src)
        build_src = _repo_file("app/kb_pipeline/graph/build.py")
        self.assertIn("build_concepts(all_facts", build_src)
        self.assertIn("reconcile_facts(all_facts)", build_src)
        self.assertIn('graph["conflicts"]', build_src)


class ViewLayerTests(unittest.TestCase):
    """Third batch of the 2026-09-07 plan (view layer): the four page kinds subject / timeline / source / index are
    deterministic projections of the structure layer, written to the workspace wiki/ and into the page vector
    collection; subject-page narration is written by the summary model, falling back to the deterministic summary
    on failure."""

    def _graph(self):
        def ent(key, title, etype, parent, upper, **kw):
            row = {"key": key, "title": title, "type": etype, "parent_type": parent, "upper": upper, "description": kw.pop("description", ""),
                   "frequency": kw.pop("frequency", 1), "degree": 1, "pagerank": kw.pop("pagerank", 0.1), "doc_ids": kw.pop("doc_ids", ["d1"]),
                   "aliases": kw.pop("aliases", []), "unit_ids": ["u1"]}
            row.update(kw)
            return row

        def fact(fid, doc, when, value, *, prop="总胆固醇", ckey="c1", flag="", series=None, unit="mmol/L", subject_key="zhangsan"):
            return {"id": fid, "subject": "张三", "subject_key": subject_key, "subject_keys": [subject_key], "property": prop, "symbol": "",
                    "concept": prop, "concept_key": ckey, "value": value, "value_num": float(value), "unit": unit, "doc_id": doc,
                    "rel_path": f"{doc}.pdf", "section": "检验", "valid_from": when, "axis": when, "point_ids": [f"p-{doc}"],
                    "flag": flag, "ref_min": "0", "ref_max": "5.2", "conditions": {}, **(series or {})}

        entities = [
            ent("zhangsan", "张三", "person", "entity", "entity", description="被检者,男。", frequency=9, pagerank=0.4, doc_ids=["d1", "d2"], aliases=["Zhang San"]),
            ent("tc", "总胆固醇", "indicator", "attribute", "attribute", frequency=4, doc_ids=["d1", "d2"]),
            ent("toc", "目录", "section", "entity", "entity", boilerplate=True),
            ent("hospital", "某医院", "organization", "entity", "entity", frequency=2, doc_ids=["d2"]),
        ]
        relations = [
            {"source_key": "zhangsan", "target_key": "tc", "source": "张三", "target": "总胆固醇", "predicate": "indicates", "description": "总胆固醇偏高", "weight": 6.1},
            {"source_key": "hospital", "target_key": "zhangsan", "source": "某医院", "target": "张三", "predicate": "related_to", "description": "", "weight": 2.0},
            {"source_key": "toc", "target_key": "zhangsan", "predicate": "related_to", "description": "", "weight": 1.0, "boilerplate": True},
        ]
        facts = [
            fact("f1", "d1", "2024-03-01", "4.6", series={"series_key": "s1", "series_index": 0, "series_len": 2}),
            fact("f2", "d2", "2025-03-01", "5.4", flag="↑", series={"series_key": "s1", "series_index": 1, "series_len": 2}),
            fact("f3", "d2", "2025-03-01", "4.7", prop="血糖", ckey="c2"),
        ]
        graph = {
            "kb_id": "kb_002", "entities": entities, "relations": relations, "specs": facts,
            "concepts": [{"key": "c1", "label": "总胆固醇", "facts": 2, "docs": ["d1", "d2"]}, {"key": "c2", "label": "血糖", "facts": 1, "docs": ["d2"]}],
            "conflicts": [{"key": "x1", "subject": "张三", "concept": "血糖", "axis": "2025-03-01", "values": [
                {"value": "4.7", "doc_id": "d2", "rel_path": "d2.pdf"}, {"value": "5.0", "doc_id": "d3", "rel_path": "d3.pdf"}]}],
            "documents": {"d1": {"rel_path": "d1.pdf", "kind": "date", "value": "2024-03-01"}, "d2": {"rel_path": "d2.pdf", "kind": "date", "value": "2025-03-01"}},
            "profile": {"subject_types": ["person"], "axis": "date", "conclusion_headings": [], "extension_predicates": ["indicates"]},
            "mentions": [{"entity_key": "zhangsan", "point_id": "p-d1", "count": 2}, {"entity_key": "zhangsan", "point_id": "p-zero", "count": 0}],
            "unit_kinds": {"u2": "conclusion"},
        }
        u1 = _unit("u1", doc="d1", points=("p-d1",))
        u2 = _unit("u2", doc="d2", points=("p-d2",), order=1)
        u2.text = "异常结果汇总:总胆固醇偏高,建议复查。"
        return graph, [u1, u2]

    def test_symbol_only_facts_stay_in_the_page_but_not_in_the_narration_input(self) -> None:
        """A spot check found an assessment marked "-" for both eyes narrated as a positive finding. Facts whose value is
        only a symbol stay in the page and in the specs but not in the narration input; a concept with nothing but
        symbol values drops out of it entirely; relations and sources after the facts are the same in both inputs."""
        from kb_pipeline.graph.compile import compile_pages, narrate_pages
        from kb_pipeline.graph.facts import symbol_only_value

        for value, expect in (("-", True), (" — ", True), ("/", True), ("N/A", True), ("-1.5", False), ("无", False), ("", False), ("0", False)):
            self.assertEqual(symbol_only_value({"value": value}), expect, value)
        graph, units = self._graph()
        base = graph["specs"][2]
        graph["specs"].append({**base, "id": "f4", "property": "眼底评估 类似", "concept": "眼底评估 类似", "concept_key": "c3",
                               "value": "-", "value_num": None, "unit": "", "ref_min": "", "ref_max": ""})
        graph["specs"].append({**base, "id": "f5", "value": "-", "value_num": None, "doc_id": "d1", "rel_path": "d1.pdf",
                               "valid_from": "2024-03-01", "axis": "2024-03-01"})
        pages, _ = compile_pages(graph, units, out_dir=None, language="Chinese")
        subject = next(p for p in pages if p["kind"] == "subject")
        self.assertIn("### 眼底评估 类似", subject["text"])
        self.assertIn("| 2025-03-01 | - |", subject["text"])
        self.assertNotIn("眼底评估", subject["narrate_text"])                     # a concept with only symbol values drops out of the narration
        self.assertIn("### 血糖", subject["narrate_text"])                        # a concept with real values stays, minus its symbol rows
        self.assertIn(f"| 2025-03-01 | {base['value']} mmol/L |", subject["narrate_text"])   # the fixture's real value stays
        self.assertNotIn("| 2024-03-01 | - |", subject["narrate_text"])
        self.assertIn("## 关系\n\n- related_to ← 某医院", subject["narrate_text"])
        prompts: list[str] = []

        class Client:
            stats: dict = {}

            def chat(self, prompt, **kw):
                prompts.append(prompt)
                return "叙述"

            def run_parallel(self, items, work, progress=None):
                return [(it, work(it), None) for it in items]

        narrate_pages(Client(), pages, language="Chinese")
        self.assertTrue(prompts)
        self.assertNotIn("眼底评估", prompts[0])
        self.assertIn("血糖", prompts[0])
        self.assertEqual(subject["summary"].split("\n")[0], "叙述")

    def test_pages_are_projected_from_the_structure_layer(self) -> None:
        from kb_pipeline.graph.compile import compile_pages, labels_for

        self.assertEqual(labels_for("Chinese")["facts"], "事实")
        self.assertEqual(labels_for("English")["facts"], "Facts")
        graph, units = self._graph()
        pages, stats = compile_pages(graph, units, out_dir=None, language="Chinese")
        kinds = Counter(p["kind"] for p in pages)
        self.assertEqual(kinds, {"subject": 1, "timeline": 1, "source": 2, "index": 1})   # profile subject person; the hospital is an entity but has no facts and is not a subject type
        subject = next(p for p in pages if p["kind"] == "subject" and p["title"] == "张三")
        self.assertEqual(subject["id"], next(p["id"] for p in pages if p["title"] == "张三"))
        text = subject["text"]
        self.assertIn("# 张三", text)
        self.assertIn("别名: Zhang San", text)
        self.assertIn("## 概述\n\n被检者,男。", text)
        self.assertIn("### 总胆固醇 (mmol/L)", text)
        self.assertIn("| 2024-03-01 | 4.6 mmol/L |  | 0~5.2 |  | d1.pdf |", text)
        self.assertIn("| 2025-03-01 | 5.4 mmol/L | ↑ | 0~5.2 |  | d2.pdf |", text)
        self.assertLess(text.index("### 总胆固醇"), text.index("### 血糖"))          # the concept with more facts comes first
        self.assertIn("## 延伸\n\n- indicates → 总胆固醇: 总胆固醇偏高", text)     # the profile's extension predicates are listed separately, before the relations
        self.assertIn("## 关系\n\n- related_to ← 某医院", text)
        self.assertNotIn("目录", text)                                             # layout relations do not enter the page
        self.assertIn("- d1.pdf (2024-03-01)", text)
        self.assertIn("总胆固醇:2024-03-01 4.6 mmol/L → 2025-03-01 5.4 mmol/L(上升,2 个点)", subject["summary"])
        self.assertEqual(subject["concept_keys"], ["c1", "c2"])
        self.assertEqual(subject["point_ids"], ["p-d1", "p-d2"])                   # attribution points with 0 hits do not count
        self.assertEqual(subject["spec_ids"], ["f1", "f2", "f3"])
        timeline = next(p for p in pages if p["kind"] == "timeline")
        self.assertEqual(timeline["title"], "张三 · 总胆固醇")
        self.assertEqual((timeline["axis_from"], timeline["axis_to"], timeline["points"], timeline["flags"]), ("2024-03-01", "2025-03-01", 2, 1))
        self.assertIn("越界 1 次", timeline["summary"])
        self.assertEqual(timeline["spec_ids"], ["f1", "f2"])
        self.assertEqual(timeline["concept_keys"], ["c1"])
        source2 = next(p for p in pages if p["kind"] == "source" and p["title"] == "d2.pdf")
        self.assertIn("轴: 2025-03-01 (date)", source2["text"])
        self.assertIn("正文 0 · 结论 1", source2["text"])
        self.assertIn("## 结论段\n\n异常结果汇总:总胆固醇偏高,建议复查。", source2["text"])
        self.assertIn("- 张三 (person)", source2["text"])
        self.assertIn("## 冲突\n\n- 张三 · 血糖 @ 2025-03-01: 4.7 (d2.pdf); 5.0 (d3.pdf)", source2["text"])
        self.assertEqual(source2["point_ids"], ["p-d2"])
        index = pages[-1]
        self.assertEqual(index["kind"], "index")
        self.assertIn("## 主体页 (1)", index["text"])
        self.assertIn("- [张三 · 总胆固醇](timelines/", index["text"])
        self.assertIn("- 总胆固醇:2 条事实,2 份文档", index["text"])
        self.assertEqual((stats["pages"], stats["subjects"], stats["timelines"], stats["sources"]), (5, 1, 1, 2))
        # Source pages sort by axis, subject pages by score
        self.assertEqual([p["title"] for p in pages if p["kind"] == "source"], ["d1.pdf", "d2.pdf"])
        self.assertEqual([p["title"] for p in pages if p["kind"] == "subject"][0], "张三")

    def test_subject_pages_list_the_facts_of_their_parts(self) -> None:
        """10-24 改版二十二:事实挂到各部分名下以后,上级实体的主体页说不全它由什么组成。组成关系(两个方向都认)连着的
        部分一个一行、紧跟概述(放在事实表后面的话叙述输入和页面向量都截不到),每行带它名下最多 FACTS_PER_PART 条事实;
        另有主体页的部分只列名字和事实数;类别名、非组成关系连着的东西不算。"""
        from kb_pipeline.graph.compile import FACTS_PER_PART, compile_pages, parts_index
        from kb_pipeline.ontology import composition_ends

        self.assertEqual(composition_ends("has_component", "a", "b"), ("a", "b"))
        self.assertEqual(composition_ends("Part-Of", "b", "a"), ("a", "b"))
        self.assertIsNone(composition_ends("uses", "a", "b"))
        self.assertIsNone(composition_ends("belongs_to", "b", "a"))           # 归属不是组成

        def ent(key, title, etype, upper, **kw):
            return {"key": key, "title": title, "type": etype, "parent_type": upper, "upper": upper, "description": "", "frequency": 3,
                    "degree": 1, "pagerank": 0.1, "doc_ids": ["d1"], "aliases": [], "unit_ids": ["u1"], **kw}

        def fact(fid, subject_key, prop, value, when="2025-01-01", flag=""):
            return {"id": fid, "subject": subject_key, "subject_key": subject_key, "subject_keys": [subject_key], "property": prop,
                    "concept": prop, "concept_key": f"c-{prop}", "value": value, "unit": "", "doc_id": "d1", "rel_path": "d1.md",
                    "section": "S", "valid_from": when, "axis": when, "point_ids": [f"p-{fid}"], "flag": flag, "conditions": {}}

        entities = [ent("plat", "平台甲", "product", "entity"), ent("mod", "调度模块", "module", "part"),
                    ent("db", "数据库乙", "product", "entity"), ent("api", "接口层", "module", "part"),
                    ent("skill", "技能", "concept", "concept", kind="class"), ent("tool", "工具丙", "product", "entity")]
        relations = [{"source_key": "plat", "target_key": "mod", "predicate": "has_component", "weight": 3.0},
                     {"source_key": "plat", "target_key": "db", "predicate": "composed_of", "weight": 2.0},
                     {"source_key": "api", "target_key": "plat", "predicate": "part_of", "weight": 2.0},
                     {"source_key": "plat", "target_key": "skill", "predicate": "includes", "weight": 2.0},
                     {"source_key": "plat", "target_key": "tool", "predicate": "uses", "weight": 5.0},
                     {"source_key": "plat", "target_key": "api", "predicate": "has_component", "weight": 1.0, "boilerplate": True}]
        facts = [fact(f"o{i}", "plat", f"own{i}", str(i)) for i in range(3)]
        facts += [fact(f"m{i}", "mod", f"参数{i}", str(i), when=f"202{i}-01-01") for i in range(6)]
        facts += [fact("m-flag", "mod", "告警阈值", "90", when="2020-01-01", flag="↑")]
        facts += [fact(f"d{i}", "db", f"容量{i}", str(i)) for i in range(3)]
        facts += [fact("a0", "api", "接口数", "12"), fact("s0", "skill", "数量", "8"), fact("t0", "tool", "版本", "2")]
        self.assertEqual(parts_index(relations, entities), {"plat": ["mod", "db", "api"]})    # 类别名、uses、版式关系不算
        graph = {"kb_id": "kb_x", "entities": entities, "relations": relations, "specs": facts, "concepts": [], "conflicts": [],
                 "documents": {"d1": {"rel_path": "d1.md", "kind": "none", "value": ""}}, "profile": {"subject_types": ["product"]},
                 "mentions": [], "unit_kinds": {}}
        pages, _ = compile_pages(graph, [_unit("u1", doc="d1")], out_dir=None, language="Chinese")
        page = next(p for p in pages if p["kind"] == "subject" and p["title"] == "平台甲")
        text = page["text"]
        self.assertLess(text.index("## 概述"), text.index("## 组成部分"))
        self.assertLess(text.index("## 组成部分"), text.index("## 事实"))
        section = text.split("## 组成部分", 1)[1].split("## 事实", 1)[0]
        self.assertIn("3 个组成部分", text)
        self.assertLess(section.index("- 调度模块 (module): "), section.index("- 接口层 (module): "))     # 事实多的部分在前
        mod_line = next(line for line in section.splitlines() if line.startswith("- 调度模块"))
        self.assertEqual(mod_line.count("参数") + mod_line.count("告警阈值"), FACTS_PER_PART)
        self.assertLess(mod_line.index("告警阈值 90 ↑"), mod_line.index("参数5"))     # 有标记的在前,其余从新到旧
        self.assertLess(mod_line.index("参数5"), mod_line.index("参数4"))
        self.assertTrue(mod_line.endswith("另有 4 条未列出"))
        self.assertIn("- 接口层 (module): 接口数 12(2025-01-01)", section)
        self.assertIn("数据库乙(3 条事实,另有主体页)", section)                 # 自己有主体页:只列名字,不搬事实
        self.assertNotIn("容量", section)
        self.assertNotIn("技能", section)
        self.assertNotIn("工具丙", section)
        self.assertIn("## 组成部分", page["narrate_text"])
        self.assertEqual(page["point_ids"][:3], ["p-o0", "p-o1", "p-o2"])         # 自己的事实在前,部分的事实在后
        self.assertIn("p-a0", page["point_ids"])
        self.assertEqual((page["parts"], page["part_facts"]), (3, FACTS_PER_PART + 1))
        db_page = next(p for p in pages if p["kind"] == "subject" and p["title"] == "数据库乙")
        self.assertNotIn("## 组成部分", db_page["text"])

    def test_series_line_prefers_numeric_endpoints(self) -> None:
        from kb_pipeline.graph.compile import _series_line, labels_for

        L = labels_for("Chinese")
        rows = [{"concept": "总胆固醇", "value": "增高", "value_num": None, "unit": "", "valid_from": "2022-06-05"},
                {"concept": "总胆固醇", "value": "6.1", "value_num": 6.1, "unit": "mmol/L", "valid_from": "2022-06-05"},
                {"concept": "总胆固醇", "value": "6.35", "value_num": 6.35, "unit": "mmol/L", "valid_from": "2023-09-12"},
                {"concept": "总胆固醇", "value": "增高", "value_num": None, "unit": "", "valid_from": "2023-09-12"}]
        self.assertEqual(_series_line(rows, L), "总胆固醇:2022-06-05 6.1 mmol/L → 2023-09-12 6.35 mmol/L(上升,4 个点)")
        only_flags = [rows[0], rows[3]]
        self.assertEqual(_series_line(only_flags, L), "总胆固醇:2022-06-05 增高 → 2023-09-12 增高(持平,2 个点)")

    def test_trend_counts_each_axis_once(self) -> None:
        """The same value stated repeatedly on one axis used to count as one point per row: the difference between the
        repeated rows is 0, so a monotonic series was called "changing" and one with numbers on a single axis "flat".
        The trend now compares axes; different values on the same axis leave no single trend."""
        from kb_pipeline.graph.compile import _series_line, _trend, labels_for

        L = labels_for("Chinese")

        def row(when, value, unit="mmol/L"):
            return {"concept": "总胆固醇", "value": value, "unit": unit, "valid_from": when}

        self.assertEqual(_trend([row("2023", "5.0"), row("2023", "5.00"), row("2024", "6.0")], L), L["trend_up"])
        self.assertEqual(_trend([row("2023", "6.0"), row("2024", "5.5"), row("2024", "5.5"), row("2025", "5.0")], L), L["trend_down"])
        self.assertEqual(_trend([row("2023", "5.0"), row("2023", "5.0"), row("2024", "5.0")], L), L["trend_flat"])
        self.assertEqual(_trend([row("2024", "6.0"), row("2023", "5.0"), row("2023", "5.0")], L), L["trend_up"])          # ordered by axis, not by row order
        self.assertEqual(_trend([row("2023", "5.0"), row("2023", "5.4"), row("2024", "6.0")], L), L["trend_mixed"])       # two values on one axis
        # the rows with numbers all sit on one axis: nothing was compared across axes, so a zero difference between
        # repeated rows does not make it flat
        self.assertEqual(_trend([row("2023", "5.0"), row("2023", "5.0"), row("2024", "增高", unit="")], L), L["trend_mixed"])
        self.assertEqual(_trend([row("2023", "5.0"), row("2023", "5.0")], L), L["trend_flat"])
        # the point count still follows the evidence rows, matching the rows of the page's table
        self.assertEqual(_series_line([row("2023", "5.0"), row("2023", "5.0"), row("2024", "6.0")], L),
                         "总胆固醇:2023 5.0 mmol/L → 2024 6.0 mmol/L(上升,3 个点)")

    def test_subject_page_series_stay_within_one_kind_of_axis(self) -> None:
        from kb_pipeline.graph.compile import compile_pages

        graph, units = self._graph()
        base = graph["specs"][0]

        def fact(fid, prop, value, when, *, ckey):
            return {**base, "id": fid, "property": prop, "concept": prop, "concept_key": ckey, "value": value, "value_num": float(value),
                    "unit": "万", "valid_from": when, "axis": when, "flag": "", "ref_min": "", "ref_max": "", "series_key": None}

        graph["specs"] = [
            fact("m1", "支撑用户量", "50", "2023-10-31", ckey="c5"), fact("m2", "支撑用户量", "80", "v6.0", ckey="c5"),       # one date, one version
            fact("k1", "并发数", "10", "2023-01", ckey="c6"), fact("k2", "并发数", "20", "2024-01", ckey="c6"),
            fact("k3", "并发数", "90", "v6.10", ckey="c6"), fact("k4", "并发数", "70", "v6.2", ckey="c6"), fact("k5", "并发数", "5", "", ckey="c6"),
        ]
        pages, _ = compile_pages(graph, units, out_dir=None, language="Chinese")
        subject = next(p for p in pages if p["kind"] == "subject")
        self.assertEqual(subject["series"], ["并发数:2023-01 10 万 → 2024-01 20 万(上升,2 个点)",
                                             "并发数:v6.2 70 万 → v6.10 90 万(上升,2 个点)"])
        self.assertNotIn("支撑用户量:", subject["summary"])                          # one date point and one version point: no series
        self.assertIn("| v6.0 | 80 万 |", subject["text"])                          # still listed in the fact table

    def test_subject_selection_degrades_without_profile_or_facts(self) -> None:
        from kb_pipeline.graph.compile import SUBJECT_FALLBACK, select_subjects

        ents = [{"key": f"e{i}", "title": f"E{i}", "type": "t", "parent_type": "", "upper": "entity", "frequency": 30 - i, "pagerank": 0.01,
                 "doc_ids": ["d1"]} for i in range(30)]
        ents.append({"key": "attr", "title": "A", "type": "a", "parent_type": "", "upper": "attribute", "frequency": 99, "doc_ids": ["d1"]})
        ents.append({"key": "bp", "title": "B", "type": "t", "parent_type": "", "upper": "entity", "frequency": 99, "doc_ids": ["d1"], "boilerplate": True})
        chosen = select_subjects(ents, {}, {})
        self.assertEqual(len(chosen), SUBJECT_FALLBACK)
        self.assertEqual(chosen[0]["key"], "e0")
        self.assertNotIn("attr", {e["key"] for e in chosen})
        self.assertNotIn("bp", {e["key"] for e in chosen})
        # An entity with facts gets a page even when it is not a subject type
        rich = select_subjects(ents, {"e29": [{}, {}, {}]}, {"subject_types": ["nothing"]})
        self.assertEqual([e["key"] for e in rich], ["e29"])

    def test_pages_are_written_and_narrated(self) -> None:
        from kb_pipeline.graph.compile import compile_pages, write_pages

        graph, units = self._graph()
        asked: list[str] = []

        class FakeClient:
            stats = {"calls": 1}

            def chat(self, prompt, max_tokens=0):
                asked.append(prompt)
                return "张三两年间总胆固醇从 4.6 升到 5.4,2025 年越界。"

            def run_parallel(self, items, fn, *, workers=None, progress=None):
                return [(item, fn(item), None) for item in items]

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "wiki"
            out.mkdir()
            (out / "stale.md").write_text("old", encoding="utf-8")
            pages, stats = compile_pages(graph, units, out_dir=out, language="Chinese", client=FakeClient())
            self.assertFalse((out / "stale.md").exists())                          # the whole directory is rewritten
            self.assertTrue((out / "index.md").exists())
            # Paths carry a stable identity suffix: same-named subjects / same-named files in different directories no
            # longer write to the same file (Codex review F11)
            subject_files = sorted(x.name for x in (out / "subjects").glob("张三--*.md"))
            self.assertEqual(len(subject_files), 1)
            self.assertRegex(subject_files[0], r"^张三--[0-9a-f]{8}\.md$")
            self.assertEqual(len(list((out / "sources").glob("d1--*.md"))), 1)
            listed = json.loads((out / "pages.json").read_text(encoding="utf-8"))
            self.assertEqual(len(listed), len(pages))
            self.assertNotIn("text", listed[0])
            self.assertEqual(len({p["path"] for p in listed}), len(listed))                  # every path in the manifest is unique
            # Even if two pages compute the same path, writing does not overwrite: the latter gets a sequence number, and
            # the manifest records the actual file
            dup = [{"id": "a", "path": "subjects/x.md", "text": "A"}, {"id": "b", "path": "subjects/x.md", "text": "B"}]
            self.assertEqual(write_pages(Path(tmp) / "w2", dup), 2)
            self.assertEqual(sorted(x.name for x in (Path(tmp) / "w2" / "subjects").iterdir()), ["x-2.md", "x.md"])
            self.assertEqual([p["path"] for p in dup], ["subjects/x.md", "subjects/x-2.md"])
            self.assertEqual((Path(tmp) / "w2" / "subjects" / "x-2.md").read_text(encoding="utf-8"), "B")
        self.assertEqual(stats["narrate"], {"candidates": 1, "narrated": 1, "failed": 0})   # only subject pages with ≥ 2 facts are narrated
        self.assertEqual(len(asked), 1)
        self.assertIn("compiled reference page about \"张三\"", asked[0])
        self.assertIn("| 2024-03-01 | 4.6 mmol/L", asked[0])
        subject = next(p for p in pages if p["title"] == "张三")
        self.assertTrue(subject["summary"].startswith("张三两年间"))
        self.assertIn("## 概述\n\n张三两年间总胆固醇从 4.6 升到 5.4,2025 年越界。\n\n被检者,男。", subject["text"])
        self.assertEqual(stats["llm"], {"calls": 1})
        # Narration off: no model call
        asked.clear()
        _, stats2 = compile_pages(graph, units, out_dir=None, language="Chinese", client=FakeClient(), narrate=False)
        self.assertEqual((asked, stats2["narrate"]), ([], {}))

    def test_compile_phase_sits_between_facts_and_enrich(self) -> None:
        from kb_pipeline.graph.build import GRAPH_PHASE_LABELS, GRAPH_PHASES

        names = [p for p, _ in GRAPH_PHASES]
        self.assertEqual(names.index("compile"), names.index("facts") + 1)
        self.assertEqual(names.index("enrich"), names.index("compile") + 1)
        self.assertEqual(GRAPH_PHASE_LABELS["compile"], "Compiling view pages")
        src = (Path(__file__).resolve().parents[1] / "kb_pipeline" / "graph" / "build.py").read_text(encoding="utf-8")
        self.assertIn('run_phase("compile", compile_views', src)
        self.assertIn('graph["pages"] = [{k: v for k, v in p.items() if k != "narrate_text"} for p in pages]', src)   # narration input is not persisted
        self.assertIn('paths.work_dir / "wiki"', src)
        js = (Path(__file__).resolve().parents[1] / "kb_server" / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn('"Compiling view pages"', js)
        self.assertIn('w("compile")', js)


class MeasurementNormalizationTests(unittest.TestCase):
    """Synthetic measurement tables exercise subject ownership, flag suffixes and
    multi-year axes. Re-attribution stays deterministic and profile-driven."""

    def _entities(self):
        def ent(key, title, etype, upper, docs, freq=1, scope=""):
            return {"key": key, "title": title, "type": etype, "parent_type": upper, "upper": upper, "frequency": freq, "doc_ids": docs,
                    "aliases": [], "scope": scope}
        return [ent("p1", "李华", "patient", "entity", ["d1", "d2", "d3"], freq=7),
                ent("p2", "李先生_001", "patient", "entity", ["d4"], freq=1),
                ent("dm", "高血压", "medical condition", "entity", ["d1", "d2"], freq=16),
                ent("cbc", "血常规", "laboratory test", "process", ["d2"]),
                ent("ecg", "心电图", "medical imaging", "process", ["d1"]),
                ent("fb", "本体反馈单", "feedback form", "process", ["d1"]),
                ent("tc", "总胆固醇", "biomarker", "property", ["d2"]),
                ent("ua", "尿酸", "biomarker", "property", ["d1"]),
                ent("bmi", "体重指数", "health indicator", "property", ["d1", "d2"]),
                ent("fat", "超重", "risk factor", "property", ["d1"])]

    def test_document_subjects_prefer_profile_types(self) -> None:
        from kb_pipeline.graph.facts import document_subjects, profile_subject_by_doc

        ents = self._entities()
        self.assertEqual(document_subjects(ents)["d1"][0], "高血压")                                 # old rule: by frequency
        self.assertEqual(document_subjects(ents, subject_types=["patient"])["d1"][:2], ["李华", "高血压"])   # profile subject types come first
        by_doc = profile_subject_by_doc(ents, ["patient"])
        self.assertEqual((by_doc["d1"]["key"], by_doc["d4"]["key"]), ("p1", "p2"))
        # The profile gives three subject types: their order is the priority, not frequency (diabetes 16 > the patient 7,
        # but patient comes first)
        multi = profile_subject_by_doc(ents, ["patient", "medical condition", "health report"])
        self.assertEqual(multi["d1"]["key"], "p1")
        self.assertEqual(document_subjects(ents, subject_types=["patient", "medical condition"])["d1"][:2], ["李华", "高血压"])
        self.assertEqual(profile_subject_by_doc(ents, ["medical condition"])["d1"]["key"], "dm")
        self.assertNotIn("d5", by_doc)
        self.assertEqual(profile_subject_by_doc(ents, []), {})

    def test_measurements_move_to_the_document_subject(self) -> None:
        from kb_pipeline.graph.facts import normalize_measurements

        units = {"u1": _unit("u1", doc="d1"), "u2": _unit("u2", doc="d2"), "u4": _unit("u4", doc="d4")}
        facts = [
            {"id": "a", "unit_id": "u1", "doc_id": "d1", "subject": "心电图", "property": "心率", "value": "68", "value_num": 68.0, "unit": "bpm", "conditions": {}},
            {"id": "b", "unit_id": "u2", "doc_id": "d2", "subject": "血常规", "property": "血红蛋白", "value": "138", "value_num": 138.0, "unit": "g/L", "conditions": {}},
            {"id": "c", "unit_id": "u2", "doc_id": "d2", "subject": "总胆固醇", "property": "测量结果", "value": "5.93", "value_num": 5.93, "unit": "mmol/L", "ref_min": "0", "ref_max": "5.2", "flag": "↑", "conditions": {}},
            {"id": "d", "unit_id": "u1", "doc_id": "d1", "subject": "体重指数", "property": "体重指数", "value": "22.9", "value_num": 22.9, "unit": "", "ref_min": "18.5", "ref_max": "23.99", "flag": "↑", "conditions": {}},
            {"id": "e", "unit_id": "u1", "doc_id": "d1", "subject": "超重", "property": "尿酸", "value": "468", "value_num": 468.0, "unit": "umol/L", "ref_min": "210", "ref_max": "420", "flag": "↑", "conditions": {}},
            {"id": "f", "unit_id": "u2", "doc_id": "d2", "subject": "总胆固醇", "property": "总胆固醇增高", "value": "增高", "value_num": None, "unit": "", "conditions": {}},
            {"id": "g", "unit_id": "u2", "doc_id": "d2", "subject": "总胆固醇", "property": "增高原因", "value": "高脂蛋白血症等", "value_num": None, "unit": "", "conditions": {}},
            {"id": "h", "unit_id": "u2", "doc_id": "d2", "subject": "体重指数", "property": "测量结果", "value": "24.3", "value_num": 24.3, "unit": "", "conditions": {"年份": "2022/06"}},
            {"id": "i", "unit_id": "u2", "doc_id": "d2", "subject": "高血压", "property": "关注等级", "value": "一般", "value_num": None, "unit": "", "conditions": {}},
            {"id": "j", "unit_id": "u4", "doc_id": "d4", "subject": "李先生_001", "property": "心率", "value": "72", "value_num": 72.0, "unit": "bpm", "conditions": {}},
            {"id": "k", "unit_id": "u2", "doc_id": "d2", "subject": "李华", "property": "年龄", "value": "44", "value_num": 44.0, "unit": "岁", "conditions": {}},
            {"id": "l", "unit_id": "u1", "doc_id": "d1", "subject": "本体反馈单", "property": "提交人", "value": "某成员", "value_num": None, "unit": "", "conditions": {}},
        ]
        docs = {"d1": "李华/体检报告/2024.pdf", "d2": "李华/体检报告/2025.pdf", "d4": "其他/心电图.pdf"}     # d1、d2 由路径定归属人
        stats = normalize_measurements(facts, self._entities(), units_by_id=units, profile={"subject_types": ["patient"], "axis": "date"},
                                       documents=docs)
        by = {f["id"]: f for f in facts}
        self.assertEqual((by["a"]["subject"], by["a"]["context"], by["a"]["property"], by["a"]["subject_key"]), ("李华", "心电图", "心率", "p1"))   # lab sheet / examination item → examinee
        self.assertEqual((by["b"]["subject"], by["b"]["context"]), ("李华", "血常规"))
        self.assertEqual((by["c"]["subject"], by["c"]["property"], by["c"]["subject_raw"]), ("李华", "总胆固醇", "总胆固醇"))               # indicator + generic property → the indicator becomes the property
        self.assertEqual((by["d"]["subject"], by["d"]["property"]), ("李华", "体重指数"))                                              # subject and property share a name
        self.assertEqual((by["e"]["subject"], by["e"]["property"], by["e"]["context"]), ("李华", "尿酸", "超重"))                     # abnormal-findings summary: a measurement with a reference range under a diagnosis
        self.assertEqual((by["f"]["property"], by["f"]["flag"], by["f"]["subject"]), ("总胆固醇", "↑", "李华"))                          # the trailing flag word is stripped into flag
        self.assertEqual((by["g"]["subject"], by["g"]["property"]), ("总胆固醇", "增高原因"))                                             # not a measurement: knowledge about the indicator, left alone
        self.assertNotIn("subject_raw", by["g"])
        self.assertEqual((by["h"]["valid_from"], by["h"]["period_text"], by["h"]["conditions"], by["h"]["subject"], by["h"]["property"]),
                         ("2022-06", "年份: 2022/06", {}, "李华", "体重指数"))                                                           # a period condition goes into the axis
        self.assertEqual((by["i"]["subject"], by["i"].get("subject_raw")), ("高血压", None))                                          # text value, no reference range: stays under the diagnosis
        self.assertEqual(by["j"]["subject"], "李先生_001")                                                                            # this document's profile subject is itself
        self.assertEqual((by["k"]["subject"], by["k"].get("subject_raw")), ("李华", None))
        self.assertEqual((by["l"]["subject"], by["l"].get("subject_raw")), ("本体反馈单", None))   # a business fact under a process subject (not a measurement): stays with the original subject (Codex 2026-09-13 F03)
        self.assertEqual(stats, {"flag_words": 1, "period_conditions": 1, "resubjected_process": 2, "resubjected_indicator": 1,
                                 "resubjected_generic": 4, "resubjected_context": 0, "resubjected_identity": 0, "identity_kept_shared": 0})     # 肥胖在本库是 risk factor(上层 property):走指标规则,原主体留作 context
        # 归属人只是按频次猜的(没有路径 / 序列证据):不归位。合成汇编中子项目的数字不能归到出现最多的主体名下
        facts3 = [{"id": "a", "unit_id": "u1", "doc_id": "d1", "subject": "心电图", "property": "心率", "value": "65", "value_num": 65.0, "unit": "bpm", "conditions": {}},
                  {"id": "c", "unit_id": "u2", "doc_id": "d2", "subject": "总胆固醇", "property": "测量结果", "value": "6.05", "value_num": 6.15, "unit": "mmol/L", "conditions": {}}]
        stats3 = normalize_measurements(facts3, self._entities(), units_by_id=units, profile={"subject_types": ["patient"], "axis": "date"})
        self.assertEqual([(f["subject"], f.get("subject_raw")) for f in facts3], [("心电图", None), ("总胆固醇", None)])
        self.assertEqual(stats3["resubjected_process"] + stats3["resubjected_generic"], 0)
        # 没有画像主体类型(手册库):只剥标记词、搬时期,主体一律不动
        facts2 = [{"id": "x", "unit_id": "u1", "doc_id": "d1", "subject": "总胆固醇", "property": "测量结果", "value": "6", "value_num": 6.1, "unit": "mmol/L", "conditions": {}}]
        stats2 = normalize_measurements(facts2, self._entities(), units_by_id=units, profile={"subject_types": [], "axis": "version"})
        self.assertEqual((facts2[0]["subject"], stats2["resubjected_generic"]), ("总胆固醇", 0))
        # Concept keys and series follow the re-attributed property
        from kb_pipeline.graph.concepts import build_concepts
        from kb_pipeline.graph.reconcile import reconcile_facts
        series = [{"id": "s1", "subject": "李华", "subject_key": "p1", "property": "总胆固醇", "value": "5.0", "value_num": 5.0, "unit": "mmol/L", "doc_id": "d1", "unit_id": "u1", "valid_from": "2021-03-14", "conditions": {}},
                  {"id": "s2", "subject": "李华", "subject_key": "p1", "property": "总胆固醇增高", "value": "增高", "value_num": None, "unit": "", "doc_id": "d2", "unit_id": "u2", "valid_from": "2023-09-12", "conditions": {}}]
        normalize_measurements(series, self._entities(), units_by_id=units, profile={"subject_types": ["patient"], "axis": "date"})
        build_concepts(series)
        self.assertEqual(series[0]["concept_key"], series[1]["concept_key"])

    def test_flag_words_are_only_stripped_from_text_valued_facts(self) -> None:
        """High / Low, "decrease" or "abnormal" at the end of a property name used to be split off as a flag every
        time: Price High and Price Low merged into one property, both carrying an out-of-range arrow, and the decrease
        in "consumption down 30%" was read as the consumption itself. When the value is a quantity the tail word is
        part of the property name and stays; when the value is text it is still split off, and the original name is
        recorded."""
        from kb_pipeline.graph.concepts import build_concepts
        from kb_pipeline.graph.facts import normalize_fact, normalize_measurements

        def fact(fid, prop, **fields):
            row = normalize_fact({"subject": "某基金", "property": prop, **fields})
            row.update(id=fid, unit_id="u1", doc_id="d1")
            return row

        facts = [fact("hi", "Price High", value="105.2"), fact("lo", "Price Low", value="98.7"),
                 fact("cut", "消耗下降", value="30%"), fact("p50", "首 token 延迟 p50 降低", min="20", max="35", unit="%"),
                 fact("rng", "Score-Q High", value="1.2 to 1.8"), fact("expr", "Input voltage high", max="V_CC + 0.5"),
                 fact("tc", "总胆固醇增高", value="增高"), fact("mk", "Marker-X positive", value="阳性", flag="+"),
                 fact("note", "检查结果异常", value="详见附页"), fact("bare", "异常", value="有")]
        stats = normalize_measurements(facts, [], units_by_id={"u1": _unit("u1", doc="d1")}, profile={"subject_types": ["fund"], "axis": "date"})
        by = {f["id"]: f for f in facts}
        for fid, prop in (("hi", "Price High"), ("lo", "Price Low"), ("cut", "消耗下降"), ("p50", "首 token 延迟 p50 降低"),
                          ("rng", "Score-Q High"), ("expr", "Input voltage high")):
            self.assertEqual((by[fid]["property"], by[fid].get("flag"), by[fid].get("property_raw")), (prop, None, None), fid)
        self.assertEqual((by["tc"]["property"], by["tc"]["flag"], by["tc"]["property_raw"]), ("总胆固醇", "↑", "总胆固醇增高"))
        self.assertEqual((by["mk"]["property"], by["mk"]["flag"], by["mk"]["property_raw"]), ("Marker-X", "+", "Marker-X positive"))   # the flag given by the model wins
        self.assertEqual((by["note"]["property"], by["note"]["flag"], by["note"]["property_raw"]), ("检查结果", "异常", "检查结果异常"))
        self.assertEqual((by["bare"]["property"], by["bare"].get("property_raw")), ("异常", None))          # the whole property name is a flag word: left alone
        self.assertEqual(stats["flag_words"], 3)
        build_concepts(facts)
        self.assertNotEqual(by["hi"]["concept_key"], by["lo"]["concept_key"])

    def test_bad_endpoint_constraints_are_relaxed(self) -> None:
        from kb_pipeline.graph.merge import predicate_health, relax_bad_endpoints

        rels = [{"source_key": f"a{i}", "target_key": f"b{i}", "predicate": "recommends", "type_violation": True} for i in range(30)]
        rels += [{"source_key": f"c{i}", "target_key": f"d{i}", "predicate": "has_pin", "type_violation": i < 3} for i in range(30)]
        rels += [{"source_key": "x", "target_key": "y", "predicate": "rare", "type_violation": True}]
        relaxed = relax_bad_endpoints(rels)
        self.assertEqual(relaxed, ["recommends"])                                          # over half violating with enough edges: relaxed; rare has too few edges and stays
        self.assertFalse(any(r["type_violation"] for r in rels if r["predicate"] == "recommends"))
        self.assertTrue(all(r.get("endpoints_relaxed") for r in rels if r["predicate"] == "recommends"))
        self.assertEqual(sum(1 for r in rels if r["type_violation"]), 4)
        health = {h["predicate"]: h for h in predicate_health(rels, [], relaxed=relaxed)}
        self.assertTrue(health["recommends"]["relaxed"])
        self.assertFalse(health["has_pin"]["relaxed"])
        src = (Path(__file__).resolve().parents[1] / "kb_pipeline" / "graph" / "merge.py").read_text(encoding="utf-8")
        self.assertIn("relaxed = relax_bad_endpoints(relation_rows)", src)
        self.assertIn("relaxed_pairs = relax_bad_endpoint_pairs(relation_rows, entity_rows)", src)
        build = (Path(__file__).resolve().parents[1] / "kb_pipeline" / "graph" / "build.py").read_text(encoding="utf-8")
        self.assertIn("normalize_measurements(all_facts", build)
        self.assertIn('subject_types=(graph.get("profile") or {}).get("subject_types")', build)

    def test_concentrated_endpoint_pairs_are_relaxed_per_pair(self) -> None:
        from kb_pipeline.graph.merge import predicate_health, relax_bad_endpoint_pairs, relax_bad_endpoints

        ents = [{"key": f"e{i}", "upper": "entity"} for i in range(60)] + [{"key": f"p{i}", "upper": "property"} for i in range(60)] \
            + [{"key": f"c{i}", "upper": "process"} for i in range(60)]
        # associated_with 100 条:entity → property 25 条越界(集中,25%)、process → property 3 条越界(零星)、其余 72 条合规
        rels = [{"source_key": f"e{i}", "target_key": f"p{i}", "predicate": "associated_with", "type_violation": True} for i in range(25)]
        rels += [{"source_key": f"c{i}", "target_key": f"p{i}", "predicate": "associated_with", "type_violation": True} for i in range(3)]
        rels += [{"source_key": f"e{i}", "target_key": f"e{i + 1}", "predicate": "associated_with", "type_violation": False} for i in range(72)]
        # has_pin: 22 of 40 edges violate entity → property (55%): the whole-predicate rule relaxes it first
        rels += [{"source_key": f"e{i}", "target_key": f"p{i}", "predicate": "has_pin", "type_violation": i < 22} for i in range(40)]
        # rare: 5 violations on the same pair, below the threshold of 20
        rels += [{"source_key": f"c{i}", "target_key": f"e{i}", "predicate": "rare", "type_violation": True} for i in range(5)]
        relaxed = relax_bad_endpoints(rels)
        self.assertEqual(relaxed, ["has_pin"])
        pairs = relax_bad_endpoint_pairs(rels, ents)
        self.assertEqual(pairs, [{"predicate": "associated_with", "source_parent": "entity", "target_parent": "property", "count": 25}])
        self.assertEqual(sum(1 for r in rels if r["type_violation"]), 3 + 5)              # only the 3 sporadic ones and rare's 5 remain
        self.assertTrue(all(r.get("endpoints_relaxed") for r in rels[:25]))
        health = {h["predicate"]: h for h in predicate_health(rels, ents, relaxed=relaxed, relaxed_pairs=pairs)}
        self.assertEqual(health["associated_with"]["relaxed_pairs"], [{"source_parent": "entity", "target_parent": "property", "count": 25}])
        self.assertEqual((health["associated_with"]["violations"], health["associated_with"]["relaxed"]), (3, False))
        self.assertEqual(health["rare"]["relaxed_pairs"], [])


    def test_subject_identity_from_path_and_series(self) -> None:
        from kb_pipeline.graph.facts import document_subjects, normalize_measurements, subject_evidence_by_doc

        def ent(key, title, etype, upper, docs, freq=1):
            return {"key": key, "title": title, "type": etype, "parent_type": upper, "upper": upper, "frequency": freq, "doc_ids": docs, "aliases": [], "scope": ""}
        ents = [ent("p1", "李华", "patient", "entity", ["d1", "d2"], freq=9),
                ent("p2", "李先生_2000000000002_1", "patient", "entity", ["d3"], freq=3),        # ECG report: anonymous ID
                ent("p3", "钱小丽", "patient", "entity", ["d4"], freq=1),                       # lab sheet: the issuer was extracted as the examinee
                ent("p4", "张三", "patient", "entity", ["d5", "d6"], freq=4),
                ent("hr", "心率", "biomarker", "property", ["d3"])]
        docs = {"d1": "李华/体检报告/2021体检报告.pdf", "d2": "李华/体检报告/2022体检报告.pdf", "d3": "李华/体检报告/2022-心电图.pdf",
                "d4": "李华/体检报告/2022-专项检查.pdf", "d5": "张三/2021.pdf", "d6": "张三/2022.pdf", "d7": "其他/说明.pdf"}
        ev = subject_evidence_by_doc(ents, ["patient"], docs)
        self.assertEqual({d: (e["key"], how) for d, (e, how) in ev.items()},
                         {"d1": ("p1", "path"), "d2": ("p1", "path"), "d3": ("p1", "path"), "d4": ("p1", "path"), "d5": ("p4", "path"), "d6": ("p4", "path")})
        # No name in the path: same-directory series evidence — over half the documents point to S, so ID-style subjects /
        # documents without a subject go to S; those with a real name of their own stay
        docs2 = {"d1": "报告/2021.pdf", "d2": "报告/2022.pdf", "d3": "报告/心电图.pdf", "d4": "报告/检验单.pdf", "d7": "报告/说明.pdf"}
        ev2 = subject_evidence_by_doc(ents, ["patient"], docs2)
        self.assertEqual((ev2["d1"][0]["key"], ev2["d1"][1]), ("p1", "frequency"))
        self.assertEqual((ev2["d3"][0]["key"], ev2["d3"][1]), ("p1", "series"))            # p2's name is ID-style
        self.assertEqual((ev2["d4"][0]["key"], ev2["d4"][1]), ("p3", "frequency"))         # p3 has a real name: series evidence does not change it, only the path can
        self.assertEqual((ev2["d7"][0]["key"], ev2["d7"][1]), ("p1", "series"))            # a document with no subject-type entity goes to the series subject
        self.assertNotIn("d7", subject_evidence_by_doc(ents, ["patient"], {"d7": "单独/说明.pdf"}))   # alone in its directory: not attributed
        self.assertEqual(subject_evidence_by_doc(ents, ["patient"], None)["d3"][1], "frequency")
        self.assertEqual(document_subjects(ents, subject_types=["patient"], documents=docs)["d3"][0], "李华")
        # Re-attribution: facts filed under the document's own "subject type" entity all move to the profile subject (not
        # only measurements), recording subject_raw
        units = {"u3": _unit("u3", doc="d3"), "u4": _unit("u4", doc="d4")}
        facts = [{"id": "a", "unit_id": "u3", "doc_id": "d3", "subject": "李先生_2000000000002_1", "property": "心率", "value": "72", "value_num": 72.0, "unit": "bpm", "conditions": {}},
                 {"id": "b", "unit_id": "u3", "doc_id": "d3", "subject": "李先生_2000000000002_1", "property": "心电图结论", "value": "窦性心律", "value_num": None, "unit": "", "conditions": {}},
                 {"id": "c", "unit_id": "u4", "doc_id": "d4", "subject": "钱小丽", "property": "指标Q", "value": "1.4", "value_num": 1.4, "unit": "", "conditions": {}}]
        stats = normalize_measurements(facts, ents, units_by_id=units, profile={"subject_types": ["patient"], "axis": "date"}, documents=docs)
        self.assertEqual([(f["subject"], f.get("subject_raw"), f["subject_key"]) for f in facts],
                         [("李华", "李先生_2000000000002_1", "p1"), ("李华", "李先生_2000000000002_1", "p1"), ("李华", "钱小丽", "p1")])
        self.assertEqual(stats["resubjected_identity"], 3)
        # Without path / series evidence (a frequency-determined subject) this rule does not apply: the document's own
        # subject is itself
        facts2 = [{"id": "a", "unit_id": "u3", "doc_id": "d3", "subject": "李先生_2000000000002_1", "property": "心率", "value": "72", "value_num": 72.0, "unit": "bpm", "conditions": {}}]
        normalize_measurements(facts2, ents, units_by_id=units, profile={"subject_types": ["patient"], "axis": "date"})
        self.assertEqual(facts2[0]["subject"], "李先生_2000000000002_1")



    def test_period_in_a_condition_wins_over_the_date_inherited_from_the_document(self) -> None:
        """link_facts fills the document's axis value into facts without a validity; in a multi-year comparison table
        each row's year sits in the conditions and has to move to valid_from, otherwise the whole table lands on the
        single point of the report date. Facts that state their own validity are left alone."""
        from kb_pipeline.graph.facts import normalize_measurements

        facts = [
            {"id": "a", "subject": "李华", "property": "体重指数", "value": "21.9", "value_num": 21.9, "axis": "2023-09-12",
             "valid_from": "2023-09-12", "conditions": {"年份": "2021/05"}},
            {"id": "b", "subject": "李华", "property": "体重指数", "value": "22.6", "value_num": 22.6, "axis": "2023-09-12",
             "valid_from": "2023-09-12", "conditions": {"年份": "2022/06"}},
            {"id": "c", "subject": "李华", "property": "体重指数", "value": "23.0", "value_num": 23.0, "axis": "2023-09-12",
             "valid_from": "2023-09-12", "conditions": {"体位": "立位"}},
            {"id": "d", "subject": "李华", "property": "体重指数", "value": "21.4", "value_num": 21.4, "axis": "2023-09-12",
             "valid_from": "2020-04", "conditions": {"年份": "2019"}},
        ]
        stats = normalize_measurements(facts, [], units_by_id={})
        by = {f["id"]: f for f in facts}
        self.assertEqual([(by[k]["valid_from"], by[k].get("period_text"), by[k]["conditions"]) for k in "abcd"], [
            ("2021-05", "年份: 2021/05", {}),
            ("2022-06", "年份: 2022/06", {}),
            ("2023-09-12", None, {"体位": "立位"}),
            ("2020-04", None, {"年份": "2019"}),
        ])
        self.assertEqual({f["axis"] for f in facts}, {"2023-09-12"})
        self.assertEqual(stats["period_conditions"], 2)

class FinerEvalTests(unittest.TestCase):
    """Finer evaluation (2026-09-08): cross-document questions check how many documents the evidence comes from and
    strict questions need every keyword hit; the fact-level comparison looks the gold standard up directly in the
    facts table."""


    def test_factcheck_finds_facts_series_and_wrong_subjects(self) -> None:
        from kb_pipeline.graph.factcheck import factcheck, factcheck_markdown

        graph = {"graph_version": "v1",
                 "entities": [{"key": "p1", "title": "李华", "aliases": ["李先生_001"]}, {"key": "d1", "title": "ZK14B108L/ZK14B108N", "aliases": ["ZK14B108L"]},
                              {"key": "x1", "title": "超重", "aliases": []}],
                 "specs": [
                     {"id": "f7", "subject": "ZK14B108L/ZK14B108N", "subject_key": "d1", "property": "输出延迟", "concept": "输出延迟", "value": "20/25/45", "unit": "ns", "rel_path": "x.pdf"},
                     {"id": "f8", "subject": "李华", "subject_key": "p1", "property": "指标Q", "concept": "指标Q", "value": "<1.5", "unit": "ng/mL", "valid_from": "2022-06-05", "rel_path": "2025.pdf"},
                     {"id": "f1", "subject": "李华", "subject_key": "p1", "property": "体重指数", "concept": "体重指数", "value": "22.9", "valid_from": "2021-03-14", "series_key": "s1", "rel_path": "2024.pdf"},
                     {"id": "f2", "subject": "李华", "subject_key": "p1", "property": "体重指数", "concept": "体重指数", "value": "24.3", "valid_from": "2022-06-05", "series_key": "s1", "rel_path": "2025.pdf"},
                     {"id": "f3", "subject": "李华", "subject_key": "p1", "property": "BMI", "concept": "体重指数", "value": "24.2", "valid_from": "2023-09-12", "series_key": "s1", "rel_path": "2023.pdf", "conflict_group": "x9"},
                     {"id": "f4", "subject": "超重", "subject_key": "x1", "property": "尿酸", "concept": "尿酸", "value": "468", "unit": "umol/L", "valid_from": "2021-03-14", "rel_path": "2024.pdf"},
                     {"id": "f5", "subject": "ZK14B108L/ZK14B108N", "subject_key": "d1", "property": "电源电压", "symbol": "V_CC", "min": "2.7", "typ": "3.0", "max": "3.6", "unit": "V", "conditions": {}},
                     {"id": "f6", "subject": "ZK14B108L/ZK14B108N", "subject_key": "d1", "property": "地址访问时间", "symbol": "t_AA", "max": "25", "unit": "ns", "conditions": {"速度等级": "25 ns"}},
                 ]}
        gold = [
            {"subject": "李华", "property": "体重指数", "series": [["2021-03-14", "22.9"], ["2022-06", "24.3"], ["2023", "24.2"]]},
            {"subject": "李华", "property": "尿酸", "when": "2021-03-14", "value": "468"},                     # filed under "obesity"
            {"subject": "ZK14B108L", "property": "电源电压", "symbol": "V_CC", "min": "2.7", "typ": "3", "max": "3.6"},    # alias + numeric 3 = 3.0
            {"subject": "ZK14B108L", "symbol": "t_AA", "max": "25", "conditions": {"速度等级": "25 ns"}},
            {"subject": "ZK14B108L", "symbol": "t_AA", "max": "20", "conditions": {"速度等级": "20 ns"}},         # the graph has no 20 ns grade
            {"subject": "李华", "property": "体重指数", "when": "2022-06-05", "value": "27"},                   # wrong value
            # Codex review F05: it used to take the first number and ignore the unit, so the next three were falsely
            # reported found
            {"subject": "ZK14B108L", "symbol": "t_AA", "max": "25", "unit": "ms", "conditions": {"速度等级": "25 ns"}},   # wrong unit
            {"subject": "ZK14B108L", "property": "输出延迟", "value": "20"},                                              # the graph has 20/25/45
            {"subject": "李华", "property": "指标Q", "when": "2022", "value": "1.5"},                                    # the graph has <1.5
            {"subject": "李华", "property": "指标Q", "when": "2022", "value": "<1.5"},
        ]
        report = factcheck(graph, gold)
        s = report["summary"]
        self.assertEqual((s["found"], s["wrong_subject"], s["wrong_value"], s["missing"]), (6, 1, 4, 1))
        series = report["rows"][0]
        self.assertEqual((series["points_found"], series["points_total"], series["one_series"], series["false_conflicts"]), (3, 3, True, 1))
        self.assertEqual((s["series"], s["series_complete"], s["series_ok"], s["false_conflicts"]), (1, 1, 0, 1))
        self.assertEqual(report["rows"][1]["status"], "wrong_subject")
        self.assertEqual(report["rows"][1]["subject"], "超重")
        self.assertEqual([r["status"] for r in report["rows"][2:]],
                         ["found", "found", "missing", "wrong_value", "wrong_value", "wrong_value", "wrong_value", "found"])
        from kb_pipeline.graph.factcheck import value_equal
        self.assertTrue(value_equal("2.7 to 3.6", "2.7 ~ 3.6"))
        self.assertFalse(value_equal("2.7 to 3.6", "2.7"))
        self.assertTrue(value_equal("≤ 10", "<= 10"))
        self.assertFalse(value_equal("10", "≤ 10"))
        md = factcheck_markdown(report, "kb_x")
        self.assertIn("filed under “超重”", md)
        self.assertIn("| 李华 · 体重指数 | 3/3 | single series, conflicts 1 |", md)

    def test_rejected_pairs_are_replayed_on_append(self) -> None:
        from kb_pipeline.graph import resolution

        ents = [{"key": "a", "title": "Demo", "type": "process", "descriptions": ["演示"], "unit_ids": ["t1"], "frequency": 2, "aliases": []},
                {"key": "b", "title": "Demo阶段", "type": "process", "descriptions": ["阶段"], "unit_ids": ["t2"], "frequency": 1, "aliases": []},
                {"key": "c", "title": "Cognition", "type": "ORG", "descriptions": ["公司"], "unit_ids": ["t3"], "frequency": 2, "aliases": []},
                {"key": "d", "title": "Cognition AI", "type": "ORG", "descriptions": ["公司"], "unit_ids": ["t4"], "frequency": 1, "aliases": []}]
        prior = {"map": {"d": "c"}, "judged": [["a", "b"], ["c", "d"]],
                 "rejected": [{"a": "a", "b": "b", "a_title": "Demo", "b_title": "Demo阶段", "source": "lexical", "category": "alias", "reason": "recheck", "verdict": "different"},
                              {"a": "zz", "b": "b", "a_title": "没了", "b_title": "Demo阶段", "source": "lexical", "category": "alias", "reason": "recheck", "verdict": "different"}]}
        client = _client(["1: yes alias"])
        merged, _, stats = resolution.resolve(client, ents, [], prior=prior, type_words=["阶段"])
        self.assertEqual({e["key"] for e in merged}, {"a", "b", "c"})                     # c/d still merged, a/b still refused, neither asked again
        self.assertEqual(client.stats["calls"], 0)
        self.assertEqual(stats["replayed_rejected"], 1)                                  # pairs blocked in the previous version carry over; those whose entity is gone are dropped
        self.assertEqual([(r["a_title"], r["b_title"], r["source"], r["verdict"]) for r in stats["_rejected"]], [("Demo", "Demo阶段", "replay", "different")])


class QualityStateProjectionTests(unittest.TestCase):
    """Codex re-review 2026-09-09, graph side: N02 the quality state enters both projections; N04 cache loading
    re-applies the marks; N05 bucketing carries identity tokens and sets are mutually exclusive."""

    def test_projections_carry_the_quality_state(self) -> None:
        from kb_pipeline.graph.vectors import spec_payload

        bad = {"id": "f1", "subject": "S", "subject_key": "s", "property": "综合风险指数", "value": "31", "value_num": 25.0,
               "kinds": {"value": "scalar"}, "confidence": "low",
               "evidence_conflict": {"label": "综合风险指数", "text_value": "34", "model_value": "31"}, "cmps": {}}
        good = {"id": "f2", "subject": "S", "subject_key": "s", "property": "V_OL", "max": "<1.3", "max_num": 1.3,
                "kinds": {"max": "scalar"}, "cmps": {"max": "<"}}
        p1 = spec_payload(bad, kb_id="kb", source_collection="kb", graph_version="v")
        p2 = spec_payload(good, kb_id="kb", source_collection="kb", graph_version="v")
        self.assertEqual((p1["confidence"], p1["evidence_conflict"]["text_value"], p1["comparable"]), ("low", "34", False))
        self.assertNotIn("evidence_conflict", p2)
        self.assertEqual((p2["cmps"], p2["comparable"]), ({"max": "<"}, False))       # a value with a comparator cannot be compared as a number
        self.assertTrue(spec_payload({"id": "f3", "subject": "S", "property": "x", "value": "5"}, kb_id="kb", source_collection="kb",
                                     graph_version="v")["comparable"])
        src = (Path(__file__).resolve().parents[2] / "app/kb_pipeline/graph/neo4j_import.py").read_text(encoding="utf-8")
        for needle in ('"confidence": f.get("confidence") or None', '"evidence_conflict": json.dumps(f["evidence_conflict"]',
                       '"comparable": comparable_number(f) is not None'):
            self.assertIn(needle, src)

    def test_cached_facts_are_requalified_from_the_current_unit(self) -> None:
        from kb_pipeline.graph.facts import apply_unit_quality, comparable_number

        unit = _unit("u1")
        unit.value_conflicts = [{"label": "综合风险指数", "text_value": "34", "model_value": "31"}]
        unit.ambiguous_values = ["757557"]
        facts = [{"subject": "S", "property": "综合风险指数", "value": "31"}, {"subject": "S", "property": "血糖", "value": "5.2"},
                 {"subject": "S", "property": "编号", "value": "757557"}]
        counts = apply_unit_quality(facts, unit)
        self.assertEqual(counts, {"ambiguous": 1, "evidence_conflicts": 1})
        self.assertEqual((facts[0]["confidence"], comparable_number(facts[0])), ("low", None))
        self.assertNotIn("evidence_conflict", facts[1])
        self.assertEqual(facts[2].get("quality"), "ambiguous_source")
        self.assertEqual(apply_unit_quality(facts, _unit("u2")), {"ambiguous": 0, "evidence_conflicts": 0})   # no metadata, nothing changes
        src = (Path(__file__).resolve().parents[2] / "app/kb_pipeline/graph/build.py").read_text(encoding="utf-8")
        self.assertIn("apply_unit_quality(unit_facts, u)", src)

    def test_the_same_property_with_and_without_its_symbol_is_one_concept(self) -> None:
        """合成样本中同一属性有时带符号、有时不带,两种写法应归入同一序列。
        没写符号的桶并进唯一一个同属性名、写了符号的桶;两个符号共用一个属性名时不并。"""
        from kb_pipeline.graph.concepts import build_concepts

        facts = [{"id": "f1", "property": "体重指数", "symbol": "BMI", "value": "24", "doc_id": "d1"},
                 {"id": "f2", "property": "体重指数", "symbol": "", "value": "25", "doc_id": "d2"},
                 {"id": "f3", "property": "电压", "symbol": "VCC", "value": "3.3", "unit": "V", "doc_id": "d1"},
                 {"id": "f4", "property": "电压", "symbol": "VDD", "value": "1.8", "unit": "V", "doc_id": "d1"},
                 {"id": "f5", "property": "电压", "symbol": "", "value": "5", "unit": "V", "doc_id": "d2"},
                 {"id": "f6", "property": "低密度脂蛋白胆固醇", "symbol": "LDL-C", "value": "3.4", "unit": "mmol/L", "doc_id": "d1"},
                 {"id": "f7", "property": "低密度脂蛋白胆固醇", "symbol": "LDL-C", "value": "3.6", "unit": "mmol/L", "doc_id": "d2"},
                 {"id": "f8", "property": "低密度脂蛋白胆固醇", "symbol": "", "value": "增高", "doc_id": "d2"}]
        _, stats = build_concepts(facts)
        by = {f["id"]: f["concept_key"] for f in facts}
        self.assertEqual(by["f1"], by["f2"])
        self.assertEqual(len({by["f3"], by["f4"], by["f5"]}), 3)          # 归谁说不清:三个都留着
        self.assertNotEqual(by["f8"], by["f6"])      # 文字结论(异常汇总表的「增高」)不并进数值桶:并了序列同一年两个点
        self.assertEqual(by["f6"], by["f7"])
        self.assertEqual(stats["same_property_merged"], 1)

    def test_concept_sets_cannot_be_bridged_across_identity_tokens(self) -> None:
        from kb_pipeline.graph.concepts import build_concepts

        facts = [
            {"id": "a", "subject": "S", "property": "GENEX rs0000133 基因型", "symbol": "", "value": "TT", "doc_id": "d1"},
            {"id": "b", "subject": "S", "property": "基因型", "symbol": "", "value": "AG", "doc_id": "d2"},
            {"id": "c", "subject": "S", "property": "GENEX rs0000131 基因型", "symbol": "", "value": "GG", "doc_id": "d3"},
        ]
        vectors = {"GENEX rs0000133 基因型": [1.0, 0.0, 0.0], "基因型": [0.99, 0.14, 0.0], "GENEX rs0000131 基因型": [0.98, 0.2, 0.0]}
        _, stats = build_concepts(facts, embed=lambda texts: [vectors[t] for t in texts], auto_threshold=0.95, ask_threshold=0.9)
        keys = {f["id"]: f["concept_key"] for f in facts}
        # 2026-09-12: the bare label "genotype" no longer merges into any concept carrying a locus (it used to merge into
        # one side; the generic label must not absorb identity-bearing property names)
        self.assertEqual(len({keys["a"], keys["b"], keys["c"]}), 3)
        self.assertEqual(stats["blocked_bridge"], 0)
        self.assertTrue(stats["blocked"] and all(m["reason"] == "identity" for m in stats["blocked"]))
        # Same symbol, different locus: separated at bucketing, not left to the vector step
        same_symbol = [
            {"id": "x", "subject": "S", "property": "GENEX rs0000133 基因型", "symbol": "GENEX", "value": "TT"},
            {"id": "y", "subject": "S", "property": "GENEX rs0000131 基因型", "symbol": "GENEX", "value": "GG"},
            {"id": "z", "subject": "S", "property": "GENEX rs0000133 风险", "symbol": "GENEX rs0000133", "value": "低"},
        ]
        _, stats2 = build_concepts(same_symbol)
        self.assertEqual(stats2["norms"], 3)
        self.assertNotEqual(same_symbol[0]["concept_key"], same_symbol[1]["concept_key"])


class TypedValueFlowTests(unittest.TestCase):
    """Codex review 2026-09-09, third batch (graph side): F04 typed values throughout compile / reconcile, the second
    half of F01 (facts with estimated-reading conflicts are degraded), F02 identity-token / measure-word protection in
    concept merging plus a merge log, F10 split-in-half retry after truncation plus a partial-completion state."""

    def test_comparators_survive_normalization_and_only_plain_scalars_are_comparable(self) -> None:
        from kb_pipeline.graph.facts import comparable_number, normalize_fact

        fact = normalize_fact({"subject": "S", "property": "Output low voltage", "symbol": "V_OL", "max": "<1.3", "unit": "V"})
        self.assertEqual((fact["cmps"], fact["max_num"], fact["kinds"]), ({"max": "<"}, 1.3, {"max": "scalar"}))
        self.assertNotIn("cmps", normalize_fact({"subject": "S", "property": "x", "value": "1.3"}))
        # Compile / reconcile only accept scalars without a comparator: expressions, ranges, multi-values, text and
        # compared values are not comparable numbers
        self.assertEqual(comparable_number({"value": "6.35"}), 6.35)
        self.assertEqual(comparable_number({"value": "", "typ": "4.1"}), 4.1)
        for raw in ("V_CC + 0.5", "2.7 to 3.6", "20/25/45", "<1.3", "TI-RADS 3类", "增高", ""):
            self.assertIsNone(comparable_number({"value": raw}), raw)
        self.assertIsNone(comparable_number({"value": "V_CC + 0.5", "typ": "4.1"}))       # an incomparable value does not fall back to typ
        self.assertIsNone(comparable_number({"value": "5", "evidence_conflict": {"label": "x"}}))
        self.assertIsNone(comparable_number({"value": "5", "quality": "ambiguous_source"}))

    def test_compile_trend_and_endpoints_use_typed_values(self) -> None:
        from kb_pipeline.graph.compile import _numeric, _series_line, _trend, labels_for

        L = labels_for("Chinese")
        self.assertIsNone(_numeric({"value": "V_CC + 0.5", "value_num": None}))
        self.assertIsNone(_numeric({"value": "2.7 to 3.6", "value_num": None}))
        self.assertIsNone(_numeric({"value": "20/25/45", "value_num": None}))
        self.assertEqual(_numeric({"value": "6.35", "value_num": 6.35}), 6.35)
        rows = [{"concept": "总胆固醇", "value": "228", "unit": "mg/dL", "valid_from": "2023-01-01"},
                {"concept": "总胆固醇", "value": "6.1", "unit": "mmol/L", "valid_from": "2022-06-05"},
                {"concept": "总胆固醇", "value": "6.35", "unit": "mmol/L", "valid_from": "2023-09-12"}]
        self.assertEqual(_trend(rows, L), L["trend_mixed"])                       # numbers in two real units are not compared
        self.assertEqual(_trend(rows[1:], L), L["trend_up"])
        exprs = [{"concept": "V_IH", "value": "V_CC + 0.5", "unit": "V", "valid_from": "2020"},
                 {"concept": "V_IH", "value": "2.0", "unit": "V", "valid_from": "2021"}]
        self.assertEqual(_trend(exprs, L), L["trend_mixed"])                      # the expression is no longer read as 0.5
        self.assertIn("2.0 V", _series_line(exprs, L))                             # endpoints pick only comparable numbers

    def test_values_equal_respects_comparators_and_ranges_and_conflicted_facts_stay_out(self) -> None:
        from kb_pipeline.graph.reconcile import _values_equal, reconcile_facts

        self.assertFalse(_values_equal({"value": "<1.3", "value_num": 1.3}, {"value": "1.3", "value_num": 1.3}))
        self.assertTrue(_values_equal({"value": "1.3", "value_num": 1.3}, {"value": "1.30", "value_num": 1.3}))
        self.assertTrue(_values_equal({"value": "2.7 to 3.6"}, {"value": "2.7~3.6"}))
        self.assertFalse(_values_equal({"value": "2.7 to 3.6"}, {"value": "2.7"}))

        def fact(fid, doc, value, **extra):
            return {"id": fid, "subject": "张三", "subject_key": "zhangsan", "property": "综合风险指数", "symbol": "", "value": value,
                    "value_num": float(value), "unit": "", "doc_id": doc, "unit_id": f"u-{fid}", "valid_from": "2023-11-06", "conditions": {}, **extra}

        rows = [fact("a", "d1", "28"),
                fact("b", "d2", "25", evidence_conflict={"label": "综合风险指数", "text_value": "34", "model_value": "31"}, confidence="low")]
        out = reconcile_facts(rows)
        self.assertEqual((out["stats"]["conflicts"], out["stats"]["evidence_conflict_skipped"]), (0, 1))
        self.assertNotIn("conflict_group", rows[1])
        self.assertNotIn("series_key", rows[1])

    def test_concepts_keep_identity_tokens_apart_and_log_merges(self) -> None:
        from kb_pipeline.graph.concepts import build_concepts, compatible

        self.assertEqual(compatible({"label": "GENEX rs0000133 基因型"}, {"label": "GENEX rs0000131 基因型"}), "identity")
        self.assertEqual(compatible({"label": "GENEX rs0000133 基因型"}, {"label": "GENEX rs0000133 人口比例"}), "measure")
        self.assertEqual(compatible({"label": "Vitamin D3"}, {"label": "Vitamin B12"}), "identity")
        self.assertEqual(compatible({"label": "Supply voltage"}, {"label": "Operating voltage"}), "ok")
        self.assertEqual(compatible({"label": "Address setup", "symbols_text": "tAS"}, {"label": "Address setup time", "symbols_text": "t_AS"}), "ok")
        self.assertEqual(compatible({"label": "综合风险指数"}, {"label": "综合风险"}), "ok")
        facts = [
            {"id": "f1", "subject": "S", "property": "GENEX rs0000133 基因型", "symbol": "", "value": "TT", "unit": "", "doc_id": "d1"},
            {"id": "f2", "subject": "S", "property": "GENEX rs0000131 基因型", "symbol": "", "value": "GG", "unit": "", "doc_id": "d2"},
            {"id": "f3", "subject": "S", "property": "GENEX rs0000133 人口比例", "symbol": "", "value": "35%", "unit": "", "doc_id": "d1"},
            {"id": "f4", "subject": "S", "property": "Supply voltage", "symbol": "", "value": "4.1", "unit": "V", "doc_id": "d1"},
            {"id": "f5", "subject": "S", "property": "Operating voltage", "symbol": "", "value": "3.0", "unit": "V", "doc_id": "d2"},
        ]
        vectors = {"GENEX rs0000133 基因型": [1.0, 0.0, 0.0], "GENEX rs0000131 基因型": [0.999, 0.04, 0.0], "GENEX rs0000133 人口比例": [0.99, 0.0, 0.14],
                   "Supply voltage": [0.0, 1.0, 0.0], "Operating voltage": [0.0, 0.99, 0.14]}

        def embed(texts):
            return [vectors[t] for t in texts]

        asked: list[str] = []

        class Judge:
            def chat(self, prompt, max_tokens=0):
                asked.append(prompt)
                return "\n".join(f"{i + 1}. no" for i in range(prompt.count("  vs  ")))

        _, stats = build_concepts(facts, embed=embed, client=Judge(), auto_threshold=0.95, ask_threshold=0.82)
        by_key = {f["id"]: f.get("concept_key") for f in facts}
        self.assertNotEqual(by_key["f1"], by_key["f2"])                   # rs0000133 and rs0000131 are never merged nor asked about, however similar
        self.assertNotEqual(by_key["f1"], by_key["f3"])                   # genotype vs population ratio: left to the model, which says no
        self.assertEqual(by_key["f4"], by_key["f5"])
        # Two pairs blocked: rs0000133 genotype / rs0000131 genotype, and rs0000131 genotype / rs0000133 population ratio
        self.assertEqual((stats["blocked_identity"], stats["measure_to_judge"], stats["auto_merged"], stats["judged_yes"]), (2, 1, 1, 0))
        self.assertEqual(len(asked), 1)
        self.assertIn("人口比例", asked[0])
        self.assertNotIn("rs0000131", asked[0])
        self.assertEqual([(m["a"], m["b"], m["via"]) for m in stats["merges"]], [("Supply voltage", "Operating voltage", "auto")])
        self.assertEqual(stats["blocked"][0]["reason"], "identity")
        self.assertEqual({stats["blocked"][0]["a"], stats["blocked"][0]["b"]}, {"GENEX rs0000133 基因型", "GENEX rs0000131 基因型"})

    def test_units_carry_visual_value_conflicts_and_facts_from_them_are_degraded(self) -> None:
        from kb_pipeline.graph.facts import FactExtractor, comparable_number

        conflict = {"label": "综合风险指数", "text_value": "34", "model_value": "31"}
        c1 = _chunk(0, "季度报告 " * 20)
        c2 = _chunk(1, "综合风险指数 " * 20)
        c2.value_conflicts = [conflict]
        units = build_units([c1, c2], kb_id="kb_003", unit_chunks=2)
        self.assertEqual(units[0].value_conflicts, [conflict])
        self.assertEqual(Unit.from_json(units[0].to_json()).value_conflicts, [conflict])
        self.assertEqual(Unit.from_json({"unit_id": "u", "doc_id": "d"}).value_conflicts, [])
        client = _client(['{"facts": [{"subject": "张三", "property": "综合风险指数", "value": "31"}, {"subject": "张三", "property": "血糖", "value": "5.2", "unit": "mmol/L"}]}'])
        res = FactExtractor(client).extract(units[0])
        by = {f["property"]: f for f in res.facts}
        self.assertEqual((by["综合风险指数"]["confidence"], by["综合风险指数"]["evidence_conflict"]), ("low", conflict))
        self.assertIsNone(comparable_number(by["综合风险指数"]))
        self.assertEqual(by["综合风险指数"]["value"], "31")                                  # the original value and both readings are kept, they just stay out of comparisons
        self.assertNotIn("evidence_conflict", by["血糖"])
        self.assertEqual((res.stats["evidence_conflicts"], res.stats["partial"], res.stats["split"]), (1, 0, 0))

    def test_fact_extractor_splits_a_unit_after_two_truncated_responses(self) -> None:
        from kb_pipeline.graph.facts import FactExtractor, split_unit_text

        first_half, second_half = "第一段甲乙丙丁。" * 40, "第二段戊己庚辛。" * 40
        unit = _unit("u1")
        unit.text = first_half + "\n\n" + second_half
        self.assertEqual(split_unit_text(unit.text), [first_half, second_half])
        self.assertIsNone(split_unit_text("太短"))
        truncated = '{"facts": [{"subject": "S", "property": "a", "value": "1"}, {"subject": "S", "property": "b", "val'
        prompts: list[str] = []
        queue = [truncated, truncated,
                 '{"facts": [{"subject": "S", "property": "a", "value": "1"}]}',
                 '{"facts": [{"subject": "S", "property": "c", "value": "3"}]}']

        def chat(messages):
            prompts.append(messages[-1]["content"])
            return queue.pop(0)

        res = FactExtractor(_client(chat)).extract(unit)
        self.assertEqual(sorted(f["property"] for f in res.facts), ["a", "c"])
        self.assertEqual((res.calls, res.stats["split"], res.stats["partial"], res.stats["truncated"]), (4, 1, 0, 0))
        self.assertIn(first_half, prompts[2])
        self.assertNotIn(second_half, prompts[2])
        self.assertIn(second_half, prompts[3])
        # Still incomplete after splitting in half: keep the salvaged facts and record partial — the graph build completes
        # as usual, and the status card shows "facts incomplete"
        queue[:] = [truncated, truncated, truncated, '{"facts": [{"subject": "S", "property": "c", "value": "3"}]}']
        res2 = FactExtractor(_client(chat)).extract(unit)
        self.assertEqual(sorted(f["property"] for f in res2.facts), ["a", "c"])
        self.assertEqual((res2.stats["partial"], res2.stats["truncated"], res2.stats["split"]), (1, 1, 1))
        # Too short to split: kept as is after two truncations, recording partial but not split
        short = _unit("u2")
        short.text = "短单元"
        res3 = FactExtractor(_client([truncated, truncated])).extract(short)
        self.assertEqual((res3.calls, res3.stats["partial"], res3.stats["split"], len(res3.facts)), (2, 1, 0, 1))
        build_src = (Path(__file__).resolve().parents[2] / "app/kb_pipeline/graph/build.py").read_text(encoding="utf-8")
        self.assertIn('"partial_units"', build_src)
        self.assertIn('"evidence_conflict_facts"', build_src)


class FactsFixRegressionTests(_CodexAudit20260906TestsSupport, _CodexFinalTestsSupport, unittest.TestCase):
    """Regressions for problems found by past re-reviews, health checks and audits; each case's docstring names the
    source and the symptom at the time."""

    def test_f08_fact_identity_includes_the_unit(self) -> None:
        from kb_pipeline.graph.facts import fact_id

        base = {"subject": "ZK14B108L", "property": "读周期时间", "symbol": "t_RC", "value": "10", "min": "", "typ": "", "max": "",
                "conditions": {}}
        ns, ms = fact_id("u1", {**base, "unit": "ns"}), fact_id("u1", {**base, "unit": "ms"})
        self.assertNotEqual(ns, ms)
        self.assertEqual(fact_id("u1", {**base, "unit": "ns"}), fact_id("u1", {**base, "unit": " NS "}))   # case / whitespace are not a difference

    def test_f01_relative_expressions_are_never_turned_into_numbers(self) -> None:
        """F01: relative values like V_CC + 0.5 used to be recorded as 0.5 by "find the first number" (V_SS - 0.5 even
        came out positive). Only a string that parses as a single number gets *_num; ranges give lo / hi; expressions
        and text have no numeric value."""
        from kb_pipeline.graph.facts import classify_value, normalize_fact, parse_number
        from kb_pipeline.graph.vectors import spec_payload

        for t in ("V_CC + 0.5", "V_SS - 0.5", "V_DD × 0.8", "0.8 × V_DD", "V_CC-0.5[6]", "V_CC+0.2[4]", "VCC/2", "V_{CC} + 0.3"):
            self.assertEqual(classify_value(t)["kind"], "expression", t)
            self.assertIsNone(parse_number(t), t)
        scalars = {"75": 75.0, "2.7V": 2.7, "-40°C": -40.0, "1,024": 1024.0, "0.160": 0.16, "+85 °C": 85.0,
                   "≤ 10": 10.0, "±5 %": 5.0, "10 mA (typ)": 10.0, "1e-3": 0.001, "3.6 V[8]": 3.6, "−2": -2.0}
        for t, want in scalars.items():
            self.assertEqual(classify_value(t)["kind"], "scalar", t)
            self.assertEqual(parse_number(t), want, t)
        self.assertEqual(classify_value("≤ 10")["cmp"], "≤")
        self.assertEqual(classify_value("2.7 to 3.6"), {"kind": "range", "num": None, "lo": 2.7, "hi": 3.6})
        self.assertEqual(classify_value("-40 ~ +85")["hi"], 85.0)
        self.assertEqual(classify_value("20 – 45 ns")["lo"], 20.0)
        for t in ("n/a", "Max", "V_CC", "20/25/45", "75 75 57", ""):
            self.assertIsNone(parse_number(t), t)
            self.assertIn(classify_value(t)["kind"], {"text", "empty"}, t)
        norm = normalize_fact({"subject": "ZK14B108L", "property": "输入高电平电压", "symbol": "V_IH",
                               "min": "2.0", "max": "V_CC + 0.5", "unit": "V"})
        self.assertEqual((norm["min_num"], norm["max_num"]), (2.0, None))
        self.assertEqual(norm["kinds"], {"min": "scalar", "max": "expression"})
        self.assertNotIn("ranges", norm)
        rng = normalize_fact({"subject": "X", "property": "工作温度", "value": "-40 to +85", "unit": "°C"})
        self.assertIsNone(rng["value_num"])
        self.assertEqual(rng["ranges"], {"value": [-40.0, 85.0]})
        payload = spec_payload({**norm, "id": "f1"}, kb_id="kb_001", source_collection="c", graph_version="v")
        self.assertEqual(payload["kinds"], {"min": "scalar", "max": "expression"})
        self.assertNotIn("max_num", payload)          # None fields stay out of the payload: filters will not treat the expression as 0.5

    def test_only_grouped_commas_are_thousands_separators(self) -> None:
        """Commas not followed by groups of three digits (a decimal comma, two comma-separated values) used to be
        dropped to read one large number: 12,50 -> 1250. Now only thousands-separator notation counts as one number;
        anything else is text, has no numeric value and takes no part in trends or reconciliation."""
        from kb_pipeline.graph.facts import classify_value, comparable_number, normalize_fact, parse_number
        from kb_pipeline.graph.reconcile import _values_equal

        for t, want in {"1,024": 1024.0, "12,345.6": 12345.6, "1,234,567": 1234567.0, "-1,500 mA": -1500.0, "≤ 1,024": 1024.0,
                        "１，０２４": 1024.0}.items():
            self.assertEqual(classify_value(t)["kind"], "scalar", t)
            self.assertEqual(parse_number(t), want, t)
        for t in ("12,50", "2,86", "34,57", "172,2576", "12,26,9", "0,5", "1,0245", "1234,567", "1.024,5", "12,50 mm"):
            self.assertEqual(classify_value(t), {"kind": "text", "num": None}, t)
        self.assertEqual(classify_value("1,024 to 2,048"), {"kind": "range", "num": None, "lo": 1024.0, "hi": 2048.0})
        self.assertEqual(classify_value("2,5 - 3,5")["kind"], "text")
        fact = normalize_fact({"subject": "X", "property": "参数", "value": "12,50", "min": "2,86", "ref_min": "1,5", "ref_max": "1,500"})
        self.assertEqual((fact["value"], fact["value_num"], fact["min_num"], fact["kinds"]), ("12,50", None, None, {"value": "text", "min": "text"}))
        self.assertEqual((fact["ref_min_num"], fact["ref_max_num"]), (None, 1500.0))
        self.assertIsNone(comparable_number(fact))
        self.assertFalse(_values_equal({"value": "12,50"}, {"value": "1250"}))

    def test_a_currency_sign_in_front_of_a_plain_number_is_its_unit(self) -> None:
        """Once values were kept as written, "$85,000" as a whole was no longer a number: amounts became text and could
        be neither compared nor put in a series. A currency sign directly in front of a number is parsed as usual and
        the currency goes to the unit field when that is empty; when a magnitude abbreviation follows ($85K, $1.05M),
        85 is not the number and the value remains text."""
        from kb_pipeline.graph.facts import classify_value, comparable_number, normalize_fact

        for t, want in {"$85,000": 85000.0, "¥245.00": 245.0, "A$21,050": 21050.0, "US$ 1,200": 1200.0, "€99": 99.0,
                        "$350,000 (2025)": 350000.0}.items():
            info = classify_value(t)
            self.assertEqual((info["kind"], info["num"]), ("scalar", want), t)
        self.assertEqual(classify_value("~$238,000"), {"kind": "scalar", "num": 238000.0, "currency": "$", "cmp": "~"})
        self.assertEqual(classify_value("$310,000-$485,000"), {"kind": "range", "num": None, "lo": 310000.0, "hi": 485000.0, "currency": "$"})
        for t in ("$85K", "$1.05M", "$10亿", "约$200,000", "$12,50", "${HOME}_dir", "$defs.item.$ref", "$", "¥?/kg", "高级 $310,000-$485,000"):
            self.assertEqual(classify_value(t), {"kind": "text", "num": None}, t)
        fact = normalize_fact({"subject": "某岗位", "property": "平均薪资", "value": "$196,489"})
        self.assertEqual((fact["value"], fact["value_num"], fact["unit"], fact["unit_canonical"], fact["kinds"]),
                         ("$196,489", 196489.0, "$", "USD", {"value": "scalar"}))
        self.assertEqual(comparable_number(fact), 196489.0)
        kept = normalize_fact({"subject": "某商品", "property": "单价", "value": "¥245.00", "unit": "RMB/件"})
        self.assertEqual((kept["value_num"], kept["unit"]), (245.0, "RMB/件"))                 # a unit that was written stays

    def test_value_fields_are_not_rewritten_as_entity_names(self) -> None:
        """Values, units, conditions and notes used to go through the LaTeX restoration of entity names: every $ or
        backslash was deleted, rewriting currencies, variable names and paths. Now these fields stay as in the source
        and only recognisable $...$ fragments are restored; subject, property and symbol are still normalised like
        entity names."""
        from kb_pipeline.graph.facts import classify_value, normalize_fact

        def value_of(raw: str) -> str:
            return normalize_fact({"subject": "S", "property": "p", "value": raw})["value"]

        for raw in ("$85K", "$10亿", "A$21,050", "$85K – $120K", "${HOME}_dir", "${DB_HOST}:${DB_PORT}", "{name}_{id}.log",
                    "C:\\Users\\foo_bar", "CRM系统\\ERP", "$defs.item.$ref"):
            self.assertEqual(value_of(raw), raw, raw)
        money = normalize_fact({"subject": "某公司", "property": "薪资范围", "min": "$85K", "max": "$120K", "unit": "$",
                                "conditions": {"币种": "US$"}, "note": "含 $5K 签约奖金"})
        self.assertEqual((money["min"], money["max"], money["unit"], money["unit_canonical"]), ("$85K", "$120K", "$", "USD"))
        self.assertEqual((money["conditions"], money["note"]), ({"币种": "US$"}, "含 $5K 签约奖金"))
        self.assertEqual((money["min_num"], money["kinds"]), (None, {"min": "text", "max": "text"}))      # a number with a currency sign is not a bare number
        # a paired $...$ fragment that really holds LaTeX notation is still restored; one with commands that are not
        # all recognised stays as it is, rather than losing its backslashes into a run of letters and digits
        self.assertEqual(value_of("$V_{CC}$ + 0.5"), "VCC + 0.5")
        self.assertEqual(value_of("$\\overline{\\mathrm{CE}}$ low"), "CE# low")
        self.assertEqual(value_of("$5\\%$"), "5%")
        self.assertEqual(value_of("$\\frac{1}{2} V_{CC}$"), "$\\frac{1}{2} V_{CC}$")
        self.assertEqual(value_of("V_{CC} + 0.3"), "V_{CC} + 0.3")
        self.assertEqual(classify_value(value_of("V_{CC} + 0.3"))["kind"], "expression")
        # full-width characters, surrounding quotes and whitespace are normalised as before
        self.assertEqual((value_of("  “１２．５　ｍＡ”  "), value_of('"a   b"')), ("12.5 mA", "a b"))
        # an exponent is not merged into its base: a superscript 9 after 10 and $10^{9}$ are both written 10^9, not 109
        blood = normalize_fact({"subject": "S", "property": "白细胞计数", "value": "5.2", "unit": "10⁹/L", "ref_max": "$10^{9}$"})
        self.assertEqual((blood["unit"], blood["unit_canonical"], blood["ref_max"], blood["ref_max_num"]), ("10^9/L", "10^9/L", "10^9", None))
        self.assertEqual(classify_value("10⁹"), {"kind": "text", "num": None})
        self.assertEqual(value_of("5 m²"), "5 m2")
        # name fields: LaTeX notation is restored so that they match entity names
        named = normalize_fact({"subject": "$t_{\\text{AS}}$", "property": "Setup time", "symbol": "$V_{CC}$", "value": "1"})
        self.assertEqual((named["subject"], named["symbol"]), ("tAS", "VCC"))

    def test_f03_malformed_llm_output_is_retried_uncached_and_fails_the_unit(self) -> None:
        """F03: an HTTP success whose content was not the requested JSON used to be cached anyway and counted as a
        "successful empty result". Now: ask again with a correction prompt; if still malformed raise
        LLMMalformedResponse, do not cache, do not count towards the circuit breaker; an old malformed response in the
        cache is discarded and re-requested when it fails validation; the facts phase does not count as complete while
        any unit failed extraction, unless partial publication is explicitly accepted."""
        from kb_pipeline.graph.build import facts_phase_gate
        from kb_pipeline.graph.facts import FactExtractor, facts_fingerprint, facts_response_ok
        from kb_pipeline.graph.llm import ChatClient, LLMCache, LLMMalformedResponse, LLMSpec, cache_key

        spec = LLMSpec(name="m", base_url="http://x", api_key="", model_id="m")
        good = '{"facts": [{"subject": "X", "property": "p", "value": "1", "unit": "V"}]}'

        def client_with(replies, cache):
            seen: list[list[dict]] = []

            def chat(spec, messages, **kw):
                seen.append(list(messages))
                return replies[min(len(seen) - 1, len(replies) - 1)]
            return ChatClient(spec, cache=cache, chat=chat, attempts=3, workers=1), seen

        with tempfile.TemporaryDirectory() as tmp:
            cache = LLMCache(Path(tmp) / "c.sqlite")
            client, seen = client_with(["I cannot produce that.", good], cache)
            out = client.chat("q1", validate=facts_response_ok)
            self.assertEqual(out, good)
            self.assertEqual(len(seen), 2)
            self.assertEqual([m["role"] for m in seen[1]], ["user", "assistant", "user"])       # carries the previous reply and the correction prompt
            self.assertEqual((client.stats["retries"], client.stats["malformed"], cache.count()), (1, 0, 1))
            # Malformed every time: raises, not cached, not counted as consecutive failures (that is the circuit breaker's signal)
            client, seen = client_with(["nope", "still nope", "nope again"], cache)
            with self.assertRaises(LLMMalformedResponse):
                client.chat("q2", validate=facts_response_ok)
            self.assertEqual((len(seen), client.stats["malformed"], client.stats["consecutive_failures"], cache.count()), (3, 1, 0, 1))
            # An old cache holds a malformed response: discarded and re-requested when it fails validation
            client, seen = client_with([good], cache)
            key = cache_key(spec, [{"role": "user", "content": "q3"}], client._params(client.max_tokens, client.temperature))
            cache.put(key, "m", "garbage from an earlier run")
            self.assertEqual(client.chat("q3", validate=facts_response_ok), good)
            self.assertEqual((client.stats["cache_evicted"], len(seen), cache.get(key)), (1, 1, good))
            # Calls without validate behave as before
            client, seen = client_with(["free text"], cache)
            self.assertEqual(client.chat("q4"), "free text")
            # Fact extraction treats it as a failed unit, not as an empty result
            client, _ = client_with(["not json", "not json either"], LLMCache(None))
            unit = SimpleNamespace(unit_id="u-1", text="| a | b |", section_label="s", rel_path="d.pdf", kind="body",
                                   block_types=["table"], doc_id="d")
            with self.assertRaises(LLMMalformedResponse):
                FactExtractor(client).extract(unit, document="d.pdf", subjects=["X"])
            cache.close()
        facts_phase_gate([], partial_ok=False)
        failed = [(SimpleNamespace(unit_id="u-1"), LLMMalformedResponse("x")), (SimpleNamespace(unit_id="u-2"), RuntimeError("boom"))]
        with self.assertRaisesRegex(RuntimeError, "failed for 2 units.*u-1, u-2.*KB_GRAPH_FACTS_PARTIAL_OK"):
            facts_phase_gate(failed, partial_ok=False)
        facts_phase_gate(failed, partial_ok=True)
        # 2026-09-29 audit: a unit the provider rejected for its content is a loss inherent to the corpus, the same
        # rule as in the extraction phase; it does not fail the facts phase
        from kb_pipeline.graph.llm import LLMInputRejected

        rejected = [(SimpleNamespace(unit_id="u-3"), LLMInputRejected("Content Exists Risk"))]
        facts_phase_gate(rejected, partial_ok=False)
        with self.assertRaisesRegex(RuntimeError, "failed for 2 units.*u-1, u-2"):
            facts_phase_gate(failed + rejected, partial_ok=False)          # real failures still stop the phase, rejections are not counted
        build_src = _repo_file("app/kb_pipeline/graph/build.py")
        self.assertIn('"rejected_units": len(rejected_units)', build_src)
        # 走规则抽取的单元(代码、配置、结构化 markdown)不送模型抽事实
        self.assertIn("todo_units = [u for u in units if not det.wants(u) and u.unit_id not in table_ids and wants_facts(u, kinds.get(u.unit_id))]", build_src)
        self.assertIn("facts-v4", _repo_file("app/kb_pipeline/graph/facts.py"))     # 09-07 加轴 / 边界字段升到 v3;数值与取值清洗改了升到 v4
        self.assertNotEqual(facts_fingerprint(spec), "")
        src = _repo_file("app/kb_pipeline/graph/build.py")
        self.assertIn('facts_phase_gate(failed, partial_ok=_env_flag("KB_GRAPH_FACTS_PARTIAL_OK"))', src)
        self.assertIn("validate=facts_response_ok", _repo_file("app/kb_pipeline/graph/facts.py"))

    def test_f02_unverified_glued_values_never_become_trusted_numbers(self) -> None:
        """An unverified glued value reaching fact extraction: that fact's number is cleared, the field kind becomes
        ambiguous, quality=ambiguous_source; the payload and the chunk list carry the mark and the front end shows a
        badge; the parser route version was bumped so that historical files get re-parsed."""
        from kb_pipeline.graph.facts import FactExtractor, mark_ambiguous_source
        from kb_pipeline.graph.llm import ChatClient, LLMCache, LLMSpec
        from kb_pipeline.graph.units import ChunkRef, Unit, build_units
        from kb_pipeline.graph.vectors import spec_payload

        fact = {"subject": "ZK14B108L", "property": "平均电流", "symbol": "I_CC1", "value": "", "min": "", "typ": "", "max": "757557",
                "unit": "mA", "max_num": 757557.0, "kinds": {"max": "scalar"}, "conditions": {}}
        self.assertTrue(mark_ambiguous_source(fact, ["757557"]))
        self.assertEqual((fact["max_num"], fact["kinds"]["max"], fact["quality"]), (None, "ambiguous", "ambiguous_source"))
        self.assertFalse(mark_ambiguous_source({"value": "75", "value_num": 75.0}, ["757557"]))
        payload = spec_payload({**fact, "id": "f1"}, kb_id="kb_001", source_collection="c", graph_version="v")
        self.assertEqual(payload["quality"], "ambiguous_source")
        self.assertNotIn("max_num", payload)
        # Chunk payload → unit → extraction
        ref = ChunkRef(point_id="p1", chunk_uid="c1", doc_id="d1", content_version="v1", chunk_index=0, block_id="b1",
                       block_type="table", section_path=["电气特性"], text="| I_CC1 | 757557 | mAmAmA |", n_tokens=20,
                       ambiguous_values=["757557"])
        units = build_units([ref], kb_id="kb_001", unit_chunks=1)
        self.assertEqual(units[0].ambiguous_values, ["757557"])
        self.assertEqual(Unit.from_json(units[0].to_json()).ambiguous_values, ["757557"])
        reply = '{"facts": [{"subject": "ZK14B108L", "property": "平均电流", "symbol": "I_CC1", "max": "757557", "unit": "mA"}, {"subject": "ZK14B108L", "property": "电源", "typ": "3.0", "unit": "V"}]}'
        client = ChatClient(LLMSpec(name="m", base_url="http://x", api_key="", model_id="m"), cache=LLMCache(None),
                            chat=lambda *a, **k: reply, attempts=1, workers=1)
        res = FactExtractor(client).extract(units[0], document="d.pdf", subjects=["ZK14B108L"])
        self.assertEqual(res.stats["ambiguous"], 1)
        by_symbol = {f.get("symbol") or f["property"]: f for f in res.facts}
        self.assertEqual((by_symbol["I_CC1"]["max_num"], by_symbol["I_CC1"]["quality"]), (None, "ambiguous_source"))
        self.assertEqual((by_symbol["电源"]["typ_num"], by_symbol["电源"].get("quality")), (3.0, None))
        # Payload, console, parser version
        self.assertIn('"table_flags": block.metadata.get("table_flags") or None', _repo_file("app/kb_pipeline/pipeline/parse_job.py"))
        self.assertIn('"table_flags": pl.get("table_flags") or None', _repo_file("app/kb_server/service.py"))
        js = _repo_file("app/kb_server/static/app.js")
        self.assertIn("function tableBadge(c)", js)
        self.assertIn('⚠ ${t("表格粘连")}', js)
        self.assertIn('chip("ambiguous", t("表格粘连"), ambiguous)', js)
        html = _repo_file("app/kb_server/static/index.html")
        self.assertIn(".pv-badge.warn", html)
        self.assertRegex(html, r"app\.js\?v=\d{8}-\d+")
        self.assertEqual(parser_profile_for_path(Path("x.pdf")), "pdf-mineru-table-vlm-v15")
        self.assertEqual(parser_profile_for_path(Path("x.docx")), "docx-mineru-ooxml-vlm-v10")
        for name in ("mineru_pdf.py", "mineru_docx.py"):
            self.assertIn("table_ambiguity_flags(", _repo_file(f"app/kb_pipeline/parsers/{name}"), name)
        for name in ("pdf_enhanced.py", "docx_enhanced.py"):
            self.assertIn("repair_ambiguous_tables(", _repo_file(f"app/kb_pipeline/parsers/{name}"), name)

    def test_f03_truncated_fact_json_is_salvaged_and_retried_with_a_bigger_budget(self) -> None:
        """Real box: the fact JSON of a large table was cut off midway by max_tokens. A truncation is a valid prefix,
        not malformed: salvage the complete objects and ask again with double the budget; if still truncated keep the
        salvaged part and record truncated."""
        from kb_pipeline.graph.facts import FactExtractor, facts_response_ok, parse_facts_json, parse_facts_response
        from kb_pipeline.graph.llm import ChatClient, LLMCache, LLMSpec

        full = ('{"facts": [{"subject": "X", "property": "a", "value": "1", "unit": "V"}, '
                '{"subject": "X", "property": "b", "value": "2", "unit": "V"}, {"subject": "X", "property": "c", "value": "3", "unit": "V"}]}')
        cut = full[:full.index('"property": "c"') + 10]           # the third one is cut off halfway
        parsed = parse_facts_response(cut)
        self.assertEqual((len(parsed["facts"]), parsed["malformed"], parsed["truncated"]), (2, 0, True))
        self.assertEqual(parse_facts_response(full)["truncated"], False)
        self.assertEqual(parse_facts_response('{"facts": []}'), {"facts": [], "malformed": 0, "truncated": False})
        self.assertEqual(parse_facts_response("I cannot help."), {"facts": [], "malformed": 1, "truncated": False})
        self.assertEqual(parse_facts_json(cut), (parsed["facts"], 0))
        self.assertTrue(facts_response_ok(cut))
        self.assertFalse(facts_response_ok("prose only"))
        spec = LLMSpec(name="m", base_url="http://x", api_key="", model_id="m")
        unit = SimpleNamespace(unit_id="u-1", text="| a | b |", section_label="s", rel_path="d.pdf", kind="body",
                               block_types=["table"], doc_id="d", ambiguous_values=[])

        def extractor(replies):
            seen: list[int] = []

            def chat(spec, messages, max_tokens=None, **kw):
                seen.append(max_tokens)
                return replies[min(len(seen) - 1, len(replies) - 1)]
            return FactExtractor(ChatClient(spec, cache=LLMCache(None), chat=chat, attempts=1, workers=1), max_tokens=1000), seen

        ex, seen = extractor([cut, full])
        res = ex.extract(unit, document="d.pdf", subjects=["X"])
        self.assertEqual((len(res.facts), res.stats["truncated"], seen), (3, 0, [1000, 2000]))    # double the budget gets the complete one
        ex, seen = extractor([cut, cut])
        res = ex.extract(unit, document="d.pdf", subjects=["X"])
        self.assertEqual((len(res.facts), res.stats["truncated"]), (2, 1))                       # still truncated: keep the two salvaged
        src = _repo_file("app/kb_pipeline/graph/build.py")
        self.assertIn('run_phase("facts", facts, retries=max(2, int(retry_cfg["retries"])))', src)
        self.assertIn('"truncated_units"', src)
        self.assertIn("FACTS_MAX_TOKENS = 12288", _repo_file("app/kb_pipeline/graph/facts.py"))
        self.assertIn("_MALFORMED_ATTEMPTS = 3", _repo_file("app/kb_pipeline/graph/llm.py"))

    def test_unit_canonicalisation_keeps_case_that_changes_magnitude(self) -> None:
        from kb_pipeline.graph.facts import canonical_unit

        self.assertEqual(canonical_unit("MW"), "MW")          # megawatt is not milliwatt
        self.assertEqual(canonical_unit("mHz"), "mHz")        # millihertz is not megahertz
        self.assertEqual(canonical_unit("MV"), "MV")
        self.assertEqual(canonical_unit("mw"), "mW")          # all-lowercase is just a spelling difference
        self.assertEqual(canonical_unit("MHZ"), "MHz")
        self.assertEqual(canonical_unit("mmHg"), "mmHg")
        self.assertEqual(canonical_unit("MMOL/L"), "mmol/L")  # what follows the first letter is not a unit: an all-caps writing habit, normalised as usual
        self.assertEqual(canonical_unit("毫瓦"), "mW")
        self.assertEqual(canonical_unit("unknownunit"), "unknownunit")

    def test_fact_budget_cap_is_reported_as_partial(self) -> None:
        from kb_pipeline.graph.facts import FactExtractor
        from kb_pipeline.graph.units import Unit

        unit = Unit(unit_id="u", doc_id="d", rel_path="d.pdf", section_path=["S"], block_ids=["b"], chunk_uids=["c"],
                    point_ids=["p"], n_tokens=10, text="t", order=0)
        body = '{"facts": [' + ", ".join(f'{{"subject": "S", "property": "p{i}", "value": "{i}"}}' for i in range(4)) + "]}"
        res = FactExtractor(_fake_chat_client([body]), max_facts=2).extract(unit)
        self.assertEqual((len(res.facts), res.stats["capped"], res.stats["partial"], res.stats["truncated"]), (2, 1, 1, 0))
        res2 = FactExtractor(_fake_chat_client([body]), max_facts=8).extract(unit)
        self.assertEqual((len(res2.facts), res2.stats["capped"], res2.stats["partial"]), (4, 0, 0))

    def test_unit_key_keeps_case_all_the_way_down(self) -> None:
        from kb_pipeline.graph.compile import _trend, labels_for
        from kb_pipeline.graph.concepts import _unit_family
        from kb_pipeline.graph.reconcile import unit_of
        from kb_pipeline.measure_units import unit_key
        from kb_pipeline.parsers.visual_blocks import reconcile_visual_facts

        self.assertNotEqual(unit_key("MW"), unit_key("mW"))
        self.assertEqual(unit_key("MMOL/L"), unit_key("mmol/L"))
        self.assertEqual(unit_of({"unit": "MW"}), "MW")
        self.assertNotEqual(_unit_family("MHz"), _unit_family("mHz"))
        self.assertEqual(reconcile_visual_facts(["功率 1000000000 mW"], "功率 1 MW")[::2], (["功率 1000000000 mW"], []))
        L = labels_for("Chinese")
        rows = [{"concept": "功率", "value": "2", "unit": "mW", "valid_from": "2024"},
                {"concept": "功率", "value": "1", "unit": "MW", "valid_from": "2025"}]
        self.assertEqual(_trend(rows, L), L["trend_mixed"])                         # numbers in different units are not compared


class WideTableFactsPromptTests(unittest.TestCase):
    def test_facts_prompt_uses_expanded_rows_when_enabled(self) -> None:
        from unittest.mock import patch

        from kb_pipeline.graph import tabletext
        from kb_pipeline.graph.facts import render_facts_prompt
        from kb_pipeline.graph.units import Unit

        text = ("SHEET: 表\nROWS: 1-2\nHEADER: 功能 | 描述 | 协作甲 | 协作乙 | 协作丙 | 协作丁\n"
                "标签 |  | 1 | 1 | 0 | 1")
        unit = Unit(unit_id="u", doc_id="d", rel_path="a.xlsx", section_path=[], block_ids=[], chunk_uids=[], point_ids=[], n_tokens=10, text=text)
        with patch.object(tabletext, "FACTS_EXPAND_WIDE_TABLES", True), patch("kb_pipeline.graph.facts.FACTS_EXPAND_WIDE_TABLES", True):
            self.assertIn("功能: 标签 | 协作甲: 1 | 协作乙: 1 | 协作丙: 0 | 协作丁: 1", render_facts_prompt(unit, document="a", subjects=["协作丙"]))
        with patch("kb_pipeline.graph.facts.FACTS_EXPAND_WIDE_TABLES", False):
            prompt = render_facts_prompt(unit, document="a", subjects=["协作丙"])
            self.assertIn("标签 |  | 1 | 1 | 0 | 1", prompt)
            self.assertNotIn("协作甲: 1", prompt)


class CurrencyTests(unittest.TestCase):
    """2026-10-24 破坏性改版(方案附录第 3 条与第 7 条):关系与事实同一套时间与新旧字段。
    关系带观察时间(来源文档的日期)和原文写明的起止;只有显式声明单值的关系才判新旧;
    事实序列里最新那一档是现行值;版本串不再写进有效期;按日期组织的资料里叙述段也抽状态事实。"""

    def _units(self):
        u1 = _unit("u1", doc="kb:1"); u1.doc_date = "2025-03-01"
        u2 = _unit("u2", doc="kb:2", order=1); u2.doc_date = "2026-01-10"
        u3 = _unit("u3", doc="kb:3", order=2)
        return {u.unit_id: u for u in (u1, u2, u3)}

    def test_only_things_with_changing_states_have_their_relations_superseded(self) -> None:
        """一般概念上的不同说法可以并存。只有具备可变状态的实例才按时间判断关系的新旧。"""
        from kb_pipeline.graph.currency import relation_currency

        units = self._units()
        rels = [{"source_key": "某病", "target_key": "部位a", "predicate": "located_in", "unit_ids": ["u1"]},
                {"source_key": "某病", "target_key": "部位b", "predicate": "located_in", "unit_ids": ["u2"]},
                {"source_key": "张三", "target_key": "城市a", "predicate": "located_in", "unit_ids": ["u1"]},
                {"source_key": "张三", "target_key": "城市b", "predicate": "located_in", "unit_ids": ["u2"]}]
        ents = [{"key": "某病", "upper": "concept", "kind": "instance"}, {"key": "张三", "upper": "agent", "kind": "instance"}]
        stats = relation_currency(rels, units, predicates=[{"name": "located_in", "multivalued": False}], entities=ents)
        self.assertEqual([bool(r.get("superseded_by")) for r in rels], [False, False, True, False])
        self.assertEqual((stats["superseded"], stats["stateless_sources_skipped"]), (1, 2))

    def test_only_stated_periods_of_a_state_lapse(self) -> None:
        """单次事件的发生时间不等于有效期。只有实例明确的一段状态才会在止期后失效。"""
        from kb_pipeline.graph.currency import is_interval, mark_fact_lapses, relation_currency

        self.assertTrue(is_interval("2014", "2016"))
        self.assertTrue(is_interval("", "2025-06"))                     # 只写了止期:合同到 2025 年 6 月
        self.assertFalse(is_interval("2025-10-18", "2025-10-18"))       # 某天做的一次检查
        self.assertFalse(is_interval("2025", "2025"))
        self.assertFalse(is_interval("2021", ""))
        self.assertFalse(is_interval("", "v2.0"))
        units = self._units()
        rels = [{"source_key": "张三", "target_key": "公司a", "predicate": "works_at", "unit_ids": ["u1"],
                 "valid_from": "2014", "valid_until": "2016"},
                {"source_key": "张三", "target_key": "机构b", "predicate": "related_to", "unit_ids": ["u1"],
                 "valid_from": "2025-03-01", "valid_until": "2025-03-01"},
                {"source_key": "张三", "target_key": "年会", "predicate": "attends", "unit_ids": ["u1"],
                 "valid_from": "2025-03-01", "valid_until": "2025-03-03"},
                {"source_key": "张三", "target_key": "公司a", "predicate": "related_to", "unit_ids": ["u1"]},
                {"source_key": "张三", "target_key": "检查c", "predicate": "related_to", "unit_ids": ["u1"],      # 兜底关系不说状态
                 "valid_from": "2026-08-31", "valid_until": "2026-09-16"},
                {"source_key": "某病", "target_key": "某项风险", "predicate": "has_biomarker", "unit_ids": ["u1"],   # 报告的检测期
                 "valid_from": "2025-10-11", "valid_until": "2025-12-11"}]
        ents = [{"key": "张三", "upper": "agent"}, {"key": "公司a", "upper": "agent"}, {"key": "机构b", "upper": "agent"},
                {"key": "年会", "upper": "event"}, {"key": "检查c", "upper": "process"}, {"key": "某病", "upper": "concept"},
                {"key": "某项风险", "upper": "property"}]
        stats = relation_currency(rels, units, entities=ents)
        self.assertEqual([r["lapses"] for r in rels], [True, False, False, False, False, False])
        self.assertEqual(stats["relations_lapsing"], 1)
        facts = [{"subject_key": "张三", "valid_from": "2024-01", "valid_until": "2024-06"},
                 {"subject_key": "年会", "valid_from": "2025-03-01", "valid_until": "2025-03-03"},
                 {"subject_key": "某病", "valid_from": "2020", "valid_until": "2022"},
                 {"subject_key": "张三", "valid_from": "2024-07-26", "valid_until": "2024-07-26"},
                 {"subject_key": "张三", "valid_from": "2024-07-26"}]
        self.assertEqual(mark_fact_lapses(facts, ents), 1)
        self.assertEqual([f["lapses"] for f in facts], [True, False, False, False, False])

    def test_merged_relations_do_not_make_a_period_out_of_two_dates(self) -> None:
        """两次检查各一条起止相同的关系,归并后不能变成「从第一次到第二次」的一段状态(过了第二次就被当成已结束)。"""
        from kb_pipeline.graph.resolution import merged_period

        self.assertEqual(merged_period({"valid_from": "2024-07-26", "valid_until": "2024-07-26"},
                                       {"valid_from": "2025-06-12", "valid_until": "2025-06-12"}), ("2024-07-26", "2024-07-26"))
        self.assertEqual(merged_period({"valid_from": "2021"}, {"valid_until": "2025-06"}), ("2021", "2025-06"))
        self.assertEqual(merged_period({"valid_from": "2014", "valid_until": "2016"},
                                       {"valid_from": "2015", "valid_until": "2018"}), ("2014", "2018"))
        self.assertEqual(merged_period({}, {"valid_from": "2020-05-01", "valid_until": "2020-05-01"}), ("2020-05-01", "2020-05-01"))
        self.assertEqual(merged_period({}, {}), ("", ""))

    def test_single_valued_relations_are_superseded_by_the_newest_statement(self) -> None:
        from kb_pipeline.graph.currency import relation_currency
        from kb_pipeline.graph.vectors import relation_id

        units = self._units()
        rels = [
            {"source_key": "张三", "target_key": "组a", "predicate": "reports_to", "unit_ids": ["u1"]},
            {"source_key": "张三", "target_key": "组b", "predicate": "reports_to", "unit_ids": ["u2"]},
            {"source_key": "张三", "target_key": "组c", "predicate": "reports_to", "unit_ids": ["u3"]},      # 没有日期:不判
            {"source_key": "张三", "target_key": "项目x", "predicate": "works_on", "unit_ids": ["u1"]},
            {"source_key": "张三", "target_key": "项目y", "predicate": "works_on", "unit_ids": ["u2"]},
            {"source_key": "李四", "target_key": "组a", "predicate": "reports_to", "unit_ids": ["u1"], "valid_from": "2026-05"},
            {"source_key": "李四", "target_key": "组b", "predicate": "reports_to", "unit_ids": ["u2"]},
        ]
        stats = relation_currency(rels, units, predicates=[{"name": "reports_to", "multivalued": False}, {"name": "works_on"}])
        by = {(r["source_key"], r["target_key"]): r for r in rels}
        self.assertEqual((by[("张三", "组a")]["observed_first"], by[("张三", "组a")]["observed_last"]), ("2025-03-01", "2025-03-01"))
        self.assertEqual(by[("张三", "组a")]["superseded_by"], relation_id("张三", "组b", "reports_to"))
        self.assertEqual(by[("张三", "组a")]["superseded_on"], "2026-01-10")
        self.assertNotIn("superseded_by", by[("张三", "组b")])
        self.assertNotIn("superseded_by", by[("张三", "组c")])             # 没有时间的说法不凭空判旧
        self.assertNotIn("superseded_by", by[("张三", "项目x")])           # 多值关系不判新旧
        # 原文写明的起点优先于观察时间:李四 2026-05 起向组 a 汇报,比 2026-01 的文档新
        self.assertEqual(by[("李四", "组b")]["superseded_by"], relation_id("李四", "组a", "reports_to"))
        self.assertEqual((stats["superseded"], stats["single_valued_groups"]), (2, 2))

    def test_fact_series_mark_the_newest_value_current(self) -> None:
        from kb_pipeline.graph.reconcile import reconcile_facts

        facts = [{"id": f"f{y}", "subject": "张三", "property": "体重", "value": str(60 + i), "unit": "kg", "doc_id": f"d{y}",
                  "valid_from": str(y)} for i, y in enumerate((2023, 2024, 2025))]
        reconcile_facts(facts)
        by = {f["id"]: f for f in facts}
        self.assertEqual((by["f2023"]["superseded_by"], by["f2023"]["superseded_on"]), ("f2024", "2024"))
        self.assertEqual(by["f2024"]["superseded_by"], "f2025")
        self.assertNotIn("superseded_by", by["f2025"])

    def test_versions_stay_out_of_the_validity_dates(self) -> None:
        from kb_pipeline.graph.facts import link_facts

        unit = _unit("u1", doc="kb:1"); unit.axis = "rev *L"; unit.doc_date = "2024-05"
        dated = _unit("u2", doc="kb:2", order=1); dated.axis = "2025-01-02"; dated.doc_date = "2025-01-02"
        facts = [{"unit_id": "u1", "subject": "x", "property": "p", "value": "1"},
                 {"unit_id": "u2", "subject": "x", "property": "p", "value": "2"}]
        link_facts(facts, [], units_by_id={"u1": unit, "u2": dated})
        self.assertEqual((facts[0]["axis"], facts[0]["valid_from"], facts[0]["observed"]), ("rev *L", "2024-05", "2024-05"))
        self.assertEqual(facts[1]["valid_from"], "2025-01-02")

    def test_narrative_state_facts_only_for_prose_units_and_with_the_document_date(self) -> None:
        from kb_pipeline.graph.facts import render_facts_prompt, wants_narrative_facts

        prose = _unit("u1"); prose.text = "项目负责人改为王五,上线日期推迟到下月底。" * 10; prose.n_tokens = 120; prose.doc_date = "2026-03-14"
        table = _unit("u2"); table.text = "| a | b |\n| 1 | 2 |\n| 3 | 4 |"; table.block_types = ["table"]; table.n_tokens = 120
        short = _unit("u3"); short.text = "好的"; short.n_tokens = 3
        self.assertTrue(wants_narrative_facts(prose))
        self.assertFalse(wants_narrative_facts(table))                     # 表格单元走表格事实
        self.assertFalse(wants_narrative_facts(short))
        self.assertFalse(wants_narrative_facts(prose, "boilerplate"))
        text = render_facts_prompt(prose, document="周报.docx", subjects=["项目X"], narrative=True)
        self.assertIn("Document date, if known: 2026-03-14", text)
        self.assertIn("state facts", text)
        self.assertIn("Document version, if known: (unknown)", render_facts_prompt(table, document="t.xlsx", subjects=[]))
        build = _repo_file("app/kb_pipeline/graph/build.py")
        self.assertIn('if str((graph.get("profile") or {}).get("axis") or "") == "date":', build)    # 只对按日期组织的库开


class SeriesConditionTests(unittest.TestCase):
    """合成的跨年记录验证条件分组:记录上下文不能被随意丢弃,真正区分不同取值的条件应保留。"""

    def _fact(self, fid, doc, when, value, conditions):
        return {"id": fid, "subject": "某甲", "subject_key": "某甲", "property": "血红蛋白", "concept_key": "c-hb", "value": value,
                "value_num": float(value), "unit": "g/L", "valid_from": when, "doc_id": doc, "conditions": conditions}

    def test_unclassified_cross_document_conditions_are_preserved(self) -> None:
        from kb_pipeline.graph.reconcile import reconcile_facts

        facts = [self._fact("f1", "d1", "2024", "147", {"检查项目": "血红蛋白", "科室": "血常规", "检查者": "某医生"}),
                 self._fact("f2", "d2", "2025", "159", {}),
                 self._fact("f3", "d3", "2026", "154", {"检查类型": "血常规"}),
                 self._fact("f4", "d3", "2024-07", "147", {"时间": "2024/07"})]       # 历年对照表:时期是轴,不是条件
        out = reconcile_facts(facts)
        self.assertIsNone(facts[0].get("superseded_by"))
        self.assertIsNone(facts[2].get("superseded_by"))
        self.assertNotEqual(facts[0].get("series_key"), facts[1].get("series_key"))
        self.assertEqual(out["stats"]["conflicts"], 0)

    def test_a_condition_that_tells_values_apart_still_splits(self) -> None:
        from kb_pipeline.graph.reconcile import reconcile_facts

        facts = [self._fact("f1", "d1", "2024", "3.3", {"温度": "25°C"}), self._fact("f2", "d1", "2024", "3.1", {"温度": "85°C"}),
                 self._fact("f3", "d2", "2025", "3.3", {"温度": "25°C"}), self._fact("f4", "d2", "2025", "3.0", {"温度": "85°C"})]
        reconcile_facts(facts)
        self.assertEqual(facts[0]["series_key"], facts[2]["series_key"])
        self.assertEqual(facts[1]["series_key"], facts[3]["series_key"])
        self.assertNotEqual(facts[0]["series_key"], facts[1]["series_key"])


class SubjectEvidenceTests(unittest.TestCase):
    def test_events_named_like_the_file_do_not_become_the_document_subject(self) -> None:
        """虚构的事件与文件同名时,不能压过上级目录所指的人员主体。"""
        from kb_pipeline.graph.facts import subject_evidence_by_doc

        ents = [{"key": "某甲", "title": "某甲", "type": "patient", "upper": "agent", "kind": "instance", "doc_ids": ["d1", "d2"], "frequency": 9},
                {"key": "2024体检报告", "title": "2024体检报告", "type": "health examination", "upper": "event", "kind": "instance",
                 "doc_ids": ["d1"], "frequency": 3}]
        ev = subject_evidence_by_doc(ents, ["patient", "health examination"],
                                     documents={"d1": "某甲/体检报告/2024体检报告.pdf", "d2": "某甲/体检报告/2025体检报告.pdf"})
        self.assertEqual({d: (e["title"], how) for d, (e, how) in ev.items()}, {"d1": ("某甲", "path"), "d2": ("某甲", "path")})


class TimeScopeTests(unittest.TestCase):
    """检索时的时间口径(关系、事实、文档状态共用一套):默认现行、as_of 回看某一天、history 全给。
    Cypher 条件与 Python 里的判定必须是同一个口径,否则清单和计数对不上。"""

    def test_as_of_dates_close_the_period_they_name(self) -> None:
        from kb_pipeline.graph.currency import as_of_day

        self.assertEqual(as_of_day("2024"), "2024-12-31")
        self.assertEqual(as_of_day("2024-02"), "2024-02-29")                 # 「截至 2024 年 2 月」:那月的最后一天
        self.assertEqual(as_of_day("2023-2-3"), "2023-02-03")
        self.assertEqual(as_of_day(""), "")
        for bad in ("2024-13", "2023-02-30", "去年", "2024/01"):
            with self.assertRaises(ValueError):
                as_of_day(bad)

    def test_row_status_by_scope(self) -> None:
        from kb_pipeline.graph.currency import in_scope, query_scope, row_status

        today = "2026-10-24"
        self.assertEqual(row_status({}, day=today), "current")
        self.assertEqual(row_status({"valid_until": "2014", "lapses": True}, day=today), "ended")
        self.assertEqual(row_status({"valid_until": "2026", "lapses": True}, day=today), "current")   # 只写到年:那一整年都算
        self.assertEqual(row_status({"valid_from": "2014-05-01", "valid_until": "2014-05-01"}, day=today), "current")  # 发生过的事不过期
        self.assertEqual(row_status({"superseded_by": "r2"}, day=today), "superseded")
        self.assertEqual(row_status({"doc_status": "archived"}, day=today), "doc_archived")
        self.assertEqual(row_status({"doc_status": "superseded"}, day=today), "doc_superseded")
        self.assertEqual(row_status({"valid_until": "v2.0", "lapses": True}, day=today), "current")   # 不是日期的值不比
        self.assertEqual(row_status({"valid_until": "2014", "lapses": True}, day="2015-01-01", as_of=True), "ended")
        self.assertEqual(row_status({"valid_from": "2014-05-01", "valid_until": "2014-05-01"}, day="2015-01-01", as_of=True), "current")
        # 回看:被取代的时间晚于那一天就还算数;那天还没开始(或还没有文档提到)的不算;文档状态是现在的事,不看
        old = {"superseded_by": "r2", "superseded_on": "2025-03", "observed_first": "2024-05-01", "doc_status": "archived"}
        self.assertEqual(row_status(old, day="2025-02-28", as_of=True), "current")
        self.assertEqual(row_status(old, day="2025-03-31", as_of=True), "superseded")
        self.assertEqual(row_status(old, day="2024-01-01", as_of=True), "not_started")
        self.assertEqual(row_status({"superseded_by": "s2", "superseded_on": "v7.0"}, day="2020-01-01", as_of=True), "current")  # 版本轴说不出哪天
        scope = query_scope(as_of="2025-02")
        self.assertEqual((scope["mode"], scope["day"]), ("as_of", "2025-02-28"))
        self.assertTrue(in_scope(old, scope))
        self.assertTrue(in_scope({"superseded_by": "x"}, query_scope(history=True)))
        self.assertFalse(in_scope({"superseded_by": "x"}, query_scope(today=today)))
        self.assertEqual(query_scope(history=True, as_of="2024")["day"], "2024-12-31")       # history 照样按回看那天标状态

    def test_cypher_condition_carries_the_same_rule(self) -> None:
        """Cypher 条件与 row_status 是同一个口径:现行看取代 / 文档状态 / 有效期,回看看起点 / 有效期 / 取代的时间、不看文档状态。
        (条件在真库上的行为由上线前的实库核对覆盖,这里钉住条件的构成。)"""
        from kb_pipeline.graph.currency import cypher_in_effect, cypher_in_scope, query_scope, row_status

        cur = cypher_in_effect("r", observed="observed_first", as_of=False)
        for piece in ("r.superseded_on", "r.valid_from", "coalesce(r.doc_status, '') = ''", "NOT coalesce(r.lapses, false)",
                      "r.valid_until >= left($day, size(r.valid_until))", "NOT coalesce(r.valid_until, '') =~ '[0-9]{4}.*'"):
            self.assertIn(piece, cur)
        self.assertIn("NOT coalesce(f.lapses, false)", cypher_in_effect("f", observed="observed", as_of=True))
        back = cypher_in_effect("f", observed="observed", as_of=True)
        for piece in ("coalesce(f.valid_from, f.observed, '') <= left($day, size(coalesce(f.valid_from, f.observed, '')))",
                      "f.superseded_on <= left($day, size(f.superseded_on))", "NOT (coalesce(f.superseded_by, '') <> ''"):
            self.assertIn(piece, back)
        self.assertNotIn("doc_status", back)                     # 回看不看文档的现在状态
        self.assertEqual(cypher_in_scope("r", observed="observed_first", scope=query_scope(history=True)), "true")
        self.assertEqual(row_status({"valid_from": "2027"}, day="2026-10-24"), "not_started")  # Current is evaluated at today's date.


class FactAttributionTests(unittest.TestCase):
    """10-24 新旧图对比里暴露的挂错与噪声:长文档(一份讲好几样东西)里一节的数值挂到别的小节的主体上;报告里按等级
    列出的参考建议当成了本人的事实;证据里同一条事实重复好几遍;中心实体扩展出来的边只按强度挑,跟问题对不上。"""

    def _units(self):
        u1 = _unit("u1", doc="d1"); u1.rel_path = "合集/手册.md"
        u2 = _unit("u2", doc="d1", order=1); u2.rel_path = "合集/手册.md"
        u3 = _unit("u3", doc="d2", order=2); u3.rel_path = "李华/体检报告/2025.pdf"
        return [u1, u2, u3]

    def test_units_get_their_own_subjects_before_the_document_ones(self) -> None:
        from kb_pipeline.graph.facts import unit_subjects

        ents = [
            {"key": "harness", "title": "评测框架甲", "type": "framework", "upper": "entity", "frequency": 30, "doc_ids": ["d1"], "unit_ids": ["u1"]},
            {"key": "bench", "title": "题库乙", "type": "benchmark", "upper": "entity", "frequency": 5, "doc_ids": ["d1"], "unit_ids": ["u2"]},
            {"key": "lh", "title": "李华", "type": "person", "upper": "agent", "frequency": 3, "doc_ids": ["d2"], "unit_ids": ["u3"]},
            {"key": "org", "title": "某医院", "type": "organization", "upper": "agent", "frequency": 9, "doc_ids": ["d2"], "unit_ids": ["u3"]},
            {"key": "cls", "title": "患者", "type": "person", "upper": "agent", "kind": "class", "frequency": 50, "doc_ids": ["d2"], "unit_ids": ["u3"]},
        ]
        units = self._units()
        out = unit_subjects(units, ents, subject_types=["person", "framework"], documents={u.doc_id: u.rel_path for u in units})
        self.assertEqual(out["u2"][:2], ["题库乙", "评测框架甲"])           # 这一节讲题库乙:排在整份文档的主要主体前面
        self.assertEqual(out["u1"][0], "评测框架甲")
        self.assertEqual(out["u3"][0], "李华")                             # 路径说明这份报告就是李华的:排最前,医院在后
        self.assertIn("某医院", out["u3"])
        self.assertNotIn("患者", out["u3"])                                # 泛指的类别名不当主体

    def test_a_section_heading_names_the_topic_of_a_compilation(self) -> None:
        """无归属人的合成汇编中,小节里的值不能都归到整份文档的主要主体名下。小节标题点名的那样东西排在单元自己点名的东西后面、整份文档的主要主体前面;有归属人的
        文档不加;英文名按整词认,类别名、文档不算。"""
        from kb_pipeline.graph.facts import unit_subjects

        u1, u2, u3 = _unit("u1", doc="d1"), _unit("u2", doc="d1", order=1), _unit("u3", doc="d1", order=2)
        u4 = _unit("u4", doc="d2", order=3)
        for u in (u1, u2, u3):
            u.rel_path = "合集/手册.md"
        u4.rel_path = "李华/体检报告/2025.pdf"
        u1.section_path = ["21点实验报告", "4. 实验怎么做的"]
        u2.section_path = ["21点实验报告", "6. 结果"]
        u3.section_path = ["Autoresearch Feature Discovery", "笔记本结构"]
        u4.section_path = ["血常规检查"]
        ents = [
            {"key": "harness", "title": "评测框架甲", "type": "framework", "upper": "entity", "frequency": 30, "doc_ids": ["d1"], "unit_ids": []},
            {"key": "exp", "title": "21点实验", "type": "experiment", "upper": "event", "scope": "d1", "frequency": 2, "doc_ids": ["d1"]},
            {"key": "rep", "title": "21点实验报告", "type": "report", "upper": "document", "scope": "d1", "frequency": 1, "doc_ids": ["d1"]},
            {"key": "judge", "title": "裁判", "type": "role", "upper": "agent", "frequency": 3, "doc_ids": ["d1"], "unit_ids": ["u2"]},
            {"key": "redis", "title": "Redis", "type": "database", "upper": "entity", "frequency": 4, "doc_ids": ["d3"]},
            {"key": "nb", "title": "笔记本结构", "type": "structure", "upper": "part", "kind": "class", "frequency": 6, "doc_ids": ["d1"]},
            {"key": "lh", "title": "李华", "type": "person", "upper": "agent", "frequency": 3, "doc_ids": ["d2"], "unit_ids": ["u4"]},
            {"key": "cbc", "title": "血常规检查", "type": "examination", "upper": "process", "frequency": 2, "doc_ids": ["d2"]},
        ]
        units = [u1, u2, u3, u4]
        out = unit_subjects(units, ents, subject_types=["person", "framework"], documents={u.doc_id: u.rel_path for u in units})
        self.assertEqual(out["u1"][:2], ["21点实验", "评测框架甲"])           # 这一节自己没点名任何东西:标题点名的实验在框架前面
        self.assertEqual(out["u2"][:3], ["裁判", "21点实验", "评测框架甲"])   # 单元自己点名的东西仍排最前
        self.assertNotIn("Redis", out["u3"])                                  # Feature Discovery 里没有 Redis
        self.assertNotIn("笔记本结构", out["u3"])                             # 类别名不是哪一样东西
        self.assertEqual(out["u4"], ["李华"])                                 # 有归属人的文档:数值归归属人,不加标题主题

    def test_prompts_ask_for_the_subject_the_values_belong_to(self) -> None:
        from kb_pipeline.graph import prompts

        for text in (prompts.FACTS_PROMPT, prompts.NARRATIVE_FACTS_PROMPT):
            self.assertIn("things named in this part of the document first", text)
        self.assertIn("use the likely subject above that the values belong to", prompts.FACTS_PROMPT)
        self.assertIn("Skip advice addressed to the reader", prompts.FACTS_PROMPT)
        self.assertIn("A recommended value or setting of a thing", prompts.FACTS_PROMPT)     # 产品的推荐工作条件照样是事实

    def test_duplicate_facts_collapse_into_one_row(self) -> None:
        from kb_pipeline.graph.recall import collapse_duplicate_facts

        base = {"subject": "李华", "subject_id": "e1", "property": "GENE_A rs0000111 风险/正常", "valid_from": "2023-05-10"}
        rows = [{**base, "gr_id": "f1", "value": "正常", "conditions": {"基因": "GENE_A"}, "point_ids": ["p1"]},
                {**base, "gr_id": "f2", "value": "正常", "conditions": {"基因": "GENE_A", "位点名称": "rs0000111"}, "point_ids": ["p7"]},
                {**base, "gr_id": "f3", "value": "风险", "conditions": {"基因": "GENE_A"}, "point_ids": ["p3"]},
                {"gr_id": "f4", "subject": "器件甲", "property": "tAS", "min": "0.5", "unit": "ns", "conditions": {"速度等级": "-250"}},
                {"gr_id": "f5", "subject": "器件甲", "property": "tAS", "min": "0.5", "unit": "ns", "conditions": {"速度等级": "-200"}}]
        out = collapse_duplicate_facts(rows)
        # 条件只是复述属性名里已有的内容(位点名称)不算区别;取值不同、真正的条件(速度等级)不同的照留
        self.assertEqual([r["gr_id"] for r in out], ["f1", "f3", "f4", "f5"])
        self.assertEqual((out[0]["point_ids"], out[0]["_duplicates"]), (["p1", "p7"], 1))

    def test_expansion_prefers_edges_that_match_the_question(self) -> None:
        from kb_pipeline.graph.recall import expansion_relevance

        q = "DSE 需要具备哪些能力？"
        skill = {"title": "业务理解能力", "type": "requires", "description": "DSE 需要懂业务的能力"}
        years = {"title": "3-5年经验", "type": "has_parameter", "description": "DSE 岗位常见的工作年限"}
        self.assertGreater(expansion_relevance(q, "DSE", skill) * 3.0, expansion_relevance(q, "DSE", years) * 5.0)
        self.assertEqual(expansion_relevance("DSE", "DSE", years), 1.0)               # 问题只点了实体名:不改按强度的排序
        self.assertEqual(expansion_relevance("dse 的能力", "DSE", skill), expansion_relevance("能力", "", skill))   # 起点名字大小写不论都去掉
