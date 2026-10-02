"""Local regressions for the search service (kb_search): fusion, cleaning, dedupe, routing, orchestration and
auth, all with stubs, never connecting to any external service."""
from __future__ import annotations

import json
import math
import unittest
from types import SimpleNamespace
from unittest import mock

import io
import os
import pathlib

from kb_search import catalog as catalog_mod
from kb_search import channels, evalset, graphwalk, images, service
from kb_search.config import SearchSettings, load_search_settings
from kb_search.evidence import assemble_sources, doc_aggs, page_rows, select_hits, spec_rows, stitch_short_hit
from kb_search.fusion import interleave, rrf_merge
from kb_search.rerank import Reranker, rerank, rerank_documents
from kb_search.text import clean_for_rerank, dedupe_overlaps, is_boilerplate, jaccard, overlap_ratio, position, token_set, truncate_tokens, windows


def _settings(**over) -> SearchSettings:
    base = load_search_settings().__dict__ | {"token": "", "rerank_enabled": True, "rerank_threshold": 0.0, "neighbor_span": 1,
                                             "context_tokens": 400, "top_k": 3, "rerank_n": 8, "final_lex_weight": 0.0}
    base.update(over)
    return SearchSettings(**base)


class TextTests(unittest.TestCase):
    def test_clean_for_rerank_strips_prefixes_tables_and_markdown(self) -> None:
        raw = "TITLE: 3.2 Timing\nCAPTION: Table 5\n| 参数 | 最小 | 最大 |\n|---|---|---|\n| tAS | 5 | 10 ns |\n**说明**:# 标题\n`code`"
        out = clean_for_rerank(raw)
        self.assertNotIn("|", out); self.assertNotIn("---", out); self.assertNotIn("TITLE", out); self.assertNotIn("**", out)
        self.assertIn("参数, 最小, 最大", out); self.assertIn("tAS, 5, 10 ns", out); self.assertIn("3.2 Timing", out)

    def test_windows_split_long_text_with_overlap(self) -> None:
        sentence = "这是一个用来测试分窗的句子。"
        text = sentence * 200
        wins = windows(text, window_tokens=120, overlap_tokens=20)
        self.assertGreater(len(wins), 1)
        self.assertTrue(all(w for w in wins))
        self.assertEqual(windows("短句。", window_tokens=120), ["短句。"])

    def test_position_strings(self) -> None:
        self.assertEqual(position({"page_idx": 46, "section_path": ["3. AC Characteristics", "3.2 Timing"]}), "page 46 · 3. AC Characteristics / 3.2 Timing")
        self.assertEqual(position({"slide_idx": 4}), "slide 4")                                       # page / slide numbers in the payload already start at 1
        self.assertEqual(position({"sheet_name": "价格表", "row_start": 2, "row_end": 9}), "sheet 价格表 rows 2–9")
        self.assertEqual(position({}), "")

    def test_numbers_and_documents_are_not_merged_by_dedupe(self) -> None:
        a = {"kb_id": "k", "doc_id": "A", "content_version": "1", "text": "产品甲的额定电压为3.3 V。"}
        b = {"kb_id": "k", "doc_id": "B", "content_version": "1", "text": "产品乙的额定电压为33 V。"}
        self.assertLess(overlap_ratio("额定电压为3.3 V。", "额定电压为33 V。"), 1.0)                    # the decimal point must survive normalisation (Codex S01)
        same_a = {"kb_id": "k", "doc_id": "A", "content_version": "1", "text": "产品甲的额定电压为3.3 V。补充说明一句。"}
        kept, dropped = dedupe_overlaps([a, b, dict(a)], threshold=0.85)
        self.assertEqual((len(kept), dropped), (2, 1))                                                 # different documents are always kept; only same-document duplicates go
        kept2, dropped2 = dedupe_overlaps([a, same_a], threshold=0.85)
        self.assertEqual((dropped2, kept2[0]["text"]), (1, same_a["text"]))                              # same-document containment: drop the shorter, keep the longer

    def test_dedupe_requires_adjacency_and_matching_numbers(self) -> None:
        from kb_search.text import numbers_of, same_facts
        self.assertEqual(numbers_of("上限温度为-40摄氏度,精度 0.5"), ["-40", "0.5"]); self.assertFalse(same_facts("上限温度为40摄氏度。", "上限温度为-40摄氏度。"))
        self.assertLess(overlap_ratio("上限温度为40摄氏度。", "上限温度为-40摄氏度。"), 1.0)                     # the minus sign is kept (Codex R2)
        far_a = {"kb_id": "k", "doc_id": "A", "content_version": "1", "chunk_index": 1, "text": "同一句话出现在两处。" * 5}
        far_b = {"kb_id": "k", "doc_id": "A", "content_version": "1", "chunk_index": 99, "text": "同一句话出现在两处。" * 5}
        near = {"kb_id": "k", "doc_id": "A", "content_version": "1", "chunk_index": 2, "text": "同一句话出现在两处。" * 5 + "补一句。"}
        kept, dropped = dedupe_overlaps([far_a, far_b, near], threshold=0.85)
        self.assertEqual((len(kept), dropped), (2, 1)); self.assertIn(far_b, kept)                        # non-adjacent kept; adjacent: drop the shorter, keep the longer
        num_a = {"kb_id": "k", "doc_id": "A", "content_version": "1", "chunk_index": 5, "text": "上限温度为40摄氏度,其余参数相同。" * 3}
        num_b = {"kb_id": "k", "doc_id": "A", "content_version": "1", "chunk_index": 6, "text": "上限温度为-40摄氏度,其余参数相同。" * 3}
        self.assertEqual(dedupe_overlaps([num_a, num_b], threshold=0.85)[1], 0)                          # mismatched numbers are not duplicates

    def test_windows_split_unpunctuated_text_and_truncate(self) -> None:
        long_run = "无标点长文" * 800
        wins = windows(long_run, window_tokens=120, overlap_tokens=20)
        from kb_pipeline.utils import count_tokens
        self.assertGreater(len(wins), 3); self.assertTrue(all(count_tokens(w) <= 120 + 20 for w in wins))  # splits by token even without sentence breaks (S07); window + overlap
        cut = truncate_tokens("这是一段用来截断的正文。" * 100, 50)
        self.assertLessEqual(count_tokens(cut), 50); self.assertTrue(cut)

    def test_body_tokens_ignore_repeated_headers_and_coverage(self) -> None:
        from kb_search.text import body_token_set, question_coverage
        a = "SHEET: 功能清单 ROWS: 3-3 HEADER: 模块 | 功能 | 描述\n消息 | 群投票 | 支持发起投票"
        b = "SHEET: 功能清单 ROWS: 9-9 HEADER: 模块 | 功能 | 描述\n日历 | 会议室预订 | 支持预订会议室"
        self.assertLess(jaccard(body_token_set(a), body_token_set(b)), jaccard(token_set(a), token_set(b)))   # header rows do not count towards similarity (rows of the same sheet)
        self.assertGreater(question_coverage("PC端消息模块的群投票功能支持什么", a), question_coverage("PC端消息模块的群投票功能支持什么", b))
        self.assertEqual(question_coverage("", a), 0.0)

    def test_overlap_dedupe_keeps_the_longer_piece(self) -> None:
        a = {"text": "供电电压 3.3 V,工作温度 -40 到 85 度。存取时间 10 ns。"}
        b = {"text": "供电电压 3.3 V,工作温度 -40 到 85 度。存取时间 10 ns。功耗 100 mW。" + "补充说明。" * 3}
        c = {"text": "完全不同的一段话,讲的是封装类型和引脚定义。"}
        self.assertGreaterEqual(overlap_ratio(a["text"], a["text"]), 1.0)
        kept, dropped = dedupe_overlaps([a, b, c], threshold=0.5, scoped=False)
        self.assertEqual(dropped, 1); self.assertEqual(len(kept), 2); self.assertIs(kept[0], b)                  # contained with matching numbers: drop the shorter, keep the longer
        d = {"text": "工作温度 -40 到 85 度。存取时间 10 ns。功耗 100 mW。" + "补充说明。" * 3}
        self.assertEqual(dedupe_overlaps([a, d], threshold=0.5, scoped=False)[1], 0)                          # each has numbers the other lacks: both kept (R2)


    def test_token_set_jaccard_and_boilerplate(self) -> None:
        a = token_set("MACD 指标怎么计算"); b = token_set("MACD 指标的计算方法")
        self.assertIn("macd", a); self.assertIn("指标", a); self.assertGreaterEqual(jaccard(a, b), 0.3); self.assertEqual(jaccard(set(), a), 0.0)
        self.assertTrue(is_boilerplate({"section_path": ["目录"], "title": ""})); self.assertTrue(is_boilerplate({"title": "Revision History"}))
        self.assertTrue(is_boilerplate({"section_path": ["版权声明 Copyright"]})); self.assertFalse(is_boilerplate({"section_path": ["产品功能目录说明"], "title": "3.2 Timing"}))
        self.assertFalse(is_boilerplate({"section_path": ["3. AC Characteristics"]}))


class PlaceTests(unittest.TestCase):
    def test_place_is_the_short_locator_for_citations(self) -> None:
        """The short locator appended to the file path in a citation: page / slide / sheet rows when there are any,
        the deepest heading for documents without pages. It cannot be cut out of the position string: headings
        themselves contain " / " and " · ", and a cut takes half a heading for the locator."""
        from kb_search.evidence import source_row
        from kb_search.text import place, position

        pdf = {"page_idx": 46, "section_path": ["3. AC Characteristics", "3.2 Timing"]}
        self.assertEqual((position(pdf), place(pdf)), ("page 46 · 3. AC Characteristics / 3.2 Timing", "page 46"))
        self.assertEqual(place({"slide_idx": 3, "section_path": ["Overview"]}), "slide 3")
        self.assertEqual(place({"sheet_name": "Prices", "row_start": 6, "row_end": 13}), "sheet Prices rows 6–13")
        self.assertEqual(place({"sheet_name": "Summary"}), "sheet Summary")
        md = {"section_path": ["Volume 3 · Bundles (A / B / C)", "Bundles · A / C / Summer offer"]}
        self.assertEqual(place(md), "Bundles · A / C / Summer offer")           # the whole heading, not the half after a "/"
        self.assertEqual(position(md), "Volume 3 · Bundles (A / B / C) / Bundles · A / C / Summer offer")
        long = {"section_path": ["1. " + "a very long heading " * 10]}
        self.assertEqual((len(place(long)), place(long)[-1]), (60, "…"))
        self.assertEqual((place({}), place({"page_idx": "", "section_path": ["", " "]})), ("", ""))
        row = source_row(1, {"point_id": "p1", "kb_id": "kb_001"}, _payload("p1", "d1", 0, "body"))
        self.assertEqual(row["place"], place(_payload("p1", "d1", 0, "body")))             # source rows carry the same short locator


class ConfigTests(unittest.TestCase):
    def test_defaults_are_the_calibrated_values(self) -> None:
        env = {k: v for k, v in os.environ.items() if not k.startswith("KB_SEARCH_")}
        with mock.patch.dict(os.environ, env, clear=True):
            ss = load_search_settings()
        self.assertEqual(ss.rerank_threshold, 0.1); self.assertEqual(ss.route_gap, 0.2); self.assertEqual(ss.route_floor, 0.45)
        self.assertEqual((ss.stitch_min_chars, ss.stitch_max_chars, ss.mmr_lambda, ss.quota_min_hits), (350, 850, 0.7, 2))
        self.assertEqual((ss.graph_hops, ss.final_lex_weight, ss.query_instruction), (1, 0.2, ""))                # fixed values from Q27 / the lexical-weight grid / the instruction ablation
        self.assertTrue(ss.visual_enabled and ss.mmr_enabled and ss.route_widen)

    def test_env_example_lists_every_key_with_the_code_default(self) -> None:
        import re

        text = (pathlib.Path(__file__).resolve().parents[2] / "config/knowledge-base.env.example").read_text(encoding="utf-8")
        example = dict(re.findall(r"^#?\s*(KB_SEARCH_[A-Z0-9_]+)=(.*)$", text, re.M))
        read: set[str] = set()

        class Recording(dict):
            def get(self, key, default=None):
                read.add(key)
                return super().get(key, default)

        defaults = load_search_settings(Recording())
        self.assertEqual(set(example), read)                                     # the example lacks no key and has none the code does not know
        self.assertEqual(load_search_settings(example), defaults)                # copying the example verbatim gives the calibrated values
        self.assertEqual((defaults.rerank_threshold, defaults.graph_hops, defaults.neighborhoods), (0.1, 1, 4))

    def test_search_keys_follow_the_env_file_without_a_restart(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            env_file = pathlib.Path(tmp) / "knowledge-base.env"
            env_file.write_text("# comment\nKB_SEARCH_TOKEN=old-token\nKB_SEARCH_RERANK_THRESHOLD=0.1\nKB_SEARCH_TOP_K=7\nKB_SEARCH_GRAPH_K=60\n"
                                "KB_SEARCH_TOP_K=9\nQDRANT_URL=http://q\n", encoding="utf-8")
            settings = SimpleNamespace(env_file=env_file, qdrant_url="http://127.0.0.1:1", qdrant_api_key=None, sources={})
            # Process environment: the first three keys were read from the file at startup (left there by load_env_file's
            # setdefault; for a repeated key the first one wins); GRAPH_K disagrees with the file and VECTOR_K is not in the
            # file, so those two were set explicitly
            stale = {"KB_SEARCH_TOKEN": "old-token", "KB_SEARCH_RERANK_THRESHOLD": "0.1", "KB_SEARCH_TOP_K": "7",
                     "KB_SEARCH_GRAPH_K": "77", "KB_SEARCH_VECTOR_K": "33"}
            fresh = {"settings": None, "search": None, "q": None, "q_key": None, "at": 0.0, "explicit": None}
            with mock.patch.dict(os.environ, stale), mock.patch.dict(service._state, fresh), \
                    mock.patch.object(service, "QdrantClient", side_effect=lambda **kw: SimpleNamespace(**kw)), \
                    mock.patch.object(service, "load_settings", return_value=settings):
                ss = service.runtime()[1]
                self.assertEqual((ss.token, ss.rerank_threshold, ss.top_k, ss.graph_k, ss.vector_k), ("old-token", 0.1, 7, 77, 33))
                env_file.write_text("KB_SEARCH_TOKEN=new-token\nKB_SEARCH_RERANK_THRESHOLD=0.3\nKB_SEARCH_GRAPH_K=50\n", encoding="utf-8")
                self.assertEqual(service.runtime()[1].token, "old-token")                       # within a minute the cached settings are used
                service._state["at"] = 0.0
                ss = service.runtime()[1]
                self.assertEqual((ss.token, ss.rerank_threshold), ("new-token", 0.3))          # changed keys follow the file, and the old token stops working
                self.assertEqual(ss.top_k, 12)                                                  # a key deleted from the file returns to its default
                self.assertEqual((ss.graph_k, ss.vector_k), (77, 33))                           # keys set explicitly in the process environment still take precedence
                env_file.unlink()
                service._state["at"] = 0.0
                ss = service.runtime()[1]
                self.assertEqual((ss.token, ss.rerank_threshold), ("new-token", 0.3))          # file unreadable for a moment: the previous settings are kept, not the defaults
                env_file.write_text("KB_SEARCH_TOKEN=new-token\nKB_SEARCH_RERANK_THRESHOLD=0.3\n", encoding="utf-8")
                q = service.runtime()[2]
                self.assertEqual((q.timeout, q.check_compatibility), (25, False))                # the timeout follows the search scale, not the 120 s used when building
                service._state["at"] = 0.0
                self.assertIs(service.runtime()[2], q)                                          # connection details unchanged: the client is not rebuilt
                settings.qdrant_url = "http://127.0.0.1:2"
                service._state["at"] = 0.0
                self.assertIsNot(service.runtime()[2], q)


    def test_service_reads_its_settings_once_at_startup(self) -> None:
        import uvicorn

        from kb_search import main as main_mod

        order = []
        with mock.patch("kb_pipeline.config.load_settings", side_effect=lambda *a, **k: order.append("load")), \
                mock.patch.object(service, "runtime", side_effect=lambda: order.append("runtime")), \
                mock.patch.object(uvicorn, "run", side_effect=lambda *a, **k: order.append("serve")):
            main_mod.run()
        self.assertEqual(order, ["load", "runtime", "serve"])       # explicitly set keys must be recognised while the file has just been loaded into the environment


class EvidenceTests(unittest.TestCase):
    def test_stitch_short_hit_alternates_neighbors_within_limits(self) -> None:
        row = {"block_type": "text", "chunk_index": 5, "text": "短片。" * 10}
        nbs = [{"chunk_index": 4, "block_type": "text", "text": "前一片。" * 30}, {"chunk_index": 6, "block_type": "text", "text": "后一片。" * 30},
               {"chunk_index": 7, "block_type": "table", "text": "| a | b |"}, {"chunk_index": 3, "block_type": "text", "text": "再前。" * 500}]
        self.assertTrue(stitch_short_hit(row, nbs, min_chars=350, max_chars=850))
        self.assertTrue(row["text"].startswith("前一片。")); self.assertIn("后一片。", row["text"]); self.assertNotIn("| a |", row["text"])
        self.assertEqual((row["stitched"]["chunk_from"], row["stitched"]["chunk_to"], row["stitched"]["own_chars"]), (4, 6, 30)); self.assertLessEqual(len(row["text"]), 850)
        self.assertEqual([pc["chunk_index"] for pc in row["stitched"]["pieces"]], [4, 5, 6])                # each piece can be cited on its own (S03)
        self.assertIsNone(row["degraded"]); self.assertTrue(all("degraded" not in pc for pc in row["stitched"]["pieces"]))
        lossy = {"block_type": "text", "chunk_index": 5, "text": "短片。" * 10}
        self.assertTrue(stitch_short_hit(lossy, [{**nbs[0], "degraded": "text_layer_cjk_lost"}, nbs[1]], min_chars=350, max_chars=850))
        self.assertEqual(lossy["degraded"], "text_layer_cjk_lost")                                  # a neighbour's parse degradation marker travels with its text
        self.assertEqual([pc.get("degraded") for pc in lossy["stitched"]["pieces"]], ["text_layer_cjk_lost", None, None])
        long_row = {"block_type": "text", "chunk_index": 1, "text": "x" * 400}
        self.assertFalse(stitch_short_hit(long_row, nbs, min_chars=350, max_chars=850))
        self.assertFalse(stitch_short_hit({"block_type": "table", "chunk_index": 1, "text": "x"}, nbs, min_chars=350, max_chars=850))

    def test_select_hits_quota_then_mmr(self) -> None:
        def c(pid, score, text, bucket=None):
            return {"point_id": pid, "score_final": score, "bucket": bucket, "payload": {"text": text}}
        ordered = [c("a1", 0.9, "尿酸 433 参考 208-428", "尿酸"), c("a2", 0.85, "尿酸 433 参考 208-428 复查", "尿酸"),
                   c("a3", 0.8, "尿酸 433 参考 208-428。", "尿酸"), c("b1", 0.2, "血糖 5.3 空腹", "血糖"), c("x1", 0.5, "甘油三酯 1.7", None)]
        hits, stats = select_hits(ordered, 4, buckets={"mode": "subject", "keys": ["尿酸", "血糖"]}, min_hits=2, mmr_lambda=0.7)
        ids = [h["point_id"] for h in hits]
        self.assertEqual(ids[:2], ["a1", "a2"]); self.assertIn("b1", ids); self.assertEqual(stats["quota_filled"], {"尿酸": 2, "血糖": 1})
        self.assertIn("x1", ids); self.assertEqual(ids, sorted(ids, key=lambda i: -next(c["score_final"] for c in ordered if c["point_id"] == i)))   # selection fixed; presentation ordered by score
        self.assertNotIn("a3", ids)                                                                 # MMR: a3, a duplicate of what is already selected, yields to different content
        self.assertGreater(stats["avg_redundancy"], 0.0)
        hits2, _ = select_hits(ordered, 3, buckets=None, min_hits=2, mmr_lambda=None)
        self.assertEqual([h["point_id"] for h in hits2], ["a1", "a2", "a3"])                         # no MMR, no buckets: cut in order
        ordered[3]["scores"] = {"score_visual": 0.97}; ordered[4]["scores"] = {"score_visual": 0.6}
        hits3, st3 = select_hits(ordered, 3, buckets=None, min_hits=2, mmr_lambda=None, priority_key="score_visual", priority_n=2)
        self.assertEqual(sorted(h["point_id"] for h in hits3[:3]), ["a1", "b1", "x1"]); self.assertEqual(st3["priority_filled"], 2)   # visual-score floor (S02)

    def test_mmr_incremental_similarity_picks_the_same_hits(self) -> None:
        import random

        from kb_search.text import body_token_set

        def reference(ordered, k, lam):
            """The naive version, comparing every candidate with all picked entries again in each round, as the reference."""
            scores = [float(c["score_final"]) for c in ordered]
            rel = [x / max(scores) for x in scores]
            toks = [body_token_set(c["payload"]["text"]) for c in ordered]
            chosen, remaining = [], list(range(len(ordered)))
            while remaining and len(chosen) < k:
                best = max(remaining, key=lambda i: lam * rel[i] - (1 - lam) * max((jaccard(toks[i], toks[j]) for j in chosen), default=0.0))
                chosen.append(best); remaining.remove(best)
            return sorted((ordered[i]["point_id"] for i in chosen))

        words = ["尿酸", "血糖", "肌酐", "甘油三酯", "参考范围", "复查", "空腹", "偏高", "正常", "报告"]
        for seed in range(5):
            rnd = random.Random(seed)
            ordered = sorted(({"point_id": f"p{i}", "score_final": round(rnd.random(), 3),
                               "payload": {"text": " ".join(rnd.choice(words) for _ in range(rnd.randint(3, 9)))}} for i in range(40)),
                             key=lambda c: -c["score_final"])
            hits, _ = select_hits(ordered, 12, mmr_lambda=0.7)
            self.assertEqual(sorted(h["point_id"] for h in hits), reference(ordered, 12, 0.7))

    def test_budget_hard_bound_truncates_hits_with_position(self) -> None:
        hits = [{"point_id": f"p{i}", "kb_id": "kb_001", "scores": {}, "recall_sources": ["text"],
                 "payload": _payload(f"p{i}", f"d{i}", 0, f"第 {i} 篇正文。" * 200, token_count=0)} for i in range(4)]
        rows, stats = assemble_sources(hits, budget_tokens=600, neighbors=None)
        total = sum(int(r["token_count"]) for r in rows)
        self.assertEqual(len(rows), 4); self.assertLessEqual(total, 600); self.assertGreaterEqual(stats["truncated_hits"], 2)    # the total stays within budget (R3)
        cut = [r for r in rows if r.get("text_truncated")]
        self.assertTrue(cut and all(r["full_tokens"] > r["token_count"] and r["position"] and r["token_count"] >= 40 for r in cut))   # each keeps room for an excerpt
        rows2, _ = assemble_sources(hits, budget_tokens=100, neighbors=None)
        self.assertLessEqual(sum(int(r["token_count"]) for r in rows2), 100); self.assertEqual(len(rows2), 4)

    def test_spec_rows_verify_source_points(self) -> None:
        specs = [{"subject": "李", "property": "尿酸", "value": "433", "point_ids": ["p1"], "score": 0.9},
                 {"subject": "李", "property": "肌酐", "value": "78", "point_ids": ["zz"], "score": 0.8}]
        rows = spec_rows(specs, source_ns={"p1": 1}, limit_hints=8, point_active={"p1": True, "zz": False})
        self.assertEqual((rows[0]["verified"], rows[0]["sources_active"], rows[0]["hint"]), (True, "1/1", "李 · 尿酸 = 433"))
        self.assertEqual((rows[1]["verified"], rows[1]["sources_active"], rows[1]["hint"]), (False, "0/1", None))     # source deactivated: no hint (S03)

    def test_spec_rows_hints_series_and_conflicts(self) -> None:
        specs = [{"subject": "李", "property": "尿酸", "value": "433", "unit": "umol/L", "when": "2024", "series_key": "s1", "series_index": 0, "point_ids": ["p1"], "score": 0.9},
                 {"subject": "李", "property": "尿酸", "value": "401", "unit": "umol/L", "when": "2025", "series_key": "s1", "series_index": 1, "point_ids": ["p9"], "conflict_group": "g1", "score": 0.8},
                 {"subject": "器件", "property": "tAS", "symbol": "tAS", "min": "5", "max": "10", "unit": "ns", "conditions_text": "VCC=3.3V", "quality": "ambiguous_source", "point_ids": [], "score": 0.7}]
        rows = spec_rows(specs, source_ns={"p1": 1}, limit_hints=8)
        self.assertEqual(rows[0]["hint"], "李 · 尿酸 = 433 umol/L · 2024"); self.assertEqual(rows[0]["sources"], [1])
        self.assertEqual(rows[0]["series_text"], "433 umol/L(2024)、401 umol/L(2025)")
        self.assertTrue(rows[1]["conflict"]); self.assertIn("Conflicting", rows[1]["conflict_note"]); self.assertFalse(rows[0]["conflict"])
        self.assertIsNone(rows[2]["hint"])                                                          # a fact with unclear provenance gets no hint

    def test_page_rows_budget_and_subject_first(self) -> None:
        pages = [{"kind": "source", "title": "报告 A", "score": 0.9, "text": "来源页正文", "series": []},
                 {"kind": "timeline", "title": "李 时间线", "score": 0.7, "text": "\n".join(f"2024-0{i} 指标 {i}" for i in range(1, 9)), "series": ["x"]},
                 {"kind": "subject", "title": "尿酸", "score": 0.6, "text": "主体页正文 " * 50, "series": ["a", "b"]}]
        rows = page_rows(pages, subjects=["尿酸"], text_budget_tokens=40)
        self.assertEqual(rows[0]["title"], "尿酸"); self.assertTrue(all(r["compiled"] for r in rows))
        self.assertIsNone([r for r in rows if r["kind"] == "source"][0].get("text"))                   # source pages only get the overview
        tl = [r for r in rows if r["kind"] == "timeline"][0]
        self.assertTrue(tl.get("text") is None or tl.get("text_truncated") or len(tl["text"]) < 200)   # over budget: lines are trimmed first

    def test_assemble_sources_adds_table_head_and_stitches(self) -> None:
        hit = {"point_id": "t2", "kb_id": "kb_001", "scores": {}, "recall_sources": ["bm25"],
               "payload": _payload("t2", "d1", 4, "| 3 | 4 |", block_type="table", block_id="blk")}
        short = {"point_id": "s1", "kb_id": "kb_001", "scores": {}, "recall_sources": ["text"], "payload": _payload("s1", "d2", 1, "短。")}
        head_pl = _payload("t1", "d1", 3, "| 参数 | 值 |", block_type="table", block_id="blk")
        nb = {"s1": [_payload("s0", "d2", 0, "前文。" * 60), _payload("s2", "d2", 2, "后文。" * 60)]}
        rows, stats = assemble_sources([hit, short], budget_tokens=5000, neighbors=lambda r, span: nb.get(r["point_id"], []),
                                       table_head=lambda r: head_pl if r["point_id"] == "t2" else None, stitch=(350, 850), neighbor_span=1)
        roles = [(r["role"], r["point_id"]) for r in rows]
        self.assertIn(("table_head", "t1"), roles); self.assertEqual(stats["table_heads"], 1); self.assertEqual(stats["stitched"], 1)
        s1 = next(r for r in rows if r["point_id"] == "s1")
        self.assertIn("前文。", s1["text"]); self.assertEqual(s1["stitched"]["chunk_from"], 0)
        self.assertEqual(next(r for r in rows if r["role"] == "table_head")["of"], next(r for r in rows if r["point_id"] == "t2")["n"])


class ImagesTests(unittest.TestCase):
    def test_resolve_bbox_conventions_and_crop(self) -> None:
        from PIL import Image

        self.assertEqual(images.resolve_bbox([0.1, 0.2, 0.5, 0.6], 1000, 500), (100, 100, 500, 300))
        self.assertEqual(images.resolve_bbox([100, 200, 500, 600], 2000, 1000), (200, 200, 1000, 600))          # per-mille
        self.assertEqual(images.resolve_bbox([0.9, 0.9, 0.99, 0.99], 100, 100), (90, 90, 99, 99))
        self.assertEqual(images.resolve_bbox([120, 80, 480, 360], 918, 1262), (110, 100, 441, 454))          # any value up to 1000 is read as per-mille; pixels are not accepted
        with self.assertRaises(ValueError):
            images.resolve_bbox([0, 0, 5000, 5000], 100, 100)
        with self.assertRaises(ValueError):
            images.resolve_bbox([0.5, 0.5, 0.5, 0.5], 100, 100)
        img = Image.new("RGB", (400, 200), (255, 255, 255)); buf = io.BytesIO(); img.save(buf, format="PNG")
        out = images.crop_image({"bytes": buf.getvalue(), "mime": "image/png", "source": "cache"}, [0.25, 0.0, 0.75, 1.0], pad=10)
        self.assertEqual(out["mime"], "image/png"); self.assertEqual(out["box"], [90, 0, 310, 200]); self.assertEqual((out["width"], out["height"]), (220, 200))


    def test_locate_rejects_ids_that_are_not_point_ids(self) -> None:
        asked = []

        class FakeQ:
            def retrieve(self, collection_name, ids, with_payload, with_vectors):
                asked.append(ids)
                return [SimpleNamespace(id=ids[0], payload={"visual_ref": "parse/x.jpg", "is_active": True})]

        for bad in ("kb_001:12:abc:pdf-v12:blk:0", "not-a-uuid", "", "12"):
            with self.assertRaises(ValueError):
                images.locate(FakeQ(), "kb_001", bad)
        self.assertEqual(asked, [])                                                                          # a bad argument is stopped locally, the main store is not asked
        pid = "3f2b8c1e-5d47-5a09-9c1b-0e6f2a7d4b10"
        self.assertEqual(images.locate(FakeQ(), "kb_001", pid)["point_id"], pid)

    def test_original_image_picks_the_placed_image_not_the_biggest(self) -> None:
        import tempfile

        try:
            import pymupdf
        except ImportError:
            self.skipTest("pymupdf not installed")
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            red = Image.new("RGB", (100, 100), (255, 0, 0)); blue = Image.new("RGB", (400, 400), (0, 0, 255))
            rb, bb = io.BytesIO(), io.BytesIO(); red.save(rb, format="PNG"); blue.save(bb, format="PNG")
            doc = pymupdf.open(); page = doc.new_page(width=600, height=800)
            page.insert_image(pymupdf.Rect(50, 50, 150, 150), stream=rb.getvalue())        # red: small image top-left
            page.insert_image(pymupdf.Rect(200, 300, 560, 660), stream=bb.getvalue())      # blue: same aspect ratio, more pixels
            (root / "mirror").mkdir(); (root / "cache" / "parse").mkdir(parents=True)
            doc.save(str(root / "mirror" / "doc.pdf")); doc.close()
            cached = root / "cache" / "parse" / "red.jpg"; red.save(cached, format="JPEG")
            settings = SimpleNamespace(mirror_root=root / "mirror", cache_dir=root / "cache")
            import hashlib
            sha = hashlib.sha256((root / "mirror" / "doc.pdf").read_bytes()).hexdigest()
            payload = {"source_path": "doc.pdf", "page_idx": 1, "visual_ref": "parse/red.jpg", "content_version": sha,
                       "bbox": [50 / 600 * 1000, 50 / 800 * 1000, 150 / 600 * 1000, 150 / 800 * 1000]}
            out = images.original_image(settings, payload)
            got = Image.open(io.BytesIO(out["bytes"])).convert("RGB")
            self.assertEqual(out["source"], "pdf-embedded"); self.assertEqual(got.size, (100, 100)); self.assertEqual(got.getpixel((50, 50)), (255, 0, 0))
            stale = images.original_image(settings, {**payload, "content_version": "0" * 64})          # content identity mismatch: fall back to the cache (R4)
            self.assertEqual(stale["source"], "cache")
            self.assertEqual(images.original_image(settings, {k: v for k, v in payload.items() if k != "content_version"})["source"], "cache")   # no version given: no guessing
            nobbox = images.original_image(settings, {k: v for k, v in payload.items() if k != "bbox"})
            self.assertEqual(nobbox["source"], "pdf-embedded"); self.assertEqual(Image.open(io.BytesIO(nobbox["bytes"])).size, (100, 100))   # no bbox: aspect ratio + no smaller than the cached image


class EvalsetTests(unittest.TestCase):
    def test_evaluate_metrics_and_compare(self) -> None:
        def fake_search(q, kbs=None, top_k=12):
            if "负例" in q:
                return {"kbs": ["kb_001"], "sources": [], "specs": [], "pages": [],
                        "retrieval_summary": {"rerank": "below_threshold", "rerank_max": 0.02, "low_confidence": True, "evidence_state": "diagnostic",
                                              "no_relevant_content": True, "timings_ms": {"total_ms": 100}}}
            return {"kbs": ["kb_001"], "sources": [{"role": "hit", "point_id": "p2", "doc_id": "d2", "rel_path": "x/d2.pdf", "doc": "d2.pdf", "text": "无关"},
                                                     {"role": "hit", "point_id": "p1", "doc_id": "d1", "rel_path": "x/d1.pdf", "doc": "d1.pdf", "text": "尿酸 433 umol/L"},
                                                     {"role": "neighbor", "point_id": "n1", "doc_id": "d1", "text": "参考范围 208-428"}],
                    "specs": [{"hint": "李 · 尿酸 = 433"}], "pages": [], "retrieval_summary": {"rerank": "ok", "rerank_max": 0.9, "low_confidence": False, "timings_ms": {"total_ms": 300}}}
        items = [{"q": "尿酸多少", "gold_point_ids": ["p1"], "gold_docs": ["x/d2.pdf"], "expect": ["433"], "expect_all": ["433", "参考范围"], "min_docs": 2, "tag": "t"},
                 {"q": "负例问题", "negative": True, "tag": "neg"},
                 {"q": "报错的题", "gold_point_ids": ["p9"], "tag": "err"}]

        def flaky(q, kbs=None, top_k=12):
            if "报错" in q:
                raise RuntimeError("boom")
            return fake_search(q, kbs=kbs, top_k=top_k)
        res = evalset.evaluate(items, flaky)
        r0, r1, r2 = res["items"]
        self.assertEqual((r0["rank"], r0["basis"], r0["doc_rank"], r0["hit@3"], r0["doc_hit@3"]), (2, "point", 1, True, True))   # chunk and document metrics kept apart (S08)
        self.assertEqual((r0["expect"], r0["expect_all"], r0["min_docs"]), (True, True, True))
        self.assertTrue(r1["negative"]); self.assertIn("boom", r2["error"]); self.assertFalse(r2["hit@5"])
        self.assertEqual(res["summary"]["hit@3"], "1/2"); self.assertEqual(res["summary"]["doc_hit@3"], "1/1"); self.assertEqual(res["summary"]["mrr"], 0.25)   # errored questions count in the denominator
        self.assertEqual((res["summary"]["errors"], res["summary"]["negative"], res["summary"]["avg_ms"]), (1, "1/1", 200))
        table = evalset.format_table(evalset.compare(res, {"summary": {"hit@3": "0/2", "mrr": 0.05}}))
        self.assertIn("+0.500", table); self.assertIn("+0.200", table)

    def test_negative_follows_what_the_service_tells_the_caller(self) -> None:
        def summary(**kw):
            return {"kbs": ["kb_001"], "sources": [{"role": "hit", "point_id": "p1", "text": "沾点边的内容"}], "specs": [], "pages": [], "retrieval_summary": kw}
        item = {"q": "库里没有的问题", "negative": True}
        rejected = evalset.evaluate_item(item, summary(rerank="below_threshold", rerank_max=0.03, evidence_state="diagnostic", no_relevant_content=True))
        # top score between the floor and the threshold: the service returned accepted evidence, so the evaluation must not
        # count it as a rejection just because the score is below 0.1
        accepted = evalset.evaluate_item(item, summary(rerank="floor_lowered", rerank_max=0.08, evidence_state="accepted", no_relevant_content=False))
        unranked = evalset.evaluate_item(item, summary(rerank="skipped", rerank_max=None, evidence_state="unranked", no_relevant_content=False))
        self.assertEqual((rejected["negative"], accepted["negative"], unranked["negative"]), (True, False, False))

    def test_stale_gold_is_listed_and_kept_out_of_chunk_metrics(self) -> None:
        def search(q, kbs=None, top_k=12):
            return {"kbs": ["kb_002"], "sources": [{"role": "hit", "point_id": "new-1", "doc_id": "d1", "rel_path": "x/d1.xlsx", "doc": "d1.xlsx", "text": "企业版 99 元"},
                                                     {"role": "hit", "point_id": "p-live", "doc_id": "d2", "rel_path": "x/d2.pdf", "doc": "d2.pdf", "text": "标准版 49 元"}],
                    "specs": [], "pages": [], "retrieval_summary": {"rerank": "ok", "rerank_max": 0.9, "timings_ms": {"total_ms": 100}}}
        items = [{"q": "企业版多少钱", "gold_point_ids": ["old-1"], "gold_docs": ["x/d1.xlsx"], "expect": ["99"], "kbs": ["kb_002"]},      # the gold point got a new id in a re-parse
                 {"q": "标准版多少钱", "gold_point_ids": ["p-live", "old-2"], "gold_docs": ["x/d2.pdf"], "expect": ["49"], "kbs": ["kb_002"]},
                 {"q": "只有文档金标", "gold_docs": ["x/d2.pdf"]}]
        asked = []

        def active(item, gold):
            asked.append((item["q"], gold))
            return {g for g in gold if g == "p-live"}
        self.assertEqual(evalset.check_gold(items, active), 1)
        self.assertEqual([q for q, _ in asked], ["企业版多少钱", "标准版多少钱"])                      # questions without chunk gold need no check
        self.assertEqual((items[0]["gold_stale"], items[1]["gold_point_ids"], items[1].get("gold_stale")), (True, ["p-live"], None))
        res = evalset.evaluate(items, search)
        r0, r1, r2 = res["items"]
        self.assertEqual((r0["gold_stale"], r0["basis"], r0["doc_hit@3"], r0["expect"]), (True, "stale", True, True))
        self.assertTrue(all(k not in r0 for k in ("rank", "rr", "hit@3", "hit@5", "hit@12")))          # stale questions stay out of the chunk metrics; document gold and expect as usual
        self.assertEqual((r1["rank"], r1["basis"], r2["basis"]), (2, "point", "doc"))
        sm = res["summary"]
        self.assertEqual((sm["stale_gold"], sm["hit@3"], sm["doc_hit@3"], sm["mrr"], sm["expect"]), (1, "2/2", "3/3", 0.5, "2/2"))
        self.assertIn("stale_gold", evalset.format_table(evalset.compare(res, None)))

        def boom(q, kbs=None, top_k=12):
            raise RuntimeError("down")
        failed = evalset.evaluate(items, boom)["items"][0]
        self.assertTrue(failed["gold_stale"]); self.assertNotIn("hit@5", failed); self.assertFalse(failed["doc_hit@5"])

    def test_gold_points_are_checked_in_the_libraries_the_question_names(self) -> None:
        settings = SimpleNamespace(sources={"kb_001": SimpleNamespace(collection="c1"), "kb_002": SimpleNamespace(collection="c2")})
        asked = []

        def meta(q, collection, ids, **kw):
            asked.append(collection)
            return {i: {"active": collection == "c2" and i == "g2"} for i in ids}
        with mock.patch.object(service, "runtime", return_value=(settings, _settings(), object())), \
                mock.patch.object(channels, "point_meta", side_effect=meta):
            active = evalset._active_points(None)
            self.assertEqual(active({"q": "x", "kbs": ["kb_001", "kb_999"]}, ["g1", "g2"]), set()); self.assertEqual(asked, ["c1"])
            self.assertEqual(active({"q": "x"}, ["g1", "g2"]), {"g2"}); self.assertEqual(asked, ["c1", "c1", "c2"])   # no knowledge base named: look in all of them
        with mock.patch.object(service, "runtime", return_value=(settings, _settings(), object())), \
                mock.patch.object(channels, "point_meta", side_effect=RuntimeError("400: not a valid point id")):
            items = [{"q": "手写的题", "gold_point_ids": ["chunk-12"], "gold_docs": ["x/d1.pdf"]}]
            self.assertEqual(evalset.check_gold(items, evalset._active_points(None)), 0)
            self.assertEqual((items[0]["gold_point_ids"], items[0].get("gold_stale")), (["chunk-12"], None))          # gold that cannot be checked is kept as is and the evaluation still runs

    def test_run_eval_checks_gold_and_refuses_an_empty_set(self) -> None:
        import contextlib
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            good = pathlib.Path(tmp) / "set.json"
            good.write_text(json.dumps({"items": [{"question": "企业版多少钱", "gold_point_ids": ["old-1"], "gold_docs": ["x/d1.xlsx"], "expect_all": "99"}]}), encoding="utf-8")
            search = lambda q, kbs=None, top_k=12: {"kbs": [], "sources": [{"role": "hit", "point_id": "n", "rel_path": "x/d1.xlsx", "text": "99 元"}],
                                                    "specs": [], "pages": [], "retrieval_summary": {}}
            res = evalset.run_eval(str(good), search_fn=search, gold_active=lambda item, gold: set())
            self.assertEqual((res["summary"]["stale_gold"], res["summary"]["hit@5"], res["items"][0]["expect_all"]), (1, None, True))   # expect_all written as a string is accepted too
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), mock.patch.object(evalset, "run_eval", return_value=res):
                evalset.cli_eval(SimpleNamespace(set=str(good), out=None, kbs=None, top_k=12, compare=None, auto=False))
            self.assertIn("!! chunk gold is stale for 1 question(s)", buf.getvalue())
            for body in ({"label": "x", "questions": [{"question": "q"}]}, [], {"items": {}}):
                bad = pathlib.Path(tmp) / "bad.json"
                bad.write_text(json.dumps(body), encoding="utf-8")
                with self.assertRaises(ValueError):
                    evalset.load_set(bad)                                                              # an unrecognised structure is no longer silently taken as 0 questions


class FusionTests(unittest.TestCase):
    def test_rrf_merge_keeps_every_channel_score_and_source(self) -> None:
        merged = rrf_merge({"text": [{"point_id": "p1", "score": 0.9}, {"point_id": "p2", "score": 0.8}],
                            "bm25": [{"point_id": "p2", "score": 12.0}, {"point_id": "p3", "score": 8.0}],
                            "graph": [{"point_id": "p2", "score": 0.5, "entities": ["X"]}]}, k=60)
        by = {c["point_id"]: c for c in merged}
        self.assertEqual(merged[0]["point_id"], "p2")                                   # hit by all three channels: ranks first
        self.assertEqual(by["p2"]["scores"], {"score_text": 0.8, "score_bm25": 12.0, "score_graph": 0.5})
        self.assertEqual(by["p2"]["recall_sources"], ["text", "bm25", "graph"]); self.assertEqual(by["p2"]["entities"], ["X"])
        self.assertEqual(by["p3"]["scores"], {"score_bm25": 8.0})
        self.assertAlmostEqual(by["p2"]["rrf"], 1 / 62 + 1 / 61 + 1 / 61)

    def test_interleave_gives_every_group_a_slot_and_dedupes(self) -> None:
        a = [{"point_id": "a1"}, {"point_id": "a2"}, {"point_id": "a3"}]
        b = [{"point_id": "b1"}, {"point_id": "a1"}, {"point_id": "b2"}]
        out = interleave([a, b], limit=5)
        self.assertEqual([c["point_id"] for c in out], ["a1", "b1", "a2", "b2", "a3"])


class ChannelTests(unittest.TestCase):
    def test_bm25_query_boosts_identifiers_as_phrases(self) -> None:
        body = channels.bm25_query("ZK7C4021KV13 的 tAS 是多少", ["ZK7C4021KV13", "tAS"])
        should = body["bool"]["should"]
        phrases = [s for s in should if s["multi_match"].get("type") == "phrase"]
        self.assertGreaterEqual(len(phrases), 2)
        self.assertTrue(all(s["multi_match"]["boost"] == channels.IDENTIFIER_BOOST for s in phrases))
        self.assertIn("ZK7C4021KV13", [s["multi_match"]["query"] for s in phrases])
        self.assertEqual(channels.question_identifiers("VCC 是 3.3 V,型号 ZK7C4021KV13")[:2], ["VCC", "ZK7C4021KV13"])

    def test_query_text_instruction_form(self) -> None:
        self.assertEqual(channels.query_text("尿酸多少", None), "尿酸多少")
        self.assertEqual(channels.query_text("尿酸多少", "Given a question, retrieve passages that answer it"),
                         "Instruct: Given a question, retrieve passages that answer it\nQuery: 尿酸多少")

    def test_hints_parse_and_filters(self) -> None:
        used, ignored = channels.parse_hints({"doc_ids": ["d1", " "], "block_types": "table", "content_version": "v1", "subject": "李", "rel_paths": []})
        self.assertEqual(used, {"doc_ids": ["d1"], "block_types": ["table"], "content_version": "v1"}); self.assertEqual(ignored, ["subject", "rel_paths"])
        flt = channels.hint_filter(used)
        self.assertEqual(len(flt.must), 3)
        self.assertEqual(channels.hint_os_filters(used), [{"terms": {"doc_id": ["d1"]}}, {"term": {"content_version": "v1"}}])
        body = channels.bm25_query("q", [], channels.hint_os_filters(used))
        self.assertEqual(len(body["bool"]["filter"]), 2)
        self.assertTrue(channels.hint_allows(used, {"doc_id": "d1", "content_version": "v1"})); self.assertFalse(channels.hint_allows(used, {"doc_id": "d2", "content_version": "v1"}))

    def test_table_head_is_the_first_piece_and_is_skipped_when_redundant(self) -> None:
        asked = []
        first = {"doc_id": "d1", "block_id": "b", "chunk_index": 10, "text": "TITLE: 表 3 时序参数\n| 参数 | 最小 | 最大 |\n|---|---|---|\n| tAS | 5 | 10 |"}

        class FakeQ:
            def scroll(self, **kw):
                asked.append(kw)
                return [SimpleNamespace(id="t0", payload=dict(first))], None

        hit = {"doc_id": "d1", "block_id": "b", "chunk_index": 40, "content_version": "v1", "text": "| 参数 | 最小 | 最大 |\n|---|---|---|\n| tHD | 1 | 2 |"}
        head = channels.table_head(FakeQ(), "kb_004", hit)
        self.assertEqual((head["point_id"], head["chunk_index"]), ("t0", 10))                                 # the first chunk has a table title the continuation lacks: attached
        self.assertEqual((asked[0]["limit"], asked[0]["order_by"].key, str(asked[0]["order_by"].direction.value)), (1, "chunk_index", "asc"))
        first["text"] = "| 参数 | 最小 | 最大 |\n|---|---|---|\n| tAS | 5 | 10 |"
        self.assertIsNone(channels.table_head(FakeQ(), "kb_004", hit))                                         # the hit already carries the header: no extra chunk
        first["text"] = "| 参数 | 最小 | 最大 |\n|---|---|---|\n注:典型值在 25 摄氏度下测得\n| tAS | 5 | 10 |"
        self.assertEqual(channels.table_head(FakeQ(), "kb_004", hit)["point_id"], "t0")                        # the table note is only in the first chunk
        first["text"] = "参数 最小 最大\ntAS 5 10"
        self.assertEqual(channels.table_head(FakeQ(), "kb_004", hit)["point_id"], "t0")                        # the first chunk is not a markdown table: when in doubt, attach it
        n = len(asked)
        native = {"doc_id": "d1", "block_id": "b", "chunk_index": 40, "text": "SHEET: 价格表 ROWS: 30-39 HEADER: 型号 | 单价\nA1 | 3.2"}
        self.assertIsNone(channels.table_head(FakeQ(), "kb_002", native)); self.assertEqual(len(asked), n)     # every chunk of a native table carries its HEADER: no lookup

    def test_rerank_documents_windows_map_back_to_candidates(self) -> None:
        long_text = "这是很长的一段正文。" * 300
        cands = [{"payload": {"filename": "a.pdf", "section_path": ["1"], "text": "短正文。"}}, {"payload": {"text": long_text}}]
        docs, owner = rerank_documents(cands, window_tokens=120, overlap_tokens=20)
        self.assertEqual(owner[0], 0); self.assertTrue(docs[0].startswith("a.pdf › 1\n"))
        self.assertGreater(owner.count(1), 1)
        scores = [0.2] + [0.1, 0.9, 0.3][:owner.count(1)] + [0.0] * max(0, owner.count(1) - 3)
        rr = Reranker("http://127.0.0.1:1/v1", model_id="m")
        with mock.patch.object(Reranker, "score", return_value=scores):
            best = rerank(rr, "q", cands, window_tokens=120, overlap_tokens=20)
        self.assertEqual(best[0], 0.2); self.assertEqual(best[1], max(scores[1:]))
        self.assertEqual(rr.url, "http://127.0.0.1:1/v1/rerank")


def _render_plain(src: str, query: str, document: str) -> str:
    """Stand-in used when jinja2 is not installed: the two templates hold only {{ expression }} blocks, evaluated
    for the few forms they use; the one newline at the end of the source is dropped, as Jinja does by default."""
    import ast
    import re

    def value(m):
        expr = " ".join(m.group(1).split())
        if expr[:1] in "\"'":
            return ast.literal_eval(expr)
        if expr.startswith("messages") and '"query"' in expr:
            return query
        if expr.startswith("messages") and '"document"' in expr:
            return document
        if expr.startswith("instruction"):
            return re.search(r'first \| default\("([^"]+)", true\)', expr).group(1)
        raise AssertionError(f"unexpected template expression: {expr}")

    return re.sub(r"\{\{(.*?)\}\}", value, src[:-1] if src.endswith("\n") else src, flags=re.S)


class RerankTemplateTests(unittest.TestCase):
    """The rendered rerank templates must match the model card's prompt format character for character: the score
    is the probability of yes against no after the prompt's last token, so with one newline missing from the suffix
    every rerank score shifts in meaning."""

    SYSTEM = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. '
              'Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')

    @staticmethod
    def _render(name: str, query: str, document: str) -> str:
        src = (pathlib.Path(__file__).resolve().parents[2] / "deployment/compose/assets" / name).read_text(encoding="utf-8")
        try:
            from jinja2.sandbox import ImmutableSandboxedEnvironment       # the settings vLLM uses when it renders templates through transformers
        except ImportError:
            return _render_plain(src, query, document)
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        return env.from_string(src).render(messages=[{"role": "query", "content": query}, {"role": "document", "content": document}])

    def test_text_reranker_prompt_matches_the_model_card(self) -> None:
        out = self._render("qwen3_reranker.jinja", "尿酸多少", "尿酸 433 umol/L")
        self.assertEqual(out, self.SYSTEM + "<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n"
                         "<Query>: 尿酸多少\n<Document>: 尿酸 433 umol/L<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")

    def test_visual_reranker_prompt_ends_with_the_generation_prefix(self) -> None:
        out = self._render("qwen3_vl_reranker.jinja", "架构图", "一张架构图")
        self.assertEqual(out, self.SYSTEM + "<Instruct>: Given a search query, retrieve relevant candidates that answer the query."
                         "<Query>:架构图\n<Document>:一张架构图<|im_end|>\n<|im_start|>assistant\n")


class RerankerReuseTests(unittest.TestCase):
    def test_model_id_is_resolved_once_and_forgotten_when_rejected(self) -> None:
        from kb_search import rerank as rerank_mod

        class Resp:
            def __init__(self, body, ok=True): self.body, self.ok = body, ok
            def json(self): return self.body
            def raise_for_status(self):
                if not self.ok:
                    raise RuntimeError("404")

        gets, posts = [], []

        def fake_get(url, timeout=None):
            gets.append(url)
            return Resp({"data": [{"id": f"model-{len(gets)}"}]})

        def fake_post(url, json=None, timeout=None):
            posts.append(json["model"])
            return Resp({"results": [{"index": 0, "relevance_score": 0.7}]}, ok=json["model"] != "gone")

        with mock.patch.dict(rerank_mod._model_ids, {}, clear=True), \
                mock.patch.object(rerank_mod._session, "get", side_effect=fake_get), \
                mock.patch.object(rerank_mod._session, "post", side_effect=fake_post):
            for _ in range(3):
                self.assertEqual(Reranker("http://rr/v1").score("q", ["d"]), [0.7])         # objects created per request share the resolved name
            self.assertEqual((gets, posts), (["http://rr/v1/models"], ["model-1"] * 3))
            rerank_mod._model_ids["http://rr/v1/models"] = "gone"                           # the service switched models: the old name is rejected once, then resolved again
            with self.assertRaises(RuntimeError):
                Reranker("http://rr/v1").score("q", ["d"])
            self.assertEqual(Reranker("http://rr/v1").score("q", ["d"]), [0.7])
            self.assertEqual(len(gets), 2)


class _GraphSession:
    """Neo4j session stand-in for the graph channel: returns fixed rows keyed on marker strings in the query and
    records every query."""

    def __init__(self, log: list) -> None:
        self.log = log

    def __enter__(self): return self
    def __exit__(self, *a): return False

    def run(self, cypher, **kw):
        text = str(getattr(cypher, "text", cypher))
        self.log.append((text, getattr(cypher, "timeout", None), kw))
        rows: list[dict] = []
        if "active_graph_version" in text:
            rows = [{"version": "v1"}]
        elif "max(e.pagerank)" in text:
            rows = [{"m": 0.5}]
        elif "count(c) AS n" in text:
            rows = [{"n": 10}]
        elif "count(m) AS n" in text:
            rows = [{"n": 40}]
        elif "m.count AS count" in text:
            rows = [{"eid": "e1", "point_id": "p1", "chunk_uid": "u1", "rel_path": "a.pdf", "count": 2, "hub": 3, "kind": "body"},
                    {"eid": "e1", "point_id": "p2", "chunk_uid": "u2", "rel_path": "a.pdf", "count": 1, "hub": 1, "kind": "body"},
                    {"eid": "e1", "point_id": "p3", "chunk_uid": "u3", "rel_path": "b.pdf", "count": 1, "hub": 1, "kind": "body"}]
        elif "EVIDENCES" in text:
            rows = [{"rid": "r1", "point_id": "p3", "chunk_uid": "u3", "rel_path": "b.pdf", "kind": "body"}]
        return SimpleNamespace(data=lambda: rows, single=lambda: rows[0] if rows else None)


class _GraphDriver:
    def __init__(self) -> None:
        self.log: list = []
        self.closed = 0

    def session(self): return _GraphSession(self.log)
    def close(self): self.closed += 1


class _GraphQdrant:
    """Stand-in for the graph collections and the main store: the entity / relation / page collections return one
    row each, the fact collection does not exist, and candidates fetch their text and vectors from the main store."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def query_points(self, collection_name, query, limit, with_payload, query_filter=None, using=None, **kw):
        self.calls.append(f"query:{collection_name}")
        if collection_name.endswith("_entity"):
            return SimpleNamespace(points=[SimpleNamespace(id="ge1", score=0.8, payload={"gr_id": "e1", "title": "尿酸", "type": "biomarker", "graph_version": "v1"})])
        if collection_name.endswith("_relation"):
            return SimpleNamespace(points=[SimpleNamespace(id="gr1", score=0.6, payload={"gr_id": "r1", "source": "尿酸", "target": "痛风", "type": "indicates",
                                                                                        "source_id": "e1", "target_id": "e2", "graph_version": "v1"})])
        if collection_name.endswith("_page"):
            return SimpleNamespace(points=[SimpleNamespace(id="gp1", score=0.3, payload={"gr_id": "pg1", "kind": "timeline", "title": "尿酸 时间线", "text": "2024 433",
                                                                                        "point_ids": [f"s{i}" for i in range(40)] * 2})])
        raise RuntimeError("collection not found")

    def scroll(self, collection_name, scroll_filter, limit, with_payload, with_vectors, **kw):
        self.calls.append(f"scroll:{collection_name}")
        return [], None

    def retrieve(self, collection_name, ids, with_payload, with_vectors, **kw):
        self.calls.append(f"retrieve:{collection_name}")
        return [SimpleNamespace(id=i, payload={"chunk_uid": f"u{i[1:]}", "rel_path": "a.pdf", "text": "尿酸 433 umol/L", "section_path": ["S"]},
                                vector={"text": [1.0, 0.0]}) for i in ids]


class GraphRecallTests(unittest.TestCase):
    """The graph channel's recall runs the whole graph_query against stand-ins: the shared client and driver, and
    graph-wide statistics cached per version."""

    def _query(self, **kw):
        from kb_pipeline.graph import recall

        source = SimpleNamespace(kb_id="kb_001", collection="kb_001")
        settings = SimpleNamespace(qdrant_url="http://q", qdrant_api_key=None)
        return recall.graph_query(settings, source, "尿酸多少", hops=1, chunk_limit=5, candidate_limit=5, vector=[1.0, 0.0], **kw)

    def test_shared_client_and_driver_are_used_and_left_open(self) -> None:
        from kb_pipeline.graph import recall

        q, driver = _GraphQdrant(), _GraphDriver()
        with mock.patch.dict(recall._version_stats, {}, clear=True), \
                mock.patch.object(recall, "qdrant_client", side_effect=AssertionError("must not create a Qdrant client")), \
                mock.patch.object(recall, "neo4j_driver", side_effect=AssertionError("must not create a Neo4j driver")):
            res = self._query(q=q, driver=driver)
        self.assertEqual(driver.closed, 0)                                                   # the shared driver is not its to close
        self.assertEqual(res["graph_version"], "v1"); self.assertEqual([c["point_id"] for c in res["chunks"]][:1], ["p3"])   # relation evidence is the main signal
        self.assertEqual({c["point_id"] for c in res["chunks"]}, {"p1", "p2", "p3"})
        self.assertEqual(res["entities"][0]["docs"], ["a.pdf", "b.pdf"])
        self.assertEqual(res["entities"][0]["point_ids"], ["p1", "p3"])                      # one source point per document, for the search service to check it is still active
        self.assertEqual(res["pages"][0]["point_ids"], [f"s{i}" for i in range(0, 40, 5)] * 2)   # a page's 16 source points are taken at even intervals, not just the first 16
        self.assertEqual(next(r for r in res["relations"] if r["id"] == "r1")["point_ids"], ["p3"])
        self.assertIn("retrieve:kb_001", q.calls)

    def test_source_points_sample_one_per_document_and_spread_out(self) -> None:
        from kb_pipeline.graph.recall import source_points

        rows = [{"eid": "e1", "point_id": f"p{d}-{i}", "rel_path": f"dir{d // 10}/doc{d:02d}.pdf"} for d in range(40) for i in range(3)]
        rows += [{"eid": "e2", "point_id": "q1", "rel_path": "x.pdf"}, {"eid": "e2", "point_id": "q2", "rel_path": None}, {"eid": "e2", "point_id": None, "rel_path": "y.pdf"}]
        got = source_points(rows, "eid", limit=8)
        self.assertEqual(got["e1"], [f"p{d}-0" for d in range(0, 40, 5)])                     # 8 out of 40 documents, all four directories represented
        self.assertEqual(sorted(got["e2"]), ["q1", "q2"])                                     # points without a path count as one document each; rows without a point id do not count

    def test_without_shared_driver_it_builds_one_and_closes_it(self) -> None:
        from kb_pipeline.graph import recall

        driver = _GraphDriver()
        with mock.patch.dict(recall._version_stats, {}, clear=True), \
                mock.patch.object(recall, "qdrant_client", return_value=_GraphQdrant()) as make_q, \
                mock.patch.object(recall, "neo4j_driver", return_value=driver):
            self._query()
        self.assertEqual((make_q.call_count, driver.closed), (1, 1))

    def test_version_statistics_are_computed_once_per_graph_version(self) -> None:
        from kb_pipeline.graph import recall

        def counted(driver):
            return sum(1 for text, _, _ in driver.log if "count(m) AS n" in text or "count(c) AS n" in text or "max(e.pagerank)" in text)

        driver = _GraphDriver()
        with mock.patch.dict(recall._version_stats, {}, clear=True):
            first = self._query(q=_GraphQdrant(), driver=driver)
            self.assertEqual(counted(driver), 3)
            second = self._query(q=_GraphQdrant(), driver=driver)
            self.assertEqual(counted(driver), 3)                                             # a second query on the same version does not recompute
            self.assertEqual([c["graph_score"] for c in first["chunks"]], [c["graph_score"] for c in second["chunks"]])
            self.assertEqual(recall._version_stats["kb_001"][2], {"max_rank": 0.5, "n_chunks": 10, "total_mentions": 40})
            recall._version_stats["kb_001"] = ("v0", *recall._version_stats["kb_001"][1:])   # version changed: recompute
            self._query(q=_GraphQdrant(), driver=driver)
            self.assertEqual(counted(driver), 6)
            recall._version_stats["kb_001"] = ("v1", 0.0, recall._version_stats["kb_001"][2])   # kept too long: recompute
            self._query(q=_GraphQdrant(), driver=driver)
            self.assertEqual(counted(driver), 9)
        mentions = [text for text, _, _ in driver.log if "count(m) AS n" in text]
        self.assertTrue(all("e.id IS NOT NULL" in text for text in mentions))                # the mention total can use the composite index on the entity id


class BackendLimitTests(unittest.TestCase):
    """Every back-end call in the search process is bounded: a job its caller has abandoned ends on its own when its
    time is up instead of holding on to the shared thread pool."""

    def test_collect_reports_how_long_it_waited(self) -> None:
        import time as _t
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=3) as pool:
            jobs = {"fast": pool.submit(lambda: "ok"), "slow": pool.submit(_t.sleep, 0.5), "slower": pool.submit(_t.sleep, 0.5)}
            out = service._collect(jobs, _t.time() + 0.2)
        self.assertEqual(out["fast"], (True, "ok"))
        self.assertRegex(str(out["slow"][1]), r"^deadline exceeded after 0\.[2-4]s$")             # gives up only after waiting the full time, no longer "0.2s left"
        self.assertRegex(str(out["slower"][1]), r"^deadline exceeded after 0\.[2-4]s$")

    def test_opensearch_client_does_not_retry_timeouts_and_every_call_is_bounded(self) -> None:
        client = channels.os_client("http://127.0.0.1:1")
        self.assertIs(channels.os_client("http://127.0.0.1:1"), client)
        self.assertEqual((client.transport.max_retries, client.transport.retry_on_timeout), (1, False))   # no retry on timeout; a broken old connection is reconnected once
        seen = []

        class Indices:
            def exists(self, index, request_timeout=None): seen.append(("exists", request_timeout)); return True
            def analyze(self, body, request_timeout=None): seen.append(("analyze", request_timeout)); return {"tokens": [{"token": "尿酸"}]}

        class FakeOS:
            indices = Indices()
            def search(self, index, body, request_timeout=None): seen.append(("search", request_timeout)); return {"hits": {"hits": []}}
            def msearch(self, body, request_timeout=None): seen.append(("msearch", request_timeout)); return {"responses": []}

        with mock.patch.object(channels, "os_client", return_value=FakeOS()):
            channels.bm25_channel("http://os", "尿酸", "kb_001", limit=5, timeout=3.5)
            channels.lexical_profile("http://os", "尿酸", ["kb_001"], timeout=3.5)
        self.assertEqual(seen, [("exists", 3.5), ("search", 3.5), ("exists", 3.5), ("analyze", 3.5), ("msearch", 3.5)])

    def test_qdrant_calls_carry_whole_second_timeouts(self) -> None:
        seen = []

        class FakeQ:
            def query_points(self, **kw): seen.append(kw["timeout"]); return SimpleNamespace(points=[])
            def retrieve(self, **kw): seen.append(kw["timeout"]); return []
            def scroll(self, **kw): seen.append(kw["timeout"]); return [], None

        q = FakeQ()
        channels.vector_channel(q, "kb_001", [0.1], limit=5, timeout=0.4)
        channels.visual_channel(q, "kb_001", [0.1], limit=5, timeout=2.0)
        channels.fetch_payloads(q, "kb_001", ["p1"], timeout=2.2)
        channels.point_meta(q, "kb_001", ["p1"], timeout=24.9)
        channels.neighbor_payloads(q, "kb_001", {"doc_id": "d1", "chunk_index": 3}, span=1, timeout=0.01)
        channels.table_head(q, "kb_001", {"doc_id": "d1", "chunk_index": 3, "block_id": "b", "text": "| 1 | 2 |"})
        self.assertEqual(seen, [1, 2, 3, 25, 1, None])                                             # rounded up, at least 1 second; without one the client's own limit applies

    def test_graph_queries_run_with_a_transaction_timeout(self) -> None:
        from kb_pipeline.graph import recall

        driver = _GraphDriver()
        source = SimpleNamespace(kb_id="kb_001", collection="kb_001")
        with mock.patch.dict(recall._version_stats, {}, clear=True):
            recall.graph_query(SimpleNamespace(), source, "尿酸多少", hops=1, chunk_limit=5, vector=[1.0, 0.0], q=_GraphQdrant(), driver=driver, timeout=7)
        timed = [t for text, t, _ in driver.log if "active_graph_version" not in text]
        self.assertGreaterEqual(len(timed), 5); self.assertEqual(set(timed), {7.0})


class SharedDriverTests(unittest.TestCase):
    def test_one_driver_per_connection_settings(self) -> None:
        import neo4j

        made = []

        def fake_driver(uri, auth=None, **kw):
            made.append(SimpleNamespace(uri=uri, auth=auth, options=kw, closed=False, close=lambda: made[0].__setattr__("closed", True)))
            return made[-1]

        settings = SimpleNamespace(neo4j_uri="bolt://n:7687", neo4j_user="neo4j", neo4j_password="pw")
        with mock.patch.dict(channels._driver, {"key": None, "driver": None}), \
                mock.patch.object(neo4j.GraphDatabase, "driver", side_effect=fake_driver):
            first = channels.shared_driver(settings)
            self.assertIs(channels.shared_driver(settings), first); self.assertEqual(len(made), 1)
            self.assertEqual((first.auth, first.options["liveness_check_timeout"]), (("neo4j", "pw"), channels.NEO4J_IDLE_CHECK_SECONDS))
            moved = SimpleNamespace(neo4j_uri="bolt://m:7687", neo4j_user="neo4j", neo4j_password="pw")
            self.assertIsNot(channels.shared_driver(moved), first); self.assertTrue(first.closed)        # connection details changed: a new driver, the old one closed
            with self.assertRaises(RuntimeError):
                channels.shared_driver(SimpleNamespace(neo4j_uri="bolt://x", neo4j_user="neo4j", neo4j_password=None))


class CatalogTests(unittest.TestCase):
    """KB routing only looks at the evidence the two cheap channels retrieve from each KB; profiles, graphs and
    subject matter play no part."""

    @staticmethod
    def _probe(**per_kb):
        return {kb: {"text": [{"point_id": f"{kb}-t{i}", "score": s} for i, s in enumerate(vec)], "bm25": []} for kb, vec in per_kb.items()}

    def test_evidence_is_top_n_mean_and_lexical(self) -> None:
        ev = catalog_mod.library_evidence(self._probe(kb_001=[0.9, 0.7, 0.5, 0.1], kb_002=[]), {"kb_001": 0.8}, top_n=3)
        self.assertEqual(ev["kb_001"], {"vec": 0.7, "lex": 0.8, "vis": 0.0}); self.assertEqual(ev["kb_002"], {"vec": 0.0, "lex": 0.0, "vis": 0.0})

    def test_lexical_evidence_uses_density_and_cross_library_rarity(self) -> None:
        cols = {"kb_001": "kb_001", "kb_002": "kb_002", "kb_003": "kb_003"}
        profile = {"sizes": {"kb_001": 500, "kb_002": 10000, "kb_003": 200},
                   "terms": {"尿酸": {"kb_001": 20},                                   # only one KB has it: max weight; kb_001 = 1 after density normalisation
                             "企业": {"kb_001": 2, "kb_002": 3000, "kb_003": 60},      # every KB has it: weight 0
                             "workbuddy": {"kb_002": 900, "kb_003": 8},                 # two KBs have it: kb_002 density 0.09 far above kb_003 at 0.04
                             "不存在": {}}}
        lex = catalog_mod.lexical_evidence(profile, cols)
        w1, w2 = math.log(4 / 2), math.log(4 / 3)                                            # three KBs: token weight for "only 1 KB has it" / "2 KBs have it"
        self.assertEqual(lex["kb_001"], round(w1 / (w1 + w2), 4))
        self.assertGreater(lex["kb_002"], lex["kb_003"]); self.assertEqual(lex["kb_003"], round(w2 * (0.04 / 0.09) / (w1 + w2), 4))
        self.assertEqual(catalog_mod.lexical_evidence({"sizes": {}, "terms": {"企业": {"kb_001": 1, "kb_002": 1, "kb_003": 1}}}, cols),
                         {"kb_001": 0.0, "kb_002": 0.0, "kb_003": 0.0})

    def test_route_by_evidence(self) -> None:
        kw = dict(max_kbs=2, gap=0.2, floor=0.45, lexical_weight=0.3)
        ev = catalog_mod.library_evidence(self._probe(kb_001=[0.72, 0.70, 0.66], kb_002=[0.48, 0.45, 0.44], kb_003=[0.40]),
                                          {"kb_001": 0.9, "kb_002": 0.2, "kb_003": 0.0})
        r = catalog_mod.route(ev, **kw)
        self.assertEqual((r["mode"], r["chosen"]), ("auto", ["kb_001"]))                       # one KB clearly ahead: query only that one
        self.assertEqual(r["scores"]["kb_001"]["score"], 1.0); self.assertLess(r["scores"]["kb_002"]["score"], 0.8)
        self.assertFalse(r["weak"])
        ev2 = catalog_mod.library_evidence(self._probe(kb_001=[0.70, 0.68], kb_002=[0.69, 0.66], kb_003=[0.3]), {"kb_001": 0.5, "kb_002": 0.5})
        r2 = catalog_mod.route(ev2, **kw)
        self.assertEqual(r2["chosen"], ["kb_001", "kb_002"])                                      # both look alike: query both, up to max_kbs
        r3 = catalog_mod.route(ev2, **(kw | {"max_kbs": 1}))
        self.assertEqual(r3["chosen"], ["kb_001"])
        ev3 = catalog_mod.library_evidence(self._probe(kb_001=[], kb_002=[]), {"kb_001": 0.1, "kb_002": 0.9})
        r4 = catalog_mod.route(ev3, **kw)
        self.assertEqual(r4["chosen"], ["kb_002"])                                                # vector channel unavailable altogether: all weight goes to the lexical evidence
        self.assertEqual(r4["scores"]["kb_002"]["score"], 1.0)
        ev4 = catalog_mod.library_evidence(self._probe(kb_001=[0.31, 0.30], kb_002=[0.28]), {})
        r5 = catalog_mod.route(ev4, **kw)
        self.assertTrue(r5["weak"]); self.assertEqual(r5["chosen"][0], "kb_001")                  # none looks right: still pick the closest, just flag weak
        r6 = catalog_mod.route_explicit(["kb_002", "kb_999"], [{"kb_id": "kb_001"}, {"kb_id": "kb_002"}])
        self.assertEqual((r6["mode"], r6["chosen"], r6["unknown"]), ("explicit", ["kb_002"], ["kb_999"]))


class CatalogCacheTests(unittest.TestCase):
    """Catalog cache: a catalog built while the main store was out of reach must not be cached for ten minutes as
    "0 chunks, no graph"; when a knowledge base is enrolled or closed the catalog is rebuilt at once."""

    class Q:
        def __init__(self) -> None:
            self.fail: Exception | None = None
            self.missing: set[str] = set()
            self.broken: set[str] = set()
            self.calls: list[str] = []

        def count(self, collection_name, count_filter, exact):
            self.calls.append(f"count:{collection_name}")
            if self.fail:
                raise self.fail
            if collection_name in self.missing or collection_name in self.broken:
                err = RuntimeError("Not found: Collection doesn't exist" if collection_name in self.missing else "Service internal error")
                err.status_code = 404 if collection_name in self.missing else 500
                raise err
            return SimpleNamespace(count=581)

        def get_aliases(self):
            self.calls.append("aliases")
            if self.fail:
                raise self.fail
            return SimpleNamespace(aliases=[SimpleNamespace(alias_name="graph_001_entity", collection_name="graph_001_entity__v7")])

    def _env(self, tmp: str, kbs=("kb_001", "kb_002")):
        from kb_pipeline import db

        path = pathlib.Path(tmp) / "state.db"
        db.init_db(path)
        with db.connect(path) as con:          # one active chunk per knowledge base: having a graph also requires active documents
            for kb in kbs:
                con.execute("INSERT OR IGNORE INTO chunks(chunk_uid, file_id, content_version, chunk_index, point_id, collection, status, "
                            "created_at) VALUES (?, ?, 'v1', 0, ?, ?, 'active', 0)", (f"{kb}:c0", f"{kb}:1", f"{kb}-p0", kb))
        src = lambda kb: SimpleNamespace(kb_id=kb, collection=kb, source_root=kb, graph_profile={}, graph_language="Chinese", graph_entity_types=[])
        return SimpleNamespace(state_db=path, sources={kb: src(kb) for kb in kbs})

    def test_degraded_build_is_marked_unknown_and_expires_quickly(self) -> None:
        import tempfile
        import time as _t

        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(catalog_mod._cache, {"at": 0.0, "entries": []}), \
                mock.patch.object(catalog_mod, "_active_schema", return_value={"domain": "", "persona": "", "schema_id": None}):
            settings, q = self._env(tmp), self.Q()
            q.fail = ConnectionError("qdrant is starting")
            down = catalog_mod.get_catalog(settings, q, ttl=600)
            self.assertEqual([(e["kb_id"], e["chunks"], e["has_graph"], e["degraded"]) for e in down],
                             [("kb_001", None, None, "qdrant: ConnectionError"), ("kb_002", None, None, "qdrant: ConnectionError")])
            self.assertEqual(q.calls, ["count:kb_001"])                                           # after one failure the other knowledge bases are not tried, each would wait for the timeout
            q.fail = None
            self.assertIs(catalog_mod.get_catalog(settings, q, ttl=600), down)                    # just built: used for now
            catalog_mod._cache["at"] = _t.time() - catalog_mod.DEGRADED_TTL - 1
            up = catalog_mod.get_catalog(settings, q, ttl=600)
            self.assertEqual([(e["kb_id"], e["chunks"], e["has_graph"], e["graph_version"], e.get("degraded")) for e in up],
                             [("kb_001", 581, True, "v7", None), ("kb_002", 581, False, None, None)])   # recovers within about fifteen seconds once the main store is back
            catalog_mod._cache["at"] = _t.time() - 300
            self.assertIs(catalog_mod.get_catalog(settings, q, ttl=600), up)                      # a catalog built normally is cached for ttl

    def test_missing_collection_is_an_empty_library_not_an_outage(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(catalog_mod._cache, {"at": 0.0, "entries": []}), \
                mock.patch.object(catalog_mod, "_active_schema", return_value={"domain": "", "persona": "", "schema_id": None}):
            settings, q = self._env(tmp), self.Q()
            q.missing = {"kb_001"}
            rows = catalog_mod.get_catalog(settings, q, ttl=600)
            self.assertEqual([(e["kb_id"], e["chunks"], e.get("degraded")) for e in rows], [("kb_001", 0, None), ("kb_002", 581, None)])
            q.missing, q.broken = set(), {"kb_001"}
            rows = catalog_mod.get_catalog(settings, q, ttl=600, force=True)
            self.assertEqual([(e["kb_id"], e["chunks"], e["has_graph"], e.get("degraded")) for e in rows],
                             [("kb_001", None, None, "qdrant: RuntimeError"), ("kb_002", 581, False, None)])   # the main store answered one knowledge base with an error: only that one becomes unknown

    def test_enrolling_or_closing_a_library_rebuilds_the_catalog_at_once(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(catalog_mod._cache, {"at": 0.0, "entries": []}), \
                mock.patch.object(catalog_mod, "_active_schema", return_value={"domain": "", "persona": "", "schema_id": None}):
            settings, q = self._env(tmp, kbs=("kb_001",)), self.Q()
            self.assertEqual([e["kb_id"] for e in catalog_mod.get_catalog(settings, q, ttl=600)], ["kb_001"])
            grown = self._env(tmp, kbs=("kb_001", "kb_007"))
            self.assertEqual([e["kb_id"] for e in catalog_mod.get_catalog(grown, q, ttl=600)], ["kb_001", "kb_007"])   # a newly enrolled knowledge base does not wait for the cache to expire
            self.assertEqual([e["kb_id"] for e in catalog_mod.get_catalog(settings, q, ttl=600)], ["kb_001"])


def _payload(pid: str, doc: str, idx: int, text: str, **extra):
    return {"point_id": pid, "chunk_uid": f"u{pid}", "doc_id": doc, "content_version": "v1", "chunk_index": idx, "chunk_total": 5,
            "filename": f"{doc}.pdf", "rel_path": f"dir/{doc}.pdf", "page_idx": idx, "section_path": ["S"], "block_type": "text",
            "text": text, "token_count": 20, "is_active": True, **extra}


class OrchestrationTests(unittest.TestCase):
    """Orchestration with stubs: two channels hitting the same chunk, the graph channel carrying entities, BM25
    returning only ids that need backfilling, the visual channel hitting an image chunk; rerank working and
    failing; multi-subject / multi-document quotas, boilerplate downweighting, widening."""

    def _run(self, *, rerank_fn=None, kbs=None, settings_over=None, question="我的尿酸多少", graph_entities=None, extra_vec=None, extra_bm25=None,
             image=None, visual_fn=None, vec_delay=None, graph_specs=None, hints=None, graph_delay=None, broken=(),
             visual_query_fn=None, entries=None, graph_relations=None, graph_pages=None, hoods=None):
        docs = {"p1": _payload("p1", "d1", 2, "尿酸 433 umol/L,参考范围 208-428。" * 12),
                "p2": _payload("p2", "d1", 3, "甘油三酯 1.7 mmol/L。" * 20),
                "p3": _payload("p3", "d2", 0, "封装类型 361-ball FCBGA。" * 15, visual_ref="parse/x.jpg", visual_summary="图", visual_value_conflicts=[{"a": 1}], visual_confidence="high"),
                "p4": _payload("p4", "d2", 1, "血糖 5.3 mmol/L,空腹。" * 15),
                "p5": _payload("p5", "d1", 0, "第一章 …… 1;第二章 …… 5" * 10, section_path=["目录"]),
                "n1": _payload("n1", "d1", 1, "血常规检查结果如下。"), "n3": _payload("n3", "d1", 4, "尿常规正常。"),
                "p9": _payload("p9", "d9", 0, "产品部署手册:安装步骤。" * 20),
                "p8": _payload("p8", "d8", 0, "架构图。", visual_ref="parse/y.jpg", visual_summary="架构图", visual_confidence="high")}
        src = SimpleNamespace(kb_id="kb_001", collection="kb_001", source_root="health-samples", graph_language="Chinese", graph_entity_types=[], graph_profile={})
        src2 = SimpleNamespace(kb_id="kb_002", collection="kb_002", source_root="产品资料", graph_language="Chinese", graph_entity_types=[], graph_profile={})
        settings = SimpleNamespace(sources={"kb_001": src, "kb_002": src2}, opensearch_url="http://os", reranker_base_url="http://rr/v1",
                                   embedding_base_url="", embedding_api_key="", embedding_model_id="", embedding_dim=2, qdrant_url="", qdrant_api_key=None,
                                   state_db=":memory:", visual_embedding_enabled=True, visual_embedding_base_url="http://vis/v1", visual_embedding_model_id="m",
                                   visual_embedding_api_key="", visual_embedding_dim=4, visual_embedding_instruction="Represent the user's input.")
        ss = _settings(**(settings_over or {}))
        entries = entries or [{"kb_id": "kb_001", "has_graph": True}, {"kb_id": "kb_002", "has_graph": False}]
        calls = {"rerank_docs": [], "rerank_calls": [], "graph": [], "visual_query": 0}
        ents = graph_entities if graph_entities is not None else [{"title": "尿酸", "type": "biomarker", "score": 0.7, "hop": 0, "via": "lexical", "docs": ["dir/d1.pdf"]}]

        def fake_rerank(self, query, documents):
            calls["rerank_docs"] = list(documents)
            calls["rerank_calls"].append(list(documents))
            if rerank_fn:
                return rerank_fn(documents)
            return [0.9 if "尿酸 433" in d else (0.5 if "甘油" in d else (0.3 if "血糖" in d else 0.1)) for d in documents]

        def fake_graph(settings, source, question, *, limit, hops, vector=None, lexical_only=False, **kw):
            calls["graph"].append(source.kb_id)
            calls["graph_timeout"] = kw.get("timeout")
            if graph_delay:
                import time as _t
                _t.sleep(graph_delay)
            return {"chunks": [{"point_id": "p1", "score": 0.4, "rel_path": "dir/d1.pdf", "entities": ["尿酸"], "relations": []}], "entities": ents,
                    "relations": graph_relations or [],
                    "specs": graph_specs if graph_specs is not None else [{"id": "s1", "subject": "李", "property": "尿酸", "value": "433", "score": 0.9, "point_ids": ["p1"]}],
                    "pages": graph_pages if graph_pages is not None else [{"kind": "subject", "title": "尿酸", "score": 0.8, "summary": "主体页", "text": "尿酸 时间线 433", "series": ["x"], "point_ids": ["p1"]}],
                    "graph_version": "001-v", "seeds": {"entities": 1}}

        def fake_visual_query(settings, *, text=None, image_bytes=None, timeout=20.0):
            calls["visual_query"] += 1
            calls["visual_image"] = image_bytes
            calls["visual_text"] = text
            return [0.1, 0.2, 0.3, 0.4]

        def default_visual(q, c, v, limit, timeout=None, query_filter=None):
            return [{"point_id": "p3", "score": 0.7, "payload": docs["p3"]}] if c == "kb_001" else []

        def fake_neighbors(q, collection, payload, *, span, **kw):
            if "neighbors" in broken:
                raise RuntimeError("qdrant timed out")
            idx = int(payload["chunk_index"])
            return [dict(p) for p in docs.values() if p["doc_id"] == payload["doc_id"] and p["chunk_index"] != idx and abs(p["chunk_index"] - idx) <= span]

        def fake_profile(url, question, collections, **kw):
            if "profile" in broken:
                raise RuntimeError("opensearch timed out")
            return {"sizes": {"kb_001": 5, "kb_002": 1}, "terms": {"尿酸": {"kb_001": 2}}}

        def fake_meta(q, c, ids, **kw):
            if "meta" in broken:
                raise RuntimeError("qdrant timed out")
            return {i: ({"active": True, "doc_id": docs[i]["doc_id"], "rel_path": docs[i]["rel_path"], "content_version": "v1"} if i in docs else {"active": False})
                    for i in ids}

        def fake_fetch(q, c, ids, **kw):
            if "fetch" in broken:
                raise RuntimeError("qdrant timed out")
            return {i: docs[i] for i in ids if i in docs}

        def fake_points(q, c, ids, *, keys=None, timeout=None):
            calls.setdefault("points", []).append(sorted(ids))
            if "points" in broken:
                raise RuntimeError("qdrant timed out")
            return {i: docs[i] for i in ids if i in docs}

        def fake_vec(q, c, v, limit, timeout=None, query_filter=None):
            calls["vec_filter"] = query_filter
            if "vec" in broken:
                raise RuntimeError("qdrant unreachable")
            if vec_delay and c == "kb_002":
                import time as _t
                _t.sleep(vec_delay)
            if c == "kb_001":
                return [{"point_id": "p1", "score": 0.8, "payload": docs["p1"]}, {"point_id": "p3", "score": 0.6, "payload": docs["p3"]}] + list(extra_vec or [])
            return [{"point_id": "p9", "score": 0.3, "payload": docs["p9"]}]

        def fake_hoods(settings, sources, graph_versions, question, seeds, *, limit, driver=None, timeout=None):
            calls["hoods"] = {"kbs": list(sources), "question": question, "seeds": [e.get("title") for e in seeds], "graph_versions": graph_versions,
                              "limit": limit, "timeout": timeout}
            if callable(hoods):
                return hoods()
            return list(hoods or [])

        def fake_bm25(url, question, c, limit, identifiers=None, timeout=None, filters=None):
            calls["bm25_filters"] = filters
            if "bm25" in broken:
                raise RuntimeError("opensearch unreachable")
            if c == "kb_001":
                return [{"point_id": "p2", "score": 9.0, "payload": {"doc_id": "d1", "rel_path": "dir/d1.pdf"}}, {"point_id": "p1", "score": 7.0, "payload": {"doc_id": "d1", "rel_path": "dir/d1.pdf"}}] + list(extra_bm25 or [])
            return []

        with mock.patch.object(service, "runtime", return_value=(settings, ss, object())), \
                mock.patch.object(catalog_mod, "get_catalog", return_value=entries), \
                mock.patch.object(channels, "embed_question", side_effect=lambda settings, question, timeout=10.0, instruction=None: [1.0, 0.0]), \
                mock.patch.object(channels, "vector_channel", side_effect=fake_vec), \
                mock.patch.object(channels, "point_meta", side_effect=fake_meta), \
                mock.patch.object(channels, "bm25_channel", side_effect=fake_bm25), \
                mock.patch.object(channels, "graph_channel", side_effect=fake_graph), \
                mock.patch.object(channels, "lexical_profile", side_effect=fake_profile), \
                mock.patch.object(channels, "embed_visual_query", side_effect=visual_query_fn or fake_visual_query), \
                mock.patch.object(channels, "visual_channel", side_effect=visual_fn or default_visual), \
                mock.patch.object(channels, "table_head", return_value=None), \
                mock.patch.object(channels, "fetch_payloads", side_effect=fake_fetch), \
                mock.patch.object(channels, "neighbor_payloads", side_effect=fake_neighbors), \
                mock.patch.object(channels, "shared_driver", return_value=object()), \
                mock.patch.object(graphwalk, "neighborhoods", side_effect=fake_hoods), \
                mock.patch.object(graphwalk, "point_payloads", side_effect=fake_points), \
                mock.patch.object(Reranker, "score", fake_rerank):
            out = service.search(question, kbs=kbs, explain=True, image_bytes=image, hints=hints)
        return out, calls

    def test_search_facts_carry_the_chunks_they_rest_on(self) -> None:
        """Facts of /search carry their source chunks: locators enough for /context, the chunk that holds the value first
        when that can be told; a failed lookup is only a degraded note and the facts are returned all the same."""
        specs = [{"id": "s1", "subject": "李", "property": "尿酸", "value": "433", "score": 0.9, "point_ids": ["p1"]},
                 {"id": "s2", "subject": "系统", "property": "结构", "value": "架构图", "score": 0.8, "point_ids": ["p1", "p8"]}]
        out, calls = self._run(graph_specs=specs)
        by = {sp["id"]: sp for sp in out["specs"]}
        self.assertEqual([(e["point_id"], e["doc_id"], e["chunk_index"], e["active"]) for e in by["s1"]["evidence"]], [("p1", "d1", 2, True)])
        self.assertNotIn("located", by["s1"]["evidence"][0])                          # a single chunk: nothing to tell apart
        self.assertEqual([(e["point_id"], e.get("located")) for e in by["s2"]["evidence"]], [("p8", True), ("p1", None)])   # the value is in p8 only
        self.assertEqual((by["s2"]["evidence"][0]["doc_id"], by["s2"]["evidence"][0]["chunk_index"]), ("d8", 0))
        self.assertEqual(calls["points"], [["p8"]])                                   # a chunk already among the sources (p1) is not fetched again
        self.assertIn("spec_evidence_ms", out["retrieval_summary"]["timings_ms"])
        broken, _ = self._run(graph_specs=specs, broken=("points",))
        self.assertIn("kb_001:spec_evidence: RuntimeError", broken["retrieval_summary"]["degraded"])
        by = {sp["id"]: sp for sp in broken["specs"]}
        self.assertEqual(by["s1"]["evidence"][0]["chunk_index"], 2)                   # chunks among the sources are unaffected
        self.assertEqual([(e["point_id"], e.get("active")) for e in by["s2"]["evidence"]], [("p1", True), ("p8", None)])   # not fetched: unknown, not marked inactive

    def test_search_carries_the_neighbourhood_of_named_subjects(self) -> None:
        """A search carries the one-hop neighbourhood of its subjects: the caller reads it and decides whether to call the
        neighbours endpoint and walk on. These are leads that come along: switching them off, a scoped query or a failure
        to fetch them leaves the search itself untouched."""
        block = {"kb_id": "kb_001", "id": "e1", "title": "尿酸", "type": "biomarker", "named": True, "relations": 3, "facts": 2,
                 "predicates": [{"type": "measured_in", "count": 3}],
                 "neighbors": [{"relation_id": "r1", "type": "measured_in", "direction": "out", "directed": True, "weight": 2.0,
                                "other": {"id": "e2", "title": "血液", "type": "specimen", "degree": 9}}]}
        out, calls = self._run(hoods=[block])
        self.assertEqual(out["neighborhoods"], [{"n": 1, **block}])
        self.assertEqual((calls["hoods"]["kbs"], calls["hoods"]["seeds"], calls["hoods"]["graph_versions"], calls["hoods"]["limit"]),
                         (["kb_001"], ["尿酸"], {"kb_001": "001-v"}, 4))                             # only knowledge bases with a graph; the graph route's entity rows are the seeds that fill up
        self.assertLessEqual(calls["hoods"]["timeout"], service.NEIGHBORHOOD_TIMEOUT)               # a slow graph database does not hold the request up
        self.assertIn("neighborhood_ms", out["retrieval_summary"]["timings_ms"]); self.assertEqual(out["retrieval_summary"]["degraded"], [])
        off, calls_off = self._run(hoods=[block], settings_over={"neighborhoods": 0})
        self.assertEqual(off["neighborhoods"], []); self.assertNotIn("hoods", calls_off)
        scoped, calls_scoped = self._run(hoods=[block], kbs=["kb_001"], hints={"doc_ids": ["d1"]})
        self.assertEqual(scoped["neighborhoods"], []); self.assertNotIn("hoods", calls_scoped)      # scoped queries do not carry them: relations have no document information

        def boom():
            raise RuntimeError("neo4j timed out")
        broken, _ = self._run(hoods=boom)
        self.assertEqual(broken["neighborhoods"], []); self.assertIn("neighborhoods: RuntimeError", broken["retrieval_summary"]["degraded"])
        self.assertTrue([r for r in broken["sources"] if r["role"] == "hit"])                       # a failure only records a degradation, the evidence is returned as usual

    def test_hints_restrict_documents_and_prefer_block_types(self) -> None:
        out, calls = self._run(kbs=["kb_001"], hints={"doc_ids": ["d2"], "block_types": ["image"], "subject": "李"}, settings_over={"top_k": 5})
        s = out["retrieval_summary"]
        self.assertTrue(s["hints_applied"]); self.assertEqual(s["hints_used"], {"doc_ids": ["d2"], "block_types": ["image"]}); self.assertEqual(s["hints_ignored"], ["subject"])
        self.assertIsNotNone(calls["vec_filter"]); self.assertEqual(calls["bm25_filters"], [{"terms": {"doc_id": ["d2"]}}])   # the filter was pushed into both channels
        hits = [r for r in out["sources"] if r["role"] == "hit"]
        self.assertTrue(hits and all(r["doc_id"] == "d2" for r in hits))                                                       # the d1 candidate from the graph channel is filtered out too
        self.assertEqual(s["hints_scope"], {"specs_dropped": 1, "pages_dropped": 1, "entities_dropped": 1, "relations_omitted": 0})   # derived evidence is restricted as well (R1)
        self.assertEqual((out["specs"], out["pages"], out["entities"]), ([], [], []))
        out2, _ = self._run(kbs=["kb_001"], hints={"block_types": ["text"]}, settings_over={"top_k": 5})
        p2 = next(r for r in out2["sources"] if r["point_id"] == "p2")
        self.assertEqual(p2["scores"]["score_final"], 0.6); self.assertEqual(out2["retrieval_summary"]["downweighted"]["hinted_blocks"], 3)   # soft preference ×1.2

    def test_search_returns_numbered_evidence_with_all_scores(self) -> None:
        out, calls = self._run(kbs=["kb_001"])
        s = out["retrieval_summary"]
        self.assertEqual(s["kbs"], ["kb_001"]); self.assertEqual(s["routing"]["mode"], "explicit")
        self.assertEqual(s["channels"]["kb_001"]["text"]["candidates"], 2); self.assertEqual(s["channels"]["kb_001"]["bm25"]["candidates"], 2)
        self.assertEqual(s["channels"]["kb_001"]["graph"]["candidates"], 1); self.assertEqual(s["channels"]["kb_001"]["visual"]["candidates"], 1)
        self.assertEqual(s["channels"]["kb_001"]["merged"], 3); self.assertEqual(s["visual"], "ok"); self.assertEqual(calls["visual_query"], 1)
        self.assertEqual(s["rerank"], "ok"); self.assertEqual(s["degraded"], []); self.assertFalse(s["low_confidence"]); self.assertEqual(s["rerank_max"], 0.9)
        hits = [r for r in out["sources"] if r["role"] == "hit"]
        self.assertEqual([h["point_id"] for h in hits], ["p1", "p2", "p3"])                       # the rerank score decides the order
        self.assertEqual(hits[0]["scores"], {"score_text": 0.8, "score_bm25": 7.0, "score_graph": 0.4, "score_rerank": 0.9, "score_final": 0.9})
        self.assertEqual(hits[0]["recall_sources"], ["text", "bm25", "graph"]); self.assertEqual(hits[0]["entities"], ["尿酸"])
        self.assertEqual(hits[2]["recall_sources"], ["text", "visual"])                            # RRF of the visual and text channels
        self.assertEqual(hits[1]["scores"]["score_bm25"], 9.0); self.assertIn("甘油三酯 1.7 mmol/L。", hits[1]["text"])   # the id-only hit got its text backfilled
        self.assertEqual(hits[1]["stitched"]["own_chars"], 320)                                             # the short chunk was stitched with its neighbours
        self.assertTrue(len(hits[1]["text"]) > 320 or hits[1].get("text_truncated"))                        # test budget is 400 tokens: the excess is cut to an excerpt (S07)
        self.assertEqual(s["sources"]["truncated_hits"], 2); self.assertTrue(hits[1]["stitched"]["pieces"][0]["point_id"])
        self.assertEqual(hits[0]["position"], "page 2 · S")
        self.assertEqual([r["n"] for r in out["sources"]], list(range(1, len(out["sources"]) + 1)))
        nbs = [r for r in out["sources"] if r["role"] == "neighbor"]
        self.assertTrue(all(r["doc_id"] in ("d1", "d2") for r in nbs)); self.assertGreaterEqual(s["sources"]["stitched"], 2)   # short chunks stitched their neighbours in
        self.assertEqual(hits[2]["visual"]["value_conflicts"], 1); self.assertIn("conflicts", hits[2]["visual"]["note"])
        self.assertEqual(out["specs"][0]["n"], 1); self.assertEqual(out["specs"][0]["hint"], "李 · 尿酸 = 433"); self.assertEqual(out["specs"][0]["sources"], [1])
        self.assertEqual((out["specs"][0]["verified"], out["specs"][0]["id"]), (True, "s1")); self.assertEqual(s["evidence_state"], "accepted")
        self.assertTrue(all(h["accepted"] for h in hits)); self.assertFalse(s["no_relevant_content"]); self.assertEqual(s["graph_versions"], {"kb_001": "001-v"})
        self.assertEqual(out["entities"][0]["title"], "尿酸"); self.assertTrue(out["pages"][0]["compiled"]); self.assertEqual(out["pages"][0]["text"], "尿酸 时间线 433")
        self.assertEqual(out["doc_aggs"][0]["doc"], "d1.pdf"); self.assertEqual(out["doc_aggs"][0]["hits"], 2)
        self.assertTrue(all("|" not in d for d in calls["rerank_docs"]))
        self.assertIn("avg_redundancy", s["selection"]); self.assertEqual(s["downweighted"], {"boilerplate": 0, "visual_low": 0, "hinted_blocks": 0})

    def test_rerank_failure_falls_back_to_fused_order_and_is_reported(self) -> None:
        def boom(documents):
            raise RuntimeError("8102 down")
        out, _ = self._run(rerank_fn=boom, kbs=["kb_001"])
        s = out["retrieval_summary"]
        self.assertEqual(s["rerank"], "skipped"); self.assertIn("rerank: RuntimeError", s["degraded"]); self.assertTrue(s["low_confidence"])
        hits = [r for r in out["sources"] if r["role"] == "hit"]
        self.assertEqual(hits[0]["point_id"], "p1")                                                 # fused order: hit by three channels comes first
        self.assertNotIn("score_rerank", hits[0]["scores"]); self.assertEqual(hits[0]["scores"]["score_final"], 1.0); self.assertIsNone(s["rerank_max"])
        self.assertEqual(s["evidence_state"], "unranked"); self.assertIsNone(hits[0]["accepted"])

    def test_auto_routing_picks_the_library_with_evidence(self) -> None:
        out, calls = self._run()
        r = out["retrieval_summary"]["routing"]
        self.assertEqual((r["mode"], r["probed"], out["kbs"]), ("auto", ["kb_001", "kb_002"], ["kb_001"]))
        self.assertEqual(r["scores"]["kb_001"], {"vec": 0.7, "lex": 1.0, "score": 1.0})           # both KBs probed; the one with weak evidence loses
        self.assertEqual(r["scores"]["kb_002"]["lex"], 0.0); self.assertLess(r["scores"]["kb_002"]["score"], 0.8)
        self.assertEqual(r["terms"], ["尿酸"])
        self.assertFalse(r["widened"]); self.assertEqual(calls["graph"], ["kb_001"])                  # the graph channel only runs on the chosen KB
        self.assertNotIn("kb_002", out["retrieval_summary"]["channels"])

    def test_library_without_graph_searches_with_cheap_channels_only(self) -> None:
        out, calls = self._run(kbs=["kb_002"])
        s = out["retrieval_summary"]
        self.assertEqual(out["kbs"], ["kb_002"]); self.assertEqual(calls["graph"], [])
        self.assertEqual(set(s["channels"]["kb_002"]), {"text", "bm25", "visual", "merged"}); self.assertEqual(s["degraded"], [])
        self.assertEqual([r["point_id"] for r in out["sources"] if r["role"] == "hit"], ["p9"])
        self.assertEqual(out["entities"], []); self.assertEqual(out["specs"], []); self.assertEqual(out["pages"], [])

    def test_widening_when_chosen_libraries_have_nothing_above_threshold(self) -> None:
        out, calls = self._run(rerank_fn=lambda docs: [0.05] * len(docs), settings_over={"rerank_threshold": 0.9})
        s = out["retrieval_summary"]
        self.assertEqual(s["rerank"], "below_threshold"); self.assertTrue(s["routing"]["widened"])
        self.assertEqual(out["kbs"], ["kb_001", "kb_002"]); self.assertEqual(calls["graph"], ["kb_001"])   # kb_002 has no graph; widening does not run the graph channel
        self.assertIn("widen_ms", s["timings_ms"]); self.assertTrue(s["low_confidence"]); self.assertEqual(s["rerank_max"], 0.05)
        self.assertIn("p9", [r["point_id"] for r in out["sources"] if r["role"] == "hit"])
        self.assertTrue(s["no_relevant_content"]); self.assertEqual(s["evidence_state"], "diagnostic")                 # everything below the floor: diagnostic candidates (S05)
        self.assertTrue(all(r["accepted"] is False for r in out["sources"] if r["role"] == "hit")); self.assertEqual(out["specs"][0]["state"], "diagnostic")
        first, second = calls["rerank_calls"]
        self.assertEqual(len(first), 3); self.assertEqual(len(second), 1); self.assertIn("产品部署手册", second[0])   # candidates scored in the first round are not sent again
        self.assertEqual(s["reranked"], 4)
        out2, _ = self._run(rerank_fn=lambda docs: [0.05] * len(docs), settings_over={"rerank_threshold": 0.9, "route_widen": False})
        self.assertFalse(out2["retrieval_summary"]["routing"]["widened"]); self.assertEqual(out2["kbs"], ["kb_001"])

    def test_image_query_routes_by_visual_evidence_and_keeps_visual_hits(self) -> None:
        def vis(q, c, v, limit, timeout=None, query_filter=None):
            return [{"point_id": "p8", "score": 0.99, "payload": {"doc_id": "d8", "rel_path": "dir/d8.pdf", "visual_ref": "parse/y.jpg", "text": "架构图。", "chunk_index": 0}}] if c == "kb_002" else []
        out, calls = self._run(image=b"IMG", question="找与这张图相似的图", visual_fn=vis, settings_over={"top_k": 3})
        s = out["retrieval_summary"]; r = s["routing"]
        self.assertEqual(calls["visual_image"], b"IMG"); self.assertIsNone(calls["visual_text"]); self.assertTrue(s["image_query"])
        self.assertIn("kb_002", out["kbs"]); self.assertGreater(r["scores"]["kb_002"]["vis"], r["scores"]["kb_001"]["vis"])   # visual evidence takes part in KB selection (S02)
        hits = [x for x in out["sources"] if x["role"] == "hit"]
        p8 = next(x for x in hits if x["point_id"] == "p8")
        self.assertEqual(p8["scores"]["score_visual"], 0.99); self.assertGreaterEqual(p8["scores"]["score_final"], 0.99)       # the text rerank cannot push the image chunk down
        self.assertTrue(p8["accepted"]); self.assertGreaterEqual(s["selection"]["priority_filled"], 1)

    def test_lexical_tiebreak_in_final_score(self) -> None:
        out, _ = self._run(kbs=["kb_001"], question="尿酸 433 参考范围", settings_over={"final_lex_weight": 0.2, "top_k": 3})
        hits = [r for r in out["sources"] if r["role"] == "hit"]
        p1 = next(r for r in hits if r["point_id"] == "p1")
        self.assertIn("score_lex", p1["scores"]); self.assertGreater(p1["scores"]["score_lex"], 0.5)
        self.assertAlmostEqual(p1["scores"]["score_final"], round(0.8 * 0.9 + 0.2 * p1["scores"]["score_lex"], 4), places=3)

    def test_stale_graph_sources_are_marked_unverified(self) -> None:
        specs = [{"id": "s1", "subject": "李", "property": "尿酸", "value": "433", "score": 0.9, "point_ids": ["p1"]},
                 {"id": "s2", "subject": "李", "property": "肌酐", "value": "78", "score": 0.8, "point_ids": ["gone"]}]
        out, _ = self._run(kbs=["kb_001"], graph_specs=specs)
        by = {sp["id"]: sp for sp in out["specs"]}
        self.assertEqual((by["s1"]["verified"], by["s2"]["verified"], by["s2"]["sources_active"]), (True, False, "0/1"))
        self.assertIsNone(by["s2"].get("hint")); self.assertIn("verify_ms", out["retrieval_summary"]["timings_ms"])

    def test_entities_relations_and_pages_from_deleted_documents_are_marked(self) -> None:
        ents = [{"id": "e1", "title": "尿酸", "type": "biomarker", "score": 0.7, "hop": 0, "via": "lexical", "docs": ["dir/d1.pdf"], "point_ids": ["p1", "gone"]},
                {"id": "e2", "title": "旧套餐", "type": "product", "score": 0.6, "hop": 0, "via": "vector", "docs": ["dir/deleted.xlsx"], "point_ids": ["gone", "gone2"]},
                {"id": "e3", "title": "痛风", "type": "disease", "score": 0.5, "hop": 1, "via": "expand", "docs": [], "point_ids": []}]
        rels = [{"id": "r1", "source": "尿酸", "type": "indicates", "target": "痛风", "score": 0.6, "hop": 0, "via": "vector", "point_ids": ["p2"]},
                {"id": "r2", "source": "旧套餐", "type": "includes", "target": "权益", "score": 0.5, "hop": 0, "via": "vector", "point_ids": ["gone"]}]
        pages = [{"id": "pg1", "kind": "timeline", "title": "旧套餐 时间线", "score": 0.9, "summary": "概述", "text": "旧套餐 2024 99 元", "series": ["x"], "point_ids": ["gone"]},
                 {"id": "pg2", "kind": "timeline", "title": "尿酸 时间线", "score": 0.8, "summary": "概述", "text": "尿酸 2024 433", "series": ["x"], "point_ids": ["p1"]}]
        out, _ = self._run(kbs=["kb_001"], graph_entities=ents, graph_relations=rels, graph_pages=pages)
        e = {r["id"]: r for r in out["entities"]}; r = {x["id"]: x for x in out["relationships"]}; pg = {x["id"]: x for x in out["pages"]}
        self.assertEqual((e["e1"]["verified"], e["e1"]["sources_active"]), (True, "1/2"))
        self.assertEqual((e["e2"]["verified"], e["e2"]["sources_active"]), (False, "0/2"))          # only from deleted documents: marked, not a current basis
        self.assertNotIn("verified", e["e3"]); self.assertNotIn("point_ids", e["e1"])                # rows without source point information are not marked
        self.assertEqual((r["r1"]["verified"], r["r2"]["verified"], r["r2"]["sources_active"]), (True, False, "0/1"))
        self.assertEqual((pg["pg1"]["verified"], pg["pg1"].get("text"), pg["pg1"]["summary"]), (False, None, "概述"))   # a page whose sources are gone gets no body text
        self.assertEqual((pg["pg2"]["verified"], pg["pg2"]["text"]), (True, "尿酸 2024 433"))

    def test_source_check_failure_leaves_derived_evidence_unmarked(self) -> None:
        ents = [{"id": "e1", "title": "尿酸", "type": "biomarker", "score": 0.7, "hop": 0, "via": "lexical", "docs": ["dir/d9.pdf"], "point_ids": ["x1"]}]
        specs = [{"id": "s1", "subject": "李", "property": "肌酐", "value": "78", "score": 0.9, "point_ids": ["x2"]}]
        pages = [{"id": "pg1", "kind": "timeline", "title": "肌酐 时间线", "score": 0.9, "summary": "概述", "text": "肌酐 2024 78", "series": ["x"], "point_ids": ["x3"]}]
        out, _ = self._run(kbs=["kb_001"], graph_entities=ents, graph_specs=specs, graph_pages=pages, broken=("meta",))
        self.assertIn("kb_001:point_meta: RuntimeError", out["retrieval_summary"]["degraded"])
        self.assertNotIn("verified", out["entities"][0])                                              # not checked means "unknown", not "no longer valid"
        self.assertNotIn("verified", out["specs"][0]); self.assertEqual(out["specs"][0]["hint"], "李 · 肌酐 = 78")
        self.assertNotIn("verified", out["pages"][0]); self.assertEqual(out["pages"][0]["text"], "肌酐 2024 78")

    def test_slow_channel_is_dropped_at_the_deadline(self) -> None:
        import time as _t
        t = _t.time()
        out, _ = self._run(vec_delay=0.8, settings_over={"channel_timeout": 0.15})
        s = out["retrieval_summary"]
        self.assertLess(_t.time() - t, 0.7); self.assertIn("kb_002:text", s["degraded"])                              # a slow channel is dropped at the deadline instead of holding the whole request (S06)
        self.assertEqual(out["kbs"], ["kb_001"]); self.assertTrue([r for r in out["sources"] if r["role"] == "hit"])

    def test_image_query_without_a_visual_vector_is_reported_as_degraded(self) -> None:
        def down(settings, *, text=None, image_bytes=None, timeout=20.0):
            raise ConnectionError("8103 down")
        out, _ = self._run(image=b"IMG", question="找与这张图相似的图", visual_query_fn=down)
        s = out["retrieval_summary"]
        self.assertTrue(s["image_query"]); self.assertTrue(s["visual"].startswith("skipped: ConnectionError"))
        self.assertIn("visual_query: ConnectionError", s["degraded"]); self.assertTrue(s["low_confidence"])   # the result comes from the text alone, and that has to be said
        self.assertEqual(s["selection"]["priority_filled"], 0)
        off, _ = self._run(image=b"IMG", question="找与这张图相似的图", settings_over={"visual_enabled": False})
        self.assertIn("visual_query: disabled", off["retrieval_summary"]["degraded"])
        text_only, _ = self._run(kbs=["kb_001"], visual_query_fn=down)
        self.assertEqual(text_only["retrieval_summary"]["degraded"], [])                                       # for a text question the visual channel is only an aid: its failure is no degradation

    def test_lexical_profile_failure_is_reported(self) -> None:
        out, _ = self._run(broken=("profile",))
        s = out["retrieval_summary"]
        self.assertIn("lexical_profile: RuntimeError", s["degraded"]); self.assertTrue(s["routing"]["lexical"].startswith("skipped: RuntimeError"))
        self.assertEqual(out["kbs"], ["kb_001"])                                                                # selection falls back to the vector evidence alone

    def test_unknown_graph_state_still_tries_the_graph_channel(self) -> None:
        entries = [{"kb_id": "kb_001", "has_graph": None, "chunks": None, "degraded": "qdrant: ResponseHandlingException"},
                   {"kb_id": "kb_002", "has_graph": None, "chunks": None, "degraded": "qdrant: ResponseHandlingException"}]
        out, calls = self._run(kbs=["kb_001"], entries=entries)
        s = out["retrieval_summary"]
        self.assertEqual(calls["graph"], ["kb_001"]); self.assertEqual(s["channels"]["kb_001"]["graph"]["candidates"], 1)
        self.assertEqual(s["degraded"], ["catalog: qdrant: ResponseHandlingException"]); self.assertTrue(s["low_confidence"])

    def test_stage_timeouts_shrink_with_the_request_budget(self) -> None:
        import time as _t
        t = _t.time()
        out, calls = self._run(kbs=["kb_001"], graph_delay=0.8, settings_over={"request_budget": 0.3, "channel_timeout": 5.0})
        s = out["retrieval_summary"]
        self.assertLess(_t.time() - t, 0.7)                                                     # the graph channel is waited for by the remaining budget, not its own 5 seconds
        self.assertLessEqual(calls["graph_timeout"], 0.3); self.assertIn("kb_001:graph", s["degraded"])
        self.assertRegex(s["channels"]["kb_001"]["graph"]["skipped"], r"deadline exceeded after 0\.\ds$")   # reports how long it actually waited
        self.assertEqual(s["rerank"], "skipped"); self.assertIn("rerank: TimeoutError", s["degraded"])          # budget spent: no rerank, back to the fusion order
        self.assertEqual(calls["rerank_calls"], []); self.assertIn("budget: context skipped", s["degraded"])
        hits = [r for r in out["sources"] if r["role"] == "hit"]
        self.assertTrue(hits); self.assertNotIn("neighbor", {r["role"] for r in out["sources"]})                 # hits are returned as usual, only without neighbourhood backfill
        out2, calls2 = self._run(kbs=["kb_001"], graph_delay=0.2, settings_over={"request_budget": 0, "channel_timeout": 5.0})
        self.assertEqual((calls2["graph_timeout"], out2["retrieval_summary"]["degraded"]), (5.0, []))            # 0 = no overall budget

    def test_widening_is_skipped_when_the_budget_cannot_cover_it(self) -> None:
        out, calls = self._run(rerank_fn=lambda docs: [0.05] * len(docs), graph_delay=0.5,
                               settings_over={"rerank_threshold": 0.9, "request_budget": 0.9, "channel_timeout": 5.0})
        s = out["retrieval_summary"]
        self.assertIn("budget: widen skipped", s["degraded"]); self.assertFalse(s["routing"]["widened"])
        self.assertEqual((out["kbs"], s["rerank"], s["no_relevant_content"]), (["kb_001"], "below_threshold", True))   # the first round's conclusion is returned as usual
        self.assertEqual(len(calls["rerank_calls"]), 1); self.assertNotIn("widen_ms", s["timings_ms"])

    def test_failed_widening_keeps_the_first_round_verdict(self) -> None:
        rounds = []

        def rerank_fn(docs):
            rounds.append(len(docs))
            if len(rounds) > 1:
                raise RuntimeError("8102 timed out")
            return [0.05] * len(docs)
        out, _ = self._run(rerank_fn=rerank_fn, settings_over={"rerank_threshold": 0.9})
        s = out["retrieval_summary"]
        self.assertEqual(len(rounds), 2); self.assertIn("widen: rerank: RuntimeError", s["degraded"])
        self.assertEqual((out["kbs"], s["routing"]["widened"], s["rerank"]), (["kb_001"], False, "below_threshold"))
        self.assertEqual((s["evidence_state"], s["no_relevant_content"]), ("diagnostic", True))   # unranked candidates do not displace the "everything below the floor" verdict
        self.assertNotIn("p9", [r["point_id"] for r in out["sources"]])

    def test_backend_failures_in_backfill_and_context_only_degrade(self) -> None:
        out, _ = self._run(kbs=["kb_001"], broken=("fetch", "neighbors"))
        s = out["retrieval_summary"]
        self.assertEqual([d for d in s["degraded"] if "backfill" in d or "context" in d], ["kb_001:backfill: RuntimeError", "context: RuntimeError"])
        hits = [r["point_id"] for r in out["sources"] if r["role"] == "hit"]
        self.assertEqual(hits, ["p1", "p3"]); self.assertTrue(s["low_confidence"])                # p2 carries only an id and its text cannot be fetched; the rest as usual

    def test_subject_buckets_guarantee_each_subject_a_slot(self) -> None:
        ents = [{"title": "尿酸", "type": "biomarker", "score": 0.7, "hop": 0, "via": "lexical"}, {"title": "血糖", "type": "biomarker", "score": 0.6, "hop": 0, "via": "vector"},
                {"title": "肌酐", "type": "biomarker", "score": 0.2, "hop": 1, "via": "expand"}, {"title": "空腹血糖", "type": "biomarker", "score": 0.5, "hop": 0, "via": "vector"}]
        out, _ = self._run(kbs=["kb_001"], question="尿酸和血糖分别是多少", graph_entities=ents,
                           extra_vec=[{"point_id": "p4", "score": 0.2, "payload": None}], settings_over={"top_k": 3})
        s = out["retrieval_summary"]
        self.assertEqual(s["buckets"], {"mode": "subject", "keys": ["尿酸", "血糖"]})                      # entities from 2-hop expansion or absent from the question are not subjects
        hits = [r["point_id"] for r in out["sources"] if r["role"] == "hit"]
        self.assertIn("p4", hits); self.assertEqual(hits[0], "p1"); self.assertEqual(s["selection"]["quota_filled"], {"尿酸": 1, "血糖": 1})
        self.assertEqual(next(r for r in out["sources"] if r["point_id"] == "p4")["bucket"], "血糖")
        self.assertEqual(out["pages"][0]["title"], "尿酸")

    def test_doc_buckets_for_cross_document_questions(self) -> None:
        out, _ = self._run(kbs=["kb_001"], question="两份报告分别怎么说", settings_over={"top_k": 2, "quota_min_hits": 1})
        s = out["retrieval_summary"]
        self.assertEqual(s["buckets"]["mode"], "doc"); self.assertEqual(sorted(s["buckets"]["keys"]), ["dir/d1.pdf", "dir/d2.pdf"])
        hits = [r for r in out["sources"] if r["role"] == "hit"]
        self.assertEqual({h["doc_id"] for h in hits}, {"d1", "d2"})                                   # the small document was not crowded out by the big one

    def test_boilerplate_and_low_confidence_visual_are_downweighted_not_dropped(self) -> None:
        out, _ = self._run(kbs=["kb_001"], extra_bm25=[{"point_id": "p5", "score": 5.0, "payload": {"doc_id": "d1", "rel_path": "dir/d1.pdf"}}], settings_over={"top_k": 5})
        s = out["retrieval_summary"]
        p5 = next(r for r in out["sources"] if r["point_id"] == "p5")
        self.assertTrue(p5["boilerplate"]); self.assertEqual(p5["scores"]["score_final"], 0.05); self.assertEqual(s["downweighted"]["boilerplate"], 1)
        self.assertEqual([r["point_id"] for r in out["sources"] if r["role"] == "hit"][-1], "p5")

    def test_embedding_down_keeps_bm25_and_lexical_graph(self) -> None:
        out, calls = self._run_with_embed_error()
        s = out["retrieval_summary"]
        self.assertIn("embedding: RuntimeError", s["degraded"]); self.assertEqual(s["channels"]["kb_001"]["text"], {"skipped": "embedding unavailable"})
        self.assertEqual(calls["graph"], ["kb_001"]); self.assertTrue(calls["lexical_only"])            # the graph channel still runs, with lexical seeds only
        self.assertTrue([r for r in out["sources"] if r["role"] == "hit"])

    def _run_with_embed_error(self):
        calls = {"graph": [], "lexical_only": None}
        docs = {"p1": _payload("p1", "d1", 2, "尿酸 433 umol/L。" * 20), "p2": _payload("p2", "d1", 3, "甘油三酯 1.7 mmol/L。" * 20)}
        src = SimpleNamespace(kb_id="kb_001", collection="kb_001", source_root="health-samples", graph_language="Chinese", graph_entity_types=[], graph_profile={})
        settings = SimpleNamespace(sources={"kb_001": src}, opensearch_url="http://os", reranker_base_url="http://rr/v1", embedding_base_url="", embedding_api_key="",
                                   embedding_model_id="", embedding_dim=2, qdrant_url="", qdrant_api_key=None, state_db=":memory:", visual_embedding_enabled=False)
        ss = _settings(visual_enabled=False)

        def fake_graph(settings, source, question, *, limit, hops, vector=None, lexical_only=False, **kw):
            calls["graph"].append(source.kb_id); calls["lexical_only"] = lexical_only
            return {"chunks": [{"point_id": "p1", "score": 0.4, "entities": ["尿酸"], "relations": []}], "entities": [], "relations": [], "specs": [], "pages": [], "graph_version": "v"}

        with mock.patch.object(service, "runtime", return_value=(settings, ss, object())), \
                mock.patch.object(catalog_mod, "get_catalog", return_value=[{"kb_id": "kb_001", "has_graph": True}]), \
                mock.patch.object(channels, "embed_question", side_effect=RuntimeError("8101 down")), \
                mock.patch.object(channels, "bm25_channel", return_value=[{"point_id": "p2", "score": 9.0, "payload": {"doc_id": "d1"}}, {"point_id": "p1", "score": 7.0, "payload": {"doc_id": "d1"}}]), \
                mock.patch.object(channels, "graph_channel", side_effect=fake_graph), \
                mock.patch.object(channels, "point_meta", return_value={}), \
                mock.patch.object(channels, "table_head", return_value=None), \
                mock.patch.object(channels, "fetch_payloads", side_effect=lambda q, c, ids, **kw: {i: docs[i] for i in ids if i in docs}), \
                mock.patch.object(channels, "neighbor_payloads", return_value=[]), \
                mock.patch.object(channels, "shared_driver", return_value=object()), \
                mock.patch.object(graphwalk, "neighborhoods", return_value=[]), \
                mock.patch.object(graphwalk, "point_payloads", return_value={}), \
                mock.patch.object(Reranker, "score", lambda self, query, documents: [0.8] * len(documents)):
            out = service.search("尿酸多少", kbs=["kb_001"])
        return out, calls



    def test_backends_out_of_reach_are_an_error_not_an_empty_answer(self) -> None:
        """The vector and keyword channels failed on every knowledge base, or not one candidate's text came back:
        the answer must not be "no relevant content". With only one channel broken the search returns as usual and
        records the degradation."""
        plain = {"entries": [{"kb_id": "kb_001", "has_graph": False}, {"kb_id": "kb_002", "has_graph": False}],
                 "visual_fn": lambda *a, **kw: []}             # no graph channel and no visual hits: candidates can only come from the vector and keyword channels
        with self.assertRaises(service.RetrievalUnavailable):
            self._run(broken=("vec", "bm25"), **plain)
        with self.assertRaises(service.RetrievalUnavailable):
            self._run(broken=("vec", "fetch"), **plain)        # the keyword channel brings back ids only, and their text cannot be fetched
        out, _ = self._run(broken=("vec",))
        self.assertTrue(out["sources"])
        self.assertIn("kb_001:text", out["retrieval_summary"]["degraded"])

class GraphWalkTests(unittest.TestCase):
    """The parts of the graph lookups that need no backend: excerpts of the original text, entities named in the
    question, the subject neighbourhoods /search carries."""

    ENTITIES = [
        {"id": "e1", "title": "ZK200", "type": "product", "aliases": ["ZK 200 Pro"], "degree": 50},
        {"id": "e2", "title": "Northwind Gateway", "type": "product", "aliases": ["Gateway"], "degree": 30},
        {"id": "e3", "title": "Northwind", "type": "vendor", "aliases": [], "degree": 99},
        {"id": "e4", "title": "网关", "type": "module", "aliases": [], "degree": 7},
        {"id": "e5", "title": "网关", "type": "module", "aliases": [], "degree": 70},
        {"id": "e6", "title": "AI", "type": "capability", "aliases": [], "degree": 5},
        {"id": "e7", "title": "延迟", "type": "property", "aliases": ["时延"], "degree": 12},
        {"id": "e8", "title": "热备份", "type": "feature", "aliases": None, "degree": 3},
        {"id": "e9", "title": "Gateway", "type": "module", "aliases": [], "degree": 2},
    ]
    TOP = [{"relation_id": "r1", "type": "part_of", "outgoing": False, "directed": True, "weight": 5.0,
            "other_id": "e10", "other_title": "控制模块", "other_type": "module", "other_degree": 4},
           {"relation_id": "r3", "type": "requires", "outgoing": True, "directed": True, "weight": 4.0,
            "other_id": "e10", "other_title": "控制模块", "other_type": "module", "other_degree": 4},
           {"relation_id": "r2", "type": "supports", "outgoing": True, "directed": False, "weight": 2.5,
            "other_id": "e5", "other_title": "网关", "other_type": "module", "other_degree": 70}]
    RELATIONS = {"e1": {"total": 3, "kinds": ["part_of", "supports", "part_of"], "top": TOP},
                 "e4": {"total": 1, "kinds": ["part_of"], "top": TOP[:1]}}
    FACTS = {"e1": 12}

    def setUp(self) -> None:
        graphwalk._names.clear()

    def _driver(self, log):
        entities, relations, facts = self.ENTITIES, self.RELATIONS, self.FACTS

        class Result:
            def __init__(self, rows): self.rows = rows
            def data(self): return self.rows

        class Session:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def run(self, cypher, **kw):
                log.append((cypher, kw))
                if "e.aliases AS aliases" in cypher:
                    return Result(entities)
                if "collect(r.type) AS kinds" in cypher:
                    return Result([{"eid": i, **relations[i]} for i in kw["ids"] if i in relations])
                if "HAS_SPEC" in cypher:
                    return Result([{"eid": i, "facts": facts.get(i, 0)} for i in kw["ids"]])
                raise AssertionError(cypher)

        class Driver:
            def session(self): return Session()
            def close(self): raise AssertionError("the shared driver must not be closed")

        return Driver()

    def test_facts_point_at_the_chunk_that_holds_the_value(self) -> None:
        """A fact comes from an extraction unit that may span several chunks: when the value's wording appears in exactly
        one of them that chunk goes first and is marked located, the property name is tried when the value does not
        settle it, and nothing is marked otherwise; inactive and missing chunks go last and are never located."""
        sources = [{"n": 1, "point_id": "p1", "doc_id": "d1", "rel_path": "dir/a.pdf", "doc": "a.pdf", "chunk_index": 4, "content_version": "v1",
                    "page_idx": 2, "position": "a.pdf · page 3", "place": "page 3", "text": "ZK200 概述:支持双机热备,切换很快。"}]
        stored = {"p2": {"is_active": True, "doc_id": "d1", "rel_path": "dir/a.pdf", "filename": "a.pdf", "chunk_index": 5, "content_version": "v1",
                         "page_idx": 3, "text": "切换时间 2 s,待机功耗 3 W。ZK200 概述见上页。"},
                  "p3": {"is_active": False, "doc_id": "d1", "rel_path": "dir/a.pdf", "filename": "a.pdf", "chunk_index": 6, "content_version": "v0",
                         "page_idx": 4, "text": "切换时间 2 s"},
                  "p5": {"is_active": True, "doc_id": "d7", "rel_path": "dir/b.pdf", "filename": "b.pdf", "chunk_index": 0, "content_version": "v1",
                         "page_idx": 0, "text": "重量 5 kg。"}}
        log = []

        class Q:
            def __init__(self, down=()): self.down = down
            def retrieve(self, collection_name, ids, with_payload, with_vectors, **kw):
                log.append((collection_name, sorted(ids), "text" in with_payload, kw))
                if collection_name in self.down:
                    raise RuntimeError("qdrant timed out")
                return [SimpleNamespace(id=i, payload=stored[i]) for i in ids if i in stored]

        specs = [{"kb_id": "kb_001", "id": "a", "property": "切换时间", "value": "2 s", "point_ids": ["p1", "p2"]},        # the value is in p2 only
                 {"kb_id": "kb_001", "id": "b", "property": "待机功耗", "value": "三瓦", "point_ids": ["p1", "p2"]},       # a paraphrased value: the property name is in p2 only
                 {"kb_id": "kb_001", "id": "c", "property": "名称", "value": "ZK200", "point_ids": ["p1", "p2"]},          # the value in both, the property name in neither: undecided
                 {"kb_id": "kb_001", "id": "d", "property": "切换时间", "value": "2 s", "point_ids": ["p3", "p9", "p2", "p1"]},   # inactive and missing ones go last; p1, beyond the per-fact limit, is not taken
                 {"kb_id": "kb_001", "id": "e", "property": "概述", "value": "热备", "point_ids": ["p1"]},
                 {"kb_id": "kb_002", "id": "f", "property": "重量", "value": "5", "point_ids": ["p5", "p6"]},             # another knowledge base reads its own collection
                 {"kb_id": "kb_001", "id": "g", "property": "x", "value": "1", "point_ids": []}]
        self.assertEqual(graphwalk.locate_facts(Q(), {"kb_001": "c1", "kb_002": "c2"}, specs, sources, timeout=2.4), [])
        by = {sp["id"]: sp for sp in specs}
        shape = lambda sid: [(e["point_id"], e["active"], e.get("located")) for e in by[sid]["evidence"]]
        self.assertEqual(shape("a"), [("p2", True, True), ("p1", True, None)])
        self.assertEqual((by["a"]["evidence"][0]["doc_id"], by["a"]["evidence"][0]["chunk_index"], by["a"]["evidence"][0]["content_version"]), ("d1", 5, "v1"))
        self.assertTrue(by["a"]["evidence"][0]["place"]); self.assertEqual(by["a"]["evidence"][1]["place"], "page 3")
        self.assertEqual(shape("b"), [("p2", True, True), ("p1", True, None)])
        self.assertEqual(shape("c"), [("p1", True, None), ("p2", True, None)])
        self.assertEqual(shape("d"), [("p2", True, None), ("p3", False, None), ("p9", False, None)])        # one live chunk left: nothing to tell apart, but it goes first
        self.assertEqual(shape("e"), [("p1", True, None)])
        self.assertEqual(shape("f"), [("p5", True, None), ("p6", False, None)])
        self.assertNotIn("evidence", by["g"])
        self.assertEqual(log, [("c1", ["p2", "p3", "p9"], True, {"timeout": 3}), ("c2", ["p5", "p6"], True, {"timeout": 3})])   # p1, already among the sources, is not fetched; the timeout is rounded up to whole seconds
        # one knowledge base cannot be read: its chunks keep only their ids (unknown, not marked inactive), the others are unaffected
        again = [dict(sp, evidence=None) for sp in specs if sp["id"] in ("a", "f")]
        self.assertEqual(graphwalk.locate_facts(Q(down=("c2",)), {"kb_001": "c1", "kb_002": "c2"}, again, sources), ["kb_002:spec_evidence: RuntimeError"])
        self.assertEqual([e.get("located") for e in again[0]["evidence"]], [True, None])
        self.assertEqual(again[1]["evidence"], [{"point_id": "p5"}, {"point_id": "p6"}])

    def test_excerpt_takes_the_sentences_that_name_the_other_end(self) -> None:
        sentence = "ZK200 通过 Northwind 网关接入,延迟低于 5 ms。"
        text = "前言。" * 40 + sentence + "后记。" * 100
        body, match = graphwalk.excerpt_of(text, ["Northwind 网关", "NW-GW"], ["ZK200"])
        self.assertEqual(match, "both")
        self.assertTrue(body.startswith(sentence)); self.assertTrue(body.endswith("后记。")); self.assertNotIn("…", body)   # from the start of a sentence to the end of one
        self.assertLessEqual(len(body), graphwalk.EXCERPT_CHARS)
        self.assertEqual(graphwalk.excerpt_of("Northwind 网关支持双机热备。", ["Northwind 网关"], ["ZK200"]), ("Northwind 网关支持双机热备。", "other"))
        self.assertEqual(graphwalk.excerpt_of("ZK200 的外壳\n是铝合金。", ["Northwind 网关"], ["ZK200"]), ("ZK200 的外壳 是铝合金。", "center"))   # a line break becomes a space
        # no sentence boundary on either side: the cut sides get an ellipsis
        body, match = graphwalk.excerpt_of("甲" * 300 + "Northwind 网关" + "乙" * 300, ["Northwind 网关"], [])
        self.assertEqual(match, "other"); self.assertTrue(body.startswith("…甲") and body.endswith("乙…")); self.assertIn("Northwind 网关", body)
        self.assertEqual(graphwalk.excerpt_of("丙" * 500, ["Northwind 网关"], ["ZK200"]), ("丙" * graphwalk.EXCERPT_CHARS + "…", "none"))   # neither found: the head of the chunk
        # a name of letters / digits does not match inside a longer word; for a name written with spaces the spelling without them counts
        self.assertEqual(graphwalk.excerpt_of("FAIL 指示灯亮起表示故障。AI 模块负责推理。", ["AI"], []), ("AI 模块负责推理。", "other"))
        self.assertEqual(graphwalk.excerpt_of("ZK200 支持热插拔。", ["ZK 200"], [])[1], "other")
        # with both ends present, the occurrence of the far end closest to the centre entity is taken
        far = "网关的说明。" + "无关。" * 60 + "ZK200 内置网关,支持冗余。"
        self.assertEqual(graphwalk.excerpt_of(far, ["网关"], ["ZK200"]), ("ZK200 内置网关,支持冗余。", "both"))

    def test_excerpt_prefers_document_text_over_picture_descriptions(self) -> None:
        payloads = {"a": {"is_active": True, "rel_path": "x.pdf", "text": "FACTS: ZK200 与 Northwind 网关的连接示意图。", "visual_ref": "parse/a.jpg", "block_type": "image"},
                    "b": {"is_active": True, "rel_path": "x.pdf", "text": "Northwind 网关支持双机热备。", "block_type": "text"},
                    "c": {"is_active": True, "rel_path": "x.pdf", "text": "ZK200 的外壳是铝合金。", "block_type": "text"},
                    "d": {"is_active": False, "rel_path": "old.pdf", "text": "ZK200 通过 Northwind 网关接入。", "block_type": "text"}}

        class Q:
            def retrieve(self, collection_name, ids, with_payload, with_vectors):
                return [SimpleNamespace(id=i, payload=payloads[i]) for i in ids if i in payloads]

        def pick(*points):
            rows = [{"relation_id": "r1", "_names": ["Northwind 网关"], "evidence": [{"point_id": p} for p in points]}]
            graphwalk.backfill_evidence(Q(), "kb_005", rows, anchors=["ZK200"])
            first = rows[0]["evidence"][0]
            return first["point_id"], first.get("excerpt_match"), first.get("visual", False)

        # a text chunk that names the far end (even without the centre entity) comes before a picture description that names both;
        # a picture chunk is used, and marked visual, when it is the only one naming the far end
        self.assertEqual(pick("a", "c", "b"), ("b", "other", False))
        self.assertEqual(pick("c", "a"), ("a", "both", True))
        self.assertEqual(pick("c"), ("c", "center", False))
        self.assertEqual(pick("d", "c"), ("c", "center", False))                    # deactivated evidence is not excerpted
        rows = [{"relation_id": "r1", "_names": ["Northwind 网关"], "evidence": [{"point_id": "b"}]}]
        graphwalk.backfill_evidence(Q(), "kb_005", rows)                           # no excerpt wanted (the facts route): the text is not fetched
        self.assertNotIn("excerpt", rows[0]["evidence"][0])

    def test_entities_named_in_the_question(self) -> None:
        log: list = []
        with self._driver(log).session() as session:
            index = graphwalk.name_index(session, "kb_005", "v1")
        named = graphwalk.named_entities("zk200 和 northwind gateway 的网关延迟对比,FAIL 时怎么热备份", index)
        # case and spaces are ignored; names covered by a longer one (Northwind, Gateway) do not count; AI does not match inside FAIL;
        # of same-named entities the one with more relations is taken (e5); names with letters or digits, or of four characters and
        # more, come first, three-character ones next, two-character words without letters last
        self.assertEqual([(e["id"], e["_rank"]) for e in named], [("e1", 2), ("e2", 2), ("e8", 1), ("e5", 0), ("e7", 0)])
        # aliases count; of several sharing the name the one with the most relations is taken (e2 shares it as an alias, e9 as its
        # title), the same entity a lookup by name on the neighbours endpoint lands on
        self.assertEqual([e["id"] for e in graphwalk.named_entities("Gateway 怎么配", index)], ["e2"])
        self.assertEqual([e["id"] for e in graphwalk.named_entities("ZK 200 的时延", index)], ["e1", "e7"])           # an extra space in the question, and an alias
        self.assertEqual(graphwalk.named_entities("今天天气怎么样", index), [])
        self.assertEqual(graphwalk.named_entities("", index), [])

    def test_neighbourhoods_take_named_entities_first_then_seeds(self) -> None:
        log: list = []
        sources = {"kb_005": SimpleNamespace(kb_id="kb_005", collection="kb_005")}
        seed = lambda eid, title, score: {"kb_id": "kb_005", "id": eid, "title": title, "score": score}
        seeds = [seed("e3", "Northwind", 0.9), seed("e1", "ZK200", 0.8), seed("zz", "not in the graph", 0.7)]
        run = lambda question, seeds, gv="v1", limit=3: graphwalk.neighborhoods(SimpleNamespace(), sources, {"kb_005": gv}, question, seeds, limit=limit,
                                                                                driver=self._driver(log))
        out = run("zk200 和 northwind gateway 对比", seeds)
        # the two the question names come first (Northwind, the best-scoring graph route seed, does not get ahead of them); a seed
        # used to fill up that is connected to nothing takes no slot
        self.assertEqual([(b["kb_id"], b["id"], b["named"], b["relations"], b["facts"]) for b in out],
                         [("kb_005", "e1", True, 3, 12), ("kb_005", "e2", True, 0, 0)])
        self.assertEqual(out[0]["predicates"], [{"type": "part_of", "count": 2}, {"type": "supports", "count": 1}])
        # a far end is listed once, by its strongest relation (the control module also has a requires), so one far end cannot fill the list
        self.assertEqual([(n["type"], n["direction"], n["directed"], n["other"]["title"], n["other"]["degree"]) for n in out[0]["neighbors"]],
                         [("part_of", "in", True, "控制模块", 4), ("supports", "out", False, "网关", 70)])
        self.assertNotIn("evidence", out[0]["neighbors"][0])                                             # leads only; evidence and excerpts come from the neighbours endpoint
        rel = next(kw for c, kw in log if "collect(r.type) AS kinds" in c)
        self.assertEqual((rel["ids"], rel["gv"]), (["e1", "e2", "e3"], "v1"))
        # a two- or three-character name without letters or digits only counts when the entity is also a graph route seed:
        # 网关 is among the seeds (of the same-named ones e5 has more relations), 延迟 is not
        self.assertEqual([(b["id"], b["named"]) for b in run("网关的延迟", [seed("e5", "网关", 0.6)])], [("e5", True)])
        self.assertEqual(run("网关的延迟", []), [])
        # seeds that fill up: of those sharing a name within one knowledge base only the first is taken
        self.assertEqual([(b["id"], b["named"], b["relations"]) for b in run("今天天气怎么样", [seed("e4", "网关", 0.6), seed("e5", "网关", 0.5)])],
                         [("e4", False, 1)])
        self.assertEqual([b["id"] for b in run("zk200 和 northwind gateway 的热备份", seeds, limit=1)], ["e1"])   # the total across knowledge bases is capped
        # the names stay in the process: the same graph version is not read again, a new one is
        loads = lambda: sum(1 for c, _ in log if "e.aliases AS aliases" in c)
        self.assertEqual(loads(), 1)
        run("热备份", [], gv="v2")
        self.assertEqual(loads(), 2)


class ApiTests(unittest.TestCase):
    def test_bearer_token_is_required_when_configured(self) -> None:
        from fastapi.testclient import TestClient

        from kb_search.main import create_app

        ss = _settings(token="secret")
        with mock.patch.object(service, "runtime", return_value=(SimpleNamespace(sources={}), ss, SimpleNamespace(get_collections=lambda: None))), \
                mock.patch.object(service, "search", return_value={"ok": 1}) as search_mock, \
                mock.patch.object(service, "image", return_value={"bytes": b"PNG", "mime": "image/png", "source": "pdf-embedded", "width": 3, "height": 2}), \
                mock.patch.object(service, "crop", return_value={"bytes": b"CROP", "mime": "image/png", "source": "cache", "width": 1, "height": 1, "box": [0, 0, 1, 1]}):
            client = TestClient(create_app())
            auth = {"Authorization": "Bearer secret"}
            self.assertEqual(client.get("/health").status_code, 200)                                  # the health check needs no auth
            self.assertEqual(client.post("/search", json={"question": "q"}).status_code, 401)
            self.assertEqual(client.post("/search", json={"question": "q"}, headers={"Authorization": "Bearer wrong"}).status_code, 401)
            self.assertEqual(client.post("/search", json={"question": "q"}, headers=auth).json(), {"ok": 1})
            self.assertEqual(client.post("/search", json={"question": "q"}, headers={"Authorization": "bearer secret"}).status_code, 200)   # the prefix is case-insensitive
            for wrong in ("Bearer secre", "Bearer secretx", "Bearer", "secret", "Basic secret"):
                self.assertEqual(client.post("/search", json={"question": "q"}, headers={"Authorization": wrong}).status_code, 401)
            self.assertEqual(client.post("/search", json={"question": ""}, headers=auth).status_code, 422)
            from PIL import Image
            b64 = lambda data: __import__("base64").b64encode(data).decode()
            buf = io.BytesIO(); Image.new("RGB", (4, 4), (255, 0, 0)).save(buf, format="PNG")
            client.post("/search", json={"question": "q", "image_b64": "data:image/png;base64," + b64(buf.getvalue())}, headers=auth)
            self.assertEqual(search_mock.call_args.kwargs["image_bytes"], buf.getvalue())
            calls_before = search_mock.call_count
            bad = client.post("/search", json={"question": "q", "image_b64": b64(b"not an image")}, headers=auth)
            self.assertEqual((bad.status_code, search_mock.call_count), (422, calls_before))            # an unreadable image is rejected outright instead of silently searching by text
            self.assertIn("not an image", bad.json()["detail"])
            r = client.get("/image/kb_001/p1", headers=auth)
            self.assertEqual((r.status_code, r.headers["content-type"], r.headers["x-image-source"], r.content), (200, "image/png", "pdf-embedded", b"PNG"))
            self.assertEqual(client.get("/image/kb_001/p1").status_code, 401)
            r2 = client.post("/crop", json={"kb_id": "kb_001", "point_id": "p1", "bbox": [0.1, 0.1, 0.5, 0.5]}, headers=auth)
            self.assertEqual((r2.status_code, r2.headers["x-crop-box"], r2.content), (200, "0,0,1,1", b"CROP"))
            self.assertEqual(client.post("/crop", json={"kb_id": "kb_001", "point_id": "p1", "bbox": [0.1, 0.1]}, headers=auth).status_code, 422)

    def test_malformed_point_id_is_a_client_error(self) -> None:
        from fastapi.testclient import TestClient

        from kb_search.main import create_app

        settings = SimpleNamespace(sources={"kb_001": SimpleNamespace(collection="kb_001")})
        q = SimpleNamespace(retrieve=mock.Mock(side_effect=AssertionError("must not ask the main store")))
        with mock.patch.object(service, "runtime", return_value=(settings, _settings(token="secret"), q)):
            client = TestClient(create_app(), raise_server_exceptions=False)
            auth = {"Authorization": "Bearer secret"}
            r = client.get("/image/kb_001/kb_001:12:abc:pdf-v12:blk:0", headers=auth)
            self.assertEqual(r.status_code, 422); self.assertIn("point_id must be a UUID", r.json()["detail"])      # this used to be a 500 with a stack trace
            r2 = client.post("/crop", json={"kb_id": "kb_001", "point_id": "not-a-uuid", "bbox": [0.1, 0.1, 0.5, 0.5]}, headers=auth)
            self.assertEqual(r2.status_code, 422)

    def test_every_search_leaves_one_log_line_without_the_question(self) -> None:
        import contextlib

        from fastapi.testclient import TestClient

        from kb_search.main import create_app

        summary = {"kbs": ["kb_001", "kb_002"], "routing": {"widened": True}, "rerank": "below_threshold", "evidence_state": "diagnostic",
                   "degraded": ["kb_002:graph", "budget: context skipped"], "image_query": False, "sources": {"hits": 3},
                   "timings_ms": {"embed_ms": 12, "recall_ms": 800, "total_ms": 1900}}
        class Out:
            def __init__(self): self.writes = []
            def write(self, text): self.writes.append(text)
            def flush(self): pass

        class Broken:
            def write(self, text): raise OSError("stdout closed")
            def flush(self): pass

        out = Out()
        with mock.patch.object(service, "runtime", return_value=(SimpleNamespace(sources={}), _settings(token="secret"), None)), \
                mock.patch.object(service, "search", return_value={"question": "我的尿酸多少", "retrieval_summary": summary}):
            client = TestClient(create_app())
            with contextlib.redirect_stdout(out):
                r = client.post("/search", json={"question": "我的尿酸多少"}, headers={"Authorization": "Bearer secret"})
            with contextlib.redirect_stdout(Broken()):
                r2 = client.post("/search", json={"question": "我的尿酸多少"}, headers={"Authorization": "Bearer secret"})
        self.assertEqual((r.status_code, r2.status_code), (200, 200))                               # failing to write the log does not affect the response
        lines = [w for w in out.writes if w.startswith("[search.request] ")]
        self.assertEqual(len(lines), 1); self.assertTrue(lines[0].endswith("}\n"))                  # the whole line is written at once, so concurrent lines do not run together
        self.assertNotIn("尿酸", lines[0])
        self.assertEqual(json.loads(lines[0][len("[search.request] "):]),
                         {"timings_ms": summary["timings_ms"], "kbs": 2, "widened": True, "rerank": "below_threshold", "evidence_state": "diagnostic",
                          "hits": 3, "image_query": False, "degraded": ["kb_002:graph", "budget: context skipped"]})

    def test_graph_neighbors_route_and_shaping(self) -> None:
        from fastapi.testclient import TestClient

        from kb_search import graphwalk
        from kb_search.main import create_app

        class FakeResult:
            def __init__(self, rows): self.rows = rows
            def data(self): return self.rows

        class FakeSession:
            def __init__(self): self.calls = []
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def run(self, cypher, **kw):
                self.calls.append((cypher, kw))
                if "count(r) AS n" in cypher:
                    return FakeResult([{"type": "has_stage", "n": 5}, {"type": "part_of", "n": 2}])
                if "RELATED_TO" in cypher:
                    return FakeResult([{"relation_id": "r1", "type": "has_stage", "outgoing": True, "directed": True, "weight": 3.0, "npmi": 0.4, "cooccur": 2,
                                        "description": "FDE 有阶段二", "type_violation": False, "other_id": "e2", "other_title": "阶段二", "other_type": "stage", "other_scope": None, "other_pagerank": 0.1,
                                        "other_degree": 3, "other_aliases": ["Stage Two"]}])
                if "EVIDENCES" in cypher:
                    return FakeResult([{"rid": "r1", "point_id": p, "chunk_uid": "u" + p[1:], "rel_path": "x/fde.pdf", "doc_id": None, "chunk_index": None, "content_version": None, "page_idx": None}
                                       for p in ("p6", "p7")])
                if "MENTIONED_IN" in cypher:
                    return FakeResult([{"rel_path": "x/fde.pdf"}])
                if "HAS_SPEC" in cypher:
                    return FakeResult([{"n": 4}])
                if "id: $id" in cypher:
                    return FakeResult([])
                return FakeResult([{"id": "e1", "title": "FDE", "type": "role", "parent_type": None, "scope": None, "description": "前向部署工程师", "pagerank": 0.5, "degree": 9, "aliases": ["Forward Deployed Engineer"]}])

        class FakeDriver:
            def __init__(self): self.s = FakeSession()
            def session(self): return self.s
            def close(self): pass

        src = SimpleNamespace(kb_id="kb_003", collection="kb_003", source_root="reports")
        class FakeQ:
            def retrieve(self, collection_name, ids, with_payload, with_vectors):
                self.keys = list(with_payload)
                base = {"is_active": True, "doc_id": "d7", "content_version": "v1", "page_idx": 4, "rel_path": "x/fde.pdf", "filename": "fde.pdf", "section_path": ["2. Roles"],
                        "block_type": "text"}
                return [SimpleNamespace(id="p6", payload={**base, "chunk_index": 2, "text": "FDE 是前向部署工程师。"}),
                        SimpleNamespace(id="p7", payload={**base, "chunk_index": 3, "text": "TITLE: 2. Roles\nFDE 的工作分三个阶段。阶段二由 FDE 驻场交付,周期约三个月。阶段三转入运维。"})]

        with mock.patch("kb_pipeline.graph.neo4j_import.neo4j_driver", return_value=FakeDriver()), \
                mock.patch("kb_pipeline.graph.neo4j_import.active_neo4j_graph_version", return_value="003-v"):
            fq = FakeQ()
            out = graphwalk.neighbors(SimpleNamespace(), src, entity="fde", entity_id=None, limit=5, types=None, direction="both", q=fq)
            only = graphwalk.neighbors(SimpleNamespace(), src, entity="fde", entity_id=None, limit=5, types=["part_of"], direction="both", q=fq)
        self.assertTrue(out["found"]); self.assertEqual(out["entity"]["title"], "FDE"); self.assertEqual(out["entity"]["docs"], ["x/fde.pdf"])
        nb = out["neighbors"][0]
        self.assertEqual((nb["type"], nb["direction"], nb["other"]["title"]), ("has_stage", "out", "阶段二"))            # predicate, direction, far end
        # every relation carries an excerpt of the original text: the evidence chunk that names the far end is picked (p7, moved to
        # the front) and cut from the start of a sentence; the chunk that does not name it (p6) carries none
        self.assertEqual([ev["point_id"] for ev in nb["evidence"]], ["p7", "p6"])
        self.assertEqual((nb["evidence"][0]["excerpt"], nb["evidence"][0]["excerpt_match"], nb["evidence"][0]["block_type"]),
                         ("阶段二由 FDE 驻场交付,周期约三个月。阶段三转入运维。", "both", "text"))
        self.assertNotIn("excerpt", nb["evidence"][1]); self.assertNotIn("visual", nb["evidence"][0]); self.assertNotIn("_names", nb)
        self.assertIn("text", fq.keys)                                                                    # the text is fetched for the excerpt only
        # relation counts per predicate are not bounded by limit; the far end's own relation count and the number of facts under the
        # centre entity come along, and the caller decides by them where to walk
        self.assertEqual(out["predicates"], [{"type": "has_stage", "count": 5}, {"type": "part_of", "count": 2}])
        self.assertEqual((nb["other"]["degree"], out["entity"]["facts"]), (3, 4))
        self.assertEqual((only["total"], only["has_more"]), (2, True))                                    # with a predicate filter the total counts that kind only
        self.assertEqual((nb["evidence"][0]["doc_id"], nb["evidence"][0]["chunk_index"], nb["evidence"][0]["active"]), ("d7", 3, True))   # the evidence chunk can be fed to /context directly
        self.assertEqual((nb["evidence"][0]["position"], nb["evidence"][0]["place"]), ("page 4 · 2. Roles", "page 4"))   # the position string and the short locator for citations, written like the sources of /search
        self.assertEqual((out["count"], out["total"], out["has_more"], out["kb_name"]), (1, 7, True, "reports"))   # the total is not bounded by limit: an incomplete answer shows
        ss = _settings(token="secret")
        with mock.patch.object(service, "runtime", return_value=(SimpleNamespace(sources={}), ss, SimpleNamespace(get_collections=lambda: None))), \
                mock.patch.object(service, "graph_neighbors", return_value={"found": True}) as gn:
            client = TestClient(create_app())
            r = client.post("/graph/neighbors", json={"kb_id": "kb_003", "entity": "FDE", "types": ["has_stage"], "direction": "out"}, headers={"Authorization": "Bearer secret"})
            self.assertEqual((r.status_code, r.json()), (200, {"found": True})); self.assertEqual(gn.call_args.kwargs["types"], ["has_stage"])
            self.assertEqual(client.post("/graph/neighbors", json={"kb_id": "kb_003", "entity": "FDE", "direction": "sideways"}, headers={"Authorization": "Bearer secret"}).status_code, 422)

    def test_graph_neighbors_reuses_the_shared_client_and_driver(self) -> None:
        from kb_search import graphwalk

        driver = _GraphDriver()
        q = _GraphQdrant()
        src = SimpleNamespace(kb_id="kb_001", collection="kb_001")
        with mock.patch("kb_pipeline.graph.neo4j_import.neo4j_driver", side_effect=AssertionError("must not create a Neo4j driver")), \
                mock.patch("kb_pipeline.vector.qdrant.client", side_effect=AssertionError("must not create a Qdrant client")), \
                mock.patch.object(channels, "embed_question", return_value=[1.0, 0.0]):
            out = graphwalk.neighbors(SimpleNamespace(), src, entity="没有这个实体", entity_id=None, q=q, driver=driver)
        self.assertFalse(out["found"]); self.assertEqual([c["id"] for c in out["candidates"]], ["e1"])     # no name match: the vector candidates go through the shared client
        self.assertEqual(driver.closed, 0); self.assertEqual(q.calls, ["query:graph_001_entity"])
        by_name = next(text for text, _, _ in driver.log if "toLower(e.title)" in text)
        self.assertIn("e.id IS NOT NULL", by_name)                                                          # the lookup by name can use the composite index too

    def test_graph_entities_lists_by_type_with_total_and_paging(self) -> None:
        """ "All of them" questions: entities by type / upper class / name, with a total and paging; the first page
        also carries the number of entities per type."""
        from fastapi.testclient import TestClient

        from kb_search import graphwalk
        from kb_search.main import create_app

        log: list = []

        class Session:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def run(self, cypher, **kw):
                text = str(getattr(cypher, "text", cypher))
                log.append((text, getattr(cypher, "timeout", None), kw))
                if "count(e) AS total" in text:
                    rows = [{"total": 3}]
                elif "AS etype" in text:
                    rows = [{"etype": "product", "parent": "entity", "n": 3}, {"etype": "module", "parent": "part", "n": 9}]
                elif "MENTIONED_IN" in text:
                    rows = [{"eid": "e1", "docs": ["a/whitepaper.pdf", "b/manual.pdf", "c/comparison.xlsx"], "doc_count": 7}]
                else:
                    rows = [{"id": "e1", "title": "Northwind Suite", "type": "product", "parent_type": "entity", "scope": None, "description": "An office suite. " * 40,
                             "pagerank": 0.9, "degree": 40, "aliases": ["NW Suite"]},
                            {"id": "e2", "title": "Northwind Notes", "type": "product", "parent_type": "entity", "scope": None, "description": None,
                             "pagerank": 0.5, "degree": 12, "aliases": []}]
                return SimpleNamespace(data=lambda: rows)

        driver = SimpleNamespace(session=lambda: Session(), close=mock.Mock())
        src = SimpleNamespace(kb_id="kb_005", collection="kb_005", source_root="products")
        with mock.patch("kb_pipeline.graph.neo4j_import.neo4j_driver", side_effect=AssertionError("must not create a Neo4j driver")), \
                mock.patch("kb_pipeline.graph.neo4j_import.active_neo4j_graph_version", return_value="005-v"):
            out = graphwalk.list_entities(SimpleNamespace(), src, types=["Product", " product "], parent_types=["Entity"], name=" Northwind ",
                                          limit=2, offset=0, driver=driver, timeout=7.0)
            page2 = graphwalk.list_entities(SimpleNamespace(), src, types=["product"], limit=999, offset=2, driver=driver)
        self.assertEqual((out["kb_name"], out["graph_version"], out["total"], out["count"], out["has_more"]), ("products", "005-v", 3, 2, True))
        self.assertEqual(out["filters"], {"types": ["product"], "parent_types": ["entity"], "name": "northwind"})    # case-insensitive, de-duplicated
        self.assertEqual([e["id"] for e in out["entities"]], ["e1", "e2"])
        self.assertEqual(len(out["entities"][0]["description"]), 200); self.assertIsNone(out["entities"][1]["description"])   # a listing keeps only the start of a description
        self.assertEqual((out["entities"][0]["docs"], out["entities"][0]["doc_count"]), (["a/whitepaper.pdf", "b/manual.pdf", "c/comparison.xlsx"], 7))   # items carry their documents: a listing needs no query per item
        self.assertEqual((out["entities"][1]["docs"], out["entities"][1]["doc_count"]), ([], 0))
        docs_query = next((t, kw) for t, _, kw in log if "MENTIONED_IN" in t)
        self.assertEqual((docs_query[1]["ids"], docs_query[1]["top"]), (["e1", "e2"], 3)); self.assertIn("ORDER BY mentions DESC", docs_query[0])
        self.assertEqual(out["types"], [{"type": "product", "parent_type": "entity", "count": 3}, {"type": "module", "parent_type": "part", "count": 9}])
        self.assertNotIn("types", page2); self.assertEqual((page2["limit"], page2["offset"]), (200, 2))             # the type summary only on the first page; limit is capped
        listing = next(t for t, _, _ in log if "SKIP $offset LIMIT $limit" in t)
        for piece in ("e.id IS NOT NULL", "coalesce(e.boilerplate, false) = false", "coalesce(e.reference, false) = false",
                      "toLower(e.type) IN $types", "ORDER BY coalesce(e.pagerank, 0) DESC, e.id"):
            self.assertIn(piece, listing)                                                                           # uses the composite index, skips boilerplate and references, stable order
        self.assertTrue(all(timeout == 7.0 for _, timeout, kw in log[:4]))                                          # every query carries the transaction timeout
        self.assertEqual(driver.close.call_count, 0)                                                                # the shared driver is not closed
        ss = _settings(token="secret")
        auth = {"Authorization": "Bearer secret"}
        with mock.patch.object(service, "runtime", return_value=(SimpleNamespace(sources={}), ss, None)), \
                mock.patch.object(service, "graph_entities", return_value={"total": 0}) as ge:
            client = TestClient(create_app())
            r = client.post("/graph/entities", json={"kb_id": "kb_005", "types": ["product"], "limit": 20, "offset": 40}, headers=auth)
            self.assertEqual((r.status_code, r.json()), (200, {"total": 0}))
            self.assertEqual((ge.call_args.kwargs["types"], ge.call_args.kwargs["limit"], ge.call_args.kwargs["offset"]), (["product"], 20, 40))
            self.assertEqual(client.post("/graph/entities", json={"kb_id": "kb_005", "limit": 201}, headers=auth).status_code, 422)
            self.assertEqual(client.post("/graph/entities", json={"kb_id": "kb_005"}).status_code, 401)
        with mock.patch.object(service, "runtime", return_value=(SimpleNamespace(sources={}), ss, None)):
            self.assertEqual(TestClient(create_app()).post("/graph/entities", json={"kb_id": "kb_009"}, headers=auth).status_code, 404)

    def test_graph_facts_by_subject_and_property(self) -> None:
        """Facts by subject / property: the subject goes along HAS_SPEC, the property is matched exactly first and by
        containment second, bringing every spelling of the same concept along; rows come from the payloads of the fact
        collection, the same projection as the specs of /search, with evidence locators and source verification."""
        from fastapi.testclient import TestClient

        from kb_pipeline.graph.vectors import point_id_for
        from kb_search import graphwalk
        from kb_search.main import create_app

        log: list = []
        gv = "002-v"

        class Session:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def run(self, cypher, **kw):
                text = str(getattr(cypher, "text", cypher))
                log.append((text, kw))
                rows: list = []
                if "replace(toLower(e.title), ' ', '') = $name" in text:
                    rows = [{"id": "e-small", "title": "ZK200", "type": "device", "parent_type": "entity", "scope": None, "description": "", "pagerank": 0.1, "degree": 2, "aliases": []},
                            {"id": "e-main", "title": "ZK200", "type": "device", "parent_type": "entity", "scope": None, "description": "", "pagerank": 0.2, "degree": 30, "aliases": []}]
                elif "RETURN DISTINCT f.concept_key AS key" in text:
                    rows = [] if " = $p" in text else [{"key": "c-vcc"}]            # nothing exact, containment hits one concept
                elif "count(f) AS total" in text:
                    rows = [{"total": 5}]
                elif "AS row" in text:
                    rows = [{"row": {"id": "f1", "subject": "ZK200", "property": "Supply voltage", "concept": "supply voltage", "concept_key": "c-vcc", "value": "3.3", "unit": "V", "valid_from": "rev A"}},
                            {"row": {"id": "f2", "subject": "ZK200", "property": "VCC", "concept": "supply voltage", "concept_key": "c-vcc", "value": "3.6", "unit": "V", "valid_from": "rev B"}},
                            {"row": {"id": "f3", "subject": "ZK200", "property": "Supply voltage result", "concept": "supply voltage", "concept_key": "c-vcc", "value": "3.5", "unit": "V", "valid_from": "rev C"}}]
                elif "count(f) AS n" in text:
                    rows = [{"concept": "supply voltage", "concept_key": "c-vcc", "n": 5}, {"concept": "access time", "concept_key": "c-taa", "n": 2}]
                return SimpleNamespace(data=lambda: rows)

        def spec_payload(fid, prop, value, when, series_index, points):
            return {"gr_id": fid, "graph_version": gv, "subject": "ZK200", "property": prop, "concept": "supply voltage", "concept_key": "c-vcc", "value": value,
                    "unit": "V", "unit_canonical": "V", "when": when, "valid_from": when, "series_key": "s-vcc", "series_index": series_index,
                    "conditions": {}, "kinds": {"value": "scalar"}, "comparable": True, "text": f"ZK200 {prop} {value} V", "rel_path": f"datasheets/{when}.pdf",
                    "point_ids": points}

        specs = {point_id_for(gv, "f1"): spec_payload("f1", "Supply voltage", "3.3", "rev A", 0, ["p1", "p1b", "p1c", "p1d"]),
                 point_id_for(gv, "f2"): spec_payload("f2", "VCC", "3.6", "rev B", 1, ["p2"])}        # the payload of f3 cannot be fetched (the alias is switching versions)
        chunks = {"p1": {"is_active": True, "doc_id": "d1", "rel_path": "datasheets/rev A.pdf", "filename": "rev A.pdf", "chunk_index": 4, "content_version": "v1", "page_idx": 2,
                         "section_path": ["DC Characteristics"]},
                  "p1b": {"is_active": False, "doc_id": "d1", "rel_path": "datasheets/rev A.pdf", "chunk_index": 5, "content_version": "v0", "page_idx": 2}}      # p1c and p2 are not in the main collection

        class Q:
            calls: list = []
            def retrieve(self, collection_name, ids, with_payload, with_vectors):
                Q.calls.append((collection_name, list(ids), with_payload))
                table = specs if collection_name == "graph_002_spec" else chunks
                return [SimpleNamespace(id=i, payload=table[i]) for i in ids if i in table]

        driver = SimpleNamespace(session=lambda: Session(), close=mock.Mock())
        src = SimpleNamespace(kb_id="kb_002", collection="kb_002", source_root="datasheets")
        with mock.patch("kb_pipeline.graph.neo4j_import.active_neo4j_graph_version", return_value=gv):
            out = graphwalk.list_facts(SimpleNamespace(), src, subject=" ZK 200 ", prop=" Voltage ", limit=3, offset=2, fields=service.FACT_FIELDS, q=Q(), driver=driver)
        self.assertEqual((out["found"], out["subject"]["id"], [m["id"] for m in out["matches"]]), (True, "e-main", ["e-small"]))    # several with the same name: the best connected one, the rest listed
        self.assertEqual(next(kw for t, kw in log if "$name" in t)["name"], "zk200")                                               # case and spaces do not matter
        self.assertEqual(out["property"], {"query": "Voltage", "matched": "contains", "concepts": 1})                              # containment only when nothing matched exactly
        self.assertEqual((out["total"], out["offset"], out["count"], out["has_more"], out["kb_name"]), (5, 2, 3, False, "datasheets"))
        page = next(t for t, kw in log if "AS row" in t)
        self.assertIn("(e:Entity {kb_id: $kb, graph_version: $gv, id: $sid})-[:HAS_SPEC]->(f:Spec)", page)                         # along the subject's fact edges
        self.assertIn("f.concept_key IN $keys OR", page)                                                                         # every spelling of the same concept comes along
        self.assertEqual(next(kw for t, kw in log if "AS row" in t)["keys"], ["c-vcc"])
        f1, f2, f3 = out["facts"]
        self.assertEqual([f["n"] for f in out["facts"]], [3, 4, 5])                                                                # numbering continues from the page offset
        self.assertEqual((f1["hint"], f1["series_text"]), ("ZK200 · Supply voltage = 3.3 V · rev A", "3.3 V(rev A)、3.6 V(rev B)"))     # the same hints and series as the fact rows of /search
        self.assertEqual([(e["point_id"], e["active"]) for e in f1["evidence"]], [("p1", True), ("p1b", False), ("p1c", False)])      # at most three evidence chunks per fact
        self.assertEqual((f1["evidence"][0]["doc_id"], f1["evidence"][0]["chunk_index"], f1["evidence"][0]["position"], f1["evidence"][0]["place"]),
                         ("d1", 4, "page 2 · DC Characteristics", "page 2"))
        self.assertEqual((f1["verified"], f1["sources_active"]), (True, "1/3"))
        self.assertEqual((f2["verified"], f2["sources_active"], f2.get("hint")), (False, "0/1", None))                             # every source point inactive: no hint, not a current fact
        self.assertEqual((f3["id"], f3["value"], f3["evidence"]), ("f3", "3.5", [])); self.assertNotIn("text", f3)               # payload not fetched: only the basic graph-database fields
        self.assertEqual(out["degraded"], ["spec_payload_missing: 1"])
        self.assertNotIn("properties", out)                                                                                      # the property list only on the first page
        self.assertNotIn("sources", f1); self.assertNotIn("score", f1)
        self.assertEqual([c[0] for c in Q.calls], ["graph_002_spec", "kb_002"])
        # only a property (no subject): across subjects, ordered by subject first; no property list on the first page either
        log.clear()
        with mock.patch("kb_pipeline.graph.neo4j_import.active_neo4j_graph_version", return_value=gv):
            cross = graphwalk.list_facts(SimpleNamespace(), src, prop="voltage", match="contains", fields=service.FACT_FIELDS, q=None, driver=driver)
        page = next(t for t, kw in log if "AS row" in t)
        self.assertIn("MATCH (f:Spec {kb_id: $kb, graph_version: $gv})", page); self.assertIn("ORDER BY coalesce(f.subject, ''),", page)
        self.assertEqual((cross["subject"], cross["property"]["matched"], [f["n"] for f in cross["facts"]]), (None, "contains", [1, 2, 3]))
        self.assertNotIn("properties", cross); self.assertTrue(all(f["evidence"] == [] and f.get("verified") is None for f in cross["facts"]))
        self.assertFalse([t for t, kw in log if " = $p" in t])                                                                   # match=contains does not try exact first
        # the name did not match: vector candidates, no fact query
        with mock.patch("kb_pipeline.graph.neo4j_import.active_neo4j_graph_version", return_value=gv), \
                mock.patch.object(graphwalk, "resolve_entity", return_value=[]), \
                mock.patch.object(graphwalk, "dense_candidates", return_value=[{"id": "e9", "title": "ZK201"}]):
            miss = graphwalk.list_facts(SimpleNamespace(), src, subject="ZK20", q=None, driver=driver)
        self.assertEqual((miss["found"], miss["total"], miss["facts"], [c["id"] for c in miss["candidates"]]), (False, 0, [], ["e9"]))
        ss = _settings(token="secret")
        auth = {"Authorization": "Bearer secret"}
        with mock.patch.object(service, "runtime", return_value=(SimpleNamespace(sources={"kb_002": src}), ss, None)), \
                mock.patch.object(graphwalk, "list_facts", return_value={"total": 1}) as lf, \
                mock.patch.object(channels, "shared_driver", return_value=driver):
            client = TestClient(create_app())
            r = client.post("/graph/facts", json={"kb_id": "kb_002", "subject": "ZK200", "property": "VCC", "match": "exact", "limit": 10}, headers=auth)
            self.assertEqual((r.status_code, r.json()), (200, {"total": 1}))
            self.assertEqual((lf.call_args.kwargs["prop"], lf.call_args.kwargs["match"], lf.call_args.kwargs["fields"]), ("VCC", "exact", service.FACT_FIELDS))
            self.assertEqual(client.post("/graph/facts", json={"kb_id": "kb_002"}, headers=auth).status_code, 422)                 # at least one of subject and property
            self.assertEqual(client.post("/graph/facts", json={"kb_id": "kb_002", "property": "  "}, headers=auth).status_code, 422)
            self.assertEqual(client.post("/graph/facts", json={"kb_id": "kb_002", "subject": "ZK200", "match": "fuzzy"}, headers=auth).status_code, 422)
            self.assertEqual(client.post("/graph/facts", json={"kb_id": "kb_404", "subject": "ZK200"}, headers=auth).status_code, 404)

    def test_fact_rows_share_one_projection(self) -> None:
        """Graph recall and the facts endpoint use one payload projection: a field added on one side appears on the other."""
        from kb_pipeline.graph.recall import spec_result_row

        row = spec_result_row({"gr_id": "f1", "subject": "X", "property": "VCC", "value": "3.3", "unit": "V", "when": "rev B", "point_ids": ("p1",), "kinds": None})
        self.assertEqual((row["id"], row["subject"], row["value"], row["when"], row["point_ids"], row["kinds"], row["conditions"]), ("f1", "X", "3.3", "rev B", ["p1"], {}, {}))
        for key in ("concept", "concept_key", "valid_from", "series_key", "conflict_group", "conditions_text", "text", "rel_path", "section", "comparable"):
            self.assertIn(key, row)
        self.assertNotIn("score", row)                                         # the retrieval score is added by graph recall itself
        self.assertIn("evidence", service.FACT_FIELDS + ("evidence",)); self.assertNotIn("sources", service.FACT_FIELDS)

    def test_search_and_context_carry_the_library_folder_name(self) -> None:
        """A cited path is "folder name / rel_path": /context carries kb_name, so the caller needs no catalog call to
        build it."""
        src = SimpleNamespace(kb_id="kb_005", collection="kb_005", source_root="products")
        q = SimpleNamespace(scroll=lambda **kw: ([SimpleNamespace(id="p1", payload=_payload("p1", "d1", 0, "body text"))], None))
        with mock.patch.object(service, "runtime", return_value=(SimpleNamespace(sources={"kb_005": src}), _settings(), q)):
            out = service.context("kb_005", "d1", 0, 0)
        self.assertEqual((out["kb_name"], len(out["sources"])), ("products", 1))

    def test_source_row_budget_and_doc_aggs(self) -> None:
        hits = [{"point_id": f"p{i}", "kb_id": "kb_001", "scores": {"score_text": 0.5}, "recall_sources": ["text"],
                 "payload": _payload(f"p{i}", "d1" if i < 2 else "d2", i, f"第 {i} 段正文内容,各不相同。" * (i + 1))} for i in range(3)]
        rows, stats = assemble_sources(hits, budget_tokens=10_000, neighbors=None)
        self.assertEqual([r["n"] for r in rows], [1, 2, 3]); self.assertEqual(stats["neighbors"], 0)
        aggs = doc_aggs(rows)
        self.assertEqual([(a["doc"], a["hits"]) for a in aggs], [("d1.pdf", 2), ("d2.pdf", 1)])


class SkillClientTests(unittest.TestCase):
    """The agent-side client under skills/carrel-search. The agent pastes its citations into answers as they are,
    so their shape is a contract: a path from the knowledge base's top-level folder plus where in the file."""

    @staticmethod
    def _client():
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parents[2] / "skills" / "carrel-search" / "scripts" / "carrel_search.py"
        spec = importlib.util.spec_from_file_location("carrel_search_client", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_citations_start_at_the_knowledge_base_folder(self) -> None:
        c = self._client()
        result = {"kbs": ["kb_005"], "kb_names": {"kb_005": "products"},
                  "sources": [{"n": 1, "kb_id": "kb_005", "rel_path": "vendor/manual.pdf", "position": "page 6 · 1. Setup", "place": "page 6"},
                              {"n": 2, "kb_id": "kb_005", "rel_path": "vendor/prices.xlsx", "position": "sheet Prices rows 6–13", "place": "sheet Prices rows 6–13"},
                              {"n": 3, "kb_id": "kb_005", "rel_path": "notes/guide.md", "position": "Part 1 · Overview / A / B install", "place": "A / B install"},
                              {"n": 4, "kb_id": "kb_009", "rel_path": "x/y.pdf", "position": "", "place": ""}],
                  "specs": [{"n": 1, "kb_id": "kb_005", "rel_path": "vendor/manual.pdf", "sources": [1]},
                            {"n": 2, "kb_id": "kb_005", "rel_path": "vendor/manual.pdf", "sources": []}],
                  "entities": [{"n": 1, "kb_id": "kb_005", "docs": ["vendor/manual.pdf", "notes/guide.md"]}]}
        c.add_cites(result)
        self.assertEqual([s["cite"] for s in result["sources"]],
                         ["products/vendor/manual.pdf page 6", "products/vendor/prices.xlsx sheet Prices rows 6–13",
                          "products/notes/guide.md A / B install",       # the service's locator as it is: the client never cuts a position string
                          "x/y.pdf"])                                    # folder name unknown: the path inside the knowledge base
        self.assertEqual([s["cite"] for s in result["specs"]], ["products/vendor/manual.pdf page 6", "products/vendor/manual.pdf"])   # a fact borrows the locator of the source it points at
        self.assertEqual(result["entities"][0]["docs_cite"], ["products/vendor/manual.pdf", "products/notes/guide.md"])
        single = {"kb_id": "kb_002", "kb_name": "datasheets",
                  "facts": [{"n": 1, "rel_path": "a.pdf", "evidence": [{"point_id": "p1", "rel_path": "a.pdf", "position": "page 2 · DC", "place": "page 2"},
                                                                      {"point_id": "p2", "rel_path": "b.pdf", "place": "page 9"}]}],
                  "entities": [{"id": "e1", "docs": ["a.pdf"], "doc_count": 4}]}
        c.add_cites(single)
        fact = single["facts"][0]
        self.assertEqual((fact["cite"], [e["cite"] for e in fact["evidence"]]), ("datasheets/a.pdf page 2", ["datasheets/a.pdf page 2", "datasheets/b.pdf page 9"]))
        split = {"kb_id": "kb_002", "kb_name": "datasheets",
                 "sources": [{"n": 3, "rel_path": "a.xlsx", "place": "sheet S rows 2–10"}, {"n": 16, "rel_path": "a.xlsx", "place": "sheet S rows 9–21"}],
                 "specs": [{"n": 1, "rel_path": "a.xlsx", "sources": [3, 16]}, {"n": 2, "rel_path": "a.xlsx", "sources": [16]}]}
        c.add_cites(split)
        self.assertEqual([s["cite"] for s in split["specs"]], ["datasheets/a.xlsx", "datasheets/a.xlsx sheet S rows 9–21"])   # chunks that disagree on the location: the fact stops at the document instead of naming the wrong rows
        self.assertEqual(single["entities"][0]["docs_cite"], ["datasheets/a.pdf"])       # a listed entity is cited by its documents
        old_service = {"kb_id": "kb_002", "kb_name": "datasheets", "sources": [{"n": 1, "rel_path": "a.pdf", "position": "page 2 · DC"}]}
        c.add_cites(old_service)
        self.assertEqual(old_service["sources"][0]["cite"], "datasheets/a.pdf")          # a service without `place`: the path alone, nothing guessed from the position
        self.assertNotIn("](", result["sources"][0]["cite"])                # plain text, not a link: files may sit in the mirror directly

    SEARCH = {"call_id": "aaaaaa", "operation": "search", "result": {
        "question": "compare ZK200 and Northwind Gateway", "kbs": ["kb_005"], "kb_names": {"kb_005": "products"},
        "sources": [
            {"n": 1, "role": "hit", "accepted": True, "kb_id": "kb_005", "point_id": "p1", "doc_id": "d1", "content_version": "v1", "chunk_index": 7,
             "doc": "manual.pdf", "rel_path": "vendor/manual.pdf", "place": "page 6", "text": "ZK200 supports hot standby.",
             "stitched": {"chunk_from": 6, "chunk_to": 7}},
            {"n": 2, "role": "hit", "accepted": False, "kb_id": "kb_005", "point_id": "p2", "doc_id": "d2", "content_version": "v3", "chunk_index": 0,
             "doc": "board.pdf", "rel_path": "vendor/board.pdf", "place": "page 1", "text": "FACTS: a wiring diagram.\nENTITIES: ZK200, Northwind\nKEYWORDS: wiring",
             "visual": {"confidence": "high", "value_conflicts": 2}},
            {"n": 3, "role": "neighbor", "of": 1, "accepted": None, "kb_id": "kb_005", "point_id": "p3", "doc_id": "d1", "content_version": "v1",
             "chunk_index": 8, "doc": "manual.pdf", "rel_path": "vendor/manual.pdf", "place": "page 6", "text": "The standby unit takes over in 2 s."}]
        + [{"n": n, "role": "hit", "accepted": True, "kb_id": "kb_005", "point_id": "p%d" % n, "doc_id": "d3", "content_version": "v1", "chunk_index": n,
            "doc": "notes.md", "rel_path": "vendor/notes.md", "place": "Setup", "text": ("Step %d of the setup. " % n) * 12} for n in (4, 5, 6)],
        "doc_aggs": [{"doc": "manual.pdf", "rel_path": "vendor/manual.pdf", "source_ns": [1]}, {"doc": "board.pdf", "rel_path": "vendor/board.pdf", "source_ns": [2]}],
        "specs": [{"n": 1, "kb_id": "kb_005", "hint": "ZK200 · failover time = 2 s", "rel_path": "vendor/manual.pdf", "sources": [1], "conflict": True},
                  {"n": 2, "kb_id": "kb_005", "text": "ZK200 · weight: 3 kg", "rel_path": "vendor/sheet.pdf", "sources": [], "verified": False,
                   "series_text": "3 kg (2024), 2.8 kg (2025)"},
                  {"n": 3, "kb_id": "kb_005", "subject": "ZK200", "property": "standby power", "value": "3", "unit": "W", "conditions_text": "mode: idle",
                   "when": "2025-03", "text": "ZK200 · standby power: 3 W | mode: idle | when: 2025-03", "rel_path": "vendor/manual.pdf", "sources": [],
                   "evidence": [{"point_id": "p12", "doc_id": "d1", "content_version": "v1", "chunk_index": 12, "rel_path": "vendor/manual.pdf", "place": "page 9",
                                 "active": True, "located": True},
                                {"point_id": "p11", "doc_id": "d1", "content_version": "v1", "chunk_index": 11, "rel_path": "vendor/manual.pdf", "place": "page 8",
                                 "active": True}]}],
        "pages": [{"n": 1, "kb_id": "kb_005", "kind": "subject", "title": "ZK200", "summary": "A gateway controller.", "sources_active": "2/2",
                   "docs": ["vendor/manual.pdf", "vendor/sheet.pdf"], "text": "# ZK200\nReleased in 2024."},
                  {"n": 2, "kb_id": "kb_005", "kind": "source", "title": "manual.pdf", "summary": "manual.pdf · ZK200, Northwind", "docs": ["vendor/manual.pdf"]}],
        "neighborhoods": [{"n": 1, "kb_id": "kb_005", "id": "e1", "title": "ZK200", "type": "product", "named": True, "relations": 30, "facts": 12,
                           "predicates": [{"type": "part_of", "count": 20}, {"type": "supports", "count": 10}],
                           "neighbors": [{"relation_id": "r1", "type": "part_of", "direction": "in", "directed": True, "other": {"id": "e7", "title": "Control module", "degree": 4}},
                                         {"relation_id": "r2", "type": "related_to", "direction": "out", "directed": False, "other": {"id": "e2", "title": "Northwind Gateway", "degree": 9}}]}],
        "entities": [{"n": 1, "kb_id": "kb_005", "id": "e1", "title": "ZK200", "type": "product"},
                     {"n": 2, "kb_id": "kb_005", "id": "e9", "title": "ZK200", "type": "product"}],
        "relationships": [{"n": 1, "kb_id": "kb_005", "source": "ZK200", "type": "supports", "target": "hot standby"},
                          {"n": 2, "kb_id": "kb_005", "source": "ZK200", "type": "supports", "target": "hot standby"}],
        "retrieval_summary": {"evidence_state": "accepted", "degraded": ["kb_005:graph"], "routing": {"weak": True}}}}
    NEIGHBORS = {"call_id": "bbbbbb", "operation": "neighbors", "result": {
        "kb_id": "kb_005", "kb_name": "products", "found": True, "entity": {"id": "e1", "title": "ZK200", "type": "product", "facts": 12},
        "total": 30, "count": 3, "predicates": [{"type": "part_of", "count": 20}],
        "matches": [{"id": "e9", "title": "ZK200", "type": "module", "degree": 2}],
        "neighbors": [
            {"type": "part_of", "direction": "in", "directed": True, "type_violation": True, "other": {"id": "e7", "title": "Control module", "type": "module", "degree": 4},
             "evidence": [{"point_id": "p8", "doc_id": "d1", "content_version": "v1", "chunk_index": 3, "rel_path": "vendor/manual.pdf", "place": "page 2",
                           "active": True, "excerpt": "The control module is part of ZK200.", "excerpt_match": "both"}]},
            {"type": "supports", "direction": "out", "directed": True, "other": {"id": "e2", "title": "Northwind Gateway", "type": "product", "degree": 9},
             "evidence": [{"point_id": "p9", "doc_id": "d1", "content_version": "v1", "chunk_index": 5, "rel_path": "vendor/manual.pdf", "place": "page 2",
                           "active": True, "excerpt": "A block diagram.", "excerpt_match": "center", "visual": True}]},
            {"type": "requires", "direction": "out", "directed": True, "other": {"id": "e3", "title": "Power unit", "type": "module", "degree": 1},
             "evidence": [{"point_id": "p9", "doc_id": "d1", "content_version": "v1", "chunk_index": 5, "rel_path": "vendor/manual.pdf", "place": "page 2",
                           "active": True, "excerpt": "A block diagram.", "excerpt_match": "center", "visual": True}]}]}}
    ENTITIES = {"call_id": "cccccc", "operation": "entities", "result": {
        "kb_id": "kb_005", "kb_name": "products", "total": 1, "offset": 0, "types": [{"type": "product", "parent_type": "entity", "count": 3}],
        "entities": [{"id": "e1", "title": "ZK200", "type": "product", "degree": 30, "doc_count": 2, "docs": ["vendor/manual.pdf"]}]}}
    NO_FACTS = {"call_id": "dddddd", "operation": "facts", "result": {"kb_id": "kb_005", "kb_name": "products", "found": True, "total": 0, "offset": 0, "facts": []}}
    CONTEXT = {"call_id": "eeeeee", "operation": "context", "result": {
        "kb_id": "kb_005", "kb_name": "products", "doc_id": "d2", "tokens_total": 10,
        "sources": [{"n": 1, "role": "context", "kb_id": "kb_005", "point_id": "p2", "doc_id": "d2", "doc": "board.pdf", "rel_path": "vendor/board.pdf",
                     "place": "page 1", "text": "FACTS: a wiring diagram.\nKEYWORDS: wiring", "visual": {"confidence": "high"}}]}}

    def _stored(self, c, work, *envelopes):
        import copy

        for envelope in envelopes:
            envelope = copy.deepcopy(envelope)
            c.add_cites(envelope["result"])
            c.number_rows(envelope["result"])
            (work / (envelope["call_id"] + ".json")).write_text(json.dumps(envelope, ensure_ascii=False), encoding="utf-8")
        return [c.stored(work, e["call_id"]) for e in envelopes]

    def test_compact_view_keeps_what_the_agent_reads_and_points_at(self) -> None:
        """A command prints a compact view instead of the complete response: the text of the sources with ready citations,
        facts, page summaries and graph leads, every entry under a label later commands can point at."""
        import contextlib
        import tempfile
        from pathlib import Path

        c = self._client()
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            search, neighbors = self._stored(c, work, self.SEARCH, self.NEIGHBORS)
            view = c.compact(search, work / "aaaaaa.json").splitlines()
            near = c.compact(neighbors, work / "bbbbbb.json").splitlines()
            tight = c.view_search(search, limit=1500)
            shown = io.StringIO()
            with contextlib.redirect_stdout(shown):
                c.show(work, ["aaaaaa:S3", "aaaaaa:P1", "aaaaaa:H1.2"])
        self.assertEqual(view[0], "call aaaaaa · search · knowledge base: products (kb_005) · evidence: accepted · 5 hits + 1 neighbouring chunks")
        self.assertIn("gaps in this retrieval: kb_005:graph", view); self.assertIn("routing: weak vector evidence (weak)", view)
        at = view.index
        # a hit is printed with its text; its neighbouring chunks are named, not printed; a citation is printed once and pointed at afterwards
        self.assertEqual(view[at("[S1] products/vendor/manual.pdf page 6   (stitched with adjacent chunks)"):][:3],
                         ["[S1] products/vendor/manual.pdf page 6   (stitched with adjacent chunks)", "ZK200 supports hot standby.",
                          "     neighbouring chunks (context; show when needed): S3 (page 6)"])
        picture = ("[S2] products/vendor/board.pdf page 1   (below the relevance threshold; a lead only; picture · confidence high; "
                   "2 conflicts between the text in the picture and estimated readings; trust the text in the picture; see the picture: image --ref aaaaaa:S2)")
        self.assertEqual(view[at(picture):][:2], [picture, "FACTS: a wiring diagram."])                # the index lines of a picture chunk are left out
        self.assertNotIn("The standby unit takes over in 2 s.", view)
        self.assertEqual(view[at("[S5] same citation as S4"):][1], ("Step 5 of the setup. " * 12).strip())
        self.assertIn("documents: manual.pdf (S1); board.pdf (S2)", view)
        self.assertIn("[F1] ZK200 · failover time = 2 s [conflict] — same citation as S1", view)
        self.assertEqual(view[at("[F2] ZK200 · weight: 3 kg [sources no longer active] — products/vendor/sheet.pdf"):][1], "     series: 3 kg (2024), 2.8 kg (2025)")
        # a fact the service gave no hint reads the same way, and is cited at the chunk its value was found in
        self.assertIn("[F3] ZK200 · standby power = 3 W @ mode: idle · 2025-03 — products/vendor/manual.pdf page 9", view)
        self.assertFalse([line for line in view if line.startswith("[P2]")])                           # a source page without body text is left out
        self.assertIn("[P1] subject · ZK200 · sources active 2/2: A gateway controller. (has body text, show aaaaaa:P1) — products/vendor/manual.pdf (one of 2 documents)", view)
        self.assertIn("[H1] ZK200 (product) · named in the question · 30 relations · 12 facts · part_of 20, supports 10", view)
        self.assertIn("     H1.1 ←part_of Control module (4); H1.2 —related_to— Northwind Gateway (9)", view)
        # entities and relations that read the same are listed once
        self.assertIn("Entities: [E1] ZK200 (product)", view); self.assertIn("Relations: [R1] ZK200 —supports→ hot standby", view)
        self.assertEqual(view[-1], "complete response: " + str(work / "aaaaaa.json"))
        # the reminder to tell the user the conclusion before looking further stands where the agent decides its next step
        self.assertTrue(view[-3].startswith("▶ If you mean to look further: first write the user two or three sentences, as reply text"))
        nothing = dict(search, result=dict(search["result"], retrieval_summary={"evidence_state": "diagnostic", "no_relevant_content": True}))
        self.assertFalse([line for line in c.view_search(nothing) if line.startswith("▶")])            # nothing to conclude from: no reminder
        # the view stays within its budget: the best hits are printed in full whatever it is, later ones by their opening words
        self.assertIn(("Step 5 of the setup. " * 12).strip(), tight)
        self.assertEqual(tight[tight.index("[S6] same citation as S4"):][:2], ["[S6] same citation as S4", "     " + ("Step 6 of the setup. " * 12)[:59] + "…"])
        self.assertIn("only the opening is shown; full text: show aaaaaa:S6", tight)
        self.assertLess(c.nbytes(view), c.VIEW_BYTES)
        # show prints what the view left out: a neighbouring chunk and a page's body as text, anything else with every field
        out = shown.getvalue().splitlines()
        self.assertEqual(out[:2], ["[aaaaaa:S3] products/vendor/manual.pdf page 6   (neighbouring chunk of S1)", "The standby unit takes over in 2 s."])
        self.assertEqual(out[2:5], ["[aaaaaa:P1] subject · ZK200: A gateway controller. — products/vendor/manual.pdf; products/vendor/sheet.pdf", "# ZK200", "Released in 2024."])
        self.assertEqual(json.loads("\n".join(out[6:]))["other"]["id"], "e2")
        self.assertEqual(near[0], "call bbbbbb · neighbors · knowledge base: products (kb_005) · ZK200 (product) · 30 relations in all, 3 here (by weight) · 12 facts under it")
        self.assertIn("relations by kind: part_of 20 (one kind only: --type <predicate>)", near)
        self.assertIn("other entities with this name: ZK200 (module · 2 relations · id=e9)", near)
        self.assertEqual(near[near.index("[N1] ←part_of Control module (module · 4 relations)   (the end types do not fit this kind of relation; judge it by the excerpt)"):][1],
                         '     "The control module is part of ZK200." — products/vendor/manual.pdf page 2')
        self.assertEqual(near[near.index("[N2] →supports Northwind Gateway (product · 9 relations)"):][1],
                         '     "A block diagram." (picture description; the excerpt did not locate the far end; context --ref bbbbbb:N2 when needed) — same citation as N1')
        # an excerpt shared by several relations is printed once
        self.assertEqual(near[near.index("[N3] →requires Power unit (module · 1 relations)"):][1],
                         "     same excerpt as N2 (picture description; the excerpt did not locate the far end; context --ref bbbbbb:N3 when needed) — same citation as N1")

    def test_listings_and_read_back_use_the_same_labels_and_remarks(self) -> None:
        import tempfile
        from pathlib import Path

        c = self._client()
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            entities, none, context = self._stored(c, work, self.ENTITIES, self.NO_FACTS, self.CONTEXT)
            listed, empty, read = c.view_entities(entities), c.view_facts(none), c.view_context(context)
        self.assertIn("entities per type (--type takes the type name; the upper class in brackets goes with --parent-type): product 3 (entity)", listed)
        self.assertEqual(listed[-1], "[E1] ZK200 (product · 30 relations · 2 documents) — products/vendor/manual.pdf")
        self.assertEqual(empty[1:], ["nothing matches"])                                              # no rows: no hint on how to point at one
        # a picture chunk read back says so and how to fetch the picture; its index lines are left out
        self.assertEqual(read[1:], ["[S1] products/vendor/board.pdf page 1   (picture · confidence high; see the picture: image --ref eeeeee:S1)", "FACTS: a wiring diagram."])

    def test_later_commands_point_at_entries_of_stored_responses(self) -> None:
        """--ref "<call id>:<label>" stands for an entry of an earlier response, so the agent neither retypes ids nor writes
        request files: read the original around a chunk, walk the graph from an entity, list the facts under it, search
        inside a document."""
        import tempfile
        from pathlib import Path

        c = self._client()
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            self._stored(c, work, self.SEARCH, self.NEIGHBORS)
            build = lambda *argv: c.payload(c.parser().parse_args(list(argv)), work)
            # the original around a source: a stitched hit is read over its whole range, one chunk either side by default
            self.assertEqual(build("context", "--ref", "aaaaaa:S1"), {"kb_id": "kb_005", "doc_id": "d1", "chunk_from": 5, "chunk_to": 8, "content_version": "v1"})
            self.assertEqual(build("context", "--ref", "aaaaaa:S2", "--before", "3", "--after", "0"),
                             {"kb_id": "kb_005", "doc_id": "d2", "chunk_from": 0, "chunk_to": 0, "content_version": "v3"})
            self.assertEqual(build("context", "--ref", "bbbbbb:N2")["chunk_from"], 4)                    # a relation: its first evidence chunk
            self.assertEqual(build("context", "--ref", "aaaaaa:F1")["doc_id"], "d1")                     # a fact of /search without evidence: the first source it points at
            self.assertEqual(build("context", "--ref", "aaaaaa:F3"),                                      # with evidence: the chunk that holds its value
                             {"kb_id": "kb_005", "doc_id": "d1", "chunk_from": 11, "chunk_to": 13, "content_version": "v1"})
            # walking on: a listed entity, a subject of the neighbourhood block, one of its far ends, the far end of a relation
            self.assertEqual(build("neighbors", "--ref", "aaaaaa:E1"), {"kb_id": "kb_005", "entity_id": "e1"})
            self.assertEqual(build("neighbors", "--ref", "aaaaaa:H1.2", "--type", "supports", "--limit", "5"),
                             {"kb_id": "kb_005", "entity_id": "e2", "limit": 5, "types": ["supports"]})
            self.assertEqual(build("neighbors", "--ref", "bbbbbb:N1"), {"kb_id": "kb_005", "entity_id": "e7"})
            self.assertEqual(build("facts", "--ref", "aaaaaa:H1", "--property", "weight"), {"kb_id": "kb_005", "subject_id": "e1", "property": "weight"})
            self.assertEqual(build("crop", "--ref", "aaaaaa:S2", "--bbox", "0.1,0.2,0.5,0.6"), {"kb_id": "kb_005", "point_id": "p2", "bbox": [0.1, 0.2, 0.5, 0.6]})
            self.assertEqual(build("search", "--question", "failover time", "--in-doc", "aaaaaa:S1", "--block-type", "table"),
                             {"question": "failover time", "kbs": ["kb_005"], "hints": {"block_types": ["table"], "doc_ids": ["d1"]}})
            self.assertEqual(c.chunk_of(*c.resolve(work, "aaaaaa:S2")[:3])[1]["point_id"], "p2")       # the chunk whose picture `image --ref` fetches
            for bad in ("aaaaaa:S9", "aaaaaa:H1.5", "aaaaaa:X1", "cccccc:S1", "S1"):
                with self.assertRaises(ValueError):
                    c.resolve(work, bad)
            with self.assertRaises(ValueError):
                build("neighbors", "--ref", "aaaaaa:S1")                                                # a source is not an entity
            with self.assertRaises(ValueError):
                build("context", "--ref", "aaaaaa:E1")                                                  # an entity is not a chunk
            with self.assertRaises(ValueError):
                build("context", "--ref", "aaaaaa:F2")                                                  # a fact that names no chunk of the response

    def test_work_directory_keeps_recent_responses_only(self) -> None:
        import os
        import re
        import tempfile
        import time
        from pathlib import Path

        c = self._client()
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "work"
            work.mkdir()
            old, new, picture, foreign = work / "aaaaaa.json", work / "bbbbbb.json", work / "cccccc-aaaaaa-S2.png", work / "notes.json"
            for f in (old, new, picture, foreign):
                f.write_text("{}", encoding="utf-8")
            stale = time.time() - c.KEEP_SECONDS - 60
            for f in (old, picture, foreign):
                os.utime(f, (stale, stale))
            self.assertEqual(c.work_dir({"work_dir": str(work)}), work)
            self.assertEqual(sorted(f.name for f in work.iterdir()), ["bbbbbb.json", "notes.json"])     # only what this client stored is removed
            self.assertTrue(re.fullmatch(r"[0-9a-f]{6}", c.new_call(work)))
        rows = {"offset": 50, "entities": [{"id": "e1"}, {"id": "e2"}], "neighbors": [{"type": "part_of"}]}
        c.number_rows(rows)
        self.assertEqual(([e["n"] for e in rows["entities"]], rows["neighbors"][0]["n"]), ([51, 52], 1))   # listed entities are numbered across pages

    def test_listing_commands_build_their_requests(self) -> None:
        c = self._client()
        args = c.parser().parse_args(["entities", "--kb", "kb_005", "--type", "product", "--type", "module", "--parent-type", "entity", "--name", "suite",
                                      "--limit", "20", "--offset", "40"])
        self.assertEqual(c.payload(args), {"kb_id": "kb_005", "limit": 20, "offset": 40, "types": ["product", "module"], "parent_types": ["entity"], "name": "suite"})
        args = c.parser().parse_args(["facts", "--kb", "kb_002", "--subject", "ZK200", "--property", "VCC", "--match", "exact"])
        self.assertEqual(c.payload(args), {"kb_id": "kb_002", "subject": "ZK200", "property": "VCC", "match": "exact"})
        with self.assertRaises(ValueError):
            c.payload(c.parser().parse_args(["facts", "--kb", "kb_002"]))          # at least one of subject and property
        with self.assertRaises(ValueError):
            c.payload(c.parser().parse_args(["entities"]))


if __name__ == "__main__":
    unittest.main()


class RerankSwitchTests(unittest.TestCase):
    """2026-09-28: an empty RERANKER_BASE_URL means there is no rerank service. Even with the switch on, no
    request goes to the empty address; the health endpoint reports rerank=false and the search status is
    disabled rather than degraded."""

    def test_empty_reranker_url_disables_rerank(self) -> None:
        from types import SimpleNamespace

        from kb_search.service import rerank_active

        on = SimpleNamespace(rerank_enabled=True)
        self.assertTrue(rerank_active(SimpleNamespace(reranker_base_url="http://127.0.0.1:8102/v1"), on))
        self.assertFalse(rerank_active(SimpleNamespace(reranker_base_url=""), on))
        self.assertFalse(rerank_active(SimpleNamespace(reranker_base_url="  "), on))
        self.assertFalse(rerank_active(SimpleNamespace(), on))
        self.assertFalse(rerank_active(SimpleNamespace(reranker_base_url="http://r"), SimpleNamespace(rerank_enabled=False)))
        src = (pathlib.Path(__file__).resolve().parents[2] / "app/kb_search/service.py").read_text(encoding="utf-8")
        self.assertIn("if rerank_active(settings, ss) and cands:", src)
        self.assertIn('"rerank": rerank_active(settings, ss)', src)


class ImageCacheBoundaryTests(unittest.TestCase):
    """2026-09-28 security review 4.3: visual_ref must stay inside the parse cache directory; an absolute path or
    a .. escape is treated as "no image"."""

    def test_visual_ref_must_stay_inside_the_cache(self) -> None:
        import tempfile

        from kb_search.images import _cached_file

        with tempfile.TemporaryDirectory() as tmp:
            cache = pathlib.Path(tmp) / "cache"; cache.mkdir()
            settings = SimpleNamespace(cache_dir=str(cache))
            self.assertEqual(_cached_file(settings, {"visual_ref": "kb/doc/img.png"}), (cache / "kb/doc/img.png").resolve())
            for bad in ("../outside.png", "/etc/passwd", "kb/../../x.png", ""):
                with self.assertRaises(KeyError):
                    _cached_file(settings, {"visual_ref": bad})
