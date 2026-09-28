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
from kb_search import channels, evalset, images, service
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


class ConfigTests(unittest.TestCase):
    def test_defaults_are_the_calibrated_values(self) -> None:
        env = {k: v for k, v in os.environ.items() if not k.startswith("KB_SEARCH_")}
        with mock.patch.dict(os.environ, env, clear=True):
            ss = load_search_settings()
        self.assertEqual(ss.rerank_threshold, 0.1); self.assertEqual(ss.route_gap, 0.2); self.assertEqual(ss.route_floor, 0.45)
        self.assertEqual((ss.stitch_min_chars, ss.stitch_max_chars, ss.mmr_lambda, ss.quota_min_hits), (350, 850, 0.7, 2))
        self.assertEqual((ss.graph_hops, ss.final_lex_weight, ss.query_instruction), (1, 0.2, ""))                # fixed values from Q27 / the lexical-weight grid / the instruction ablation
        self.assertTrue(ss.visual_enabled and ss.mmr_enabled and ss.route_widen)


class EvidenceTests(unittest.TestCase):
    def test_stitch_short_hit_alternates_neighbors_within_limits(self) -> None:
        row = {"block_type": "text", "chunk_index": 5, "text": "短片。" * 10}
        nbs = [{"chunk_index": 4, "block_type": "text", "text": "前一片。" * 30}, {"chunk_index": 6, "block_type": "text", "text": "后一片。" * 30},
               {"chunk_index": 7, "block_type": "table", "text": "| a | b |"}, {"chunk_index": 3, "block_type": "text", "text": "再前。" * 500}]
        self.assertTrue(stitch_short_hit(row, nbs, min_chars=350, max_chars=850))
        self.assertTrue(row["text"].startswith("前一片。")); self.assertIn("后一片。", row["text"]); self.assertNotIn("| a |", row["text"])
        self.assertEqual((row["stitched"]["chunk_from"], row["stitched"]["chunk_to"], row["stitched"]["own_chars"]), (4, 6, 30)); self.assertLessEqual(len(row["text"]), 850)
        self.assertEqual([pc["chunk_index"] for pc in row["stitched"]["pieces"]], [4, 5, 6])                # each piece can be cited on its own (S03)
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
        with self.assertRaises(ValueError):
            images.resolve_bbox([0, 0, 5000, 5000], 100, 100)
        with self.assertRaises(ValueError):
            images.resolve_bbox([0.5, 0.5, 0.5, 0.5], 100, 100)
        img = Image.new("RGB", (400, 200), (255, 255, 255)); buf = io.BytesIO(); img.save(buf, format="PNG")
        out = images.crop_image({"bytes": buf.getvalue(), "mime": "image/png", "source": "cache"}, [0.25, 0.0, 0.75, 1.0], pad=10)
        self.assertEqual(out["mime"], "image/png"); self.assertEqual(out["box"], [90, 0, 310, 200]); self.assertEqual((out["width"], out["height"]), (220, 200))


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
                return {"kbs": ["kb_001"], "sources": [], "specs": [], "pages": [], "retrieval_summary": {"rerank": "below_threshold", "rerank_max": 0.02, "low_confidence": True, "timings_ms": {"total_ms": 100}}}
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


def _payload(pid: str, doc: str, idx: int, text: str, **extra):
    return {"point_id": pid, "chunk_uid": f"u{pid}", "doc_id": doc, "content_version": "v1", "chunk_index": idx, "chunk_total": 5,
            "filename": f"{doc}.pdf", "rel_path": f"dir/{doc}.pdf", "page_idx": idx, "section_path": ["S"], "block_type": "text",
            "text": text, "token_count": 20, "is_active": True, **extra}


class OrchestrationTests(unittest.TestCase):
    """Orchestration with stubs: two channels hitting the same chunk, the graph channel carrying entities, BM25
    returning only ids that need backfilling, the visual channel hitting an image chunk; rerank working and
    failing; multi-subject / multi-document quotas, boilerplate downweighting, widening."""

    def _run(self, *, rerank_fn=None, kbs=None, settings_over=None, question="我的尿酸多少", graph_entities=None, extra_vec=None, extra_bm25=None,
             image=None, visual_fn=None, vec_delay=None, graph_specs=None, hints=None):
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
        entries = [{"kb_id": "kb_001", "has_graph": True}, {"kb_id": "kb_002", "has_graph": False}]
        calls = {"rerank_docs": [], "graph": [], "visual_query": 0}
        ents = graph_entities if graph_entities is not None else [{"title": "尿酸", "type": "biomarker", "score": 0.7, "hop": 0, "via": "lexical", "docs": ["dir/d1.pdf"]}]

        def fake_rerank(self, query, documents):
            calls["rerank_docs"] = list(documents)
            if rerank_fn:
                return rerank_fn(documents)
            return [0.9 if "尿酸 433" in d else (0.5 if "甘油" in d else (0.3 if "血糖" in d else 0.1)) for d in documents]

        def fake_graph(settings, source, question, *, limit, hops, vector=None, lexical_only=False):
            calls["graph"].append(source.kb_id)
            return {"chunks": [{"point_id": "p1", "score": 0.4, "rel_path": "dir/d1.pdf", "entities": ["尿酸"], "relations": []}], "entities": ents,
                    "relations": [], "specs": graph_specs if graph_specs is not None else [{"id": "s1", "subject": "李", "property": "尿酸", "value": "433", "score": 0.9, "point_ids": ["p1"]}],
                    "pages": [{"kind": "subject", "title": "尿酸", "score": 0.8, "summary": "主体页", "text": "尿酸 时间线 433", "series": ["x"], "point_ids": ["p1"]}],
                    "graph_version": "001-v", "seeds": {"entities": 1}}

        def fake_visual_query(settings, *, text=None, image_bytes=None, timeout=20.0):
            calls["visual_query"] += 1
            calls["visual_image"] = image_bytes
            calls["visual_text"] = text
            return [0.1, 0.2, 0.3, 0.4]

        def default_visual(q, c, v, limit, timeout=None, query_filter=None):
            return [{"point_id": "p3", "score": 0.7, "payload": docs["p3"]}] if c == "kb_001" else []

        def fake_neighbors(q, collection, payload, *, span):
            idx = int(payload["chunk_index"])
            return [dict(p) for p in docs.values() if p["doc_id"] == payload["doc_id"] and p["chunk_index"] != idx and abs(p["chunk_index"] - idx) <= span]

        def fake_vec(q, c, v, limit, timeout=None, query_filter=None):
            calls["vec_filter"] = query_filter
            if vec_delay and c == "kb_002":
                import time as _t
                _t.sleep(vec_delay)
            if c == "kb_001":
                return [{"point_id": "p1", "score": 0.8, "payload": docs["p1"]}, {"point_id": "p3", "score": 0.6, "payload": docs["p3"]}] + list(extra_vec or [])
            return [{"point_id": "p9", "score": 0.3, "payload": docs["p9"]}]

        def fake_bm25(url, question, c, limit, identifiers=None, timeout=None, filters=None):
            calls["bm25_filters"] = filters
            if c == "kb_001":
                return [{"point_id": "p2", "score": 9.0, "payload": {"doc_id": "d1", "rel_path": "dir/d1.pdf"}}, {"point_id": "p1", "score": 7.0, "payload": {"doc_id": "d1", "rel_path": "dir/d1.pdf"}}] + list(extra_bm25 or [])
            return []

        with mock.patch.object(service, "runtime", return_value=(settings, ss, object())), \
                mock.patch.object(catalog_mod, "get_catalog", return_value=entries), \
                mock.patch.object(channels, "embed_question", side_effect=lambda settings, question, timeout=10.0, instruction=None: [1.0, 0.0]), \
                mock.patch.object(channels, "vector_channel", side_effect=fake_vec), \
                mock.patch.object(channels, "point_meta", side_effect=lambda q, c, ids: {i: ({"active": True, "doc_id": docs[i]["doc_id"], "rel_path": docs[i]["rel_path"], "content_version": "v1"} if i in docs else {"active": False}) for i in ids}), \
                mock.patch.object(channels, "bm25_channel", side_effect=fake_bm25), \
                mock.patch.object(channels, "graph_channel", side_effect=fake_graph), \
                mock.patch.object(channels, "lexical_profile", return_value={"sizes": {"kb_001": 5, "kb_002": 1}, "terms": {"尿酸": {"kb_001": 2}}}), \
                mock.patch.object(channels, "embed_visual_query", side_effect=fake_visual_query), \
                mock.patch.object(channels, "visual_channel", side_effect=visual_fn or default_visual), \
                mock.patch.object(channels, "table_head", return_value=None), \
                mock.patch.object(channels, "fetch_payloads", side_effect=lambda q, c, ids: {i: docs[i] for i in ids if i in docs}), \
                mock.patch.object(channels, "neighbor_payloads", side_effect=fake_neighbors), \
                mock.patch.object(Reranker, "score", fake_rerank):
            out = service.search(question, kbs=kbs, explain=True, image_bytes=image, hints=hints)
        return out, calls

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

    def test_slow_channel_is_dropped_at_the_deadline(self) -> None:
        import time as _t
        t = _t.time()
        out, _ = self._run(vec_delay=0.8, settings_over={"channel_timeout": 0.15})
        s = out["retrieval_summary"]
        self.assertLess(_t.time() - t, 0.7); self.assertIn("kb_002:text", s["degraded"])                              # a slow channel is dropped at the deadline instead of holding the whole request (S06)
        self.assertEqual(out["kbs"], ["kb_001"]); self.assertTrue([r for r in out["sources"] if r["role"] == "hit"])

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

        def fake_graph(settings, source, question, *, limit, hops, vector=None, lexical_only=False):
            calls["graph"].append(source.kb_id); calls["lexical_only"] = lexical_only
            return {"chunks": [{"point_id": "p1", "score": 0.4, "entities": ["尿酸"], "relations": []}], "entities": [], "relations": [], "specs": [], "pages": [], "graph_version": "v"}

        with mock.patch.object(service, "runtime", return_value=(settings, ss, object())), \
                mock.patch.object(catalog_mod, "get_catalog", return_value=[{"kb_id": "kb_001", "has_graph": True}]), \
                mock.patch.object(channels, "embed_question", side_effect=RuntimeError("8101 down")), \
                mock.patch.object(channels, "bm25_channel", return_value=[{"point_id": "p2", "score": 9.0, "payload": {"doc_id": "d1"}}, {"point_id": "p1", "score": 7.0, "payload": {"doc_id": "d1"}}]), \
                mock.patch.object(channels, "graph_channel", side_effect=fake_graph), \
                mock.patch.object(channels, "point_meta", return_value={}), \
                mock.patch.object(channels, "table_head", return_value=None), \
                mock.patch.object(channels, "fetch_payloads", side_effect=lambda q, c, ids: {i: docs[i] for i in ids if i in docs}), \
                mock.patch.object(channels, "neighbor_payloads", return_value=[]), \
                mock.patch.object(Reranker, "score", lambda self, query, documents: [0.8] * len(documents)):
            out = service.search("尿酸多少", kbs=["kb_001"])
        return out, calls


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
            self.assertEqual(client.post("/search", json={"question": ""}, headers=auth).status_code, 422)
            client.post("/search", json={"question": "q", "image_b64": "data:image/png;base64," + __import__("base64").b64encode(b"IMG").decode()}, headers=auth)
            self.assertEqual(search_mock.call_args.kwargs["image_bytes"], b"IMG")
            r = client.get("/image/kb_001/p1", headers=auth)
            self.assertEqual((r.status_code, r.headers["content-type"], r.headers["x-image-source"], r.content), (200, "image/png", "pdf-embedded", b"PNG"))
            self.assertEqual(client.get("/image/kb_001/p1").status_code, 401)
            r2 = client.post("/crop", json={"kb_id": "kb_001", "point_id": "p1", "bbox": [0.1, 0.1, 0.5, 0.5]}, headers=auth)
            self.assertEqual((r2.status_code, r2.headers["x-crop-box"], r2.content), (200, "0,0,1,1", b"CROP"))
            self.assertEqual(client.post("/crop", json={"kb_id": "kb_001", "point_id": "p1", "bbox": [0.1, 0.1]}, headers=auth).status_code, 422)

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
                if "RELATED_TO" in cypher:
                    return FakeResult([{"relation_id": "r1", "type": "has_stage", "outgoing": True, "directed": True, "weight": 3.0, "npmi": 0.4, "cooccur": 2,
                                        "description": "FDE 有阶段二", "type_violation": False, "other_id": "e2", "other_title": "阶段二", "other_type": "stage", "other_scope": None, "other_pagerank": 0.1}])
                if "EVIDENCES" in cypher:
                    return FakeResult([{"rid": "r1", "point_id": "p7", "chunk_uid": "u7", "rel_path": "x/fde.pdf", "doc_id": None, "chunk_index": None, "content_version": None, "page_idx": None}])
                if "MENTIONED_IN" in cypher:
                    return FakeResult([{"rel_path": "x/fde.pdf"}])
                if "id: $id" in cypher:
                    return FakeResult([])
                return FakeResult([{"id": "e1", "title": "FDE", "type": "role", "parent_type": None, "scope": None, "description": "前向部署工程师", "pagerank": 0.5, "degree": 9, "aliases": ["Forward Deployed Engineer"]}])

        class FakeDriver:
            def __init__(self): self.s = FakeSession()
            def session(self): return self.s
            def close(self): pass

        src = SimpleNamespace(kb_id="kb_003", collection="kb_003")
        class FakeQ:
            def retrieve(self, collection_name, ids, with_payload, with_vectors):
                return [SimpleNamespace(id="p7", payload={"is_active": True, "doc_id": "d7", "chunk_index": 3, "content_version": "v1", "page_idx": 4, "rel_path": "x/fde.pdf", "filename": "fde.pdf"})]

        with mock.patch("kb_pipeline.graph.neo4j_import.neo4j_driver", return_value=FakeDriver()), \
                mock.patch("kb_pipeline.graph.neo4j_import.active_neo4j_graph_version", return_value="003-v"):
            out = graphwalk.neighbors(SimpleNamespace(), src, entity="fde", entity_id=None, limit=5, types=None, direction="both", q=FakeQ())
        self.assertTrue(out["found"]); self.assertEqual(out["entity"]["title"], "FDE"); self.assertEqual(out["entity"]["docs"], ["x/fde.pdf"])
        nb = out["neighbors"][0]
        self.assertEqual((nb["type"], nb["direction"], nb["other"]["title"]), ("has_stage", "out", "阶段二"))            # predicate, direction, far end
        self.assertEqual((nb["evidence"][0]["doc_id"], nb["evidence"][0]["chunk_index"], nb["evidence"][0]["active"]), ("d7", 3, True))   # the evidence chunk can be fed to /context directly
        ss = _settings(token="secret")
        with mock.patch.object(service, "runtime", return_value=(SimpleNamespace(sources={}), ss, SimpleNamespace(get_collections=lambda: None))), \
                mock.patch.object(service, "graph_neighbors", return_value={"found": True}) as gn:
            client = TestClient(create_app())
            r = client.post("/graph/neighbors", json={"kb_id": "kb_003", "entity": "FDE", "types": ["has_stage"], "direction": "out"}, headers={"Authorization": "Bearer secret"})
            self.assertEqual((r.status_code, r.json()), (200, {"found": True})); self.assertEqual(gn.call_args.kwargs["types"], ["has_stage"])
            self.assertEqual(client.post("/graph/neighbors", json={"kb_id": "kb_003", "entity": "FDE", "direction": "sideways"}, headers={"Authorization": "Bearer secret"}).status_code, 422)

    def test_source_row_budget_and_doc_aggs(self) -> None:
        hits = [{"point_id": f"p{i}", "kb_id": "kb_001", "scores": {"score_text": 0.5}, "recall_sources": ["text"],
                 "payload": _payload(f"p{i}", "d1" if i < 2 else "d2", i, f"第 {i} 段正文内容,各不相同。" * (i + 1))} for i in range(3)]
        rows, stats = assemble_sources(hits, budget_tokens=10_000, neighbors=None)
        self.assertEqual([r["n"] for r in rows], [1, 2, 3]); self.assertEqual(stats["neighbors"], 0)
        aggs = doc_aggs(rows)
        self.assertEqual([(a["doc"], a["hits"]) for a in aggs], [("d1.pdf", 2), ("d2.pdf", 1)])


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
