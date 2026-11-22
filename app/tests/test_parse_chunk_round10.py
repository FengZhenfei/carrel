from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kb_pipeline.chunking.chunker import (
    EMBED_CONTEXT_MAX_CHARS,
    blocks_to_chunks,
    chunk_text,
    embedding_context_for,
)
from kb_pipeline.headings import (
    HeadingResolver,
    detect_toc_pages,
    explicit_toc_entries,
    infer_heading_level,
)
from kb_pipeline.models import ParsedBlock
from kb_pipeline.parsers.common import SectionTracker, html_table_to_markdown
from kb_pipeline.parsers.html_dom import parse_html_dom
from kb_pipeline.parsers import mineru_pdf
from kb_pipeline.parsers.pdf_enhanced import merge_mineru_text_blocks
from kb_pipeline.utils import split_sentences


class MergedCellRepetitionTests(unittest.TestCase):
    def test_merged_cells_preserve_per_row_attribution_including_long_notes(self) -> None:
        short_label = "Group A"
        long_note = (
            "这是一段跨多行的长说明文本，详细描述了该参数在各种边界条件、温度范围与异常供电下的具体行为、"
            "兼容性限制以及推荐的排查与处理步骤，每一从属行单独回读或跨切块后都必须保留该说明。"
        )
        html = (
            "<table>"
            "<tr><th>Group</th><th>Item</th><th>Note</th></tr>"
            f"<tr><td rowspan='3'>{short_label}</td><td>R1</td><td rowspan='3'>{long_note}</td></tr>"
            "<tr><td>R2</td></tr>"
            "<tr><td>R3</td></tr>"
            "</table>"
        )
        md = html_table_to_markdown(html)
        lines = [line for line in md.splitlines() if line.strip()]
        self.assertEqual(len(lines), 5)  # header + sep + 3 body rows
        # Both short labels and long shared notes repeat on every spanned row so each row stays self-contained
        self.assertEqual(md.count(short_label), 3)
        self.assertEqual(md.count(long_note), 3)
        for line in lines:
            self.assertEqual(line.count("|"), 4, line)
        self.assertIn(f"| {short_label} | R2 | {long_note} |", lines[3])
        self.assertIn(f"| {short_label} | R3 | {long_note} |", lines[4])

    def test_colspan_and_rowspan_html_dom_parity(self) -> None:
        short_val = "A" * 80
        long_val = "B" * 120
        html = (
            "<table>"
            "<tr><th>C1</th><th>C2</th><th>C3</th></tr>"
            f"<tr><td colspan='2'>{short_val}</td><td>X</td></tr>"
            f"<tr><td colspan='2'>{long_val}</td><td>Y</td></tr>"
            "</table>"
        )
        md_common = html_table_to_markdown(html)
        self.assertEqual(md_common.count(short_val), 1)
        self.assertEqual(md_common.count(long_val), 1)
        self.assertIn(f"| [COLS: C1 / C2] {short_val} |  | X |", md_common)
        self.assertIn(f"| [COLS: C1 / C2] {long_val} |  | Y |", md_common)

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "t.html"
            p.write_text(f"<html><body>{html}</body></html>", encoding="utf-8")
            blocks = parse_html_dom(p)
        table_blocks = [b for b in blocks if b.block_type == "table"]
        self.assertEqual(len(table_blocks), 1)
        self.assertEqual(table_blocks[0].table_markdown.count(short_val), 1)
        self.assertEqual(table_blocks[0].table_markdown.count(long_val), 1)
        self.assertIn(f"| [COLS: C1 / C2] {long_val} |  | Y |", table_blocks[0].table_markdown)


class ChunkerHeadingTrustAndF13Tests(unittest.TestCase):
    @staticmethod
    def _para(n: int, word: str = "word") -> str:
        return " ".join([word] * n) + "."

    def test_unmarked_numbered_body_lines_do_not_split_chunks_or_bind_titles(self) -> None:
        # Unmarked numbered steps inside body text must not trigger chunker-side heading splits (B14)
        body = self._para(220) + "\n\n1. Disconnect the power\n\n2. Reconnect the cable\n\n" + self._para(80, "tail")
        chunks = chunk_text(body, 400, 80)
        self.assertEqual(len(chunks), 1)
        self.assertIn("1. Disconnect the power", chunks[0])

        # Unmarked numbered body line right before a table does NOT become TITLE: when block has no heading_lines
        table_md = "| k | v |\n| --- | --- |\n| 1 | 2 |"
        blocks = [
            ParsedBlock(parser="m", parser_profile="p", doc_type="pdf", block_type="text",
                        text="1.2 First check the wiring", block_id="b1",
                        metadata={"section_path": ["Setup"]}),
            ParsedBlock(parser="m", parser_profile="p", doc_type="pdf", block_type="table",
                        text=table_md, table_markdown=table_md, block_id="t1",
                        metadata={"section_path": ["Setup"]}),
        ]
        out = blocks_to_chunks(kb_id="k", file_key=1, content_version="v", parser_profile="p",
                               blocks=blocks, max_tokens=400, overlap_tokens=80)
        self.assertFalse(any("TITLE: 1.2 First check the wiring" in c.text for c in out))

        # When the parser DOES mark heading_lines, it still becomes TITLE: for the table
        blocks[0].metadata["heading_lines"] = ["1.2 First check the wiring"]
        out_marked = blocks_to_chunks(kb_id="k", file_key=1, content_version="v", parser_profile="p",
                                      blocks=blocks, max_tokens=400, overlap_tokens=80)
        self.assertTrue(any("TITLE: 1.2 First check the wiring" in c.text for c in out_marked))

    def test_note_label_only_drops_before_real_title_block(self) -> None:
        table_md = "| a | b |\n| --- | --- |\n| 1 | 2 |"
        blocks = [
            ParsedBlock(parser="m", parser_profile="p", doc_type="pdf", block_type="table",
                        text=table_md, table_markdown=table_md, block_id="t1", page_idx=1,
                        metadata={"section_path": ["S1"]}),
            ParsedBlock(parser="m", parser_profile="p", doc_type="pdf", block_type="text",
                        text="Note:", block_id="n1", page_idx=1,
                        metadata={"section_path": ["S1"]}),
            ParsedBlock(parser="m", parser_profile="p", doc_type="pdf", block_type="text",
                        text="1. Check voltage before powering on.", block_id="b1", page_idx=1,
                        metadata={"section_path": ["S1"]}),
        ]
        merged = merge_mineru_text_blocks(blocks, target_tokens=400)
        # Note: is NOT dropped before an unmarked numbered body line (B14)
        self.assertTrue(any("Note:" in b.text for b in merged))

    def test_f13_dashed_numbers_removed(self) -> None:
        for sample in ("1-3 SQL概要", "15-20 日内交付", "2-1 数据库种类", "3-5 个工作日"):
            self.assertIsNone(infer_heading_level(sample), sample)
        self.assertEqual(infer_heading_level("1.3 SQL概要"), 2)
        self.assertEqual(infer_heading_level("2.1.4 细节"), 3)


class ChapterWordIntegerSubsectionTests(unittest.TestCase):
    @staticmethod
    def _walk(seq: list[tuple[str, int | None]], **kw) -> list[tuple[int | None, list[str]]]:
        resolver = HeadingResolver((t for t, lv in seq if lv), **kw)
        sections = SectionTracker()
        out = []
        for text, parser_level in seq:
            level, _ = resolver.resolve(parser_level, text)
            sections.observe(level, text)
            out.append((level, list(sections.path)))
        return out

    def test_integer_subsections_stay_consistent_under_chapter_word(self) -> None:
        seq = [
            ("第一章 总则", 1),
            ("1 目的", 1),
            ("注意事项", 1),
            ("2 范围", 1),
            ("3 术语", 1),
            ("第二章 规范", 1),
            ("1 要求", 1),
            ("2 验收", 1),
        ]
        got = self._walk(seq)
        self.assertEqual([lv for lv, _ in got], [1, 2, 3, 2, 2, 1, 2, 2])
        self.assertEqual(got[1][1], ["第一章 总则", "1 目的"])
        self.assertEqual(got[2][1], ["第一章 总则", "1 目的", "注意事项"])
        self.assertEqual(got[3][1], ["第一章 总则", "2 范围"])
        self.assertEqual(got[4][1], ["第一章 总则", "3 术语"])
        self.assertEqual(got[6][1], ["第二章 规范", "1 要求"])
        self.assertEqual(got[7][1], ["第二章 规范", "2 验收"])

    def test_section_word_level2_and_appendix_and_english_chapter(self) -> None:
        seq = [
            ("第一章 总则", 1),
            ("第一节 基本规定", 1),
            ("1 定义", 1),
            ("2 符号", 1),
            ("第二节 引用", 1),
            ("1 标准", 1),
            ("附录 A 术语表", 1),
            ("1 通用术语", 1),
            ("2 专用术语", 1),
        ]
        got = self._walk(seq)
        self.assertEqual([lv for lv, _ in got], [1, 2, 3, 3, 2, 3, 1, 2, 2])
        self.assertEqual(got[2][1], ["第一章 总则", "第一节 基本规定", "1 定义"])
        self.assertEqual(got[3][1], ["第一章 总则", "第一节 基本规定", "2 符号"])
        self.assertEqual(got[7][1], ["附录 A 术语表", "1 通用术语"])
        self.assertEqual(got[8][1], ["附录 A 术语表", "2 专用术语"])

        eng = self._walk([("Chapter 1 Overview", 1), ("1 Goals", 1), ("2 Scope", 1)])
        self.assertEqual([lv for lv, _ in eng], [1, 2, 2])
        self.assertEqual(eng[2][1], ["Chapter 1 Overview", "2 Scope"])

    def test_quantity_headings_under_chapter_word_do_not_displace_sections(self) -> None:
        # After dotted 1.1 under 第一章, "1 级能效" must NOT start a level-2 integer sequence
        seq = [
            ("第一章 总则", 1),
            ("1.1 背景", 1),
            ("1 级能效", 1),
            ("1.2 目标", 1),
        ]
        got = self._walk(seq)
        self.assertEqual([lv for lv, _ in got], [1, 2, 3, 2])
        self.assertEqual(got[2][1], ["第一章 总则", "1.1 背景", "1 级能效"])
        self.assertEqual(got[3][1], ["第一章 总则", "1.2 目标"])

        # After 2 范围, "8 位微控制器" and "1 级能效" stay level 3 under 2 范围
        seq2 = [
            ("第一章 总则", 1),
            ("1 目的", 1),
            ("2 范围", 1),
            ("8 位微控制器", 1),
            ("1 级能效", 1),
            ("3 术语", 1),
        ]
        got2 = self._walk(seq2)
        self.assertEqual([lv for lv, _ in got2], [1, 2, 2, 3, 3, 2])
        self.assertEqual(got2[5][1], ["第一章 总则", "3 术语"])

    def test_recovered_chapter_enters_chapter_word_mode(self) -> None:
        r = HeadingResolver(["1 目的", "2 范围"])
        sections = SectionTracker()
        lv, rec = r.recover(1)
        self.assertEqual((lv, rec), (1, True))
        sections.observe(lv, "第一章 总则")
        for heading in ("1 目的", "2 范围"):
            lv, _ = r.resolve(1, heading)
            sections.observe(lv, heading)
            self.assertEqual(lv, 2)
        self.assertEqual(sections.path, ["第一章 总则", "2 范围"])


class SentenceSplitMarkersTests(unittest.TestCase):
    def test_multi_letter_roman_and_word_number_labels_stay_attached(self) -> None:
        self.assertEqual(split_sentences("II. Scope of the agreement. Next sentence."),
                         ["II. Scope of the agreement.", "Next sentence."])
        self.assertEqual(split_sentences("IV. Method and setup. Details follow."),
                         ["IV. Method and setup.", "Details follow."])
        self.assertEqual(split_sentences("vii. Lowercase roman item. Second."),
                         ["vii. Lowercase roman item.", "Second."])
        self.assertEqual(split_sentences("Step 1. Connect the power cable. Then boot."),
                         ["Step 1. Connect the power cable.", "Then boot."])
        self.assertEqual(split_sentences("Figure 2.1. System architecture overview. See below."),
                         ["Figure 2.1. System architecture overview.", "See below."])
        self.assertEqual(split_sentences("Option A. Enable TLS verification. Otherwise warn."),
                         ["Option A. Enable TLS verification.", "Otherwise warn."])
        self.assertEqual(split_sentences("步骤 1. 安装运行依赖。然后启动服务。"),
                         ["步骤 1. 安装运行依赖。", "然后启动服务。"])
        self.assertEqual(split_sentences("步骤1. 安装运行依赖。然后启动服务。"),
                         ["步骤1. 安装运行依赖。", "然后启动服务。"])

    def test_ordinary_sentences_and_years_still_split(self) -> None:
        self.assertEqual(split_sentences("See Figure 1. Next sentence."),
                         ["See Figure 1.", "Next sentence."])
        self.assertEqual(split_sentences("January 2024. The rollout began."),
                         ["January 2024.", "The rollout began."])
        self.assertEqual(split_sentences("It worked. Then it stopped."),
                         ["It worked.", "Then it stopped."])
        self.assertEqual(split_sentences("Step 1.\nConnect power."),
                         ["Step 1.", "Connect power."])


class TocDetectionAndShortChapterTocTests(unittest.TestCase):
    def test_detect_toc_pages_rejects_bare_short_line_pages_and_keeps_real_tocs(self) -> None:
        # Page 1: 30 bare short spec lines (NO TOC title, NO leaders, NO page numbers) -> NOT TOC
        spec = [(1, f"功能特性条目 {i}") for i in range(30)]
        # Page 2: 22 short lines WITH "目录" title -> TOC
        titled = [(2, "目录")] + [(2, f"第{i}章 章节{i}") for i in range(1, 22)]
        # Page 3: continuation of page 2 (leaderless, no title on p.3, 22 short lines) -> TOC
        cont3 = [(3, f"第{i}章 章节{i}") for i in range(22, 44)]
        # Page 4: continuation of page 3 (3rd page of leaderless TOC) -> TOC
        cont4 = [(4, f"第{i}章 章节{i}") for i in range(44, 66)]
        # Page 10: dot-leader TOC page -> TOC
        leader = [(10, f"Section {i} ...... {i + 5}") for i in range(10)]
        # Page 11: dense short-line list right after a leader TOC (no leaders, no page nums, no TOC title) -> NOT TOC
        after_leader = [(11, f"PIN_{i}") for i in range(25)]
        # Page 15: dense short lines with trailing page numbers (>= 60%), including recurring per-chapter labels
        page_nums = [(15, f"Section {i} Topic {i + 10}") for i in range(15)] + [
            (15, f"Summary {26 + i * 3}") for i in range(5)
        ] + [(15, f"Exercises {27 + i * 3}") for i in range(5)]
        # Page 16: dense key-value lines with non-monotonic trailing numbers -> NOT TOC
        kv_vals = [(16, f"Param_{i} {(i * 37) % 91 + 1}") for i in range(25)]
        got = detect_toc_pages(spec + titled + cont3 + cont4 + leader + after_leader + page_nums + kv_vals)
        self.assertEqual(got, {2, 3, 4, 10, 15})

    def test_explicit_toc_entries_catches_short_chapter_word_toc_and_fixes_body_chapters(self) -> None:
        toc_items = [
            {"type": "text", "text": "CONTENTS", "text_level": 1, "page_idx": 2},
            {"type": "text", "text": "目录", "text_level": 1, "page_idx": 2},
            {"type": "text", "text": "第一章 概述", "text_level": 1, "page_idx": 2},
            {"type": "text", "text": "Overview and Scope", "page_idx": 2},
            {"type": "text", "text": "第二章 架构", "text_level": 1, "page_idx": 2},
            {"type": "text", "text": "Architecture", "page_idx": 2},
            {"type": "text", "text": "第三章 实施", "text_level": 1, "page_idx": 2},
            {"type": "text", "text": "Implementation", "page_idx": 2},
        ]
        body_items = [
            {"type": "text", "text": "第一章 概述", "text_level": 1, "page_idx": 4},
            {"type": "text", "text": "第一章的正文内容较长，说明整体背景和适用范围。", "page_idx": 4},
            {"type": "text", "text": "第二章 架构", "text_level": 1, "page_idx": 8},
            {"type": "text", "text": "第二章的正文内容较长，介绍核心组件之间的关系。", "page_idx": 8},
            {"type": "text", "text": "第三章 实施", "text_level": 1, "page_idx": 12},
            {"type": "text", "text": "第三章的正文内容较长，列出交付里程碑和检查项。", "page_idx": 12},
        ]
        entries = [(it["page_idx"] + 1, it["text"]) for it in toc_items + body_items]
        self.assertEqual(explicit_toc_entries(entries), set(range(len(toc_items))))

        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(mineru_pdf, "call_mineru_sync", return_value={"content_list": toc_items + body_items}):
            blocks = mineru_pdf.mineru_pdf_blocks(mineru_url="http://x", path=Path(tmp) / "doc.pdf", cache_dir=Path(tmp))
        toc_blocks = [b for b in blocks if b.page_idx == 3]
        self.assertTrue(toc_blocks)
        self.assertTrue(all(not b.metadata["section_path"] for b in toc_blocks))
        body_titles = [(b.text, b.metadata["section_path"]) for b in blocks if b.block_type == "title"]
        self.assertEqual(body_titles, [
            ("第一章 概述", ["第一章 概述"]),
            ("第二章 架构", ["第二章 架构"]),
            ("第三章 实施", ["第三章 实施"]),
        ])

    def test_explicit_toc_entries_rejects_mixed_or_non_sequential_pages(self) -> None:
        # Mixed page with prose sentence ending in 。 must NOT be wiped as chapter_toc
        mixed = [(1, "目录"), (1, "第一章 概述"), (1, "第二章 架构"), (1, "本文档介绍系统架构与部署步骤。")]
        self.assertEqual(explicit_toc_entries(mixed), set())
        # Page with an embedded image placeholder must NOT be wiped as chapter_toc
        with_img = [(1, "目录"), (1, "第一章 概述"), (1, "\ufffc"), (1, "第二章 架构")]
        self.assertEqual(explicit_toc_entries(with_img), set())
        # Duplicate chapter numbers (e.g. TOC + body start on same page) are not a strict sequence
        dup = [(1, "目录"), (1, "第一章 概述"), (1, "第二章 架构"), (1, "第一章 概述")]
        self.assertEqual(explicit_toc_entries(dup), set())
        # Contents-only page where PDF reading order places "目录" after "第一章": whole page is TOC
        reordered = [(5, "CONTENT"), (5, "第一章"), (5, "目录"), (5, "概述 01"), (5, "第二章"), (5, "架构 07")]
        self.assertEqual(explicit_toc_entries(reordered), {0, 1, 2, 3, 4, 5})
        # Mixed page with prose BEFORE "目录": prose before "目录" must NOT be marked as TOC
        prose_before = [(1, "前言"), (1, "本文档介绍系统架构与部署步骤。"), (1, "目录"), (1, "第一章 概述"), (1, "第二章 架构")]
        self.assertEqual(explicit_toc_entries(prose_before), {2, 3, 4})


class EmbeddingContextFallbackTests(unittest.TestCase):
    def test_embedding_context_falls_back_to_section_paths_when_common_path_empty(self) -> None:
        raw_blocks = [
            ParsedBlock(parser="m", parser_profile="p", doc_type="pdf", block_type="title",
                        text="第一章 总则", block_id="t1", page_idx=1,
                        metadata={"section_path": ["第一章 总则"]}),
            ParsedBlock(parser="m", parser_profile="p", doc_type="pdf", block_type="text",
                        text="正文一。", block_id="b1", page_idx=1,
                        metadata={"section_path": ["第一章 总则"]}),
            ParsedBlock(parser="m", parser_profile="p", doc_type="pdf", block_type="title",
                        text="第二章 规范", block_id="t2", page_idx=1,
                        metadata={"section_path": ["第二章 规范"]}),
            ParsedBlock(parser="m", parser_profile="p", doc_type="pdf", block_type="text",
                        text="正文二。", block_id="b2", page_idx=1,
                        metadata={"section_path": ["第二章 规范"]}),
        ]
        merged = merge_mineru_text_blocks(raw_blocks, target_tokens=400)
        chunks = blocks_to_chunks(kb_id="k", file_key=1, content_version="v", parser_profile="p",
                                  blocks=merged, max_tokens=400, overlap_tokens=80)
        # Small sibling chapters stay packed into 1 chunk (zero fragmentation)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].block.metadata["section_path"], [])
        ctx = embedding_context_for(chunks[0], doc_name="spec.pdf")
        self.assertEqual(ctx, "spec.pdf > 第一章 总则 / 第二章 规范")

    def test_embedding_context_single_non_empty_path_dedup_and_length_cap(self) -> None:
        raw_blocks = [
            ParsedBlock(parser="m", parser_profile="p", doc_type="pdf", block_type="text",
                        text="封面说明。", block_id="b0", page_idx=1, metadata={"section_path": []}),
            ParsedBlock(parser="m", parser_profile="p", doc_type="pdf", block_type="title",
                        text="第一章 总则", block_id="t1", page_idx=1,
                        metadata={"section_path": ["第一章 总则", "1.1 范围"]}),
            ParsedBlock(parser="m", parser_profile="p", doc_type="pdf", block_type="text",
                        text="正文。", block_id="b1", page_idx=1,
                        metadata={"section_path": ["第一章 总则", "1.1 范围"]}),
        ]
        merged = merge_mineru_text_blocks(raw_blocks, target_tokens=400)
        chunks = blocks_to_chunks(kb_id="k", file_key=1, content_version="v", parser_profile="p",
                                  blocks=merged, max_tokens=400, overlap_tokens=80)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(embedding_context_for(chunks[0], doc_name="doc.pdf"), "doc.pdf > 第一章 总则 > 1.1 范围")

        # Length cap is enforced when many long paths are packed
        many = [
            ParsedBlock(parser="m", parser_profile="p", doc_type="pdf", block_type="text",
                        text=f"短句{i}。", block_id=f"x{i}", page_idx=1,
                        metadata={"section_path": [f"第{i}章 " + "标题" * 15]})
            for i in range(1, 10)
        ]
        packed = blocks_to_chunks(kb_id="k", file_key=1, content_version="v", parser_profile="p",
                                  blocks=merge_mineru_text_blocks(many, target_tokens=500),
                                  max_tokens=500, overlap_tokens=80)
        self.assertEqual(len(packed), 1)
        ctx = embedding_context_for(packed[0], doc_name="long.pdf")
        self.assertLessEqual(len(ctx), EMBED_CONTEXT_MAX_CHARS)
        self.assertTrue(ctx.startswith("long.pdf > 第1章 "))


if __name__ == "__main__":
    unittest.main()


class AdversarialReviewRegressionTests(unittest.TestCase):
    def test_table_repair_expands_declared_spans_for_headers_and_body(self) -> None:
        from kb_pipeline.parsers.table_repair import _expand_review_spans, _identity_columns

        long_hdr = "H" * 90
        long_body = "N" * 90
        html = (
            "<table>"
            f"<tr><th colspan='2'>{long_hdr}</th><th>Val</th></tr>"
            f"<tr><td rowspan='2'>{long_body}</td><td>Row1</td><td>10</td></tr>"
            "<tr><td>Row2</td><td>20</td></tr>"
            "</table>"
        )
        md = html_table_to_markdown(html)
        header_line = md.splitlines()[0]
        self.assertEqual(header_line.count(long_hdr), 2)
        self.assertEqual(md.count(long_body), 2)

        old_grid = [[long_hdr, long_hdr, "Val"], ["Row1", "10", long_body], ["Row2", "20", long_body]]
        vlm_grid = [[long_hdr, "", "Val"], ["Row1", "10", long_body], ["Row2", "20", ""]]
        html_note_col2 = (
            "<table>"
            f"<tr><th colspan='2'>{long_hdr}</th><th>Note</th></tr>"
            f"<tr><td>Row1</td><td>10</td><td rowspan='2'>{long_body}</td></tr>"
            "<tr><td>Row2</td><td>20</td></tr>"
            "</table>"
        )
        expanded = _expand_review_spans(old_grid, vlm_grid, html_note_col2)
        self.assertEqual(expanded[0], [long_hdr, long_hdr, "Val"])
        self.assertEqual(expanded[1][2], long_body)
        self.assertEqual(expanded[2][2], long_body)
        self.assertEqual(_identity_columns(old_grid), (0,))

    def test_gbt_multipart_and_quantity_heading_counterexamples(self) -> None:
        walk = ChapterWordIntegerSubsectionTests._walk
        # GB/T multi-part standard: Part 1 -> 1 Scope -> 1.1 ... -> 2 ... -> 3 -> 3.1 -> 4 -> 4.1 -> 4.1.1
        gbt = [
            ("第1部分：通用要求", 1),
            ("1 范围", 1),
            ("1.1 适用对象", 1),
            ("2 规范性引用文件", 1),
            ("3 术语和定义", 1),
            ("3.1 基本术语", 1),
            ("4 技术要求", 1),
            ("4.1 外观", 1),
            ("4.1.1 表面", 1),
        ]
        got = walk(gbt)
        self.assertEqual([lv for lv, _ in got], [1, 2, 3, 2, 2, 3, 2, 3, 4])
        self.assertEqual(got[2][1], ["第1部分：通用要求", "1 范围", "1.1 适用对象"])
        self.assertEqual(got[5][1], ["第1部分：通用要求", "3 术语和定义", "3.1 基本术语"])
        self.assertEqual(got[8][1], ["第1部分：通用要求", "4 技术要求", "4.1 外观", "4.1.1 表面"])

        # Counterexample A: quantity headings across dotted sections must not hijack level 2
        got_a = walk([("第一章 总则", 1), ("1.1 能效等级", 1), ("4 级标准", 1), ("1.2 目标", 1), ("5 级标准", 1)])
        self.assertEqual([lv for lv, _ in got_a], [1, 2, 3, 2, 3])
        self.assertEqual(got_a[4][1], ["第一章 总则", "1.2 目标", "5 级标准"])

        # Counterexample B: consecutive quantities inside a dotted section
        got_b = walk([("第一章 总则", 1), ("1.1 能效等级", 1), ("1 级能效", 1), ("2 级能效", 1)])
        self.assertEqual([lv for lv, _ in got_b], [1, 2, 3, 3])

        # Counterexample C: consecutive quantities inside integer subsections must not displace section 3
        got_c = walk([("第一章 总则", 1), ("1 目的", 1), ("2 范围", 1), ("8 位微控制器", 1), ("9 位数据帧", 1), ("3 术语", 1)])
        self.assertEqual([lv for lv, _ in got_c], [1, 2, 2, 3, 3, 2])
        self.assertEqual(got_c[5][1], ["第一章 总则", "3 术语"])

    def test_roman_vi_and_no_mid_paragraph_false_merges(self) -> None:
        # Roman numerals 1..39 (including VI and vi!) all stay attached
        romans = [
            "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
            "XI", "XIV", "XV", "XVI", "XIX", "XX", "XXVI", "XXXIX",
        ]
        for r in romans:
            for form in (r, r.lower()):
                self.assertEqual(split_sentences(f"{form}. Item body. Next."), [f"{form}. Item body.", "Next."], form)
        # Mid-paragraph dates, versions, short statements, and CJK cross-references still split
        self.assertEqual(split_sentences("Updated yesterday. January 15. The deployment finished."),
                         ["Updated yesterday.", "January 15.", "The deployment finished."])
        self.assertEqual(split_sentences("Check the runtime. Python 3.12. It is installed by default."),
                         ["Check the runtime.", "Python 3.12.", "It is installed by default."])
        self.assertEqual(split_sentences("Result is ready. Grade A. Return 0. Done."),
                         ["Result is ready.", "Grade A.", "Return 0.", "Done."])
        self.assertEqual(split_sentences("详见图1. 本节介绍系统架构。"),
                         ["详见图1.", "本节介绍系统架构。"])

    def test_toc_variants_split_columns_brochure_and_prose_chain_guard(self) -> None:
        # GB/T "目 次" + bilingual "目录 CONTENTS" + chapter-relative "1-1" + split-column TOC
        gbt_toc = [(1, "目 次")] + [(1, f"1.{i} 条款{i} 1-{i}") for i in range(1, 25)]
        bilingual = [(2, "目录 / Contents")] + [(2, f"第{i}章 标题{i}") for i in range(1, 22)]
        split_col = [(3, x) for i in range(1, 16) for x in (f"1.{i} 术语{i}", str(i * 2))]
        # Prose page with "Contents" heading must NOT pollute page 5 pin list
        prose_p4 = [(4, "Contents"), (4, "This section explains the package contents in detail over several long sentences that exceed thirty characters easily.")]
        pins_p5 = [(5, f"PIN_GPIO_{i:02d}") for i in range(25)]
        self.assertEqual(detect_toc_pages(gbt_toc + bilingual + split_col + prose_p4 + pins_p5), {1, 2, 3})

        # Brochure with "Contents" + 2 Parts + 7 spec lines must NOT be wiped by chapter_toc
        brochure = [
            (1, "Contents"),
            (1, "Part 1: Main Controller Unit"),
            (1, "Part 2: Power Supply Module"),
            (1, "Key Specifications"),
            (1, "Input voltage: 24V DC"),
            (1, "Max current: 5A"),
            (1, "Operating temp: -20 to 70C"),
            (1, "Dimensions: 120 x 80 x 35 mm"),
            (1, "Weight: 420 g"),
            (1, "Protection rating: IP65"),
        ]
        self.assertEqual(explicit_toc_entries(brochure), set())

    def test_diagnose_heading_only_matches_b14_and_embedding_context_dedups_prefixes(self) -> None:
        from kb_pipeline.chunking.diagnose import _heading_only as diag_heading_only

        body_block = ParsedBlock(parser="m", parser_profile="p", doc_type="pdf", block_type="text",
                                 text="1. Open settings panel", block_id="b1", metadata={})
        self.assertFalse(diag_heading_only("1. Open settings panel", body_block))
        title_block = ParsedBlock(parser="m", parser_profile="p", doc_type="pdf", block_type="title",
                                  text="1.2 Setup", block_id="t1", metadata={})
        self.assertTrue(diag_heading_only("1.2 Setup", title_block))

        raw_blocks = [
            ParsedBlock(parser="m", parser_profile="p", doc_type="pdf", block_type="title",
                        text="第一章 总则", block_id="t1", page_idx=1, metadata={"section_path": ["第一章 总则"]}),
            ParsedBlock(parser="m", parser_profile="p", doc_type="pdf", block_type="title",
                        text="1.1 范围", block_id="t2", page_idx=1, metadata={"section_path": ["第一章 总则", "1.1 范围"]}),
            ParsedBlock(parser="m", parser_profile="p", doc_type="pdf", block_type="text",
                        text="正文一。", block_id="b1", page_idx=1, metadata={"section_path": ["第一章 总则", "1.1 范围"]}),
            ParsedBlock(parser="m", parser_profile="p", doc_type="pdf", block_type="title",
                        text="第二章 规范", block_id="t3", page_idx=1, metadata={"section_path": ["第二章 规范"]}),
            ParsedBlock(parser="m", parser_profile="p", doc_type="pdf", block_type="text",
                        text="正文二。", block_id="b2", page_idx=1, metadata={"section_path": ["第二章 规范"]}),
        ]
        chunks = blocks_to_chunks(kb_id="k", file_key=1, content_version="v", parser_profile="p",
                                  blocks=merge_mineru_text_blocks(raw_blocks, target_tokens=400),
                                  max_tokens=400, overlap_tokens=80)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(embedding_context_for(chunks[0], doc_name="doc.pdf"),
                         "doc.pdf > 第一章 总则 > 1.1 范围 / 第二章 规范")
