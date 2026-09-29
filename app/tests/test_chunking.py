"""Chunking layer: merging, headings, table pieces, diagnostics, token budget."""
from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from kb_pipeline import db
from kb_pipeline.models import ParsedBlock, SourceFile
from kb_pipeline.parsers.common import parser_profile_for_path

from _support import _CodexAudit20260906TestsSupport, _chunk, _local_file, _mb, _pin_tokenizer, _repo_file


class ChunkerRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        # token-budget assertions must run on a fixed tokenizer, otherwise the same assertion tests two different
        # things on machines with / without a tiktoken cache
        if not _pin_tokenizer():
            self.skipTest("tiktoken o200k_base 不可用(离线且无缓存)")

    def test_overlap_tail_never_accumulates_past_max_tokens(self) -> None:
        from kb_pipeline.chunking.chunker import chunk_text
        from kb_pipeline.utils import count_tokens

        # The overlap tail always keeps at least one sentence however large it
        # is. Before the guard, the main loop appended the next sentence to
        # that tail without rechecking, producing chunks near 2x the limit
        # (observed 720 against 400 in the real corpus).
        sentence = "one two three four five six seven eight nine ten eleven twelve."
        t = count_tokens(sentence)
        max_tokens = int(t * 1.3)
        text = " ".join([sentence] * 12)
        chunks = chunk_text(text, max_tokens, overlap_tokens=t - 1)
        self.assertGreater(len(chunks), 1)
        worst = max(count_tokens(c) for c in chunks)
        # the newline join adds a token or two that current_tokens ignores
        self.assertLessEqual(worst, max_tokens + 5)

    def test_slide_blocks_are_split_like_text(self) -> None:
        from kb_pipeline.chunking.chunker import blocks_to_chunks
        from kb_pipeline.models import ParsedBlock
        from kb_pipeline.utils import count_tokens

        long_text = "。".join(f"这是幻灯片正文里比较长的一句话第{i}句" for i in range(80)) + "。"
        slide = ParsedBlock(
            parser="mineru", parser_profile="p", doc_type="pptx",
            block_type="slide", text=long_text, block_id="s-0001", slide_idx=3,
        )
        image = ParsedBlock(
            parser="mineru", parser_profile="p", doc_type="pptx",
            block_type="image", text="VISUAL SUMMARY: 架构图", block_id="i-0001", slide_idx=3,
        )
        chunks = blocks_to_chunks(
            kb_id="k", file_key=1, content_version="v", parser_profile="p",
            blocks=[slide, image], max_tokens=100, overlap_tokens=20,
        )
        slide_parts = [c for c in chunks if c.block.block_id == "s-0001"]
        image_parts = [c for c in chunks if c.block.block_id == "i-0001"]
        # slides used to bypass chunking entirely (single 725-token chunks)
        self.assertGreater(len(slide_parts), 1)
        for chunk in slide_parts:
            self.assertLessEqual(count_tokens(chunk.text), 105)
        # figures stay one chunk: their VLM-derived text is bounded
        self.assertEqual(len(image_parts), 1)


class ChunkLimitBoundTests(unittest.TestCase):
    """Both bounds follow the embedding service's max_model_len; there is no hard-coded cap any more.

    After max_model_len went from 4096 to 8192 on 2026-08-22, the hard-coded 3200 in the old implementation's
    min(HARD_CAP_TOKENS, ...) swallowed the whole increase: the service got more capable while the configurable
    cap did not move an inch. It is now taken proportionally, so adjusting compose moves it.
    """

    def _limits(self, served):
        from unittest import mock

        from kb_pipeline import limits as limits_module

        payload = {"data": [{"max_model_len": served}]} if served else {"data": [{}]}
        with mock.patch.object(
            limits_module.requests, "get",
            return_value=SimpleNamespace(json=lambda: payload),
        ):
            return limits_module.chunk_limits("http://x/v1")

    def test_cap_tracks_the_service_context_window(self) -> None:
        self.assertEqual(self._limits(8192)["max_tokens_cap"], 6553)   # int(8192 * 0.8)
        self.assertEqual(self._limits(4096)["max_tokens_cap"], 3276)   # int(4096 * 0.8)
        # after the service grows the cap must grow with it, exactly what the old implementation could not do
        self.assertGreater(
            self._limits(8192)["max_tokens_cap"], self._limits(4096)["max_tokens_cap"]
        )

    def test_floor_is_128(self) -> None:
        self.assertEqual(self._limits(8192)["max_tokens_min"], 128)

    def test_overlap_cap_takes_the_tighter_of_the_two_bounds(self) -> None:
        from kb_pipeline import limits as limits_module

        lim = self._limits(8192)
        # normal range: "no more than half" is the tighter bound
        self.assertEqual(limits_module.overlap_cap(400, lim), 200)
        self.assertEqual(limits_module.overlap_cap(3200, lim), 1600)
        # as max_tokens approaches the model limit, "remaining model budget" becomes the tighter one
        self.assertEqual(limits_module.overlap_cap(6553, lim), 1639)  # 8192-6553 < 6553//2

    def test_overlap_validation_uses_the_dynamic_bound(self) -> None:
        from kb_pipeline import limits as limits_module

        lim = self._limits(8192)
        self.assertEqual(limits_module.validate_chunk_config(400, 199, lim), [])
        self.assertTrue(limits_module.validate_chunk_config(400, 200, lim))    # exactly at the bound: rejected
        self.assertEqual(limits_module.validate_chunk_config(6553, 1638, lim), [])
        # 1700 < 6553/2 yet > 8192-6553: the old "less than half" would let it through, the new bound stops it
        self.assertTrue(limits_module.validate_chunk_config(6553, 1700, lim))

    def test_max_tokens_error_does_not_also_report_a_bogus_overlap_bound(self) -> None:
        from kb_pipeline import limits as limits_module

        # when max_tokens itself is out of range, deriving the overlap bound from it only yields a misleading number
        errors = limits_module.validate_chunk_config(99999, 80, self._limits(8192))
        self.assertEqual(len(errors), 1)
        self.assertIn("max_tokens", errors[0])

    def test_unreachable_service_falls_back_without_crashing(self) -> None:
        from unittest import mock

        from kb_pipeline import limits as limits_module

        with mock.patch.object(limits_module.requests, "get", side_effect=OSError("down")):
            lim = limits_module.chunk_limits("http://x/v1")
        self.assertIsNone(lim["embedding_max_model_len"])
        self.assertFalse(lim["live"])
        self.assertEqual(lim["effective_max_model_len"], 4096)
        self.assertEqual(lim["max_tokens_cap"], 3276)
        self.assertIn("falling back", limits_module.validate_chunk_config(9999, 80, lim)[0])


class ChunkDiagnosticsTests(unittest.TestCase):
    """Chunk acceptance: it only warns, never blocks, but the warnings must be accurate: short table/image
    chunks must not be taken for fragments, and real fragments must not hide behind the mean."""

    @staticmethod
    def _chunk(tokens: int, block_type: str = "text", section=()):
        from kb_pipeline.models import UnifiedChunk

        # with a full stop: the diagnostics treat a "single short line without punctuation" fragment as an orphan
        # heading, so the synthetic text has to look like body text
        block = ParsedBlock(parser="t", parser_profile="p", doc_type="d", block_type=block_type,
                            text="x" * tokens + "。", block_id=f"b{tokens}",
                            metadata={"section_path": list(section)})
        return UnifiedChunk(chunk_uid=f"u{tokens}", chunk_index=0, text=block.text, block=block,
                            token_count=tokens)

    def test_healthy_document_passes(self) -> None:
        from kb_pipeline.chunking.diagnose import chunk_diagnostics

        chunks = [self._chunk(t) for t in (380, 390, 360, 400, 120)]
        diag = chunk_diagnostics(chunks, max_tokens=400)
        self.assertTrue(diag["ok"], diag)
        self.assertEqual(diag["stats"]["chunks"], 5)
        self.assertEqual(diag["stats"]["tiny_threshold"], 50)
        self.assertEqual(diag["stats"]["over_count"], 0)

    def test_fragmented_document_is_flagged(self) -> None:
        from kb_pipeline.chunking.diagnose import chunk_diagnostics

        chunks = [self._chunk(t) for t in (12, 9, 15, 20, 380, 390)]
        diag = chunk_diagnostics(chunks, max_tokens=400)
        self.assertFalse(diag["ok"])
        self.assertEqual([r["key"] for r in diag["reasons"]], ["fragmented"])
        self.assertEqual(diag["stats"]["tiny_count"], 4)

    def test_table_rows_and_figures_are_exempt(self) -> None:
        """Tables are split per row, and the last row or two left over are short anyway, so they are not fragments;
        images are one per chunk, and their length is not bound by max_tokens."""
        from kb_pipeline.chunking.diagnose import chunk_diagnostics

        chunks = [self._chunk(t, "table") for t in (8, 9, 10, 11)] + [self._chunk(900, "image")]
        diag = chunk_diagnostics(chunks, max_tokens=400)
        self.assertTrue(diag["ok"], diag)
        self.assertEqual(diag["stats"]["tiny_count"], 0)
        self.assertEqual(diag["stats"]["over_count"], 0)
        self.assertEqual(diag["stats"]["by_block_type"]["table"]["chunks"], 4)

    def test_an_oversized_table_chunk_is_reported(self) -> None:
        """Table chunks keep to the budget too (rows and cells over the budget are split); an over-long table chunk
        used to be exempt as a whole class, so the console kept reporting the check as passed."""
        from kb_pipeline.chunking.diagnose import chunk_diagnostics

        chunks = [self._chunk(t, "table") for t in (380, 390, 5921)]
        diag = chunk_diagnostics(chunks, max_tokens=400)
        self.assertEqual([r["key"] for r in diag["reasons"]], ["oversized"])
        self.assertEqual(diag["stats"]["over_count"], 1)

    def test_never_filling_the_budget_is_flagged(self) -> None:
        from kb_pipeline.chunking.diagnose import chunk_diagnostics

        chunks = [self._chunk(t) for t in (60, 70, 65, 80, 75, 90)]
        diag = chunk_diagnostics(chunks, max_tokens=400)
        self.assertIn("all_tiny", [r["key"] for r in diag["reasons"]])

    def test_block_level_stats_and_log_line(self) -> None:
        from kb_pipeline.chunking.diagnose import chunk_diagnostics, summarize_line

        chunks = [self._chunk(380, section=("第一章",)), self._chunk(390, section=("第一章", "1.1"))]
        blocks = [c.block for c in chunks] + [
            ParsedBlock(parser="t", parser_profile="p", doc_type="d", block_type="title", text="第一章", block_id="h")]
        diag = chunk_diagnostics(chunks, max_tokens=400, blocks=blocks)
        self.assertEqual(diag["stats"]["blocks"], 3)
        self.assertEqual(diag["stats"]["headings"], 1)
        self.assertEqual(diag["stats"]["section_depths"], {"0": 1, "1": 1, "2": 1})
        line = summarize_line(diag)
        self.assertIn("verdict=ok", line)
        self.assertIn("chunks=2", line)

    def test_diag_persisted_and_surfaced_in_file_list(self) -> None:
        from unittest import mock

        from kb_pipeline import discovery
        from kb_server import service

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "资料").mkdir()
            state = Path(tmp) / "s.db"
            db.init_db(state)
            with db.connect(state) as con:
                src, _ = discovery.enroll(con, root, "资料")
                f = _local_file("a.txt")
                f = SourceFile(**{**f.__dict__, "kb_id": src.kb_id, "collection": src.collection})
                db.upsert_file(con, f)
                row = db.get_file_by_path(con, src.kb_id, f.source_path)
                diag = {"ok": False, "reasons": [{"key": "fragmented", "message": "碎片过多"}],
                        "stats": {"chunks": 7, "tokens_mean": 33.0}}
                db.mark_file_indexed(con, str(row["file_id"]), f.content_version, "p", chunk_diag=diag)
                stored = json.loads(db.get_file_by_id(con, str(row["file_id"]))["chunk_diag_json"])
                self.assertEqual(stored, diag)
                con.commit()
            with mock.patch.object(service, "settings", lambda: SimpleNamespace(state_db=state)):
                files = service.kb_files(src.kb_id)
            self.assertEqual(files[0]["dot"], "green")
            self.assertEqual(files[0]["chunk_diag"], {"ok": False, "chunks": 7, "tokens_mean": 33.0,
                                                      "reasons": ["碎片过多"]})

    def test_console_previews_stored_chunks_in_a_drawer(self) -> None:
        """2026-09-06 redesign: the chunk preview is no longer a tab and does not re-chunk; the "chunk preview"
        in the file table shows the chunks currently stored, in a drawer on the right, and "re-parse" is the way
        to see the effect of new rules. The workspace has only three tabs, config / files / graph, with config
        first; the "enable knowledge base" switch sits on the right of the title row of the knowledge-base
        section on the config page, at the same level as "enable knowledge graph"."""
        html = _repo_file("app/kb_server/static/index.html")
        js = _repo_file("app/kb_server/static/app.js")
        self.assertEqual(re.findall(r'data-tab="(\w+)"', html), ["config", "files", "graph"])
        self.assertIn('<button data-tab="graph">图谱预览</button>', html)
        self.assertIn('<button data-tab="config">配置管理</button>', html)
        self.assertIn('<button data-tab="files">文件管理<span class="n" id="tabn-files"></span></button>', html)
        # the four buttons build / merge / pause / delete go back to the knowledge-graph section of config
        # management; the graph preview tab keeps only the status card and the preview
        graph_tab = html.split('id="tab-graph"', 1)[1].split('id="tab-config"', 1)[0]
        config_tab = html.split('id="tab-config"', 1)[1]
        for bid in ('id="cfg-gbuild"', 'id="cfg-gappend"', 'id="cfg-gpause"', 'id="cfg-gdelete"'):
            self.assertNotIn(bid, graph_tab, bid)
            self.assertIn(bid, config_tab, bid)
        for gone in ('id="tab-chunks"', 'id="tab-jobs"', 'id="pv-run"', 'id="cfg-preview"', 'id="jb-list"'):
            self.assertNotIn(gone, html, gone)
        self.assertIn('id="side-win"', html)
        self.assertIn('id="side-body"', html)
        before_card, card = html.split('id="config-card"', 1)
        self.assertNotIn('id="sel-toggle"', before_card)               # no longer in the workspace title row
        self.assertIn('id="sel-toggle"', card.split('id="cfg-fields"', 1)[0])   # in the "knowledge base" section title row
        self.assertIn('const TABS = ["config", "files", "graph"]', js)
        self.assertIn("/files/${encodeURIComponent(fileId)}/chunks", js)
        self.assertIn("function openChunkDrawer(", js)
        self.assertIn('>${t("切块预览")}</button>', js)     # copy goes through t(): the Chinese original is still the key
        self.assertIn('>${t("重新解析")}</button>', js)
        self.assertNotIn(">任务</button>", js)
        self.assertNotIn("chunk_preview", js)
        # a file being re-parsed: the dot at the row start pulses yellow and the status column shows the stage
        # (the server-side dot stays green throughout, so on its own it does not reveal a run in progress)
        self.assertIn('<td><span class="dot ${fileDot(f)}"></span></td>', js)
        cell = js.split("function fileStatusCell(f) {", 1)[1].split("\n}\n", 1)[0]
        self.assertLess(cell.index('f.job_status === "running"'), cell.index('f.dot === "green"'))
        dot = js.split("function fileDot(f) {", 1)[1].split("\n}\n", 1)[0]
        self.assertIn('return "yellow pulse"', dot)

    def test_sidebar_draws_graph_progress_like_parse_progress(self) -> None:
        """One progress bar under the KB label in the sidebar: it shows parsing while a parse runs, otherwise the
        graph build while that runs (stage + percentage), with the percentage from the same stage interpolation
        the graph page uses."""
        js = _repo_file("app/kb_server/static/app.js")
        nav = js.split("function renderKbNav()", 1)[1].split("\nfunction ", 1)[0]
        self.assertIn("graphPctFloor(kb.kb_id, gb.started_at, graphStagePct(gb.stage, gb.stage_weights))", nav)
        self.assertIn('class="mini${busy ? "" : " graph"}"', nav)
        self.assertNotIn('"建图中"', nav)
        # the three kinds of in-progress work share one bar + short text; the stage name only goes into the tooltip
        for needle in ('text = t("解析 {0}/{1} · {2}%", p.done, p.total, p.pct)', 'text = t("建图 · {0}%", bar)', "tip = shortStage("):
            self.assertIn(needle, nav, needle)
        # single-file / whole-KB "re-parse": the file stays indexed throughout, so counted over the whole KB it
        # reads "113/114 · 99%"; now it is counted per round
        self.assertIn("reparseRound(kb.kb_id, reparsing)", nav)
        self.assertIn('t("重新解析") + (round > 1', nav)
        # the graph page's status card draws no progress bar / percentage any more: progress is only in the sidebar
        panel = js.split("function renderGraphPanel()", 1)[1].split("/* ── graph preview", 1)[0]
        self.assertNotIn("pbar", panel)
        self.assertNotIn("${pct}%", panel)
        # the graph build history table is gone: size / merge numbers are stated on the status card instead
        html = _repo_file("app/kb_server/static/index.html")
        self.assertNotIn('id="gb-list"', html)
        self.assertNotIn("建图记录", html)
        self.assertNotIn("renderGraphBuilds(", js)
        # a built graph: entity / relation / unit cards + an "updated at" line; no version, mode or merge count
        self.assertIn('class="pv-stats gstats"', panel)
        self.assertIn('${t("更新于")} ${esc(fmtTs(b.counts_at))}', panel)
        self.assertNotIn("graph_version", panel)
        self.assertNotIn("归并", panel)
        svc = _repo_file("app/kb_server/service.py")
        self.assertIn('info["counts"] = summary.get("counts")', svc)
        self.assertIn('info["counts_at"] = done_row["finished_at"] or done_row["started_at"]', svc)
        css = _repo_file("app/kb_server/static/index.html")
        self.assertIn(".kb-row-sub .mini{flex:1 1 56px;min-width:56px", css)   # the bar cannot be squeezed out by the text
        self.assertIn("kb.jobs_waiting > 0", nav)               # the queued stage must show too
        self.assertIn("function reparseRound(", js)
        self.assertIn("function shortStage(", js)


class HeadingInferenceTests(unittest.TestCase):
    """Headings MinerU failed to tag are recognised from their layout; numbers and list items must be kept
    out, otherwise section_path gets polluted and chunk boundaries land in the wrong place."""

    def test_levels(self) -> None:
        from kb_pipeline.headings import infer_heading_level as lv

        self.assertEqual(lv("第一章 总则"), 1)
        self.assertEqual(lv("第 3 章：安装与配置"), 1)
        self.assertEqual(lv("第十二节 时序参数"), 2)
        self.assertEqual(lv("Chapter 4: Timing Characteristics"), 1)
        self.assertEqual(lv("Part II"), 1)
        self.assertEqual(lv("Section 2.1"), 2)
        self.assertEqual(lv("附录 A 引脚定义"), 1)
        self.assertEqual(lv("1.2.3 电气特性"), 3)
        self.assertEqual(lv("2. 概述"), 1)
        self.assertEqual(lv("IV. Results"), 1)
        self.assertEqual(lv("§3.3 闭区间上连续函数的基本性质"), 2)
        self.assertEqual(lv("§ 2 序列极限"), 1)
        self.assertEqual(lv("1.2概述"), 2)                 # Chinese typesetting often omits the space
        self.assertEqual(lv("第一章总则"), 1)
        self.assertEqual(lv("第二节安装与配置"), 2)

    def test_numbering_depth_overrides_a_flattened_parser_level(self) -> None:
        """MinerU tags a whole book's headings as level 1; the depth carried by the numbering is more reliable."""
        from kb_pipeline.headings import resolve_heading_level as rl

        self.assertEqual(rl(1, "5.4.3 函数的凹凸性"), (3, False))
        self.assertEqual(rl(1, "第一章 函数"), (1, False))        # inference no deeper than the parser: keep the parser's level
        self.assertEqual(rl(2, "1.2 概述"), (2, False))
        self.assertEqual(rl(None, "1.2 概述"), (2, True))        # the parser did not tag it: inferred
        self.assertEqual(rl(None, "普通正文一句。"), (None, False))
        self.assertEqual(rl(None, ""), (None, False))

    def test_non_headings(self) -> None:
        from kb_pipeline.headings import infer_heading_level as lv

        for text in (
            "1.5 V typical",                       # number + unit
            "3.3 V",
            "4.7 kΩ pull-up",
            "1. 典型值仅供参考，并未得以保证，也未经过测试。",   # numbered list item
            "1. Turn on the power.",
            "第3章的说明请参见附录，见下",             # a punctuated sentence, not a heading
            "第3章的说明请参见附录并对照第4章的接线图",   # too long for the run-together form
            "1.2 " + "很长的标题" * 20,             # too long
            "第一章\n正文接着写",                    # multi-line
            "",
        ):
            self.assertIsNone(lv(text), text)

    def test_page_footers_are_noise(self) -> None:
        from kb_pipeline.headings import is_page_footer
        from kb_pipeline.parsers.pdf_enhanced import is_noise_block

        for text in ("Page 3 of 12", "Seite 4 von 9", "第 5 页", "第5页/共20页", "- 12 -", "3 / 20", "页 7"):
            self.assertTrue(is_page_footer(text), text)
        for text in ("3/4 英寸接口", "Page layout", "第 5 页的内容如下", "1/2"):   # 1/2 may be a formula fragment
            self.assertFalse(is_page_footer(text), text)
        block = ParsedBlock(parser="mineru", parser_profile="p", doc_type="pdf", block_type="text",
                            text="Page 3 of 12", block_id="b", metadata={"source_type": "text"})
        self.assertTrue(is_noise_block(block))
        block.text = "正文"
        self.assertFalse(is_noise_block(block))

    def test_parsers_wire_the_inference_and_bump_profiles(self) -> None:
        # since 2026-09-05 both parsers use HeadingResolver (per document: repeated labels demoted, levels inherited)
        for name in ("mineru_pdf", "mineru_docx"):
            src = _repo_file(f"app/kb_pipeline/parsers/{name}.py")
            self.assertIn("resolver.resolve(heading_level(item)", src, name)
            self.assertIn("HeadingResolver(", src, name)
            self.assertIn("heading_inferred", src, name)
        self.assertEqual(parser_profile_for_path(Path("a.pdf")), "pdf-mineru-table-vlm-v15")     # 09-30 heading levels, headings at a block's end
        self.assertEqual(parser_profile_for_path(Path("a.docx")), "docx-mineru-ooxml-vlm-v10")
        self.assertEqual(parser_profile_for_path(Path("a.py")), "py-symbols-v2")                 # 09-30 line splitting
        self.assertEqual(parser_profile_for_path(Path("a.ts")), "code-symbols-v2")

    def test_merged_blocks_remember_their_heading_lines(self) -> None:
        from kb_pipeline.parsers.pdf_enhanced import merge_mineru_text_blocks

        def blk(i, text, kind="text"):
            return ParsedBlock(parser="mineru", parser_profile="p", doc_type="pdf", block_type=kind,
                               text=text, block_id=f"b{i}", metadata={"source_type": kind, "section_path": []})

        blocks = [blk(1, "开头一段。"), blk(2, "功能描述", "title"), blk(3, "正文第二段。"),
                  blk(4, "Page 2 of 9"), blk(5, "1.2 时序", "title"), blk(6, "尾段。")]
        merged = merge_mineru_text_blocks(blocks, target_tokens=800)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].metadata["heading_lines"], ["功能描述", "1.2 时序"])
        self.assertNotIn("Page 2 of 9", merged[0].text)   # the page-number block is dropped as noise


class HeadingAwareChunkingTests(unittest.TestCase):
    """Headings are structural cut points: past the halfway mark the chunk breaks before the heading with no
    overlap; when a break is forced by length, a heading at the chunk tail moves to the start of the next
    chunk."""

    @staticmethod
    def _para(n: int, word: str = "word") -> str:
        return " ".join([word] * n) + "."

    def test_heading_starts_a_fresh_chunk_without_overlap(self) -> None:
        from kb_pipeline.chunking.chunker import chunk_text

        text = self._para(250) + "\n\n1.2 Second Section\n\n" + self._para(100, "next")
        chunks = chunk_text(text, 400, 80)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[1].startswith("1.2 Second Section\n"), chunks[1][:60])
        self.assertNotIn("word", chunks[1])          # no tail of the previous section leaked in

    def test_small_sections_are_packed_together(self) -> None:
        from kb_pipeline.chunking.chunker import chunk_text

        text = self._para(60) + "\n\n1.2 Second Section\n\n" + self._para(60, "next")
        self.assertEqual(len(chunk_text(text, 400, 80)), 1)

    def test_trailing_heading_moves_to_the_next_chunk(self) -> None:
        from kb_pipeline.chunking.chunker import chunk_text

        # the chunk is only 1/4 full when the heading arrives, so it is merged in first; the big paragraph right
        # after does not fit → break by length, and the heading must follow the paragraph, not stay at the tail
        text = self._para(100) + "\n\n功能描述\n\n" + self._para(350, "next")
        chunks = chunk_text(text, 400, 80, headings=frozenset({"功能描述"}))
        self.assertEqual(len(chunks), 2)
        self.assertFalse(chunks[0].endswith("功能描述"))
        self.assertTrue(chunks[1].startswith("功能描述\n"))

    def test_known_headings_come_from_the_block(self) -> None:
        from kb_pipeline.chunking.chunker import blocks_to_chunks

        text = self._para(250) + "\n\n功能描述\n\n" + self._para(100, "next")
        block = ParsedBlock(parser="m", parser_profile="p", doc_type="pdf", block_type="text", text=text,
                            block_id="b1", metadata={"heading_lines": ["功能描述"]})
        chunks = blocks_to_chunks(kb_id="k", file_key=1, content_version="v", parser_profile="p",
                                  blocks=[block], max_tokens=400, overlap_tokens=80)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[1].text.startswith("功能描述"))

    def test_heading_before_an_oversized_unit_is_not_dropped(self) -> None:
        """A heading moved off the end of a chunk, and the unit after it is over the budget and split on its own:
        the moved heading used to be picked up by nobody and ended up in no chunk at all."""
        from kb_pipeline.chunking.chunker import blocks_to_chunks, chunk_text

        long_sentence = "没有句读的长段落" * 400
        fence = "```yaml\n" + "\n".join(f"key{i}: value{i}" for i in range(400)) + "\n```"
        for heading, unit, known in (("第3章 部署说明", long_sentence, ()), ("配置示例", fence, ("配置示例",))):
            text = f"正文第一句。正文第二句。\n\n{heading}\n\n{unit}"
            for overlap in (0, 80):
                parts = chunk_text(text, 600, overlap, headings=frozenset(known))
                self.assertEqual(sum(p.count(heading) for p in parts), 1, (heading, overlap))
                self.assertLess(parts.index(heading), len(parts) - 1)            # before the pieces of the over-long unit
            block = ParsedBlock(parser="m", parser_profile="p", doc_type="pdf", block_type="text", text=text,
                                block_id="b1", metadata={"heading_lines": list(known)})
            chunks = blocks_to_chunks(kb_id="k", file_key=1, content_version="v", parser_profile="p",
                                      blocks=[block], max_tokens=600, overlap_tokens=80)
            first = chunks[0].text
            self.assertIn(f"{heading}\n{unit[:20]}", first)                      # after the fragment fallback the heading leads the content below it

    def test_heading_at_the_end_of_the_text_is_not_dropped(self) -> None:
        """The block text ends with a heading (a merged block filled up right after the heading): the final flush
        used to drop the heading as well."""
        from kb_pipeline.chunking.chunker import blocks_to_chunks, chunk_text

        text = "正文第一句。正文第二句。\n\n第3章 部署说明"
        self.assertEqual(chunk_text(text, 600, 80), ["正文第一句。\n正文第二句。", "第3章 部署说明"])
        self.assertEqual(chunk_text("第3章 部署说明", 600, 80), ["第3章 部署说明"])
        long_text = self._para(350) + "\n\n" + self._para(300, "more") + "\n\n功能描述"
        parts = chunk_text(long_text, 400, 80, headings=frozenset({"功能描述"}))
        self.assertEqual(parts[-1], "功能描述")
        self.assertEqual(sum(p.count("功能描述") for p in parts), 1)
        # A stranded heading is left to the fragment fallback: right before a table block it becomes the table's
        # TITLE, otherwise it merges back into the previous chunk of the same block
        body, grid = self._para(200), "| a | b |\n| --- | --- |\n| 1 | 2 |"
        tail = ParsedBlock(parser="m", parser_profile="p", doc_type="pdf", block_type="text",
                           text=f"{body}\n\n第3章 部署说明", block_id="b1")
        table = ParsedBlock(parser="m", parser_profile="p", doc_type="pdf", block_type="table", text=grid,
                            table_markdown=grid, block_id="t1")
        other = ParsedBlock(parser="m", parser_profile="p", doc_type="pdf", block_type="text", text=self._para(200, "next"),
                            block_id="b2", metadata={"section_path": ["第3章 部署说明"]})
        chunks = lambda blocks: [c.text for c in blocks_to_chunks(
            kb_id="k", file_key=1, content_version="v", parser_profile="p", blocks=blocks, max_tokens=600, overlap_tokens=80)]
        self.assertEqual(chunks([tail, table]), [body, f"TITLE: 第3章 部署说明\n{grid}"])
        self.assertEqual(chunks([tail, other]), [f"{body}\n第3章 部署说明", other.text])


class HeadingRuleTests(unittest.TestCase):
    def test_dashed_numbering_is_a_level_two_heading(self) -> None:
        from kb_pipeline.headings import infer_heading_level

        self.assertEqual(infer_heading_level("1-3 SQL概要"), 2)
        self.assertEqual(infer_heading_level("2-1 数据库的种类"), 2)
        self.assertIsNone(infer_heading_level("3-2 V 供电"))                    # number + unit
        self.assertIsNone(infer_heading_level("1-2 月完成安装,3 月验收。"))     # a sentence

    def test_clean_heading_text_strips_markup(self) -> None:
        from kb_pipeline.headings import clean_heading_text

        self.assertEqual(clean_heading_text("**2.1 Windows 安装**"), "2.1 Windows 安装")
        self.assertEqual(clean_heading_text("==📅 2026-01-15== 周会"), "📅 2026-01-15 周会")
        self.assertEqual(clean_heading_text("`config.yaml` 说明:"), "config.yaml 说明")

    def test_resolver_demotes_labels_and_list_items_and_inherits_levels(self) -> None:
        from kb_pipeline.headings import HeadingResolver

        titles = ["KEYWORD", "1-3 SQL概要", "标准SQL", "KEYWORD", "- 加粗的列表项", "KEYWORD", "注释:"]
        r = HeadingResolver(titles)
        self.assertEqual(r.resolve(1, "KEYWORD"), (None, False))          # a sidebar label that appears 3 times
        self.assertEqual(r.resolve(1, "1-3 SQL概要")[0], 2)                # inferred from the numbering
        self.assertEqual(r.resolve(1, "标准SQL")[0], 3)                    # hangs one level below the numbered heading
        self.assertEqual(r.resolve(1, "- 加粗的列表项"), (None, False))
        self.assertEqual(r.resolve(1, "注释:"), (None, False))
        self.assertEqual(r.resolve(None, "普通正文一句话。"), (None, False))
        self.assertEqual(r.resolve(2, "1.2.1 细节")[0], 3)                # unchanged when the docx level and the numbering agree

    @staticmethod
    def _walk(seq, **kw):
        """[(text, parser level)] -> [(final level, section path)], with levels and paths walked the way the parsers
        do it."""
        from kb_pipeline.headings import HeadingResolver
        from kb_pipeline.parsers.common import SectionTracker

        resolver = HeadingResolver((t for t, lv in seq if lv), **kw)
        sections = SectionTracker()
        out = []
        for text, parser_level in seq:
            level, _ = resolver.resolve(parser_level, text)
            sections.observe(level, text)
            out.append((level, list(sections.path)))
        return out

    def test_integer_chapter_headings_are_numbered_headings(self) -> None:
        """An integer chapter number followed by a space ("4 Technical requirements") used not to count as numbered:
        after a "3.1" it was demoted to level 3 as an unnumbered heading, the next "4.1" popped it off again, and
        all the subsections of the chapter hung under the previous chapter."""
        got = self._walk([("3 术语", 1), ("3.1 定义", 1), ("4 技术要求", 1), ("4.1 电气", 1), ("4.1.1 直流", 1),
                          ("试验条件", 1), ("5 试验方法", 1)])
        self.assertEqual([lv for lv, _ in got], [1, 2, 1, 2, 3, 4, 1])
        self.assertEqual(got[2][1], ["4 技术要求"])
        self.assertEqual(got[3][1], ["4 技术要求", "4.1 电气"])
        self.assertEqual(got[5][1], ["4 技术要求", "4.1 电气", "4.1.1 直流", "试验条件"])      # an unnumbered heading still hangs under the numbered one
        self.assertEqual(got[6][1], ["5 试验方法"])
        # A document with integer chapter numbers only: every chapter is level 1 and unnumbered subheadings hang
        # under their chapter. A chapter name that opens with a one- or two-letter word or contains a comma is fine
        got = self._walk([("Abstract", 1), ("1 Introduction", 1), ("2 A Survey of Methods", 1), ("Datasets", 1),
                          ("3 Results, Analysis and Discussion", 1), ("4 UI 设计", 1)])
        self.assertEqual([lv for lv, _ in got], [1, 1, 1, 2, 1, 1])
        self.assertEqual(got[3][1], ["2 A Survey of Methods", "Datasets"])

    def test_integer_chapter_numbering_survives_gaps(self) -> None:
        """A chapter number that does not connect is handled as an unnumbered heading, but it must not throw off
        the chapters after it: one unmarked chapter still connects; after a longer break, or when the numbering
        starts over (several articles bound together), the sequence picks up again from the next chapter."""
        got = self._walk([("1 范围", 1), ("2 规范性引用文件", 1), ("4 技术要求", 1), ("5 试验方法", 1)])       # chapter 3 is not marked
        self.assertEqual([lv for lv, _ in got], [1, 1, 1, 1])
        got = self._walk([("1 范围", 1), ("2 引用", 1), ("5 试验方法", 1), ("6 检验规则", 1), ("7 标志", 1)])
        self.assertEqual([lv for lv, _ in got], [1, 1, 2, 1, 1])
        got = self._walk([("1 Introduction", 1), ("2 Method", 1), ("1 Introduction", 1), ("2 Background", 1), ("3 Approach", 1)])
        self.assertEqual([lv for lv, _ in got], [1, 1, 2, 1, 1])
        self.assertEqual(got[4][1], ["3 Approach"])

    def test_integer_heading_must_be_marked_by_the_parser_and_continue_the_numbering(self) -> None:
        # A heading that opens with a quantity (its chapter number does not connect) stays an unnumbered heading
        # and does not displace the chapter it is in
        got = self._walk([("2 产品", 1), ("2.1 产品线", 1), ("8 位微控制器", 1), ("2.2 封装", 1), ("3 订购信息", 1)])
        self.assertEqual([lv for lv, _ in got], [1, 2, 3, 2, 1])
        self.assertEqual(got[3][1], ["2 产品", "2.2 封装"])
        got = self._walk([("3 术语", 1), ("3.1 定义", 1), ("1 级能效", 1), ("3.2 缩略语", 1)])          # numbering is under way, so 1 is not a fresh start
        self.assertEqual(got[3][1], ["3 术语", "3.2 缩略语"])
        # Body text the parser did not mark as a heading is not recovered; "5 V" is a value
        got = self._walk([("4.2 供电", 1), ("5 个步骤完成安装", None), ("5 V", 1), ("5 供电要求", 1)])
        self.assertEqual([lv for lv, _ in got], [2, None, 3, 1])

    def test_docx_headings_are_not_demoted_below_inferred_ones(self) -> None:
        """docx levels come from Word styles: once an unstyled "2.3 xxx" body paragraph had been recovered as a
        level-2 heading, every level-1 heading after it used to be demoted to level 3."""
        seq = [("概述", 1), ("2.3 没有标题样式的正文段", None), ("安装", 1), ("配置", 2), ("1.2.1 细节", 2),
               ("- 加粗的列表项", 1), ("升级", 1)]
        got = self._walk(seq, trust_parser_levels=True)
        self.assertEqual([lv for lv, _ in got], [1, 2, 1, 2, 3, None, 1])
        self.assertEqual(got[4][1], ["安装", "配置", "1.2.1 细节"])
        self.assertEqual(got[6][1], ["升级"])
        self.assertEqual([lv for lv, _ in self._walk(seq)], [1, 2, 3, 3, 3, None, 4])       # unchanged for PDF
        # Only the document title carries a style and the chapters are typed by hand: the chapters are still
        # recovered from the layout; when every style is level 1, the depth of the numbering still counts
        got = self._walk([("XX 系统设计说明书", 1), ("第一章 总则", None), ("1.1 范围", None)], trust_parser_levels=True)
        self.assertEqual([lv for lv, _ in got], [1, 1, 2])
        got = self._walk([("1 概述", 1), ("1.1 背景", 1), ("1.1.1 细节", 1)], trust_parser_levels=True)
        self.assertEqual([lv for lv, _ in got], [1, 2, 3])

    def test_section_paths_in_parsed_blocks(self) -> None:
        """Blocks from the two parsers: PDF integer chapter numbers are not demoted; docx style headings are not
        pushed down by numbered headings recovered by inference."""
        from kb_pipeline.parsers import mineru_docx, mineru_pdf

        def item(text, level=None):
            return {"type": "text", "text": text, "page_idx": 0, **({"text_level": level} if level else {})}

        pdf = {"content_list": [item("3 术语", 1), item("3.1 定义", 1), item("定义的正文。"),
                                item("4 技术要求", 1), item("4.1 电气", 1), item("电气的正文。")]}
        docx = {"content_list": [item("概述", 1), item("2.3 没有标题样式的一段"), item("安装", 1), item("安装的正文。")]}
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(mineru_pdf, "call_mineru_sync", return_value=pdf):
                blocks = mineru_pdf.mineru_pdf_blocks(mineru_url="http://x", path=Path(tmp) / "a.pdf", cache_dir=Path(tmp) / "p")
            self.assertEqual(blocks[-1].metadata["section_path"], ["4 技术要求", "4.1 电气"])
            self.assertEqual([b.metadata.get("heading_level") for b in blocks], [1, 2, None, 1, 2, None])
            self.assertEqual({b.parser_profile for b in blocks}, {"mineru-3.4.4-pdf-v3"})     # the provenance tag no longer claims the pipeline backend
            with patch.object(mineru_docx, "call_mineru_sync", return_value=docx):
                blocks = mineru_docx.mineru_docx_blocks(mineru_url="http://x", path=Path(tmp) / "a.docx", cache_dir=Path(tmp) / "d")
            self.assertEqual([b.metadata.get("heading_level") for b in blocks], [1, 2, 1, None])
            self.assertEqual([b.metadata["section_path"] for b in blocks],
                             [["概述"], ["概述", "2.3 没有标题样式的一段"], ["安装"], ["安装"]])


class ChunkStructureTests(unittest.TestCase):
    """Chunk structure: table pieces carry the header, fenced code travels whole, fragment fallback, code
    grouped by class, decorative images folded into body text, embedding prefix."""

    @staticmethod
    def _chunks(blocks, max_tokens=120, overlap=0):
        from kb_pipeline.chunking.chunker import blocks_to_chunks
        return blocks_to_chunks(kb_id="kb", file_key=1, content_version="v", parser_profile="p",
                                blocks=blocks, max_tokens=max_tokens, overlap_tokens=overlap)

    def test_table_pieces_repeat_the_header(self) -> None:
        from kb_pipeline.chunking.chunker import split_table_text
        from kb_pipeline.utils import count_tokens

        rows = "\n".join(f"| 参数{i} | 这是第{i}行的说明文字,用来把表撑长 | {i * 3} mV |" for i in range(40))
        head = "TITLE: 表 3-1 电气特性\n| 参数 | 说明 | 数值 |\n| --- | --- | --- |"
        pieces = split_table_text(f"{head}\n{rows}", 120)
        self.assertGreater(len(pieces), 2)
        for piece in pieces:
            self.assertTrue(piece.startswith(head + "\n"), piece[:80])
            self.assertLessEqual(count_tokens(piece), 120)
        body = [l for p in pieces for l in p.splitlines()[3:]]
        self.assertEqual(body, rows.splitlines())          # no duplicated, no lost rows

    def test_native_table_prefix_is_repeated_per_piece(self) -> None:
        from kb_pipeline.chunking.chunker import split_table_text

        rows = "\n".join(f"R{i}: 值{i} | 说明{i} | {i}" for i in range(60))
        prefix = "SHEET: 清单\nROWS: 00001-00060\nHEADER: 名称 | 说明 | 数量"
        pieces = split_table_text(f"{prefix}\n{rows}", 100)
        self.assertGreater(len(pieces), 1)
        self.assertTrue(all(p.startswith(prefix + "\n") for p in pieces))

    def test_over_long_row_is_split_at_cell_boundaries(self) -> None:
        from kb_pipeline.chunking.chunker import split_table_text

        long_cell = "很长的描述" * 30
        text = "| 名称 | 描述 |\n| --- | --- |\n| A | 短 |\n| B | " + long_cell + " | " + long_cell + " |"
        pieces = split_table_text(text, 80)
        self.assertEqual(pieces[0], "| 名称 | 描述 |\n| --- | --- |\n| A | 短 |")
        # Every cell of a split row carries its own column name (a column without a name in the header is called
        # "column N"), and such chunks no longer carry the header rows
        self.assertTrue(all(p.startswith("名称: B | ") and "---" not in p for p in pieces[1:]), pieces[1:])
        for name in ("描述", "column 3"):
            self.assertEqual("".join(p.split(f"{name}: ", 1)[1] for p in pieces if f"{name}: " in p), long_cell)

    def test_code_fence_travels_whole_and_splits_by_line_when_too_big(self) -> None:
        from kb_pipeline.chunking.chunker import chunk_text

        fence = "```python\n" + "\n".join(f"x{i} = {i}" for i in range(6)) + "\n```"
        chunks = chunk_text(f"下面是示例。\n{fence}\n结束。", 200, 0)
        self.assertEqual(len(chunks), 1)
        self.assertIn(fence, chunks[0])
        big = "```python\n" + "\n".join(f"line_{i} = {i}" for i in range(200)) + "\n```"
        chunks = chunk_text(big, 120, 0)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertTrue(c.startswith("```python\n") and c.endswith("```"), c[:40])

    def test_tiny_text_pieces_are_glued_to_neighbours_in_the_same_section(self) -> None:
        blocks = [_mb("a", "第一段正文,足够长。" * 3, metadata={"section_path": ["第一章"]}),
                  _mb("b", "短句。", metadata={"section_path": ["第一章"]}),
                  _mb("c", "第三段正文,也足够长。" * 3, metadata={"section_path": ["第一章"]})]
        chunks = self._chunks(blocks, max_tokens=120)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[1].text.startswith("短句。"))
        self.assertEqual(chunks[1].chunk_uid, "kb:1:v:p:c:0")
        self.assertEqual([c.chunk_index for c in chunks], [0, 1])

    def test_heading_only_piece_before_a_table_becomes_its_title(self) -> None:
        table = "| a | b |\n| --- | --- |\n| 1 | 2 |"
        blocks = [_mb("p", "前面的正文段落,内容足够长,不会被当碎片。" * 2),
                  _mb("h", "直流电气特性"),
                  _mb("t", table, block_type="table", table_markdown=table)]
        chunks = self._chunks(blocks, max_tokens=120)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[1].text.startswith("TITLE: 直流电气特性\n| a | b |"))
        self.assertEqual(chunks[1].block.block_id, "t")

    def test_class_and_its_methods_are_grouped(self) -> None:
        def code(bid, kind, name, cls=None):
            sym = {"kind": kind, "name": name, "qualname": f"{cls}.{name}" if cls else name}
            if cls:
                sym["class"] = cls
            text = f"def {name}(self):\n    return {name!r}" if kind == "method" else f"class {name}:\n    ..."
            return _mb(bid, text, block_type="code", metadata={"symbol": sym})

        blocks = [code("c", "class", "Base"), code("m1", "method", "a", "Base"), code("m2", "method", "b", "Base"),
                  code("f", "function", "helper")]
        chunks = self._chunks(blocks, max_tokens=200)
        self.assertEqual([c.block.block_id for c in chunks], ["c", "f"])
        self.assertIn("def a(self)", chunks[0].text)
        self.assertIn("def b(self)", chunks[0].text)
        chunks = self._chunks(blocks, max_tokens=20)
        self.assertEqual(chunks[0].block.block_id, "c")
        self.assertGreaterEqual(len(chunks), 3)

    def test_decorative_images_fold_into_the_previous_chunk(self) -> None:
        blocks = [_mb("p", "正文段落,说明产品外观。" * 3),
                  _mb("img", "VISUAL SUMMARY: 产品外观照片", block_type="image", visual_summary="产品外观照片",
                      metadata={"decorative": True}),
                  _mb("img2", "VISUAL SUMMARY: 装饰背景", block_type="image", visual_summary="装饰背景",
                      metadata={"decorative": True}),
                  _mb("chart", "VISUAL SUMMARY: 销量柱状图", block_type="chart", visual_summary="销量柱状图")]
        chunks = self._chunks(blocks, max_tokens=120)
        self.assertEqual([c.block.block_id for c in chunks], ["p", "chart"])
        self.assertTrue(chunks[0].text.endswith("IMAGE: 产品外观照片\nIMAGE: 装饰背景"))

    def test_decorative_fold_keeps_the_adopted_lead_in_text(self) -> None:
        """Codex review F03: the short lines before an image (report date, testing lab) were absorbed by
        pdf_enhanced into the title of the seal image; the seal was then judged decorative and folded away, and
        the date was gone. The absorbed text must first flow back as body lines, followed by the description;
        with no preceding body text to attach to, it becomes a chunk of its own."""
        stamp = dict(block_type="image", visual_summary="红色圆形公章", metadata={"decorative": True, "adopted_lead_in": True},
                     title="报告日期:2023-11-06\n\n检测技术:基因芯片")
        chunks = self._chunks([_mb("p", "营养代谢基因检测报告。" * 3), _mb("seal", "VISUAL SUMMARY: 红色圆形公章", **stamp)], max_tokens=120)
        self.assertEqual([c.block.block_id for c in chunks], ["p"])
        self.assertTrue(chunks[0].text.endswith("报告日期:2023-11-06\n\n检测技术:基因芯片\nIMAGE: 红色圆形公章"), chunks[0].text)
        # the seal comes first, with no previous chunk: the text is kept as a chunk of its own, not dropped
        alone = self._chunks([_mb("seal", "VISUAL SUMMARY: 红色圆形公章", **stamp), _mb("p", "正文段落,说明检测项目与方法。" * 9)], max_tokens=120)
        self.assertEqual([c.block.block_id for c in alone], ["seal", "p"])
        self.assertIn("报告日期:2023-11-06", alone[0].text)
        # decorative images with no absorbed text are unchanged: one description line, dropped if nothing precedes
        plain = self._chunks([_mb("seal", "VISUAL SUMMARY: 装饰背景", block_type="image", visual_summary="装饰背景", metadata={"decorative": True}), _mb("p", "正文。")], max_tokens=120)
        self.assertEqual([c.block.block_id for c in plain], ["p"])

    def test_embedding_context_prefixes_doc_and_section(self) -> None:
        from kb_pipeline.chunking.chunker import embedding_context_for, embedding_input

        blocks = [_mb("p", "正文。", metadata={"section_path": ["第一章", "1.1 概述"]})]
        chunk = self._chunks(blocks, max_tokens=120)[0]
        self.assertIsNone(chunk.embedding_context)
        self.assertEqual(embedding_input(chunk), "正文。")
        chunk.embedding_context = embedding_context_for(chunk, doc_name="产品手册 > 第1卷")
        self.assertEqual(chunk.embedding_context, "产品手册 > 第1卷 > 第一章 > 1.1 概述")
        self.assertEqual(embedding_input(chunk), "产品手册 > 第1卷 > 第一章 > 1.1 概述\n正文。")

    def test_parse_job_embeds_with_context_and_stores_it(self) -> None:
        src = _repo_file("app/kb_pipeline/pipeline/parse_job.py")
        self.assertIn("embedder.embed([embedding_input(chunk) for chunk in chunks])", src)
        self.assertIn('"embedding_context": getattr(chunk, "embedding_context", None)', src)


class TableBudgetTests(unittest.TestCase):
    """Table chunks keep to the budget too: a row over the budget is split by cell, a cell over the budget by
    sentence and then by token, every piece carrying the row identity and the column name; when the header takes
    up most of the budget the column names travel with the cells instead of being repeated in every chunk; rows
    of a native table already expanded with column names are not labelled by position a second time in the
    chunker."""

    _chunks = staticmethod(ChunkStructureTests._chunks)

    def setUp(self) -> None:
        if not _pin_tokenizer():
            self.skipTest("tiktoken o200k_base unavailable (offline, no cache)")

    def test_native_rows_split_by_column_reach_the_index_under_their_own_names(self) -> None:
        """A row with a long cell over the budget and an empty cell before it: the native table used to split it
        into "name: value" first, then the chunker applied the header by position once more, so the chunk text read
        "Module: Module: Contacts | Level 1: Description: ...", an extra fragment held only the row identity, and
        the long cell's chunk was still over the budget."""
        from kb_pipeline.parsers.native_table import chunk_rows_to_blocks
        from kb_pipeline.utils import count_tokens

        header = ["模块", "一级", "二级", "功能描述", "v6.2", "公网", "私有云"]
        long_cell = "在云文档中使用当前用户组。" * 150
        rows = [(1, header), (2, ["通讯录", "", "用户组", long_cell, "❌", "✅", "✅"]),
                (3, ["通讯录", "组织", "动态", "短说明", "1", "0", "1"])]
        blocks = chunk_rows_to_blocks("xlsx", rows, "功能清单", max_tokens=400, overlap_tokens=0)
        chunks = self._chunks(blocks[1:], max_tokens=400)                                   # the first block is the summary
        self.assertEqual([c.text for c in chunks], [b.text for b in blocks[1:]])           # the native table's blocks are within the budget, the chunker passes them through
        self.assertTrue(all(count_tokens(c.text) <= 400 for c in chunks), [count_tokens(c.text) for c in chunks])
        pieces = [c.text.splitlines() for c in chunks if c.block.block_id.startswith("xlsx-功能清单-rows-2-2-p")]
        self.assertGreater(len(pieces), 2)
        self.assertTrue(all(p[:2] == ["SHEET: 功能清单", "ROWS: 2-2"] and len(p) == 3 for p in pieces))   # pieces that carry column names have no HEADER line
        lines = [p[2] for p in pieces]
        self.assertEqual(lines[0], "模块: 通讯录 | 二级: 用户组")                              # the empty level-1 cell is skipped, later column names stay in place
        self.assertTrue(all(l.startswith("模块: 通讯录 | 功能描述: 在云文档中使用当前用户组。") for l in lines[1:]))
        self.assertTrue(lines[-1].endswith("。 | v6.2: ❌ | 公网: ✅ | 私有云: ✅"))
        self.assertEqual("".join(l.split("功能描述: ", 1)[1].split(" | ")[0] for l in lines[1:]), long_cell)   # the long cell loses nothing and repeats nothing
        joined = "\n".join(lines)
        for wrong in ("模块: 模块", "一级: ", "二级: 功能描述", "功能描述: v6.2"):
            self.assertNotIn(wrong, joined)
        self.assertNotIn("模块: 通讯录", lines)                                             # no piece holding only the row identity
        self.assertEqual(chunks[-1].text.splitlines()[2:], ["HEADER: " + " | ".join(header), "通讯录 | 组织 | 动态 | 短说明 | 1 | 0 | 1"])

    def test_rows_that_already_carry_their_column_names_are_not_labelled_again(self) -> None:
        """Rows of a native table expanded with column names carry no HEADER line; should such a block need splitting
        again, its cells are used as they are: no "column N" labels, and no piece holding only the row identity."""
        from kb_pipeline.chunking.chunker import split_table_text
        from kb_pipeline.utils import count_tokens

        answer = "减少库存积压,减少断货。" * 120
        text = f"SHEET: 问卷\nROWS: 7-7\n提交时间: 2026-08-24 08:42:00 | 问题二: {answer} | 问题三: 近 30 天销量"
        pieces = split_table_text(text, 120)
        self.assertGreater(len(pieces), 3)
        for piece in pieces:
            self.assertLessEqual(count_tokens(piece), 120)
            self.assertTrue(piece.startswith("SHEET: 问卷\nROWS: 7-7\n提交时间: 2026-08-24 08:42:00 | 问题"), piece[:80])
        self.assertNotIn("column 1", "\n".join(pieces))
        self.assertTrue(pieces[-1].endswith(" | 问题三: 近 30 天销量"))
        self.assertEqual("".join(p.split("问题二: ", 1)[1].split(" | ")[0] for p in pieces if "问题二: " in p), answer)

    def test_native_rows_with_an_empty_first_cell_keep_their_columns_when_split(self) -> None:
        """A row whose first cell is empty starts with "| " in the text, one empty cell short; the cell is restored
        before the column names are paired by position, otherwise the whole row shifts one column to the left."""
        from kb_pipeline.chunking.chunker import split_table_text

        text = ("SHEET: 版本对比\nROWS: 2-3\nHEADER: | 产品规格 | 协作版 | 专业版\n"
                "| 在线文档 | " + "支持多人在线协作编辑。" * 60 + " | Web 端\n存储 | 容量 | 100G | 1T")
        pieces = split_table_text(text, 120)
        joined = "\n".join(pieces)
        self.assertIn("产品规格: 在线文档 | 协作版: 支持多人在线协作编辑。", joined)
        self.assertTrue(pieces[-2].endswith(" | 专业版: Web 端") or " | 专业版: Web 端\n" in pieces[-1], pieces[-2:])
        self.assertNotIn("协作版: Web 端", joined)
        self.assertNotIn("column 1: 在线文档", joined)
        self.assertIn("\n存储 | 容量 | 100G | 1T", pieces[-1])                                # a row that fits stays as it is

    def test_a_cell_longer_than_the_budget_is_cut_at_sentences_and_keeps_row_and_column(self) -> None:
        """A single cell over the budget: the whole cell used to become one chunk as it was, with no upper bound on the
        chunk length (the longest in production was 5,921 tokens against a budget of 400)."""
        from kb_pipeline.chunking.chunker import split_table_text
        from kb_pipeline.utils import count_tokens

        note = "这是第一句说明。" * 80
        head = "| 编号 | 名称 | 备注 |\n| --- | --- | --- |"
        pieces = split_table_text(f"TITLE: 表 1\n{head}\n| A-001 | 甲 | {note} |\n| A-002 | 乙 | 短 |\nFOOTNOTE: 注", 100)
        for piece in pieces:
            self.assertTrue(piece.startswith("TITLE: 表 1\n"), piece[:60])                  # every piece carries the title
            self.assertLessEqual(count_tokens(piece), 100)
        self.assertEqual(pieces[-1], f"TITLE: 表 1\n{head}\n| A-002 | 乙 | 短 |\nFOOTNOTE: 注")   # rows that fit are unchanged: positional, with the header
        rows = [l for p in pieces[:-1] for l in p.splitlines()[1:]]
        self.assertEqual(rows[0], "编号: A-001 | 名称: 甲")
        self.assertGreater(len(rows), 4)
        self.assertTrue(all(l.startswith("编号: A-001 | 备注: 这是第一句说明。") and l.endswith("。") for l in rows[1:]), rows[1:3])   # the cuts fall at sentence ends
        self.assertEqual("".join(l.split("备注: ", 1)[1] for l in rows[1:]), note)
        self.assertNotIn("---", "\n".join(pieces[:-1]))                                   # pieces that carry column names have no header rows
        # A long cell without a single sentence (no sentence-final punctuation at all) is split by token
        blob = "没有标点的一整段内容" * 100
        pieces = split_table_text(f"{head}\n| A-003 | 丙 | {blob} |", 100)
        self.assertTrue(all(count_tokens(p) <= 100 for p in pieces))
        self.assertEqual("".join(p.split("备注: ", 1)[1] for p in pieces if "备注: " in p), blob)

    def test_a_header_that_eats_the_budget_is_not_repeated_in_every_piece(self) -> None:
        """A header longer than the budget: every chunk used to be "the whole header + a row or two of data", more than
        twice the budget, and the vectors of one table were nearly identical. Now every value is preceded by its own
        column name and the header rows are not repeated; the title line is still carried by every chunk."""
        from kb_pipeline.chunking.chunker import split_table_text
        from kb_pipeline.utils import count_tokens

        names = [f"第{i}题:您认为当前流程里最需要改进的环节是什么" for i in range(30)]
        rows = [[f"回答{r}-{c}" if (r + c) % 7 else "" for c in range(30)] for r in range(4)]
        text = "\n".join(["TITLE: 调研结果", "| " + " | ".join(names) + " |", "| " + " | ".join(["---"] * 30) + " |",
                          *("| " + " | ".join(r) + " |" for r in rows), "FOOTNOTE: 数据截至 2026 年 8 月。"])
        self.assertGreater(count_tokens("| " + " | ".join(names) + " |"), 400)
        pieces = split_table_text(text, 400)
        joined = "\n".join(pieces)
        for piece in pieces:
            self.assertLessEqual(count_tokens(piece), 400)
            self.assertTrue(piece.startswith("TITLE: 调研结果\n"), piece[:40])
        self.assertNotIn("| --- |", joined)
        self.assertNotIn(" | ".join(names[:2]), joined)
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                if value:
                    self.assertIn(f"{names[c]}: {value}", joined)                          # every value sits under its own column name
        self.assertEqual(joined.count("FOOTNOTE: 数据截至 2026 年 8 月。"), 1)
        self.assertNotIn(": FOOTNOTE", joined)
        # A table whose header takes only a small part of the budget is unchanged: every chunk still carries the
        # header, and rows stay positional
        small = "| a | b |\n| --- | --- |\n" + "\n".join(f"| 行{i} | 值{i} |" for i in range(200))
        self.assertTrue(all(p.startswith("| a | b |\n| --- | --- |\n| 行") for p in split_table_text(small, 120)))

    def test_a_table_that_is_one_long_cell_is_cut_as_text(self) -> None:
        """A "table" into which a whole page of text was recognized as one cell: a header row and no data rows, so the
        header is the content. It used to become one chunk as it was, however long."""
        from kb_pipeline.chunking.chunker import split_table_text
        from kb_pipeline.utils import count_tokens

        page = "整页的文字被识别进了一个格子里,一个句末标点都没有" * 200
        for text in (f"TITLE: 第 12 页\n| {page} |\n| --- |", f"<table><tr><td>{page}</td></tr></table>"):
            pieces = split_table_text(text, 400)
            self.assertGreater(len(pieces), 5)
            self.assertTrue(all(count_tokens(p) <= 400 for p in pieces), [count_tokens(p) for p in pieces])
            self.assertEqual("".join(p.removeprefix("TITLE: 第 12 页\n") for p in pieces).count("整页的文字被识别进了一个格子里"), 200)
            self.assertNotIn("column 1", "\n".join(pieces))

    def test_native_table_with_a_header_longer_than_the_budget(self) -> None:
        """A 40-column questionnaire sheet with a header of 700-odd tokens and a budget of 400: 15 rows of data used to
        be cut into 330 chunks, each carrying the whole header (median 788 tokens)."""
        from kb_pipeline.parsers.native_table import chunk_rows_to_blocks
        from kb_pipeline.utils import count_tokens

        names = ["提交时间"] + [f"第{i}题:您认为当前流程里最需要改进的环节是什么" for i in range(1, 40)]
        rows = [(1, names)] + [(r + 2, [f"2026-08-{r + 1:02d}"] + [f"回答{r}-{c}" if (r + c) % 5 else "" for c in range(1, 40)])
                               for r in range(15)]
        blocks = chunk_rows_to_blocks("xlsx", rows, "问卷", max_tokens=400, overlap_tokens=0)
        body = [b for b in blocks if not b.metadata.get("summary")]
        self.assertLess(len(body), 60)
        self.assertTrue(all(count_tokens(b.text) <= 400 for b in body), [count_tokens(b.text) for b in body])
        self.assertTrue(all("HEADER:" not in b.text for b in body))
        self.assertEqual(len({b.block_id for b in blocks}), len(blocks))
        joined = "\n".join(b.text for b in body)
        for r, (_, row) in enumerate(rows[1:]):
            for c, value in enumerate(row):
                if value and c:
                    self.assertIn(f"{names[c]}: {value}", joined)
        for b in body:                                                                     # every piece tells which row it belongs to
            self.assertTrue(b.text.splitlines()[2].startswith("提交时间: 2026-08-"), b.text[:80])
        chunks = self._chunks(blocks, max_tokens=400)
        self.assertTrue(all(count_tokens(c.text) <= 400 for c in chunks))
        self.assertEqual([c.text for c in chunks if c.block.block_id != blocks[0].block_id], [b.text for b in body])
        summary = "\n".join(c.text for c in chunks if c.block.block_id == blocks[0].block_id)
        self.assertTrue(all(name in summary for name in names))                            # the whole header is still in the summary chunks

    def test_notes_around_a_table_are_cut_as_text_not_as_cells(self) -> None:
        """A footnote after a table is not a table row: over the budget it used to be treated as a one-cell row,
        labelled with the name of the first column, and became one chunk as it was."""
        from kb_pipeline.chunking.chunker import split_table_text
        from kb_pipeline.utils import count_tokens

        note = "FOOTNOTE: " + "本表数值均为实测结果,仅供参考。" * 40
        head = "| 参数 | 数值 |\n| --- | --- |"
        pieces = split_table_text(f"{head}\n| 电压 | 3.3 |\n{note}", 100)
        self.assertTrue(all(p.startswith(head + "\n") and count_tokens(p) <= 100 for p in pieces))
        rest = [l for p in pieces for l in p.splitlines()[2:]]
        self.assertEqual(rest[0], "| 电压 | 3.3 |")
        self.assertEqual("".join(rest[1:]), note)
        self.assertNotIn("参数: ", "\n".join(pieces))

    def test_a_caption_that_eats_the_budget_is_written_once_before_the_table(self) -> None:
        """A caption that alone takes up most of the budget: every chunk used to carry it and ran to twice the budget.
        Now it appears once at the start, split as a passage of text, while the short title line is still carried
        by every chunk."""
        from kb_pipeline.chunking.chunker import split_table_text
        from kb_pipeline.utils import count_tokens

        caption = "CAPTION: " + "本表列出各型号在不同工况下的实测结果。" * 30
        head = "| 型号 | 数值 |\n| --- | --- |"
        rows = [f"| M{i} | {i} |" for i in range(60)]
        pieces = split_table_text("\n".join(["TITLE: 实测", caption, head, *rows]), 200)
        self.assertGreater(count_tokens(caption), 200)
        self.assertTrue(all(p.startswith("TITLE: 实测\n") and count_tokens(p) <= 200 for p in pieces), [count_tokens(p) for p in pieces])
        bodies = [p.removeprefix("TITLE: 实测\n") for p in pieces]
        text = [b for b in bodies if not b.startswith(head)]
        self.assertEqual(bodies[:len(text)], text)                                          # caption first, table after
        self.assertEqual("".join(text), caption)
        self.assertEqual([l for b in bodies[len(text):] for l in b.splitlines()[2:]], rows)
        # A header and no data rows, and the header does not fit: the header becomes body text after the caption
        wide = "| " + " | ".join(f"第{i}列的名字比较长" for i in range(40)) + " |"
        pieces = split_table_text("\n".join(["TITLE: 实测", caption, wide, "| " + " | ".join(["---"] * 40) + " |"]), 200)
        self.assertTrue(all(count_tokens(p) <= 200 for p in pieces))
        joined = "".join(p.removeprefix("TITLE: 实测\n") for p in pieces)
        self.assertEqual(joined.replace(" ", ""), (caption + wide).replace(" ", ""))        # spaces at the cut points are dropped

    def test_a_fence_line_longer_than_the_budget_is_cut(self) -> None:
        """JSON / base64 squashed into one line inside a fence: fences used to be split by line only, so the chunk was
        as long as that line."""
        from kb_pipeline.chunking.chunker import _split_fence
        from kb_pipeline.utils import count_tokens

        blob = '{"k": "' + "abc123XYZ+/" * 3000 + '"}'
        parts = _split_fence(f"```json\nconst a = 1;\n{blob}\nconst b = 2;\n```", 200)
        self.assertGreater(len(parts), 10)
        for part in parts:
            self.assertLessEqual(count_tokens(part), 200)
            self.assertTrue(part.startswith("```json\n") and part.endswith("\n```"), part[:30])
        inner = [l for p in parts for l in p.splitlines()[1:-1]]
        self.assertEqual((inner[0], inner[-1]), ("const a = 1;", "const b = 2;"))
        self.assertEqual("".join(inner[1:-1]), blob)


class EmbeddingRejectionTests(unittest.TestCase):
    """The embedding service rejecting a chunk for its content (over the length limit of the model) is
    deterministic: the job used to retry 5 times with backoff anyway, the scan queued another round after the
    24-hour cooldown, the same file failed forever, and the error did not say which chunk it was."""

    class _Http(Exception):
        def __init__(self, status: int, message: str) -> None:
            super().__init__(message)
            self.status_code = status

    def _client(self, accepts, *, batch_size: int = 3):
        from kb_pipeline.embedding.client import EmbeddingClient

        calls: list[list[str]] = []
        http = self._Http

        class Embeddings:
            def create(self, *, model, input, dimensions):
                calls.append(list(input))
                verdict = accepts(list(input))
                if verdict is not True:
                    raise http(*verdict)
                return SimpleNamespace(data=[SimpleNamespace(embedding=[0.0] * dimensions) for _ in input])

        client = EmbeddingClient(base_url="http://embed.test/v1", api_key="k", model_id="m", dim=4,
                                 batch_size=batch_size, retry=2, sleep_seconds=0)
        client.client = SimpleNamespace(embeddings=Embeddings())
        return client, calls

    def test_the_client_names_the_rejected_input(self) -> None:
        from kb_pipeline.embedding.client import EmbeddingInputRejected
        from kb_pipeline.pipeline.worker import _should_retry

        too_long = (400, "This model's maximum context length is 8192 tokens")
        client, calls = self._client(lambda batch: too_long if any(len(t) > 100 for t in batch) else True)
        texts = ["短一", "短二", "短三", "短四", "长" * 500, "短六", "短七"]
        with self.assertRaises(EmbeddingInputRejected) as caught:
            client.embed(texts)
        self.assertEqual(caught.exception.index, 4)                     # the index in the whole file, not within the batch
        self.assertIn("maximum context length", caught.exception.reason)
        self.assertEqual(calls[:2], [texts[:3], texts[3:6]])            # the first batch passes, the second is rejected
        self.assertEqual(calls[2:], [["短四"], ["长" * 500], ["长" * 32]])   # sent one at a time to find the culprit; later batches are not sent
        self.assertFalse(_should_retry({"retry_count": 0}, SimpleNamespace(job_max_retries=5), caught.exception))
        self.assertEqual(len(client.embed(["短一", "短二"])), 2)

    def test_a_service_that_rejects_everything_is_not_blamed_on_the_input(self) -> None:
        """The service answers 400 to any input (a parameter / configuration problem), or it rejects the batch as a
        whole: no single chunk is at fault, the original error goes through, and the job-level retries and the
        requeue after the cooldown both remain."""
        from kb_pipeline.pipeline.worker import _should_retry

        client, _ = self._client(lambda batch: (400, "dimensions is not supported by this model"))
        with self.assertRaises(self._Http) as caught:
            client.embed(["一", "二"])
        self.assertTrue(_should_retry({"retry_count": 0}, SimpleNamespace(job_max_retries=5), caught.exception))
        client, _ = self._client(lambda batch: (400, "batch too large") if len(batch) > 1 else True)
        with self.assertRaises(self._Http):
            client.embed(["一", "二"])
        client, calls = self._client(lambda batch: (401, "bad key"))     # other 4xx statuses are not tried one by one
        with self.assertRaises(self._Http):
            client.embed(["一", "二"])
        self.assertEqual(len(calls), 1)

    def test_a_rejected_chunk_fails_the_file_once_and_says_which_chunk(self) -> None:
        from unittest import mock

        from kb_pipeline.embedding.client import EmbeddingInputRejected
        from kb_pipeline.pipeline import parse_job, scheduler, worker as worker_mod

        class Rejecting:
            def __init__(self, **kwargs) -> None:
                pass

            def embed(self, texts):
                raise EmbeddingInputRejected(1, "maximum context length is 8192 tokens")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = root / "s.db"
            db.init_db(state)
            doc = root / "a.md"
            doc.write_text("正文", encoding="utf-8")
            file = SourceFile(**{**_local_file("a.md").__dict__, "physical_path": str(doc)})
            settings = SimpleNamespace(
                state_db=state, parse_enabled=True, parse_job_lease_seconds=600, metadata_job_lease_seconds=600,
                job_max_retries=5, job_retry_base_seconds=300, job_retry_max_seconds=3600, max_file_bytes=0,
                cache_dir=root / "cache", sources={}, vlm_failure_retry_ratio=0.5,
                embedding_base_url="http://embed.test/v1", embedding_api_key="k", embedding_model_id="m", embedding_dim=4,
                embedding_batch=20, embedding_retry=1, embedding_sleep_seconds=0,
                qdrant_url="http://q", qdrant_api_key="", opensearch_url="http://o",
            )
            blocks = [_mb(f"b{i}", f"第{i}段正文,写得足够长,不会被当成碎片并进邻片。" * 6, metadata={"section_path": [f"第{i}节"]})
                      for i in (1, 2, 3)]
            with db.connect(state) as con:
                db.upsert_file(con, file, status="seen")
                fid = db.file_id_for(file.kb_id, file.file_key)
                job_id = db.enqueue_job(con, ingest_run_id=None, file_id=fid, kb_id=file.kb_id, collection=file.collection,
                                        file_key=file.file_key, job_type="parse", dedupe_key=f"parse:{fid}")
                con.commit()
                with mock.patch.object(parse_job, "_source_for_file", return_value=SimpleNamespace(max_tokens=400, overlap_tokens=0)), \
                        mock.patch.object(parse_job, "verify_source_file"), \
                        mock.patch.object(parse_job, "_parse_blocks", return_value=blocks), \
                        mock.patch.object(parse_job, "EmbeddingClient", Rejecting):
                    result = worker_mod.run_once(con, settings=settings)
                row = con.execute("SELECT status, retry_count, error, finished_at FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        self.assertTrue(result.startswith("failed:"), result)
        self.assertEqual((row["status"], row["retry_count"]), ("failed", 0))           # settled at once, no retries with backoff
        self.assertIn("The embedding service rejected chunk 2/3 (block_id=b2, ", row["error"])
        self.assertIn("maximum context length is 8192 tokens", row["error"])
        long_ago = {"error": row["error"], "finished_at": 1}
        self.assertTrue(scheduler._failed_parse_blocks_auto_requeue(long_ago))          # past the cooldown the scan still does not requeue it
        self.assertFalse(scheduler._failed_parse_blocks_auto_requeue(long_ago, "parser_changed"))   # a parser version bump gives it another chance


class DiagnosisRuleTests(unittest.TestCase):
    _chunk = staticmethod(ChunkDiagnosticsTests._chunk)

    def test_oversized_has_tolerance_and_code_slides_are_not_tiny(self) -> None:
        from kb_pipeline.chunking.diagnose import chunk_diagnostics

        chunks = ([self._chunk(t) for t in (380, 390, 440, 480)]
                  + [self._chunk(t, "code") for t in (5, 6, 7, 8)] + [self._chunk(9, "slide")])
        diag = chunk_diagnostics(chunks, max_tokens=400)
        self.assertTrue(diag["ok"], diag)
        self.assertEqual(diag["stats"]["over_count"], 0)
        self.assertEqual(diag["stats"]["tiny_count"], 0)
        diag = chunk_diagnostics([self._chunk(t) for t in (380, 520)], max_tokens=400)
        self.assertEqual([r["key"] for r in diag["reasons"]], ["oversized"])

    def test_long_visual_headerless_tables_and_orphan_headings(self) -> None:
        from kb_pipeline.chunking.diagnose import chunk_diagnostics
        from kb_pipeline.models import UnifiedChunk

        def mk(i, text, bt, tokens):
            block = ParsedBlock(parser="t", parser_profile="p", doc_type="d", block_type=bt, text=text,
                                block_id=f"b{i}", metadata={"section_path": []})
            return UnifiedChunk(chunk_uid=f"u{i}", chunk_index=i, text=text, block=block, token_count=tokens)

        chunks = [mk(0, "x" * 400, "text", 380), mk(1, "图片识读", "image", 1100),
                  mk(2, "| a | b |\n| 1 | 2 |", "table", 30), mk(3, "| c | d |\n| 3 | 4 |", "table", 30),
                  mk(4, "第一章 绪论", "text", 3), mk(5, "1.2 概述", "text", 5), mk(6, "学习要点", "text", 4)]
        diag = chunk_diagnostics(chunks, max_tokens=400)
        keys = {r["key"] for r in diag["reasons"]}
        self.assertTrue({"long_visual", "headerless_tables", "orphan_headings"} <= keys, keys)
        self.assertEqual(diag["stats"]["long_visual"], 1)
        self.assertEqual(diag["stats"]["headerless_tables"], 2)
        self.assertEqual(diag["stats"]["orphan_headings"], 3)
        ok = chunk_diagnostics([mk(0, "x", "text", 380), mk(1, "| a | b |\n| --- | --- |\n| 1 | 2 |", "table", 30)], max_tokens=400)
        self.assertEqual(ok["stats"]["headerless_tables"], 0)


class ChunkGlueIntoFiguresTests(unittest.TestCase):
    _chunks = staticmethod(ChunkStructureTests._chunks)
    T0 = "| a | b |\n| --- | --- |\n| 1 | 2 |"
    T1 = "| 任务类型 | Pro | flash |\n| --- | --- | --- |\n| 普通任务 | deepseek-v4-pro | deepseek-v4-flash |"

    def test_tiny_lead_in_paragraph_joins_the_table_chunk(self) -> None:
        blocks = [_mb("t0", self.T0, block_type="table", table_markdown=self.T0),
                  _mb("p", "企业通过额度使用时无需理解每个模块配置什么模型,默认以下模型:"),
                  _mb("t1", self.T1, block_type="table", table_markdown=self.T1)]
        chunks = self._chunks(blocks, max_tokens=400)
        self.assertEqual([c.block.block_id for c in chunks], ["t0", "t1"])
        self.assertTrue(chunks[1].text.startswith("企业通过额度使用时"))
        self.assertIn("| 任务类型 |", chunks[1].text)

    def test_tiny_text_between_a_table_and_an_image_goes_forward(self) -> None:
        blocks = [_mb("t0", self.T0, block_type="table", table_markdown=self.T0),
                  _mb("p", "只要企业有额度,默认全企业员工可用。"),
                  _mb("img", "VISUAL SUMMARY: 配置界面截图", block_type="image", visual_summary="配置界面截图")]
        chunks = self._chunks(blocks, max_tokens=400)
        self.assertEqual([c.block.block_id for c in chunks], ["t0", "img"])
        self.assertTrue(chunks[1].text.startswith("只要企业有额度"))

    def test_short_text_joins_a_figure_within_the_slack_only(self) -> None:
        """The slack for merging short text into an adjacent image chunk is 25% of the budget, the same as for
        body fragments (2026-09-11, agreed with the user: a slide's title belongs with its own image); beyond
        the slack they stay separate chunks, and the image chunk is not pushed any further."""
        from kb_pipeline.utils import count_tokens

        big = "VISUAL SUMMARY: " + "很长的图片描述。" * 60                       # far beyond 120 × 1.25
        blocks = [_mb("img", big, block_type="image", visual_summary=big[16:]),
                  _mb("p", "一句短话。")]
        chunks = self._chunks(blocks, max_tokens=120)
        self.assertEqual([c.block.block_id for c in chunks], ["img", "p"])
        near = "VISUAL SUMMARY: " + "图片描述。" * 22                             # right at the budget (about 110)
        blocks = [_mb("title", "六个财务 Agent,六条业务链"), _mb("img2", near, block_type="image", visual_summary=near[16:])]
        chunks = self._chunks(blocks, max_tokens=120)
        self.assertEqual([c.block.block_id for c in chunks], ["img2"])          # the title merged into the image chunk
        self.assertTrue(chunks[0].text.startswith("六个财务 Agent,六条业务链\n"))
        self.assertLessEqual(chunks[0].token_count, int(120 * 1.25))


class VisualTextCompositionTests(unittest.TestCase):
    """When the VLM has a result, MinerU's OCR text of screenshots / photos is dropped; for charts /
    architecture diagrams the OCR text goes after the VLM sections; small icons removed by the size filter
    are marked decorative and do not become chunks of their own."""

    @staticmethod
    def _enrich(blocks, kind, tmp, *, summary="摘要", facts=("f1",)):
        from kb_pipeline.parsers import visual_blocks

        def fake_caption(jobs, **kwargs):
            return {job_id: {"kind": kind, "title": "", "summary": summary, "text_verbatim": "",
                             "entities": [], "facts": list(facts), "keywords": ["k1"], "confidence": "high",
                             "decorative": False}
                    for job_id, *_ in jobs}

        with patch.object(visual_blocks, "caption_images_parallel", fake_caption):
            visual_blocks.enrich_blocks_with_vlm(blocks, base_url="u", api_key="k", model_id="m",
                                                 cache_dir=Path(tmp), filter_decorative=False)

    @staticmethod
    def _img(bid, text, ref):
        return ParsedBlock(parser="p", parser_profile="p", doc_type="pdf", block_type="image", text=text,
                           block_id=bid, visual_ref=ref)

    def test_screenshot_own_text_is_replaced_by_vlm_sections(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            img = Path(tmp) / "i.png"; Image.new("RGB", (300, 200)).save(img)
            shot = self._img("a", "首页\n密码\n密码", str(img))
            diagram = self._img("b", "A -> B", str(img))
            empty = self._img("c", "首页\n密码", str(img))
            self._enrich([shot], "screenshot", tmp)
            self._enrich([diagram], "diagram", tmp)
            self._enrich([empty], "screenshot", tmp, summary="", facts=())
        self.assertEqual(shot.text.splitlines()[0], "VISUAL SUMMARY: 摘要")
        self.assertIn("FACTS: f1", shot.text)
        self.assertNotIn("密码", shot.text)
        self.assertTrue(shot.metadata.get("own_text_dropped"))
        self.assertTrue(diagram.text.startswith("VISUAL SUMMARY: 摘要"))
        self.assertTrue(diagram.text.endswith("\nA -> B"))
        self.assertFalse(diagram.metadata.get("own_text_dropped"))
        self.assertIn("首页", empty.text)                      # VLM returned nothing: the OCR text is kept as before
        self.assertFalse(empty.metadata.get("own_text_dropped"))

    def test_size_filtered_icons_are_marked_decorative(self) -> None:
        from PIL import Image
        from kb_pipeline.parsers import visual_blocks

        with tempfile.TemporaryDirectory() as tmp:
            img = Path(tmp) / "icon.png"; Image.new("RGB", (20, 20)).save(img)
            icon = self._img("i", "", str(img))
            with patch.object(visual_blocks, "caption_images_parallel", lambda jobs, **kw: {}):
                visual_blocks.enrich_blocks_with_vlm([icon], base_url="u", api_key="k", model_id="m", cache_dir=Path(tmp))
        self.assertTrue(icon.metadata.get("decorative"))
        self.assertEqual(icon.metadata.get("vlm_skip_reason"), "tiny_icon")
        # chunking: a decorative image with no body chunk before it simply produces no chunk
        icon.title = "AIHub 使用手册"
        chunks = ChunkStructureTests._chunks([icon, _mb("p", "正文段落,说明产品外观。" * 3)], max_tokens=120)
        self.assertEqual([c.block.block_id for c in chunks], ["p"])


class ChunkGlueAcrossSlidesTests(unittest.TestCase):
    _chunks = staticmethod(ChunkStructureTests._chunks)

    def test_heading_glue_stays_on_the_same_slide_and_drops_title_lines(self) -> None:
        slide = _mb("s5", "**客户墙**", block_type="slide", title="deck slide 5", page_idx=5, slide_idx=5)
        img1 = _mb("img1", "VISUAL SUMMARY: 医院大楼", block_type="image", visual_summary="医院大楼", page_idx=1, slide_idx=1)
        chunks = self._chunks([slide, img1], max_tokens=400)
        self.assertEqual([c.block.block_id for c in chunks], ["s5", "img1"])       # different pages: no merge
        img5 = _mb("img5", "VISUAL SUMMARY: 客户 logo 墙", block_type="image", visual_summary="客户 logo 墙", page_idx=5, slide_idx=5)
        chunks = self._chunks([slide, img5], max_tokens=400)
        self.assertEqual([c.block.block_id for c in chunks], ["img5"])
        self.assertTrue(chunks[0].text.startswith("TITLE: 客户墙\nVISUAL SUMMARY"), chunks[0].text)

    def test_pdf_pages_still_flow(self) -> None:
        h = _mb("h", "直流电气特性", page_idx=3)
        t = _mb("t", "| a | b |\n| --- | --- |\n| 1 | 2 |", block_type="table", table_markdown="| a | b |\n| --- | --- |\n| 1 | 2 |", page_idx=4)
        chunks = self._chunks([h, t], max_tokens=400)
        self.assertEqual([c.block.block_id for c in chunks], ["t"])
        self.assertTrue(chunks[0].text.startswith("TITLE: 直流电气特性"))


class HeadingRuleFollowupTests(unittest.TestCase):
    def test_zero_padded_steps_and_multiline_text_are_not_headings(self) -> None:
        from kb_pipeline.headings import HeadingResolver, infer_heading_level

        self.assertIsNone(infer_heading_level("06. 设置端口号"))
        self.assertIsNone(infer_heading_level("06.\n设置端口号"))
        self.assertEqual(infer_heading_level("6. 设置端口号"), 1)
        self.assertIsNone(infer_heading_level("2-1 SELECT语句基础\n列的查询"))
        r = HeadingResolver()
        self.assertEqual(r.resolve(1, "06.\n设置端口号"), (None, False))


class TocPageTests(unittest.TestCase):
    def test_toc_pages_are_detected_by_short_line_density_or_leaders(self) -> None:
        from kb_pipeline.headings import detect_toc_pages

        entries = [(3, f"{i}-1 第{i}节的标题") for i in range(25)]                       # book TOC page: a whole page of short lines
        entries += [(4, "本书面向完全没有编程和系统开发经验的读者,循序渐进地讲解。" * 2)] * 5 + [(4, "前言"), (4, "关于本书"), (4, "读者对象")]
        entries += [(5, f"第 {i} 节 …… {i + 3}") for i in range(10)] + [(5, "目录")]      # datasheet TOC page: dot leaders with page numbers
        entries += [(None, "无页码的项")]
        self.assertEqual(detect_toc_pages(entries), {3, 5})

    def test_step_labels_are_not_headings_even_when_the_parser_says_so(self) -> None:
        from kb_pipeline.headings import HeadingResolver

        r = HeadingResolver()
        self.assertEqual(r.resolve(1, "06. 设置端口号"), (None, False))
        self.assertEqual(r.resolve(1, "01、下载安装程序"), (None, False))
        self.assertEqual(r.resolve(1, "6. 设置端口号")[0], 1)

    def test_pdf_parser_skips_headings_on_toc_pages(self) -> None:
        src = _repo_file("app/kb_pipeline/parsers/mineru_pdf.py")
        self.assertIn("toc_pages = detect_toc_pages(", src)
        self.assertIn("page_idx(item) not in toc_pages", src)

    def test_toc_page_log_uses_the_same_page_numbers_as_the_blocks(self) -> None:
        """The page numbers of TOC pages in the log used to be one more than those on the blocks (already
        1-based numbers had one added again)."""
        import contextlib
        import io

        from kb_pipeline.parsers import mineru_pdf

        toc = [{"type": "text", "text": f"{i}.1 第{i}节的标题", "text_level": 1, "page_idx": 2} for i in range(1, 26)]
        body = [{"type": "text", "text": "1.1 第1节的标题", "text_level": 1, "page_idx": 3},
                {"type": "text", "text": "正文从这里开始,讲第一节的内容。", "page_idx": 3}]
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(out), \
                patch.object(mineru_pdf, "call_mineru_sync", return_value={"content_list": toc + body}):
            blocks = mineru_pdf.mineru_pdf_blocks(mineru_url="http://x", path=Path(tmp) / "a.pdf", cache_dir=Path(tmp))
        self.assertEqual({b.page_idx for b in blocks if not b.metadata["section_path"]}, {3})     # lines on the TOC page are not sections
        self.assertEqual(blocks[-1].metadata["section_path"], ["1.1 第1节的标题"])
        self.assertIn("pdf toc pages=[3] ", out.getvalue())


class ChunkingFixRegressionTests(_CodexAudit20260906TestsSupport, unittest.TestCase):
    """Regressions for problems found by successive re-reviews, health checks and audits; each test's
    docstring records where it came from and the symptom observed at the time."""

    def test_fallback_title_keeps_vlm_seen_figures_alive(self) -> None:  # issue 2
        from kb_pipeline.chunking.chunker import text_for_block
        from kb_pipeline.models import ParsedBlock
        from kb_pipeline.parsers.visual_blocks import ensure_visual_blocks_have_text

        bare = ParsedBlock(parser="mineru", parser_profile="p", doc_type="pdf", block_type="image",
                           text="", block_id="img1", page_idx=3, visual_ref="/x.jpg",
                           metadata={"vlm_status": "failed"})
        captioned = ParsedBlock(parser="mineru", parser_profile="p", doc_type="pdf", block_type="chart",
                                text="", block_id="img2", caption="图2", visual_ref="/y.jpg",
                                metadata={"vlm_status": "empty"})
        not_seen = ParsedBlock(parser="mineru", parser_profile="p", doc_type="pdf", block_type="image",
                               text="", block_id="img3", visual_ref="/z.jpg",
                               metadata={"vlm_status": "skipped"})
        patched = ensure_visual_blocks_have_text([bare, captioned, not_seen], doc_name="报告.pdf")
        self.assertEqual(patched, 1)
        self.assertEqual(bare.title, "报告.pdf p3 image")
        self.assertTrue(text_for_block(bare))          # the block now yields a chunk
        self.assertIsNone(captioned.title)             # had a caption; untouched
        self.assertIsNone(not_seen.title)              # never went through the VLM

    def test_chinese_sentences_split_without_trailing_space(self) -> None:  # issue 11
        from kb_pipeline.utils import split_sentences

        parts = split_sentences("第一句。第二句！第三句？收尾")
        self.assertEqual(parts, ["第一句。", "第二句！", "第三句？", "收尾"])
        self.assertEqual(split_sentences("圆周率是3.14159,路径是a.b.c"), ["圆周率是3.14159,路径是a.b.c"])

    def test_one_term_per_line_pages_stay_within_budget(self) -> None:
        """Pages with one term per line, such as a glossary index / TOC: the budget used to be the sum of the
        sentence tokens, ignoring the newlines between sentences, so a block of 261 "sentences" produced a
        783-token chunk (budget 600, 2026-09-11, the library KB). The real token count must stay within budget."""
        from kb_pipeline.chunking.chunker import chunk_text
        from kb_pipeline.utils import count_tokens

        terms = [f"术语{i}" for i in range(300)]
        text = "\n\n".join(terms)
        for overlap in (0, 30):
            parts = chunk_text(text, 100, overlap)
            self.assertGreater(len(parts), 3)
            self.assertTrue(all(count_tokens(p) <= 100 for p in parts), [count_tokens(p) for p in parts])
        self.assertEqual("\n".join(chunk_text(text, 100, 0)).split("\n"), terms)     # no term lost, none duplicated

    def test_long_single_line_chunks_fast_and_within_budget(self) -> None:  # issue 11
        import time as time_module

        from kb_pipeline.chunking.chunker import chunk_text
        from kb_pipeline.utils import count_tokens

        blob = ("x" * 97 + ",yz") * 2000  # 200 KB, no sentence boundaries
        started = time_module.time()
        parts = chunk_text(blob, max_tokens=400, overlap_tokens=0)
        elapsed = time_module.time() - started
        self.assertLess(elapsed, 20.0)
        self.assertTrue(parts)
        self.assertTrue(all(count_tokens(piece) <= 400 for piece in parts))
        self.assertEqual("".join(parts), blob)

    def test_f09_short_slides_with_the_same_title_do_not_merge(self) -> None:
        """Two slides both titled "product features", each with only a short body: the text-joining branch used
        to look at the section path alone, merged the two into one chunk and kept only the later slide_idx; now
        the slide boundary wins. PDF pages still flow continuously."""
        s1 = _mb("s1", "产品特点\n低功耗。", block_type="slide", page_idx=1, slide_idx=1, doc_type="pptx",
                 metadata={"section_path": ["产品特点"]})
        s2 = _mb("s2", "产品特点\n高可靠。", block_type="slide", page_idx=2, slide_idx=2, doc_type="pptx",
                 metadata={"section_path": ["产品特点"]})
        chunks = self._chunks([s1, s2], max_tokens=400)
        self.assertEqual([c.block.slide_idx for c in chunks], [1, 2])
        self.assertEqual([c.block.block_id for c in chunks], ["s1", "s2"])
        p1 = _mb("p1", "第一段很短。", page_idx=3, metadata={"section_path": ["直流特性"]})
        p2 = _mb("p2", "第二段也短。", page_idx=4, metadata={"section_path": ["直流特性"]})
        self.assertEqual(len(self._chunks([p1, p2], max_tokens=400)), 1)
