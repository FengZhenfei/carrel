"""Regression tests for Patch 13 (narrowing Patch 12 embedded-image heading & horizontal colspan rules).
Synthetic fixtures only; no real document filenames or body text."""
from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kb_pipeline.chunking.chunker import blocks_to_chunks, split_table_text
from kb_pipeline.models import ParsedBlock
from kb_pipeline.parsers import mineru_pdf
from kb_pipeline.parsers.common import (
    grid_to_markdown,
    html_table_spans,
    html_table_to_grid,
    html_table_to_markdown,
    strip_colspan_prefixes,
)
from kb_pipeline.parsers.html_dom import parse_html_dom
from kb_pipeline.parsers.pdf_enhanced import merge_mineru_text_blocks
from kb_pipeline.parsers.table_repair import _expand_review_spans, review_pdf_tables


def _png_bytes(width: int = 300, height: int = 200, color=(235, 235, 235)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


class Fix1EmbeddedImageHeadingScopeTests(unittest.TestCase):
    def test_standalone_title_badge_and_templated_slide_title_prevent_cross_page_hijack(self) -> None:
        """A landscape slide deck where page N+1 has an unmarked top-slot slide title and a standalone
        product-badge graphic title (sole text occupying >=25% of its small image) must attribute page N+1
        capabilities to page N+1's topic and never leak page N's trailing subsection into chunks or spans."""
        import pymupdf

        img_bytes = _png_bytes(200, 100)
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "slides.pdf"
            doc = pymupdf.open()
            # 5 landscape 16:9 slides (960x540): slides 1-3 have MinerU-marked top titles; slides 4-5 have
            # unmarked top banners in the exact same top slot (20pt vs 12pt body), and slide 5 has a badge graphic.
            slide_specs = [
                ("Topic A | Overview One", ["Detail A1", "Detail A2"]),
                ("Topic B | Overview Two", ["Detail B1", "Detail B2"]),
                ("Topic C | Overview Three", ["Detail C1", "Detail C2"]),
                ("Topic D | Knowledge Base", ["Connect External Systems", "Open API integration for frontline staff."]),
                ("Topic E | Smart Assistant", ["Preset Prompts", "Custom Commands", "Read Messages", "Check Mail"]),
            ]
            for idx, (banner, lines) in enumerate(slide_specs):
                pg = doc.new_page(width=960, height=540)
                pg.insert_text((34, 52), banner, fontsize=20)
                if idx == 4:
                    # Standalone product badge image at x=36..255, y=103..160 (norm [37.5, 191, 265, 296])
                    pg.insert_image(pymupdf.Rect(36, 103, 255, 160), stream=img_bytes)
                for li, ln in enumerate(lines):
                    pg.insert_text((60, 195 + li * 42), ln, fontsize=12)
            doc.save(str(pdf_path))
            doc.close()

            content = [
                # Pages 1-3: marked top titles at consistent slot y=64..107, x=33..450
                {"type": "text", "text": "Topic A | Overview One", "text_level": 1, "page_idx": 0, "bbox": [33, 64, 420, 107]},
                {"type": "text", "text": "Detail A1", "page_idx": 0, "bbox": [60, 340, 300, 375]},
                {"type": "text", "text": "Topic B | Overview Two", "text_level": 1, "page_idx": 1, "bbox": [33, 66, 430, 107]},
                {"type": "text", "text": "Detail B1", "page_idx": 1, "bbox": [60, 340, 300, 375]},
                {"type": "text", "text": "Topic C | Overview Three", "text_level": 1, "page_idx": 2, "bbox": [34, 65, 440, 108]},
                {"type": "text", "text": "Detail C1", "page_idx": 2, "bbox": [60, 340, 300, 375]},
                # Page 4: unmarked top banner + trailing subsection
                {"type": "text", "text": "Topic D | Knowledge Base", "page_idx": 3, "bbox": [33, 66, 410, 107]},
                {"type": "text", "text": "Connect External Systems", "text_level": 1, "page_idx": 3, "bbox": [60, 340, 420, 375]},
                {"type": "text", "text": "Open API integration for frontline staff.", "page_idx": 3, "bbox": [60, 410, 540, 460]},
                # Page 5: unmarked top banner + badge icon + badge title (no native text inside badge) + capability items
                {"type": "text", "text": "Topic E | Smart Assistant", "page_idx": 4, "bbox": [34, 66, 475, 107]},
                {"type": "image", "img_path": "", "page_idx": 4, "bbox": [38, 192, 95, 294]},
                {"type": "text", "text": "Smart Assistant Pro", "text_level": 1, "page_idx": 4, "bbox": [105, 195, 252, 288]},
                {"type": "text", "text": "Preset Prompts", "page_idx": 4, "bbox": [60, 340, 200, 375]},
                {"type": "text", "text": "Custom Commands", "page_idx": 4, "bbox": [60, 420, 220, 455]},
                {"type": "text", "text": "Read Messages", "page_idx": 4, "bbox": [60, 500, 200, 535]},
                {"type": "text", "text": "Check Mail", "page_idx": 4, "bbox": [60, 580, 180, 615]},
            ]
            with patch.object(mineru_pdf, "call_mineru_sync", return_value={"content_list": content}):
                blocks = mineru_pdf.mineru_pdf_blocks(
                    mineru_url="http://x", path=pdf_path, cache_dir=Path(tmp) / "cache"
                )
            merged = merge_mineru_text_blocks(blocks, target_tokens=800)
            chunks = blocks_to_chunks(
                kb_id="kb", file_key=1, content_version="v1", parser_profile="p",
                blocks=merged, max_tokens=400, overlap_tokens=40,
            )

            p5_blocks = [b for b in blocks if b.page_idx == 5]
            by_text = {b.text: b for b in p5_blocks}
            self.assertEqual(by_text["Topic E | Smart Assistant"].block_type, "title")
            self.assertEqual(by_text["Smart Assistant Pro"].block_type, "title")
            for cap in ("Preset Prompts", "Custom Commands", "Read Messages", "Check Mail"):
                self.assertEqual(by_text[cap].metadata["section_path"], ["Smart Assistant Pro"])
            # Neither final chunks nor paragraph spans on page 5 may reference page 4's trailing subsection
            for b in p5_blocks:
                self.assertNotIn("Connect External Systems", b.metadata.get("section_path") or [])
            for ch in chunks:
                for sp in (ch.block.metadata.get("section_spans") or []):
                    pages = {r.get("page") for r in (sp.get("source_regions") or [])}
                    if 5 in pages:
                        self.assertNotIn("Connect External Systems", sp.get("section_path") or [])
                if ch.block.metadata.get("page_start") == 5:
                    self.assertNotIn("Connect External Systems", ch.block.metadata.get("section_path") or [])

    def test_in_image_card_headings_preserve_local_attribution_without_leaking_outside(self) -> None:
        """An infographic card strip containing 3 Challenge-Solution pairs retains per-card local attribution
        nested under the outer section, stays packed without forced 6-chunk splitting, and restores the outer
        section cleanly once the image ends."""
        import pymupdf

        img_bytes = _png_bytes(400, 220)
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "case.pdf"
            doc = pymupdf.open()
            pg = doc.new_page(width=600, height=800)
            pg.insert_text((45, 50), "Company Overview", fontsize=16)
            pg.insert_text((45, 90), "An enterprise manufacturer deploying modern office workflows. " * 4, fontsize=11)
            # Embedded infographic card strip: x=48..360, y=240..440 (norm x=80..600, y=300..550)
            pg.insert_image(pymupdf.Rect(48, 240, 360, 440), stream=img_bytes)
            pg.insert_text((45, 510), "Post-card paragraph belonging to Company Overview.", fontsize=11)
            pg.insert_text((45, 580), "Detailed Architecture", fontsize=16)
            pg.insert_text((45, 620), "Architecture body text.", fontsize=11)
            doc.save(str(pdf_path))
            doc.close()

            content = [
                {"type": "text", "text": "Company Overview", "text_level": 1, "page_idx": 0, "bbox": [75, 45, 260, 75]},
                {"type": "text", "text": "An enterprise manufacturer deploying modern office workflows.", "page_idx": 0, "bbox": [75, 100, 560, 160]},
                # 3 Challenge-Solution cards inside embedded image [80..600, 300..550]
                {"type": "text", "text": "Challenge 1", "text_level": 1, "page_idx": 0, "bbox": [95, 315, 180, 332]},
                {"type": "text", "text": "Projects lack real-time progress tracking.", "page_idx": 0, "bbox": [95, 338, 300, 365]},
                {"type": "text", "text": "Solution", "text_level": 1, "page_idx": 0, "bbox": [350, 315, 430, 332]},
                {"type": "text", "text": "Collect daily updates with structured tables.", "page_idx": 0, "bbox": [350, 338, 560, 365]},
                {"type": "text", "text": "Challenge 2", "text_level": 1, "page_idx": 0, "bbox": [95, 395, 180, 412]},
                {"type": "text", "text": "Invoice payment status updates are delayed.", "page_idx": 0, "bbox": [95, 418, 300, 445]},
                {"type": "text", "text": "Solution", "text_level": 1, "page_idx": 0, "bbox": [350, 395, 430, 412]},
                {"type": "text", "text": "Automate collection with smart forms.", "page_idx": 0, "bbox": [350, 418, 560, 445]},
                {"type": "text", "text": "Challenge 3", "text_level": 1, "page_idx": 0, "bbox": [95, 475, 180, 492]},
                {"type": "text", "text": "Ad-hoc orders are tracked manually.", "page_idx": 0, "bbox": [95, 498, 300, 525]},
                {"type": "text", "text": "Solution", "text_level": 1, "page_idx": 0, "bbox": [350, 475, 430, 492]},
                {"type": "text", "text": "Centralize order reminders in one table.", "page_idx": 0, "bbox": [350, 498, 560, 525]},
                # Outside image: must restore outer section ["Company Overview"] without leaking Challenge 3 / Solution!
                {"type": "text", "text": "Post-card paragraph belonging to Company Overview.", "page_idx": 0, "bbox": [75, 620, 540, 655]},
                {"type": "text", "text": "Detailed Architecture", "text_level": 1, "page_idx": 0, "bbox": [75, 710, 300, 740]},
                {"type": "text", "text": "Architecture body text.", "page_idx": 0, "bbox": [75, 760, 450, 795]},
            ]
            with patch.object(mineru_pdf, "call_mineru_sync", return_value={"content_list": content}):
                blocks = mineru_pdf.mineru_pdf_blocks(
                    mineru_url="http://x", path=pdf_path, cache_dir=Path(tmp) / "cache"
                )
            by_text = {b.text: b for b in blocks}
            self.assertEqual(by_text["Projects lack real-time progress tracking."].metadata["section_path"],
                             ["Company Overview", "Challenge 1"])
            self.assertEqual(by_text["Collect daily updates with structured tables."].metadata["section_path"],
                             ["Company Overview", "Challenge 1", "Solution"])
            self.assertEqual(by_text["Invoice payment status updates are delayed."].metadata["section_path"],
                             ["Company Overview", "Challenge 2"])
            self.assertEqual(by_text["Automate collection with smart forms."].metadata["section_path"],
                             ["Company Overview", "Challenge 2", "Solution"])
            self.assertEqual(by_text["Ad-hoc orders are tracked manually."].metadata["section_path"],
                             ["Company Overview", "Challenge 3"])
            self.assertEqual(by_text["Centralize order reminders in one table."].metadata["section_path"],
                             ["Company Overview", "Challenge 3", "Solution"])
            # Strictly scoped: post-card prose returns to outer section
            self.assertEqual(by_text["Post-card paragraph belonging to Company Overview."].metadata["section_path"],
                             ["Company Overview"])
            # Merged into 1 chunk (not forcibly split into 6 chunks)
            merged = merge_mineru_text_blocks(blocks, target_tokens=800)
            chunks = blocks_to_chunks(
                kb_id="kb", file_key=2, content_version="v1", parser_profile="p",
                blocks=merged, max_tokens=400, overlap_tokens=40,
            )
            self.assertEqual(len(chunks), 1)
            self.assertIn(["Company Overview", "Challenge 2", "Solution"], chunks[0].block.metadata["section_paths"])

    def test_sidebar_mini_toc_ui_form_dialog_and_captioned_figure_stay_demoted_while_continuation_preserved(self) -> None:
        """Negative controls: (1) UI form screenshot with required '*' labels and back arrow stays under outer
        section; (2) webpage screenshot with sidebar mini-TOC never promotes sidebar TOC heading and never
        corrupts subsequent numbered chapters; (3) captioned example screenshot stays demoted; (4) real
        cross-page prose continuation in a portrait manual keeps inheriting the previous page's section."""
        import pymupdf

        img_bytes = _png_bytes(400, 300)
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "manual.pdf"
            doc = pymupdf.open()
            # Page 1: Chapter 4 + 4.1 prose that continues onto Page 2
            p1 = doc.new_page(width=600, height=800)
            p1.insert_text((45, 60), "4. Directory Sync", fontsize=16)
            p1.insert_text((45, 110), "4.1 Setup Overview", fontsize=14)
            p1.insert_text((45, 150), "First part of the directory synchronization guide. " * 6, fontsize=11)
            # Page 2: Continuation prose at top + webpage screenshot with sidebar mini-TOC + native 4.2 below
            p2 = doc.new_page(width=600, height=800)
            p2.insert_text((45, 55), "Continuation sentence from section 4.1 on the next page.", fontsize=11)
            p2.insert_image(pymupdf.Rect(30, 120, 510, 360), stream=img_bytes, keep_proportion=False)
            p2.insert_text((45, 420), "4.2 Sync Members", fontsize=14)
            # UI form dialog screenshot on lower half
            p2.insert_image(pymupdf.Rect(30, 460, 420, 680), stream=img_bytes, keep_proportion=False)
            p2.insert_text((45, 720), "5. Maintenance", fontsize=16)
            p2.insert_text((45, 760), "Routine maintenance instructions.", fontsize=11)
            # Page 3: Captioned example figure
            p3 = doc.new_page(width=600, height=800)
            p3.insert_image(pymupdf.Rect(60, 80, 520, 400), stream=img_bytes, keep_proportion=False)
            p3.insert_text((60, 425), "Figure 1: Example report preview", fontsize=11)
            p3.insert_text((60, 470), "Post-figure body still under Maintenance.", fontsize=11)
            doc.save(str(pdf_path))
            doc.close()

            content = [
                {"type": "text", "text": "4. Directory Sync", "text_level": 1, "page_idx": 0, "bbox": [75, 65, 300, 92]},
                {"type": "text", "text": "4.1 Setup Overview", "text_level": 1, "page_idx": 0, "bbox": [75, 125, 300, 150]},
                {"type": "text", "text": "First part of the directory synchronization guide.", "page_idx": 0, "bbox": [75, 180, 560, 220]},
                # Page 2 top continuation prose
                {"type": "text", "text": "Continuation sentence from section 4.1 on the next page.", "page_idx": 1, "bbox": [75, 58, 560, 82]},
                # Inside webpage screenshot [50..850, 150..450]: main pane + right sidebar mini-TOC
                {"type": "text", "text": "1. Create Dept", "text_level": 1, "page_idx": 1, "bbox": [160, 165, 320, 185]},
                {"type": "text", "text": "POST /v1/depts creates a department.", "page_idx": 1, "bbox": [160, 195, 520, 220]},
                {"type": "text", "text": "Page Outline", "text_level": 1, "page_idx": 1, "bbox": [660, 165, 760, 178]},
                {"type": "text", "text": "1. Create Dept", "page_idx": 1, "bbox": [665, 185, 790, 198]},
                {"type": "text", "text": "2. Update Dept", "page_idx": 1, "bbox": [665, 205, 790, 218]},
                {"type": "text", "text": "3. Delete Dept", "page_idx": 1, "bbox": [665, 225, 790, 238]},
                {"type": "text", "text": "4. Add Member", "page_idx": 1, "bbox": [665, 245, 790, 258]},
                # Native 4.2 heading must remain under "4. Directory Sync", NOT hijacked by "1. Create Dept"
                {"type": "text", "text": "4.2 Sync Members", "text_level": 1, "page_idx": 1, "bbox": [75, 515, 300, 542]},
                # UI form dialog screenshot [50..700, 575..850]
                {"type": "text", "text": "Create Connector", "text_level": 1, "page_idx": 1, "bbox": [80, 595, 240, 612]},
                {"type": "text", "text": "<- Back to templates", "text_level": 1, "page_idx": 1, "bbox": [80, 625, 260, 642]},
                {"type": "text", "text": "* Connector Name", "text_level": 1, "page_idx": 1, "bbox": [80, 660, 240, 678]},
                {"type": "text", "text": "Enter name", "page_idx": 1, "bbox": [90, 685, 240, 700]},
                {"type": "text", "text": "* Client ID", "page_idx": 1, "bbox": [80, 715, 200, 732]},
                {"type": "text", "text": "Description", "text_level": 1, "page_idx": 1, "bbox": [80, 748, 200, 765]},
                {"type": "text", "text": "Optional description", "page_idx": 1, "bbox": [90, 772, 260, 788]},
                # Native 5. heading
                {"type": "text", "text": "5. Maintenance", "text_level": 1, "page_idx": 1, "bbox": [75, 888, 280, 915]},
                {"type": "text", "text": "Routine maintenance instructions.", "page_idx": 1, "bbox": [75, 938, 460, 965]},
                # Page 3: Captioned figure [100..867, 100..500]
                {"type": "text", "text": "Executive Summary", "text_level": 1, "page_idx": 2, "bbox": [140, 140, 380, 165]},
                {"type": "text", "text": "Embedded report text inside screenshot.", "page_idx": 2, "bbox": [140, 180, 800, 260]},
                {"type": "text", "text": "Figure 1: Example report preview", "page_idx": 2, "bbox": [100, 520, 480, 545]},
                {"type": "text", "text": "Post-figure body still under Maintenance.", "page_idx": 2, "bbox": [100, 575, 560, 605]},
            ]
            with patch.object(mineru_pdf, "call_mineru_sync", return_value={"content_list": content}):
                blocks = mineru_pdf.mineru_pdf_blocks(
                    mineru_url="http://x", path=pdf_path, cache_dir=Path(tmp) / "cache"
                )
            by_text = {b.text: b for b in blocks}
            # 1) Real cross-page continuation preserved
            self.assertEqual(by_text["Continuation sentence from section 4.1 on the next page."].metadata["section_path"],
                             ["4. Directory Sync", "4.1 Setup Overview"])
            # 2) Sidebar mini-TOC never enters section_path and stays searchable
            self.assertEqual(by_text["Page Outline"].block_type, "text")
            for b in blocks:
                self.assertNotIn("Page Outline", b.metadata.get("section_path") or [])
            # 3) Outer numbered outline completely unpolluted by in-screenshot "1. Create Dept"
            self.assertEqual(by_text["4.2 Sync Members"].metadata["section_path"], ["4. Directory Sync", "4.2 Sync Members"])
            self.assertEqual(by_text["5. Maintenance"].metadata["section_path"], ["5. Maintenance"])
            # 4) UI form dialog controls demoted to searchable body under 4.2
            for ctrl in ("Create Connector", "<- Back to templates", "Description"):
                self.assertEqual(by_text[ctrl].block_type, "text")
                self.assertEqual(by_text[ctrl].metadata["section_path"], ["4. Directory Sync", "4.2 Sync Members"])
            # 5) Captioned figure screenshot heading demoted; post-figure body stays under 5. Maintenance
            self.assertEqual(by_text["Executive Summary"].block_type, "text")
            self.assertEqual(by_text["Post-figure body still under Maintenance."].metadata["section_path"], ["5. Maintenance"])


class Fix2MergeDirectionTableTests(unittest.TestCase):
    def test_horizontal_colspan_expressed_once_with_covered_columns_and_no_false_splitting(self) -> None:
        note = (
            "Provided by customer third-party model; if unavailable, an open-source model may be selected "
            "and the customer prepares dedicated compute resources after pre-sales sizing."
        )
        html = (
            "<table>"
            "<tr><th>Model</th><th>x86 Package</th><th>ARM Package</th><th>GPU-A Count</th><th>GPU-B Count</th><th>NPU Count</th></tr>"
            "<tr><td>Built-in</td><td>pkg_x86.tgz</td><td>pkg_arm.tgz</td><td>4</td><td>2</td><td>2</td></tr>"
            f"<tr><td>Customer LLM</td><td colspan='5'>{note}</td></tr>"
            "</table>"
        )
        md = html_table_to_markdown(html)
        self.assertIsNotNone(md)
        self.assertEqual(md.count(note), 1)
        self.assertIn("[COLS: x86 Package / ARM Package / GPU-A Count / GPU-B Count / NPU Count] " + note, md)
        # Fits cleanly in 1 chunk under 400 tokens (instead of splitting into 3 duplicated chunks)
        chunks = split_table_text(md, 400)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].count(note), 1)
        # When forced to split row-by-cell under a tight budget, applicable columns become the field label once
        tight_chunks = split_table_text(md, 90)
        joined = "\n".join(tight_chunks)
        self.assertIn("x86 Package / ARM Package / GPU-A Count / GPU-B Count / NPU Count: " + note[:40], joined)
        self.assertNotIn("GPU-A Count: " + note[:20], joined)
        self.assertNotIn("[COLS:", joined)

    def test_combined_rowspan_and_colspan_repeats_vertically_once_per_row_never_horizontally(self) -> None:
        shared = "Shared deployment policy across tenant and overseas region."
        html = (
            "<table>"
            "<tr><th>Region</th><th>Country</th><th>Tenant</th><th>Data Node</th></tr>"
            f"<tr><td rowspan='2'>APAC</td><td>Country A</td><td rowspan='2' colspan='2'>{shared}</td></tr>"
            "<tr><td>Country B</td></tr>"
            "</table>"
        )
        md = html_table_to_markdown(html)
        self.assertIsNotNone(md)
        lines = md.splitlines()
        self.assertEqual(len(lines), 4)
        # Vertical rowspan: appears on BOTH Country A and Country B rows; horizontal colspan: once per row!
        self.assertEqual(lines[2].count(shared), 1)
        self.assertEqual(lines[3].count(shared), 1)
        self.assertIn(f"| APAC | Country A | [COLS: Tenant / Data Node] {shared} |  |", lines[2])
        self.assertIn(f"| APAC | Country B | [COLS: Tenant / Data Node] {shared} |  |", lines[3])

        # HTML DOM parity
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "mixed.html"
            p.write_text(f"<html><body>{html}</body></html>", encoding="utf-8")
            blocks = parse_html_dom(p)
        tbl = [b for b in blocks if b.block_type == "table"][0]
        self.assertEqual(tbl.table_markdown, md)

    def test_identical_adjacent_independent_cells_are_never_deduplicated_and_same_header_colspan_stays_clean(self) -> None:
        same_long = "Identical independent specification text repeated intentionally across two separate columns."
        html_indep = (
            "<table>"
            "<tr><th>Item</th><th>Plan A</th><th>Plan B</th></tr>"
            f"<tr><td>SLA</td><td>{same_long}</td><td>{same_long}</td></tr>"
            "</table>"
        )
        md_indep = html_table_to_markdown(html_indep)
        # Two independent cells (colspan=1 each) MUST both remain verbatim!
        self.assertEqual(md_indep.count(same_long), 2)
        self.assertNotIn("[COLS:", md_indep)

        # Body colspan under a single group header name needs no redundant [COLS: ...] prefix
        html_group = (
            "<table>"
            "<tr><th>Phase</th><th colspan='3'>Core Actions</th></tr>"
            "<tr><td rowspan='2'>Prep</td><td colspan='3'>Define criteria and benchmark metrics.</td></tr>"
            "<tr><td>Sub1</td><td>Sub2</td><td>Sub3</td></tr>"
            "</table>"
        )
        md_group = html_table_to_markdown(html_group)
        self.assertEqual(md_group.count("Define criteria and benchmark metrics."), 1)
        self.assertNotIn("[COLS:", md_group)
        self.assertIn("| Prep | Define criteria and benchmark metrics. |  |  |", md_group)

    def test_empty_tr_does_not_corrupt_spans_and_table_repair_handles_colspan_prefixes(self) -> None:
        html = (
            "<table>"
            "<tr><th>Col A</th><th>Col B</th><th>Col C</th></tr>"
            "<tr><td colspan='3'></td></tr>"
            "<tr><td>R1</td><td colspan='2'>Shared note for B and C</td></tr>"
            "</table>"
        )
        spans = html_table_spans(html)
        # Empty row dropped -> remaining body row is at index 1, not shifted or duplicated
        self.assertEqual(spans, [(1, 1, 1, 2)])
        grid = html_table_to_grid(html)
        self.assertEqual(grid, [["Col A", "Col B", "Col C"], ["R1", "Shared note for B and C", ""]])
        md = html_table_to_markdown(html)
        self.assertIn("| R1 | [COLS: Col B / Col C] Shared note for B and C |  |", md)
        stripped = strip_colspan_prefixes(
            [["Col A", "Col B", "Col C"], ["R1", "[COLS: Col B / Col C] Shared note for B and C", ""]],
            spans,
        )
        self.assertEqual(stripped, grid)

        # _expand_review_spans expands Row 0 colspan + vertical rowspan, never duplicates body colspan horizontally
        html_mix = (
            "<table>"
            "<tr><th>K</th><th colspan='2'>Grp</th></tr>"
            "<tr><td rowspan='2'>A</td><td colspan='2'>Wide</td></tr>"
            "<tr><td>x</td><td>y</td></tr>"
            "</table>"
        )
        old = html_table_to_grid(html_mix)
        new = [["K", "Grp", ""], ["A", "Wide", ""], ["", "x", "y"]]
        expanded = _expand_review_spans(old, new, html_mix)
        self.assertEqual(expanded, [["K", "Grp", "Grp"], ["A", "Wide", ""], ["A", "x", "y"]])

        # End-to-end review_pdf_tables strips [COLS:] before comparing and re-applies it on accepted markdown
        html_ocr_err = (
            "<table>"
            "<tr><th>Col A</th><th>Col B</th><th>Col C</th></tr>"
            "<tr><td>R1</td><td colspan='2'>Shared note for B and X</td></tr>"
            "</table>"
        )
        md_ocr_err = html_table_to_markdown(html_ocr_err)
        with tempfile.TemporaryDirectory() as tmp:
            img = Path(tmp) / "tbl.png"
            img.write_bytes(_png_bytes(100, 60))
            block = ParsedBlock(
                parser="mineru",
                parser_profile="p",
                doc_type="pdf",
                block_type="table",
                text=md_ocr_err,
                table_markdown=md_ocr_err,
                block_id="t1",
                page_idx=1,
                metadata={
                    "source_item": {"table_body": html_ocr_err},
                    "table_review_reasons": ["native_text_missing"],
                    "table_missing_spans": ["verified"],
                    "table_source_text": "Col A Col B Col C R1 Shared note for B and C verified",
                    "table_native_spans": [],
                    "table_review_image": str(img),
                },
            )
            with patch(
                "kb_pipeline.parsers.table_repair.transcribe_table",
                return_value={"rows": [["Col A", "Col B", "Col C"], ["R1", "Shared note for B and C verified", ""]], "legend": []},
            ):
                stats = review_pdf_tables([block], base_url="http://x", api_key="k", model_id="m", cache_dir=Path(tmp))
            self.assertEqual(stats["accepted"], 1)
            self.assertIn("[COLS: Col B / Col C] Shared note for B and C verified", block.table_markdown)
            self.assertEqual(block.table_markdown.count("Shared note for B and C verified"), 1)

    def test_full_row_span_omits_cols_prefix_and_populated_sibling_never_gets_stale_cols(self) -> None:
        from kb_pipeline.parsers.common import grid_to_markdown

        # Full-table-width divider / note row omits noisy [COLS: all headers] while still expressing text once
        html_full = (
            "<table>"
            "<tr><th>Plan</th><th>Basic</th><th>Standard</th><th>Enterprise</th></tr>"
            "<tr><td colspan='4'>Add-on Benefits</td></tr>"
            "<tr><td>Storage</td><td colspan='3'>Shared cloud storage quota applies across tiers</td></tr>"
            "</table>"
        )
        md = html_table_to_markdown(html_full)
        self.assertIn("| Add-on Benefits |  |  |  |", md)
        self.assertNotIn("[COLS: Plan / Basic", md)
        self.assertIn("[COLS: Basic / Standard / Enterprise] Shared cloud storage quota applies across tiers", md)

        # Populated sibling column never receives a bogus [COLS: ...] even if stale spans are passed
        repaired_grid = [
            ["Dimension", "Phase 1", "Phase 2"],
            ["Knowledge", "Personal experience", "Enterprise KB"],
        ]
        stale_4col_spans = [(0, 0, 1, 2), (1, 0, 1, 2)]
        md_safe = grid_to_markdown(repaired_grid, spans=stale_4col_spans)
        self.assertNotIn("[COLS:", md_safe)
        self.assertIn("| Knowledge | Personal experience | Enterprise KB |", md_safe)

    def test_headerless_continuation_table_rerenders_colspans_with_parent_header(self) -> None:
        from kb_pipeline.parsers.pdf_enhanced import merge_split_tables

        html_p1 = (
            "<table>"
            "<tr><th>Component</th><th>x86</th><th>ARM</th><th>GPU</th></tr>"
            "<tr><td>OCR</td><td>pkg-x86</td><td>pkg-arm</td><td>1</td></tr>"
            "</table>"
        )
        html_p2 = (
            "<table>"
            "<tr><td>FirstCont</td><td colspan='3'>Row0 shared note on page 2</td></tr>"
            "<tr><td>SecondCont</td><td colspan='3'>Body shared note on page 2</td></tr>"
            "</table>"
        )
        md1 = html_table_to_markdown(html_p1)
        md2 = html_table_to_markdown(html_p2)
        b1 = ParsedBlock(
            parser="mineru",
            parser_profile="p",
            doc_type="pdf",
            block_type="table",
            text=md1,
            table_markdown=md1,
            block_id="t1",
            page_idx=1,
            bbox=[50.0, 720.0, 950.0, 950.0],
            metadata={"section_path": ["Specs"], "source_item": {"table_body": html_p1}},
        )
        b2 = ParsedBlock(
            parser="mineru",
            parser_profile="p",
            doc_type="pdf",
            block_type="table",
            text=md2,
            table_markdown=md2,
            block_id="t2",
            page_idx=2,
            bbox=[50.0, 60.0, 950.0, 260.0],
            metadata={"section_path": ["Specs"], "source_item": {"table_body": html_p2}},
        )
        merged = merge_split_tables([b1, b2])
        self.assertEqual(len(merged), 1)
        out = merged[0].table_markdown
        # Row 0 of page 2 must NOT duplicate "Row0 shared note on page 2" 3x, and both rows use real parent headers
        self.assertEqual(out.count("Row0 shared note on page 2"), 1)
        self.assertIn("[COLS: x86 / ARM / GPU] Row0 shared note on page 2", out)
        self.assertEqual(out.count("Body shared note on page 2"), 1)
        self.assertIn("[COLS: x86 / ARM / GPU] Body shared note on page 2", out)
        self.assertNotIn("FirstCont", out.split("Body shared note")[0].split("SecondCont")[-1])


class Fix1AdversarialEdgeCaseTests(unittest.TestCase):
    def test_native_heading_over_background_image_never_demoted_and_row_major_cards_keep_column_owner(self) -> None:
        """Guards against HIGH-1 (native title over decorative background image + OCR badge) and HIGH-2
        (row-major interleaved multi-column cards T1, T2, T3, B1, B2, B3)."""
        import pymupdf

        img_bytes = _png_bytes(400, 300)
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "adv_layout.pdf"
            doc = pymupdf.open()
            # Page 1: Decorative background covering top half + native title + native body + small OCR status badge
            p1 = doc.new_page(width=600, height=800)
            p1.insert_image(pymupdf.Rect(30, 0, 560, 420), stream=img_bytes, keep_proportion=False)
            p1.insert_text((45, 70), "Vitamin A Metabolism", fontsize=16)
            p1.insert_text((310, 175), "Your baseline capacity is normal and well balanced.", fontsize=11)
            p1.insert_text((45, 550), "Health Advice", fontsize=14)
            p1.insert_text((45, 600), "Eat carrots and leafy greens.", fontsize=11)
            # Page 2: Row-major 3-card infographic (T1, T2, T3 on same y, then B1, B2, B3 below)
            p2 = doc.new_page(width=600, height=800)
            p2.insert_text((45, 60), "Three Pillars", fontsize=16)
            p2.insert_image(pymupdf.Rect(30, 120, 570, 560), stream=img_bytes, keep_proportion=False)
            p2.insert_text((45, 620), "Closing Summary", fontsize=14)
            p2.insert_text((45, 660), "After cards closing text.", fontsize=11)
            doc.save(str(pdf_path))
            doc.close()

            content = [
                # Page 1: native title over background image + OCR badge "Normal" + native prose
                {"type": "text", "text": "Vitamin A Metabolism", "text_level": 1, "page_idx": 0, "bbox": [75, 72, 420, 102]},
                {"type": "text", "text": "Normal", "text_level": 1, "page_idx": 0, "bbox": [515, 165, 635, 192]},
                {"type": "text", "text": "Your baseline capacity is normal and well balanced.", "page_idx": 0, "bbox": [515, 205, 900, 260]},
                {"type": "text", "text": "Health Advice", "text_level": 1, "page_idx": 0, "bbox": [75, 670, 240, 700]},
                {"type": "text", "text": "Eat carrots and leafy greens.", "page_idx": 0, "bbox": [75, 730, 480, 770]},
                # Page 2: row-major 3 side-by-side cards inside image [50..950, 150..700]
                {"type": "text", "text": "Three Pillars", "text_level": 1, "page_idx": 1, "bbox": [75, 60, 300, 88]},
                {"type": "text", "text": "Pillar Alpha", "text_level": 1, "page_idx": 1, "bbox": [80, 200, 260, 225]},
                {"type": "text", "text": "Pillar Beta", "text_level": 1, "page_idx": 1, "bbox": [380, 200, 560, 225]},
                {"type": "text", "text": "Pillar Gamma", "text_level": 1, "page_idx": 1, "bbox": [680, 200, 880, 225]},
                {"type": "text", "text": "Alpha body explains governance.", "page_idx": 1, "bbox": [80, 250, 310, 340]},
                {"type": "text", "text": "Beta body explains security.", "page_idx": 1, "bbox": [380, 250, 610, 340]},
                {"type": "text", "text": "Gamma body explains velocity.", "page_idx": 1, "bbox": [680, 250, 910, 340]},
                {"type": "text", "text": "Closing Summary", "text_level": 1, "page_idx": 1, "bbox": [75, 760, 300, 788]},
                {"type": "text", "text": "After cards closing text.", "page_idx": 1, "bbox": [75, 815, 420, 845]},
            ]
            with patch.object(mineru_pdf, "call_mineru_sync", return_value={"content_list": content}):
                blocks = mineru_pdf.mineru_pdf_blocks(
                    mineru_url="http://x", path=pdf_path, cache_dir=Path(tmp) / "cache"
                )
            by_text = {b.text: b for b in blocks}
            # HIGH-1: Native title "Vitamin A Metabolism" stays intact as title & outer section; OCR "Normal" demoted
            self.assertEqual(by_text["Vitamin A Metabolism"].block_type, "title")
            self.assertEqual(
                by_text["Your baseline capacity is normal and well balanced."].metadata["section_path"],
                ["Vitamin A Metabolism"],
            )
            self.assertEqual(by_text["Normal"].block_type, "text")
            # HIGH-2: Row-major 3-card infographic maps each body back to its own column heading
            self.assertEqual(by_text["Alpha body explains governance."].metadata["section_path"], ["Three Pillars", "Pillar Alpha"])
            self.assertEqual(by_text["Beta body explains security."].metadata["section_path"], ["Three Pillars", "Pillar Beta"])
            self.assertEqual(by_text["Gamma body explains velocity."].metadata["section_path"], ["Three Pillars", "Pillar Gamma"])
            self.assertEqual(by_text["After cards closing text."].metadata["section_path"], ["Closing Summary"])

    def test_ascii_back_chevron_horizontal_tabs_and_interleaved_sidebar_toc_stay_demoted(self) -> None:
        import pymupdf

        img_bytes = _png_bytes(400, 300)
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "ui_edge.pdf"
            doc = pymupdf.open()
            p1 = doc.new_page(width=600, height=800)
            p1.insert_text((45, 55), "3. Display Settings", fontsize=16)
            p1.insert_image(pymupdf.Rect(30, 100, 570, 380), stream=img_bytes, keep_proportion=False)
            p1.insert_image(pymupdf.Rect(30, 420, 570, 720), stream=img_bytes, keep_proportion=False)
            p1.insert_text((45, 755), "Native footer paragraph under Display Settings explaining how display customization works across every department view in the workspace.", fontsize=11)
            doc.save(str(pdf_path))
            doc.close()

            content = [
                {"type": "text", "text": "3. Display Settings", "text_level": 1, "page_idx": 0, "bbox": [75, 52, 340, 78]},
                # Image 1 [50..950, 125..475]: ASCII "< Back" chevron + horizontal tab bar over wide body
                {"type": "text", "text": "< Back Custom Rules", "text_level": 1, "page_idx": 0, "bbox": [80, 145, 320, 165]},
                {"type": "text", "text": "Tab One", "text_level": 1, "page_idx": 0, "bbox": [80, 190, 180, 210]},
                {"type": "text", "text": "Tab Two", "text_level": 1, "page_idx": 0, "bbox": [220, 190, 320, 210]},
                {"type": "text", "text": "Tab Three", "text_level": 1, "page_idx": 0, "bbox": [360, 190, 460, 210]},
                {"type": "text", "text": "Wide settings panel body spanning all three tabs across the screenshot.", "page_idx": 0, "bbox": [80, 240, 860, 340]},
                # Image 2 [50..950, 525..900]: y-interleaved main pane + sidebar mini-TOC with marked numbered entry
                {"type": "text", "text": "Page Outline", "text_level": 1, "page_idx": 0, "bbox": [700, 545, 860, 565]},
                {"type": "text", "text": "1. Alpha", "page_idx": 0, "bbox": [705, 575, 850, 592]},
                {"type": "text", "text": "Main pane paragraph interleaved by y.", "page_idx": 0, "bbox": [80, 585, 560, 620]},
                {"type": "text", "text": "2. Beta", "text_level": 1, "page_idx": 0, "bbox": [705, 602, 850, 618]},
                {"type": "text", "text": "3. Gamma", "page_idx": 0, "bbox": [705, 628, 850, 644]},
                {"type": "text", "text": "Native footer paragraph under Display Settings explaining how display customization works across every department view in the workspace.", "page_idx": 0, "bbox": [75, 930, 920, 968]},
            ]
            with patch.object(mineru_pdf, "call_mineru_sync", return_value={"content_list": content}):
                blocks = mineru_pdf.mineru_pdf_blocks(
                    mineru_url="http://x", path=pdf_path, cache_dir=Path(tmp) / "cache"
                )
            by_text = {b.text: b for b in blocks}
            for ctrl in ("< Back Custom Rules", "Tab One", "Tab Two", "Tab Three", "Page Outline", "2. Beta"):
                self.assertEqual(by_text[ctrl].block_type, "text", f"{ctrl} should be demoted")
            for b in blocks:
                for ctrl in ("< Back Custom Rules", "Tab One", "Tab Two", "Tab Three", "Page Outline", "2. Beta"):
                    self.assertNotIn(ctrl, b.metadata.get("section_path") or [])


class ReviewScopeRegressionTests(unittest.TestCase):
    def test_unmarked_data_and_declared_column_names_survive_both_table_routes(self):
        from kb_pipeline.chunking.chunker import split_table_text

        # The last row is one physical cell across columns 2-4 in each layout.
        tail = '<tr><td>07</td><td colspan="3">Independent case</td></tr>'
        cases = [
            ("<tr><td>01</td><td>One</td><td>02</td><td>Two</td></tr>", "2-4"),
            ("<tr><td>Owner:</td><td>Example</td><td>Ref:</td><td>Other</td></tr>", "2-4"),
            ("<tr><th scope='row'>Owner</th><td>Kim</td><td>Office</td><td>West</td></tr>", "2-4"),
            ("<tr><td>Item</td><td>East</td><td>West</td><td>South</td></tr>", "East / West / South"),
            ("<thead><tr><td>Item</td><td>2024</td><td>2025</td><td>v3.2</td></tr></thead>", "2024 / 2025 / v3.2"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for first, label in cases:
                with self.subTest(first=first):
                    html = f"<table>{first}{tail}</table>"
                    path = Path(tmp) / "input.html"
                    path.write_text(html)
                    mineru = html_table_to_markdown(html)
                    native = next(b for b in parse_html_dom(path) if b.block_type == "table")
                    for text in (mineru, native.table_markdown):
                        self.assertEqual(text.count("Independent case"), 1)
                        self.assertIn(f"[COLS: {label}] Independent case", text)
                    long_table = mineru.replace("Independent case", "Independent case " + "detail " * 100)
                    pieces = split_table_text(long_table, 35)
                    self.assertGreater(len(pieces), 1)
                    joined = "\n".join(pieces)
                    if label == "2-4":
                        self.assertIn(mineru.splitlines()[0], joined)
                        self.assertIn("COLS 2-4: Independent case", joined)
                        self.assertIn("列1: 07", joined)
                    else:
                        self.assertIn(f"{label}: Independent case", joined)

        # A record value in a merged first-row cell is printed once, including after VLM grid repair.
        form = "<table><tr><td>Owner:</td><td colspan='2'>Example</td><td>Reference:</td><td>00123456</td></tr></table>"
        row = html_table_to_grid(form)[0]
        self.assertEqual(row, ["Owner:", "Example", "", "Reference:", "00123456"])
        repeated = [row.copy()]
        repeated[0][2] = "Example"
        self.assertEqual(_expand_review_spans([row], repeated, form), [row])

    def test_unconfirmed_numbered_region_ends_old_cards_without_stealing_neighbour_text(self):
        import pymupdf

        def item(text, box, level=None):
            data = {"type": "text", "text": text, "bbox": box, "page_idx": 0}
            if level is not None:
                data["text_level"] = level
            return data

        content = [
            item("Parent", [50, 28, 170, 60], 1),
            item("1. Left", [100, 240, 220, 265], 1),
            item("Left details", [100, 280, 295, 310]),
            item("2. Right", [530, 240, 670, 265], 1),
            item("Right details", [530, 280, 730, 310]),
            item("Offset right details", [800, 280, 920, 310]),
            item("3. Lower region", [100, 500, 235, 530], 1),
            item("Offset lower details", [550, 615, 835, 655]),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cards.pdf"
            with pymupdf.open() as doc:
                page = doc.new_page(width=1000, height=1000)
                page.insert_text((50, 50), "Parent", fontsize=18)
                page.insert_image(pymupdf.Rect(80, 200, 950, 740), stream=_png_bytes(), keep_proportion=False)
                page.insert_text((50, 830), "Ordinary native text outside the illustration. " * 4)
                doc.save(path)
            with patch.object(mineru_pdf, "call_mineru_sync", return_value={"content_list": content}):
                blocks = mineru_pdf.mineru_pdf_blocks(mineru_url="offline", path=path, cache_dir=Path(tmp)/"cache")
        paths = {b.text: b.metadata["section_path"] for b in blocks}
        self.assertEqual(paths["Left details"], ["Parent", "1. Left"])
        self.assertEqual(paths["Offset right details"], ["Parent", "2. Right"])
        self.assertEqual(paths["3. Lower region"], ["Parent"])
        self.assertEqual(paths["Offset lower details"], ["Parent"])


if __name__ == "__main__":
    unittest.main()
