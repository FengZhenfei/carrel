"""Fixes from the review of two trial knowledge bases (2026-11). Synthetic fixtures; no document content."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest

from kb_pipeline.chunking.diagnose import chunk_diagnostics
from kb_pipeline.headings import HeadingResolver
from kb_pipeline.models import ParsedBlock
from kb_pipeline.parsers.common import SectionTracker
from kb_pipeline.parsers.mineru_quality import recover_pdf_items
from kb_pipeline.vision import vlm


def walk(seq, **kw):
    """[(text, parser level)] -> [(final level, section path)], the way the parsers walk a document."""
    resolver = HeadingResolver((t for t, lv in seq if lv), **kw)
    sections = SectionTracker()
    out = []
    for text, parser_level in seq:
        level, _ = resolver.resolve(parser_level, text)
        sections.observe(level, text)
        out.append((level, list(sections.path)))
    return out


# ── A title that recurs is a sub-section, not a label ───────────────────────────────────────────────────────────

def test_a_recurring_title_never_replaces_the_section_above_it():
    """The parser marks every heading as level 1. Three projects, each with the same two sub-headings: as level-1
    headings they would replace the project title; as body text (the rule until now) they were no heading at all."""
    doc = []
    for project in ("项目一", "项目二", "项目三"):
        doc += [(project, 1), ("项目背景", 1), ("这个项目在一家工厂落地。", None), ("遇到的问题", 1), ("排产靠人工。", None)]
    got = walk(doc)
    assert [level for level, _ in got] == [1, 2, None, 2, None] * 3
    assert [path for _, path in got][:5] == [["项目一"], ["项目一", "项目背景"], ["项目一", "项目背景"],
                                             ["项目一", "遇到的问题"], ["项目一", "遇到的问题"]]
    assert got[-1][1] == ["项目三", "遇到的问题"]


def test_a_recurring_title_goes_below_the_nearest_heading_that_does_not_recur():
    """Whether it is a sibling or a child of that heading cannot be told from its recurring. Below keeps what the
    path held before and adds the title; beside would be right in a book and drop a correct parent elsewhere."""
    got = walk([("第1章 总则", 1), ("适用范围", 1), ("正文。", None), ("结论", 1), ("本章小结。", None),
                ("第2章 流程", 1), ("结论", 1), ("第3章 附则", 1), ("结论", 1)])
    assert got[1] == (2, ["第1章 总则", "适用范围"])                  # unnumbered, under the numbered chapter
    assert got[3] == (3, ["第1章 总则", "适用范围", "结论"])           # below the heading before it
    assert got[4][1] == ["第1章 总则", "适用范围", "结论"]             # its body is filed under it
    assert got[6] == (2, ["第2章 流程", "结论"]) and got[8] == (2, ["第3章 附则", "结论"])
    # the chapters are placed as before: a recurring title has no say in the numbering
    assert [got[i] for i in (0, 5, 7)] == [(1, ["第1章 总则"]), (1, ["第2章 流程"]), (1, ["第3章 附则"])]


def test_recurring_does_not_make_a_list_item_or_a_note_label_a_heading():
    resolver = HeadingResolver(["- 加粗的列表项"] * 3 + ["注释:"] * 3 + ["第1章 概述"])
    assert resolver.resolve(1, "第1章 概述")[0] == 1
    assert resolver.resolve(1, "- 加粗的列表项") == (None, False)
    assert resolver.resolve(1, "注释:") == (None, False)


def test_a_word_document_keeps_the_style_level_of_a_recurring_title():
    """In docx the levels come from Word styles. A sub-heading styled the same in every chapter was turned into
    body text there as well."""
    doc = []
    for chapter in ("第一章", "第二章", "第三章"):
        doc += [(chapter, 1), (f"{chapter}的背景", 2), ("小结", 2)]     # "小结" is styled as a sibling of the heading before it
    got = walk(doc + [("细节", 3)], trust_parser_levels=True)
    assert [level for level, _ in got] == [1, 2, 2] * 3 + [3]
    assert got[2][1] == ["第一章", "小结"] and got[-1][1] == ["第三章", "小结", "细节"]


def test_pdf_blocks_file_text_under_a_recurring_title():
    from kb_pipeline.parsers import mineru_pdf

    def item(text, page, level=None):
        return {"type": "text", "text": text, "page_idx": page, **({"text_level": level} if level else {})}

    content = []
    for page, chapter in enumerate(("第1章 总则", "第2章 流程", "第3章 附则")):      # one chapter per page
        content += [item(chapter, page, 1), item(f"{chapter}介绍一种做法。", page), item("结论", page, 1),
                    item(f"{chapter}的适用范围。", page)]
    with tempfile.TemporaryDirectory() as tmp, patch.object(mineru_pdf, "call_mineru_sync", return_value={"content_list": content}):
        blocks = mineru_pdf.mineru_pdf_blocks(mineru_url="http://x", path=Path(tmp) / "a.pdf", cache_dir=Path(tmp) / "c")
    assert [b.block_type for b in blocks] == ["title", "text", "title", "text"] * 3
    assert blocks[2].metadata["heading_level"] == 2
    assert blocks[3].metadata["section_path"] == ["第1章 总则", "结论"]
    assert blocks[-1].metadata["section_path"] == ["第3章 附则", "结论"]


# ── A footer that stands on one page only is content ────────────────────────────────────────────────────────────

def test_a_footer_that_stands_on_one_page_only_is_body_text():
    items = [
        {"type": "text", "text": "Body of page one.", "page_idx": 0, "bbox": [50, 100, 500, 400]},
        {"type": "image", "img_path": "figure.png", "page_idx": 0, "bbox": [50, 420, 500, 800]},
        {"type": "footer", "text": "Annual report 2026 · 1", "page_idx": 0, "bbox": [50, 950, 500, 970]},
        {"type": "text", "text": "Body of page two.", "page_idx": 1, "bbox": [50, 100, 500, 400]},
        {"type": "footer", "text": "Annual report 2026 · 2", "page_idx": 1, "bbox": [50, 950, 500, 970]},
        {"type": "footer", "text": "- 2 -", "page_idx": 1, "bbox": [250, 975, 300, 990]},
        {"type": "footer", "text": "12 / 2026", "page_idx": 1, "bbox": [400, 975, 500, 990]},
        {"type": "footer", "text": "Body of page two.", "page_idx": 1, "bbox": [50, 900, 500, 920]},
        # the layout model lists what it discards after the page body: a caption it took for a footer
        {"type": "footer", "text": "Figure 3: retrieval over a structured store", "page_idx": 0, "bbox": [50, 810, 500, 840]},
    ]
    fixed = recover_pdf_items(items, {})
    recovered = [x for x in fixed if x.get("original_source_type")]
    assert [(x["type"], x["original_source_type"], x["text"]) for x in recovered] == \
        [("text", "footer", "Figure 3: retrieval over a structured store")]
    assert [x.get("text") or x.get("img_path") for x in fixed][:3] == \
        ["Body of page one.", "figure.png", "Figure 3: retrieval over a structured store"]      # back under its figure
    # not content: a running footer whose page number changes, a page number, a line without a letter, a repeat of the body
    assert [x["type"] for x in fixed if x.get("text") != "Figure 3: retrieval over a structured store"] == \
        ["text", "image", "footer", "text", "footer", "footer", "footer", "footer"]
    assert items[-1]["type"] == "footer"                                    # the cached raw payload is not rewritten


def test_a_recovered_footer_reaches_the_blocks():
    from kb_pipeline.parsers import mineru_pdf

    content = [{"type": "text", "text": "The store answers structured queries.", "page_idx": 0, "bbox": [50, 100, 500, 400]},
               {"type": "footer", "text": "Source: internal survey of twelve teams", "page_idx": 0, "bbox": [50, 900, 500, 920]},
               {"type": "footer", "text": "Page 1 of 9", "page_idx": 0, "bbox": [50, 960, 500, 980]}]
    with tempfile.TemporaryDirectory() as tmp, patch.object(mineru_pdf, "call_mineru_sync", return_value={"content_list": content}):
        blocks = mineru_pdf.mineru_pdf_blocks(mineru_url="http://x", path=Path(tmp) / "a.pdf", cache_dir=Path(tmp) / "c")
    by_text = {b.text: b for b in blocks}
    assert by_text["Source: internal survey of twelve teams"].block_type == "text"
    assert by_text["Source: internal survey of twelve teams"].metadata["source_type"] == "text"
    assert by_text["Page 1 of 9"].metadata["source_type"] == "footer"       # still furniture, dropped by the merge step


# ── A picture whose answer was cut off or unreadable has no description: that is a failure ──────────────────────

EMPTY_ANSWER = {"kind": "other", "title": "", "summary": "", "text_verbatim": "", "entities": [], "facts": [],
                "keywords": [], "confidence": "low", "decorative": False}


def _answer(content, finish="stop"):
    return NS(choices=[NS(message=NS(content=content if isinstance(content, str) else json.dumps(content)), finish_reason=finish)], usage=None)


def _caption(tmp_path, answers, **kwargs):
    create = Mock(side_effect=answers)
    client = NS(chat=NS(completions=NS(create=create)))
    cache = tmp_path / "caption.json"
    with patch.object(vlm, "_client_for", return_value=client), patch.object(vlm, "image_data_url", return_value="data:,"):
        out = vlm.caption_image(image_path=tmp_path / "image.png", base_url="http://fixture/v1", api_key="fixture",
                                model_id="fixture", cache_json=cache, **kwargs)
    return out, create.call_count, cache.exists()


@pytest.mark.parametrize("answers,why", [
    ([_answer("", "length"), _answer("", "length")], "cut off at the output limit"),          # the limit was spent before any answer
    ([_answer('{"kind": "chart", "summary": "', "length")], "cut off at the output limit"),   # cut off before the first field
    ([_answer("I cannot help with that."), _answer("   ")], "could not be read"),             # prose, then blank
])
def test_no_description_from_an_incomplete_answer_is_a_failed_description(tmp_path, answers, why):
    """It used to be stored like a picture the model had nothing to say about, and the document went into the index
    with the picture undescribed."""
    out, calls, cached = _caption(tmp_path, answers)
    assert why in out["error"] and calls == len(answers) and not cached
    from kb_pipeline.parsers import visual_blocks
    block = ParsedBlock("fixture", "fixture", "pdf", "image", "", "image", page_idx=1, visual_ref=str(tmp_path / "image.png"))
    (tmp_path / "image.png").write_bytes(b"x")
    with patch.object(visual_blocks, "image_size", return_value=(800, 600)), patch.object(visual_blocks, "file_hash", return_value="0" * 64), \
            patch.object(visual_blocks, "caption_images_parallel", return_value={"image": out}):
        visual_blocks.enrich_blocks_with_vlm([block], base_url="fixture", api_key="fixture", model_id="fixture", cache_dir=tmp_path)
    assert block.metadata["vlm_status"] == "failed" and not block.metadata["decorative"]
    assert visual_blocks.vlm_failure_summary([block]) == {"visual_blocks": 1, "vlm_failed": 1}     # what holds the document back


@pytest.mark.parametrize("structured,calls", [(True, 1), (False, 2)])
def test_a_complete_answer_that_says_nothing_is_not_a_failure(tmp_path, structured, calls):
    """The model looked and had nothing to say (three of the four undescribed pictures in the trial were icons of
    about 50 pixels). Holding a document back for that would keep it out of the index for good."""
    out, sent, _ = _caption(tmp_path, [_answer(EMPTY_ANSWER), _answer(EMPTY_ANSWER)], structured=structured)
    assert "error" not in out and sent == calls
    assert not out["summary"] and not out["vlm_truncated"]


def test_a_cut_off_answer_that_still_describes_the_picture_is_kept(tmp_path):
    cut = json.dumps({"kind": "chart", "title": "", "summary": "A bar chart of yearly output.", "facts": ["2024: 12 units"]})[:-2]
    out, _, _ = _caption(tmp_path, [_answer(cut, "length"), _answer(cut, "length")])
    assert "error" not in out and out["summary"] == "A bar chart of yearly output." and out["vlm_truncated"]


def test_answer_is_readable_tells_an_empty_object_from_no_object():
    assert vlm.answer_is_readable(json.dumps(EMPTY_ANSWER)) and vlm.answer_is_readable("```json\n{}\n```")
    assert vlm.answer_is_readable('{"summary": "cut off he')                                  # repairable prefix
    assert not vlm.answer_is_readable("") and not vlm.answer_is_readable("   ") and not vlm.answer_is_readable("No picture found.")


# ── The chunk check says so when pictures are in the index without a description ────────────────────────────────

def test_pictures_left_without_a_description_are_reported(tmp_path):
    def picture(block_id, status, decorative=False):
        return ParsedBlock("fixture", "fixture", "pdf", "image", "", block_id, page_idx=1, visual_ref=str(tmp_path / "p.png"),
                           metadata={"vlm_status": status, "decorative": decorative})
    blocks = [picture("a", "empty"), picture("b", "empty"), picture("c", "empty", decorative=True), picture("d", "success"),
              ParsedBlock("fixture", "fixture", "pdf", "text", "body", "e", metadata={"vlm_status": "empty"})]
    diag = chunk_diagnostics([], max_tokens=400, blocks=blocks)
    assert diag["stats"]["visual_outputs_empty"] == 2
    assert {"key": "visual_output_empty", "params": [2]}.items() <= next(r for r in diag["reasons"] if r["key"] == "visual_output_empty").items()
    quiet = chunk_diagnostics([], max_tokens=400, blocks=blocks[2:])
    assert quiet["stats"]["visual_outputs_empty"] == 0 and "visual_output_empty" not in {r["key"] for r in quiet["reasons"]}
