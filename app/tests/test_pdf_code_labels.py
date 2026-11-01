"""Printed listing labels and missed boxes, using synthetic PDF content only."""
from copy import deepcopy
from unittest.mock import patch

import pytest

from kb_pipeline.parsers.pdf_code import repair_code_items
from kb_pipeline.parsers.common import item_text, caption_of
from kb_pipeline.parsers.mineru_pdf import mineru_pdf_blocks
from kb_pipeline.chunking.chunker import blocks_to_chunks, embedding_input
from kb_pipeline.graph.units import build_units, fetch_chunk_payloads


def item(text, box, kind="text", page=0, **extra):
    return {"type": kind, "text": text, "bbox": box, "page_idx": page, **extra}


def native_pdf(tmp_path, lines, font="cour"):
    fitz = pytest.importorskip("pymupdf")
    path = tmp_path / "example.pdf"
    with fitz.open() as doc:
        page = doc.new_page(width=600, height=800)
        page.insert_text((60, 90), "PYTHON", fontsize=8)
        page.draw_rect((55, 100, 560, 180), fill=(.95, .95, .95))
        for index, line in enumerate(lines):
            page.insert_text((60, 120 + 14*index), line, fontname=font, fontsize=10)
        doc.save(path)
    return path


@pytest.mark.parametrize("language,lines", [
    ("PYTHON", ["# Keep the input value.", "sample_value = 'safe'", "print(sample_value)"]),
    ("JAVASCRIPT", ["const value = 17;", "console.log(value);"]),
    ("R", ["value <- 17", "print(value)"]),
])
def test_native_box_recovers_newlines_without_execution(tmp_path, language, lines):
    path = native_pdf(tmp_path, lines)
    raw = " ".join(lines).replace("_", "\\_").replace("#", "\\#")
    items = [item(language, [100, 100, 180, 115], text_level=1),
             item(raw, [95, 135, 950, 205])]
    before = deepcopy(items)
    result = repair_code_items(items, path)
    assert items == before
    assert result[0]["_pdf_code_label_owned"]
    assert result[1]["type"] == "code"
    assert result[1]["code_body"] == "\n".join(lines) + "\n"
    assert result[1]["_pdf_code_recovery"]["ocr_text"] == raw
    assert caption_of(result[1]) == ""
    assert result[1]["_pdf_code_language"] == language.casefold()
    assert result[1]["_pdf_code_label_sources"] == [{"page": 1, "bbox": items[0]["bbox"], "text": language, "kind": "language"}]
    assert repair_code_items(result, path) == result


@pytest.mark.parametrize("case", ["prose", "font", "missing", "different", "heading", "unlabelled", "column", "single_line"])
def test_ambiguous_or_incomplete_text_keeps_original(tmp_path, case):
    lines = ["# Keep the value.", "value = 17"]
    font = "cour"
    if case == "prose": lines = ["This is a short explanation.", "Keep its original paragraph."]
    if case == "font": font = "helv"
    if case == "single_line": lines = ["value = 17"]
    path = native_pdf(tmp_path, lines, font)
    raw = " ".join(lines)
    if case == "missing": raw += " other_value = 123"
    if case == "different": raw = "# Other text. value = 71"
    tag = item("PYTHON", [100, 100, 180, 115], text_level=1)
    if case == "heading": tag["bbox"][3] = 130
    if case == "unlabelled": tag["text"] = "Short explanation"
    if case == "column": tag["bbox"] = [960, 100, 995, 115]
    items = [tag, item(raw, [95, 135, 950, 205])]
    assert repair_code_items(items, path) == items


def test_non_python_prose_with_inline_assignment_is_not_a_listing(tmp_path):
    lines = ["Use value = 17 for this example.", "Keep its original paragraph."]
    path = native_pdf(tmp_path, lines)
    items = [item("JAVASCRIPT", [100, 100, 180, 115]), item(" ".join(lines), [95, 135, 950, 205])]
    assert repair_code_items(items, path) == items


def test_split_counter_closes_previous_and_language_opens_next(tmp_path):
    previous = item("x = 17", [100, 100, 900, 700], "code")
    counter = item("Code block split (2/3)", [400, 710, 600, 725])
    language = item("PYTHON", [100, 745, 180, 758], text_level=1)
    following = item("y = 18", [100, 100, 900, 700], "code", page=1)
    result = repair_code_items([previous, counter, language, following], tmp_path / "scan.pdf")
    assert caption_of(result[0]) == ""
    assert result[0]["_pdf_code_label_sources"][0]["text"] == counter["text"]
    assert result[3]["_pdf_code_language"] == language["text"].casefold()
    assert result[0]["text"] == previous["text"] and result[3]["text"] == following["text"]
    assert result[3]["_pdf_code_label_sources"][0]["page"] == 1


@pytest.mark.parametrize("case", ["section", "prose", "column", "large_heading", "late_code", "nonconsecutive", "other_language", "counter_before"])
def test_labels_do_not_cross_intervening_content_or_layout(tmp_path, case):
    tag = item("PYTHON", [100, 745, 180, 758], text_level=1)
    code = item("y = 18", [100, 100, 900, 700], "code", page=1)
    items = [tag, code]
    if case in {"section", "prose"}:
        items.insert(1, item("A new section" if case == "section" else "Use this option.",
                             [100, 50, 900, 75], page=1, **({"text_level": 1} if case == "section" else {})))
    if case == "column": code["bbox"] = [600, 100, 900, 700]
    if case == "large_heading": tag["bbox"] = [100, 200, 180, 230]
    if case == "late_code": code["bbox"][1] = 400
    if case == "nonconsecutive": code["page_idx"] = 2
    if case == "other_language": code["code_caption"] = ["RUBY"]
    if case == "counter_before": tag["text"] = "Code block split (1/2)"
    assert repair_code_items(items, tmp_path / "scan.pdf") == items


def test_page_break_and_indentation_use_panel_column_not_token_guess(tmp_path):
    fitz = pytest.importorskip("pymupdf")
    path = tmp_path / "indented.pdf"
    with fitz.open() as doc:
        doc.new_page(width=600, height=800)
        page = doc.new_page(width=600, height=800)
        page.draw_rect((50, 70, 560, 180), fill=(.9, .9, .9))
        page.insert_text((160, 100), "value = 17", fontname="cour", fontsize=10)
        doc.save(path)
    tag = item("PYTHON", [100, 300, 180, 313])
    code = item("value = 17", [260, 100, 400, 140], "code", page=1)
    result = repair_code_items([tag, code], path)
    assert result[0]["_pdf_code_label_owned"] and result[1]["_pdf_code_language"] == "python"
    assert result[1]["bbox"] == code["bbox"]


def test_label_provenance_reaches_chunks_and_graph_without_becoming_heading(tmp_path):
    from types import SimpleNamespace
    items = [item("2. Setup", [100, 500, 400, 525], text_level=1),
             item("A short explanation.", [100, 600, 900, 640]),
             item("PYTHON", [100, 745, 180, 758], text_level=1),
             item("value = 17", [100, 100, 900, 700], "code", page=1)]
    with patch("kb_pipeline.parsers.mineru_pdf.call_mineru_sync", return_value={}), \
         patch("kb_pipeline.parsers.mineru_pdf.extract_content_list", return_value=items):
        blocks = mineru_pdf_blocks(mineru_url="", path=tmp_path / "scan.pdf", cache_dir=tmp_path)
    assert len(blocks) == 4 and blocks[-1].block_id == "mineru-code-00004"
    assert blocks[-1].metadata["section_path"] == ["2. Setup"]
    chunks = blocks_to_chunks(kb_id="kb_fixture", file_key=1, content_version="v", parser_profile="p",
                             blocks=blocks, max_tokens=512, overlap_tokens=0)
    code = chunks[-1]
    assert code.text == "value = 17"
    assert code.block.metadata["code_language"] == "python"
    assert code.block.metadata["code_context_sources"] == [{"page": 1, "bbox": items[2]["bbox"], "text": "PYTHON", "kind": "language"}]
    # A normal short cross-page explanation remains a separate text chunk.
    assert any("A short explanation." in c.text and c.block.block_type != "code" for c in chunks)
    payload = {"text": code.text, "block_type": "code", "code_context": code.block.metadata.get("code_context"),
               "code_context_sources": code.block.metadata["code_context_sources"], "chunk_uid": code.chunk_uid,
               "doc_id": "d", "content_version": "v", "token_count": code.token_count}
    q = SimpleNamespace(retrieve=lambda **kw: [SimpleNamespace(id="p", payload=payload)])
    units = build_units(fetch_chunk_payloads(q, "fixture", [{"point_id": "p"}]), kb_id="kb_fixture", unit_chunks=3)
    assert "value = 17" in units[0].text


def test_label_on_another_tile_restores_its_own_original_coordinates():
    from kb_pipeline.models import ParsedBlock
    from kb_pipeline.parsers.pdf_longpage import restore_page_coordinates
    segments = [{"page": 3, "width": 600, "height": 1600, "y0": i*800,
                 "y1": (i+1)*800, "tiled": True} for i in range(2)]
    label = {"page": 1, "bbox": [100, 800, 180, 815], "text": "PYTHON"}
    b = ParsedBlock("fixture", "fixture", "pdf", "code", "value = 17", "code",
                    page_idx=2, bbox=[100, 100, 900, 700], metadata={"code_context_sources": [label]})
    restore_page_coordinates([b], segments)
    assert b.page_idx == 3 and b.bbox == [100, 550, 900, 850]
    assert label == {"page": 3, "bbox": [100, 400, 180, 407.5], "text": "PYTHON"}


@pytest.mark.parametrize("case,exempt", [("owned", True), ("prose", False), ("page", False), ("unbound", False)])
def test_diagnostics_distinguish_owned_labels_from_short_body(case, exempt):
    from kb_pipeline.models import ParsedBlock
    from kb_pipeline.chunking.diagnose import chunk_diagnostics
    box = [100, 700, 180, 715]
    label = ParsedBlock("fixture", "fixture", "pdf", "text", "PYTHON", "label", page_idx=1, bbox=box)
    code = ParsedBlock("fixture", "fixture", "pdf", "code", "value = 17", "code", page_idx=2,
        bbox=[100, 100, 900, 200], metadata={"code_context_sources": [
            {"page": 1, "bbox": box, "kind": "language", "text": "PYTHON"}]})
    if case == "prose": label.text += "\nA short but useful explanation."
    if case == "page": label.page_idx = 3
    if case == "unbound": code.metadata.clear()
    blocks = [label, code]
    chunks = blocks_to_chunks(kb_id="fixture", file_key=1, content_version="v", parser_profile="p",
                             blocks=blocks, max_tokens=512, overlap_tokens=0)
    before = deepcopy(chunks)
    stats = chunk_diagnostics(chunks, max_tokens=512, blocks=blocks)["stats"]
    assert stats["code_label_chunks"] == int(exempt)
    assert stats["tiny_count"] == int(not exempt)
    assert chunks == before
