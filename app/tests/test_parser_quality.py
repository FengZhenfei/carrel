"""Source fidelity fixes with controls for neighboring document structures."""
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from kb_pipeline.chunking.chunker import blocks_to_chunks
from kb_pipeline.chunking.diagnose import _heading_only
from kb_pipeline.models import ParsedBlock
from kb_pipeline.parsers.common import parser_profile_for_path
from kb_pipeline.parsers.html_dom import parse_html_dom
from kb_pipeline.parsers.mineru_pdf import mineru_pdf_blocks
from kb_pipeline.parsers.mineru_quality import recover_pdf_items
from kb_pipeline.parsers.pdf_longpage import restore_page_coordinates
from kb_pipeline.parsers.text_formats import parse_text_format
from kb_pipeline.parsers.visual_blocks import enrich_blocks_with_vlm


def test_pdf_cover_and_body_survive_running_header_filter():
    items = [
        {"type": "header", "text": "Device Model X", "page_idx": 0, "bbox": [50, 50, 400, 120]},
        {"type": "header", "text": "Shared manual", "page_idx": 0},
        {"type": "header", "text": "Shared manual", "page_idx": 1},
        {"type": "header", "text": "A substantive introductory paragraph.", "page_idx": 0},
        {"type": "footer", "text": "Copyright Example", "page_idx": 0},
        {"type": "header", "text": "Page 1 of 2", "page_idx": 0},
    ]
    fixed = recover_pdf_items(items, {})
    assert [x["type"] for x in fixed] == ["title", "header", "header", "text", "footer", "header"]
    assert items[0]["type"] == "header"  # cached raw payload is not rewritten


def _html(rows):
    return "<table>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</table>"


def _table(image, rows):
    return {"type": "table", "blocks": [{"type": "table_body", "lines": [{"spans": [
        {"type": "table", "image_path": image, "html": _html(rows)}]}]}]}


def test_upstream_merged_table_recovers_page_range(tmp_path):
    head, a, b, c = ["Item", "Value"], ["Alpha", "12"], ["Beta", "34"], ["Gamma", "56"]
    payload = {"content_list": [{"type": "table", "img_path": "images/a.jpg", "page_idx": 0,
                                  "table_body": _html([head, a, b, c])}],
               "middle_json": {"pdf_info": [
                   {"page_idx": 0, "preproc_blocks": [_table("a.jpg", [head, a])],
                    "para_blocks": [_table("a.jpg", [head, a, b, c])]},
                   {"page_idx": 1, "preproc_blocks": [_table("b.jpg", [b])], "para_blocks": []},
                   {"page_idx": 2, "preproc_blocks": [_table("c.jpg", [head, c])], "para_blocks": []}]}}
    with patch("kb_pipeline.parsers.mineru_pdf.call_mineru_sync", return_value=payload):
        blocks = mineru_pdf_blocks(mineru_url="unused", path=tmp_path / "sample.pdf", cache_dir=tmp_path)
    assert blocks[0].page_idx == 1
    assert blocks[0].metadata["page_end"] == 3
    assert "Beta" in blocks[0].text and "Gamma" in blocks[0].text


@pytest.mark.parametrize("continuation", [[["Different", "34"]], [["Item", "Value"], ["Alpha", "12"]]])
def test_similar_table_on_next_page_does_not_extend_source(continuation):
    first = [["Item", "Value"], ["Alpha", "12"]]
    items = [{"type": "table", "img_path": "a.jpg", "page_idx": 0, "table_body": _html(first)}]
    payload = {"middle_json": {"pdf_info": [
        {"page_idx": 0, "preproc_blocks": [_table("a.jpg", first)], "para_blocks": [_table("a.jpg", first)]},
        {"page_idx": 1, "preproc_blocks": [_table("b.jpg", continuation)], "para_blocks": []}]}}
    assert "source_page_end" not in recover_pdf_items(items, payload)[0]


def test_table_range_is_restored_from_tiles_to_original_pages():
    block = ParsedBlock(parser="p", parser_profile="p", doc_type="pdf", block_type="table", text="x",
                        block_id="a", page_idx=1, metadata={"page_end": 2})
    segments = [{"page": 1, "tiled": False}, {"page": 1, "tiled": False}]
    restore_page_coordinates([block], segments)
    assert block.page_idx == block.metadata["page_end"] == 1


@pytest.mark.parametrize("own,keep", [("A diagram of an unrelated device.", False),
                                       ("```mermaid\ngraph LR\nA-->B\n```", True),
                                       ("Voltage: 12 V", True)])
def test_generated_description_does_not_override_image_reading(tmp_path, own, keep):
    image = tmp_path / "figure.png"
    Image.new("RGB", (400, 300)).save(image)
    block = ParsedBlock(parser="mineru", parser_profile="p", doc_type="pdf", block_type="image", text=own,
                        visual_ref=str(image), block_id="a", metadata={"image_text_source": "model"})
    result = {"a": {"kind": "diagram", "summary": "Device schematic.", "text_verbatim": "Voltage: 12 V",
                    "facts": ["Voltage: 12 V"], "entities": [], "keywords": []}}
    with patch("kb_pipeline.parsers.visual_blocks.caption_images_parallel", return_value=result):
        enrich_blocks_with_vlm([block], base_url="unused", api_key="test", model_id="test", cache_dir=tmp_path)
    assert (own in block.text) is keep
    assert "Device schematic." in block.text


def test_tex_hierarchy_preserves_commands_equations_and_literal_examples(tmp_path):
    path = tmp_path / "manual.tex"
    source = "\\section{Setup}\n\\subsection{Inputs}\nUse $x=2$.\n\\subsubsection{Nested {title}}\nMore text.\n\\begin{verbatim}\n\\section{Example only}\n\\end{verbatim}\n\\section{Next} Inline body.\n\\subsection{Details}\nFinal body."
    path.write_text(source)
    blocks = parse_text_format(path)
    assert blocks[0].metadata["section_path"] == ["Setup", "Inputs"]
    assert blocks[1].metadata["section_path"] == ["Setup", "Inputs", "Nested {title}"]
    assert blocks[2].metadata["section_path"] == ["Next"]
    assert "Inline body." in blocks[2].text
    assert all(line in "\n".join(b.text for b in blocks) for line in source.splitlines())
    assert blocks[0].metadata["source_locator"] == {"kind": "lines", "start": 1, "end": 3}


@pytest.mark.parametrize("text,expected", [
    ("TITLE: convert(value)\nConvert a value to a string.", False),
    ("TITLE: 参数说明\n返回转换结果", False),
    ('"Show an example"', False), ("A complete sentence.", False),
    ("TITLE: Setup\nSetup", True), ("1.2 Setup", True), ("学习要点", True)])
def test_heading_diagnostics_distinguish_short_body(text, expected):
    assert _heading_only(text) is expected


@pytest.mark.parametrize("sectioned", [True, False])
def test_html_table_markers_and_normal_cells_survive_once(tmp_path, sectioned):
    path = tmp_path / "template.html"
    path.write_text("<html><body><h1>Overview</h1>" + ("<h2>Results</h2>" if sectioned else "") +
                    "<table><caption>Values</caption><tr><th>Name</th><th>Result</th></tr>"
                    "<tbody>{{ROWS}}<tr><td>Alpha</td><td>12</td></tr></tbody></table></body></html>")
    blocks = parse_html_dom(path)
    text = "\n".join(b.text for b in blocks)
    assert text.count("{{ROWS}}") == 1 and text.count("Alpha") == 1
    assert "Values" in text and "12" in text
    chunks = blocks_to_chunks(blocks=blocks, kb_id="kb_test", file_key=1, content_version="v", parser_profile="p",
                              max_tokens=400, overlap_tokens=80)
    assert "{{ROWS}}" in "\n".join(c.text for c in chunks)


def test_existing_route_versions_do_not_trigger_historical_reparse():
    assert parser_profile_for_path(Path("old.pdf")) == "pdf-mineru-table-vlm-v15"
    assert parser_profile_for_path(Path("old.md")) == "md-sections-v3"
    assert parser_profile_for_path(Path("old.tex")) == "tex-sections-v1"
    assert parser_profile_for_path(Path("old.html")) == "html-dom-v2"
