"""Source line recovery must retain small values without assigning table cells."""
import re
from unittest.mock import patch

import pytest

from kb_pipeline.models import ParsedBlock
from kb_pipeline.parsers.pdf_evidence import add_pdf_region_evidence
from kb_pipeline.parsers.table_repair import review_pdf_tables
from kb_pipeline.chunking.chunker import blocks_to_chunks
from kb_pipeline.utils import count_tokens


GRID = "| Item | Value |\n| --- | --- |\n| Alpha | 17 |"


def table(**kwargs):
    return ParsedBlock(parser="fixture", parser_profile="fixture-v1", doc_type="pdf",
                       block_type="table", block_id="table", text=GRID,
                       table_markdown=GRID, **kwargs)


@pytest.mark.parametrize("parts", [
    ["Recorded: ", "2031", "-", "7", "-", "9"],
    ["Dilution: ", "1", ":", "4"],
    ["Measured: ", "5", ".", "7", " V"],
    ["Offset: ", "-", "2", " V"],
])
@pytest.mark.parametrize("reverse_drawing_order", [False, True])
def test_font_spans_keep_complete_values_and_source_order(tmp_path, parts, reverse_drawing_order):
    pdf = pytest.importorskip("pymupdf")
    path = tmp_path / "source.pdf"
    with pdf.open() as doc:
        page = doc.new_page(width=480, height=240)
        x, pieces = 30, []
        for i, text in enumerate(parts):
            font, size = ("helv", 11) if i % 2 == 0 else ("hebo", 10)
            pieces.append((x, text, font, size))
            x += pdf.get_text_length(text, fontname=font, fontsize=size)
        for x, text, font, size in reversed(pieces) if reverse_drawing_order else pieces:
            page.insert_text((x, 100), text, fontname=font, fontsize=size)
        page.insert_text((30, 150), "Unrelated footer 999")
        doc.save(path)
    b = table(page_idx=1, bbox=[0, 0, 1000, 500])
    add_pdf_region_evidence([b], path, tmp_path)
    review_pdf_tables([b], base_url="fixture", api_key="", model_id="fixture", cache_dir=tmp_path)
    assert b.text.startswith(GRID + "\nSOURCE TEXT")
    assert re.sub(r"\s+", "", "".join(parts)) in re.sub(r"\s+", "", b.text)
    assert "Unrelated footer" not in b.text
    assert b.metadata["table_review"]["status"] == "unverified"


@pytest.mark.parametrize("response", [None, {"rows": [["Item", "Value"], ["Alpha", "17"]]}])
def test_review_failure_or_unchanged_rows_do_not_approve_missing_note(tmp_path, response):
    image = tmp_path / "crop.png"
    image.write_bytes(b"fixture")
    b = table(metadata={"table_review_reasons": ["native_text_missing"],
                        "table_missing_spans": ["Recorded:", "2031"],
                        "table_source_text": "Item Value\nAlpha 17\nRecorded: 2031-7-9\nUnrelated footer 999",
                        "table_review_image": str(image)})
    with patch("kb_pipeline.parsers.table_repair.transcribe_table", return_value=response,
               side_effect=TimeoutError if response is None else None):
        review_pdf_tables([b], base_url="fixture", api_key="fixture", model_id="fixture", cache_dir=tmp_path)
    assert b.text.startswith(GRID + "\nSOURCE TEXT")
    assert b.text.count("Recorded: 2031-7-9") == 1
    assert "Unrelated footer" not in b.text
    assert b.metadata["table_review"]["status"] != "accepted"


@pytest.mark.parametrize("native", ["", "Other source line 42"])
def test_unmatched_source_span_is_not_lost(tmp_path, native):
    b = table(metadata={"table_review_reasons": ["native_text_missing"],
                        "table_missing_spans": ["Preserved note"], "table_source_text": native})
    review_pdf_tables([b], base_url="fixture", api_key="", model_id="fixture", cache_dir=tmp_path)
    assert "Preserved note" in b.text
    assert "Other source" not in b.text


def test_long_source_supplement_still_obeys_chunk_budget(tmp_path):
    lines = [f"Record {i}: 2031-7-9; measured 5.7 V; offset -2 V." for i in range(80)]
    b = table(metadata={"table_review_reasons": ["native_text_missing"],
                        "table_missing_spans": [f"Record {i}:" for i in range(80)],
                        "table_source_text": "\n".join(lines)})
    review_pdf_tables([b], base_url="fixture", api_key="", model_id="fixture", cache_dir=tmp_path)
    result = blocks_to_chunks(kb_id="kb_fixture", file_key=1, content_version="fixture", parser_profile="p",
                             blocks=[b], max_tokens=100, overlap_tokens=0)
    assert len(result) > 1
    assert all(count_tokens(c.text) <= 100 for c in result)
    assert all(any(line in c.text for c in result) for line in lines)
    assert "| Alpha | 17 |" in b.text
