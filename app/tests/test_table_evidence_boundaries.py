"""Ordinary table layouts and source fonts must not become false evidence."""
from copy import deepcopy
from unittest.mock import patch

import pytest

from kb_pipeline.models import ParsedBlock
from kb_pipeline.parsers.common import grid_to_markdown
from kb_pipeline.parsers import pdf_evidence as pe, table_repair as tr


def table(tmp_path, rows):
    image = tmp_path / "crop.png"
    image.write_bytes(b"stubbed model input")
    text = grid_to_markdown(rows)
    return ParsedBlock("fixture", "fixture", "pdf", "table", text, "table",
        page_idx=1, bbox=[0, 0, 1000, 1000], table_markdown=text,
        metadata={"table_review_reasons": ["native_text_missing"], "table_review_image": str(image)})


@pytest.mark.parametrize("prefix", [[], [""]])
def test_blank_corner_does_not_block_review_or_change_body_keys(tmp_path, prefix):
    rows = [prefix + row for row in [["", "Left", "Right"], ["Alpha", "17 V", "9 V"], ["Beta", "5 V", "3 V"]]]
    block = table(tmp_path, rows)
    with patch.object(tr, "transcribe_table", return_value={"rows": rows}) as call:
        result = tr.review_pdf_tables([block], base_url="fixture", api_key="fixture", model_id="fixture", cache_dir=tmp_path)
    call.assert_called_once()
    assert result == {"requested": 1, "accepted": 1, "unverified": 0}
    assert tr._pipe_grid(block.text) == rows


@pytest.mark.parametrize("change", ["fill_corner", "move_header", "missing_body_key", "duplicate_body_key", "number", "unit", "sign"])
def test_blank_corner_is_not_permission_to_change_cells(tmp_path, change):
    rows = [["", "Value"], ["Alpha", "17 V"], ["Beta", "9 V"]]
    new = deepcopy(rows)
    if change == "fill_corner": new[0][0] = "Invented header"
    if change == "move_header": new[0], new[1] = new[1], new[0]
    if change == "missing_body_key": rows[2][0] = new[2][0] = ""
    if change == "duplicate_body_key": rows[2] = new[2] = rows[1][:]
    if change in {"number", "unit", "sign"}: new[1][1] = {"number": "19 V", "unit": "17 A", "sign": "-17 V"}[change]
    block = table(tmp_path, rows)
    with patch.object(tr, "transcribe_table", return_value={"rows": new}):
        result = tr.review_pdf_tables([block], base_url="fixture", api_key="fixture", model_id="fixture", cache_dir=tmp_path)
    assert not result["accepted"]
    assert tr._pipe_grid(block.text) == rows


@pytest.mark.parametrize("old,new", [
    ("Refs 【A018-A021】", "Refs [A018-A021]"),
    ("A，B：17 V", "A,B:17 V"),
    ("Examples......", "Examples······"),
    ("Remarks…", "Remarks..."),
    ("Observation (A).", "Observation（A）"),
    ("已完成。", "已完成"),
])
def test_typography_alone_does_not_need_independent_confirmation(old, new):
    assert tr._review_preserves_rows([["Item", old]], [["Item", new]], "")
    assert pe.source_span_present(old, pe.table_evidence_text(new), [])


def test_table_body_keeps_final_punctuation_for_separate_native_spans():
    body = pe.table_evidence_text(grid_to_markdown([["Item", "Note"], ["Alpha", "Click「Confirm」。"]]))
    assert pe.source_span_present("」。", body, [])
    assert pe.source_span_present("Click「Confirm」。", body, [])
    assert pe.source_span_present("Click「Confirm」", body, [])
    assert not pe.source_span_present("〕。", body, [])


@pytest.mark.parametrize("old,new", [
    ("17 V", "17 A"), ("-17 V", "17 V"), ("<17", ">17"),
    ("[1,2]", "(1,2)"), ("1.7", "17"), ("1·7", "17"),
    ("Refs [A018-A021]", "Refs [A018-A022]"), ("Not allowed", "Allowed"),
    ("未⻅异常", "未1异常"), ("未⻅异常", "未短异常"),
])
def test_presentation_comparison_preserves_meaningful_symbols(old, new):
    assert not tr._review_preserves_rows([["Item", old]], [["Item", new]], "")
    assert not pe.source_span_present(old, pe.table_evidence_text(new), [])


def native_pdf(tmp_path, broken=False, rotation=0, actual_text=False):
    pdf = pytest.importorskip("pymupdf")
    doc = pdf.open()
    page = doc.new_page(width=500, height=300)
    font = pdf.Font("helv")
    xref = page.insert_font(fontname="fixture", fontbuffer=font.buffer)
    page.insert_text((80, 180), "Alpha 17 V", fontname="fixture", rotate=rotation)
    page.insert_text((160, 180), "Beta 9 V", fontname="fixture", rotate=rotation)
    if broken:
        target = int(doc.xref_get_key(xref, "ToUnicode")[1].split()[0])
        cmap = doc.xref_stream(target).decode()
        glyph = font.has_glyph(ord("p"))
        cmap = cmap.replace("endcmap", f"1 beginbfchar\n<{glyph:04x}> <0020>\nendbfchar\nendcmap")
        doc.update_stream(target, cmap.encode())
    if actual_text:
        stream = page.get_contents()[0]
        text = ("\ufeffAlpha 17 V").encode("utf-16-be").hex()
        doc.update_stream(stream, f"/Span << /ActualText <{text}> >> BDC\n".encode() + doc.xref_stream(stream) + b"\nEMC")
    path = tmp_path / "native.pdf"
    doc.save(path)
    doc.close()
    return path


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_real_spaces_remain_trusted_and_missing_values_survive(tmp_path, rotation):
    pdf = pytest.importorskip("pymupdf")
    path = native_pdf(tmp_path, rotation=rotation)
    with pdf.open(path) as doc:
        assert pe._unreadable_native_glyphs(doc[0]) == []
    block = table(tmp_path, [["Item", "Value"], ["Alpha", ""], ["Beta", ""]])
    block.metadata.clear()
    pe.add_pdf_region_evidence([block], path, tmp_path)
    tr.review_pdf_tables([block], base_url="fixture", api_key="", model_id="fixture", cache_dir=tmp_path)
    assert "17 V" in block.text and "9 V" in block.text
    assert not block.metadata.get("pdf_native_lines_omitted")


def test_visible_glyph_mapped_to_space_cannot_contaminate_ocr_or_other_lines(tmp_path):
    pdf = pytest.importorskip("pymupdf")
    path = native_pdf(tmp_path, broken=True)
    with pdf.open(path) as doc:
        assert "Al ha" in doc[0].get_text()
        assert len(pe._unreadable_native_glyphs(doc[0])) == 1
    block = table(tmp_path, [["Item", "Value"], ["Alpha", "17 V"], ["Beta", ""]])
    block.metadata.clear()
    pe.add_pdf_region_evidence([block], path, tmp_path)
    assert block.metadata["pdf_native_lines_omitted"] == 1
    assert "Al ha" not in block.metadata["table_source_text"]
    assert not block.metadata["table_native_grid"]
    tr.review_pdf_tables([block], base_url="fixture", api_key="", model_id="fixture", cache_dir=tmp_path)
    assert "Alpha" in block.text and "17 V" in block.text and "Beta 9 V" in block.text
    assert "Al ha" not in block.text


def test_broken_font_context_is_not_sent_to_the_image_reader(tmp_path):
    path = native_pdf(tmp_path, broken=True)
    block = table(tmp_path, [["Item", "Value"]])
    block.block_type = "image"
    block.metadata.clear()
    pe.add_pdf_region_evidence([block], path, tmp_path)
    assert block.metadata["visual_source_context"] == "Beta 9 V"


def test_pdf_actual_text_can_supply_valid_text_despite_broken_raw_mapping(tmp_path):
    pdf = pytest.importorskip("pymupdf")
    path = native_pdf(tmp_path, broken=True, actual_text=True)
    with pdf.open(path) as doc:
        assert "Alpha17V" in "".join(doc[0].get_text().split())
        assert pe._unreadable_native_glyphs(doc[0]) == []
    block = table(tmp_path, [["Item", "Value"], ["Alpha", "17 V"], ["Beta", "9 V"]])
    block.metadata.clear()
    pe.add_pdf_region_evidence([block], path, tmp_path)
    assert not block.metadata.get("pdf_native_lines_omitted")


@pytest.mark.parametrize("radical,letter", [("⻣", "骨"), ("⻆", "角"), ("⻅", "见"), ("⻓", "长")])
def test_standard_font_variants_match_exact_equivalents_not_wildcards(radical, letter):
    assert pe.source_span_present("Prefix" + radical, "Prefix" + letter, [])
    for different in ["1", "-", "A", "错"]:
        assert not pe.source_span_present("Prefix" + radical, "Prefix" + different, [])
