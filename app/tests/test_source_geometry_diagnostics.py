"""Synthetic geometry, diagnostic and table-format boundaries."""
from copy import deepcopy

import pytest

from kb_pipeline.models import ParsedBlock
from kb_pipeline.chunking.chunker import blocks_to_chunks
from kb_pipeline.chunking.context import evidence_text
from kb_pipeline.chunking.diagnose import _heading_only
from kb_pipeline.chunking.packing import owns_code_intro
from kb_pipeline.parsers.pdf_evidence import _norm, _span_present, normalize_ranges
from kb_pipeline.parsers.table_repair import _review_preserves_rows


def block(text, kind="text", **kw):
    return ParsedBlock(parser="fixture", parser_profile="v1", doc_type="pdf", block_type=kind,
                       block_id=kind, text=text, **kw)


def pair():
    intro = block("Convert the input before storing it.", page_idx=2, metadata={
        "section_path": ["Procedure"], "source_fields": [], "page_end": 2, "source_regions": [
            {"page": 2, "bbox": [100, 200, 400, 220]}, {"page": 2, "bbox": [100, 225, 400, 245]}]})
    code = block("    return convert(value)\n", "code", page_idx=2, bbox=[110, 270, 450, 350],
                 metadata={"section_path": ["Procedure"]})
    return intro, code


def test_merged_pdf_intro_uses_all_regions_and_keeps_literal_source():
    intro, code = pair()
    original = deepcopy([intro, code])
    assert intro.bbox is None and owns_code_intro(intro, code)
    cs = blocks_to_chunks(kb_id="fixture", file_key=1, content_version="v", parser_profile="p",
                          blocks=[intro, code], max_tokens=128, overlap_tokens=0)
    assert len(cs) == 1 and cs[0].text == code.text
    assert intro.text in evidence_text(cs[0].text, cs[0].block.metadata)
    assert cs[0].block.metadata["code_context_sources"] == intro.metadata["source_regions"]
    assert [intro, code] == original


@pytest.mark.parametrize("bad", ["other_page", "other_column", "below", "missing_box", "missing_page",
                                "nan", "inverted", "page_end", "section"])
def test_partial_or_ambiguous_regions_do_not_establish_ownership(bad):
    intro, code = pair()
    region = intro.metadata["source_regions"][-1]
    if bad == "other_page": region["page"] = 3
    elif bad == "other_column": region["bbox"] = [600, 225, 900, 245]
    elif bad == "below": region["bbox"] = [100, 400, 400, 420]
    elif bad == "missing_box": region.pop("bbox")
    elif bad == "missing_page": region.pop("page")
    elif bad == "nan": region["bbox"][1] = float("nan")
    elif bad == "inverted": region["bbox"] = [400, 200, 100, 220]
    elif bad == "page_end": intro.metadata["page_end"] = 3
    else: code.metadata["section_path"] = ["Other procedure"]
    # Even a broad fallback rectangle must not override conflicting source regions.
    intro.bbox = [50, 100, 950, 250]
    assert not owns_code_intro(intro, code)


def test_nonempty_source_field_boundary_is_preserved():
    intro, code = pair()
    intro.metadata["source_fields"] = ["first_field"]
    assert not owns_code_intro(intro, code)


@pytest.mark.parametrize("text", ["Open", "Weekly Summary", "Location: North", "Count: 312 → 298",
                                 "The available options are:", "学习要点", "系统分为以下几层："])
def test_short_body_is_not_a_heading_without_structural_evidence(text):
    assert not _heading_only(text, block(text))


@pytest.mark.parametrize("metadata", [{"heading_lines": ["Inspection"]}, {"section_path": ["Inspection"]}])
def test_parser_identified_heading_remains_diagnostic(metadata):
    assert _heading_only("Inspection", block("Inspection", metadata=metadata))
    assert _heading_only("Inspection", block("Inspection", "title"))
    assert not _heading_only("Inspection\nUse the supplied procedure.", block("", "title"))


@pytest.mark.parametrize("native,ocr", [("$73K–$141K", "$73K-$141K"), ("2.5万—6万元", "2.5万-6万元"),
                                      ("9kg – 17kg", "9kg-17kg"), ("13%–29%", "13%-29%"),
                                      ("4–8", "4-8")])
def test_numeric_range_typography_does_not_trigger_missing_source(native, ocr):
    assert _norm(native) == _norm(ocr)
    assert _span_present(native, _norm(ocr))
    assert _review_preserves_rows([["Range", ocr]], [["Range", native]], native)


@pytest.mark.parametrize("native", ["−17", "−17–−9", "17−9", "–17", "alpha–beta", "x—y"])
def test_signs_operators_and_non_numeric_dashes_remain_distinct(native):
    assert normalize_ranges(native) == native


@pytest.mark.parametrize("wrong", ["73K-140K", "73M-141M", "-73K-141K", "<73K-141K"])
def test_range_formatting_does_not_accept_changed_value_unit_sign_or_condition(wrong):
    assert not _review_preserves_rows([["Range", "73K-141K"]], [["Range", wrong]], "73K–141K")


def test_table_bullet_prefix_requires_the_complete_item_to_survive():
    assert _span_present("• Keep the original value", _norm("Keep the original value"))
    assert _span_present("<br>• Up to 17 MB", _norm("Up to 17 MB"))
    assert not _span_present("• Do not open", _norm("Do open"))
    assert not _span_present("• 17 MB", _norm("18 MB"))
    assert not _span_present("<br>", _norm("An unrelated tag"))


def test_equal_table_values_in_another_column_or_row_do_not_validate_a_move():
    old = [["Item", "First", "Second"], ["A", "11 kg", "17 kg"], ["B", "23 kg", "29 kg"]]
    new = deepcopy(old)
    new[1][1:] = list(reversed(new[1][1:]))
    assert not _review_preserves_rows(old, new, str(new))
    assert not _review_preserves_rows(old, [old[0], old[2], old[1]], str(old))
