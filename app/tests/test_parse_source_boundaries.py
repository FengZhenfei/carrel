"""Source-supported corrections, with no corpus-specific words or thresholds."""
from copy import deepcopy
import json
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from kb_pipeline.models import ParsedBlock
from kb_pipeline.parsers import table_repair as tr
from kb_pipeline.parsers.pdf_evidence import source_span_present
from kb_pipeline.vision import vlm


def response(raw, finish="stop"):
    return NS(choices=[NS(message=NS(content=raw), finish_reason=finish)])


@pytest.mark.parametrize("tail", ["1" * 3000, "?\\n" * 1000, "Repeated label " * 200])
@pytest.mark.parametrize("retry", ["success", "empty", "error", "runaway"])
def test_unfinished_repetition_uses_bounded_retry_and_keeps_complete_facts(tmp_path, tail, retry):
    first = '{"facts":["Input: 17 V"],"summary":"' + tail
    second = (RuntimeError("offline") if retry == "error" else
              response('{}') if retry == "empty" else response(first, "length") if retry == "runaway" else
              response('{"facts":["Input: 17 V"],"summary":"A circuit"}'))
    call = Mock(side_effect=[response(first, "length"), second])
    with patch.object(vlm, "_client_for", return_value=NS(chat=NS(completions=NS(create=call)))), \
         patch.object(vlm, "image_data_url", return_value="data:image/png;base64,AA=="):
        result = vlm.caption_image(image_path=tmp_path / "image.png", base_url="http://fixture/v1",
                                  api_key="fixture", model_id="fixture", cache_json=tmp_path / "caption.json")
    assert [c.kwargs["max_tokens"] for c in call.call_args_list] == [4096, 4096]
    assert result["facts"] == ["Input: 17 V"] and result["vlm_degraded_retry"]
    assert result["vlm_degraded_reason"] == "repetitive_output"
    assert not (tmp_path / "caption.json").exists()


def test_complete_repeated_cells_and_normal_unfinished_prose_are_not_raw_runaway():
    raw = json.dumps({"facts": [f"Pin {i}: 0 V" for i in range(200)]})
    assert not vlm._raw_tail_runaway(raw[:-1], "length")
    assert not vlm._raw_tail_runaway('{"summary":"' + "1" * 1000, "stop")
    prose = " ".join(f"Stage {i} uses input {i*17} and yields output {i*29}." for i in range(100))
    assert not vlm._raw_tail_runaway('{"summary":"' + prose, "length")


def test_empty_first_response_does_not_hide_a_failed_retry(tmp_path):
    call = Mock(side_effect=[response('{"summary":"' + '1' * 3000, 'length'), RuntimeError('offline')])
    with patch.object(vlm, "_client_for", return_value=NS(chat=NS(completions=NS(create=call)))), \
         patch.object(vlm, "image_data_url", return_value="data:image/png;base64,AA=="):
        with pytest.raises(RuntimeError, match="offline"):
            vlm.caption_image(image_path=tmp_path / "image.png", base_url="http://fixture/v1",
                              api_key="fixture", model_id="fixture")


def make_table(tmp_path, source, old):
    image = tmp_path / "table.png"
    Image.new("RGB", (400, 200)).save(image)
    return ParsedBlock("fixture", "fixture", "pdf", "table", old, "table", page_idx=2,
        bbox=[10, 20, 900, 500], table_markdown=old, visual_ref=str(image), metadata={
        "table_review_reasons": ["native_text_missing"], "table_review_image": str(image),
        "table_source_text": "\n".join(" ".join(r) for r in source), "table_native_grid": source})


SOURCE = [["Spec", "Minimum", "Maximum"], ["Alpha", "-17 mA", "29 mA"], ["Beta", "Not approved", "Approved"]]


@pytest.mark.parametrize("old", ["<table><tr><td><img src='crop.png'/></td></tr></table>",
    "| Spec | Spec | Value |\n| --- | --- | --- |\n| Alpha | Alpha | 1729 |"])
def test_independent_cells_can_replace_wrong_ocr_structure(tmp_path, old):
    b = make_table(tmp_path, SOURCE, old)
    with patch.object(tr, "transcribe_table", return_value={"rows": SOURCE}):
        stats = tr.review_pdf_tables([b], base_url="fixture", api_key="fixture", model_id="fixture", cache_dir=tmp_path)
    assert stats["accepted"] == 1
    assert b.metadata["table_review"]["verification"] == "native_cells"
    assert tr._pipe_grid(b.text) == SOURCE and "<img" not in b.text
    assert b.page_idx == 2 and b.bbox == [10, 20, 900, 500]


@pytest.mark.parametrize("change", ["columns", "row", "unit", "sign", "negation", "extra", "empty"])
def test_structural_correction_rejects_cell_damage(tmp_path, change):
    candidate = deepcopy(SOURCE)
    if change == "columns": candidate[1][1:] = candidate[1][1:][::-1]
    if change == "row": candidate.pop()
    if change == "unit": candidate[1][1] = "-17 A"
    if change == "sign": candidate[1][1] = "17 mA"
    if change == "negation": candidate[2][1] = "Approved"
    if change == "extra": candidate.append(["Gamma", "12", "13"])
    if change == "empty": candidate[1][1] = ""
    old = "<table><tr><td><img src='crop.png'/></td></tr></table>"
    b = make_table(tmp_path, SOURCE, old)
    with patch.object(tr, "transcribe_table", return_value={"rows": candidate}):
        stats = tr.review_pdf_tables([b], base_url="fixture", api_key="fixture", model_id="fixture", cache_dir=tmp_path)
    assert stats["accepted"] == 0 and b.text == old


@pytest.mark.parametrize("scope", ["source_regions", "table_source_regions", "page_end", "same_page_outside"])
def test_local_cells_cannot_replace_a_table_with_wider_source_scope(tmp_path, scope):
    rows = [["Item", "Value"], ["Alpha", "17 V"], ["Beta", "29 V"]]
    local = rows[:2]
    b = make_table(tmp_path, local, tr.grid_to_markdown(rows))
    if scope == "page_end":
        b.metadata["page_end"] = 3
    else:
        key = "source_regions" if scope == "same_page_outside" else scope
        b.metadata[key] = [{"page": 2, "bbox": b.bbox},
                           {"page": 2 if scope == "same_page_outside" else 3,
                            "bbox": [10, 600, 900, 950]}]
    with patch.object(tr, "transcribe_table", return_value={"rows": local}):
        stats = tr.review_pdf_tables([b], base_url="fixture", api_key="fixture", model_id="fixture", cache_dir=tmp_path)
    assert stats["accepted"] == 0
    assert tr._pipe_grid(b.text) == rows


def test_regions_inside_the_evidence_crop_do_not_block_correction(tmp_path):
    b = make_table(tmp_path, SOURCE, "| Item | Merged |\n| --- | --- |\n| Alpha | 1729 |")
    b.metadata["source_regions"] = [{"page": 2, "bbox": [10, 20, 900, 200]},
                                    {"page": 2, "bbox": [10, 200, 900, 500]}]
    with patch.object(tr, "transcribe_table", return_value={"rows": SOURCE}):
        stats = tr.review_pdf_tables([b], base_url="fixture", api_key="fixture", model_id="fixture", cache_dir=tmp_path)
    assert stats["accepted"] == 1 and tr._pipe_grid(b.text) == SOURCE


@pytest.mark.parametrize("changed_shape", [False, True])
def test_native_blank_cannot_erase_text_recognized_inside_an_image(tmp_path, changed_shape):
    rows = [["Item", "Image text"], ["Alpha", "Recorded value: 17 V"]]
    local = [["Item", "Image text"], ["Alpha", ""]]
    if changed_shape:
        rows = [[r[0], r[0], r[1]] for r in rows]
    b = make_table(tmp_path, local, tr.grid_to_markdown(rows))
    with patch.object(tr, "transcribe_table", return_value={"rows": local}):
        stats = tr.review_pdf_tables([b], base_url="fixture", api_key="fixture", model_id="fixture", cache_dir=tmp_path)
    assert stats["accepted"] == 0
    assert "Recorded value: 17 V" in b.text
    assert len(tr._pipe_grid(b.text)[1]) == len(rows[1])


def test_source_blanks_preserved_by_ocr_do_not_block_other_cell_corrections(tmp_path):
    source = [["Item", "Value", "Image"], ["Alpha", "17 V", ""]]
    old = [["Item", "Value", "Image"], ["Alpha", "wrong OCR", ""]]
    b = make_table(tmp_path, source, tr.grid_to_markdown(old))
    with patch.object(tr, "transcribe_table", return_value={"rows": source}):
        stats = tr.review_pdf_tables([b], base_url="fixture", api_key="fixture", model_id="fixture", cache_dir=tmp_path)
    assert stats["accepted"] == 1 and tr._pipe_grid(b.text) == source


def test_borderless_wrapped_cells_require_all_source_content(tmp_path):
    source = [["Item", "Condition"], ["", ""], ["Alpha", "Open only"], ["", "after inspection"], ["Beta", "Closed"]]
    b = make_table(tmp_path, source, "")
    b.metadata["table_native_grid_strategy"] = "text"
    cells = tr._source_cells(b)
    assert tr._matches_source_cells(cells, [["Item", "Condition"], ["Alpha", "Open only after inspection"], ["Beta", "Closed"]])
    assert not tr._matches_source_cells(cells, [["Item", "Condition"], ["Alpha", "Open only"], ["Beta", "Closed"]])
    assert not tr._matches_source_cells([["未⻅", "17"]], [["未1", "17"]])
    assert tr._matches_source_cells([["未⻅", "17"]], [["未⻅", "17"]])
    assert tr._matches_source_cells([["⻓期", "17"]], [["长期", "17"]])
    assert not tr._matches_source_cells([["⻓期", "17"]], [["短期", "17"]])


def test_image_only_table_reaches_visual_reader_without_certifying_cells(tmp_path):
    from kb_pipeline.parsers import visual_blocks
    old = "<table><tr><td><img src='crop.png'/></td></tr></table>"
    b = make_table(tmp_path, [], old)
    b.metadata["table_source_text"] = "Printed note: inspect before use."
    tr.route_image_only_tables([b])
    with patch.object(visual_blocks, "caption_images_parallel", return_value={"table": {
        "kind": "table", "summary": "A permissions matrix", "facts": ["Viewer can read."], "confidence": "high"}}) as call:
        visual_blocks.enrich_blocks_with_vlm([b], base_url="fixture", api_key="fixture", model_id="fixture", cache_dir=tmp_path)
    assert call.call_count == 1 and "Viewer can read." in b.text and "inspect before use" in b.text
    assert b.metadata["table_review"]["status"] == "unverified" and b.page_idx == 2
    assert b.visual_ref and not b.table_markdown
    b2 = make_table(tmp_path, SOURCE, tr.grid_to_markdown(SOURCE))
    tr.route_image_only_tables([b2])
    assert b2.block_type == "table"


def test_image_only_fallback_retains_source_when_no_model_is_configured(tmp_path):
    from kb_pipeline.parsers.visual_blocks import enrich_blocks_with_vlm
    b = make_table(tmp_path, [], "Caption\n<table><tr><td><img src='crop.png'/></td></tr></table>")
    # The raw table body excludes its surrounding source caption.
    b.metadata["source_item"] = {"table_body": "<table><tr><td><img src='crop.png'/></td></tr></table>"}
    b.metadata["table_source_text"] = "Printed condition: not approved."
    tr.route_image_only_tables([b])
    enrich_blocks_with_vlm([b], base_url="fixture", api_key="", model_id="fixture", cache_dir=tmp_path)
    assert "Caption" in b.text and "not approved" in b.text and "<img" not in b.text
    assert b.metadata["table_review"]["status"] == "unverified"


def test_borderless_pdf_cells_independently_verify_changed_structure(tmp_path):
    import pymupdf as pdf
    from kb_pipeline.parsers.pdf_evidence import add_pdf_region_evidence
    source = [["Item", "Minimum", "Maximum"], ["Alpha", "-17 mA", "29 mA"],
              ["Beta", "23 mA", "40 mA"], ["Gamma", "12 mA", "19 mA"]]
    path = tmp_path / "borderless.pdf"
    with pdf.open() as doc:
        page = doc.new_page(width=500, height=300)
        for i, row in enumerate(source):
            for x, text in zip([40, 200, 350], row):
                page.insert_text((x, 60 + i * 30), text, fontsize=11)
        boxes = [pdf.Rect(w[:4]) for w in page.get_text("words")]
        area = boxes[0]
        for box in boxes[1:]: area |= box
        bbox = [area.x0 * 2, area.y0 / .3, area.x1 * 2, area.y1 / .3]
        doc.save(path)
    b = make_table(tmp_path, [], "| Item | Merged |\n| --- | --- |\n| Alpha | 1729 |")
    b.page_idx, b.bbox, b.metadata = 1, bbox, {}
    add_pdf_region_evidence([b], path, tmp_path)
    assert b.metadata["table_native_grid_strategy"] == "text"
    with patch.object(tr, "transcribe_table", return_value={"rows": source}):
        stats = tr.review_pdf_tables([b], base_url="fixture", api_key="fixture", model_id="fixture", cache_dir=tmp_path)
    assert stats["accepted"] == 1 and tr._pipe_grid(b.text) == source


def test_line_wrap_matching_needs_adjacent_same_column_source():
    spans = [{"text": "Control architec-", "bbox": [10, 10, 100, 20]},
             {"text": "ture review", "bbox": [10, 23, 90, 33]}]
    assert source_span_present("Control architec-", "Controlarchitecturereview", spans)
    assert not source_span_present("Control architec-", "Controlarchitecturereview", [])
    spans[1]["bbox"] = [150, 23, 250, 33]
    assert not source_span_present("Control architec-", "Controlarchitecturereview", spans)
    assert not source_span_present("-17 mA", "17mA", [])
    assert not source_span_present("ABC-200", "ABC200", [])
