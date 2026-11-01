"""Synthetic source-fidelity cases; no corpus names, text or expected answers."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

from kb_pipeline.models import ParsedBlock
from kb_pipeline.parsers import visual_blocks as visual
from kb_pipeline.parsers.pdf_enhanced import merge_mineru_text_blocks, merge_split_tables
from kb_pipeline.parsers.pdf_evidence import supplement_visible_forms, add_pdf_region_evidence
from kb_pipeline.parsers.table_repair import _review_preserves_rows, review_pdf_tables
from kb_pipeline.chunking.chunker import blocks_to_chunks


def block(text="", kind="text", **kw):
    return ParsedBlock(parser="fixture", parser_profile="fixture-v1", doc_type="pdf", block_type=kind,
                       block_id=kw.pop("block_id", "item"), text=text, **kw)


def chunks(blocks, budget=80):
    return blocks_to_chunks(kb_id="kb_fixture", file_key=1, content_version="fixture", parser_profile="p",
                            blocks=blocks, max_tokens=budget, overlap_tokens=0)


def test_prompt_contains_only_source_owned_context():
    b = block("invented upstream table 999", "image", title="Unrelated conclusion", caption="Output: 17 kW",
              metadata={"image_text_source": "model", "visual_source_context": "Legend: dashed = planned"})
    p = visual.visual_prompt(b, "Read this image.")
    assert "Output: 17 kW" in p and "dashed = planned" in p
    assert "Unrelated conclusion" not in p and "999" not in p
    b.caption = "Output: 23 kW"
    assert visual.visual_prompt(b, "Read this image.") != p


@pytest.mark.parametrize("kind", ["chart", "diagram", "table", "formula"])
def test_generated_structured_text_is_not_promoted_to_source(tmp_path, kind):
    from PIL import Image
    p = tmp_path / "figure.png"
    Image.new("RGB", (300, 200), "white").save(p)
    old = "| Output | 999 |\n| --- | --- |"
    generated = block(old, "image", visual_ref=str(p), metadata={"image_text_source": "model"})
    native = block(old, "image", block_id="native", visual_ref=str(p))
    response = {"kind": kind, "summary": "A measurement", "facts": ["Output: 17 kW"], "text_verbatim": "Output: 17 kW"}
    with patch.object(visual, "caption_images_parallel", return_value={"item": response, "native": response}):
        visual.enrich_blocks_with_vlm([generated, native], base_url="fixture", api_key="fixture", model_id="fixture",
                                     cache_dir=tmp_path, filter_decorative=False)
    assert "999" not in generated.text
    assert old in native.text


def test_all_visual_facts_reach_chunks_with_conditions():
    facts = [f"Branch {i}: run stage {i + 1} only if gate {i} is closed; otherwise do not run." for i in range(25)]
    b = block("VISUAL SUMMARY: A workflow\n" + "\n".join("FACTS: " + f for f in facts), "chart",
              metadata={"vlm_status": "success", "visual_kind": "diagram", "visual_facts": facts})
    visual.tidy_visual_blocks([b], max_tokens=80)
    result = chunks([b])
    assert len(result) > 1
    for fact in facts:
        owners = [c for c in result if "FACTS: " + fact in c.text]
        assert len(owners) == 1 and fact in owners[0].block.metadata["visual_facts"]
        assert owners[0].block.block_id == b.block_id


def test_visual_fragments_preserve_graph_inputs_and_search_evidence(tmp_path):
    from types import SimpleNamespace
    from PIL import Image
    from kb_pipeline.pipeline.parse_job import _payload_for_chunk, select_visual_chunks
    from kb_pipeline.graph.units import ChunkRef, build_units
    from kb_search.evidence import source_row
    p = tmp_path / "workflow.png"
    Image.new("RGB", (200, 200)).save(p)
    facts = [f"Step {i}: do not open gate {i} unless timer {i} exceeds 17 seconds." for i in range(22)]
    b = block("\n".join("FACTS: " + f for f in facts), "chart", page_idx=3, visual_ref=str(p),
              metadata={"vlm_status": "success", "visual_facts": facts, "visual_kind": "diagram"})
    cs = chunks([b])
    row = {"kb_id": "kb_fixture", "file_key": 1, "source_path": "workflow.pdf", "rel_path": "workflow.pdf",
           "filename": "workflow.pdf", "dir": "", "content_version": "v", "size": 1, "mtime": 1, "mime_type": "application/pdf"}
    settings = SimpleNamespace(cache_dir=tmp_path, embedding_model_id="fixture")
    payloads = [_payload_for_chunk(settings, row, c, len(cs)) for c in cs]
    evidence = [source_row(i, {}, p) for i, p in enumerate(payloads)]
    for fact in facts:
        assert any(fact in e["text"] and fact in e["visual"]["facts"] for e in evidence)
    refs = [ChunkRef(point_id=str(i), chunk_uid=c.chunk_uid, doc_id="d", content_version="v", chunk_index=i,
                     block_id=c.block.block_id, block_type="chart", section_path=[], text=c.text,
                     n_tokens=c.token_count, page_idx=3) for i, c in enumerate(cs)]
    units = build_units(refs, kb_id="kb_fixture", unit_chunks=3)
    assert all(any(f in u.text for u in units) for f in facts)
    assert len(select_visual_chunks(cs)) == 1
    assert len({c.chunk_uid for c in cs}) == len(cs)


def test_previous_conclusion_does_not_become_next_chart_title():
    a = block("First result", "chart", block_id="first", page_idx=1, bbox=[50, 100, 400, 300])
    note = block("Conclusion: no further action.", block_id="note", page_idx=1, bbox=[50, 310, 400, 350])
    b = block("Second result", "chart", block_id="second", page_idx=1, bbox=[50, 400, 400, 700])
    merged = merge_mineru_text_blocks([a, note, b], target_tokens=800)
    assert not b.title
    result = chunks(merged, budget=400)
    assert all("Conclusion" not in c.text for c in result if c.block.block_id == "second")
    assert any("no further action" in c.text for c in result)


def test_heading_in_other_column_is_not_adopted():
    a = block("Installation", "title", block_id="heading", page_idx=1, bbox=[50, 50, 450, 100])
    b = block("A figure", "image", block_id="figure", page_idx=1, bbox=[550, 150, 950, 400])
    merge_mineru_text_blocks([a, b], target_tokens=800)
    assert b.title is None


def test_aligned_heading_is_adopted():
    a = block("Installation", "title", block_id="heading", page_idx=1, bbox=[50, 50, 450, 100])
    b = block("A figure", "image", block_id="figure", page_idx=1, bbox=[50, 150, 450, 400])
    merge_mineru_text_blocks([a, b], target_tokens=800)
    assert b.title == "Installation"


def test_continued_table_keeps_first_data_row_and_regions():
    a = block("| Item | Value |\n| --- | --- |\n| Alpha | 7 |", "table", block_id="a", page_idx=1,
              bbox=[50, 500, 950, 950])
    b = block("| Beta | 8 |\n| --- | --- |\n| Gamma | 9 |", "table", block_id="b", page_idx=2,
              bbox=[50, 50, 950, 300])
    a.table_markdown, b.table_markdown = a.text, b.text
    merged = merge_split_tables([a, b])
    assert len(merged) == 1 and merged[0].text.count("Beta") == 1
    assert len(merged[0].metadata["table_source_regions"]) == 2
    assert merged[0].metadata["page_end"] == 2


@pytest.mark.parametrize("same_page,bottom,top", [(True, 950, 50), (False, 500, 50), (False, 950, 500)])
def test_independent_tables_are_not_merged(same_page, bottom, top):
    a = block("| Item | Value |\n| --- | --- |\n| Alpha | 7 |", "table", block_id="a", page_idx=1,
              bbox=[50, 100, 950, bottom])
    b = block("| Beta | 8 |\n| --- | --- |", "table", block_id="b", page_idx=1 if same_page else 2,
              bbox=[50, top, 950, 700])
    a.table_markdown, b.table_markdown = a.text, b.text
    assert len(merge_split_tables([a, b])) == 2


def test_visible_forms_are_read_hidden_fields_are_not_and_replay_is_idempotent(tmp_path):
    pdf = pytest.importorskip("pymupdf")
    path = tmp_path / "form.pdf"
    with pdf.open() as doc:
        page = doc.new_page(width=400, height=400)
        for index, value in enumerate(["Approved after inspection", "Hidden field value"]):
            widget = pdf.Widget()
            widget.field_name = f"field_{index}"
            widget.field_type = pdf.PDF_WIDGET_TYPE_TEXT
            widget.rect = pdf.Rect(40, 40 + index * 80, 360, 90 + index * 80)
            widget.field_value = value
            annot = page.add_widget(widget)
            if index:
                doc.xref_set_key(annot.xref, "F", "2")
        doc.save(path)
    first = supplement_visible_forms([], path)
    assert len(first) == 1 and first[0].text == "Approved after inspection"
    assert first[0].page_idx == 1 and first[0].bbox
    assert len(supplement_visible_forms(deepcopy(first), path)) == 1


def test_blank_cell_alone_does_not_trigger_review(tmp_path):
    pdf = pytest.importorskip("pymupdf")
    path = tmp_path / "table.pdf"
    with pdf.open() as doc:
        p = doc.new_page(width=400, height=400)
        p.insert_text((40, 60), "Item")
        p.insert_text((200, 60), "Value")
        p.insert_text((40, 90), "Alpha")
        doc.save(path)
    b = block("| Item | Value |\n| --- | --- |\n| Alpha | |", "table", page_idx=1, bbox=[50, 50, 950, 300])
    b.table_markdown = b.text
    add_pdf_region_evidence([b], path, tmp_path)
    assert "table_review_reasons" not in b.metadata


def test_review_cannot_move_measurements_to_another_column():
    old = [["Item", "Input", "Output"], ["Alpha", "17", "29"]]
    assert _review_preserves_rows(old, deepcopy(old), "17 29")
    assert not _review_preserves_rows(old, [["Item", "Input", "Output"], ["Alpha", "29", "17"]], "17 29")
    assert not _review_preserves_rows(old, [["Item", "Input", "Output"], ["Beta", "17", "29"]], "17 29")
    assert not _review_preserves_rows([["Alpha", ""]], [["Alpha", "999"]], "17 29")
    assert not _review_preserves_rows([["Alpha", ""]], [["Alpha", "17 A"]], "Alpha 17 V")
    assert not _review_preserves_rows([["Alpha", "17"]], [["Alpha", "17 A"]], "Alpha 17 V")
    assert not _review_preserves_rows([["Alpha", "17 V"]], [["Alpha", "17 A"]], "17 A")
    assert not _review_preserves_rows([["Alpha", "< 17"]], [["Alpha", "> 17"]], "17")


def test_native_radical_font_mapping_is_not_missing_text():
    from kb_pipeline.parsers.pdf_evidence import _span_present
    assert _span_present("检查所⻅", "检查所见")
    assert _span_present("未⻅异常", "未见异常")
    assert not _span_present("压力 17", "压力18")


def test_duplicate_images_share_one_request_but_not_different_context(tmp_path):
    from PIL import Image
    from kb_pipeline.vision import vlm
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    Image.new("RGB", (200, 200)).save(a)
    b.write_bytes(a.read_bytes())
    with patch.object(vlm, "caption_image", return_value={"facts": ["Source fact"]}) as call:
        result = vlm.caption_images_parallel([("a", a, "context A", None), ("b", b, "context A", None),
                                             ("c", b, "context B", None)],
                                            base_url="fixture", api_key="fixture", model_id="fixture", concurrency=3)
    assert call.call_count == 2 and set(result) == {"a", "b", "c"}
    result["a"]["facts"].append("Changed")
    assert result["b"]["facts"] == ["Source fact"]


def test_unverified_review_retains_source_without_guessing_cells(tmp_path):
    b = block("| Item | Value |\n| --- | --- |\n| Alpha | |", "table", metadata={
        "table_review_reasons": ["native_text_missing"], "table_missing_spans": ["Batch identifier"],
        "table_source_text": "Batch identifier"})
    b.table_markdown = b.text
    original = b.text
    review_pdf_tables([b], base_url="fixture", api_key="", model_id="fixture", cache_dir=tmp_path)
    assert b.text.startswith(original) and "Batch identifier" in b.text
    assert b.metadata["table_review"]["status"] == "unverified"


def test_table_review_failure_keeps_readable_source(tmp_path):
    p = tmp_path / "crop.png"
    p.write_bytes(b"fixture")
    b = block("| Item | Value |\n| --- | --- |\n| Alpha | |", "table", metadata={
        "table_review_reasons": ["native_text_missing"], "table_missing_spans": ["A printed field"],
        "table_source_text": "A printed field", "table_review_image": str(p)})
    b.table_markdown = b.text
    with patch("kb_pipeline.parsers.table_repair.transcribe_table", side_effect=TimeoutError):
        review_pdf_tables([b], base_url="fixture", api_key="fixture", model_id="fixture", cache_dir=tmp_path)
    assert "Alpha" in b.text and "A printed field" in b.text
    assert b.metadata["table_review_error"] == "TimeoutError"


def test_review_preserves_labels_polarity_order_and_rejects_invented_rows():
    original = [["Item", "Status"], ["Alpha", "Not approved"], ["Beta", "Approved"]]
    assert _review_preserves_rows(original, deepcopy(original), "")
    changed = deepcopy(original)
    changed[1][1] = "Approved"
    assert not _review_preserves_rows(original, changed, "Alpha Not approved Beta Approved")
    assert not _review_preserves_rows(original, [original[0], original[2], original[1]], "")
    assert not _review_preserves_rows(original, original + [["Invented", "17"]], "")
    assert _review_preserves_rows([["Field", ""]], [["Field", "Batch identifier"]], "Field Batch identifier")


def test_valid_table_review_fills_printed_cells_and_keeps_notes(tmp_path):
    p = tmp_path / "crop.png"
    p.write_bytes(b"fixture")
    b = block("Table 1\n| Item | Value |\n| --- | --- |\n| Alpha | |\nNote: calibrated.", "table", metadata={
        "table_review_reasons": ["native_text_missing"], "table_missing_spans": ["17 V"],
        "table_source_text": "Item Value Alpha 17 V", "table_review_image": str(p),
        "table_native_grid": [["Item", "Value"], ["Alpha", "17 V"]]})
    b.table_markdown = b.text
    with patch("kb_pipeline.parsers.table_repair.transcribe_table", return_value={
            "rows": [["Item", "Value"], ["Alpha", "17 V"]], "legend": []}):
        stats = review_pdf_tables([b], base_url="fixture", api_key="fixture", model_id="fixture", cache_dir=tmp_path)
    assert stats["accepted"] == 1 and "17 V" in b.text and "Note: calibrated." in b.text


def test_visual_prose_budget_includes_repeated_title():
    from kb_pipeline.chunking.chunker import split_visual_text
    from kb_pipeline.utils import count_tokens
    text = "TITLE: Equipment operation\n" + "A long descriptive sentence. " * 100
    pieces = split_visual_text(text, 80)
    assert len(pieces) > 1 and all(p.startswith("TITLE: Equipment operation") for p in pieces)
    assert all(count_tokens(p) <= 80 for p in pieces)


def test_search_evidence_preserves_parse_review_and_source_regions():
    from kb_search.evidence import source_row
    payload = {"text": "A source cell", "source_regions": [{"page_idx": 2, "bbox": [1, 2, 3, 4]}],
               "source_fields": ["field_1"], "table_review": {"status": "unverified"}}
    evidence = source_row(1, {}, payload)
    for name in ("source_regions", "source_fields", "table_review"):
        assert evidence[name] == payload[name]


def test_repeated_row_keys_do_not_call_a_reviewer_that_cannot_align_them(tmp_path):
    p = tmp_path / "crop.png"
    p.write_bytes(b"fixture")
    b = block("| Item | Value |\n| --- | --- |\n| Alpha | |\n| Alpha | |", "table", metadata={
        "table_review_reasons": ["native_text_missing"], "table_missing_spans": ["17"],
        "table_source_text": "17", "table_review_image": str(p)})
    b.table_markdown = b.text
    with patch("kb_pipeline.parsers.table_repair.transcribe_table") as call:
        stats = review_pdf_tables([b], base_url="fixture", api_key="fixture", model_id="fixture", cache_dir=tmp_path)
    call.assert_not_called()
    assert stats["unverified"] == 1 and "17" in b.text


def test_truncated_caption_drops_the_partial_fact_and_partial_field():
    import json
    from kb_pipeline.vision.vlm import complete_json_prefix
    samples = [
        ('{"facts":["Gate is closed.","Open gate only if', {"facts": ["Gate is closed."]}),
        ('{"facts":["Gate is closed."],"summary":"Open gate only if', {"facts": ["Gate is closed."]}),
        ('{"facts":["Gate is closed."],"summ', {"facts": ["Gate is closed."]}),
        ('{"facts":["Gate is closed.","escaped \\', {"facts": ["Gate is closed."]}),
    ]
    for raw, expected in samples:
        assert json.loads(complete_json_prefix(raw)) == expected


def test_truncated_response_is_marked_and_not_cached(tmp_path):
    from types import SimpleNamespace as NS
    from PIL import Image
    from kb_pipeline.vision import vlm
    p = tmp_path / "figure.png"
    Image.new("RGB", (200, 200)).save(p)
    response = NS(choices=[NS(message=NS(content='{"summary":"A gate.","facts":["Gate is closed.","Open gate only if'), finish_reason="length")])
    client = NS(chat=NS(completions=NS(create=lambda **kw: response)))
    cache = tmp_path / "caption.json"
    with patch.object(vlm, "_client_for", return_value=client):
        result = vlm.caption_image(image_path=p, base_url="fixture", api_key="fixture", model_id="fixture", cache_json=cache)
    assert result["vlm_truncated"] and result["facts"] == ["Gate is closed."]
    assert not cache.exists()


def test_caption_output_budget_changes_cache_identity(tmp_path):
    from types import SimpleNamespace as NS
    from unittest.mock import Mock
    from PIL import Image
    from kb_pipeline.vision import vlm
    p = tmp_path / "figure.png"
    Image.new("RGB", (200, 200)).save(p)
    create = Mock(return_value=NS(choices=[NS(message=NS(content='{"summary":"A gate.","facts":["Gate is closed."]}'), finish_reason="stop")]))
    client = NS(chat=NS(completions=NS(create=create)))
    with patch.object(vlm, "_client_for", return_value=client):
        for budget in (200, 200, 400):
            vlm.caption_image(image_path=p, base_url="fixture", api_key="fixture", model_id="fixture",
                              cache_json=tmp_path / "caption.json", max_tokens=budget)
    assert create.call_count == 2


def test_partial_table_review_retains_rejected_row_but_fills_supported_row(tmp_path):
    p = tmp_path / "crop.png"
    p.write_bytes(b"fixture")
    b = block("| Item | Value |\n| --- | --- |\n| Alpha | |\n| Beta | 29 V |", "table", metadata={
        "table_review_reasons": ["native_text_missing"], "table_missing_spans": ["17 V"],
        "table_source_text": "Item Value Alpha 17 V Beta 29 V", "table_review_image": str(p),
        "table_native_grid": [["Item", "Value"], ["Alpha", "17 V"], ["Beta", "29 V"]]})
    b.table_markdown = b.text
    with patch("kb_pipeline.parsers.table_repair.transcribe_table", return_value={
            "rows": [["Item", "Value"], ["Alpha", "17 V"], ["Beta", "29 A"]], "legend": []}):
        review_pdf_tables([b], base_url="fixture", api_key="fixture", model_id="fixture", cache_dir=tmp_path)
    assert "| Alpha | 17 V |" in b.text and "| Beta | 29 V |" in b.text and "29 A" not in b.text
    assert b.metadata["table_review"]["status"] == "partial"


@pytest.mark.parametrize("kind", ["screenshot", "chart", "diagram", "table", "formula"])
def test_content_bearing_visual_is_not_discarded_by_decorative_flag(kind):
    b = block("FACTS: Valve is closed.", kind if kind in {"chart"} else "image", visual_ref="fixture.png",
              metadata={"visual_kind": kind, "visual_facts": ["Valve is closed."], "vlm_status": "success"})
    b.metadata["decorative"] = visual.is_decorative_result(kind, "An instrument", facts=["Valve is closed."],
                                                         entities=[], verbatim="", flagged=True)
    assert not b.metadata["decorative"]
    assert any("Valve is closed." in c.text for c in chunks([b]))


def test_explanatory_footnote_is_not_automatically_a_symbol_legend(tmp_path):
    pdf = pytest.importorskip("pymupdf")
    path = tmp_path / "table.pdf"
    with pdf.open() as doc:
        page = doc.new_page(width=400, height=400)
        page.insert_text((40, 60), "Item")
        page.insert_text((200, 60), "Value")
        page.insert_text((40, 90), "Alpha")
        doc.save(path)
    b = block("| Item | Value |\n| --- | --- |\n| Alpha | |", "table", page_idx=1, bbox=[50, 50, 950, 300],
              metadata={"source_footnote": "Results represent observations under the stated conditions."})
    b.table_markdown = b.text
    add_pdf_region_evidence([b], path, tmp_path)
    assert "table_review_reasons" not in b.metadata



def test_missing_table_text_follows_page_order_not_pdf_drawing_order(tmp_path):
    import pymupdf
    from kb_pipeline.parsers.pdf_evidence import add_pdf_region_evidence
    p=tmp_path/'source.pdf';doc=pymupdf.open();page=doc.new_page(width=400,height=400)
    page.insert_text((40,160),'second line')
    page.insert_text((40,140),'first line')
    page.insert_text((40,100),'Key Value')
    doc.save(p);doc.close()
    b=block('| Key | Value |\n| --- | --- |','table',page_idx=1,bbox=[50,150,950,500])
    b.table_markdown=b.text
    add_pdf_region_evidence([b],p,tmp_path)
    assert b.metadata['table_missing_spans']==['first line','second line']
    review_pdf_tables([b],base_url='fixture',api_key='',model_id='fixture',cache_dir=tmp_path)
    assert 'first line\nsecond line' in b.text
