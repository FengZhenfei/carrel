"""Synthetic PDF layouts and adversarial evidence checks, independent of any corpus."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from kb_pipeline.models import ParsedBlock
from kb_pipeline.parsers.pdf_code import recover_native_code
from kb_pipeline.parsers.mineru_quality import recover_pdf_items
from kb_pipeline.parsers.pdf_evidence import _norm, _span_present
from kb_pipeline.parsers.table_repair import _review_preserves_rows, _verified_row_key
from kb_pipeline.chunking.chunker import blocks_to_chunks, embedding_context_for
from kb_pipeline.graph.units import fetch_chunk_payloads, build_units


def listing(text, **kw):
    return ParsedBlock("fixture", "fixture-v1", "pdf", "code", text, "listing",
                       page_idx=1, bbox=[0, 0, 1000, 1000], **kw)


def pdf(tmp_path, lines, font="cour", positioned=False):
    fitz = pytest.importorskip("pymupdf")
    dest = tmp_path / "source.pdf"
    with fitz.open() as doc:
        page = doc.new_page(width=600, height=800)
        for i, line in enumerate(lines):
            spaces = len(line) - len(line.lstrip()) if positioned else 0
            page.insert_text((40 + spaces * 6, 50 + i * 14), line[spaces:], fontname=font, fontsize=10)
        doc.save(dest)
    return dest


@pytest.mark.parametrize("positioned", [False, True])
def test_missing_definition_and_identifier_recovered_with_indent(tmp_path, positioned):
    source = ["message = 'Keep the original instructions.'", "def schedule(value):",
              "    result = client.requests.create(value)", "    return result"]
    path = pdf(tmp_path, source, positioned=positioned)
    b = listing("def schedule(value):\n    result = client.requestts.create(value)\n    return result")
    assert recover_native_code([b], path) == 1
    assert b.text == "\n".join(source) + "\n"
    assert "requestts" in b.metadata["pdf_code_recovery"]["ocr_text"]
    assert b.page_idx == 1 and b.bbox == [0, 0, 1000, 1000]


@pytest.mark.parametrize("font,old,kind", [
    ("helv", "def schedule(value):\n    return value", "code"),
    ("cour", "class Different:\n    def calculate(self, other):\n        return other * 123", "code"),
    ("cour", "def schedule(value):\n    return value", "text"),
])
def test_unrelated_or_non_monospace_or_body_text_is_unchanged(tmp_path, font, old, kind):
    path = pdf(tmp_path, ["def schedule(value):", "    return value"], font=font)
    b = listing(old); b.block_type = kind
    before = deepcopy(b)
    assert recover_native_code([b], path) == 0
    assert b == before


def test_incomplete_native_layer_does_not_delete_ocr_lines(tmp_path):
    path = pdf(tmp_path, ["def schedule(value):", "    return value"])
    b = listing("def schedule(value):\n    total = value + 123\n    record(total)\n    return value")
    assert recover_native_code([b], path) == 0


def test_one_missing_line_in_a_long_native_listing_keeps_ocr(tmp_path):
    source = [f"value_{i} = {i}" for i in range(20)]
    original = source[:10] + ["audit_result = check_constraints(required_policy)"] + source[10:]
    path = pdf(tmp_path, source)
    b = listing("\n".join(original))
    assert recover_native_code([b], path) == 0
    assert "required_policy" in b.text


def test_native_recovery_does_not_read_the_adjacent_column(tmp_path):
    fitz = pytest.importorskip("pymupdf")
    path = pdf(tmp_path, ["def schedule(value):", "    return value"])
    with fitz.open(path) as doc:
        doc[0].insert_text((350, 50), "secret = 'neighbour'", fontname="cour", fontsize=10)
        doc.saveIncr()
    b = listing("def schedule(value):\n    return value"); b.bbox = [0, 0, 500, 1000]
    recover_native_code([b], path)
    assert "neighbour" not in b.text and "return value" in b.text


def test_printed_line_number_gutter_is_not_code(tmp_path):
    fitz = pytest.importorskip("pymupdf")
    source = ["def schedule(value):", "    amount = value + 17", "    return amount"]
    path = pdf(tmp_path, source)
    with fitz.open(path) as doc:
        for i in range(3):
            doc[0].insert_text((20, 50 + i*14), str(i+1), fontname="helv", fontsize=5)
        doc.saveIncr()
    b = listing("```python\n" + "\n".join(source) + "\n```")
    assert recover_native_code([b], path) == 1
    assert b.text == "\n".join(source) + "\n"
    assert b.metadata["code_language"] == "python"


def test_numeric_code_values_are_not_a_printed_gutter(tmp_path):
    source = ["values = [", "    1,", "    2,", "    3,", "]"]
    path = pdf(tmp_path, source)
    b = listing("\n".join(source))
    recover_native_code([b], path)
    assert b.text == "\n".join(source) + "\n"


def item(text, box, kind="text", page=0):
    return {"type": kind, "text": text, "bbox": box, "page_idx": page}


def test_restored_header_returns_to_its_column_without_sorting_body():
    left = item("left paragraph", [40, 250, 440, 400])
    right = item("right paragraph", [550, 100, 950, 400])
    heading = item("Local heading", [40, 100, 440, 150], "header")
    lower = item("left continuation", [40, 450, 440, 550])
    result = recover_pdf_items([left, lower, right, heading], {})
    assert [r["text"] for r in result] == ["Local heading", "left paragraph", "left continuation", "right paragraph"]
    assert result[0]["original_source_type"] == "header"


def test_repeated_running_header_is_not_restored():
    result = recover_pdf_items([item("body", [40, 100, 900, 300]),
        item("Manual", [40, 10, 900, 25], "header"), item("Manual", [40, 10, 900, 25], "header", 1)], {})
    assert [i["type"] for i in result] == ["text", "header", "header"]


@pytest.mark.parametrize("span,body,equal", [
    ("3–7 kg", "3-7 kg", True), ("$10—$20", "$10-$20", True),
    ("3−7", "3-7", False), ("−7", "7", False), ("≤7", "7", False),
    ("3–7 kg", "3-7 mg", False), ("3–7", "3-8", False),
])
def test_numeric_range_typography_preserves_signs_units_and_values(span, body, equal):
    assert _span_present(span, _norm(body)) is equal


def table_case():
    old = [["Name", "Count"], ["a110_base", "14"], ["beta", "21"]]
    new = [["Name", "Count"], ["al10_base", "14"], ["beta", "21"]]
    spans = [{"text": row[0], "bbox": [20, 20 + 30*i, 85, 30 + 30*i]} for i, row in enumerate(new)]
    return old, new, spans


def test_row_key_typo_requires_independent_geometry_and_unchanged_cells():
    old, new, spans = table_case()
    assert not _review_preserves_rows([old[1]], [new[1]], "al10_base 14")
    assert _verified_row_key(old, new, 1, spans)
    assert not _verified_row_key(old, new, 1, [])


def test_row_anchor_can_span_fonts_without_joining_columns():
    old, new, spans = table_case()
    old[2][0] = new[2][0] = "beta (B)"
    spans[-1] = {"text": "beta", "bbox": [20, 80, 50, 90]}
    spans.append({"text": " (B)", "bbox": [50, 79, 85, 89]})
    assert _verified_row_key(old, new, 1, spans)


@pytest.mark.parametrize("attack", ["value", "column", "order", "duplicate", "old_present", "unit"])
def test_correct_key_cannot_authorize_other_changes(attack):
    old, new, spans = table_case()
    if attack == "value": new[1][1] = "21"
    if attack == "column": spans[1]["bbox"] = [150, 50, 200, 60]
    if attack == "order": spans[1]["bbox"][1:] = [90, 85, 100]
    if attack == "duplicate": spans.append(deepcopy(spans[1]))
    if attack == "old_present": spans.append({"text": old[1][0], "bbox": [20, 140, 85, 150]})
    if attack == "unit": new[1][1] = "14 kg"
    assert not _verified_row_key(old, new, 1, spans)


def test_compact_label_remains_in_embedding_and_graph_without_extra_chunk():
    b = listing("result = 17\n", caption="Listing-A")
    cs = blocks_to_chunks(kb_id="kb_fixture", file_key=1, content_version="v", parser_profile="p",
                          blocks=[b], max_tokens=100, overlap_tokens=0)
    assert len(cs) == 1 and cs[0].text == b.text
    assert "Listing-A" in embedding_context_for(cs[0], doc_name="Example")
    payload = {"text": cs[0].text, "block_type": "code", "caption": b.caption,
               "code_context": cs[0].block.metadata["code_context"], "chunk_uid": cs[0].chunk_uid,
               "doc_id": "d", "content_version": "v", "token_count": cs[0].token_count}
    q = SimpleNamespace(retrieve=lambda **kw: [SimpleNamespace(id="p", payload=payload)])
    refs = fetch_chunk_payloads(q, "fixture", [{"point_id": "p"}])
    from kb_pipeline.utils import count_tokens
    assert refs[0].n_tokens == count_tokens(refs[0].text)
    units = build_units(refs, kb_id="kb_fixture", unit_chunks=3)
    assert "CAPTION: Listing-A" in units[0].text and "result = 17" in units[0].text
