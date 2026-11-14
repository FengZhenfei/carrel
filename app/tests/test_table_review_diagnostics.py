"""Review empty columns without changing cell ownership or hiding uncertainty."""
from copy import deepcopy
from unittest.mock import patch

import pytest

from kb_pipeline.chunking.diagnose import chunk_diagnostics
from kb_pipeline.models import ParsedBlock
from kb_pipeline.parsers.common import grid_to_markdown
from kb_pipeline.parsers.table_repair import (
    _identity_columns, _native_row_text, _pipe_grid, review_pdf_tables,
)


def table(tmp_path, rows, **metadata):
    image = tmp_path / "table.png"
    image.write_bytes(b"synthetic image; model response is stubbed")
    text = grid_to_markdown(rows) if rows else "Source caption without table rows"
    return ParsedBlock(parser="fixture", parser_profile="fixture", doc_type="pdf",
        block_type="table", block_id="table", text=text, table_markdown=text,
        metadata={"table_review_reasons": ["native_text_missing"],
                  "table_review_image": str(image), **metadata})


def review(tmp_path, block, **kwargs):
    return review_pdf_tables([block], base_url="fixture", api_key="fixture",
                            model_id="fixture", cache_dir=tmp_path, **kwargs)


@pytest.mark.parametrize("positions", [(0,), (0, 1), (1,), (2,)])
def test_empty_columns_keep_alignment_for_native_cell_fills(tmp_path, positions):
    old = [["Item", "Value"], ["Alpha", ""], ["Beta", "9 V"]]
    native = [["Item", "Value"], ["Alpha", "17 V"], ["Beta", "9 V"]]
    for position in positions:
        for rows in (old, native):
            for row in rows:
                row.insert(position, "")
    block = table(tmp_path, old, table_native_grid=native,
                  table_source_text="Item Value Alpha 17 V Beta 9 V", table_missing_spans=["17 V"])
    with patch("kb_pipeline.parsers.table_repair.transcribe_table", return_value={"rows": native}) as call:
        assert review(tmp_path, block) == {"requested": 1, "accepted": 1, "unverified": 0}
    call.assert_called_once()
    assert _pipe_grid(block.table_markdown) == native
    assert all(row[i] == "" for row in _pipe_grid(block.text) for i in positions)


def test_identity_skips_only_wholly_empty_columns_and_retains_key_limit():
    assert _identity_columns([["", "Part", "", "x1", "4"], ["", "Part", "", "x2", "4"]]) == (1, 3)
    for rows in (
        [["", "Part", "4"], ["", "Part", "4"]],  # duplicate rows
        [["Item", "Value"], ["", "4"], ["Beta", "5"]],  # missing row key
        [["", ""], ["", ""]], [], [["", "A"], [""]],
        [["A", "B", "C", "x1"], ["A", "B", "C", "x2"]],  # fourth key not searched
    ):
        assert _identity_columns(rows) == ()


def test_row_native_text_uses_the_selected_column():
    rows = [["", "Item", "Value"], ["", "Alpha", "suffix"], ["", "Beta", "other"]]
    spans = [{"text": text, "bbox": box} for text, box in [
        ("Item", [20, 0, 50, 10]), ("Alpha", [20, 20, 50, 30]),
        ("Full suffix", [60, 20, 100, 30]), ("Beta", [20, 40, 50, 50]),
        ("Other row", [60, 40, 100, 50]),
    ]]
    text = _native_row_text(rows, 1, _identity_columns(rows), spans)
    assert "Full suffix" in text and "Other row" not in text


@pytest.mark.parametrize("before,replacement", [
    ("17 V", "19 V"), ("17 V", "17 A"), ("17 V", "-17 V"),
    (">17 V", "<17 V"), ("17 V", "⚠"),
])
def test_empty_leading_column_does_not_weaken_value_checks(tmp_path, before, replacement):
    old = [["", "Item", "Value"], ["", "Alpha", before]]
    new = deepcopy(old)
    new[1][2] = replacement
    block = table(tmp_path, old, table_source_text="Alpha " + replacement)
    with patch("kb_pipeline.parsers.table_repair.transcribe_table", return_value={"rows": new}):
        assert review(tmp_path, block)["accepted"] == 0
    assert _pipe_grid(block.text) == old
    assert block.metadata["table_review"]["status"] == "partial"


def test_native_cell_grid_must_match_selected_row_keys(tmp_path):
    old = [["", "Item", "Value"], ["", "Alpha", ""], ["", "Beta", ""]]
    new = deepcopy(old)
    new[1][2] = "17 V"
    native = deepcopy(new)
    native[1][1], native[2][1] = native[2][1], native[1][1]
    block = table(tmp_path, old, table_native_grid=native, table_source_text="Alpha Beta 17 V")
    with patch("kb_pipeline.parsers.table_repair.transcribe_table", return_value={"rows": new}):
        assert review(tmp_path, block)["accepted"] == 0
    assert _pipe_grid(block.text) == old
    assert block.metadata["table_cell_alignment_unverified"] is True


@pytest.mark.parametrize("rows,reason", [([], "missing_table_body"),
    ([["", "A"], ["", "A"]], "ambiguous_row_keys")])
def test_review_skips_report_the_actual_reason_and_keep_source(tmp_path, rows, reason):
    block = table(tmp_path, rows, table_missing_spans=["Printed source note"])
    with patch("kb_pipeline.parsers.table_repair.transcribe_table") as call:
        assert review(tmp_path, block) == {"requested": 0, "accepted": 0, "unverified": 1}
    call.assert_not_called()
    assert block.metadata["table_review"]["reason"] == reason
    assert "Printed source note" in block.text


def test_failed_request_is_distinct_from_an_unconfirmed_response(tmp_path):
    for response, reason in [(TimeoutError("private endpoint detail"), "request_failed"),
                             ({}, "verification_incomplete")]:
        block = table(tmp_path, [["Item", "Value"], ["Alpha", "17 V"]])
        options = {"side_effect": response} if isinstance(response, Exception) else {"return_value": response}
        with patch("kb_pipeline.parsers.table_repair.transcribe_table", **options):
            assert review(tmp_path, block)["unverified"] == 1
        assert block.metadata["table_review"]["reason"] == reason
        assert "private endpoint detail" not in str(block.metadata)


def test_diagnostics_group_outcomes_and_count_each_table_once(tmp_path):
    cases = [
        ("unverified", "missing_table_body", "table_review_missing_body"),
        ("unverified", "ambiguous_row_keys", "table_review_ambiguous_rows"),
        ("unverified", "missing_model_credentials", "table_review_missing_model"),
        ("unverified", "missing_image", "table_review_missing_image"),
        ("unverified", "request_failed", "table_review_request_failed"),
        ("unverified", "verification_incomplete", "table_review_unconfirmed"),
        ("partial", "rows_unverified", "table_review_partial"),
        ("partial", "source_text_missing", "table_review_partial"),
        ("unverified", "unknown", "table_source_alignment"),
    ]
    blocks = []
    for status, reason, _ in cases + [("accepted", "", "")]:
        blocks.append(table(tmp_path, [["Item", "Value"]], table_review={"status": status, "reason": reason}))
    result = chunk_diagnostics([], max_tokens=400, blocks=blocks)
    assert result["stats"]["table_reviews_unverified"] == len(cases)
    flags = {r["key"]: r["params"] for r in result["reasons"] if r["key"].startswith("table_")}
    assert flags == {key: [2 if key == "table_review_partial" else 1] for _, _, key in cases}
