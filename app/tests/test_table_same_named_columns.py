"""Same-named columns in table files (2026-11-20, follow-up to C11): the parser tells them apart while the column
positions are still known, so a value is filed under the same column whichever way its row is written. A row written
by position keeps empty cells; a row written column by column ("name: value | name: value") skips them, and with two
columns both called "Price" a lone "Price: 30" no longer says which of the two it came from.
Synthetic inputs only, independent of any deployed knowledge base."""
import csv

import pytest

from kb_pipeline.graph import tables, tabletext
from kb_pipeline.graph.extract import ExtractionSchema
from kb_pipeline.graph.tables import TableExtractor, parse_table_text
from kb_pipeline.parsers.native_table import chunk_rows_to_blocks, distinct_names, parse_native_table

from _support import _client, _unit

WIDTH, FIRST, SECOND = 83, 40, 80          # a wide sheet and the 0-based positions of its two "Price" columns
BUDGETS = (800, 400, 300, 120)


def _data(blocks):
    return [b for b in blocks if not (b.metadata or {}).get("summary")]


def _written_by_position(block) -> bool:
    return "\nHEADER: " in block.text


def _read(blocks, key: str = "Item") -> dict[str, dict[str, str]]:
    """{row identifier: {column: value}}, the non-empty values as the graph's table reader sees these blocks."""
    out: dict[str, dict[str, str]] = {}
    for block in _data(blocks):
        for section in parse_table_text(block.text):
            for row in section["rows"]:
                values = tables._row_values(section, row) or {}
                if values.get(key):
                    out.setdefault(values[key], {}).update({k: v for k, v in values.items() if v})
    return out


def _numbered(rows: list[list[str]]) -> list[tuple[int, list[str]]]:
    return list(enumerate(rows, start=1))


def _wide_rows() -> list[list[str]]:
    header = [f"Field{i:02d}" for i in range(WIDTH)]
    header[0], header[FIRST], header[SECOND] = "Item", "Price", "Price"
    rows = [header]
    for n in range(1, 7):
        row = [""] * WIDTH
        row[0], row[5], row[60] = f"item-{n}", f"note-{n}", f"tag-{n}"
        rows.append(row)
    rows[1][FIRST], rows[1][SECOND] = "10", "11"       # both prices
    rows[2][SECOND] = "30"                             # only the second
    rows[3][FIRST] = "50"                              # only the first
    return rows


def _prices(values: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in values.items() if k.startswith("Price")}


# ── the naming rule ──────────────────────────────────────────────────────

def test_same_named_columns_are_numbered_by_position():
    assert distinct_names(["Item", "Price", "Note"]) == ["Item", "Price", "Note"]
    assert distinct_names(["A", "A", "", "", "B", "A"]) == ["A", "A (2)", "", "", "B", "A (3)"]
    assert distinct_names(["A", "A (2)", "A"]) == ["A", "A (2)", "A (3)"]            # a real "A (2)" keeps its name
    assert distinct_names([]) == []


@pytest.mark.parametrize("names", [["A", "A", "A"], ["A", "A (2)", "A"], ["", "x", "", "x"], ["Price", "Price (2)", "Price", "Price"]])
def test_naming_twice_changes_nothing(names):
    once = distinct_names(names)
    assert distinct_names(once) == once
    assert len({n for n in once if n}) == len([n for n in once if n])


def test_the_reader_uses_the_rule_of_the_writer():
    assert tables._distinct is distinct_names


# ── what the parser writes ───────────────────────────────────────────────

def test_header_and_column_list_tell_same_named_columns_apart():
    blocks = chunk_rows_to_blocks("csv", _numbered([["Item", "Price", "Price", "Note"], ["a", "10", "11", "x"],
                                                    ["b", "", "30", "y"], ["c", "50", "", "z"]]), None, 400, 0)
    summary, (data,) = blocks[0], _data(blocks)
    assert "Columns: Item | Price | Price (2) | Note" in summary.text.splitlines()
    assert data.text.splitlines() == ["ROWS: 2-4", "HEADER: Item | Price | Price (2) | Note",
                                      "a | 10 | 11 | x", "b |  | 30 | y", "c | 50 |  | z"]
    assert _read(blocks) == {"a": {"Item": "a", "Price": "10", "Price (2)": "11", "Note": "x"},
                             "b": {"Item": "b", "Price (2)": "30", "Note": "y"},
                             "c": {"Item": "c", "Price": "50", "Note": "z"}}


def test_a_table_without_same_named_columns_is_written_as_before():
    blocks = chunk_rows_to_blocks("xlsx", _numbered([["Item", "Unit\nPrice", "", "Note (2)"], ["a", "10", "k", "x"],
                                                     ["b", "", "", "y"]]), "Prices", 400, 0)
    assert blocks[0].text.splitlines() == ["Sheet: Prices", "Rows: 3", "Columns: Item | Unit Price |  | Note (2)"]
    assert blocks[1].text.splitlines() == ["SHEET: Prices", "ROWS: 2-3", "HEADER: Item | Unit Price |  | Note (2)",
                                           "a | 10 | k | x", "b |  |  | y"]


def test_a_row_written_column_by_column_says_which_column_a_value_came_from():
    long_note = " ".join(f"word{i}" for i in range(400))          # one cell longer than any budget: the row is expanded
    blocks = chunk_rows_to_blocks("csv", _numbered([["Item", "Price", "Price", "Note"], ["a", "10", "11", "x"],
                                                    ["b", "", "30", long_note], ["c", "50", "", "z"]]), None, 120, 0)
    expanded = [b for b in _data(blocks) if not _written_by_position(b) and "Item: b" in b.text]
    assert expanded and "Price (2): 30" in expanded[0].text
    assert not any("Price: 30" in b.text for b in blocks)
    got = _read(blocks)
    assert _prices(got["a"]) == {"Price": "10", "Price (2)": "11"}
    assert _prices(got["b"]) == {"Price (2)": "30"}
    assert _prices(got["c"]) == {"Price": "50"}


def test_names_that_differ_only_in_line_breaks_are_the_same_name():
    # "Unit⏎Price" and "Unit Price" are both written "Unit Price": apart on the header line, apart in an expanded row
    blocks = chunk_rows_to_blocks("xlsx", _numbered([["Item", "Unit\nPrice", "Unit  Price"], ["a", "10", "11"]]), "S", 400, 0)
    assert "HEADER: Item | Unit Price | Unit Price (2)" in blocks[1].text.splitlines()


def test_a_header_cell_merged_across_columns():
    blocks = chunk_rows_to_blocks("xlsx", _numbered([["Item", "Amount", "Amount", "Amount"], ["a", "1", "2", "3"]]), "S", 400, 0)
    assert "HEADER: Item | Amount | Amount (2) | Amount (3)" in blocks[1].text.splitlines()
    assert _read(blocks)["a"] == {"Item": "a", "Amount": "1", "Amount (2)": "2", "Amount (3)": "3"}


def test_two_header_rows_name_the_columns_before_they_are_told_apart():
    # "Price" spans two columns over "List" / "List": the merged names collide and are numbered; "Qty" does not collide
    rows = _numbered([["Item", "Price", "Price", "Qty"], ["Item", "List", "List", "Qty"], ["a", "10", "11", "2"], ["b", "", "30", "1"]])
    blocks = chunk_rows_to_blocks("xlsx", rows, "S", 400, 0)
    assert "HEADER: Item | Price/List | Price/List (2) | Qty" in blocks[1].text.splitlines()
    assert _prices(_read(blocks)["b"]) == {"Price/List (2)": "30"}


# ── the same answer whichever way the rows are written ───────────────────

def test_a_wide_table_reads_the_same_at_every_budget(tmp_path):
    path = tmp_path / "wide.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(_wide_rows())
    ways = set()
    for budget in BUDGETS:
        blocks = parse_native_table(path, max_tokens=budget, overlap_tokens=0)
        ways |= {_written_by_position(b) for b in _data(blocks)}
        got = _read(blocks)
        assert _prices(got["item-1"]) == {"Price": "10", "Price (2)": "11"}, budget
        assert _prices(got["item-2"]) == {"Price (2)": "30"}, budget            # was filed under "Price" once the row was expanded
        assert _prices(got["item-3"]) == {"Price": "50"}, budget
        assert not _prices(got["item-4"]), budget
    assert ways == {True, False}                                              # both ways of writing a row were exercised


def test_same_named_columns_in_different_pieces_of_one_row_are_both_kept():
    width = 30
    header = [f"Field{i:02d}" for i in range(width)]
    header[0], header[2], header[27] = "Item", "Price", "Price"
    row = [f"value number {i} of the row" for i in range(width)]
    row[0], row[2], row[27] = "item-1", "10", "11"
    blocks = chunk_rows_to_blocks("csv", _numbered([header, row, ["item-2"] + [""] * 26 + ["30"]]), None, 80, 0)
    pieces = [line for b in _data(blocks) for line in b.text.splitlines() if line.startswith("Item: item-1")]
    assert len(pieces) > 2
    assert [i for i, line in enumerate(pieces) if "| Price: 10" in line] != [i for i, line in enumerate(pieces) if "| Price (2): 11" in line]
    got = _read(blocks)
    assert _prices(got["item-1"]) == {"Price": "10", "Price (2)": "11"}          # the second one used to be dropped
    assert _prices(got["item-2"]) == {"Price (2)": "30"}


def test_an_xlsx_sheet_takes_the_same_path(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Prices"
    for row in (["Item", "Price", "Price"], ["a", 10, 11], ["b", None, 30], ["c", 50, None]):
        ws.append(row)
    path = tmp_path / "prices.xlsx"
    wb.save(path)
    blocks = parse_native_table(path, max_tokens=400, overlap_tokens=0)
    assert "HEADER: Item | Price | Price (2)" in _data(blocks)[0].text.splitlines()
    assert _prices(_read(blocks)["b"]) == {"Price (2)": "30"}


# ── downstream readers of the same text ──────────────────────────────────

def test_the_rule_route_files_the_value_under_its_own_column(tmp_path):
    path = tmp_path / "wide.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(_wide_rows())
    blocks = parse_native_table(path, max_tokens=300, overlap_tokens=0)
    assert not any(_written_by_position(b) for b in _data(blocks))            # every row is written column by column
    unit = _unit("u1", doc="kb:t")
    unit.rel_path, unit.text, unit.block_types = "sales/wide.csv", "\n\n".join(b.text for b in blocks), ["table"]
    tx = TableExtractor([unit], schema=ExtractionSchema(entity_types=("product",), parent_types={"product": "entity"}))
    tx.profile(_client(lambda messages: '{"row_type": "product", "key_column": "", "name_column": "Item", "references": [], "date_column": ""}'))
    assert tx.wants(unit)
    facts = {(f["subject"], f["property"], f["value"]) for f in tx.facts(unit)}
    assert {("item-1", "Price", "10"), ("item-1", "Price (2)", "11"), ("item-2", "Price (2)", "30"), ("item-3", "Price", "50")} <= facts
    assert ("item-2", "Price", "30") not in facts


def test_the_text_shown_to_the_model_names_the_columns_apart():
    rows = [["Item", "A", "B", "C", "D", "Price", "Price"], ["x", "1", "2", "3", "4", "", "30"]]
    blocks = chunk_rows_to_blocks("csv", _numbered(rows), None, 400, 0)
    rendered = tabletext.expand_wide_tables(blocks[1].text) or blocks[1].text
    assert "Price (2): 30" in rendered and "Price: 30" not in rendered


def test_a_chunk_written_before_this_change_is_still_read():
    # same-named columns on an old header line are numbered by position, exactly as the parser now names them
    (section,) = parse_table_text("ROWS: 2-2\nHEADER: Item | Price | Price | Note\nb |  | 30 | y")
    assert tables._row_values(section, section["rows"][0]) == {"Item": "b", "Price": "", "Price (2)": "30", "Note": "y"}
