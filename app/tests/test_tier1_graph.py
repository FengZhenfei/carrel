"""Graph-side fixes of 2026-11-20: reference names (C1), relation strength (C2), the rule route for table files
(C11) and the date of a fact at recall (C12). Synthetic inputs only, independent of any deployed knowledge base."""
from types import SimpleNamespace

import pytest

from kb_pipeline.graph import prompts
from kb_pipeline.graph.extract import ExtractionSchema, parse_records
from kb_pipeline.graph.merge import is_reference_name, merge_extractions
from kb_pipeline.graph.recall import collapse_duplicate_facts, spec_seeds
from kb_pipeline.graph.tables import TableExtractor, parse_table_text

from _support import _client, _ext, _unit


# ── C1: a reference is a keyword followed by its number ──────────────────

@pytest.mark.parametrize("name", [
    "Python", "PostgreSQL", "Pfizer", "PayPal", "PLL", "PCB", "Product", "Products", "Plan", "Participant", "PM2.5",   # p + letters
    "Notes", "Notebook", "Figma", "Tableau", "tablets", "Security", "Equinix",                                        # other keywords + letters
    "Review", "Revisions", "Revert",                                                                                  # rev + letters
    "P. Smith", "table tennis", "Note Taking", "Section Chief", "Equation Solver",                                    # keyword, separator, a word
    "表 Orders", "表Items",                                                                                            # a database table, not "Table 3"
])
def test_ordinary_names_are_not_references(name):
    assert not is_reference_name(name)


@pytest.mark.parametrize("name", [
    "Section 3", "Sec. 4.2", "Fig. 2", "Fig.2", "Figure 5a", "Table 4", "Table 3.2", "Table 3-1", "table1", "Chapter 7", "Page 12",
    "p. 12", "p 12", "p12", "Note 15", "Footnote 3", "Eq. 5", "Equation 12", "Appendix 1",
    "图5", "图 5", "表 3.2", "注 2", "公式 7",
    "Appendix A", "Appendix B.1", "Table A1", "Figure A-2", "附录A", "附录 A", "图a",          # a letter label: one letter, optional number
    "000-12345", "001-79553 *J", "Rev *J", "Rev. *J", "Rev A", "Rev. B", "Revision 2", "rev2", "*J",
])
def test_numbered_references_are_still_references(name):
    assert is_reference_name(name)


def test_only_the_numbered_reference_is_flagged_when_the_graph_is_merged():
    extraction = _ext(
        [("Acme Gateway", "device", "a gateway"), ("Python", "technology", "the language"), ("Notes", "document", "release notes"),
         ("Figure 5", "document", "block diagram")],
        [("Acme Gateway", "Python", "implemented_in", "written in Python", 8), ("Acme Gateway", "Notes", "has_document", "has notes", 5),
         ("Acme Gateway", "Figure 5", "has_document", "shown in figure 5", 3)])
    out = merge_extractions([_unit("u1")], {"u1": extraction}, entity_types=["device", "technology", "document"])
    flagged = {e["title"] for e in out["entities"] if e.get("reference")}
    assert flagged == {"Figure 5"}
    assert {(r["target"], bool(r.get("reference"))) for r in out["relations"]} == {("Python", False), ("Notes", False), ("Figure 5", True)}
    assert (out["stats"]["reference_entities"], out["stats"]["reference_relations"]) == (1, 1)


# ── C2: the strength of a relation is the score the reply gives, between 1 and 10 ──

D = prompts.TUPLE_DELIMITER
META = '{"polarity":"affirmed","target_polarity":"affirmed","modality":"actual","conditions":{}}'


def _relation(*fields: str) -> dict:
    """Parse one relationship record made of the given fields (after the record kind) and return the relation."""
    text = '("relationship"' + "".join(D + f for f in fields) + ")" + prompts.RECORD_DELIMITER + prompts.COMPLETION_DELIMITER
    _entities, relations, stats = parse_records(text)
    assert stats["malformed"] == 0 and len(relations) == 1, (stats, relations)
    return relations[0]


@pytest.mark.parametrize("period", ["2021..present", "2021年至今", "from 2021 onwards"])
def test_an_unreadable_period_does_not_move_the_strength(period):
    r = _relation("Alice", "Acme", "works_at", "Alice works at Acme", "9", period, META)
    assert (r["strength"], r["description"], r["predicate"]) == (9.0, "Alice works at Acme", "works_at")
    assert (r["valid_from"], r["valid_until"]) == ("", "")          # the period itself stays unread
    assert r["polarity"] == "affirmed"


def test_a_period_glued_to_the_metadata_does_not_move_the_strength():
    # the reply wrote the delimiter before the metadata as "|>": period and metadata arrive as one field
    r = _relation("Alice", "Acme", "works_at", "Alice works at Acme", "8", "2023|>" + META)
    assert (r["strength"], r["description"]) == (8.0, "Alice works at Acme")
    assert (r["valid_from"], r["valid_until"]) == ("", "")


def test_readable_periods_and_older_record_formats_parse_as_before():
    r = _relation("Alice", "Acme", "works_at", "joined in 2021", "8", "2021-03..", META)
    assert (r["strength"], r["description"], r["valid_from"], r["valid_until"]) == (8.0, "joined in 2021", "2021-03", "")
    r = _relation("Alice", "Acme", "works_at", "no time given", "3", "", META)
    assert (r["strength"], r["description"], r["valid_from"]) == (3.0, "no time given", "")
    r = _relation("Alice", "Acme", "works_at", "a bare year is a period", "6", "2021")
    assert (r["strength"], r["valid_from"]) == (6.0, "2021")
    r = _relation("Alice", "Acme", "works_at", "old style", "2")                          # no period field
    assert (r["strength"], r["description"], r["predicate"]) == (2.0, "old style", "works_at")
    r = _relation("Alice", "Acme", "works_at", "ratio of", "42", "7")                     # old style, delimiter inside the description
    assert (r["strength"], r["description"]) == (7.0, "ratio of 42")
    r = _relation("Alice", "Acme", "oldest style", "4")                                   # no predicate field
    assert (r["strength"], r["description"], r["predicate"]) == (4.0, "oldest style", prompts.DEFAULT_PREDICATE)


@pytest.mark.parametrize("written, expected", [("15", 10.0), ("2023", 10.0), ("0", 1.0), ("-3", 1.0), ("0.5", 1.0), ("7.5", 7.5),
                                               ("1", 1.0), ("10", 10.0), ("strong", 1.0), ("8/10", 8.0)])
def test_strength_stays_within_the_range_the_prompt_asks_for(written, expected):
    assert "from 1 to 10" in prompts.GRAPH_EXTRACTION_PROMPT
    assert _relation("Alice", "Acme", "works_at", "old style", written)["strength"] == expected


# ── C11: table files on the rule route ───────────────────────────────────

TABLE_SCHEMA = ExtractionSchema(entity_types=("product",), parent_types={"product": "entity"})
PRODUCT_ROWS = '{"row_type": "product", "key_column": "", "name_column": "Name", "references": [], "date_column": ""}'


def _table(text: str, profile: str = PRODUCT_ROWS, *, seen: list | None = None):
    """One table unit and its extractor, profiled with the given model answer."""
    unit = _unit("u1", doc="kb:t")
    unit.rel_path, unit.text, unit.block_types = "sales/prices.xlsx", text, ["table"]
    tx = TableExtractor([unit], schema=TABLE_SCHEMA)

    def answer(messages):
        if seen is not None:
            seen.append(messages[-1]["content"])
        return profile

    tx.profile(_client(answer))
    return tx, unit


def _facts(tx, unit) -> set[tuple[str, str, str]]:
    return {(f["subject"], f["property"], f["value"]) for f in tx.facts(unit)}


def test_columns_with_the_same_name_keep_their_own_values():
    prompts_seen: list[str] = []
    tx, unit = _table("SHEET: Prices\nROWS: 2-4\nHEADER: Name | Price | Price | Note\n"
                      "Widget A | 100 | 80 | list / net\nWidget B | 200 | 150\nWidget C | 300 | 250", seen=prompts_seen)
    assert "Name | Price | Price (2) | Note" in prompts_seen[0]          # the model is shown, and can name, both columns
    assert tx.wants(unit)
    facts = _facts(tx, unit)
    assert {("Widget A", "Price", "100"), ("Widget A", "Price (2)", "80"), ("Widget A", "Note", "list / net"),
            ("Widget B", "Price", "200"), ("Widget B", "Price (2)", "150"), ("Widget C", "Price (2)", "250")} <= facts
    (entity,) = [e for e in tx.extract(unit).entities if e["name"] == "Widget A"]
    assert "Price: 100; Price (2): 80" in entity["descriptions"][0]


def test_a_row_written_column_by_column_keeps_same_named_columns_apart():
    (section,) = parse_table_text("SHEET: Prices\nROWS: 2-2\nName: Widget A | Price: 100 | Price: 80 | Note: list / net")
    assert section["rows"][0]["labelled"] == {"Name": "Widget A", "Price": "100", "Price (2)": "80", "Note": "list / net"}


def test_distinct_column_names_keep_position_and_avoid_existing_names():
    from kb_pipeline.graph.tables import _distinct

    assert _distinct(["Name", "Price", "Note"]) == ["Name", "Price", "Note"]
    assert _distinct(["A", "A", "", "", "B", "A"]) == ["A", "A (2)", "", "", "B", "A (3)"]
    assert _distinct(["A", "A (2)", "A"]) == ["A", "A (2)", "A (3)"]


def test_a_table_without_repeated_names_reads_as_before():
    tx, unit = _table("SHEET: Prices\nROWS: 2-4\nHEADER: Name | Price | Note\nWidget A | 100 | x\nWidget B | 200\nWidget C | 300 | z")
    assert _facts(tx, unit) == {("Widget A", "Price", "100"), ("Widget A", "Note", "x"), ("Widget B", "Price", "200"),
                                ("Widget C", "Price", "300"), ("Widget C", "Note", "z")}


def test_an_empty_first_header_cell_does_not_end_the_table():
    # the table starts in column B: the first cell is empty in the header and in every row
    tx, unit = _table("SHEET: List\nROWS: 2-6\nHEADER: | Name | Price\n" + "\n".join(f"| Widget {c} | {i}00" for i, c in enumerate("ABCDE", 1)))
    rows, total, unmapped = tx._rows(unit)
    assert (len(rows), total, unmapped) == (5, 5, 0)
    assert tx.wants(unit)
    assert [e["name"] for e in tx.extract(unit).entities] == [f"Widget {c}" for c in "ABCDE"]
    # an unlabelled running number in column A, blank in one row: the rows after the blank one are still read
    lines = [f"{i} | Widget {c} | {i}00" for i, c in enumerate("ABCDEFGHIJ", 1)]
    lines[6] = "| Widget G | 700"
    tx, unit = _table("SHEET: List\nROWS: 2-11\nHEADER: | Name | Price\n" + "\n".join(lines))
    assert [e["name"] for e in tx.extract(unit).entities] == [f"Widget {c}" for c in "ABCDEFGHIJ"]
    assert ("Widget J", "Price", "1000") in _facts(tx, unit)


def test_a_repeated_header_row_still_ends_the_table():
    tx, unit = _table("SHEET: List\nROWS: 2-8\nHEADER: Name | Price\nWidget A | 100\nWidget B | 200\nWidget C | 300\nWidget D | 400\n"
                      "Name | Stock\nWidget E | 5\nWidget F | 6")
    rows, total, unmapped = tx._rows(unit)
    assert (len(rows), total, unmapped) == (4, 7, 3)                  # the second header and the rows under it are not this table's
    assert [e["name"] for e in tx.extract(unit).entities] == ["Widget A", "Widget B", "Widget C", "Widget D"]


def test_a_first_column_called_title_is_data_outside_the_summary_block():
    # a row written column by column whose first column happens to be called Title (or Columns)
    for first in ("Title", "Columns", "Name"):
        (section,) = parse_table_text(f"SHEET: Books\nROWS: 7-7\n{first}: alpha | Author: Bo | Year: 2020")
        assert section["rows"] == [{"labelled": {first: "alpha", "Author": "Bo", "Year": "2020"}, "row_number": 7}]
    book = '{"row_type": "product", "key_column": "", "name_column": "Title", "references": [], "date_column": ""}'
    tx, unit = _table("SHEET: Books\nROWS: 2-4\nTitle: Short Stories | Author: Ann Lee | Year: 2019\n"
                      "Title: Field Notes | Author: Cy Dow | Year: 2020\nTitle: Last One | Author: Di Eng | Year: 2021", book)
    assert tx.wants(unit)
    assert [e["name"] for e in tx.extract(unit).entities] == ["Short Stories", "Field Notes", "Last One"]
    assert ("Field Notes", "Author", "Cy Dow") in _facts(tx, unit)


def test_summary_lines_are_still_not_rows():
    rows = "SHEET: Books\nROWS: 2-3\nHEADER: Title | Author\nShort Stories | Ann Lee\nField Notes | Cy Dow"
    summary = "Sheet: Books\nRows: 2\nTitle: Catalogue of the reading room, 2020 edition\nColumns: Title | Author\n"
    (section,) = parse_table_text(summary + rows)                          # a workbook sheet
    assert [r["cells"] for r in section["rows"]] == [["Short Stories", "Ann Lee"], ["Field Notes", "Cy Dow"]]
    # the summary of a second sheet follows the rows of the first one
    sections = parse_table_text(rows + "\n" + summary.replace("Books", "More") + rows.replace("Books", "More"))
    assert [(s["sheet"], len(s["rows"])) for s in sections] == [("Books", 2), ("More", 2)]
    # a csv file has no sheet name: its summary starts with the file type and "Rows:"
    (section,) = parse_table_text("CSV\nRows: 2\nTitle: exported on request\nColumns: Title | Author\n"
                                  "ROWS: 2-3\nHEADER: Title | Author\nShort Stories | Ann Lee\nField Notes | Cy Dow")
    assert len(section["rows"]) == 2


# ── C12: at recall a fact is dated by its effective date, else its observation date, else the document's ──

def _reading(gr_id: str, **fields) -> dict:
    return {"gr_id": gr_id, "subject": "Patient Li", "subject_id": "e1", "property": "Systolic pressure", "value": "120",
            "unit": "mmHg", "conditions": {}, "point_ids": [f"chunk-{gr_id}"], **fields}


def test_the_same_reading_on_two_observation_dates_stays_two_facts():
    rows = [_reading("f1", observed_at="2024-03-10"), _reading("f2", observed_at="2025-03-12")]
    assert [r["gr_id"] for r in collapse_duplicate_facts(rows)] == ["f1", "f2"]
    # both readings listed in one report: the document date is the same, the observation dates are not
    rows = [_reading("f1", observed_at="2024-03-10", axis="2025-03-20"), _reading("f2", observed_at="2025-03-12", axis="2025-03-20")]
    assert [r["gr_id"] for r in collapse_duplicate_facts(rows)] == ["f1", "f2"]


def test_one_observation_quoted_twice_is_still_one_fact():
    rows = [_reading("f1", observed_at="2024-03-10", axis="2024-03-20"), _reading("f2", observed_at="2024-03-10", axis="2025-01-05")]
    (kept,) = collapse_duplicate_facts(rows)                          # two reports quote the reading of the same day
    assert (kept["gr_id"], kept["point_ids"], kept["_duplicates"]) == ("f1", ["chunk-f1", "chunk-f2"], 1)


def test_facts_dated_by_effective_date_or_document_collapse_as_before():
    same = [_reading("f1", valid_from="2024-03-10"), _reading("f2", valid_from="2024-03-10")]
    assert [r["gr_id"] for r in collapse_duplicate_facts(same)] == ["f1"]
    apart = [_reading("f1", valid_from="2024-03-10"), _reading("f2", valid_from="2025-03-12")]
    assert len(collapse_duplicate_facts(apart)) == 2
    assert len(collapse_duplicate_facts([_reading("f1", axis="2024-03-20"), _reading("f2", axis="2024-03-20")])) == 1
    assert len(collapse_duplicate_facts([_reading("f1", axis="2024-03-20"), _reading("f2", axis="2025-01-05")])) == 2
    assert len(collapse_duplicate_facts([_reading("f1"), _reading("f2")])) == 1
    # the effective date comes first: the same period observed on two days is one statement
    stated = [_reading("f1", valid_from="2024-01", observed_at="2024-03-10"), _reading("f2", valid_from="2024-01", observed_at="2025-03-12")]
    assert len(collapse_duplicate_facts(stated)) == 1


def _window_factors(dates: dict[str, dict]) -> dict[str, float]:
    """Score of each fact after spec_seeds, divided by its similarity: 1.1 inside the question's window, 0.8 outside."""
    class Store:
        def query_points(self, collection_name, query, limit, with_payload, query_filter=None, using=None):
            rows = dates if using == "text" else {}
            return SimpleNamespace(points=[SimpleNamespace(id=k, score=0.5, payload={"gr_id": k, "subject": "Patient Li", "property": k, **v})
                                           for k, v in rows.items()])

        def scroll(self, collection_name, scroll_filter, limit, with_payload, with_vectors):
            return [], None

    seeds = spec_seeds(Store(), "graph_x_spec", [0.1], "systolic pressure of patient li from 2024 to 2025",
                       window={"from": "2024", "to": "2025"})
    return {r["gr_id"]: round(r["_score"] / 0.5, 2) for r in seeds}


def test_the_time_window_weighs_a_fact_by_its_observation_date():
    got = _window_factors({
        "observed inside": {"observed_at": "2024-06-01"},
        "observed outside": {"observed_at": "2019-05-01"},
        "observed outside, report inside": {"observed_at": "2019-05-01", "axis": "2024-08-01"},
        "observed inside, report outside": {"observed_at": "2024-06-01", "axis": "2027-01-15"},
    })
    assert got == {"observed inside": 1.1, "observed outside": 0.8, "observed outside, report inside": 0.8,
                   "observed inside, report outside": 1.1}


def test_the_time_window_reads_effective_and_document_dates_as_before():
    got = _window_factors({
        "effective inside": {"valid_from": "2024-06-01"},
        "effective outside": {"valid_from": "2019-05-01"},
        "effective inside, observed outside": {"valid_from": "2024-06-01", "observed_at": "2019-05-01"},
        "report inside": {"axis": "2025-02-01"},
        "report outside": {"axis": "2019-02-01"},
        "report with a version, no date": {"axis": "rev *L"},
        "undated": {},
    })
    assert got == {"effective inside": 1.1, "effective outside": 0.8, "effective inside, observed outside": 1.1, "report inside": 1.1,
                   "report outside": 0.8, "report with a version, no date": 1.1, "undated": 1.1}
