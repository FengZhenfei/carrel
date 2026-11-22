"""What a workbook hides is indexed like what it shows: hidden sheets, hidden rows, hidden columns, pictures on or
in them, and pictures set to "not shown". (A hidden slide of a presentation always was.) Hidden sheets and rows used
to be left out, hidden columns for a while as well.

A very hidden sheet is the exception. That state cannot be chosen from the menus; programs set it for records of
their own, and WPS keeps a copy of every picture placed in a cell on such a sheet. It is not read.

Reading hidden rows and columns does not change which rows are the header: the header of a sheet is looked for on the
sheet as it shows, so it stays the row that was the header before."""
import re
import zipfile
from pathlib import Path
from unittest import mock

import pytest
from openpyxl import Workbook
from openpyxl.drawing.image import Image as SheetImage
from PIL import Image

from kb_pipeline.parsers import native_table
from kb_pipeline.parsers.xlsx_images import extract_xlsx_images


def rewrite(path, change):
    with zipfile.ZipFile(path) as z:
        parts = {name: z.read(name) for name in z.namelist()}
    change(parts)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in parts.items():
            z.writestr(name, data)


def workbook(tmp_path, state="hidden"):
    wb = Workbook()
    ws = wb.active
    ws.title = "Prices"
    for row in (["name", "cost", "price"], ["bolt", 1.5, 2.1], ["washer", 0.1, 0.2], ["nut", 0.5, 0.75]):
        ws.append(row)
    ws.row_dimensions[3].hidden = True
    ws.column_dimensions["B"].hidden = True
    other = wb.create_sheet("Suppliers")
    other.append(["supplier", "city"])
    other.append(["ACME", "Lyon"])
    other.sheet_state = state
    path = tmp_path / "prices.xlsx"
    wb.save(path)
    return path


EVERYTHING = [("Prices", [(1, ["name", "cost", "price"]), (2, ["bolt", "1.5", "2.1"]), (3, ["washer", "0.1", "0.2"]),
                          (4, ["nut", "0.5", "0.75"])]),
              ("Suppliers", [(1, ["supplier", "city"]), (2, ["ACME", "Lyon"])])]


def test_hidden_sheets_rows_and_columns_are_read(tmp_path):
    path = workbook(tmp_path)
    assert [(name, rows) for name, rows, _ in native_table.read_xlsx_rows(path)] == EVERYTHING
    with mock.patch.object(native_table, "load_workbook_compat", side_effect=ValueError("bad backend")):
        assert [(name, rows) for name, rows, _ in native_table.read_xlsx_rows(path)] == EVERYTHING     # the fallback reader too


def test_a_very_hidden_sheet_is_not_read(tmp_path):
    path = workbook(tmp_path, "veryHidden")
    assert [(name, rows) for name, rows, _ in native_table.read_xlsx_rows(path)] == EVERYTHING[:1]       # hidden row and column of the first sheet: read
    with mock.patch.object(native_table, "load_workbook_compat", side_effect=ValueError("bad backend")):
        assert [(name, rows) for name, rows, _ in native_table.read_xlsx_rows(path)] == EVERYTHING[:1]
    text = "\n".join(b.text for b in native_table.parse_native_table(path, max_tokens=400, overlap_tokens=80))
    assert "washer | 0.1 | 0.2" in text and "Suppliers" not in text and "ACME" not in text


def test_an_old_excel_workbook_reads_its_hidden_sheet_and_not_its_very_hidden_one(tmp_path):
    import xlrd
    from types import SimpleNamespace

    def sheet(name, visibility):
        grid = [["name"], [name.lower()]]
        return SimpleNamespace(name=name, visibility=visibility, nrows=2, ncols=1, merged_cells=[], rowinfo_map={}, colinfo_map={},
                               cell_value=lambda row, col: grid[row][col], cell_type=lambda row, col: xlrd.XL_CELL_TEXT,
                               cell_xf_index=lambda row, col: 99)

    book = SimpleNamespace(datemode=0, xf_list=[], format_map={}, sheets=lambda: [sheet("Shown", 0), sheet("Hidden", 1), sheet("Store", 2)])
    with mock.patch("xlrd.open_workbook", return_value=book):
        sheets = native_table.read_xls_rows(tmp_path / "any.xls")
    assert [(name, rows) for name, rows, _ in sheets] == [("Shown", [(1, ["name"]), (2, ["shown"])]), ("Hidden", [(1, ["name"]), (2, ["hidden"])])]


def test_the_chunks_of_a_workbook_hold_its_hidden_content(tmp_path):
    text = "\n".join(b.text for b in native_table.parse_native_table(workbook(tmp_path), max_tokens=400, overlap_tokens=80))
    for line in ("bolt | 1.5 | 2.1", "washer | 0.1 | 0.2", "SHEET: Suppliers", "ACME | Lyon"):
        assert line in text


def test_a_workbook_that_hides_nothing_still_reads_the_same(tmp_path):
    wb = Workbook()
    ws = wb.active
    ws.title = "Prices"
    for row in (["name", "cost", "price"], ["bolt", 1.5, 2.1]):
        ws.append(row)
    path = tmp_path / "plain.xlsx"
    wb.save(path)
    assert [(name, rows) for name, rows, _ in native_table.read_xlsx_rows(path)] == [("Prices", [(1, ["name", "cost", "price"]), (2, ["bolt", "1.5", "2.1"])])]


def picture_book(tmp_path):
    picture = tmp_path / "diagram.png"
    Image.new("RGB", (120, 60), "blue").save(picture)
    wb = Workbook()
    ws = wb.active
    ws.title = "Measurements"
    ws.append(["Item", "Value"])
    ws.append(["Gate", 17])
    ws.append(["Valve", 4])
    ws.add_image(SheetImage(picture), "C3")
    path = tmp_path / "sample.xlsx"
    wb.save(path)
    return path


def test_a_floating_picture_set_to_not_shown_is_extracted(tmp_path):
    path = picture_book(tmp_path)

    def hide(parts):
        drawing, n = re.subn(r"<((?:xdr:)?cNvPr) ", r'<\1 hidden="1" ', parts["xl/drawings/drawing1.xml"].decode(), count=1)
        assert n == 1
        parts["xl/drawings/drawing1.xml"] = drawing.encode()

    rewrite(path, hide)
    blocks = extract_xlsx_images(path, tmp_path / "cache")
    assert [(b.sheet_name, b.metadata["source_locator"]["range"]) for b in blocks] == [("Measurements", "C3")]
    assert Path(blocks[0].visual_ref).read_bytes() == (tmp_path / "diagram.png").read_bytes()


def hidden_header_book(tmp_path):
    """A sheet as accounting software exports it: row 1, hidden, holds the field codes; row 2 is the header in view.
    Row 4 is a hidden data row."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Items"
    for row in (["FNumber", "FName", "FSpec", "FPrice"], ["Code", "Name", "Size", "Price"], ["M001", "bolt", "M8", 1.5],
                ["M000", "retired", "M6", 0.9], ["M002", "washer", "8mm", 0.2]):
        ws.append(row)
    ws.row_dimensions[1].hidden = True
    ws.row_dimensions[4].hidden = True
    path = tmp_path / "items.xlsx"
    wb.save(path)
    return path


@pytest.mark.parametrize("fallback", [False, True])
def test_a_hidden_row_is_read_but_does_not_become_the_header(tmp_path, fallback):
    path = hidden_header_book(tmp_path)
    with mock.patch.object(native_table, "load_workbook_compat", side_effect=ValueError("bad backend")) if fallback else mock.MagicMock():
        (name, rows, _), = native_table.read_xlsx_rows(path)
        blocks = native_table.parse_native_table(path, max_tokens=400, overlap_tokens=80)
    assert [number for number, _ in rows] == [1, 2, 3, 4, 5] and rows.hidden == {1, 4}
    summary, table = [b.text for b in blocks]
    assert summary == "Sheet: Items\nRows: 5\nTitle: FNumber FName FSpec FPrice\nColumns: Code | Name | Size | Price"
    assert table == ("SHEET: Items\nROWS: 3-5\nHEADER: Code | Name | Size | Price\n"
                     "M001 | bolt | M8 | 1.5\nM000 | retired | M6 | 0.9\nM002 | washer | 8mm | 0.2")


def test_a_hidden_row_of_an_old_excel_sheet_does_not_become_the_header_either(tmp_path):
    import xlrd
    from types import SimpleNamespace

    grid = [["FNumber", "FName"], ["Code", "Name"], ["M001", "bolt"]]
    sheet = SimpleNamespace(name="Items", visibility=0, nrows=3, ncols=2, merged_cells=[], colinfo_map={},
                            rowinfo_map={0: SimpleNamespace(hidden=1)}, cell_value=lambda row, col: grid[row][col],
                            cell_type=lambda row, col: xlrd.XL_CELL_TEXT, cell_xf_index=lambda row, col: 99)
    book = SimpleNamespace(datemode=0, xf_list=[], format_map={}, sheets=lambda: [sheet])
    with mock.patch("xlrd.open_workbook", return_value=book), mock.patch.object(native_table, "require_source_content"):
        blocks = native_table.parse_native_table(tmp_path / "any.xls", max_tokens=400, overlap_tokens=80)
    assert [b.text for b in blocks] == ["Sheet: Items\nRows: 3\nTitle: FNumber FName\nColumns: Code | Name",
                                        "SHEET: Items\nROWS: 3-3\nHEADER: Code | Name\nM001 | bolt"]


def sheet_texts(tmp_path, rows, hidden_columns="", merges=()):
    wb = Workbook()
    ws = wb.active
    ws.title = "S"
    for row in rows:
        ws.append(row)
    for merged in merges:
        ws.merge_cells(merged)
    for column in hidden_columns:
        ws.column_dimensions[column].hidden = True
    path = tmp_path / "sheet.xlsx"
    wb.save(path)
    with mock.patch.object(native_table, "load_workbook_compat", side_effect=ValueError("bad backend")):
        fallback = [b.text for b in native_table.parse_native_table(path, max_tokens=400, overlap_tokens=80)]
    texts = [b.text for b in native_table.parse_native_table(path, max_tokens=400, overlap_tokens=80)]
    assert fallback == texts                                   # both readers
    return texts


def test_the_cells_of_a_hidden_column_do_not_change_which_rows_are_the_header(tmp_path):
    """The header is looked for on the sheet as it shows. Read into the row, the cells of a hidden column would
    make a title row look like a header (a counter beside it), a header look like a note (a link in it), or a second
    header row look like data (a number in it)."""
    counter = sheet_texts(tmp_path, [[1, "2024 price list"], [2, "Name", "Size", "Price"], [3, "bolt", "M8", 1.5], [4, "washer", "8mm", 0.2]], "A")
    assert counter == ["Sheet: S\nRows: 4\nTitle: 1 2024 price list\nColumns: 2 | Name | Size | Price",
                       "SHEET: S\nROWS: 3-4\nHEADER: 2 | Name | Size | Price\n3 | bolt | M8 | 1.5\n4 | washer | 8mm | 0.2"]
    link = sheet_texts(tmp_path, [["Name", "http://example.com/source-of-the-list", "Price"], ["bolt", "x", 0.35], ["washer", "y", 0.2]], "B")
    assert link[1].startswith("SHEET: S\nROWS: 2-3\nHEADER: Name | http://example.com/source-of-the-list | Price\nbolt | x | 0.35")
    # the second header row is known by the cell it repeats ("Name" under "Name") and by holding no data-shaped cell
    tiers = sheet_texts(tmp_path, [["Name", "Stock", "Stock", "No"], ["Name", "Warehouse A", "Warehouse B", 7], ["bolt", 10, 20, 1], ["nut", 5, 6, 2]], "D")
    assert tiers[1] == "SHEET: S\nROWS: 3-4\nHEADER: Name | Stock/Warehouse A | Stock/Warehouse B | No/7\nbolt | 10 | 20 | 1\nnut | 5 | 6 | 2"


def test_a_merged_cell_that_starts_in_a_hidden_column_is_read_once_where_the_sheet_shows_it(tmp_path):
    """ "Notes" is merged over B1:C1 and column B is hidden: the sheet shows the label above column C, and that is
    where it is read, as it was while hidden columns were left out. The column in view keeps its name, and nothing is
    written twice. The same holds in a data row and down the rows of a taller range."""
    rows = [["Item", "Notes", None], ["Item", "internal", "public"], ["bolt", "x1", "y1"], ["nut", "see the remark", None],
            ["screw", "both rows", None], ["pin", None, None]]
    texts = sheet_texts(tmp_path, rows, "B", merges=["B1:C1", "A1:A2", "B4:C4", "B5:C6"])
    assert texts[1] == ("SHEET: S\nROWS: 3-6\nHEADER: Item | internal | Notes/public\nbolt | x1 | y1\n"
                        "nut |  | see the remark\nscrew |  | both rows\npin |  | both rows")
    # a title merged across a sheet whose first column is hidden is written once
    title = sheet_texts(tmp_path, [["Stock 2026", None, None], ["Code", "Name", "Count"], ["M001", "bolt", 40]], "A", merges=["A1:C1"])
    assert title[0] == "Sheet: S\nRows: 3\nTitle: Stock 2026\nColumns: Code | Name | Count"
    # a range that lies wholly in hidden columns keeps its label in its first column
    inside = sheet_texts(tmp_path, [["Name", "Group", None], ["bolt", "g1", None], ["nut", "g2", "x"]], "BC", merges=["B2:C2"])
    assert inside[1] == "SHEET: S\nROWS: 2-3\nHEADER: Name | Group\nbolt | g1\nnut | g2 | x"
    # the label of a range that runs past the last column with a value shows there, beyond the end of every row
    # (the reader of the workbook's XML never showed a label beyond its last column: there it stays in its own cell)
    wb = Workbook()
    ws = wb.active
    ws.title = "S"
    for row in (["Item", "Notes"], ["Item", "internal"], ["bolt", "x1"]):
        ws.append(row)
    ws.merge_cells("B1:D1")
    ws.merge_cells("A1:A2")
    ws.column_dimensions["B"].hidden = True
    path = tmp_path / "wide.xlsx"
    wb.save(path)
    (_, rows, _), = native_table.read_xlsx_rows(path)
    assert rows == [(1, ["Item", "", "Notes"]), (2, ["Item", "internal", ""]), (3, ["bolt", "x1", ""])]
    assert rows.in_view() == [(1, ["Item", "", "Notes"]), (2, ["Item", "", ""]), (3, ["bolt", "", ""])]
    with mock.patch.object(native_table, "load_workbook_compat", side_effect=ValueError("bad backend")):
        (_, fallback, _), = native_table.read_xlsx_rows(path)
    assert fallback == [(1, ["Item", "Notes"]), (2, ["Item", "internal"]), (3, ["bolt", "x1"])]
    assert fallback.in_view() == [(1, ["Item", ""]), (2, ["Item", ""]), (3, ["bolt", ""])]


def test_a_column_in_view_keeps_its_name_when_a_hidden_column_has_the_same(tmp_path):
    """Same-named columns are numbered from the second one on. A hidden column to the left of a column in view with
    the same name would take the plain name and push "(2)" onto the one in view: hidden columns are numbered last."""
    rows = [["Code", "Price", "Name", "Price", "Price"], ["M001", 1.1, "bolt", 2.5, 3], ["M002", 0.1, "nut", 0.4, 1]]
    texts = sheet_texts(tmp_path, rows, "B")
    assert texts[1].splitlines()[2] == "HEADER: Code | Price (3) | Name | Price | Price (2)"
    assert native_table.distinct_names(["a", "b", "a", "a"]) == ["a", "b", "a (2)", "a (3)"]                # nothing hidden: as before
    assert native_table.distinct_names(["a", "b", "a", "a"], later=[0]) == ["a (3)", "b", "a", "a (2)"]


def test_an_old_excel_sheet_hands_over_its_hidden_columns_and_the_labels_they_start(tmp_path):
    sheet = mock.MagicMock()
    sheet.name, sheet.visibility, sheet.nrows, sheet.ncols = "S", 0, 3, 3
    cells = [["Item", "Notes", ""], ["Item", "internal", "public"], ["bolt", "x1", "y1"]]
    sheet.cell_type.side_effect = lambda r, c: 1 if cells[r][c] else 0
    sheet.cell_value.side_effect = lambda r, c: cells[r][c]
    sheet.merged_cells = [(0, 1, 1, 3), (0, 2, 0, 1)]                 # B1:C1 and A1:A2, ends exclusive
    sheet.rowinfo_map = {}
    sheet.colinfo_map = {1: mock.MagicMock(hidden=1)}
    book = mock.MagicMock(datemode=0)
    book.sheets.return_value = [sheet]
    xlrd = mock.MagicMock()
    xlrd.open_workbook.return_value = book
    with mock.patch.dict("sys.modules", {"xlrd": xlrd}):
        (_, rows, spans), = native_table.read_xls_rows(tmp_path / "old.xls")
    assert rows == [(1, ["Item", "", "Notes"]), (2, ["Item", "internal", "public"]), (3, ["bolt", "x1", "y1"])]
    assert rows.hidden_columns == {2}
    assert rows.in_view() == [(1, ["Item", "", "Notes"]), (2, ["Item", "", "public"]), (3, ["bolt", "", "y1"])]
    text = [b.text for b in native_table.chunk_rows_to_blocks("xls", rows, "S", 400, 80, header_spans=spans)]
    assert text[1] == "SHEET: S\nROWS: 3-3\nHEADER: Item | internal | Notes/public\nbolt | x1 | y1"
    # a taller range: the value runs down its rows in the column that shows it; and a value whose shown column lies
    # beyond the sheet stays in its own cell (the sheet has three columns, the range reaches a fourth)
    cells[:] = [["Item", "", ""], ["Group", "both rows", ""], ["bolt", "", ""], ["nut", "kept", "z"]]
    sheet.nrows = 4
    sheet.merged_cells = [(1, 3, 1, 3), (3, 4, 1, 2)]                # B2:C3; B4 alone in its hidden column
    with mock.patch.dict("sys.modules", {"xlrd": xlrd}):
        (_, rows, _), = native_table.read_xls_rows(tmp_path / "old.xls")
    assert rows == [(1, ["Item", "", ""]), (2, ["Group", "", "both rows"]), (3, ["bolt", "", "both rows"]), (4, ["nut", "kept", "z"])]
    cells[:] = [["Item", "", "wide"], ["bolt", "", "y"]]
    cells[0][2] = "wide"
    sheet.nrows = 2
    sheet.colinfo_map = {2: mock.MagicMock(hidden=1)}
    sheet.merged_cells = [(0, 1, 2, 4)]                               # C1:D1, column C hidden, column D not in the sheet
    with mock.patch.dict("sys.modules", {"xlrd": xlrd}):
        (_, rows, _), = native_table.read_xls_rows(tmp_path / "old.xls")
    assert rows == [(1, ["Item", "", "wide"]), (2, ["bolt", "", "y"])]


def test_the_rows_in_view_are_what_was_read_while_hidden_content_was_left_out(tmp_path):
    wb = Workbook()
    ws = wb.active
    ws.title = "S"
    for row in (["Region", None, None, "Total"], ["x", "North", "South", "t"], [1, 2, 3, 6], ["only", None, None, None], [7, 8, 9, 24]):
        ws.append(row)
    ws.merge_cells("A1:C1")                       # the label of this range shows in column B once column A is hidden,
                                                  # and that is where it is read
    ws.column_dimensions["A"].hidden = True
    ws.row_dimensions[3].hidden = True
    path = tmp_path / "view.xlsx"
    wb.save(path)
    (_, rows, _), = native_table.read_xlsx_rows(path)
    assert rows == [(1, ["", "Region", "", "Total"]), (2, ["x", "North", "South", "t"]), (3, ["1", "2", "3", "6"]),
                    (4, ["only", "", "", ""]), (5, ["7", "8", "9", "24"])]
    assert (rows.hidden, rows.hidden_columns) == ({3}, {1})
    assert rows.in_view() == [(1, ["", "Region", "", "Total"]), (2, ["", "North", "South", "t"]), (5, ["", "8", "9", "24"])]
    with mock.patch.object(native_table, "load_workbook_compat", side_effect=ValueError("bad backend")):
        (_, fallback, _), = native_table.read_xlsx_rows(path)
    assert fallback == rows and fallback.in_view() == rows.in_view()


def test_a_sheet_whose_rows_are_all_hidden_is_still_read(tmp_path):
    wb = Workbook()
    ws = wb.active
    ws.title = "Old"
    ws.append(["name", "price"])
    ws.append(["bolt", 1.5])
    ws.row_dimensions[1].hidden = True
    ws.row_dimensions[2].hidden = True
    path = tmp_path / "old.xlsx"
    wb.save(path)
    text = "\n".join(b.text for b in native_table.parse_native_table(path, max_tokens=400, overlap_tokens=80))
    assert "HEADER: name | price" in text and "bolt | 1.5" in text


@pytest.mark.parametrize("hidden", [range(4, 13), range(4, 8), range(8, 13)])
def test_empty_hidden_rows_do_not_end_the_sheet_early(tmp_path, hidden):
    # a long run of empty rows ends a sheet (its used range can reach row 1048576 by formatting alone). Hidden rows
    # were never part of that run while they were left out; counted now, they would cut off what stands after them
    wb = Workbook()
    ws = wb.active
    for row in (["name", "value"], ["bolt", 1], ["nut", 2]):
        ws.append(row)
    ws.cell(13, 1, "total")
    ws.cell(13, 2, 3)
    for number in hidden:
        ws.row_dimensions[number].hidden = True
    path = tmp_path / "gap.xlsx"
    wb.save(path)
    with mock.patch.object(native_table, "_XLSX_EMPTY_RUN_LIMIT", 6):
        (_, rows, _), = native_table.read_xlsx_rows(path)
    assert [number for number, _ in rows] == [1, 2, 3, 13]
    # and nine empty rows in view still end it
    for number in hidden:
        ws.row_dimensions[number].hidden = False
    wb.save(path)
    with mock.patch.object(native_table, "_XLSX_EMPTY_RUN_LIMIT", 6):
        (_, rows, _), = native_table.read_xlsx_rows(path)
    assert [number for number, _ in rows] == [1, 2, 3]


def wps_book(tmp_path, state):
    """What WPS writes for a picture placed in a cell: the cell holds =DISPIMG("…"), xl/cellimages.xml names the image
    file, and a sheet called WpsReserved_CellImgList holds a floating copy of every such picture."""
    picture = tmp_path / "diagram.png"
    Image.new("RGB", (120, 60), "blue").save(picture)
    wb = Workbook()
    ws = wb.active
    ws.title = "Measurements"
    ws.append(["Item", "Value"])
    ws.append(["Gate", 17])
    store = wb.create_sheet("WpsReserved_CellImgList")
    store.append(["kept by the program"])       # WPS leaves the sheet without values; one here shows that its rows are not read either
    store.add_image(SheetImage(picture), "A1")
    store.sheet_state = state
    path = tmp_path / "wps.xlsx"
    wb.save(path)

    def place_in_cell(parts):
        sheet = parts["xl/worksheets/sheet1.xml"].decode()
        sheet, n = re.subn(r'(<row r="2"[^>]*>.*?)(</row>)', r'\1<c r="C2"><f>_xlfn.DISPIMG("ID_A",1)</f><v/></c>\2', sheet, count=1, flags=re.S)
        assert n == 1
        parts["xl/worksheets/sheet1.xml"] = sheet.encode()
        parts["xl/cellimages.xml"] = (
            '<etc:cellImages xmlns:etc="http://www.wps.cn/officeDocument/2017/etCustomData" '
            'xmlns:xdr="http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing" '
            'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><etc:cellImage><xdr:pic><xdr:nvPicPr>'
            '<xdr:cNvPr id="1" name="ID_A"/></xdr:nvPicPr><xdr:blipFill><a:blip r:embed="rImage"/></xdr:blipFill></xdr:pic>'
            '</etc:cellImage></etc:cellImages>').encode()
        parts["xl/_rels/cellimages.xml.rels"] = (
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rImage" '
            'Target="media/image1.png" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"/>'
            '</Relationships>').encode()

    rewrite(path, place_in_cell)
    return path


def test_the_very_hidden_sheet_where_wps_keeps_copies_of_cell_pictures_does_not_give_each_picture_twice(tmp_path):
    blocks = extract_xlsx_images(wps_book(tmp_path, "veryHidden"), tmp_path / "cache")
    assert [(b.sheet_name, b.metadata["source_locator"]["range"]) for b in blocks] == [("Measurements", "C2")]     # once, at its cell
    assert [name for name, *_ in native_table.read_xlsx_rows(tmp_path / "wps.xlsx")] == ["Measurements"]


def test_a_picture_on_a_sheet_hidden_from_the_menu_is_extracted(tmp_path):
    blocks = extract_xlsx_images(wps_book(tmp_path, "hidden"), tmp_path / "cache")
    assert [(b.sheet_name, b.metadata["source_locator"]["range"]) for b in blocks] == [("Measurements", "C2"), ("WpsReserved_CellImgList", "A1")]
    assert [name for name, *_ in native_table.read_xlsx_rows(tmp_path / "wps.xlsx")] == ["Measurements", "WpsReserved_CellImgList"]


@pytest.mark.parametrize("hide", ["row", "column", "sheet"])
def test_a_picture_placed_in_a_cell_of_a_hidden_row_column_or_sheet_is_extracted(tmp_path, hide):
    path = wps_book(tmp_path, "veryHidden")

    def conceal(parts):
        sheet = parts["xl/worksheets/sheet1.xml"].decode()
        if hide == "row":
            sheet, n = re.subn(r'<row r="2"', '<row r="2" hidden="1"', sheet, count=1)
        elif hide == "column":
            sheet, n = re.subn(r"<sheetData>", '<cols><col min="3" max="3" width="9" hidden="1" customWidth="1"/></cols><sheetData>', sheet, count=1)
        else:
            book, n = re.subn(r'(name="Measurements"[^>]*?state=")visible"', r'\1hidden"', parts["xl/workbook.xml"].decode(), count=1)
            parts["xl/workbook.xml"] = book.encode()
        assert n == 1
        parts["xl/worksheets/sheet1.xml"] = sheet.encode()

    rewrite(path, conceal)
    blocks = extract_xlsx_images(path, tmp_path / "cache")
    assert [(b.sheet_name, b.metadata["source_locator"]["range"]) for b in blocks] == [("Measurements", "C2")]
