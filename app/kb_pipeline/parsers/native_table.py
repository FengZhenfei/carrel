from __future__ import annotations

import csv
import datetime
import math
import re
import string
import zipfile
import xml.etree.ElementTree as ET
from decimal import ROUND_HALF_UP, Context, Decimal
from pathlib import Path
from typing import Any, Iterable

from ..chunking.chunker import pack_labelled_cells
from ..models import ParsedBlock
from ..utils import count_tokens
from .common import clean_text
from .openpyxl_compat import load_workbook_compat


def parse_native_table(
    path: Path,
    *,
    max_tokens: int,
    overlap_tokens: int,
) -> list[ParsedBlock]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        rows = read_csv_rows_numbered(path)          # with source-file row numbers: provenance no longer shifts after blank rows are filtered (final review S01)
        return chunk_rows_to_blocks("csv", rows, None, max_tokens, overlap_tokens)
    if suffix == ".xlsx":
        blocks: list[ParsedBlock] = []
        for sheet_name, rows, spans in read_xlsx_rows(path):
            blocks.extend(chunk_rows_to_blocks("xlsx", rows, sheet_name, max_tokens, overlap_tokens, header_spans=spans))
        return blocks
    if suffix == ".xls":
        blocks = []
        for sheet_name, rows, spans in read_xls_rows(path):
            blocks.extend(chunk_rows_to_blocks("xls", rows, sheet_name, max_tokens, overlap_tokens, header_spans=spans))
        return blocks
    raise RuntimeError(f"native table parser does not support {suffix}")


def read_csv_rows_numbered(path: Path) -> list[tuple[int, list[str]]]:
    """(source-file row number, cells). Blank rows are filtered out but row numbers follow the source
    file, so provenance lines up (final review S01)."""
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return _read_csv_rows_with_encoding(path, encoding=encoding, errors="strict")
        except Exception as exc:
            last_error = exc
    try:
        return _read_csv_rows_with_encoding(path, encoding="utf-8", errors="ignore")
    except Exception as exc:
        raise RuntimeError(f"cannot read csv {path}: {last_error or exc!r}") from exc


def _read_csv_rows_with_encoding(path: Path, *, encoding: str, errors: str) -> list[tuple[int, list[str]]]:
    rows: list[tuple[int, list[str]]] = []
    with path.open("r", encoding=encoding, errors=errors, newline="") as handle:
        for row_no, row in enumerate(csv.reader(handle), start=1):
            cleaned = [normalize_cell(cell) for cell in row]
            if any(cleaned):
                rows.append((row_no, cleaned))
    return rows


_XLSX_EMPTY_RUN_LIMIT = 5000


HEADER_SPAN_MAX_ROWS = 3     # a vertical merge starting on a row counts as evidence of a header tier only if it spans at most this many rows (category labels spanning dozens of rows do not)


def vertical_spans(ranges: Iterable[tuple[int, int, int, int]], *, max_rows: int = HEADER_SPAN_MAX_ROWS) -> dict[int, int]:
    """Footprint of vertical merges: {start row: end row}, keeping only those spanning 2..max_rows rows.
    bounds is (min_col, min_row, max_col, max_row). A vertical merge starting on a header row (C1:C2)
    shows the next row is the second tier of the header, even if its cells look like data (product
    names carrying version numbers) -- Codex 2026-09-14 R02: all 336 chunks of a feature comparison
    sheet had headers missing the product column names."""
    spans: dict[int, int] = {}
    for min_col, min_row, max_col, max_row in ranges:
        if 0 < max_row - min_row < max_rows:
            spans[min_row] = max(spans.get(min_row, 0), max_row)
    return spans


def _xlsx_content_extent(ws: Any) -> tuple[int, int]:
    """The range of the sheet that really holds content: (last row, last column). ws.max_row / ws.max_column
    depend on the farthest cell that appears in the XML, even an empty cell that only carries a style: a sheet
    with a whole row formatted has max_column 16384, and in non-read-only mode iter_rows builds that many Cell
    objects for every row, about 4.7 MB per row and tens of GB for a few thousand rows (2026-09-29 audit: a
    157 KB file read with 508 MB). Only cells with a value are counted here; the value of a merged range sits
    in its top-left cell, the rows it is filled down to are added by the caller. When the internal cell table
    is not available (a different openpyxl implementation) the declared range is returned."""
    cells = getattr(ws, "_cells", None)
    if not isinstance(cells, dict):
        return int(ws.max_row or 0), int(ws.max_column or 0)
    last_row = last_col = 0
    for (row_no, col_no), cell in cells.items():
        value = cell.value
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        if row_no > last_row:
            last_row = row_no
        if col_no > last_col:
            last_col = col_no
    return last_row, last_col


def read_xlsx_rows(path: Path) -> list[tuple[str, list[tuple[int, list[str]]], dict[int, int]]]:
    try:
        wb = load_workbook_compat(str(path), data_only=True, read_only=False)
        output: list[tuple[str, list[tuple[int, list[str]]], dict[int, int]]] = []
        for ws in wb.worksheets:
            if str(getattr(ws, "sheet_state", "visible") or "visible") != "visible":
                # Hidden sheets are mostly archived old data / helper calculation sheets: not indexed
                # (health check R8)
                print(f"[XLSX] skip hidden sheet {path.name}:{ws.title}", flush=True)
                continue
            hidden_rows = {int(r) for r, dim in ws.row_dimensions.items() if getattr(dim, "hidden", False)}
            merged_values: dict[tuple[int, int], str] = {}
            # Clamp fills to the used range: an accidental whole-column merge
            # (A1:A1048576 is one Excel/WPS click away) would otherwise build
            # a ~1M-entry dict per merge before a single row is read.
            used_rows = int(ws.max_row or 0)
            content_rows, content_cols = _xlsx_content_extent(ws)
            for merged in ws.merged_cells.ranges:
                min_col, min_row, max_col, max_row = merged.bounds
                anchor = ws.cell(min_row, min_col)
                value = normalize_cell(anchor.value, getattr(anchor, "number_format", None))
                if not value:
                    continue
                # Fill vertically, not horizontally. Every **row** covered by the merged range gets
                # the value, because a category label such as "WPS AI" spanning 54 rows is exactly what
                # gives each row chunk its ownership; but the other columns of the same row are not
                # filled -- that would only copy the same value N times within one row.
                #
                # Measured on the "instructions" sheet of a product feature list workbook: one 7x4
                # merged block poured the same note into 28 cells, rendering as note|note|note|note
                # repeated over 7 rows, which chunked into dozens of verbatim-identical chunks -- the
                # same vector stored dozens of times, surfacing in clusters at retrieval. 9.2% of all
                # table-native-v1 chunks came from this.
                for row_no in range(min_row, min(max_row, used_rows) + 1):
                    merged_values[(row_no, min_col)] = value
            rows: list[tuple[int, list[str]]] = []
            empty_run = 0
            # Read only up to the last row / column with content (rows a merged range is filled down to count
            # as content): no cells are built for formatted but empty areas
            last_row = max([content_rows] + [row_no for row_no, _col in merged_values])
            last_col = max([content_cols] + [col_no for _row, col_no in merged_values])
            for row in (ws.iter_rows(max_row=last_row, max_col=last_col) if last_row > 0 and last_col > 0 else ()):
                if row and row[0].row in hidden_rows:
                    continue
                values: list[str] = []
                for cell in row:
                    values.append(normalize_cell(cell.value, cell.number_format) or merged_values.get((cell.row, cell.column), ""))
                if any(values):
                    empty_run = 0
                    rows.append((row[0].row if row else len(rows) + 1, values))
                    continue
                # ws.max_row is affected by formatting: a sheet with a whole column formatted "uses"
                # row 1048576, with nothing but empty rows after the data. Several thousand consecutive
                # empty rows are taken as the end, so no more idle spinning (health check R8)
                empty_run += 1
                if empty_run >= _XLSX_EMPTY_RUN_LIMIT:
                    break
            if rows:
                output.append((ws.title or "Sheet", rows, vertical_spans(m.bounds for m in ws.merged_cells.ranges)))
        return output
    except Exception as exc:
        print(f"[XLSX WARN] openpyxl failed, fallback XML: {path} -> {exc!r}", flush=True)
        return read_xlsx_rows_via_xml(path)


def read_xlsx_rows_via_xml(path: Path) -> list[tuple[str, list[tuple[int, list[str]]], dict[int, int]]]:
    output: list[tuple[str, list[tuple[int, list[str]]], dict[int, int]]] = []
    with zipfile.ZipFile(str(path), "r") as archive:
        shared = _xlsx_parse_shared_strings(archive)
        for sheet_name, sheet_path in _xlsx_sheet_targets(archive):
            try:
                rows = _xlsx_parse_sheet_rows(archive, sheet_path, shared)
            except Exception as exc:
                print(f"[XLSX XML WARN] parse sheet failed: {path} {sheet_name} {sheet_path} -> {exc!r}", flush=True)
                continue
            if rows:
                output.append((sheet_name or "Sheet", rows, {}))
    return output


def _xlsx_col_to_index(col_letters: str) -> int:
    index = 0
    for char in col_letters.upper():
        if "A" <= char <= "Z":
            index = index * 26 + (ord(char) - ord("A") + 1)
    return index


def _xlsx_parse_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        xml = archive.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(xml)
    ns = _xml_namespace(root)
    values: list[str] = []
    for item in _xml_findall(root, ".//si", ns):
        parts: list[str] = []
        for text_node in _xml_findall(item, ".//t", ns):
            if text_node.text:
                parts.append(text_node.text)
        values.append("".join(parts))
    return values


def _xlsx_sheet_targets(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    output: list[tuple[str, str]] = []
    try:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    except KeyError:
        return _xlsx_guess_sheet_targets(archive)

    workbook_ns = _xml_namespace(workbook)
    rels_ns = _xml_namespace(rels)
    rel_targets: dict[str, str] = {}
    for rel in _xml_findall(rels, ".//Relationship", rels_ns):
        rel_id = rel.attrib.get("Id")
        target = rel.attrib.get("Target")
        if rel_id and target:
            rel_targets[rel_id] = target

    for sheet in _xml_findall(workbook, ".//sheets/sheet", workbook_ns):
        name = sheet.attrib.get("name") or "Sheet"
        rel_id = sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id") or sheet.attrib.get("r:id")
        target = rel_targets.get(rel_id or "")
        if target:
            target = target.lstrip("/")
            sheet_path = target if target.startswith("xl/") else f"xl/{target}"
            output.append((name, sheet_path))
    return output or _xlsx_guess_sheet_targets(archive)


def _xlsx_guess_sheet_targets(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    output: list[tuple[str, str]] = []
    index = 1
    while True:
        sheet_path = f"xl/worksheets/sheet{index}.xml"
        try:
            archive.getinfo(sheet_path)
        except KeyError:
            break
        output.append((f"Sheet{index}", sheet_path))
        index += 1
    return output


def _xlsx_parse_sheet_rows(
    archive: zipfile.ZipFile,
    sheet_path: str,
    shared_strings: list[str],
) -> list[tuple[int, list[str]]]:
    root = ET.fromstring(archive.read(sheet_path))
    ns = _xml_namespace(root)
    rows: list[tuple[int, list[str]]] = []
    for row_el in _xml_findall(root, ".//sheetData/row", ns):
        row_no = int(row_el.attrib.get("r") or len(rows) + 1)
        values_by_col: dict[int, str] = {}
        max_col = 0
        for cell in _xml_findall(row_el, "c", ns):
            ref = cell.attrib.get("r", "")
            match = re.match(r"^([A-Z]+)", ref)
            if not match:
                continue
            col_idx = _xlsx_col_to_index(match.group(1))
            value = normalize_cell(_xlsx_cell_value(cell, shared_strings, ns))
            if value:
                values_by_col[col_idx] = value
                max_col = max(max_col, col_idx)
        if not values_by_col:
            continue
        values = ["" for _ in range(max_col)]
        for col_idx, value in values_by_col.items():
            values[col_idx - 1] = value
        rows.append((row_no, values))
    return rows


def _xlsx_cell_value(cell: ET.Element, shared_strings: list[str], ns: str | None) -> str:
    cell_type = cell.attrib.get("t", "")
    inline = _xml_find(cell, "is", ns)
    if cell_type == "inlineStr" and inline is not None:
        return "".join(node.text or "" for node in _xml_findall(inline, ".//t", ns)).strip()
    value_el = _xml_find(cell, "v", ns)
    if value_el is None or value_el.text is None:
        return ""
    raw = value_el.text
    if cell_type == "s":
        try:
            return shared_strings[int(raw)]
        except Exception:
            return raw
    return raw


def _xml_namespace(root: ET.Element) -> str | None:
    if root.tag.startswith("{") and "}" in root.tag:
        return root.tag.split("}", 1)[0].strip("{")
    return None


def _xml_findall(root: ET.Element, path: str, ns: str | None) -> list[ET.Element]:
    if not ns:
        return list(root.findall(path))
    namespaced = "/".join(
        part if not part or part == "." or part.startswith("@") else f"{{{ns}}}{part}"
        for part in path.split("/")
    )
    return list(root.findall(namespaced))


def _xml_find(root: ET.Element, path: str, ns: str | None) -> ET.Element | None:
    if not ns:
        return root.find(path)
    return root.find(f"{{{ns}}}{path}")


def read_xls_rows(path: Path) -> list[tuple[str, list[tuple[int, list[str]]], dict[int, int]]]:
    try:
        import xlrd
    except Exception as exc:
        raise RuntimeError("xlrd is required to parse .xls files") from exc

    book = xlrd.open_workbook(str(path))
    output: list[tuple[str, list[tuple[int, list[str]]], dict[int, int]]] = []
    for sheet in book.sheets():
        rows: list[tuple[int, list[str]]] = []
        for idx in range(sheet.nrows):
            values = [
                xls_cell_text(sheet.cell_type(idx, col), sheet.cell_value(idx, col), book.datemode)
                for col in range(sheet.ncols)
            ]
            if any(values):
                rows.append((idx + 1, values))
        if rows:
            output.append((sheet.name or "Sheet", rows, vertical_spans((clo + 1, rlo + 1, chi, rhi) for rlo, rhi, clo, chi in getattr(sheet, 'merged_cells', []))))
    return output


def xls_cell_text(ctype: int, value: Any, datemode: int) -> str:
    """xlrd hands dates back as raw serial floats (45123.0) and booleans as
    1/0; without ctype handling every date column indexed as meaningless
    integers."""
    import xlrd

    if ctype == xlrd.XL_CELL_DATE:
        try:
            moment = xlrd.xldate.xldate_as_datetime(value, datemode)
        except Exception:
            return normalize_cell(value)
        if float(value) < 1:
            return moment.strftime("%H:%M:%S")
        if moment.hour == moment.minute == moment.second == 0:
            return moment.strftime("%Y-%m-%d")
        return moment.strftime("%Y-%m-%d %H:%M:%S")
    if ctype == xlrd.XL_CELL_BOOLEAN:
        return "TRUE" if value else "FALSE"
    if ctype == xlrd.XL_CELL_ERROR:
        return ""
    return normalize_cell(value)


def chunk_rows_to_blocks(
    doc_type: str,
    rows: list[tuple[int, list[str]]],
    sheet_name: str | None,
    max_tokens: int,
    overlap_tokens: int,
    header_spans: dict[int, int] | None = None,
) -> list[ParsedBlock]:
    if not rows:
        return []
    header_rows = detect_header_rows(rows, spans=header_spans)
    header_row_nos = {row_no for row_no, _ in header_rows}
    header = merge_header_rows([row for _, row in header_rows])
    header_line = header_text(header)
    # Title / description rows before the header: not emitted as data rows, written into the summary chunk
    first_header_no = min(header_row_nos) if header_row_nos else None
    lead_rows = [(row_no, row) for row_no, row in rows if first_header_no is not None and row_no < first_header_no and any((c or "").strip() for c in row)]
    lead_row_nos = {row_no for row_no, _ in lead_rows}
    # A vertically merged note block is spread over every row it covers (read_xlsx_rows), so by now it is
    # several identical lines of text: of adjacent identical lines only one is kept.
    # A note line is a passage of text, not a table row: its cells are joined with spaces, not " | ", so that
    # when it is over budget the chunker splits it by sentence instead of by cell
    lead_lines: list[str] = []
    for _, row in lead_rows:
        line = " ".join(cell for cell in map(flat_cell, row) if cell)
        if not lead_lines or line != lead_lines[-1]:
            lead_lines.append(line)
    blocks = [summary_block(doc_type, sheet_name, rows, header_line, lead_lines=lead_lines)]

    # Every emitted block carries the SHEET/ROWS/HEADER prefix in front of the
    # packed rows, so the row budget is what is left after that prefix;
    # otherwise a wide header silently pushes every block past max_tokens
    # (measured: 55 of 60 blocks at 450-507 against a 400 limit).
    # The budget is counted the same way as in the chunker (the prefix and the newline after every row count),
    # so the chunker does not split the block a second time.
    # Rows expanded with column names ("name: value | ...") go without the HEADER line: the column name already
    # stands before every cell, so the whole header would only repeat it.
    bare_prefix = "\n".join(x for x in (f"SHEET: {sheet_name}" if sheet_name else "", "ROWS: 00000-00000") if x)
    labelled_budget = max_tokens - count_tokens(bare_prefix) - 1
    row_budget = labelled_budget - (count_tokens(f"HEADER: {header_line}") + 1 if header_line else 0)
    # When the header takes more than three quarters of the budget, "whole header + positional rows" leaves no
    # room for data and every block is mostly header (a 40-column questionnaire sheet produced 330 chunks, each
    # 700-odd tokens of header next to a few dozen tokens of data): then every row of the table is expanded with
    # column names
    labelled_only = bool(header_line) and row_budget < max_tokens // 4

    current: list[tuple[int, str]] = []
    current_tokens = 0

    def flush(piece: int | None = None, *, labelled: bool = labelled_only) -> None:
        nonlocal current, current_tokens
        if not current:
            return
        body = "\n".join(line for _, line in current)
        prefix = []
        if sheet_name:
            prefix.append(f"SHEET: {sheet_name}")
        prefix.append(f"ROWS: {current[0][0]}-{current[-1][0]}")
        if header_line and not labelled:
            prefix.append(f"HEADER: {header_line}")
        text = "\n".join(prefix) + "\n" + body
        blocks.append(
            ParsedBlock(
                parser="native_table",
                parser_profile="table-native-v1",
                doc_type=doc_type,
                block_type="table",
                text=text,
                table_markdown=text,
                sheet_name=sheet_name,
                row_start=current[0][0],
                row_end=current[-1][0],
                # Split pieces of one over-budget row all share the same row
                # span; without the piece suffix they collide on block_id and
                # therefore on chunk_uid/point id, so only the last piece
                # would survive the upsert.
                block_id=f"{doc_type}-{safe_sheet(sheet_name)}-rows-{current[0][0]}-{current[-1][0]}"
                + (f"-p{piece}" if piece else ""),
                metadata={
                    "col_start": "A",
                    "col_end": col_name(max((len(row) for _, row in rows), default=1) - 1),
                },
            )
        )
        if overlap_tokens <= 0:
            current = []
            current_tokens = 0
            return
        tail: list[tuple[int, str]] = []
        tail_tokens = 0
        for item in reversed(current):
            tokens = count_tokens(item[1]) + 1
            if tail and tail_tokens + tokens > overlap_tokens:
                break
            tail.append(item)
            tail_tokens += tokens
        current = list(reversed(tail))
        current_tokens = tail_tokens

    budget = labelled_budget if labelled_only else row_budget
    for row_no, row in rows:
        if row_no in header_row_nos or row_no in lead_row_nos:
            continue
        line = row_to_text(row)
        if not line:
            continue
        over = not labelled_only and count_tokens(line) + 1 > row_budget
        pieces = split_long_row(row, labelled_budget - 1, header) if over or labelled_only else [line]
        if over or len(pieces) > 1:
            # A single row that alone exceeds the budget (a long free-text cell)
            # is split at cell boundaries into several lines, each emitted as
            # its own block.
            flush()
            for piece_no, piece_line in enumerate(pieces, start=1):
                current = [(row_no, piece_line)]
                flush(piece=piece_no, labelled=True)
            current = []
            current_tokens = 0
            continue
        line = pieces[0]
        tokens = count_tokens(line) + 1              # +1: the newline when joining back
        if current and current_tokens + tokens > budget:
            flush()
            # flush() keeps an overlap tail that always retains at least one
            # row however large; without this recheck the tail plus the new
            # row accumulates past the budget (same fix as chunker.chunk_text).
            if current and current_tokens + tokens > budget:
                current = []
                current_tokens = 0
        current.append((row_no, line))
        current_tokens += tokens
    flush()
    return blocks


def summary_block(
    doc_type: str,
    sheet_name: str | None,
    rows: list[tuple[int, list[str]]],
    header_line: str,
    lead_lines: list[str] | None = None,
) -> ParsedBlock:
    target = f"Sheet: {sheet_name}" if sheet_name else "CSV"
    text = f"{target}\nRows: {len(rows)}"
    for line in (lead_lines or []):
        if line.strip():
            text += f"\nTitle: {line}"          # not truncated: note lines are indexed only here; the chunker splits a long one to the budget
    if header_line:
        text += f"\nColumns: {header_line}"
    return ParsedBlock(
        parser="native_table",
        parser_profile="table-native-v1",
        doc_type=doc_type,
        block_type="table",
        text=text,
        table_markdown=text,
        sheet_name=sheet_name,
        row_start=rows[0][0] if rows else None,
        row_end=rows[-1][0] if rows else None,
        title=f"{sheet_name or doc_type} summary",
        block_id=f"{doc_type}-{safe_sheet(sheet_name)}-summary",
        metadata={"summary": True},
    )


_DISPIMG_RE = re.compile(r"^=\s*DISPIMG\(", re.IGNORECASE)


# The parts of a number format that are not format symbols: quoted literals, escaped characters,
# [conditions / colours], padding (_x) and fill (*x)
_FORMAT_LITERAL_RE = re.compile(r'"[^"]*"|\\.|\[[^\]]*\]|[_*].')
_FORMAT_DECIMALS_RE = re.compile(r"\.([0#?]+)")
_PERCENT_CONTEXT = Context(prec=400)      # the default 28 significant digits cannot hold a very large number, and rounding would raise


def percent_decimals(number_format: str | None) -> int | None:
    """For a percentage number format, how many decimals it displays (0% -> 0, 0.00% -> 2); None when it is
    not a percentage. Only the first section (the positive-number format) is considered."""
    section = _FORMAT_LITERAL_RE.sub("", str(number_format or "")).split(";")[0]
    if "%" not in section:
        return None
    decimals = _FORMAT_DECIMALS_RE.search(section)
    return len(decimals.group(1)) if decimals else 0


def normalize_cell(value: Any, number_format: str | None = None) -> str:
    """Cell value -> the form that is indexed, as close as possible to what the sheet shows (ignoring the number
    format, a cell showing 6% was indexed as 0.06, a formula result of 92% as 0.916666666666667, and a whole
    date carried 00:00:00). number_format is the cell's number format, used only for percentages; dates are
    always written in ISO form, while currency symbols and thousands separators are not restored."""
    if value is None:
        return ""
    if isinstance(value, str) and _DISPIMG_RE.match(value.strip()):
        return "(image)"                       # WPS embedded-image formula; storing it verbatim is only noise
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"   # the same form as the xls path
    if isinstance(value, datetime.datetime):
        if value.hour == value.minute == value.second == 0:
            return value.strftime("%Y-%m-%d")
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, (int, float)):
        decimals = percent_decimals(number_format)
        if decimals is not None and math.isfinite(value * 100):
            # Reduce to 15 significant digits before rounding: 0.545 is 0.54500000000000004 in the machine,
            # and the sheet shows 55%
            shown = Decimal(format(value * 100, ".15g")).quantize(
                Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP, context=_PERCENT_CONTEXT)
            return f"{shown:f}%"
        if isinstance(value, float):
            # Integral values lose the ".0"; the rest lose the binary floating-point tail (0.30000000000000004 -> 0.3)
            value = int(value) if value.is_integer() else format(value, ".15g")
    return clean_text(str(value))


_NUMERIC_RE = re.compile(r"^[\d.,%￥$€\-+/:\s]+$")
# date / version / id shaped values are data even when short (20260122, v7.0.2412a, SP2B09-x86)
# date-like runs, dotted versions, and CODE-123 / AB1234 style ids. Plain
# arch/platform words with a short number (arm64, x86_64, IPv6) stay headers.
_DATAISH_RE = re.compile(r"\d{6,}|\bv?\d+\.\d+(\.\d+)?[a-z]?\b|[A-Z]{2,}-\d{2,}|[A-Z]{2,}\d{3,}", re.I)


def _looks_like_header_cell(cell: str) -> bool:
    cell = (cell or "").strip()
    if not cell:
        return True  # blank cells are fine in a header (merged spans)
    if len(cell) > 40:
        return False
    if _NUMERIC_RE.match(cell) or _DATAISH_RE.search(cell):
        return False
    return True


def detect_header_rows(rows: list[tuple[int, list[str]]], max_header_rows: int = 3,
                       spans: dict[int, int] | None = None) -> list[tuple[int, list[str]]]:
    """Return the leading header rows (1..max_header_rows).

    Real product spreadsheets often carry a tiered header: a merged category
    row (Image name | v6.0 app version | v6.0 app version | Notes) above the true
    column names (Image name | x86_64 | arm64 | Notes). The reliable tell for a
    second tier is not cell shape (short product codes look just like labels)
    but the merge footprint: a row extends the header only if, column by column,
    it either repeats the row above, is blank where the row above is set, or the
    row above was blank -- and it does so in most columns while still adding
    at least one new value. A row with fresh values across the board is data.

    spans: the workbook's own vertical-merge footprint ({start row: end row}, see
    vertical_spans). A merge that starts on an accepted header row and reaches
    the next row proves that row is another header tier, whatever its cells
    look like (a product column named "Sample Collab v4.3->v4.11" is
    version-shaped, and the shape test alone would have rejected it -- Codex
    2026-09-14 R02).
    """
    first = first_nonempty_row(rows)
    if first is None:
        return []
    start = next(i for i, (rn, _) in enumerate(rows) if rn == first[0])
    header: list[tuple[int, list[str]]] = [first]
    spans = spans or {}
    for row_no, row in rows[start + 1 : start + max_header_rows]:
        if not any((c or "").strip() for c in row):
            break
        prev = header[-1][1]
        spanned = any(spans.get(rn, 0) >= row_no for rn, _ in header)
        if spanned or _extends_header(prev, row):
            header.append((row_no, row))
        else:
            break
    return header


def _extends_header(prev: list[str], row: list[str]) -> bool:
    """A second header row is recognised by one unambiguous tell only: some
    column repeats the cell directly above it verbatim ("Image name" over
    "Image name", "Mobile" over "Mobile" -- the footprint of a merged span),
    while the row also brings at least one new label and contains no
    data-shaped cell. Shape-based guesses ("looks short / looks like a label")
    were tried and oscillate on real sheets; a verbatim repeat almost never
    occurs in a data row.
    """
    width = max(len(prev), len(row))
    repeat = fresh = 0
    for col in range(width):
        a = (prev[col] if col < len(prev) else "").strip()
        b = (row[col] if col < len(row) else "").strip()
        if not b:
            continue
        if not _looks_like_header_cell(b):
            return False
        if a and b == a:
            repeat += 1
        elif b != a:
            fresh += 1
    return repeat >= 1 and fresh >= 1


def merge_header_rows(rows: list[list[str]]) -> list[str]:
    """Combine tiered header rows column-wise into 'upper/lower' names,
    dropping a repeated upper value and carrying merged spans forward."""
    if not rows:
        return []
    width = max(len(r) for r in rows)
    merged: list[str] = []
    for col in range(width):
        parts: list[str] = []
        last = ""
        for r in rows:
            v = (r[col] if col < len(r) else "").strip()
            if not v:
                continue
            if v == last:
                continue
            parts.append(v)
            last = v
        merged.append("/".join(parts))
    return merged


def split_long_row(row: list[str], budget: int, header: list[str] | None = None) -> list[str]:
    """A row over budget: split it into several pieces by cell, each cell carrying its column name
    (the column letter when the header has none), and later pieces start by repeating the row
    identifier (the first non-empty cell); empty cells are skipped, but the column names make the
    positions clear. Previously the non-empty cells were simply packed by budget and empty columns
    vanished, so a continuation piece no longer lined up with the full-row header (2026-09-13 Codex
    F01: a 40-column questionnaire sheet whose continuation held only the W-AE stretch under the
    complete header). No piece exceeds budget: a single cell that is still over budget has its value
    split by token (for the packing see chunker.pack_labelled_cells)."""
    items = []
    for idx, cell in enumerate(row):
        text = flat_cell(cell)
        if not text:
            continue
        name = (header[idx] if header and idx < len(header) and str(header[idx] or "").strip() else col_name(idx))
        items.append(f"{flat_cell(str(name))}: {text}")
    if not items:
        return [row_to_text(row)]
    return pack_labelled_cells(items, budget)


def _is_note_row(cells: list[str]) -> bool:
    return any(len(c) > 40 or "http://" in c or "https://" in c for c in cells)


def first_nonempty_row(rows: list[tuple[int, list[str]]]) -> tuple[int, list[str]] | None:
    """Header candidate row. Skips leading title and description rows: a row with only one filled cell
    (or a merged title whose cells are all identical) followed by wider rows; or a description row
    where some cell is long text / a URL ("open-source component licence address: Apache2.0
    :http://..."). Falls back to the first non-empty row when none qualifies."""
    window = rows[:12]
    widest = max((sum(1 for c in row if (c or "").strip()) for _, row in window), default=0)
    for row_no, row in window:
        cells = [(c or "").strip() for c in row if (c or "").strip()]
        if not cells:
            continue
        if len(set(cells)) == 1 and (len(cells) < widest or len(cells) >= 2):
            continue                          # merged title row: a single cell, or several cells with the same text
        if _is_note_row(cells):
            continue                          # long text / URL: a description row, not a header
        return row_no, row
    for row_no, row in window:
        if any(row):
            return row_no, row
    return None


_CELL_WS_RE = re.compile(r"\s+")


def flat_cell(value: str | None) -> str:
    """Flatten a cell to one line: newlines inside the cell become spaces and runs of
    whitespace collapse. A spot check found in-cell newlines ("Session list<LF>@owner",
    header "WPS Collab<LF>public") splitting one record over two or three lines, so neither
    readers nor the model could line values up with columns."""
    return _CELL_WS_RE.sub(" ", value or "").strip()


def _joined_cells(values: list[str]) -> str:
    """Join cells positionally, trimming only trailing empties. Dropping
    interior holes shifted every later cell left, so a sparse row no longer
    lined up with the HEADER prefix and values got attributed to the wrong
    column."""
    cells = [flat_cell(value) for value in values]
    while cells and not cells[-1]:
        cells.pop()
    if not any(cells):
        return ""
    return " | ".join(cells).strip()


def header_text(header: list[str]) -> str:
    return _joined_cells(header)[:4000]


def row_to_text(row: list[str]) -> str:
    return _joined_cells(row)


def col_name(idx: int) -> str:
    letters = ""
    idx += 1
    while idx:
        idx, rem = divmod(idx - 1, 26)
        letters = string.ascii_uppercase[rem] + letters
    return letters


def safe_sheet(sheet_name: str | None) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in (sheet_name or "csv"))[:80]
