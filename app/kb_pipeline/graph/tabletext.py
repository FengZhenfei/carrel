"""Wide tables rendered as "column: value" pairs for the model only.

A spot check of a 15-column feature-comparison sheet (8 columns of per-product 1/0 flags) found roughly one in ten
"supports / does not support" statements in entity descriptions wrong, each wrong value equal to a neighbouring
column: the model miscounted columns in wide rows, while the facts stage read the same text almost without error.
So rows of tables with many columns are rewritten as "column: value | column: value" before they reach the
extraction model (the facts model too, GRAPH_FACTS_WIDE_TABLES=0 turns that off); chunk text and retrieval are
untouched. Rewritten units carry the render version in their unit_id (table_render_tag), so only their extraction
and facts caches are invalidated; every other unit keeps hitting the cache.

Two table forms are recognised: native table blocks (SHEET: / ROWS: / HEADER: a | b | c prefix lines, data rows
"v1 | v2 | v3" with empty cells kept in position) and markdown pipe tables (a "| a | b |" header row followed by
a |---| separator). Tables with fewer than WIDE_TABLE_MIN_COLUMNS header columns are left alone; a blank line
ends a table.

Positions are checked before names are assigned by position (2026-09-29 audit: on the three kinds of row below
the first version handed the model wrong column names):
- A native row or header whose first cell is empty: native_table._joined_cells strips the whitespace at the start
  of the joined line, so the row starts with "| " and has lost one empty cell; pairing by position then shifts the
  whole row one column to the left. The empty cell is put back.
- The continuation pieces of an over-long row (native_table.split_long_row, the chunker's _split_table_row) are
  already "column: value" and skip empty cells; pairing them by position again gives "id: id: A-001 | name: note:
  ...". Such rows are left as they are, and where the chunk text already carries a second, positional layer of
  names added by the chunker, the outer layer is removed.
- A markdown row whose cell count differs from the header (a row flattened from merged cells) has no reliable
  positions and is left as it is.
Units whose rendering differs from the first version for one of these reasons use RENDER_TAG_REALIGNED; every
other wide-table unit keeps RENDER_TAG and its cache."""
from __future__ import annotations

import os
import re

WIDE_TABLE_MIN_COLUMNS = 6
FACTS_EXPAND_WIDE_TABLES = os.getenv("GRAPH_FACTS_WIDE_TABLES", "1").strip().lower() in {"1", "true", "yes", "on"}
_FACTS_SUFFIX = "+facts" if FACTS_EXPAND_WIDE_TABLES else ""
RENDER_TAG = "wide-table-v1" + _FACTS_SUFFIX
RENDER_TAG_REALIGNED = "wide-table-v2" + _FACTS_SUFFIX

_MD_SEP_RE = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
# columns the header leaves unnamed: native tables use the column letter, the chunker "column N"
_AUTO_NAME_RE = re.compile(r"^(?:[A-Z]{1,3}|column \d+)$")


def _native_cells(line: str) -> tuple[list[str], bool]:
    """Split one native-table line into cells by position; returns (cells, whether the leading empty cell was put back)."""
    s = line
    padded = s.startswith("|")
    if padded:
        s = " " + s
    if s.endswith("|"):
        s = s + " "
    return [c.strip() for c in s.split(" | ")], padded


def _md_cells(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _col_name(idx: int) -> str:
    letters = ""
    idx += 1
    while idx:
        idx, rem = divmod(idx - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def _pairs(header: list[str], cells: list[str]) -> str:
    items = []
    for idx, cell in enumerate(cells):
        if not cell:
            continue
        name = header[idx] if idx < len(header) and header[idx] else _col_name(idx)
        items.append(f"{name}: {cell}")
    return " | ".join(items)


def _label_of(cell: str, names: set[str]) -> str | None:
    """The column name a cell starts with ("column: value"); None when it does not start with a column of this table.
    The longest match wins, since a column name may contain a colon itself."""
    best = ""
    for name in names:
        if len(name) > len(best) and cell.startswith(name + ": "):
            best = name
    if best:
        return best
    head, sep, _rest = cell.partition(": ")
    return head if sep and _AUTO_NAME_RE.match(head) else None


def _labelled_cells(cells: list[str], names: set[str]) -> list[str] | None:
    """The cells to show the model when the row already carries column names; None for an ordinary positional row.
    Every non-empty cell has to start with a column name of this table. When each cell carries a second column name
    after the first, the chunker paired a continuation piece by position once more: the inner name is the right one
    and the outer layer is dropped."""
    filled = [c for c in cells if c]
    if not filled:
        return None
    labels = [_label_of(c, names) for c in filled]
    if not all(labels):
        return None
    inner = [c[len(label) + 2:].strip() for c, label in zip(filled, labels)]
    if all(_label_of(c, names) for c in inner):
        return inner
    return filled


def _render(text: str, min_columns: int) -> tuple[str | None, bool]:
    """Returns (the expanded text, None when nothing changed; whether the rendering differs from the first version)."""
    lines = str(text or "").split("\n")
    out: list[str] = []
    header: list[str] | None = None        # current header of a native table block
    md_header: list[str] | None = None     # current header of a markdown table
    names: set[str] = set()
    changed = False
    realigned = False
    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if s.startswith("HEADER:"):
            cells, padded = _native_cells(s[len("HEADER:"):].strip())
            header = cells if len(cells) >= min_columns else None      # the empty cell put back counts as a column
            md_header = None
            names = {c for c in (header or []) if c}
            realigned = realigned or (header is not None and padded)
            out.append(line)
            i += 1
            continue
        if s.startswith("|") and i + 1 < len(lines) and _MD_SEP_RE.match(lines[i + 1].strip() or "x"):
            cells = _md_cells(s)
            md_header = cells if len(cells) >= min_columns else None
            header = None
            names = {c for c in (md_header or []) if c}
            out.append(line)
            out.append(lines[i + 1])
            i += 2
            continue
        if not s:
            header = None
            md_header = None
            names = set()
            out.append(line)
            i += 1
            continue
        if md_header is not None and s.startswith("|"):
            cells = _md_cells(s)
            kept = _labelled_cells(cells, names)
            if kept is not None:
                rendered = "| " + " | ".join(kept) + " |"
                out.append(rendered if kept != [c for c in cells if c] else line)
                changed = changed or kept != [c for c in cells if c]
                realigned = True
            elif len(cells) != len(md_header):
                out.append(line)
                realigned = True
            else:
                rendered = _pairs(md_header, cells)
                out.append(rendered or line)
                changed = changed or bool(rendered)
        elif header is not None and " | " in s and not s.startswith(("SHEET:", "ROWS:", "TITLE:")):
            cells, padded = _native_cells(s)
            kept = _labelled_cells(cells, names)
            if kept is not None:
                unchanged = kept == [c for c in cells if c]
                out.append(line if unchanged else " | ".join(kept))
                changed = changed or not unchanged
                realigned = True
            else:
                rendered = _pairs(header, cells)
                out.append(rendered or line)
                changed = changed or bool(rendered)
                realigned = realigned or padded
        else:
            out.append(line)
        i += 1
    return ("\n".join(out) if changed else None), realigned


def expand_wide_tables(text: str, *, min_columns: int = WIDE_TABLE_MIN_COLUMNS) -> str | None:
    """Rewrite the data rows of wide tables as "column: value | ..."; None when there is no wide table (the caller
    keeps the original text)."""
    return _render(text, min_columns)[0]


def table_render_tag(text: str) -> str:
    """The render version (part of the unit_id) when the unit text gets expanded, otherwise an empty string: only
    wide-table units lose their cache. Units whose rendering differs from the first version (the three kinds of row
    in the module docstring) return RENDER_TAG_REALIGNED; only those are extracted again."""
    rendered, realigned = _render(text, WIDE_TABLE_MIN_COLUMNS)
    if rendered is None:
        return ""
    return RENDER_TAG_REALIGNED if realigned else RENDER_TAG
