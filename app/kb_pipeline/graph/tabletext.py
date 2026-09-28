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
ends a table."""
from __future__ import annotations

import os
import re

WIDE_TABLE_MIN_COLUMNS = 6
FACTS_EXPAND_WIDE_TABLES = os.getenv("GRAPH_FACTS_WIDE_TABLES", "1").strip().lower() in {"1", "true", "yes", "on"}
RENDER_TAG = "wide-table-v1" + ("+facts" if FACTS_EXPAND_WIDE_TABLES else "")

_MD_SEP_RE = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def _native_cells(line: str) -> list[str]:
    return [c.strip() for c in line.split(" | ")]


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


def expand_wide_tables(text: str, *, min_columns: int = WIDE_TABLE_MIN_COLUMNS) -> str | None:
    """Rewrite the data rows of wide tables as "column: value | ..."; returns None when there is no wide table
    (callers keep the original text)."""
    lines = str(text or "").split("\n")
    out: list[str] = []
    header: list[str] | None = None        # current native-table header
    md_header: list[str] | None = None     # current markdown-table header
    changed = False
    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if s.startswith("HEADER:"):
            cells = _native_cells(s[len("HEADER:"):].strip())
            header = cells if len(cells) >= min_columns else None
            md_header = None
            out.append(line)
            i += 1
            continue
        if s.startswith("|") and i + 1 < len(lines) and _MD_SEP_RE.match(lines[i + 1].strip() or "x"):
            cells = _md_cells(s)
            md_header = cells if len(cells) >= min_columns else None
            header = None
            out.append(line)
            out.append(lines[i + 1])
            i += 2
            continue
        if not s:
            header = None
            md_header = None
            out.append(line)
            i += 1
            continue
        if md_header is not None and s.startswith("|"):
            rendered = _pairs(md_header, _md_cells(s))
            out.append(rendered or line)
            changed = changed or bool(rendered)
        elif header is not None and " | " in s and not s.startswith(("SHEET:", "ROWS:", "TITLE:")):
            rendered = _pairs(header, _native_cells(s))
            out.append(rendered or line)
            changed = changed or bool(rendered)
        else:
            out.append(line)
        i += 1
    return "\n".join(out) if changed else None


def table_render_tag(text: str) -> str:
    """The render version (goes into unit_id) when the unit text would be expanded, else an empty string: only
    wide-table units lose their cache."""
    return RENDER_TAG if expand_wide_tables(text) is not None else ""
