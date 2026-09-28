"""Table ambiguity detection (Codex review F02).

MinerU's table recognition concatenates lines stacked vertically inside one cell (75 / 75 / 57 mA
listed per test condition) straight into `757557` and `mAmAmA`, and the values then pass through
chunking, embedding and fact extraction looking perfectly normal. The merging itself happens upstream;
this module does only two things: flag suspicious rows from structural signals (zero cost,
deterministic), and provide the grid and expansion tools that table_repair uses for screenshot
verification repair. It never guesses how to split 757557 into numbers.

Signals (per row):
  - a unit cell is the same unit repeated: mAmAmA = mA x 3, giving the number of stacked lines n;
  - a condition cell holds >= 2 "symbol = value" segments: t_RC = 20 ns t_RC = 25 ns ...;
  - a value cell is a long run of plain digits / has several decimal points (7.57.55.7).
"n >= 2" together with a long digit run in the same row is a high-confidence merge; a long digit run
alone, clearly longer than the other values in its column, is low confidence.
"""
from __future__ import annotations

import re
from typing import Any

from .common import html_table_to_grid

_UNIT_REPEAT_RE = re.compile(r"^(?P<u>[^\d\s|]{1,6}?)(?P=u)+$")
_UNIT_LETTER_RE = re.compile(r"[A-Za-z°µμΩ℧%‰℃℉]")
_ASSIGN_RE = re.compile(r"[A-Za-z_}\]\$)]\s*[=≤≥<>]\s*[-+]?\d")
_DIGITS_RE = re.compile(r"^[-+]?\d{4,}$")
_MULTI_DOT_RE = re.compile(r"^[-+]?\d+(?:\.\d+){2,}$")
_NUMERIC_RE = re.compile(r"^[-+]?\d+(?:[.,]\d+)?$")
GLUED_KINDS = ("glued_values", "glued_units")


def unit_repeat(cell: str) -> int:
    """'mAmAmA' -> 3; '°C°C' -> 2; 'mA' / 'Vpp' / 'mm' / '**' -> 1.
    The repeated unit must be at least two characters and look like a unit (letters / degree / micro /
    ohm / percent sign): 'mm' is m x 2 but is a unit in itself, '**' is markdown bold, and a Chinese
    word such as "output data" x 6 is stacked text in a truth table; none of these is what we are
    after here."""
    text = (cell or "").strip()
    if len(text) > 40:
        return 1
    m = _UNIT_REPEAT_RE.fullmatch(text)
    if not m:
        return 1
    base = m.group("u")
    if len(base) < 2 or not _UNIT_LETTER_RE.search(base):
        return 1
    return len(text) // len(base)


def assignment_count(cell: str) -> int:
    return len(_ASSIGN_RE.findall(cell or ""))


def _digit_len(cell: str) -> int:
    return sum(ch.isdigit() for ch in cell)


def detect_table_ambiguity(rows: list[list[str]]) -> list[dict[str, Any]]:
    """Grid -> list of suspicious cells, each {row, col, kind, n, raw, key, confidence}."""
    flags: list[dict[str, Any]] = []
    if not rows:
        return flags
    width = max(len(r) for r in rows)
    # Median digit count of the numeric cells per column (used to judge "clearly longer than the other
    # values in the column")
    col_digits: dict[int, list[int]] = {}
    for r in rows[1:]:
        for j, cell in enumerate(r):
            if _NUMERIC_RE.fullmatch((cell or "").strip()):
                col_digits.setdefault(j, []).append(_digit_len(cell))
    for i, row in enumerate(rows):
        key = (row[0] if row else "").strip()
        n_from_units = 0
        n_from_conditions = 0
        cells: list[tuple[int, str, str, int]] = []      # (col, kind, raw, n)
        for j in range(width):
            cell = (row[j] if j < len(row) else "").strip()
            if not cell:
                continue
            n = unit_repeat(cell)
            if n >= 2:
                cells.append((j, "glued_units", cell, n))
                n_from_units = max(n_from_units, n)
                continue
            n = assignment_count(cell)
            if n >= 2:
                cells.append((j, "multi_condition", cell, n))
                n_from_conditions = max(n_from_conditions, n)
                continue
            if _DIGITS_RE.fullmatch(cell) or _MULTI_DOT_RE.fullmatch(cell):
                cells.append((j, "glued_values", cell, 0))
        if not cells:
            continue
        n_row = n_from_units or n_from_conditions
        values = [c for c in cells if c[1] == "glued_values"]
        if n_from_units >= 2 or (n_from_conditions >= 2 and values):
            # A repeated unit is hard proof of stacking by itself; multiple conditions count only
            # together with a long digit run (three conditions per row is normal in a capacitor table)
            confidence = "high"
        elif values:
            # Only a long digit run: suspicious only when clearly longer than the other values in the
            # column (a column of 5-digit order codes is 5 digits throughout, which does not count)
            confidence = "low"
            keep = []
            for j, kind, raw, _ in values:
                others = [d for d in col_digits.get(j, []) if d != _digit_len(raw)]
                median = sorted(others)[len(others) // 2] if others else 0
                if median and _digit_len(raw) >= 2 * median:
                    keep.append((j, kind, raw, 0))
            if not keep:
                continue
            cells = keep
        else:
            continue
        for j, kind, raw, n in cells:
            flags.append({"row": i, "col": j, "kind": kind, "n": n or (n_row if kind == "glued_values" else 0),
                          "raw": raw, "key": key, "confidence": confidence})
    return flags


def table_ambiguity_flags(html: str) -> list[dict[str, Any]]:
    rows = html_table_to_grid(html or "")
    return detect_table_ambiguity(rows) if rows else []


def high_confidence(flags: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return [f for f in (flags or []) if f.get("confidence") == "high"]


def unrepaired_glued_values(flags: list[dict[str, Any]] | None, repair: dict[str, Any] | None) -> list[str]:
    """Merged raw strings not yet split by screenshot verification (fact extraction uses this to mark
    facts that picked up these values as untrusted)."""
    if repair and str(repair.get("status")) == "verified":
        fixed = set(int(r) for r in (repair.get("rows") or []))
    else:
        fixed = set()
    out: list[str] = []
    for f in high_confidence(flags):
        if f.get("kind") == "glued_values" and int(f.get("row", -1)) not in fixed:
            raw = str(f.get("raw") or "")
            if raw and raw not in out:
                out.append(raw)
    return out


def expand_glued_rows(rows: list[list[str]], repairs: dict[int, dict[int, list[str]]]) -> list[list[str]]:
    """Expand verified rows into n sub-rows: {row index: {column index: [stacked lines]}}; n must be the
    same for every column of a row, and columns without a split are copied into every sub-row."""
    out: list[list[str]] = []
    for i, row in enumerate(rows):
        plan = repairs.get(i)
        if not plan:
            out.append(list(row))
            continue
        n = max(len(parts) for parts in plan.values())
        if n < 2 or any(len(parts) != n for parts in plan.values()):
            out.append(list(row))
            continue
        for k in range(n):
            out.append([plan[j][k] if j in plan else cell for j, cell in enumerate(row)])
    return out


def table_ambiguity_summary(blocks: list[Any]) -> dict[str, int]:
    flagged = verified = unverified = 0
    for b in blocks:
        meta = getattr(b, "metadata", None) or {}
        if not high_confidence(meta.get("table_flags")):
            continue
        flagged += 1
        repair = meta.get("table_repair") or {}
        if str(repair.get("status")) == "verified":
            verified += 1
        else:
            unverified += 1
    return {"flagged": flagged, "verified": verified, "unverified": unverified}
