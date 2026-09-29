"""Axis and time: a document's axis value (date or version), date normalisation, the time window in a question.
All rules, no model calls; both Chinese and English are recognised.

There are no domain words in the pipeline: whether a KB is ordered by date (checkup reports, test records,
meeting minutes) or by version (datasheets, specifications, code) is decided by the scenario profile's axis
field; when no value can be found it is left empty, never guessed (plan chapter 6: no axis, no timeline page).
The parsing ladder borrows from Semantica's TemporalNormalizer: ISO -> year / year-month -> Chinese
year-month-day -> English month names -> bare year; ambiguous forms (01/02/03) are not interpreted and
return None.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Any, Iterable

AXIS_KINDS = ("date", "version", "none")

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_ISO_RE = re.compile(r"(?<!\d)(?P<y>(?:19|20)\d{2})[-/.](?P<m>0?[1-9]|1[0-2])(?:[-/.](?P<d>0?[1-9]|[12]\d|3[01]))?(?!\d)")
_COMPACT_RE = re.compile(r"(?<!\d)(?P<y>(?:19|20)\d{2})(?P<m>0[1-9]|1[0-2])(?P<d>0[1-9]|[12]\d|3[01])(?!\d)")
_CJK_RE = re.compile(r"(?P<y>(?:19|20)\d{2})\s*年\s*(?:(?P<m>0?[1-9]|1[0-2])\s*月\s*(?:(?P<d>0?[1-9]|[12]\d|3[01])\s*[日号])?)?")
_EN_MDY_RE = re.compile(r"\b(?P<mon>[A-Za-z]{3,9})\.?\s+(?P<d>0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?,?\s+(?P<y>(?:19|20)\d{2})\b")
_EN_DMY_RE = re.compile(r"\b(?P<d>0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?\s+(?P<mon>[A-Za-z]{3,9})\.?,?\s+(?P<y>(?:19|20)\d{2})\b")
_EN_MY_RE = re.compile(r"\b(?P<mon>[A-Za-z]{3,9})\.?,?\s+(?P<y>(?:19|20)\d{2})\b")
_YEAR_RE = re.compile(r"(?<![\d.])(?P<y>(?:19|20)\d{2})(?![\d.])")
_VERSION_RE = re.compile(
    r"(?:\brev(?:ision)?\.?\s*(?P<rev>\*?[A-Za-z0-9]{1,4}(?:\.[0-9]{1,3})*)\b)|"
    r"(?:\bv(?:ersion)?\.?\s*(?P<ver>\d{1,3}(?:\.\d{1,3}){0,3})\b)|"
    r"(?:版本\s*[::]?\s*(?P<cver>[A-Za-z]?\d{1,3}(?:\.\d{1,3}){0,3}))",
    re.IGNORECASE)


def _iso(y: str, m: str | None = None, d: str | None = None) -> str | None:
    year = int(y)
    if m is None:
        return f"{year:04d}"
    month = int(m)
    if not 1 <= month <= 12:
        return None
    if d is None:
        return f"{year:04d}-{month:02d}"
    day = int(d)
    try:
        date(year, month, day)
    except ValueError:
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


def parse_date(text: Any) -> str | None:
    """The first date in a text, normalised to YYYY / YYYY-MM / YYYY-MM-DD; None when nothing is recognised.
    Only year-first forms and forms with a month name / Chinese date units count; ambiguous ones like 01/02/2024
    are not interpreted."""
    s = str(text or "")
    for rx in (_ISO_RE, _COMPACT_RE, _CJK_RE):
        m = rx.search(s)
        if m:
            out = _iso(m.group("y"), m.group("m"), m.group("d"))
            if out:
                return out
    for rx in (_EN_MDY_RE, _EN_DMY_RE, _EN_MY_RE):
        m = rx.search(s)
        if m and m.group("mon").lower() in _MONTHS:
            out = _iso(m.group("y"), str(_MONTHS[m.group("mon").lower()]), m.groupdict().get("d"))
            if out:
                return out
    m = _YEAR_RE.search(s)
    return _iso(m.group("y")) if m else None


def find_dates(text: Any) -> list[str]:
    """All normalisable dates in a text (deduplicated, order kept); bare-year matches count too."""
    s = str(text or "")
    found: list[tuple[int, str]] = []
    spans: list[tuple[int, int]] = []

    def add(value: str | None, span: tuple[int, int]) -> None:
        if value and not any(a <= span[0] < b or a < span[1] <= b for a, b in spans):
            spans.append(span)
            found.append((span[0], value))

    for rx in (_ISO_RE, _COMPACT_RE, _CJK_RE):
        for m in rx.finditer(s):
            add(_iso(m.group("y"), m.group("m"), m.group("d")), m.span())
    for rx in (_EN_MDY_RE, _EN_DMY_RE, _EN_MY_RE):
        for m in rx.finditer(s):
            if m.group("mon").lower() in _MONTHS:
                add(_iso(m.group("y"), str(_MONTHS[m.group("mon").lower()]), m.groupdict().get("d")), m.span())
    for m in _YEAR_RE.finditer(s):
        add(_iso(m.group("y")), m.span())
    out: list[str] = []
    for _, value in sorted(found):          # ordered by position: the earliest date on the first page is the document date
        if value not in out:
            out.append(value)
    return out


def find_versions(text: Any) -> list[str]:
    """Rev *L, Rev. 1.2, Version 3, v2.1.0, the Chinese "version 2.1" -> normalised to short strings such as
    'rev *L' / 'v1.2' (deduplicated, order kept)."""
    out: list[str] = []
    for m in _VERSION_RE.finditer(str(text or "")):
        if m.group("rev"):
            value = f"rev {m.group('rev').upper() if m.group('rev').startswith('*') else m.group('rev')}"
        elif m.group("ver"):
            value = f"v{m.group('ver')}"
        else:
            value = f"v{m.group('cver')}"
        if value not in out:
            out.append(value)
    return out


def document_axis(rel_path: str, head_text: str = "", *, kind: str = "auto") -> dict[str, str]:
    """The document's axis value. Returns {"kind": date|version|none, "value": ..., "date": ..., "version": ...}.
    Date: the one in the file name ("2024 checkup report" -> 2024) takes precedence over the first page (when
    the first page gives a more precise date, it completes the file-name date); the first-page text only counts
    dates written to the month or day: a lone four-digit number there (1920 pixels, 2048 tokens, 2023 employees,
    "from 1988 to 2023") is mostly not the document's date, and taking it would make it the validity of every
    fact in the document, so an empty value is better; version: the Rev / Version at the start of the first
    page; with kind=auto a date wins when present, a version is used when only it exists, and none is recorded
    when neither is found."""
    name = str(rel_path or "").rsplit("/", 1)[-1]
    head = str(head_text or "")[:1500]
    name_date = parse_date(name)
    head_dates = [d for d in find_dates(head) if len(d) > 4]
    head_date = head_dates[0] if head_dates else None
    date_value = name_date or ""
    if name_date and head_date and head_date.startswith(name_date) and len(head_date) > len(name_date):
        date_value = head_date          # the file name has only a year, the first page a full date of the same year
    elif not name_date and head_date:
        date_value = head_date
    versions = find_versions(name) or find_versions(head)
    version_value = versions[0] if versions else ""
    wanted = str(kind or "auto").lower()
    if wanted == "date":
        chosen = ("date", date_value) if date_value else ("none", "")
    elif wanted == "version":
        chosen = ("version", version_value) if version_value else (("date", date_value) if date_value else ("none", ""))
    else:
        chosen = ("date", date_value) if date_value else (("version", version_value) if version_value else ("none", ""))
    return {"kind": chosen[0], "value": chosen[1], "date": date_value, "version": version_value}


def axis_value_kind(value: Any) -> str:
    """Whether an axis value is a date or a version (an empty string for an empty value). In a KB whose profile
    is version, documents without a version number get a date axis value, and validity periods the model copies
    from the text are dates as well: both kinds coexist in one KB. Dates and versions have no order between
    them, so each kind can only be sorted on its own."""
    s = str(value or "")
    if not s:
        return ""
    return "date" if _ISO_RE.match(s) or _YEAR_RE.fullmatch(s) else "version"


def axis_sort_key(value: Any) -> tuple:
    """Sort key by axis value: dates lexicographically (ISO prefixes are ordered); versions by natural numeric
    segments; empty values last.
    Dates sort before versions only to keep the sort stable, not to mean earlier; code that compares order
    separates the two by axis_value_kind first."""
    s = str(value or "")
    kind = axis_value_kind(s)
    if not kind:
        return (2, ())
    if kind == "date":
        return (0, (s,))
    parts = tuple(int(p) if p.isdigit() else p.lower() for p in re.findall(r"\d+|[A-Za-z*]+", s))
    return (1, parts)


# ── Time window in a question ──────────────────────────────────────────────────

_RANGE_CJK_RE = re.compile(r"(?P<a>(?:19|20)\d{2})\s*年?\s*(?:到|至|—|-|~|～)\s*(?P<b>(?:19|20)\d{2})\s*年?")
_RANGE_EN_RE = re.compile(r"\b(?:from\s+)?(?P<a>(?:19|20)\d{2})\s*(?:to|-|–|through)\s*(?P<b>(?:19|20)\d{2})\b", re.IGNORECASE)
_RECENT_RE = re.compile(r"(?:这|近|最近|过去|前)\s*(?P<n>[一二两三四五六七八九十\d]+)\s*年|(?:last|past|recent)\s+(?P<en>\d+|two|three|four|five)\s+years?", re.IGNORECASE)
_CJK_NUM = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
_EN_NUM = {"two": 2, "three": 3, "four": 4, "five": 5}
_SINCE_RE = re.compile(r"(?:自|从)\s*(?P<y>(?:19|20)\d{2})\s*年?\s*(?:以来|起|开始)?|\bsince\s+(?P<ey>(?:19|20)\d{2})\b", re.IGNORECASE)


def question_time_window(question: str, *, today: date | None = None) -> dict[str, str] | None:
    """A time expression in a question -> {"from": YYYY, "to": YYYY, "text": the original wording}.
    "these two years" / "the last two years" / "the last three years" / "the past five years" = counting back
    from and including this year; this year, last year, the year before last; 2024 to 2026; since 2023; a bare
    "2024" = that year. Returns None when there is none."""
    q = str(question or "")
    now = today or date.today()
    m = _RANGE_CJK_RE.search(q) or _RANGE_EN_RE.search(q)
    if m:
        a, b = sorted((m.group("a"), m.group("b")))
        return {"from": a, "to": b, "text": m.group(0)}
    m = _RECENT_RE.search(q)
    if m:
        raw = m.group("n") or m.group("en") or ""
        n = _CJK_NUM.get(raw) or _EN_NUM.get(raw.lower()) or (int(raw) if raw.isdigit() else 0)
        if n > 0:
            return {"from": str(now.year - n + 1), "to": str(now.year), "text": m.group(0)}
    m = _SINCE_RE.search(q)
    if m:
        y = m.group("y") or m.group("ey")
        return {"from": y, "to": str(now.year), "text": m.group(0)}
    if re.search(r"今年|this\s+year", q, re.IGNORECASE):
        return {"from": str(now.year), "to": str(now.year), "text": "今年"}
    if re.search(r"去年|last\s+year", q, re.IGNORECASE):
        return {"from": str(now.year - 1), "to": str(now.year - 1), "text": "去年"}
    if re.search(r"前年", q):
        return {"from": str(now.year - 2), "to": str(now.year - 2), "text": "前年"}
    years = [y for y in find_dates(q) if len(y) >= 4]
    if years:
        ys = sorted({y[:4] for y in years})
        return {"from": ys[0], "to": ys[-1], "text": ", ".join(years)}
    return None


def in_window(value: Any, window: dict[str, str] | None) -> bool:
    """Whether an axis value (ISO date or year) falls inside the time window; with no window, or a value that is
    not a date, it counts as inside (no filtering, only weighting)."""
    if not window:
        return True
    s = str(value or "")
    if not s or not (_ISO_RE.match(s) or _YEAR_RE.fullmatch(s)):
        return True
    return str(window.get("from") or "0000") <= s[:4] <= str(window.get("to") or "9999")


def axis_for_units(units: Iterable[Any], *, kind: str = "auto") -> dict[str, dict[str, str]]:
    """Axis value per document: the text of each document's earliest unit serves as its first page. Returns
    {doc_id: document_axis(...)}."""
    first_text: dict[str, tuple[int, str, str]] = {}
    for u in units:
        cur = first_text.get(u.doc_id)
        if cur is None or int(getattr(u, "order", 0)) < cur[0]:
            first_text[u.doc_id] = (int(getattr(u, "order", 0)), str(getattr(u, "rel_path", "") or ""), str(getattr(u, "text", "") or ""))
    return {doc: document_axis(rel, text, kind=kind) for doc, (_, rel, text) in first_text.items()}
