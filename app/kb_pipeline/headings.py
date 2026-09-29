"""Layout inference: headings MinerU failed to mark, and footers that are nothing but a page number.

A leaf module depending only on re: both the parsers (parsers/common.py) and the chunker
(chunking/chunker.py) need it, and either importing the other would create a cycle.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

# Headings MinerU cannot recognize are recovered from layout conventions (borrowing the regex set from
# WeKnora's patterns.go, tightened for our corpus). Only whole short lines are considered: anything that is
# not a heading -- values like "1.5 V typical", numbered list items like "1. Typical values are for
# reference only, ..." -- must be kept out, otherwise section_path gets polluted and the chunker's
# structural cut points land in the wrong places.
_CJK_NUM = r"[一二三四五六七八九十百千零〇0-9]+"
_HEADING_TAIL = r"(?=\s|[:：.．、—\-]|$)"
# Chinese typesetting often glues the number to the title (e.g. "1.2" fused with the title text, or
# "Chapter One" fused with "General Provisions"): such unseparated forms are only accepted on very short
# whole lines, otherwise a sentence like "for the notes on chapter 3 see the appendix" would be taken for
# a heading too
_TIGHT_CJK_TAIL = r"(?=[一-鿿])"
TIGHT_HEADING_MAX_CHARS = 20
_CHAPTER_RES: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(rf"^\s*第\s*{_CJK_NUM}\s*(?:章|部分|篇|卷){_HEADING_TAIL}"), 1),
    (re.compile(rf"^\s*第\s*{_CJK_NUM}\s*(?:节|節){_HEADING_TAIL}"), 2),
    (re.compile(r"^\s*(?:Chapter|Part)\s+(?:\d+|[IVX]{1,5})(?=[\.:\s]|$)", re.IGNORECASE), 1),
    (re.compile(r"^\s*Section\s+(?:\d+|[IVX]{1,5})(?=[\.:\s]|$)", re.IGNORECASE), 2),
    (re.compile(r"^\s*(?:附录|Appendix)\s*[A-Z一二三四五六七八九十0-9]{1,2}(?=[\.:：\s]|$)", re.IGNORECASE), 1),
)
# "1.2.3 Electrical characteristics", "§3.3 Closed interval": depth = number of dotted segments; "2. Overview",
# "IV. Results" are level 1
_DOTTED_RE = re.compile(r"^\s*(?:§\s*(\d+(?:\.\d+){0,3})|(\d+(?:\.\d+){1,3})\.?)(?:\s+|(?=[一-鿿]))(\S.*)$")
_TIGHT_CHAPTER_RES: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(rf"^\s*第\s*{_CJK_NUM}\s*(?:章|部分|篇|卷){_TIGHT_CJK_TAIL}"), 1),
    (re.compile(rf"^\s*第\s*{_CJK_NUM}\s*(?:节|節){_TIGHT_CJK_TAIL}"), 2),
)
# Zero-padded numbers like "01." and "06." are step lists, not headings
_NUMBERED_RE = re.compile(r"^\s*(?:[1-9]\d*|[IVX]{1,5})\.\s+(\S.*)$")
# "1-3 SQL overview", "4-2 Updating data": dash-separated section numbers (common in Japanese technical
# books). Both parts are 1-2 digit numbers followed by the title text; "2020-2021 fiscal year" has 4-digit
# years and does not count, and "1-5 V" is blocked by the unit rule below
_DASHED_RE = re.compile(r"^\s*(\d{1,2})-(\d{1,2})(?:\s+|(?=[一-鿿]))(\S.*)$")
# "2 Overview", "4 Technical requirements": integer chapter numbers followed by a space (national standards,
# industry specifications and papers are written this way). Phrases that open with a quantity ("3 package
# types", "8 bit microcontrollers") look exactly the same, so this stays out of infer_heading_level: it is only
# accepted when the parser already marked the line as a heading and the chapter number continues from what
# came before, see HeadingResolver
_INTEGER_RE = re.compile(r"^\s*(\d{1,2})\s+(\S.*)$")
CHAPTER_GAP = 2       # how far an integer chapter number may run ahead of the previous chapter: 2 = one unmarked chapter in between still connects
_LEADING_NUMBER_RE = re.compile(r"^\s*§?\s*(\d{1,3})(?!\d)")
# Typographic markup in headings: markdown bold / underline / inline code / strikethrough, Obsidian highlight
# "==📅 2026-05-27=="
_MARKUP_RE = re.compile(r"\*\*|__|`|~~")
_HIGHLIGHT_RE = re.compile(r"==([^=\n]*)==")   # Obsidian highlight: ==📅 2026-01-15== → 📅 2026-01-15
# List-item shapes: "- default rules", "• custom limits", "\- user rules" -- MinerU marks bold list items as
# headings
_LIST_ITEM_RE = re.compile(r"^\s*\\?[-*•·●▪◦]\s+\S")
# Note / remark labels ("Note:" and its Chinese equivalents); when a line holds only the label, the body is
# usually elsewhere (or was dropped by the parser)
_NOTE_LABEL_RE = re.compile(r"^\s*(?:注释|注|备注|備考|Notes?|NOTE)\s*[:：]?\s*$", re.IGNORECASE)
_NOTE_PREFIX_RE = re.compile(r"^\s*(?:注释|注|备注|備考|Notes?|NOTE)\s*[:：]", re.IGNORECASE)
# A 1-2 letter unit (V / mA / mm / kΩ) right after the number -- that is a value, not a heading
_UNIT_AFTER_NUMBER_RE = re.compile(r"^[A-Za-zΩµ°%]{1,2}(?:\s|$)")
_SENTENCE_PUNCT_RE = re.compile(r"[，。；！？,;!?]")
_LEADER_RE = re.compile(r"\.{3,}|…{2,}|(?:\.\s){3,}")   # TOC / parameter-table dot leaders: "Support....18"
HEADING_MAX_CHARS = 60

# Prose blocks that are nothing but a page number: "Page 3 of 12", "Seite 4 von 9", the Chinese "page 5"
# form, "- 12 -", "3 / 20"
PAGE_FOOTER_RE = re.compile(
    r"^\s*(?:(?:Seite|Page|页码?)\s*\d+(?:\s*(?:von|of|/)\s*\d+)?"
    r"|第\s*\d+\s*页(?:\s*[/,,]\s*(?:共\s*)?\d+\s*页)?"
    r"|\d+\s+/\s+\d+|-\s*\d+\s*-)\s*$",
    re.IGNORECASE,
)


def resolve_heading_level(parser_level: int | None, text: str) -> tuple[int | None, bool]:
    """Combine the parser's level with layout inference into the final level; returns (level, recovered by
    inference).

    MinerU often marks every heading of a whole book as level 1 ("§3.3" and "5.4.3" side by side); the
    depth carried by the number is more trustworthy, so a deeper inferred level wins. Only a heading the
    parser did not mark but inference recognized counts as "recovered".
    """
    inferred = infer_heading_level(text)
    if parser_level is None:
        return inferred, inferred is not None
    if inferred is not None and inferred > parser_level:
        return inferred, False
    return parser_level, False


def infer_heading_level(text: str) -> int | None:
    """Infer the heading level from the text itself; returns None when it is not a heading.

    Chinese chapter / part / volume → 1, Chinese section → 2, Chapter/Part → 1, Section → 2,
    dotted numbers "1.2.3 xxx" → number of segments, "2. xxx" / "IV. xxx" → 1.
    """
    if "\n" in str(text or "").strip():
        return None                    # multi-line is not a heading ("06.\nSet the port" is step + body)
    line = (text or "").strip()
    if not line or "\n" in line or len(line) > HEADING_MAX_CHARS:
        return None
    for pattern, level in _CHAPTER_RES:
        if pattern.match(line):
            return level
    if len(line) <= TIGHT_HEADING_MAX_CHARS and not _SENTENCE_PUNCT_RE.search(line):
        for pattern, level in _TIGHT_CHAPTER_RES:
            if pattern.match(line):
                return level
    m = _DOTTED_RE.match(line)
    if m:
        number, rest = (m.group(1) or m.group(2)), m.group(3)
        if _UNIT_AFTER_NUMBER_RE.match(rest) or _SENTENCE_PUNCT_RE.search(rest):
            return None
        return min(6, number.count(".") + 1)
    m = _NUMBERED_RE.match(line)
    if m:
        rest = m.group(1)
        if _UNIT_AFTER_NUMBER_RE.match(rest) or _SENTENCE_PUNCT_RE.search(rest) or rest.endswith("."):
            return None
        return 1
    m = _DASHED_RE.match(line)
    if m:
        rest = m.group(3)
        if _UNIT_AFTER_NUMBER_RE.match(rest) or _SENTENCE_PUNCT_RE.search(rest):
            return None
        return 2
    return None


def clean_heading_text(text: str) -> str:
    """Strip typographic markup (bold, inline code, highlighted date markers) from a heading / section name,
    collapse whitespace, and trim leading / trailing colons and dashes. Section paths, chunk TITLEs and
    heading detection all use it, so "**2.1 Windows installation**" and "2.1 Windows installation" are
    the same heading."""
    out = _MARKUP_RE.sub("", _HIGHLIGHT_RE.sub(r"\1", str(text or "")))
    return re.sub(r"\s+", " ", out).strip(" \t-—:：")


def looks_like_list_item(text: str) -> bool:
    return bool(_LIST_ITEM_RE.match(str(text or "")))


_STEP_LABEL_RE = re.compile(r"^\s*0\d+\s*[.、)）]")


def looks_like_step_label(text: str) -> bool:
    """Zero-padded step numbers such as "01." and "06. Set the port" are procedure steps, not sections."""
    return bool(_STEP_LABEL_RE.match(str(text or "")))


def detect_toc_pages(entries: Iterable[tuple[int | None, str]], *, min_short_lines: int = 20,
                     short_ratio: float = 0.9, min_leaders: int = 8) -> set[int]:
    """Table-of-contents pages: the whole page is short lines (measured: a book's TOC pages have 32 lines
    each, all ≤30 characters), or ≥ min_leaders lines on the page carry dot-leader page numbers ("Pin
    configuration....4"). Numbered lines on TOC pages are not treated as headings, otherwise the section
    path gets polluted by the TOC (measured: the "Copyright notice" on body page 10 ended up under "9-3
    Connecting to PostgreSQL via Java"). entries = a sequence of (page_idx, text), body / heading items
    only."""
    per_page: dict[int, list[str]] = {}
    for page, text in entries:
        if page is None:
            continue
        line = str(text or "").strip()
        if line:
            per_page.setdefault(int(page), []).append(line)
    pages: set[int] = set()
    for page, lines in per_page.items():
        short = sum(1 for l in lines if "\n" not in l and len(l) <= 30)
        leaders = sum(1 for l in lines if _LEADER_RE.search(l))
        if leaders >= min_leaders or (short >= min_short_lines and short >= short_ratio * len(lines)):
            pages.add(page)
    return pages


def is_note_label(text: str) -> bool:
    """The whole block is just a label such as "Note:", with no body."""
    return bool(_NOTE_LABEL_RE.match(str(text or "").strip()))


def starts_with_note(text: str) -> bool:
    return bool(_NOTE_PREFIX_RE.match(str(text or "")))


def is_short_lead_in(text: str) -> bool:
    """One short line without sentence punctuation: can serve as a lead-in for a figure / table ("Three access
    methods", "Case notes"), not a paragraph."""
    line = str(text or "").strip()
    return (bool(line) and "\n" not in line and len(line) <= HEADING_MAX_CHARS
            and not _SENTENCE_PUNCT_RE.search(line) and not _LEADER_RE.search(line))


class HeadingResolver:
    """Parser-marked heading level + layout inference + document-level correction, combined into the final
    level.

    On PDFs MinerU marks every heading as level 1 and also marks bold list items and sidebar labels
    (KEYWORD / Note: / Key points) as headings. Three corrections: list-item shapes are not headings; a
    text that appears as a heading ≥ repeat_limit times in the document is a label and is demoted to
    body text; when a numbered heading precedes, an unnumbered heading counts as its next level (in a
    book, "Standard SQL" is a subsection of "1-3 SQL overview", not a sibling chapter). An integer chapter
    number such as "4 Technical requirements" that continues from what came before is a numbered heading
    too and is not demoted by the third rule.

    trust_parser_levels (docx): the levels come from Word styles, so the third rule is skipped -- otherwise,
    once an unstyled body paragraph shaped like "2.3 xxx" is recovered as a heading, every level-1 heading
    after it would be demoted.
    """

    def __init__(self, title_texts: Iterable[str] = (), *, repeat_limit: int = 3,
                 trust_parser_levels: bool = False) -> None:
        counts = Counter(clean_heading_text(t) for t in title_texts if clean_heading_text(t))
        self.labels = {t for t, n in counts.items() if n >= repeat_limit}
        self.trust_parser_levels = trust_parser_levels
        self.numbered_level: int | None = None
        self.chapter: int | None = None      # chapter number of the latest numbered heading ("3.1 Definitions" -> 3)
        self.pending: int | None = None      # the last integer chapter number that did not connect

    def _continues_chapters(self, number: int) -> bool:
        """An integer chapter number continues from what came before: it follows the previous chapter (one
        unmarked chapter in between still counts); or it follows the last integer chapter number that did not
        connect -- the break was earlier, and the sequence picks up again from here; or the document has no
        numbered heading yet and the numbering starts at 0 / 1."""
        if self.chapter is not None and self.chapter < number <= self.chapter + CHAPTER_GAP:
            return True
        if self.pending is not None and number == self.pending + 1:
            return True
        return number <= 1 and self.numbered_level is None

    def resolve(self, parser_level: int | None, text: str) -> tuple[int | None, bool]:
        clean = clean_heading_text(text)
        if parser_level is not None and (
            not clean or clean in self.labels or looks_like_list_item(text) or looks_like_step_label(text)
            or is_note_label(clean) or len(clean) > HEADING_MAX_CHARS or "\n" in str(text or "").strip()
        ):
            parser_level = None          # label / list item / note label / whole block mis-marked: not a heading
        level, inferred = resolve_heading_level(parser_level, text)
        if level is None:
            return None, False
        numbered = infer_heading_level(text) is not None
        integer = None if numbered or parser_level is None else _INTEGER_RE.match(clean)
        if integer and not _UNIT_AFTER_NUMBER_RE.fullmatch(integer.group(2)):      # "5 V" is a value
            numbered = self._continues_chapters(int(integer.group(1)))
            self.pending = None if numbered else int(integer.group(1))
        elif numbered:
            self.pending = None
        if numbered:
            self.numbered_level = level    # a numbered heading: later unnumbered headings hang under it
            number = _LEADING_NUMBER_RE.match(clean)
            if number:
                self.chapter = int(number.group(1))
        elif (parser_level is not None and not self.trust_parser_levels
                and self.numbered_level is not None and level <= self.numbered_level):
            level = self.numbered_level + 1
        return level, inferred


def is_page_footer(text: str) -> bool:
    return bool(PAGE_FOOTER_RE.match((text or "").strip()))
