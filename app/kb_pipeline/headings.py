"""Layout inference: headings MinerU failed to mark, and footers that are nothing but a page number.

A leaf module depending only on re: both the parsers (parsers/common.py) and the chunker
(chunking/chunker.py) need it, and either importing the other would create a cycle.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Iterable

# Headings MinerU cannot recognize are recovered from layout conventions (borrowing the regex set from
# WeKnora's patterns.go, tightened for our corpus). Only whole short lines are considered: anything that is
# not a heading -- values like "1.5 V typical", numbered list items like "1. Typical values are for
# reference only, ..." -- must be kept out, otherwise section_path gets polluted and the chunker's
# structural cut points land in the wrong places.
# The parser uses these rules to set the level of a block it has already marked as a heading (see
# HeadingResolver). An ordinary paragraph becomes a heading in one case only: it is a chapter heading of the
# document that the parser failed to mark (see recoverable_chapter_lines); numbered lines never are. The chunker
# uses the rules to find structural cut points inside body text.
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
    return None


def _is_chapter_word_heading(text: str) -> bool:
    """Whether a single heading line opens with an explicit chapter/part/section/appendix word."""
    if "\n" in str(text or "").strip():
        return False
    line = (text or "").strip()
    if not line or len(line) > HEADING_MAX_CHARS:
        return False
    if any(pattern.match(line) for pattern, _ in _CHAPTER_RES):
        return True
    if len(line) <= TIGHT_HEADING_MAX_CHARS and not _SENTENCE_PUNCT_RE.search(line):
        return any(pattern.match(line) for pattern, _ in _TIGHT_CHAPTER_RES)
    return False


# ── Chapter headings the parser did not mark ─────────────────────────────────────────────────────────────────────
# What such a heading must look like. Narrower than _CHAPTER_RES, which only sets the level of a heading the parser
# has marked: the number is followed by a space, a colon or a dash or ends the line ("Section 2.1 Scope", "附录A.1"
# carry a compound number and are left alone); an English chapter word is written "Chapter" or "CHAPTER" and
# carries at most two digits ("Section 906" is a law, "Part 4410" a part number); an appendix is named by one
# capital letter or a number ("Appendix to the report" is the title of something else); and a volume is not a
# chapter ("第 35 卷 第 4 期" at the head of a journal paper names the publication, not a part of the file).
_RECOVER_RES: tuple[tuple[re.Pattern[str], int, bool], ...] = (       # pattern, level, written in Latin script
    (re.compile(rf"^第\s*(?P<n>{_CJK_NUM})\s*(?P<w>章|部分|篇)(?=\s|[:：、—–\-]|$)"), 1, False),
    (re.compile(rf"^第\s*(?P<n>{_CJK_NUM})\s*(?P<w>节|節)(?=\s|[:：、—–\-]|$)"), 2, False),
    (re.compile(r"^(?P<w>Chapter|CHAPTER|Part|PART)\s+(?P<n>\d{1,2}|[IVX]{1,5})(?=\s|[:：—–]|$)"), 1, True),
    (re.compile(r"^(?P<w>Section|SECTION)\s+(?P<n>\d{1,2}|[IVX]{1,5})(?=\s|[:：—–]|$)"), 2, True),
    (re.compile(r"^(?P<w>附录)\s*(?P<n>[A-Z]|\d{1,2}|[一二三四五六七八九十]{1,3})(?=\s|[:：—–]|$)"), 1, False),
    (re.compile(r"^(?P<w>Appendix|APPENDIX)\s+(?P<n>[A-Z]|\d{1,2}|[IVX]{1,5})(?=\s|[:：—–]|$)"), 1, True),
)
# Any line that opens with a chapter word, however it goes on: a heading, a contents entry, a mention
_NAMED_RE = re.compile(
    rf"^\s*(?:第\s*(?P<cn>{_CJK_NUM})\s*(?P<cw>章|部分|篇|卷|节|節)"
    r"|(?P<lw>Chapter|Part|Section)\s+(?P<ln>\d+|[IVX]{1,5})(?![A-Za-z\d])"
    r"|(?P<aw>附录|Appendix)\s*(?P<an>\d{1,2}|[一二三四五六七八九十]{1,3}(?![一-鿿])|[IVX]{2,5}|[A-Z])(?![A-Za-z\d]))",
    re.IGNORECASE)
_CONTENTS_ENTRY_RE = re.compile(r"\s\d{1,4}\s*$")          # a contents entry ends with its page number: "第一章 总则 3"
_DOT_RUN_RE = re.compile(r"[·•‧∙．‥⋯]{2,}")                # leaders other than "...": "第一章 总则····3"
_CITED_PART_RE = re.compile(r"^of\b", re.IGNORECASE)       # "Section 8 of the Act", "Part 2 of 3"
_CLOCK_RE = re.compile(r"\d{1,2}[:：]\d{2}")               # "第一节 08:00-09:30 开幕致辞": an entry of a timetable
_LETTER_RE = re.compile(r"[^\W\d_]")
_CJK_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CJK_UNITS = {"十": 10, "百": 100, "千": 1000}
_ROMAN = {"I": 1, "V": 5, "X": 10}


def _cjk_number(text: str) -> int | None:
    """ "十二" -> 12, "一百零五" -> 105, "二〇二四" -> 2024; None when the text is not a Chinese numeral."""
    if not text or any(ch not in _CJK_DIGITS and ch not in _CJK_UNITS for ch in text):
        return None
    if not any(ch in _CJK_UNITS for ch in text):
        return int("".join(str(_CJK_DIGITS[ch]) for ch in text))
    total = digit = 0
    for ch in text:
        if ch in _CJK_DIGITS:
            digit = _CJK_DIGITS[ch]
        else:
            total += (digit or 1) * _CJK_UNITS[ch]
            digit = 0
    return total + digit


def _roman_number(text: str) -> int:
    total = 0
    for ch, following in zip(text, text[1:] + " "):
        value = _ROMAN[ch]
        total += -value if _ROMAN.get(following, 0) > value else value
    return total


def _chapter_key(word: str, number: str) -> tuple[str, str]:
    """Which chapter a chapter word and its number name, however the number is written: ("章", "3") for "第 3 章",
    "第三章" and "第03章", ("chapter", "4") for "CHAPTER 04" and "Chapter IV", ("appendix", "A")."""
    kind = "节" if word == "節" else word.casefold()
    text = number.strip()
    if text.isdigit():
        return kind, str(int(text))
    value = _cjk_number(text)
    if value is not None:
        return kind, str(value)
    if re.fullmatch(r"[IVXivx]{1,5}", text) and not (kind in ("附录", "appendix") and len(text) == 1):
        return kind, str(_roman_number(text.upper()))
    return kind, text.upper()


def _named_chapter(line: str, marked: bool = False) -> tuple[str, str] | None:
    """The chapter a single line names by the way it opens, None when it names none. A contents entry ("第一章 总则
    …… 3") names one too, and so does a heading the parser marked that is written without a separator ("第一章总则");
    an unmarked line written that way is a sentence ("第3章的说明请参见附录")."""
    match = _NAMED_RE.match(line)
    if not match:
        return None
    if match.group("cw"):
        after = line[match.end():]
        if after and not re.match(r"\s|[:：.．、—–\-·…]", after) and not (marked and re.match(r"[一-鿿]", after)):
            return None
        return _chapter_key(match.group("cw"), match.group("cn"))
    if match.group("lw"):
        return _chapter_key(match.group("lw"), match.group("ln"))
    return _chapter_key(match.group("aw"), match.group("an"))


def chapter_word_level(text: str) -> int | None:
    """The level of a line that reads like a chapter title on a line of its own: "第 3 章 安装", "第二节 范围",
    "Chapter 4: Timing", "Part II", "Section 2 Scope", "附录 A 术语表". None for everything else.

    This wording alone does not make an unmarked line a heading (see recoverable_chapter_lines), but nothing else
    can: so it is narrower than infer_heading_level. Besides what _RECOVER_RES asks of the chapter word, what
    follows it must be a title: not a sentence ("第 3 章 介绍了安装方法，见下文", "Chapter 3 describes the setup."),
    not a contents entry ("第一章 总则 3", "第一章 总则····3"), not a citation ("Section 8 of the Act"), not another
    chapter word ("第一章 第一节 概述"), not a time of day ("第一节 08:00-09:30 开幕致辞" is an entry of a
    timetable), and after an English chapter word it starts with a capital or a digit
    ("Section 3 applies only to members" is a sentence). Emphasis marks around the line do not count ("**第一章 总则**")."""
    raw = str(text or "").strip()
    line = clean_heading_text(raw)
    if not line or "\n" in raw or len(line) > HEADING_MAX_CHARS:
        return None
    for pattern, level, latin in _RECOVER_RES:
        match = pattern.match(line)
        if not match:
            continue
        rest = line[match.end():]
        title = rest.lstrip(" \t　:：、—–-")
        if (_SENTENCE_PUNCT_RE.search(title) or title.endswith((".", "。")) or _LEADER_RE.search(rest)
                or _DOT_RUN_RE.search(rest) or _CONTENTS_ENTRY_RE.search(rest) or _NAMED_RE.match(title)
                or _CLOCK_RE.search(title) or (latin and (_CITED_PART_RE.match(title) or title[:1].islower()))):
            return None
        return level
    return None


def _one_sequence(numbers: list[str]) -> bool:
    """Chapter numbers as _chapter_key writes them count on without a gap: "4" "5" "6", or "A" "B" "C"."""
    if all(n.isdigit() for n in numbers):
        values = [int(n) for n in numbers]
    elif all(len(n) == 1 and n.isalpha() for n in numbers):
        values = [ord(n) for n in numbers]
    else:
        return False
    return all(b == a + 1 for a, b in zip(values, values[1:]))


# An enumerator at the start of a short line: "二、培训内容", "（三）范围", "2.2 投标报价", "3. 提交", "(4)", "2 Installation"
_LIST_NUMBER_RES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("cjk", re.compile(r"^([一二三四五六七八九十百]+)\s*[、.．]")),
    ("(cjk)", re.compile(r"^[（(]\s*([一二三四五六七八九十百]+)\s*[)）]")),
    ("dotted", re.compile(r"^§?\s*(\d+(?:\.\d+){1,3})\.?(?=\s|[一-鿿])")),
    ("arabic", re.compile(r"^(\d{1,3})\s*[.、．)）](?!\d)")),
    ("(arabic)", re.compile(r"^[（(]\s*(\d{1,3})\s*[)）]")),
    ("integer", re.compile(r"^(\d{1,2})\s+(?=\S)")),
)
_LOWER_CHAPTER_WORDS = frozenset({"节", "section"})        # chapter words of level 2; all others are level 1


def _list_number(line: str) -> tuple[tuple[str, str], int] | None:
    """The place of a line in a numbered list, by the enumerator it opens with: (the list it belongs to, its number).
    "二、培训内容" -> (("cjk", ""), 2), "2.2 投标报价" -> (("dotted", "2"), 2); None for a line without one."""
    for style, pattern in _LIST_NUMBER_RES:
        match = pattern.match(line)
        if not match:
            continue
        text = match.group(1)
        if style == "dotted":
            parent, _, last = text.rpartition(".")
            return (style, parent), int(last)
        number = int(text) if text.isdigit() else _cjk_number(text)
        return None if number is None else ((style, ""), number)
    return None


def recoverable_chapter_lines(entries: Iterable[tuple[Any, str | None, bool]],
                              contents: Iterable[Any] = ()) -> dict[Any, int]:
    """The lines the parser did not mark that are chapter headings of this document: {entry id: heading level}.

    entries: the document in reading order as (id, text, marked as a heading by the parser). text None stands for
    content that is not a line of text: a table, a picture, a formula, a code listing. contents: the ids of lines
    known to stand on a contents page; they are neither recovered nor taken for headings.

    Recognisers miss a chapter heading now and then, and a Word file whose chapters were typed without heading
    styles has no marked headings at all. A line is recovered when all of this holds:
      1. it reads like a chapter title on a line of its own (chapter_word_level);
      2. it is not an entry of a list: between it and the nearest line with the same chapter word, before it and
         after it, stands body text or other content. A contents list, a list of enclosures ("附录 A …", "附录 B
         …") and a summary of the chapters ("第一章 介绍背景", "第二章 介绍方法") are runs of such lines with
         nothing but headings and page numbers between them. And the lines to be recovered are no list with a
         note under each entry either: an entry of such a list is followed by a single paragraph and then by the
         next unmarked line with the same chapter word ("第一部分 商务部分", one sentence, "第二部分 技术部分",
         one sentence). When every line to be recovered but the last is such an entry, or the last one is, none
         is recovered. Overviews, the parts of a tender or of a device and annotated contents are written this
         way, and the last entry of such a list took everything that came after it;
      3. no heading the parser marked names the same chapter: then the line is a contents entry, a running head
         or a mention of it;
      4. the chapters of its kind form one sequence. The headings of that chapter word which the parser marked and
         the lines to be recovered, taken together in reading order, are at least two and count on without a gap
         (3, 4, 5; A, B, C). And when the last of them is a recovered line, nothing after it names the chapter that
         would come next without being recovered itself: its text would be filed under the chapter before. Such a
         line reads like a contents entry ("第三章 展望 2030"), is a heading broken over two lines or written
         without a separator, stands on a page taken for contents, or was taken for the entry of a list because
         nothing but numbered lines stands below it. Only an entry of a list of the chapters of this very
         sequence (an index at the end) is not meant. Otherwise no
         line of that chapter word is recovered: when an attachment starts again at "第一章", when a running head
         comes back page after page, when a preface describes later chapters, when the heading of one chapter
         cannot be read, some of those lines are headings and some are not, or one is missing, and recovering the
         rest would file whole stretches of the document under the wrong chapter. A lone "第 2 部分：试验方法" on
         the cover of a standard is no sequence;
      5. the document does not write this chapter word without a separator as well ("第三章配置" on a short line
         the parser did not mark): then that is how it writes its headings, and the separated line is a mention;
      6. the headings the parser marked leave room for it. A line that stands before the last heading the parser
         marked with the same chapter word fills a place among those headings ("第二章" between a marked "第一章"
         and a marked "第三章"). The lines after that heading -- all of them, when the parser marked no heading
         with this chapter word -- are recovered only when no heading the parser marked stands after the first of
         them: then they are all the structure that part of the document has (a Word file typed without heading
         styles). Where the parser did mark the headings that follow, a run of "第一部分 …", "第二部分 …" lines
         with a sentence under each is a list inside one of those sections -- the parts of a tender, the parts of
         a device, the chapters of the next volume -- and as level-1 headings such lines took the sections after
         them away from their chapter;
      7. those lines after the last marked heading are not held between two neighbours of one numbered list:
         "二、培训内容" before them and "三、培训时间" after them, or "2.1" and "2.2". Then they divide one item of
         that list, and the items after it are not theirs;
      8. a section ("第 N 节", "Section N") is recovered only when every line that names a chapter of the level
         above is a heading, marked or recovered, and the document does not write that level without a separator
         ("第一章总则"). Sections recovered below chapters that are not headings would take the chapters after
         them.
    A line that fails any of them stays body text, as every unmarked line did before. Numbered steps and values
    ("1. Switch the device on", "2.4 GHz band") are never recovered: they read like headings, and as level-1
    headings they wiped the section path before them."""
    known = set(contents)
    rows: list[tuple[Any, tuple[str, str] | None, int | None, bool]] = []      # id, chapter named, level if it reads like a title, is content
    marked_keys: set[tuple[str, str]] = set()
    sequence: dict[str, list[tuple[int, str]]] = defaultdict(list)             # chapter word -> (position, number) of its marked headings
    written_tight: set[str] = set()
    unread: dict[int, tuple[str, str]] = {}       # unmarked lines that name a chapter without reading like its title
                                                  # (written without a separator, too)
    naming: dict[int, tuple[str, str]] = {}       # every unmarked line that names a chapter, lines on contents pages too
    numbered: list[tuple[int, tuple[str, str], int]] = []      # short lines that open with an enumerator: position, list, number
    last_marked = -1                              # position of the last heading the parser marked
    marked_rows: set[int] = set()
    for ident, text, marked in entries:
        if text is None:
            rows.append((ident, None, None, True))
            continue
        if marked:
            marked_rows.add(len(rows))
        if marked and ident not in known:
            last_marked = len(rows)
        raw = str(text).strip()
        line = clean_heading_text(raw)
        single = bool(line) and "\n" not in raw and len(line) <= HEADING_MAX_CHARS
        key = _named_chapter(line, marked) if single else None
        if key is not None and marked and ident not in known:
            marked_keys.add(key)
            sequence[key[0]].append((len(rows), key[1]))
        elif key is None and single and not marked and len(line) <= TIGHT_HEADING_MAX_CHARS and not _SENTENCE_PUNCT_RE.search(line):
            tight = _named_chapter(line, True)
            if tight is not None:
                written_tight.add(tight[0])
        level = chapter_word_level(raw) if key is not None and not marked and ident not in known else None
        if not marked:
            opening = key
            if key is None and not single and line:      # several lines, or a long one: what its first line names
                opening = _named_chapter(clean_heading_text(raw.split("\n", 1)[0])[:HEADING_MAX_CHARS])
            elif key is None and single and not _SENTENCE_PUNCT_RE.search(line):
                opening = _named_chapter(line, True)     # written without a separator: "第三章关于未来五年的安排"
            if opening is not None:
                naming[len(rows)] = opening
                if ident not in known and level is None:
                    unread[len(rows)] = opening
        if key is None and single and ident not in known and not _SENTENCE_PUNCT_RE.search(line):
            place = _list_number(line)
            if place is not None:
                numbered.append((len(rows), *place))
        heading_like = marked or key is not None or (single and infer_heading_level(line) is not None)
        rows.append((ident, key, level, not heading_like and bool(_LETTER_RE.search(line)) and not is_page_footer(line)))

    def list_of(index: int) -> set[tuple[str, str]]:
        """The chapters named by the lines that stand in one list with this line: lines with the same chapter word
        before and after it, with nothing but headings and page numbers between them."""
        kind = rows[index][1][0]
        found: set[tuple[str, str]] = set()
        for step in (1, -1):
            other = index + step
            while 0 <= other < len(rows) and not rows[other][3]:
                key = rows[other][1]
                if key is not None and key[0] == kind:
                    found.add(key)
                other += step
        return found

    def listed(index: int) -> bool:
        return bool(list_of(index))

    def note_entry(index: int) -> bool:
        """The line is followed by at most one item and then by an unmarked line with the same chapter word."""
        kind = rows[index][1][0]
        for other in range(index + 1, min(index + 3, len(rows))):
            key = rows[other][1]
            if key is not None and key[0] == kind:
                return other not in marked_rows
        return False

    def held(first: int, last: int) -> bool:
        """The lines first..last stand between two neighbours of one numbered list (rule 7)."""
        before = {group: number for position, group, number in numbered if position < first}       # the last of each list
        after: dict[tuple[str, str], int] = {}
        for position, group, number in numbered:
            if position > last:
                after.setdefault(group, number)                                                      # the first of each list
        return any(group in before and number == before[group] + 1 for group, number in after.items())

    lines: dict[str, list[int]] = defaultdict(list)            # chapter word -> its lines that pass 1 to 3, in reading order
    for i, row in enumerate(rows):
        if row[2] is not None and row[1] not in marked_keys and not listed(i):
            lines[row[1][0]].append(i)

    def recovered(kind: str) -> list[int]:
        """The lines of one chapter word that pass 4 to 7."""
        if kind in written_tight:
            return []
        last_of_kind = max((position for position, _ in sequence[kind]), default=-1)
        among = [i for i in lines[kind] if i < last_of_kind]
        beyond = [i for i in lines[kind] if i > last_of_kind]
        if beyond and (last_marked > beyond[0] or held(beyond[0], beyond[-1])):
            beyond = []
        chosen = among + beyond
        places = sorted(sequence[kind] + [(i, rows[i][1][1]) for i in chosen])
        if not chosen or len(places) < 2 or not _one_sequence([number for _, number in places]):
            return []
        entries = [note_entry(i) for i in chosen]
        if entries[-1] or (len(chosen) > 1 and all(entries[:-1])):
            return []                              # a list with a note under each entry (rule 2)
        end, number = places[-1]
        if end in chosen:
            following = (kind, str(int(number) + 1) if number.isdigit() else chr(ord(number) + 1))
            own = {(kind, number) for _, number in places}
            if any(key == following and position > end and (rows[position][1] is None or not list_of(position) & own)
                   for position, key in naming.items()):
                return []
        return chosen

    result = {kind: recovered(kind) for kind in lines}
    # rule 8: a chapter of the level above that is named by a line which is no heading
    upper_open = (any(set(result[kind]) != set(lines[kind]) for kind in lines if kind not in _LOWER_CHAPTER_WORDS)
                  or any(key[0] not in _LOWER_CHAPTER_WORDS and (rows[position][1] is None or not listed(position))
                         for position, key in unread.items()))
    return {rows[i][0]: rows[i][2] for kind, chosen in result.items() for i in chosen
            if not (upper_open and kind in _LOWER_CHAPTER_WORDS)}


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


_TOC_TITLE_CN = r"(?:(?:总|简明|详细|图|表)\s*)?目\s*[录次]"
_TOC_TITLE_EN = r"(?:(?:table\s+of\s+)?contents(?:\s*\(cont(?:\.|inued)?\))?|list\s+of\s+(?:figures|tables))"
_TOC_TITLE_RE = re.compile(
    rf"^(?:{_TOC_TITLE_CN}|{_TOC_TITLE_EN}"
    rf"|{_TOC_TITLE_CN}\s*[/|·\s]+\s*{_TOC_TITLE_EN}"
    rf"|{_TOC_TITLE_EN}\s*[/|·\s]+\s*{_TOC_TITLE_CN})$",
    re.IGNORECASE,
)
_TOC_PAGE_NUM_RE = re.compile(r"^(.*?)(?:\s+|[\.·…]{2,})(?:([1-9]\d{0,2})-([1-9]\d{0,3})|([1-9]\d{0,3}))\s*$")
_TOC_BARE_PAGE_NUM_RE = re.compile(r"^(?:([1-9]\d{0,2})-([1-9]\d{0,3})|([1-9]\d{0,3}))$")


def _parse_toc_page_key(match: re.Match[str], offset: int) -> tuple[int, int]:
    if match.group(offset):
        return int(match.group(offset)), int(match.group(offset + 1))
    return 0, int(match.group(offset + 2))


def _is_valid_toc_sequence(prefixes: list[str], nums: list[tuple[int, int]]) -> bool:
    if len(nums) < 2:
        return False
    distinct = len(set(prefixes))
    if distinct < 5 or distinct < 0.4 * len(nums):
        return False
    non_decreasing = sum(b >= a for a, b in zip(nums, nums[1:]))
    return non_decreasing >= 0.85 * (len(nums) - 1) and max(nums) > min(nums)


def _has_toc_page_numbers(lines: list[str], ratio: float) -> bool:
    """Whether a dense short-line page carries real printed TOC page numbers (inline or split-column, plain
    or chapter-relative like '1-1') rather than repeated-label series ('Item 1', 'Item 2') or arbitrary
    parameter-value lines ('Voltage 5', 'Temp 85', 'Pins 16')."""
    prefixes: list[str] = []
    nums: list[tuple[int, int]] = []
    for line in lines:
        m = _TOC_PAGE_NUM_RE.match(line)
        if not m:
            continue
        prefix = m.group(1).strip(" \t.·…")
        if not prefix:
            continue
        prefixes.append(prefix.casefold())
        nums.append(_parse_toc_page_key(m, 2))
    if len(nums) >= ratio * len(lines) and _is_valid_toc_sequence(prefixes, nums):
        return True
    # Split-column TOC fallback: MinerU splits leaderless two-column entries into alternating (title, page_num)
    split_prefixes: list[str] = []
    split_nums: list[tuple[int, int]] = []
    for idx in range(1, len(lines)):
        m = _TOC_BARE_PAGE_NUM_RE.match(lines[idx])
        if not m:
            continue
        prev = lines[idx - 1].strip(" \t.·…")
        if not prev or _TOC_BARE_PAGE_NUM_RE.match(prev):
            continue
        split_prefixes.append(prev.casefold())
        split_nums.append(_parse_toc_page_key(m, 1))
    return 2 * len(split_nums) >= ratio * len(lines) and _is_valid_toc_sequence(split_prefixes, split_nums)


def detect_toc_pages(entries: Iterable[tuple[int | None, str]], *, min_short_lines: int = 20,
                     short_ratio: float = 0.9, min_leaders: int = 8,
                     page_num_ratio: float = 0.6) -> set[int]:
    """Table-of-contents pages: >= min_leaders lines carrying dot-leader page numbers ("Pin configuration....4"),
    or a dense short-line page (>= min_short_lines lines <= 30 chars at >= short_ratio) backed by structural TOC
    evidence -- either an explicit contents heading on this or the immediately preceding page, or at least
    page_num_ratio of lines ending with a printed page number. Short-line density alone misclassifies spec
    lists, bullet slides and glossary pages."""
    per_page: dict[int, list[str]] = {}
    for page, text in entries:
        if page is None:
            continue
        line = str(text or "").strip()
        if line:
            per_page.setdefault(int(page), []).append(line)
    toc_title_pages = {
        page for page, lines in per_page.items()
        if any(_TOC_TITLE_RE.match(clean_heading_text(line)) for line in lines[:5] + lines[-2:])
    }
    title_chain: set[int] = set()
    pages: set[int] = set()
    for page in sorted(per_page):
        lines = per_page[page]
        short = sum(1 for l in lines if "\n" not in l and len(l) <= 30)
        leaders = sum(1 for l in lines if _LEADER_RE.search(l))
        if leaders >= min_leaders:
            pages.add(page)
            continue
        dense_short = short >= short_ratio * len(lines)
        if page in toc_title_pages and dense_short:
            title_chain.add(page)
        if short >= min_short_lines and dense_short:
            has_title = page in title_chain or (page - 1) in title_chain
            if has_title or _has_toc_page_numbers(lines, page_num_ratio):
                pages.add(page)
                if has_title:
                    title_chain.add(page)
    return pages


def explicit_toc_entries(entries: list[tuple[int | None, str]]) -> set[int]:
    """Recognize short contents lists without making nearby prose a TOC.

    Mixed pages require printed page-number leaders. Contents-only pages may
    use numbered entries or a sequential chapter-word list without page numbers.
    The explicit title and list structure are both required; a short-line count
    alone is not sufficient.
    """
    pages: dict[int, list[tuple[int, str]]] = {}
    for index, (page, text) in enumerate(entries):
        if page is not None and text.strip():
            pages.setdefault(page, []).append((index, text.strip()))
    found: set[int] = set()

    def title(text):
        return bool(_TOC_TITLE_RE.match(clean_heading_text(text)))

    def lines(text):
        return [line.strip() for line in text.splitlines() if line.strip()]

    def numbered(line):
        return len(line) <= HEADING_MAX_CHARS and bool(re.match(r"^\d+(?:[.\-]\d+)*[.、．]?\s+\S", line))

    def chapter_toc(cand_lines: list[str]) -> bool:
        if len(cand_lines) < 2:
            return False
        if any(line == "\ufffc" or len(line) > HEADING_MAX_CHARS
               or re.search(r"[。！？!?]", line) or line.endswith(".") for line in cand_lines):
            return False
        by_kind: dict[str, list[str]] = defaultdict(list)
        structured = 0
        for line in cand_lines:
            key = _named_chapter(clean_heading_text(line), marked=True)
            if key is not None:
                by_kind[key[0]].append(key[1])
                structured += 1
            elif _LEADER_RE.search(line) or numbered(line):
                structured += 1
        return (structured >= 0.4 * len(cand_lines)
                and any(len(nums) >= 2 and _one_sequence(nums) for nums in by_kind.values()))

    for items in pages.values():
        for offset, (index, text) in enumerate(items):
            if not title(text):
                continue
            whole_rest = [(i, t) for i, t in items if not title(t)]
            whole_lines = [line for _, t in whole_rest for line in lines(t)]
            if len(whole_lines) >= 2 and (
                all(_LEADER_RE.search(line) or numbered(line) for line in whole_lines)
                or chapter_toc(whole_lines)
            ):
                found.update(i for i, _ in items)
                break
            rest = [(i, t) for i, t in items[offset + 1:] if not title(t)]
            all_lines = [line for _, text in rest for line in lines(text)]
            if len(all_lines) >= 2 and (
                all(_LEADER_RE.search(line) or numbered(line) for line in all_lines)
                or chapter_toc(all_lines)
            ):
                found.update(i for i, _ in items[offset:])
                break
            run = []
            for i, t in rest:
                if not all(_LEADER_RE.search(line) for line in lines(t)):
                    break
                run.append((i, t))
            if sum(len(lines(t)) for _, t in run) >= 2:
                found.add(index)
                found.update(i for i, _ in run)
    return found


def is_note_label(text: str) -> bool:
    """The whole block is just a label such as "Note:", with no body."""
    return bool(_NOTE_LABEL_RE.match(str(text or "").strip()))


def starts_with_note(text: str) -> bool:
    return bool(_NOTE_PREFIX_RE.match(str(text or "")))


def is_mismarked_heading(text: str) -> bool:
    """Whether a parser-marked heading is clearly not a heading (empty, pure punctuation, list item,
    zero-padded step label, bare note label, multi-line, or longer than HEADING_MAX_CHARS)."""
    raw = str(text or "").strip()
    clean = clean_heading_text(raw)
    has_word_char = bool(_LETTER_RE.search(clean) or any(ch.isdigit() for ch in clean))
    return (
        not clean
        or not has_word_char
        or looks_like_list_item(raw)
        or looks_like_step_label(raw)
        or is_note_label(clean)
        or len(clean) > HEADING_MAX_CHARS
        or "\n" in raw
    )


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
    text that appears as a heading ≥ repeat_limit times in the document is a recurring title and goes one
    level below the nearest heading that does not recur; when a numbered heading precedes, an unnumbered
    heading counts as its next level (in a book, "Standard SQL" is a subsection of "1.3 SQL overview", not
    a sibling chapter). An integer chapter number such as "4 Technical requirements" that continues from
    what came before is a numbered heading too and is not demoted by the third rule.

    Recurring titles used to become body text, on the reasoning that they are sidebar labels. On two trial
    knowledge bases 561 of 2140 marked headings recurred, and every one that could be read was a real
    sub-section title ("Conclusion", "References" and "Key points" under each chapter of a book, the same
    six questions under each case of a casebook). As body text they were no cut point, and their sections
    were filed under whatever stood before them. Whether such a title is a sibling or a child of the
    heading before it cannot be told from its recurring: it is a sibling in the book and a child in
    another report. Placing it below that heading keeps every section path as it was and adds the title
    to it; placing it beside would be right for the book and would drop a correct parent in the report.
    A recurring title takes no part in the numbering, and it cannot replace a section above it.

    trust_parser_levels (docx): the levels come from Word styles, so the third rule is skipped -- otherwise,
    once an unstyled body paragraph shaped like "2.3 xxx" is recovered as a heading, every level-1 heading
    after it would be demoted. A recurring title keeps its style level there for the same reason.

    Layout inference only sets the level of a block the parser has already marked as a heading; resolve() does
    not promote an ordinary paragraph. Numbered steps and values ("1. Switch the device on", "2.4 GHz band") read
    like headings: once promoted to level 1 they wiped the section path before them, and every chunk after them
    was filed under the wrong section. The one kind of unmarked line that does become a heading is chosen by
    looking at the whole document (recoverable_chapter_lines) and handed over with recover().
    """

    def __init__(self, title_texts: Iterable[str] = (), *, repeat_limit: int = 3,
                 trust_parser_levels: bool = False, demoted_recurring: Iterable[str] = ()) -> None:
        counts = Counter(clean_heading_text(t) for t in title_texts if clean_heading_text(t))
        self.labels = {t for t, n in counts.items() if n >= repeat_limit}
        self.demoted_recurring = {clean_heading_text(t) for t in demoted_recurring if clean_heading_text(t)}
        self.trust_parser_levels = trust_parser_levels
        self.numbered_level: int | None = None
        self.unique_level: int | None = None  # level of the latest heading that does not recur
        self.chapter: int | None = None      # chapter number of the latest numbered heading ("3.1 Definitions" -> 3)
        self.pending: int | None = None      # the last integer chapter number that did not connect
        self.chapter_word_level: int | None = None  # level of active chapter-word heading ("第一章" -> 1, "第一节" -> 2)
        self.sub_chapter: int | None = None         # latest integer subsection number under chapter_word_level
        self.sub_pending: int | None = None         # unconnected integer subsection under chapter_word_level

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

    def _continues_sub_chapters(self, number: int) -> bool:
        """An integer subsection number under an active chapter-word heading ("第一章 / 1 目的 / 2 范围")."""
        if self.sub_chapter is not None:
            return self.sub_chapter < number <= self.sub_chapter + CHAPTER_GAP
        if self.sub_pending is not None and number == self.sub_pending + 1:
            return True
        return number <= 1 and self.numbered_level == self.chapter_word_level

    def resolve(self, parser_level: int | None, text: str) -> tuple[int | None, bool]:
        if parser_level is None:
            return None, False           # the parser did not mark it as a heading: body text, no promotion by wording
        clean = clean_heading_text(text)
        mis_marked = is_mismarked_heading(text)
        recurring = clean in self.labels and not self.trust_parser_levels
        if recurring and not mis_marked and infer_heading_level(text) is None:
            if clean in self.demoted_recurring:
                return None, False          # recurring non-section UI control / trailing label: keep as body text
            return (self.unique_level or 0) + 1, False      # below the nearest heading that does not recur
        if mis_marked or recurring:
            # list item / note label / whole block mis-marked: not a heading. A recurring title that carries a
            # number is left to that number, as before
            parser_level = None
        level, inferred = resolve_heading_level(parser_level, text)
        if level is None:
            return None, False
        numbered = infer_heading_level(text) is not None
        sub_integer = False
        integer = None if numbered or parser_level is None else _INTEGER_RE.match(clean)
        if integer and not _UNIT_AFTER_NUMBER_RE.fullmatch(integer.group(2)):      # "5 V" is a value
            n = int(integer.group(1))
            if self.chapter_word_level is not None and not self.trust_parser_levels:
                if self._continues_sub_chapters(n):
                    numbered = True
                    sub_integer = True
                    level = self.chapter_word_level + 1
                    self.sub_chapter = n
                    self.sub_pending = None
                elif self.sub_chapter is None and self.numbered_level == self.chapter_word_level:
                    self.sub_pending = n
            else:
                numbered = self._continues_chapters(n)
                self.pending = None if numbered else n
        elif numbered:
            self.pending = None
            self.sub_pending = None
            if _is_chapter_word_heading(clean):
                self.chapter_word_level = level
                self.sub_chapter = None
                self.chapter = None
            elif level == 1:
                self.chapter_word_level = None
                self.sub_chapter = None
            elif (self.chapter_word_level is not None and self.sub_chapter is not None
                    and not self.trust_parser_levels):
                # Dotted subsections under an active integer subsection ("第1部分 / 1 范围 / 1.1 适用"):
                # nest under the integer subsection rather than popping it at the same level.
                lead = _LEADING_NUMBER_RE.match(clean)
                if lead and int(lead.group(1)) == self.sub_chapter:
                    level = self.chapter_word_level + level
        if numbered:
            self.numbered_level = level    # a numbered heading: later unnumbered headings hang under it
            if not sub_integer:
                number = _LEADING_NUMBER_RE.match(clean)
                if number:
                    self.chapter = int(number.group(1))
        elif (parser_level is not None and not self.trust_parser_levels
                and self.numbered_level is not None and level <= self.numbered_level):
            level = self.numbered_level + 1
        self.unique_level = level
        return level, inferred

    def recover(self, level: int) -> tuple[int, bool]:
        """Take in a chapter heading the parser did not mark (chosen by recoverable_chapter_lines). The headings
        after it are placed as after a numbered heading of that level, which is what a marked chapter heading is;
        returns (level, True), True meaning recovered by wording."""
        self.chapter = None
        self.pending = None
        self.chapter_word_level = level
        self.sub_chapter = None
        self.sub_pending = None
        self.numbered_level = level
        self.unique_level = level
        return level, True


def is_page_footer(text: str) -> bool:
    return bool(PAGE_FOOTER_RE.match((text or "").strip()))
