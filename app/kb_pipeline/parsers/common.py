from __future__ import annotations

import ast
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable

from ..models import ParsedBlock
from ..utils import infer_doc_type
from ..vision.images import IMAGE_SUFFIXES


def decode_text_bytes(data: bytes) -> str:
    """utf-8 first (self-validating), then gb18030 (supersets gbk/gb2312),
    then lossy utf-8 as the last resort. The old errors="ignore" read deleted
    every multi-byte GBK sequence, silently stripping all CJK from legacy
    files; the csv reader always had this ladder -- text/markdown/html now
    share it."""
    # BOM first: gb18030 accepts almost any byte sequence, so a UTF-16 file (common for
    # .txt/.csv exported from Windows) would decode into mojibake and get stored as body text.
    if data[:2] in (b"\xff\xfe", b"\xfe\xff") or data[:4] in (b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff"):
        for encoding in ("utf-32", "utf-16"):
            try:
                return data.decode(encoding)
            except (UnicodeDecodeError, LookupError):
                pass
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            pass
    try:
        return data.decode("gb18030")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="ignore")


def read_text_smart(path: Path) -> str:
    return decode_text_bytes(path.read_bytes())


def clean_text(text: str) -> str:
    return "\n".join(
        line.rstrip()
        for line in (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    ).strip()


def walk_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_dicts(child)
    elif isinstance(value, list):
        for item in value:
            yield from walk_dicts(item)


def maybe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def page_idx(item: dict[str, Any]) -> int | None:
    for key in ("page_idx", "page_no", "page", "page_id"):
        value = maybe_int(item.get(key))
        if value is not None:
            return value + 1 if key in {"page_idx", "page_id"} else value
    prov = item.get("prov") or item.get("provenance")
    if isinstance(prov, list) and prov:
        first = prov[0]
        if isinstance(first, dict):
            value = maybe_int(first.get("page_no") or first.get("page"))
            if value is not None:
                return value
    if isinstance(prov, dict):
        value = maybe_int(prov.get("page_no") or prov.get("page"))
        if value is not None:
            return value
    return None


def bbox(item: dict[str, Any]) -> list[float] | None:
    for key in ("bbox", "box", "position"):
        value = item.get(key)
        if isinstance(value, list) and len(value) >= 4:
            try:
                return [float(x) for x in value[:4]]
            except (TypeError, ValueError):
                return None
    return None


def label_of(item: dict[str, Any]) -> str:
    raw = item.get("label") or item.get("type") or item.get("block_type") or item.get("category") or item.get("name") or ""
    if isinstance(raw, dict):
        raw = raw.get("name") or raw.get("value") or ""
    return str(raw).strip().lower()


_TABLE_TAG_RE = re.compile(r"<\s*(table|tr)\b", re.IGNORECASE)
# A value like colspan="10000" (hand-written or misrecognised) would inflate one row into tens of
# thousands of cells. No real table is that wide, so clamping beats blowing up.
_MAX_SPAN = 64


def _span_of(value: object) -> int:
    try:
        n = int(str(value).strip())
    except (TypeError, ValueError):
        return 1
    return max(1, min(n, _MAX_SPAN))


# Tags that can appear in table HTML; any < outside this whitelist is treated as a plain character.
# Checking only "is the character after < a letter" is not enough: a math cell such as $S_T<K$ has
# exactly the letter K after <, and HTMLParser would take that < as the start of a tag name and
# swallow the whole cell -- in practice 3 cells out of 3,374 real tables were lost this way, with no
# warning at all.
_HTML_TAG_RE = re.compile(
    r"</?(?:table|thead|tbody|tfoot|tr|td|th|caption|col|colgroup|"
    r"br|hr|b|i|u|s|em|strong|sup|sub|span|p|div|font|a|img|ul|ol|li)\b[^>]*>",
    re.IGNORECASE,
)


def _escape_bare_lt(html: str) -> str:
    """Escape any < that does not start a known tag as &lt;; convert_charrefs turns it back into <.

    For markup outside the whitelist (a genuinely unknown tag) the worst case is that it stays in the
    text as-is, which is better than having the content swallowed wholesale.
    """
    out: list[str] = []
    pos = 0
    while True:
        i = html.find("<", pos)
        if i < 0:
            out.append(html[pos:])
            return "".join(out)
        out.append(html[pos:i])
        m = _HTML_TAG_RE.match(html, i)
        if m:
            out.append(m.group(0))
            pos = m.end()
        else:
            out.append("&lt;")
            pos = i + 1


class _TableReader(HTMLParser):
    """<table> -> list of rows. Only cell text is kept; every attribute except colspan/rowspan is dropped.

    A cell spanning several rows or columns has its value **repeated** into every position it covers
    instead of leaving them empty: markdown tables have no notion of spans, and empty positions would
    shift all the following columns, and a misaligned table is harder to read than a repetitive one.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._span = (1, 1)
        self._col = 0
        self._carry: dict[int, list] = {}   # column index -> [rows still to fill, value]

    def _drain(self) -> None:
        """When this column's turn comes, first fill in the placeholders left by rowspans above."""
        while self._row is not None and self._col in self._carry and self._carry[self._col][0] > 0:
            self._carry[self._col][0] -= 1
            self._row.append(self._carry[self._col][1])
            self._col += 1

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag == "tr":
            self._flush_row()
            self._row, self._col = [], 0
        elif tag in {"td", "th"}:
            if self._row is None:          # some tables lack <tr>; open a row for them
                self._row, self._col = [], 0
            self._drain()
            self._cell = []
            a = {k.lower(): v for k, v in attrs}
            self._span = (_span_of(a.get("colspan")), _span_of(a.get("rowspan")))

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def _flush_cell(self) -> None:
        if self._cell is None or self._row is None:
            return
        value = re.sub(r"\s+", " ", "".join(self._cell)).strip().replace("|", "\\|")
        cols, rows = self._span
        for _ in range(cols):
            self._row.append(value)
            if rows > 1:
                self._carry[self._col] = [rows - 1, value]
            self._col += 1
        self._cell = None

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"}:
            self._flush_cell()
        elif tag == "tr":
            self._flush_row()

    def _flush_row(self) -> None:
        self._flush_cell()
        if self._row is None:
            return
        self._drain()
        if any(cell for cell in self._row):
            self.rows.append(self._row)
        self._row = None

    def close(self) -> None:
        # Truncated HTML must still hand over whatever was read -- losing the tail of one table
        # beats falling back to a pile of <td> for the whole table.
        super().close()
        self._flush_row()


def html_table_to_markdown(html: str) -> str | None:
    """MinerU table HTML -> markdown pipe table. Returns None when it cannot be recognised, and the
    caller keeps the original.

    MinerU gives tables a single table_body key whose content is <table><tr><td>..., with no markdown
    alternative. That markup would go into chunk text verbatim: measured, markup makes up a median of
    57%~61% of the body characters, and kb_005 alone has 312,105 tags. The tokenizer compresses the
    repeated </td><td> fairly well, so the real waste in tokens is 33% -- 3,374 real tables went from
    1,391,102 to 928,911 tokens. The waste is not only budget: a 399-token chunk can consist entirely
    of <td></td>, and its vector then represents "this is a table" rather than what the table holds.
    """
    rows = html_table_to_grid(html)
    return grid_to_markdown(rows) if rows else None


def html_table_to_grid(html: str) -> list[list[str]] | None:
    """MinerU table HTML -> cell grid (rows x columns, padded to equal width). Returns None when it
    cannot be recognised. Table ambiguity detection (table_check) and screenshot verification repair
    (table_repair) both operate on this grid."""
    if not html or not _TABLE_TAG_RE.search(html):
        return None
    reader = _TableReader()
    try:
        reader.feed(_escape_bare_lt(html))
        reader.close()
    except Exception:      # noqa: BLE001 - fall back to the original text; one table must not ruin a whole document
        return None
    rows = reader.rows
    if not rows:
        return None
    width = max(len(row) for row in rows)
    if width <= 0:
        return None
    return [row + [""] * (width - len(row)) for row in rows]


def grid_to_markdown(rows: list[list[str]] | None) -> str | None:
    if not rows:
        return None
    width = max(len(row) for row in rows)
    if width <= 0:
        return None
    padded = [row + [""] * (width - len(row)) for row in rows]
    lines = ["| " + " | ".join(padded[0]) + " |",
             "| " + " | ".join(["---"] * width) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in padded[1:])
    return "\n".join(lines)


def item_text(item: dict[str, Any], *, fallback_json: bool = False) -> str:
    values: list[str] = []
    # code_body: the text of MinerU's code listings / algorithm boxes ({"type": "code", "sub_type": "code" |
    # "algorithm", "code_body", "code_caption"}) lives entirely in this key. It used not to be read: such items
    # yielded an empty string, have no image, and were skipped as a whole, caption included, without a line in
    # the log; on one deployment 23 documents had lost about 667k characters, some technical books nearly half
    # of their text (2026-09-29 audit)
    for key in ("text", "content", "orig", "caption", "latex", "md", "markdown", "html", "table_body", "code_body", "name"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            if key in {"html", "table_body"}:
                # Convert table HTML to a pipe table before it enters the body. If conversion fails keep
                # it as-is -- better to leave markup in than to lose the content of an unrecognised table.
                value = html_table_to_markdown(value) or value
            values.append(value.strip())
    # MinerU 3.x list items: {"type": "list", "list_items": ["1. ...", "2. ..."]}; the whole body is in
    # list_items and text is empty -- this key used to be ignored, so the "expert advice" sections of
    # medical check-up reports were dropped wholesale (2026-09-07 health knowledge base audit)
    items = item.get("list_items")
    if isinstance(items, list):
        lines = [str(x).strip() for x in items if str(x).strip()]
        if lines:
            values.append("\n".join(lines))
    text = clean_text("\n".join(values))
    if text:
        return text
    return json.dumps(item, ensure_ascii=False) if fallback_json and item else ""


_FRONTMATTER_LIST_RE = re.compile(r"^\[(.*)\]$")


def parse_frontmatter(md: str) -> tuple[dict[str, Any], str]:
    """"key: value" metadata between the two leading --- lines (the SKILL.md / static site / Obsidian
    convention) -> (fields, body). Only the most common shapes are recognised: one key per line;
    multi-line values starting with `>-` / `|` absorb the indented lines that follow; `[a, b]` and
    comma-separated lists are kept verbatim as strings. Anything that is not frontmatter returns
    ({}, original text)."""
    if not md.startswith("---"):
        return {}, md
    lines = md.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, md
    end = next((i for i in range(1, len(lines)) if lines[i].strip() in ("---", "...")), None)
    if end is None:
        return {}, md
    fields: dict[str, Any] = {}
    key: str | None = None
    folded: list[str] = []
    for raw in lines[1:end]:
        if raw[:1] in (" ", "\t") and key is not None:
            folded.append(raw.strip())
            continue
        if key is not None and folded:
            fields[key] = (fields[key] + " " + " ".join(folded)).strip()
            folded = []
        if ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        key, value = key.strip(), value.strip()
        if not key:
            key = None
            continue
        if value in (">-", ">", "|", "|-"):
            value = ""
        fields[key] = value.strip("\"'")
    if key is not None and folded:
        fields[key] = (fields[key] + " " + " ".join(folded)).strip()
    return fields, "\n".join(lines[end + 1:])


_HR_RE = re.compile(r"^(?:-{3,}|\*{3,}|_{3,}|⸻+|—{3,})\s*$")
_FENCE_RE = re.compile(r"^(`{3,}|~{3,})(.*)$")
_SETEXT_H1_RE = re.compile(r"^=+\s*$")
_SETEXT_H2_RE = re.compile(r"^-{2,}\s*$")
_SETEXT_NOT_TEXT_RE = re.compile(r"^(?:[#>|]|[-*+]\s|\d+[.)]\s)")


def _atx_heading(line: str) -> tuple[str | None, int]:
    """"# Heading" -> (heading text, level); (None, 0) when the line is not a heading."""
    if not line.startswith("#"):
        return None, 0
    body = line.lstrip("#")
    if not body.startswith(" ") or not body.strip():
        return None, 0
    return body.strip(), len(line) - len(body)


def _setext_level(text: str, underline: str, *, paragraph_open: bool) -> int:
    """Setext heading: a line of text followed by a full line of === (level 1) or --- (level 2).
    Only a "short text standing on its own line" counts: if body text sits directly above it (the
    paragraph has not ended) it is not a heading but a paragraph followed by a rule; list items,
    quotes, table rows and full sentences ending in punctuation do not count either (health check B9)."""
    if paragraph_open or not text or len(text) > 120:
        return 0
    if _SETEXT_NOT_TEXT_RE.match(text) or _HR_RE.match(text) or text[-1] in ".。!?;:,，":
        return 0
    if _SETEXT_H1_RE.match(underline):
        return 1
    if _SETEXT_H2_RE.match(underline):
        return 2
    return 0


def markdown_to_blocks(
    md: str,
    *,
    parser: str,
    parser_profile: str,
    doc_type: str,
) -> list[ParsedBlock]:
    from ..headings import clean_heading_text

    md = clean_text(md)
    if not md:
        return []
    frontmatter, md = parse_frontmatter(md)
    blocks: list[ParsedBlock] = []
    if frontmatter:
        # The metadata becomes a block of its own: name / description are the best retrieval text and
        # also the entity source for rule-based extraction in the graph build
        text = "\n".join(f"{k}: {v}" for k, v in frontmatter.items() if str(v).strip())
        blocks.append(ParsedBlock(
            parser=parser, parser_profile=parser_profile, doc_type=doc_type, block_type="text",
            text=text, title=str(frontmatter.get("name") or frontmatter.get("title") or ""), block_id="md-fm",
            metadata={"section_path": [], "frontmatter": frontmatter},
        ))
    tracker = SectionTracker()
    sections: list[tuple[str | None, list[str], list[str]]] = []
    current_title: str | None = None
    current_lines: list[str] = []
    # Character and length of the opening fence: it is closed only by a line of the same character,
    # at least as long, with nothing else after it (``` inside ```` is content); a "# comment" or "---"
    # inside a fence is code, not a heading / rule.
    fence: tuple[str, int] | None = None
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.lstrip()
        i += 1
        fence_match = _FENCE_RE.match(stripped)
        if fence_match:
            marker = fence_match.group(1)
            if fence is None:
                fence = (marker[0], len(marker))
                current_lines.append(line)
                continue
            if marker[0] == fence[0] and len(marker) >= fence[1] and not fence_match.group(2).strip():
                fence = None
                current_lines.append(line)
                continue
        if fence is not None:
            current_lines.append(line)
            continue
        heading, level = _atx_heading(line)
        if heading is None and i < len(lines):
            level = _setext_level(stripped, lines[i].rstrip(),
                                  paragraph_open=bool(current_lines and current_lines[-1].strip()))
            if level:
                heading = stripped
                i += 1      # consume the underline too; it must not end up in the body
        if heading is not None:
            if current_lines:
                sections.append((current_title, current_lines, tracker.path))
            current_title = clean_heading_text(heading) or heading
            tracker.observe(level, current_title)
            # The heading is carried by title + section_path; keeping it in the
            # body too would embed and index the same string twice per chunk.
            current_lines = []
        elif _HR_RE.match(stripped):
            # A rule (--- / *** / ⸻) is not content; on its own it would only become a fragment, so
            # drop it together with the blank line before it
            if current_lines and not current_lines[-1].strip():
                current_lines.pop()
            continue
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_title, current_lines, tracker.path))

    for idx, (title, lines, section_path) in enumerate(sections, start=1):
        text = clean_text("\n".join(lines))
        if not text:
            continue
        blocks.append(
            ParsedBlock(
                parser=parser,
                parser_profile=parser_profile,
                doc_type=doc_type,
                block_type="text",
                text=text,
                title=title,
                block_id=f"md-{idx:04d}",
                metadata={"section_path": list(section_path)},
            )
        )
    return blocks


def caption_of(item: dict[str, Any]) -> str:
    """MinerU puts captions in type-specific keys, never in a plain `caption`."""
    values: list[str] = []
    for key in ("image_caption", "table_caption", "chart_caption", "code_caption"):
        values.extend(_as_str_list(item.get(key)))
    return clean_text("\n".join(values))


def footnote_of(item: dict[str, Any]) -> str:
    values: list[str] = []
    for key in ("image_footnote", "table_footnote", "chart_footnote", "code_footnote"):
        values.extend(_as_str_list(item.get(key)))
    return clean_text("\n".join(values))


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    if not text or text in {"[]", "''", '""'}:
        return []
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(text)
        except Exception:
            continue
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
        if isinstance(parsed, str) and parsed.strip():
            return [parsed.strip()]
    return [text]


# Layout-based heading / page footer detection lives in kb_pipeline/headings.py (a leaf module): the
# chunker needs it too, and keeping it in the parsers package would create a circular import. This is
# only a re-export.
from ..headings import HEADING_MAX_CHARS, PAGE_FOOTER_RE, HeadingResolver, clean_heading_text, infer_heading_level, is_page_footer, resolve_heading_level  # noqa: E402,F401


def heading_level(item: dict[str, Any]) -> int | None:
    """MinerU marks headings with text_level (1 = top level)."""
    level = maybe_int(item.get("text_level"))
    return level if level and level > 0 else None


class SectionTracker:
    """Walks blocks in document order and keeps the current heading stack."""

    def __init__(self) -> None:
        self._stack: list[tuple[int, str]] = []

    def observe(self, level: int | None, title: str) -> None:
        from ..headings import clean_heading_text

        title = clean_heading_text(clean_text(title))
        if not level or not title:
            return
        self._stack = [(lv, tt) for lv, tt in self._stack if lv < level]
        self._stack.append((level, title))

    @property
    def path(self) -> list[str]:
        return [title for _, title in self._stack]


def parser_profile_for_path(path: Path) -> str:
    """Route-level profile: decides which parsing chain a file takes purely by extension. It is a
    separate thing from the block-level profile (the one inside chunk_uid, with the +text-merge-vN
    suffix).

    detect_change compares this value -- when it changes, scan marks every file of that type as
    parser_changed and dispatches parse tasks. So when changing a parser implementation, bump the
    version here only if historical files genuinely need to be re-parsed.

    v2 -> v3 (pdf) / v1 -> v2 (docx), 2026-08-29: formulas merged into the body buffer, docx gained
    the merge step. Both change the chunking result, so historical files must be re-parsed to get it.
    v3 -> v4 (pdf) / v2 -> v3 (docx), same day: a short heading sandwiched between tables/images is
    merged into the figure/table it describes.
    pdf v4->v5 / docx v3->v4 / pptx v1->v2, same day: table HTML converted to pipe tables (saves 33%
    of tokens).
    pdf v5->v6 / docx v4->v5, 2026-09-03: headings MinerU missed ("Chapter N / 1.2.3 / CJK chapter numbering") are
    recognised from layout, whole-line page-number blocks count as noise; chunking breaks before
    headings and overlap never crosses a heading.
    The 2026-09-05 chunking clean-up (tables split by row with header, fragments merged into
    neighbouring chunks, captions attached to figures/tables, notes attached to figures/tables,
    heading level correction, code grouped by class, section prefix in embeddings, VLM image text
    tidy-up) did **not** touch the version here: the content in the knowledge bases is all test data
    and not worth an automatic full re-parse. To give historical files the new chunking, bump the
    corresponding version by 1 and scan will re-parse them one by one.
    pdf v6->v7 / docx v5->v6, evening of 2026-09-06 (Codex review F02): markers for merged stacked
    table rows + screenshot verification row splitting are now in the blocks; parameter table values
    are directly affected, so historical PDF / DOCX need re-parsing.
    pdf v7->v8 / docx v6->v7, same evening: detection rules tightened (mm / ** / multi-condition
    cells no longer misjudged), verification now matches the transcription by content; fixed the
    first real-machine round where all 9 tables went unverified.
    pdf v8->v9 / docx v7->v8, 2026-09-07 (health knowledge base audit): MinerU list_items used to be
    dropped wholesale; a table split across pages into a header-less fragment is merged back into the
    previous one; QR codes / barcodes / icons / signatures are folded into the body, and photos with a
    substantive description (ultrasound image pages) are no longer folded away as decorative images.
    pdf v14->v15 / docx v9->v10 / py-symbols, code-symbols v1->v2 / md v2->v3 / other text types v1->v2,
    2026-09-30: integer chapter numbers ("4 Technical requirements") count as numbered headings and docx
    style headings are no longer demoted; a heading at the end of a block is no longer lost (every type
    that goes through prose chunking) and travels with the body text below it when merging; tables that
    span three or more pages are merged back; code is split into lines the same way the syntax tree
    numbers them, and a decorator belongs to the definition it decorates.
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        # v10 (2026-09-09 Codex review): original text absorbed while folding decorative images flows
        # back (F03), cross-page page ranges enter the payload (F09), conflicts between in-image text
        # and estimated values resolve to the text and leave a trace (F01), runaway repetitive
        # descriptions are collapsed / retried
        return "pdf-mineru-table-vlm-v15"
    if suffix == ".pptx":
        return "pptx-mineru-slide-vlm-v4"     # v4 (2026-09-30): repeated content on a slide is no longer de-duplicated
    if suffix == ".docx":
        # v10 (2026-09-30): repeated paragraphs / tables / images are no longer de-duplicated; native charts
        # become a data table read series by series, placed back in their section
        return "docx-mineru-ooxml-vlm-v10"
    if suffix == ".doc":
        return "doc-unsupported-v1"
    if suffix == ".html":
        return "html-dom-v2"          # v2 (2026-09-30): tables follow their section with its section path, cells spanning rows / columns are filled in
    if suffix in {".md", ".markdown"}:
        return "md-sections-v3"       # v2 (2026-09-04): frontmatter parsed into fields and given its own block
    if suffix == ".py":
        return "py-symbols-v2"        # 2026-09-04: chunked by function / class / method, blocks carry symbol metadata
    from .code_symbols import LANGUAGE_BY_SUFFIX

    if suffix in LANGUAGE_BY_SUFFIX:
        return "code-symbols-v2"      # 2026-09-05: tree-sitter chunking by symbol (JS/TS/Go/Java/Rust/C/C++/C#/PHP/Ruby/Swift/Kotlin/Scala/Shell/Lua/PowerShell)
    if suffix in {".xlsx", ".xls", ".csv"}:
        # v4 (2026-09-30): percentages / dates are written as the sheet displays them, split rows no longer carry
        # the whole header, long cells are split, and the notes before the header are neither truncated nor repeated
        return "table-native-v4"
    if suffix in IMAGE_SUFFIXES:
        return "image-vlm-v1"
    return f"{infer_doc_type(path.name)}-native-v2"
