from __future__ import annotations

import copy
import hashlib
import re
from html import unescape
from pathlib import Path

from lxml import etree, html

from ..models import ParsedBlock
from .common import SectionTracker, _span_of, clean_text, decode_text_bytes


PARSER_PROFILE = "html-dom-v2"
NOISE_XPATHS = (
    ".//script",
    ".//style",
    ".//noscript",
    ".//template",
    ".//*[contains(concat(' ', normalize-space(@class), ' '), ' topbar ')]",
    ".//*[contains(concat(' ', normalize-space(@class), ' '), ' sidebar ')]",
    ".//*[contains(concat(' ', normalize-space(@class), ' '), ' menu-toggle ')]",
    ".//*[@role='navigation']",
    ".//nav",
    ".//footer",
)
SPECIAL_BLOCK_XPATHS = (
    ".//*[contains(concat(' ', normalize-space(@class), ' '), ' scene ')]",
    ".//*[contains(concat(' ', normalize-space(@class), ' '), ' faq-item ')]",
    ".//*[contains(concat(' ', normalize-space(@class), ' '), ' e2e-stage ')]",
    ".//*[contains(concat(' ', normalize-space(@class), ' '), ' skill-goal ')]",
    ".//*[contains(concat(' ', normalize-space(@class), ' '), ' build-path ')]",
    ".//*[contains(concat(' ', normalize-space(@class), ' '), ' prompt-box ')]",
    ".//table",
)
HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
# Pages with these custom class names are the tailored site pages (scene / faq ...), which keep the
# tailored extraction; only generic pages are sectioned by heading
_TAILORED_XPATHS = tuple(xp for xp in SPECIAL_BLOCK_XPATHS if xp != ".//table")
BLOCK_BREAK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "br",
    "dd",
    "div",
    "dl",
    "dt",
    "figcaption",
    "figure",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "td",
    "th",
    "tr",
    "ul",
}


def parse_html_dom(path: Path) -> list[ParsedBlock]:
    # Parse bytes, not text: lxml then honours a declared charset (GBK legacy
    # pages survive) and accepts XHTML's XML declaration, which raises
    # ValueError on unicode input and used to shunt whole XHTML files into
    # the raw fallback.
    raw_bytes = path.read_bytes()
    try:
        root = html.document_fromstring(raw_bytes, parser=_parser_for(raw_bytes))
    except (etree.ParserError, ValueError) as exc:
        return _raw_fallback(decode_text_bytes(raw_bytes), title=path.stem, reason=repr(exc))

    _remove_noise(root)
    doc_title = _document_title(root) or path.stem
    pages = _page_elements(root)
    blocks: list[ParsedBlock] = []
    used_page_ids: set[str] = set()
    for page_idx, page in enumerate(pages, start=1):
        page_id = str(page.get("id") or f"page-{page_idx}")
        # An explicit id="page-2" must not collide with the synthetic name of
        # the second unnamed page (or a duplicated id): colliding page ids
        # produce identical block_ids and the points overwrite each other.
        if page_id in used_page_ids:
            page_id = f"{page_id}-{page_idx}"
        used_page_ids.add(page_id)
        page_title = _first_heading(page) or doc_title
        blocks.extend(_blocks_for_page(page, page_id=page_id, page_title=page_title))

    if not _substantial(blocks):
        return _raw_fallback(decode_text_bytes(raw_bytes), title=doc_title, reason="dom extraction returned too little text")
    return blocks


def _blocks_for_page(page: etree._Element, *, page_id: str, page_title: str) -> list[ParsedBlock]:
    blocks: list[ParsedBlock] = []
    seen: set[str] = set()

    overview = copy.deepcopy(page)
    tailored = any(page.xpath(xp) for xp in _TAILORED_XPATHS)
    sectioned = None if tailored else _sectioned_overview(overview)
    if sectioned is not None:
        # Generic HTML (health check R9): sectioned by h1-h6, one block per section with a
        # section_path, so chunks get a section prefix; previously the whole page was one block and
        # long documents carried no section information
        preface, sections, tables = sectioned
        tracker = SectionTracker()

        def section_tables(section: int, title: str) -> list[ParsedBlock]:
            # A table follows the section it is in: placed after that section's body, with the same section
            # path and titled with that section's heading. Put at the end of the page with every title hung
            # on the page's first heading, the table of the second section was attributed to the first
            found = []
            for number, (at, table) in enumerate(tables, start=1):
                table_md = _table_markdown(table) if at == section else ""
                if table_md:
                    found.append(_block(block_type="table", text=table_md, title=f"{title} / Table {number}",
                                        block_id=f"{_safe_id(page_id)}-table-{number:04d}", page_id=page_id,
                                        page_title=page_title, html_block_type="table", table_markdown=table_md,
                                        section_path=tracker.path))
            return found

        if preface:
            blocks.append(_block(block_type="text", text=preface, title=page_title,
                                 block_id=f"{_safe_id(page_id)}-overview", page_id=page_id,
                                 page_title=page_title, html_block_type="overview"))
        blocks.extend(section_tables(0, page_title))
        for idx, (title, level, text) in enumerate(sections, start=1):
            tracker.observe(level, title)
            if text:            # otherwise the next heading follows directly: no body to store, the path is already in the tracker
                blocks.append(_block(block_type="text", text=text, title=title or page_title,
                                     block_id=f"{_safe_id(page_id)}-s{idx:03d}", page_id=page_id,
                                     page_title=page_title, html_block_type="section", section_path=tracker.path))
            blocks.extend(section_tables(idx, title or page_title))
    else:
        _remove_nodes(overview, SPECIAL_BLOCK_XPATHS)
        overview_text = _element_text(overview)
        if overview_text:
            blocks.append(
                _block(
                    block_type="text",
                    text=overview_text,
                    title=page_title,
                    block_id=f"{_safe_id(page_id)}-overview",
                    page_id=page_id,
                    page_title=page_title,
                    html_block_type="overview",
                )
            )

    for idx, element in enumerate(_class_elements(page, "scene"), start=1):
        if _skip_nested_or_seen(seen, element):
            continue
        title = _first_text(element, ".//*[contains(concat(' ', normalize-space(@class), ' '), ' scene-title ')]")
        title = title or _first_heading(element) or f"Scene {idx}"
        blocks.append(
            _block(
                block_type="scene",
                text=_element_text(element),
                title=f"{page_title} / {title}",
                block_id=f"{_safe_id(page_id)}-scene-{idx:04d}",
                page_id=page_id,
                page_title=page_title,
                html_block_type="scene",
            )
        )

    for idx, element in enumerate(_class_elements(page, "faq-item"), start=1):
        if _skip_nested_or_seen(seen, element):
            continue
        question = _first_text(element, ".//*[contains(concat(' ', normalize-space(@class), ' '), ' q-text ')]")
        answer = _first_text(element, ".//*[contains(concat(' ', normalize-space(@class), ' '), ' faq-a ')]")
        text = clean_text(f"QUESTION: {question}\nANSWER: {answer}") if question or answer else _element_text(element)
        title = question or f"FAQ {idx}"
        blocks.append(
            _block(
                block_type="faq",
                text=text,
                title=f"{page_title} / {title}",
                block_id=f"{_safe_id(page_id)}-faq-{idx:04d}",
                page_id=page_id,
                page_title=page_title,
                html_block_type="faq",
            )
        )

    for idx, element in enumerate(_class_elements(page, "e2e-stage"), start=1):
        if _skip_nested_or_seen(seen, element):
            continue
        title = _first_text(element, ".//*[contains(concat(' ', normalize-space(@class), ' '), ' stage-title ')]")
        title = title or _first_heading(element) or f"Stage {idx}"
        blocks.append(
            _block(
                block_type="stage",
                text=_element_text(element),
                title=f"{page_title} / {title}",
                block_id=f"{_safe_id(page_id)}-stage-{idx:04d}",
                page_id=page_id,
                page_title=page_title,
                html_block_type="e2e-stage",
            )
        )

    for idx, element in enumerate(_class_elements(page, "skill-goal"), start=1):
        if _skip_nested_or_seen(seen, element):
            continue
        title = _first_text(element, ".//*[contains(concat(' ', normalize-space(@class), ' '), ' skill-goal-name ')]")
        title = title or f"Skill Goal {idx}"
        blocks.append(
            _block(
                block_type="card",
                text=_element_text(element),
                title=f"{page_title} / {title}",
                block_id=f"{_safe_id(page_id)}-skill-goal-{idx:04d}",
                page_id=page_id,
                page_title=page_title,
                html_block_type="skill-goal",
            )
        )

    for idx, element in enumerate(_class_elements(page, "build-path"), start=1):
        if _skip_nested_or_seen(seen, element):
            continue
        title = _first_text(element, ".//*[contains(concat(' ', normalize-space(@class), ' '), ' title ')]")
        title = title or f"Build Path {idx}"
        blocks.append(
            _block(
                block_type="card",
                text=_element_text(element),
                title=f"{page_title} / {title}",
                block_id=f"{_safe_id(page_id)}-build-path-{idx:04d}",
                page_id=page_id,
                page_title=page_title,
                html_block_type="build-path",
            )
        )

    for idx, element in enumerate(_class_elements(page, "prompt-box"), start=1):
        if _skip_nested_or_seen(seen, element):
            continue
        title = _first_text(element, ".//*[contains(concat(' ', normalize-space(@class), ' '), ' prompt-label ')]")
        title = title or f"Prompt {idx}"
        blocks.append(
            _block(
                block_type="prompt",
                text=_element_text(element),
                title=f"{page_title} / {title}",
                block_id=f"{_safe_id(page_id)}-prompt-{idx:04d}",
                page_id=page_id,
                page_title=page_title,
                html_block_type="prompt",
            )
        )

    for idx, table in enumerate(() if sectioned is not None else page.xpath(".//table"), start=1):
        if _skip_nested_or_seen(seen, table):
            continue
        table_md = _table_markdown(table)
        if not table_md:
            continue
        blocks.append(
            _block(
                block_type="table",
                text=table_md,
                title=f"{page_title} / Table {idx}",
                block_id=f"{_safe_id(page_id)}-table-{idx:04d}",
                page_id=page_id,
                page_title=page_title,
                html_block_type="table",
                table_markdown=table_md,
            )
        )

    return [block for block in blocks if block.text.strip() or (block.table_markdown or "").strip()]


def _block(
    *,
    block_type: str,
    text: str,
    title: str,
    block_id: str,
    page_id: str,
    page_title: str,
    html_block_type: str,
    table_markdown: str | None = None,
    section_path: list[str] | None = None,
) -> ParsedBlock:
    metadata: dict = {
        "html_page_id": page_id,
        "html_page_title": page_title,
        "html_block_type": html_block_type,
    }
    if section_path is not None:
        metadata["section_path"] = list(section_path)
    return ParsedBlock(
        parser="html_dom",
        parser_profile=PARSER_PROFILE,
        doc_type="html",
        block_type=block_type,
        text=clean_text(text),
        title=clean_text(title),
        table_markdown=table_markdown,
        block_id=block_id,
        metadata=metadata,
    )


def _sectioned_overview(
    overview: etree._Element,
) -> tuple[str, list[tuple[str, int, str]], list[tuple[int, etree._Element]]] | None:
    """Section a generic page by h1-h6: returns (text before the first heading, [(heading, level,
    body) ...], [(section it is in, table) ...]); with fewer than 2 headings it returns None and the
    whole page stays one block. The heading text does not go into the body (title + section_path
    already carry it); nor do tables: the section each one is in is recorded (0 = before the first
    heading) and it becomes a block of its own. Headings inside a table do not start a section."""
    parts: list[list[str]] = [[]]
    titles: list[tuple[str, int]] = []
    tables: list[tuple[int, etree._Element]] = []

    def visit(node: etree._Element) -> None:
        tag = _tag(node)
        if tag in {"script", "style", "noscript", "template"}:
            return
        if tag == "table":
            tables.append((len(titles), node))
            return
        if tag in HEADING_TAGS:
            titles.append((_normalize_text("".join(node.itertext())).replace("\n", " "), int(tag[1])))
            parts.append([])
            return
        if tag in BLOCK_BREAK_TAGS:
            parts[-1].append("\n")
        if tag == "li":
            parts[-1].append("- ")
        if node.text:
            parts[-1].append(node.text)
        for child in node:
            visit(child)
            if child.tail:
                parts[-1].append(child.tail)
        if tag in BLOCK_BREAK_TAGS:
            parts[-1].append("\n")

    visit(overview)
    if len(titles) < 2:
        return None
    preface = _normalize_text("".join(parts[0]))
    sections = [(title, level, _normalize_text("".join(body))) for (title, level), body in zip(titles, parts[1:])]
    return preface, sections, tables


_CHARSET_RE = re.compile(rb"""charset\s*=\s*["']?\s*([A-Za-z0-9_.:-]+)""", re.IGNORECASE)
_XML_ENCODING_RE = re.compile(rb"""^\s*<\?xml[^>]*encoding\s*=\s*["']([A-Za-z0-9_.:-]+)["']""", re.IGNORECASE)


def _parser_for(raw: bytes):
    """Pages without a declared charset are decoded as UTF-8: lxml guesses Latin-1 for an undeclared
    byte stream, turning a whole Chinese page into mojibake (found in passing during the 2026-09-06
    health check R9). With a declared charset lxml follows the declaration (legacy GBK pages work as
    before); anything that is not valid UTF-8 is also left to lxml to guess."""
    head = raw[:4096]
    if _CHARSET_RE.search(head) or _XML_ENCODING_RE.match(head):
        return None
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return html.HTMLParser(encoding="utf-8")


def _document_title(root: etree._Element) -> str:
    return _first_text(root, ".//title")


def _page_elements(root: etree._Element) -> list[etree._Element]:
    # Only top-level pages: a .page nested inside another .page would have its
    # content extracted twice (once as its own page, once inside the parent).
    pages = [page for page in _class_elements(root, "page") if not _has_page_ancestor(page)]
    if pages:
        return pages
    body = root.find(".//body")
    return [body if body is not None else root]


def _has_page_ancestor(element: etree._Element) -> bool:
    parent = element.getparent()
    while parent is not None:
        classes = f" {' '.join((parent.get('class') or '').split())} "
        if " page " in classes:
            return True
        parent = parent.getparent()
    return False


def _remove_noise(root: etree._Element) -> None:
    _remove_nodes(root, NOISE_XPATHS)


def _remove_nodes(root: etree._Element, xpaths: tuple[str, ...]) -> None:
    for xpath in xpaths:
        for element in list(root.xpath(xpath)):
            parent = element.getparent()
            if parent is not None:
                parent.remove(element)


def _class_elements(root: etree._Element, class_name: str) -> list[etree._Element]:
    return list(root.xpath(f".//*[contains(concat(' ', normalize-space(@class), ' '), ' {class_name} ')]"))


def _first_heading(root: etree._Element) -> str:
    return _first_text(root, ".//h1|.//h2|.//h3|.//h4|.//h5|.//h6")


def _first_text(root: etree._Element, xpath: str) -> str:
    for element in root.xpath(xpath):
        text = _element_text(element)
        if text:
            return text
    return ""


def _element_text(element: etree._Element) -> str:
    parts: list[str] = []

    def visit(node: etree._Element) -> None:
        tag = _tag(node)
        if tag in {"script", "style", "noscript", "template"}:
            return
        if tag in BLOCK_BREAK_TAGS:
            parts.append("\n")
        if tag == "li":
            parts.append("- ")
        if node.text:
            parts.append(node.text)
        for child in node:
            visit(child)
            if child.tail:
                parts.append(child.tail)
        if tag in BLOCK_BREAK_TAGS:
            parts.append("\n")

    visit(element)
    return _normalize_text("".join(parts))


def _normalize_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    lines = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = re.sub(r"[ \t\f\v]+", " ", raw).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def _table_markdown(table: etree._Element) -> str:
    rows: list[list[str]] = []
    carry: dict[int, list] = {}      # column -> [rows still to fill below, value]: placeholders left by a rowspan above
    for tr in table.xpath(".//tr"):
        # Rows of a nested table belong to that table; pulling them up here
        # duplicated the inner content (the outer cell's text already inlines
        # it) and misaligned the outer table's rows.
        nearest_table = tr.xpath("ancestor::table[1]")
        if nearest_table and nearest_table[0] is not table:
            continue
        cells: list[str] = []

        def fill_from_above() -> None:
            while carry.get(len(cells), [0])[0] > 0:
                carry[len(cells)][0] -= 1
                cells.append(carry[len(cells)][1])

        for cell in tr.xpath("./th|./td"):
            fill_from_above()
            value = _escape_table_cell(_element_text(cell).replace("\n", " "))
            down = _span_of(cell.get("rowspan")) - 1
            # A cell spanning rows / columns fills its value into every position it covers (the same rule as
            # common._TableReader): a markdown table has no merged cells, and leaving them empty would shift
            # the columns after them
            for _ in range(_span_of(cell.get("colspan"))):
                if down:
                    carry[len(cells)] = [down, value]
                cells.append(value)
        fill_from_above()
        if any(cells):
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    header = rows[0]
    body = rows[1:]
    output = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    output.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(output)


def _escape_table_cell(text: str) -> str:
    return clean_text(text).replace("\n", " ").replace("|", "\\|")


def _skip_nested_or_seen(seen: set[str], element: etree._Element) -> bool:
    return _inside_seen(seen, element) or _mark_seen(seen, element)


def _mark_seen(seen: set[str], element: etree._Element) -> bool:
    element_key = _node_key(element)
    if element_key in seen:
        return True
    seen.add(element_key)
    return False


def _inside_seen(seen: set[str], element: etree._Element) -> bool:
    parent = element.getparent()
    while parent is not None:
        if _node_key(parent) in seen:
            return True
        parent = parent.getparent()
    return False


def _node_key(element: etree._Element) -> str:
    return element.getroottree().getpath(element)


def _tag(node: etree._Element) -> str:
    return str(node.tag).lower() if isinstance(node.tag, str) else ""


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_.-]+", "-", value).strip("-")
    if safe:
        return safe[:80]
    if value.strip():
        # A pure-CJK id sanitised to nothing must stay distinct: every such
        # page collapsing to "html" made block_ids (and point ids) collide.
        return "id-" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return "html"


def _substantial(blocks: list[ParsedBlock]) -> bool:
    return sum(len(block.text.strip()) for block in blocks) >= 80


_RAW_FALLBACK_MAX_CHARS = 50_000


def _raw_fallback(raw: str, *, title: str, reason: str) -> list[ParsedBlock]:
    # The fallback used to index the raw source verbatim -- <script> bodies
    # included -- so a JS-rendered shell embedded its whole bundle. Strip
    # scripts/styles/comments and tags, decode entities, and cap the size.
    text = re.sub(r"(?is)<(script|style|noscript|template)\b[^>]*>.*?</\1\s*>", " ", raw)
    text = re.sub(r"(?s)<!--.*?-->", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = clean_text(unescape(text))
    if len(text) > _RAW_FALLBACK_MAX_CHARS:
        text = text[:_RAW_FALLBACK_MAX_CHARS]
    return [
        ParsedBlock(
            parser="html_dom",
            parser_profile=PARSER_PROFILE,
            doc_type="html",
            block_type="text",
            text=text,
            title=clean_text(title),
            block_id="html-raw-fallback",
            metadata={"html_fallback_reason": reason},
        )
    ]
