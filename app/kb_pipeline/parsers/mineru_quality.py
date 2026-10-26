"""Recover PDF content and provenance from MinerU's existing structured output."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from lxml import etree, html

from ..headings import is_page_footer
from .common import bbox, item_text, label_of, page_idx, walk_dicts


def recover_pdf_items(items: list[dict], payload: dict) -> list[dict]:
    items = [dict(item) for item in items]
    body_texts = {"".join(item_text(i).casefold().split()) for i in items if label_of(i) in {"text", "title"}}
    headers = defaultdict(list)
    for item in items:
        if label_of(item) in {"header", "page_header"}:
            headers["".join(item_text(item).casefold().split())].append(item)
    first_page = min((page_idx(i) for i in items if page_idx(i) is not None), default=1)
    for group in headers.values():
        pages = {page_idx(i) for i in group}
        for index, item in enumerate(group):
            text = item_text(item).strip()
            if not text or is_page_footer(text):
                continue
            if "".join(text.casefold().split()) in body_texts or (len(pages) == 1 and index > 0):
                continue
            box = bbox(item)
            height = box[3] - box[1] if box and len(box) == 4 else 0
            other_heights = [b[3] - b[1] for i in group if page_idx(i) != first_page
                             and (b := bbox(i)) and len(b) == 4]
            # Repeated running headers stay noise. A unique header can be a
            # cover title or body misclassified by the layout model. A larger
            # cover occurrence of a repeated title also carries document content.
            cover = (page_idx(item) == first_page and other_heights
                     and height > 1.5 * max(other_heights))
            if len(pages) > 1 and not cover:
                continue
            item["original_source_type"] = label_of(item)
            item["type"] = "title" if cover or (page_idx(item) == first_page and height >= 25) else "text"
            if item["type"] == "title":
                item["text_level"] = 1
    ends = _merged_table_ends(payload)
    for item in items:
        if "table" in label_of(item):
            key = (page_idx(item), Path(str(item.get("img_path") or item.get("image_path") or "")).name)
            if key in ends:
                item["source_page_end"] = ends[key]
    return items


def _middle_pages(payload: dict) -> list[dict]:
    for obj in walk_dicts(payload):
        middle = obj.get("middle_json")
        if isinstance(middle, str):
            try:
                middle = json.loads(middle)
            except (ValueError, TypeError):
                continue
        if isinstance(middle, dict) and isinstance(middle.get("pdf_info"), list):
            return middle["pdf_info"]
    return []


def _table_rows(value: str) -> list[tuple[str, ...]]:
    try:
        root = html.fromstring(value)
    except (ValueError, etree.ParserError):
        return []
    return [tuple("".join("".join(cell.itertext()).split()) for cell in row.xpath("./th|./td"))
            for row in root.xpath(".//tr") if row.xpath("./th|./td")]


def _tables(blocks: list[dict]) -> list[tuple[str, list[tuple[str, ...]]]]:
    out = []
    for block in blocks:
        if block.get("type") != "table":
            continue
        for span in walk_dicts(block):
            if span.get("type") == "table" and span.get("html"):
                out.append((Path(str(span.get("image_path") or "")).name, _table_rows(span["html"])))
    return out


def _merged_table_ends(payload: dict) -> dict[tuple[int, str], int]:
    # preproc_blocks retain physical-page fragments; para_blocks may already
    # contain a merged table whose content_list entry names only its first page.
    pages = _middle_pages(payload)
    originals = {page_idx(p): _tables(p.get("preproc_blocks") or []) for p in pages}
    ends = {}
    for page in pages:
        start = page_idx(page)
        if start is None:
            continue
        local = dict(originals.get(start, []))
        for image_name, merged in _tables(page.get("para_blocks") or []):
            first = local.get(image_name, [])
            if not image_name or not first or len(merged) <= len(first) or merged[:len(first)] != first:
                continue
            remaining = merged[len(first):]
            end = start
            while remaining and end + 1 in originals:
                # Require an exact, unique row-sequence match on the adjacent
                # page. Shared column labels alone never establish provenance.
                candidates = []
                for _, rows in originals[end + 1]:
                    if rows and rows[0] == first[0]:
                        rows = rows[1:]
                    if rows and remaining[:len(rows)] == rows:
                        candidates.append(rows)
                if len(candidates) != 1:
                    break
                remaining = remaining[len(candidates[0]):]
                end += 1
            if not remaining:
                ends[(start, image_name)] = end
    return ends
