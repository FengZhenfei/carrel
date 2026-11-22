"""Recover PDF content and provenance from MinerU's existing structured output."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from lxml import etree, html

from ..headings import is_page_footer
from .common import bbox, caption_of, footnote_of, item_text, label_of, page_idx, walk_dicts


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
    # The same for footers: a line the layout model called a footer and that stands on one page only is a
    # figure caption, a note or a credit, not page furniture (trial: 37 such blocks in 15 PDFs, among them
    # captions and a 222-character note). Footers carry page numbers in many forms, so texts that differ only
    # in their digits count as one footer; a line without a single letter is not content.
    footers = defaultdict(list)
    for item in items:
        if label_of(item) in {"footer", "page_footer"}:
            footers[re.sub(r"\d+", "0", "".join(item_text(item).casefold().split()))].append(item)
    for group in footers.values():
        item, text = group[0], item_text(group[0]).strip()
        if (len({page_idx(i) for i in group}) != 1 or not text or is_page_footer(text)
                or not re.search(r"[^\W\d_]", text) or "".join(text.casefold().split()) in body_texts):
            continue
        item["original_source_type"] = label_of(item)
        item["type"] = "text"
    continuations = {}
    ends = _merged_table_ends(payload, continuations=continuations)
    for item in items:
        if "table" in label_of(item):
            key = (page_idx(item), Path(str(item.get("img_path") or item.get("image_path") or "")).name)
            if key in ends:
                item["source_page_end"] = ends[key]
                merged, regions = continuations[key]
                # The content_list body must be the same complete table, not
                # just a similarly named or partially recovered entry.
                if _table_rows(item.get("table_body") or item.get("html") or "") == merged:
                    item["_merged_table_continuations"] = regions
    # MinerU can append discarded headers and footers after the page body. Restore only the
    # recovered items to their geometric neighbours; keep the other reading order.
    for item in list(items):
        box = bbox(item)
        if not item.get("original_source_type") or not box or len(box) != 4:
            continue
        neighbours = []
        for i, other in enumerate(items):
            ob = bbox(other)
            if other is item or page_idx(other) != page_idx(item) or not ob or len(ob) != 4:
                continue
            if label_of(other) in {"header", "page_header", "footer", "page_footer", "page_number"}:
                continue
            overlap = min(box[2], ob[2]) - max(box[0], ob[0])
            if overlap > 0 and overlap >= .5 * min(box[2] - box[0], ob[2] - ob[0]):
                neighbours.append((i, other, ob))
        below = [n for n in neighbours if n[2][1] >= box[3]]
        above = [n for n in neighbours if n[2][3] <= box[1]]
        target = min(below, key=lambda n: n[2][1]) if below else max(above, key=lambda n: n[2][3]) if above else None
        if target:
            items.remove(item)
            index = next(i for i, v in enumerate(items) if v is target[1])
            items.insert(index if below else index + 1, item)
    return items


def merge_table_placeholders(items: list[dict], path=None) -> list[dict]:
    """Fold proven upstream continuation placeholders into their owning table.

    Run after note attachment so continuation footnotes and their locations
    travel with the parent. An uncertain or nonempty entry remains independent.
    """
    items = [dict(item) for item in items]
    claims = defaultdict(list)
    for parent_index, parent in enumerate(items):
        for region in parent.pop("_merged_table_continuations", []):
            box = region["bbox"]
            candidates = [i for i, child in enumerate(items)
                if i != parent_index and label_of(child) == "table"
                and page_idx(child) == region["page"] and not item_text(child) and not caption_of(child)
                and not (child.get("img_path") or child.get("image_path"))
                and (cb := bbox(child)) and max(abs(a-b) for a,b in zip(cb, box)) <= 2]
            if len(candidates) == 1:
                claims[candidates[0]].append((parent_index, region))
    removed = set()
    for child_index, owners in claims.items():
        if len(owners) != 1:
            continue
        parent_index, region = owners[0]
        parent, child = items[parent_index], items[child_index]
        if not _continuation_text_covered(parent, child, path):
            continue
        parent["table_footnote"] = list(dict.fromkeys(s for s in (footnote_of(parent), footnote_of(child)) if s))
        parent["_pdf_table_note_sources"] = (list(parent.get("_pdf_table_note_sources") or [])
                                            + list(child.get("_pdf_table_note_sources") or []))
        parent["_pdf_merged_table_sources"] = list(parent.get("_pdf_merged_table_sources") or []) + [region]
        removed.add(child_index)
    return [item for i, item in enumerate(items) if i not in removed]


def _continuation_text_covered(parent: dict, child: dict, path) -> bool:
    # Retain a placeholder if its native PDF text still contributes evidence
    # missing from the merged OCR table. This is the same span check used by
    # table review; an upstream merge alone cannot prove OCR completeness.
    if path is None:
        return True
    try:
        import pymupdf
        doc = pymupdf.open(path)
    except Exception:
        return True  # No native evidence is available to the review layer either.
    from .pdf_evidence import _norm, _rect, _span_present, table_evidence_text
    with doc:
        page = page_idx(child)
        if page is None or not 1 <= page <= len(doc):
            return False
        page = doc[page - 1]
        spans = [s.get("text", "").strip() for block in page.get_text("dict", clip=_rect(page, bbox(child))).get("blocks", [])
                 for line in block.get("lines", []) for s in line.get("spans", [])]
    if any("\ufffd" in s or "\x00" in s for s in spans):
        return True
    body = table_evidence_text("\n".join([item_text(parent), caption_of(parent), footnote_of(parent), footnote_of(child)]))
    return all(_span_present(s, body) for s in spans
               if len(_norm(s)) >= 2 and not re.fullmatch(r"[\s\-‐‑‒–—]+", s))


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


def _table_region(page: dict, image_name: str) -> dict | None:
    size = page.get("page_size") or []
    if len(size) != 2 or not all(isinstance(v, (int, float)) and v > 0 for v in size):
        return None
    matches = [bbox(block) for block in page.get("preproc_blocks") or []
               if any(name == image_name for name, _ in _tables([block]))]
    if len(matches) != 1 or not matches[0]:
        return None
    box = [v / size[i % 2] * 1000 for i, v in enumerate(matches[0])]
    if not (0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000):
        return None
    return {"page": page_idx(page), "bbox": box}


def _merged_table_ends(payload: dict, *, continuations: dict | None = None) -> dict[tuple[int, str], int]:
    # preproc_blocks retain physical-page fragments; para_blocks may already
    # contain a merged table whose content_list entry names only its first page.
    pages = _middle_pages(payload)
    pages_by_number = {page_idx(p): p for p in pages}
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
            regions = []
            while remaining and end + 1 in originals:
                # Require an exact, unique row-sequence match on the adjacent
                # page. Shared column labels alone never establish provenance.
                candidates = []
                for name, rows in originals[end + 1]:
                    if rows and rows[0] == first[0]:
                        rows = rows[1:]
                    if rows and remaining[:len(rows)] == rows:
                        candidates.append((name, rows))
                if len(candidates) != 1:
                    break
                name, rows = candidates[0]
                remaining = remaining[len(rows):]
                end += 1
                region = _table_region(pages_by_number[end], name)
                if region is not None:
                    regions.append(region)
            if not remaining:
                ends[(start, image_name)] = end
                if continuations is not None:
                    continuations[(start, image_name)] = (merged, regions)
    return ends
