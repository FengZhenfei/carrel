"""Ordered packing within structural boundaries, with per-member provenance."""
from __future__ import annotations

from dataclasses import replace
import math
from typing import Any

from ..models import ParsedBlock
from ..headings import infer_heading_level
from ..utils import count_tokens

PROSE = frozenset({"text", "title", "list", "page", "slide"})


def source_regions(block: ParsedBlock) -> list[dict[str, Any]]:
    regions = block.metadata.get("source_regions") or block.metadata.get("table_source_regions")
    if regions:
        return [dict(r) for r in regions]
    return [{"block_id": block.block_id, "page": block.page_idx, "bbox": block.bbox,
             "slide_idx": block.slide_idx, "source_locator": block.metadata.get("source_locator"),
             "embedded_source": block.metadata.get("embedded_source")}]


def combined_owner(owner: ParsedBlock, members: list[ParsedBlock]) -> ParsedBlock:
    regions: list[dict[str, Any]] = []
    for block in members:
        for region in source_regions(block):
            if region not in regions:
                regions.append(region)
    metadata = {**owner.metadata, "source_regions": regions}
    if any("_chunk_section_paths" in b.metadata for b in members):
        paths = []
        for block in members:
            for path in block.metadata.get("_chunk_section_paths", [list(block.metadata.get("section_path") or [])]):
                if path not in paths:
                    paths.append(path)
        metadata["_chunk_section_paths"] = paths
    pages = [p for b in members for p in (b.page_idx, b.metadata.get("page_end")) if p is not None]
    bbox = owner.bbox
    if pages:
        metadata.update(page_start=min(pages), page_end=max(pages))
        if min(pages) != max(pages):
            bbox = None
        elif all(b.bbox for b in members):
            bbox = [min(b.bbox[0] for b in members), min(b.bbox[1] for b in members),
                    max(b.bbox[2] for b in members), max(b.bbox[3] for b in members)]
    locators = [b.metadata.get("source_locator") or {} for b in members]
    if all(l.get("kind") == "lines" for l in locators):
        metadata["source_locator"] = {"kind": "lines", "start": min(l["start"] for l in locators),
                                      "end": max(l["end"] for l in locators)}
    else:
        metadata.pop("source_locator", None)
    return replace(owner, metadata=metadata, bbox=bbox, page_idx=min(pages) if pages else owner.page_idx)


def same_structure(a: ParsedBlock, b: ParsedBlock) -> bool:
    if (a.slide_idx, a.sheet_name, a.doc_type) != (b.slide_idx, b.sheet_name, b.doc_type):
        return False
    for key in ("section_path", "source_field", "source_fields", "frontmatter", "degraded"):
        left, right = a.metadata.get(key), b.metadata.get(key)
        if key in {"section_path", "source_fields"}:
            left, right = left or [], right or []
        if left != right:
            return False
    # Without a section path, an explicit new title is still a boundary.
    if not a.metadata.get("section_path") and (b.block_type == "title" or a.title != b.title):
        return False
    return True


def _page_boxes(block: ParsedBlock) -> list[list[float]]:
    """Read every source rectangle; a merged block need not have a single bbox."""
    if block.page_idx is None or block.metadata.get("page_end", block.page_idx) != block.page_idx:
        return []
    regions = block.metadata.get("source_regions") or [{"page": block.page_idx, "bbox": block.bbox}]
    boxes = []
    for region in regions:
        box = region.get("bbox")
        if region.get("page") != block.page_idx or not isinstance(box, (list, tuple)) or len(box) != 4:
            return []
        if not all(isinstance(v, (float, int)) and math.isfinite(v) for v in box):
            return []
        if box[0] >= box[2] or box[1] >= box[3]:
            return []
        boxes.append(box)
    return boxes


def owns_code_intro(prose: ParsedBlock, code: ParsedBlock) -> bool:
    if prose.block_type not in PROSE or prose.visual_ref or code.metadata.get("symbol"):
        return False
    if not same_structure(prose, code):
        return False
    parent = (code.metadata.get("embedded_source") or {}).get("block_id")
    if parent:
        return parent == (prose.metadata.get("embedded_source") or {}).get("block_id")
    if prose.page_idx != code.page_idx:
        return False
    if prose.page_idx is not None:
        prose_boxes, code_boxes = _page_boxes(prose), _page_boxes(code)
        if not prose_boxes or not code_boxes:
            return False
        return all(p[3] <= min(c[1] for c in code_boxes) + 5 and
                   any(min(p[2], c[2]) > max(p[0], c[0]) for c in code_boxes)
                   for p in prose_boxes)
    return False


def pack_prose(pieces: list[dict[str, Any]], max_tokens: int) -> list[dict[str, Any]]:
    """Minimize chunk count, then balance slack; never reorder or drop source text.

    Candidates are bounded by the token budget. Images, tables, listings and
    structural transitions terminate a run. Existing oversized atoms stay intact.
    """
    def eligible(p):
        b = p["block"]
        return b.block_type in PROSE and not b.visual_ref and not b.metadata.get("symbol")

    def starts_heading(p):
        lines = [s.strip() for s in p["text"].splitlines() if s.strip() and not s.startswith("TITLE:")]
        return bool(lines and (lines[0] in (p["block"].metadata.get("heading_lines") or [])
                               or infer_heading_level(lines[0]) is not None))

    def pack(run):
        n = len(run)
        costs = [(0, 0)] * (n + 1)
        ends = [0] * n
        for i in range(n - 1, -1, -1):
            best = None
            joined = ""
            for j in range(i, n):
                joined += ("\n" if j > i else "") + run[j]["text"]
                tokens = count_tokens(joined)
                if j > i and tokens > max_tokens:
                    break
                score = (1 + costs[j + 1][0], (max_tokens - tokens) ** 2 + costs[j + 1][1])
                if best is None or score < best:
                    best, ends[i] = score, j + 1
                if tokens >= max_tokens:
                    break
            costs[i] = best
        out = []
        i = 0
        while i < n:
            group = run[i:ends[i]]
            if len(group) == 1:
                out.append(group[0])
            else:
                text = "\n".join(p["text"] for p in group)
                owner = next((p["block"] for p in group if p["block"].block_type != "title"), group[0]["block"])
                out.append({"text": text, "tokens": count_tokens(text),
                            "block": combined_owner(owner, [p["block"] for p in group])})
            i = ends[i]
        return out

    out, run = [], []
    for piece in pieces:
        if run and (not eligible(piece) or starts_heading(piece)
                    or not same_structure(run[-1]["block"], piece["block"])):
            out.extend(pack(run))
            run = []
        if eligible(piece):
            run.append(piece)
        else:
            out.append(piece)
    return out + pack(run)
