"""Chunking diagnostics: statistics + acceptance check of the chunking result after every parse.

Advisory only, never blocking -- parsing and indexing proceed as usual, the verdict is stored in
files.chunk_diag_json for the console to display, and the "chunk preview" uses the same statistics. The
rules borrow from WeKnora's ValidateChunks, with thresholds rewritten for our "split within a block,
never across blocks" model: an image is one chunk per block, so its length is not bounded by max_tokens;
tables are split by row, and the last row or two left over are naturally short, so they take no part in
the fragment verdict.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from ..headings import infer_heading_level, is_short_lead_in
from ..models import ParsedBlock, UnifiedChunk

# Block types whose length is not bounded by max_tokens: images (one per chunk). Tables are not among them:
# a row over the budget is split by cell and a cell over the budget by token, so an over-long table chunk
# means chunking went wrong, and an over-long chunk is rejected by the embedding service
EXEMPT_BLOCK_TYPES = frozenset({"vision", "image", "chart"})
# Types additionally exempt from the fragment verdict: tables are split by row and the last row or two left
# over are short anyway; code is chunked by symbol and small methods are short anyway; slides are chunked by
# page, and a title-only page is still a page
TINY_EXEMPT_BLOCK_TYPES = EXEMPT_BLOCK_TYPES | frozenset({"table", "code", "slide"})
# Over-budget tolerance: sentence packing is off by a few percent, only beyond this factor is it a problem
OVER_TOLERANCE = 1.25
# An image chunk whose text (VLM / MinerU recognition) exceeds the budget by this factor means the
# recognition ran away (repetition, fabrication)
LONG_VISUAL_FACTOR = 2.5
_NOTE_RE = re.compile(r"^\s*(?:注释|注|备注|Notes?|NOTE)\s*[:：]?\s*$", re.IGNORECASE)


def tiny_threshold(max_tokens: int) -> int:
    """Below this token count a chunk is a "fragment": 1/8 of the budget, but no less than 16."""
    return max(16, int(max_tokens) // 8)


def chunk_diagnostics(
    chunks: list[UnifiedChunk],
    *,
    max_tokens: int,
    blocks: list[ParsedBlock] | None = None,
) -> dict[str, Any]:
    tokens = [int(c.token_count) for c in chunks]
    total = sum(tokens)
    n = len(tokens)
    mean = total / n if n else 0.0
    stddev = math.sqrt(sum((t - mean) ** 2 for t in tokens) / n) if n else 0.0
    threshold = tiny_threshold(max_tokens)

    text_tokens = [int(c.token_count) for c in chunks if c.block.block_type not in TINY_EXEMPT_BLOCK_TYPES]
    tiny = sum(1 for t in text_tokens if t < threshold)
    over = sum(1 for c in chunks if c.block.block_type not in EXEMPT_BLOCK_TYPES and int(c.token_count) > max_tokens * OVER_TOLERANCE)
    # Heading-only fragments: the whole chunk is one short heading line (should be gone after the chunking
    # overhaul; kept as a regression signal)
    orphan_headings = sum(1 for c in chunks if c.block.block_type not in TINY_EXEMPT_BLOCK_TYPES
                          and int(c.token_count) < threshold and _heading_only(c.text))
    # Runaway image chunks: text exceeds the budget by LONG_VISUAL_FACTOR
    long_visual = sum(1 for c in chunks if c.block.block_type in {"vision", "image", "chart"}
                      and int(c.token_count) > max_tokens * LONG_VISUAL_FACTOR)
    # Table continuation chunks missing the header: pipe rows without a header row (markdown separator row or
    # HEADER: line)
    table_chunks = [c for c in chunks if c.block.block_type == "table"]
    headerless_tables = sum(1 for c in table_chunks if _table_without_header(c.text))

    by_type: dict[str, dict[str, int]] = {}
    for c in chunks:
        slot = by_type.setdefault(c.block.block_type, {"chunks": 0, "tokens": 0})
        slot["chunks"] += 1
        slot["tokens"] += int(c.token_count)

    stats: dict[str, Any] = {
        "chunks": n,
        "tokens_total": total,
        "tokens_min": min(tokens) if tokens else 0,
        "tokens_max": max(tokens) if tokens else 0,
        "tokens_mean": round(mean, 1),
        "tokens_stddev": round(stddev, 1),
        "tiny_threshold": threshold,
        "tiny_count": tiny,
        "tiny_ratio": round(tiny / len(text_tokens), 3) if text_tokens else 0.0,
        "over_count": over,
        "over_tolerance": OVER_TOLERANCE,
        "orphan_headings": orphan_headings,
        "long_visual": long_visual,
        "headerless_tables": headerless_tables,
        "by_block_type": by_type,
    }
    if blocks is not None:
        # Whether heading detection worked: a document of dozens of pages with every section_path empty
        # means the parser recognized no heading at all and chunking could only cut by length. The depth
        # histogram says more than a plain yes / no.
        depths = Counter(len(b.metadata.get("section_path") or []) for b in blocks)
        stats["blocks"] = len(blocks)
        # In PDF/DOCX, headings get merged into prose blocks (merge_mineru_text_blocks), so count both the
        # standalone heading blocks and the heading lines recorded by merged blocks
        stats["headings"] = (sum(1 for b in blocks if b.block_type == "title")
                             + sum(len(b.metadata.get("heading_lines") or ()) for b in blocks))
        stats["inferred_headings"] = (sum(1 for b in blocks if b.metadata.get("heading_inferred"))
                                      + sum(int(b.metadata.get("inferred_headings") or 0) for b in blocks))
        stats["section_depths"] = {str(k): v for k, v in sorted(depths.items())}

    reasons: list[dict[str, str]] = []

    def flag(key: str, message: str) -> None:
        reasons.append({"key": key, "message": message})

    if n == 0:
        flag("no_chunks", "No chunks were produced")
    elif n == 1 and total > 2 * max_tokens:
        flag("single_oversized", f"The whole document is a single chunk of {total} tokens, over twice the budget")
    if tiny > 2 and tiny * 4 > len(text_tokens):
        flag("fragmented", f"{tiny}/{len(text_tokens)} text chunks are under {threshold} tokens: too fragmented")
    if n > 1 and total > max_tokens and stats["tokens_max"] < max_tokens // 4:
        flag("all_tiny", f"The longest chunk has only {stats['tokens_max']} tokens, far below the budget of {max_tokens}")
    if over:
        flag("oversized", f"{over} text chunks exceed {OVER_TOLERANCE:g}× the budget of {max_tokens} tokens")
    if orphan_headings >= 3:
        flag("orphan_headings", f"{orphan_headings} heading-only fragments")
    if long_visual:
        flag("long_visual", f"{long_visual} images whose recognized text exceeds {LONG_VISUAL_FACTOR:g}× the budget; VLM / MinerU output may have run away")
    if headerless_tables and headerless_tables * 4 > len(table_chunks):
        flag("headerless_tables", f"{headerless_tables}/{len(table_chunks)} table chunks have no header")
    return {"ok": not reasons, "reasons": reasons, "stats": stats}


def _heading_only(text: str) -> bool:
    lines = [l.strip() for l in str(text or "").splitlines() if l.strip() and not l.strip().startswith("TITLE:")]
    if len(lines) != 1:
        return False
    line = lines[0]
    return _NOTE_RE.match(line) is not None or infer_heading_level(line) is not None or is_short_lead_in(line)


def _table_without_header(text: str) -> bool:
    lines = [l.strip() for l in str(text or "").splitlines() if l.strip()]
    pipe_rows = [l for l in lines if l.startswith("|")]
    if len(pipe_rows) < 2:
        return False
    has_sep = any(re.match(r"^\|?\s*:?-{3,}", l) for l in lines)
    has_native = any(l.startswith("HEADER:") for l in lines)
    return not (has_sep or has_native)


def summarize_line(diag: dict[str, Any]) -> str:
    """One log line: the part that follows [parse] chunk-check."""
    s = diag.get("stats") or {}
    verdict = "ok" if diag.get("ok") else ",".join(r["key"] for r in diag.get("reasons") or [])
    return (
        f"verdict={verdict} chunks={s.get('chunks', 0)} tokens={s.get('tokens_total', 0)} "
        f"mean={s.get('tokens_mean', 0)} min={s.get('tokens_min', 0)} max={s.get('tokens_max', 0)} "
        f"tiny={s.get('tiny_count', 0)} over={s.get('over_count', 0)}"
    )
