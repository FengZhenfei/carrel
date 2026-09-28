from __future__ import annotations

import re

from pathlib import Path

from ..chunking.chunker import text_for_block
from ..headings import infer_heading_level, is_note_label, is_page_footer, is_short_lead_in, starts_with_note
from ..models import ParsedBlock
from ..utils import count_tokens
from .mineru_pdf import MinerUServiceError, mineru_pdf_blocks
from .pdf_textlayer import mark_degraded_pages, repair_lost_text_layer
from ..vision.vlm import compose_prompt
from .table_check import high_confidence
from .table_repair import repair_ambiguous_tables
from .visual_blocks import enrich_blocks_with_vlm, ensure_visual_blocks_have_text, tidy_visual_blocks


NOISE_SOURCE_TYPES = {"header", "footer", "page_header", "page_footer", "page_number"}

# Block types that can be merged into the body buffer. equation is here on purpose: MinerU breaks
# every connecting sentence between formulas into a separate text block, and if formulas were not
# mergeable the buffer would be flushed at every formula -- "where / $$formula$$ / therefore" would
# become three chunks, two of them a single token. Measured on the library corpus, 1,424 of 44,043
# chunks were <=5 tokens and 51% were under 100 tokens (limit 800). The other end is worse: 14,338
# formula chunks (median 83 tokens) were cut off from the prose explaining them, and nobody searches
# in LaTeX -- what makes a formula findable is precisely the two sentences around it.
#
# table / image / chart remain non-mergeable: they are large on their own, have their own
# 2048-dimensional visual vector, and retrieving them individually is meaningful.
MERGEABLE_TYPES = {"text", "title", "page", "equation"}


def parse_pdf_enhanced(
    *,
    mineru_url: str,
    vlm_base_url: str,
    vlm_api_key: str,
    vlm_model_id: str,
    vlm_concurrency: int,
    path: Path,
    cache_dir: Path,
    merge_target_tokens: int = 800,
    timeout: int = 43200,
    vlm_options: dict | None = None,
    caption_cache_root: Path | None = None,
    vlm_prompt: str | None = None,
    progress_cb=None,
) -> list[ParsedBlock]:
    print(f"[parser] pdf mineru start file={path.name}", flush=True)
    try:
        blocks = mineru_pdf_blocks(mineru_url=mineru_url, path=path, cache_dir=cache_dir, timeout=timeout)
        print(f"[parser] pdf mineru done blocks={len(blocks)} file={path.name}", flush=True)
    except MinerUServiceError as exc:
        print(f"[parser] pdf mineru failed file={path.name} error={exc!r}", flush=True)
        raise

    # Text-layer cross-check: when MinerU cannot render non-embedded Chinese fonts whole pages lose their CJK text;
    # render those pages with PyMuPDF and send them through MinerU once more (pdf_textlayer)
    blocks, layer_info = repair_lost_text_layer(
        blocks, path=path, rendered_path=cache_dir / "rendered" / f"{path.stem}.lost-pages.pdf",
        parse_rendered=lambda rendered: mineru_pdf_blocks(mineru_url=mineru_url, path=rendered, cache_dir=cache_dir / "rendered",
                                                          timeout=timeout))
    if layer_info.get("lost_before"):
        print(f"[parser] pdf text-layer cjk lost pages={layer_info['lost_before']} rendered={layer_info['rendered']} "
              f"recovered={layer_info['recovered']} still_lost={layer_info['lost']}"
              + (f" error={layer_info['error']}" if layer_info.get("error") else "") + f" file={path.name}", flush=True)

    blocks = merge_mineru_text_blocks(blocks, target_tokens=merge_target_tokens)
    blocks = merge_split_tables(blocks)
    if layer_info.get("lost"):
        marked = mark_degraded_pages(blocks, layer_info["lost"])
        print(f"[parser] pdf degraded blocks={marked} pages={layer_info['lost']} file={path.name}", flush=True)

    image_blocks = [block for block in blocks if block.block_type in {"image", "chart"} and block.visual_ref]
    print(f"[parser] pdf vlm candidates={len(image_blocks)} file={path.name}", flush=True)
    enrich_blocks_with_vlm(
        image_blocks,
        base_url=vlm_base_url,
        api_key=vlm_api_key,
        model_id=vlm_model_id,
        cache_dir=cache_dir,
        concurrency=vlm_concurrency,
        prompt=compose_prompt("figure", vlm_prompt),
        filter_decorative=True,
        vlm_options=vlm_options,
        caption_cache_root=caption_cache_root,
        progress_cb=progress_cb,
    )
    tidy_visual_blocks(image_blocks, max_tokens=max(100, merge_target_tokens // 2))
    ensure_visual_blocks_have_text(image_blocks, doc_name=path.name)
    # Tables flagged as having merged cells: give the VLM MinerU's table screenshot and have it
    # transcribe only those rows; split them only when the concatenation matches (F02)
    table_blocks = [b for b in blocks if b.block_type == "table" and b.metadata.get("table_flags")]
    if table_blocks:
        stats = repair_ambiguous_tables(table_blocks, base_url=vlm_base_url, api_key=vlm_api_key, model_id=vlm_model_id,
                                        cache_dir=cache_dir, vlm_options=vlm_options, caption_cache_root=caption_cache_root)
        print(f"[parser] pdf table-repair {stats} file={path.name}", flush=True)
    return blocks


# ── Merging tables split across pages ─────────────────────────────────────────
# MinerU emits tables per page: the part of a lab report continuing on the next page has no header
# ("| Total protein | TP | 73.4 | ... |"), or the header is printed again. The fragment becomes a block
# of its own, and rows without their header are meaningless (2026-09-07 health knowledge base audit:
# 5 header-less tables in a 2026 check-up report). Two adjacent table blocks with the same column
# count, on the same or consecutive pages, where the second has no title of its own and its first row
# either equals the first table's header or looks like a data row, are merged back into the first.
# Tables flagged for merged cells are not merged (screenshot verification aligns rows on the original
# grid).

def _table_lines(md: str) -> tuple[list[str], list[str], list[str]]:
    """Table text -> (lines before, table rows, lines after). Before is TITLE / CAPTION, after is FOOTNOTE."""
    lines = (md or "").splitlines()
    first = next((i for i, l in enumerate(lines) if l.lstrip().startswith("|")), None)
    if first is None:
        return lines, [], []
    last = max(i for i, l in enumerate(lines) if l.lstrip().startswith("|"))
    return lines[:first], [l.strip() for l in lines[first:last + 1] if l.strip().startswith("|")], lines[last + 1:]


def _cells(row: str) -> list[str]:
    return [c.strip() for c in row.strip().strip("|").split("|")]


def _is_separator(row: str) -> bool:
    cells = _cells(row)
    return bool(cells) and all(c and set(c) <= set("-: ") for c in cells)


def _looks_like_data(cells: list[str]) -> bool:
    return any(re.search(r"\d", c) for c in cells)


def merge_split_tables(blocks: list[ParsedBlock]) -> list[ParsedBlock]:
    out: list[ParsedBlock] = []
    for block in blocks:
        prev = out[-1] if out else None
        if (prev is None or block.block_type != "table" or prev.block_type != "table"
                or not prev.table_markdown or not block.table_markdown
                or block.caption or block.title
                or high_confidence(prev.metadata.get("table_flags")) or high_confidence(block.metadata.get("table_flags"))):
            out.append(block)
            continue
        if block.page_idx is not None and prev.page_idx is not None and block.page_idx not in (prev.page_idx, prev.page_idx + 1):
            out.append(block)
            continue
        pre_a, rows_a, post_a = _table_lines(prev.table_markdown)
        pre_b, rows_b, post_b = _table_lines(block.table_markdown)
        if len(rows_a) < 2 or not rows_b or not _is_separator(rows_a[1]):
            out.append(block)
            continue
        header = _cells(rows_a[0])
        first_b = _cells(rows_b[0])
        if len(first_b) != len(header):
            out.append(block)
            continue
        # Note: html_table_to_markdown always treats the first row as the header and follows it with a
        # separator line, continuation tables included -- so "first row looks like data" is the real
        # sign of a continuation, and separator lines are always stripped
        if first_b == header:
            body_b = [r for r in rows_b[1:] if not _is_separator(r)]        # repeated header: drop it
        elif not _is_separator(rows_b[0]) and _looks_like_data(first_b):
            body_b = [r for r in rows_b if not _is_separator(r)]           # header-less continuation
        else:
            out.append(block)
            continue
        if not body_b:
            continue
        extra_post = [l for l in post_b if l.strip() and l not in post_a]  # a page footer taken as a table note is identical in both parts; do not repeat it
        merged = "\n".join(pre_a + rows_a + body_b + post_a + extra_post)
        prev.table_markdown = merged
        prev.text = merged
        prev.metadata.setdefault("merged_tables", []).append(block.block_id)
        if block.page_idx is not None:
            prev.metadata["page_end"] = block.page_idx
    return out


def min_merge_tokens(target_tokens: int) -> int:
    """A buffer below this count does not become a block of its own; it folds back into the previous
    merged block.

    Folding back is only possible when the buffer sits **directly** after the previous merged block --
    not when a table or image lies in between, because that would scramble the order, and the
    render-ready document is assembled exactly in block order. So this rule catches the remainder
    after an overflow (750+100 exceeds 800 -> flush 750, the remaining 100 would become a block on its
    own), not short prose sandwiched between tables; the latter can only be helped by MERGEABLE_TYPES
    flushing less often at the source.
    """
    return max(1, target_tokens // 8)


def merge_mineru_text_blocks(
    blocks: list[ParsedBlock],
    *,
    target_tokens: int,
    id_prefix: str = "mineru-merged-text",
) -> list[ParsedBlock]:
    """Merge MinerU body blocks up to target_tokens; tables / images are not mergeable and act as the
    boundaries of the merge stream.

    Rules at a boundary (refined after the 2026-09-05 sampling, see the chunking optimisation plan on
    the desktop):
      - Title blocks and one short lead-in line at the tail of the buffer go to the following
        figure/table as its TITLE. A title block goes to it whether or not the figure/table has its own
        caption (the title names the section, the figure/table is the section's first content); a short
        lead-in goes to it only when the figure/table has no caption. Paragraphs earlier in the buffer
        do **not** follow into the TITLE -- they are emitted as body blocks as usual (short ones fold
        back into the previous merged block).
      - A short buffer starting with "Note: ..." attaches to the **previous** figure/table as its
        FOOTNOTE (a note belongs to the figure / table above it, not to the start of the next section);
        if only the "Note:" label remains because the parser dropped the body, the whole thing is
        discarded.
      - Other short buffers fold back into the previous merged block (overflowed prose tails), in line
        with the min_merge_tokens description.
      - Short lines before a title (table-of-contents entries such as "Technical support....18",
        parameter lines) do not follow the title into the TITLE; inline icons are not boundaries.
    """
    if not target_tokens or target_tokens <= 0:
        return blocks

    floor_tokens = min_merge_tokens(target_tokens)
    merged: list[ParsedBlock] = []
    buf: list[ParsedBlock] = []
    buf_tokens = 0
    merged_idx = 0
    dropped_noise = 0
    folded = 0
    adopted = 0
    footnoted = 0
    dropped_labels = 0
    dropped_icons = 0

    def rendered(block: ParsedBlock) -> str:
        """What goes into the buffer is the **rendered** text, not block.text.

        A formula's body lives in latex; block.text is either empty or the same latex -- joining
        block.text directly would lose the "EQUATION: " marker meant for the extraction model.
        text_for_block's output for an ordinary body block is exactly block.text.strip(), so this
        change is equivalent to the old behaviour and merely wires the formulas in.
        """
        return text_for_block(block).strip()

    def is_lead_in(block: ParsedBlock) -> bool:
        return block.block_type == "title" or is_short_lead_in(rendered(block))

    def emit(run: list[ParsedBlock]) -> None:
        """Emit a buffer run as one merged block; if it is too short and directly follows the previous
        merged block, fold it back into that one."""
        nonlocal merged_idx, folded
        if not run:
            return
        text = "\n\n".join(part for part in (rendered(block) for block in run) if part)
        if not text:
            return
        tokens = count_tokens(text)
        pages = [block.page_idx for block in run if block.page_idx is not None]
        source_types = [source_type(block) for block in run if source_type(block)]
        # Record the merged-in heading lines verbatim: after merging they are just lines in the body,
        # and the chunker relies on this list to know where to break (headings of chunker.chunk_text)
        heading_lines = [rendered(block) for block in run if block.block_type == "title" and rendered(block)]
        inferred_headings = sum(1 for block in run if block.metadata.get("heading_inferred"))

        previous = merged[-1] if merged else None
        if tokens < floor_tokens and previous is not None and previous.metadata.get("merged"):
            # Fold back into the previous merged block. One block fewer, and the block_id stays the
            # previous one -- downstream chunk_uid is keyed by block_id, so no new id may be invented here.
            folded += 1
            previous.text = f"{previous.text}\n\n{text}" if previous.text else text
            previous.metadata["merged_block_count"] += len(run)
            previous.metadata["source_block_ids"].extend(block.block_id for block in run)
            previous.metadata["source_types"].extend(source_types)
            previous.metadata.setdefault("heading_lines", []).extend(heading_lines)
            previous.metadata["inferred_headings"] = previous.metadata.get("inferred_headings", 0) + inferred_headings
            if pages:
                starts = [p for p in (previous.metadata.get("page_start"), min(pages)) if p is not None]
                ends = [p for p in (previous.metadata.get("page_end"), max(pages)) if p is not None]
                previous.metadata["page_start"] = min(starts) if starts else None
                previous.metadata["page_end"] = max(ends) if ends else None
            return

        merged_idx += 1
        template = run[0]
        merged.append(
            ParsedBlock(
                parser=template.parser,
                parser_profile=f"{template.parser_profile}+text-merge-v3",
                doc_type=template.doc_type,
                block_type="text",
                text=text,
                block_id=f"{id_prefix}-{merged_idx:05d}",
                page_idx=min(pages) if pages else template.page_idx,
                title=template.title,
                metadata={
                    "section_path": list(template.metadata.get("section_path") or []),
                    "merged": True,
                    "merged_block_count": len(run),
                    "source_block_ids": [block.block_id for block in run],
                    "source_types": source_types,
                    "heading_lines": heading_lines,
                    "inferred_headings": inferred_headings,
                    "page_start": min(pages) if pages else None,
                    "page_end": max(pages) if pages else None,
                },
            )
        )

    def flush() -> None:
        nonlocal buf, buf_tokens
        emit(buf)
        buf = []
        buf_tokens = 0

    def attach_footnote(target: ParsedBlock, run: list[ParsedBlock], text: str) -> None:
        nonlocal footnoted
        target.text = f"{target.text}\nFOOTNOTE: {text}".strip() if (target.text or "").strip() else f"FOOTNOTE: {text}"
        if target.block_type == "table" and target.table_markdown is not None:
            # A table block's text must match table_markdown, otherwise text_for_block emits the whole
            # table a second time
            target.table_markdown = target.text
        target.metadata.setdefault("footnote_block_ids", []).extend(block.block_id for block in run)
        footnoted += 1

    def hand_over(target: ParsedBlock) -> None:
        """A non-mergeable block arrives: the title / lead-in at the tail of the buffer becomes its
        TITLE, a note becomes the FOOTNOTE of the previous figure/table, and the rest is emitted as
        usual (short runs fold back into the previous merged block)."""
        nonlocal buf, buf_tokens, adopted, dropped_labels
        if not buf:
            return
        previous = merged[-1] if merged else None
        # 1) Title + short lead-in at the tail (at most 3 blocks)
        tail: list[ParsedBlock] = []
        while buf and len(tail) < 3 and is_lead_in(buf[-1]):
            tail.insert(0, buf.pop())
        if tail:
            has_title = any(block.block_type == "title" for block in tail)
            if has_title:
                # Short lines before the title (table-of-contents entries, parameter lines) are not
                # lead-ins; they stay in the body
                while tail and tail[0].block_type != "title":
                    buf.append(tail.pop(0))
            titles_only = all(block.block_type == "title" for block in tail)
            # A prose tail directly after a just-emitted merged block is not a lead-in: fold it back into
            # the previous run (consistent with min_merge_tokens)
            prose_tail = not has_title and not buf and previous is not None and previous.metadata.get("merged")
            can_adopt = (
                not prose_tail
                and not (target.title or "").strip()
                and (has_title or not (target.caption or "").strip())
            )
            if can_adopt:
                text = "\n\n".join(part for part in (rendered(block) for block in tail) if part)
                if text:
                    target.title = text
                    target.metadata["adopted_lead_in"] = True
                    target.metadata["adopted_block_ids"] = [block.block_id for block in tail]
                    target.metadata["adopted_titles_only"] = titles_only
                    adopted += 1
                    tail = []
            if tail:
                buf.extend(tail)
        if not buf:
            buf_tokens = 0
            return
        # 2) Notes: a short buffer starting with "Note: ..." belongs to the previous figure/table
        text_all = "\n\n".join(part for part in (rendered(block) for block in buf) if part)
        tokens_all = count_tokens(text_all) if text_all else 0
        if text_all and tokens_all < floor_tokens and starts_with_note(text_all):
            if is_note_label(text_all):
                dropped_labels += 1
                buf = []
                buf_tokens = 0
                return
            if previous is not None and previous.block_type in {"table", "image", "chart"}:
                attach_footnote(previous, buf, text_all)
                buf = []
                buf_tokens = 0
                return
        # 3) Everything else as usual
        flush()

    for block in blocks:
        if is_noise_block(block):
            dropped_noise += 1
            continue
        if is_inline_icon(block):
            # Inline icons (bbox smaller than ICON_MAX_PT square): nothing retrievable in them, and they
            # would cut a paragraph in half and steal the preceding heading as their TITLE. The VLM side
            # filters by size the same way, so here they simply are not boundaries
            dropped_icons += 1
            continue
        if not is_mergeable_text_block(block):
            hand_over(block)
            merged.append(block)
            continue
        text = rendered(block)
        tokens = count_tokens(text)
        # A "Note: ..." directly after a figure/table attaches to it as FOOTNOTE right away instead of
        # waiting for the next boundary -- otherwise a note in the page footer would be merged with the
        # next page's heading. A lone "Note:" label block waits for its body; if a heading arrives
        # instead, the label is dropped.
        previous = merged[-1] if merged else None
        labels_only = bool(buf) and all(is_note_label(rendered(b)) for b in buf)
        if previous is not None and previous.block_type in {"table", "image", "chart"} and (not buf or labels_only):
            if labels_only and (block.block_type == "title" or infer_heading_level(text) is not None):
                dropped_labels += 1
                buf = []
                buf_tokens = 0
            else:
                joined = " ".join(part for part in (rendered(b) for b in buf + [block]) if part)
                if starts_with_note(joined) and count_tokens(joined) < floor_tokens:
                    if is_note_label(joined):
                        buf.append(block)
                        buf_tokens += tokens
                        continue
                    attach_footnote(previous, buf + [block], joined)
                    buf = []
                    buf_tokens = 0
                    continue
        if buf and buf_tokens + tokens > target_tokens:
            flush()
        buf.append(block)
        buf_tokens += tokens
        if buf_tokens >= target_tokens:
            flush()
    # End of document: drop a note that is only a label, emit the rest
    if buf:
        text_all = "\n\n".join(part for part in (rendered(block) for block in buf) if part)
        if text_all and is_note_label(text_all):
            dropped_labels += 1
            buf = []
    flush()
    print(
        f"[parser] pdf merge-text output={len(merged)} dropped_noise={dropped_noise} "
        f"folded_short={folded} adopted_lead_in={adopted} footnotes_attached={footnoted} "
        f"label_only_dropped={dropped_labels} icons_dropped={dropped_icons} target_tokens={target_tokens} floor_tokens={floor_tokens}",
        flush=True,
    )
    return merged

ICON_MAX_PT = 48


def is_inline_icon(block: ParsedBlock) -> bool:
    """MinerU bboxes are page coordinates (pt); an image under ICON_MAX_PT on both sides is an inline
    icon / bullet."""
    if block.block_type != "image" or not block.bbox or len(block.bbox) < 4:
        return False
    try:
        x0, y0, x1, y1 = (float(v) for v in block.bbox[:4])
    except (TypeError, ValueError):
        return False
    return (x1 - x0) < ICON_MAX_PT and (y1 - y0) < ICON_MAX_PT


def source_type(block: ParsedBlock) -> str:
    value = block.metadata.get("source_type") or block.metadata.get("label") or ""
    return str(value).strip().lower()


def is_noise_block(block: ParsedBlock) -> bool:
    if source_type(block) in NOISE_SOURCE_TYPES:
        return True
    # MinerU occasionally takes a page footer for body text: a short block whose whole line is
    # "Page 3 of 12 / a CJK page number / - 12 -"
    return block.block_type == "text" and is_page_footer(block.text or "")


def is_mergeable_text_block(block: ParsedBlock) -> bool:
    if block.block_type not in MERGEABLE_TYPES:
        return False
    value = source_type(block)
    if value in NOISE_SOURCE_TYPES or "code" in value:
        return False
    return bool(text_for_block(block).strip())
