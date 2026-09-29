"""Chunking: blocks -> chunks of at most max_tokens.

Overhaul after the 2026-09-05 sampling audit (plan: the chunking-logic optimization document in the
2026-09-05 chunk-sampling folder on the Desktop):
  · Tables are split by row and every chunk carries the header (TITLE/CAPTION, SHEET/ROWS/HEADER lines,
    and for markdown tables the header row + separator row); a row over the budget is split by cell, each
    cell carrying its column name (such chunks no longer carry the header lines that spell out the column
    names), and a cell that alone exceeds the budget is split further by sentence, then by token; in a
    table whose header takes up most of the budget, every row is written this way;
  · Code fences and pipe tables inside prose are atomic segments: they travel with the prose whole when
    they fit, and are split by line on their own only when they do not (a single line over the budget is
    split by token);
  · A fallback pass after chunking: fragments (below diagnose.tiny_threshold) are merged into an adjacent
    chunk of the same section (preferring the following one), never across tables / images; a
    heading-only fragment right before a figure or table becomes that block's TITLE;
  · Code is grouped by class: class header + methods become one chunk when they fit, otherwise split by
    method;
  · Decorative images never form their own chunk; only one description line is left in the preceding
    prose chunk.
"""
from __future__ import annotations

import re
from typing import Any

from ..headings import clean_heading_text, infer_heading_level, is_short_lead_in
from ..models import ParsedBlock, UnifiedChunk
from ..utils import count_tokens, split_sentences

VISUAL_TYPES = frozenset({"vision", "image", "chart"})
TEXT_LIKE_TYPES = frozenset({"text", "title", "list", "equation", "page", "slide"})
# How far a merged chunk may exceed the budget after chunking: used when gluing fragments into a neighbour
GLUE_SLACK = 1.25
_FENCE_OPEN_RE = re.compile(r"^\s*(```|~~~)")
_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}")
_CELL_BAR_RE = re.compile(r"(?<!\\)\|")          # the bars separating the cells of a pipe table; a bar inside a cell is written \|
_NATIVE_TABLE_PREFIXES = ("SHEET:", "ROWS:", "HEADER:")
_HEAD_PREFIXES = ("TITLE:", "CAPTION:") + _NATIVE_TABLE_PREFIXES


def _split_long_sentence(sentence: str, max_tokens: int) -> list[str]:
    """Split a single over-long sentence on real token counts.

    Slices only at character boundaries -- binary searching the longest prefix
    that still fits -- so no multi-byte character is ever cut, while the pieces
    honour the same token budget as the rest of the chunker.
    """
    limit = max(1, max_tokens - 1)
    parts: list[str] = []
    start = 0
    total = len(sentence)
    # No prefix of ~8 chars/token or longer is needed to fill the budget;
    # without the cap a multi-MB single-line file (minified JS/JSON)
    # re-tokenised megabyte prefixes ~20x per piece -- O(n^2), hours of CPU.
    window = max(16, limit * 8)
    while start < total:
        lo, hi = 1, min(total - start, window)
        best = 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if count_tokens(sentence[start : start + mid]) <= limit:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        piece = sentence[start : start + best].strip()
        if piece:
            parts.append(piece)
        start += best
    return parts


# ── Atomic segments: code fences, pipe tables ─────────────────────────────

def _segments(text: str) -> list[tuple[str, str]]:
    """Split text into a sequence of ('prose' | 'fence' | 'table', segment text). Fences and runs of
    consecutive pipe-table lines are kept whole."""
    out: list[tuple[str, str]] = []
    prose: list[str] = []
    lines = text.splitlines()
    i = 0

    def flush_prose() -> None:
        if prose:
            out.append(("prose", "\n".join(prose)))
            prose.clear()

    while i < len(lines):
        line = lines[i]
        if _FENCE_OPEN_RE.match(line):
            fence = _FENCE_OPEN_RE.match(line).group(1)
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith(fence):
                j += 1
            flush_prose()
            out.append(("fence", "\n".join(lines[i : min(j + 1, len(lines))])))
            i = j + 1
            continue
        if _TABLE_LINE_RE.match(line):
            j = i
            while j < len(lines) and _TABLE_LINE_RE.match(lines[j]):
                j += 1
            if j - i >= 2:
                flush_prose()
                out.append(("table", "\n".join(lines[i:j])))
                i = j
                continue
        prose.append(line)
        i += 1
    flush_prose()
    return out


_ROW_IDENT_MAX_CHARS = 40


def _native_cells(line: str) -> list[str]:
    """Split a row of a native table block (or the part of a HEADER line after the colon) into cells by
    position: separated by " | ", without an outer frame. When the first cell is empty the whole line starts
    with "| " (native_table._joined_cells strips the leading whitespace); the empty cell is restored first,
    otherwise every later cell would shift one column to the left."""
    s = line.strip()
    s = (" " + s if s.startswith("|") else s) + (" " if s.endswith("|") else "")
    return [c.strip() for c in s.split(" | ")]


def _framed_cells(line: str) -> list[str]:
    """Strip the outer frame of a pipe-table row and split it into cells by position."""
    return [c.strip() for c in _CELL_BAR_RE.split(line.strip().strip("|"))]


def _table_header_cells(lines: list[str]) -> list[str]:
    """Column names of the header: the markdown header row (| a | b |) or the "HEADER: a | b" line of a native
    table block; an empty list when there is none."""
    for line in lines:
        s = line.strip()
        if s.startswith("HEADER:"):
            return _native_cells(s[len("HEADER:"):])
        if _TABLE_LINE_RE.match(s) and not _TABLE_SEP_RE.match(s):
            return _framed_cells(s)
    return []


def _split_long_line(text: str, max_tokens: int) -> list[str]:
    """Split one long line of text (a long table cell, a table note) into pieces of at most max_tokens: pack by
    sentence first, and split a sentence that alone exceeds the budget by token. No newline is added inside a
    piece, so one record stays one line."""
    parts: list[str] = []
    cur = ""
    for sentence in split_sentences(text):
        gap = "" if not cur or cur[-1] in "。！？；" else " "       # no space follows Chinese sentence-final punctuation
        joined = cur + gap + sentence
        if count_tokens(joined) <= max_tokens:
            cur = joined
            continue
        if cur:
            parts.append(cur)
        cur = sentence
        if count_tokens(cur) > max_tokens:
            *done, cur = _split_long_sentence(cur, max_tokens)
            parts.extend(done)
    return parts + [cur] if cur else parts


def pack_labelled_cells(items: list[str], budget: int) -> list[str]:
    """Pack labelled cells ("name: value") into lines "name: value | name: value" within the budget; shared by
    native tables (native_table.split_long_row) and pipe tables. From the second line on, each line starts
    with the row identifier repeated (the first cell, only when its value is short); a line holding nothing
    but the row identifier is not emitted, since the identifier opens every following line anyway. A cell
    that does not fit by itself has its value split into pieces, each carrying the row identifier and the
    column name -- a whole cell kept as one line would leave the chunk length unbounded, and once it exceeds
    the limit of the embedding service the whole file cannot be indexed."""
    ident = items[0] if items and len(items[0].partition(": ")[2]) <= _ROW_IDENT_MAX_CHARS else ""
    lines: list[str] = []
    cur: list[str] = []
    for item in items:
        if cur and count_tokens(" | ".join(cur + [item])) <= budget:
            cur.append(item)
            continue
        bare_ident = cur == [ident]
        if cur and not bare_ident:
            lines.append(" | ".join(cur))
        start = [ident] if ident and item != ident else []
        cur = start + [item]
        if count_tokens(" | ".join(cur)) <= budget:
            continue
        name, sep, text = item.partition(": ")
        lead = " | ".join(start + [name + sep])
        room = budget - count_tokens(lead)
        if sep and room >= budget // 2:
            parts = [lead + part for part in _split_long_line(text, room)]
        else:
            # The row identifier and the column name alone take more than half the budget: this cell goes
            # without the row identifier and, when it still does not fit, is split as one line of text
            if bare_ident and start:
                lines.append(ident)
            parts = _split_long_line(item, budget)
        lines.extend(parts[:-1])
        cur = parts[-1:]          # the last piece is usually not full; the following cells continue on it
    if cur and (cur != [ident] or not lines):
        lines.append(" | ".join(cur))
    return lines


def _labelled_cells(line: str, header_cells: list[str] | None, native: bool) -> list[str]:
    """Split a row by position and write each cell as "name: value": a column without a name in the header is
    called "column N", and empty cells are skipped. In a native table block without a HEADER line the rows
    were already expanded with column names by native_table (empty cells skipped, positions unrelated to
    the header): those cells are used as they are, without pairing column names a second time."""
    raw = _native_cells(line) if native else _framed_cells(line)
    if native and not header_cells:
        return [c for c in raw if c]
    items = []
    for idx, text in enumerate(raw):
        if not text:
            continue
        name = (header_cells[idx] if header_cells and idx < len(header_cells) and header_cells[idx].strip() else f"column {idx + 1}")
        items.append(f"{name}: {text}")
    return items


def _split_table_row(line: str, budget: int, header_cells: list[str] | None = None, *, native: bool = False) -> list[str]:
    """A row over budget: split it into several lines by cell, each cell carrying its column name ("column N"
    when there is no header), and the following lines start with the row identifier (the first non-empty
    cell) repeated. Empty cells are skipped but the column names keep the positions clear -- empty cells
    used to be squeezed out, so continuation lines no longer lined up with the header (2026-09-13 Codex
    F01, the same rule as for native tables). The split lines do not keep the outer frame of a pipe table:
    the chunk they land in carries no header row, so it is no longer a pipe table."""
    items = _labelled_cells(line, header_cells, native)
    return pack_labelled_cells(items, budget) if items else [line]


def split_table_text(text: str, max_tokens: int) -> list[str]:
    """Split a table by row, every chunk carrying the header. Header = the leading TITLE/CAPTION lines, the
    SHEET/ROWS/HEADER lines, and for markdown tables the header row + separator row. The whole table is
    one chunk when it fits.
    A row over the budget is split by cell with each cell preceded by its own column name, and such chunks
    leave out the header lines that spell out the column names (the HEADER line, the markdown header row +
    separator row): for them the whole header is mere repetition that eats into the budget for the body.
    When the header takes up more than three quarters of the budget, carrying it in every chunk leaves no
    room for data (a 40-column questionnaire table had 700-odd tokens of header per chunk next to a few
    dozen tokens of data, so the vector was decided almost entirely by the header): then every row is
    written this way."""
    if count_tokens(text) <= max_tokens:
        return [text]
    lines = text.splitlines()
    # Every non-table line before the table is header context: besides TITLE / CAPTION there is the second
    # line of a MinerU table note ("operator: xxx reviewer: xxx"). This used to collect only prefixed lines
    # and stopped at such a lead-in line, so the header rows never made it into the later chunks
    # (2026-09-07 health KB sampling: lab report continuation chunks were all numeric rows without column
    # names). When the lead-in is too long, keep only the prefixed lines and put the rest into the first
    # chunk as body lines.
    first_table = next((k for k, l in enumerate(lines) if _TABLE_LINE_RE.match(l)), len(lines))
    lead = [l for l in lines[:first_table] if l.strip()]
    extra_body: list[str] = []
    if count_tokens("\n".join(lead)) > max_tokens // 3:
        extra_body = [l for l in lead if not l.lstrip().startswith(_HEAD_PREFIXES)]
        lead = [l for l in lead if l.lstrip().startswith(_HEAD_PREFIXES)]
    columns = [l for l in lead if l.lstrip().startswith("HEADER:")]         # the header lines that name the columns
    context = [l for l in lead if l not in columns]                         # the rest: title, caption, SHEET / ROWS
    body = extra_body + lines[first_table:]
    if len(body) >= len(extra_body) + 2 and _TABLE_LINE_RE.match(body[len(extra_body)]) and _TABLE_SEP_RE.match(body[len(extra_body) + 1]):
        columns = body[len(extra_body):len(extra_body) + 2]
        body = extra_body + body[len(extra_body) + 2:]
    header_cells = _table_header_cells(columns)
    native = any(l.lstrip().startswith(_NATIVE_TABLE_PREFIXES) for l in lead)
    # Every line of a native table block is a data row; in other tables a line without a bar is a table note, a
    # footnote or original text that was not turned into a table: not split by cell, no column names
    entries = [(l, native or "|" in l) for l in body if l.strip()]

    def room(head_lines: list[str]) -> int:
        return max_tokens - (count_tokens("\n".join(head_lines)) + 1 if head_lines else 0)

    floor = max_tokens // 4
    while context and room(context) < floor:
        # Title / caption take up most of the budget: the longest line is no longer carried by every chunk; it
        # appears once at the start, ahead of the table, and is split like body text
        context.remove(max(context, key=count_tokens))
    opening = [l for l in lead if l not in columns and l not in context]
    has_rows = any(is_row for _, is_row in entries)
    entries = [(l, False) for l in opening] + entries
    expand_all = bool(columns) and has_rows and room(context + columns) < floor
    if columns and not has_rows:
        # A header and no data rows (a "table" whose single cell holds a whole passage): the header is the
        # content, split it as body text
        at = len(opening) + len(extra_body)
        entries[at:at] = [(l, False) for l in columns if not _TABLE_SEP_RE.match(l)]
    if expand_all or not has_rows:
        columns = []
    budgets = {False: room(context + columns), True: room(context)}          # key: the chunk leaves out the column header lines (True) or not
    groups: list[tuple[bool, list[str]]] = []
    cur: list[str] = []
    cur_tokens = 0
    cur_named = expand_all
    for k, (line, is_row) in enumerate(entries):
        fits = count_tokens(line) + 1 <= budgets[False]
        if is_row:
            named = expand_all or not fits
            parts = _split_table_row(line, budgets[True] - 1, header_cells, native=native) if named else [line]
        else:
            named = k < len(opening) or (cur_named if cur else expand_all)   # table notes and footnotes follow the chunk before them
            parts = [line] if count_tokens(line) + 1 <= budgets[named] else _split_long_line(line, budgets[named] - 1)
        for part in parts:
            t = count_tokens(part) + 1              # +1: the newline when joining back
            if cur and (named != cur_named or cur_tokens + t > budgets[named]):
                groups.append((cur_named, cur))
                cur, cur_tokens = [], 0
            cur.append(part)
            cur_tokens += t
            cur_named = named
    if cur:
        groups.append((cur_named, cur))
    if not groups:
        return [text]
    return ["\n".join(context + ([] if named else columns) + rows) for named, rows in groups]


def _split_fence(text: str, max_tokens: int) -> list[str]:
    """A code fence that does not fit: split by line, each piece re-wrapped with the opening fence line and the
    closing fence. A single line over the budget (JSON squashed into one line, base64) is split by token into
    several lines."""
    lines = text.splitlines()
    if not lines:
        return []
    opener = lines[0]
    closer = lines[-1] if len(lines) > 1 and _FENCE_OPEN_RE.match(lines[-1]) else "```"
    body = lines[1:-1] if len(lines) > 1 and _FENCE_OPEN_RE.match(lines[-1]) else lines[1:]
    overhead = count_tokens(opener) + count_tokens(closer) + 2
    budget = max(max_tokens // 4, max_tokens - overhead)
    pieces: list[str] = []
    cur: list[str] = []
    cur_tokens = 0
    for line in body:
        for part in ([line] if count_tokens(line) + 1 <= budget else _split_long_sentence(line, budget)):
            t = count_tokens(part) + 1
            if cur and cur_tokens + t > budget:
                pieces.append("\n".join([opener, *cur, closer]))
                cur, cur_tokens = [], 0
            cur.append(part)
            cur_tokens += t
    if cur:
        pieces.append("\n".join([opener, *cur, closer]))
    return pieces or [text]


# ── Prose ─────────────────────────────────────────────────────────────────

def _chunk_prose(
    units: list[tuple[str, bool]],
    max_tokens: int,
    overlap_tokens: int,
    *,
    headings: frozenset[str] = frozenset(),
) -> list[str]:
    """Greedy sentence packing with whole-sentence overlap; heading lines are structural cut points.
    units = [(text, is_atomic)].

    headings are the known heading lines of this text (MinerU headings that merged blocks record in
    metadata.heading_lines); whole lines shaped like a Chinese "Chapter N" heading / 1.2.3 / Chapter N
    count too. Two rules:
      · A heading arrives while the current chunk is already past half -> cut before the heading,
        **without overlap** (the new chunk starts at the heading; the tail of the previous section must
        not bleed in); below half, keep packing so small sections stay together
      · When cutting for length and the chunk ends with a heading line, move it to the start of the next
        chunk -- the heading is the nameplate of the next passage; when there is no next chunk to attach
        it to (the text ends there, or the following unit is split on its own), it becomes a chunk of its own
    An atomic unit (fence / table) is packed like a sentence when it fits; only when it does not is it
    split by line on its own.
    """
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    def is_heading(sentence: str) -> bool:
        return sentence in headings or infer_heading_level(sentence) is not None

    def cost(sentence: str) -> int:
        # Sentences are joined with newlines, and that newline is a token too: a prose chunk has only a few
        # sentences, so a few tokens off do not matter; but an index / table of contents page with one word
        # per line packs two hundred-odd "sentences" per chunk, and counting only the sum of the sentences
        # would overfill by a hundred-odd tokens (2026-09-11: an index page of a library book chunked to
        # 783 tokens against a budget of 600)
        return count_tokens(sentence) + 1

    def flush(*, keep_overlap: bool = True) -> list[str]:
        nonlocal current, current_tokens
        carry = [current.pop()] if len(current) > 1 and is_heading(current[-1]) else []
        if current:
            chunks.append("\n".join(current).strip())
        tail: list[str] = []
        tail_tokens = 0
        if overlap_tokens > 0 and keep_overlap:
            for sentence in reversed(current):
                t = cost(sentence)
                if tail and tail_tokens + t > overlap_tokens:
                    break
                tail.insert(0, sentence)
                tail_tokens += t
        current = tail + carry
        current_tokens = tail_tokens + sum(cost(s) for s in carry)
        return carry

    for sentence, atomic in units:
        t = cost(sentence)
        if t > max_tokens:
            # A heading moved off the end of the chunk cannot attach to a unit that is split on its own: it becomes
            # a chunk of its own, and the fragment fallback (_glue_pieces) merges it into a neighbouring chunk
            chunks.extend(flush())
            if atomic:
                kind = "fence" if _FENCE_OPEN_RE.match(sentence) else "table"
                chunks.extend(_split_fence(sentence, max_tokens) if kind == "fence" else split_table_text(sentence, max_tokens))
            else:
                chunks.extend(_split_long_sentence(sentence, max_tokens))
            current = []
            current_tokens = 0
            continue
        if current and current_tokens >= max_tokens // 2 and is_heading(sentence):
            flush(keep_overlap=False)
        if current and current_tokens + t > max_tokens:
            carry = flush()
            # flush() keeps an overlap tail, and that tail always retains at
            # least one sentence however large it is. Without this second check
            # a big tail plus the incoming sentence would accumulate well past
            # max_tokens. The carried heading stays.
            if current and current_tokens + t > max_tokens:
                current = list(carry)
                current_tokens = sum(cost(s) for s in carry)
        current.append(sentence)
        current_tokens += t
    chunks.extend(flush())      # the text ends with a heading: no sentence follows to attach it to, so it stands alone too
    return [c for c in chunks if c]


def chunk_text(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
    *,
    headings: frozenset[str] = frozenset(),
) -> list[str]:
    """Chunk prose: fences and pipe tables are atomic segments (see _segments), the rest is packed by sentence
    (see _chunk_prose)."""
    units: list[tuple[str, bool]] = []
    for kind, seg in _segments(text):
        if kind == "prose":
            units.extend((s, False) for s in split_sentences(seg))
        else:
            seg = seg.strip()
            if seg:
                units.append((seg, True))
    if not units:
        return []
    return _chunk_prose(units, max_tokens, overlap_tokens, headings=headings)


def text_for_block(block: ParsedBlock) -> str:
    parts: list[str] = []
    if block.title:
        parts.append(f"TITLE: {block.title}")
    if block.caption:
        parts.append(f"CAPTION: {block.caption}")
    if block.table_markdown:
        parts.append(block.table_markdown)
    if block.latex:
        parts.append(f"EQUATION: {block.latex}")
    if block.text and block.text not in {block.table_markdown, block.latex}:
        parts.append(block.text)
    # enrich_blocks_with_vlm already writes "VISUAL SUMMARY: ..." into the text
    # of a block that had none of its own; do not repeat it a second time.
    # Compare only the prefix: the summary in text may have been truncated to the budget by tidy_visual_text,
    # and a full-string comparison would fail and append the uncapped original a second time
    summary = str(block.visual_summary or "").strip()
    if summary and summary[:60] not in (block.text or ""):
        parts.append(f"VISUAL SUMMARY: {summary}")
    return "\n".join(p.strip() for p in parts if p and p.strip()).strip()


# ── Blocks → chunks ───────────────────────────────────────────────────────

def tiny_threshold(max_tokens: int) -> int:
    """Same criterion as diagnose.tiny_threshold: 1/8 of the budget, no less than 16."""
    return max(16, int(max_tokens) // 8)


def _symbol(block: ParsedBlock) -> dict[str, Any]:
    sym = block.metadata.get("symbol") if isinstance(block.metadata, dict) else None
    return sym if isinstance(sym, dict) else {}


def _group_code_blocks(blocks: list[ParsedBlock], max_tokens: int) -> list[tuple[ParsedBlock, str]]:
    """Group code blocks by class: the class header + its method blocks become one chunk when they fit
    (attributed to the class header block); otherwise pack greedily, and methods that do not fit become
    chunks of their own. Other code blocks are left as they are. Returns [(owner block, text)]."""
    out: list[tuple[ParsedBlock, str]] = []
    i = 0
    while i < len(blocks):
        block = blocks[i]
        sym = _symbol(block)
        if block.block_type != "code" or sym.get("kind") != "class":
            out.append((block, text_for_block(block)))
            i += 1
            continue
        qual = str(sym.get("qualname") or sym.get("name") or "")
        members: list[ParsedBlock] = []
        j = i + 1
        while j < len(blocks):
            nxt = blocks[j]
            nsym = _symbol(nxt)
            if nxt.block_type == "code" and nsym.get("kind") == "method" and str(nsym.get("class") or "") == qual:
                members.append(nxt)
                j += 1
                continue
            break
        group: list[ParsedBlock] = [block]
        group_tokens = count_tokens(text_for_block(block))
        for member in members:
            t = count_tokens(text_for_block(member))
            if group_tokens + t > max_tokens and len(group) > 1:
                out.append((group[0], "\n\n".join(text_for_block(b) for b in group)))
                group, group_tokens = [member], t
                continue
            group.append(member)
            group_tokens += t
        out.append((group[0], "\n\n".join(text_for_block(b) for b in group)))
        i = j
    return out


def _heading_only(text: str) -> bool:
    """The whole chunk is a single heading / lead-in (only one short line remains after dropping the
    TITLE: lines)."""
    lines = [l.strip() for l in text.splitlines() if l.strip() and not l.strip().startswith("TITLE:")]
    if len(lines) != 1:
        return False
    line = lines[0]
    return infer_heading_level(line) is not None or is_short_lead_in(line)


def _same_section(a: ParsedBlock, b: ParsedBlock) -> bool:
    return list(a.metadata.get("section_path") or []) == list(b.metadata.get("section_path") or [])


def _same_slide(a: ParsedBlock, b: ParsedBlock) -> bool:
    """Slides are discrete units, so fragments never merge across slides; PDF pages are continuous, so merging
    across pages is allowed."""
    if a.slide_idx is None or b.slide_idx is None:
        return True
    return int(a.slide_idx) == int(b.slide_idx)


def _glue_pieces(pieces: list[dict[str, Any]], max_tokens: int) -> list[dict[str, Any]]:
    """Fragment fallback. piece = {"text", "block", "tokens"}."""
    threshold = tiny_threshold(max_tokens)
    limit = int(max_tokens * GLUE_SLACK)

    def text_like(p: dict[str, Any]) -> bool:
        return p["block"].block_type in TEXT_LIKE_TYPES

    for _ in range(2):
        changed = False
        i = 0
        while i < len(pieces):
            p = pieces[i]
            if not text_like(p) or p["tokens"] >= threshold:
                i += 1
                continue
            nxt = pieces[i + 1] if i + 1 < len(pieces) else None
            prev = pieces[i - 1] if i > 0 else None
            # A heading-only fragment right before a figure / table: becomes the TITLE of that block
            if (nxt is not None and nxt["block"].block_type in VISUAL_TYPES | {"table"} and _heading_only(p["text"])
                    and _same_slide(p["block"], nxt["block"])):
                title = " ".join(clean_heading_text(l) for l in p["text"].splitlines()
                                 if l.strip() and not l.strip().startswith("TITLE:")).strip()
                if title and not nxt["text"].startswith("TITLE:"):
                    nxt["text"] = f"TITLE: {title}\n{nxt['text']}"
                    nxt["tokens"] = count_tokens(nxt["text"])
                    pieces.pop(i)
                    changed = True
                    continue
            # A prose fragment only merges with a neighbour of the same section and the same slide: two short
            # slides with the same title used to be merged into one chunk, leaving only the later slide's
            # slide_idx as the source page (Codex review F09); PDF pages are continuous, so _same_slide
            # lets them through
            if (nxt is not None and text_like(nxt) and _same_section(p["block"], nxt["block"])
                    and _same_slide(p["block"], nxt["block"]) and p["tokens"] + nxt["tokens"] <= limit):
                nxt["text"] = f"{p['text']}\n{nxt['text']}"
                nxt["tokens"] = count_tokens(nxt["text"])
                pieces.pop(i)
                changed = True
                continue
            if (prev is not None and text_like(prev) and _same_section(prev["block"], p["block"])
                    and _same_slide(prev["block"], p["block"]) and prev["tokens"] + p["tokens"] <= limit):
                prev["text"] = f"{prev['text']}\n{p['text']}"
                prev["tokens"] = count_tokens(prev["text"])
                pieces.pop(i)
                changed = True
                continue
            # No prose neighbour in the same section (sandwiched between tables / images): a short paragraph
            # merges into the adjacent table / image chunk -- before it, it is that block's lead-in ("the
            # enterprise defaults to the following models:"); after it, its supplementary note. The slack is
            # 25% of the budget, same as for prose fragments: a PDF exported from slides has one big image
            # plus one title line per page, and the image chunk already sits right at the budget after its
            # description; not a single token over used to be allowed, so titles had to become chunks of
            # their own, and 8 of the 11 text chunks of a 16-page deck were fragments (2026-09-11)
            glued = False
            for cand, prepend in ((nxt, True), (prev, False)):
                if (cand is not None and cand["block"].block_type in VISUAL_TYPES | {"table"}
                        and _same_slide(p["block"], cand["block"]) and cand["tokens"] + p["tokens"] <= limit):
                    cand["text"] = f"{p['text']}\n{cand['text']}" if prepend else f"{cand['text']}\n{p['text']}"
                    cand["tokens"] = count_tokens(cand["text"])
                    pieces.pop(i)
                    changed = glued = True
                    break
            if glued:
                continue
            i += 1
        if not changed:
            break
    return pieces


def blocks_to_chunks(
    *,
    kb_id: str,
    file_key: int,
    content_version: str,
    parser_profile: str,
    blocks: list[ParsedBlock],
    max_tokens: int,
    overlap_tokens: int,
) -> list[UnifiedChunk]:
    pieces: list[dict[str, Any]] = []
    for block, text in _group_code_blocks(blocks, max_tokens):
        if not text:
            continue
        if block.block_type == "code":
            parts = [text] if count_tokens(text) <= max_tokens else chunk_text(text, max_tokens, 0)
        elif block.block_type == "table":
            parts = split_table_text(text, max_tokens)
        elif block.block_type in VISUAL_TYPES:
            if block.metadata.get("decorative"):
                # A decorative image never forms its own chunk: one description line is left in the preceding
                # prose chunk. The original text absorbed into the title before the image (short lines like
                # report date, testing lab, testing technique -- adopted_lead_in from pdf_enhanced.hand_over)
                # is document body text and must not be folded away together with the stamp (Codex review
                # F03: the "report date: 2023-11-06" on the first page of a nutrition-metabolism report was
                # lost exactly that way): first flow it back as a body line, then append one description
                # line; when there is no preceding chunk to attach to, keep it as a chunk of its own
                line = str(block.visual_summary or block.caption or "").strip()
                adopted = str(block.title or "").strip() if block.metadata.get("adopted_lead_in") else ""
                if not line and not adopted:
                    line = str(block.title or "").strip()
                extra = "\n".join(x for x in ((adopted if adopted and adopted not in line else ""), f"IMAGE: {line}" if line else "") if x)
                if not extra:
                    continue
                if pieces and pieces[-1]["block"].block_type in TEXT_LIKE_TYPES and pieces[-1]["tokens"] < int(max_tokens * GLUE_SLACK):
                    pieces[-1]["text"] = f"{pieces[-1]['text']}\n{extra}"
                    pieces[-1]["tokens"] = count_tokens(pieces[-1]["text"])
                elif adopted:
                    pieces.append({"text": extra, "block": block, "tokens": count_tokens(extra)})
                continue
            # VLM-derived figure text is bounded; keep one chunk per figure.
            # Slide blocks are deliberately NOT here: merged per-slide text can
            # exceed the budget (measured up to 725 tokens against a 400
            # limit), so slides chunk like ordinary text.
            parts = [text]
        else:
            parts = chunk_text(text, max_tokens, overlap_tokens,
                               headings=frozenset(block.metadata.get("heading_lines") or ()))
        for part in parts:
            pieces.append({"text": part, "block": block, "tokens": count_tokens(part)})

    pieces = _glue_pieces(pieces, max_tokens)

    result: list[UnifiedChunk] = []
    per_block: dict[str, int] = {}
    for piece in pieces:
        block = piece["block"]
        idx = per_block.get(block.block_id, 0)
        per_block[block.block_id] = idx + 1
        chunk_uid = f"{kb_id}:{file_key}:{content_version}:{parser_profile}:{block.block_id}:{idx}"
        result.append(
            UnifiedChunk(
                chunk_uid=chunk_uid,
                chunk_index=len(result),
                text=piece["text"],
                block=block,
                token_count=piece["tokens"],
            )
        )
    return result


# ── Embedding prefix ──────────────────────────────────────────────────────

EMBED_CONTEXT_MAX_CHARS = 200


def embedding_context_for(chunk: UnifiedChunk, *, doc_name: str) -> str:
    """Embedding prefix: "document name > section path". Titles and captions are already written into the text
    by text_for_block and are not repeated here."""
    parts = [str(doc_name or "").strip()] + [str(x).strip() for x in (chunk.block.metadata.get("section_path") or []) if str(x).strip()]
    context = " > ".join(p for p in parts if p)
    return context[:EMBED_CONTEXT_MAX_CHARS]


def embedding_input(chunk: UnifiedChunk) -> str:
    """The text actually sent for embedding = prefix + body; the text in the payload stays the original body."""
    context = getattr(chunk, "embedding_context", None)
    return f"{context}\n{chunk.text}" if context else chunk.text
