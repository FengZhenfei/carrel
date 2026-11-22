from __future__ import annotations

import posixpath
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from ..models import ParsedBlock
from .common import clean_text
from .mineru_docx import mineru_docx_blocks
from .native_table import normalize_cell
from .pdf_enhanced import merge_mineru_text_blocks, merge_split_tables
from ..vision.vlm import compose_prompt
from .visual_blocks import enrich_blocks_with_vlm, ensure_visual_blocks_have_text, tidy_visual_blocks


def parse_docx_enhanced(
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
    blocks = mineru_docx_blocks(
        mineru_url=mineru_url,
        path=path,
        cache_dir=cache_dir,
        timeout=timeout,
    )
    # docx used to have no merge step at all: every MinerU paragraph and every sub-heading line was a
    # block of its own. Measured, the median mineru-3.4.4-docx-v1 chunk was only 20 tokens and 70%
    # were under 50 tokens -- the most fragmented of all parsing routes (a sub-heading like
    # "**3.2.2.1 Connecting QQ**" was a chunk by itself). The block structure is identical to the PDF
    # route (same block_type set, same source_type metadata), so the same merger is reused directly,
    # and a heading therefore merges with the body paragraph below it.
    blocks = merge_mineru_text_blocks(
        blocks,
        target_tokens=merge_target_tokens,
        id_prefix="mineru-docx-merged-text",
    )
    blocks = merge_split_tables(blocks)

    media_blocks = [block for block in blocks if block.block_type in {"image", "chart"} and block.visual_ref]
    if not media_blocks:
        media_blocks = extract_docx_media_blocks(path=path, cache_dir=cache_dir)
        blocks.extend(media_blocks)
    enrich_blocks_with_vlm(
        media_blocks,
        base_url=vlm_base_url,
        api_key=vlm_api_key,
        model_id=vlm_model_id,
        cache_dir=cache_dir,
        concurrency=vlm_concurrency,
        prompt=compose_prompt("docx_image", vlm_prompt),
        filter_decorative=True,
        vlm_options=vlm_options,
        caption_cache_root=caption_cache_root,
        progress_cb=progress_cb,
    )
    tidy_visual_blocks(media_blocks)
    ensure_visual_blocks_have_text(media_blocks, doc_name=path.name)
    blocks = place_chart_blocks(blocks, extract_docx_chart_blocks(path))
    return attach_note_blocks(blocks, extract_docx_note_blocks(path, blocks))


def extract_docx_media_blocks(*, path: Path, cache_dir: Path) -> list[ParsedBlock]:
    media_dir = cache_dir / "docx_media"
    media_dir.mkdir(parents=True, exist_ok=True)
    blocks: list[ParsedBlock] = []
    try:
        with zipfile.ZipFile(path) as zf:
            names = sorted(
                name
                for name in zf.namelist()
                if name.startswith("word/media/") and not name.endswith("/")
            )
            for idx, name in enumerate(names, start=1):
                suffix = Path(name).suffix.lower()
                if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}:
                    continue
                out = media_dir / Path(name).name
                if not out.exists():
                    out.write_bytes(zf.read(name))
                blocks.append(
                    ParsedBlock(
                        parser="docx-ooxml-media",
                        parser_profile="docx-ooxml-media-vlm-v1",
                        doc_type="docx",
                        block_type="image",
                        text="",
                        title=f"{path.name} embedded image {idx}",
                        visual_ref=str(out),
                        block_id=f"docx-image-{idx:04d}",
                        metadata={"ooxml_path": name},
                    )
                )
    except zipfile.BadZipFile:
        return []
    return blocks


def extract_docx_chart_blocks(path: Path) -> list[ParsedBlock]:
    """One data-table block for each native chart in the document (word/charts/*.xml), in the order the charts
    appear in the body. metadata.anchor_text is the nearest paragraph before the chart; place_chart_blocks uses
    it to put the chart back where it belongs."""
    blocks: list[ParsedBlock] = []
    try:
        with zipfile.ZipFile(path) as zf:
            names = sorted(
                name
                for name in zf.namelist()
                if name.startswith("word/charts/") and name.endswith(".xml")
            )
            try:
                anchors = chart_anchors(zf) if names else {}
            except Exception:
                anchors = {}
            order = list(anchors)
            names.sort(key=lambda name: order.index(name) if name in anchors else len(order))
            for idx, name in enumerate(names, start=1):
                try:
                    title, axes, table = chart_xml_table(zf.read(name))
                except Exception:
                    table = ""
                if not table:
                    continue
                blocks.append(
                    ParsedBlock(
                        parser="docx-ooxml-chart",
                        parser_profile="docx-ooxml-chart-v2",
                        doc_type="docx",
                        # A chart's data is a table: chunked like a table (split by row when there are many rows,
                        # every chunk with the header), not kept whole like an image
                        block_type="table",
                        text=table,
                        table_markdown=table,
                        title=title or f"{path.name} chart {idx}",
                        caption="图表数据" + (f";坐标轴: {' / '.join(axes)}" if axes else ""),
                        block_id=f"docx-chart-{idx:04d}",
                        metadata={"ooxml_path": name, "anchor_text": anchors.get(name, "")},
                    )
                )
    except zipfile.BadZipFile:
        return []
    return blocks


_NOTE_PARTS = (("footnote", "word/footnotes.xml", "footnoteReference", "FOOTNOTE"),
               ("endnote", "word/endnotes.xml", "endnoteReference", "ENDNOTE"))


def _iter_ooxml_paragraphs(root: ET.Element):
    """Yield <w:p> elements while skipping <mc:Fallback> duplicates inside <mc:AlternateContent>."""
    stack = [root]
    while stack:
        elem = stack.pop()
        tag = elem.tag
        if tag.endswith("}Fallback") or tag == "Fallback":
            continue
        if tag == f"{_WORD_NS}p":
            yield elem
        # Push children in reverse so traversal stays in document order
        children = list(elem)
        for child in reversed(children):
            stack.append(child)


def _direct_paragraph_parts(para: ET.Element, marks: dict[str, tuple[str, str, str]]) -> tuple[str, list[tuple[str, str, str, str | None, str]]]:
    """Return (paragraph_text, [(kind, part, label, note_id, text_before_mark)]) for this <w:p> without
    descending into nested textbox paragraphs (<w:txbxContent>), avoiding double-counting nested <w:p>."""
    chunks: list[str] = []
    refs: list[tuple[str, str, str, str | None, str]] = []
    before = ""

    def walk(node: ET.Element) -> None:
        nonlocal before
        for child in node:
            tag = child.tag
            if tag.endswith("}Fallback") or tag == "Fallback" or tag == f"{_WORD_NS}txbxContent":
                continue
            if tag == f"{_WORD_NS}t":
                t = child.text or ""
                chunks.append(t)
                before += t
            elif tag in marks:
                kind, part, label = marks[tag]
                refs.append((kind, part, label, child.get(f"{_WORD_NS}id"), before))
            else:
                walk(child)

    walk(para)
    return clean_text("".join(chunks)), refs


def extract_docx_note_blocks(path: Path, blocks: list[ParsedBlock] = ()) -> list[ParsedBlock]:
    """One block for each footnote and endnote, in the order their reference marks appear in the body. The
    recogniser's answer for a Word file holds no notes at all, so without this they are not in the index.
    metadata.anchor_text is the paragraph that carries the mark, anchor_tail the last few characters before the
    mark and anchor_paragraph the number of that paragraph; attach_note_blocks uses them to write the note where
    it belongs. A note that the recogniser already delivered (in an explicit note block/line, attached to its
    anchored paragraph unit, or in an unmatched standalone text block) is left out without letting unrelated
    body phrasing, merged table repetitions, charts, or image captions suppress a real footnote."""
    found: list[ParsedBlock] = []
    try:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            notes: dict[tuple[str, str | None], str] = {}
            for kind, part, _mark, _label in _NOTE_PARTS:
                if part not in names:
                    continue
                for note in ET.fromstring(zf.read(part)).iter(f"{_WORD_NS}{kind}"):
                    text = "\n".join(line for line in (_paragraph_text(para) for para in note.iter(f"{_WORD_NS}p")) if line)
                    if text:
                        notes[(kind, note.get(f"{_WORD_NS}id"))] = text
            if not notes:
                return []

            def plain(text: str) -> str:
                return _MARKUP_RE.sub("", _LINK_RE.sub(r"\1", text))

            # 1. Explicit note texts already recognised (dedicated note blocks or FOOTNOTE:/ENDNOTE: lines only).
            explicit_note_texts: list[str] = []
            for block in blocks:
                if block.metadata.get("source_type") in {"footnote", "endnote"}:
                    p = plain(block.text or "")
                    if p:
                        explicit_note_texts.append(p)
                else:
                    for line in (block.text or "").splitlines():
                        stripped = line.lstrip()
                        if stripped.startswith(("FOOTNOTE:", "ENDNOTE:")):
                            body = plain(stripped.split(":", 1)[1])
                            if body:
                                explicit_note_texts.append(body)
            explicit_used: list[bool] = [False] * len(explicit_note_texts)

            # 2. Decompose eligible body blocks into paragraph units (same granularity as attach_note_blocks)
            #    so merged 800-token blocks are checked per paragraph unit, and exclude VLM/chart/media blocks.
            doc_root = ET.fromstring(zf.read("word/document.xml"))
            marks = {f"{_WORD_NS}{mark}": (kind, part, label) for kind, part, mark, label in _NOTE_PARTS}
            doc_entries: list[tuple[str, str, list[tuple[str, str, str, str | None, str]]]] = []
            for para in _iter_ooxml_paragraphs(doc_root):
                ptxt, prefs = _direct_paragraph_parts(para, marks)
                doc_entries.append((ptxt, plain(ptxt), prefs))
            ooxml_body_plains = [p for _, p, _ in doc_entries if p]

            unit_entries: list[tuple[str, bool, bool]] = []  # (plain_text, is_plain_text_block, is_body_para)
            for block in blocks:
                if block.parser in {"docx-ooxml-chart", "docx-ooxml-media"} or block.visual_ref:
                    continue
                bu = _paragraph_units(block)
                if not bu:
                    continue
                is_text_block = block.block_type in {"text", "list"}
                for start, end, _span in bu:
                    up = plain((block.text or "")[start:end])
                    is_body = bool(up) and any(up == bp or (len(up) >= 6 and up in bp) for bp in ooxml_body_plains)
                    unit_entries.append((up, is_text_block, is_body))

            # Unmatched standalone text units (candidate recogniser-emitted note blocks, e.g. endnotes at tail)
            unmatched_text_units: list[tuple[int, str]] = [
                (idx, up) for idx, (up, is_text_block, is_body) in enumerate(unit_entries)
                if is_text_block and up and not is_body
            ]
            unmatched_used: set[int] = set()

            cursor = -1
            last_para = None
            for number, (para_text, para_plain, refs) in enumerate(doc_entries):
                for kind, part, label, note_id, before in refs:
                    text = notes.pop((kind, note_id), "")
                    needle = _MARKUP_RE.sub("", text)
                    if not text or not needle:
                        continue
                    # Check 1: explicit note block / FOOTNOTE: line (consume once per match)
                    already = False
                    for ei, etxt in enumerate(explicit_note_texts):
                        if not explicit_used[ei] and needle in etxt:
                            explicit_used[ei] = True
                            already = True
                            break
                    # Check 2: anchored paragraph unit & its immediate next non-body unit
                    if not already and unit_entries:
                        short = _MARKUP_RE.sub("", before)[-16:]
                        short = short if len(short) >= 6 else ""
                        if cursor >= 0 and number == last_para:
                            places = [cursor]
                        else:
                            places = [*range(cursor + 1, len(unit_entries)), *([cursor] if cursor >= 0 else [])]
                        anchored = next(
                            (n for k in (para_plain, short) if k for n in places if k in unit_entries[n][0]),
                            None,
                        )
                        if anchored is not None:
                            cursor, last_para = anchored, number
                            up = unit_entries[anchored][0]
                            if up.count(needle) > para_plain.count(needle):
                                already = True
                            elif anchored + 1 < len(unit_entries):
                                nxt_up, nxt_is_text, nxt_is_body = unit_entries[anchored + 1]
                                if nxt_is_text and not nxt_is_body and (anchored + 1) not in unmatched_used:
                                    # Also join one more consecutive unmatched text unit for split multi-para notes
                                    combined = nxt_up
                                    span_ids = [anchored + 1]
                                    if anchored + 2 < len(unit_entries):
                                        n2_up, n2_text, n2_body = unit_entries[anchored + 2]
                                        if n2_text and not n2_body and (anchored + 2) not in unmatched_used:
                                            combined += n2_up
                                            span_ids.append(anchored + 2)
                                    if needle in nxt_up:
                                        unmatched_used.add(anchored + 1)
                                        already = True
                                    elif needle in combined:
                                        unmatched_used.update(span_ids)
                                        already = True
                    # Check 3: unmatched standalone text unit elsewhere (e.g. endnote emitted at end of doc)
                    if not already and unmatched_text_units:
                        for ui, uplain in unmatched_text_units:
                            if ui not in unmatched_used and needle in uplain:
                                unmatched_used.add(ui)
                                already = True
                                break
                    if already:
                        continue
                    found.append(ParsedBlock(
                        parser="docx-ooxml-note", parser_profile="docx-ooxml-note-v1", doc_type="docx",
                        block_type="text", text=f"{label}: {text}", block_id=f"docx-{kind}-{len(found) + 1:04d}",
                        metadata={"ooxml_path": part, "source_type": kind, "anchor_text": para_text,
                                  "anchor_tail": _MARKUP_RE.sub("", before)[-16:], "anchor_paragraph": number}))
    except (zipfile.BadZipFile, KeyError, ET.ParseError):
        return []
    return found


def _paragraph_text(para: ET.Element) -> str:
    return clean_text("".join(node.text or "" for node in para.iter(f"{_WORD_NS}t")))


_CHART_NS = "{http://schemas.openxmlformats.org/drawingml/2006/chart}"
_WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_REL_ID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
_AXIS_TAGS = frozenset(f"{_CHART_NS}{kind}" for kind in ("catAx", "dateAx", "valAx", "serAx"))
_MARKUP_RE = re.compile(r"[\s*_`\\#]+")          # whitespace and markdown marks are ignored when comparing text
_LINK_RE = re.compile(r"\[([^\]\n]*)\]\([^)\n]*\)")


def chart_anchors(zf: zipfile.ZipFile) -> dict[str, str]:
    """{chart part path: the nearest paragraph before it}, in the order the charts appear in the body."""
    targets = {}
    for rel in ET.fromstring(zf.read("word/_rels/document.xml.rels")):
        target = str(rel.get("Target") or "")
        if str(rel.get("Type") or "").endswith("/chart") and target:
            targets[rel.get("Id")] = target.lstrip("/") if target.startswith("/") else posixpath.normpath(posixpath.join("word", target))
    anchors: dict[str, str] = {}
    last_text = ""
    for para in ET.fromstring(zf.read("word/document.xml")).iter(f"{_WORD_NS}p"):
        text = clean_text("".join(node.text or "" for node in para.iter(f"{_WORD_NS}t")))
        for ref in para.iter(f"{_CHART_NS}chart"):
            name = targets.get(ref.get(_REL_ID))
            if name:
                anchors.setdefault(name, text or last_text)
        last_text = text or last_text
    return anchors


def place_chart_blocks(blocks: list[ParsedBlock], charts: list[ParsedBlock]) -> list[ParsedBlock]:
    """Put chart blocks back into the body: a chart goes right after the block that contains its anchor text and
    takes over that block's section and page; a chart whose anchor is not found goes to the end. Charts used to
    be appended at the very end of the document without a section, so nothing told which section they
    belonged to."""
    if not charts:
        return blocks
    out = list(blocks)
    plain = [_MARKUP_RE.sub("", block.text or "") for block in out]
    tail: list[ParsedBlock] = []
    last_key, last_at = "", -1
    for chart in charts:
        key = _MARKUP_RE.sub("", str(chart.metadata.get("anchor_text") or ""))
        if key and key == last_key:
            at = last_at                                 # several charts after the same paragraph: one after another
        else:
            at = next((i for i in range(last_at + 1, len(out)) if key and key in plain[i]), None)
        if at is None:
            tail.append(chart)
            continue
        chart.metadata["section_path"] = list(out[at].metadata.get("section_path") or [])
        chart.page_idx = out[at].page_idx
        out.insert(at + 1, chart)
        plain.insert(at + 1, "")
        last_key, last_at = key, at + 1
    return out + tail


def _paragraph_units(block: ParsedBlock) -> list[tuple[int, int, dict | None]] | None:
    """The stretches of a block's text that came from one paragraph of the source each: (start, end, span record).
    A merged text block lists them in metadata.section_spans; any other text or table block is one stretch.
    None for a block that takes no note (a picture, a formula) or whose span list does not fit its text."""
    if block.block_type not in {"text", "list", "table"} or block.visual_ref:
        return None
    text = block.text or ""
    spans = block.metadata.get("section_spans")
    if not spans:
        return [(0, len(text), None)]
    units, at = [], 0
    for span in spans:
        start, end = span.get("start"), span.get("end")
        if not isinstance(start, int) or not isinstance(end, int) or not at <= start <= end <= len(text):
            return None
        units.append((start, end, span))
        at = end
    return units


def attach_note_blocks(blocks: list[ParsedBlock], notes: list[ParsedBlock]) -> list[ParsedBlock]:
    """Write every footnote and endnote into the block that holds its paragraph, on a line of its own right after
    that paragraph (after the table when the mark is in a table cell, where the note lines of a table already go).
    A reader of the chunk then finds the note under the sentence it explains, as on the printed page, and the note
    shares that paragraph's section. Only a note whose paragraph is not found becomes a block of its own, at the
    end of the document.

    The paragraph is looked for as it stands and, failing that (the recogniser may leave out a field or a
    cross-reference), by the last characters before the mark. A link the recogniser wrote as [text](address) is
    compared by its text. Notes come in the order of their marks, so the search goes on from the last place
    found: a note from another paragraph looks further down first and only then in that same place (two cells
    of one table, two paragraphs the recogniser delivered as one)."""
    if not notes:
        return blocks
    units = [_paragraph_units(block) for block in blocks]
    order = [(i, j) for i, block_units in enumerate(units) for j in range(len(block_units or ()))]
    seen: dict[tuple[int, int], str] = {}

    def plain(text: str) -> str:
        return _MARKUP_RE.sub("", _LINK_RE.sub(r"\1", text))

    def holds(place: tuple[int, int], needle: str) -> bool:
        if place not in seen:
            start, end, _span = units[place[0]][place[1]]
            seen[place] = plain((blocks[place[0]].text or "")[start:end])
        return needle in seen[place]

    tail: list[ParsedBlock] = []
    at = -1                                               # position in order of the last place found
    last_paragraph = None
    for note in notes:
        key = plain(str(note.metadata.get("anchor_text") or ""))
        short = plain(str(note.metadata.get("anchor_tail") or ""))
        short = short if len(short) >= 6 else ""          # a handful of characters would match anywhere
        paragraph = note.metadata.get("anchor_paragraph")
        if at >= 0 and paragraph is not None and paragraph == last_paragraph:
            places = [at]                                 # another mark in the same paragraph: one note after the other
        else:
            places = [*range(at + 1, len(order)), *([at] if at >= 0 else [])]
        found = next((n for needle in (key, short) if needle for n in places if holds(order[n], needle)), None)
        if found is None:
            tail.append(note)
            continue
        at, last_paragraph = found, paragraph
        i, j = order[found]
        block = blocks[i]
        end = units[i][j][1]
        line = "\n" + note.text
        block.text = block.text[:end] + line + block.text[end:]
        if block.block_type == "table" and block.table_markdown is not None:
            block.table_markdown = block.text             # a table's text and table_markdown must stay the same (as in the PDF route)
        # That stretch grows by the note, the ones after it move down
        for n, (a, b, record) in enumerate(units[i]):
            if n < j:
                continue
            a, b = (a, b + len(line)) if n == j else (a + len(line), b + len(line))
            if record is not None:
                record["start"], record["end"] = a, b
            units[i][n] = (a, b, record)
        seen.pop((i, j), None)
        block.metadata.setdefault("footnote_block_ids", []).append(note.block_id)
    return blocks + tail


def chart_xml_table(raw: bytes) -> tuple[str, list[str], str]:
    """Chart XML -> (title, axis titles, data table). The data table has one row per point: series | category |
    value; for scatter / bubble charts the X value goes into the category. All the text in the XML used to be
    collected and de-duplicated by value: when two quarters were both 10 the second 10 vanished and every later
    value shifted; formula references (Sheet1!$B$2:$B$5) were written into the body as content too."""
    root = ET.fromstring(raw)
    chart = root.find(f"{_CHART_NS}chart")
    if chart is None:
        # Charts with the other structure (chartEx: waterfall, sunburst and the like): not paired by series; the
        # cached text and values are listed in their original order, without de-duplication or formula references
        return "", [], "\n".join(text for text in (clean_text(el.text or "") for el in root.iter()
                                                  if el.tag.endswith(("}pt", "}v", "}t"))) if text)
    title = _chart_text(chart.find(f"{_CHART_NS}title"))
    axes = [text for text in (_chart_text(axis.find(f"{_CHART_NS}title")) for axis in chart.iter() if axis.tag in _AXIS_TAGS) if text]
    rows: list[list[str]] = []
    sized = False
    for number, ser in enumerate(chart.iter(f"{_CHART_NS}ser"), start=1):
        name = _chart_text(ser.find(f"{_CHART_NS}tx")) or f"系列{number}"
        cats = _chart_points(ser.find(f"{_CHART_NS}cat")) or _chart_points(ser.find(f"{_CHART_NS}xVal"))
        vals = _chart_points(ser.find(f"{_CHART_NS}val")) or _chart_points(ser.find(f"{_CHART_NS}yVal"))
        sizes = _chart_points(ser.find(f"{_CHART_NS}bubbleSize"))
        sized = sized or bool(sizes)
        for idx in sorted(set(cats) | set(vals)):
            rows.append([name, cats.get(idx) or str(idx + 1), vals.get(idx, ""), sizes.get(idx, "")])
    if not rows:
        return title, axes, ""
    width = 4 if sized else 3
    lines = [["系列", "类别", "值", "大小"][:width], ["---"] * width] + [row[:width] for row in rows]
    return title, axes, "\n".join("| " + " | ".join(cell.replace("|", "\\|") for cell in line) + " |" for line in lines)


def _chart_text(node: ET.Element | None) -> str:
    """Title / series name: rich text (a:t) or a cached string (c:v). Formula references (c:f) do not count."""
    if node is None:
        return ""
    text = "".join(el.text or "" for el in node.iter() if el.tag.endswith(("}t", "}v")))
    return " ".join(clean_text(text).split())


def _chart_points(node: ET.Element | None) -> dict[int, str]:
    """The cached points of a category or value reference: {index: written form}. Numbers are written with the
    cached number format (percentages, dates); multi-level categories (c:lvl, the inner level first in the
    file) are joined with "/" from the outer to the inner level, and an outer label is written only on the
    first point of its group and carried forward."""
    if node is None:
        return {}
    number_format = next((el.text for el in node.iter(f"{_CHART_NS}formatCode")), None)
    numeric = any(el.tag in (f"{_CHART_NS}numCache", f"{_CHART_NS}numLit") for el in node.iter())
    levels = [{int(pt.get("idx") or 0): (pt.findtext(f"{_CHART_NS}v") or "").strip() for pt in level.iter(f"{_CHART_NS}pt")}
              for level in (list(node.iter(f"{_CHART_NS}lvl")) or [node])]
    indexes = sorted({idx for values in levels for idx in values})
    points: dict[int, str] = {}
    for values in reversed(levels):
        carried = ""
        for idx in indexes:
            text = values.get(idx, "")
            if values is not levels[0]:
                text = carried = text or carried
            if not text:
                continue
            text = _chart_number(text, number_format) if numeric else clean_text(text)
            points[idx] = f"{points[idx]}/{text}" if idx in points else text
    return points


def _chart_number(raw: str, number_format: str | None) -> str:
    try:
        number = float(raw)
    except ValueError:
        return clean_text(raw)
    try:
        from openpyxl.styles.numbers import is_date_format
        from openpyxl.utils.datetime import from_excel

        if number_format and is_date_format(number_format):
            return normalize_cell(from_excel(number))       # a date axis caches serial numbers
    except Exception:
        pass
    return normalize_cell(number, number_format)
