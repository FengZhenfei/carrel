"""Source-owned PDF evidence: visible forms, local text and table review regions.

All geometry uses the parser's 0..1000 page coordinates. Document text is data;
no domain vocabulary or known answers participate in selecting a region.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from ..models import ParsedBlock

# Unicode 17.0 Equivalent_Unified_Ideograph mappings for the CJK Radicals
# Supplement block not already covered by NFKC. Unknown glyphs stay literal.
# https://www.unicode.org/Public/17.0.0/ucd/EquivalentUnifiedIdeograph.txt
# Unicode data license: THIRD_PARTY_LICENSES/Unicode-3.0.txt.
SOURCE_RADICALS = str.maketrans(
    '⺁⺂⺃⺄⺅⺆⺇⺈⺉⺊⺋⺌⺍⺎⺏⺐⺑⺒⺓⺔⺕⺖⺗⺘⺙⺛⺜⺝⺞⺠⺡⺢⺣⺤⺥⺦⺧⺨⺩⺪⺫⺬⺭⺮⺯⺰⺱⺲⺳⺴⺵⺶⺷⺸⺹⺺⺻⺼⺽⺾⺿⻀⻁⻂⻃⻄⻅⻆⻇⻈⻉⻊⻋⻌⻍⻎⻏⻐⻑⻒⻓⻔⻕⻖⻗⻘⻙⻚⻛⻜⻝⻞⻟⻠⻡⻢⻣⻤⻥⻦⻧⻨⻩⻪⻫⻬⻭⻮⻯⻰⻱⻲',
    '厂乛乚乙亻冂𠘨刀刂卜㔾小小兀尣尢𡯂巳幺彑𫜹忄心扌攵旡日月歺民氵氺灬爫爫丬牛犭王𤴔目示礻𥫗糹纟罓罒㓁冗𦉫羊𦍌𦍋耂肀聿肉𦥑艹艹艹虎衤覀西见角𧢲讠贝𧾷车辶辶辶邑钅長镸长门𨸏阝雨青韦页风飞食𩙿飠饣𩠐马骨鬼鱼鸟卤麦黄黾斉齐歯齿竜龙龜亀')


def _norm(text: str) -> str:
    text = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text or ""))
    # Typographic hyphens are spelling variants; mathematical minus is not.
    text = text.replace("‐", "-").replace("‑", "-")
    return normalize_ranges(normalize_table_presentation(text, strip_sentence_end=False))


def normalize_table_presentation(text: str, *, strip_sentence_end: bool = True) -> str:
    """Compare typographic variants without changing stored text or fact symbols."""
    text = unicodedata.normalize("NFKC", text.strip()).translate(SOURCE_RADICALS).translate(str.maketrans("【】", "[]"))
    # Ellipsis glyphs and repeated dots are presentation, not decimal points.
    text = re.sub(r"\.{3,}|[·…]{2,}", "…", text)
    # Sentence punctuation is optional in prose cells; bare numbers and their
    # decimal points, operators, brackets and ellipses remain distinct.
    return text.rstrip(".。") if strip_sentence_end and any(c.isalpha() for c in text) else text


def table_evidence_text(text: str) -> str:
    """Remove table markup, preserving literal bars inside cells."""
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    return _norm(re.sub(r"(?<!\\)\|", " ", text).replace(r"\|", "|"))


def normalize_ranges(text: str) -> str:
    # Typography of a numeric range, not signs, inequalities or arbitrary dashes.
    # U+2212 is a mathematical minus and deliberately stays distinct.
    return re.sub(r"(\d(?:\s*(?:[^\W\d_]|[%‰°]))*)\s*[‐‑‒–—]\s*(?=[$€£¥]?\d)",
                  r"\1-", text)


def _span_present(span: str, body: str) -> bool:
    text = normalize_table_presentation(_norm(span))
    body = _norm(body)
    if text in body:
        return True
    # A list bullet or a printed HTML line break may survive native extraction
    # while the OCR table keeps its complete item text. Compare that item, not
    # the presentation prefix; ordinary text and standalone markup stay exact.
    item = re.sub(r"^(?:<br\s*/?>)?[•●◦▪]", "", text, flags=re.I)
    if item != text and item and item in body:
        return True
    return False


def source_span_present(span: str, body: str, spans: list[dict]) -> bool:
    if _span_present(span, body):
        return True
    # A word split at the end of a printed line can be joined by OCR. Require
    # an adjacent source line in the same column, not arbitrary dash removal.
    if not re.search(r"[A-Za-z]{2,}-$", span.rstrip()):
        return False
    owners = [s for s in spans if s.get("text", "").strip() == span.strip() and s.get("bbox")]
    if len(owners) != 1:
        return False
    box = owners[0]["bbox"]
    height = max(box[3] - box[1], 1)
    following = [s for s in spans if s.get("bbox")
        and 0 < s["bbox"][1] - box[1] <= height * 1.8
        and abs(s["bbox"][0] - box[0]) <= height
        and re.match(r"^[A-Za-z]+", s.get("text", "").strip())]
    if len(following) != 1:
        return False
    return _span_present(span.rstrip()[:-1] + following[0]["text"].strip(), body)


def _rect(page, box):
    import pymupdf
    return pymupdf.Rect(box[0] * page.rect.width / 1000, box[1] * page.rect.height / 1000,
                        box[2] * page.rect.width / 1000, box[3] * page.rect.height / 1000) & page.rect


def _box(page, rect) -> list[float]:
    return [rect.x0 * 1000 / page.rect.width, rect.y0 * 1000 / page.rect.height,
            rect.x1 * 1000 / page.rect.width, rect.y1 * 1000 / page.rect.height]


def _native_line_spans(line: dict) -> list[dict]:
    """Rejoin adjacent single-glyph spans, never separate lines or cells.

    PDF styling can split a word or number into one span per glyph. Preserve
    spaces, baseline changes and cell gaps; these are evidence boundaries.
    """
    result = []
    previous = None
    for source in line.get("spans", []):
        span = {**source, "line_dir": line.get("dir")}
        text = span.get("text", "")
        box = span.get("bbox")
        size = float(span.get("size") or 0)
        single = len(text) == 1 and text.isalnum()
        if (single and previous and len(previous.get("text", "")) == 1
                and previous["text"].isalnum() and box and previous.get("bbox")
                and tuple(line.get("dir") or ()) == (1, 0) and size > 0
                and abs(size - float(previous.get("size") or 0)) < .1
                and abs(box[1] - previous["bbox"][1]) < .1
                and -.15 * size <= box[0] - previous["bbox"][2] <= .15 * size):
            joined = result[-1]
            joined["text"] += text
            a = joined["bbox"]
            joined["bbox"] = (min(a[0], box[0]), min(a[1], box[1]), max(a[2], box[2]), max(a[3], box[3]))
        else:
            result.append(span)
        previous = source
    return result


def _unreadable_native_glyphs(page, fonts=None) -> list[tuple[float, float]]:
    """Locate broken mappings, including visible glyphs encoded as spaces.

    The PDF's Unicode map and its embedded font must agree on a space glyph.
    Unknown or non-embedded fonts are left alone; glyph boxes are not proof
    because some valid fonts report an ink box even for a real space.
    """
    import pymupdf
    fonts = {} if fonts is None else fonts
    # Structured extraction may apply /ActualText that repairs a bad font map.
    # A raw trace must not invalidate that correctly decoded text.
    spaces = {(tuple(c["origin"]), ord(c["c"]))
              for b in page.get_text("rawdict", flags=pymupdf.TEXTFLAGS_RAWDICT & ~pymupdf.TEXT_PRESERVE_IMAGES).get("blocks", [])
              for line in b.get("lines", []) for span in line.get("spans", [])
              for c in span.get("chars", []) if len(c["c"]) == 1 and c["c"].isspace() and not c.get("synthetic")}
    name = lambda value: re.sub(r"[\s-]", "", value.split("+")[-1]).casefold()
    by_name = {}
    for item in page.get_fonts():
        xref = item[0]
        if xref not in fonts:
            data = page.parent.extract_font(xref)[3]
            try:
                font = pymupdf.Font(fontbuffer=data) if data else None
                fonts[xref] = {c: font.has_glyph(c) for c in font.valid_codepoints()
                               if chr(c).isspace()} if font else {}
            except RuntimeError:
                fonts[xref] = {}
        by_name.setdefault(name(item[3]), []).append(fonts[xref])
    bad = []
    for span in page.get_texttrace():
        candidates = by_name.get(name(span["font"]), [])
        for code, glyph, origin, _ in span["chars"]:
            if (tuple(origin), code) in spaces and candidates:
                # Space/NBSP/width variants may share a decoded Unicode space.
                expected = {g for font in candidates for g in font.values()}
                if all(font.get(code) for font in candidates) and glyph not in expected:
                    bad.append(tuple(origin))
    return bad


def _native_region(page, rect, bad_glyphs):
    """Keep intact native lines; a damaged line cannot supply independent facts."""
    import pymupdf
    lines, omitted = [], 0
    for block in page.get_text("dict", clip=rect).get("blocks", []):
        for line in block.get("lines", []):
            box = pymupdf.Rect(line["bbox"])
            if (any(point in box for point in bad_glyphs) or
                    any("\ufffd" in s.get("text", "") or "\x00" in s.get("text", "") for s in line.get("spans", []))):
                omitted += 1
            else:
                lines.append(line)
    spans = [s for line in lines for s in _native_line_spans(line)]
    spans.sort(key=lambda s: (s["bbox"][1], s["bbox"][0]))
    native = ("\n".join("".join(s.get("text", "") for s in line.get("spans", []))
                         for line in sorted(lines, key=lambda x: (x["bbox"][1], x["bbox"][0])))
              if omitted else page.get_text("text", clip=rect, sort=True)).strip()
    return native, spans, omitted


def supplement_visible_forms(blocks: list[ParsedBlock], path: Path) -> list[ParsedBlock]:
    """Read visible field values without executing actions or flattening the PDF."""
    try:
        import pymupdf
    except ImportError:
        return blocks
    try:
        doc = pymupdf.open(path)
    except Exception:
        return blocks
    out = list(blocks)
    with doc:
        for page_no, page in enumerate(doc, 1):
            for widget in page.widgets() or []:
                if widget.field_type not in {pymupdf.PDF_WIDGET_TYPE_TEXT,
                        pymupdf.PDF_WIDGET_TYPE_COMBOBOX, pymupdf.PDF_WIDGET_TYPE_LISTBOX}:
                    continue
                value = str(widget.field_value or "").strip()
                kind, flag = doc.xref_get_key(widget.xref, "F")
                flags = int(flag) if kind == "int" else 0
                if not value or flags & (1 | 2 | 32) or widget.rect.is_empty:
                    continue  # Invisible, hidden, or no-view annotations.
                region = widget.rect & page.rect
                if region.is_empty:
                    continue
                candidates = [b for b in out if b.page_idx == page_no and b.bbox
                              and (_rect(page, b.bbox) & region).get_area() > 0]
                if any(_norm(value) in _norm(b.text) for b in candidates):
                    continue
                label = str(widget.field_label or "").strip()
                body = f"{label}: {value}" if label and _norm(label) not in _norm(value) else value
                block = ParsedBlock(parser="pdf-form", parser_profile="pdf-evidence-v1", doc_type="pdf",
                    block_type="text", text=body, block_id=f"pdf-form-{page_no}-{widget.xref}",
                    page_idx=page_no, bbox=_box(page, region), metadata={"source_type": "form",
                    "source_field": widget.field_name, "source_xref": widget.xref, "section_path": []})
                index = next((i for i, b in enumerate(out) if b.page_idx is not None and
                    (b.page_idx > page_no or (b.page_idx == page_no and b.bbox and
                     b.bbox[1] > block.bbox[1] and min(b.bbox[2], block.bbox[2]) > max(b.bbox[0], block.bbox[0])))), len(out))
                if index and out[index - 1].page_idx == page_no:
                    block.metadata["section_path"] = list(out[index - 1].metadata.get("section_path") or [])
                out.insert(index, block)
    return out


def add_pdf_region_evidence(blocks: list[ParsedBlock], path: Path, cache_dir: Path) -> None:
    """Attach native text inside the owning region; flag local loss for review.

    Blank cells alone never trigger review. A printed source span missing from a
    table or a symbol legend near it provides an independent reason to inspect.
    """
    try:
        import pymupdf
    except ImportError:
        return
    try:
        doc = pymupdf.open(path)
    except Exception:
        return
    with doc:
        bad_by_page, fonts = {}, {}
        for block in blocks:
            if block.block_type not in {"image", "chart", "table"} or not block.bbox or not block.page_idx:
                continue
            if not 1 <= block.page_idx <= len(doc):
                continue
            page = doc[block.page_idx - 1]
            rect = _rect(page, block.bbox)
            if rect.is_empty:
                continue
            if block.page_idx not in bad_by_page:
                bad_by_page[block.page_idx] = _unreadable_native_glyphs(page, fonts)
            bad_glyphs = bad_by_page[block.page_idx]
            native, native_spans, omitted = _native_region(page, rect, bad_glyphs)
            if omitted:
                block.metadata["pdf_native_lines_omitted"] = omitted
            if block.block_type != "table":
                context = "\n".join(x for x in (native, block.metadata.get("source_footnote", "")) if x)
                if context:
                    block.metadata["visual_source_context"] = context
                continue
            got = table_evidence_text("\n".join([block.table_markdown or block.text, block.caption or "",
                                    str(block.metadata.get("source_footnote") or "")]))
            # get_text(sort=True) inserts spaces between cells. Compare individual
            # spans, not whole rows joined across Markdown column separators.
            spans = [s.get("text", "").strip() for s in native_spans]
            missing = list(dict.fromkeys(s for s in spans if len(_norm(s)) >= 2
                and not re.fullmatch(r"[\s\-‐‑‒–—]+", s) and _norm(s) != "…"
                and not source_span_present(s, got, native_spans)))
            # Look only below this table, in its column, stopping before another
            # content region. A distant or adjacent-column legend is not adopted.
            bottom = min(page.rect.height, rect.y1 + page.rect.height * .12)
            for other in blocks:
                if other is block or other.page_idx != block.page_idx or not other.bbox:
                    continue
                if other.block_type not in {"table", "image", "chart", "title"}:
                    continue
                r = _rect(page, other.bbox)
                if r.y0 >= rect.y1 and min(r.x1, rect.x1) > max(r.x0, rect.x0):
                    bottom = min(bottom, r.y0)
            below, _, _ = _native_region(page, pymupdf.Rect(rect.x0, rect.y1, rect.x1, bottom), bad_glyphs)
            legend_terms = (r"(?:\blegend\b|\bsymbols?\b|图例|符号说明|"
                            r"[“\"‘'][^“”\"‘’'\n]{1,8}[”\"’']\s*(?:指的是|表示|代表|means|denotes|indicates))")
            footnote = str(block.metadata.get("source_footnote") or "")
            legend = "\n".join(dict.fromkeys(s for s in (footnote, below) if s and re.search(legend_terms, s, re.I)))
            if not missing and not legend:
                continue
            reasons = (["native_text_missing"] if missing else []) + (["symbol_legend"] if legend else [])
            # Native ruled cells are independent evidence for filling an OCR
            # blank. Whole-table text cannot establish a row or column.
            candidates = []
            try:
                for table in page.find_tables().tables:
                    area = pymupdf.Rect(table.bbox)
                    if (area & rect).get_area() / max(area.get_area(), rect.get_area(), 1) >= .85:
                        candidates.append([[cell or "" for cell in row] for row in table.extract()])
            except Exception:
                pass
            if not candidates and native:
                # Borderless tables still have independently positioned native
                # cells. Use only this region, never a page-wide text guess.
                try:
                    for table in page.find_tables(clip=rect, strategy="text").tables:
                        area = pymupdf.Rect(table.bbox)
                        if (area & rect).get_area() / max(area.get_area(), rect.get_area(), 1) >= .85:
                            candidates.append([[cell or "" for cell in row] for row in table.extract()])
                    if len(candidates) == 1:
                        block.metadata["table_native_grid_strategy"] = "text"
                except Exception:
                    pass
            block.metadata["table_native_grid"] = candidates[0] if native and not omitted and len(candidates) == 1 else []
            block.metadata["table_source_text"] = native
            block.metadata["table_native_spans"] = [{"text": s.get("text", ""), "bbox": list(s["bbox"]),
                "origin": list(s.get("origin", [])), "line_dir": list(s.get("line_dir") or [])}
                for s in native_spans if s.get("bbox") and native]
            block.metadata["table_source_legend"] = legend
            block.metadata["table_review_reasons"] = reasons
            block.metadata["table_missing_spans"] = missing
            review_rect = pymupdf.Rect(rect.x0, rect.y0, rect.x1, bottom if legend else rect.y1)
            dest = cache_dir / "evidence" / f"{block.block_id}.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            page.get_pixmap(clip=review_rect, dpi=160, alpha=False).save(dest)
            block.metadata["table_review_image"] = str(dest)
