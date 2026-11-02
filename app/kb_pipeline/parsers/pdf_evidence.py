"""Source-owned PDF evidence: visible forms, local text and table review regions.

All geometry uses the parser's 0..1000 page coordinates. Document text is data;
no domain vocabulary or known answers participate in selecting a region.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from ..models import ParsedBlock


def _norm(text: str) -> str:
    text = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text or ""))
    # Typographic hyphens are spelling variants; mathematical minus is not.
    text = text.replace("‐", "-").replace("‑", "-")
    return normalize_ranges(text)


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
    text = _norm(span)
    if text in body:
        return True
    # A list bullet or a printed HTML line break may survive native extraction
    # while the OCR table keeps its complete item text. Compare that item, not
    # the presentation prefix; ordinary text and standalone markup stay exact.
    item = re.sub(r"^(?:<br\s*/?>)?[•●◦▪]", "", text, flags=re.I)
    if item != text and item and item in body:
        return True
    # Some PDF font maps use Unicode radical characters for their visually
    # identical ideographs. Treat only those characters as a single glyph here;
    # numbers, letters, operators and ordinary ideographs still match exactly.
    if any("CJK RADICAL" in unicodedata.name(ch, "") for ch in text):
        pattern = "".join("." if "CJK RADICAL" in unicodedata.name(ch, "") else re.escape(ch) for ch in text)
        return re.search(pattern, body) is not None
    return False


def _rect(page, box):
    import pymupdf
    return pymupdf.Rect(box[0] * page.rect.width / 1000, box[1] * page.rect.height / 1000,
                        box[2] * page.rect.width / 1000, box[3] * page.rect.height / 1000) & page.rect


def _box(page, rect) -> list[float]:
    return [rect.x0 * 1000 / page.rect.width, rect.y0 * 1000 / page.rect.height,
            rect.x1 * 1000 / page.rect.width, rect.y1 * 1000 / page.rect.height]


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
        for block in blocks:
            if block.block_type not in {"image", "chart", "table"} or not block.bbox or not block.page_idx:
                continue
            if not 1 <= block.page_idx <= len(doc):
                continue
            page = doc[block.page_idx - 1]
            rect = _rect(page, block.bbox)
            if rect.is_empty:
                continue
            native = page.get_text("text", clip=rect, sort=True).strip()
            # A broken font map is not independent evidence.
            if "\ufffd" in native or "\x00" in native:
                native = ""
            if block.block_type != "table":
                context = "\n".join(x for x in (native, block.metadata.get("source_footnote", "")) if x)
                if context:
                    block.metadata["visual_source_context"] = context
                continue
            got = table_evidence_text("\n".join([block.table_markdown or block.text, block.caption or "",
                                    str(block.metadata.get("source_footnote") or "")]))
            # get_text(sort=True) inserts spaces between cells. Compare individual
            # spans, not whole rows joined across Markdown column separators.
            native_spans = [s for b in page.get_text("dict", clip=rect).get("blocks", [])
                            for line in b.get("lines", []) for s in line.get("spans", [])]
            # PDF drawing order can write a wrapped line bottom-first. Evidence
            # supplements follow visible top-to-bottom, then left-to-right order.
            native_spans.sort(key=lambda s: (s.get("bbox", [0,0,0,0])[1], s.get("bbox", [0,0,0,0])[0]))
            spans = [s.get("text", "").strip() for s in native_spans]
            missing = list(dict.fromkeys(s for s in spans if len(_norm(s)) >= 2
                and not re.fullmatch(r"[\s\-‐‑‒–—]+", s) and not _span_present(s, got)))
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
            below = page.get_text("text", clip=pymupdf.Rect(rect.x0, rect.y1, rect.x1, bottom), sort=True).strip()
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
            block.metadata["table_native_grid"] = candidates[0] if native and len(candidates) == 1 else []
            block.metadata["table_source_text"] = native
            block.metadata["table_native_spans"] = [{"text": s.get("text", ""), "bbox": list(s["bbox"])}
                for s in native_spans if s.get("bbox") and native]
            block.metadata["table_source_legend"] = legend
            block.metadata["table_review_reasons"] = reasons
            block.metadata["table_missing_spans"] = missing
            review_rect = pymupdf.Rect(rect.x0, rect.y0, rect.x1, bottom if legend else rect.y1)
            dest = cache_dir / "evidence" / f"{block.block_id}.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            page.get_pixmap(clip=review_rect, dpi=160, alpha=False).save(dest)
            block.metadata["table_review_image"] = str(dest)
