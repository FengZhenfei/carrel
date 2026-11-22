"""Text-layer cross-check for PDFs.

A recogniser that cannot render fonts a PDF does not embed returns such pages without their Chinese text (SimSun
with the GBK-EUC-H encoding, common in PDFs exported by WPS and in many Chinese e-books: only digits and Latin text
came back). The PDF's own text layer still holds the text, so the loss can be seen:
  1. compare, page by page, the CJK character counts of the recogniser's output and of the PyMuPDF text layer;
  2. confirm suspicious pages through the fonts: only CJK drawn with non-embedded fonts and outside figure regions
     counts, so tables of contents and pages the recogniser turned into images are not flagged.
A page confirmed this way stops the parse (pdf_enhanced raises a NonRetryableParseError and drops the cached
recognition result): indexing the document without the text of those pages would lose content without a trace. The cause lies in the recogniser's installation (no CJK
fonts in its container), which a retry does not change; once the fonts are there the file is parsed again.

Until 2026-11 lost pages were rendered into images and sent through the recogniser a second time, and what was
still lost was indexed with a "degraded" mark. That repair was removed: the cause was fixed where it belongs (the
recogniser's image installs CJK fonts), the repair never ran after that, and splicing the re-recognised pages back
lost or repeated the rows of tables merged across pages.
Without PyMuPDF the whole check is skipped. Page numbers follow the pipeline convention, 1-based
(common.page_idx adds one to MinerU's 0-based index); PyMuPDF's 0-based numbers are converted inside this module."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

from ..models import ParsedBlock

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
MIN_LAYER_CJK = 20          # a page needs this many CJK characters drawn with non-embedded fonts to count as "has Chinese to lose"
KEEP_RATIO = 0.5            # MinerU keeping fewer than half of the text layer's CJK characters counts as lost
NON_TEXT_TYPES = {"image", "chart"}   # figure text is written by the VLM stage, not taken from the text layer

DetailFn = Callable[[list[int], dict[int, list[list[float]]]], dict[int, tuple[int, int]]]


def cjk_count(text: str | None) -> int:
    return len(CJK_RE.findall(text or ""))


def text_layer_cjk_by_page(path: Path) -> dict[int, int] | None:
    """CJK characters in the text layer of each page (1-based); None when PyMuPDF is missing or the file
    cannot be read, which skips the check."""
    try:
        import pymupdf  # PyMuPDF, optional dependency
    except ImportError:
        return None
    try:
        with pymupdf.open(str(path)) as doc:
            return {i + 1: cjk_count(page.get_text()) for i, page in enumerate(doc)}
    except Exception:
        return None


def blocks_cjk_by_page(blocks: list[ParsedBlock]) -> dict[int, int]:
    out: dict[int, int] = {}
    for block in blocks:
        if block.page_idx is None or block.block_type in NON_TEXT_TYPES:
            continue
        out[int(block.page_idx)] = out.get(int(block.page_idx), 0) + cjk_count(block.text)
    return out


def candidate_pages(layer: dict[int, int], got: dict[int, int], *, min_layer: int = MIN_LAYER_CJK,
                    keep_ratio: float = KEEP_RATIO) -> list[int]:
    """Cheap first pass: pages whose text layer has Chinese that MinerU did not keep (totals only; whether the
    page really lost text is confirmed by page_details)."""
    return [page for page, n in sorted(layer.items()) if n >= min_layer and got.get(page, 0) < n * keep_ratio]


def figure_boxes_by_page(blocks: list[ParsedBlock]) -> dict[int, list[list[float]]]:
    """Regions MinerU recognised as images / charts, per page; text-layer CJK inside them does not count as
    lost (the VLM stage adds it)."""
    out: dict[int, list[list[float]]] = {}
    for block in blocks:
        if block.page_idx is not None and block.block_type in NON_TEXT_TYPES and block.bbox and len(block.bbox) >= 4:
            out.setdefault(int(block.page_idx), []).append([float(v) for v in block.bbox[:4]])
    return out


def _box_fraction(box: list[float], width: float, height: float) -> tuple[float, float, float, float]:
    """MinerU 3.x bboxes are per-mille page coordinates (0-1000); values above 1000 are taken as PDF points."""
    if all(0 <= v <= 1000 for v in box):
        return box[0] / 1000, box[1] / 1000, box[2] / 1000, box[3] / 1000
    return box[0] / width, box[1] / height, box[2] / width, box[3] / height


def page_detail(page: Any, boxes: list[list[float]]) -> tuple[int, int]:
    """CJK characters of one page's text layer outside figure regions, and how many of them are drawn with
    non-embedded fonts."""
    import pymupdf

    fonts = page.get_fonts(full=True)
    nonembedded = {str(f[3]).split("+", 1)[-1] for f in fonts if f[1] == "n/a"} | {str(f[4]).split("+", 1)[-1] for f in fonts if f[1] == "n/a"}
    width, height = float(page.rect.width), float(page.rect.height)
    rects = []
    for box in boxes:
        x0, y0, x1, y1 = _box_fraction(box, width, height)
        rects.append(pymupdf.Rect(x0 * width, y0 * height, x1 * width, y1 * height))
    total = non = 0
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                count = cjk_count(span.get("text", ""))
                if not count:
                    continue
                rect = pymupdf.Rect(span["bbox"])
                if rect.get_area() > 0 and any((rect & fig).get_area() > 0.5 * rect.get_area() for fig in rects):
                    continue
                total += count
                if str(span.get("font", "")).split("+", 1)[-1] in nonembedded:
                    non += count
    return total, non


def page_details(path: Path, pages: list[int], boxes: dict[int, list[list[float]]]) -> dict[int, tuple[int, int]]:
    """Confirm candidate pages one by one (reads fonts and spans, slower than text_layer_cjk_by_page, hence
    candidates only)."""
    try:
        import pymupdf
    except ImportError:
        return {}
    out: dict[int, tuple[int, int]] = {}
    try:
        with pymupdf.open(str(path)) as doc:
            for page_no in pages:
                if 1 <= page_no <= doc.page_count:
                    out[page_no] = page_detail(doc[page_no - 1], boxes.get(page_no, []))
    except Exception:
        return out
    return out


def lost_text_pages(blocks: list[ParsedBlock], path: Path, *, layer: dict[int, int] | None = None,
                    detail: DetailFn | None = None) -> list[int]:
    """Pages (1-based) whose text layer holds Chinese text, drawn with non-embedded fonts, that the recogniser's
    blocks do not have. Empty when nothing is lost or when the check cannot run (no PyMuPDF, unreadable file).
    layer and detail replace the two reads of the file in tests."""
    layer = text_layer_cjk_by_page(path) if layer is None else layer
    if not layer:
        return []
    got = blocks_cjk_by_page(blocks)
    candidates = candidate_pages(layer, got)
    if not candidates:
        return []
    detail_fn = detail or (lambda pages, boxes: page_details(path, pages, boxes))
    details = detail_fn(candidates, figure_boxes_by_page(blocks))
    return sorted(p for p, (outside, non) in details.items() if non >= MIN_LAYER_CJK and got.get(p, 0) < outside * KEEP_RATIO)
