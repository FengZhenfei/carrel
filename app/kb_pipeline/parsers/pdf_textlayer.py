"""Text-layer cross-check for PDFs.

The MinerU container cannot render non-embedded Chinese fonts (SimSun with the GBK-EUC-H encoding, common in
PDFs exported by WPS and in many Chinese e-books): whole pages of CJK text are blank to it and its output keeps
only digits and Latin text. PyMuPDF reads the text layer in full and ships CJK fallback fonts for rendering. So:
  1. compare, page by page, the CJK character counts of MinerU's output and of the PyMuPDF text layer; suspicious
     pages are confirmed through the fonts (only CJK drawn with non-embedded fonts and outside figure regions count,
     so tables of contents and pages MinerU turned into images are not flagged);
  2. render only the lost pages with PyMuPDF into image pages, send them through MinerU once more and splice the
     result back into the corresponding pages; other pages are untouched;
  3. pages that are still lost get `degraded` on their blocks, carried through the chunk payload to
     `sources[].degraded` in search results. When the resend hits a transient failure (MinerU unreachable,
     5xx) the job first goes through its normal retries; only its last attempt indexes the file as degraded.
Spot check: a 7-page ECG report had 640 CJK characters in the text layer of 6 pages and MinerU kept 7; after the
rendered pass all 6 pages came back. Without PyMuPDF the whole check is skipped and behaviour is unchanged.
Page numbers follow the pipeline convention, 1-based (common.page_idx adds one to MinerU's 0-based index);
PyMuPDF's 0-based numbers are converted inside this module."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

from ..models import ParsedBlock

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
MIN_LAYER_CJK = 20          # a page needs this many CJK characters drawn with non-embedded fonts to count as "has Chinese to lose"
KEEP_RATIO = 0.5            # MinerU keeping fewer than half of the text layer's CJK characters counts as lost
RENDER_DPI = 200
RENDER_JPEG_QUALITY = 90
DEGRADED_REASON = "text_layer_cjk_lost"
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


def render_pages(path: Path, pages: list[int], out: Path, *, dpi: int = RENDER_DPI, quality: int = RENDER_JPEG_QUALITY) -> Path:
    """Render the given pages as JPEG image pages (MuPDF supplies CJK fallback fonts) into one small PDF; page
    sizes are kept and the page order follows `pages`."""
    import pymupdf

    out.parent.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(str(path)) as doc, pymupdf.open() as rendered:
        for page_no in pages:
            page = doc[page_no - 1]
            pix = page.get_pixmap(dpi=dpi)
            target = rendered.new_page(width=page.rect.width, height=page.rect.height)
            target.insert_image(target.rect, stream=pix.tobytes("jpeg", jpg_quality=quality))
        rendered.save(str(out))
    return out


def splice_rendered(blocks: list[ParsedBlock], rendered: list[ParsedBlock], pages: list[int], need: dict[int, int],
                    *, keep_ratio: float = KEEP_RATIO) -> tuple[list[ParsedBlock], list[int], list[int]]:
    """Splice the blocks parsed from the small PDF back by page: its page i (1-based) is pages[i-1]; a recovered
    page (enough CJK against need * keep_ratio) is replaced wholesale by the rendered blocks, other pages stay.
    Rendered blocks without a section_path inherit the one of the preceding original block.
    Returns (merged blocks, recovered pages, pages still lost)."""
    by_page: dict[int, list[ParsedBlock]] = {}
    for block in rendered:
        if block.page_idx is None or not 1 <= int(block.page_idx) <= len(pages):
            continue
        target = pages[int(block.page_idx) - 1]
        block.page_idx = target
        block.block_id = f"rendered-{block.block_id}"
        block.metadata["rendered_page"] = True
        by_page.setdefault(target, []).append(block)
    recovered = [p for p in pages if sum(cjk_count(b.text) for b in by_page.get(p, []) if b.block_type not in NON_TEXT_TYPES)
                 >= need.get(p, 0) * keep_ratio and by_page.get(p)]
    still = [p for p in pages if p not in recovered]
    wanted = set(recovered)
    out: list[ParsedBlock] = []
    emitted: set[int] = set()
    last_section: list[str] = []

    def emit(page_no: int) -> None:
        for rb in by_page[page_no]:
            if not rb.metadata.get("section_path"):
                rb.metadata["section_path"] = list(last_section)
        out.extend(by_page[page_no])
        emitted.add(page_no)

    for block in blocks:
        page_no = None if block.page_idx is None else int(block.page_idx)
        if page_no in wanted:
            if page_no not in emitted:
                emit(page_no)
            continue
        out.append(block)
        if block.metadata.get("section_path"):
            last_section = list(block.metadata["section_path"])
    for page_no in recovered:
        if page_no in emitted:
            continue
        # the original result has no block at all on this page: insert after the last block with a smaller page number
        idx = max((i for i, b in enumerate(out) if b.page_idx is not None and int(b.page_idx) < page_no), default=-1) + 1
        last_section = next((list(b.metadata.get("section_path") or []) for b in reversed(out[:idx]) if b.metadata.get("section_path")), [])
        for rb in by_page[page_no]:
            if not rb.metadata.get("section_path"):
                rb.metadata["section_path"] = list(last_section)
        out[idx:idx] = by_page[page_no]
        emitted.add(page_no)
    return out, recovered, still


def repair_lost_text_layer(blocks: list[ParsedBlock], *, path: Path, rendered_path: Path,
                           parse_rendered: Callable[[Path], list[ParsedBlock]] | None,
                           layer: dict[int, int] | None = None, detail: DetailFn | None = None,
                           raise_transient: bool = False) -> tuple[list[ParsedBlock], dict[str, Any]]:
    """Lost-text check with per-page fallback. Returns (blocks to use, diagnostics): lost_before are the confirmed
    lost pages, recovered the pages the rendered pass brought back, lost the pages still lost at the end.
    With parse_rendered None the pages are only checked, not repaired, and lost pages stay lost. raise_transient:
    a transient failure of the second parse (connection failure, 5xx) is raised so the job goes through its
    normal retries -- degrading in place would index the file as usual, scan would then judge it unchanged and
    those pages would never get another chance; an explicit rejection by MinerU (4xx) and a rendering failure
    would fail the same way on retry, so they always degrade in place."""
    info: dict[str, Any] = {"checked": False, "candidates": [], "lost_before": [], "recovered": [], "lost": [], "rendered": False}
    layer = text_layer_cjk_by_page(path) if layer is None else layer
    if not layer:
        return blocks, info
    info["checked"] = True
    got = blocks_cjk_by_page(blocks)
    candidates = candidate_pages(layer, got)
    info["candidates"] = candidates
    if not candidates:
        return blocks, info
    detail_fn = detail or (lambda pages, boxes: page_details(path, pages, boxes))
    details = detail_fn(candidates, figure_boxes_by_page(blocks))
    need = {p: outside for p, (outside, non) in details.items() if non >= MIN_LAYER_CJK and got.get(p, 0) < outside * KEEP_RATIO}
    lost = sorted(need)
    info["lost_before"] = lost
    info["lost"] = lost
    if not lost or parse_rendered is None:
        return blocks, info
    try:
        render_pages(path, lost, rendered_path)
    except Exception as exc:              # rendering failed: keep the original result, lost pages still get degraded
        info["error"] = f"{type(exc).__name__}: {exc}"
        return blocks, info
    try:
        rendered = parse_rendered(rendered_path)
    except Exception as exc:
        if raise_transient and not getattr(exc, "deterministic", False):
            raise
        info["error"] = f"{type(exc).__name__}: {exc}"
        return blocks, info
    info["rendered"] = True
    merged, recovered, still = splice_rendered(blocks, rendered, lost, need)
    info["recovered"] = recovered
    info["lost"] = still
    return merged, info


def page_range(block: ParsedBlock) -> tuple[int, int] | None:
    """Pages a block covers: merged cross-page blocks carry page_start / page_end in metadata, otherwise page_idx."""
    meta = block.metadata if isinstance(block.metadata, dict) else {}
    start = meta.get("page_start", block.page_idx)
    end = meta.get("page_end", block.page_idx)
    if start is None and end is None:
        return None
    start = end if start is None else start
    end = start if end is None else end
    return int(start), int(end)


def mark_degraded_pages(blocks: list[ParsedBlock], pages: list[int], *, reason: str = DEGRADED_REASON) -> int:
    """Mark blocks on lost pages (including the page range of merged blocks) as degraded; returns the count."""
    wanted = set(pages)
    if not wanted:
        return 0
    marked = 0
    for block in blocks:
        rng = page_range(block)
        if rng is not None and any(p in wanted for p in range(rng[0], rng[1] + 1)):
            block.metadata["degraded"] = reason
            marked += 1
    return marked
