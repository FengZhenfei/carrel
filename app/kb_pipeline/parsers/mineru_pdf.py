from __future__ import annotations

import base64
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import re
import statistics
from typing import Any

from ..models import ParsedBlock
from ..headings import (
    detect_toc_pages,
    explicit_toc_entries,
    infer_heading_level,
    is_mismarked_heading,
    recoverable_chapter_lines,
)
from .common import HeadingResolver, SectionTracker, bbox, caption_of, clean_heading_text, clean_text, footnote_of, heading_level, item_text, label_of, markdown_to_blocks, page_idx, walk_dicts
from .errors import FAILURE_SCOPE_MINERU
from .service_clients import call_mineru_sync, extract_content_list, extract_images, extract_markdown, mineru_parser_profile
from .mineru_quality import merge_table_placeholders, recover_pdf_items
from .pdf_code import repair_code_items
from .pdf_table_layout import attach_table_notes

# Block-level provenance tag, written into parser_profile in the chunk payload; whether a file is re-parsed is
# decided by the route-level version (common.parser_profile_for_path). The tag does not name the MinerU
# backend: the backend is chosen at run time (MINERU_BACKEND or the parser container, see mineru_backend), and
# a name hard-coded here would not match the one actually used
PDF_PARSER_PROFILE = "mineru-3.4.4-pdf-v3"
_PAGE_FURNITURE = frozenset({"header", "footer", "page_header", "page_footer", "page_number"})
_NON_SECTION_FURNITURE = _PAGE_FURNITURE | frozenset({"aside_text"})
_SUBSTANTIVE_BODY_TYPES = frozenset({
    "text", "list", "table", "equation", "formula", "interline_equation", "code", "algorithm",
})
_SUBSTANTIVE_IMAGE_AREA = 2000.0   # in 0-1000 normalized page coords (~45x45); smaller crops are icons/avatars
_NAV_ARROW_RE = re.compile(
    r"^\s*(?:[←→↩↪⬅➡◀▶◁▷❮❯‹›«»]|<-|->|[<＜](?!\s*[+-]?\d)(?![A-Za-z][A-Za-z0-9_-]*>)|>(?!\s*[+-]?\d)\s)"
    r"|\s(?:[←→↩↪⬅➡◀▶◁▷❮❯‹›«»]|<-|->)\s*$"
)
_REQ_STAR_RE = re.compile(r"^\s*\\?\*\s*\S|\S\s*\\?\*\s*$")
_FIG_CAPTION_RE = re.compile(r"^\s*(?:图|圖|Figure|Fig\.?)\s*\d", re.IGNORECASE)
_SEQ_NUM_LINE_RE = re.compile(r"^\s*(\d{1,2})\s*[.、．]\s*\S")
_TOP_TITLE_END_PUNCT_RE = re.compile(r"[。！？!?；;，,:：]$")


@dataclass
class _PdfLayoutSignals:
    """Layout-derived heading signals from the source PDF, computed in a single PyMuPDF pass."""

    embedded_ocr: set[int] = field(default_factory=set)
    demoted_embedded: set[int] = field(default_factory=set)
    local_image_titles: dict[int, tuple[int, int]] = field(default_factory=dict)
    local_image_members: dict[int, tuple[int, int]] = field(default_factory=dict)
    recovered_top_titles: set[int] = field(default_factory=set)


def _box_overlap_frac(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    area_a = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    return ((ix1 - ix0) * (iy1 - iy0)) / area_a


def _same_column_below(
    top_box: tuple[float, float, float, float] | None,
    bot_box: tuple[float, float, float, float] | None,
) -> bool:
    if not top_box or not bot_box:
        return False
    if bot_box[1] < top_box[3] - 8.0:
        return False
    tw = max(1.0, top_box[2] - top_box[0])
    bw = max(1.0, bot_box[2] - bot_box[0])
    ov = min(top_box[2], bot_box[2]) - max(top_box[0], bot_box[0])
    return ov >= 0.3 * min(tw, bw) or abs(bot_box[0] - top_box[0]) <= 35.0


def _sidebar_toc_run(
    title_pos: int,
    img_items: list[tuple[int, dict[str, Any]]],
) -> list[int]:
    """Return item indices of a sidebar mini-TOC starting at title_pos inside an embedded webpage screenshot
    (a heading followed directly in the same column by >= 3 sequential numbered short outline items with no
    intervening prose), or [] if title_pos does not head a sidebar mini-TOC."""
    t_idx, t_it = img_items[title_pos]
    tbb_raw = bbox(t_it)
    tbb = (float(tbb_raw[0]), float(tbb_raw[1]), float(tbb_raw[2]), float(tbb_raw[3])) if tbb_raw and len(tbb_raw) == 4 else None
    nums: list[int] = []
    prose = 0
    run_indices = [t_idx]
    for pos in range(title_pos + 1, len(img_items)):
        idx, it = img_items[pos]
        lbl = label_of(it)
        if lbl not in ("text", "title", "list"):
            continue
        bb_raw = bbox(it)
        bb = (float(bb_raw[0]), float(bb_raw[1]), float(bb_raw[2]), float(bb_raw[3])) if bb_raw and len(bb_raw) == 4 else None
        if tbb and bb and not _same_column_below(tbb, bb):
            continue
        txt = item_text(it).strip()
        if not txt:
            continue
        if _effective_heading_level(it) and not _SEQ_NUM_LINE_RE.match(txt):
            break
        run_indices.append(idx)
        for ln in txt.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            m = _SEQ_NUM_LINE_RE.match(ln)
            if m and len(ln) <= 45:
                nums.append(int(m.group(1)))
            elif len(ln) > 35:
                prose += 1
    if len(nums) >= 3 and prose == 0:
        inc = sum(b == a + 1 for a, b in zip(nums, nums[1:]))
        if inc >= 2:
            return run_indices
    return []


def _effective_heading_level(item: dict[str, Any]) -> int | None:
    """Treat legacy/fallback type='title' items without text_level uniformly as parser level 1 so every
    heading filter and HeadingResolver sees the exact same candidates (closes M1)."""
    lv = heading_level(item)
    if lv is not None:
        return lv
    return 1 if "title" in label_of(item) else None


def _norm_alnum_cjk(text: str) -> str:
    return "".join(ch.casefold() for ch in str(text or "") if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")


def _analyze_pdf_heading_layout(path: Path, items: list[dict[str, Any]]) -> _PdfLayoutSignals:
    """Classify OCR-only text inside embedded raster images and recover unmarked templated top-of-page slide
    titles in a single PyMuPDF pass.

    Three roles for marked titles OCR'd inside a non-full-page embedded image (with no overlapping native text):
      1. Standalone graphic title / logo-lockup: sole text-like item in the image, title covers >= 25% of the
         image area, no figure caption below, no navigation arrow, followed by substantive content on the page ->
         normal document heading.
      2. Local in-image card/section heading: inside a multi-item infographic/card image (neither captioned as an
         illustrative figure nor a UI form dialog), not a navigation arrow or sidebar mini-TOC, and directly
         leading substantive body content in the same column inside the same image -> kept as a local heading
         strictly scoped to its image so it preserves card attribution without leaking into outer sections.
      3. UI screenshot controls, form labels, sidebar mini-TOCs, captioned example figures, and unsupported
         labels -> demoted to searchable body text.

    Scanned PDFs (even with digital headers/footers/Bates stamps), rotated pages, tiled-vs-untiled mismatches,
    and native titles drawn over a background image are left untouched.
    """
    signals = _PdfLayoutSignals()
    if not path or not path.is_file():
        return signals
    try:
        import pymupdf
    except ImportError:
        return signals
    try:
        doc = pymupdf.open(str(path))
    except Exception:
        return signals
    with doc:
        if doc.page_count <= 0:
            return signals
        max_page = max((page_idx(it) or 0) for it in items) if items else 0
        if max_page > doc.page_count:
            return signals  # tiled MinerU result paired with untiled source PDF: skip to avoid misaligned coords
        sample_count = min(doc.page_count, 10)
        native_chars = sum(len(doc[p].get_text("text").strip()) for p in range(sample_count))
        if native_chars < 100 and doc.page_count > 10:
            step = max(1, doc.page_count // 10)
            native_chars += sum(len(doc[p].get_text("text").strip()) for p in range(10, doc.page_count, step))
        if native_chars < 100:
            return signals  # scanned PDF: rely on MinerU OCR headings

        text_items_by_page: dict[int, list[tuple[int, tuple[float, float, float, float], str]]] = {}
        nf_items_by_page: dict[int, list[tuple[int, dict[str, Any], tuple[float, float, float, float] | None]]] = {}
        for idx, it in enumerate(items):
            p1 = page_idx(it)
            if not p1 or not (1 <= p1 <= doc.page_count):
                continue
            lbl = label_of(it)
            bb = bbox(it)
            box: tuple[float, float, float, float] | None = None
            if bb and len(bb) == 4:
                cand = (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))
                if all(-5.0 <= v <= 1005.0 for v in cand) and cand[2] > cand[0] and cand[3] > cand[1]:
                    box = cand
            if lbl not in _NON_SECTION_FURNITURE:
                nf_items_by_page.setdefault(p1, []).append((idx, it, box))
            if lbl in ("text", "title") and box is not None:
                text_items_by_page.setdefault(p1, []).append((idx, box, item_text(it)))

        # Part A: Embedded raster image classification
        for p1, plist in text_items_by_page.items():
            page = doc[p1 - 1]
            if page.rotation:
                continue  # rotated page coordinates differ from MinerU's upright render
            pw, ph = float(page.rect.width) or 1.0, float(page.rect.height) or 1.0
            img_rects: list[tuple[float, float, float, float]] = []
            seen_rects: set[tuple[float, float, float, float]] = set()
            for im in page.get_image_info(xrefs=False):
                ib = im.get("bbox")
                if not ib or len(ib) != 4:
                    continue
                x0 = max(0.0, min(1000.0, ib[0] / pw * 1000.0))
                y0 = max(0.0, min(1000.0, ib[1] / ph * 1000.0))
                x1 = max(0.0, min(1000.0, ib[2] / pw * 1000.0))
                y1 = max(0.0, min(1000.0, ib[3] / ph * 1000.0))
                w_f = (x1 - x0) / 1000.0
                h_f = (y1 - y0) / 1000.0
                # Exclude full-page background/scan canvases and tiny inline icons.
                if not (w_f > 0.85 and h_f > 0.85) and w_f >= 0.05 and h_f >= 0.05:
                    key = (round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1))
                    if key not in seen_rects:
                        seen_rects.add(key)
                        img_rects.append((x0, y0, x1, y1))
            if not img_rects:
                continue
            inside_cands: list[tuple[int, tuple[float, float, float, float]]] = []
            outside_items: list[tuple[tuple[float, float, float, float], str]] = []
            for idx, bb, txt in plist:
                if any(_box_overlap_frac(bb, ir) >= 0.7 for ir in img_rects):
                    inside_cands.append((idx, bb))
                elif all(_box_overlap_frac(bb, ir) < 0.5 for ir in img_rects):
                    outside_items.append((bb, txt))
            if not inside_cands or not outside_items:
                # Without non-furniture body/title text outside the embedded images, this page may be a
                # scanned page with only a digital header/footer/Bates stamp: leave OCR headings alone.
                continue
            words = page.get_text("words")
            if not words:
                continue
            wentries = [
                ((w[0] / pw * 1000.0, w[1] / ph * 1000.0, w[2] / pw * 1000.0, w[3] / ph * 1000.0), str(w[4] or ""))
                for w in words
            ]
            # Confirm MinerU coordinates align with this page's native body text outside embedded images.
            aligned = False
            for obb, otxt in outside_items:
                onorm = _norm_alnum_cjk(otxt)
                if not onorm:
                    continue
                padded = (obb[0] - 12.0, obb[1] - 12.0, obb[2] + 12.0, obb[3] + 12.0)
                overlap_text = "".join(
                    wtxt for wr, wtxt in wentries
                    if _box_overlap_frac(wr, padded) > 0.1 or _box_overlap_frac(obb, wr) > 0.1
                )
                wnorm = _norm_alnum_cjk(overlap_text)
                if wnorm and (wnorm in onorm or onorm in wnorm or len(set(wnorm) & set(onorm)) >= min(2, len(set(onorm)))):
                    aligned = True
                    break
            if not aligned:
                continue

            def _has_native_overlap(bb: tuple[float, float, float, float] | None) -> bool:
                if bb is None:
                    return False
                padded = (bb[0] - 8.0, bb[1] - 8.0, bb[2] + 8.0, bb[3] + 8.0)
                return any(
                    _box_overlap_frac(wr, padded) > 0.1 or _box_overlap_frac(bb, wr) > 0.1
                    for wr, _ in wentries
                )

            page_emb: set[int] = set()
            for idx, bb in inside_cands:
                if not _has_native_overlap(bb):
                    page_emb.add(idx)
            if not page_emb:
                continue
            signals.embedded_ocr.update(page_emb)

            # Map every non-furniture item on this page to its smallest containing embedded image rect.
            p_nf = nf_items_by_page.get(p1, [])
            owner_img: dict[int, int] = {}
            by_img: dict[int, list[tuple[int, dict[str, Any]]]] = {}
            for idx, it, box in p_nf:
                if box is None:
                    continue
                cands = [j for j, ir in enumerate(img_rects) if _box_overlap_frac(box, ir) >= 0.7]
                if cands:
                    best = min(cands, key=lambda j: (img_rects[j][2] - img_rects[j][0]) * (img_rects[j][3] - img_rects[j][1]))
                    owner_img[idx] = best
                    by_img.setdefault(best, []).append((idx, it))

            # Merge nested single-item title pills into a larger enclosing multi-item OCR image so a card
            # badge sitting inside a multi-card infographic is treated as Role 2 (local) instead of Role 1.
            merged_into: dict[int, int] = {}
            for j, j_items in list(by_img.items()):
                if len(j_items) != 1:
                    continue
                ir = img_rects[j]
                area_j = max(1.0, (ir[2] - ir[0]) * (ir[3] - ir[1]))
                parents = [
                    k for k, k_items in by_img.items()
                    if k != j
                    and (img_rects[k][2] - img_rects[k][0]) * (img_rects[k][3] - img_rects[k][1]) > area_j * 1.2
                    and _box_overlap_frac(ir, img_rects[k]) >= 0.85
                    and any(i in page_emb for i, _ in k_items)
                ]
                if parents:
                    parent = min(
                        parents,
                        key=lambda k: (img_rects[k][2] - img_rects[k][0]) * (img_rects[k][3] - img_rects[k][1]),
                    )
                    merged_into[j] = parent
            for j, parent in merged_into.items():
                while parent in merged_into:
                    parent = merged_into[parent]
                for idx, it in by_img.pop(j, []):
                    owner_img[idx] = parent
                    by_img.setdefault(parent, []).append((idx, it))
            for k in set(merged_into.values()):
                if k in by_img:
                    by_img[k].sort(key=lambda pair: pair[0])

            for j, img_items in by_img.items():
                ir = img_rects[j]
                img_area = max(1.0, (ir[2] - ir[0]) * (ir[3] - ir[1]))
                emb_in_img = [idx for idx, _ in img_items if idx in page_emb]
                emb_titles = [
                    (pos, idx, it)
                    for pos, (idx, it) in enumerate(img_items)
                    if idx in page_emb and _effective_heading_level(it)
                ]
                if not emb_titles:
                    signals.demoted_embedded.update(emb_in_img)
                    continue

                # Check for an explicit figure caption directly below this image (horizontally overlapping)
                # or stored on an image item's image_caption field.
                has_fig_caption = any(
                    owner_img.get(i2) != j
                    and b2 is not None
                    and min(ir[2], b2[2]) > max(ir[0], b2[0])
                    and ir[3] - 10.0 <= b2[1] <= ir[3] + 65.0
                    and _FIG_CAPTION_RE.match(item_text(it2))
                    for i2, it2, b2 in p_nf
                ) or any(
                    _FIG_CAPTION_RE.match(caption_of(it2))
                    for _, it2 in img_items
                )

                # Role 1: Standalone graphic title / logo-lockup (sole text-like item, dominant area >= 0.25).
                text_like = [(idx, it) for idx, it in img_items if label_of(it) in ("text", "title", "list", "table")]
                if len(text_like) == 1 and len(emb_titles) == 1 and not has_fig_caption:
                    _, idx, it = emb_titles[0]
                    txt = item_text(it)
                    bb = bbox(it)
                    area_f = ((float(bb[2]) - float(bb[0])) * (float(bb[3]) - float(bb[1]))) / img_area if bb and len(bb) == 4 else 0.0
                    p_pos = next((k for k, (pi, _, _) in enumerate(p_nf) if pi == idx), -1)
                    has_after = False
                    if p_pos >= 0:
                        for _, nit, _ in p_nf[p_pos + 1 :]:
                            if _effective_heading_level(nit) and label_of(nit) in ("text", "title"):
                                break  # do not scan past the next heading into an unrelated section
                            if label_of(nit) in _SUBSTANTIVE_BODY_TYPES and item_text(nit).strip():
                                has_after = True
                                break
                    if area_f >= 0.25 and not is_mismarked_heading(txt) and not _NAV_ARROW_RE.search(txt) and has_after:
                        continue  # Role 1: flows normally through outer HeadingResolver & SectionTracker

                # Image-level disqualifiers for multi-item images: captioned example figure or UI form dialog.
                req_stars = sum(
                    1 for idx, it in img_items
                    if idx in page_emb
                    and label_of(it) in ("text", "title")
                    and "\n" not in item_text(it).strip()
                    and len(item_text(it).strip()) <= 30
                    and _REQ_STAR_RE.search(item_text(it).strip())
                )
                navigation = [
                    idx for idx, it in img_items
                    if idx in page_emb
                    and label_of(it) in ("text", "title")
                    and _NAV_ARROW_RE.search(item_text(it))
                ]
                # Navigation plus a source-captioned illustration identifies an example screen, too.
                captioned_screen = navigation and navigation[0] < emb_titles[0][1] and any(
                    caption_of(it) for _, it in img_items if label_of(it) == "image"
                )
                if has_fig_caption or req_stars >= 2 or (req_stars and navigation) or captioned_screen:
                    signals.demoted_embedded.update(emb_in_img)
                    continue

                # Non-native body items inside this image (OCR-only text/titles plus non-native lists/tables).
                non_native_in_img: set[int] = set(emb_in_img)
                for idx, it in img_items:
                    if idx in non_native_in_img:
                        continue
                    if label_of(it) in ("list", "table"):
                        bb_raw = bbox(it)
                        b_it = (float(bb_raw[0]), float(bb_raw[1]), float(bb_raw[2]), float(bb_raw[3])) if bb_raw and len(bb_raw) == 4 else None
                        if b_it is not None and not _has_native_overlap(b_it):
                            non_native_in_img.add(idx)

                # Per-item Role 2 (local in-image card/section heading) vs Role 3 (demoted) check.
                local_key = (p1, j)
                accepted_local: set[int] = set()
                toc_run_indices: set[int] = set()
                title_body_match: dict[int, tuple[tuple[float, float, float, float] | None, int | None, tuple[float, float, float, float] | None]] = {}
                for pos, idx, it in emb_titles:
                    if idx in toc_run_indices:
                        signals.demoted_embedded.add(idx)
                        continue
                    txt = item_text(it)
                    if is_mismarked_heading(txt) or _NAV_ARROW_RE.search(txt):
                        signals.demoted_embedded.add(idx)
                        continue
                    toc_run = _sidebar_toc_run(pos, img_items)
                    if toc_run:
                        signals.demoted_embedded.add(idx)
                        toc_run_indices.update(toc_run)
                        continue
                    bb = bbox(it)
                    tbb = (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])) if bb and len(bb) == 4 else None
                    matched_body_idx: int | None = None
                    matched_body_box: tuple[float, float, float, float] | None = None
                    for pos2 in range(pos + 1, len(img_items)):
                        i2, it2 = img_items[pos2]
                        if i2 not in non_native_in_img:
                            continue  # ignore native PDF text drawn over a background image
                        bb2_raw = bbox(it2)
                        bb2 = (float(bb2_raw[0]), float(bb2_raw[1]), float(bb2_raw[2]), float(bb2_raw[3])) if bb2_raw and len(bb2_raw) == 4 else None
                        if _effective_heading_level(it2):
                            if _same_column_below(tbb, bb2):
                                break  # another heading/control sits directly below before any body prose
                            continue
                        if label_of(it2) in _SUBSTANTIVE_BODY_TYPES and item_text(it2).strip():
                            if _same_column_below(tbb, bb2):
                                matched_body_idx = i2
                                matched_body_box = bb2
                                break
                    if matched_body_idx is not None:
                        accepted_local.add(idx)
                        title_body_match[idx] = (tbb, matched_body_idx, matched_body_box)
                    else:
                        signals.demoted_embedded.add(idx)

                # Reject horizontal tab/navbar bars: >= 2 same-row titles all claiming the same much-wider body block.
                if len(accepted_local) >= 2:
                    by_body: dict[int, list[tuple[int, tuple[float, float, float, float], tuple[float, float, float, float]]]] = defaultdict(list)
                    for tidx in accepted_local:
                        tbb, bidx, bbb = title_body_match[tidx]
                        if tbb and bidx is not None and bbb:
                            by_body[bidx].append((tidx, tbb, bbb))
                    for bidx, group in by_body.items():
                        if len(group) < 2:
                            continue
                        bw = max(1.0, group[0][2][2] - group[0][2][0])
                        for tidx, tbb, _ in group:
                            tw = max(1.0, tbb[2] - tbb[0])
                            same_row_peers = sum(
                                1 for oidx, obb, _ in group
                                if oidx != tidx and abs(obb[1] - tbb[1]) <= 15.0
                            )
                            if same_row_peers >= 1 and bw > tw * 1.8:
                                accepted_local.discard(tidx)
                                signals.demoted_embedded.add(tidx)

                if accepted_local:
                    for idx in accepted_local:
                        signals.local_image_titles[idx] = local_key
                    # Only attach OCR-only / non-native items inside this image — never native PDF text or
                    # scattered sub-icon image crops — so native headings and adjacent panels stay untouched.
                    for idx in non_native_in_img:
                        if idx not in toc_run_indices:
                            signals.local_image_members[idx] = local_key
                    for idx in emb_in_img:
                        if idx not in accepted_local:
                            signals.demoted_embedded.add(idx)
                else:
                    signals.demoted_embedded.update(emb_in_img)

        # Part B: Recover unmarked top-of-page slide/page titles that match a consistent document-wide
        # top-title template slot (prevents new slides with clear own topics from inheriting the previous
        # slide's trailing subsection while keeping continuous cross-page prose untouched).
        if doc.page_count >= 3:
            top_cands_by_page: dict[int, list[tuple[int, dict[str, Any], tuple[float, float, float, float]]]] = defaultdict(list)
            for p1, plist in text_items_by_page.items():
                if doc[p1 - 1].rotation:
                    continue
                for idx, box, _ in plist:
                    if idx in signals.embedded_ocr or box[1] > 130.0 or box[3] > 200.0:
                        continue
                    top_cands_by_page[p1].append((idx, items[idx], box))
            top_counts = Counter(
                clean_heading_text(item_text(it))
                for cands in top_cands_by_page.values()
                for _, it, _ in cands
                if clean_heading_text(item_text(it))
            )
            marked_slots = []
            for p1, cands in top_cands_by_page.items():
                page_marked = [
                    (p1, idx, box)
                    for idx, it, box in cands
                    if _effective_heading_level(it)
                    and not is_mismarked_heading(item_text(it))
                    and len(clean_heading_text(item_text(it))) >= 2
                    and top_counts[clean_heading_text(item_text(it))] == 1
                ]
                if page_marked:
                    marked_slots.append(min(page_marked, key=lambda s: s[2][1]))
            if len(marked_slots) >= 3:
                med_y0 = statistics.median(b[1] for _, _, b in marked_slots)
                med_y1 = statistics.median(b[3] for _, _, b in marked_slots)
                med_x0 = statistics.median(b[0] for _, _, b in marked_slots)
                med_h = statistics.median(b[3] - b[1] for _, _, b in marked_slots)
                consistent = [
                    s for s in marked_slots
                    if abs(s[2][1] - med_y0) <= 15.0
                    and abs(s[2][3] - med_y1) <= 18.0
                    and abs(s[2][0] - med_x0) <= 35.0
                ]
                land_pages = sum(1 for p in doc if not p.rotation and p.rect.width > p.rect.height * 1.2)
                is_slide_deck = land_pages >= 0.5 * doc.page_count
                if len(consistent) >= 3 and (is_slide_deck or med_h >= 28.0):
                    def _span_font_metrics(
                        p1: int, box: tuple[float, float, float, float], cand_text: str
                    ) -> tuple[float | None, float | None, bool]:
                        pg = doc[p1 - 1]
                        pw, ph = float(pg.rect.width) or 1.0, float(pg.rect.height) or 1.0
                        rx0, ry0 = box[0] * pw / 1000.0 - 1.5, box[1] * ph / 1000.0 - 1.5
                        rx1, ry1 = box[2] * pw / 1000.0 + 1.5, box[3] * ph / 1000.0 + 1.5
                        slot_sizes: list[float] = []
                        slot_txt_parts: list[str] = []
                        body_sizes: list[float] = []
                        for blk in pg.get_text("dict").get("blocks", []):
                            for ln in blk.get("lines", []):
                                for sp in ln.get("spans", []):
                                    sb = sp.get("bbox")
                                    stxt = str(sp.get("text") or "").strip()
                                    if not sb or not stxt:
                                        continue
                                    sz = float(sp.get("size") or 0.0)
                                    # Check center or >= 30% overlap with slot box so adjacent slide numbers don't bleed in.
                                    ix0, iy0 = max(rx0, sb[0]), max(ry0, sb[1])
                                    ix1, iy1 = min(rx1, sb[2]), min(ry1, sb[3])
                                    sp_area = max(1e-6, (sb[2] - sb[0]) * (sb[3] - sb[1]))
                                    ov = ((ix1 - ix0) * (iy1 - iy0)) / sp_area if ix1 > ix0 and iy1 > iy0 else 0.0
                                    if ov >= 0.3:
                                        slot_sizes.extend([sz] * min(len(stxt), 20))
                                        slot_txt_parts.append(stxt)
                                    elif sb[1] / ph * 1000.0 > box[3]:
                                        body_sizes.extend([sz] * min(len(stxt), 20))
                        cnorm = _norm_alnum_cjk(cand_text)
                        snorm = _norm_alnum_cjk("".join(slot_txt_parts))
                        aligned_txt = bool(
                            cnorm and snorm
                            and (snorm in cnorm or cnorm in snorm or len(set(cnorm) & set(snorm)) >= min(2, len(set(cnorm))))
                        )
                        return (
                            statistics.median(slot_sizes) if slot_sizes else None,
                            statistics.median(body_sizes) if body_sizes else None,
                            aligned_txt,
                        )

                    template_sizes: list[float] = []
                    for tp1, tidx, tbox in consistent:
                        tfs, _, taligned = _span_font_metrics(tp1, tbox, item_text(items[tidx]))
                        if tfs is not None and taligned:
                            template_sizes.append(tfs)
                    med_slot_fs = statistics.median(template_sizes) if len(template_sizes) >= 2 else None

                    if med_slot_fs is not None:
                        for p1, cands in top_cands_by_page.items():
                            for idx, it, box in cands:
                                if _effective_heading_level(it):
                                    continue
                                txt = item_text(it).strip()
                                ct = clean_heading_text(txt)
                                if (
                                    len(ct) < 2
                                    or top_counts[ct] > 1
                                    or is_mismarked_heading(txt)
                                    or _TOP_TITLE_END_PUNCT_RE.search(txt)
                                ):
                                    continue
                                if not (
                                    abs(box[1] - med_y0) <= 15.0
                                    and abs(box[3] - med_y1) <= 18.0
                                    and abs(box[0] - med_x0) <= 35.0
                                    and (box[3] - box[1]) >= med_h * 0.75
                                ):
                                    continue
                                fs, bfs, aligned_txt = _span_font_metrics(p1, box, txt)
                                # Fail closed: require aligned native font size matching the template title slot.
                                if not aligned_txt or fs is None or fs < med_slot_fs * 0.85:
                                    continue
                                if bfs is not None and fs <= bfs + 1.5:
                                    continue
                                signals.recovered_top_titles.add(idx)
                                break  # at most one recovered top-slot title per page

    return signals


def _embedded_image_text_indices(path: Path, items: list[dict[str, Any]]) -> set[int]:
    """Compatibility helper returning embedded-image OCR items demoted from heading eligibility."""
    return _analyze_pdf_heading_layout(path, items).demoted_embedded


def _unsupported_recurring_titles(
    items: list[dict[str, Any]],
    *,
    excluded_indices: set[int],
    recovered_top_titles: set[int] | None = None,
    repeat_limit: int = 3,
) -> set[str]:
    """Recurring unnumbered marked headings that lack layout/content ownership across their occurrences
    (e.g., repeated UI collapse controls at card footers or column labels followed by unrelated columns).

    A recurring unnumbered heading stays a section heading when at least half of its occurrences introduce
    either substantive content (text/list/table/equation/code or a non-icon figure/diagram) or a same-flow
    child heading below it, and it is not predominantly a trailing right-shifted label with no direct content.
    """
    extra_titles = recovered_top_titles or set()
    nf: list[tuple[int, dict[str, Any]]] = [
        (i, it) for i, it in enumerate(items) if label_of(it) not in _NON_SECTION_FURNITURE
    ]

    def is_active_title(orig_idx: int, it: dict[str, Any]) -> bool:
        return (
            orig_idx not in excluded_indices
            and label_of(it) in ("text", "title")
            and (bool(_effective_heading_level(it)) or orig_idx in extra_titles)
            and not is_mismarked_heading(item_text(it))
        )

    counts = Counter(
        clean_heading_text(item_text(it))
        for orig_idx, it in nf
        if is_active_title(orig_idx, it)
        and clean_heading_text(item_text(it))
        and infer_heading_level(item_text(it)) is None
    )
    recurring = {t for t, n in counts.items() if n >= repeat_limit}
    if not recurring:
        return set()

    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])  # total, owns_content, owns_child_title, trailing_shifted
    for pos, (orig_idx, it) in enumerate(nf):
        if not is_active_title(orig_idx, it):
            continue
        ct = clean_heading_text(item_text(it))
        if ct not in recurring:
            continue
        st = stats[ct]
        st[0] += 1
        bb = bbox(it)
        pg = page_idx(it)
        page_centered = bool(bb and len(bb) == 4 and abs(((float(bb[0]) + float(bb[2])) / 2.0) - 500.0) <= 80.0)
        if pos > 0 and bb and len(bb) == 4 and not page_centered:
            prev_idx, prev_it = nf[pos - 1]
            pbb = bbox(prev_it)
            # Only count as trailing-shifted when strictly below the preceding body block in vertical flow
            # (never a 2-column wrap from column 1 bottom to column 2 top).
            if (
                page_idx(prev_it) == pg
                and pbb
                and len(pbb) == 4
                and label_of(prev_it) in _SUBSTANTIVE_BODY_TYPES
                and not is_active_title(prev_idx, prev_it)
                and float(bb[1]) >= float(pbb[3]) - 10.0
                and (float(bb[0]) - float(pbb[0])) > 40.0
            ):
                st[3] += 1

        owns_content = False
        next_title: dict[str, Any] | None = None
        saw_small_image = False
        for nxt_idx, nxt_it in nf[pos + 1 :]:
            if is_active_title(nxt_idx, nxt_it):
                next_title = nxt_it
                break
            nl = label_of(nxt_it)
            if nl in _SUBSTANTIVE_BODY_TYPES and item_text(nxt_it).strip():
                owns_content = True
                break
            if any(k in nl for k in ("image", "figure", "chart", "picture")):
                nbb = bbox(nxt_it)
                if not nbb or len(nbb) != 4:
                    owns_content = True
                    break
                area = max(0.0, float(nbb[2]) - float(nbb[0])) * max(0.0, float(nbb[3]) - float(nbb[1]))
                if area >= _SUBSTANTIVE_IMAGE_AREA:
                    owns_content = True
                    break
                saw_small_image = True
        if owns_content:
            st[1] += 1
        elif next_title is not None:
            nt_bb = bbox(next_title)
            nt_pg = page_idx(next_title)
            if bb is None or len(bb) != 4 or nt_bb is None or len(nt_bb) != 4:
                st[2] += 1
            elif nt_pg == pg and float(nt_bb[1]) >= float(bb[1]) - 10.0 and (
                float(nt_bb[0]) >= float(bb[0]) - 25.0 or (page_centered and not saw_small_image)
            ):
                st[2] += 1
            elif pg is not None and nt_pg == pg + 1 and (
                float(nt_bb[0]) >= float(bb[0]) - 25.0 or (page_centered and not saw_small_image)
            ):
                st[2] += 1

    demoted: set[str] = set()
    for ct, (total, owns_content, owns_child_title, trailing_shifted) in stats.items():
        if (owns_content + owns_child_title) < 0.5 * total or (
            trailing_shifted >= 0.5 * total and owns_content < 0.5 * total
        ):
            demoted.add(ct)
    return demoted


class MinerUServiceError(RuntimeError):
    """Carries the HTTP status code: retrying a 4xx (corrupt document, rejected by MinerU) is
    pointless, while connection / read timeouts should be retried. The two used to be squashed into
    the same string exception."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code

    @property
    def deterministic(self) -> bool:
        return self.status_code in {400, 413, 415, 422}

    @property
    def config_error(self) -> bool:
        return self.status_code in {401, 403, 404, 409}

    @property
    def config_scope(self) -> str | None:
        return FAILURE_SCOPE_MINERU if self.config_error else None


def mineru_pdf_blocks(
    *,
    mineru_url: str,
    path: Path,
    cache_dir: Path,
    timeout: int = 43200,
) -> list[ParsedBlock]:
    out_json = cache_dir / "mineru" / "result.json"
    out_md = cache_dir / "mineru" / "result.md"
    try:
        payload = call_mineru_sync(
            mineru_url,
            path,
            out_json,
            out_md,
            table_enable=True,
            return_middle_json=True,
            timeout=timeout,
        )
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        raise MinerUServiceError(str(exc), status_code=status) from exc

    image_map = save_mineru_images(extract_images(payload), cache_dir / "mineru" / "images")
    content = extract_content_list(payload)
    items = recover_pdf_items(content if content else list(walk_dicts(payload)), payload)
    items = attach_table_notes(repair_code_items(items, path), path)
    items = merge_table_placeholders(items, path)

    blocks: list[ParsedBlock] = []
    seen: set[tuple] = set()
    sections = SectionTracker()
    # Heading levels: MinerU's text_level is combined with layout inference (CJK chapter numbering /
    # 1.2.3 / Chapter N), then corrected at document level -- recurring labels (KEYWORD / Note:) and
    # list items are not headings, and unnumbered headings hang under the nearest numbered one.
    # Body lines MinerU did not mark stay body text: inference only sets the level of a block that is
    # already marked as a heading (see headings.HeadingResolver). The one exception is a chapter heading of the
    # document that MinerU failed to mark ("第 3 章 安装", "CHAPTER 02"), found by looking at the whole document
    # (headings.recoverable_chapter_lines)
    toc_pages = detect_toc_pages((page_idx(i), item_text(i)) for i in items if label_of(i) in ("text", "title"))
    toc_items = explicit_toc_entries([(page_idx(i), "" if label_of(i) in
        {"header", "footer", "page_header", "page_footer", "page_number"} else item_text(i) or "\ufffc") for i in items])
    explicit_pages = {page_idx(items[i]) for i in toc_items}
    if toc_pages:       # page_idx() already returns 1-based page numbers
        print(f"[parser] pdf toc pages={sorted(toc_pages)} (headings there are not sections)", flush=True)
    layout = _analyze_pdf_heading_layout(path, items)
    demoted_embedded = layout.demoted_embedded
    local_image_titles = layout.local_image_titles
    local_image_members = layout.local_image_members
    recovered_top_titles = layout.recovered_top_titles
    non_outer_embedded = demoted_embedded | set(local_image_titles)
    excluded_title_indices = (
        toc_items
        | non_outer_embedded
        | {i for i, item in enumerate(items)
           if page_idx(item) in (toc_pages - explicit_pages) or item.get("_pdf_code_label_owned")}
    )
    demoted_recurring = _unsupported_recurring_titles(
        items,
        excluded_indices=excluded_title_indices,
        recovered_top_titles=recovered_top_titles,
    )
    resolver = HeadingResolver(
        (item_text(item) for i, item in enumerate(items)
         if i not in excluded_title_indices
         and (_effective_heading_level(item) or i in recovered_top_titles)
         and label_of(item) in ("text", "title")),
        demoted_recurring=demoted_recurring,
    )
    local_titles_by_img: dict[tuple[int, int], list[str]] = defaultdict(list)
    for i, img_key in local_image_titles.items():
        local_titles_by_img[img_key].append(item_text(items[i]))
    local_trackers: dict[tuple[int, int], tuple[SectionTracker, HeadingResolver]] = {
        img_key: (SectionTracker(), HeadingResolver(titles, repeat_limit=2))
        for img_key, titles in local_titles_by_img.items()
    }
    local_seen_titles: dict[tuple[int, int], list[tuple[tuple[float, float, float, float] | None, list[str]]]] = defaultdict(list)
    # Page headers and footers are no lines of the document; whatever is not a line of text (a table, a picture,
    # a formula, a listing) is content between two lines; a line on a contents page or inside an embedded image
    # is never recovered as a document chapter heading
    recovered = recoverable_chapter_lines(
        ((i, item_text(item) if label_of(item) in ("text", "title") else None,
          i not in non_outer_embedded and label_of(item) in ("text", "title")
          and (bool(_effective_heading_level(item)) or i in recovered_top_titles))
         for i, item in enumerate(items)
         if label_of(item) not in _PAGE_FURNITURE and not item.get("_pdf_code_label_owned")),
        contents={i for i, item in enumerate(items)
                  if i in toc_items or i in non_outer_embedded
                  or (i in layout.embedded_ocr and not _effective_heading_level(item))
                  or page_idx(item) in toc_pages})
    count = 0
    for item_index, item in enumerate(items):
        typ = label_of(item)
        text = item_text(item)
        local_level_offset = 0
        raw_bb = bbox(item)
        item_box = (float(raw_bb[0]), float(raw_bb[1]), float(raw_bb[2]), float(raw_bb[3])) if raw_bb and len(raw_bb) == 4 else None
        if (
            item_index in local_image_titles
            and typ in ("text", "title")
            and item_index not in toc_items
            and page_idx(item) not in toc_pages - explicit_pages
            and not item.get("_pdf_code_label_owned")
        ):
            img_key = local_image_titles[item_index]
            local_sections, local_resolver = local_trackers[img_key]
            level, inferred = local_resolver.resolve(_effective_heading_level(item), text)
            local_sections.observe(level, text)
            local_path = list(local_sections.path)
            local_seen_titles[img_key].append((item_box, list(local_path)))
            outer_path = list(sections.path)
            if outer_path and local_path and outer_path[-1] == local_path[0]:
                local_path = local_path[1:]
            effective_section_path = outer_path + local_path
            local_level_offset = len(outer_path)
        elif (
            item_index in local_image_members
            and item_index not in toc_items
            and item_index not in recovered
            and item_index not in recovered_top_titles
            and not (item_index not in layout.embedded_ocr and _effective_heading_level(item))
        ):
            level, inferred = None, False
            img_key = local_image_members[item_index]
            image_headings = local_seen_titles[img_key]
            local_path: list[str] = []
            # An OCR heading can end a numbered section without supplying enough evidence
            # to open a new one. Keep verified ancestors and headings in adjacent columns.
            boundary = infer_heading_level(text) if _effective_heading_level(item) else None
            if boundary is not None and image_headings:
                tracker, _ = local_trackers[img_key]
                tracker.end_at(boundary)
                local_path = tracker.path
                image_headings[:] = [
                    (box, path) for box, path in image_headings
                    if box and item_box and box[1] < item_box[3] and item_box[1] < box[3]
                ]
                image_headings.append((item_box, local_path))
            elif image_headings:
                local_path = image_headings[-1][1]
                for tbox, tpath in reversed(image_headings):
                    if _same_column_below(tbox, item_box):
                        local_path = tpath
                        break
            outer_path = list(sections.path)
            if outer_path and local_path and outer_path[-1] == local_path[0]:
                local_path = local_path[1:]
            effective_section_path = outer_path + local_path
        else:
            if item_index in recovered:
                level, inferred = resolver.recover(recovered[item_index])
            else:
                level, inferred = (
                    resolver.resolve(heading_level(item) or (1 if ("title" in typ or item_index in recovered_top_titles) else None), text)
                    if typ in ("text", "title") and item_index not in toc_items
                    and item_index not in non_outer_embedded
                    and page_idx(item) not in toc_pages - explicit_pages
                    and not item.get("_pdf_code_label_owned") else (None, False)
                )
                if item_index in recovered_top_titles and level is not None:
                    inferred = True
            sections.observe(level, text)
            if level is not None and local_seen_titles:
                # Entering a new outer section resets active in-image local trackers so no earlier
                # local heading can leak across a native section boundary.
                local_seen_titles.clear()
                for k in list(local_trackers):
                    local_trackers[k] = (SectionTracker(), HeadingResolver(local_titles_by_img[k], repeat_limit=2))
            effective_section_path = list(sections.path)
        footnote = footnote_of(item)
        if footnote:
            # image_footnote/table_footnote carry the "Note:" / "Data source:" lines;
            # they were extracted (common.footnote_of) but never attached, so
            # units and data sources silently vanished from the index.
            text = f"{text}\nFOOTNOTE: {footnote}".strip()
        image_path = resolve_image_path(item, image_map, allowed_root=cache_dir / "mineru")
        if not text and not image_path:
            continue
        if "table" in typ:
            block_type = "table"
            latex = None
        elif any(token in typ for token in ("equation", "formula", "interline_equation")):
            block_type = "equation"
            latex = str(item.get("latex") or text)
        elif any(token in typ for token in ("image", "figure", "chart", "picture")) or image_path:
            block_type = "chart" if "chart" in typ else "image"
            latex = None
        elif "code" in typ or "algorithm" in typ:
            # A code listing / algorithm box: a block of its own, not merged with the prose around it
            # (pdf_enhanced.is_mergeable_text_block), chunked line by line; its caption (code_caption) is
            # attached by caption_of below
            block_type = "code"
            latex = None
        elif item_index in toc_items:
            block_type = "text"
            latex = None
        elif level is not None:
            # Lines MinerU marked with text_level (or type='title') that passed HeadingResolver, embedded-image,
            # and TOC filters become title blocks; rejected headings remain searchable body text.
            block_type = "title"
            text = clean_heading_text(text) or text      # "**2.1 Installation**" -> "2.1 Installation"
            latex = None
        else:
            block_type = "text"
            latex = None

        page = page_idx(item)
        key = (block_type, hashlib.sha256(text.encode("utf-8")).hexdigest(),
               str(image_path or ""), page, tuple(bbox(item) or ()))
        if key in seen:
            continue
        seen.add(key)
        count += 1
        metadata = {
            "section_path": [] if item_index in toc_items else effective_section_path,
            "source_type": typ,
            "source_item": {k: v for k, v in item.items() if k not in {"text", "content", "html", "latex"}},
        }
        if item_index in toc_items:
            metadata["is_toc"] = True
        if footnote:
            metadata["source_footnote"] = footnote
        if item.get("_pdf_table_note_sources"):
            metadata["table_note_sources"] = item["_pdf_table_note_sources"]
        if item.get("_pdf_table_note_sources") or item.get("_pdf_merged_table_sources"):
            metadata["source_regions"] = ([{"page": page, "bbox": bbox(item)}]
                + list(item.get("_pdf_merged_table_sources") or []) + list(item.get("_pdf_table_note_sources") or []))
        if item.get("_pdf_code_label_sources"):
            metadata["code_context_sources"] = [dict(r) for r in item["_pdf_code_label_sources"]]
        if item.get("_pdf_code_language"):
            metadata["code_language"] = item["_pdf_code_language"]
        if item.get("_pdf_code_recovery"):
            metadata["pdf_code_recovery"] = item["_pdf_code_recovery"]
        if item.get("source_page_end") is not None:
            metadata["page_end"] = item["source_page_end"]
        if block_type in {"image", "chart"} and item.get("content") and not item.get("text"):
            metadata["image_text_source"] = "model"
        if block_type == "title":
            metadata["heading_level"] = level + local_level_offset
            metadata["heading_inferred"] = inferred
        blocks.append(
            ParsedBlock(
                parser="mineru",
                parser_profile=mineru_parser_profile(payload, PDF_PARSER_PROFILE),
                doc_type="pdf",
                block_type=block_type,
                text=text,
                table_markdown=text if block_type == "table" else None,
                block_id=f"mineru-{block_type}-{count:05d}",
                page_idx=page,
                bbox=bbox(item),
                latex=latex,
                caption=caption_of(item) or None,
                visual_ref=image_path,
                metadata=metadata,
            )
        )

    if blocks:
        return blocks

    md = extract_markdown(payload)
    if not md and out_md.exists():
        md = out_md.read_text(encoding="utf-8", errors="ignore")
    return markdown_to_blocks(
        clean_text(md),
        parser="mineru",
        parser_profile=mineru_parser_profile(payload, PDF_PARSER_PROFILE),
        doc_type="pdf",
    )


def save_mineru_images(images: dict[str, str], target_dir: Path) -> dict[str, Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    for name, encoded in images.items():
        if not encoded:
            continue
        payload = encoded.split(",", 1)[1] if "," in encoded else encoded
        try:
            raw = base64.b64decode(payload)
        except Exception:
            continue
        suffix = Path(name).suffix.lower() or suffix_from_data_url(encoded) or ".png"
        safe_name = Path(name).name or f"{hashlib.sha256(raw).hexdigest()[:16]}{suffix}"
        if not Path(safe_name).suffix:
            safe_name = f"{safe_name}{suffix}"
        out = target_dir / safe_name
        if out.exists() and out.read_bytes() != raw:
            out = target_dir / f"{hashlib.sha256(raw).hexdigest()}{suffix}"
        if not out.exists():
            out.write_bytes(raw)
        result[str(name)] = out
        result[Path(name).name] = out
    return result


def suffix_from_data_url(value: str) -> str:
    header = value.split(",", 1)[0].lower()
    if "jpeg" in header or "jpg" in header:
        return ".jpg"
    if "webp" in header:
        return ".webp"
    if "png" in header:
        return ".png"
    return ".png"


def resolve_image_path(
    item: dict[str, Any],
    images: dict[str, Path],
    *,
    allowed_root: Path | None = None,
) -> str | None:
    for key in ("img_path", "image_path", "path"):
        value = item.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        raw = value.strip()
        if raw in images:
            return str(images[raw])
        name = Path(raw).name
        if name in images:
            return str(images[name])
        candidate = Path(raw)
        if candidate.exists():
            # Accept a loose path only when it lives under the parse cache:
            # anything else (a MinerU-side output dir on a shared FS, a CWD
            # artifact) would bake a foreign absolute path into visual_ref
            # and dangle when its owner cleans up.
            if allowed_root is None:
                return str(candidate)
            try:
                candidate.resolve().relative_to(allowed_root.resolve())
            except (ValueError, OSError):
                continue
            return str(candidate)
    return None
