from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from ..models import ParsedBlock
from .common import bbox, caption_of, clean_text, footnote_of, item_text, label_of, markdown_to_blocks, maybe_int
from .mineru_pdf import MinerUServiceError, resolve_image_path, save_mineru_images
from .service_clients import call_mineru_sync, extract_content_list, extract_images, extract_markdown


def mineru_pptx_blocks(
    *,
    mineru_url: str,
    path: Path,
    cache_dir: Path,
    timeout: int = 43200,
) -> list[ParsedBlock]:
    out_json = cache_dir / "mineru" / "pptx-result.json"
    out_md = cache_dir / "mineru" / "pptx-result.md"
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

    content = extract_content_list(payload)
    if not content:
        md = extract_markdown(payload)
        if not md and out_md.exists():
            md = out_md.read_text(encoding="utf-8", errors="ignore")
        return markdown_to_blocks(
            md,
            parser="mineru",
            parser_profile="mineru-3.4.4-pptx-v1",
            doc_type="pptx",
        )

    image_map = save_mineru_images(extract_images(payload), cache_dir / "mineru" / "images")
    slide_by_index = normalize_pptx_slide_indexes(content)
    blocks: list[ParsedBlock] = []
    count = 0
    for idx, item in enumerate(content):
        raw_type = label_of(item)
        if raw_type in {"page_footnote", "footer", "header"}:
            continue
        image_path = resolve_image_path(item, image_map, allowed_root=cache_dir / "mineru")
        text = item_text(item)
        footnote = footnote_of(item)
        if footnote:
            text = f"{text}\nFOOTNOTE: {footnote}".strip()
        if not text and not image_path:
            continue

        if "table" in raw_type:
            block_type = "table"
        elif any(token in raw_type for token in ("image", "figure", "chart", "picture")) or image_path:
            block_type = "chart" if "chart" in raw_type else "image"
        else:
            block_type = "text"

        # No de-duplication by content: short labels repeated on one slide (four cards all saying "Supported")
        # are part of the layout, see mineru_docx
        slide_idx = slide_by_index.get(idx)
        count += 1
        blocks.append(
            ParsedBlock(
                parser="mineru",
                parser_profile="mineru-3.4.4-pptx-v1",
                doc_type="pptx",
                block_type=block_type,
                text=text,
                table_markdown=text if block_type == "table" else None,
                block_id=f"mineru-pptx-{block_type}-{count:05d}",
                page_idx=slide_idx,
                slide_idx=slide_idx,
                bbox=bbox(item),
                caption=caption_of(item) or None,
                visual_ref=image_path,
                metadata={
                    "source_type": raw_type,
                    "raw_page": raw_page(item),
                    "source_item": {k: v for k, v in item.items() if k not in {"text", "content", "html"}},
                },
            )
        )
    return blocks


def aggregate_mineru_slide_blocks(blocks: list[ParsedBlock], path: Path) -> list[ParsedBlock]:
    text_by_slide: dict[int, list[ParsedBlock]] = defaultdict(list)
    passthrough: list[ParsedBlock] = []
    orphan_text: list[ParsedBlock] = []

    for block in blocks:
        if block.block_type == "table":
            passthrough.append(block)
            continue
        if block.block_type == "text" and (block.text or "").strip():
            if block.slide_idx is not None:
                text_by_slide[int(block.slide_idx)].append(block)
            else:
                orphan_text.append(block)
            continue
        # image/chart blocks used to fall through every branch and vanish, which
        # is why a whole-slide render existed to make up for them. They now pass
        # through and get VLM-enriched exactly like PDF and DOCX figures.
        passthrough.append(block)

    from ..headings import clean_heading_text

    merged: list[ParsedBlock] = []
    slide_titles: dict[int, str] = {}
    for slide in sorted(text_by_slide):
        slide_blocks = text_by_slide[slide]
        text = clean_text("\n\n".join(block.text or "" for block in slide_blocks))
        if not text:
            continue
        # The slide's first line (usually the slide title) serves as the section: both the retrieval
        # prefix and the section path used in graph recall depend on it
        first_line = next((l.strip() for l in text.splitlines() if l.strip()), "")
        heading = clean_heading_text(first_line) if len(first_line) <= 80 else ""
        if heading:
            slide_titles[slide] = heading
        merged.append(
            ParsedBlock(
                parser="mineru",
                parser_profile="mineru-3.4.4-pptx-v1+slide-merge-v1",
                doc_type="pptx",
                block_type="slide",
                text=text,
                page_idx=slide,
                slide_idx=slide,
                title=f"{path.name} slide {slide}",
                block_id=f"pptx-mineru-slide-text-{slide:04d}",
                metadata={
                    "merged": True,
                    "merged_block_count": len(slide_blocks),
                    "source_block_ids": [block.block_id for block in slide_blocks],
                    "section_path": [heading] if heading else [],
                },
            )
        )
    # Image / table blocks on the same slide hang under the same section
    for block in passthrough:
        if block.slide_idx is not None and int(block.slide_idx) in slide_titles and not block.metadata.get("section_path"):
            block.metadata["section_path"] = [slide_titles[int(block.slide_idx)]]

    if not merged:
        passthrough.extend(orphan_text)
    print(
        f"[parser] pptx mineru slide-merge slides={len(merged)} passthrough={len(passthrough)} output={len(merged) + len(passthrough)}",
        flush=True,
    )
    # Interleave by slide: each slide's images / tables directly follow that slide's text block
    # (previously it was all text blocks + all images, and the chunking fallback merged the last
    # slide's title into the first slide's image)
    order = {id(block): i for i, block in enumerate(blocks)}

    def key(block: ParsedBlock) -> tuple[int, int, int]:
        slide = int(block.slide_idx) if block.slide_idx is not None else 10**9
        return (slide, 0 if block.block_type == "slide" else 1, order.get(id(block), 0))

    return sorted(merged + passthrough, key=key)


def normalize_pptx_slide_indexes(content: list[dict[str, Any]]) -> dict[int, int | None]:
    raw_pages = [raw_page(item) for item in content]
    numeric_pages = [page for page in raw_pages if page is not None]
    if not numeric_pages:
        return {idx: None for idx, _ in enumerate(content)}

    first_numeric_idx = next((idx for idx, page in enumerate(raw_pages) if page is not None), None)
    has_leading_none = first_numeric_idx is not None and any(
        meaningful_item(item) and raw_pages[idx] is None for idx, item in enumerate(content[:first_numeric_idx])
    )
    min_page = min(numeric_pages)
    if min_page == 0:
        offset = 1
    elif min_page == 1 and has_leading_none:
        offset = 1
    else:
        offset = 0

    result: dict[int, int | None] = {}
    current_slide = 1 if has_leading_none else None
    for idx, page in enumerate(raw_pages):
        if page is None:
            result[idx] = current_slide
            continue
        current_slide = page + offset
        result[idx] = current_slide
    return result


def raw_page(item: dict[str, Any]) -> int | None:
    for key in ("page_idx", "page", "page_id", "page_no"):
        value = maybe_int(item.get(key))
        if value is not None:
            return value
    return None


def meaningful_item(item: dict[str, Any]) -> bool:
    return bool(item_text(item) or item.get("img_path") or item.get("image_path") or item.get("path"))
