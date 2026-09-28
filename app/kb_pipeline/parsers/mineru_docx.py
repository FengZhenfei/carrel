from __future__ import annotations

from pathlib import Path

from ..models import ParsedBlock
from .table_check import table_ambiguity_flags
from .common import HeadingResolver, SectionTracker, bbox, caption_of, clean_heading_text, footnote_of, heading_level, item_text, label_of, markdown_to_blocks, page_idx
from .mineru_pdf import MinerUServiceError, resolve_image_path, save_mineru_images
from .service_clients import call_mineru_sync, extract_content_list, extract_images, extract_markdown

def _table_meta(item: dict, block_type: str) -> dict:
    """Table blocks: rows whose stacked lines MinerU concatenated into one string are flagged first
    (F02); the enhancement layer then verifies them against the screenshot."""
    if block_type != "table":
        return {}
    flags = table_ambiguity_flags(str(item.get("table_body") or item.get("html") or ""))
    return {"table_flags": flags} if flags else {}



def mineru_docx_blocks(
    *,
    mineru_url: str,
    path: Path,
    cache_dir: Path,
    timeout: int = 43200,
) -> list[ParsedBlock]:
    out_json = cache_dir / "mineru" / "docx-result.json"
    out_md = cache_dir / "mineru" / "docx-result.md"
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
            parser_profile="mineru-3.4.4-docx-v1",
            doc_type="docx",
        )

    image_map = save_mineru_images(extract_images(payload), cache_dir / "mineru" / "images")
    blocks: list[ParsedBlock] = []
    seen: set[tuple[str, str, str, int | None]] = set()
    sections = SectionTracker()
    # Document-level heading correction (recurring labels and list items are not headings; unnumbered
    # headings hang under numbered ones), see headings.HeadingResolver
    resolver = HeadingResolver(item_text(i) for i in content if heading_level(i) and label_of(i) in ("text", "title"))
    count = 0
    for item in content:
        raw_type = label_of(item)
        if raw_type in {"page_footnote", "footer", "header"}:
            continue
        image_path = resolve_image_path(item, image_map, allowed_root=cache_dir / "mineru")
        text = item_text(item)
        level, inferred = resolver.resolve(heading_level(item), text) if raw_type in ("text", "title") else (None, False)
        sections.observe(level, text)
        footnote = footnote_of(item)
        if footnote:
            text = f"{text}\nFOOTNOTE: {footnote}".strip()
        if not text and not image_path:
            continue

        latex = None
        if "table" in raw_type:
            block_type = "table"
        elif any(token in raw_type for token in ("equation", "formula", "interline_equation")):
            block_type = "equation"
            latex = str(item.get("latex") or text)
        elif any(token in raw_type for token in ("image", "figure", "chart", "picture")) or image_path:
            block_type = "chart" if "chart" in raw_type else "image"
        elif "title" in raw_type or level is not None:
            block_type = "title"
            text = clean_heading_text(text) or text      # "**2.1 Installation**" -> "2.1 Installation"
        else:
            block_type = "text"

        page = page_idx(item)
        key = (block_type, text[:160], Path(image_path).name if image_path else "", page)
        if key in seen:
            continue
        seen.add(key)
        count += 1
        blocks.append(
            ParsedBlock(
                parser="mineru",
                parser_profile="mineru-3.4.4-docx-v1",
                doc_type="docx",
                block_type=block_type,
                text=text,
                table_markdown=text if block_type == "table" else None,
                latex=latex,
                block_id=f"mineru-docx-{block_type}-{count:05d}",
                page_idx=page,
                bbox=bbox(item),
                caption=caption_of(item) or None,
                visual_ref=image_path,
                metadata={
                    "section_path": list(sections.path),
                    "source_type": raw_type,
                    "source_item": {k: v for k, v in item.items() if k not in {"text", "content", "html", "latex"}},
                    **({"heading_level": level, "heading_inferred": inferred} if block_type == "title" else {}),
                    **_table_meta(item, block_type),
                },
            )
        )
    return blocks
