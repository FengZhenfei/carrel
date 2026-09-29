from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import Any

from ..models import ParsedBlock
from ..headings import detect_toc_pages
from .table_check import table_ambiguity_flags
from .common import HeadingResolver, SectionTracker, bbox, caption_of, clean_heading_text, clean_text, footnote_of, heading_level, item_text, label_of, markdown_to_blocks, page_idx, walk_dicts
from .service_clients import call_mineru_sync, extract_content_list, extract_images, extract_markdown

# v2 (2026-09-06): merged-cell table flags + screenshot verification repair are now in the blocks; the
# version bump makes scan re-parse via parser_changed
PDF_PARSER_PROFILE = "mineru-3.4.4-pipeline-v2"


class MinerUServiceError(RuntimeError):
    """Carries the HTTP status code: retrying a 4xx (corrupt document, rejected by MinerU) is
    pointless, while connection / read timeouts should be retried. The two used to be squashed into
    the same string exception."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code

    @property
    def deterministic(self) -> bool:
        return self.status_code is not None and 400 <= self.status_code < 500 and self.status_code != 429

    pass


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
    items = content if content else list(walk_dicts(payload))

    blocks: list[ParsedBlock] = []
    seen: set[tuple[str, str, str, int | None]] = set()
    sections = SectionTracker()
    # Heading levels: MinerU's text_level is combined with layout inference (CJK chapter numbering /
    # 1.2.3 / Chapter N), then corrected at document level -- recurring labels (KEYWORD / Note:) and
    # list items are not headings, and unnumbered headings hang under the nearest numbered one.
    # Ordinary body lines MinerU did not mark are recognised by inference (see headings.HeadingResolver)
    toc_pages = detect_toc_pages((page_idx(i), item_text(i)) for i in items if label_of(i) in ("text", "title"))
    if toc_pages:
        print(f"[parser] pdf toc pages={sorted(p + 1 for p in toc_pages)} (headings there are not sections)", flush=True)
    resolver = HeadingResolver(item_text(i) for i in items if heading_level(i) and label_of(i) in ("text", "title"))
    count = 0
    for item in items:
        typ = label_of(item)
        text = item_text(item)
        level, inferred = (resolver.resolve(heading_level(item), text)
                           if typ in ("text", "title") and page_idx(item) not in toc_pages else (None, False))
        sections.observe(level, text)
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
        elif "title" in typ or level is not None:
            # Lines MinerU marked with text_level also become title blocks (previously only those
            # recognised by layout inference did): the chunker breaks before headings using the
            # heading_lines recorded from title blocks, which is what makes section cut points line up
            block_type = "title"
            text = clean_heading_text(text) or text      # "**2.1 Installation**" -> "2.1 Installation"
            latex = None
        else:
            block_type = "text"
            latex = None

        page = page_idx(item)
        key = (block_type, text[:160], Path(image_path).name if image_path else "", page)
        if key in seen:
            continue
        seen.add(key)
        count += 1
        metadata = {
            "section_path": list(sections.path),
            "source_type": typ,
            "source_item": {k: v for k, v in item.items() if k not in {"text", "content", "html", "latex"}},
        }
        if block_type == "title":
            metadata["heading_level"] = level
            metadata["heading_inferred"] = inferred
        if block_type == "table":
            # Rows whose stacked lines MinerU concatenated into one string (757557 / mAmAmA) are flagged
            # first; the enhancement layer then verifies them against the screenshot (F02)
            flags = table_ambiguity_flags(str(item.get("table_body") or item.get("html") or ""))
            if flags:
                metadata["table_flags"] = flags
        blocks.append(
            ParsedBlock(
                parser="mineru",
                parser_profile=PDF_PARSER_PROFILE,
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
        parser_profile=PDF_PARSER_PROFILE,
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
