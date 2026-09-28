from __future__ import annotations

import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from ..models import ParsedBlock
from .common import clean_text
from .mineru_docx import mineru_docx_blocks
from .pdf_enhanced import merge_mineru_text_blocks, merge_split_tables
from ..vision.vlm import compose_prompt
from .table_repair import repair_ambiguous_tables
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
    table_blocks = [b for b in blocks if b.block_type == "table" and b.metadata.get("table_flags")]
    if table_blocks:
        stats = repair_ambiguous_tables(table_blocks, base_url=vlm_base_url, api_key=vlm_api_key, model_id=vlm_model_id,
                                        cache_dir=cache_dir, vlm_options=vlm_options, caption_cache_root=caption_cache_root)
        print(f"[parser] docx table-repair {stats} file={path.name}", flush=True)
    chart_blocks = extract_docx_chart_blocks(path)
    return blocks + chart_blocks


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
    blocks: list[ParsedBlock] = []
    try:
        with zipfile.ZipFile(path) as zf:
            names = sorted(
                name
                for name in zf.namelist()
                if name.startswith("word/charts/") and name.endswith(".xml")
            )
            for idx, name in enumerate(names, start=1):
                try:
                    text = chart_xml_text(zf.read(name))
                except Exception:
                    text = ""
                if not text:
                    continue
                blocks.append(
                    ParsedBlock(
                        parser="docx-ooxml-chart",
                        parser_profile="docx-ooxml-chart-v1",
                        doc_type="docx",
                        block_type="chart",
                        text=f"DOCX CHART DATA:\n{text}",
                        title=f"{path.name} chart {idx}",
                        block_id=f"docx-chart-{idx:04d}",
                        metadata={"ooxml_path": name},
                    )
                )
    except zipfile.BadZipFile:
        return []
    return blocks


def chart_xml_text(raw: bytes) -> str:
    root = ET.fromstring(raw)
    values: list[str] = []
    for elem in root.iter():
        tag = elem.tag.rsplit("}", 1)[-1]
        if tag in {"v", "ptCount", "f", "tx", "t"} and elem.text:
            value = clean_text(elem.text)
            if value:
                values.append(value)
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value[:200]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return clean_text("\n".join(deduped[:300]))
