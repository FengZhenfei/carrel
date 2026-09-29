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
    return place_chart_blocks(blocks, extract_docx_chart_blocks(path))


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
                        caption="Chart data" + (f"; axes: {' / '.join(axes)}" if axes else ""),
                        block_id=f"docx-chart-{idx:04d}",
                        metadata={"ooxml_path": name, "anchor_text": anchors.get(name, "")},
                    )
                )
    except zipfile.BadZipFile:
        return []
    return blocks


_CHART_NS = "{http://schemas.openxmlformats.org/drawingml/2006/chart}"
_WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_REL_ID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
_AXIS_TAGS = frozenset(f"{_CHART_NS}{kind}" for kind in ("catAx", "dateAx", "valAx", "serAx"))
_MARKUP_RE = re.compile(r"[\s*_`\\#]+")          # whitespace and markdown marks are ignored when comparing text


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
        name = _chart_text(ser.find(f"{_CHART_NS}tx")) or f"Series {number}"
        cats = _chart_points(ser.find(f"{_CHART_NS}cat")) or _chart_points(ser.find(f"{_CHART_NS}xVal"))
        vals = _chart_points(ser.find(f"{_CHART_NS}val")) or _chart_points(ser.find(f"{_CHART_NS}yVal"))
        sizes = _chart_points(ser.find(f"{_CHART_NS}bubbleSize"))
        sized = sized or bool(sizes)
        for idx in sorted(set(cats) | set(vals)):
            rows.append([name, cats.get(idx) or str(idx + 1), vals.get(idx, ""), sizes.get(idx, "")])
    if not rows:
        return title, axes, ""
    width = 4 if sized else 3
    lines = [["Series", "Category", "Value", "Size"][:width], ["---"] * width] + [row[:width] for row in rows]
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
