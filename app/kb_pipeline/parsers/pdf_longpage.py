"""Tile unusually tall text PDFs before OCR, retaining original page coordinates.

Long exported web pages can be downscaled until a price or a glyph is only a few
pixels tall. Split only at gaps between text lines and images. Ordinary pages and
pages without a safe text-based split retain their existing parsing path.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..models import ParsedBlock

VERSION = "long-page-v1"
MIN_ASPECT = 3.0
TILE_ASPECT = 1.4


def _cuts(page) -> list[float]:
    width, height = float(page.rect.width), float(page.rect.height)
    if height < 2000 or height / max(width, 1) < MIN_ASPECT or page.rotation:
        return [0.0, height]
    blocks = page.get_text("dict").get("blocks", [])
    lines = [line for b in blocks for line in b.get("lines", [])]
    if sum(len(s.get("text", "")) for line in lines for s in line.get("spans", [])) < 100:
        return [0.0, height]
    occupied = [line["bbox"] for line in lines]
    occupied += [b["bbox"] for b in blocks if b.get("type") == 1]
    cuts = [0.0]
    while height - cuts[-1] > width * TILE_ASPECT * 1.15:
        aim = cuts[-1] + width * TILE_ASPECT
        candidates = [y for y in range(int(aim - width * .22), int(min(height, aim + width * .22)))
                      if all(not (box[1] - 2 < y < box[3] + 2) for box in occupied)]
        if not candidates:
            return [0.0, height]  # Never cut a text line or a figure to force a tile.
        cuts.append(float(min(candidates, key=lambda y: abs(y - aim))))
    return cuts + [height]


def prepare_long_pages(path: Path, cache_dir: Path) -> tuple[Path, list[dict]]:
    """Return the original file unchanged unless at least one safe split exists."""
    try:
        import pymupdf
        pdf = pymupdf.open(str(path))
    except (ImportError, RuntimeError, OSError):
        return path, []
    with pdf:
        cuts = [_cuts(page) for page in pdf]
        if not any(len(c) > 2 for c in cuts):
            return path, []
        root = cache_dir / VERSION
        target, manifest = root / "segments.pdf", root / "segments.json"
        source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if target.exists() and manifest.exists():
            saved = json.loads(manifest.read_text())
            if saved.get("source_sha256") == source_sha:
                return target, saved["segments"]
        root.mkdir(parents=True, exist_ok=True)
        segments = []
        with pymupdf.open() as output:
            for i, page in enumerate(pdf):
                width, height = float(page.rect.width), float(page.rect.height)
                if len(cuts[i]) == 2:
                    output.insert_pdf(pdf, from_page=i, to_page=i)
                    segments.append(dict(page=i + 1, y0=0.0, y1=height, width=width, height=height, tiled=False))
                    continue
                for y0, y1 in zip(cuts[i], cuts[i][1:]):
                    clip = pymupdf.Rect(0, y0, width, y1)
                    tile = output.new_page(width=width, height=y1 - y0)
                    tile.show_pdf_page(tile.rect, pdf, i, clip=clip)
                    segments.append(dict(page=i + 1, y0=y0, y1=y1, width=width, height=height, tiled=True))
            tmp = root / "segments.tmp.pdf"
            output.save(str(tmp))
            tmp.replace(target)
        manifest.write_text(json.dumps(dict(version=VERSION, source_sha256=source_sha, segments=segments), indent=2))
        return target, segments


def restore_page_coordinates(blocks: list[ParsedBlock], segments: list[dict]) -> list[ParsedBlock]:
    """Convert MinerU's normalized tile bbox to the original PDF's 0-1000 bbox."""
    for block in blocks:
        if block.page_idx is None or not 1 <= block.page_idx <= len(segments):
            raise ValueError("segmented PDF block has no valid source page")
        tile_no = block.page_idx
        seg = segments[tile_no - 1]
        block.page_idx = seg["page"]
        if seg["tiled"]:
            tile_height = seg["y1"] - seg["y0"]
            if block.bbox and len(block.bbox) >= 4:
                x0, y0, x1, y1 = map(float, block.bbox[:4])
                if any(v > 1000 for v in (x0, y0, x1, y1)):
                    x0, x1 = x0 / seg["width"] * 1000, x1 / seg["width"] * 1000
                    y0, y1 = y0 / tile_height * 1000, y1 / tile_height * 1000
                block.bbox = [x0, (seg["y0"] + y0 * tile_height / 1000) / seg["height"] * 1000,
                              x1, (seg["y0"] + y1 * tile_height / 1000) / seg["height"] * 1000]
            block.parser_profile += "+" + VERSION
        block.metadata["source_tile"] = {"tile": tile_no, **seg}
        # source_item describes the parser input. Keep its tile coordinates nested,
        # while the public page_idx/bbox always locate the original document.
    return blocks
