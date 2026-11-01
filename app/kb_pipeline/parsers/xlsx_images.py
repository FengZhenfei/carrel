"""Read embedded XLSX pictures and their sheet/cell ownership, without executing formulas."""
from __future__ import annotations

import hashlib
import posixpath
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from ..models import ParsedBlock
from ..vision.images import IMAGE_SUFFIXES
from .native_table import _xlsx_parse_shared_strings, col_name

S = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
XDR = "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing"
NS = {"s": S, "a": A, "xdr": XDR}
_DISPIMG = re.compile(r'^\s*=?\s*(?:_xlfn\.)?DISPIMG\(\s*"([^"\r\n]+)"', re.I)


def _xml(z: zipfile.ZipFile, name: str):
    try:
        return ET.fromstring(z.read(name))
    except KeyError:
        return None


def _rels(z: zipfile.ZipFile, part: str) -> dict[str, str]:
    parent, name = posixpath.split(part)
    root = _xml(z, posixpath.join(parent, "_rels", name + ".rels"))
    out = {}
    if root is None:
        return out
    for rel in root:
        target = rel.get("Target", "")
        if rel.get("TargetMode", "").lower() == "external" or ":" in target or "\\" in target:
            continue
        target = posixpath.normpath(target.lstrip("/") if target.startswith("/") else posixpath.join(parent, target))
        if target.startswith("../") or target not in z.namelist():
            continue
        out[rel.get("Id", "")] = target
    return out


def _cell(marker) -> tuple[str, int]:
    row = int(marker.findtext("xdr:row", "0", NS)) + 1
    col = int(marker.findtext("xdr:col", "0", NS))
    return f"{col_name(col)}{row}", row


def extract_xlsx_images(path: Path, cache_dir: Path) -> list[ParsedBlock]:
    """Only referenced, internal pictures on visible sheets are read.

    Supports DrawingML anchors and WPS DISPIMG cells. External links and
    formulas are never fetched or evaluated; unreferenced media are not data.
    """
    blocks = []
    with zipfile.ZipFile(path) as z:
        workbook = _xml(z, "xl/workbook.xml")
        if workbook is None:
            return []
        book_rels = _rels(z, "xl/workbook.xml")
        cell_images = _xml(z, "xl/cellimages.xml")
        cell_rels = _rels(z, "xl/cellimages.xml")
        named = {}
        if cell_images is not None:
            for pic in cell_images.findall(".//xdr:pic", NS):
                prop = pic.find("xdr:nvPicPr/xdr:cNvPr", NS)
                blip = pic.find(".//a:blip", NS)
                if prop is not None and blip is not None:
                    target = cell_rels.get(blip.get(f"{{{R}}}embed", ""))
                    if target and prop.get("hidden") not in {"1", "true"}:
                        named[prop.get("name", "")] = target
        shared = _xlsx_parse_shared_strings(z)
        for sheet in workbook.findall("s:sheets/s:sheet", NS):
            if sheet.get("state", "visible") != "visible":
                continue
            sheet_part = book_rels.get(sheet.get(f"{{{R}}}id", ""))
            root = _xml(z, sheet_part) if sheet_part else None
            if root is None:
                continue
            sheet_name = sheet.get("name", "Sheet")
            refs = []
            hidden_rows = {int(row.get("r", "0")) for row in root.findall("s:sheetData/s:row", NS)
                           if row.get("hidden") in {"1", "true"}}
            for cell in root.findall("s:sheetData/s:row/s:c", NS):
                address = cell.get("r", "")
                m = re.fullmatch(r"([A-Z]+)([1-9][0-9]*)", address)
                if not m or int(m[2]) in hidden_rows:
                    continue
                value = cell.findtext("s:f", "", NS) or cell.findtext("s:v", "", NS)
                if cell.get("t") == "s" and value.isdigit() and int(value) < len(shared):
                    value = shared[int(value)]
                elif cell.get("t") == "inlineStr":
                    value = "".join(t.text or "" for t in cell.findall(".//s:t", NS))
                match = _DISPIMG.match(value)
                if match and match[1] in named:
                    refs.append((named[match[1]], address, address, int(m[2]), int(m[2]), f"cell:{address}"))
            sheet_rels = _rels(z, sheet_part)
            for drawing in root.findall("s:drawing", NS):
                part = sheet_rels.get(drawing.get(f"{{{R}}}id", ""))
                tree = _xml(z, part) if part else None
                if tree is None:
                    continue
                rels = _rels(z, part)
                for anchor_index, anchor in enumerate(tree):
                    start = anchor.find("xdr:from", NS)
                    end = anchor.find("xdr:to", NS)
                    first, row_start = _cell(start) if start is not None else ("", 0)
                    last, row_end = _cell(end) if end is not None else (first, row_start)
                    for picture_index, pic in enumerate(anchor.findall(".//xdr:pic", NS)):
                        prop = pic.find("xdr:nvPicPr/xdr:cNvPr", NS)
                        if prop is not None and prop.get("hidden") in {"1", "true"}:
                            continue
                        blip = pic.find(".//a:blip", NS)
                        target = rels.get(blip.get(f"{{{R}}}embed", "")) if blip is not None else None
                        if target:
                            refs.append((target, first, last, row_start, row_end,
                                         f"{part}:anchor:{anchor_index}:picture:{picture_index}"))
            for target, first, last, start, end, occurrence in refs:
                data = z.read(target)
                digest = hashlib.sha256(data).hexdigest()
                location = first + (":" + last if last and last != first else "")
                identity = hashlib.sha256(f"{sheet_part}|{occurrence}|{digest}".encode()).hexdigest()[:24]
                suffix = Path(target).suffix.lower()
                supported = suffix in IMAGE_SUFFIXES
                asset = cache_dir / "xlsx-images" / (digest + suffix)
                asset.parent.mkdir(parents=True, exist_ok=True)
                asset.write_bytes(data)
                blocks.append(ParsedBlock(parser="xlsx-ooxml-image", parser_profile="xlsx-image-v1", doc_type="xlsx",
                    block_type="image", text="" if supported else f"Embedded image format: {suffix}",
                    block_id="xlsx-image-" + identity, sheet_name=sheet_name, row_start=start or None, row_end=end or None,
                    title=f"{sheet_name}!{location}" if location else sheet_name,
                    visual_ref=str(asset) if supported else None,
                    metadata={"ooxml_path": target, "source_locator": {"kind": "cells" if location else "drawing",
                              "sheet": sheet_name, "range": location, "occurrence": occurrence},
                              **({} if supported else {"degraded": "unsupported_embedded_image", "source_asset": str(asset)})}))
    return blocks
