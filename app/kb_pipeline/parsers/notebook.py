"""Read saved notebook cells and outputs without starting a kernel."""
from __future__ import annotations

import base64
import binascii
import json
import re
from pathlib import Path

from ..models import ParsedBlock
from .common import markdown_to_blocks
from .errors import NonRetryableParseError
from .html_dom import parse_html_dom
from .source_guard import require_source_content

PROFILE = "notebook-cells-v1"
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        return "".join(value)
    if value is None:
        return ""
    raise NonRetryableParseError("Invalid notebook text: expected a string or list of strings")


def parse_notebook(path: Path, cache_dir: Path) -> list[ParsedBlock]:
    require_source_content(path)
    try:
        notebook = json.loads(path.read_text(encoding="utf-8-sig"))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise NonRetryableParseError(f"Invalid notebook JSON: {exc}") from exc
    if not isinstance(notebook, dict):
        raise NonRetryableParseError("Invalid notebook: expected an object")
    if notebook.get("nbformat") not in (3, 4):
        raise NonRetryableParseError("Notebook format version must be 3 or 4")
    cells = notebook.get("cells")
    if notebook.get("nbformat") == 3:
        sheets = notebook.get("worksheets")
        if not isinstance(sheets, list) or any(not isinstance(s, dict) or not isinstance(s.get("cells"), list) for s in sheets):
            raise NonRetryableParseError("Invalid notebook worksheets")
        cells = [c for sheet in sheets for c in sheet["cells"]]
    if not isinstance(cells, list):
        raise NonRetryableParseError("Invalid notebook: cells are missing")
    assets = cache_dir / "notebook"
    assets.mkdir(parents=True, exist_ok=True)
    blocks: list[ParsedBlock] = []
    metadata = notebook.get("metadata") or {}
    if not isinstance(metadata, dict) or any(metadata.get(k) is not None and not isinstance(metadata[k], dict) for k in ("language_info", "kernelspec")):
        raise NonRetryableParseError("Invalid notebook metadata")
    language = (metadata.get("language_info") or {}).get("name") or (metadata.get("kernelspec") or {}).get("language") or ""

    def make(text: str, ci: int, oi: int | None = None, *, kind="text", mime=None) -> ParsedBlock:
        label = f"Cell {ci}" + (f" output {oi}" if oi is not None else "")
        block = ParsedBlock(parser="notebook", parser_profile=PROFILE, doc_type="ipynb", block_type=kind,
                            text=_ANSI.sub("", text), title=label,
                            block_id=f"cell-{ci:05d}" + (f"-out-{oi:04d}" if oi is not None else ""),
                            metadata={"section_path": [label], "source_locator": {"kind": "notebook", "cell": ci}, "language": language})
        if oi is not None:
            block.metadata["source_locator"].update(output=oi, mime=mime)
        return block

    for ci, cell in enumerate(cells, 1):
        if not isinstance(cell, dict):
            raise NonRetryableParseError(f"Invalid notebook cell {ci}")
        kind = cell.get("cell_type")
        source = _text(cell.get("source", cell.get("input")))
        if kind == "markdown":
            sections = markdown_to_blocks(source, parser="notebook", parser_profile=PROFILE, doc_type="ipynb")
            for section in sections:
                section.block_id = f"cell-{ci:05d}-{section.block_id}"
                section.metadata["source_locator"] = {"kind": "notebook", "cell": ci}
                section.metadata["section_path"] = [f"Cell {ci}", *section.metadata.get("section_path", [])]
                blocks.append(section)
        elif source.strip():
            block = make(source, ci, kind="code" if kind == "code" else "text")
            block.metadata["execution_count"] = cell.get("execution_count", cell.get("prompt_number"))
            blocks.append(block)
        outputs = cell.get("outputs", [])
        if not isinstance(outputs, list):
            raise NonRetryableParseError(f"Invalid outputs in notebook cell {ci}")
        for oi, output in enumerate(outputs, 1):
            if not isinstance(output, dict):
                raise NonRetryableParseError(f"Invalid notebook output at cell {ci}, output {oi}")
            out_type = output.get("output_type")
            if out_type == "stream":
                text = _text(output.get("text"))
                if text.strip():
                    blocks.append(make(text, ci, oi, mime="text/plain"))
                continue
            if out_type in {"error", "pyerr"}:
                trace = output.get("traceback") or []
                text = "\n".join(str(line) for line in trace) if isinstance(trace, list) else str(trace)
                text = f"Saved error: {output.get('ename', '')}: {output.get('evalue', '')}\n{text}"
                blocks.append(make(text, ci, oi, mime="text/plain"))
                continue
            data = output.get("data", {})
            if notebook.get("nbformat") == 3:
                data = {mime: output[key] for key, mime in (("text", "text/plain"), ("html", "text/html"), ("png", "image/png"), ("jpeg", "image/jpeg")) if key in output}
            if not isinstance(data, dict):
                raise NonRetryableParseError(f"Invalid MIME bundle at cell {ci}, output {oi}")
            supported = {"text/html", "text/plain", "text/markdown", "text/latex", "application/json", "image/png", "image/jpeg"}
            if data and not (supported & data.keys()):
                raise NonRetryableParseError(f"Unsupported saved output MIME at cell {ci}, output {oi}: {', '.join(data)}")
            # MIME alternatives describe the same output; choose one text representation.
            text, text_mime = "", "text/plain"
            if "text/html" in data:
                html_path = assets / f"cell-{ci}-output-{oi}.html"
                html_path.write_text(_text(data["text/html"]), encoding="utf-8")
                sections = [section for section in parse_html_dom(html_path) if section.text.strip()]
                for i, section in enumerate(sections, 1):
                    section.parser, section.parser_profile, section.doc_type = "notebook", PROFILE, "ipynb"
                    section.block_id = f"cell-{ci:05d}-out-{oi:04d}-html-{i}"
                    section.title = f"Cell {ci} output {oi}" + (f": {section.title}" if section.title else "")
                    section.page_idx = None
                    section.metadata.update(source_locator={"kind": "notebook", "cell": ci, "output": oi, "mime": "text/html"}, section_path=[f"Cell {ci}", f"Output {oi}"])
                    blocks.append(section)
                if not sections:
                    text = _text(data.get("text/plain"))
            elif "text/plain" in data:
                text = _text(data["text/plain"])
            elif "text/markdown" in data:
                text = _text(data["text/markdown"])
                text_mime = "text/markdown"
            elif "application/json" in data:
                text = json.dumps(data["application/json"], ensure_ascii=False, indent=2)
                text_mime = "application/json"
            elif "text/latex" in data:
                text = _text(data["text/latex"])
                text_mime = "text/latex"
            if text.strip():
                blocks.append(make(text, ci, oi, mime=text_mime))
            # Keep saved raster plots as visual evidence, beside their textual alternative.
            for mime, suffix in (("image/png", ".png"), ("image/jpeg", ".jpg")):
                if mime not in data:
                    continue
                try:
                    image = base64.b64decode(re.sub(r"\s+", "", _text(data[mime])), validate=True)
                    asset = assets / f"cell-{ci}-output-{oi}{suffix}"
                    asset.write_bytes(image)
                    from PIL import Image
                    with Image.open(asset) as im:
                        im.verify()
                except (ValueError, binascii.Error, OSError) as exc:
                    raise NonRetryableParseError(f"Invalid saved image at cell {ci}, output {oi}: {exc}") from exc
                block = make("", ci, oi, kind="image", mime=mime)
                block.block_id += "-image"
                block.visual_ref = str(asset)
                blocks.append(block)
                break
    return blocks
