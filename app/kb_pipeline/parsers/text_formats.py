"""Static, source-preserving readers for sectioned text and record formats."""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..models import ParsedBlock
from .common import SectionTracker, read_text_smart, parser_profile_for_path
from .errors import NonRetryableParseError


def _block(path: Path, text: str, index: int, start: int, end: int, title: str = "", **metadata) -> ParsedBlock:
    return ParsedBlock(
        parser="static-text", parser_profile=parser_profile_for_path(path), doc_type=path.suffix[1:].lower(),
        block_type="text", text=text, title=title or None, block_id=f"section-{index:05d}",
        metadata={"section_path": [title] if title else [],
                  "source_locator": {"kind": "lines", "start": start, "end": end}, **metadata},
    )


def parse_jsonl(path: Path) -> list[ParsedBlock]:
    blocks = []
    # Read one record at a time; the file size limit is enforced by the pipeline.
    try:
        with path.open(encoding="utf-8-sig") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except (ValueError, RecursionError) as exc:
                    raise NonRetryableParseError(f"Invalid JSONL at line {number}: {exc}") from exc
                identifier = next((value[k] for k in ("id", "_id", "name", "title") if k in value), None) if isinstance(value, dict) else None
                title = f"Record {number}" + (f": {str(identifier)[:160]}" if identifier is not None else "")
                block = _block(path, json.dumps(value, ensure_ascii=False, indent=2), len(blocks) + 1, number, number, title)
                block.metadata["source_locator"] = {"kind": "jsonl", "line": number}
                blocks.append(block)
    except UnicodeError as exc:
        raise NonRetryableParseError(f"JSONL must contain UTF-8 text: {exc}") from exc
    return blocks


def parse_xsd(path: Path) -> list[ParsedBlock]:
    from lxml import etree

    try:
        root = etree.fromstring(path.read_bytes(), parser=etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False))
    except etree.XMLSyntaxError as exc:
        raise NonRetryableParseError(f"Invalid XML schema: {exc}") from exc
    if etree.QName(root).localname != "schema":
        raise NonRetryableParseError("XSD root must be schema")
    blocks = []
    namespace = root.get("targetNamespace", "")
    # Keep the root namespace and defaults together with each definition. Imports remain references.
    for node in root:
        if not isinstance(node.tag, str):
            continue
        kind = etree.QName(node).localname
        name = node.get("name") or node.get("ref") or node.get("schemaLocation") or kind
        text = etree.tostring(node, encoding="unicode", with_tail=False)
        context = "\n".join(f"schema {key}: {value}" for key, value in root.attrib.items())
        block = _block(path, context + "\n" + text, len(blocks) + 1, node.sourceline or 1, node.sourceline or 1, f"{kind}: {name}")
        block.metadata["source_locator"] = {"kind": "xml", "line": node.sourceline or 1, "name": name, "namespace": namespace}
        blocks.append(block)
    if not blocks:
        blocks.append(_block(path, etree.tostring(root, encoding="unicode"), 1, 1, 1, "schema"))
    return blocks


def parse_text_format(path: Path) -> list[ParsedBlock]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return parse_jsonl(path)
    if suffix == ".xsd":
        return parse_xsd(path)
    if suffix == ".tex":
        return _parse_tex(path)
    lines = read_text_smart(path).splitlines()
    starts: list[tuple[int, str]] = [(0, "")]
    for i, line in enumerate(lines):
        title = ""
        if suffix == ".rst" and i + 1 < len(lines) and line and not line[0].isspace():
            underline = lines[i + 1].strip()
            if len(underline) >= len(line.strip()) and re.fullmatch(r"([=~^`:+*#\"_-])\1{2,}", underline):
                title = line.strip()
        elif suffix == ".tex":
            match = re.match(r"\\(?:part|chapter|(?:sub){0,2}section|paragraph)\*?(?:\[[^\]]*\])?\{([^\n]+?)\}", line)
            if match:
                title = match[1]
        elif suffix == ".bib":
            match = re.match(r"\s*@([a-zA-Z]+)\s*[({]\s*([^,\s]+)", line)
            if match:
                title = f"{match[1]}: {match[2]}"
        elif suffix == ".feature":
            match = re.match(r"\s*(Feature|Rule|Background|Scenario(?: Outline)?|功能|规则|背景|场景(?:大纲)?)\s*[:：]\s*(.*)", line)
            if match:
                title = f"{match[1]}: {match[2]}"
        if title:
            if i == 0:
                starts[0] = (i, title)
            else:
                starts.append((i, title))
    blocks = []
    for n, (start, title) in enumerate(starts):
        end = starts[n + 1][0] if n + 1 < len(starts) else len(lines)
        text = "\n".join(lines[start:end]).strip()
        if text:
            blocks.append(_block(path, text, len(blocks) + 1, start + 1, end, title))
    return blocks


def _parse_tex(path: Path) -> list[ParsedBlock]:
    """Keep the source intact while carrying TeX's explicit section hierarchy."""
    levels = {name: i for i, name in enumerate(
        ("part", "chapter", "section", "subsection", "subsubsection", "paragraph", "subparagraph"), 1)}
    lines = read_text_smart(path).splitlines()
    starts = [(0, "", None)]
    literal = None
    for i, line in enumerate(lines):
        # Section commands inside literal examples are code, not document structure.
        if literal:
            if re.search(r"\\end\{" + re.escape(literal) + r"\}", line):
                literal = None
            continue
        begin = re.search(r"\\begin\{(verbatim\*?|lstlisting|minted)\}", line)
        if begin:
            literal = begin[1]
            continue
        match = re.match(r"\s*\\(part|chapter|section|subsection|subsubsection|paragraph|subparagraph)\*?(?:\[[^\]]*\])?\{", line)
        if not match:
            continue
        depth, end = 1, match.end()
        while end < len(line) and depth:
            if line[end] == "\\":
                end += 2
                continue
            depth += (line[end] == "{") - (line[end] == "}")
            end += 1
        if depth:
            continue
        entry = (i, line[match.end():end - 1], levels[match[1]])
        if i == 0:
            starts[0] = entry
        else:
            starts.append(entry)
    tracker = SectionTracker()
    blocks = []
    pending_start = None
    for n, (start, title, level) in enumerate(starts):
        tracker.observe(level, title)
        end = starts[n + 1][0] if n + 1 < len(starts) else len(lines)
        # An empty parent contributes its path and original command to the next
        # section, rather than becoming an isolated embedding of its heading.
        body = lines[start + 1:end] if title else lines[start:end]
        heading_line_only = bool(title and re.fullmatch(r"\s*\\[a-z]+\*?(?:\[[^\]]*\])?\{.*\}\s*", lines[start]))
        if heading_line_only and not any(s.strip() for s in body) and n + 1 < len(starts):
            if pending_start is None:
                pending_start = start
            continue
        first = start if pending_start is None else pending_start
        text = "\n".join(lines[first:end]).strip()
        if text:
            blocks.append(_block(path, text, len(blocks) + 1, first + 1, end, title,
                                 section_path=list(tracker.path)))
        pending_start = None
    return blocks
