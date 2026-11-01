"""Recover printed code from the native text inside an existing PDF code region."""
from __future__ import annotations

import re
import ast
import textwrap
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from statistics import median

from ..models import ParsedBlock
from .pdf_evidence import _rect


_LANGUAGE_LABEL = re.compile(
    r"(?:python|javascript|typescript|java|c|c\+\+|c#|csharp|go|rust|ruby|php|"
    r"swift|kotlin|scala|r|julia|sql|bash|sh|shell|powershell|html|css|xml|json|yaml|toml)", re.I)
_SPLIT_LABEL = re.compile(r"code (?:block|listing) split\s*\((\d+)\s*/\s*(\d+)\)", re.I)


def repair_code_items(items: list[dict], path: Path) -> list[dict]:
    """Recover labelled native listings and bind small printed labels to their boxes.

    A language label introduces the next listing; a split counter closes the
    preceding one. Only adjacent body items with compatible geometry qualify.
    Bound labels retain their source positions; the section resolver excludes
    them from the heading hierarchy without repacking the surrounding prose.
    """
    from .common import bbox, caption_of, item_text, label_of, page_idx

    items = [dict(i) for i in items]
    body = [i for i in items if label_of(i) not in
            {"header", "footer", "page_number", "discarded"} and (item_text(i) or i.get("img_path"))]
    panels: dict[int, list[list[float]]] = {}

    def label(item):
        text = item_text(item).strip()
        box = bbox(item)
        if item.get("_pdf_code_label_owned") or label_of(item) not in {"text", "title"} or not box or not 0 < box[3] - box[1] <= 20:
            return None
        if _LANGUAGE_LABEL.fullmatch(text):
            return "language"
        match = _SPLIT_LABEL.fullmatch(text)
        if match and 1 <= int(match[1]) <= int(match[2]):
            return "split"
        return None

    def adjacent(tag, owner, kind, *, cross_page=False):
        a, b = bbox(tag), bbox(owner)
        if not a or not b:
            return False
        aligned = min(a[2], b[2]) > max(a[0], b[0])
        # A continuation may start deeply indented. A native enclosing panel
        # supplies the column boundary; never extend the code text rectangle.
        if not aligned:
            aligned = any(p[1] <= b[1] and b[3] <= p[3]
                          and min(p[2], b[2]) - max(p[0], b[0]) >= (b[2] - b[0]) * .6
                          and min(a[2], p[2]) > max(a[0], p[0])
                          for p in panels.get(page_idx(owner), []))
        if not aligned:
            return False
        if page_idx(tag) == page_idx(owner):
            gap = b[1] - a[3] if kind == "language" else a[1] - b[3]
            return -2 <= gap <= 40
        return (cross_page and kind == "language" and page_idx(tag) is not None
                and page_idx(owner) == page_idx(tag) + 1 and b[1] <= 250)

    # Classification requires independent native font, syntax and OCR agreement.
    # Ordinary prose, scans and incomplete/contradictory native layers stay as-is.
    try:
        import pymupdf
        doc = pymupdf.open(path)
    except Exception:
        doc = None
    if doc is not None:
        with doc:
            for page_no in {page_idx(i) for i in body if label_of(i) == "code"}:
                if page_no is None or not 1 <= page_no <= len(doc):
                    continue
                page = doc[page_no - 1]
                panels[page_no] = [[r.x0 * 1000 / page.rect.width, r.y0 * 1000 / page.rect.height,
                                   r.x1 * 1000 / page.rect.width, r.y1 * 1000 / page.rect.height]
                    for drawing in page.get_drawings() if drawing.get("fill") is not None
                    and (r := drawing["rect"]).get_area() < page.rect.get_area() * .85]
            for tag, item in zip(body, body[1:]):
                page = page_idx(item)
                if (label(tag) != "language" or label_of(item) != "text" or label(item)
                        or not adjacent(tag, item, "language") or not page or not 1 <= page <= len(doc)):
                    continue
                native = _listing(doc[page - 1], _rect(doc[page - 1], bbox(item)))
                source = re.sub(r"\\([\\`*_{}\[\]()#+.!>-])", r"\1", item_text(item))
                compact = lambda s: re.sub(r"\s+", "", s)
                if not native.strip() or compact(source) != compact(native):
                    continue
                lines = [line.strip() for line in native.splitlines() if line.strip()]
                if len(lines) < 2:
                    continue
                if item_text(tag).casefold() == "python":
                    try:
                        tree = ast.parse(textwrap.dedent(native))
                    except (SyntaxError, ValueError):
                        continue
                    if not any(isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign,
                            ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import,
                            ast.ImportFrom, ast.For, ast.While, ast.If, ast.With, ast.Call))
                            for n in ast.walk(tree)):
                        continue
                else:
                    statements = [line for line in lines if not line.startswith(("#", "//", "--"))]
                    syntax = re.compile(r"^(?:(?:(?:const|let|var)\s+)?[\w.$\[\]]+\s*(?:=(?!=)|:=|<-)\s*\S"
                                        r"|[\w.$]+\s*\([^)]*\)\s*;?$|[{}\[\]]\s*[,;]?$)")
                    if not statements or not all(syntax.search(line) for line in statements):
                        continue
                item.update(type="code", code_body=native,
                            _pdf_code_recovery={"source": "native_text", "ocr_text": item_text(item)})
                # item_text prefers text over code_body; retain the source in metadata instead.
                item.pop("text", None)
                item.pop("text_level", None)

    for index, tag in enumerate(body):
        kind = label(tag)
        if not kind:
            continue
        owner_index = index + 1 if kind == "language" else index - 1
        if not 0 <= owner_index < len(body):
            continue
        owner = body[owner_index]
        if label_of(owner) != "code" or not adjacent(tag, owner, kind, cross_page=True):
            continue
        text = item_text(tag).strip()
        old = caption_of(owner)
        # An explicitly different language caption makes ownership ambiguous.
        if kind == "language" and _LANGUAGE_LABEL.fullmatch(old) and old.casefold() != text.casefold():
            continue
        # Printed language/counter labels stay as source context. Do not repeat
        # this layout text in every code embedding or change existing captions.
        owner.setdefault("_pdf_code_label_sources", []).append(
            {"page": page_idx(tag), "bbox": bbox(tag), "text": text, "kind": kind})
        if kind == "language":
            owner["_pdf_code_language"] = text.casefold()
        tag["_pdf_code_label_owned"] = True
    return items


def _listing(page, rect) -> str:
    # Select complete lines before clipping horizontally. Clipping first can
    # turn descenders from the preceding line into stray letters at the top.
    valid_lines = [line for b in page.get_text("dict").get("blocks", [])
                   for line in b.get("lines", [])
                   if line["bbox"][1] >= rect.y0 - 2 and line["bbox"][3] <= rect.y1 + 2]
    spans = [s for b in page.get_text("dict", clip=rect).get("blocks", [])
             for line in b.get("lines", []) if tuple(line.get("dir", (1, 0))) == (1, 0)
             and any(abs(line["bbox"][3] - full["bbox"][3]) < 1 for full in valid_lines)
             for s in line.get("spans", []) if s.get("text", "").strip()]
    if not spans:
        return ""
    mono = [s for s in spans if s.get("flags", 0) & 8]
    chars = sum(len(s["text"].strip()) for s in spans)
    if sum(len(s["text"].strip()) for s in mono) < chars * .6:
        return ""  # Prose, scanned code and unreliable font maps keep the OCR result.
    advances = [(s["bbox"][2] - s["bbox"][0]) / len(s["text"]) for s in mono if s["text"]]
    step = median(advances)
    if step <= 0:
        return ""
    rows: list[list[dict]] = []
    for span in sorted(spans, key=lambda s: (s["origin"][1], s["origin"][0])):
        if not rows or abs(span["origin"][1] - rows[-1][0]["origin"][1]) > 1.5:
            rows.append([])
        rows[-1].append(span)
    mono_left = min(s["origin"][0] for s in mono)
    mono_size = median(s["size"] for s in mono)
    gutter = [s for s in spans if re.fullmatch(r"\d{1,6}", s["text"].strip())
              and s["bbox"][2] < mono_left - step and s["size"] < mono_size * .8]
    numbers = [int(s["text"]) for s in sorted(gutter, key=lambda s: s["origin"][1])]
    gutter_ids = {id(s) for s in gutter} if len(numbers) >= 2 and all(b > a for a, b in zip(numbers, numbers[1:])) else set()
    left = min(s["origin"][0] for s in spans if id(s) not in gutter_ids)
    gaps = [b[0]["origin"][1] - a[0]["origin"][1] for a, b in zip(rows, rows[1:])]
    leading = median(gaps) if gaps else 0
    lines = []
    for i, row in enumerate(rows):
        if i and leading and gaps[i - 1] > leading * 1.6:
            lines.append("")
        end = left
        line = ""
        wrapped = False
        for s in sorted(row, key=lambda s: s["origin"][0]):
            if id(s) in gutter_ids:
                continue
            if s["text"].strip() == "↪" and not s.get("flags", 0) & 8 and s["origin"][0] < mono_left + step:
                wrapped = True
                end = s["bbox"][2]
                continue
            spaces = max(0, round((s["origin"][0] - end) / step))
            line += " " * spaces + s["text"]
            end = s["bbox"][2]
        if wrapped and lines:
            # A separately printed continuation marker is layout, not source code.
            lines[-1] += line.lstrip().rstrip()
        else:
            lines.append(line.rstrip())
    return "\n".join(lines).rstrip() + "\n"


def recover_native_code(blocks: list[ParsedBlock], path: Path) -> int:
    """Use visible native monospace text only where it agrees with the OCR region.

    This does not classify new regions as code, execute code or rewrite identifiers.
    OCR is retained when either text layer is substantially different or incomplete.
    """
    try:
        import pymupdf
        doc = pymupdf.open(path)
    except Exception:
        return 0
    changed = 0
    with doc:
        for block in blocks:
            if block.block_type != "code" or not block.bbox or not block.page_idx or not 1 <= block.page_idx <= len(doc):
                continue
            page = doc[block.page_idx - 1]
            native = _listing(page, _rect(page, block.bbox))
            if not native.strip() or any(c in native for c in ("\ufffd", "\x00")):
                continue
            source = re.sub(r"\A\s*```[\w+-]*[^\S\n]*\n|\n```\s*\Z", "", block.text)
            old = re.sub(r"\s+", "", source)
            new = re.sub(r"\s+", "", native)
            if len(new) < len(old) * .85:
                continue
            # Agreement must cover both the ordered characters and OCR identifiers;
            # a different hidden layer is not a correction of this listing.
            alignment = SequenceMatcher(None, old, new, autojunk=False)
            match = sum(m.size for m in alignment.get_matching_blocks())
            tokens = Counter(re.findall(r"[A-Za-z_]\w*", source))
            common = tokens & Counter(re.findall(r"[A-Za-z_]\w*", native))
            if match < len(old) * .75 or (tokens and sum(common.values()) < sum(tokens.values()) * .65):
                continue
            if any(tag in {"delete", "replace"} and a1-a0 >= 12
                   and sum(ch.isalnum() for ch in old[a0:a1]) >= 8
                   for tag, a0, a1, _, _ in alignment.get_opcodes()):
                continue  # Keep OCR when a native layer would remove a substantial source run.
            language = re.match(r"\s*```([\w+-]+)\s*\n", block.text)
            try:
                ast.parse(textwrap.dedent(source))
            except (SyntaxError, ValueError):
                pass  # Printed page fragments need not be executable programs.
            else:
                try:
                    ast.parse(textwrap.dedent(native))
                except (SyntaxError, ValueError):
                    continue  # A clipped text layer must not break a complete listing.
            if block.text != native:
                if language:
                    block.metadata.setdefault("code_language", language.group(1))
                block.metadata.setdefault("pdf_code_recovery", {"source": "native_text", "ocr_text": block.text})
                block.text = native
                changed += 1
    return changed
