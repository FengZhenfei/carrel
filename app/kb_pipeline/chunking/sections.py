"""Resolve section provenance after packing, without changing text boundaries."""
from __future__ import annotations

from dataclasses import replace
import re

from ..models import ParsedBlock


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def common_path(paths: list[list[str]]) -> list[str]:
    common = list(paths[0]) if paths else []
    for path in paths[1:]:
        common = common[:next((i for i, (a, b) in enumerate(zip(common, path)) if a != b),
                              min(len(common), len(path)))]
    return common


class SectionSources:
    """Match ordered chunk text to the original premerged source spans.

    Whitespace may change during sentence packing. Matching otherwise stays exact;
    an unmatched fragment retains all possible sections instead of guessing one.
    """
    def __init__(self, blocks: list[ParsedBlock]):
        self.sources = {}
        self.cursors = {}
        for block in blocks:
            spans = block.metadata.get("section_spans")
            if not spans:
                continue
            self.sources[block.block_id] = (block.text, compact(block.text), [
                {**s, "start": len(compact(block.text[:s["start"]])),
                 "end": len(compact(block.text[:s["end"]]))} for s in spans])

    def locate(self, block: ParsedBlock, text: str, *, advance: bool = True) -> ParsedBlock:
        embedded = block.metadata.get("embedded_source") or {}
        key = embedded.get("block_id", block.block_id)
        if key not in self.sources:
            return block
        original, source, spans = self.sources[key]
        needle = compact(text)
        # The merged block may render its title again ahead of its original text.
        title = compact("TITLE: " + block.title) if block.title else ""
        if title and needle.startswith(title) and not source.startswith(needle):
            needle = needle[len(title):]
        lower = len(compact(original[:embedded["start"]])) if "start" in embedded else 0
        upper = len(compact(original[:embedded["end"]])) if "end" in embedded else len(source)
        cursor = max(lower, self.cursors.get(key, 0))
        start = source.find(needle, cursor, upper) if needle else -1
        ambiguous = start < 0
        if start >= 0:
            selected = [s for s in spans if s["start"] < start + len(needle) and s["end"] > start]
            # Repeated passages can match several sections. Unless this starts at
            # the known source boundary, retain the alternatives, not a guessed owner.
            if start != lower:
                other = source.find(needle, start + 1, upper)
                while other >= 0:
                    alternatives = [s for s in spans if s["start"] < other + len(needle) and s["end"] > other]
                    if {tuple(s["section_path"]) for s in alternatives} != {tuple(s["section_path"]) for s in selected}:
                        ambiguous = True
                        selected.extend(s for s in alternatives if s not in selected)
                    other = source.find(needle, other + 1, upper)
            # Allow whole-sentence overlap, but do not reuse an identical occurrence.
            if advance:
                self.cursors[key] = start + 1
        else:
            selected = spans
        paths = []
        for span in selected:
            path = list(span.get("section_path") or [])
            if path not in paths:
                paths.append(path)
        metadata = {**block.metadata, "_chunk_section_paths": paths}
        if ambiguous:
            metadata["section_match"] = "ambiguous"
        return replace(block, metadata=metadata)


def resolve_sections(block: ParsedBlock) -> ParsedBlock:
    paths = block.metadata.get("_chunk_section_paths")
    if paths is None:
        return block
    metadata = {k: v for k, v in block.metadata.items() if k != "_chunk_section_paths"}
    metadata["section_path"] = common_path(paths)
    metadata["section_paths"] = paths
    return replace(block, metadata=metadata)
