"""Source ranges for code blocks, including discontiguous module-level statements."""
from __future__ import annotations


def line_ranges(numbers: list[int]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for number in numbers:
        if ranges and number == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], number)
        else:
            ranges.append((number, number))
    return ranges


def source_text(lines: list[str], ranges: list[tuple[int, int]]) -> tuple[str, list[dict]]:
    """Join source runs without changing their contents; record inserted run separators."""
    text = ""
    spans = []
    for start, end in ranges:
        part = "\n".join(lines[start - 1:end])
        if text:
            text += "\n"
        offset = len(text)
        text += part
        spans.append({"start": start, "end": end, "offset": offset, "length": len(part)})
    return text, spans


def module_source(lines: list[str], covered: list[bool]) -> tuple[str, list[dict]]:
    numbers = [i for i in range(1, len(lines) + 1) if not covered[i]]
    while numbers and lines[numbers[0] - 1] == "":
        numbers.pop(0)
    while numbers and lines[numbers[-1] - 1] == "":
        numbers.pop()
    return source_text(lines, line_ranges(numbers))


def code_locator(text: str, spans: list[dict], start: int, end: int) -> dict:
    """Locate a fragment in its original source runs; columns are zero-based."""
    ranges = []
    for span in spans:
        offset = span["offset"]
        lo, hi = max(start, offset), min(end, offset + span["length"])
        if hi <= lo:
            continue
        prefix = text[offset:lo]
        through = text[offset:hi]
        ranges.append({"start": span["start"] + prefix.count("\n"),
                       "end": span["start"] + through.count("\n"),
                       "start_column": len(prefix.rsplit("\n", 1)[-1]),
                       "end_column": len(through.rsplit("\n", 1)[-1])})
    if len(ranges) == 1:
        return {"kind": "lines", **ranges[0]}
    return {"kind": "line_ranges", "ranges": ranges} if ranges else {}
