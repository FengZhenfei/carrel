"""Read-only field paths and value locations in structured configuration source."""
from __future__ import annotations

import json
import re
from typing import Any


def field_name(path: tuple) -> str:
    text = ""
    for part in path:
        if isinstance(part, int):
            text += f"[{part}]"
        elif re.fullmatch(r"[\w$-]+", str(part)):
            text += ("." if text else "") + str(part)
        else:
            text += "[" + json.dumps(str(part), ensure_ascii=False) + "]"
    return text or "(root)"


def config_fields(data: Any, path: tuple = (), ancestors: frozenset[int] = frozenset()):
    """All leaves, retaining scalar arrays as one field and guarding recursive YAML aliases."""
    if isinstance(data, (dict, list)):
        if id(data) in ancestors:
            return
        ancestors = ancestors | {id(data)}
    if isinstance(data, dict) and data:
        for key, value in data.items():
            yield from config_fields(value, path + (key,), ancestors)
    elif isinstance(data, list) and any(isinstance(x, (dict, list)) for x in data):
        for i, value in enumerate(data):
            yield from config_fields(value, path + (i,), ancestors)
    else:
        yield path, data


def config_locations(text: str, suffix: str) -> dict[tuple, dict]:
    """JSON is a YAML syntax subset; compose reads source marks without constructing objects."""
    if suffix not in {".json", ".json5", ".yaml", ".yml"}:
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    try:
        docs = list(yaml.compose_all(text, Loader=yaml.SafeLoader))
    except yaml.YAMLError:
        return {}
    out = {}

    def visit(node, path, ancestors=frozenset()):
        if node is None or id(node) in ancestors:
            return
        ancestors = ancestors | {id(node)}
        out[path] = {"kind": "lines", "start": node.start_mark.line + 1, "end": node.end_mark.line + 1,
                     "start_column": node.start_mark.column, "end_column": node.end_mark.column,
                     "field_path": list(path)}
        if isinstance(node, yaml.MappingNode):
            for key, value in node.value:
                visit(value, path + (key.value,), ancestors)
        elif isinstance(node, yaml.SequenceNode):
            for i, value in enumerate(node.value):
                visit(value, path + (i,), ancestors)

    for i, node in enumerate(docs):
        visit(node, (i,) if len(docs) > 1 else ())
    return out


def overlaps(a: dict, b: dict) -> bool:
    if a.get("kind") == "line_ranges":
        return any(overlaps({"kind": "lines", **r}, b) for r in a.get("ranges") or [])
    if a.get("kind") != "lines" or b.get("kind") != "lines":
        return False
    left = (a["start"], a.get("start_column", 0)), (a["end"], a.get("end_column", 10**12))
    right = (b["start"], b.get("start_column", 0)), (b["end"], b.get("end_column", 10**12))
    return left[0] < right[1] and right[0] < left[1]
