"""Exact, declarative evidence checks for retrieval evaluation; no model judgement."""
from __future__ import annotations

import unicodedata

CHECKS = ("required_facts", "forbidden_facts", "required_relations", "forbidden_relations",
          "required_entities", "forbidden_entities", "required_paths", "required_documents")


def _normal(value):
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def matches(row, expected):
    """All fields must occur on the same record. Conditions are a subset, values are exact."""
    if not isinstance(expected, dict) or not expected:
        return False
    for key, wanted in expected.items():
        actual = row.get(key)
        if key == "predicate":
            actual = row.get("predicate", row.get("type"))
        if isinstance(wanted, dict):
            if not isinstance(actual, dict):
                return False
            if (wanted and not matches(actual, wanted)) or (not wanted and actual):
                return False
        elif actual is None or _normal(actual) != _normal(wanted):
            return False
    return True


def evaluate_structured(item, output):
    result = {}
    facts = output.get("specs") or []
    relations = output.get("relationships") or []
    entities = output.get("entities") or []
    for name, rows in (("facts", facts), ("relations", relations), ("entities", entities)):
        for prefix in ("required", "forbidden"):
            field = f"{prefix}_{name}"
            expected = item.get(field)
            if expected is None:
                continue
            if not isinstance(expected, list) or not expected or any(not isinstance(r, dict) or not r for r in expected):
                raise ValueError(f"{field} must be a nonempty list of field selectors")
            found = [any(matches(row, selector) for row in rows) for selector in expected]
            passed = found if prefix == "required" else [not v for v in found]
            result[field] = all(passed)
            result[field + "_coverage"] = {"passed": sum(passed), "total": len(passed)}
    paths = item.get("required_paths")
    if paths is not None:
        if not isinstance(paths, list) or not paths:
            raise ValueError("required_paths must contain directed edge sequences")
        passed = []
        for path in paths:
            if not isinstance(path, list) or not path or any(not isinstance(edge, dict) or
                    not all(edge.get(k) for k in ("source", "predicate", "target")) for edge in path):
                raise ValueError("each path edge needs source, predicate and target")
            if any(_normal(a["target"]) != _normal(b["source"]) for a, b in zip(path, path[1:])):
                raise ValueError("path edges must be connected and ordered")
            passed.append(all(any(matches(row, edge) for row in relations) for edge in path))
        result["required_paths"] = all(passed)
        result["required_paths_coverage"] = {"passed": sum(passed), "total": len(passed)}
    docs = item.get("required_documents")
    if docs is not None:
        if not isinstance(docs, list) or not docs or any(not isinstance(x, str) or not x for x in docs):
            raise ValueError("required_documents must contain document paths or ids")
        got = {str(row.get(k)) for row in output.get("sources") or [] for k in ("doc_id", "rel_path", "doc") if row.get(k)}
        result["required_documents"] = all(d in got for d in docs)
        result["required_documents_coverage"] = {"passed": sum(d in got for d in docs), "total": len(docs)}
    return result
