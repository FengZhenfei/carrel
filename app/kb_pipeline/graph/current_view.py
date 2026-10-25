"""Source-valid projections for summaries and compiled pages; raw evidence is retained."""
from __future__ import annotations

from copy import deepcopy
from .corrections import document_status, status_on


def filter_descriptions(rows, corrections, *, today):
    for row in rows:
        texts = list(row.get("descriptions") or [])
        sources = list(row.get("description_sources") or [])
        valid = []
        for i, text in enumerate(texts):
            origin = sources[i] if i < len(sources) and isinstance(sources[i], dict) else {}
            path = str(origin.get("source") or origin.get("rel_path") or "")
            if corrections.get("documents") and not path:
                continue
            if status_on(document_status(path, corrections), today=today) == "current":
                valid.append((text, origin))
        row["descriptions"] = [text for text, _ in valid]
        row["description_sources"] = [origin for _, origin in valid]
        row["description"] = ""  # never reuse a summary that included excluded descriptions


def graph_for_pages(graph, units, *, today):
    from .currency import in_scope
    result = deepcopy(graph)
    docs = result.get("documents") or {}
    allowed = {str(key) for key, value in docs.items() if status_on(value, today=today) == "current"}
    # Legacy/in-memory bundles may not have a document map; unit provenance still counts.
    allowed |= {u.doc_id for u in units if u.doc_id not in docs}
    live_units = [u for u in units if u.doc_id in allowed]
    live_points = {p for u in live_units for p in u.point_ids}
    result["documents"] = {key: value for key, value in docs.items() if key in allowed}
    def valid(row):
        ids = list(row.get("doc_ids") or ([row["doc_id"]] if row.get("doc_id") else []))
        return not row.get("doc_status") and (not ids or any(str(d) in allowed for d in ids))
    result["entities"] = [e for e in result.get("entities", []) if valid(e)]
    result["relations"] = [r for r in result.get("relations", []) if valid(r) and in_scope(r, {"mode":"current", "day":today})]
    result["specs"] = [f for f in result.get("specs", []) if valid(f)]
    for row in result["entities"] + result["relations"]:
        if "doc_ids" in row:
            row["doc_ids"] = [d for d in row["doc_ids"] if d in allowed]
    if units:
        result["mentions"] = [m for m in result.get("mentions", []) if m.get("point_id") in live_points]
    return result, live_units
