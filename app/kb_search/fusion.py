"""Multi-channel candidate fusion: merged and deduped by point_id, each candidate keeping every
channel's raw score and origin; RRF only decides the order in which candidates enter the rerank (Q05)."""
from __future__ import annotations

from typing import Any, Iterable

CHANNELS = ("text", "bm25", "graph", "visual")


def rrf_merge(channels: dict[str, list[dict[str, Any]]], *, k: int = 60) -> list[dict[str, Any]]:
    """channels: {channel name: [candidate, ...]}, each candidate carrying at least point_id and score;
    ranks within a channel follow the given order. Returns candidates in descending RRF order, each with
    scores{score_<channel>}, recall_sources, rrf and the per-channel ranks."""
    merged: dict[str, dict[str, Any]] = {}
    for name, rows in channels.items():
        for rank, row in enumerate(rows, 1):
            pid = str(row.get("point_id") or "")
            if not pid:
                continue
            slot = merged.get(pid)
            if slot is None:
                slot = merged[pid] = {"point_id": pid, "scores": {}, "ranks": {}, "recall_sources": [], "rrf": 0.0}
                for key in ("payload", "chunk_uid", "kb_id"):
                    if row.get(key) is not None:
                        slot[key] = row[key]
            if name in slot["ranks"]:
                continue                       # a repeat within the same channel counts only the first time
            slot["scores"][f"score_{name}"] = row.get("score")
            slot["ranks"][name] = rank
            slot["recall_sources"].append(name)
            slot["rrf"] += 1.0 / (k + rank)
            for key in ("payload", "chunk_uid", "kb_id", "entities", "relations"):
                if slot.get(key) is None and row.get(key) is not None:
                    slot[key] = row[key]
    return sorted(merged.values(), key=lambda c: (-c["rrf"], c["point_id"]))


def interleave(groups: Iterable[list[dict[str, Any]]], *, limit: int) -> list[dict[str, Any]]:
    """Round-robin interleaving of candidates across knowledge bases / subjects, deduped, so that every
    group gets a share (Q08 / Q20)."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    lists = [list(g) for g in groups if g]
    idx = 0
    while len(out) < limit and any(lists):
        group = lists[idx % len(lists)]
        idx += 1
        while group:
            row = group.pop(0)
            pid = str(row.get("point_id") or "")
            if pid and pid not in seen:
                seen.add(pid)
                out.append(row)
                break
        lists = [g for g in lists if g]
        if not lists:
            break
    return out
