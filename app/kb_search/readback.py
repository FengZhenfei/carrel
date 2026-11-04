"""Read one bounded round of graph provenance after direct text ranking."""
from __future__ import annotations

from collections import defaultdict, deque
import json
import time

from kb_pipeline.chunking.context import payload_text
from kb_pipeline.utils import count_tokens

from . import channels, docstatus
from .evidence import source_row
from .rerank import Reranker, rerank
from .text import token_set

POOL_SIZE = 24
MAX_SOURCES = 6
TOKEN_BUDGET = 2400
LEAD_FIELDS = ("title", "source", "target", "type", "subject", "property", "value", "hint",
               "conditions", "conditions_text", "when", "scope", "description")


def candidates(question, result):
    """Prioritize relevant leads, then rotate among their documents before capping the pool."""
    seen = set()
    known_docs = set()
    for source in result["sources"]:
        kb = source["kb_id"]
        known_docs.add((kb, source.get("doc_id")))
        seen.add((kb, source["point_id"]))
        for piece in (source.get("stitched") or {}).get("pieces", []):
            if piece.get("kept", True) and not piece.get("partial"):
                seen.add((kb, piece.get("point_id")))
    query = token_set(question)
    leads = []
    for group in ("specs", "relationships", "entities"):
        for row in result[group]:
            kb = row["kb_id"]
            if kb not in result["kbs"] or row.get("source_valid") is False:
                continue
            description = {key: row[key] for key in LEAD_FIELDS if row.get(key) not in (None, "")}
            words = token_set(json.dumps(description, ensure_ascii=False))
            relevance = len(query & words) / max(1, len(query))
            for locator in row.get("evidence") or []:
                if (not locator.get("point_id") or not locator.get("content_version")
                        or not locator.get("doc_id") or locator.get("active") is False):
                    continue
                new_doc = (kb, locator["doc_id"]) not in known_docs
                leads.append((relevance + 0.05 * new_doc, kb, locator))
    buckets = defaultdict(deque)
    for _, kb, locator in sorted(leads, key=lambda item: -item[0]):
        key = (kb, locator["point_id"])
        if key not in seen:
            seen.add(key)
            buckets[kb, locator["doc_id"]].append((kb, locator))
    pending = deque(buckets.values())
    pool = []
    while pending and len(pool) < POOL_SIZE:
        bucket = pending.popleft()
        pool.append(bucket.popleft())
        if bucket:
            pending.append(bucket)
    return pool


def read_originals(settings, ss, q, result, scope, hint, until, driver):
    """Return additional numbered sources and the verified locators examined this round."""
    pool = candidates(result["question"], result)
    by_kb = defaultdict(dict)
    for kb, locator in pool:
        by_kb[kb][locator["point_id"]] = locator
    eligible = []
    examined = []
    for kb, locators in by_kb.items():
        source = settings.sources[kb]
        points = q.retrieve(source.collection, ids=list(locators), with_payload=True, with_vectors=False)
        for point in points:
            pid, payload = str(point.id), point.payload or {}
            locator = locators.get(pid)
            if (not locator or not payload.get("is_active") or payload.get("kb_id", kb) != kb
                    or payload.get("doc_id") != locator["doc_id"]
                    or payload.get("content_version") != locator["content_version"]
                    or not channels.hint_allows(hint, payload)
                    or not docstatus.allowed(settings, source, payload.get("rel_path"), scope)):
                continue
            examined.append({"kb_id": kb, "point_id": pid, "content_version": payload["content_version"]})
            eligible.append({"kb_id": kb, "point_id": pid, "payload": payload})

    added = []
    used = 0
    if eligible:
        remaining = min(ss.rerank_timeout, until - time.time())
        if remaining <= 0:
            raise TimeoutError("graph readback budget exhausted")
        ranker = Reranker(settings.reranker_base_url, model_id=getattr(settings, "reranker_model_id", ""),
                          api_key=getattr(settings, "reranker_api_key", ""), timeout=remaining)
        scores = rerank(ranker, result["question"], eligible, window_tokens=ss.rerank_window_tokens,
                        overlap_tokens=ss.rerank_overlap_tokens)
        ranked = sorted(zip(scores, eligible), key=lambda x: (-x[0], x[1]["kb_id"], x[1]["point_id"]))
        texts = [" ".join(s["text"].split()) for s in result["sources"]]
        for score, item in ranked:
            if score < ss.rerank_threshold or len(added) >= MAX_SOURCES:
                break
            payload = item["payload"]
            text = payload_text(payload)
            normalized = " ".join(text.split())
            if not normalized or any(normalized in existing for existing in texts):
                continue
            size = count_tokens(text)
            if used + size > TOKEN_BUDGET:
                continue
            texts.append(normalized)
            used += size
            item.update(score_rerank=score, recall_sources=["graph_readback"], accepted=True)
            row = source_row(len(result["sources"]) + len(added) + 1, item, payload, role="graph_followup")
            if payload.get("doc_date"):
                row["doc_date"] = payload["doc_date"]
            added.append(row)
        docstatus.annotate(settings, added, scope)
        if scope.get("as_of") and added:
            remaining = until - time.time()
            if remaining <= 0:
                raise TimeoutError("graph readback date lookup budget exhausted")
            docstatus.annotate_dates(settings, added, scope, driver=driver, timeout=remaining)
    return added, {"status": "completed" if pool else "no_candidates", "examined": examined,
                   "source_numbers": [row["n"] for row in added], "added": len(added), "tokens": used}
