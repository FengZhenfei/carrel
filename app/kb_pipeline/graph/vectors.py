"""Graph vectors: the entity collection (title: description) and the relation collection (A -[predicate]-> B:
description) are written into Qdrant.

Replaces the old enrichment script (which only back-filled payloads; the vectors were written
upstream). Now vectors and payloads are written together and the point id is determined by (graph_version,
entity/relation id) -- re-running the same version is an idempotent upsert. The payload fields stay compatible
with the old collections (gr_id / title / search_text / type / degree / frequency ...), minus the community
fields, plus pagerank / parent_type / predicate / point_ids.

Incremental append (reuse_from): rows whose gr_id exists in the previous version's collection with unchanged
embedding text (same embed_sha in the payload) take over the previous version's vector; only new rows and rows
whose description changed are embedded. Retrieval is batched, so the previous collection is never read into
memory as a whole.

Reading the previous version and writing the new one go in batches of UPSERT_BATCH, independent of the embedding
batch size (the embedding client splits its own batches); writes are not awaited batch by batch, and the point
count is checked once a collection has been written.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Callable

from ..vector.qdrant import (
    GRAPH_VECTOR_TYPES,
    ensure_graph_collection,
    graph_collection_alias,
    graph_collection_name,
    graph_vector_layout,
)

EMBED_TEXT_MAX_CHARS = 2000
POINT_IDS_LIMIT = 32
# 256 points per batch: a fact point carries three vectors, so the request body is about 10 MB, well within
# Qdrant's per-request limit (32 MB)
UPSERT_BATCH = 256
# payload fields needed to reuse a vector: the embedding fingerprint and the embedding model; old versions
# without a fingerprint recompute it from the remaining fields (see _base_embed_sha)
BASE_PAYLOAD_FIELDS = ["embed_sha", "embed_model", "graph_type", "title", "description", "source", "type", "target"]


def entity_id(key: str) -> str:
    return "e" + hashlib.sha256(str(key).encode("utf-8")).hexdigest()[:20]


def relation_id(source_key: str, target_key: str, predicate: str) -> str:
    raw = f"{source_key}|{target_key}|{predicate}"
    return "r" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def point_id_for(graph_version: str, gr_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"graph:{graph_version}:{gr_id}"))


def embed_sha(text: str) -> str:
    """Fingerprint of the embedding text, written into the payload: the next version uses it to decide whether the
    vector can be reused."""
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()[:16]


def _base_embed_sha(graph_type: str, payload: dict[str, Any]) -> str | None:
    """Embedding fingerprint of a point in the previous version: newer payloads carry it directly; older versions
    (built before embed_sha was written) recompute it from the payload fields -- the embedding text of an entity /
    relation depends only on title / description (source / predicate / target / description); facts have too many
    text fields for a stable recomputation, so old fact versions are always re-embedded."""
    if payload.get("embed_sha"):
        return str(payload["embed_sha"])
    if graph_type == "entity":
        return embed_sha(entity_embed_text({"title": payload.get("title"), "description": payload.get("description")}))
    if graph_type == "relation":
        return embed_sha(relation_embed_text({"source": payload.get("source"), "predicate": payload.get("type"),
                                              "target": payload.get("target"), "description": payload.get("description")}))
    return None


def _retrieve_base(q: Any, collection: str, ids: list[str]) -> dict[str, tuple[str | None, Any, str]] | None:
    """(embedding fingerprint, vector, embedding model) of these points in the previous version's collection. When
    the collection is gone (garbage-collected) or cannot be read -> None, and this collection is re-embedded in
    full. Collections with named vectors (facts) return a {name: vector} dict, reused only if it matches the new
    layout."""
    try:
        points = q.retrieve(collection_name=collection, ids=ids, with_payload=BASE_PAYLOAD_FIELDS, with_vectors=True)
    except Exception as exc:
        print(f"[graph] vector reuse from {collection} skipped: {exc!r}", flush=True)
        return None
    out: dict[str, tuple[str | None, Any, str]] = {}
    for p in points or []:
        payload = dict(p.payload or {})
        vector = p.vector if isinstance(p.vector, (list, dict)) else None
        out[str(p.id)] = (_base_embed_sha(str(payload.get("graph_type") or ""), payload), vector,
                          str(payload.get("embed_model") or ""))
    return out


def entity_embed_text(e: dict[str, Any]) -> str:
    text = f"{e.get('title') or ''}: {e.get('description') or ''}".strip(": ").strip()
    return text[:EMBED_TEXT_MAX_CHARS] or str(e.get("title") or "")


def relation_embed_text(r: dict[str, Any]) -> str:
    text = f"{r.get('source') or ''} -[{r.get('predicate') or 'related_to'}]-> {r.get('target') or ''}: {r.get('description') or ''}"
    return text.strip()[:EMBED_TEXT_MAX_CHARS]


def spec_payload(f: dict[str, Any], *, kb_id: str, source_collection: str, graph_version: str) -> dict[str, Any]:
    from .facts import comparable_number, conditions_text, spec_text, values_text, when_text

    payload = {
        "gr_id": str(f.get("id") or ""),
        "kb_id": kb_id, "graph_version": graph_version, "graph_type": "spec",
        "source_collection": source_collection,
        "title": f"{f.get('subject')} · {f.get('property')}",
        "subject": f.get("subject"), "subject_id": entity_id(f["subject_key"]) if f.get("subject_key") else None,
        "property": f.get("property"), "property_id": entity_id(f["property_key"]) if f.get("property_key") else None,
        "symbol": f.get("symbol") or None,
        # 2026-09-07 fact skeleton: concept key, canonical unit, flag, reference range, validity / axis, series and
        # conflict group
        "concept": f.get("concept") or None, "concept_key": f.get("concept_key") or None,
        "unit_canonical": f.get("unit_canonical") or None, "flag": f.get("flag") or None,
        "ref_min": f.get("ref_min") or None, "ref_max": f.get("ref_max") or None,
        "ref_min_num": f.get("ref_min_num"), "ref_max_num": f.get("ref_max_num"), "bound_distance": f.get("bound_distance"),
        "valid_from": f.get("valid_from") or None, "valid_until": f.get("valid_until") or None, "axis": f.get("axis") or None,
        "when": when_text(f) or None, "period_text": f.get("period_text") or None,
        "series_key": f.get("series_key") or None, "series_len": f.get("series_len"), "series_index": f.get("series_index"),
        "conflict_group": f.get("conflict_group") or None,
        # the quality state is projected together with the fact (Codex re-review N02): downgraded values cannot rely
        # on upstream Python callers voluntarily excluding them
        "confidence": f.get("confidence") or None, "evidence_conflict": f.get("evidence_conflict") or None,
        "cmps": f.get("cmps") or None, "comparable": comparable_number(f) is not None,
        "value": f.get("value") or None, "min": f.get("min") or None, "typ": f.get("typ") or None, "max": f.get("max") or None,
        "unit": f.get("unit") or None,
        "value_num": f.get("value_num"), "min_num": f.get("min_num"), "typ_num": f.get("typ_num"), "max_num": f.get("max_num"),
        # each value field is scalar / range / expression / text: numeric filters trust only scalar, and expressions
        # have no *_num (F01)
        "kinds": f.get("kinds") or None, "ranges": f.get("ranges") or None, "quality": f.get("quality") or None,
        "conditions": f.get("conditions") or {}, "conditions_text": conditions_text(f.get("conditions")),
        "values_text": values_text(f), "text": spec_text(f), "note": f.get("note") or None,
        "search_text": _search_text(f.get("subject"), f.get("property"), f.get("symbol"), f.get("concept"), values_text(f),
                                    conditions_text(f.get("conditions")), f.get("note")),
        "unit_id": f.get("unit_id"), "doc_id": f.get("doc_id"), "rel_path": f.get("rel_path"), "section": f.get("section"),
        "point_ids": list(f.get("point_ids") or [])[:POINT_IDS_LIMIT],
    }
    return {k: v for k, v in payload.items() if v is not None}


def spec_embed_texts(f: dict[str, Any]) -> dict[str, str]:
    """The three embedding texts of a fact: whole sentence, property side, value side (OG-RAG)."""
    from .facts import property_text, spec_text, value_text

    return {"text": spec_text(f), "property": property_text(f) or spec_text(f), "value": value_text(f)}


def page_payload(p: dict[str, Any], *, kb_id: str, source_collection: str, graph_version: str) -> dict[str, Any]:
    """View-layer pages (compile.py) go into the vector store: subject / timeline / source / index pages, competing
    in the same arena as the chunks."""
    payload = {
        "gr_id": str(p.get("id") or ""),
        "kb_id": kb_id, "graph_version": graph_version, "graph_type": "page",
        "source_collection": source_collection,
        "title": p.get("title"), "type": p.get("kind"), "kind": p.get("kind"),
        "text": str(p.get("text") or "")[:8000], "description": str(p.get("summary") or "")[:2000] or None,
        "search_text": _search_text(p.get("title"), p.get("kind"), p.get("summary"), str(p.get("text") or "")[:4000]),
        "entity_keys": list(p.get("entity_keys") or [])[:32], "concept_keys": list(p.get("concept_keys") or [])[:32],
        "doc_ids": list(p.get("doc_ids") or [])[:32], "point_ids": list(p.get("point_ids") or [])[:POINT_IDS_LIMIT],
        "spec_ids": list(p.get("spec_ids") or [])[:64], "path": p.get("path"),
        "series": [str(x)[:200] for x in (p.get("series") or [])][:80] or None,     # the subject page's series lines: used as Q&A context
        "boilerplate": False, "reference": False, "evidence_kind": "page",
    }
    return {k: v for k, v in payload.items() if v is not None}


def page_embed_text(p: dict[str, Any]) -> str:
    text = f"{p.get('title') or ''}: {p.get('summary') or ''}\n{str(p.get('text') or '')}".strip()
    return text[:EMBED_TEXT_MAX_CHARS] or str(p.get("title") or "")


def _search_text(*values: Any) -> str:
    parts: list[str] = []
    for v in values:
        if isinstance(v, (list, tuple)):
            v = " ".join(str(x) for x in v if x)
        text = str(v or "").strip()
        if text and text not in parts:
            parts.append(text)
    return "\n".join(parts)


def _mentions_index(mentions: list[dict[str, Any]]) -> dict[str, list[str]]:
    by_entity: dict[str, list[tuple[int, str]]] = {}
    for m in mentions:
        by_entity.setdefault(str(m["entity_key"]), []).append((int(m.get("count") or 0), str(m["point_id"])))
    return {k: [pid for _, pid in sorted(v, key=lambda x: -x[0])[:POINT_IDS_LIMIT]] for k, v in by_entity.items()}


def entity_payload(e: dict[str, Any], *, kb_id: str, source_collection: str, graph_version: str,
                   point_ids: list[str]) -> dict[str, Any]:
    payload = {
        "gr_id": entity_id(e["key"]),
        "kb_id": kb_id, "graph_version": graph_version, "graph_type": "entity",
        "source_collection": source_collection,
        "title": e.get("title"), "type": e.get("type"), "parent_type": e.get("parent_type") or None,
        # scope of a document-scoped entity (doc_id): same-named entities exist once per document, and search
        # results tell them apart by it
        "scope": e.get("scope") or None,
        "description": e.get("description"),
        "search_text": _search_text(e.get("title"), e.get("aliases"), e.get("type"), e.get("description")),
        "degree": int(e.get("degree") or 0), "frequency": int(e.get("frequency") or 0),
        "pagerank": float(e.get("pagerank") or 0.0),
        "aliases": list(e.get("aliases") or [])[:16],
        "doc_ids": list(e.get("doc_ids") or [])[:16],
        "point_ids": point_ids,
        "unit_count": len(e.get("unit_ids") or []),
        # noise flags: boilerplate-only / reference-number or downgraded type; recall seeds filter them out by default
        "boilerplate": bool(e.get("boilerplate")),
        "reference": bool(e.get("reference")),
        "evidence_kind": str(e.get("evidence_kind") or "body"),
        "attributes": list(e.get("attributes") or [])[:24] or None,
    }
    return {k: v for k, v in payload.items() if v is not None}


def relation_payload(r: dict[str, Any], *, kb_id: str, source_collection: str, graph_version: str,
                     point_ids: list[str]) -> dict[str, Any]:
    payload = {
        "gr_id": relation_id(r["source_key"], r["target_key"], r["predicate"]),
        "kb_id": kb_id, "graph_version": graph_version, "graph_type": "relation",
        "source_collection": source_collection,
        "title": f"{r.get('source')} -> {r.get('target')}",
        "source": r.get("source"), "target": r.get("target"),
        "source_id": entity_id(r["source_key"]), "target_id": entity_id(r["target_key"]),
        "type": r.get("predicate"), "directed": bool(r.get("directed")),
        "description": r.get("description"),
        "search_text": _search_text(r.get("source"), r.get("target"), r.get("predicate"), r.get("description")),
        "weight": float(r.get("weight") or 1.0), "strength_sum": float(r.get("strength_sum") or 0.0),
        "evidence": int(r.get("evidence") or 0), "cooccur": int(r.get("cooccur") or 0),
        "npmi": float(r.get("npmi") or 0.0), "combined_degree": int(r.get("combined_degree") or 0),
        "type_violation": bool(r.get("type_violation")),
        "boilerplate": bool(r.get("boilerplate")),
        "reference": bool(r.get("reference")),
        "evidence_kind": str(r.get("evidence_kind") or "body"),
        "point_ids": point_ids,
    }
    return {k: v for k, v in payload.items() if v is not None}


def _delete_points_not_in(q: Any, collection: str, expected_ids: set[str], *, batch: int = 1000) -> int:
    """Re-running the same version is an idempotent upsert, but points the previous run wrote and this one does not
    (say a few relations fewer after a filter was tightened) would stay in the collection and break the count.
    Delete the points that are not part of this run."""
    from qdrant_client.http import models as qm

    stale: list[str] = []
    offset = None
    while True:
        points, offset = q.scroll(collection_name=collection, limit=batch, offset=offset,
                                  with_payload=False, with_vectors=False)
        stale.extend(str(p.id) for p in points if str(p.id) not in expected_ids)
        if offset is None or not points:
            break
    if stale:
        q.delete(collection_name=collection, points_selector=qm.PointIdsList(points=stale), wait=True)
    return len(stale)


def write_graph_vectors(
    q: Any,
    embed: Callable[[list[str]], list[list[float]]],
    *,
    kb_id: str,
    source_collection: str,
    graph_version: str,
    bundle: dict[str, Any],
    units_by_id: dict[str, Any],
    vector_size: int,
    stage: Callable[[str], None] | None = None,
    batch: int = UPSERT_BATCH,
    reuse_from: dict[str, str] | None = None,
    base_version: str | None = None,
    embed_model: str = "",
    check_stop: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """reuse_from: {graph_type: previous version's collection name}, together with base_version; rows whose
    embedding text is unchanged in the previous version and were computed by the same embedding model reuse the
    vector. embed_model is written into every point's payload: when the model changes at the same dimension the
    database would not object, and comparing text fingerprints alone would mix the old and new models' vectors in
    one collection (Codex review F05).
    check_stop: called at the start of every batch and after each named vector of the facts is embedded; it
    raises the interruption when the build has been asked to stop -- this function waits on the embedding and
    database HTTP calls in the main thread throughout, and a stop signal landing inside such a call would be
    swallowed by the embedding client's retries."""
    from qdrant_client.http import models as qm

    entities = list(bundle.get("entities") or [])
    relations = list(bundle.get("relations") or [])
    specs = list(bundle.get("specs") or [])
    pages = list(bundle.get("pages") or [])
    mention_points = _mentions_index(list(bundle.get("mentions") or []))
    summary: dict[str, Any] = {"collections": {}}

    def unit_points(unit_ids: list[str]) -> list[str]:
        out: list[str] = []
        for uid in unit_ids:
            unit = units_by_id.get(uid)
            if unit is None:
                continue
            for pid in unit.point_ids:
                if pid not in out:
                    out.append(pid)
                if len(out) >= POINT_IDS_LIMIT:
                    return out
        return out

    specs_by_type = {
        "entity": (entities, lambda e: entity_payload(
            e, kb_id=kb_id, source_collection=source_collection, graph_version=graph_version,
            point_ids=mention_points.get(e["key"]) or unit_points(e.get("unit_ids") or [])), entity_embed_text),
        "relation": (relations, lambda r: relation_payload(
            r, kb_id=kb_id, source_collection=source_collection, graph_version=graph_version,
            point_ids=unit_points(r.get("unit_ids") or [])), relation_embed_text),
        "spec": (specs, lambda f: spec_payload(
            f, kb_id=kb_id, source_collection=source_collection, graph_version=graph_version), spec_embed_texts),
        "page": (pages, lambda p: page_payload(
            p, kb_id=kb_id, source_collection=source_collection, graph_version=graph_version), page_embed_text),
    }
    labels = {"entity": "entities", "relation": "relations", "spec": "facts", "page": "pages"}
    for graph_type in GRAPH_VECTOR_TYPES:
        rows, to_payload, to_text = specs_by_type[graph_type]
        if graph_type == "page" and not rows:
            continue                  # no pages generated (view layer off, or no subjects / axis): no empty collection, the alias is optional
        collection = graph_collection_name(source_collection, graph_type, graph_version)
        layout = graph_vector_layout(graph_type)
        status = ensure_graph_collection(q, collection, vector_size, layout=layout)
        base_collection = (reuse_from or {}).get(graph_type) if base_version else None
        written = reused = embedded_rows = 0
        label = labels[graph_type]
        expected_ids: set[str] = set()
        for start in range(0, len(rows), batch):
            if check_stop is not None:
                check_stop()
            chunk = rows[start:start + batch]
            texts = [to_text(r) for r in chunk]          # named vectors: {name: text}; otherwise a single text
            shas = [embed_sha(json.dumps(t, ensure_ascii=False, sort_keys=True) if isinstance(t, dict) else t) for t in texts]
            payloads = [to_payload(r) for r in chunk]
            vectors: list[Any] = [None] * len(chunk)
            if base_collection:
                base_ids = [point_id_for(base_version, p["gr_id"]) for p in payloads]
                base_points = _retrieve_base(q, base_collection, base_ids)
                if base_points is None:
                    if check_stop is not None:
                        check_stop()            # the read may also fail because a stop landed in this call: do not re-embed everything then
                    base_collection = None      # the previous version's collection is gone: re-embed this collection in full
                else:
                    for k, base_id in enumerate(base_ids):
                        hit = base_points.get(base_id)
                        if hit is None or hit[0] != shas[k] or hit[1] is None or hit[2] != (embed_model or ""):
                            continue
                        if layout and not (isinstance(hit[1], dict) and set(hit[1]) == set(layout)):
                            continue            # the old version is a single-vector fact collection: different layout, re-embed
                        if not layout and isinstance(hit[1], dict):
                            continue
                        vectors[k] = hit[1]
            todo = [k for k, v in enumerate(vectors) if v is None]
            if todo:
                if layout:
                    per_name: dict[str, list[list[float]]] = {}
                    for name in layout:
                        per_name[name] = embed([str(texts[k][name]) for k in todo])
                        if check_stop is not None:
                            check_stop()        # a batch is embedded three times: honour a stop before the whole batch is done
                    for pos, k in enumerate(todo):
                        vectors[k] = {name: per_name[name][pos] for name in layout}
                else:
                    fresh = embed([texts[k] for k in todo])
                    for k, vector in zip(todo, fresh, strict=True):
                        vectors[k] = vector
            reused += len(chunk) - len(todo)
            embedded_rows += len(todo)
            points = []
            for payload, vector, sha in zip(payloads, vectors, shas, strict=True):
                payload["embed_sha"] = sha
                if embed_model:
                    payload["embed_model"] = embed_model
                points.append(qm.PointStruct(id=point_id_for(graph_version, payload["gr_id"]), vector=vector, payload=payload))
            expected_ids.update(str(p.id) for p in points)
            # writes to one collection are applied in submission order: waiting for the last batch is enough, once it
            # has landed so have the earlier ones; the point count is checked below
            q.upsert(collection_name=collection, points=points, wait=start + batch >= len(rows))
            written += len(points)
            if stage is not None:
                stage(f"Writing vectors · {label} {written}/{len(rows)}" + (f" (reused {reused})" if reused else ""))
        removed = _delete_points_not_in(q, collection, expected_ids)
        actual = int(q.count(collection_name=collection, exact=True).count)
        if actual != len(rows):
            raise RuntimeError(f"Qdrant {collection} point count mismatch: actual={actual} expected={len(rows)}")
        summary["collections"][graph_type] = {
            "collection": collection, "alias": graph_collection_alias(source_collection, graph_type),
            "status": status, "points": written, "stale_removed": removed,
            "reused": reused, "embedded": embedded_rows,
        }
    return summary
