"""Graph neighbourhood interface: the agent walks the graph step by step itself (multi-hop reasoning
lives on the agent side; blindly expanding three to five hops showed no gain at all in evaluation).
Given an entity (name or id), returns its one-hop relations in the current graph version: predicate,
direction, weight, the entity at the other end and evidence chunks (ready for /context). The name is
first matched exactly by title / alias; if nothing matches, a few candidates are offered by vector for
the agent to choose from, with no generation of any kind."""
from __future__ import annotations

from typing import Any

from kb_pipeline.config import Settings
from kb_pipeline.models import KBSource

ENTITY_FIELDS = ("id", "title", "type", "parent_type", "scope", "description", "pagerank", "degree", "aliases")
EVIDENCE_PER_RELATION = 3
CANDIDATE_LIMIT = 5


def _entity_row(rec: Any) -> dict[str, Any]:
    return {k: rec.get(k) for k in ENTITY_FIELDS}


def resolve_entity(session: Any, kb_id: str, gv: str, *, entity_id: str | None, name: str | None) -> list[dict[str, Any]]:
    """Find an entity by id or by name (title / alias, case-insensitive); several entities with the same
    name are all returned in descending pagerank order, and the caller decides which to use."""
    if entity_id:
        rows = session.run(
            "MATCH (e:Entity {kb_id: $kb, graph_version: $gv, id: $id}) "
            "RETURN e.id AS id, e.title AS title, e.type AS type, e.parent_type AS parent_type, e.scope AS scope, e.description AS description, "
            "e.pagerank AS pagerank, e.degree AS degree, e.aliases AS aliases", kb=kb_id, gv=gv, id=str(entity_id)).data()
        return [_entity_row(r) for r in rows]
    key = str(name or "").strip().casefold()
    if not key:
        return []
    # e.id IS NOT NULL does not change the result (every entity has an id); it is only there so that the plan uses
    # the (kb_id, graph_version, id) index; without it the query is a label scan across all knowledge bases
    rows = session.run(
        "MATCH (e:Entity {kb_id: $kb, graph_version: $gv}) "
        "WHERE e.id IS NOT NULL AND coalesce(e.boilerplate, false) = false "
        "AND (toLower(e.title) = $name OR any(a IN coalesce(e.aliases, []) WHERE toLower(a) = $name)) "
        "RETURN e.id AS id, e.title AS title, e.type AS type, e.parent_type AS parent_type, e.scope AS scope, e.description AS description, "
        "e.pagerank AS pagerank, e.degree AS degree, e.aliases AS aliases ORDER BY coalesce(e.pagerank, 0) DESC LIMIT 10",
        kb=kb_id, gv=gv, name=key).data()
    return [_entity_row(r) for r in rows]


def dense_candidates(settings: Settings, source: KBSource, name: str, *, limit: int = CANDIDATE_LIMIT, q: Any = None) -> list[dict[str, Any]]:
    """When the name has no exact match, take the few most similar entities from the entity collection
    by vector for the caller to choose from (with ids, so the next query can go by id directly)."""
    from kb_pipeline.graph.recall import _query_points, seed_filter
    from kb_pipeline.vector.qdrant import client as qdrant_client, graph_collection_alias

    from .channels import embed_question

    vector = embed_question(settings, name, timeout=10.0)
    if q is None:
        q = qdrant_client(settings.qdrant_url, settings.qdrant_api_key)
    rows = _query_points(q, graph_collection_alias(source.collection, "entity"), vector, int(limit), seed_filter())
    return [{"id": r.get("gr_id"), "title": r.get("title"), "type": r.get("type"), "scope": r.get("scope"),
             "description": str(r.get("description") or "")[:200], "score": round(float(r.get("_score") or 0.0), 4)} for r in rows]


def neighbor_rows(session: Any, kb_id: str, gv: str, entity_id: str, *, limit: int, types: list[str] | None,
                  direction: str) -> list[dict[str, Any]]:
    """One-hop relations: predicate, direction, weight, the entity at the other end; in descending
    weight order, excluding boilerplate relations and reference-type counterparts."""
    rows = session.run(
        """
        MATCH (e:Entity {kb_id: $kb, graph_version: $gv, id: $id})-[r:RELATED_TO]-(n:Entity)
        WHERE coalesce(r.boilerplate, false) = false AND coalesce(n.reference, false) = false
          AND ($types = [] OR r.type IN $types)
        WITH r, n, (startNode(r).id = $id) AS outgoing
        WHERE $direction = 'both' OR ($direction = 'out' AND outgoing) OR ($direction = 'in' AND NOT outgoing)
        RETURN r.id AS relation_id, r.type AS type, outgoing AS outgoing, coalesce(r.directed, false) AS directed,
               r.weight AS weight, r.npmi AS npmi, r.cooccur AS cooccur, r.description AS description,
               coalesce(r.type_violation, false) AS type_violation,
               n.id AS other_id, n.title AS other_title, n.type AS other_type, n.scope AS other_scope, n.pagerank AS other_pagerank
        ORDER BY coalesce(r.weight, 0) DESC, coalesce(n.pagerank, 0) DESC
        LIMIT $limit
        """, kb=kb_id, gv=gv, id=str(entity_id), types=[str(t) for t in (types or [])], direction=direction, limit=int(limit)).data()
    out = []
    for r in rows:
        out.append({
            "relation_id": r.get("relation_id"), "type": r.get("type"), "direction": "out" if r.get("outgoing") else "in",
            "directed": bool(r.get("directed")), "weight": r.get("weight"), "npmi": r.get("npmi"), "cooccur": r.get("cooccur"),
            "description": r.get("description"), "type_violation": bool(r.get("type_violation")),
            "other": {"id": r.get("other_id"), "title": r.get("other_title"), "type": r.get("other_type"), "scope": r.get("other_scope"),
                      "pagerank": r.get("other_pagerank")},
            "evidence": [],
        })
    return out


def attach_evidence(session: Any, kb_id: str, gv: str, rows: list[dict[str, Any]], *, per_relation: int = EVIDENCE_PER_RELATION) -> None:
    """Attach a few evidence chunks to each relation (back to the chunk snapshot via EVIDENCES ->
    CONTRIBUTES_TO), together with the locators /context needs."""
    rids = [str(r["relation_id"]) for r in rows if r.get("relation_id")]
    if not rids:
        return
    data = session.run(
        """
        UNWIND $ids AS rid
        MATCH (tu:TextUnit {kb_id: $kb, graph_version: $gv})-[:EVIDENCES]->(rel:Relation {kb_id: $kb, graph_version: $gv, id: rid})
        WHERE coalesce(tu.kind, 'body') <> 'boilerplate'
        MATCH (c:QdrantChunkSnapshot)-[:CONTRIBUTES_TO]->(tu)
        RETURN rid, c.point_id AS point_id, c.chunk_uid AS chunk_uid, c.rel_path AS rel_path, c.doc_id AS doc_id,
               c.chunk_index AS chunk_index, c.content_version AS content_version, c.page_idx AS page_idx
        """, ids=rids, kb=kb_id, gv=gv).data()
    by_rid: dict[str, list[dict[str, Any]]] = {}
    for d in data:
        slot = by_rid.setdefault(str(d["rid"]), [])
        if len(slot) >= per_relation or any(x["point_id"] == str(d.get("point_id")) for x in slot):
            continue
        slot.append({"point_id": str(d.get("point_id")), "chunk_uid": d.get("chunk_uid"), "rel_path": d.get("rel_path"), "doc_id": d.get("doc_id"),
                     "chunk_index": d.get("chunk_index"), "content_version": d.get("content_version"), "page_idx": d.get("page_idx")})
    for r in rows:
        r["evidence"] = by_rid.get(str(r.get("relation_id")), [])


def entity_docs(session: Any, kb_id: str, gv: str, entity_id: str, *, limit: int = 12) -> list[str]:
    rows = session.run(
        "MATCH (e:Entity {kb_id: $kb, graph_version: $gv, id: $id})-[:MENTIONED_IN]->(c:QdrantChunkSnapshot) "
        "WHERE c.rel_path IS NOT NULL RETURN DISTINCT c.rel_path AS rel_path LIMIT $limit", kb=kb_id, gv=gv, id=str(entity_id), limit=int(limit)).data()
    return sorted(str(r["rel_path"]) for r in rows if r.get("rel_path"))


def backfill_evidence(q: Any, collection: str, rows: list[dict[str, Any]]) -> None:
    """Chunk snapshot nodes carry no doc_id / chunk index / version; fill them in from the main
    collection by point_id so the evidence can be fed straight to /context; deactivated points are
    marked."""
    ids = sorted({str(ev["point_id"]) for r in rows for ev in r.get("evidence") or [] if ev.get("point_id")})
    if not ids or q is None:
        return
    meta: dict[str, dict[str, Any]] = {}
    for start in range(0, len(ids), 256):
        for rec in q.retrieve(collection_name=collection, ids=ids[start:start + 256],
                              with_payload=["is_active", "doc_id", "rel_path", "filename", "chunk_index", "content_version", "page_idx"], with_vectors=False):
            meta[str(rec.id)] = dict(rec.payload or {})
    for r in rows:
        for ev in r.get("evidence") or []:
            pl = meta.get(str(ev.get("point_id")))
            if pl is None:
                ev["active"] = False
                continue
            ev["active"] = bool(pl.get("is_active", True))
            for key in ("doc_id", "rel_path", "filename", "chunk_index", "content_version", "page_idx"):
                if ev.get(key) in (None, "") and pl.get(key) not in (None, ""):
                    ev[key] = pl.get(key)


def neighbors(settings: Settings, source: KBSource, *, entity: str | None, entity_id: str | None, limit: int = 20,
              types: list[str] | None = None, direction: str = "both", q: Any = None, driver: Any = None,
              timeout: float | None = None) -> dict[str, Any]:
    """q / driver: the Qdrant client and Neo4j driver shared by the service; without a driver one is
    created here and closed afterwards. timeout: transaction timeout (seconds) of each Neo4j query, except
    the point lookup of the active version number."""
    from kb_pipeline.graph.neo4j_import import active_neo4j_graph_version, neo4j_driver
    from kb_pipeline.graph.recall import TimedSession

    direction = direction if direction in ("both", "out", "in") else "both"
    own_driver = driver is None
    if own_driver:
        driver = neo4j_driver(settings)
    try:
        gv = active_neo4j_graph_version(driver, source.kb_id)
        if not gv:
            raise KeyError(f"{source.kb_id} has no active graph")
        with driver.session() as session:
            if timeout:
                session = TimedSession(session, timeout)
            matches = resolve_entity(session, source.kb_id, gv, entity_id=entity_id, name=entity)
            if not matches:
                cands = dense_candidates(settings, source, entity or "", q=q) if entity else []
                return {"kb_id": source.kb_id, "graph_version": gv, "found": False, "entity": None, "matches": [], "candidates": cands,
                        "neighbors": [], "note": "no entity with that title or alias; pick one of the candidates by id"}
            # Several entities with the same name (MACD in different scopes): take the one with the most
            # relations, list the rest in matches so the caller can query by id instead
            matches.sort(key=lambda m: (-int(m.get("degree") or 0), -float(m.get("pagerank") or 0.0)))
            center = matches[0]
            rows = neighbor_rows(session, source.kb_id, gv, str(center["id"]), limit=max(1, min(100, int(limit))), types=types, direction=direction)
            attach_evidence(session, source.kb_id, gv, rows)
            center["docs"] = entity_docs(session, source.kb_id, gv, str(center["id"]))
    finally:
        if own_driver:
            driver.close()
    backfill_evidence(q, source.collection, rows)
    return {"kb_id": source.kb_id, "graph_version": gv, "found": True, "entity": center, "matches": matches[1:], "candidates": [],
            "neighbors": rows, "count": len(rows), "direction": direction, "types": list(types or [])}
