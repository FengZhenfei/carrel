"""Structured graph lookups: the agent walks the graph and lists things step by step itself (multi-hop
reasoning lives on the agent side; blindly expanding three to five hops showed no gain at all in evaluation).
All three endpoints only read the current graph version and generate nothing:
- neighbours: given an entity (name or id), its one-hop relations: predicate, direction, weight, the entity
  at the other end and evidence chunks (ready for /context); the name is first matched exactly by title /
  alias, and if nothing matches a few candidates are offered by vector for the agent to choose from;
- entities: entities by type / upper class / name, with a total and paging ("all of them" questions);
- facts: qualified facts by subject / property, with a total and paging, rows shaped like the specs of /search.
What is listed is what the graph registered (the result of model extraction), not a complete inventory of the
documents."""
from __future__ import annotations

from typing import Any, Iterable

from kb_pipeline.config import Settings
from kb_pipeline.models import KBSource

from .text import place, position

ENTITY_FIELDS = ("id", "title", "type", "parent_type", "scope", "description", "pagerank", "degree", "aliases")
EVIDENCE_PER_RELATION = 3
EVIDENCE_PER_FACT = 3
CANDIDATE_LIMIT = 5
LIST_LIMIT_MAX = 200
LIST_DESCRIPTION_CHARS = 200        # a listing keeps only the start of each description; the full text comes from the neighbours endpoint
LIST_DOCS_PER_ENTITY = 3            # how many of the documents mentioning it most a listed entity carries (for citations); the document count comes with them
PROPERTY_SUMMARY_LIMIT = 50
CONCEPT_KEYS_MAX = 200
# Payload keys fetched from the main collection for an evidence chunk: the locators of /context, the position
# string, whether it is still active
EVIDENCE_PAYLOAD_KEYS = ["is_active", "doc_id", "rel_path", "filename", "chunk_index", "content_version", "page_idx",
                         "slide_idx", "sheet_name", "row_start", "row_end", "section_path"]
# Entities a listing leaves out: those found only on boilerplate pages and reference numbers (the same rule graph
# recall uses for its seeds).
# e.id IS NOT NULL does not change the result (every entity has an id); it is only there so that the plan uses
# the (kb_id, graph_version, id) index; without it the query is a label scan across all knowledge bases
LIVE_ENTITY = "e.id IS NOT NULL AND coalesce(e.boilerplate, false) = false AND coalesce(e.reference, false) = false"
_FACT_TEXT_KEYS = ("f.property", "f.symbol", "f.concept")
FACT_MATCH = {
    "exact": "(" + " OR ".join(f"toLower(coalesce({k}, '')) = $p" for k in _FACT_TEXT_KEYS) + ")",
    "contains": "(" + " OR ".join(f"toLower(coalesce({k}, '')) CONTAINS $p" for k in _FACT_TEXT_KEYS) + ")",
}
# The basic graph-database fields a fact falls back to when its payload cannot be fetched from the fact
# collection (the alias is switching versions)
_FACT_BASIC = ("id", "subject", "property", "symbol", "concept", "concept_key", "value", "min", "typ", "max", "unit", "unit_canonical",
               "valid_from", "valid_until", "series_key", "series_index", "conflict_group", "section")


def kb_name(source: Any) -> str | None:
    """The knowledge base's folder name (its top-level directory under the mirror): callers build the cited
    path as "folder name / rel_path"."""
    root = getattr(source, "source_root", None)
    return str(root) if root else None


def _entity_row(rec: Any) -> dict[str, Any]:
    return {k: rec.get(k) for k in ENTITY_FIELDS}


def resolve_entity(session: Any, kb_id: str, gv: str, *, entity_id: str | None, name: str | None) -> list[dict[str, Any]]:
    """Find an entity by id or by name (title / alias, ignoring case and spaces); several entities with the
    same name are all returned in descending pagerank order, and the caller decides which to use.
    Ignoring spaces: a question writes "ZK 200" where the graph registered "ZK200"; spellings that differ
    only by spaces count as the same name."""
    if entity_id:
        rows = session.run(
            "MATCH (e:Entity {kb_id: $kb, graph_version: $gv, id: $id}) "
            "RETURN e.id AS id, e.title AS title, e.type AS type, e.parent_type AS parent_type, e.scope AS scope, e.description AS description, "
            "e.pagerank AS pagerank, e.degree AS degree, e.aliases AS aliases", kb=kb_id, gv=gv, id=str(entity_id)).data()
        return [_entity_row(r) for r in rows]
    key = "".join(str(name or "").casefold().split())
    if not key:
        return []
    # e.id IS NOT NULL does not change the result (every entity has an id); it is only there so that the plan uses
    # the (kb_id, graph_version, id) index; without it the query is a label scan across all knowledge bases
    rows = session.run(
        "MATCH (e:Entity {kb_id: $kb, graph_version: $gv}) "
        "WHERE e.id IS NOT NULL AND coalesce(e.boilerplate, false) = false "
        "AND (replace(toLower(e.title), ' ', '') = $name OR any(a IN coalesce(e.aliases, []) WHERE replace(toLower(a), ' ', '') = $name)) "
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


def neighbor_total(session: Any, kb_id: str, gv: str, entity_id: str, *, types: list[str] | None, direction: str) -> int | None:
    """Number of relations under the same filter (not bounded by limit): tells the caller whether what it got
    is everything. Returns None when it cannot be counted, never 0."""
    rows = session.run(
        """
        MATCH (e:Entity {kb_id: $kb, graph_version: $gv, id: $id})-[r:RELATED_TO]-(n:Entity)
        WHERE coalesce(r.boilerplate, false) = false AND coalesce(n.reference, false) = false
          AND ($types = [] OR r.type IN $types)
        WITH r, (startNode(r).id = $id) AS outgoing
        WHERE $direction = 'both' OR ($direction = 'out' AND outgoing) OR ($direction = 'in' AND NOT outgoing)
        RETURN count(r) AS total
        """, kb=kb_id, gv=gv, id=str(entity_id), types=[str(t) for t in (types or [])], direction=direction).data()
    try:
        return int(rows[0]["total"])
    except (IndexError, KeyError, TypeError, ValueError):
        return None


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


def point_payloads(q: Any, collection: str, point_ids: Iterable[Any]) -> dict[str, dict[str, Any]]:
    """Payloads of source chunks in the main collection (locators and active state); points the main collection
    does not have are absent from the result."""
    ids = sorted({str(p) for p in point_ids if p})
    meta: dict[str, dict[str, Any]] = {}
    for start in range(0, len(ids), 256):
        for rec in q.retrieve(collection_name=collection, ids=ids[start:start + 256], with_payload=EVIDENCE_PAYLOAD_KEYS, with_vectors=False):
            meta[str(rec.id)] = dict(rec.payload or {})
    return meta


def evidence_entry(ev: dict[str, Any], payload: dict[str, Any] | None) -> dict[str, Any]:
    """Complete one evidence chunk with the locators /context needs (doc_id / chunk index / version), the
    position string for people, and whether it is still active; a point the main collection does not have is
    marked active=False."""
    if payload is None:
        ev["active"] = False
        return ev
    ev["active"] = bool(payload.get("is_active", True))
    for key in ("doc_id", "rel_path", "filename", "chunk_index", "content_version", "page_idx"):
        if ev.get(key) in (None, "") and payload.get(key) not in (None, ""):
            ev[key] = payload.get(key)
    ev["position"] = position(payload) or None
    ev["place"] = place(payload) or None
    return ev


def backfill_evidence(q: Any, collection: str, rows: list[dict[str, Any]]) -> None:
    """Chunk snapshot nodes carry no doc_id / chunk index / version; fill them in from the main
    collection by point_id so the evidence can be fed straight to /context; deactivated points are
    marked."""
    ids = [ev.get("point_id") for r in rows for ev in r.get("evidence") or [] if ev.get("point_id")]
    if not ids or q is None:
        return
    meta = point_payloads(q, collection, ids)
    for r in rows:
        for ev in r.get("evidence") or []:
            evidence_entry(ev, meta.get(str(ev.get("point_id"))))


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
            total = neighbor_total(session, source.kb_id, gv, str(center["id"]), types=types, direction=direction)
            attach_evidence(session, source.kb_id, gv, rows)
            center["docs"] = entity_docs(session, source.kb_id, gv, str(center["id"]))
    finally:
        if own_driver:
            driver.close()
    backfill_evidence(q, source.collection, rows)
    return {"kb_id": source.kb_id, "kb_name": kb_name(source), "graph_version": gv, "found": True, "entity": center, "matches": matches[1:],
            "candidates": [], "neighbors": rows, "count": len(rows), "total": total, "has_more": total is not None and total > len(rows),
            "direction": direction, "types": list(types or [])}


def _lowered(values: Iterable[Any] | None) -> list[str]:
    return sorted({str(v).strip().lower() for v in (values or []) if str(v).strip()})


def list_entities(settings: Settings, source: KBSource, *, types: list[str] | None = None, parent_types: list[str] | None = None,
                  name: str | None = None, limit: int = 50, offset: int = 0, driver: Any = None,
                  timeout: float | None = None) -> dict[str, Any]:
    """List the entities of the current graph version by type / upper class / name, with a total and paging,
    in descending pagerank order (for "all of them" questions). The first page (offset=0) also carries the
    number of entities per type, so the caller sees which types this knowledge base has and how large they are.
    Every entity carries the few documents that mention it most (docs) and the number of documents (doc_count),
    so a listed item can be cited directly.
    types / parent_types are case-insensitive, each a union, intersected with each other; name is a piece of
    text contained in the title or an alias.
    What is listed is what the graph registered (the result of model extraction), not a complete inventory of
    the documents; entities found only on boilerplate pages and reference numbers are left out."""
    from kb_pipeline.graph.neo4j_import import active_neo4j_graph_version, neo4j_driver
    from kb_pipeline.graph.recall import TimedSession

    limit = max(1, min(LIST_LIMIT_MAX, int(limit)))
    offset = max(0, int(offset))
    filters = {"types": _lowered(types), "parent_types": _lowered(parent_types), "name": str(name or "").strip().lower()}
    where = (LIVE_ENTITY + " AND ($types = [] OR toLower(e.type) IN $types)"
             " AND ($parents = [] OR toLower(coalesce(e.parent_type, '')) IN $parents OR toLower(coalesce(e.upper, '')) IN $parents)"
             " AND ($name = '' OR toLower(e.title) CONTAINS $name OR any(a IN coalesce(e.aliases, []) WHERE toLower(a) CONTAINS $name))")
    own_driver = driver is None
    if own_driver:
        driver = neo4j_driver(settings)
    try:
        gv = active_neo4j_graph_version(driver, source.kb_id)
        if not gv:
            raise KeyError(f"{source.kb_id} has no active graph")
        params = {"kb": source.kb_id, "gv": gv, "types": filters["types"], "parents": filters["parent_types"], "name": filters["name"]}
        with driver.session() as session:
            if timeout:
                session = TimedSession(session, timeout)
            head = "MATCH (e:Entity {kb_id: $kb, graph_version: $gv}) WHERE "
            total = int((session.run(head + where + " RETURN count(e) AS total", **params).data() or [{}])[0].get("total") or 0)
            rows = session.run(
                head + where + " RETURN e.id AS id, e.title AS title, e.type AS type, e.parent_type AS parent_type, e.scope AS scope, "
                "e.description AS description, e.pagerank AS pagerank, e.degree AS degree, e.aliases AS aliases "
                "ORDER BY coalesce(e.pagerank, 0) DESC, e.id SKIP $offset LIMIT $limit", offset=offset, limit=limit, **params).data()
            type_rows = None
            if offset == 0:
                type_rows = session.run(
                    head + LIVE_ENTITY + " RETURN e.type AS etype, coalesce(e.parent_type, e.upper) AS parent, count(e) AS n "
                    "ORDER BY n DESC, etype LIMIT 100", kb=source.kb_id, gv=gv).data()
            # which documents each entity comes from: a listed item can be cited directly, without one more query
            # per item just to find its source
            doc_rows = session.run(
                "UNWIND $ids AS eid "
                "MATCH (e:Entity {kb_id: $kb, graph_version: $gv, id: eid})-[:MENTIONED_IN]->(c:QdrantChunkSnapshot) "
                "WHERE c.rel_path IS NOT NULL "
                "WITH eid, c.rel_path AS rel_path, count(c) AS mentions ORDER BY mentions DESC, rel_path "
                "RETURN eid, collect(rel_path)[..$top] AS docs, count(rel_path) AS doc_count",
                ids=[str(r.get("id")) for r in rows], kb=source.kb_id, gv=gv, top=LIST_DOCS_PER_ENTITY).data() if rows else []
    finally:
        if own_driver:
            driver.close()
    docs_of = {str(d.get("eid")): d for d in doc_rows}
    entities = []
    for r in rows:
        row = _entity_row(r)
        row["description"] = str(row.get("description") or "")[:LIST_DESCRIPTION_CHARS] or None
        found = docs_of.get(str(row.get("id"))) or {}
        row["docs"] = [str(d) for d in (found.get("docs") or [])]
        row["doc_count"] = int(found.get("doc_count") or 0)
        entities.append(row)
    out = {"kb_id": source.kb_id, "kb_name": kb_name(source), "graph_version": gv, "filters": filters,
           "total": total, "offset": offset, "limit": limit, "count": len(entities), "has_more": offset + len(entities) < total,
           "entities": entities}
    if type_rows is not None:
        out["types"] = [{"type": r.get("etype"), "parent_type": r.get("parent"), "count": int(r.get("n") or 0)} for r in type_rows]
    return out


def list_facts(settings: Settings, source: KBSource, *, subject: str | None = None, subject_id: str | None = None,
               prop: str | None = None, match: str = "auto", limit: int = 50, offset: int = 0,
               fields: tuple[str, ...] | None = None, q: Any = None, driver: Any = None, timeout: float | None = None) -> dict[str, Any]:
    """List the qualified facts of the current graph version by subject / property, with a total and paging
    ("every parameter of this subject", "this property across subjects").
    The subject is found by id or name (title / alias) and its facts are taken along the HAS_SPEC edges; with
    several entities of the same name the one with the most relations is used and the rest are listed in
    matches. The property is matched against the property name / symbol / canonical concept name: match=auto
    tries exact first and falls back to containment only when nothing matched; every spelling under the matched
    concept keys comes back with it (the ones concept normalisation folded into one, such as "TC" and "total
    cholesterol result").
    Fact rows are taken from the payloads of the fact collection, the same projection as the specs of /search,
    so hint strings, series, conflicts and source verification follow the same rules; every row also carries
    the locators of its evidence chunks (ready for /context). With only a subject, the first page also lists the
    properties this subject has."""
    from kb_pipeline.graph.neo4j_import import active_neo4j_graph_version, neo4j_driver
    from kb_pipeline.graph.recall import TimedSession, spec_result_row
    from kb_pipeline.graph.vectors import point_id_for
    from kb_pipeline.vector.qdrant import graph_collection_alias

    from .evidence import numbered, spec_rows

    limit = max(1, min(LIST_LIMIT_MAX, int(limit)))
    offset = max(0, int(offset))
    needle = str(prop or "").strip().lower()
    match = match if match in ("auto", "exact", "contains") else "auto"
    base = {"kb_id": source.kb_id, "kb_name": kb_name(source)}
    own_driver = driver is None
    if own_driver:
        driver = neo4j_driver(settings)
    try:
        gv = active_neo4j_graph_version(driver, source.kb_id)
        if not gv:
            raise KeyError(f"{source.kb_id} has no active graph")
        base["graph_version"] = gv
        with driver.session() as session:
            if timeout:
                session = TimedSession(session, timeout)
            center: dict[str, Any] | None = None
            matches: list[dict[str, Any]] = []
            if subject or subject_id:
                matches = resolve_entity(session, source.kb_id, gv, entity_id=subject_id, name=subject)
                if not matches:
                    cands = dense_candidates(settings, source, subject or "", q=q) if subject else []
                    return {**base, "found": False, "subject": None, "matches": [], "candidates": cands, "total": 0, "count": 0, "facts": [],
                            "note": "no entity with that title or alias; pick one of the candidates by id"}
                matches.sort(key=lambda m: (-int(m.get("degree") or 0), -float(m.get("pagerank") or 0.0)))
                center = matches[0]
            scope = ("MATCH (e:Entity {kb_id: $kb, graph_version: $gv, id: $sid})-[:HAS_SPEC]->(f:Spec)" if center is not None
                     else "MATCH (f:Spec {kb_id: $kb, graph_version: $gv})")
            params: dict[str, Any] = {"kb": source.kb_id, "gv": gv, "p": needle, "keys": []}
            if center is not None:
                params["sid"] = str(center["id"])
            where, matched, keys = "true", None, []
            if needle:
                for mode in (("exact", "contains") if match == "auto" else (match,)):
                    found_keys = session.run(f"{scope} WHERE {FACT_MATCH[mode]} RETURN DISTINCT f.concept_key AS key LIMIT $top",
                                             top=CONCEPT_KEYS_MAX, **params).data()
                    if found_keys:
                        matched, keys = mode, [str(r["key"]) for r in found_keys if r.get("key")]
                        break
                params["keys"] = keys
                # every spelling under the matched concept keys, plus the facts the text matches directly (facts of
                # older graphs without concept keys can only be reached by the latter)
                where = f"(f.concept_key IN $keys OR {FACT_MATCH[matched]})" if matched else "false"
            total = int((session.run(f"{scope} WHERE {where} RETURN count(f) AS total", **params).data() or [{}])[0].get("total") or 0)
            order = "coalesce(f.concept, f.property, ''), coalesce(f.valid_from, ''), coalesce(f.series_index, 0), f.id"
            if center is None:
                order = "coalesce(f.subject, ''), " + order
            picked = [dict(r.get("row") or {}) for r in session.run(
                f"{scope} WHERE {where} RETURN f {{" + ", ".join(f".{k}" for k in _FACT_BASIC) + f"}} AS row "
                f"ORDER BY {order} SKIP $offset LIMIT $limit", offset=offset, limit=limit, **params).data()] if total else []
            properties = None
            if center is not None and offset == 0:
                properties = session.run(
                    f"{scope} RETURN coalesce(f.concept, f.property) AS concept, f.concept_key AS concept_key, count(f) AS n "
                    "ORDER BY n DESC, concept LIMIT $top", top=PROPERTY_SUMMARY_LIMIT, **params).data()
    finally:
        if own_driver:
            driver.close()
    degraded: list[str] = []
    payloads: dict[str, dict[str, Any]] = {}
    fact_ids = [str(r["id"]) for r in picked]
    if q is not None and fact_ids:
        by_point = {point_id_for(gv, fid): fid for fid in fact_ids}
        try:
            ids = list(by_point)
            for start in range(0, len(ids), 256):
                for rec in q.retrieve(collection_name=graph_collection_alias(source.collection, "spec"), ids=ids[start:start + 256],
                                      with_payload=True, with_vectors=False):
                    pl = dict(rec.payload or {})
                    if str(pl.get("graph_version") or gv) == gv and str(rec.id) in by_point:
                        payloads[by_point[str(rec.id)]] = pl
        except Exception as exc:
            degraded.append(f"spec_payload: {type(exc).__name__}")
    raw = [spec_result_row(payloads[fid]) if fid in payloads else {**{k: r.get(k) for k in _FACT_BASIC}, "point_ids": []}
           for fid, r in zip(fact_ids, picked)]
    missing = sum(1 for fid in fact_ids if fid not in payloads)
    if missing and not degraded:
        # the alias of the fact collection points at another version (the few seconds of a publish switch): only
        # the basic graph-database fields are available
        degraded.append(f"spec_payload_missing: {missing}")
    cited = {fid: [str(p) for p in (row.get("point_ids") or [])[:EVIDENCE_PER_FACT]] for fid, row in zip(fact_ids, raw)}
    meta: dict[str, dict[str, Any]] | None = None
    if q is not None and any(cited.values()):
        try:
            meta = point_payloads(q, source.collection, [p for pids in cited.values() for p in pids])
        except Exception as exc:
            degraded.append(f"point_meta: {type(exc).__name__}")
    # conclusions only from source points that were checked: when the main collection gave nothing back the state
    # is "unknown", not "no longer valid"
    point_active = None if meta is None else {p: bool(meta[p].get("is_active", True)) if p in meta else False
                                              for pids in cited.values() for p in pids}
    shaped = spec_rows(raw, source_ns={}, limit_hints=len(raw), point_active=point_active)
    for fid, row in zip(fact_ids, shaped):
        row["evidence"] = [evidence_entry({"point_id": p}, meta.get(p)) if meta is not None else {"point_id": p} for p in cited[fid]]
    facts = numbered(shaped, tuple(fields or ()) + ("evidence",))
    for i, row in enumerate(facts):
        row["n"] = offset + i + 1
    out = {**base, "found": True, "subject": center, "matches": matches[1:], "candidates": [],
           "property": {"query": str(prop).strip(), "matched": matched, "concepts": len(keys)} if needle else None,
           "total": total, "offset": offset, "limit": limit, "count": len(facts), "has_more": offset + len(facts) < total,
           "facts": facts}
    if properties is not None:
        out["properties"] = [{"concept": r.get("concept"), "concept_key": r.get("concept_key"), "count": int(r.get("n") or 0)}
                             for r in properties]
    if degraded:
        out["degraded"] = degraded
    return out
