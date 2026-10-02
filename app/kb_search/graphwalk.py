"""Structured graph lookups: the agent walks the graph and lists things step by step itself (multi-hop
reasoning lives on the agent side; blindly expanding three to five hops showed no gain at all in evaluation).
All three endpoints only read the current graph version and generate nothing:
- neighbours: given an entity (name or id), its one-hop relations: predicate, direction, weight, the entity
  at the other end and evidence chunks (ready for /context); every relation carries an excerpt of the original
  text that names the far end, and the entity's relation counts per predicate come along; the name is first
  matched exactly by title / alias, and if nothing matches a few candidates are offered by vector for the agent
  to choose from;
- entities: entities by type / upper class / name, with a total and paging ("all of them" questions);
- facts: qualified facts by subject / property, with a total and paging, rows shaped like the specs of /search.
/search additionally carries the one-hop neighbourhood of its subjects (neighborhoods): for each entity the question
names, a few of its strongest relations, so the agent can decide where to walk next.
What is listed is what the graph registered (the result of model extraction), not a complete inventory of the
documents."""
from __future__ import annotations

import math
import threading
from typing import Any, Iterable

from kb_pipeline.config import Settings
from kb_pipeline.models import KBSource

from .text import place, position

ENTITY_FIELDS = ("id", "title", "type", "parent_type", "scope", "description", "pagerank", "degree", "aliases")
EVIDENCE_PER_RELATION = 3
EVIDENCE_PER_FACT = 3
# Every relation carries an excerpt of the original text: only the few sentences of the evidence that name the far
# end, so the caller need not read a chunk back to check one relation; the full text still comes from /context.
# Measured on four entities on 2026-10-02 (20 relations per hop): the full evidence text is 19,000-30,000 characters
# per hop, 200-character excerpts are about 3,600-3,900, the same order as the model-written descriptions already
# returned (2,200-5,900); for 19-20 of 20 relations the far end's name is found literally in the evidence
EXCERPT_CHARS = 200
EXCERPT_PAYLOAD_KEYS = ["text", "block_type", "visual_ref"]
EXCERPT_BOUNDS = "\n。；;！？!?"
_MATCH_RANK = {"both": 0, "other": 1, "center": 2, "none": 3}
NEIGHBORHOOD_SIZE = 8               # subject neighbourhoods of /search: how many of its strongest relations each subject lists
NEIGHBORHOOD_SCAN = 4               # same: to give each far end one line only, take this many times NEIGHBORHOOD_SIZE and then dedupe
NEIGHBORHOOD_KINDS = 8              # same: at most how many predicates the per-predicate counts list
NAME_MIN_CHARS = 2
NAME_MAX_CHARS = 48                 # longest name looked for in a question (99% of the names in a graph are under 35 characters; longer ones never appear whole in a question)
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
               n.id AS other_id, n.title AS other_title, n.type AS other_type, n.scope AS other_scope, n.pagerank AS other_pagerank,
               n.degree AS other_degree, n.aliases AS other_aliases
        ORDER BY coalesce(r.weight, 0) DESC, coalesce(n.pagerank, 0) DESC
        LIMIT $limit
        """, kb=kb_id, gv=gv, id=str(entity_id), types=[str(t) for t in (types or [])], direction=direction, limit=int(limit)).data()
    out = []
    for r in rows:
        out.append({
            "relation_id": r.get("relation_id"), "type": r.get("type"), "direction": "out" if r.get("outgoing") else "in",
            "directed": bool(r.get("directed")), "weight": r.get("weight"), "npmi": r.get("npmi"), "cooccur": r.get("cooccur"),
            "description": r.get("description"), "type_violation": bool(r.get("type_violation")),
            # degree: how many relations the far end has itself; the caller judges by it whether walking on that way is worthwhile
            "other": {"id": r.get("other_id"), "title": r.get("other_title"), "type": r.get("other_type"), "scope": r.get("other_scope"),
                      "pagerank": r.get("other_pagerank"), "degree": r.get("other_degree")},
            "evidence": [],
            "_names": [r.get("other_title"), *(r.get("other_aliases") or [])],       # to locate the far end in the original text when cutting the excerpt; not returned
        })
    return out


def neighbor_kinds(session: Any, kb_id: str, gv: str, entity_id: str, *, direction: str) -> list[dict[str, Any]]:
    """How many relations of each kind (predicate) this entity has, bounded neither by limit nor by the predicate
    filter: an entity often has more than a thousand relations while one call returns the few dozen strongest, so
    the caller learns from this which other kinds it can ask for by predicate; whether what it got is everything
    is computed from it as well."""
    rows = session.run(
        """
        MATCH (e:Entity {kb_id: $kb, graph_version: $gv, id: $id})-[r:RELATED_TO]-(n:Entity)
        WHERE coalesce(r.boilerplate, false) = false AND coalesce(n.reference, false) = false
        WITH r, (startNode(r).id = $id) AS outgoing
        WHERE $direction = 'both' OR ($direction = 'out' AND outgoing) OR ($direction = 'in' AND NOT outgoing)
        RETURN r.type AS type, count(r) AS n ORDER BY n DESC, type
        """, kb=kb_id, gv=gv, id=str(entity_id), direction=direction).data()
    return [{"type": r.get("type"), "count": int(r.get("n") or 0)} for r in rows]


def fact_count(session: Any, kb_id: str, gv: str, entity_id: str) -> int:
    """How many qualified facts are registered under this entity (tells the caller whether the facts endpoint has
    anything for it)."""
    rows = session.run("MATCH (e:Entity {kb_id: $kb, graph_version: $gv, id: $id})-[:HAS_SPEC]->(f:Spec) RETURN count(f) AS n",
                       kb=kb_id, gv=gv, id=str(entity_id)).data()
    return int((rows or [{}])[0].get("n") or 0)


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


def point_payloads(q: Any, collection: str, point_ids: Iterable[Any], *, keys: list[str] | None = None,
                   timeout: float | None = None) -> dict[str, dict[str, Any]]:
    """Payloads of source chunks in the main collection (locators and active state; keys names other payload keys
    to fetch instead); points the main collection does not have are absent from the result."""
    ids = sorted({str(p) for p in point_ids if p})
    meta: dict[str, dict[str, Any]] = {}
    extra = {"timeout": max(1, math.ceil(float(timeout)))} if timeout else {}      # Qdrant takes whole seconds only
    for start in range(0, len(ids), 256):
        for rec in q.retrieve(collection_name=collection, ids=ids[start:start + 256], with_payload=keys or EVIDENCE_PAYLOAD_KEYS,
                              with_vectors=False, **extra):
            meta[str(rec.id)] = dict(rec.payload or {})
    return meta


def _ascii_word(ch: str) -> bool:
    return ch.isascii() and ch.isalnum()


def _name_spans(low: str, name: Any) -> list[tuple[int, int]]:
    """Where a name occurs in the (lower-cased) text, as start / end offsets. A name that starts or ends with a
    letter or digit must not sit inside a longer word ("AI" must not match in "FAIL"); for a name written with
    spaces, like "ZK 200", the spelling without them counts too."""
    key = " ".join(str(name or "").lower().split())
    out: list[tuple[int, int]] = []
    for variant in dict.fromkeys((key, key.replace(" ", ""))):
        if len(variant) < NAME_MIN_CHARS:
            continue
        start = low.find(variant)
        while start >= 0 and len(out) < 16:
            end = start + len(variant)
            inside = (_ascii_word(variant[0]) and start > 0 and _ascii_word(low[start - 1])) \
                or (_ascii_word(variant[-1]) and end < len(low) and _ascii_word(low[end]))
            if not inside:
                out.append((start, end))
            start = low.find(variant, start + 1)
    return out


def excerpt_of(text: Any, names: Iterable[Any], anchors: Iterable[Any], *, limit: int = EXCERPT_CHARS) -> tuple[str, str]:
    """Cut the few sentences that name the far end out of one evidence chunk. names: the far end's title and
    aliases; anchors: the centre entity's title and aliases. Returns (excerpt, what was found): both = the names of
    both ends are in this chunk, other = only the far end, center = only the centre entity, none = neither (the
    excerpt is the head of the chunk). The search is literal, no model is involved; when both ends are present the
    occurrence of the far end closest to the centre entity is taken. The window starts at the beginning of a
    sentence (or line) where it can and ends at the end of one; a cut side gets an ellipsis."""
    raw = str(text or "")
    low = raw.lower()
    if len(low) != len(raw):          # lower-casing lengthens a few characters and the offsets no longer line up: search the text as it is
        low = raw
    other = sorted({s for n in names for s in _name_spans(low, n)})
    center = sorted({s for n in anchors for s in _name_spans(low, n)})
    if other:
        hit = min(other, key=lambda s: min(abs(s[0] - c[0]) for c in center)) if center else other[0]
        match = "both" if center else "other"
    elif center:
        hit, match = center[0], "center"
    else:
        hit, match = (0, 0), "none"
    lo = max(0, hit[0] - limit // 3)
    cut = max(raw.rfind(b, lo, hit[0]) for b in EXCERPT_BOUNDS)
    start = cut + 1 if cut >= 0 else lo
    end = min(len(raw), start + limit)
    if end < len(raw):
        tail = max(raw.rfind(b, max(start + limit // 2, hit[1]), end) for b in EXCERPT_BOUNDS)
        if tail >= 0:
            end = tail + 1
    body = " ".join(raw[start:end].split())
    if start > 0 and raw[start - 1] not in EXCERPT_BOUNDS:
        body = "…" + body
    if end < len(raw) and raw[end - 1] not in EXCERPT_BOUNDS:
        body += "…"
    return body, match


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


def _squash(text: Any) -> str:
    return "".join(str(text or "").casefold().split())


def locate_facts(q: Any, collections: dict[str, str], specs: list[dict[str, Any]], sources: list[dict[str, Any]], *,
                 timeout: float | None = None) -> list[str]:
    """Give every fact of /search the chunks it rests on (EVIDENCE_PER_FACT at most): the locators /context needs,
    the place string and whether the chunk is still active, in the row shape of the facts endpoint's evidence. A
    fact comes from one extraction unit that may span several chunks, and extraction does not record which chunk
    holds the value: when the value's wording appears in the text of exactly one of them (the property name is
    tried when the value does not settle it), that chunk moves to the front and is marked located, so the caller's
    citation and read-back land on it; otherwise the order stays and nothing is marked. Chunks already among the
    sources reuse their rows, the rest are fetched from the main collection; inactive or missing chunks go last.
    When one knowledge base cannot be read, its chunks keep only their ids (unknown, not marked inactive) and the
    return value carries a degraded note."""
    by_point = {str(s.get("point_id")): s for s in sources if s.get("point_id")}
    cited = [[str(p) for p in (sp.get("point_ids") or [])[:EVIDENCE_PER_FACT]] for sp in specs]
    fetched: dict[str, dict[str, Any]] = {}
    failed: set[str] = set()
    degraded: list[str] = []
    for kb, collection in collections.items():
        want = {p for sp, pids in zip(specs, cited) if sp.get("kb_id") == kb for p in pids if p not in by_point}
        if want:
            try:
                fetched.update(point_payloads(q, collection, want, keys=EVIDENCE_PAYLOAD_KEYS + ["text"], timeout=timeout))
            except Exception as exc:
                failed.add(kb)
                degraded.append(f"{kb}:spec_evidence: {type(exc).__name__}")
    squashed: dict[str, str] = {}

    def body(pid: str) -> str:
        if pid not in squashed:
            squashed[pid] = _squash((by_point.get(pid) or fetched.get(pid) or {}).get("text"))
        return squashed[pid]

    for sp, pids in zip(specs, cited):
        entries: list[dict[str, Any]] = []
        for pid in pids:
            row = by_point.get(pid)
            if row is not None:
                entries.append({"point_id": pid, "active": True, "doc_id": row.get("doc_id"), "rel_path": row.get("rel_path"),
                                "filename": row.get("doc"), "chunk_index": row.get("chunk_index"), "content_version": row.get("content_version"),
                                "page_idx": row.get("page_idx"), "position": row.get("position") or None, "place": row.get("place") or None})
            elif sp.get("kb_id") in failed:
                entries.append({"point_id": pid})
            else:
                entries.append(evidence_entry({"point_id": pid}, fetched.get(pid)))
        live = [i for i, ev in enumerate(entries) if ev.get("active")]
        if len(live) > 1:
            values = [v for v in (_squash(sp.get(k)) for k in ("value", "min", "typ", "max")) if len(v) >= 2]     # a value of one or two characters matches by chance
            held = [i for i in live if any(v in body(pids[i]) for v in values)]
            if len(held) != 1:
                name = _squash(sp.get("property"))
                named = [i for i in (held or live) if len(name) >= 2 and name in body(pids[i])]
                held = named if len(named) == 1 else []
            if held:
                entries[held[0]]["located"] = True
                entries.insert(0, entries.pop(held[0]))
        entries.sort(key=lambda ev: {True: 0, None: 1, False: 2}[ev.get("active")])       # stable: live chunks first, the caller reads back the first one
        if entries:
            sp["evidence"] = entries
    return degraded


def backfill_evidence(q: Any, collection: str, rows: list[dict[str, Any]], *, anchors: list[Any] | None = None) -> None:
    """Chunk snapshot nodes carry no doc_id / chunk index / version; fill them in from the main
    collection by point_id so the evidence can be fed straight to /context; deactivated points are
    marked.
    With anchors (the centre entity's title and aliases) the original text is excerpted along the way: of a
    relation's active evidence chunks one is picked - chunks that name the far end before those that do not, among
    them text chunks before picture chunks (the text of a picture chunk is a description written by the vision
    model, not the document's own words), then whether both ends are named - the excerpt goes on that evidence
    entry (excerpt / excerpt_match) and the entry moves to the front; one taken from a picture chunk is marked
    visual=true."""
    ids = [ev.get("point_id") for r in rows for ev in r.get("evidence") or [] if ev.get("point_id")]
    if not ids or q is None:
        return
    meta = point_payloads(q, collection, ids, keys=EVIDENCE_PAYLOAD_KEYS + EXCERPT_PAYLOAD_KEYS if anchors is not None else None)
    for r in rows:
        best: tuple[tuple[bool, bool, int, bool, int], dict[str, Any], str, str, dict[str, Any]] | None = None
        for i, ev in enumerate(r.get("evidence") or []):
            payload = meta.get(str(ev.get("point_id")))
            evidence_entry(ev, payload)
            if anchors is None or payload is None or not ev.get("active") or not payload.get("text"):
                continue
            body, match = excerpt_of(payload["text"], r.get("_names") or [], anchors)
            named, visual = match in ("both", "other"), bool(payload.get("visual_ref"))
            rank = (not named, visual and named, _MATCH_RANK[match], visual, i)
            if best is None or rank < best[0]:
                best = (rank, ev, body, match, payload)
        if best is not None:
            _, ev, body, match, payload = best
            ev["excerpt"], ev["excerpt_match"] = body, match
            if payload.get("block_type"):
                ev["block_type"] = payload["block_type"]
            if payload.get("visual_ref"):
                ev["visual"] = True
            r["evidence"].sort(key=lambda e: e is not ev)


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
            kinds = neighbor_kinds(session, source.kb_id, gv, str(center["id"]), direction=direction)
            attach_evidence(session, source.kb_id, gv, rows)
            center["docs"] = entity_docs(session, source.kb_id, gv, str(center["id"]))
            center["facts"] = fact_count(session, source.kb_id, gv, str(center["id"]))
    finally:
        if own_driver:
            driver.close()
    backfill_evidence(q, source.collection, rows, anchors=[center.get("title"), *(center.get("aliases") or [])])
    for r in rows:
        r.pop("_names", None)
    # Number of relations under the same filter (not bounded by limit): tells the caller whether what it got is everything
    wanted = {str(t) for t in (types or [])}
    total = sum(k["count"] for k in kinds if not wanted or str(k["type"]) in wanted)
    return {"kb_id": source.kb_id, "kb_name": kb_name(source), "graph_version": gv, "found": True, "entity": center, "matches": matches[1:],
            "candidates": [], "neighbors": rows, "count": len(rows), "total": total, "has_more": total > len(rows),
            "predicates": kinds, "direction": direction, "types": list(types or [])}


_names_lock = threading.Lock()
_names: dict[str, dict[str, Any]] = {}      # kb_id -> {"gv", "names": {normalized name: [(entity id, is it the title)]}, "entities": {id: title / type / relation count}}


def _name_key(name: Any) -> str:
    return "".join(str(name or "").casefold().split())


def name_index(session: Any, kb_id: str, gv: str) -> dict[str, Any]:
    """The names of every entity in this knowledge base's current graph version (titles and aliases, case and
    spaces ignored), kept in the process and rebuilt when the graph version changes.
    Finding entities by name in a question means checking every span of the question against every name; asking
    the graph database each time is a scan of the whole knowledge base, whereas after one read it is a few hundred
    dictionary lookups per question. Measured on 2026-10-02: a knowledge base of 49,000 entities takes 0.7 s to
    read and about 9 MB, three smaller ones 0.1-0.2 s. Entities that only appear on boilerplate pages and
    reference-number entities are left out."""
    with _names_lock:
        cur = _names.get(kb_id)
        if cur is not None and cur["gv"] == gv:
            return cur
    rows = session.run(
        "MATCH (e:Entity {kb_id: $kb, graph_version: $gv}) WHERE " + LIVE_ENTITY +
        " RETURN e.id AS id, e.title AS title, e.type AS type, e.aliases AS aliases, e.degree AS degree", kb=kb_id, gv=gv).data()
    names: dict[str, list[tuple[str, bool]]] = {}
    entities: dict[str, dict[str, Any]] = {}
    for r in rows:
        eid = str(r.get("id") or "")
        if not eid:
            continue
        entities[eid] = {"id": eid, "title": r.get("title"), "type": r.get("type"), "degree": int(r.get("degree") or 0)}
        for i, name in enumerate([r.get("title"), *(r.get("aliases") or [])]):
            key = _name_key(name)
            if NAME_MIN_CHARS <= len(key) <= NAME_MAX_CHARS:
                names.setdefault(key, []).append((eid, i == 0))
    built = {"gv": gv, "names": names, "entities": entities}
    with _names_lock:
        _names[kb_id] = built
    return built


def named_entities(question: str, index: dict[str, Any]) -> list[dict[str, Any]]:
    """Entities the question names: a title or alias appears whole in the question (case and spaces ignored; names
    in scripts without word spacing count too).
    A name of letters / digits must not sit inside a longer word; an occurrence wholly covered by a longer name does
    not count (the "Northwind" inside "Northwind Gateway"); of several entities sharing the name (as title or alias)
    the one with the most relations is taken, the same choice the neighbours endpoint makes for a name, so what the
    preview shows is where the next hop will land.
    Each carries _rank: 2 for a name with letters or digits or of four characters and more, 1 for three characters,
    0 for a two-character word without them (a common word can be an entity too; whether such a short name counts is
    for the caller to decide); the result is ordered by _rank, then by relation count."""
    question = str(question or "")
    flat: list[str] = []
    at: list[int] = []                 # where each character of flat sits in the question (word boundaries are checked in the original after whitespace is dropped and case folded)
    for i, ch in enumerate(question):
        if ch.isspace():
            continue
        for c in ch.casefold():
            flat.append(c)
            at.append(i)
    text = "".join(flat)
    names = index["names"]
    spans: list[tuple[int, int]] = []
    for start in range(len(text)):
        if _ascii_word(text[start]) and at[start] > 0 and _ascii_word(question[at[start] - 1]):
            continue
        for end in range(start + NAME_MIN_CHARS, min(len(text), start + NAME_MAX_CHARS) + 1):
            if text[start:end] not in names:
                continue
            last = at[end - 1]
            if _ascii_word(text[end - 1]) and last + 1 < len(question) and _ascii_word(question[last + 1]):
                continue
            spans.append((start, end))
    spans.sort(key=lambda s: (s[0] - s[1], s[0]))
    kept: list[tuple[int, int]] = []
    for s in spans:
        if not any(k[0] <= s[0] and s[1] <= k[1] and k[1] - k[0] > s[1] - s[0] for k in kept):
            kept.append(s)
    entities = index["entities"]
    found: dict[str, dict[str, Any]] = {}
    for start, end in kept:
        key = text[start:end]
        eid, _ = max(names[key], key=lambda m: (entities[m[0]]["degree"], m[1]))
        rank = 2 if len(key) >= 4 or any(_ascii_word(c) for c in key) else len(key) - 2
        if eid not in found or rank > found[eid]["_rank"]:
            found[eid] = {**entities[eid], "_rank": rank, "_key": key}
    return sorted(found.values(), key=lambda e: (-e["_rank"], -e["degree"], str(e.get("title") or "")))


def neighborhoods(settings: Settings, sources: dict[str, KBSource], graph_versions: dict[str, str], question: str,
                  seeds: list[dict[str, Any]], *, limit: int, per_entity: int = NEIGHBORHOOD_SIZE, driver: Any = None,
                  timeout: float | None = None) -> list[dict[str, Any]]:
    """The one-hop neighbourhood of the subjects that /search carries: the caller (the agent) reads it and then
    decides whether to call the neighbours endpoint and walk on, instead of first looking each subject up.
    graph_versions: the chosen knowledge bases that have a graph -> current graph version (in routing order);
    seeds: the entity rows the graph route returned (with kb_id, already in descending score order).
    Subjects are first the entities the question names (named_entities), then, to fill up, the graph route's seeds
    by score - limit in total across the knowledge bases:
    - the seeds are the entities semantically closest to the question, which often are not the subject it asks
      about (measured on a comparison question on 2026-10-02: the five best-scoring seeds were small entities for
      the comparison dimensions, neither of the two products being compared made the top five), so named entities
      come first;
    - a name of two or three characters without letters or digits is hit too easily (a question about "which
      products the knowledge base lists" names entities called "product" and "knowledge base"), so such a short
      name only counts when the entity is also a graph route seed; short names that really are asked about were
      among the seeds in every case measured;
    - of the seeds used to fill up, only the first of those sharing a name within one knowledge base is taken
      (same-named entities of different scopes are often hit together), and one connected to nothing takes no slot.
    Each subject gives: its relation total, the number of facts under it, relation counts per predicate, and its
    per_entity strongest relations (predicate, direction, the far end and the far end's own relation count); a
    far end is listed once, by its strongest relation (two entities often share several relations with different
    predicates, and one far end must not fill the list).
    These are leads only, without evidence or excerpts: to check a relation, call the neighbours endpoint."""
    from kb_pipeline.graph.neo4j_import import neo4j_driver
    from kb_pipeline.graph.recall import TimedSession

    seeded = {(str(e.get("kb_id")), str(e.get("id"))) for e in seeds}
    own_driver = driver is None
    if own_driver:
        driver = neo4j_driver(settings)
    try:
        with driver.session() as session:
            if timeout:
                session = TimedSession(session, timeout)
            indexes = {kb: name_index(session, kb, gv) for kb, gv in graph_versions.items()}
            named = [{**e, "kb_id": kb, "named": True} for kb, index in indexes.items() for e in named_entities(question, index)
                     if e["_rank"] >= 2 or (kb, e["id"]) in seeded]
            order = {kb: i for i, kb in enumerate(graph_versions)}
            named.sort(key=lambda e: (-e["_rank"], order[e["kb_id"]], -e["degree"]))
            subjects = named[:limit]
            taken = {(s["kb_id"], s["id"]) for s in subjects}
            used = {(s["kb_id"], s["_key"]) for s in subjects}
            for seed in seeds:
                if len(subjects) >= limit:
                    break
                kb, eid = str(seed.get("kb_id")), str(seed.get("id") or "")
                entity = (indexes.get(kb) or {"entities": {}})["entities"].get(eid)
                if entity is None or (kb, eid) in taken or (kb, _name_key(entity.get("title"))) in used:
                    continue
                taken.add((kb, eid))
                used.add((kb, _name_key(entity.get("title"))))
                subjects.append({**entity, "kb_id": kb, "named": False})
            rel_rows: dict[tuple[str, str], dict[str, Any]] = {}
            facts: dict[tuple[str, str], int] = {}
            for kb in graph_versions:
                ids = [s["id"] for s in subjects if s["kb_id"] == kb]
                if not ids:
                    continue
                for r in session.run(
                        """
                        UNWIND $ids AS eid
                        MATCH (e:Entity {kb_id: $kb, graph_version: $gv, id: eid})-[r:RELATED_TO]-(n:Entity)
                        WHERE coalesce(r.boilerplate, false) = false AND coalesce(n.reference, false) = false
                        WITH eid, r, n, (startNode(r).id = eid) AS outgoing ORDER BY coalesce(r.weight, 0) DESC, coalesce(n.pagerank, 0) DESC
                        RETURN eid, count(r) AS total, collect(r.type) AS kinds,
                               collect({relation_id: r.id, type: r.type, outgoing: outgoing, directed: coalesce(r.directed, false), weight: r.weight,
                                        other_id: n.id, other_title: n.title, other_type: n.type, other_degree: n.degree})[0..$k] AS top
                        """, ids=ids, kb=kb, gv=graph_versions[kb], k=int(per_entity) * NEIGHBORHOOD_SCAN).data():
                    rel_rows[(kb, str(r.get("eid")))] = r
                for r in session.run(
                        "UNWIND $ids AS eid MATCH (e:Entity {kb_id: $kb, graph_version: $gv, id: eid}) "
                        "RETURN eid, size([(e)-[:HAS_SPEC]->(f:Spec) | 1]) AS facts", ids=ids, kb=kb, gv=graph_versions[kb]).data():
                    facts[(kb, str(r.get("eid")))] = int(r.get("facts") or 0)
    finally:
        if own_driver:
            driver.close()
    out: list[dict[str, Any]] = []
    for s in subjects:
        key = (s["kb_id"], s["id"])
        row = rel_rows.get(key) or {}
        counts: dict[str, int] = {}
        for kind in row.get("kinds") or []:
            counts[str(kind)] = counts.get(str(kind), 0) + 1
        total = int(row.get("total") or 0)
        if not s["named"] and not total and not facts.get(key):
            continue
        top: dict[str, dict[str, Any]] = {}
        for t in row.get("top") or []:
            if len(top) < per_entity:
                top.setdefault(str(t.get("other_id")), t)
        out.append({"kb_id": s["kb_id"], "id": s["id"], "title": s.get("title"), "type": s.get("type"), "named": bool(s["named"]),
                    "relations": total, "facts": facts.get(key, 0),
                    "predicates": [{"type": k, "count": n} for k, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:NEIGHBORHOOD_KINDS]],
                    "neighbors": [{"relation_id": t.get("relation_id"), "type": t.get("type"), "direction": "out" if t.get("outgoing") else "in",
                                   "directed": bool(t.get("directed")), "weight": t.get("weight"),
                                   "other": {"id": t.get("other_id"), "title": t.get("other_title"), "type": t.get("other_type"),
                                             "degree": t.get("other_degree")}} for t in top.values()]})
    return out


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
    facts = numbered(shaped, tuple(dict.fromkeys(tuple(fields or ()) + ("evidence",))))
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
