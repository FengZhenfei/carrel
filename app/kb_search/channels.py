"""Candidate fetching for the three recall channels: vector (the text vector of the main Qdrant
collection), keyword (OpenSearch) and graph (kb_pipeline.graph.recall). Every channel returns
candidates of the same shape [{point_id, score, payload?, ...}] and does no fusion; a channel's failure
is recorded in degraded by the caller."""
from __future__ import annotations

import base64
import io
import math
import re
import threading
from typing import Any

import requests
from opensearchpy import OpenSearch
from qdrant_client import models

from kb_pipeline import search_fts
from kb_pipeline.config import Settings
from kb_pipeline.embedding.client import EmbeddingClient
from kb_pipeline.graph.recall import identifier_variants, lexical_tokens
from kb_pipeline.models import KBSource

TEXT_VECTOR = "text"
VISUAL_VECTOR = "visual"
ACTIVE_FILTER = models.Filter(must=[models.FieldCondition(key="is_active", match=models.MatchValue(value=True))])
VISUAL_MAX_PIXELS = 1_000_000
BM25_SOURCE = ["chunk_uid", "kb_id", "doc_id", "content_version", "chunk_index", "rel_path", "filename", "page_idx", "block_type"]
IDENTIFIER_BOOST = 3.0


def _secs(timeout: float | None) -> int | None:
    """Qdrant takes timeouts in whole seconds only: round up, at least 1 second (a fraction of a second
    computed from the remaining budget must not be truncated to 0)."""
    return max(1, math.ceil(float(timeout))) if timeout else None


_os_clients: dict[str, OpenSearch] = {}


def os_client(url: str) -> OpenSearch:
    """The search side's own OpenSearch client: no retry on timeout. The pipeline's client retries twice
    more after a timeout, so a keyword channel job the caller has already abandoned could drag on for up
    to 90 seconds, holding on to the shared thread pool all along. One retry is kept for connection
    errors: after an OpenSearch restart an old pooled connection breaks on first use, and reconnecting
    right away fixes it."""
    cached = _os_clients.get(url)
    if cached is None:
        cached = _os_clients[url] = OpenSearch(hosts=[url], timeout=30, max_retries=1, retry_on_timeout=False)
    return cached


def query_text(question: str, instruction: str | None) -> str:
    """Query-side embedding text: with a task instruction, the Qwen3-Embedding model card form
    "Instruct: instruction\nQuery: question"; otherwise the question verbatim. The "document name >
    section path" carried by corpus vectors is a content prefix, not a task instruction; the two are
    different things."""
    instruction = str(instruction or "").strip()
    return f"Instruct: {instruction}\nQuery: {question}" if instruction else str(question)


def embed_question(settings: Settings, question: str, *, timeout: float = 10.0, instruction: str | None = None) -> list[float]:
    """Question vector: no prefix by default (the pipeline's "document name > section path" prefix goes
    into corpus vectors only, Q02 / section 5 R9); a configured query instruction is added in the model
    card form. Uses the pipeline client's connection pool directly, but the request timeout follows the
    query's scale (seconds) rather than the 120 s used when building (Codex S06)."""
    client = EmbeddingClient(base_url=settings.embedding_base_url, api_key=settings.embedding_api_key,
                             model_id=settings.embedding_model_id, dim=settings.embedding_dim, batch_size=1, retry=1, sleep_seconds=0)
    resp = client.client.embeddings.create(model=settings.embedding_model_id, input=[query_text(question, instruction)],
                                           dimensions=int(settings.embedding_dim), timeout=float(timeout))
    vec = [float(x) for x in resp.data[0].embedding]
    if len(vec) != int(settings.embedding_dim):
        raise RuntimeError(f"embedding dim mismatch: {len(vec)}")
    return vec


def question_identifiers(question: str) -> list[str]:
    """Tokens in the question worth exact matching (model numbers, parameter symbols, two-letter
    all-caps, filenames): reuses the graph channel's lexical seed rules (Q03)."""
    out: list[str] = []
    for tok in lexical_tokens(question):
        if tok not in out:
            out.append(tok)
    return out


HINT_KEYS = ("doc_ids", "rel_paths", "content_version", "block_types")


def parse_hints(hints: dict[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    """Caller hints: only a few generic, deterministically applicable keys are accepted -- doc_ids /
    rel_paths / content_version are hard filters, block_types is a soft preference; keys of the wrong
    type or unknown keys are ignored and listed in ignored, without an error. Things verified against
    the source text, such as subject or time, are not hard-filtered."""
    used: dict[str, Any] = {}
    ignored: list[str] = []
    for key, val in (hints or {}).items():
        if key in ("doc_ids", "rel_paths", "block_types"):
            items = [str(x).strip() for x in (val if isinstance(val, (list, tuple)) else [val]) if str(x).strip()]
            if items:
                used[key] = items
            else:
                ignored.append(key)
        elif key == "content_version":
            if isinstance(val, str) and val.strip():
                used[key] = val.strip()
            else:
                ignored.append(key)
        else:
            ignored.append(str(key))
    return used, ignored


def hint_filter(used: dict[str, Any]) -> models.Filter:
    """Qdrant filter: active points + the document / path / version constraints from the hints."""
    must: list[Any] = [models.FieldCondition(key="is_active", match=models.MatchValue(value=True))]
    if used.get("doc_ids"):
        must.append(models.FieldCondition(key="doc_id", match=models.MatchAny(any=list(used["doc_ids"]))))
    if used.get("rel_paths"):
        must.append(models.FieldCondition(key="rel_path", match=models.MatchAny(any=list(used["rel_paths"]))))
    if used.get("content_version"):
        must.append(models.FieldCondition(key="content_version", match=models.MatchValue(value=str(used["content_version"]))))
    return models.Filter(must=must)


def hint_os_filters(used: dict[str, Any]) -> list[dict[str, Any]]:
    """OpenSearch filter clauses (doc_id / rel_path / content_version are all keyword fields in the index)."""
    out: list[dict[str, Any]] = []
    if used.get("doc_ids"):
        out.append({"terms": {"doc_id": list(used["doc_ids"])}})
    if used.get("rel_paths"):
        out.append({"terms": {"rel_path": list(used["rel_paths"])}})
    if used.get("content_version"):
        out.append({"term": {"content_version": str(used["content_version"])}})
    return out


def hint_allows(used: dict[str, Any], payload: dict[str, Any]) -> bool:
    """Candidates that bypassed the filter (graph channel etc.) are screened by the same constraints
    once their payload is filled in."""
    if used.get("doc_ids") and str(payload.get("doc_id")) not in set(used["doc_ids"]):
        return False
    if used.get("rel_paths") and str(payload.get("rel_path")) not in set(used["rel_paths"]):
        return False
    if used.get("content_version") and str(payload.get("content_version")) != str(used["content_version"]):
        return False
    return True


def vector_channel(q: Any, collection: str, vector: list[float], *, limit: int, timeout: float | None = None,
                   query_filter: Any = None) -> list[dict[str, Any]]:
    """Dense recall from the main collection: the named vector text, active points only (Q02: an
    unnamed vector on a named-vector collection is a straight 400, so using is always passed here)."""
    res = q.query_points(collection_name=collection, query=vector, using=TEXT_VECTOR, query_filter=query_filter or ACTIVE_FILTER,
                         limit=int(limit), with_payload=True, timeout=_secs(timeout))
    return [{"point_id": str(p.id), "score": float(p.score), "payload": dict(p.payload or {})} for p in res.points]


def bm25_query(question: str, identifiers: list[str], filters: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """OpenSearch query body: whole-sentence multi_match + filename / path ngram, plus an exact phrase
    match for every token, with tokens weighted above the whole sentence (Q03); filters are the
    document / path / version constraints from the hints."""
    should: list[dict[str, Any]] = [
        {"multi_match": {"query": question, "fields": search_fts.SEARCH_FIELDS, "type": "most_fields"}},
        {"multi_match": {"query": question, "fields": ["filename.ngram^2", "path.ngram"]}},
    ]
    for tok in identifiers:
        for variant in identifier_variants(tok)[:3]:
            should.append({"multi_match": {"query": variant, "fields": ["body^3", "title^3", "visual^2", "filename^4", "path^2"],
                                           "type": "phrase", "boost": IDENTIFIER_BOOST}})
    body: dict[str, Any] = {"bool": {"should": should, "minimum_should_match": 1}}
    if filters:
        body["bool"]["filter"] = list(filters)
    return body


def bm25_channel(url: str, question: str, collection: str, *, limit: int, identifiers: list[str] | None = None,
                 timeout: float | None = None, filters: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    client = os_client(url)
    if not client.indices.exists(index=collection, request_timeout=timeout):
        return []
    body = {"size": int(limit), "query": bm25_query(question, identifiers or [], filters), "_source": BM25_SOURCE}
    response = client.search(index=collection, body=body, request_timeout=timeout)
    rows: list[dict[str, Any]] = []
    for hit in response.get("hits", {}).get("hits", []):
        rows.append({"point_id": str(hit.get("_id")), "score": float(hit.get("_score") or 0.0), "payload": dict(hit.get("_source") or {})})
    return rows


PROFILE_FIELDS = ["body", "title", "visual", "filename", "path"]
PROFILE_MAX_TERMS = 16


def lexical_profile(url: str, question: str, collections: list[str], *, max_terms: int = PROFILE_MAX_TERMS,
                    timeout: float | None = None) -> dict[str, Any]:
    """Raw material for knowledge-base-level lexical evidence (Q20): the question is split into tokens
    by the same cjk analyzer as the index (Chinese bigrams, Latin words, numbers), and a single msearch
    counts how many chunks each token appears in per knowledge base and how many chunks each knowledge
    base has in total. Returns {"terms": {token: {index: chunk count}}, "sizes": {index: chunk count}}.
    Raw BM25 scores are not comparable across indexes (a topic word has the lowest IDF precisely in the
    knowledge base of that topic); chunk density is the quantity that can be compared across knowledge
    bases."""
    client = os_client(url)
    existing = [c for c in collections if client.indices.exists(index=c, request_timeout=timeout)]
    if not existing:
        return {"terms": {}, "sizes": {}}
    analyzed = client.indices.analyze(body={"analyzer": "cjk", "text": question}, request_timeout=timeout)
    terms: list[str] = []
    for t in analyzed.get("tokens") or []:
        tok = str(t.get("token") or "").strip()
        if len(tok) >= 2 and tok not in terms:
            terms.append(tok)
    terms = terms[:max(1, int(max_terms))]
    target = ",".join(existing)
    agg = {"by_index": {"terms": {"field": "_index", "size": max(10, len(existing))}}}
    body: list[dict[str, Any]] = []
    for tok in terms:
        body.append({"index": target})
        body.append({"size": 0, "query": {"multi_match": {"query": tok, "fields": PROFILE_FIELDS, "type": "phrase"}}, "aggs": agg})
    body.append({"index": target})
    body.append({"size": 0, "query": {"match_all": {}}, "aggs": agg})
    responses = client.msearch(body=body, request_timeout=timeout).get("responses") or []

    def buckets(resp: dict[str, Any]) -> dict[str, int]:
        rows = ((resp.get("aggregations") or {}).get("by_index") or {}).get("buckets") or []
        return {str(b.get("key")): int(b.get("doc_count") or 0) for b in rows}

    out = {"terms": {tok: buckets(resp) for tok, resp in zip(terms, responses)}, "sizes": {}}
    if len(responses) > len(terms):
        out["sizes"] = buckets(responses[len(terms)])
    return out


NEO4J_IDLE_CHECK_SECONDS = 10.0
_driver_lock = threading.Lock()
_driver: dict[str, Any] = {"key": None, "driver": None}


def shared_driver(settings: Settings) -> Any:
    """The Neo4j driver shared by the search process (thread-safe, with its own connection pool): the
    graph channel no longer creates, verifies and closes a driver on every query. A connection idle for
    more than a few seconds is probed before use, so dead connections left in the pool by a Neo4j restart
    are not used to run queries; when the connection details change the driver is replaced."""
    from neo4j import GraphDatabase

    key = (settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    with _driver_lock:
        if _driver["driver"] is not None and _driver["key"] == key:
            return _driver["driver"]
        if not settings.neo4j_password:
            raise RuntimeError("Neo4j password is not configured. Set NEO4J_PASSWORD in config/knowledge-base.env.")
        old = _driver["driver"]
        _driver["driver"] = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password),
                                                 liveness_check_timeout=NEO4J_IDLE_CHECK_SECONDS)
        _driver["key"] = key
        if old is not None:
            old.close()
        return _driver["driver"]


def graph_channel(settings: Settings, source: KBSource, question: str, *, limit: int, hops: int,
                  vector: list[float] | None = None, lexical_only: bool = False, q: Any = None,
                  timeout: float | None = None) -> dict[str, Any]:
    """Graph channel as candidate expansion (Q04): calls graph_query and takes only chunk ids and graph
    scores as features; entities / relations / facts / pages are brought back separately for the
    evidence tables. Any failure (alias version mismatch, Neo4j unreachable, graph collection missing)
    only skips the graph channel and the query proceeds as usual. vector: pass the question vector in
    if already computed (saves one embedding); lexical_only: the embedding service is down, use
    lexical seeds only (Q19). q: the Qdrant client shared by the service (creating one each time starts
    a compatibility-check thread and builds an SSL context, a cost multiplied under concurrency).
    timeout: transaction timeout of each Neo4j query; the database terminates it when time is up, so an
    abandoned job does not keep holding a thread."""
    from kb_pipeline.graph.recall import graph_query

    try:
        res = graph_query(settings, source, question, hops=int(hops), chunk_limit=int(limit), candidate_limit=int(limit),
                          with_text=True, vector=vector, lexical_only=lexical_only, q=q, driver=shared_driver(settings),
                          timeout=timeout)
    except Exception as exc:
        return {"chunks": [], "skipped": f"{type(exc).__name__}: {str(exc)[:160]}"}
    chunks = [{"point_id": str(c.get("point_id")), "score": float(c.get("graph_score") or c.get("score") or 0.0),
               "chunk_uid": c.get("chunk_uid"), "rel_path": c.get("rel_path"),
               "entities": c.get("entities") or [], "relations": c.get("relations") or []}
              for c in res.get("chunks") or [] if c.get("point_id")]
    return {"chunks": chunks, "entities": res.get("entities") or [], "relations": res.get("relations") or [],
            "specs": res.get("specs") or [], "pages": res.get("pages") or [], "graph_version": res.get("graph_version"),
            "window": res.get("window"), "seeds": res.get("seeds") or {}}


def _visual_data_url(image_bytes: bytes) -> str:
    """The caller's image: downscaled first if over 1 million pixels, then converted to a PNG data URL
    (8103 only accepts image_url inside messages)."""
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes))
    img.load()
    w, h = img.size
    if w * h > VISUAL_MAX_PIXELS:
        scale = (VISUAL_MAX_PIXELS / (w * h)) ** 0.5
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def embed_visual_query(settings: Settings, *, text: str | None = None, image_bytes: bytes | None = None,
                       timeout: float = 20.0) -> list[float]:
    """The question (or the caller's image) into the cross-modal space (Q17): must go through 8103's
    messages form with the model card's system instruction; the `input=` form lands in a different
    space (the same text gets a cosine of only 0.47). Text-to-image and image-to-image share this
    path."""
    content: list[dict[str, Any]] = []
    if image_bytes:
        content.append({"type": "image_url", "image_url": {"url": _visual_data_url(image_bytes)}})
    if text:
        content.append({"type": "text", "text": str(text)})
    if not content:
        raise ValueError("visual query needs text or an image")
    body = {"model": settings.visual_embedding_model_id,
            "messages": [{"role": "system", "content": settings.visual_embedding_instruction}, {"role": "user", "content": content}],
            "dimensions": int(settings.visual_embedding_dim), "encoding_format": "float"}
    headers = {"Content-Type": "application/json"}
    if settings.visual_embedding_api_key:
        headers["Authorization"] = f"Bearer {settings.visual_embedding_api_key}"
    resp = requests.post(f"{settings.visual_embedding_base_url.rstrip('/')}/embeddings", json=body, headers=headers, timeout=timeout,
                         allow_redirects=False)
    resp.raise_for_status()
    items = resp.json().get("data") or []
    if len(items) != 1:
        raise RuntimeError(f"visual embedding response size mismatch: {len(items)}")
    vec = [float(x) for x in items[0]["embedding"]]
    if len(vec) != int(settings.visual_embedding_dim):
        raise RuntimeError(f"visual embedding dim mismatch: {len(vec)}")
    return vec


def visual_channel(q: Any, collection: str, vector: list[float], *, limit: int, timeout: float | None = None,
                   query_filter: Any = None) -> list[dict[str, Any]]:
    """Visual channel (Q17): fetch active image chunks by the visual vector; Qdrant naturally omits
    points that have no visual vector."""
    res = q.query_points(collection_name=collection, query=vector, using=VISUAL_VECTOR, query_filter=query_filter or ACTIVE_FILTER,
                         limit=int(limit), with_payload=True, timeout=_secs(timeout))
    return [{"point_id": str(p.id), "score": float(p.score), "payload": dict(p.payload or {})} for p in res.points]


_TABLE_SEP_RE = re.compile(r"^\|?\s*:?-{3,}")


def table_context(text: str) -> list[str]:
    """The lines of a markdown table chunk that are not data rows: lead-in lines outside the table (table
    title, table note), the header row and the separator row; empty when there is no separator row (not a
    markdown table)."""
    lines = [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]
    sep = next((i for i, ln in enumerate(lines) if _TABLE_SEP_RE.match(ln)), None)
    if sep is None:
        return []
    return [ln for i, ln in enumerate(lines) if i <= sep or not ln.startswith("|")]


def table_head(q: Any, collection: str, payload: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any] | None:
    """When a table continuation chunk is hit, find the first chunk of the same block (doc_id +
    content_version + block_id) -- that is where the table title and header live (Q16). Returns None when
    the hit is itself the first chunk, the text already carries a HEADER label (every chunk of a native
    table does), or the block has only one chunk; markdown tables repeat the header in every chunk when
    split, so when the hit already contains all lead-in and header lines of the first chunk it also
    returns None rather than spending evidence budget for nothing. The first chunk is the first one in
    ascending chunk_index order: an unordered scroll returns points by id, and when a block has more
    than one page of earlier chunks it would hand back some chunk from the middle of the table."""
    block_id = payload.get("block_id")
    doc_id = payload.get("doc_id")
    idx = payload.get("chunk_index")
    if not block_id or doc_id is None or idx is None:
        return None
    text = str(payload.get("text") or "")
    if "HEADER:" in text[:400]:
        return None
    must: list[Any] = [
        models.FieldCondition(key="is_active", match=models.MatchValue(value=True)),
        models.FieldCondition(key="doc_id", match=models.MatchValue(value=str(doc_id))),
        models.FieldCondition(key="block_id", match=models.MatchValue(value=str(block_id))),
        models.FieldCondition(key="chunk_index", range=models.Range(lt=int(idx))),
    ]
    if payload.get("content_version"):
        must.append(models.FieldCondition(key="content_version", match=models.MatchValue(value=str(payload["content_version"]))))
    points, _ = q.scroll(collection_name=collection, scroll_filter=models.Filter(must=must), limit=1, with_payload=True, with_vectors=False,
                         order_by=models.OrderBy(key="chunk_index", direction=models.Direction.ASC), timeout=_secs(timeout))
    if not points:
        return None
    head = dict(points[0].payload or {})
    head["point_id"] = str(points[0].id)
    context = table_context(str(head.get("text") or ""))
    own = {ln.strip() for ln in text.splitlines()}
    if context and all(ln in own for ln in context):
        return None
    return head


def fetch_payloads(q: Any, collection: str, point_ids: list[str], *, timeout: float | None = None) -> dict[str, dict[str, Any]]:
    """Fetch payloads from the main collection by point_id; deactivated points are not returned (Q02:
    retrieval always follows the published version)."""
    ids = [pid for pid in dict.fromkeys(str(p) for p in point_ids) if pid]
    out: dict[str, dict[str, Any]] = {}
    for start in range(0, len(ids), 256):
        batch = ids[start:start + 256]
        for rec in q.retrieve(collection_name=collection, ids=batch, with_payload=True, with_vectors=False, timeout=_secs(timeout)):
            payload = dict(rec.payload or {})
            if payload.get("is_active") is False:
                continue
            out[str(rec.id)] = payload
    return out


def point_meta(q: Any, collection: str, point_ids: list[str], *, timeout: float | None = None) -> dict[str, dict[str, Any]]:
    """Active state and document identity (doc_id / rel_path / content_version) of the source points:
    under a constrained query, derived evidence is checked against it for scope (Codex R1). Missing
    points are recorded as active=False."""
    ids = [pid for pid in dict.fromkeys(str(p) for p in point_ids) if pid]
    out: dict[str, dict[str, Any]] = {pid: {"active": False} for pid in ids}
    for start in range(0, len(ids), 256):
        batch = ids[start:start + 256]
        for rec in q.retrieve(collection_name=collection, ids=batch, with_payload=["is_active", "doc_id", "rel_path", "content_version"],
                              with_vectors=False, timeout=_secs(timeout)):
            pl = rec.payload or {}
            out[str(rec.id)] = {"active": bool(pl.get("is_active", True)), "doc_id": pl.get("doc_id"), "rel_path": pl.get("rel_path"),
                                "content_version": pl.get("content_version")}
    return out


def neighbor_payloads(q: Any, collection: str, payload: dict[str, Any], *, span: int, timeout: float | None = None) -> list[dict[str, Any]]:
    """Active chunks of the same document and version whose chunk_index is within +-span, sorted by
    index (Q07 neighbourhood backfill, never crossing a document boundary)."""
    doc_id = payload.get("doc_id")
    version = payload.get("content_version")
    idx = payload.get("chunk_index")
    if doc_id is None or idx is None or span <= 0:
        return []
    try:
        idx = int(idx)
    except (TypeError, ValueError):
        return []
    must: list[Any] = [
        models.FieldCondition(key="is_active", match=models.MatchValue(value=True)),
        models.FieldCondition(key="doc_id", match=models.MatchValue(value=str(doc_id))),
        models.FieldCondition(key="chunk_index", range=models.Range(gte=max(0, idx - span), lte=idx + span)),
    ]
    if version:
        must.append(models.FieldCondition(key="content_version", match=models.MatchValue(value=str(version))))
    points, _ = q.scroll(collection_name=collection, scroll_filter=models.Filter(must=must), limit=2 * span + 2,
                         with_payload=True, with_vectors=False, timeout=_secs(timeout))
    rows = []
    for p in points:
        pl = dict(p.payload or {})
        if int(pl.get("chunk_index", -1)) == idx:
            continue
        pl["point_id"] = str(p.id)
        rows.append(pl)
    return sorted(rows, key=lambda r: int(r.get("chunk_index") or 0))


def context_range(q: Any, collection: str, *, doc_id: str, chunk_from: int, chunk_to: int,
                  content_version: str | None = None) -> list[dict[str, Any]]:
    """/context: active chunks in an index range of one document (for follow-up questions), again
    returning only the active version."""
    must: list[Any] = [
        models.FieldCondition(key="is_active", match=models.MatchValue(value=True)),
        models.FieldCondition(key="doc_id", match=models.MatchValue(value=str(doc_id))),
        models.FieldCondition(key="chunk_index", range=models.Range(gte=int(chunk_from), lte=int(chunk_to))),
    ]
    if content_version:
        must.append(models.FieldCondition(key="content_version", match=models.MatchValue(value=str(content_version))))
    points, _ = q.scroll(collection_name=collection, scroll_filter=models.Filter(must=must),
                         limit=max(1, int(chunk_to) - int(chunk_from) + 1), with_payload=True, with_vectors=False)
    rows = []
    for p in points:
        pl = dict(p.payload or {}); pl["point_id"] = str(p.id); rows.append(pl)
    return sorted(rows, key=lambda r: int(r.get("chunk_index") or 0))
