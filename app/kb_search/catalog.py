"""Knowledge base catalog and knowledge base routing (Q20).

Catalog: each knowledge base's profile (name, domain, subject types, type table, language, size,
whether it has a graph, filename sample) grows entirely out of the knowledge base's own data, so a
knowledge base without a graph still gets a complete catalog entry; no assumption is made about the
topic, and a newly created knowledge base of any subject works the same way.

Routing does not guess from the profile: the vector and keyword channels are cheap anyway (tens of
milliseconds), so a small handful of candidates is first taken from every knowledge base, each is
scored by "how similar are the most similar chunks in this knowledge base", and only those above the
line go on to the graph channel (seconds) and reranking. Having a graph or not and the subject matter
have no effect on the score; when the caller names the knowledge bases there is no routing."""
from __future__ import annotations

import math
import threading
import time
from typing import Any

from kb_pipeline import db, discovery
from kb_pipeline.config import Settings
from kb_pipeline.vector.qdrant import graph_alias_targets, graph_collection_alias

from .channels import ACTIVE_FILTER

DOC_SAMPLE = 20
DEGRADED_TTL = 15.0  # a catalog built while the main store was out of reach is trusted only this long (seconds): after recovery it must not keep claiming "no graph" for ten minutes
TOP_N = 3            # knowledge base evidence is the mean of each channel's top few chunks: a single top score is easily inflated by one chunk of coincidentally similar boilerplate, the mean of the top 3 is steadier
_lock = threading.Lock()
_cache: dict[str, Any] = {"at": 0.0, "entries": []}


def _active_schema(con: Any, kb_id: str) -> dict[str, Any]:
    cfg = discovery.get_config(con, kb_id)
    versions = cfg.get("graph_schema_versions") or []
    active = str(cfg.get("graph_schema_active") or "")
    cur = next((v for v in versions if str(v.get("id")) == active), None) or (versions[-1] if versions else {})
    return {"domain": str(cur.get("domain") or ""), "persona": str(cur.get("persona") or "")[:300], "schema_id": cur.get("id")}


def _doc_sample(con: Any, collection: str) -> list[str]:
    """Filenames of the files with the most chunks: shows the client (or a person) roughly what this
    knowledge base holds, without depending on the graph."""
    rows = con.execute(
        "SELECT f.filename, COUNT(*) AS n FROM chunks c JOIN files f ON f.file_id = c.file_id "
        "WHERE c.collection = ? AND c.status = 'active' GROUP BY c.file_id ORDER BY n DESC, f.filename LIMIT ?",
        (collection, DOC_SAMPLE)).fetchall()
    return [str(r[0]) for r in rows]


def build_catalog(settings: Settings, q: Any) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    down: str | None = None         # main store unreachable or timing out: the remaining knowledge bases are not tried, every attempt would wait for the timeout
    with db.connect(settings.state_db) as con:
        for kb_id, source in sorted(settings.sources.items()):
            profile = dict(getattr(source, "graph_profile", None) or {})
            schema = _active_schema(con, kb_id)
            chunks, entity_target, failed = 0, None, down
            if failed is None:
                try:
                    chunks = int(q.count(collection_name=source.collection, count_filter=ACTIVE_FILTER, exact=True).count)
                except Exception as exc:
                    status = getattr(exc, "status_code", None)          # with a status code the main store answered with an error that concerns only this knowledge base
                    if status != 404:                                    # 404 means the collection does not exist yet (knowledge base just opened): really 0 chunks
                        failed = type(exc).__name__
                        down = failed if status is None else None
            docs = int(con.execute("SELECT COUNT(DISTINCT file_id) FROM chunks WHERE collection = ? AND status = 'active'",
                                   (source.collection,)).fetchone()[0])
            if failed is None:
                try:
                    targets = graph_alias_targets(q, source.collection, ("entity",))
                    entity_target = targets.get(graph_collection_alias(source.collection, "entity"))
                except Exception as exc:
                    failed = type(exc).__name__
                    down = failed if getattr(exc, "status_code", None) is None else None
            entries.append({
                "kb_id": kb_id, "name": str(source.source_root), "collection": source.collection,
                "language": getattr(source, "graph_language", None), "domain": schema["domain"], "persona": schema["persona"],
                "subject_types": list(profile.get("subject_types") or []), "axis": profile.get("axis"),
                "entity_types": list(getattr(source, "graph_entity_types", None) or [])[:40],
                "chunks": chunks, "docs": docs, "docs_sample": _doc_sample(con, source.collection),
                # between a knowledge base being emptied and its graph being retired (the next graph maintenance
                # check) the alias still points at the old graph: without active documents the graph channel is skipped
                "has_graph": bool(entity_target) and docs > 0,
                "graph_version": (str(entity_target).split("__", 1)[1] if entity_target and "__" in str(entity_target) else None),
            })
            if failed:
                # chunks that cannot be counted and a graph alias that cannot be seen mean "unknown", not "0 chunks,
                # no graph": whether there is a graph is recorded as unknown, and search still tries the graph channel
                entries[-1].update({"chunks": None, "has_graph": None, "degraded": f"qdrant: {failed}"})
    return entries


def get_catalog(settings: Settings, q: Any, *, ttl: float, force: bool = False) -> list[dict[str, Any]]:
    """The catalog is cached for ttl; it is rebuilt at once when the set of enabled knowledge bases changes
    (one opened or closed), and a catalog built while the main store was out of reach is cached only briefly."""
    with _lock:
        entries = _cache["entries"]
        if not force and entries and {e["kb_id"] for e in entries} == set(settings.sources):
            keep = min(ttl, DEGRADED_TTL) if any(e.get("degraded") for e in entries) else ttl
            if time.time() - float(_cache["at"]) < keep:
                return entries
        entries = build_catalog(settings, q)
        _cache["entries"], _cache["at"] = entries, time.time()
        return entries


def public_view(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(e) for e in entries]


def _top_mean(rows: list[dict[str, Any]], n: int) -> float:
    scores = sorted((float(r.get("score") or 0.0) for r in rows), reverse=True)[:max(1, n)]
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def lexical_evidence(profile: dict[str, Any], collections: dict[str, str]) -> dict[str, float]:
    """Knowledge-base-level lexical evidence (0-1): each token's occurrence density per knowledge base
    (chunk count / knowledge base size) divided by the highest density, then a weighted average by the
    token's rarity across knowledge bases; the fewer knowledge bases a token appears in, the higher its
    weight, a token present in every knowledge base (words like "enterprise" or "these") has weight 0,
    and a token present in none is not counted. Unlike BM25 scores, density is comparable across
    knowledge bases: a topic word has the lowest IDF precisely in the knowledge base of that topic,
    where a raw BM25 score would push it down."""
    kbs = list(collections)
    sizes = {c: max(1, int(n or 0)) for c, n in (profile.get("sizes") or {}).items()}
    acc = {kb: 0.0 for kb in kbs}
    wsum = 0.0
    for counts in (profile.get("terms") or {}).values():
        dens = {kb: int(counts.get(col, 0) or 0) / sizes.get(col, 1) for kb, col in collections.items()}
        present = sum(1 for d in dens.values() if d > 0)
        if present == 0:
            continue
        w = math.log((len(kbs) + 1) / (1 + present))
        if w <= 0:
            continue
        dmax = max(dens.values())
        for kb in kbs:
            acc[kb] += w * dens[kb] / dmax
        wsum += w
    if wsum <= 0:
        return {kb: 0.0 for kb in kbs}
    return {kb: round(acc[kb] / wsum, 4) for kb in kbs}


def library_evidence(probe: dict[str, dict[str, Any]], lexical: dict[str, float] | None = None, *,
                     top_n: int = TOP_N) -> dict[str, dict[str, float]]:
    """Amount of evidence per knowledge base: vec = mean cosine of the vector channel's top_n chunks (0
    with no results), lex = knowledge-base-level lexical evidence (0 if absent), vis = mean cosine of
    the visual channel's top_n chunks (the visual channel is probed only for requests carrying an
    image, otherwise 0)."""
    lexical = lexical or {}
    return {kb: {"vec": _top_mean(chans.get("text") or [], top_n), "lex": float(lexical.get(kb, 0.0) or 0.0),
                 "vis": _top_mean(chans.get("visual") or [], top_n) if isinstance(chans.get("visual"), list) else 0.0}
            for kb, chans in probe.items()}


def route(evidence: dict[str, dict[str, float]], *, max_kbs: int, gap: float, floor: float,
          lexical_weight: float, visual_weight: float = 0.0) -> dict[str, Any]:
    """Choose knowledge bases by evidence: each quantity is divided by its maximum over all knowledge
    bases to give 0-1 and combined by weight; knowledge bases within gap of the top score are queried
    together, at most max_kbs of them. A quantity that is 0 in every knowledge base (vectors
    unavailable, no discriminative words in the question) hands its whole weight to the others.
    visual_weight > 0 means a request carrying an image (Codex S02): visual evidence takes part with
    this weight and the two text channels share the rest. weak: the vector evidence of every knowledge
    base is below floor, meaning the question barely relates to any of them; this is only a hint and
    does not change the selection."""
    if not evidence:
        return {"mode": "auto", "chosen": [], "scores": {}, "weak": True}
    vec_max = max(e["vec"] for e in evidence.values())
    lex_max = max(e["lex"] for e in evidence.values())
    vis_max = max(float(e.get("vis") or 0.0) for e in evidence.values())
    w_vis = float(visual_weight) if vis_max > 0 else 0.0
    text_share = 1.0 - w_vis
    w_lex = text_share * float(lexical_weight) if lex_max > 0 else 0.0
    w_vec = text_share * (1.0 - float(lexical_weight)) if vec_max > 0 else 0.0
    total_w = (w_vec + w_lex + w_vis) or 1.0
    scores: dict[str, dict[str, float]] = {}
    for kb, e in evidence.items():
        vec_rel = e["vec"] / vec_max if vec_max > 0 else 0.0
        lex_rel = e["lex"] / lex_max if lex_max > 0 else 0.0
        vis_rel = float(e.get("vis") or 0.0) / vis_max if vis_max > 0 else 0.0
        scores[kb] = {"vec": e["vec"], "lex": e["lex"], "score": round((w_vec * vec_rel + w_lex * lex_rel + w_vis * vis_rel) / total_w, 4)}
        if w_vis:
            scores[kb]["vis"] = float(e.get("vis") or 0.0)
    ranked = sorted(scores, key=lambda k: (-scores[k]["score"], k))
    best = scores[ranked[0]]["score"]
    chosen = [k for k in ranked if scores[k]["score"] >= best - float(gap)][:max(1, int(max_kbs))]
    return {"mode": "auto", "chosen": chosen, "scores": scores, "weak": bool(0.0 < vec_max < float(floor))}


def route_explicit(explicit: list[str], entries: list[dict[str, Any]]) -> dict[str, Any]:
    """The caller named the knowledge bases: use them as given, listing unrecognised ones in unknown."""
    known = {e["kb_id"] for e in entries}
    return {"mode": "explicit", "chosen": [k for k in explicit if k in known], "unknown": [k for k in explicit if k not in known], "scores": {}}
