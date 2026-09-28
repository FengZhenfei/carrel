"""Retrieval orchestration: the two cheap channels probe every knowledge base -> knowledge bases are
chosen by evidence -> the graph / visual channels run only on the chosen ones -> RRF fusion per
knowledge base, interleaved by subject / document buckets -> cross-encoder rerank (threshold + floor)
-> boilerplate chunks / low-confidence images down-weighted -> quota reservation + MMR cut to top-k ->
evidence assembly -> summary. Returns evidence only and never generates answers. When the chosen
knowledge bases yield no decent evidence, the search automatically widens to all knowledge bases and
runs again (the cheap channels' candidates are already in hand, only the graph channel is added).
Requests carrying an image (image-to-image): the visual channel probes every knowledge base and takes
part in the selection, and visual candidates are exempt from the text rerank threshold and get slots
reserved by visual score (Codex S02). The whole service shares one bounded thread pool, every stage
collects results against a deadline, and a slow channel is only recorded as degraded instead of
holding up the whole request (Codex S06)."""
from __future__ import annotations

import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any

from kb_pipeline.config import Settings, load_settings
from kb_pipeline.vector.qdrant import client as qdrant_client

from . import catalog as catalog_mod
from . import channels
from .config import SearchSettings, load_search_settings
from .evidence import assemble_sources, doc_aggs, numbered, page_rows, select_hits, spec_rows
from .fusion import interleave, rrf_merge
from .rerank import Reranker, rerank

_state: dict[str, Any] = {"settings": None, "search": None, "q": None, "at": 0.0, "pool": None, "pool_size": 0}
_state_lock = threading.Lock()
SETTINGS_TTL = 60.0

ENTITY_FIELDS = ("id", "title", "type", "scope", "score", "hop", "via", "docs", "description")
RELATION_FIELDS = ("id", "source", "type", "target", "score", "hop", "via", "description")
SPEC_FIELDS = ("id", "hint", "sources", "sources_active", "verified", "series_text", "conflict", "conflict_note",
               "subject", "property", "symbol", "concept", "value", "min", "typ", "max", "unit", "unit_canonical", "flag", "ref_min", "ref_max",
               "when", "valid_from", "valid_until", "series_key", "series_index", "conflict_group", "conditions", "conditions_text", "kinds", "quality",
               "confidence", "comparable", "text", "note", "score", "rel_path", "section", "point_ids")
PAGE_FIELDS = ("id", "kind", "title", "score", "summary", "text", "text_truncated", "compiled", "verified", "sources_active", "docs", "series", "point_ids")
# Generic words pointing across documents (Q08): when the question asks "both / each / respectively /
# all / compare" and the candidates come from only a few documents, quotas are split by document
CROSS_DOC_RE = re.compile(r"(两份|两个文档|各自|各有|各是|分别|都有|都是|对比|比较|区别|差异|异同|共同|哪几份|每份|每个文档)")


def runtime() -> tuple[Settings, SearchSettings, Any]:
    """Settings and the Qdrant client: the env is re-read once a minute (knowledge base toggles and
    parameter changes need no restart)."""
    with _state_lock:
        if _state["settings"] is None or time.time() - float(_state["at"]) > SETTINGS_TTL:
            settings = load_settings()
            _state["settings"] = settings
            _state["search"] = load_search_settings()
            _state["q"] = qdrant_client(settings.qdrant_url, settings.qdrant_api_key)
            _state["at"] = time.time()
        return _state["settings"], _state["search"], _state["q"]


def _pool(ss: SearchSettings) -> ThreadPoolExecutor:
    """Thread pool shared by the whole service: the thread count is capped and does not grow with
    requests (Codex S06)."""
    with _state_lock:
        size = max(4, int(ss.pool_workers))
        if _state["pool"] is None or _state["pool_size"] != size:
            _state["pool"] = ThreadPoolExecutor(max_workers=size, thread_name_prefix="kb-search")
            _state["pool_size"] = size
        return _state["pool"]


def _collect(jobs: dict[Any, Future], deadline: float) -> dict[Any, tuple[bool, Any]]:
    """Collect results against a shared deadline: whatever has not returned by then is recorded as a
    TimeoutError and abandoned (the call in the thread finishes as usual, it is just no longer waited
    for)."""
    out: dict[Any, tuple[bool, Any]] = {}
    for key, fut in jobs.items():
        remaining = deadline - time.time()
        try:
            out[key] = (True, fut.result(timeout=max(0.01, remaining)))
        except FutureTimeout:
            fut.cancel()
            out[key] = (False, TimeoutError(f"deadline exceeded ({remaining:.1f}s left)"))
        except Exception as exc:
            out[key] = (False, exc)
    return out


def _err(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {str(exc)[:120]}"


def rerank_active(settings: Any, ss: Any) -> bool:
    """Reranking counts as enabled only when the switch is on and an address is configured: an empty
    RERANKER_BASE_URL means "this machine has no rerank service", and then no request should be sent to
    an empty address only to record MissingSchema as a degradation."""
    return bool(ss.rerank_enabled and str(getattr(settings, "reranker_base_url", "") or "").strip())


def health() -> dict[str, Any]:
    settings, ss, q = runtime()
    checks: dict[str, Any] = {}
    try:
        q.get_collections(); checks["qdrant"] = "ok"
    except Exception as exc:
        checks["qdrant"] = f"error: {type(exc).__name__}"
    import importlib.util

    checks["pdf_images"] = "ok" if importlib.util.find_spec("pymupdf") else "unavailable: install carrel[pdf-images] (PyMuPDF) for original-resolution PDF pictures"
    return {"ok": True, "service": "kb_search", "kbs": sorted(settings.sources), "auth": "bearer" if ss.token else "loopback-only",
            "rerank": rerank_active(settings, ss), "visual": ss.visual_enabled, "checks": checks}


def catalog(force: bool = False) -> dict[str, Any]:
    settings, ss, q = runtime()
    entries = catalog_mod.get_catalog(settings, q, ttl=ss.catalog_ttl, force=force)
    return {"kbs": catalog_mod.public_view(entries)}


def _probe(settings: Settings, ss: SearchSettings, q: Any, kb_ids: list[str], question: str, vector: list[float] | None,
           identifiers: list[str], *, with_profile: bool, image_bytes: bytes | None = None,
           hint: dict[str, Any] | None = None) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """The two cheap channels (vector, keyword) fetch candidates from the given knowledge bases in
    parallel; with auto routing a knowledge-base-level lexical evidence count is taken along the way,
    and the visual channel's question vector is computed in this step too; for a request carrying an
    image the visual channel also probes every knowledge base (it takes part in the selection). A
    failing channel is only recorded as degraded and does not affect other knowledge bases or channels."""
    out: dict[str, dict[str, Any]] = {kb: {"text": [], "bm25": [], "channels": {}, "degraded": []} for kb in kb_ids}
    profile: dict[str, Any] = {"lexical": None, "visual_vector": None, "visual": "disabled"}
    pool = _pool(ss)
    deadline = time.time() + float(ss.channel_timeout)
    qfilter = channels.hint_filter(hint) if hint else None
    os_filters = channels.hint_os_filters(hint) if hint else None
    jobs: dict[tuple[str, str], Future] = {}
    for kb in kb_ids:
        src = settings.sources[kb]
        if vector is not None:
            jobs[(kb, "text")] = pool.submit(channels.vector_channel, q, src.collection, vector, limit=ss.vector_k, timeout=ss.channel_timeout,
                                             query_filter=qfilter)
        else:
            out[kb]["channels"]["text"] = {"skipped": "embedding unavailable"}
        jobs[(kb, "bm25")] = pool.submit(channels.bm25_channel, settings.opensearch_url, question, src.collection,
                                         limit=ss.bm25_k, identifiers=identifiers, timeout=ss.channel_timeout, filters=os_filters)
    if with_profile:
        jobs[("*", "profile")] = pool.submit(channels.lexical_profile, settings.opensearch_url, question,
                                             [settings.sources[kb].collection for kb in kb_ids])
    if ss.visual_enabled and getattr(settings, "visual_embedding_enabled", True):
        jobs[("*", "visual_vector")] = pool.submit(channels.embed_visual_query, settings, text=None if image_bytes else question,
                                                   image_bytes=image_bytes, timeout=ss.visual_timeout)
    results = _collect(jobs, deadline)
    for (kb, name), (ok, val) in results.items():
        if kb == "*":
            continue
        if not ok:
            out[kb]["channels"][name] = {"skipped": _err(val)}
            out[kb]["degraded"].append(f"{kb}:{name}")
            continue
        out[kb][name] = val
        out[kb]["channels"][name] = {"candidates": len(val)}
    if with_profile:
        ok, val = results[("*", "profile")]
        if ok:
            profile["lexical"] = catalog_mod.lexical_evidence(val, {kb: settings.sources[kb].collection for kb in kb_ids})
            profile["terms"] = list((val.get("terms") or {}).keys())
        else:
            profile["skipped"] = _err(val)
    if ("*", "visual_vector") in results:
        ok, val = results[("*", "visual_vector")]
        if ok:
            profile["visual_vector"], profile["visual"] = val, "ok"
        else:
            profile["visual"] = f"skipped: {_err(val)[:80]}"
    if image_bytes and profile["visual_vector"] is not None:
        # Request carrying an image: the visual channel probes every knowledge base and its evidence
        # takes part in the selection (Codex S02)
        vjobs = {kb: pool.submit(channels.visual_channel, q, settings.sources[kb].collection, profile["visual_vector"],
                                 limit=ss.visual_k, timeout=ss.channel_timeout, query_filter=qfilter) for kb in kb_ids}
        for kb, (ok, val) in _collect(vjobs, time.time() + float(ss.channel_timeout)).items():
            if ok:
                out[kb]["visual"] = val
                out[kb]["channels"]["visual"] = {"candidates": len(val)}
            else:
                out[kb]["visual"] = {"skipped": _err(val)}
                out[kb]["channels"]["visual"] = {"skipped": _err(val)}
                out[kb]["degraded"].append(f"{kb}:visual")
    return out, profile


def _graph_probe(settings: Settings, ss: SearchSettings, kb_ids: list[str], entries: dict[str, dict[str, Any]],
                 question: str, vector: list[float] | None) -> dict[str, dict[str, Any]]:
    """The graph channel runs only on knowledge bases that have a graph (one without a graph simply
    lacks this channel, no switch needed); a failure or timeout only skips the graph channel. The
    question vector is passed in to save one embedding; when the embedding service is down only lexical
    seeds are used (Q19)."""
    out: dict[str, dict[str, Any]] = {}
    graph_kbs = [kb for kb in kb_ids if entries.get(kb, {}).get("has_graph")]
    if not graph_kbs:
        return out
    pool = _pool(ss)
    jobs = {kb: pool.submit(channels.graph_channel, settings, settings.sources[kb], question, limit=ss.graph_k, hops=ss.graph_hops,
                            vector=vector, lexical_only=vector is None) for kb in graph_kbs}
    for kb, (ok, val) in _collect(jobs, time.time() + float(ss.channel_timeout)).items():
        out[kb] = val if ok else {"chunks": [], "skipped": _err(val)}
    return out


def _visual_probe(settings: Settings, ss: SearchSettings, q: Any, kb_ids: list[str], vector: list[float] | None,
                  probe: dict[str, dict[str, Any]], hint: dict[str, Any] | None = None) -> dict[str, Any]:
    """The visual channel (Q17) queries only the chosen knowledge bases; those already queried in the
    probe stage (requests carrying an image) are reused directly; without a visual vector the whole
    channel is skipped."""
    out: dict[str, Any] = {}
    if vector is None:
        return out
    pool = _pool(ss)
    qfilter = channels.hint_filter(hint) if hint else None
    jobs = {}
    for kb in kb_ids:
        if "visual" in probe.get(kb, {}):
            out[kb] = probe[kb]["visual"]
        else:
            jobs[kb] = pool.submit(channels.visual_channel, q, settings.sources[kb].collection, vector, limit=ss.visual_k, timeout=ss.channel_timeout,
                                   query_filter=qfilter)
    for kb, (ok, val) in _collect(jobs, time.time() + float(ss.channel_timeout)).items():
        out[kb] = val if ok else {"skipped": _err(val)}
    return out


def _fuse(kb: str, probe: dict[str, Any], graph: dict[str, Any] | None, visual: Any, *, k: int) -> dict[str, Any]:
    """RRF fusion of the channels within one knowledge base, together with per-channel stats and
    degradation entries."""
    results: dict[str, list[dict[str, Any]]] = {"text": probe["text"], "bm25": probe["bm25"]}
    stats = dict(probe["channels"])
    degraded = list(probe["degraded"])
    if graph is not None:
        if graph.get("skipped"):
            stats["graph"] = {"skipped": graph["skipped"]}
            degraded.append(f"{kb}:graph")
        else:
            results["graph"] = graph["chunks"]
            stats["graph"] = {"candidates": len(graph["chunks"]), "graph_version": graph.get("graph_version"), "seeds": graph.get("seeds")}
    if isinstance(visual, dict) and visual.get("skipped"):
        stats["visual"] = {"skipped": visual["skipped"]}
        if f"{kb}:visual" not in degraded:
            degraded.append(f"{kb}:visual")
    elif isinstance(visual, list):
        results["visual"] = visual
        stats["visual"] = {"candidates": len(visual)}
    fused = rrf_merge(results, k=k)
    for c in fused:
        c["kb_id"] = kb
    stats["merged"] = len(fused)
    return {"fused": fused, "channels": stats, "degraded": degraded, "graph": None if (graph or {}).get("skipped") else graph}


def _backfill(q: Any, settings: Settings, cands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fill in payloads before the rerank (BM25 / graph channels carry ids only); deactivated points are
    dropped here."""
    by_kb: dict[str, list[dict[str, Any]]] = {}
    for c in cands:
        by_kb.setdefault(str(c["kb_id"]), []).append(c)
    kept: list[dict[str, Any]] = []
    for kb_id, rows in by_kb.items():
        need = [c["point_id"] for c in rows if not (c.get("payload") or {}).get("text")]
        found = channels.fetch_payloads(q, settings.sources[kb_id].collection, need) if need else {}
        for c in rows:
            if not (c.get("payload") or {}).get("text"):
                pl = found.get(c["point_id"])
                if pl is None:
                    continue
                c["payload"] = pl
            kept.append(c)
    order = {id(c): i for i, c in enumerate(cands)}
    return sorted(kept, key=lambda c: order[id(c)])


def _cand_text(c: dict[str, Any]) -> str:
    pl = c.get("payload") or {}
    return " ".join(str(pl.get(k) or "") for k in ("filename", "title", "text")).casefold()


def _bucket_of(c: dict[str, Any], mode: str, keys: list[str]) -> str | None:
    if mode == "doc":
        rp = str((c.get("payload") or {}).get("rel_path") or c.get("rel_path") or "")
        return rp if rp in keys else None
    ents = {str(e).casefold() for e in (c.get("entities") or [])}
    text = None
    for key in sorted(keys, key=len, reverse=True):
        kf = key.casefold()
        if kf in ents:
            return key
        if text is None:
            text = _cand_text(c)
        if kf in text:
            return key
    return None


def _detect_buckets(question: str, chosen: list[str], per_kb: dict[str, dict[str, Any]], *, max_buckets: int) -> dict[str, Any] | None:
    """Multi-subject / multi-document quotas (Q08): when the question names >= 2 subjects from the
    graph (seed titles occurring in the question, entities hit by token, or fact subjects), buckets are
    by subject; otherwise, when the question points across documents and some knowledge base's
    candidates come from only a few documents, buckets are by that knowledge base's documents. Every
    candidate is tagged with its bucket."""
    q_fold = str(question or "").casefold()
    subjects: list[str] = []
    for kb in chosen:
        g = per_kb[kb].get("graph") or {}
        for e in g.get("entities") or []:
            title = str(e.get("title") or "").strip()
            if len(title) < 2 or (e.get("hop") or 0) != 0:
                continue
            if (title.casefold() in q_fold or e.get("via") == "lexical") and title not in subjects:
                subjects.append(title)
        for sp in g.get("specs") or []:
            subj = str(sp.get("subject") or "").strip()
            if len(subj) >= 2 and subj.casefold() in q_fold and subj not in subjects:
                subjects.append(subj)
    # "WorkBuddy" and "WorkBuddy Enterprise" are the same thing in a question: a subject that is a
    # substring of another does not get a bucket of its own
    subjects = [a for a in subjects if not any(a != b and a.casefold() in b.casefold() for b in subjects)]
    mode, keys = None, []
    if 2 <= len(subjects) <= max_buckets:
        mode, keys = "subject", subjects
    elif CROSS_DOC_RE.search(question):
        # Per knowledge base: where the candidates come from only a few documents, each of those
        # documents becomes a bucket; a knowledge base whose candidates are spread over many documents
        # is not split
        docs: list[str] = []
        for kb in chosen:
            kb_docs: list[str] = []
            for c in per_kb[kb]["fused"]:
                rp = str((c.get("payload") or {}).get("rel_path") or c.get("rel_path") or "")
                if rp and rp not in kb_docs:
                    kb_docs.append(rp)
            if 2 <= len(kb_docs) <= max_buckets:
                docs.extend(d for d in kb_docs if d not in docs)
        if len(docs) >= 2:
            mode, keys = "doc", docs
    if not mode:
        return None
    for kb in chosen:
        for c in per_kb[kb]["fused"]:
            c["bucket"] = _bucket_of(c, mode, keys)
    return {"mode": mode, "keys": keys}


def _groups(chosen: list[str], per_kb: dict[str, dict[str, Any]], buckets: dict[str, Any] | None, limit: int) -> list[list[dict[str, Any]]]:
    """Groups for interleaved candidate selection: with buckets, one group per knowledge base per bucket
    (candidates outside any bucket form a group of their own); without, one group per knowledge base."""
    groups: list[list[dict[str, Any]]] = []
    for kb in chosen:
        fused = per_kb[kb]["fused"][:limit * 2]
        if buckets:
            for key in buckets["keys"]:
                rows = [c for c in fused if c.get("bucket") == key]
                if rows:
                    groups.append(rows)
            rest = [c for c in fused if c.get("bucket") is None]
            if rest:
                groups.append(rest)
        else:
            groups.append(fused[:limit])
    return groups


def _rank(settings: Settings, ss: SearchSettings, q: Any, question: str, groups: list[list[dict[str, Any]]],
          buckets: dict[str, Any] | None = None, *, image_query: bool = False, hint: dict[str, Any] | None = None) -> dict[str, Any]:
    """Several groups: round-robin interleaving so every group gets a share; a single group: the top
    rerank_n of the fusion order directly. After payload backfill, cross-encoder rerank with threshold
    and floor (Q06 / Q19). Candidates carrying only an id get their text at backfill, so bucketing is
    redone here once. In a request carrying an image, visual channel candidates are exempt from the
    text threshold (Codex S02)."""
    if len(groups) > 1:
        cands = interleave(groups, limit=ss.rerank_n)
    else:
        cands = list(groups[0][:ss.rerank_n]) if groups else []
    cands = _backfill(q, settings, cands)
    if hint:
        cands = [c for c in cands if channels.hint_allows(hint, c.get("payload") or {})]   # candidates that bypassed the filter (graph channel etc.) are screened by the same constraints
    if buckets:
        for c in cands:
            if c.get("bucket") is None:
                c["bucket"] = _bucket_of(c, buckets["mode"], buckets["keys"])
    status, error, ordered, floor_used = "disabled", None, cands, None
    if rerank_active(settings, ss) and cands:
        try:
            scores = rerank(Reranker(settings.reranker_base_url, timeout=ss.rerank_timeout), question, cands,
                            window_tokens=ss.rerank_window_tokens, overlap_tokens=ss.rerank_overlap_tokens)
            for c, s in zip(cands, scores):
                c["score_rerank"] = round(float(s), 4)
            ordered = sorted(cands, key=lambda c: -float(c.get("score_rerank") or 0.0))
            status = "ok"
            if ss.rerank_threshold > 0:
                floor_used = ss.rerank_threshold
                is_visual = (lambda c: image_query and (c.get("scores") or {}).get("score_visual") is not None)
                kept = [c for c in ordered if float(c.get("score_rerank") or 0.0) >= floor_used or is_visual(c)]
                if not kept:
                    floor_used *= 0.7
                    kept = [c for c in ordered if float(c.get("score_rerank") or 0.0) >= floor_used]
                    status = "floor_lowered" if kept else "below_threshold"
                if kept:
                    ordered = kept
        except Exception as exc:
            status, error, ordered = "skipped", f"rerank: {type(exc).__name__}", cands
    return {"cands": cands, "ordered": ordered, "status": status, "error": error, "floor": floor_used}


def _final_scores(ordered: list[dict[str, Any]], ss: SearchSettings, *, reranked: bool, image_query: bool = False,
                  block_types: list[str] | None = None, question: str = "") -> dict[str, int]:
    """Final ranking score: the rerank score (normalised RRF when there was no rerank) blended with
    question token coverage by final_lex_weight (in table-heavy knowledge bases the rerank score
    saturates at 0.99 across whole runs, and coverage tells the rows of one table apart); boilerplate
    chunks (Q18) and low-confidence images (Q17) are multiplied by a down-weighting factor, not
    removed; in a request carrying an image, visual channel candidates take the higher of visual
    similarity and rerank score (the text rerank cannot see the image, Codex S02); chunks matching
    hints.block_types are multiplied by the soft preference factor."""
    from .text import is_boilerplate, question_coverage

    top_rrf = max((float(c.get("rrf") or 0.0) for c in ordered), default=0.0) or 1.0
    counts = {"boilerplate": 0, "visual_low": 0, "hinted_blocks": 0}
    wanted = {str(b).casefold() for b in (block_types or [])}
    w_lex = max(0.0, min(1.0, float(ss.final_lex_weight))) if question else 0.0
    for c in ordered:
        base = float(c.get("score_rerank") or 0.0) if reranked else float(c.get("rrf") or 0.0) / top_rrf
        pl = c.get("payload") or {}
        if w_lex:
            cov = question_coverage(question, str(pl.get("text") or ""))
            c.setdefault("scores", {})["score_lex"] = cov
            base = (1.0 - w_lex) * base + w_lex * cov
        factor = 1.0
        if is_boilerplate(pl):
            c["boilerplate"] = True
            factor *= ss.boilerplate_factor
            counts["boilerplate"] += 1
        if pl.get("visual_ref") and str(pl.get("visual_confidence") or "").lower() == "low":
            factor *= ss.visual_low_confidence_factor
            counts["visual_low"] += 1
        if wanted and str(pl.get("block_type") or "").casefold() in wanted:
            factor *= ss.hint_block_boost
            counts["hinted_blocks"] += 1
        score = base * factor
        vis = (c.get("scores") or {}).get("score_visual")
        if image_query and vis is not None:
            score = max(score, float(vis))
        c["score_final"] = round(score, 4)
    ordered.sort(key=lambda c: -float(c.get("score_final") or 0.0))
    return counts


def _mark_accepted(ordered: list[dict[str, Any]], status: str, floor: float | None, *, image_query: bool) -> str:
    """Tag every candidate with accepted (Codex S05): only a rerank score above the floor counts as an
    accepted basis; when all are below the floor they are diagnostic candidates and the whole package
    has evidence_state = diagnostic; without a rerank (down / off) the tag is None and evidence_state
    = unranked. In a request carrying an image, visual channel candidates are judged by visual score."""
    if status in ("ok", "floor_lowered", "below_threshold") and floor is not None:
        for c in ordered:
            vis = (c.get("scores") or {}).get("score_visual")
            if image_query and vis is not None:
                c["accepted"] = float(vis) >= 0.5
            else:
                c["accepted"] = status != "below_threshold" and float(c.get("score_rerank") or 0.0) >= floor
        return "accepted" if any(c.get("accepted") for c in ordered) else "diagnostic"
    if status == "ok":
        for c in ordered:
            c["accepted"] = True
        return "accepted"
    for c in ordered:
        c["accepted"] = None
    return "unranked"


def search(question: str, *, kbs: list[str] | None = None, top_k: int | None = None, hints: dict[str, Any] | None = None,
           with_context: bool = True, explain: bool = False, image_bytes: bytes | None = None) -> dict[str, Any]:
    t0 = time.time()
    question = str(question or "").strip()
    if not question:
        raise ValueError("question is empty")
    settings, ss, q = runtime()
    image_query = bool(image_bytes)
    hint_used, hint_ignored = channels.parse_hints(hints)
    doc_hint = {k: v for k, v in hint_used.items() if k != "block_types"} or None
    timings: dict[str, int] = {}
    degraded: list[str] = []
    entries = catalog_mod.get_catalog(settings, q, ttl=ss.catalog_ttl)
    by_id = {e["kb_id"]: e for e in entries}
    identifiers = channels.question_identifiers(question)
    vector: list[float] | None = None
    t = time.time()
    try:
        vector = channels.embed_question(settings, question, timeout=ss.embed_timeout, instruction=ss.query_instruction or None)
    except Exception as exc:
        degraded.append(f"embedding: {type(exc).__name__}")
    timings["embed_ms"] = int((time.time() - t) * 1000)
    # Probe scope: with knowledge bases named, only those are queried; otherwise the two cheap channels
    # run on every enrolled knowledge base
    if kbs:
        routing = catalog_mod.route_explicit(kbs, entries)
        probed = [k for k in routing["chosen"] if k in settings.sources]
        if not probed:
            raise ValueError(f"no known knowledge base among {kbs}")
    else:
        probed = [k for k in sorted(settings.sources) if k in by_id]
        if not probed:
            raise ValueError("no knowledge base enrolled")
        routing = None
    t = time.time()
    probe, profile = _probe(settings, ss, q, probed, question, vector, identifiers, with_profile=routing is None, image_bytes=image_bytes,
                            hint=doc_hint)
    timings["probe_ms"] = int((time.time() - t) * 1000)
    if routing is None:
        routing = catalog_mod.route(catalog_mod.library_evidence(probe, profile.get("lexical")), max_kbs=ss.route_max_kbs,
                                    gap=ss.route_gap, floor=ss.route_floor, lexical_weight=ss.route_lexical_weight,
                                    visual_weight=ss.visual_route_weight if image_query else 0.0)
        routing["terms"] = profile.get("terms") or []
        if profile.get("skipped"):
            routing["lexical"] = f"skipped: {profile['skipped']}"
    routing["probed"] = probed
    chosen = [k for k in routing["chosen"] if k in probe]
    # The graph / visual channels run only on the chosen knowledge bases, then fusion per knowledge base
    t = time.time()
    graphs = _graph_probe(settings, ss, chosen, by_id, question, vector)
    visuals = _visual_probe(settings, ss, q, chosen, profile.get("visual_vector"), probe, doc_hint)
    per_kb: dict[str, dict[str, Any]] = {kb: _fuse(kb, probe[kb], graphs.get(kb), visuals.get(kb), k=ss.rrf_k) for kb in chosen}
    timings["recall_ms"] = int((time.time() - t) * 1000)
    buckets = _detect_buckets(question, chosen, per_kb, max_buckets=ss.quota_max_buckets)
    t = time.time()
    ranked = _rank(settings, ss, q, question, _groups(chosen, per_kb, buckets, ss.rerank_n), buckets, image_query=image_query, hint=doc_hint)
    timings["rerank_ms"] = int((time.time() - t) * 1000)
    # Widening: the knowledge bases chosen by auto routing yield no decent evidence (no candidates, or
    # every rerank score below the floor) -> add the graph channel for the remaining knowledge bases,
    # merge and rank again
    routing["widened"] = False
    rest = [k for k in probed if k not in chosen]
    if ss.route_widen and routing["mode"] == "auto" and rest and (not ranked["cands"] or ranked["status"] == "below_threshold"):
        t = time.time()
        graphs.update(_graph_probe(settings, ss, rest, by_id, question, vector))
        visuals.update(_visual_probe(settings, ss, q, rest, profile.get("visual_vector"), probe, doc_hint))
        for kb in rest:
            per_kb[kb] = _fuse(kb, probe[kb], graphs.get(kb), visuals.get(kb), k=ss.rrf_k)
        chosen = chosen + rest
        buckets = _detect_buckets(question, chosen, per_kb, max_buckets=ss.quota_max_buckets)
        ranked = _rank(settings, ss, q, question, _groups(chosen, per_kb, buckets, ss.rerank_n), buckets, image_query=image_query, hint=doc_hint)
        routing["widened"] = True
        routing["chosen"] = chosen
        timings["widen_ms"] = int((time.time() - t) * 1000)
    for kb in chosen:
        degraded.extend(per_kb[kb]["degraded"])
    for kb in probed:
        if kb not in chosen:
            degraded.extend(x for x in probe[kb]["degraded"] if x not in degraded)    # degradation in the probe stage affects the selection, so it is reported too
    if ranked["error"]:
        degraded.append(ranked["error"])
    cands, ordered, rerank_status = ranked["cands"], ranked["ordered"], ranked["status"]
    reranked = rerank_status in ("ok", "floor_lowered", "below_threshold")
    downweighted = _final_scores(ordered, ss, reranked=reranked, image_query=image_query, block_types=hint_used.get("block_types"), question=question)
    evidence_state = _mark_accepted(ordered, rerank_status, ranked.get("floor"), image_query=image_query)
    merged_n = sum(len(per_kb[kb]["fused"]) for kb in chosen)
    k = int(top_k or ss.top_k)
    hits, selection = select_hits(ordered, k, buckets=buckets, min_hits=ss.quota_min_hits,
                                  mmr_lambda=ss.mmr_lambda if ss.mmr_enabled else None,
                                  priority_key="score_visual" if image_query else None, priority_n=ss.visual_quota if image_query else 0)
    # Evidence
    t = time.time()

    def neighbors(row: dict[str, Any], span: int) -> list[dict[str, Any]]:
        kb = str(row.get("kb_id"))
        return channels.neighbor_payloads(q, settings.sources[kb].collection, row, span=span)

    def head_of(row: dict[str, Any]) -> dict[str, Any] | None:
        kb = str(row.get("kb_id"))
        return channels.table_head(q, settings.sources[kb].collection, row)

    sources, src_stats = assemble_sources(
        hits, budget_tokens=ss.context_tokens, neighbors=neighbors if with_context else None,
        table_head=head_of if with_context else None, stitch=(ss.stitch_min_chars, ss.stitch_max_chars) if with_context else None,
        neighbor_span=ss.neighbor_span)
    timings["evidence_ms"] = int((time.time() - t) * 1000)
    source_ns = {str(r.get("point_id")): int(r["n"]) for r in sources}
    entities: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    specs: list[dict[str, Any]] = []
    pages: list[dict[str, Any]] = []
    graph_versions: dict[str, str | None] = {}
    for kb in chosen:
        g = per_kb[kb].get("graph") or {}
        if g:
            graph_versions[kb] = g.get("graph_version")
        for row in g.get("entities") or []:
            entities.append({**row, "kb_id": kb})
        for row in g.get("relations") or []:
            relations.append({**row, "kb_id": kb})
        for row in g.get("specs") or []:
            specs.append({**row, "kb_id": kb})
        for row in g.get("pages") or []:
            pages.append({**row, "kb_id": kb})
    for rows in (entities, relations, specs):
        rows.sort(key=lambda r: -float(r.get("score") or 0.0))
    specs = specs[:24]
    # Source point verification for derived evidence (Codex S03 / R1): whether the points that facts /
    # pages refer to are still active and which document they belong to; a point already in Sources
    # uses its payload directly
    meta: dict[str, dict[str, Any]] = {str(r.get("point_id")): {"active": True, "doc_id": r.get("doc_id"), "rel_path": r.get("rel_path"),
                                                                 "content_version": r.get("content_version")} for r in sources}
    t = time.time()
    for kb in chosen:
        want = {str(p) for row in specs + pages if row.get("kb_id") == kb for p in (row.get("point_ids") or [])} - set(meta)
        if want:
            try:
                meta.update(channels.point_meta(q, settings.sources[kb].collection, sorted(want)))
            except Exception as exc:
                degraded.append(f"{kb}:point_meta: {type(exc).__name__}")
    timings["verify_ms"] = int((time.time() - t) * 1000)
    point_active = {pid: bool(m.get("active")) for pid, m in meta.items()}
    hints_scope: dict[str, int] = {}
    if doc_hint:
        # Constrained query (Codex R1): derived evidence follows the same document / path / version
        # constraints -- a fact needs an active source point within scope; pages are compiled across
        # documents and are not attached as soon as any source is out of scope; entities are screened
        # by their source document list; relations carry no document information and are never
        # attached; nothing out of scope may serve as a basis for this request
        in_scope = {pid: bool(m.get("active")) and channels.hint_allows(doc_hint, m) for pid, m in meta.items()}
        allowed_paths = {str(m.get("rel_path")) for pid, m in meta.items() if in_scope.get(pid) and m.get("rel_path")}
        allowed_paths |= {str(x) for x in (doc_hint.get("rel_paths") or [])}
        kept_specs = []
        for sp in specs:
            pids = [str(x) for x in (sp.get("point_ids") or [])]
            if pids and any(in_scope.get(x) for x in pids):
                sp["point_ids"] = [x for x in pids if in_scope.get(x)]
                kept_specs.append(sp)
        hints_scope["specs_dropped"] = len(specs) - len(kept_specs)
        specs = kept_specs
        kept_pages = [pg for pg in pages if (pg.get("point_ids") or []) and all(in_scope.get(str(x)) for x in pg["point_ids"])]
        hints_scope["pages_dropped"] = len(pages) - len(kept_pages)
        pages = kept_pages
        kept_entities = [e for e in entities if (e.get("docs") or []) and set(map(str, e["docs"])) <= allowed_paths]
        hints_scope["entities_dropped"] = len(entities) - len(kept_entities)
        entities = kept_entities
        hints_scope["relations_omitted"] = len(relations)
        relations = []
    specs = spec_rows(specs, source_ns=source_ns, limit_hints=ss.spec_hint_limit, point_active=point_active)
    subjects = list(buckets["keys"]) if buckets and buckets.get("mode") == "subject" else []
    pages = page_rows(pages, subjects=subjects, text_budget_tokens=ss.page_text_tokens, point_active=point_active)[:6]
    if evidence_state != "accepted":
        for row in specs + pages:
            row["state"] = evidence_state
    rerank_max = max((float(c.get("score_rerank") or 0.0) for c in cands), default=0.0) if reranked else None
    summary = {
        "kbs": chosen, "routing": routing, "identifiers": identifiers, "image_query": image_query,
        "channels": {kb: per_kb[kb]["channels"] for kb in chosen}, "graph_versions": graph_versions,
        "merged": merged_n, "reranked": len(cands) if reranked else 0,
        "rerank": rerank_status, "rerank_max": rerank_max, "rerank_floor": ranked.get("floor"), "visual": profile.get("visual"),
        "buckets": buckets, "selection": selection, "downweighted": downweighted,
        "degraded": degraded,
        "evidence_state": evidence_state,
        "low_confidence": bool(degraded) or evidence_state != "accepted",
        "no_relevant_content": (not hits) or evidence_state == "diagnostic",
        "sources": {**src_stats, "page_tokens": sum(int(r.get("_tokens") or 0) for r in pages)},
        "timings_ms": {**timings, "total_ms": int((time.time() - t0) * 1000)},
        "hints": hints or {}, "hints_applied": bool(hint_used), "hints_used": hint_used, "hints_ignored": hint_ignored, "hints_scope": hints_scope,
    }
    if not explain:
        for r in sources:
            r.pop("entities", None); r.pop("relations", None)
    return {
        "question": question, "kbs": chosen,
        "sources": sources, "doc_aggs": doc_aggs(sources),
        "entities": numbered(entities[:24], ENTITY_FIELDS + ("kb_id",)),
        "relationships": numbered(relations[:24], RELATION_FIELDS + ("kb_id",)),
        "specs": numbered(specs, SPEC_FIELDS + ("state", "kb_id")),
        "pages": numbered(pages, PAGE_FIELDS + ("state", "kb_id")),
        "retrieval_summary": summary,
    }


def context(kb_id: str, doc_id: str, chunk_from: int, chunk_to: int, *, content_version: str | None = None) -> dict[str, Any]:
    settings, ss, q = runtime()
    if kb_id not in settings.sources:
        raise KeyError(kb_id)
    if chunk_to < chunk_from or chunk_to - chunk_from > 60:
        raise ValueError("chunk range must be ascending and at most 60 chunks")
    rows = channels.context_range(q, settings.sources[kb_id].collection, doc_id=doc_id, chunk_from=chunk_from, chunk_to=chunk_to,
                                  content_version=content_version)
    from .evidence import source_row

    sources = [source_row(i, {"point_id": r.get("point_id"), "kb_id": kb_id}, r, role="context") for i, r in enumerate(rows, 1)]
    return {"kb_id": kb_id, "doc_id": doc_id, "sources": sources, "tokens_total": sum(int(s["token_count"]) for s in sources)}


def image(kb_id: str, point_id: str) -> dict[str, Any]:
    """Original image bytes of an image chunk (Q26): the bitmap embedded in the PDF first, then a
    high-resolution render of the page bbox, and only last the image in the parse cache."""
    from . import images

    settings, ss, q = runtime()
    if kb_id not in settings.sources:
        raise KeyError(kb_id)
    payload = images.locate(q, settings.sources[kb_id].collection, point_id)
    return images.original_image(settings, payload)


def crop(kb_id: str, point_id: str, bbox: list[float], *, pad: int = 16) -> dict[str, Any]:
    """Crop the original image along the caller's box (Q25): the box may be 0-1 fractions or 0-1000
    per-mille; the DGX does a deterministic crop only, no recognition of any kind."""
    from . import images

    settings, ss, q = runtime()
    if kb_id not in settings.sources:
        raise KeyError(kb_id)
    payload = images.locate(q, settings.sources[kb_id].collection, point_id)
    full = images.original_image(settings, payload)
    return images.crop_image(full, bbox, pad=pad)


def graph_neighbors(kb_id: str, *, entity: str | None = None, entity_id: str | None = None, limit: int = 20,
                    types: list[str] | None = None, direction: str = "both") -> dict[str, Any]:
    """Graph neighbourhood (for the agent to walk the graph itself): one entity's one-hop relations with
    predicate, direction, weight, counterpart and evidence chunks; a knowledge base without a graph
    answers 404."""
    from . import graphwalk

    settings, ss, q = runtime()
    if kb_id not in settings.sources:
        raise KeyError(kb_id)
    if not entity and not entity_id:
        raise ValueError("entity or entity_id is required")
    return graphwalk.neighbors(settings, settings.sources[kb_id], entity=entity, entity_id=entity_id, limit=limit, types=types, direction=direction, q=q)

