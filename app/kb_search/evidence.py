"""Evidence assembly (Q07 / Q08 / Q11 / Q12 / Q16 / Q18 / Q21): hits carry a position string and scores,
short hits are stitched with neighbours from the same document, table continuation chunks bring their
header, the neighbourhood is backfilled by chunk index without crossing document boundaries, overlaps
are deduped, and the token budget accumulates by token_count; numbered tables Sources / Entities /
Relationships / Specs / Pages, facts carry a hint string and source numbers, series are merged,
conflicts are marked, and compiled pages are labelled separately from source chunks; the client cites
by number."""
from __future__ import annotations

from typing import Any, Callable

from kb_pipeline.utils import count_tokens

from .text import adjacent, body_token_set, dedupe_overlaps, is_boilerplate, jaccard, place, position, same_facts, scope_of, token_set, truncate_tokens

EXCERPT_TOKENS = 80          # length of the excerpt kept for a hit over budget (with its position; the full text comes later via /context)

VISUAL_TEXT_LIMIT = 600
VISUAL_FACTS_LIMIT = 12
STITCHABLE_BLOCKS = {"text", "slide", "", "None"}


def visual_block(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Visual evidence of an image / chart chunk: description, in-image text, facts, confidence, conflict
    marker (Q17); returns None for a chunk that is not an image."""
    src = str(payload.get("embedding_text_source") or "")
    if not payload.get("visual_ref") and not src.startswith("visual") and src != "text_plus_visual_summary":
        return None
    conflicts = payload.get("visual_value_conflicts") or []
    return {
        "kind": payload.get("visual_kind"), "confidence": payload.get("visual_confidence"),
        "summary": payload.get("visual_summary"), "text": str(payload.get("visual_text") or "")[:VISUAL_TEXT_LIMIT],
        "facts": list(payload.get("visual_facts") or [])[:VISUAL_FACTS_LIMIT],
        "ref": payload.get("visual_ref"), "sha256": payload.get("visual_sha256"),
        "value_conflicts": len(conflicts),
        "note": "The text in the figure conflicts with the estimated reading; the figure text takes precedence" if conflicts else None,
    }


def source_row(n: int, cand: dict[str, Any], payload: dict[str, Any], *, role: str = "hit", of: int | None = None) -> dict[str, Any]:
    text = str(payload.get("text") or "")
    row = {
        "n": n, "role": role, "of": of,
        "kb_id": cand.get("kb_id") or payload.get("kb_id"), "point_id": cand.get("point_id") or payload.get("point_id"),
        "chunk_uid": payload.get("chunk_uid"), "doc_id": payload.get("doc_id"), "content_version": payload.get("content_version"),
        "chunk_index": payload.get("chunk_index"), "chunk_total": payload.get("chunk_total"),
        "doc": payload.get("filename"), "rel_path": payload.get("rel_path"), "doc_type": payload.get("doc_type"),
        "position": position(payload), "place": place(payload), "page_idx": payload.get("page_idx"),
        "section_path": list(payload.get("section_path") or []),
        "block_type": payload.get("block_type"), "block_id": payload.get("block_id"), "text": text,
        "token_count": int(payload.get("token_count") or 0) or count_tokens(text),
        "scores": dict(cand.get("scores") or {}), "recall_sources": list(cand.get("recall_sources") or []),
        "entities": list(cand.get("entities") or [])[:8], "relations": list(cand.get("relations") or [])[:6],
        "visual": visual_block(payload),
        "boilerplate": bool(cand.get("boilerplate")) or is_boilerplate(payload),
        "bucket": cand.get("bucket"),
        "degraded": payload.get("degraded"),      # parse-time degradation reason (e.g. text_layer_cjk_lost), distinct from the channel-level degraded list
        "accepted": cand.get("accepted"),
    }
    for key in ("score_rerank", "score_final"):
        if cand.get(key) is not None:
            row["scores"][key] = cand[key]
    return row


def stitch_short_hit(row: dict[str, Any], neighbors: list[dict[str, Any]], *, min_chars: int, max_chars: int) -> bool:
    """When a hit is shorter than min_chars, stitch neighbours along the document's chunk index in both
    directions (next first, then previous, alternating), stopping once >= min_chars, with the total
    never exceeding max_chars; only body-type chunks are stitched (tables follow the header rule,
    images already come with their paragraph neighbours). Rewrites the row's text / token_count and
    records the stitched range."""
    if str(row.get("block_type") or "") not in STITCHABLE_BLOCKS:
        return False
    text = str(row.get("text") or "")
    if len(text) >= min_chars:
        return False
    try:
        idx = int(row.get("chunk_index"))
    except (TypeError, ValueError):
        return False
    prev = sorted((p for p in neighbors if int(p.get("chunk_index", -1)) < idx and str(p.get("block_type") or "") in STITCHABLE_BLOCKS),
                  key=lambda p: -int(p["chunk_index"]))
    nxt = sorted((p for p in neighbors if int(p.get("chunk_index", -1)) > idx and str(p.get("block_type") or "") in STITCHABLE_BLOCKS),
                 key=lambda p: int(p["chunk_index"]))
    pieces: dict[int, dict[str, Any]] = {idx: {"text": text, "point_id": row.get("point_id"), "chunk_index": idx,
                                               "page_idx": row.get("page_idx"), "position": row.get("position"),
                                               "degraded": row.get("degraded")}}
    total = len(text)
    order = []
    for i in range(max(len(prev), len(nxt))):
        if i < len(nxt):
            order.append(nxt[i])
        if i < len(prev):
            order.append(prev[i])
    for p in order:
        if total >= min_chars:
            break
        t = str(p.get("text") or "")
        if not t or total + len(t) + 1 > max_chars:
            continue
        pieces[int(p["chunk_index"])] = {"text": t, "point_id": p.get("point_id"), "chunk_index": int(p["chunk_index"]),
                                         "page_idx": p.get("page_idx"), "position": position(p), "degraded": p.get("degraded")}
        total += len(t) + 1
    if len(pieces) == 1:
        return False
    keys = sorted(pieces)
    row["text"] = "\n".join(pieces[k]["text"] for k in keys)
    row["token_count"] = count_tokens(row["text"])
    # Every piece can be cited on its own (Codex S03): a stitched-in neighbour carries its own point_id
    # / chunk index / page number / position string; when a neighbour sits on a page whose parse was
    # degraded, its text is now part of this row, so the degraded marker is carried along with it
    row["stitched"] = {"chunk_from": keys[0], "chunk_to": keys[-1], "own_chars": len(text),
                       "pieces": [{k: v for k, v in pieces[i].items() if k != "text" and (k != "degraded" or v)} | {"chars": len(pieces[i]["text"])}
                                  for i in keys]}
    row["degraded"] = row.get("degraded") or next((pieces[i]["degraded"] for i in keys if pieces[i].get("degraded")), None)
    return True


def dedupe_against(base: list[dict[str, Any]], extra: list[dict[str, Any]], *, threshold: float, key: str = "text") -> tuple[list[dict[str, Any]], int]:
    """A neighbour / header chunk overlapping a hit (or a previously kept neighbour) at or above the
    threshold is dropped; a hit never yields (it is the ranked evidence)."""
    from .text import overlap_ratio

    kept: list[dict[str, Any]] = []
    dropped = 0
    for row in extra:
        text = str(row.get(key) or "")
        scope = scope_of(row)
        if any(scope_of(other) == scope and adjacent(row, other) and overlap_ratio(text, str(other.get(key) or "")) >= threshold
               and same_facts(text, str(other.get(key) or "")) for other in base + kept):
            dropped += 1
            continue
        kept.append(row)
    return kept, dropped


def _trim_pieces(row: dict[str, Any]) -> None:
    """After a stitched chunk is truncated, mark in pieces which segments are still in the text (kept)
    and which one was cut in half (partial)."""
    st = row.get("stitched")
    if not st or not st.get("pieces"):
        return
    kept_len = len(str(row.get("text") or ""))
    offset = 0
    for pc in st["pieces"]:
        chars = int(pc.get("chars") or 0)
        start, end = offset, offset + chars
        pc["kept"] = start < kept_len
        pc["partial"] = start < kept_len < end
        offset = end + 1              # one newline between segments
    st["truncated"] = True


def assemble_sources(hits: list[dict[str, Any]], *, budget_tokens: int,
                     neighbors: Callable[[dict[str, Any], int], list[dict[str, Any]]] | None,
                     table_head: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
                     stitch: tuple[int, int] | None = None, neighbor_span: int = 1,
                     overlap_threshold: float = 0.85) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """hits are already ranked and each carries a payload. Hits are first deduped by text and all
    numbered first; short hits are then stitched with neighbours (Q16; the material excludes other hits,
    chunks a higher-ranked hit already stitched in, and boilerplate chunks); table continuation chunks
    bring their header chunk; neighbours come after, ordered by distance, backfilled only within budget,
    and dropped when a stitched hit already shows them or they overlap a hit. Returns (sources, stats)."""
    rows: list[dict[str, Any]] = []
    for i, cand in enumerate(hits, 1):
        rows.append(source_row(i, cand, cand.get("payload") or {}))
    rows, dropped_hits = dedupe_overlaps(rows, threshold=overlap_threshold)
    stats = {"hits": len(rows), "hit_tokens": 0, "neighbors": 0, "table_heads": 0, "stitched": 0, "truncated_hits": 0, "dropped_overlaps": dropped_hits, "budget_tokens": budget_tokens}
    hit_ids = {str(r.get("point_id")) for r in rows}
    taken = set(hit_ids)            # a chunk goes into one hit only: two short hits close together would otherwise both stitch it in
    neighbor_cache: dict[str, list[dict[str, Any]]] = {}
    if neighbors is not None and stitch:
        span = max(neighbor_span, 2)
        for r in rows:
            if str(r.get("block_type") or "") in STITCHABLE_BLOCKS and len(str(r.get("text") or "")) < stitch[0]:
                nbs = neighbors(r, span)
                neighbor_cache[str(r["point_id"])] = nbs
                material = [p for p in nbs if str(p.get("point_id")) not in taken and not is_boilerplate(p)]
                if stitch_short_hit(r, material, min_chars=stitch[0], max_chars=stitch[1]):
                    stats["stitched"] += 1
                    taken.update(str(pc.get("point_id")) for pc in r["stitched"]["pieces"])
    # Hard budget boundary (Codex S07 / R3): every hit first reserves a minimum excerpt allowance (the
    # smaller of 80 tokens and budget / count), the rest is handed out in order; the total never exceeds
    # the budget, and a hit that overflows is cut to an excerpt with its position and marked
    # text_truncated, with the full text fetched later via /context
    used = 0
    n_rows = len(rows)
    floor = min(EXCERPT_TOKENS, budget_tokens // n_rows) if n_rows else 0
    for i, r in enumerate(rows):
        n = int(r["token_count"])
        allowance = max(floor, budget_tokens - used - floor * (n_rows - i - 1))
        if n <= allowance:
            used += n
            continue
        r["full_tokens"] = n
        r["text"] = truncate_tokens(str(r.get("text") or ""), allowance)
        r["token_count"] = count_tokens(r["text"])
        r["text_truncated"] = True
        _trim_pieces(r)
        used += int(r["token_count"])
        stats["truncated_hits"] = stats.get("truncated_hits", 0) + 1
    stats["hit_tokens"] = used
    extra: list[dict[str, Any]] = []
    # a chunk whose text a stitched hit still shows is not offered again: the overlap check below would not always
    # catch it, since joining two pieces can run their numbers together
    seen = hit_ids | {str(pc.get("point_id")) for r in rows for pc in (r.get("stitched") or {}).get("pieces") or [] if pc.get("kept", True)}
    if table_head is not None:
        for r in rows:
            if str(r.get("block_type") or "") != "table":
                continue
            head = table_head(r)
            if head and str(head.get("point_id")) not in seen:
                hb = source_row(0, {"point_id": str(head.get("point_id")), "kb_id": r.get("kb_id")}, head, role="table_head", of=r["n"])
                hb["distance"] = 0
                extra.append(hb)
                seen.add(str(head.get("point_id")))
    if neighbors is not None:
        for r in rows:
            nbs = neighbor_cache.get(str(r["point_id"]))
            if nbs is None:
                nbs = neighbors(r, neighbor_span)
            for pl in nbs:
                pid = str(pl.get("point_id"))
                if pid in seen:
                    continue
                if abs(int(pl.get("chunk_index") or 0) - int(r.get("chunk_index") or 0)) > neighbor_span:
                    continue
                nb = source_row(0, {"point_id": pid, "kb_id": r.get("kb_id")}, pl, role="neighbor", of=r["n"])
                nb["distance"] = abs(int(pl.get("chunk_index") or 0) - int(r.get("chunk_index") or 0))
                extra.append(nb)
                seen.add(pid)
    if extra:
        extra.sort(key=lambda x: (x["distance"], x["of"]))
        kept_extra, dropped_nb = dedupe_against(rows, extra, threshold=overlap_threshold)
        stats["dropped_overlaps"] += dropped_nb
        picked: list[dict[str, Any]] = []
        for nb in kept_extra:
            if used + int(nb["token_count"]) > budget_tokens:
                continue
            used += int(nb["token_count"])
            picked.append(nb)
        rows = rows + picked
        stats["neighbors"] = sum(1 for r in picked if r["role"] == "neighbor")
        stats["table_heads"] = sum(1 for r in picked if r["role"] == "table_head")
    old_to_new: dict[int, int] = {}
    for i, r in enumerate(rows, 1):
        if r["role"] == "hit":
            old_to_new[int(r["n"])] = i
        r["n"] = i
    for r in rows:
        if r["role"] != "hit" and r.get("of") is not None:
            r["of"] = old_to_new.get(int(r["of"]), r["of"])
    stats["tokens_total"] = used
    return rows, stats


def select_hits(ordered: list[dict[str, Any]], k: int, *, score_key: str = "score_final", buckets: dict[str, Any] | None = None,
                min_hits: int = 2, mmr_lambda: float | None = 0.7, priority_key: str | None = None,
                priority_n: int = 0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Cut to top-k (Q08 / Q16): first reserve priority_n slots by priority_key (the visual score of a
    request carrying an image, Codex S02), then reserve min_hits per subject / document bucket (taken
    in rank order), and finally fill up with MMR (lambda * relevance - (1 - lambda) * max Jaccard with
    the already chosen); with mmr_lambda None, fill in rank order. Returns (hits, {avg_redundancy,
    quota_filled, priority_filled})."""
    k = max(0, int(k))
    picked: list[dict[str, Any]] = []
    chosen_ids: set[int] = set()
    filled: dict[str, int] = {}
    priority_filled = 0
    if priority_key and priority_n > 0:
        pri = sorted((c for c in ordered if (c.get("scores") or {}).get(priority_key) is not None),
                     key=lambda c: -float((c.get("scores") or {}).get(priority_key) or 0.0))
        for c in pri[:priority_n]:
            if len(picked) >= k:
                break
            picked.append(c)
            chosen_ids.add(id(c))
            priority_filled += 1
    if buckets and buckets.get("keys"):
        for key in buckets["keys"]:
            n = 0
            for c in ordered:
                if len(picked) >= k:
                    break
                if n >= min_hits:
                    break
                if id(c) in chosen_ids or c.get("bucket") != key:
                    continue
                picked.append(c)
                chosen_ids.add(id(c))
                n += 1
            filled[key] = n
    rest = [c for c in ordered if id(c) not in chosen_ids]
    if mmr_lambda is None or not rest:
        for c in rest:
            if len(picked) >= k:
                break
            picked.append(c)
    else:
        lam = float(mmr_lambda)
        scores = [float(c.get(score_key) or 0.0) for c in rest]
        top = max(scores) if scores else 0.0
        rel = [s / top if top > 0 else 0.0 for s in scores]
        toks = [body_token_set(str((c.get("payload") or {}).get("text") or "")) for c in rest]
        chosen_toks = [body_token_set(str((c.get("payload") or {}).get("text") or "")) for c in picked]
        remaining = list(range(len(rest)))
        # Each candidate's highest similarity to the entries already picked: every newly picked entry is compared
        # once, instead of recomputing against all picked entries in every round
        sim = [max((jaccard(toks[i], t) for t in chosen_toks), default=0.0) for i in remaining]
        while remaining and len(picked) < k:
            best_i = max(remaining, key=lambda i: lam * rel[i] - (1.0 - lam) * sim[i])
            picked.append(rest[best_i])
            remaining.remove(best_i)
            for i in remaining:
                sim[i] = max(sim[i], jaccard(toks[i], toks[best_i]))
    toks_all = [body_token_set(str((c.get("payload") or {}).get("text") or "")) for c in picked]
    pairs = [(i, j) for i in range(len(toks_all)) for j in range(i + 1, len(toks_all))]
    redundancy = round(sum(jaccard(toks_all[i], toks_all[j]) for i, j in pairs) / len(pairs), 4) if pairs else 0.0
    # Quotas and MMR decide "which ones"; the presentation order still follows the final score: a
    # quota pick must not be listed ahead of a more relevant one
    picked.sort(key=lambda c: -float(c.get(score_key) or 0.0))
    return picked, {"avg_redundancy": redundancy, "quota_filled": filled, "priority_filled": priority_filled}


def doc_aggs(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate by document (the shape of RAGFlow's doc_aggs): hits per document, best rank, source
    knowledge base."""
    agg: dict[tuple[str, str], dict[str, Any]] = {}
    for r in sources:
        if r.get("role") != "hit":
            continue
        key = (str(r.get("kb_id")), str(r.get("doc_id")))
        slot = agg.setdefault(key, {"kb_id": r.get("kb_id"), "doc_id": r.get("doc_id"), "doc": r.get("doc"), "rel_path": r.get("rel_path"),
                                    "hits": 0, "best_n": r["n"], "source_ns": []})
        slot["hits"] += 1
        slot["source_ns"].append(r["n"])
    return sorted(agg.values(), key=lambda a: (a["best_n"], -a["hits"]))


def numbered(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    """Entity / relation / fact tables: numbered in order, keeping only the fields the client cites."""
    out = []
    for i, r in enumerate(rows, 1):
        item = {"n": i}
        for f in fields:
            if r.get(f) is not None:
                item[f] = r[f]
        out.append(item)
    return out


def _value_repr(sp: dict[str, Any]) -> str:
    unit = str(sp.get("unit_canonical") or sp.get("unit") or "").strip()
    if sp.get("value") not in (None, ""):
        body = str(sp["value"])
    else:
        lo, typ, hi = sp.get("min"), sp.get("typ"), sp.get("max")
        parts = []
        if lo not in (None, ""):
            parts.append(f"min {lo}")
        if typ not in (None, ""):
            parts.append(f"typ {typ}")
        if hi not in (None, ""):
            parts.append(f"max {hi}")
        body = " / ".join(parts)
    if not body:
        return ""
    return f"{body} {unit}".strip() if unit and unit.casefold() not in body.casefold() else body


def spec_hint(sp: dict[str, Any]) -> str | None:
    """Hint string of one fact (Q11): "subject · property(symbol) = value unit @ conditions · time"; a
    fact with an unclear source (ambiguous_source) gets no hint."""
    if str(sp.get("quality") or "") == "ambiguous_source":
        return None
    value = _value_repr(sp)
    if not value:
        return None
    head = " · ".join(x for x in (str(sp.get("subject") or "").strip(), str(sp.get("property") or sp.get("concept") or "").strip()) if x)
    if sp.get("symbol"):
        head += f"({sp['symbol']})"
    tail = []
    cond = str(sp.get("conditions_text") or "").strip()
    if not cond and isinstance(sp.get("conditions"), dict) and sp["conditions"]:
        cond = ", ".join(f"{k}={v}" for k, v in sp["conditions"].items())
    if cond:
        tail.append(f"@ {cond}")
    when = str(sp.get("when") or sp.get("valid_from") or "").strip()
    if when:
        tail.append(f"· {when}")
    return f"{head} = {value}" + (" " + " ".join(tail) if tail else "")


def spec_rows(specs: list[dict[str, Any]], *, source_ns: dict[str, int], limit_hints: int = 8,
              point_active: dict[str, bool] | None = None) -> list[dict[str, Any]]:
    """Fact table (Q11): with hint string, source chunk numbers, the other values of the same series
    ("30(2024), 31(2025)") and conflict markers; numeric filtering trusts only fields whose kinds is
    scalar, and kinds is passed through as-is here. point_active: whether the source points are still
    active (Codex S03): a fact whose source points are all deactivated is marked verified=False and
    gets no hint, rather than silently passing as a current fact. Conclusions rest only on source points
    that were actually checked: when none could be checked (the main store did not return them) the
    answer is "unknown", not "no longer valid"."""
    series: dict[str, list[dict[str, Any]]] = {}
    for sp in specs:
        key = str(sp.get("series_key") or "")
        if key:
            series.setdefault(key, []).append(sp)
    out: list[dict[str, Any]] = []
    hints_given = 0
    for sp in specs:
        row = dict(sp)
        pids = [str(p) for p in (sp.get("point_ids") or [])]
        row["sources"] = sorted({source_ns[p] for p in pids if p in source_ns})
        checked = [p for p in pids if p in point_active or p in source_ns] if point_active is not None else []
        if checked:
            active = [p for p in checked if point_active.get(p, True)]
            row["sources_active"] = f"{len(active)}/{len(checked)}"
            row["verified"] = bool(active)
        else:
            row["verified"] = None if not pids else bool(row["sources"]) or None
        hint = spec_hint(sp) if hints_given < limit_hints and row.get("verified") is not False else None
        if hint:
            hints_given += 1
        row["hint"] = hint
        key = str(sp.get("series_key") or "")
        if key and len(series.get(key, [])) > 1:
            members = sorted(series[key], key=lambda x: (str(x.get("valid_from") or x.get("when") or ""), int(x.get("series_index") or 0)))
            row["series_text"] = "、".join(f"{_value_repr(m)}({m.get('when') or m.get('valid_from') or '?'})" for m in members)
        conflict = bool(sp.get("conflict_group")) or bool(sp.get("evidence_conflict"))
        row["conflict"] = conflict
        if conflict:
            row["conflict_note"] = "Conflicting records for the same property are listed in groups rather than merged silently"
        out.append(row)
    return out


def page_rows(pages: list[dict[str, Any]], *, subjects: list[str], text_budget_tokens: int,
              point_active: dict[str, bool] | None = None) -> list[dict[str, Any]]:
    """Compiled page table (Q12): pages of the named subjects come first; the body text is given only
    for timeline pages and subject pages with series rows, cut to the total budget, the rest get only
    the overview; every row is marked compiled=True as a reminder that it is second-hand knowledge,
    and answers cite pages together with chunks. A page whose source points are all deactivated
    (verified=False) gets no body text: it was compiled from content that has since been deleted or
    replaced, so only the overview is kept as a lead."""
    subj = [s.casefold() for s in subjects if s]

    def named(pg: dict[str, Any]) -> int:
        title = str(pg.get("title") or "").casefold()
        return 0 if any(s and s in title for s in subj) else 1

    ordered = sorted(pages, key=lambda pg: (named(pg), -float(pg.get("score") or 0.0)))
    used = 0
    out: list[dict[str, Any]] = []
    for pg in ordered:
        row = {k: v for k, v in pg.items() if k != "text"}
        row["compiled"] = True
        checked = [str(p) for p in (pg.get("point_ids") or []) if str(p) in point_active] if point_active is not None else []
        if checked:                                 # conclusions rest only on checked source points; a page none of whose points were checked is not marked
            active = [p for p in checked if point_active[p]]
            row["sources_active"] = f"{len(active)}/{len(checked)}"
            row["verified"] = bool(active)
        kind = str(pg.get("kind") or "")
        text = str(pg.get("text") or "")
        wants_text = text and row.get("verified") is not False and (kind == "timeline" or (kind == "subject" and (pg.get("series") or [])))
        if wants_text:
            n = count_tokens(text)
            if used + n <= text_budget_tokens:
                row["text"] = text
                row["_tokens"] = n
                used += n
            else:
                lines = text.splitlines()
                kept: list[str] = []
                for line in lines:
                    t = count_tokens(line)
                    if used + t > text_budget_tokens:
                        break
                    kept.append(line)
                    used += t
                row["text"] = "\n".join(kept) if kept else None
                row["text_truncated"] = True
                row["_tokens"] = count_tokens(row["text"] or "")
        out.append(row)
    return out
