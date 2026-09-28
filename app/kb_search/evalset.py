"""Retrieval regression evaluation (Q10): question set JSON -> service.search per question -> hit@k /
MRR / expect / min_docs / negative rejection / latency, with the results written to JSON and diffed
against the previous run. Question set generation (make-set) runs only during development: random body
chunks are drawn from the knowledge base, the extraction model configured for that knowledge base
writes one question and one verbatim answer per chunk, and the gold standard is that chunk's point_id
and document; it never enters the retrieval service."""
from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path
from typing import Any, Callable

HIT_KS = (3, 5, 12)


def load_set(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data.get("items") if isinstance(data, dict) else data
    out = []
    for it in items or []:
        if isinstance(it, dict) and (it.get("q") or it.get("question")):
            it = dict(it)
            it["q"] = str(it.get("q") or it.get("question"))
            out.append(it)
    return out


def _contains(hay: str, needle: str) -> bool:
    return str(needle or "").casefold().replace(" ", "") in hay


def evaluate_item(item: dict[str, Any], out: dict[str, Any]) -> dict[str, Any]:
    """Judgement of one question: rank of the gold point_id / document, whether expect (any) /
    expect_all (all) occur in the evidence text, whether the number of documents hit is >= min_docs,
    and whether a negative was judged low confidence."""
    sources = out.get("sources") or []
    hits = [r for r in sources if r.get("role") == "hit"]
    summary = out.get("retrieval_summary") or {}
    texts = [str(r.get("text") or "") for r in sources]
    texts += [str(sp.get("hint") or "") + " " + str(sp.get("series_text") or "") + " " + str(sp.get("text") or "") for sp in out.get("specs") or []]
    texts += [str(pg.get("text") or "") + " " + str(pg.get("summary") or "") for pg in out.get("pages") or []]
    hay = " ".join(texts).casefold().replace(" ", "")
    res: dict[str, Any] = {"q": item["q"], "tag": item.get("tag"), "kbs": out.get("kbs"), "hits": len(hits),
                           "rerank": summary.get("rerank"), "rerank_max": summary.get("rerank_max"),
                           "low_confidence": summary.get("low_confidence"), "total_ms": (summary.get("timings_ms") or {}).get("total_ms")}
    gold_pids = {str(x) for x in (item.get("gold_point_ids") or [])}
    gold_docs = {str(x) for x in (item.get("gold_docs") or [])}
    point_rank = next((i for i, r in enumerate(hits, 1) if gold_pids and str(r.get("point_id")) in gold_pids), None)
    doc_rank = next((i for i, r in enumerate(hits, 1) if gold_docs and (str(r.get("rel_path")) in gold_docs or str(r.get("doc")) in gold_docs)), None)
    # Chunk gold and document gold are computed separately (Codex S08): hit@k by chunk, doc_hit@k by
    # document; for a question with only document gold, hit@k is on the document basis and marked as such
    if gold_pids:
        res["rank"], res["basis"] = point_rank, "point"
    elif gold_docs:
        res["rank"], res["basis"] = doc_rank, "doc"
    if gold_pids or gold_docs:
        rank = res["rank"]
        res["rr"] = round(1.0 / rank, 4) if rank else 0.0
        for k in HIT_KS:
            res[f"hit@{k}"] = bool(rank and rank <= k)
    if gold_docs:
        res["doc_rank"] = doc_rank
        for k in HIT_KS:
            res[f"doc_hit@{k}"] = bool(doc_rank and doc_rank <= k)
    expect = item.get("expect") or []
    if isinstance(expect, str):
        expect = [expect]
    expect_all = item.get("expect_all")
    if expect:
        res["expect"] = any(_contains(hay, e) for e in expect)
    if isinstance(expect_all, list) and expect_all:
        res["expect_all"] = all(_contains(hay, e) for e in expect_all)
    elif expect_all is True and expect:                       # the 09-08 cross-document set's notation: expect_all: true = every entry of expect must occur
        res["expect_all"] = all(_contains(hay, e) for e in expect)
    if item.get("min_docs"):
        docs = {str(r.get("doc_id")) for r in hits}
        res["docs"] = len(docs)
        res["min_docs"] = len(docs) >= int(item["min_docs"])
    if item.get("negative"):
        res["negative"] = bool(summary.get("rerank") == "below_threshold" or summary.get("no_relevant_content") or
                               (summary.get("rerank_max") is not None and float(summary.get("rerank_max") or 0) < 0.1))
    return res


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def rate(key: str) -> tuple[int, int]:
        have = [r for r in rows if key in r]
        return sum(1 for r in have if r[key]), len(have)

    out: dict[str, Any] = {"n": len(rows), "errors": sum(1 for r in rows if r.get("error"))}
    for k in HIT_KS:
        got, tot = rate(f"hit@{k}")
        out[f"hit@{k}"] = f"{got}/{tot}" if tot else None
    for k in HIT_KS:
        got, tot = rate(f"doc_hit@{k}")
        out[f"doc_hit@{k}"] = f"{got}/{tot}" if tot else None
    ranked = [r for r in rows if "rr" in r]
    out["mrr"] = round(sum(float(r["rr"]) for r in ranked) / len(ranked), 4) if ranked else None
    for key in ("expect", "expect_all", "min_docs", "negative"):
        got, tot = rate(key)
        out[key] = f"{got}/{tot}" if tot else None
    lat = [int(r["total_ms"]) for r in rows if r.get("total_ms") is not None]
    out["avg_ms"] = int(sum(lat) / len(lat)) if lat else None
    out["low_confidence"] = sum(1 for r in rows if r.get("low_confidence"))
    return out


def evaluate(items: list[dict[str, Any]], search_fn: Callable[..., dict[str, Any]], *, kbs: list[str] | None = None,
             top_k: int = 12, auto: bool = False) -> dict[str, Any]:
    """With auto=True the question's own kbs are ignored and the service routes by itself (which also
    tests routing); otherwise the question's kbs take precedence, then those given on the command line."""
    rows = []
    for it in items:
        try:
            out = search_fn(it["q"], kbs=None if auto else (it.get("kbs") or kbs), top_k=top_k)
            rows.append(evaluate_item(it, out))
        except Exception as exc:
            # A question that errored counts in the denominator as "not answered" (Codex S08); it must
            # not be quietly dropped from the statistics
            row: dict[str, Any] = {"q": it["q"], "tag": it.get("tag"), "error": f"{type(exc).__name__}: {str(exc)[:160]}"}
            if it.get("gold_point_ids") or it.get("gold_docs"):
                row["rank"], row["rr"] = None, 0.0
                for k in HIT_KS:
                    row[f"hit@{k}"] = False
            if it.get("gold_docs"):
                for k in HIT_KS:
                    row[f"doc_hit@{k}"] = False
            for key in ("expect", "expect_all", "min_docs", "negative"):
                if it.get(key):
                    row[key] = False
            rows.append(row)
    return {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "top_k": top_k, "summary": summarize(rows), "items": rows}


def _num(s: Any) -> float | None:
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    m = re.match(r"^(\d+)/(\d+)$", str(s))
    if m and int(m.group(2)):
        return int(m.group(1)) / int(m.group(2))
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def compare(current: dict[str, Any], previous: dict[str, Any] | None) -> list[tuple[str, Any, Any, str]]:
    """Metrics table: name, this run, previous run, delta (computed only when a previous run exists)."""
    cur, prev = current.get("summary") or {}, (previous or {}).get("summary") or {}
    rows = []
    for key in ("n", "errors", "hit@3", "hit@5", "hit@12", "doc_hit@3", "doc_hit@5", "doc_hit@12", "mrr", "expect", "expect_all", "min_docs", "negative", "avg_ms", "low_confidence"):
        a, b = cur.get(key), prev.get(key)
        na, nb = _num(a), _num(b)
        delta = "" if na is None or nb is None else f"{na - nb:+.3f}"
        rows.append((key, a, b, delta))
    return rows


def format_table(rows: list[tuple[str, Any, Any, str]]) -> str:
    lines = [f"{'metric':16s} {'now':>10s} {'prev':>10s} {'delta':>9s}"]
    for name, a, b, d in rows:
        lines.append(f"{name:16s} {str(a if a is not None else '-'):>10s} {str(b if b is not None else '-'):>10s} {d:>9s}")
    return "\n".join(lines)


def run_eval(set_path: str, *, out_path: str | None = None, kbs: list[str] | None = None, top_k: int = 12,
             compare_path: str | None = None, search_fn: Callable[..., dict[str, Any]] | None = None, auto: bool = False) -> dict[str, Any]:
    if search_fn is None:
        from . import service

        search_fn = lambda q, kbs=None, top_k=12: service.search(q, kbs=kbs, top_k=top_k)   # noqa: E731
    items = load_set(set_path)
    result = evaluate(items, search_fn, kbs=kbs, top_k=top_k, auto=auto)
    result["set"] = str(set_path)
    result["auto_routing"] = bool(auto)
    prev = None
    prev_path = Path(compare_path) if compare_path else (Path(out_path) if out_path else None)
    if prev_path and prev_path.exists():
        try:
            prev = json.loads(prev_path.read_text(encoding="utf-8"))
        except Exception:
            prev = None
    result["compared_to"] = str(prev_path) if prev else None
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    result["table"] = format_table(compare(result, prev))
    return result


# ── Question set generation (development only) ─────────────────────────────────

MAKE_PROMPT = """Below is a passage from a knowledge base. Write one specific question a user might ask that this passage answers, in the same language as the passage. Do not refer to "this passage" or "the text above"; include the concrete names, identifiers or concepts that locate the passage. The answer must be findable verbatim in the passage. Then give the answer as a verbatim excerpt (at most 40 characters, identical to the passage: no rewording, no added punctuation). Output exactly one JSON object: {{"question": "...", "answer": "..."}}

Passage (from file {filename}):
{text}
"""


def _sample_chunks(q: Any, collection: str, n: int, *, seed: int, min_chars: int = 300) -> list[dict[str, Any]]:
    from qdrant_client import models

    flt = models.Filter(must=[models.FieldCondition(key="is_active", match=models.MatchValue(value=True)),
                              models.FieldCondition(key="block_type", match=models.MatchAny(any=["text", "table", "slide"]))])
    pool: list[dict[str, Any]] = []
    offset = None
    while True:
        points, offset = q.scroll(collection_name=collection, scroll_filter=flt, limit=1000, offset=offset, with_payload=True, with_vectors=False)
        for p in points:
            pl = dict(p.payload or {})
            if len(str(pl.get("text") or "")) >= min_chars:
                pl["point_id"] = str(p.id)
                pool.append(pl)
        if offset is None:
            break
    rnd = random.Random(seed)
    rnd.shuffle(pool)
    # At most two chunks per document, so a large document cannot fill the whole set
    per_doc: dict[str, int] = {}
    picked: list[dict[str, Any]] = []
    for pl in pool:
        d = str(pl.get("doc_id"))
        if per_doc.get(d, 0) >= 2:
            continue
        per_doc[d] = per_doc.get(d, 0) + 1
        picked.append(pl)
        if len(picked) >= n * 2:
            break
    return picked


def make_set(kb_id: str, n: int, out_path: str, *, seed: int = 7) -> dict[str, Any]:
    from kb_pipeline.graph.build import resolve_llm_specs
    from kb_pipeline.graph.llm import ChatClient, LLMCache

    from . import service

    settings, ss, q = service.runtime()
    if kb_id not in settings.sources:
        raise KeyError(kb_id)
    source = settings.sources[kb_id]
    spec = resolve_llm_specs(settings, source, ("extract",))["extract"]
    client = ChatClient(spec, cache=LLMCache(None), max_tokens=400, temperature=0.3)
    items: list[dict[str, Any]] = []
    skipped = 0
    for pl in _sample_chunks(q, source.collection, n, seed=seed):
        if len(items) >= n:
            break
        text = str(pl.get("text") or "")[:2500]
        try:
            reply = client.chat(MAKE_PROMPT.format(filename=pl.get("filename"), text=text), max_tokens=400)
            m = re.search(r"\{.*\}", reply, re.S)
            data = json.loads(m.group(0)) if m else {}
        except Exception:
            skipped += 1
            continue
        qtext = str(data.get("question") or "").strip()
        answer = str(data.get("answer") or "").strip()
        if not qtext or not answer or answer not in text or len(answer) > 60:
            skipped += 1
            continue
        items.append({"q": qtext, "expect": [answer], "gold_point_ids": [pl["point_id"]], "gold_docs": [pl.get("rel_path")],
                      "kbs": [kb_id], "tag": "synthetic", "block_type": pl.get("block_type")})
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps({"kb_id": kb_id, "made_at": time.strftime("%Y-%m-%d %H:%M:%S"), "model": spec.name,
                                          "items": items}, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"kb_id": kb_id, "items": len(items), "skipped": skipped, "out": out_path}


def cli_eval(args: Any) -> int:
    result = run_eval(args.set, out_path=args.out, kbs=args.kbs, top_k=int(args.top_k), compare_path=args.compare, auto=bool(getattr(args, "auto", False)))
    print(f"set: {result['set']}  auto_routing: {result['auto_routing']}  compared_to: {result.get('compared_to') or '-'}")
    print(result["table"])
    for r in result["items"]:
        flag = "ERR " if r.get("error") else ("    " if all(r.get(k, True) for k in ("hit@5", "expect", "expect_all", "min_docs", "negative")) else "FAIL")
        print(f"{flag} {r.get('tag') or '':10s} rank={r.get('rank', '-')!s:>3}({r.get('basis', '-')[:1]}) doc={r.get('doc_rank', '-')!s:>3} exp={r.get('expect', '-')!s:5} all={r.get('expect_all', '-')!s:5} "
              f"docs={r.get('docs', '-')!s:>2}/{r.get('min_docs', '-')!s:5} neg={r.get('negative', '-')!s:5} rr={r.get('rerank')} max={r.get('rerank_max')} "
              f"ms={r.get('total_ms')} {r['q'][:40]}" + (f"  !! {r['error']}" if r.get("error") else ""))
    return 0


def cli_make_set(args: Any) -> int:
    res = make_set(args.kb, int(args.n), args.out, seed=int(args.seed))
    print(json.dumps(res, ensure_ascii=False))
    return 0
