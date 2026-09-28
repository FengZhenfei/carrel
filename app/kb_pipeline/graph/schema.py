"""Schema sampling (the console's "extract / re-extract labels now"): sample chunks from this KB and let the
LLM induce the domain, output language, entity type list, type parents and relation predicate list, then
draft a set of capability questions.

The prompts of the first four steps come from the upstream prompt_tune (domain -> language -> persona ->
entity_types); the sampling logic and task constraints are kept as they were: fixed-seed random sampling
with a fallback cap on the total token count. The last three steps are local (schema layer, section 4.10).
7 calls in total, all through ChatClient (cached: clicking "re-extract" again on the same sample returns
instantly).

The output structure matches the old script (domain / language / entity_types / sampled / documents /
chunks_total / truncated) plus parent_types / predicates; the version-ring code needs no change.
"""
from __future__ import annotations

import json
import random
import re
from typing import Any, Iterable

from ..utils import count_tokens
from . import prompts
from .extract import normalize_predicate
from .llm import ChatClient
from .units import classify_unit_text

SAMPLE_TOKEN_BUDGET = 80_000
SAMPLE_SEED = 20260823
# Candidate pool cap: first take this many chunks stratified by document, then pick representatives from them
# (GraphRAG prompt-tune's n_subset_max is 300 as well)
SAMPLE_POOL_MAX = 300
MAX_PARENT_TYPES = 6
MIN_PREDICATES = 8
MAX_PREDICATES = 20
MAX_ENTITY_TYPES = 30            # the type menu with definitions goes into the extraction prompt; kb_001 once induced 82 types
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _allocate(sizes: list[int], total: int) -> list[int]:
    """Split total slots among documents by sqrt(chunk count): big documents get more, but no longer dominate
    linearly by chunk count; when there are enough slots every document gets at least 1. Largest-remainder
    method, deterministic."""
    n = len(sizes)
    if n == 0 or total <= 0:
        return [0] * n
    if total <= n:
        return [1 if i < total else 0 for i in range(n)]
    weights = [max(1.0, float(s)) ** 0.5 for s in sizes]
    scale = (total - n) / sum(weights)
    raw = [1 + w * scale for w in weights]
    quotas = [min(sizes[i], int(raw[i])) for i in range(n)]
    remainder = total - sum(quotas)
    order = sorted(range(n), key=lambda i: (-(raw[i] - int(raw[i])), i))
    while remainder > 0:
        progressed = False
        for i in order:
            if remainder <= 0:
                break
            if quotas[i] < sizes[i]:
                quotas[i] += 1
                remainder -= 1
                progressed = True
        if not progressed:
            break
    return quotas


def _spread(items: list[Any], k: int) -> list[Any]:
    """Take k chunks from one document at even spacing: the beginning, middle and end are all covered, instead of
    random picks landing in the same chapter."""
    if k <= 0 or not items:
        return []
    if k >= len(items):
        return list(items)
    if k == 1:
        return [items[len(items) // 2]]
    step = (len(items) - 1) / (k - 1)
    seen: set[int] = set()
    out: list[Any] = []
    for i in range(k):
        idx = int(round(i * step))
        if idx not in seen:
            seen.add(idx)
            out.append(items[idx])
    return out


def _farthest_point(pool: list[dict[str, Any]], vectors: dict[str, list[float]], size: int) -> list[dict[str, Any]]:
    """Pick the size candidates (among those with vectors) that are least alike: start from the chunk closest to
    the centroid, then repeatedly add the chunk whose cosine similarity to its nearest chosen chunk is lowest.
    Deterministic, O(size x pool)."""
    keyed = [(c, vectors[str(c.get("point_id") or "")]) for c in pool if vectors.get(str(c.get("point_id") or ""))]
    if not keyed:
        return []

    def unit(v: list[float]) -> list[float]:
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    vecs = [unit(v) for _, v in keyed]
    dim = len(vecs[0])
    centroid = unit([sum(v[i] for v in vecs) / len(vecs) for i in range(dim)])
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))
    first = max(range(len(vecs)), key=lambda i: (dot(vecs[i], centroid), -i))
    chosen = [first]
    nearest = [dot(vecs[i], vecs[first]) for i in range(len(vecs))]
    while len(chosen) < min(size, len(vecs)):
        nxt = min((i for i in range(len(vecs)) if i not in chosen), key=lambda i: (nearest[i], i))
        chosen.append(nxt)
        for i in range(len(vecs)):
            nearest[i] = max(nearest[i], dot(vecs[i], vecs[nxt]))
    return [keyed[i][0] for i in chosen]


def sample_chunks(chunks: list[dict[str, Any]], size: int, *, seed: int = SAMPLE_SEED,
                  budget: int = SAMPLE_TOKEN_BUDGET, pool_max: int = SAMPLE_POOL_MAX,
                  fetch_vectors=None) -> dict[str, Any]:
    """The sample for label extraction (health check D1, replacing the old uniform random sample over the KB):
    1. structural rules (units.classify_unit_text, zero calls) exclude boilerplate chunks such as TOC / revision
       history / legal notices;
    2. a candidate pool stratified by document: slots allocated by sqrt(chunk count), at least one chunk per
       document, evenly spaced within a document;
    3. with vectors (fetch_vectors reads the ready-made text vectors from the main collection), farthest-point
       sampling picks the size chunks of the pool that are least alike; without vectors, fixed-seed random;
    4. then truncate to the token budget.
    chunks carry at least text; optionally doc_id / chunk_index / chunk_uid / point_id / section. Sampling the
    same corpus twice gives the same result (sorting + fixed seed). Returns texts / truncated plus sampling stats."""
    ordered = sorted(chunks, key=lambda c: (str(c.get("doc_id") or ""), int(c.get("chunk_index") or 0), str(c.get("chunk_uid") or "")))
    ordered = [c for c in ordered if str(c.get("text") or "").strip()]
    empty = {"texts": [], "truncated": False, "picked": [], "pool": 0, "documents_covered": 0, "documents_total": 0,
             "excluded_boilerplate": 0, "method": "none"}
    if not ordered:
        return empty
    kept = [c for c in ordered
            if classify_unit_text(str(c.get("text") or ""), [str(c.get("section") or "")]) != "boilerplate"]
    excluded = len(ordered) - len(kept)
    if len(kept) < max(1, size):
        kept = ordered          # almost all boilerplate (e.g. only a TOC): exclude nothing, sample whatever there is
        excluded = 0
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for c in kept:
        by_doc.setdefault(str(c.get("doc_id") or ""), []).append(c)
    docs = list(by_doc)
    pool_n = min(len(kept), max(size, pool_max))
    quotas = _allocate([len(by_doc[d]) for d in docs], pool_n)
    pool: list[dict[str, Any]] = []
    for d, q in zip(docs, quotas):
        pool.extend(_spread(by_doc[d], q))
    vectors: dict[str, list[float]] = {}
    if fetch_vectors is not None and len(pool) > size:
        try:
            vectors = dict(fetch_vectors(pool) or {})
        except Exception as exc:     # main collection unreachable: fall back to stratified random rather than failing label extraction
            print(f"[schema] sample vectors unavailable, falling back to stratified random: {exc!r}", flush=True)
            vectors = {}
    picked = _farthest_point(pool, vectors, size) if len(vectors) >= min(size, len(pool)) else []
    method = "vectors"
    if not picked:
        method = "stratified"
        rng = random.Random(seed)
        shuffled = list(pool)
        rng.shuffle(shuffled)
        picked = shuffled[:size]
    texts: list[str] = []
    total = 0
    truncated = False
    for c in picked:
        text = str(c.get("text") or "")
        n = count_tokens(text)
        if texts and total + n > budget:
            truncated = True
            break
        texts.append(text)
        total += n
    used = picked[:len(texts)]
    return {"texts": texts, "truncated": truncated, "picked": used, "pool": len(pool),
            "documents_covered": len({str(c.get("doc_id") or "") for c in used}), "documents_total": len(by_doc),
            "excluded_boilerplate": excluded, "method": method}


def sample_texts(chunks: list[dict[str, Any]], size: int, *, seed: int = SAMPLE_SEED,
                 budget: int = SAMPLE_TOKEN_BUDGET) -> tuple[list[str], bool]:
    """Take size chunks, then truncate to the token budget. Returns (sample, whether the budget truncated it).
    A thin wrapper over sample_chunks (no vectors: stratified + fixed-seed random)."""
    out = sample_chunks(chunks, size, seed=seed, budget=budget)
    return out["texts"], out["truncated"]


def repair_json(body: str) -> str:
    """Complete unfinished JSON: the model occasionally stops right at max_tokens, missing the last } or ] (kb_001
    once lost a whole predicate list that way). Close brackets by the stack; if it stopped inside a string, close
    the string first; drop dangling commas."""
    stack: list[str] = []
    in_str = esc = False
    for ch in body:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]" and stack:
            stack.pop()
    out = body + ('"' if in_str else "")
    out = out.rstrip().rstrip(",")
    for ch in reversed(stack):
        out += "}" if ch == "{" else "]"
    return out


def parse_json_object(text: str) -> dict[str, Any]:
    """The model sometimes adds talk around the JSON or fences it in ```json; parse the first {...}. If that fails,
    complete it by the bracket stack and retry; failing that, fall back to the last complete element (dropping
    the truncated tail)."""
    body = str(text or "").strip()
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", body, flags=re.IGNORECASE | re.MULTILINE)
    candidates = [body]
    m = _JSON_RE.search(body)
    if m:
        candidates.append(m.group(0))
    start = body.find("{")
    if start >= 0:
        tail = body[start:]
        candidates.append(repair_json(tail))
        cut = tail
        for _ in range(3):
            idx = cut.rfind("}")
            if idx <= 0:
                break
            cut = cut[:idx + 1]
            candidates.append(repair_json(cut))
            cut = cut[:-1]
    for cand in candidates:
        try:
            data = json.loads(cand)
        except ValueError:
            continue
        if isinstance(data, dict):
            return data
    return {}


def parse_entity_types(value: Any, *, limit: int = MAX_ENTITY_TYPES) -> list[str]:
    if isinstance(value, dict):
        value = value.get("entity_types") or value.get("entities")
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for v in value:
            name = str(v).strip()
            if name and name.casefold() not in {o.casefold() for o in out}:
                out.append(name)
        return out[:limit]
    text = str(value or "")
    m = re.search(r"\[(.*?)\]", text, re.S)
    body = m.group(1) if m else text
    items = re.split(r"[,\n]", body)
    return [i.strip().strip("\"'` ") for i in items if i.strip().strip("\"'` ")]


def normalize_parent_types(value: Any, entity_types: list[str]) -> dict[str, str]:
    """{type: parent type}; parents are restricted to the six upper-ontology classes (limits.UPPER_PARENTS);
    unrecognised ones are mapped by legacy name, else default to entity."""
    from ..limits import upper_parent_of

    mapping = value.get("parent_types") if isinstance(value, dict) and "parent_types" in value else value
    out: dict[str, str] = {}
    if not isinstance(mapping, dict):
        return out
    lookup = {t.casefold(): t for t in entity_types}
    for k, v in mapping.items():
        key = lookup.get(str(k).strip().casefold())
        parent = upper_parent_of(v) or ("entity" if str(v or "").strip() else "")
        if key and parent:
            out[key] = parent
    return out


def upper_parents_text() -> str:
    from ..limits import UPPER_PARENT_DESCRIPTIONS

    return "\n".join(f"- {name}: {desc}" for name, desc in UPPER_PARENT_DESCRIPTIONS.items())


def map_types_to_upper(client: ChatClient, entity_types: list[str], parent_types: dict[str, str] | None = None) -> dict[str, str]:
    """Old schema versions (parents are made-up names) -> the six upper classes, in one call; on failure or for
    missing items fall back to the legacy-name mapping, else entity."""
    from ..limits import DOCUMENT_SCOPED_PARENTS, UPPER_PARENTS, upper_parent_of  # noqa: F401

    parent_types = parent_types or {}
    listed = "\n".join(f"- {t}" + (f" (author's category: {parent_types[t]})" if parent_types.get(t) else "") for t in entity_types)
    mapped: dict[str, str] = {}
    try:
        raw = client.chat(prompts.UPPER_MAPPING_PROMPT.format(entity_types=listed, upper_parents=upper_parents_text()),
                          max_tokens=1024)
        mapped = normalize_parent_types(parse_json_object(raw), list(entity_types))
    except Exception as exc:
        print(f"[graph] upper mapping call failed, falling back to legacy names: {exc!r}", flush=True)
    out: dict[str, str] = {}
    for t in entity_types:
        out[t] = mapped.get(t) or upper_parent_of(parent_types.get(t)) or "entity"
    return out


def normalize_predicates(value: Any, *, limit: int = MAX_PREDICATES) -> list[dict[str, Any]]:
    items = value.get("predicates") if isinstance(value, dict) and "predicates" in value else value
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    if not isinstance(items, (list, tuple)):
        return out
    for item in items:
        if isinstance(item, str):
            item = {"name": item}
        if not isinstance(item, dict):
            continue
        name = normalize_predicate(str(item.get("name") or ""))
        if not name or name == prompts.DEFAULT_PREDICATE or name in seen:
            continue
        seen.add(name)
        out.append({
            "name": name,
            "description": str(item.get("description") or "").strip()[:200],
            "source_parents": [str(s).strip().lower() for s in (item.get("source_parents") or []) if str(s).strip()][:8],
            "target_parents": [str(s).strip().lower() for s in (item.get("target_parents") or []) if str(s).strip()][:8],
        })
        if len(out) >= limit:
            break
    return out


def normalize_definitions(value: Any, entity_types: list[str]) -> dict[str, str]:
    from ..limits import normalize_type_definitions

    mapping = value.get("definitions") if isinstance(value, dict) and "definitions" in value else value
    return normalize_type_definitions(mapping, entity_types)


def normalize_profile_suggestion(value: Any, *, entity_types: list[str], predicates: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep only the profile's types and predicates that exist in the lists; axis accepts date / version / none."""
    from ..limits import normalize_profile

    profile = normalize_profile(value)
    if not profile:
        return {}
    lookup = {t.casefold(): t for t in entity_types}
    profile["subject_types"] = [lookup[t.casefold()] for t in profile.get("subject_types") or [] if t.casefold() in lookup]
    names = {str(p.get("name") or "") for p in predicates}
    profile["extension_predicates"] = [p for p in profile.get("extension_predicates") or [] if p in names]
    return profile


EXAMPLE_MAX_TEXT_CHARS = 1800
EXAMPLE_MIN_TEXT_CHARS = 120
EXAMPLE_MAX_COUNT = 2
EXAMPLE_MAX_TRIES = 8
EXAMPLE_MAX_RELATED_RATIO = 0.3
EXAMPLE_RELAX_RATIO = 0.5


def score_example(ents: list[dict[str, Any]], rels: list[dict[str, Any]], schema: Any, *, related_predicate: str) -> dict[str, Any]:
    """Quality of one sample extraction: among relations whose both ends are in the entities, the share of
    related_to and the number of relations with a real predicate. Predicate endpoint constraints follow the same
    rule as the pipeline's merge stage (merge.relax_bad_endpoints): when more than half of a predicate's relations
    in this sample violate the ends, the constraint is wrong rather than the extraction, and the whole predicate
    is let through; otherwise only the violating relations are removed."""
    allowed = schema.allowed_ends()
    parent_of = {t.casefold(): str(p).casefold() for t, p in (schema.parent_types or {}).items()}
    names = {str(e["name"]).casefold(): str(e["type"]) for e in ents}
    well_formed = [r for r in rels if str(r["source"]).casefold() in names and str(r["target"]).casefold() in names]
    uses: dict[str, int] = {}
    bad: dict[str, list[int]] = {}
    for idx, r in enumerate(well_formed):
        pred = str(r["predicate"])
        uses[pred] = uses.get(pred, 0) + 1
        ends = allowed.get(pred)
        if ends is None:
            continue
        src_t, tgt_t = names[str(r["source"]).casefold()], names[str(r["target"]).casefold()]
        src_ok = not ends[0] or "*" in ends[0] or parent_of.get(src_t.casefold(), "") in ends[0] or src_t.casefold() in ends[0]
        tgt_ok = not ends[1] or "*" in ends[1] or parent_of.get(tgt_t.casefold(), "") in ends[1] or tgt_t.casefold() in ends[1]
        if not (src_ok and tgt_ok):
            bad.setdefault(pred, []).append(idx)
    relaxed = {p for p, idxs in bad.items() if len(idxs) / max(1, uses[p]) > EXAMPLE_RELAX_RATIO}
    drop = {i for p, idxs in bad.items() if p not in relaxed for i in idxs}
    kept = [r for i, r in enumerate(well_formed) if i not in drop]
    related = sum(1 for r in kept if str(r["predicate"]) == related_predicate)
    return {"relations": kept, "related_ratio": (related / len(kept)) if kept else 1.0, "typed": len(kept) - related,
            "violations_removed": len(drop), "relaxed_predicates": sorted(relaxed)}


def generate_examples(client: ChatClient, texts: list[str], schema: Any, *, max_examples: int = EXAMPLE_MAX_COUNT) -> tuple[str, dict[str, Any]]:
    """Generate few-shot examples from this KB's corpus (GraphRAG prompt tune's idea, with validation added): run
    the current extraction prompt over a few samples, score them with score_example, prefer those with a
    related_to share <= 0.3; if none qualifies take the ones with the most predicate-bearing relations (an example
    from this corpus beats a news example), and only reject samples that yield no relations at all. Returns
    (example text, stats). Returns an empty string when no sample survives (extraction then uses the generic
    examples).
    2026-09-08: the health KB used to keep no sample at all -- short chunks (mostly < 200 characters), the endpoint
    constraints removed 12 of 12 relations, and the rest were all related_to; now the length floor is 120, the
    endpoint constraints follow the pipeline's relaxation rule, and the hard threshold became a ranking."""
    from . import prompts
    from .extract import parse_records, render_extract_prompt

    known = {t.casefold(): t for t in schema.entity_types}
    stats: dict[str, Any] = {"tried": 0, "kept": 0, "dropped_empty": 0, "violations_removed": 0, "fallback": 0, "relaxed_predicates": []}
    candidates = sorted((t for t in texts if EXAMPLE_MIN_TEXT_CHARS <= len(t) <= EXAMPLE_MAX_TEXT_CHARS * 2), key=len, reverse=True)
    scored: list[tuple[tuple[float, int, int], str, list[dict[str, Any]], list[dict[str, Any]]]] = []
    for text in candidates[:EXAMPLE_MAX_TRIES]:
        stats["tried"] += 1
        sample = text[:EXAMPLE_MAX_TEXT_CHARS]
        prompt = render_extract_prompt(sample, section="(sample)", schema=schema, document="(sample)", unit_kind="body")
        try:
            raw = client.chat(prompt, max_tokens=2048)
        except Exception as exc:
            print(f"[schema] example extraction failed: {exc!r}", flush=True)
            continue
        ents, rels, _ = parse_records(raw, schema)
        ents = [e for e in ents if str(e.get("type") or "").casefold() in known]
        sc = score_example(ents, rels, schema, related_predicate=prompts.DEFAULT_PREDICATE)
        stats["violations_removed"] += sc["violations_removed"]
        for p in sc["relaxed_predicates"]:
            if p not in stats["relaxed_predicates"]:
                stats["relaxed_predicates"].append(p)
        if not ents or not sc["relations"]:
            stats["dropped_empty"] += 1
            continue
        scored.append(((sc["related_ratio"], -sc["typed"], -len(ents)), sample, ents, sc["relations"]))
    scored.sort(key=lambda row: row[0])
    good = [row for row in scored if row[0][0] <= EXAMPLE_MAX_RELATED_RATIO]
    chosen = good[:max_examples]
    if not chosen:
        chosen = [row for row in scored if -row[0][1] > 0][:max_examples]     # at least one relation with a real predicate
        stats["fallback"] = len(chosen)
    blocks: list[str] = []
    type_names = ", ".join(str(t) for t in schema.entity_types)
    predicate_names = ", ".join(str(p.get("name") if isinstance(p, dict) else p) for p in (schema.predicates or ())) or prompts.DEFAULT_PREDICATE
    for _, sample, ents, kept_rels in chosen:
        records = ['("unit"<|>body<|>sample from this corpus)']
        for e in ents[:12]:
            records.append(f'("entity"<|>{e["name"]}<|>{e["type"]}<|>{e["description"]})')
        for r in kept_rels[:12]:
            records.append(f'("relationship"<|>{r["source"]}<|>{r["target"]}<|>{r["predicate"]}<|>{r["description"]}<|>{int(r.get("strength") or 5)})')
        body = "\n##\n".join(records) + "\n" + prompts.COMPLETION_DELIMITER
        # the menus in the examples list names only: the main prompt already carries the full menu with definitions,
        # and repeating it in every example would add ten thousand characters of duplication
        blocks.append(
            f"Example {len(blocks) + 1}:\nEntity_types: {type_names}\nPredicates: {predicate_names}\n"
            f"Document: (sample)\nSection: (sample)\nText:\n{sample}\n######################\nOutput:\n{body}")
        stats["kept"] += 1
    return "\n\n######################\n".join(blocks), stats


def _prior_types_text(prior: dict[str, Any] | None) -> str:
    """The previous version's type list plus the observation counts from the ledger (entity count per type, type
    names outside the list that showed up during extraction). Without prior returns an empty string: the prompt
    of a first label extraction does not change by a single character."""
    types = [str(t) for t in ((prior or {}).get("entity_types") or []) if str(t).strip()]
    if not types:
        return ""
    text = ("\nThe previous version of this schema used these entity types: " + ", ".join(types)
            + ". Keep every name that still fits this corpus (renaming a type invalidates the existing graph), "
            "drop only types that do not occur in the corpus, and add what is missing.")
    observed = (prior or {}).get("observed") or {}
    counts = observed.get("type_counts") or {}
    if counts:
        lookup = {str(k).casefold(): int(v or 0) for k, v in counts.items()}
        text += ("\nHow many entities the current graph holds under each type: "
                 + ", ".join(f"{t} ({lookup.get(t.casefold(), 0)})" for t in types)
                 + ". A type with no entities is not used by this corpus; a type holding most of the graph may be too broad.")
    unknown = observed.get("unknown_types") or {}
    if unknown:
        text += ("\nThe extractor also labelled entities with type names that are not in the list: "
                 + ", ".join(f"{t} ({int(n or 0)})" for t, n in list(unknown.items())[:10])
                 + ". Add a type for a name that is a real, recurring category of this corpus; if it is only another "
                 "wording of an existing type, keep the existing name.")
    return text


def _prior_predicates_text(prior: dict[str, Any] | None) -> str:
    """The previous version's predicate list plus the endpoint ledger, appended after the parent list in the
    predicate prompt: the model revises on that basis instead of guessing from scratch."""
    preds = list((prior or {}).get("predicates") or [])
    if not preds:
        return ""
    observed = (prior or {}).get("observed") or {}
    edges = observed.get("edges") or {}
    violations = observed.get("violations") or {}
    pairs = observed.get("pairs") or {}
    confirmed: dict[str, list[str]] = {}
    for row in observed.get("confirmed") or []:
        confirmed.setdefault(str(row.get("predicate") or ""), []).append(
            f"{row.get('source_parent')} -> {row.get('target_parent')} ({int(row.get('count') or 0)} edges)")
    lines = []
    for p in preds:
        name = str(p.get("name") or "").strip()
        if not name:
            continue
        ends = f"{', '.join(str(x) for x in (p.get('source_parents') or ['*']))} -> {', '.join(str(x) for x in (p.get('target_parents') or ['*']))}"
        line = f"- {name} [{ends}]: {str(p.get('description') or '').strip()}"
        if name in edges:
            n_edges = int(edges[name])
            line += f" (used by {n_edges} edges in the current graph"
            bad = int(violations.get(name) or 0)
            if n_edges and bad:
                line += f"; {bad / n_edges:.0%} of them fall outside the declared ends"
            top = list((pairs.get(name) or {}).items())[:3]
            if top:
                line += "; most common endpoint pairs: " + ", ".join(
                    f"{str(k).replace('->', ' -> ')} ({int(v or 0)})" for k, v in top)
            if confirmed.get(name):
                line += "; data-confirmed endpoint pairs: " + "; ".join(confirmed[name])
            line += ")"
        lines.append(line)
    text = ("\n\nThe previous version of this schema defined these predicates (with the parent categories allowed at "
            "their source -> target ends, and how the current graph actually uses them):\n" + "\n".join(lines)
            + "\nRevise this list rather than starting over: keep predicates that carry many edges, keep every "
            "data-confirmed endpoint pair allowed, refine descriptions, drop predicates that never occur, add what is missing.")
    if violations:
        text += (" When most edges of a predicate fall outside its declared ends, the ends were declared too narrowly: "
                 "widen them to the pairs the data shows (or split the predicate) rather than dropping it.")
    return text


def _prior_profile_text(prior: dict[str, Any] | None) -> str:
    profile = (prior or {}).get("profile") or {}
    if not profile:
        return ""
    return ("\n\nThe previous version of this schema described the corpus with this profile; keep it unless the sample "
            "says otherwise:\n" + json.dumps(profile, ensure_ascii=False))


def apply_observed_guard(predicates: Iterable[dict[str, Any]], prior: dict[str, Any] | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Data guard: endpoint pairs confirmed in the previous version's ledger (>= 20 edges and >= 5% at build time)
    are kept by rule, not left to the model -- if the model dropped that predicate, the previous version's entry
    is restored; if it narrowed the ends, the parent type is added back. Returns (predicate list, record)."""
    from .merge import confirmed_pairs

    out = [dict(p) for p in predicates]
    record: dict[str, Any] = {"kept_predicates": [], "widened": []}
    confirmed = confirmed_pairs((prior or {}).get("observed") or {})
    if not confirmed:
        return out, record
    prior_preds = {normalize_predicate(str(p.get("name") or "")): p for p in ((prior or {}).get("predicates") or [])}
    by_name = {normalize_predicate(str(p.get("name") or "")): p for p in out}
    for pred, src, tgt in confirmed:
        key = normalize_predicate(pred)
        if key not in by_name:
            base = prior_preds.get(key)
            if base is None:
                continue
            row = {**base, "source_parents": list(base.get("source_parents") or ["*"]),
                   "target_parents": list(base.get("target_parents") or ["*"])}
            out.append(row)
            by_name[key] = row
            record["kept_predicates"].append(key)
        row = by_name[key]
        for side, parent in (("source_parents", src), ("target_parents", tgt)):
            ends = [str(x).strip() for x in (row.get(side) or []) if str(x).strip()]
            if not ends or "*" in ends or parent in ("", "-"):
                continue
            if parent.casefold() not in {e.casefold() for e in ends}:
                ends.append(parent)
                row[side] = ends
                record["widened"].append(f"{key}.{side}+{parent}")
    return out, record


def suggest(client: ChatClient, docs: list[str], *, domain: str | None = None, examples: bool = True,
            prior: dict[str, Any] | None = None) -> dict[str, Any]:
    """Nine calls: domain -> language -> persona -> type list -> parents -> predicates -> type definitions ->
    scenario profile -> corpus examples (one call for each of <= 2 samples).
    (Capability questions were removed 2026-09-06: self-check questions for acceptance; neither the build nor
    retrieval depends on them.)
    prior: the active schema version (type list, predicates, profile, endpoint ledger); when given, the model
    revises on that basis; endpoint pairs confirmed in the ledger are kept by rule in apply_observed_guard,
    regardless of whether the model complies."""
    joined = " ".join(docs)
    if not domain:
        domain = client.chat(prompts.GENERATE_DOMAIN_PROMPT.format(input_text=joined), max_tokens=256).strip().strip('"')
    language = client.chat(prompts.DETECT_LANGUAGE_PROMPT.format(input_text=joined), max_tokens=64).strip().strip('".')
    task = prompts.ENTITY_TYPE_TASK.format(domain=domain) + _prior_types_text(prior)
    persona = client.chat(prompts.GENERATE_PERSONA_PROMPT.format(sample_task=task), max_tokens=512).strip()
    raw_types = client.chat(
        [{"role": "system", "content": persona},
         {"role": "user", "content": prompts.ENTITY_TYPE_GENERATION_PROMPT.format(task=task, input_text=joined)}],
        max_tokens=1024,
    )
    entity_types = parse_entity_types(parse_json_object(raw_types) or raw_types)
    parent_types: dict[str, str] = {}
    predicates: list[dict[str, Any]] = []
    guard: dict[str, Any] = {}
    if entity_types:
        raw_parents = client.chat(prompts.PARENT_TYPES_PROMPT.format(upper_parents=upper_parents_text(),
            persona=persona, domain=domain, entity_types="\n".join(f"- {t}" for t in entity_types),
            max_parents=MAX_PARENT_TYPES), max_tokens=1024)
        parent_types = normalize_parent_types(parse_json_object(raw_parents), entity_types)
        parents_text = "\n".join(
            f"- {parent}: {', '.join(t for t, p in parent_types.items() if p == parent)}"
            for parent in dict.fromkeys(parent_types.values())
        ) or "\n".join(f"- {t}" for t in entity_types)
        parents_text += _prior_predicates_text(prior)
        # the predicate list is half of the extraction constraints; a response with no parseable predicates is not
        # cached and is requested again with a correction prompt (ChatClient.validate)
        raw_predicates = client.chat(prompts.PREDICATES_PROMPT.format(
            persona=persona, domain=domain, parent_types=parents_text, input_text=joined,
            min_predicates=MIN_PREDICATES, max_predicates=MAX_PREDICATES), max_tokens=4096,
            validate=lambda t: bool(normalize_predicates(parse_json_object(t))))
        predicates, guard = apply_observed_guard(normalize_predicates(parse_json_object(raw_predicates)), prior)
    type_definitions: dict[str, str] = {}
    profile: dict[str, Any] = {}
    example_text = ""
    example_stats: dict[str, int] = {}
    if entity_types:
        typed = "\n".join(f"- {t} (parent: {parent_types.get(t, '')})" for t in entity_types)
        try:
            raw_defs = client.chat(prompts.TYPE_DEFINITIONS_PROMPT.format(
                persona=persona, domain=domain, entity_types=typed, input_text=joined[:20000]), max_tokens=4096,
                validate=lambda t: bool(normalize_definitions(parse_json_object(t), entity_types)))
            type_definitions = normalize_definitions(parse_json_object(raw_defs), entity_types)
        except Exception as exc:
            print(f"[schema] type definitions call failed: {exc!r}", flush=True)
        try:
            raw_profile = client.chat(prompts.PROFILE_PROMPT.format(
                persona=persona, domain=domain, entity_types=typed,
                predicates=("\n".join(f"- {p['name']}: {p.get('description') or ''}" for p in predicates) or "- (none)")
                + _prior_profile_text(prior),
                input_text=joined[:20000]), max_tokens=1024,
                validate=lambda t: "axis" in parse_json_object(t))
            profile = normalize_profile_suggestion(parse_json_object(raw_profile), entity_types=entity_types, predicates=predicates)
        except Exception as exc:
            print(f"[schema] profile call failed: {exc!r}", flush=True)
        if examples:
            from .extract import ExtractionSchema

            schema = ExtractionSchema(entity_types=tuple(entity_types), predicates=tuple(predicates), language=language,
                                      parent_types=dict(parent_types), type_definitions=dict(type_definitions))
            example_text, example_stats = generate_examples(client, docs, schema)
    return {
        "domain": domain, "language": language, "persona": persona,
        "entity_types": entity_types, "parent_types": parent_types,
        "predicates": predicates,
        "type_definitions": type_definitions, "profile": profile,
        "examples": example_text, "example_stats": example_stats,
        "guard": guard,
    }
