"""模式采样(控制台「立即/重新抽取标签」、空白图建图前的自动归纳):从本库切片里采样,用 LLM 归纳出
领域、输出语言、实体类型表、类型父类(八个上层类之一或表内另一个类型)、关系谓词表(带两端、单值与否)、
类型定义与作用域、场景画像,再从本库语料生成 few-shot。

前四段的提示词取自上游 prompt_tune(domain → language → persona → entity_types),采样逻辑与任务约束
原样保留:固定种子、按 token 总量兜底。后面几段是本地的。全部走 ChatClient(带缓存:同一份样本再点一次
「重新抽取」秒回)。产物用 ontology.normalize_model 规范化成模式字典,由 schema_flow 存成 LinkML 版本。
"""
from __future__ import annotations

import json
import random
import re
from typing import Any, Iterable

from ..utils import count_tokens
from . import prompts
from .extract import normalize_predicate
from .llm import ChatClient, LLMInputBudgetExceeded, LLMInterrupted
from .sampling_budget import estimate_text
from .units import classify_unit_text

SAMPLE_TOKEN_BUDGET = 80_000
SAMPLE_SEED = 20260823
# 候选池基础规模;自动模式随预算和文档数扩大,不再把后面的文档排除在 300 片之外。
SAMPLE_POOL_MAX = 300
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


def _farthest_point(pool: list[dict[str, Any]], vectors: dict[str, list[float]], size: int,
                    *, cover_documents: bool = False, coverage_budget: int | None = None) -> list[dict[str, Any]]:
    """Pick the size candidates (among those with vectors) that are least alike: start from the chunk closest to
    the centroid, then repeatedly add the chunk whose cosine similarity to its nearest chosen chunk is lowest.
    Deterministic, O(size x pool)."""
    keyed = [(c, vectors[str(c.get("point_id") or "")]) for c in pool if vectors.get(str(c.get("point_id") or ""))]
    if not keyed:
        return []

    import numpy as np

    vecs = np.asarray([v for _, v in keyed], dtype=np.float64)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.where(norms > 0, norms, 1)
    centroid = vecs.mean(axis=0)
    chosen: list[int] = []
    if cover_documents:
        by_doc: dict[str, list[int]] = {}
        for i, (c, _) in enumerate(keyed):
            by_doc.setdefault(str(c.get("doc_id") or ""), []).append(i)
        # One central representative per document before semantic expansion.
        for ids in by_doc.values():
            local = vecs[ids].mean(axis=0)
            chosen.append(max(ids, key=lambda i: (float(np.dot(vecs[i], local)), -i)))
        if coverage_budget is not None:
            costs = [estimate_text(str(c.get("text") or "")) + 2 for c, _ in keyed]
            if sum(costs[i] for i in chosen) > coverage_budget:
                # Prefer coverage to a few long representatives when context is
                # tight; semantic diversity expands the set after this seed.
                chosen = [min(ids, key=lambda i: (costs[i], i)) for ids in by_doc.values()]
        random.Random(SAMPLE_SEED).shuffle(chosen)
        chosen = chosen[:size]
    if not chosen:
        chosen = [int(np.argmax(vecs @ centroid))]
    nearest = np.full(len(vecs), -np.inf)
    selected = np.zeros(len(vecs), dtype=bool)
    for i in chosen:
        nearest = np.maximum(nearest, np.einsum("ij,j->i", vecs, vecs[i]))
        selected[i] = True
    while len(chosen) < min(size, len(vecs)):
        nxt = int(np.argmin(np.where(selected, np.inf, nearest)))
        chosen.append(nxt)
        selected[nxt] = True
        nearest = np.maximum(nearest, np.einsum("ij,j->i", vecs, vecs[nxt]))
    return [keyed[i][0] for i in chosen]


def sample_chunks(chunks: list[dict[str, Any]], size: int, *, seed: int = SAMPLE_SEED,
                  budget: int = SAMPLE_TOKEN_BUDGET, pool_max: int = SAMPLE_POOL_MAX,
                  fetch_vectors=None) -> dict[str, Any]:
    """抽标签用的样本;size=0 按模型预算自动确定数量(生产入口只使用自动模式):
    1. 结构规则(units.classify_unit_text,零调用)排除目录 / 修订记录 / 法律声明这类样板片;
    2. 按文档分层取候选池:名额按 sqrt(切片数) 分配,每份文档至少一片,文档内等距取;
    3. 有向量(fetch_vectors 取主库里现成的 text 向量)就在池里做最远点采样挑彼此最不像的
       size 片;取不到向量就固定种子随机;
    4. 按含估算余量的 token 预算选择完整片段,跳过超长片段,不截断正文。
    chunks 至少带 text;可带 doc_id / chunk_index / chunk_uid / point_id / section。同一份语料
    两次采样结果一样(排序 + 固定种子)。返回 texts / truncated 与采样统计。"""
    ordered = sorted(chunks, key=lambda c: (str(c.get("doc_id") or ""), int(c.get("chunk_index") or 0), str(c.get("chunk_uid") or "")))
    ordered = [c for c in ordered if str(c.get("text") or "").strip()]
    empty = {"texts": [], "truncated": False, "picked": [], "pool": 0, "documents_covered": 0, "documents_total": 0,
             "excluded_boilerplate": 0, "method": "none"}
    if not ordered:
        return empty
    kept = [c for c in ordered
            if classify_unit_text(str(c.get("text") or ""), [str(c.get("section") or "")]) != "boilerplate"]
    excluded = len(ordered) - len(kept)
    if not kept:
        kept = ordered          # 几乎全是样板(比如只有一份目录):不排除,有什么采什么
        excluded = 0
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for c in kept:
        by_doc.setdefault(str(c.get("doc_id") or ""), []).append(c)
    docs = list(by_doc)
    automatic = size == 0
    costs = {id(c): estimate_text(str(c.get("text") or "")) + 2 for c in kept}
    if automatic:
        average = sum(costs.values()) / len(kept)
        size = min(len(kept), max(len(docs), int(budget / max(1, average))))
    pool_n = min(len(kept), max(size * 2 if automatic else size, pool_max, len(docs)))
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
    picked = _farthest_point(pool, vectors, size, cover_documents=automatic,
                             coverage_budget=budget if automatic else None) if len(vectors) >= len(pool) else []
    method = "vectors"
    if not picked:
        method = "stratified"
        rng = random.Random(seed)
        shuffled = list(pool)
        rng.shuffle(shuffled)
        if automatic:
            # Without vectors retain document coverage rather than letting a
            # long document dominate. Stable input + seed gives reproducibility.
            grouped: dict[str, list[dict[str, Any]]] = {}
            for c in shuffled:
                grouped.setdefault(str(c.get("doc_id") or ""), []).append(c)
            shuffled = [cs[i] for i in range(max(map(len, grouped.values())))
                        for cs in grouped.values() if i < len(cs)]
        picked = shuffled[:size]
    if automatic:
        seen = {id(c) for c in picked}
        # Estimated average determines the diverse core, not a hard count cap.
        # Fill remaining budget with stratified candidates, never just a prefix.
        remaining = [c for c in pool if id(c) not in seen]
        random.Random(seed).shuffle(remaining)
        picked.extend(remaining)
    texts: list[str] = []
    used: list[dict[str, Any]] = []
    total = 0
    truncated = False
    for c in picked:
        text = str(c.get("text") or "")
        n = costs[id(c)]
        if total + n > budget:
            truncated = True
            continue
        texts.append(text)
        used.append(c)
        total += n
    return {"texts": texts, "truncated": truncated, "picked": used, "pool": len(pool),
            "documents_covered": len({str(c.get("doc_id") or "") for c in used}), "documents_total": len(by_doc),
            "excluded_boilerplate": excluded, "method": method,
            "estimated_tokens": total, "token_budget": budget,
            "algorithm": "model-budget-diverse-v1" if automatic else "manual-legacy",
            "stop_reason": "token_budget" if truncated else ("corpus_covered" if len(used) == len(kept) else "candidate_pool")}


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


_TYPE_ALTERNATIVES_RE = re.compile(r"\s+/\s+")


def entity_type_name(value: Any) -> str:
    """模型给的类型名规整成一个类型:「vendor / service provider」这种把两种叫法用前后带空格的斜杠拼在一起的,
    取前一个(类型定义里两种说法都会写到);「I/O pin」「TCP/IP」这种词内斜杠不动。"""
    name = " ".join(str(value or "").strip().strip("\"'` ").split())
    head = _TYPE_ALTERNATIVES_RE.split(name, maxsplit=1)[0].strip()
    return head or name


def clean_domain(value: Any, *, limit: int = 80) -> str:
    """领域只要一个短语:模型偶尔回一段带 Markdown 的说明,取第一行非空的,去掉加粗 / 标题符号、引号和「Domain:」前缀,
    截到 limit 个字符。"""
    for line in str(value or "").splitlines():
        line = re.sub(r"[*#`]+", "", line).strip().strip("\"'“”「」 ")
        line = re.sub(r"^(domain|领域)\s*[:：]\s*", "", line, flags=re.I).strip()
        if line:
            return line.rstrip("。.").strip()[:limit]
    return ""


def parse_entity_types(value: Any, *, limit: int = MAX_ENTITY_TYPES) -> list[str]:
    """归纳出的类型名单;与上层类同名的不要(ontology.UPPER_NAMES)。"""
    from ..ontology import is_upper_name

    if isinstance(value, dict):
        value = value.get("entity_types") or value.get("entities")
    if isinstance(value, (list, tuple)):
        items = [str(v) for v in value]
    else:
        text = str(value or "")
        m = re.search(r"\[(.*?)\]", text, re.S)
        items = re.split(r"[,\n]", m.group(1) if m else text)
    out: list[str] = []
    for item in items:
        name = entity_type_name(item)
        if name and not is_upper_name(name) and name.casefold() not in {o.casefold() for o in out}:
            out.append(name)
    return out[:limit]


def normalize_parent_types(value: Any, entity_types: list[str]) -> dict[str, str]:
    """{类型: 父类}:父类是八个上层类之一(含同义词)或表内另一个类型(多层);都认不出的按 entity。"""
    from ..ontology import upper_of_name

    mapping = value.get("parent_types") if isinstance(value, dict) and "parent_types" in value else value
    out: dict[str, str] = {}
    if not isinstance(mapping, dict):
        return out
    lookup = {t.casefold(): t for t in entity_types}
    for k, v in mapping.items():
        key = lookup.get(str(k).strip().casefold())
        raw = str(v or "").strip()
        if not key or not raw:
            continue
        other = lookup.get(raw.casefold())
        if other and other != key:
            out[key] = other
        else:
            out[key] = upper_of_name(raw) or "entity"
    return out


def upper_parents_text() -> str:
    from ..ontology import UPPER_DESCRIPTIONS

    return "\n".join(f"- {name}: {desc}" for name, desc in UPPER_DESCRIPTIONS.items())


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
        single = item.get("single_valued")
        if "multivalued" in item and single is None:
            single = item.get("multivalued") is False
        out.append({
            "name": name,
            "description": str(item.get("description") or "").strip()[:200],
            "source_parents": [str(s).strip() for s in (item.get("source_parents") or []) if str(s).strip()][:8],
            "target_parents": [str(s).strip() for s in (item.get("target_parents") or []) if str(s).strip()][:8],
            "multivalued": not (single is True or str(single).strip().lower() == "true"),
        })
        if len(out) >= limit:
            break
    return out


def normalize_definitions(value: Any, entity_types: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    """TYPE_DEFINITIONS_PROMPT 的回答 → ({类型: 定义}, {类型: 作用域})。也认只有定义的旧形状 {"definitions": {...}}。"""
    from ..limits import normalize_type_definitions, normalize_type_scopes

    if isinstance(value, dict) and isinstance(value.get("types"), dict):
        defs: dict[str, Any] = {}
        scopes: dict[str, Any] = {}
        for k, v in value["types"].items():
            if isinstance(v, dict):
                defs[k] = v.get("definition") or v.get("description") or ""
                scopes[k] = v.get("scope") or ""
            else:
                defs[k] = v
        lookup = {t.casefold(): t for t in entity_types}
        scoped = {lookup[str(k).strip().casefold()]: v for k, v in scopes.items() if str(k).strip().casefold() in lookup}
        return normalize_type_definitions(defs, entity_types), normalize_type_scopes(scoped)
    mapping = value.get("definitions") if isinstance(value, dict) and "definitions" in value else value
    return normalize_type_definitions(mapping, entity_types), {}


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
        if not (schema.endpoint_ok(src_t, ends[0]) and schema.endpoint_ok(tgt_t, ends[1])):
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
    2026-09-08: one evaluation corpus used to keep no sample at all -- short chunks (mostly < 200 characters), the endpoint
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
        except (LLMInputBudgetExceeded, LLMInterrupted):
            raise
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
            records.append(f'("entity"<|>{e["name"]}<|>{e["type"]}<|>{e["description"]}<|>{e.get("kind") or "instance"})')
        for r in kept_rels[:12]:
            period = f'{r.get("valid_from") or ""}..{r.get("valid_until") or ""}' if (r.get("valid_from") or r.get("valid_until")) else ""
            records.append(f'("relationship"<|>{r["source"]}<|>{r["target"]}<|>{r["predicate"]}<|>{r["description"]}'
                           f'<|>{int(r.get("strength") or 5)}<|>{period})')
        body = "\n##\n".join(records) + "\n" + prompts.COMPLETION_DELIMITER
        # the menus in the examples list names only: the main prompt already carries the full menu with definitions,
        # and repeating it in every example would add ten thousand characters of duplication
        blocks.append(
            f"Example {len(blocks) + 1}:\nEntity_types: {type_names}\nPredicates: {predicate_names}\n"
            f"Document: (sample)\nDocument date: (unknown)\nSection: (sample)\nText:\n{sample}\n######################\nOutput:\n{body}")
        stats["kept"] += 1
    return "\n\n######################\n".join(blocks), stats


def _prior_types_text(prior: dict[str, Any] | None) -> str:
    """上一版类型表 + 账里的体检数(每类实体数、抽取时冒出来的表外类型名)。没有 prior 就返回空串:
    首次抽标签的提示词一个字都不变。上一版里与上层类同名的类型不列进「保留」,单独说明它们装了多少实体、要换成
    更具体的类型(只是从名单里删掉的话,模型照抄上一版,那些实体就被硬塞进最接近的别的类型)。"""
    from ..ontology import is_upper_name

    named = [str(t) for t in ((prior or {}).get("entity_types") or []) if str(t).strip()]
    types = [t for t in named if not is_upper_name(t)]
    upper_named = [t for t in named if is_upper_name(t)]
    if not named:
        return ""
    observed = (prior or {}).get("observed") or {}
    lookup = {str(k).casefold(): int(v or 0) for k, v in (observed.get("type_counts") or {}).items()}
    text = ("\nThe previous version of this schema used these entity types: " + ", ".join(types)
            + ". Keep every name that still fits this corpus (renaming a type invalidates the existing graph), "
            "drop only types that do not occur in the corpus, and add what is missing.") if types else ""
    if upper_named:
        text += ("\nIt also used top-level class names as types: "
                 + ", ".join(f"{t} ({lookup[t.casefold()]} entities)" if t.casefold() in lookup else t for t in upper_named)
                 + ". Those are not allowed as types; add the more specific kinds of thing those entities are.")
    if lookup and types:
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
            prior: dict[str, Any] | None = None, progress=None) -> dict[str, Any]:
    """九次调用:领域 → 语言 → 人设 → 类型表 → 父类 → 谓词 → 类型定义与作用域 → 场景画像 → 语料示例(≤2 段各一次)。
    prior:现行模式版本(类型表、谓词表、画像、端点账),给了就让模型在它基础上修订;账里确认过的端点组合由
    apply_observed_guard 按规则保住,不依赖模型听不听话。
    返回 model(ontology.normalize_model 规范化过的模式字典)与领域、人设、语料示例等。"""
    from ..ontology import SchemaError, normalize_model

    progress = progress or (lambda phase: None)
    joined = " ".join(docs)
    progress("domain")
    if not domain:
        domain = clean_domain(client.chat(prompts.GENERATE_DOMAIN_PROMPT.format(input_text=joined), max_tokens=256))
    progress("language")
    language = client.chat(prompts.DETECT_LANGUAGE_PROMPT.format(input_text=joined), max_tokens=64).strip().strip('".')
    task = prompts.ENTITY_TYPE_TASK.format(domain=domain) + _prior_types_text(prior)
    progress("persona")
    persona = client.chat(prompts.GENERATE_PERSONA_PROMPT.format(sample_task=task), max_tokens=512).strip()
    progress("types")
    raw_types = client.chat(
        [{"role": "system", "content": persona},
         {"role": "user", "content": prompts.ENTITY_TYPE_GENERATION_PROMPT.format(task=task, input_text=joined)}],
        max_tokens=1024,
        validate=lambda text: bool(parse_entity_types(parse_json_object(text) or text)),
    )
    entity_types = parse_entity_types(parse_json_object(raw_types) or raw_types)
    parent_types: dict[str, str] = {}
    predicates: list[dict[str, Any]] = []
    guard: dict[str, Any] = {}
    if entity_types:
        progress("parents")
        raw_parents = client.chat(prompts.PARENT_TYPES_PROMPT.format(upper_parents=upper_parents_text(),
            persona=persona, domain=domain, entity_types="\n".join(f"- {t}" for t in entity_types)), max_tokens=1024)
        parent_types = normalize_parent_types(parse_json_object(raw_parents), entity_types)
        uppers = _uppers_of(entity_types, parent_types)
        parents_text = "\n".join(
            f"- {upper}: {', '.join(t for t in entity_types if uppers.get(t) == upper)}"
            for upper in dict.fromkeys(uppers[t] for t in entity_types)
        ) or "\n".join(f"- {t}" for t in entity_types)
        parents_text += _prior_predicates_text(prior)
        # 谓词表是抽取约束的一半;解析不出谓词的响应不进缓存、带纠正提示再要一次(ChatClient.validate)
        progress("predicates")
        raw_predicates = client.chat(prompts.PREDICATES_PROMPT.format(
            persona=persona, domain=domain, parent_types=parents_text, input_text=joined,
            min_predicates=MIN_PREDICATES, max_predicates=MAX_PREDICATES), max_tokens=4096,
            validate=lambda t: bool(normalize_predicates(parse_json_object(t))))
        predicates, guard = apply_observed_guard(normalize_predicates(parse_json_object(raw_predicates)), prior)
    type_definitions: dict[str, str] = {}
    type_scopes: dict[str, str] = {}
    profile: dict[str, Any] = {}
    example_text = ""
    example_stats: dict[str, int] = {}
    model: dict[str, Any] = {}
    if entity_types:
        typed = "\n".join(f"- {t} (parent: {parent_types.get(t, 'entity')})" for t in entity_types)
        try:
            progress("definitions")
            raw_defs = client.chat(prompts.TYPE_DEFINITIONS_PROMPT.format(
                persona=persona, domain=domain, entity_types=typed, input_text=joined), max_tokens=4096,
                validate=lambda t: bool(normalize_definitions(parse_json_object(t), entity_types)[0]))
            type_definitions, type_scopes = normalize_definitions(parse_json_object(raw_defs), entity_types)
        except (LLMInputBudgetExceeded, LLMInterrupted):
            raise
        except Exception as exc:
            print(f"[schema] type definitions call failed: {exc!r}", flush=True)
        try:
            progress("profile")
            raw_profile = client.chat(prompts.PROFILE_PROMPT.format(
                persona=persona, domain=domain, entity_types=typed,
                predicates=("\n".join(f"- {p['name']}: {p.get('description') or ''}" for p in predicates) or "- (none)")
                + _prior_profile_text(prior),
                input_text=joined), max_tokens=1024,
                validate=lambda t: "axis" in parse_json_object(t))
            profile = normalize_profile_suggestion(parse_json_object(raw_profile), entity_types=entity_types, predicates=predicates)
        except (LLMInputBudgetExceeded, LLMInterrupted):
            raise
        except Exception as exc:
            print(f"[schema] profile call failed: {exc!r}", flush=True)
        try:
            model, warnings = normalize_model({
                "language": language,
                "types": [{"name": t, "is_a": parent_types.get(t, "entity"), "description": type_definitions.get(t, ""),
                           "scope": type_scopes.get(t, "")} for t in entity_types],
                "predicates": predicates, "profile": profile})
        except SchemaError as exc:
            # 归纳结果不合规(父类成环、类型太多):类型表截到上限、父类一律拉平到上层类再规范化一次,不让整轮抽取作废
            print(f"[schema] induced schema rejected ({exc}); retrying flattened and trimmed", flush=True)
            from ..ontology import MAX_MENU_TYPES

            kept = entity_types[:MAX_MENU_TYPES]
            flat = _uppers_of(entity_types, {t: p for t, p in parent_types.items() if p not in entity_types})
            model, warnings = normalize_model({
                "language": language,
                "types": [{"name": t, "is_a": flat.get(t, "entity"), "description": type_definitions.get(t, ""),
                           "scope": type_scopes.get(t, "")} for t in kept],
                "predicates": predicates, "profile": profile})
        if warnings:
            print(f"[schema] normalized with warnings: {warnings}", flush=True)
        if examples:
            from .extract import ExtractionSchema

            schema = ExtractionSchema.from_model(model)
            progress("examples")
            example_text, example_stats = generate_examples(client, docs, schema)
    return {
        "domain": domain, "language": language, "persona": persona,
        "model": model,
        "examples": example_text, "example_stats": example_stats,
        "guard": guard,
    }


def _uppers_of(entity_types: list[str], parent_types: dict[str, str]) -> dict[str, str]:
    """父类可能是表内另一个类型:沿链走到上层类(谓词提示词按上层类分组)。"""
    from ..ontology import UPPER_CLASSES

    out: dict[str, str] = {}
    for t in entity_types:
        cur, hops = t, 0
        while cur not in UPPER_CLASSES and hops <= len(entity_types):
            cur = parent_types.get(cur) or "entity"
            hops += 1
        out[t] = cur if cur in UPPER_CLASSES else "entity"
    return out
