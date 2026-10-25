"""Graph recall: question → entity / relation vector seeds → Neo4j expansion along RELATED_TO →
aggregation into chunks.

Used in two places: the graph route of the search service (kb_search/channels.graph_channel) and `kb graph query`
for manual checks. Scoring follows plan 4.7: entity = sim × pagerank_norm; relation = sim × weight_norm; the score at hop
i = seed score / (2 + i); chunk score = the scores of the entities hitting it (via MENTIONED_IN, weighted by hit
count and idf) + relation scores (via EVIDENCES → CONTRIBUTES_TO, likewise by idf). It returns chunk ids with
scores so the layer above can treat it as one recall route and fuse it with vectors and BM25 by RRF.
"""
from __future__ import annotations

import json
import math
import re
import time
from typing import Any

from neo4j import Query

from ..config import Settings
from ..embedding.client import EmbeddingClient
from ..models import KBSource
from ..vector.qdrant import client as qdrant_client, graph_collection_alias, graph_collection_name
from .neo4j_import import active_neo4j_graph_version, neo4j_driver


def mention_weight(entity_score: float, *, count: int, df: int, n_chunks: int) -> float:
    """Contribution of one "entity → chunk" attribution to the chunk score: entity score × idf × saturated count.

    idf is the key. The device name in the question is a seed entity of every question and appears in hundreds
    of chunks: the document history page, the ordering information table and the contents page were pushed to
    the top on every question by it, while what really tells the answer apart are entities like PE# and tAS that
    appear in only a few chunks. In the evaluation of kb_003's 32 capability questions, the chunks from graph
    recall could not answer 10 of them without idf, although for several of those the relation seeds themselves
    had already hit the answer.
    """
    idf = math.log1p(max(1, n_chunks) / max(1, df))
    return float(entity_score) * idf * (1.0 + math.log1p(max(0, int(count or 0))))


def evidence_weight(relation_score: float, *, df: int, n_chunks: int) -> float:
    """Contribution of relation evidence (EVIDENCES → CONTRIBUTES_TO) to a chunk, with idf over the chunks it
    covers as well, so it stays on the same scale as entity attribution: a relation is usually supported by only
    one or two units (a few chunks), and that is exactly what pins the chunk to where the answer is."""
    idf = math.log1p(max(1, n_chunks) / max(1, df))
    return float(relation_score) * idf


# ── Second stage: re-rank candidate chunks by "question ↔ chunk text" ───────────────────────────────────
# The graph score only gathers the candidates (for an evaluation set of 26 answerable questions the top 60
# candidates always contain the answer chunk); ranking is left to the chunk's own similarity to the question +
# lexical overlap: hit@5 was 11/26 ordered by graph score and 20/26 after re-ranking, against 17/26 for vector
# search alone and 15/26 for BM25.
CANDIDATE_LIMIT = 60
TEXT_VECTOR_NAME = "text"     # name of the text vector in the main collection (two named vectors: text + visual)
LEX_WEIGHT = 0.3
GRAPH_PRIOR = 0.5
_CJK_STOP = {"是什", "什么", "么？", "的是", "在什", "如何", "哪些", "有何", "分别", "是多", "多少", "少？", "些？", "何不", "不同"}
_TOKEN_RE = re.compile(r"[A-Za-z0-9_#]+(?:[\[\]:./+-][A-Za-z0-9_#]+)*")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]{2,}")


def _tokens(text: str) -> tuple[set[str], set[str]]:
    alnum = {t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) >= 2}
    cjk: set[str] = set()
    for run in _CJK_RE.findall(text or ""):
        cjk.update(run[k:k + 2] for k in range(len(run) - 1))
    return alnum, cjk


def lexical_overlap(question: str, text: str) -> float:
    """Share of the question's symbols (alphanumeric runs such as part numbers, parameter names, pin names) and
    CJK bigrams that appear in the chunk text; alphanumeric runs weigh double, since almost all the
    discrimination in datasheet questions lies in symbols like tAS, PE#, IDCODE. Ranges 0–3."""
    qa, qc = _tokens(question)
    qc -= _CJK_STOP
    ta, tc = _tokens(text)
    a = len(qa & ta) / len(qa) if qa else 0.0
    c = len(qc & tc) / len(qc) if qc else 0.0
    return 2.0 * a + c


EXPAND_FETCH_FACTOR = 4         # 第一跳:每个实体先按关系强度取几倍邻边,再按与问题的相关度留 per_node 条(后面的跳数不多取)
EXPAND_RELEVANCE_WEIGHT = 3.0   # 没有问题向量时的退路:相关度系数 = 1 + 3 × 词重合,封顶 3
EXPAND_RELEVANCE_MAX = 3.0
# Relevance selects neighbours without multiplying their retrieval scores.


def expansion_relevance(question: str, anchor: str, neighbour: dict[str, Any]) -> float:
    """一跳邻边与问题的相关度系数(1–3):问题里去掉起点自己的名字,剩下的记号在「对端名字 + 关系类型 + 关系描述」里
    出现的比例。中心实体的每条边都带着它自己的名字,不去掉的话条条都算相关;问题只点了实体名时系数都是 1,
    排序和只按强度一样(强度高的边不一定与本次问题相关)。"""
    q = re.sub(re.escape(anchor), " ", question, flags=re.IGNORECASE) if anchor else question
    text = f"{neighbour.get('title') or ''} {neighbour.get('type') or ''} {neighbour.get('description') or ''}"
    return min(EXPAND_RELEVANCE_MAX, 1.0 + EXPAND_RELEVANCE_WEIGHT * lexical_overlap(q, text))


def relation_cosines(q: Any, relation_collection: str, graph_version: str, vector: list[float] | None,
                     rids: list[str]) -> dict[str, float]:
    """候选邻边与问题的余弦:关系向量建图时已经写进图集合,点编号由版本和关系编号算出来,一次取回;取不到就是空。"""
    if vector is None or not rids:
        return {}
    from .vectors import point_id_for

    ids = {point_id_for(graph_version, rid): rid for rid in dict.fromkeys(rids)}
    try:
        points = q.retrieve(collection_name=relation_collection, ids=list(ids), with_payload=False, with_vectors=True)
    except Exception:
        return {}
    out: dict[str, float] = {}
    for p in points or []:
        vec = p.vector
        if isinstance(vec, dict):
            vec = next(iter(vec.values()), None)
        rid = ids.get(str(p.id))
        if rid and vec is not None:
            out[rid] = _cosine(vector, list(vec))
    return out


def pick_neighbours(neighbours: list[dict[str, Any]], per_node: int, cosines: dict[str, float], question: str,
                    anchor: str) -> list[dict[str, Any]]:
    """一个实体留哪 per_node 条邻边:有关系向量时按与问题的余弦挑(平局按强度);没有(只走词法、向量取不回来)
    就按「强度 × 词重合系数」(expansion_relevance)。候选本来就是按强度取的前几倍,强度的先验已经在里面。"""
    if cosines:
        ranked = sorted(neighbours, key=lambda nb: (-cosines.get(str(nb.get("rid") or ""), -1.0), -float(nb.get("weight") or 1.0)))
    else:
        ranked = sorted(neighbours, key=lambda nb: -expansion_relevance(question, anchor, nb) * float(nb.get("weight") or 1.0))
    return ranked[: max(0, int(per_node))]


def rerank_score(cos: float, lex: float, graph_rank: int) -> float:
    """Final score of a candidate chunk: cosine similarity + lexical overlap + a small graph-rank prior (higher
    ranked candidates get a slight edge)."""
    return float(cos or 0.0) + LEX_WEIGHT * float(lex or 0.0) + GRAPH_PRIOR / (10.0 + max(1, int(graph_rank or 1)))


def _cosine(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


ENTITY_PART_WEIGHT = 0.5      # weight of entity attribution relative to relation evidence
ENTITY_TOP_K = 3              # a chunk is scored on its three strongest entity attributions only
EVIDENCE_TOP_K = 5            # likewise for relation evidence, the five strongest


def chunk_score(entity_weights: list[float], evidence_weights: list[float], *, hub: int, mean_hub: float) -> float:
    """Graph recall score of one chunk.

    Relation evidence is the main signal: the chunks of the units supporting a relation are where the answer is
    (the evidence for "tAS is a timing parameter" is in the AC table, that for "uses a 361-ball FCBGA" in the
    ordering table). Entity attribution is the secondary signal, and only the strongest few count: the document
    history page, the ordering information table and the contents page list dozens of names from a datasheet,
    and summing "how many seed entities are mentioned" ranked them first on every question; the chunk's own
    total mention count (hub) then applies a BM25-like length normalization.
    """
    ent = sorted((float(w) for w in entity_weights), reverse=True)[:ENTITY_TOP_K]
    evi = sorted((float(w) for w in evidence_weights), reverse=True)[:EVIDENCE_TOP_K]
    norm = 1.0 + math.log1p(max(0, int(hub or 0)) / max(mean_hub, 1e-9)) if mean_hub > 0 else 1.0
    return sum(evi) + ENTITY_PART_WEIGHT * sum(ent) / norm


LEXICAL_SEED_SCORE = {"title": 0.85, "alias": 0.8, "relation": 0.8}
LEXICAL_MAX_TOKENS = 8
LEXICAL_PER_TOKEN = 5


def identifier_variants(token: str) -> list[str]:
    """A symbol from the question may appear in the graph as any of tAS / TAS / t_AS."""
    tok = str(token or "").strip()
    out: list[str] = []
    for v in (tok, tok.upper(), tok.lower(), tok.replace("_", ""), tok.replace(" ", "")):
        if v and v not in out:
            out.append(v)
    return out


def lexical_tokens(question: str) -> list[str]:
    """Symbols worth an exact entity lookup by name: containing letters, not a bare number / unit (667, MHz do not
    count), at most a few."""
    from .merge import _VALUE_TOKEN_RE, question_identifiers

    out: list[str] = []
    for tok in question_identifiers(question):
        if len(tok) < 2 or _VALUE_TOKEN_RE.match(tok) or not any(ch.isalpha() for ch in tok):
            continue
        if sum(1 for ch in tok if ch.isalpha()) < 2:
            continue
        out.append(tok)
        if len(out) >= LEXICAL_MAX_TOKENS:
            break
    return out


def lexical_seeds(q: Any, entity_collection: str, relation_collection: str, question: str, *,
                  per_token: int = LEXICAL_PER_TOKEN, scope: dict[str, Any] | None = None,
                  vector: list[float] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """按问题中的记号精确找种子(标题、别名、外部编号以及关系端点),补充稠密召回。
    短记号在向量中可能分量不足,但对定位相关证据很重要。关系按时间口径筛(scope,默认现行)。
    端点匹配的关系在有问题向量时按语义相关度取前几条;否则按存储顺序取。
    这样中心实体有很多边时,仍能优先选中与问题有关的关系。分数口径保持 LEXICAL_SEED_SCORE。"""
    from qdrant_client.http import models as qm

    from .currency import in_scope, query_scope

    scope = scope or query_scope()
    tokens = lexical_tokens(question)
    if not tokens:
        return [], []
    exclude = seed_filter().must_not
    rel_must = relation_filter(scope).must
    entities: dict[str, dict[str, Any]] = {}
    relations: dict[str, dict[str, Any]] = {}
    for tok in tokens:
        variants = identifier_variants(tok)
        e_filter = qm.Filter(should=[qm.FieldCondition(key="title", match=qm.MatchAny(any=variants)),
                                     qm.FieldCondition(key="aliases", match=qm.MatchAny(any=variants)),
                                     qm.FieldCondition(key="external_ids", match=qm.MatchAny(any=variants))],
                             must_not=exclude)
        points, _ = q.scroll(collection_name=entity_collection, scroll_filter=e_filter, limit=per_token,
                             with_payload=True, with_vectors=False)
        for p in points:
            payload = dict(p.payload or {})
            eid = str(payload.get("gr_id") or "")
            if not eid or eid in entities:
                continue
            exact = str(payload.get("title") or "") in variants
            payload["_score"] = LEXICAL_SEED_SCORE["title" if exact else "alias"]
            payload["_point_id"] = str(p.id)
            payload["_token"] = tok
            entities[eid] = payload
        r_filter = qm.Filter(should=[qm.FieldCondition(key="source", match=qm.MatchAny(any=variants)),
                                     qm.FieldCondition(key="target", match=qm.MatchAny(any=variants))],
                             must=rel_must, must_not=exclude)
        found = None
        if vector is not None:
            try:
                found = _query_points(q, relation_collection, vector, per_token * 2, r_filter)
            except Exception:
                found = None
        if found is None:
            points, _ = q.scroll(collection_name=relation_collection, scroll_filter=r_filter, limit=per_token * 2,
                                 with_payload=True, with_vectors=False)
            found = [{**dict(p.payload or {}), "_point_id": str(p.id)} for p in points]
        for row in found:
            payload = dict(row)
            rid = str(payload.get("gr_id") or "")
            if not rid or rid in relations or not in_scope(payload, scope):
                continue
            payload["_score"] = LEXICAL_SEED_SCORE["relation"]
            payload["_token"] = tok
            relations[rid] = payload
    return list(entities.values()), list(relations.values())


SPEC_VECTOR_NAMES = ("text", "property", "value")
WINDOW_IN_BOOST = 1.1
WINDOW_OUT_PENALTY = 0.8


def _spec_dense_rows(q: Any, spec_collection: str, vector: list[float], limit: int, query_filter: Any = None) -> list[dict[str, Any]]:
    """Dense hits in the facts collection: new collections have three named vectors text / property / value,
    queried separately and merged (the same fact takes its highest score, plus a small bonus when both the property
    and the value side hit); old collections have a single vector, queried as the unnamed vector."""
    merged: dict[str, dict[str, Any]] = {}
    named_ok = False
    for name in SPEC_VECTOR_NAMES:
        try:
            rows = _query_points(q, spec_collection, vector, limit, query_filter, using=name)
        except Exception:
            break
        named_ok = True
        for r in rows:
            sid = str(r.get("gr_id") or "")
            if not sid:
                continue
            cur = merged.get(sid)
            if cur is None:
                r["_via"] = f"vector:{name}"
                r["_named_hits"] = {name}
                merged[sid] = r
            else:
                cur["_named_hits"].add(name)
                if float(r["_score"]) > float(cur["_score"]):
                    cur["_score"] = float(r["_score"])
                    cur["_via"] = f"vector:{name}"
    if not named_ok:
        rows = _query_points(q, spec_collection, vector, limit, query_filter)
        for r in rows:
            sid = str(r.get("gr_id") or "")
            if sid:
                r["_via"] = "vector"
                merged[sid] = r
        return list(merged.values())
    for r in merged.values():
        hits = r.pop("_named_hits", set())
        if {"property", "value"} <= hits:
            r["_score"] = min(1.0, float(r["_score"]) + 0.05)     # both property and value side match: more likely the answer
        r["_via"] = "vector"
    return sorted(merged.values(), key=lambda r: -float(r["_score"]))[:limit]


def greedy_cover(seeds: list[dict[str, Any]], question: str, *, limit: int) -> list[dict[str, Any]]:
    """Greedy cover (OG-RAG): cover the question's symbols and CJK bigrams with as few facts as possible, each
    time picking the fact covering the most uncovered tokens, ties going to the higher score; once everything is
    covered or the limit is reached, the rest are appended by score. Returns the re-ordered list."""
    qa, qc = _tokens(question)
    needles = {t for t in qa | (qc - _CJK_STOP) if len(t) >= 2}
    if not needles or not seeds:
        return seeds
    fields = {}
    for s in seeds:
        text = " ".join(str(s.get(k) or "") for k in ("subject", "property", "symbol", "concept", "values_text", "conditions_text", "search_text"))
        ta, tc = _tokens(text)
        fields[id(s)] = (ta | tc) & needles
    uncovered = set(needles)
    chosen: list[dict[str, Any]] = []
    pool = list(seeds)
    while pool and uncovered and len(chosen) < limit:
        best = max(pool, key=lambda s: (len(fields[id(s)] & uncovered), float(s.get("_score") or 0)))
        gain = fields[id(best)] & uncovered
        if not gain:
            break
        chosen.append(best)
        pool.remove(best)
        uncovered -= gain
    rest = sorted(pool, key=lambda s: -float(s.get("_score") or 0))
    return (chosen + rest)[:max(limit, len(chosen))]


def fact_content_key(row: dict[str, Any]) -> tuple:
    """一条事实说的内容:主体、属性(有概念键用概念键)、取值、时间,加上真正起区分作用的条件 ——
    条件值已经写在主体或属性名里的(「位点名称: rs0000111」配属性「GENE_A rs0000111 …」)不算区别。"""
    subject = str(row.get("subject_id") or row.get("subject") or "").casefold()
    concept = str(row.get("concept_key") or row.get("property") or "").casefold()
    named = f"{row.get('subject') or ''} {row.get('property') or ''} {row.get('symbol') or ''}".casefold()
    conditions = row.get("conditions") or {}
    if isinstance(conditions, str):
        try:
            conditions = json.loads(conditions)
        except ValueError:
            conditions = {"_": conditions}
    kept = tuple(sorted((str(k).casefold(), str(v).casefold()) for k, v in dict(conditions).items()
                        if str(v).strip() and str(v).casefold() not in named))
    values = tuple(str(row.get(k) or "").strip().casefold() for k in ("value", "min", "typ", "max", "unit"))
    return subject, concept, values, str(row.get("valid_from") or row.get("axis") or ""), kept


def collapse_duplicate_facts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """内容相同的几条事实(同一行在报告里重复列了几次、两份报告各抄了一遍)只留排在最前的一条,来源切片并进去、
    记下重复了几条。rows 已按分数从高到低排(10-24 评估:同一个事实在证据里占了 12 条里的 8 条)。"""
    out: list[dict[str, Any]] = []
    first: dict[tuple, dict[str, Any]] = {}
    for r in rows:
        key = fact_content_key(r)
        keep = first.get(key)
        if keep is None:
            first[key] = r
            out.append(r)
            continue
        keep["point_ids"] = list(dict.fromkeys([*(keep.get("point_ids") or []), *(r.get("point_ids") or [])]))
        keep["_duplicates"] = int(keep.get("_duplicates") or 0) + 1
    return out


def spec_seeds(q: Any, spec_collection: str, vector: list[float] | None, question: str, *, limit: int = 12,
               per_token: int = LEXICAL_PER_TOKEN, window: dict[str, str] | None = None,
               scope: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """限定事实种子:稠密检索(属性侧 / 值侧 / 整句)+ 记号精确匹配(主体 / 符号 / 属性)+ 时间窗加权 + 贪心覆盖。
    按时间口径筛(scope,默认现行:序列里更早的档位、来源文档被取代或归档的、过了有效期的不作种子)。
    集合不存在(旧图)就返回空。"""
    from qdrant_client.http import models as qm

    from .currency import in_scope, query_scope
    from .temporal import in_window

    scope = scope or query_scope()
    flt_status = relation_filter(scope)
    try:
        rows = _spec_dense_rows(q, spec_collection, vector, limit, flt_status) if vector is not None else []
    except Exception:
        return []
    seeds: dict[str, dict[str, Any]] = {}
    for r in rows:
        sid = str(r.get("gr_id") or "")
        if sid and in_scope(r, scope):
            seeds[sid] = r
    # Lexical lookup by symbol / property name only: a subject name (some product) would hit all of its facts,
    # which is no relevance evidence and only counts as a bonus
    for tok in lexical_tokens(question):
        variants = identifier_variants(tok)
        flt = qm.Filter(should=[qm.FieldCondition(key="symbol", match=qm.MatchAny(any=variants)),
                                qm.FieldCondition(key="property", match=qm.MatchAny(any=variants))],
                        must=flt_status.must)
        try:
            points, _ = q.scroll(collection_name=spec_collection, scroll_filter=flt, limit=per_token * 4,
                                 with_payload=True, with_vectors=False)
        except Exception:
            break
        for p in points:
            payload = dict(p.payload or {})
            sid = str(payload.get("gr_id") or "")
            if not sid or not in_scope(payload, scope):
                continue
            score = LEXICAL_SEED_SCORE["title"]
            cur = seeds.get(sid)
            if cur is None:
                payload["_score"] = score
                payload["_point_id"] = str(p.id)
                payload["_via"] = "lexical"
                payload["_token"] = tok
                seeds[sid] = payload
            elif cur.get("_via") == "vector":
                # Both dense and lexical hit: the two signals stack, ranked first
                cur["_score"] = min(1.0, float(cur["_score"]) + 0.1)
                cur["_via"] = "vector+lexical"
    # The more of the question's symbols a fact hits, the likelier it is the answer (subject + symbol matches rank
    # first)
    tokens_cf = {t.casefold() for t in lexical_tokens(question)}
    for r in seeds.values():
        fields = [str(r.get("subject") or ""), str(r.get("symbol") or ""), str(r.get("property") or ""), str(r.get("concept") or "")]
        hits = sum(1 for t in tokens_cf if any(t == f.casefold() or t in f.casefold() for f in fields))
        r["_hits"] = hits
        r["_score"] = float(r["_score"]) + 0.05 * hits
        # The question has a time window ("these two years", 2024 to 2026): facts inside get a small boost, those
        # outside a small penalty, no filtering
        if window:
            when = str(r.get("valid_from") or r.get("axis") or "")
            r["_score"] = float(r["_score"]) * (WINDOW_IN_BOOST if in_window(when, window) else WINDOW_OUT_PENALTY)
    ranked = collapse_duplicate_facts(sorted(seeds.values(), key=lambda r: -float(r["_score"])))
    return greedy_cover(ranked[: limit * 2], question, limit=limit)


SPEC_EVIDENCE_LIMIT = 4
SPEC_EVIDENCE_MIN_SCORE = 0.8
CONCLUSION_BOOST = 1.25


def spec_evidence_seeds(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Which facts' source chunks enter the candidates: only the top few with a high enough score (dense similarity
    or symbol hit). The facts themselves are all still returned to the caller; this only limits their influence on
    chunk ranking, so a pile of weakly related table pages does not push out the body text."""
    return [sp for sp in specs[:SPEC_EVIDENCE_LIMIT] if float(sp.get("_score") or 0) >= SPEC_EVIDENCE_MIN_SCORE]


def graph_collections_for(source_collection: str, graph_version: str | None = None) -> dict[str, str]:
    """The four graph collections to query (entities, relations, facts, pages): without a version, go through the
    aliases (the active version); with one, query by version name directly (to evaluate a trial build whose
    aliases were not switched, or to compare two versions)."""
    types = ("entity", "relation", "spec", "page")
    if graph_version:
        return {t: graph_collection_name(source_collection, t, graph_version) for t in types}
    return {t: graph_collection_alias(source_collection, t) for t in types}


PAGE_SEED_LIMIT = 6
PAGE_TEXT_LIMIT = 6000
PAGE_EVIDENCE_MIN_SCORE = 0.6


def page_seeds(q: Any, page_collection: str, vector: list[float], question: str, *,
               limit: int = PAGE_SEED_LIMIT) -> list[dict[str, Any]]:
    """Dense hits among view pages (subject / timeline / source / index pages); returns empty when the collection
    does not exist. Not filtered by the question's time window: timeline and subject pages span time by nature."""
    try:
        rows = _query_points(q, page_collection, vector, limit)
    except Exception:
        return []
    for r in rows:
        r["_via"] = "vector"
        r["lex"] = round(lexical_overlap(question, str(r.get("text") or "")), 4)
        r["_score"] = float(r["_score"]) + 0.1 * float(r["lex"])
    return sorted(rows, key=lambda r: -float(r["_score"]))


def seed_filter() -> Any:
    """Seeds only come from body-text entities / relations: those found only on boilerplate pages (contents,
    revision records), references and demoted types are never seeds. Older graphs lack these two fields, and
    must_not does not apply to points missing the field, so their behaviour is unchanged."""
    from qdrant_client.http import models as qm

    return qm.Filter(must_not=[
        qm.FieldCondition(key="boilerplate", match=qm.MatchValue(value=True)),
        qm.FieldCondition(key="reference", match=qm.MatchValue(value=True)),
    ])


def relation_filter(scope: dict[str, Any] | None = None) -> Any:
    """关系 / 事实种子:在 seed_filter 之上按时间口径先在 Qdrant 里筛掉被取代的、来源文档标了状态的(现行口径);
    回看某一天与 history 不在这里筛,取回来按 currency.in_scope 判。"""
    from qdrant_client.http import models as qm

    from .currency import qdrant_current_conditions

    base = seed_filter()
    must = qdrant_current_conditions() if (scope or {"mode": "current"})["mode"] == "current" else None
    return qm.Filter(must=must, must_not=base.must_not)


def _query_points(q: Any, collection: str, vector: list[float], limit: int, query_filter: Any = None,
                  using: str | None = None) -> list[dict[str, Any]]:
    """using: the name of a named vector (text / property / value of the facts collection); None queries the
    unnamed vector."""
    try:
        kwargs = {"using": using} if using else {}
        response = q.query_points(collection_name=collection, query=vector, limit=limit, with_payload=True,
                                  query_filter=query_filter, **kwargs)
        points = response.points
    except AttributeError:
        points = q.search(collection_name=collection, query_vector=(using, vector) if using else vector, limit=limit,
                          with_payload=True, query_filter=query_filter)
    out: list[dict[str, Any]] = []
    for p in points:
        payload = dict(p.payload or {})
        payload["_score"] = float(p.score)
        payload["_point_id"] = str(p.id)
        out.append(payload)
    return out


SOURCE_SAMPLE = 16
SERIES_MEMBERS_MAX = 40
RELATION_TIME_KEYS = ("valid_from", "valid_until", "observed_first", "observed_last", "superseded_by", "superseded_on", "doc_status",
                      "lapses")


def spread(items: list[Any], limit: int) -> list[Any]:
    """With more than limit items, take limit of them at even spacing: taking the first limit would put the whole
    sample on the first few documents."""
    return items if len(items) <= limit else [items[i * len(items) // limit] for i in range(limit)]


def source_points(rows: list[dict[str, Any]], owner_key: str, *, limit: int = SOURCE_SAMPLE) -> dict[str, list[str]]:
    """Each entity / relation carries a few representative source points, so the search service can check that
    its sources are still active (when a document is deleted or re-parsed, the graph only catches up with the next
    version). One point per document: a deleted document, changed content or an upgraded parse profile all replace
    the chunks of a whole document at once. With many documents they are sorted by path and taken at even spacing."""
    by_owner: dict[str, dict[str, str]] = {}
    for row in rows:
        pid = str(row.get("point_id") or "")
        if pid:
            by_owner.setdefault(str(row[owner_key]), {}).setdefault(str(row.get("rel_path") or pid), pid)
    return {owner: [by_doc[d] for d in spread(sorted(by_doc), limit)] for owner, by_doc in by_owner.items()}


def spec_result_row(sp: dict[str, Any]) -> dict[str, Any]:
    """Payload of one point of the fact collection -> the fact row handed to the caller. Shared by graph recall
    and the "facts by subject / property" endpoint (kb_search.graphwalk), so the two row shapes cannot drift
    apart."""
    return {
        "id": sp.get("gr_id"), "subject": sp.get("subject"), "property": sp.get("property"), "symbol": sp.get("symbol"),
        "concept": sp.get("concept"), "concept_key": sp.get("concept_key"),
        "value": sp.get("value"), "min": sp.get("min"), "typ": sp.get("typ"), "max": sp.get("max"), "unit": sp.get("unit"),
        "flag": sp.get("flag"), "ref_min": sp.get("ref_min"), "ref_max": sp.get("ref_max"), "bound_distance": sp.get("bound_distance"),
        "when": sp.get("when"), "valid_from": sp.get("valid_from"), "valid_until": sp.get("valid_until"),
        "series_key": sp.get("series_key"), "conflict_group": sp.get("conflict_group"),
        "conditions": sp.get("conditions") or {}, "conditions_text": sp.get("conditions_text"), "text": sp.get("text"), "note": sp.get("note"),
        "unit_canonical": sp.get("unit_canonical"), "kinds": sp.get("kinds") or {}, "quality": sp.get("quality"),
        "evidence_conflict": sp.get("evidence_conflict"), "comparable": sp.get("comparable"), "confidence": sp.get("confidence"),
        "series_index": sp.get("series_index"), "series_len": sp.get("series_len"), "axis": sp.get("axis"), "values_text": sp.get("values_text"),
        "rel_path": sp.get("rel_path"), "section": sp.get("section"), "point_ids": list(sp.get("point_ids") or []),
        # 时间与状态(方案附录第 3 条):观察时间、被哪条取代、来源文档状态、出自表格 / 叙述
        "observed": sp.get("observed"), "superseded_by": sp.get("superseded_by"), "superseded_on": sp.get("superseded_on"),
        "doc_status": sp.get("doc_status"), "source_kind": sp.get("source_kind"), "lapses": sp.get("lapses"),
    }


class TimedSession:
    """Adds a transaction timeout to every query of the session: Neo4j sets no limit by default, and a query
    keeps running after the caller has given up at its deadline, holding a connection and a thread."""

    def __init__(self, session: Any, timeout: float) -> None:
        self._session, self._timeout = session, float(timeout)

    def run(self, text: str, **params: Any) -> Any:
        return self._session.run(Query(text, timeout=self._timeout), **params)


STATS_TTL = 600.0
_version_stats: dict[str, tuple[str, float, dict[str, float]]] = {}      # kb_id -> (graph version, computed at, statistics)


def version_stats(session: Any, kb_id: str, gv: str) -> dict[str, float]:
    """Three whole-graph statistics of a graph version: the maximum pagerank, the number of chunks and the total
    number of mentions. A version's content no longer changes once it is active, so they need not be recomputed
    for every query (the mention total expands every MENTIONED_IN, about half a second on a large KB); each KB
    keeps one copy, for its active version, recomputed after a while so that a version re-imported by hand is
    picked up too. The mention total carries e.id IS NOT NULL so that the (kb_id, graph_version, id) composite
    index is used: with only the first two equality conditions the plan falls back to a label scan across all KBs."""
    cached = _version_stats.get(kb_id)
    if cached and cached[0] == gv and time.time() - cached[1] < STATS_TTL:
        return cached[2]
    max_rank = session.run(
        "MATCH (e:Entity {kb_id: $kb, graph_version: $gv}) RETURN max(e.pagerank) AS m", kb=kb_id, gv=gv,
    ).single()
    n_chunks = session.run(
        "MATCH (c:QdrantChunkSnapshot {kb_id: $kb, graph_version: $gv}) RETURN count(c) AS n", kb=kb_id, gv=gv,
    ).single()
    total_mentions = session.run(
        "MATCH (e:Entity {kb_id: $kb, graph_version: $gv})-[m:MENTIONED_IN]->() WHERE e.id IS NOT NULL RETURN count(m) AS n",
        kb=kb_id, gv=gv,
    ).single()
    stats = {"max_rank": float((max_rank or {}).get("m") or 0.0) or 1.0,
             "n_chunks": int((n_chunks or {}).get("n") or 0),
             "total_mentions": int((total_mentions or {}).get("n") or 0)}
    _version_stats[kb_id] = (gv, time.time(), stats)
    return stats


def graph_query(
    settings: Settings,
    source: KBSource,
    question: str,
    *,
    top_entities: int = 12,
    top_relations: int = 12,
    hops: int = 2,
    per_node: int = 8,
    chunk_limit: int = 12,
    graph_version: str | None = None,
    with_text: bool = True,
    explain: bool = False,
    candidate_limit: int = CANDIDATE_LIMIT,
    vector: list[float] | None = None,
    lexical_only: bool = False,
    q: Any = None,
    driver: Any = None,
    timeout: float | None = None,
    history: bool = False,
    as_of: str | None = None,
) -> dict[str, Any]:
    """history / as_of:关系与事实的时间口径(currency.query_scope;默认现行)。实体没有时间,不受影响。
    vector:调用方已经算好的问题向量(检索服务里一次嵌入三路共用);lexical_only:嵌入服务不可用时
    只走词法种子(记号精确匹配实体 / 关系 / 事实的符号与属性),稠密种子、页面与余弦项全部为空(Q19)。
    q / driver:常驻的检索服务把自己共用的 Qdrant 客户端和 Neo4j driver 传进来;不传(命令行)就现建、用完关掉。
    timeout:图路各条 Neo4j 查询的事务超时(秒),查现行版本号的那一条点查除外;不传不限时。Qdrant 那边的超时跟着传进来的客户端走。"""
    from .currency import cypher_in_scope, in_scope, mark_status, query_scope
    from .temporal import question_time_window

    scope = query_scope(history=history, as_of=as_of)
    from kb_search import docstatus
    if scope["mode"] == "current" and docstatus.graph_refresh_pending(settings, source):
        return {"kb_id": source.kb_id, "degraded": ["graph_refresh_pending"], "entities": [], "relations": [], "specs": [], "pages": [], "chunks": []}
    if vector is None and not lexical_only:
        embedder = EmbeddingClient(
            base_url=settings.embedding_base_url, api_key=settings.embedding_api_key,
            model_id=settings.embedding_model_id, dim=settings.embedding_dim, batch_size=1,
        )
        vector = embedder.embed([question])[0]
    if q is None:
        q = qdrant_client(settings.qdrant_url, settings.qdrant_api_key)
    seeds_only = seed_filter()
    collections = graph_collections_for(source.collection, graph_version)
    entity_alias, relation_alias = collections["entity"], collections["relation"]
    window = question_time_window(question)
    if vector is not None:
        entity_seeds = _query_points(q, entity_alias, vector, top_entities, seeds_only)
        relation_seeds = [r for r in _query_points(q, relation_alias, vector, top_relations, relation_filter(scope)) if in_scope(r, scope)]
    else:
        entity_seeds, relation_seeds = [], []
    # 词法种子:问题里的记号按名字精确命中的实体 / 关系,与稠密种子合并(同 id 取高分)
    lex_entities, lex_relations = lexical_seeds(q, entity_alias, relation_alias, question, scope=scope, vector=vector)
    dense_e = {str(s.get("gr_id")): s for s in entity_seeds}
    for s in lex_entities:
        cur = dense_e.get(str(s.get("gr_id")))
        if cur is None or float(s["_score"]) > float(cur["_score"]):
            s["_via"] = "lexical"
            dense_e[str(s.get("gr_id"))] = s
    entity_seeds = list(dense_e.values())
    dense_r = {str(s.get("gr_id")): s for s in relation_seeds}
    for s in lex_relations:
        cur = dense_r.get(str(s.get("gr_id")))
        if cur is None or float(s["_score"]) > float(cur["_score"]):
            s["_via"] = "lexical"
            dense_r[str(s.get("gr_id"))] = s
    relation_seeds = list(dense_r.values())
    specs = spec_seeds(q, collections["spec"], vector, question, limit=top_relations, window=window, scope=scope)
    pages = page_seeds(q, collections["page"], vector, question) if vector is not None else []
    doc_paths: dict[str, str] = {}          # 页面 doc_ids -> 文件路径;评测按它判主体页出自哪份文档
    series_of: dict[str, list[dict[str, Any]]] = {}      # 事实序列键 -> 全部档位(不按时间口径筛)
    seed_summary = {"entities": len(entity_seeds), "relations": len(relation_seeds), "specs": len(specs), "pages": len(pages),
                    "lexical_entities": sum(1 for s in entity_seeds if s.get("_via") == "lexical"), "lexical_only": vector is None}

    own_driver = driver is None
    if own_driver:
        driver = neo4j_driver(settings)
    try:
        gv = graph_version or active_neo4j_graph_version(driver, source.kb_id)
        if not gv:
            raise RuntimeError(f"Neo4j has no active graph version for {source.kb_id}")
        if entity_seeds and str(entity_seeds[0].get("graph_version") or gv) != gv:
            raise RuntimeError("The Qdrant graph collections and the active Neo4j version disagree; the alias switch may be incomplete")

        entity_scores: dict[str, dict[str, Any]] = {}
        for seed in entity_seeds:
            eid = str(seed.get("gr_id") or "")
            if not eid:
                continue
            entity_scores[eid] = {"id": eid, "title": seed.get("title"), "type": seed.get("type"),
                                  "upper": seed.get("upper"), "kind": seed.get("kind"),
                                  "description": seed.get("description"), "scope": seed.get("scope"),
                                  "score": seed["_score"], "hop": 0, "via": seed.get("_via") or "vector"}
        relation_scores: dict[str, dict[str, Any]] = {}
        for seed in relation_seeds:
            rid = str(seed.get("gr_id") or "")
            if rid:
                relation_scores[rid] = {"id": rid, "source": seed.get("source"), "target": seed.get("target"),
                                        "type": seed.get("type"), "description": seed.get("description"),
                                        "score": seed["_score"], "hop": 0, "via": seed.get("_via") or "vector",
                                        **{k: seed.get(k) for k in RELATION_TIME_KEYS if seed.get(k)}}
                for end in (seed.get("source_id"), seed.get("target_id")):
                    end = str(end or "")
                    if end and end not in entity_scores:
                        entity_scores[end] = {"id": end, "title": None, "type": None, "score": seed["_score"] / 2,
                                              "hop": 0, "via": "relation"}

        with driver.session() as session:
            if timeout:
                session = TimedSession(session, timeout)
            stats = version_stats(session, source.kb_id, gv)
            max_rank = stats["max_rank"]          # pagerank normalization factor
            frontier = list(entity_scores)
            seed_ids = set(entity_scores)          # relations with both ends seeded (in-network) do not decay by hop (GraphRAG local)
            for hop in range(1, max(0, int(hops)) + 1):
                if not frontier:
                    break
                rows = session.run(
                    """
                    UNWIND $ids AS eid
                    MATCH (e:Entity {kb_id: $kb, graph_version: $gv, id: eid})-[r:RELATED_TO]-(n:Entity)
                    WHERE coalesce(r.boilerplate, false) = false AND coalesce(n.reference, false) = false
                      AND """ + cypher_in_scope("r", observed="observed_first", scope=scope) + """
                    WITH eid, r, n ORDER BY r.weight DESC
                    WITH eid, collect({rid: r.id, type: r.type, weight: r.weight, description: r.description,
                                       valid_from: r.valid_from, valid_until: r.valid_until, observed_first: r.observed_first,
                                       observed_last: r.observed_last, superseded_by: r.superseded_by, superseded_on: r.superseded_on,
                                       doc_status: r.doc_status, lapses: r.lapses,
                                       nid: n.id, title: n.title, ntype: n.type, nupper: n.upper, nkind: n.kind, scope: n.scope,
                                       pagerank: n.pagerank, degree: n.degree})[0..$k] AS neighbours
                    RETURN eid, neighbours
                    """,
                    ids=frontier, kb=source.kb_id, gv=gv, k=int(per_node) * (EXPAND_FETCH_FACTOR if hop == 1 else 1), day=scope["day"],
                ).data()
                next_frontier: list[str] = []
                cosines = relation_cosines(q, relation_alias, gv, vector,
                                           [str(nb.get("rid") or "") for row in rows for nb in row["neighbours"] or [] if nb.get("rid")]) \
                    if hop == 1 and EXPAND_FETCH_FACTOR > 1 else {}
                for row in rows:
                    base = entity_scores.get(str(row["eid"]), {}).get("score", 0.0)
                    anchor = str(entity_scores.get(str(row["eid"]), {}).get("title") or "")
                    for nb in pick_neighbours(list(row["neighbours"] or []), per_node, cosines, question, anchor):
                        nid = str(nb["nid"])
                        rank = float(nb.get("pagerank") or 0.0) / max_rank
                        degree = int(nb.get("degree") or 0)
                        hub_penalty = 1.0 / (1.0 + math.log1p(max(0, degree - 20))) if degree > 20 else 1.0
                        score = base / (2 + hop) * (0.5 + 0.5 * rank) * hub_penalty
                        slot = entity_scores.get(nid)
                        if slot is None or score > slot["score"]:
                            if slot is None:
                                next_frontier.append(nid)
                            entity_scores[nid] = {"id": nid, "title": nb.get("title"), "type": nb.get("ntype"),
                                                  "upper": nb.get("nupper"), "kind": nb.get("nkind"),
                                                  "scope": nb.get("scope"), "score": score, "hop": hop, "via": "expand"}
                        rid = str(nb.get("rid") or "")
                        if rid:
                            in_network = nid in seed_ids and hop == 1
                            rscore = (base if in_network else base / (2 + hop)) * (float(nb.get("weight") or 1.0) / 10.0)
                            rslot = relation_scores.get(rid)
                            if rslot is None or rscore > rslot["score"]:
                                relation_scores[rid] = {"id": rid, "source": None, "target": None,
                                                        "type": nb.get("type"), "description": nb.get("description"),
                                                        "score": rscore, "hop": hop, "via": "expand",
                                                        **{k: nb.get(k) for k in RELATION_TIME_KEYS if nb.get(k)}}
                frontier = next_frontier
            # Fill in missing titles
            missing = [eid for eid, s in entity_scores.items() if not s.get("title")]
            if missing:
                for row in session.run(
                    "UNWIND $ids AS eid MATCH (e:Entity {kb_id: $kb, graph_version: $gv, id: eid}) "
                    "RETURN eid, e.title AS title, e.type AS type, e.upper AS upper, e.kind AS kind, e.scope AS scope",
                    ids=missing, kb=source.kb_id, gv=gv,
                ).data():
                    entity_scores[str(row["eid"])].update({"title": row["title"], "type": row["type"], "upper": row.get("upper"),
                                                           "kind": row.get("kind"), "scope": row.get("scope")})
            rel_missing = [rid for rid, s in relation_scores.items() if not s.get("source")]
            if rel_missing:
                for row in session.run(
                    "UNWIND $ids AS rid MATCH (r:Relation {kb_id: $kb, graph_version: $gv, id: rid}) "
                    "RETURN rid, r.source AS source, r.target AS target", ids=rel_missing, kb=source.kb_id, gv=gv,
                ).data():
                    relation_scores[str(row["rid"])].update({"source": row["source"], "target": row["target"]})
            # 事实序列的全部档位:现行口径下更早的档位不作种子,现行那一条的 series_text 里仍要看得到它们
            series_keys = sorted({str(sp.get("series_key")) for sp in specs[:top_relations] if sp.get("series_key")})
            if series_keys:
                for row in session.run(
                    "MATCH (f:Spec {kb_id: $kb, graph_version: $gv}) WHERE f.series_key IN $keys "
                    "RETURN f.series_key AS series_key, f.id AS id, f.value AS value, f.min AS min, f.typ AS typ, f.max AS max, "
                    "f.unit AS unit, f.unit_canonical AS unit_canonical, f.valid_from AS valid_from, f.axis AS axis, "
                    "f.series_index AS series_index ORDER BY series_key, coalesce(f.series_index, 0), f.id",
                    kb=source.kb_id, gv=gv, keys=series_keys,
                ).data():
                    members = series_of.setdefault(str(row["series_key"]), [])
                    if len(members) < SERIES_MEMBERS_MAX:
                        members.append({**{k: row.get(k) for k in ("id", "value", "min", "typ", "max", "unit", "unit_canonical",
                                                                    "valid_from", "series_index")},
                                        "when": row.get("valid_from") or row.get("axis")})
            page_doc_ids = sorted({str(d) for pg in pages for d in (pg.get("doc_ids") or [])})
            if page_doc_ids:
                for row in session.run(
                    "MATCH (d:Document {kb_id: $kb, graph_version: $gv}) WHERE d.doc_id IN $ids "
                    "RETURN d.doc_id AS doc_id, d.rel_path AS rel_path", ids=page_doc_ids, kb=source.kb_id, gv=gv,
                ).data():
                    if row.get("rel_path"):
                        doc_paths[str(row["doc_id"])] = str(row["rel_path"])

            # Chunk aggregation: entity attribution and relation evidence each weighted by idf (see mention_weight /
            # evidence_weight)
            n_chunks = stats["n_chunks"]
            mean_hub = stats["total_mentions"] / n_chunks if n_chunks else 0.0
            mention_rows = session.run(
                """
                UNWIND $ids AS eid
                MATCH (e:Entity {kb_id: $kb, graph_version: $gv, id: eid})-[m:MENTIONED_IN]->(c:QdrantChunkSnapshot)
                WHERE coalesce(c.kind, 'body') <> 'boilerplate'
                RETURN eid, c.point_id AS point_id, c.chunk_uid AS chunk_uid, c.rel_path AS rel_path, m.count AS count,
                       size([(c)<-[:MENTIONED_IN]-() | 1]) AS hub, coalesce(c.kind, 'body') AS kind
                """,
                ids=list(entity_scores), kb=source.kb_id, gv=gv,
            ).data()
            entity_df: dict[str, int] = {}
            entity_docs: dict[str, set[str]] = {}
            for row in mention_rows:
                entity_df[str(row["eid"])] = entity_df.get(str(row["eid"]), 0) + 1
                if row.get("rel_path"):
                    entity_docs.setdefault(str(row["eid"]), set()).add(str(row["rel_path"]))
            entity_points = source_points(mention_rows, "eid")
            for eid, slot in entity_scores.items():
                slot["docs"] = sorted(entity_docs.get(eid, ()))      # docs of the chunks mentioning it; evaluation locates a cited entity's document by it
                slot["point_ids"] = entity_points.get(eid, [])
            chunk_scores: dict[str, dict[str, Any]] = {}
            parts: dict[str, dict[str, Any]] = {}
            for row in mention_rows:
                eid = str(row["eid"])
                escore = entity_scores[eid]["score"]
                pid = str(row["point_id"])
                slot = chunk_scores.setdefault(pid, {"point_id": pid, "chunk_uid": row.get("chunk_uid"),
                                                     "rel_path": row.get("rel_path"), "score": 0.0, "entities": [], "relations": []})
                part = parts.setdefault(pid, {"entity": [], "evidence": [], "hub": int(row.get("hub") or 0),
                                              "entity_detail": [], "evidence_detail": []})
                weight = mention_weight(escore, count=int(row.get("count") or 0), df=entity_df.get(eid, 1), n_chunks=n_chunks)
                if str(row.get("kind") or "body") == "conclusion":
                    weight *= CONCLUSION_BOOST          # the document's own conclusion section: the same hit is likelier the answer
                part["entity"].append(weight)
                if explain:
                    part["entity_detail"].append({**{k: entity_scores[eid].get(k) for k in ("title", "type", "score", "hop", "via")},
                                                  "count": int(row.get("count") or 0), "df": entity_df.get(eid, 1), "weight": weight})
                slot["entities"].append(entity_scores[eid].get("title"))
            if relation_scores:
                evidence_rows = session.run(
                    """
                    UNWIND $ids AS rid
                    MATCH (tu:TextUnit {kb_id: $kb, graph_version: $gv})-[:EVIDENCES]->(r:Relation {kb_id: $kb, graph_version: $gv, id: rid})
                    WHERE coalesce(tu.kind, 'body') <> 'boilerplate'
                    MATCH (c:QdrantChunkSnapshot)-[:CONTRIBUTES_TO]->(tu)
                    RETURN rid, c.point_id AS point_id, c.chunk_uid AS chunk_uid, c.rel_path AS rel_path, coalesce(tu.kind, 'body') AS kind
                    """,
                    ids=list(relation_scores), kb=source.kb_id, gv=gv,
                ).data()
                relation_df: dict[str, int] = {}
                for row in evidence_rows:
                    relation_df[str(row["rid"])] = relation_df.get(str(row["rid"]), 0) + 1
                for rid, pids in source_points(evidence_rows, "rid").items():
                    relation_scores[rid]["point_ids"] = pids
                for row in evidence_rows:
                    rid = str(row["rid"])
                    rscore = relation_scores[rid]["score"]
                    pid = str(row["point_id"])
                    slot = chunk_scores.setdefault(pid, {"point_id": pid, "chunk_uid": row.get("chunk_uid"),
                                                         "rel_path": row.get("rel_path"), "score": 0.0, "entities": [], "relations": []})
                    part = parts.setdefault(pid, {"entity": [], "evidence": [], "hub": 0,
                                                  "entity_detail": [], "evidence_detail": []})
                    weight = evidence_weight(rscore, df=relation_df.get(rid, 1), n_chunks=n_chunks)
                    if str(row.get("kind") or "body") == "conclusion":
                        weight *= CONCLUSION_BOOST
                    part["evidence"].append(weight)
                    r = relation_scores[rid]
                    if explain:
                        part["evidence_detail"].append({"label": f"{r.get('source')} -[{r.get('type')}]-> {r.get('target')}",
                                                        "score": r.get("score"), "hop": r.get("hop"), "via": r.get("via"),
                                                        "df": relation_df.get(rid, 1), "weight": weight})
                    slot["relations"].append(f"{r.get('source')} -[{r.get('type')}]-> {r.get('target')}")
            # Fact hits: their source chunks score as evidence (a fact is usually supported by a few chunks of one
            # unit, exactly where the answer is)
            for sp in spec_evidence_seeds(specs):
                pids = [str(x) for x in (sp.get("point_ids") or [])]
                for pid in pids:
                    slot = chunk_scores.setdefault(pid, {"point_id": pid, "chunk_uid": None, "rel_path": sp.get("rel_path"),
                                                         "score": 0.0, "entities": [], "relations": []})
                    part = parts.setdefault(pid, {"entity": [], "evidence": [], "hub": 0, "entity_detail": [], "evidence_detail": []})
                    weight = evidence_weight(float(sp["_score"]), df=len(pids) or 1, n_chunks=n_chunks)
                    part["evidence"].append(weight)
                    slot["relations"].append(f"spec: {sp.get('subject')} · {sp.get('property')}")
                    if explain:
                        part["evidence_detail"].append({"label": f"spec: {sp.get('text')}", "score": sp["_score"], "hop": 0,
                                                        "via": sp.get("_via"), "df": len(pids), "weight": weight})
            # Page hits: the source chunks of subject / timeline pages count as evidence too (the pages themselves
            # are returned separately)
            for pg in pages[:3]:
                if float(pg.get("_score") or 0) < PAGE_EVIDENCE_MIN_SCORE:
                    continue
                pids = [str(x) for x in (pg.get("point_ids") or [])][:8]
                for pid in pids:
                    slot = chunk_scores.setdefault(pid, {"point_id": pid, "chunk_uid": None, "rel_path": None,
                                                         "score": 0.0, "entities": [], "relations": []})
                    part = parts.setdefault(pid, {"entity": [], "evidence": [], "hub": 0, "entity_detail": [], "evidence_detail": []})
                    weight = evidence_weight(float(pg["_score"]) * 0.8, df=len(pids) or 1, n_chunks=n_chunks)
                    part["evidence"].append(weight)
                    slot["relations"].append(f"page: {pg.get('title')}")
            for pid, part in parts.items():
                chunk_scores[pid]["score"] = chunk_score(part["entity"], part["evidence"], hub=part["hub"], mean_hub=mean_hub)
                if explain:
                    chunk_scores[pid]["explain"] = {"hub": part["hub"], "mean_hub": mean_hub, "n_chunks": n_chunks,
                                                    "entities": part["entity_detail"], "relations": part["evidence_detail"]}
    finally:
        if own_driver:
            driver.close()

    candidates = sorted(chunk_scores.values(), key=lambda c: -c["score"])[:max(int(candidate_limit), int(chunk_limit))]
    for rank, c in enumerate(candidates, 1):
        c["graph_score"] = c["score"]
        c["graph_rank"] = rank
    if candidates:
        points = q.retrieve(collection_name=source.collection, ids=[c["point_id"] for c in candidates],
                            with_payload=True, with_vectors=[TEXT_VECTOR_NAME])
        by_id = {str(p.id): p for p in points}
        for c in candidates:
            p = by_id.get(c["point_id"])
            payload = dict((p.payload if p is not None else None) or {})
            if not c.get("chunk_uid"):
                c["chunk_uid"] = payload.get("chunk_uid")
                c["rel_path"] = c.get("rel_path") or payload.get("rel_path")
            vec = p.vector.get(TEXT_VECTOR_NAME) if p is not None and isinstance(p.vector, dict) else (p.vector if p is not None else None)
            text = str(payload.get("text") or "")
            c["cos"] = round(_cosine(vector, list(vec) if vec is not None else None), 4) if vector is not None else 0.0
            c["lex"] = round(lexical_overlap(question, text), 4)
            c["score"] = rerank_score(c["cos"], c["lex"], c["graph_rank"])
            c["section"] = " > ".join(str(s) for s in (payload.get("section_path") or []))
            if with_text:
                c["snippet"] = text[:400]
    chunks = sorted(candidates, key=lambda c: -c["score"])[:chunk_limit]
    for c in chunks:
        c["entities"] = sorted({e for e in c["entities"] if e})[:8]
        c["relations"] = sorted(set(c["relations"]))[:6]
    spec_rows = [mark_status({**spec_result_row(sp), "score": round(float(sp.get("_score") or 0), 4), "via": sp.get("_via"),
                              "hits": sp.get("_hits", 0), **({"series_members": series_of[str(sp["series_key"])]}
                                                             if sp.get("series_key") and series_of.get(str(sp["series_key"])) else {})},
                             scope) for sp in specs[:top_relations]]
    for r in relation_scores.values():
        mark_status(r, scope)
    pages = [pg for pg in pages if docstatus.page_allowed(settings, source, pg, scope, paths=doc_paths)]
    page_rows = [{
        "id": pg.get("gr_id"), "kind": pg.get("kind"), "title": pg.get("title"), "score": round(float(pg.get("_score") or 0), 4),
        "summary": pg.get("description"), "text": str(pg.get("text") or "")[:PAGE_TEXT_LIMIT] if with_text else None,
        "series": list(pg.get("series") or [])[:80],
        "doc_ids": list(pg.get("doc_ids") or []), "spec_ids": list(pg.get("spec_ids") or [])[:32],
        "docs": sorted({doc_paths[str(d)] for d in (pg.get("doc_ids") or []) if doc_paths.get(str(d))}),
        "entity_keys": list(pg.get("entity_keys") or [])[:16], "point_ids": spread(list(pg.get("point_ids") or []), SOURCE_SAMPLE),
        "path": pg.get("path"), "source_paths": pg.get("source_paths"),
        "compiled_day": pg.get("compiled_day"), "time_mode": pg.get("time_mode"),
    } for pg in pages]
    return {
        "question": question, "kb_id": source.kb_id, "graph_version": gv,
        "window": window, "time": {"mode": scope["mode"], "day": scope["day"]},
        "entities": sorted(entity_scores.values(), key=lambda e: -e["score"])[:top_entities * 2],
        "relations": sorted(relation_scores.values(), key=lambda r: -r["score"])[:top_relations * 2],
        "specs": spec_rows,
        "pages": page_rows,
        "chunks": chunks,
        "seeds": seed_summary,
    }


