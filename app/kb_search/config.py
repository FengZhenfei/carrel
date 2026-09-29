"""The retrieval service's own tunable parameters: all read from environment variables (the env file is
loaded into os.environ by kb_pipeline.config.load_settings; the long-running service rereads KB_SEARCH_*
from the file's current content every minute, see service.runtime), with the origin of every value noted
next to it; starting values follow the 2026-09-09 to-do list Q05 / Q15, and after tuning the measurement
date and regression numbers go into the comments."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


def env_file_values(path: str | Path) -> dict[str, str] | None:
    """The env file's current content (parsed by the same rules as kb_pipeline.utils.load_env_file: when a
    key is written twice the first one wins); None when the file cannot be read."""
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return None
    out: dict[str, str] = {}
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out.setdefault(key.strip(), value.strip().strip("'\""))
    return out


def _int(env: Mapping[str, str], name: str, default: int) -> int:
    try:
        return int(env.get(name, default))
    except (TypeError, ValueError):
        return default


def _float(env: Mapping[str, str], name: str, default: float) -> float:
    try:
        return float(env.get(name, default))
    except (TypeError, ValueError):
        return default


def _bool(env: Mapping[str, str], name: str, default: bool = True) -> bool:
    raw = env.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in ("0", "false", "no", "off")


@dataclass(frozen=True)
class SearchSettings:
    host: str
    port: int
    token: str                    # static Bearer; empty string = only loopback requests are accepted
    vector_k: int                 # vector channel candidate count (Q05 start 50)
    bm25_k: int                   # keyword channel candidate count (Q05 start 30)
    graph_k: int                  # graph channel candidate count (Q04 start 60)
    visual_k: int                 # visual channel candidate count (Q17: text-to-image takes only the top 20, image chunks are few anyway)
    rerank_n: int                 # number of candidates entering the rerank (Q05 start 64)
    top_k: int                    # default number of results returned (Q15 start 12)
    rrf_k: int                    # RRF constant (Q05: 60)
    rerank_enabled: bool
    # Rerank score floor (Q06). Calibrated on 2026-09-15 with the local qwen3-reranker-0.6b: score
    # histogram (0.0-0.1 ... 0.9-1.0) of 64 candidates for each of 15 real questions =
    # [124, 6, 10, 14, 14, 21, 27, 33, 56, 628], 10 cross-document questions on the health knowledge base =
    # [418, 36, 32, 20, 8, 16, 13, 12, 6, 11] (top scores 0.32-0.98), 8 out-of-corpus questions =
    # [494, 0, 0, 0, 0, 0, 0, 1, 3, 0] (top score <= 0.043, the sole exception being one question that
    # happened to have matching content at 0.85). The valley is at 0.1-0.2: 0.1 is used, floor x0.7 =
    # 0.07; no real question loses a result, and every out-of-corpus question lands in below_threshold.
    rerank_threshold: float
    rerank_window_tokens: int     # windowed scoring of long chunks (Q06: about 480 tokens, 32 overlap)
    rerank_overlap_tokens: int
    rerank_timeout: float
    context_tokens: int           # token budget of answer_context (Q07)
    neighbor_span: int            # neighbourhood backfill: how many chunks before and after each hit (Q07)
    graph_hops: int               # graph channel expansion hops. Q27 comparison on 2026-09-16 (seven question sets, auto routing): 0 / 1 / 2 hops
                                  # gave identical hit@k, expect, cross-document and negative results and differed only in latency (products
                                  # 1071 / 1219 / 1279 ms, library 1535 / 1604 / 1755 ms); 1 is used: the entity / relation tables keep one-hop
                                  # neighbours, and it saves a tenth of the time compared with 2 hops
    route_max_kbs: int            # maximum number of knowledge bases auto routing selects (Q20)
    route_gap: float              # knowledge bases whose evidence score (0-1 relative) is within this gap of the top score are queried together (Q20)
    route_floor: float            # weak is flagged when every knowledge base's vector evidence (mean cosine of the top 3 chunks) is below it; a hint only, selection unchanged (Q20)
    route_lexical_weight: float   # weight of the keyword channel in knowledge base evidence, the rest goes to the vector channel (Q20; BM25 is not comparable across indexes, so the weight should stay under half)
    route_widen: bool             # when the knowledge bases chosen by auto routing yield no decent evidence, widen to all knowledge bases and rank again (Q20)
    catalog_ttl: float            # catalog cache lifetime in seconds. 60: matches the settings refresh period, so a graph built for the first time is used by the graph channel within a minute (a rebuild takes about 0.4 s)
    channel_timeout: float        # per-channel recall timeout (seconds); single Qdrant / OpenSearch / Neo4j calls in the search process are capped by it as well
    request_budget: float         # time budget of the whole request (seconds): each stage gets the smaller of its own timeout and the remaining budget; when the
                                  # budget runs short there is no widening and no neighbourhood backfill, and budget is recorded in degraded. 45: the caller
                                  # (the agent's skill) times out at 60 seconds by default without retrying, which leaves a margin;
                                  # 0 = no overall budget. To be reviewed against the latency distribution under real concurrency once in service
    visual_enabled: bool          # visual channel (Q17): the question is embedded through 8103's messages form and queried against the visual vector; silently skipped when 8103 is unavailable
    visual_timeout: float
    visual_low_confidence_factor: float   # image chunks with visual_confidence = low: the final ranking score is multiplied by it, down-weighted but not filtered (Q17)
    visual_quota: int             # requests carrying an image: when cutting to top-k, reserve this many slots for visual channel candidates by visual score (Codex S02: the text rerank must not squeeze out image blocks)
    visual_route_weight: float    # requests carrying an image: weight of visual evidence in knowledge base selection, the two text channels share the rest (Codex S02)
    embed_timeout: float          # request timeout for embedding the question (seconds), not the build client's 120 s (Codex S06)
    query_instruction: str        # query-side task instruction (the Qwen3-Embedding model card form "Instruct: ...\nQuery: ..."); empty = embed the question verbatim.
                                  # The "document name > section path" content prefix carried by corpus vectors is a different thing. Ablation on 2026-09-16
                                  # (seven question sets, auto routing): with vs without the instruction, chunk hit@5 products 5/15=5/15, reports 3/4=3/4,
                                  # library 10/15 vs 11/15, projects 8/15=8/15, expect 14/13, 4/4, 14/15, 14/12, negatives 8/8=8/8, health / report
                                  # cross-document all pass; one or two questions swap either way, latency identical, no consistent gain, so the
                                  # default stays empty.
    hint_block_boost: float       # chunks matching hints.block_types have their final score multiplied by it (soft preference, no filtering)
    final_lex_weight: float       # weight of question token coverage in the final score (RAGFlow-style tkweight): in table-heavy knowledge bases the rerank
                                  # score saturates at 0.99 across whole runs, and "how many of the question's concrete words land in this chunk" tells the
                                  # rows of one table apart. Grid on 2026-09-16 (chunk hit@5 / MRR, auto routing):
                                  # weight 0: products 6/15 0.32, reports 3/4 0.29, library 11/15 0.69, projects 10/15 0.40;
                                  # 0.1: 10/15 0.63, 4/4 1.0, 15/15 0.95, 12/15 0.80; 0.2: 11/15 0.70, 4/4 1.0, 14/15 0.94, 12/15 0.80;
                                  # 0.3: 12/15 0.72, 4/4 1.0, 14/15 0.94, 12/15 0.80; negatives and the health / report cross-document sets identical at
                                  # every level. 0.2 is used: most of the gain while the rerank still carries eighty percent (synthetic questions are
                                  # naturally close to the source wording, and a higher weight risks a bias towards that kind of question)
    pool_workers: int             # cap of the thread pool shared by the whole service (Codex S06: thread count does not grow with requests)
    stitch_min_chars: int         # a hit shorter than this is stitched with its neighbours in both directions along the document's chunk index (Q16: 350)
    stitch_max_chars: int         # upper limit of the stitched length (Q16: 850)
    mmr_enabled: bool             # run MMR after the rerank and before cutting to top-k (Q16)
    mmr_lambda: float             # MMR relevance weight (Q16: 0.7, similarity by token Jaccard)
    quota_min_hits: int           # multi-subject / multi-document quota: at least this many per bucket in the final hits (Q08)
    quota_max_buckets: int        # no quota once buckets exceed this number ("per document" is meaningless with too many documents, Q08)
    boilerplate_factor: float     # boilerplate chunks (table of contents / revision history / copyright page) have their final ranking score multiplied by it (Q18: down-weighted, not removed)
    page_text_tokens: int         # total budget for compiled page text in the context (Q12: only timeline pages and subject pages with series rows)
    spec_hint_limit: int          # maximum number of fact hints (Q11)


def load_search_settings(env: Mapping[str, str] | None = None) -> SearchSettings:
    """env: where the values come from, the process environment by default; the long-running service passes
    the set computed from the env file's current content (service._search_env)."""
    env = os.environ if env is None else env
    return SearchSettings(
        host=str(env.get("KB_SEARCH_HOST", "0.0.0.0")),
        port=_int(env, "KB_SEARCH_PORT", 9810),
        token=str(env.get("KB_SEARCH_TOKEN", "") or "").strip(),
        vector_k=_int(env, "KB_SEARCH_VECTOR_K", 50),
        bm25_k=_int(env, "KB_SEARCH_BM25_K", 30),
        graph_k=_int(env, "KB_SEARCH_GRAPH_K", 60),
        visual_k=_int(env, "KB_SEARCH_VISUAL_K", 20),
        rerank_n=_int(env, "KB_SEARCH_RERANK_N", 64),
        top_k=_int(env, "KB_SEARCH_TOP_K", 12),
        rrf_k=_int(env, "KB_SEARCH_RRF_K", 60),
        rerank_enabled=_bool(env, "KB_SEARCH_RERANK", True),
        rerank_threshold=_float(env, "KB_SEARCH_RERANK_THRESHOLD", 0.1),
        rerank_window_tokens=_int(env, "KB_SEARCH_RERANK_WINDOW_TOKENS", 480),
        rerank_overlap_tokens=_int(env, "KB_SEARCH_RERANK_OVERLAP_TOKENS", 32),
        rerank_timeout=_float(env, "KB_SEARCH_RERANK_TIMEOUT", 20.0),
        context_tokens=_int(env, "KB_SEARCH_CONTEXT_TOKENS", 6000),
        neighbor_span=_int(env, "KB_SEARCH_NEIGHBOR_SPAN", 1),
        graph_hops=_int(env, "KB_SEARCH_GRAPH_HOPS", 1),
        route_max_kbs=_int(env, "KB_SEARCH_ROUTE_MAX_KBS", 2),
        route_gap=_float(env, "KB_SEARCH_ROUTE_GAP", 0.20),
        route_floor=_float(env, "KB_SEARCH_ROUTE_FLOOR", 0.45),
        route_lexical_weight=_float(env, "KB_SEARCH_ROUTE_LEXICAL_WEIGHT", 0.3),
        route_widen=_bool(env, "KB_SEARCH_ROUTE_WIDEN", True),
        catalog_ttl=_float(env, "KB_SEARCH_CATALOG_TTL", 60.0),
        channel_timeout=_float(env, "KB_SEARCH_CHANNEL_TIMEOUT", 25.0),
        request_budget=_float(env, "KB_SEARCH_REQUEST_BUDGET", 45.0),
        visual_enabled=_bool(env, "KB_SEARCH_VISUAL", True),
        visual_timeout=_float(env, "KB_SEARCH_VISUAL_TIMEOUT", 20.0),
        visual_low_confidence_factor=_float(env, "KB_SEARCH_VISUAL_LOW_CONFIDENCE_FACTOR", 0.85),
        visual_quota=_int(env, "KB_SEARCH_VISUAL_QUOTA", 3),
        visual_route_weight=_float(env, "KB_SEARCH_VISUAL_ROUTE_WEIGHT", 0.6),
        embed_timeout=_float(env, "KB_SEARCH_EMBED_TIMEOUT", 10.0),
        query_instruction=str(env.get("KB_SEARCH_QUERY_INSTRUCTION", "") or "").strip(),
        hint_block_boost=_float(env, "KB_SEARCH_HINT_BLOCK_BOOST", 1.2),
        final_lex_weight=_float(env, "KB_SEARCH_FINAL_LEX_WEIGHT", 0.2),
        pool_workers=_int(env, "KB_SEARCH_POOL_WORKERS", 32),
        stitch_min_chars=_int(env, "KB_SEARCH_STITCH_MIN_CHARS", 350),
        stitch_max_chars=_int(env, "KB_SEARCH_STITCH_MAX_CHARS", 850),
        mmr_enabled=_bool(env, "KB_SEARCH_MMR", True),
        mmr_lambda=_float(env, "KB_SEARCH_MMR_LAMBDA", 0.7),
        quota_min_hits=_int(env, "KB_SEARCH_QUOTA_MIN_HITS", 2),
        quota_max_buckets=_int(env, "KB_SEARCH_QUOTA_MAX_BUCKETS", 8),
        boilerplate_factor=_float(env, "KB_SEARCH_BOILERPLATE_FACTOR", 0.5),
        page_text_tokens=_int(env, "KB_SEARCH_PAGE_TEXT_TOKENS", 1200),
        spec_hint_limit=_int(env, "KB_SEARCH_SPEC_HINT_LIMIT", 8),
    )
