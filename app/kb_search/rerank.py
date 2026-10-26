"""Cross-encoder rerank on 8102 (Q06): only cleaned body text is fed in, long chunks are windowed and
scored per window taking the max; when the service is unavailable the caller falls back to the fusion
order."""
from __future__ import annotations

from typing import Any
import hashlib
import math
import time

import requests

from .text import clean_for_rerank, head_line, windows

# Shared by the whole process: connections are reused and the model name is resolved once, instead of opening
# a new connection and asking /models first on every rerank
_session = requests.Session()
_model_ids: dict[str, str] = {}       # /models address -> resolved model name


class Reranker:
    def __init__(self, base_url: str, *, model_id: str = "", api_key: str = "", timeout: float = 20.0) -> None:
        base = str(base_url or "").rstrip("/")
        # Accept an API base or an explicit rerank endpoint, including /v2/rerank.
        if base.endswith("/rerank"):
            self.url = base
            self.models_url = base.rsplit("/", 1)[0] + "/models"
        else:
            self.url = f"{base}/rerank" if base.endswith("/v1") else f"{base}/v1/rerank"
            self.models_url = f"{base}/models" if base.endswith("/v1") else f"{base}/v1/models"
        self.model_id = model_id
        self.timeout = timeout
        self.request_options = {"headers": {"Authorization": f"Bearer {api_key}"}} if api_key else {}
        # Model discovery may depend on the account; never share it across keys.
        self.cache_key = self.models_url + (":" + hashlib.sha256(api_key.encode()).hexdigest() if api_key else "")

    def resolve_model(self) -> str:
        if self.model_id:
            return self.model_id
        cached = _model_ids.get(self.cache_key)
        if cached:
            return cached
        resp = _session.get(self.models_url, timeout=min(5.0, self.timeout), **self.request_options)
        resp.raise_for_status()
        data = resp.json().get("data") or []
        if not data:
            raise RuntimeError("reranker lists no model")
        model_id = _model_ids[self.cache_key] = str(data[0].get("id"))
        return model_id

    def score(self, query: Any, documents: list[Any], *, strict: bool = True) -> list[float]:
        if not documents:
            return []
        body = {"model": self.resolve_model(), "query": query, "documents": documents}
        resp = _session.post(self.url, json=body, timeout=self.timeout, **self.request_options)
        if not resp.ok:
            _model_ids.pop(self.cache_key, None)       # Resolve again after the service changes models.
        resp.raise_for_status()
        results = resp.json().get("results") or []
        out = [0.0] * len(documents)
        seen: set[int] = set()
        for item in results:
            idx = int(item.get("index", -1))
            if strict and (idx in seen or not 0 <= idx < len(documents)):
                raise ValueError("reranker returned a duplicate or invalid document index")
            if 0 <= idx < len(documents):
                value = float(item.get("relevance_score") or 0.0)
                if strict and (item.get("relevance_score") is None or not math.isfinite(value) or not 0 <= value <= 1):
                    raise ValueError("reranker returned an invalid relevance score")
                out[idx] = value
                seen.add(idx)
        if strict and len(seen) != len(documents):
            raise ValueError("reranker omitted document scores")
        return out


def rerank_images(settings: Any, question: str, candidates: list[dict[str, Any]], *,
                  image_bytes: bytes | None, until: float) -> dict[str, Any]:
    """Score admitted image candidates using cached pixels and bounded source text.

    Only cache-relative, validated paths become data URLs; the model never fetches
    a URL supplied by a document. The original-image endpoint remains the path for
    detailed inspection. Return scores instead of mutating caller-owned candidates,
    so a late worker cannot change a response after its deadline.
    """
    from .channels import _visual_data_url
    from .images import _cached_file
    from .text import truncate_tokens

    scores: dict[str, float] = {}
    errors: list[str] = []
    query: Any = question
    if image_bytes:
        query = {"content": [{"type": "text", "text": question},
                             {"type": "image_url", "image_url": {"url": _visual_data_url(image_bytes)}}]}
    docs: list[dict[str, Any]] = []
    ids: list[str] = []
    for cand in candidates:
        if time.time() >= until:
            errors.append("budget")
            break
        payload = cand.get("payload") or {}
        try:
            path = _cached_file(settings, payload)
            if path.stat().st_size > 20_000_000:
                raise ValueError("image exceeds input byte budget")
            url = _visual_data_url(path.read_bytes())
            text = truncate_tokens(clean_for_rerank(str(payload.get("text") or "")), 384)
            docs.append({"content": [{"type": "text", "text": head_line(payload) + "\n" + text},
                                     {"type": "image_url", "image_url": {"url": url}}]})
            ids.append(str(cand["point_id"]))
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"image:{type(exc).__name__}")
    # Small batches bound model memory and preserve completed batches if the next
    # one fails. The deadline also includes image loading and HTTP model discovery.
    for start in range(0, len(docs), 4):
        remaining = until - time.time()
        if remaining <= 0:
            errors.append("budget")
            break
        try:
            client = Reranker(settings.visual_reranker_base_url, timeout=remaining)
            client.resolve_model()
            client.timeout = max(0.001, until - time.time())
            if time.time() >= until:
                raise TimeoutError("image rerank budget exhausted")
            values = client.score(query, docs[start:start + 4], strict=True)
            scores.update(zip(ids[start:start + 4], values))
        except Exception as exc:
            errors.append(f"model:{type(exc).__name__}")
            break
    return {"scores": scores, "errors": sorted(set(errors)), "requested": len(candidates)}


def rerank_documents(candidates: list[dict[str, Any]], *, window_tokens: int, overlap_tokens: int) -> tuple[list[str], list[int]]:
    """Candidates -> rerank input: a head line "filename › section path" + cleaned body, windowed when
    long; returns (list of documents, which candidate each document belongs to)."""
    docs: list[str] = []
    owner: list[int] = []
    for i, cand in enumerate(candidates):
        payload = cand.get("payload") or {}
        head = head_line(payload)
        body = clean_for_rerank(str(payload.get("text") or ""))
        for win in windows(body, window_tokens=window_tokens, overlap_tokens=overlap_tokens):
            docs.append(f"{head}\n{win}" if head else win)
            owner.append(i)
    return docs, owner


def rerank(reranker: Reranker, question: str, candidates: list[dict[str, Any]], *, window_tokens: int = 480,
           overlap_tokens: int = 32) -> list[float]:
    """Each candidate's rerank score = the maximum over its window scores (Q06: long chunks are not
    truncated)."""
    docs, owner = rerank_documents(candidates, window_tokens=window_tokens, overlap_tokens=overlap_tokens)
    scores = reranker.score(question, docs)
    best = [0.0] * len(candidates)
    seen = [False] * len(candidates)
    for s, i in zip(scores, owner):
        if not seen[i] or s > best[i]:
            best[i] = s
            seen[i] = True
    return best
