"""Cross-encoder rerank on 8102 (Q06): only cleaned body text is fed in, long chunks are windowed and
scored per window taking the max; when the service is unavailable the caller falls back to the fusion
order."""
from __future__ import annotations

from typing import Any

import requests

from .text import clean_for_rerank, head_line, windows


class Reranker:
    def __init__(self, base_url: str, *, model_id: str = "", timeout: float = 20.0) -> None:
        base = str(base_url or "").rstrip("/")
        self.url = f"{base}/rerank" if base.endswith("/v1") else f"{base}/v1/rerank"
        self.models_url = f"{base}/models" if base.endswith("/v1") else f"{base}/v1/models"
        self.model_id = model_id
        self.timeout = timeout

    def resolve_model(self) -> str:
        if self.model_id:
            return self.model_id
        resp = requests.get(self.models_url, timeout=min(5.0, self.timeout))
        resp.raise_for_status()
        data = resp.json().get("data") or []
        if not data:
            raise RuntimeError("reranker lists no model")
        self.model_id = str(data[0].get("id"))
        return self.model_id

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        body = {"model": self.resolve_model(), "query": query, "documents": documents}
        resp = requests.post(self.url, json=body, timeout=self.timeout)
        resp.raise_for_status()
        results = resp.json().get("results") or []
        out = [0.0] * len(documents)
        for item in results:
            idx = int(item.get("index", -1))
            if 0 <= idx < len(documents):
                out[idx] = float(item.get("relevance_score") or 0.0)
        return out


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
