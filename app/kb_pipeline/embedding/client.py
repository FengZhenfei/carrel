from __future__ import annotations

import threading
import time

from openai import OpenAI


_CLIENT_LOCK = threading.Lock()
_CLIENTS: dict[tuple[str, str], OpenAI] = {}


def _client_for(base_url: str, api_key: str) -> OpenAI:
    """One pooled client per endpoint. A parse job used to build a fresh
    client each time, paying a new connection per document; against the
    local vLLM that is pure overhead across a full re-ingest."""
    key = (base_url, api_key)
    with _CLIENT_LOCK:
        client = _CLIENTS.get(key)
        if client is None:
            client = OpenAI(api_key=api_key, base_url=base_url, timeout=120, max_retries=0)
            _CLIENTS[key] = client
    return client


class EmbeddingDimensionError(RuntimeError):
    """The vector dimension returned by the service does not match the configuration. Deterministic error, not
    retried."""


class EmbeddingInputRejected(RuntimeError):
    """The service rejected one of the inputs because of its content (most likely it exceeds the length limit
    of the model). Retrying the same input changes nothing, so it is not retried. index is the position of the
    rejected input in embed(texts)."""

    deterministic = True

    def __init__(self, index: int, reason: str) -> None:
        super().__init__(f"embedding input #{index + 1} rejected: {reason}")
        self.index = index
        self.reason = reason


def _status_of(exc: Exception) -> int | None:
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _is_deterministic_client_error(exc: Exception) -> bool:
    status = _status_of(exc)
    # 408 timeout / 429 rate limit are transient; retrying any other 4xx is pointless
    return status is not None and 400 <= status < 500 and status not in (408, 409, 429)


class EmbeddingClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model_id: str,
        dim: int,
        batch_size: int,
        retry: int = 5,
        sleep_seconds: float = 0.1,
    ) -> None:
        self.client = _client_for(base_url, api_key)
        self.model_id = model_id
        self.dim = dim
        self.batch_size = max(1, batch_size)
        self.retry = max(1, retry)
        self.sleep_seconds = max(0.0, sleep_seconds)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            end_idx = i + len(batch)
            last_error: Exception | None = None
            for attempt in range(1, self.retry + 1):
                try:
                    resp = self.client.embeddings.create(model=self.model_id, input=batch, dimensions=self.dim)
                    batch_vectors = [item.embedding for item in resp.data]
                    if len(batch_vectors) != len(batch):
                        raise RuntimeError(
                            f"embedding response size mismatch: input={len(batch)} output={len(batch_vectors)}"
                        )
                    # Dimension check: when the service ignores dimensions (a non-Matryoshka model), this
                    # used to fail only at the Qdrant upsert with a 400, and that 400 was treated as retryable.
                    bad = next((len(v) for v in batch_vectors if len(v) != self.dim), None)
                    if bad is not None:
                        raise EmbeddingDimensionError(
                            f"embedding dim mismatch: expected {self.dim}, got {bad}"
                        )
                    vectors.extend(batch_vectors)
                    last_error = None
                    break
                except EmbeddingDimensionError:
                    raise                      # deterministic error: retrying changes nothing
                except Exception as exc:
                    if _is_deterministic_client_error(exc):
                        # A 400 (input too long / invalid) retried 5 times just wastes 20 seconds, and then a
                        # job-level retry round follows anyway.
                        rejected = self._rejected_input(batch, exc)
                        if rejected is not None:
                            raise EmbeddingInputRejected(i + rejected, str(exc)) from exc
                        raise
                    last_error = exc
                    print(
                        f"[embed] retry batch={i + 1}-{end_idx}/{len(texts)} "
                        f"attempt={attempt}/{self.retry} error={exc!r}",
                        flush=True,
                    )
                    if attempt < self.retry:
                        time.sleep(min(60.0, 2.0 * attempt))
            if last_error is not None:
                raise last_error
            if self.sleep_seconds > 0:
                time.sleep(self.sleep_seconds)
        return vectors

    def _rejected_input(self, batch: list[str], exc: Exception) -> int | None:
        """When a batch is rejected with 400 / 422, find the input responsible: send the inputs one at a time and
        pick the one that is rejected on its own while a short prefix of it is accepted. When every input
        passes on its own (the batch as a whole was rejected), or even the short prefix is rejected (the
        service rejects any input, a configuration problem), return None: no single input is at fault, and
        the original error goes on to the job-level retry."""
        if _status_of(exc) not in (400, 422):
            return None
        for idx, text in enumerate(batch):
            if not self._accepts(text):
                return idx if self._accepts(text[:32]) else None
        return None

    def _accepts(self, text: str) -> bool:
        try:
            self.client.embeddings.create(model=self.model_id, input=[text], dimensions=self.dim)
        except Exception as exc:
            if _status_of(exc) in (400, 422):
                return False
            raise
        return True
