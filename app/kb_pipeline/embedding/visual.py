from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests

from ..parsers.errors import ConfigParseError, FAILURE_SCOPE_VISUAL_EMBEDDING
from ..vision.images import DEFAULT_MAX_PIXELS, file_hash, image_data_url


# Bump when the request shape changes in a way that makes cached vectors stale.
VISUAL_EMBED_CONTRACT_VERSION = 2  # v2: EXIF transpose + 16-bit normalisation changed the pixels


class VisualEmbedContentError(RuntimeError):
    """The endpoint rejected this image deterministically (HTTP 4xx): the
    picture itself is the problem, so retrying -- or failing the whole parse
    job over it -- cannot help. Callers skip the visual vector for this one
    image and keep the document."""


class VisualEmbedConfigError(ConfigParseError):
    """Auth/endpoint/dimension configuration error from the visual embedding service."""

    config_error = True

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message, scope=FAILURE_SCOPE_VISUAL_EMBEDDING)
        self.status_code = status_code

# The model card wraps every input in this system prompt (config_sentence_
# transformers.json "default"); documents are embedded with it unchanged.
# Measured on a DGX Spark (GB10): the plain `input=` path (no chat template) lands in a
# different space -- cos(messages, input) ~ 0.47 for the same text and
# cross-modal similarity collapses -- so everything, queries included, must go
# through `messages`.
DEFAULT_INSTRUCTION = "Represent the user's input."

_SESSION_LOCK = threading.Lock()
_SESSIONS: dict[str, requests.Session] = {}


def _session_for(base_url: str) -> requests.Session:
    with _SESSION_LOCK:
        session = _SESSIONS.get(base_url)
        if session is None:
            session = requests.Session()
            session.trust_env = False  # loopback vLLM; never route through a proxy
            _SESSIONS[base_url] = session
        return session


class VisualEmbeddingClient:
    """Image -> vector via the OpenAI-compatible embeddings endpoint of a vLLM
    pooling server. Batched requests contain separate conversations, each
    with one image; concurrency limits the number of in-flight requests.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model_id: str,
        dim: int,
        instruction: str = DEFAULT_INSTRUCTION,
        concurrency: int = 2,
        batch_size: int = 1,
        retry: int = 5,
        timeout: float = 300.0,
        max_pixels: int | None = DEFAULT_MAX_PIXELS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_id = model_id
        self.dim = int(dim)
        self.instruction = instruction
        self.concurrency = max(1, int(concurrency or 1))
        self.batch_size = max(1, int(batch_size))
        self.retry = max(1, int(retry))
        self.timeout = timeout
        self.max_pixels = max_pixels
        self.session = _session_for(self.base_url)

    # -- cache identity -----------------------------------------------------
    def cache_identity(self, image_sha256: str) -> dict[str, Any]:
        return {
            "contract_version": VISUAL_EMBED_CONTRACT_VERSION,
            "model_id": self.model_id,
            "base_url": self.base_url,
            "dim": self.dim,
            "instruction_sha256": hashlib.sha256(self.instruction.encode("utf-8")).hexdigest(),
            "max_pixels": self.max_pixels,
            "image_sha256": image_sha256,
        }

    # -- cache and request contract ----------------------------------------
    def _cached(self, cache_json: Path | None, identity: dict) -> list[float] | None:
        if cache_json and cache_json.exists():
            try:
                cached = json.loads(cache_json.read_text(encoding="utf-8"))
                vector = cached.get("vector") if isinstance(cached, dict) else None
                if (isinstance(cached, dict) and cached.get("_cache") == identity
                        and self._valid_vector(vector)):
                    return [float(x) for x in vector]
            except (OSError, ValueError, TypeError):
                pass
        return None

    def _valid_vector(self, vector) -> bool:
        return (isinstance(vector, list) and len(vector) == self.dim
                and all(not isinstance(x, bool) and isinstance(x, (int, float))
                        and math.isfinite(x) for x in vector))

    @staticmethod
    def _save(cache_json: Path | None, identity: dict, vector: list[float]) -> None:
        if cache_json:
            cache_json.parent.mkdir(parents=True, exist_ok=True)
            cache_json.write_text(json.dumps({"_cache": identity, "vector": vector}), encoding="utf-8")

    def _messages(self, image_path: Path) -> list[dict]:
        return [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_data_url(image_path, self.max_pixels)}},
            ]},
        ]

    def _request(self, conversations: list[list[dict]]) -> list[list[float]]:
        # A batch is a list of independent conversations, never several images
        # in one conversation (which would produce a fused vector).
        body = {
            "model": self.model_id,
            "messages": conversations[0] if len(conversations) == 1 else conversations,
            "dimensions": self.dim,
            "encoding_format": "float",
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_error: Exception | None = None
        for attempt in range(1, self.retry + 1):
            try:
                response = self.session.post(
                    f"{self.base_url}/embeddings", json=body, headers=headers, timeout=self.timeout,
                    allow_redirects=False,
                )
                if response.status_code in (401, 403, 404, 409):
                    raise VisualEmbedConfigError(
                        f"HTTP {response.status_code}: {response.text[:300]}",
                        status_code=response.status_code,
                    )
                if response.status_code in (408, 429):
                    raise RuntimeError(f"HTTP {response.status_code}(暂时性): {response.text[:200]}")
                if response.status_code in (413, 415) or (
                        response.status_code in (400, 422) and any(term in response.text.lower() for term in
                            ("invalid image", "image too", "image size", "image dimensions", "cannot identify image", "unsupported image", "decode image"))):
                    raise VisualEmbedContentError(f"HTTP {response.status_code}: {response.text[:300]}")
                if response.status_code >= 400:
                    raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
                items = response.json().get("data") or []
                if len(items) != len(conversations):
                    raise RuntimeError(f"visual embedding response size mismatch: input={len(conversations)} output={len(items)}")
                vectors = [None] * len(items)
                for item in items:
                    # Legacy single-image endpoints may omit index; batches may not.
                    index = item.get("index", 0 if len(items) == 1 else None)
                    if (isinstance(index, bool) or not isinstance(index, int)
                            or not 0 <= index < len(items) or vectors[index] is not None):
                        raise RuntimeError("visual embedding returned duplicate or invalid indices")
                    vector = item.get("embedding")
                    if isinstance(vector, list) and len(vector) != self.dim:
                        raise VisualEmbedConfigError(
                            f"visual embedding dim mismatch: expected {self.dim}, got {len(vector)}"
                        )
                    if not self._valid_vector(vector):
                        raise RuntimeError("visual embedding returned invalid vector values")
                    vectors[index] = [float(x) for x in vector]
                return vectors
            except (VisualEmbedContentError, VisualEmbedConfigError):
                raise
            except Exception as exc:
                last_error = exc
                print(f"[visual-embed] retry images={len(conversations)} attempt={attempt}/{self.retry} error={exc!r}", flush=True)
                if attempt < self.retry:
                    time.sleep(min(60.0, 2.0 * attempt))
        assert last_error is not None
        raise last_error

    def embed_image(self, image_path: Path, *, cache_json: Path | None = None) -> list[float]:
        identity = self.cache_identity(file_hash(image_path))
        cached = self._cached(cache_json, identity)
        if cached is not None:
            return cached
        vector = self._request([self._messages(image_path)])[0]
        self._save(cache_json, identity, vector)
        return vector

    def _embed_batch(self, jobs: list[tuple[Path, Path | None]]) -> list[list[float] | None]:
        results = [None] * len(jobs)
        pending = []
        for index, (path, cache) in enumerate(jobs):
            identity = self.cache_identity(file_hash(path))
            results[index] = self._cached(cache, identity)
            if results[index] is None:
                pending.append((index, path, cache, identity))
        if not pending:
            return results
        try:
            vectors = self._request([self._messages(path) for _, path, _, _ in pending])
        except VisualEmbedContentError:
            # A rejected batch does not identify the bad image (413 may also
            # concern the combined payload). Isolate it before skipping content.
            for index, path, cache, _ in pending:
                try:
                    results[index] = self.embed_image(path, cache_json=cache)
                except VisualEmbedContentError as exc:
                    print(f"[visual-embed] rejected image={path.name}: {exc}", flush=True)
            return results
        for (index, _, cache, identity), vector in zip(pending, vectors, strict=True):
            results[index] = vector
            self._save(cache, identity, vector)
        return results

    # -- many images --------------------------------------------------------
    def embed_images(
        self,
        jobs: list[tuple[Path, Path | None]],
        *, progress_cb=None,
    ) -> list[list[float] | None]:
        """Embed images in order; `jobs` are (image_path, cache_json) pairs.

        Transport failures raise after per-image retries -- a missing visual
        vector must fail the parse job into the worker's retry schedule, not
        pass silently. A deterministic rejection (VisualEmbedContentError,
        explicit image-content rejection) yields None for that image instead: one poison picture must
        not make the whole document unindexable."""
        if not jobs:
            return []

        def one(job: tuple[Path, Path | None]) -> list[float] | None:
            path, cache = job
            try:
                return self.embed_image(path, cache_json=cache)
            except VisualEmbedContentError as exc:
                print(f"[visual-embed] rejected image={path.name}: {exc}", flush=True)
                return None

        batches = [jobs[i:i + self.batch_size] for i in range(0, len(jobs), self.batch_size)]
        workers = min(self.concurrency, len(batches))

        def one_batch(batch):
            return [one(batch[0])] if len(batch) == 1 else self._embed_batch(batch)
        print(f"[visual-embed] start images={len(jobs)} batch_size={self.batch_size} concurrency={workers} model={self.model_id} dim={self.dim}", flush=True)
        started = time.time()
        if progress_cb is not None:
            progress_cb(0, len(jobs))

        def collect(results):
            vectors = []
            for batch_vectors in results:
                for vector in batch_vectors:
                    vectors.append(vector)
                    if progress_cb is not None:
                        progress_cb(len(vectors), len(jobs))
            return vectors

        if workers == 1:
            vectors = collect(map(one_batch, batches))
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                try:
                    vectors = collect(executor.map(one_batch, batches))
                except Exception:
                    # One image failed in transport, so the whole file has to be redone: the queued images
                    # need not each be tried again
                    executor.shutdown(wait=True, cancel_futures=True)
                    raise
        rejected = sum(1 for vector in vectors if vector is None)
        print(
            f"[visual-embed] done images={len(jobs)} rejected={rejected} elapsed={time.time() - started:.1f}s",
            flush=True,
        )
        return vectors
