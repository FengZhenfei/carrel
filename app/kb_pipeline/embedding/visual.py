from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests

from ..vision.images import DEFAULT_MAX_PIXELS, file_hash, image_data_url


# Bump when the request shape changes in a way that makes cached vectors stale.
VISUAL_EMBED_CONTRACT_VERSION = 2  # v2: EXIF transpose + 16-bit normalisation changed the pixels


class VisualEmbedContentError(RuntimeError):
    """The endpoint rejected this image deterministically (HTTP 4xx): the
    picture itself is the problem, so retrying -- or failing the whole parse
    job over it -- cannot help. Callers skip the visual vector for this one
    image and keep the document."""

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
    pooling server running Qwen3-VL-Embedding. One request per image (the
    server runs --limit-mm-per-prompt image=1); parallelism comes from threads.
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
        self.retry = max(1, int(retry))
        self.timeout = timeout
        self.max_pixels = max_pixels
        self.session = _session_for(self.base_url)

    # -- cache identity -----------------------------------------------------
    def cache_identity(self, image_sha256: str) -> dict[str, Any]:
        return {
            "contract_version": VISUAL_EMBED_CONTRACT_VERSION,
            "model_id": self.model_id,
            "dim": self.dim,
            "instruction_sha256": hashlib.sha256(self.instruction.encode("utf-8")).hexdigest(),
            "max_pixels": self.max_pixels,
            "image_sha256": image_sha256,
        }

    # -- single image -------------------------------------------------------
    def embed_image(self, image_path: Path, *, cache_json: Path | None = None) -> list[float]:
        image_sha256 = file_hash(image_path)
        identity = self.cache_identity(image_sha256)
        if cache_json and cache_json.exists():
            try:
                cached = json.loads(cache_json.read_text(encoding="utf-8"))
            except Exception:
                cached = None
            if isinstance(cached, dict) and cached.get("_cache") == identity:
                vector = cached.get("vector")
                if isinstance(vector, list) and len(vector) == self.dim:
                    return [float(x) for x in vector]

        body = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": self.instruction},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_data_url(image_path, self.max_pixels)}},
                    ],
                },
            ],
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
                    allow_redirects=False,      # a request carrying the key does not follow redirects
                )
                if response.status_code in (408, 429):
                    # A full queue / rate limit is transient: this used to fall into "the picture itself is
                    # the problem", so the image was skipped permanently (no cache written, the job succeeded
                    # as usual, never filled in later).
                    raise RuntimeError(f"HTTP {response.status_code} (transient): {response.text[:200]}")
                if 400 <= response.status_code < 500:
                    raise VisualEmbedContentError(
                        f"HTTP {response.status_code} for {image_path.name}: {response.text[:300]}"
                    )
                if response.status_code >= 500:
                    raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
                data = response.json()
                items = data.get("data") or []
                if len(items) != 1:
                    raise RuntimeError(f"visual embedding response size mismatch: output={len(items)}")
                vector = [float(x) for x in items[0]["embedding"]]
                if len(vector) != self.dim:
                    raise RuntimeError(f"visual embedding dim mismatch: got={len(vector)} expected={self.dim}")
                if cache_json:
                    cache_json.parent.mkdir(parents=True, exist_ok=True)
                    cache_json.write_text(json.dumps({"_cache": identity, "vector": vector}), encoding="utf-8")
                return vector
            except VisualEmbedContentError:
                raise
            except Exception as exc:
                last_error = exc
                print(
                    f"[visual-embed] retry image={image_path.name} attempt={attempt}/{self.retry} error={exc!r}",
                    flush=True,
                )
                if attempt < self.retry:
                    time.sleep(min(60.0, 2.0 * attempt))
        assert last_error is not None
        raise last_error

    # -- many images --------------------------------------------------------
    def embed_images(
        self,
        jobs: list[tuple[Path, Path | None]],
    ) -> list[list[float] | None]:
        """Embed images in order; `jobs` are (image_path, cache_json) pairs.

        Transport failures raise after per-image retries -- a missing visual
        vector must fail the parse job into the worker's retry schedule, not
        pass silently. A deterministic rejection (VisualEmbedContentError,
        HTTP 4xx) yields None for that image instead: one poison picture must
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

        workers = min(self.concurrency, len(jobs))
        print(f"[visual-embed] start images={len(jobs)} concurrency={workers} model={self.model_id} dim={self.dim}", flush=True)
        started = time.time()
        if workers == 1:
            vectors = [one(job) for job in jobs]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                try:
                    vectors = list(executor.map(one, jobs))
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
