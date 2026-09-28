"""The graph side's shared LLM caller: cache, retries, concurrency, circuit breaker.

Entity extraction, description summaries, entity resolution, schema sampling and structured facts all go
through this one. Trade-offs:

· Cache key = sha256 of model name + protocol + sampling parameters + full message text (upstream GraphRAG
  leaves the model name out, so switching models without switching directories silently returned the old
  model's responses). The cache is one SQLite file per KB and stores successful responses only.
· Retries: RAGFlow's defaults, exponential backoff 2s → 60s, at most 5 attempts; only network / timeout /
  429 / 5xx / empty responses are retried, other 4xx are deterministic errors where retrying is pointless.
· Circuit breaker: N consecutive failed calls (after retries are exhausted) stop the build. When the provider is
  down as a whole every retry burns money, while "come back later" is almost free: the calls already finished
  are in the cache. The criterion comes straight from call results, no more guessing from upstream logs.
· Concurrency: a thread pool, 8 workers by default (the ceiling measured on 2026-08-25 against the two models
  in use).
· enable_thinking=false adaptively: reasoning models burn the output budget on thinking; a server that does not
  recognize the parameter answers 4xx, in which case it is dropped, the request re-sent and that base_url
  remembered.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import requests
from requests.adapters import HTTPAdapter

DEFAULT_TIMEOUT = 300
DEFAULT_ATTEMPTS = 5
BACKOFF_BASE = 2.0
BACKOFF_MAX = 60.0
DEFAULT_WORKERS = 8
DEFAULT_CIRCUIT_FAILS = 20
DEFAULT_MAX_TOKENS = 4096
CACHE_VERSION = "v1"


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


class LLMCallError(RuntimeError):
    """A call still failed after its retries were exhausted. The caller decides whether to isolate it (unit-level
    failure) or abort."""


class LLMCircuitOpen(RuntimeError):
    """Consecutive failures passed the threshold: the provider is down as a whole, stop the build and keep the
    cache."""


class LLMInterrupted(RuntimeError):
    """The stop event was set (the build received SIGTERM / a pause), worker threads exit on their own."""


class LLMEmptyResponse(RuntimeError):
    """The model returned an empty body. Mostly deterministic (content filtering, a prompt that stumped the model),
    worth exactly one more try."""


class LLMInputRejected(LLMCallError):
    """The server explicitly rejected this input (content moderation, context length exceeded): only a different
    input would change that, retrying is pointless, and it is no sign of the service being down. A unit-level
    failure that counts neither towards the circuit breaker nor towards the "extraction is broken" failure
    ratio."""


class LLMTransientResponse(RuntimeError):
    """HTTP succeeded, but the server says this generation was interrupted (DeepSeek's official
    finish_reason=insufficient_system_resource / aborted): the body may be empty or cut short, neither is usable,
    so retry with backoff like any ordinary retryable error."""


class LLMReasoningExhausted(LLMEmptyResponse):
    """A reasoning model spent the whole output budget on thinking (finish_reason=length, empty body). Retrying the
    same input would only burn it again, so no retry; this is a systemic problem (the endpoint ignores
    enable_thinking or the model cannot turn thinking off), counted towards the circuit breaker and spelled out
    in the circuit reason."""


class LLMMalformedResponse(RuntimeError):
    """HTTP succeeded, but the content is not in the format the caller asked for (e.g. fact extraction wanted JSON
    and got prose). Such responses used to go into the cache and count as success, and a resume reused them as-is
    (Codex review F03)."""


_MALFORMED_ATTEMPTS = 3          # bad format: at most two more asks, previous reply + correction hint sent back (sporadic on the real box, retry usually works)
_MALFORMED_NUDGE = "Your previous reply was not in the required format. Reply again with ONLY the requested JSON object, nothing else."
_EMPTY_RESPONSE_ATTEMPTS = 2


@dataclass(frozen=True)
class LLMSpec:
    name: str
    base_url: str
    api_key: str
    model_id: str
    protocol: str = "openai"

    @classmethod
    def from_row(cls, row: Any) -> "LLMSpec":
        keys = row.keys() if hasattr(row, "keys") else ()
        protocol = str(row["protocol"] if "protocol" in keys else "openai").strip().lower() or "openai"
        return cls(
            name=str(row["name"] if "name" in keys else row["model_id"]),
            base_url=str(row["base_url"]).rstrip("/"),
            api_key=str(row["api_key"] or ""),
            model_id=str(row["model_id"]),
            protocol=protocol,
        )


# ── Cache ────────────────────────────────────────────────────────────────

def _token_count(text: str) -> int:
    try:
        from ..utils import count_tokens
        return int(count_tokens(text))
    except Exception:
        return max(1, len(text) // 3)


class LLMCache:
    """Response cache addressed by call content. One SQLite file per KB, one connection shared by all threads under
    a lock. With path=None it is a no-op (tests and dry-run).

    Managed per version (user decision 2026-09-09): during a build LLMCache.build_tag is set to the build id, and
    entries hit or written are tagged with it; after a full build prune_unused(build_tag) deletes the entries this
    version did not use, so the cache only ever holds what the current version needs. Incremental appends do not
    prune (the same-entity answers for old pairs are not touched, and deleting them would make the next full
    rebuild ask everything again). Tags accumulate in memory and are committed every 64 entries and on close; a
    process dying midway loses a few dozen tags at most, which only means a few extra questions on the next
    rebuild."""

    build_tag: str | None = None          # process-wide: set when build_graph starts, cleared when it ends

    def __init__(self, path: Path | str | None) -> None:
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._con: sqlite3.Connection | None = None
        self._pending = 0
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._con = sqlite3.connect(self.path, check_same_thread=False)
            self._con.execute("PRAGMA journal_mode=WAL")
            self._con.execute(
                "CREATE TABLE IF NOT EXISTS responses ("
                "key TEXT PRIMARY KEY, model TEXT NOT NULL, response TEXT NOT NULL, created_at INTEGER NOT NULL)"
            )
            cols = {str(r[1]) for r in self._con.execute("PRAGMA table_info(responses)").fetchall()}
            if "used_by" not in cols:
                self._con.execute("ALTER TABLE responses ADD COLUMN used_by TEXT")
            self._con.commit()

    def _mark(self, key: str) -> None:
        """Tag a hit with this build's marker (skipped when already tagged); the caller holds the lock."""
        tag = LLMCache.build_tag
        if not tag or self._con is None:
            return
        cur = self._con.execute(
            "UPDATE responses SET used_by = ? WHERE key = ? AND (used_by IS NULL OR used_by <> ?)", (tag, key, tag))
        if cur.rowcount:
            self._pending += 1
            if self._pending >= 64:
                self._con.commit()
                self._pending = 0

    def get(self, key: str) -> str | None:
        if self._con is None:
            return None
        with self._lock:
            row = self._con.execute("SELECT response FROM responses WHERE key = ?", (key,)).fetchone()
            if row:
                self._mark(key)
        return str(row[0]) if row else None

    def put(self, key: str, model: str, response: str) -> None:
        if self._con is None:
            return
        with self._lock:
            self._con.execute(
                "INSERT OR REPLACE INTO responses(key, model, response, created_at, used_by) VALUES (?, ?, ?, ?, ?)",
                (key, model, response, int(time.time()), LLMCache.build_tag),
            )
            self._con.commit()
            self._pending = 0

    def count(self) -> int:
        if self._con is None:
            return 0
        with self._lock:
            row = self._con.execute("SELECT COUNT(*) FROM responses").fetchone()
        return int(row[0] if row else 0)

    def prune_unused(self, tag: str) -> int:
        """Delete entries the build tag did not use, returning the count; VACUUM afterwards to shrink the file."""
        if self._con is None or not tag:
            return 0
        with self._lock:
            self._con.commit()
            cur = self._con.execute("DELETE FROM responses WHERE used_by IS NULL OR used_by <> ?", (tag,))
            removed = int(cur.rowcount or 0)
            self._con.commit()
            if removed:
                self._con.execute("VACUUM")
        return removed

    def delete(self, key: str) -> None:
        if self._con is None:
            return
        with self._lock:
            self._con.execute("DELETE FROM responses WHERE key = ?", (key,))
            self._con.commit()

    def close(self) -> None:
        if self._con is not None:
            with self._lock:
                try:
                    self._con.commit()
                except sqlite3.Error:
                    pass
                self._con.close()
                self._con = None


def cache_count(path: Path | str | None) -> int:
    """Count the entries without holding a connection open (for the console's "cache entries" figure)."""
    if not path or not Path(path).is_file():
        return 0
    try:
        con = sqlite3.connect(Path(path))
        try:
            row = con.execute("SELECT COUNT(*) FROM responses").fetchone()
            return int(row[0] if row else 0)
        finally:
            con.close()
    except sqlite3.Error:
        return 0


def cache_key(spec: LLMSpec, messages: list[dict[str, str]], params: dict[str, Any]) -> str:
    payload = {
        "v": CACHE_VERSION,
        "model": spec.model_id,
        "protocol": spec.protocol,
        "messages": messages,
        "params": {k: params[k] for k in sorted(params)},
    }
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# ── Single call ──────────────────────────────────────────────────────────

_THINK_RE = re.compile(r"^.*?</think>", re.DOTALL)
# Reasoning models burn the whole output budget on thinking when the input is table-heavy (finish_reason=length,
# empty body), so the graph build always turns thinking off. The parameter differs per vendor, so all are sent
# together: enable_thinking=False is the Qwen (Alibaba Cloud DashScope) / SGLang spelling; thinking.type=disabled is
# DeepSeek's official one (since V4 in 2026 thinking is on by default with effort=high, and enable_thinking is
# silently ignored; compared on the real box 2026-09-12). A server that does not recognize a parameter answers
# 4xx naming it, in which case only the named one is dropped and this base_url remembers not to send it; when
# none is named all are dropped and the request re-sent, remembered only if the resend really succeeds.
# 2026-09-11 on the real box: a 400 from content moderation was once taken as "parameter rejected", the whole
# process stopped turning thinking off, every unit's output budget was then eaten by reasoning and 20 empty
# responses in a row tripped the circuit breaker.
_NO_THINKING_PARAMS: dict[str, Any] = {"enable_thinking": False, "thinking": {"type": "disabled"}}
_THINKING_PARAMS_REJECTED: dict[str, set[str]] = {}       # base_url → parameter names this server does not accept
# The two "generation interrupted" finish_reason values in DeepSeek's docs: insufficient resources / aborted; retry
_TRANSIENT_FINISH = {"insufficient_system_resource", "aborted"}
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
# 4xx where the server rejects "this particular input": content moderation, context length exceeded. Each
# vendor's codes / wording: Alibaba Cloud data_inspection_failed, OpenAI / Azure content_filter, DeepSeek
# "Content Exists Risk", Volcano / Zhipu / Baidu moderation / sensitive / violation / unsafe, plus the generic
# context_length_exceeded.
_INPUT_REJECTED_RE = re.compile(
    r"data_inspection_failed|inappropriate content|content[_ ]?filter|content_policy_violation|content exists risk"
    r"|moderation|unsafe content|敏感|违规|context[_ ]length|maximum context|too many tokens|prompt is too long"
    r"|input (?:is )?too long",
    re.IGNORECASE,
)


def _input_rejected(status: int, body: str) -> bool:
    if status in _RETRYABLE_STATUS or status < 400 or status >= 500:
        return False
    return status == 413 or bool(_INPUT_REJECTED_RE.search(body or ""))


def _named_thinking_params(body: str, sent: dict[str, Any]) -> set[str]:
    """Which thinking-off parameters the error body names (OpenAI's "Unrecognized request argument supplied: x"
    and pydantic's "extra fields not permitted" both carry the field name, so it can be matched)."""
    text = (body or "").lower()
    # Match whole words: thinking is a substring of enable_thinking, naming the latter must not count the former
    return {k for k in sent if re.search(r"(?<![a-z0-9_])" + re.escape(k.lower()) + r"(?![a-z0-9_])", text)}


def _error_message(body: str) -> str:
    """The message inside the error body (OpenAI-style error.message / top-level message); non-JSON bodies are cut
    to the first 160 characters."""
    try:
        data = json.loads(body or "")
    except (ValueError, TypeError):
        return (body or "")[:160]
    err = data.get("error") if isinstance(data, dict) else None
    if isinstance(err, dict):
        code = str(err.get("code") or "").strip()
        msg = str(err.get("message") or "").strip()
        return f"{code}: {msg}"[:200] if code else msg[:200]
    if isinstance(data, dict) and data.get("message"):
        return str(data["message"])[:200]
    return (body or "")[:160]


def _reasoning_tokens(data: dict[str, Any], message: dict[str, Any]) -> int:
    """Tokens this response spent on reasoning: from usage when present, otherwise estimated from the length of
    reasoning_content."""
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    details = usage.get("completion_tokens_details") if isinstance(usage.get("completion_tokens_details"), dict) else {}
    try:
        n = int(details.get("reasoning_tokens") or 0)
    except (TypeError, ValueError):
        n = 0
    if n > 0:
        return n
    reasoning = str(message.get("reasoning_content") or message.get("reasoning") or "")
    return max(1, len(reasoning) // 3) if reasoning.strip() else 0


class HTTPStatusError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status} {body[:200]}")
        self.status = status


def strip_think(text: str) -> str:
    return _THINK_RE.sub("", text or "", count=1).strip()


def as_messages(prompt_or_messages: str | list[dict[str, str]]) -> list[dict[str, str]]:
    if isinstance(prompt_or_messages, str):
        return [{"role": "user", "content": prompt_or_messages}]
    return [dict(m) for m in prompt_or_messages]


def chat_once(
    spec: LLMSpec,
    messages: list[dict[str, str]],
    *,
    timeout: int = DEFAULT_TIMEOUT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
    session: requests.Session | None = None,
    meta: dict[str, Any] | None = None,
) -> str:
    """One chat call, sent according to the registry row's protocol (openai-compatible / anthropic), returning the
    body. When meta is given it is filled with this response's finish_reason / completion_tokens / truncated
    (output cut off by max_tokens)."""
    base = spec.base_url.rstrip("/")
    headers = {"Content-Type": "application/json"}
    post = (session or requests).post
    if spec.protocol == "anthropic":
        url = base if base.endswith("/v1/messages") else base + "/v1/messages"
        headers["anthropic-version"] = "2023-06-01"
        if spec.api_key:
            headers["x-api-key"] = spec.api_key
        system = "\n\n".join(m["content"] for m in messages if m.get("role") == "system")
        body: dict[str, Any] = {
            "model": spec.model_id,
            "messages": [m for m in messages if m.get("role") != "system"],
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
        if system:
            body["system"] = system
        resp = post(url, headers=headers, json=body, timeout=timeout, allow_redirects=False)
    else:
        url = base + "/chat/completions"
        if spec.api_key:
            headers["Authorization"] = f"Bearer {spec.api_key}"
        body = {
            "model": spec.model_id,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
        rejected = _THINKING_PARAMS_REJECTED.get(base, set())
        sent = {k: v for k, v in _NO_THINKING_PARAMS.items() if k not in rejected}
        body.update(sent)
        resp = post(url, headers=headers, json=body, timeout=timeout, allow_redirects=False)
        for _ in range(len(_NO_THINKING_PARAMS)):
            if not (400 <= resp.status_code < 500 and resp.status_code not in _RETRYABLE_STATUS and sent
                    and not _input_rejected(resp.status_code, resp.text)):
                break        # retryable 4xx such as 429 are load problems, not parameter problems
            # The server may not accept the thinking-off parameters: drop only the named one, or all of them when
            # none is named, and resend. Named ones are remembered right away; unnamed ones only when the resend
            # really succeeds, so content moderation and other parameter errors do not touch the process-wide switch
            named = _named_thinking_params(resp.text, sent)
            drop = named or set(sent)
            for k in drop:
                body.pop(k, None)
                sent.pop(k, None)
            retry = post(url, headers=headers, json=body, timeout=timeout, allow_redirects=False)
            if named or retry.status_code == 200:
                _THINKING_PARAMS_REJECTED.setdefault(base, set()).update(drop)
                resp = retry
            else:
                break
    if 300 <= resp.status_code < 400:
        # Requests carrying credentials do not follow redirects (a cross-host one would hand over the key); this is
        # a configuration problem, not a load problem
        raise LLMInputRejected(f"HTTP {resp.status_code}: the service redirected to {str(resp.headers.get('location') or '')[:120]}; set the model address to the final address")
    if resp.status_code != 200:
        if _input_rejected(resp.status_code, resp.text):
            raise LLMInputRejected(f"HTTP {resp.status_code}: the server rejected this input (content moderation / length): {_error_message(resp.text)}")
        raise HTTPStatusError(resp.status_code, resp.text)
    data = resp.json()
    finish = ""
    message: dict[str, Any] = {}
    if spec.protocol == "anthropic":
        content = "".join(str(b.get("text") or "") for b in data.get("content") or [] if b.get("type") == "text")
        finish = str(data.get("stop_reason") or "")
    else:
        choices = data.get("choices") or []
        choice = (choices[0] if choices else None) or {}
        message = choice.get("message") or {}
        content = str(message.get("content") or "")
        finish = str(choice.get("finish_reason") or "")
    if finish in _TRANSIENT_FINISH:
        raise LLMTransientResponse(f"The server interrupted this generation (finish_reason={finish}); retrying later")
    content = strip_think(content)
    if not content:
        if finish == "content_filter":
            raise LLMInputRejected("The server emptied this reply by content filtering (finish_reason=content_filter)")
        reasoning = _reasoning_tokens(data, message) if spec.protocol != "anthropic" else 0
        if finish in ("length", "max_tokens") and reasoning:
            raise LLMReasoningExhausted(
                f"The model spent all {int(max_tokens)} output tokens on reasoning (finish_reason={finish}, reasoning_tokens={reasoning}) "
                "and returned no text; this endpoint ignores enable_thinking or the model cannot turn thinking off. "
                "Use a non-reasoning model, or one whose thinking can be disabled, for graph building"
            )
        raise LLMEmptyResponse("empty response" + (f" (finish_reason={finish})" if finish else ""))
    if meta is not None:
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        try:
            completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        except (TypeError, ValueError):
            completion = 0
        meta.update(finish_reason=finish, completion_tokens=completion, truncated=finish in ("length", "max_tokens"), cached=False)
    return content


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (LLMInputRejected, LLMReasoningExhausted)):
        return False          # only a different input / model would change this; resending burns it again
    if isinstance(exc, HTTPStatusError):
        return exc.status in _RETRYABLE_STATUS
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True
    if isinstance(exc, requests.RequestException):
        return True
    # Empty responses, JSON parse failures and the like: sporadic on the provider side, worth another try
    return isinstance(exc, (RuntimeError, ValueError)) and not isinstance(exc, LLMInterrupted)


# ── Caller ───────────────────────────────────────────────────────────────

class ChatClient:
    """Caller with cache / retries / circuit breaker / concurrency. One instance per model slot."""

    def __init__(
        self,
        spec: LLMSpec,
        *,
        cache: LLMCache | None = None,
        timeout: int | None = None,
        attempts: int = DEFAULT_ATTEMPTS,
        workers: int | None = None,
        circuit_fails: int | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
        stop: threading.Event | None = None,
        chat: Callable[..., str] = chat_once,
        backoff_base: float = BACKOFF_BASE,
        backoff_max: float = BACKOFF_MAX,
    ) -> None:
        self.spec = spec
        self.cache = cache or LLMCache(None)
        self.timeout = int(timeout if timeout is not None else env_int("KB_GRAPH_LLM_TIMEOUT", DEFAULT_TIMEOUT))
        self.attempts = max(1, int(attempts))
        self.workers = max(1, int(workers if workers is not None else env_int("KB_GRAPH_LLM_CONCURRENCY", DEFAULT_WORKERS)))
        self.circuit_fails = max(0, int(circuit_fails if circuit_fails is not None else env_int("KB_GRAPH_CIRCUIT_FAILS", DEFAULT_CIRCUIT_FAILS)))
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.stop = stop or threading.Event()
        self._chat = chat
        self.backoff_base = float(backoff_base)
        self.backoff_max = float(backoff_max)
        self._lock = threading.Lock()
        self._session = requests.Session()
        # Size the keep-alive pool by concurrency: requests keeps only 10 per host by default, and with higher
        # concurrency the rest are discarded after use and re-handshake every time
        pool = max(10, self.workers)
        for scheme in ("https://", "http://"):
            self._session.mount(scheme, HTTPAdapter(pool_connections=pool, pool_maxsize=pool))
        self.stats: dict[str, int] = {
            "calls": 0, "cache_hits": 0, "cache_evicted": 0, "malformed": 0, "rejected": 0, "retries": 0, "failures": 0,
            "consecutive_failures": 0,
            "prompt_tokens": 0, "completion_tokens": 0,      # tokens of real calls (estimated with the local tokenizer), cache hits excluded
        }
        self.circuit_reason: str | None = None

    # Sampling parameters in the cache key: everything that decides the output, nothing from the network layer
    # (timeout / base_url / key)
    def _params(self, max_tokens: int, temperature: float) -> dict[str, Any]:
        return {"max_tokens": int(max_tokens), "temperature": float(temperature)}

    def key_for(self, prompt_or_messages: str | list[dict[str, str]], *, max_tokens: int | None = None,
                temperature: float | None = None) -> str:
        messages = as_messages(prompt_or_messages)
        return cache_key(self.spec, messages, self._params(
            self.max_tokens if max_tokens is None else max_tokens,
            self.temperature if temperature is None else temperature))

    def chat(
        self,
        prompt_or_messages: str | list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        use_cache: bool = True,
        validate: Callable[[str], bool] | None = None,
        correction: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> str:
        """validate: the caller's check of the content format (e.g. "parses as facts JSON"). A response that fails it
        is not cached and is requested once more with a correction hint; if it still fails, LLMMalformedResponse
        is raised. Format problems do not count towards the circuit breaker's consecutive failures, which signal
        whether the service is alive. Old cached responses go through the check too and are dropped and re-called
        when they fail. When meta is given it is filled with this response's finish_reason / truncated / cached;
        a response cut off by max_tokens is not cached (if it were, the next resume would never see the
        "truncated" signal and would reuse the incomplete result as-is)."""
        messages = as_messages(prompt_or_messages)
        max_tokens = self.max_tokens if max_tokens is None else int(max_tokens)
        temperature = self.temperature if temperature is None else float(temperature)
        key = cache_key(self.spec, messages, self._params(max_tokens, temperature))
        if use_cache:
            hit = self.cache.get(key)
            if hit is not None and validate is not None and not validate(hit):
                self.cache.delete(key)
                with self._lock:
                    self.stats["cache_evicted"] += 1
                hit = None
            if hit is not None:
                with self._lock:
                    self.stats["cache_hits"] += 1
                if meta is not None:
                    meta.update(finish_reason="", completion_tokens=0, truncated=False, cached=True)
                return hit
        if self.circuit_reason:
            raise LLMCircuitOpen(self.circuit_reason)
        last: BaseException | None = None
        nudged: list[dict[str, str]] | None = None
        for attempt in range(1, self.attempts + 1):
            if self.stop.is_set():
                raise LLMInterrupted("stop requested")
            try:
                content = self._chat(
                    self.spec, nudged or messages, timeout=self.timeout, max_tokens=max_tokens,
                    temperature=temperature, session=self._session, meta=meta,
                )
            except LLMInterrupted:
                raise
            except Exception as exc:
                last = exc
                # An empty response gets one more try only, not the full 5-attempt backoff (health check R7); other
                # retryable errors as usual
                limit = min(self.attempts, _EMPTY_RESPONSE_ATTEMPTS) if isinstance(exc, LLMEmptyResponse) else self.attempts
                if not is_retryable(exc) or attempt >= limit:
                    break
                with self._lock:
                    self.stats["retries"] += 1
                delay = min(self.backoff_max, self.backoff_base * (2 ** (attempt - 1)))
                if self.stop.wait(delay):
                    raise LLMInterrupted("stop requested")
                continue
            with self._lock:
                self.stats["calls"] += 1
                self.stats["consecutive_failures"] = 0
                self.stats["prompt_tokens"] += _token_count("\n".join(str(m.get("content") or "") for m in (nudged or messages)))
                self.stats["completion_tokens"] += _token_count(content)
            if validate is not None and not validate(content):
                last = LLMMalformedResponse(f"{self.spec.name}: the response is not in the requested format: {content[:160]!r}")
                if attempt >= min(self.attempts, _MALFORMED_ATTEMPTS):
                    break
                with self._lock:
                    self.stats["retries"] += 1
                # The correction hint must fit the caller's protocol (final review F09): the entity record protocol
                # cannot be corrected with "output only JSON"
                nudged = messages + [{"role": "assistant", "content": content},
                                     {"role": "user", "content": correction or _MALFORMED_NUDGE}]
                continue        # no backoff: format problems have nothing to do with service load
            if use_cache and not (meta or {}).get("truncated"):
                self.cache.put(key, self.spec.model_id, content)
            return content
        if isinstance(last, LLMMalformedResponse):
            with self._lock:
                self.stats["malformed"] += 1
            raise last
        if isinstance(last, LLMInputRejected):
            # The server rejected this input, the service is not down: no consecutive-failure count, the unit-level
            # failure is raised to the caller as-is
            with self._lock:
                self.stats["rejected"] += 1
            raise last
        with self._lock:
            self.stats["failures"] += 1
            self.stats["consecutive_failures"] += 1
            streak = self.stats["consecutive_failures"]
            if self.circuit_fails and streak >= self.circuit_fails and not self.circuit_reason:
                self.circuit_reason = (
                    f"Model “{self.spec.name}” failed {streak} calls in a row (threshold {self.circuit_fails}); "
                    f"last error: {last!r}"
                )
        if self.circuit_reason:
            self.stop.set()
            raise LLMCircuitOpen(self.circuit_reason)
        raise LLMCallError(f"{self.spec.name}: {last!r}") from last

    def run_parallel(
        self,
        items: Iterable[Any],
        fn: Callable[[Any], Any],
        *,
        workers: int | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> list[tuple[Any, Any, Exception | None]]:
        """Run fn(item) for every item and return [(item, result, error)] in input order.

        A single failure (LLMCallError or any other exception) is recorded in error and the run continues; a
        circuit break or an interruption stops the whole pool immediately and re-raises, since carrying on in
        those two cases only burns money or delays the exit.
        """
        items = list(items)
        total = len(items)
        results: list[tuple[Any, Any, Exception | None]] = [(item, None, None) for item in items]
        if not items:
            return results
        workers = max(1, int(workers or self.workers))
        done_count = 0
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="graph-llm")
        pending = {pool.submit(fn, item): idx for idx, item in enumerate(items)}
        try:
            while pending:
                finished, _ = wait(list(pending), timeout=1.0, return_when=FIRST_COMPLETED)
                for future in finished:
                    idx = pending.pop(future)
                    try:
                        results[idx] = (items[idx], future.result(), None)
                    except (LLMCircuitOpen, LLMInterrupted):
                        raise
                    except Exception as exc:
                        results[idx] = (items[idx], None, exc)
                    done_count += 1
                    if progress is not None:
                        progress(done_count, total)
                if self.stop.is_set():
                    raise LLMInterrupted("stop requested")
        except BaseException:
            self.stop.set()
            for future in pending:
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        pool.shutdown(wait=True)
        return results
