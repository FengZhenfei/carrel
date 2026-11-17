from __future__ import annotations

import os
import copy
import hashlib
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context

from ..parse_cache import reusable_in_run, write_cache
from pathlib import Path
from typing import Any

from openai import DefaultHttpxClient, OpenAI, omit

from ..llm_compat import CompatibleRequest, generation_parameters, sdk_arguments
from ..parsers.errors import JobCancelled, service_unreachable
from .images import DEFAULT_MAX_PIXELS
from .images import image_data_url as _image_data_url


# Bump when the schema, the field semantics or the default prompts change, so
# that cached captions from an earlier contract are not reused.
CAPTION_CONTRACT_VERSION = 7  # Preserve distinct source details without secondary field-length caps.

# text_verbatim is deliberately the LAST property: guided decoding emits
# fields in schema order, and it is the only field that can run long. If a
# repetition loop eats the token budget, the truncation lands in verbatim's
# tail while summary/entities/facts are already complete and recoverable by
# complete_json_prefix().
RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {
            "type": "string",
            "enum": [
                "screenshot", "diagram", "chart", "table", "photo",
                "formula", "illustration", "slide", "other",
            ],
        },
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "entities": {"type": "array", "items": {"type": "string"}},
        "facts": {"type": "array", "items": {"type": "string"}},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "decorative": {"type": "boolean"},
        "text_verbatim": {"type": "string"},
    },
    "required": [
        "kind", "title", "summary", "entities",
        "facts", "keywords", "confidence", "decorative", "text_verbatim",
    ],
    "additionalProperties": False,
}

_FIELD_GUIDE = """Field guide (output in this order):
- kind: screenshot (UI screenshot) / diagram (architecture, flow or topology diagram) / chart (statistical chart) /
  table / photo / formula / illustration / slide (a whole slide) / other
- title: the title text in the image or on the page. Empty string when there is none.
- summary: one or two sentences on what it conveys. Empty string when there is no substance.
- entities: product, system, module, organization and person names that appear explicitly in the image
- facts: facts that can be read directly off the image, one self-contained sentence each, including values,
  states, hierarchy or process order. Do not write meta descriptions such as "the figure shows…" or
  "this page introduces…".
- keywords: keywords useful for search
- confidence: high (clearly legible) / medium (partly blurred or occluded) / low (hard to make out)
- decorative: true = purely decorative (no text, data or chart, and no photo of a recognizable product /
  person / organization: background images, dividers, watermarks); otherwise false
- text_verbatim: the text in the image, transcribed as-is in reading order. Keep the original case,
  numbering, units and symbols; do not translate, rephrase or complete elided content. Transcribe a
  repeated passage once. Empty string when there is no text.

Extra requirements by image type:
- UI screenshot (screenshot): facts lists the key visible text item by item -- menu entries, field names,
  buttons, table headers, status values, hints -- in reading order; do not describe layout or colours,
  do not draw flows.
- Architecture / flow diagram (diagram): facts records nodes and their links as "A → B: note"; never
  output mermaid, code blocks or any markup language.
- Several logos, name lists, participant or customer walls: entities lists every name that appears,
  never summarized as "several hospitals" or "some companies".
- Photo (photo): summary states the subject in one sentence; with no text or data, leave facts empty
  and set decorative as appropriate.

Output only what is really visible in the image. Use ? for illegible characters.
Return an empty string for string fields with no content and an empty array for array fields with no
matching items; never fill them with placeholders, field names or "none". Leaving a field empty beats guessing."""

DEFAULT_PROMPT = (
    "This is an image from a knowledge-base document; examine it carefully and output a structured result.\n"
    "Common kinds: UI screenshots, architecture diagrams, flow charts, topologies, data charts, tables, photos, "
    "formulas, whole slides; decide the kind first, then extract the information item by item.\n"
    "Transcribe text, values, units and identifiers exactly as written; ignore headers, footers, page numbers, "
    "watermarks and purely decorative elements, and extract only content that carries information."
)
# The console's prompt textarea placeholder (app/kb_server/static/index.html)
# shows this default text; keep the two in sync.


def compose_prompt(route: str, custom: str | None = None, *, filename: str | None = None) -> str:
    """Effective caption prompt. Every route ("figure" / "docx_image" /
    "image_file") shares the same default paragraph; a per-KB custom
    instruction replaces that paragraph only -- the field guide is always
    appended, because it defines the JSON contract the parser depends on.
    Standalone image files additionally carry their filename, the only
    context such a picture comes with."""
    if route == "image_file" and not filename:
        raise ValueError("image_file route requires a filename")
    parts = [(custom or "").strip() or DEFAULT_PROMPT]
    if filename:
        parts.append(f"(File name: “{filename}”, usable as a hint for the topic; title still follows the actual text in the image.)")
    parts.append(_FIELD_GUIDE)
    return "\n\n".join(parts)


def language_is_chinese(language: str | None) -> bool:
    """Whether the output language recorded by label extraction is Chinese (empty = not extracted yet)."""
    value = str(language or "").strip().lower()
    if not value:
        return False
    return value.startswith("zh") or "chinese" in value or "中文" in value or "汉语" in value or "漢語" in value


def prompt_with_language(custom: str | None, language: str | None) -> str:
    """Append the output-language instruction to the caption prompt so that
    summary / facts / keywords match the corpus language (health check D2):
    otherwise the keyword index mixes, say, English text with descriptions in
    another language. With a known language (from label extraction) the
    instruction names it; without one, the model follows the language of the
    text in the image. The result is part of the caption cache key."""
    lang = str(language or "").strip()
    base = (custom or "").strip() or DEFAULT_PROMPT
    if not lang:
        return (f"{base}\n\nOutput language: write title, summary, facts and keywords in the language of the "
                "text in the image (proper nouns, part numbers and code stay as they are); text_verbatim is "
                "always transcribed as it appears, never translated.")
    return (f"{base}\n\nOutput language: {lang}. Write title, summary, facts and keywords in {lang} "
            "(proper nouns, part numbers and code stay as they are); text_verbatim is always transcribed "
            "as it appears, never translated.")


EMPTY_RESULT: dict[str, Any] = {
    "kind": "other", "title": "", "summary": "", "text_verbatim": "",
    "entities": [], "facts": [], "keywords": [], "confidence": "low", "decorative": False,
}


_CLIENT_LOCK = threading.Lock()
_CLIENTS: dict[tuple[str, str], OpenAI] = {}


def _client_for(base_url: str, api_key: str) -> OpenAI:
    """One client per endpoint. The OpenAI client is thread-safe and pools
    connections; building a fresh one per image made every caption pay a new
    TCP/TLS handshake, which adds up once hundreds of figures hit a local
    vLLM."""
    key = (base_url, api_key)
    with _CLIENT_LOCK:
        client = _CLIENTS.get(key)
        if client is None:
            # The SDK default of a 600s timeout + 2 retries = half an hour worst case per image; a
            # PDF with two hundred images could hold a worker for over ten hours while the service
            # hangs. Give it a cap proportionate to an image description (the embedding client has
            # done this for a long time).
            client = OpenAI(
                api_key=api_key or "not-required",
                base_url=base_url,
                http_client=DefaultHttpxClient(follow_redirects=False),
                timeout=float(os.getenv("VLM_TIMEOUT_SECONDS", "180")),
                max_retries=int(os.getenv("VLM_MAX_RETRIES", "1")),
            )
            _CLIENTS[key] = client
    return client


def _compatible_completion(client: OpenAI, request: CompatibleRequest, *, api_key: str | None = None):
    while True:
        body = copy.deepcopy(request.body)
        if "response_format" not in body:
            body["messages"][0]["content"][0]["text"] += "\n\nOutput exactly one JSON object and nothing else."
        try:
            return client.chat.completions.create(**sdk_arguments(body),
                **({"extra_headers": {"Authorization": omit}} if api_key == "" else {}))
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            # SDK errors expose the server's JSON body separately from their display text.
            detail = getattr(exc, "body", None)
            error = json.dumps(detail) if isinstance(detail, (dict, list)) else str(exc)
            if not request.adapt(status, error):
                raise


def image_data_url(path: Path, max_pixels: int | None = DEFAULT_MAX_PIXELS) -> str:
    """Kept as the VLM module's entry point; the normalisation (pass-through
    for PNG/JPEG/WebP within budget, re-encode/downscale otherwise) lives in
    vision.images so the visual-embedding client sends the same bytes."""
    return _image_data_url(path, max_pixels)


# Match literal loops only in unfinished, output-limited responses. Complete
# model fields, including repeated table rows, are preserved as returned.
_LITERAL_LOOP = re.compile(r"(?P<unit>.{4,160}?)(?:[ \t\r\n]*(?P=unit)){7,}", re.DOTALL)


def normalize_result(data: Any) -> dict[str, Any]:
    """Normalize the schema without imposing a second content-length budget."""
    result = copy.deepcopy(EMPTY_RESULT)
    if not isinstance(data, dict):
        return result
    for key in ("kind", "title", "summary", "text_verbatim", "confidence"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            result[key] = value.strip()
    for key in ("entities", "facts", "keywords"):
        value = data.get(key)
        items = value if isinstance(value, list) else [value] if isinstance(value, str) else []
        kept = []
        for item in items:
            text = str(item).strip()
            if text:
                kept.append(text)
        result[key] = list(dict.fromkeys(kept))
    if result["confidence"] not in {"high", "medium", "low"}:
        result["confidence"] = "low"
    decorative = data.get("decorative")
    result["decorative"] = decorative if isinstance(decorative, bool) else str(decorative).strip().lower() == "true"
    return result


def _raw_tail_runaway(raw: str, finish_reason: str) -> bool:
    """Check a truncated string before prefix repair discards it."""
    if finish_reason != "length":
        return False
    start = None
    escaped = False
    for i, char in enumerate(raw):
        if escaped:
            escaped = False
        elif char == "\\" and start is not None:
            escaped = True
        elif char == '"':
            start = i + 1 if start is None else None
    if start is None:
        return False
    tail = raw[start:]
    # Completed rows can legitimately repeat. Only a dominant literal loop in
    # an unfinished string is grounds to avoid spending a larger output budget.
    return any(len(match.group()) >= max(200, len(tail) * 0.8)
               for match in _LITERAL_LOOP.finditer(tail))


def complete_json_prefix(raw: str) -> str | None:
    """A truncated schema-constrained response is a valid-JSON *prefix* (the
    grammar guarantees no syntax error before the cut). Closing the open
    string and brackets recovers every field that finished generating.
    Returns completed JSON text, or None when the input is not a repairable
    prefix."""
    stack: list[str] = []
    in_str = esc = False
    string_start = 0
    for offset, ch in enumerate(raw):
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
            string_start = offset
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack or stack[-1] != ch:
                return None
            stack.pop()
    if not stack and not in_str:
        return raw  # already complete
    # A closed JSON string is a complete value. Never promote a half sentence
    # (possibly missing its condition or negation) by adding its closing quote.
    fixed = raw[:string_start] if in_str else raw
    tail = fixed.rstrip()
    if tail.endswith(":"):
        # the value after this key never started; drop the dangling key
        cut = max(tail.rfind(","), tail.rfind("{"))
        if cut >= 0:
            tail = tail[: cut + 1]
    if tail.endswith(","):
        tail = tail[:-1]
    fixed = tail + "".join(reversed(stack))
    try:
        json.loads(fixed)
    except Exception:
        return None
    return fixed


def parse_structured(content: str) -> dict[str, Any] | None:
    """Parse a schema-constrained response, repairing a truncated prefix.
    None means unrecoverable."""
    try:
        obj = json.loads(content)
    except Exception:
        fixed = complete_json_prefix(content)
        if fixed is None:
            return None
        try:
            obj = json.loads(fixed)
        except Exception:
            return None
    return obj if isinstance(obj, dict) else None


def parse_jsonish(text: str) -> dict[str, Any]:
    """Fallback for servers that cannot constrain decoding to the schema."""
    raw = text.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    try:
        parsed = json.loads(raw)
    except Exception:
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(raw[start : end + 1])
            except Exception:
                parsed = None
        else:
            parsed = None
        if parsed is None and start >= 0:
            # e.g. truncated output: a JSON prefix with no closing brace
            fixed = complete_json_prefix(raw[start:])
            if fixed is not None:
                try:
                    parsed = json.loads(fixed)
                except Exception:
                    parsed = None
    if isinstance(parsed, dict):
        return normalize_result(parsed)
    # Unparseable and unrepairable: salvage the summary field if its text
    # survived. NEVER embed the raw model output as a summary -- a broken
    # JSON blob in the chunk text poisons the vector and the keyword index.
    result = dict(EMPTY_RESULT)
    m = re.search(r'"summary"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
    if m:
        try:
            result["summary"] = str(json.loads(f'"{m.group(1)}"')).strip()
        except Exception:
            result["summary"] = m.group(1).strip()
    return result


def caption_image(
    *,
    image_path: Path,
    base_url: str,
    api_key: str,
    model_id: str,
    prompt: str | None = None,
    cache_json: Path | None = None,
    temperature: float = 0.1,
    top_p: float = 0.8,
    max_tokens: int = 4096,
    structured: bool = True,
    repetition_penalty: float = 1.05,
    max_pixels: int | None = DEFAULT_MAX_PIXELS,
) -> dict[str, Any]:
    effective_prompt = prompt or compose_prompt("figure")
    body: dict[str, Any] = {
        "model": model_id,
        **generation_parameters(model_id, base_url=base_url, max_tokens=max_tokens,
                                temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty),
    }
    if structured:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "visual_block", "schema": RESULT_SCHEMA, "strict": True},
        }
    request = CompatibleRequest(base_url, model_id, "openai", body)
    cache_identity = {
        "contract_version": CAPTION_CONTRACT_VERSION,
        "model_id": model_id,
        "endpoint_sha256": hashlib.sha256(base_url.rstrip("/").encode("utf-8")).hexdigest(),
        "prompt_sha256": hashlib.sha256(effective_prompt.encode("utf-8")).hexdigest(),
        "request_parameters": request.parameters(),
        "max_pixels": max_pixels,
    }
    if cache_json and cache_json.exists():
        try:
            cached = json.loads(cache_json.read_text(encoding="utf-8"))
        except Exception:
            cached = None
        # An empty summary is now a legitimate answer, so identity is the only
        # thing that decides whether a cached caption can be reused.
        if isinstance(cached, dict) and reusable_in_run(cached) and cached.get("_cache") == cache_identity:
            result = normalize_result(cached)
            result["vlm_cache_hit"] = True
            return result

    client = _client_for(base_url, api_key)
    request.body["messages"] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": effective_prompt},
                {"type": "image_url", "image_url": {"url": image_data_url(image_path, max_pixels)}},
            ],
        }
    ]
    resp = _compatible_completion(client, request, api_key=api_key)
    constrained = "response_format" in request.body

    content = resp.choices[0].message.content or ""
    finish_reason = str(getattr(resp.choices[0], "finish_reason", "") or "")
    data: dict[str, Any] | None
    if constrained:
        obj = parse_structured(content)
        data = normalize_result(obj) if obj is not None else None
    else:
        data = parse_jsonish(content)
        if not (data["summary"] or data["title"] or data["facts"] or data["entities"]
                or (data.get("decorative") is True and data.get("confidence") == "high")):
            data = None

    if data is not None and _raw_tail_runaway(content, finish_reason):
        data["runaway"] = True
    first_data = data if data and any(data.get(k) for k in ("summary", "title", "facts", "text_verbatim", "entities")) else None
    first_finish = finish_reason
    length_retried = False
    _log_caption_usage(resp, max_tokens)
    if (finish_reason == "length" and data and not data.get("runaway") and max_tokens < 8192
            and any(data.get(k) for k in ("summary", "facts", "text_verbatim", "entities"))):
        # One larger-budget attempt, including recoverable JSON prefixes. Do
        # not spend more tokens on detected repetition; that uses the existing
        # degraded retry below. A failed retry cannot erase a usable prefix.
        length_retried = True
        retry_body = copy.deepcopy(request.body)
        token_key = "max_completion_tokens" if "max_completion_tokens" in retry_body else "max_tokens"
        retry_body[token_key] = min(8192, max_tokens * 2)
        try:
            retry_response = _compatible_completion(client, CompatibleRequest(base_url, model_id, "openai", retry_body), api_key=api_key)
            _log_caption_usage(retry_response, retry_body[token_key])
            candidate_content = retry_response.choices[0].message.content or ""
            candidate = parse_jsonish(candidate_content)
            candidate_finish = str(getattr(retry_response.choices[0], "finish_reason", "") or "")
            if _raw_tail_runaway(candidate_content, candidate_finish):
                candidate["runaway"] = True
            if (not candidate.get("runaway") and any(candidate.get(k) for k in ("summary", "facts", "text_verbatim", "entities"))
                    and (candidate_finish == "stop" or len(json.dumps(candidate)) >= len(json.dumps(data or {})))):
                data, finish_reason = candidate, candidate_finish
        except JobCancelled:
            raise
        except Exception as exc:
            print(f"[vlm] output-limit retry failed ({type(exc).__name__}); retaining first result", flush=True)
        data = data or parse_jsonish(content)
        data["vlm_length_retry"] = True

    degraded_reason = "repetitive_output" if data and data.get("runaway") else "invalid_output"
    if data is not None and data.get("runaway") and not length_retried:
        print(f"[vlm] caption has an unfinished repeated field; degraded retry", flush=True)
        data = None
    if data is None and not length_retried:
        # Unrecoverable output -- usually a repetition loop that truncated
        # mid-string beyond repair. One degraded retry: stronger anti-repeat
        # and verbatim disabled, trading the transcription for a clean
        # structured description instead of garbage.
        print(f"[vlm] caption unparseable or runaway (finish_reason={finish_reason}); degraded retry", flush=True)
        retry = copy.deepcopy(request.body)
        if "repetition_penalty" in retry:  # do not reintroduce a rejected parameter
            retry["repetition_penalty"] = max(1.15, float(retry["repetition_penalty"]))
        retry["messages"] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": effective_prompt + "\n\nNote: this time text_verbatim must be an empty string."},
                    request.body["messages"][0]["content"][1],
                ],
            }
        ]
        degraded_request = CompatibleRequest(base_url, model_id, "openai", retry)
        try:
            resp = _compatible_completion(client, degraded_request, api_key=api_key)
        except JobCancelled:
            raise
        except Exception:
            if not first_data:
                raise
            resp = None
        constrained = "response_format" in degraded_request.body
        content = (resp.choices[0].message.content or "") if resp else ""
        finish_reason = str(getattr(resp.choices[0], "finish_reason", "") or "") if resp else first_finish
        obj = parse_structured(content) if constrained else None
        data = normalize_result(obj) if obj is not None else parse_jsonish(content)
        if _raw_tail_runaway(content, finish_reason):
            data["runaway"] = True
        if first_data and not any(data.get(k) for k in ("summary", "facts", "text_verbatim", "entities")):
            data, finish_reason = first_data, first_finish
        data["vlm_degraded_retry"] = True
        data["vlm_degraded_reason"] = degraded_reason
        if data.get("runaway"):
            # The unfinished repeated field was discarded by prefix repair.
            # Keep complete fields, lower confidence and report incomplete recovery.
            print(f"[vlm] still running away after retry; kept {len(data.get('facts') or [])} facts, confidence low", flush=True)
            data["confidence"] = "low"
            data["vlm_runaway"] = True
    data["vlm_truncated"] = finish_reason == "length"
    data["vlm_cache_hit"] = False
    # An explicit, confident decoration classification is a complete result, even without text.
    # Empty unknown images and degraded transcriptions still need a future full attempt.
    degraded = bool(data.get("vlm_degraded_retry"))
    empty = not any(str(data.get(k) or "").strip() for k in ("summary", "title", "text_verbatim")) \
        and not (data.get("facts") or data.get("entities") or data.get("keywords"))
    decoration = data.get("decorative") is True and data.get("confidence") == "high"
    cacheable = not degraded and not data.get("runaway") and not data.get("vlm_truncated") and (not empty or decoration)
    if cacheable:
        request.accept()
        cache_identity["request_parameters"] = request.parameters()
    if cache_json and cacheable:
        cache_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {**data, "_cache": cache_identity}
        payload.pop("vlm_cache_hit", None)
        write_cache(cache_json, payload)
    elif cache_json:
        print(f"[vlm] not caching the {'truncated' if data.get('vlm_truncated') else 'degraded' if degraded else 'empty'} result; retried next time", flush=True)
    return data


def _log_caption_usage(response, budget: int) -> None:
    usage = getattr(response, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)
    print("[vlm] " + json.dumps({"finish_reason": getattr(response.choices[0], "finish_reason", None),
        "output_budget": budget, "completion_tokens": getattr(usage, "completion_tokens", None),
        "reasoning_tokens": getattr(details, "reasoning_tokens", None)}), flush=True)


def caption_images_parallel(
    jobs: list[tuple[str, Path, str | None, Path | None]],
    *,
    base_url: str,
    api_key: str,
    model_id: str,
    concurrency: int = 1,
    temperature: float = 0.1,
    top_p: float = 0.8,
    max_tokens: int = 4096,
    structured: bool = True,
    repetition_penalty: float = 1.05,
    max_pixels: int | None = DEFAULT_MAX_PIXELS,
    progress_cb: Any = None,
) -> dict[str, dict[str, Any]]:
    if not jobs:
        return {}
    workers = max(1, min(int(concurrency or 1), len(jobs)))
    results: dict[str, dict[str, Any]] = {}
    groups: dict[tuple[str, str], list[tuple[str, Path, str | None, Path | None]]] = {}
    for job in jobs:
        _, image_path, prompt, _ = job
        try:
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        except OSError:
            digest = str(image_path)
        groups.setdefault((digest, prompt or compose_prompt("figure")), []).append(job)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_id = {
            executor.submit(
                copy_context().run, caption_image,
                image_path=image_path,
                base_url=base_url,
                api_key=api_key,
                model_id=model_id,
                prompt=prompt,
                cache_json=cache_json,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                structured=structured,
                repetition_penalty=repetition_penalty,
                max_pixels=max_pixels,
            ): group
            for group in groups.values()
            for job_id, image_path, prompt, cache_json in group[:1]
        }
        image_by_id = {job_id: image_path for job_id, image_path, _p, _c in jobs}
        completed = 0
        for future in as_completed(future_to_id):
            group = future_to_id[future]
            job_id = group[0][0]
            try:
                results[job_id] = future.result()
            except Exception as exc:
                if service_unreachable(exc):
                    # The model service cannot be reached: the remaining images need not be tried, the whole
                    # run stops; the worker then returns the job to the queue until the service is ready
                    print(f"[vlm] endpoint unreachable, caption run stopped at {completed}/{len(jobs)}: {exc!r}", flush=True)
                    executor.shutdown(wait=True, cancel_futures=True)
                    raise
                # This used to swallow the exception whole into results[job_id]["error"] and stop
                # there: the log only carried the upstream summary vlm_failed=N/M, and nobody could
                # see the real cause. Yet one failed image is enough to keep a whole document out of
                # the index (VLM_FAILURE_RETRY_RATIO defaults to 0.001 = zero tolerance), so "some
                # document retries repeatedly and finally fails" became a fault with no clue at all.
                print(
                    f"[vlm] caption failed image={image_by_id.get(job_id, job_id)} {exc!r}",
                    flush=True,
                )
                failed = dict(EMPTY_RESULT)
                failed["confidence"] = "low"
                failed["error"] = repr(exc)
                results[job_id] = failed
            for alias, _, _, _ in group[1:]:
                results[alias] = copy.deepcopy(results[job_id])
            completed += len(group)
            if completed == 1 or completed == len(jobs) or completed % workers == 0:
                print(f"[vlm] progress {completed}/{len(jobs)}", flush=True)
            if progress_cb is not None:
                try:
                    progress_cb(completed, len(jobs))
                except JobCancelled:
                    # The job was cancelled: images not yet started are no longer sent to the model, the ones in
                    # flight are allowed to finish; the cancellation goes back to the parse job unchanged
                    executor.shutdown(wait=True, cancel_futures=True)
                    raise
                except Exception:
                    pass  # progress reporting must never fail a caption run
    return results
