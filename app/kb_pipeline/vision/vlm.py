from __future__ import annotations

import os
import hashlib
import json
import re
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI

from ..parsers.errors import JobCancelled, service_unreachable
from .images import DEFAULT_MAX_PIXELS
from .images import image_data_url as _image_data_url


# Bump when the schema, the field semantics or the default prompts change, so
# that cached captions from an earlier contract are not reused.
CAPTION_CONTRACT_VERSION = 5  # v5: decorative field; extraction requirements per image kind (screenshot OCR goes into facts, logo walls list every name, no mermaid for architecture diagrams)

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
                api_key=api_key,
                base_url=base_url,
                timeout=float(os.getenv("VLM_TIMEOUT_SECONDS", "180")),
                max_retries=int(os.getenv("VLM_MAX_RETRIES", "1")),
            )
            _CLIENTS[key] = client
    return client


def image_data_url(path: Path, max_pixels: int | None = DEFAULT_MAX_PIXELS) -> str:
    """Kept as the VLM module's entry point; the normalisation (pass-through
    for PNG/JPEG/WebP within budget, re-encode/downscale otherwise) lives in
    vision.images so the visual-embedding client sends the same bytes."""
    return _image_data_url(path, max_pixels)


VERBATIM_MAX_CHARS = 4000


def _collapse_repeats(text: str) -> str:
    """Fold runs of identical lines -- degenerate repetition loops must not
    reach the index even when the JSON itself parsed fine."""
    lines = text.splitlines()
    out: list[str] = []
    for line in lines:
        if out and line.strip() and line.strip() == out[-1].strip():
            continue
        out.append(line)
    collapsed = "\n".join(out)
    return collapsed[:VERBATIM_MAX_CHARS]


_ARRAY_FIELD_CAP = int(os.getenv("VLM_ARRAY_FIELD_CAP", "64"))
_GROWING_MIN_CHAIN = 3          # this many consecutive items each extending the previous one by prefix count as a repetition chain ("pending folder", "folder inside pending folder", ...)
_ITEM_MAX_CHARS = 200           # a single item longer than this and highly repetitive inside (one 4-char gram over half of it) counts as runaway
# Caps for string fields: by contract title is one line and summary one or two sentences; anything
# beyond is not a description but a runaway, or FACTS written into the wrong place
_STRING_CAPS = {"title": 120, "summary": 600}
# Numbering, punctuation, whitespace: in "1. task list; 2. task list; ..." they dilute the share of
# the 4-char gram below half, so they are stripped before judging repetition
_NOISE_RE = re.compile(r"[\s\d\W_]+")
_GRAM = 4
_COMPRESS_MIN_BYTES = 400       # check the compression ratio only on strings this long: a loop body longer than 4 chars (a whole sentence repeated) escapes the gram-share test
_COMPRESS_MAX_RATIO = 0.15      # measured: repetition 0.01-0.14; legitimate repetition such as pinout / parameter tables / menu vocabularies >= 0.20
_CYCLE_MAX_GAP = 60             # the same gram occurring three times in a row, each at most this many chars apart, marks where the loop begins


def _compressible(text: str) -> bool:
    raw = (text or "").encode("utf-8")
    return len(raw) >= _COMPRESS_MIN_BYTES and len(zlib.compress(raw, 6)) < len(raw) * _COMPRESS_MAX_RATIO


def _top_gram(t: str) -> tuple[str, int]:
    grams: dict[str, int] = {}
    for i in range(len(t) - _GRAM + 1):
        g = t[i:i + _GRAM]
        grams[g] = grams.get(g, 0) + 1
    if not grams:
        return "", 0
    top = max(grams, key=grams.get)
    return top, grams[top]


def _repetitive(text: str) -> bool:
    """One 4-char gram recurring within an item (over half of it) is repetition; long enough items
    are also checked by compression ratio."""
    t = text.replace(" ", "")
    if len(t) < 40:
        return False
    _, n = _top_gram(t)
    return n * _GRAM >= len(t) * 0.5 or _compressible(t)


def _repetitive_prose(text: str) -> bool:
    """Prose fields such as title / summary: strip numbering, punctuation and whitespace before judging
    (2026-09-10 product material, the "automated tasks" screenshot: "1. task list; 2. task list; ..."
    516 times, and with only spaces removed the gram share was 45%, so it was missed)."""
    t = _NOISE_RE.sub("", text or "")
    if len(t) < 40:
        return False
    _, n = _top_gram(t)
    return n * _GRAM >= len(t) * 0.5 or _compressible(t)


def _cut_at_cycle(text: str) -> str:
    """Cut a repetitive string at the start of the loop: find the most frequent 4-char gram, locate it
    in the original text (numbering / punctuation may sit in between), and the first place where three
    consecutive occurrences are all close together is the loop; keep up to the loop's first item. If
    it cannot be located, return the text unchanged and leave it to the cap truncation."""
    top, _ = _top_gram(_NOISE_RE.sub("", text))
    if not top:
        return text
    pattern = re.compile(r"[\s\d\W_]*".join(re.escape(ch) for ch in top))
    starts = [m.start() for m in pattern.finditer(text)]
    for i in range(len(starts) - 2):
        if starts[i + 1] - starts[i] <= _CYCLE_MAX_GAP and starts[i + 2] - starts[i + 1] <= _CYCLE_MAX_GAP:
            return text[: starts[i + 1]].rstrip(" \t;；,，、.。:：0123456789")
    return text


def _tame_string(text: str, cap: int) -> tuple[str, bool]:
    """title / summary: a repetitive one is cut at the start of the loop and reported as runaway; a
    non-repetitive one is still truncated at the cap. Returns (text, runaway)."""
    text = (text or "").strip()
    if not text:
        return text, False
    if _repetitive_prose(text):
        return _cut_at_cycle(text)[:cap].rstrip(), True
    return text[:cap].rstrip(), False


def _collapse_growing(items: list[str]) -> tuple[list[str], int]:
    """Collapse a repetition chain that grows item by item: an item that merely extends the previous
    one (same prefix, only ever longer) is dropped, keeping only the chain head; an over-long item that
    repeats internally is dropped too. Returns (kept items, number dropped). Exact duplicate lines are
    deduped separately in normalize_result."""
    kept: list[str] = []
    dropped = 0
    chain = 0
    for item in items:
        prev = kept[-1] if kept else ""
        head = prev[: max(12, len(prev) * 2 // 3)] if prev else ""
        growing = bool(prev) and len(item) > len(prev) and bool(head) and item.startswith(head)
        if growing:
            chain += 1
            if chain >= _GROWING_MIN_CHAIN - 1:
                dropped += 1
                continue
        else:
            chain = 0
        if len(item) > _ITEM_MAX_CHARS and _repetitive(item):
            dropped += 1
            continue
        kept.append(item)
    # The first two items after the chain head are part of the chain too: only the head survives
    if dropped:
        pruned: list[str] = []
        for item in kept:
            prev = pruned[-1] if pruned else ""
            head = prev[: max(12, len(prev) * 2 // 3)] if prev else ""
            if prev and len(item) > len(prev) and item.startswith(head):
                dropped += 1
                continue
            pruned.append(item)
        kept = pruned
    return kept, dropped


def normalize_result(data: Any) -> dict[str, Any]:
    """Coerce whatever came back into the documented shape."""
    result = dict(EMPTY_RESULT)
    if not isinstance(data, dict):
        return result
    runaway_dropped = 0
    for key in ("kind", "title", "summary", "text_verbatim", "confidence"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            result[key] = value.strip()
    # String fields run away too (2026-09-10 product material, the "automated tasks" screenshot: the
    # model wrote FACTS into summary as "1. task list; 2. task list; ..." 516 times, over 5,000 chars;
    # title / summary used to have neither a repetition check nor a cap, so it went into the chunk
    # verbatim (3,157 tokens) and into the cache). Repetitive ones are cut at the loop start and
    # recorded as runaway; non-repetitive ones are capped as well.
    for key, cap in _STRING_CAPS.items():
        tamed, ran = _tame_string(result[key], cap)
        result[key] = tamed
        if ran:
            runaway_dropped += 1
    for key in ("entities", "facts", "keywords"):
        value = data.get(key)
        if isinstance(value, list):
            items = [str(x).strip() for x in value if str(x).strip()]
        elif isinstance(value, str) and value.strip():
            items = [value.strip()]
        else:
            items = []
        # A model stuck in a repetition loop fills the whole array with the same sentence
        # (text_verbatim already gets collapsed; these three fields used to be kept verbatim and went
        # straight into the payload and the keyword index).
        seen: set[str] = set()
        deduped: list[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
            if len(deduped) >= _ARRAY_FIELD_CAP:
                break
        # Item-by-item growing repetition (each item wraps one more layer than the last) and
        # repetition inside a single item: collapsed and recorded as runaway (2026-09-09 product
        # material, one "cloud antivirus overview page" screenshot had FACTS of 3955 tokens). A
        # runaway result is retried / degraded by the caller and never cached
        collapsed, dropped = _collapse_growing(deduped)
        if dropped:
            runaway_dropped += dropped
        result[key] = collapsed
    if result["confidence"] not in {"high", "medium", "low"}:
        result["confidence"] = "low"
    decorative = data.get("decorative")
    result["decorative"] = decorative if isinstance(decorative, bool) else str(decorative).strip().lower() == "true"
    verbatim = _collapse_repeats(result["text_verbatim"])
    if _compressible(verbatim):
        # Line-wise collapsing cannot catch repetition written on a single line ("Home Password Home
        # Password ..."): if the transcription is garbage, drop it entirely
        verbatim = ""
        runaway_dropped += 1
    result["text_verbatim"] = verbatim
    if runaway_dropped:
        result["runaway"] = True
        result["runaway_dropped"] = runaway_dropped
    return result


def complete_json_prefix(raw: str) -> str | None:
    """A truncated schema-constrained response is a valid-JSON *prefix* (the
    grammar guarantees no syntax error before the cut). Closing the open
    string and brackets recovers every field that finished generating.
    Returns completed JSON text, or None when the input is not a repairable
    prefix."""
    stack: list[str] = []
    in_str = esc = False
    for ch in raw:
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
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack or stack[-1] != ch:
                return None
            stack.pop()
    if not stack and not in_str:
        return raw  # already complete
    fixed = raw[:-1] if esc else raw
    if in_str:
        fixed += '"'
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
            result["summary"] = str(json.loads(f'"{m.group(1)}"')).strip()[:500]
        except Exception:
            result["summary"] = m.group(1).strip()[:500]
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
    cache_identity = {
        "contract_version": CAPTION_CONTRACT_VERSION,
        "model_id": model_id,
        "prompt_sha256": hashlib.sha256(effective_prompt.encode("utf-8")).hexdigest(),
        "temperature": temperature,
        "top_p": top_p,
        # Mild anti-repeat: degenerate loops die within a few lines while a
        # pinout's legitimately repeated labels survive. Part of the cache
        # identity -- changing it must re-caption.
        "repetition_penalty": repetition_penalty,
        "structured": bool(structured),
        "max_pixels": max_pixels,
    }
    if cache_json and cache_json.exists():
        try:
            cached = json.loads(cache_json.read_text(encoding="utf-8"))
        except Exception:
            cached = None
        # An empty summary is now a legitimate answer, so identity is the only
        # thing that decides whether a cached caption can be reused.
        if isinstance(cached, dict) and cached.get("_cache") == cache_identity:
            result = normalize_result(cached)
            if result.get("runaway"):
                print(f"[vlm] cached caption looks like a repetition runaway ({result.get('runaway_dropped')} items dropped); re-captioning", flush=True)
            else:
                result["vlm_cache_hit"] = True
                return result

    client = _client_for(base_url, api_key)
    request: dict[str, Any] = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": effective_prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url(image_path, max_pixels)}},
                ],
            }
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "extra_body": {"repetition_penalty": repetition_penalty},
    }
    constrained = bool(structured)
    if structured:
        request["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "visual_block", "schema": RESULT_SCHEMA, "strict": True},
        }

    try:
        resp = client.chat.completions.create(**request)
    except Exception as exc:
        if not structured:
            raise
        # Degrade only when the endpoint rejects json_schema. Connection errors / timeouts / 5xx used
        # to land here too and be re-sent with the unconstrained prompt -- one wasted inference, and
        # the result was still written to the cache as structured.
        status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
        try:
            status = int(status) if status is not None else None
        except (TypeError, ValueError):
            status = None
        if status is not None and not (400 <= status < 500):
            raise
        if status is None and not isinstance(exc, (TypeError, ValueError, KeyError)):
            raise
        # Endpoint does not support schema-constrained decoding; fall back to
        # asking for JSON in the prompt and parsing leniently.
        print(f"[vlm] structured output rejected, retrying unconstrained: {exc!r}", flush=True)
        constrained = False
        request.pop("response_format", None)
        request["messages"][0]["content"][0]["text"] = (
            effective_prompt + "\n\nOutput exactly one JSON object and nothing else."
        )
        resp = client.chat.completions.create(**request)

    content = resp.choices[0].message.content or ""
    finish_reason = str(getattr(resp.choices[0], "finish_reason", "") or "")
    data: dict[str, Any] | None
    if constrained:
        obj = parse_structured(content)
        data = normalize_result(obj) if obj is not None else None
    else:
        data = parse_jsonish(content)
        if not (data["summary"] or data["title"] or data["facts"] or data["entities"]):
            data = None

    if data is not None and data.get("runaway"):
        print(f"[vlm] caption ran away ({data.get('runaway_dropped')} repeated items); degraded retry", flush=True)
        data = None
    if data is None:
        # Unrecoverable output -- usually a repetition loop that truncated
        # mid-string beyond repair. One degraded retry: stronger anti-repeat
        # and verbatim disabled, trading the transcription for a clean
        # structured description instead of garbage.
        print(f"[vlm] caption unparseable or runaway (finish_reason={finish_reason}); degraded retry", flush=True)
        retry = dict(request)
        retry["extra_body"] = {"repetition_penalty": max(1.15, repetition_penalty)}
        retry["messages"] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": effective_prompt + "\n\nNote: this time text_verbatim must be an empty string."},
                    request["messages"][0]["content"][1],
                ],
            }
        ]
        resp = client.chat.completions.create(**retry)
        content = resp.choices[0].message.content or ""
        obj = parse_structured(content) if constrained else None
        data = normalize_result(obj) if obj is not None else parse_jsonish(content)
        data["vlm_degraded_retry"] = True
        if data.get("runaway"):
            # Still repeating despite the stronger repetition penalty: the runaway items were already
            # collapsed in normalize_result, the remaining facts are kept (2026-09-10 user decision:
            # occasional noise is preferable to a vague description), confidence drops to low
            print(f"[vlm] still running away after retry; kept {len(data.get('facts') or [])} facts, confidence low", flush=True)
            data["confidence"] = "low"
            data["vlm_runaway"] = True
    data["vlm_cache_hit"] = False
    # A degraded retry (verbatim forced empty) and an all-empty result are not written to the cache:
    # otherwise the same image with the same prompt would never get another attempt at a full
    # transcription -- one service hiccup would freeze this image's description as the crippled
    # version forever.
    degraded = bool(data.get("vlm_degraded_retry"))
    empty = not any(str(data.get(k) or "").strip() for k in ("summary", "title", "text_verbatim")) \
        and not (data.get("facts") or data.get("entities") or data.get("keywords"))
    if cache_json and not degraded and not empty:
        cache_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {**data, "_cache": cache_identity}
        payload.pop("vlm_cache_hit", None)
        cache_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    elif cache_json and (degraded or empty):
        print(f"[vlm] not caching the {'degraded' if degraded else 'empty'} result; retried next time", flush=True)
    return data


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
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_id = {
            executor.submit(
                caption_image,
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
            ): job_id
            for job_id, image_path, prompt, cache_json in jobs
        }
        image_by_id = {job_id: image_path for job_id, image_path, _p, _c in jobs}
        completed = 0
        for future in as_completed(future_to_id):
            job_id = future_to_id[future]
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
            completed += 1
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
