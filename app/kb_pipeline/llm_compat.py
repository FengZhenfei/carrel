"""Small request-parameter policy shared by model probes, graph calls and VLM.

Model IDs remain free-form. Known families supply defaults; exact deployment
overrides cover aliases. Only explicit parameter rejections permit a retry.
The output format is the exception: it steps down (schema, then JSON mode,
then none) on a refusal that names it in any words, and on a refusal that
names nothing at all. A thinking control is given up on a refusal that names
it in any words; the request then runs at the service's own default.
No prompts, credentials or response text are kept in the capability cache.
"""
from __future__ import annotations

import copy
import json
import os
import re
import threading
from typing import Any


_TOKEN_FIELDS = {"max_tokens", "max_completion_tokens"}
_OPTIONAL = {
    "temperature", "top_p", "repetition_penalty", "response_format",
    "enable_thinking", "thinking", "reasoning_effort", "chat_template_kwargs",
}
_MANAGED = _TOKEN_FIELDS | _OPTIONAL
_EXTRA_BODY = {"repetition_penalty", "enable_thinking", "thinking", "chat_template_kwargs"}
_THINKING_FIELDS = {"enable_thinking", "thinking", "reasoning_effort", "chat_template_kwargs"}
MAX_COMPAT_RETRIES = 4
_ACCEPTED: dict[tuple[str, str, str, str], tuple[set[str], dict[str, Any]]] = {}
MAX_FUTILE_GUESSES = 3
_FUTILE: dict[tuple[str, str, str], int] = {}     # endpoint -> failures in a row that followed a guessed step
_LOCK = threading.Lock()


def _overrides(model_id: str, base_url: str, protocol: str) -> list[dict[str, Any]]:
    raw = os.getenv("KB_LLM_COMPAT_OVERRIDES", "").strip()
    if not raw:
        return []
    try:
        entries = json.loads(raw)
    except ValueError as exc:
        raise ValueError("KB_LLM_COMPAT_OVERRIDES must be a JSON array") from exc
    if not isinstance(entries, list):
        raise ValueError("KB_LLM_COMPAT_OVERRIDES must be a JSON array")
    matched = []
    allowed = {"model_id", "base_url", "protocol", "max_tokens_field", "omit", "set", "thinking_parameters"}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) - allowed or not isinstance(entry.get("model_id"), str):
            raise ValueError("Invalid model compatibility override")
        if entry.get("protocol", "openai") not in ("openai", "anthropic"):
            raise ValueError("Compatibility override protocol must be openai or anthropic")
        if "base_url" in entry and not isinstance(entry["base_url"], str):
            raise ValueError("Compatibility override base_url must be a string")
        field = entry.get("max_tokens_field", "max_tokens")
        if not isinstance(field, str) or field not in _TOKEN_FIELDS or (entry.get("protocol") == "anthropic" and field != "max_tokens"):
            raise ValueError("Invalid completion-token field for compatibility override")
        omit, fixed = entry.get("omit", []), entry.get("set", {})
        if not isinstance(omit, list) or any(not isinstance(k, str) or k not in _OPTIONAL for k in omit):
            raise ValueError("Compatibility override omit must list optional generation parameters")
        if not isinstance(fixed, dict) or set(fixed) - (_OPTIONAL - {"response_format"}):
            raise ValueError("Compatibility override set must contain generation parameters")
        thinking = entry.get("thinking_parameters", {})
        if not isinstance(thinking, dict) or set(thinking) - _THINKING_FIELDS:
            raise ValueError("Compatibility override thinking_parameters must contain thinking controls")
        if (entry["model_id"] == model_id and entry.get("protocol", "openai") == protocol
                and ("base_url" not in entry or entry["base_url"].rstrip("/") == base_url.rstrip("/"))):
            matched.append(entry)
    return matched


def generation_parameters(
    model_id: str, *, protocol: str = "openai", base_url: str = "",
    max_tokens: int, temperature: float | None = None, top_p: float | None = None,
    repetition_penalty: float | None = None, disable_thinking: bool = False,
) -> dict[str, Any]:
    """Resolve wire parameters, preserving explicit zero and the output limit."""
    if protocol not in ("openai", "anthropic"):
        raise ValueError("protocol must be 'openai' or 'anthropic'")
    model = model_id.strip().lower().rsplit("/", 1)[-1]
    gpt5 = bool(re.match(r"gpt-5(?:[.-]|$)", model))
    o_series = bool(re.match(r"o[134](?:-|$)", model))
    claude = model.startswith("claude-")
    version = re.search(r"(?:^|-)(\d+)[.-](\d+)(?:-|$)", model) if claude else None
    new_claude = bool(version and tuple(map(int, version.groups())) >= (4, 7))
    kimi = bool(re.match(r"kimi-k2[.-][56](?:-|$)", model))
    params: dict[str, Any] = {"max_tokens": int(max_tokens)}
    if temperature is not None:
        params["temperature"] = float(temperature)
    if top_p is not None:
        params["top_p"] = float(top_p)
    if repetition_penalty is not None:
        params["repetition_penalty"] = float(repetition_penalty)
    if protocol == "openai":
        if gpt5 or o_series:
            params["max_completion_tokens"] = params.pop("max_tokens")
        if disable_thinking:
            if gpt5 and not re.search(r"-(?:pro|chat)(?:-|$)", model):
                # GPT-5.1/5.2/5.4 accept none; original GPT-5 accepts minimal.
                if re.match(r"gpt-5\.(?:1|2|4)(?:-|$)", model):
                    params["reasoning_effort"] = "none"
                elif re.match(r"gpt-5(?:-(?:mini|nano))?(?:-\d{4}-\d{2}-\d{2})?$", model):
                    params["reasoning_effort"] = "minimal"
            elif o_series:
                params["reasoning_effort"] = "low"
            elif model.startswith(("deepseek", "kimi")):
                params["thinking"] = {"type": "disabled"}
            elif model.startswith("qwen"):
                params["enable_thinking"] = False
            elif not (gpt5 or claude or model.startswith("gpt-")):
                # Preserve the existing thinking-off request for unknown compatible hosts.
                params.update(enable_thinking=False, thinking={"type": "disabled"})
        if model.startswith("gpt-") or o_series or claude or kimi:
            params.pop("repetition_penalty", None)
    # Sampling restrictions are model properties, including through a proxy.
    if new_claude or kimi or o_series or (gpt5 and params.get("reasoning_effort") != "none"):
        params.pop("temperature", None)
        params.pop("top_p", None)
    elif protocol == "anthropic" and "temperature" in params:
        params.pop("top_p", None)  # Anthropic accepts at most one sampling control.
    for entry in _overrides(model_id, base_url, protocol):
        if "max_tokens_field" in entry:
            for key in _TOKEN_FIELDS:
                params.pop(key, None)
            params[entry["max_tokens_field"]] = int(max_tokens)
        for key in entry.get("omit", []):
            params.pop(key, None)
        params.update(copy.deepcopy(entry.get("set", {})))
        if disable_thinking and "thinking_parameters" in entry:
            for key in _THINKING_FIELDS:
                params.pop(key, None)
            params.update(copy.deepcopy(entry["thinking_parameters"]))
    return params


def endpoint_url(base_url: str, protocol: str) -> str:
    base = base_url.rstrip("/")
    if protocol == "anthropic":
        if base.endswith("/messages"):
            return base
        return base + ("/messages" if base.endswith("/v1") else "/v1/messages")
    return base if base.endswith("/chat/completions") else base + "/chat/completions"


def sdk_arguments(body: dict[str, Any]) -> dict[str, Any]:
    """Move provider extensions into the OpenAI SDK's extra_body transport slot."""
    result = {k: v for k, v in body.items() if k not in _EXTRA_BODY}
    extra = {k: v for k, v in body.items() if k in _EXTRA_BODY}
    if extra:
        result["extra_body"] = extra
    return result


_NON_PARAMETER_ERROR = re.compile(
    r"api[_ -]?key|unauthorized|authentication|permission|content[_ ]?filter|moderation"
    r"|data_inspection|content exists risk|inappropriate content|context[_ ]length|maximum context"
    r"|too many tokens|prompt is too long|input (?:is )?too long|quota|rate[_ ]limit", re.I,
)
_UNSUPPORTED = re.compile(
    r"unsupported|not supported|does not support|unrecognized|unknown (?:request )?(?:argument|parameter|field)"
    r"|extra(?: fields| inputs)?(?: are)? not permitted|unexpected keyword|not allowed|not permitted"
    r"|only.{0,60}(?:default|supported)|cannot.{0,40}(?:both|specified together)", re.I,
)


def _lesser_format(body: dict[str, Any]) -> dict[str, Any] | None:
    """The same request one step down the output-format ladder: schema-constrained output, then JSON mode
    (parseable JSON, no schema), then no output format. None when nothing is left to give up."""
    if "response_format" not in body:
        return None
    result = copy.deepcopy(body)
    value = result.pop("response_format")
    if isinstance(value, dict) and value.get("type") == "json_schema":
        result["response_format"] = {"type": "json_object"}
    return result


def _adjust(body: dict[str, Any], status: int | None, error: str, protocol: str) -> dict[str, Any] | None:
    if status not in (400, 422) or _NON_PARAMETER_ERROR.search(error):
        return None
    named = {k for k in _MANAGED if re.search(r"(?<![\w])" + re.escape(k) + r"(?![\w])", error, re.I)}
    rejected: set[str] = set()
    if _UNSUPPORTED.search(error):
        # Structured error.param identifies the rejected field more precisely than
        # a message which may also name its replacement.
        try:
            data = json.loads(error)
            detail = data.get("error", data) if isinstance(data, dict) else {}
            param = detail.get("param") if isinstance(detail, dict) else None
        except (ValueError, TypeError):
            param = None
        rejected = {param} if isinstance(param, str) and param in _MANAGED else named
    result = copy.deepcopy(body)
    for key in _TOKEN_FIELDS & rejected & body.keys():
        other = (_TOKEN_FIELDS - {key}).pop()
        # Never remove a token budget or translate a size/range error into a new field.
        if protocol == "openai" and other in named and re.search(r"use|instead|replace", error, re.I):
            result[other] = result.pop(key)
            return result
    for key in (_OPTIONAL - {"response_format"}) & rejected & body.keys():
        result.pop(key, None)
    # A thinking control is given up on a refusal that names it, whatever the wording: services refuse its value
    # ("thinking.type: Input should be 'enabled'", "reasoning_effort must be one of low, medium, high") as readily
    # as the field. Without it the request runs at the service's own default, which is where it stood before the
    # caller asked; a caller for whom that default is not good enough finds out from the answer, not from a 400.
    for key in _THINKING_FIELDS & named & body.keys():
        result.pop(key, None)
    if "response_format" in named and "response_format" in body:
        # A refusal that names the output format is read from any wording, because services word it freely
        # ("This response_format type is unavailable now", "response_format.type: Input should be 'text' or
        # 'json_object'"), and it costs one step, not the parameter: most services without schema-constrained
        # output still have JSON mode. Every other parameter keeps the explicit rule above, because
        # "temperature must be between 0 and 2" is a wrong value, not a refusal.
        result = _lesser_format(result)
    return result if result != body else None


class CompatibleRequest:
    """Bounded parameter retries, with successful adaptations scoped to one request profile."""

    def __init__(self, base_url: str, model_id: str, protocol: str, body: dict[str, Any]) -> None:
        self.protocol = protocol
        self.body = copy.deepcopy(body)
        self._original = self.parameters()
        self._key = (base_url.rstrip("/"), model_id, protocol,
                     json.dumps(self._original, sort_keys=True, separators=(",", ":")))
        with _LOCK:
            removed, changed = _ACCEPTED.get(self._key, (set(), {}))
            for key in removed:
                self.body.pop(key, None)
            self.body.update(copy.deepcopy(changed))
        self._attempts = 0
        self._guessed = False

    def parameters(self) -> dict[str, Any]:
        return {k: copy.deepcopy(v) for k, v in self.body.items() if k in _MANAGED}

    def adapt(self, status: int | None, error: str) -> bool:
        if self._attempts >= MAX_COMPAT_RETRIES:
            return False
        changed = _adjust(self.body, status, error, self.protocol)
        if changed is None and status in (400, 422) and not _NON_PARAMETER_ERROR.search(error):
            # The refusal does not say what was refused ("invalid parameter" and nothing more is common). The
            # output format is the one thing given up on a guess: the answer is parsed and checked whichever
            # way it was asked for, so a lesser format cannot change what it means. No other parameter is
            # touched without being named. An endpoint where such guesses keep failing is left alone.
            with _LOCK:
                futile = _FUTILE.get(self._key[:3], 0) >= MAX_FUTILE_GUESSES
            changed = None if futile else _lesser_format(self.body)
            self._guessed = self._guessed or changed is not None
        if changed is None:
            return False
        self.body = changed
        self._attempts += 1
        return True

    def fail(self) -> None:
        """Call when the request failed for good. A failure that followed a guess counts against guessing at
        this endpoint: its refusals have another cause, and every guess costs requests that cannot succeed."""
        if self._guessed:
            with _LOCK:
                if len(_FUTILE) >= 256:
                    _FUTILE.pop(next(iter(_FUTILE)))
                _FUTILE[self._key[:3]] = _FUTILE.get(self._key[:3], 0) + 1

    def accept(self) -> None:
        """Call only after a valid response, never on a 400 or a failed retry."""
        if self._guessed:
            with _LOCK:
                _FUTILE.pop(self._key[:3], None)     # a guess was right: keep guessing here
        final = self.parameters()
        if final == self._original:
            return
        removed = set(self._original) - set(final)
        changed = {k: v for k, v in final.items() if k not in self._original or v != self._original[k]}
        with _LOCK:
            if len(_ACCEPTED) >= 256:
                _ACCEPTED.pop(next(iter(_ACCEPTED)))
            _ACCEPTED[self._key] = (removed, changed)
