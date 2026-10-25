"""Model-specific label-induction input budgets (tokens, not characters).

The configured capacity is supplied by the operator. The 30% ratio is a policy,
not a model quality guarantee. o200k counts are estimates for other tokenizers;
reserve 20% plus message framing rather than claiming provider-exact counts.
"""
from __future__ import annotations

import math
from typing import Any

from ..utils import count_tokens

INPUT_RATIO = 0.30
TOKEN_MARGIN = 1.20
PROMPT_RESERVE = 8192


def estimate_text(text: str) -> int:
    return math.ceil(count_tokens(text) * TOKEN_MARGIN)


def estimate_messages(messages: list[dict[str, str]]) -> int:
    return 16 + sum(8 + estimate_text(str(m.get("content") or "")) for m in messages)


def sampling_budget(spec: Any) -> dict[str, Any]:
    capacity = getattr(spec, "max_input_tokens", None)
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
        raise ValueError("Set Max Input (tokens) in the model card before extracting labels")
    limit = capacity * 3 // 10
    # Leave room for templates, prior schema, generated definitions and retries.
    reserve = min(PROMPT_RESERVE, limit // 2)
    return {"max_input_tokens": capacity, "input_ratio": INPUT_RATIO,
            "input_token_limit": limit, "sample_token_budget": max(0, limit - reserve),
            "prompt_reserve": reserve, "token_estimator": "o200k_base_or_heuristic_1.2_margin"}
