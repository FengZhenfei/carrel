"""A listing keeps literal code separately from its attached source explanation."""
from __future__ import annotations

from typing import Any, Mapping


def evidence_text(text: str, metadata: Mapping[str, Any]) -> str:
    context = str(metadata.get("code_context") or "")
    return context + "\n" + text if context else text


def payload_text(payload: Mapping[str, Any]) -> str:
    return evidence_text(str(payload.get("text") or ""), payload)
