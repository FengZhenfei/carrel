"""Shared interpretation of graph assertions; source availability is a separate concern."""
from __future__ import annotations

import json
import re
from typing import Any

RELATION_FIELDS = ("polarity", "target_polarity", "modality", "conditions")


def relation_semantics(row: dict[str, Any]) -> dict[str, Any]:
    def choice(name: str, values: set[str], default: str) -> str:
        value = str(row.get(name) or default).strip().lower()
        return value if value in values else "uncertain"

    conditions = row.get("conditions") or row.get("conditions_json") or {}
    if isinstance(conditions, str):
        try:
            conditions = json.loads(conditions)
        except (ValueError, TypeError):
            conditions = {"context": conditions} if conditions.strip() else {}
    if not isinstance(conditions, dict):
        conditions = {}
    return {
        "polarity": choice("polarity", {"affirmed", "negated", "uncertain"}, "affirmed"),
        "target_polarity": choice("target_polarity", {"affirmed", "negated", "uncertain"}, "affirmed"),
        "modality": choice("modality", {"actual", "recommended", "planned", "hypothetical", "uncertain"}, "actual"),
        "conditions": {str(k).strip(): str(v).strip() for k, v in conditions.items() if str(k).strip() and str(v).strip()},
    }


def relation_qualifier(row: dict[str, Any]) -> str:
    """Contradictory or differently qualified assertions must never share an edge ID."""
    value = relation_semantics(row)
    # Ordinary unconditional edges retain the compact key used by deterministic extractors.
    if value == {"polarity": "affirmed", "target_polarity": "affirmed", "modality": "actual", "conditions": {}}:
        return ""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def relation_quality(row: dict[str, Any]) -> str:
    sem = relation_semantics(row)
    if (row.get("derived") or row.get("evidence_kind") == "derived"
            or row.get("type_violation") or row.get("evidence_conflict")
            or row.get("quality") == "needs_review" or row.get("quality_state") == "needs_review"):
        return "needs_review"
    if "uncertain" in (sem["polarity"], sem["target_polarity"], sem["modality"]):
        return "needs_review"
    return "extracted"


def traversable_relation(row: dict[str, Any]) -> bool:
    sem = relation_semantics(row)
    return (relation_quality(row) != "needs_review" and sem["polarity"] == "affirmed"
            and sem["target_polarity"] == "affirmed" and sem["modality"] == "actual" and not sem["conditions"])


def relation_label(row: dict[str, Any]) -> str:
    sem = relation_semantics(row)
    tags = [f"{k}={sem[k]}" for k, default in (("polarity", "affirmed"), ("target_polarity", "affirmed"), ("modality", "actual"))
            if sem[k] != default]
    tags.extend(f"{k}={v}" for k, v in sem["conditions"].items())
    if relation_quality(row) == "needs_review":
        tags.append("needs_review")
    return str(row.get("predicate") or row.get("type") or "related_to") + (" [" + "; ".join(tags) + "]" if tags else "")


def fact_quality(fact: dict[str, Any]) -> str:
    if (fact.get("evidence_conflict") or fact.get("quality") in {"ambiguous_source", "needs_review"}
            or fact.get("subject_origin") == "unresolved"):
        return "needs_review"
    return "extracted"


_NORMAL = re.compile(r"^(?:正常|未见异常|未见明显异常|符合范围|normal|within\s+(?:normal\s+)?range|in\s+range)$", re.I)
_ABNORMAL = re.compile(r"^(?:[↑↓]|[HL]|异常|超标|超出范围|偏高|偏低|增高|降低|abnormal|high|low|out\s+of\s+range|above\s+(?:the\s+)?limit|below\s+(?:the\s+)?limit)$", re.I)


def flag_status(fact: dict[str, Any]) -> str:
    """Interpret an explicit marker or a scalar's stated bounds, never a marker's mere presence."""
    if fact_quality(fact) == "needs_review":
        return "uncertain"
    raw = str(fact.get("flag") or "").strip()
    marked = "normal" if _NORMAL.fullmatch(raw) else "abnormal" if _ABNORMAL.fullmatch(raw) else "unknown"
    from .facts import comparable_number, parse_number

    number = comparable_number(fact)
    lo, hi = parse_number(fact.get("ref_min")), parse_number(fact.get("ref_max"))
    bounded = "unknown"
    if number is not None and (lo is not None or hi is not None) and not (lo is not None and hi is not None and lo > hi):
        bounded = "abnormal" if (lo is not None and number < lo) or (hi is not None and number > hi) else "normal"
    if marked != "unknown" and bounded != "unknown" and marked != bounded:
        return "uncertain"
    return marked if marked != "unknown" else bounded
