"""Qualified facts (structured facts): "the value of some property of X under some condition" from table /
list units.

Fact = (subject, property, symbol?, value | min/typ/max, unit?, conditions{k: v}, note?, source unit).
Domain-independent: product parameters, financial indicators per period and dosages per condition all have
this shape. Triples cannot express conditional values, and folding the value into a property string only hid
the information inside a description; here they are extracted in a separate pass, stored in graph_facts
(cached per unit + fingerprint, like extraction results), written to Qdrant's spec collection and Neo4j's
Spec nodes, so recall can hit facts directly.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, replace
from typing import Any, Iterable

from . import prompts
from .extract import normalize_name, strip_math
from .llm import ChatClient, LLMSpec
from .tabletext import FACTS_EXPAND_WIDE_TABLES, expand_wide_tables
from .temporal import axis_value_kind
from .units import Unit, unit_signals

FACTS_MAX_PER_UNIT = 120
FACTS_TABLE_RATIO = 0.3
# A big table unit can yield hundreds of facts and 6144 cut the JSON midway (observed in a large-table extraction); after a
# truncation another request follows at twice the budget
FACTS_MAX_TOKENS = 12288
_NUMBER_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?(?:\s*[eE][-+]?\d+)?")
# Value field classification (Codex review F01): only a string that parses as one number in its entirety is a
# scalar; relative values with variables (V_CC + 0.5) are never evaluated. "Find the first number" used to
# record it as 0.5, and V_SS - 0.5 even as a positive 0.5.
_UNIT_CLASS = r"%°℃℉µμΩ℧A-Za-z/·"
_FOOTNOTE_RE = re.compile(r"\s*(?:\[\s*\d{1,3}\s*\]|\(\s*\d{1,2}\s*\)|（\s*\d{1,2}\s*）)+\s*$")   # trailing footnote markers [6], (4)
# a comma is a thousands separator only in groups of three (1,024, 12,345.6); "12,50" and "2,86" are a decimal
# comma or two comma-separated values, and dropping the comma to read 1250 or 286 would change the number, so
# such a string does not count as one number
_PLAIN_NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_SCALAR_RE = re.compile(
    rf"^(?P<cmp><=|>=|[<>≤≥~≈±])?\s*(?P<num>{_PLAIN_NUMBER}(?:\s*[eE][-+]?\d+)?)"
    rf"\s*(?P<unit>[{_UNIT_CLASS}]*)(?:\s*[\(（][^\)）]{{0,40}}[\)）])?$"
)
_RANGE_RE = re.compile(
    rf"^(?P<lo>{_PLAIN_NUMBER})\s*[{_UNIT_CLASS}]*\s*(?:to|~|–|—|-|至|到|\.\.\.?)\s*"
    rf"(?P<hi>{_PLAIN_NUMBER})\s*[{_UNIT_CLASS}]*$"
)
# A currency sign directly in front of a number ($85,000, A$21,050) is its unit, not part of the number: the value
# stays as written and the number is parsed as usual, so amounts can be compared and put in a series. When letters
# follow ($85K, $1.05M) they abbreviate a magnitude, 85 is not the number, and the value remains text
_CURRENCY = r"(?:[A-Z]{1,3})?[$¥€£₩₹]"
_MONEY_RE = re.compile(
    rf"^(?P<cmp><=|>=|[<>≤≥~≈±])?\s*(?P<cur>{_CURRENCY})\s*(?P<num>{_PLAIN_NUMBER})(?:\s*[\(（][^\)）]{{0,40}}[\)）])?$"
)
_MONEY_RANGE_RE = re.compile(
    rf"^(?P<cur>{_CURRENCY})\s*(?P<lo>{_PLAIN_NUMBER})\s*(?:to|~|–|—|-|至|到)\s*(?:{_CURRENCY})?\s*(?P<hi>{_PLAIN_NUMBER})$"
)
_EXPR_RE = re.compile(r"[A-Za-z_}\)]\s*[+\-×x\*/]\s*[\d(]|[\d\)]\s*[+\-×x\*/]\s*[A-Za-z_({\\]")
_IDENT_RE = re.compile(r"[A-Za-z]+_[A-Za-z0-9{}]+|\\[A-Za-z]+|[A-Za-z]+\{")
_COMPOUND_SPLIT_RE = re.compile(r"\s*(?:/|,|;|、|\band\b|与|和)\s*")
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


@dataclass
class FactResult:
    unit_id: str
    facts: list[dict[str, Any]]
    calls: int
    stats: dict[str, Any]


def wants_facts(unit: Unit, kind: str | None = None) -> bool:
    """Which units get structured extraction: those containing a table block, or with a high enough table-row
    ratio, or the document's own conclusion sections (abnormal result lists, conclusions, recommendations: the
    source of the view layer's "conclusion facts"); boilerplate units are skipped."""
    effective = kind or unit.kind or "body"
    if effective == "boilerplate":
        return False
    if effective == "conclusion":
        return True
    if any(str(b).lower() == "table" for b in (unit.block_types or [])):
        return True
    return unit_signals(unit.text, [unit.section_label])["table_ratio"] >= FACTS_TABLE_RATIO


from ..measure_units import _UNIT_ALIASES, canonical_unit, unit_key  # noqa: E402,F401  unit table + canonical forms (final review F04: separate module shared by parser and graph sides)


def _to_float(raw: str) -> float | None:
    try:
        return float(raw.replace(",", "").replace(" ", ""))
    except ValueError:
        return None


_SUPERSCRIPT_RE = re.compile(r"(?<=\d)[⁰¹²³⁴-⁹⁺⁻]+")


def _nfkc(text: str) -> str:
    """Full-width / half-width normalisation. A superscript after a digit is first written as ^n: NFKC would
    flatten 10 with a superscript 9 into 109, which is another number."""
    text = _SUPERSCRIPT_RE.sub(lambda m: "^" + unicodedata.normalize("NFKC", m.group(0)).replace("−", "-"), text)
    return unicodedata.normalize("NFKC", text)


def classify_value(value: Any) -> dict[str, Any]:
    """What kind of thing a value field is:
      scalar     the whole string is one number (comparator, unit, footnote, bracketed remark allowed): '0.160',
                 '-40°C', '≤ 10', '10 mA (typ)'; also a number with a currency sign directly in front of it
                 ('$85,000', reported with currency)
      range      two numbers around a range marker: '2.7 to 3.6', '-40 ~ +85', '$310,000-$485,000'
      expression a relative value with variables / arithmetic: 'V_CC + 0.5', '0.8 × V_DD', 'VCC/2'; cannot be
                 evaluated without variable bindings
      text       anything else ('n/a', 'Max', '20/25/45', '12,50')
      empty
    Only scalar yields num; range yields lo / hi."""
    text = _nfkc(str(value or "")).replace("−", "-").strip()
    text = _FOOTNOTE_RE.sub("", text).strip()
    if not text:
        return {"kind": "empty", "num": None}
    m = _SCALAR_RE.fullmatch(text)
    if m:
        num = _to_float(m.group("num"))
        if num is not None:
            out: dict[str, Any] = {"kind": "scalar", "num": num}
            if m.group("cmp"):
                out["cmp"] = m.group("cmp")
            return out
    m = _RANGE_RE.fullmatch(text)
    if m:
        lo, hi = _to_float(m.group("lo")), _to_float(m.group("hi"))
        if lo is not None and hi is not None:
            return {"kind": "range", "num": None, "lo": lo, "hi": hi}
    m = _MONEY_RE.fullmatch(text)
    if m:
        num = _to_float(m.group("num"))
        if num is not None:
            out = {"kind": "scalar", "num": num, "currency": m.group("cur")}
            if m.group("cmp"):
                out["cmp"] = m.group("cmp")
            return out
    m = _MONEY_RANGE_RE.fullmatch(text)
    if m:
        lo, hi = _to_float(m.group("lo")), _to_float(m.group("hi"))
        if lo is not None and hi is not None:
            return {"kind": "range", "num": None, "lo": lo, "hi": hi, "currency": m.group("cur")}
    if _NUMBER_RE.search(text) and (_EXPR_RE.search(text) or _IDENT_RE.search(text)):
        return {"kind": "expression", "num": None}
    return {"kind": "text", "num": None}


def parse_number(value: Any) -> float | None:
    """'0.160' → 0.16; '1,024' → 1024; '-40°C' → -40; non-scalars (expressions, ranges, text) return None."""
    return classify_value(value)["num"]


def _complete_json_prefix(raw: str) -> str | None:
    """JSON cut off by max_tokens is a valid prefix: close the unterminated string and brackets, and every object
    that was fully generated can be recovered. Returns None when the prefix is not repairable. (Same logic as
    vision.vlm.complete_json_prefix; the graph build does not depend on the vision module.)"""
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
        return raw
    fixed = raw[:-1] if esc else raw
    if in_str:
        fixed += '"'
    tail = fixed.rstrip()
    if tail.endswith(":"):
        cut = max(tail.rfind(","), tail.rfind("{"))
        if cut >= 0:
            tail = tail[: cut + 1]
    if tail.endswith(","):
        tail = tail[:-1]
    fixed = tail + "".join(reversed(stack))
    try:
        json.loads(fixed)
        return fixed
    except ValueError:
        # The last object is half written (a key without a value): back off to the comma after the previous
        # complete object and recompute the brackets for the new prefix
        cut = tail.rfind("},")
        if cut < 0:
            return None
        return _complete_json_prefix(tail[: cut + 1])


_BAD_ESCAPE_RE = re.compile(r'\\(?!["\\/bfnrtu])')     # a backslash not followed by a JSON escape character: \| \_ \% (markdown / SQL spellings)


def _strip_trailing_commas(text: str) -> str:
    """Remove commas right before } / ], touching only text outside strings (Codex 2026-09-14 R04: the old
    whole-text regex turned a,]b inside a string into a]b)."""
    out: list[str] = []
    in_str = esc = False
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
            out.append(ch)
        elif ch == ",":
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j < n and text[j] in "}]":
                i = j
                continue
            out.append(ch)
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def _lenient_json(text: str) -> Any:
    """Parse the model's JSON leniently: as-is first; failing that, double the invalid escapes and retry; failing
    that, strip trailing commas outside strings. Control characters inside strings are always allowed, and string
    contents are never altered. On 2026-09-13 one sampled unit hit six "malformed
    response" errors in a row: the model wrote the pipes of a table as "backslash + pipe", json raised Invalid
    escape and the whole build failed. That is spelling noise and must not let one unit sink the whole graph.
    Raises ValueError when it still cannot be parsed."""
    try:
        return json.loads(text, strict=False)
    except ValueError:
        pass
    fixed = _BAD_ESCAPE_RE.sub(r"\\\\", text)
    try:
        return json.loads(fixed, strict=False)
    except ValueError:
        pass
    return json.loads(_strip_trailing_commas(fixed), strict=False)


def parse_facts_response(text: str) -> dict[str, Any]:
    """Model output → {facts, malformed, truncated}. Accepts fences, surrounding prose, invalid escapes and control
    characters; facts not being a list counts as malformed; a truncated JSON prefix recovers the complete objects
    and records truncated (distinguishing "valid empty facts / invalid JSON / truncated", Codex review F03)."""
    body = _FENCE_RE.sub("", str(text or "")).strip()
    start = body.find("{")
    if start < 0:
        return {"facts": [], "malformed": 1, "truncated": False}
    end = body.rfind("}")
    data: Any = None
    if end > start:
        try:
            data = _lenient_json(body[start:end + 1])
        except ValueError:
            data = None
    truncated = False
    if data is None:
        tail = _BAD_ESCAPE_RE.sub(r"\\\\", body[start:])
        fixed = _complete_json_prefix(tail)
        if fixed is None:
            return {"facts": [], "malformed": 1, "truncated": False}
        try:
            data = json.loads(fixed, strict=False)
        except ValueError:
            return {"facts": [], "malformed": 1, "truncated": False}
        truncated = fixed != tail
    facts = data.get("facts") if isinstance(data, dict) else data
    if not isinstance(facts, list):
        return {"facts": [], "malformed": 1, "truncated": False}
    return {"facts": [f for f in facts if isinstance(f, dict)], "malformed": 0, "truncated": truncated}


def parse_facts_json(text: str) -> tuple[list[dict[str, Any]], int]:
    """Legacy signature: (facts, malformed)."""
    parsed = parse_facts_response(text)
    return parsed["facts"], parsed["malformed"]


def _clean(value: Any, limit: int = 200) -> str:
    """Subject, property, symbol: normalised the same way as entity names, so facts can be matched to entities
    by name."""
    return normalize_name(str(value if value is not None else ""))[:limit]


def _clean_value(value: Any, limit: int = 200) -> str:
    """Value, unit, conditions, note: full-width / half-width normalisation, surrounding quotes and whitespace
    stripped, whitespace collapsed, everything else as in the source.
    These fields are not names and skip the LaTeX restoration of entity names ("$85K" would lose its currency,
    "${HOME}_dir" would become HOMEdir); only recognisable $...$ fragments are restored (see strip_math)."""
    text = _nfkc(strip_math(str(value if value is not None else "")))
    text = text.strip().strip("\"'`“”‘’").strip()
    return re.sub(r"\s+", " ", text)[:limit]


def normalize_fact(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Canonical form of one fact: field trimming, condition table cleanup, number parsing; facts missing subject /
    property or without any value are dropped."""
    subject = _clean(raw.get("subject"), 160)
    prop = _clean(raw.get("property"), 160)
    symbol = _clean(raw.get("symbol"), 60)
    value = _clean_value(raw.get("value"), 120)
    lo, typ, hi = _clean_value(raw.get("min"), 60), _clean_value(raw.get("typ"), 60), _clean_value(raw.get("max"), 60)
    unit = _clean_value(raw.get("unit"), 40)
    note = _clean_value(raw.get("note"), 300)
    flag = _clean_value(raw.get("flag"), 40)
    ref_min, ref_max = _clean_value(raw.get("ref_min"), 60), _clean_value(raw.get("ref_max"), 60)
    period_text = _clean_value(raw.get("period_text"), 120)
    if not subject or not prop:
        return None
    if not any((value, lo, typ, hi)):
        return None
    conditions: dict[str, str] = {}
    raw_conditions = raw.get("conditions")
    if isinstance(raw_conditions, dict):
        for k, v in list(raw_conditions.items())[:8]:
            ck, cv = _clean_value(k, 60), _clean_value(v, 80)
            if ck and cv:
                conditions[ck] = cv
    elif isinstance(raw_conditions, str) and raw_conditions.strip():
        conditions["condition"] = _clean_value(raw_conditions, 120)
    kinds: dict[str, str] = {}
    nums: dict[str, float | None] = {}
    ranges: dict[str, list[float]] = {}
    cmps: dict[str, str] = {}
    for field, text in (("value", value), ("min", lo), ("typ", typ), ("max", hi)):
        info = classify_value(text)
        nums[field] = info["num"]
        if info.get("currency") and not unit:
            unit = str(info["currency"])              # empty unit field and the currency written in front of the number: the currency is the unit
        if info["kind"] != "empty":
            kinds[field] = info["kind"]
        if info["kind"] == "range":
            ranges[field] = [info["lo"], info["hi"]]
        if info.get("cmp"):
            cmps[field] = str(info["cmp"])            # the comparator of "<1.3" is part of the value (Codex review F04), keep it
    out = {
        "subject": subject, "property": prop, "symbol": symbol,
        "value": value, "min": lo, "typ": typ, "max": hi, "unit": unit, "unit_canonical": canonical_unit(unit),
        "value_num": nums["value"], "min_num": nums["min"], "typ_num": nums["typ"], "max_num": nums["max"],
        "kinds": kinds,
        "conditions": conditions, "note": note,
    }
    if ranges:
        out["ranges"] = ranges
    if cmps:
        out["cmps"] = cmps
    # Axis and bounds (2026-09-07): the document's own abnormal flags, reference ranges and validity period. Dates
    # are only accepted in the ISO form the model copied from the source (temporal.parse_date normalizes once
    # more); a missing validity period is filled in by the build phase from the document axis value
    if flag:
        out["flag"] = flag
    if ref_min or ref_max:
        out["ref_min"], out["ref_max"] = ref_min, ref_max
        out["ref_min_num"], out["ref_max_num"] = parse_number(ref_min), parse_number(ref_max)
    from .temporal import parse_date
    for field in ("valid_from", "valid_until"):
        text = _clean_value(raw.get(field), 40)
        parsed = parse_date(text) if text else None
        if parsed:
            out[field] = parsed
    if period_text and (out.get("valid_from") or out.get("valid_until")):
        out["period_text"] = period_text
    return out


def bound_distance(fact: dict[str, Any]) -> float | None:
    """Relative distance of a scalar value from the reference range / spec limits: 0 inside the range, otherwise
    the overshoot / range width (1 when there is no width); None without a scalar value or without bounds. The
    view layer and the extension rules use it to find "close to the limit" items."""
    num = fact.get("value_num")
    if num is None:
        num = fact.get("typ_num")
    if num is None:
        return None
    lo, hi = fact.get("ref_min_num"), fact.get("ref_max_num")
    if lo is None and hi is None:
        return None
    width = (hi - lo) if (lo is not None and hi is not None and hi > lo) else None
    over = 0.0
    if lo is not None and num < lo:
        over = lo - num
    elif hi is not None and num > hi:
        over = num - hi
    if over == 0.0:
        return 0.0
    return round(over / width, 4) if width else 1.0


def _squash(text: Any) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def comparable_number(fact: dict[str, Any]) -> float | None:
    """The value used for trends / numeric comparison: only a scalar without comparator counts (value, then typ);
    ranges, expressions, comparator values (<1.3), text, multi-values (20/25/45) and facts flagged as evidence
    conflicts or glued sources all return None. The compile layer no longer guesses numbers out of strings (Codex
    review F04: V_CC + 0.5 was once taken as 0.5, 2.7 to 3.6 as 2.7, and <1.3 treated as equal to 1.3)."""
    if fact.get("evidence_conflict") or str(fact.get("quality") or "") == "ambiguous_source":
        return None
    for field in ("value", "typ"):
        raw = fact.get(field)
        if raw in (None, ""):
            continue
        info = classify_value(raw)
        if info["kind"] == "scalar" and not info.get("cmp"):
            return info["num"]
        return None
    return None


def mark_evidence_conflict(fact: dict[str, Any], conflicts: list[dict[str, Any]]) -> bool:
    """Indicators where the in-image text and the model's reading conflicted in this unit (the chunk's
    visual_value_conflicts): when a fact's property matches, mark evidence_conflict, lower confidence to low and
    keep the number out of trend and conflict judgements, retaining the source text and both values (Codex review
    F01). Property and conflict label match when equal after whitespace removal and casefold, or when one contains
    the other (at least 2 characters)."""
    prop = _squash(fact.get("property"))
    if not prop:
        return False
    for c in conflicts or []:
        label = _squash(c.get("label"))
        if not label:
            continue
        shorter = min(label, prop, key=len)
        if label == prop or (len(shorter) >= 2 and (label in prop or prop in label)):
            fact["evidence_conflict"] = {"label": str(c.get("label") or ""), "text_value": str(c.get("text_value") or ""),
                                         "model_value": str(c.get("model_value") or "")}
            fact["confidence"] = "low"
            return True
    return False


def apply_unit_quality(facts: list[dict[str, Any]], unit: Any) -> dict[str, int]:
    """Apply quality flags to facts from the unit's current evidence metadata: glued source strings in tables that
    were split without verification → ambiguous_source, conflicts between in-image text and the model's reading →
    evidence_conflict. Done once at extraction and again when loading from the facts cache: the unit id is
    addressed by text and position only, so after a re-parse that leaves the text unchanged but changes this
    metadata, cached facts would not know by themselves (Codex re-review N04). Flags are only added, never
    withdrawn: withdrawing could not restore the confidence the model originally gave."""
    glued = [str(v) for v in (getattr(unit, "ambiguous_values", None) or []) if str(v).strip()]
    conflicts = [c for c in (getattr(unit, "value_conflicts", None) or []) if isinstance(c, dict)]
    out = {"ambiguous": 0, "evidence_conflicts": 0}
    for fact in facts:
        if glued and mark_ambiguous_source(fact, glued):
            out["ambiguous"] += 1
        if conflicts and mark_evidence_conflict(fact, conflicts):
            out["evidence_conflicts"] += 1
    return out


def split_unit_text(text: str, *, min_chars: int = 200) -> list[str] | None:
    """Split the unit text in two at a paragraph boundary near the middle (by newline when there is no blank
    line); returns None when either half is too short. Used for the split-in-half retry when the facts response
    was truncated twice (Codex review F10): the unit is too long for the model to finish in one go, the model is
    not broken."""
    body = str(text or "")
    if len(body) < min_chars * 2:
        return None
    mid = len(body) // 2
    for sep in ("\n\n", "\n", "。", ". "):
        left = body.rfind(sep, 0, mid)
        right = body.find(sep, mid)
        cut = max(left, right) if left < 0 or right < 0 else (left if mid - left <= right - mid else right)
        if cut > 0:
            a, b = body[:cut + len(sep)].strip(), body[cut + len(sep):].strip()
            if len(a) >= min_chars and len(b) >= min_chars:
                return [a, b]
    return None


def mark_ambiguous_source(fact: dict[str, Any], glued: list[str]) -> bool:
    """A value hits a glued string from a table that was split without verification (757557): clear the numeric
    fields, record the field kind as ambiguous and flag the fact quality=ambiguous_source. The source text stays
    as evidence but must not take part in numeric filtering (F02)."""
    squashed = {_squash(g) for g in glued if _squash(g)}
    if not squashed:
        return False
    hit = False
    for field in ("value", "min", "typ", "max"):
        text = _squash(fact.get(field))
        if text and any(g in text for g in squashed):
            fact[f"{field}_num"] = None
            fact.setdefault("kinds", {})[field] = "ambiguous"
            ranges = fact.get("ranges")
            if isinstance(ranges, dict):
                ranges.pop(field, None)
                if not ranges:
                    fact.pop("ranges", None)
            hit = True
    if hit:
        fact["quality"] = "ambiguous_source"
    return hit


def fact_id(unit_id: str, fact: dict[str, Any]) -> str:
    # The unit is part of the identity too: two facts identical except for the unit (10 ns / 10 ms) used to
    # collide on one ID, and the later one was dropped as a duplicate (Codex review F08)
    raw = "|".join([
        str(unit_id), fact.get("subject", ""), fact.get("property", ""), fact.get("symbol", ""),
        json.dumps(fact.get("conditions") or {}, ensure_ascii=False, sort_keys=True),
        fact.get("value", ""), fact.get("min", ""), fact.get("typ", ""), fact.get("max", ""),
        str(fact.get("unit", "")).casefold().strip(),
        str(fact.get("valid_from") or ""),        # the same property split by period in one table (2024 / 2025 columns) is two facts
    ])
    return "f" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def when_text(fact: dict[str, Any]) -> str:
    """Readable form of the time / version a fact applies to (validity period or document axis value)."""
    lo, hi = str(fact.get("valid_from") or ""), str(fact.get("valid_until") or "")
    if lo and hi and lo != hi:
        return f"{lo}~{hi}"
    return lo or hi or str(fact.get("axis") or "")


def conditions_text(conditions: dict[str, Any] | None) -> str:
    return "; ".join(f"{k}: {v}" for k, v in (conditions or {}).items())


_SYMBOL_ONLY_RE = re.compile(r"^[\s\-—–―/\\_.·•~～＿－]+$")


def symbol_only_value(fact: dict[str, Any]) -> bool:
    """A fact whose value is only a symbol ("-", "—", "/") or N/A: a table placeholder, "not applicable" or "none",
    with a meaning that depends on the document. Such facts stay in the facts layer for retrieval but are kept out
    of the subject page's narration input: the summary model reads a lone dash as a positive finding (a spot check
    found an assessment marked "-" for both eyes narrated as "shows ... abnormality")."""
    value = str(fact.get("value") or "").strip()
    if not value:
        return False
    return bool(_SYMBOL_ONLY_RE.match(value)) or value.casefold() in {"n/a", "na", "n.a.", "none", "null"}


def values_text(fact: dict[str, Any]) -> str:
    """Readable form of the values: a single value or min / typ / max, with unit."""
    unit = str(fact.get("unit") or "")
    if fact.get("value"):
        return f"{fact['value']} {unit}".strip()
    parts = [f"{label} {fact[key]}" for key, label in (("min", "min"), ("typ", "typ"), ("max", "max")) if fact.get(key)]
    return (" / ".join(parts) + (f" {unit}" if unit else "")).strip()


def spec_text(fact: dict[str, Any]) -> str:
    """One line for embedding: subject · property (symbol): value | conditions."""
    head = f"{fact.get('subject')} · {fact.get('property')}"
    if fact.get("symbol") and str(fact["symbol"]).casefold() != str(fact.get("property") or "").casefold():
        head += f" ({fact['symbol']})"
    text = f"{head}: {values_text(fact)}"
    if fact.get("flag"):
        text += f" {fact['flag']}"
    cond = conditions_text(fact.get("conditions"))
    if cond:
        text += f" | {cond}"
    when = when_text(fact)
    if when:
        text += f" | when: {when}"
    if fact.get("ref_min") or fact.get("ref_max"):
        text += f" | ref: {fact.get('ref_min') or ''}~{fact.get('ref_max') or ''}"
    if fact.get("note"):
        text += f" | {fact['note']}"
    return text[:1000]


def property_text(fact: dict[str, Any]) -> str:
    """Property-side embedding text (OG-RAG: property name and value are embedded separately): concept name /
    property / symbol / subject."""
    parts = [str(fact.get("concept") or ""), str(fact.get("property") or ""), str(fact.get("symbol") or ""),
             str(fact.get("subject") or "")]
    return " · ".join(dict.fromkeys(p for p in parts if p))[:400]


def value_text(fact: dict[str, Any]) -> str:
    """Value-side embedding text: value, unit, flag, conditions, time."""
    text = values_text(fact)
    if fact.get("flag"):
        text += f" {fact['flag']}"
    cond = conditions_text(fact.get("conditions"))
    if cond:
        text += f" | {cond}"
    when = when_text(fact)
    if when:
        text += f" | {when}"
    return text[:400] or "(no value)"


def render_facts_prompt(unit: Unit, *, document: str, subjects: Iterable[str], max_facts: int = FACTS_MAX_PER_UNIT,
                        axis: str = "", narrative: bool = False) -> str:
    """表格 / 列表单元用 FACTS_PROMPT;narrative=True(按日期组织的资料里的叙述段)用 NARRATIVE_FACTS_PROMPT。
    两段提示词都带文档日期:相对日期、缺年份的日期按它换算。"""
    from .temporal import axis_value_kind

    subject_text = ", ".join(str(s) for s in subjects if str(s).strip()) or "(unknown)"
    axis_value = axis or getattr(unit, "axis", "") or ""
    doc_date = str(getattr(unit, "doc_date", "") or "") or (axis_value if axis_value_kind(axis_value) == "date" else "")
    doc_version = axis_value if axis_value_kind(axis_value) == "version" else ""
    if narrative:
        return prompts.NARRATIVE_FACTS_PROMPT.format(
            document=document or unit.rel_path or "(unknown)", section=unit.section_label,
            document_date=doc_date or "(unknown)", subjects=subject_text, input_text=unit.text, max_facts=int(max_facts))
    return prompts.FACTS_PROMPT.format(
        document=document or unit.rel_path or "(unknown)", section=unit.section_label,
        document_date=doc_date or "(unknown)", document_version=doc_version or "(unknown)",
        subjects=subject_text, input_text=((expand_wide_tables(unit.text) if FACTS_EXPAND_WIDE_TABLES else None) or unit.text),
        max_facts=int(max_facts),
    )


def facts_prompt_hash() -> str:
    return hashlib.sha256((prompts.FACTS_PROMPT + prompts.NARRATIVE_FACTS_PROMPT).encode("utf-8")).hexdigest()[:16]


NARRATIVE_FACTS_MIN_TOKENS = 40


def wants_narrative_facts(unit: Unit, kind: str | None = None) -> bool:
    """按日期组织的资料(画像 axis = date)里,不走表格事实抽取的叙述段也抽一遍状态事实:负责人、进度、截止日期、
    决定、叙述里给出的数值。样板与罗列不抽,太短的段不抽。是否开由调用方按画像判。"""
    effective = kind or unit.kind or "body"
    if effective in ("boilerplate", "listing"):
        return False
    if wants_facts(unit, kind):
        return False
    return int(unit.n_tokens or 0) >= NARRATIVE_FACTS_MIN_TOKENS


# 候选主体怎么挑也进指纹:候选写在提示词的 Likely subjects 行,规则变了,按旧候选抽出的单元不能沿用(提示词没变的
# 单元重抽时走响应缓存,不花调用)。subjects-v2:小节标题点名的主题(改版十八)
SUBJECT_CANDIDATES_VERSION = "subjects-v2"


def facts_fingerprint(spec: LLMSpec) -> str:
    # 版本标签跟着 normalize_fact 的产出走:落库的是规范化之后的事实,规范化变了旧行就不能复用。
    # facts-v4:逗号只认千分位、取值类字段不再按实体名还原 LaTeX;facts-v5:有效期只收日期(版本在 axis 里)、带观察时间
    raw = "|".join(["facts-v5", SUBJECT_CANDIDATES_VERSION, spec.model_id, spec.protocol, facts_prompt_hash(), str(FACTS_MAX_PER_UNIT)])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def facts_response_ok(text: str) -> bool:
    """For ChatClient.chat(validate=…): a response is acceptable only if a facts list can be parsed from it
    (complete, or truncated but recoverable)."""
    return parse_facts_response(text)["malformed"] == 0


class FactExtractor:
    """One call per unit, producing that unit's qualified facts."""

    def __init__(self, client: ChatClient, *, max_facts: int = FACTS_MAX_PER_UNIT, max_tokens: int = FACTS_MAX_TOKENS) -> None:
        self.client = client
        self.max_facts = int(max_facts)
        self.max_tokens = int(max_tokens)

    @property
    def fingerprint(self) -> str:
        return facts_fingerprint(self.client.spec)

    def extract(self, unit: Unit, *, document: str = "", subjects: Iterable[str] = (), axis: str = "",
                narrative: bool = False) -> FactResult:
        prompt = render_facts_prompt(unit, document=document, subjects=subjects, max_facts=self.max_facts, axis=axis,
                                     narrative=narrative)
        # 不是合格 JSON 的响应:客户端带纠正提示再要一次,仍不合格就抛 LLMMalformedResponse ——
        # 调用方把这个单元记成失败,不落 graph_facts、不进响应缓存(Codex 审查 F03)
        response = self.client.chat([{"role": "user", "content": prompt}], max_tokens=self.max_tokens,
                                    validate=facts_response_ok)
        parsed = parse_facts_response(response)
        calls = 1
        split = 0
        if parsed["truncated"]:
            # Cut off by max_tokens: ask again at twice the budget; if still truncated keep the recovered part and
            # record truncated
            bigger = self.client.chat([{"role": "user", "content": prompt}], max_tokens=self.max_tokens * 2,
                                      validate=facts_response_ok)
            calls += 1
            again = parse_facts_response(bigger)
            if not again["truncated"] or len(again["facts"]) > len(parsed["facts"]):
                parsed = again
        if parsed["truncated"]:
            # Incomplete both times: split the unit in half, extract each and merge the facts (Codex review F10).
            # Still truncated is recorded as partial: the build completes as usual, but this unit shows as "facts
            # incomplete" in the stats and on the status card instead of pretending full success
            halves = split_unit_text(unit.text)
            if halves:
                merged: list[dict[str, Any]] = []
                still = False
                for part in halves:
                    sub = replace(unit, text=part)
                    sub_prompt = render_facts_prompt(sub, document=document, subjects=subjects, max_facts=self.max_facts, axis=axis,
                                                     narrative=narrative)
                    got = parse_facts_response(self.client.chat([{"role": "user", "content": sub_prompt}], max_tokens=self.max_tokens * 2,
                                                                validate=facts_response_ok))
                    calls += 1
                    merged.extend(got["facts"])
                    still = still or bool(got["truncated"])
                parsed = {"facts": merged, "malformed": parsed["malformed"], "truncated": still}
                split = 1
        capped = False
        raw_facts, malformed = parsed["facts"], parsed["malformed"]
        facts: list[dict[str, Any]] = []
        seen: set[str] = set()
        dropped = 0
        ambiguous = 0
        conflicted = 0
        glued = [str(v) for v in (getattr(unit, "ambiguous_values", None) or []) if str(v).strip()]
        value_conflicts = [c for c in (getattr(unit, "value_conflicts", None) or []) if isinstance(c, dict)]
        for i, raw in enumerate(raw_facts[: self.max_facts * 2]):
            fact = normalize_fact(raw)
            if fact is None:
                dropped += 1
                continue
            if glued and mark_ambiguous_source(fact, glued):
                ambiguous += 1
            if value_conflicts and mark_evidence_conflict(fact, value_conflicts):
                conflicted += 1
            fid = fact_id(unit.unit_id, fact)
            if fid in seen:
                dropped += 1
                continue
            seen.add(fid)
            fact["id"] = fid
            fact["unit_id"] = unit.unit_id
            if narrative:
                fact["source_kind"] = "narrative"
            facts.append(fact)
            if len(facts) >= self.max_facts:
                capped = i + 1 < len(raw_facts)      # some left: stopping at the budget is not "input fully processed" (Codex re-review N03)
                break
        partial = int(bool(parsed["truncated"]) or capped)
        return FactResult(unit_id=unit.unit_id, facts=facts, calls=calls,
                          stats={"facts": len(facts), "dropped": dropped, "malformed_json": malformed, "raw": len(raw_facts),
                                 "ambiguous": ambiguous, "evidence_conflicts": conflicted,
                                 "truncated": int(bool(parsed["truncated"])), "split": split, "capped": int(capped),
                                 "partial": partial})


def link_facts(facts: list[dict[str, Any]], entities: list[dict[str, Any]], *, units_by_id: dict[str, Unit]) -> dict[str, int]:
    """把事实的 subject / property 对到图里的实体(先本文档作用域,再全局;标题或别名相等,大小写不敏感)。
    对上的写 subject_key / property_key;对不上保留文本。返回计数。"""
    by_title: dict[tuple[str, str], set[str]] = {}
    by_ref: dict[tuple[str, str], set[str]] = {}
    for e in entities:
        scope = str(e.get("scope") or "")
        names = [str(e.get("title") or "")] + [str(a) for a in (e.get("aliases") or [])]
        for n in names:
            n = n.casefold().strip()
            if n:
                by_title.setdefault((scope, n), set()).add(e["key"])
                for ext in e.get("external_ids") or []:
                    ref = f"{ext.get('scheme') or ''}:{ext.get('value') or ''}"
                    by_ref.setdefault((n, ref), set()).add(e["key"])
    stats = {"subjects_linked": 0, "properties_linked": 0, "compound_subjects": 0}

    def lookup(doc: str, name: str) -> str:
        name = name.casefold().strip()
        found = by_title.get((doc, name)) or by_title.get(("", name)) or set()
        return next(iter(found)) if len(found) == 1 else ""

    for f in facts:
        unit = units_by_id.get(str(f.get("unit_id") or ""))
        doc = unit.doc_id if unit else ""
        for field, key_field, counter in (("subject", "subject_key", "subjects_linked"), ("property", "property_key", "properties_linked")):
            name = str(f.get(field) or "")
            ref = str(f.get("subject_ref") or "") if field == "subject" else ""
            matches = by_ref.get((name.casefold().strip(), ref), set()) if ref else set()
            key = (next(iter(matches)) if len(matches) == 1 else "") if ref else lookup(doc, name)
            f.pop(key_field, None)
            if not key and f.get("symbol") and field == "property":
                key = lookup(doc, str(f["symbol"]))
            if not key and field == "subject" and not ref:
                # 合写主体「A, B」「A/B」:一条事实同时属于几个东西,每个都挂上
                pieces = [x.strip() for x in _COMPOUND_SPLIT_RE.split(name) if x.strip()]
                keys = [k for k in (lookup(doc, x) for x in pieces) if k] if len(pieces) >= 2 else []
                if keys:
                    key = keys[0]
                    f["subject_keys"] = list(dict.fromkeys(keys))
                    stats["compound_subjects"] += 1
            if key:
                f[key_field] = key
                stats[counter] += 1
        if unit is not None:
            f["doc_id"] = unit.doc_id
            f["rel_path"] = unit.rel_path
            f["section"] = unit.section_label
            f["point_ids"] = list(unit.point_ids)
            # 文档轴值:事实自己没写有效期就按文档日期算;版本只进 axis,不进 valid_from(有效期只收日期,
            # 以前把 rev *L 这种版本串写进 valid_from,时间窗与新旧判断都把它当日期比)
            axis = str(getattr(unit, "axis", "") or "")
            doc_date = str(getattr(unit, "doc_date", "") or "")
            if axis:
                f["axis"] = axis
            if doc_date:
                f["observed"] = doc_date
            default_from = axis if axis_value_kind(axis) == "date" else doc_date
            if default_from and not f.get("valid_from") and not f.get("valid_until"):
                f["valid_from"] = default_from
            if f.get("ref_min_num") is not None or f.get("ref_max_num") is not None:
                dist = bound_distance(f)
                if dist is not None:
                    f["bound_distance"] = dist
    return stats


SUBJECT_UPPERS = ("agent", "entity")    # 与 compile.SUBJECT_UPPERS 同一口径


def document_subjects(entities: list[dict[str, Any]], *, per_doc: int = 4, subject_types: Iterable[str] = (),
                      documents: dict[str, str] | None = None) -> dict[str, list[str]]:
    """每份文档里的默认主体(表格没写主体时用):画像主体(路径 / 同目录序列证据优先,见 profile_subject_by_doc)排最前,
    然后是画像主体类型里的其他实体,其余按频次;只算上层类为 agent / entity 的全局实例(泛指的类别名不算)。
    documents:doc_id → rel_path。"""
    priority = subject_type_priority(subject_types)
    scores: dict[str, dict[str, tuple[int, int]]] = {}
    for e in entities:
        if e.get("scope") or e.get("reference") or e.get("boilerplate"):
            continue
        if str(e.get("upper") or "entity") not in SUBJECT_UPPERS or e.get("kind") == "class":
            continue
        typed = _type_rank(e, priority)
        for doc in e.get("doc_ids") or []:
            scores.setdefault(str(doc), {})[str(e.get("title") or "")] = (typed, int(e.get("frequency") or 0))
    out = {doc: [t for t, _ in sorted(rows.items(), key=lambda kv: (-kv[1][0], -kv[1][1]))[:per_doc]] for doc, rows in scores.items()}
    if priority:
        for doc, ent in profile_subject_by_doc(entities, subject_types, documents=documents).items():
            title = str(ent.get("title") or "")
            rows = [t for t in out.get(doc, []) if t != title]
            out[doc] = ([title] + rows)[:per_doc]
    return out


def unit_subjects(units: Iterable[Unit], entities: list[dict[str, Any]], *, per_unit: int = 4, subject_types: Iterable[str] = (),
                  documents: dict[str, str] | None = None) -> dict[str, list[str]]:
    """每个单元的候选主体(表格 / 叙述没写主体时用):文档路径或同目录序列说明这份文档就是讲某个画像主体的(某人
    文件夹里的报告),它排最前;然后是这个单元里抽到的人员与组织、实体的实例(画像主体类型优先,再按频次);
    再是小节标题点名的那样东西(只给没有归属人的文档,见 HeadingTopics);最后用整份文档的主要主体补足
    (只按频次认出来的画像主体在这里)。
    一份文档里讲好几样东西(汇编手册、合集)时,只给整份文档的几个主要主体,一节里的数值会挂到别的小节的主体上
    (各小节的值不能误归到其他子项目名下)。"""
    units = list(units)
    priority = subject_type_priority(subject_types)
    by_doc = document_subjects(entities, per_doc=per_unit, subject_types=subject_types, documents=documents)
    owner = {doc: str(e.get("title") or "") for doc, (e, how) in subject_evidence_by_doc(entities, subject_types, documents).items()
             if how in ("path", "series")} if priority else {}
    in_unit: dict[str, dict[str, tuple[int, int]]] = {}
    for e in entities:
        if e.get("scope") or e.get("reference") or e.get("boilerplate"):
            continue
        if str(e.get("upper") or "entity") not in SUBJECT_UPPERS or e.get("kind") == "class":
            continue
        rank = (_type_rank(e, priority), int(e.get("frequency") or 0))
        for uid in e.get("unit_ids") or []:
            in_unit.setdefault(str(uid), {})[str(e.get("title") or "")] = rank
    topics = HeadingTopics(entities) if any(u.doc_id not in owner for u in units) else None
    out: dict[str, list[str]] = {}
    for u in units:
        local = [t for t, _ in sorted((in_unit.get(u.unit_id) or {}).items(), key=lambda kv: (-kv[1][0], -kv[1][1]))]
        topic = topics.topic(u) if topics is not None and u.doc_id not in owner else ""
        names = [owner.get(u.doc_id, ""), *local, topic, *by_doc.get(u.doc_id, [])]
        out[u.unit_id] = [t for t in dict.fromkeys(names) if t][:per_unit]
    return out


HEADING_TOPIC_EXCLUDED_UPPERS = ("property", "document")
_HEADING_SPLIT_RE = re.compile(r"\s+[>|]\s+")
_HEADING_WORD_RE = re.compile(r"[0-9a-z]+")


def _heading_lower(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text or ""))).casefold().strip()


class HeadingTopics:
    """小节标题点名的实体(10-24 改版十八)。没有归属人的汇编、手册里,一段自己没点名任何东西时,候选只剩整份文档的
    主要主体,子实验的局数、题数、笔记本结构都挂到了框架名下;小节标题点名的那样东西才是这一节讲的。
    从最近的一级标题往上找,一级里点名了几样取名字最长的。非 ASCII 的名字去掉空白和标点后按子串认(中文没有词边界),
    ASCII 名字按整词认(Feature Discovery 里没有 Redis);名字至少 3 个字符,纯数字不算。
    只认实例(类别名不是哪一样东西),属性、文档不算(数值不归它们),样板、引用不算;作用域是全局或本文档。"""

    def __init__(self, entities: list[dict[str, Any]]) -> None:
        self.wide: dict[str, dict[str, list[tuple[str, str]]]] = {}       # 作用域 → 首字符 → [(规范名, 标题)]
        self.ascii: dict[str, dict[str, list[tuple[str, str]]]] = {}      # 作用域 → 首个词 → [(小写名, 标题)]
        for e in entities:
            if e.get("boilerplate") or e.get("reference") or e.get("kind") == "class":
                continue
            if str(e.get("upper") or "entity") in HEADING_TOPIC_EXCLUDED_UPPERS:
                continue
            title = str(e.get("title") or "")
            scope = str(e.get("scope") or "")
            for name in {title, *[str(a) for a in (e.get("aliases") or [])]}:
                canon = _path_canon(name)
                if len(canon) < 3 or canon.isdigit():
                    continue
                if canon.isascii():
                    low = _heading_lower(name)
                    word = _HEADING_WORD_RE.search(low)
                    if word:
                        self.ascii.setdefault(scope, {}).setdefault(word.group(0), []).append((low, title))
                else:
                    self.wide.setdefault(scope, {}).setdefault(canon[0], []).append((canon, title))

    def topic(self, unit: Unit) -> str:
        levels = [s for s in reversed(unit.section_path) if str(s).strip()]
        for extra in unit.extra_sections or []:
            levels += [s for s in reversed(_HEADING_SPLIT_RE.split(str(extra))) if s.strip()]
        scopes = ("", str(unit.doc_id))
        for level in levels:
            canon, low = _path_canon(level), _heading_lower(level)
            words = set(_HEADING_WORD_RE.findall(low))
            hits: list[tuple[int, str]] = []
            for scope in scopes:
                wide = self.wide.get(scope) or {}
                for ch in set(canon):
                    hits += [(len(c), t) for c, t in wide.get(ch, ()) if c in canon]
                ascii_names = self.ascii.get(scope) or {}
                for word in words:
                    hits += [(len(n), t) for n, t in ascii_names.get(word, ())
                             if re.search(r"(?<![0-9a-z])" + re.escape(n) + r"(?![0-9a-z])", low)]
            if hits:
                return max(hits)[1]
        return ""


def subject_type_priority(subject_types: Iterable[str]) -> dict[str, int]:
    """The profile orders subject types by priority: the first names the main subject,
    while later types provide context. Frequency alone can promote a supporting entity."""
    names = [str(t).casefold() for t in subject_types if str(t).strip()]
    return {t: len(names) - i for i, t in enumerate(names)}


def _type_rank(e: dict[str, Any], priority: dict[str, int]) -> int:
    """画像主体类型可以写在类型链的任何一级(类型本身、父类型、上层类)。"""
    chain = [str(t).casefold() for t in (e.get("type_path") or [e.get("type") or ""])]
    return max([priority.get(t, 0) for t in chain] or [0])


# ── Re-homing measurement facts: a table row may name a metric or a report instead
# of the subject that owns the measured value. Use document evidence to resolve it. ──
_GENERIC_PROPERTY_WORDS = {"测量结果", "检查结果", "检测结果", "检验结果", "结果", "值", "数值", "测定值", "参数", "指标",
                           "measurement", "measured value", "value", "result", "reading", "level"}
_FLAG_SUFFIX_RE = re.compile(
    r"(?:增高|偏高|升高|过高|降低|偏低|减低|下降|过低|异常|阳性|阴性|超标|"
    r"\s+(?:elevated|increased|raised|high|decreased|reduced|low|abnormal|positive|negative))\s*$", re.IGNORECASE)
_FLAG_SYMBOL = {"增高": "↑", "偏高": "↑", "升高": "↑", "过高": "↑", "超标": "↑", "elevated": "↑", "increased": "↑", "raised": "↑", "high": "↑",
                "降低": "↓", "偏低": "↓", "减低": "↓", "下降": "↓", "过低": "↓", "decreased": "↓", "reduced": "↓", "low": "↓"}
_PERIOD_KEY_RE = re.compile(r"年份|年度|日期|时间|期间|date|year|period|time", re.IGNORECASE)


def _entity_lookup(entities: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for e in entities:
        scope = str(e.get("scope") or "")
        for n in [str(e.get("title") or "")] + [str(a) for a in (e.get("aliases") or [])]:
            n = n.casefold().strip()
            if n:
                out.setdefault((scope, n), e)
    return out


_NUMBERED_NAME_RE = re.compile(r"\d{4,}")


def _path_canon(text: str) -> str:
    return re.sub(r"[\s_\-\.·/:：()（）]+", "", unicodedata.normalize("NFKC", str(text or ""))).casefold()


def subject_evidence_by_doc(entities: list[dict[str, Any]], subject_types: Iterable[str],
                            documents: dict[str, str] | None = None) -> dict[str, tuple[dict[str, Any], str]]:
    """The profile subject entity of each document and its basis, three levels of evidence:
    · path: a segment of the document path (directory or file name) is the name / alias of some profile subject
      entity: "Li Hua/checkup report/2025-ECG.pdf" belongs to Li Hua, material under "ZK7C1049GN/" to that device,
      "Party A XX/contract" to Party A;
    · series: more than half of the documents in the same directory (at least two) with a real-name subject point
      to S, while this document's own subject is a numbered name (Mr. Li_1000000000001_2) or there is no entity of
      a profile type at all (the subject was extracted as the issuer's name or the lab sheet name): the directory
      is one report series, so it belongs to S;
    · frequency: the highest-ranked entity of the profile subject types in this document by type priority /
      frequency / document count (the original rule).
    Returns empty without profile subject types."""
    priority = subject_type_priority(subject_types)
    if not priority:
        return {}
    best: dict[str, tuple[tuple[int, int, int], dict[str, Any]]] = {}
    typed_entities: list[dict[str, Any]] = []
    for e in entities:
        if e.get("scope") or e.get("reference") or e.get("boilerplate") or e.get("kind") == "class":
            continue
        # 与 document_subjects 同一口径:文档主体是人员与组织或实体。事件、概念也可能在画像类型中,
        # 但不能因为名字与文件名匹配,就压过真正持有测量值的主体。
        if str(e.get("upper") or "entity") not in SUBJECT_UPPERS:
            continue
        typed = _type_rank(e, priority)
        if not typed:
            continue
        typed_entities.append(e)
        rank = (typed, int(e.get("frequency") or 0), len(e.get("doc_ids") or []))     # type priority first, then frequency, then document count
        for doc in e.get("doc_ids") or []:
            cur = best.get(str(doc))
            if cur is None or rank > cur[0]:
                best[str(doc)] = (rank, e)
    out: dict[str, tuple[dict[str, Any], str]] = {doc: (e, "frequency") for doc, (_, e) in best.items()}
    docs = {str(k): str(v) for k, v in (documents or {}).items() if str(v or "").strip()}
    if not docs:
        return out
    by_name: dict[str, dict[str, Any]] = {}
    for e in typed_entities:
        for n in [str(e.get("title") or "")] + [str(a) for a in (e.get("aliases") or [])]:
            c = _path_canon(n)
            if len(c) >= 2:
                by_name.setdefault(c, e)
    for doc, rel in docs.items():
        parts = [p for p in re.split(r"[\\/]+", rel) if p]
        if parts:
            parts[-1] = parts[-1].rsplit(".", 1)[0]
        hit = next((by_name[_path_canon(p)] for p in reversed(parts) if _path_canon(p) in by_name), None)
        if hit is not None:
            out[doc] = (hit, "path")
    # Same-directory series: only when more than half of the documents with a basis (path / frequency) in the
    # directory point to one subject are the numbered / subject-less documents assigned to it
    by_dir: dict[str, list[str]] = {}
    for doc, rel in docs.items():
        parts = [p for p in re.split(r"[\\/]+", rel) if p]
        by_dir.setdefault("/".join(parts[:-1]), []).append(doc)
    for folder, members in by_dir.items():
        if len(members) < 2:
            continue
        votes: dict[str, int] = {}
        keyed: dict[str, dict[str, Any]] = {}
        for doc in members:
            ent, how = out.get(doc, (None, ""))
            if ent is None or how == "series":
                continue
            title = str(ent.get("title") or "")
            if how == "frequency" and _NUMBERED_NAME_RE.search(title):
                continue
            votes[title] = votes.get(title, 0) + 1
            keyed[title] = ent
        if not votes:
            continue
        top, n = max(votes.items(), key=lambda kv: kv[1])
        if n < 2 or n * 2 < sum(votes.values()):
            continue
        for doc in members:
            ent, how = out.get(doc, (None, ""))
            if how == "path":
                continue
            if ent is None or (how == "frequency" and _NUMBERED_NAME_RE.search(str(ent.get("title") or "")) and str(ent.get("title") or "") != top):
                out[doc] = (keyed[top], "series")
    return out


def profile_subject_by_doc(entities: list[dict[str, Any]], subject_types: Iterable[str],
                           documents: dict[str, str] | None = None) -> dict[str, dict[str, Any]]:
    """The profile subject entity of each document (basis: see subject_evidence_by_doc)."""
    return {doc: e for doc, (e, _) in subject_evidence_by_doc(entities, subject_types, documents).items()}


def _document_own_entity(ent: dict[str, Any]) -> bool:
    """Whether this entity "belongs to this one document only and is a subject in its own right": a global type
    (not a document-scoped component like part / property) that appears in this one document only, as checkup
    numbers, issuer names and numbered subjects do. When path / series evidence assigns a document to the profile
    subject, only the facts under such an entity move over as a whole. Facts under cross-document global entities
    (the products and competitors in a feature comparison table) and document-scoped components (a product's
    modules, features) are their own and stay put. 2026-09-12 evaluation: matching a top-level directory
    to a shared parent entity incorrectly reassigned all 26k facts to that entity."""
    return not ent.get("scope") and len(ent.get("doc_ids") or []) <= 1


def is_measurement(fact: dict[str, Any]) -> bool:
    return (fact.get("value_num") is not None or fact.get("min_num") is not None or fact.get("max_num") is not None
            or bool(str(fact.get("flag") or "").strip()) or bool(fact.get("ref_min") or fact.get("ref_max")) or bool(fact.get("unit")))


def _is_quantity(fact: dict[str, Any]) -> bool:
    """The value is a quantity: a number was parsed, or a value field is a range, an expression or an unverified
    glued number."""
    if any(fact.get(f"{field}_num") is not None for field in ("value", "min", "typ", "max")):
        return True
    return any(kind != "text" for kind in (fact.get("kinds") or {}).values())


def normalize_measurements(facts: list[dict[str, Any]], entities: list[dict[str, Any]], *, units_by_id: dict[str, Unit],
                           profile: dict[str, Any] | None = None, documents: dict[str, str] | None = None) -> dict[str, int]:
    """确定性归位,在 link_facts 之后跑(缓存里的旧事实也能修):
    · 属性名尾巴上的标记词(总胆固醇增高 / elevated)剥到 flag,概念才能并到「总胆固醇」;只剥取值是文字的,
      原属性名记 property_raw;
    · 条件里的时期(年份: 2024/07)搬到 valid_from / period_text,历年对照表的每一行各成一个时间点;
    · 文档有归属人(路径 / 同目录序列证据定下的画像主体:某人文件夹里的报告、某器件目录下的资料)时,测量事实的主体
      改成归属人:原主体是化验单 / 检查项目(上层 process)或指标本身(上层 property)或与属性同名、属性是「测量结果」
      这类泛词的;日期轴的语料里,带参考范围或标记的测量即使挂在诊断名下(异常汇总表:肥胖 → 尿酸 479↑)也归受检者,
      原主体记到 context。只按频次认出来的画像主体不归位,避免把文档中的不同对象归到最常出现的名字下。
    · 文档主体由路径 / 同目录序列证据定下来(subject_evidence_by_doc)时,挂在本文档自己那个「主体类型」实体
      (李先生_体检号、签发人名)名下的事实整个归到画像主体(resubjected_identity)。
    不改数值,不删事实;改过主体的记 subject_raw。"""
    from .temporal import parse_date

    profile = profile or {}
    evidence = subject_evidence_by_doc(entities, profile.get("subject_types") or (), documents)
    subj_by_doc = {doc: e for doc, (e, how) in evidence.items() if how in ("path", "series")}
    priority = subject_type_priority(profile.get("subject_types") or ())
    date_axis = str(profile.get("axis") or "") == "date"
    lookup = _entity_lookup(entities)
    stats = {"flag_words": 0, "period_conditions": 0, "resubjected_process": 0, "resubjected_indicator": 0,
             "resubjected_generic": 0, "resubjected_context": 0, "resubjected_identity": 0, "identity_kept_shared": 0}
    # How many subject-type entities carry facts in each document (excluding the profile subject itself): two or
    # more means a comparison table / multi-subject document, and the path subject does not take everything over.
    # A checkup report has the examinee as its only subject; a feature comparison table has one product per column
    subjects_by_doc: dict[str, set[str]] = {}
    for f in facts:
        unit = units_by_id.get(str(f.get("unit_id") or ""))
        doc = unit.doc_id if unit else str(f.get("doc_id") or "")
        target = subj_by_doc.get(doc)
        subject = str(f.get("subject") or "")
        if target is None or not subject or subject.casefold() == str(target.get("title") or "").casefold():
            continue
        ent = lookup.get((doc, subject.casefold())) or lookup.get(("", subject.casefold()))
        if ent is not None and _type_rank(ent, priority):
            subjects_by_doc.setdefault(doc, set()).add(subject.casefold())
    for f in facts:
        prop = str(f.get("property") or "")
        m = _FLAG_SUFFIX_RE.search(prop)
        # not split off when the value is a quantity: in "Price High 105.2" or "consumption down 30%" the tail word
        # is part of the property name, the very thing the number measures; splitting it off would merge Price High
        # and Price Low into one property and read the decrease as the indicator itself. Only a text value
        # (elevated, positive, a passage of findings) makes it a flag
        if m and len(prop) > len(m.group(0)) and not _is_quantity(f):
            word = m.group(0).strip().casefold()
            f["property_raw"] = prop
            f["property"] = prop[: m.start()].strip(" :：-")
            if not str(f.get("flag") or "").strip():
                f["flag"] = _FLAG_SYMBOL.get(word, m.group(0).strip())
            stats["flag_words"] += 1
        conds = dict(f.get("conditions") or {})
        # a validity inherited from the document axis (filled in by link_facts) is not one the fact states itself:
        # a multi-year comparison table has each row's year in the conditions, and unless it is moved the whole
        # table lands on the single point of the report date
        inherited = not f.get("valid_until") and str(f.get("valid_from") or "") == str(f.get("axis") or "")
        for k in list(conds):
            if _PERIOD_KEY_RE.search(str(k)) and (inherited or not f.get("valid_from")):
                parsed = parse_date(str(conds[k]))
                if parsed:
                    f["valid_from"] = parsed
                    f["period_text"] = f"{k}: {conds[k]}"
                    conds.pop(k)
                    f["conditions"] = conds
                    stats["period_conditions"] += 1
                    break
        unit = units_by_id.get(str(f.get("unit_id") or ""))
        doc = unit.doc_id if unit else str(f.get("doc_id") or "")
        target = subj_by_doc.get(doc)
        if target is None:
            continue
        subject = str(f.get("subject") or "")
        if not subject or subject.casefold() == str(target.get("title") or "").casefold() or f.get("subject_key") == target.get("key"):
            continue
        ent = lookup.get((doc, subject.casefold())) or lookup.get(("", subject.casefold()))
        upper = str((ent or {}).get("upper") or "").casefold()
        prop = str(f.get("property") or "")
        generic = prop.casefold() in _GENERIC_PROPERTY_WORDS or not prop
        same = prop.casefold() == subject.casefold()
        measure = is_measurement(f)
        rule = ""
        identity_case = ent is not None and _type_rank(ent, priority) and evidence.get(doc, (None, ""))[1] in ("path", "series")
        if identity_case and (not _document_own_entity(ent) or len(subjects_by_doc.get(doc, ())) > 1):
            stats["identity_kept_shared"] += 1     # cross-document entity / component / multi-subject document: facts keep their own subject
            identity_case = False
        if identity_case:
            rule = "resubjected_identity"          # 主体是本文档自己的「主体类型」实体,但路径 / 序列证据说文档属于画像主体
        elif measure and upper in ("process", "event"):
            rule = "resubjected_process"       # 化验单 / 检查项目 / 某次检查名下的读数归受检者;「反馈单的提交人是谁」这种业务事实留在原主体(Codex 2026-09-13 F03)
        elif measure and (same or generic):
            rule = "resubjected_generic"
        elif measure and upper == "property":
            rule = "resubjected_indicator"
        elif measure and date_axis and (f.get("ref_min") or f.get("ref_max") or str(f.get("flag") or "").strip()) and not generic:
            rule = "resubjected_context"
        if not rule:
            continue
        f["subject_raw"] = subject
        new_prop = subject if (rule in ("resubjected_generic", "resubjected_indicator") and (generic or same)) else prop
        if new_prop.casefold() != subject.casefold():
            f["context"] = subject            # original subject is not the measured property (lab sheet, diagnosis): kept as context, out of the grouping key
        f["property"] = new_prop
        f["subject"] = str(target.get("title") or subject)
        f["subject_key"] = str(target.get("key") or "")
        f["subject_keys"] = [f["subject_key"]]
        stats[rule] += 1
    return stats


