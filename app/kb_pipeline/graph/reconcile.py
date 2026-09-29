"""Cross-document fact reconciliation (plan 5.A): facts with the same subject, concept, conditions and unit are
ordered along the axis into a series; only facts with the same axis value (or none at all) and different
values are recorded as conflicts. Annotation only, values are never changed (the record shape of Semantica's
conflicts, without its adjudication). Age 30 and 31 from two reports two years apart are a series, not a
contradiction.

The output is written back into the facts (series_key / series_len / series_index / conflict_group), and the
conflict list and stats are returned for the timeline pages and the health-check panel.
"""
from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from typing import Any

from .concepts import concept_norm
from .facts import classify_value
from ..measure_units import unit_key
from .temporal import axis_sort_key, axis_value_kind

VALUE_TOLERANCE = 1e-6


def _conditions_norm(conditions: dict[str, Any] | None) -> str:
    items = sorted((str(k).casefold().strip(), unicodedata.normalize("NFKC", str(v)).casefold().replace(" ", ""))
                   for k, v in (conditions or {}).items())
    return json.dumps(items, ensure_ascii=False)


def _subject_norm(fact: dict[str, Any]) -> str:
    key = str(fact.get("subject_key") or "")
    if key:
        return key
    return unicodedata.normalize("NFKC", str(fact.get("subject") or "")).casefold().replace(" ", "")


def _concept(fact: dict[str, Any]) -> str:
    return str(fact.get("concept_key") or "") or concept_norm(fact.get("property"), fact.get("symbol"))


def group_key(fact: dict[str, Any]) -> tuple[str, str, str]:
    """The group key excludes the unit: unit-less facts ("total cholesterol elevated" in an abnormality summary)
    must join the measurement series that carries a unit; a group is only split by unit when several units occur
    in it (see split_by_unit)."""
    return (_subject_norm(fact), _concept(fact), _conditions_norm(fact.get("conditions")))


def unit_of(fact: dict[str, Any]) -> str:
    # the canonical key keeps case (final review F04): 2 mW and 1 MW are not one group and must not yield "falling"
    return unit_key(fact.get("unit_canonical") or fact.get("unit") or "")


def split_by_unit(rows: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    """Split a group's facts by unit: a single unit (or none at all) is not split; several units form one group
    each, and unit-less facts join the largest group."""
    units = [u for u in (unit_of(r) for r in rows) if u]
    if len(set(units)) <= 1:
        return [(units[0] if units else "", rows)]
    from collections import Counter

    main = Counter(units).most_common(1)[0][0]
    out: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(unit_of(r) or main, []).append(r)
    return sorted(out.items(), key=lambda kv: (-len(kv[1]), kv[0]))


def fact_axis(fact: dict[str, Any]) -> str:
    return str(fact.get("valid_from") or fact.get("axis") or "")


def _values_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Whether two facts on the same axis value agree: scalars by relative tolerance, everything else by casefolded
    text without whitespace, min / typ / max item by item. A number on one side and only a flag word on the other
    (the summary table's "elevated") is not a disagreement: they are two spellings of one measurement."""
    a_num = any(a.get(f"{k}_num") is not None for k in ("value", "min", "typ", "max"))
    b_num = any(b.get(f"{k}_num") is not None for k in ("value", "min", "typ", "max"))
    if a_num != b_num and not (unit_of(a) and unit_of(b)):
        return True
    for field in ("value", "min", "typ", "max"):
        ai, bi = classify_value(a.get(field)), classify_value(b.get(field))
        if ai["kind"] == "scalar" and bi["kind"] == "scalar":
            # the comparator is part of the value: "<1.3" and "1.3" are not the same value (Codex review F04)
            if (ai.get("cmp") or "") != (bi.get("cmp") or ""):
                return False
            if not math.isclose(float(ai["num"]), float(bi["num"]), rel_tol=VALUE_TOLERANCE, abs_tol=VALUE_TOLERANCE):
                return False
            continue
        if ai["kind"] == "range" and bi["kind"] == "range":
            if not (math.isclose(ai["lo"], bi["lo"], rel_tol=VALUE_TOLERANCE, abs_tol=VALUE_TOLERANCE)
                    and math.isclose(ai["hi"], bi["hi"], rel_tol=VALUE_TOLERANCE, abs_tol=VALUE_TOLERANCE)):
                return False
            continue
        at = unicodedata.normalize("NFKC", str(a.get(field) or "")).casefold().replace(" ", "")
        bt = unicodedata.normalize("NFKC", str(b.get(field) or "")).casefold().replace(" ", "")
        if at != bt:
            return False
    return True


def reconcile_facts(facts: list[dict[str, Any]]) -> dict[str, Any]:
    """Returns {"conflicts": [...], "stats": {...}}; series_* / conflict_group are written into the facts in place."""
    raw_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    skipped_conflict = 0
    for f in facts:
        if not (f.get("subject") and (f.get("property") or f.get("symbol"))):
            continue
        if f.get("evidence_conflict"):
            skipped_conflict += 1          # image text vs estimated reading conflicted: no series, no conflict check (Codex review F01); the fact is kept
            continue
        raw_groups.setdefault(group_key(f), []).append(f)
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for key, rows in raw_groups.items():
        for unit, part in split_by_unit(rows):
            groups[(*key, unit)] = part
    stats = {"groups": len(groups), "multi_doc_groups": 0, "series": 0, "series_facts": 0, "conflicts": 0, "conflict_facts": 0,
             "same_document_variants": 0, "evidence_conflict_skipped": skipped_conflict}
    conflicts: list[dict[str, Any]] = []
    for key, rows in groups.items():
        docs = {str(r.get("doc_id") or "") for r in rows}
        if len(rows) < 2:
            continue
        if len(docs) >= 2:
            stats["multi_doc_groups"] += 1
        by_axis: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            by_axis.setdefault(fact_axis(r), []).append(r)
        # date axes and version axes form separate series: "2023-10-31" and "v6.0" have no order between them, so
        # in one series the start, the end and the trend would all be arbitrary
        axes_by_kind: dict[str, list[str]] = {}
        for a in by_axis:
            if a:
                axes_by_kind.setdefault(axis_value_kind(a), []).append(a)
        for kind, axes in axes_by_kind.items():
            if len(axes) < 2:
                continue
            stats["series"] += 1
            skey = "s" + hashlib.sha256("|".join((*key, kind)).encode("utf-8")).hexdigest()[:16]
            ordered = sorted(axes, key=axis_sort_key)
            for idx, axis in enumerate(ordered):
                for r in by_axis[axis]:
                    r["series_key"] = skey
                    r["series_len"] = len(ordered)
                    r["series_index"] = idx
                    stats["series_facts"] += 1
        for axis, same in by_axis.items():
            if len(same) < 2:
                continue
            distinct: list[dict[str, Any]] = []
            for r in same:
                if not any(_values_equal(r, d) for d in distinct):
                    distinct.append(r)
            if len(distinct) < 2:
                continue
            if len({str(r.get("doc_id") or "") for r in distinct}) < 2:
                # two values within one document: in the same unit it is mostly a two-column table whose condition
                # was not extracted; in different paragraphs the same property was measured once per occasion (two
                # heart rates in a checkup, a parameter given in two manual sections). Neither is a cross-document
                # conflict, so they are only counted (2026-09-08: all 7 groups of the health KB were of this kind)
                stats["same_document_variants"] += 1
                continue
            ckey = "x" + hashlib.sha256(f"{'|'.join(key)}|{axis}".encode("utf-8")).hexdigest()[:16]
            for r in same:
                r["conflict_group"] = ckey
            stats["conflicts"] += 1
            stats["conflict_facts"] += len(same)
            conflicts.append({
                "key": ckey, "subject": str(same[0].get("subject") or ""), "concept": str(same[0].get("concept") or same[0].get("property") or ""),
                "axis": axis, "conditions": same[0].get("conditions") or {},
                "values": [{"fact_id": r.get("id"), "value": r.get("value") or "", "min": r.get("min") or "", "typ": r.get("typ") or "",
                            "max": r.get("max") or "", "unit": r.get("unit") or "", "doc_id": r.get("doc_id"), "rel_path": r.get("rel_path"),
                            "section": r.get("section"), "quality": r.get("quality")} for r in distinct],
            })
    conflicts.sort(key=lambda c: (c["subject"], c["concept"], c["axis"]))
    return {"conflicts": conflicts, "stats": stats}


def series_of(facts: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """{series_key: facts ordered by axis}, for the timeline pages."""
    out: dict[str, list[dict[str, Any]]] = {}
    for f in facts:
        if f.get("series_key"):
            out.setdefault(str(f["series_key"]), []).append(f)
    for rows in out.values():
        rows.sort(key=lambda r: (axis_sort_key(fact_axis(r)), str(r.get("doc_id") or "")))
    return out
