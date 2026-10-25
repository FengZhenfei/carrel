"""跨文档事实对账(方案 5.A):同主体、同概念、同条件、同单位的事实按轴排成序列;
轴相同(或都没有轴)而值不同的才记冲突。只标注,不改值(Semantica conflicts 的记录形状,
不要它的裁决)。年龄 30 与 31 来自两年的两份报告,是序列不是矛盾。

「同条件」保守采用所有非时期条件。跨文档不同条件不能仅因每篇只有一个值就被忽略;
上下文归类需要显式语义,不从样本频率推断。

产出写回事实(series_key / series_len / series_index / conflict_group / superseded_by / superseded_on),
并返回冲突清单与统计,时间线页与体检面板用;序列里最新那一档是现行值,更早的标被取代(检索默认只给现行值)。
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
    """分组键不含单位:没单位的事实(异常汇总里的「总胆固醇 增高」)要并进带单位的测量序列;
    同一组里出现多个单位才按单位拆开(见 split_by_unit)。条件只算全部条件,真正参与分组的是 qualifier_keys 挑出来的那几个。"""
    return (_subject_norm(fact), _concept(fact), _conditions_norm(fact.get("conditions")))


def _condition_items(fact: dict[str, Any]) -> dict[str, str]:
    from .facts import _PERIOD_KEY_RE

    return {str(k).casefold().strip(): unicodedata.normalize("NFKC", str(v)).casefold().replace(" ", "")
            for k, v in (fact.get("conditions") or {}).items() if not _PERIOD_KEY_RE.search(str(k))}


def qualifier_keys(rows: list[dict[str, Any]]) -> set[str]:
    """Until context has explicit semantics, every non-period condition qualifies identity.

    Observing only one temperature per document does not make temperature report metadata.
    Missing conditions also remain distinct from an explicitly qualified measurement.
    """
    return {key for row in rows for key in _condition_items(row)}



def _qualified(fact: dict[str, Any], keys: set[str]) -> str:
    items = _condition_items(fact)
    return json.dumps(sorted((k, items.get(k, "")) for k in keys), ensure_ascii=False)


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
        for key in ("superseded_by", "superseded_on", "series_key", "series_len", "series_index", "conflict_group"):
            f.pop(key, None)
    for f in facts:
        if not (f.get("subject") and (f.get("property") or f.get("symbol"))):
            continue
        if f.get("evidence_conflict"):
            skipped_conflict += 1          # image text vs estimated reading conflicted: no series, no conflict check (Codex review F01); the fact is kept
            continue
        raw_groups.setdefault((_subject_norm(f), _concept(f), ""), []).append(f)
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for (subj, concept, _), rows in raw_groups.items():
        keys = qualifier_keys(rows)
        by_cond: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            by_cond.setdefault(_qualified(r, keys), []).append(r)
        for cond, part in by_cond.items():
            for unit, unit_rows in split_by_unit(part):
                groups[(subj, concept, cond, unit)] = unit_rows
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
            # 新旧(方案第 7 条):序列里最新那一档是现行值,更早的每一档被它后面那一档取代(沿 superseded_by 能一路走到现行值)
            for idx, axis in enumerate(ordered[:-1]):
                newer = ordered[idx + 1]
                for r in by_axis[axis]:
                    r["superseded_by"] = by_axis[newer][0].get("id")
                    r["superseded_on"] = newer
                    stats["superseded_facts"] = stats.get("superseded_facts", 0) + 1
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
                # conflict, so they are only counted (2026-09-08: all 7 groups in one evaluation corpus were of this kind)
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
