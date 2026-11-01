"""Source time evidence and explicit retrieval scope.

Default retrieval retains dated, historical and conflicting source assertions.
Dates and single-valued predicates do not establish authority or supersession.
Only explicit document governance excludes evidence by default. An explicit
as_of query filters by the source's stated interval or observation date; it does
not decide which assertion is true. Agent/human governance owns that decision.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

from . import prompts
from .quality import relation_qualifier, traversable_relation
from .extract import normalize_predicate
from .units import Unit


STATEFUL_UPPERS = ("agent", "entity", "part")
EVENT_UPPER = "event"


def is_interval(start: Any, until: Any) -> bool:
    """原文写明的是一段时间:止期是日期,且没写起期或止期晚于起期。起止相同的是一件事发生的时间,不是一段状态。"""
    s, u = str(start or ""), str(until or "")
    return is_date(u) and (not is_date(s) or u > s)


def stateful_keys(entities: Iterable[dict[str, Any]]) -> set[str]:
    """有状态会变的实例(人员与组织、实体、部件,不含类别名):用于识别原文中的持续状态区间。"""
    return {str(e["key"]) for e in entities if str(e.get("upper") or "entity") in STATEFUL_UPPERS and e.get("kind") != "class"}


def relation_lapses(r: dict[str, Any], stateful: set[str] | None, upper_of: dict[str, str]) -> bool:
    """这条关系过了止期就不再成立吗(见模块说明):有类型的关系、起点是有状态的实例、写明的是一段时间、另一端不是事件。
    stateful 是 None 时不按起点筛(调用方没给实体)。"""
    if str(r.get("predicate") or prompts.DEFAULT_PREDICATE) == prompts.DEFAULT_PREDICATE:
        return False
    if stateful is not None and str(r.get("source_key") or "") not in stateful:
        return False
    ends = (upper_of.get(str(r.get("source_key") or "")), upper_of.get(str(r.get("target_key") or "")))
    return is_interval(r.get("valid_from"), r.get("valid_until")) and EVENT_UPPER not in ends


def mark_fact_lapses(facts: Iterable[dict[str, Any]], entities: Iterable[dict[str, Any]]) -> int:
    """标记原文明确限定的状态区间,仅用于显式时间查询。"""
    marked = 0
    for f in facts:
        # A policy/document can also have an explicitly bounded state. Events and
        # measurements remain observations; an explicitly one-day state expires.
        start, end = str(f.get("valid_from") or ""), str(f.get("valid_until") or "")
        f["lapses"] = (f.get("fact_kind") == "state" and is_date(end)
                       and (not is_date(start) or start <= end))
        marked += int(f["lapses"])
    return marked


def relation_currency(relations: list[dict[str, Any]], units_by_id: dict[str, Unit], *,
                      predicates: Iterable[dict[str, Any]] = (), entities: Iterable[dict[str, Any]] | None = None) -> dict[str, int]:
    """Attach source dates and interval metadata; never infer supersession.
    Single-valued alternatives are counted for review, without selecting a winner.
    """

    single = {normalize_predicate(str(p.get("name") or "")) for p in predicates if p.get("multivalued") is False}
    stateful = None
    upper_of: dict[str, str] = {}
    if entities is not None:
        entities = list(entities)
        stateful = stateful_keys(entities)
        upper_of = {str(e["key"]): str(e.get("upper") or "entity") for e in entities}
    stats = {"relations_dated": 0, "relations_with_period": 0, "relations_lapsing": 0, "single_valued_groups": 0,
             "superseded": 0, "undecided_groups": 0, "stateless_sources_skipped": 0}
    for r in relations:
        dates = sorted({str(getattr(units_by_id.get(u), "doc_date", "") or "") for u in (r.get("unit_ids") or [])} - {""})
        r["observed_first"] = dates[0] if dates else ""
        r["observed_last"] = dates[-1] if dates else ""
        r.pop("superseded_by", None)
        r.pop("superseded_on", None)
        r["lapses"] = relation_lapses(r, stateful, upper_of)
        if dates:
            stats["relations_dated"] += 1
        if r.get("valid_from") or r.get("valid_until"):
            stats["relations_with_period"] += 1
        stats["relations_lapsing"] += int(r["lapses"])
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in relations:
        if r.get("derived") or not traversable_relation(r) or str(r.get("predicate") or "") not in single:
            continue
        if stateful is not None and str(r["source_key"]) not in stateful:
            stats["stateless_sources_skipped"] += 1
            continue
        groups.setdefault((str(r["source_key"]), str(r["predicate"]), relation_qualifier(r)), []).append(r)
    for rows in groups.values():
        if len({str(r["target_key"]) for r in rows}) < 2:
            continue
        stats["single_valued_groups"] += 1
        stats["undecided_groups"] += 1
    return stats


# Default: all source assertions except explicitly governed documents.
# as_of: source time scope; history: include explicitly governed documents too.
STATUS_CURRENT = "current"
STATUS_SUPERSEDED = "superseded"            # 有更新的说法(单值关系 / 事实序列里的更早一档)
STATUS_ENDED = "ended"                      # 原文写明的一段状态,止期已过
STATUS_NOT_STARTED = "not_started"          # 回看某一天时:那天还没开始(或还没有任何文档提到)
STATUS_DOC_SUPERSEDED = "doc_superseded"    # 来源文档全都被更正文件标成被取代 / 过了有效期
STATUS_DOC_ARCHIVED = "doc_archived"        # 来源文档全都被标成归档
_DATE_RE = re.compile(r"^[0-9]{4}")       # 与 Cypher 里的 '[0-9]{4}.*' 同一口径:四位年份开头的才当日期比


def is_date(value: Any) -> bool:
    return bool(_DATE_RE.match(str(value or "")))


def as_of_day(value: Any) -> str:
    """回看日期规整成 YYYY-MM-DD:只给年就是那年的最后一天,给到月就是那月的最后一天(「截至 2024 年」);空串原样返回;
    认不出来就 ValueError(接口回 422)。"""
    import calendar

    text = str(value or "").strip()
    if not text:
        return ""
    m = re.fullmatch(r"(\d{4})(?:-(\d{1,2})(?:-(\d{1,2}))?)?", text)
    if not m:
        raise ValueError(f"as_of must be YYYY, YYYY-MM or YYYY-MM-DD, got {text!r}")
    y, mo, d = int(m.group(1)), m.group(2), m.group(3)
    if mo is None:
        return f"{y:04d}-12-31"
    mo_i = int(mo)
    if not 1 <= mo_i <= 12:
        raise ValueError(f"as_of month out of range: {text!r}")
    last = calendar.monthrange(y, mo_i)[1]
    if d is None:
        return f"{y:04d}-{mo_i:02d}-{last:02d}"
    if not 1 <= int(d) <= last:
        raise ValueError(f"as_of day out of range: {text!r}")
    return f"{y:04d}-{mo_i:02d}-{int(d):02d}"


def _before(a: str, day: str) -> bool:
    """a 早于 day(按 a 的精度比):「2024」早于「2025-01-01」,不早于「2024-06-30」。"""
    return bool(a) and is_date(a) and a < day[: len(a)]


def _after(a: str, day: str) -> bool:
    return bool(a) and is_date(a) and a > day[: len(a)]


def _ended(row: dict[str, Any], day: str) -> bool:
    return bool(row.get("lapses")) and _before(str(row.get("valid_until") or ""), day)


def row_status(row: dict[str, Any], *, day: str, as_of: bool = False) -> str:
    """Query-scope status, not a judgment of business validity or truth."""
    if as_of:
        start = str(row.get("valid_from") or row.get("observed_at") or row.get("observed_first") or row.get("observed") or "")
        if _after(start, day):
            return STATUS_NOT_STARTED
        if _ended(row, day):
            return STATUS_ENDED
        return STATUS_CURRENT
    doc = str(row.get("doc_status") or "")
    if doc:
        return STATUS_DOC_ARCHIVED if doc == "archived" else STATUS_DOC_SUPERSEDED
    return STATUS_CURRENT


def next_change_day(rows: Iterable[dict[str, Any]], *, today: str) -> str:
    """Source dates do not expire the default evidence projection.

    Explicit document governance has its own corrections digest/refresh path.
    Kept as a stable build hook for callers recording temporal_refresh_on.
    """
    return ""


def cypher_in_effect(var: str, *, observed: str, as_of: bool) -> str:
    """Same source-scope semantics as row_status, with empty-date fallback."""
    v = var
    if not as_of:
        return f"(coalesce({v}.doc_status, '') = '')"
    until = (f"(NOT coalesce({v}.lapses, false) OR NOT coalesce({v}.valid_until, '') =~ '[0-9]{{4}}.*' "
             f"OR {v}.valid_until >= left($day, size({v}.valid_until)))")
    start = (f"CASE WHEN coalesce({v}.valid_from, '') <> '' THEN {v}.valid_from "
             f"WHEN coalesce({v}.observed_at, '') <> '' THEN {v}.observed_at "
             f"ELSE coalesce({v}.{observed}, '') END")
    started = f"(NOT ({start}) =~ '[0-9]{{4}}.*' OR ({start}) <= left($day, size({start})))"
    return f"({started} AND {until})"


def query_scope(*, history: bool = False, as_of: Any = None, today: str | None = None) -> dict[str, Any]:
    """一次查询的时间口径:{"mode": current | as_of | history, "day": YYYY-MM-DD, "as_of": 状态按回看那天标}。
    history 优先(状态按 as_of 那天或今天标);as_of 写法认不出来 → ValueError。"""
    import time as _time

    day = as_of_day(as_of)
    now = today or _time.strftime("%Y-%m-%d")
    if history:
        return {"mode": "history", "day": day or now, "as_of": bool(day)}
    if day:
        return {"mode": "as_of", "day": day, "as_of": True}
    return {"mode": "current", "day": now, "as_of": False}


def cypher_in_scope(var: str, *, observed: str, scope: dict[str, Any]) -> str:
    """Cypher 条件:关系边 / 事实节点按口径算不算数(参数 $day 取 scope["day"]);history 一律算。"""
    if scope["mode"] == "history":
        return "true"
    return cypher_in_effect(var, observed=observed, as_of=scope["mode"] == "as_of")


def in_scope(row: dict[str, Any], scope: dict[str, Any]) -> bool:
    """Python source-scope filter, matching cypher_in_scope."""
    return scope["mode"] == "history" or row_status(row, day=scope["day"], as_of=bool(scope.get("as_of"))) == STATUS_CURRENT


def mark_status(row: dict[str, Any], scope: dict[str, Any]) -> dict[str, Any]:
    row["status"] = row_status(row, day=scope["day"], as_of=bool(scope.get("as_of")))
    return row


def qdrant_current_conditions() -> list[Any]:
    """Default evidence excludes only explicitly governed documents."""
    from qdrant_client.http import models as qm

    return [qm.IsEmptyCondition(is_empty=qm.PayloadField(key="doc_status"))]
