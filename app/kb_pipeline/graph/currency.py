"""关系与事实的时间和新旧(2026-10-24 破坏性改版,方案附录第 3 条:关系与事实用同一套时间与状态字段)。

· 观察时间:一条关系 / 事实来自哪些文档,那些文档的日期就是它被看到的时间(observed_first / observed_last)。
· 起止:原文写明的 valid_from / valid_until(抽取时已按文档日期换算过相对日期)。
· 新旧:只对模式里显式声明单值的关系(multivalued: false)做 —— 同一起点、同一谓词有几个终点时,按时间(起点,
  没写就用观察时间)排,最新的是现行,更早的记 superseded_by(取代它的那条)与 superseded_on(被取代的时间)。
  时间相同或都没有时间就不判,几条都算现行(宁可留着让人看见冲突,也不凭空藏掉)。
  起点只限有状态会变的东西(人员与组织、实体、部件,且是实例):概念、属性、过程、事件、文档上的关系是一般性的说法,
  后来的文档换个说法不等于旧的作废(多种可能的属性值可以并存)。
  事实的新旧在 reconcile 里按序列判(同主体、同概念、同条件的值按轴排,最新的现行)。
· 会不会过期(lapses):只有一段状态(某人 2014–2016 在某公司、某台设备的保修到 2025 年 6 月)过了止期才不再成立。
  一件事发生的时间(起止同一天 / 同一个月 / 同一年:参加某次会议、某天完成的测试)过去了,事情照样发生过;另一端是事件的,
  时间说的是事件什么时候发生;起点(事实是主体)跟新旧一样只限有状态会变的实例 —— 概念、属性、过程、文档上的说法
  带的时间多半是资料的检测期 / 统计期,不是状态;兜底关系 related_to 只说两者有关联。这几种都不过期,时间照样带着给
  调用方看(观察时间不代表关系的有效期)。
  建图时定下写在关系 / 事实上;「已结束」(会过期的、止期早于今天)随日期变,检索时现算。
全是规则,不调模型。
"""
from __future__ import annotations

import re
from typing import Any, Iterable

from . import prompts
from .extract import normalize_predicate
from .units import Unit


def _when(r: dict[str, Any]) -> str:
    return str(r.get("valid_from") or r.get("observed_last") or "")


STATEFUL_UPPERS = ("agent", "entity", "part")
EVENT_UPPER = "event"


def is_interval(start: Any, until: Any) -> bool:
    """原文写明的是一段时间:止期是日期,且没写起期或止期晚于起期。起止相同的是一件事发生的时间,不是一段状态。"""
    s, u = str(start or ""), str(until or "")
    return is_date(u) and (not is_date(s) or u > s)


def stateful_keys(entities: Iterable[dict[str, Any]]) -> set[str]:
    """有状态会变的实例(人员与组织、实体、部件,不含类别名):只有它们的关系判新旧、算过期。"""
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
    """事实同一条规则(主体得是有状态的实例),原地写 lapses;返回会过期的条数。"""
    stateful = stateful_keys(entities)
    marked = 0
    for f in facts:
        f["lapses"] = is_interval(f.get("valid_from"), f.get("valid_until")) and str(f.get("subject_key") or "") in stateful
        marked += int(f["lapses"])
    return marked


def relation_currency(relations: list[dict[str, Any]], units_by_id: dict[str, Unit], *,
                      predicates: Iterable[dict[str, Any]] = (), entities: Iterable[dict[str, Any]] | None = None) -> dict[str, int]:
    """原地写 observed_first / observed_last / lapses / superseded_by / superseded_on;返回计数。
    entities 给了就按起点的上层类与类 / 实例筛、按端点是不是事件定会不会过期(见模块说明);不给不筛。"""
    from .vectors import relation_id

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
        if r.get("derived") or str(r.get("predicate") or "") not in single:
            continue
        if stateful is not None and str(r["source_key"]) not in stateful:
            stats["stateless_sources_skipped"] += 1
            continue
        groups.setdefault((str(r["source_key"]), str(r["predicate"])), []).append(r)
    for rows in groups.values():
        if len({str(r["target_key"]) for r in rows}) < 2:
            continue
        stats["single_valued_groups"] += 1
        dated = [r for r in rows if _when(r)]
        if not dated:
            stats["undecided_groups"] += 1
            continue
        top = max(_when(r) for r in dated)
        latest = [r for r in dated if _when(r) == top]
        if len({str(r["target_key"]) for r in latest}) > 1:
            stats["undecided_groups"] += 1          # 最新的时间上就有两个终点:冲突,不判
            continue
        head = latest[0]
        head_id = relation_id(head["source_key"], head["target_key"], head["predicate"])
        for r in dated:
            if r is head or str(r["target_key"]) == str(head["target_key"]) or _when(r) >= top:
                continue
            r["superseded_by"] = head_id
            r["superseded_on"] = top
            stats["superseded"] += 1
    return stats


# ── 检索时的口径(关系与事实共用):现行 / 回看某一天 / 全部 ─────────────────────
# 默认只给现行的:没被更新的说法取代、来源文档不是全都被取代或归档、有效期没过(只看会过期的,lapses)。带 as_of
# 回看某一天:那天已经开始(起点没写就看最早出现在哪天的文档里)、还没结束、取代它的说法那天还没出现;文档状态不看
# (被取代是现在的事,那天它还是现行的)。带 history 全部给,每行标状态。日期按前缀比:只写到年的「2014」与
# 「2014-05-01」比的是年。
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
    """一条关系 / 事实在 day 那天的状态(见上面的口径)。as_of=False 时 day 是今天。"""
    start = str(row.get("valid_from") or row.get("observed_first") or row.get("observed") or "")
    if _after(start, day):
        return STATUS_NOT_STARTED
    if _ended(row, day):
        return STATUS_ENDED
    on = str(row.get("superseded_on") or "")
    if row.get("superseded_by") and ((is_date(on) and not _after(on, day)) or (not as_of and not is_date(on))):
        return STATUS_SUPERSEDED
    if as_of:
        return STATUS_CURRENT

    doc = str(row.get("doc_status") or "")
    if doc:
        return STATUS_DOC_ARCHIVED if doc == "archived" else STATUS_DOC_SUPERSEDED
    if _ended(row, day):
        return STATUS_ENDED
    return STATUS_CURRENT


def next_change_day(rows: Iterable[dict[str, Any]], *, today: str) -> str:
    """Earliest future day when a compiled current projection may need refreshing."""
    from datetime import date, timedelta
    candidates = []
    for row in rows:
        for value in (row.get("valid_from") or row.get("observed_first") or row.get("observed"), row.get("superseded_on")):
            value = str(value or "")
            if is_date(value):
                try:
                    day = date.fromisoformat(value + {4: "-01-01", 7: "-01", 10: ""}[len(value)]).isoformat()
                    if day > today:
                        candidates.append(day)
                except (ValueError, KeyError):
                    pass
        if row.get("lapses") and is_date(str(row.get("valid_until") or "")):
            try:
                day = (date.fromisoformat(as_of_day(row["valid_until"])) + timedelta(days=1)).isoformat()
                if day > today:
                    candidates.append(day)
            except ValueError:
                pass
    return min(candidates, default="")


def cypher_in_effect(var: str, *, observed: str, as_of: bool) -> str:
    """Cypher 条件:变量 var(关系边或事实节点)在 $day 那天算数;observed 是它的观察时间字段
    (关系 observed_first、事实 observed)。日期字段只按前缀比,不是日期的值(版本号轴)不参与比较;止期只对会过期的算。"""
    v = var
    until = (f"(NOT coalesce({v}.lapses, false) OR NOT coalesce({v}.valid_until, '') =~ '[0-9]{{4}}.*' "
             f"OR {v}.valid_until >= left($day, size({v}.valid_until)))")
    start = f"coalesce({v}.valid_from, {v}.{observed}, '')"
    started = f"(NOT {start} =~ '[0-9]{{4}}.*' OR {start} <= left($day, size({start})))"
    replaced = (f"(coalesce({v}.superseded_by, '') <> '' AND coalesce({v}.superseded_on, '') =~ '[0-9]{{4}}.*' "
                f"AND {v}.superseded_on <= left($day, size({v}.superseded_on)))")
    if not as_of:
        replaced = f"(coalesce({v}.superseded_by, '') <> '' AND (NOT coalesce({v}.superseded_on, '') =~ '[0-9]{{4}}.*' OR {v}.superseded_on <= left($day, size({v}.superseded_on))))"
        return f"({started} AND {until} AND NOT {replaced} AND coalesce({v}.doc_status, '') = '')"
    return f"({started} AND {until} AND NOT {replaced})"


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
    """同一口径在 Python 里判一行(向量种子取回来之后用;Qdrant 里只能先按「没被取代、文档没标状态」筛)。"""
    return scope["mode"] == "history" or row_status(row, day=scope["day"], as_of=bool(scope.get("as_of"))) == STATUS_CURRENT


def mark_status(row: dict[str, Any], scope: dict[str, Any]) -> dict[str, Any]:
    row["status"] = row_status(row, day=scope["day"], as_of=bool(scope.get("as_of")))
    return row


def qdrant_current_conditions() -> list[Any]:
    """现行口径能先交给 Qdrant 的那部分:没被取代、来源文档没标状态(有效期过没过要按日期比,取回来再判)。"""
    from qdrant_client.http import models as qm

    return [qm.IsEmptyCondition(is_empty=qm.PayloadField(key="doc_status"))]
