"""整版图的统计(方案第 2 条):导入 Neo4j 时按 graph.json 与单元算好,写在 GraphVersion 节点上,
检索服务的 /graph/stats 直接给、/catalog 摘一份概览。

覆盖面按「涉及几份文档、落在几个周」算,不按提及次数:同一个名字在一份周报里被提二十次,和在二十份周报里
各提一次,分量不一样。另给每个谓词的「起点类型 → 终点类型」组合、别名最多的实体、落到兜底关系 related_to
的比例与最常见的实体对、类型表以外的类型、资料覆盖的时间跨度、核心主体。全是计数,不调模型。
"""
from __future__ import annotations

import datetime as _dt
from collections import Counter
from typing import Any, Iterable

from . import prompts
from .units import Unit

TOP_TYPES = 40
TOP_PREDICATES = 30
TOP_PAIRS = 5
TOP_SUBJECTS = 12
TOP_ALIASES = 20
TOP_UNCLASSIFIED = 15
TOP_PROPERTIES = 20
SUBJECT_UPPERS = ("agent", "entity")


def iso_week(day: str) -> str:
    """YYYY-MM-DD → ISO 年周(2026-W09);只有年或年月的日期落不到周,返回空串。"""
    try:
        y, w, _ = _dt.date.fromisoformat(str(day)[:10]).isocalendar()
    except ValueError:
        return ""
    return f"{y}-W{w:02d}"


def document_dates(units: Iterable[Unit]) -> dict[str, str]:
    """doc_id → 文档日期(建图时按文件名 / 正文开头定下的,见 temporal.document_axis;没有就是空串)。"""
    return {u.doc_id: str(getattr(u, "doc_date", "") or "") for u in units}


def entity_coverage(entity: dict[str, Any], doc_dates: dict[str, str]) -> dict[str, Any]:
    """一个实体出现在几份文档(完整的,不受负载里来源只存前 16 个的限制)、几个周,最早 / 最晚的文档日期。"""
    docs = [str(d) for d in (entity.get("doc_ids") or [])]
    dates = sorted({doc_dates.get(d, "") for d in docs} - {""})
    return {"doc_count": len(docs), "week_count": len({w for w in map(iso_week, dates) if w}),
            "first_seen": dates[0] if dates else "", "last_seen": dates[-1] if dates else ""}


def graph_stats(graph: dict[str, Any], units: Iterable[Unit]) -> dict[str, Any]:
    """整版图的统计(见模块说明)。graph 是 graph.json 的内容;类型表取建图时用的那份(graph["schema"])。"""
    doc_dates = document_dates(units)
    entities = list(graph.get("entities") or [])
    relations = list(graph.get("relations") or [])
    specs = list(graph.get("specs") or [])
    known = {str(t).casefold() for t in ((graph.get("schema") or {}).get("entity_types") or [])}
    dated = sorted(d for d in doc_dates.values() if d)
    by_key = {e["key"]: e for e in entities}
    out: dict[str, Any] = {
        "graph_version": graph.get("graph_version"), "built_at": graph.get("built_at"),
        "schema_version_id": (graph.get("schema") or {}).get("version_id"),
        "documents": len(doc_dates),
        "time_span": {"from": dated[0] if dated else "", "to": dated[-1] if dated else "", "dated_documents": len(dated),
                      "by_year": dict(sorted(Counter(d[:4] for d in dated).items()))},
        "entities": {"total": len(entities),
                     "instances": sum(1 for e in entities if e.get("kind") != "class"),
                     "classes": sum(1 for e in entities if e.get("kind") == "class"),
                     "with_external_ids": sum(1 for e in entities if e.get("external_ids")),
                     "by_upper": dict(Counter(str(e.get("upper") or "entity") for e in entities).most_common())},
    }
    # 类型的覆盖面:实体数、涉及的文档数与周数、最早 / 最晚
    per_type: dict[str, dict[str, Any]] = {}
    for e in entities:
        t = str(e.get("type") or "")
        row = per_type.setdefault(t, {"type": t, "upper": str(e.get("upper") or "entity"), "entities": 0, "classes": 0,
                                      "_docs": set(), "in_schema": (not known) or bool(e.get("exact")) or t.casefold() in known})
        row["entities"] += 1
        row["classes"] += int(e.get("kind") == "class")
        row["_docs"].update(str(d) for d in (e.get("doc_ids") or []))
    type_rows = []
    for row in per_type.values():
        docs = row.pop("_docs")
        dates = sorted({doc_dates.get(d, "") for d in docs} - {""})
        type_rows.append({**row, "docs": len(docs), "weeks": len({w for w in map(iso_week, dates) if w}),
                          "first_seen": dates[0] if dates else "", "last_seen": dates[-1] if dates else ""})
    type_rows.sort(key=lambda r: (-r["docs"], -r["entities"], r["type"]))
    out["types"] = type_rows[:TOP_TYPES]
    # 谓词:边数、涉及的文档数、最常见的「起点类型 → 终点类型」组合;兜底关系的占比
    p_edges: Counter = Counter()
    p_docs: dict[str, set[str]] = {}
    p_pairs: dict[str, Counter] = {}
    fallback_pairs: Counter = Counter()
    for r in relations:
        p = str(r.get("predicate") or prompts.DEFAULT_PREDICATE)
        src, tgt = by_key.get(r.get("source_key")) or {}, by_key.get(r.get("target_key")) or {}
        p_edges[p] += 1
        p_docs.setdefault(p, set()).update(str(d) for d in (r.get("doc_ids") or []))
        p_pairs.setdefault(p, Counter())[(str(src.get("type") or "-"), str(tgt.get("type") or "-"))] += 1
        if p == prompts.DEFAULT_PREDICATE:
            fallback_pairs[(str(src.get("title") or ""), str(tgt.get("title") or ""))] += int(r.get("evidence") or 1)
    fallback = p_edges.get(prompts.DEFAULT_PREDICATE, 0)
    out["relations"] = {
        "total": len(relations), "related_to": fallback,
        "related_to_share": round(fallback / len(relations), 3) if relations else 0.0,
        "superseded": sum(1 for r in relations if r.get("superseded_by")),
        "non_current_documents": sum(1 for r in relations if r.get("doc_status")),
        "with_period": sum(1 for r in relations if r.get("valid_from") or r.get("valid_until")),
        "lapsing": sum(1 for r in relations if r.get("lapses")),
    }
    out["predicates"] = [
        {"predicate": p, "edges": n, "docs": len(p_docs.get(p, ())),
         "pairs": [{"source_type": a, "target_type": b, "count": c} for (a, b), c in p_pairs[p].most_common(TOP_PAIRS)]}
        for p, n in p_edges.most_common(TOP_PREDICATES)]
    # 事实:总数、被取代的、来源文档不现行的、叙述 / 表格来源各多少、最常见的属性(涉及几个主体)
    prop_facts: Counter = Counter()
    prop_subjects: dict[str, set[str]] = {}
    for f in specs:
        c = str(f.get("concept") or f.get("property") or "")
        if c:
            prop_facts[c] += 1
            prop_subjects.setdefault(c, set()).add(str(f.get("subject_key") or f.get("subject") or ""))
    out["facts"] = {
        "total": len(specs), "superseded": sum(1 for f in specs if f.get("superseded_by")),
        "non_current_documents": sum(1 for f in specs if f.get("doc_status")),
        "lapsing": sum(1 for f in specs if f.get("lapses")),
        "narrative": sum(1 for f in specs if f.get("source_kind") == "narrative"),
        "table": sum(1 for f in specs if f.get("source_kind") == "table"),
        "properties": [{"property": c, "facts": n, "subjects": len(prop_subjects.get(c, ()))}
                       for c, n in prop_facts.most_common(TOP_PROPERTIES)],
    }
    # 核心主体:先取这个库场景画像里的主体类型(模式阶段归纳的,如受检者、项目),各类型按 pagerank 轮流取(每类先取
    # 第一名,再各取第二名……,一类实体多、排名又高时不至于把别的主体类型挤掉),再用人员与组织、实体两个上层类里的
    # 实例补足;都不含文档内作用域的、样板里的、参考文献。只按 pagerank 会让高频的辅助概念占满,
    # 只按类型优先级又可能遗漏其他主体类型,所以同时考虑类型覆盖和相关度。
    def eligible(e: dict[str, Any]) -> bool:
        return not e.get("scope") and not e.get("boilerplate") and not e.get("reference")

    def rank(e: dict[str, Any]) -> tuple:
        return (-float(e.get("pagerank") or 0.0), -len(e.get("doc_ids") or []), str(e.get("title") or ""))

    subject_types = list(dict.fromkeys(str(t).casefold() for t in ((graph.get("profile") or {}).get("subject_types") or [])))
    by_type = {t: sorted((e for e in entities if eligible(e) and str(e.get("type") or "").casefold() == t), key=rank)
               for t in subject_types}
    typed = [rows[i] for i in range(TOP_SUBJECTS) for rows in by_type.values() if i < len(rows)][:TOP_SUBJECTS]
    picked = {id(e) for e in typed}
    rest = sorted((e for e in entities if eligible(e) and id(e) not in picked and e.get("kind") != "class"
                   and str(e.get("upper") or "entity") in SUBJECT_UPPERS), key=rank)
    out["subjects"] = [{"title": e.get("title"), "type": e.get("type"), "upper": e.get("upper"),
                        **entity_coverage(e, doc_dates)} for e in (typed + rest)[:TOP_SUBJECTS]]
    aliased = sorted((e for e in entities if e.get("aliases")),
                     key=lambda e: (-len(e.get("aliases") or []), str(e.get("title") or "")))
    out["alias_clusters"] = [{"title": e.get("title"), "type": e.get("type"), "count": len(e.get("aliases") or []),
                              "aliases": list(e.get("aliases") or [])[:10]} for e in aliased[:TOP_ALIASES]]
    # 没归好类的:类型表以外的类型(规则抽取的类型不算)、经兜底关系相连证据最多的实体对
    outside = Counter(str(e.get("type") or "") for e in entities
                      if known and not e.get("exact") and str(e.get("type") or "").casefold() not in known)
    out["unclassified"] = {
        "types_outside_schema": [{"type": t, "entities": n} for t, n in outside.most_common(TOP_UNCLASSIFIED)],
        "frequent_related_to": [{"source": a, "target": b, "evidence": n}
                                for (a, b), n in fallback_pairs.most_common(TOP_UNCLASSIFIED)],
    }
    return out


def overview(stats: dict[str, Any], *, types: int = 12, predicates: int = 10, subjects: int = 8) -> dict[str, Any]:
    """/catalog 里每个库的概览:会话开头让智能体知道这个库讲什么、有哪些类型和关系、覆盖哪段时间。"""
    if not stats:
        return {}
    return {
        "documents": stats.get("documents"),
        "time_span": {k: (stats.get("time_span") or {}).get(k) for k in ("from", "to", "dated_documents")},
        "entities": (stats.get("entities") or {}).get("total"),
        "facts": (stats.get("facts") or {}).get("total"),
        "related_to_share": (stats.get("relations") or {}).get("related_to_share"),
        "types": [{k: t.get(k) for k in ("type", "upper", "entities", "docs")} for t in (stats.get("types") or [])[:types]],
        "predicates": [{"predicate": p.get("predicate"), "edges": p.get("edges"),
                        "pair": (f"{p['pairs'][0]['source_type']} -> {p['pairs'][0]['target_type']}" if p.get("pairs") else "")}
                       for p in (stats.get("predicates") or [])[:predicates]],
        "subjects": [s.get("title") for s in (stats.get("subjects") or [])[:subjects]],
    }
