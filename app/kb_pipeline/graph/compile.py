"""View layer (plan 5.C): the structure layer (entities / relations / facts / concepts / reconciliation) is
projected deterministically into four kinds of pages, written as markdown under wiki/ in the workspace and
added to graph.json as pages -> the page vector collection, competing in the same arena as the chunks.

Pages are a projection, not the truth: neither the evidence layer (chunks) nor the structure layer is changed;
every graph build regenerates them as a whole (the compile step of LLM Wiki, without its manual upkeep). Four
templates that degrade gracefully in any scenario (no axis means no timeline pages; no subject types means
subjects are picked by frequency / centrality):
  subject page   one subject's facts across the KB (grouped by concept, ordered by axis), its relations
                 (extension predicates first) and its sources
  timeline page  the series of one property concept of one subject along the axis (reconcile's series), with
                 the first / last values and the trend
  source page    one document: axis value, subjects, fact count, conclusion excerpts, conflicts
  index page     the catalogue of subjects, timelines, sources and concepts
The subject page may get a narrative written by the summary model (capped at NARRATE_MAX_PAGES; on failure it
falls back to the deterministic summary); everything else is pure projection.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from . import prompts
from .facts import comparable_number, conditions_text, symbol_only_value, values_text, when_text
from .llm import ChatClient, LLMCallError
from .reconcile import fact_axis, series_of, unit_of
from .temporal import axis_sort_key, axis_value_kind

SUBJECT_MAX_PAGES = 80
# 主体页与文档主体只从这两类里挑:人员与组织(受检者、客户、合同方)和独立存在的东西(产品、项目、系统)
SUBJECT_UPPERS = ("agent", "entity")
SUBJECT_MIN_FACTS = 3           # 没被画像列为主体类型的实体,事实够多也给一页
SUBJECT_FALLBACK = 20           # 既没主体类型也没事实的库:按频次 / 中心度挑这么多
CONCEPTS_PER_PAGE = 40
ROWS_PER_CONCEPT = 12
RELATIONS_PER_PAGE = 30
# 组成部分(10-24 改版二十二):主体页按组成关系汇总各部分名下的事实。事实挂到各部分名下以后(改版十六、十八),
# 问「X 由哪些部分组成、各管什么」时,上级实体的主体页只剩它自己的几条。一个部分一行、紧跟概述:放在事实表后面的话,
# 大页面上要到第一万字才出现,叙述输入(NARRATE_INPUT_CHARS)、检索返回的页面正文和页面向量都截不到
PARTS_PER_PAGE = 10
FACTS_PER_PART = 3
PART_FACT_CHARS = 80
TIMELINE_MAX_PAGES = 400
INDEX_ROWS = 200
POINT_IDS_PER_PAGE = 32
NARRATE_MAX_PAGES = 40
NARRATE_MIN_FACTS = 2
NARRATE_INPUT_CHARS = 6000
CONCLUSION_EXCERPT_CHARS = 700
# the X in the stage string "Compiling view pages · X": the console splits the progress bar by these literals
# (app.js GRAPH_STAGES)
STAGE_PARTS = {"subjects": "subjects", "timelines": "timelines", "sources": "sources", "narrate": "narration"}

_LABELS = {
    "zh": {
        "overview": "概述", "facts": "事实", "relations": "关系", "extensions": "延伸", "sources": "来源",
        "when": "时间 / 版本", "value": "值", "flag": "标记", "ref": "参考范围", "conditions": "条件", "source": "来源",
        "type": "类型", "aliases": "别名", "docs": "文档", "facts_n": "事实", "relations_n": "关系", "more": "另有 {n} 条未列出",
        "timeline": "时间线", "first": "起点", "last": "终点", "trend_up": "上升", "trend_down": "下降", "trend_flat": "持平",
        "trend_mixed": "有变动", "out_of_range": "越界 {n} 次", "axis": "轴", "subjects": "主体", "units": "单元",
        "conclusion": "结论段", "conflicts": "冲突", "index": "索引", "concepts": "属性概念", "documents": "文档",
        "subject_pages": "主体页", "timeline_pages": "时间线页", "source_pages": "来源页", "no_axis": "(无轴)",
        "kinds": "正文 {body} · 结论 {conclusion} · 目录 / 列表 {listing} · 版式 {boilerplate}",
        "series_summary": "{concept}:{first_when} {first} → {last_when} {last}({trend},{n} 个点)",
        "concept_line": "{label}:{n} 条事实,{docs} 份文档",
        "parts": "组成部分", "parts_n": "个组成部分", "property": "属性", "part_ref": "{title}({n} 条事实,另有主体页)", "list_sep": "、",
        "evidence_conflict": "证据冲突", "ambiguous_source": "来源值未核实", "confidence": "置信度",
        "text_value": "文字读数", "model_value": "模型读数", "unresolved": "待核实",
        "quality_count": "抽取事实待核实: {n} 条；原始记录保留，不纳入趋势，部分条目可能未在本页展开。",
        "quality_omitted": "另有 {n} 条待核实事实未纳入趋势",
        "quality_no_trend": "{n} 条事实待核实；其余有效时间点不足，不生成趋势",
    },
    "en": {
        "overview": "Overview", "facts": "Facts", "relations": "Relationships", "extensions": "Extensions", "sources": "Sources",
        "when": "When", "value": "Value", "flag": "Flag", "ref": "Reference", "conditions": "Conditions", "source": "Source",
        "type": "Type", "aliases": "Aliases", "docs": "documents", "facts_n": "facts", "relations_n": "relationships", "more": "{n} more not listed",
        "timeline": "Timeline", "first": "First", "last": "Last", "trend_up": "rising", "trend_down": "falling", "trend_flat": "flat",
        "trend_mixed": "changing", "out_of_range": "{n} out of range", "axis": "Axis", "subjects": "Subjects", "units": "Units",
        "conclusion": "Conclusions", "conflicts": "Conflicts", "index": "Index", "concepts": "Properties", "documents": "Documents",
        "subject_pages": "Subject pages", "timeline_pages": "Timeline pages", "source_pages": "Source pages", "no_axis": "(no axis)",
        "kinds": "body {body} · conclusion {conclusion} · listing {listing} · boilerplate {boilerplate}",
        "series_summary": "{concept}: {first_when} {first} → {last_when} {last} ({trend}, {n} points)",
        "concept_line": "{label}: {n} facts, {docs} documents",
        "parts": "Parts", "parts_n": "parts", "property": "Property", "part_ref": "{title} ({n} facts, own page)", "list_sep": ", ",
        "evidence_conflict": "Evidence conflict", "ambiguous_source": "Unverified source value", "confidence": "Confidence",
        "text_value": "Text reading", "model_value": "Model reading", "unresolved": "Unresolved",
        "quality_count": "Extracted facts awaiting verification: {n}; original records retained and excluded from trends. Some may not be expanded on this page.",
        "quality_omitted": "{n} unresolved facts excluded from the trend",
        "quality_no_trend": "{n} unresolved facts; insufficient remaining time points for a trend",
    },
}
_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def labels_for(language: str) -> dict[str, str]:
    lang = str(language or "").strip().casefold()
    return _LABELS["zh"] if lang.startswith(("chinese", "zh", "中文", "汉")) else _LABELS["en"]


def page_id(kind: str, key: str) -> str:
    return "pg" + hashlib.sha256(f"{kind}|{key}".encode("utf-8")).hexdigest()[:16]


def _slug(text: str, limit: int = 60) -> str:
    s = re.sub(r"[^\w\u4e00-\u9fff.-]+", "-", str(text or "")).strip("-").casefold()
    return (s or "page")[:limit]


def _path_tag(key: Any) -> str:
    """A stable identity suffix in the path: same-named subjects / timelines with the same slug / same-named files
    in different directories no longer land in the same file (Codex review F11)."""
    return hashlib.sha1(str(key or "").encode("utf-8")).hexdigest()[:8]


def _numeric(fact: dict[str, Any]) -> float | None:
    """The number used for trends and series endpoints: only scalars without a comparator count; the first number
    is no longer searched out of the value / typ / max / min strings (Codex review F04: V_CC + 0.5 -> 0.5,
    20/25/45 -> 20, 2.7 to 3.6 -> 2.7, TI-RADS category 3 -> 3 all came from that path)."""
    return comparable_number(fact)


def _ref_text(fact: dict[str, Any]) -> str:
    lo, hi = str(fact.get("ref_min") or ""), str(fact.get("ref_max") or "")
    if lo and hi:
        return f"{lo}~{hi}"
    return lo or hi


def _md_cell(text: Any) -> str:
    return str(text or "").replace("|", "\\|").replace("\n", " ").strip()


def _unresolved(f: dict[str, Any]) -> bool:
    return bool(f.get("evidence_conflict")) or f.get("quality") == "ambiguous_source"


def _quality_note(f: dict[str, Any], L: dict[str, str], *, readings: bool = True) -> str:
    notes = []
    conflict = f.get("evidence_conflict")
    if conflict:
        notes.append(L["evidence_conflict"])
        if readings and isinstance(conflict, dict):
            for key in ("text_value", "model_value"):
                if conflict.get(key) not in (None, ""):
                    notes.append(f"{L[key]}={conflict[key]}")
    if f.get("quality") == "ambiguous_source":
        notes.append(L["ambiguous_source"])
    if f.get("confidence"):
        notes.append(f"{L['confidence']}={f['confidence']}")
    return "; ".join(notes)


def _quality_summary(facts: list[dict[str, Any]], L: dict[str, str]) -> str:
    pending = {str(f.get("id") or id(f)) for f in facts if _unresolved(f)}
    return L["quality_count"].format(n=len(pending)) if pending else ""


def _fact_row(f: dict[str, Any], *, rel_paths: dict[str, str], L: dict[str, str] | None = None,
              narrative: bool = False) -> str:
    L = L or labels_for("en")
    src = str(f.get("rel_path") or rel_paths.get(str(f.get("doc_id") or ""), "") or "")
    # Retain both readings in the evidence table. The narrative model receives the
    # unresolved status, not a disputed value it could turn into a settled claim.
    value = L["unresolved"] if narrative and _unresolved(f) else values_text(f) or str(f.get("value") or "")
    flag = "; ".join(x for x in (str(f.get("flag") or ""), _quality_note(f, L, readings=not narrative)) if x)
    return "| " + " | ".join(_md_cell(x) for x in (
        when_text(f) or "-", value, flag, _ref_text(f),
        conditions_text(f.get("conditions")), Path(src).name if src else "")) + " |"


def _part_fact_text(f: dict[str, Any], *, L: dict[str, str] | None = None, narrative: bool = False) -> str:
    """组成部分一行里的一条事实:属性 值 标记(条件; 时间)。"""
    L = L or labels_for("en")
    value = L["unresolved"] if narrative and _unresolved(f) else values_text(f) or str(f.get("value") or "")
    text = f"{f.get('property') or f.get('concept') or ''} {value}".strip()
    if str(f.get("flag") or "").strip():
        text += f" {f['flag']}"
    text = " ".join(text.split())
    if len(text) > PART_FACT_CHARS:
        text = text[:PART_FACT_CHARS] + "…"
    extra = "; ".join(x for x in (conditions_text(f.get("conditions")), when_text(f)) if x)
    quality = _quality_note(f, L, readings=not narrative)
    return text + (f"({extra[:40]})" if extra else "") + (f" [{quality}]" if quality else "")


def parts_index(relations: list[dict[str, Any]], entities: list[dict[str, Any]]) -> dict[str, list[str]]:
    """组成关系 → {整体: [部分]}:两个方向都认(ontology.composition_ends),版式、引用关系不算,自己连自己不算。
    部分只认实例:类别名(「技能」「模块」)不是哪一个具体的组成部分。"""
    from ..ontology import composition_ends

    by_key = {str(e["key"]): e for e in entities}
    out: dict[str, list[str]] = defaultdict(list)
    for r in relations:
        if r.get("boilerplate") or r.get("reference"):
            continue
        ends = composition_ends(r.get("predicate"), str(r.get("source_key") or ""), str(r.get("target_key") or ""))
        if ends is None:
            continue
        whole, part = ends
        e = by_key.get(part)
        if not whole or whole == part or e is None or e.get("kind") == "class" or e.get("boilerplate") or e.get("reference"):
            continue
        if part not in out[whole]:
            out[whole].append(part)
    return dict(out)


def _parts_section(parts: list[dict[str, Any]], facts_by_subject: dict[str, list[dict[str, Any]]], paged: set[str], *,
                   L: dict[str, str]) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """主体页的「组成部分」一节:一个部分一行,带它名下最多 FACTS_PER_PART 条事实(有标记的在前,其余按轴从新到旧),事实多的
    部分在前。另有主体页的部分只列名字和事实数:组成关系里常有「平台 包含 某个独立产品」这种噪声,把一个上百条事实的实体
    整段搬进来会淹没这一页。返回 (页面行, 叙述输入行, 列出的事实)。"""
    with_facts = sorted((p for p in parts if facts_by_subject.get(str(p["key"]))),
                        key=lambda p: (-len(facts_by_subject[str(p["key"])]), str(p.get("title") or "")))
    if not with_facts:
        return [], [], []
    lines = [f"## {L['parts']}", ""]
    narrate = list(lines)
    shown_facts: list[dict[str, Any]] = []
    refs: list[str] = []
    listed = 0
    for p in with_facts:
        rows = facts_by_subject[str(p["key"])]
        if str(p["key"]) in paged:
            refs.append(L["part_ref"].format(title=p.get("title") or p["key"], n=len(rows)))
            continue
        if listed >= PARTS_PER_PAGE:
            continue
        listed += 1
        # 三次稳定排序:有标记的在前;再按轴的种类(日期、版本、没有轴);同一种里从新到旧
        ordered = sorted(rows, key=lambda r: axis_sort_key(fact_axis(r)), reverse=True)
        ordered = sorted(ordered, key=lambda r: axis_sort_key(fact_axis(r))[0])
        ordered = sorted(ordered, key=lambda r: 0 if str(r.get("flag") or "").strip() else 1)
        shown = ordered[:FACTS_PER_PART]
        head = f"- {p.get('title') or p['key']}" + (f" ({p.get('type')})" if p.get("type") else "") + ": "
        more = [L["more"].format(n=len(rows) - FACTS_PER_PART)] if len(rows) > FACTS_PER_PART else []
        lines.append(head + "; ".join([_part_fact_text(r, L=L) for r in shown] + more))
        kept = [r for r in shown if not symbol_only_value(r)]          # 与事实一节同一口径:符号值不进叙述输入
        if kept:
            narrate.append(head + "; ".join([_part_fact_text(r, L=L, narrative=True) for r in kept] + more))
        shown_facts += shown
    if refs:
        line = "- " + L["list_sep"].join(refs)
        lines.append(line)
        narrate.append(line)
    return lines + [""], narrate + [""], shown_facts


def _facts_by_subject(facts: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for f in facts:
        keys = list(f.get("subject_keys") or ([f["subject_key"]] if f.get("subject_key") else []))
        for k in dict.fromkeys(str(k) for k in keys if k):
            out[k].append(f)
    return out


def select_subjects(entities: list[dict[str, Any]], facts_by_subject: dict[str, list[dict[str, Any]]],
                    profile: dict[str, Any], *, limit: int = SUBJECT_MAX_PAGES) -> list[dict[str, Any]]:
    """Who gets a subject page: entities of the profile's subject_types and entities with enough facts; when there
    are none, pick by frequency / centrality."""
    subject_types = {str(t).casefold() for t in (profile.get("subject_types") or [])}
    pool = [e for e in entities
            if not e.get("scope") and not e.get("reference") and not e.get("boilerplate")
            and e.get("kind") != "class" and str(e.get("upper") or "entity") in SUBJECT_UPPERS]

    def score(e: dict[str, Any]) -> float:
        return 3.0 * len(facts_by_subject.get(e["key"], ())) + float(e.get("frequency") or 0) \
            + 2.0 * len(e.get("doc_ids") or []) + 50.0 * float(e.get("pagerank") or 0)

    typed = [e for e in pool if subject_types and any(str(t).casefold() in subject_types
                                                      for t in (e.get("type_path") or [e.get("type") or ""]))]
    rich = [e for e in pool if len(facts_by_subject.get(e["key"], ())) >= SUBJECT_MIN_FACTS]
    chosen = {e["key"]: e for e in typed + rich}
    if not chosen:
        chosen = {e["key"]: e for e in sorted(pool, key=score, reverse=True)[:SUBJECT_FALLBACK]}
    return sorted(chosen.values(), key=score, reverse=True)[:limit]


def _trend(rows: list[dict[str, Any]], L: dict[str, str]) -> str:
    pending = sum(_unresolved(r) for r in rows)
    rows = [r for r in rows if not _unresolved(r)]
    if pending and len({fact_axis(r) for r in rows}) < 2:
        return L["quality_no_trend"].format(n=pending)
    numeric_rows = [r for r in rows if _numeric(r) is not None]
    # the trend compares axis values: one value stated repeatedly on the same axis (once by each of two adjacent
    # units, copied by several documents) is a single point; otherwise the difference between the repeated rows
    # is 0, a monotonic series reads as "changing" and one with numbers on a single axis reads as "flat"
    by_axis: dict[str, list[float]] = {}
    for r in numeric_rows:
        seen = by_axis.setdefault(fact_axis(r), [])
        num = _numeric(r)
        if not any(abs(num - v) < 1e-9 for v in seen):
            seen.append(num)
    if len(by_axis) < 2:
        return L["trend_mixed"] if len({values_text(r) for r in rows}) > 1 else L["trend_flat"]
    if len({unit_of(r) for r in numeric_rows if unit_of(r)}) > 1:
        return L["trend_mixed"]            # numbers in two different real units cannot be compared directly
    if any(len(values) > 1 for values in by_axis.values()):
        return L["trend_mixed"]            # several different values on one axis: there is no single trend
    nums = [by_axis[axis][0] for axis in sorted(by_axis, key=axis_sort_key)]
    diffs = [b - a for a, b in zip(nums, nums[1:])]
    if all(d > 0 for d in diffs):
        return L["trend_up"]
    if all(d < 0 for d in diffs):
        return L["trend_down"]
    if all(abs(d) < 1e-9 for d in diffs):
        return L["trend_flat"]
    return L["trend_mixed"]


def _series_line(rows: list[dict[str, Any]], L: dict[str, str]) -> str:
    pending = sum(_unresolved(r) for r in rows)
    rows = [r for r in rows if not _unresolved(r)]
    if pending and len({fact_axis(r) for r in rows}) < 2:
        return L["quality_no_trend"].format(n=pending)
    # 起点 / 终点优先取带数值的行:同一轴上「6.22 mmol/L」与汇总表的「增高」并存时,说数值
    numeric = [r for r in rows if _numeric(r) is not None]
    first = next((r for r in rows if _numeric(r) is not None), rows[0]) if numeric else rows[0]
    last = next((r for r in reversed(rows) if _numeric(r) is not None), rows[-1]) if numeric else rows[-1]
    line = L["series_summary"].format(
        concept=str(first.get("concept") or first.get("property") or ""), first_when=when_text(first) or "-",
        first=values_text(first), last_when=when_text(last) or "-", last=values_text(last), trend=_trend(rows, L), n=len(rows))
    return line + ("; " + L["quality_omitted"].format(n=pending) if pending else "")


def build_subject_page(entity: dict[str, Any], facts: list[dict[str, Any]], relations: list[dict[str, Any]],
                       *, L: dict[str, str], documents: dict[str, dict[str, Any]], mention_points: dict[str, list[str]],
                       profile: dict[str, Any], entity_titles: dict[str, str], parts: list[dict[str, Any]] | None = None,
                       facts_by_subject: dict[str, list[dict[str, Any]]] | None = None, paged: set[str] | None = None) -> dict[str, Any]:
    """parts:组成部分实体(parts_index);facts_by_subject:各部分的事实;paged:另有主体页的实体键。"""
    key = str(entity["key"])
    title = str(entity.get("title") or key)
    rel_paths = {d: str(v.get("rel_path") or "") for d, v in documents.items()}
    part_lines, part_narrate, part_facts = _parts_section(parts or [], facts_by_subject or {}, paged or set(), L=L)
    n_parts = sum(1 for p in parts or [] if (facts_by_subject or {}).get(str(p["key"])))
    lines = [f"# {title}", ""]
    path = [str(t) for t in (entity.get("type_path") or [])[1:]]
    meta = [f"{L['type']}: {entity.get('type') or '-'}" + (f" ({' > '.join(path)})" if path else "")]
    if entity.get("aliases"):
        meta.append(f"{L['aliases']}: {', '.join(str(a) for a in entity['aliases'][:8])}")
    docs = list(dict.fromkeys([str(d) for d in (entity.get("doc_ids") or [])]
                + [str(f["doc_id"]) for f in facts + part_facts if f.get("doc_id")]
                + [str(d) for r in relations for d in (r.get("doc_ids") or [])]
                + [str(d) for p in parts or [] for d in (p.get("doc_ids") or [])]))
    meta.append(f"{len(docs)} {L['docs']} · {len(facts)} {L['facts_n']} · {len(relations)} {L['relations_n']}"
                + (f" · {n_parts} {L['parts_n']}" if n_parts else ""))
    lines += [" · ".join(meta), "", f"## {L['overview']}", "", str(entity.get("description") or "").strip() or "-", ""]
    quality_note = _quality_summary(facts + part_facts, L)
    if quality_note:
        lines += [quality_note, ""]
    narrate_lines = list(lines)          # 给叙述模型的输入:同页面正文,但去掉取值只是符号的事实(symbol_only_value)
    lines += part_lines
    narrate_lines += part_narrate

    by_concept: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for f in facts:
        by_concept[str(f.get("concept") or f.get("property") or f.get("symbol") or "")].append(f)
    concept_keys: list[str] = []
    series_lines: list[str] = []
    if by_concept:
        lines += [f"## {L['facts']}", ""]
        narrate_lines += [f"## {L['facts']}", ""]
        # flagged (out-of-range) concepts first, then those with the most facts: the subject page leads with what
        # deserves attention
        ordered = sorted(by_concept.items(), key=lambda kv: (-sum(1 for r in kv[1] if str(r.get("flag") or "").strip()), -len(kv[1]), kv[0]))
        for label, rows in ordered[:CONCEPTS_PER_PAGE]:
            rows = sorted(rows, key=lambda r: (axis_sort_key(fact_axis(r)), str(r.get("rel_path") or ""), str(r.get("section") or "")))
            ck = next((str(r.get("concept_key")) for r in rows if r.get("concept_key")), "")
            if ck and ck not in concept_keys:
                concept_keys.append(ck)
            unit = next((str(r.get("unit") or "") for r in rows if r.get("unit")), "")
            head = [f"### {label}" + (f" ({unit})" if unit else ""), "",
                    f"| {L['when']} | {L['value']} | {L['flag']} | {L['ref']} | {L['conditions']} | {L['source']} |",
                    "|---|---|---|---|---|---|"]
            # a series line only covers axis values of one kind: dates and versions have no order between them,
            # so one cannot be the start and the other the end
            by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for r in rows:
                by_kind[axis_value_kind(fact_axis(r))].append(r)
            for kind, part in by_kind.items():
                if not kind or len({fact_axis(r) for r in part}) < 2:
                    continue
                line = _series_line(part, L)
                flagged = sum(1 for r in part if not _unresolved(r) and str(r.get("flag") or "").strip())
                series_lines.append(line + ("," + L["out_of_range"].format(n=flagged) if flagged else ""))
            shown = rows[:ROWS_PER_CONCEPT]
            more = [L["more"].format(n=len(rows) - ROWS_PER_CONCEPT)] if len(rows) > ROWS_PER_CONCEPT else []
            lines += head + [_fact_row(r, rel_paths=rel_paths, L=L) for r in shown] + more + [""]
            # 叙述输入里去掉取值只是符号的事实:摘要模型会把孤立的「-」读成阳性发现(2026-09-29 抽查);页面正文和 specs 照留
            kept = [r for r in shown if not symbol_only_value(r)]
            if kept:
                narrate_lines += head + [_fact_row(r, rel_paths=rel_paths, L=L, narrative=True) for r in kept] + more + [""]
        if len(ordered) > CONCEPTS_PER_PAGE:
            lines += [L["more"].format(n=len(ordered) - CONCEPTS_PER_PAGE), ""]
            narrate_lines += [L["more"].format(n=len(ordered) - CONCEPTS_PER_PAGE), ""]

    tail_start = len(lines)              # relations, extensions and sources after the facts are the same in both inputs
    ext = {str(p).casefold() for p in (profile.get("extension_predicates") or [])}
    rels = sorted(relations, key=lambda r: -float(r.get("weight") or 0))
    ext_rows = [r for r in rels if str(r.get("predicate") or "").casefold() in ext]
    other_rows = [r for r in rels if r not in ext_rows]

    def rel_line(r: dict[str, Any]) -> str:
        if str(r.get("source_key")) == key:
            arrow, other = "→", entity_titles.get(str(r.get("target_key")), str(r.get("target") or ""))
        else:
            arrow, other = "←", entity_titles.get(str(r.get("source_key")), str(r.get("source") or ""))
        desc = str(r.get("description") or "").strip().replace("\n", " ")
        return f"- {r.get('predicate')} {arrow} {other}" + (f": {desc[:240]}" if desc else "")

    if ext_rows:
        lines += [f"## {L['extensions']}", ""] + [rel_line(r) for r in ext_rows[:RELATIONS_PER_PAGE]] + [""]
    if other_rows:
        lines += [f"## {L['relations']}", ""] + [rel_line(r) for r in other_rows[:RELATIONS_PER_PAGE]]
        if len(other_rows) > RELATIONS_PER_PAGE:
            lines.append(L["more"].format(n=len(other_rows) - RELATIONS_PER_PAGE))
        lines.append("")
    if docs:
        lines += [f"## {L['sources']}", ""]
        for d in sorted(docs, key=lambda d: axis_sort_key(str((documents.get(d) or {}).get("value") or ""))):
            info = documents.get(d) or {}
            axis = str(info.get("value") or "")
            lines.append(f"- {info.get('rel_path') or d}" + (f" ({axis})" if axis else ""))
        lines.append("")
    series_lines.sort(key=lambda line: (0 if L["out_of_range"].split("{")[0] in line else 1))    # 越界的序列先说
    summary_bits = [str(entity.get("description") or "").strip().split("\n")[0][:300], quality_note] + series_lines[:6]
    summary = "\n".join(b for b in summary_bits if b)
    point_ids = list(dict.fromkeys(list(mention_points.get(key) or []) + [p for f in facts + part_facts for p in (f.get("point_ids") or [])]))[:POINT_IDS_PER_PAGE]
    return {
        "id": page_id("subject", key), "kind": "subject", "title": title, "summary": summary, "text": "\n".join(lines).strip() + "\n",
        "quality_note": quality_note,
        "narrate_text": "\n".join(narrate_lines + lines[tail_start:]).strip() + "\n",     # 只给 narrate_pages 用,不落盘
        "path": f"subjects/{_slug(title)}--{_path_tag(key)}.md", "entity_keys": [key], "concept_keys": concept_keys, "doc_ids": docs,
        "point_ids": point_ids, "spec_ids": [str(f.get("id")) for f in facts + part_facts if f.get("id")][:200],
        "facts": len(facts), "relations": len(relations), "series": series_lines, "parts": n_parts, "part_facts": len(part_facts),
    }


def build_timeline_page(series_key: str, rows: list[dict[str, Any]], *, L: dict[str, str],
                        documents: dict[str, dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    subject = str(first.get("subject") or "")
    concept = str(first.get("concept") or first.get("property") or first.get("symbol") or "")
    unit = next((str(r.get("unit") or "") for r in rows if r.get("unit")), "")
    rel_paths = {d: str(v.get("rel_path") or "") for d, v in documents.items()}
    title = f"{subject} · {concept}"
    flags = sum(1 for r in rows if not _unresolved(r) and str(r.get("flag") or "").strip())
    line = _series_line(rows, L)
    if flags:
        line += "," + L["out_of_range"].format(n=flags)
    lines = [f"# {title}" + (f" ({unit})" if unit else ""), "", line, "",
             f"| {L['when']} | {L['value']} | {L['flag']} | {L['ref']} | {L['conditions']} | {L['source']} |", "|---|---|---|---|---|---|"]
    lines += [_fact_row(r, rel_paths=rel_paths, L=L) for r in rows]
    cond = conditions_text(first.get("conditions"))
    if cond:
        lines += ["", f"{L['conditions']}: {cond}"]
    docs = list(dict.fromkeys(str(r.get("doc_id") or "") for r in rows if r.get("doc_id")))
    return {
        "id": page_id("timeline", series_key), "kind": "timeline", "title": title, "summary": line, "text": "\n".join(lines).strip() + "\n",
        "path": f"timelines/{_slug(subject)}--{_slug(concept)}--{_path_tag(series_key)}.md",
        "entity_keys": list(dict.fromkeys(str(k) for r in rows for k in (r.get("subject_keys") or [r.get("subject_key")]) if k)),
        "concept_keys": [str(first.get("concept_key"))] if first.get("concept_key") else [], "doc_ids": docs,
        "point_ids": list(dict.fromkeys(p for r in rows for p in (r.get("point_ids") or [])))[:POINT_IDS_PER_PAGE],
        "spec_ids": [str(r.get("id")) for r in rows if r.get("id")], "axis_from": fact_axis(rows[0]), "axis_to": fact_axis(rows[-1]),
        "points": len(rows), "flags": flags,
    }


def build_source_page(doc_id: str, info: dict[str, Any], units: list[Any], *, L: dict[str, str],
                      entities: list[dict[str, Any]], facts: list[dict[str, Any]], conflicts: list[dict[str, Any]],
                      unit_kinds: dict[str, str]) -> dict[str, Any]:
    rel_path = str(info.get("rel_path") or (units[0].rel_path if units else doc_id))
    axis = str(info.get("value") or "")
    kinds = Counter(unit_kinds.get(u.unit_id, getattr(u, "kind", "body")) for u in units)
    subjects = sorted([e for e in entities if doc_id in (e.get("doc_ids") or []) and not e.get("boilerplate") and not e.get("scope")],
                      key=lambda e: -float(e.get("frequency") or 0))[:12]
    doc_facts = [f for f in facts if str(f.get("doc_id") or "") == doc_id]
    quality_note = _quality_summary(doc_facts, L)
    lines = [f"# {Path(rel_path).name}", "", f"{L['axis']}: {axis or L['no_axis']}" + (f" ({info.get('kind')})" if info.get("kind") and axis else ""),
             L["kinds"].format(body=kinds.get("body", 0), conclusion=kinds.get("conclusion", 0), listing=kinds.get("listing", 0),
                               boilerplate=kinds.get("boilerplate", 0)) + f" · {len(doc_facts)} {L['facts_n']}", ""]
    if subjects:
        lines += [f"## {L['subjects']}", ""] + [f"- {e.get('title')} ({e.get('type') or '-'})" for e in subjects] + [""]
    if quality_note:
        pending = [f for f in doc_facts if _unresolved(f)]
        lines += [f"## {L['unresolved']}", "", quality_note, ""]
        lines += ["- " + _part_fact_text(f, L=L) for f in pending[:20]] + [""]
    excerpts = [u.text.strip() for u in units if unit_kinds.get(u.unit_id, getattr(u, "kind", "body")) == "conclusion" and u.text.strip()]
    if excerpts:
        text = "\n\n".join(excerpts)
        lines += [f"## {L['conclusion']}", "", text[:CONCLUSION_EXCERPT_CHARS] + ("…" if len(text) > CONCLUSION_EXCERPT_CHARS else ""), ""]
    doc_conflicts = [c for c in conflicts if any(str(v.get("doc_id") or "") == doc_id for v in (c.get("values") or []))]
    if doc_conflicts:
        lines += [f"## {L['conflicts']}", ""]
        for c in doc_conflicts[:20]:
            vals = "; ".join(f"{v.get('value') or values_text(v)} ({Path(str(v.get('rel_path') or '')).name})" for v in c.get("values") or [])
            lines.append(f"- {c.get('subject')} · {c.get('concept')} @ {c.get('axis') or '-'}: {vals}")
        lines.append("")
    summary = f"{Path(rel_path).name}" + (f" ({axis})" if axis else "") + " · " + ", ".join(str(e.get("title")) for e in subjects[:5])
    if quality_note:
        summary += "\n" + quality_note
    return {
        "id": page_id("source", doc_id), "kind": "source", "title": Path(rel_path).name, "summary": summary,
        "text": "\n".join(lines).strip() + "\n", "path": f"sources/{_slug(Path(rel_path).stem)}--{_path_tag(doc_id)}.md",
        "entity_keys": [str(e["key"]) for e in subjects], "concept_keys": [], "doc_ids": [doc_id],
        "point_ids": list(dict.fromkeys(p for u in units for p in u.point_ids))[:POINT_IDS_PER_PAGE],
        "spec_ids": [str(f.get("id")) for f in doc_facts if f.get("id")][:200], "axis": axis, "rel_path": rel_path,
        "conflicts": len(doc_conflicts),
    }


def build_index_page(pages: list[dict[str, Any]], concepts: list[dict[str, Any]], *, L: dict[str, str], kb_id: str) -> dict[str, Any]:
    lines = [f"# {L['index']} · {kb_id}", ""]
    for kind, heading in (("subject", L["subject_pages"]), ("timeline", L["timeline_pages"]), ("source", L["source_pages"])):
        rows = [p for p in pages if p["kind"] == kind]
        if not rows:
            continue
        lines += [f"## {heading} ({len(rows)})", ""]
        for p in rows[:INDEX_ROWS]:
            # The current index links to history pages without quoting historical values.
            lines.append(f"- [{p['title']}]({p['path']})")
        if len(rows) > INDEX_ROWS:
            lines.append(L["more"].format(n=len(rows) - INDEX_ROWS))
        lines.append("")
    if concepts:
        lines += [f"## {L['concepts']} ({len(concepts)})", ""]
        for c in concepts[:INDEX_ROWS]:
            lines.append("- " + L["concept_line"].format(label=c.get("label"), n=c.get("facts", 0), docs=len(c.get("docs") or [])))
        lines.append("")
    return {
        "id": page_id("index", kb_id), "kind": "index", "title": f"{L['index']} · {kb_id}", "summary": "",
        "text": "\n".join(lines).strip() + "\n", "path": "index.md", "entity_keys": [], "concept_keys": [], "doc_ids": [],
        "point_ids": [], "spec_ids": [],
    }


def narrate_pages(client: ChatClient, pages: list[dict[str, Any]], *, language: str,
                  max_pages: int = NARRATE_MAX_PAGES, progress: Callable[[int, int], None] | None = None) -> dict[str, int]:
    """Narration of subject pages: hand the deterministic page to the summary model for a paragraph (using only
    what the page contains); on failure the deterministic summary stays."""
    todo = [p for p in pages if p["kind"] == "subject" and int(p.get("facts") or 0) >= NARRATE_MIN_FACTS]
    todo = sorted(todo, key=lambda p: -(int(p.get("facts") or 0) * 3 + int(p.get("relations") or 0)))[:max_pages]
    stats = {"candidates": len(todo), "narrated": 0, "failed": 0}
    if not todo:
        return stats

    def work(page: dict[str, Any]) -> str:
        prompt = prompts.PAGE_NARRATE_PROMPT.format(language=language or "English", title=page["title"],
                                                    page=str(page.get("narrate_text") or page["text"])[:NARRATE_INPUT_CHARS])
        return client.chat(prompt, max_tokens=700).strip()

    for page, text, error in client.run_parallel(todo, work, progress=progress):
        if error is None and text:
            page["narrative"] = text
            page["summary"] = "\n".join([x for x in (text, page.get("quality_note")) if x] + list(page.get("series") or [])[:6])
            page["text"] = page["text"].replace("## " + labels_for(language)["overview"] + "\n\n",
                                                "## " + labels_for(language)["overview"] + "\n\n" + text + "\n\n", 1)
            stats["narrated"] += 1
        else:
            stats["failed"] += 1
            if error is not None and not isinstance(error, LLMCallError):
                print(f"[graph] narrate failed for {page['title']!r}: {error!r}", flush=True)
    return stats


def compile_pages(graph: dict[str, Any], units: list[Any], *, out_dir: Path | None, language: str,
                  client: ChatClient | None = None, stage: Callable[[str], None] | None = None,
                  narrate: bool = True) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Returns (pages, stats). When out_dir is given, the markdown is written to out_dir/<path> (the whole directory
is rewritten)."""
    import time
    from .current_view import graph_for_pages
    from .currency import in_scope
    today = time.strftime("%Y-%m-%d")
    graph, units = graph_for_pages(graph, units, today=today)
    L = labels_for(language)
    entities = list(graph.get("entities") or [])
    relations = list(graph.get("relations") or [])
    facts = list(graph.get("specs") or [])
    concepts = list(graph.get("concepts") or [])
    conflicts = list(graph.get("conflicts") or [])
    documents = {str(k): dict(v or {}) for k, v in (graph.get("documents") or {}).items()}
    profile = dict(graph.get("profile") or {})
    unit_kinds = {str(k): str(v) for k, v in (graph.get("unit_kinds") or {}).items()}
    mention_points: dict[str, list[str]] = defaultdict(list)
    for m in graph.get("mentions") or []:
        if int(m.get("count") or 0) > 0 and m.get("point_id"):
            mention_points[str(m.get("entity_key"))].append(str(m["point_id"]))
    entity_titles = {str(e["key"]): str(e.get("title") or e["key"]) for e in entities}
    facts_by_subject = _facts_by_subject([f for f in facts if in_scope(f, {"mode": "current", "day": today})])
    rels_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in relations:
        if r.get("boilerplate") or r.get("reference"):
            continue
        rels_by_key[str(r.get("source_key"))].append(r)
        rels_by_key[str(r.get("target_key"))].append(r)
    units_by_doc: dict[str, list[Any]] = defaultdict(list)
    for u in units:
        units_by_doc[str(u.doc_id)].append(u)
    for rows in units_by_doc.values():
        rows.sort(key=lambda u: int(getattr(u, "order", 0) or 0))
    for doc in units_by_doc:
        documents.setdefault(doc, {"rel_path": units_by_doc[doc][0].rel_path, "kind": "none", "value": ""})

    if stage:
        stage(f"Compiling view pages · {STAGE_PARTS['subjects']}")
    pages: list[dict[str, Any]] = []
    subjects = select_subjects(entities, facts_by_subject, profile)
    paged = {str(e["key"]) for e in subjects}
    by_key = {str(e["key"]): e for e in entities}
    parts_of = parts_index(relations, entities)
    for e in subjects:
        parts = [by_key[k] for k in parts_of.get(str(e["key"]), []) if k in by_key]
        pages.append(build_subject_page(e, facts_by_subject.get(e["key"], []), rels_by_key.get(e["key"], []), L=L,
                                        documents=documents, mention_points=mention_points, profile=profile, entity_titles=entity_titles,
                                        parts=parts, facts_by_subject=facts_by_subject, paged=paged))
    if stage:
        stage(f"Compiling view pages · {STAGE_PARTS['timelines']}")
    series = series_of(facts)
    ordered_series = sorted(series.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:TIMELINE_MAX_PAGES]
    for skey, rows in ordered_series:
        pages.append(build_timeline_page(skey, rows, L=L, documents=documents))
    if stage:
        stage(f"Compiling view pages · {STAGE_PARTS['sources']}")
    for doc in sorted(documents, key=lambda d: (axis_sort_key(str(documents[d].get("value") or "")), d)):
        pages.append(build_source_page(doc, documents[doc], units_by_doc.get(doc, []), L=L, entities=entities, facts=facts,
                                       conflicts=conflicts, unit_kinds=unit_kinds))
    pages.append(build_index_page(pages, concepts, L=L, kb_id=str(graph.get("kb_id") or "")))
    stats: dict[str, Any] = {"pages": len(pages), "subjects": len(subjects), "timelines": len(ordered_series),
                             "sources": len(documents), "narrate": {}}
    if narrate and client is not None:
        if stage:
            stage(f"Compiling view pages · {STAGE_PARTS['narrate']}")
        stats["narrate"] = narrate_pages(client, pages, language=language,
                                         progress=(lambda d, t: stage(f"Compiling view pages · {STAGE_PARTS['narrate']} {d}/{t}")) if stage else None)
        stats["llm"] = dict(client.stats)
    for page in pages:
        page["compiled_day"] = today
        page["time_mode"] = "history" if page.get("kind") in ("timeline", "source") else "current"
        # Keep complete provenance even when vector payload doc_ids/point_ids are capped.
        ids = page.get("doc_ids") or list(documents)
        page["source_paths"] = sorted({str(documents[d].get("rel_path") or "") for d in ids if d in documents})
    if out_dir is not None:
        write_pages(out_dir, pages)
        stats["out_dir"] = str(out_dir)
    return pages, stats


def write_pages(out_dir: Path, pages: list[dict[str, Any]]) -> int:
    """Rewrite the whole directory: the previous version's pages are not kept (pages are a projection, with no
    manual upkeep)."""
    import shutil

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # paths are unique across all pages: a collision gets a sequence number, and the manifest (pages.json) records
    # the id together with the physical file, so two pages never end up in one file
    seen: set[str] = set()
    for p in pages:
        base = str(p["path"])
        path, n = base, 2
        while path in seen:
            path = re.sub(r"\.md$", "", base) + f"-{n}.md"
            n += 1
        seen.add(path)
        p["path"] = path
    for p in pages:
        target = out_dir / str(p["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(p["text"]), encoding="utf-8")
    (out_dir / "pages.json").write_text(json.dumps(
        [{k: v for k, v in p.items() if k not in ("text", "narrate_text")} for p in pages], ensure_ascii=False, indent=1), encoding="utf-8")
    return len(pages)
