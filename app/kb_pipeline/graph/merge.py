"""合并、过滤、权重、切片归属 —— 全是本地计算,不调 LLM(描述摘要在 summarize.py)。

· 实体键 = 规范名(NFKC、空白折叠、大小写不敏感),按文档隔开的类型前面加 doc_id::,同名不同上层类拆开的
  第二个节点后面加 ||上层类(规则见 merge_extractions);类型取众数;描述去重保序;
  frequency = 各单元提及次数之和;unit_ids 并集。
· 关系:受控谓词有方向(HAS_PIN 是 A → B),兜底谓词 related_to 无向(sorted 端点)。
  同 (源, 目标, 谓词) 合并:strength_sum 求和、evidence = 单元数、描述去重。
· 过滤:自环、孤儿端点、否定描述(「无明确关系」「no clear relationship」…)。
· 端点类型门(4.10):谓词允许的端点父类对不符合的标 type_violation,不删。
· 权重:degree(无向去重邻居数)、combined_degree、带边权 pagerank、cooccur
  (两端共同出现的单元数)、NPMI(三个概率统一以单元总数为分母、保留负值)、
  最终 weight = 1 + 9 × (a·norm(strength_sum) + b·norm(cooccur) + c·(npmi+1)/2)。
· MENTIONED_IN:在单元覆盖的各切片文本(与切片的 section_path)里查实体名,命中哪片
  挂哪片并计数;一片都没命中的挂单元全部切片(count=0,表示推断而非命中)。
· 降噪(2026-09-04):单元类型合成(结构规则 + 模型判断)、
  样板 / 罗列单元的 strength 打折并给关系与实体打 boilerplate 标记;值(1.3V、667 MHz)
  不成节点、折成端点实体的属性描述;编号(Figure 5、001-79553)标 reference;合写名字
  (A/B)拆到各段;类型体检把编号 / 样板占大头的类型整体降为 reference。
"""
from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from collections import Counter
from typing import Any, Iterable

from . import prompts
from .resolution import is_identifier_like
from .extract import entity_key, normalize_name
from .units import UNIT_KINDS, Unit

# Contribution of each unit kind to relation strength: "relations" on boilerplate pages are mostly revision
# records and table-of-contents entries
UNIT_KIND_STRENGTH = {"body": 1.0, "conclusion": 1.0, "listing": 0.5, "boilerplate": 0.25}
_KIND_RANK = {"body": 0, "conclusion": 0, "listing": 1, "boilerplate": 2}
# Type health check: types where references + boilerplate-only + values reach this ratio (with enough instances)
# get their remaining entities flagged reference as a whole
TYPE_DEMOTE_RATIO = 0.6
TYPE_DEMOTE_MIN = 5


def combine_unit_kind(structural: str | None, llm: str | None) -> str:
    """Combine structural rules with the model's judgement: boilerplate if either side says so; listing only on the
    structural rule (the model also calls register bit definition tables and truth tables listing; on kb_003 half
    of 32 such units were spec tables); otherwise body. A wrong boilerplate call only keeps the unit out of recall
    seeds and candidates and discounts strength, it deletes no data, so over-flagging is the safer side."""
    s = str(structural or "body")
    m = str(llm or "body")
    s = s if s in UNIT_KINDS else "body"
    m = m if m in UNIT_KINDS else "body"
    if "boilerplate" in (s, m):
        return "boilerplate"
    if s == "listing":
        return "listing"
    if "conclusion" in (s, m):
        return "conclusion"      # conclusion: either side counts; a kind of body text that feeds the view layer's conclusion facts
    return "body"


def best_kind(kinds: Iterable[str]) -> str:
    ranked = sorted((str(k or "body") for k in kinds), key=lambda k: _KIND_RANK.get(k, 0))
    return ranked[0] if ranked else "body"

NEGATION_PATTERNS = (
    "no clear relationship", "no relationship", "not related", "unrelated", "no direct relationship",
    "no explicit relationship", "relationship is unclear", "no apparent relationship",
    "not explicitly", "cannot be determined", "relationship is not",
    "无明确关系", "没有明确关系", "无直接关系", "没有直接关系", "无关", "没有关系", "不相关", "关系不明确",
    "未明确列出关系", "未明确说明关系", "未明确提及关系", "无法确定关系", "文本中未明确",
)
_NEGATION_RE = re.compile("|".join(re.escape(p) for p in NEGATION_PATTERNS), re.IGNORECASE)

# The three coefficients of the final edge weight; can be overridden with
# KB_GRAPH_WEIGHT_COEFFICIENTS='{"strength":0.5,"cooccur":0.3,"npmi":0.2}'
DEFAULT_WEIGHT_COEFFICIENTS = {"strength": 0.5, "cooccur": 0.3, "npmi": 0.2}
PAGERANK_DAMPING = 0.85
PAGERANK_ITERATIONS = 60


def weight_coefficients() -> dict[str, float]:
    raw = os.getenv("KB_GRAPH_WEIGHT_COEFFICIENTS")
    coeffs = dict(DEFAULT_WEIGHT_COEFFICIENTS)
    if raw:
        try:
            override = json.loads(raw)
            for k in coeffs:
                if k in override:
                    coeffs[k] = float(override[k])
        except (ValueError, TypeError):
            pass
    return coeffs


def is_negated(description: str) -> bool:
    return bool(_NEGATION_RE.search(str(description or "")))


def _majority(counter: dict[str, int], known: set[str] | None = None) -> str:
    """Majority vote; ties prefer a type from the type table, then the name."""
    if not counter:
        return ""
    known = known or set()
    return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0].casefold() not in known, kv[0]))[0][0]


def merge_extractions(
    units: list[Unit],
    extractions: dict[str, dict[str, Any]],
    *,
    entity_types: Iterable[str] = (),
    upper_parents: dict[str, str] | None = None,
    type_parents: dict[str, str] | None = None,
    type_scopes: dict[str, str] | None = None,
    allowed_ends: dict[str, tuple[set[str], set[str]]] | None = None,
    unit_kinds: dict[str, str] | None = None,
    prior_pairs: Iterable[tuple[str, str, str]] = (),
    exact_types: dict[str, str] | None = None,
) -> dict[str, Any]:
    """把各单元的抽取结果合成一张图。返回 {"entities": [...], "relations": [...], "stats": {...}}。

    实体身份(2026-10-24 改版,方案附录第 4 条):
    · 先收齐一个名字在全库的所有提及,按类型众数定作用域(type_scopes 逐类型判定,没判定的按上层类默认;
      单字符一类的符号名、按文档隔开的类型 → 每份文档一个节点)。
      以前按每个单元自己的类型定,同一个名字在一份文档里能同时成为全局节点和文档内节点。
    · 事件按名字定:名字里带日期(年份、年月、月日)的跨文档认同一个(「2026-03-14 设计评审」在纪要和周报里是同一场);
      不带日期的事件出现在几份日期不同的文档里,多半是同名的几次(每年的「健康体检」、每周的「例会」),每份文档各一个节点,
      不管类型的作用域判的是什么(例如不同年份的同名事件,不能并成一次发生)。
    · 全局的名字,如果有另一组上层类的提及占到 UPPER_SPLIT_MIN_SHARE、且来自至少 UPPER_SPLIT_MIN_DOCS 份文档,
      认定是同名的两样东西(公司与它的同名产品),拆成两个节点,第二个的键带「||上层类」。组见 UPPER_GROUPS:
      同组里的分歧(一个病名这里算病症、那里算风险因素;一项检测这里算过程、那里算指标)是同一个东西的类型噪声,
      照旧按众数归到一个节点。
    · 规则抽取的实体(代码、配置)保持各自的作用域(跨文件引用由 scope_doc 指到被引用的文件)。
    upper_parents:{类型: 上层类};type_parents:{类型: 直接父类}(类型树,端点约束按整条链判);
    prior_pairs:上一版模式版本账里已确认的端点组合 (谓词, 起点上层类, 终点上层类),这一版直接放行;
    unit_kinds:合成后的单元类型(见 combine_unit_kind);没给就用单元自己的结构类型;
    exact_types:规则抽取自己的类型 → 上层类(deterministic.UPPER),只用于规则抽取的实体 —— 模式里同名的类型
    (产品的 module、报价里的 package)不受它影响。"""
    from ..ontology import DEFAULT_SCOPE, UPPER_CLASSES, type_paths, upper_of_name

    allowed_ends = allowed_ends or {}
    known_types = {str(t).casefold() for t in entity_types}
    upper_of_type = {str(t).casefold(): str(up) for t, up in (upper_parents or {}).items()}
    tree = {str(k): str(v) for k, v in (type_parents or {}).items()}
    for t, up in (upper_parents or {}).items():
        tree.setdefault(str(t), str(up))
    paths = type_paths(tree)
    scope_of_type = {str(t).casefold(): str(s) for t, s in (type_scopes or {}).items()}

    def upper_for(etype: str) -> str:
        """类型 → 上层类:按类型表;类型表外的名字(模型说不出具体类型时退回写上层类名,或写了 standard、method
        这类同义词)按本体认,认不出才算实体(10-24 真机:表外的 concept 全被当成了实体)。"""
        key = str(etype or "").casefold()
        return upper_of_type.get(key) or (paths.get(key) or [None])[-1] or upper_of_name(key) or "entity"

    def path_for(etype: str) -> list[str]:
        key = str(etype or "").casefold()
        if not key:
            return ["entity"]
        return paths.get(key) or ([key] if key in UPPER_CLASSES else [key, upper_for(etype)])

    def scope_for(etype: str) -> str:
        key = str(etype or "").casefold()
        return scope_of_type.get(key) or DEFAULT_SCOPE.get(upper_for(etype), "global")

    exact_upper = {str(t).casefold(): str(u) for t, u in (exact_types or {}).items()}

    kind_of = {u.unit_id: str((unit_kinds or {}).get(u.unit_id) or u.kind or "body") for u in units}
    entities: dict[str, dict[str, Any]] = {}
    relations: dict[tuple[str, str, str], dict[str, Any]] = {}
    stats: Counter = Counter({k: 0 for k in (
        "units_without_extraction", "self_loops_dropped", "negated_dropped", "orphan_dropped",
        "schema_drift_types", "schema_drift_predicates", "type_violations",
        "boilerplate_units", "listing_units", "boilerplate_relations", "boilerplate_entities",
        "value_entities_dropped", "value_relations_folded", "reference_entities", "reference_relations",
        "combined_names_split", "demoted_types", "homonyms_split")})
    unit_order = {u.unit_id: u.order for u in units}
    # Source label of each description (file name + axis value): the summary prompt gets JSON lines, and claims
    # from different sources / times must be attributed side by side
    source_of = {u.unit_id: {"source": (u.rel_path or u.doc_id).rsplit("/", 1)[-1], "when": str(getattr(u, "axis", "") or "")}
                 for u in units}
    for kind in kind_of.values():
        if kind == "boilerplate":
            stats["boilerplate_units"] += 1
        elif kind == "listing":
            stats["listing_units"] += 1

    # ── 第一遍:收齐每个名字的提及(模型抽取的按名字分组;规则抽取的各自定键)──
    present: list[tuple[Unit, dict[str, Any]]] = []
    votes: dict[str, Counter] = {}
    surfaces: dict[str, Counter] = {}
    upper_docs: dict[str, dict[str, set[str]]] = {}
    doc_dates: dict[str, set[str]] = {}         # 名字 → 提到它的文档的日期(判不带日期的事件是不是同名的几次)
    for unit in units:
        result = extractions.get(unit.unit_id)
        if not result:
            stats["units_without_extraction"] += 1
            continue
        present.append((unit, result))
        for e in result.get("entities") or []:
            if e.get("exact"):
                continue
            base = entity_key(str(e.get("name") or ""))
            if not base:
                continue
            n = int(e.get("mentions") or 1)
            surfaces.setdefault(base, Counter())[normalize_name(str(e.get("name") or ""))] += n
            if getattr(unit, "doc_date", ""):
                doc_dates.setdefault(base, set()).add(str(unit.doc_date))
            for t, c in (e.get("types") or {str(e.get("type") or ""): 1}).items():
                if t:
                    votes.setdefault(base, Counter())[str(t)] += int(c) if e.get("types") else n
                    upper_docs.setdefault(base, {}).setdefault(upper_for(str(t)), set()).add(unit.doc_id)

    decided: dict[str, dict[str, Any]] = {}
    for base, counter in votes.items():
        etype = _majority(counter, known_types)
        upper = upper_for(etype)
        name = surfaces[base].most_common(1)[0][0] if surfaces.get(base) else base     # 符号名的判定看写法(AA 全局、aa 文档内)
        scope = "document" if symbol_like_name(name) else scope_for(etype)
        if upper == "event":
            if _DATED_NAME_RE.search(name):
                scope = "global"
            elif len(doc_dates.get(base, ())) > 1:
                scope = "document"
        split: set[str] = set()
        if scope == "global":
            total = sum(counter.values()) or 1
            by_group: Counter = Counter()
            first_upper: dict[str, str] = {}
            for t, c in counter.most_common():
                up = upper_for(t)
                by_group[UPPER_GROUP.get(up, up)] += c
                first_upper.setdefault(UPPER_GROUP.get(up, up), up)
            own = UPPER_GROUP.get(upper, upper)
            for group, c in by_group.items():
                up = first_upper[group]
                docs = set().union(*(upper_docs.get(base, {}).get(u, set()) for u in UPPER_GROUP_MEMBERS.get(group, (group,))))
                if group != own and c / total >= UPPER_SPLIT_MIN_SHARE and len(docs) >= UPPER_SPLIT_MIN_DOCS:
                    split.add(up)
        if split:
            stats["homonyms_split"] += len(split)
        decided[base] = {"scope": scope, "upper": upper, "split": split}

    def key_for(base: str, etype: str, doc_id: str) -> tuple[str, str]:
        """(键, 作用域 doc_id 或 "")。"""
        d = decided.get(base)
        if d is None:
            return base, ""
        if d["scope"] == "document":
            return scoped_key(base, doc_id), doc_id
        up = upper_for(etype) if etype else d["upper"]
        for other in d["split"]:
            if UPPER_GROUP.get(up, up) == UPPER_GROUP.get(other, other):
                return f"{base}{UPPER_KEY_SEPARATOR}{other}", ""
        return base, ""

    # ── 第二遍:按定好的键合并 ──
    for unit, result in present:
        kind = kind_of.get(unit.unit_id, "body")
        # Keys (with scope) of this unit's names; relation endpoints use the same table so both sides agree on keys
        keys_in_unit: dict[str, str] = {}
        ambiguous_keys: set[str] = set()
        refs_in_unit: dict[tuple[str, str], str] = {}
        for e in result.get("entities") or []:
            # Keys are recomputed from the name rather than read from the stored extraction: once the
            # normalize_name rules change (e.g. LaTeX restoration), stored extractions still merge under the new
            # rules without re-extraction
            name = str(e.get("name") or "")
            etype = str(e.get("type") or "")
            base = entity_key(name)
            if e.get("exact"):
                # 规则抽取:按自己的类型定作用域;跨文件引用的实体属于被引用的那份文件
                scope_doc = str(e.get("scope_doc") or "") or unit.doc_id
                up = exact_upper.get(etype.casefold()) or upper_for(etype)
                if base and (DEFAULT_SCOPE.get(up, "global") == "document" or symbol_like_name(name)):
                    key, scope = scoped_key(base, scope_doc), scope_doc
                else:
                    key, scope = base, ""
            elif base and e.get("identity"):
                # 表格里同名不同编号的行(两个叫张三的员工):按编号各成一个节点,只写名字的提及不并进任何一个
                key, scope = f"{base}{UPPER_KEY_SEPARATOR}{e['identity']}", ""
            elif base:
                key, scope = key_for(base, etype, unit.doc_id)
            else:
                key, scope = str(e.get("key") or ""), ""
            if not key:
                continue
            if base in keys_in_unit and keys_in_unit[base] != key:
                ambiguous_keys.add(base)
            keys_in_unit[base] = key
            if e.get("record_ref"):
                refs_in_unit[(base, str(e["record_ref"]))] = key
            slot = entities.get(key)
            if slot is None:
                slot = entities[key] = {
                    "key": key, "surface": Counter(), "types": Counter(), "descriptions": [], "description_sources": [],
                    "frequency": 0, "unit_ids": [], "doc_ids": [], "kinds": Counter(), "scope": scope,
                    "exact": False, "aliases": [], "kind_votes": Counter(), "external_ids": [],
                }
            if e.get("exact"):
                slot["exact"] = True
            for a in e.get("aliases") or []:
                a = normalize_name(str(a))
                if a and a not in slot["aliases"]:
                    slot["aliases"].append(a)
            for x in e.get("external_ids") or []:
                if isinstance(x, dict) and x.get("value") and x not in slot["external_ids"]:
                    slot["external_ids"].append({"scheme": str(x.get("scheme") or ""), "value": str(x["value"])})
            slot["kinds"][kind] += 1
            slot["surface"][normalize_name(name)] += int(e.get("mentions") or 1)
            for t, n in (e.get("types") or {etype: 1}).items():
                if t:
                    slot["types"][str(t)] += int(n)
            ek = str(e.get("kind") or "").strip().lower()
            if ek in ENTITY_KINDS:
                slot["kind_votes"][ek] += int(e.get("mentions") or 1)
            for d in e.get("descriptions") or ([e["description"]] if e.get("description") else []):
                d = str(d).strip()
                if d and d not in slot["descriptions"]:
                    slot["descriptions"].append(d)
                    slot["description_sources"].append(dict(source_of.get(unit.unit_id) or {}))
            slot["frequency"] += int(e.get("mentions") or 1)
            if unit.unit_id not in slot["unit_ids"]:
                slot["unit_ids"].append(unit.unit_id)
            if unit.doc_id not in slot["doc_ids"]:
                slot["doc_ids"].append(unit.doc_id)
        for r in result.get("relations") or []:
            sb, tb = entity_key(str(r.get("source") or "")), entity_key(str(r.get("target") or ""))
            def endpoint(base, field):
                ref = str(r.get(field + "_ref") or "")
                if ref:
                    return refs_in_unit.get((base, ref), "")
                if base in ambiguous_keys:
                    return ""  # An ambiguous name is not an identity.
                return keys_in_unit.get(base) or (key_for(base, "", unit.doc_id)[0] if base else "") or str(r.get(field + "_key") or "")
            sk, tk = endpoint(sb, "source"), endpoint(tb, "target")
            predicate = str(r.get("predicate") or prompts.DEFAULT_PREDICATE)
            if not sk or not tk:
                continue
            if sk == tk:
                stats["self_loops_dropped"] += 1
                continue
            descriptions = [str(d).strip() for d in (r.get("descriptions") or ([r["description"]] if r.get("description") else [])) if str(d).strip()]
            if descriptions and all(is_negated(d) for d in descriptions):
                stats["negated_dropped"] += 1
                continue
            if predicate == prompts.DEFAULT_PREDICATE and tk < sk:
                sk, tk = tk, sk
            rkey = (sk, tk, predicate)
            slot = relations.get(rkey)
            if slot is None:
                slot = relations[rkey] = {
                    "source_key": sk, "target_key": tk, "predicate": predicate,
                    "directed": predicate != prompts.DEFAULT_PREDICATE,
                    "descriptions": [], "description_sources": [], "strength_sum": 0.0, "evidence": 0, "unit_ids": [],
                    "predicate_raw": Counter(), "kinds": Counter(), "doc_ids": [], "periods": Counter(),
                }
            slot["kinds"][kind] += 1
            slot["strength_sum"] += float(r.get("strength") or 1.0) * UNIT_KIND_STRENGTH.get(kind, 1.0)
            if unit.unit_id not in slot["unit_ids"]:
                slot["unit_ids"].append(unit.unit_id)
                slot["evidence"] += 1
            if unit.doc_id not in slot["doc_ids"]:
                slot["doc_ids"].append(unit.doc_id)
            raw = str(r.get("predicate_raw") or "")
            if raw:
                slot["predicate_raw"][raw] += 1
            if r.get("valid_from") or r.get("valid_until"):
                slot["periods"][(str(r.get("valid_from") or ""), str(r.get("valid_until") or ""))] += 1
            for d in descriptions:
                if d not in slot["descriptions"]:
                    slot["descriptions"].append(d)
                    slot["description_sources"].append(dict(source_of.get(unit.unit_id) or {}))

    # Orphan edges: an endpoint never appeared as an entity
    kept: dict[tuple[str, str, str], dict[str, Any]] = {}
    for rkey, rel in relations.items():
        if rel["source_key"] not in entities or rel["target_key"] not in entities:
            stats["orphan_dropped"] += 1
            continue
        kept[rkey] = rel

    entity_rows: list[dict[str, Any]] = []
    for key, slot in entities.items():
        title = slot["surface"].most_common(1)[0][0] if slot["surface"] else key
        etype = _majority(slot["types"], known_types)
        if etype and etype.casefold() not in known_types and known_types:
            stats["schema_drift_types"] += 1
        kinds = slot["kinds"]
        boilerplate_only = bool(kinds) and all(k == "boilerplate" for k in kinds)
        if boilerplate_only:
            stats["boilerplate_entities"] += 1
        kv = slot["kind_votes"]
        if slot.get("exact") and etype.casefold() in exact_upper:
            upper, tpath = exact_upper[etype.casefold()], [etype.casefold(), exact_upper[etype.casefold()]]
        else:
            upper, tpath = (upper_for(etype), path_for(etype)) if etype else ("entity", ["entity"])
        entity_rows.append({
            "key": key, "title": title, "type": etype,
            "upper": upper,
            "type_path": tpath,
            "scope": slot.get("scope") or "",
            # 类与实例(方案第 13 条):提及里多数说是泛指的类别名才算 class,平手按实例
            "kind": "class" if kv and kv.get("class", 0) > kv.get("instance", 0) else "instance",
            "external_ids": slot["external_ids"],
            "descriptions": slot["descriptions"], "description_sources": slot["description_sources"], "description": "",
            "frequency": int(slot["frequency"]),
            "unit_ids": sorted(slot["unit_ids"], key=lambda u: unit_order.get(u, 0)),
            "doc_ids": slot["doc_ids"],
            "aliases": list(dict.fromkeys([s for s, _ in slot["surface"].most_common() if s != title]
                                          + [a for a in slot.get("aliases") or [] if a != title])),
            "attributes": [],
            "boilerplate": boilerplate_only,
            "reference": False,
            "evidence_kind": best_kind(kinds.keys()),
            "exact": bool(slot.get("exact")),
        })
    by_key = {e["key"]: e for e in entity_rows}
    relation_rows: list[dict[str, Any]] = []
    for rel in kept.values():
        src, tgt = by_key[rel["source_key"]], by_key[rel["target_key"]]
        violation = False
        ends = allowed_ends.get(rel["predicate"])
        if ends is not None:
            src_ok = not ends[0] or "*" in ends[0] or any(t in ends[0] for t in src["type_path"])
            tgt_ok = not ends[1] or "*" in ends[1] or any(t in ends[1] for t in tgt["type_path"])
            violation = not (src_ok and tgt_ok)
            if violation:
                stats["type_violations"] += 1
        raw = rel["predicate_raw"]
        if rel["predicate"] == prompts.DEFAULT_PREDICATE and any(k != prompts.DEFAULT_PREDICATE for k in raw):
            stats["schema_drift_predicates"] += 1
        kinds = rel["kinds"]
        boilerplate_only = bool(kinds) and all(k == "boilerplate" for k in kinds)
        if boilerplate_only:
            stats["boilerplate_relations"] += 1
        period = rel["periods"].most_common(1)[0][0] if rel["periods"] else ("", "")
        relation_rows.append({
            "source_key": rel["source_key"], "target_key": rel["target_key"],
            "source": src["title"], "target": tgt["title"],
            "predicate": rel["predicate"], "directed": rel["directed"],
            "descriptions": rel["descriptions"], "description_sources": rel["description_sources"], "description": "",
            "strength_sum": round(float(rel["strength_sum"]), 3), "evidence": int(rel["evidence"]),
            "unit_ids": sorted(rel["unit_ids"], key=lambda u: unit_order.get(u, 0)),
            "doc_ids": list(rel["doc_ids"]),
            # 关系的时间(方案附录第 3 条):原文写明的起止(抽取时按文档日期换算过);观察时间在 temporal 阶段按文档补
            "valid_from": period[0], "valid_until": period[1],
            "type_violation": violation,
            "predicate_raw": _majority(raw) if raw else "",
            "boilerplate": boilerplate_only,
            "reference": False,
            "evidence_kind": best_kind(kinds.keys()),
        })
    entity_rows, relation_rows, split = dissolve_combined_names(entity_rows, relation_rows)
    stats["combined_names_split"] = split
    entity_rows, relation_rows, admit_stats = admit_entities(entity_rows, relation_rows)
    for k, v in admit_stats.items():
        if not str(k).startswith("_"):
            stats[k] += int(v)
    # The type health check only reports, it no longer demotes whole types: the second rebuild of kb_003 demoted
    # product family (the PSoC marketing pages contributed 36 boilerplate-only entities) and took legitimate
    # question targets such as the ordering code ZK14B108L-ZS20XIT out of the seeds with it. The two per-entity
    # flags (boilerplate-only and reference) are enough; a type-level cut is left to a human who has read the
    # report.
    health = type_health(entity_rows, value_counts=admit_stats.get("_values_by_type") or {})
    demoted = [row["type"] for row in health if row.get("demoted")]
    stats["demoted_types"] = 0
    derived = derive_variant_edges(entity_rows, relation_rows)
    relation_rows.extend(derived)
    stats["derived_variant_edges"] = len(derived)
    stats["reference_entities"] = sum(1 for e in entity_rows if e.get("reference"))
    stats["reference_relations"] = sum(1 for r in relation_rows if r.get("reference"))
    stats.update({"entities": len(entity_rows), "relations": len(relation_rows)})
    out_stats = {k: v for k, v in stats.items() if not str(k).startswith("_")}
    out_stats["type_health"] = health
    out_stats["demoted_type_names"] = []
    out_stats["noisy_type_names"] = demoted      # just a hint: these types have a high reference / boilerplate share, worth a look
    # Endpoint constraint self-healing: when more than half of a predicate's edges are "violations", the
    # constraint itself is wrong (evaluation: 527/527 violations under one predicate); flagging them is meaningless and drags
    # scores down, so this version relaxes it and notes it in the report
    # Check the account first: endpoint combinations confirmed by a previous build pass directly, without having to
    # reach the threshold again every version (the same for appends, small KBs and corpus jitter); it runs before
    # the threshold relaxation so the report shows which pairs the account released and which this version earned
    prior_applied = relax_prior_pairs(relation_rows, entity_rows, prior_pairs)
    if prior_applied:
        out_stats["endpoint_pairs_prior"] = prior_applied
        print("[graph] endpoint pairs relaxed from the schema account: "
              + ", ".join(f"{p['predicate']} {p['source_parent']}->{p['target_parent']} x{p['count']}" for p in prior_applied), flush=True)
    relaxed = relax_bad_endpoints(relation_rows)
    if relaxed:
        out_stats["endpoints_relaxed"] = relaxed
        print(f"[graph] predicate endpoint constraints relaxed (violation ratio > {RELAX_ENDPOINTS_RATIO:.0%}): {relaxed}", flush=True)
    # 再按「谓词 × 端点上层类对」放行:没过半、但某一对端点集中越界的(associated_with 的 entity → property)
    relaxed_pairs = relax_bad_endpoint_pairs(relation_rows, entity_rows)
    if relaxed_pairs:
        out_stats["endpoint_pairs_relaxed"] = relaxed_pairs
        print(f"[graph] endpoint pairs relaxed (>= {RELAX_PAIR_MIN_EDGES} edges and >= {RELAX_PAIR_RATIO:.0%} of the predicate): "
              + ", ".join(f"{p['predicate']} {p['source_parent']}->{p['target_parent']} x{p['count']}" for p in relaxed_pairs), flush=True)
    if relaxed or relaxed_pairs or prior_applied:
        stats["type_violations"] = sum(1 for r in relation_rows if r.get("type_violation"))
        out_stats["type_violations"] = stats["type_violations"]
    all_pairs = list(relaxed_pairs) + list(prior_applied)
    out_stats["predicate_health"] = predicate_health(relation_rows, entity_rows, relaxed=relaxed, relaxed_pairs=all_pairs)
    # Endpoint account: written back to the schema version at the end of the build; the next merge and label
    # re-extraction both start from it (schema_flow.record_schema_observation)
    out_stats["endpoint_observed"] = endpoint_observation(relation_rows, entity_rows, relaxed=relaxed, relaxed_pairs=all_pairs,
                                                          known_types=known_types)
    out_stats["scoped_entities"] = sum(1 for e in entity_rows if e.get("scope"))
    out_stats["class_entities"] = sum(1 for e in entity_rows if e.get("kind") == "class")
    return {"entities": entity_rows, "relations": relation_rows, "stats": out_stats}


def _linkable_id(value: str) -> bool:
    """能拿来对正文提及的外部编号:带数字,且带字母或至少四位(R04、F12、CUST-0091、10086);纯短数字不认。"""
    v = str(value or "").strip()
    return 2 <= len(v) <= 40 and any(ch.isdigit() for ch in v) and (any(ch.isalpha() for ch in v) or len(v) >= 4)


def link_external_ids(entities: list[dict[str, Any]], relations: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """正文或别的表里直接写编号的提及(「关联场景:R04、R08」)并到带这个编号的实体上:名字或别名与某个外部编号完全相同、
    且全库只有一个实体带这个编号才并(两个实体同号就不知道指谁)。规则,不调模型;编号不同的绝不合并由
    resolution.merge_entities 保证。"""
    from .resolution import merge_entities

    by_value: dict[str, list[int]] = {}
    for i, e in enumerate(entities):
        for x in e.get("external_ids") or []:
            v = str((x or {}).get("value") or "").strip()
            if _linkable_id(v):
                by_value.setdefault(v.casefold(), [])
                if i not in by_value[v.casefold()]:
                    by_value[v.casefold()].append(i)
    pairs: list[tuple[int, int]] = []
    for j, e in enumerate(entities):
        if e.get("external_ids") or e.get("exact"):
            continue
        for n in [str(e.get("title") or "")] + [str(a) for a in (e.get("aliases") or [])]:
            hits = by_value.get(n.strip().casefold())
            if hits and len(hits) == 1 and hits[0] != j:
                pairs.append((hits[0], j))
                break
    if not pairs:
        return entities, relations, {"external_id_mentions_linked": 0}
    merged, rels, _ = merge_entities(entities, relations, pairs)
    return merged, rels, {"external_id_mentions_linked": len(pairs)}


SCOPE_SEPARATOR = "::"
# 同名不同上层类拆开时,第二个节点的键:规范名 + 分隔符 + 上层类(规范名里几乎不会出现 ||)
UPPER_KEY_SEPARATOR = "||"
UPPER_SPLIT_MIN_SHARE = 0.2
UPPER_SPLIT_MIN_DOCS = 2
# 判同名两样东西时的上层类分组:人员与组织、具体的东西(实体 / 部件 / 文档)、抽象的东西(概念 / 属性 / 过程 / 事件)。
# 组与组之间的分歧是真的两样东西(公司与同名产品、产品与同名概念),组内的分歧是同一个东西的类型噪声
UPPER_GROUP_MEMBERS = {"agent": ("agent",), "thing": ("entity", "part", "document"),
                       "abstract": ("concept", "property", "process", "event")}
UPPER_GROUP = {u: g for g, members in UPPER_GROUP_MEMBERS.items() for u in members}
ENTITY_KINDS = ("instance", "class")
# 名字里带日期:单独的年份(2024 年体检、2024 Q3 review)、2026-03-14、2026/3/14、2026.03、2026年3月14日、3月14日
_DATED_NAME_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)|\d{1,2}\s*月\s*\d{1,2}\s*日")


_SYMBOL_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def symbol_like_name(name: str) -> bool:
    """A single non-CJK character (variables A, r, σ), or two characters including a lower-case letter / digit /
    symbol (x0, r1, σ2): such names are not the same thing in two books, so they only become nodes within a
    document (2026-09-12 evaluation: A, C, r, N were merged into global entities across four subject
    directories). Two-letter upper-case abbreviations (OA, IM, WB) stay global."""
    text = normalize_name(str(name or ""))
    if not text or _SYMBOL_CJK_RE.search(text):
        return False
    if len(text) == 1:
        return True
    return len(text) == 2 and not text.isupper()


def scoped_key(base_key: str, doc_id: str) -> str:
    """Key of a document-scoped entity: doc_id::canonical name. Entities of part / property / process types belong
    to something, so the same name in different documents is not the same node."""
    return f"{doc_id}{SCOPE_SEPARATOR}{base_key}" if doc_id else base_key


_VARIANT_SEP_RE = re.compile(r"^(?P<base>.+?)[\-_/\s]+(?P<suffix>[A-Za-z0-9][A-Za-z0-9.\-]*)$")
_CJK_TITLE_RE = re.compile(r"[\u4e00-\u9fff]")
_VARIANT_PAREN_RE = re.compile(r"^(?P<base>[^()（）]+)[(（][^()（）]{1,40}[)）]\s*$")


def derive_variant_edges(entities: list[dict[str, Any]], relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rule-derived edges: among global entities of the same type, one name being the other plus a separator and
    suffix (ZK7C4021KV13-667FCXC is a variant of ZK7C4021KV13; iPhone 15 Pro a variant of iPhone 15) →
    variant_of. Pairs that already have any relation are not added again. Domain-independent, no model; the edge
    is flagged derived with a fixed strength of 1."""
    globals_by_type: dict[str, dict[str, dict[str, Any]]] = {}
    for e in entities:
        if e.get("scope") or e.get("reference") or e.get("boilerplate"):
            continue
        title = str(e.get("title") or "")
        if len(title) < (2 if _CJK_TITLE_RE.search(title) else 4):
            continue
        globals_by_type.setdefault(str(e.get("type") or ""), {})[title.casefold()] = e
    existing = {(r["source_key"], r["target_key"]) for r in relations} | {(r["target_key"], r["source_key"]) for r in relations}
    out: list[dict[str, Any]] = []
    for rows in globals_by_type.values():
        if len(rows) < 2:
            continue
        for title_cf, e in rows.items():
            title = str(e.get("title") or "")
            base = None
            m = _VARIANT_SEP_RE.match(title)
            if m and len(m.group("base")) >= 4:
                base = rows.get(m.group("base").casefold())
            if base is None:
                # Package suffixes of ordering codes (ZK14B108L-ZS25XIT is a variant of ZK14B108L-ZS25XI) and the
                # bracket form (stroke (ischemic stroke) is a variant of stroke): resolution no longer treats them as
                # one entity, so a variant_of edge is drawn here
                if is_identifier_like(title):
                    for k in (1, 2, 3):
                        if len(title) - k >= 6 and title[-k:].isalnum():
                            base = rows.get(title[:-k].casefold())
                            if base is not None:
                                break
                if base is None:
                    m3 = _VARIANT_PAREN_RE.match(title)
                    if m3 and len(m3.group("base").strip()) >= 2:
                        base = rows.get(m3.group("base").strip().casefold())
            if base is None or base is e:
                continue
            pair = (e["key"], base["key"])
            if pair in existing:
                continue
            existing.add(pair)
            out.append({
                "source_key": e["key"], "target_key": base["key"], "source": e.get("title"), "target": base.get("title"),
                "predicate": "variant_of", "directed": True,
                "descriptions": [f"{e.get('title')} is a name variant of {base.get('title')} (derived from the names)"],
                "description": "", "strength_sum": 1.0, "evidence": 0,
                "unit_ids": list(e.get("unit_ids") or [])[:4], "type_violation": False, "predicate_raw": "variant_of",
                "boilerplate": False, "reference": False, "evidence_kind": "derived", "derived": True,
            })
    return out


RELAX_ENDPOINTS_RATIO = 0.5
RELAX_ENDPOINTS_MIN_EDGES = 20
# Relax per "predicate × endpoint parent pair": when one pair of endpoint parents accumulates enough violating
# edges under a predicate, and a non-trivial share of that predicate's edges, it is stable usage rather than a
# sporadic error (evaluation: 119 entity → property edges under one predicate); the constraint was too narrow,
# so that pair is released
RELAX_PAIR_MIN_EDGES = 20
RELAX_PAIR_RATIO = 0.05      # 20 edges is the main bar; the ratio only guards fragments under big predicates (evaluation: process → process 38/407 under one predicate is stable usage too)


def relax_bad_endpoints(relations: list[dict[str, Any]], *, ratio: float = RELAX_ENDPOINTS_RATIO,
                        min_edges: int = RELAX_ENDPOINTS_MIN_EDGES) -> list[str]:
    """Predicates with more than half violations (and enough edges): drop their type_violation flags for this
    version and return the predicate names."""
    edges: Counter = Counter()
    bad: Counter = Counter()
    for r in relations:
        pred = str(r.get("predicate") or prompts.DEFAULT_PREDICATE)
        edges[pred] += 1
        if r.get("type_violation"):
            bad[pred] += 1
    relaxed = sorted(p for p, n in edges.items() if n >= min_edges and bad[p] / n > ratio)
    if relaxed:
        gone = set(relaxed)
        for r in relations:
            if r.get("type_violation") and str(r.get("predicate") or prompts.DEFAULT_PREDICATE) in gone:
                r["type_violation"] = False
                r["endpoints_relaxed"] = True
    return relaxed


def _parent_of(entities: list[dict[str, Any]]) -> dict[str, str]:
    """端点账与放行按上层类记(八个固定的类,跨版本可比);谓词约束里写的是上层类。"""
    return {e["key"]: (str(e.get("upper") or "").casefold() or "-") for e in entities}


def relax_bad_endpoint_pairs(relations: list[dict[str, Any]], entities: list[dict[str, Any]], *,
                             min_edges: int = RELAX_PAIR_MIN_EDGES, ratio: float = RELAX_PAIR_RATIO) -> list[dict[str, Any]]:
    """Relax per "predicate × (source parent, target parent)": when one violating endpoint parent pair under a
    predicate accumulates ≥ min_edges edges and ≥ ratio of that predicate's edges, it is stable extraction usage
    rather than a sporadic error and the constraint was too narrow: drop those edges' type_violation flags and
    return the released pairs (with counts). Predicates over the half mark were already released by
    relax_bad_endpoints; this handles the ones below it where one endpoint pair is concentrated."""
    parent_of = _parent_of(entities)
    edges: Counter = Counter()
    bad_pairs: Counter = Counter()
    for r in relations:
        pred = str(r.get("predicate") or prompts.DEFAULT_PREDICATE)
        edges[pred] += 1
        if r.get("type_violation"):
            bad_pairs[(pred, parent_of.get(r["source_key"], "-"), parent_of.get(r["target_key"], "-"))] += 1
    relaxed = {k for k, n in bad_pairs.items() if n >= min_edges and n / max(1, edges[k[0]]) >= ratio}
    if relaxed:
        for r in relations:
            if not r.get("type_violation"):
                continue
            k = (str(r.get("predicate") or prompts.DEFAULT_PREDICATE), parent_of.get(r["source_key"], "-"), parent_of.get(r["target_key"], "-"))
            if k in relaxed:
                r["type_violation"] = False
                r["endpoints_relaxed"] = True
    return [{"predicate": p, "source_parent": a, "target_parent": b, "count": bad_pairs[(p, a, b)]}
            for (p, a, b) in sorted(relaxed, key=lambda k: (-bad_pairs[k], k))]


def relax_prior_pairs(relations: list[dict[str, Any]], entities: list[dict[str, Any]],
                      prior_pairs: Iterable[tuple[str, str, str]]) -> list[dict[str, Any]]:
    """Endpoint combinations (predicate, source parent, target parent) confirmed in the previous schema version's
    account: passed directly this version without reaching the threshold again. Returns the released pairs (with
    this version's counts, prior=True)."""
    wanted = {(str(p), str(a).casefold(), str(b).casefold()) for p, a, b in prior_pairs}
    if not wanted:
        return []
    parent_of = _parent_of(entities)
    hit: Counter = Counter()
    for r in relations:
        if not r.get("type_violation"):
            continue
        k = (str(r.get("predicate") or prompts.DEFAULT_PREDICATE), parent_of.get(r["source_key"], "-"), parent_of.get(r["target_key"], "-"))
        if k in wanted:
            r["type_violation"] = False
            r["endpoints_relaxed"] = True
            r["endpoints_prior"] = True
            hit[k] += 1
    return [{"predicate": p, "source_parent": a, "target_parent": b, "count": hit[(p, a, b)], "prior": True}
            for (p, a, b) in sorted(hit, key=lambda k: (-hit[k], k))]


def endpoint_observation(relations: list[dict[str, Any]], entities: list[dict[str, Any]], *, relaxed: Iterable[str] = (),
                         relaxed_pairs: Iterable[dict[str, Any]] = (), known_types: Iterable[str] = ()) -> dict[str, Any]:
    """Endpoint account: how many edges each predicate actually has per "source parent->target parent" combination
    (derived edges excluded), the predicates relaxed this version, the released pairs, and the combinations
    confirmed by the same threshold (≥ RELAX_PAIR_MIN_EDGES edges and ≥ RELAX_PAIR_RATIO). Written back to the
    schema version at the end of the build; the next merge passes the confirmed pairs directly and label
    re-extraction keeps them by rule (schema.apply_observed_guard).
    Three health figures are recorded too (2026-09-13, handed to the model on label re-extraction so its
    revisions have evidence, not targeting any particular KB): per predicate, the number of edges outside the
    declared endpoints (violations, relaxed ones included, because they are exactly the evidence of a narrow
    constraint); entities per type (type_counts); and how many entities used names outside the type table at
    extraction (unknown_types, only counted when known_types is given, at most 20)."""
    parent_of = _parent_of(entities)
    pairs: dict[str, Counter] = {}
    edges: Counter = Counter()
    violations: Counter = Counter()
    for r in relations:
        if r.get("derived"):
            continue
        pred = str(r.get("predicate") or prompts.DEFAULT_PREDICATE)
        edges[pred] += 1
        if r.get("type_violation") or r.get("endpoints_relaxed"):
            violations[pred] += 1
        pairs.setdefault(pred, Counter())[f"{parent_of.get(r['source_key'], '-')}->{parent_of.get(r['target_key'], '-')}"] += 1
    known = {str(t).casefold() for t in known_types if str(t)}
    type_counts: Counter = Counter(str(e.get("type") or "") for e in entities)
    unknown: Counter = Counter({t: n for t, n in type_counts.items() if known and t and t.casefold() not in known})
    confirmed: dict[tuple[str, str, str], int] = {}
    for rp in relaxed_pairs:
        key = (str(rp.get("predicate") or ""), str(rp.get("source_parent") or ""), str(rp.get("target_parent") or ""))
        if all(key):
            confirmed[key] = max(confirmed.get(key, 0), int(rp.get("count") or 0))
    for pred, counter in pairs.items():
        for pair, n in counter.items():
            src, tgt = pair.split("->", 1)
            if "-" in (src, tgt) or n < RELAX_PAIR_MIN_EDGES or n / max(1, edges[pred]) < RELAX_PAIR_RATIO:
                continue
            confirmed[(pred, src, tgt)] = max(confirmed.get((pred, src, tgt), 0), n)
    return {
        "edges": dict(edges),
        "pairs": {p: dict(c.most_common()) for p, c in pairs.items()},
        "violations": dict(violations),
        "type_counts": dict(type_counts.most_common()),
        "unknown_types": dict(unknown.most_common(20)),
        "relaxed_predicates": sorted(str(p) for p in relaxed),
        "relaxed_pairs": [{"predicate": rp.get("predicate"), "source_parent": rp.get("source_parent"),
                           "target_parent": rp.get("target_parent"), "count": int(rp.get("count") or 0)} for rp in relaxed_pairs],
        "confirmed": [{"predicate": p, "source_parent": a, "target_parent": b, "count": n}
                      for (p, a, b), n in sorted(confirmed.items(), key=lambda kv: (kv[0][0], -kv[1], kv[0][1], kv[0][2]))],
    }


def confirmed_pairs(observed: dict[str, Any] | None) -> list[tuple[str, str, str]]:
    """Endpoint combinations confirmed in the account [(predicate, source parent, target parent)]: passed directly
    at merge time and kept by rule on label re-extraction."""
    out: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in (observed or {}).get("confirmed") or []:
        key = (str(row.get("predicate") or ""), str(row.get("source_parent") or ""), str(row.get("target_parent") or ""))
        if all(key) and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def predicate_health(relations: list[dict[str, Any]], entities: list[dict[str, Any]], *, relaxed: Iterable[str] = (),
                     relaxed_pairs: Iterable[dict[str, Any]] = ()) -> list[dict[str, Any]]:
    """Per predicate: edge count, violation count and ratio, the (source parent, target parent) combinations with
    the most violations, fan-out p95. Predicates over the half mark had their constraint relaxed this version
    (relax_bad_endpoints) and are marked relaxed in the report; pairs relaxed per endpoint pair
    (relax_bad_endpoint_pairs) are listed in relaxed_pairs, for a human to judge later whether the constraint was
    too narrow or the extraction drifted."""
    relaxed_set = {str(p) for p in relaxed}
    pairs_by_pred: dict[str, list[dict[str, Any]]] = {}
    for rp in relaxed_pairs:
        pairs_by_pred.setdefault(str(rp.get("predicate") or ""), []).append({"source_parent": rp.get("source_parent"), "target_parent": rp.get("target_parent"), "count": rp.get("count")})
    parent_of = _parent_of(entities)
    rows: dict[str, dict[str, Any]] = {}
    fanout: dict[tuple[str, str], int] = {}
    for r in relations:
        pred = str(r.get("predicate") or prompts.DEFAULT_PREDICATE)
        slot = rows.setdefault(pred, {"predicate": pred, "edges": 0, "violations": 0, "derived": 0, "pairs": Counter()})
        slot["edges"] += 1
        if r.get("derived"):
            slot["derived"] += 1
        if r.get("type_violation"):
            slot["violations"] += 1
            slot["pairs"][(parent_of.get(r["source_key"], "-"), parent_of.get(r["target_key"], "-"))] += 1
        fanout[(pred, r["source_key"])] = fanout.get((pred, r["source_key"]), 0) + 1
    out: list[dict[str, Any]] = []
    for pred, slot in rows.items():
        outs = sorted(n for (p, _), n in fanout.items() if p == pred)
        p95 = outs[max(0, int(math.ceil(len(outs) * 0.95)) - 1)] if outs else 0
        out.append({
            "predicate": pred, "edges": slot["edges"], "violations": slot["violations"],
            "violation_ratio": round(slot["violations"] / slot["edges"], 3) if slot["edges"] else 0.0,
            "top_violating_pairs": [{"source_parent": a, "target_parent": b, "count": n}
                                    for (a, b), n in slot["pairs"].most_common(3)],
            "fanout_p95": int(p95), "derived": slot["derived"], "relaxed": pred in relaxed_set,
            "relaxed_pairs": pairs_by_pred.get(pred, []),
        })
    return sorted(out, key=lambda r: (-r["violations"], -r["edges"], r["predicate"]))


# ── Entity admission: values and references ─────────────────────────────

_WORD_RE = re.compile(r"[A-Za-z0-9µμ°℃Ω%][A-Za-z0-9µμ°℃Ω%.,\-]*|[\u4e00-\u9fff]+|[×/~～\-–—+±]")
_VALUE_TOKEN_RE = re.compile(r"^[+\-±~≥≤<>]?\d[\d.,]*(?:[A-Za-zµμ°℃Ω%]{1,3})?$")
_UNIT_TOKEN_RE = re.compile(r"^[A-Za-zµμ°℃Ω%]{1,3}$")
_CJK_SHORT_RE = re.compile(r"^[\u4e00-\u9fff]{1,2}$")
_CONNECTOR_RE = re.compile(r"^(?:to|至|到|and|或|或者|x|×|/|-|–|—|~|～|\+|±)$", re.IGNORECASE)
_REFERENCE_RE = re.compile(
    r"^(?:(?:fig(?:ure)?|table|tab|图|表|附录|appendix|chapter|section|sec|章|节|page|p|note|注|footnote|equation|eq|公式)\.?\s*"
    r"[\dA-Za-z][\dA-Za-z.\-]*|\d{2,4}-\d{3,6}(?:\s*\*?[A-Za-z]{1,2})?|rev(?:ision)?\.?\s*\*?[A-Za-z0-9]{1,3}|\*[A-Za-z]{1,2})$",
    re.IGNORECASE)


def is_value_name(title: str) -> bool:
    """The name is just a value (number + unit / range / organization): 1.3V, 667 MHz, 2.7 V to 3.6 V, 1024 K × 8,
    65nm. ZK7C4021KV13 is one mixed alphanumeric run, not a value; 361-ball FCBGA has a long letter run, not a
    value."""
    text = unicodedata.normalize("NFKC", str(title or "")).strip()
    tokens = _WORD_RE.findall(text)
    if not tokens:
        return False
    numeric = 0
    for tok in tokens:
        if _VALUE_TOKEN_RE.match(tok):
            numeric += 1
        elif _CONNECTOR_RE.match(tok) or _UNIT_TOKEN_RE.match(tok) or _CJK_SHORT_RE.match(tok):
            continue
        else:
            return False
    return numeric >= 1


def is_reference_name(title: str) -> bool:
    """Figure / chapter / page / document number / revision marker: Figure 5, Table 3.2, 001-79553, Rev *J."""
    text = unicodedata.normalize("NFKC", str(title or "")).strip()
    return bool(text) and bool(_REFERENCE_RE.match(text))


def admit_entities(entities: list[dict[str, Any]], relations: list[dict[str, Any]]
                   ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Values do not become nodes: A -[pred]-> value folds into A's attribute description "pred = value", and the
    relation and value node are removed; references keep their node but are flagged reference, as are the
    relations touching them."""
    stats: dict[str, Any] = {"value_entities_dropped": 0, "value_relations_folded": 0,
                             "reference_entities": 0, "reference_relations": 0, "_values_by_type": {}}
    value_keys: dict[str, dict[str, Any]] = {}
    for e in entities:
        if e.get("exact"):
            continue           # rule-extracted entities (constant names, version numbers) are real, not values or references
        if is_value_name(e.get("title") or ""):
            value_keys[e["key"]] = e
            etype = str(e.get("type") or "")
            stats["_values_by_type"][etype] = stats["_values_by_type"].get(etype, 0) + 1
        elif is_reference_name(e.get("title") or ""):
            e["reference"] = True
    by_key = {e["key"]: e for e in entities}
    kept_relations: list[dict[str, Any]] = []
    for r in relations:
        sv, tv = r["source_key"] in value_keys, r["target_key"] in value_keys
        if sv or tv:
            stats["value_relations_folded"] += 1
            if sv and tv:
                continue
            owner = by_key.get(r["target_key"] if sv else r["source_key"])
            value = value_keys[r["source_key"] if sv else r["target_key"]]
            if owner is not None:
                attr = f"{r.get('predicate') or prompts.DEFAULT_PREDICATE} = {value.get('title')}"
                attrs = owner.setdefault("attributes", [])
                if attr not in attrs:
                    attrs.append(attr)
            continue
        kept_relations.append(r)
    kept_entities = [e for e in entities if e["key"] not in value_keys]
    stats["value_entities_dropped"] = len(value_keys)
    for e in kept_entities:
        if e.get("attributes"):
            line = f"{e.get('title')}: " + "; ".join(e["attributes"][:24])
            if line not in e["descriptions"]:
                e["descriptions"].append(line)
    ref_keys = {e["key"] for e in kept_entities if e.get("reference")}
    for r in kept_relations:
        if r["source_key"] in ref_keys or r["target_key"] in ref_keys:
            r["reference"] = True
    stats["reference_entities"] = len(ref_keys)
    stats["reference_relations"] = sum(1 for r in kept_relations if r.get("reference"))
    return kept_entities, kept_relations, stats


def dissolve_combined_names(entities: list[dict[str, Any]], relations: list[dict[str, Any]]
                            ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """同类实体的「A/B」并列名可拆开;类型不同的前缀/名称不能据斜杠推定为并列。

    例如组织/产品、作者/作品:把后者的描述和关系复制给前者会制造错误别名。
    不确定的合写名保留,交给后续实体归并判断,不丢掉原始证据。
    """
    by_key = {e["key"]: e for e in entities}
    split: dict[str, list[str]] = {}
    for e in entities:
        title = str(e.get("title") or "")
        if "/" not in title:
            continue
        parts = [p.strip() for p in title.split("/")]
        scope = str(e.get("scope") or "")
        keys = [scoped_key(entity_key(p), scope) if scope else entity_key(p) for p in parts]
        # 拆成同名异类的那一个(键带 ||上层类)也认:合写名的上层类与它一致时
        keys = [k if k in by_key else f"{k}{UPPER_KEY_SEPARATOR}{e.get('upper')}" for k in keys]
        if (len(parts) >= 2 and all(len(p) >= 3 for p in parts)
                and all(k in by_key and k != e["key"] for k in keys)
                and all((by_key[k].get("type"), by_key[k].get("upper"), by_key[k].get("scope"))
                        == (e.get("type"), e.get("upper"), e.get("scope")) for k in keys)):
            split[e["key"]] = keys
    if not split:
        return entities, relations, 0
    for combined_key, part_keys in split.items():
        combined = by_key[combined_key]
        for pk in part_keys:
            part = by_key[pk]
            for field in ("unit_ids", "doc_ids"):
                for item in combined.get(field) or []:
                    if item not in part[field]:
                        part[field].append(item)
            c_sources = list(combined.get("description_sources") or [])
            for i, d in enumerate(combined.get("descriptions") or []):
                if d not in part["descriptions"]:
                    part["descriptions"].append(d)
                    part.setdefault("description_sources", []).append(dict(c_sources[i]) if i < len(c_sources) else {})
            part["frequency"] = int(part.get("frequency") or 0) + int(combined.get("frequency") or 0)
            if combined.get("title") and combined["title"] not in part.get("aliases", []):
                part.setdefault("aliases", []).append(str(combined["title"]))
    kept_entities = [e for e in entities if e["key"] not in split]
    title_of = {e["key"]: e["title"] for e in kept_entities}
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    out: list[dict[str, Any]] = []
    for r in relations:
        sources = split.get(r["source_key"], [r["source_key"]])
        targets = split.get(r["target_key"], [r["target_key"]])
        placed: set[tuple[str, str, str]] = set()
        for source in sources:
            for target in targets:
                # the ends of an undirected edge are ordered by key: the swap holds for this pair of ends only and
                # must not change the loop variables carried into the next pass
                sk, tk = (target, source) if not r.get("directed") and target < source else (source, target)
                if sk == tk:
                    continue
                key = (sk, tk, str(r.get("predicate")))
                if key in placed:
                    continue             # two end pairs split from one relation land on the same key: counted once
                placed.add(key)
                if key in merged:
                    # an existing key absorbs the relation whichever arrived first: when the original relation comes
                    # after a split-off relation with the same key, its evidence counts all the same
                    slot = merged[key]
                    slot["strength_sum"] = round(float(slot.get("strength_sum") or 0) + float(r.get("strength_sum") or 0), 3)
                    r_sources = list(r.get("description_sources") or [])
                    for i, d in enumerate(r.get("descriptions") or []):
                        if d not in slot["descriptions"]:
                            slot["descriptions"].append(d)
                            slot.setdefault("description_sources", []).append(dict(r_sources[i]) if i < len(r_sources) else {})
                    for u in r.get("unit_ids") or []:
                        if u not in slot["unit_ids"]:
                            slot["unit_ids"].append(u)
                    slot["evidence"] = len(slot["unit_ids"])      # evidence = distinct units, the same unit is not counted twice
                    slot["boilerplate"] = bool(slot.get("boilerplate")) and bool(r.get("boilerplate"))
                    slot["evidence_kind"] = best_kind([slot.get("evidence_kind") or "body", r.get("evidence_kind") or "body"])
                    continue
                row = dict(r)
                row["source_key"], row["target_key"] = sk, tk
                row["source"], row["target"] = title_of.get(sk, row.get("source")), title_of.get(tk, row.get("target"))
                row["descriptions"] = list(r.get("descriptions") or [])
                row["description_sources"] = [dict(s) for s in (r.get("description_sources") or [])]
                row["unit_ids"] = list(r.get("unit_ids") or [])
                merged[key] = row
                out.append(row)
    return kept_entities, out, len(split)


def type_health(entities: list[dict[str, Any]], *, value_counts: dict[str, int] | None = None,
                ratio: float = TYPE_DEMOTE_RATIO, minimum: int = TYPE_DEMOTE_MIN) -> list[dict[str, Any]]:
    """Noise share per type: values (before admission), references, boilerplate-only sources. Types with share ≥
    ratio and instances ≥ minimum are marked demoted (a hint for the report; the merge phase no longer demotes
    whole types on it)."""
    value_counts = value_counts or {}
    rows: dict[str, dict[str, Any]] = {}
    for e in entities:
        etype = str(e.get("type") or "")
        slot = rows.setdefault(etype, {"type": etype, "entities": 0, "values": int(value_counts.get(etype, 0)),
                                       "references": 0, "boilerplate_only": 0})
        slot["entities"] += 1
        if e.get("reference"):
            slot["references"] += 1
        if e.get("boilerplate"):
            slot["boilerplate_only"] += 1
    for etype, count in value_counts.items():
        rows.setdefault(etype, {"type": etype, "entities": 0, "values": int(count), "references": 0, "boilerplate_only": 0})
    out: list[dict[str, Any]] = []
    for slot in rows.values():
        total = slot["entities"] + slot["values"]
        noise = slot["values"] + slot["references"] + slot["boilerplate_only"]
        slot["noise_ratio"] = round(noise / total, 3) if total else 0.0
        slot["demoted"] = bool(total >= minimum and slot["noise_ratio"] >= ratio and slot["entities"] - slot["references"] > 0)
        out.append(slot)
    return sorted(out, key=lambda r: (-r["noise_ratio"], r["type"]))


# ── Grounding check for capability questions ────────────────────────────

_QUESTION_TOKEN_RE = re.compile(r"[A-Za-z0-9_#]+(?:[\[\]:./+-][A-Za-z0-9_#]+)*")


def question_identifiers(question: str) -> list[str]:
    """Symbols in the question: containing digits, or all upper-case (≥ 2 letters: two-letter pin names like AP,
    CE, WE count), or mixed case (tAS, PE#, ZK7C4021KV13, IDCODE). Lexical seeds match titles exactly, so a
    two-letter word only matters when the graph really has that name."""
    out: list[str] = []
    for tok in _QUESTION_TOKEN_RE.findall(str(question or "")):
        if len(tok) < 2:
            continue
        upper = sum(1 for ch in tok if ch.isupper())
        lower = sum(1 for ch in tok if ch.islower())
        letters = upper + lower
        # Names from code: snake_case, dotted module names, file names with path / extension (team_wiki_search,
        # lib.v7_env, lib/http_client.py)
        code_like = letters >= 2 and any(ch in tok for ch in "_./")
        if any(ch.isdigit() for ch in tok) or upper >= 3 or (upper >= 2 and lower == 0) or (upper >= 1 and lower >= 1) or code_like:
            if tok not in out:
                out.append(tok)
    return out




def finalize_descriptions(rows: list[dict[str, Any]], *, max_descriptions: int = 8) -> None:
    """Rows that were not summarized (a single description, or the summary failed): description = the
    deduplicated descriptions joined."""
    for row in rows:
        if not row.get("description"):
            row["description"] = "\n".join((row.get("descriptions") or [])[:max_descriptions])[:4000]


def pagerank(nodes: Iterable[str], edges: Iterable[tuple[str, str, float]], *, damping: float = PAGERANK_DAMPING,
             iterations: int = PAGERANK_ITERATIONS) -> dict[str, float]:
    """Edge-weighted undirected pagerank (power iteration). The graph is small (tens of thousands of nodes), not
    worth pulling in networkx."""
    nodes = list(dict.fromkeys(nodes))
    n = len(nodes)
    if n == 0:
        return {}
    index = {node: i for i, node in enumerate(nodes)}
    out_weight = [0.0] * n
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for a, b, w in edges:
        if a not in index or b not in index or a == b:
            continue
        w = max(0.0, float(w))
        if w <= 0:
            continue
        ia, ib = index[a], index[b]
        adjacency[ia].append((ib, w))
        adjacency[ib].append((ia, w))
        out_weight[ia] += w
        out_weight[ib] += w
    rank = [1.0 / n] * n
    base = (1.0 - damping) / n
    for _ in range(iterations):
        nxt = [base] * n
        dangling = sum(rank[i] for i in range(n) if out_weight[i] == 0)
        share = damping * dangling / n
        for i in range(n):
            if out_weight[i] == 0:
                continue
            push = damping * rank[i] / out_weight[i]
            for j, w in adjacency[i]:
                nxt[j] += push * w
        rank = [v + share for v in nxt]
        total = sum(rank)
        if total > 0:
            rank = [v / total for v in rank]
    return {node: rank[index[node]] for node in nodes}


def compute_weights(entities: list[dict[str, Any]], relations: list[dict[str, Any]], *, n_units: int) -> dict[str, Any]:
    """Fill in degree / combined_degree / pagerank / cooccur / npmi / weight in place."""
    by_key = {e["key"]: e for e in entities}
    neighbours: dict[str, set[str]] = {k: set() for k in by_key}
    for r in relations:
        neighbours.setdefault(r["source_key"], set()).add(r["target_key"])
        neighbours.setdefault(r["target_key"], set()).add(r["source_key"])
    for e in entities:
        e["degree"] = len(neighbours.get(e["key"], ()))
    ranks = pagerank(by_key.keys(), ((r["source_key"], r["target_key"], r["strength_sum"]) for r in relations))
    max_rank = max(ranks.values(), default=0.0) or 1.0
    for e in entities:
        e["pagerank"] = round(ranks.get(e["key"], 0.0), 8)
        e["pagerank_norm"] = round(ranks.get(e["key"], 0.0) / max_rank, 6)
    units_of = {e["key"]: set(e.get("unit_ids") or []) for e in entities}
    n = max(1, int(n_units))
    for r in relations:
        src, tgt = units_of.get(r["source_key"], set()), units_of.get(r["target_key"], set())
        both = len(src & tgt)
        r["cooccur"] = both
        r["combined_degree"] = int(by_key[r["source_key"]]["degree"] + by_key[r["target_key"]]["degree"])
        px, py, pxy = len(src) / n, len(tgt) / n, both / n
        if pxy > 0 and px > 0 and py > 0 and pxy < 1:
            pmi = math.log(pxy / (px * py))
            r["npmi"] = round(pmi / (-math.log(pxy)), 6)
        elif pxy >= 1:
            r["npmi"] = 1.0
        else:
            r["npmi"] = 0.0
    coeffs = weight_coefficients()
    max_strength = max((r["strength_sum"] for r in relations), default=0.0) or 1.0
    max_cooccur = max((r["cooccur"] for r in relations), default=0) or 1
    for r in relations:
        score = (coeffs["strength"] * (r["strength_sum"] / max_strength)
                 + coeffs["cooccur"] * (r["cooccur"] / max_cooccur)
                 + coeffs["npmi"] * (r["npmi"] + 1.0) / 2.0)
        denominator = sum(coeffs.values()) or 1.0
        r["weight"] = round(1.0 + 9.0 * max(0.0, min(1.0, score / denominator)), 4)
    return {"coefficients": coeffs, "n_units": n}


_FOLD_RE = re.compile(r"\s+")


def fold_text(text: str) -> str:
    return _FOLD_RE.sub(" ", unicodedata.normalize("NFKC", str(text or ""))).casefold()


_ASCII_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+/-]*")


def _needle_matcher(needle: str):
    """Counter of an entity name in text. Pure-ASCII names are matched on word boundaries: CE must not hit inside
    CELL, nor AP inside CAPACITY (health check B10); Chinese and other names still match as substrings, since
    Chinese has no word boundaries."""
    if _ASCII_NAME_RE.fullmatch(needle):
        pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(needle) + r"(?![A-Za-z0-9])")
        return lambda text: len(pattern.findall(text))
    return lambda text: text.count(needle)


def attribute_mentions(entities: list[dict[str, Any]], units_by_id: dict[str, Unit],
                       chunk_texts: dict[str, str], chunk_sections: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Exact entity → chunk attribution (MENTIONED_IN). Returns [{entity_key, point_id, chunk_uid, count}].
    chunk_sections: the chunks' section paths; a section-heading entity ("TAP Registers") appears only in the
    heading and is not repeated in the body, so searching the body alone would attach it to the contents page."""
    folded: dict[str, str] = {}
    folded_sections: dict[str, str] = {}
    chunk_sections = chunk_sections or {}
    out: list[dict[str, Any]] = []
    for e in entities:
        needles = {fold_text(e["title"])} | {fold_text(a) for a in (e.get("aliases") or [])}
        needles = {n for n in needles if len(n) >= 2}
        matchers = [_needle_matcher(n) for n in sorted(needles)]
        rows: dict[str, dict[str, Any]] = {}
        for unit_id in e.get("unit_ids") or []:
            unit = units_by_id.get(unit_id)
            if unit is None:
                continue
            hits: list[tuple[str, str, int]] = []
            for point_id, chunk_uid in zip(unit.point_ids, unit.chunk_uids):
                text = folded.get(point_id)
                if text is None:
                    text = folded[point_id] = fold_text(chunk_texts.get(point_id, ""))
                section = folded_sections.get(point_id)
                if section is None:
                    section = folded_sections[point_id] = fold_text(chunk_sections.get(point_id, ""))
                count = sum(m(text) for m in matchers) if text else 0
                if section:
                    count += sum(1 for m in matchers if m(section))
                if count:
                    hits.append((point_id, chunk_uid, count))
            if not hits:
                hits = [(pid, cid, 0) for pid, cid in zip(unit.point_ids, unit.chunk_uids)]
            for point_id, chunk_uid, count in hits:
                slot = rows.get(point_id)
                if slot is None:
                    rows[point_id] = {"entity_key": e["key"], "point_id": point_id, "chunk_uid": chunk_uid, "count": count}
                else:
                    slot["count"] += count
        out.extend(rows.values())
    return out
