"""上层本体与图谱模式(2026-10-24 破坏性改版)。

上层类固定八个,不按库配置:
  agent 人员与组织 · entity 实体 · part 部件 · property 属性 · process 过程(可重复的做法)·
  event 事件(发生在某个时间的事)· concept 概念(标准、协议、术语、分类)· document 文档。

每个库的模式(自动归纳的、人工上传的)以 LinkML 子集为原生格式存进版本环:
· 类型是 class,is_a 指向上层类或另一个类型(多层);abstract 的类型不进抽取菜单,只用来按上层查询;
· 关系是 slot,domain / range(或 any_of)指向类;没写 multivalued 的关系按多值处理,
  只有显式写 multivalued: false 的才做「新值取代旧值」;
· 作用域(global 跨文档 / document 按文档隔开)、语言、场景画像放 annotations。
Carrel 只读写这个子集;完整校验、生成 OWL / SHACL 留在插件侧用 LinkML 工具链做。

模块之间传的是规范化后的模式字典(model):
  {"language": str,
   "types": [{"name", "is_a", "description", "scope", "abstract"}],
   "predicates": [{"name", "description", "source_parents", "target_parents", "multivalued", "core"}],
   "profile": {...}}
expand_model 把它展开成建图、抽取、合并、检索目录实际读的 graph_* 生效字段。
"""
from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Iterable

ROOT_CLASS = "thing"
UPPER_CLASSES = ("agent", "entity", "part", "property", "process", "event", "concept", "document")
UPPER_DESCRIPTIONS = {
    "agent": "people and organizations that act or decide: named persons, companies, institutions, departments, teams, committees",
    "entity": "independent things that exist on their own: products, devices, systems, projects, places, materials, datasets",
    "part": "constituents that belong to an entity: components, pins, modules, members, fields, ingredients",
    "property": "measurable or descriptive attributes: parameters, metrics, characteristics, ratings, settings",
    "process": "repeatable ways of doing things: procedures, methods, workflows, operating modes, practices",
    "event": "things that happen at a point or span of time: meetings, decisions, incidents, inspections, examinations, releases, action items",
    "concept": "standards, protocols, technologies, terms, categories and ideas shared across documents",
    "document": "documents, sections, figures, tables and other references to material",
}
SCOPES = ("global", "document")
# 没有逐类型判定时的默认作用域:部件 / 属性 / 过程 / 事件属于某份材料里的某个东西,按文档隔开;其余跨文档。
# 事件另有一条规则(merge 里):名字里带日期的事件跨文档认同一个(「2026-03-14 设计评审」在纪要和周报里是同一场)。
DEFAULT_SCOPE = {"part": "document", "property": "document", "process": "document", "event": "document"}
# 模型或人写的上层类别名 → 八类之一(只认意思明确的;认不出的由调用方决定兜底)
UPPER_SYNONYMS = {
    "person": "agent", "people": "agent", "organization": "agent", "organisation": "agent", "actor": "agent",
    "人员": "agent", "人物": "agent", "组织": "agent", "机构": "agent", "人员与组织": "agent",
    "实体": "entity", "object": "entity", "thing": "entity", "product": "entity", "device": "entity", "system": "entity",
    "部件": "part", "零件": "part", "组件": "part", "component": "part", "constituent": "part", "interface": "part",
    "pin": "part", "signal": "part", "module": "part", "接口": "part",
    "mode": "process", "模式": "process", "操作": "process",
    "location": "entity", "place": "entity", "material": "entity",
    "属性": "property", "参数": "property", "指标": "property", "attribute": "property", "parameter": "property",
    "metric": "property", "characteristic": "property",
    "过程": "process", "流程": "process", "方法": "process", "procedure": "process", "method": "process",
    "workflow": "process", "operation": "process",
    "事件": "event", "occurrence": "event", "happening": "event", "meeting": "event", "incident": "event",
    "概念": "concept", "标准": "concept", "术语": "concept", "standard": "concept", "term": "concept",
    "technology": "concept", "protocol": "concept", "category": "concept", "idea": "concept",
    "文档": "document", "资料": "document", "reference": "document",
}
# Leaf types add specificity to an upper class; reusing an upper-class name adds no information.
UPPER_NAMES = frozenset({*UPPER_CLASSES, ROOT_CLASS, "人员与组织", "实体", "部件", "属性", "过程", "事件", "概念", "文档"})


def is_upper_name(value: Any) -> bool:
    return unicodedata.normalize("NFKC", str(value or "")).strip().casefold() in UPPER_NAMES


# 事件之间的核心关系:模式里有事件类型就自动带上,不占归纳出来的谓词名额。follows 归一成反向的 precedes。
EVENT_PREDICATES = (
    {"name": "precedes", "description": "the source event happened before the target event",
     "source_parents": ["event"], "target_parents": ["event"]},
    {"name": "concurrent_with", "description": "the two events happened at the same time or overlapped",
     "source_parents": ["event"], "target_parents": ["event"]},
    {"name": "causes", "description": "the source event or process brought about the target event",
     "source_parents": ["event", "process"], "target_parents": ["event"]},
    {"name": "results_in", "description": "the source event or process produced the target outcome: a decision, a document, a thing or a state",
     "source_parents": ["event", "process"], "target_parents": ["*"]},
)
INVERSE_PREDICATES = {"follows": "precedes", "preceded_by": "precedes", "caused_by": "causes", "result_of": "results_in"}

# 组成关系:谓词是各库归纳出来的(has_component、composed_of、part_of……),按名字认两个方向。主体页按它汇总
# 各组成部分的事实(compile)。归属类(belongs_to、member_of)不算:人属于部门不是部门的组成部分
WHOLE_TO_PART_PREDICATES = frozenset({
    "has_part", "has_parts", "has_component", "has_components", "has_module", "has_modules", "has_subsystem", "has_subsystems",
    "composed_of", "consists_of", "comprises", "contains", "includes", "made_of", "made_up_of"})
PART_TO_WHOLE_PREDICATES = frozenset({"part_of", "component_of", "module_of", "subsystem_of"})


def composition_ends(predicate: Any, source_key: str, target_key: str) -> tuple[str, str] | None:
    """组成关系的 (整体, 部分);不是组成关系返回 None。"""
    name = normalize_predicate_name(predicate)
    if name in WHOLE_TO_PART_PREDICATES:
        return source_key, target_key
    if name in PART_TO_WHOLE_PREDICATES:
        return target_key, source_key
    return None

# 没有任何模式版本时的兜底类型表(正常流程里空白图建图前会先自动归纳一版,这一份只在归纳失败时用)
DEFAULT_TYPES = (
    ("organization", "agent", "a company, institution, department, team or committee"),
    ("person", "agent", "a named human being"),
    ("product", "entity", "a product, system or service that is offered or used"),
    ("project", "entity", "a named project or initiative"),
    ("module", "part", "a component, module or feature of a product or system"),
    ("technology", "concept", "a technology, standard, protocol or method shared across documents"),
    ("requirement", "concept", "a requirement, need or constraint stated for something"),
    ("risk", "concept", "a risk, issue or problem"),
    ("procedure", "process", "a repeatable procedure, workflow or method"),
    ("meeting", "event", "a meeting, review, decision or incident that happened at a particular time"),
    ("document", "document", "a document, report or file referred to in the text"),
)

MAX_TYPES = 64              # 含 abstract 的中间层
MAX_MENU_TYPES = 32         # 进抽取菜单的(非 abstract)类型
MAX_PREDICATES = 28         # 含事件核心关系
TYPE_NAME_MAX_LEN = 64
DESCRIPTION_MAX_CHARS = 240
LINKML_PREFIXES = {"linkml": "https://w3id.org/linkml/", "carrel": "urn:carrel:"}
_PRIMITIVE_RANGES = {"string", "integer", "float", "double", "decimal", "boolean", "date", "datetime", "time", "uri",
                     "uriorcurie", "curie", "ncname", "objectidentifier", "nodeidentifier", "jsonpointer", "jsonpath",
                     "sparqlpath", "date_or_datetime"}
_PREDICATE_CLEAN_RE = re.compile(r"[^a-z0-9_]+")


class SchemaIssue(str):
    def __new__(cls, code, zh, en, **params):
        obj = super().__new__(cls, zh)
        obj.detail = {"code": code, "params": params, "message": {"zh": zh, "en": en}}
        return obj


class SchemaError(ValueError):
    """上传或归纳出来的模式不合规(带全部错误,控制台逐条显示)。"""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = list(errors)
        self.details = [getattr(e, "detail", {"code": "schema_invalid", "params": {},
                        "message": {"zh": str(e), "en": str(e)}}) for e in errors]


def upper_of_name(value: Any) -> str | None:
    """上层类名或同义词 → 八类之一;认不出返回 None。"""
    key = unicodedata.normalize("NFKC", str(value or "")).strip().casefold()
    if not key:
        return None
    if key in UPPER_CLASSES:
        return key
    return UPPER_SYNONYMS.get(key)


def normalize_predicate_name(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip().lower().replace("-", "_").replace(" ", "_")
    return _PREDICATE_CLEAN_RE.sub("", text).strip("_")


def _text(value: Any, limit: int = DESCRIPTION_MAX_CHARS) -> str:
    return " ".join(str(value or "").split())[:limit]


def _ends(value: Any, types: dict[str, str]) -> list[str]:
    """端点清单:上层类、类型名或 "*";类型名按大小写不敏感对齐成表里的写法。"""
    if isinstance(value, str):
        value = re.split(r"[,\s|]+", value)
    if not isinstance(value, (list, tuple)):
        return []
    out: list[str] = []
    for v in value:
        raw = str(v or "").strip()
        if not raw:
            continue
        if raw == "*" or raw.casefold() == ROOT_CLASS:
            name = "*"
        else:
            name = upper_of_name(raw) if raw.casefold() not in types else None
            name = name or types.get(raw.casefold(), "")
        if name and name not in out:
            out.append(name)
    return ["*"] if "*" in out else out[:12]


def normalize_model(raw: dict[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    """把归纳结果或 LinkML 读出来的模式字典规范化;返回 (model, 警告)。硬错误抛 SchemaError。
    规则:类型名去重(大小写不敏感,保留首次写法)、is_a 只能指向上层类或表内类型且不能成环、作用域不认识的
    按上层默认、谓词名 lower_snake 并去掉兜底的 related_to;有事件类型就补上事件核心关系。"""
    raw = raw or {}
    errors: list[str] = []
    warnings: list[str] = []
    types: list[dict[str, Any]] = []
    lookup: dict[str, str] = {}
    for item in raw.get("types") or []:
        if isinstance(item, str):
            item = {"name": item}
        if not isinstance(item, dict):
            continue
        name = unicodedata.normalize("NFKC", str(item.get("name") or "")).strip()
        if not name:
            continue
        if len(name) > TYPE_NAME_MAX_LEN:
            errors.append(SchemaIssue("type_name_too_long", f"类型名过长(上限 {TYPE_NAME_MAX_LEN} 字符):{name[:20]}…", f"Type name exceeds {TYPE_NAME_MAX_LEN} characters: {name[:20]}…", path=f"classes.{name}", limit=TYPE_NAME_MAX_LEN))
            continue
        if name.casefold() == ROOT_CLASS:
            warnings.append(f"类型名 {name} 是保留的根类名,已忽略")
            continue
        if name.casefold() in UPPER_CLASSES:
            # 上层类直接当类型用(归纳出来的类型表里常有 document / event):名字统一成上层类的写法,父类就是它自己
            name = name.casefold()
        if name.casefold() in lookup:
            warnings.append(f"类型 {name} 重复,只保留第一个")
            continue
        lookup[name.casefold()] = name
        scope = str(item.get("scope") or "").strip().lower()
        types.append({"name": name, "is_a": str(item.get("is_a") or item.get("parent") or "").strip(),
                      "description": _text(item.get("description")),
                      "scope": scope if scope in SCOPES else "",
                      "abstract": bool(item.get("abstract"))})
    # is_a:上层类(含同义词)或表内类型;都不是的挂到 entity 并给警告。上层类直接当类型用的,父类固定是它自己
    for t in types:
        parent = t["is_a"]
        if t["name"] in UPPER_CLASSES:
            if parent and parent.casefold() != t["name"]:
                warnings.append(f"类型 {t['name']} 就是上层类,父类 {parent} 不生效")
            t["is_a"] = t["name"]
            continue
        if parent.casefold() in lookup and lookup[parent.casefold()] != t["name"]:
            t["is_a"] = lookup[parent.casefold()]
        elif upper_of_name(parent):
            t["is_a"] = upper_of_name(parent)
        else:
            if parent:
                warnings.append(f"类型 {t['name']} 的父类 {parent} 不是上层类也不在类型表里,按 entity 处理")
            t["is_a"] = "entity"
    by_name = {t["name"]: t for t in types}
    for t in types:
        seen = {t["name"]}
        cur = t["is_a"]
        while cur in by_name and cur not in UPPER_CLASSES:
            if cur in seen:
                errors.append(SchemaIssue("inheritance_cycle", f"类型 {t['name']} 的父类链成环", f"Inheritance cycle at type {t['name']}", path=f"classes.{t['name']}.is_a"))
                break
            seen.add(cur)
            cur = by_name[cur]["is_a"]
    if not errors:
        for t in types:
            if not t["scope"]:
                t["scope"] = _inherited_scope(t["name"], by_name)
    menu = [t for t in types if not t["abstract"]]
    if not menu:
        errors.append(SchemaIssue("no_concrete_types", "至少要有一个可抽取的(非 abstract)类型", "At least one non-abstract type is required", path="classes"))
    if len(menu) > MAX_MENU_TYPES:
        errors.append(SchemaIssue("too_many_concrete_types", f"可抽取的类型最多 {MAX_MENU_TYPES} 个(当前 {len(menu)} 个);中间层可以标 abstract", f"At most {MAX_MENU_TYPES} non-abstract types are allowed (got {len(menu)}); intermediate classes can be abstract", path="classes", limit=MAX_MENU_TYPES, actual=len(menu)))
    if len(types) > MAX_TYPES:
        errors.append(SchemaIssue("too_many_types", f"类型总数最多 {MAX_TYPES} 个(当前 {len(types)} 个)", f"At most {MAX_TYPES} types are allowed (got {len(types)})", path="classes", limit=MAX_TYPES, actual=len(types)))
    predicates: list[dict[str, Any]] = []
    seen_preds: set[str] = set()
    for item in raw.get("predicates") or []:
        if isinstance(item, str):
            item = {"name": item}
        if not isinstance(item, dict):
            continue
        name = normalize_predicate_name(item.get("name"))
        name = INVERSE_PREDICATES.get(name, name) if item.get("core") else name
        if not name or name == "related_to" or name in seen_preds:
            continue
        seen_preds.add(name)
        multivalued = item.get("multivalued")
        predicates.append({
            "name": name,
            "description": _text(item.get("description"), 200),
            "source_parents": _ends(item.get("source_parents"), lookup) or ["*"],
            "target_parents": _ends(item.get("target_parents"), lookup) or ["*"],
            "multivalued": multivalued is not False and str(multivalued).strip().lower() not in ("false", "0", "no"),
            "core": bool(item.get("core")),
        })
    has_event = any(upper_of(t["name"], by_name) == "event" for t in types) if not errors else False
    if has_event:
        for core in EVENT_PREDICATES:
            if core["name"] not in seen_preds:
                predicates.append({**core, "source_parents": list(core["source_parents"]),
                                   "target_parents": list(core["target_parents"]), "multivalued": True, "core": True})
                seen_preds.add(core["name"])
    if len(predicates) > MAX_PREDICATES:
        errors.append(SchemaIssue("too_many_relations", f"关系最多 {MAX_PREDICATES} 种(当前 {len(predicates)} 种,含事件核心关系)", f"At most {MAX_PREDICATES} relations are allowed (got {len(predicates)}, including event relations)", path="slots", limit=MAX_PREDICATES, actual=len(predicates)))
    profile = _normalize_profile(raw.get("profile"), lookup, seen_preds)
    if errors:
        raise SchemaError(errors)
    return {"language": _text(raw.get("language"), 40), "types": types, "predicates": predicates, "profile": profile}, warnings


def _inherited_scope(name: str, by_name: dict[str, dict[str, Any]]) -> str:
    """没判定作用域的类型:沿 is_a 找最近一个判定过的祖先,都没有就按上层类的默认。"""
    cur = by_name.get(name)
    hops = 0
    while cur is not None and hops <= len(by_name):
        if cur.get("scope") in SCOPES and cur["name"] != name:
            return cur["scope"]
        parent = cur.get("is_a") or ""
        if parent in by_name and parent not in UPPER_CLASSES:
            cur = by_name[parent]
            hops += 1
            continue
        return DEFAULT_SCOPE.get(upper_of_name(parent) or "entity", "global")
    return "global"


def upper_of(name: str, by_name: dict[str, dict[str, Any]]) -> str:
    """沿 is_a 走到上层类;走不到(数据不完整)按 entity。"""
    cur = name
    for _ in range(len(by_name) + 1):
        if cur in UPPER_CLASSES:
            return cur
        row = by_name.get(cur)
        if row is None:
            return upper_of_name(cur) or "entity"
        cur = row.get("is_a") or "entity"
    return "entity"


def ancestors(name: str, by_name: dict[str, dict[str, Any]]) -> list[str]:
    """类型的祖先链(不含自己),由近到远,最后是上层类;上层类直接当类型用的,链是空的。"""
    out: list[str] = []
    if name in UPPER_CLASSES:
        return out
    cur = (by_name.get(name) or {}).get("is_a") or ""
    for _ in range(len(by_name) + 1):
        if not cur:
            break
        out.append(cur)
        if cur in UPPER_CLASSES:
            break
        row = by_name.get(cur)
        if row is None:
            break
        cur = row.get("is_a") or ""
    return out


def type_paths(tree: dict[str, str]) -> dict[str, list[str]]:
    """{类型: [类型, 父类, …, 上层类]}(casefold 后的键):按祖先过滤、端点约束按祖先判都用这一份。
    tree 是生效字段 graph_type_parents。表外的类型(抽取冒出来的)只有它自己,调用方再补上层类。"""
    by_name = {str(k): {"name": str(k), "is_a": str(v)} for k, v in (tree or {}).items()}
    out: dict[str, list[str]] = {}
    for name in by_name:
        out[name.casefold()] = [name.casefold()] + [a.casefold() for a in ancestors(name, by_name)]
    return out


def _normalize_profile(value: Any, types: dict[str, str], predicates: set[str]) -> dict[str, Any]:
    from .limits import normalize_profile

    profile = normalize_profile(value)
    if not profile:
        return {}
    profile["subject_types"] = [types[t.casefold()] for t in profile.get("subject_types") or [] if t.casefold() in types]
    profile["extension_predicates"] = [p for p in profile.get("extension_predicates") or [] if p in predicates]
    return profile


def default_model() -> dict[str, Any]:
    model, _ = normalize_model({"types": [{"name": n, "is_a": p, "description": d} for n, p, d in DEFAULT_TYPES]})
    return model


def entry_model(entry: dict[str, Any] | None) -> dict[str, Any] | None:
    """版本条目 → 规范化的模式字典(条目里存的是 LinkML);条目坏了返回 None。"""
    if not entry or not isinstance(entry.get("linkml"), dict):
        return None
    try:
        model, _ = from_linkml(entry["linkml"])
    except SchemaError:
        return None
    return model


def expand_model(model: dict[str, Any]) -> dict[str, Any]:
    """规范化的模式 → 各模块读的生效字段。抽取菜单只放非 abstract 的类型;父类表给的是上层类(作用域、端点约束、
    检索目录都按它),类型树另存一份供按祖先查询。"""
    types = list(model.get("types") or [])
    by_name = {t["name"]: t for t in types}
    menu = [t["name"] for t in types if not t.get("abstract")]
    return {
        "graph_entity_types": menu or None,
        "graph_language": model.get("language") or None,
        "graph_predicates": [dict(p) for p in model.get("predicates") or []] or None,
        "graph_parent_types": {t["name"]: upper_of(t["name"], by_name) for t in types} or None,
        "graph_type_parents": {t["name"]: t["is_a"] for t in types} or None,
        "graph_type_scopes": {t["name"]: t.get("scope") or "global" for t in types} or None,
        "graph_type_definitions": {t["name"]: t["description"] for t in types if t.get("description")} or None,
        "graph_profile": dict(model.get("profile") or {}) or None,
    }


def model_from_effective(config: dict[str, Any]) -> dict[str, Any]:
    """生效字段 → 模式字典(控制台在现行配置上改一处再存成新版本时用)。"""
    types_menu = [str(t) for t in (config.get("graph_entity_types") or []) if str(t).strip()]
    tree = dict(config.get("graph_type_parents") or {})
    uppers = dict(config.get("graph_parent_types") or {})
    scopes = dict(config.get("graph_type_scopes") or {})
    defs = dict(config.get("graph_type_definitions") or {})
    names = list(dict.fromkeys([*tree.keys(), *types_menu]))
    menu = {t.casefold() for t in types_menu}
    types = [{"name": n, "is_a": tree.get(n) or uppers.get(n) or "entity", "description": defs.get(n, ""),
              "scope": scopes.get(n, ""), "abstract": n.casefold() not in menu} for n in names]
    return {"language": config.get("graph_language") or "", "types": types,
            "predicates": list(config.get("graph_predicates") or []), "profile": dict(config.get("graph_profile") or {})}


# ── LinkML 子集 ─────────────────────────────────────────────────────────


def upper_linkml_classes() -> dict[str, Any]:
    """固定上层类的 LinkML 定义(根类 thing + 八个 abstract 类)。每份导出的模式都自带这一段,
    不依赖外部 import 也能被 LinkML 工具链直接加载。"""
    classes: dict[str, Any] = {ROOT_CLASS: {"abstract": True, "description": "root of the Carrel upper ontology",
                                            "annotations": {"carrel_upper": "true"}}}
    for name in UPPER_CLASSES:
        classes[name] = {"is_a": ROOT_CLASS, "abstract": True, "description": UPPER_DESCRIPTIONS[name],
                         "annotations": {"carrel_upper": "true", "carrel_scope": DEFAULT_SCOPE.get(name, "global")}}
    return classes


def to_linkml(model: dict[str, Any], *, name: str, version: str = "", title: str = "", description: str = "") -> dict[str, Any]:
    """模式字典 → LinkML 文档(dict,自包含:上层类内联,只 import linkml:types)。"""
    schema_name = re.sub(r"[^A-Za-z0-9_]+", "_", str(name or "schema")).strip("_") or "schema"
    doc: dict[str, Any] = {
        "id": f"urn:carrel:schema:{schema_name}",
        "name": schema_name,
    }
    if title:
        doc["title"] = str(title)
    if description:
        doc["description"] = _text(description, 1000)
    if version:
        doc["version"] = str(version)
    doc.update({
        "prefixes": dict(LINKML_PREFIXES),
        "default_prefix": "carrel",
        "default_range": "string",
        "imports": ["linkml:types"],
    })
    annotations: dict[str, Any] = {"carrel_format": "carrel-linkml-1"}
    if model.get("language"):
        annotations["carrel_language"] = str(model["language"])
    if model.get("profile"):
        annotations["carrel_profile"] = json.dumps(model["profile"], ensure_ascii=False, sort_keys=True)
    doc["annotations"] = annotations
    uppers = upper_linkml_classes()
    used = {t["name"] for t in model.get("types") or [] if t["name"] in UPPER_CLASSES}
    # 没被直接当类型用的上层类排在前面;直接当类型用的放在类型表里它原来的位置(类型顺序就是抽取菜单的顺序)
    classes = {k: v for k, v in uppers.items() if k not in used}
    predicates = list(model.get("predicates") or [])
    for t in model.get("types") or []:
        if t["name"] in UPPER_CLASSES:
            # 上层类直接当类型用:那一个上层类不再是 abstract,作用域与定义按本库的
            upper_row = uppers[t["name"]]
            upper_row.pop("abstract", None)
            if t.get("description"):
                upper_row["description"] = t["description"]
            upper_row["annotations"]["carrel_scope"] = t.get("scope") or DEFAULT_SCOPE.get(t["name"], "global")
            classes[t["name"]] = upper_row
            continue
        row: dict[str, Any] = {"is_a": t.get("is_a") or "entity"}
        if t.get("abstract"):
            row["abstract"] = True
        if t.get("description"):
            row["description"] = t["description"]
        row["annotations"] = {"carrel_scope": t.get("scope") or "global"}
        classes[t["name"]] = row
    # 关系挂到来源端的类上(LinkML 的惯用写法:类声明自己有哪些 slot);来源是 * 的挂到根类
    for p in predicates:
        for src in p.get("source_parents") or ["*"]:
            owner = ROOT_CLASS if src == "*" else src
            if owner in classes:
                classes[owner].setdefault("slots", [])
                if p["name"] not in classes[owner]["slots"]:
                    classes[owner]["slots"].append(p["name"])
    doc["classes"] = classes
    slots: dict[str, Any] = {}
    for p in predicates:
        targets = [ROOT_CLASS if t == "*" else t for t in (p.get("target_parents") or ["*"])]
        row = {}
        if p.get("description"):
            row["description"] = p["description"]
        if len(targets) == 1:
            row["range"] = targets[0]
        else:
            row["range"] = ROOT_CLASS
            row["any_of"] = [{"range": t} for t in targets]
        row["multivalued"] = bool(p.get("multivalued", True))
        sources = list(p.get("source_parents") or ["*"])
        if len(sources) == 1 and sources[0] != "*":
            row["domain"] = sources[0]
        ann = {"carrel_source_parents": json.dumps(sources, ensure_ascii=False), "carrel_target_parents": json.dumps(p.get("target_parents") or ["*"], ensure_ascii=False)}
        if p.get("core"):
            ann["carrel_core"] = "true"
        row["annotations"] = ann
        slots[p["name"]] = row
    if slots:
        doc["slots"] = slots
    return doc


def dump_linkml(doc: dict[str, Any]) -> str:
    import yaml

    return yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=1000)


def load_linkml(text: str | bytes | dict[str, Any]) -> dict[str, Any]:
    if isinstance(text, dict):
        return text
    import yaml

    if isinstance(text, bytes):
        text = text.decode("utf-8-sig")
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SchemaError([SchemaIssue("invalid_yaml", f"不是合法的 YAML:{exc}", f"Invalid YAML: {exc}", path="$")]) from exc
    if not isinstance(doc, dict):
        raise SchemaError([SchemaIssue("invalid_root", "LinkML 文档的顶层必须是映射(id / name / classes / slots …)", "The LinkML root must be a mapping (id / name / classes / slots)", path="$")])
    return doc


def _ann(row: dict[str, Any], key: str) -> Any:
    """读 annotation:紧凑写法 {tag: value} 与展开写法 {tag: {tag, value}} 都认。"""
    anns = row.get("annotations") or {}
    if isinstance(anns, list):
        anns = {str(a.get("tag")): a for a in anns if isinstance(a, dict)}
    val = anns.get(key) if isinstance(anns, dict) else None
    if isinstance(val, dict) and "value" in val:
        val = val.get("value")
    return val


def _named(section: Any) -> dict[str, dict[str, Any]]:
    """classes / slots 既可能是 {名字: 定义} 也可能是 [{name: …}] 列表。"""
    if isinstance(section, dict):
        return {str(k): (v if isinstance(v, dict) else {}) for k, v in section.items()}
    if isinstance(section, list):
        return {str(v.get("name")): v for v in section if isinstance(v, dict) and v.get("name")}
    return {}


def from_linkml(doc: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """LinkML 文档 → (规范化的模式字典, 警告)。认的子集:classes(is_a / abstract / description / attributes /
    slots / slot_usage / annotations.carrel_scope)、slots(domain / range / any_of / exactly_one_of / multivalued /
    description / annotations.carrel_source_parents / carrel_target_parents)、顶层 annotations(carrel_language /
    carrel_profile)。range 是基本类型的 slot 是属性不是关系,跳过并给警告;enums / types / subsets 不读。"""
    warnings: list[str] = []
    classes = _named(doc.get("classes"))
    slots = _named(doc.get("slots"))
    for section in ("enums", "types", "subsets"):
        if doc.get(section):
            warnings.append(f"{section} 不在 Carrel 读取的子集里,已忽略")
    types: list[dict[str, Any]] = []
    owners: dict[str, list[str]] = {}
    inline: dict[str, dict[str, Any]] = {}
    for cname, row in classes.items():
        is_upper = cname.casefold() == ROOT_CLASS or cname.casefold() in UPPER_CLASSES
        owner = "*" if cname.casefold() == ROOT_CLASS else (upper_of_name(cname) if is_upper else cname)
        for sname in list(row.get("slots") or []) + list((_named(row.get("slot_usage")) or {}).keys()):
            owners.setdefault(str(sname), [])
            if owner not in owners[str(sname)]:
                owners[str(sname)].append(owner)
        for aname, arow in _named(row.get("attributes")).items():
            inline[aname] = {**arow, "_owner": owner}
        if is_upper:
            # 内联的上层类什么时候算「本库直接拿它当类型用」:Carrel 导出的(带 carrel_upper 注解)看有没有 abstract;
            # 别处写的,不是 abstract 且没有别的类挂在它下面才算 —— 手写的模式常把上层类抄一份只为了让别的类能 is_a 它
            referenced = any(str(r.get("is_a") or "").casefold() == cname.casefold() for r in classes.values())
            ours = str(_ann(row, "carrel_upper") or "").lower() == "true"
            concrete = row.get("abstract") is not True and (ours or not referenced)
            if owner and owner != "*" and concrete:
                types.append({"name": owner, "is_a": owner, "description": row.get("description") or "",
                              "scope": str(_ann(row, "carrel_scope") or ""), "abstract": False})
            continue
        parent = row.get("is_a") or (row.get("mixins") or [None])[0] or ""
        if row.get("mixins"):
            warnings.append(f"类型 {cname} 的 mixins 不读,只按 is_a 归类")
        types.append({"name": cname, "is_a": str(parent or ""), "description": row.get("description") or "",
                      "scope": str(_ann(row, "carrel_scope") or ""), "abstract": bool(row.get("abstract"))})
    class_names = {c.casefold() for c in classes} | set(UPPER_CLASSES) | {ROOT_CLASS}

    def _is_class(r: Any) -> bool:
        return bool(r) and str(r).casefold() in class_names

    def endpoint_annotation(value):
        if isinstance(value, list):
            return [str(x) for x in value]
        text = str(value).strip()
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except ValueError as exc:
                raise SchemaError(["Invalid endpoint annotation JSON"]) from exc
            if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
                raise SchemaError(["Endpoint annotation must be a list of class names"])
            return parsed
        return [text] if _is_class(text) or text == "*" else text.split()

    predicates: list[dict[str, Any]] = []
    for sname, row in {**slots, **inline}.items():
        ranges = [str(x.get("range")) for x in (row.get("any_of") or row.get("exactly_one_of") or []) if isinstance(x, dict) and x.get("range")]
        if not ranges and row.get("range"):
            ranges = [str(row["range"])]
        ann_targets = _ann(row, "carrel_target_parents")
        if ann_targets:
            targets = endpoint_annotation(ann_targets)
        elif ranges and all(_is_class(r) for r in ranges):
            targets = ["*" if r.casefold() == ROOT_CLASS else r for r in ranges]
            if len(targets) > 1 and "*" in targets:
                targets = [t for t in targets if t != "*"]
        else:
            if ranges and any(str(r).casefold() in _PRIMITIVE_RANGES for r in ranges):
                warnings.append(f"slot {sname} 的取值是基本类型({ranges[0]}),是属性不是关系,已跳过")
            elif not ranges and not row.get("_owner"):
                warnings.append(f"slot {sname} 没有 range,已跳过")
            else:
                warnings.append(f"slot {sname} 的 range 不是模式里的类,已跳过")
            continue
        ann_sources = _ann(row, "carrel_source_parents")
        if ann_sources:
            sources = endpoint_annotation(ann_sources)
        elif row.get("_owner"):
            sources = [row["_owner"]]
        elif owners.get(sname):
            sources = owners[sname]
        elif row.get("domain"):
            sources = [str(row["domain"])]
        else:
            sources = ["*"]
        unknown = [x for x in sources + targets if x != "*" and not _is_class(x)]
        if unknown:
            raise SchemaError([f"slot {sname}: unknown endpoint classes: {unknown}"])
        multivalued = row.get("multivalued")
        predicates.append({"name": sname, "description": row.get("description") or "",
                           "source_parents": sources, "target_parents": targets,
                           "multivalued": multivalued is not False,
                           "core": str(_ann(row, "carrel_core") or "").lower() == "true"})
    profile_raw = _ann(doc, "carrel_profile")
    if isinstance(profile_raw, str):
        try:
            profile_raw = json.loads(profile_raw)
        except ValueError:
            warnings.append("carrel_profile 不是合法的 JSON,已忽略")
            profile_raw = {}
    model, more = normalize_model({"language": _ann(doc, "carrel_language") or "", "types": types,
                                   "predicates": predicates, "profile": profile_raw or {}})
    return model, warnings + more


def model_summary(model: dict[str, Any]) -> dict[str, Any]:
    """控制台与目录用的概览:每个类型的上层类、作用域、父类,关系的两端。"""
    types = list(model.get("types") or [])
    by_name = {t["name"]: t for t in types}
    return {
        "language": model.get("language") or "",
        "types": [{"name": t["name"], "is_a": t["is_a"], "upper": upper_of(t["name"], by_name),
                   "scope": t.get("scope") or "global", "abstract": bool(t.get("abstract")),
                   "description": t.get("description") or ""} for t in types],
        "predicates": [{"name": p["name"], "description": p.get("description") or "",
                        "source_parents": list(p.get("source_parents") or ["*"]),
                        "target_parents": list(p.get("target_parents") or ["*"]),
                        "multivalued": bool(p.get("multivalued", True)), "core": bool(p.get("core"))}
                       for p in model.get("predicates") or []],
        "profile": dict(model.get("profile") or {}),
    }


def descendants(names: Iterable[str], tree: dict[str, str]) -> set[str]:
    """按类型或上层类查询时带上子类:返回 names 本身及所有经 is_a 挂在它们下面的类型(大小写不敏感)。"""
    wanted = {str(n).casefold() for n in names if str(n).strip()}
    if not wanted:
        return set()
    out = set(wanted)
    parent = {str(k).casefold(): str(v).casefold() for k, v in (tree or {}).items()}
    for t in parent:
        cur, hops = t, 0
        while cur in parent and hops <= len(parent):
            cur = parent[cur]
            hops += 1
            if cur in wanted:
                out.add(t)
                break
    return out
