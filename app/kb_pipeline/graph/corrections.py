"""更正文件(覆盖层,方案第 8、9 条):每个库一份声明式的更正,人或插件维护,建图与检索时套用。

位置:<base>/config/corrections/<kb_id>.yaml(KB_CORRECTIONS_DIR 可改),只在这台机器上,不在资料镜像里、不随资料同步。
2026-10-25 以前放在库文件夹下的 .carrel/corrections.yaml:镜像是单向 rsync --delete 的目标,机器上写不住,已不再读。
生命周期跟着库走,不跟着图谱走:开启新库时写一份带注释的空模板,彻底删除库时删掉;关闭库、删图谱、重建图谱都不动它,
重建之后照样套用。每晚的过期库清理顺带对账:登记着的库缺文件就补模板,登记表里已经没有的库的文件挪进 orphaned/。
文件头 kb.id 写着它属于哪个库,和文件名对不上就整份不套用;整库初始化以后库编号会从头再排,所以新编号第一次开库时,
同名的旧文件先挪进 orphaned/ 再写模板,旧库的更正不会套到新库上。
每次建图都在抽取结果之上重新套一遍,回退到旧版本时用的也是那一版建图时的更正。Carrel 只写模板、只删整份文件,内容从不改。

    kb:
      id: kb_004                                       # 这份更正属于哪个库
    entities:
      merge:     [[Acme Inc, Acme, ACME 公司]]        # 强制合并:第一个名字是规范名
      separate:  [[苹果公司, 苹果]]                    # 禁止合并:这几个名字永远是不同的实体
      aliases:   {Acme Inc: [艾克米]}                  # 补充别名
      types:     {张三: person}                        # 指定类型(必须是本库模式里的类型)
      reject:    [某个噪声实体]                        # 整个删掉,连同它的关系与事实
    relations:
      reject:    [{source: A, predicate: owned_by, target: B}]
    facts:
      reject:    [{subject: X, property: 价格}, {subject: X, property: 价格, value: "100"}]
    documents:                                         # 文档状态:检索默认只给现行文档
      - {path: 2024/价格表.xlsx, status: superseded, superseded_by: 2025/价格表.xlsx}
      - {path: 旧资料/**, status: archived}
      - {path: 合同/框架协议.pdf, valid_from: 2024-01-01, valid_until: 2025-12-31}

名字按规范名比较(全半角、空白、大小写不敏感),标题和别名都认。结构错误记进 errors;
建图在前置校验时停止,现行检索暂不返回该库的候选并报告降级,避免把错误规则当成无限制。
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

from .extract import entity_key, normalize_name

DOC_STATUSES = ("current", "superseded", "archived")
MAX_BYTES = 1024 * 1024
ORPHANED_DIR = "orphaned"

TEMPLATE = """\
# 更正文件:{kb_id}({folder})。在这里对自动建图的个别错误做定点修正,建图和检索时套用;用不到的节留空。
# 开库时自动生成,彻底删除库时自动删除;关闭库、删图谱、重建图谱都不动它,重建后照样套用。
# 名字按规范名比较(全半角、空白、大小写不敏感),标题和别名都认;格式错误会阻止建图并暂缓该库的现行召回。
kb:
  id: {kb_id}            # 这份更正属于哪个库;和文件名对不上时整份不套用,不要改
  folder: {folder_yaml}  # 开库时的文件夹名,只给人看
entities:
  merge: []              # 强制合并,第一个是规范名:[[某公司, 某公司有限公司]]
  separate: []           # 禁止合并:[[苹果公司, 苹果]]
  aliases: {{}}            # 补别名:{{某公司: [简称]}}
  types: {{}}              # 指定类型,必须是本库类型表里的:{{张三: person}}
  reject: []             # 整个删掉,连同它的关系与事实:[某个噪声实体]
relations:
  reject: []             # 驳回关系:[{{source: 某项目, predicate: recommends, target: 某方案}}]
facts:
  reject: []             # 驳回事实:[{{subject: 某产品, property: 价格}}];带上 value 只驳回那一个取值
documents: []            # 文档状态:[{{path: 2024/价格表.xlsx, status: superseded, superseded_by: 2025/价格表.xlsx}}]
                         # status 还可以是 archived;也可以只写有效期 valid_from / valid_until;path 可用通配(旧资料/**)
"""


def corrections_path(settings: Any, kb_id: str) -> Path | None:
    """这个库的更正文件;设置里没有更正目录(测试替身)时是 None,等于没有更正。"""
    base = getattr(settings, "corrections_dir", None)
    if not kb_id or not isinstance(base, (str, os.PathLike)):
        return None
    return Path(base) / f"{kb_id}.yaml"


def _set_aside(path: Path) -> str:
    """挪进同目录的 orphaned/(加时间戳,不覆盖已有的);返回新路径。"""
    aside = path.parent / ORPHANED_DIR
    aside.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target, n = aside / f"{path.stem}-{stamp}{path.suffix}", 1
    while target.exists():
        target, n = aside / f"{path.stem}-{stamp}-{n}{path.suffix}", n + 1
    os.replace(path, target)
    return str(target)


def ensure_corrections_file(settings: Any, kb_id: str, *, folder: str, fresh: bool = False) -> dict[str, Any]:
    """保证这个库有更正文件(开库时、每晚对账时)。fresh=True 是新编号第一次开库:同名的旧文件只可能属于整库初始化
    之前的某个库,挪进 orphaned/ 再写模板;其余情况已有的文件原样保留,没有才写模板。返回 {"path", "created", "orphaned"}。"""
    path = corrections_path(settings, kb_id)
    if path is None:
        return {}
    out: dict[str, Any] = {"path": str(path), "created": False, "orphaned": None}
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not fresh:
            return out
        out["orphaned"] = _set_aside(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(TEMPLATE.format(kb_id=kb_id, folder=folder, folder_yaml=json.dumps(str(folder), ensure_ascii=False)),
                   encoding="utf-8")
    os.replace(tmp, path)
    out["created"] = True
    return out


def remove_corrections_file(settings: Any, kb_id: str) -> str | None:
    """彻底删除库以后删掉它的更正文件(每晚备份里还留着最近几份);返回删掉的路径。"""
    path = corrections_path(settings, kb_id)
    if path is None or not path.is_file():
        return None
    path.unlink()
    return str(path)


def reconcile_corrections_files(settings: Any, registered: dict[str, str], *, keep: Iterable[str] = ()) -> dict[str, Any]:
    """每晚对账。registered:{kb_id: 文件夹名},登记着、不在彻底删除中的库,缺文件就补模板;
    keep:删到一半的库,文件留着等删完一起删;此外目录里登记表没有的库的文件挪进 orphaned/。"""
    base = getattr(settings, "corrections_dir", None)
    if not isinstance(base, (str, os.PathLike)):
        return {}
    created = [kb for kb, folder in sorted(registered.items())
               if ensure_corrections_file(settings, kb, folder=folder).get("created")]
    known = set(registered) | set(keep)
    orphaned = [_set_aside(p) for p in sorted(Path(base).glob("*.yaml")) if p.stem not in known]
    return {"created": created, "orphaned": orphaned}


def corrections_sha(path: Path | None) -> str:
    """更正的指纹,按解析后的有效条目算:没有文件、只有空模板、整份不套用的都是空串,改注释不算改动。
    增量并入据此判断「只改了更正也要重新合并」(开库补的空模板不该让已建好的图重新合并一遍)。"""
    return effective_sha(load_corrections(path))


def effective_sha(corr: dict[str, Any]) -> str:
    if is_empty(corr):
        return ""
    keys = ("merge", "separate", "aliases", "types", "reject_entities", "reject_relations", "reject_facts", "documents")
    body = json.dumps({k: corr.get(k) for k in keys}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def status_fingerprint(corr, paths, *, today):
    """Changes only when an input document actually enters/leaves current scope."""
    states = sorted((str(path), status_on(document_status(str(path), corr), today=today)) for path in set(paths))
    states = [(path, state) for path, state in states if state != "current"]
    return hashlib.sha256(json.dumps(states, ensure_ascii=False).encode()).hexdigest()[:16] if states else ""


def _names(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    return [normalize_name(str(v)) for v in value if normalize_name(str(v))]


def parse_corrections(doc: Any) -> dict[str, Any]:
    """YAML 解析出来的对象 → 规范化的更正。返回 {"merge", "separate", "aliases", "types", "reject_entities",
    "reject_relations", "reject_facts", "documents", "errors"}。"""
    out: dict[str, Any] = {"merge": [], "separate": [], "aliases": {}, "types": {}, "reject_entities": [],
                           "reject_relations": [], "reject_facts": [], "documents": [], "errors": []}
    if doc is None:
        return out
    if not isinstance(doc, dict):
        out["errors"].append("更正文件的顶层必须是映射(entities / relations / facts / documents)")
        return out
    # Validate containers before iterating. A legal YAML scalar is not a rule list.
    doc = dict(doc)
    for section in ("entities", "relations", "facts"):
        value = doc.get(section)
        if value is not None and not isinstance(value, dict):
            out["errors"].append(f"{section} must be a mapping")
            doc[section] = {}
        else:
            doc[section] = dict(value or {})
    for section, keys in (("entities", ("merge", "separate", "reject")),
                          ("relations", ("reject",)), ("facts", ("reject",))):
        for key in keys:
            value = doc[section].get(key)
            if value is not None and not isinstance(value, list):
                out["errors"].append(f"{section}.{key} must be a list")
                doc[section][key] = []
    for key in ("aliases", "types"):
        value = doc["entities"].get(key)
        if value is not None and not isinstance(value, dict):
            out["errors"].append(f"entities.{key} must be a mapping")
            doc["entities"][key] = {}
    if doc.get("documents") is not None and not isinstance(doc["documents"], list):
        out["errors"].append("documents must be a list")
        doc["documents"] = []
    # Reject nested containers where scalars are expected instead of stringifying them
    # into names, wildcards or dates. Keep locations for operator diagnostics.
    def scalar(value):
        return value is not None and not isinstance(value, (dict, list, tuple, set))
    for key in ("merge", "separate", "reject"):
        for i, group in enumerate(doc["entities"].get(key) or []):
            names = group.get("names") if isinstance(group, dict) and key == "merge" else group
            values = names if isinstance(names, list) else [names]
            if any(not scalar(v) for v in values) or (isinstance(group, dict) and not scalar(group.get("canonical", ""))):
                out["errors"].append(f"entities.{key}[{i}] must contain scalar names")
    for key in ("aliases", "types"):
        for name, value in (doc["entities"].get(key) or {}).items():
            values = value if key == "aliases" and isinstance(value, list) else [value]
            if not scalar(name) or any(not scalar(v) for v in values):
                out["errors"].append(f"entities.{key}.{name} must contain scalar values")
    for section, fields in (("relations", ("source", "target", "predicate")),
                            ("facts", ("subject", "property", "value")),
                            ("documents", ("path", "status", "superseded_by", "valid_from", "valid_until", "note"))):
        items = doc.get("documents") if section == "documents" else doc[section].get("reject")
        for i, item in enumerate(items or []):
            if isinstance(item, dict):
                for key in fields:
                    if item.get(key) is not None and not scalar(item[key]):
                        out["errors"].append(f"{section}[{i}].{key} must be a scalar")
    if out["errors"]:
        return out
    ents = doc.get("entities") or {}
    if not isinstance(ents, dict):
        out["errors"].append("entities 必须是映射")
        ents = {}
    for group in ents.get("merge") or []:
        names = _names(group.get("names") if isinstance(group, dict) else group)
        if isinstance(group, dict) and group.get("canonical"):
            names = _names(group["canonical"]) + [n for n in names if n != normalize_name(str(group["canonical"]))]
        if len(names) >= 2:
            out["merge"].append(names)
        else:
            out["errors"].append(f"entities.merge 的一组至少要两个名字:{group!r}"[:200])
    for group in ents.get("separate") or []:
        names = _names(group)
        if len(names) >= 2:
            out["separate"].append(names)
        else:
            out["errors"].append(f"entities.separate 的一组至少要两个名字:{group!r}"[:200])
    for name, aliases in (ents.get("aliases") or {}).items() if isinstance(ents.get("aliases"), dict) else []:
        if normalize_name(str(name)) and _names(aliases):
            out["aliases"][normalize_name(str(name))] = _names(aliases)
    for name, etype in (ents.get("types") or {}).items() if isinstance(ents.get("types"), dict) else []:
        if normalize_name(str(name)) and str(etype or "").strip():
            out["types"][normalize_name(str(name))] = str(etype).strip()
    out["reject_entities"] = _names(ents.get("reject") or [])
    rels = (doc.get("relations") or {}).get("reject") if isinstance(doc.get("relations"), dict) else None
    for item in rels or []:
        if isinstance(item, dict) and item.get("source") and item.get("target"):
            out["reject_relations"].append({"source": normalize_name(str(item["source"])), "target": normalize_name(str(item["target"])),
                                            "predicate": str(item.get("predicate") or "").strip()})
        else:
            out["errors"].append(f"relations.reject 每条要有 source 与 target:{item!r}"[:200])
    facts = (doc.get("facts") or {}).get("reject") if isinstance(doc.get("facts"), dict) else None
    for item in facts or []:
        if isinstance(item, dict) and item.get("subject"):
            out["reject_facts"].append({"subject": normalize_name(str(item["subject"])), "property": str(item.get("property") or "").strip(),
                                        "value": str(item["value"]).strip() if item.get("value") is not None else ""})
        else:
            out["errors"].append(f"facts.reject 每条至少要有 subject:{item!r}"[:200])
    for item in doc.get("documents") or []:
        if not isinstance(item, dict) or not str(item.get("path") or "").strip():
            out["errors"].append(f"documents 每条要有 path:{item!r}"[:200])
            continue
        status = str(item.get("status") or "current").strip().lower()
        if status not in DOC_STATUSES:
            out["errors"].append(f"文档状态只能是 current / superseded / archived:{item.get('path')}")
            continue
        from datetime import date
        dates = {}
        for field in ("valid_from", "valid_until"):
            value = str(item.get(field) or "").strip()
            try:
                if value:
                    date.fromisoformat(value + {4: "-01-01", 7: "-01", 10: ""}[len(value)])
                dates[field] = value
            except (ValueError, KeyError):
                out["errors"].append(f"documents.{item['path']}.{field} must be an ISO date")
        if len(dates) != 2:
            continue
        if dates["valid_from"] and dates["valid_until"] and dates["valid_from"][:len(dates["valid_until"])] > dates["valid_until"]:
            out["errors"].append(f"documents.{item['path']}: valid_from is after valid_until")
            continue
        out["documents"].append({
            "path": str(item["path"]).strip().lstrip("/"), "status": status,
            "superseded_by": str(item.get("superseded_by") or "").strip().lstrip("/"),
            **dates,
            "note": str(item.get("note") or "").strip()[:300],
        })
    return out


def load_corrections(path: Path | None) -> dict[str, Any]:
    """读更正文件;没有文件返回空的更正,YAML 坏了返回空的更正并带 errors(调用方拒绝发布/现行召回)。
    文件头 kb.id 和文件名(库编号)对不上的,整份不套用,只记一条 error。"""
    empty = parse_corrections(None)
    if path is None or not path.is_file():
        return empty
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return {**empty, "errors": [f"更正文件读不了:{exc}"]}
    if len(raw) > MAX_BYTES:
        return {**empty, "errors": [f"更正文件超过 {MAX_BYTES // 1024} KB,没有读"]}
    import yaml

    try:
        doc = yaml.safe_load(raw.decode("utf-8-sig"))
    except (yaml.YAMLError, UnicodeDecodeError) as exc:
        return {**empty, "errors": [f"更正文件不是合法的 YAML:{exc}"[:300]]}
    owner = doc.get("kb") if isinstance(doc, dict) else None
    owner_id = str(owner.get("id") or "").strip() if isinstance(owner, dict) else ""
    if owner_id and owner_id != path.stem:
        return {**empty, "errors": [f"更正文件头写的是 {owner_id},不是 {path.stem},整份没有套用"]}
    out = parse_corrections(doc)
    out["sha"] = hashlib.sha256(raw).hexdigest()[:16]
    return out


def is_empty(corr: dict[str, Any] | None) -> bool:
    return not corr or not any(corr.get(k) for k in ("merge", "separate", "aliases", "types", "reject_entities",
                                                     "reject_relations", "reject_facts", "documents"))


# ── 文档状态 ─────────────────────────────────────────────────────────────

def _path_matches(pattern: str, rel_path: str) -> bool:
    """path 可以是文件、目录(以 / 结尾或本身就是目录前缀)或通配(* / ** / ?)。"""
    pat = pattern.strip().lstrip("/")
    rel = str(rel_path or "").lstrip("/")
    if not pat:
        return False
    if any(ch in pat for ch in "*?["):
        return fnmatch.fnmatchcase(rel, pat) or fnmatch.fnmatchcase(rel, pat.replace("**/", ""))
    return rel == pat or rel.startswith(pat.rstrip("/") + "/")


def document_status(rel_path: str, corr: dict[str, Any] | None) -> dict[str, Any] | None:
    """一份文档的状态条目(更正文件里第一条匹配的);没写就 None(= 现行)。"""
    for item in (corr or {}).get("documents") or []:
        if _path_matches(item["path"], rel_path):
            return dict(item)
    return None


def status_on(item: dict[str, Any] | None, *, today: str) -> str:
    """文档在某一天的状态:写明 superseded / archived 的就是;有效期止于 today 之前的算 superseded;其余 current。"""
    if not item:
        return "current"
    if item.get("status") in ("superseded", "archived"):
        return str(item["status"])
    start = str(item.get("valid_from") or "")
    if start and start > today[:len(start)]:
        return "not_started"
    until = str(item.get("valid_until") or "")
    if until and until < today[: len(until)]:
        return "superseded"
    return "current"


def document_in_scope(item: dict[str, Any] | None, scope: dict[str, Any]) -> bool:
    """检索时这份文档按时间口径(currency.query_scope)给不给:现行口径按 status_on;回看某一天只看有效期
    (那天还没生效的、那天之前已经失效的不给;写明被取代 / 归档但没写日期的,那天它还算数);history 全给。"""
    if not item or scope.get("mode") == "history":
        return True
    day = str(scope.get("day") or "")
    if scope.get("mode") != "as_of":
        return status_on(item, today=day) == "current"
    start, until = str(item.get("valid_from") or ""), str(item.get("valid_until") or "")
    if start and start > day[: len(start)]:
        return False
    if until and until < day[: len(until)]:
        return False
    return True


# ── 建图时套用 ───────────────────────────────────────────────────────────

def _index(entities: list[dict[str, Any]]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for i, e in enumerate(entities):
        for n in [str(e.get("title") or "")] + [str(a) for a in (e.get("aliases") or [])]:
            k = entity_key(n)
            if k:
                out.setdefault(k, [])
                if i not in out[k]:
                    out[k].append(i)
    return out


def apply_entity_corrections(entities: list[dict[str, Any]], relations: list[dict[str, Any]], corr: dict[str, Any], *,
                             known_types: Iterable[str] = (), upper_of: Any = None, path_of: Any = None,
                             ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], set[frozenset[str]]]:
    """删实体 → 指定类型 → 补别名 → 强制合并;返回 (实体, 关系, 计数, 禁止合并的键对)。禁止合并交给归并阶段守住。
    upper_of / path_of:类型 → 上层类 / 类型链(改了类型要跟着改)。"""
    from .resolution import merge_entities

    stats = {"rejected_entities": 0, "retyped": 0, "aliases_added": 0, "forced_merges": 0, "unknown_names": [], "unknown_types": []}
    if is_empty(corr):
        return entities, relations, stats, set()
    rejected = {entity_key(n) for n in corr.get("reject_entities") or []}
    if rejected:
        drop = {e["key"] for e in entities if entity_key(str(e.get("title") or "")) in rejected}
        entities = [e for e in entities if e["key"] not in drop]
        relations = [r for r in relations if r["source_key"] not in drop and r["target_key"] not in drop]
        stats["rejected_entities"] = len(drop)
    types = {str(t).casefold(): str(t) for t in known_types}
    idx = _index(entities)
    for name, etype in (corr.get("types") or {}).items():
        canonical = types.get(str(etype).casefold())
        if not canonical:
            stats["unknown_types"].append(f"{name}: {etype}")
            continue
        hits = idx.get(entity_key(name)) or []
        if not hits:
            stats["unknown_names"].append(name)
        for i in hits:
            e = entities[i]
            e["type"] = canonical
            if upper_of is not None:
                e["upper"] = upper_of(canonical)
            if path_of is not None:
                e["type_path"] = path_of(canonical)
            e["type_corrected"] = True
            stats["retyped"] += 1
    for name, aliases in (corr.get("aliases") or {}).items():
        hits = idx.get(entity_key(name)) or []
        if not hits:
            stats["unknown_names"].append(name)
        for i in hits[:1]:
            e = entities[i]
            for a in aliases:
                if a != e.get("title") and a not in (e.get("aliases") or []):
                    e.setdefault("aliases", []).append(a)
                    stats["aliases_added"] += 1
    pairs: list[tuple[int, int]] = []
    heads: dict[int, str] = {}
    for group in corr.get("merge") or []:
        found = [(n, (idx.get(entity_key(n)) or [None])[0]) for n in group]
        present = [(n, i) for n, i in found if i is not None]
        stats["unknown_names"].extend(n for n, i in found if i is None)
        if len(present) < 2:
            continue
        first = present[0][1]
        heads[first] = present[0][0]
        for _, i in present[1:]:
            if i != first:
                pairs.append((first, i))
    if pairs:
        # 规范名:更正里写的第一个名字;合并后的代表不管是谁,标题改成它
        wanted = {entities[i]["key"]: title for i, title in heads.items()}
        entities, relations, mstats = merge_entities(entities, relations, pairs)
        member_of = {str(k): e for e in entities for k in (e.get("member_keys") or [e["key"]])}
        for key, title in wanted.items():
            e = member_of.get(key)
            if e is not None and e.get("title") != title:
                old = e.get("title")
                e["title"] = title
                e["aliases"] = [a for a in (e.get("aliases") or []) if a != title] + ([old] if old and old not in (e.get("aliases") or []) else [])
        stats["forced_merges"] = int(mstats.get("merged_away") or 0)
    forbidden: set[frozenset[str]] = set()
    idx = _index(entities)
    for group in corr.get("separate") or []:
        keys = [entities[i]["key"] for n in group for i in (idx.get(entity_key(n)) or [])[:1]]
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                if keys[a] != keys[b]:
                    forbidden.add(frozenset((keys[a], keys[b])))
    stats["unknown_names"] = sorted(set(stats["unknown_names"]))[:50]
    return entities, relations, stats, forbidden


def apply_relation_corrections(relations: list[dict[str, Any]], entities: list[dict[str, Any]], corr: dict[str, Any]
                               ) -> tuple[list[dict[str, Any]], int]:
    """驳回的关系(两端按名字、谓词不写就不限)删掉;返回 (关系, 删掉的条数)。"""
    rules = (corr or {}).get("reject_relations") or []
    if not rules:
        return relations, 0
    names: dict[str, set[str]] = {}
    for e in entities:
        names[e["key"]] = {entity_key(str(e.get("title") or ""))} | {entity_key(str(a)) for a in (e.get("aliases") or [])}
    out = []
    dropped = 0
    for r in relations:
        src, tgt = names.get(r["source_key"], set()), names.get(r["target_key"], set())
        hit = any(entity_key(x["source"]) in src and entity_key(x["target"]) in tgt
                  and (not x["predicate"] or x["predicate"] == r.get("predicate")) for x in rules)
        if hit:
            dropped += 1
            continue
        out.append(r)
    return out, dropped


def apply_fact_corrections(facts: list[dict[str, Any]], corr: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    """驳回的事实:主体按名字,属性不写就是这个主体的全部事实,值写了只删这个值。"""
    rules = (corr or {}).get("reject_facts") or []
    if not rules:
        return facts, 0
    out = []
    dropped = 0
    for f in facts:
        subj = entity_key(str(f.get("subject") or ""))
        prop = str(f.get("property") or "").casefold()
        value = str(f["value"]).strip() if f.get("value") is not None else ""
        hit = any(entity_key(x["subject"]) == subj and (not x["property"] or x["property"].casefold() == prop)
                  and (not x["value"] or x["value"] == value) for x in rules)
        if hit:
            dropped += 1
            continue
        out.append(f)
    return out, dropped


def mark_document_status(rows: Iterable[dict[str, Any]], corr: dict[str, Any], *, rel_of_doc: dict[str, str], today: str) -> int:
    """事实 / 关系:来源文档全都不是现行的(被取代、归档、过了有效期),记 doc_status;返回标记的条数。
    一条关系在现行文档里也说过,就还是现行的。"""
    if not (corr or {}).get("documents"):
        return 0
    marked = 0
    for row in rows:
        docs = [str(d) for d in (row.get("doc_ids") or ([row.get("doc_id")] if row.get("doc_id") else [])) if d]
        if not docs:
            continue
        states = [status_on(document_status(rel_of_doc.get(d, ""), corr), today=today) for d in docs]
        if states and all(s != "current" for s in states):
            row["doc_status"] = "archived" if all(s == "archived" for s in states) else "superseded"
            marked += 1
        else:
            row.pop("doc_status", None)
    return marked
