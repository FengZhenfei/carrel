"""表格的规则建图路线(2026-10-24 破坏性改版,方案第 4、5 条):xlsx / xls / csv 里「每行是一个东西」的表,
每行建一个实体,编号列写进外部编号,引用别的东西的列建成关系,其余列进事实层。产出与模型抽取同一种记录,下游共用。

每张表(同一文件、同一工作表、同一表头)先问一次模型:每行是不是同一类东西(类型从本库模式里选)、哪一列是编号、
哪一列是名字、哪些列引用别的东西(连到什么类型、用哪个谓词)。只这一次调用,后面逐行按规则算;编号列的唯一性
按数据复核,不唯一就不当编号。判成「不是实体表」的(功能 × 产品的对比矩阵、报表版式、汇总),这张表的单元
照旧交给模型逐单元抽取 —— 宽表读错列的问题恰恰出在模型逐格读,规则路线只接得住确实是清单的表。

行从切片正文里读(SHEET / ROWS / HEADER 前缀 + 按位置排的格子,或「列名: 值」展开的行),与入库的内容一致。
一个单元里映射不上表头的行太多(子表、说明行),整个单元退回模型抽取,不丢内容。
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable

from . import prompts
from .extract import ExtractionResult, ExtractionSchema, entity_name_ok, normalize_name, normalize_predicate, predicate_menu, type_menu
from .units import Unit

VERSION = "tables-v4"
TABLE_SUFFIXES = (".xlsx", ".xls", ".csv")
PROFILE_SAMPLE_ROWS = 8
MIN_DATA_ROWS = 3
KEY_UNIQUE_RATIO = 0.9
UNMAPPED_FALLBACK_RATIO = 0.4      # 单元里映射不上表头的行超过这个比例:整个单元退回模型抽取
NAME_MAX_CHARS = 60
FACT_VALUE_MAX_CHARS = 300
_REF_SPLIT_RE = re.compile(r"\s*(?:[,，、;；/]|\band\b|\s和\s)\s*")
_PREFIX_RE = re.compile(r"^(SHEET|ROWS|HEADER|Sheet|Rows|Columns|Title):\s?(.*)$")


def is_table_file(rel_path: str) -> bool:
    return str(rel_path or "").lower().endswith(TABLE_SUFFIXES)


def _cells(line: str) -> list[str]:
    s = line.strip()
    s = (" " + s if s.startswith("|") else s) + (" " if s.endswith("|") else "")
    return [c.strip() for c in s.split(" | ")]


def _labelled(cells: list[str]) -> dict[str, str] | None:
    """「列名: 值 | 列名: 值」展开的一行 → {列名: 值};不是这种写法返回 None。"""
    out: dict[str, str] = {}
    for c in cells:
        name, sep, value = c.partition(": ")
        if not sep or not name.strip():
            return None
        out[name.strip()] = (out.get(name.strip(), "") + " " + value.strip()).strip() if name.strip() in out else value.strip()
    return out


def parse_table_text(text: str) -> list[dict[str, Any]]:
    """单元正文 → 按出现顺序的表格段 [{sheet, header, rows: [{cells} 或 {labelled}]}]。
    一个单元可能由几个切片拼成,每个切片各带 SHEET / ROWS / HEADER 前缀;摘要片(Sheet / Rows / Columns)跳过。"""
    sections: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    summary = False
    for raw in str(text or "").splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        m = _PREFIX_RE.match(line.strip())
        if m:
            tag, value = m.group(1), m.group(2)
            if tag in ("Sheet", "Rows", "Columns", "Title"):
                summary = True               # 摘要片:不是数据行
                continue
            summary = False
            if tag == "SHEET":
                cur = {"sheet": value.strip(), "header": [], "rows": []}
                sections.append(cur)
            elif tag == "ROWS":
                if cur is None or cur["rows"]:
                    cur = {"sheet": (cur or {}).get("sheet", ""), "header": [], "rows": []}
                    sections.append(cur)
            elif tag == "HEADER":
                if cur is None:
                    cur = {"sheet": "", "header": [], "rows": []}
                    sections.append(cur)
                cur["header"] = _cells(value)
            continue
        if summary or cur is None:
            continue
        cells = _cells(line)
        lab = _labelled(cells) if not cur["header"] else None
        if lab is not None:
            # 同一行被拆成几行展开时,后面几行开头重复行标识(第一个格子):并回同一行
            rows = cur["rows"]
            first = next(iter(lab.items()))
            if rows and "labelled" in rows[-1] and rows[-1]["labelled"].get(first[0]) == first[1]:
                rows[-1]["labelled"].update({k: v for k, v in lab.items() if k not in rows[-1]["labelled"]})
            else:
                rows.append({"labelled": lab})
        else:
            cur["rows"].append({"cells": cells})
    return [s for s in sections if s["rows"]]


def _row_values(section: dict[str, Any], row: dict[str, Any]) -> dict[str, str] | None:
    """一行 → {列名: 值};按位置排的行格子数要和表头对得上,对不上返回 None(子表的表头、说明行)。"""
    if "labelled" in row:
        return dict(row["labelled"])
    header = section.get("header") or []
    cells = list(row.get("cells") or [])
    if not header or len(cells) > len(header) or len(cells) < 2:
        return None
    cells += [""] * (len(header) - len(cells))      # 解析器去掉了行尾的空格子(native_table._joined_cells)
    return {h: c for h, c in zip(header, cells) if h}


def table_key(rel_path: str, section: dict[str, Any]) -> str:
    """一张表的身份:文件 + 工作表 + 表头(展开的行用出现过的列名)。"""
    header = section.get("header") or sorted({k for r in section["rows"] if "labelled" in r for k in r["labelled"]})
    raw = json.dumps([rel_path, section.get("sheet") or "", header], ensure_ascii=False)
    return "t" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _unique_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out = []
    for r in rows:
        mark = json.dumps(r, ensure_ascii=False, sort_keys=True)
        if mark not in seen:
            seen.add(mark)
            out.append(r)
    return out


def _columns(section: dict[str, Any]) -> list[str]:
    if section.get("header"):
        return [h for h in section["header"] if h]
    cols: list[str] = []
    for r in section["rows"]:
        for k in (r.get("labelled") or {}):
            if k not in cols:
                cols.append(k)
    return cols


def render_profile_prompt(rel_path: str, section: dict[str, Any], schema: ExtractionSchema) -> str:
    cols = _columns(section)
    samples: list[str] = []
    for r in section["rows"]:
        values = _row_values(section, r)
        if values:
            samples.append(" | ".join(f"{k}: {v}" for k, v in values.items() if v)[:600])
        if len(samples) >= PROFILE_SAMPLE_ROWS:
            break
    return prompts.TABLE_PROFILE_PROMPT.format(
        document=rel_path, sheet=section.get("sheet") or "(none)", entity_types=type_menu(schema),
        predicates=predicate_menu(schema), columns=" | ".join(cols), rows="\n".join(samples) or "(none)")


def running_numbers(ids: list[str]) -> bool:
    """编号列其实是行号:全是不大的整数,从 1 附近数上来(1、2、3…;中间删过几行、开头几行落在别的段里也算)。
    这种列不是外部编号,换一张表同样的数指的是别的行(10-24 真机:需求清单的「序号」被当成编号,3 号、4 号成了
    需求的外部编号和事实)。"""
    if not ids or not all(v.isdigit() for v in ids):
        return False
    nums = [int(v) for v in ids]
    lo, hi = min(nums), max(nums)
    return lo <= max(2, hi // 10) and hi <= 2 * len(set(nums)) + 10


def normalize_profile(raw: dict[str, Any], section: dict[str, Any], schema: ExtractionSchema, *,
                      sheet_values: dict[str, list[str]] | None = None) -> dict[str, Any]:
    """模型给的表画像 → 校验过的画像;不是实体表或者校验不过返回 {"row_type": ""}。
    列名必须在表里;类型必须是本库的类型;谓词不在本库就用 related_to;编号列按数据复核唯一性。
    sheet_values 是同一个工作表里各列的全部取值(带表头的段和「列名: 值」的散行段都算):编号列是不是行号按它判。"""
    cols = _columns(section)
    lookup = {c.casefold(): c for c in cols}
    types = schema.type_lookup()
    preds = schema.predicate_lookup()

    def col(value: Any) -> str:
        return lookup.get(str(value or "").strip().casefold(), "")

    row_type = types.get(str(raw.get("row_type") or "").strip().casefold(), "")
    if not row_type:
        return {"row_type": ""}
    key, name = col(raw.get("key_column")), col(raw.get("name_column"))
    if not (key or name):
        return {"row_type": ""}
    values = [v for v in (_row_values(section, r) for r in section["rows"]) if v]
    skip: list[str] = []
    if key:
        ids = [str(v.get(key) or "").strip() for v in values if str(v.get(key) or "").strip()]
        whole = [x for x in ((sheet_values or {}).get(key) or ()) if x] or ids
        if ids and running_numbers(whole):
            skip, key = [key], ""            # 行号:不当外部编号,也不当一条事实;也不能当名字
            if name in skip:
                name = ""
        elif not ids or len(set(ids)) / len(ids) < KEY_UNIQUE_RATIO:
            key = ""                         # 编号列不唯一:不当外部编号(两行同号的表多半是分组列)
    refs: list[dict[str, str]] = []
    for item in raw.get("references") or []:
        if not isinstance(item, dict):
            continue
        c = col(item.get("column"))
        target = types.get(str(item.get("target_type") or "").strip().casefold(), "")
        if not c or not target or c in (key, name):
            continue
        pred = normalize_predicate(str(item.get("predicate") or ""))
        refs.append({"column": c, "target_type": target, "predicate": normalize_predicate(preds[pred]) if pred in preds else prompts.DEFAULT_PREDICATE})
    date_col = col(raw.get("date_column"))
    if not (name or key):
        return {"row_type": ""}
    out = {"row_type": row_type, "key_column": key, "name_column": name or key, "references": refs, "date_column": date_col}
    if skip:
        out["skip_columns"] = skip
    return out


class TableExtractor:
    """表格单元的规则抽取。profile() 先给每张表问一次模型(结果走 LLM 缓存,同一张表再建图秒回),
    之后 wants / extract / facts 都是纯规则。"""

    def __init__(self, units: Iterable[Unit], *, schema: ExtractionSchema) -> None:
        self.schema = schema
        self.units = [u for u in units if is_table_file(u.rel_path)]
        self.profiles: dict[str, dict[str, Any]] = {}
        self.unit_tables: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        self.stats: dict[str, Any] = {"tables": 0, "entity_tables": 0, "profile_calls": 0, "fallback_units": 0}
        for u in self.units:
            self.unit_tables[u.unit_id] = [(table_key(u.rel_path, s), s) for s in parse_table_text(u.text)]

    def profile(self, client: Any) -> dict[str, Any]:
        """每张表问一次模型。长表按行切成了好几个单元时,各段的行合回整张表(重叠切块重复的行只算一次;表头不同的
        段分开合,取行最多的那组):够不够算清单、唯一不唯一按整张表判,给模型看的样例也从整张表里取;编号是不是行号
        按整个工作表里这一列的取值判 —— 切块器有时把长行写成「列名: 值」的散行,那些段算成别的表,行号也散在里面
        (10-24 真机:一百多行的需求清单切成一段两三行,带表头的段合起来序号从 3 起,1、2、5 号在散行段里)。"""
        groups: dict[str, dict[tuple[str, ...], tuple[str, dict[str, Any]]]] = {}
        sheets: dict[tuple[str, str], dict[str, list[str]]] = {}     # (文件, 工作表) → 列 → 全部取值
        for u in self.units:
            for tkey, section in self.unit_tables.get(u.unit_id, []):
                columns = sheets.setdefault((u.rel_path, str(section.get("sheet") or "")), {})
                for r in section["rows"]:
                    for k, v in (_row_values(section, r) or {}).items():
                        if str(v or "").strip():
                            columns.setdefault(k, []).append(str(v).strip())
                by_header = groups.setdefault(tkey, {})
                header = tuple(section.get("header") or [])
                if header not in by_header:
                    by_header[header] = (u.rel_path, {**section, "rows": []})
                by_header[header][1]["rows"].extend(section["rows"])
        by_table: dict[str, tuple[str, dict[str, Any]]] = {}
        for tkey, by_header in groups.items():
            for _, whole in by_header.values():
                whole["rows"] = _unique_rows(whole["rows"])
            by_table[tkey] = max(by_header.values(), key=lambda p: len(p[1]["rows"]))
        self.stats["tables"] = len(by_table)
        for tkey, (rel, section) in sorted(by_table.items()):
            if len(section["rows"]) < MIN_DATA_ROWS:        # 整张表都不到 3 行:不是清单,交给模型
                self.profiles[tkey] = {"row_type": ""}
                continue
            try:
                from .schema import parse_json_object

                raw = client.chat(render_profile_prompt(rel, section, self.schema), max_tokens=1024,
                                  validate=lambda t: "row_type" in parse_json_object(t))
                self.stats["profile_calls"] += 1
                prof = normalize_profile(parse_json_object(raw), section, self.schema,
                                         sheet_values=sheets.get((rel, str(section.get("sheet") or ""))))
            except Exception as exc:             # 问不出来:这张表照旧交给模型逐单元抽
                print(f"[graph] table profile failed for {rel} / {section.get('sheet')}: {exc!r}", flush=True)
                prof = {"row_type": ""}
            prof["rel_path"], prof["sheet"] = rel, section.get("sheet") or ""
            self.profiles[tkey] = prof
            if prof.get("row_type"):
                self.stats["entity_tables"] += 1
        return dict(self.profiles)

    def load_profiles(self, profiles: dict[str, dict[str, Any]]) -> None:
        self.profiles = dict(profiles or {})
        self._ambiguous = None

    def ambiguous_names(self) -> set[str]:
        """全库表格里对着两个以上不同编号的名字(两个叫张三的员工):这些行按编号各成一个节点,
        正文里只写名字的提及不并进任何一个。"""
        if getattr(self, "_ambiguous", None) is None:
            ids: dict[str, set[str]] = {}
            for u in self.units:
                for prof, values in self._rows(u)[0]:
                    name, key = self._name(prof, values)
                    if name and key and name != key:
                        ids.setdefault(name.casefold(), set()).add(key)
            self._ambiguous = {n for n, keys in ids.items() if len(keys) > 1}
        return self._ambiguous

    def _rows(self, unit: Unit) -> tuple[list[tuple[dict[str, Any], dict[str, str]]], int, int]:
        """这个单元里能映射的行 (画像, {列: 值}),以及表格行总数、映射不上的行数。"""
        out: list[tuple[dict[str, Any], dict[str, str]]] = []
        total = unmapped = 0
        for tkey, section in self.unit_tables.get(unit.unit_id, []):
            prof = self.profiles.get(tkey) or {}
            header = [h.casefold() for h in (section.get("header") or [])]
            subtable = False
            for r in section["rows"]:
                total += 1
                cells = r.get("cells") or []
                if header and cells and cells[0].casefold() == header[0] and len(cells) >= 2:
                    subtable = True          # 又一行表头(同一个工作表里的下一张子表):后面的行不是这张表的
                values = None if subtable else _row_values(section, r)
                if not prof.get("row_type") or values is None:
                    unmapped += 1
                    continue
                name_col, key_col = prof.get("name_column") or "", prof.get("key_column") or ""
                if not str(values.get(name_col) or values.get(key_col) or "").strip():
                    unmapped += 1
                    continue
                out.append((prof, values))
        return out, total, unmapped

    def wants(self, unit: Unit) -> bool:
        if unit.unit_id not in self.unit_tables:
            return False
        rows, total, unmapped = self._rows(unit)
        if not rows or not total:
            return False
        if unmapped / total > UNMAPPED_FALLBACK_RATIO:
            return False
        return True

    @staticmethod
    def _name(prof: dict[str, Any], values: dict[str, str]) -> tuple[str, str]:
        """(实体名, 外部编号)。名字太长或不像名字的(带分号、等号)退回用编号。"""
        key = str(values.get(prof.get("key_column") or "") or "").strip()
        name = normalize_name(str(values.get(prof.get("name_column") or "") or "").strip())
        if not name or len(name) > NAME_MAX_CHARS or not entity_name_ok(name):
            name = key if key else name[:NAME_MAX_CHARS].rstrip()
        return name, key

    def extract(self, unit: Unit) -> ExtractionResult:
        rows, total, unmapped = self._rows(unit)
        entities: list[dict[str, Any]] = []
        relations: list[dict[str, Any]] = []
        ambiguous = self.ambiguous_names()
        seen: set[tuple[str, str]] = set()
        for prof, values in rows:
            name, key = self._name(prof, values)
            if not name or (name, key) in seen:          # 切片有重叠:同一行在相邻两片里各出现一次
                continue
            seen.add((name, key))
            used = ({prof.get("key_column"), prof.get("name_column")} | {r["column"] for r in prof.get("references") or []}
                    | set(prof.get("skip_columns") or []))
            attrs = [f"{k}: {v}" for k, v in values.items() if v and k not in used]
            desc = "; ".join(attrs)[:800]
            rec: dict[str, Any] = {"name": name, "type": prof["row_type"], "descriptions": [desc] if desc else [], "mentions": 1,
                                   "types": {prof["row_type"]: 1}, "kind": "instance", "source_kind": "table"}
            if key:
                rec["record_ref"] = f"{prof.get('key_column') or ''}:{key}"
                rec["external_ids"] = [{"scheme": prof.get("key_column") or "", "value": key}]
                if name != key:
                    rec["aliases"] = [key]
                if name.casefold() in ambiguous:
                    rec["identity"] = f"{prof.get('key_column') or ''}:{key}"
            entities.append(rec)
            for ref in prof.get("references") or []:
                cell = str(values.get(ref["column"]) or "").strip()
                for target in [normalize_name(t) for t in _REF_SPLIT_RE.split(cell) if t.strip()][:12]:
                    if not target or target == name or len(target) > NAME_MAX_CHARS or not entity_name_ok(target):
                        continue
                    entities.append({"name": target, "type": ref["target_type"], "descriptions": [], "mentions": 1,
                                     "types": {ref["target_type"]: 1}, "kind": "instance", "source_kind": "table_reference"})
                    relations.append({"source": name, "source_ref": rec.get("record_ref", ""), "target": target, "predicate": ref["predicate"], "predicate_raw": ref["predicate"],
                                      "descriptions": [f"{name} — {ref['column']}: {target}"], "strength": 5.0})
        return ExtractionResult(unit_id=unit.unit_id, entities=entities, relations=relations, calls=0,
                                stats={"builder": "table", "rows": len(rows), "table_rows": total, "unmapped_rows": unmapped,
                                       "unit_kind": "body"})

    def facts(self, unit: Unit) -> list[dict[str, Any]]:
        """属性列 → 事实(主体是这一行的实体,属性是列名);日期列给了就是这一行事实的起点。"""
        from .facts import fact_id, normalize_fact
        from .temporal import parse_date

        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        rows, _, _ = self._rows(unit)
        for prof, values in rows:
            name, key = self._name(prof, values)
            if not name:
                continue
            used = {prof.get("name_column")} | {r["column"] for r in prof.get("references") or []} | set(prof.get("skip_columns") or [])
            when = parse_date(values.get(prof.get("date_column") or "") or "") if prof.get("date_column") else None
            for column, value in values.items():
                value = str(value or "").strip()
                if not value or column in used or len(value) > FACT_VALUE_MAX_CHARS:
                    continue
                raw = {"subject": name, "property": column, "value": value}
                if when and column != prof.get("date_column"):
                    raw["valid_from"] = when
                fact = normalize_fact(raw)
                if fact is None:
                    continue
                if key:
                    fact["subject_ref"] = f"{prof.get('key_column') or ''}:{key}"
                fid = fact_id(unit.unit_id + (":" + fact["subject_ref"] if key else ""), fact)
                if fid in seen:
                    continue
                seen.add(fid)
                fact.update({"id": fid, "unit_id": unit.unit_id, "source_kind": "table", "exact": True})
                out.append(fact)
        return out
