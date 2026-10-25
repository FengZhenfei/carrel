"""Live lower and upper bounds for the chunking configuration.

Both bounds follow the embedding service's --max-model-len, so they move by adjusting compose without
code changes:

- **max_tokens**: MIN_TOKENS to CAP_RATIO of max_model_len. The headroom left absorbs two kinds of
  over-budget risk: first, the TITLE/CAPTION markers that text_for_block prepends (they are spliced into
  the text before chunking and are counted in the budget, but they crowd out the body); second,
  vision/image/chart blocks take the "one image, one chunk, never split further" path, and the VLM
  description length is not bounded by max_tokens.
- **overlap_tokens**: min(max_model_len - max_tokens, max_tokens / 2). The second term is a semantic
  requirement (an overlap above half is just feeding the same content twice); the first tightens the
  overlap as max_tokens approaches the model limit, so the tail and the body cannot blow the budget
  together.

The overlap bound depends on the max_tokens currently entered, so it is not a static number -- the
frontend must recompute it live as the max_tokens input changes (see setOverlapHint in app.js).
"""

from __future__ import annotations

import re
from typing import Any, Iterable

import requests

MIN_TOKENS = 128
CAP_RATIO = 0.8
FALLBACK_MAX_MODEL_LEN = 4096


def chunk_limits(embedding_base_url: str) -> dict[str, Any]:
    max_model_len = None
    try:
        base = embedding_base_url.rstrip("/")
        data = requests.get(f"{base}/models", timeout=5).json()
        raw = (data.get("data") or [{}])[0].get("max_model_len")
        max_model_len = int(raw) if raw else None
    except Exception:
        pass
    effective = max_model_len or FALLBACK_MAX_MODEL_LEN
    cap = max(MIN_TOKENS, int(effective * CAP_RATIO))
    return {
        "embedding_max_model_len": max_model_len,
        # Falls back to FALLBACK when the service cannot be probed; both the frontend and overlap_cap
        # compute bounds from this value, so the two sides never end up with different caps because one
        # used None and the other the fallback.
        "effective_max_model_len": effective,
        "max_tokens_cap": cap,
        "max_tokens_min": MIN_TOKENS,
        "cap_ratio": CAP_RATIO,
        "overlap_rule": "overlap_tokens < min(max_model_len - max_tokens, max_tokens / 2)",
        "live": max_model_len is not None,
    }


# Extraction unit = several consecutive raw chunks merged into one passage fed to the LLM. 1 = no merging
# (one chunk per unit); too large and a passage holds too much, so exhaustive extraction misses things
# (one or two gleaning rounds cannot recover them) and calls take longer too. Default 3: on kb_003 that
# averages 650 tokens, comparable to what upstream's 1200/100 sliding window produces.
GRAPH_UNIT_CHUNKS_MIN = 1
GRAPH_UNIT_CHUNKS_MAX = 8
GRAPH_UNIT_CHUNKS_DEFAULT = 3
# Gleaning rounds: 0 = extract once only; 1 = always two calls (upstream default); 2 = ask "anything
# else?" before the second round.
GRAPH_MAX_GLEANINGS_MIN = 0
GRAPH_MAX_GLEANINGS_MAX = 2
GRAPH_MAX_GLEANINGS_DEFAULT = 1


def validate_graph_unit_config(unit_chunks: Any, max_gleanings: Any) -> list[str]:
    errors: list[str] = []
    if unit_chunks not in (None, ""):
        try:
            value = int(unit_chunks)
        except (TypeError, ValueError):
            errors.append("graph_unit_chunks must be an integer")
        else:
            if not (GRAPH_UNIT_CHUNKS_MIN <= value <= GRAPH_UNIT_CHUNKS_MAX):
                errors.append(
                    f"Chunks per extraction unit must be between {GRAPH_UNIT_CHUNKS_MIN} and {GRAPH_UNIT_CHUNKS_MAX} "
                    f"(default {GRAPH_UNIT_CHUNKS_DEFAULT}; 1 = no merging; large values make entity extraction miss more)"
                )
    if max_gleanings not in (None, ""):
        try:
            value = int(max_gleanings)
        except (TypeError, ValueError):
            errors.append("graph_max_gleanings must be an integer")
        else:
            if not (GRAPH_MAX_GLEANINGS_MIN <= value <= GRAPH_MAX_GLEANINGS_MAX):
                errors.append(f"Gleaning rounds must be between {GRAPH_MAX_GLEANINGS_MIN} and {GRAPH_MAX_GLEANINGS_MAX}")
    return errors


# 「立即/重新抽取标签」采样多少个片段喂给 LLM 归纳领域/语言/实体类型。
# 兼容旧配置字段的校验;生产采样已忽略此字段,统一按模型输入预算自动决定数量。
GRAPH_TUNE_SAMPLE_MIN = 0
GRAPH_TUNE_SAMPLE_MAX = 1_000_000
GRAPH_TUNE_SAMPLE_DEFAULT = 0

# Cap on the type table length. The value of a closed type table lies in being "narrow": with many types
# the model's attention is diluted when choosing, which is worse than giving no table at all (upstream's
# default has only 4 types).
# Cap on the historical versions of the entity labels. 3 are kept: enough to compare and roll back
# between "just extracted", "previous" and "the one before", without turning the dropdown into an
# archaeological dig. The active version is never pushed out.
GRAPH_SCHEMA_VERSION_MAX = 3

# Id of the "no version info" placeholder entry in the read-only view. It is never stored; selecting it
# means "keep things as they are".
CURRENT_SCHEMA_VERSION_ID = "__current__"



def push_schema_version(versions, entry, *, active_id=None, limit=GRAPH_SCHEMA_VERSION_MAX, protected_ids=()):
    """新版本入环,最新的排最前,超出上限就淘汰最旧的 —— 但**绝不淘汰正在生效的
    那一版**。淘汰掉生效版意味着下拉框里选不回当前建图用的标签,而图还是按它建的。
    """
    kept = [dict(v) for v in (versions or []) if str(v.get("id")) != str(entry.get("id"))]
    out = [dict(entry)] + kept
    if len(out) <= limit:
        return out
    protected = {str(active_id or ""), *map(str, protected_ids)}
    trimmed = list(out)
    # Evict from the oldest forward, skipping the active version
    for index in range(len(trimmed) - 1, 0, -1):
        if len(trimmed) <= limit:
            break
        if str(trimmed[index].get("id")) in protected:
            continue
        trimmed.pop(index)
    return trimmed  # Referenced versions may temporarily exceed the display retention limit.


GRAPH_ENTITY_TYPES_MAX = 32
GRAPH_ENTITY_TYPE_MAX_LEN = 64


def normalize_entity_types(value: Any) -> tuple[str, ...]:
    """Normalize the entity types sent by the console into an ordered, de-duplicated tuple.

    Accepts a list/tuple, and also a whole text separated by commas / semicolons / newlines -- pasting a
    block into the input box is common. De-duplication compares case-insensitively but keeps the spelling
    of the first occurrence: types are spliced verbatim into the prompt, and entity merging matches
    (title, type) exactly, so having both Pin and pin in the table would only split the same thing into
    two nodes in the graph.
    """
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        items = [str(v) for v in value]
    else:
        # Full-width comma / enumeration comma / semicolon are written as code points so the source is not
        # folded into half-width forms in transit
        items = re.split(r"[,;\uFF0C\u3001\uFF1B\n\r]+", str(value))
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        name = item.strip()
        if not name:
            continue
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(name)
    return tuple(out)


def validate_graph_tune_config(entity_types: Any, sample_size: Any) -> list[str]:
    errors: list[str] = []
    types = normalize_entity_types(entity_types)
    if len(types) > GRAPH_ENTITY_TYPES_MAX:
        errors.append(
            f"At most {GRAPH_ENTITY_TYPES_MAX} entity types (currently {len(types)}); "
            "a narrower type table extracts more precisely, and an empty one falls back to the global default"
        )
    for name in types:
        if len(name) > GRAPH_ENTITY_TYPE_MAX_LEN:
            errors.append(
                f"Entity type {name[:20]!r}… is too long (at most {GRAPH_ENTITY_TYPE_MAX_LEN} characters)")
            break
    if sample_size is None:
        # Key absent = unset = use the default; it is not invalid input. The effective view of config may
        # not have this key at all for old KBs / old tests, and that must not raise an error.
        return errors
    try:
        size = int(sample_size)
    except (TypeError, ValueError):
        errors.append("Sample size must be an integer")
        return errors
    if not (GRAPH_TUNE_SAMPLE_MIN <= size <= GRAPH_TUNE_SAMPLE_MAX):
        errors.append(
            f"Sample size must be between {GRAPH_TUNE_SAMPLE_MIN} and {GRAPH_TUNE_SAMPLE_MAX} "
            "(0 is automatic; the model input budget determines the actual sample)"
        )
    return errors


GRAPH_PREDICATES_MAX = 28
_PREDICATE_NAME_RE = re.compile(r"[^a-z0-9_]+")


def normalize_predicate_name(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return _PREDICATE_NAME_RE.sub("", text).strip("_")


def normalize_predicates(value: Any, *, limit: int | None = GRAPH_PREDICATES_MAX) -> tuple[dict[str, Any], ...]:
    """谓词表:[{name, description, source_parents, target_parents, multivalued, core}];名字 lower_snake、
    去重、去掉兜底的 related_to,最多留 limit 条(None = 不截,校验数量时用)。端点是上层类、类型名或 "*",
    原样保留写法(类型名大小写有意义)。字符串形式(逗号/换行分隔的名字)也接受。"""
    if value is None:
        return ()
    if isinstance(value, str):
        items: list[Any] = [n for n in re.split(r"[,;\uFF0C\u3001\uFF1B\n\r]+", value) if n.strip()]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        return ()
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        if isinstance(item, str):
            item = {"name": item}
        if not isinstance(item, dict):
            continue
        name = normalize_predicate_name(item.get("name"))
        if not name or name == "related_to" or name in seen:
            continue
        seen.add(name)
        multivalued = item.get("multivalued")
        out.append({
            "name": name,
            "description": str(item.get("description") or "").strip()[:200],
            "source_parents": [str(p).strip() for p in (item.get("source_parents") or []) if str(p).strip()][:12],
            "target_parents": [str(p).strip() for p in (item.get("target_parents") or []) if str(p).strip()][:12],
            # 没写就是多值:只有模式里显式声明单值的关系才做「新值取代旧值」(ontology 模块说明)
            "multivalued": multivalued is not False,
            "core": bool(item.get("core")),
        })
        if limit is not None and len(out) >= limit:
            break
    return tuple(out)


def normalize_parent_types(value: Any) -> dict[str, str]:
    """{类型: 上层类}:生效字段,由模式版本展开(ontology.expand_model),值只能是八个上层类之一。"""
    from .ontology import upper_of_name

    if not isinstance(value, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in value.items():
        key = str(k or "").strip()
        parent = upper_of_name(v)
        if key and parent:
            out[key] = parent
    return out


def normalize_type_parents(value: Any) -> dict[str, str]:
    """{类型: 直接父类}(类型树,父类是上层类或另一个类型):按祖先查询、端点约束按祖先判。"""
    if not isinstance(value, dict):
        return {}
    return {str(k).strip(): str(v).strip() for k, v in value.items() if str(k or "").strip() and str(v or "").strip()}


def normalize_type_scopes(value: Any) -> dict[str, str]:
    """{类型: global | document}:模式阶段逐类型判定的作用域。"""
    if not isinstance(value, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in value.items():
        scope = str(v or "").strip().lower()
        if str(k or "").strip() and scope in ("global", "document"):
            out[str(k).strip()] = scope
    return out


# 2026-09-07 schema layer completion: normalization of type definitions / corpus examples / scenario
# profile (derived data, like the type table and the predicate table)
GRAPH_TYPE_DEFINITION_MAX_CHARS = 240
GRAPH_EXAMPLES_MAX_CHARS = 16000
GRAPH_PROFILE_AXES = ("date", "version", "none", "auto")
GRAPH_PROFILE_LIST_MAX = 12


def normalize_type_definitions(value: Any, entity_types: Iterable[str] = ()) -> dict[str, str]:
    """{type: one-sentence definition}; when entity_types is given, type names are aligned case-insensitively
    and those outside the table are dropped."""
    if isinstance(value, str):
        try:
            import json as _json
            value = _json.loads(value)
        except ValueError:
            return {}
    if not isinstance(value, dict):
        return {}
    lookup = {str(t).casefold(): str(t) for t in entity_types}
    out: dict[str, str] = {}
    for k, v in value.items():
        key = str(k or "").strip()
        if lookup:
            key = lookup.get(key.casefold(), "")
        text = " ".join(str(v or "").split())[:GRAPH_TYPE_DEFINITION_MAX_CHARS]
        if key and text:
            out[key] = text
    return out


def normalize_examples(value: Any) -> str:
    text = str(value or "").strip()
    return text[:GRAPH_EXAMPLES_MAX_CHARS]


def _str_list(value: Any, limit: int = GRAPH_PROFILE_LIST_MAX) -> list[str]:
    if isinstance(value, str):
        value = re.split(r"[,\n;，、；]", value)
    if not isinstance(value, (list, tuple)):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for v in value:
        text = " ".join(str(v or "").split())
        if text and text.casefold() not in seen:
            seen.add(text.casefold())
            out.append(text[:80])
        if len(out) >= limit:
            break
    return out


def normalize_profile(value: Any) -> dict[str, Any]:
    """Scenario profile: subject_types / axis / conclusion_headings / boilerplate_headings / listing_headings /
    type_words / extension_predicates. An empty object means no profile."""
    if isinstance(value, str):
        try:
            import json as _json
            value = _json.loads(value)
        except ValueError:
            return {}
    if not isinstance(value, dict):
        return {}
    raw_axis = str(value.get("axis") or "").strip().lower()
    axis = raw_axis if raw_axis in GRAPH_PROFILE_AXES else "auto"
    out = {
        "subject_types": _str_list(value.get("subject_types")),
        "axis": axis,
        "conclusion_headings": _str_list(value.get("conclusion_headings")),
        "boilerplate_headings": _str_list(value.get("boilerplate_headings")),
        "listing_headings": _str_list(value.get("listing_headings")),
        "type_words": [w for w in _str_list(value.get("type_words")) if " " not in w.strip()][:10],
        "extension_predicates": [p for p in (normalize_predicate_names(value.get("extension_predicates")))],
    }
    if not any((out["subject_types"], out["conclusion_headings"], out["boilerplate_headings"], out["listing_headings"],
                out["type_words"], out["extension_predicates"])) and axis == "auto":
        return {}
    return out


def normalize_predicate_names(value: Any) -> list[str]:
    names: list[str] = []
    for p in _str_list(value):
        key = re.sub(r"[^a-z0-9_]+", "", p.strip().lower().replace("-", "_").replace(" ", "_")).strip("_")
        if key and key not in names:
            names.append(key)
    return names


def validate_graph_schema_config(predicates: Any, parent_types: Any) -> list[str]:
    errors: list[str] = []
    if predicates not in (None, "", (), []):
        if not isinstance(predicates, (list, tuple, str)):
            errors.append("graph_predicates must be a list")
        elif len(normalize_predicates(predicates, limit=None)) > GRAPH_PREDICATES_MAX:
            errors.append(f"At most {GRAPH_PREDICATES_MAX} predicates")
    if parent_types not in (None, "", {}):
        if not isinstance(parent_types, dict):
            errors.append("graph_parent_types must be an object")
        elif len(normalize_parent_types(parent_types)) < len(parent_types):
            errors.append("The values of graph_parent_types must be one of the eight upper classes")
    return errors


def overlap_cap(max_tokens: int, limits: dict[str, Any]) -> int:
    """Exclusive upper bound of overlap_tokens under the current max_tokens (the bound itself cannot be
    used)."""
    effective = int(limits.get("effective_max_model_len") or FALLBACK_MAX_MODEL_LEN)
    return max(0, min(effective - int(max_tokens), int(max_tokens) // 2))


def validate_chunk_config(max_tokens: Any, overlap_tokens: Any, limits: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        max_tokens = int(max_tokens)
    except (TypeError, ValueError):
        return ["max_tokens must be an integer"]
    try:
        overlap_tokens = int(overlap_tokens)
    except (TypeError, ValueError):
        return ["overlap_tokens must be an integer"]

    cap = int(limits["max_tokens_cap"])
    floor = int(limits["max_tokens_min"])
    if not (floor <= max_tokens <= cap):
        served = limits.get("embedding_max_model_len")
        ratio = int(float(limits.get("cap_ratio", CAP_RATIO)) * 100)
        source = (
            f"the embedding service's max_model_len={served}"
            if served
            else f"the embedding service did not respond, falling back to {limits.get('effective_max_model_len', FALLBACK_MAX_MODEL_LEN)}"
        )
        errors.append(f"max_tokens must be between {floor} and {cap} (the cap is {ratio}% of {source})")
        # When max_tokens itself is invalid, also reporting the overlap bound would only give a number
        # computed from a wrong baseline and add confusion.
        return errors

    ov_cap = overlap_cap(max_tokens, limits)
    if not (0 <= overlap_tokens < ov_cap):
        effective = int(limits.get("effective_max_model_len") or FALLBACK_MAX_MODEL_LEN)
        errors.append(
            f"overlap_tokens must be between 0 and {max(0, ov_cap - 1)} "
            f"(it must be below min({effective} − {max_tokens}, {max_tokens} ÷ 2) = {ov_cap})"
        )
    return errors


def apply_schema_version(current: dict[str, Any], updates: dict[str, Any]) -> None:
    """把 updates 里的 graph_schema_active 展开成真正生效的模式字段(原地改)。

    展开而不是让建图去查版本表,是刻意的:生效值只有一处,建图、重建策略、
    提示词渲染、检索目录读的都是它。版本列表只负责「能看见、能回退」。
    版本条目里存的是 LinkML(ontology.to_linkml 的产物),展开走 ontology.expand_model;
    语料示例不在 LinkML 里,单独随版本生效。
    """
    from .ontology import entry_model, expand_model

    raw = updates.get("graph_schema_active")
    if raw is None:
        return
    chosen = str(raw).strip()
    if not chosen or chosen == CURRENT_SCHEMA_VERSION_ID:
        # The fallback entry is not a real choice: keep things as they are and drop the key altogether.
        updates.pop("graph_schema_active", None)
        return
    versions = list(current.get("graph_schema_versions") or [])
    match = next((v for v in versions if str(v.get("id")) == chosen), None)
    if match is None:
        raise ValueError(f"The label version does not exist or has been rotated out: {chosen}")
    model = entry_model(match)
    if model is None:
        raise ValueError(f"Cannot read the schema of label version {chosen}")
    updates.update(expand_model(model))
    updates["graph_examples"] = normalize_examples(match.get("examples")) or None
    updates["graph_schema_active"] = chosen


# 模式展开出来的生效字段:只由 apply_schema_version 写,控制台与 CLI 不能直接改(改模式 = 上传或抽一版新的)
SCHEMA_EFFECTIVE_KEYS = (
    "graph_entity_types", "graph_language", "graph_predicates", "graph_parent_types", "graph_type_parents",
    "graph_type_scopes", "graph_type_definitions", "graph_examples", "graph_profile",
)
