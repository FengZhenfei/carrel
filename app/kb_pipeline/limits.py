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


# How many passages "extract / re-extract labels" samples to feed the LLM for inducing the domain /
# language / entity types. Cap 32: this step splices the full text of the passages into one prompt
# (generate_domain / generate_entity_types both do " ".join(docs) internally), so more would overflow
# the context window -- and inducing the domain never needed that many samples anyway.
GRAPH_TUNE_SAMPLE_MIN = 1
GRAPH_TUNE_SAMPLE_MAX = 32
GRAPH_TUNE_SAMPLE_DEFAULT = 8

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

# Id used when the placeholder entry above is materialized into the version ring. It must be a **real
# id** rather than the placeholder id -- the placeholder id is treated as a no-op on the save path, while
# the materialized entry is a real version that can be selected again.
LEGACY_SCHEMA_VERSION_ID = "v-legacy"


def push_schema_version(versions, entry, *, active_id=None, limit=GRAPH_SCHEMA_VERSION_MAX):
    """Push a new version into the ring, newest first, evicting the oldest beyond the limit -- but **never
    the version currently in effect**. Evicting the active version would mean the dropdown can no longer
    select the labels the current build used, while the graph is still built on them.
    """
    kept = [dict(v) for v in (versions or []) if str(v.get("id")) != str(entry.get("id"))]
    out = [dict(entry)] + kept
    if len(out) <= limit:
        return out
    protected = str(active_id or "")
    trimmed = list(out)
    # Evict from the oldest forward, skipping the active version
    for index in range(len(trimmed) - 1, 0, -1):
        if len(trimmed) <= limit:
            break
        if str(trimmed[index].get("id")) == protected:
            continue
        trimmed.pop(index)
    return trimmed[:limit] if len(trimmed) > limit else trimmed


# Global default for the graph extraction type table. KBs that leave graph_entity_types empty use this one.
#
# Why here rather than settings.yaml: settings.yaml is expanded as plain text via Template.substitute,
# which cannot express "keep the literal YAML value when this key is not injected" -- build.graph_env can
# only inject either "this KB's table" or "this default". So the source of truth must live on the Python
# side.
GRAPH_ENTITY_TYPES_DEFAULT = (
    "organization", "person", "product", "module", "feature", "technology",
    "customer", "process", "risk", "requirement", "document",
)

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
            f"(default {GRAPH_TUNE_SAMPLE_DEFAULT}; the full sample text goes into one prompt)"
        )
    return errors


GRAPH_PREDICATES_MAX = 24
GRAPH_PARENT_TYPES_MAX = 8
_PREDICATE_NAME_RE = re.compile(r"[^a-z0-9_]+")


def normalize_predicate_name(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return _PREDICATE_NAME_RE.sub("", text).strip("_")


def normalize_predicates(value: Any, *, limit: int | None = GRAPH_PREDICATES_MAX) -> tuple[dict[str, Any], ...]:
    """Predicate table: [{name, description, source_parents, target_parents}]; names are lower_snake,
    de-duplicated, with the fallback related_to removed, keeping at most limit entries (None = no cut, used
    when validating the count). A string form (names separated by commas / newlines) is accepted too."""
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
        out.append({
            "name": name,
            "description": str(item.get("description") or "").strip()[:200],
            "source_parents": [str(p).strip().lower() for p in (item.get("source_parents") or []) if str(p).strip()][:8],
            "target_parents": [str(p).strip().lower() for p in (item.get("target_parents") or []) if str(p).strip()][:8],
        })
        if limit is not None and len(out) >= limit:
            break
    return tuple(out)


def normalize_parent_types(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in value.items():
        key = str(k or "").strip()
        parent = str(v or "").strip().lower()
        if key and parent:
            out[key] = parent
    return out


# ── Upper ontology (domain-independent, fixed six classes) ─────────────────────────────────────────
# Every sampled entity type maps to one of these six classes. They decide two general rules:
# · Scope: part / property / process belong to something and are scoped within the document (the tAS in
#   two manuals are two nodes); entity / standard / document are global.
# · The "concept" vs "instance" distinction for retrieval and health checks is not done yet; only the
#   scope is used for now.
UPPER_PARENTS = ("entity", "part", "property", "process", "standard", "document")
UPPER_PARENT_DESCRIPTIONS = {
    "entity": "independent things that exist on their own: products, devices, organizations, people, places, systems, materials",
    "part": "constituents that belong to an entity: components, pins, modules, members, fields, ingredients",
    "property": "measurable or descriptive attributes: parameters, metrics, characteristics, ratings, settings",
    "process": "operations, modes, states, procedures, events, instructions, methods",
    "standard": "standards, technologies, protocols, specifications, concepts and terms shared across documents",
    "document": "documents, sections, figures, tables and other references to material",
}
DOCUMENT_SCOPED_PARENTS = frozenset({"part", "property", "process"})
# Self-invented parent type names from old schema versions → upper class (the model's mapping takes
# precedence; this is the fallback)
UPPER_LEGACY_MAP = {
    "entity": "entity", "component": "entity", "组件": "entity", "device": "entity", "product": "entity", "organization": "entity",
    "person": "entity", "location": "entity", "place": "entity", "system": "entity", "material": "entity", "实体": "entity",
    "part": "part", "interface": "part", "接口": "part", "signal": "part", "pin": "part", "register": "part", "module": "part",
    "部件": "part", "零件": "part",
    "property": "property", "parameter": "property", "参数": "property", "attribute": "property", "metric": "property",
    "characteristic": "property", "属性": "property", "指标": "property",
    "process": "process", "mode": "process", "模式": "process", "operation": "process", "event": "process", "state": "process",
    "procedure": "process", "instruction": "process", "过程": "process", "操作": "process",
    "standard": "standard", "标准": "standard", "technology": "standard", "protocol": "standard", "concept": "standard",
    "term": "standard", "概念": "standard", "技术": "standard",
    "document": "document", "文档": "document", "reference": "document", "figure": "document", "table": "document",
}


def upper_parent_of(parent: Any) -> str | None:
    """Parent type name → upper class; returns None when unrecognized (a one-off model mapping at build
    time, or the default entity, serves as the fallback)."""
    key = str(parent or "").strip().lower()
    if not key:
        return None
    if key in UPPER_PARENTS:
        return key
    return UPPER_LEGACY_MAP.get(key)


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


def normalize_upper_parents(value: Any, entity_types: Iterable[str] = ()) -> dict[str, str]:
    """{type: upper class}, keeping only the six classes; when entity_types is given, type names are aligned
    case-insensitively."""
    if not isinstance(value, dict):
        return {}
    lookup = {str(t).casefold(): str(t) for t in entity_types}
    out: dict[str, str] = {}
    for k, v in value.items():
        key = str(k or "").strip()
        if lookup:
            key = lookup.get(key.casefold(), "")
        upper = upper_parent_of(v)
        if key and upper:
            out[key] = upper
    return out


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
        elif len(set(normalize_parent_types(parent_types).values())) > GRAPH_PARENT_TYPES_MAX:
            errors.append(f"At most {GRAPH_PARENT_TYPES_MAX} parent types")
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
    """Expand graph_schema_active in updates into the labels and language that actually take effect (in
    place).

    Expanding, rather than having the graph build look up the version table, is deliberate: the effective
    value lives in one place, and the graph build, the rebuild policy and the prompt rendering all read
    it. The version list is only responsible for "visible and revertible".
    Moved here from kb_server.service on 2026-09-08: the pipeline itself also needs to activate versions
    (automatic label extraction for a blank graph, re-extraction before a full rebuild) and must not
    depend on the server side the other way round.
    """
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
    types = list(normalize_entity_types(match.get("entity_types")))
    updates["graph_entity_types"] = types or None
    updates["graph_language"] = str(match.get("language") or "").strip() or None
    # Schema layer (4.10): the predicate table and parent types take effect together with the version
    predicates = list(normalize_predicates(match.get("predicates")))
    updates["graph_predicates"] = predicates or None
    parents = normalize_parent_types(match.get("parent_types"))
    updates["graph_parent_types"] = parents or None
    # 2026-09-07: type definitions, corpus examples and the scenario profile take effect with the version
    # (cleared when an old version lacks them, falling back to no definitions / generic examples)
    definitions = normalize_type_definitions(match.get("type_definitions"), types)
    updates["graph_type_definitions"] = definitions or None
    updates["graph_examples"] = normalize_examples(match.get("examples")) or None
    updates["graph_profile"] = normalize_profile(match.get("profile")) or None
    updates["graph_schema_active"] = chosen
