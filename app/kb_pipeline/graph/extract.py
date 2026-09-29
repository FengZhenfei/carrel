"""Entity / relation extraction: prompt rendering, the gleaning loop, parsing, and storing results per unit.

The record syntax and the gleaning loop follow GraphRAG (see prompts.py); the parser fixes three upstream
defects: (1) `<|COMPLETE|>` is not preceded by `##` and gets glued onto the last field -- strip it before
splitting; (2) the gleaning rounds' output was appended straight after the first round -- each round is
parsed separately and then merged; (3) the Y/N verdict was neither stripped nor case-insensitive.
The first field is accepted with or without quotes, with half-width or full-width parentheses.

Each unit's result is stored in graph_extractions under (kb_id, unit_id, fingerprint): fingerprint
= model + type list + predicate list + language + gleaning rounds + prompt hash. An existing key is
skipped -- this table is at once the cache, the in-stage resume point and the basis of incremental builds.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from . import prompts
from .tabletext import expand_wide_tables
from .llm import ChatClient, LLMSpec
from .units import UNIT_KINDS, Unit

RECORD_RE = re.compile(r"^[\(（\[]?\s*(.*?)\s*[\)）\]]?$", re.DOTALL)
_STRENGTH_RE = re.compile(r"-?\d+(?:\.\d+)?")
# Units below this many tokens get no gleaning: pressing "many were missed" on a hundred-odd tokens only
# forces the model to invent numeric values and pin numbers as entities (observed on kb_003).
GLEANING_MIN_TOKENS = 300
_WS_RE = re.compile(r"\s+")
_PREDICATE_CLEAN_RE = re.compile(r"[^a-z0-9_]+")


# Output caps per unit kind (LightRAG's quantity limits): boilerplate and listing pages spewing hundreds of
# related_to edges in one go were the main source of boilerplate entities on kb_003 / kb_002; spec tables
# (body units that contain tables) get a looser cap
RECORD_CAPS = {"body": (120, 50), "conclusion": (80, 40), "listing": (30, 20), "boilerplate": (8, 5)}
RECORD_CAPS_TABLE_BODY = (160, 80)
# Output budget of one extraction call. Measured 2026-09-12 on the 40 densest units of the product-documentation
# KB: median output 1285 tokens, maximum 3496, leaving only 15% headroom under 4096; by the record caps (table
# body: 160 entities + 80 relations) the output could in theory reach ten thousand. Tokens not generated cost
# nothing, so 8192; anything still truncated is requested again with twice the budget (see GraphExtractor._ask)
EXTRACT_MAX_TOKENS = 8192


@dataclass(frozen=True)
class ExtractionSchema:
    entity_types: tuple[str, ...]
    predicates: tuple[dict[str, Any], ...] = ()
    language: str = "English"
    parent_types: dict[str, str] = field(default_factory=dict)
    # type definitions (induced in the schema stage): the prompt's type menu "type: definition (parent: ...)"
    type_definitions: dict[str, str] = field(default_factory=dict)
    # few-shot examples generated from this KB's corpus (output of schema.generate_examples); empty means
    # prompts.DEFAULT_EXAMPLES
    examples: str = ""
    # subject types from the scenario profile (what a document is "about"); unused by extraction itself, used by
    # the view layer after the build
    subject_types: tuple[str, ...] = ()

    @property
    def predicate_names(self) -> tuple[str, ...]:
        return tuple(str(p.get("name") or "").strip() for p in self.predicates if str(p.get("name") or "").strip())

    def type_lookup(self) -> dict[str, str]:
        return {t.casefold(): t for t in self.entity_types}

    def predicate_lookup(self) -> dict[str, str]:
        return {normalize_predicate(n): n for n in self.predicate_names}

    def allowed_ends(self) -> dict[str, tuple[set[str], set[str]]]:
        out: dict[str, tuple[set[str], set[str]]] = {}
        for p in self.predicates:
            name = normalize_predicate(str(p.get("name") or ""))
            if not name:
                continue
            src = {str(s).strip().casefold() for s in (p.get("source_parents") or []) if str(s).strip()}
            tgt = {str(s).strip().casefold() for s in (p.get("target_parents") or []) if str(s).strip()}
            out[name] = (src, tgt)
        return out


_LATEX_WRAP_RE = re.compile(r"\\(?:text|mathrm|mathit|mathbf|textrm|operatorname)\s*\{([^{}]*)\}")
_LATEX_OVERLINE_RE = re.compile(r"\\overline\s*\{([^{}]*)\}")
_LATEX_SUBSUP_RE = re.compile(r"[_^]\s*\{([^{}]*)\}")
_LATEX_SUBSUP_BARE_RE = re.compile(r"[_^]([A-Za-z0-9])")
# recognisably LaTeX: a braced subscript / superscript, a known wrapper command, or a paired $...$ holding a
# subscript, superscript, command or brace
_LATEX_EVIDENCE_RE = re.compile(
    r"[_^]\s*\{|\\(?:text|mathrm|mathit|mathbf|textrm|operatorname|overline)\s*\{|\$[^$]*[_^\\{}][^$]*\$")


def _subsup(m: re.Match) -> str:
    """A subscript / superscript drops its marker and joins the preceding name (t_{AS} -> tAS); a numeric
    superscript after a digit keeps its ^ -- 10^{9} joined into 109 would be another number."""
    body = m.group(1)
    after_digit = m.start() > 0 and m.string[m.start() - 1].isdigit()
    if m.group(0)[0] == "^" and after_digit and re.match(r"[-+−]?\d", body):
        return "^" + body
    return body


def strip_latex(name: str) -> str:
    """Names parsed out of datasheets often carry LaTeX: $t_{\\text{AS}}$ -> tAS, $V_{CC}$ -> VCC,
    $\\overline{\\mathrm{CE}}$ -> CE# (overline = active low, written CE# in the body text). The same symbol is
    LaTeX in tables and plain text in the body; without normalisation they become two entities, and the LaTeX
    form has a very low vector similarity to questions.
    Only restored when the name is recognisably LaTeX: a lone $, backslash or underscore belongs to variable
    names, currencies and paths ($SKILL_DIR, A$, C:\\data\\my_file), and treating such names as LaTeX would turn
    them into different names."""
    text = str(name or "")
    if not _LATEX_EVIDENCE_RE.search(text):
        return text
    text = re.sub(r"\s+(?=\$?[_^]\{)", "", text)      # "t $_{AA}$": the space before a subscript
    text = text.replace("$", "")
    for _ in range(4):   # nesting: \overline{\mathrm{CE}}
        before = text
        text = _LATEX_WRAP_RE.sub(r"\1", text)
        text = _LATEX_OVERLINE_RE.sub(r"\1#", text)
        text = _LATEX_SUBSUP_RE.sub(_subsup, text)
        if text == before:
            break
    text = _LATEX_SUBSUP_BARE_RE.sub(_subsup, text)
    text = text.replace("\\", "").replace("{", "").replace("}", "")
    return text


_MATH_RE = re.compile(r"\$([^$]+)\$")
_LATEX_ESCAPED_RE = re.compile(r"\\([%#&])")


def _restore_math(m: re.Match) -> str:
    inner = m.group(1)
    plain = _LATEX_ESCAPED_RE.sub(r"\1", inner)
    for _ in range(4):
        before = plain
        plain = _LATEX_OVERLINE_RE.sub(r"\1", _LATEX_WRAP_RE.sub(r"\1", plain))
        if plain == before:
            break
    if "\\" in plain or (plain == inner and not _LATEX_SUBSUP_RE.search(inner)):
        return m.group(0)
    return strip_latex(m.group(0))


def strip_math(text: str) -> str:
    """LaTeX in value text (a fact's value, unit and conditions): only paired $...$ fragments are restored, and
    only when the fragment holds a braced subscript / superscript, a known wrapper command or an escaped symbol
    and no other command ($V_{CC}$ + 0.5 -> VCC + 0.5, $5\\%$ -> 5%). Everything else is left exactly as it is:
    a lone $, backslash or brace belongs to currencies, paths and variable names ($85K, ${HOME}_dir,
    team\\members), all of which strip_latex would delete; commands such as \\frac{1}{2} or \\pm leave only a run
    of letters and digits once the backslash is gone, so they are better kept verbatim."""
    return _MATH_RE.sub(_restore_math, str(text or ""))


def normalize_name(name: str) -> str:
    """Display name: strip quotes and surrounding whitespace, collapse whitespace, restore LaTeX symbols. Case is
    left alone -- the case of device part numbers is meaningful; merging relies on entity_key's case-insensitive
    comparison."""
    text = unicodedata.normalize("NFKC", strip_latex(str(name or "")))
    text = text.strip().strip("\"'`“”‘’").strip()
    return _WS_RE.sub(" ", text)


def entity_key(name: str) -> str:
    return normalize_name(name).casefold()


# ── Entity-name validity gate (2026-09-12 spot check of five KBs: the library KB turned formula fragments, SQL
#    statements and whole questionnaire sentences into entity names) ──
_NAME_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_NAME_WORD_RE = re.compile(r"[\w\u4e00-\u9fff]")
_CODE_STMT_RE = re.compile(r"^(select|insert|update|delete|create|alter|drop|with)\b", re.IGNORECASE)
_LATEX_WORD_RE = re.compile(r"\b(langle|rangle|frac|sqrt|hat|dagger|sum|int|partial|prime|nabla|varepsilon)\b")
_SHORT_CALL_RE = re.compile(r"^[A-Za-z]{1,8}\([a-z0-9,\s]{1,3}\)$")


def entity_name_ok(name: str) -> bool:
    """Whether a name can be an entity: domain-independent, looks at form only. Rejects equations / formula
    fragments, code statements, half sentences ending in punctuation, questions, pure symbols and over-long
    strings; single Chinese characters (liver, spleen) and single letters (variable names) pass -- single-letter /
    symbol names are handled by entity resolution per document scope and are not merged across documents."""
    text = normalize_name(name)
    if not text or len(text) > 60:
        return False
    if not _NAME_WORD_RE.search(text):
        return False                                   # pure symbols / punctuation
    if "=" in text or ";" in text:
        return False                                   # equations, assignments, statements
    if text[-1] in ",，、:：?？!！" or text.endswith("...") or text.endswith("…"):
        return False                                   # half sentence / question / truncated
    if "?" in text or "？" in text:
        return False
    if _CODE_STMT_RE.match(text):
        return False
    if text.startswith("(") and text.endswith(")"):
        return False                                   # formula fragments like (hat a dagger)
    if len(_LATEX_WORD_RE.findall(text)) >= 2 or _SHORT_CALL_RE.match(text):
        return False                                   # LaTeX-restored symbol strings, function notation like fr(q)
    return True


# ── Grounding filter: example names from the prompt leaking into the extraction results (2026-09-12: an example
#    product name showed up in documents that never mention it) ──
_EXAMPLE_LEAD_RE = re.compile(r"(?:such as|like|e\.g\.|for example|for instance|例如|比如|譬如)\s*([^;.;。]+)", re.IGNORECASE)
_EXAMPLE_ENTITY_RE = re.compile(r'\(\s*["“]?entity["”]?\s*<\|>\s*([^<]+?)\s*<\|>')


def _ground_norm(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(text or ""))).casefold()


def prompt_example_names(schema: "ExtractionSchema") -> set[str]:
    """Names that appear as examples in the prompt: those listed after such as / e.g. in the type definitions, and
    the entity records in the corpus examples."""
    names: set[str] = set()
    for definition in (schema.type_definitions or {}).values():
        for m in _EXAMPLE_LEAD_RE.finditer(str(definition or "")):
            for n in re.split(r",|、|/| or |;", m.group(1)):
                n = n.strip(" '\"()（）")
                if 1 < len(n) <= 40:
                    names.add(_ground_norm(n))
    for m in _EXAMPLE_ENTITY_RE.finditer(str(schema.examples or "")):
        n = _ground_norm(normalize_name(m.group(1)))
        if n:
            names.add(n)
    return names


def ground_records(entities: list[dict[str, Any]], relations: list[dict[str, Any]], *, context: str,
                   example_names: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Drop entities whose name is a prompt example yet cannot be found in the unit text + document name + section,
    together with their relations. Names that really occur in the text are never touched; the comparison
    normalises full / half width, whitespace and case. Returns (entities, relations, number of dropped entities)."""
    if not example_names or not entities:
        return entities, relations, 0
    ctx = _ground_norm(context)
    dropped: set[str] = set()
    kept: list[dict[str, Any]] = []
    for ent in entities:
        n = _ground_norm(ent.get("name"))
        if n and n in example_names and n not in ctx:
            dropped.add(entity_key(str(ent.get("name") or "")))
            continue
        kept.append(ent)
    if not dropped:
        return entities, relations, 0
    rels = [r for r in relations if entity_key(str(r.get("source") or "")) not in dropped and entity_key(str(r.get("target") or "")) not in dropped]
    return kept, rels, len(dropped)


def normalize_predicate(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip().lower()
    text = text.replace("-", "_").replace(" ", "_")
    text = _PREDICATE_CLEAN_RE.sub("", text).strip("_")
    return text


def predicate_menu(schema: ExtractionSchema) -> str:
    """The predicate menu in the prompt. It used to list names only, and the descriptions and endpoint constraints
    (source_parents -> target_parents) were used only afterwards to flag type violations -- the model had never
    seen the constraints, so the violation rate was naturally high (Codex review F05). Now each name carries its
    description and "who -> whom"; the related_to fallback still comes last."""
    items: list[str] = []
    for p in schema.predicates:
        name = str(p.get("name") or "").strip()
        if not name:
            continue
        detail = str(p.get("description") or "").strip()
        src = [str(x).strip() for x in (p.get("source_parents") or ()) if str(x).strip()]
        tgt = [str(x).strip() for x in (p.get("target_parents") or ()) if str(x).strip()]
        if src or tgt:
            detail = (detail + " " if detail else "") + f"({'|'.join(src) or 'any'} -> {'|'.join(tgt) or 'any'})"
        items.append(f"{name}: {detail}" if detail else name)
    if prompts.DEFAULT_PREDICATE not in schema.predicate_names:
        items.append(prompts.DEFAULT_PREDICATE)
    return "; ".join(items)


def type_menu(schema: ExtractionSchema) -> str:
    """The type menu in the prompt: "type: definition (parent: parent type)". Types without a definition get only
    the name and parent. Definitions are induced in the schema stage (prompts.TYPE_DEFINITIONS_PROMPT): for
    neighbouring types such as pin vs signal or parameter vs characteristic, names alone gave a high violation
    rate (34.6% on kb_001); the model has to see "what it is and what it is not" to choose accurately."""
    items: list[str] = []
    for t in schema.entity_types:
        definition = str((schema.type_definitions or {}).get(t) or "").strip().rstrip(".")
        parent = str((schema.parent_types or {}).get(t) or "").strip()
        head = f"{t}: {definition}" if definition else t
        items.append(f"{head} (parent: {parent})" if parent else head)
    return "; ".join(items)


def record_caps(unit_kind: str | None, *, has_table: bool = False) -> tuple[int, int]:
    """(total record cap, entity record cap): by unit kind; body units that contain tables get a looser cap."""
    kind = str(unit_kind or "body")
    if kind == "body" and has_table:
        return RECORD_CAPS_TABLE_BODY
    return RECORD_CAPS.get(kind, RECORD_CAPS["body"])


def render_extract_prompt(unit_text: str, *, section: str, schema: ExtractionSchema, document: str = "",
                          unit_kind: str | None = None, has_table: bool = False) -> str:
    total, entities = record_caps(unit_kind, has_table=has_table)
    return prompts.GRAPH_EXTRACTION_PROMPT.format(
        entity_types=type_menu(schema),
        predicates=predicate_menu(schema),
        language=schema.language or "English",
        examples=(schema.examples or "").strip() or prompts.DEFAULT_EXAMPLES,
        document=document or "(unknown)",
        section=section or "(no section)",
        max_records=int(total), max_entities=int(entities),
        input_text=unit_text,
    )


def _split_records(text: str) -> list[str]:
    body = str(text or "").replace(prompts.COMPLETION_DELIMITER, "")
    records: list[str] = []
    for raw in body.split(prompts.RECORD_DELIMITER):
        raw = raw.strip()
        if not raw:
            continue
        m = RECORD_RE.match(raw)
        records.append(m.group(1) if m else raw)
    return records


def _strength(value: str) -> float:
    m = _STRENGTH_RE.search(str(value or ""))
    if not m:
        return 1.0
    try:
        return float(m.group(0))
    except ValueError:
        return 1.0


_EXTRACTION_NUDGE = ("Your previous reply was not in the required record format. Reply again using ONLY the record format "
                     "described in the instructions (\"entity\" / \"relationship\" tuples with the given delimiters), "
                     "and finish with " + prompts.COMPLETION_DELIMITER + ". If there is genuinely nothing to extract, "
                     "reply with " + prompts.COMPLETION_DELIMITER + " only.")


def extraction_response_ok(text: str) -> bool:
    """For ChatClient.chat(validate=...) (final review F09): a response must either contain at least one well-formed
    record (entity / relation / unit classification) or state "nothing to extract" with the completion delimiter
    alone. Plain prose or nothing but malformed records fails -- such responses used to be parsed as zero
    entities, cached as successes, and hit forever after."""
    body = str(text or "")
    ents, rels, stats = parse_records(body)
    if ents or rels or stats.get("unit_kind"):
        return True
    return prompts.COMPLETION_DELIMITER in body and stats["records"] == 0


def parse_records(text: str, schema: ExtractionSchema | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Parse one model output into (entities, relations, stats). No merging across records."""
    entities: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    stats = {"records": 0, "malformed": 0, "unknown_types": 0, "unknown_predicates": 0}
    types = schema.type_lookup() if schema else {}
    predicates = schema.predicate_lookup() if schema else {}
    for record in _split_records(text):
        fields = [f.strip() for f in record.split(prompts.TUPLE_DELIMITER)]
        stats["records"] += 1
        if len(fields) < 2:
            stats["malformed"] += 1
            continue
        kind = fields[0].strip().strip("\"'“”‘’").lower()
        if kind == "unit":
            value = fields[1].strip().strip("\"'“”‘’").lower()
            stats["unit_kind"] = value if value in UNIT_KINDS else "body"   # type: ignore[assignment]
            continue
        if kind == "entity":
            if len(fields) < 4:
                stats["malformed"] += 1
                continue
            name = normalize_name(fields[1])
            raw_type = normalize_name(fields[2])
            if not name:
                stats["malformed"] += 1
                continue
            if not entity_name_ok(name):
                stats["bad_names"] = int(stats.get("bad_names", 0)) + 1
                continue
            canonical = types.get(raw_type.casefold()) if types else raw_type
            if canonical is None:
                stats["unknown_types"] += 1
                canonical = raw_type.lower()
            entities.append({
                "name": name, "type": canonical, "type_raw": raw_type,
                "description": " ".join(fields[3:]).strip(),
            })
        elif kind == "relationship":
            if len(fields) < 5:
                stats["malformed"] += 1
                continue
            source = normalize_name(fields[1])
            target = normalize_name(fields[2])
            if len(fields) >= 6:
                predicate_raw, description, strength = fields[3], " ".join(fields[4:-1]).strip(), fields[-1]
            else:
                predicate_raw, description, strength = "", fields[3], fields[4]
            if not source or not target:
                stats["malformed"] += 1
                continue
            if not entity_name_ok(source) or not entity_name_ok(target):
                stats["bad_names"] = int(stats.get("bad_names", 0)) + 1
                continue
            predicate = normalize_predicate(predicate_raw)
            if predicate and predicate != prompts.DEFAULT_PREDICATE:
                if predicates:
                    if predicate not in predicates:
                        stats["unknown_predicates"] += 1
                        predicate = prompts.DEFAULT_PREDICATE
                    else:
                        predicate = normalize_predicate(predicates[predicate])
            else:
                predicate = prompts.DEFAULT_PREDICATE
            relations.append({
                "source": source, "target": target, "predicate": predicate,
                "predicate_raw": normalize_predicate(predicate_raw),
                "description": description, "strength": _strength(strength),
            })
        else:
            stats["malformed"] += 1
    return entities, relations, stats


def consolidate(entities: list[dict[str, Any]], relations: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deduplication within one unit: entities are merged by key (descriptions deduplicated, first type kept,
    mentions counted), relations by (source, target, predicate) (maximum strength, descriptions deduplicated)."""
    by_key: dict[str, dict[str, Any]] = {}
    for e in entities:
        key = entity_key(e["name"])
        slot = by_key.get(key)
        if slot is None:
            slot = by_key[key] = {
                "name": e["name"], "key": key, "type": e["type"], "descriptions": [], "mentions": 0,
                "types": {},
            }
        slot["mentions"] += 1
        slot["types"][e["type"]] = slot["types"].get(e["type"], 0) + 1
        desc = str(e.get("description") or "").strip()
        if desc:
            _absorb_description(slot["descriptions"], desc)
    rels: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in relations:
        sk, tk = entity_key(r["source"]), entity_key(r["target"])
        if not sk or not tk or sk == tk:
            continue
        rkey = (sk, tk, r["predicate"])
        slot = rels.get(rkey)
        if slot is None:
            slot = rels[rkey] = {
                "source": r["source"], "target": r["target"], "source_key": sk, "target_key": tk,
                "predicate": r["predicate"], "predicate_raw": r.get("predicate_raw", ""),
                "descriptions": [], "strength": float(r.get("strength") or 1.0),
            }
        slot["strength"] = max(slot["strength"], float(r.get("strength") or 1.0))
        desc = str(r.get("description") or "").strip()
        if desc:
            _absorb_description(slot["descriptions"], desc)
    return list(by_key.values()), list(rels.values())


def _absorb_description(descriptions: list[str], desc: str) -> None:
    """Descriptions of the same entity within one unit: gleaning rounds often rewrite the first round's description
    at greater length (LightRAG's "longer wins"). A new description that contains an old one replaces it, one
    contained in an old one is dropped, otherwise both are kept; exact duplicates are not repeated."""
    folded = desc.casefold()
    for i, old in enumerate(descriptions):
        old_f = old.casefold()
        if folded == old_f or folded in old_f:
            return
        if old_f in folded:
            descriptions[i] = desc
            return
    descriptions.append(desc)


def prompt_hash() -> str:
    parts = [prompts.GRAPH_EXTRACTION_PROMPT, prompts.DEFAULT_EXAMPLES, prompts.CONTINUE_PROMPT, prompts.LOOP_PROMPT]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def extraction_fingerprint(spec: LLMSpec, schema: ExtractionSchema, *, max_gleanings: int) -> str:
    # the type menu (with definitions and parents), the KB's few-shot examples and the per-kind caps all go into
    # the fingerprint -- they are all in the prompt, so changing them invalidates the cache; the predicate menu is
    # still hashed in its full prompt form (name + description + endpoints).
    # The version label follows the stored output: from extract-v6 on, names are only restored when they are
    # recognisably LaTeX
    caps = ",".join(f"{k}={v[0]}/{v[1]}" for k, v in sorted(RECORD_CAPS.items())) + f",table={RECORD_CAPS_TABLE_BODY[0]}/{RECORD_CAPS_TABLE_BODY[1]}"
    raw = "|".join([
        "extract-v6", spec.model_id, spec.protocol,
        type_menu(schema), predicate_menu(schema), schema.language or "",
        hashlib.sha256((schema.examples or "").encode("utf-8")).hexdigest()[:16], caps,
        str(int(max_gleanings)), str(GLEANING_MIN_TOKENS), prompt_hash(),
    ])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


@dataclass
class ExtractionResult:
    unit_id: str
    entities: list[dict[str, Any]]
    relations: list[dict[str, Any]]
    calls: int
    stats: dict[str, Any]

    @property
    def unit_kind(self) -> str:
        """The model's verdict on this unit (body / listing / boilerplate); body when not given."""
        value = str(self.stats.get("unit_kind") or "body")
        return value if value in UNIT_KINDS else "body"


class GraphExtractor:
    """One call (plus gleaning) per unit, producing the unit's entities and relations."""

    def __init__(self, client: ChatClient, schema: ExtractionSchema, *, max_gleanings: int = 1,
                 max_tokens: int = EXTRACT_MAX_TOKENS) -> None:
        self.client = client
        self.schema = schema
        self.max_gleanings = max(0, int(max_gleanings))
        self.max_tokens = int(max_tokens)
        self.example_names = prompt_example_names(schema)

    @property
    def fingerprint(self) -> str:
        return extraction_fingerprint(self.client.spec, self.schema, max_gleanings=self.max_gleanings)

    def extract(self, unit: Unit) -> ExtractionResult:
        has_table = any(str(b).lower() == "table" for b in (unit.block_types or []))
        # wide tables are shown to the model as "column: value" pairs (tabletext): the model miscounts columns in rows of a dozen cells
        prompt = render_extract_prompt(expand_wide_tables(unit.text) or unit.text, section=unit.section_label, schema=self.schema,
                                       document=unit.document_label, unit_kind=unit.kind, has_table=has_table)
        messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
        entities: list[dict[str, Any]] = []
        relations: list[dict[str, Any]] = []
        stats: dict[str, Any] = {"records": 0, "malformed": 0, "unknown_types": 0, "unknown_predicates": 0, "gleanings": 0,
                                 "truncated": 0}
        calls = 0
        response, n = self._ask(messages, stats)
        calls += n
        self._absorb(response, entities, relations, stats)
        rounds = self.max_gleanings if unit.n_tokens >= GLEANING_MIN_TOKENS else 0
        for round_index in range(rounds):
            messages = messages + [{"role": "assistant", "content": response},
                                   {"role": "user", "content": prompts.CONTINUE_PROMPT}]
            response, n = self._ask(messages, stats)
            calls += n
            stats["gleanings"] += 1
            self._absorb(response, entities, relations, stats)
            if round_index >= rounds - 1:
                break
            messages = messages + [{"role": "assistant", "content": response},
                                   {"role": "user", "content": prompts.LOOP_PROMPT}]
            # the answer to the yes/no question becomes response: it is what the next round appends to the history.
            # Appending the previous gleaning output instead would put that output into the history twice, as if
            # the model had answered the yes/no question with it
            response = self.client.chat(messages, max_tokens=8)
            calls += 1
            if not response.strip().upper().startswith("Y"):
                break
        context = "\n".join([unit.text or "", unit.document_label or "", unit.section_label or "", unit.rel_path or ""])
        entities, relations, ungrounded = ground_records(entities, relations, context=context, example_names=self.example_names)
        stats["ungrounded"] = ungrounded
        entities, relations = consolidate(entities, relations)
        return ExtractionResult(unit_id=unit.unit_id, entities=entities, relations=relations, calls=calls, stats=stats)

    def _ask(self, messages: list[dict[str, str]], stats: dict[str, Any]) -> tuple[str, int]:
        """One extraction call. When the output is truncated by max_tokens, the records already written are usable
        but the rest is lost, and this step used to ignore finish_reason, so the loss left no trace: now the call
        is repeated with twice the budget; if still truncated, the longer reply is kept and truncated is counted.
        Returns (text, number of calls)."""
        meta: dict[str, Any] = {}
        response = self.client.chat(messages, max_tokens=self.max_tokens, validate=extraction_response_ok,
                                    correction=_EXTRACTION_NUDGE, meta=meta)
        calls = 1
        if meta.get("truncated"):
            again_meta: dict[str, Any] = {}
            again = self.client.chat(messages, max_tokens=self.max_tokens * 2, validate=extraction_response_ok,
                                     correction=_EXTRACTION_NUDGE, meta=again_meta)
            calls += 1
            if not again_meta.get("truncated") or len(again) > len(response):
                response, meta = again, again_meta
        if meta.get("truncated"):
            stats["truncated"] = int(stats.get("truncated", 0)) + 1
        return response, calls

    def _absorb(self, response: str, entities: list[dict[str, Any]], relations: list[dict[str, Any]],
                stats: dict[str, int]) -> None:
        ents, rels, s = parse_records(response, self.schema)
        entities.extend(ents)
        relations.extend(rels)
        for k, v in s.items():
            if k == "unit_kind":
                stats.setdefault(k, v)       # the first round's verdict wins; gleaning rounds do not override it
            else:
                stats[k] = stats.get(k, 0) + v
