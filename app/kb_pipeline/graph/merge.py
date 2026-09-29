"""Merging, filtering, weights and chunk attribution: all local computation, no LLM calls (description
summaries live in summarize.py).

· Entity key = canonical name (NFKC, whitespace folded, case-insensitive); type by majority vote; descriptions
  deduplicated in order; frequency = sum of mentions across units; unit_ids unioned.
· Relations: controlled predicates are directed (HAS_PIN is A → B), the fallback predicate related_to is
  undirected (sorted endpoints). Same (source, target, predicate) merges: strength_sum summed, evidence = number
  of units, descriptions deduplicated.
· Filtering: self-loops, orphan endpoints, negated descriptions ("no clear relationship" and its Chinese
  equivalents …).
· Endpoint type gate (4.10): edges whose endpoint parent types do not satisfy the predicate's allowed pairs are
  flagged type_violation, not deleted.
· Weights: degree (undirected distinct neighbours), combined_degree, edge-weighted pagerank, cooccur (units
  where both endpoints appear), NPMI (all three probabilities over the unit total as denominator, negatives
  kept), final weight = 1 + 9 × (a·norm(strength_sum) + b·norm(cooccur) + c·(npmi+1)/2).
· MENTIONED_IN: look for the entity name in the text of each chunk the unit covers (and in the chunk's
  section_path), attach it to the chunks that hit with a count; an entity that hits no chunk is attached to all
  of the unit's chunks (count=0, meaning inferred rather than matched).
· Noise reduction (2026-09-04, plan in the desktop document on graph build noise reduction): unit kind
  composition (structural rules + model judgement), discounted strength for boilerplate / listing units with a
  boilerplate flag on relations and entities; values (1.3V, 667 MHz) do not become nodes but fold into attribute
  descriptions of the endpoint entity; references (Figure 5, 001-79553) are flagged reference; compound names
  (A/B) are split into their parts; the type health check demotes types dominated by references / boilerplate
  to reference as a whole.
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
from ..limits import DOCUMENT_SCOPED_PARENTS
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
    parent_types: dict[str, str] | None = None,
    allowed_ends: dict[str, tuple[set[str], set[str]]] | None = None,
    unit_kinds: dict[str, str] | None = None,
    upper_parents: dict[str, str] | None = None,
    prior_pairs: Iterable[tuple[str, str, str]] = (),
) -> dict[str, Any]:
    """Combine the per-unit extraction results into one graph. Returns {"entities": [...], "relations": [...],
    "stats": {...}}.
    prior_pairs: endpoint combinations (predicate, source parent, target parent) already confirmed in the previous
    schema version's account, passed directly this version (see relax_prior_pairs).
    unit_kinds: the combined unit kinds (see combine_unit_kind); defaults to each unit's own structural kind.
    upper_parents: {type: upper class}; entities of part / property / process types are scoped to the document
    (the key carries doc_id), so tAS in two manuals is two nodes. Without it everything is global (old
    behaviour)."""
    parent_types = {str(k).casefold(): str(v) for k, v in (parent_types or {}).items()}
    allowed_ends = allowed_ends or {}
    known_types = {str(t).casefold() for t in entity_types}
    scoped_types = {str(t).casefold() for t, up in (upper_parents or {}).items() if str(up) in DOCUMENT_SCOPED_PARENTS}
    upper_of = {str(t).casefold(): str(up) for t, up in (upper_parents or {}).items()}
    kind_of = {u.unit_id: str((unit_kinds or {}).get(u.unit_id) or u.kind or "body") for u in units}
    entities: dict[str, dict[str, Any]] = {}
    relations: dict[tuple[str, str, str], dict[str, Any]] = {}
    stats: Counter = Counter({k: 0 for k in (
        "units_without_extraction", "self_loops_dropped", "negated_dropped", "orphan_dropped",
        "schema_drift_types", "schema_drift_predicates", "type_violations",
        "boilerplate_units", "listing_units", "boilerplate_relations", "boilerplate_entities",
        "value_entities_dropped", "value_relations_folded", "reference_entities", "reference_relations",
        "combined_names_split", "demoted_types")})
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

    for unit in units:
        result = extractions.get(unit.unit_id)
        if not result:
            stats["units_without_extraction"] += 1
            continue
        kind = kind_of.get(unit.unit_id, "body")
        # Keys (with scope) of this unit's names; relation endpoints use the same table so both sides agree on keys
        keys_in_unit: dict[str, str] = {}

        def key_for(name: str, etype: str, scope_doc: str | None = None) -> str:
            # scope_doc: an entity referenced across files in deterministic extraction (file A calls a function of
            # file B) belongs to file B's scope
            base = entity_key(name)
            if not base:
                return ""
            if str(etype or "").casefold() in scoped_types or symbol_like_name(name):
                return scoped_key(base, scope_doc or unit.doc_id)
            return base

        for e in result.get("entities") or []:
            # Keys are recomputed from the name rather than read from the stored extraction: once the
            # normalize_name rules change (e.g. LaTeX restoration), stored extractions still merge under the new
            # rules without re-extraction
            name = str(e.get("name") or "")
            etype = str(e.get("type") or "")
            scope_doc = str(e.get("scope_doc") or "") or None
            key = key_for(name, etype, scope_doc) or str(e.get("key") or "")
            if not key:
                continue
            keys_in_unit[entity_key(name)] = key
            slot = entities.get(key)
            if slot is None:
                slot = entities[key] = {
                    "key": key, "surface": Counter(), "types": Counter(), "descriptions": [], "description_sources": [],
                    "frequency": 0, "unit_ids": [], "doc_ids": [], "kinds": Counter(),
                    "scope": key.rsplit(SCOPE_SEPARATOR, 1)[0] if SCOPE_SEPARATOR in key else "",
                    "exact": False, "aliases": [],
                }
            if e.get("exact"):
                slot["exact"] = True
            for a in e.get("aliases") or []:
                a = normalize_name(str(a))
                if a and a not in slot["aliases"]:
                    slot["aliases"].append(a)
            slot["kinds"][kind] += 1
            slot["surface"][normalize_name(str(e.get("name") or ""))] += int(e.get("mentions") or 1)
            for t, n in (e.get("types") or {str(e.get("type") or ""): 1}).items():
                if t:
                    slot["types"][str(t)] += int(n)
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
            sk = keys_in_unit.get(sb) or sb or str(r.get("source_key") or "")
            tk = keys_in_unit.get(tb) or tb or str(r.get("target_key") or "")
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
                    "predicate_raw": Counter(), "kinds": Counter(),
                }
            slot["kinds"][kind] += 1
            slot["strength_sum"] += float(r.get("strength") or 1.0) * UNIT_KIND_STRENGTH.get(kind, 1.0)
            if unit.unit_id not in slot["unit_ids"]:
                slot["unit_ids"].append(unit.unit_id)
                slot["evidence"] += 1
            raw = str(r.get("predicate_raw") or "")
            if raw:
                slot["predicate_raw"][raw] += 1
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
        parent = parent_types.get(etype.casefold(), "") if etype else ""
        kinds = slot["kinds"]
        boilerplate_only = bool(kinds) and all(k == "boilerplate" for k in kinds)
        if boilerplate_only:
            stats["boilerplate_entities"] += 1
        entity_rows.append({
            "key": key, "title": title, "type": etype, "parent_type": parent,
            "upper": upper_of.get(etype.casefold(), "") if etype else "",
            "scope": slot.get("scope") or "",
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
            src_ok = not ends[0] or "*" in ends[0] or (src["parent_type"] or "").casefold() in ends[0] or (src["type"] or "").casefold() in ends[0]
            tgt_ok = not ends[1] or "*" in ends[1] or (tgt["parent_type"] or "").casefold() in ends[1] or (tgt["type"] or "").casefold() in ends[1]
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
        relation_rows.append({
            "source_key": rel["source_key"], "target_key": rel["target_key"],
            "source": src["title"], "target": tgt["title"],
            "predicate": rel["predicate"], "directed": rel["directed"],
            "descriptions": rel["descriptions"], "description_sources": rel["description_sources"], "description": "",
            "strength_sum": round(float(rel["strength_sum"]), 3), "evidence": int(rel["evidence"]),
            "unit_ids": sorted(rel["unit_ids"], key=lambda u: unit_order.get(u, 0)),
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
    # constraint itself is wrong (health KB: recommends 527/527 violations); flagging them is meaningless and drags
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
    # Then relax per "predicate × endpoint parent pair": below the half mark, but one endpoint pair concentrates
    # the violations (entity → property under associated_with)
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
    return {"entities": entity_rows, "relations": relation_rows, "stats": out_stats}


SCOPE_SEPARATOR = "::"


_SYMBOL_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def symbol_like_name(name: str) -> bool:
    """A single non-CJK character (variables A, r, σ), or two characters including a lower-case letter / digit /
    symbol (x0, r1, σ2): such names are not the same thing in two books, so they only become nodes within a
    document (2026-09-12 library KB: A, C, r, N were merged into global entities across four subject
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
# sporadic error (health KB: 119 entity → property edges under associated_with); the constraint was too narrow,
# so that pair is released
RELAX_PAIR_MIN_EDGES = 20
RELAX_PAIR_RATIO = 0.05      # 20 edges is the main bar; the ratio only guards fragments under big predicates (health KB: process → process 38/407 under part_of is stable usage too)


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
    return {e["key"]: (str(e.get("parent_type") or e.get("upper") or "").casefold() or "-") for e in entities}


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
    """Compound names like "A/B" get no combined node when both parts are already entities: relations attach to
    each part, and mentions and descriptions merge into them."""
    by_key = {e["key"]: e for e in entities}
    split: dict[str, list[str]] = {}
    for e in entities:
        title = str(e.get("title") or "")
        if "/" not in title:
            continue
        parts = [p.strip() for p in title.split("/")]
        scope = str(e.get("scope") or "")
        keys = [scoped_key(entity_key(p), scope) if scope else entity_key(p) for p in parts]
        if len(parts) >= 2 and all(len(p) >= 3 for p in parts) and all(k in by_key and k != e["key"] for k in keys):
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
