"""Preserve still-supported schema definitions during automatic resampling.

Usage does not validate a new endpoint. Reuse requires a current source and an
assertion that already conforms to the previous definition; violations stay out.
Supported meanings are retained; grounded additive endpoints can evolve them.
"""
from __future__ import annotations

from typing import Any

from .extract import ExtractionSchema
from .quality import relation_quality


def supported_definitions(graph: dict[str, Any], model: dict[str, Any], live_units: set[str]) -> dict[str, list[str]]:
    schema = ExtractionSchema.from_model(model)
    types = set(schema.entity_types)
    entities = {str(e["key"]): e for e in graph.get("entities", [])}

    def live(row):
        return (not row.get("boilerplate") and not row.get("derived")
                and bool(live_units.intersection(str(x) for x in row.get("unit_ids", []))))

    kept_types = {str(e.get("type")) for e in entities.values() if live(e) and str(e.get("type")) in types}
    predicates = {p["name"]: p for p in model.get("predicates", [])}
    kept_preds = set()
    for row in graph.get("relations", []):
        p = predicates.get(str(row.get("predicate")))
        a, b = entities.get(str(row.get("source_key"))), entities.get(str(row.get("target_key")))
        if not p or not a or not b or not live(row) or relation_quality(row) == "needs_review":
            continue
        if str(a.get("type")) not in types or str(b.get("type")) not in types:
            continue
        if (schema.endpoint_ok(str(a["type"]), set(p.get("source_parents") or []))
                and schema.endpoint_ok(str(b["type"]), set(p.get("target_parents") or []))):
            kept_preds.add(p["name"])
            kept_types.update((a["type"], b["type"]))
    return {"types": sorted(kept_types), "predicates": sorted(kept_preds)}


def reuse_supported_model(model: dict[str, Any], prior: dict[str, Any] | None, *,
                          endpoint_evidence: dict[str, Any] | None = None,
                          samples: list[str] = ()) -> tuple[dict[str, Any], dict[str, Any]]:
    from ..ontology import normalize_model

    support = (prior or {}).get("source_support") or {}
    old = (prior or {}).get("schema_model") or {}
    guard = {"kept_predicates": [], "widened": []}
    old_types = {t["name"]: t for t in old.get("types", [])}
    old_preds = {p["name"]: p for p in old.get("predicates", [])}
    wanted = set(support.get("types") or []) & old_types.keys()
    preds = set(support.get("predicates") or []) & old_preds.keys()
    # A retained constraint may name a parent category, including a custom one.
    for name in preds:
        for end in ("source_parents", "target_parents"):
            wanted.update(x for x in old_preds[name].get(end, []) if x in old_types)
    pending = list(wanted)
    while pending:
        parent = old_types[pending.pop()].get("is_a")
        if parent in old_types and parent not in wanted:
            wanted.add(parent)
            pending.append(parent)
    if not wanted and not preds:
        return model, guard
    guard.update(kept_types=[], deferred_changes=[])
    proposed_types = {t["name"]: t for t in model.get("types", [])}
    proposed_preds = {p["name"]: p for p in model.get("predicates", [])}
    kept = [dict(t) for t in old.get("types", []) if t["name"] in wanted]
    guard["kept_types"] = [t["name"] for t in kept]
    for name in sorted(wanted):
        if name in proposed_types and proposed_types[name] != old_types[name]:
            guard["deferred_changes"].append({"kind": "type", "name": name})
    # Proposal budgets are not lifetime quotas. A source-supported old category
    # must not consume the slots needed to represent a newly documented topic.
    for t in model.get("types", []):
        if t["name"] in wanted:
            continue
        kept.append(dict(t))
    names = {t["name"] for t in kept}
    for t in kept:
        if t.get("is_a") in proposed_types and t["is_a"] not in names:
            t["is_a"] = "entity"
    kept_preds = [dict(p) for p in old.get("predicates", []) if p["name"] in preds]
    guard["kept_predicates"] = [p["name"] for p in kept_preds]
    combined_schema = ExtractionSchema.from_model({**model, "types": kept})
    for predicate in kept_preds:
        name = predicate["name"]
        proposal = proposed_preds.get(name) or {}
        # Replacing a direction, cardinality or meaning is a governance action.
        # An additive extension needs an actual quote and typed named endpoints,
        # not a count of historical extraction violations.
        if (proposal.get("description") != predicate.get("description")
                or proposal.get("multivalued") != predicate.get("multivalued")
                or any(not set(predicate.get(k, [])) <= set(proposal.get(k, []))
                       for k in ("source_parents", "target_parents"))):
            continue
        for evidence in (endpoint_evidence or {}).get(name, []):
            if not isinstance(evidence, dict):
                continue
            quote = str(evidence.get("quote") or "").strip()
            a, b = str(evidence.get("source") or ""), str(evidence.get("target") or "")
            st, tt = str(evidence.get("source_type") or ""), str(evidence.get("target_type") or "")
            if (len(quote) < 16 or not any(quote in sample for sample in samples)
                    or not a or not b or a == b or a not in quote or b not in quote
                    or st not in names or tt not in names
                    or not combined_schema.endpoint_ok(st, set(proposal.get("source_parents", [])))
                    or not combined_schema.endpoint_ok(tt, set(proposal.get("target_parents", [])))):
                continue
            added = {}
            for key, typ in (("source_parents", st), ("target_parents", tt)):
                if not combined_schema.endpoint_ok(typ, set(predicate.get(key, []))):
                    predicate[key] = [*predicate.get(key, []), typ]
                    added[key] = typ
            if added:
                guard["widened"].append({"name": name, **added, "evidence": dict(evidence)})
    def contract(p):
        return (str(p.get("description") or ""), p.get("multivalued") is not False, bool(p.get("core")),
                tuple(sorted(str(x).casefold() for x in p.get("source_parents", []))),
                tuple(sorted(str(x).casefold() for x in p.get("target_parents", []))))

    applied = {p["name"]: p for p in kept_preds}
    for name in sorted(preds):
        if name in proposed_preds and contract(proposed_preds[name]) != contract(applied[name]):
            guard["deferred_changes"].append({"kind": "predicate", "name": name})
    # normalize_model will restore the core event predicates if needed.
    core = [p for p in model.get("predicates", []) if p.get("core") and p["name"] not in preds]
    kept_preds.extend(core)
    for p in model.get("predicates", []):
        if p["name"] in {x["name"] for x in kept_preds}:
            continue
        ends = set(p.get("source_parents", [])) | set(p.get("target_parents", []))
        if not ((ends & proposed_types.keys()) - names):
            kept_preds.append(dict(p))
    result, _ = normalize_model({**model, "types": kept, "predicates": kept_preds})
    return result, guard
