"""Scoped correction records. Carrel validates and applies records; callers own business decisions."""
from __future__ import annotations

import copy
import time

from .comparison import digest
from .extract import entity_key

ACTIONS = {"entity.type", "entity.aliases", "entity.merge", "entity.separate", "entity.reject",
           "relation.reject", "fact.reject"}
FIELDS = {"key", "name", "type", "scope", "doc_id", "unit_id", "external_id", "id", "subject_key",
          "subject", "property", "value", "conditions", "valid_from", "valid_until", "observed_at",
          "source_key", "target_key", "predicate", "polarity", "modality"}


def validate_record(record):
    if not isinstance(record, dict) or record.get("action") not in ACTIONS:
        raise ValueError("unknown correction action")
    if (not isinstance(record.get("id"), str) or not record["id"].strip()
            or len(record["id"]) > 128 or any(c in record["id"] for c in "/\\\r\n")):
        raise ValueError("correction id is required")
    targets = record.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("correction targets are required")
    for target in targets:
        if not isinstance(target, dict) or not target or set(target) - FIELDS:
            raise ValueError("invalid target selector")
        if not any(target.get(k) for k in ("key", "id", "name", "subject_key", "subject", "source_key", "external_id")):
            raise ValueError("target needs an identity")
        for key, value in target.items():
            if key == "external_id":
                if not isinstance(value, dict) or set(value) != {"scheme", "value"} or not all(isinstance(x, str) and x for x in value.values()):
                    raise ValueError("external_id needs scheme and value")
            elif key == "conditions":
                if not isinstance(value, dict) or any(not isinstance(v, (str, int, float, bool)) for v in value.values()):
                    raise ValueError("conditions must contain scalar values")
            elif not isinstance(value, (str, int, float, bool)):
                raise ValueError("target fields must be scalar")
    if record["action"] in {"entity.merge", "entity.separate"}:
        if len(targets) < 2:
            raise ValueError("merge/separate needs at least two targets")
    elif len(targets) != 1:
        raise ValueError("this action requires one target")
    if record["action"] == "entity.type" and not isinstance(record.get("value"), str):
        raise ValueError("entity.type needs a type name")
    if record["action"] == "entity.aliases" and (not isinstance(record.get("value"), list) or
            any(not isinstance(a, str) or not a.strip() for a in record["value"])):
        raise ValueError("entity.aliases needs a list of names")
    if record.get("state", "active") not in {"active", "revoked"}:
        raise ValueError("invalid correction state")
    if "basis" in record:
        basis = record["basis"]
        if (not isinstance(basis, dict) or not isinstance(basis.get("source_units"), dict)
                or any(not isinstance(k, str) or not isinstance(v, str) for k, v in basis["source_units"].items())
                or any(not isinstance(basis.get(k), str) for k in ("kb_identity", "graph_version", "schema_sha"))):
            raise ValueError("invalid correction basis")


def matches(row, target):
    for key, value in target.items():
        if key == "name":
            if entity_key(str(value)) not in {entity_key(str(n)) for n in [row.get("title", "")] + list(row.get("aliases") or [])}:
                return False
        elif key == "external_id":
            if value not in (row.get("external_ids") or []):
                return False
        elif key in {"doc_id", "unit_id"}:
            if str(value) not in {str(v) for v in (row.get(key + "s") or []) + [row.get(key)] if v is not None}:
                return False
        elif key == "conditions":
            if any(str((row.get(key) or {}).get(k)) != str(v) for k, v in value.items()):
                return False
        elif key not in row or str(row[key]) != str(value):
            return False
    return True


def resolve(record, rows):
    selected = []
    for target in record["targets"]:
        hits = [r for r in rows if matches(r, target)]
        if len(hits) != 1:
            return [], "ambiguous" if len(hits) > 1 else "unmatched"
        selected.append(hits[0])
    if len({id(r) for r in selected}) != len(selected):
        return [], "duplicate_target"
    return selected, "ready"


def applicable(record, corr):
    basis = record.get("basis") or {}
    context = corr.get("_context") or {}
    if not basis or not context:
        return "missing_basis"
    if basis.get("kb_identity") != corr.get("identity"):
        return "stale_library"
    if basis.get("schema_sha") != context.get("schema_sha"):
        return "stale_schema"
    evidence = basis.get("source_units") or {}
    if not evidence or any(context.get("source_units", {}).get(k) != v for k, v in evidence.items()):
        return "stale_source"
    return "ready"


def apply_records(entities, relations, facts, corr, *, phase, known_types=(), upper_of=None, path_of=None):
    """Explicitly report unmatched, stale and ambiguous rules; never choose the first match."""
    from .resolution import merge_entities
    forbidden = set()
    for record in corr.get("records") or []:
        if record.get("state", "active") != "active" or not record["action"].startswith(phase + "."):
            continue
        status = applicable(record, corr)
        rows = entities if phase == "entity" else relations if phase == "relation" else facts
        picked = []
        if status == "ready":
            picked, status = resolve(record, rows)
        if status == "ready":
            action = record["action"]
            if action == "entity.type":
                value = record["value"]
                if value not in known_types:
                    status = "unknown_type"
                else:
                    picked[0]["type"] = value
                    if upper_of:
                        picked[0]["upper"] = upper_of(value)
                    if path_of:
                        picked[0]["type_path"] = path_of(value)
                    picked[0]["type_corrected"] = True
            elif action == "entity.aliases":
                picked[0]["aliases"] = sorted(set(picked[0].get("aliases") or []) | set(record["value"]))
            elif action == "entity.merge":
                indices = [next(i for i, row in enumerate(entities) if row is p) for p in picked]
                entities, relations, _ = merge_entities(entities, relations, [(indices[0], i) for i in indices[1:]])
            elif action == "entity.separate":
                forbidden |= {frozenset((a["key"], b["key"])) for i, a in enumerate(picked) for b in picked[i + 1:]}
            elif action == "entity.reject":
                key = picked[0]["key"]
                corr.setdefault("_rejected_entities", []).append(copy.deepcopy(picked[0]))
                entities = [r for r in entities if r["key"] != key]
                relations = [r for r in relations if key not in (r.get("source_key"), r.get("target_key"))]
            elif action == "relation.reject":
                relations = [r for r in relations if r is not picked[0]]
            elif action == "fact.reject":
                facts = [r for r in facts if r is not picked[0]]
        corr.setdefault("_outcomes", {})[record["id"]] = "applied" if status == "ready" else status
    return entities, relations, facts, forbidden


def context(schema, units):
    from dataclasses import asdict, is_dataclass
    model = asdict(schema) if is_dataclass(schema) else dict(schema)
    model.pop("examples", None)
    return {"schema_sha": digest(model), "source_units": {u.unit_id: digest(u.text) for u in units}}


def save_state(graph, corr):
    graph["correction_state"] = {k: copy.deepcopy(corr[k]) for k in ("_outcomes", "_rejected_entities") if k in corr}


def restore_state(graph, corr):
    for key, value in (graph.get("correction_state") or {}).items():
        if key in {"_outcomes", "_rejected_entities"}:
            corr[key] = copy.deepcopy(value)


def prepare_record(payload, graph, corr, *, graph_version, schema, units):
    """Preview is also the validator used at commit, with a fresh current-version check."""
    record = copy.deepcopy(payload)
    validate_record(record)
    if record.get("base_graph_version") != graph_version:
        raise ValueError("graph version changed; preview again")
    if (not isinstance(record.get("submitted_by"), str) or not record["submitted_by"].strip()
            or not isinstance(record.get("evidence"), list) or not record["evidence"]
            or any(not isinstance(e, str) or not e.strip() for e in record["evidence"])):
        raise ValueError("submitted_by and evidence references are required")
    phase = record["action"].split(".")[0]
    rows = graph.get({"entity": "entities", "relation": "relations", "fact": "specs"}[phase]) or []
    picked, status = resolve(record, rows)
    if status != "ready":
        raise ValueError(f"correction target is {status}")
    if record["action"] == "entity.type" and record["value"] not in schema.entity_types:
        raise ValueError("unknown target type")
    ctx = context(schema, units)
    ids = {str(uid) for r in picked for uid in (r.get("unit_ids") or [r.get("unit_id")]) if uid}
    if not ids or not ids <= ctx["source_units"].keys():
        raise ValueError("target source evidence is unavailable")
    if not corr.get("identity"):
        raise ValueError("correction library identity is unavailable")
    record["basis"] = {"kb_identity": corr["identity"], "graph_version": graph_version,
                       "schema_sha": ctx["schema_sha"], "source_units": {i: ctx["source_units"][i] for i in sorted(ids)}}
    record["created_at"] = int(time.time())
    record["state"] = "active"
    return record
