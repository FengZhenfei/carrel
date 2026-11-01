"""Source-aware graph comparisons. Differences are evidence to inspect, not truth judgements."""
from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

REPORT_LIMIT = 20


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str,
                                     separators=(",", ":")).encode()).hexdigest()


def _get(row, key, default=None):
    return row.get(key, default) if isinstance(row, dict) else getattr(row, key, default)


def source_snapshot(units):
    """Hash source text per document, preserving reading order and duplicate occurrences."""
    docs = defaultdict(list)
    for u in units:
        docs[str(_get(u, "doc_id", ""))].append((int(_get(u, "order", 0)), str(_get(u, "text", "")),
                                                 _get(u, "section_path", []), _get(u, "doc_date", ""),
                                                 _get(u, "axis", {}), _get(u, "source_locations", [])))
    return {doc: digest(sorted(rows, key=lambda r: (r[0], r[1]))) for doc, rows in sorted(docs.items())}


def _docs(row):
    return sorted(set(str(d) for d in row.get("doc_ids") or [row.get("doc_id")] if d))


def _record(row, kind):
    if kind == "entities":
        identity = [row.get("key"), row.get("scope", "")]
        fields = ("key", "title", "type", "kind", "scope", "aliases", "external_ids", "type_path")
    elif kind == "relations":
        identity = [row.get("source_key"), row.get("predicate"), row.get("target_key"),
                    row.get("polarity", "affirmed"), row.get("target_polarity", "affirmed"),
                    row.get("modality", "actual"), row.get("conditions", {}),
                    row.get("valid_from", ""), row.get("valid_until", "")]
        fields = ("source_key", "source", "target_key", "target", "predicate", "polarity", "target_polarity",
                  "modality", "conditions", "valid_from", "valid_until", "type_violation", "quality_state",
                  "derived", "evidence_conflict")
    else:
        identity = [row.get("subject_key") or row.get("subject"), row.get("property"),
                    row.get("conditions", {}), row.get("valid_from", ""), row.get("valid_until", ""),
                    row.get("observed_at", ""), row.get("doc_id"), row.get("source_location")]
        fields = ("subject_key", "subject", "property", "conditions", "valid_from", "valid_until", "observed_at",
                  "source_location", "value", "min", "typ", "max", "unit", "fact_kind", "quality", "time_basis")
    data = {k: row[k] for k in fields if k in row}
    for key in ("aliases", "external_ids"):
        if key in data:
            data[key] = sorted(data[key], key=lambda x: json.dumps(x, sort_keys=True))
    data["doc_ids"] = _docs(row)
    data["unit_ids"] = sorted(set(row.get("unit_ids") or [row.get("unit_id")] or []) - {None, ""})
    return digest(identity), data


def _index(graph, kind):
    result = defaultdict(list)
    for row in graph.get("specs" if kind == "facts" else kind) or []:
        key, data = _record(row, kind)
        result[key].append(data)
    return {k: sorted(v, key=digest) for k, v in result.items()}


def compare_graphs(before, after, *, before_sources=None, after_sources=None,
                   before_manifest=None, after_manifest=None, limit=50):
    """Keep exact counts and bounded examples; changed values are not silently treated as equal."""
    limit = max(0, min(500, int(limit)))
    bm, am = before_manifest or {}, after_manifest or {}
    bs, ns = before_sources or {}, after_sources or {}
    def schema_identity(graph, manifest):
        from ..ontology import entry_model
        from dataclasses import asdict
        from .extract import ExtractionSchema
        model = entry_model(manifest.get("schema_snapshot") or {})
        if model:
            return asdict(ExtractionSchema.from_model(model))
        return {k: v for k, v in (graph.get("schema") or {}).items() if k != "version_id"}
    comparable = {
        "sources": bs == ns if before_sources is not None and after_sources is not None else None,
        "schema": digest(schema_identity(before, bm)) == digest(schema_identity(after, am)),
        "models": (digest(bm["model_identity"]) == digest(am["model_identity"]))
                  if bm.get("model_identity") and am.get("model_identity") else None,
        "corrections": bm.get("corrections_sha", "") == am.get("corrections_sha", ""),
        "pipeline": bm.get("pipeline_revision") == am.get("pipeline_revision"),
    }
    report = {"format": 1, "before": before.get("graph_version"), "after": after.get("graph_version"),
              "comparable": comparable, "changed_inputs": [k for k, v in comparable.items() if v is False],
              "unknown_inputs": [k for k, v in comparable.items() if v is None],
              "source_documents": {"before": len(bs), "after": len(ns),
                                   "added": len(ns.keys() - bs.keys()), "removed": len(bs.keys() - ns.keys()),
                                   "changed": sum(bs[k] != ns[k] for k in bs.keys() & ns.keys())}, "kinds": {}}
    for kind in ("entities", "relations", "facts"):
        old, new = _index(before, kind), _index(after, kind)
        added, removed = sorted(new.keys() - old.keys()), sorted(old.keys() - new.keys())
        changed = sorted(k for k in old.keys() & new.keys() if old[k] != new[k])
        review = []
        for key in removed:
            docs = {d for row in old[key] for d in row["doc_ids"]}
            if docs and all(d in bs and bs[d] == ns.get(d) for d in docs):
                review.append(key)
        report["kinds"][kind] = {
            "before": sum(map(len, old.values())), "after": sum(map(len, new.values())),
            "added": len(added), "removed": len(removed), "changed": len(changed),
            "removed_with_unchanged_source": len(review),
            "examples": {"added": [{"key": k, "after": new[k]} for k in added[:limit]],
                         "removed": [{"key": k, "before": old[k], "source_unchanged": k in review} for k in removed[:limit]],
                         "changed": [{"key": k, "before": old[k], "after": new[k]} for k in changed[:limit]]},
        }
    def violations(g):
        rows = g.get("relations") or []
        return {"violations": sum(bool(r.get("type_violation")) for r in rows), "relations": len(rows)}
    report["endpoint_constraints"] = {"before": violations(before), "after": violations(after)}
    report["review_required"] = any(v["removed_with_unchanged_source"] for v in report["kinds"].values())
    report["interpretation"] = "Source unchanged means evidence can be revisited; it does not prove an old assertion correct."
    return report


def save_report(directory: Path, report):
    """Small reports outlive full graph workspaces; each KB has its own bounded history."""
    directory.mkdir(parents=True, exist_ok=True)
    key = str(report["after"])
    if not key or Path(key).name != key or key in {".", ".."}:
        raise ValueError("invalid graph version")
    path = directory / (key + ".json")
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)
    for old in sorted(directory.glob("*.json"), key=lambda p: (p.stat().st_mtime_ns, p.name), reverse=True)[REPORT_LIMIT:]:
        if old != path:
            old.unlink()
    return path
