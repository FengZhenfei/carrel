"""Version-checked correction API backed by the existing per-library YAML file."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import yaml

from kb_pipeline import db
from kb_pipeline.graph.build import extraction_schema_for, graph_paths
from kb_pipeline.graph.corrections import MAX_BYTES, corrections_path, parse_corrections
from kb_pipeline.graph.governance import prepare_record
from kb_pipeline.graph.units import read_units


def _source(settings, kb_id):
    found = next(((k, s) for k, s in settings.sources.items() if s.kb_id == kb_id), None)
    if found is None:
        raise KeyError(kb_id)
    return found


def _read(path, kb_id):
    if path is None or not path.is_file():
        raise ValueError("corrections file is unavailable")
    raw = path.read_bytes()
    if len(raw) > MAX_BYTES:
        raise ValueError("corrections file exceeds size limit")
    doc = yaml.safe_load(raw) or {}
    if not isinstance(doc, dict) or not isinstance(doc.get("kb"), dict) or doc["kb"].get("id") != kb_id:
        raise ValueError("corrections library identity mismatch")
    corr = parse_corrections(doc)
    if corr["errors"]:
        raise ValueError("; ".join(corr["errors"]))
    return doc, corr, hashlib.sha256(raw).hexdigest()


def _write(path, doc):
    raw = yaml.safe_dump(doc, allow_unicode=True, sort_keys=False).encode()
    if len(raw) > MAX_BYTES:
        raise ValueError("corrections file exceeds size limit")
    temp = path.with_suffix(".yaml.pending")
    temp.write_bytes(raw)
    os.replace(temp, path)


def _current(settings, source_key, source, con):
    row = db.latest_successful_graph_build(con, source_key)
    if row is None:
        raise ValueError("no completed graph")
    paths = graph_paths(settings, source, str(row["graph_version"]))
    graph = json.loads(paths.graph_file.read_text())
    # The graph's schema, not a newly selected schema waiting for a rebuild.
    manifest = json.loads(row["manifest_json"] or "{}")
    from kb_pipeline.ontology import entry_model
    from kb_pipeline.graph.extract import ExtractionSchema
    model = entry_model(manifest.get("schema_snapshot") or {})
    schema = ExtractionSchema.from_model(model) if model else extraction_schema_for(source)
    return str(row["graph_version"]), graph, schema, read_units(paths.units_file)


def corrections(settings, kb_id, *, payload=None, commit=False, revoke=None):
    source_key, source = _source(settings, kb_id)
    path = corrections_path(settings, kb_id)
    with db.connect(settings.state_db) as con:
        # Serialize API writers and registry deletion. Manual file edits are checked
        # through the revision token immediately before replacement as well.
        if commit or revoke:
            con.execute("BEGIN IMMEDIATE")
        doc, corr, revision = _read(path, kb_id)
        if payload is None:
            latest = db.latest_successful_graph_build(con, source_key)
            manifest = json.loads(latest["manifest_json"] or "{}") if latest is not None else {}
            return {"identity": corr.get("identity", ""), "revision": revision,
                    "records": corr.get("records") or [], "last_build_outcomes": manifest.get("correction_records") or {}}
        if payload.get("library_identity") != corr.get("identity") or not corr.get("identity"):
            raise ValueError("library identity changed")
        request = {k: v for k, v in payload.items() if k not in {"revision", "library_identity"}}
        request_sha = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        existing = next((r for r in doc.get("records") or [] if r["id"] == (revoke or payload.get("id"))), None)
        if not revoke and existing:
            if existing.get("request_sha") != request_sha:
                raise ValueError("correction id already has different content")
            return {"record": existing, "revision": revision, "duplicate": True}
        if payload.get("revision") != revision:
            raise ValueError("corrections changed; read and preview again")
        if revoke:
            if not existing:
                raise KeyError(revoke)
            if not payload.get("submitted_by") or not payload.get("reason"):
                raise ValueError("revocation needs submitted_by and reason")
            if existing.get("state") != "revoked":
                existing.update(state="revoked", revoked_at=int(time.time()), revoked_by=payload["submitted_by"],
                                revoke_reason=payload["reason"])
            record = existing
        else:
            version, graph, schema, units = _current(settings, source_key, source, con)
            record = prepare_record(request, graph, corr, graph_version=version, schema=schema, units=units)
            record["request_sha"] = request_sha
            if not commit:
                return {"record": record, "revision": revision, "status": "ready", "applies_on": "next_graph_build"}
            doc.setdefault("records", []).append(record)
        if _read(path, kb_id)[2] != revision:
            raise ValueError("corrections changed during validation")
        _write(path, doc)
        return {"record": record, "revision": _read(path, kb_id)[2], "applies_on": "next_graph_build"}


def graph_comparisons(settings, kb_id, *, version=""):
    _, source = _source(settings, kb_id)
    short = graph_paths(settings, source).source_short
    directory = settings.graph_work_dir / "reports" / short
    if version:
        if Path(version).name != version or version in {".", ".."}:
            raise ValueError("invalid graph version")
        path = directory / (version + ".json")
        if not path.is_file():
            raise KeyError(version)
        return json.loads(path.read_text())
    reports = []
    for path in sorted(directory.glob("*.json"), reverse=True):
        report = json.loads(path.read_text())
        reports.append({k: report[k] for k in ("before", "after", "status", "review_required", "changed_inputs") if k in report})
    return {"reports": reports}
