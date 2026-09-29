"""Project a finished graph version into Neo4j.

The input is the output of the local graph build (graph.json + units.jsonl in the workspace); parquet is no
longer read. Node labels, edge types, property names, the (kb_id, graph_version) version isolation and the
KB.active_graph_version activation scheme are all carried over from the old mode -- the recall side
(wiki-search) has Cypher that depends on them:

  KB -HAS_GRAPH_VERSION-> GraphVersion
  Document -HAS_TEXT_UNIT-> TextUnit (extraction unit, body text not stored)
  QdrantChunkSnapshot -CONTRIBUTES_TO-> TextUnit
  TextUnit -MENTIONS-> Entity;  TextUnit -EVIDENCES-> Relation
  Entity -RELATED_TO{type, weight, ...}-> Entity;  Relation -SOURCE/TARGET-> Entity
  Entity -MENTIONED_IN{count}-> QdrantChunkSnapshot          (new: attribution precise to the chunk)

Community-related nodes and edges are no longer produced.
"""
from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase

from .. import db
from ..config import Settings
from ..models import KBSource
from ..vector.qdrant import (
    GRAPH_OPTIONAL_TYPES,
    GRAPH_VECTOR_TYPES,
    client as qdrant_client,
    collection_exists,
    graph_collection_alias,
    graph_collection_name,
    graph_version_timestamp,
)
from .build import GraphBuildLock, graph_paths
from .units import read_units
from .vectors import entity_id, relation_id

NODE_LABELS = (
    "GraphVersion",
    "QdrantChunkSnapshot",
    "Document",
    "TextUnit",
    "Entity",
    "Relation",
    "Spec",
)


def import_graph_to_neo4j(
    settings: Settings,
    *,
    source_key: str,
    source: KBSource,
    graph_version: str | None = None,
    replace: bool = False,
    activate: bool = True,
    dry_run: bool = False,
    batch_size: int = 1000,
    build_record: dict[str, Any] | None = None,
    acquire_lock: bool = False,
    require_qdrant_aliases: bool = True,
) -> dict[str, Any]:
    if acquire_lock and not dry_run:
        lock = GraphBuildLock(settings)
        lock.acquire()
        try:
            return import_graph_to_neo4j(
                settings, source_key=source_key, source=source, graph_version=graph_version,
                replace=replace, activate=activate, dry_run=dry_run, batch_size=batch_size,
                build_record=build_record, acquire_lock=False, require_qdrant_aliases=require_qdrant_aliases,
            )
        finally:
            lock.release()

    resolved = build_record or resolve_graph_build(settings, source_key=source_key, source=source, graph_version=graph_version)
    graph_version = str(resolved["graph_version"])
    output_dir = Path(str(resolved.get("output_dir") or graph_paths(settings, source, graph_version).output_dir))
    preflight_result = preflight(
        settings, source=source, graph_version=graph_version, output_dir=output_dir,
        require_qdrant_aliases=require_qdrant_aliases,
    )
    bundle = load_bundle(output_dir, source=source, graph_version=graph_version)
    expected = bundle["expected_counts"]
    expected["graph_version"] = 1
    expected["kb"] = 1

    result: dict[str, Any] = {
        "dry_run": dry_run, "source": source_key, "kb_id": source.kb_id,
        "source_collection": source.collection, "graph_version": graph_version,
        "output_dir": str(output_dir), "replace": replace, "activate": activate,
        "batch_size": batch_size, "preflight": preflight_result, "expected_counts": expected,
        "steps": [],
    }
    if dry_run:
        result["steps"].append("dry_run")
        return result

    driver = neo4j_driver(settings)
    try:
        ensure_constraints(driver)
        result["steps"].append("ensure_constraints")

        existing_nodes = count_version_nodes(driver, source.kb_id, graph_version)
        if existing_nodes and not replace:
            raise RuntimeError(
                f"Neo4j graph version already exists for {source.kb_id}:{graph_version}; "
                "rerun with --replace to rebuild this Neo4j projection"
            )
        if existing_nodes and replace:
            result["deleted_existing_nodes"] = delete_version(driver, source.kb_id, graph_version)
            result["steps"].append("delete_existing")

        graph_version_uid = graph_uid(source.kb_id, graph_version, "GraphVersion", graph_version)
        upsert_kb_and_graph_version(
            driver, source_key=source_key, source=source, graph_version=graph_version,
            graph_version_uid=graph_version_uid, output_dir=output_dir, build=resolved, activate=False,
        )
        result["steps"].append("upsert_graph_version")

        import_nodes(driver, "Document", bundle["documents"], graph_version_uid, batch_size)
        import_nodes(driver, "TextUnit", bundle["text_units"], graph_version_uid, batch_size)
        import_nodes(driver, "QdrantChunkSnapshot", bundle["qdrant_chunks"], graph_version_uid, batch_size)
        import_nodes(driver, "Entity", bundle["entities"], graph_version_uid, batch_size)
        import_nodes(driver, "Relation", bundle["relations"], graph_version_uid, batch_size)
        import_nodes(driver, "Spec", bundle.get("specs") or [], graph_version_uid, batch_size)
        result["steps"].append("import_nodes")

        import_edges(driver, bundle, batch_size)
        result["steps"].append("import_edges")

        actual = neo4j_counts(driver, source.kb_id, graph_version)
        result["actual_counts"] = actual
        result["matches"] = counts_match(expected, actual)
        if not result["matches"]:
            raise RuntimeError(f"Neo4j import count mismatch: expected={expected}, actual={actual}")
        result["steps"].append("verify")

        if activate:
            activate_graph_version(driver, source=source, graph_version=graph_version, graph_version_uid=graph_version_uid)
            result["steps"].append("activate")
        return result
    finally:
        driver.close()


def neo4j_status(settings: Settings, *, source: KBSource, graph_version: str | None = None) -> dict[str, Any]:
    driver = neo4j_driver(settings)
    try:
        active_version = active_neo4j_graph_version(driver, source.kb_id)
        selected_version = graph_version or active_version
        result: dict[str, Any] = {
            "kb_id": source.kb_id, "source_collection": source.collection,
            "active_graph_version": active_version, "selected_graph_version": selected_version,
        }
        if selected_version:
            result["counts"] = neo4j_counts(driver, source.kb_id, selected_version)
        return result
    finally:
        driver.close()


def activate_neo4j_graph_version(settings: Settings, *, source: KBSource, graph_version: str) -> dict[str, Any]:
    driver = neo4j_driver(settings)
    try:
        nodes = count_version_nodes(driver, source.kb_id, graph_version)
        if nodes <= 0:
            raise RuntimeError(f"cannot activate missing Neo4j graph version: {source.kb_id}:{graph_version}")
        graph_version_uid = graph_uid(source.kb_id, graph_version, "GraphVersion", graph_version)
        activate_graph_version(driver, source=source, graph_version=graph_version, graph_version_uid=graph_version_uid)
        active_version = active_neo4j_graph_version(driver, source.kb_id)
        if active_version != graph_version:
            raise RuntimeError(
                f"Neo4j graph activation verification failed: expected={graph_version}, actual={active_version}"
            )
        return {"kb_id": source.kb_id, "graph_version": graph_version, "nodes": nodes, "active_graph_version": active_version}
    finally:
        driver.close()


def delete_neo4j_graph_version(settings: Settings, *, source: KBSource, graph_version: str, dry_run: bool = False) -> dict[str, Any]:
    driver = neo4j_driver(settings)
    try:
        count = count_version_nodes(driver, source.kb_id, graph_version)
        result = {"dry_run": dry_run, "kb_id": source.kb_id, "graph_version": graph_version, "nodes": count}
        if not dry_run:
            result["deleted_nodes"] = delete_version(driver, source.kb_id, graph_version)
        return result
    finally:
        driver.close()


def delete_old_neo4j_graph_versions(
    settings: Settings,
    *,
    sources: Iterable[KBSource],
    retention_days: int,
    dry_run: bool = False,
    delete_unparseable: bool = False,
    keep_latest: int | None = None,
    discard_versions: Iterable[str] = (),
) -> dict[str, Any]:
    """Delete superseded Neo4j graph versions with the same rule as vector.qdrant.delete_old_graph_collections:
    with ``keep_latest`` the active version plus the ``keep_latest - 1`` newest other versions are kept
    unconditionally and everything older is deleted, age plays no part; without ``keep_latest`` days only.
    The active version is never deleted.
    ``discard_versions``: cancelled / failed / rolled-back versions take no keep slot and are deleted outright;
    nodes left behind by an import killed midway are removed by version id as well."""
    if retention_days < 1:
        raise ValueError("retention_days must be >= 1")
    if keep_latest is not None and int(keep_latest) < 1:
        raise ValueError("keep_latest must be >= 1")
    cutoff_ts = int(time.time()) - retention_days * 24 * 3600
    discard = {str(v) for v in discard_versions if str(v)}
    driver = neo4j_driver(settings)
    try:
        result: dict[str, Any] = {
            "dry_run": dry_run, "retention_days": retention_days, "cutoff_ts": cutoff_ts, "keep_latest": keep_latest,
            "delete_unparseable": delete_unparseable, "discard_versions": sorted(discard), "sources": {},
            "total_deleted_nodes": 0,
        }
        for source in sources:
            active_version = active_neo4j_graph_version(driver, source.kb_id)
            source_result: dict[str, Any] = {
                "kb_id": source.kb_id, "source_collection": source.collection,
                "active_graph_version": active_version, "deleted": [], "skipped": [],
            }
            rows = list(neo4j_graph_versions(driver, source.kb_id))
            stamps = {str(r.get("graph_version") or ""): (graph_version_timestamp(str(r.get("graph_version") or ""))
                                                          or _int_or_none(r.get("imported_at"))) for r in rows}
            # The active version takes a slot of its own; the others compete for the remaining slots, newest first
            ranked = sorted((v for v, ts in stamps.items() if v and ts is not None and v not in discard and v != active_version),
                            key=lambda v: -stamps[v])
            rank_of = {v: i for i, v in enumerate(ranked)}
            slots = max(0, int(keep_latest) - (1 if active_version else 0)) if keep_latest is not None else None
            # Half-imported versions without a GraphVersion marker (killed right after the import started):
            # their nodes are removed by version id too
            listed = {str(r.get("graph_version") or "") for r in rows}
            for version in sorted(discard - listed):
                if version == active_version:
                    continue
                nodes = count_version_nodes(driver, source.kb_id, version)      # another base's version counts 0 under this kb_id
                if nodes:
                    deleted_nodes = nodes if dry_run else delete_version(driver, source.kb_id, version)
                    source_result["deleted"].append({"graph_version": version, "version_ts": None, "nodes": nodes,
                                                     "deleted_nodes": deleted_nodes, "reason": "unsuccessful"})
                    result["total_deleted_nodes"] += deleted_nodes
            for row in rows:
                version = str(row.get("graph_version") or "")
                if not version:
                    source_result["skipped"].append({"reason": "missing_graph_version", **row})
                    continue
                if version == active_version:
                    source_result["skipped"].append({"graph_version": version, "reason": "active"})
                    continue
                if version in discard:
                    nodes = count_version_nodes(driver, source.kb_id, version)
                    deleted_nodes = nodes if dry_run else delete_version(driver, source.kb_id, version)
                    source_result["deleted"].append({"graph_version": version, "version_ts": stamps.get(version), "nodes": nodes,
                                                     "deleted_nodes": deleted_nodes, "reason": "unsuccessful"})
                    result["total_deleted_nodes"] += deleted_nodes
                    continue
                version_ts = stamps.get(version)
                if version_ts is None:
                    if not delete_unparseable:
                        source_result["skipped"].append({"graph_version": version, "reason": "unparseable_timestamp"})
                        continue
                    reason = "unparseable_forced"
                elif slots is not None:
                    rank = rank_of.get(version, 0)
                    if rank < slots:
                        source_result["skipped"].append({"graph_version": version, "reason": "within_keep_latest",
                                                         "version_ts": version_ts, "rank": rank})
                        continue
                    reason = "beyond_keep_latest"
                elif version_ts >= cutoff_ts:
                    source_result["skipped"].append({"graph_version": version, "reason": "not_expired", "version_ts": version_ts})
                    continue
                else:
                    reason = "expired"
                nodes = count_version_nodes(driver, source.kb_id, version)
                deleted_nodes = nodes if dry_run else delete_version(driver, source.kb_id, version)
                source_result["deleted"].append({"graph_version": version, "version_ts": version_ts, "nodes": nodes,
                                                 "deleted_nodes": deleted_nodes, "reason": reason})
                result["total_deleted_nodes"] += deleted_nodes
            result["sources"][source.kb_id] = source_result
        return result
    finally:
        driver.close()


def resolve_graph_build(settings: Settings, *, source_key: str, source: KBSource, graph_version: str | None) -> dict[str, Any]:
    db.init_db(settings.state_db)
    with db.connect(settings.state_db) as con:
        if graph_version is None:
            row = db.latest_successful_graph_build(con, source_key)
        else:
            row = con.execute(
                "SELECT * FROM graph_builds WHERE source_collection = ? AND graph_version = ? AND status = 'done' LIMIT 1",
                (source.collection, graph_version),
            ).fetchone()
    if row is None:
        raise RuntimeError(f"no successful graph build found for {source_key} graph_version={graph_version!r}")
    return dict(row)


def bundle_type_counts(bundle: dict[str, Any]) -> dict[str, int]:
    """Row count per graph vector type in graph.json; both writing the vector store and the preflight check
    reconcile against this table."""
    return {"entity": len(bundle.get("entities") or []), "relation": len(bundle.get("relations") or []),
            "spec": len(bundle.get("specs") or []), "page": len(bundle.get("pages") or [])}


def preflight(
    settings: Settings,
    *,
    source: KBSource,
    graph_version: str,
    output_dir: Path,
    require_qdrant_aliases: bool = True,
) -> dict[str, Any]:
    if not settings.neo4j_password:
        raise RuntimeError(
            "Neo4j password is not configured: set NEO4J_PASSWORD in config/knowledge-base.env "
            "(or set GRAPH_NEO4J_IMPORT_AFTER_BUILD=0 to skip the Neo4j import)."
        )
    missing = [str(output_dir / name) for name in ("graph.json", "units.jsonl") if not (output_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing graph build artefact(s): {missing}")
    bundle = json.loads((output_dir / "graph.json").read_text(encoding="utf-8"))
    bundle_counts = bundle_type_counts(bundle)

    q = qdrant_client(settings.qdrant_url, settings.qdrant_api_key)
    aliases = {item.alias_name: item.collection_name for item in q.get_aliases().aliases}
    alias_checks: dict[str, dict[str, Any]] = {}
    qdrant_counts: dict[str, int] = {}
    for graph_type in GRAPH_VECTOR_TYPES:
        alias = graph_collection_alias(source.collection, graph_type)
        expected = graph_collection_name(source.collection, graph_type, graph_version)
        actual = aliases.get(alias)
        alias_checks[graph_type] = {"alias": alias, "expected": expected, "actual": actual, "matches": actual == expected}
        if graph_type in GRAPH_OPTIONAL_TYPES and bundle_counts[graph_type] == 0 and not collection_exists(q, expected):
            qdrant_counts[graph_type] = 0          # optional collections (facts / pages) are not created when this version has no content; preflight expects 0
            continue
        if require_qdrant_aliases and actual != expected:
            raise RuntimeError(f"Qdrant graph alias mismatch for {alias}: expected={expected}, actual={actual}")
        qdrant_counts[graph_type] = int(q.count(collection_name=expected, exact=True).count)
        if qdrant_counts[graph_type] != bundle_counts[graph_type]:
            raise RuntimeError(
                f"{graph_type} count mismatch: qdrant={qdrant_counts[graph_type]} graph.json={bundle_counts[graph_type]}"
            )
    return {
        "neo4j_uri": settings.neo4j_uri, "neo4j_user": settings.neo4j_user, "output_dir": str(output_dir),
        "qdrant_aliases": alias_checks, "require_qdrant_aliases": require_qdrant_aliases,
        "qdrant_counts": qdrant_counts, "bundle_counts": bundle_counts,
    }


def load_bundle(output_dir: Path, *, source: KBSource, graph_version: str) -> dict[str, Any]:
    graph = json.loads((output_dir / "graph.json").read_text(encoding="utf-8"))
    units = read_units(output_dir / "units.jsonl")
    kb, gv = source.kb_id, graph_version
    # combined unit kinds (structural rules + model verdict, see merge.combine_unit_kind); older bundles lack them,
    # so fall back to the unit's own kind
    unit_kinds = {str(k): str(v) for k, v in (graph.get("unit_kinds") or {}).items()}

    documents: dict[str, dict[str, Any]] = {}
    text_units: list[dict[str, Any]] = []
    qdrant_chunks: dict[str, dict[str, Any]] = {}
    document_edges: list[dict[str, Any]] = []
    contributes_to: list[dict[str, Any]] = []
    unit_uid: dict[str, str] = {}
    chunk_uid_of_point: dict[str, str] = {}
    for unit in units:
        doc_uid = graph_uid(kb, gv, "Document", unit.doc_id)
        if doc_uid not in documents:
            props = common_props(source, gv, "Document", unit.doc_id)
            props.update(clean_props({
                "id": unit.doc_id, "doc_id": unit.doc_id, "title": unit.rel_path or unit.doc_id,
                "rel_path": unit.rel_path, "filename": Path(unit.rel_path).name if unit.rel_path else None,
                "source_collection": source.collection, "units": 0,
            }))
            documents[doc_uid] = props
        documents[doc_uid]["units"] = int(documents[doc_uid].get("units") or 0) + 1
        tu_uid = graph_uid(kb, gv, "TextUnit", unit.unit_id)
        unit_uid[unit.unit_id] = tu_uid
        props = common_props(source, gv, "TextUnit", unit.unit_id)
        kind = unit_kinds.get(unit.unit_id) or unit.kind or "body"
        props.update(clean_props({
            "id": unit.unit_id, "document_id": unit.doc_id, "n_tokens": unit.n_tokens,
            "section_path": list(unit.section_path), "section": unit.section_label,
            "chunk_count": len(unit.chunk_uids), "order": unit.order,
            "block_types": list(unit.block_types), "kind": kind,
        }))
        text_units.append(props)
        document_edges.append({"document_uid": doc_uid, "text_unit_uid": tu_uid})
        for order, (point_id, chunk_uid) in enumerate(zip(unit.point_ids, unit.chunk_uids)):
            chunk_uid_of_point[point_id] = chunk_uid
            c_uid = graph_uid(kb, gv, "QdrantChunkSnapshot", point_id)
            if c_uid not in qdrant_chunks:
                cprops = common_props(source, gv, "QdrantChunkSnapshot", point_id)
                cprops.update(clean_props({
                    "point_id": point_id, "chunk_uid": chunk_uid, "doc_id": unit.doc_id,
                    "rel_path": unit.rel_path, "filename": Path(unit.rel_path).name if unit.rel_path else None,
                    "kind": kind,
                }))
                qdrant_chunks[c_uid] = cprops
            edge_uid = f"{c_uid}->{tu_uid}"
            contributes_to.append({
                "uid": edge_uid, "chunk_uid": c_uid, "text_unit_uid": tu_uid,
                "props": clean_props({"uid": edge_uid, "order": order, "point_id": point_id, "chunk_uid": chunk_uid}),
            })

    entities: list[dict[str, Any]] = []
    entity_uid_by_key: dict[str, str] = {}
    for e in graph.get("entities") or []:
        eid = entity_id(e["key"])
        uid = graph_uid(kb, gv, "Entity", eid)
        entity_uid_by_key[e["key"]] = uid
        props = common_props(source, gv, "Entity", eid)
        props.update(clean_props({
            "id": eid, "title": e.get("title"), "type": e.get("type"), "parent_type": e.get("parent_type") or None,
            "description": e.get("description"), "frequency": e.get("frequency"), "degree": e.get("degree"),
            "pagerank": e.get("pagerank"), "aliases": list(e.get("aliases") or [])[:16],
            "unit_count": len(e.get("unit_ids") or []),
            "boilerplate": bool(e.get("boilerplate")), "reference": bool(e.get("reference")),
            "evidence_kind": e.get("evidence_kind") or "body",
            "scope": e.get("scope") or None, "upper": e.get("upper") or None,
        }))
        entities.append(props)

    relations: list[dict[str, Any]] = []
    relation_entity_edges: list[dict[str, Any]] = []
    mentions: list[dict[str, Any]] = []
    evidences: list[dict[str, Any]] = []
    seen_mentions: set[tuple[str, str]] = set()
    seen_evidence: set[tuple[str, str]] = set()
    for e in graph.get("entities") or []:
        uid = entity_uid_by_key[e["key"]]
        for unit_id in e.get("unit_ids") or []:
            tu = unit_uid.get(unit_id)
            if tu and (tu, uid) not in seen_mentions:
                seen_mentions.add((tu, uid))
                mentions.append({"text_unit_uid": tu, "target_uid": uid})
    for r in graph.get("relations") or []:
        src_uid, tgt_uid = entity_uid_by_key.get(r["source_key"]), entity_uid_by_key.get(r["target_key"])
        if not src_uid or not tgt_uid:
            raise RuntimeError(f"relationship endpoint not found: {r.get('source')!r}->{r.get('target')!r}")
        rid = relation_id(r["source_key"], r["target_key"], r["predicate"])
        uid = graph_uid(kb, gv, "Relation", rid)
        props = common_props(source, gv, "Relation", rid)
        props.update(clean_props({
            "id": rid, "source": r.get("source"), "target": r.get("target"),
            "source_entity_uid": src_uid, "target_entity_uid": tgt_uid,
            "type": r.get("predicate"), "directed": bool(r.get("directed")),
            "description": r.get("description"), "weight": r.get("weight"),
            "strength_sum": r.get("strength_sum"), "evidence": r.get("evidence"), "cooccur": r.get("cooccur"),
            "npmi": r.get("npmi"), "combined_degree": r.get("combined_degree"),
            "type_violation": bool(r.get("type_violation")),
            "boilerplate": bool(r.get("boilerplate")), "reference": bool(r.get("reference")),
            "evidence_kind": r.get("evidence_kind") or "body",
        }))
        relations.append(props)
        relation_entity_edges.append({
            "relation_uid": uid, "source_uid": src_uid, "target_uid": tgt_uid,
            "related_props": clean_props({
                "uid": uid, "id": rid, "type": r.get("predicate"), "weight": r.get("weight"),
                "combined_degree": r.get("combined_degree"), "description": r.get("description"),
                "npmi": r.get("npmi"), "cooccur": r.get("cooccur"), "strength_sum": r.get("strength_sum"),
                "directed": bool(r.get("directed")), "type_violation": bool(r.get("type_violation")),
                "boilerplate": bool(r.get("boilerplate")), "reference": bool(r.get("reference")),
                "evidence_kind": r.get("evidence_kind") or "body",
            }),
        })
        for unit_id in r.get("unit_ids") or []:
            tu = unit_uid.get(unit_id)
            if tu and (tu, uid) not in seen_evidence:
                seen_evidence.add((tu, uid))
                evidences.append({"text_unit_uid": tu, "target_uid": uid})

    mentioned_in: list[dict[str, Any]] = []
    seen_mi: set[tuple[str, str]] = set()
    for m in graph.get("mentions") or []:
        e_uid = entity_uid_by_key.get(str(m.get("entity_key")))
        point_id = str(m.get("point_id") or "")
        c_uid = graph_uid(kb, gv, "QdrantChunkSnapshot", point_id)
        if not e_uid or c_uid not in qdrant_chunks or (e_uid, c_uid) in seen_mi:
            continue
        seen_mi.add((e_uid, c_uid))
        mentioned_in.append({"entity_uid": e_uid, "chunk_uid": c_uid, "count": int(m.get("count") or 0)})

    specs: list[dict[str, Any]] = []
    has_spec_edges: list[dict[str, Any]] = []
    of_property_edges: list[dict[str, Any]] = []
    states_edges: list[dict[str, Any]] = []
    from .facts import comparable_number

    for f in graph.get("specs") or []:
        fid = str(f.get("id") or "")
        if not fid:
            continue
        uid = graph_uid(kb, gv, "Spec", fid)
        props = common_props(source, gv, "Spec", fid)
        props.update(clean_props({
            "id": fid, "subject": f.get("subject"), "property": f.get("property"), "symbol": f.get("symbol") or None,
            "value": f.get("value") or None, "min": f.get("min") or None, "typ": f.get("typ") or None, "max": f.get("max") or None,
            "unit": f.get("unit") or None,
            "value_num": f.get("value_num"), "min_num": f.get("min_num"), "typ_num": f.get("typ_num"), "max_num": f.get("max_num"),
            "kinds": json.dumps(f.get("kinds") or {}, ensure_ascii=False, sort_keys=True),
            "quality": f.get("quality") or None,
            "ranges": json.dumps(f.get("ranges") or {}, ensure_ascii=False, sort_keys=True) if f.get("ranges") else None,
            "conditions": json.dumps(f.get("conditions") or {}, ensure_ascii=False, sort_keys=True),
            "note": f.get("note") or None, "unit_id": f.get("unit_id"), "doc_id": f.get("doc_id"),
            "section": f.get("section"),
            # 2026-09-07 fact skeleton: concept, canonical unit, flag, reference range, validity / axis, series and
            # conflict group
            "concept": f.get("concept") or None, "concept_key": f.get("concept_key") or None,
            "unit_canonical": f.get("unit_canonical") or None, "flag": f.get("flag") or None,
            "ref_min": f.get("ref_min") or None, "ref_max": f.get("ref_max") or None,
            "ref_min_num": f.get("ref_min_num"), "ref_max_num": f.get("ref_max_num"), "bound_distance": f.get("bound_distance"),
            "valid_from": f.get("valid_from") or None, "valid_until": f.get("valid_until") or None, "axis": f.get("axis") or None,
            "period_text": f.get("period_text") or None,
            "series_key": f.get("series_key") or None, "series_len": f.get("series_len"), "series_index": f.get("series_index"),
            "conflict_group": f.get("conflict_group") or None,
            # quality state (Codex re-review N02): low confidence, image-text conflict, comparators, whether it is
            # numerically comparable; consistent with the Qdrant projection
            "confidence": f.get("confidence") or None,
            "evidence_conflict": json.dumps(f["evidence_conflict"], ensure_ascii=False, sort_keys=True) if f.get("evidence_conflict") else None,
            "cmps": json.dumps(f["cmps"], ensure_ascii=False, sort_keys=True) if f.get("cmps") else None,
            "comparable": comparable_number(f) is not None,
        }))
        specs.append(props)
        for skey in dict.fromkeys([str(f.get("subject_key") or ""), *(str(k) for k in (f.get("subject_keys") or []))]):
            su = entity_uid_by_key.get(skey)
            if su:
                has_spec_edges.append({"entity_uid": su, "spec_uid": uid})
        pu = entity_uid_by_key.get(str(f.get("property_key") or ""))
        if pu:
            of_property_edges.append({"spec_uid": uid, "entity_uid": pu})
        tu = unit_uid.get(str(f.get("unit_id") or ""))
        if tu:
            states_edges.append({"text_unit_uid": tu, "spec_uid": uid})

    expected_counts = {
        "documents": len(documents), "text_units": len(text_units), "qdrant_chunks": len(qdrant_chunks),
        "specs": len(specs), "has_spec_edges": len(has_spec_edges), "of_property_edges": len(of_property_edges),
        "states_edges": len(states_edges),
        "entities": len(entities), "relations": len(relations),
        "document_text_unit_edges": len(document_edges), "contributes_to_edges": len(contributes_to),
        "mentions_edges": len(mentions), "evidences_edges": len(evidences),
        "related_to_edges": len(relation_entity_edges), "relation_source_edges": len(relation_entity_edges),
        "relation_target_edges": len(relation_entity_edges), "mentioned_in_edges": len(mentioned_in),
    }
    return {
        "specs": specs, "has_spec_edges": has_spec_edges, "of_property_edges": of_property_edges, "states_edges": states_edges,
        "documents": list(documents.values()), "text_units": text_units, "qdrant_chunks": list(qdrant_chunks.values()),
        "entities": entities, "relations": relations, "document_edges": document_edges,
        "contributes_to_edges": contributes_to, "mentions_edges": mentions, "evidences_edges": evidences,
        "relation_entity_edges": relation_entity_edges, "mentioned_in_edges": mentioned_in,
        "expected_counts": expected_counts,
    }


def common_props(source: KBSource, graph_version: str, label: str, identifier: str) -> dict[str, Any]:
    return {"uid": graph_uid(source.kb_id, graph_version, label, identifier), "kb_id": source.kb_id, "graph_version": graph_version}


def graph_uid(kb_id: str, graph_version: str, label: str, identifier: Any) -> str:
    return f"{kb_id}:{graph_version}:{label}:{identifier}"


def clean_props(values: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in values.items():
        prop = neo4j_prop(value)
        if prop is not None:
            cleaned[key] = prop
    return cleaned


def neo4j_prop(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        items = [neo4j_prop(item) for item in value]
        scalar_items = [item for item in items if isinstance(item, (str, bool, int, float))]
        return scalar_items or None
    return str(value)


def neo4j_driver(settings: Settings):
    if not settings.neo4j_password:
        raise RuntimeError("Neo4j password is not configured. Set NEO4J_PASSWORD in config/knowledge-base.env.")
    driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
    driver.verify_connectivity()
    return driver


def ensure_constraints(driver) -> None:
    statements = [
        "CREATE CONSTRAINT ak_kb_id IF NOT EXISTS FOR (n:KB) REQUIRE n.kb_id IS UNIQUE",
        *[
            f"CREATE CONSTRAINT ak_{label.lower()}_uid IF NOT EXISTS FOR (n:{label}) REQUIRE n.uid IS UNIQUE"
            for label in NODE_LABELS
        ],
        "CREATE INDEX ak_entity_title IF NOT EXISTS FOR (n:Entity) ON (n.kb_id, n.graph_version, n.title)",
        "CREATE INDEX ak_entity_type IF NOT EXISTS FOR (n:Entity) ON (n.kb_id, n.graph_version, n.type)",
        "CREATE INDEX ak_chunk_lookup IF NOT EXISTS FOR (n:QdrantChunkSnapshot) ON (n.kb_id, n.graph_version, n.chunk_uid)",
        "CREATE INDEX ak_chunk_point_id IF NOT EXISTS FOR (n:QdrantChunkSnapshot) ON (n.kb_id, n.graph_version, n.point_id)",
        "CREATE INDEX ak_entity_pagerank IF NOT EXISTS FOR (n:Entity) ON (n.kb_id, n.graph_version, n.pagerank)",
        # Graph recall fetches entities / relations by id in batches (UNWIND $ids ... {kb_id, graph_version, id}):
        # without these two indexes every id is filtered over the whole graph version; on kb_003 one hop took 2
        # seconds and two hops 5-10 seconds.
        "CREATE INDEX ak_entity_id IF NOT EXISTS FOR (n:Entity) ON (n.kb_id, n.graph_version, n.id)",
        "CREATE INDEX ak_relation_id IF NOT EXISTS FOR (n:Relation) ON (n.kb_id, n.graph_version, n.id)",
        "CREATE INDEX ak_spec_id IF NOT EXISTS FOR (n:Spec) ON (n.kb_id, n.graph_version, n.id)",
        "CREATE INDEX ak_spec_subject IF NOT EXISTS FOR (n:Spec) ON (n.kb_id, n.graph_version, n.subject)",
    ]
    with driver.session() as session:
        for statement in statements:
            session.run(statement).consume()


def upsert_kb_and_graph_version(
    driver, *, source_key: str, source: KBSource, graph_version: str, graph_version_uid: str,
    output_dir: Path, build: dict[str, Any], activate: bool,
) -> None:
    props = clean_props({
        "uid": graph_version_uid, "kb_id": source.kb_id, "source_key": source_key,
        "source_collection": source.collection, "source_root": source.source_root,
        "graph_version": graph_version, "graph_build_id": build.get("graph_build_id"),
        "started_at": build.get("started_at"), "finished_at": build.get("finished_at"),
        "input_rows": build.get("input_rows"), "active_chunk_count": build.get("active_chunk_count"),
        "active_doc_count": build.get("active_doc_count"), "source_content_hash": build.get("source_content_hash"),
        "output_dir": str(output_dir), "imported_at": int(time.time()), "mode": "entity_graph",
    })
    with driver.session() as session:
        session.run(
            """
            MERGE (kb:KB {kb_id: $kb_id})
            SET kb.title = $title,
                kb.source_collection = $source_collection,
                kb.updated_at = timestamp()
            MERGE (gv:GraphVersion {uid: $graph_version_uid})
            SET gv += $props
            MERGE (kb)-[:HAS_GRAPH_VERSION]->(gv)
            """,
            kb_id=source.kb_id, title=source.source_root, source_collection=source.collection,
            graph_version_uid=graph_version_uid, props=props,
        ).consume()
        if activate:
            activate_graph_version(driver, source=source, graph_version=graph_version, graph_version_uid=graph_version_uid)


def import_nodes(driver, label: str, rows: list[dict[str, Any]], graph_version_uid: str, batch_size: int) -> None:
    relation_name = {
        "Document": "HAS_DOCUMENT", "TextUnit": "HAS_TEXT_UNIT", "QdrantChunkSnapshot": "HAS_QDRANT_CHUNK",
        "Entity": "HAS_ENTITY", "Relation": "HAS_RELATION", "Spec": "HAS_FACT",
    }[label]
    cypher = f"""
    MATCH (gv:GraphVersion {{uid: $graph_version_uid}})
    UNWIND $rows AS row
    MERGE (n:{label} {{uid: row.uid}})
    SET n += row
    MERGE (gv)-[:{relation_name}]->(n)
    """
    run_batches(driver, cypher, rows, batch_size, graph_version_uid=graph_version_uid)


def import_edges(driver, bundle: dict[str, Any], batch_size: int) -> None:
    run_batches(driver, """
        UNWIND $rows AS row
        MATCH (d:Document {uid: row.document_uid})
        MATCH (tu:TextUnit {uid: row.text_unit_uid})
        MERGE (d)-[:HAS_TEXT_UNIT]->(tu)
        """, bundle["document_edges"], batch_size)
    run_batches(driver, """
        UNWIND $rows AS row
        MATCH (c:QdrantChunkSnapshot {uid: row.chunk_uid})
        MATCH (tu:TextUnit {uid: row.text_unit_uid})
        MERGE (c)-[r:CONTRIBUTES_TO {uid: row.uid}]->(tu)
        SET r += row.props
        """, bundle["contributes_to_edges"], batch_size)
    run_batches(driver, """
        UNWIND $rows AS row
        MATCH (tu:TextUnit {uid: row.text_unit_uid})
        MATCH (e:Entity {uid: row.target_uid})
        MERGE (tu)-[:MENTIONS]->(e)
        """, bundle["mentions_edges"], batch_size)
    run_batches(driver, """
        UNWIND $rows AS row
        MATCH (tu:TextUnit {uid: row.text_unit_uid})
        MATCH (rel:Relation {uid: row.target_uid})
        MERGE (tu)-[:EVIDENCES]->(rel)
        """, bundle["evidences_edges"], batch_size)
    run_batches(driver, """
        UNWIND $rows AS row
        MATCH (rel:Relation {uid: row.relation_uid})
        MATCH (s:Entity {uid: row.source_uid})
        MATCH (t:Entity {uid: row.target_uid})
        MERGE (s)-[r:RELATED_TO {uid: row.relation_uid}]->(t)
        SET r += row.related_props
        MERGE (rel)-[:SOURCE]->(s)
        MERGE (rel)-[:TARGET]->(t)
        """, bundle["relation_entity_edges"], batch_size)
    run_batches(driver, """
        UNWIND $rows AS row
        MATCH (e:Entity {uid: row.entity_uid})
        MATCH (c:QdrantChunkSnapshot {uid: row.chunk_uid})
        MERGE (e)-[r:MENTIONED_IN]->(c)
        SET r.count = row.count
        """, bundle["mentioned_in_edges"], batch_size)
    # qualified facts: subject -HAS_SPEC-> Spec -OF_PROPERTY-> property entity; source unit -STATES-> Spec
    run_batches(driver, """
        UNWIND $rows AS row
        MATCH (e:Entity {uid: row.entity_uid})
        MATCH (f:Spec {uid: row.spec_uid})
        MERGE (e)-[:HAS_SPEC]->(f)
        """, bundle.get("has_spec_edges") or [], batch_size)
    run_batches(driver, """
        UNWIND $rows AS row
        MATCH (f:Spec {uid: row.spec_uid})
        MATCH (e:Entity {uid: row.entity_uid})
        MERGE (f)-[:OF_PROPERTY]->(e)
        """, bundle.get("of_property_edges") or [], batch_size)
    run_batches(driver, """
        UNWIND $rows AS row
        MATCH (tu:TextUnit {uid: row.text_unit_uid})
        MATCH (f:Spec {uid: row.spec_uid})
        MERGE (tu)-[:STATES]->(f)
        """, bundle.get("states_edges") or [], batch_size)


def run_batches(driver, cypher: str, rows: list[dict[str, Any]], batch_size: int, **params: Any) -> None:
    if not rows:
        return
    batch_size = max(1, int(batch_size))
    with driver.session() as session:
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            session.execute_write(lambda tx, batch=batch: tx.run(cypher, rows=batch, **params).consume())


def activate_graph_version(driver, *, source: KBSource, graph_version: str, graph_version_uid: str) -> None:
    with driver.session() as session:
        session.run(
            """
            MATCH (kb:KB {kb_id: $kb_id})
            SET kb.active_graph_version = $graph_version,
                kb.active_graph_version_uid = $graph_version_uid,
                kb.updated_at = timestamp()
            """,
            kb_id=source.kb_id, graph_version=graph_version, graph_version_uid=graph_version_uid,
        ).consume()


def active_neo4j_graph_version(driver, kb_id: str) -> str | None:
    with driver.session() as session:
        row = session.run("MATCH (kb:KB {kb_id: $kb_id}) RETURN kb.active_graph_version AS version", kb_id=kb_id).single()
    return str(row["version"]) if row and row["version"] else None


def neo4j_graph_versions(driver, kb_id: str) -> list[dict[str, Any]]:
    with driver.session() as session:
        return session.run(
            """
            MATCH (gv:GraphVersion {kb_id: $kb_id})
            RETURN gv.graph_version AS graph_version, gv.imported_at AS imported_at,
                   gv.started_at AS started_at, gv.finished_at AS finished_at
            ORDER BY gv.graph_version
            """,
            kb_id=kb_id,
        ).data()


def count_version_nodes(driver, kb_id: str, graph_version: str) -> int:
    with driver.session() as session:
        row = session.run(
            "MATCH (n {kb_id: $kb_id, graph_version: $graph_version}) RETURN count(n) AS count",
            kb_id=kb_id, graph_version=graph_version,
        ).single()
    return int(row["count"] if row else 0)


DELETE_BATCH_MIN = 100


def _delete_matching(driver, node_pattern: str, params: dict[str, Any], batch_size: int) -> int:
    """Delete the nodes matching node_pattern in batches: first their relationships batch by batch, then the nodes
    batch by batch, one auto-commit transaction per batch. When a batch hits the transaction memory limit
    (dbms.memory.transaction.total.max, 70% of the heap by default) the batch size is halved and retried, and the
    error is only raised once the size is down to DELETE_BATCH_MIN and it still fails -- on 2026-09-13 an old
    library-KB version with 470k nodes hit the 1.4 GiB limit when one DETACH DELETE removed 5000 nodes (with their
    hundreds of thousands of relationships) and the whole cleanup aborted. Returns the number of deleted nodes."""
    from neo4j.exceptions import TransientError

    total = 0
    queries = (
        (f"MATCH (n {node_pattern})-[r]-() WITH DISTINCT r LIMIT $batch_size DELETE r RETURN count(r) AS n", False),
        (f"MATCH (n {node_pattern}) WITH n LIMIT $batch_size DETACH DELETE n RETURN count(n) AS n", True),
    )
    with driver.session() as session:
        for query, counts in queries:
            size = max(DELETE_BATCH_MIN, int(batch_size))
            while True:
                try:
                    row = session.run(query, **params, batch_size=size).single()
                except TransientError:
                    if size <= DELETE_BATCH_MIN:
                        raise
                    size = max(DELETE_BATCH_MIN, size // 2)
                    continue
                n = int(row["n"] if row else 0)
                if counts:
                    total += n
                if n == 0:
                    break
    return total


def delete_kb_projection(driver, kb_id: str, batch_size: int = 5000) -> int:
    """Delete a KB's entire projection: all versions, the GraphVersion markers and the KB anchor (all carry kb_id)."""
    return _delete_matching(driver, "{kb_id: $kb_id}", {"kb_id": kb_id}, batch_size)


def delete_version(driver, kb_id: str, graph_version: str, batch_size: int = 5000) -> int:
    return _delete_matching(driver, "{kb_id: $kb_id, graph_version: $graph_version}",
                            {"kb_id": kb_id, "graph_version": graph_version}, batch_size)


_COUNT_QUERIES = {
    "kb": "MATCH (n:KB {kb_id: $kb_id}) RETURN count(n) AS count",
    "graph_version": "MATCH (n:GraphVersion {kb_id: $kb_id, graph_version: $graph_version}) RETURN count(n) AS count",
    "documents": "MATCH (n:Document {kb_id: $kb_id, graph_version: $graph_version}) RETURN count(n) AS count",
    "text_units": "MATCH (n:TextUnit {kb_id: $kb_id, graph_version: $graph_version}) RETURN count(n) AS count",
    "qdrant_chunks": "MATCH (n:QdrantChunkSnapshot {kb_id: $kb_id, graph_version: $graph_version}) RETURN count(n) AS count",
    "entities": "MATCH (n:Entity {kb_id: $kb_id, graph_version: $graph_version}) RETURN count(n) AS count",
    "relations": "MATCH (n:Relation {kb_id: $kb_id, graph_version: $graph_version}) RETURN count(n) AS count",
    "document_text_unit_edges": "MATCH (:Document {kb_id: $kb_id, graph_version: $graph_version})-[r:HAS_TEXT_UNIT]->(:TextUnit {kb_id: $kb_id, graph_version: $graph_version}) RETURN count(r) AS count",
    "contributes_to_edges": "MATCH (:QdrantChunkSnapshot {kb_id: $kb_id, graph_version: $graph_version})-[r:CONTRIBUTES_TO]->(:TextUnit {kb_id: $kb_id, graph_version: $graph_version}) RETURN count(r) AS count",
    "mentions_edges": "MATCH (:TextUnit {kb_id: $kb_id, graph_version: $graph_version})-[r:MENTIONS]->(:Entity {kb_id: $kb_id, graph_version: $graph_version}) RETURN count(r) AS count",
    "evidences_edges": "MATCH (:TextUnit {kb_id: $kb_id, graph_version: $graph_version})-[r:EVIDENCES]->(:Relation {kb_id: $kb_id, graph_version: $graph_version}) RETURN count(r) AS count",
    "related_to_edges": "MATCH (:Entity {kb_id: $kb_id, graph_version: $graph_version})-[r:RELATED_TO]->(:Entity {kb_id: $kb_id, graph_version: $graph_version}) RETURN count(r) AS count",
    "relation_source_edges": "MATCH (:Relation {kb_id: $kb_id, graph_version: $graph_version})-[r:SOURCE]->(:Entity {kb_id: $kb_id, graph_version: $graph_version}) RETURN count(r) AS count",
    "relation_target_edges": "MATCH (:Relation {kb_id: $kb_id, graph_version: $graph_version})-[r:TARGET]->(:Entity {kb_id: $kb_id, graph_version: $graph_version}) RETURN count(r) AS count",
    "mentioned_in_edges": "MATCH (:Entity {kb_id: $kb_id, graph_version: $graph_version})-[r:MENTIONED_IN]->(:QdrantChunkSnapshot {kb_id: $kb_id, graph_version: $graph_version}) RETURN count(r) AS count",
    "specs": "MATCH (n:Spec {kb_id: $kb_id, graph_version: $graph_version}) RETURN count(n) AS count",
    "has_spec_edges": "MATCH (:Entity {kb_id: $kb_id, graph_version: $graph_version})-[r:HAS_SPEC]->(:Spec {kb_id: $kb_id, graph_version: $graph_version}) RETURN count(r) AS count",
    "of_property_edges": "MATCH (:Spec {kb_id: $kb_id, graph_version: $graph_version})-[r:OF_PROPERTY]->(:Entity {kb_id: $kb_id, graph_version: $graph_version}) RETURN count(r) AS count",
    "states_edges": "MATCH (:TextUnit {kb_id: $kb_id, graph_version: $graph_version})-[r:STATES]->(:Spec {kb_id: $kb_id, graph_version: $graph_version}) RETURN count(r) AS count",
}


def neo4j_counts(driver, kb_id: str, graph_version: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    with driver.session() as session:
        for key, query in _COUNT_QUERIES.items():
            row = session.run(query, kb_id=kb_id, graph_version=graph_version).single()
            counts[key] = int(row["count"] if row else 0)
    return counts


def counts_match(expected: dict[str, int], actual: dict[str, int]) -> bool:
    return all(int(actual.get(key, -1)) == int(value) for key, value in expected.items())


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
