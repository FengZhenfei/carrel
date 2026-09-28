from __future__ import annotations

import os

import json
import re
import sys
import time
from typing import Any, Iterable

from qdrant_client import QdrantClient
from qdrant_client.http import models

from ..utils import dir_ancestors_from_rel_path
from .layout import TEXT_VECTOR, VectorLayout


PAYLOAD_INDEX_FIELDS = {
    # every entry here must back an actual filter; unused indexes cost RAM
    "kb_id": models.PayloadSchemaType.KEYWORD,
    "doc_id": models.PayloadSchemaType.KEYWORD,
    "source_path": models.PayloadSchemaType.KEYWORD,
    "dir_ancestors": models.PayloadSchemaType.KEYWORD,
    "content_version": models.PayloadSchemaType.KEYWORD,
    "chunk_index": models.PayloadSchemaType.INTEGER,
    "is_active": models.PayloadSchemaType.BOOL,
    "inactive_at": models.PayloadSchemaType.INTEGER,
}

GRAPH_PAYLOAD_INDEX_FIELDS = {
    "kb_id": models.PayloadSchemaType.KEYWORD,
    "graph_version": models.PayloadSchemaType.KEYWORD,
    "graph_type": models.PayloadSchemaType.KEYWORD,
    "source_collection": models.PayloadSchemaType.KEYWORD,
    "gr_id": models.PayloadSchemaType.KEYWORD,
    "title": models.PayloadSchemaType.KEYWORD,
    "search_text": models.TextIndexParams(
        type=models.TextIndexType.TEXT,
        tokenizer=models.TokenizerType.MULTILINGUAL,
        lowercase=True,
        phrase_matching=True,
    ),
    "type": models.PayloadSchemaType.KEYWORD,
    "parent_type": models.PayloadSchemaType.KEYWORD,
    "degree": models.PayloadSchemaType.INTEGER,
    "frequency": models.PayloadSchemaType.INTEGER,
    "pagerank": models.PayloadSchemaType.FLOAT,
    "weight": models.PayloadSchemaType.FLOAT,
    "source_id": models.PayloadSchemaType.KEYWORD,
    "target_id": models.PayloadSchemaType.KEYWORD,
    "level": models.PayloadSchemaType.INTEGER,
    "doc_id": models.PayloadSchemaType.KEYWORD,
    "content_version": models.PayloadSchemaType.KEYWORD,
    # Noise-reduction flags (2026-09-04): recall seeds are filtered by them
    "boilerplate": models.PayloadSchemaType.BOOL,
    "reference": models.PayloadSchemaType.BOOL,
    "evidence_kind": models.PayloadSchemaType.KEYWORD,
    # Lexical seeds (recall.lexical_seeds): exact lookup by title / alias / relation endpoint
    "aliases": models.PayloadSchemaType.KEYWORD,
    "source": models.PayloadSchemaType.KEYWORD,
    "target": models.PayloadSchemaType.KEYWORD,
    # Structured-fact collection (spec): exact lookup by subject / symbol / property
    "subject": models.PayloadSchemaType.KEYWORD,
    "symbol": models.PayloadSchemaType.KEYWORD,
    "property": models.PayloadSchemaType.KEYWORD,
}

# The entity graph's collections (entities, relations, facts). community (the old pipeline's community reports)
# and raptor (the removed summary-tree mode) are historical types: no longer produced, but GC and alias
# cleanup must still recognize them to reclaim the old collections.
GRAPH_VECTOR_TYPES = ("entity", "relation", "spec", "page")
# Older graph versions have no spec / page collections: a missing one is not an error when switching aliases,
# it is simply not switched
GRAPH_OPTIONAL_TYPES = frozenset({"spec", "page"})
LEGACY_GRAPH_VECTOR_TYPES = ("community", "raptor")
ALL_GRAPH_VECTOR_TYPES = GRAPH_VECTOR_TYPES + LEGACY_GRAPH_VECTOR_TYPES
GRAPH_COLLECTION_RE = re.compile(r"^graph_(?P<source>.+)_(?P<graph_type>entity|relation|spec|page|community|raptor)__(?P<version>.+)$")
# The fact collection uses three named vectors (OG-RAG: property name and value matched separately): text
# for the whole sentence, property for the attribute side, value for the value side; the other graph
# collections remain a single unnamed vector
GRAPH_VECTOR_LAYOUTS: dict[str, tuple[str, ...]] = {"spec": ("text", "property", "value")}


def graph_vector_layout(graph_type: str) -> tuple[str, ...] | None:
    return GRAPH_VECTOR_LAYOUTS.get(graph_type)


GRAPH_VERSION_TS_RE = re.compile(r"(\d{8}-\d{6})")


def client(url: str, api_key: str | None) -> QdrantClient:
    """qdrant-client's default REST read timeout is 5 seconds, while the callers here do 30MB batch upserts
    (wait=True), whole-collection scrolls and filter-based deletes -- easily over 5 seconds while the
    optimizer merges segments or the disk stutters. A timeout makes the client treat a batch the server
    actually wrote as failed, and the job needlessly retries a whole embedding round."""
    timeout = int(os.getenv("QDRANT_TIMEOUT_SECONDS", "120"))
    return QdrantClient(url=url, api_key=api_key, timeout=timeout)


def vectors_config_for(layout: VectorLayout) -> dict[str, models.VectorParams]:
    return {
        name: models.VectorParams(size=size, distance=models.Distance.COSINE)
        for name, size in layout.sizes().items()
    }


def ensure_collection(q: QdrantClient, collection: str, layout: VectorLayout, *, dry_run: bool = False) -> str:
    """Create a KB collection with the named-vector layout (text + visual), or
    validate that an existing one already has exactly that layout."""
    collections = {c.name for c in q.get_collections().collections}
    if collection not in collections:
        if dry_run:
            return f"would-create:{collection}:{layout.describe()}"
        q.create_collection(
            collection_name=collection,
            vectors_config=vectors_config_for(layout),
        )
        created = True
    else:
        created = False
    actual = validate_collection_layout(q, collection, layout)
    if not dry_run:
        for field, schema in PAYLOAD_INDEX_FIELDS.items():
            try:
                q.create_payload_index(
                    collection_name=collection,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception as exc:
                print(
                    f"[qdrant] payload index skipped field={field} schema={schema}: {exc!r}",
                    file=sys.stderr,
                    flush=True,
                )
    return f"{'created' if created else 'exists'}:{collection}:vectors={_describe_layout(actual)}"


def _describe_layout(sizes: dict[str, int]) -> str:
    return ",".join(f"{name or '<unnamed>'}={size}" for name, size in sizes.items())


def collection_vector_layout(q: QdrantClient, collection: str) -> dict[str, int]:
    """Named-vector sizes of a collection; a legacy single unnamed vector is
    reported under the key ""."""
    info = q.get_collection(collection_name=collection)
    vectors = info.config.params.vectors
    size = getattr(vectors, "size", None)
    if size is not None:
        return {"": int(size)}
    if isinstance(vectors, dict):
        result: dict[str, int] = {}
        for name, params in vectors.items():
            inner = _extract_vector_size(params)
            if inner is not None:
                result[str(name)] = inner
        return result
    return {}


def validate_collection_layout(q: QdrantClient, collection: str, layout: VectorLayout) -> dict[str, int]:
    actual = collection_vector_layout(q, collection)
    expected = layout.sizes()
    if actual != expected:
        if "" in actual:
            hint = (
                "the collection uses a single unnamed vector from before the text+visual layout; "
                "Qdrant cannot add named vectors afterwards -- recreate it (kb reset / delete + ensure-collections)"
            )
        else:
            hint = "the set of named vectors or their sizes differ; recreate the collection"
        raise RuntimeError(
            f"Qdrant collection {collection!r} vector layout mismatch: "
            f"actual={_describe_layout(actual)} expected={_describe_layout(expected)}. {hint}."
        )
    return actual


def graph_collection_short_name(source_collection: str) -> str:
    if source_collection.startswith("kb_") and len(source_collection) > 3:
        return source_collection[3:]
    return source_collection


def graph_collection_alias(source_collection: str, graph_type: str) -> str:
    _validate_graph_type(graph_type)
    return f"graph_{graph_collection_short_name(source_collection)}_{graph_type}"


def graph_collection_name(source_collection: str, graph_type: str, graph_version: str) -> str:
    alias = graph_collection_alias(source_collection, graph_type)
    return f"{alias}__{graph_version}"


def graph_alias_targets(
    q: QdrantClient,
    source_collection: str,
    graph_types: Iterable[str] = GRAPH_VECTOR_TYPES,
) -> dict[str, str | None]:
    aliases = {item.alias_name: item.collection_name for item in q.get_aliases().aliases}
    return {
        graph_collection_alias(source_collection, graph_type): aliases.get(
            graph_collection_alias(source_collection, graph_type)
        )
        for graph_type in graph_types
    }


def activate_graph_aliases(
    q: QdrantClient,
    *,
    source_collection: str,
    graph_version: str,
    graph_types: Iterable[str] = GRAPH_VECTOR_TYPES,
) -> dict[str, Any]:
    graph_types = tuple(graph_types)
    previous = graph_alias_targets(q, source_collection, graph_types)
    targets = {
        graph_collection_alias(source_collection, graph_type): graph_collection_name(
            source_collection,
            graph_type,
            graph_version,
        )
        for graph_type in graph_types
    }
    optional_aliases = {graph_collection_alias(source_collection, t) for t in graph_types if t in GRAPH_OPTIONAL_TYPES}
    missing = [target for alias, target in targets.items() if not collection_exists(q, target) and alias not in optional_aliases]
    if missing:
        raise RuntimeError(f"cannot activate missing graph collection(s): {missing}")
    skipped = [alias for alias in optional_aliases if not collection_exists(q, targets[alias])]
    for alias in skipped:
        targets.pop(alias, None)
        previous.pop(alias, None)
    changed = _set_alias_targets(q, previous=previous, targets=targets)
    return {"previous": previous, "targets": targets, "changed": changed, "skipped": skipped}


def restore_graph_aliases(q: QdrantClient, previous: dict[str, str | None]) -> bool:
    current_aliases = {item.alias_name: item.collection_name for item in q.get_aliases().aliases}
    current = {alias: current_aliases.get(alias) for alias in previous}
    return _set_alias_targets(q, previous=current, targets=previous)


def _set_alias_targets(
    q: QdrantClient,
    *,
    previous: dict[str, str | None],
    targets: dict[str, str | None],
) -> bool:
    operations = []
    for alias, target in targets.items():
        current = previous.get(alias)
        if current == target:
            continue
        if current:
            operations.append(
                models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias))
            )
        if target:
            operations.append(
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(collection_name=target, alias_name=alias)
                )
            )
    if operations:
        q.update_collection_aliases(operations)
    return bool(operations)


def parse_graph_collection_name(collection: str) -> dict[str, str] | None:
    match = GRAPH_COLLECTION_RE.match(collection)
    if not match:
        return None
    graph_type = match.group("graph_type")
    _validate_graph_type(graph_type)
    return {
        "source_short": match.group("source"),
        "graph_type": graph_type,
        "graph_version": match.group("version"),
    }


def graph_version_timestamp(graph_version: str) -> int | None:
    matches = GRAPH_VERSION_TS_RE.findall(graph_version)
    if not matches:
        return None
    try:
        return int(time.mktime(time.strptime(matches[-1], "%Y%m%d-%H%M%S")))
    except ValueError:
        return None


def delete_old_graph_collections(
    q: QdrantClient,
    *,
    source_collections: list[str],
    retention_days: int,
    dry_run: bool = False,
    delete_unparseable: bool = False,
    keep_latest: int | None = None,
    discard_versions: Iterable[str] = (),
) -> dict[str, object]:
    """Delete expired graph version collections: past retention_days, or (when keep_latest is given) not
    among the KB's latest keep_latest versions -- incremental append makes versions arrive often, each
    version is a full set of collections, and keeping by days alone would fill the disk. A collection an
    alias points to is never deleted.
    discard_versions: versions left by paused / failed builds (db.unsuccessful_graph_versions); they take no
    slot among the latest keep_latest versions and are deleted outright -- a half-finished version ranked
    by timestamp would occupy a slot and push out the previous good graph (2026-09-13)."""
    if retention_days < 1:
        raise ValueError("retention_days must be >= 1")

    cutoff_ts = int(time.time()) - retention_days * 24 * 3600
    discard = {str(v) for v in discard_versions if str(v)}
    selected_sources = {graph_collection_short_name(collection) for collection in source_collections}
    alias_targets = {item.collection_name for item in q.get_aliases().aliases}
    deleted: list[dict[str, object]] = []
    skipped_collections: list[dict[str, object]] = []

    candidates: list[tuple[str, dict[str, object]]] = []
    for item in sorted(q.get_collections().collections, key=lambda value: value.name):
        parsed = parse_graph_collection_name(item.name)
        if parsed is None or parsed["source_short"] not in selected_sources:
            continue
        candidates.append((item.name, parsed))
    # Rank each KB's versions by time, newest first (the entity / relation / fact collections of one version
    # share a rank)
    rank_of: dict[tuple[str, str], int] = {}
    if keep_latest is not None:
        by_source: dict[str, dict[str, int]] = {}
        for _, parsed in candidates:
            ts = graph_version_timestamp(str(parsed["graph_version"]))
            if ts is not None and str(parsed["graph_version"]) not in discard:
                by_source.setdefault(str(parsed["source_short"]), {})[str(parsed["graph_version"])] = ts
        for short, versions in by_source.items():
            for rank, version in enumerate(sorted(versions, key=lambda v: -versions[v])):
                rank_of[(short, version)] = rank

    for collection, parsed in candidates:
        record: dict[str, object] = {"collection": collection, **parsed}
        if collection in alias_targets:
            skipped_collections.append({**record, "reason": "aliased"})
            continue
        if str(parsed["graph_version"]) in discard:
            delete_status = delete_collection(q, collection, dry_run=dry_run)
            deleted.append({**record, "reason": "unsuccessful", "status": delete_status})
            continue

        version_ts = graph_version_timestamp(str(parsed["graph_version"]))
        if version_ts is None:
            if not delete_unparseable:
                skipped_collections.append({**record, "reason": "unparseable_version_timestamp"})
                continue
            delete_status = delete_collection(q, collection, dry_run=dry_run)
            deleted.append({**record, "reason": "unparseable_forced", "status": delete_status})
            continue

        record["version_ts"] = version_ts
        rank = rank_of.get((str(parsed["source_short"]), str(parsed["graph_version"])))
        beyond = keep_latest is not None and rank is not None and rank >= keep_latest
        if version_ts >= cutoff_ts and not beyond:
            skipped_collections.append({**record, "reason": "within_retention"})
            continue

        delete_status = delete_collection(q, collection, dry_run=dry_run)
        deleted.append({**record, "reason": "beyond_keep_latest" if (beyond and version_ts >= cutoff_ts) else "expired",
                        "status": delete_status})

    return {
        "dry_run": dry_run,
        "retention_days": retention_days,
        "keep_latest": keep_latest,
        "discard_versions": sorted(discard),
        "cutoff_ts": cutoff_ts,
        "selected_sources": sorted(selected_sources),
        "deleted": deleted,
        "skipped_collections": skipped_collections,
        "total_deleted": len(deleted),
    }


def ensure_graph_collection(
    q: QdrantClient,
    collection: str,
    vector_size: int,
    *,
    dry_run: bool = False,
    layout: Iterable[str] | None = None,
) -> str:
    """With layout given, create with named vectors (the fact collection's text / property / value); an existing
    collection has its names and dimensions validated."""
    names = tuple(str(n) for n in (layout or ()))
    collections = {c.name for c in q.get_collections().collections}
    if collection not in collections:
        if dry_run:
            return f"would-create:{collection}"
        if names:
            vectors_config: Any = {n: models.VectorParams(size=vector_size, distance=models.Distance.COSINE) for n in names}
        else:
            vectors_config = models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
        q.create_collection(collection_name=collection, vectors_config=vectors_config)
        created = True
    else:
        created = False
    if names and not created:
        actual = collection_vector_layout(q, collection)
        expected = {n: int(vector_size) for n in names}
        if actual != expected:
            raise RuntimeError(
                f"Qdrant graph collection {collection!r} vector layout mismatch: actual={_describe_layout(actual)} "
                f"expected={_describe_layout(expected)}; recreate the collection (it is versioned, a new build makes a new one)"
            )
        actual_size = int(vector_size)
    elif names:
        actual_size = int(vector_size)      # just created with named vectors: the layout is what we gave
    else:
        actual_size = validate_collection_vector_size(q, collection, vector_size)
    if not dry_run:
        for field, schema in GRAPH_PAYLOAD_INDEX_FIELDS.items():
            try:
                q.create_payload_index(
                    collection_name=collection,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception as exc:
                print(
                    f"[qdrant] graph payload index skipped field={field} schema={schema}: {exc!r}",
                    file=sys.stderr,
                    flush=True,
                )
    return f"{'created' if created else 'exists'}:{collection}:vector_size={actual_size}"


def _validate_graph_type(graph_type: str) -> None:
    if graph_type not in ALL_GRAPH_VECTOR_TYPES:
        raise ValueError(f"unsupported graph vector type {graph_type!r}; expected one of {ALL_GRAPH_VECTOR_TYPES}")


def drop_graph_aliases(q: QdrantClient, source_collection: str, graph_types: Iterable[str]) -> list[str]:
    """Remove the aliases of certain graph collection types of this KB (the collections themselves are left
    for the GC to reclaim by retention period). Used when switching graph build modes: aliases left by
    the other path no longer represent the current graph."""
    aliases = {item.alias_name for item in q.get_aliases().aliases}
    targets = [graph_collection_alias(source_collection, t) for t in graph_types]
    dropped = [alias for alias in targets if alias in aliases]
    if dropped:
        q.update_collection_aliases([
            models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias)) for alias in dropped
        ])
    return dropped


def collection_vector_size(q: QdrantClient, collection: str) -> int | None:
    info = q.get_collection(collection_name=collection)
    return _extract_vector_size(info.config.params.vectors)


def validate_collection_vector_size(q: QdrantClient, collection: str, expected_size: int) -> int:
    actual_size = collection_vector_size(q, collection)
    if actual_size is None:
        raise RuntimeError(
            f"Qdrant collection {collection!r} vector size is unavailable; "
            f"expected EMBEDDING_DIM={expected_size}"
        )
    if actual_size != int(expected_size):
        raise RuntimeError(
            f"Qdrant collection {collection!r} vector size mismatch: "
            f"actual={actual_size}, EMBEDDING_DIM={expected_size}. "
            "Use the matching embedding model or reset/recreate this collection."
        )
    return actual_size


def _extract_vector_size(vectors_config: Any) -> int | None:
    size = getattr(vectors_config, "size", None)
    if size is not None:
        return int(size)
    if isinstance(vectors_config, dict):
        if "size" in vectors_config and vectors_config["size"] is not None:
            return int(vectors_config["size"])
        unnamed = vectors_config.get("") or vectors_config.get("default")
        if unnamed is not None:
            return _extract_vector_size(unnamed)
        if len(vectors_config) == 1:
            return _extract_vector_size(next(iter(vectors_config.values())))
        return None
    model_dump = getattr(vectors_config, "model_dump", None)
    if callable(model_dump):
        return _extract_vector_size(model_dump())
    to_dict = getattr(vectors_config, "dict", None)
    if callable(to_dict):
        return _extract_vector_size(to_dict())
    return None


def collection_exists(q: QdrantClient, collection: str) -> bool:
    # Client 1.16+ has a dedicated endpoint; the old way listed every collection on the server, and a single
    # parse job calls this three or four times.
    try:
        return bool(q.collection_exists(collection))
    except AttributeError:
        return collection in {item.name for item in q.get_collections().collections}


def delete_collection(q: QdrantClient, collection: str, *, dry_run: bool = False) -> str:
    if not collection_exists(q, collection):
        return f"missing:{collection}"
    if dry_run:
        return f"would-delete:{collection}"
    q.delete_collection(collection_name=collection)
    return f"deleted:{collection}"


def doc_filter(kb_id: str, file_key: int) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(
                key="doc_id",
                match=models.MatchValue(value=f"{kb_id}:{file_key}"),
            )
        ]
    )


def active_doc_filter(kb_id: str, file_key: int) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(
                key="doc_id",
                match=models.MatchValue(value=f"{kb_id}:{file_key}"),
            ),
            models.FieldCondition(key="is_active", match=models.MatchValue(value=True)),
        ]
    )


def active_doc_point_count(q: QdrantClient, collection: str, *, kb_id: str, file_key: int) -> int:
    if not collection_exists(q, collection):
        return 0
    return int(
        q.count(
            collection_name=collection,
            count_filter=active_doc_filter(kb_id, file_key),
            exact=True,
        ).count
    )


def doc_version_filter(kb_id: str, file_key: int, content_version: str) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(
                key="doc_id",
                match=models.MatchValue(value=f"{kb_id}:{file_key}"),
            ),
            models.FieldCondition(
                key="content_version",
                match=models.MatchValue(value=content_version),
            ),
        ]
    )


def metadata_payload_from_file_row(row) -> dict:
    payload = {
        "source_path": row["source_path"],
        "rel_path": row["rel_path"],
        "filename": row["filename"],
        "dir": row["dir"],
        "dir_ancestors": dir_ancestors_from_rel_path(str(row["rel_path"])),
        "file_size": int(row["size"]),
        "file_mtime": int(row["mtime"]),
    }
    return {k: v for k, v in payload.items() if v is not None}


def update_file_metadata(q: QdrantClient, row) -> None:
    if not collection_exists(q, row["collection"]):
        return
    q.set_payload(
        collection_name=row["collection"],
        payload=metadata_payload_from_file_row(row),
        points=doc_filter(str(row["kb_id"]), int(row["file_key"])),
    )


def reactivate_file_metadata(q: QdrantClient, row, *, point_ids: Iterable[str] | None = None) -> None:
    """A file that returned after deletion: flip its points back to active and refresh the path fields.
    With point_ids given, only those points are flipped (the batch marked deleted in SQLite -- active at
    the moment of KB close / deletion); without it, the whole "file + version" batch is flipped, which
    also revives old chunks replaced within the same version, so that is left to legacy callers without
    a ledger."""
    if not collection_exists(q, row["collection"]):
        return
    if point_ids is not None:
        ids = [str(p) for p in point_ids]
        if not ids:
            return
        selector: Any = ids
    else:
        selector = doc_version_filter(str(row["kb_id"]), int(row["file_key"]), str(row["content_version"]))
    payload = metadata_payload_from_file_row(row)
    payload["is_active"] = True
    q.set_payload(
        collection_name=row["collection"],
        payload=payload,
        points=selector,
    )
    q.delete_payload(
        collection_name=row["collection"],
        keys=["inactive_at"],
        points=selector,
        wait=True,
    )


def file_version_point_count(q: QdrantClient, row) -> int:
    """How many points of this file's current version remain in Qdrant. A file that returned after deletion
    uses it to decide whether it can be restored directly: inactive points past the retention period are
    cleared by the GC, and then only a re-parse remains."""
    if not collection_exists(q, row["collection"]):
        return 0
    selector = doc_version_filter(str(row["kb_id"]), int(row["file_key"]), str(row["content_version"]))
    return int(q.count(collection_name=row["collection"], count_filter=selector, exact=True).count)


def mark_file_inactive(q: QdrantClient, row) -> None:
    if not collection_exists(q, row["collection"]):
        return
    q.set_payload(
        collection_name=row["collection"],
        payload={"is_active": False, "inactive_at": int(time.time())},
        points=active_doc_filter(str(row["kb_id"]), int(row["file_key"])),
    )


def mark_stale_file_points_inactive(
    q: QdrantClient,
    collection: str,
    *,
    kb_id: str,
    file_key: int,
    active_chunk_uids: set[str],
    batch_size: int = 256,
) -> int:
    stale_point_ids: list[str] = []
    offset = None
    while True:
        records, offset = q.scroll(
            collection_name=collection,
            scroll_filter=doc_filter(kb_id, file_key),
            limit=batch_size,
            offset=offset,
            with_payload=["chunk_uid", "is_active"],
            with_vectors=False,
        )
        for record in records:
            payload = record.payload or {}
            if payload.get("is_active") is False:
                continue
            chunk_uid = payload.get("chunk_uid")
            if not chunk_uid or str(chunk_uid) not in active_chunk_uids:
                stale_point_ids.append(str(record.id))
        if offset is None:
            break

    for start in range(0, len(stale_point_ids), batch_size):
        q.set_payload(
            collection_name=collection,
            payload={"is_active": False, "inactive_at": int(time.time())},
            points=stale_point_ids[start : start + batch_size],
        )
    return len(stale_point_ids)


def upsert_chunks(
    q: QdrantClient,
    collection: str,
    point_ids: list[str],
    vectors: list[dict[str, list[float]]],
    payloads: list[dict[str, Any]],
    *,
    max_bytes: int = 30 * 1024 * 1024,
) -> None:
    """`vectors[i]` maps vector name -> values for point i, e.g. {"text": [...]}
    or {"text": [...], "visual": [...]}; a point carries only the named
    vectors it has."""
    batch: list[models.PointStruct] = []
    batch_bytes = 0

    def flush() -> None:
        nonlocal batch, batch_bytes
        if not batch:
            return
        q.upsert(collection_name=collection, points=batch, wait=True)
        batch = []
        batch_bytes = 0

    for point_id, vector, payload in zip(point_ids, vectors, payloads, strict=True):
        if not isinstance(vector, dict) or not vector:
            raise ValueError(f"point {point_id} must carry a non-empty mapping of named vectors")
        if TEXT_VECTOR not in vector:
            raise ValueError(f"point {point_id} is missing the {TEXT_VECTOR!r} vector")
        point = models.PointStruct(id=point_id, vector=vector, payload=payload)
        payload_bytes = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        vector_bytes = sum(len(values) for values in vector.values()) * 24
        point_bytes = len(str(point_id).encode("utf-8")) + payload_bytes + vector_bytes + 128
        if max_bytes > 0 and point_bytes > max_bytes:
            raise ValueError(f"single Qdrant point is too large: {point_bytes} bytes")
        if max_bytes > 0 and batch and batch_bytes + point_bytes > max_bytes:
            flush()
        batch.append(point)
        batch_bytes += point_bytes
    flush()


def mark_old_versions_inactive(
    q: QdrantClient,
    collection: str,
    *,
    kb_id: str,
    file_key: int,
    active_content_version: str,
) -> None:
    q.set_payload(
        collection_name=collection,
        payload={"is_active": False, "inactive_at": int(time.time())},
        points=models.Filter(
            must=[
                models.FieldCondition(
                    key="doc_id",
                    match=models.MatchValue(value=f"{kb_id}:{file_key}"),
                ),
                models.FieldCondition(key="is_active", match=models.MatchValue(value=True)),
            ],
            must_not=[
                models.FieldCondition(
                    key="content_version",
                    match=models.MatchValue(value=active_content_version),
                )
            ],
        ),
    )


def inactive_older_than_filter(cutoff_ts: int) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(key="is_active", match=models.MatchValue(value=False)),
            models.FieldCondition(key="inactive_at", range=models.Range(lt=float(cutoff_ts))),
        ]
    )


def inactive_doc_versions_older_than(
    q: QdrantClient,
    collection: str,
    cutoff_ts: int,
    *,
    batch_size: int = 512,
) -> list[dict[str, Any]]:
    if not collection_exists(q, collection):
        return []

    groups: dict[tuple[str, str], dict[str, Any]] = {}
    offset = None
    while True:
        records, offset = q.scroll(
            collection_name=collection,
            scroll_filter=inactive_older_than_filter(cutoff_ts),
            limit=batch_size,
            offset=offset,
            with_payload=["doc_id", "kb_id", "content_version", "inactive_at"],
            with_vectors=False,
        )
        for record in records:
            payload = record.payload or {}
            doc_id = str(payload.get("doc_id") or "")
            content_version = str(payload.get("content_version") or "")
            if not doc_id or not content_version:
                continue
            key = (doc_id, content_version)
            group = groups.setdefault(
                key,
                {
                    "collection": collection,
                    "doc_id": doc_id,
                    "kb_id": str(payload.get("kb_id") or doc_id.rsplit(":", 1)[0]),
                    "content_version": content_version,
                    # The payload has no sha256 key (the parse side writes visual_sha256 and
                    # content_version); keeping one would only make maintainers think it has a value.
                    "sha256": str(payload.get("visual_sha256") or ""),
                    "inactive_at": None,
                    "points": 0,
                },
            )
            group["points"] += 1
            inactive_at = payload.get("inactive_at")
            if inactive_at is not None:
                try:
                    inactive_at_int = int(float(inactive_at))
                except (TypeError, ValueError):
                    inactive_at_int = None
                if inactive_at_int is not None:
                    previous = group["inactive_at"]
                    group["inactive_at"] = inactive_at_int if previous is None else min(int(previous), inactive_at_int)
        if offset is None:
            break
    return sorted(groups.values(), key=lambda item: (str(item["collection"]), str(item["doc_id"]), str(item["content_version"])))


def delete_inactive_doc_version_older_than(
    q: QdrantClient,
    collection: str,
    *,
    doc_id: str,
    content_version: str,
    cutoff_ts: int,
    dry_run: bool = False,
) -> int:
    if not collection_exists(q, collection):
        return 0
    query_filter = models.Filter(
        must=[
            models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
            models.FieldCondition(key="content_version", match=models.MatchValue(value=content_version)),
            models.FieldCondition(key="is_active", match=models.MatchValue(value=False)),
            models.FieldCondition(key="inactive_at", range=models.Range(lt=float(cutoff_ts))),
        ]
    )
    count = int(q.count(collection_name=collection, count_filter=query_filter, exact=True).count)
    if dry_run or count == 0:
        return count
    q.delete(
        collection_name=collection,
        points_selector=models.FilterSelector(filter=query_filter),
        wait=True,
    )
    return count


def delete_malformed_inactive_points(
    q: QdrantClient,
    collection: str,
    cutoff_ts: int,
    *,
    dry_run: bool = False,
) -> int:
    """Expired inactive points whose payload lost doc_id or content_version:
    the per-version GC groups by exactly those fields, so it can never select
    them, and they would otherwise accumulate forever."""
    if not collection_exists(q, collection):
        return 0
    total = 0
    for key in ("doc_id", "content_version"):
        query_filter = models.Filter(
            must=[
                models.FieldCondition(key="is_active", match=models.MatchValue(value=False)),
                models.FieldCondition(key="inactive_at", range=models.Range(lt=float(cutoff_ts))),
                models.IsEmptyCondition(is_empty=models.PayloadField(key=key)),
            ]
        )
        count = int(q.count(collection_name=collection, count_filter=query_filter, exact=True).count)
        if count and not dry_run:
            q.delete(collection_name=collection, points_selector=models.FilterSelector(filter=query_filter), wait=True)
        # a point missing both fields is counted once per pass but the second
        # delete is a no-op; the number is a report, not an invariant
        total += count
    return total


def delete_inactive_points_older_than(
    q: QdrantClient,
    collection: str,
    cutoff_ts: int,
    *,
    dry_run: bool = False,
) -> int:
    if not collection_exists(q, collection):
        return 0
    query_filter = inactive_older_than_filter(cutoff_ts)
    count = int(q.count(collection_name=collection, count_filter=query_filter, exact=True).count)
    if dry_run or count == 0:
        return count
    q.delete(
        collection_name=collection,
        points_selector=models.FilterSelector(filter=query_filter),
        wait=True,
    )
    return count


def backfill_inactive_at(
    q: QdrantClient,
    collection: str,
    inactive_at_ts: int,
    *,
    dry_run: bool = False,
    batch_size: int = 256,
) -> int:
    if not collection_exists(q, collection):
        return 0

    query_filter = models.Filter(
        must=[models.FieldCondition(key="is_active", match=models.MatchValue(value=False))]
    )
    point_ids: list[str] = []
    total = 0
    offset = None

    def flush() -> None:
        nonlocal point_ids
        if dry_run or not point_ids:
            point_ids = []
            return
        q.set_payload(
            collection_name=collection,
            payload={"inactive_at": inactive_at_ts},
            points=point_ids,
        )
        point_ids = []

    while True:
        records, offset = q.scroll(
            collection_name=collection,
            scroll_filter=query_filter,
            limit=batch_size,
            offset=offset,
            with_payload=["inactive_at", "is_active"],
            with_vectors=False,
        )
        for record in records:
            payload = record.payload or {}
            if payload.get("inactive_at") is None:
                total += 1
                point_ids.append(str(record.id))
                if len(point_ids) >= batch_size:
                    flush()
        if offset is None:
            break
    flush()
    return total
