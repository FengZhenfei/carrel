"""Keyword index maintained in OpenSearch.

This module keeps the name it had when the index lived in SQLite FTS5 so the
call sites (worker, parse job, CLI, reset) stay put. The contract is
unchanged: the keyword index is a projection of Qdrant active points, synced
per document before a job is marked done, and rebuildable from Qdrant at any
time. What changed is the backend: one OpenSearch index per collection,
documents keyed by Qdrant point_id so every write is an idempotent upsert.

Analysis relies on the built-in ``cjk`` analyzer (overlapping CJK bigrams), so
two-character Chinese terms match natively -- the gap that motivated leaving
trigram FTS5. ASCII identifiers survive as whole lowercase tokens. Substring
matching inside identifiers is covered by an ngram subfield on filename and
path only; ngramming the body would balloon the index for little gain.
"""

from __future__ import annotations

import json
import time
from typing import Any

from opensearchpy import OpenSearch, helpers
from qdrant_client import QdrantClient
from qdrant_client.http import models


PAYLOAD_FIELDS = [
    "kb_id",
    "doc_id",
    "source_path",
    "rel_path",
    "filename",
    "dir",
    "content_version",
    "chunk_uid",
    "chunk_index",
    "text",
    "doc_type",
    "block_type",
    "page_idx",
    "slide_idx",
    "sheet_name",
    "row_start",
    "row_end",
    "title",
    "caption",
    "section_path",
    "visual_summary",
    "visual_entities",
    "visual_facts",
    "visual_keywords",
    "visual_text",
    "is_active",
]

# Field boosts mirror the old bm25(fts, 4.0, 2.0, 2.0, 1.0, 1.0) weights.
SEARCH_FIELDS = ["filename^4", "path^2", "title^2", "body", "visual"]

INDEX_BODY: dict[str, Any] = {
    "settings": {
        "index": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "max_ngram_diff": 2,
        },
        "analysis": {
            "tokenizer": {
                "kb_ngram_tokenizer": {
                    "type": "ngram",
                    "min_gram": 3,
                    "max_gram": 4,
                    "token_chars": ["letter", "digit", "punctuation"],
                }
            },
            "analyzer": {
                "kb_ngram": {
                    "type": "custom",
                    "tokenizer": "kb_ngram_tokenizer",
                    "filter": ["lowercase"],
                }
            },
        },
    },
    "mappings": {
        "dynamic": "strict",
        "properties": {
            # analyzed search fields (composition mirrors the old FTS columns)
            "filename": {
                "type": "text",
                "analyzer": "cjk",
                "fields": {
                    "keyword": {"type": "keyword"},
                    "ngram": {"type": "text", "analyzer": "kb_ngram"},
                },
            },
            "path": {
                "type": "text",
                "analyzer": "cjk",
                "fields": {"ngram": {"type": "text", "analyzer": "kb_ngram"}},
            },
            "title": {"type": "text", "analyzer": "cjk"},
            "body": {"type": "text", "analyzer": "cjk"},
            "visual": {"type": "text", "analyzer": "cjk"},
            # exact-match metadata (drives metadata routing and doc addressing)
            "chunk_uid": {"type": "keyword"},
            "kb_id": {"type": "keyword"},
            "doc_id": {"type": "keyword"},
            "content_version": {"type": "keyword"},
            "chunk_index": {"type": "integer"},
            "source_path": {"type": "keyword"},
            "rel_path": {"type": "keyword"},
            "dir": {"type": "keyword"},
            "doc_type": {"type": "keyword"},
            "block_type": {"type": "keyword"},
            "section_path": {"type": "keyword"},
            "page_idx": {"type": "integer"},
            "slide_idx": {"type": "integer"},
            "sheet_name": {"type": "keyword"},
            "row_start": {"type": "integer"},
            "row_end": {"type": "integer"},
            "updated_at": {"type": "long"},
        },
    },
}


_CLIENTS: dict[str, OpenSearch] = {}


def client(url: str) -> OpenSearch:
    cached = _CLIENTS.get(url)
    if cached is None:
        cached = OpenSearch(hosts=[url], timeout=30, max_retries=2, retry_on_timeout=True)
        _CLIENTS[url] = cached
    return cached


def ensure_index(os_client: OpenSearch, collection: str) -> bool:
    if os_client.indices.exists(index=collection):
        return False
    os_client.indices.create(index=collection, body=INDEX_BODY)
    return True


def doc_source(payload: dict[str, Any]) -> dict[str, Any]:
    """Build the index document for one Qdrant point payload."""
    fields = _search_fields(payload)
    source: dict[str, Any] = {
        "filename": fields["filename"],
        "path": fields["path"],
        "title": fields["title"],
        "body": fields["body"],
        "visual": fields["visual"],
        "chunk_uid": _text(payload.get("chunk_uid")),
        "kb_id": _text(payload.get("kb_id")),
        "doc_id": _text(payload.get("doc_id")),
        "content_version": _text(payload.get("content_version")),
        "chunk_index": _int_or_none(payload.get("chunk_index")),
        "source_path": _text(payload.get("source_path")),
        "rel_path": _text(payload.get("rel_path")),
        "dir": _text(payload.get("dir")),
        "doc_type": _text(payload.get("doc_type")),
        "block_type": _text(payload.get("block_type")),
        "section_path": [str(x) for x in (payload.get("section_path") or []) if str(x).strip()],
        "page_idx": _int_or_none(payload.get("page_idx")),
        "slide_idx": _int_or_none(payload.get("slide_idx")),
        "sheet_name": _text(payload.get("sheet_name")),
        "row_start": _int_or_none(payload.get("row_start")),
        "row_end": _int_or_none(payload.get("row_end")),
        "updated_at": int(time.time()),
    }
    return {key: value for key, value in source.items() if value not in (None, "")}


def _bulk_index(os_client: OpenSearch, actions: list[dict[str, Any]]) -> int:
    if not actions:
        return 0
    success, _ = helpers.bulk(os_client, actions, raise_on_error=True, request_timeout=120)
    return int(success)


def _scroll_active(
    qdrant: QdrantClient,
    collection: str,
    *,
    doc_id: str | None = None,
    batch_size: int = 512,
):
    must: list[Any] = [models.FieldCondition(key="is_active", match=models.MatchValue(value=True))]
    if doc_id is not None:
        must.append(models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)))
    offset = None
    while True:
        records, offset = qdrant.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(must=must),
            limit=batch_size,
            offset=offset,
            with_payload=PAYLOAD_FIELDS,
            with_vectors=False,
        )
        yield from records
        if offset is None:
            break


def sync_doc_from_qdrant(
    *,
    url: str,
    qdrant: QdrantClient,
    collection: str,
    doc_id: str,
    batch_size: int = 512,
) -> dict[str, Any]:
    """Projection sync for one document: drop its rows, refill from Qdrant."""
    started = time.time()
    os_client = client(url)
    collection_present = _collection_exists(qdrant, collection)
    if not collection_present and not os_client.indices.exists(index=collection):
        # The whole collection is gone (kb_sources GC won the race with this
        # job); creating a fresh empty index here would resurrect an orphan
        # that nothing ever deletes.
        return {
            # the worker prints result["collection"]; without this key the "skipped" branch raises KeyError
            "collection": collection,
            "doc_id": doc_id,
            "deleted_rows": 0,
            "inserted_rows": 0,
            "skipped": "collection and index both absent",
            "elapsed_ms": int((time.time() - started) * 1000),
        }
    ensure_index(os_client, collection)
    # No separate refresh after deleting the old rows: one refresh after the bulk write below is enough; the
    # window in which old and new versions are both visible exists before that refresh anyway, so one
    # refresh fewer changes no semantics and halves the refreshes per document (health check D6)
    deleted = os_client.delete_by_query(
        index=collection,
        body={"query": {"term": {"doc_id": doc_id}}},
        params={"refresh": "false", "conflicts": "proceed"},
    ).get("deleted", 0)

    actions: list[dict[str, Any]] = []
    if collection_present:
        actions = [
            {
                "_op_type": "index",
                "_index": collection,
                "_id": str(record.id),
                "_source": doc_source(record.payload or {}),
            }
            for record in _scroll_active(qdrant, collection, doc_id=doc_id, batch_size=batch_size)
        ]
    inserted = _bulk_index(os_client, actions)
    os_client.indices.refresh(index=collection)
    return {
        "collection": collection,
        "doc_id": doc_id,
        "deleted_rows": int(deleted),
        "inserted_rows": inserted,
        "elapsed_seconds": round(time.time() - started, 3),
    }


def rebuild_from_qdrant(
    *,
    url: str,
    qdrant: QdrantClient,
    collections: list[str],
    batch_size: int = 512,
) -> dict[str, Any]:
    started = time.time()
    os_client = client(url)
    per_collection: dict[str, int] = {}
    for collection in collections:
        if os_client.indices.exists(index=collection):
            os_client.indices.delete(index=collection)
        ensure_index(os_client, collection)
        if not _collection_exists(qdrant, collection):
            per_collection[collection] = 0
            continue
        actions = [
            {
                "_op_type": "index",
                "_index": collection,
                "_id": str(record.id),
                "_source": doc_source(record.payload or {}),
            }
            for record in _scroll_active(qdrant, collection, batch_size=batch_size)
        ]
        per_collection[collection] = _bulk_index(os_client, actions)
        os_client.indices.refresh(index=collection)
    return {
        "url": url,
        "rebuilt": per_collection,
        "total_rows": sum(per_collection.values()),
        "elapsed_seconds": round(time.time() - started, 3),
    }


def ensure_indices(url: str, collections: list[str]) -> dict[str, Any]:
    os_client = client(url)
    return {collection: ("created" if ensure_index(os_client, collection) else "exists") for collection in collections}


def status(url: str, collections: list[str] | None = None) -> dict[str, Any]:
    os_client = client(url)
    indices: dict[str, Any] = {}
    total = 0
    names = collections or sorted(os_client.indices.get_alias(index="kb_*").keys())
    for name in names:
        if not os_client.indices.exists(index=name):
            indices[name] = {"exists": False}
            continue
        count = int(os_client.count(index=name).get("count", 0))
        stats = os_client.indices.stats(index=name, metric="store")
        size = int(stats["indices"][name]["total"]["store"]["size_in_bytes"])
        indices[name] = {"exists": True, "docs": count, "size_bytes": size}
        total += count
    return {"url": url, "indices": indices, "total_rows": total}


def compare_with_qdrant(url: str, qdrant: QdrantClient, collections: list[str]) -> dict[str, Any]:
    os_client = client(url)
    by_collection: dict[str, Any] = {}
    matches = True
    for collection in collections:
        qdrant_active = 0
        if _collection_exists(qdrant, collection):
            qdrant_active = int(
                qdrant.count(
                    collection_name=collection,
                    count_filter=models.Filter(
                        must=[models.FieldCondition(key="is_active", match=models.MatchValue(value=True))]
                    ),
                    exact=True,
                ).count
            )
        index_docs = int(os_client.count(index=collection).get("count", 0)) if os_client.indices.exists(index=collection) else 0
        diff = qdrant_active - index_docs
        matches = matches and diff == 0
        by_collection[collection] = {
            "qdrant_active": qdrant_active,
            "index_docs": index_docs,
            "diff": diff,
        }
    return {"matches": matches, "by_collection": by_collection}


def delete_collection(url: str, *, collection: str) -> dict[str, Any]:
    os_client = client(url)
    if not os_client.indices.exists(index=collection):
        return {"url": url, "collection": collection, "deleted_rows": 0, "exists": False}
    count = int(os_client.count(index=collection).get("count", 0))
    os_client.indices.delete(index=collection)
    return {"url": url, "collection": collection, "deleted_rows": count, "exists": True}


def search(
    url: str,
    query: str,
    *,
    collections: list[str],
    limit: int = 20,
) -> list[dict[str, Any]]:
    os_client = client(url)
    live = [c for c in collections if os_client.indices.exists(index=c)]
    if not live:
        return []
    body = {
        "size": limit,
        "query": {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": SEARCH_FIELDS,
                            "type": "most_fields",
                        }
                    },
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["filename.ngram^2", "path.ngram"],
                        }
                    },
                ],
                "minimum_should_match": 1,
            }
        },
        "_source": [
            "chunk_uid", "kb_id", "doc_id", "content_version", "chunk_index",
            "source_path", "rel_path", "filename", "dir", "doc_type",
            "block_type", "page_idx", "slide_idx", "sheet_name",
            "row_start", "row_end",
        ],
    }
    response = os_client.search(index=",".join(live), body=body)
    rows: list[dict[str, Any]] = []
    for hit in response.get("hits", {}).get("hits", []):
        row = dict(hit.get("_source") or {})
        row["collection"] = hit.get("_index")
        row["point_id"] = hit.get("_id")
        row["score"] = hit.get("_score")
        rows.append(row)
    return rows


def _collection_exists(qdrant: QdrantClient, collection: str) -> bool:
    """Deliberately does not swallow exceptions: Qdrant being temporarily unreachable and "collection does
    not exist" are two different things. This used to return False in both cases, so sync_doc first deleted
    every FTS row of the document, then skipped the refill because of "does not exist", and the job still
    finished normally -- keyword search silently lost documents. Let the exception propagate; the caller
    retries anyway."""
    names = {c.name for c in qdrant.get_collections().collections}
    return collection in names


def _search_fields(payload: dict[str, Any]) -> dict[str, str]:
    filename = _text(payload.get("filename"))
    path = "\n".join(
        item
        for item in (
            _text(payload.get("source_path")),
            _text(payload.get("rel_path")),
            _text(payload.get("dir")),
        )
        if item
    )
    title = "\n".join(
        item
        for item in (
            _text(payload.get("title")),
            _text(payload.get("caption")),
            " / ".join(str(x) for x in (payload.get("section_path") or []) if str(x).strip()),
        )
        if item
    )
    body = _text(payload.get("text"))
    visual = "\n".join(
        item
        for item in (
            _text(payload.get("visual_summary")),
            _text(payload.get("visual_text")),
            _jsonish(payload.get("visual_entities")),
            _jsonish(payload.get("visual_facts")),
            _jsonish(payload.get("visual_keywords")),
        )
        if item
    )
    return {"filename": filename, "path": path, "title": title, "body": body, "visual": visual}


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _jsonish(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
