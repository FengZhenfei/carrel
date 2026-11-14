"""Standard adapters. Existing algorithms remain the implementation of this backend.

Database-specific filters, pooled connections and driver types are kept here or
in the legacy implementation modules called here. Instantiation is lazy: a text
parse must not open Neo4j, and an optional keyword failure must not stop ingestion.
"""
from __future__ import annotations

from typing import Any
import threading

from ..vector import qdrant


def vector_client(url: str, api_key: str | None, *, timeout: float | None = None):
    if timeout is None:
        return qdrant.client(url, api_key)
    from qdrant_client import QdrantClient
    return QdrantClient(url=url, api_key=api_key, timeout=timeout, check_compatibility=False)


def _filters(hints=None, excluded=None):
    from kb_search import channels
    from qdrant_client import models
    vector = channels.hint_filter(hints) if hints else None
    keyword = channels.hint_os_filters(hints) if hints else None
    if excluded:
        base = vector or channels.ACTIVE_FILTER
        vector = models.Filter(must=list(base.must or []), must_not=[
            models.FieldCondition(key="rel_path", match=models.MatchAny(any=list(excluded)))])
        keyword = list(keyword or []) + [{"bool": {"must_not": [{"terms": {"rel_path": list(excluded)}}]}}]
    return vector, keyword


class StandardStorage:
    def __init__(self, settings, *, vector_client=None, query_timeout=None):
        self.settings = settings
        self._client = vector_client
        self._client_lock = threading.Lock()
        self._query_timeout = query_timeout
        self._client_key = (getattr(settings, "qdrant_url", None), getattr(settings, "qdrant_api_key", None), query_timeout)
        self.chunks = StandardChunks(self)
        self.keywords = StandardKeywords(self)
        self.graph = StandardGraph(self)

    @property
    def _q(self):
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    self._client = vector_client(self.settings.qdrant_url, self.settings.qdrant_api_key,
                                                 timeout=self._query_timeout)
        return self._client

    def catalog(self, *, ttl, force=False):
        from kb_search.catalog import get_catalog
        return get_catalog(self.settings, self._q, ttl=ttl, force=force, driver=self.graph.optional_driver())

    def with_deadline(self, deadline):
        from ..graph.recall import _DeadlineClient
        from .publication import guard
        return guard(StandardStorage(self.settings, vector_client=_DeadlineClient(self._q, deadline),
                                     query_timeout=self._query_timeout))


class StandardChunks:
    def __init__(self, storage):
        self._storage = storage

    @property
    def _q(self):
        return self._storage._q

    def exists(self, collection):
        return qdrant.collection_exists(self._q, collection)

    def ensure(self, collection, layout, *, dry_run=False):
        return qdrant.ensure_collection(self._q, collection, layout, dry_run=dry_run)

    def validate(self, collection, layout):
        return qdrant.validate_collection_layout(self._q, collection, layout)

    def write(self, collection, point_ids, vectors, payloads, *, max_bytes):
        return qdrant.upsert_chunks(self._q, collection, point_ids, vectors, payloads, max_bytes=max_bytes)

    def retire_previous(self, collection, **options):
        return qdrant.mark_old_versions_inactive(self._q, collection, **options)

    def retire_stale(self, collection, **options):
        return qdrant.mark_stale_file_points_inactive(self._q, collection, **options)

    def update_metadata(self, row):
        return qdrant.update_file_metadata(self._q, row)

    def reactivate(self, row, *, point_ids=None):
        return qdrant.reactivate_file_metadata(self._q, row, point_ids=point_ids)

    def version_count(self, row):
        return qdrant.file_version_point_count(self._q, row)

    def deactivate(self, row):
        return qdrant.mark_file_inactive(self._q, row)

    def payloads(self, collection, point_ids, *, timeout=None, active_only=True):
        from kb_search.channels import fetch_payloads, _secs
        if active_only:
            return fetch_payloads(self._q, collection, point_ids, timeout=timeout)
        # Console previews follow the SQLite ledger, including its previous version
        # while a reparse is publishing; ordinary retrieval only sees active points.
        found = {}
        ids = list(dict.fromkeys(str(pid) for pid in point_ids if pid))
        for start in range(0, len(ids), 200):
            for row in self._q.retrieve(collection_name=collection, ids=ids[start:start + 200],
                                        with_payload=True, with_vectors=False, timeout=_secs(timeout)):
                found[str(row.id)] = dict(row.payload or {})
        return found

    def image_payload(self, collection, point_id):
        from kb_search.images import locate
        return locate(self._q, collection, point_id)

    def vectors(self, collection, point_ids):
        from ..vector.layout import TEXT_VECTOR
        found = {}
        for start in range(0, len(point_ids), 128):
            for point in self._q.retrieve(collection_name=collection, ids=point_ids[start:start + 128],
                                          with_vectors=[TEXT_VECTOR], with_payload=False):
                vector = point.vector.get(TEXT_VECTOR) if isinstance(point.vector, dict) else point.vector
                if vector:
                    found[str(point.id)] = list(vector)
        return found

    def active_count(self, collection):
        from kb_search.channels import ACTIVE_FILTER
        return int(self._q.count(collection_name=collection, count_filter=ACTIVE_FILTER, exact=True).count)

    def vector_candidates(self, collection, vector, *, limit, hints=None, excluded=None, visual=False, timeout=None):
        from kb_search import channels
        fn = channels.visual_channel if visual else channels.vector_channel
        return fn(self._q, collection, vector, limit=limit, timeout=timeout,
                  query_filter=_filters(hints, excluded)[0])

    def context(self, collection, **options):
        from kb_search.channels import context_range
        return context_range(self._q, collection, **options)

    def table_head(self, collection, payload, *, timeout=None):
        from kb_search.channels import table_head
        return table_head(self._q, collection, payload, timeout=timeout)

    def neighbors(self, collection, payload, *, span, timeout=None):
        from kb_search.channels import neighbor_payloads
        return neighbor_payloads(self._q, collection, payload, span=span, timeout=timeout)

    def metadata(self, collection, point_ids, *, timeout=None):
        from kb_search.channels import point_meta
        return point_meta(self._q, collection, point_ids, timeout=timeout)

    def check(self):
        self._q.get_collections()


class StandardKeywords:
    def __init__(self, storage):
        self._storage = storage

    @property
    def _url(self):
        return self._storage.settings.opensearch_url

    def ensure(self, collections):
        from .. import search_fts
        return search_fts.ensure_indices(self._url, collections)

    def sync_document(self, collection, doc_id):
        from .. import search_fts
        return search_fts.sync_doc_from_qdrant(url=self._url, qdrant=self._storage._q,
                                             collection=collection, doc_id=doc_id)

    def delete(self, collection):
        from .. import search_fts
        return search_fts.delete_collection(self._url, collection=collection)

    def candidates(self, question, collection, *, limit, identifiers=None, hints=None, excluded=None, timeout=None):
        from kb_search.channels import bm25_channel
        return bm25_channel(self._url, question, collection, limit=limit, identifiers=identifiers,
                            timeout=timeout, filters=_filters(hints, excluded)[1])

    def profile(self, question, collections, *, timeout=None):
        from kb_search.channels import lexical_profile
        return lexical_profile(self._url, question, collections, timeout=timeout)

    def grep(self, source, phrases, **options):
        from kb_search import channels, library
        return library.grep(self._storage.settings, source, phrases,
                            client=channels.os_client(self._url), **options)


class StandardGraph:
    def __init__(self, storage):
        self._storage = storage

    @property
    def _settings(self):
        return self._storage.settings

    def corpus_payloads(self, collection, refs):
        from ..graph.units import fetch_chunk_payloads
        return fetch_chunk_payloads(self._storage._q, collection, refs)

    def write_vectors(self, embed, **options):
        from ..graph.vectors import write_graph_vectors
        return write_graph_vectors(self._storage._q, embed, **options)

    def import_version(self, **options):
        from ..graph.neo4j_import import import_graph_to_neo4j
        return import_graph_to_neo4j(self._settings, **options)

    def activate_vectors(self, **options):
        return qdrant.activate_graph_aliases(self._storage._q, **options)

    def restore_vectors(self, previous):
        return qdrant.restore_graph_aliases(self._storage._q, previous)

    def activate(self, **options):
        from ..graph.neo4j_import import activate_neo4j_graph_version
        return activate_neo4j_graph_version(self._settings, **options)

    def query(self, source, question, **options):
        from kb_search.channels import graph_channel
        return graph_channel(self._settings, source, question, q=self._storage._q, **options)

    def neighbors(self, source, **options):
        from kb_search import channels, graphwalk
        return graphwalk.neighbors(self._settings, source, q=self._storage._q,
                                   driver=channels.shared_driver(self._settings), store=self._storage.chunks, **options)

    def entities(self, source, **options):
        from kb_search import channels, graphwalk
        return graphwalk.list_entities(self._settings, source,
                                       driver=channels.shared_driver(self._settings), **options)

    def facts(self, source, **options):
        from kb_search import channels, graphwalk
        return graphwalk.list_facts(self._settings, source, q=self._storage._q,
                                    driver=channels.shared_driver(self._settings), store=self._storage.chunks, **options)

    def pages(self, source, **options):
        from kb_search.graphwalk import list_pages
        return list_pages(self._settings, source, q=self._storage._q, **options)

    def neighborhoods(self, sources, versions, question, seeds, **options):
        from kb_search import channels, graphwalk
        return graphwalk.neighborhoods(self._settings, sources, versions, question, seeds,
                                       driver=channels.shared_driver(self._settings), **options)

    def locate_facts(self, collections, specs, sources, *, timeout=None):
        from kb_search.graphwalk import locate_facts
        return locate_facts(self._storage._q, collections, specs, sources, timeout=timeout, store=self._storage.chunks)

    def stats(self, kb_id, *, timeout=None):
        from kb_search import channels, graphwalk
        return graphwalk.version_stats(channels.shared_driver(self._settings), kb_id, timeout=timeout)


    def optional_driver(self):
        from kb_search.channels import shared_driver
        try:
            return shared_driver(self._settings)
        except Exception:
            return None

    def document_dates(self, sources, versions, *, timeout=None):
        from kb_search.docstatus import document_dates
        return document_dates(self._settings, sources, versions,
                              driver=self.optional_driver() if versions else None, timeout=timeout)

    def annotate_dates(self, sources, scope, *, timeout=None):
        from kb_search.docstatus import annotate_dates
        return annotate_dates(self._settings, sources, scope, driver=self.optional_driver(), timeout=timeout)

    def read_originals(self, search_settings, result, scope, hint, until):
        from kb_search.readback import read_originals
        return read_originals(self._settings, search_settings, self._storage._q, result, scope, hint, until,
                              self.optional_driver(), store=self._storage)


    def check(self):
        from ..graph.neo4j_import import neo4j_driver
        driver = neo4j_driver(self._settings)
        try:
            driver.verify_connectivity()
        finally:
            driver.close()

    def vector_collections(self):
        return {item.name for item in self._storage._q.get_collections().collections}

    def vector_aliases(self):
        return {item.alias_name: item.collection_name for item in self._storage._q.get_aliases().aliases}

    def alias_targets(self, collection):
        return qdrant.graph_alias_targets(self._storage._q, collection)

    def vector_exists(self, collection):
        return qdrant.collection_exists(self._storage._q, collection)

    def vector_count(self, collection):
        return int(self._storage._q.count(collection_name=collection, exact=True).count)

    def drop_aliases(self, collection, types):
        return qdrant.drop_graph_aliases(self._storage._q, collection, types)

    def collect_vectors(self, **options):
        return qdrant.delete_old_graph_collections(self._storage._q, **options)

    def collect_versions(self, **options):
        from ..graph.neo4j_import import delete_old_neo4j_graph_versions
        return delete_old_neo4j_graph_versions(self._settings, **options)

    def projection_matches(self, kb_id, version, expected):
        from ..graph.neo4j_import import count_version_nodes, counts_match, neo4j_counts, neo4j_driver
        driver = neo4j_driver(self._settings)
        try:
            nodes = count_version_nodes(driver, kb_id, version)
            return bool(nodes > 0 and expected and counts_match(expected, neo4j_counts(driver, kb_id, version)))
        finally:
            driver.close()

    def deactivate(self, source):
        from ..graph.neo4j_import import deactivate_neo4j_graph_version
        return deactivate_neo4j_graph_version(self._settings, source=source)
