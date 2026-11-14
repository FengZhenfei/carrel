"""Light application adapter: Lance data, DuckDB graph joins, SQLite publication."""
from contextlib import contextmanager
from copy import copy
import fcntl
import json
import time
from pathlib import Path

from .light_data import LightChunks, LightKeywords
from .light_graph import LightGraphProjection
from .light_vectors import LightVectors
from .light_reader import LightReader
from ..vector.qdrant import GRAPH_VECTOR_TYPES, GRAPH_OPTIONAL_TYPES, graph_collection_name, graph_collection_alias, parse_graph_collection_name


class LightStorage:
    def __init__(self, settings):
        self.settings = settings
        self.chunks = LightChunks(getattr(settings, 'storage_root', None) or settings.runtime_dir / 'storage')
        self.keywords = LightKeywords(self.chunks, settings=settings)
        self.graph = LightGraph(self)
        self.deadline = None

    def with_deadline(self, deadline):
        scoped = copy(self)
        scoped.deadline = deadline
        from .publication import PublishedChunks, guard
        scoped.chunks = copy(self.chunks.raw if isinstance(self.chunks, PublishedChunks) else self.chunks)
        scoped.chunks.deadline = deadline
        scoped.keywords = LightKeywords(scoped.chunks, settings=self.settings)
        scoped.graph = LightGraph(scoped)
        return guard(scoped)

    def catalog(self, *, ttl, force=False):
        from kb_search.catalog import build_catalog
        return build_catalog(self.settings, None, store=self)


class LightGraph:
    def __init__(self, store):
        self.store, self.settings = store, store.settings
        self.projection = LightGraphProjection(store.chunks)
        self.vectors = LightVectors(store.chunks)
        with store.chunks._db() as con:
            con.execute('''CREATE TABLE IF NOT EXISTS graph_metadata(kb_id TEXT, version TEXT, collection TEXT,
                counts TEXT, stats TEXT, created_at REAL, sealed INTEGER DEFAULT 0, PRIMARY KEY(kb_id,version))''')

    @contextmanager
    def _read_lock(self, exclusive=False):
        with (self.store.chunks.root / 'graph-read.lock').open('a+b') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            try:
                yield
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    @contextmanager
    def reader(self, source, version=None, *, timeout=None):
        deadline = self.store.deadline
        if timeout is not None:
            deadline = min(deadline or float('inf'), time.monotonic() + timeout)
        with self._read_lock(), self.projection.session(source.kb_id, version, deadline=deadline) as (gv, con):
            yield LightReader(self.store, source, gv, con)

    def check(self):
        self.store.chunks.check()
        with self.store.chunks._db() as con:
            rows = con.execute('SELECT kb_id,version FROM graph_active').fetchall()
        for kb, version in rows:
            with self.projection.session(kb, version) as (_, con):
                con.execute('SELECT count(*) FROM nodes').fetchone()

    def corpus_payloads(self, collection, refs):
        from ..graph.units import fetch_chunk_payloads
        return fetch_chunk_payloads(None, collection, refs, store=self.store.chunks)

    def write_vectors(self, embed, **options):
        from ..graph.vectors import write_graph_vectors
        with self.store.chunks._db() as con:
            sealed = con.execute('SELECT sealed FROM graph_metadata WHERE kb_id=? AND version=?',
                (options['kb_id'],options['graph_version'])).fetchone()
        if sealed and sealed[0]:
            raise ValueError('published graph versions are immutable')
        return write_graph_vectors(None, embed, storage=self.vectors, **options)

    def import_version(self, *, source_key, source, graph_version=None, replace=False, activate=True,
                       dry_run=False, batch_size=1000, build_record=None, acquire_lock=False, require_qdrant_aliases=True):
        if acquire_lock and not dry_run:
            from ..graph.lock import GraphBuildLock
            guard = GraphBuildLock(self.settings)
            guard.acquire()
            try:
                return self.import_version(source_key=source_key, source=source, graph_version=graph_version,
                    replace=replace, activate=activate, dry_run=dry_run, batch_size=batch_size,
                    build_record=build_record, require_qdrant_aliases=require_qdrant_aliases)
            finally:
                guard.release()
        from ..graph.neo4j_import import load_bundle, resolve_graph_build
        from ..graph.build import graph_paths
        record = build_record or resolve_graph_build(self.settings, source_key=source_key, source=source, graph_version=graph_version)
        version = str(record['graph_version'])
        output = Path(record.get('output_dir') or graph_paths(self.settings, source, version).output_dir)
        bundle = load_bundle(output, source=source, graph_version=version)
        expected = dict(bundle['expected_counts'], kb=1, graph_version=1, pages=len(json.loads((output / 'graph.json').read_text()).get('pages') or []))
        if not dry_run:
            self.projection.stage(source.kb_id, version, bundle)
            with self.store.chunks._db() as con:
                con.execute('''INSERT INTO graph_metadata(kb_id,version,collection,counts,stats,created_at) VALUES(?,?,?,?,?,?)
                    ON CONFLICT(kb_id,version) DO NOTHING''', (source.kb_id,version,source.collection,
                    json.dumps(expected),json.dumps(bundle.get('stats') or {}),time.time()))
            self._validate(source.kb_id,version)
            if activate:
                self.activate(source=source, graph_version=version)
        return dict(dry_run=dry_run,kb_id=source.kb_id,graph_version=version,expected_counts=expected,
                    actual_counts=expected,counts_match=True,steps=['light_projection'],output_dir=str(output))

    def _validate(self, kb, version):
        with self.store.chunks._db() as con:
            row = con.execute('SELECT collection,counts FROM graph_metadata WHERE kb_id=? AND version=?',(kb,version)).fetchone()
        if not row:
            raise ValueError('graph projection has not completed')
        collection, counts = row[0], json.loads(row[1])
        for kind in GRAPH_VECTOR_TYPES:
            name = graph_collection_name(collection,kind,version)
            expected = int(counts.get(kind, counts.get({'entity':'entities','relation':'relations','spec':'specs','page':'pages'}[kind],0)))
            if name not in self.vectors.names():
                if kind in GRAPH_OPTIONAL_TYPES and expected == 0:
                    continue
                raise ValueError(f'missing graph vectors: {kind}')
            if self.vectors.count(name) != expected:
                raise ValueError(f'graph vector count mismatch: {kind}')

    def activate_vectors(self, *, source_collection, graph_version):
        # Prepare only. The one graph_active pointer publishes both stores below.
        source = next(s for s in self.settings.sources.values() if s.collection == source_collection)
        self._validate(source.kb_id,graph_version)
        return {'previous': self.alias_targets(source_collection), 'active': {
            graph_collection_alias(source_collection,k): graph_collection_name(source_collection,k,graph_version)
            for k in GRAPH_VECTOR_TYPES if graph_collection_name(source_collection,k,graph_version) in self.vectors.names()}}

    def activate(self, *, source, graph_version):
        # GC cannot remove the staged generation between validation and publication.
        # Sealing and the active pointer commit together, under the vector writer lock.
        with self._read_lock():
            return self.projection.activate(source.kb_id, graph_version, seal=True,
                validate=lambda: self._validate(source.kb_id, graph_version))

    def restore_vectors(self, previous):
        # A failed preparation never moved the publication pointer.
        return {'restored': previous}

    def deactivate(self, source):
        return self.projection.deactivate(source.kb_id)

    def vector_collections(self):
        return self.vectors.names()

    def alias_targets(self, collection):
        with self.store.chunks._db() as con:
            row = con.execute('''SELECT m.version FROM graph_metadata m JOIN graph_active a
                ON a.kb_id=m.kb_id AND a.version=m.version WHERE m.collection=?''',(collection,)).fetchone()
        if not row:
            return {}
        names = self.vectors.names()
        return {graph_collection_alias(collection,k):graph_collection_name(collection,k,row[0]) for k in GRAPH_VECTOR_TYPES
                if graph_collection_name(collection,k,row[0]) in names}

    def vector_aliases(self):
        with self.store.chunks._db() as con:
            collections = [r[0] for r in con.execute('SELECT DISTINCT collection FROM graph_metadata')]
        return {k:v for col in collections for k,v in self.alias_targets(col).items()}

    def vector_exists(self, collection):
        return collection in self.vectors.names()

    def vector_count(self, collection):
        return self.vectors.count(collection)

    def drop_aliases(self, collection, types):
        return {'removed': []}  # Light has no independent mutable vector aliases.

    def projection_matches(self, kb_id, version, expected):
        from ..graph.neo4j_import import counts_match
        with self.store.chunks._db() as con:
            row = con.execute('SELECT counts FROM graph_metadata WHERE kb_id=? AND version=?',(kb_id,version)).fetchone()
        if not row:
            return False
        self._validate(kb_id,version)
        return counts_match(expected,json.loads(row[0]))

    def stats(self, kb_id, *, timeout=None):
        with self.store.chunks._db() as con:
            row = con.execute('''SELECT m.version,m.stats FROM graph_metadata m JOIN graph_active a
                ON a.kb_id=m.kb_id AND a.version=m.version WHERE m.kb_id=?''',(kb_id,)).fetchone()
        return (row[0],json.loads(row[1])) if row else (None,{})

    def query(self, source, question, *, limit, hops, **options):
        from kb_search.channels import graph_channel
        try:
            with self.reader(source, timeout=options.get('timeout')) as native:
                return graph_channel(self.settings,source,question,limit=limit,hops=hops,native=native,**options)
        except KeyError:
            return {'chunks': [], 'skipped': 'no active graph'}

    def _walk(self, method, source, options):
        from kb_search import graphwalk
        with self.reader(source, timeout=options.get('timeout')) as native:
            return getattr(graphwalk,method)(self.settings,source,native=native,**options)

    def neighbors(self,source,**options): return self._walk('neighbors',source,options)
    def entities(self,source,**options): return self._walk('list_entities',source,options)
    def facts(self,source,**options): return self._walk('list_facts',source,options)
    def pages(self,source,**options): return self._walk('list_pages',source,options)

    def locate_facts(self,collections,specs,sources,*,timeout=None):
        from kb_search.graphwalk import locate_facts
        return locate_facts(None,collections,specs,sources,timeout=timeout,store=self.store.chunks)

    def document_dates(self,sources,versions,*,timeout=None):
        from kb_search.docstatus import document_dates
        sources = list(sources)
        dates = document_dates(self.settings,sources,{})
        expected = {(r.get("kb_id"), r.get("doc_id")): r.get("content_version") for r in sources}
        deadline = time.monotonic() + timeout if timeout is not None else None
        for kb,version in versions.items():
            if not version or kb not in self.settings.sources:
                continue
            with self.reader(self.settings.sources[kb],version,
                    timeout=max(0,deadline-time.monotonic()) if deadline is not None else None) as reader:
                for row in reader.nodes('Document'):
                    if row.get('date') and row.get('content_version') and row['content_version'] == expected.get((kb, row.get('doc_id'))):
                        dates[kb,str(row['doc_id'])] = str(row['date'])
        return dates

    def annotate_dates(self,sources,scope,*,timeout=None):
        from kb_search.docstatus import annotate_dates
        return annotate_dates(self.settings,sources,scope,timeout=timeout,provider=self.document_dates)

    def read_originals(self,search_settings,result,scope,hint,until):
        from kb_search.readback import read_originals
        return read_originals(self.settings,search_settings,None,result,scope,hint,until,None,store=self.store)

    def neighborhoods(self,sources,versions,question,seeds,**options):
        from .light_neighborhoods import neighborhoods
        return neighborhoods(self,sources,versions,question,seeds,**options)

    def collect_vectors(self, **options):
        from .light_maintenance import collect_graph
        return collect_graph(self, **options)

    def collect_versions(self, *, sources, **options):
        return self.collect_vectors(source_collections=[s.collection for s in sources], **options)

    def drop(self, kb_id, collection):
        from .light_maintenance import drop_graph
        return drop_graph(self,kb_id,collection)
