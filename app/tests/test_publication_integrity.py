"""Failure injection across SQLite publication, LanceDB and local Qdrant."""
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from qdrant_client import QdrantClient

import pytest

from kb_pipeline import db, discovery
from kb_pipeline.pipeline import parse_job, worker
from kb_pipeline.storage import get_storage
from kb_pipeline.storage import standard
from kb_pipeline.storage.publication import repair_indexes
from test_scanner_worker import _parse_harness, _rescan


@pytest.fixture(params=['light', 'standard'])
def library(tmp_path, request):
    real_parse = parse_job._parse_blocks
    functions = {name: getattr(standard.qdrant, name) for name in (
        'collection_exists', 'ensure_collection', 'validate_collection_layout',
        'upsert_chunks', 'mark_old_versions_inactive', 'mark_stale_file_points_inactive')}
    with _parse_harness(str(tmp_path), text='ORION valve pressure is 15 kPa.') as h, ExitStack() as stack:
        h.settings.vlm_temperature = .1
        h.settings.vlm_top_p = .8
        h.settings.vlm_max_tokens = 64
        h.settings.vlm_structured_output = True
        h.settings.storage_backend = request.param
        h.settings.storage_root = tmp_path / 'store'
        if request.param == 'standard':
            h.q = QdrantClient(':memory:')
            stack.callback(h.q.close)
            stack.enter_context(patch.object(standard, 'vector_client', return_value=h.q))
            for name, fn in functions.items():
                stack.enter_context(patch.object(standard.qdrant, name, fn))
        with patch.object(parse_job, '_parse_blocks', real_parse):
            assert worker.run_once(h.con, settings=h.settings).startswith('parse-done:' )
            h.con.commit()
            h.source = discovery.enrolled_sources(h.settings.state_db, h.settings.mirror_root)[h.kb_id]
            h.store = get_storage(h.settings)
            h.path = Path(db.get_file_by_id(h.con, h.file_id)['physical_path'])
            yield h


def texts(h):
    rows = h.store.chunks.context(h.source.collection, doc_id=f'{h.kb_id}:{db.get_file_by_id(h.con,h.file_id)["file_key"]}',
                                  chunk_from=0, chunk_to=100)
    return [r['text'] for r in rows]


def test_retirement_failure_publishes_only_new_version_and_queues_cleanup(library, monkeypatch):
    h = library
    old = db.active_chunk_refs(h.con, h.source.collection)
    h.path.write_text('ORION valve pressure is now 27 kPa.')
    _rescan(h)
    with monkeypatch.context() as m:
        m.setattr(type(h.store.chunks.raw), 'retire_stale', lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('injected cleanup failure')))
        assert worker.run_once(h.con, settings=h.settings).startswith('parse-done:')
    h.con.commit()
    assert '27 kPa' in ' '.join(texts(h)) and '15 kPa' not in ' '.join(texts(h))
    assert h.store.chunks.metadata(h.source.collection, [old[0]['point_id']])[old[0]['point_id']]['active'] is False
    h.settings.sources = {h.kb_id: h.source}
    from kb_search.config import load_search_settings
    import time
    package = {'question': 'What is the ORION pressure?', 'kbs': [h.kb_id],
               'sources': [], 'entities': [], 'relationships': [],
               'specs': [{'kb_id': h.kb_id, 'evidence': [old[0]]}]}
    with patch('kb_search.readback.rerank', side_effect=AssertionError('retired text must not reach ranking')):
        additions, _ = h.store.graph.read_originals(load_search_settings({}), package, {}, {}, time.time()+5)
    assert additions == []
    facts = [{'kb_id': h.kb_id, 'point_ids': [old[0]['point_id']]}]
    assert h.store.graph.locate_facts({h.kb_id: h.source.collection}, facts, []) == []
    assert facts[0]['evidence'][0]['active'] is False
    assert h.con.execute("SELECT count(*) FROM jobs WHERE job_type='fts_sync' AND status='queued'").fetchone()[0] == 1
    assert worker.run_once(h.con, settings=h.settings).startswith('fts-sync-done:')
    h.con.commit()
    assert h.store.chunks.raw.active_count(h.source.collection) == len(db.active_chunk_refs(h.con, h.source.collection))


def test_partial_vector_write_never_changes_published_evidence(library, monkeypatch):
    h = library
    before = texts(h)
    h.path.write_text('ORION valve pressure is now 27 kPa.')
    _rescan(h)
    raw_type = type(h.store.chunks.raw)
    original = raw_type.write
    def partial(self, collection, ids, vectors, payloads, **kw):
        original(self, collection, ids[:1], vectors[:1], payloads[:1], **kw)
        raise RuntimeError('injected partial write')
    with monkeypatch.context() as m:
        m.setattr(raw_type, 'write', partial)
        result = worker.run_once(h.con, settings=h.settings)
        assert 'parse-done' not in result
    h.con.commit()
    assert texts(h) == before
    # A retry publishes the new complete set and retires staged leftovers.
    h.con.execute("UPDATE jobs SET next_attempt_at=0 WHERE status='retry'")
    h.con.commit()
    assert worker.run_once(h.con, settings=h.settings).startswith('parse-done:')
    h.con.commit()
    assert '27 kPa' in ' '.join(texts(h))


def test_cleanup_follows_published_version_while_a_new_file_waits(library, monkeypatch):
    h = library
    h.path.write_text('ORION valve pressure is now 27 kPa.')
    _rescan(h)
    with monkeypatch.context() as m:
        m.setattr(type(h.store.chunks.raw), 'retire_stale',
                    lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('cleanup failed')))
        assert worker.run_once(h.con, settings=h.settings).startswith('parse-done:')
    h.con.commit()
    h.path.write_text('ORION valve pressure is now 36 kPa.')
    _rescan(h, 'third')
    # A new parse is pending, but compensation still belongs to published 27 kPa.
    assert worker.run_once(h.con, settings=h.settings).startswith('fts-sync-done:')
    h.con.commit()
    assert h.store.chunks.raw.active_count(h.source.collection) == len(db.active_chunk_refs(h.con, h.source.collection))
    assert '27 kPa' in ' '.join(texts(h))


def test_sqlite_publication_failure_rolls_back_the_entire_ledger(library, monkeypatch):
    h = library
    before = texts(h)
    h.path.write_text('ORION valve pressure is now 27 kPa.')
    _rescan(h)
    with monkeypatch.context() as m:
        m.setattr(db, 'mark_file_indexed', lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('injected SQLite publish failure')))
        assert 'parse-done' not in worker.run_once(h.con, settings=h.settings)
    h.con.commit()
    assert texts(h) == before


def test_restore_missing_indexes_queues_normal_parse_and_reuses_cache(library):
    h = library
    raw = h.store.chunks.raw
    if h.settings.storage_backend == 'light':
        raw._table(h.source.collection).delete('true')
        assert texts(h) == []
    else:
        h.q.delete_collection(h.source.collection)
    assert repair_indexes(h.con, h.store, h.kb_id, ingest_run_id='restore', dry_run=True) == 1
    assert repair_indexes(h.con, h.store, h.kb_id, ingest_run_id='restore') == 1
    h.con.commit()
    pending = h.con.execute("SELECT payload_json FROM jobs WHERE job_type='parse' AND status='queued'").fetchone()[0]
    assert 'refresh_parse_cache' not in pending
    assert worker.run_once(h.con, settings=h.settings).startswith('parse-done:')
    h.con.commit()
    assert '15 kPa' in ' '.join(texts(h))


def test_index_connection_failure_does_not_turn_into_a_rebuild(library, monkeypatch):
    h = library
    monkeypatch.setattr(h.store.chunks.raw, 'exists', lambda *a: (_ for _ in ()).throw(TimeoutError('offline')))
    with pytest.raises(TimeoutError):
        repair_indexes(h.con, h.store, h.kb_id, ingest_run_id='restore')
    assert db.get_file_by_id(h.con, h.file_id)['status'] == 'indexed'


def test_embedding_endpoint_change_requires_an_empty_index(library):
    from kb_server.model_endpoints import _check_embedding_change
    h = library
    h.settings.graph_work_dir = h.settings.state_db.parent/'graph'
    h.settings.runtime_dir = h.settings.state_db.parent/'runtime'
    current = {'base_url':'https://first.invalid/v1', 'model_id':'same-model', 'api_key':'', 'dim':4, 'enabled':True}
    with pytest.raises(ValueError, match='reinitialize'):
        _check_embedding_change(h.con,h.settings,'embedding',current,{**current,'base_url':'https://other.invalid/v1'})


def test_rechunking_changes_references_and_rejects_an_old_context_revision(library):
    from kb_pipeline.pipeline.scheduler import requeue_single_file
    from kb_search import service
    from kb_search.library import StaleReference
    h=library
    original = db.active_chunk_refs(h.con,h.source.collection)
    payload=h.store.chunks.payloads(h.source.collection,[original[0]['point_id']])[original[0]['point_id']]
    # Alter the indexed representation with identical file bytes.
    discovery.set_config(h.con,h.kb_id,{'max_tokens':128})
    requeue_single_file(h.con,ingest_run_id='rechunk',file_row=db.get_file_by_id(h.con,h.file_id))
    h.con.commit()
    with patch.object(parse_job,'_parse_blocks',return_value=[__import__('test_scanner_worker')._block('b1','A different extracted representation of the same file.')]):
        assert worker.run_once(h.con,settings=h.settings).startswith('parse-done:')
    h.con.commit()
    assert h.store.chunks.metadata(h.source.collection,[original[0]['point_id']])[original[0]['point_id']]['active'] is False
    h.settings.sources={h.kb_id:h.source}
    with patch.object(service,'runtime',return_value=(h.settings,None,h.store)), pytest.raises(StaleReference):
        service.context(h.kb_id,payload['doc_id'],0,0,content_version=payload['content_version'],chunk_revision=payload['chunk_revision'])


def test_unpublished_points_do_not_contribute_keyword_counts(library):
    h = library
    refs = db.active_chunk_refs(h.con, h.source.collection)
    if h.settings.storage_backend == 'standard':
        # The backend must constrain both snippets and aggregates to this ID set.
        with patch('kb_search.library.grep', return_value={}) as grep:
            h.store.keywords.grep(h.source, ['ORION'])
        assert set(grep.call_args.kwargs['published_ids']) == {r['point_id'] for r in refs}
        return
    import uuid
    raw = h.store.chunks.raw
    payload = h.store.chunks.payloads(h.source.collection, [refs[0]['point_id']])[refs[0]['point_id']]
    raw.write(h.source.collection, [str(uuid.uuid4())], [{'text': [0.] * 4}],
              [{**payload, 'chunk_uid': 'unpublished'}], max_bytes=4096)
    out = h.store.keywords.grep(h.source, ['ORION'])['results'][0]
    assert out['total_chunks'] == 1
    assert len(out['hits']) == 1
    assert out['docs'][0]['chunks'] == 1
