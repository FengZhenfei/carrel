"""Keyword navigation uses the same temporal boundaries as evidence search."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from kb_pipeline import db
from kb_search import channels, docstatus, library, service
from kb_search.main import create_app


@pytest.fixture
def runtime(tmp_path):
    state = tmp_path / 'state.db'
    db.init_db(state)
    settings = SimpleNamespace(state_db=state, sources={'sample': SimpleNamespace(kb_id='sample', collection='sample', source_root='Sample')},
                               opensearch_url='http://unused')
    with patch.object(service, 'runtime', return_value=(settings, SimpleNamespace(token='test-token', channel_timeout=3), None)):
        yield settings


@pytest.mark.parametrize('extra,mode,day', [({}, 'current', None), ({'history': True}, 'history', None),
                                            ({'as_of': '2024-02'}, 'as_of', '2024-02-29')])
def test_keyword_api_propagates_time_and_keeps_version(runtime, extra, mode, day):
    excluded = ['superseded.md', 'archived.md'] if mode != 'history' else []
    result = {'results': [{'hits': [{'point_id': 'p', 'doc_id': 'sample:1', 'content_version': 'v1', 'rel_path': 'record.md'}]}]}
    with patch.object(docstatus, 'corrections_of', return_value={}), \
         patch.object(docstatus, 'excluded_paths', return_value=excluded) as paths, \
         patch.object(channels, 'os_client', return_value=object()), \
         patch.object(library, 'grep', return_value=result) as lookup:
        response = TestClient(create_app()).post('/grep', json={'kb_id': 'sample', 'phrases': ['valve'], **extra},
                                                headers={'Authorization': 'Bearer test-token'})
    assert response.status_code == 200
    data = response.json()
    assert data['time']['mode'] == mode
    if day:
        assert data['time']['day'] == day
    assert data['time']['excluded_documents']['sample'] == len(excluded)
    assert lookup.call_args.kwargs['excluded_paths'] == excluded
    assert paths.call_args.kwargs['max_paths'] is None
    assert data['results'][0]['hits'][0]['content_version'] == 'v1'
    assert data['results'][0]['hits'][0]['kb_id'] == 'sample'


def test_unreadable_status_rules_are_not_reported_as_zero_hits(runtime):
    with patch.object(docstatus, 'corrections_of', return_value={'errors': ['invalid']}), \
         patch.object(library, 'grep') as lookup:
        response = TestClient(create_app()).post('/grep', json={'kb_id': 'sample', 'phrases': ['valve']},
                                                headers={'Authorization': 'Bearer test-token'})
    assert response.status_code == 503
    lookup.assert_not_called()


def test_exclusion_list_for_counts_is_not_cut_at_search_budget(runtime):
    from kb_pipeline.graph.currency import query_scope
    paths = [f'old/{n}.md' for n in range(2101)]
    with patch.object(docstatus, 'corrections_of', return_value={'documents': [{'path': 'old/**', 'status': 'archived'}]}), \
         patch.object(docstatus, '_doc_paths', return_value=paths), \
         patch.object(docstatus, 'document_in_scope', return_value=False):
        source = runtime.sources['sample']
        assert len(docstatus.excluded_paths(runtime, source, query_scope())) == 2000
        assert docstatus.excluded_paths(runtime, source, query_scope(), max_paths=None) == paths


def test_keyword_query_excludes_paths_before_counts_and_aggregations(runtime):
    captured = []
    fake = SimpleNamespace(indices=SimpleNamespace(exists=lambda **kw: True))
    def msearch(**kw):
        captured.extend(kw['body'])
        return {'responses': [{'hits': {'total': {'value': 1}, 'hits': [{'_id': 'p', '_source': {
            'doc_id': 'sample:1', 'rel_path': 'current.md', 'content_version': 'v9', 'body': 'valve', 'chunk_index': 2}}]},
            'aggregations': {'by_doc': {'buckets': []}}}]}
    fake.msearch = msearch
    out = library.grep(runtime, runtime.sources['sample'], ['valve'], client=fake, excluded_paths=['old.md'])
    assert {'bool': {'must_not': [{'terms': {'rel_path': ['old.md']}}]}} in captured[1]['query']['bool']['filter']
    assert out['results'][0]['hits'][0]['content_version'] == 'v9'


@pytest.mark.parametrize('as_of', ['no-date', '2024-99-01'])
def test_invalid_keyword_date_is_rejected(runtime, as_of):
    response = TestClient(create_app()).post('/grep', json={'kb_id': 'sample', 'phrases': ['valve'], 'as_of': as_of},
                                            headers={'Authorization': 'Bearer test-token'})
    assert response.status_code == 422


@pytest.mark.parametrize('scope,expected', [
    ({'mode': 'current', 'day': '2026-06-01'}, ['old.md', 'future.md']),
    ({'mode': 'history', 'day': '2026-06-01'}, []),
    ({'mode': 'as_of', 'day': '2024-06-01'}, ['future.md']),
])
def test_real_status_rules_filter_before_keyword_counting(runtime, scope, expected):
    rules = {'documents': [{'path': 'old.md', 'status': 'superseded', 'valid_until': '2024-12-31'},
                           {'path': 'future.md', 'valid_from': '2027-01-01'}]}
    with patch.object(docstatus, 'corrections_of', return_value=rules), \
         patch.object(docstatus, '_doc_paths', return_value=['old.md', 'current.md', 'future.md']):
        assert docstatus.excluded_paths(runtime, runtime.sources['sample'], scope, max_paths=None) == expected
