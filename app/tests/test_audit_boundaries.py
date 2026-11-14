"""Synthetic regressions for authentication, transport and evidence boundaries."""
import json
import os
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from kb_pipeline.embedding.client import EmbeddingClient, EmbeddingDimensionError
from kb_pipeline.vector import qdrant
from kb_pipeline.vector.layout import VectorLayout
from kb_search import service, docstatus
from test_search_skill import client
from test_embedding_throughput import text_embedder


def test_embedding_out_of_order_items_follow_input_indices(monkeypatch):
    embed = text_embedder(monkeypatch, lambda **kw: NS(data=[NS(index=1,embedding=[2.,0.]),NS(index=0,embedding=[1.,0.])]))
    assert embed.embed(['a','b']) == [[1.,0.],[2.,0.]]


@pytest.mark.parametrize('indices', [[0,0], [0,2], [0], [False,1], [None,1]])
def test_embedding_invalid_correspondence_is_rejected(monkeypatch, indices):
    embed = text_embedder(monkeypatch, lambda **kw: NS(data=[NS(index=i,embedding=[1.,0.]) for i in indices]))
    with pytest.raises(EmbeddingDimensionError):
        embed.embed(['a','b'])


@pytest.mark.parametrize('prefix', ['', '/carrel', '/parent/carrel'])
def test_console_auth_protects_mounted_routes_before_body(monkeypatch, prefix):
    from kb_server import main, api
    monkeypatch.setenv('KB_WEB_TOKEN', 'synthetic-token')
    reveal = Mock(return_value={'api_key':'synthetic-key'})
    monkeypatch.setattr(api.service, 'reveal_llm_key', reveal)
    child = main.create_app()
    app = FastAPI()
    if prefix:
        app.mount(prefix, child)
    else:
        app = child
    client = TestClient(app)
    target = prefix + '/api/llms/example/key'
    assert client.post(target).status_code == 401
    assert client.post(prefix + '/api/model-endpoints/embedding/key', content='{').status_code == 401
    reveal.assert_not_called()
    assert client.post(target, headers={'Authorization':'Bearer synthetic-token'}).status_code == 200


def test_search_auth_precedes_json_parsing(monkeypatch):
    from kb_search.main import create_app
    monkeypatch.setattr(service, 'runtime', lambda: (None, NS(token='synthetic-token'), None))
    app = FastAPI()
    app.mount('/carrel', create_app())
    c = TestClient(app)
    assert c.post('/carrel/search', content='{').status_code == 401
    assert c.post('/carrel/search', content='{', headers={'Authorization':'Bearer synthetic-token'}).status_code == 422


@contextmanager
def redirect_service():
    received = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            self.rfile.read(int(self.headers.get('Content-Length', '0')))
            received.append(self.path)
            if self.path == '/sink':
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{}')
            else:
                self.send_response(307)
                self.send_header('Location', '/sink')
                self.send_header('Content-Length','0')
                self.end_headers()
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', received
    finally:
        server.shutdown(); server.server_close(); thread.join()


@pytest.mark.parametrize('role', ['embedding','reranker','mineru','vlm'])
def test_runtime_never_forwards_input_to_a_redirect(role, tmp_path):
    with redirect_service() as (url, received):
        with pytest.raises(Exception):
            if role == 'embedding':
                EmbeddingClient(base_url=url+'/v1', api_key='', model_id='synthetic', dim=2,
                                batch_size=1, retry=1).embed(['private synthetic text'])
            elif role == 'reranker':
                from kb_search.rerank import Reranker
                Reranker(url, model_id='synthetic').score('synthetic question',['synthetic evidence'])
            elif role == 'vlm':
                from kb_pipeline.vision.vlm import _client_for
                _client_for(url+'/v1','synthetic').chat.completions.create(model='synthetic', messages=[{'role':'user','content':'synthetic image description'}])
            else:
                from kb_pipeline.parsers.service_clients import call_mineru_sync
                path = tmp_path/'document.pdf'; path.write_bytes(b'synthetic file')
                call_mineru_sync(url, path, tmp_path/'result.json', tmp_path/'result.md')
        assert received and '/sink' not in received


def test_alias_rollback_removes_optional_newer_pages():
    q = QdrantClient(':memory:')
    try:
        for version, kinds in [('old',('entity','relation','spec')), ('new',('entity','relation','spec','page'))]:
            for kind in kinds:
                q.create_collection(qdrant.graph_collection_name('kb_003',kind,version), vectors_config={'size':2,'distance':'Cosine'})
        qdrant.activate_graph_aliases(q, source_collection='kb_003', graph_version='new')
        change = qdrant.activate_graph_aliases(q, source_collection='kb_003', graph_version='old')
        aliases = qdrant.graph_alias_targets(q,'kb_003')
        assert aliases['graph_003_page'] is None
        qdrant.restore_graph_aliases(q,change['previous'])
        assert qdrant.graph_alias_targets(q,'kb_003')['graph_003_page'].endswith('__new')
    finally:
        q.close()


def test_dates_do_not_leak_from_an_older_document_version():
    driver = Mock()
    driver.session.return_value.__enter__ = Mock(return_value=driver)
    driver.session.return_value.__exit__ = Mock(return_value=False)
    driver.run.return_value.data.return_value = [{'doc_id':'d','date':'2020-01-01','content_version':'old'}]
    row = {'kb_id':'k','doc_id':'d','content_version':'new','rel_path':'manual.md'}
    assert docstatus.document_dates(None,[row],{'k':'graph'},driver=driver) == {}
    row['content_version'] = 'old'
    assert docstatus.document_dates(None,[row],{'k':'graph'},driver=driver) == {('k','d'):'2020-01-01'}


@pytest.mark.parametrize('failure', ['version','layout','missing','duplicate'])
def test_skill_context_batches_cannot_mix_or_silently_truncate(client, monkeypatch, failure):
    calls = []
    def request(url, token, timeout, method, route, data):
        calls.append(data)
        rows = [{'chunk_index':i,'chunk_total':100,'content_version':'v1','chunk_revision':'r1','text':'synthetic'}
                for i in range(data['chunk_from'],min(data['chunk_to']+1,100))]
        if len(calls)>1:
            if failure == 'missing': rows = []
            elif failure == 'duplicate': rows[-1]['chunk_index']=rows[0]['chunk_index']
            else:
                for r in rows: r['content_version' if failure=='version' else 'chunk_revision']='changed'
        return json.dumps({'sources':rows}).encode(), {}
    monkeypatch.setattr(client,'request',request)
    with pytest.raises(ValueError):
        client.read_context('http://synthetic','',5,{'chunk_from':0,'chunk_to':99},whole=True)
    assert calls[1]['content_version']=='v1' and calls[1]['chunk_revision']=='r1'


def test_skill_private_cache_rejects_symlink_and_repairs_permissions(client,tmp_path,monkeypatch):
    monkeypatch.delenv('CARREL_SEARCH_WORK_DIR',raising=False)
    root=tmp_path/'cache'; root.mkdir(mode=0o777); root.chmod(0o777)
    assert client.work_dir({'work_dir':str(root)}) == root
    assert root.stat().st_mode & 0o777 == 0o700
    link=tmp_path/'link'; link.symlink_to(root,target_is_directory=True)
    with pytest.raises(ValueError): client.work_dir({'work_dir':str(link)})
    saved=root/'new.json'; client.save_new(saved,b'{}')
    assert saved.stat().st_mode & 0o777 == 0o600


def test_mixed_query_preserves_text_evidence_when_visual_score_is_low():
    rows=[{'score_rerank':.9,'scores':{'score_visual':.1}}]
    assert service._mark_accepted(rows,'ok',.5,image_query=True)=='accepted'
    assert rows[0]['accepted']


def test_graph_gc_failure_preserves_build_records(tmp_path, monkeypatch):
    from kb_pipeline import db
    from kb_pipeline.graph import build
    state = tmp_path/'state.db'; db.init_db(state)
    cfg = NS(state_db=state, graph_work_dir=tmp_path/'graph', graph_gc_retention_days=7,
             graph_neo4j_import_after_build=True, neo4j_graph_retention_days=7)
    source = NS(kb_id='kb_001', collection='kb_001')
    store = NS(graph=NS(collect_vectors=Mock(side_effect=TimeoutError('unavailable')),
                       collect_versions=Mock(return_value={})))
    prune = Mock(side_effect=AssertionError('must preserve rollback records'))
    monkeypatch.setattr(db,'prune_graph_builds',prune)
    monkeypatch.setattr(build,'_prune_graph_workspaces',lambda *a, **kw: {})
    result = build.gc_graph_versions(cfg,source,storage=store,graph_version='v2',discard=set())
    assert 'graph_gc' in result['errors']
    assert result['result']['build_records_pruned']==0
    prune.assert_not_called()


def test_shell_and_python_read_the_same_generated_password(tmp_path, monkeypatch):
    import subprocess
    from kb_runtime import bootstrap
    from kb_pipeline.utils import load_env_file
    # No real credential; shell metacharacters must remain literal in both readers.
    value = "synthetic&value'with space$and`quotes`"
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setattr(os,'environ',{'PATH':os.environ['PATH'], 'KB_LOCAL_BASE_DIR':str(tmp_path),
        'KB_STATE_DB':str(tmp_path/'state.db'), 'KB_STORAGE_BACKEND':'standard',
        'KB_CONFIG_TEMPLATE':str(root/'config/knowledge-base.env.example'), 'NEO4J_PASSWORD':value})
    bootstrap.initialize()
    path=tmp_path/'config/knowledge-base.env'
    shell = subprocess.run(['sh','-c','. "$1"; printf %s "$NEO4J_PASSWORD"','sh',str(path)],capture_output=True,text=True,check=True)
    assert shell.stdout == value
    monkeypatch.delenv('NEO4J_PASSWORD')
    load_env_file(path)
    assert os.environ['NEO4J_PASSWORD'] == value
