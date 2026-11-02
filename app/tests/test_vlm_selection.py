"""Per-KB visual routing, with synthetic documents and an isolated HTTP model endpoint."""
import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image

from kb_pipeline import db, discovery
from kb_pipeline.models import ParsedBlock
from kb_pipeline.pipeline import parse_job
from kb_pipeline.parsers import table_repair
from kb_pipeline.vision import vlm
from kb_server import service


@pytest.fixture
def setup(tmp_path):
    state = tmp_path / 'state.db'
    db.init_db(state)
    mirror = tmp_path / 'mirror'
    (mirror / 'documents').mkdir(parents=True)
    with db.connect(state) as con:
        source, _ = discovery.enroll(con, mirror, 'documents')
        db.upsert_llm(con, name='vision-a', base_url='http://vision-a.invalid/v1', api_key='key-a', model_id='image-a')
        db.upsert_llm(con, name='vision-b', base_url='http://vision-b.invalid/v1', api_key='key-b', model_id='image-b')
        db.upsert_llm(con, name='messages-model', base_url='http://messages.invalid', api_key='', model_id='message', protocol='anthropic')
    cfg = SimpleNamespace(state_db=state, mirror_root=mirror, sources={source.kb_id: source},
        cache_dir=tmp_path / 'cache', embedding_base_url='http://embedding.invalid/v1',
        mineru_url='http://parser.invalid', mineru_timeout_seconds=5,
        vlm_base_url='http://default.invalid/v1', vlm_api_key='default-key', vlm_model_id='default-image',
        vlm_concurrency=1, vlm_temperature=0.1, vlm_top_p=0.8, vlm_max_tokens=4096,
        vlm_structured_output=True, image_max_pixels=200000)
    return source, cfg


@pytest.fixture
def model_server():
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            calls.append((self.path, self.headers.get('Authorization'), body))
            data = {'kind': 'diagram', 'summary': 'Synthetic device with a blue rectangle.',
                    'text_verbatim': 'Synthetic device', 'confidence': 'high', 'decorative': False,
                    'entities': [], 'facts': [], 'keywords': [],
                    'rows': [{'key': 'Alpha', 'cells': ['Alpha', '12', '34']}]}
            response = json.dumps({'id': 'synthetic', 'object': 'chat.completion', 'created': 0,
                'model': body['model'], 'choices': [{'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': json.dumps(data)}}]}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(response)))
            self.end_headers()
            self.wfile.write(response)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1', calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_selection_is_passive_and_fresh_for_each_job(setup):
    source, cfg = setup
    limits = {'max_tokens_cap': 3276, 'max_tokens_min': 128}
    with patch.object(service, 'settings', return_value=cfg), patch.object(service, 'chunk_limits', return_value=limits):
        for name, model in [('vision-a', 'image-a'), ('vision-b', 'image-b')]:
            service.update_kb_config(source.kb_id, {'vlm_model': name})
            fresh = parse_job._source_for_file(cfg, source.kb_id)
            assert fresh.vlm_model == name
            assert parse_job._vlm_endpoint(cfg, fresh)[2] == model
        with db.connect(cfg.state_db) as con:
            assert con.execute('SELECT count(*) FROM jobs').fetchone()[0] == 0
            assert con.execute('SELECT count(*) FROM graph_builds').fetchone()[0] == 0
        for bad in [None, '', '   ', ' vision-a ', 'missing', 'messages-model']:
            with pytest.raises(ValueError):
                service.update_kb_config(source.kb_id, {'vlm_model': bad})
        assert service.get_kb_config(source.kb_id)['config']['vlm_model'] == 'vision-b'


def test_invalid_type_rejected_before_persistence():
    from fastapi import HTTPException
    from kb_server.api import _reject_bad_config
    for value in [True, 12, [], {}]:
        with pytest.raises(HTTPException):
            _reject_bad_config({'vlm_model': value})
    _reject_bad_config({'vlm_model': None})
    _reject_bad_config({'vlm_model': 'vision-a'})


def test_deleted_reference_is_cleared_and_secret_never_listed(setup):
    source, cfg = setup
    with db.connect(cfg.state_db) as con:
        discovery.set_config(con, source.kb_id, {'vlm_model': 'vision-a', 'graph_llm': {'extract': 'vision-b'}})
    with patch.object(service, 'settings', return_value=cfg):
        listed = service.list_llms()
        assert 'key-a' not in json.dumps(listed)
        model = next(m for m in listed if m['name'] == 'vision-a')
        assert model['used_by'][0]['steps'] in [['Multimodal model'], ['多模态模型']]
        out = service.remove_llm('vision-a')
        assert out['cleared'][0]['kb_id'] == source.kb_id
    with db.connect(cfg.state_db) as con:
        config = discovery.get_config(con, source.kb_id)
    assert 'vlm_model' not in config
    assert config['graph_llm'] == {'extract': 'vision-b'}
    with pytest.raises(parse_job.NonRetryableParseError, match='No multimodal model configured'):
        parse_job._vlm_endpoint(cfg, parse_job._source_for_file(cfg, source.kb_id))


def test_parse_save_requires_model_but_graph_only_save_is_independent(setup):
    source, cfg = setup
    limits = {'max_tokens_cap': 3276, 'max_tokens_min': 128}
    with patch.object(service, 'settings', return_value=cfg), patch.object(service, 'chunk_limits', return_value=limits):
        for updates in ({'vlm_model': None}, {'vlm_model': ''}, {'max_tokens': 600}, {'vlm_prompt': 'Read the image.'}):
            with pytest.raises(ValueError, match='Select a multimodal model'):
                service.update_kb_config(source.kb_id, updates)
        service.update_kb_config(source.kb_id, {'graph_auto_append': False})
        service.update_kb_config(source.kb_id, {'vlm_model': 'vision-a'})
        service.update_kb_config(source.kb_id, {'max_tokens': 600})
        saved = service.get_kb_config(source.kb_id)['config']
        assert saved['vlm_model'] == 'vision-a'
        assert saved['max_tokens'] == 600
        assert saved['graph_auto_append'] is False


def test_enrollment_rejects_unselected_model_before_creating_library(setup):
    source, cfg = setup
    (cfg.mirror_root / 'new-documents').mkdir()
    limits = {'max_tokens_cap': 3276, 'max_tokens_min': 128}
    with patch.object(service, 'settings', return_value=cfg), patch.object(service, 'chunk_limits', return_value=limits), \
         patch.object(service, 'qdrant_client') as vector:
        with pytest.raises(ValueError, match='Select a multimodal model'):
            service.enroll('new-documents', {'max_tokens': 400, 'overlap_tokens': 80, 'vlm_model': None})
        vector.assert_not_called()
    with db.connect(cfg.state_db) as con:
        assert con.execute('SELECT count(*) FROM kb_sources').fetchone()[0] == 1
        assert con.execute('SELECT count(*) FROM jobs').fetchone()[0] == 0


def test_api_rejects_clearing_model_without_changing_saved_config(setup, monkeypatch):
    from fastapi.testclient import TestClient
    from kb_server.main import create_app

    source, cfg = setup
    monkeypatch.delenv('KB_WEB_TOKEN', raising=False)
    with db.connect(cfg.state_db) as con:
        discovery.set_config(con, source.kb_id, {'vlm_model': 'vision-a'})
        before = con.execute('SELECT config_json FROM kb_sources WHERE kb_id=?', (source.kb_id,)).fetchone()[0]
    with patch.object(service, 'settings', return_value=cfg), \
         patch.object(service, 'chunk_limits', return_value={'max_tokens_cap': 3276, 'max_tokens_min': 128}):
        client = TestClient(create_app())
        for empty in [None, '', '   ']:
            result = client.put(f'/api/kbs/{source.kb_id}/config', json={'vlm_model': empty})
            assert result.status_code == 422
            assert 'Select a multimodal model' in result.json()['detail']
    with db.connect(cfg.state_db) as con:
        assert con.execute('SELECT config_json FROM kb_sources WHERE kb_id=?', (source.kb_id,)).fetchone()[0] == before
        assert con.execute('SELECT count(*) FROM jobs').fetchone()[0] == 0


def test_referenced_model_cannot_change_to_incompatible_protocol(setup):
    source, cfg = setup
    with db.connect(cfg.state_db) as con:
        discovery.set_config(con, source.kb_id, {'vlm_model': 'vision-a'})
    with patch.object(service, 'settings', return_value=cfg), patch.object(service, '_probe_llm'):
        with pytest.raises(ValueError, match='OpenAI'):
            service.save_llm({'name': 'vision-a', 'base_url': 'http://vision-a.invalid/v1',
                              'model_id': 'image-a', 'protocol': 'anthropic', 'clear_api_key': True})
    assert parse_job._vlm_endpoint(cfg, replace(source, vlm_model='vision-a')) == ('http://vision-a.invalid/v1', 'key-a', 'image-a')


@pytest.mark.parametrize('suffix,parser', [('.pdf','parse_pdf_enhanced'), ('.docx','parse_docx_enhanced'),
                                         ('.pptx','parse_pptx_enhanced'), ('.png','parse_image_file')])
def test_each_visual_route_uses_selected_model(setup, tmp_path, suffix, parser):
    source, cfg = setup
    path = tmp_path / ('sample' + suffix)
    path.write_bytes(b'synthetic content')
    with patch.object(parse_job, parser, return_value=[]) as route:
        parse_job._parse_blocks(cfg, replace(source, vlm_model='vision-b'), path, tmp_path/'parse', 'profile', {})
    args = route.call_args.kwargs
    assert (args['vlm_base_url'], args['vlm_api_key'], args['vlm_model_id']) == ('http://vision-b.invalid/v1','key-b','image-b')
    assert cfg.vlm_model_id == 'default-image'


def test_notebook_route_uses_selected_model(setup, tmp_path):
    source, cfg = setup
    path = tmp_path / 'sample.ipynb'
    path.write_text('{}')
    with patch('kb_pipeline.parsers.notebook.parse_notebook', return_value=[]), \
         patch('kb_pipeline.parsers.visual_blocks.enrich_blocks_with_vlm') as route:
        parse_job._parse_blocks(cfg, replace(source, vlm_model='vision-b'), path, tmp_path/'parse', 'profile', {})
    args = route.call_args.kwargs
    assert (args['base_url'], args['api_key'], args['model_id']) == ('http://vision-b.invalid/v1','key-b','image-b')


def test_invalid_vision_model_does_not_break_text_route(setup, tmp_path):
    source, cfg = setup
    path = tmp_path / 'sample.md'
    path.write_text('# Guide\n\nThis is synthetic ordinary text.\n')
    blocks = parse_job._parse_blocks(cfg, replace(source, vlm_model='missing'), path, tmp_path/'parse', 'profile', {})
    assert any('synthetic ordinary text' in b.text for b in blocks)
    with pytest.raises(parse_job.NonRetryableParseError, match='not in the registry|不在模型列表中'):
        parse_job._vlm_endpoint(cfg, replace(source, vlm_model='missing'))


def test_image_http_request_and_model_endpoint_cache_isolation(setup, tmp_path, model_server):
    source, cfg = setup
    url, calls = model_server
    image = tmp_path/'diagram.png'
    Image.new('RGB', (320, 240), 'blue').save(image)
    source = replace(source, vlm_model='vision-a')
    def register(endpoint, model='shared-model', key='first-key'):
        with db.connect(cfg.state_db) as con:
            db.upsert_llm(con, name='vision-a', base_url=endpoint, api_key=key, model_id=model)
    def parse():
        return parse_job._parse_blocks(cfg, source, image, tmp_path/'parse', 'profile', {})
    register(url)
    first = parse()
    assert first[0].metadata['vlm_model'] == 'shared-model'
    assert len(calls) == 1
    assert calls[0][0] == '/v1/chat/completions'
    assert calls[0][1] == 'Bearer first-key'
    assert calls[0][2]['messages'][0]['content'][1]['image_url']['url'].startswith('data:image/')
    parse()
    assert len(calls) == 1
    register(url + '/other', key='second-key')
    parse()
    assert len(calls) == 2
    assert calls[-1][0] == '/v1/other/chat/completions'
    assert calls[-1][1] == 'Bearer second-key'
    register(url + '/other', model='different-model', key='second-key')
    parse()
    assert len(calls) == 3 and calls[-1][2]['model'] == 'different-model'


def test_table_review_uses_same_model_and_endpoint_cache_identity(setup, tmp_path, model_server):
    source, cfg = setup
    url, calls = model_server
    image = tmp_path/'table.png'
    Image.new('RGB', (320,240), 'white').save(image)
    with db.connect(cfg.state_db) as con:
        db.upsert_llm(con, name='vision-a', base_url=url, api_key='review-key', model_id='review-model')
    endpoint, key, model = parse_job._vlm_endpoint(cfg, replace(source, vlm_model='vision-a'))
    args = dict(image_path=image, base_url=endpoint, api_key=key, model_id=model, row_keys=['Alpha'], cache_json=tmp_path/'rows.json')
    rows = table_repair.transcribe_rows(**args)
    assert rows[0]['cells'] == [['Alpha'], ['12'], ['34']]
    table_repair.transcribe_rows(**args)
    assert len(calls) == 1
    table_repair.transcribe_rows(**{**args, 'base_url': endpoint+'/other'})
    assert len(calls) == 2
    assert calls[-1][1] == 'Bearer review-key' and calls[-1][2]['model'] == 'review-model'


@pytest.mark.parametrize('legacy_default', [False, True])
def test_unselected_model_never_uses_global_default(setup, legacy_default):
    source, cfg = setup
    if not legacy_default:
        cfg.vlm_base_url = cfg.vlm_model_id = ''
    with patch.object(service.requests, 'get') as request:
        with pytest.raises(parse_job.NonRetryableParseError):
            parse_job._vlm_endpoint(cfg, source)
    request.assert_not_called()


def test_health_does_not_probe_selected_or_default_vision_models(setup, tmp_path):
    source, cfg = setup
    cfg.sources = {source.kb_id: replace(source, vlm_model='vision-b')}
    cfg.console_services = ('database', 'mineru', 'embedding', 'visual_embedding', 'reranker')
    cfg.qdrant_url = 'http://vector.invalid'
    cfg.opensearch_url = 'http://keyword.invalid'
    cfg.neo4j_uri = ''
    cfg.visual_embedding_enabled = True
    cfg.visual_embedding_base_url = 'http://visual-embedding.invalid/v1'
    cfg.reranker_base_url = 'http://reranker.invalid/v1'
    cfg.visual_reranker_base_url = 'http://visual-reranker.invalid/v1'
    cfg.parse_enabled = True
    cfg.runtime_dir = tmp_path
    with patch.object(service.requests, 'get', return_value=SimpleNamespace(status_code=200)) as request, \
         patch.object(service, 'settings', return_value=cfg), \
         patch.object(service, '_probe_neo4j', return_value=False), \
         patch.object(service, '_systemctl_show', return_value={}), \
         patch.object(service, '_systemctl_list_timers', return_value={}), \
         patch.object(service, '_maintenance_defers', return_value={}), \
         patch.object(service, 'timer_health', return_value=[]), \
         patch.dict(service._health_cache, {'at': 0, 'value': None}):
        result = service.health()
        assert 'vlm' not in result
        assert {call.args[0] for call in request.call_args_list} == {
            'http://vector.invalid/healthz', 'http://keyword.invalid/_cluster/health',
            'http://parser.invalid/health', 'http://embedding.invalid/health',
            'http://visual-embedding.invalid/health', 'http://reranker.invalid/health',
            'http://visual-reranker.invalid/health',
        }
        for key in ('qdrant', 'opensearch', 'mineru', 'embedding', 'visual_embedding',
                    'reranker', 'visual_reranker'):
            assert result[key] is True
        assert not {'carrel-vlm', 'qwen3-vl-8b-instruct-fp8'} & set(service._all_service_containers())
        with pytest.raises(KeyError):
            service.restart_service('vlm')


def test_switch_between_documents_keeps_prior_result_and_uses_new_choice(setup, tmp_path, model_server):
    source, cfg = setup
    url, calls = model_server
    image = tmp_path/'synthetic.png'
    Image.new('RGB', (320, 240), 'blue').save(image)
    with db.connect(cfg.state_db) as con:
        for name in ['vision-a', 'vision-b']:
            db.upsert_llm(con, name=name, base_url=url, api_key='synthetic-key', model_id=name)
        discovery.set_config(con, source.kb_id, {'vlm_model':'vision-a'})
    in_flight = parse_job._source_for_file(cfg, source.kb_id)
    with db.connect(cfg.state_db) as con:
        discovery.set_config(con, source.kb_id, {'vlm_model':'vision-b'})
    first = parse_job._parse_blocks(cfg, in_flight, image, tmp_path/'first', 'profile', {})
    next_source = parse_job._source_for_file(cfg, source.kb_id)
    second = parse_job._parse_blocks(cfg, next_source, image, tmp_path/'second', 'profile', {})
    assert [call[2]['model'] for call in calls] == ['vision-a','vision-b']
    assert first[0].metadata['vlm_model'] == 'vision-a'
    assert second[0].metadata['vlm_model'] == 'vision-b'
