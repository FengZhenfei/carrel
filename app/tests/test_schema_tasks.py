"""Background induction: single flight, cancellation fencing and reload/restart state."""
import json
import os
import socket
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from kb_pipeline import db, discovery
from kb_pipeline.graph import schema_flow as flow
from kb_pipeline.graph.llm import ChatClient, LLMSpec, LLMInterrupted
from kb_server import schema_tasks as tasks, service
from kb_server.main import create_app
from _support import _client
import test_graph_schema as schema_tests


@pytest.fixture
def setup(tmp_path):
    settings, source = schema_tests.SchemaFeedbackTests()._state(str(tmp_path), {'graph_enabled': True})
    settings.graph_work_dir = tmp_path/'graph'
    spec = LLMSpec('test', 'http://invalid', '', 'test', max_input_tokens=128000)
    yield settings, source, spec
    deadline = time.monotonic()+3
    while tasks._active and time.monotonic()<deadline:
        time.sleep(.01)
    assert not tasks._active


def wait_done(settings, source):
    deadline = time.monotonic()+3
    while time.monotonic()<deadline:
        out = tasks.status(settings, source.kb_id)
        if out['status'] != 'running':
            return out
        time.sleep(.01)
    pytest.fail('worker did not finish')


def test_api_returns_before_sampling_finishes_and_restores_result(setup):
    settings, source, spec = setup
    entered, release = threading.Event(), threading.Event()
    renew_seen = threading.Event()
    real_renew = flow.renew_suggest
    def renew(*args):
        renew_seen.set()
        return real_renew(*args)
    def sample(*a, **k):
        entered.set()
        assert release.wait(3)
        return {'texts':['a job role requires skills'], 'chunks_total':2,'documents':1}
    fake = _client(list(schema_tests.SchemaFeedbackTests.ANSWERS))
    with patch.object(service, '_prepare_graph_schema', return_value=setup), \
         patch.object(service, 'settings', return_value=settings), \
         patch.object(flow, 'sample_docs', side_effect=sample) as sampler, \
         patch('kb_pipeline.graph.llm.ChatClient', return_value=fake) as factory, \
         patch.object(flow, 'SUGGEST_RENEW_SECONDS', .02), \
         patch.object(flow, 'renew_suggest', side_effect=renew):
        api = TestClient(create_app())
        response = api.post('/api/kbs/kb_x/graph_schema')
        assert response.status_code == 202
        token = response.json()['task_id']
        assert entered.wait(1)
        assert renew_seen.wait(1), 'sampling must be covered by heartbeat'
        assert api.post('/api/kbs/kb_x/graph_schema').json()['task_id'] == token
        status = api.get('/api/kbs/kb_x/graph_schema').json()
        assert status['status']=='running' and status['phase']=='sampling'
        release.set()
        done = wait_done(settings, source)
        assert done['status']=='completed', done
        assert sampler.call_count==1
        assert factory.call_args.kwargs['timeout']==900
        assert factory.call_args.kwargs['connect_timeout']==10
        assert factory.call_args.kwargs['attempts']==3
        result = done['result']
        assert result['sampled']==1 and result['pending'] is True
        assert result['schema']['active'] != result['version_id']
        assert tasks.status(settings, source.kb_id)==done, 'reload can read the same result'
        with db.connect(settings.state_db) as con:
            discovery.set_config(con, source.kb_id, {'graph_schema_active': result['version_id']})
        assert tasks.status(settings, source.kb_id)['result']['pending'] is False
        assert api.post('/api/kbs/kb_x/graph_schema/cancel',json={}).status_code==422


@pytest.mark.parametrize('action', ['cancel','delete'])
def test_late_model_completion_cannot_publish_after_cancel_or_delete(setup, action):
    settings, source, spec = setup
    entered, release = threading.Event(), threading.Event()
    answers = list(schema_tests.SchemaFeedbackTests.ANSWERS)
    def reply(messages):
        entered.set()
        assert release.wait(3)
        return answers.pop(0)
    with patch.object(flow,'sample_docs',return_value={'texts':['sample']}), \
         patch('kb_pipeline.graph.llm.ChatClient',return_value=_client(reply)):
        task = tasks.start(*setup)
        assert entered.wait(1)
        if action=='cancel':
            assert tasks.cancel(settings,source.kb_id,task['task_id'])['status']=='cancelled'
        else:
            with db.connect(settings.state_db) as con:
                flow.discard_suggest(con,source.kb_id)
        release.set()
        deadline=time.monotonic()+2
        while tasks._active and time.monotonic()<deadline: time.sleep(.01)
        with db.connect(settings.state_db) as con:
            assert not discovery.get_config(con,source.kb_id).get('graph_schema_versions')
            assert not db.get_app_config(con,flow.SUGGEST_MARK_PREFIX+source.kb_id)
            if action=='delete': assert not db.get_app_config(con,flow.SUGGEST_TASK_PREFIX+source.kb_id)


def test_failure_retry_and_old_cancel_do_not_affect_new_task(setup):
    settings, source, spec = setup
    with patch.object(flow,'suggest_schema_version',side_effect=RuntimeError('network test')):
        first=tasks.start(*setup)
        failed=wait_done(settings,source)
        assert failed['status']=='failed' and 'network test' in failed['error']
    entered, release=threading.Event(),threading.Event()
    def hold(*a,**k):
        entered.set()
        assert release.wait(3)
        raise LLMInterrupted('test cancelled')
    with patch.object(flow,'suggest_schema_version',side_effect=hold):
        second=tasks.start(*setup)
        assert entered.wait(1)
        assert second['task_id']!=first['task_id']
        assert tasks.cancel(settings,source.kb_id,first['task_id'])['status']=='running'
        assert flow.suggest_owner(settings,source.kb_id)==second['task_id']
        release.set()
        assert wait_done(settings,source)['status']=='cancelled'


def test_startup_marks_previous_worker_interrupted_without_losing_cache(setup):
    settings, source, spec=setup
    token=flow.claim_suggest(settings,source.kb_id,'manual')
    with db.connect(settings.state_db) as con:
        db.set_app_config(con,flow.SUGGEST_TASK_PREFIX+source.kb_id,
            {'task_id':token,'status':'running','worker_pid':os.getpid(),
             'worker_host':socket.gethostname(),'worker_instance':'previous process'})
    tasks.recover_interrupted(settings)
    assert tasks.status(settings,source.kb_id)['status']=='interrupted'
    assert flow.suggest_owner(settings,source.kb_id) is None


def test_closing_kb_between_preflight_and_start_blocks_worker(setup):
    settings, source, spec=setup
    with db.connect(settings.state_db) as con:
        discovery.set_config(con,source.kb_id,{'graph_enabled':False})
    with patch.object(flow,'suggest_schema_version') as worker:
        with pytest.raises(ValueError,match='is disabled'): tasks.start(*setup)
        worker.assert_not_called()
    assert flow.suggest_owner(settings,source.kb_id) is None


def test_wait_for_healthy_task_is_cancellable_without_old_total_deadline(setup):
    settings, source, spec=setup
    token=flow.claim_suggest(settings,source.kb_id,'manual')
    stop=threading.Event()
    stop.set()
    with pytest.raises(LLMInterrupted): flow.wait_for_suggest(settings,source.kb_id,stop=stop)
    flow.release_suggest(settings,source.kb_id,token)


def test_schema_connect_read_timeout_does_not_change_other_clients():
    spec=LLMSpec('test','http://invalid','','test')
    seen=[]
    def chat(*a,**k):
        seen.append(k['timeout'])
        return 'OK'
    ChatClient(spec,chat=chat,timeout=900,connect_timeout=10).chat('one')
    ChatClient(spec,chat=chat,timeout=300).chat('two')
    assert seen==[(10,900),300]
