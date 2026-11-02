"""Four local retrieval models, external vision endpoints, and legacy upgrade cleanup."""
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def deploy_functions(tmp_path):
    # Load functions without running installation or Docker.
    source = (ROOT/'deploy.sh').read_text().split('case "$CMD" in\n', 1)[0]
    source = source[:source.index('for arg in ')] + source[source.index('log()'): ]
    script = tmp_path/'functions.sh'
    script.write_text(source)
    return script


@pytest.mark.parametrize('with_models', [False, True])
def test_model_profiles_keep_retrieval_dependencies_local(with_models):
    data = yaml.safe_load((ROOT/'deployment/compose/docker-compose.yml').read_text())
    services = data['services']
    active = {name for name, spec in services.items() if not spec.get('profiles') or (with_models and 'local-models' in spec['profiles'])}
    assert 'mineru' in active
    retrieval = {'embedding', 'reranker', 'vl-embedding', 'vl-reranker'}
    assert active & retrieval == (retrieval if with_models else set())
    assert 'vlm' not in services
    assert {profile for spec in services.values() for profile in spec.get('profiles', [])} == {'local-models'}
    for name in active:
        assert set(services[name].get('depends_on', {})) <= active


@pytest.mark.parametrize('with_models,existing_url,existing_model,port,expected_url', [
    (False, None, None, '8105', ''),
    (True, None, None, '8105', ''),
    (True, 'http://127.0.0.1:8105/v1', 'qwen3-vl-8b-instruct-fp8', '8105', ''),
    (True, 'http://localhost:8105/v1/', 'qwen3-vl-8b-instruct-fp8', '8105', ''),
    (True, 'http://127.0.0.1:18105/v1/', 'qwen3-vl-8b-instruct-fp8', '18105', ''),
    (True, 'https://vision.example.test/v1', 'qwen3-vl-8b-instruct-fp8', '8105', 'https://vision.example.test/v1'),
    (True, 'http://127.0.0.1:8105/v1', 'custom-vision', '8105', 'http://127.0.0.1:8105/v1'),
    (True, 'http://127.0.0.1:18106/v1', 'qwen3-vl-8b-instruct-fp8', '8105', 'http://127.0.0.1:18106/v1'),
])
def test_deploy_writes_wait_target_and_managed_services(tmp_path, deploy_functions, with_models, existing_url, existing_model, port, expected_url):
    compose = tmp_path/'compose'
    compose.mkdir()
    (compose/'.env.example').write_text((ROOT/'deployment/compose/.env.example').read_text())
    (compose/'.env').write_text(f'NEO4J_PASSWORD=synthetic\nQWEN3_VL_INSTRUCT_PORT={port}\n')
    config = tmp_path/'kb.env'
    if existing_url:
        config.write_text('NEO4J_PASSWORD=synthetic\nKB_PARSE_ENABLED=1\nKB_CONSOLE_SERVICES=database,mineru,embedding,visual_embedding, VLM ,reranker\n'
                          f'VLM_BASE_URL={existing_url}\nVLM_MODEL_ID={existing_model}\nVLM_API_KEY=synthetic-key\n')
    command = '''source "$1"
ROOT="$2"; COMPOSE_DIR="$3"; COMPOSE_ENV="$3/.env"; APP_ENV="$4"
WITH_MODELS="$5"
gpu_usable() { return 0; }
hub_image() { printf '%s' "$1"; }
migrate_legacy_vlm
write_compose_env
write_app_config
# A second run must preserve the migrated configuration.
migrate_legacy_vlm
write_compose_env
write_app_config
'''
    subprocess.run(['bash','-c', command, 'test', str(deploy_functions), str(ROOT), str(compose), str(config), str(int(with_models))],check=True,capture_output=True,text=True)
    def env(path):
        return dict(line.split('=',1) for line in path.read_text().splitlines() if '=' in line and not line.startswith('#'))
    c,a=env(compose/'.env'),env(config)
    assert c['COMPOSE_PROFILES'] == ('local-models' if with_models else '')
    assert c['MINERU_WAIT_FOR_URL'] == ('http://vl-reranker:8000/health' if with_models else '')
    assert not any(key.startswith('QWEN3_VL_INSTRUCT_') for key in c)
    assert a['KB_CONSOLE_SERVICES'] == ('database,mineru,embedding,visual_embedding,reranker' if with_models else 'database,mineru')
    assert a['KB_PARSE_ENABLED'] == '1'
    assert a.get('VLM_BASE_URL', '') == expected_url
    if not expected_url:
        assert a.get('VLM_MODEL_ID', '') == a.get('VLM_API_KEY', '') == ''
        if existing_url is None:
            assert {'VLM_BASE_URL', 'VLM_MODEL_ID', 'VLM_API_KEY'}.isdisjoint(a)
    elif existing_url:
        assert a['VLM_MODEL_ID'] == existing_model
        assert a['VLM_API_KEY'] == 'synthetic-key'


def test_container_only_upgrade_keeps_metadata_for_later_app_migration(tmp_path, deploy_functions):
    compose = tmp_path/'compose.env'
    compose.write_text('QWEN3_VL_INSTRUCT_PORT=18105\nQWEN3_VL_INSTRUCT_MODEL_PATH=synthetic-path\n')
    app = tmp_path/'app.env'
    original = ('KB_CONSOLE_SERVICES=database,mineru,vlm\n'
                'VLM_BASE_URL=http://localhost:18105/v1/\n'
                'VLM_MODEL_ID=qwen3-vl-8b-instruct-fp8\nVLM_API_KEY=synthetic\n')
    app.write_text(original)
    command = '''source "$1"
COMPOSE_ENV="$2"; APP_ENV="$3"; NO_APP="$4"
migrate_legacy_vlm
'''
    def migrate(no_app):
        subprocess.run(['bash','-c',command,'test',str(deploy_functions),str(compose),str(app),str(no_app)],check=True,capture_output=True,text=True)
    migrate(1)
    assert app.read_text() == original
    assert 'QWEN3_VL_INSTRUCT_PORT=18105' in compose.read_text()
    migrate(0)
    assert 'QWEN3_VL_INSTRUCT_' not in compose.read_text()
    assert app.read_text() == 'KB_CONSOLE_SERVICES=database,mineru\nVLM_BASE_URL=\nVLM_MODEL_ID=\nVLM_API_KEY=\n'


@pytest.mark.parametrize('with_models', [False, True])
def test_container_start_removes_orphans_and_waits_only_for_selected_services(tmp_path, deploy_functions, with_models):
    compose_env = tmp_path/'compose.env'
    compose_env.write_text('MINERU_FLAVOR=gpu\n')
    calls = tmp_path/'calls'
    command = '''source "$1"
COMPOSE_ENV="$2"; WITH_MODELS="$3"; CALLS="$4"
prepare_dirs() { :; }
verify_gpu_in_container() { :; }
compose() { printf 'compose %s\\n' "$*" >> "$CALLS"; }
wait_healthy() { printf 'wait %s\\n' "$*" >> "$CALLS"; }
containers_up
'''
    subprocess.run(['bash','-c',command,'test',str(deploy_functions),str(compose_env),str(int(with_models)),str(calls)],check=True,capture_output=True,text=True)
    lines = calls.read_text().splitlines()
    assert 'compose up -d --remove-orphans' in lines
    waited = next(line.split()[2:] for line in lines if line.startswith('wait '))
    expected = {'carrel-qdrant','carrel-opensearch','carrel-neo4j','carrel-mineru'}
    if with_models:
        expected |= {'carrel-embedding','carrel-reranker','carrel-vl-embedding','carrel-vl-reranker'}
    assert set(waited) == expected
