"""Public search client: quality cues, retained evidence, and source-image references."""
import copy
import importlib.util
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def client():
    path = Path(__file__).resolve().parents[2] / "skills/carrel-search/scripts/carrel_search.py"
    spec = importlib.util.spec_from_file_location("carrel_search_client", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_uncertain_readings_override_a_confident_hint(client):
    row = {"subject": "Pump", "property": "pressure", "value": 25, "unit": "kPa",
           "hint": "Pump pressure is certainly 25 kPa", "confidence": "low", "comparable": False,
           "evidence_conflict": {"text_value": 28, "model_value": 25}}
    text = client.fact_line(row)
    assert row["hint"] not in text
    assert "text reading:28" in text and "extracted reading:25" in text
    assert "Awaiting verification" in text and "not directly comparable" in text
    assert "low confidence" in text


def test_missing_conflict_details_do_not_invent_a_second_reading(client):
    row = {"subject": "Pump", "property": "pressure", "value": 25, "unit": "kPa",
           "conflict": True, "quality": "ambiguous_source", "confidence": "low"}
    text = client.fact_line(row)
    assert "original extracted value: 25" in text
    assert "text reading" not in text
    assert "Awaiting verification" in text


@pytest.mark.parametrize("operation", ["search", "facts"])
def test_history_summary_is_retained_without_exposing_unqualified_values(client, operation):
    row = {"n": 1, "subject": "Pump", "property": "pressure", "value": 0, "unit": "kPa",
           "series_text": "938271 (2024), 0 (2025)"}
    result = {"specs": [row], "sources": [], "retrieval_summary": {}} if operation == "search" else {"facts": [row]}
    envelope = {"call_id": "review", "operation": operation, "result": result, "request": {}}
    before = copy.deepcopy(envelope)
    text = "\n".join((client.view_search if operation == "search" else client.view_facts)(envelope))
    assert "= 0 kPa" in text
    assert "938271" not in text and "Historical series" in text
    assert envelope == before


@pytest.mark.parametrize("method,status,expected", [
    ("rrf", "ok", "text/image fusion"),
    ("rrf", "partial", "partial image rerank failure"),
    ("text", "skipped", "image reranking unavailable"),
])
def test_compact_view_reports_fusion_and_fallback(client, method, status, expected):
    envelope = {"call_id": "review", "operation": "search", "request": {},
                "result": {"sources": [], "retrieval_summary": {
                    "ranking": {"method": method, "visual": {"status": status}}}}}
    assert expected in "\n".join(client.view_search(envelope))


@pytest.mark.parametrize("command", ["image", "crop"])
def test_search_reference_fetches_an_image_with_provenance(client, tmp_path, monkeypatch, capsys, command):
    # Synthetic evidence and an in-process HTTP service exercise the public CLI's
    # configuration, authentication, cached references and binary output together.
    for key in ("CARREL_SEARCH_BASE_URL", "CARREL_SEARCH_TOKEN", "KB_SEARCH_TOKEN", "CARREL_SEARCH_WORK_DIR"):
        monkeypatch.delenv(key, raising=False)
    pixels = io.BytesIO()
    Image.new("RGB", (12, 8), "white").save(pixels, format="PNG")
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, self.headers.get("Authorization"), body))
            if self.path == "/crop":
                return self.send_image()
            data = {"kbs": ["example"], "kb_names": {"example": "Sample Library"},
                    "sources": [{"n": 1, "role": "hit", "kb_id": "example", "point_id": "image-point",
                                 "doc_id": "document", "content_version": "v1", "chunk_index": 0,
                                 "rel_path": "Design overview.pdf", "place": "page 2", "text": "Example workflow"}],
                    "retrieval_summary": {"evidence_state": "accepted"}}
            raw = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            calls.append((self.path, self.headers.get("Authorization"), None))
            self.send_image()

        def send_image(self):
            self.send_response(200)
            for key, value in {"Content-Type": "image/png", "X-Image-Source": "cache",
                               "X-Image-Width": "12", "X-Image-Height": "8"}.items():
                self.send_header(key, value)
            if self.path == "/crop":
                self.send_header("X-Crop-Box", "0,0,6,8")
            self.end_headers()
            self.wfile.write(pixels.getvalue())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    token = tmp_path / "token"
    token.write_text("synthetic-test-token")
    config = tmp_path / "client.json"
    config.write_text(json.dumps({"base_url": f"http://127.0.0.1:{server.server_port}",
                                  "token_file": str(token), "work_dir": str(tmp_path / "client work")}))
    base = ["--config", str(config)]
    try:
        assert client.main(base + ["search", "--question", "Example workflow", "--json"]) == 0
        result = json.loads(capsys.readouterr().out)
        args = base + [command, "--ref", result["call_id"] + ":S1", "--json"]
        if command == "crop":
            args += ["--bbox", "0,0,0.5,1"]
        assert client.main(args) == 0
        output = capsys.readouterr()
        image = json.loads(output.out)["result"]
        path = Path(image["path"])
        assert path.read_bytes() == pixels.getvalue()
        assert image["source_file"] == "Sample Library/Design overview.pdf"
        assert "page 2" in image["source_cite"]
        assert "(<" + str(path) + ">)" in image["display_markdown"]
        assert ("crop" in image["display_markdown"].lower()) == (command == "crop")
        assert all(call[1] == "Bearer synthetic-test-token" for call in calls)
        assert "synthetic-test-token" not in output.out + output.err
        assert calls[-1][0] == ("/crop" if command == "crop" else "/image/example/image-point")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize('start,end', [(0,20),(100,125)])
def test_excerpt_provenance_keeps_evidence_and_quality(client,start,end):
    row={'text':'Original evidence, unchanged', 'text_truncated':True,
         'excerpt':{'char_start':start,'char_end':end},'accepted':False,
         'doc_status':'superseded','degraded':'parser fallback'}
    before=copy.deepcopy(row)
    text=' '.join(client.source_notes(row,'abcdef:S1'))
    assert f'{start}–{end}' in text
    assert 'context --ref abcdef:S1' in text
    assert client.TXT['not_accepted'] in text
    assert client.TXT['status_doc_superseded'] in text
    assert 'parser fallback' in text
    assert row==before


@pytest.mark.parametrize('excerpt',[None,{}, {'char_start':-1,'char_end':20},
    {'char_start':2,'char_end':2},{'char_start':True,'char_end':8},
    {'char_start':'2','char_end':20},'invalid',[1,2]])
def test_missing_or_invalid_excerpt_retains_readback_cue(client,excerpt):
    row={'text':'Source','text_truncated':True,'excerpt':excerpt}
    assert client.TXT['truncated'].format(ref='abcdef:S2') in client.source_notes(row,'abcdef:S2')


def test_full_source_does_not_acquire_excerpt_label(client):
    row={'text':'Full source','excerpt':{'char_start':0,'char_end':11}}
    assert not client.source_notes(row,'abcdef:S2')


@pytest.mark.parametrize('command,extra',[('search',['--question','Pump pressure']),
    ('neighbors',['--entity','Pump']),('facts',['--subject','Pump'])])
@pytest.mark.parametrize('scope,field',[(['--history'],'history'),(['--as-of','2024'],'as_of')])
def test_time_and_library_scope_are_preserved(client,command,extra,scope,field):
    args=client.parser().parse_args([command,'--kb','example']+extra+scope)
    result=client.payload(args)
    assert result.get('kbs',result.get('kb_id')) in (['example'],'example')
    assert result.get(field) == (True if field=='history' else '2024')


def test_document_and_phrase_references_preserve_scope_and_version(client, tmp_path, monkeypatch, capsys):
    calls = []
    work = tmp_path / 'cache'
    config = tmp_path / 'client.json'
    config.write_text(json.dumps({'base_url': 'http://unused', 'work_dir': str(work)}))
    for key in ('CARREL_SEARCH_BASE_URL', 'CARREL_SEARCH_WORK_DIR', 'CARREL_SEARCH_TOKEN', 'KB_SEARCH_TOKEN'):
        monkeypatch.delenv(key, raising=False)
    def request(url, token, timeout, method, route, data=None):
        calls.append((method, route, copy.deepcopy(data)))
        common = {'kb_id': 'example', 'kb_name': 'Sample Library'}
        if route == '/docs':
            result = dict(common, docs=[{'n': 9, 'doc_id': 'doc', 'rel_path': 'folder/Guide.md', 'indexed': True,
                                        'content_version': 'v8', 'chunk_total': 4}], total=1, count=1, limit=10, offset=8)
        elif route == '/grep':
            result = dict(common, time={'mode': 'as_of', 'day': '2024-12-31'}, results=[{'phrase': 'term',
                'total_chunks': 1, 'hits': [{'point_id': 'p', 'doc_id': 'doc', 'rel_path': 'folder/Guide.md',
                    'chunk_index': 2, 'content_version': 'v8', 'literal': True, 'snippet': 'term', 'field': 'body'}]}])
        else:
            result = dict(common, sources=[], retrieval_summary={})
        return json.dumps(result).encode(), {}
    monkeypatch.setattr(client, 'request', request)
    base = ['--config', str(config)]
    assert client.main(base + ['docs', '--kb', 'example', '--name', 'Guide', '--offset', '8', '--limit', '10', '--json']) == 0
    d = json.loads(capsys.readouterr().out)
    dref = d['call_id'] + ':D9'
    assert client.main(base + ['grep', '--phrase', 'term', '--in-doc', dref, '--as-of', '2024', '--json']) == 0
    g = json.loads(capsys.readouterr().out)
    assert calls[-1] == ('POST', '/grep', {'kb_id': 'example', 'doc_ids': ['doc'], 'phrases': ['term'], 'as_of': '2024', 'limit': 8})
    gref = g['call_id'] + ':G1'
    assert client.main(base + ['context', '--ref', gref, '--json']) == 0
    capsys.readouterr()
    assert calls[-1] == ('POST', '/context', {'kb_id': 'example', 'doc_id': 'doc', 'chunk_from': 1,
                                             'chunk_to': 3, 'content_version': 'v8'})
    assert client.main(base + ['search', '--question', 'Meaning?', '--in-doc', dref, '--json']) == 0
    capsys.readouterr()
    assert calls[-1][2] == {'question': 'Meaning?', 'kbs': ['example'], 'hints': {'doc_ids': ['doc']}}
    count = len(calls)
    assert client.main(base + ['grep', '--kb', 'other', '--in-doc', dref, '--phrase', 'term']) == 1
    capsys.readouterr()
    assert len(calls) == count
    assert client.main(base + ['show', dref, gref]) == 0
    capsys.readouterr()
    assert len(calls) == count


def test_phrase_labels_are_unique_across_phrases_and_quality_survives(client):
    r = {'kb_id': 'example', 'kb_name': 'Sample Library', 'degraded': ['index_missing'],
         'time': {'mode': 'history'}, 'results': [
             {'phrase': 'one', 'total_chunks': 4, 'docs_truncated': True, 'hits': [
                 {'rel_path': 'a.pdf', 'field': 'visual', 'literal': False, 'doc_status': 'archived', 'snippet': 'first'}]},
             {'phrase': 'two', 'hits': [{'rel_path': 'b.md', 'literal': True, 'snippet': 'second'}]}]}
    client.number_rows(r)
    client.add_cites(r)
    before = copy.deepcopy(r)
    text = '\n'.join(client.view_grep({'call_id': 'abcdef', 'operation': 'grep', 'request': {}, 'result': r}))
    assert '[G1]' in text and '[G2]' in text
    for value in ('index_missing', client.TXT['grep_approx'], client.TXT['picture_text'],
                  client.TXT['status_doc_archived'], client.TXT['grep_docs_truncated'], 'first', 'second'):
        assert value in text
    assert r == before


def test_old_keyword_server_is_identified_without_inventing_time_scope(client):
    text = '\n'.join(client.view_grep({'call_id': 'abcdef', 'operation': 'grep', 'request': {}, 'result': {'results': []}}))
    assert client.TXT['grep_legacy'] in text


def test_keyword_image_body_and_future_document_keep_provenance(client):
    row = {'n': 1, 'block_type': 'image', 'field': 'body', 'snippet': 'Extracted diagram description',
           'literal': True, 'doc_status': 'not_started'}
    e = {'call_id': 'abcdef', 'operation': 'grep', 'request': {},
         'result': {'time': {'mode': 'history'}, 'results': [{'hits': [row]}]}}
    text = '\n'.join(client.view_grep(e))
    assert client.TXT['picture_text'] in text
    assert 'source document not yet in effect today' in text
    assert 'source document not yet in effect today' in client.source_notes(row, 'abcdef:S1')


@pytest.mark.parametrize('operation,label', [('docs','D1'),('grep','G1')])
def test_empty_navigation_has_no_dangling_reference(client, operation, label):
    envelope = {'call_id': 'abcdef', 'operation': operation, 'request': {},
                'result': {'docs': [], 'results': [{'hits': [], 'total_chunks': 0}], 'total': 0}}
    text = '\n'.join(client.VIEWS[operation](envelope))
    assert 'abcdef:' + label not in text


@pytest.mark.parametrize('request_limit,cli_limit,expected', [
    (None, None, 8), (None, 0, 0), (None, 30, 30), (17, None, 17), (17, 2, 2),
])
def test_phrase_limit_default_preserves_explicit_requests(client, tmp_path, request_limit, cli_limit, expected):
    args = ['grep', '--kb', 'example', '--phrase', 'term']
    if request_limit is not None:
        path = tmp_path / 'request.json'
        path.write_text(json.dumps({'limit': request_limit}))
        args += ['--request', str(path)]
    if cli_limit is not None:
        args += ['--limit', str(cli_limit)]
    assert client.payload(client.parser().parse_args(args))['limit'] == expected
