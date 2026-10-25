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
