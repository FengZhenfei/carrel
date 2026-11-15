"""Producer versions must follow each response, including historical cache reads."""
import json
from unittest.mock import Mock

import pytest

from kb_pipeline.parsers import mineru_docx, mineru_pdf, mineru_pptx, service_clients
from kb_pipeline.parsers.service_clients import mineru_parser_profile


@pytest.mark.parametrize("payload, expected", [
    ({"version": "3.4.5"}, "mineru-3.4.5-pdf-v3"),
    ({"version": "3.4.4"}, "mineru-3.4.4-pdf-v3"),
    ({"_kb_mineru": {"version": "4.2.2"}, "version": "1.0.0"}, "mineru-4.2.2-pdf-v3"),
    ({"_kb_mineru": "invalid", "version": "3.4.5"}, "mineru-3.4.5-pdf-v3"),
    ({"version": "invalid"}, "mineru-3.4.4-pdf-v3"),
    ({}, "mineru-3.4.4-pdf-v3"),
])
def test_native_response_and_cached_producer_versions(payload, expected):
    assert mineru_parser_profile(payload, "mineru-3.4.4-pdf-v3") == expected


@pytest.mark.parametrize("module, extension, revision", [
    (mineru_pdf, "pdf", "v3"),
    (mineru_docx, "docx", "v1"),
    (mineru_pptx, "pptx", "v1"),
])
@pytest.mark.parametrize("version", ["3.4.4", "3.4.5"])
@pytest.mark.parametrize("structured", [True, False], ids=["content-list", "markdown"])
def test_producer_version_survives_conversion_and_slide_merge(
    module, extension, revision, version, structured, monkeypatch, tmp_path
):
    text = "Synthetic source evidence remains unchanged."
    payload = {"version": version}
    if structured:
        payload["content_list"] = [{"type": "text", "text": text, "page_idx": 0}]
    else:
        payload["md_content"] = text
    monkeypatch.setattr(module, "call_mineru_sync", lambda *a, **k: payload)
    path = tmp_path / f"input.{extension}"
    blocks = getattr(module, f"mineru_{extension}_blocks")(
        mineru_url="http://example.invalid", path=path, cache_dir=tmp_path
    )
    assert blocks
    assert text in "\n".join(block.text for block in blocks)
    expected = f"mineru-{version}-{extension}-{revision}"
    assert {block.parser_profile for block in blocks} == {expected}
    if extension == "pptx" and structured:
        merged = mineru_pptx.aggregate_mineru_slide_blocks(blocks, path)
        assert len(merged) == 1
        assert merged[0].parser_profile == expected + "+slide-merge-v1"
        assert text in merged[0].text


def test_cache_reuse_keeps_old_producer_after_service_upgrade(monkeypatch, tmp_path):
    source = tmp_path / "input.pdf"
    source.write_bytes(b"synthetic unchanged source")
    output = tmp_path / "result.json"
    old = {"version": "3.4.4", "md_content": "Previously parsed evidence."}
    output.write_text(json.dumps(old), encoding="utf-8")
    monkeypatch.setenv("MINERU_REUSE_CACHE", "1")
    monkeypatch.setenv("MINERU_VERSION", "3.4.5")
    monkeypatch.setenv("MINERU_BACKEND", "pipeline")
    session = Mock(side_effect=AssertionError("cached evidence must not call the service"))
    monkeypatch.setattr(service_clients, "_local_service_session", session)
    payload = service_clients.call_mineru_sync(
        "http://example.invalid", source, output, output.with_suffix(".md")
    )
    assert payload == old
    assert mineru_parser_profile(payload, "mineru-3.4.4-pdf-v3") == "mineru-3.4.4-pdf-v3"
    session.assert_not_called()
