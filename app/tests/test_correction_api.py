"""Correction publication exercises actual files, revision checks and serialized writers."""
import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest

from kb_pipeline import db
from kb_pipeline.graph.corrections import ensure_corrections_file, load_corrections, effective_sha
from kb_pipeline.graph.extract import ExtractionSchema
from kb_server import governance
from _support import _unit


@pytest.fixture
def env(tmp_path, monkeypatch):
    source = SimpleNamespace(kb_id="test", source_root="Example", collection="test")
    settings = SimpleNamespace(state_db=tmp_path / "state.db", corrections_dir=tmp_path / "corrections",
                               sources={"test": source})
    db.init_db(settings.state_db)
    ensure_corrections_file(settings, "test", folder="Example", fresh=True)
    unit = replace(_unit("u", "d"), text="A person called Avery leads Team X.")
    graph = {"entities": [{"key": "avery", "title": "Avery", "type": "object", "unit_ids": ["u"], "doc_ids": ["d"]}]}
    schema = ExtractionSchema(("object", "person"))
    current = ["v1", graph, schema, [unit]]
    monkeypatch.setattr(governance, "_current", lambda *a: tuple(current))
    get = governance.corrections(settings, "test")
    payload = {"id": "r1", "action": "entity.type", "targets": [{"key": "avery"}], "value": "person",
               "base_graph_version": "v1", "submitted_by": "reviewer", "evidence": ["document:d"],
               "revision": get["revision"], "library_identity": get["identity"]}
    return settings, payload, current


def test_preview_commit_duplicate_and_revoke(env):
    settings, payload, _ = env
    path = settings.corrections_dir / "test.yaml"
    original = path.read_bytes()
    assert governance.corrections(settings, "test", payload=payload)["status"] == "ready"
    assert path.read_bytes() == original
    submitted = governance.corrections(settings, "test", payload=payload, commit=True)
    assert effective_sha(load_corrections(path))
    assert governance.corrections(settings, "test", payload=payload, commit=True)["duplicate"]
    revoked = governance.corrections(settings, "test", payload={"library_identity": payload["library_identity"],
        "revision": submitted["revision"], "submitted_by": "reviewer", "reason": "new evidence"}, revoke="r1")
    assert revoked["record"]["state"] == "revoked"
    assert not effective_sha(load_corrections(path))
    assert len(governance.corrections(settings, "test")["records"]) == 1


@pytest.mark.parametrize("change", ["version", "revision", "identity", "target", "type", "evidence"])
def test_invalid_preconditions_never_write(env, change):
    settings, payload, current = env
    path = settings.corrections_dir / "test.yaml"
    original = path.read_bytes()
    if change == "version": current[0] = "v2"
    if change == "revision": payload["revision"] = "old"
    if change == "identity": payload["library_identity"] = "old-library"
    if change == "target": payload["targets"] = [{"key": "missing"}]
    if change == "type": payload["value"] = "missing-type"
    if change == "evidence": payload["evidence"] = []
    with pytest.raises(ValueError):
        governance.corrections(settings, "test", payload=payload, commit=True)
    assert path.read_bytes() == original


def test_concurrent_writes_have_one_winner(env):
    settings, payload, _ = env
    def submit(i):
        try:
            governance.corrections(settings, "test", payload={**payload, "id": f"r{i}"}, commit=True)
            return True
        except ValueError:
            return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(submit, [1, 2])) == [False, True]
    assert len(governance.corrections(settings, "test")["records"]) == 1


def test_reused_library_id_rejects_old_submission(env):
    settings, payload, _ = env
    ensure_corrections_file(settings, "test", folder="Unrelated", fresh=True)
    with pytest.raises(ValueError, match="identity"):
        governance.corrections(settings, "test", payload=payload, commit=True)


def test_duplicate_id_with_changed_content_is_not_idempotent(env):
    settings, payload, _ = env
    governance.corrections(settings, "test", payload=payload, commit=True)
    with pytest.raises(ValueError, match="different content"):
        governance.corrections(settings, "test", payload={**payload, "value": "object"}, commit=True)
