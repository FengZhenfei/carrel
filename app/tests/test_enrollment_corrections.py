"""A new library must never become visible with a previous library's rules."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
import sqlite3
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from kb_pipeline import db, discovery
from kb_pipeline.graph import corrections
from kb_server import service
from kb_server.main import create_app


OLD_RULES = "kb:\n  id: kb_001\n  folder: Previous Library\nentities:\n  reject: [Shared Entity]\n"


class Interrupted(BaseException):
    """Simulate a process interruption, not a recoverable external-index error."""


@pytest.fixture
def env(tmp_path, monkeypatch):
    cfg = SimpleNamespace(
        state_db=tmp_path / "state.sqlite", mirror_root=tmp_path / "mirror",
        corrections_dir=tmp_path / "corrections", embedding_base_url="http://invalid.test",
        qdrant_url="http://invalid.test", qdrant_api_key="", vector_layout=None,
        opensearch_url="http://invalid.test",
    )
    for name in ("New Library", "Other Library"):
        (cfg.mirror_root / name).mkdir(parents=True)
    db.init_db(cfg.state_db)
    cfg.corrections_dir.mkdir()
    path = cfg.corrections_dir / "kb_001.yaml"
    path.write_text(OLD_RULES, encoding="utf-8")
    monkeypatch.setattr(service, "settings", lambda: cfg)
    monkeypatch.setattr(service, "chunk_limits", lambda _: {
        "max_tokens_cap": 3276, "max_tokens_min": 128,
        "effective_max_model_len": 4096, "embedding_max_model_len": 4096,
    })
    effects = {}
    for owner, name in ((service, "qdrant_client"), (service, "ensure_collection"),
                        (service.search_fts, "ensure_indices"), (service, "kick_scan"),
                        (service, "kick_worker")):
        effects[name] = Mock(return_value=True)
        monkeypatch.setattr(owner, name, effects[name])
    monkeypatch.delenv("KB_WEB_TOKEN", raising=False)
    return SimpleNamespace(cfg=cfg, path=path, effects=effects)


def registry(env):
    with db.connect(env.cfg.state_db) as con:
        return [dict(row) for row in discovery.known_sources(con)]


def assert_unpublished(env):
    assert registry(env) == []
    with db.connect(env.cfg.state_db) as con:
        assert con.execute("SELECT value FROM app_config WHERE key='kb_seq'").fetchone() is None
    for effect in env.effects.values():
        effect.assert_not_called()


def assert_clean_rules(env):
    corr = corrections.load_corrections(env.path)
    assert not corr.get("errors")
    assert corrections.is_empty(corr)
    entities = [{"key": corrections.entity_key("Shared Entity"), "title": "Shared Entity",
                 "type": "entity", "aliases": [], "unit_ids": [], "descriptions": []}]
    kept, _, stats, _ = corrections.apply_entity_corrections(entities, [], corr, known_types=["entity"])
    assert len(kept) == 1
    assert stats["rejected_entities"] == 0


@pytest.mark.parametrize("stage", ["archive", "write", "publish"])
def test_io_failure_is_visible_and_retry_cannot_inherit_rules(env, monkeypatch, stage):
    with monkeypatch.context() as fault:
        if stage == "archive":
            fault.setattr(corrections, "_set_aside", Mock(side_effect=PermissionError("read-only archive")))
        elif stage == "write":
            original_write = Path.write_text

            def fail_write(path, *args, **kwargs):
                if path.name == "kb_001.yaml.tmp":
                    raise OSError("disk full")
                return original_write(path, *args, **kwargs)

            fault.setattr(Path, "write_text", fail_write)
        else:
            original_replace = corrections.os.replace

            def fail_publish(src, dst):
                if Path(src).name == "kb_001.yaml.tmp":
                    raise OSError("cannot publish template")
                return original_replace(src, dst)

            fault.setattr(corrections.os, "replace", fail_publish)
        # The real API must report a failed operation, not a success notification.
        response = TestClient(create_app()).post("/api/enroll", json={
            "dir": "New Library", "config": {"max_tokens": 600, "overlap_tokens": 90}})
        assert response.status_code == 422
        assert "correction rules" in response.json()["detail"]
        assert_unpublished(env)
    if stage == "archive":
        assert env.path.read_text(encoding="utf-8") == OLD_RULES
    else:
        assert not env.path.exists()
        assert [p.read_text(encoding="utf-8") for p in
                (env.cfg.corrections_dir / "orphaned").glob("*.yaml")] == [OLD_RULES]
    result = service.enroll("New Library")
    assert (result["outcome"], result["kb_id"]) == ("new", "kb_001")
    assert_clean_rules(env)


@pytest.mark.parametrize("after_prepare", [False, True])
def test_interruption_before_commit_rolls_back_and_retry_is_safe(env, monkeypatch, after_prepare):
    prepare = corrections.ensure_corrections_file

    def interrupt(*args, **kwargs):
        if after_prepare:
            prepare(*args, **kwargs)
        raise Interrupted()

    with monkeypatch.context() as fault:
        fault.setattr(corrections, "ensure_corrections_file", interrupt)
        with pytest.raises(Interrupted):
            service.enroll("New Library")
        assert_unpublished(env)
    assert service.enroll("New Library")["outcome"] == "new"
    assert_clean_rules(env)


def test_failed_commit_does_not_restore_old_rules(env, monkeypatch):
    connect = db.connect

    @contextmanager
    def fail_commit(path):
        with connect(path) as con:
            yield con
            if con.in_transaction and con.execute("SELECT count(*) FROM kb_sources").fetchone()[0]:
                # Deny the actual COMMIT: sqlite must roll back before closing.
                con.set_authorizer(lambda action, arg, *_: sqlite3.SQLITE_DENY
                                   if action == sqlite3.SQLITE_TRANSACTION and arg == "COMMIT"
                                   else sqlite3.SQLITE_OK)

    with monkeypatch.context() as fault:
        fault.setattr(db, "connect", fail_commit)
        with pytest.raises(sqlite3.DatabaseError):
            service.enroll("New Library")
    assert_unpublished(env)
    assert_clean_rules(env)
    assert service.enroll("New Library")["outcome"] == "new"
    assert_clean_rules(env)


def test_rules_are_prepared_before_registration_is_visible(env, monkeypatch):
    prepare = corrections.ensure_corrections_file

    def inspect_prepare(*args, **kwargs):
        assert registry(env) == []  # an independent connection cannot see a partial enrollment
        return prepare(*args, **kwargs)

    def inspect_external_index(*args, **kwargs):
        assert registry(env)[0]["status"] == "active"
        assert_clean_rules(env)
        return True

    monkeypatch.setattr(corrections, "ensure_corrections_file", inspect_prepare)
    env.effects["ensure_collection"].side_effect = inspect_external_index
    assert service.enroll("New Library")["outcome"] == "new"


def test_interruption_after_commit_leaves_only_new_rules(env):
    env.effects["ensure_collection"].side_effect = Interrupted()
    with pytest.raises(Interrupted):
        service.enroll("New Library")
    assert registry(env)[0]["status"] == "active"
    assert_clean_rules(env)
    env.effects["ensure_collection"].side_effect = None
    assert service.enroll("New Library")["outcome"] == "already"
    assert_clean_rules(env)


@pytest.mark.parametrize("second_name", ["New Library", "Other Library"])
def test_concurrent_enrollments_do_not_publish_partial_state(env, monkeypatch, second_name):
    prepare = corrections.ensure_corrections_file
    preparing, release, second_started = Event(), Event(), Event()

    def hold_first(*args, **kwargs):
        if not preparing.is_set():
            preparing.set()
            assert release.wait(10), "test did not release enrollment"
        return prepare(*args, **kwargs)

    def second():
        second_started.set()
        return service.enroll(second_name)

    monkeypatch.setattr(corrections, "ensure_corrections_file", hold_first)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(service.enroll, "New Library")
        try:
            assert preparing.wait(10)
            other = pool.submit(second)
            assert second_started.wait(10)
            assert registry(env) == []
        finally:
            release.set()
        results = [first.result(timeout=10), other.result(timeout=10)]
    assert results[0]["outcome"] == "new"
    if second_name == "New Library":
        assert results[1]["outcome"] == "already"
        assert [r["kb_id"] for r in registry(env)] == ["kb_001"]
    else:
        assert results[1]["outcome"] == "new"
        assert [r["kb_id"] for r in registry(env)] == ["kb_001", "kb_002"]
        assert "Other Library" in (env.cfg.corrections_dir / "kb_002.yaml").read_text(encoding="utf-8")
    assert_clean_rules(env)
    assert len(list((env.cfg.corrections_dir / "orphaned").glob("*.yaml"))) == 1


def test_existing_library_retains_its_own_rules_on_save_and_reactivation(env):
    service.enroll("New Library")
    own_rules = OLD_RULES.replace("Previous Library", "New Library")
    env.path.write_text(own_rules, encoding="utf-8")
    assert service.enroll("New Library")["outcome"] == "already"
    assert env.path.read_text(encoding="utf-8") == own_rules
    with db.connect(env.cfg.state_db) as con:
        discovery.mark_inactive(con, "kb_001", reason="user_disabled")
    assert service.enroll("New Library")["outcome"] == "reactivated"
    assert env.path.read_text(encoding="utf-8") == own_rules


def test_reactivation_failure_keeps_status_and_config(env, monkeypatch):
    service.enroll("New Library", {"max_tokens": 600, "overlap_tokens": 90})
    with db.connect(env.cfg.state_db) as con:
        discovery.mark_inactive(con, "kb_001", reason="user_disabled")
    before = registry(env)
    env.path.unlink()  # a missing template must be recreated before reactivation
    for effect in env.effects.values():
        effect.reset_mock()
    monkeypatch.setattr(corrections, "ensure_corrections_file", Mock(side_effect=OSError("disk full")))
    with pytest.raises(ValueError, match="correction rules"):
        service.enroll("New Library", {"max_tokens": 800})
    assert registry(env) == before
    for effect in env.effects.values():
        effect.assert_not_called()
