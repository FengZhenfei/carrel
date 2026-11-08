"""Lifecycle tests use synthetic tasks; no production endpoints or model calls."""

import json
import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from kb_runtime import queue, bootstrap
from kb_runtime import __main__ as runner

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def runtime_env(tmp_path, monkeypatch):
    monkeypatch.setattr(
        os,
        "environ",
        {
            "PATH": os.environ["PATH"],
            "KB_LOCAL_BASE_DIR": str(tmp_path),
            "KB_RUNTIME_DIR": str(tmp_path / "runtime"),
            "KB_STATE_DB": str(tmp_path / "runtime/state/kb-pipeline.db"),
            "KB_MIRROR_ROOT": str(tmp_path / "runtime/mirror"),
            "KB_LOG_DIR": str(tmp_path / "logs"),
            "KB_ENV_FILE": str(tmp_path / "config/app.env"),
            "KB_STORAGE_BACKEND": "light",
            "KB_RUNTIME": "container",
            "NEO4J_PASSWORD": "synthetic-test-only",
            "KB_CONFIG_TEMPLATE": str(ROOT / "config/knowledge-base.env.example"),
        },
    )
    return tmp_path


@pytest.mark.parametrize("edition", ["light", "standard"])
def test_initialize_preserves_saved_config_and_fixes_edition(
    runtime_env, monkeypatch, edition
):
    monkeypatch.setenv("KB_STORAGE_BACKEND", edition)
    bootstrap.initialize()
    config = Path(os.environ["KB_ENV_FILE"])
    assert config.stat().st_mode & 0o777 == 0o600
    content = config.read_text()
    assert (
        "EMBEDDING_BASE_URL=\n" in content and "VISUAL_EMBEDDING_ENABLED=0" in content
    )
    assert "KB_SEARCH_TOKEN=\n" not in content
    config.write_text(content + "\nVLM_API_KEY=synthetic-secret\n")
    bootstrap.initialize()
    assert config.read_text() == content + "\nVLM_API_KEY=synthetic-secret\n"
    (runtime_env / "runtime/deployment-mode").unlink()
    monkeypatch.setenv(
        "KB_STORAGE_BACKEND", "standard" if edition == "light" else "light"
    )
    with pytest.raises(ValueError, match="editions differ"):
        bootstrap.initialize()
    assert config.read_text().endswith("VLM_API_KEY=synthetic-secret\n")


def test_queue_coalesces_waiting_rounds_and_retains_one_followup(runtime_env):
    first = queue.enqueue("scan")
    assert queue.enqueue("scan") == first
    with queue.connect() as con:
        con.execute("UPDATE requests SET status='running' WHERE id=?", (first,))
    second = queue.enqueue("scan")
    assert second != first and queue.enqueue("scan") == second
    runner.recover_requests()
    with queue.connect() as con:
        assert (
            con.execute("SELECT status FROM requests WHERE id=?", (first,)).fetchone()[
                0
            ]
            == "interrupted"
        )
        assert (
            con.execute(
                "SELECT COUNT(*) FROM requests WHERE kind='scan' AND status='queued'"
            ).fetchone()[0]
            == 1
        )


def test_manual_graph_dispatch_resume_and_cancel(runtime_env):
    payload = {
        "kb_id": "kb_001",
        "graph_version": "v1",
        "append": False,
        "force": False,
    }
    queue.enqueue("graph-build", payload)
    with pytest.raises(ValueError, match="already running"):
        queue.enqueue("graph-build", payload)
    assert queue.active_graph("kb_001")["status"] == "queued"
    cmd = runner.command("graph-build", payload)
    assert cmd[-3:] == ["--graph-version", "v1", "--allow-existing-graph-version"]
    assert queue.cancel_graph("kb_001")
    assert queue.active_graph() is None
    for invalid in [{"kb_id": "../../invalid"}, {"kb_id": "kb_001", "append": "yes"}]:
        with pytest.raises(ValueError):
            queue.enqueue("graph-build", invalid)
    with pytest.raises(ValueError):
        queue.enqueue("unknown")
    assert (
        runner.command(
            "graph-build", {"kb_id": "kb_001", "append": True, "force": True}
        )[-1]
        == "--force"
    )


def test_child_env_refreshes_saved_values_and_preserves_runtime(
    runtime_env, monkeypatch
):
    p = Path(os.environ["KB_ENV_FILE"])
    p.parent.mkdir()
    p.write_text("EMBEDDING_MODEL_ID=new\nKB_LOCAL_BASE_DIR=wrong\n")
    monkeypatch.setenv("EMBEDDING_MODEL_ID", "old")
    monkeypatch.setenv("KB_RUNTIME_BOOT_TIME", "123")
    env = runner.child_env()
    assert "EMBEDDING_MODEL_ID" not in env
    assert (
        env["KB_LOCAL_BASE_DIR"] == str(runtime_env)
        and env["KB_RUNTIME_BOOT_TIME"] == "123"
    )


def test_runtime_single_flight_catchup_and_web_restart(runtime_env, monkeypatch):
    monkeypatch.setenv("KB_WEB_MODULE", "site_console")
    runtime = runner.Runtime()
    procs = []
    commands = []

    def spawn(_cmd, **kwargs):
        commands.append(_cmd)
        p = Mock(pid=100 + len(procs))
        p.poll.return_value = None
        procs.append(p)
        return p

    with patch.object(runner.subprocess, "Popen", side_effect=spawn):
        queue.enqueue("worker")
        queue.enqueue("gc")
        queue.enqueue("graph-build", {"kb_id": "kb_001"})
        runtime.tick(1000)
        assert commands[0][-2:] == ["-m", "site_console"]
        assert commands[1][-2:] == ["-m", "kb_search"]
        assert len(runtime.children) == 4  # web, search, worker, graph; GC waits
        assert queue.active_graph()["status"] == "running"
        queue.enqueue("worker")
        runtime.tick(1300)
        runtime.tick(1301)
        assert (
            sum(bool(r) and r["kind"] == "worker" for _, r in runtime.children.values())
            == 1
        )
        graph_proc = next(
            p for p, r in runtime.children.values() if r and r["kind"] == "graph-build"
        )
        runtime.children["web"][0].poll.return_value = 1
        runtime.tick(1302)
        runtime.tick(1307)
        assert "web" in runtime.children and graph_proc.poll() is None
        with queue.connect() as con:
            assert (
                con.execute(
                    "SELECT COUNT(*) FROM requests WHERE kind='worker' AND status='queued'"
                ).fetchone()[0]
                == 1
            )
        for p, r in runtime.children.values():
            if r:
                p.poll.return_value = 0
        runtime.tick(1310)
        assert any(r and r["kind"] == "gc" for _, r in runtime.children.values())
        with queue.connect() as con:
            con.execute("UPDATE schedules SET next_due=0 WHERE kind='logs'")
        runtime.tick(1400)
        runtime.tick(1401)
        with queue.connect() as con:
            assert (
                con.execute(
                    "SELECT COUNT(*) FROM requests WHERE kind='logs' AND status='queued'"
                ).fetchone()[0]
                == 1
            )


def test_stopping_marks_shutdown_and_kills_task_groups(runtime_env, monkeypatch):
    from kb_pipeline.graph.build import _shutdown_probe
    from kb_pipeline import db

    runtime = runner.Runtime()
    queue.state_dir().mkdir(parents=True)
    assert not _shutdown_probe()["shutdown"]
    process = Mock(pid=424242)
    process.poll.return_value = 0
    runtime.children = {"web": (process, None)}
    monkeypatch.setenv("KB_SHUTDOWN_GRACE_SECONDS", "0")
    with patch.object(runner.os, "killpg") as kill:
        runtime.stop()
    assert kill.call_args_list[0].args[0] == 424242
    assert _shutdown_probe()["shutdown"]
    assert json.loads((queue.state_dir() / "runtime.json").read_text())["stopping"]
    monkeypatch.setenv("KB_RUNTIME_BOOT_TIME", "123")
    assert db.boot_time() == 123


def test_console_uses_container_dispatch_and_no_host_management(runtime_env):
    from kb_server import service

    with patch.object(
        service.subprocess, "run", side_effect=AssertionError("host command")
    ):
        assert service.kick("carrel-worker.service", min_interval=0) == "container-queue"
        assert service.managed_services() == ()
        assert service._spawn_graph_build(None, "kb_001")["queued"]
    assert queue.active_graph("kb_001")


def test_parser_cleanup_preserves_recent_and_linked_paths(runtime_env):
    root = runtime_env / "runtime/mineru-output"
    root.mkdir(parents=True)
    old = root / "old"
    old.mkdir()
    (old / "asset").write_text("old")
    os.utime(old, (1, 1))
    recent = root / "recent"
    recent.mkdir()
    outside = runtime_env / "original"
    outside.mkdir()
    (outside / "keep").write_text("keep")
    (root / "linked").symlink_to(outside, target_is_directory=True)
    (root / "carrel-mineru.json").write_text("{}")
    bootstrap.cleanup_mineru_output()
    assert not old.exists()
    assert (
        recent.exists()
        and (outside / "keep").exists()
        and (root / "carrel-mineru.json").exists()
    )


def test_pending_graph_pause_keeps_assets_and_operator_intent(runtime_env, monkeypatch):
    from kb_pipeline import db, discovery
    from kb_server import service
    from kb_pipeline.config import load_settings

    bootstrap.initialize()
    cfg = load_settings(os.environ["KB_ENV_FILE"])
    folder = cfg.mirror_root / "Synthetic"
    folder.mkdir(parents=True)
    original = folder / "guide.md"
    original.write_text("Keep the original")
    with db.connect(cfg.state_db) as con:
        source, _ = discovery.enroll(con, cfg.mirror_root, "Synthetic")
        discovery.set_config(con, source.kb_id, {"graph_enabled": True})
    monkeypatch.setattr(service, "settings", lambda: cfg)
    service._spawn_graph_build(cfg, source.kb_id)
    with db.connect(cfg.state_db) as con:
        assert service._graph_status(con, source.kb_id, "Synthetic", {}) == "running"
        assert service._graph_build_info(con, source.kb_id)["queued"]
    assert service.pause_graph_build(source.kb_id)["paused"]
    assert queue.active_graph() is None
    with db.connect(cfg.state_db) as con:
        config = discovery.get_config(con, source.kb_id)
        assert config["graph_paused"] and config["graph_enabled"]
    assert original.read_text() == "Keep the original"


@pytest.mark.parametrize("status,expected", [("done", "ok"), ("failed", "failed"), ("cancelled", "stopped")])
def test_terminal_graph_is_not_queued_while_dispatch_finishes(runtime_env, monkeypatch, status, expected):
    """The child records its outcome before cleanup / exit and the runtime's next poll."""
    from kb_pipeline import db, discovery
    from kb_pipeline.config import load_settings
    from kb_server import service

    bootstrap.initialize()
    cfg = load_settings(os.environ["KB_ENV_FILE"])
    (cfg.mirror_root / "Synthetic").mkdir(parents=True)
    monkeypatch.setattr(service, "settings", lambda: cfg)
    with db.connect(cfg.state_db) as con:
        source, _ = discovery.enroll(con, cfg.mirror_root, "Synthetic")
        discovery.set_config(con, source.kb_id, {"graph_enabled": True})
        bid = db.begin_graph_build(con, source_key=source.kb_id, kb_id=source.kb_id,
                                   source_collection=source.collection, graph_version="v1")
        con.execute("UPDATE graph_builds SET status=?,stage='Done',started_at=100,finished_at=120,worker_pid=700 "
                    "WHERE graph_build_id=?", (status, bid))
    request_id = queue.enqueue("graph-build", {"kb_id": source.kb_id})
    with queue.connect() as con:
        con.execute("UPDATE requests SET status='running',created=100.1,started=100.2,pid=700 WHERE id=?", (request_id,))
    with db.connect(cfg.state_db) as con:
        assert service._graph_status(con, source.kb_id, "Synthetic", {"graph_enabled": True}) == expected
        info = service._graph_build_info(con, source.kb_id)
        assert info["status"] == status and info["graph_version"] == "v1"
        assert not info.get("queued")
        assert con.execute("SELECT COUNT(*) FROM graph_builds").fetchone()[0] == 1
    # The dispatch still owns the process: concurrency and cancellation gates must keep seeing it.
    assert queue.active_graph(source.kb_id)["status"] == "running"


@pytest.mark.parametrize("request_status,stage", [("queued", "Queued"), ("running", "Starting graph build")])
def test_new_graph_dispatch_is_visible_over_previous_success(runtime_env, monkeypatch, request_status, stage):
    from kb_pipeline import db, discovery
    from kb_pipeline.config import load_settings
    from kb_server import service

    bootstrap.initialize()
    cfg = load_settings(os.environ["KB_ENV_FILE"])
    (cfg.mirror_root / "Synthetic").mkdir(parents=True)
    monkeypatch.setattr(service, "settings", lambda: cfg)
    with db.connect(cfg.state_db) as con:
        source, _ = discovery.enroll(con, cfg.mirror_root, "Synthetic")
        bid = db.begin_graph_build(con, source_key=source.kb_id, kb_id=source.kb_id,
                                   source_collection=source.collection, graph_version="v1")
        # Old and new dispatch may start in the same integer second; PID distinguishes them.
        con.execute("UPDATE graph_builds SET status='done',started_at=100,finished_at=100,worker_pid=700 "
                    "WHERE graph_build_id=?", (bid,))
    request_id = queue.enqueue("graph-build", {"kb_id": source.kb_id})
    with queue.connect() as con:
        con.execute("UPDATE requests SET status=?,created=100.3,started=100.4,pid=701 WHERE id=?", (request_status, request_id))
    with db.connect(cfg.state_db) as con:
        assert service._graph_status(con, source.kb_id, "Synthetic", {"graph_enabled": True}) == "running"
        info = service._graph_build_info(con, source.kb_id)
        assert info["stage"] == stage
        assert info["queued"] is (request_status == "queued")
