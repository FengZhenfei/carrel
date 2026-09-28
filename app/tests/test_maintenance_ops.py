"""Maintenance and operations: cleanup jobs, the yield policy, systemd units, scripts."""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from kb_pipeline import db

from _support import _CodexAudit20260906TestsSupport, _CodexFinalTestsSupport, _repo_file


def _render_unit(template: Path) -> str:
    """The installed version is the copy that install-systemd.sh produces by replacing __CARREL_HOME__ with the
    repository path; render by the same rule before comparing, otherwise the template always looks "drifted"."""
    home = template.resolve().parents[2]
    return template.read_text(encoding="utf-8").replace("__CARREL_HOME__", str(home))


class MaintenanceCleanupTests(unittest.TestCase):
    """The cleanup actions had only ever been run on the real box, and they delete things from disk."""

    def test_job_history_pruning_keeps_live_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"; db.init_db(state)
            old = int(time.time()) - 60 * 86400
            with db.connect(state) as con:
                for jid, status, finished in (("done-old", "done", old),
                                              ("done-new", "done", int(time.time())),
                                              ("failed-old", "failed", old),
                                              ("queued", "queued", None)):
                    con.execute(
                        "INSERT INTO jobs(job_id, kb_id, collection, job_type, status, priority, "
                        "next_attempt_at, created_at, updated_at, finished_at) "
                        "VALUES(?, 'kb', 'c', 'parse', ?, 100, 0, 1, ?, ?)",
                        (jid, status, finished or int(time.time()), finished))
                con.commit()
                removed = db.prune_job_history(con, retention_days=30)
                left = {r[0] for r in con.execute("SELECT job_id FROM jobs").fetchall()}
            self.assertEqual(removed["jobs"], 1)          # only long-finished rows are removed
            self.assertEqual(left, {"done-new", "failed-old", "queued"})  # failed rows are kept for troubleshooting

    def test_weekly_cleanup_never_touches_the_vlm_cache(self) -> None:
        """The VLM description cache is reused across versions: letting the weekly rotation move it away means
        every image gets burned through the GPU again."""
        from unittest import mock

        from kb_pipeline import maintenance

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"; (cache / "vlm-cache").mkdir(parents=True)
            (cache / "vlm-cache" / "a.json").write_text("{}", encoding="utf-8")
            (cache / "parse").mkdir(); (cache / "parse" / "keep.txt").write_text("x", encoding="utf-8")
            (cache / "scratch").mkdir(); (cache / "scratch" / "tmp.bin").write_text("x", encoding="utf-8")
            settings = SimpleNamespace(runtime_dir=root, cache_dir=cache, log_dir=root / "logs",
                                       state_db=root / "s.db", qdrant_inactive_retention_days=7)
            (root / "logs").mkdir()
            with mock.patch.object(maintenance, "service_busy", return_value=(False, [])):
                maintenance.weekly_cache_cleanup(settings)
            self.assertTrue((cache / "vlm-cache" / "a.json").exists())   # kept
            self.assertTrue((cache / "parse" / "keep.txt").exists())     # parse assets are kept too
            self.assertFalse((cache / "scratch" / "tmp.bin").exists())   # everything else is rotated away


class ScheduledCleanupCoverageTests(unittest.TestCase):
    """The cleanup subcommands the timers actually run must cover everything that grows monotonically.

    This guards against one specific kind of bug: the code hangs off a function that **no timer ever calls**,
    while the unit name looks as if it did. The graph_stale cleanup slipped through a whole round exactly this
    way: purge_stale lived in qdrant_inactive_gc, while the unit named carrel-qdrant-gc actually ran
    parse-assets-gc in its ExecStart. Code, comments and unit name each made sense on their own; put together,
    nobody did the work.
    """

    # subcommand -> handler function in maintenance, mirroring the dispatch in cli.cmd_cleanup.
    # The first test below checks it against the real dispatch in cli.py; if the routing changes, so must this.
    HANDLERS = {
        "status": "maintenance_status",
        "weekly": "weekly_cache_cleanup",
        "monthly": "monthly_log_cleanup",
        "qdrant-gc": "qdrant_inactive_gc",
        "qdrant-graph-gc": "qdrant_graph_collection_gc",
        "neo4j-graph-gc": "neo4j_graph_gc",
        "parse-assets-gc": "parse_assets_gc",
    }

    def _scheduled_subcommands(self) -> set[str]:
        """Derive from the systemd units in the repository which cleanup subcommands are really scheduled."""
        import re

        units = Path(__file__).resolve().parents[2] / "deployment" / "systemd"
        if not units.is_dir():
            self.skipTest("systemd 单元不在仓库里")
        found: set[str] = set()
        for service in units.glob("*.service"):
            text = service.read_text(encoding="utf-8")
            # only units with a matching .timer count: no timer, no recurring job
            if not service.with_suffix(".timer").exists():
                continue
            for line in text.splitlines():
                if not line.startswith("ExecStart="):
                    continue
                m = re.search(r"kb-cleanup\.sh\s+([a-z0-9-]+)", line)
                if m:
                    found.add(m.group(1))
        return found

    def test_scheduled_subcommands_all_route_somewhere(self) -> None:
        """Every subcommand written in a unit must have a branch in cli.cmd_cleanup, otherwise the timer runs a
        command that argparse rejects immediately, round after round, and a oneshot unit fails very quietly."""
        import inspect

        from kb_pipeline import cli

        dispatch = inspect.getsource(cli.cmd_cleanup)
        scheduled = self._scheduled_subcommands()
        self.assertTrue(scheduled, "没有从单元里解析出任何 cleanup 子命令")
        for sub in sorted(scheduled):
            self.assertIn(f'"{sub}"', dispatch, f"cmd_cleanup 里没有 {sub} 分支")
            self.assertIn(sub, self.HANDLERS, f"本用例的 HANDLERS 表缺 {sub}")

    def test_recurring_growth_is_actually_swept(self) -> None:
        """Things that only ever grow must be swept by a round that **really runs**.

        Each entry corresponds to an incident that happened or nearly happened:
        - prune_job_history: jobs/failures/ingest_runs only grow; failure rows carry 4000-character stack traces
        """
        import inspect

        from kb_pipeline import maintenance

        scheduled = self._scheduled_subcommands()
        swept = "\n".join(
            inspect.getsource(getattr(maintenance, self.HANDLERS[sub]))
            for sub in sorted(scheduled)
            if sub in self.HANDLERS and hasattr(maintenance, self.HANDLERS[sub])
        )
        for needle, what in (
            ("prune_job_history", "jobs / failures / ingest_runs 的历史行"),
        ):
            # assertTrue rather than assertIn: on failure the latter dumps hundreds of lines of function source,
            # burying the information that matters (what nobody sweeps, what the current schedule is).
            self.assertTrue(
                needle in swept,
                f"{what} 没有被任何**已排期**的 cleanup 子命令清理"
                f"({needle} 不在其中任何一个的实现里)。当前排期:{sorted(scheduled)}",
            )

    def test_repo_units_match_what_is_installed(self) -> None:
        """When the repository units drift from the installed ones, the two guards above check against thin air."""
        import shutil
        import subprocess as sp

        if not shutil.which("systemctl"):
            self.skipTest("没有 systemctl")
        units = Path(__file__).resolve().parents[2] / "deployment" / "systemd"
        if not units.is_dir():
            self.skipTest("systemd 单元不在仓库里")
        for service in sorted(units.glob("*.service")):
            proc = sp.run(["systemctl", "--user", "cat", service.name],
                          capture_output=True, text=True, timeout=30)
            if proc.returncode != 0:
                self.skipTest(f"{service.name} 未安装(非部署机)")
            installed = {l for l in proc.stdout.splitlines() if l.startswith("ExecStart=")}
            in_repo = {l for l in _render_unit(service).splitlines() if l.startswith("ExecStart=")}
            self.assertEqual(installed, in_repo, f"{service.name} 的 ExecStart 与仓库不一致")


class MaintenanceDeferPolicyTests(unittest.TestCase):
    """The yield policy for maintenance jobs that run into service_busy. Both extremes have bitten us: a silent
    exit 0 makes whole GC rounds vanish for days with nobody noticing; an exit 75 every time paints normal
    behaviour such as "avoiding the ingestion peak" as failed for long stretches, so real faults stop being
    believed. Now the unit only turns red once the run of consecutive deferrals exceeds the limit."""

    REPO = Path(__file__).resolve().parents[2]

    def _fake_repo(self, tmp: Path, exit_code: int) -> None:
        py = tmp / "app" / ".venv" / "bin" / "python"
        py.parent.mkdir(parents=True, exist_ok=True)
        py.write_text(f"#!/bin/sh\nexit {exit_code}\n", encoding="utf-8")
        py.chmod(0o755)

    def _fake_curl(self, tmp: Path, exit_code: int) -> None:
        """The script probes Qdrant with curl before starting; tests cannot depend on a real instance, so a fake
        curl goes at the front of PATH."""
        curl = tmp / "bin" / "curl"
        curl.parent.mkdir(parents=True, exist_ok=True)
        curl.write_text(f"#!/bin/sh\nexit {exit_code}\n", encoding="utf-8")
        curl.chmod(0o755)

    def _run(self, tmp: Path, script: str, *args: str):
        import subprocess

        if not (tmp / "bin" / "curl").exists():
            self._fake_curl(tmp, 0)             # Qdrant is up by default
        env = dict(os.environ)
        env.update({
            "PATH": str(tmp / "bin") + os.pathsep + os.environ.get("PATH", ""),
            "KB_LOCAL_BASE_DIR": str(tmp),
            "KB_MAINT_STATE_DIR": str(tmp / "maint"),
            "KB_MAINT_DEFER_LIMIT": "3",
            "KB_CLEANUP_BUSY_ATTEMPTS": "1",   # give up at once; do not sleep 15 minutes in a test
            "KB_GRAPH_BUSY_ATTEMPTS": "1",
            "KB_QDRANT_WAIT_ROUNDS": "1",      # one probe decides; do not wait 5 minutes in a test
            "KB_QDRANT_WAIT_SECONDS": "0",
        })
        return subprocess.run(
            ["bash", str(self.REPO / "scripts" / script), *args],
            env=env, capture_output=True, text=True, timeout=60)

    def test_occasional_deferral_is_quiet_and_a_run_of_them_is_not(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._fake_repo(tmp, 75)            # the CLI says "yield" every time
            codes = [self._run(tmp, "kb-cleanup.sh", "parse-assets-gc").returncode
                     for _ in range(3)]
            self.assertEqual(codes, [0, 0, 75],
                             "前两轮该安静让路,第三轮才让单元变红")

            counter = tmp / "maint" / "cleanup-parse-assets-gc.defers"
            self.assertEqual(counter.read_text(encoding="utf-8").strip(), "3")

            # one successful run resets the count; the next deferral starts quiet again
            self._fake_repo(tmp, 0)
            self.assertEqual(self._run(tmp, "kb-cleanup.sh", "parse-assets-gc").returncode, 0)
            self.assertFalse(counter.exists(), "成功一轮必须清掉连续计数")
            self._fake_repo(tmp, 75)
            self.assertEqual(self._run(tmp, "kb-cleanup.sh", "parse-assets-gc").returncode, 0)

    def test_real_failures_still_surface_immediately(self) -> None:
        """Yielding is reserved for exit code 75. Any other non-zero exit code must surface unchanged rather than
        being swallowed by the counter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._fake_repo(tmp, 1)
            self.assertEqual(self._run(tmp, "kb-cleanup.sh", "parse-assets-gc").returncode, 1)
            self.assertFalse((tmp / "maint").exists(), "真失败不该记进让路计数")

    def test_graph_rebuild_check_shares_the_same_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._fake_repo(tmp, 75)
            codes = [self._run(tmp, "kb-graph-rebuild-check.sh").returncode for _ in range(3)]
            self.assertEqual(codes, [0, 0, 75])

    def test_both_schedulers_use_the_shared_policy(self) -> None:
        """There must be a single policy. A script with its own exit 75 would make the two sides diverge."""
        for name in ("kb-cleanup.sh", "kb-graph-rebuild-check.sh"):
            text = (self.REPO / "scripts" / name).read_text(encoding="utf-8")
            self.assertIn("lib/kb-maint-defer.sh", text, f"{name} 没有共用让路策略")
            self.assertIn("maint_defer_give_up", text)
            self.assertIn("maint_defer_clear", text)
            self.assertNotIn("exit 75", text, f"{name} 里还留着自己的 exit 75")

    def test_qdrant_not_ready_is_a_deferral_not_a_failure(self) -> None:
        """2026-09-06: when Qdrant could not be reached, the unit's ExecStartPre used to exit 75 directly,
        painting the unit failed every round while the services were deliberately stopped. The wait now lives in
        the script and goes through the same deferral counter: quiet exit, red only once the run exceeds the
        limit; once Qdrant is back the CLI runs as usual. Cleanup subcommands that never touch Qdrant neither
        wait nor count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._fake_repo(tmp, 1)             # the CLI would fail with 1 if it were reached, proving it was not
            self._fake_curl(tmp, 22)            # the health probe keeps failing
            codes = [self._run(tmp, "kb-graph-rebuild-check.sh").returncode for _ in range(3)]
            self.assertEqual(codes, [0, 0, 75])
            self.assertEqual((tmp / "maint" / "graph-rebuild.defers").read_text(encoding="utf-8").strip(), "3")
            codes = [self._run(tmp, "kb-cleanup.sh", "parse-assets-gc").returncode for _ in range(3)]
            self.assertEqual(codes, [0, 0, 75])
            self.assertEqual(self._run(tmp, "kb-cleanup.sh", "weekly").returncode, 1)
            self.assertFalse((tmp / "maint" / "cleanup-weekly.defers").exists())
            # Qdrant is back: straight into the CLI, and one successful run resets the count
            self._fake_curl(tmp, 0)
            self._fake_repo(tmp, 0)
            self.assertEqual(self._run(tmp, "kb-graph-rebuild-check.sh").returncode, 0)
            self.assertFalse((tmp / "maint" / "graph-rebuild.defers").exists())
            self.assertEqual(self._run(tmp, "kb-cleanup.sh", "parse-assets-gc").returncode, 0)
            self.assertFalse((tmp / "maint" / "cleanup-parse-assets-gc.defers").exists())
        for name in ("carrel-graph-rebuild.service", "carrel-qdrant-gc.service"):
            unit = (self.REPO / "deployment" / "systemd" / name).read_text(encoding="utf-8")
            self.assertNotIn("\nExecStartPre=", unit, f"{name} 还在 ExecStartPre 里自己 exit 75")   # mentioning the word in a comment does not count
        lib = (self.REPO / "scripts" / "lib" / "kb-maint-defer.sh").read_text(encoding="utf-8")
        self.assertIn("maint_wait_qdrant_or_defer()", lib)
        self.assertIn('maint_wait_qdrant_or_defer "graph-rebuild"', (self.REPO / "scripts" / "kb-graph-rebuild-check.sh").read_text(encoding="utf-8"))
        self.assertIn('maint_wait_qdrant_or_defer "cleanup-$COMMAND"', (self.REPO / "scripts" / "kb-cleanup.sh").read_text(encoding="utf-8"))


class SystemdUnitInstallTests(unittest.TestCase):
    """The installed units are **independent copies** (not symlinks); editing the repo files tells systemd nothing.

    Between intent (git) and effect (systemd) sit two steps a human has to remember, "cp + daemon-reload";
    forgetting them raises no error, the behaviour just differs from what you think it is; that is configuration
    drift. The script turns the sync into one idempotent command, and this test makes forgetting to sync
    visible.
    """

    UNIT_DIR = Path("deployment/systemd")

    def _installed_dir(self) -> Path:
        base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
        return Path(base) / "systemd" / "user"

    def test_installed_units_match_the_repo(self) -> None:
        repo = Path(__file__).resolve().parents[2] / self.UNIT_DIR
        units = sorted(p for p in repo.iterdir() if p.suffix in {".service", ".timer"})
        self.assertTrue(units, "仓库里应当有 unit 文件")
        installed = self._installed_dir()
        if not any((installed / u.name).exists() for u in units):
            self.skipTest(f"这台机器没装这些 unit（{installed}）")
        drifted = []
        for unit in units:
            target = installed / unit.name
            if not target.exists():
                drifted.append(f"{unit.name}: 未安装")
            elif target.read_text(encoding="utf-8") != _render_unit(unit):
                drifted.append(f"{unit.name}: 内容不一致")
        self.assertEqual(drifted, [], "跑 ./scripts/install-systemd.sh 同步:\n" + "\n".join(drifted))

    def test_install_script_is_idempotent_and_never_restarts(self) -> None:
        """Restarting a running worker loses the parse progress of the current file, so the install script only
        reloads and never restarts. A reload only makes systemd re-read the configuration; it does not touch
        the processes."""
        script = _repo_file("scripts/install-systemd.sh")
        self.assertIn("daemon-reload", script)
        self.assertNotIn("systemctl --user restart", script)
        self.assertIn("cmp -s", script)          # no copy when the content is unchanged
        self.assertIn("--check", script)         # there is a read-only drift-check mode


class WorkerScriptAndTimerTests(unittest.TestCase):
    def test_worker_script_tolerates_state_db_locks(self) -> None:
        """R1 / R2: a state database lock timeout is not a crash; wait a while and retry. B11: the graph-rebuild
        unit loads the env file into its environment."""
        script = _repo_file("scripts/kb-pipeline-worker-once.sh")
        self.assertIn("database is locked", script)
        self.assertIn("MAX_LOCK_WAITS", script)
        self.assertIn("lock_waits=0", script)
        unit = _repo_file("deployment/systemd/carrel-graph-rebuild.service")
        self.assertIn("EnvironmentFile=-__CARREL_HOME__/config/knowledge-base.env", unit)

    def test_graph_check_runs_every_two_hours(self) -> None:
        """2026-09-06: the graph maintenance check moved from every 30 minutes to every 2 hours; the deferral
        limit went from 48 rounds to 12 accordingly (still one day)."""
        timer = _repo_file("deployment/systemd/carrel-graph-rebuild.timer")
        self.assertIn("OnUnitActiveSec=2h", timer)
        self.assertNotIn("OnCalendar", timer)                 # 2026-09-09 switched to relative time: no wall clock, no time zone
        self.assertNotIn("07/30", timer)
        self.assertIn('KB_MAINT_DEFER_LIMIT:-12}', _repo_file("scripts/kb-graph-rebuild-check.sh"))
        for rel in ("app/kb_server/static/app.js", "app/kb_server/static/index.html"):
            text = _repo_file(rel)
            self.assertIn("每 2 小时", text, rel)
            self.assertNotIn("每 30 分钟", text, rel)


class OpsFixRegressionTests(_CodexAudit20260906TestsSupport, _CodexFinalTestsSupport, unittest.TestCase):
    """Regressions for problems found by successive re-reviews, health checks and audits; each test's
    docstring records where it came from and the symptom observed at the time."""

    def test_log_rotation_keeps_lines_appended_during_gzip(self) -> None:  # issue 23
        import gzip as gzip_module

        from kb_pipeline import maintenance

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "kb.log"; dst = Path(tmp) / "kb.log.gz"
            src.write_text("OLD-LINE\n", encoding="utf-8")
            real_open = gzip_module.open
            appended = {"done": False}

            def open_and_append(*args, **kwargs):
                handle = real_open(*args, **kwargs)
                if not appended["done"]:
                    appended["done"] = True
                    with src.open("a", encoding="utf-8") as live:
                        live.write("APPENDED-DURING-ROTATION\n")
                return handle

            with patch.object(maintenance.gzip, "open", open_and_append):
                maintenance.gzip_file_and_truncate(src, dst)
            self.assertEqual(src.read_text(encoding="utf-8"), "APPENDED-DURING-ROTATION\n")
            with gzip_module.open(dst, "rt", encoding="utf-8") as archived:
                self.assertEqual(archived.read(), "OLD-LINE\n")

    def test_unenrolled_kb_is_hard_deleted_when_retention_expires(self) -> None:
        """S3: a knowledge base closed from the console still has its directory, so the old GC skipped it
        forever and the "deleted automatically after 7 days" promised by the dialog never happened. The old
        semantics of directory-missing deactivation must stay unchanged."""
        from unittest import mock

        from kb_pipeline import discovery, maintenance

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); mirror = root / "mirror"
            (mirror / "关掉的库").mkdir(parents=True)
            (mirror / "还在的库").mkdir()
            state = root / "s.db"; db.init_db(state)
            with db.connect(state) as con:
                a, _ = discovery.enroll(con, mirror, "关掉的库")
                b, _ = discovery.enroll(con, mirror, "还在的库")
                discovery.mark_inactive(con, a.kb_id, reason="unenrolled")        # closed from the console
                discovery.mark_inactive(con, b.kb_id, reason="directory_missing")  # the directory went missing once
                con.execute("UPDATE kb_sources SET inactive_at = ?", (int(time.time()) - 30 * 86400,))
                con.commit()
            stub = SimpleNamespace(state_db=state, mirror_root=mirror,
                                   qdrant_url="http://q", qdrant_api_key="",
                                   opensearch_url="http://o", cache_dir=root / "cache",
                                   graph_work_dir=root / "gw")
            with mock.patch.object(maintenance, "service_busy", return_value=(False, [])), \
                    mock.patch.object(maintenance, "_hard_delete_kb",
                                      side_effect=lambda *a, **k: {"forgotten": True}) as hard_delete:
                result = maintenance.kb_sources_gc(stub, retention_days=7)
            dropped = {entry["kb_id"] for entry in result["dropped"]}
            self.assertEqual(dropped, {a.kb_id})            # the closed KB is deleted once retention expires
            self.assertEqual(hard_delete.call_count, 1)
            # a directory-missing deactivation whose directory exists is still skipped (the scan reactivates it)
            self.assertNotIn(b.kb_id, dropped)

    def test_failed_external_delete_keeps_the_registry_row(self) -> None:
        """S4: the registry row is the only lead back to the collection / Neo4j projection. If any side fails
        to delete, the row must stay, marked delete_failed, otherwise nothing can ever find the leftovers."""
        from unittest import mock

        from kb_pipeline import discovery, maintenance

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); mirror = root / "mirror"; (mirror / "库X").mkdir(parents=True)
            state = root / "s.db"; db.init_db(state)
            with db.connect(state) as con:
                src, _ = discovery.enroll(con, mirror, "库X")
                con.commit()
            stub = SimpleNamespace(state_db=state, mirror_root=mirror,
                                   qdrant_url="http://q", qdrant_api_key="",
                                   opensearch_url="http://o", cache_dir=root / "cache",
                                   graph_work_dir=root / "gw", neo4j_password="")
            boom = mock.Mock(side_effect=RuntimeError("qdrant unreachable"))
            with db.connect(state) as con:
                errors: list[str] = []
                with mock.patch("kb_pipeline.vector.qdrant.client", return_value=mock.Mock()), \
                        mock.patch("kb_pipeline.vector.qdrant.delete_collection", boom), \
                        mock.patch.object(maintenance, "_drop_graph_data", return_value={}), \
                        mock.patch("kb_pipeline.search_fts.delete_collection", return_value={}):
                    entry = maintenance._hard_delete_kb(
                        stub, con, kb_id=src.kb_id, collection=src.collection, errors=errors)
                con.commit()
                row = con.execute("SELECT status, inactive_reason FROM kb_sources WHERE kb_id=?",
                                  (src.kb_id,)).fetchone()
            self.assertTrue(errors)                          # the failure is recorded
            self.assertFalse(entry["forgotten"])             # the row was not forgotten
            self.assertIsNotNone(row)                        # the row is still there; the next GC round can retry
            self.assertEqual(str(row["inactive_reason"]), "delete_failed")

    def test_side_notes_readme_and_retention(self) -> None:
        # The schedule table moved from the README into the operations guide (2026-09-28 restructure).
        self.assertNotIn("每 30 分钟", _repo_file("docs/operations.zh-CN.md"))
        self.assertIn("每 2 小时", _repo_file("docs/operations.zh-CN.md"))
        self.assertIn('os.getenv("GRAPH_GC_KEEP_VERSIONS", "2")', _repo_file("app/kb_pipeline/config.py"))
        self.assertIn("GRAPH_GC_KEEP_VERSIONS=2", _repo_file("config/knowledge-base.env.example"))

    def test_stop_path_recognises_kb_launched_processes_and_separates_dead_from_foreign(self) -> None:
        from unittest import mock

        from kb_pipeline import maintenance
        from kb_pipeline.utils import looks_like_kb_process, looks_like_worker_command

        self.assertTrue(looks_like_kb_process("/x/.venv/bin/python /x/.venv/bin/kb worker --once --source kb_002"))
        self.assertTrue(looks_like_kb_process("/x/.venv/bin/python -m kb_pipeline --env-file e graph build --source kb_001"))
        self.assertTrue(looks_like_kb_process("kb graph build"))
        self.assertFalse(looks_like_kb_process("/usr/bin/python3 -m http.server"))
        self.assertFalse(looks_like_kb_process("bash /x/scripts/kb-pipeline-worker-once.sh"))
        self.assertTrue(looks_like_worker_command("/x/.venv/bin/kb worker --once"))
        self.assertFalse(looks_like_worker_command("/x/.venv/bin/kb scan --source kb_002"))
        with mock.patch.object(maintenance.Path, "read_bytes", return_value=b"/x/.venv/bin/kb\x00worker\x00--once\x00"):
            self.assertTrue(maintenance._process_matches(4242, "kb_pipeline"))
        with mock.patch.object(maintenance.Path, "read_bytes", return_value=b"/usr/bin/python3\x00-m\x00http.server\x00"):
            self.assertFalse(maintenance._process_matches(4242, "kb_pipeline"))
        # alive but not ours: left alone and not counted as stopped; only a missing pid counts as "confirmed dead"
        with mock.patch.object(maintenance, "_process_matches", return_value=False):
            self.assertEqual(maintenance._terminate_pid(os.getppid()), "not-ours")
            self.assertEqual(maintenance._terminate_pid(4194300), "already-gone")
        src = _repo_file("app/kb_pipeline/maintenance.py")
        self.assertIn('entry["stopped"] = (not remaining) or dead', src)

    def test_delete_failed_rows_are_retried_even_if_the_directory_is_still_there(self) -> None:
        from unittest import mock

        from kb_pipeline import discovery, maintenance

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); mirror = root / "mirror"
            (mirror / "删失败的库").mkdir(parents=True)
            state = root / "s.db"; db.init_db(state)
            with db.connect(state) as con:
                a, _ = discovery.enroll(con, mirror, "删失败的库")
                con.execute("UPDATE kb_sources SET status='inactive', inactive_reason='delete_failed', inactive_at=? WHERE kb_id=?",
                            (int(time.time()) - 30 * 86400, a.kb_id))
                con.commit()
            stub = SimpleNamespace(state_db=state, mirror_root=mirror, qdrant_url="http://q", qdrant_api_key="",
                                   opensearch_url="http://o", cache_dir=root / "cache", graph_work_dir=root / "gw")
            with mock.patch.object(maintenance, "service_busy", return_value=(False, [])), \
                    mock.patch.object(maintenance, "_hard_delete_kb", side_effect=lambda *a, **k: {"forgotten": True}) as hard_delete:
                result = maintenance.kb_sources_gc(stub, retention_days=7)
            self.assertEqual(hard_delete.call_count, 1)
            self.assertEqual({e["kb_id"] for e in result["dropped"]}, {a.kb_id})
