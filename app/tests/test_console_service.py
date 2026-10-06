"""Console service layer: label versions, pause / close / delete semantics, build yielding, model registry."""
from __future__ import annotations

import os
import socket
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from kb_pipeline import db
from kb_pipeline.graph.llm import LLMCache
from kb_pipeline.models import KBSource

from _support import _CodexFinalTestsSupport, _repo_file


class StopSemanticsTests(unittest.TestCase):
    """Close = stop and keep the cache; delete = terminate and wipe. The difference must stay in the code."""

    def test_terminate_lease_refuses_foreign_hosts_and_junk(self) -> None:
        from kb_pipeline.maintenance import _terminate_lease

        self.assertTrue(_terminate_lease("some-other-box:4242").startswith("skipped:other-host"))
        self.assertTrue(_terminate_lease("").startswith("skipped:bad-lease"))
        self.assertTrue(_terminate_lease(f"{socket.gethostname()}:notanint")
                        .startswith("skipped:bad-lease"))
        # This host but the pid does not exist: already dead, not an error
        self.assertEqual(_terminate_lease(f"{socket.gethostname()}:0"), "skipped:bad-pid")

    def test_interrupted_build_is_cancelled_not_failed(self) -> None:
        """A graph build interrupted by a signal is recorded as cancelled. Recording it as failed would turn
        the panel red and let "last failed" mask "stopped by the user, cache intact, can be resumed"."""
        source = _repo_file("app/kb_pipeline/graph/build.py")
        # A run that received a stop signal counts too (a client library may have wrapped the signal)
        self.assertIn(
            'terminal_status = ("cancelled" if isinstance(exc, (GraphBuildInterrupted, LLMInterrupted)) or interrupted.is_set()',
            source)
        self.assertIn("status=terminal_status,", source)

    def test_graph_status_maps_cancelled_to_stopped(self) -> None:
        """If cancelled fell into the fallback branch of _graph_status it would show "building" forever, and
        build now and delete graph would be refused with it -- the graph could never be touched again."""
        source = _repo_file("app/kb_server/service.py")
        head = source.split("def _graph_artifacts_exist", 1)[0]
        self.assertIn('if status == "cancelled":', head)
        self.assertIn('return "stopped"', head)

    def test_close_keeps_the_graph_cache_and_delete_drops_it(self) -> None:
        """Close only stops the process (stop_graph_build_now, touching no directory); only delete goes through
        _drop_graph_data -- and cache must be on delete's cleanup list, otherwise "delete progress" is a
        misnomer and the next build would hit the stale cache and pull the discarded results back in."""
        source = _repo_file("app/kb_pipeline/maintenance.py")
        stop_fn = source.split("def stop_graph_build_now", 1)[1].split("\ndef ", 1)[0]
        self.assertNotIn("rmtree", stop_fn)
        self.assertNotIn("_drop_graph_data", stop_fn)
        drop_fn = source.split("def _drop_graph_data", 1)[1].split("\ndef ", 1)[0]
        self.assertIn('"cache"', drop_fn)
        self.assertIn('"work"', drop_fn)
        self.assertIn("delete_graph_extractions", drop_fn)

    def test_delete_paths_stop_running_work_instead_of_refusing(self) -> None:
        """Both delete paths must "terminate first, then delete". The only refusal still allowed is when
        **another** knowledge base holds the build lock -- that build has nothing to do with this KB, and this
        KB's delete must not be used to kill it."""
        source = _repo_file("app/kb_pipeline/maintenance.py")
        for fn in ("def delete_graph_now", "def delete_kb_now"):
            body = source.split(fn, 1)[1].split("\ndef ", 1)[0]
            self.assertIn("stop_graph_build_now", body, fn)
            for refusal in ("建图任务进行中,等它结束后再删除知识图谱",
                            "该知识库有解析任务正在运行"):
                self.assertNotIn(refusal, body, f"{fn} 仍在拒绝而不是终止")
        kb_body = source.split("def delete_kb_now", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("stop_kb_parse_jobs", kb_body)
        self.assertIn("Another knowledge base is building its graph", kb_body)   # the only refusal kept


class ClosePreservesConfigTests(unittest.TestCase):
    """Close and delete must treat the config in clearly distinct ways:

        close graph / close knowledge base  ->  not a single key touched; reopening restores it as it was
        delete graph                        ->  only the derived items are cleared (labels, language, switch)

    The derived items are what the LLM induced from **this version of the corpus**; once the graph is deleted,
    the basis for that induction is gone too. Close is only deactivation and the basis is still there --
    clearing it would just make the user rerun an extraction for nothing.
    """

    TYPES = ["organization", "memory device", "signal"]

    def _enrolled(self, tmp: str):
        from kb_pipeline import discovery

        state = Path(tmp) / "s.db"
        db.init_db(state)
        mirror = Path(tmp) / "mirror"
        (mirror / "库A").mkdir(parents=True)
        with db.connect(state) as con:
            source, _ = discovery.enroll(con, mirror, "库A")
            discovery.set_config(con, source.kb_id, {
                "graph_enabled": True,
                "graph_entity_types": self.TYPES,
                "graph_language": "Chinese",
                "graph_llm": {"extract": "M1", "tune": "M2"},
                "graph_tune_sample_size": 20,
            })
            con.commit()
        return state, source

    def test_turning_the_graph_off_keeps_every_other_key(self) -> None:
        from kb_pipeline import discovery

        with tempfile.TemporaryDirectory() as tmp:
            state, source = self._enrolled(tmp)
            with db.connect(state) as con:
                # Turning the switch off in the console sends only this one key
                # (app.js: body {graph_enabled:false})
                discovery.set_config(con, source.kb_id, {"graph_enabled": False})
                con.commit()
                cfg = discovery.get_config(con, source.kb_id)
            self.assertIs(cfg["graph_enabled"], False)
            self.assertEqual(cfg["graph_entity_types"], self.TYPES)
            self.assertEqual(cfg["graph_language"], "Chinese")
            self.assertEqual(cfg["graph_tune_sample_size"], 20)
            self.assertEqual(cfg["graph_llm"], {"extract": "M1", "tune": "M2"})

    def test_closing_and_reopening_the_kb_keeps_the_graph_schema(self) -> None:
        """Closing a knowledge base only starts the deactivated retention period, and the config must not go
        with it -- reopening within the retention period is "open and restored at once", and if the restored
        KB is missing its labels the next graph build would silently switch to a different set of
        constraints."""
        from kb_pipeline import discovery

        with tempfile.TemporaryDirectory() as tmp:
            state, source = self._enrolled(tmp)
            mirror = Path(tmp) / "mirror"
            with db.connect(state) as con:
                discovery.mark_inactive(con, source.kb_id, reason="unenrolled")
                con.commit()
                self.assertEqual(
                    discovery.get_config(con, source.kb_id)["graph_entity_types"], self.TYPES)
                _, outcome = discovery.enroll(con, mirror, "库A")
                con.commit()
                cfg = discovery.get_config(con, source.kb_id)
            self.assertEqual(outcome, "reactivated")
            self.assertEqual(cfg["graph_entity_types"], self.TYPES)
            self.assertEqual(cfg["graph_language"], "Chinese")
            self.assertIs(cfg["graph_enabled"], True)


class SchemaVersionSelectionTests(unittest.TestCase):
    """Selecting a version and saving = expanding that version's labels and language into the effective values.

    Expanding, rather than having the graph build look up the version table, is deliberate: there are only two
    effective values, graph_entity_types / graph_language, and the graph build, the rebuild policy and prompt
    rendering all read those, so nothing else has to change.
    """

    def _current(self):
        return {"graph_schema_versions": [
            {"id": "v2", "entity_types": ["alpha", "beta"], "language": "English"},
            {"id": "v1", "entity_types": ["甲"], "language": "Chinese"},
        ]}

    def test_selecting_a_version_expands_to_the_effective_values(self) -> None:
        from kb_server.service import _apply_schema_version

        updates = {"graph_schema_active": "v1"}
        _apply_schema_version(self._current(), updates)
        self.assertEqual(updates["graph_entity_types"], ["甲"])
        self.assertEqual(updates["graph_language"], "Chinese")
        self.assertEqual(updates["graph_schema_active"], "v1")

    def test_the_legacy_placeholder_changes_nothing(self) -> None:
        """The fallback entry for legacy knowledge bases is not a real selection: saving it must be a complete
        no-op, otherwise merely opening the config page and clicking save would rewrite the labels."""
        from kb_pipeline.limits import CURRENT_SCHEMA_VERSION_ID
        from kb_server.service import _apply_schema_version

        updates = {"graph_schema_active": CURRENT_SCHEMA_VERSION_ID, "max_tokens": 400}
        _apply_schema_version(self._current(), updates)
        self.assertEqual(updates, {"max_tokens": 400})

    def test_an_evicted_version_is_refused_not_silently_ignored(self) -> None:
        from kb_server.service import _apply_schema_version

        with self.assertRaises(ValueError) as ctx:
            _apply_schema_version(self._current(), {"graph_schema_active": "gone"})
        self.assertIn("gone", str(ctx.exception))

    def test_absent_key_leaves_the_derived_values_alone(self) -> None:
        from kb_server.service import _apply_schema_version

        updates = {"graph_tune_sample_size": 20}
        _apply_schema_version(self._current(), updates)
        self.assertEqual(updates, {"graph_tune_sample_size": 20})


class LegacyLabelsSurviveFirstExtractionTests(unittest.TestCase):
    """The first extraction must not push out the labels that predate version management.

    Those labels live only in graph_entity_types; the fallback entry the view gives them is read-only and never
    stored. On 2026-08-24 kb_003 lost them exactly this way: the 14 labels the graph build actually used were
    replaced by 10 freshly extracted ones, and could only be reconstructed from the web access log plus the
    entity type distribution in Neo4j. Yet that was precisely the version most worth keeping -- the graph that
    already existed was built with it.
    """

    def test_pre_version_labels_are_materialised_before_the_new_one_lands(self) -> None:
        from kb_pipeline.limits import LEGACY_SCHEMA_VERSION_ID, push_schema_version
        from kb_pipeline.graph.schema_flow import ring_with_legacy

        stored = {"graph_entity_types": ["organization", "signal"], "graph_language": "Chinese"}
        ring = ring_with_legacy(stored)
        self.assertEqual(len(ring), 1)
        self.assertEqual(ring[0]["id"], LEGACY_SCHEMA_VERSION_ID)
        self.assertEqual(ring[0]["entity_types"], ["organization", "signal"])
        self.assertEqual(ring[0]["language"], "Chinese")
        # After a new version enters the ring, the old one is still there and can be selected again
        after = push_schema_version(ring, {"id": "v2", "entity_types": ["x"]})
        self.assertEqual([v["id"] for v in after], ["v2", LEGACY_SCHEMA_VERSION_ID])

    def test_the_materialised_id_is_selectable_not_the_readonly_placeholder(self) -> None:
        """The id used to materialise must be a real id. The placeholder id is treated on the save path as a
        "keep as is" no-op -- materialising with it would put a version in the dropdown that does nothing
        when selected."""
        from kb_pipeline.limits import CURRENT_SCHEMA_VERSION_ID, LEGACY_SCHEMA_VERSION_ID
        from kb_pipeline.graph.schema_flow import ring_with_legacy
        from kb_server.service import _apply_schema_version

        self.assertNotEqual(LEGACY_SCHEMA_VERSION_ID, CURRENT_SCHEMA_VERSION_ID)
        ring = ring_with_legacy({"graph_entity_types": ["signal"], "graph_language": "Chinese"})
        updates = {"graph_schema_active": LEGACY_SCHEMA_VERSION_ID}
        _apply_schema_version({"graph_schema_versions": ring}, updates)
        self.assertEqual(updates["graph_entity_types"], ["signal"])
        self.assertEqual(updates["graph_language"], "Chinese")

    def test_metadata_is_left_blank_rather_than_invented(self) -> None:
        """Nothing was recorded at the time. Inventing a plausible timestamp would make the whole version record
        untrustworthy -- and the entire value of rolling back a version rests on it being trustworthy."""
        from kb_pipeline.graph.schema_flow import ring_with_legacy

        entry = ring_with_legacy({"graph_entity_types": ["signal"]})[0]
        self.assertIsNone(entry["created_at"])
        self.assertIsNone(entry["model"])
        self.assertIsNone(entry["sample_size"])
        self.assertTrue(entry["legacy"])

    def test_an_existing_ring_is_left_alone(self) -> None:
        from kb_pipeline.graph.schema_flow import ring_with_legacy

        ring = [{"id": "v1", "entity_types": ["a"]}]
        self.assertEqual(ring_with_legacy(
            {"graph_schema_versions": ring, "graph_entity_types": ["b"]}), ring)

    def test_a_kb_that_never_had_labels_gets_no_phantom_version(self) -> None:
        from kb_pipeline.graph.schema_flow import ring_with_legacy

        self.assertEqual(ring_with_legacy({}), [])
        self.assertEqual(ring_with_legacy({"graph_entity_types": []}), [])


class SchemaVersionContractTests(unittest.TestCase):
    def test_clients_cannot_write_the_version_ring(self) -> None:
        """The version ring is maintained by the server during extraction. If clients could write it, "when this
        version was extracted, with which model and at what sample size" would become something anyone can
        make up -- and the entire value of rolling back a version rests on that record being trustworthy."""
        from fastapi import HTTPException

        from kb_server.api import _reject_bad_config

        with self.assertRaises(HTTPException) as ctx:
            _reject_bad_config({"graph_schema_versions": [{"id": "fake"}]})
        self.assertEqual(ctx.exception.status_code, 422)
        with self.assertRaises(HTTPException):
            _reject_bad_config({"graph_schema_active": ["not", "a", "string"]})
        _reject_bad_config({"graph_schema_active": "v1"})          # a normal value passes

    def test_labels_row_spans_the_right_grid_columns(self) -> None:
        """The dropdown takes column 1 (aligned with "Label extraction model"); the label box spans columns 2-4
        (left edge aligned with "Sample size", right edge aligned with "Output language")."""
        css = _repo_file("app/kb_server/static/index.html")
        self.assertIn("#cfg-tune .form-grid{grid-template-columns:1.4fr 1.4fr .75fr 1fr}", css)
        self.assertIn("#cfg-ver-row > :last-child{grid-column:2 / -1}", css)
        # The dropdown must sit in the same grid, otherwise there is nothing to align
        markup = css.split('id="cfg-ver-row"', 1)[1].split("</div>", 3)[0]
        self.assertIn('id="cfg-gver"', markup)

    def test_deleting_the_graph_also_clears_the_version_ring(self) -> None:
        """Versions, like labels, are induced from this version of the corpus. Once the graph is deleted the
        basis is gone; keeping a ring of old versions would only let someone pick a set of constraints of
        unknown origin back from the dropdown."""
        source = _repo_file("app/kb_pipeline/maintenance.py")
        body = source.split("def _drop_graph_data", 1)[1].split("\ndef ", 1)[0]
        for key in ("graph_entity_types", "graph_language",
                    "graph_schema_versions", "graph_schema_active"):
            self.assertIn(f'"{key}": None', body, key)


class RebuildGateScopeTests(unittest.TestCase):
    """The graph build's yield decision must be per knowledge base, not site-wide.

    The reason is not quota contention -- the embedding concurrency budget is already partitioned (parse
    pipeline 20 + graph build 10 + 2 reserved for retrieval = max_num_seqs 32). The real reason is "this KB's
    corpus is still growing": building while ingesting produces half a graph. Other KBs being parsed has
    nothing to do with this one.

    The automatic rebuild used to go through the global service_busy (any running job of any KB, any worker
    process, five locks, the MinerU queue), so during the hours a large KB was being ingested, KBs that had
    long finished parsing never got a turn -- and this check runs only once a day. The manual "build now" was
    per KB all along -- the two paths applied opposite rules.
    """

    def _job(self, con, job_id, *, kb_id="kb_1", status="queued",
             job_type="parse", next_attempt_at=0) -> None:
        con.execute(
            "INSERT INTO jobs(job_id, kb_id, collection, job_type, status, priority, "
            "next_attempt_at, created_at, updated_at) VALUES(?,?,'c',?,?,100,?,1,1)",
            (job_id, kb_id, job_type, status, next_attempt_at),
        )

    def test_busy_counts_only_this_kbs_claimable_parse_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"
            db.init_db(state)
            now = int(time.time())
            with db.connect(state) as con:
                self._job(con, "mine-running", status="running")
                self._job(con, "mine-queued")
                self._job(con, "mine-backoff", status="retry", next_attempt_at=now + 9999)
                self._job(con, "mine-done", status="done")
                self._job(con, "mine-meta", job_type="metadata_update")
                self._job(con, "other-kb", kb_id="kb_2", status="running")
                con.commit()
                self.assertEqual(db.kb_parse_busy(con, "kb_1"), 2)   # running + queued
                self.assertEqual(db.kb_parse_busy(con, "kb_2"), 1)
                self.assertEqual(db.kb_parse_busy(con, "kb_3"), 0)

    def _settings(self, tmp: str):
        state = Path(tmp) / "s.db"
        db.init_db(state)
        runtime = Path(tmp) / "runtime"
        (runtime / "state").mkdir(parents=True)
        return SimpleNamespace(state_db=state, runtime_dir=runtime)

    def test_another_kb_parsing_no_longer_blocks_this_rebuild(self) -> None:
        """This is the change itself: while KB B is being ingested, KB A should rebuild as usual."""
        from kb_pipeline.cli import _rebuild_blocked

        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._settings(tmp)
            with db.connect(cfg.state_db) as con:
                for i in range(50):
                    self._job(con, f"busy{i}", kb_id="kb_2", status="queued")
                con.commit()
            self.assertIsNone(_rebuild_blocked(cfg, SimpleNamespace(kb_id="kb_1")))

    def test_this_kb_parsing_does_block_its_own_rebuild(self) -> None:
        from kb_pipeline.cli import _rebuild_blocked

        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._settings(tmp)
            with db.connect(cfg.state_db) as con:
                self._job(con, "mine", status="running")
                con.commit()
            blocked = _rebuild_blocked(cfg, SimpleNamespace(kb_id="kb_1"))
            self.assertEqual(blocked["reason"], "kb parsing")
            self.assertEqual(blocked["parse_jobs"], 1)

    def test_a_live_build_lock_still_defers_everyone(self) -> None:
        """The build lock is the only global condition kept: it serialises graph builds (one at a time), which
        is queueing rather than interference -- after yielding, the unit retry picks it up. The lock is a flock
        (2026-09-09): while it is really held the check must yield; once released it must not."""
        from kb_pipeline.cli import _rebuild_blocked
        from kb_pipeline.graph.lock import GraphBuildLock

        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._settings(tmp)
            lock = GraphBuildLock(cfg)
            lock.acquire()
            try:
                blocked = _rebuild_blocked(cfg, SimpleNamespace(kb_id="kb_1"))
                self.assertEqual(blocked["reason"], "another graph build is running")
            finally:
                lock.release()
            self.assertIsNone(_rebuild_blocked(cfg, SimpleNamespace(kb_id="kb_1")))

    def test_a_dead_build_lock_is_reclaimed_not_obeyed(self) -> None:
        """A process killed by SIGKILL leaves the lock directory and pid file behind. Yielding to them would
        stall the automatic rebuild forever. A flock follows the process: with nobody holding it, neither the
        directory nor the pid file counts as a lock, and the leftover pid file is cleaned up on the way."""
        from kb_pipeline.cli import _rebuild_blocked

        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._settings(tmp)
            lock = cfg.runtime_dir / "state" / "graph_build.lock.d"
            lock.mkdir()
            (lock / "pid").write_text("999999999\n", encoding="utf-8")   # a process that does not exist
            self.assertIsNone(_rebuild_blocked(cfg, SimpleNamespace(kb_id="kb_1")))
            self.assertFalse((lock / "pid").exists(), "残留的 pid 文件应当被顺手清掉")

    def test_manual_and_automatic_paths_share_one_predicate(self) -> None:
        """Two separately written copies of the SQL would drift sooner or later: manual can build while
        automatic yields, or the other way round."""
        cli = _repo_file("app/kb_pipeline/cli.py")
        svc = _repo_file("app/kb_server/service.py")
        self.assertIn("db.kb_parse_busy(con, source.kb_id)", cli)
        self.assertEqual(svc.count("db.kb_parse_busy(con, kb_id)"), 2)   # graph build + label extraction
        self.assertIn("_CLAIMABLE_SQL = db.CLAIMABLE_JOB_SQL", svc)
        # The automatic rebuild must not fall back to the global check
        branch = cli.split('args.graph_command == "check-rebuild"', 1)[1].split("\n    if args.", 1)[0]
        self.assertNotIn("service_busy", branch)


class GraphPauseTests(unittest.TestCase):
    """Pausing a graph build = stop the process, keep every artifact and the cache, leave the switch alone, and
    record one persistent intent.

    It shares its mechanism with "close graph"; the only difference is graph_enabled: close means "do not
    rebuild automatically any more", pause means "stop for now, carry on in a while". Folding the two into one
    entry point would leave the user pausing by turning the switch off -- which also stops automatic rebuilds.

    This class originally asserted "pause writes nothing". That assertion described the implementation of the
    time, not an invariant: stopping only the process leaves a cancelled record rather than a successful build,
    and evaluate_rebuild accepts only successful builds as the baseline, so for a KB that had never built a
    graph successfully due was always true after a pause and it was resumed at 00:00 that same night. The
    invariant now is: apart from the single graph_paused intent, nothing else may change.
    """

    def _pause_code(self) -> str:
        source = _repo_file("app/kb_server/service.py")
        body = source.split("def pause_graph_build", 1)[1].split("\ndef ", 1)[0]
        # Code only: the docstring explains "does not touch graph_enabled", and comments mention it too
        quote = chr(34) * 3
        code = body.split(quote, 2)[-1] if body.count(quote) >= 2 else body
        return "\n".join(line for line in code.splitlines()
                         if not line.lstrip().startswith("#"))

    def test_pause_stops_the_process_and_touches_nothing_else(self) -> None:
        code = self._pause_code()
        self.assertIn("stop_graph_build_now", code)
        for forbidden in ("graph_enabled", "_drop_graph_data", "rmtree"):
            self.assertNotIn(forbidden, code, f"暂停不该动 {forbidden}")

    def test_pause_writes_the_intent_and_only_that(self) -> None:
        """Write, but only this key -- a pause that also changed other config would be another surprise."""
        code = self._pause_code()
        self.assertIn('set_config(con, kb_id, {"graph_paused": True})', code)
        self.assertEqual(code.count("set_config"), 1)

    def test_resuming_clears_the_intent(self) -> None:
        """Both "build now / rebuild" and "enable graph" are explicit intents to resume. If the flag is not
        cleared, the automatic rebuild would still consider the KB paused after this run finishes and would
        never resume it on its own again."""
        source = _repo_file("app/kb_server/service.py")
        build = source.split("def trigger_graph_build", 1)[1].split("\ndef ", 1)[0]
        self.assertIn('{"graph_paused": None}', build)
        update = source.split("def update_kb_config", 1)[1].split("\ndef ", 1)[0]
        self.assertIn('updates.get("graph_enabled") is True', update)
        self.assertIn('updates["graph_paused"] = None', update)

    def test_auto_rebuild_honours_the_intent(self) -> None:
        """This is the only link in the whole chain that actually does anything: the yield check
        (_rebuild_blocked) only looks at whether this KB is being parsed and whether the build lock is held --
        after a pause neither holds, so it cannot stop the rebuild."""
        build = _repo_file("app/kb_pipeline/graph/build.py")
        body = build.split("def evaluate_rebuild", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("source.graph_paused", body)
        self.assertIn("paused_by_operator", body)
        # Must come before the successful-build check, otherwise no_successful_build returns due=True first
        self.assertLess(body.index("source.graph_paused"),
                        body.index("latest_successful_graph_build"))

    def test_the_paused_branch_actually_runs(self) -> None:
        """Actually call it once instead of reading the source.

        The first version of this branch was correct source plus a raising runtime: it referenced current_ids,
        a variable that only exists after active_source_chunks. Every structural assertion above was green,
        yet running it raised UnboundLocalError -- tests that only read the source cannot catch such bugs.

        It also proves the branch comes before all I/O: settings is a fake that blows up on any access, so
        returning at all shows a paused KB does not scan the whole collection for nothing every night.
        """
        from kb_pipeline.graph.build import evaluate_rebuild
        from kb_pipeline.models import KBSource

        class Boom:
            def __getattr__(self, name):
                raise AssertionError(f"暂停的库不该访问 settings.{name}")

        def source(paused: bool) -> KBSource:
            return KBSource(kb_id="kb_x", collection="kb_x", source_root="r",
                            source_type="local", max_tokens=400, overlap_tokens=80,
                            graph_enabled=True, graph_paused=paused)

        out = evaluate_rebuild(Boom(), source_key="kb_x", source=source(True))
        self.assertEqual((out["due"], out["reason"]), (False, "paused_by_operator"))

        # The negative case: when not paused it should carry on as usual (and so raise on the fake settings),
        # proving the early return above really comes from graph_paused, not from something else in the way.
        with self.assertRaises(AssertionError):
            evaluate_rebuild(Boom(), source_key="kb_x", source=source(False))

    def test_the_console_cannot_set_it_by_hand(self) -> None:
        """It is the trace of an action, not an option to tick at will: a form with pause ticked while nothing
        is actually paused is an inconsistent state."""
        api = _repo_file("app/kb_server/api.py")
        self.assertIn('key == "graph_paused"', api)
        self.assertIn("The paused state is maintained by the build operations", api)

    def test_pause_is_not_folded_into_graph_enabled(self) -> None:
        """The two keys must stay separate. graph_enabled is "whether to keep building in future", graph_paused
        is "stop for now" -- merged, a pause would also cancel the automatic rebuild policy."""
        from kb_pipeline import discovery

        self.assertIn("graph_paused", discovery.DEFAULTS)
        self.assertIn("graph_paused", discovery.CONFIG_KEYS)
        self.assertIs(discovery.DEFAULTS["graph_paused"], False)
        self.assertIsNot(discovery.DEFAULTS["graph_paused"],
                         discovery.DEFAULTS["graph_enabled"] or None)

    def test_pause_refuses_when_nothing_is_building(self) -> None:
        """Clicking pause while nothing is building must be refused explicitly rather than succeed silently --
        otherwise the user would believe they had stopped something."""
        source = _repo_file("app/kb_server/service.py")
        body = source.split("def pause_graph_build", 1)[1].split("\ndef ", 1)[0]
        self.assertIn('!= "running"', body)
        self.assertIn("No graph build is running", body)

    def test_cache_is_counted_only_for_a_stopped_build(self) -> None:
        """The cache directory may hold thousands of files. The overview is polled every 2.5 seconds and should
        not walk it for a number that is only shown in the paused state."""
        source = _repo_file("app/kb_server/service.py")
        body = source.split("def _graph_build_info", 1)[1].split("\ndef ", 1)[0]
        index = body.index("graph_cache_entries")
        guard = body.rindex('"cancelled"', 0, index)
        self.assertGreater(index, guard, "数缓存必须被停止态判定守着")

    def test_console_offers_pause_while_building_and_resume_after(self) -> None:
        """While building, pause is the only option; after a pause the button is relabelled "resume build" --
        calling it "build now / rebuild" would suggest starting over, whereas the cache actually makes the
        completed calls return instantly."""
        js = _repo_file("app/kb_server/static/app.js")
        self.assertIn('$("#cfg-gpause").disabled = !(active && building)', js)
        self.assertIn('$("#cfg-gbuild").disabled = !(active && $("#cfg-graph").checked) || building', js)
        # "Resume" is a promise about cost, used only when the server confirms the cache can really carry on --
        # after switching the build model, re-extracting labels or changing chunking parameters the old cache
        # is useless, and clicking it then means hours of a full rerun. See service._paused_cache_reuse.
        self.assertIn('canResume ? "继续建图" : "立即/重新建图"', js)
        self.assertIn("gb.cache_reusable === true", js)
        html = _repo_file("app/kb_server/static/index.html")
        self.assertIn('id="cfg-gpause"', html)

    def test_the_pause_dialog_warns_about_cache_invalidation(self) -> None:
        """Changing the model / labels / language / chunking during a pause invalidates the whole cache, turning
        "resume" into a rerun. It is the one pitfall of pausing and must be stated in the dialog, not only in
        a commit message."""
        js = _repo_file("app/kb_server/static/app.js")
        # The first $("#cfg-gpause") is the disabled assignment in renderConfig; we want the event handler
        dialog = js.split('$("#cfg-gpause").addEventListener', 1)[1].split("\n});", 1)[0]
        self.assertIn("缓存", dialog)
        for knob in ("模型", "标签", "谓词", "语言", "合并切片数"):
            self.assertIn(knob, dialog, knob)

    def test_cache_count_reads_the_per_kb_sqlite(self) -> None:
        from kb_pipeline.graph.llm import LLMCache
        from kb_server.service import graph_cache_entries

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "work"
            cache = LLMCache(work / "cache" / "007.sqlite")
            cache.put("k1", "m", "a"); cache.put("k2", "m", "b"); cache.close()
            other = LLMCache(work / "cache" / "008.sqlite")      # another KB does not count
            other.put("k3", "m", "c"); other.close()
            cfg = SimpleNamespace(graph_work_dir=work)
            self.assertEqual(graph_cache_entries(cfg, "kb_007"), 2)
            self.assertEqual(graph_cache_entries(cfg, "kb_404"), 0)   # file does not exist


class GraphSaveAndPauseSemanticsTests(unittest.TestCase):
    """2026-09-29 audit: the graph form sends graph_enabled and graph_llm with every save. Judged by the presence
    of the keys, saving any setting after a pause cleared the pause, and saving any policy while a build ran was
    refused. What counts now is whether the values changed."""

    def _env(self, tmp: str, config: dict):
        from unittest import mock

        from kb_pipeline import db as dbm
        from kb_pipeline import discovery

        root = Path(tmp)
        state_db = root / "state.sqlite3"
        dbm.init_db(state_db)
        mirror = root / "mirror"
        (mirror / "docs").mkdir(parents=True)
        with dbm.connect(state_db) as con:
            source, _ = discovery.enroll(con, mirror, "docs")
            dbm.upsert_llm(con, name="m1", base_url="http://llm.invalid/v1", api_key="", model_id="m1")
            dbm.upsert_llm(con, name="m2", base_url="http://llm.invalid/v1", api_key="", model_id="m2")
            discovery.set_config(con, source.kb_id, config)
            con.commit()
        stub = mock.Mock()
        stub.state_db = state_db
        stub.embedding_base_url = "http://embedding.invalid/v1"
        limits = {"embedding_max_model_len": 4096, "effective_max_model_len": 4096, "max_tokens_cap": 3276,
                  "max_tokens_min": 128, "cap_ratio": 0.8, "overlap_rule": "", "live": False}
        return source, stub, limits

    def test_saving_a_setting_keeps_the_pause_and_turning_the_graph_on_clears_it(self) -> None:
        from unittest import mock

        from kb_pipeline import db as dbm
        from kb_pipeline import discovery
        from kb_server import service

        with tempfile.TemporaryDirectory() as tmp:
            llm = {"extract": "m1", "summarize": "m1", "tune": "m1"}
            source, stub, limits = self._env(tmp, {"graph_enabled": True, "graph_paused": True, "graph_llm": llm})
            with mock.patch.object(service, "settings", return_value=stub), \
                    mock.patch.object(service, "chunk_limits", return_value=limits):
                # the form carries the switch and the models as they are, only the rebuild interval changed: still paused
                service.update_kb_config(source.kb_id, {"graph_enabled": True, "graph_llm": dict(llm),
                                                        "graph_rebuild_interval": "2w"})
                with dbm.connect(stub.state_db) as con:
                    cfg = discovery.get_config(con, source.kb_id)
                self.assertEqual((cfg.get("graph_paused"), cfg.get("graph_rebuild_interval")), (True, "2w"))
                # off and on again: only a switch that really goes from off to on overrides the earlier pause
                service.update_kb_config(source.kb_id, {"graph_enabled": False})
                service.update_kb_config(source.kb_id, {"graph_enabled": True, "graph_llm": dict(llm)})
                with dbm.connect(stub.state_db) as con:
                    cfg = discovery.get_config(con, source.kb_id)
                self.assertTrue(cfg.get("graph_enabled"))
                self.assertFalse(cfg.get("graph_paused"))

    def test_a_running_build_only_blocks_a_real_change_of_models(self) -> None:
        import socket
        from unittest import mock

        from kb_pipeline import db as dbm
        from kb_pipeline import discovery
        from kb_server import service

        with tempfile.TemporaryDirectory() as tmp:
            llm = {"extract": "m1", "summarize": "m1", "tune": ""}
            source, stub, limits = self._env(tmp, {"graph_enabled": True, "graph_llm": llm})
            with dbm.connect(stub.state_db) as con:
                con.execute(
                    "INSERT INTO graph_builds(graph_build_id, source_key, kb_id, source_collection, graph_version, status, "
                    "started_at, worker_host, worker_pid, heartbeat_at) VALUES('g1', ?, ?, ?, 'v1', 'running', ?, ?, ?, ?)",
                    (source.kb_id, source.kb_id, source.collection, int(time.time()), socket.gethostname(), os.getpid(),
                     int(time.time())))
                con.commit()
            with mock.patch.object(service, "settings", return_value=stub), \
                    mock.patch.object(service, "chunk_limits", return_value=limits):
                # a build is running, the form carries the same switch and models, automatic append changes: saved as usual
                saved = service.update_kb_config(source.kb_id, {"graph_enabled": True, "graph_llm": dict(llm),
                                                                "graph_auto_append": False})
                self.assertIs(saved["config"].get("graph_auto_append"), False)
                with self.assertRaisesRegex(ValueError, "step models cannot be changed now"):
                    service.update_kb_config(source.kb_id, {"graph_enabled": True, "graph_llm": {**llm, "extract": "m2"}})
                with dbm.connect(stub.state_db) as con:
                    self.assertEqual(discovery.get_config(con, source.kb_id)["graph_llm"]["extract"], "m1")

    def test_pause_that_meets_a_finished_build_records_nothing(self) -> None:
        """A pause request that meets the end of the build: the version is built. Writing the pause mark anyway
        would stop automatic appends and rebuilds of a finished base for good."""
        import socket
        from unittest import mock

        from kb_pipeline import db as dbm
        from kb_pipeline import discovery
        from kb_server import service

        with tempfile.TemporaryDirectory() as tmp:
            source, stub, _limits = self._env(tmp, {"graph_enabled": True})
            with dbm.connect(stub.state_db) as con:
                con.execute(
                    "INSERT INTO graph_builds(graph_build_id, source_key, kb_id, source_collection, graph_version, status, "
                    "started_at, worker_host, worker_pid, heartbeat_at) VALUES('g1', ?, ?, ?, 'v1', 'running', ?, ?, ?, ?)",
                    (source.kb_id, source.kb_id, source.collection, int(time.time()), socket.gethostname(), os.getpid(),
                     int(time.time())))
                con.commit()
            for outcome in ({"running": False, "stopped": True}, {"running": True, "stopped": True, "status": "done"}):
                with mock.patch.object(service, "settings", return_value=stub), \
                        mock.patch("kb_pipeline.maintenance.stop_graph_build_now", return_value=dict(outcome)):
                    with self.assertRaisesRegex(ValueError, "already finished"):
                        service.pause_graph_build(source.kb_id)
                with dbm.connect(stub.state_db) as con:
                    self.assertFalse(discovery.get_config(con, source.kb_id).get("graph_paused"))
            with mock.patch.object(service, "settings", return_value=stub), \
                    mock.patch.object(service, "graph_cache_entries", return_value=0), \
                    mock.patch("kb_pipeline.maintenance.stop_graph_build_now",
                               return_value={"running": True, "stopped": True, "status": "cancelled"}):
                service.pause_graph_build(source.kb_id)                       # a running build was really stopped: the pause is recorded
            with dbm.connect(stub.state_db) as con:
                self.assertIs(discovery.get_config(con, source.kb_id).get("graph_paused"), True)

    def test_an_interrupted_append_continues_as_an_append(self) -> None:
        """What stopped midway was an append and a built version is still live: "continue" appends once more.
        Resuming the same version would run a full build."""
        from unittest import mock

        from kb_pipeline import db as dbm
        from kb_server import service

        with tempfile.TemporaryDirectory() as tmp:
            source, stub, _limits = self._env(tmp, {"graph_enabled": True, "graph_paused": True,
                                                   "graph_llm": {"extract": "m1", "summarize": "m1", "tune": "m1"}})
            with dbm.connect(stub.state_db) as con:
                for bid, version, status, kind, started in (("g1", "v1", "done", "full", 100), ("g2", "v2", "cancelled", "append", 200)):
                    con.execute(
                        "INSERT INTO graph_builds(graph_build_id, source_key, kb_id, source_collection, graph_version, status, "
                        "started_at, finished_at, build_kind) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (bid, source.kb_id, source.kb_id, source.collection, version, status, started, started + 10, kind))
                con.commit()
            spawned: list[dict] = []

            def spawn(cfg, kb_id, **kwargs):
                spawned.append(kwargs)
                return {"started": True}

            gate = lambda con, cfg, kb_id: (None, {}, con.execute(
                "SELECT * FROM graph_builds WHERE kb_id=? ORDER BY started_at DESC LIMIT 1", (kb_id,)).fetchone())
            with mock.patch.object(service, "settings", return_value=stub), \
                    mock.patch.object(service, "_graph_action_gate", side_effect=gate), \
                    mock.patch.object(service, "_spawn_graph_build", side_effect=spawn):
                service.trigger_graph_build(source.kb_id)
                self.assertEqual(spawned, [{"append": True, "force": True}])
                with dbm.connect(stub.state_db) as con:
                    con.execute("UPDATE graph_builds SET build_kind = 'full' WHERE graph_build_id = 'g2'")
                    con.commit()
                with mock.patch.object(service, "_paused_cache_reuse", return_value={"cache_reusable": True, "cache_stale": ""}), \
                        mock.patch.object(dbm, "graph_phases_done", return_value=["extract"]):
                    service.trigger_graph_build(source.kb_id)
                self.assertEqual(spawned[-1], {"graph_version": "v2"})            # a full build that stopped midway still resumes its version
        js = _repo_file("app/kb_server/static/app.js")
        self.assertIn("resumeAppend ?", js)
        self.assertIn("const liveGraph = kb.graph_status === \"ok\" || (halted && !!(kb.graph_build || {}).active_graph_version);", js)


class SchemaVersionDeleteTests(unittest.TestCase):
    """Label versions must be deletable -- the ring has only 3 slots, and one badly extracted version would
    occupy a slot for good.

    But the version in effect cannot be deleted: the effective values graph_entity_types / graph_language were
    copied from it, and the graph already built follows it. After deleting it the dropdown would silently fall
    back to the newest version, so the UI would show A's labels while the build used B's -- both are just
    lists of words, and nobody would spot the mismatch. push_schema_version likewise skips it when evicting
    old versions; the two are outlets of one rule."""

    def _kb(self, tmp: str):
        from kb_pipeline import discovery

        root = Path(tmp); (root / "产品资料").mkdir()
        state = root / "s.db"
        db.init_db(state)
        ring = [{"id": f"v{n}", "created_at": n, "model": "M", "sample_size": 8,
                 "language": "Chinese", "entity_types": [f"t{n}"]} for n in (3, 2, 1)]
        with db.connect(state) as con:
            src, _ = discovery.enroll(con, root, "产品资料")
            discovery.set_config(con, src.kb_id, {
                "graph_entity_types": ["t3"], "graph_language": "Chinese",
                "graph_schema_versions": ring, "graph_schema_active": "v3",
            })
            con.commit()
        return state, src.kb_id

    def _call(self, state: Path, kb_id: str, version_id: str):
        from unittest import mock

        from kb_server import service

        with mock.patch.object(service, "settings",
                               lambda: SimpleNamespace(state_db=state)):
            return service.delete_graph_schema_version(kb_id, version_id)

    def _ring(self, state: Path, kb_id: str):
        from kb_pipeline import discovery

        with db.connect(state) as con:
            return discovery.get_config(con, kb_id).get("graph_schema_versions")

    def test_the_active_version_cannot_be_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state, kb_id = self._kb(tmp)
            with self.assertRaises(ValueError) as ctx:
                self._call(state, kb_id, "v3")
            self.assertIn("version in effect", str(ctx.exception))
            self.assertEqual(len(self._ring(state, kb_id)), 3, "拒绝之后环不能被动过")

    def test_the_readonly_placeholder_cannot_be_deleted(self) -> None:
        """schema_versions_view fabricates a __current__ entry on the fly for old KBs without version records;
        it is never stored -- deleting it would delete something that does not exist."""
        from kb_pipeline.limits import CURRENT_SCHEMA_VERSION_ID

        with tempfile.TemporaryDirectory() as tmp:
            state, kb_id = self._kb(tmp)
            with self.assertRaises(ValueError) as ctx:
                self._call(state, kb_id, CURRENT_SCHEMA_VERSION_ID)
            self.assertIn("read-only view", str(ctx.exception))

    def test_unknown_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state, kb_id = self._kb(tmp)
            for bad in ("v9", ""):
                with self.assertRaises(ValueError):
                    self._call(state, kb_id, bad)

    def test_a_spare_version_is_removed_and_the_rest_survive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state, kb_id = self._kb(tmp)
            out = self._call(state, kb_id, "v2")
            self.assertEqual(out["deleted"], "v2")
            self.assertEqual([v["id"] for v in self._ring(state, kb_id)], ["v3", "v1"])
            # The returned view is enough for the console to redraw without fetching the config again
            self.assertEqual([v["id"] for v in out["schema"]["versions"]], ["v3", "v1"])
            self.assertEqual(out["schema"]["active"], "v3", "生效版不受影响")

    def test_emptying_the_ring_clears_the_key_instead_of_leaving_an_empty_list(self) -> None:
        """set_config uses None to mean "remove this key". Writing [] would leave an empty array in
        config_json -- not the same state as "never extracted", and the view treats the two differently."""
        from kb_pipeline import discovery

        with tempfile.TemporaryDirectory() as tmp:
            state, kb_id = self._kb(tmp)
            with db.connect(state) as con:      # remove the active marker first, otherwise v3 cannot be deleted
                discovery.set_config(con, kb_id, {"graph_schema_active": None})
                con.commit()
            for vid in ("v1", "v2", "v3"):
                self._call(state, kb_id, vid)
            with db.connect(state) as con:
                stored = discovery.get_config(con, kb_id)
            self.assertNotIn("graph_schema_versions", stored)

    def test_console_exposes_delete_next_to_the_version_picker(self) -> None:
        html = _repo_file("app/kb_server/static/index.html")
        js = _repo_file("app/kb_server/static/app.js")
        api = _repo_file("app/kb_server/api.py")
        # The button and the dropdown share one cell (grid contract: see SchemaVersionContractTests)
        cell = html.split('id="cfg-ver-row"', 1)[1].split("</div>", 3)[0]
        self.assertIn('id="cfg-gver"', cell)
        self.assertIn('id="cfg-gvdel"', cell)
        # .ri-row cannot be reused: that rule pins the select at 88px, and a version label is a long string
        self.assertIn(".ver-row select{flex:1", html)
        self.assertIn('@router.delete("/kbs/{kb_id}/graph_schema/{version_id}")', api)
        # Whether it can be deleted depends on the version the server holds as active, not on the dropdown's
        # current choice -- changing the option is only an intention; until saved, builds still use the old one.
        self.assertIn("savedActive", js)
        self.assertIn("state.graphSchema.savedActive", js)


class LlmDeletionDecouplingTests(unittest.TestCase):
    """Deleting a model is no longer refused because "a knowledge base references it".

    It used to be a hard block: as long as the name appeared in any KB's graph_llm it could not be deleted.
    But **referenced is not the same as in use** -- once the graph is off, the whole graph-fields block (three
    model slots + the label extraction button) is disabled, none of those names can run, yet they still locked
    the model registry. Hit for real on 2026-08-30: the graphs of all three KBs were off, and deleting a model
    was still answered with "in use by: Semiconductors, Product docs, Library".

    Now decoupled: the delete goes through, and the KBs referencing it get that slot **cleared** (rather than
    keeping a dangling name pointing at a deleted model -- that would make the build report "not in the
    registry" when the truth is "you have not chosen one yet"). The console lists the affected KBs and slots
    in the confirmation dialog first, so it is not silent.
    """

    def _kb(self, tmp: str, graph_enabled: bool):
        from kb_pipeline import discovery

        root = Path(tmp); (root / "产品资料").mkdir(exist_ok=True)
        state = root / "s.db"
        if not state.exists():
            db.init_db(state)
        with db.connect(state) as con:
            for name in ("甲模型", "乙模型"):
                db.upsert_llm(con, name=name, base_url="http://x/v1",
                              api_key="k", model_id=name)
            src, _ = discovery.enroll(con, root, "产品资料")
            discovery.set_config(con, src.kb_id, {
                "graph_enabled": graph_enabled,
                "graph_llm": {"extract": "甲模型", "summarize": "乙模型", "tune": "甲模型"},
            })
            con.commit()
        return state, src.kb_id

    def _svc(self, state: Path):
        from unittest import mock

        from kb_server import service

        return mock.patch.object(service, "settings",
                                 lambda: SimpleNamespace(state_db=state))

    def test_a_referenced_model_can_be_deleted(self) -> None:
        from kb_pipeline import discovery
        from kb_server import service

        for enabled in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                state, kb_id = self._kb(tmp, enabled)
                with self._svc(state):
                    out = service.remove_llm("甲模型")
                self.assertEqual(out["deleted"], "甲模型")
                self.assertTrue(out["cleared"], "受影响的库要如实报出来")
                with db.connect(state) as con:
                    self.assertIsNone(db.get_llm(con, "甲模型"))
                    left = discovery.get_config(con, kb_id).get("graph_llm") or {}
                # The reference is cleared rather than left as a dangling name pointing at a deleted model
                self.assertNotIn("甲模型", left.values())
                # The slot that did not reference it is untouched
                self.assertEqual(left.get("summarize"), "乙模型")

    def test_the_cleared_references_are_reported(self) -> None:
        """Decoupled does not mean silent: the console must be able to say which slots will become empty after
        the delete."""
        from kb_server import service

        with tempfile.TemporaryDirectory() as tmp:
            state, _ = self._kb(tmp, True)
            with self._svc(state):
                listed = service.list_llms()
                out = service.remove_llm("甲模型")
        used = {m["name"]: m["used_by"] for m in listed}
        self.assertTrue(used["甲模型"], "list_llms 要带上 used_by,弹窗才有得说")
        steps = out["cleared"][0]["steps"]
        for label in ("Entity extraction model", "Label extraction model"):
            self.assertIn(label, steps)
        self.assertNotIn("Description summary model", steps)   # that slot uses the second model

    def test_deleting_an_unknown_model_still_fails(self) -> None:
        from kb_server import service

        with tempfile.TemporaryDirectory() as tmp:
            state, _ = self._kb(tmp, True)
            with self._svc(state), self.assertRaises(KeyError):
                service.remove_llm("根本不存在")

    def test_console_warns_before_clearing_and_redraws_after(self) -> None:
        js = _repo_file("app/kb_server/static/app.js")
        handler = js.split('.c-del").addEventListener', 1)[1].split("\n  });", 1)[0]
        self.assertIn("used_by", handler, "弹窗要列出受影响的库")
        self.assertIn("对应栏位会清空", handler)
        self.assertIn("renderConfig()", handler, "删完要重画三个下拉框")

    def test_the_three_model_slots_default_to_one_model(self) -> None:
        """All three model slots (extraction / summary / entity labels) preselect the default model when none is
        chosen, under the same name as the backend fallback in resolve_llm_specs: what the form shows is what
        the graph build / label extraction will actually use. Since 2026-09-09 the entity label slot is
        preselected too -- the automatic label extraction before building a blank graph uses it, and a user
        seeing "not selected" would assume there is no default. The community report slot was retired along
        with GraphRAG; the console has no fourth slot."""
        js = _repo_file("app/kb_server/static/app.js")
        self.assertIn('const DEFAULT_GRAPH_LLM = "DeepSeek V4 Flash"', js)
        # 2026-09-12: when the registry holds only one model, that is the default -- same rule as the backend's
        # build.default_graph_llm
        self.assertIn("state.llms.length === 1 ? state.llms[0].name", js)
        self.assertIn("if (!picked && fallback) picked = defaultGraphLlm();", js)
        svc = _repo_file("app/kb_server/service.py")
        self.assertIn("if missing and default_graph_llm(con) is None:", svc)
        for sel in ("#cfg-ge", "#cfg-gs", "#cfg-gt"):
            self.assertEqual(js.count(f'$("{sel}").innerHTML = llmOptions('), 2)
        self.assertNotIn("#cfg-gc\"", js)
        # Each slot passes fallback at both sites (load / draft); the model-list refresh loops over all three
        self.assertEqual(js.count("{ fallback: true }"), 7)
        self.assertIn('["#cfg-ge", "#cfg-gs", "#cfg-gt"].forEach', js)
        tune_lines = [ln for ln in js.splitlines() if '$("#cfg-gt").innerHTML' in ln]
        self.assertEqual(len(tune_lines), 2)
        for ln in tune_lines:
            self.assertIn("fallback: true", ln)

    def test_spawned_builds_reread_the_env_file_instead_of_inheriting_the_web_snapshot(self) -> None:
        """2026-09-12 on the real box: after KB_GRAPH_LLM_CONCURRENCY was changed from 16 to 128 in the env
        file, builds started by the web process still ran at 16 -- the web process had snapshotted the file
        into its own environment at startup, the child inherited it, and load_env_file is a setdefault, so the
        file cannot override existing values. The child's environment must drop the variables defined in the
        file so that it rereads the current file; the other variables are inherited as usual (systemd-run
        needs them)."""
        from kb_server import service as svc_mod

        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / "kb.env"
            env_file.write_text("# 注释\nKB_GRAPH_LLM_CONCURRENCY=128\nKB_GRAPH_LLM_TIMEOUT = 300\n", encoding="utf-8")
            saved = dict(os.environ)
            try:
                os.environ.update({"KB_GRAPH_LLM_CONCURRENCY": "16", "KB_GRAPH_LLM_TIMEOUT": "60",
                                   "PATH": "/usr/bin", "XDG_RUNTIME_DIR": "/run/user/1"})
                env = svc_mod._child_env(str(env_file), Path(tmp))
            finally:
                os.environ.clear(); os.environ.update(saved)
            self.assertNotIn("KB_GRAPH_LLM_CONCURRENCY", env)           # the child rereads the current file
            self.assertNotIn("KB_GRAPH_LLM_TIMEOUT", env)
            self.assertEqual((env["PATH"], env["XDG_RUNTIME_DIR"]), ("/usr/bin", "/run/user/1"))
            self.assertEqual((env["KB_ENV_FILE"], env["KB_LOCAL_BASE_DIR"]), (str(env_file), str(Path(tmp))))
        svc = _repo_file("app/kb_server/service.py")
        self.assertEqual(svc.count("env=_child_env("), 3)                 # two build Popens + the fallback script

    def test_spawned_builds_run_at_batch_oom_priority(self) -> None:
        """The console unit's oom_score_adj is 100 and child processes inherit it; before a build starts it is raised
        back to 200 and the process is then replaced by the build, the same on both launch paths, with the command
        line arguments passed through unchanged."""
        import subprocess
        import sys
        from unittest import mock

        from kb_server import service as svc_mod

        launched: list[list[str]] = []

        def popen(cmd, **kwargs):
            launched.append(list(cmd))
            if cmd[0] == "systemd-run" and len(launched) > 1:
                raise FileNotFoundError("systemd-run")
            return mock.Mock(pid=4242)

        with tempfile.TemporaryDirectory() as tmp:
            cfg = SimpleNamespace(state_db=Path(tmp) / "runtime" / "state" / "s.db", log_dir=Path(tmp) / "logs")
            with mock.patch.object(svc_mod.subprocess, "Popen", side_effect=popen):
                self.assertEqual(svc_mod._spawn_graph_build(cfg, "kb_001", append=True, force=True)["detached"], "systemd-scope")
                self.assertEqual(svc_mod._spawn_graph_build(cfg, "kb_001")["detached"], "popen")
        scoped, _, fallback = launched
        wrapper = svc_mod.BATCH_OOM_SCORE
        at = scoped.index(wrapper[0])
        self.assertEqual(scoped[:3], ["systemd-run", "--user", "--scope"])
        self.assertEqual(scoped[at:at + len(wrapper)], wrapper)
        self.assertEqual(scoped[at + len(wrapper) + 1:][:2] + scoped[-5:], ["-m", "kb_pipeline", "graph", "append", "--source", "kb_001", "--force"])
        self.assertEqual(fallback[:len(wrapper)], wrapper)
        self.assertEqual(fallback[-4:], ["graph", "build", "--source", "kb_001"])
        # Run the wrapper for real: the arguments (spaces included) arrive unchanged and nothing extra is printed; on a
        # machine with /proc the child's score is at least 200
        probe = ("import pathlib, sys; p = pathlib.Path('/proc/self/oom_score_adj'); "
                 "print(sys.argv[1:], p.read_text().strip() if p.exists() else 'n/a', sep='|')")
        out = subprocess.run(wrapper + [sys.executable, "-c", probe, "a b", "--x"], capture_output=True, text=True, timeout=30)
        self.assertEqual((out.returncode, out.stderr), (0, ""))
        argv, score = out.stdout.strip().split("|")
        self.assertEqual(argv, "['a b', '--x']")
        self.assertTrue(score == "n/a" or int(score) >= 200, score)

    def test_resolution_judge_batches_run_at_the_global_concurrency(self) -> None:
        """Decided by the user on 2026-09-12: entity-resolution judge batches no longer have their own cap of 5
        and follow KB_GRAPH_LLM_CONCURRENCY (the batches are independent of each other)."""
        src = _repo_file("app/kb_pipeline/graph/resolution.py")
        self.assertNotIn("RESOLUTION_WORKERS", src)
        self.assertEqual(src.count("workers=client.workers"), 2)         # two sites: judging + hypernym re-check

    def test_refresh_does_not_clobber_the_default_with_a_deleted_name(self) -> None:
        """Refreshing the model list keeps the dropdown's current value. But the name just deleted is no longer
        an option, and forcing it back in would replace the default llmOptions picked with an empty value --
        which is exactly why "the slot still looks empty after the delete"."""
        js = _repo_file("app/kb_server/static/app.js")
        self.assertIn("if ([...el.options].some(o => o.value === cur)) el.value = cur;", js)


class ConsoleStatusContractTests(unittest.TestCase):
    """Health check B2 / B3 / B4 / D4 / D5 / D7 / D8 / R6: the console and the service layer behind it."""

    def test_console_review_fixes(self) -> None:
        js = _repo_file("app/kb_server/static/app.js")
        self.assertNotIn("setQuestionsView", js)                       # B2: deleted function no longer called
        self.assertNotIn('setTab("jobs")', js)                          # B4: there is no jobs tab
        for gone in ("GRAPH_MODE_ZH", "BUILD_KIND_ZH", "BUILD_STATUS_ZH", "buildCountsText"):
            self.assertNotIn(gone, js, gone)                            # D4: dead code
        preview = js.split("async function refreshGraphPreview", 1)[1].split("\n}\n", 1)[0]
        self.assertIn("active_graph_version", preview)                  # B3: no refetch by new version mid-build
        self.assertIn("extract_failed_units", js)                       # R3: failed units shown on the card
        self.assertRegex(_repo_file("app/kb_server/static/index.html"), r'app\.js\?v=\d{8}-\d+')   # the cache-busting version bumps with every frontend change

    def test_files_endpoint_answers_304_when_unchanged(self) -> None:
        """D5: the file table gets a content-based ETag and answers 304 when unchanged."""
        from unittest import mock

        from fastapi.testclient import TestClient

        from kb_server import service
        from kb_server.main import create_app

        rows = [{"file_id": "f1", "rel_path": "a.md", "dot": "green"}]
        with mock.patch.object(service, "kb_files", return_value=rows):
            client = TestClient(create_app(), raise_server_exceptions=False)
            first = client.get("/api/kbs/kb_1/files")
            self.assertEqual(first.status_code, 200)
            etag = first.headers.get("ETag")
            self.assertTrue(etag)
            self.assertEqual(client.get("/api/kbs/kb_1/files", headers={"If-None-Match": etag}).status_code, 304)
            rows.append({"file_id": "f2", "rel_path": "b.md", "dot": "green"})
            changed = client.get("/api/kbs/kb_1/files", headers={"If-None-Match": etag})
            self.assertEqual(changed.status_code, 200)
            self.assertNotEqual(changed.headers.get("ETag"), etag)
        self.assertIn("If-None-Match", _repo_file("app/kb_server/static/app.js"))

    def test_graph_check_decisions_are_recorded_and_shown(self) -> None:
        """D7: the check-rebuild verdict is stored, and the status card speaks from it."""
        from kb_pipeline.graph.build import check_kind

        self.assertEqual(check_kind({"build": {}}), "full")
        self.assertEqual(check_kind({"append": {"build": {}}}), "append")
        self.assertEqual(check_kind({"build_skipped": {"retry": True, "reason": "parse_busy"}}), "deferred")
        self.assertEqual(check_kind({"build_skipped": {"reason": "llm_not_configured"}}), "skipped")
        self.assertEqual(check_kind({"error": "boom"}), "error")
        self.assertEqual(check_kind({"append": {"reason": "source_unchanged"}}), "none")
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"
            db.init_db(state)
            with db.connect(state) as con:
                self.assertIsNone(db.latest_graph_check(con, "kb_1"))
                db.record_graph_check(con, "kb_1", {"kind": "none", "append_reason": "config_changed_needs_full_rebuild"})
                got = db.latest_graph_check(con, "kb_1")
        self.assertEqual(got["append_reason"], "config_changed_needs_full_rebuild")
        self.assertGreater(got["at"], 0)
        self.assertIn("db.record_graph_check(", _repo_file("app/kb_pipeline/cli.py"))
        # The check verdict is kept in the database (for troubleshooting); the status card line "last check
        # ...: config changed ..." was removed on 2026-09-08 at the user's request -- after a manual rebuild it
        # still showed the verdict from two hours earlier, misleading more than it helped
        js = _repo_file("app/kb_server/static/app.js")
        self.assertNotIn("graphCheckLine", js)
        self.assertNotIn("上次检查", js)

    def test_stage_weights_follow_the_last_build(self) -> None:
        """D8: progress bar weights come from the actual per-stage durations of the last successful build; with
        too little data it falls back to the fixed table."""
        from kb_pipeline.graph.build import stage_weights_from_phases

        ts = [("prepare_input", 110), ("extract", 1010), ("merge", 1310), ("facts", 1400), ("enrich", 1430), ("neo4j_import", 1450)]
        self.assertEqual(stage_weights_from_phases(100, ts),
                         {"prepare_input": 10.0, "extract": 900.0, "merge": 300.0, "facts": 90.0, "enrich": 30.0, "neo4j_import": 20.0})
        self.assertEqual(stage_weights_from_phases(100, ts[:2]), {})
        from kb_pipeline.graph.build import merge_stage_weights

        cached = {"prepare_input": 1.0, "extract": 3.0, "merge": 11.0, "facts": 1.0, "enrich": 39.0, "neo4j_import": 1.0}
        merged = merge_stage_weights([cached, stage_weights_from_phases(100, ts)])
        self.assertEqual(merged["extract"], 900.0)      # per stage, take the slowest run
        self.assertEqual(merged["enrich"], 39.0)
        self.assertEqual(merge_stage_weights([{}, {"extract": 1.0}]), {})
        js = _repo_file("app/kb_server/static/app.js")
        self.assertIn("function graphStagesFor", js)
        self.assertIn("graphStagePct(gb.stage, gb.stage_weights)", js)

    def test_health_reports_timer_state(self) -> None:
        """R6: the service status carries the last result / next trigger / yield count of the six timers; a
        failed one turns the top-bar dot red."""
        from kb_server import service

        show = ("Id=carrel-scan.service\nActiveState=failed\nSubState=failed\nResult=exit-code\n\n"
                "Id=carrel-worker.service\nActiveState=inactive\nSubState=dead\nResult=success\n\n"
                "Id=carrel-graph-rebuild.service\nActiveState=activating\nSubState=start\nResult=success\n")
        timers = service.parse_systemctl_show(
            "Id=carrel-scan.timer\nActiveState=active\nNextElapseUSecRealtime=Sat 2026-09-06 12:01:00 CST\n"
            "LastTriggerUSec=Sat 2026-09-06 12:00:00 CST\n")
        rows = service.timer_health(service.parse_systemctl_show(show), timers, {"graph-rebuild": 2})
        by = {r["key"]: r for r in rows}
        self.assertEqual((by["scan"]["ok"], by["scan"]["result"]), (False, "exit-code"))
        self.assertEqual((by["scan"]["last"], by["scan"]["next"]), ("09-06 12:00", "09-06 12:01"))
        self.assertTrue(by["worker"]["ok"])
        self.assertFalse(by["worker"]["running"])
        self.assertTrue(by["graph"]["running"])
        self.assertEqual(by["graph"]["defers"], 2)
        self.assertIsNone(by["gc"]["ok"])          # systemd has no record: unknown, not broken
        self.assertEqual(service.parse_systemctl_show(""), {})
        # A relative-time timer's show output has no NextElapseUSecRealtime: fill it from the wall-clock
        # microseconds given by list-timers (2026-09-09)
        rel = service.parse_systemctl_show("Id=carrel-gc.timer\nActiveState=active\nNextElapseUSecRealtime=\nLastTriggerUSec=\n"
                                           "\nId=carrel-worker.timer\nActiveState=active\nNextElapseUSecRealtime=Sat 2026-09-06 12:05:00 CST\n")
        merged = service.merge_timer_next(rel, {"carrel-gc.timer": int(time.mktime((2026, 9, 10, 4, 2, 30, 0, 0, -1))) * 1_000_000,
                                                "carrel-worker.timer": 1})
        self.assertEqual(service._short_systemd_ts(merged["carrel-gc.timer"]["NextElapseUSecRealtime"]), "09-10 04:02")
        self.assertEqual(service._short_systemd_ts(merged["carrel-worker.timer"]["NextElapseUSecRealtime"]), "09-06 12:05")   # an existing value is not overwritten
        js = _repo_file("app/kb_server/static/app.js")
        self.assertIn("h.timers", js)
        self.assertIn("定时任务", js)


class ServiceFixRegressionTests(_CodexFinalTestsSupport, unittest.TestCase):
    """Regressions for problems found in successive re-checks, health checks and reviews; each test's docstring
    gives the source and the symptom observed at the time."""

    def test_kb_config_service_layers_dir_defaults_and_validates_policy(self) -> None:
        from unittest import mock

        from kb_pipeline import db as dbm
        from kb_pipeline import discovery
        from kb_server import service

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_db = root / "state.sqlite3"
            dbm.init_db(state_db)
            mirror = root / "mirror"
            (mirror / "产品资料").mkdir(parents=True)
            with dbm.connect(state_db) as con:
                source, outcome = discovery.enroll(con, mirror, "产品资料")
                con.commit()
            self.assertEqual(outcome, "new")

            stub = mock.Mock()
            stub.state_db = state_db
            stub.embedding_base_url = "http://embedding.invalid/v1"
            limits = {"embedding_max_model_len": 4096, "effective_max_model_len": 4096,
                      "max_tokens_cap": 3276, "max_tokens_min": 128, "cap_ratio": 0.8,
                      "overlap_rule": "", "live": False}
            with mock.patch.object(service, "settings", return_value=stub), \
                    mock.patch.object(service, "chunk_limits", return_value=limits):
                got = service.get_kb_config(source.kb_id)
                # The product-docs directory preset (auto rebuild 1m AND 20%) shows up in effective;
                # graph_enabled has no preset, enabling the graph must be an explicit action
                self.assertFalse(got["effective"]["graph_enabled"])
                self.assertEqual(got["effective"]["graph_rebuild_interval"], "1m")

                saved = service.update_kb_config(source.kb_id, {
                    "graph_rebuild_interval": "2w",
                    "graph_rebuild_new_chunk_pct": "30%",
                    "graph_rebuild_operator": "or",
                })
                self.assertEqual(saved["config"]["graph_rebuild_interval"], "2w")

                # After clearing the key, effective falls back to the directory preset, matching the layering
                # in build_source
                service.update_kb_config(source.kb_id, {"graph_rebuild_interval": None})
                got2 = service.get_kb_config(source.kb_id)
                self.assertNotIn("graph_rebuild_interval", got2["config"])
                self.assertEqual(got2["effective"]["graph_rebuild_interval"], "1m")

                with self.assertRaises(ValueError):
                    service.update_kb_config(source.kb_id, {"graph_rebuild_interval": "abc"})
                with self.assertRaises(ValueError):
                    service.update_kb_config(source.kb_id, {"graph_rebuild_operator": "xor"})

    def test_overview_names_a_directory_that_became_a_refused_link(self) -> None:
        """Second review R01: a registered directory swapped for a symbolic link disappears from the live sources
        but not from the console: it is shown as directory_linked (not as vanished), so nothing is deleted and
        the reason is visible."""
        import shutil
        from unittest import mock

        from kb_pipeline import db as dbm
        from kb_pipeline import discovery
        from kb_server import service

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_db = root / "state.sqlite3"
            dbm.init_db(state_db)
            mirror = root / "mirror"; (mirror / "linked").mkdir(parents=True)
            elsewhere = root / "elsewhere"; elsewhere.mkdir()
            with dbm.connect(state_db) as con:
                discovery.enroll(con, mirror, "linked"); con.commit()
            shutil.rmtree(mirror / "linked"); (mirror / "linked").symlink_to(elsewhere, target_is_directory=True)
            stub = mock.Mock()
            stub.state_db = state_db
            stub.mirror_root = mirror
            stub.qdrant_inactive_retention_days = 7
            with mock.patch.object(service, "settings", return_value=stub), \
                    mock.patch.object(service, "_graph_status", return_value="disabled"), \
                    mock.patch.object(service, "_graph_build_info", return_value=None), \
                    mock.patch.object(service, "_graph_artifacts_exist", return_value=False), \
                    mock.patch.dict(os.environ, {"KB_MIRROR_ALLOW_LINKED_DIRS": ""}):
                by_dir = {kb["dir"]: kb for kb in service.overview()["kbs"]}
            self.assertEqual(by_dir["linked"]["state"], "directory_linked")
        js = _repo_file("app/kb_server/static/app.js")
        self.assertIn('kb.state === "directory_linked"', js)

    def test_gc_countdown_follows_whether_the_gc_will_actually_fire(self) -> None:
        """The countdown used to be shown the wrong way round: the branch where the directory still exists (GC
        skips it forever) gave a day count that ran down to 0 and then showed "permanently deleted in 0 days"
        for ever, while the branch where the directory is really gone (the retention period really running)
        gave no number at all. This pins both directions."""
        from unittest import mock

        from kb_pipeline import db as dbm
        from kb_pipeline import discovery
        from kb_server import service

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_db = root / "state.sqlite3"
            dbm.init_db(state_db)
            mirror = root / "mirror"
            for name in ("目录还在", "目录没了", "控制台注销"):
                (mirror / name).mkdir(parents=True)
            with dbm.connect(state_db) as con:
                for name in ("目录还在", "目录没了", "控制台注销"):
                    discovery.enroll(con, mirror, name)
                # Deactivated because the directory vanished, then put back (here it is simply never removed)
                discovery.mark_inactive(con, "kb_001", reason="directory_missing")
                # Deactivated because the directory vanished, and it really is gone
                discovery.mark_inactive(con, "kb_002", reason="directory_missing")
                # Unenrolled from the console: directory untouched, but the retention period is really running
                discovery.mark_inactive(con, "kb_003", reason="unenrolled")
                con.commit()
            (mirror / "目录没了").rmdir()

            stub = mock.Mock()
            stub.state_db = state_db
            stub.mirror_root = mirror
            stub.qdrant_inactive_retention_days = 7
            with mock.patch.object(service, "settings", return_value=stub), \
                    mock.patch.object(service, "_graph_status", return_value="disabled"), \
                    mock.patch.object(service, "_graph_build_info", return_value=None), \
                    mock.patch.object(service, "_graph_artifacts_exist", return_value=False):
                by_dir = {kb["dir"]: kb for kb in service.overview()["kbs"]}

            # The directory is back -> kb_sources_gc's exemption holds, never hard-deleted: no day count
            back = by_dir["目录还在"]
            self.assertEqual(back["state"], "inactive")
            self.assertTrue(back["gc_exempt"])
            self.assertIsNone(back["gc_in_seconds"])

            # The directory is really gone -> the retention period runs, and on expiry the collection and graph
            # are deleted together: a day count is required
            gone = by_dir["目录没了"]
            self.assertEqual(gone["state"], "inactive")
            self.assertFalse(gone["gc_exempt"])
            self.assertGreater(gone["gc_in_seconds"], 6 * 86400)

            # Unenrolling from the console gets no such exemption; the countdown runs whether or not the
            # directory exists
            unenrolled = by_dir["控制台注销"]
            self.assertFalse(unenrolled["gc_exempt"])
            self.assertGreater(unenrolled["gc_in_seconds"], 6 * 86400)

    def test_dir_file_count_matches_scanner_rules(self) -> None:
        from kb_server import service

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d = root / "资料"
            (d / "sub").mkdir(parents=True)
            (d / "a.pdf").write_bytes(b"x")
            (d / "sub" / "b.md").write_text("x", encoding="utf-8")
            (d / "Dockerfile").write_text("x", encoding="utf-8")   # whitelisted file name
            (d / ".DS_Store").write_bytes(b"x")                    # hidden file, skipped
            (d / "~$tmp.docx").write_bytes(b"x")                   # editor temp file, skipped
            (d / "c.unknown").write_bytes(b"x")                    # unsupported extension
            (d / "sub" / ".hidden" ).mkdir()
            (d / "sub" / ".hidden" / "d.md").write_text("x", encoding="utf-8")  # inside a hidden directory, skipped

            service._dir_count_cache.clear()
            self.assertEqual(service._dir_file_count(root, "资料"), 3)
            # TTL cache in effect: the directory changed but the cached value is still returned for a while
            (d / "e.txt").write_text("x", encoding="utf-8")
            self.assertEqual(service._dir_file_count(root, "资料"), 3)
            self.assertEqual(service._dir_file_count(root, "资料", ttl=0), 4)
            service._dir_count_cache.clear()

    def test_enroll_with_config_validates_first_and_applies(self) -> None:
        from unittest import mock

        from kb_pipeline import db as dbm
        from kb_pipeline import discovery
        from kb_server import service

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_db = root / "state.sqlite3"
            dbm.init_db(state_db)
            mirror = root / "mirror"
            (mirror / "新目录").mkdir(parents=True)
            stub = mock.Mock()
            stub.state_db = state_db
            stub.mirror_root = mirror
            limits = {"embedding_max_model_len": 4096, "effective_max_model_len": 4096,
                      "max_tokens_cap": 3276, "max_tokens_min": 128, "cap_ratio": 0.8,
                      "overlap_rule": "", "live": False}
            with mock.patch.object(service, "settings", return_value=stub), \
                    mock.patch.object(service, "chunk_limits", return_value=limits), \
                    mock.patch.object(service, "qdrant_client"), \
                    mock.patch.object(service, "ensure_collection", return_value=True), \
                    mock.patch.object(service.search_fts, "ensure_indices"), \
                    mock.patch.object(service, "kick_scan"), \
                    mock.patch.object(service, "kick_worker"):
                # Bad draft: validate before enrolling; on failure nothing has happened
                with self.assertRaises(ValueError):
                    service.enroll("新目录", {"max_tokens": 8, "overlap_tokens": 80})
                with dbm.connect(state_db) as con:
                    self.assertEqual(len(discovery.known_sources(con)), 0)
                # Good draft: enrolled, and the first parse already carries the user's policy
                r = service.enroll("新目录", {"max_tokens": 600, "overlap_tokens": 90})
                self.assertEqual(r["outcome"], "new")
                with dbm.connect(state_db) as con:
                    saved = discovery.get_config(con, r["kb_id"])
                self.assertEqual(saved["max_tokens"], 600)
                self.assertEqual(saved["overlap_tokens"], 90)

    def test_console_delete_kb_drops_everything(self) -> None:
        from unittest import mock

        from kb_pipeline import db as dbm
        from kb_pipeline import discovery, maintenance
        from kb_pipeline.vector.qdrant import graph_collection_short_name

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_db = root / "state.sqlite3"
            dbm.init_db(state_db)
            mirror = root / "mirror"
            (mirror / "库A").mkdir(parents=True)
            with dbm.connect(state_db) as con:
                source, _ = discovery.enroll(con, mirror, "库A")
                con.execute(
                    "INSERT INTO jobs(job_id, kb_id, collection, job_type, status, created_at, updated_at) "
                    "VALUES('j1', ?, ?, 'parse', 'queued', 0, 0)", (source.kb_id, source.collection))
                con.commit()

            # Pre-create the graph work directory / cache; delete must clear them as well
            short = graph_collection_short_name(source.collection)
            gw = root / "gw"
            (gw / "work" / short / "v1").mkdir(parents=True)
            (gw / "work" / short / "v1" / "graph.json").write_bytes(b"{}")
            (gw / "cache").mkdir(parents=True)
            (gw / "cache" / f"{short}.sqlite").write_bytes(b"x")

            stub = mock.Mock()
            stub.state_db = state_db
            stub.runtime_dir = root / "rt"
            stub.graph_work_dir = gw
            stub.cache_dir = root / "cache"
            with mock.patch("kb_pipeline.vector.qdrant.client", return_value=mock.Mock()), \
                    mock.patch("kb_pipeline.vector.qdrant.delete_collection", return_value={"deleted": True}), \
                    mock.patch("kb_pipeline.search_fts.delete_collection", return_value={"deleted": True}), \
                    mock.patch.object(maintenance, "_drop_graph_artifacts", return_value={}), \
                    mock.patch.object(maintenance, "_drop_neo4j_projection", return_value={}), \
                    mock.patch.object(maintenance, "_mineru_busy", return_value=None), \
                    mock.patch.object(maintenance, "_pgrep", return_value=False):
                # Deleting terminates first, then deletes: with a job running under a valid lease whose holder cannot
                # be confirmed dead (the lease carries no "host:pid"), the delete is refused, so a worker still alive
                # cannot recreate the collection that was just deleted
                with dbm.connect(state_db) as con:
                    con.execute("UPDATE jobs SET status='running', locked_until=? WHERE job_id='j1'",
                                (int(time.time()) + 3600,))
                    con.commit()
                # The 10 seconds of waiting for the lease to be released run on a fake clock instead of real sleep
                clock = mock.Mock(wraps=time)
                started = time.time()
                now = [started]
                clock.time.side_effect = lambda: now[0]
                clock.sleep.side_effect = lambda seconds: now.__setitem__(0, now[0] + seconds)
                with mock.patch.object(maintenance, "time", clock), \
                        self.assertRaisesRegex(ValueError, "Parse jobs could not be terminated"):
                    maintenance.delete_kb_now(stub, kb_id=source.kb_id)
                self.assertGreaterEqual(now[0] - started, 10.0)                   # gave up only after the full wait
                with dbm.connect(state_db) as con:
                    con.execute("UPDATE jobs SET status='queued' WHERE job_id='j1'")
                    con.commit()

                entry = maintenance.delete_kb_now(stub, kb_id=source.kb_id)
                self.assertTrue(entry["forgotten"])
                self.assertEqual(entry["errors"], [])
                self.assertFalse((gw / "work" / short).exists())               # graph work directory cleared
                self.assertFalse((gw / "cache" / f"{short}.sqlite").exists())   # graph build cache cleared
                with dbm.connect(state_db) as con:
                    self.assertEqual(len(discovery.known_sources(con)), 0)   # row forgotten
                    left = con.execute("SELECT COUNT(*) FROM jobs WHERE kb_id=?", (source.kb_id,)).fetchone()[0]
                self.assertEqual(int(left), 0)                        # queued jobs removed with the delete
                with self.assertRaises(KeyError):
                    maintenance.delete_kb_now(stub, kb_id=source.kb_id)

    def test_graph_status_mirrors_parse_dot_logic(self) -> None:
        from kb_pipeline import db as dbm
        from kb_pipeline import discovery
        from kb_server import service

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_db = root / "state.sqlite3"
            dbm.init_db(state_db)
            mirror = root / "mirror"
            (mirror / "产品资料").mkdir(parents=True)   # directory preset: graph enabled
            (mirror / "普通库").mkdir()
            with dbm.connect(state_db) as con:
                s1, _ = discovery.enroll(con, mirror, "产品资料")
                s2, _ = discovery.enroll(con, mirror, "普通库")
                # The directory preset does not enable the graph by itself: a freshly opened KB reports the
                # graph status as disabled
                self.assertEqual(
                    service._graph_status(con, s1.kb_id, "产品资料", discovery.get_config(con, s1.kb_id)),
                    "disabled")
                discovery.set_config(con, s1.kb_id, {"graph_enabled": True})
                cfg1 = discovery.get_config(con, s1.kb_id)
                self.assertEqual(service._graph_status(con, s2.kb_id, "普通库", {}), "disabled")
                self.assertEqual(service._graph_status(con, s1.kb_id, "产品资料", cfg1), "pending")
                con.execute(
                    "INSERT INTO graph_builds(graph_build_id, source_key, kb_id, source_collection, graph_version, status, started_at) "
                    "VALUES('g1', ?, ?, ?, 'v1', 'running', 1)", (s1.kb_id, s1.kb_id, s1.collection))
                self.assertEqual(service._graph_status(con, s1.kb_id, "产品资料", cfg1), "running")
                con.execute("UPDATE graph_builds SET status='failed' WHERE graph_build_id='g1'")
                self.assertEqual(service._graph_status(con, s1.kb_id, "产品资料", cfg1), "failed")
                con.execute(
                    "INSERT INTO graph_builds(graph_build_id, source_key, kb_id, source_collection, graph_version, status, started_at) "
                    "VALUES('g2', ?, ?, ?, 'v2', 'done', 2)", (s1.kb_id, s1.kb_id, s1.collection))
                self.assertEqual(service._graph_status(con, s1.kb_id, "产品资料", cfg1), "ok")
                # Stage check-in chain: begin -> set_graph_build_stage -> visible on the status panel
                bid = dbm.begin_graph_build(con, source_key=s1.kb_id, kb_id=s1.kb_id,
                                            source_collection=s1.collection, graph_version="v3")
                dbm.set_graph_build_stage(con, bid, "GraphRAG 索引")
                self.assertEqual(service._graph_status(con, s1.kb_id, "产品资料", cfg1), "running")
                info = service._graph_build_info(con, s1.kb_id)
                self.assertEqual(info["stage"], "GraphRAG 索引")

    def test_trigger_graph_build_validations(self) -> None:
        from unittest import mock

        from kb_pipeline import db as dbm
        from kb_pipeline import discovery
        from kb_server import service

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_db = root / "state.sqlite3"
            dbm.init_db(state_db)
            mirror = root / "mirror"
            (mirror / "库B").mkdir(parents=True)
            with dbm.connect(state_db) as con:
                s, _ = discovery.enroll(con, mirror, "库B")
                con.execute(
                    "INSERT INTO jobs(job_id, kb_id, collection, job_type, status, created_at, updated_at) "
                    "VALUES('jq', ?, ?, 'parse', 'queued', 0, 0)", (s.kb_id, s.collection))
                con.commit()
            stub = mock.Mock()
            stub.state_db = state_db
            stub.mirror_root = mirror          # the directory existence check needs a real path
            stub.runtime_dir = root / "runtime"   # where the global build lock lives
            with mock.patch.object(service, "settings", return_value=stub), \
                    mock.patch.object(service, "_spawn_graph_build", return_value={"started": True}) as spawn:
                with self.assertRaises(ValueError) as ctx:   # parsing: refused with the agreed message
                    service.trigger_graph_build(s.kb_id)
                self.assertIn("is being parsed", str(ctx.exception))
                with dbm.connect(state_db) as con:
                    con.execute("DELETE FROM jobs")
                    con.commit()
                with self.assertRaises(ValueError):          # graph not enabled
                    service.trigger_graph_build(s.kb_id)
                with dbm.connect(state_db) as con:
                    discovery.set_config(con, s.kb_id, {"graph_enabled": True})
                    con.commit()
                with self.assertRaises(ValueError):          # not all models chosen
                    service.trigger_graph_build(s.kb_id)
                with dbm.connect(state_db) as con:
                    discovery.set_config(con, s.kb_id,
                                         {"graph_llm": {"extract": "m", "summarize": "m"}})
                    con.commit()
                # Another KB holds the global build lock: say so plainly, never answer "started"
                # (Codex 2026-09-13 F04)
                with mock.patch.object(service, "build_lock_held", return_value=True):
                    with self.assertRaises(ValueError) as ctx_lock:
                        service.trigger_graph_build(s.kb_id)
                    self.assertIn("Another knowledge base is building its graph", str(ctx_lock.exception))
                spawn.assert_not_called()
                self.assertEqual(service.trigger_graph_build(s.kb_id), {"started": True})
                # When the directory has vanished (not yet marked deactivated by the scan) give a readable
                # reason instead of letting the detached child exit quietly with unknown source
                import shutil as _shutil
                _shutil.rmtree(mirror / "库B")
                with self.assertRaises(ValueError) as ctx2:
                    service.trigger_graph_build(s.kb_id)
                self.assertIn("directory", str(ctx2.exception))
                spawn.assert_called_once()

    def test_delete_graph_only_keeps_the_kb(self) -> None:
        from unittest import mock

        from kb_pipeline import db as dbm
        from kb_pipeline import discovery, maintenance
        from kb_pipeline.vector.qdrant import graph_collection_short_name

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_db = root / "state.sqlite3"
            dbm.init_db(state_db)
            mirror = root / "mirror"
            (mirror / "库C").mkdir(parents=True)
            with dbm.connect(state_db) as con:
                s, _ = discovery.enroll(con, mirror, "库C")
                discovery.set_config(con, s.kb_id, {"graph_enabled": True})
                # Represent a genuinely running build with "this host + a live pid + a fresh heartbeat",
                # otherwise reconcile_stale_graph_builds would (correctly) judge it a stale record.
                con.execute(
                    "INSERT INTO graph_builds(graph_build_id, source_key, kb_id, source_collection, graph_version, "
                    "status, started_at, worker_host, worker_pid, heartbeat_at) "
                    "VALUES('gb', ?, ?, ?, 'v1', 'running', ?, ?, ?, ?)",
                    (s.kb_id, s.kb_id, s.collection, int(time.time()),
                     socket.gethostname(), os.getpid(), int(time.time())))
                con.commit()
            short = graph_collection_short_name(s.collection)
            gw = root / "gw"
            (gw / "work" / short / "v1").mkdir(parents=True)
            (gw / "work" / short / "v1" / "x").write_bytes(b"x")

            stub = mock.Mock()
            stub.state_db = state_db
            stub.graph_work_dir = gw
            stub.runtime_dir = root
            with mock.patch("kb_pipeline.vector.qdrant.client", return_value=mock.Mock()), \
                    mock.patch.object(maintenance, "_drop_graph_artifacts", return_value={}), \
                    mock.patch.object(maintenance, "_drop_neo4j_projection", return_value={}):
                # A build in progress no longer blocks the delete: terminate first, then clear the artifacts
                # and the cache. worker_pid is this very process, and _terminate_pid recognises "that is me"
                # and refuses to shoot -- without that line of defence this test would SIGTERM pytest itself.
                entry = maintenance.delete_graph_now(stub, kb_id=s.kb_id)
            self.assertEqual(entry["build_stopped"]["terminated"], "skipped:self")
            self.assertTrue(entry["build_stopped"]["stopped"])
            self.assertEqual(entry["errors"], [])
            self.assertTrue(entry["graph_enabled_cleared"])
            self.assertFalse((gw / "work" / short).exists())          # graph workspace cleared
            with dbm.connect(state_db) as con:
                self.assertEqual(len(discovery.known_sources(con)), 1)  # the knowledge base itself stays
                left = con.execute("SELECT COUNT(*) FROM graph_builds WHERE kb_id=?", (s.kb_id,)).fetchone()[0]
                self.assertEqual(int(left), 0)                          # build records cleared
                cfgd = discovery.get_config(con, s.kb_id)
            self.assertNotIn("graph_enabled", cfgd)                     # the switch is greyed out again
            self.assertFalse(cfgd.get("graph_paused"))

    def test_partly_failed_graph_delete_is_not_rebuilt_by_the_scheduler(self) -> None:
        """2026-09-29 audit: when an external store fails to delete, the build records are cleared all the same and
        the switch stays on; the base is then "switch on, never built" and the next maintenance round rebuilds
        in full the graph the user asked to delete. The pause is recorded as well now, so automatic maintenance
        yields; a retried delete that succeeds clears it."""
        from types import SimpleNamespace
        from unittest import mock

        from kb_pipeline import db as dbm
        from kb_pipeline import discovery, maintenance
        from kb_pipeline.graph.build import evaluate_rebuild

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_db = root / "state.sqlite3"
            dbm.init_db(state_db)
            mirror = root / "mirror"
            (mirror / "docs").mkdir(parents=True)
            with dbm.connect(state_db) as con:
                s, _ = discovery.enroll(con, mirror, "docs")
                discovery.set_config(con, s.kb_id, {"graph_enabled": True})
                con.execute(
                    "INSERT INTO graph_builds(graph_build_id, source_key, kb_id, source_collection, graph_version, status, "
                    "started_at, finished_at) VALUES('g1', ?, ?, ?, 'v1', 'done', 100, 110)", (s.kb_id, s.kb_id, s.collection))
                con.commit()
            stub = mock.Mock()
            stub.state_db = state_db
            stub.graph_work_dir = root / "gw"
            stub.runtime_dir = root
            with mock.patch("kb_pipeline.vector.qdrant.client", return_value=mock.Mock()), \
                    mock.patch.object(maintenance, "_drop_graph_artifacts", return_value={}), \
                    mock.patch.object(maintenance, "_drop_neo4j_projection", side_effect=RuntimeError("neo4j restarting")):
                entry = maintenance.delete_graph_now(stub, kb_id=s.kb_id)
            self.assertFalse(entry["graph_enabled_cleared"])
            self.assertEqual(len(entry["errors"]), 1)
            with dbm.connect(state_db) as con:
                cfgd = discovery.get_config(con, s.kb_id)
                row = con.execute("SELECT * FROM kb_sources WHERE kb_id = ?", (s.kb_id,)).fetchone()
            self.assertEqual((cfgd.get("graph_enabled"), cfgd.get("graph_paused")), (True, True))
            source = discovery.source_from_row(mirror, row)
            decision = evaluate_rebuild(SimpleNamespace(state_db=state_db, sources={s.kb_id: source}), source_key=s.kb_id, source=source)
            self.assertEqual((decision["due"], decision["reason"]), (False, "paused_by_operator"))   # not rebuilt behind the user's back
            with mock.patch("kb_pipeline.vector.qdrant.client", return_value=mock.Mock()), \
                    mock.patch.object(maintenance, "_drop_graph_artifacts", return_value={}), \
                    mock.patch.object(maintenance, "_drop_neo4j_projection", return_value={}):
                entry = maintenance.delete_graph_now(stub, kb_id=s.kb_id)                          # the retry succeeds
            self.assertTrue(entry["graph_enabled_cleared"])
            with dbm.connect(state_db) as con:
                cfgd = discovery.get_config(con, s.kb_id)
            self.assertNotIn("graph_enabled", cfgd)
            self.assertFalse(cfgd.get("graph_paused"))

    def test_saved_key_is_not_reused_for_a_different_endpoint(self) -> None:
        """Security review 2026-09-28 F01 / F07: when editing a model, an empty key is reused only for the same
        endpoint (scheme + host + port, same protocol); if the address or protocol changes it must be entered
        again, and the stored key must not follow the probe to the new address. The probe does not follow
        redirects."""
        from unittest import mock

        from kb_pipeline import db as dbm
        from kb_server import service

        with tempfile.TemporaryDirectory() as tmp:
            state_db = Path(tmp) / "state.sqlite3"
            dbm.init_db(state_db)
            stub = mock.Mock()
            stub.state_db = state_db

            def resp(status=200, content="OK", location=None):
                r = mock.Mock()
                r.status_code = status
                r.text = "body"
                r.headers = {"location": location} if location else {}
                r.json.return_value = {"choices": [{"message": {"content": content}}]}
                return r

            with mock.patch.object(service, "settings", return_value=stub):
                with mock.patch.object(service.requests, "post", return_value=resp()) as post:
                    service.save_llm({"name": "m", "base_url": "http://a:8000/v1", "model_id": "x", "api_key": "sk-old"})
                    self.assertFalse(post.call_args.kwargs["allow_redirects"])      # probe with key: no redirects
                    # Same endpoint, different path: reused
                    service.save_llm({"name": "m", "base_url": "http://a:8000/v2", "model_id": "x", "api_key": ""})
                    self.assertIn("Bearer sk-old", str(post.call_args))
                    post.reset_mock()
                    # Different host: refused, and no probe was even sent
                    with self.assertRaisesRegex(ValueError, "enter the API key again"):
                        service.save_llm({"name": "m", "base_url": "http://b:8000/v1", "model_id": "x", "api_key": ""})
                    post.assert_not_called()
                    # A different port / protocol is refused likewise
                    with self.assertRaisesRegex(ValueError, "enter the API key again"):
                        service.save_llm({"name": "m", "base_url": "http://a:9000/v1", "model_id": "x", "api_key": ""})
                    with self.assertRaisesRegex(ValueError, "enter the API key again"):
                        service.save_llm({"name": "m", "base_url": "http://a:8000/v1", "model_id": "x", "api_key": "", "protocol": "anthropic"})
                    # With the key explicitly cleared the address may change (an endpoint needing no key);
                    # changing the address together with a new key is fine too
                    service.save_llm({"name": "m", "base_url": "http://b:8000/v1", "model_id": "x", "api_key": "", "clear_api_key": True})
                    self.assertNotIn("Bearer", str(post.call_args))
                    service.save_llm({"name": "m", "base_url": "http://c:8000/v1", "model_id": "x", "api_key": "sk-new"})
                    self.assertIn("Bearer sk-new", str(post.call_args))
                with mock.patch.object(service.requests, "post", return_value=resp(307, location="https://elsewhere.example/v1")):
                    with self.assertRaisesRegex(ValueError, "redirected"):
                        service.save_llm({"name": "r", "base_url": "http://a:8000/v1", "model_id": "x", "api_key": "sk-r"})
        llm_src = _repo_file("app/kb_pipeline/graph/llm.py")
        self.assertEqual(llm_src.count("allow_redirects=False"), 3)
        self.assertIn("allow_redirects=False", _repo_file("app/kb_pipeline/embedding/visual.py"))
        self.assertIn("allow_redirects=False", _repo_file("app/kb_search/channels.py"))

    def test_graph_status_ignores_rolled_back_versions(self) -> None:
        """The version rejected by a rollback keeps its record (status rolled_back); the status card and the graph status
        ignore it and show the version rolled back to."""
        from kb_pipeline import db as dbm
        from kb_server import service

        with tempfile.TemporaryDirectory() as tmp:
            state_db = Path(tmp) / "s.db"
            dbm.init_db(state_db)
            with dbm.connect(state_db) as con:
                con.execute("INSERT INTO kb_sources(kb_id, collection, source_root, status, first_seen_at, last_seen_at, config_json) "
                            "VALUES('kb_1', 'kb_1', 'dir', 'active', 0, 0, '{}')")
                for bid, status, started in (("good", "done", 100), ("bad", "rolled_back", 200)):
                    con.execute(
                        "INSERT INTO graph_builds(graph_build_id, source_key, kb_id, source_collection, graph_version, status, "
                        "started_at, finished_at, build_kind) VALUES(?, 'k', 'kb_1', 'kb_1', ?, ?, ?, ?, 'full')",
                        (bid, "v-" + bid, status, started, started + 10))
                self.assertEqual(service._graph_status(con, "kb_1", "dir", {"graph_enabled": True}), "ok")
                info = service._graph_build_info(con, "kb_1")
                self.assertEqual((info["graph_version"], info["status"]), ("v-good", "done"))

    def test_save_llm_requires_connectivity_ok(self) -> None:
        from unittest import mock

        from kb_pipeline import db as dbm
        from kb_server import service

        with tempfile.TemporaryDirectory() as tmp:
            state_db = Path(tmp) / "state.sqlite3"
            dbm.init_db(state_db)
            stub = mock.Mock()
            stub.state_db = state_db

            def fake_resp(content, status=200):
                r = mock.Mock()
                r.status_code = status
                r.text = "err-body"
                r.json.return_value = {"choices": [{"message": {"content": content}}]}
                return r

            with mock.patch.object(service, "settings", return_value=stub):
                with mock.patch.object(service.requests, "post", return_value=fake_resp("OK")) as post:
                    saved = service.save_llm({"name": "m1", "base_url": "http://x/v1",
                                              "model_id": "mid", "api_key": "sk-1"})
                    self.assertEqual(saved["name"], "m1")
                    self.assertEqual(saved["protocol"], "openai")          # default protocol
                    self.assertIn("Bearer sk-1", str(post.call_args))     # the probe carried the key
                    probe_body = post.call_args.kwargs["json"]
                    self.assertNotIn("enable_thinking", probe_body)       # thinking mode left to the model default
                with mock.patch.object(service.requests, "post", return_value=fake_resp("你好")):
                    with self.assertRaises(ValueError):                    # reply lacks ok: refuse to save
                        service.save_llm({"name": "m2", "base_url": "http://x/v1",
                                          "model_id": "mid", "api_key": ""})
                with dbm.connect(state_db) as con:
                    self.assertIsNone(dbm.get_llm(con, "m2"))
                with mock.patch.object(service.requests, "post", return_value=fake_resp("ok!")) as post2:
                    service.save_llm({"name": "m1", "base_url": "http://x/v1",
                                      "model_id": "mid2", "api_key": ""})  # editing with an empty key
                    self.assertIn("Bearer sk-1", str(post2.call_args))     # probe with the stored key
                with mock.patch.object(service.requests, "post",
                                       side_effect=service.requests.ConnectionError("boom")):
                    with self.assertRaises(ValueError):                    # network unreachable: refuse to save
                        service.save_llm({"name": "m3", "base_url": "http://x/v1",
                                          "model_id": "mid", "api_key": "k"})

                def fake_anthropic_resp(text):
                    r = mock.Mock()
                    r.status_code = 200
                    r.text = "err-body"
                    r.json.return_value = {"content": [{"type": "thinking", "thinking": "…"},
                                                       {"type": "text", "text": text}]}
                    return r

                with mock.patch.object(service.requests, "post",
                                       return_value=fake_anthropic_resp("OK")) as post3:
                    saved = service.save_llm({"name": "m4", "base_url": "https://open.bigmodel.cn/api/anthropic",
                                              "model_id": "glm-x", "api_key": "ak", "protocol": "anthropic"})
                    self.assertEqual(saved["protocol"], "anthropic")
                    # URL joining follows the same litellm rule as the build side: /v1/messages is appended
                    # automatically
                    self.assertEqual(post3.call_args[0][0],
                                     "https://open.bigmodel.cn/api/anthropic/v1/messages")
                    hdrs = post3.call_args.kwargs["headers"]
                    self.assertEqual(hdrs["x-api-key"], "ak")
                    self.assertEqual(hdrs["anthropic-version"], "2023-06-01")
                    self.assertNotIn("Authorization", hdrs)                # no Bearer mixed in
                with self.assertRaises(ValueError):                        # unknown protocol refused
                    service.save_llm({"name": "m5", "base_url": "http://x/v1",
                                      "model_id": "mid", "protocol": "grpc"})

    def test_connectivity_probe_does_not_hold_the_state_db_write_lock(self) -> None:
        """Saving a model with the key cleared used to write the row first and then probe connectivity inside the
        same uncommitted transaction: with a slow endpoint the probe takes tens of seconds, and the state
        database's write lock was held all that time while parse and graph build writes waited. The probe now runs
        outside the transaction; a failed probe changes nothing."""
        import sqlite3
        from unittest import mock

        from kb_pipeline import db as dbm
        from kb_server import service

        with tempfile.TemporaryDirectory() as tmp:
            state_db = Path(tmp) / "state.sqlite3"
            dbm.init_db(state_db)
            with dbm.connect(state_db) as con:
                dbm.upsert_llm(con, name="m", base_url="http://a:8000/v1", api_key="sk-old", model_id="x")
            stub = mock.Mock()
            stub.state_db = state_db
            seen: dict[str, Any] = {}

            def probe(base_url, api_key, model_id, protocol="openai"):
                seen["probe_key"] = api_key
                other = sqlite3.connect(str(state_db), timeout=0.2)        # another process writes meanwhile
                try:
                    other.execute("INSERT INTO app_config(key, value, updated_at) VALUES('written-during-probe', '1', 0)")
                    other.commit()
                    seen["stored_key_during_probe"] = other.execute(
                        "SELECT api_key FROM llm_registry WHERE name='m'").fetchone()[0]
                finally:
                    other.close()

            with mock.patch.object(service, "settings", return_value=stub), \
                    mock.patch.object(service, "_probe_llm", side_effect=probe):
                saved = service.save_llm({"name": "m", "base_url": "http://b:8000/v1", "model_id": "x",
                                          "api_key": "", "clear_api_key": True})
            self.assertEqual(seen, {"probe_key": "", "stored_key_during_probe": "sk-old"})
            self.assertFalse(saved["has_api_key"])
            with dbm.connect(state_db) as con:
                row = dbm.get_llm(con, "m")
                self.assertEqual((row["api_key"], row["base_url"]), ("", "http://b:8000/v1"))
                dbm.upsert_llm(con, name="k", base_url="http://a:8000/v1", api_key="sk-keep", model_id="x")
            with mock.patch.object(service, "settings", return_value=stub), \
                    mock.patch.object(service, "_probe_llm", side_effect=ValueError("Connectivity test failed: HTTP 500")):
                with self.assertRaises(ValueError):
                    service.save_llm({"name": "k", "base_url": "http://c:8000/v1", "model_id": "y",
                                      "api_key": "", "clear_api_key": True})
            with dbm.connect(state_db) as con:
                row = dbm.get_llm(con, "k")
            self.assertEqual((row["api_key"], row["base_url"], row["model_id"]), ("sk-keep", "http://a:8000/v1", "x"))

    def test_delete_cancels_pending_jobs_before_touching_storage(self) -> None:
        """C1: deleting a KB spans several external network calls, during which SQLite holds no write lock.
        Queued jobs must be cancelled and committed first, otherwise a worker started in that window would
        claim one and recreate the deleted collection via ensure_collection, leaving an ownerless zombie
        collection."""
        from unittest import mock

        from kb_pipeline import discovery, maintenance

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); mirror = root / "mirror"; (mirror / "库Y").mkdir(parents=True)
            state = root / "s.db"; db.init_db(state)
            with db.connect(state) as con:
                src, _ = discovery.enroll(con, mirror, "库Y")
                for i in range(3):
                    con.execute(
                        "INSERT INTO jobs(job_id, kb_id, collection, job_type, status, priority, "
                        "next_attempt_at, created_at, updated_at) VALUES(?, ?, ?, 'parse', 'queued', 100, 0, 1, 1)",
                        (f"j{i}", src.kb_id, src.collection),
                    )
                con.commit()
            stub = SimpleNamespace(state_db=state, mirror_root=mirror, runtime_dir=root / "rt",
                                   qdrant_url="http://q", qdrant_api_key="", opensearch_url="http://o",
                                   cache_dir=root / "cache", graph_work_dir=root / "gw")
            seen: dict[str, int] = {}

            def spy_hard_delete(settings, con, *, kb_id, collection, errors):
                # The moment external storage starts being deleted, the queue must already be empty
                seen["pending_at_delete_time"] = int(con.execute(
                    "SELECT COUNT(*) FROM jobs WHERE kb_id=? AND status IN ('queued','retry')",
                    (kb_id,)).fetchone()[0])
                return {"forgotten": True}

            with mock.patch.object(maintenance, "_hard_delete_kb", side_effect=spy_hard_delete):
                result = maintenance.delete_kb_now(stub, kb_id=src.kb_id)
            self.assertEqual(seen["pending_at_delete_time"], 0)   # emptied before the delete
            self.assertEqual(result["cancelled_jobs"], 3)

    def test_backoff_jobs_do_not_count_as_parsing(self) -> None:
        """H3: a retry job waiting out its backoff cannot be claimed by a worker and must not count as
        "parsing" -- otherwise the console looks busy for nothing, kicks the worker every 10 seconds to no
        effect, and graph operations are blocked for up to an hour."""
        from unittest import mock

        from kb_server import service

        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"
            db.init_db(state)
            now = int(time.time())
            with db.connect(state) as con:
                def put(job_id: str, status: str, next_attempt_at: int) -> None:
                    con.execute(
                        "INSERT INTO jobs(job_id, kb_id, collection, job_type, status, priority, "
                        "next_attempt_at, created_at, updated_at) VALUES(?, 'kb_001', 'kb_001', 'parse', ?, 100, ?, 1, 1)",
                        (job_id, status, next_attempt_at),
                    )

                put("waiting", "retry", now + 1800)      # in backoff: not busy
                con.commit()
                stats = service._kb_stats(con, "kb_001")
                self.assertEqual(stats["jobs_active"], 0)
                self.assertEqual(stats["jobs_waiting"], 1)

                put("claimable", "retry", now - 5)       # claimable now: busy
                put("running", "running", 0)
                con.commit()
                stats = service._kb_stats(con, "kb_001")
                self.assertEqual(stats["jobs_active"], 2)
                self.assertEqual(stats["jobs_waiting"], 1)

    def test_rebuild_check_is_cached_and_only_for_graph_enabled_kbs(self) -> None:
        """2026-09-23: the progress of the auto-rebuild conditions is wired to the console -- through
        evaluate_rebuild, cached per KB for 60 seconds; KBs without a graph do not count."""
        from types import SimpleNamespace
        from unittest import mock

        from kb_pipeline.graph import build as build_mod
        from kb_server import service

        service._rebuild_check_cache.clear()
        verdict = {"source": "kb_x", "graph_enabled": True, "due": False, "operator": "and", "appends_since_full": 2,
                   "latest_graph_version": "v2", "baseline_graph_version": "v1",
                   "conditions": [{"name": "interval", "due": False, "interval_days": 14, "elapsed_days": 5.2, "last_finished_at": 1},
                                  {"name": "new_chunk_ratio", "due": False, "threshold": 0.2, "ratio": 0.007, "new_chunks": 76}]}
        cfg = SimpleNamespace(sources={"kb_x": SimpleNamespace(graph_enabled=True), "kb_off": SimpleNamespace(graph_enabled=False)})
        with mock.patch.object(build_mod, "evaluate_rebuild", return_value=verdict) as ev:
            first = service.rebuild_check_cached(cfg, "kb_x", now=1000.0)
            again = service.rebuild_check_cached(cfg, "kb_x", now=1030.0)
            later = service.rebuild_check_cached(cfg, "kb_x", now=1100.0)
            self.assertIsNone(service.rebuild_check_cached(cfg, "kb_off", now=1000.0))
            self.assertIsNone(service.rebuild_check_cached(cfg, "kb_unknown", now=1000.0))
        self.assertEqual(ev.call_count, 2)                                   # reused within 60 s, recomputed after
        self.assertIs(first, again)
        self.assertEqual(later["conditions"][1]["new_chunks"], 76)
        self.assertEqual(sorted(first), ["appends_since_full", "baseline_graph_version", "conditions", "due", "operator"])
        self.assertNotIn("latest_graph_build_id", first)                      # only the fields the frontend draws
        with mock.patch.object(build_mod, "evaluate_rebuild", side_effect=RuntimeError("boom")):
            service._rebuild_check_cache.clear()
            broken = service.rebuild_check_cached(cfg, "kb_x", now=2000.0)
        self.assertEqual((broken["due"], broken["reason"]), (False, "check_failed"))   # a failed check must not sink overview
        service._rebuild_check_cache.clear()
        src = _repo_file("app/kb_server/service.py")
        self.assertIn('entry["rebuild_check"] = rebuild_check_cached(cfg, entry["kb_id"])', src)
        html = _repo_file("app/kb_server/static/index.html")
        self.assertIn('自动重建·时间条件<span class="f-note" id="cfg-ri-progress" hidden></span>', html)      # the percentage sits after the label
        self.assertIn('自动重建·新增内容条件<span class="f-note" id="cfg-rp-progress" hidden></span>', html)
        self.assertNotIn("cfg-rebuild-progress", html)
        js = _repo_file("app/kb_server/static/app.js")
        self.assertIn("function rebuildProgressParts(rc)", js)
        self.assertIn('rebuildProgressParts(kb.rebuild_check)', js)
        for key in ("时间进度:{0}%", "新增占比:{0}%", "新增:{0} 条"):
            self.assertIn(f't("{key}"', js)

    def test_overview_reports_the_version_the_preview_draws(self) -> None:
        """B3 (backend): while a build is running, graph_build.graph_version is the new version and
        active_graph_version is the one the preview draws."""
        from kb_pipeline import discovery
        from kb_server import service

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "库").mkdir()
            state = root / "s.db"
            db.init_db(state)
            with db.connect(state) as con:
                src, _ = discovery.enroll(con, root, "库")
                done = db.begin_graph_build(con, source_key=src.kb_id, kb_id=src.kb_id,
                                            source_collection=src.collection, graph_version="v1")
                db.finish_graph_build(con, done, status="done", source_content_hash="h")
                con.execute("UPDATE graph_builds SET started_at=1000, finished_at=2000 WHERE graph_build_id=?", (done,))
                running = db.begin_graph_build(con, source_key=src.kb_id, kb_id=src.kb_id,
                                               source_collection=src.collection, graph_version="v2")
                con.execute("UPDATE graph_builds SET started_at=3000 WHERE graph_build_id=?", (running,))
                con.commit()
                info = service._graph_build_info(con, src.kb_id, src.collection)
        self.assertEqual((info["status"], info["graph_version"]), ("running", "v2"))
        self.assertEqual(info["active_graph_version"], "v1")
        self.assertIsNone(info["last_check"])          # D7: not checked yet
        self.assertEqual(info["stage_weights"], {})    # D8: no stage check-ins, no weights

    def test_overview_no_longer_reinitialises_the_schema_per_poll(self) -> None:
        """D3: the schema is created only once, in lifespan."""
        src = _repo_file("app/kb_server/service.py")
        body = src.split("def overview()", 1)[1].split("\ndef ", 1)[0]
        self.assertNotIn("db.init_db(", body)
        self.assertIn("db.init_db(", src.split("def init_state()", 1)[1].split("\ndef ", 1)[0])

    def test_stop_without_confirmed_death_never_fakes_cancelled_nor_deletes(self) -> None:
        from unittest import mock

        from kb_pipeline import maintenance

        with tempfile.TemporaryDirectory() as tmp:
            stub, s = self._running_build(tmp)
            for verdict in ("not-ours", "denied", "still-alive"):
                with mock.patch.object(maintenance, "_terminate_pid", return_value=verdict), \
                        mock.patch.object(maintenance, "_STOP_GRACE_SECONDS", 0.0):
                    entry = maintenance.stop_graph_build_now(stub, kb_id=s.kb_id, reason="test")
                    self.assertEqual((entry["stopped"], entry["status"], entry["terminated"]), (False, "running", verdict))
                    with mock.patch.object(maintenance, "_drop_graph_data", side_effect=AssertionError("不该删")) as drop:
                        with self.assertRaises(ValueError):
                            maintenance.delete_graph_now(stub, kb_id=s.kb_id)
                        self.assertEqual(drop.call_count, 0)
            # Confirmed dead (the process had no time to write its terminal state after SIGTERM): the fallback
            # writes cancelled
            with mock.patch.object(maintenance, "_terminate_pid", return_value="terminated"), \
                    mock.patch.object(maintenance, "_STOP_GRACE_SECONDS", 0.0):
                entry = maintenance.stop_graph_build_now(stub, kb_id=s.kb_id, reason="test")
            self.assertEqual((entry["stopped"], entry["status"]), (True, "cancelled"))
        src = _repo_file("app/kb_pipeline/maintenance.py")
        self.assertIn("guard = GraphBuildLock(settings)", src)                    # the lock is held throughout the graph delete

    def test_closing_a_kb_stops_its_build_and_blocks_publishing(self) -> None:
        from kb_pipeline import discovery
        from kb_pipeline.graph.build import kb_still_active

        with tempfile.TemporaryDirectory() as tmp:
            stub, s = self._running_build(tmp)
            self.assertTrue(kb_still_active(stub, s.kb_id))
            with db.connect(stub.state_db) as con:
                discovery.mark_inactive(con, s.kb_id, reason="unenrolled")
                con.commit()
            self.assertFalse(kb_still_active(stub, s.kb_id))
            self.assertTrue(kb_still_active(stub, "kb_nonexistent"))                # an unregistered KB is not blocked
        svc = _repo_file("app/kb_server/service.py")
        self.assertIn('stop_graph_build_now(cfg, kb_id=kb_id, reason="cancelled because the knowledge base was closed")', svc)
        build = _repo_file("app/kb_pipeline/graph/build.py")
        self.assertIn("if not dry_run and not kb_still_active(settings, source.kb_id):", build)

    def test_build_lock_covers_label_preparation_and_start_button_is_guarded(self) -> None:
        build = _repo_file("app/kb_pipeline/graph/build.py")
        self.assertEqual(build.count("lock.acquire()"), 1)
        self.assertLess(build.index("lock.acquire()"), build.index("source, schema_auto = ensure_schema_before_build(settings, source, stop=stop_event)"))
        js = _repo_file("app/kb_server/static/app.js")
        self.assertIn("if (buildStarting) return;", js)


class HardDeleteLifecycleTests(unittest.TestCase):
    """"Delete permanently": while it runs others can see it and cannot get in its way; one that did not finish is
    carried on by the next maintenance run and cannot be switched back on as it is."""

    def _kb(self, tmp: str, name: str = "库D"):
        from kb_pipeline import discovery

        root = Path(tmp)
        state = root / "state.sqlite3"
        if not state.exists():
            db.init_db(state)
        mirror = root / "mirror"
        (mirror / name).mkdir(parents=True, exist_ok=True)
        with db.connect(state) as con:
            source, _ = discovery.enroll(con, mirror, name)
            con.commit()
        stub = SimpleNamespace(state_db=state, mirror_root=mirror, runtime_dir=root / "rt", graph_work_dir=root / "gw",
                               cache_dir=root / "cache", qdrant_url="http://q", qdrant_api_key="", opensearch_url="http://o",
                               neo4j_password="x", qdrant_inactive_retention_days=7, parse_enabled=True)
        return stub, source

    @staticmethod
    def _stores(*, neo4j=None):
        """Stubs for the external stores: by default everything deletes; an exception for neo4j makes the graph store
        side fail."""
        from contextlib import ExitStack
        from unittest import mock

        from kb_pipeline import maintenance

        stack = ExitStack()
        stack.enter_context(mock.patch("kb_pipeline.vector.qdrant.client", return_value=mock.Mock()))
        stack.enter_context(mock.patch("kb_pipeline.vector.qdrant.delete_collection", return_value={"deleted": True}))
        stack.enter_context(mock.patch("kb_pipeline.search_fts.delete_collection", return_value={"deleted": True}))
        stack.enter_context(mock.patch.object(maintenance, "_drop_graph_artifacts", return_value={}))
        stack.enter_context(mock.patch.object(maintenance, "_drop_neo4j_projection",
                                              **({"side_effect": neo4j} if neo4j else {"return_value": {}})))
        stack.enter_context(mock.patch.object(maintenance, "service_busy", return_value=(False, [])))
        return stack

    def _row(self, stub, kb_id: str):
        with db.connect(stub.state_db) as con:
            return con.execute("SELECT * FROM kb_sources WHERE kb_id = ?", (kb_id,)).fetchone()

    def test_a_delete_that_partly_failed_is_retried_at_the_next_maintenance_run(self) -> None:
        """When a delete fails, inactive_at records the moment of the delete, and it used to get the retention period
        of an ordinary deactivated KB: the message said "retried at the next maintenance run", yet it waited the
        full 7 days, and the text left undeleted stayed in the keyword index and the graph store 7 more days."""
        from kb_pipeline import discovery, maintenance

        with tempfile.TemporaryDirectory() as tmp:
            stub, source = self._kb(tmp)
            _, closed = self._kb(tmp, "刚关的库")
            with db.connect(stub.state_db) as con:
                discovery.mark_inactive(con, closed.kb_id, reason="unenrolled")
                con.commit()
            with self._stores(neo4j=RuntimeError("neo4j restarting")):
                with self.assertRaises(maintenance.PartialDeleteError):
                    maintenance.delete_kb_now(stub, kb_id=source.kb_id)
            row = self._row(stub, source.kb_id)
            self.assertEqual((row["status"], row["inactive_reason"]), ("inactive", "delete_failed"))
            self.assertGreater(int(row["inactive_at"]), int(time.time()) - 60)          # deactivated just now, far from the end of retention
            with self._stores():
                result = maintenance.kb_sources_gc(stub, retention_days=7)               # the maintenance run that night
            self.assertEqual([e["kb_id"] for e in result["dropped"]], [source.kb_id])    # the KB closed just now waits for its retention as usual
            self.assertEqual(result["errors"], [])
            self.assertIsNone(self._row(stub, source.kb_id))
            self.assertIsNotNone(self._row(stub, closed.kb_id))

    def test_a_delete_cut_short_by_a_restart_is_picked_up_too(self) -> None:
        """The process died halfway through a delete and the registry row stayed at "deleting": maintenance carries on
        deleting it as well, and a directory still on disk does not exempt it."""
        from kb_pipeline import maintenance

        with tempfile.TemporaryDirectory() as tmp:
            stub, source = self._kb(tmp)
            with db.connect(stub.state_db) as con:
                con.execute("UPDATE kb_sources SET status='inactive', inactive_reason='deleting', inactive_at=? WHERE kb_id=?",
                            (int(time.time()), source.kb_id))
                con.commit()
            with self._stores():
                result = maintenance.kb_sources_gc(stub, retention_days=7)
            self.assertEqual([e["kb_id"] for e in result["dropped"]], [source.kb_id])
            self.assertIsNone(self._row(stub, source.kb_id))

    def test_a_half_deleted_kb_cannot_be_reopened_or_adopted(self) -> None:
        """For a KB whose delete failed the console used to say "removed (directory is back), re-enable to restore";
        doing so revived a half-deleted KB: the state database already cleared, points still left in the external
        stores. Now the overview reports "deletion unfinished", and re-enabling and adopting are both refused."""
        from unittest import mock

        from kb_pipeline import discovery
        from kb_server import service

        with tempfile.TemporaryDirectory() as tmp:
            stub, source = self._kb(tmp)
            (stub.mirror_root / "新名字").mkdir()
            for reason in ("delete_failed", "deleting"):
                with db.connect(stub.state_db) as con:
                    con.execute("UPDATE kb_sources SET status='inactive', inactive_reason=?, inactive_at=? WHERE kb_id=?",
                                (reason, int(time.time()) - 3 * 86400, source.kb_id))
                    con.commit()
                with mock.patch.object(service, "settings", return_value=stub):
                    entry = {kb["dir"]: kb for kb in service.overview()["kbs"]}["库D"]
                    self.assertEqual(entry["state"], "delete_failed", reason)
                    self.assertNotIn("gc_exempt", entry)                       # not "the directory is back, never deleted"
                    self.assertNotIn("gc_in_seconds", entry)                   # no countdown either: the next maintenance run deletes it
                    with self.assertRaisesRegex(ValueError, "permanent deletion of .* has not finished"):
                        service.enroll("库D")
                    with self.assertRaisesRegex(ValueError, "permanent deletion of .* has not finished"):
                        service.adopt_kb(source.kb_id, "新名字")
                row = self._row(stub, source.kb_id)
                self.assertEqual((row["status"], row["inactive_reason"], row["source_root"]), ("inactive", reason, "库D"))
            # The branch where the directory is gone too reports "deletion unfinished" as well
            (stub.mirror_root / "库D").rmdir()
            with mock.patch.object(service, "settings", return_value=stub):
                entry = {kb["dir"]: kb for kb in service.overview()["kbs"]}["库D"]
            self.assertEqual(entry["state"], "delete_failed")
            # An ordinary closed KB is not affected: it can be switched back on as usual
            _, other = self._kb(tmp, "另一个库")
            with db.connect(stub.state_db) as con:
                discovery.mark_inactive(con, other.kb_id, reason="unenrolled")
                _, outcome = discovery.enroll(con, stub.mirror_root, "另一个库")
            self.assertEqual(outcome, "reactivated")

    def test_delete_marks_the_row_and_holds_the_build_lock_until_it_is_done(self) -> None:
        """Deleting a KB spans several external stores and takes minutes for a large one. It used to hold no build lock
        and the registry row stayed active throughout: the scheduled check could start a build of the same KB in
        that window, and the check before a build publishes could not tell the KB was being deleted."""
        from unittest import mock

        from kb_pipeline import maintenance
        from kb_pipeline.graph.build import kb_still_active
        from kb_pipeline.graph.lock import GraphBuildLock, build_lock_held, build_lock_path
        from kb_pipeline.graph.schema_flow import SUGGEST_MARK_PREFIX

        with tempfile.TemporaryDirectory() as tmp:
            stub, source = self._kb(tmp)
            with db.connect(stub.state_db) as con:
                db.record_graph_check(con, source.kb_id, {"kind": "none"})
                db.set_app_config(con, SUGGEST_MARK_PREFIX + source.kb_id, {"origin": "manual", "started_at": 1})
                db.set_app_config(con, db.GRAPH_AUTO_RESUME_PREFIX + source.kb_id, {"graph_build_id": "g", "count": 1})
                con.commit()
            seen: dict[str, Any] = {}
            real = maintenance._hard_delete_kb

            def spy(settings, con, **kw):
                seen["lock_held"] = build_lock_held(build_lock_path(stub))
                row = self._row(stub, source.kb_id)                 # read on another connection: sees the committed state
                seen["row"] = (row["status"], row["inactive_reason"])
                seen["still_active"] = kb_still_active(stub, source.kb_id)
                with self.assertRaises(RuntimeError):               # a build arriving now cannot take the lock
                    GraphBuildLock(stub).acquire()
                return real(settings, con, **kw)

            with self._stores(), mock.patch.object(maintenance, "_hard_delete_kb", side_effect=spy):
                entry = maintenance.delete_kb_now(stub, kb_id=source.kb_id)
            self.assertEqual(seen, {"lock_held": True, "row": ("inactive", "deleting"), "still_active": False})
            self.assertTrue(entry["forgotten"])
            self.assertFalse(build_lock_held(build_lock_path(stub)))             # the lock is released once done
            with db.connect(stub.state_db) as con:
                self.assertIsNone(db.latest_graph_check(con, source.kb_id))      # ids are never reused, so these two would stay forever
                self.assertIsNone(db.get_app_config(con, SUGGEST_MARK_PREFIX + source.kb_id))
                self.assertIsNone(db.get_app_config(con, db.GRAPH_AUTO_RESUME_PREFIX + source.kb_id))   # the resume count
            # A failed delete releases the lock too
            stub2, source2 = self._kb(tmp, "库E")
            with self._stores(neo4j=RuntimeError("boom")):
                with self.assertRaises(maintenance.PartialDeleteError):
                    maintenance.delete_kb_now(stub2, kb_id=source2.kb_id)
            self.assertFalse(build_lock_held(build_lock_path(stub2)))

    def test_scan_leaves_a_kb_that_is_being_deleted_alone(self) -> None:
        """Once the registry row is marked as deleting, the scan that runs every minute took it for a deactivated KB and
        queued a soft-delete job for every file: the whole collection is being torn down, so those jobs only fail and
        retry, and pending jobs also make maintenance skip the KB. A closed KB is soft-deleted as usual."""
        import argparse
        from unittest import mock

        from kb_pipeline import cli, discovery

        with tempfile.TemporaryDirectory() as tmp:
            stub, doomed = self._kb(tmp)
            _, closed = self._kb(tmp, "关掉的库")
            _, other = self._kb(tmp, "别的库")
            for name in ("库D", "关掉的库", "别的库"):
                doc = stub.mirror_root / name / "a.md"
                doc.write_text(f"# {name}\n\n内容", encoding="utf-8")
                os.utime(doc, (time.time() - 3600, time.time() - 3600))
            stub.min_file_age_seconds = 30
            args = argparse.Namespace(env_file=None, source=None, limit=None, verbose=False, dry_run=False, rehash=False,
                                      requeue_failed=False, no_detect_deletes=False, force_kb_teardown=False,
                                      exit_code_on_recent=False)

            def scan() -> None:
                stub.sources = discovery.enrolled_sources(stub.state_db, stub.mirror_root)
                with mock.patch.object(cli, "load_settings", return_value=stub):
                    self.assertEqual(cli.cmd_scan(args), 0)

            scan()
            with db.connect(stub.state_db) as con:
                discovery.mark_inactive(con, closed.kb_id, reason="unenrolled")
                con.commit()
            for reason in ("deleting", "delete_failed"):
                with db.connect(stub.state_db) as con:
                    con.execute("UPDATE kb_sources SET status='inactive', inactive_reason=?, inactive_at=? WHERE kb_id=?",
                                (reason, int(time.time()), doomed.kb_id))
                    con.commit()
                scan()
                with db.connect(stub.state_db) as con:
                    live = {str(r[0]): int(r[1]) for r in con.execute(
                        "SELECT kb_id, COUNT(*) FROM files WHERE status != 'deleted' GROUP BY kb_id")}
                    deletes = [str(r[0]) for r in con.execute("SELECT kb_id FROM jobs WHERE job_type = 'delete'")]
                self.assertEqual(live, {doomed.kb_id: 1, other.kb_id: 1}, reason)
                self.assertEqual(deletes, [closed.kb_id], reason)

    def test_delete_is_refused_with_the_actual_reason_when_the_lock_is_taken(self) -> None:
        from kb_pipeline import maintenance
        from kb_pipeline.graph.lock import GraphBuildLock
        from kb_pipeline.graph.schema_flow import SUGGEST_MARK_PREFIX

        with tempfile.TemporaryDirectory() as tmp:
            stub, source = self._kb(tmp)
            holder = GraphBuildLock(stub)
            holder.acquire()
            try:
                with self._stores() as stores:
                    with self.assertRaisesRegex(ValueError, "Another knowledge base is building its graph or being deleted"):
                        maintenance.delete_kb_now(stub, kb_id=source.kb_id)
                    # This KB's own build holds the lock and is still extracting labels, with no build record yet:
                    # the message says so, not "another knowledge base"
                    with db.connect(stub.state_db) as con:
                        db.set_app_config(con, SUGGEST_MARK_PREFIX + source.kb_id,
                                          {"origin": "auto_blank", "started_at": int(time.time())})
                        con.commit()
                    with self.assertRaisesRegex(ValueError, "is extracting labels"):
                        maintenance.delete_kb_now(stub, kb_id=source.kb_id)
                    del stores
                row = self._row(stub, source.kb_id)
                self.assertEqual((row["status"], row["inactive_reason"]), ("active", None))      # refused = nothing touched
            finally:
                holder.release()

    def test_overview_reports_a_delete_in_progress_and_a_second_click_is_turned_away(self) -> None:
        """A permanent delete runs synchronously in the request thread. The page used to only grey out the button, and
        after a reload not even that was left, so it looked as if nothing happened. While the delete runs the overview
        reports deleting (visible after a reload or in another browser); once it is done or has failed it no longer
        does."""
        from unittest import mock

        from kb_pipeline import discovery, maintenance
        from kb_server import service

        with tempfile.TemporaryDirectory() as tmp:
            stub, source = self._kb(tmp)
            _, other = self._kb(tmp, "要建图的库")
            with db.connect(stub.state_db) as con:
                discovery.set_config(con, other.kb_id, {"graph_enabled": True, "graph_llm": {"extract": "m", "summarize": "m"}})
                con.commit()
            seen: dict[str, Any] = {}

            def during(settings, *, kb_id):
                seen["state"] = {kb["dir"]: kb for kb in service.overview()["kbs"]}["库D"]["state"]
                try:
                    service.delete_kb(kb_id)
                except ValueError as exc:
                    seen["second_click"] = str(exc)
                with mock.patch.object(service, "build_lock_held", return_value=True):      # the delete holds the build lock throughout
                    try:
                        service.trigger_graph_build(other.kb_id)
                    except ValueError as exc:
                        seen["build_elsewhere"] = str(exc)
                return {"kb_id": kb_id, "errors": []}

            with mock.patch.object(service, "settings", return_value=stub), \
                    mock.patch.object(service, "_spawn_graph_build", return_value={"started": True}) as spawn:
                with mock.patch.object(maintenance, "delete_kb_now", side_effect=during):
                    service.delete_kb(source.kb_id)
                self.assertEqual(seen, {"state": "deleting",
                                        "second_click": "This knowledge base is already being deleted; wait for it to finish",
                                        "build_elsewhere": "Another knowledge base is being permanently deleted; build the graph when that finishes"})
                spawn.assert_not_called()
                self.assertEqual(service._KB_DELETING, set())
                self.assertEqual({kb["dir"]: kb for kb in service.overview()["kbs"]}["库D"]["state"], "active")
                failed = maintenance.PartialDeleteError(source.kb_id, {}, ["kb_001: neo4j projection drop failed"])
                with mock.patch.object(maintenance, "delete_kb_now", side_effect=failed):
                    with self.assertRaisesRegex(ValueError, "Deletion incomplete.*continues at the next maintenance run.*neo4j projection drop failed"):
                        service.delete_kb(source.kb_id)
                self.assertEqual(service._KB_DELETING, set())                     # removed on failure too, or it shows as deleting forever

    def test_adopt_checks_the_directory_name_before_walking_it(self) -> None:
        """Adopting a directory used to join the name into a path, walk the whole tree and checksum a sample before it
        checked the name: a name of ".." or an absolute path walked outside the mirror first and was only then
        refused."""
        from unittest import mock

        from kb_pipeline import discovery
        from kb_server import service

        with tempfile.TemporaryDirectory() as tmp:
            stub, source = self._kb(tmp)
            outside = Path(tmp) / "outside"
            outside.mkdir()
            (outside / "a.md").write_text("x", encoding="utf-8")
            with mock.patch.object(service, "settings", return_value=stub), \
                    mock.patch.object(discovery, "directory_match_report",
                                      side_effect=AssertionError("walked before the name was checked")) as report:
                for bad in ("..", "../outside", str(outside), "/", "不存在的目录", ".hidden"):
                    with self.assertRaisesRegex(ValueError, "does not exist under the mirror root"):
                        service.adopt_kb(source.kb_id, bad)
                self.assertEqual(report.call_count, 0)
