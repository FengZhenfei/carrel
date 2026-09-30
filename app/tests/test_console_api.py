"""Console API layer and static pages: JS / HTML contracts, i18n, previews and progress."""
from __future__ import annotations

import json
import re
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from kb_pipeline import db
from kb_pipeline.parsers.common import page_idx

from _support import _CodexAudit20260906TestsSupport, _block, _local_file, _repo_file



def _scratch_dir(case: unittest.TestCase) -> Path:
    """A temporary directory removed when the test case ends. The tests used to call mkdtemp() directly and
    left their directories in the system temp dir on every run (2026-09-29 audit)."""
    tmp = tempfile.TemporaryDirectory()
    case.addCleanup(tmp.cleanup)
    return Path(tmp.name)


class ApiLayerTests(unittest.TestCase):
    """The API layer previously had zero coverage: error-code mapping, the same-origin guard and payload type
    validation had only ever been clicked through by hand on the real box. They are the first gate of every
    console action."""

    def _client(self, **service_patches):
        from unittest import mock

        from fastapi.testclient import TestClient

        from kb_server.main import create_app
        from kb_server import service

        patches = [mock.patch.object(service, name, value) for name, value in service_patches.items()]
        for patch_obj in patches:
            patch_obj.start()
            self.addCleanup(patch_obj.stop)
        # raise_server_exceptions=False: makes TestClient return the response the handler produced instead of
        # raising the exception straight into the test (in production this is the response uvicorn gets)
        return TestClient(create_app(), raise_server_exceptions=False)

    def test_error_mapping(self) -> None:
        from unittest import mock

        client = self._client(
            get_kb_config=mock.Mock(side_effect=KeyError("kb_404")),
            reparse_kb=mock.Mock(side_effect=ValueError("知识库未开启")),
            kb_files=mock.Mock(side_effect=RuntimeError("qdrant 崩了")),
        )
        self.assertEqual(client.get("/api/kbs/kb_404/config").status_code, 404)
        r = client.post("/api/kbs/kb_1/reparse", json={})
        self.assertEqual(r.status_code, 422)
        self.assertIn("未开启", r.json()["detail"])
        # Other exceptions used to be a bare 500 "Internal Server Error"; now they carry the type and the reason
        r = client.get("/api/kbs/kb_1/files")
        self.assertEqual(r.status_code, 500)
        self.assertIn("qdrant", r.json()["detail"])

    def test_cross_site_writes_are_rejected(self) -> None:
        from unittest import mock

        client = self._client(parse_now=mock.Mock(return_value={"kicked": True}))
        # A cross-site browser fetch always carries Origin; a forged site must be blocked before the business logic
        r = client.post("/api/kbs/kb_001/parse_now", json={}, headers={"Origin": "http://evil.example"})
        self.assertEqual(r.status_code, 403)
        # Same-origin is allowed
        r = client.post("/api/kbs/kb_001/parse_now", json={}, headers={"Origin": "http://testserver"})
        self.assertEqual(r.status_code, 200)
        # No Origin (curl / scripts) is allowed, keeping the command line usable
        self.assertEqual(client.post("/api/kbs/kb_001/parse_now", json={}).status_code, 200)
        # 2026-09-28 security review F01: same-origin compares the full origin — same host on another port and
        # localhost are no longer allowed
        r = client.post("/api/kbs/kb_001/parse_now", json={}, headers={"Origin": "http://testserver:8080"})
        self.assertEqual(r.status_code, 403)
        r = client.post("/api/kbs/kb_001/parse_now", json={}, headers={"Origin": "http://localhost"})
        self.assertEqual(r.status_code, 403)
        r = client.post("/api/kbs/kb_001/parse_now", json={}, headers={"Origin": "https://testserver"})
        self.assertEqual(r.status_code, 403)                                  # a different scheme is not same-origin either
        r = client.post("/api/kbs/kb_001/parse_now", json={},
                        headers={"Origin": "https://kb.example", "Host": "kb.example", "X-Forwarded-Proto": "https"})
        self.assertEqual(r.status_code, 200)                                  # reverse proxy: judged by the forwarded headers
        r = client.post("/api/kbs/kb_001/parse_now", json={}, headers={"Origin": "http://testserver:80"})
        self.assertEqual(r.status_code, 200)                                  # an explicit default port = same-origin

    def test_console_token_guards_the_api_when_set(self) -> None:
        """With KB_WEB_TOKEN set the whole /api requires a Bearer token; static pages are served as usual; when it is
        unset the behaviour is unchanged."""
        import os
        from unittest import mock

        # Second review R03: a token typed into the prompt is kept in memory for the page's lifetime, so a
        # browser that blocks localStorage still authenticates the retried request
        js = _repo_file("app/kb_server/static/app.js")
        self.assertIn('let memToken = ""', js)
        self.assertIn("memToken = v.trim();", js)
        self.assertIn("let token = memToken;", js)

        from fastapi.testclient import TestClient

        from kb_server import service
        from kb_server.main import create_app

        with mock.patch.object(service, "health", return_value={"ok": True}), \
                mock.patch.dict(os.environ, {"KB_WEB_TOKEN": "s3cret-token"}):
            client = TestClient(create_app(), raise_server_exceptions=False)
            self.assertEqual(client.get("/api/health").status_code, 401)
            self.assertEqual(client.get("/api/health", headers={"Authorization": "Bearer wrong"}).status_code, 401)
            self.assertEqual(client.get("/api/health", headers={"Authorization": "Bearer s3cret-token"}).status_code, 200)
            self.assertEqual(client.get("/").status_code, 200)                # the page itself needs no token; the front end will ask for it
        with mock.patch.object(service, "health", return_value={"ok": True}), \
                mock.patch.dict(os.environ, {"KB_WEB_TOKEN": ""}):
            client = TestClient(create_app(), raise_server_exceptions=False)
            self.assertEqual(client.get("/api/health").status_code, 200)
        js = _repo_file("app/kb_server/static/app.js")
        self.assertIn("function authHeaders()", js)
        # asked once: requests that were in flight while the token was typed resend with it (2026-09-29 audit)
        self.assertIn('res.status === 401 && ((authHeaders().Authorization || "") !== sentWith || await askToken())', js)
        self.assertIn('localStorage.getItem("kb.token")', js)
        example = _repo_file("config/knowledge-base.env.example")
        self.assertIn("KB_WEB_HOST=127.0.0.1", example)
        self.assertIn("KB_WEB_TOKEN=", example)
        self.assertIn('os.getenv("KB_WEB_HOST", "127.0.0.1")', _repo_file("app/kb_server/main.py"))

    def test_config_payload_type_guard(self) -> None:
        from unittest import mock

        client = self._client(update_kb_config=mock.Mock(return_value={"config": {}, "warnings": []}))
        bad = client.put("/api/kbs/kb_1/config", json={"max_tokens": "八百"})
        self.assertEqual(bad.status_code, 422)
        self.assertIn("must be an integer", bad.json()["detail"])
        self.assertEqual(client.put("/api/kbs/kb_1/config", json={"graph_enabled": "yes"}).status_code, 422)
        self.assertEqual(client.put("/api/kbs/kb_1/config", json={"graph_llm": ["a"]}).status_code, 422)
        # A valid payload is accepted as usual
        self.assertEqual(client.put("/api/kbs/kb_1/config", json={"max_tokens": 800}).status_code, 200)

    def test_endpoints_the_console_stopped_calling_are_gone(self) -> None:
        """Endpoints the front end stopped calling after the redesign do not stay exposed on the LAN: one of them,
        the preview that re-chunks with given parameters, re-ran parsing synchronously inside the web process. The
        chunk preview drawer uses files/{id}/chunks, which reads the chunks already stored, and the job timeline
        uses jobs/{id}; those two remain."""
        from kb_server import service
        from kb_server.api import router

        paths = {route.path for route in router.routes}
        for gone in ("/api/kbs/{kb_id}/chunk_preview", "/api/kbs/{kb_id}/jobs", "/api/kbs/{kb_id}/graph_builds",
                     "/api/jobs/failed", "/api/kbs/{kb_id}/unenroll_info"):
            self.assertNotIn(gone, paths)
        for kept in ("/api/kbs/{kb_id}/files/{file_id}/chunks", "/api/jobs/{job_id}", "/api/kbs/{kb_id}/graph_preview"):
            self.assertIn(kept, paths)
        for name in ("chunk_preview", "kb_jobs", "graph_builds", "failed_jobs", "unenroll_info"):
            self.assertFalse(hasattr(service, name), name)
        client = self._client()
        self.assertIn(client.post("/api/kbs/kb_1/chunk_preview", json={"file_id": "f"}).status_code, (404, 405))
        self.assertEqual(client.get("/api/kbs/kb_1/jobs").status_code, 404)
        self.assertEqual(client.get("/api/kbs/kb_1/graph_builds").status_code, 404)
        self.assertEqual(client.get("/api/kbs/kb_1/unenroll_info").status_code, 404)
        js = _repo_file("app/kb_server/static/app.js")
        for gone in ("chunk_preview", "/graph_builds", "jobs/failed", "unenroll_info", "state.corpus"):
            self.assertNotIn(gone, js, gone)

    def test_required_fields_and_unknown_service(self) -> None:
        from unittest import mock

        client = self._client(
            enroll=mock.Mock(return_value={"kb_id": "kb_1"}),
            retry_file=mock.Mock(return_value={"job_id": "j"}),
            restart_service=mock.Mock(side_effect=KeyError("nope")),
        )
        self.assertEqual(client.post("/api/enroll", json={"dir": ""}).status_code, 422)
        self.assertEqual(client.post("/api/files/retry", json={}).status_code, 422)
        self.assertEqual(client.post("/api/services/nope/restart", json={}).status_code, 404)


class BusyDotParityTests(unittest.TestCase):
    """Parsing and graph building use the same pulsing dot. Previously only the graph dot pulsed, so two dots in
    the same list expressed the same "running" state yet looked different."""

    def _app_js(self) -> str:
        return _repo_file("app/kb_server/static/app.js")

    def test_both_dots_pulse_while_work_is_running(self) -> None:
        source = self._app_js()
        kb_dot = source.split("function kbDot", 1)[1].split("\nfunction ", 1)[0]
        graph_dot = source.split("function graphDot", 1)[1].split("\nfunction ", 1)[0]
        self.assertIn('kb.jobs_active > 0) return "yellow pulse"', kb_dot)
        self.assertIn('"yellow pulse"', graph_dot)
        # Queued but not started does not pulse: pulsing means "really moving right now"
        self.assertIn('kb.files_pending > 0) return "yellow"', kb_dot)

    def test_pulse_css_keeps_both_layers(self) -> None:
        """.dot.pulse has two rules (the dot's own brightness + the ::after halo); only their combination gives the
        current effect. An earlier comment claimed it was "never used", and cleaning up by that comment would have
        changed the effect."""
        css = _repo_file("app/kb_server/static/index.html")
        self.assertIn(".dot.pulse::after{", css)
        self.assertIn(".dot.pulse{animation:ping", css)
        self.assertNotIn("却从未使用", css)


class DerivedSchemaPersistenceTests(unittest.TestCase):
    """Entity labels and the output language are products of "label extraction": read-only display, not form fields.

    2026-08-24: saving the configuration wiped them from the server although the user had never touched that
    field. The cause: saving unconditionally sent state.graphSchema (a purely front-end in-memory state) — while
    enrolling a new KB clears it and a failed config request leaves it empty, after which any save overwrites the
    values stored on the server with null. What was lost was a label table that had taken 2 hours 25 minutes to
    extract.
    """

    def _collect_body(self) -> str:
        source = _repo_file("app/kb_server/static/app.js")
        return source.split("function collectGraphConfig", 1)[1].split("\n}", 1)[0]

    def test_console_never_sends_the_derived_label_fields(self) -> None:
        """This is the invariant of that incident, and it is now stronger than the original fix: the console
        **never sends** labels or language at all, only "which version to use" (graph_schema_active), which the
        server expands. Without a version id nothing changes — no path is left that can write the labels as null."""
        body = self._collect_body()
        self.assertNotIn("graph_entity_types", body)
        self.assertNotIn("graph_language", body)
        self.assertIn("graph_schema_active", body)

    def test_the_selection_is_only_sent_when_the_user_actually_chose(self) -> None:
        """Not sent by default. The backend set_config has merge semantics: omitting the key = keep the old value.
        Only after you touched the dropdown, or just extracted a version, is there any "I want this version"."""
        body = self._collect_body()
        index = body.index("graph_schema_active")
        gate = body.rindex("state.graphSchema.dirty", 0, index)
        self.assertGreater(index, gate, "graph_schema_active 必须在 dirty 判定之内")

    def test_only_deliberate_actions_originate_dirty(self) -> None:
        """dirty can only be **produced** by two actions: extracting labels and switching version.

        The config-loading site also writes dirty: true, but that **preserves** an existing unsaved state rather
        than producing one — so it must be guarded by keepPending. Without that guard, opening the config page and
        casually clicking save once would turn into an "I chose this version" out of thin air.
        """
        source = _repo_file("app/kb_server/static/app.js")
        tune = source.split('$("#cfg-gtune")', 1)[1]
        self.assertIn("dirty: true", tune.split('$("#cfg-', 1)[0])
        change = source.split('$("#cfg-gver").addEventListener', 1)[1].split("});", 1)[0]
        self.assertIn("dirty = true", change)
        load = source.split("if (state.cfgKey !== key)", 1)[1].split("} else if (draft)", 1)[0]
        index = load.index("dirty: true")
        guard = load.rindex("keepPending", 0, index)
        self.assertGreater(index, guard, "载入路径的 dirty 必须被 keepPending 守着")

    def test_build_manifest_records_the_entity_types(self) -> None:
        """The graph-build output must record the type table used for this run. Recording only the language means
        that once the config is overwritten, the only clue left is the entity type distribution in Neo4j — that is
        reverse inference, and the model occasionally writes variants never configured (OPERATING_MODE /
        FUNCTIONAL_BLOCK seen in practice), so no clean original table can be inferred."""
        source = _repo_file("app/kb_pipeline/graph/build.py")
        block = source.split('"schema": {', 1)[1].split("},", 1)[0]
        self.assertIn('"language"', block)
        self.assertIn('"entity_types"', block)
        self.assertIn('"predicates"', block)


class SchemaVersionViewTests(unittest.TestCase):
    def test_pre_version_labels_surface_as_one_honest_placeholder(self) -> None:
        """Version management was added later. Labels extracted before it have no date / model / sampling record —
        the view gives them one fallback entry with the metadata honestly left empty, rather than fabricating one
        that looks real."""
        from kb_pipeline.limits import CURRENT_SCHEMA_VERSION_ID
        from kb_server.service import schema_versions_view

        view = schema_versions_view({
            "graph_entity_types": ["organization", "signal"],
            "graph_language": "Chinese",
        })
        self.assertEqual(view["active"], CURRENT_SCHEMA_VERSION_ID)
        self.assertEqual(len(view["versions"]), 1)
        entry = view["versions"][0]
        self.assertTrue(entry["legacy"])
        self.assertIsNone(entry["created_at"])
        self.assertIsNone(entry["model"])
        self.assertEqual(entry["entity_types"], ["organization", "signal"])

    def test_no_labels_at_all_means_no_versions(self) -> None:
        from kb_server.service import schema_versions_view

        view = schema_versions_view({})
        self.assertEqual(view, {"versions": [], "active": ""})

    def test_a_dangling_active_id_falls_back_instead_of_pointing_nowhere(self) -> None:
        from kb_pipeline.limits import CURRENT_SCHEMA_VERSION_ID
        from kb_server.service import schema_versions_view

        view = schema_versions_view({
            "graph_schema_versions": [],
            "graph_schema_active": "evicted",
            "graph_entity_types": ["signal"],
        })
        self.assertEqual(view["active"], CURRENT_SCHEMA_VERSION_ID)


class PendingSchemaSelectionTests(unittest.TestCase):
    """The selection of a freshly extracted, not yet saved version must not be washed away by an unrelated refetch.

    2026-08-24 kb_004: 21 labels were extracted (dirty=true), then the graph switch was turned off — toggling the
    switch triggers a config refetch, which reset dirty to false; the "save config" that followed therefore carried
    no selection, none of the 21 labels took effect, and the UI showed no hint at all. In the log the two PUTs were
    4 seconds apart, both 200; nothing visible tells you which one lost something.
    """

    def _app_js(self) -> str:
        return _repo_file("app/kb_server/static/app.js")

    def test_refetch_keeps_an_unsaved_selection_for_the_same_kb(self) -> None:
        source = self._app_js()
        block = source.split("if (state.cfgKey !== key)", 1)[1].split("} else if (draft)", 1)[0]
        self.assertIn("state.graphSchema.dirty && state.graphSchema.kbId === kb.kb_id", block)
        self.assertIn("dirty: true", block)
        self.assertIn("active: state.graphSchema.active", block)

    def test_the_pending_selection_is_scoped_to_its_own_kb(self) -> None:
        """A pending selection belongs to the KB it was extracted for. If it survives switching KBs, KB A's version
        id gets stored under KB B — the backend cannot find that id and the save is a straight 422."""
        source = self._app_js()
        self.assertIn("kbId: String(kbId || \"\")", source)
        tune = source.split('$("#cfg-gtune")', 1)[1].split('$("#cfg-', 1)[0]
        self.assertIn("kbId: kb.kb_id", tune)

    def test_the_unsaved_state_is_visible(self) -> None:
        """The silent loss was hard to track down because the UI gave no sign that "something is not saved yet"."""
        source = self._app_js()
        render = source.split("function renderGraphSchema", 1)[1].split("\nfunction ", 1)[0]
        self.assertIn("state.graphSchema.dirty", render)
        self.assertIn("未保存", render)


class GraphTabAndConfigPanelTests(unittest.TestCase):
    """2026-09-04 console reworked for the local entity-graph pipeline: retired fields removed, two model slots,
    unit chunk count / gleaning rounds, the schema layer (parent types / predicates / capability questions), and
    the graph tab's recall test and extraction preview."""

    def test_retired_fields_are_gone_and_new_knobs_are_wired(self) -> None:
        html = _repo_file("app/kb_server/static/index.html")
        js = _repo_file("app/kb_server/static/app.js")
        for retired in ('id="cfg-gcs"', 'id="cfg-gco"', 'id="cfg-gmcs"', 'id="cfg-glcc"', 'id="cfg-gc"'):
            self.assertNotIn(retired, html, retired)
        for needle in ('id="cfg-guc"', 'id="cfg-ggl"', 'id="cfg-gparents"', 'id="cfg-gpreds"', 'id="cfg-ge"', 'id="cfg-gs"', 'id="cfg-gt"'):
            self.assertIn(needle, html, needle)
        for gone in ('id="cfg-gq"', 'id="cfg-gq-draft"', 'id="gq-pick"'):        # capability questions withdrawn 2026-09-06
            self.assertNotIn(gone, html, gone)
        self.assertNotIn("competency", js)
        body = js.split("function collectGraphConfig", 1)[1].split("\n}", 1)[0]
        self.assertIn('graph_unit_chunks: requireInt("#cfg-guc"', body)
        self.assertIn('graph_max_gleanings: requireInt("#cfg-ggl"', body)
        for retired in ("graph_chunk_size", "graph_chunk_overlap", "graph_max_cluster_size", "graph_use_lcc", "community"):
            self.assertNotIn(retired, body, retired)
        self.assertNotIn("setGraphChunkHints", js)

    def test_schema_view_shows_parents_and_predicates(self) -> None:
        js = _repo_file("app/kb_server/static/app.js")
        render = js.split("function renderGraphSchema", 1)[1].split("\nfunction ", 1)[0]
        for needle in ('$("#cfg-gparents")', '$("#cfg-gpreds")'):
            self.assertIn(needle, render, needle)
        view = js.split("function setSchemaView", 1)[1].split("\nfunction ", 1)[0]
        for field in ("predicates", "parent_types"):
            self.assertIn(field, view, field)
        self.assertIn("个谓词", js.split("function versionLabel", 1)[1].split("\nfunction ", 1)[0])

    def test_schema_view_reads_the_shapes_the_server_actually_stores(self) -> None:
        """Parent types are {type: parent} and the predicate endpoint fields are called source_parents /
        target_parents — both as normalize_* in limits defines them. The first console version wrote
        {parent: [types]} and source/target, so loading the config threw a TypeError, which renderConfig's catch
        swallowed into a "config failed to load" message plus endless refetching, and even the corpus hints stopped
        refreshing. Both ends must be pinned to the same shape."""
        from kb_pipeline.limits import normalize_parent_types, normalize_predicates

        parents = normalize_parent_types({"pin": "component", "signal": "interface"})
        self.assertEqual(parents, {"pin": "component", "signal": "interface"})
        (pred,) = normalize_predicates([{"name": "has_pin", "source_parents": ["component"],
                                         "target_parents": ["component"], "description": "d"}])
        self.assertEqual(set(pred) >= {"name", "source_parents", "target_parents"}, True)
        js = _repo_file("app/kb_server/static/app.js")
        render = js.split("function renderGraphSchema", 1)[1].split("\nfunction ", 1)[0]
        self.assertIn("groupParentTypes(", render)
        self.assertIn("pr.source_parents", render)
        self.assertIn("pr.target_parents", render)
        self.assertNotIn("(map[name] || []).join", render)
        group = js.split("function groupParentTypes", 1)[1].split("\nfunction ", 1)[0]
        self.assertIn("(groups[String(value)] ||= []).push(key)", group)

    def test_config_panel_drops_the_four_explanations(self) -> None:
        """2026-09-06: the four explanation paragraphs under unit chunk count / gleaning rounds / relation predicates /
        auto rebuild in config management are removed; the controls themselves stay."""
        html = _repo_file("app/kb_server/static/index.html")
        js = _repo_file("app/kb_server/static/app.js")
        for gone in ('id="cfg-guc-hint"', "还有漏的吗", "只认这张表里的谓词", "类型越界", "每 30 分钟检查一次,解析忙时自动让路", "只做首建"):
            self.assertNotIn(gone, html, gone)
        for gone in ("renderUnitHint", "合成一个抽取单元", "个切片 ≈"):
            self.assertNotIn(gone, js, gone)
        for kept in ('id="cfg-guc"', 'id="cfg-ggl"', 'id="cfg-gpreds"', 'id="cfg-ro"'):
            self.assertIn(kept, html, kept)

    def test_graph_tab_is_only_the_preview(self) -> None:
        """The graph recall test and extraction preview tools were withdrawn on 2026-09-06, with their endpoints and
        service functions deleted; the graph preview canvas is taller."""
        html = _repo_file("app/kb_server/static/index.html")
        js = _repo_file("app/kb_server/static/app.js")
        api = _repo_file("app/kb_server/api.py")
        for gone in ('id="gq-q"', 'id="gq-run"', 'id="gx-load"', 'id="gx-unit"', 'id="gx-run"', "图召回测试", "抽取预览"):
            self.assertNotIn(gone, html, gone)
        for gone in ("/graph_query", "/graph_units", "/graph_extract_preview", "renderGraphTools", "renderQueryResult"):
            self.assertNotIn(gone, js, gone)
        for gone in ("graph_units", "graph_extract_preview", '"/kbs/{kb_id}/graph_query"'):
            self.assertNotIn(gone, api, gone)
        self.assertIn('@router.get("/kbs/{kb_id}/graph_preview")', api)
        self.assertIn("#gp-canvas{display:block;width:100%;height:calc(100vh - 330px);min-height:640px", html)
        # 2026-09-11 mobile: grid items can shrink (the predicate table no longer widens the whole page), the canvas
        # takes over touch gestures, the empty state no longer carries an explanation
        self.assertIn(".form-grid>*{min-width:0}", html)
        self.assertIn("touch-action:none", html)
        # On phones the action row is a two-column grid: primary / danger buttons each take a full row, and with an odd
        # number of ordinary buttons the last one takes a full row too; the predicate table hides its description column
        self.assertIn("@media (max-width:600px){", html)
        self.assertIn(".cfg-actions{display:grid;grid-template-columns:repeat(2,minmax(0,1fr))", html)
        self.assertIn(".cfg-actions > button:not(.primary):not(.danger):nth-of-type(even):nth-last-of-type(2){grid-column:1 / -1}", html)
        self.assertIn(".pred-table td:nth-child(5),.pred-table th:nth-child(5){display:none}", html)
        self.assertIn('replace("→", "→<wbr>")', _repo_file("app/kb_server/static/app.js"))
        js = _repo_file("app/kb_server/static/app.js")
        for needle in ('addEventListener("touchstart"', 'addEventListener("touchmove"', 'addEventListener("touchend"', 'mode: "pinch"'):
            self.assertIn(needle, js, needle)

    def test_progress_reads_the_real_stage_strings(self) -> None:
        """Progress is computed from the phase name + the N/M in the stage string; "Description summaries · entities"
        and "· relations" each take half the span. When the position is unknown the lower bound is used (see the
        comment on graphStagePct)."""
        js = _repo_file("app/kb_server/static/app.js")
        fn = js.split("function graphStagePct", 1)[1].split("\nfunction ", 1)[0]
        self.assertIn("stg.startsWith(n)", fn)
        self.assertIn('rest.includes("· " + p)', fn)
        self.assertIn("frac = m && +m[2] > 0", fn)
        self.assertIn('["Description summaries",                  0.50, 0.68, ["entities", "relations"]]', js)
        self.assertIn('["Structured facts",                       0.68, 0.82]', js)
        self.assertIn('["Compiling view pages",                   0.82, 0.87, ["subjects", "timelines", "sources", "narration"]]', js)
        self.assertIn('["Writing vectors",                        0.87, 0.94, ["entities", "relations", "facts", "pages"]]', js)

    def test_build_summary_carries_extract_and_summary_stats(self) -> None:
        from kb_server.service import _graph_build_summary

        manifest = {
            "input": {"documents": 3, "units": 118, "unit_chunks": 3, "avg_unit_chunks": 2.5, "avg_unit_tokens": 489,
                      "max_unit_tokens": 1536, "max_gleanings": 1},
            "extract": {"units": 118, "extracted": 118, "cached": 0, "failed_units": 0, "entities_total": 2700,
                        "llm": {"calls": 236, "cache_hits": 0, "retries": 1, "failures": 0},
                        "parse": {"records": 9000, "malformed": 30, "gleanings": 110, "unknown_types": 2, "unknown_predicates": 5}},
            "graph": {"entities": 1783, "relations": 4471, "units": 118, "mentions": 4200,
                      "merge": {"type_violations": 1300, "schema_drift_types": 2, "schema_drift_predicates": 5,
                                "negated_dropped": 20, "orphan_dropped": 60},
                      "resolution": {"candidates": 1349, "yes": 75, "merged_away": 73, "groups": 54, "batches": 24, "failed_batches": 0},
                      "summaries": {"entities": {"rows": 1783, "summarized": 511, "skipped": 1272, "failed": 0},
                                    "relations": {"rows": 4471, "summarized": 315, "skipped": 4156, "failed": 0}},
                      "llm": {"calls": 600, "cache_hits": 254}},
            "steps": ["preflight", "prepare_input", "extract", "merge"],
        }
        row = {"graph_build_id": "b1", "graph_version": "003-x", "status": "done", "stage": "完成",
               "started_at": 1, "finished_at": 2, "manifest_json": json.dumps(manifest), "error": None,
               "input_rows": 3, "active_chunk_count": 295}
        out = _graph_build_summary(row, [], {})
        self.assertEqual(out["mode"], "entity_graph")
        self.assertEqual(out["counts"]["text_units"], 118)
        self.assertEqual(out["counts"]["type_violations"], 1300)
        self.assertEqual(out["counts"]["negated_dropped"], 20)
        self.assertEqual(out["stats"]["input"]["unit_chunks"], 3)
        self.assertEqual(out["stats"]["extract"]["llm"]["calls"], 236)
        self.assertEqual(out["stats"]["extract"]["parse"]["malformed"], 30)
        self.assertEqual(out["stats"]["summaries"]["entities"]["summarized"], 511)
        self.assertEqual(out["resolution"]["merged_away"], 73)
        # A record still running with no manifest: all statistics empty, no guessing
        running = dict(row, manifest_json=None, status="running", stage="实体抽取 3/118 单元(缓存 0)")
        out2 = _graph_build_summary(running, [], {})
        self.assertIsNone(out2["counts"])
        self.assertEqual(out2["stats"], {"input": None, "extract": None, "summaries": None, "llm": None, "facts": None,
                                         "predicate_health": None, "reuse": None})


class FileChunksDrawerTests(unittest.TestCase):
    """Data source of the file table's "chunking preview" drawer: the chunks currently in the KB (state database
    ledger + Qdrant payload) and the diagnostics from indexing time, without re-chunking."""

    def test_route_and_service_read_stored_points(self) -> None:
        from unittest import mock

        from fastapi.testclient import TestClient

        from kb_pipeline.models import UnifiedChunk
        from kb_server import service
        from kb_server.main import create_app

        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"; db.init_db(state)
            with db.connect(state) as con:
                file = _local_file("a.pdf", checksum="c1")
                db.upsert_file(con, file, status="indexed")
                fid = db.file_id_for(file.kb_id, file.file_key)
                db.replace_chunks(
                    con, file_id=fid, collection=file.collection, content_version="v1",
                    chunks=[UnifiedChunk(chunk_uid="u0", chunk_index=0, text="第一段", block=_block("b0", "第一段")),
                            UnifiedChunk(chunk_uid="u1", chunk_index=1, text="| a |", block=_block("b1", "| a |"))],
                    point_ids=["p0", "p1"])
                con.execute("UPDATE files SET chunk_diag_json = ? WHERE file_id = ?",
                            (json.dumps({"ok": True, "reasons": [], "stats": {"chunks": 2, "tiny_threshold": 50}}), fid))
                con.commit()

            class FakeQ:
                def retrieve(self, collection_name, ids, with_payload=True, with_vectors=False):
                    pl = {"p0": {"text": "第一段", "token_count": 3, "block_type": "text", "block_id": "b0",
                                 "section_path": ["第一章"], "page_idx": 0},
                          "p1": {"text": "| a |", "token_count": 4, "block_type": "table", "block_id": "b1", "section_path": []}}
                    return [SimpleNamespace(id=i, payload=pl[i]) for i in ids if i in pl]

            src = SimpleNamespace(kb_id=file.kb_id, collection=file.collection, max_tokens=400, overlap_tokens=80)
            cfg = SimpleNamespace(state_db=state, qdrant_url="http://x", qdrant_api_key="", sources={"k": src})
            with mock.patch.object(service, "settings", lambda: cfg), \
                 mock.patch.object(service, "qdrant_client", lambda *a, **k: FakeQ()):
                r = service.file_chunks(file.kb_id, fid)
                self.assertEqual([c["chunk_index"] for c in r["chunks"]], [0, 1])
                self.assertEqual(r["chunks"][0]["text"], "第一段")
                self.assertEqual(r["chunks"][0]["section_path"], ["第一章"])
                self.assertEqual(r["chunks"][1]["tokens"], 4)
                self.assertEqual((r["file"]["chunks_total"], r["max_tokens"], r["overlap_tokens"]), (2, 400, 80))
                self.assertTrue(r["diagnostics"]["ok"])
                self.assertFalse(r["truncated"])
                with self.assertRaises(KeyError):
                    service.file_chunks("kb_other", fid)
            fake = mock.Mock(return_value={"chunks": []})
            with mock.patch.object(service, "file_chunks", fake):
                client = TestClient(create_app(), raise_server_exceptions=False)
                self.assertEqual(client.get(f"/api/kbs/{file.kb_id}/files/{fid}/chunks").status_code, 200)
            fake.assert_called_once_with(file.kb_id, fid)


class GraphPreviewTests(unittest.TestCase):
    """Graph preview: picks entities by degree from the current version's graph.json, or takes the neighbours around
    one entity; the console draws it on a canvas."""

    GRAPH = {
        "entities": [
            {"key": "a", "title": "ZK7C1049GN", "type": "semiconductor device", "upper": "产品", "frequency": 9, "doc_ids": ["d1", "d2"], "description": "SRAM"},
            {"key": "b", "title": "VCC", "type": "electrical parameter", "upper": "参数", "frequency": 5, "doc_ids": ["d1"]},
            {"key": "c", "title": "TSOP", "type": "package type", "upper": "产品", "frequency": 2, "doc_ids": ["d1"]},
            {"key": "d", "title": "孤立实体", "type": "other", "upper": "其它", "frequency": 1, "doc_ids": []},
            {"key": "e", "title": "目录", "type": "document", "upper": "文档", "frequency": 40, "doc_ids": ["d1"], "boilerplate": True},
        ],
        "relations": [
            {"source_key": "a", "target_key": "b", "predicate": "has_parameter", "strength_sum": 3},
            {"source_key": "a", "target_key": "c", "predicate": "has_package", "strength_sum": 2},
            {"source_key": "e", "target_key": "a", "predicate": "mentions", "strength_sum": 1},
        ],
    }

    def test_pick_by_degree_and_focus(self) -> None:
        from kb_server.service import graph_preview_pick

        out = graph_preview_pick(self.GRAPH, limit=3)
        self.assertEqual([n["key"] for n in out["nodes"]], ["a", "b", "c"])       # boilerplate entity e is not picked, the isolated d comes last
        self.assertEqual(out["nodes"][0]["degree"], 3)
        self.assertEqual(sorted(e["predicate"] for e in out["edges"]), ["has_package", "has_parameter"])
        self.assertEqual(out["totals"], {"entities": 5, "relations": 3, "boilerplate": 1})
        self.assertEqual(out["upper_counts"], {"产品": 2, "参数": 1, "其它": 1})
        focus = graph_preview_pick(self.GRAPH, limit=10, q="vcc")
        self.assertEqual(focus["focus"], "b")
        self.assertEqual([n["key"] for n in focus["nodes"]], ["b", "a"])          # centre + one-hop neighbours
        only = graph_preview_pick(self.GRAPH, limit=10, upper="参数")
        self.assertEqual([n["key"] for n in only["nodes"]], ["b"])

    def test_full_graph_and_exact_focus(self) -> None:
        """The "whole graph": limit=0 means no cap; every entity of the version (boilerplate and isolated ones
        included) and every relation is drawn, matching the numbers on the status card; boilerplate entities carry
        the boilerplate flag and the front end draws them faded. key locates exactly, unaffected by q's fuzzy match."""
        from kb_server.service import graph_preview_pick

        whole = graph_preview_pick(self.GRAPH, limit=0)
        self.assertEqual([n["key"] for n in whole["nodes"]], ["a", "e", "b", "c", "d"])
        self.assertEqual(len(whole["edges"]), 3)
        self.assertEqual([n["key"] for n in whole["nodes"] if n["boilerplate"]], ["e"])
        self.assertEqual(whole["totals"]["boilerplate"], 1)
        self.assertNotIn("e", [n["key"] for n in graph_preview_pick(self.GRAPH, limit=10)["nodes"]])   # with a cap, boilerplate is still excluded
        exact = graph_preview_pick(self.GRAPH, limit=0, q="随便写的", key="c")
        self.assertEqual(exact["focus"], "c")
        self.assertEqual([n["key"] for n in exact["nodes"]], ["c", "a"])
        self.assertIsNone(graph_preview_pick(self.GRAPH, limit=0, key="no-such")["focus"])

    def test_scoped_entities_carry_their_document_label(self) -> None:
        """Document-local entities (one per document for part / property / process) carry a short label of the
        document they come from: the date, else the version, else the filename. Three check-up reports each have a
        "urinalysis" entity, drawn on the canvas as "urinalysis · 2021-03-14", with a hover note saying which
        document it comes from. Global entities have scope / doc both None."""
        from kb_server.service import graph_preview_pick

        graph = {
            "documents": {"kb:1": {"rel_path": "李/2021体检报告.pdf", "date": "2021-03-14", "version": ""},
                          "kb:2": {"rel_path": "手册/rev-b.pdf", "date": "", "version": "B"},
                          "kb:3": {"rel_path": "杂/说明.pdf", "date": "", "version": ""}},
            "entities": [
                {"key": "kb:1::尿常规", "title": "尿常规", "type": "laboratory test", "upper": "process", "scope": "kb:1", "frequency": 1, "doc_ids": ["kb:1"]},
                {"key": "kb:2::尿常规", "title": "尿常规", "type": "laboratory test", "upper": "process", "scope": "kb:2", "frequency": 1, "doc_ids": ["kb:2"]},
                {"key": "kb:3::尿常规", "title": "尿常规", "type": "laboratory test", "upper": "process", "scope": "kb:3", "frequency": 1, "doc_ids": ["kb:3"]},
                {"key": "kb:9::x", "title": "x", "type": "t", "upper": "process", "scope": "kb:9", "frequency": 1, "doc_ids": ["kb:9"]},
                {"key": "李华", "title": "李华", "type": "person", "upper": "entity", "frequency": 3, "doc_ids": ["kb:1", "kb:2", "kb:3"]},
            ],
            "relations": [{"source_key": "李华", "target_key": k, "predicate": "has_test", "strength_sum": 1} for k in ("kb:1::尿常规", "kb:2::尿常规", "kb:3::尿常规", "kb:9::x")],
        }
        out = graph_preview_pick(graph, limit=0)
        by = {n["key"]: n for n in out["nodes"]}
        self.assertEqual((by["kb:1::尿常规"]["scope"], by["kb:1::尿常规"]["doc"]), ("kb:1", "2021-03-14"))
        self.assertEqual(by["kb:2::尿常规"]["doc"], "B")                    # no date, so the version
        self.assertEqual(by["kb:3::尿常规"]["doc"], "说明")                  # neither, so the filename
        self.assertEqual(by["kb:9::x"]["doc"], "9")                          # document metadata missing: falls back to the tail of the document id
        self.assertEqual((by["李华"]["scope"], by["李华"]["doc"]), (None, None))
        js = _repo_file("app/kb_server/static/app.js")
        for needle in ("gpLabelOf", "dupTitles", "文档内实体 · 来自", "另有", "UPPER_ZH", "marqueeMergeCells", "seg-clone"):
            self.assertIn(needle, js, needle)
        html = _repo_file("app/kb_server/static/index.html")
        # 2026-09-10 the user asked to remove the line of usage instructions under the canvas (the interaction is
        # self-explanatory through hover tips and button titles), together with its dictionary entry
        self.assertNotIn('gp-hint', html)
        self.assertNotIn('点一个实体以它为中心看邻居', _repo_file("app/kb_server/static/i18n.js"))
        self.assertIn(".merge-tbl td .cell.scroll .cell-in{animation:llm-marquee", html)

    def test_service_reads_the_current_version_and_console_draws_it(self) -> None:
        from unittest import mock

        from fastapi.testclient import TestClient

        from kb_server import service
        from kb_server.main import create_app

        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"; db.init_db(state)
            src = SimpleNamespace(kb_id="kb_1", collection="kb_1")
            cfg = SimpleNamespace(state_db=state, sources={"k1": src})
            with db.connect(state) as con:
                con.execute("INSERT INTO graph_builds(graph_build_id, source_key, kb_id, source_collection, graph_version, status, "
                            "started_at, finished_at, build_kind) VALUES('g1', 'k1', 'kb_1', 'kb_1', 'v1', 'done', 10, 20, 'append')")
                con.commit()
            gdir = Path(tmp) / "work"; gdir.mkdir()
            (gdir / "graph.json").write_text(json.dumps(self.GRAPH), encoding="utf-8")
            paths = SimpleNamespace(graph_file=gdir / "graph.json")
            with mock.patch.object(service, "settings", lambda: cfg), \
                 mock.patch("kb_pipeline.graph.build.graph_paths", lambda *a, **k: paths):
                out = service.graph_preview("kb_1", limit=5)
                self.assertEqual((out["version"], out["build_kind"]), ("v1", "append"))
                self.assertEqual(len(out["nodes"]), 4)
                self.assertEqual(len(service.graph_preview("kb_1", limit=0)["nodes"]), 5)   # whole graph, boilerplate included
                with self.assertRaises(KeyError):
                    service.graph_preview("kb_x")
            fake = mock.Mock(return_value={"nodes": []})
            with mock.patch.object(service, "graph_preview", fake):
                client = TestClient(create_app(), raise_server_exceptions=False)
                self.assertEqual(client.get("/api/kbs/kb_1/graph_preview?limit=40&q=vcc").status_code, 200)
            fake.assert_called_once_with("kb_1", limit=40, q="vcc", upper="", key="")
        html = _repo_file("app/kb_server/static/index.html")
        js = _repo_file("app/kb_server/static/app.js")
        for needle in ('id="gp-canvas"', 'id="gp-q"', 'id="gp-legend"', "<h2>图谱预览</h2>"):
            self.assertIn(needle, html, needle)
        for needle in ("function layoutGraph(", "function renderGraphPreview(", "/graph_preview?", "refreshGraphPreview(force)"):
            self.assertIn(needle, js, needle)

    def test_console_levels_pan_zoom_and_whole_graph(self) -> None:
        """2026-09-06: ← → in the canvas's top-left corner jump between levels, greyed out when there is no previous /
        next level; the wheel zooms, dragging pans; the explanatory line above the canvas is removed.
        2026-09-12, decided by the user: the "whole graph" button is removed, the entity-count steps run 100 ~ 3000
        (1000 / 1500 / 2000 / 3000 added above 500) and the server cap is raised to 3000 to match; changing the
        count refetches in place on the same level, and the view no longer has an all state. The endpoint's limit=0
        still returns the whole version (test_full_graph_and_exact_focus)."""
        html = _repo_file("app/kb_server/static/index.html")
        js = _repo_file("app/kb_server/static/app.js")
        svc = _repo_file("app/kb_server/service.py")
        for needle in ('id="gp-back" class="ghost" title="上一层" disabled', 'id="gp-fwd" class="ghost" title="下一层" disabled',
                       '<option value="100">', '<option value="200" selected>', '<option value="500">', '<option value="1000">',
                       '<option value="1500">', '<option value="2000">', '<option value="3000">3000 个</option></select>',
                       'title="画多少个实体"', ".gp-nav{position:absolute;top:10px;left:10px"):
            self.assertIn(needle, html, needle)
        for gone in ('id="gp-info"', 'id="gp-reset"', '<option value="40">', '<option value="80"', 'id="gp-all"', "全图不受这个限制", ">全图<"):
            self.assertNotIn(gone, html, gone)
        for needle in ("function gpGo(", "function gpStep(", "function renderGraphNav(", '$("#gp-back").disabled = g.cur <= 0',
                       '$("#gp-fwd").disabled = g.cur >= g.hist.length - 1', 'limit: $("#gp-limit").value',
                       '$("#gp-limit").addEventListener("change", () => gpLoadView());',
                       'gpCanvas.addEventListener("wheel"', 'gpCanvas.addEventListener("mousedown"', "state.gp.cam.tx += dx",
                       "function gpQuadTree(", "gpGo({ q: n.title, key: n.key })"):
            self.assertIn(needle, js, needle)
        for gone in ("gp-info", "悬停看说明,点击以它为中心", "画出连接最多的", "gp-all", "view.all", "all: false", "all: cur.all"):
            self.assertNotIn(gone, js, gone)
        self.assertIn("limit = 0 if limit <= 0 else max(10, min(limit, 3000))", svc)


class GraphFileLoadTests(unittest.TestCase):
    """The preview and the merge drawer read graph.json: for a large knowledge base the file is several hundred MB.
    Read whole and then parsed, it took several GB in the web process for a moment and other endpoints did not
    respond while it parsed; the cache kept a single copy and dropped the old one only after reading the new one, so
    switching back and forth between two large knowledge bases re-read the file every time."""

    GRAPH = {
        "kb_id": "kb_1", "graph_version": "v1", "built_at": 1726000000, "schema": {"entity_types": ["device"]},
        "documents": {"kb_1:1": {"rel_path": "手册/a.pdf", "date": "2026-01-02", "version": "B", "doc_type": "manual", "pages": 12}},
        "unit_kinds": {"u1": "body", "u2": "listing"},
        "entities": [
            {"key": "a", "title": "存储器 𠮷", "type": "device", "upper": "entity", "frequency": 9, "doc_ids": ["kb_1:1"],
             "description": "长" * 500, "descriptions": ["长" * 500, "别的说法"], "unit_ids": ["u1", "u2"], "aliases": ["SRAM"],
             "pagerank": 0.61, "degree": 2, "types": {"device": 3}},
            {"key": "kb_1:1::b", "title": "电压", "type": "parameter", "upper": "property", "frequency": 5, "doc_ids": ["kb_1:1"],
             "scope": "kb_1:1", "description": "供电 \"VCC\"\\n3.3 V", "unit_ids": ["u1"]},
            {"key": "c", "title": "目录", "type": "document", "upper": "document", "frequency": 40, "boilerplate": True},
        ],
        "relations": [
            {"source_key": "a", "target_key": "kb_1:1::b", "source": "存储器", "target": "电压", "predicate": "has_parameter",
             "strength_sum": 3, "description": "述" * 400, "descriptions": ["述" * 400], "weight": 7.5e-1, "unit_ids": ["u1"]},
            {"source_key": "c", "target_key": "a", "predicate": "mentions", "strength_sum": 1, "npmi": -0.25},
        ],
        "mentions": [{"entity_key": "a", "point_id": "p1", "chunk_uid": "cu1", "count": 12345}] * 40,
        "resolution_map": {"a2": "a"}, "resolution_judged": [["a", "c", "no"], ["a", "kb_1:1::b", "no"]],
        "resolution_log": [{"kept": "a", "kept_title": "存储器", "merged": "a2", "merged_title": "SRAM 存储器", "source": "lexical", "category": "alias"}],
        "resolution_rejected": [{"a_title": "甲", "b_title": "乙", "source": "embedding", "reason": "recheck", "verdict": "narrower"}],
        "stats": {"units": 2, "resolution": {"candidates": 3, "yes": 1, "entities_before": 4, "entities_after": 3}, "llm": {"calls": 9}},
        "pages": [{"page_id": "p", "markdown": "# 页\\n" + "文" * 300}], "specs": [], "conflicts": [],
    }

    def setUp(self) -> None:
        from kb_server import service

        service._GRAPH_JSON_CACHE.clear()
        self.addCleanup(service._GRAPH_JSON_CACHE.clear)

    def test_the_file_is_read_piece_by_piece_and_only_what_the_views_use_is_kept(self) -> None:
        from unittest import mock

        from kb_server import service

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "graph.json"
            for layout in ({}, {"indent": 1}, {"separators": (",", ":"), "ensure_ascii": True}):
                path.write_text(json.dumps(self.GRAPH, **{"ensure_ascii": False, **layout}), encoding="utf-8")
                whole = json.loads(path.read_text(encoding="utf-8"))
                for chars in (1, 3, 50, 1 << 20):           # wherever the buffer breaks, the result is the same
                    with mock.patch.object(service, "_GRAPH_READ_CHARS", chars):
                        pieces = list(service._iter_graph_json(path))
                        slim = service._read_graph_json(path)
                    self.assertEqual([p for p in pieces if p[0] == "mentions"], [("mentions", whole["mentions"][0], True)] * 40)
                    self.assertIn(("graph_version", "v1", False), pieces)
                    self.assertIn(("resolution_judged", ["a", "c", "no"], True), pieces)
                    self.assertEqual(sorted(slim), ["documents", "entities", "relations", "resolution_log", "resolution_rejected", "stats"])
                    self.assertEqual(sorted(slim["entities"][0]),
                                     ["aliases", "description", "doc_ids", "frequency", "key", "title", "type", "upper"])
                    self.assertEqual((len(slim["entities"][0]["description"]), len(slim["relations"][0]["description"])), (240, 160))
                    self.assertEqual(slim["documents"], {"kb_1:1": {"rel_path": "手册/a.pdf", "date": "2026-01-02", "version": "B"}})
                    self.assertEqual(slim["stats"], {"resolution": whole["stats"]["resolution"]})
                    self.assertEqual((slim["resolution_log"], slim["resolution_rejected"]),
                                     (whole["resolution_log"], whole["resolution_rejected"]))
                    for view in ({"limit": 10}, {"limit": 0}, {"limit": 10, "q": "sram"}, {"limit": 10, "key": "kb_1:1::b"},
                                 {"limit": 10, "upper": "property"}):
                        self.assertEqual(service.graph_preview_pick(slim, **view), service.graph_preview_pick(whole, **view), view)
            for broken in ('{"entities": [{"key": "a"}', "[1, 2]", '{"entities": [1 2]}', ""):      # an incomplete read raises instead of returning half a graph
                path.write_text(broken, encoding="utf-8")
                with self.assertRaises(ValueError):
                    service._read_graph_json(path)

    def test_memory_held_while_reading_does_not_grow_with_the_file(self) -> None:
        import io
        from unittest import mock

        from kb_server import service

        class Tracked(io.StringIO):
            asked: list[int] = []

            def read(self, size=-1):
                self.asked.append(size)
                return super().read(size)

        graph = {"kb_id": "kb_1", "documents": {f"d{i}": {"rel_path": f"{i}.pdf"} for i in range(5)},
                 "entities": [{"key": f"e{i}", "title": "名" * 20, "unit_ids": ["u1", "u2"]} for i in range(400)],
                 "mentions": [{"entity_key": f"e{i}", "point_id": "p", "count": 1.5e3} for i in range(400)],
                 "stats": {"resolution": {"yes": 1}}}
        text = json.dumps(graph, ensure_ascii=False)
        stream = Tracked(text)
        with mock.patch.object(service, "_GRAPH_READ_CHARS", 64):
            pieces = list(service._iter_graph_json(SimpleNamespace(open=lambda encoding: stream)))
        self.assertEqual(len(pieces), 1 + 1 + 400 + 400 + 1)
        self.assertGreater(len(text), 64 * 200)
        # Each read asks for a small piece; only a single value larger than the buffer asks for more (the documents
        # item is about 150 characters)
        self.assertLessEqual(max(stream.asked), 64 * 4)

    def test_graphs_are_cached_per_kb_within_a_budget_and_the_old_one_goes_first(self) -> None:
        import os
        from unittest import mock

        from kb_server import service

        with tempfile.TemporaryDirectory() as tmp:
            paths = {}
            for name in ("a1", "a2", "b", "c"):
                paths[name] = Path(tmp) / name / "graph.json"
                paths[name].parent.mkdir()
                paths[name].write_text(json.dumps({"entities": [{"key": name}], "pad": "x" * 1000}), encoding="utf-8")
            size = paths["a1"].stat().st_size
            held_while_reading: list[list[str]] = []
            real = service._read_graph_json

            def reader(path):
                held_while_reading.append(sorted(service._GRAPH_JSON_CACHE))
                return real(path)

            with mock.patch.object(service, "_read_graph_json", side_effect=reader) as read, \
                    mock.patch.object(service, "_GRAPH_CACHE_MAX_BYTES", size * 2 + 10):
                first = service._load_graph_json("kb_a", paths["a1"])
                self.assertIs(service._load_graph_json("kb_a", paths["a1"]), first)            # same version: not read again
                service._load_graph_json("kb_b", paths["b"])
                self.assertIs(service._load_graph_json("kb_a", paths["a1"]), first)            # another KB and back: still there
                self.assertEqual(read.call_count, 2)
                # A new version: the old copy is let go before the new one is read
                second = service._load_graph_json("kb_a", paths["a2"])
                self.assertEqual(second["entities"], [{"key": "a2", "description": ""}])
                self.assertEqual(held_while_reading[-1], ["kb_b"])
                # The same path rewritten (mtime changed) is read again too
                os.utime(paths["a2"], (1, 1))
                service._load_graph_json("kb_a", paths["a2"])
                self.assertEqual(read.call_count, 4)
                # Over the budget: the least recently viewed (kb_b) goes first, kb_a viewed just now stays
                service._load_graph_json("kb_c", paths["c"])
                self.assertEqual(held_while_reading[-1], ["kb_a"])
                self.assertEqual(sorted(service._GRAPH_JSON_CACHE), ["kb_a", "kb_c"])
                # A single graph over the budget on its own is still read; everything else makes room
                with mock.patch.object(service, "_GRAPH_CACHE_MAX_BYTES", 10):
                    service._load_graph_json("kb_b", paths["b"])
                self.assertEqual(sorted(service._GRAPH_JSON_CACHE), ["kb_b"])
            # Deleting the graph or the KB lets go of that KB's cached copy
            for call, target in ((service.delete_graph, "kb_pipeline.maintenance.delete_graph_now"),
                                 (service.delete_kb, "kb_pipeline.maintenance.delete_kb_now")):
                service._GRAPH_JSON_CACHE["kb_b"] = ("p", 0.0, 1, {})
                with mock.patch.object(service, "settings"), mock.patch(target, return_value={"errors": []}):
                    call("kb_b")
                self.assertNotIn("kb_b", service._GRAPH_JSON_CACHE)

    def test_identical_requests_arriving_together_read_the_file_once(self) -> None:
        import threading
        from unittest import mock

        from kb_server import service

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "graph.json"
            path.write_text(json.dumps(self.GRAPH, ensure_ascii=False), encoding="utf-8")
            real = service._read_graph_json
            started = threading.Event()

            def slow(p):
                started.set()
                time.sleep(0.2)
                return real(p)

            got: list[Any] = []
            with mock.patch.object(service, "_read_graph_json", side_effect=slow) as read:
                first = threading.Thread(target=lambda: got.append(service._load_graph_json("kb_1", path)))
                first.start()
                self.assertTrue(started.wait(5))
                others = [threading.Thread(target=lambda: got.append(service._load_graph_json("kb_1", path))) for _ in range(3)]
                for t in others:
                    t.start()
                for t in [first, *others]:
                    t.join(10)
            self.assertEqual(read.call_count, 1)
            self.assertEqual(len(got), 4)
            self.assertTrue(all(g is got[0] for g in got))


class ServiceDrawerBulkTests(unittest.TestCase):
    """2026-09-06: service status drawer — the "system services" group gains one-click restart all / stop all;
    timer tasks show only their last run time."""

    def test_bulk_actions_drive_every_container(self) -> None:
        from unittest import mock

        from kb_server import service

        calls: list[list[str]] = []

        def fake_popen(cmd, **kw):
            calls.append(list(cmd))
            return mock.Mock()

        with mock.patch.object(service.subprocess, "Popen", side_effect=fake_popen), \
                mock.patch.object(service, "settings", return_value=SimpleNamespace(log_dir=_scratch_dir(self))):
            out = service.stop_all_services(force=True)
            self.assertEqual(out["stopping"], service._all_service_containers())
            self.assertTrue(all(c[:2] == ["docker", "stop"] for c in calls))
            self.assertEqual({c[2] for c in calls}, set(service._all_service_containers()))
            calls.clear()
            out = service.restart_all_services(force=True)
            self.assertEqual(out["restarting"], service._all_service_containers())
            self.assertTrue(all(c[:2] == ["docker", "restart"] for c in calls))
        # Without force while the system is busy: refused, with the reasons returned so the front end can ask for confirmation
        with mock.patch("kb_pipeline.maintenance.service_busy", return_value=(True, ["解析任务 3 个"])), \
                mock.patch.object(service, "settings", return_value=SimpleNamespace(log_dir=_scratch_dir(self))):
            with self.assertRaisesRegex(ValueError, "a shutdown would interrupt"):
                service.stop_all_services(force=False)
            with self.assertRaisesRegex(ValueError, "a restart would interrupt"):
                service.restart_all_services(force=False)

    def test_routes_and_console(self) -> None:
        from unittest import mock

        from fastapi.testclient import TestClient

        from kb_server import service
        from kb_server.main import create_app

        with mock.patch.object(service, "stop_all_services", return_value={"stopping": ["a"]}) as stop, \
                mock.patch.object(service, "restart_all_services", return_value={"restarting": ["a"]}) as restart:
            client = TestClient(create_app(), raise_server_exceptions=False)
            self.assertEqual(client.post("/api/services/stop_all", json={"force": True}).status_code, 200)
            stop.assert_called_once_with(True)
            self.assertEqual(client.post("/api/services/restart_all", json={}).status_code, 200)
            restart.assert_called_once_with(False)
        js = _repo_file("app/kb_server/static/app.js")
        health = js.split("async function _refreshHealth", 1)[1].split("\nasync function ", 1)[0]
        self.assertIn("系统服务", health)
        self.assertIn('id="svc-restart-all"', health)
        self.assertIn('id="svc-stop-all"', health)
        self.assertNotIn("下次 ${t.next}", health)                 # timer tasks show only the last run
        self.assertNotIn("运行中", health)                          # nor the in-progress state; the dot only reflects the last result
        self.assertNotIn("yellow pulse", health)
        self.assertNotIn("t.running", health)
        self.assertNotIn("连续让路", health)                        # the yield count is not shown either; over the limit the unit itself turns red
        self.assertIn("function onBulkService", js)
        self.assertIn("/services/stop_all", js)
        self.assertIn("/services/restart_all", js)

    def test_sidebar_has_a_kb_filter_box(self) -> None:
        """The sidebar gains a filter box: filters by directory name / id substring, while the counts still cover all KBs."""
        html = _repo_file("app/kb_server/static/index.html")
        self.assertIn('id="kb-q"', html)
        js = _repo_file("app/kb_server/static/app.js")
        nav = js.split("function renderKbNav()", 1)[1].split("\nfunction ", 1)[0]
        self.assertIn('$("#kb-q")', nav)
        self.assertIn("k.dir.toLowerCase().includes(q)", nav)
        self.assertIn('$("#kb-q")?.addEventListener("input", renderKbNav)', js)
        # 2026-09-07: an idle sidebar row shows no status text (not enabled / graph built / graph build failed) and
        # only draws progress while a task is running; a removed KB (with a deletion countdown) still gets a line
        self.assertNotIn('"已建图"', nav)
        self.assertNotIn("建图失败", nav)
        self.assertNotIn("失败 ${kb.files_failed}", nav)
        self.assertIn('kb.state === "inactive" || kb.state === "directory_missing"', nav)
        self.assertNotIn("} else if (!active) {", nav)

    def test_neo4j_is_part_of_the_database_service(self) -> None:
        """The graph database is managed under "database services": health probing, individual restart and stop all /
        restart all cover Neo4j; no separate row."""
        from unittest import mock

        from kb_server import service

        self.assertEqual(service.SERVICE_CONTAINERS["database"], ["carrel-qdrant", "carrel-opensearch", "carrel-neo4j"])
        self.assertNotIn("neo4j", service.SERVICE_CONTAINERS)
        with mock.patch.object(service, "settings", return_value=SimpleNamespace(console_services=("database", "mineru"))):
            self.assertIn("carrel-neo4j", service._all_service_containers())
        self.assertFalse(service._probe_neo4j("bolt://127.0.0.1:1", "neo4j", "x"))    # unreachable = unhealthy, without raising
        self.assertFalse(service._probe_neo4j("", "neo4j", None))
        js = _repo_file("app/kb_server/static/app.js")
        self.assertIn("h => h.qdrant && h.opensearch && h.neo4j !== false", js)
        self.assertNotIn("图谱库服务", js)
        health = _repo_file("app/kb_server/service.py").split("def health()", 1)[1].split("\ndef ", 1)[0]
        self.assertIn('"neo4j": lambda: _probe_neo4j(', health)

    def test_reranker_row_covers_both_rerankers(self) -> None:
        """2026-09-06: the "reranker service" row manages the text and the cross-modal reranker containers. The
        pipeline does not use them: an individual restart skips the busy check; stop all / restart all cover them;
        each is probed on its own, and with both addresses empty the value is None and the row is hidden."""
        from unittest import mock

        from kb_server import service

        self.assertEqual(service.SERVICE_CONTAINERS["reranker"], ["carrel-reranker", "carrel-vl-reranker"])
        all_rows = tuple(service.SERVICE_CONTAINERS)
        with mock.patch.object(service, "settings", return_value=SimpleNamespace(console_services=all_rows)):
            for name in service.SERVICE_CONTAINERS["reranker"]:
                self.assertIn(name, service._all_service_containers())
        calls: list[list[str]] = []

        def fake_popen(cmd, **kw):
            calls.append(list(cmd))
            return mock.Mock()

        with mock.patch.object(service.subprocess, "Popen", side_effect=fake_popen), \
                mock.patch("kb_pipeline.maintenance.service_busy", return_value=(True, ["解析任务 3 个"])), \
                mock.patch.object(service, "settings", return_value=SimpleNamespace(log_dir=_scratch_dir(self), console_services=all_rows)):
            out = service.restart_service("reranker", force=False)          # restarts even while the system is busy
            self.assertEqual(out["restarting"], service.SERVICE_CONTAINERS["reranker"])
            self.assertEqual([c[2] for c in calls], service.SERVICE_CONTAINERS["reranker"])
            with self.assertRaisesRegex(ValueError, "a restart would interrupt"):        # services the pipeline depends on are still blocked
                service.restart_service("vlm", force=False)

        def cfg(**over):
            base = dict(qdrant_url="http://q", opensearch_url="http://o", neo4j_uri="", mineru_url="http://m",
                        embedding_base_url="http://e/v1", vlm_base_url="http://v/v1", visual_embedding_enabled=False,
                        visual_embedding_base_url="", parse_enabled=True, state_db=_scratch_dir(self) / "s.db",
                        reranker_base_url="http://127.0.0.1:8102/v1", visual_reranker_base_url="http://127.0.0.1:8104/v1",
                        console_services=all_rows, runtime_dir=_scratch_dir(self))
            base.update(over)
            return SimpleNamespace(**base)

        seen: list[str] = []

        def fake_get(url, timeout=0):
            seen.append(url)
            return SimpleNamespace(status_code=200 if ":8102/" in url else 500)

        with mock.patch.object(service.requests, "get", side_effect=fake_get), \
                mock.patch.object(service, "_probe_neo4j", return_value=False), \
                mock.patch.object(service, "_systemctl_show", return_value={}), \
                mock.patch.object(service, "_maintenance_defers", return_value={}), \
                mock.patch.object(service, "timer_health", return_value=[]):
            with mock.patch.object(service, "settings", return_value=cfg()):
                service._health_cache["value"] = None
                h = service.health()
                self.assertTrue(h["reranker"])
                self.assertFalse(h["visual_reranker"])
                self.assertIn("http://127.0.0.1:8102/health", seen)          # probes vLLM's /health with /v1 stripped
                self.assertIn("http://127.0.0.1:8104/health", seen)
            with mock.patch.object(service, "settings", return_value=cfg(reranker_base_url="", visual_reranker_base_url="")):
                service._health_cache["value"] = None
                h = service.health()
                self.assertIsNone(h["reranker"])
                self.assertIsNone(h["visual_reranker"])
        service._health_cache["value"] = None
        js = _repo_file("app/kb_server/static/app.js")
        self.assertIn('["重排序服务",     "reranker",', js)
        self.assertIn("h.reranker == null && h.visual_reranker == null ? null", js)
        self.assertIn("h.reranker !== false && h.visual_reranker !== false", js)
        example = _repo_file("config/knowledge-base.env.example")
        self.assertIn("RERANKER_BASE_URL=", example)
        self.assertIn("VISUAL_RERANKER_BASE_URL=", example)   # the code defaults are checked through load_settings in ConfigLoadingTests


class GraphPreviewClusterTests(unittest.TestCase):
    """2026-09-07: the graph preview is laid out in partitions by upper ontology (the 3D plan was rejected; this is
    the replacement). Anchors form a ring with wider sectors for bigger classes, nodes start near their own anchor
    and are pulled towards it; "class name · count" is written above each cluster; the switch is remembered in the
    browser, on by default."""

    def test_cluster_layout_is_wired(self) -> None:
        js = _repo_file("app/kb_server/static/app.js")
        for needle in ("function gpGroupAnchors(", "function gpDrawGroupLabels(", "function gpClusterOn(",
                       "layoutGraph(d.nodes, d.edges, { cluster: gpClusterOn(), palette })",
                       'localStorage.getItem("gp.cluster") !== "0"', "gpDrawGroupLabels(ctx, sim, pos, state.gp.cam, palette, cssW, cssH)",
                       "(g.x - p.x) * gc", "cross ? 0.02 : 0.06", "single: order.length < 2", '$("#gp-cluster")?.addEventListener("click"', "state.gp.sim = null; state.gp.pos = null;"):
            self.assertIn(needle, js, needle)
        layout = js.split("function layoutGraph(", 1)[1].split("\nfunction ", 1)[0]
        self.assertIn("const anchorOf = groups && !groups.single", layout)          # a single class is not partitioned
        self.assertIn("const g = 0.015 + Math.min(deg[i], 20) * 0.002;", layout)   # switched off, it is still the original centripetal pull
        html = _repo_file("app/kb_server/static/index.html")
        self.assertIn('id="gp-cluster" class="ghost on"', html)
        self.assertIn("#gp-cluster.on{", html)
        self.assertRegex(html, r"app\.js\?v=\d{8}-\d+")


class PollingResumeTests(unittest.TestCase):
    def test_returning_to_the_tab_restarts_the_poll_chain(self) -> None:
        """2026-09-08: when the page returns from hidden to visible it must restart the polling chain, not just refetch
        once and then wait for the old 60-second timer."""
        js = _repo_file("app/kb_server/static/app.js")
        handler = js.split('document.addEventListener("visibilitychange"', 1)[1].split("\n});", 1)[0]
        self.assertIn("tick();", handler)
        self.assertNotIn("refreshOverview();", handler)
        tick = js.split("async function tick()", 1)[1].split("\n}\n", 1)[0]
        self.assertIn("if (tickBusy) return;", tick)
        self.assertIn("clearTimeout(tickTimer);", tick)
        self.assertIn("tickTimer = setTimeout(tick, document.hidden ? 60000 : (busy ? 2500 : 8000));", tick)


class ParseProgressEstimateTests(unittest.TestCase):
    """2026-09-08: the parse progress bar assigns spans from the actual per-stage durations of this KB's recent
    parses and interpolates within a stage by elapsed time; a freshly enabled KB shows the on-disk count before the
    scan registers files; a scan that queued jobs kicks the worker instead of waiting for the 5-minute timer."""

    def test_stage_weights_are_medians_of_recent_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"
            db.init_db(state)
            with db.connect(state) as con:
                file = _local_file("a.pdf", checksum="c1")
                db.upsert_file(con, file, status="indexed")
                fid = db.file_id_for(file.kb_id, file.file_key)

                def job(started, stages, finished):
                    jid = db.enqueue_job(con, ingest_run_id=None, file_id=fid, kb_id=file.kb_id, collection=file.collection,
                                         file_key=file.file_key, job_type="parse", payload={}, dedupe_key=f"j:{started}")
                    con.execute("UPDATE jobs SET status='done', started_at=?, finished_at=? WHERE job_id=?", (started, finished, jid))
                    for ts, text in stages:
                        con.execute("INSERT INTO job_events(job_id, ts, kind, text) VALUES (?, ?, 'stage', ?)", (jid, ts, text))
                    return jid

                job(1000, [(1000, "文档解析"), (1100, "图片描述(VLM 1/9)"), (1140, "切块"), (1142, "写入向量库")], 1145)
                job(2000, [(2000, "文档解析"), (2020, "图片描述(VLM 1/3)"), (2030, "切块"), (2031, "写入向量库")], 2033)
                job(3000, [(3000, "文档解析"), (3060, "切块"), (3061, "写入向量库")], 3062)      # a file without images
                from kb_server.service import parse_stage_weights, stage_elapsed

                w = parse_stage_weights(con, file.kb_id)
                self.assertEqual(w["文档解析"], 60.0)      # median of 100 / 20 / 60
                self.assertEqual(w["图片描述"], 25.0)      # 40 / 10
                self.assertEqual(w["切块"], 1.0)           # 2 / 1 / 1
                self.assertEqual(w["写入向量库"], 2.0)     # the last stage counts up to finished_at: 3 / 2 / 1
                self.assertNotIn("完成", w)
                running = db.enqueue_job(con, ingest_run_id=None, file_id=fid, kb_id=file.kb_id, collection=file.collection,
                                         file_key=file.file_key, job_type="parse", payload={}, dedupe_key="j:run")
                con.execute("INSERT INTO job_events(job_id, ts, kind, text) VALUES (?, 5000, 'stage', '文档解析')", (running,))
                self.assertEqual(stage_elapsed(con, running, 5030), 30)
                self.assertIsNone(stage_elapsed(con, "nope", 5030))

    def test_scan_script_kicks_the_worker_when_it_queued_jobs(self) -> None:
        """Kicking the worker lives in the scan shell script (reading jobs=N from the summary line); the pipeline code
        is untouched."""
        script = _repo_file("scripts/kb-pipeline-scan.sh")
        self.assertIn("scan_code=${PIPESTATUS[0]}", script)
        self.assertIn("grep -oE 'jobs=[0-9]+'", script)
        # The script reads the summary line the cli prints, so both formats must be pinned: change the summary line
        # and the worker kick silently stops working
        cli_src = _repo_file("app/kb_pipeline/cli.py")
        self.assertIn('"scan summary: "', cli_src)
        self.assertIn("jobs={stats.jobs}", cli_src)
        self.assertIn("systemctl --user start --no-block carrel-worker.service", script)
        self.assertIn('"${KB_SCAN_KICK_WORKER:-1}"', script)
        # Run the same parsing logic over a fake summary to confirm jobs=N takes the number from the summary line
        import subprocess as sp
        out = sp.run(["bash", "-c", "grep -oE 'scan summary: .*jobs=[0-9]+' <<< \"$1\" | grep -oE 'jobs=[0-9]+' | tail -1 | cut -d= -f2",
                      "_", "  new  a.pdf\nscan summary: seen=3 added=1 jobs=4 recent_pending=0 dry_run=False\n"],
                     capture_output=True, text=True)
        self.assertEqual(out.stdout.strip(), "4")

    def test_console_estimates_progress_from_history_and_shows_unregistered_files(self) -> None:
        svc = _repo_file("app/kb_server/service.py")
        self.assertIn('entry["parse_stage_weights"] = _parse_stage_weights_cached(con, entry["kb_id"])', svc)
        self.assertIn('item["stage_elapsed"] = stage_elapsed(con, str(item["job_id"]), now_ts)', svc)
        self.assertIn("def parse_stage_weights(con, kb_id: str", svc)      # the query lives in the console layer, the pipeline code is untouched
        self.assertIn('if stats["files_total"] == 0:', svc)
        js = _repo_file("app/kb_server/static/app.js")
        self.assertIn("function parseStageSpans(weights)", js)
        self.assertIn("stageFraction(j.stage, j, kb.parse_stage_weights)", js)
        self.assertIn("Math.min(0.95, elapsed / secs)", js)
        self.assertIn("等待扫描登记", js)
        self.assertIn("kb.files_total || kb.dir_files || 0", js)


class ConsoleI18nTests(unittest.TestCase):
    """Console Chinese / English switch (2026-09-09): strings are keyed by the Chinese original, static pages are
    translated wholesale on load, strings assembled in app.js go through t(), and stage names / timer tasks /
    diagnostic reasons from the server are translated by whole sentence or by pattern; the language is stored in
    localStorage kb.lang and switching reloads the page. Data such as directory names, filenames and entity names
    is not translated."""

    _ZH = re.compile(r"[一-鿿]")

    def _dict_keys(self) -> set[str]:
        i18n = _repo_file("app/kb_server/static/i18n.js")
        unesc = lambda x: x.replace("\\n", "\n").replace('\\"', '"')
        return {unesc(m.group(1)) for m in re.finditer(r'^\s*"((?:[^"\\]|\\.)*)": "', i18n, re.M)}

    def test_i18n_module_loads_before_app_and_the_switch_sits_by_the_title(self) -> None:
        html = _repo_file("app/kb_server/static/index.html")
        self.assertLess(html.index('<script src="i18n.js'), html.index('<script src="app.js'))
        brand = html.split('<div class="brand">', 1)[1].split("</div></div>", 1)[0]
        # Both language titles are stacked in the same cell and the invisible English one sets the width: the switch
        # does not move with the language; on narrow screens the title shrinks instead of wrapping
        self.assertIn('<h1><span>Carrel 控制台</span><span class="ghost" aria-hidden="true">Carrel Console</span></h1>', brand)
        css = html.split("<style>", 1)[1].split("</style>", 1)[0]
        self.assertIn(".brand h1>span{grid-area:1/1;overflow:hidden;text-overflow:ellipsis}", css)
        self.assertIn(".brand h1 .ghost{visibility:hidden}", css)
        self.assertIn("#btn-llms{min-width:80px}#btn-health{min-width:96px}", css)
        self.assertIn("white-space:nowrap}", css.split(".tabs button{", 1)[1].split("\n", 1)[0])
        self.assertIn('id="lang-switch"', brand)
        self.assertIn('data-lang="zh"', brand)
        self.assertIn('data-lang="en"', brand)
        js = _repo_file("app/kb_server/static/app.js")
        self.assertNotIn('toLocaleString("zh-CN"', js)        # times / numbers follow the UI language
        self.assertNotIn('toLocaleTimeString("zh-CN"', js)
        self.assertIn("I18N.locale", js)
        i18n = _repo_file("app/kb_server/static/i18n.js")
        self.assertIn('localStorage.getItem("kb.lang")', i18n)
        self.assertIn("location.reload()", i18n)
        self.assertIn("applyStatic(document.body)", i18n)
        # The header banners (CSS content) also have an English version in the English UI
        self.assertIn("html[lang=en] body.backend-offline header::after{content:", html)
        self.assertIn("html[lang=en] body.parse-disabled header::before{content:", html)

    def test_chunk_preview_renders_markdown_and_formulas_with_the_vendored_libraries(self) -> None:
        """The chunk preview draws each chunk as Markdown (tables, code, headings) with the pipeline's marker lines as
        labels and EQUATION / $$ blocks through KaTeX, next to the raw text the model actually sees. Both renderers are
        vendored under static/vendor so the console works without the internet; document text never becomes HTML
        (markdown-it html:false, KaTeX trust:false)."""
        static = Path(__file__).resolve().parents[1] / "kb_server" / "static"
        for rel in ("vendor/markdown-it.umd.min.js", "vendor/katex.min.js", "vendor/katex.min.css"):
            self.assertTrue((static / rel).is_file(), rel)
        self.assertEqual(len(list((static / "vendor" / "fonts").glob("KaTeX_*.woff2"))), 20)
        html = _repo_file("app/kb_server/static/index.html")
        self.assertIn('<link rel="stylesheet" href="vendor/katex.min.css', html)
        self.assertLess(html.index('<script src="vendor/markdown-it.umd.min.js'), html.index('<script src="i18n.js'))
        self.assertLess(html.index('<script src="vendor/katex.min.js'), html.index('<script src="i18n.js'))
        self.assertIn(".pv-chunk .pv-md{", html)
        js = _repo_file("app/kb_server/static/app.js")
        self.assertIn("markdownit({ html: false, linkify: false", js)
        self.assertIn("throwOnError: false, trust: false", js)
        self.assertIn('data-v="raw"', js)                      # the raw text stays one click away
        self.assertIn('localStorage.setItem("kb.pvview"', js)
        self.assertIn('.pv-filters .chip[data-f]', js)          # view chips are not filter chips
        # Spreadsheet blocks (SHEET / ROWS / HEADER: a | b, then one positional row per line) are not Markdown:
        # they are split like chunker._native_cells (a leading "|" is an empty first cell) and drawn as a table
        self.assertIn('/^HEADER: ?(.*)$/', js)
        self.assertIn('if (s.startsWith("|")) s = " " + s;', js)
        self.assertIn('s.split(" | ").map((c) => c.trim())', js)
        for marker in ("SHEET", "ROWS", "QUESTION", "ANSWER", "Sheet", "Rows", "Title", "Columns"):
            self.assertRegex(js, r"const PV_MARKER_RE = /\^\([^/]*\b" + marker + r"\b[^/]*\): \?\(\.\*\)\$/")

    def test_every_console_string_has_a_dictionary_entry(self) -> None:
        """Every Chinese string on the static page (text, placeholder, title) and every t("…") key in app.js must be
        in the dictionary; one missing entry leaks a Chinese sentence into the English UI."""
        keys = self._dict_keys()
        self.assertGreater(len(keys), 400)
        html = _repo_file("app/kb_server/static/index.html")
        body = re.sub(r"<style>[\s\S]*?</style>", "", html)
        body = re.sub(r"<!--[\s\S]*?-->", "", body)
        missing = []
        for m in re.finditer(r">([^<>]*)<", body):
            txt = m.group(1).strip()
            if txt and self._ZH.search(txt) and txt not in keys:
                missing.append(txt)
        for m in re.finditer(r'(placeholder|title|aria-label)="([^"]*)"', body):
            for line in m.group(2).split("\n"):
                line = line.strip()
                if line and self._ZH.search(line) and line not in keys:
                    missing.append(line)
        js = _repo_file("app/kb_server/static/app.js")
        for m in re.finditer(r'\bt\("((?:[^"\\]|\\.)*)"', js):
            k = m.group(1).replace("\\n", "\n").replace('\\"', '"')
            if self._ZH.search(k) and k not in keys:
                missing.append(k)
        self.assertEqual(missing, [])

    def _dict_values(self) -> set[str]:
        i18n = _repo_file("app/kb_server/static/i18n.js")
        unesc = lambda x: x.replace("\\n", "\n").replace('\\"', '"')
        return {unesc(m.group(1)) for m in re.finditer(r'^\s*"(?:[^"\\]|\\.)*": "((?:[^"\\]|\\.)*)"', i18n, re.M)}

    def test_server_stage_and_timer_labels_translate(self) -> None:
        """The server now writes English directly: stage names, timer task names, reasons an append was blocked and
        model step names must each be some English value in the dictionary so the Chinese UI can map them back."""
        values = self._dict_values()
        from kb_pipeline.graph.build import GRAPH_PHASE_LABELS, GRAPH_STEP_LABELS
        for label in list(GRAPH_PHASE_LABELS.values()) + list(GRAPH_STEP_LABELS.values()):
            self.assertIn(label, values, label)
        for label in ("Parsing document", "Describing images", "Chunking", "Text embedding", "Visual embedding",
                      "Writing vectors", "Keyword indexing", "Done",
                      "Entity resolution", "Description summaries", "Switching version aliases", "Cleaning up old versions",
                      "Mirror scan", "Parse queue", "Graph maintenance check", "Inactive point / parse asset GC",
                      "Weekly cache cleanup", "Monthly log rotation"):
            self.assertIn(label, values, label)
        from kb_server.service import APPEND_SKIP
        for text in APPEND_SKIP.values():
            self.assertIn(text, values, text)

    def test_server_messages_have_a_chinese_rendering(self) -> None:
        """Fixed strings the server throws at the console (ValueError / HTTP detail / validation errors / stage names)
        must all map back to Chinese in the dictionary; sentences with variables go through EN_ZH_PATTERNS in
        i18n.js, and only pure literals are checked here."""
        values = self._dict_values()
        call = re.compile(r'(?:raise ValueError|raise RuntimeError|raise NoGraphCorpus|raise GraphBuildInterrupted|\bstage|'
                          r'errors\.append|warnings\.append|HTTPException\(status_code=\d+, detail=|content=\{"detail": )'
                          r'\(?\s*"((?:[^"\\\n]|\\.)+)"\s*[,)]')
        missing = []
        for rel in ("app/kb_server/service.py", "app/kb_server/api.py", "app/kb_server/main.py",
                    "app/kb_pipeline/maintenance.py", "app/kb_pipeline/graph/build.py", "app/kb_pipeline/graph/schema_flow.py",
                    "app/kb_pipeline/pipeline/parse_job.py"):
            for m in call.finditer(_repo_file(rel)):
                text = m.group(1).replace('\\"', '"')
                if text and text not in values:
                    missing.append(f"{rel}: {text}")
        self.assertEqual(missing, [])

    # Where the errors the console API returns to the page come from: the whole API and service layers; in the
    # pipeline only the functions the console can reach
    _SERVER_MESSAGE_SOURCES = {
        "app/kb_server/api.py": None, "app/kb_server/main.py": None, "app/kb_server/service.py": None,
        "app/kb_pipeline/maintenance.py": {"stop_graph_build_now", "delete_graph_now", "delete_kb_now"},
        "app/kb_pipeline/discovery.py": {"enroll", "adopt_directory"},
        "app/kb_pipeline/limits.py": None,
        "app/kb_pipeline/db.py": {"delete_llm"},
        "app/kb_pipeline/graph/schema_flow.py": None,
        "app/kb_pipeline/graph/build.py": {"resolve_llm_specs"},
    }
    # Shown as they are in both languages: low-level errors passed through verbatim (a missing key, an exception with
    # its type name, a malformed graph.json) and the directory checks of adopt
    _UNTRANSLATED_MESSAGES = {"not found: 12", "12: 12", "Expecting one of 12",
                              "directory 12 does not exist under the mirror root", "12 already lives in 12"}

    @classmethod
    def _message_sample(cls, node) -> str | None:
        """Error expression -> one sample message: a literal as it is, the variable slots of an f-string filled with
        12, the two sides of a concatenation taken separately."""
        import ast

        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.JoinedStr):
            return "".join(str(v.value) if isinstance(v, ast.Constant) else "12" for v in node.values)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = cls._message_sample(node.left), cls._message_sample(node.right)
            return None if left is None and right is None else (left or "12") + (right or "12")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get" and len(node.args) == 2:
            return cls._message_sample(node.args[1])          # X.get(key, fallback message)
        return None

    def _server_messages(self) -> list[tuple[str, int, str]]:
        import ast

        found = []
        for rel, only in self._SERVER_MESSAGE_SOURCES.items():
            tree = ast.parse(_repo_file(rel))
            owner: dict[int, str] = {}
            for fn in ast.walk(tree):
                if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for child in ast.walk(fn):
                        owner.setdefault(id(child), fn.name)
            for node in ast.walk(tree):
                if only is not None and owner.get(id(node)) not in only:
                    continue
                expr = None
                if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
                    call = node.exc
                    detail = [kw.value for kw in call.keywords if kw.arg == "detail"]
                    expr = detail[0] if detail else (call.args[0] if call.args else None)
                elif isinstance(node, ast.Call) and getattr(node.func, "id", "") == "JSONResponse":
                    for kw in node.keywords:
                        if kw.arg == "content" and isinstance(kw.value, ast.Dict):
                            expr = next((v for k, v in zip(kw.value.keys, kw.value.values)
                                         if isinstance(k, ast.Constant) and k.value == "detail"), None)
                elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "append"
                      and getattr(node.func.value, "id", "") in ("errors", "warnings", "stale") and node.args):
                    expr = node.args[0]
                text = self._message_sample(expr) if expr is not None else None
                if text:
                    found.append((rel, node.lineno, text))
        return found

    def test_server_error_messages_translate(self) -> None:
        """Server errors go through t(): in the Chinese UI the whole sentence maps back through the dictionary, or an
        EN_ZH_PATTERNS entry matches it when it carries variables. test_server_messages_have_a_chinese_rendering only
        sees plain literals; this one collects every error the console API can return, f-strings and concatenations
        included, so a sentence with variables cannot reach the Chinese UI untranslated."""
        values = self._dict_values()
        i18n = _repo_file("app/kb_server/static/i18n.js")
        block = i18n.split("const EN_ZH_PATTERNS = [", 1)[1].split("\n];", 1)[0]
        patterns = [re.compile(m.group(1)) for m in re.finditer(r"^\s*\[/((?:[^/\\\n]|\\.)+)/, ", block, re.M)]
        self.assertGreater(len(patterns), 50)
        messages = self._server_messages()
        self.assertGreater(len(messages), 60)
        missing = [f"{rel}:{line} {text}" for rel, line, text in messages
                   if text not in self._UNTRANSLATED_MESSAGES and text not in values
                   and not any(p.search(text) for p in patterns)]
        self.assertEqual(missing, [])

    def test_sidebar_search_box_aligns_with_its_header(self) -> None:
        """The generic input rule (width:100%) outranked .side-search, so the search box was once 20px wider than
        the header row and touched the right edge; the selector must beat it for the box to get the same 10px
        padding as "knowledge bases / count"."""
        css = _repo_file("app/kb_server/static/index.html")
        self.assertIn(".sidebar input.side-search{display:block;width:calc(100% - 20px);margin:0 10px 8px", css)


class TuneButtonStateTests(unittest.TestCase):
    """The "extracting" state of the "extract labels" button used to be just text changed by that one click: switch
    KB, change page or reload and it was gone while the extraction was still running on the server; switching to
    another KB during extraction even filled the result into the other KB's form. Now the state comes from the
    server: every KB in the overview carries schema_suggest (the mark schema_flow sets while extracting labels),
    the console displays from it, and when the mark disappears the config is refetched."""

    def test_overview_reports_a_running_label_extraction(self) -> None:
        from kb_pipeline.graph import schema_flow
        from kb_server import service

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.db"
            db.init_db(path)
            with db.connect(path) as con:
                self.assertIsNone(service._schema_suggest_info(con, "kb_x"))
                db.set_app_config(con, schema_flow.SUGGEST_MARK_PREFIX + "kb_x", {"origin": "manual", "started_at": int(time.time()) - 30})
                info = service._schema_suggest_info(con, "kb_x")
                self.assertEqual(info["origin"], "manual")
                self.assertGreaterEqual(info["seconds"], 30)
                db.set_app_config(con, schema_flow.SUGGEST_MARK_PREFIX + "kb_x", {"origin": "manual", "started_at": 1})
                self.assertIsNone(service._schema_suggest_info(con, "kb_x"))     # a stale mark left by a killed process
        svc = _repo_file("app/kb_server/service.py")
        self.assertIn('entry["schema_suggest"] = _schema_suggest_info(con, entry["kb_id"])', svc)

    def test_console_draws_the_button_from_server_state_and_keeps_results_in_their_kb(self) -> None:
        js = _repo_file("app/kb_server/static/app.js")
        self.assertIn("function syncTuneButton(kb)", js)
        render = js.split("async function renderConfig()", 1)[1].split("\n}\n", 1)[0]
        self.assertIn("syncTuneButton(kb);", render)
        handler = js.split('$("#cfg-gtune").addEventListener', 1)[1].split("\n});", 1)[0]
        self.assertNotIn('btn.textContent = t("抽取中…")', handler)        # the click handler no longer changes the text itself
        self.assertIn("tuneInFlight = kb.kb_id", handler)
        self.assertIn("selectedEntry()?.kb_id !== kb.kb_id", handler)       # KB switched during extraction: do not fill another KB's form
        sync = js.split("function syncTuneButton(kb)", 1)[1].split("\n}\n", 1)[0]
        self.assertIn("kb.schema_suggest", sync)
        self.assertIn("state.cfgKey = null", sync)                           # refetch the config after extraction so the new version appears


class GraphPreviewRaceTests(unittest.TestCase):
    """Codex review F08: the graph preview's cache check looked only at KB and version, so clicking two views in
    quick succession let the earlier, slower response overwrite the later one — the UI had B selected but drew A;
    the pending mechanism was in turn swallowed by the "same version, within five seconds" throttle. Now the request
    identity = KB + version + view parameters, every request carries a sequence number, and a response that is not
    the latest sequence number is dropped."""

    def test_stale_preview_responses_are_dropped(self) -> None:
        js = _repo_file("app/kb_server/static/app.js")
        fn = js.split("async function refreshGraphPreview(", 1)[1].split("\n}\n", 1)[0]
        self.assertIn("const seq = ++gpReqSeq;", fn)
        self.assertIn("if (seq !== gpReqSeq) return;", fn)
        self.assertIn('JSON.stringify([kb.kb_id, version, view.q || "", view.key || "", view.upper || "", $("#gp-limit").value])', fn)
        self.assertIn("state.gp.viewKey === viewKey", fn)
        self.assertNotIn("state.gp.pending", js)
        self.assertNotIn("state.gp.busy", js)


def _js_function(js: str, head: str) -> str:
    """Source of one top-level function in app.js (from its head to the next closing brace at column 0)."""
    return js.split(head, 1)[1].split("\n}\n", 1)[0]


class ConsoleFormAndDrawerStateTests(unittest.TestCase):
    """The form draft stashed when switching KBs, the content cache of the side drawer, and the preview when the
    latest build did not finish."""

    def test_a_draft_is_only_what_was_edited_on_the_loaded_form(self) -> None:
        """The form of a KB that is not enabled holds placeholder values (400 / 80 / empty prompt), and until the
        config arrives it holds the previous KB's values. Stashed as a draft and filled back in once the KB is
        enabled, they were shown instead of the saved config, and one more click on save wrote them. Only edits made
        against the filled-in form are kept."""
        js = _repo_file("app/kb_server/static/app.js")
        stash = _js_function(js, "function stashDraft(dir) {")
        self.assertIn("if (!dir || !formBase || formBase.dir !== dir) return;", stash)
        self.assertIn("draftCache.delete(dir)", stash)                       # nothing edited, no draft kept
        render = _js_function(js, "async function renderConfig() {")
        self.assertIn("if (!active) draftCache.delete(kb.dir);", render)      # KB closed / deleted / not yet enabled: the draft is void
        self.assertIn("if (!(active && formBase && formBase.dir === kb.dir)) formBase = null;", render)
        filled = render.index("formBase = { dir: kb.dir, ...parseFormValues() };")
        self.assertLess(render.index('$("#cfg-prompt").value = cfg.config.vlm_prompt || "";'), filled)   # counts only once filled in
        self.assertLess(filled, render.index("restoreDraft(kb.dir)"))
        self.assertIn('t("有未保存的修改")', render)                           # a draft filled back in must be visible
        save = js.split('$("#cfg-save").addEventListener', 1)[1].split("\n});", 1)[0]
        self.assertIn("formBase = { dir: kb.dir, ...parseFormValues() };", save)

    def test_every_write_to_the_side_drawer_goes_through_the_html_cache(self) -> None:
        """setHtml remembers what it last wrote into a container and skips an identical write. With direct innerHTML
        writes mixed into the drawer the cache and the page disagreed: reopening the timeline of the same finished
        job, the content equalled last time's, was judged "unchanged", and the drawer stayed at "Loading"."""
        js = _repo_file("app/kb_server/static/app.js")
        for head in ("async function openChunkDrawer(fileId) {", "function renderChunkPreview() {",
                     "async function openMergesDrawer() {", "function renderMerges() {", "function openJob(jobId) {",
                     "async function refreshJobDetail() {"):
            body = _js_function(js, head)
            self.assertNotIn(".innerHTML", body, head)
            self.assertIn("setHtml(", body, head)

    def test_preview_keeps_drawing_the_current_version_when_the_latest_build_did_not_finish(self) -> None:
        """When the latest build failed or was paused, the version built before still serves search and the status
        card still shows its size; the preview must not say "no version has been built yet"."""
        js = _repo_file("app/kb_server/static/app.js")
        fn = _js_function(js, "async function refreshGraphPreview(")
        self.assertIn('const ok = !!version && !!kb.graph_status && kb.graph_status !== "disabled";', fn)
        draw = _js_function(js, "function renderGraphPreview() {")
        self.assertIn('kb.graph_status === "failed" || kb.graph_status === "stopped"', draw)
        self.assertIn('t("画的是现行版本;最近一次建图没有完成")', draw)


class ConsoleRaceAndFeedbackTests(unittest.TestCase):
    """Button disabled states, late responses, leftovers on screen after switching KBs, and messages that did not
    match what actually happened."""

    def test_single_service_restart_keeps_its_button_disabled_across_redraws(self) -> None:
        """The service panel is redrawn wholesale every few seconds. The disabled state of a single service's
        "Restart" button was set only on the old node, so it was clickable again after about 2 seconds and the
        database containers could be restarted a second time while starting up; the "database service" row has no
        key of that name in /health either, so turning green was never seen. The disabled state is kept in a table,
        turning green is judged by the row's own check, and the service must first be seen going down."""
        js = _repo_file("app/kb_server/static/app.js")
        health = js.split("async function _refreshHealth", 1)[1].split("\nasync function ", 1)[0]
        self.assertIn('${busy || svcRestarting.has(key) ? "disabled" : ""}', health)
        restart = _js_function(js, "async function onRestartService(ev) {")
        self.assertIn("if (svcRestarting.has(key)) return;", restart)
        self.assertIn("SERVICES.find(([, k]) => k === key)", restart)
        self.assertIn("if ((mark.wentDown && up) || Date.now() > mark.until)", restart)
        self.assertNotIn("state.health[key]", js)

    def test_merges_drawer_drops_a_response_that_is_no_longer_wanted(self) -> None:
        js = _repo_file("app/kb_server/static/app.js")
        fn = _js_function(js, "async function openMergesDrawer() {")
        self.assertIn("const seq = ++mergesReqSeq;", fn)
        self.assertIn('seq !== mergesReqSeq || state.side.kind !== "merges" || selectedEntry()?.kb_id !== kb.kb_id', fn)
        self.assertEqual(fn.count("if (stale()) return;"), 2)             # both the success and the failure path check

    def test_preview_shows_loading_after_a_switch_and_sends_one_request_per_view(self) -> None:
        """After a KB switch the canvas and legend kept the previous KB's graph until the new one arrived; while a
        request for a view was in flight, every poll sent another identical one."""
        js = _repo_file("app/kb_server/static/app.js")
        reset = _js_function(js, "function resetWorkspace() {")
        self.assertIn("loading: GP_PENDING", reset)
        self.assertIn("renderGraphPreview();", reset)
        fn = _js_function(js, "async function refreshGraphPreview(")
        guard = fn.index("if (state.gp.loading === viewKey && Date.now() - state.gp.loadingAt < GP_LOAD_PATIENCE_MS) return;")
        self.assertLess(guard, fn.index("const seq = ++gpReqSeq;"))
        self.assertIn("if (!same) renderGraphPreview();", fn)
        draw = _js_function(js, "function renderGraphPreview() {")
        self.assertIn(': state.gp.loading ? t("载入中…")', draw)

    def test_file_table_retry_says_so_when_a_job_is_already_running(self) -> None:
        js = _repo_file("app/kb_server/static/app.js")
        table = _js_function(js, "function renderFilesTable() {")
        self.assertIn('toast(r.already_active ? t("已有任务在飞,不重复排队") : t("已重新排队"));', table)

    def test_a_permanent_delete_is_visible_while_it_runs_and_reports_its_outcome(self) -> None:
        """A permanent delete of a large KB takes minutes. The page used to only grey out the button, without a word;
        users thought nothing happened and reloaded, the browser dropped the pending request and the completion
        message never appeared. Now a click shows a message and the deleting state at once; the overview carries the
        deleting state, so after a reload it still shows and polling runs at the busy pace; the outcome is reported
        by the page that sent the request, or, when that request was lost, by the poll noticing the state changed."""
        js = _repo_file("app/kb_server/static/app.js")
        click = js.split('$("#cfg-delete").addEventListener("click"', 1)[1].split("\n});", 1)[0]
        sent = click.index('api(`/kbs/${encodeURIComponent(kb.kb_id)}`, { method: "DELETE" })')
        for before in ("pendingDeletes.add(kb.kb_id);", 't("正在彻底删除「{0}」,数据多的要几分钟", kb.dir)', 'kb.state = "deleting";',
                       "renderKbNav();", "renderConfig();"):
            self.assertLess(click.index(before), sent, before)                  # the page has changed before the request goes out
        self.assertIn("if (!kb || !kb.kb_id || pendingDeletes.has(kb.kb_id)) return;", click)
        self.assertIn("await refreshOverview(state.selected === kb.dir);", click)   # moved on to another KB while waiting: its form is not refilled
        seen = _js_function(js, "function noteDeleteOutcomes(kbs) {")
        self.assertIn('if (k.kb_id && pendingDeletes.has(k.kb_id)) k.state = "deleting";', seen)
        self.assertIn("if (pendingDeletes.has(id)) continue;", seen)             # a page reports its own request, not twice
        self.assertIn('now.state === "delete_failed"', seen)
        poll = _js_function(js, "async function refreshOverview(")
        self.assertLess(poll.index("noteDeleteOutcomes(data.kbs);"), poll.index("state.overview = data;"))
        tick = _js_function(js, "async function tick() {")
        self.assertIn('.some(k => k.state === "deleting")', tick)
        render = _js_function(js, "async function renderConfig() {")
        self.assertIn('|| kb.state === "deleting" || kb.state === "delete_failed";', render)       # the switch cannot be flipped
        self.assertIn('$("#cfg-delete").disabled = !(kb.kb_id && !draft) || kb.state === "deleting";', render)
        text = _js_function(js, "function kbStateText(kb) {")
        self.assertLess(text.index('kb.state === "delete_failed"'), text.index("kb.gc_exempt"))    # no longer says "re-enable to restore"

    def test_draft_form_load_checks_the_kb_has_not_changed(self) -> None:
        """Switching to another KB while the draft branch waited for /limits and /llms filled the directory presets
        into that KB's form afterwards."""
        js = _repo_file("app/kb_server/static/app.js")
        render = _js_function(js, "async function renderConfig() {")
        draft = render.split("} else if (draft) {", 1)[1]
        self.assertLess(draft.index("if (state.cfgKey !== key) return;"), draft.index('$("#cfg-max").value = dd.max_tokens ?? 400;'))


class ConsoleFixRegressionTests(_CodexAudit20260906TestsSupport, unittest.TestCase):
    """Regressions for problems found by past re-reviews, health checks and audits; each case's docstring names the
    source and the symptom at the time."""

    def test_f10_console_shows_the_stored_page_number_as_is(self) -> None:
        """The parsing layer already converts MinerU's 0-based page numbers to 1-based (common.page_idx); the front end
        must not add 1 again."""
        js = _repo_file("app/kb_server/static/app.js")
        self.assertNotIn("Number(c.page_idx) + 1", js)
        self.assertIn("p${esc(String(c.page_idx))}", js)
        self.assertEqual(page_idx({"page_idx": 0}), 1)
