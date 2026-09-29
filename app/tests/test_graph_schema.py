"""Label system: type table, version ring, sampling, corpus examples and constraint feedback."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from kb_pipeline import db
from kb_pipeline.graph import prompts, schema as graph_schema
from kb_pipeline.graph.extract import ExtractionSchema
from kb_pipeline.graph.merge import merge_extractions
from kb_pipeline.models import KBSource

from _support import _client, _ext, _unit


class GraphEntityTypesTests(unittest.TestCase):
    """The graph extraction type table is live per-KB configuration: SQLite -> graph_env -> settings.yaml, empty
    falls back to the global default, and dropping the graph clears it."""

    def test_normalize_dedups_case_insensitively_and_splits_wide_separators(self) -> None:
        from kb_pipeline.limits import normalize_entity_types

        self.assertEqual(normalize_entity_types(["Pin", " Voltage ", "pin", ""]),
                         ("Pin", "Voltage"))
        # Full-width comma / enumeration comma / semicolon — bound to appear when Chinese is pasted into the input box
        wide = "Pin" + chr(0xFF0C) + "Voltage" + chr(0x3001) + "Register" + chr(0xFF1B) + "Clock"
        self.assertEqual(normalize_entity_types(wide),
                         ("Pin", "Voltage", "Register", "Clock"))
        self.assertEqual(normalize_entity_types(None), ())

    def test_unset_sample_size_is_not_an_error(self) -> None:
        """The effective view of the config may lack the key entirely (old KBs, old test cases); that is "unset",
        not "invalid"."""
        from kb_pipeline.limits import validate_graph_tune_config

        self.assertEqual(validate_graph_tune_config(None, None), [])
        self.assertTrue(validate_graph_tune_config(None, 0))

    def test_no_undefined_global_names_in_pipeline_and_server(self) -> None:
        import builtins
        import symtable

        root = Path(__file__).resolve().parents[1]
        problems: list[str] = []
        for path in sorted(p for pkg in ("kb_pipeline", "kb_server", "kb_search") for p in (root / pkg).rglob("*.py")):
            src = path.read_text(encoding="utf-8")
            top = symtable.symtable(src, str(path), "exec")
            defined = {s.get_name() for s in top.get_symbols() if s.is_assigned() or s.is_imported() or s.is_parameter()}
            defined |= set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__spec__", "__builtins__", "__package__", "__loader__",
                                             "__conditional_annotations__"}   # implicit module name on Python 3.14+ (PEP 649 deferred annotations)
            stack = [top]
            while stack:
                table = stack.pop()
                stack.extend(table.get_children())
                for sym in table.get_symbols():
                    if sym.is_referenced() and (sym.is_global() or (table is top and not sym.is_assigned() and not sym.is_imported())):
                        if sym.get_name() not in defined and not sym.is_assigned() and not sym.is_imported():
                            problems.append(f"{path.relative_to(root)}: {sym.get_name()} (in {table.get_name()})")
        self.assertEqual(problems, [])

    def test_extra_sources_carry_the_same_graph_fields(self) -> None:
        """KB_EXTRA_SOURCES_JSON is the second KBSource construction path. One missing field and a dry-run started
        through it runs on a configuration different from the real graph build, with the difference invisible."""
        import inspect

        from kb_pipeline import config as config_module

        src = inspect.getsource(config_module._load_extra_sources)
        for field in ("graph_entity_types", "graph_language", "graph_tune_sample_size"):
            self.assertIn(field, src, f"extra source 少了 {field}")

    def test_dropping_the_graph_clears_derived_schema_only(self) -> None:
        """The type table and language are derived data induced from this version of the corpus; without the graph
        their basis is gone too. The sample size and the entity-label model are set by hand and must not be
        cleared."""
        from unittest import mock

        from kb_pipeline import discovery, maintenance

        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"
            db.init_db(state)
            with db.connect(state) as con:
                discovery.init_schema(con)
                con.execute(
                    "INSERT INTO kb_sources(kb_id, collection, source_root, status,"
                    " first_seen_at, last_seen_at, config_json) VALUES(?,?,?,?,?,?,?)",
                    ("kb_003", "kb_003", "半导体", "active", 0, 0, json.dumps({
                        "graph_enabled": True,
                        "graph_entity_types": ["Pin"],
                        "graph_language": "Chinese",
                        "graph_predicates": [{"name": "has_pin"}],
                        "graph_parent_types": {"Pin": "interface"},
                        "graph_tune_sample_size": 12,
                        "graph_llm": {"extract": "m", "tune": "m"},
                    })))
                con.commit()

                class S:
                    graph_work_dir = Path(tmp) / "work"
                    qdrant_url = "http://127.0.0.1:1"      # unreachable: 6333 on a deployment host's loopback is a live store
                    qdrant_api_key = ""

                errors: list[str] = []
                with mock.patch("kb_pipeline.vector.qdrant.client", return_value=mock.Mock()), \
                     mock.patch.object(maintenance, "_drop_graph_artifacts", return_value={}), \
                     mock.patch.object(maintenance, "_drop_neo4j_projection", return_value={}):
                    maintenance._drop_graph_data(S, con, kb_id="kb_003",
                                                 collection="kb_003", errors=errors)
                con.commit()
                left = discovery.get_config(con, "kb_003")

        self.assertEqual(errors, [])
        self.assertNotIn("graph_entity_types", left)
        self.assertNotIn("graph_language", left)
        self.assertNotIn("graph_predicates", left)
        self.assertNotIn("graph_parent_types", left)
        self.assertEqual(left.get("graph_tune_sample_size"), 12)
        self.assertEqual(left.get("graph_llm", {}).get("tune"), "m")

    def test_tune_slot_is_accepted_but_not_required_for_building(self) -> None:
        """The tune slot sits in graph_llm alongside the three build steps, but it is not a precondition for
        building — an existing KB must not become unbuildable just because this slot appeared."""
        from kb_pipeline.graph.build import GRAPH_LLM_STEPS, GRAPH_TUNE_STEP
        from kb_server import service

        self.assertNotIn(GRAPH_TUNE_STEP, GRAPH_LLM_STEPS)
        self.assertFalse(hasattr(service, "GRAPH_LLM_STEPS"))      # the console keeps no copy of its own; validation takes it from graph.build

        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"
            db.init_db(state)
            limits = {"embedding_max_model_len": 4096, "effective_max_model_len": 4096,
                      "max_tokens_cap": 3276, "max_tokens_min": 128, "cap_ratio": 0.8,
                      "overlap_rule": "", "live": False}
            with db.connect(state) as con:
                effective = {"max_tokens": 400, "overlap_tokens": 80,
                             "graph_chunk_size": 1200, "graph_chunk_overlap": 100}
                errors = service._validate_config(
                    con, effective, {"graph_llm": {"tune": ""}}, limits)
                self.assertEqual(errors, [])
                bad = service._validate_config(
                    con, effective, {"graph_llm": {"nope": ""}}, limits)
                self.assertTrue(any("Unknown graph step" in e for e in bad))

    def test_extraction_schema_uses_kb_list_or_falls_back_to_default(self) -> None:
        from kb_pipeline import discovery
        from kb_pipeline.graph.build import extraction_schema_for
        from kb_pipeline.limits import GRAPH_ENTITY_TYPES_DEFAULT

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "mirror" / "半导体").mkdir(parents=True)

            def schema_for(config):
                return extraction_schema_for(discovery.build_source(root / "mirror", "半导体", config, kb_id="kb_003"))

            self.assertEqual(schema_for({}).entity_types, tuple(GRAPH_ENTITY_TYPES_DEFAULT))
            mine = schema_for({"graph_entity_types": ["Memory Device", "Pin", "电压轨"],
                               "graph_predicates": [{"name": "has_pin", "source_parents": ["component"], "target_parents": ["interface"]}],
                               "graph_parent_types": {"Pin": "interface"}, "graph_language": "Chinese"})
            self.assertEqual(mine.entity_types, ("Memory Device", "Pin", "电压轨"))
            self.assertEqual(mine.predicate_names, ("has_pin",))
            self.assertEqual(mine.allowed_ends()["has_pin"], ({"component"}, {"interface"}))
            self.assertEqual((mine.language, mine.parent_types), ("Chinese", {"Pin": "interface"}))


class SchemaVersionRingTests(unittest.TestCase):
    """Label version ring: at most 3 versions, newest first, **never evicting the version currently in effect**.

    Evicting the active version means the dropdown can no longer select the labels the current graph was built
    with, while the graph is still built on them — UI and output are permanently out of step, with no hint at all.
    """

    def test_newest_first_and_capped(self) -> None:
        from kb_pipeline.limits import GRAPH_SCHEMA_VERSION_MAX, push_schema_version

        ring = []
        for i in range(6):
            ring = push_schema_version(ring, {"id": f"v{i}"})
            self.assertLessEqual(len(ring), GRAPH_SCHEMA_VERSION_MAX)
        self.assertEqual([v["id"] for v in ring], ["v5", "v4", "v3"])

    def test_active_version_is_never_evicted(self) -> None:
        from kb_pipeline.limits import push_schema_version

        ring = [{"id": "c"}, {"id": "b"}, {"id": "a"}]      # a is the oldest
        # The active version is the oldest, a: the one evicted must be b, not a
        out = push_schema_version(ring, {"id": "d"}, active_id="a")
        ids = [v["id"] for v in out]
        self.assertEqual(ids[0], "d")
        self.assertIn("a", ids)
        self.assertEqual(len(ids), 3)

    def test_reextracting_the_same_id_does_not_duplicate(self) -> None:
        from kb_pipeline.limits import push_schema_version

        ring = [{"id": "b"}, {"id": "a"}]
        out = push_schema_version(ring, {"id": "b", "entity_types": ["x"]})
        self.assertEqual([v["id"] for v in out], ["b", "a"])
        self.assertEqual(out[0]["entity_types"], ["x"])      # the new copy is used


class SchemaSamplingTests(unittest.TestCase):
    def test_sampling_is_deterministic_and_budgeted(self) -> None:
        chunks = [{"doc_id": "d", "chunk_index": i, "chunk_uid": f"c{i}", "text": f"text {i} " * 5} for i in range(20)]
        a, _ = graph_schema.sample_texts(chunks, 5)
        b, _ = graph_schema.sample_texts(list(reversed(chunks)), 5)
        self.assertEqual(a, b)
        small, truncated = graph_schema.sample_texts(chunks, 20, budget=30)
        self.assertTrue(truncated)
        self.assertLess(len(small), 20)

    def test_normalizers(self) -> None:
        self.assertEqual(graph_schema.parse_entity_types({"entity_types": ["a", " b "]}), ["a", "b"])
        self.assertEqual(graph_schema.parse_entity_types('"entity_types": [x, y]'), ["x", "y"])
        self.assertEqual(len(graph_schema.parse_entity_types({"entity_types": [f"t{i}" for i in range(80)]})), graph_schema.MAX_ENTITY_TYPES)
        self.assertEqual(graph_schema.parse_entity_types({"entity_types": ["Pin", "pin", "signal"]}), ["Pin", "signal"])
        self.assertIn("Return at most 30 entity types", prompts.ENTITY_TYPE_GENERATION_PROMPT)
        self.assertEqual(graph_schema.parse_json_object('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(graph_schema.parse_json_object('Sure: {"a": 1} done'), {"a": 1})
        # A response stopped at the max_tokens edge: a missing }, stopped inside a string, last element unfinished — all
        # must be salvageable (kb_001 once lost a whole predicate table)
        self.assertEqual(graph_schema.parse_json_object('{"predicates": [{"name": "a"}, {"name": "b"}]'), {"predicates": [{"name": "a"}, {"name": "b"}]})
        self.assertEqual(graph_schema.parse_json_object('```json\n{"x": [1, {"y": "z'), {"x": [1, {"y": "z"}]})
        self.assertEqual(graph_schema.parse_json_object('{"predicates": [{"name": "a", "description": "d"}, {"name": "b", "desc'),
                         {"predicates": [{"name": "a", "description": "d"}]})
        self.assertEqual(graph_schema.parse_json_object("not json"), {})
        self.assertEqual(graph_schema.repair_json('{"a": [1, 2'), '{"a": [1, 2]}')
        src = (Path(__file__).resolve().parents[1] / "kb_pipeline" / "graph" / "schema.py").read_text(encoding="utf-8")
        self.assertIn("validate=lambda t: bool(normalize_predicates(parse_json_object(t)))", src)   # an unparseable predicate response is requested once more
        preds = graph_schema.normalize_predicates({"predicates": [
            {"name": "Has Pin", "description": "d", "source_parents": ["Component"], "target_parents": ["interface"]},
            {"name": "related_to"}, {"name": "has-pin"}, "part_of"]})
        self.assertEqual([p["name"] for p in preds], ["has_pin", "part_of"])
        self.assertEqual(preds[0]["source_parents"], ["component"])
        parents = graph_schema.normalize_parent_types({"parent_types": {"Pin": "Interface", "ghost": "x"}}, ["pin"])
        self.assertEqual(parents, {"pin": "part"})     # the invented parent type interface → one of the six upper classes

    def test_suggest_runs_the_six_calls(self) -> None:
        answers = ["Semiconductor datasheets", "English", "You are an expert.",
                   '{"entity_types": ["memory device", "pin"]}',
                   '{"parent_types": {"memory device": "component", "pin": "interface"}}',
                   '{"predicates": [{"name": "has_pin", "description": "d", "source_parents": ["component"], "target_parents": ["interface"]}]}']
        client = _client(list(answers))
        out = graph_schema.suggest(client, ["sample text"])
        self.assertEqual(out["domain"], "Semiconductor datasheets")
        self.assertEqual(out["entity_types"], ["memory device", "pin"])
        self.assertEqual(out["parent_types"], {"memory device": "entity", "pin": "part"})
        self.assertEqual(out["predicates"][0]["name"], "has_pin")
        self.assertNotIn("competency_questions", out)     # capability questions withdrawn 2026-09-06
        self.assertEqual(client.stats["calls"], 6)


class SamplingStratificationTests(unittest.TestCase):
    """Health check D1: label-extraction sampling is stratified by document, excludes boilerplate chunks, and uses
    farthest-point sampling when vectors are available."""

    def _chunks(self) -> list[dict]:
        chunks = []
        for i in range(60):        # big document, 60 chunks
            chunks.append({"doc_id": "big", "chunk_index": i, "point_id": f"b{i}",
                           "text": f"big document body sentence number {i} about memory timing and voltage."})
        for i in range(3):         # small document, 3 chunks
            chunks.append({"doc_id": "small", "chunk_index": i, "point_id": f"s{i}",
                           "text": f"small note {i} on package thermal resistance."})
        toc = ("1.1 Overview .......... 3\n1.2 Features .......... 4\n2.1 Pinout .......... 7\n"
               "2.2 Timing .......... 9\n3 Ordering .......... 12\n4 Revision .......... 15")
        chunks.append({"doc_id": "big", "chunk_index": 99, "point_id": "toc", "text": toc, "section": "Table of Contents"})
        return chunks

    def test_stratified_pool_drops_boilerplate_and_is_deterministic(self) -> None:
        out = graph_schema.sample_chunks(self._chunks(), 6)
        self.assertEqual(out["method"], "stratified")
        self.assertEqual(out["excluded_boilerplate"], 1)
        self.assertEqual(out["documents_total"], 2)
        self.assertEqual(len(out["texts"]), 6)
        self.assertNotIn("Overview ..........", " ".join(out["texts"]))
        again = graph_schema.sample_chunks(list(reversed(self._chunks())), 6)
        self.assertEqual(out["texts"], again["texts"])       # the same corpus gives the same result twice

    def test_vectors_pick_the_most_diverse_chunks(self) -> None:
        chunks = self._chunks()

        def fetch(pool):
            # the big document's chunks point almost the same way, the small document's along another axis:
            # farthest-point sampling must pick the small document
            return {c["point_id"]: ([1.0, 0.01 * c["chunk_index"]] if c["doc_id"] == "big" else [0.0, 1.0])
                    for c in pool}

        out = graph_schema.sample_chunks(chunks, 4, fetch_vectors=fetch)
        self.assertEqual(out["method"], "vectors")
        self.assertEqual({c["doc_id"] for c in out["picked"]}, {"big", "small"})
        self.assertEqual(out["documents_covered"], 2)

        def broken(pool):
            raise RuntimeError("qdrant down")

        fallback = graph_schema.sample_chunks(chunks, 4, fetch_vectors=broken)
        self.assertEqual(fallback["method"], "stratified")     # vectors unavailable falls back to stratified random without an error
        self.assertEqual(len(fallback["texts"]), 4)

    def test_allocation_gives_every_document_a_seat(self) -> None:
        quotas = graph_schema._allocate([400, 4, 1], 20)
        self.assertEqual(sum(quotas), 20)
        self.assertTrue(all(q >= 1 for q in quotas))
        self.assertEqual(quotas[2], 1)
        self.assertLess(quotas[0], 18)      # no longer dominated linearly by chunk count
        self.assertEqual(graph_schema._allocate([5, 5], 1), [1, 0])
        self.assertEqual(graph_schema._spread(list(range(10)), 3), [0, 4, 9])


class ExampleSelectionTests(unittest.TestCase):
    """Corpus example selection (4a): endpoint constraints follow the pipeline's relaxation rules and the hard
    threshold became a ranking; the health KB previously kept no passage at all."""

    def _schema(self):
        from kb_pipeline.graph.extract import ExtractionSchema
        return ExtractionSchema(entity_types=("patient", "biomarker", "recommendation"), language="Chinese",
                                predicates=({"name": "has_biomarker", "source_parents": ["entity"], "target_parents": ["property"]},
                                            {"name": "recommends", "source_parents": ["process"], "target_parents": ["*"]}),
                                parent_types={"patient": "entity", "biomarker": "property", "recommendation": "process"})

    def test_score_relaxes_predicates_whose_constraints_are_mostly_violated(self) -> None:
        from kb_pipeline.graph.schema import score_example

        ents = [{"name": "李华", "type": "patient", "description": "a"}, {"name": "尿酸", "type": "biomarker", "description": "b"},
                {"name": "多喝水", "type": "recommendation", "description": "c"}]
        rels = [{"source": "李华", "target": "尿酸", "predicate": "has_biomarker", "description": "", "strength": 5},
                {"source": "尿酸", "target": "多喝水", "predicate": "recommends", "description": "", "strength": 5},     # the source side is property, the constraint wants process
                {"source": "尿酸", "target": "李华", "predicate": "recommends", "description": "", "strength": 5},     # same: this predicate violates 2/2 → the constraint is wrong, relaxed
                {"source": "李华", "target": "多喝水", "predicate": "related_to", "description": "", "strength": 3},
                {"source": "幽灵", "target": "尿酸", "predicate": "has_biomarker", "description": "", "strength": 5}]      # endpoint not among the entities: dropped
        sc = score_example(ents, rels, self._schema(), related_predicate="related_to")
        self.assertEqual((len(sc["relations"]), sc["typed"], sc["violations_removed"], sc["relaxed_predicates"]), (4, 3, 0, ["recommends"]))
        self.assertAlmostEqual(sc["related_ratio"], 0.25)
        # A predicate violating in under half its edges only loses the violating ones
        rels2 = rels[:1] + [{"source": "尿酸", "target": "多喝水", "predicate": "has_biomarker", "description": "", "strength": 5}] * 3 + \
                [{"source": "李华", "target": "尿酸", "predicate": "has_biomarker", "description": "", "strength": 5}] * 4
        rels2.append({"source": "尿酸", "target": "多喝水", "predicate": "recommends", "description": "", "strength": 5})
        rels2.append({"source": "尿酸", "target": "李华", "predicate": "recommends", "description": "", "strength": 5})
        sc2 = score_example(ents, rels2, self._schema(), related_predicate="related_to")
        self.assertEqual((sc2["violations_removed"], sc2["relaxed_predicates"]), (3, ["recommends"]))    # has_biomarker violates 3/8: those three are removed; recommends violates 2/2: relaxed
        rels3 = rels[:1] + [{"source": "尿酸", "target": "多喝水", "predicate": "recommends", "description": "", "strength": 5},
                            {"source": "多喝水", "target": "尿酸", "predicate": "recommends", "description": "", "strength": 5}]
        sc3 = score_example(ents, rels3, self._schema(), related_predicate="related_to")
        self.assertEqual((sc3["violations_removed"], sc3["relaxed_predicates"]), (1, []))                # exactly half is not over half: only the violating one is removed

    def test_generate_examples_ranks_and_falls_back(self) -> None:
        from kb_pipeline.graph import schema as S

        good = '("unit"<|>body<|>x)\n##\n("entity"<|>李华<|>patient<|>受检者)\n##\n("entity"<|>尿酸<|>biomarker<|>指标)\n##\n("relationship"<|>李华<|>尿酸<|>has_biomarker<|>测了<|>7)\n<|COMPLETE|>'
        lazy = '("unit"<|>body<|>x)\n##\n("entity"<|>李华<|>patient<|>受检者)\n##\n("entity"<|>多喝水<|>recommendation<|>建议)\n##\n("relationship"<|>李华<|>多喝水<|>related_to<|>有关<|>3)\n<|COMPLETE|>'
        empty = '("unit"<|>body<|>x)\n##\n("entity"<|>李华<|>patient<|>受检者)\n<|COMPLETE|>'
        texts = ["短" * 100, "甲" * 300, "乙" * 250, "丙" * 200]          # the 100-character one is too short (floor 120); the other three are tried in descending length
        client = _client([good, lazy, empty])
        text, stats = S.generate_examples(client, texts, self._schema(), max_examples=2)
        self.assertEqual((stats["tried"], stats["kept"], stats["dropped_empty"], stats["fallback"]), (3, 1, 1, 0))
        self.assertIn("has_biomarker", text)
        self.assertNotIn("多喝水", text)                                   # a passage with related_to share 1.0 ranks below the bar and is skipped when a qualified one exists
        client2 = _client([lazy, empty])
        text2, stats2 = S.generate_examples(client2, texts[1:3], self._schema(), max_examples=2)
        self.assertEqual((stats2["kept"], stats2["fallback"]), (0, 0))      # a related_to-only passage does not even count as a fallback: not one relation with a predicate
        self.assertEqual(text2, "")

    def test_console_reads_the_merge_log_of_the_current_version(self) -> None:
        import json as _json
        import tempfile
        from types import SimpleNamespace
        from unittest import mock

        from kb_pipeline import db
        from kb_server import service

        graph = {"resolution_log": [{"kept": "a", "merged": "b", "kept_title": "FDE", "merged_title": "FDE (Forward-Deployed Engineer)", "source": "auto", "category": "identifier"}],
                 "resolution_rejected": [{"a": "c", "b": "d", "a_title": "Demo", "b_title": "Demo阶段", "source": "lexical", "category": "alias", "reason": "recheck"}],
                 "stats": {"resolution": {"candidates": 3, "yes": 1, "rechecked": 1, "recheck_dropped": 1, "entities_before": 10, "entities_after": 9, "secret": 1}}}
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"; db.init_db(state)
            src = SimpleNamespace(kb_id="kb_1", collection="kb_1")
            cfg = SimpleNamespace(state_db=state, sources={"k1": src})
            with db.connect(state) as con:
                con.execute("INSERT INTO graph_builds(graph_build_id, source_key, kb_id, source_collection, graph_version, status, "
                            "started_at, finished_at, build_kind) VALUES('g1', 'k1', 'kb_1', 'kb_1', 'v1', 'done', 10, 20, 'full')")
                con.commit()
            gdir = Path(tmp) / "work"; gdir.mkdir()
            (gdir / "graph.json").write_text(_json.dumps(graph), encoding="utf-8")
            paths = SimpleNamespace(graph_file=gdir / "graph.json")
            with mock.patch.object(service, "settings", lambda: cfg), mock.patch("kb_pipeline.graph.build.graph_paths", lambda *a, **k: paths):
                out = service.graph_merges("kb_1")
                self.assertEqual((out["version"], out["totals"], out["merges"][0]["merged_title"], out["rejected"][0]["reason"]),
                                 ("v1", {"merges": 1, "rejected": 1}, "FDE (Forward-Deployed Engineer)", "recheck"))
                self.assertNotIn("secret", out["stats"])
                with self.assertRaises(KeyError):
                    service.graph_merges("kb_x")
        api = (Path(__file__).resolve().parents[1] / "kb_server" / "api.py").read_text(encoding="utf-8")
        self.assertIn('@router.get("/kbs/{kb_id}/graph_merges")', api)
        html = (Path(__file__).resolve().parents[1] / "kb_server" / "static" / "index.html").read_text(encoding="utf-8")
        js = (Path(__file__).resolve().parents[1] / "kb_server" / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn('id="gp-merges"', html)
        for needle in ("openMergesDrawer", "graph_merges", "resolution_log", "合并实体", "未合并实体", "并入实体"):
            self.assertIn(needle, js, needle)
        # The schema version keeps the example-selection account
        # 2026-09-08 the ring-insertion code moved into the pipeline (schema_flow), shared by the console and the automatic flow
        flow = (Path(__file__).resolve().parents[1] / "kb_pipeline" / "graph" / "schema_flow.py").read_text(encoding="utf-8")
        self.assertIn('"example_stats": result.get("example_stats")', flow)


class SchemaFeedbackTests(unittest.TestCase):
    """Constraint feedback (2026-09-08 evening): the graph build writes the endpoint account back into the schema
    version, merging consults the account first, re-extracting labels starts from the current version + account and
    a rule keeps the data-confirmed endpoints; a blank graph auto-extracts one label version before building; a
    threshold-triggered full rebuild re-extracts first, then builds; an unselected model slot uses the default
    model."""

    ANSWERS = ["Job market reports", "Chinese", "You are an expert.",
               '{"entity_types": ["role", "skill", "standard"]}',
               '{"parent_types": {"role": "entity", "skill": "property", "standard": "standard"}}',
               '{"predicates": [{"name": "requires_skill", "description": "a role needs a skill", "source_parents": ["entity"], "target_parents": ["property"]}]}',
               '{"definitions": {"role": "a job role", "skill": "an ability", "standard": "a norm"}}',
               '{"subject_types": ["role"], "axis": "none", "conclusion_headings": [], "extension_predicates": ["requires_skill"]}']
    PRIOR = {"id": "v1", "entity_types": ["role", "skill", "standard", "tool"],
             "parent_types": {"role": "entity", "skill": "property", "standard": "standard", "tool": "entity"},
             "predicates": [{"name": "requires_skill", "description": "d", "source_parents": ["entity"], "target_parents": ["property"]},
                            {"name": "calls", "description": "invokes", "source_parents": ["entity"], "target_parents": ["entity"]}],
             "profile": {"subject_types": ["role"], "axis": "none"},
             "observed": {"edges": {"requires_skill": 400, "calls": 60},
                          "confirmed": [{"predicate": "requires_skill", "source_parent": "entity", "target_parent": "standard", "count": 34},
                                        {"predicate": "calls", "source_parent": "entity", "target_parent": "entity", "count": 40}]}}

    def test_prior_pairs_relax_and_endpoint_account(self) -> None:
        from kb_pipeline.graph.merge import confirmed_pairs

        units = [_unit("u1")]
        ext = {"u1": _ext([("Northwind", "organization", "maker"), ("SRAM", "memory device", "chip")],
                          [("Northwind", "SRAM", "produces", "makes it", 8)])}
        kw = dict(entity_types=("organization", "memory device"), parent_types={"organization": "entity", "memory device": "part"},
                  allowed_ends={"produces": ({"entity"}, {"process"})})
        plain = merge_extractions(units, ext, **kw)
        self.assertTrue(plain["relations"][0]["type_violation"])                         # part is not in the target constraint
        observed = plain["stats"]["endpoint_observed"]
        self.assertEqual((observed["edges"], observed["pairs"]), ({"produces": 1}, {"produces": {"entity->part": 1}}))
        self.assertEqual(observed["confirmed"], [])                                       # 1 edge is below the threshold
        # Health-check numbers: violations per predicate, entities per type, type names outside the table (none here)
        self.assertEqual((observed["violations"], observed["type_counts"], observed["unknown_types"]),
                         ({"produces": 1}, {"organization": 1, "memory device": 1}, {}))
        # A combination confirmed in the account: released directly in this version, no need to reach 20 edges
        prior = merge_extractions(units, ext, prior_pairs=[("produces", "entity", "part")], **kw)
        rel = prior["relations"][0]
        self.assertEqual((rel["type_violation"], rel.get("endpoints_prior")), (False, True))
        self.assertEqual(prior["stats"]["type_violations"], 0)
        self.assertEqual(prior["stats"]["endpoint_pairs_prior"], [{"predicate": "produces", "source_parent": "entity", "target_parent": "part", "count": 1, "prior": True}])
        acct = prior["stats"]["endpoint_observed"]
        self.assertEqual(acct["confirmed"], [{"predicate": "produces", "source_parent": "entity", "target_parent": "part", "count": 1}])   # the released pair stays in the account
        self.assertEqual(acct["violations"], {"produces": 1})        # released, yet still outside the declared endpoints: exactly the evidence the constraint is too narrow
        self.assertEqual(confirmed_pairs(acct), [("produces", "entity", "part")])
        self.assertEqual(confirmed_pairs(None), [])
        health = {r["predicate"]: r for r in prior["stats"]["predicate_health"]}
        self.assertEqual(health["produces"]["relaxed_pairs"][0]["source_parent"], "entity")

    def test_endpoint_account_lists_type_names_outside_the_schema(self) -> None:
        units = [_unit("u1")]
        ext = {"u1": _ext([("Northwind", "organization", "maker"), ("Gizmo", "gadget", "a thing"), ("Widget", "gadget", "another")],
                          [("Northwind", "Gizmo", "produces", "makes it", 8)])}
        out = merge_extractions(units, ext, entity_types=("organization",), parent_types={"organization": "entity"})
        acct = out["stats"]["endpoint_observed"]
        self.assertEqual(acct["unknown_types"], {"gadget": 2})
        self.assertEqual(acct["type_counts"], {"gadget": 2, "organization": 1})
        self.assertEqual(out["stats"]["schema_drift_types"], 2)
        # No type table given (rule-based extraction, old callers): no guessing what is outside the table
        self.assertEqual(merge_extractions(units, ext)["stats"]["endpoint_observed"]["unknown_types"], {})

    def test_facts_json_tolerates_markdown_escapes_and_control_chars(self) -> None:
        """2026-09-13: the model wrote the pipes of table output as \\|, json reported Invalid \\escape, and one unit
        brought down the whole KB's graph build; notation noise must be parsed leniently."""
        from kb_pipeline.graph.facts import parse_facts_response

        raw = '{"facts": [{"subject": "InnoDB", "property": "锁", "value": "mysql> select 1;+----+\\| a \\| b \\|", "unit": ""}]}'
        out = parse_facts_response(raw)
        self.assertEqual((out["malformed"], out["truncated"], len(out["facts"])), (0, False, 1))
        self.assertIn("| a \\| b", out["facts"][0]["value"])                                      # the backslash is kept as is
        ctrl = '{"facts": [{"subject": "S", "property": "p", "value": "第一行\n第二行", "unit": ""}]}'   # a real newline inside the string
        self.assertEqual(parse_facts_response(ctrl)["facts"][0]["value"], "第一行\n第二行")
        cut = '{"facts": [{"subject": "S", "property": "p", "value": "a \\| b", "unit": ""}, {"subject": "T", "property": "q", "value": "x", "unit":'
        out2 = parse_facts_response(cut)
        self.assertEqual((out2["malformed"], out2["truncated"], len(out2["facts"])), (0, True, 2))     # truncation + invalid escape salvaged together (the half object is kept too, minus its dangling key)
        self.assertEqual(parse_facts_response("{\"facts\": \"no\"}")["malformed"], 1)                  # truly malformed is still malformed
        # Codex 2026-09-14 R04: repairing JSON syntax must not alter string bodies — with an invalid escape and a
        # literal comma both present, the comma must stay
        both = '{"facts": [{"subject": "S", "property": "p", "value": "a,]b\\|c", "unit": ""}]}'
        got = parse_facts_response(both)
        self.assertEqual((got["malformed"], got["facts"][0]["value"]), (0, "a,]b\\|c"))
        trailing = '{"facts": [{"subject": "S", "property": "p", "value": "x, ]", "unit": "",}, ]}'          # a real trailing comma outside the string
        self.assertEqual(parse_facts_response(trailing)["facts"][0]["value"], "x, ]")

    def test_short_symbols_do_not_bucket_different_quantities_together(self) -> None:
        """2026-09-13 Codex F02: the principal quantum number n and the number of periods N share a symbol but not a
        meaning, so bucketing cannot look at the symbol alone; a long symbol like VCC still forms one bucket."""
        from kb_pipeline.graph.concepts import _bucket_norm, build_concepts

        facts = [
            {"id": "f1", "subject": "氢原子", "property": "主量子数", "symbol": "n", "value": "2", "unit": "", "doc_id": "d1"},
            {"id": "f2", "subject": "债券", "property": "number of periods", "symbol": "N", "value": "30", "unit": "", "doc_id": "d2"},
            {"id": "f3", "subject": "氢原子", "property": "主量子数 n", "symbol": "n", "value": "3", "unit": "", "doc_id": "d3"},
            {"id": "f4", "subject": "S", "property": "Supply voltage", "symbol": "V_CC", "value": "3.3", "unit": "V", "doc_id": "d1"},
            {"id": "f5", "subject": "S", "property": "供电电压", "symbol": "VCC", "value": "3.0", "unit": "V", "doc_id": "d2"},
        ]
        self.assertNotEqual(_bucket_norm(facts[0]), _bucket_norm(facts[1]))          # n and N: different properties, different buckets
        self.assertEqual(_bucket_norm(facts[0]), _bucket_norm(facts[2]))             # same bucket only when the property name matches (whitespace / symbol suffix stripped)
        self.assertEqual(_bucket_norm(facts[3]), _bucket_norm(facts[4]))             # a long symbol is still one bucket by symbol
        self.assertEqual(_bucket_norm({"property": "n", "symbol": "n"}), "n")            # the property is the symbol itself: only the symbol is usable
        concepts, stats = build_concepts([dict(f) for f in facts])
        self.assertEqual(len(concepts), 3)
        labels = sorted(str(c["label"]) for c in concepts)
        self.assertIn("number of periods", labels)
        # Same short symbol, same unit, one property name containing the other (a check-up sheet's HR uric acid / uric
        # acid are both UA µmol/L): merged back deterministically; the symbol-only "t value" and "node time t (years)"
        # differ in unit and neither contains the other, so they stay two concepts
        more = [
            {"id": "g1", "subject": "P", "property": "尿酸", "symbol": "UA", "value": "300", "unit": "umol/L", "doc_id": "d1"},
            {"id": "g2", "subject": "P", "property": "HR尿酸", "symbol": "UA", "value": "310", "unit": "umol/L", "doc_id": "d2"},
            {"id": "g3", "subject": "B", "property": "t值", "symbol": "t", "value": "2.1", "unit": "", "doc_id": "d3"},
            {"id": "g4", "subject": "B", "property": "节点时间 t（年）", "symbol": "t", "value": "3", "unit": "年", "doc_id": "d4"},
        ]
        more = [dict(f) for f in more]
        concepts2, stats2 = build_concepts(more)
        self.assertEqual((len(concepts2), stats2["symbol_contained"]), (3, 1))
        self.assertEqual(len({f.get("concept_key") for f in more[:2]}), 1)
        # Codex 2026-09-14 R03: this route passes the guard too — a Chinese qualifier (leakage current) goes to the
        # sameness judge, an identity token (point P2) is always blocked, and only a Latin-letter code prefix (HR)
        # merges directly
        guard = [
            {"id": "h1", "subject": "S", "property": "电流", "symbol": "I", "value": "1", "unit": "A", "doc_id": "d1"},
            {"id": "h2", "subject": "S", "property": "漏电流", "symbol": "I", "value": "0.2", "unit": "A", "doc_id": "d2"},
            {"id": "h3", "subject": "B", "property": "价格", "symbol": "P2", "value": "10", "unit": "元", "doc_id": "d3"},
            {"id": "h4", "subject": "B", "property": "P2点对应价格", "symbol": "P2", "value": "12", "unit": "元", "doc_id": "d4"},
        ]
        concepts3, stats3 = build_concepts([dict(f) for f in guard])
        self.assertEqual(len(concepts3), 4)                                                          # no sameness model: all kept
        self.assertEqual((stats3["symbol_contained"], stats3["symbol_to_judge"], stats3["blocked_identity"]), (0, 2, 0))
        # With vectors and a sameness model: both pairs become judge candidates; the yes one merges, the no one stays
        seen: list[str] = []

        def reply(messages):
            seen.append(messages if isinstance(messages, str) else json.dumps(messages, ensure_ascii=False))
            return "1. yes\n2. no"

        vec = lambda texts: [[1.0, 0.0] if "价格" in t else [0.0, 1.0] for t in texts]
        concepts4, stats4 = build_concepts([dict(f) for f in guard], embed=vec, client=_client(reply))
        self.assertEqual((stats4["asked_pairs"], stats4["judged_yes"], len(concepts4)), (2, 1, 3))
        self.assertIn("漏电流", seen[0]); self.assertIn("P2点对应价格", seen[0])
        # A bucket whose identity token is only in the property, not the symbol, carries a third segment and never enters
        # this route at all (price P vs price at point P2, P)
        self.assertEqual(len(build_concepts([dict(guard[2], symbol="P"), dict(guard[3], symbol="P")])[0]), 2)

    def test_observed_guard_keeps_data_confirmed_predicates(self) -> None:
        from kb_pipeline.graph.schema import apply_observed_guard

        narrow = [{"name": "requires_skill", "description": "new", "source_parents": ["entity"], "target_parents": ["property"]},
                  {"name": "part_of", "description": "x", "source_parents": ["*"], "target_parents": ["*"]}]
        out, record = apply_observed_guard(narrow, self.PRIOR)
        by = {p["name"]: p for p in out}
        self.assertEqual(by["requires_skill"]["target_parents"], ["property", "standard"])      # the narrowed endpoint is added back
        self.assertEqual(by["requires_skill"]["description"], "new")                             # the description follows the model
        self.assertEqual((by["calls"]["description"], by["calls"]["source_parents"]), ("invokes", ["entity"]))   # a deleted predicate is restored from the previous version
        self.assertEqual(by["part_of"]["source_parents"], ["*"])                                 # wildcard endpoints are left alone
        self.assertEqual(record, {"kept_predicates": ["calls"], "widened": ["requires_skill.target_parents+standard"]})
        self.assertEqual(apply_observed_guard(narrow, None), ([dict(p) for p in narrow], {"kept_predicates": [], "widened": []}))
        self.assertEqual(apply_observed_guard(narrow, {"predicates": [], "observed": {}})[1], {"kept_predicates": [], "widened": []})

    def test_suggest_with_prior_feeds_prompts_and_guards(self) -> None:
        seen: list[str] = []
        answers = list(self.ANSWERS)

        def reply(messages):
            seen.append(messages if isinstance(messages, str) else json.dumps(messages, ensure_ascii=False))
            return answers.pop(0)

        out = graph_schema.suggest(_client(reply), ["sample text"], examples=False, prior=self.PRIOR)
        self.assertEqual(len(seen), 8)
        self.assertIn("previous version of this schema used these entity types: role, skill, standard, tool", seen[3])
        self.assertIn("data-confirmed endpoint pairs: entity -> standard (34 edges)", seen[5])     # the predicate prompt carries the previous version + the account
        self.assertIn("used by 60 edges", seen[5])
        self.assertIn("previous version of this schema described the corpus", seen[7])
        self.assertEqual(out["guard"], {"kept_predicates": ["calls"], "widened": ["requires_skill.target_parents+standard"]})
        self.assertEqual({p["name"] for p in out["predicates"]}, {"requires_skill", "calls"})
        # No prior: the prompts lack these passages and the guard is empty
        seen.clear(); answers[:] = list(self.ANSWERS)
        out2 = graph_schema.suggest(_client(reply), ["sample text"], examples=False)
        self.assertNotIn("previous version", seen[3]); self.assertNotIn("previous version", seen[5])
        self.assertEqual(out2["guard"], {"kept_predicates": [], "widened": []})

    def test_usage_stats_feed_the_resuggest_prompts_but_not_the_first_suggest(self) -> None:
        seen: list[str] = []
        answers = list(self.ANSWERS)

        def reply(messages):
            seen.append(messages if isinstance(messages, str) else json.dumps(messages, ensure_ascii=False))
            return answers.pop(0)

        prior = json.loads(json.dumps(self.PRIOR))
        prior["observed"].update({
            "pairs": {"requires_skill": {"entity->property": 260, "entity->standard": 34, "process->property": 6}, "calls": {"entity->entity": 60}},
            "violations": {"requires_skill": 140},
            "type_counts": {"role": 120, "skill": 300, "standard": 40},
            "unknown_types": {"gadget": 7, "vendor": 3},
        })
        graph_schema.suggest(_client(reply), ["sample text"], examples=False, prior=prior)
        self.assertIn("each type: role (120), skill (300), standard (40), tool (0)", seen[3])          # the type table carries entity counts in the previous version's order
        self.assertIn("not in the list: gadget (7), vendor (3)", seen[3])                                  # names outside the table that surfaced during extraction
        self.assertIn("used by 400 edges in the current graph; 35% of them fall outside the declared ends", seen[5])
        self.assertIn("most common endpoint pairs: entity -> property (260), entity -> standard (34), process -> property (6)", seen[5])
        self.assertIn("data-confirmed endpoint pairs: entity -> standard (34 edges)", seen[5])          # the existing account is unaffected
        self.assertIn("declared too narrowly", seen[5])
        self.assertNotIn("fall outside", seen[5].split("- calls")[1].split("Revise this list")[0])      # calls has no violations, so nothing is written
        # The first label extraction (no prior) keeps its prompts word for word: none of these passages
        seen.clear(); answers[:] = list(self.ANSWERS)
        graph_schema.suggest(_client(reply), ["sample text"], examples=False)
        for phrase in ("each type:", "not in the list", "fall outside", "most common endpoint pairs", "declared too narrowly"):
            self.assertNotIn(phrase, seen[3]); self.assertNotIn(phrase, seen[5])
        self.assertEqual((graph_schema._prior_types_text(None), graph_schema._prior_predicates_text(None)), ("", ""))
        # An old account (no health-check numbers) works as before: only the original sentences
        seen.clear(); answers[:] = list(self.ANSWERS)
        graph_schema.suggest(_client(reply), ["sample text"], examples=False, prior=self.PRIOR)
        self.assertNotIn("each type:", seen[3]); self.assertNotIn("fall outside", seen[5]); self.assertNotIn("declared too narrowly", seen[5])
        self.assertIn("used by 60 edges", seen[5])

    def _state(self, tmp: str, config: dict):
        from kb_pipeline import discovery

        state = Path(tmp) / "s.db"
        db.init_db(state)
        with db.connect(state) as con:
            con.execute("INSERT INTO kb_sources(kb_id, collection, source_root, status, first_seen_at, last_seen_at, config_json) "
                        "VALUES ('kb_x', 'kb_x', '库', 'active', 1, 1, ?)", (json.dumps(config, ensure_ascii=False),))
            con.commit()
        settings = SimpleNamespace(state_db=state, mirror_root=Path(tmp), graph_llm_timeout_seconds=60)
        source = discovery.build_source(Path(tmp), "库", config, kb_id="kb_x", collection="kb_x")
        return settings, source

    def test_suggest_schema_version_adopts_records_account_and_revises_from_prior(self) -> None:
        from kb_pipeline import discovery
        from kb_pipeline.graph import schema_flow

        with tempfile.TemporaryDirectory() as tmp:
            settings, source = self._state(tmp, {"graph_enabled": True})
            with db.connect(settings.state_db) as con:
                self.assertTrue(schema_flow.schema_missing(discovery.get_config(con, "kb_x")))
            first = schema_flow.suggest_schema_version(settings, source, origin=schema_flow.ORIGIN_BLANK, adopt=True, use_prior=False,
                                                       client=_client(list(self.ANSWERS)), docs=["sample"], sample={"chunks_total": 3, "documents": 1})
            with db.connect(settings.state_db) as con:
                cfg = discovery.get_config(con, "kb_x")
            self.assertEqual((first["adopted"], first["origin"], first["prior_id"]), (True, "auto_blank", None))
            self.assertEqual(cfg["graph_schema_active"], first["version_id"])
            self.assertEqual((cfg["graph_entity_types"], cfg["graph_language"]), (["role", "skill", "standard"], "Chinese"))   # takes effect directly
            self.assertEqual(cfg["graph_predicates"][0]["name"], "requires_skill")
            entry = cfg["graph_schema_versions"][0]
            self.assertEqual((entry["origin"], entry["prior_id"], entry["sampled"], entry["documents"]), ("auto_blank", None, 1, 1))
            self.assertFalse(schema_flow.schema_missing(cfg))
            # The graph build writes the account at the end
            with db.connect(settings.state_db) as con:
                self.assertTrue(schema_flow.record_schema_observation(con, "kb_x", version_id=first["version_id"], observed=self.PRIOR["observed"], graph_version="g1"))
                self.assertFalse(schema_flow.record_schema_observation(con, "kb_x", version_id="nope", observed={}, graph_version="g1"))
                acct = schema_flow.active_schema_entry(discovery.get_config(con, "kb_x"))["observed"]
            self.assertEqual((acct["graph_version"], acct["edges"]), ("g1", {"requires_skill": 400, "calls": 60}))
            self.assertIsNotNone(acct.get("at"))
            self.assertEqual(schema_flow.active_schema_entry_for(settings, "kb_x")["id"], first["version_id"])
            self.assertIsNone(schema_flow.active_schema_entry_for(settings, "kb_missing"))
            # With a version present there is no automatic extraction
            same, info = schema_flow.ensure_schema_before_build(settings, source)
            self.assertIs(same, source); self.assertIsNone(info)
            # Re-extraction before rebuild: based on the current version + account; calls, confirmed in the account, is
            # kept by the rule (the model did not mention it)
            second = schema_flow.suggest_schema_version(settings, source, origin=schema_flow.ORIGIN_REBUILD, adopt=True, use_prior=True,
                                                        client=_client(list(self.ANSWERS)), docs=["sample"])
            self.assertEqual(second["prior_id"], first["version_id"])
            # entity -> standard, confirmed in the account, is added back to the targets by the rule; calls is absent from
            # the previous version's predicate table and cannot be restored (restoration follows the previous version only)
            self.assertEqual(second["guard"], {"kept_predicates": [], "widened": ["requires_skill.target_parents+standard"]})
            with db.connect(settings.state_db) as con:
                cfg = discovery.get_config(con, "kb_x")
            self.assertEqual(cfg["graph_schema_active"], second["version_id"])
            self.assertNotEqual(second["version_id"], first["version_id"])
            self.assertEqual([(p["name"], p["target_parents"]) for p in cfg["graph_predicates"]], [("requires_skill", ["property", "standard"])])
            # The account passes down with the version: the new version carries the previous one's account (marked
            # inherited_from), so merging can release pairs on the very first build
            inherited = cfg["graph_schema_versions"][0]["observed"]
            self.assertEqual((inherited["inherited_from"], inherited["edges"]), (first["version_id"], {"requires_skill": 400, "calls": 60}))
            self.assertEqual(schema_flow.active_schema_entry(cfg)["observed"]["confirmed"][0]["predicate"], "requires_skill")
            self.assertEqual([v["id"] for v in cfg["graph_schema_versions"]], [second["version_id"], first["version_id"]])
            # With the switch off there is no re-extraction
            with db.connect(settings.state_db) as con:
                discovery.set_config(con, "kb_x", {"graph_rebuild_resuggest": False}); con.commit()
            src3, info3 = schema_flow.resuggest_for_rebuild(settings, source)
            self.assertEqual((src3 is source, info3), (True, {"skipped": "disabled"}))
            # 2026-09-29 audit: one baseline is re-suggested only once. After a failed full build the baseline is
            # unchanged; re-suggesting every round changes the fingerprint every round, invalidates the whole
            # extraction cache and soon pushes the versions saved by hand out of the ring
            with db.connect(settings.state_db) as con:
                discovery.set_config(con, "kb_x", {"graph_rebuild_resuggest": None}); con.commit()
            calls: list[str] = []

            def fake_suggest(settings_, source_, **kw):
                calls.append(kw["origin"])
                return {"version_id": f"r{len(calls)}", "origin": kw["origin"]}

            from unittest import mock
            with mock.patch.object(schema_flow, "suggest_schema_version", side_effect=fake_suggest), \
                    mock.patch.object(schema_flow, "reload_source", side_effect=lambda s, src: src):
                _, once = schema_flow.resuggest_for_rebuild(settings, source, baseline_version="g-full-1")
                _, again = schema_flow.resuggest_for_rebuild(settings, source, baseline_version="g-full-1")
                _, later = schema_flow.resuggest_for_rebuild(settings, source, baseline_version="g-full-2")
                _, forced = schema_flow.resuggest_for_rebuild(settings, source)
            self.assertEqual(once["version_id"], "r1")
            self.assertEqual(again, {"skipped": "already_resuggested_for_baseline", "baseline": "g-full-1", "version_id": "r1"})
            self.assertEqual((later["version_id"], forced["version_id"]), ("r2", "r3"))      # a new baseline, or none (forced by the operator), re-suggests as usual
            self.assertEqual(len(calls), 3)
            cli_src = (Path(__file__).resolve().parents[1] / "kb_pipeline" / "cli.py").read_text(encoding="utf-8")
            self.assertIn('baseline_version=decision.get("baseline_graph_version")', cli_src)
            # A manual extraction (console) enters the ring but does not take effect
            manual = schema_flow.suggest_schema_version(settings, source, origin=schema_flow.ORIGIN_MANUAL, adopt=False,
                                                        client=_client(list(self.ANSWERS)), docs=["sample"])
            with db.connect(settings.state_db) as con:
                cfg = discovery.get_config(con, "kb_x")
            self.assertEqual(cfg["graph_schema_active"], second["version_id"])
            self.assertEqual(cfg["graph_schema_versions"][0]["id"], manual["version_id"])
            self.assertEqual(cfg["graph_schema_versions"][0]["origin"], "manual")

    def test_blank_graph_runs_auto_suggest_then_reloads_source(self) -> None:
        from unittest import mock

        from kb_pipeline import discovery
        from kb_pipeline.graph import schema_flow

        with tempfile.TemporaryDirectory() as tmp:
            settings, source = self._state(tmp, {"graph_enabled": True})
            calls = []

            def fake(settings_, source_, **kw):
                calls.append(kw)
                with db.connect(settings_.state_db) as con:
                    discovery.set_config(con, source_.kb_id, {"graph_entity_types": ["role"], "graph_schema_active": "v9",
                                                              "graph_schema_versions": [{"id": "v9", "entity_types": ["role"]}]})
                    con.commit()
                return {"version_id": "v9", "origin": kw["origin"]}

            with mock.patch.object(schema_flow, "suggest_schema_version", side_effect=fake):
                reloaded, info = schema_flow.ensure_schema_before_build(settings, source)
            self.assertEqual((calls[0]["origin"], calls[0]["adopt"], calls[0]["use_prior"]), ("auto_blank", True, False))
            self.assertEqual((reloaded.graph_entity_types, info["version_id"]), (("role",), "v9"))
        # A KB without a version is left to the graph build itself on the rebuild path; no re-extraction here
        with tempfile.TemporaryDirectory() as tmp2:
            settings2, source2 = self._state(tmp2, {"graph_enabled": True})
            self.assertEqual(schema_flow.resuggest_for_rebuild(settings2, source2)[1], {"skipped": "no_schema"})

    def test_blank_graph_adopts_an_unsaved_version_and_waits_for_a_running_suggest(self) -> None:
        """2026-09-09 02:09 kb_001: the timer's automatic extraction and the user's manual click were 20 seconds
        apart and each extracted an identical version. Now the extraction is marked throughout: the build waits
        when it sees the mark, and an extracted but unsaved version in the ring is adopted directly instead of
        extracting again."""
        from unittest import mock

        from kb_pipeline import discovery
        from kb_pipeline.graph import schema_flow

        with tempfile.TemporaryDirectory() as tmp:
            settings, source = self._state(tmp, {"graph_enabled": True})
            seen: list[bool] = []
            real_set = discovery.set_config

            def spy(con, kb_id, updates):
                seen.append(schema_flow.suggest_in_progress(settings, "kb_x") is not None)   # the mark is still there when the version lands
                return real_set(con, kb_id, updates)

            with mock.patch.object(discovery, "set_config", side_effect=spy):
                manual = schema_flow.suggest_schema_version(settings, source, origin=schema_flow.ORIGIN_MANUAL, adopt=False,
                                                            client=_client(list(self.ANSWERS)), docs=["sample"], sample={"chunks_total": 3, "documents": 1})
            self.assertEqual(seen, [True])
            self.assertIsNone(schema_flow.suggest_in_progress(settings, "kb_x"))            # released only after landing
            with mock.patch.object(schema_flow, "suggest_schema_version", side_effect=AssertionError("环里已有版本,不该再抽")):
                reloaded, info = schema_flow.ensure_schema_before_build(settings, source)
            self.assertEqual((info["version_id"], info["adopted_existing"], info["origin"]), (manual["version_id"], True, "manual"))
            with db.connect(settings.state_db) as con:
                cfg = discovery.get_config(con, "kb_x")
            self.assertEqual(cfg["graph_schema_active"], manual["version_id"])
            self.assertEqual(cfg["graph_entity_types"], ["role", "skill", "standard"])
            self.assertEqual(reloaded.graph_entity_types, ("role", "skill", "standard"))
            # A version already in effect: nothing else is adopted, nothing extracted
            self.assertIsNone(schema_flow.adopt_latest_schema_version(settings, source))
        # The console is extracting (mark present): wait for the mark to disappear, then adopt the version it extracted
        with tempfile.TemporaryDirectory() as tmp2:
            settings, source = self._state(tmp2, {"graph_enabled": True})
            token = schema_flow.claim_suggest(settings, "kb_x", "manual")
            self.assertTrue(token)
            self.assertEqual(schema_flow.suggest_in_progress(settings, "kb_x")["origin"], "manual")
            # Claiming is atomic (Codex review F06): a second one cannot get in; someone else's token cannot release it;
            # extracting again while holding the mark is refused outright
            self.assertIsNone(schema_flow.claim_suggest(settings, "kb_x", "auto_blank"))
            self.assertFalse(schema_flow.release_suggest(settings, "kb_x", "not-mine"))
            self.assertIsNotNone(schema_flow.suggest_in_progress(settings, "kb_x"))
            with self.assertRaises(schema_flow.SuggestBusy):
                schema_flow.suggest_schema_version(settings, source, origin=schema_flow.ORIGIN_MANUAL, adopt=False,
                                                   client=_client(list(self.ANSWERS)), docs=["sample"])

            def console_finishes(_seconds):
                self.assertTrue(schema_flow.release_suggest(settings, "kb_x", token))          # the console's extraction finishes: release and land the version
                schema_flow.suggest_schema_version(settings, source, origin=schema_flow.ORIGIN_MANUAL, adopt=False,
                                                   client=_client(list(self.ANSWERS)), docs=["sample"])

            with mock.patch.object(schema_flow.time, "sleep", side_effect=console_finishes) as sleep:
                with mock.patch.object(schema_flow, "suggest_schema_version", wraps=schema_flow.suggest_schema_version) as spy:
                    reloaded, info = schema_flow.ensure_schema_before_build(settings, source)
            self.assertEqual((sleep.call_count, spy.call_count), (1, 1))                    # only the console's one extraction
            self.assertTrue(info["adopted_existing"])
            self.assertEqual(reloaded.graph_entity_types, ("role", "skill", "standard"))
            # A stale mark left by a killed process does not count as extracting and cannot block a new claim
            with db.connect(settings.state_db) as con:
                db.set_app_config(con, schema_flow.SUGGEST_MARK_PREFIX + "kb_x", {"origin": "manual", "started_at": 1, "token": "dead"})
                con.execute("UPDATE app_config SET updated_at = 1 WHERE key = ?", (schema_flow.SUGGEST_MARK_PREFIX + "kb_x",)); con.commit()
            self.assertIsNone(schema_flow.suggest_in_progress(settings, "kb_x"))
            fresh = schema_flow.claim_suggest(settings, "kb_x", "manual")
            self.assertTrue(fresh)
            self.assertTrue(schema_flow.release_suggest(settings, "kb_x", fresh))
        # An error midway through extraction must clear the mark too
        with tempfile.TemporaryDirectory() as tmp3:
            settings, source = self._state(tmp3, {"graph_enabled": True})
            from kb_pipeline.graph import schema as graph_schema
            with mock.patch.object(graph_schema, "suggest", side_effect=RuntimeError("boom")):
                with self.assertRaises(RuntimeError):
                    schema_flow.suggest_schema_version(settings, source, origin=schema_flow.ORIGIN_MANUAL, adopt=False,
                                                       client=_client([]), docs=["sample"])
            self.assertIsNone(schema_flow.suggest_in_progress(settings, "kb_x"))
        # On the console side: refuse to extract again when the mark is seen
        svc = (Path(__file__).resolve().parents[1] / "kb_server" / "service.py").read_text(encoding="utf-8")
        self.assertIn("if suggest_in_progress(cfg, kb_id) is not None:", svc)

    def test_default_graph_llm_fallback(self) -> None:
        from kb_pipeline.graph.build import DEFAULT_GRAPH_LLM, llm_ready, resolve_llm_specs

        with tempfile.TemporaryDirectory() as tmp:
            settings, source = self._state(tmp, {"graph_enabled": True})
            self.assertIn("selected", llm_ready(settings, source) or "")
            with db.connect(settings.state_db) as con:
                db.upsert_llm(con, name=DEFAULT_GRAPH_LLM, base_url="http://x/v1", api_key="k", model_id="deepseek-v4-flash")
                con.commit()
            specs = resolve_llm_specs(settings, source, ("extract", "summarize", "tune"))
            self.assertEqual({s.name for s in specs.values()}, {DEFAULT_GRAPH_LLM})
            self.assertIsNone(llm_ready(settings, source))
            with db.connect(settings.state_db) as con:
                db.upsert_llm(con, name="Other", base_url="http://x/v1", api_key="k", model_id="o")
                from kb_pipeline import discovery
                discovery.set_config(con, "kb_x", {"graph_llm": {"extract": "Other"}}); con.commit()
            from kb_pipeline import discovery
            source = discovery.build_source(Path(tmp), "库", {"graph_enabled": True, "graph_llm": {"extract": "Other"}}, kb_id="kb_x", collection="kb_x")
            specs = resolve_llm_specs(settings, source, ("extract", "summarize"))
            self.assertEqual((specs["extract"].name, specs["summarize"].name), ("Other", DEFAULT_GRAPH_LLM))   # a selected slot uses its selection, an unselected one uses the default

    def test_a_lone_registered_model_is_the_default_for_every_slot(self) -> None:
        """2026-09-12 the user kept only one model (not named DEFAULT_GRAPH_LLM): unselected slots use it rather than
        requiring a per-KB selection; only once a second one is registered without a default name does it count as
        unselected."""
        from kb_pipeline.graph.build import default_graph_llm, llm_ready, resolve_llm_specs

        with tempfile.TemporaryDirectory() as tmp:
            settings, source = self._state(tmp, {"graph_enabled": True})
            with db.connect(settings.state_db) as con:
                self.assertIsNone(default_graph_llm(con))
                db.upsert_llm(con, name="DeepSeek Flash", base_url="https://api.deepseek.com", api_key="k", model_id="deepseek-flash")
                con.commit()
                self.assertEqual(default_graph_llm(con), "DeepSeek Flash")
            specs = resolve_llm_specs(settings, source, ("extract", "summarize", "tune"))
            self.assertEqual({s.name for s in specs.values()}, {"DeepSeek Flash"})
            self.assertIsNone(llm_ready(settings, source))
            with db.connect(settings.state_db) as con:
                db.upsert_llm(con, name="Second", base_url="http://x/v1", api_key="k", model_id="o")
                con.commit()
                self.assertIsNone(default_graph_llm(con))
            self.assertIn("selected", llm_ready(settings, source) or "")

    def test_check_rebuild_resuggests_before_threshold_rebuild(self) -> None:
        import argparse
        from unittest import mock

        from kb_pipeline import cli
        from kb_pipeline.graph import schema_flow

        with tempfile.TemporaryDirectory() as tmp:
            src = KBSource(kb_id="kb_x", collection="kb_x", source_root="r", source_type="local", max_tokens=400, overlap_tokens=80, graph_enabled=True)
            src2 = KBSource(kb_id="kb_x", collection="kb_x", source_root="r", source_type="local", max_tokens=400, overlap_tokens=80, graph_enabled=True,
                            graph_entity_types=("role",))
            stub = SimpleNamespace(state_db=Path(tmp) / "s.db", sources={"kb_x": src})
            order: list[str] = []
            base = dict(env_file=None, source=None, collection=None, all=False)

            def resuggest(settings, source, **kwargs):
                order.append("resuggest"); return src2, {"version_id": "v2"}

            def build(settings, **kw):
                order.append("build:" + ("new" if kw["source"] is src2 else "old")); return {"ok": True}

            def run(decision, **extra):
                order.clear()
                args = argparse.Namespace(graph_command="check-rebuild", execute=True, dry_run=False, force_full=False, **base, **extra)
                with mock.patch.object(cli, "load_settings", return_value=stub), \
                        mock.patch.object(cli, "evaluate_rebuild", return_value=dict(decision)), \
                        mock.patch.object(cli, "llm_ready", return_value=None), \
                        mock.patch.object(cli, "_rebuild_blocked", return_value=None), \
                        mock.patch.object(cli, "build_graph", side_effect=build), \
                        mock.patch.object(schema_flow, "resuggest_for_rebuild", side_effect=resuggest), \
                        mock.patch.object(cli.db, "record_graph_check", return_value=None):
                    self.assertEqual(cli.cmd_graph(args), 0)
                return list(order)

            # Threshold reached: re-extract first, then build with the new config
            self.assertEqual(run({"source": "kb_x", "due": True, "reason": "interval"}), ["resuggest", "build:new"])
            # First build: no re-extraction
            self.assertEqual(run({"source": "kb_x", "due": True, "reason": "no_successful_build"}), ["build:old"])
            # --force-full: skips the conditions, same path
            args = argparse.Namespace(graph_command="check-rebuild", execute=True, dry_run=False, force_full=True, **base)
            order.clear()
            with mock.patch.object(cli, "load_settings", return_value=stub), \
                    mock.patch.object(cli, "evaluate_rebuild", side_effect=AssertionError("forced 不该评估条件")), \
                    mock.patch.object(cli, "llm_ready", return_value=None), \
                    mock.patch.object(cli, "_rebuild_blocked", return_value=None), \
                    mock.patch.object(cli, "build_graph", side_effect=build), \
                    mock.patch.object(schema_flow, "resuggest_for_rebuild", side_effect=resuggest), \
                    mock.patch.object(cli.db, "record_graph_check", return_value=None):
                self.assertEqual(cli.cmd_graph(args), 0)
            self.assertEqual(order, ["resuggest", "build:new"])
            with mock.patch.object(cli, "load_settings", return_value=stub):
                with self.assertRaises(ValueError):       # --force-full without --execute is not accepted
                    cli.cmd_graph(argparse.Namespace(graph_command="check-rebuild", execute=False, dry_run=False, force_full=True, **base))
            # A failed re-extraction does not block the rebuild: build on the current version
            order.clear()
            args = argparse.Namespace(graph_command="check-rebuild", execute=True, dry_run=False, force_full=False, **base)
            with mock.patch.object(cli, "load_settings", return_value=stub), \
                    mock.patch.object(cli, "evaluate_rebuild", return_value={"source": "kb_x", "due": True, "reason": "interval"}), \
                    mock.patch.object(cli, "llm_ready", return_value=None), \
                    mock.patch.object(cli, "_rebuild_blocked", return_value=None), \
                    mock.patch.object(cli, "build_graph", side_effect=build), \
                    mock.patch.object(schema_flow, "resuggest_for_rebuild", side_effect=RuntimeError("llm down")), \
                    mock.patch.object(cli.db, "record_graph_check", return_value=None):
                self.assertEqual(cli.cmd_graph(args), 0)
            self.assertEqual(order, ["build:old"])

    def test_suggest_claim_is_renewed_and_a_lost_claim_cannot_publish(self) -> None:
        """Codex re-review N08: the claim lasted only 12 minutes while the extraction flow is several model calls, so a
        slightly slow run expired; after expiry someone else claimed it and the old run still published when done.
        Now the claim is renewed during model calls and the token is checked before publishing; on a mismatch only
        a log line is left."""
        from kb_pipeline import discovery
        from kb_pipeline.graph import schema_flow

        with tempfile.TemporaryDirectory() as tmp:
            settings, source = self._state(tmp, {"graph_enabled": True})
            key = schema_flow.SUGGEST_MARK_PREFIX + "kb_x"
            token = schema_flow.claim_suggest(settings, "kb_x", "manual")
            with db.connect(settings.state_db) as con:
                con.execute("UPDATE app_config SET updated_at = 1, value = json_set(value, '$.started_at', 1) WHERE key = ?", (key,))
                con.commit()
            self.assertIsNone(schema_flow.suggest_in_progress(settings, "kb_x"))          # expired
            self.assertFalse(schema_flow.renew_suggest(settings, "kb_x", "not-mine"))
            self.assertTrue(schema_flow.renew_suggest(settings, "kb_x", token))
            self.assertEqual(schema_flow.suggest_owner(settings, "kb_x"), token)           # renewed
            self.assertTrue(schema_flow.release_suggest(settings, "kb_x", token))
            # The mark expires mid-extraction and the other side claims it: this run's result must not be published,
            # the ring stays empty, and the other party's mark is left as is
            answers = list(self.ANSWERS)
            stolen: dict[str, str] = {}

            def chat(messages):
                if not stolen:
                    with db.connect(settings.state_db) as con:
                        con.execute("UPDATE app_config SET updated_at = 1 WHERE key = ?", (key,))
                        con.commit()
                    stolen["token"] = schema_flow.claim_suggest(settings, "kb_x", "auto_rebuild") or ""
                return answers.pop(0)

            with self.assertRaises(schema_flow.SuggestBusy) as ctx:
                schema_flow.suggest_schema_version(settings, source, origin=schema_flow.ORIGIN_MANUAL, adopt=True,
                                                   client=_client(chat), docs=["sample"], sample={"chunks_total": 3, "documents": 1})
            self.assertEqual(str(ctx.exception), schema_flow.SUGGEST_SUPERSEDED_MESSAGE)
            self.assertTrue(stolen["token"])
            with db.connect(settings.state_db) as con:
                cfg = discovery.get_config(con, "kb_x")
            self.assertEqual((cfg.get("graph_schema_versions") or [], cfg.get("graph_schema_active")), ([], None))
            self.assertEqual(schema_flow.suggest_owner(settings, "kb_x"), stolen["token"])
