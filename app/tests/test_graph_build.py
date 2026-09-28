"""Graph build flow: phases and resume, fingerprints, locks, incremental append, artifacts and records."""
from __future__ import annotations

import json
import os
import re
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from kb_pipeline import db
from kb_pipeline.graph import resolution
from kb_pipeline.graph.build import graph_paths
from kb_pipeline.graph.llm import ChatClient, LLMCache, LLMSpec
from kb_pipeline.graph.units import ChunkRef, Unit
from kb_pipeline.localfs.scanner import stable_int
from kb_pipeline.models import KBSource, SourceFile
from kb_pipeline.vector.qdrant import activate_graph_aliases, restore_graph_aliases

from _support import _CodexAudit20260906TestsSupport, _CodexFinalTestsSupport, _block, _bundle_dir, _repo_file


class GraphPathRegressionTests(unittest.TestCase):
    def test_future_build_artifacts_are_versioned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = SimpleNamespace(graph_work_dir=root)
            source = KBSource(
                kb_id="work_product",
                collection="kb_product",
                source_root="产品资料",
                source_type="local_mirror",
                max_tokens=400,
                overlap_tokens=80,
            )
            legacy = graph_paths(settings, source)
            versioned = graph_paths(settings, source, "product-20260713-120000-test")
        self.assertEqual(legacy.output_dir, root / "work" / "product")
        self.assertEqual(
            versioned.output_dir,
            root / "work" / "product" / "product-20260713-120000-test",
        )
        self.assertEqual(versioned.units_file, versioned.work_dir / "units.jsonl")
        self.assertEqual(versioned.cache_file, root / "cache" / "product.sqlite")

    def test_graph_alias_switch_and_rollback_are_atomic_batches(self) -> None:
        old_version = "product-20260627-223336"
        new_version = "product-20260713-120000-test"

        class FakeQdrant:
            def __init__(self):
                self.aliases = {
                    "graph_product_entity": f"graph_product_entity__{old_version}",
                    "graph_product_relation": f"graph_product_relation__{old_version}",
                }
                self.collections = {
                    *self.aliases.values(),
                    f"graph_product_entity__{new_version}",
                    f"graph_product_relation__{new_version}",
                }
                self.batch_sizes: list[int] = []

            def get_aliases(self):
                return SimpleNamespace(
                    aliases=[
                        SimpleNamespace(alias_name=alias, collection_name=target)
                        for alias, target in self.aliases.items()
                    ]
                )

            def get_collections(self):
                return SimpleNamespace(
                    collections=[SimpleNamespace(name=name) for name in self.collections]
                )

            def update_collection_aliases(self, operations):
                self.batch_sizes.append(len(operations))
                for operation in operations:
                    delete_alias = getattr(operation, "delete_alias", None)
                    create_alias = getattr(operation, "create_alias", None)
                    if delete_alias is not None:
                        self.aliases.pop(delete_alias.alias_name, None)
                    if create_alias is not None:
                        self.aliases[create_alias.alias_name] = create_alias.collection_name

        q = FakeQdrant()
        activated = activate_graph_aliases(
            q,
            source_collection="kb_product",
            graph_version=new_version,
        )
        self.assertEqual(set(q.aliases.values()), {
            f"graph_product_entity__{new_version}",
            f"graph_product_relation__{new_version}",
        })
        self.assertTrue(restore_graph_aliases(q, activated["previous"]))
        self.assertEqual(set(q.aliases.values()), {
            f"graph_product_entity__{old_version}",
            f"graph_product_relation__{old_version}",
        })
        self.assertEqual(q.batch_sizes, [4, 4])


class RebuildPolicyTests(unittest.TestCase):
    """The combinations of the automatic rebuild policy had zero coverage before -- they decide when hours of
    LLM spend are committed."""

    def _prepare(self, tmp: str, *, policy: dict, chunks: list[dict], baseline: list[str] | None,
                 finished_days_ago: float = 10.0, content_hash: str | None = None):
        from unittest import mock

        from kb_pipeline import discovery
        from kb_pipeline.graph import build as build_mod

        root = Path(tmp); (root / "库").mkdir()
        state = root / "s.db"; db.init_db(state)
        with db.connect(state) as con:
            src, _ = discovery.enroll(con, root, "库")
            discovery.set_config(con, src.kb_id, {"graph_enabled": True, **policy})
            if baseline is not None:
                build_id = db.begin_graph_build(con, source_key=src.kb_id, kb_id=src.kb_id,
                                                source_collection=src.collection, graph_version="v1")
                db.finish_graph_build(con, build_id, status="done", source_content_hash=content_hash or "h0")
                con.execute("UPDATE graph_builds SET finished_at = ? WHERE graph_build_id = ?",
                            (int(time.time() - finished_days_ago * 86400), build_id))
                db.replace_graph_build_chunks(con, build_id, [{"point_id": pid} for pid in baseline])
            con.commit()
        settings = SimpleNamespace(state_db=state, mirror_root=root,
                                   qdrant_url="http://q", qdrant_api_key="")
        source = discovery.enrolled_sources(state, root)[src.kb_id]
        patcher = mock.patch.object(build_mod, "active_source_chunks", return_value=chunks)
        patcher.start(); self.addCleanup(patcher.stop)
        return build_mod, settings, src.kb_id, source

    def test_interval_and_ratio_combinations(self) -> None:
        base = [{"point_id": f"p{i}", "chunk_uid": f"u{i}", "doc_id": "d", "content_version": "v"}
                for i in range(10)]
        with tempfile.TemporaryDirectory() as tmp:
            # Interval elapsed + corpus changed -> should rebuild
            mod, settings, key, source = self._prepare(
                tmp, policy={"graph_rebuild_interval": "7d"},
                chunks=base + [{"point_id": "new", "chunk_uid": "un", "doc_id": "d", "content_version": "v2"}],
                baseline=[c["point_id"] for c in base], finished_days_ago=10)
            decision = mod.evaluate_rebuild(settings, source_key=key, source=source)
            self.assertTrue(decision["due"], decision)

    def test_interval_due_but_corpus_unchanged_is_skipped(self) -> None:
        chunks = [{"point_id": "p1", "chunk_uid": "u1", "doc_id": "d", "content_version": "v"}]
        with tempfile.TemporaryDirectory() as tmp:
            from kb_pipeline.graph.build import source_snapshot_hash

            mod, settings, key, source = self._prepare(
                tmp, policy={"graph_rebuild_interval": "1d"}, chunks=chunks,
                baseline=["p1"], finished_days_ago=30,
                content_hash=source_snapshot_hash(chunks))
            decision = mod.evaluate_rebuild(settings, source_key=key, source=source)
            self.assertFalse(decision["due"])      # not a character of the corpus changed; no need to burn hours of LLM again
            self.assertEqual(decision.get("skipped_reason"), "source_unchanged_since_last_build")

    def test_no_policy_means_first_build_only(self) -> None:
        chunks = [{"point_id": "p1", "chunk_uid": "u1", "doc_id": "d", "content_version": "v"}]
        with tempfile.TemporaryDirectory() as tmp:
            mod, settings, key, source = self._prepare(tmp, policy={}, chunks=chunks, baseline=["p1"])
            decision = mod.evaluate_rebuild(settings, source_key=key, source=source)
            self.assertFalse(decision["due"])
            self.assertEqual(decision["reason"], "no_policy")

    def test_invalid_operator_falls_back_to_or(self) -> None:
        """Health check B12: an operator other than or / and is treated as or and logged, so one KB does not
        fail the whole check-rebuild round."""
        base = [{"point_id": f"p{i}", "chunk_uid": f"u{i}", "doc_id": "d", "content_version": "v"}
                for i in range(10)]
        with tempfile.TemporaryDirectory() as tmp:
            mod, settings, key, source = self._prepare(
                tmp, policy={"graph_rebuild_interval": "7d", "graph_rebuild_operator": "xor"},
                chunks=base + [{"point_id": "new", "chunk_uid": "un", "doc_id": "d", "content_version": "v2"}],
                baseline=[c["point_id"] for c in base], finished_days_ago=10)
            decision = mod.evaluate_rebuild(settings, source_key=key, source=source)
        self.assertTrue(decision["due"], decision)
        self.assertEqual(decision["operator"], "or")

    def test_new_chunk_count_threshold(self) -> None:
        """Triggered by an absolute count. A ratio is not comparable between KBs of very different size --
        20% growth on a 1496-file KB is hundreds of chunks, 20% on a 28-file KB is just a few."""
        base = [{"point_id": f"p{i}", "chunk_uid": f"u{i}", "doc_id": "d", "content_version": "v"}
                for i in range(10)]
        baseline = [c["point_id"] for c in base]

        def decide(new_count: int) -> dict:
            fresh = [{"point_id": f"n{i}", "chunk_uid": f"un{i}", "doc_id": "d",
                      "content_version": "v2"} for i in range(new_count)]
            with tempfile.TemporaryDirectory() as tmp:
                mod, settings, key, source = self._prepare(
                    tmp, policy={"graph_rebuild_new_chunk_count": 5},
                    chunks=base + fresh, baseline=baseline)
                return mod.evaluate_rebuild(settings, source_key=key, source=source)

        self.assertFalse(decide(4)["due"])          # one short does not count
        due = decide(5)
        self.assertTrue(due["due"], due)            # exactly at the threshold counts
        cond = [c for c in due["conditions"] if c["name"] == "new_chunk_count"]
        self.assertEqual(len(cond), 1, due)
        self.assertEqual(cond[0]["threshold"], 5)
        self.assertEqual(cond[0]["new_chunks"], 5)

    def test_percent_over_100_is_clamped_at_save_and_at_read(self) -> None:
        """Clamped in both places: at save time the stored value is rewritten (otherwise the UI would show 150
        while the behaviour follows 100), and at read time it is caught again (the config can also be changed
        through the CLI or by writing the database directly)."""
        from kb_pipeline import discovery
        from kb_server.service import normalize_graph_updates

        updates = {"graph_rebuild_new_chunk_pct": "150%"}
        self.assertEqual(
            normalize_graph_updates(updates)["graph_rebuild_new_chunk_pct"], "100%")
        # Values within range are untouched
        self.assertEqual(
            normalize_graph_updates({"graph_rebuild_new_chunk_pct": "20%"})
            ["graph_rebuild_new_chunk_pct"], "20%")

        with tempfile.TemporaryDirectory() as tmp:
            src = discovery.build_source(Path(tmp), "库",
                                         {"graph_rebuild_new_chunk_pct": "300%"}, kb_id="kb_001")
            self.assertEqual(src.graph_rebuild_policy.new_chunk_ratio, 1.0)

    def test_two_new_chunk_units_are_mutually_exclusive(self) -> None:
        """If both units of the same condition were in effect at once, the operator's semantics would be
        undefined. The console dropdown is already either-or; this blocks direct API calls that bypass the
        UI."""
        from kb_server import service

        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"; db.init_db(state)
            limits = {"embedding_max_model_len": 4096, "effective_max_model_len": 4096,
                      "max_tokens_cap": 3276, "max_tokens_min": 128, "cap_ratio": 0.8,
                      "overlap_rule": "", "live": False}
            base = {"max_tokens": 400, "overlap_tokens": 80,
                    "graph_chunk_size": 1200, "graph_chunk_overlap": 100}
            with db.connect(state) as con:
                both = service._validate_config(
                    con, {**base, "graph_rebuild_new_chunk_pct": "20%",
                          "graph_rebuild_new_chunk_count": 300}, {}, limits)
                self.assertTrue(any("either a percentage or a chunk count" in e for e in both), both)
                # Each on its own is valid
                for one in ({"graph_rebuild_new_chunk_pct": "20%"},
                            {"graph_rebuild_new_chunk_count": 300}):
                    self.assertEqual(service._validate_config(con, {**base, **one}, {}, limits), [])
                bad = service._validate_config(
                    con, {**base, "graph_rebuild_new_chunk_count": 0}, {}, limits)
                self.assertTrue(any("new chunk count" in e for e in bad), bad)

    def test_never_built_is_always_due(self) -> None:
        chunks = [{"point_id": "p1", "chunk_uid": "u1", "doc_id": "d", "content_version": "v"}]
        with tempfile.TemporaryDirectory() as tmp:
            mod, settings, key, source = self._prepare(tmp, policy={"graph_rebuild_interval": "7d"},
                                                       chunks=chunks, baseline=None)
            decision = mod.evaluate_rebuild(settings, source_key=key, source=source)
            self.assertTrue(decision["due"])       # never built means due


class EmptyGraphSourceTests(unittest.TestCase):
    """A KB with not a single file parsed must not trigger a graph build, nor be fed to GraphRAG."""

    def _settings(self, tmp: Path):
        return SimpleNamespace(
            runtime_dir=tmp / "runtime", state_db=tmp / "state.db",
            graphrag_root=tmp / "graphrag", graph_work_dir=tmp / "work",
        )

    def _source(self):
        return KBSource(
            kb_id="project_materials", collection="kb_project", source_root="项目资料",
            source_type="local_mirror", max_tokens=400, overlap_tokens=80, graph_enabled=True,
        )

    def test_rebuild_policy_reports_no_content_instead_of_due(self) -> None:
        # Otherwise the nightly rebuild check would leave a failed record for every new KB
        from kb_pipeline.graph import build as build_mod

        with tempfile.TemporaryDirectory() as tmp:
            settings = self._settings(Path(tmp))
            db.init_db(settings.state_db)
            verdict = build_mod.evaluate_rebuild(settings, source_key="项目资料", source=self._source())
        self.assertFalse(verdict["due"])
        self.assertEqual(verdict["reason"], "no_active_content")

    def test_prepare_graph_input_refuses_an_empty_source(self) -> None:
        # A manual build bypasses the policy layer, so this must block on its own -- an empty input directory
        # would only make GraphRAG report an error unrelated to the root cause.
        from kb_pipeline.graph import build as build_mod

        with tempfile.TemporaryDirectory() as tmp:
            settings = self._settings(Path(tmp))
            db.init_db(settings.state_db)
            source = self._source()
            paths = build_mod.graph_paths(settings, source, "v1")
            with self.assertRaises(build_mod.NoGraphCorpus) as caught:
                build_mod.prepare_graph_input(settings, source, paths=paths, q=object())
        self.assertIn("no active chunks", str(caught.exception))


class KbIsolationTests(unittest.TestCase):
    """With two knowledge bases side by side, every boundary of the graph build must see only its own KB.

    Cross-KB leakage is invisible: entities from another KB end up in the graph while every count and
    reconciliation still passes. These assertions pin down each boundary.
    """

    def _settings(self, tmp: Path):
        return SimpleNamespace(
            runtime_dir=tmp / "runtime", state_db=tmp / "state.db",
            graphrag_root=tmp / "g", graph_work_dir=tmp / "w", embedding_dim=1024,
        )

    def _seed(self, con, settings, *, collection: str, source_root: str, rel_path: str, text: str):
        from kb_pipeline.localfs.scanner import stable_int
        from kb_pipeline.models import SourceFile, UnifiedChunk

        file_key = stable_int(f"file:{collection}:{rel_path}")
        parent = str(Path(rel_path).parent)
        file = SourceFile(
            kb_id=collection, collection=collection, source_root=source_root,
            source_type="local_mirror", file_key=file_key,
            source_path=f"{source_root}/{rel_path}", rel_path=rel_path,
            filename=Path(rel_path).name, dir="" if parent == "." else parent,
            physical_path="/x", mime_type="application/pdf", size=10, mtime=1,
            checksum=f"c{file_key}",
        )
        db.upsert_file(con, file, status="indexed")
        fid = db.file_id_for(collection, file_key)
        db.replace_chunks(
            con, file_id=fid, collection=collection, content_version="v1",
            chunks=[UnifiedChunk(chunk_uid=f"{collection}:{file_key}:v1:pdf:b0:0",
                                 chunk_index=0, text="x", block=_block("b0", "x"))],
            point_ids=[f"pt-{collection}"],
        )
        return KBSource(kb_id=collection, collection=collection, source_root=source_root,
                        source_type="local_mirror", max_tokens=400, overlap_tokens=80)

    def test_every_graph_boundary_stays_inside_one_kb(self) -> None:
        from kb_pipeline.graph import build as build_mod
        from kb_pipeline.vector.qdrant import graph_collection_name

        with tempfile.TemporaryDirectory() as tmp:
            settings = self._settings(Path(tmp))
            db.init_db(settings.state_db)
            with db.connect(settings.state_db) as con:
                a = self._seed(con, settings, collection="kb_003", source_root="知识库",
                               rel_path="技术/编译原理.pdf", text="A库独有内容")
                b = self._seed(con, settings, collection="kb_004", source_root="产品资料",
                               rel_path="手册/安装.pdf", text="B库独有内容")
                con.commit()

                # Chunk ledger: each sees only its own
                for source in (a, b):
                    refs = db.active_chunk_refs(con, source.collection)
                    self.assertEqual(len(refs), 1)
                    self.assertTrue(refs[0]["chunk_uid"].startswith(source.collection))

                paths_a = build_mod.graph_paths(settings, a, "v1")
                paths_b = build_mod.graph_paths(settings, b, "v1")

            # Graph workspace and cache: one per KB
            self.assertNotEqual(paths_a.work_dir, paths_b.work_dir)
            self.assertNotEqual(paths_a.cache_file, paths_b.cache_file)

            # Qdrant graph collections: names do not collide
            names_a = {graph_collection_name("kb_003", t, "v1") for t in ("entity", "relation")}
            names_b = {graph_collection_name("kb_004", t, "v1") for t in ("entity", "relation")}
            self.assertEqual(names_a & names_b, set())


class GraphCacheFingerprintTests(unittest.TestCase):
    """The "resume build" button is a promise about cost: LLM calls that already ran come back instantly.

    But the cache key is a hash of a single call's input_args, and the exclusion list holds only metrics /
    stream / timeout / base_url / api_base / api_key / drop_params -- model and messages are both in the key.
    Switch the build model, re-extract the labels once, tweak a chunking parameter, and the old cache is all
    useless; clicking "resume build" then is hours of a full rerun while the button says "resume".

    So a config fingerprint is recorded when the build starts, and after a pause it is compared with the
    current config."""

    REPO = Path(__file__).resolve().parents[2]

    # _resolve_graph_llm reads graph_llm from the **database** (by kb_id), not from the KBSource passed in,
    # so the fixture must create real kb_sources rows + a model registry.
    def _prep(self, tmp: str):
        from kb_pipeline import discovery

        root = Path(tmp); (root / "产品资料").mkdir(exist_ok=True)
        state = root / "s.db"
        db.init_db(state)
        with db.connect(state) as con:
            for name, model in (("抽取", "m-extract"), ("摘要", "m-sum"),
                                ("社区", "m-comm"), ("另一个", "m-other")):
                db.upsert_llm(con, name=name, base_url="http://x/v1",
                              api_key="k", model_id=model)
            src, _ = discovery.enroll(con, root, "产品资料")
            con.commit()
        return root, state, src.kb_id

    def _fp(self, root: Path, state: Path, kb_id: str, config: dict | None = None) -> str:
        from kb_pipeline import discovery
        from kb_pipeline.graph.build import graph_cache_fingerprint

        # Write the full set of relevant keys every time: set_config merges, so writing only the difference
        # would let one test case's changes leak into the next.
        full = {
            "graph_llm": {"extract": "抽取", "summarize": "摘要"},
            "graph_entity_types": ["alpha", "beta"],
            "graph_language": "Chinese",
            "graph_predicates": [{"name": "part_of"}],
            "graph_parent_types": {"alpha": "thing"},
            "graph_unit_chunks": 3,
            "graph_max_gleanings": 1,
        }
        full.update(config or {})
        with db.connect(state) as con:
            discovery.set_config(con, kb_id, full)
            con.commit()
            stored = discovery.get_config(con, kb_id)
        source = discovery.build_source(root, "产品资料", stored,
                                        kb_id=kb_id, collection=kb_id)
        return graph_cache_fingerprint(self._settings(state), source)

    def _settings(self, state: Path):
        """The fingerprint hashes settings.yaml, the three upstream prompt constants and the render script --
        pointing at the real repository, these tests also guard that those paths still exist."""
        return SimpleNamespace(state_db=state, runtime_dir=self.REPO / "runtime")

    def test_everything_that_invalidates_the_cache_changes_the_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, state, kb_id = self._prep(tmp)
            base = self._fp(root, state, kb_id)
            for label, cfg in (
                ("换建图模型", {"graph_llm": {"extract": "另一个", "summarize": "摘要"}}),
                ("改类型表", {"graph_entity_types": ["pin", "voltage"]}),
                ("改输出语言", {"graph_language": "English"}),
                ("改谓词表", {"graph_predicates": [{"name": "requires"}]}),
                ("改父类", {"graph_parent_types": {"alpha": "other"}}),
                ("改合并切片数", {"graph_unit_chunks": 2}),
                ("改补漏轮数", {"graph_max_gleanings": 0}),
            ):
                self.assertNotEqual(self._fp(root, state, kb_id, cfg), base,
                                    f"{label} 之后指纹没变")

    def test_the_build_record_carries_the_fingerprint(self) -> None:
        """The fingerprint must be stored **when the build starts**: a pause goes through the exception path,
        and computing it then would hash the already-changed config, so it would always compare equal."""
        source = _repo_file("app/kb_pipeline/graph/build.py")
        begin = source.split("build_id = db.begin_graph_build(", 1)[1].split("\n\n", 1)[0]
        self.assertIn("cache_fingerprint=graph_cache_fingerprint(", begin,
                      "指纹必须在开建时落库:暂停走的是异常路径,那时再算就算的是"
                      "被改过的配置,等于永远相等")
        self.assertIn("ALTER TABLE graph_builds ADD COLUMN cache_fingerprint TEXT",
                      _repo_file("app/kb_pipeline/db.py"))

    def _paused_row(self, root: Path, state: Path, kb_id: str, *,
                    fingerprint: str, corpus: str | None):
        """Create a build record "paused here" and return (con, row) for the server-side verdict."""
        from kb_pipeline import discovery

        con = db.connect(state).__enter__()
        stored = discovery.get_config(con, kb_id)
        source = discovery.build_source(root, "产品资料", stored,
                                        kb_id=kb_id, collection=kb_id)
        bid = db.begin_graph_build(
            con, source_key="产品资料", kb_id=kb_id,
            source_collection=source.collection, graph_version="v-test",
            cache_fingerprint=fingerprint)
        con.execute("UPDATE graph_builds SET status='cancelled', source_content_hash=? "
                    "WHERE graph_build_id=?", (corpus, bid))
        con.commit()
        row = con.execute(
            "SELECT cache_fingerprint, source_content_hash FROM graph_builds "
            "WHERE graph_build_id=?", (bid,)).fetchone()
        return con, row, source

    def _verdict(self, root: Path, state: Path, kb_id: str, *,
                 fingerprint: str, corpus: str | None):
        from unittest import mock

        from kb_server import service

        con, row, _ = self._paused_row(root, state, kb_id,
                                       fingerprint=fingerprint, corpus=corpus)
        try:
            with mock.patch.object(service, "settings", lambda: SimpleNamespace(
                    state_db=state, mirror_root=root,
                    graphrag_root=self.REPO / "graphrag",
                    runtime_dir=self.REPO / "runtime")):
                return service._paused_cache_reuse(con, kb_id, row)
        finally:
            con.close()

    def test_it_does_say_resume_when_nothing_changed(self) -> None:
        """The other half: when neither config nor corpus changed, the verdict must be resumable. Without this
        the mechanism would degrade into "never resumable" -- worse than nothing, since "resume build" would
        never appear again."""
        from kb_pipeline import discovery
        from kb_pipeline.graph.build import source_snapshot_hash

        with tempfile.TemporaryDirectory() as tmp:
            root, state, kb_id = self._prep(tmp)
            fp = self._fp(root, state, kb_id)
            with db.connect(state) as con:
                corpus = source_snapshot_hash(db.active_chunk_refs(con, kb_id))
            out = self._verdict(root, state, kb_id, fingerprint=fp, corpus=corpus)
        self.assertTrue(out["cache_reusable"], out)
        self.assertEqual(out["cache_stale"], "")

    def test_changing_the_tags_flips_it_back_to_a_full_rebuild(self) -> None:
        """The user's own words: once the labels change, it should go back to "build now / rebuild"."""
        from kb_pipeline.graph.build import source_snapshot_hash

        with tempfile.TemporaryDirectory() as tmp:
            root, state, kb_id = self._prep(tmp)
            old_fp = self._fp(root, state, kb_id)
            with db.connect(state) as con:
                corpus = source_snapshot_hash(db.active_chunk_refs(con, kb_id))
            # The record keeps the old fingerprint while the current config has a different set of labels
            self._fp(root, state, kb_id, {"graph_entity_types": ["pin", "voltage"]})
            out = self._verdict(root, state, kb_id, fingerprint=old_fp, corpus=corpus)
        self.assertFalse(out["cache_reusable"])
        self.assertIn("labels", out["cache_stale"])

    def test_an_old_record_without_a_fingerprint_never_promises_resume(self) -> None:
        """This column was added later. Old records do not know which config was used, so no promise can be
        made out of thin air."""
        svc = _repo_file("app/kb_server/service.py")
        body = svc.split("def _paused_cache_reuse", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("if not stored_fp:", body)
        self.assertIn("source_content_hash", body)   # the corpus dimension is compared too
        self.assertIn('"cache_reusable": False', body)   # no promise when it cannot be computed


class GraphPhaseResumeTests(unittest.TestCase):
    """Graph build phase marks + per-phase retries + resuming the same version.

    When an hours-long build stopped at the last step (Neo4j hiccuped), the only option used to be starting
    over: the LLM cache answers instantly, but the write-back and import had to be redone in full. Now every
    heavy phase leaves a mark when it completes, and when neither config nor corpus changed, resuming the
    same version skips it outright.
    """

    def _state(self, tmp: str) -> Path:
        state = Path(tmp) / "s.db"
        db.init_db(state)
        return state

    def _build(self, state: Path, version: str = "v1", **kw) -> str:
        with db.connect(state) as con:
            return db.begin_graph_build(con, source_key="k", kb_id="kb_x", source_collection="kb_x",
                                        graph_version=version, **kw)

    def test_run_phase_skips_done_retries_failures_and_marks_success(self) -> None:
        from kb_pipeline.graph.build import _run_phase

        with tempfile.TemporaryDirectory() as tmp:
            state = self._state(tmp)
            bid = self._build(state)
            settings = SimpleNamespace(state_db=state)
            stages: list[str] = []
            calls = {"n": 0}

            def flaky():
                calls["n"] += 1
                if calls["n"] < 3:
                    raise RuntimeError("neo4j hiccup")
                return "ok"

            done: set[str] = set()
            out = _run_phase(settings, build_id=bid, phase="neo4j_import", label="Graph database import",
                             fn=flaky, done=done, stage=stages.append, retries=3, backoff=0)
            self.assertEqual(out, "ok")
            self.assertEqual(calls["n"], 3)
            self.assertIn("Graph database import (attempt 1 failed, retrying in 0s)", stages)
            self.assertEqual(done, {"neo4j_import"})
            with db.connect(state) as con:
                self.assertEqual(db.graph_phases_done(con, bid), ["neo4j_import"])

            # A marked phase: fn is never called, only "done, skipped" is checked in
            never = mock_fn = lambda: (_ for _ in ()).throw(AssertionError("must not run"))
            stages.clear()
            self.assertIsNone(_run_phase(settings, build_id=bid, phase="neo4j_import", label="Graph database import",
                                         fn=never, done=done, stage=stages.append))
            self.assertEqual(stages, ["Graph database import (done, skipped)"])

            # Retries exhausted: the exception propagates, no mark is written
            def always():
                raise RuntimeError("down")
            with self.assertRaises(RuntimeError):
                _run_phase(settings, build_id=bid, phase="enrich", label="回写图谱负载",
                           fn=always, done=done, stage=stages.append, retries=2, backoff=0)
            with db.connect(state) as con:
                self.assertEqual(db.graph_phases_done(con, bid), ["neo4j_import"])

    def test_resume_keeps_the_previous_runs_assets(self) -> None:
        """When resuming the same version, the chunk ledger, corpus fingerprint and phase marks are assets of
        the previous run and must not be zeroed as for a fresh build -- once cleared, there is nothing left
        to resume."""
        with tempfile.TemporaryDirectory() as tmp:
            state = self._state(tmp)
            bid = self._build(state, cache_fingerprint="fp1")
            chunks = [{"point_id": "p1", "chunk_uid": "u1", "doc_id": "kb_x:1", "content_version": "c1"}]
            with db.connect(state) as con:
                db.replace_graph_build_chunks(con, bid, chunks)
                db.record_graph_build_input(con, bid, input_rows=1, chunks=chunks, source_content_hash="h1")
                db.mark_graph_phase_done(con, bid, "prepare_input")
                db.mark_graph_phase_done(con, bid, "index")
                db.finish_graph_build(con, bid, status="cancelled", input_rows=1, active_chunk_count=1,
                                      active_doc_count=1, source_content_hash="h1", error="stopped")
            again = self._build(state, allow_existing=True, cache_fingerprint="fp1")
            self.assertEqual(again, bid)
            with db.connect(state) as con:
                row = db.graph_build_by_version(con, "kb_x", "v1")
                self.assertEqual(str(row["status"]), "running")
                self.assertEqual(str(row["source_content_hash"]), "h1")
                self.assertEqual(int(row["active_chunk_count"]), 1)
                back = [{k: r[k] for k in ("point_id", "chunk_uid", "doc_id", "content_version")}
                        for r in db.graph_build_chunk_refs(con, bid)]
                self.assertEqual(back, chunks)
                self.assertEqual(db.graph_phases_done(con, bid), ["prepare_input", "index"])

    def test_phases_count_only_when_config_and_corpus_are_unchanged(self) -> None:
        from unittest import mock

        from kb_pipeline.graph import build as gb

        with tempfile.TemporaryDirectory() as tmp:
            state = self._state(tmp)
            bid = self._build(state, cache_fingerprint="fp1")
            settings = SimpleNamespace(state_db=state)
            source = KBSource(kb_id="kb_x", collection="kb_x", source_root="r", source_type="local",
                              max_tokens=400, overlap_tokens=80, graph_enabled=True)
            with db.connect(state) as con:
                corpus = gb.source_snapshot_hash(db.active_chunk_refs(con, "kb_x"))
                db.record_graph_build_input(con, bid, input_rows=0, chunks=[], source_content_hash=corpus)
                db.mark_graph_phase_done(con, bid, "index")
                previous = db.graph_build_by_version(con, "kb_x", "v1")
                with mock.patch.object(gb, "graph_cache_fingerprint", return_value="fp1"):
                    self.assertEqual(gb._resumable_phases(con, previous, settings=settings, source=source),
                                     {"index"})
                with mock.patch.object(gb, "graph_cache_fingerprint", return_value="fp2"):
                    self.assertEqual(gb._resumable_phases(con, previous, settings=settings, source=source),
                                     set())
                # Once judged not resumable the marks are cleared, so even if the fingerprint matches again
                # later nothing is wrongly skipped
                self.assertEqual(db.graph_phases_done(con, bid), [])

    def test_trigger_resumes_the_same_version_only_when_reusable(self) -> None:
        from unittest import mock

        from kb_pipeline import discovery
        from kb_server import service

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); mirror = root / "mirror"; (mirror / "库C").mkdir(parents=True)
            state = self._state(tmp)
            with db.connect(state) as con:
                s, _ = discovery.enroll(con, mirror, "库C")
                discovery.set_config(con, s.kb_id, {
                    "graph_enabled": True,
                    "graph_llm": {"extract": "m", "summarize": "m"}})
                bid = db.begin_graph_build(con, source_key=s.kb_id, kb_id=s.kb_id,
                                           source_collection=s.collection, graph_version="v-halt",
                                           cache_fingerprint="fp")
                db.mark_graph_phase_done(con, bid, "extract")
                db.finish_graph_build(con, bid, status="failed", error="neo4j down")
                con.commit()
            stub = SimpleNamespace(state_db=state, mirror_root=mirror)
            with mock.patch.object(service, "settings", return_value=stub), \
                    mock.patch.object(service, "_spawn_graph_build", return_value={"started": True}) as spawn, \
                    mock.patch.object(service, "_paused_cache_reuse",
                                      return_value={"cache_reusable": True, "cache_stale": ""}):
                service.trigger_graph_build(s.kb_id)
                spawn.assert_called_once_with(stub, s.kb_id, graph_version="v-halt")
            with mock.patch.object(service, "settings", return_value=stub), \
                    mock.patch.object(service, "_spawn_graph_build", return_value={"started": True}) as spawn, \
                    mock.patch.object(service, "_paused_cache_reuse",
                                      return_value={"cache_reusable": False, "cache_stale": "标签改过"}):
                service.trigger_graph_build(s.kb_id)
                spawn.assert_called_once_with(stub, s.kb_id, graph_version=None)
            # Panel data: the failed state also reports the completed phases and resumability
            with db.connect(state) as con, \
                    mock.patch.object(service, "settings", return_value=SimpleNamespace(
                        state_db=state, mirror_root=mirror, graph_work_dir=root / "gw")), \
                    mock.patch.object(service, "_paused_cache_reuse",
                                      return_value={"cache_reusable": True, "cache_stale": ""}):
                info = service._graph_build_info(con, s.kb_id, s.collection)
            self.assertEqual(info["phases_done"], ["Entity extraction"])
            self.assertEqual(info["graph_version"], "v-halt")
            self.assertTrue(info["cache_reusable"])

    def test_spawn_passes_the_version_through(self) -> None:
        src = _repo_file("app/kb_server/service.py")
        body = src.split("def _spawn_graph_build", 1)[1].split("\ndef ", 1)[0]
        self.assertIn('"--graph-version", graph_version, "--allow-existing-graph-version"', body)

    def test_dropping_the_graph_clears_phase_marks(self) -> None:
        src = _repo_file("app/kb_pipeline/maintenance.py")
        body = src.split("def _drop_graph_data", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("graph_build_phases", body)

    def test_stage_labels_are_declared_in_the_build(self) -> None:
        """Stage names are the keys the panel uses to compute progress; this pins that the backend labels come
        from GRAPH_PHASES, that the names of unmarked steps were not renamed, and that every name in the
        console's GRAPH_STAGES has the same literal in the backend source -- change one side and forget the
        other, and the progress bar sticks at 2%."""
        import re

        from kb_pipeline.graph.build import GRAPH_PHASE_LABELS

        build = _repo_file("app/kb_pipeline/graph/build.py")
        for label in ("Switching version aliases", "Cleaning up old versions", "Entity resolution", "Description summaries"):
            self.assertIn(f'"{label}', build)
        backend = build + _repo_file("app/kb_pipeline/graph/vectors.py") + _repo_file("app/kb_pipeline/graph/compile.py")
        js = _repo_file("app/kb_server/static/app.js")
        table = js.split("const GRAPH_STAGES = [", 1)[1].split("];", 1)[0]
        names = re.findall(r'\["([^"]+)",', table)
        self.assertGreaterEqual(len(names), 8)
        for name in names:
            self.assertIn(f'"{name}', backend, f"控制台阶段「{name}」在后端找不到同名字面量")
        # The reverse: every phase name the backend writes into stage must be known to the console
        for label in GRAPH_PHASE_LABELS.values():
            if label == "Merge & resolution":
                continue   # merge itself writes no stage; its inner entity resolution / description summaries do
            self.assertIn(label, names, label)
        for runtime in ("Entity resolution", "Description summaries", "Structured facts", "Writing vectors",
                        "Switching version aliases", "Cleaning up old versions"):
            self.assertIn(runtime, names, runtime)
        self.assertNotIn("GraphRAG 索引", js)
        self.assertEqual(list(GRAPH_PHASE_LABELS.values()), ["Preparing corpus", "Entity extraction", "Merge & resolution", "Structured facts",
                                                             "Compiling view pages", "Writing vectors", "Graph database import"])


class RetiredRaptorModeTests(unittest.TestCase):
    """The summary tree (RAPTOR) mode was removed on 2026-09-05: the graph build has only the entity graph
    pipeline left. Pins that the old config key is retired, no leftover branches remain in the code, the old
    collection type is still known to GC, and old summary-tree build records can still be displayed."""

    def test_mode_is_gone_but_history_still_reads(self) -> None:
        from kb_pipeline import discovery
        from kb_pipeline.vector.qdrant import (ALL_GRAPH_VECTOR_TYPES, GRAPH_COLLECTION_RE, GRAPH_VECTOR_TYPES,
                                               graph_collection_name)
        from kb_server.service import RETIRED_CONFIG_KEYS, _graph_build_summary

        self.assertNotIn("graph_mode", discovery.DEFAULTS)
        self.assertNotIn("graph_mode", discovery.CONFIG_KEYS)
        self.assertIn("graph_mode", RETIRED_CONFIG_KEYS)
        self.assertFalse((Path(__file__).resolve().parents[1] / "kb_pipeline" / "graph" / "raptor.py").exists())
        for name in ("app/kb_pipeline/cli.py", "app/kb_pipeline/graph/build.py", "app/kb_pipeline/discovery.py",
                     "app/kb_server/service.py", "app/kb_server/static/index.html", "app/kb_server/static/app.js"):
            text = _repo_file(name)
            for leftover in ("cfg-gmode", "_build_by_mode", "graph_mode_value", "RAPTOR_LLM_STEPS", "graphrag-only"):
                self.assertNotIn(leftover, text, f"{name} 还留着 {leftover}")
        # The old collection type must still be known to GC, or historical collections could not be collected
        self.assertNotIn("raptor", GRAPH_VECTOR_TYPES)
        self.assertIn("raptor", ALL_GRAPH_VECTOR_TYPES)
        self.assertTrue(GRAPH_COLLECTION_RE.match(graph_collection_name("kb_005", "raptor", "v1")))
        maint = _repo_file("app/kb_pipeline/maintenance.py")
        self.assertIn("ALL_GRAPH_VECTOR_TYPES", maint.split("def _drop_graph_artifacts", 1)[1].split("\ndef ", 1)[0])
        # An old summary-tree build record: recognised by its manifest, a few numbers read, no error
        row = {"graph_build_id": "b0", "graph_version": "003-old", "status": "done", "stage": "完成",
               "started_at": 1, "finished_at": 2, "error": None, "input_rows": 3, "active_chunk_count": 10,
               "manifest_json": json.dumps({"mode": "raptor", "raptor": {"nodes": 134, "documents": 3}})}
        out = _graph_build_summary(row, [], {})
        self.assertEqual(out["mode"], "raptor")
        self.assertEqual(out["counts"]["nodes"], 134)
        # Current records are always entity graphs, even with an empty manifest (still running)
        out = _graph_build_summary(dict(row, manifest_json=None, status="running"), [], {})
        self.assertEqual((out["mode"], out["counts"]), ("entity_graph", None))


class GraphBuildRecordPruneTests(unittest.TestCase):
    """Now that artifacts keep only the current version (GRAPH_GC_KEEP_VERSIONS default 1), build records
    also keep only the useful rows: the running one, the latest successful one, and those after the latest
    full version; older ones are deleted together with their chunk ledger, phase check-ins and unit table."""

    def test_prune_keeps_the_rows_the_policy_needs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"; db.init_db(state)
            with db.connect(state) as con:
                def row(bid, kind, status, started, finished):
                    con.execute(
                        "INSERT INTO graph_builds(graph_build_id, source_key, kb_id, source_collection, graph_version, "
                        "status, started_at, finished_at, build_kind) VALUES(?, 'k', 'kb_1', 'kb_1', ?, ?, ?, ?, ?)",
                        (bid, "v-" + bid, status, started, finished, kind))
                row("old_full", "full", "done", 50, 60)
                row("old_append", "append", "done", 70, 80)
                row("full", "full", "done", 100, 110)
                row("a1", "append", "done", 200, 210)
                row("a2", "append", "done", 300, 310)
                row("run", "append", "running", 400, None)
                db.replace_graph_build_chunks(con, "old_full", [{"point_id": "p", "chunk_uid": "u", "doc_id": "d", "content_version": "v"}])
                db.replace_graph_build_chunks(con, "a2", [{"point_id": "p", "chunk_uid": "u", "doc_id": "d", "content_version": "v"}])
                self.assertEqual(db.prune_graph_builds(con, "kb_1"), 2)
                left = sorted(r["graph_build_id"] for r in con.execute("SELECT graph_build_id FROM graph_builds"))
                self.assertEqual(left, ["a1", "a2", "full", "run"])
                self.assertEqual(con.execute("SELECT COUNT(*) FROM graph_build_chunks WHERE graph_build_id='old_full'").fetchone()[0], 0)
                self.assertEqual(con.execute("SELECT COUNT(*) FROM graph_build_chunks WHERE graph_build_id='a2'").fetchone()[0], 1)
                self.assertEqual(db.count_graph_builds_since(con, "kb_1", kind="append", since_ts=110), 2)   # policy counts unaffected
                self.assertEqual(db.prune_graph_builds(con, "kb_1"), 0)
        src = _repo_file("app/kb_pipeline/graph/build.py")
        self.assertIn('return db.prune_graph_builds(con, source.kb_id)', src)                      # the end-of-build cleanup still prunes records
        # Codex 2026-09-13 F05: the sample list is capped at 50 and the total kept separately; the API uses the
        # total; truncation in this run and truncation in the cached assets are reported separately
        self.assertIn('"partial_units_total": sum(', src)
        self.assertIn('"truncated_cached": asset_flags.get("truncated", 0)', src)
        self.assertIn('facts_stats.get("partial_units_total")', _repo_file("app/kb_server/service.py"))
        self.assertIn('gc_step("build_records_pruned", prune_records', src)                         # 2026-09-13: its own step, not dragged down by the earlier cleanups
        self.assertIn('os.getenv("GRAPH_GC_KEEP_VERSIONS", "2")', _repo_file("app/kb_pipeline/config.py"))   # 09-06 evening: keep the previous version too so it can be switched back


class ExtractionPersistenceTests(unittest.TestCase):
    def test_extractions_are_keyed_by_unit_and_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"
            db.init_db(state)
            with db.connect(state) as con:
                db.save_graph_extraction(con, kb_id="kb_003", unit_id="u1", fingerprint="fp1",
                                         entities=[{"name": "A"}], relations=[], model="m", calls=2, stats={"records": 1})
                db.save_graph_extraction(con, kb_id="kb_003", unit_id="u2", fingerprint="fp1",
                                         entities=[], relations=[], model="m", calls=2)
                db.save_graph_extraction(con, kb_id="kb_003", unit_id="u1", fingerprint="fp2",
                                         entities=[{"name": "B"}, {"name": "C"}], relations=[], model="m2", calls=1)
                self.assertEqual(db.graph_extraction_unit_ids(con, "kb_003", "fp1"), {"u1", "u2"})
                rows = db.load_graph_extractions(con, "kb_003", "fp1", ["u1", "u9"])
                self.assertEqual(list(rows), ["u1"])
                self.assertEqual(rows["u1"]["entities"], [{"name": "A"}])
                self.assertEqual(db.graph_extraction_entity_total(con, "kb_003", "fp2", ["u1"]), 2)
                self.assertEqual(db.graph_extraction_count(con, "kb_003"), 3)
                units = [Unit(unit_id="u1", doc_id="d", rel_path="a", section_path=["S", "T"], block_ids=["b"],
                              chunk_uids=["c1", "c2"], point_ids=["p1", "p2"], n_tokens=12, text="t", order=0)]
                bid = db.begin_graph_build(con, source_key="k", kb_id="kb_003", source_collection="kb_003", graph_version="v1")
                self.assertEqual(db.replace_graph_units(con, bid, units), 1)
                row = con.execute("SELECT section, chunk_count, n_tokens FROM graph_units WHERE graph_build_id=?", (bid,)).fetchone()
                self.assertEqual((row["section"], row["chunk_count"], row["n_tokens"]), ("S > T", 2, 12))
                self.assertEqual(db.delete_graph_extractions(con, "kb_003"), 3)

    def test_prune_keeps_current_units_across_fingerprints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"
            db.init_db(state)
            with db.connect(state) as con:
                for unit, fp in (("u1", "fp1"), ("u2", "fp1"), ("u1", "fp2"), ("u3", "fp2")):
                    db.save_graph_extraction(con, kb_id="kb_003", unit_id=unit, fingerprint=fp,
                                             entities=[], relations=[], model="m", calls=1)
                db.save_graph_extraction(con, kb_id="kb_009", unit_id="u2", fingerprint="fp1",
                                         entities=[], relations=[], model="m", calls=1)
                self.assertEqual(db.prune_graph_extractions(con, "kb_003", ["u1"]), 2)   # u2 and u3 no longer exist
                self.assertEqual(db.graph_extraction_count(con, "kb_003"), 2)            # both fingerprints of u1 are kept
                self.assertEqual(db.graph_extraction_count(con, "kb_009"), 1)            # other KBs untouched
        src = (Path(__file__).resolve().parents[1] / "kb_pipeline" / "graph" / "build.py").read_text(encoding="utf-8")
        self.assertIn("db.prune_graph_extractions(con, source.kb_id, keep_ids)", src)
        self.assertIn("doc_ids is None and paths.units_file.exists()", src)   # a sample build does not clear the cache


class BundleTests(unittest.TestCase):
    """graph.json + units.jsonl -> Neo4j batches and Qdrant payloads, without connecting to a database."""

    _bundle_dir = _bundle_dir  # the fixture lives in _support; other test classes call it directly too

    def test_load_bundle_builds_nodes_edges_and_counts(self) -> None:
        from kb_pipeline.graph.neo4j_import import graph_uid, load_bundle

        source = KBSource(kb_id="kb_003", collection="kb_003", source_root="半导体", source_type="local_mirror",
                          max_tokens=400, overlap_tokens=80)
        with tempfile.TemporaryDirectory() as tmp:
            out, graph, units, entity_id = self._bundle_dir(tmp)
            bundle = load_bundle(out, source=source, graph_version="v1")
        counts = bundle["expected_counts"]
        self.assertEqual(counts["documents"], 2)
        self.assertEqual(counts["text_units"], 2)
        self.assertEqual(counts["qdrant_chunks"], 3)
        self.assertEqual((counts["entities"], counts["relations"]), (2, 1))
        self.assertEqual(counts["contributes_to_edges"], 3)
        self.assertEqual(counts["mentions_edges"], 3)         # a→u1,u2;b→u1
        self.assertEqual(counts["evidences_edges"], 1)
        self.assertEqual(counts["mentioned_in_edges"], 3)     # ghost is dropped
        self.assertNotIn("communities", counts)
        tu = bundle["text_units"][0]
        self.assertNotIn("text", tu)                          # text does not go into Neo4j
        self.assertEqual(tu["section"], "S")
        rel = bundle["relation_entity_edges"][0]["related_props"]
        self.assertEqual((rel["type"], rel["weight"], rel["directed"]), ("has_pin", 7.5, True))
        self.assertEqual(bundle["entities"][0]["uid"], graph_uid("kb_003", "v1", "Entity", entity_id("a")))
        self.assertEqual(bundle["entities"][0]["parent_type"], "p")
        self.assertNotIn("parent_type", bundle["entities"][1])   # empty values are not written

    def test_vector_payloads_and_upsert(self) -> None:
        from kb_pipeline.graph.vectors import entity_payload, point_id_for, relation_id, write_graph_vectors

        with tempfile.TemporaryDirectory() as tmp:
            out, graph, units, entity_id = self._bundle_dir(tmp)
            units_by_id = {u.unit_id: u for u in units}
            upserts: dict[str, list] = {}
            created: list[str] = []

            class FakeQ:
                def get_collections(self):
                    return SimpleNamespace(collections=[SimpleNamespace(name=n) for n in created])

                def create_collection(self, collection_name, vectors_config):
                    created.append(collection_name)

                def get_collection(self, collection_name):
                    return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=4))))

                def create_payload_index(self, **kw):
                    pass

                def upsert(self, collection_name, points, wait):
                    upserts.setdefault(collection_name, []).extend(points)

                def scroll(self, collection_name, limit, offset, with_payload, with_vectors):
                    return list(upserts.get(collection_name, [])) + stale.get(collection_name, []), None

                def delete(self, collection_name, points_selector, wait):
                    ids = set(points_selector.points)
                    stale[collection_name] = [p for p in stale.get(collection_name, []) if str(p.id) not in ids]
                    deleted.append((collection_name, sorted(ids)))

                def count(self, collection_name, exact):
                    return SimpleNamespace(count=len(upserts.get(collection_name, [])) + len(stale.get(collection_name, [])))

            # A relation point left over from the previous run of the same version: not in the graph this
            # run, it must be removed or the counts will not match
            stale = {"graph_003_relation__v1": [SimpleNamespace(id="stale-point")]}
            deleted: list = []

            embedded: list[str] = []

            def embed(texts):
                embedded.extend(texts)
                return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

            summary = write_graph_vectors(FakeQ(), embed, kb_id="kb_003", source_collection="kb_003", graph_version="v1",
                                          bundle=graph, units_by_id=units_by_id, vector_size=4)
        self.assertEqual(summary["collections"]["entity"]["points"], 2)
        self.assertEqual(summary["collections"]["relation"]["points"], 1)
        self.assertEqual(summary["collections"]["relation"]["stale_removed"], 1)
        self.assertEqual(deleted, [("graph_003_relation__v1", ["stale-point"])])
        self.assertIn("A: da", embedded)
        self.assertIn("A -[has_pin]-> B: A has B", embedded)
        ent_points = upserts["graph_003_entity__v1"]
        payload = ent_points[0].payload
        self.assertEqual((payload["gr_id"], payload["graph_type"], payload["pagerank"]), (entity_id("a"), "entity", 0.6))
        self.assertEqual(payload["point_ids"], ["p1", "p3"])            # ordered by hit count, inferred ones included
        self.assertEqual(ent_points[0].id, point_id_for("v1", entity_id("a")))
        rel_payload = upserts["graph_003_relation__v1"][0].payload
        self.assertEqual((rel_payload["type"], rel_payload["source_id"], rel_payload["point_ids"]),
                         ("has_pin", entity_id("a"), ["p1", "p2"]))
        self.assertEqual(rel_payload["gr_id"], relation_id("a", "b", "has_pin"))
        self.assertEqual(entity_payload(graph["entities"][1], kb_id="k", source_collection="c", graph_version="v", point_ids=[]).get("parent_type"), None)


class BuildWiringTests(unittest.TestCase):
    def test_fingerprint_covers_models_schema_and_unit_params(self) -> None:
        from kb_pipeline import discovery
        from kb_pipeline.graph.build import graph_cache_fingerprint

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "库").mkdir()
            state = root / "s.db"
            db.init_db(state)
            with db.connect(state) as con:
                src, _ = discovery.enroll(con, root, "库")
                db.upsert_llm(con, name="A", base_url="http://x/v1", api_key="", model_id="model-a")
                db.upsert_llm(con, name="B", base_url="http://x/v1", api_key="", model_id="model-b")
                discovery.set_config(con, src.kb_id, {"graph_llm": {"extract": "A", "summarize": "A"}})
                base_cfg = discovery.get_config(con, src.kb_id)
            settings = SimpleNamespace(state_db=state)

            def fp(extra):
                # The model slots are read from the database by kb_id, so the full set is written back every time
                cfg = dict(base_cfg); cfg.update(extra)
                with db.connect(state) as con:
                    discovery.set_config(con, src.kb_id, {**{k: None for k in discovery.CONFIG_KEYS if k.startswith("graph_")}, **cfg})
                    stored = discovery.get_config(con, src.kb_id)
                return graph_cache_fingerprint(settings, discovery.build_source(root, "库", stored, kb_id=src.kb_id))

            base = fp({})
            self.assertEqual(base, fp({}))
            for label, extra in (
                ("换模型", {"graph_llm": {"extract": "B", "summarize": "A"}}),
                ("改类型表", {"graph_entity_types": ["pin"]}),
                ("改谓词", {"graph_predicates": [{"name": "has_pin"}]}),
                ("改父类", {"graph_parent_types": {"pin": "interface"}}),
                ("改语言", {"graph_language": "Chinese"}),
                ("改合并切片数", {"graph_unit_chunks": 2}),
                ("改补漏轮数", {"graph_max_gleanings": 0}),
            ):
                self.assertNotEqual(base, fp(extra), label)
            with db.connect(state) as con:
                discovery.set_config(con, src.kb_id, {"graph_llm": {"extract": "A"}})
                cfg = discovery.get_config(con, src.kb_id)
            with self.assertRaisesRegex(RuntimeError, "Description summary model"):
                graph_cache_fingerprint(settings, discovery.build_source(root, "库", cfg, kb_id=src.kb_id))

    def test_phases_and_paths(self) -> None:
        from kb_pipeline.graph.build import GRAPH_PHASES, graph_paths

        self.assertEqual([p for p, _ in GRAPH_PHASES], ["prepare_input", "extract", "merge", "facts", "compile", "enrich", "neo4j_import"])
        settings = SimpleNamespace(graph_work_dir=Path("/w"))
        src = SimpleNamespace(collection="kb_003")
        paths = graph_paths(settings, src, "v1")
        self.assertEqual(paths.work_dir, Path("/w/work/003/v1"))
        self.assertEqual(paths.output_dir, paths.work_dir)
        self.assertEqual(paths.units_file, Path("/w/work/003/v1/units.jsonl"))
        self.assertEqual(paths.cache_file, Path("/w/cache/003.sqlite"))
        self.assertEqual(graph_paths(settings, src).work_dir, Path("/w/work/003"))

    def test_config_keys_and_validators(self) -> None:
        from kb_pipeline import discovery
        from kb_pipeline.limits import validate_graph_schema_config, validate_graph_unit_config

        for key in ("graph_unit_chunks", "graph_max_gleanings", "graph_predicates", "graph_parent_types"):
            self.assertIn(key, discovery.CONFIG_KEYS)
        self.assertNotIn("graph_competency_questions", discovery.CONFIG_KEYS)   # removed 2026-09-06
        for key in ("graph_chunk_size", "graph_chunk_overlap", "graph_encoding_model", "graph_max_cluster_size", "graph_use_lcc"):
            self.assertNotIn(key, discovery.CONFIG_KEYS)
        self.assertEqual(validate_graph_unit_config(3, 1), [])
        self.assertEqual(validate_graph_unit_config(1, 0), [])
        self.assertEqual(validate_graph_unit_config(None, None), [])
        self.assertTrue(validate_graph_unit_config(0, 1))
        self.assertTrue(validate_graph_unit_config(9, 1))
        self.assertTrue(validate_graph_unit_config(3, 5))
        self.assertTrue(validate_graph_unit_config("x", 1))
        self.assertEqual(validate_graph_schema_config([{"name": "a"}], {"t": "p"}), [])
        self.assertTrue(validate_graph_schema_config("", {"t": "p", "u": "q", "v": "r", "w": "s", "x": "t", "y": "u", "z": "v", "aa": "w", "bb": "x"}))
        with tempfile.TemporaryDirectory() as tmp:
            src = discovery.build_source(Path(tmp), "库", {
                "graph_predicates": [{"name": "Has Pin", "source_parents": ["Component"]}, "related_to"],
                "graph_parent_types": {"pin": "Interface"},
                "graph_unit_chunks": "2",
            })
        self.assertEqual(src.graph_predicates[0]["name"], "has_pin")
        self.assertEqual(len(src.graph_predicates), 1)
        self.assertEqual(src.graph_parent_types, {"pin": "interface"})
        self.assertEqual((src.graph_unit_chunks, src.graph_max_gleanings), (2, 1))

    def test_retired_console_keys_are_dropped_not_rejected(self) -> None:
        from kb_server.service import strip_retired_config

        out = strip_retired_config({"graph_chunk_size": 1200, "graph_use_lcc": False, "graph_unit_tokens": 1000, "max_tokens": 400,
                                    "graph_llm": {"extract": "a", "community": "b", "tune": "c"}})
        self.assertEqual(out, {"max_tokens": 400, "graph_llm": {"extract": "a", "tune": "c"}})

    def test_cli_has_query_and_no_eval_or_graphrag_flags(self) -> None:
        from kb_pipeline.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["graph", "query", "--source", "kb_003", "哪些器件支持 QDR", "--hops", "1"])
        self.assertEqual((args.graph_command, args.question, args.hops), ("query", "哪些器件支持 QDR", 1))
        for gone in (["graph", "eval", "--source", "kb_003", "--questions", "q.json"],        # 2026-09-09: graph QA / eval removed wholesale
                     ["graph", "answer", "--source", "kb_003", "VCC 是多少"]):
            with self.assertRaises(SystemExit):
                parser.parse_args(gone)
        fc = parser.parse_args(["graph", "factcheck", "--source", "kb_003", "--gold", "g.json"])   # fact-level comparison stays
        self.assertEqual((fc.graph_command, fc.gold), ("factcheck", "g.json"))
        with self.assertRaises(SystemExit):
            parser.parse_args(["graph", "build", "--skip-index"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["graph", "check-ready"])


class SampleBuildTests(unittest.TestCase):
    """Sample build: pick a few files, build a version without switching aliases, and evaluate it by
    version name."""

    def test_chunk_refs_filter_by_document(self) -> None:
        from kb_pipeline.graph.build import filter_chunk_refs

        chunks = [{"point_id": "p1", "doc_id": "kb_004:1"}, {"point_id": "p2", "doc_id": "kb_004:2"}, {"point_id": "p3", "doc_id": "kb_004:1"}]
        self.assertEqual([c["point_id"] for c in filter_chunk_refs(chunks, ["kb_004:1"])], ["p1", "p3"])
        self.assertEqual(filter_chunk_refs(chunks, None), chunks)
        self.assertEqual(filter_chunk_refs(chunks, []), [])

    def test_build_can_import_neo4j_without_activating(self) -> None:
        import inspect
        from kb_pipeline.graph.build import build_graph

        params = inspect.signature(build_graph).parameters
        self.assertIn("doc_ids", params)
        self.assertIn("import_neo4j", params)
        self.assertIsNone(params["import_neo4j"].default)      # defaults to following activate_aliases; old callers unchanged

    def test_recall_addresses_a_named_version_directly(self) -> None:
        from kb_pipeline.graph.recall import graph_collections_for

        self.assertEqual(graph_collections_for("kb_004"), {"entity": "graph_004_entity", "relation": "graph_004_relation",
                                                          "spec": "graph_004_spec", "page": "graph_004_page"})
        self.assertEqual(graph_collections_for("kb_004", "004-sample-1")["spec"], "graph_004_spec__004-sample-1")


class IncrementalAppendTests(unittest.TestCase):
    """Incremental append (2026-09-05): document-level delta, resolution replay, vector reuse, the append
    verdict, the rebuild baseline accepting only full versions, GC keeping only the latest few versions, and
    the console summary carrying the kind and reuse statistics."""

    def test_document_delta_detects_added_removed_and_modified_documents(self) -> None:
        from kb_pipeline.graph.build import document_delta

        base = {"d1": {("p1", "v1"), ("p2", "v1")}, "d2": {("p3", "v1")}, "d3": {("p4", "v1")}}
        chunks = [{"point_id": "p1", "content_version": "v1", "doc_id": "d1"},
                  {"point_id": "p2", "content_version": "v1", "doc_id": "d1"},
                  {"point_id": "p3b", "content_version": "v2", "doc_id": "d2"},     # d2 re-parsed: chunks changed
                  {"point_id": "p9", "content_version": "v1", "doc_id": "d9"}]      # d9 added; d3 removed
        d = document_delta(base, chunks)
        self.assertEqual((d["added_docs"], d["removed_docs"], d["modified_docs"], d["unchanged_docs"]),
                         (["d9"], ["d3"], ["d2"], 1))
        self.assertEqual((d["new_chunks"], d["removed_chunks"], d["documents"], d["base_documents"]), (2, 2, 3, 3))

    def test_document_delta_sees_changed_text_under_the_same_uid(self) -> None:
        """Without a profile upgrade block ids are stable, so after a re-parse uid and content_version stay
        the same and only the text changes: when the ledger holds a text fingerprint, changes are detected by
        it; an old ledger without fingerprints (empty string) raises no false alarm."""
        from kb_pipeline.graph.build import document_delta

        base = {"d1": {("p1", "v1", "aaaa"), ("p2", "v1", "bbbb")}, "d2": {("p3", "v1", "")}}
        chunks = [{"point_id": "p1", "content_version": "v1", "doc_id": "d1", "text_sha": "aaaa"},
                  {"point_id": "p2", "content_version": "v1", "doc_id": "d1", "text_sha": "b2b2"},   # text changed
                  {"point_id": "p3", "content_version": "v1", "doc_id": "d2", "text_sha": "cccc"}]   # old ledger has no fingerprint
        d = document_delta(base, chunks)
        self.assertEqual((d["added_docs"], d["removed_docs"], d["modified_docs"], d["unchanged_docs"]), ([], [], ["d1"], 1))
        self.assertEqual((d["new_chunks"], d["removed_chunks"]), (0, 0))
        # No fingerprint on the current side (old chunks rows) is likewise not a change; an old two-tuple
        # ledger still works
        chunks[1]["text_sha"] = ""
        self.assertEqual(document_delta(base, chunks)["modified_docs"], [])
        self.assertEqual(document_delta({"d1": {("p1", "v1"), ("p2", "v1")}}, chunks[:2])["modified_docs"], [])

    def _chat_client(self, responses):
        from kb_pipeline.graph.llm import ChatClient, LLMCache, LLMSpec

        calls: list[int] = []

        def chat(*a, **k):
            calls.append(1)
            return responses[min(len(calls) - 1, len(responses) - 1)] if responses else ""

        spec = LLMSpec(name="m", base_url="http://x", api_key="", model_id="m")
        return ChatClient(spec, cache=LLMCache(None), chat=chat, attempts=1, workers=1), calls

    @staticmethod
    def _param(key, title, desc, freq, doc="d"):
        return {"key": key, "title": title, "type": "parameter", "scope": "", "descriptions": [desc], "description": "",
                "frequency": freq, "unit_ids": [f"u-{doc}"], "doc_ids": [doc], "aliases": [], "attributes": []}

    def test_resolution_replays_prior_decisions_and_only_judges_new_pairs(self) -> None:
        ents = [self._param("tas", "tAS", "address setup", 3), self._param("tas setup time", "tAS setup time", "setup time", 1),
                self._param("vih", "VIH", "input high", 2), self._param("vih input high", "VIH input high", "input high voltage", 1),
                self._param("clk", "CLK", "clock", 2, doc="d2"), self._param("clk clock input", "CLK clock input", "clock input pin", 1, doc="d2")]
        # Previous version: the tAS pair was judged the same, the VIH pair was judged different; CLK comes
        # from a document new in this version
        prior = {"map": {"tas setup time": "tas"}, "judged": [["tas", "tas setup time"], ["vih", "vih input high"]]}
        client, calls = self._chat_client(["1: yes"])
        merged, _, stats = resolution.resolve(client, ents, [], prior=prior)
        self.assertEqual(len(calls), 1)                                          # only the CLK pair was asked
        self.assertEqual({e["key"] for e in merged}, {"tas", "vih", "vih input high", "clk"})
        self.assertEqual((stats["replayed_pairs"], stats["replayed_yes"], stats["judged_new"], stats["yes"]), (2, 1, 1, 1))
        self.assertEqual(stats["_map"], {"tas setup time": "tas", "clk clock input": "clk"})
        self.assertIn(["clk", "clk clock input"], stats["_judged"])
        self.assertIn(["vih", "vih input high"], stats["_judged"])               # negative verdicts are recorded too; not asked again next version
        # A pair merged in the previous version is merged again even if a changed title keeps it out of the
        # candidates this version; with no new pairs, nothing is asked at all
        ents2 = [self._param("tas", "tAS", "address setup", 3), self._param("tas setup time", "Address Setup", "setup time", 1)]
        client2, calls2 = self._chat_client([])
        merged2, _, stats2 = resolution.resolve(client2, ents2, [], prior=prior)
        self.assertEqual(([e["key"] for e in merged2], stats2["replayed_yes"], len(calls2)), (["tas"], 1, 0))
        # No prior (full rebuild) = everything judged afresh
        client3, calls3 = self._chat_client(["1: yes\n2: no\n3: yes"])
        _, _, stats3 = resolution.resolve(client3, ents, [])
        self.assertEqual((len(calls3), stats3["replayed_pairs"], stats3["batches"], stats3["candidates"]), (1, 0, 1, 3))

    def test_vectors_reuse_unchanged_rows_from_the_base_version(self) -> None:
        from kb_pipeline.graph.vectors import embed_sha, entity_id, point_id_for, write_graph_vectors

        def ent(key, title, desc):
            return {"key": key, "title": title, "type": "t", "description": desc, "descriptions": [desc], "frequency": 1,
                    "degree": 0, "pagerank": 0.0, "unit_ids": [], "aliases": [], "doc_ids": []}

        bundle = {"entities": [ent("a", "A", "da"), ent("b", "B", "db"), ent("c", "C", "dc")],
                  "relations": [], "specs": [], "mentions": [], "stats": {}}
        pid = lambda k: point_id_for("v0", entity_id(k))
        base = {
            pid("a"): SimpleNamespace(id=pid("a"), payload={"graph_type": "entity", "title": "A", "description": "da", "embed_sha": embed_sha("A: da")}, vector=[9.0, 9.0, 9.0, 9.0]),
            pid("b"): SimpleNamespace(id=pid("b"), payload={"graph_type": "entity", "title": "B", "description": "old"}, vector=[8.0] * 4),   # description changed
            pid("c"): SimpleNamespace(id=pid("c"), payload={"graph_type": "entity", "title": "C", "description": "dc"}, vector=[7.0] * 4),    # the old version wrote no embed_sha: recomputed from the fields
        }
        upserts: dict[str, list] = {}
        created: list[str] = []
        retrieved: list[str] = []

        class FakeQ:
            def get_collections(self):
                return SimpleNamespace(collections=[SimpleNamespace(name=n) for n in created])

            def create_collection(self, collection_name, vectors_config):
                created.append(collection_name)

            def get_collection(self, collection_name):
                return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=4))))

            def create_payload_index(self, **kw):
                pass

            def upsert(self, collection_name, points, wait):
                upserts.setdefault(collection_name, []).extend(points)

            def scroll(self, collection_name, limit, offset, with_payload, with_vectors):
                return list(upserts.get(collection_name, [])), None

            def delete(self, collection_name, points_selector, wait):
                pass

            def count(self, collection_name, exact):
                return SimpleNamespace(count=len(upserts.get(collection_name, [])))

            def retrieve(self, collection_name, ids, with_payload, with_vectors):
                retrieved.append(collection_name)
                if collection_name != "graph_003_entity__v0":
                    raise RuntimeError("no such collection")
                return [base[i] for i in ids if i in base]

        embedded: list[str] = []

        def embed(texts):
            embedded.extend(texts)
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

        reuse = {"entity": "graph_003_entity__v0", "relation": "graph_003_relation__v0", "spec": "graph_003_spec__v0"}
        summary = write_graph_vectors(FakeQ(), embed, kb_id="kb_003", source_collection="kb_003", graph_version="v1",
                                      bundle=bundle, units_by_id={}, vector_size=4, reuse_from=reuse, base_version="v0")
        self.assertEqual(embedded, ["B: db"])
        ent_summary = summary["collections"]["entity"]
        self.assertEqual((ent_summary["points"], ent_summary["reused"], ent_summary["embedded"]), (3, 2, 1))
        pts = {p.payload["gr_id"]: p for p in upserts["graph_003_entity__v1"]}
        self.assertEqual(pts[entity_id("a")].vector, [9.0, 9.0, 9.0, 9.0])
        self.assertEqual(pts[entity_id("c")].vector, [7.0] * 4)
        self.assertEqual(pts[entity_id("b")].payload["embed_sha"], embed_sha("B: db"))
        self.assertEqual(retrieved, ["graph_003_entity__v0"])                  # no relation / fact rows, so nothing fetched from the previous version
        # No reuse_from (full rebuild) = everything re-embedded
        embedded.clear()
        write_graph_vectors(FakeQ(), embed, kb_id="kb_003", source_collection="kb_003", graph_version="v2",
                            bundle=bundle, units_by_id={}, vector_size=4)
        self.assertEqual(embedded, ["A: da", "B: db", "C: dc"])

    def _seed_kb(self, root: Path, *, chunks_now, base=None, config=None):
        """A KB with the graph enabled and models chosen; base = [(version, chunk ledger, kind, same
        fingerprint?, finished days ago)] is built in order."""
        import time
        from unittest import mock

        from kb_pipeline import discovery
        from kb_pipeline.graph import build as build_mod

        (root / "库").mkdir(exist_ok=True)
        state = root / "s.db"
        db.init_db(state)
        with db.connect(state) as con:
            db.upsert_llm(con, name="甲", base_url="http://x/v1", api_key="k", model_id="m")
            src, _ = discovery.enroll(con, root, "库")
            discovery.set_config(con, src.kb_id, {"graph_enabled": True, "graph_llm": {"extract": "甲", "summarize": "甲"},
                                                  **(config or {})})
            con.commit()
        settings = SimpleNamespace(state_db=state, mirror_root=root, qdrant_url="http://q", qdrant_api_key="",
                                   graph_work_dir=root / "g", runtime_dir=root / "rt")
        source = discovery.enrolled_sources(state, root)[src.kb_id]
        fp = build_mod.graph_cache_fingerprint(settings, source)
        with db.connect(state) as con:
            for version, rows, kind, same_fp, days_ago in (base or []):
                bid = db.begin_graph_build(con, source_key=src.kb_id, kb_id=src.kb_id, source_collection=src.collection,
                                           graph_version=version, cache_fingerprint=fp if same_fp else "stale", build_kind=kind)
                db.replace_graph_build_chunks(con, bid, rows)
                db.finish_graph_build(con, bid, status="done", source_content_hash=build_mod.source_snapshot_hash(rows))
                con.execute("UPDATE graph_builds SET finished_at = ? WHERE graph_build_id = ?",
                            (int(time.time() - days_ago * 86400), bid))
            con.commit()
        patcher = mock.patch.object(build_mod, "active_source_chunks", return_value=chunks_now)
        patcher.start()
        self.addCleanup(patcher.stop)
        return build_mod, settings, src.kb_id, source

    @staticmethod
    def _rows(doc, ids, ver="v"):
        return [{"point_id": p, "chunk_uid": f"c{p}", "doc_id": doc, "content_version": ver} for p in ids]

    def test_evaluate_append_reasons_and_delta(self) -> None:
        old = self._rows("d1", ["p1", "p2"])
        new = old + self._rows("d2", ["p3"])
        with tempfile.TemporaryDirectory() as tmp:
            mod, settings, key, source = self._seed_kb(Path(tmp), chunks_now=old)
            self.assertEqual(mod.evaluate_append(settings, source_key=key, source=source)["reason"], "no_successful_build")
        with tempfile.TemporaryDirectory() as tmp:
            mod, settings, key, source = self._seed_kb(Path(tmp), chunks_now=old, base=[("v1", old, "full", True, 1)])
            self.assertEqual(mod.evaluate_append(settings, source_key=key, source=source)["reason"], "source_unchanged")
        with tempfile.TemporaryDirectory() as tmp:
            mod, settings, key, source = self._seed_kb(Path(tmp), chunks_now=new, base=[("v1", old, "full", True, 1)])
            out = mod.evaluate_append(settings, source_key=key, source=source)
            self.assertTrue(out["due"], out)
            self.assertEqual((out["base_version"], out["delta"]["added_docs"], out["delta"]["new_chunks"]), ("v1", ["d2"], 1))
        with tempfile.TemporaryDirectory() as tmp:   # config fingerprint changed: the previous extraction and resolution cannot be reused
            mod, settings, key, source = self._seed_kb(Path(tmp), chunks_now=new, base=[("v1", old, "full", False, 1)])
            self.assertEqual(mod.evaluate_append(settings, source_key=key, source=source)["reason"], "config_changed_needs_full_rebuild")
        with tempfile.TemporaryDirectory() as tmp:   # auto append off: the timer does not append, manual still does
            mod, settings, key, source = self._seed_kb(Path(tmp), chunks_now=new, base=[("v1", old, "full", True, 1)],
                                                       config={"graph_auto_append": False})
            self.assertEqual(mod.evaluate_append(settings, source_key=key, source=source)["reason"], "auto_append_off")
            self.assertTrue(mod.evaluate_append(settings, source_key=key, source=source, ignore_auto_flag=True)["due"])
        with tempfile.TemporaryDirectory() as tmp:
            mod, settings, key, source = self._seed_kb(Path(tmp), chunks_now=new, base=[("v1", old, "full", True, 1)],
                                                       config={"graph_paused": True})
            self.assertEqual(mod.evaluate_append(settings, source_key=key, source=source)["reason"], "paused_by_operator")

    def test_rebuild_policy_takes_the_last_full_build_as_baseline(self) -> None:
        old = self._rows("d1", [f"p{i}" for i in range(10)])
        now = old + self._rows("d2", ["n1", "n2"])
        with tempfile.TemporaryDirectory() as tmp:
            # Full v1 (10 days ago, 10 chunks) -> append v2 (1 hour ago, 12 chunks) -> still those 12 chunks now
            mod, settings, key, source = self._seed_kb(
                Path(tmp), chunks_now=now, config={"graph_rebuild_new_chunk_count": 2, "graph_rebuild_interval": "7d"},
                base=[("v1", old, "full", True, 10), ("v2", now, "append", True, 1 / 24)])
            out = mod.evaluate_rebuild(settings, source_key=key, source=source)
            self.assertTrue(out["due"], out)
            self.assertEqual((out["baseline_graph_version"], out["latest_graph_version"], out["appends_since_full"]), ("v1", "v2", 1))
            by_name = {c["name"]: c for c in out["conditions"]}
            self.assertEqual(by_name["new_chunk_count"]["new_chunks"], 2)      # counted against the full version; the append did not reset it
            self.assertTrue(by_name["interval"]["due"])                         # the clock also starts from the full version
            with db.connect(settings.state_db) as con:
                self.assertEqual(db.latest_successful_graph_build(con, key)["graph_version"], "v2")
                self.assertEqual(db.latest_successful_graph_build(con, key, kind="full")["graph_version"], "v1")
                # A sample build (manifest carries doc_filter) does not count as the current graph
                bid = db.begin_graph_build(con, source_key=key, kb_id=key, source_collection=source.collection, graph_version="v3-sample")
                db.finish_graph_build(con, bid, status="done", manifest={"input": {"doc_filter": ["d1"]}})
                con.commit()
                self.assertEqual(db.latest_successful_graph_build(con, key)["graph_version"], "v2")
                self.assertEqual(db.latest_successful_graph_build(con, key, exclude_samples=False)["graph_version"], "v3-sample")

    def test_graph_gc_keeps_only_the_latest_versions(self) -> None:
        import time

        from kb_pipeline.vector.qdrant import delete_old_graph_collections, graph_collection_name

        versions = [f"003-{time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time() - i * 3600))}-{i:06x}" for i in range(5)]   # newest -> oldest
        names = [graph_collection_name("kb_003", t, v) for v in versions for t in ("entity", "relation")]

        class FakeQ:
            def get_aliases(self):
                return SimpleNamespace(aliases=[SimpleNamespace(alias_name="graph_003_entity", collection_name=names[0])])

            def get_collections(self):
                return SimpleNamespace(collections=[SimpleNamespace(name=n) for n in names])

        out = delete_old_graph_collections(FakeQ(), source_collections=["kb_003"], retention_days=14, dry_run=True, keep_latest=2)
        self.assertEqual({d["collection"] for d in out["deleted"]}, set(names[4:]))          # versions 3, 4 and 5 (one of each collection)
        self.assertTrue(all(d["reason"] == "beyond_keep_latest" for d in out["deleted"]))
        self.assertEqual({s["reason"] for s in out["skipped_collections"]}, {"aliased", "within_retention"})
        untouched = delete_old_graph_collections(FakeQ(), source_collections=["kb_003"], retention_days=14, dry_run=True)
        self.assertEqual(untouched["deleted"], [])                                            # no keep_latest = by age only

    def test_graph_gc_discards_unsuccessful_versions_without_a_keep_slot(self) -> None:
        import time

        from kb_pipeline.vector.qdrant import delete_old_graph_collections, graph_collection_name

        versions = [f"003-{time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time() - i * 3600))}-{i:06x}" for i in range(4)]   # newest -> oldest
        names = {v: graph_collection_name("kb_003", "entity", v) for v in versions}

        class FakeQ:
            def get_aliases(self):
                return SimpleNamespace(aliases=[SimpleNamespace(alias_name="graph_003_entity", collection_name=names[versions[0]])])

            def get_collections(self):
                return SimpleNamespace(collections=[SimpleNamespace(name=n) for n in names.values()])

        # Without discard: the half-finished version 2 takes a keep slot by timestamp and pushes out
        # version 3 (the last good graph)
        plain = delete_old_graph_collections(FakeQ(), source_collections=["kb_003"], retention_days=14, dry_run=True, keep_latest=2)
        self.assertEqual({d["collection"] for d in plain["deleted"]}, {names[versions[2]], names[versions[3]]})
        # With discard: the half-finished version is deleted outright without taking a slot; version 3 stays
        out = delete_old_graph_collections(FakeQ(), source_collections=["kb_003"], retention_days=14, dry_run=True, keep_latest=2,
                                           discard_versions={versions[1]})
        reasons = {d["collection"]: d["reason"] for d in out["deleted"]}
        self.assertEqual(reasons, {names[versions[1]]: "unsuccessful", names[versions[3]]: "beyond_keep_latest"})
        self.assertEqual(out["discard_versions"], [versions[1]])
        # The aliased version is never deleted, even when named explicitly
        aliased = delete_old_graph_collections(FakeQ(), source_collections=["kb_003"], retention_days=14, dry_run=True, keep_latest=2,
                                               discard_versions={versions[0]})
        self.assertNotIn(names[versions[0]], {d["collection"] for d in aliased["deleted"]})

    def test_workspace_gc_discards_unsuccessful_versions_without_a_keep_slot(self) -> None:
        import os
        import time

        from kb_pipeline.graph.build import _prune_graph_workspaces

        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / "work" / "003"
            now = time.time()
            for i, v in enumerate(("v0", "v1", "v2", "v3")):          # newest -> oldest
                (parent / v).mkdir(parents=True)
                os.utime(parent / v, (now - i * 3600, now - i * 3600))
            settings = SimpleNamespace(graph_work_dir=Path(tmp))
            source = SimpleNamespace(collection="kb_003")
            out = _prune_graph_workspaces(settings, source, keep_version="v0", retention_days=14, keep_latest=2, discard_versions={"v1"})
            self.assertEqual({Path(d).name for d in out["removed_dirs"]}, {"v1", "v3"})
            self.assertEqual(sorted(p.name for p in parent.iterdir()), ["v0", "v2"])          # the last good graph stays
            self.assertEqual(out["discard_versions"], ["v1"])

    def test_neo4j_gc_discards_unsuccessful_versions_without_a_keep_slot(self) -> None:
        import time
        from unittest import mock

        from kb_pipeline.graph import neo4j_import as n4

        versions = [f"003-{time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time() - i * 3600))}-{i:06x}" for i in range(4)]
        deleted: list[str] = []
        fakes = dict(
            neo4j_driver=lambda settings: SimpleNamespace(close=lambda: None),
            active_neo4j_graph_version=lambda driver, kb_id: versions[0],
            neo4j_graph_versions=lambda driver, kb_id: [{"graph_version": v, "imported_at": None} for v in versions[:3]],  # version 4 has no GraphVersion marker
            count_version_nodes=lambda driver, kb_id, version: 10,
            delete_version=lambda driver, kb_id, version: deleted.append(version) or 10,
        )
        with mock.patch.multiple(n4, **fakes):
            out = n4.delete_old_neo4j_graph_versions(SimpleNamespace(), sources=[SimpleNamespace(kb_id="kb_003", collection="kb_003")],
                                                     retention_days=14, dry_run=False, keep_latest=2,
                                                     discard_versions={versions[1], versions[3]})
        rows = {d["graph_version"]: d.get("reason") for d in out["sources"]["kb_003"]["deleted"]}
        self.assertEqual(rows, {versions[1]: "unsuccessful", versions[3]: "unsuccessful"})   # half-finished versions deleted; version 3 (within the keep slots) stays
        self.assertEqual(sorted(deleted), sorted([versions[1], versions[3]]))
        self.assertEqual(out["total_deleted_nodes"], 20)

    def test_neo4j_version_delete_removes_rels_first_and_halves_the_batch_under_memory_pressure(self) -> None:
        from neo4j.exceptions import TransientError

        from kb_pipeline.graph.neo4j_import import delete_version

        calls: list[tuple[str, int]] = []
        remaining = {"rels": 12000, "nodes": 7000}

        class FakeSession:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def run(self, query, **params):
                kind = "rels" if "DELETE r" in query else "nodes"
                size = int(params["batch_size"])
                calls.append((kind, size))
                if kind == "rels" and size > 2500:
                    raise TransientError("The allocation of an extra 2.0 MiB would use more than the limit 1.4 GiB")
                n = min(size, remaining[kind]); remaining[kind] -= n
                return SimpleNamespace(single=lambda: {"n": n})

        driver = SimpleNamespace(session=lambda: FakeSession())
        total = delete_version(driver, "kb_003", "003-v", batch_size=5000)
        self.assertEqual(total, 7000)                                                   # the return value is the node count
        self.assertEqual([c for c in calls if c[0] == "rels"][:3], [("rels", 5000), ("rels", 2500), ("rels", 2500)])   # halved on hitting the limit
        first_node = next(i for i, c in enumerate(calls) if c[0] == "nodes")
        self.assertTrue(all(c[0] == "rels" for c in calls[:first_node]))                # nodes are deleted only after all relationships
        self.assertEqual(calls[first_node], ("nodes", 5000))                              # the node round starts from the original batch size
        self.assertEqual(remaining, {"rels": 0, "nodes": 0})

    def test_unsuccessful_graph_versions_helper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"
            db.init_db(state)
            with db.connect(state) as con:
                def add(version, status, started):
                    bid = db.begin_graph_build(con, source_key="kb_003", kb_id="kb_003", source_collection="kb_003", graph_version=version)
                    if status != "running":
                        db.finish_graph_build(con, bid, status=status)
                    con.execute("UPDATE graph_builds SET started_at = ? WHERE graph_build_id = ?", (started, bid))
                add("v-old-cancel", "cancelled", 100)
                add("v-done", "done", 200)
                add("v-new-fail", "failed", 300)
                add("v-resumed", "cancelled", 150)
                con.execute("UPDATE graph_builds SET status = 'done' WHERE graph_version = 'v-resumed'")   # later resumed successfully
                add("v-run", "running", 400)
                con.commit()
                self.assertEqual(db.unsuccessful_graph_versions(con, "kb_003"), {"v-old-cancel", "v-new-fail"})
                # Timed cleanup: only versions started before the latest success count; the newest failed
                # one is left for resume
                self.assertEqual(db.unsuccessful_graph_versions(con, "kb_003", superseded_only=True), {"v-old-cancel"})
                self.assertEqual(db.unsuccessful_graph_versions(con, "kb_999", superseded_only=True), set())

    def test_extraction_flag_counts_read_the_cached_asset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"
            db.init_db(state)
            with db.connect(state) as con:
                for uid, st in (("u1", {"truncated": 1}), ("u2", {"truncated": 0, "partial": 1}), ("u3", {})):
                    db.save_graph_extraction(con, kb_id="kb_x", unit_id=uid, fingerprint="fp", entities=[], relations=[], model="m", calls=1, stats=st)
                    db.save_graph_facts(con, kb_id="kb_x", unit_id=uid, fingerprint="fp", facts=[], model="m", calls=1, stats=st)
                con.commit()
                self.assertEqual(db.graph_extraction_flag_counts(con, "kb_x", "fp"), {"truncated": 1, "partial": 1})
                self.assertEqual(db.graph_extraction_flag_counts(con, "kb_x", "fp", flags=("partial",), table="graph_facts"), {"partial": 1})
                self.assertEqual(db.graph_extraction_flag_counts(con, "kb_x", "other"), {"truncated": 0, "partial": 0})

    def test_resumed_build_recomputes_the_extract_summary_from_the_cache(self) -> None:
        """Codex 2026-09-14 O01: when a resume skips extraction, the summary is recomputed from the cached
        assets rather than reduced to "skipped"."""
        src = (Path(__file__).resolve().parents[2] / "app/kb_pipeline/graph/build.py").read_text(encoding="utf-8")
        self.assertIn('extract_stats.update(_cached_extract_summary(settings, source, schema, specs, paths))', src)
        self.assertIn('"truncated_cached": flags.get("truncated", 0), "failed_documents"', src)

    def test_gc_callers_pass_unsuccessful_versions(self) -> None:
        root = Path(__file__).resolve().parents[2] / "app/kb_pipeline"
        build_src = (root / "graph/build.py").read_text(encoding="utf-8")
        self.assertEqual(build_src.count("discard_versions=discard"), 3)                    # Qdrant / workspace / Neo4j: one rule in three places
        self.assertIn("db.unsuccessful_graph_versions(con, source.kb_id) - {graph_version}", build_src)
        maint_src = (root / "maintenance.py").read_text(encoding="utf-8")
        self.assertEqual(maint_src.count("discard_versions=superseded_unsuccessful_versions(settings)"), 2)   # two sites in the timed cleanup
        self.assertEqual(build_src.count("gc_step(\""), 4)                                       # four independent cleanups; one failing does not drag down the others

    def test_cli_config_and_build_wiring(self) -> None:
        import inspect

        from kb_pipeline import discovery
        from kb_pipeline.cli import build_parser
        from kb_pipeline.graph.build import build_graph

        parser = build_parser()
        args = parser.parse_args(["graph", "append", "--source", "kb_003", "--force"])
        self.assertEqual((args.graph_command, args.source, args.force), ("append", ["kb_003"], True))
        self.assertIn("incremental", inspect.signature(build_graph).parameters)
        self.assertIn("graph_auto_append", discovery.CONFIG_KEYS)
        self.assertIs(discovery.DEFAULTS["graph_auto_append"], True)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(discovery.build_source(Path(tmp), "库", {}).graph_auto_append)
            self.assertFalse(discovery.build_source(Path(tmp), "库", {"graph_auto_append": False}).graph_auto_append)

    def test_build_summary_carries_kind_base_delta_and_reuse(self) -> None:
        from kb_server.service import _graph_build_summary

        manifest = {"build_kind": "append", "base_version": "v0",
                    "delta": {"added_docs": ["d9"], "removed_docs": [], "modified_docs": [], "new_chunks": 3},
                    "enrich": {"collections": {"entity": {"reused": 5, "embedded": 2}, "relation": {"reused": 7, "embedded": 0}}},
                    "graph": {"entities": 10, "relations": 12, "units": 4, "resolution": {"replayed_pairs": 3, "judged_new": 1}}}
        row = {"graph_build_id": "b", "graph_version": "v1", "status": "done", "stage": "完成", "started_at": 1, "finished_at": 2,
               "manifest_json": json.dumps(manifest), "error": None, "input_rows": 3, "active_chunk_count": 9, "build_kind": "append"}
        out = _graph_build_summary(row, [], {})
        self.assertEqual((out["build_kind"], out["base_version"], out["delta"]["added_docs"]), ("append", "v0", ["d9"]))
        self.assertEqual(out["stats"]["reuse"], {"vectors_reused": 12, "vectors_embedded": 2})
        self.assertEqual(out["resolution"]["replayed_pairs"], 3)
        # An old record: no build_kind column and none in the manifest -> full, reuse statistics empty
        old = {k: v for k, v in row.items() if k != "build_kind"}
        old["manifest_json"] = "{}"
        self.assertEqual((_graph_build_summary(old, [], {})["build_kind"], _graph_build_summary(old, [], {})["stats"]["reuse"]), ("full", None))


class LLMCacheVersioningTests(unittest.TestCase):
    """Response cache managed per version (2026-09-09): entries hit / written during a build are tagged with
    the build id, and after a full build completes the entries this version did not use are deleted."""

    def test_marks_hits_and_prunes_the_rest(self) -> None:
        from kb_pipeline.graph.llm import LLMCache

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.sqlite"
            c = LLMCache(path)
            for k in ("a", "b", "c"):
                c.put(k, "m", "r-" + k)                      # no build id: used_by is empty
            c.close()
            LLMCache.build_tag = "build-1"
            try:
                c = LLMCache(path)
                self.assertEqual(c.get("a"), "r-a")          # hit: tagged build-1
                c.put("d", "m", "r-d")                       # new write: also tagged build-1
                self.assertIsNone(c.get("zzz"))
                c.close()
                c = LLMCache(path)
                self.assertEqual(c.prune_unused("build-1"), 2)   # b and c unused, deleted
                self.assertEqual(c.count(), 2)
                self.assertEqual((c.get("a"), c.get("d"), c.get("b")), ("r-a", "r-d", None))
                c.close()
            finally:
                LLMCache.build_tag = None
            # Without a build id, hits are not tagged and nothing is deleted
            c = LLMCache(path)
            self.assertEqual(c.get("a"), "r-a")
            self.assertEqual(c.prune_unused(""), 0)
            self.assertEqual(LLMCache(None).prune_unused("x"), 0)
            c.close()

    def test_old_cache_files_get_the_column(self) -> None:
        from kb_pipeline.graph.llm import LLMCache

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.sqlite"
            con = sqlite3.connect(path)
            con.execute("CREATE TABLE responses (key TEXT PRIMARY KEY, model TEXT NOT NULL, response TEXT NOT NULL, created_at INTEGER NOT NULL)")
            con.execute("INSERT INTO responses VALUES ('k', 'm', 'v', 1)")
            con.commit(); con.close()
            c = LLMCache(path)                               # old file: the used_by column is added automatically
            self.assertEqual(c.get("k"), "v")
            cols = {r[1] for r in c._con.execute("PRAGMA table_info(responses)").fetchall()}
            self.assertIn("used_by", cols)
            c.close()
        # Build wiring: set the build id at start, clear it at the end, prune after a full build completes;
        # incremental append and resume do not prune
        src = (Path(__file__).resolve().parents[2] / "app/kb_pipeline/graph/build.py").read_text(encoding="utf-8")
        self.assertIn("LLMCache.build_tag = build_id if not dry_run else None", src)
        self.assertIn("LLMCache.build_tag = None", src)
        self.assertIn('if not dry_run and doc_ids is None and not incremental and not result.get("resumed_phases"):', src)
        self.assertIn("llm_cache.prune_unused(build_id)", src)


class GraphBuildFixRegressionTests(_CodexAudit20260906TestsSupport, _CodexFinalTestsSupport, unittest.TestCase):
    """Regressions for problems found in successive re-checks, health checks and reviews; each test's docstring
    gives the source and the symptom observed at the time."""

    def test_check_rebuild_noop_when_no_graph_enabled(self) -> None:
        """The midnight policy check must idle normally (exit 0) when no KB has the graph enabled, instead of
        ERROR: no graph-enabled source selected marking the unit failed."""
        import argparse
        from unittest import mock

        from kb_pipeline import cli

        with tempfile.TemporaryDirectory() as tmp:
            stub = SimpleNamespace(state_db=Path(tmp) / "s.db", sources={})
            base = dict(env_file=None, source=None, collection=None, all=False)
            with mock.patch.object(cli, "load_settings", return_value=stub):
                args = argparse.Namespace(graph_command="check-rebuild",
                                          execute=True, dry_run=False, **base)
                self.assertEqual(cli.cmd_graph(args), 0)   # idles successfully
                with self.assertRaises(ValueError):        # an explicit build without a target is still an error
                    cli.cmd_graph(argparse.Namespace(graph_command="build",
                                                     graph_version=None, **base))

    def test_dead_graph_build_is_reconciled_not_stuck_running(self) -> None:
        """H1: after the build process is killed (deployment restart, power loss), the running record must be
        judged failed, otherwise the console shows "building" forever and build now / delete graph are
        refused for good."""
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "s.db"
            db.init_db(state)
            with db.connect(state) as con:
                def put(build_id: str, *, host: str, pid: int, heartbeat: int) -> None:
                    con.execute(
                        "INSERT INTO graph_builds(graph_build_id, source_key, kb_id, source_collection, "
                        "graph_version, status, started_at, worker_host, worker_pid, heartbeat_at) "
                        "VALUES(?, 'k', 'k', 'kb_k', ?, 'running', ?, ?, ?, ?)",
                        (build_id, build_id, int(time.time()), host, pid, heartbeat),
                    )

                now = int(time.time())
                put("alive", host=socket.gethostname(), pid=os.getpid(), heartbeat=now)   # this process, alive
                put("dead", host=socket.gethostname(), pid=99999999, heartbeat=now)       # this host, pid gone
                put("stale-remote", host="other-host", pid=1234, heartbeat=now - 7200)    # other host, heartbeat timed out
                put("fresh-remote", host="other-host", pid=1234, heartbeat=now - 60)      # other host, heartbeat fresh
                con.commit()

                recovered = set(db.reconcile_stale_graph_builds(con))
                self.assertEqual(recovered, {"dead", "stale-remote"})
                status = dict(con.execute("SELECT graph_build_id, status FROM graph_builds").fetchall())
                self.assertEqual(status["alive"], "running")          # the live one must not be touched
                self.assertEqual(status["fresh-remote"], "running")
                self.assertEqual(status["dead"], "failed")
                self.assertEqual(status["stale-remote"], "failed")
                err = con.execute("SELECT error FROM graph_builds WHERE graph_build_id='dead'").fetchone()[0]
                self.assertIn("pid", str(err))                        # the failure reason is spelled out

    def test_graph_step_model_resolution(self) -> None:
        from kb_pipeline import discovery
        from kb_pipeline.graph.build import GRAPH_LLM_STEPS, resolve_llm_specs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "产品资料").mkdir()
            state = Path(tmp) / "s.db"
            db.init_db(state)
            with db.connect(state) as con:
                src, _ = discovery.enroll(con, root, "产品资料")
                db.upsert_llm(con, name="通用模型", base_url="http://gpu-b:8000/v1", api_key="k", model_id="glm-5.2")
                db.upsert_llm(con, name="抽取专用", base_url="http://gpu-a:8000/v1", api_key="k2",
                              model_id="qwen3.7-plus", protocol="anthropic")
                # graph models are strictly per-KB: every step must be chosen
                # in the KB's own config, there is no global fallback
                discovery.set_config(con, src.kb_id, {"graph_llm": {
                    "extract": "抽取专用", "summarize": "通用模型"}})
            settings = SimpleNamespace(state_db=state)
            specs = resolve_llm_specs(settings, src, GRAPH_LLM_STEPS)
            self.assertEqual(specs["extract"].model_id, "qwen3.7-plus")
            self.assertEqual(specs["extract"].base_url, "http://gpu-a:8000/v1")
            self.assertEqual(specs["extract"].protocol, "anthropic")
            self.assertEqual(specs["summarize"].model_id, "glm-5.2")
            self.assertEqual(specs["summarize"].protocol, "openai")
            with db.connect(state) as con:
                discovery.set_config(con, src.kb_id, {"graph_llm": {"extract": "抽取专用"}})
            with self.assertRaisesRegex(RuntimeError, "Description summary model"):
                resolve_llm_specs(settings, src, GRAPH_LLM_STEPS)      # any unchosen step fails loudly

    def test_failed_extraction_units_are_reported(self) -> None:
        """R3: the hard-coded per-document failure threshold is removed; failed unit / document counts go into
        the status card data."""
        build = _repo_file("app/kb_pipeline/graph/build.py")
        self.assertNotIn("_MAX_FAILED_UNITS_PER_DOC", build)
        self.assertIn("failed_docs = sorted(per_doc)", build)
        svc = _repo_file("app/kb_server/service.py")
        summary = svc.split("def _graph_build_summary", 1)[1].split("\ndef ", 1)[0]
        self.assertIn('"failed_documents"', summary)
        self.assertIn('info["extract_failed_units"]', svc)
        self.assertNotIn("entity and community", _repo_file("app/kb_pipeline/cli.py"))

    def test_partial_fact_units_reach_the_status_card(self) -> None:
        """Codex review F10: the number of units still incomplete after the split-in-half retry goes into an
        API field; the build does not pretend to have fully succeeded. Decided by the user 2026-09-12: the
        status card no longer shows that sentence, the field stays."""
        svc = _repo_file("app/kb_server/service.py")
        self.assertIn('info["facts_partial_units"] = int(facts_stats.get("partial_units_total")', svc)   # 2026-09-13 F05: use the real total, the old list as fallback
        js = _repo_file("app/kb_server/static/app.js")
        self.assertNotIn("个单元的事实没有写完", js)
        self.assertNotIn("partialUnits", js)
        self.assertNotIn("个单元的事实没有写完", _repo_file("app/kb_server/static/i18n.js"))

    def test_f04_snapshot_hash_is_content_based_and_uncached(self) -> None:
        """F04: the snapshot hash used to be cached by id(chunks) (id reuse after garbage collection would hit
        a stale value) and did not include the text fingerprint."""
        from kb_pipeline.graph import build as gb

        self.assertFalse(hasattr(gb, "_snapshot_hash_cache"))
        a = [{"point_id": "p1", "chunk_uid": "c1", "doc_id": "d1", "content_version": "v1", "text_sha": "s1"},
             {"point_id": "p2", "chunk_uid": "c2", "doc_id": "d1", "content_version": "v1", "text_sha": "s2"}]
        same_reordered = [dict(a[1]), dict(a[0])]
        text_changed = [dict(a[0]), {**a[1], "text_sha": "s2-changed"}]
        h = gb.source_snapshot_hash(a)
        self.assertEqual(h, gb.source_snapshot_hash(same_reordered))          # order-independent
        self.assertNotEqual(h, gb.source_snapshot_hash(text_changed))         # same point_id / version, changed text changes the hash
        ident = id(a)
        del a
        b = [{"point_id": "p9", "chunk_uid": "c9", "doc_id": "d9", "content_version": "v9", "text_sha": "s9"}]
        self.assertNotEqual(gb.source_snapshot_hash(b), h, f"id reuse={id(b) == ident}")
        src = _repo_file("app/kb_pipeline/graph/build.py")
        self.assertIn('"text_sha": str(chunk.get("text_sha") or "")', src)

    def test_f05_fingerprint_covers_predicate_definitions_and_the_embedding_model(self) -> None:
        from kb_pipeline import discovery
        from kb_pipeline.graph.build import graph_cache_fingerprint

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "库").mkdir()
            state = root / "s.db"
            db.init_db(state)
            with db.connect(state) as con:
                src, _ = discovery.enroll(con, root, "库")
                db.upsert_llm(con, name="A", base_url="http://x/v1", api_key="", model_id="model-a")
                discovery.set_config(con, src.kb_id, {"graph_llm": {"extract": "A", "summarize": "A"}})

            def fp(predicates, model="qwen3-embedding-0.6b"):
                with db.connect(state) as con:
                    discovery.set_config(con, src.kb_id, {"graph_predicates": predicates})
                    stored = discovery.get_config(con, src.kb_id)
                settings = SimpleNamespace(state_db=state, embedding_model_id=model, embedding_dim=1024)
                return graph_cache_fingerprint(settings, discovery.build_source(root, "库", stored, kb_id=src.kb_id))

            base = fp([{"name": "has_pin"}])
            self.assertEqual(base, fp([{"name": "has_pin"}]))
            self.assertNotEqual(base, fp([{"name": "has_pin", "description": "component exposes a pin"}]), "改描述")
            self.assertNotEqual(base, fp([{"name": "has_pin", "source_parents": ["component"], "target_parents": ["interface"]}]), "改端点约束")
            self.assertNotEqual(base, fp([{"name": "has_pin"}], model="other-1024d-model"), "同维换嵌入模型")

    def test_f05_vector_reuse_requires_the_same_embedding_model(self) -> None:
        from kb_pipeline.graph.vectors import embed_sha, entity_id, point_id_for, write_graph_vectors

        ent = {"key": "a", "title": "A", "type": "t", "description": "da", "descriptions": ["da"], "frequency": 1,
               "degree": 0, "pagerank": 0.0, "unit_ids": [], "aliases": [], "doc_ids": []}
        bundle = {"entities": [ent], "relations": [], "specs": [], "mentions": [], "stats": {}}
        pid = point_id_for("v0", entity_id("a"))
        upserts: dict[str, list] = {}

        def fake_q(base_model):
            base = {pid: SimpleNamespace(id=pid, payload={"graph_type": "entity", "title": "A", "description": "da",
                                                           "embed_sha": embed_sha("A: da"), "embed_model": base_model}, vector=[9.0] * 4)}

            class FakeQ:
                def get_collections(self): return SimpleNamespace(collections=[])
                def create_collection(self, collection_name, vectors_config): pass
                def get_collection(self, collection_name):
                    return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=4))))
                def create_payload_index(self, **kw): pass
                def upsert(self, collection_name, points, wait): upserts.setdefault(collection_name, []).extend(points)
                def scroll(self, collection_name, limit, offset, with_payload, with_vectors): return list(upserts.get(collection_name, [])), None
                def delete(self, collection_name, points_selector, wait): pass
                def count(self, collection_name, exact): return SimpleNamespace(count=len(upserts.get(collection_name, [])))
                def retrieve(self, collection_name, ids, with_payload, with_vectors): return [base[i] for i in ids if i in base]
            return FakeQ()

        embedded: list[str] = []

        def embed(texts):
            embedded.extend(texts)
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

        reuse = {"entity": "graph_003_entity__v0", "relation": "graph_003_relation__v0", "spec": "graph_003_spec__v0"}
        common = dict(kb_id="kb_003", source_collection="kb_003", bundle=bundle, units_by_id={}, vector_size=4,
                      reuse_from=reuse, base_version="v0")
        s = write_graph_vectors(fake_q("m-old"), embed, graph_version="v1", embed_model="m-old", **common)
        self.assertEqual((embedded, s["collections"]["entity"]["reused"]), ([], 1))          # same model: reused
        upserts.clear()
        s = write_graph_vectors(fake_q("m-old"), embed, graph_version="v2", embed_model="m-new", **common)
        self.assertEqual((embedded, s["collections"]["entity"]["reused"]), (["A: da"], 0))    # model changed: re-embedded
        self.assertEqual(upserts["graph_003_entity__v2"][0].payload["embed_model"], "m-new")
        embedded.clear(); upserts.clear()
        s = write_graph_vectors(fake_q(""), embed, graph_version="v3", embed_model="m-new", **common)
        self.assertEqual(embedded, ["A: da"])                                                # old point recorded no model: not reused

    def test_build_lock_follows_the_holder_process(self) -> None:
        """2026-09-09 14:08: a manual `kb graph build` held the lock while the timer's check guessed whether
        the lock was stale by "pid alive + command line looks like one of ours"; a wrong guess meant two
        builds running side by side. The lock is now a flock: it counts only while a live process holds it;
        if the holder is SIGKILLed the kernel releases it; a leftover pid file is not a lock; the lock file
        is never deleted."""
        from kb_pipeline.graph.lock import GraphBuildLock, build_lock_held, build_lock_holder

        with tempfile.TemporaryDirectory() as tmp:
            cfg = SimpleNamespace(runtime_dir=Path(tmp))
            lock_dir = Path(tmp) / "state" / "graph_build.lock.d"
            self.assertFalse(build_lock_held(lock_dir))                     # never built a graph
            lock = GraphBuildLock(cfg)
            lock.acquire()
            self.assertTrue(build_lock_held(lock_dir))
            self.assertEqual(build_lock_holder(lock_dir)["pid"], os.getpid())
            with self.assertRaises(RuntimeError):
                GraphBuildLock(cfg).acquire()                                # only one at a time
            lock.release()
            self.assertFalse(build_lock_held(lock_dir))
            self.assertTrue((lock_dir / "lock").exists())                   # the lock file is never deleted
            self.assertFalse((lock_dir / "pid").exists())
            # Another process holds the lock and is SIGKILLed: the kernel releases it without any pid /
            # command-line check
            app_dir = Path(__file__).resolve().parents[1]
            code = ("import sys, time; sys.path.insert(0, %r); from types import SimpleNamespace; from pathlib import Path; "
                    "from kb_pipeline.graph.lock import GraphBuildLock; "
                    "GraphBuildLock(SimpleNamespace(runtime_dir=Path(%r))).acquire(); print('held', flush=True); time.sleep(60)"
                    % (str(app_dir), tmp))
            proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
            try:
                self.assertEqual(proc.stdout.readline().strip(), "held")
                self.assertTrue(build_lock_held(lock_dir))
                self.assertNotEqual(build_lock_holder(lock_dir).get("pid"), os.getpid())
            finally:
                proc.kill()
                proc.wait(timeout=10)
            self.assertFalse(build_lock_held(lock_dir))
            self.assertTrue((lock_dir / "pid").exists())                    # the pid file it had no time to delete stays, but is not a lock
            src = _repo_file("app/kb_pipeline/cli.py")
            self.assertIn("if build_lock_held(lock):", src)
            self.assertNotIn("remove_stale_pid_lock", src)
            self.assertNotIn("remove_stale_pid_lock", _repo_file("app/kb_pipeline/maintenance.py"))

    def test_frozen_input_must_match_the_vector_store(self) -> None:
        from kb_pipeline.graph.build import input_drift
        from kb_pipeline.graph.units import ChunkRef

        def ref(pid: str, text: str, ver: str = "v1") -> ChunkRef:
            return ChunkRef(point_id=pid, chunk_uid=f"u{pid}", doc_id="d", content_version=ver, chunk_index=0, block_id="b",
                            block_type="text", section_path=[], text=text, n_tokens=1)

        ledger = [{"point_id": "p1", "content_version": "v1", "text_sha": db.chunk_text_sha("alpha")},
                  {"point_id": "p2", "content_version": "v1", "text_sha": db.chunk_text_sha("beta")},
                  {"point_id": "p3", "content_version": "v2", "text_sha": db.chunk_text_sha("gamma")},
                  {"point_id": "p4", "content_version": "v1", "text_sha": ""}]
        drift = input_drift(ledger, [ref("p1", "alpha"), ref("p2", "beta CHANGED"), ref("p3", "gamma"), ref("p4", "whatever")])
        self.assertEqual(drift, {"missing": 0, "text_mismatch": 1, "version_mismatch": 1})
        self.assertEqual(input_drift(ledger[:1], []), {"missing": 1, "text_mismatch": 0, "version_mismatch": 0})
        self.assertEqual(input_drift(ledger[:2], [ref("p1", "alpha"), ref("p2", "beta")]),
                         {"missing": 0, "text_mismatch": 0, "version_mismatch": 0})
        self.assertIn("raise GraphInputIncomplete(", _repo_file("app/kb_pipeline/graph/build.py"))
