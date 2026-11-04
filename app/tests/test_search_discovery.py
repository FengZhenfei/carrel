"""Synthetic contracts for independent evidence retrieval and graph discovery."""
from dataclasses import replace
from types import SimpleNamespace
import time
import unittest
from unittest import mock

import test_search as helpers
from kb_search import channels, docstatus, graphwalk, service
from kb_search.config import load_search_settings
from kb_pipeline.graph import recall


class DiscoveryTests(unittest.TestCase):
    def run_search(self, mode="discovery", **kwargs):
        settings = {"graph_mode": mode, "mmr_enabled": False, "visual_enabled": False}
        settings.update(kwargs.pop("settings_over", {}) or {})
        return helpers.OrchestrationTests()._run(settings_over=settings, **kwargs)

    def test_evidence_scores_order_text_and_context_are_independent_of_discovery(self):
        baseline, base_calls = self.run_search("off")
        candidate, calls = self.run_search()
        self.assertEqual(candidate["sources"], baseline["sources"])
        self.assertEqual(candidate["doc_aggs"], baseline["doc_aggs"])
        self.assertEqual(calls["rerank_docs"], base_calls["rerank_docs"])
        self.assertEqual(base_calls["graph"], [])
        self.assertTrue(calls["graph"])
        self.assertTrue(candidate["specs"])
        self.assertTrue(all(row["role"] == "discovery" for row in candidate["specs"]))
        self.assertFalse(any("graph" in row["recall_sources"] for row in candidate["sources"]))
        self.assertEqual(candidate["retrieval_summary"]["graph"]["mode"], "discovery")

    def test_off_does_not_call_graph_or_neighborhoods(self):
        out, calls = self.run_search("off")
        self.assertEqual(calls["graph"], [])
        self.assertNotIn("hoods", calls)
        for key in ("entities", "relationships", "specs", "pages", "neighborhoods"):
            self.assertEqual(out[key], [])

    def test_timeout_does_not_lower_valid_text_confidence(self):
        out, _ = self.run_search(graph_delay=0.05, settings_over={"graph_budget": 0.005})
        summary = out["retrieval_summary"]
        self.assertEqual(summary["evidence_state"], "accepted")
        self.assertFalse(summary["low_confidence"])
        self.assertEqual(summary["degraded"], [])
        self.assertTrue(summary["graph"]["degraded"])
        self.assertEqual(summary["graph"]["state"], "partial")

    def test_weak_text_does_not_relabel_sourced_facts_as_irrelevant(self):
        out, _ = self.run_search(rerank_fn=lambda documents: [0.001] * len(documents), settings_over={"rerank_threshold": 0.1})
        self.assertEqual(out["retrieval_summary"]["evidence_state"], "diagnostic")
        self.assertTrue(out["specs"])
        self.assertTrue(out["specs"][0]["source_valid"])
        self.assertNotIn("state", out["specs"][0])

    def test_unverified_and_inactive_leads_are_distinct(self):
        entities = [{"id": "a", "point_ids": ["p1", "gone"], "score": 0.8},
                    {"id": "b", "point_ids": ["gone"], "score": 0.7},
                    {"id": "c", "score": 0.6}]
        out, _ = self.run_search(graph_entities=entities)
        rows = {r["id"]: r for r in out["entities"]}
        self.assertEqual(rows["a"]["source_state"], "partial")
        self.assertEqual(rows["a"]["point_ids"], ["p1"])
        self.assertEqual(rows["a"]["evidence"][0]["content_version"], "v1")
        self.assertEqual(rows["b"]["source_state"], "invalid")
        self.assertEqual(rows["b"]["evidence"], [])
        self.assertEqual(rows["c"]["source_state"], "unverified")
        self.assertNotIn("source_valid", rows["c"])

    def test_fact_cutoff_keeps_later_retrospective_documents(self):
        for mode in ("off", "discovery", "fused"):
            with self.subTest(mode=mode):
                out, calls = self.run_search(mode, as_of="2025-12-31", doc_dates={("kb_001", "d1"): "2026-01-01"})
                later = [r for r in out["sources"] if r["doc_id"] == "d1"]
                self.assertTrue(later)
                self.assertTrue(all(r["document_after_cutoff"] for r in later))
                self.assertTrue(any("浓度 433" in text for text in calls["rerank_docs"]))
                self.assertEqual(out["retrieval_summary"]["time"]["text_time_filter"], "not_applied")

    def test_explicit_document_hints_do_not_admit_unverified_graph_scope(self):
        entities = [{"id": "a", "docs": ["dir/d1.pdf"], "point_ids": ["p1"], "sources_truncated": True}]
        out, _ = self.run_search(hints={"rel_paths": ["dir/d1.pdf"]}, graph_entities=entities,
                                graph_relations=[{"id": "r", "point_ids": ["p3"]}])
        self.assertEqual(out["entities"], [])
        self.assertEqual(out["relationships"], [])
        self.assertTrue(all(r["rel_path"] == "dir/d1.pdf" for r in out["sources"]))

    def test_configuration_is_explicit_and_legacy_default_is_preserved(self):
        self.assertEqual(load_search_settings({}).graph_mode, "fused")
        self.assertEqual(load_search_settings({"KB_SEARCH_GRAPH_MODE": "discovery"}).graph_mode, "discovery")
        with self.assertRaises(ValueError):
            load_search_settings({"KB_SEARCH_GRAPH_MODE": "automatic-magic"})


class DiscoveryRecallTests(unittest.TestCase):
    def test_clause_limit_fallback_preserves_full_query_boosts_and_filters(self):
        from opensearchpy.exceptions import TransportError
        client = mock.Mock()
        client.search.side_effect = [TransportError(500, "too_many_nested_clauses"), {"hits": {"hits": [{"_id": "p", "_score": 1}]}}]
        filters = [{"term": {"doc_id": "doc"}}]
        with mock.patch.object(channels, "os_client", return_value=client), \
             mock.patch.object(channels.time, "monotonic", side_effect=[1.0, 1.5]):
            rows = channels.bm25_channel("http://test", "An unrestricted generic question", "test", limit=10,
                                         identifiers=["ABC-120"], filters=filters, timeout=3)
        self.assertEqual(rows[0]["point_id"], "p")
        before, after = [c.kwargs for c in client.search.call_args_list]
        self.assertEqual(after["request_timeout"], 2.5)
        self.assertEqual(after["body"]["query"]["bool"]["filter"], filters)
        clauses = after["body"]["query"]["bool"]["should"]
        self.assertEqual(clauses[0]["multi_match"]["query"], "An unrestricted generic question")
        self.assertTrue(any(c["multi_match"].get("boost") for c in clauses))
        self.assertEqual(len(before["body"]["query"]["bool"]["should"]), len(clauses) + 1)

    def test_unrelated_errors_and_exhausted_budget_are_not_retried(self):
        from opensearchpy.exceptions import TransportError
        for error, times in [(TransportError(401, "unauthorized"), [0.0]),
                             (TransportError(500, "maxClauseCount"), [0.0, 5.0])]:
            client = mock.Mock()
            client.search.side_effect = error
            with mock.patch.object(channels, "os_client", return_value=client), \
                 mock.patch.object(channels.time, "monotonic", side_effect=times):
                with self.assertRaises(TransportError):
                    channels.bm25_channel("http://test", "generic", "test", limit=10, timeout=2)
            self.assertEqual(client.search.call_count, 1)

    def test_zero_budget_keeps_all_structurally_bounded_records(self):
        from kb_search.evidence import budget_discovery
        facts = [{"id": "f1", "conditions": {"detail": "long condition " * 500}}]
        result, stats = budget_discovery({"specs": facts}, 0)
        self.assertEqual(result["specs"][0]["conditions"], facts[0]["conditions"])
        self.assertFalse(stats["records"]["specs"]["has_more"])
        self.assertEqual(load_search_settings({}).graph_context_tokens, 0)

    def test_output_budget_preserves_whole_claims_and_marks_omissions(self):
        from kb_search.evidence import budget_discovery
        facts = [{"id": "f1", "value": "12", "conditions": {"region": "north", "tax": "excluded"}},
                 {"id": "f2", "value": "15", "conditions": {"detail": "long condition " * 1000}}]
        result, stats = budget_discovery({"specs": facts}, 128)
        self.assertEqual(result["specs"][0]["conditions"], facts[0]["conditions"])
        self.assertEqual(len(result["specs"]), 1)
        self.assertTrue(stats["records"]["specs"]["has_more"])
        self.assertLessEqual(stats["estimated_tokens"], 128)

    def test_discovery_does_not_aggregate_or_retrieve_text_chunk_vectors(self):
        q, driver = helpers._GraphQdrant(), helpers._GraphDriver()
        with mock.patch.object(recall, "version_stats", side_effect=AssertionError("whole-graph statistics forbidden")), \
             mock.patch.object(recall, "discovery_sources") as provenance:
            result = helpers.GraphRecallTests()._query(q=q, driver=driver, include_chunks=False)
        self.assertTrue(result["entities"])
        self.assertTrue(result["relations"])
        self.assertEqual(result["chunks"], [])
        self.assertNotIn("retrieve:kb_001", q.calls)
        provenance.assert_called_once()

    def test_provenance_sample_is_bounded_and_marks_truncation(self):
        session = mock.Mock()
        session.run.return_value.data.return_value = [
            {"owner": "e1", "point_id": f"p{i}", "rel_path": f"doc{i}.md"} for i in range(recall.SOURCE_SAMPLE + 1)]
        entities = {"e1": {}}
        recall.discovery_sources(session, "kb_test", "v1", entities, {})
        self.assertEqual(len(entities["e1"]["point_ids"]), recall.SOURCE_SAMPLE)
        self.assertTrue(entities["e1"]["sources_truncated"])
        self.assertIn("LIMIT $sample", session.run.call_args.args[0])
        self.assertEqual(session.run.call_args.kwargs["sample"], recall.SOURCE_SAMPLE + 1)

    def test_deadline_stops_database_and_vector_reads(self):
        session, client = mock.Mock(), mock.Mock()
        with self.assertRaises(TimeoutError):
            recall.TimedSession(session, 5, deadline=time.monotonic() - 1).run("RETURN 1")
        with self.assertRaises(TimeoutError):
            recall._DeadlineClient(client, time.monotonic() - 1).retrieve(ids=["p"])
        session.run.assert_not_called()
        client.retrieve.assert_not_called()

    def test_multiple_vector_reads_share_one_deadline(self):
        client = mock.Mock()
        bounded = recall._DeadlineClient(client, 1.0)
        with mock.patch.object(recall.time, "monotonic", side_effect=[0.2, 1.1]):
            bounded.retrieve(ids=["first"])
            with self.assertRaises(TimeoutError):
                bounded.retrieve(ids=["second"])
        self.assertEqual(client.retrieve.call_count, 1)

    def test_neighborhood_name_lookups_share_one_deadline(self):
        driver = mock.MagicMock()
        def index(session, kb, version):
            session.run("RETURN 1")
            return {"entities": {}}
        with mock.patch.object(graphwalk, "name_index", side_effect=index), \
             mock.patch.object(graphwalk.time, "monotonic", side_effect=[0.0, 0.2, 1.1]):
            with self.assertRaises(TimeoutError):
                graphwalk.neighborhoods(SimpleNamespace(), {}, {"a": "v1", "b": "v1"}, "generic question", [],
                                        limit=3, driver=driver, timeout=1.0)
        self.assertEqual(driver.session.return_value.__enter__.return_value.run.call_count, 1)

    def test_document_date_queries_share_one_deadline(self):
        driver = mock.MagicMock()
        driver.session.return_value.__enter__.return_value.run.return_value.data.return_value = []
        rows = [{"kb_id": "a", "doc_id": "d1"}, {"kb_id": "b", "doc_id": "d2"}]
        with mock.patch.object(recall.time, "monotonic", side_effect=[0.0, 0.2, 1.1]):
            with self.assertRaises(TimeoutError):
                docstatus.document_dates(SimpleNamespace(), rows, {"a": "v1", "b": "v1"}, driver=driver, timeout=1.0)
        self.assertEqual(driver.session.return_value.__enter__.return_value.run.call_count, 1)

    def test_non_entity_seed_version_mismatch_is_rejected(self):
        q, driver = helpers._GraphQdrant(), helpers._GraphDriver()
        def wrong(*args, **kwargs):
            return [{"gr_id": "stale", "graph_version": "older", "_score": 0.8, "point_ids": []}]
        with mock.patch.object(recall, "spec_seeds", side_effect=wrong):
            with self.assertRaisesRegex(RuntimeError, "disagree"):
                helpers.GraphRecallTests()._query(q=q, driver=driver, include_chunks=False)


class IdentityAndDateTests(unittest.TestCase):
    def test_fact_time_not_publication_time_controls_retrospective_assertions(self):
        from kb_pipeline.graph.currency import in_scope, query_scope
        scope = query_scope(as_of="2025")
        self.assertTrue(in_scope({"valid_from": "2024", "observed": "2026-01-01"}, scope))
        self.assertFalse(in_scope({"valid_from": "2026", "observed": "2026-01-01"}, scope))
        self.assertTrue(in_scope({"valid_from": "2025-12-31", "observed": "2026-01-01"}, scope))

    def test_ambiguous_name_never_selects_by_connection_count(self):
        matches = [{"id": "a", "title": "Mercury", "type": "product", "scope": "division-a", "degree": 300},
                   {"id": "b", "title": "Mercury", "type": "project", "scope": "division-b", "degree": 1}]
        driver = mock.MagicMock()
        source = SimpleNamespace(kb_id="kb_test", collection="test")
        with mock.patch("kb_pipeline.graph.neo4j_import.active_neo4j_graph_version", return_value="v1"), \
             mock.patch.object(graphwalk, "resolve_entity", return_value=matches), \
             mock.patch.object(graphwalk, "neighbor_rows") as expansion:
            result = graphwalk.neighbors(SimpleNamespace(), source, entity="Mercury", entity_id=None, driver=driver)
            facts = graphwalk.list_facts(SimpleNamespace(), source, subject="Mercury", driver=driver)
        expansion.assert_not_called()
        for result in (result, facts):
            self.assertFalse(result["found"])
            self.assertTrue(result["ambiguous"])
            self.assertEqual({row["id"] for row in result["candidates"]}, {"a", "b"})

    def test_source_date_annotations_do_not_confuse_publication_and_fact_time(self):
        rows = [{"kb_id": "kb", "doc_id": name, "rel_path": name + ".md"} for name in ("past", "equal", "future", "unknown")]
        dates = {("kb", "past"): "2020-01-01", ("kb", "equal"): "2025-06-30", ("kb", "future"): "2025-07-01"}
        with mock.patch.object(docstatus, "document_dates", return_value=dates):
            result, stats = docstatus.annotate_dates(SimpleNamespace(), rows, {"as_of": True, "day": "2025-06-30"})
        self.assertEqual([r["doc_id"] for r in result], ["past", "equal", "future", "unknown"])
        self.assertEqual(result[-1]["doc_date_state"], "unknown")
        self.assertTrue(result[2]["document_after_cutoff"])
        self.assertEqual(stats, {"documents_after_cutoff": 1, "unknown_dates": 1, "text_time_filter": "not_applied"})

    def test_context_can_read_later_retrospective_document(self):
        settings = SimpleNamespace(sources={"kb": SimpleNamespace(kb_id="kb", collection="test", source_root="docs")})
        payload = {"doc_id": "d", "point_id": "p", "rel_path": "2027-01-01.md", "text": "A later statement", "chunk_index": 1}
        with mock.patch.object(service, "runtime", return_value=(settings, load_search_settings({}), object())), \
             mock.patch.object(channels, "context_range", return_value=[payload]), \
             mock.patch.object(service, "_graph_driver", return_value=None), \
             mock.patch.object(service.catalog_mod, "get_catalog", return_value=[]):
            result = service.context("kb", "d", 1, 1, as_of="2026")
        self.assertEqual(result["sources"][0]["text"], "A later statement")
        self.assertTrue(result["sources"][0]["document_after_cutoff"])
        self.assertEqual(result["time"]["documents_after_cutoff"], 1)


if __name__ == "__main__":
    unittest.main()
