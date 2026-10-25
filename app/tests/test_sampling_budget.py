"""Regression coverage for model capacity, automatic sampling and request limits."""
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from kb_pipeline import db
from kb_pipeline.graph import schema
from kb_pipeline.graph.llm import ChatClient, LLMInputBudgetExceeded, LLMSpec
from kb_pipeline.graph.sampling_budget import estimate_messages, estimate_text, sampling_budget
from kb_server import service


class ModelCapacityTests(unittest.TestCase):
    def test_old_database_migrates_without_inventing_model_capacity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.db"
            with sqlite3.connect(path) as con:
                con.execute("CREATE TABLE llm_registry (name TEXT PRIMARY KEY, base_url TEXT NOT NULL, "
                            "api_key TEXT NOT NULL DEFAULT '', model_id TEXT NOT NULL, notes TEXT NOT NULL DEFAULT '', "
                            "builtin INTEGER NOT NULL DEFAULT 0, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL)")
                con.execute("INSERT INTO llm_registry VALUES ('old','http://x','','m','',0,1,1)")
            db.init_db(path)
            db.init_db(path)
            with db.connect(path) as con:
                self.assertIsNone(db.get_llm(con, "old")["max_input_tokens"])
                db.upsert_llm(con, name="new", base_url="http://x", api_key="secret", model_id="m", max_input_tokens=1048576)
                spec = LLMSpec.from_row(db.get_llm(con, "new"))
            self.assertEqual(spec.max_input_tokens, 1048576)
            self.assertEqual(sampling_budget(spec)["input_token_limit"], 314572)

    def test_save_roundtrip_validation_and_model_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = SimpleNamespace(state_db=Path(tmp) / "s.db")
            db.init_db(settings.state_db)
            payload = {"name": "m", "base_url": "http://x", "model_id": "one", "api_key": "secret", "max_input_tokens": 128000}
            with mock.patch.object(service, "settings", return_value=settings), mock.patch.object(service, "_probe_llm") as probe:
                saved = service.save_llm(payload)
                self.assertEqual(saved["max_input_tokens"], 128000)
                self.assertNotIn("api_key", saved)
                for bad in (0, -1, True, 1.2, "", "1M", "128000.0", 2**63):
                    probe.reset_mock()
                    with self.assertRaises(ValueError):
                        service.save_llm({**payload, "max_input_tokens": bad})
                    probe.assert_not_called()
                old_client = {k: v for k, v in payload.items() if k != "max_input_tokens"}
                self.assertEqual(service.save_llm(old_client)["max_input_tokens"], 128000)
                # A different model must not inherit an unrelated capacity.
                self.assertIsNone(service.save_llm({**old_client, "model_id": "two"})["max_input_tokens"])


class AutomaticSamplingTests(unittest.TestCase):
    @staticmethod
    def chunks(docs=400, per_doc=3):
        return [{"doc_id": f"d{d:04}", "chunk_index": i, "chunk_uid": f"c{d}-{i}", "point_id": f"p{d}-{i}",
                 "text": f"Document {d} section {i}: specifications for memory voltage and operating temperature."}
                for d in range(docs) for i in range(per_doc)]

    def test_more_than_300_documents_are_eligible_and_reproducible(self):
        chunks = self.chunks()
        result = schema.sample_chunks(chunks, 0, budget=20000)
        repeated = schema.sample_chunks(list(reversed(chunks)), 0, budget=20000)
        self.assertEqual(result["texts"], repeated["texts"])
        self.assertEqual(result["documents_covered"], 400)
        self.assertLessEqual(result["estimated_tokens"], 20000)
        self.assertGreater(len(result["texts"]), 32)
        self.assertIn("d0399", {c["doc_id"] for c in result["picked"]})

    def test_oversized_first_chunk_is_skipped_and_provenance_matches(self):
        chunks = self.chunks(docs=1, per_doc=1)
        chunks += [{"doc_id": "large", "chunk_uid": "large", "text": "oversized " * 5000}]
        result = schema.sample_chunks(chunks, 0, budget=200)
        self.assertEqual(result["texts"], [c["text"] for c in result["picked"]])
        self.assertEqual(len(result["texts"]), 1)
        self.assertLessEqual(result["estimated_tokens"], 200)
        self.assertEqual(schema.sample_chunks(chunks, 0, budget=1)["texts"], [])

    def test_selected_model_capacity_changes_automatic_count(self):
        chunks = self.chunks(docs=10, per_doc=100)
        counts = []
        for capacity in (32000, 128000):
            budget = sampling_budget(SimpleNamespace(max_input_tokens=capacity))
            result = schema.sample_chunks(chunks, 0, budget=budget["sample_token_budget"])
            counts.append(len(result["texts"]))
            self.assertLessEqual(result["estimated_tokens"], budget["sample_token_budget"])
        self.assertGreater(counts[1], counts[0])

    def test_small_corpus_fits_and_vector_selection_keeps_all_docs(self):
        chunks = self.chunks(docs=6, per_doc=10)
        vectors = lambda pool: {c["point_id"]: [1.0, float(c["chunk_index"])] for c in pool}
        result = schema.sample_chunks(chunks, 0, budget=500, fetch_vectors=vectors)
        self.assertEqual(result["documents_covered"], 6)
        self.assertEqual(result["method"], "vectors")
        whole = schema.sample_chunks(chunks, 0, budget=100000)
        self.assertEqual(len(whole["texts"]), len(chunks))
        self.assertEqual(whole["stop_reason"], "corpus_covered")


class RequestBudgetTests(unittest.TestCase):
    def test_all_messages_and_retry_history_count_before_network(self):
        spec = LLMSpec("m", "http://unused", "", "m", max_input_tokens=1000)
        messages = [{"role": "system", "content": "system " * 50}, {"role": "user", "content": "body " * 50}]
        network = mock.Mock(return_value="bad format " * 100)
        limit = estimate_messages(messages)
        client = ChatClient(spec, chat=network, input_token_limit=limit, attempts=3)
        with self.assertRaises(LLMInputBudgetExceeded):
            client.chat(messages, validate=lambda _: False)
        self.assertEqual(network.call_count, 1)  # Oversized repair request never goes out.
        with self.assertRaises(LLMInputBudgetExceeded):
            client.chat(messages + [{"role": "user", "content": "more"}])
        self.assertEqual(network.call_count, 1)

    def test_unknown_capacity_is_explicit_and_not_assumed_to_be_deepseek(self):
        with self.assertRaisesRegex(ValueError, "Max Input"):
            sampling_budget(LLMSpec("unknown", "http://unused", "", "unknown"))


if __name__ == "__main__":
    unittest.main()
