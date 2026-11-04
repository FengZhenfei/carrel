"""Readback admits only scoped, current originals and leaves the direct ranks alone."""
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from kb_search import readback
from kb_search.config import load_search_settings
import test_search as helpers


def lead(pid, doc="a"):
    return {"point_id": pid, "doc_id": doc, "content_version": "v1"}


def package(locators):
    return {"question": "Which steps are required?", "kbs": ["k"], "sources": [], "entities": [],
            "relationships": [], "specs": [{"kb_id": "k", "evidence": locators}]}


class ReadbackTests(unittest.TestCase):
    def test_stitched_pieces_and_duplicate_leads_do_not_displace_other_documents(self):
        data = package([lead("head"), lead("piece"), lead("first"), lead("first"),
                        *[lead(str(i)) for i in range(30)], lead("second", "b")])
        data["sources"] = [{"kb_id": "k", "point_id": "head", "doc_id": "a",
                            "stitched": {"pieces": [{"point_id": "piece", "kept": True}]}}]
        ids = [locator["point_id"] for _, locator in readback.candidates(data["question"], data)]
        self.assertEqual(len(ids), 24)
        self.assertNotIn("head", ids)
        self.assertNotIn("piece", ids)
        self.assertEqual(ids.count("first"), 1)
        self.assertIn("second", ids[:2])

    def test_validate_hydrated_version_scope_and_activity_before_ranking(self):
        settings = SimpleNamespace(sources={"k": SimpleNamespace(collection="collection")},
                                   reranker_base_url="http://ranker")
        names = ["ok", "changed", "retired", "outside", "moved", "archived", "copy"]
        data = package([lead(pid) for pid in names])
        data["sources"] = [{"n": 1, "kb_id": "k", "point_id": "base", "doc_id": "a",
                            "text": "Previously returned passage."}]
        points = []
        for name in names:
            payload = dict(doc_id="other" if name=="moved" else "a", is_active=name!="retired",
                           content_version="v2" if name=="changed" else "v1",
                           rel_path="other/file" if name=="outside" else "allowed/file",
                           text="Previously returned passage." if name=="copy" else "Answer in "+name)
            points.append(SimpleNamespace(id=name, payload=payload))
        q = Mock()
        q.retrieve.return_value = points
        points[5].payload["rel_path"] = "retired/file"
        with patch.object(readback.docstatus, "allowed", side_effect=lambda _, __, path, scope: path != "retired/file"), \
             patch.object(readback.docstatus, "annotate"), patch.object(readback, "rerank",
                    side_effect=lambda ranker, question, records, **kw: [0.8]*len(records)) as ranking:
            rows, info = readback.read_originals(settings, load_search_settings({}), q, data, {},
                    {"rel_paths": ["allowed/file", "retired/file"]}, time.time()+5, None)
        self.assertEqual([i["point_id"] for i in ranking.call_args.args[2]], ["ok", "copy"])
        self.assertEqual([(s["point_id"],s["n"],s["role"]) for s in rows], [("ok",2,"graph_followup")])
        self.assertEqual(info["source_numbers"], [2])
        self.assertEqual(len(data["sources"]), 1)

    def test_image_neighborhood_is_independent_of_hit_ranking(self):
        runner = helpers.OrchestrationTests()
        args = dict(image=b"fixture", kbs=["kb_001"],
                    entries=[{"kb_id": "kb_001", "has_graph": True, "graph_version": "v1"}])
        baseline, _ = runner._run(settings_over={"graph_mode": "off"}, **args)
        block = {"id": "e", "title": "Example", "kb_id": "kb_001", "neighbors": []}
        actual, calls = runner._run(hoods=[block], **args)
        self.assertEqual(actual["sources"], baseline["sources"])
        self.assertEqual(calls["graph"], [])
        self.assertEqual(calls["hoods"]["limit"], 2)
        self.assertEqual(calls["hoods"]["options"]["per_entity"], 2)
        eligible = [(r["kb_id"], r["point_id"]) for r in baseline["sources"]
                    if r["role"] == "hit" and r["accepted"]]
        self.assertTrue(eligible)
        self.assertEqual(calls["hoods"]["options"]["hit_points"], eligible[:3])
        self.assertEqual(actual["neighborhoods"][0]["role"], "discovery")
        self.assertEqual(actual["retrieval_summary"]["graph"]["mode"], "neighborhood")
        for field in ("specs", "pages", "relationships", "entities"):
            self.assertEqual(actual[field], [])
        self.assertNotIn("readback", actual["retrieval_summary"]["graph"])

        def fail():
            raise TimeoutError()
        broken, _ = runner._run(hoods=fail, **args)
        self.assertEqual(broken["sources"], baseline["sources"])
        self.assertEqual(broken["retrieval_summary"]["degraded"], baseline["retrieval_summary"]["degraded"])
        self.assertEqual(broken["retrieval_summary"]["graph"]["state"], "partial")
        scoped, calls = runner._run(hints={"doc_ids": ["d1"]}, **args)
        self.assertEqual(scoped["neighborhoods"], [])
        self.assertEqual(calls["graph"], [])
        self.assertNotIn("hoods", calls)

    def test_text_profile_preserves_image_request_ranking_and_graph_mode(self):
        runner = helpers.OrchestrationTests()
        baseline,_ = runner._run(image=b"fixture")
        actual,_ = runner._run(image=b"fixture", settings_over={"text_discovery": True})
        self.assertEqual(actual["sources"], baseline["sources"])
        self.assertEqual(actual["retrieval_summary"]["graph"], baseline["retrieval_summary"]["graph"])
