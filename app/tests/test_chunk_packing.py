"""Synthetic structural, fidelity and downstream checks for short-chunk packing."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from kb_pipeline.chunking.chunker import blocks_to_chunks, embedding_input
from kb_pipeline.chunking.context import evidence_text
from kb_pipeline.chunking.packing import pack_prose
from kb_pipeline.models import ParsedBlock
from kb_pipeline.utils import count_tokens


def block(text, bid="a", **kw):
    return ParsedBlock("fixture", "v1", kw.pop("doc_type", "pdf"), kw.pop("block_type", "text"),
                       text, bid, **kw)


def chunks(bs, limit=128):
    return blocks_to_chunks(kb_id="fixture", file_key=1, content_version="v", parser_profile="v1",
                            blocks=bs, max_tokens=limit, overlap_tokens=0)


@pytest.mark.parametrize("kind", ["pdf", "docx", "pptx", "md", "rst", "txt"])
def test_prose_packing_preserves_order_and_all_member_locations(kind):
    bs = [block(f"Statement {i}: " + "value " * 16 + ".", str(i), doc_type=kind,
                page_idx=i + 1, metadata={"section_path": ["Procedure"],
                                         "source_locator": {"kind": "lines", "start": 10*i+1, "end": 10*i+5}})
          for i in range(3)]
    old = deepcopy(bs)
    cs = chunks(bs)
    assert len(cs) == 1 and cs[0].token_count <= 128
    c = cs[0]
    assert c.text.index("Statement 0") < c.text.index("Statement 1") < c.text.index("Statement 2")
    assert c.block.page_idx == 1 and c.block.metadata["page_end"] == 3
    assert [r["block_id"] for r in c.block.metadata["source_regions"]] == ["0", "1", "2"]
    assert c.block.metadata["source_locator"] == {"kind": "lines", "start": 1, "end": 25}
    assert bs == old


@pytest.mark.parametrize("boundary", ["section", "slide", "table", "image", "code", "field"])
def test_new_packing_does_not_cross_independent_structure(boundary):
    a = block("left " * 25, "a", metadata={"section_path": ["A"]})
    b = block("right " * 25, "b", metadata={"section_path": ["A"]})
    if boundary == "section": b.metadata["section_path"] = ["B"]
    elif boundary == "slide": a.slide_idx, b.slide_idx = 1, 2
    elif boundary == "field": a.metadata["source_field"], b.metadata["source_field"] = "first", "second"
    else: b.block_type = boundary
    ps = [{"text": x.text, "block": x, "tokens": count_tokens(x.text)} for x in (a, b)]
    assert pack_prose(ps, 128) == ps


def test_balanced_partition_avoids_a_tiny_last_piece_without_exceeding_budget():
    ps = [{"text": "word " * n, "block": block("", str(i)), "tokens": n+1}
          for i, n in enumerate([45, 45, 10, 10])]
    cs = pack_prose(ps, 100)
    assert len(cs) == 2
    assert all(40 <= c["tokens"] <= 100 for c in cs)
    assert "".join(c["text"].replace("\n", "") for c in cs) == "".join(p["text"] for p in ps)


def test_embedded_intro_is_searchable_while_code_stays_literal():
    raw = "    result = emit('keep. literal!')  \n" * 16
    b = block("A short explanation.\n```python\n" + raw + "```", doc_type="md",
              metadata={"section_path": ["Example"]})
    cs = chunks([b])
    code = [c for c in cs if c.block.block_type == "code"]
    assert "".join(c.text for c in code) == raw
    assert all(c.token_count == count_tokens(evidence_text(c.text, c.block.metadata)) <= 128 for c in code)
    assert all("A short explanation." in embedding_input(c) for c in code)
    assert all(c.block.metadata["code_context_sources"][0]["embedded_source"]["block_id"] == "a" for c in code)
    for c in code:
        loc = c.block.metadata["code_fragment"]
        assert raw[loc["start"]:loc["end"]] == c.text


@pytest.mark.parametrize("bad", ["column", "page", "section", "symbol", "missing_geometry"])
def test_ambiguous_pdf_adjacency_is_not_attached(bad):
    a = block("This explanation belongs here.", page_idx=1, bbox=[0, 10, 400, 50], metadata={"section_path": ["A"]})
    b = block("return value\n", "code", block_type="code", page_idx=1, bbox=[0, 60, 400, 100], metadata={"section_path": ["A"]})
    if bad == "column": b.bbox = [500, 60, 900, 100]
    elif bad == "page": b.page_idx = 2
    elif bad == "section": b.metadata["section_path"] = ["B"]
    elif bad == "symbol": b.metadata["symbol"] = {"kind": "function"}
    else: a.bbox = None
    cs = chunks([a, b])
    assert len(cs) == 2 and not cs[1].block.metadata.get("code_context")


def test_all_consumers_and_incremental_graph_fingerprint_use_context():
    from kb_pipeline import db, search_fts
    from kb_pipeline.graph.units import fetch_chunk_payloads, build_units
    from kb_search.evidence import source_row
    from kb_search.rerank import rerank_documents
    raw = "    return value\n"
    c = chunks([block(raw, "listing", block_type="code", caption="Example operation", page_idx=2)])[0]
    body = evidence_text(c.text, c.block.metadata)
    p = {**c.block.metadata, "text": c.text, "block_type": "code", "chunk_uid": c.chunk_uid,
         "chunk_index": 0, "doc_id": "doc", "token_count": c.token_count, "is_active": True}
    assert embedding_input(c) == body
    assert search_fts.doc_source(p)["body"] == body
    assert "code_context" in search_fts.PAYLOAD_FIELDS
    assert source_row(1, {}, p)["text"] == body
    docs, _ = rerank_documents([{"payload": p}], window_tokens=128, overlap_tokens=0)
    assert "Example operation" in docs[0] and "return value" in docs[0]
    q = SimpleNamespace(retrieve=lambda **kw: [SimpleNamespace(id="p", payload=p)])
    refs = fetch_chunk_payloads(q, "fixture", [{"point_id": "p"}])
    units = build_units(refs, kb_id="fixture", unit_chunks=1)
    assert "Example operation" in units[0].text and "return value" in units[0].text
    assert units[0].source_locations[0]["context_sources"] == c.block.metadata["code_context_sources"]
    con = Mock()
    db.replace_chunks(con, file_id="f", collection="fixture", content_version="v", chunks=[c], point_ids=["p"])
    fingerprint = con.executemany.call_args.args[1][0][-1]
    assert fingerprint == db.chunk_text_sha(body) != db.chunk_text_sha(raw)
    c.block = replace(c.block, metadata={**c.block.metadata, "code_context": "Another meaning"})
    db.replace_chunks(con, file_id="f", collection="fixture", content_version="v", chunks=[c], point_ids=["p"])
    assert con.executemany.call_args.args[1][0][-1] != fingerprint


def test_long_caption_remains_independent_and_never_disappears():
    caption = "This is a complete explanation. " * 30
    cs = chunks([block("value = 7\n", block_type="code", caption=caption)])
    assert any(c.block.block_type == "text" for c in cs)
    assert "".join(c.text for c in cs if c.block.block_type == "code") == "value = 7\n"
    assert all(not c.block.metadata.get("code_context") for c in cs)


def test_same_page_pdf_explanation_keeps_literal_code_and_both_locations():
    a = block("The following function converts values before storage.", "intro",
              page_idx=3, bbox=[100, 600, 800, 700], metadata={"section_path": ["Procedure"]})
    b = block("def convert(value):\n    return value * 2\n", "code", block_type="code",
              page_idx=3, bbox=[100, 710, 800, 900], metadata={"section_path": ["Procedure"]})
    original = deepcopy([a, b])
    cs = chunks([a, b], limit=160)
    assert len(cs) == 1
    assert cs[0].text == b.text
    assert cs[0].block.metadata["code_context"] == a.text
    assert cs[0].block.metadata["code_context_sources"][0]["page"] == 3
    assert cs[0].block.bbox == b.bbox
    assert [a, b] == original
