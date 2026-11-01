"""Synthetic section ownership checks independent of chunk-size policy."""
from copy import deepcopy
from dataclasses import replace

import pytest

from kb_pipeline.models import ParsedBlock
from kb_pipeline.parsers.pdf_enhanced import merge_mineru_text_blocks
from kb_pipeline.chunking.chunker import blocks_to_chunks
from kb_pipeline.chunking.sections import SectionSources, resolve_sections
from kb_pipeline.graph.units import ChunkRef, build_units


def block(text, bid, section, page=1, kind="text"):
    return ParsedBlock("fixture", "fixture-v1", "pdf", kind, text, bid, page_idx=page,
                       metadata={"section_path": section})


def chunks(blocks, size=64, overlap=12):
    return blocks_to_chunks(kb_id="fixture", file_key=1, content_version="v1", parser_profile="v1",
                            blocks=blocks, max_tokens=size, overlap_tokens=overlap)


@pytest.mark.parametrize("overlap", [0, 12, 24])
def test_late_section_does_not_inherit_first_section(overlap):
    raw = [block("Alpha statement. " * 35, "a", ["Manual", "Alpha"]),
           block("Beta statement. " * 35, "b", ["Manual", "Beta"], page=2)]
    merged = merge_mineru_text_blocks(raw, target_tokens=1024)
    assert len(merged) == 1
    baseline = deepcopy(merged)
    for b in baseline: b.metadata.pop("section_spans")
    before, after = chunks(baseline, overlap=overlap), chunks(merged, overlap=overlap)
    assert [(c.chunk_uid, c.text) for c in before] == [(c.chunk_uid, c.text) for c in after]
    for c in after:
        paths = c.block.metadata["section_paths"]
        if "Alpha" in c.text: assert ["Manual", "Alpha"] in paths
        if "Beta" in c.text: assert ["Manual", "Beta"] in paths
        if "Beta" in c.text and "Alpha" not in c.text:
            assert c.block.metadata["section_path"] == ["Manual", "Beta"]


def test_packed_mixed_sections_retain_all_paths_through_graph_input():
    raw = [block("Alpha uses the red route.", "a", ["Manual", "Alpha"]),
           block("Beta uses the blue route.", "b", ["Manual", "Beta"])]
    cs = chunks(merge_mineru_text_blocks(raw, target_tokens=512), size=128)
    assert len(cs) == 1
    m = cs[0].block.metadata
    assert m["section_path"] == ["Manual"]
    ref = ChunkRef("p", "c", "d", "v", 0, "b", "text", m["section_path"], cs[0].text, 30,
                   section_paths=m["section_paths"])
    u = build_units([ref], kb_id="fixture")[0]
    assert "Manual > Alpha" in u.section_label and "Manual > Beta" in u.section_label
    assert u.text == cs[0].text
    changed = build_units([replace(ref, section_paths=[["Manual", "Other"]])], kb_id="fixture")[0]
    assert changed.unit_id != u.unit_id  # stale section-dependent graph output cannot be reused


def test_identical_passages_in_different_sections_match_in_source_order():
    raw = [block("Repeat this procedure.", "a", ["A"]),
           block("Repeat this procedure.", "b", ["B"])]
    merged = merge_mineru_text_blocks(raw, target_tokens=512)[0]
    sources = SectionSources([merged])
    a = resolve_sections(sources.locate(merged, raw[0].text))
    b = resolve_sections(sources.locate(merged, raw[1].text))
    assert a.metadata["section_path"] == ["A"]
    assert b.metadata["section_path"] == ["B"]


def test_unknown_section_is_not_assigned_to_neighbor_and_unmatched_is_explicit():
    raw = [block("Known paragraph.", "a", ["Known"]), block("Unlabelled paragraph.", "b", [])]
    merged = merge_mineru_text_blocks(raw, target_tokens=512)[0]
    sources = SectionSources([merged])
    unknown = resolve_sections(sources.locate(merged, raw[1].text))
    assert unknown.metadata["section_path"] == []
    unresolved = resolve_sections(sources.locate(merged, "Unmatched transformation"))
    assert unresolved.metadata["section_path"] == []
    assert unresolved.metadata["section_match"] == "ambiguous"
    assert unresolved.metadata["section_paths"] == [["Known"], []]


def test_fenced_code_keeps_literal_body_and_its_own_section():
    raw = [block("Earlier explanation.", "a", ["A"]),
           block("```python\nanswer = 42\n```", "b", ["B"])]
    cs = chunks(merge_mineru_text_blocks(raw, target_tokens=512))
    code = next(c for c in cs if c.block.block_type == "code")
    assert code.text == "answer = 42\n"
    assert code.block.metadata["section_path"] == ["B"]
    assert not code.block.metadata.get("code_context")
    assert any(c.text == "Earlier explanation." for c in cs)


def test_same_section_cross_page_continuation_and_source_blocks_are_preserved():
    raw = [block("First sentence.", "a", ["A"]), block("Second sentence.", "b", ["A"], page=2)]
    old = deepcopy(raw)
    cs = chunks(merge_mineru_text_blocks(raw, target_tokens=512))
    assert raw == old and len(cs) == 1
    assert cs[0].block.metadata["section_path"] == ["A"]
    assert cs[0].block.metadata["page_end"] == 2
    assert cs[0].text.split() == "First sentence. Second sentence.".split()


def test_crossed_boilerplate_heading_does_not_suppress_adjacent_body():
    refs = [ChunkRef("p1", "c1", "d", "v", 0, "b1", "text", [],
                     "Copyright 2025. All rights reserved.", 10, section_paths=[["Copyright"]]),
            ChunkRef("p2", "c2", "d", "v", 1, "b2", "text", ["Method"],
                     "The actuator moves the gate after the sensor confirms clearance.", 20,
                     section_paths=[["Method"]])]
    u = build_units(refs, kb_id="fixture")[0]
    assert u.kind == "body" and "actuator" in u.text
    assert set(u.chunk_uids) == {"c1", "c2"}
    assert build_units(refs[:1], kb_id="fixture")[0].kind == "boilerplate"


def test_repeated_ambiguous_fragment_does_not_guess_a_later_section():
    raw = [block("Intro. Repeat this. Repeat this.", "a", ["A"]),
           block("Repeat this.", "b", ["B"])]
    merged = merge_mineru_text_blocks(raw, target_tokens=512)[0]
    out = resolve_sections(SectionSources([merged]).locate(merged, "Repeat this."))
    assert out.metadata["section_path"] == []
    assert out.metadata["section_paths"] == [["A"], ["B"]]
    assert out.metadata["section_match"] == "ambiguous"
