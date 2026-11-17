"""Composition checks: packing must preserve established source ownership."""
from copy import deepcopy

import pytest

from kb_pipeline.models import ParsedBlock
from kb_pipeline.parsers.pdf_enhanced import merge_mineru_text_blocks
from kb_pipeline.chunking.chunker import blocks_to_chunks


def block(text, bid, section, kind="text", page=1, box=None):
    return ParsedBlock("fixture", "v1", "pdf", kind, text, bid, page_idx=page,
                       bbox=box, metadata={"section_path": section})


def chunks(raw, budget=400):
    merged = merge_mineru_text_blocks(deepcopy(raw), target_tokens=800)
    return blocks_to_chunks(kb_id="fixture", file_key=1, content_version="v1", parser_profile="v1",
                            blocks=merged, max_tokens=budget, overlap_tokens=0)


@pytest.mark.parametrize("heading", ["Configuration", "配置说明", "2. Configuration"])
@pytest.mark.parametrize("budget", [128, 400])
def test_new_heading_never_returns_to_previous_section(heading, budget):
    raw = [block("Previous section explains Alpha.", "a", ["A"], box=[10, 10, 400, 80]),
           block(heading, "b", ["B"], "title", box=[10, 850, 400, 880]),
           block("| Option | Value |\n| --- | --- |\n| Example | Enabled |", "t", ["B"], "table",
                 page=2, box=[10, 50, 400, 200])]
    cs = chunks(raw, budget)
    assert all(not ("Alpha" in c.text and heading in c.text) for c in cs)
    h = next(c for c in cs if heading in c.text)
    assert h.block.metadata["section_path"] == ["B"]
    assert h.block.page_idx == 1
    assert all(r["page"] == 1 and r["bbox"] == raw[1].bbox for r in h.block.metadata["source_regions"])
    assert cs[-1].block.page_idx == 2 and "Enabled" in cs[-1].text


@pytest.mark.parametrize("other_column", [False, True])
@pytest.mark.parametrize("small", [False, True])
def test_premerge_does_not_remove_visual_ownership_boundaries(other_column, small):
    x = 600 if other_column else 10
    raw = [block("Configuration", "h", ["B"], "title", box=[10, 10, 400, 50]),
           block("VISUAL SUMMARY: A source image.", "i", ["B"], "image",
                 box=[x, 80, x + (20 if small else 300), 100 if small else 300])]
    cs = chunks(raw)
    im = next(c for c in cs if c.block.block_type == "image")
    assert im.block.bbox == raw[1].bbox or (not small and not other_column)
    assert "A source image." in im.text
    if other_column or small:
        assert not im.text.startswith("TITLE:")
        assert any(c.text == "Configuration" for c in cs)
    else:
        assert im.text.startswith("TITLE: Configuration")


def test_fragment_uses_its_own_page_and_regions_for_the_following_image():
    raw = [block("Earlier explanation.", "a", ["A"], box=[10, 10, 400, 80]),
           block("Configuration", "h", ["B"], "title", page=2, box=[10, 10, 400, 50]),
           block("VISUAL SUMMARY: A diagram.", "i", ["B"], "image", page=2, box=[10, 80, 400, 300])]
    cs = chunks(raw)
    old = next(c for c in cs if "Earlier" in c.text)
    image = next(c for c in cs if c.block.block_type == "image")
    assert old.block.page_idx == 1 and old.block.metadata["section_path"] == ["A"]
    assert image.text.startswith("TITLE: Configuration") and image.block.page_idx == 2
    assert image.block.metadata["section_path"] == ["B"]


def test_final_heading_is_retained_without_inventing_a_previous_owner():
    cs = chunks([block("Earlier explanation.", "a", []),
                 block("Configuration", "h", [], "title")])
    assert [c.text for c in cs] == ["Earlier explanation.", "Configuration"]


def test_heading_follows_its_body_and_same_section_cross_page_prose_still_packs():
    cs = chunks([block("Configuration", "h", ["B"], "title"),
                 block("The first sentence explains the operation.", "a", ["B"]),
                 block("The second sentence continues it.", "b", ["B"], page=2)])
    assert len(cs) == 1
    assert cs[0].text.startswith("Configuration\n")
    assert cs[0].block.metadata["page_end"] == 2
    assert "second sentence" in cs[0].text


def test_complete_card_and_unowned_short_conclusion_remain_intact():
    raw = [block("A complete card", "h", ["A"], "title", box=[10, 10, 400, 40]),
           block("Its short explanation is complete.", "p", ["A"], box=[10, 50, 400, 100]),
           block("VISUAL SUMMARY: Card illustration.", "i", ["A"], "image", box=[10, 110, 400, 350]),
           block("Result: 17 V.", "note", ["A"], box=[10, 360, 400, 390]),
           block("VISUAL SUMMARY: An unrelated chart.", "other", ["A"], "image", box=[500, 400, 900, 700])]
    cs = chunks(raw)
    assert cs[0].text == "A complete card\nIts short explanation is complete."
    assert any(c.text == "Result: 17 V." for c in cs)
    assert len([c for c in cs if c.block.block_type == "image"]) == 2


@pytest.mark.parametrize("heading", ["Configuration", "配置说明", "2. Configuration"])
@pytest.mark.parametrize("side", ["left", "right"])
def test_heading_follows_prose_around_a_side_graphic_without_losing_the_graphic(heading, side):
    raw = [block(heading, "h", ["B"], "title", box=[100, 100, 700, 140]),
           block("VISUAL SUMMARY: A meaningful status symbol.", "i", ["B"], "image",
                 box=[40, 90, 90, 150] if side == "left" else [800, 108, 900, 132]),
           block("The service is available during the stated interval.", "b", ["B"],
                 box=[100, 160, 700, 260])]
    raw[1].visual_ref = "source-symbol.png"
    original = deepcopy(raw)
    cs = chunks(raw)
    body = next(c for c in cs if "service is available" in c.text)
    image = next(c for c in cs if c.block.visual_ref)
    assert body.text.startswith(heading + "\n") and body.block.metadata["section_path"] == ["B"]
    assert {tuple(r["bbox"]) for r in body.block.metadata["source_regions"]} == {tuple(raw[0].bbox), tuple(raw[2].bbox)}
    assert image.text == raw[1].text and image.block.bbox == raw[1].bbox
    assert image.block.visual_ref == raw[1].visual_ref and raw == original


@pytest.mark.parametrize("boundary", ["figure_between", "tall_side_figure", "other_column", "other_section", "missing_geometry", "new_heading"])
def test_heading_association_does_not_jump_a_real_content_or_uncertain_boundary(boundary):
    raw = [block("Configuration", "h", ["A"], "title", box=[100, 100, 700, 140]),
           block("VISUAL SUMMARY: Independent source figure.", "i", ["A"], "image", box=[800, 108, 900, 132]),
           block("A separate passage has its own context.", "b", ["A"], box=[100, 160, 700, 260])]
    if boundary == "figure_between":
        raw[1].bbox = [100, 150, 700, 300]
        raw[2].bbox = [100, 320, 700, 420]
    elif boundary == "tall_side_figure": raw[1].bbox = [800, 108, 900, 300]
    elif boundary == "other_column":
        raw[0].bbox = [100, 100, 400, 140]
        raw[2].bbox = [500, 160, 900, 260]
    elif boundary == "other_section": raw[2].metadata["section_path"] = ["B"]
    elif boundary == "missing_geometry": raw[1].bbox = None
    else: raw.insert(2, block("Another section", "h2", ["B"], "title", box=[100, 145, 700, 155]))
    cs = chunks(raw)
    assert all(not ("Configuration" in c.text and "separate passage" in c.text) for c in cs)
    assert any("Independent source figure" in c.text for c in cs)


def test_title_and_body_repack_inside_budget_with_all_body_text_and_image_sources():
    body = " ".join(f"Measurement {i} remains {i + 10} units." for i in range(18))
    raw = [block("Measurement procedure", "h", ["B"], "title", box=[100, 100, 700, 140]),
           block("VISUAL SUMMARY: Source marker.", "i", ["B"], "image", box=[800, 108, 900, 132]),
           block(body, "b", ["B"], box=[100, 160, 700, 900])]
    cs = chunks(raw, budget=80)
    texts = [c for c in cs if c.block.block_type == "text"]
    assert texts[0].text.startswith("Measurement procedure\nMeasurement 0")
    assert all(c.token_count <= 80 for c in texts)
    assert " ".join(c.text for c in texts).split() == ("Measurement procedure " + body).split()
    assert len([c for c in cs if c.block.block_type == "image"]) == 1


def test_heading_before_a_side_graphic_can_own_body_continuing_on_next_page():
    raw = [block("Configuration", "h", ["B"], "title", box=[100, 100, 700, 140]),
           block("VISUAL SUMMARY: Source marker.", "i", ["B"], "image", box=[800, 108, 900, 132]),
           block("The first paragraph starts here.", "b", ["B"], box=[100, 160, 700, 260]),
           block("The second paragraph continues it.", "c", ["B"], page=2, box=[100, 100, 700, 260])]
    cs = chunks(raw)
    body = next(c for c in cs if "first paragraph" in c.text)
    assert body.text.startswith("Configuration\n") and "second paragraph" in body.text
    assert body.block.page_idx == 1 and body.block.metadata["page_end"] == 2


def test_consecutive_headings_do_not_leave_the_first_heading_with_previous_body():
    raw = [block("Earlier material belongs to the preceding section.", "a", ["A"]),
           block("SECTION B", "h1", ["B"], "title", page=2),
           block("配置说明", "h2", ["B"], "title", page=2)]
    cs = chunks(raw)
    assert all(not ("Earlier material" in c.text and "SECTION B" in c.text) for c in cs)
    assert any("SECTION B" in c.text and "配置说明" in c.text for c in cs)


def test_heading_binds_only_the_opening_section_of_a_prepacked_body():
    raw = [block('Configuration', 'h', ['Configuration'], 'title', box=[100, 100, 700, 140]),
           block('VISUAL SUMMARY: Side symbol.', 'i', ['Configuration'], 'image', box=[800, 108, 900, 132]),
           block('Alpha uses 17 units.', 'a', ['Configuration'], box=[100, 160, 700, 200]),
           block('References', 'h2', ['References'], 'title', box=[100, 250, 700, 280]),
           block('Beta is independent.', 'b', ['References'], box=[100, 300, 700, 340])]
    cs = chunks(raw)
    owned = next(c for c in cs if 'Alpha uses' in c.text)
    assert owned.text == 'Configuration\nAlpha uses 17 units.'
    assert owned.block.metadata['section_path'] == ['Configuration']
    other = next(c for c in cs if 'Beta is independent' in c.text)
    assert other.text == 'References\nBeta is independent.'
    assert other.block.metadata['section_path'] == ['References']
    assert len([c for c in cs if c.block.block_type == 'image']) == 1


def test_consecutive_heading_names_retain_both_sources_with_their_body():
    raw = [block('OVERVIEW', 'h1', ['OVERVIEW'], 'title', box=[100, 100, 700, 130]),
           block('Configuration', 'h2', ['Configuration'], 'title', box=[100, 140, 700, 170]),
           block('VISUAL SUMMARY: Side symbol.', 'i', ['Configuration'], 'image', box=[800, 148, 900, 162]),
           block('Alpha uses 17 units.', 'a', ['Configuration'], box=[100, 200, 700, 240])]
    cs = chunks(raw)
    owned = next(c for c in cs if 'Alpha uses' in c.text)
    assert owned.text == 'OVERVIEW\nConfiguration\nAlpha uses 17 units.'
    assert owned.block.metadata['section_paths'] == [['OVERVIEW'], ['Configuration']]
    assert {r['block_id'] for r in owned.block.metadata['source_regions']} == {'h1', 'h2', 'a'}


@pytest.mark.parametrize('side', ['left', 'right'])
def test_inline_body_marker_does_not_detach_a_heading(side):
    raw = [block('Configuration', 'h', ['A'], 'title', box=[100, 100, 900, 140]),
           block('VISUAL SUMMARY: Status marker.', 'i', ['A'], 'image',
                 box=[100, 170, 150, 200] if side == 'left' else [850, 170, 900, 200]),
           block('Alpha uses 17 units.', 'a', ['A'], box=[180, 160, 800, 220])]
    cs = chunks(raw)
    assert next(c.text for c in cs if 'Alpha uses' in c.text) == 'Configuration\nAlpha uses 17 units.'
    assert next(c.block.bbox for c in cs if c.block.block_type == 'image') == raw[1].bbox


def test_cross_page_parent_and_child_headings_keep_their_known_relationship():
    raw = [block('Parent section', 'parent', ['Parent'], 'title', box=[100, 850, 700, 880]),
           block('Child section', 'child', ['Parent', 'Child'], 'title', page=2, box=[100, 100, 700, 130]),
           block('The child section contains this explanation.', 'body', ['Parent', 'Child'],
                 page=2, box=[100, 150, 700, 250])]
    cs = chunks(raw)
    assert len(cs) == 1
    assert cs[0].text == 'Parent section\nChild section\nThe child section contains this explanation.'
    assert cs[0].block.metadata['section_paths'] == [['Parent'], ['Parent', 'Child']]


def test_page_break_does_not_split_heading_group_when_upstream_levels_are_flat():
    # PDF heading levels may flatten a real parent/child pair into siblings.
    # A page break is not sufficient evidence to discard their adjacency.
    raw = [block('Model administration', 'parent', ['Model administration'], 'title',
                 box=[100, 850, 700, 880]),
           block('Publishing a model', 'child', ['Publishing a model'], 'title',
                 page=2, box=[100, 100, 700, 130])]
    cs = chunks(raw)
    assert [c.text for c in cs] == ['Model administration\nPublishing a model']


@pytest.mark.parametrize('interruption', [None, 'other_section', 'image', 'same_page', 'page_gap'])
def test_short_intro_joins_its_cross_page_continuation_before_a_later_heading(interruption):
    intro = block('Procedure\nThe following items explain the procedure:', 'intro', ['Procedure'], page=1)
    intro.metadata['heading_lines'] = ['Procedure']
    prefix = '- First item explains the operation.\n- Second item describes the expected result.'
    tail = 'References\nExternal references describe a separate subject.'
    body = block(prefix + '\n' + tail, 'body', [], page=2)
    body.metadata.update(heading_lines=['References'], section_spans=[
        {'start': 0, 'end': len(prefix), 'section_path': ['Other' if interruption == 'other_section' else 'Procedure'],
         'source_regions': [{'page': 2, 'bbox': [100, 100, 700, 200]}]},
        {'start': len(prefix) + 1, 'end': len(body.text), 'section_path': ['References'],
         'source_regions': [{'page': 2, 'bbox': [100, 250, 700, 400]}]},
    ])
    if interruption in {'same_page', 'page_gap'}:
        body.page_idx = 1 if interruption == 'same_page' else 3
        for span in body.metadata['section_spans']:
            span['source_regions'][0]['page'] = body.page_idx
    raw = [intro, body]
    if interruption == 'image':
        raw.insert(1, block('VISUAL SUMMARY: Independent diagram.', 'image', ['Procedure'], 'image'))
    cs = blocks_to_chunks(kb_id='fixture', file_key=1, content_version='v1', parser_profile='v1',
                          blocks=raw, max_tokens=400, overlap_tokens=0)
    if interruption:
        assert not any('following items' in c.text and 'First item' in c.text for c in cs)
    else:
        owned = next(c for c in cs if 'following items' in c.text)
        assert 'First item' in owned.text and 'Second item' in owned.text
        assert 'References' not in owned.text
        assert owned.block.page_idx == 1 and owned.block.metadata['page_end'] == 2
        assert owned.block.metadata['section_path'] == ['Procedure']
        assert next(c for c in cs if 'External references' in c.text).block.metadata['section_path'] == ['References']
    assert all(c.token_count <= 400 for c in cs)
