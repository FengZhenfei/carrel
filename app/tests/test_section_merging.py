"""Synthetic short-tail boundaries for the shared PDF/DOCX text merger."""
from copy import deepcopy

import pytest

from kb_pipeline.chunking.chunker import blocks_to_chunks
from kb_pipeline.models import ParsedBlock
from kb_pipeline.parsers.pdf_enhanced import merge_mineru_text_blocks


def block(ident, text, section, *, kind='pdf', page=1, block_type='text'):
    return ParsedBlock('fixture', 'v1', kind, block_type, text, ident, page_idx=page,
                       bbox=[100, 100, 800, 200],
                       metadata={'section_path': section, 'source_type': block_type})


@pytest.mark.parametrize('kind', ['pdf', 'docx'])
@pytest.mark.parametrize('section', [['Maintenance'], []])
def test_short_tail_after_overflow_keeps_its_own_section(kind, section):
    blocks = [block('a', 'Detailed setup requirement. ' * 30, ['Setup'], kind=kind),
              block('b', 'Final summary.', section, kind=kind, page=2)]
    original = deepcopy(blocks)
    merged = merge_mineru_text_blocks(blocks, target_tokens=80)
    assert len(merged) == 2
    assert [b.text for b in merged] == [b.text.strip() for b in original]
    assert [b.metadata['section_path'] for b in merged] == [['Setup'], section]
    assert [b.metadata['source_block_ids'] for b in merged] == [['a'], ['b']]
    assert merged[0].metadata['page_end'] == 1
    assert merged[1].metadata['source_regions'] == [{'page': 2, 'bbox': [100, 100, 800, 200]}]
    assert blocks == original
    chunks = blocks_to_chunks(kb_id='fixture', file_key=1, content_version='v', parser_profile='v1',
                             blocks=merged, max_tokens=80, overlap_tokens=0)
    tail = [c for c in chunks if 'Final summary.' in c.text]
    assert len(tail) == 1 and tail[0].block.metadata['section_path'] == section


@pytest.mark.parametrize('kind', ['pdf', 'docx'])
@pytest.mark.parametrize('section', [[], ['Procedure', 'Validation']])
def test_same_section_tail_still_folds_across_pages(kind, section):
    blocks = [block('a', 'Detailed validation requirement. ' * 30, section, kind=kind),
              block('b', 'Record the result.', section, kind=kind, page=2)]
    merged = merge_mineru_text_blocks(blocks, target_tokens=80)
    assert len(merged) == 1
    assert merged[0].text == '\n\n'.join(b.text.strip() for b in blocks)
    assert merged[0].metadata['section_path'] == section
    assert merged[0].metadata['source_block_ids'] == ['a', 'b']
    assert [r['page'] for r in merged[0].metadata['source_regions']] == [1, 2]


@pytest.mark.parametrize('kind', ['pdf', 'docx'])
def test_short_new_section_before_a_table_does_not_fold_back(kind):
    blocks = [block('a', 'Detailed setup requirement. ' * 30, ['Setup'], kind=kind),
              block('b', 'Observed result.', ['Measurement'], kind=kind, page=2),
              block('table', '| Field | Value |\n| --- | --- |\n| Alpha | 7 |', ['Measurement'],
                    kind=kind, page=2, block_type='table')]
    original = deepcopy(blocks)
    merged = merge_mineru_text_blocks(blocks, target_tokens=80)
    assert len(merged) == 3
    assert merged[1].text == 'Observed result.'
    assert merged[1].metadata['section_path'] == ['Measurement']
    assert merged[2] == original[2]


@pytest.mark.parametrize('kind', ['pdf', 'docx'])
def test_code_barrier_remains_separate_from_a_short_tail(kind):
    blocks = [block('a', 'Detailed setup requirement. ' * 30, ['Setup'], kind=kind),
              block('code', '    emit("keep indentation")\n', ['Setup'], kind=kind, block_type='code'),
              block('b', 'Record the result.', ['Setup'], kind=kind, page=2)]
    original = deepcopy(blocks)
    merged = merge_mineru_text_blocks(blocks, target_tokens=80)
    assert len(merged) == 3
    assert merged[1] == original[1]
    assert merged[2].text == original[2].text
    assert merged[2].metadata['source_block_ids'] == ['b']
