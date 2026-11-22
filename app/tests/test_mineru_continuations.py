"""Upstream table merges keep evidence, while empty continuations add no chunks."""
from copy import deepcopy
from unittest.mock import patch

import pytest

from kb_pipeline.chunking.chunker import blocks_to_chunks
from kb_pipeline.models import ParsedBlock
from kb_pipeline.parsers.mineru_pdf import mineru_pdf_blocks
from kb_pipeline.parsers.mineru_quality import merge_table_placeholders, recover_pdf_items
from kb_pipeline.parsers.pdf_longpage import restore_page_coordinates


def html(rows):
    return '<table>' + ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in row) + '</tr>' for row in rows) + '</table>'


def table(image, rows, box):
    return {'type': 'table', 'bbox': box, 'blocks': [{'type': 'table_body', 'lines': [{'spans': [
        {'type': 'table', 'image_path': image, 'html': html(rows)}]}]}]}


def fixture(parts=None):
    parts = parts or [[['Code', 'Value'], ['A', '12']], [['B', '34']], [['Code', 'Value'], ['C', '56']]]
    merged = list(parts[0])
    for rows in parts[1:]:
        merged += rows[1:] if rows[0] == parts[0][0] else rows
    items, pages = [], []
    for i, rows in enumerate(parts):
        box = [30, 50, 270, 200]
        items.append({'type': 'table', 'page_idx': i, 'bbox': [100, 100, 900, 400],
            'img_path': 'images/p0.jpg' if i == 0 else '', 'table_caption': [], 'table_footnote': [],
            **({'table_body': html(merged)} if i == 0 else {})})
        pages.append({'page_idx': i, 'page_size': [300, 500],
            'preproc_blocks': [table(f'p{i}.jpg', rows, box)],
            'para_blocks': [table('p0.jpg', merged, box)] if i == 0 else []})
    return {'content_list': items, 'middle_json': {'pdf_info': pages}}


def recovered(payload):
    return merge_table_placeholders(recover_pdf_items(payload['content_list'], payload))


def test_exact_multi_page_merge_removes_only_placeholders_and_keeps_notes():
    payload = fixture()
    child = payload['content_list'][1]
    child['table_footnote'] = ['Units: kPa']
    child['_pdf_table_note_sources'] = [{'page': 2, 'bbox': [100, 405, 800, 430], 'text': 'Units: kPa'}]
    original = deepcopy(payload)
    out = recovered(payload)
    assert len(out) == 1
    assert out[0]['table_body'] == original['content_list'][0]['table_body']
    assert out[0]['source_page_end'] == 3
    assert out[0]['table_footnote'] == ['Units: kPa']
    assert [r['page'] for r in out[0]['_pdf_merged_table_sources']] == [2, 3]
    assert out[0]['_pdf_table_note_sources'][0]['text'] == 'Units: kPa'
    assert payload == original


@pytest.mark.parametrize('change', ['wrong_values', 'wrong_columns', 'missing_parent', 'partial_parent',
                                   'duplicate_parent', 'duplicate_fragment', 'duplicate_placeholder',
                                   'bad_box', 'bad_size', 'own_text', 'own_caption', 'own_image'])
def test_uncertain_or_independent_tables_are_retained(change):
    p = fixture(parts=[[['Code', 'Value'], ['A', '12']], [['B', '34']]])
    items, pages = p['content_list'], p['middle_json']['pdf_info']
    if change == 'wrong_values':
        pages[1]['preproc_blocks'] = [table('p1.jpg', [['B', '35']], [30, 50, 270, 200])]
    elif change == 'wrong_columns':
        pages[1]['preproc_blocks'] = [table('p1.jpg', [['B', '34', 'other']], [30, 50, 270, 200])]
    elif change == 'missing_parent':
        items.pop(0)
    elif change == 'partial_parent':
        items[0]['table_body'] = html([['Code', 'Value'], ['A', '12']])
    elif change == 'duplicate_parent':
        items.insert(0, deepcopy(items[0]))
    elif change == 'duplicate_fragment':
        pages[1]['preproc_blocks'].append(table('other.jpg', [['B', '34']], [30, 250, 270, 400]))
    elif change == 'duplicate_placeholder':
        items.append(deepcopy(items[1]))
    elif change == 'bad_box':
        items[1]['bbox'] = [100, 500, 900, 800]
    elif change == 'bad_size':
        pages[1]['page_size'] = [0, 500]
    elif change == 'own_text':
        items[1]['text'] = 'An independent observation.'
    elif change == 'own_caption':
        items[1]['table_caption'] = ['Another table']
    elif change == 'own_image':
        items[1]['img_path'] = 'independent.jpg'
    assert len(recovered(p)) == len(items)


def test_same_page_neighbor_and_unmatched_empty_table_survive():
    p = fixture()
    other = {'type': 'table', 'page_idx': 1, 'bbox': [100, 500, 900, 700], 'table_body': html([['Other', '91']])}
    empty = {'type': 'table', 'page_idx': 2, 'bbox': [100, 500, 900, 700], 'table_footnote': ['Unresolved']}
    p['content_list'].extend([other, empty])
    out = recovered(p)
    assert len(out) == 3 and other in out and empty in out


def test_parser_carries_page_regions_and_attached_footnotes(tmp_path):
    p = fixture()
    def attach(items, _path):
        items[1]['table_footnote'] = ['Independent footnote']
        items[1]['_pdf_table_note_sources'] = [{'page': 2, 'bbox': [100, 405, 900, 425]}]
        return items
    with patch('kb_pipeline.parsers.mineru_pdf.call_mineru_sync', return_value=p), \
         patch('kb_pipeline.parsers.mineru_pdf.attach_table_notes', side_effect=attach):
        blocks = mineru_pdf_blocks(mineru_url='unused', path=tmp_path/'table.pdf', cache_dir=tmp_path)
    assert len(blocks) == 1
    b = blocks[0]
    assert 'Independent footnote' in b.text
    assert [r['page'] for r in b.metadata['source_regions']] == [1, 2, 3, 2]
    assert b.metadata['page_end'] == 3
    assert '_merged_table_continuations' not in b.metadata['source_item']


def test_long_merged_table_still_splits_to_token_budget_without_losing_rows(tmp_path):
    header = ['Identifier', 'Value', 'Description']
    rows = [[f'ITEM_{i:04d}', str(i * 17), 'A measurement with an independent source record.'] for i in range(180)]
    p = fixture(parts=[[header]+rows[:60], [header]+rows[60:120], [header]+rows[120:]])
    with patch('kb_pipeline.parsers.mineru_pdf.call_mineru_sync', return_value=p):
        blocks = mineru_pdf_blocks(mineru_url='unused', path=tmp_path/'long.pdf', cache_dir=tmp_path)
    assert len(blocks) == 1
    chunks = blocks_to_chunks(kb_id='synthetic', file_key=1, content_version='v1', parser_profile='p',
                             blocks=blocks, max_tokens=128, overlap_tokens=0)
    assert len(chunks) > 20
    assert max(c.token_count for c in chunks) <= 128
    joined = '\n'.join(c.text for c in chunks)
    for identifier, value, description in rows:
        assert joined.count(identifier) == 1
        assert f'{identifier} | {value} | {description}' in joined
    assert all(c.block.metadata['page_end'] == 3 for c in chunks)
    assert {r['page'] for c in chunks for r in c.block.metadata['source_regions']} == {1, 2, 3}


def test_region_coordinates_follow_long_page_tiles_without_mutating_shared_notes():
    note = {'page': 2, 'bbox': [100, 100, 900, 400]}
    b = ParsedBlock(parser='p', parser_profile='p', doc_type='pdf', block_type='table', text='x',
        block_id='a', page_idx=1, bbox=[100, 100, 900, 400],
        metadata={'page_end': 2, 'source_regions': [{'page': 1, 'bbox': [100, 100, 900, 400]}, note],
                  'table_note_sources': [note]})
    segments = [{'page': 1, 'tiled': True, 'width': 300, 'height': 1000, 'y0': 0, 'y1': 500},
                {'page': 1, 'tiled': True, 'width': 300, 'height': 1000, 'y0': 500, 'y1': 1000}]
    restore_page_coordinates([b], segments)
    assert b.metadata['source_regions'][1]['bbox'] == [100, 550, 900, 700]
    assert b.metadata['table_note_sources'][0] == b.metadata['source_regions'][1]
    assert note['page'] == 2 and note['bbox'] == [100, 100, 900, 400]
    assert b.metadata['page_end'] == 1


@pytest.mark.parametrize('native_value,expected', [('34', 1), ('35', 2)])
def test_native_evidence_not_in_parent_keeps_continuation(tmp_path, native_value, expected):
    import pymupdf
    p = fixture(parts=[[['Code', 'Value'], ['A', '12']], [['B', '34']]])
    path = tmp_path/'native.pdf'
    doc = pymupdf.open()
    doc.new_page(width=300, height=500)
    page = doc.new_page(width=300, height=500)
    page.insert_text((40, 80), 'B')
    page.insert_text((140, 80), native_value)
    doc.save(path)
    doc.close()
    out = merge_table_placeholders(recover_pdf_items(p['content_list'], p), path)
    assert len(out) == expected
