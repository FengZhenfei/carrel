"""Generic blank-column and native row-label recovery without fuzzy cell matching."""
from copy import deepcopy
from unittest.mock import patch

import pytest

from kb_pipeline.models import ParsedBlock
from kb_pipeline.parsers.common import grid_to_markdown
from kb_pipeline.parsers.table_repair import (
    _pipe_grid, _recover_native_row_keys, _restore_review_empty_columns, review_pdf_tables,
)


@pytest.mark.parametrize('columns', [(0,), (1,), (2,), (0, 2)])
def test_omitted_wholly_empty_columns_restore_original_positions(columns):
    compact = [['Item', 'Value'], ['Alpha', '17 V'], ['Beta', '9 V']]
    old = deepcopy(compact)
    for j in columns:
        for row in old:
            row.insert(j, '')
    original = deepcopy(old)
    assert _restore_review_empty_columns(old, compact) == old
    assert old == original


@pytest.mark.parametrize('change', ['value', 'unit', 'sign', 'polarity', 'row_order', 'column_order', 'extra_row', 'partial_empty', 'ragged'])
def test_blank_column_mapping_requires_all_retained_cells_to_match(change):
    old = [['', 'Item', 'Value'], ['', 'Alpha', '17 V'], ['', 'Beta', 'allowed']]
    new = [[r[1], r[2]] for r in old]
    if change == 'value': new[1][1] = '19 V'
    if change == 'unit': new[1][1] = '17 A'
    if change == 'sign': new[1][1] = '-17 V'
    if change == 'polarity': new[2][1] = 'not allowed'
    if change == 'row_order': new[1], new[2] = new[2], new[1]
    if change == 'column_order': new = [list(reversed(r)) for r in new]
    if change == 'extra_row': new.append(['Gamma', '7 V'])
    if change == 'partial_empty': old[2][0] = 'source condition'
    if change == 'ragged': new[1].pop()
    assert _restore_review_empty_columns(old, new) == new


def native_fixture():
    old = [['Component', 'Observation', 'Unit'], ['★Alpha', 'First wrapped description.', ''], ['', 'Second wrapped description.', '']]
    native = [['Component', 'Observation', 'Unit'], ['Alpha', 'First wrapped ', ''], ['', 'description.', ''],
              ['Beta', 'Second wrapped ', ''], ['', 'description.', '']]
    spans = [{'text': text, 'bbox': box} for text, box in [
        ('Component', [10, 0, 55, 10]), ('Observation', [80, 0, 160, 10]), ('Unit', [200, 0, 230, 10]),
        ('Alpha', [20, 20, 50, 30]), ('First wrapped ', [80, 20, 160, 30]), ('description.', [80, 30, 160, 40]),
        ('Beta', [20, 60, 50, 70]), ('Second wrapped ', [80, 60, 160, 70]), ('description.', [80, 70, 160, 80]),
    ]]
    return old, native, spans


def test_native_wrapped_rows_recover_only_missing_label_and_keep_symbol():
    old, native, spans = native_fixture()
    before = deepcopy((old, native, spans))
    new, changes = _recover_native_row_keys(old, native, spans)
    assert new == [old[0], old[1], ['Beta', *old[2][1:]]]
    assert changes == [{'row': 2, 'text': 'Beta', 'bbox': [20, 60, 50, 70]}]
    assert (old, native, spans) == before


@pytest.mark.parametrize('change', ['value', 'unit', 'sign', 'polarity', 'row_order', 'column_order', 'leading_blank',
                                   'missing_body', 'duplicate_label', 'wrong_label_column', 'wrong_label_row',
                                   'wrong_body_row', 'wrong_body_column', 'duplicate_span', 'unknown_label', 'ragged'])
def test_missing_label_is_not_guessed_when_source_or_geometry_disagrees(change):
    old, native, spans = native_fixture()
    if change in {'value', 'unit', 'sign', 'polarity'}:
        native[3][1] += {'value': '17', 'unit': ' V', 'sign': '-', 'polarity': 'not '}[change]
    if change == 'row_order': native = native[:1] + native[3:] + native[1:3]
    if change == 'column_order': native = [[r[0], r[2], r[1]] for r in native]
    if change == 'leading_blank': native.insert(0, ['', '', ''])
    if change == 'missing_body': native[3][1] = ''
    if change == 'duplicate_label': native[3][0] = 'Alpha'
    if change == 'wrong_label_column': spans[6]['bbox'] = [240, 60, 270, 70]
    if change == 'wrong_label_row': spans[6]['bbox'] = [20, 90, 50, 100]
    if change == 'wrong_body_row': spans[7]['bbox'] = [80, 90, 160, 100]
    if change == 'wrong_body_column': spans[7]['bbox'] = [0, 60, 15, 70]
    if change == 'duplicate_span': spans.append(deepcopy(spans[6]))
    if change == 'unknown_label': native[1][0] = 'Other'
    if change == 'ragged': native[3].pop()
    assert _recover_native_row_keys(old, native, spans) == (old, [])


def block(tmp_path, rows, **metadata):
    image = tmp_path/'synthetic.png'; image.write_bytes(b'mocked image')
    text = 'CAPTION: Source caption\n' + grid_to_markdown(rows) + '\nFOOTNOTE: Source note'
    return ParsedBlock(parser='fixture', parser_profile='fixture', doc_type='pdf', block_type='table',
                       block_id='t', text=text, table_markdown=text,
                       metadata={'table_review_reasons': ['native_text_missing'], 'table_review_image': str(image), **metadata})


def review(tmp_path, item, data):
    with patch('kb_pipeline.parsers.table_repair.transcribe_table', return_value=data) as call:
        stats = review_pdf_tables([item], base_url='fixture', api_key='fixture', model_id='fixture', cache_dir=tmp_path)
    return stats, call


def test_column_compatibility_does_not_hide_an_unrecovered_source_span(tmp_path):
    old = [['', 'Item', 'Value'], ['', 'Alpha', '17 V']]
    item = block(tmp_path, old, table_missing_spans=['Printed abbreviation'])
    stats, call = review(tmp_path, item, {'rows': [r[1:] for r in old]})
    assert stats == {'requested': 1, 'accepted': 0, 'unverified': 1}
    assert item.metadata['table_review']['reason'] == 'source_text_missing'
    assert _pipe_grid(item.text) == old
    assert 'Printed abbreviation' in item.text
    call.assert_called_once()


def test_native_label_unlocks_review_without_changing_measurements_or_notes(tmp_path):
    old, native, spans = native_fixture()
    item = block(tmp_path, old, table_native_grid=native, table_native_spans=spans,
                 table_source_text='Component Observation Unit Alpha First wrapped description. Beta Second wrapped description.',
                 table_missing_spans=['Beta'])
    new = deepcopy(old); new[2][0] = 'Beta'
    stats, call = review(tmp_path, item, {'rows': new})
    assert stats == {'requested': 1, 'accepted': 1, 'unverified': 0}
    assert _pipe_grid(item.text) == new
    assert item.metadata['table_row_key_recoveries'][0]['row'] == 2
    assert 'CAPTION: Source caption' in item.text and 'FOOTNOTE: Source note' in item.text
    call.assert_called_once()


def test_recovered_label_does_not_authorize_other_cell_changes(tmp_path):
    old, native, spans = native_fixture()
    item = block(tmp_path, old, table_native_grid=native, table_native_spans=spans, table_missing_spans=['Beta'])
    new = deepcopy(old); new[2] = ['Beta', 'Different observation.', '']
    stats, _ = review(tmp_path, item, {'rows': new})
    assert stats['accepted'] == 0
    assert _pipe_grid(item.text)[2] == ['Beta', *old[2][1:]]
    assert item.metadata['table_review']['status'] == 'partial'


@pytest.mark.parametrize('ocr,allowed', [('未见缺陷。', True), ('未1缺陷。', False), ('未-缺陷。', False), ('未A缺陷。', False)])
def test_native_font_radicals_do_not_match_digits_signs_or_latin_letters(ocr, allowed):
    old, native, spans = native_fixture()
    old[2][1] = ocr
    native[3][1] = '未⻅缺陷。'; native[4][1] = ''
    spans[7]['text'] = '未⻅缺陷。'; spans[8]['text'] = ''
    recovered, changes = _recover_native_row_keys(old, native, spans)
    assert bool(changes) == allowed
    assert recovered[2][1] == ocr
