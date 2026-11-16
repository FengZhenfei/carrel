"""Encoding invariance and end-to-end diagnostics, using synthetic sources."""
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
from PIL import Image
from kb_pipeline.models import ParsedBlock
from kb_pipeline.parsers.pdf_evidence import add_pdf_region_evidence, _native_line_spans
from kb_pipeline.parsers import visual_blocks
from kb_pipeline.chunking.chunker import blocks_to_chunks
from kb_pipeline.chunking.diagnose import chunk_diagnostics
from kb_pipeline.pipeline.parse_job import _payload_for_chunk


@pytest.mark.parametrize('split', [False, True])
@pytest.mark.parametrize('missing', [False, True])
def test_table_review_is_invariant_to_glyph_span_encoding(tmp_path, split, missing):
    pdf = pytest.importorskip('pymupdf')
    source = tmp_path / 'source.pdf'
    with pdf.open() as doc:
        p = doc.new_page(width=300, height=180)
        p.draw_rect(pdf.Rect(30, 30, 270, 140))
        p.draw_line((30, 85), (270, 85)); p.draw_line((170, 30), (170, 140))
        p.insert_text((40, 65), 'Metric', fontsize=12)
        p.insert_text((180, 65), 'Value', fontsize=12)
        for x, word in [(40, 'ABCD'), (180, '12')]:
            for i, text in enumerate(word if split else [word]):
                c = .01 if i % 2 else 0
                p.insert_text((x, 115), text, fontsize=12, color=(c,c,c))
                x += pdf.get_text_length(text, fontsize=12)
        doc.save(source)
    text = '| Metric | Value |\n| --- | --- |\n' + ('| ABC | 1 |' if missing else '| ABCD | 12 |')
    b = ParsedBlock('fixture','fixture','pdf','table',text,'t',page_idx=1,
                    bbox=[100,160,900,780],table_markdown=text)
    add_pdf_region_evidence([b],source,tmp_path)
    assert bool(b.metadata.get('table_review_reasons')) is missing
    assert b.text == text  # Detection cannot silently rewrite the table.
    if missing:
        assert b.metadata['table_missing_spans'] == ['ABCD','12']
        assert b.metadata['table_native_grid'] == [['Metric','Value'],['ABCD','12']]


@pytest.mark.parametrize('change', ['column_gap','baseline','font_size','space','direction','symbol'])
def test_single_glyph_join_respects_evidence_boundaries(change):
    spans = [dict(text='1',bbox=(10,10,16,22),size=12),dict(text='2',bbox=(16,10,22,22),size=12)]
    line = dict(dir=(1,0),spans=spans)
    if change == 'column_gap':spans[1]['bbox']=(22,10,28,22)
    if change == 'baseline':spans[1]['bbox']=(16,8,22,20)
    if change == 'font_size':spans[1]['size']=8
    if change == 'space':spans[1]['text']=' 2'
    if change == 'direction':line['dir']=(0,-1)
    if change == 'symbol':spans[1]['text']='-'
    assert [s['text'] for s in _native_line_spans(line)] == [s['text'] for s in spans]


@pytest.mark.parametrize('state', ['normal','recovered','truncated','repetitive','failed'])
def test_visual_diagnostic_survives_blocks_chunks_payload_and_counts_images_once(tmp_path, state):
    p=tmp_path/'figure.png';Image.new('RGB',(300,300)).save(p)
    facts=[f'Stage {i} opens only after gate {i+1} closes.' for i in range(20)]
    data=dict(summary='A workflow',kind='diagram',facts=facts,confidence='high')
    if state != 'normal':data.update(vlm_degraded_retry=True,vlm_degraded_reason='repetitive_output')
    if state == 'truncated':data['vlm_truncated']=True
    if state == 'repetitive':data['vlm_runaway']=True
    if state == 'failed':data={'error':'offline'}
    b=ParsedBlock('fixture','fixture','pdf','image','','image',page_idx=1,visual_ref=str(p))
    with patch.object(visual_blocks,'caption_images_parallel',return_value={'image':data}) as call:
        visual_blocks.enrich_blocks_with_vlm([b],base_url='fixture',api_key='fixture',model_id='fixture',cache_dir=tmp_path)
    assert call.call_count==1
    if state=='failed':
        assert b.metadata['vlm_status']=='failed'
        return
    chunks=blocks_to_chunks(kb_id='kb_fixture',file_key=1,content_version='v',parser_profile='p',blocks=[b],max_tokens=80,overlap_tokens=0)
    assert len(chunks)>1
    row=dict(kb_id='kb_fixture',file_key=1,source_path='source.pdf',rel_path='source.pdf',filename='source.pdf',dir='',content_version='v',size=1,mtime=1,mime_type='application/pdf')
    settings=NS(cache_dir=tmp_path,embedding_model_id='fixture')
    for c in chunks:
        payload=_payload_for_chunk(settings,row,c,len(chunks))
        assert payload['vlm_degraded_retry'] is (state!='normal')
        assert payload.get('vlm_degraded_reason')==(None if state=='normal' else 'repetitive_output')
    for blocks in ([b],None):
        diag=chunk_diagnostics(chunks,max_tokens=80,blocks=blocks)
        assert diag['stats']['visual_outputs_degraded']==(0 if state=='normal' else 1)
        keys={r['key'] for r in diag['reasons']}
        expected={'recovered':'visual_output_degraded','truncated':'visual_output_truncated','repetitive':'visual_output_repetitive'}.get(state)
        if expected:assert expected in keys
        assert ('visual_output_degraded' in keys) is (state=='recovered')


@pytest.mark.parametrize('title', ['目录','目 录','Table of Contents','Contents'])
def test_explicit_toc_handles_short_lists_without_suppressing_body(title):
    from kb_pipeline.headings import explicit_toc_entries
    assert explicit_toc_entries([(1,title),(1,'1. Setup\n2. Usage'),(2,'1. Setup'),(2,'Body prose.')])=={0,1}
    assert explicit_toc_entries([(1,title),(1,'1. Setup....3\n2. Usage....4'),(1,'1. Setup'),(1,'Body prose.')])=={0,1}
    assert explicit_toc_entries([(1,title),(1,'1. Setup'),(1,'This explains the contents.')])==set()
    assert explicit_toc_entries([(1,'Setup'),(1,'1. First step\n2. Second step')])==set()


def test_toc_is_searchable_but_cannot_set_following_section_or_time_context(tmp_path):
    from kb_pipeline.parsers import mineru_pdf
    from kb_pipeline.parsers.pdf_enhanced import merge_mineru_text_blocks
    pdf=pytest.importorskip('pymupdf');source=tmp_path/'book.pdf'
    with pdf.open() as doc:
        doc.new_page();doc.new_page();doc.save(source)
    items=[dict(type='text',text=t,text_level=level,page_idx=page,bbox=[10,10+n*50,900,40+n*50])
           for n,(t,level,page) in enumerate([
               ('Contents',1,0),('1. History 2021....3',1,0),('2. Plans 2030....4',1,0),
               ('1. Results',1,0),('Measurements were taken in 2024.',None,0),
               ('Additional measurements.',None,1)])]
    payload={'version':'3.4.5','content_list':items}
    with patch.object(mineru_pdf,'call_mineru_sync',return_value=payload):
        blocks=mineru_pdf.mineru_pdf_blocks(mineru_url='fixture',path=source,cache_dir=tmp_path)
    assert len(blocks)==6
    for b in blocks[:3]:
        assert b.block_type=='text' and b.metadata['is_toc'] and b.metadata['section_path']==[]
    assert all(b.metadata['section_path']==['1. Results'] for b in blocks[3:])
    merged=merge_mineru_text_blocks(blocks,target_tokens=400)
    assert merged[0].metadata.get('is_toc') and merged[0].metadata['merged_block_count']==3
    assert '2030' not in '\n'.join(b.text for b in merged[1:])
    chunks=blocks_to_chunks(kb_id='kb_fixture',file_key=1,content_version='v',parser_profile='p',blocks=merged,max_tokens=80,overlap_tokens=0)
    assert any('Plans 2030' in c.text for c in chunks)
    assert all('Plans' not in str(c.block.metadata.get('section_path')) for c in chunks)
