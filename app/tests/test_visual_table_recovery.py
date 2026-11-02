"""Generic geometry, bounded retries and literal table-cell regression cases."""
import io
import json
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from kb_pipeline.models import ParsedBlock
from kb_pipeline.parsers.pdf_images import restore_pdf_images
from kb_pipeline.parsers.common import grid_to_markdown, html_table_to_grid
from kb_pipeline.parsers.pdf_evidence import table_evidence_text, _span_present
from kb_pipeline.parsers.table_repair import _pipe_grid
from kb_pipeline.graph.tabletext import _md_cells, expand_wide_tables, table_render_tag
from kb_pipeline.vision import vlm


def fixture_pdf(tmp_path, original=(1500,900), cached=(200,120), overlay=False):
    pdf = pytest.importorskip('pymupdf')
    path=tmp_path/'document.pdf';crop=tmp_path/'crop.png'
    im=Image.new('RGB',original,'blue');buf=io.BytesIO();im.save(buf,format='PNG')
    im.resize(cached).save(crop)
    with pdf.open() as doc:
        page=doc.new_page(width=400,height=400)
        page.insert_image(pdf.Rect(20,40,220,160),stream=buf.getvalue())
        if overlay:page.draw_line((30,100),(210,100),color=(1,0,0),width=3)
        doc.save(path)
    b=ParsedBlock(parser='fixture',parser_profile='v1',doc_type='pdf',block_type='image',block_id='figure',
                  page_idx=1,bbox=[50,100,550,400],visual_ref=str(crop),text='Original caption')
    return path,b


def test_source_density_recovery_keeps_overlay_and_provenance(tmp_path):
    path,b=fixture_pdf(tmp_path,overlay=True)
    assert restore_pdf_images([b],path,tmp_path)==1
    with Image.open(b.visual_ref) as im:
        assert im.width>=1400 and im.height>=850
        assert im.getpixel((im.width//2,im.height//2))[0]>240
    assert b.page_idx==1 and b.bbox==[50,100,550,400] and b.text=='Original caption'
    assert b.metadata['visual_resolution']['original_size']==[200,120]
    assert restore_pdf_images([b],path,tmp_path)==0


@pytest.mark.parametrize('case',['low_source','already_clear','medium_crop','small_source','bad_box','missing_page','wrong_aspect','rotated'])
def test_no_unjustified_image_replacement(tmp_path,case):
    original=(150,90) if case=='low_source' else (750,450) if case=='small_source' else (1500,900)
    cached=(1500,900) if case=='already_clear' else (480,288) if case=='medium_crop' else (200,120)
    path,b=fixture_pdf(tmp_path,original,cached)
    if case=='bad_box':b.bbox[0]=float('nan')
    if case=='missing_page':b.page_idx=None
    if case=='wrong_aspect':b.bbox[2]=200
    if case=='rotated':
        import pymupdf
        with pymupdf.open(path) as d:
            d[0].set_rotation(90);d.saveIncr()
    ref=b.visual_ref
    assert restore_pdf_images([b],path,tmp_path)==0 and b.visual_ref==ref


def test_render_and_wire_have_same_pixel_bound(tmp_path):
    path,b=fixture_pdf(tmp_path)
    assert restore_pdf_images([b],path,tmp_path,max_pixels=800_000)==1
    with Image.open(b.visual_ref) as im:assert im.width*im.height<=800_000


def test_literal_bars_and_linebreaks_remain_inside_cells():
    rows=[['Name','Expression','Value'],['Alpha','A|B\nC','-17 mA'],['Beta',r'X\|Y','>= 9 V']]
    md=grid_to_markdown(rows)
    got=_pipe_grid(md)
    assert all(len(r)==3 for r in got)
    assert got[1][1]==r'A\|B<br>C'
    assert grid_to_markdown(got)==md
    assert _md_cells(md.splitlines()[2])==['Alpha','A|B<br>C','-17 mA']
    assert _md_cells(r'| Alpha | A\|')==['Alpha','A|']
    assert _span_present('A|B C',table_evidence_text(md))
    assert not _span_present('AB C',table_evidence_text(md))
    assert not _span_present('+17 mA',table_evidence_text(md))
    html='<table><tr><td>Name</td><td>Value</td></tr><tr><td>Alpha</td><td>A|B</td></tr></table>'
    assert _pipe_grid(grid_to_markdown(html_table_to_grid(html)))[1]==['Alpha',r'A\|B']
    wide=grid_to_markdown([['Key','Expression','Current','Voltage','State','Note'],
                          ['Alpha','A|B','-17 mA','9 V','off','retained']])
    expanded=expand_wide_tables(wide)
    assert 'Expression: A|B | Current: -17 mA | Voltage: 9 V' in expanded
    assert table_render_tag(wide).endswith('+literal-pipe-v1')
    assert not table_render_tag(wide.replace(r'\|','/')).endswith('+literal-pipe-v1')


@pytest.mark.parametrize('source,body,match',[('client‑server','client-server',True),('−17','-17',False),
                                            ('<17','>17',False),('12.4','124',False)])
def test_typographic_matching_retains_numeric_and_logical_boundaries(source,body,match):
    assert _span_present(source,table_evidence_text(body)) is match


def response(value,finish='stop'):
    return NS(choices=[NS(message=NS(content=value if isinstance(value,str) else json.dumps(value)),finish_reason=finish)],
              usage=NS(completion_tokens=4096,completion_tokens_details=NS(reasoning_tokens=100)))


@pytest.mark.parametrize('second',['complete','truncated','error','empty'])
def test_output_limit_gets_only_one_retry_and_retains_available_content(tmp_path,second):
    first={'summary':'A source diagram','facts':['Valve A stays closed.']}
    other=response({**first,'facts':first['facts']+['Gate B is open.']},'stop' if second=='complete' else 'length')
    if second=='error':other=RuntimeError('temporarily unavailable')
    if second=='empty':other=response({})
    create=Mock(side_effect=[response(first,'length'),other]);client=NS(chat=NS(completions=NS(create=create)))
    with patch.object(vlm,'_client_for',return_value=client),patch.object(vlm,'image_data_url',return_value='data:image/png;base64,AA=='):
        out=vlm.caption_image(image_path=tmp_path/'unused.png',base_url='http://fixture/v1',api_key='x',model_id='fixture')
    assert create.call_count==2
    assert [x.kwargs['max_tokens'] for x in create.call_args_list]==[4096,8192]
    assert 'Valve A stays closed.' in out['facts']
    assert out['vlm_truncated'] is (second!='complete')


def test_normal_caption_makes_one_call(tmp_path):
    create=Mock(return_value=response({'summary':'A figure','facts':['Output 17 V.']}))
    client=NS(chat=NS(completions=NS(create=create)))
    with patch.object(vlm,'_client_for',return_value=client),patch.object(vlm,'image_data_url',return_value='data:image/png;base64,AA=='):
        out=vlm.caption_image(image_path=tmp_path/'unused.png',base_url='http://fixture/v1',api_key='x',model_id='fixture')
    assert create.call_count==1 and not out['vlm_truncated']


def test_reasoning_only_response_keeps_existing_degraded_retry(tmp_path):
    create=Mock(side_effect=[response('', 'length'), response({'summary':'A diagram','facts':['Output 17 V.']})])
    client=NS(chat=NS(completions=NS(create=create)))
    with patch.object(vlm,'_client_for',return_value=client),patch.object(vlm,'image_data_url',return_value='data:image/png;base64,AA=='):
        out=vlm.caption_image(image_path=tmp_path/'unused.png',base_url='http://fixture/v1',api_key='x',model_id='fixture')
    assert [x.kwargs['max_tokens'] for x in create.call_args_list]==[4096,4096]
    assert out['vlm_degraded_retry'] and out['facts']==['Output 17 V.']


def test_runaway_uses_degraded_retry_without_larger_budget(tmp_path):
    first={'summary':'A menu','facts':['Home Password '*60]}
    create=Mock(side_effect=[response(first,'length'),response({'summary':'A menu','facts':['Navigation menu.']})])
    client=NS(chat=NS(completions=NS(create=create)))
    cache=tmp_path/'caption.json'
    with patch.object(vlm,'_client_for',return_value=client),patch.object(vlm,'image_data_url',return_value='data:image/png;base64,AA=='):
        out=vlm.caption_image(image_path=tmp_path/'unused.png',base_url='http://fixture/v1',api_key='x',model_id='fixture',cache_json=cache)
    assert create.call_count==2
    assert [x.kwargs['max_tokens'] for x in create.call_args_list]==[4096,4096]
    assert out['vlm_degraded_retry'] and out['facts']==['Navigation menu.']
    assert not cache.exists()


@pytest.mark.parametrize('budget,expected',[(1024,[1024,2048]),(6000,[6000,8192]),(8192,[8192]),(16384,[16384])])
def test_output_retry_budget_is_bounded(tmp_path,budget,expected):
    create=Mock(return_value=response({'summary':'A diagram','facts':['Output 17 V.']},'length'))
    client=NS(chat=NS(completions=NS(create=create)))
    with patch.object(vlm,'_client_for',return_value=client),patch.object(vlm,'image_data_url',return_value='data:image/png;base64,AA=='):
        out=vlm.caption_image(image_path=tmp_path/'unused.png',base_url='http://fixture/v1',api_key='x',model_id='fixture',max_tokens=budget)
    assert [x.kwargs['max_tokens'] for x in create.call_args_list]==expected
    assert out['vlm_truncated'] and out['facts']==['Output 17 V.']


def test_successful_length_retry_is_reused_without_another_call(tmp_path):
    value={'summary':'A diagram','facts':['Output 17 V.']}
    create=Mock(side_effect=[response(value,'length'),response(value)])
    client=NS(chat=NS(completions=NS(create=create)));cache=tmp_path/'caption.json'
    with patch.object(vlm,'_client_for',return_value=client),patch.object(vlm,'image_data_url',return_value='data:image/png;base64,AA=='):
        kwargs=dict(image_path=tmp_path/'unused.png',base_url='http://fixture/v1',api_key='x',model_id='fixture',cache_json=cache)
        first=vlm.caption_image(**kwargs);second=vlm.caption_image(**kwargs)
    assert not first['vlm_truncated'] and second['vlm_cache_hit']
    assert create.call_count==2 and second['facts']==value['facts']
