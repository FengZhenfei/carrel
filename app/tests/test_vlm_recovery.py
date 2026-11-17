"""Model-independent output handling; fixtures contain no private documents."""
import json
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from kb_pipeline.vision import vlm
from kb_pipeline.models import ParsedBlock
from kb_pipeline.parsers import visual_blocks
from kb_pipeline.chunking.chunker import blocks_to_chunks


def response(value, finish='stop'):
    return NS(choices=[NS(message=NS(content=json.dumps(value) if not isinstance(value,str) else value),finish_reason=finish)],usage=None)


def unfinished(data, field='text_verbatim'):
    fields={k:v for k,v in data.items() if k!=field}
    return json.dumps(fields)[:-1]+', "'+field+'": "'+'Home Password '*300


@pytest.mark.parametrize('field',['facts','entities','keywords'])
def test_nested_paths_and_similar_names_are_distinct(field):
    values=['Control Suite','Control Suite Office','Control Suite Office Education','Control Suite Office Education 2026']
    result=vlm.normalize_result({field:values})
    assert result[field]==values
    assert not result.get('runaway')


def test_distinct_rows_and_complete_transcription_have_no_second_length_cap():
    facts=[f'Record {i:03d}: value {i*7:04d} mV.' for i in range(180)]
    text='\n'.join(f'PIN {i:04d} | voltage {i*13:06d} mV | state {i%4}' for i in range(200))
    result=vlm.normalize_result({'facts':facts,'text_verbatim':text})
    assert len(text)>4000
    assert result['facts']==facts and result['text_verbatim']==text
    assert not result.get('runaway')


@pytest.mark.parametrize('text',['Home Password '*100,'?\n'*200,'A source sentence. '*40])
def test_completed_text_is_not_deleted_just_because_it_repeats(text):
    source='Gate A is closed. '+text+' Gate B is open.'
    result=vlm.normalize_result({'text_verbatim':source})
    assert result['text_verbatim']==source
    assert not result.get('runaway')
    assert not vlm._raw_tail_runaway(json.dumps({'text_verbatim':source}),'stop')


def test_only_dominant_unfinished_loop_prevents_larger_budget():
    assert vlm._raw_tail_runaway(unfinished({'summary':'A panel'}),'length')
    rows='\n'.join(f'PIN {i} voltage {i*13} mV' for i in range(100))
    assert not vlm._raw_tail_runaway('{"text_verbatim": "'+rows,'length')
    assert not vlm._raw_tail_runaway(json.dumps({'text_verbatim':'Home Password '*300}),'length')


def test_valid_similar_names_do_not_trigger_retry_and_are_cached(tmp_path):
    data={'summary':'Product tree','entities':['Control','Control Office','Control Office Education','Control Office Education 2026'],
          'text_verbatim':'Control\nControl Office\nControl Office Education\nControl Office Education 2026'}
    create=Mock(return_value=response(data));client=NS(chat=NS(completions=NS(create=create)))
    cache=tmp_path/'caption.json'
    with patch.object(vlm,'_client_for',return_value=client),patch.object(vlm,'image_data_url',return_value='data:,'):
        out=vlm.caption_image(image_path=tmp_path/'image.png',base_url='http://fixture/v1',api_key='fixture',model_id='fixture',cache_json=cache)
    assert create.call_count==1 and cache.exists()
    assert out['entities']==data['entities'] and out['text_verbatim']==data['text_verbatim']
    assert not out.get('vlm_degraded_retry')


@pytest.mark.parametrize('last',['repeat','empty','error'])
def test_unresolved_recovery_is_bounded_and_never_cached(tmp_path,last):
    first={'summary':'A menu','facts':['Valve A is shut.'],'text_verbatim':'Home Password '*80}
    second=response(unfinished(first),'length') if last=='repeat' else response({}) if last=='empty' else RuntimeError('unavailable')
    create=Mock(side_effect=[response(unfinished(first),'length'),second]);client=NS(chat=NS(completions=NS(create=create)))
    cache=tmp_path/'caption.json'
    with patch.object(vlm,'_client_for',return_value=client),patch.object(vlm,'image_data_url',return_value='data:,'):
        out=vlm.caption_image(image_path=tmp_path/'image.png',base_url='http://fixture/v1',api_key='fixture',model_id='fixture',cache_json=cache)
    assert create.call_count==2 and not cache.exists()
    assert out['facts']==['Valve A is shut.'] and out['vlm_runaway']


def test_long_visual_result_still_splits_and_keeps_source(tmp_path):
    image=tmp_path/'image.png';Image.new('RGB',(300,300),'white').save(image)
    facts=[f'Channel {i:03d} reports {i*7:04d} mV.' for i in range(100)]
    transcript='\n'.join(f'PIN {i:04d} voltage {i*13:06d} mV state {i%4}' for i in range(200))
    result=vlm.normalize_result({'summary':'Controller readings','facts':facts,'text_verbatim':transcript})
    block=ParsedBlock(parser='fixture',parser_profile='v',doc_type='pdf',block_type='image',block_id='image-1',text='',visual_ref=str(image),page_idx=4)
    with patch.object(visual_blocks,'caption_images_parallel',return_value={block.block_id:result}):
        visual_blocks.enrich_blocks_with_vlm([block],base_url='fixture',api_key='fixture',model_id='fixture',cache_dir=tmp_path)
    chunks=blocks_to_chunks(kb_id='fixture',file_key=1,content_version='v',parser_profile='v',blocks=[block],max_tokens=120,overlap_tokens=0)
    assert len(chunks)>1 and all(c.token_count<=150 for c in chunks)
    combined='\n'.join(c.text for c in chunks)
    assert all(f in combined for f in facts)
    assert all(c.block.visual_ref==str(image) and c.block.page_idx==4 for c in chunks)
    assert transcript==block.metadata['visual_text']
