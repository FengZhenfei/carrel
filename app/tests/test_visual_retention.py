"""Synthetic visual evidence: preserve meaning and provenance, clean model-only repetition."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from kb_pipeline.models import ParsedBlock
from kb_pipeline.parsers import visual_blocks as visual
from kb_pipeline.parsers.pdf_enhanced import merge_mineru_text_blocks
from kb_pipeline.chunking.chunker import blocks_to_chunks
from kb_pipeline.pipeline.parse_job import select_visual_chunks


def chunks(bs):
    return blocks_to_chunks(kb_id="kb_fixture",file_key=1,content_version="v",parser_profile="p",
                            blocks=bs,max_tokens=120,overlap_tokens=0)


def picture(tmp_path, name="image.png", size=(40,40), **kwargs):
    p=tmp_path/name
    Image.new("RGB",size,"white").save(p)
    return ParsedBlock(parser="fixture",parser_profile="p",doc_type="pdf",block_type="image",
                       block_id=name,text=kwargs.pop("text",""),visual_ref=str(p),page_idx=3,**kwargs)


@pytest.mark.parametrize("kind", ["screenshot","chart","diagram","table","formula","photo","illustration","slide","other"])
@pytest.mark.parametrize("evidence", ["facts","entities","text_verbatim","native"])
def test_meaning_survives_any_kind_with_conflicting_decoration(tmp_path,kind,evidence):
    b=picture(tmp_path)
    response={"kind":kind,"summary":"图标 status indicator","decorative":True,"confidence":"high"}
    value="Gate A remains closed until check 17 completes."
    if evidence=="native":b.metadata["visual_source_context"]=value
    else:response[evidence]=[value] if evidence in {"facts","entities"} else value
    with patch.object(visual,"caption_images_parallel",return_value={b.block_id:response}):
        visual.enrich_blocks_with_vlm([b],base_url="fixture",api_key="fixture",model_id="fixture",cache_dir=tmp_path)
    visual.ensure_visual_blocks_have_text([b],doc_name="fixture.pdf")
    assert not b.metadata["decorative"]
    cs=chunks([b])
    assert len(select_visual_chunks(cs))==1
    assert all(c.block.page_idx==3 for c in cs)
    if evidence=="text_verbatim":assert cs[0].block.metadata["visual_text"]==value
    assert any(value in c.text for c in cs)


@pytest.mark.parametrize("name,size",[("background-architecture.png",(400,200)),("separator-status.png",(30,30)),("image.png",(79,79))])
def test_filename_and_size_do_not_filter_before_reading(tmp_path,name,size):
    b=picture(tmp_path,name,size)
    with patch.object(visual,"caption_images_parallel",return_value={b.block_id:{"summary":"Active","facts":["Valve is closed."]}}) as call:
        visual.enrich_blocks_with_vlm([b],base_url="fixture",api_key="fixture",model_id="fixture",cache_dir=tmp_path)
    assert len(call.call_args.args[0])==1
    assert "Valve is closed." in chunks([b])[0].text


@pytest.mark.parametrize("confidence",["low","medium","high"])
def test_even_empty_decoration_retains_source_image(tmp_path,confidence):
    b=picture(tmp_path)
    with patch.object(visual,"caption_images_parallel",return_value={b.block_id:{"decorative":True,"confidence":confidence}}):
        visual.enrich_blocks_with_vlm([b],base_url="fixture",api_key="fixture",model_id="fixture",cache_dir=tmp_path)
    visual.ensure_visual_blocks_have_text([b],doc_name="fixture.pdf")
    assert b.metadata["decorative"]==(confidence=="high")
    cs=chunks([b]);assert len(cs)==1 and len(select_visual_chunks(cs))==1
    assert cs[0].block.visual_ref==b.visual_ref and cs[0].block.page_idx==3


def test_native_repeated_rows_and_code_are_not_cleaned_as_model_noise(tmp_path):
    native="\n".join(["OK"]*4)+"\n```python\n    pass\n    pass\n```"
    b=picture(tmp_path,text=native)
    response={"kind":"screenshot","summary":"A checklist","facts":["State: ready."]*5}
    with patch.object(visual,"caption_images_parallel",return_value={b.block_id:response}):
        visual.enrich_blocks_with_vlm([b],base_url="fixture",api_key="fixture",model_id="fixture",cache_dir=tmp_path)
    visual.tidy_visual_blocks([b],max_tokens=8)
    assert native in b.text
    assert b.text.count("FACTS: State: ready.")==1
    assert b.metadata["visual_facts"]==["State: ready."]


def test_small_image_keeps_caption_bbox_and_does_not_take_heading(tmp_path):
    b=picture(tmp_path,bbox=[100,100,125,125],caption="Warning status")
    heading=ParsedBlock(parser="fixture",parser_profile="p",doc_type="pdf",block_type="title",block_id="h",text="Procedure",page_idx=3)
    out=merge_mineru_text_blocks([heading,b],target_tokens=800)
    assert b in out and b.caption=="Warning status" and b.title is None
    assert any(x.text=="Procedure" for x in out)
    assert b.bbox==[100,100,125,125]


def test_model_input_rejection_does_not_erase_native_evidence(tmp_path):
    b=picture(tmp_path,size=(1200,5),metadata={"visual_source_context":"Row A = 17"})
    with patch.object(visual,"caption_images_parallel") as call:
        visual.enrich_blocks_with_vlm([b],base_url="fixture",api_key="fixture",model_id="fixture",cache_dir=tmp_path)
    call.assert_not_called()
    cs=chunks([b]);assert any("Row A = 17" in c.text for c in cs)
    assert cs[0].block.visual_ref==b.visual_ref
    assert not select_visual_chunks(cs)


@pytest.mark.parametrize("region",[None,[50,100,400,200]])
def test_docx_page_citations_require_positioned_source_blocks(tmp_path,region):
    from kb_pipeline.parsers import mineru_docx
    item={'type':'text','text':'A generic source paragraph.','page_idx':0}
    if region:item['bbox']=region
    with patch.object(mineru_docx,'call_mineru_sync',return_value={}), patch.object(mineru_docx,'extract_content_list',return_value=[item]), patch.object(mineru_docx,'extract_images',return_value={}):
        bs=mineru_docx.mineru_docx_blocks(mineru_url='fixture',path=tmp_path/'source.docx',cache_dir=tmp_path)
    assert bs[0].page_idx==(1 if region else None)
    merged=merge_mineru_text_blocks(bs,target_tokens=800)
    assert all(r['page']==(1 if region else None) for r in merged[0].metadata['source_regions'])
