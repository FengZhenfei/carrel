"""Synthetic DrawingML and cell-image packages, with no corpus material."""
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

import pytest
from PIL import Image
from openpyxl import Workbook
from openpyxl.drawing.image import Image as SheetImage

from kb_pipeline.parsers.xlsx_images import extract_xlsx_images
from kb_pipeline.parsers.native_table import parse_native_table
from kb_pipeline.pipeline import parse_job
from kb_pipeline.parsers import visual_blocks


def book(tmp_path, drawing=True):
    p=tmp_path/'diagram.png';Image.new('RGB',(120,60),'blue').save(p)
    wb=Workbook();ws=wb.active;ws.title='Measurements';ws.append(['Item','Value']);ws.append(['Gate',17])
    if drawing:ws.add_image(SheetImage(p),'C2')
    hidden=wb.create_sheet('Archive');hidden.sheet_state='hidden';hidden.add_image(SheetImage(p),'A1')
    dest=tmp_path/'sample.xlsx';wb.save(dest);return dest


def rewrite(p,changes):
    with zipfile.ZipFile(p) as z:parts={n:z.read(n) for n in z.namelist()}
    parts.update(changes)
    with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
        for n,v in parts.items():z.writestr(n,v)


def test_standard_drawing_keeps_sheet_row_pixels_and_ignores_hidden_sheet(tmp_path):
    path=book(tmp_path);bs=extract_xlsx_images(path,tmp_path/'cache')
    assert len(bs)==1
    b=bs[0];assert b.sheet_name=='Measurements' and b.row_start==b.row_end==2
    assert b.metadata['source_locator']['range']=='C2'
    assert Path(b.visual_ref).read_bytes()==(tmp_path/'diagram.png').read_bytes()
    assert extract_xlsx_images(path,tmp_path/'cache')[0].block_id==b.block_id


def test_wps_cell_image_formula_is_read_without_execution(tmp_path):
    path=book(tmp_path,False)
    with zipfile.ZipFile(path) as z:sheet=z.read('xl/worksheets/sheet1.xml').decode()
    sheet=sheet.replace('</row>', '<c r="C2"><f>_xlfn.DISPIMG("picture-a",1)</f><v/></c></row>',1)
    xml='''<etc:cellImages xmlns:etc="http://www.wps.cn/officeDocument/2017/etCustomData" xmlns:xdr="http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><etc:cellImage><xdr:pic><xdr:nvPicPr><xdr:cNvPr id="1" name="picture-a"/></xdr:nvPicPr><xdr:blipFill><a:blip r:embed="rImage"/></xdr:blipFill></xdr:pic></etc:cellImage></etc:cellImages>'''
    rel='''<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rImage" Target="media/image1.png" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"/></Relationships>'''
    rewrite(path,{'xl/worksheets/sheet1.xml':sheet,'xl/cellimages.xml':xml,'xl/_rels/cellimages.xml.rels':rel})
    bs=extract_xlsx_images(path,tmp_path/'cache');assert len(bs)==1
    assert bs[0].metadata['source_locator']['range']=='C2'
    assert bs[0].row_start==2


@pytest.mark.parametrize('absolute', [False, True])
def test_repeated_picture_occurrences_keep_separate_source_identity(tmp_path, absolute):
    from copy import deepcopy
    from xml.etree import ElementTree as ET
    from kb_pipeline.parsers.xlsx_images import XDR
    path=book(tmp_path)
    with zipfile.ZipFile(path) as z:
        tree=ET.fromstring(z.read('xl/drawings/drawing1.xml'))
    anchor=tree[0]
    if absolute:
        anchor.tag=f'{{{XDR}}}absoluteAnchor'
        anchor.remove(anchor.find(f'{{{XDR}}}from'))
        anchor.insert(0, ET.Element(f'{{{XDR}}}pos', {'x':'0', 'y':'0'}))
    second=deepcopy(anchor)
    if absolute:
        second.find(f'{{{XDR}}}pos').set('x','1000000')
    tree.append(second)
    rewrite(path, {'xl/drawings/drawing1.xml':ET.tostring(tree)})
    blocks=extract_xlsx_images(path,tmp_path/'cache')
    assert len(blocks)==2
    assert len({b.block_id for b in blocks})==2
    assert len({b.visual_ref for b in blocks})==1
    assert [b.block_id for b in extract_xlsx_images(path,tmp_path/'cache')]==[b.block_id for b in blocks]
    assert all(b.metadata['source_locator']['kind']==('drawing' if absolute else 'cells') for b in blocks)


@pytest.mark.parametrize('target,mode', [('https://example.invalid/image.png','External'),('../../../../image.png',''),('file:///secret.png','')])
def test_external_and_escaping_relationships_are_never_followed(tmp_path,target,mode):
    path=book(tmp_path)
    with zipfile.ZipFile(path) as z:
        rel=z.read('xl/drawings/_rels/drawing1.xml.rels').decode()
    import re
    rel=re.sub(r'Target="[^"]+"',f'Target="{target}" TargetMode="{mode}"',rel)
    rewrite(path,{'xl/drawings/_rels/drawing1.xml.rels':rel})
    assert extract_xlsx_images(path,tmp_path/'cache')==[]


def test_plain_workbook_does_not_require_model_registry(tmp_path):
    path=book(tmp_path,False)
    cfg=SimpleNamespace(cache_dir=tmp_path/'cache',vlm_temperature=.1,vlm_top_p=.8,vlm_max_tokens=4096,
                        vlm_structured_output=True,image_max_pixels=10000)
    source=SimpleNamespace(max_tokens=400,overlap_tokens=80,vlm_prompt=None,graph_language='en')
    with patch.object(parse_job,'_vlm_endpoint',side_effect=AssertionError('text-only workbook requested model')):
        bs=parse_job._parse_blocks(cfg,source,path,tmp_path/'cache','table-native-v1',{})
    assert all(b.block_type!='image' for b in bs)


def test_media_uses_selected_model_and_preserves_table_output(tmp_path):
    path=book(tmp_path);before=parse_native_table(path,max_tokens=400,overlap_tokens=80)
    cfg=SimpleNamespace(cache_dir=tmp_path/'cache',vlm_temperature=.1,vlm_top_p=.8,vlm_max_tokens=4096,
                        vlm_structured_output=True,image_max_pixels=10000,vlm_concurrency=1)
    source=SimpleNamespace(max_tokens=400,overlap_tokens=80,vlm_prompt=None,graph_language='en')
    def caption(jobs,**kw):
        assert kw['model_id']=='selected-model'
        return {bid:{'summary':'Gate diagram','facts':['Gate B requires check A.'],'decorative':False} for bid,*rest in jobs}
    with patch.object(parse_job,'_vlm_endpoint',return_value=('url','key','selected-model')), patch.object(visual_blocks,'caption_images_parallel',caption):
        bs=parse_job._parse_blocks(cfg,source,path,tmp_path/'cache','table-native-v1',{})
    assert [(b.block_id,b.text) for b in bs if b.block_type!='image']==[(b.block_id,b.text) for b in before]
    image=next(b for b in bs if b.block_type=='image')
    assert image.row_start==2 and 'Gate B requires check A.' in image.text
