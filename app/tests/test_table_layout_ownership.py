"""Synthetic forms, repeated row keys and continuation columns, independent of topic."""
from kb_pipeline.parsers.pdf_table_layout import attach_table_notes
from kb_pipeline.parsers.table_repair import _identity_columns, _expand_review_spans, _review_row
from kb_pipeline.parsers.common import html_table_to_grid


def table(page=0):
    return {'type':'table','page_idx':page,'bbox':[100,100,900,300],
            'table_body':'<table><tr><td>Component</td><td>Result</td></tr><tr><td>A</td><td>7</td></tr></table>'}


def note(key='Assessment', page=0, y=310):
    return [{'type':'text','text':key,'text_level':1,'page_idx':page,'bbox':[110,y,220,y+15]},
            {'type':'text','text':'Within tolerance','text_level':1,'page_idx':page,'bbox':[260,y,600,y+15]}]


def test_repeated_local_form_row_is_owned_and_source_is_preserved():
    items=[table(),*note(),table(1),*note(page=1)]
    out=attach_table_notes(items)
    assert len(out)==2
    assert out[0]['table_footnote']==['Assessment: Within tolerance']
    assert [s['text'] for s in out[0]['_pdf_table_note_sources']]==['Assessment','Within tolerance']
    assert 'table_footnote' not in items[0]  # no mutation of cached source


def test_unrelated_headings_short_prose_and_other_columns_stay_unchanged():
    for items in ([table(),*note()], [table(),*note('Result:',y=450)],
                  [table(),*note('Result:',page=1)], [table(),{'type':'text','text':'Complete short statement.'}]):
        assert attach_table_notes(items)==items
    items=[table(),*note('Result:')];items[-1]['bbox']=[910,310,990,325]
    assert attach_table_notes(items)==items


def test_notes_on_different_baselines_are_not_a_key_value_row():
    items=[table(),*note('Result:')];items[-1]['bbox']=[260,350,600,365]
    assert attach_table_notes(items)==items


def test_composite_identity_accepts_repeated_categories_but_not_duplicate_rows():
    assert _identity_columns([['Category','ID','Value'],['Part','x1','4'],['Part','x2','4']])==2
    assert _identity_columns([['Part','x1','4'],['Part','x1','4']])==0
    assert _identity_columns([['','x1'],['','x2']])==0


def test_only_declared_merged_cells_can_be_expanded():
    html='<table><tr><td>Key</td><td colspan="2">suffix</td></tr><tr><td>X</td><td>3</td><td>3</td></tr></table>'
    old=html_table_to_grid(html)
    new=[['Key','Full suffix',''],['X','3','']]
    out=_expand_review_spans(old,new,html)
    assert out==[['Key','Full suffix','Full suffix'],['X','3','']]
    assert new[0][-1]==''


def test_local_native_support_can_expand_text_but_not_move_values_or_change_units():
    assert _review_row(['Key','suffix'],['Key','Full suffix'],'Key Full suffix','Key Full suffix')
    assert not _review_row(['Key','suffix'],['Key','Full suffix'],'Key Full suffix','Different row')
    assert not _review_row(['Key','7 mg','9'],['Key','9 mg','7'],'Key 9 mg 7','Key 9 mg 7')
    assert not _review_row(['Key','7 mg'],['Key','7 g'],'Key 7 g','Key 7 g')
    assert not _review_row(['Key','allowed'],['Key','not allowed'],'Key not allowed','')


def test_unchanged_review_does_not_certify_missing_source_text(tmp_path):
    from unittest.mock import patch
    from kb_pipeline.models import ParsedBlock
    from kb_pipeline.parsers.table_repair import review_pdf_tables
    image=tmp_path/'crop.png';image.write_bytes(b'fixture')
    b=ParsedBlock(parser='fixture',parser_profile='fixture',doc_type='pdf',block_type='table',
        text='| Key | Value |\n| --- | --- |\n| A | 7 |',block_id='t',metadata={
        'table_review_reasons':['native_text_missing'],'table_missing_spans':['Additional condition'],
        'table_source_text':'Key Value A 7 Additional condition','table_review_image':str(image)})
    b.table_markdown=b.text
    with patch('kb_pipeline.parsers.table_repair.transcribe_table',return_value={
            'rows':[['Key','Value'],['A','7']],'legend':[]}):
        stats=review_pdf_tables([b],base_url='fixture',api_key='fixture',model_id='fixture',cache_dir=tmp_path)
    assert stats['accepted']==0
    assert b.metadata['table_review']['status']=='partial'
    assert 'Additional condition' in b.text


def test_native_ruled_row_proves_single_line_ownership():
    from unittest.mock import MagicMock, patch
    import pymupdf
    row=MagicMock(row_count=1,col_count=2,bbox=(100,305,900,340))
    row.extract.return_value=[['Status','Within tolerance']]
    page=MagicMock(rect=pymupdf.Rect(0,0,1000,1000))
    page.find_tables.return_value.tables=[row]
    doc=MagicMock();doc.__len__.return_value=1;doc.__getitem__.return_value=page
    item={'type':'text','text':'Status Within tolerance','text_level':1,'page_idx':0,'bbox':[110,310,600,325]}
    with patch('pymupdf.open',return_value=doc):
        result=attach_table_notes([table(),item],'fixture.pdf')
    assert len(result)==1
    assert result[0]['table_footnote']==['Status Within tolerance']
    assert result[0]['_pdf_table_note_sources'][0]['bbox']==item['bbox']
    row.extract.return_value=[['Different heading','Unrelated text']]
    with patch('pymupdf.open',return_value=doc):
        assert attach_table_notes([table(),item],'fixture.pdf')==[table(),item]


def test_native_row_anchor_does_not_require_every_other_key_to_be_one_font_span():
    from kb_pipeline.parsers.table_repair import _native_row_text
    rows=[['Heading','Value'],['Origin','suffix'],['Date','Value']]
    spans=[{'text':'Head','bbox':[10,0,30,10]},{'text':'ing','bbox':[31,0,40,10]},
           {'text':'Origin','bbox':[10,20,40,30]},{'text':'Full suffix','bbox':[50,20,100,30]},
           {'text':'Date','bbox':[10,40,40,50]},{'text':'Unrelated','bbox':[50,40,100,50]}]
    native=_native_row_text(rows,1,1,spans)
    assert 'Full suffix' in native and 'Unrelated' not in native


def test_textual_continuation_gets_header_only_with_matching_ruled_columns():
    from unittest.mock import MagicMock, patch
    import pymupdf
    from kb_pipeline.models import ParsedBlock
    from kb_pipeline.parsers.pdf_table_layout import restore_ruled_continuation_columns
    def block(text, page, box):
        return ParsedBlock(parser='fixture',parser_profile='fixture',doc_type='pdf',block_type='table',
            text=text,table_markdown=text,page_idx=page,bbox=box,block_id=str(page))
    a=block('| Item | State |\n| --- | --- |',1,[100,870,900,898])
    b=block('| Gate | Open |\n| --- | --- |\n| Valve | Closed |',2,[100,100,900,250])
    pages=[]
    for values,box in [([['Item','State']],a.bbox),([['Gate','Open'],['Valve','Closed']],b.bbox)]:
        t=MagicMock(bbox=box,col_count=2);t.extract.return_value=values
        t.rows=[MagicMock(cells=[(100,box[1],500,box[3]),(500,box[1],900,box[3])])]
        page=MagicMock(rect=pymupdf.Rect(0,0,1000,1000));page.find_tables.return_value.tables=[t];pages.append(page)
    doc=MagicMock();doc.__enter__.return_value=doc;doc.__getitem__.side_effect=lambda i:pages[i]
    with patch('pymupdf.open',return_value=doc):restore_ruled_continuation_columns([a,b],'fixture.pdf')
    assert b.table_markdown.startswith('| Item | State |')
    assert 'Gate' in b.table_markdown and 'Closed' in b.table_markdown
    assert b.metadata['inherited_table_header']['page']==1
    assert {r['page']for r in b.metadata['source_regions']}=={1,2}


def test_empty_continuation_columns_require_unique_complete_source_mapping():
    from unittest.mock import MagicMock, patch
    import pymupdf
    from kb_pipeline.models import ParsedBlock
    from kb_pipeline.parsers.pdf_table_layout import restore_ruled_continuation_columns
    from kb_pipeline.parsers.common import grid_to_markdown
    def run(native, rows):
        a=ParsedBlock(parser='fixture',parser_profile='fixture',doc_type='pdf',block_type='table',
            block_id='prior',text='| Item | Flag | Value |\n| --- | --- | --- |\n| A | | 7 |',
            page_idx=1,bbox=[100,700,900,930])
        text=grid_to_markdown(rows)
        b=ParsedBlock(parser='fixture',parser_profile='fixture',doc_type='pdf',block_type='table',
            block_id='next',text=text,table_markdown=text,page_idx=2,bbox=[100,50,900,250])
        t=MagicMock(bbox=b.bbox,col_count=3);t.extract.return_value=native
        page=MagicMock(rect=pymupdf.Rect(0,0,1000,1000));page.find_tables.return_value.tables=[t]
        doc=MagicMock();doc.__enter__.return_value=doc;doc.__getitem__.return_value=page
        with patch('pymupdf.open',return_value=doc):
            restore_ruled_continuation_columns([a,b],'fixture.pdf')
        return b,text
    b,_=run([['B','','8'],['C','','9']],[['B','8'],['C','9']])
    assert b.table_markdown==grid_to_markdown([['B','','8'],['C','','9']])
    assert b.metadata['restored_empty_columns']==[1]
    # Never insert a column if that would hide a source value or move OCR cells.
    for native,rows in [([['B','!','8'],['C','','9']],[['B','8'],['C','9']]),
                        ([['B','','8'],['C','','9']],[['C','9'],['B','8']]),
                        ([['B','',''],['C','','']],[['B',''],['C','']])]:
        b,text=run(native,rows)
        assert b.table_markdown==text and 'restored_empty_columns' not in b.metadata


def test_continuation_never_crosses_a_heading_or_an_incompatible_column_layout():
    from unittest.mock import MagicMock, patch
    import pymupdf
    from kb_pipeline.models import ParsedBlock
    from kb_pipeline.parsers.pdf_table_layout import restore_ruled_continuation_columns
    a=ParsedBlock(parser='fixture',parser_profile='fixture',doc_type='pdf',block_type='table',
        block_id='prior',text='| Item | State |\n| --- | --- |',page_idx=1,bbox=[100,870,900,898])
    text='| Gate | Open |\n| --- | --- |\n| Valve | Closed |'
    heading=ParsedBlock(parser='fixture',parser_profile='fixture',doc_type='pdf',block_type='text',
        block_id='heading',text='Another section',page_idx=2,bbox=[100,20,900,40])
    for middle,split in [([heading],500),([],650)]:
        b=ParsedBlock(parser='fixture',parser_profile='fixture',doc_type='pdf',block_type='table',
            block_id='next',text=text,table_markdown=text,page_idx=2,bbox=[100,100,900,250])
        pages=[]
        for part,values,divider in [(a,[['Item','State']],500),(b,[['Gate','Open'],['Valve','Closed']],split)]:
            t=MagicMock(bbox=part.bbox,col_count=2);t.extract.return_value=values
            t.rows=[MagicMock(cells=[(100,part.bbox[1],divider,part.bbox[3]),(divider,part.bbox[1],900,part.bbox[3])])]
            page=MagicMock(rect=pymupdf.Rect(0,0,1000,1000));page.find_tables.return_value.tables=[t];pages.append(page)
        doc=MagicMock();doc.__enter__.return_value=doc;doc.__getitem__.side_effect=lambda i:pages[i]
        with patch('pymupdf.open',return_value=doc):
            restore_ruled_continuation_columns([a,*middle,b],'fixture.pdf')
        assert b.table_markdown==text and 'inherited_table_header' not in b.metadata



def test_empty_cell_needs_its_own_native_cell_not_table_or_row_text():
    assert not _review_row(['A','',''],['A','shared note',''],'A shared note','A shared note')
    assert not _review_row(['A','',''],['A','shared note',''],'A shared note','A shared note',
                           ['A','','shared note'])
    assert _review_row(['A','',''],['A','','shared note'],'A shared note','A shared note',
                       ['A','','shared note'])


def test_uncertain_merged_row_value_stays_as_source_evidence(tmp_path):
    from unittest.mock import patch
    from kb_pipeline.models import ParsedBlock
    from kb_pipeline.parsers.table_repair import review_pdf_tables
    p=tmp_path/'crop.png';p.write_bytes(b'fixture')
    for misplaced in (['Part','x1','Shared condition'], ['Part','x2','Shared condition']):
        b=ParsedBlock(parser='fixture',parser_profile='fixture',doc_type='pdf',block_type='table',
            block_id='t',text='| Category | ID | Condition |\n| --- | --- | --- |\n| Part | x1 | |\n| Part | x2 | |',metadata={
            'table_review_reasons':['native_text_missing'],'table_missing_spans':['Shared condition'],
            'table_source_text':'Category ID Condition Part x1 Part x2 Shared condition',
            'table_review_image':str(p)})
        b.table_markdown=b.text
        rows=[['Category','ID','Condition'],['Part','x1',''],['Part','x2','']]
        rows[1 if misplaced[1]=='x1' else 2]=misplaced
        with patch('kb_pipeline.parsers.table_repair.transcribe_table',return_value={'rows':rows,'legend':[]}):
            stats=review_pdf_tables([b],base_url='fixture',api_key='fixture',model_id='fixture',cache_dir=tmp_path)
        assert stats['accepted']==0
        assert '| Part | x1 |  |' in b.text and '| Part | x2 |  |' in b.text
        assert 'SOURCE TEXT (cell alignment unverified):\nCategory ID Condition Part x1 Part x2 Shared condition' in b.text



def test_rejected_empty_cell_keeps_short_source_values_and_their_labels(tmp_path):
    from unittest.mock import patch
    from kb_pipeline.models import ParsedBlock
    from kb_pipeline.parsers.table_repair import review_pdf_tables
    p=tmp_path/'crop.png';p.write_bytes(b'fixture')
    b=ParsedBlock(parser='fixture',parser_profile='fixture',doc_type='pdf',block_type='table',block_id='t',
        text='| Field | Value |\n| --- | --- |\n| Grade | |\n| State | |',metadata={
        'table_review_reasons':['native_text_missing'],'table_missing_spans':['Ready'],
        'table_source_text':'Field Value\nGrade A\nState Ready','table_review_image':str(p)})
    b.table_markdown=b.text
    with patch('kb_pipeline.parsers.table_repair.transcribe_table',return_value={
            'rows':[['Field','Value'],['Grade','A'],['State','Ready']],'legend':[]}):
        review_pdf_tables([b],base_url='fixture',api_key='fixture',model_id='fixture',cache_dir=tmp_path)
    assert 'Grade A\nState Ready' in b.text
    assert b.metadata['table_cell_alignment_unverified']
