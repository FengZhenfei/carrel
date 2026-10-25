import tempfile
import unittest
from pathlib import Path

from kb_pipeline.models import ParsedBlock
from kb_pipeline.parsers.pdf_longpage import prepare_long_pages, restore_page_coordinates


class LongPageTests(unittest.TestCase):
    def test_tall_page_split_preserves_text_and_original_coordinates(self):
        try:
            import pymupdf as fitz
        except ImportError:
            self.skipTest('PyMuPDF unavailable')
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / 'input.pdf'
            with fitz.open() as doc:
                p = doc.new_page(width=600, height=4800)
                for i in range(90):
                    p.insert_text((40, 60+i*50), f'ROW-{i:03} Price {1000+i} USD', fontsize=15)
                doc.new_page(width=600, height=800).insert_text((40,60),'Normal second page')
                doc.save(src)
            target, segments = prepare_long_pages(src,Path(tmp)/'cache')
            self.assertNotEqual(target,src)
            with fitz.open(src) as doc:
                lines=[line['bbox'] for b in doc[0].get_text('dict')['blocks'] for line in b.get('lines',[])]
                for s in segments:
                    if s['tiled']:
                        self.assertFalse(any(y0 < s['y1'] < y1 for x0,y0,x1,y1 in lines))
            self.assertEqual(segments[-1]['page'],2)
            self.assertFalse(segments[-1]['tiled'])
            self.assertEqual(segments[0]['y0'],0)
            self.assertEqual([s for s in segments if s['page']==1][-1]['y1'],4800)
            for a,b in zip(segments,segments[1:]):
                if a['page']==b['page']:self.assertEqual(a['y1'],b['y0'])
            b=ParsedBlock('mineru','pdf','pdf','table','| amount |','b',page_idx=2,bbox=[100,100,900,900])
            restore_page_coordinates([b],segments)
            self.assertEqual(b.page_idx,1)
            self.assertAlmostEqual(b.bbox[1],(segments[1]['y0']+.1*(segments[1]['y1']-segments[1]['y0']))/4800*1000)
            self.assertEqual(prepare_long_pages(src,Path(tmp)/'cache'),(target,segments))

    def test_normal_scanned_or_unsplittable_pages_keep_old_route(self):
        try:
            import pymupdf as fitz
        except ImportError:
            self.skipTest('PyMuPDF unavailable')
        with tempfile.TemporaryDirectory() as tmp:
            for label,height,text in [('ordinary',800,True),('blank_tall',4800,False)]:
                src=Path(tmp)/(label+'.pdf')
                with fitz.open() as doc:
                    p=doc.new_page(width=600,height=height)
                    if text:p.insert_text((30,60),'Ordinary PDF contents remain unchanged. '*4)
                    doc.save(src)
                self.assertEqual(prepare_long_pages(src,Path(tmp)/'cache'),(src,[]))

    def test_missing_tile_provenance_fails_instead_of_inventing_page(self):
        b=ParsedBlock('mineru','pdf','pdf','text','x','b')
        with self.assertRaises(ValueError):restore_page_coordinates([b],[dict(page=1,tiled=True,y0=0,y1=1000,width=600,height=4000)])

    def test_continuation_table_covers_both_original_regions(self):
        from kb_pipeline.parsers.pdf_enhanced import merge_split_tables
        rows=[]
        for i, (text, box) in enumerate([('| Item | Value |\n| --- | --- |\n| A | 12 |',[100,100,800,200]),
                                       ('| B | 34 |\n| --- | --- |',[100,220,800,300])]):
            rows.append(ParsedBlock('mineru','pdf','pdf','table',text,str(i),page_idx=1,bbox=box,
                                    table_markdown=text,metadata={'source_tile':{'tile':i+1,'page':1}}))
        merged=merge_split_tables(rows)
        self.assertEqual(len(merged),1)
        self.assertEqual(merged[0].bbox,[100,100,800,300])
        self.assertIn('| B | 34 |',merged[0].text)
