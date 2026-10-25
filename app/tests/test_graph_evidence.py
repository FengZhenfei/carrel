import copy
import unittest

from kb_pipeline.graph.compile import (
    _fact_row, _parts_section, _series_line, build_subject_page,
    compile_pages, labels_for, narrate_pages,
)


def fact(fid, value, date, **extra):
    return dict(id=fid, subject='Device A', subject_key='a', subject_keys=['a'],
                property='Output', concept='Output', concept_key='output', value=str(value),
                value_num=float(value), kinds={'value':'scalar'}, conditions={}, valid_from=date,
                doc_id='d', rel_path='report.pdf', point_ids=['point'], **extra)


class QualityPageTests(unittest.TestCase):
    def fixture(self):
        bad=fact('bad',98765,'2025-01-01',confidence='low',evidence_conflict={
            'label':'Output','text_value':'54321','model_value':'98765'})
        good=fact('good',12,'2024-01-01')
        entity={'key':'a','title':'Device A','type':'product','upper':'entity','kind':'instance',
                'doc_ids':['d'],'description':'Device summary'}
        return entity,good,bad

    def test_full_pages_preserve_conflicts_without_feeding_disputed_values_to_narration(self):
        entity,good,bad=self.fixture()
        graph={'kb_id':'test','entities':[entity],'relations':[],'specs':[good,bad],
               'documents':{'d':{'rel_path':'report.pdf'}},'profile':{'subject_types':['product']}}
        before=copy.deepcopy(graph)
        pages,_=compile_pages(graph,[],out_dir=None,language='zh',narrate=False)
        subject=next(p for p in pages if p['kind']=='subject')
        source=next(p for p in pages if p['kind']=='source')
        for page in (subject,source):
            self.assertIn('54321',page['text'])
            self.assertIn('98765',page['text'])
            self.assertIn('证据冲突',page['text'])
            self.assertIn('待核实',page['summary'])
        self.assertNotIn('98765',subject['narrate_text'])
        self.assertNotIn('54321',subject['narrate_text'])
        self.assertIn('待核实',subject['narrate_text'])
        self.assertIn('bad',subject['spec_ids'])
        self.assertEqual(graph,before)

        class SummaryClient:
            def __init__(self): self.stats={};self.prompts=[]
            def chat(self,prompt,**kwargs): self.prompts.append(prompt);return 'Device summary.'
            def run_parallel(self,items,work,progress=None):
                for item in items:yield item,work(item),None
        client=SummaryClient()
        narrate_pages(client,pages,language='zh')
        self.assertTrue(client.prompts)
        self.assertNotIn('98765',client.prompts[0])
        self.assertIn('待核实',subject['summary'])

    def test_uncertain_text_fallback_does_not_manufacture_a_trend(self):
        _,good,bad=self.fixture()
        L=labels_for('zh')
        line=_series_line([good,bad],L)
        self.assertIn('不生成趋势',line)
        self.assertNotIn('98765',line)
        self.assertNotIn('上升',line)
        ambiguous=fact('ambiguous',12345,'2026-01-01',quality='ambiguous_source')
        self.assertIn('不生成趋势',_series_line([bad,ambiguous],L))
        later=fact('later',14,'2027-01-01')
        line=_series_line([good,bad,later],L)
        self.assertIn('上升',line)
        self.assertIn('1 条待核实',line)
        self.assertNotIn('98765',line)

    def test_parts_and_unexpanded_facts_keep_quality_visible(self):
        entity,good,bad=self.fixture();L=labels_for('zh')
        lines,narration,_=_parts_section([entity],{'a':[bad]},set(),L=L)
        self.assertIn('54321','\n'.join(lines))
        self.assertIn('待核实','\n'.join(narration))
        self.assertNotIn('98765','\n'.join(narration))
        many=[]
        for i in range(45):
            f=copy.deepcopy(good);f.update(id=str(i),property=f'A{i:02}',concept=f'A{i:02}');many.append(f)
        bad['property']=bad['concept']='ZZZ'
        page=build_subject_page(entity,many+[bad],[],L=L,documents={'d':{'rel_path':'report.pdf'}},
                                mention_points={},profile={},entity_titles={})
        self.assertNotIn('### ZZZ',page['text'])
        self.assertIn('待核实: 1 条',page['text'])
        self.assertIn('待核实: 1 条',page['summary'])

    def test_normal_rendering_and_trends_stay_unchanged(self):
        _,good,_=self.fixture()
        self.assertEqual(_fact_row(good,rel_paths={}), '| 2024-01-01 | 12 |  |  |  | report.pdf |')
        line=_series_line([good,fact('later',14,'2025-01-01')],labels_for('zh'))
        self.assertEqual(line,'Output:2024-01-01 12 → 2025-01-01 14(上升,2 个点)')


class CombinedNameTypeTests(unittest.TestCase):
    def test_qualified_name_does_not_copy_product_facts_to_provider(self):
        from kb_pipeline.graph.merge import dissolve_combined_names
        def ent(key,typ,upper,scope=''):
            return dict(key=key,title=key,type=typ,upper=upper,scope=scope,frequency=1,
                        descriptions=[key],description_sources=[],unit_ids=[key],doc_ids=['d'],aliases=[])
        entities=[ent('vendor','organization','agent'),ent('model','model','entity'),
                  ent('vendor/model','model','entity')]
        original=copy.deepcopy(entities)
        rows,relations,n=dissolve_combined_names(entities,[])
        self.assertEqual(n,0)
        self.assertEqual(rows,original)
        self.assertEqual(relations,[])
        # 同一上层类的细类型不同也不能复制:软件包/模型、作品/设备等。
        entities[0].update(type='package',upper='entity')
        self.assertEqual(dissolve_combined_names(entities,[])[2],0)
        # 真实同类并列名称继续拆分,不是关闭这项能力。
        entities[0].update(type='model')
        rows,_,n=dissolve_combined_names(entities,[])
        self.assertEqual(n,1)
        self.assertEqual({e['key'] for e in rows},{'vendor','model'})
