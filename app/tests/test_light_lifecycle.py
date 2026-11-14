"""Native graph API, version retention, scope and maintenance regression."""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace as NS

import pytest
pytest.importorskip('lancedb')
pytest.importorskip('duckdb')

from kb_pipeline.storage import get_storage
from kb_pipeline.storage.light_maintenance import inactive_gc, backfill_inactive_at, drop_collection
from kb_pipeline.vector.layout import VectorLayout
from kb_pipeline.vector.qdrant import graph_collection_name
from kb_pipeline.graph.vectors import point_id_for
from test_light_graph import bundle
from kb_search.service import FACT_FIELDS


@pytest.fixture
def native(tmp_path):
    source=NS(kb_id='synthetic', collection='synthetic', source_root='Synthetic', name='Synthetic')
    settings=NS(storage_backend='light', storage_root=tmp_path/'store', runtime_dir=tmp_path/'runtime',
                state_db=tmp_path/'state.db', sources={'synthetic':source})
    from kb_pipeline.storage.light import LightStorage
    store=LightStorage(settings)
    store.chunks.ensure(source.collection,VectorLayout(4,4))
    store.chunks.write(source.collection,['point-7'],[{'text':[1.,0.,0.,0.]}],[dict(kb_id=source.kb_id,
        doc_id='source:7',is_active=True,content_version='content1',chunk_index=0,chunk_uid='u1',
        text='ORION pressure: 12 in 2024; 15 and 16 in 2026.',token_count=20,rel_path='review.md')],max_bytes=4096)
    def stage(version,activate=True):
        b=bundle(version)
        b['documents'][0]['date']='2026-03-01'
        b['documents'][0]['content_version']='content1'
        b['entities'][0].update(aliases=['Alpha Unit'],ids=['old-orion'],external_ids=['system:X17'])
        b['entities'][1]['type']='property'
        b['has_spec_edges']=[dict(entity_uid=b['entities'][0]['uid'],spec_uid=f['uid']) for f in b['specs']]
        for f in b['specs']:
            f.update(subject='ORION',property='pressure',concept='pressure',concept_key='pressure',
                     doc_id='source:7',point_ids=['point-7'],gr_id=f['id'],content_version='content1')
        for r in b['relations']:
            r.update(source='ORION',target='pressure',traversable=True,weight=1)
        store.graph.projection.stage(source.kb_id,version,b)
        counts={kind:len(b[key]) for kind,key in (('entities','entities'),('relations','relations'),('specs','specs'))}
        with store.chunks._db() as con:
            con.execute('INSERT INTO graph_metadata VALUES(?,?,?,?,?,?,0)',
                (source.kb_id,version,source.collection,json.dumps(counts),'{}',time.time()))
        for kind,key in (('entity','entities'),('relation','relations'),('spec','specs')):
            name=graph_collection_name(source.collection,kind,version)
            store.graph.vectors.ensure(name,4)
            store.graph.vectors.write(name,[point_id_for(version,r['id']) for r in b[key]],
                [[1.,0.,0.,0.] for _ in b[key]],[dict(r,gr_id=r['id']) for r in b[key]])
        if activate: store.graph.activate(source=source,graph_version=version)
        return b
    stage('v1')
    return store,source,stage


def test_facts_time_conflicts_series_pagination_and_identity(native):
    store,source,_=native
    graph=store.graph
    current=graph.facts(source,fields=FACT_FIELDS,subject='AlphaUnit')
    assert current['found'] and current['total']==4
    assert graph.facts(source,fields=FACT_FIELDS,subject_id='old-orion')['total']==4
    assert graph.facts(source,fields=FACT_FIELDS,subject='X17')['total']==4
    old=graph.facts(source,fields=FACT_FIELDS,subject='ORION',as_of='2025')
    assert {r['id'] for r in old['facts']}=={'old','archived'}
    assert all('2026' not in str(r.get('series_text','')) for r in old['facts'])
    assert {r['id'] for r in graph.facts(source,fields=FACT_FIELDS,conflict_only=True)['facts']}=={'new','alternative'}
    assert graph.facts(source,fields=FACT_FIELDS,conflict_only=True,as_of='2025')['facts']==[]
    pages=[graph.facts(source,fields=FACT_FIELDS,history=True,limit=2,offset=i) for i in (0,2,4)]
    assert len({r['id'] for p in pages for r in p['facts']})==5
    assert graph.document_dates([{'kb_id':'synthetic','doc_id':'source:7','content_version':'content1'}],{'synthetic':'v1'})[('synthetic','source:7')]=='2026-03-01'
    assert graph.neighbors(source,entity='ORION',entity_id=None,direction='out')['neighbors']
    assert graph.neighbors(source,entity='ORION',entity_id=None,direction='in')['neighbors']==[]
    from kb_search.library import StaleReference
    with pytest.raises(StaleReference): graph.facts(source,fields=FACT_FIELDS,expected_graph_version='wrong')
    with pytest.raises(KeyError): graph.entities(NS(kb_id='absent',collection='absent'))


def test_pinned_readers_block_gc_and_published_vectors_are_immutable(native):
    store,source,stage=native
    stage('v2')
    entered=Event()
    def collect():
        entered.set()
        return store.graph.collect_vectors(source_collections=[source.collection],retention_days=1,keep_latest=1)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with store.graph.reader(source,'v1') as reader:
            future=pool.submit(collect)
            assert entered.wait(2)
            assert not future.done()
            assert reader.nodes('Entity')[0]['graph_version']=='v1'
        result=future.result(timeout=10)
    assert any(r['graph_version']=='v1' for r in result['deleted'])
    assert store.graph.stats(source.kb_id)[0]=='v2'
    name=graph_collection_name(source.collection,'entity','v2')
    with pytest.raises(ValueError,match='immutable'):
        store.graph.vectors.write(name,['bad'],[[1.,0.,0.,0.]],[{}])
    with pytest.raises(ValueError,match='immutable'): store.graph.vectors.delete_except(name,[])


def test_staged_retention_dry_run_and_orphan_cleanup(native):
    store,source,stage=native
    stage('v2',activate=False)
    with store.chunks._db() as con:
        con.execute('UPDATE graph_metadata SET created_at=0 WHERE version="v2"')
    options=dict(source_collections=[source.collection],retention_days=1)
    assert not store.graph.collect_vectors(**options,protect_versions=['v2'])['deleted']
    assert store.graph.collect_vectors(**options,dry_run=True)['deleted']
    assert store.graph.projection._version(source.kb_id,'v2')
    store.graph.collect_vectors(**options)
    with pytest.raises(KeyError): store.graph.projection._version(source.kb_id,'v2')
    orphan=store.graph.projection.stage(source.kb_id,'interrupted',bundle('interrupted'))
    path=store.chunks.root/'lance'/(orphan['generation']+'_nodes.lance')
    os.utime(path,(1,1))
    assert not store.graph.collect_vectors(**options,protect_versions=['interrupted'])['deleted']
    assert store.graph.collect_vectors(**options)['deleted'][0]['reason']=='orphan_projection'
    assert store.graph.stats(source.kb_id)[0]=='v1'


def test_inactive_reclamation_preserves_active_and_recent_rows(native):
    store,source,_=native
    values=[dict(kb_id=source.kb_id,doc_id=f'synthetic:{i}',content_version='c',is_active=False,
                 **({'inactive_at':stamp} if stamp is not None else {})) for i,stamp in enumerate((1,time.time(),None))]
    store.chunks.write(source.collection,['old','recent','missing'],[{'text':[1.,0.,0.,0.]}]*3,values,max_bytes=4096)
    assert inactive_gc(store.chunks,source.collection,time.time()-86400,dry_run=True)==1
    assert backfill_inactive_at(store.chunks,source.collection,time.time())==1
    assert inactive_gc(store.chunks,source.collection,time.time()-86400)==1
    assert set(store.chunks.payloads(source.collection,['old','recent','missing','point-7'],active_only=False))=={'recent','missing','point-7'}
    drop_collection(store.chunks,source.collection)
    assert not store.chunks.exists(source.collection)
    assert store.graph.stats(source.kb_id)[0]=='v1'


def test_request_deadlines_are_local_and_console_skips_system_services(native, monkeypatch):
    store,source,_=native
    scoped=store.with_deadline(time.monotonic()-1)
    with pytest.raises(TimeoutError): scoped.chunks.payloads(source.collection,['point-7'])
    with pytest.raises(TimeoutError): scoped.graph.entities(source)
    with pytest.raises(TimeoutError):
        store.graph.neighborhoods({source.kb_id:source},{source.kb_id:'v1'},'ORION',[],limit=2,timeout=0)
    assert store.chunks.payloads(source.collection,['point-7'])
    from kb_server import service
    monkeypatch.setattr(service,'settings',lambda:store.settings)
    monkeypatch.setattr(service.requests,'get',lambda *a,**kw:pytest.fail('Light console must not probe system services'))
    assert service.managed_services()==()
    assert service.health()['system_services'] is False


def test_weekly_compaction_preserves_content_vectors_keywords_and_recent_snapshots(native, monkeypatch):
    from datetime import timedelta
    from kb_pipeline import maintenance
    from lancedb.table import LanceTable
    store,source,_=native
    settings=store.settings
    settings.cache_dir=settings.runtime_dir/'cache'
    settings.log_dir=settings.runtime_dir/'logs'
    for path in (settings.cache_dir,settings.log_dir): path.mkdir(parents=True)
    for index in range(4):
        store.chunks.write(source.collection,[f'point-{index}'],[{'text':[0.,1.,0.,0.]}],
            [dict(kb_id=source.kb_id,doc_id=f'synthetic:{index}',is_active=True,content_version='c',
                  text=f'Synthetic valve {index}',rel_path=f'valve-{index}.md')],max_bytes=4096)
    ids=['point-7',*[f'point-{i}' for i in range(4)]]
    before=store.chunks.payloads(source.collection,ids)
    vectors=store.chunks.vectors(source.collection,ids)
    keywords=store.keywords.candidates('ORION',source.collection,limit=5)
    versions={v['version'] for v in store.chunks._table(source.collection).list_versions()}
    orphan='vectors_'+'a'*64
    recent='vectors_'+'b'*64
    unrelated='user_table'
    for name in (orphan,recent,unrelated):
        store.chunks.connection.create_table(name,data=[dict(id='synthetic')])
        if name!=recent:
            os.utime(store.chunks.root/'lance'/(name+'.lance'),(1,1))
    calls=[]
    original=LanceTable.optimize
    def optimize(table,**kwargs):
        calls.append(kwargs)
        return original(table,**kwargs)
    monkeypatch.setattr(LanceTable,'optimize',optimize)
    monkeypatch.setattr(maintenance,'service_busy',lambda s:(False,[]))
    assert maintenance.weekly_cache_cleanup(settings,dry_run=True)['native_storage']['collections']==[source.collection]
    assert not calls
    assert store.chunks.connection.open_table(orphan).count_rows()==1
    result=maintenance.weekly_cache_cleanup(settings)['native_storage']
    assert result['collections']==[source.collection]
    assert result['orphan_tables']==[orphan]
    assert not (store.chunks.root/'lance'/(orphan+'.lance')).exists()
    assert store.chunks.connection.open_table(recent).count_rows()==1
    assert store.chunks.connection.open_table(unrelated).count_rows()==1
    assert calls==[dict(cleanup_older_than=timedelta(days=7),delete_unverified=False)]
    assert store.chunks.payloads(source.collection,ids)==before
    assert store.chunks.vectors(source.collection,ids)==vectors
    assert store.keywords.candidates('ORION',source.collection,limit=5)==keywords
    assert versions<={v['version'] for v in store.chunks._table(source.collection).list_versions()}
    assert store.graph.facts(source,fields=FACT_FIELDS,subject='ORION')['total']==4


def test_shared_parse_asset_gc_reclaims_native_points_ledger_and_cache(tmp_path, monkeypatch):
    from test_maintenance_ops import _GcWorld
    from kb_pipeline import maintenance
    from kb_pipeline.vector import qdrant
    w=_GcWorld(str(tmp_path))
    try:
        fk=w.add_file('synthetic.pdf',version='current')
        live=w.add_batch(fk,'current','live',2,status='active')
        old=w.add_batch(fk,'old','expired',2,status='inactive',inactive_days_ago=10)
        recent=w.add_batch(fk,'recent','recent',1,status='inactive',inactive_days_ago=2)
        live_cache=w.cache(fk,'current'); old_cache=w.cache(fk,'old'); recent_cache=w.cache(fk,'recent')
        w.settings.storage_backend='light'; w.settings.storage_root=tmp_path/'native'; w.settings.runtime_dir=tmp_path/'runtime'
        store=get_storage(w.settings)
        store.chunks.ensure(w.KB,VectorLayout(4,4))
        rows=w.q.retrieve(w.KB,ids=live+old+recent,with_payload=True,with_vectors=True)
        store.chunks.write(w.KB,[str(r.id) for r in rows],[r.vector for r in rows],[r.payload for r in rows],max_bytes=4096)
        monkeypatch.setattr(maintenance,'service_busy',lambda *a:(False,[]))
        monkeypatch.setattr(qdrant,'client',lambda *a,**kw:pytest.fail('Light maintenance must not connect to Qdrant'))
        preview=maintenance.parse_assets_gc(w.settings,retention_days=7,dry_run=True)
        assert preview['errors']==[] and old_cache.exists()
        result=maintenance.parse_assets_gc(w.settings,retention_days=7)
        assert result['errors']==[],result
        assert result['totals']['qdrant_points']==2
        assert not store.chunks.payloads(w.KB,old,active_only=False)
        assert len(store.chunks.payloads(w.KB,live+recent,active_only=False))==3
        assert live_cache.exists() and recent_cache.exists() and not old_cache.exists()
        assert w.rows(fk)=={'active':2,'inactive':1}
    finally:
        w.q.close()
