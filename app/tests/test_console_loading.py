"""Bounded console transfers and isolation between file-list requests."""
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from kb_server.main import create_app


def test_file_list_compression_preserves_json_and_etag():
    rows = [{"file_id": f"demo:{n}", "rel_path": f"docs/example-{n}.md"} for n in range(1000)]
    with patch("kb_server.service.kb_files", return_value=rows):
        client = TestClient(create_app())
        response = client.get("/api/kbs/demo/files", headers={"Accept-Encoding": "gzip"})
        assert response.headers["Content-Encoding"] == "gzip"
        assert "Accept-Encoding" in response.headers["Vary"]
        assert response.json() == rows
        cached = client.get("/api/kbs/demo/files", headers={"Accept-Encoding": "gzip", "If-None-Match": response.headers["ETag"]})
        assert cached.status_code == 304 and not cached.content
        plain = client.get("/api/kbs/demo/files", headers={"Accept-Encoding": "identity"})
        assert "Content-Encoding" not in plain.headers
        assert plain.json() == rows


def test_slow_file_requests_recover_and_ignore_previous_library():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed for console behavior tests")
    source = Path(__file__).resolve().parents[1] / "kb_server/static/app.js"
    script = r'''
const assert = require('node:assert/strict'), fs = require('node:fs'), vm = require('node:vm');
const source = fs.readFileSync(process.argv[1], 'utf8');
function definition(name) {
  const start = source.search(new RegExp('^(?:async )?function ' + name + '\\(', 'm'));
  assert.ok(start >= 0, name);
  return source.slice(start, source.indexOf('\n}', start) + 2);
}
const harness = `
let filesRequest=null, selected={kb_id:'A',state:'active',jobs_active:1}, renders=0;
const state={files:{kbId:null,rows:[],page:1,at:0,etag:null},side:{},gp:{}};
const elements=new Map(), timers=new Map(); let nextTimer=0;
const $=id=>{if(!elements.has(id)) elements.set(id,{}); return elements.get(id);};
const setHtml=(el,html)=>{el.html=html;},t=s=>s,esc=s=>s;
const selectedEntry=()=>selected,authHeaders=()=>({}),askToken=async()=>false;
const renderFilesTable=()=>{renders++;$('#fl-wrap').html='rows:'+state.files.rows.map(x=>x.file_id).join(',');};
const GP_PENDING='pending',renderGraphPreview=()=>{};
const setTimeout=(fn,ms)=>{assert.equal(ms,20000);timers.set(++nextTimer,fn);return nextTimer;};
const clearTimeout=id=>timers.delete(id);
const requests=[];
const fetch=(url,options)=>new Promise((resolve,reject)=>{
 requests.push({url,options,resolve,reject});
 options.signal.addEventListener('abort',()=>reject(new Error('aborted')),{once:true});
});
const reply=(rows,status=200)=>({status,ok:status===200,statusText:'',headers:{get:()=> 'etag'},json:async()=>rows});
`;
const checks = `
(async()=>{
 const first=refreshFilesTab(true);
 await refreshFilesTab();await refreshFilesTab(true);
 assert.equal(requests.length,1,'polling must not duplicate a pending download');
 requests[0].resolve(reply([{file_id:'A:1'}]));await first;
 assert.equal(renders,1);assert.equal(timers.size,0);
 $('#fl-wrap').html='failed';
 const cached=refreshFilesTab(true);
 assert.equal(requests[1].options.headers['If-None-Match'],'etag');
 requests[1].resolve(reply([],304));await cached;
 assert.equal($('#fl-wrap').html,'rows:A:1','304 must restore the table after an error');
 let finishBody;
 const old=refreshFilesTab(true);
 requests[2].resolve({...reply([]),json:()=>new Promise(resolve=>finishBody=resolve)});
 await Promise.resolve();await Promise.resolve();assert.ok(finishBody);
 selected={...selected,kb_id:'B'};resetWorkspace();
 assert.equal(requests[2].options.signal.aborted,true);
 const current=refreshFilesTab(true);
 finishBody([{file_id:'A:late'}]);await old;
 assert.equal(state.files.kbId,null,'late A body must not replace B');
 requests[3].resolve(reply([{file_id:'B:1'}]));await current;
 assert.equal($('#fl-wrap').html,'rows:B:1');
 const hung=refreshFilesTab(true);
 [...timers.values()][0]();await hung;
 assert.equal(filesRequest,null);assert.match($('#fl-wrap').html,/请求超时/);
 const recovered=refreshFilesTab(true);
 requests[5].resolve(reply([],304));await recovered;
 assert.equal($('#fl-wrap').html,'rows:B:1');assert.equal(timers.size,0);
 const stale=refreshFilesTab(true);
 selected={...selected,kb_id:'C'};resetWorkspace();await stale;
 assert.match($('#fl-wrap').html,/载入中/);assert.doesNotMatch($('#fl-wrap').html,/失败/);
})()
`;
vm.runInNewContext(harness+definition('resetWorkspace')+'\n'+definition('refreshFilesTab')+checks,
 {assert,AbortController}).catch(error=>{console.error(error);process.exitCode=1;});
'''
    result = subprocess.run([node, "-e", script, str(source)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
