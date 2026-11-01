"""Execute the real formatter for current and saved diagnostic records."""
import json
import shutil
import subprocess
from pathlib import Path
import pytest


@pytest.mark.parametrize('lang',['zh','en'])
def test_every_diagnostic_current_and_saved(lang):
    node=shutil.which('node')
    if not node:pytest.skip('Node.js needed for frontend execution')
    root=Path(__file__).resolve().parents[1]/'kb_server/static'
    messages={
      'no_chunks':'No chunks were produced',
      'single_oversized':'The whole document is a single chunk of 900 tokens, over twice the budget',
      'fragmented':'7/18 text chunks are under 50 tokens: too fragmented',
      'all_tiny':'The longest chunk has only 18 tokens, far below the budget of 400',
      'oversized':'2 text chunks exceed 1.25× the budget of 400 tokens',
      'orphan_headings':'3 heading-only fragments',
      'long_visual':'2 images whose recognized text exceeds 2.5× the budget; VLM / MinerU output may have run away',
      'headerless_tables':'3/8 table chunks have no header',
      'table_source_alignment':'3 tables retain source text whose cell alignment needs review',
      'visual_output_truncated':'1 visual responses reached the output limit; only complete facts were retained'}
    script=r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const [root,lang,raw]=process.argv.slice(1);
const c=vm.createContext({localStorage:{getItem:()=>lang},module:{exports:{}}});
vm.runInContext(fs.readFileSync(root+'/i18n.js','utf8'),c);
const {diagnosticLabel,DIAGNOSTIC_LABELS}=c.module.exports;
const messages=JSON.parse(raw);
assert.deepEqual(Object.keys(DIAGNOSTIC_LABELS).sort(),Object.keys(messages).sort());
for(const [key,message] of Object.entries(messages)){
 const old={key,message};
 const current={key,message:'ignored',params:(message.match(/\d+(?:\.\d+)?/g)||[]).map(Number)};
 for(const short of [true,false]){
  const a=diagnosticLabel(old,short),b=diagnosticLabel(current,short);
  assert.equal(a,b);assert.ok(!a.includes('{'));
  assert.equal(/[\u4e00-\u9fff]/.test(a),lang==='zh');
 }
}
for (const [key,zh,en] of [
 ['single_oversized','整份文档只有 1 块却有 900 token,超过预算两倍',messages.single_oversized],
 ['oversized','2 个文本块超过预算 400 token 的 1.25 倍',messages.oversized],
 ['fragmented','7/18 个文本块不足 50 token,碎片过多',messages.fragmented]]) {
 assert.equal(diagnosticLabel({key,message:zh}),diagnosticLabel({key,message:en}));
}
assert.equal(diagnosticLabel({key:'unknown',message:'Example {model} 甲'}),'Example {model} 甲');
const app=fs.readFileSync(root+'/app.js','utf8');
assert.ok(app.includes('reasons.map(x => diagnosticLabel(x))'));
assert.ok(app.includes('esc(diagnosticLabel(x, true))'));
'''
    subprocess.run([node,'-e',script,str(root),lang,json.dumps(messages)],check=True,capture_output=True,text=True)
