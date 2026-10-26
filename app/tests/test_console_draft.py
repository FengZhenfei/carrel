"""Execute the enrollment form's render/save flow without a browser or live services."""
import shutil
import subprocess
from pathlib import Path

import pytest


def test_enrollment_draft_readiness_and_directory_switching():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed for console behavior tests")
    source = Path(__file__).resolve().parents[1] / "kb_server/static/app.js"
    script = r'''
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(process.argv[1], 'utf8');
function definition(name) {
  const start = source.search(new RegExp('^(?:async )?function ' + name + '\\(', 'm'));
  assert.ok(start >= 0, name);
  return source.slice(start, source.indexOf('\n}', start) + 2);
}
const definitions = ['renderConfig', 'parseFormValues', 'requireInt', 'collectParseConfig',
  'collectGraphConfig', 'collectInterval', 'collectNewChunk', 'splitInterval', 'splitNewChunk',
  'rebuildProgressParts'].map(definition).join('\n');
const saveStart = source.indexOf('$("#cfg-save").addEventListener');
const saveHandler = source.slice(saveStart, source.indexOf('\n});', saveStart) + 4);
const harness = `
let formBase = null, schemaResultApplied = '';
const draftCache = new Map(), elements = new Map(), calls = [], notices = [];
let selected = {dir: 'Library A', state: 'unenrolled'};
const state = {draft: selected.dir, cfgKey: null, llms: [], graphSchema: {}, tabByKb: {}};
function $(id) {
  if (!elements.has(id)) elements.set(id, {
    style: {}, checked: false, open: false, _value: '',
    get value() { return this._value; }, set value(v) { this._value = String(v); },
    addEventListener(event, fn) { this[event] = fn; }
  });
  return elements.get(id);
}
const selectedEntry = () => selected, t = s => s, kbDot = () => '', kbStateText = () => '';
const renderAdoptRow = () => {}, updateTabs = () => {}, syncTuneButton = () => {};
const setSampleCount = () => {}, setChunkHint = () => {}, setOverlapHint = () => {};
const refreshSchemaTaskResult = () => {}, refreshTab = () => {}, remember = () => {};
const refreshOverview = async () => {}, llmOptions = () => '', restoreDraft = () => false;
const setSchemaView = () => { state.graphSchema = {dirty: false, active: ''}; };
const toast = msg => notices.push(msg);
let read = async path => path === '/llms' ? [] : {};
async function api(path, options) {
  if (options) { calls.push({path, ...options}); return {}; }
  return read(path);
}
`;
const checks = `
(async () => {
  // Saving before defaults arrive must not submit values from another library.
  let release;
  const gate = new Promise(resolve => release = resolve);
  read = async path => { await gate; return path === '/llms' ? [] : {}; };
  const pending = renderConfig();
  await $('#cfg-save').click();
  assert.equal(calls.length, 0);
  release(); await pending;
  $('#cfg-max').value = 512;
  $('#cfg-ov').value = 64;
  await $('#cfg-save').click();
  assert.equal(calls.length, 1, 'a ready new-library form must reach enrollment');
  assert.equal(calls[0].path, '/enroll');
  assert.equal(calls[0].body.dir, 'Library A');
  assert.equal(calls[0].body.config.max_tokens, 512);
  assert.equal(calls[0].body.config.overlap_tokens, 64);
  assert.equal(calls[0].body.config.graph_enabled, false);
  assert.equal(state.draft, null);

  // A late response for A must not overwrite B's defaults or readiness.
  state.draft = selected.dir; state.cfgKey = null;
  const late = new Promise(resolve => release = resolve);
  read = async path => { await late; return path === '/llms' ? [] : {}; };
  const oldRender = renderConfig();
  selected = {dir: 'Library B', state: 'unenrolled', dir_defaults: {max_tokens: 600}};
  state.draft = selected.dir; state.cfgKey = null;
  read = async path => path === '/llms' ? [] : {};
  await renderConfig();
  release(); await oldRender;
  assert.equal(formBase.dir, 'Library B');
  assert.equal($('#cfg-max').value, '600');
  assert.equal(collectGraphConfig().graph_enabled, false);

  // Cancelling the draft clears readiness; unavailable optional hints still allow defaults.
  state.draft = null; state.cfgKey = null;
  await renderConfig();
  assert.throws(() => collectGraphConfig(), /载入中/);
  state.draft = selected.dir; state.cfgKey = null;
  read = async () => { throw new Error('temporary hint failure'); };
  await renderConfig();
  assert.equal(formBase.dir, 'Library B');
  await $('#cfg-save').click();
  assert.equal(calls.length, 2);
  assert.equal(calls[1].body.dir, 'Library B');
})()
`;
vm.runInNewContext(harness + definitions + '\n' + saveHandler + '\n' + checks, {assert})
  .catch(error => { console.error(error); process.exitCode = 1; });
'''
    result = subprocess.run([node, "-e", script, str(source)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
