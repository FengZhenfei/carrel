"""Progress calculations accept either backend language, independently of the UI language."""
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_structured_progress_labels_and_waiting(lang):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required to execute the console translation code")
    root = Path(__file__).resolve().parents[1] / "kb_server/static"
    script = r'''
const fs = require("fs"), vm = require("vm"), assert = require("assert");
const [root, lang] = process.argv.slice(1);
const c = vm.createContext({localStorage:{getItem:()=>lang}, module:{exports:{}}});
vm.runInContext(fs.readFileSync(root + "/i18n.js", "utf8"), c);
const app = fs.readFileSync(root + "/app.js", "utf8");
vm.runInContext(app.slice(app.indexOf("const PROGRESS_LABELS"), app.indexOf("// How many files are in this re-parse round")), c);
vm.runInContext(app.slice(app.indexOf("const STAGE_SPANS"), app.indexOf("const graphFloors")), c);
const run = s => vm.runInContext(s, c);
assert.equal(run('progressLabel({phase:"embed_text",completed:3,total:10})'), lang === "zh" ? "文本向量化 3/10 批" : "Text embedding 3/10 batches");
assert.equal(run('waitingLabel({wait_reason:"service",phase:"vision"})'), lang === "zh" ? "等待多模态接口" : "Waiting for vision endpoint");
assert.equal(run('waitingLabel({wait_reason:"retry"})'), lang === "zh" ? "等待重试" : "Waiting to retry");
assert.equal(run('progressLabel({phase:"compile",subphase:"sources"})'), lang === "zh" ? "编译视图页 · 来源" : "Compiling view pages · Sources");
const at = seconds => run(`stageFraction("Parsing document", {stage_elapsed:${seconds}}, {})`);
assert.equal(at(1), at(10000));
assert.ok(run('stageFraction("Text embedding", {progress:{completed:4,total:4}}, {})') > run('stageFraction("Text embedding", {progress:{completed:0,total:4}}, {})'));
assert.equal(run('t("已发布 · 清理中")'), lang === "zh" ? "已发布 · 清理中" : "Published · Cleaning up");
assert.equal(run('t("Text embedding (3/10 batches)")'), lang === "zh" ? "文本向量化(3/10 批)" : "Text embedding (3/10 batches)");
assert.equal(run('t("Visual embedding (2/8 images)")'), lang === "zh" ? "视觉向量化(2/8 张)" : "Visual embedding (2/8 images)");
'''
    subprocess.run([node, "-e", script, str(root), lang], check=True, capture_output=True, text=True)


def test_bilingual_stage_labels_and_progress():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required to execute the console translation code")
    root = Path(__file__).resolve().parents[1] / "kb_server/static"
    script = r'''
const fs = require("fs"), vm = require("vm"), assert = require("assert");
const root = process.argv[1];
for (const lang of ["zh", "en"]) {
  const context = vm.createContext({localStorage:{getItem:()=>lang}, module:{exports:{}}});
  vm.runInContext(fs.readFileSync(root + "/i18n.js", "utf8"), context);
  const app = fs.readFileSync(root + "/app.js", "utf8");
  vm.runInContext(app.slice(app.indexOf("const STAGE_SPANS"), app.indexOf("const graphFloors")), context);
  const run = text => vm.runInContext(text, context);
  for (const [zh, en] of [
    ["准备语料", "Preparing corpus"], ["实体抽取", "Entity extraction"],
    ["实体归并", "Entity resolution"], ["描述摘要", "Description summaries"],
    ["结构化事实", "Structured facts"], ["编译视图页", "Compiling view pages"],
    ["写入向量库", "Writing vectors"], ["图谱库导入", "Graph database import"],
    ["切换版本别名", "Switching version aliases"], ["清理旧版本", "Cleaning up old versions"]]) {
    assert.equal(run(`t(${JSON.stringify(en)})`), lang === "zh" ? zh : en);
    assert.equal(run(`graphStagePct(${JSON.stringify(en)})`), run(`graphStagePct(${JSON.stringify(zh)})`));
  }
  assert.equal(run("t('Entity resolution')"), lang === "zh" ? "实体归并" : "Entity resolution");
  for (const [zh, en] of [["实体归并 3/10 批", "Entity resolution 3/10 batches"],
                         ["编译视图页 · 主体 3/10", "Compiling view pages · subjects 3/10"]]) {
    assert.equal(run(`graphStagePct(${JSON.stringify(zh)})`), run(`graphStagePct(${JSON.stringify(en)})`));
  }
  assert.equal(run("stageFraction('图片描述 (VLM 3/10)')"), run("stageFraction('Describing images (VLM 3/10)')"));
  assert.equal(run("graphStagePct('完成')"), 100);
  assert.equal(run("graphStagePct('Done')"), 100);
}
'''
    subprocess.run([node, "-e", script, str(root)], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_messages_translate_templates_but_preserve_dynamic_names(lang):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required to execute the console translation code")
    script = r'''
const fs = require("fs"), vm = require("vm"), assert = require("assert");
const [file, lang] = process.argv.slice(1);
const c = vm.createContext({localStorage:{getItem:()=>lang}, module:{exports:{}}});
vm.runInContext(fs.readFileSync(file, "utf8"), c);
const {t} = c.module.exports;
for (const name of ["Example", "合成库", "Example 合成库", "甲, Phase A", "甲 $& '乙'"]) {
  const en = `Knowledge base “${name}” has no “Description summary model” selected: pick one in the console's Knowledge graph section and click Save`;
  const zh = `知识库「${name}」还没有选择「描述摘要模型」:在控制台该库的「知识图谱」一栏里选一个,然后点「保存配置」`;
  assert.equal(t(en), lang === "zh" ? zh : en);
  assert.equal(t(`ValueError: ${en}`), `ValueError: ${lang === "zh" ? zh : en}`);
  for (const original of [`Multimodal model ${name} is not in the registry`, `多模态模型 ${name} 不在模型列表中`]) {
    assert.equal(t(original), lang === "zh" ? `多模态模型 ${name} 不在模型列表中` : `Multimodal model ${name} is not in the registry`);
  }
}
for (const [zhAction, enAction] of [["这一轮不解析", "not parsed this round"], ["解析结果不写入", "the parse result was not stored"]]) {
  const zh = `文件内容在扫描之后变了(扫描时 10 字节、修改时间 20,现在 30 字节、修改时间 40),${zhAction},下一轮扫描按新内容重新排队`;
  const en = `The file content changed after the scan (10 bytes, mtime 20 at scan time, now 30 bytes, mtime 40); ${enAction}; the next scan queues it again with the new content`;
  for (const original of [zh, en]) assert.equal(t(original), lang === "zh" ? zh : en);
}
const oldError = "模型地址或协议改了,已存的 API Key 不会沿用到新地址,请重新填写 API Key";
assert.equal(t(oldError), lang === "zh" ? oldError : "The model endpoint or protocol changed; the stored API key cannot be reused for a new endpoint. Enter the API key again");
assert.equal(t("知识库控制台"), lang === "zh" ? "知识库控制台" : "Knowledge Base Console");
for (const original of ["报告 / Module-A.md", "项目 Alpha, custom_key: 未知", "Example / Draft", "未知业务声明"]) {
  assert.equal(t(original), original);
}
'''
    file = Path(__file__).resolve().parents[1] / "kb_server/static/i18n.js"
    subprocess.run([node, "-e", script, str(file), lang], check=True, capture_output=True, text=True)
