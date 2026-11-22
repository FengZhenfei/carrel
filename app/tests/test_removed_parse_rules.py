"""Parse rules taken out in 2026-11 because they did no useful work on real documents, and what the code does now
where they stood: the model's reading of a picture is kept as it is (it used to be rewritten towards the caption);
a PDF page whose text the recogniser did not return stops the parse (it used to be rendered and sent again, and
indexed with a mark when that did not help); a table cell that looks like several values run together is indexed as
recognised (it used to be marked from the look of the string and sent to the image model to be split). The tests
named "still" guard what must stay as it was."""
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from PIL import Image

from kb_pipeline import db
from kb_pipeline.models import ParsedBlock, UnifiedChunk
from kb_pipeline.parsers import mineru_docx, mineru_pdf, pdf_enhanced, table_repair, visual_blocks
from kb_pipeline.parsers import pdf_textlayer as tl
from kb_pipeline.parsers.errors import FAILURE_DETERMINISTIC, NonRetryableParseError
from kb_pipeline.pipeline import parse_job, worker


def page_block(page, text, block_type="text", **meta):
    return ParsedBlock("mineru", "p", "pdf", block_type, text, f"b-{page}-{len(text)}", page_idx=page, metadata=dict(meta))


# ---------------------------------------------------------------- the model's reading of a picture is kept as it is
def described(tmp_path, answer, **block_fields):
    png = tmp_path / "figure.png"
    Image.new("RGB", (640, 480), (200, 210, 220)).save(png)
    block = ParsedBlock("mineru", "p", "pdf", "image", "", "img", page_idx=1, visual_ref=str(png), **block_fields)
    with mock.patch.object(visual_blocks, "caption_images_parallel", lambda jobs, **kw: {jobs[0][0]: dict(answer)}):
        visual_blocks.enrich_blocks_with_vlm([block], base_url="http://x/v1", api_key="k", model_id="m", cache_dir=tmp_path)
    return block


def test_a_number_the_model_read_from_a_picture_is_not_changed_to_the_number_in_the_caption(tmp_path):
    """The caption gives one number, the picture shows others. The removed step took them for readings of the same
    metric and rewrote both to the caption's number."""
    block = described(tmp_path, {"summary": "心率趋势图，最高心率 90，最低心率 54", "text_verbatim": "", "kind": "chart",
                                 "facts": ["最高心率 90", "最低心率 54"], "confidence": "high"}, caption="心率 60")
    assert block.metadata["visual_facts"] == ["最高心率 90", "最低心率 54"]
    assert "FACTS: 最高心率 90" in block.text and "FACTS: 最低心率 54" in block.text
    assert "VISUAL SUMMARY: 心率趋势图，最高心率 90，最低心率 54" in block.text
    assert "visual_value_conflicts" not in block.metadata


def test_a_reading_that_differs_from_the_source_text_next_to_the_picture_is_kept_too(tmp_path):
    block = described(tmp_path, {"summary": "综合风险指数为 25", "text_verbatim": "", "kind": "chart",
                                 "facts": ["综合风险指数 25", "供电可靠性指数 27"], "confidence": "high"},
                      metadata={"visual_source_context": "综合风险指数 28 中低等级\n供电可靠性指数 27"})
    assert block.metadata["visual_facts"] == ["综合风险指数 25", "供电可靠性指数 27"]
    assert "综合风险指数为 25" in block.text and "FACTS: 综合风险指数 25" in block.text


def test_the_chunk_payload_has_no_field_for_rewritten_values(tmp_path):
    block = described(tmp_path, {"summary": "柱状图", "text_verbatim": "", "kind": "chart", "facts": ["华东 120"], "confidence": "high"})
    block.metadata["visual_value_conflicts"] = [{"label": "华东", "text_value": "125", "model_value": "120"}]   # as the removed step left it
    settings = SimpleNamespace(cache_dir=tmp_path, embedding_model_id="m")
    row = {"kb_id": "k", "file_key": 1, "source_path": "r/a.pdf", "rel_path": "a.pdf", "filename": "a.pdf", "dir": "",
           "content_version": "v", "size": 1, "mtime": 1, "mime_type": "application/pdf"}
    payload = parse_job._payload_for_chunk(settings, row, UnifiedChunk("u", 0, block.text, block, 1), 1)
    assert "visual_value_conflicts" not in payload
    assert payload["visual_facts"] == ["华东 120"]                 # the description itself still travels with the chunk


# ---------------------------------------------------------------- a page whose text the recogniser lost stops the parse
def test_pages_whose_chinese_text_did_not_come_back_are_found():
    blocks = [page_block(1, "封面"), page_block(2, "1 2 3"), page_block(3, "图里的字", "chart")]
    layer = {1: 2, 2: 40, 3: 30}
    assert tl.lost_text_pages(blocks, Path("/x.pdf"), layer=layer, detail=lambda pages, boxes: {2: (40, 40), 3: (30, 30)}) == [2, 3]
    # only text drawn with fonts the file does not embed counts: the recogniser renders embedded fonts itself
    assert tl.lost_text_pages(blocks, Path("/x.pdf"), layer=layer, detail=lambda pages, boxes: {2: (40, 0), 3: (30, 0)}) == []
    # text inside a region recognised as a figure is the image model's to read: detail reports it as not outside
    assert tl.lost_text_pages(blocks, Path("/x.pdf"), layer=layer, detail=lambda pages, boxes: {2: (40, 40), 3: (0, 0)}) == [2]
    # the text came back: nothing to report, and the fonts are not even looked at
    whole = [page_block(1, "封面"), page_block(2, "汉" * 30)]
    assert tl.lost_text_pages(whole, Path("/x.pdf"), layer={1: 2, 2: 40}, detail=lambda pages, boxes: 1 / 0) == []
    # no text layer to compare with (no PyMuPDF, unreadable file): the check is skipped
    assert tl.lost_text_pages(blocks, Path("/x.pdf"), layer={}) == []
    assert tl.lost_text_pages(blocks, Path("/no/such/file.pdf")) == []


def parse_pdf(tmp_path, blocks, details):
    """parse_pdf_enhanced with the recogniser and the two reads of the PDF replaced; returns (result or error,
    names of the files sent to the recogniser, cache folder)."""
    cache = tmp_path / "cache"
    (cache / "mineru").mkdir(parents=True)
    (cache / "mineru" / "result.json").write_text("{}", encoding="utf-8")
    (cache / "mineru" / "result.md").write_text("", encoding="utf-8")
    sent = []

    def fake_mineru(*, path, **kw):
        sent.append(path.name)
        return [ParsedBlock(b.parser, b.parser_profile, b.doc_type, b.block_type, b.text, b.block_id, page_idx=b.page_idx,
                            metadata=dict(b.metadata)) for b in blocks]

    with mock.patch.object(pdf_enhanced, "mineru_pdf_blocks", fake_mineru), \
            mock.patch.object(tl, "text_layer_cjk_by_page", return_value={1: 40, 2: 7}), \
            mock.patch.object(tl, "page_details", return_value=details):
        try:
            out = pdf_enhanced.parse_pdf_enhanced(mineru_url="http://x", vlm_base_url="u", vlm_api_key="", vlm_model_id="m",
                                                  vlm_concurrency=1, path=tmp_path / "a.pdf", cache_dir=cache)
        except Exception as exc:        # noqa: BLE001 - the caller looks at the type
            out = exc
    return out, sent, cache


def test_a_pdf_with_a_page_that_lost_its_text_is_not_indexed(tmp_path):
    out, sent, cache = parse_pdf(tmp_path, [page_block(1, "12 34"), page_block(2, "尾页文字")], {1: (40, 40)})
    assert isinstance(out, NonRetryableParseError)
    assert "pages 1 (present in the text layer of the PDF)" in str(out) and "fonts" in str(out)
    assert sent == ["a.pdf"]                                       # asked once: no pages are rendered and sent again
    assert not (cache / "rendered").exists()
    # the result that lost the text is not kept, so the next parse asks the recogniser again
    assert not (cache / "mineru" / "result.json").exists() and not (cache / "mineru" / "result.md").exists()
    # a retry meets the same recogniser: the job fails once and is not queued again every day
    assert worker._failure_verdict(out) == FAILURE_DETERMINISTIC


def test_a_long_list_of_lost_pages_is_cut_short_in_the_message(tmp_path):
    blocks = [page_block(n, "1 2") for n in range(1, 31)]
    with mock.patch.object(pdf_enhanced, "mineru_pdf_blocks", lambda **kw: blocks), \
            mock.patch.object(tl, "text_layer_cjk_by_page", return_value={n: 40 for n in range(1, 31)}), \
            mock.patch.object(tl, "page_details", return_value={n: (40, 40) for n in range(1, 31)}):
        with pytest.raises(NonRetryableParseError, match=r"pages 1, 2, .*, 20 and more \(present"):
            pdf_enhanced.parse_pdf_enhanced(mineru_url="http://x", vlm_base_url="u", vlm_api_key="", vlm_model_id="m",
                                            vlm_concurrency=1, path=tmp_path / "a.pdf", cache_dir=tmp_path / "cache")


def test_a_pdf_whose_text_came_back_is_still_parsed_and_carries_no_mark(tmp_path):
    out, sent, cache = parse_pdf(tmp_path, [page_block(1, "汉" * 30), page_block(2, "尾页文字")], {})
    assert [b.page_idx for b in out] and not any(b.metadata.get("degraded") for b in out)
    assert sent == ["a.pdf"] and (cache / "mineru" / "result.json").exists()
    # few Chinese characters came back, but the page draws them with embedded fonts: not this kind of loss
    out, sent, _ = parse_pdf(tmp_path / "second", [page_block(1, "12 34"), page_block(2, "尾页文字")], {1: (40, 0)})
    assert not isinstance(out, Exception) and sent == ["a.pdf"]


def test_the_pdf_route_still_gets_the_registered_vision_endpoint_and_no_repair_mode(tmp_path):
    seen = {}
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4 demo")
    state_db = tmp_path / "state.db"
    db.init_db(state_db)
    with db.connect(state_db) as con:
        db.upsert_llm(con, name="vision-a", base_url="http://vision.invalid/v1", api_key="", model_id="synthetic-vision", multimodal=True)
    settings = SimpleNamespace(max_file_bytes=0, job_max_retries=5, cache_dir=tmp_path / "cache", mineru_url="http://x",
                               state_db=state_db, vlm_concurrency=1, vlm_temperature=0.1, vlm_top_p=0.8, vlm_max_tokens=64,
                               vlm_structured_output=True, image_max_pixels=1, mineru_timeout_seconds=5)
    source = SimpleNamespace(vlm_model="vision-a", vlm_prompt=None, graph_language=None, block_merge_tokens=800, max_tokens=400, overlap_tokens=0)
    row = {"physical_path": str(pdf), "size": 13, "mtime": int(pdf.stat().st_mtime), "kb_id": "kb_x", "file_key": 1,
           "content_version": "v", "source_path": "r/a.pdf"}
    with mock.patch.object(parse_job, "parse_pdf_enhanced", lambda **kw: seen.update(kw) or []):
        parse_job._parse_blocks(settings, source, pdf, tmp_path / "cache", "p", row)
    assert (seen["vlm_base_url"], seen["vlm_api_key"], seen["vlm_model_id"]) == ("http://vision.invalid/v1", "", "synthetic-vision")
    assert "text_layer_repair" not in seen


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_the_console_shows_why_the_parse_stopped(lang):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js needed to run the console's translation table")
    message = ("text layer lost: the recognition result lacks the Chinese text of pages 3, 4 (present in the text layer of the PDF); "
               "check the CJK fonts of the recognition service and parse the file again")
    script = r"""
const fs = require('fs'), vm = require('vm');
const [file, lang, raw] = process.argv.slice(1);
const c = vm.createContext({localStorage: {getItem: () => lang}, module: {exports: {}}});
vm.runInContext(fs.readFileSync(file, 'utf8'), c);
const t = vm.runInContext(lang === 'zh' ? 'translateZh' : 'translate', c);
console.log(JSON.stringify(JSON.parse(raw).map(s => t(s))));
"""
    texts = [message, f"NonRetryableParseError('{message}')"]
    i18n = Path(__file__).resolve().parents[1] / "kb_server/static/i18n.js"
    out = json.loads(subprocess.run([node, "-e", script, str(i18n), lang, json.dumps(texts)], check=True, capture_output=True, text=True).stdout)
    if lang == "en":
        assert out == texts
    else:
        assert out[0] == "文字层丢字:识别结果里没有第 3, 4 页的中文(PDF 文字层里有);请检查识别服务的中文字体后重新解析"
        assert out[1] == "NonRetryableParseError('" + out[0] + "')"


# ---------------------------------------------------------------- values run together in a table cell are no longer marked
GLUED_TABLE = ("<table><tr><td>参数</td><td>条件</td><td>最小值</td><td>典型值</td><td>最大值</td><td>单位</td></tr>"
               "<tr><td>电源电压</td><td></td><td>2.7</td><td>3.0</td><td>3.6</td><td>V</td></tr>"
               "<tr><td>平均电流</td><td>20 ns 25 ns 45 ns</td><td>-</td><td>-</td><td>626248</td><td>mAmAmA</td></tr></table>")
GLUED_ROW = "| 平均电流 | 20 ns 25 ns 45 ns | - | - | 626248 | mAmAmA |"
OLD_MARKS = {"table_flags": [{"row": 2, "kind": "glued_values", "confidence": "high", "raw": "626248", "key": "平均电流"}],
             "table_repair": {"status": "unverified", "reason": "no verified split"}}      # as the removed steps left them


def recognised(route, folder):
    payload = {"content_list": [{"type": "table", "table_body": GLUED_TABLE, "page_idx": 0}]}
    module = mineru_pdf if route == "pdf" else mineru_docx
    with mock.patch.object(module, "call_mineru_sync", return_value=payload):
        read = module.mineru_pdf_blocks if route == "pdf" else module.mineru_docx_blocks
        return read(mineru_url="http://x", path=folder / f"a.{route}", cache_dir=folder)


@pytest.mark.parametrize("route", ["pdf", "docx"])
def test_a_table_with_values_run_together_comes_through_as_recognised_and_unmarked(tmp_path, route):
    table, = [b for b in recognised(route, tmp_path) if b.block_type == "table"]
    assert GLUED_ROW in table.text and "| 电源电压 |  | 2.7 | 3.0 | 3.6 | V |" in table.text
    assert "table_flags" not in table.metadata


def test_the_pdf_route_asks_the_model_nothing_about_values_run_together(tmp_path):
    crop = tmp_path / "table.png"
    Image.new("RGB", (400, 200), "white").save(crop)
    asked = []

    def blocks_with_a_crop(**kw):
        blocks = recognised("pdf", tmp_path)
        for block in blocks:
            block.visual_ref = str(crop)
        return blocks

    def client(*args):
        asked.append(args)
        raise RuntimeError("the image model was asked")

    with mock.patch.object(pdf_enhanced, "mineru_pdf_blocks", blocks_with_a_crop), \
            mock.patch.object(table_repair, "_client_for", client), \
            mock.patch.object(visual_blocks, "caption_images_parallel", lambda jobs, **kw: {}):
        out = pdf_enhanced.parse_pdf_enhanced(mineru_url="http://x", vlm_base_url="http://vision.invalid/v1", vlm_api_key="k",
                                              vlm_model_id="m", vlm_concurrency=1, path=tmp_path / "a.pdf", cache_dir=tmp_path / "cache")
    table, = [b for b in out if b.block_type == "table"]
    assert asked == []
    assert GLUED_ROW in table.text
    assert "table_flags" not in table.metadata and "table_repair" not in table.metadata


def test_the_chunk_payload_has_no_fields_for_values_run_together(tmp_path):
    text = "| 参数 | 最大值 |\n| --- | --- |\n| 平均电流 | 626248 |"
    block = ParsedBlock("mineru", "p", "pdf", "table", text, "t", page_idx=1, table_markdown=text, metadata=dict(OLD_MARKS))
    settings = SimpleNamespace(cache_dir=tmp_path, embedding_model_id="m")
    row = {"kb_id": "k", "file_key": 1, "source_path": "r/a.pdf", "rel_path": "a.pdf", "filename": "a.pdf", "dir": "",
           "content_version": "v", "size": 1, "mtime": 1, "mime_type": "application/pdf"}
    payload = parse_job._payload_for_chunk(settings, row, UnifiedChunk("u", 0, text, block, 1), 1)
    assert "table_flags" not in payload and "table_repair" not in payload
    assert payload["block_type"] == "table"


def test_a_table_continued_on_the_next_page_is_joined_whatever_marks_it_carries():
    """A marked table used to be left out of the joining of tables split across pages."""
    head = "| 检查项目 | 缩写 | 测量结果 | 参考区间 | 单位 |\n| --- | --- | --- | --- | --- |"

    def table(block_id, text, page, **meta):
        return ParsedBlock("mineru", "p", "pdf", "table", text, block_id, page_idx=page, table_markdown=text,
                           bbox=[50, 50, 950, 950], metadata=dict(meta))

    first = table("t1", head + "\n| 颗粒计数 | PLT | 236 | 125—350 | ×10^9/L |", 9, **OLD_MARKS)
    rest = table("t2", "| 颗粒分布宽度 | PSW | 13.4 | 11.0—18.0 | fL |\n| --- | --- | --- | --- | --- |", 10)
    out = pdf_enhanced.merge_split_tables([first, rest])
    assert [b.block_id for b in out] == ["t1"]
    assert out[0].text.endswith("| 颗粒计数 | PLT | 236 | 125—350 | ×10^9/L |\n| 颗粒分布宽度 | PSW | 13.4 | 11.0—18.0 | fL |")
    assert out[0].metadata["page_end"] == 10


def test_the_graph_still_reads_the_marks_of_chunks_indexed_before():
    """The parser writes no marks any more. Chunks indexed before keep theirs until the file is parsed again, and a
    fact that took such a value must stay untrusted for as long."""
    from kb_pipeline.graph.units import _flagged_table_values

    flags = [{"row": 2, "kind": "glued_values", "confidence": "high", "raw": "626248"},
             {"row": 2, "kind": "glued_units", "confidence": "high", "raw": "mAmAmA"},
             {"row": 3, "kind": "glued_values", "confidence": "low", "raw": "20261121"},
             {"row": 4, "kind": "glued_values", "confidence": "high", "raw": "757557"},
             {"row": 5, "kind": "glued_values", "confidence": "high", "raw": "626248"}]
    assert _flagged_table_values(flags, None) == ["626248", "757557"]
    assert _flagged_table_values(flags, {"status": "unverified"}) == ["626248", "757557"]
    assert _flagged_table_values(flags, {"status": "verified", "rows": [2, 5]}) == ["757557"]
    assert _flagged_table_values(None, None) == [] and _flagged_table_values([], {"status": "verified"}) == []


def test_the_console_has_no_badge_filter_or_message_for_values_run_together():
    app = Path(__file__).resolve().parents[1]
    script = (app / "kb_server/static/app.js").read_text(encoding="utf-8")
    assert "table_flags" not in script and "table_repair" not in script and 'chip("ambiguous"' not in script
    words = (app / "kb_server/static/i18n.js").read_text(encoding="utf-8")
    assert "表格粘连" not in words and "Merged cells in" not in words
    listing = (app / "kb_server/service.py").read_text(encoding="utf-8")
    assert '"table_flags"' not in listing and '"table_repair"' not in listing
    # the two lines the node console replaces when it serves this file are untouched
    assert "refreshHealth();\nsetInterval(refreshHealth, 30000);" in script
