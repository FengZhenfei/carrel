"""Twelve parse fixes found by testing file types the first trial did not contain (Word, PowerPoint, spreadsheets,
notebooks, Markdown, source code) and by re-checking the PDF route. Each test fails on the code before its fix."""
import base64
import io
import json
import zipfile
from pathlib import Path
from unittest import mock

import pytest

from kb_pipeline.chunking.chunker import blocks_to_chunks
from kb_pipeline.chunking.diagnose import chunk_diagnostics
from kb_pipeline.models import ParsedBlock
from kb_pipeline.parsers import docx_enhanced, mineru_pptx, visual_blocks
from kb_pipeline.parsers.common import html_table_to_markdown, markdown_to_blocks
from kb_pipeline.parsers.notebook import parse_notebook
from kb_pipeline.parsers.pdf_enhanced import merge_mineru_text_blocks, merge_split_tables
from kb_pipeline.parsers.table_repair import review_pdf_tables


def chunks_of(blocks, max_tokens=400):
    return blocks_to_chunks(kb_id="k", file_key=1, content_version="v", parser_profile="p", blocks=blocks,
                            max_tokens=max_tokens, overlap_tokens=80)


# ---------------------------------------------------------------- B6: lines inside one table cell
@pytest.mark.parametrize("cell,expected", [
    ("75<br>75<br>57", "75 75 57"),
    ("mA<br/>mA<BR>mA", "mA mA mA"),
    ("<p>输入</p><p>输出</p>", "输入 输出"),
    ("Operating<br>temperature", "Operating temperature"),
    ("<div>220</div><div>230</div>", "220 230"),
    ("<ul><li>甲</li><li>乙</li></ul>", "甲 乙"),
    ("a<b>b</b>c<span>d</span>", "abcd"),              # inline markup is not a line break
])
def test_lines_inside_a_table_cell_are_kept_apart(cell, expected):
    text = html_table_to_markdown(f"<table><tr><td>参数</td><td>值</td></tr><tr><td>项</td><td>{cell}</td></tr></table>")
    assert text.splitlines()[2] == f"| 项 | {expected} |"


# ---------------------------------------------------------------- N2: Markdown headings without body text
def md_blocks(text):
    return markdown_to_blocks(text, parser="native", parser_profile="md", doc_type="md")


def test_a_markdown_heading_without_body_text_is_kept():
    blocks = md_blocks("# Plan\n\n## Goals\n\n## Timeline\n\n### Week 1\n\n### Week 2\n\nDesign freeze.\n\n## Budget\n")
    assert [(b.block_type, b.metadata["section_path"]) for b in blocks] == [
        ("title", ["Plan", "Goals"]),                  # used to vanish
        ("title", ["Plan", "Timeline", "Week 1"]),     # used to vanish
        ("text", ["Plan", "Timeline", "Week 2"]),      # "Timeline" needs no block: this path carries it
        ("title", ["Plan", "Budget"]),
    ]
    assert [c.text for c in chunks_of(blocks)] == ["TITLE: Goals", "TITLE: Week 1", "TITLE: Week 2\nDesign freeze.", "TITLE: Budget"]


def test_a_document_title_whose_only_subsection_is_empty_survives():
    blocks = md_blocks("# Study report\n## Format E3\n\n---\n\n# Title page\n\nSponsor and protocol number.\n")
    assert [b.metadata["section_path"] for b in blocks] == [["Study report", "Format E3"], ["Title page"]]


def test_a_heading_that_states_a_conclusion_before_a_rule_is_kept():
    blocks = md_blocks("## 评估\n\n三项指标都已达标。\n\n### 结论：可以进入下一阶段\n\n⸻\n\n## 风险\n\n交付时间偏紧。\n")
    assert [b.title for b in blocks] == ["评估", "结论：可以进入下一阶段", "风险"]


def test_markdown_block_ids_of_sections_with_text_do_not_move():
    # block ids are part of every chunk id: the headings that are now kept must not renumber the other blocks
    text = "# T\n## Empty one\n## A\n\nalpha\n\n## Empty two\n\n## B\n\nbeta\n"
    blocks = md_blocks(text)
    with_text = {b.title: b.block_id for b in blocks if b.block_type == "text"}
    assert with_text == {"A": "md-0001", "B": "md-0003"}
    assert len({b.block_id for b in blocks}) == len(blocks) == 4


# ---------------------------------------------------------------- B3 / B4: merging tables
def table_block(block_id, rows, *, page, bbox, **metadata):
    text = "\n".join(rows)
    return ParsedBlock("mineru", "pdf", "pdf", "table", text, block_id, page_idx=page, bbox=bbox, table_markdown=text,
                       metadata={"section_path": ["Specs"], **metadata})


HEAD = ["| Model | Voltage |", "| --- | --- |"]


def test_two_tables_on_an_ordinary_page_stay_apart_when_another_page_of_the_pdf_was_tiled():
    # every block of a PDF that has a long page somewhere carries source_tile, ordinary pages included
    ordinary = {"tile": 4, "page": 4, "tiled": False}
    blocks = [table_block("a", HEAD + ["| A1 | 5 |"], page=4, bbox=[100, 100, 800, 200], source_tile=dict(ordinary)),
              table_block("b", HEAD + ["| B7 | 12 |"], page=4, bbox=[100, 400, 800, 500], source_tile=dict(ordinary))]
    assert len(merge_split_tables(blocks)) == 2


def test_two_parts_of_one_table_on_tiles_of_the_same_long_page_are_still_joined():
    tiles = [{"tile": n, "page": 2, "tiled": True} for n in (2, 3)]
    blocks = [table_block("a", HEAD + ["| A1 | 5 |"], page=2, bbox=[100, 100, 800, 200], source_tile=tiles[0]),
              table_block("b", ["| B7 | 12 |", "| --- | --- |"], page=2, bbox=[100, 220, 800, 300], source_tile=tiles[1])]
    merged = merge_split_tables(blocks)
    assert len(merged) == 1 and "| B7 | 12 |" in merged[0].text
    assert merged[0].bbox == [100, 100, 800, 300]


def test_notes_written_before_the_grid_of_a_continuation_survive_the_merge():
    first = table_block("a", HEAD + ["| A1 | 5 |"], page=1, bbox=[100, 300, 800, 900])
    # the table review writes the notes of a table before its grid
    second = table_block("b", ["Measured at 25 °C unless stated otherwise."] + HEAD + ["| B7 | 12 |"], page=2, bbox=[100, 80, 800, 250])
    merged = merge_split_tables([first, second])
    assert len(merged) == 1
    assert merged[0].text.splitlines() == HEAD + ["| A1 | 5 |", "| B7 | 12 |", "Measured at 25 °C unless stated otherwise."]


# ---------------------------------------------------------------- B5 / B7: describing pictures
def picture(tmp_path, text="", **metadata):
    from PIL import Image
    path = tmp_path / "figure.png"
    Image.new("RGB", (640, 480), (200, 210, 220)).save(path)
    return ParsedBlock("mineru", "p", "pdf", "image", text, "mineru-image-00001", page_idx=3, visual_ref=str(path), metadata=dict(metadata))


def describe(block, tmp_path, *, api_key="k", **reply):
    reply = {"summary": "各地区销售额的柱状图", "text_verbatim": "华东 120 华北 95", "facts": ["华东销售额 120"], "kind": "chart",
             "confidence": "high", **reply}
    seen = {}

    def fake(jobs, **kwargs):
        seen.update(kwargs)
        return {jobs[0][0]: dict(reply)}

    with mock.patch.object(visual_blocks, "caption_images_parallel", fake):
        visual_blocks.enrich_blocks_with_vlm([block], base_url="http://x/v1", api_key=api_key, model_id="m", cache_dir=tmp_path)
    return seen


NOTE = "注：数据截至 2026 年 6 月，不含港澳台地区。"


def test_a_note_attached_to_a_figure_survives_the_description(tmp_path):
    # the recogniser's own reading of the picture ("l20" for 120) is replaced; the note printed under it is not
    block = picture(tmp_path, "华东 l20 华北 95\nFOOTNOTE: " + NOTE, image_text_source="model", attached_footnotes=[NOTE])
    describe(block, tmp_path)
    assert "FOOTNOTE: " + NOTE in block.text and "l20" not in block.text
    assert block.metadata["own_text_dropped"] is True


def test_the_parsers_own_footnote_of_a_figure_survives_without_region_evidence(tmp_path):
    block = picture(tmp_path, "华东 l20 华北 95\nFOOTNOTE: " + NOTE, image_text_source="model", source_footnote=NOTE)
    describe(block, tmp_path)
    assert block.text.count(NOTE) == 1


def test_a_confirmed_reading_is_kept_once_with_its_note(tmp_path):
    block = picture(tmp_path, "华东 120 华北 95\nFOOTNOTE: " + NOTE, image_text_source="model", attached_footnotes=[NOTE])
    describe(block, tmp_path)
    # the note used to make confirmation impossible: no model writes "FOOTNOTE:" into its transcription
    assert block.text.count("华东 120 华北 95") == 1 and block.text.count(NOTE) == 1
    assert "own_text_dropped" not in block.metadata


def test_a_note_already_among_the_source_text_is_not_written_twice(tmp_path):
    block = picture(tmp_path, "华东 l20\nFOOTNOTE: " + NOTE, image_text_source="model", source_footnote=NOTE,
                    visual_source_context="华东 120 华北 95\n" + NOTE)
    describe(block, tmp_path)
    assert block.text.count(NOTE) == 1 and "SOURCE TEXT: " in block.text


def test_the_merger_records_the_note_it_attaches_to_a_figure(tmp_path):
    figure = picture(tmp_path, "", section_path=["销售"])
    note = ParsedBlock("mineru", "p", "pdf", "text", NOTE, "mineru-text-00002", page_idx=3, metadata={"section_path": ["销售"]})
    after = ParsedBlock("mineru", "p", "pdf", "text", "下一段正文，" * 30, "mineru-text-00003", page_idx=3, metadata={"section_path": ["销售"]})
    merged = merge_mineru_text_blocks([figure, note, after], target_tokens=800)
    assert merged[0].metadata["attached_footnotes"] == [NOTE] and merged[0].text == "FOOTNOTE: " + NOTE


def test_a_model_without_a_key_is_asked_for_the_description(tmp_path):
    block = picture(tmp_path)
    seen = describe(block, tmp_path, api_key="")
    assert seen["api_key"] == "" and block.metadata["vlm_status"] == "success"      # used to be "skipped", with no request
    assert "vlm_skip_reason" not in block.metadata


def test_tables_are_reviewed_through_a_model_without_a_key(tmp_path):
    image = tmp_path / "table.png"
    image.write_bytes(b"synthetic image; the model answer is stubbed")
    text = "| Item | Value |\n| --- | --- |\n| Alpha | 17 V |"
    block = ParsedBlock("fixture", "fixture", "pdf", "table", text, "table", table_markdown=text,
                        metadata={"table_review_reasons": ["native_text_missing"], "table_review_image": str(image)})
    with mock.patch("kb_pipeline.parsers.table_repair.transcribe_table", return_value={"rows": [["Item", "Value"], ["Alpha", "17 V"]]}) as call:
        stats = review_pdf_tables([block], base_url="fixture", api_key="", model_id="fixture", cache_dir=tmp_path)
    assert stats["requested"] == 1 and call.call_count == 1


# ---------------------------------------------------------------- B15: several symbols on one line
def js_blocks(source):
    from kb_pipeline.parsers.code_symbols import symbol_blocks
    blocks = symbol_blocks(source, "javascript")
    if blocks is None:
        pytest.skip("the JavaScript grammar is not installed")
    return blocks


def test_minified_source_is_not_copied_once_per_function():
    functions = [f"function calc{n}(a,b){{var t=a*{n}+b;if(t>{n * 3}){{return t-{n}}}return t+{n}}}" for n in range(1, 151)]
    source = "".join(functions) + "\n"
    blocks = js_blocks(source)
    assert sum(len(b.text) for b in blocks) == len(source.rstrip("\n"))      # used to be 150 times the file
    assert len(blocks) == 1 and blocks[0].title == "calc1"
    assert blocks[0].metadata["shared_line_symbols"][:2] == ["calc2", "calc3"]
    assert len(chunks_of(blocks)) < 20


def test_a_line_shared_by_two_functions_is_taken_once_and_the_rest_of_each_function_stays_with_it():
    source = "function a(x){return x+1} function b(y){\n  return y*2\n}\nfunction c(z){\n  return z-3\n}\n"
    blocks = js_blocks(source)
    assert [b.title for b in blocks] == ["a", "b", "c"]
    assert [b.text for b in blocks] == ["function a(x){return x+1} function b(y){", "  return y*2\n}", "function c(z){\n  return z-3\n}"]
    assert "".join(b.text + "\n" for b in blocks) == source


def test_functions_on_lines_of_their_own_are_unchanged():
    source = "function a(x){\n  return x+1\n}\n\nfunction b(y){\n  return y*2\n}\n"
    blocks = js_blocks(source)
    assert [(b.title, b.text) for b in blocks] == [("a", "function a(x){\n  return x+1\n}"), ("b", "function b(y){\n  return y*2\n}")]
    assert all("shared_line_symbols" not in b.metadata for b in blocks)


# ---------------------------------------------------------------- N1: speaker notes
def deck(tmp_path, content):
    with mock.patch.object(mineru_pptx, "call_mineru_sync", return_value={"content_list": content}):
        blocks = mineru_pptx.mineru_pptx_blocks(mineru_url="http://x", path=tmp_path / "deck.pptx", cache_dir=tmp_path)
    return blocks, mineru_pptx.aggregate_mineru_slide_blocks(blocks, tmp_path / "deck.pptx")


def test_speaker_notes_stay_with_their_slide(tmp_path):
    content = [
        {"type": "text", "text": "产品路线图", "page_idx": 0, "text_level": 2},
        {"type": "text", "text": "第三季度发布 2.0 版本", "page_idx": 0},
        {"type": "page_footnote", "text": "这里要强调 2.0 版本比原计划推迟了六周。", "page_idx": 0},
        {"type": "page_footnote", "text": "原因是安全审计。", "page_idx": 0},
        {"type": "footer", "text": "内部资料 2026", "page_idx": 0},
        {"type": "text", "text": "定价", "page_idx": 1, "text_level": 2},
        {"type": "image", "img_path": "", "page_idx": 1},
    ]
    blocks, slides = deck(tmp_path, content)
    first = next(b for b in slides if b.slide_idx == 1)
    assert first.text == "产品路线图\n\n第三季度发布 2.0 版本\n\nSPEAKER NOTES:\n这里要强调 2.0 版本比原计划推迟了六周。\n原因是安全审计。"
    assert first.metadata["section_path"] == ["产品路线图"]
    assert "内部资料" not in "".join(b.text for b in slides)                # slide footers are still left out
    assert "SPEAKER NOTES" not in next(b for b in slides if b.slide_idx == 2).text


def test_a_slide_that_has_only_notes_is_kept_and_takes_no_title_from_them(tmp_path):
    _, slides = deck(tmp_path, [{"type": "page_footnote", "text": "过渡页，提醒听众休息十分钟。", "page_idx": 0},
                                {"type": "text", "text": "下半场", "page_idx": 1}])
    first = next(b for b in slides if b.slide_idx == 1)
    assert first.text == "SPEAKER NOTES:\n过渡页，提醒听众休息十分钟。" and first.metadata["section_path"] == []


def test_notes_do_not_renumber_the_other_blocks_of_a_deck(tmp_path):
    content = [{"type": "text", "text": "标题", "page_idx": 0}, {"type": "page_footnote", "text": "备注", "page_idx": 0},
               {"type": "table", "table_body": "<table><tr><td>a</td><td>b</td></tr></table>", "page_idx": 1}]
    blocks, _ = deck(tmp_path, content)
    assert [b.block_id for b in blocks] == ["mineru-pptx-text-00001", "mineru-pptx-note-00001", "mineru-pptx-table-00002"]


# ---------------------------------------------------------------- N3: footnotes and endnotes of a Word file
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def word_file(tmp_path, body, footnotes="", endnotes=""):
    def part(root, inner):
        return f'<?xml version="1.0" encoding="UTF-8"?><w:{root} xmlns:w="{W}">{inner}</w:{root}>'
    path = tmp_path / "manual.docx"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("word/document.xml", part("document", f"<w:body>{body}</w:body>"))
        if footnotes:
            z.writestr("word/footnotes.xml", part("footnotes", '<w:footnote w:type="separator" w:id="0"><w:p><w:r><w:separator/></w:r></w:p></w:footnote>' + footnotes))
        if endnotes:
            z.writestr("word/endnotes.xml", part("endnotes", endnotes))
    return path


def para(text, mark=""):
    return f"<w:p><w:r><w:t>{text}</w:t></w:r>{mark}</w:p>"


def foot(n, text, kind="footnote"):
    return f'<w:{kind} w:id="{n}"><w:p><w:r><w:{kind}Ref/></w:r><w:r><w:t>{text}</w:t></w:r></w:p></w:{kind}>'


def ref(n, kind="footnote"):
    return f'<w:r><w:{kind}Reference w:id="{n}"/></w:r>'


def text_block(block_id, text, section):
    return ParsedBlock("mineru", "p", "docx", "text", text, block_id, metadata={"section_path": [section]})


def merged_block(block_id, parts):
    """A block the way the text merger leaves it: paragraphs joined by a blank line, one span record for each."""
    spans, at = [], 0
    for text, section in parts:
        spans.append({"start": at, "end": at + len(text), "section_path": [section]})
        at += len(text) + 2
    return ParsedBlock("mineru", "p", "docx", "text", "\n\n".join(text for text, _ in parts), block_id,
                       metadata={"section_path": [parts[0][1]], "section_spans": spans, "merged": True})


def span_texts(block):
    return [(block.text[s["start"]:s["end"]], s["section_path"]) for s in block.metadata["section_spans"]]


def test_word_footnotes_and_endnotes_are_written_under_the_paragraph_that_refers_to_them(tmp_path):
    path = word_file(tmp_path,
                     para("适用于海拔 1000 米以下的场所", ref(2)) + para("中间一段。") + para("力矩为 3.5 牛米", ref(3) + ref(1, "endnote")),
                     foot(2, "海拔超过 2000 米时额定功率下降。") + foot(3, "使用经过校准的扭力扳手。"),
                     foot(1, "参见附录 B。", "endnote"))
    blocks = [merged_block("a", [("适用范围", "范围"), ("适用于海拔 1000 米以下的场所", "范围"), ("中间一段。", "范围")]),
              text_block("c", "**力矩为 3.5 牛米**", "步骤")]
    notes = docx_enhanced.extract_docx_note_blocks(path, blocks)
    assert [n.text for n in notes] == ["FOOTNOTE: 海拔超过 2000 米时额定功率下降。", "FOOTNOTE: 使用经过校准的扭力扳手。", "ENDNOTE: 参见附录 B。"]
    placed = docx_enhanced.attach_note_blocks(blocks, notes)
    assert [b.block_id for b in placed] == ["a", "c"]                    # no block of its own, no block id moves
    assert placed[0].text == "适用范围\n\n适用于海拔 1000 米以下的场所\nFOOTNOTE: 海拔超过 2000 米时额定功率下降。\n\n中间一段。"
    # the note belongs to its paragraph's stretch, and the stretches after it still point at their own text
    assert span_texts(placed[0]) == [("适用范围", ["范围"]), ("适用于海拔 1000 米以下的场所\nFOOTNOTE: 海拔超过 2000 米时额定功率下降。", ["范围"]),
                                     ("中间一段。", ["范围"])]
    assert placed[1].text == "**力矩为 3.5 牛米**\nFOOTNOTE: 使用经过校准的扭力扳手。\nENDNOTE: 参见附录 B。"
    assert placed[0].metadata["footnote_block_ids"] == ["docx-footnote-0001"]


def test_a_word_footnote_finds_its_paragraph_when_the_recogniser_wrote_a_link_or_left_out_a_field(tmp_path):
    linked = f'<w:p><w:r><w:t>详见</w:t></w:r><w:r><w:t>安装指南</w:t></w:r><w:r><w:t>第三章的接线要求</w:t></w:r>{ref(2)}<w:r><w:t>，然后通电。</w:t></w:r></w:p>'
    # the cross-reference "图 3" is a field; the recogniser's text of this paragraph starts after it
    field = para("图 3 展示了三种接线方式，其中星形接法最常用", ref(3))
    path = word_file(tmp_path, para("概述。") + linked + field, foot(2, "接线图在附录。") + foot(3, "三角形接法见第五章。"))
    blocks = [text_block("a", "概述。", "一"), text_block("b", "详见[安装指南](https://example.com/guide)第三章的接线要求，然后通电。", "二"),
              text_block("c", "展示了三种接线方式，其中星形接法最常用", "二"), text_block("d", "结尾。", "二")]
    placed = docx_enhanced.attach_note_blocks(blocks, docx_enhanced.extract_docx_note_blocks(path, blocks))
    assert [b.text for b in placed] == ["概述。", "详见[安装指南](https://example.com/guide)第三章的接线要求，然后通电。\nFOOTNOTE: 接线图在附录。",
                                        "展示了三种接线方式，其中星形接法最常用\nFOOTNOTE: 三角形接法见第五章。", "结尾。"]


def test_a_word_footnote_whose_place_is_not_found_goes_to_the_end_and_a_delivered_one_is_not_repeated(tmp_path):
    path = word_file(tmp_path, para("第一段", ref(2)) + para("第二段", ref(3)), foot(2, "找不到位置的脚注。") + foot(3, "已经在正文里的脚注。"))
    blocks = [text_block("a", "完全不同的文字", "一"), text_block("b", "第二段\n已经在正文里的脚注。", "一")]
    placed = docx_enhanced.attach_note_blocks(blocks, docx_enhanced.extract_docx_note_blocks(path, blocks))
    assert [b.text for b in placed] == ["完全不同的文字", "第二段\n已经在正文里的脚注。", "FOOTNOTE: 找不到位置的脚注。"]
    assert placed[2].block_id == "docx-footnote-0001"


def test_the_same_sentence_in_two_chapters_gets_each_its_own_footnotes(tmp_path):
    same = "更换熔断器时必须使用同规格产品"
    path = word_file(tmp_path, para("第一章") + para(same, ref(2) + ref(4)) + para("第二章") + para(same, ref(3)),
                     foot(2, "第一章的熔断器为 5 A。") + foot(3, "第二章的熔断器为 10 A。") + foot(4, "型号见铭牌。"))
    blocks = [merged_block("a", [("第一章", "一"), (same, "一"), ("第二章", "二"), (same, "二")])]
    placed = docx_enhanced.attach_note_blocks(blocks, docx_enhanced.extract_docx_note_blocks(path, blocks))
    assert span_texts(placed[0]) == [("第一章", ["一"]), (same + "\nFOOTNOTE: 第一章的熔断器为 5 A。\nFOOTNOTE: 型号见铭牌。", ["一"]),
                                     ("第二章", ["二"]), (same + "\nFOOTNOTE: 第二章的熔断器为 10 A。", ["二"])]


def test_footnotes_of_two_cells_go_under_their_table_in_order(tmp_path):
    cell = lambda text, mark="": f"<w:tc>{para(text, mark)}</w:tc>"
    table = f"<w:tbl><w:tr>{cell('型号')}{cell('电流')}</w:tr><w:tr>{cell('PS-200 加强型', ref(2))}{cell('75 毫安以下', ref(3))}</w:tr></w:tbl>"
    path = word_file(tmp_path, para("参数见下表。") + table + para("表后的一段。"), foot(2, "2026 年起供货。") + foot(3, "25 摄氏度下测得。"))
    grid = "| 型号 | 电流 |\n| --- | --- |\n| PS-200 加强型 | 75 毫安以下 |"
    blocks = [text_block("a", "参数见下表。", "一"),
              ParsedBlock("mineru", "p", "docx", "table", grid, "t", table_markdown=grid, metadata={"section_path": ["一"]}),
              text_block("b", "表后的一段。", "一")]
    placed = docx_enhanced.attach_note_blocks(blocks, docx_enhanced.extract_docx_note_blocks(path, blocks))
    assert placed[1].text == grid + "\nFOOTNOTE: 2026 年起供货。\nFOOTNOTE: 25 摄氏度下测得。" and placed[1].table_markdown == placed[1].text
    assert [b.text for b in (placed[0], placed[2])] == ["参数见下表。", "表后的一段。"] and len(placed) == 3


def test_a_picture_or_a_block_whose_spans_do_not_fit_takes_no_note(tmp_path):
    path = word_file(tmp_path, para("图中的文字也写着这句话", ref(2)), foot(2, "脚注。"))
    picture_block = ParsedBlock("mineru", "p", "docx", "image", "IMAGE TEXT: 图中的文字也写着这句话", "img", visual_ref=str(tmp_path / "x.png"))
    broken = merged_block("m", [("图中的文字也写着这句话", "一")])
    broken.metadata["section_spans"][0]["end"] = 999
    placed = docx_enhanced.attach_note_blocks([picture_block, broken], docx_enhanced.extract_docx_note_blocks(path, [picture_block, broken]))
    assert [b.block_id for b in placed] == ["img", "m", "docx-footnote-0001"]
    assert placed[0].text == "IMAGE TEXT: 图中的文字也写着这句话" and placed[1].text == "图中的文字也写着这句话"


def test_the_word_route_delivers_footnotes_into_the_chunk_of_their_paragraph(tmp_path):
    body = [("1 范围", "title", "1 范围"), ("适用于海拔 1000 米以下的场所", "text", "1 范围"), ("环境温度不超过 40 摄氏度。", "text", "1 范围"),
            ("2 步骤", "title", "2 步骤"), ("断开上级电源，等待 5 分钟。", "text", "2 步骤")]
    path = word_file(tmp_path, "".join(para(text, ref(2) if n == 1 else "") for n, (text, _, _) in enumerate(body)),
                     foot(2, "海拔超过 2000 米时额定功率下降。"))
    recognised = [ParsedBlock("mineru", "p", "docx", kind, text, f"mineru-{n}", metadata={"section_path": [section]})
                  for n, (text, kind, section) in enumerate(body)]
    with mock.patch.object(docx_enhanced, "mineru_docx_blocks", return_value=recognised):
        blocks = docx_enhanced.parse_docx_enhanced(mineru_url="http://x", vlm_base_url="http://x/v1", vlm_api_key="k", vlm_model_id="m",
                                                   vlm_concurrency=1, path=path, cache_dir=tmp_path / "cache")
    assert not [b for b in blocks if b.parser == "docx-ooxml-note"]             # written into the text, no block of its own
    text = "\n".join(b.text for b in blocks)
    assert "适用于海拔 1000 米以下的场所\nFOOTNOTE: 海拔超过 2000 米时额定功率下降。\n\n环境温度不超过 40 摄氏度。" in text
    whole = chunks_of(blocks)
    assert len(whole) == 1 and "适用于海拔 1000 米以下的场所\nFOOTNOTE: 海拔超过 2000 米时额定功率下降。\n环境温度不超过 40 摄氏度。" in whole[0].text
    small = chunks_of(blocks, max_tokens=24)            # cut so small that the two sections part
    owner = [c for c in small if "FOOTNOTE:" in c.text]
    assert len(owner) == 1 and owner[0].block.metadata["section_path"] == ["1 范围"]
    assert [c.block.metadata["section_path"] for c in small if "断开上级电源" in c.text] == [["2 步骤"]]


def test_a_word_file_without_notes_gives_none(tmp_path):
    assert docx_enhanced.extract_docx_note_blocks(word_file(tmp_path, para("正文。"))) == []
    assert docx_enhanced.extract_docx_note_blocks(tmp_path / "not-a-zip.docx") == [] if (tmp_path / "not-a-zip.docx").write_bytes(b"x") else True


# ---------------------------------------------------------------- notebooks: N4, N5, A6
def notebook(tmp_path, *cells):
    path = tmp_path / "nb.ipynb"
    path.write_text(json.dumps({"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": list(cells)}), encoding="utf-8")
    return parse_notebook(path, tmp_path / "cache")


def code_cell(source, *outputs):
    return {"cell_type": "code", "source": source, "execution_count": 1, "metadata": {}, "outputs": list(outputs)}


def shown(data):
    return {"output_type": "display_data", "metadata": {}, "data": data}


@pytest.mark.parametrize("mime,content,expected", [
    ("text/markdown", "**Validation loss** fell from 2.31 to 0.87.", "**Validation loss** fell from 2.31 to 0.87."),
    ("text/latex", r"$\int_0^1 x^2 dx$", r"$\int_0^1 x^2 dx$"),
    ("application/json", {"learning_rate": 0.0003}, '{\n  "learning_rate": 0.0003\n}'),
])
def test_a_notebook_output_is_read_from_the_form_that_holds_its_content(tmp_path, mime, content, expected):
    blocks = notebook(tmp_path, code_cell("show(result)", shown({mime: content, "text/plain": "<IPython.core.display.Object object>"})))
    assert [b.text for b in blocks] == ["show(result)", expected]
    assert blocks[1].metadata["source_locator"]["mime"] == mime


def test_plain_text_is_still_used_when_it_is_the_only_text_form(tmp_path):
    blocks = notebook(tmp_path, code_cell("df.sum()", shown({"text/plain": "a    3\nb    7"})))
    assert blocks[1].text == "a    3\nb    7" and blocks[1].metadata["source_locator"]["mime"] == "text/plain"


def png_bytes():
    from PIL import Image
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), (10, 120, 200)).save(buffer, "PNG")
    return buffer.getvalue()


def test_one_output_this_parser_cannot_read_does_not_cost_the_whole_notebook(tmp_path):
    blocks = notebook(tmp_path,
                      {"cell_type": "markdown", "source": "# Sensor dashboard\n\nReadings every 10 minutes.", "metadata": {}},
                      code_cell("fig.show()", shown({"application/vnd.plotly.v1+json": {"data": []}})),
                      code_cell("print(summary)", {"output_type": "stream", "name": "stdout", "text": "B-07  25.92\n"}))
    texts = [b.text for b in blocks]
    assert "Readings every 10 minutes." in texts and "B-07  25.92\n" in texts          # used to raise before reaching these
    left_out = [b for b in blocks if b.metadata.get("output_omitted")]
    assert [(b.metadata["output_omitted"], b.text) for b in left_out] == [
        ("unsupported_type", "[Saved output not indexed: unsupported type application/vnd.plotly.v1+json]")]
    assert left_out[0].metadata["source_locator"] == {"kind": "notebook", "cell": 2, "output": 1, "mime": "application/vnd.plotly.v1+json"}


def test_a_damaged_picture_is_left_out_and_the_other_outputs_are_kept(tmp_path):
    damaged = base64.b64encode(png_bytes()[:60] + b"\x00" * 40).decode()
    good = base64.b64encode(png_bytes()).decode()
    blocks = notebook(tmp_path,
                      code_cell("show(board)", shown({"image/png": damaged, "text/plain": "<Figure size 640x480 with 1 Axes>"})),
                      code_cell("show(other)", shown({"image/png": good})),
                      code_cell("show(more)", shown({"image/png": "not base64 !!"})))
    assert [b.metadata.get("output_omitted") for b in blocks if b.metadata.get("output_omitted")] == ["unreadable_image", "unreadable_image"]
    assert "<Figure size 640x480 with 1 Axes>" in [b.text for b in blocks]
    assert [b.block_type for b in blocks].count("image") == 1 and next(b for b in blocks if b.block_type == "image").visual_ref


def test_the_other_saved_form_of_a_picture_is_used_when_one_is_damaged(tmp_path):
    from PIL import Image
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), (10, 120, 200)).save(buffer, "JPEG")
    blocks = notebook(tmp_path, code_cell("show(board)", shown({"image/png": "###", "image/jpeg": base64.b64encode(buffer.getvalue()).decode()})))
    pictures = [b for b in blocks if b.block_type == "image"]
    assert len(pictures) == 1 and pictures[0].metadata["source_locator"]["mime"] == "image/jpeg"
    assert not any(b.metadata.get("output_omitted") for b in blocks)


def test_the_chunk_check_reports_outputs_that_were_left_out(tmp_path):
    blocks = notebook(tmp_path, code_cell("fig.show()", shown({"application/vnd.plotly.v1+json": {}}), shown({"image/png": "###"})),
                      code_cell("print(1)", {"output_type": "stream", "name": "stdout", "text": "1\n"}))
    check = chunk_diagnostics(chunks_of(blocks), max_tokens=400, blocks=blocks)
    assert check["stats"]["outputs_omitted"] == 2
    assert {"key": "outputs_omitted", "message": "2 saved outputs were left out: unsupported type or unreadable image", "params": [2]} in check["reasons"]
    clean = notebook(tmp_path, code_cell("print(1)", {"output_type": "stream", "name": "stdout", "text": "1\n"}))
    assert chunk_diagnostics(chunks_of(clean), max_tokens=400, blocks=clean)["stats"]["outputs_omitted"] == 0
