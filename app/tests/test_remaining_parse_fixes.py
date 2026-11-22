"""Six parse fixes left over from the audit and from the file-type tests: the number of a list item in sentence
splitting, rows of tildes outside Markdown, the locator of pieces that are joined again, the error class of a
long-page tile without a page, old Excel workbooks read like new ones, and hidden columns. Every fix has tests that
fail on the code before it; the tests named "unchanged" or "still" guard what must stay as it was.

Hidden columns were first left out, like hidden rows and hidden sheets at the time. Since then hidden content is
indexed like the rest (as hidden slides of a presentation always were), and the tests about it say so now."""
import base64
import json
import zipfile
import zlib
from pathlib import Path
from unittest import mock

import pytest
from openpyxl import Workbook
from PIL import Image

from kb_pipeline.chunking import packing
from kb_pipeline.chunking.chunker import blocks_to_chunks, chunk_text, text_for_block
from kb_pipeline.models import ParsedBlock
from kb_pipeline.parsers import native_table
from kb_pipeline.parsers.common import markdown_to_blocks
from kb_pipeline.parsers.errors import FAILURE_DETERMINISTIC, NonRetryableParseError
from kb_pipeline.parsers.native import parse_text
from kb_pipeline.parsers.notebook import parse_notebook
from kb_pipeline.parsers.pdf_longpage import restore_page_coordinates
from kb_pipeline.parsers.text_formats import parse_text_format
from kb_pipeline.parsers.xlsx_images import extract_xlsx_images
from kb_pipeline.utils import split_sentences


def chunks_of(blocks, max_tokens=400, overlap_tokens=80):
    return blocks_to_chunks(kb_id="k", file_key=1, content_version="v", parser_profile="p", blocks=blocks,
                            max_tokens=max_tokens, overlap_tokens=overlap_tokens)


# ---------------------------------------------------------------- N6: the number of a list item is not a sentence
@pytest.mark.parametrize("text,expected", [
    ("1. 断开上级电源，等待 5 分钟。\n2. 拆下前面板的四颗螺钉。", ["1. 断开上级电源，等待 5 分钟。", "2. 拆下前面板的四颗螺钉。"]),
    ("12. Tighten the screws. Then close the cover.", ["12. Tighten the screws.", "Then close the cover."]),
    ("1.2. 概述 这一节说明范围。下一句。", ["1.2. 概述 这一节说明范围。", "下一句。"]),
    ("a. first item\nb. second item", ["a. first item\nb. second item"]),
    ("Q. What is it? A. A relay.", ["Q. What is it?", "A. A relay."]),
    ("TITLE: 3. Precedent Trials", ["TITLE: 3. Precedent Trials"]),
    ("完成。1. 下一步是通电。", ["完成。", "1. 下一步是通电。"]),
    # behind marks that are not words: emphasis, a quote mark, a bullet
    ("**1. Patient Demographics**\nAge and sex.", ["**1. Patient Demographics**\nAge and sex."]),
    ("> 1. Customer satisfaction rating\n> 2. Defect rate", ["> 1. Customer satisfaction rating\n> 2. Defect rate"]),
    ("- 2. No evidence of embolism. Follow up.", ["- 2. No evidence of embolism.", "Follow up."]),
    ("*J. Name* **12**, 34 (2020).", ["*J. Name* **12**, 34 (2020)."]),
    ("__3. Scope__ applies here.", ["__3. Scope__ applies here."]),
    ("§6.4. Normal pages use the lock file. Others do not.", ["§6.4. Normal pages use the lock file.", "Others do not."]),
    ("TITLE: **2. Method**", ["TITLE: **2. Method**"]),
])
def test_the_number_of_a_list_item_stays_with_its_item(text, expected):
    assert split_sentences(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("1.\n断开上级电源", ["1.", "断开上级电源"]),                    # a number alone on its line in the source stays alone
    ("See section 3. Then go on. The end.", ["See section 3.", "Then go on.", "The end."]),
    ("He got an A. Then he left.", ["He got an A.", "Then he left."]),
    ("finished in\n2024. Then it was sold.", ["finished in\n2024.", "Then it was sold."]),
    ("你好。世界！完成", ["你好。", "世界！", "完成"]),
    ("版本 3.14 已发布。文件是 a.txt。", ["版本 3.14 已发布。", "文件是 a.txt。"]),
    ("第一段\n\n第二段", ["第一段", "第二段"]),
    ("", []),
    ("- He got an A. Then he left.", ["- He got an A.", "Then he left."]),                 # a mark, then words: no list number
    ("=============== 1. Start. Next.", ["=============== 1.", "Start.", "Next."]),        # a long rule is no such mark
])
def test_other_sentence_breaks_are_unchanged(text, expected):
    assert split_sentences(text) == expected


def test_numbered_steps_recognised_from_a_document_keep_number_and_step_on_one_line():
    block = ParsedBlock("mineru", "p", "docx", "text", "操作步骤\n\n1. 断开供电单元的上级电源，等待 5 分钟。\n\n2. 拆下前面板的四颗螺钉。\n\n"
                        "3. 检查端子有无松动，力矩应为 3.5 牛米。", "b", metadata={"section_path": ["操作步骤"]})
    lines = chunks_of([block])[0].text.splitlines()
    assert "1. 断开供电单元的上级电源，等待 5 分钟。" in lines and "3. 检查端子有无松动，力矩应为 3.5 牛米。" in lines
    assert not [line for line in lines if line.strip() in {"1.", "2.", "3."}]


def test_a_numbered_markdown_heading_keeps_its_title_on_one_line():
    blocks = markdown_to_blocks("# Report\n\n## 3. Precedent Trials\n\nTwo trials were found. Both were phase II.\n",
                                parser="native", parser_profile="p", doc_type="md")
    lines = [line for chunk in chunks_of(blocks) for line in chunk.text.splitlines()]
    assert "TITLE: 3. Precedent Trials" in lines and "Precedent Trials" not in lines


def test_a_bold_or_quoted_numbered_line_keeps_its_number():
    blocks = markdown_to_blocks("# Report\n\n**1. Patient Demographics**\n\nAge 54. Male.\n\n> 2. Defect rate below 0.5%\n",
                                parser="native", parser_profile="p", doc_type="md")
    lines = [line.strip() for chunk in chunks_of(blocks) for line in chunk.text.splitlines()]
    assert "**1. Patient Demographics**" in lines and "> 2. Defect rate below 0.5%" in lines
    assert not [line for line in lines if line in {"**1.", "> 2."}]


# ---------------------------------------------------------------- B12: a row of tildes is a fence only in Markdown
RST = """User guide
==========

Intro paragraph of the guide.

Network settings
~~~~~~~~~~~~~~~~

The device obtains an address by DHCP. Set a static address when the network has no DHCP server.

Storage settings
~~~~~~~~~~~~~~~~

Volumes are created from the storage page. Each volume needs at least two disks.
"""


def test_a_tilde_underline_in_rst_does_not_turn_the_section_into_code(tmp_path):
    path = tmp_path / "guide.rst"
    path.write_text(RST, encoding="utf-8")
    chunks = chunks_of(parse_text_format(path))
    assert {c.block.block_type for c in chunks} == {"text"}
    assert sum("The device obtains an address by DHCP." in c.text for c in chunks) == 1
    assert sum("Each volume needs at least two disks." in c.text for c in chunks) == 1


def test_a_tilde_underline_in_plain_text_does_not_open_a_fence(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("Release notes\n\nOverview\n~~~~~~~~\nThis release fixes the login timeout.\n\n"
                    "Known issues\n~~~~~~~~~~~~\nPrinting from the mobile app is slow.\n", encoding="utf-8")
    chunks = chunks_of(parse_text(path, "txt", "txt-native-v2"))
    assert {c.block.block_type for c in chunks} == {"text"}
    assert any("This release fixes the login timeout." in c.text for c in chunks)


def test_a_row_of_tildes_in_text_recognised_from_a_pdf_is_not_a_fence():
    block = ParsedBlock("mineru", "p", "pdf", "text", "第一部分的说明文字。\n~~~~~~~~~~\n第二部分的说明文字，后面还有内容。第二句。\n~~~~~~~~~~\n第三部分。",
                        "b", page_idx=1, metadata={"section_path": ["说明"]})
    chunks = chunks_of([block])
    assert {c.block.block_type for c in chunks} == {"text"}
    assert "第二句。" in chunks[0].text.splitlines()             # cut by sentence like any prose: nothing was kept whole as a listing
    assert chunk_text("甲\n~~~~\n乙。丙。\n~~~~\n丁", 400, 0, tilde_fences=False) == ["甲\n~~~~\n乙。\n丙。\n~~~~\n丁"]


def test_a_long_heading_bound_to_its_text_is_cut_by_the_same_rule():
    """A heading too long to ride on its body chunk makes the chunker cut heading and body together a second time."""
    title = "安装说明：" + "适用于机柜内全部功能单元的首次安装、搬迁后的重新安装以及更换主要部件之后的复装，" * 6 + "不含软件配置"
    body = ("安装前请阅读本节。\n~~~~~~~~~~\n"
            + "".join(f"第 {i} 步要先确认上一步已经完成，再接通电源并观察指示灯的状态。" for i in range(1, 40)) + "\n~~~~~~~~~~\n结束。")
    blocks = [ParsedBlock("mineru", "p", "pdf", "title", title, "t", page_idx=1, bbox=[100, 100, 800, 130], metadata={"section_path": ["安装说明"]}),
              ParsedBlock("mineru", "p", "pdf", "text", body, "b", page_idx=1, bbox=[100, 140, 800, 900], metadata={"section_path": ["安装说明"]})]
    first = chunks_of(blocks)[0].text
    assert "安装说明：" in first and "第 1 步" in first          # as a fence, everything after the row would be one piece apart from the heading


def test_markdown_still_reads_a_tilde_fence_as_code():
    blocks = markdown_to_blocks("# Tool\n\nRun it like this:\n\n~~~python\nprint(1)\n~~~\n\nThen read the log.\n",
                                parser="native", parser_profile="p", doc_type="md")
    code = [c for c in chunks_of(blocks) if c.block.block_type == "code"]
    assert [c.text.strip() for c in code] == ["print(1)"] and code[0].block.metadata["code_language"] == "python"


def test_a_backtick_fence_in_plain_text_is_still_code(tmp_path):
    path = tmp_path / "howto.txt"
    path.write_text("How to start\n\n```\nsystemctl start demo\n```\n\nThen check the status.\n", encoding="utf-8")
    code = [c for c in chunks_of(parse_text(path, "txt", "txt-native-v2")) if c.block.block_type == "code"]
    assert [c.text.strip() for c in code] == ["systemctl start demo"]


def test_a_text_cell_of_a_notebook_still_reads_a_tilde_fence_as_code(tmp_path):
    path = tmp_path / "n.ipynb"
    cell = {"cell_type": "markdown", "metadata": {}, "source": "Steps\n\n~~~bash\nmake test\n~~~\n\nDone.\n"}
    path.write_text(json.dumps({"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": [cell]}), encoding="utf-8")
    code = [c for c in chunks_of(parse_notebook(path, tmp_path / "cache")) if c.block.block_type == "code"]
    assert [c.text.strip() for c in code] == ["make test"]


# ---------------------------------------------------------------- B17(e): the locator of pieces joined again
def located(block_id, locator):
    return ParsedBlock("t", "p", "x", "text", "t", block_id, metadata={"source_locator": locator} if locator else {})


@pytest.mark.parametrize("locator", [{"kind": "jsonl", "line": 7}, {"kind": "notebook", "cell": 3},
                                     {"kind": "xml", "line": 12, "name": "Order", "namespace": "urn:x"}])
def test_pieces_of_one_record_keep_their_locator_when_joined(locator):
    a, b = located("a", dict(locator)), located("b", dict(locator))
    assert packing.combined_owner(b, [a, b]).metadata["source_locator"] == locator


def test_pieces_of_different_records_still_have_no_single_locator():
    a, b = located("a", {"kind": "jsonl", "line": 7}), located("b", {"kind": "jsonl", "line": 8})
    merged = packing.combined_owner(b, [a, b])
    assert "source_locator" not in merged.metadata
    assert [r["source_locator"] for r in merged.metadata["source_regions"]] == [{"kind": "jsonl", "line": 7}, {"kind": "jsonl", "line": 8}]
    plain = packing.combined_owner(located("d", None), [located("c", None), located("d", None)])
    assert "source_locator" not in plain.metadata


def test_line_ranges_are_still_merged_into_one_range():
    a, b = located("a", {"kind": "lines", "start": 3, "end": 9}), located("b", {"kind": "lines", "start": 10, "end": 14})
    assert packing.combined_owner(b, [a, b]).metadata["source_locator"] == {"kind": "lines", "start": 3, "end": 14}


def test_a_schema_definition_cut_and_joined_again_keeps_its_locator(tmp_path):
    """A definition a little over the budget is cut in two; the small second piece is joined back to the first."""
    from kb_pipeline.parsers.text_formats import parse_xsd

    rejoined = False
    for fields in range(15, 45):
        lines = "".join(f'      <xs:element name="field{i}" type="xs:string"/>\n' for i in range(fields))
        path = tmp_path / "order.xsd"
        path.write_text('<?xml version="1.0"?>\n<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema" targetNamespace="urn:x">\n'
                        '  <xs:complexType name="Order">\n    <xs:sequence>\n' + lines +
                        "    </xs:sequence>\n  </xs:complexType>\n</xs:schema>\n", encoding="utf-8")
        blocks = parse_xsd(path)
        chunks = chunks_of(blocks)
        if len(chunk_text(text_for_block(blocks[0]), 400, 80)) == 2 and len(chunks) == 1:
            rejoined = True
            assert chunks[0].block.metadata.get("source_locator") == {"kind": "xml", "line": 3, "name": "Order", "namespace": "urn:x"}
    assert rejoined


# ---------------------------------------------------------------- B17(d): a long-page tile without a page
TILE = dict(page=1, tiled=True, y0=0, y1=1000, width=600, height=4000)


@pytest.mark.parametrize("block", [
    ParsedBlock("mineru", "pdf", "pdf", "text", "x", "no-page"),
    ParsedBlock("mineru", "pdf", "pdf", "table", "x", "end", page_idx=1, metadata={"page_end": 9}),
    ParsedBlock("mineru", "pdf", "pdf", "code", "x", "label", page_idx=1, metadata={"code_context_sources": [{"page": None}]}),
    ParsedBlock("mineru", "pdf", "pdf", "text", "x", "region", page_idx=1, metadata={"source_regions": [{"page": 7}]}),
])
def test_a_tile_that_cannot_be_put_back_on_its_page_is_a_failure_no_retry_can_change(block):
    from kb_pipeline.pipeline import worker

    with pytest.raises(NonRetryableParseError) as failure:
        restore_page_coordinates([block], [dict(TILE)])
    assert worker._failure_verdict(failure.value) == FAILURE_DETERMINISTIC


# ---------------------------------------------------------------- B17(a), N7, N8: old Excel files, and what a sheet hides
# hidden_demo.xls: the workbook that demo_workbook() builds, saved as .xlsx and converted to the old format with
# LibreOffice (zlib, base64). The tests carry the file because nothing in the project can write that format.
# Its texts ("不应入库", "仅内部…") date from when hidden content was left out; it is read now.
XLS = zlib.decompress(base64.b64decode("""
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
"""))
STOCK = [
    (1, ["品类", "型号", "内部编码", "库存", "", "合格率", "", "备注组"]),             # the label merged over columns 7 and 8 shows in 8
    (2, ["品类", "型号", "内部编码", "仓库A", "仓库B", "合格率", "内部备注", "对外备注"]),
    (3, ["电源", "PS-200", "X-001", "12", "30", "10%", "仅内部一", "可公开一"]),
    (4, ["电源", "PS-300", "X-002", "5", "8", "12.5%", "仅内部二", "可公开二"]),
    (5, ["停产", "PS-100", "X-000", "0", "0", "50%", "仅内部三", "可公开三"]),                 # a hidden row
    (6, ["风扇", "FN-80", "X-003", "7", "9", "92%", "仅内部四", "可公开四"]),
]                                                                                          # columns 3 and 7 are hidden
ARCHIVE = [(1, ["不应入库"]), (2, ["隐藏表里的内容"])]                                          # a hidden sheet


def demo_workbook(path: Path) -> Path:
    """Two header rows made of merged cells (one merged range starts in a hidden column), a label merged down two
    rows, percentages, a hidden row, two hidden columns and a hidden sheet."""
    wb = Workbook()
    ws = wb.active
    ws.title = "库存"
    rows = [["品类", "型号", "内部编码", "库存", None, "合格率", "备注组", None],
            [None, None, None, "仓库A", "仓库B", None, "内部备注", "对外备注"],
            ["电源", "PS-200", "X-001", 12, 30, 0.1, "仅内部一", "可公开一"],
            [None, "PS-300", "X-002", 5, 8, 0.125, "仅内部二", "可公开二"],
            ["停产", "PS-100", "X-000", 0, 0, 0.5, "仅内部三", "可公开三"],
            ["风扇", "FN-80", "X-003", 7, 9, 0.9166666, "仅内部四", "可公开四"]]
    for r, row in enumerate(rows, 1):
        for c, value in enumerate(row, 1):
            if value is not None:
                ws.cell(r, c, value)
    for ref in ("A1:A2", "B1:B2", "C1:C2", "D1:E1", "F1:F2", "G1:H1", "A3:A4"):
        ws.merge_cells(ref)
    for ref, number_format in (("F3", "0%"), ("F4", "0.0%"), ("F5", "0%"), ("F6", "0%")):
        ws[ref].number_format = number_format
    ws.row_dimensions[5].hidden = True
    ws.column_dimensions["C"].hidden = True
    ws.column_dimensions["G"].hidden = True
    old = wb.create_sheet("旧数据")
    old["A1"] = "不应入库"
    old["A2"] = "隐藏表里的内容"
    old.sheet_state = "hidden"
    wb.save(path)
    return path


def test_an_old_excel_workbook_reads_like_the_same_workbook_saved_as_xlsx(tmp_path):
    xls = tmp_path / "demo.xls"
    xls.write_bytes(XLS)
    old_format = native_table.read_xls_rows(xls)
    new_format = native_table.read_xlsx_rows(demo_workbook(tmp_path / "demo.xlsx"))
    everything = [("库存", STOCK), ("旧数据", ARCHIVE)]      # merged cells, percentages, and what the workbook hides
    assert [(name, rows) for name, rows, _ in old_format] == everything
    assert [(name, rows) for name, rows, _ in new_format] == everything
    assert old_format[0][2] == new_format[0][2] == {1: 2, 3: 4}                       # the merges that mark a second header row
    with mock.patch.object(native_table, "load_workbook_compat", side_effect=ValueError("bad backend")):
        fallback = native_table.read_xlsx_rows(tmp_path / "demo.xlsx")
    assert [(name, rows) for name, rows, _ in fallback] == everything


def test_the_two_header_rows_of_an_old_excel_workbook_are_both_header(tmp_path):
    xls = tmp_path / "demo.xls"
    xls.write_bytes(XLS)
    text = "\n".join(b.text for b in native_table.parse_native_table(xls, max_tokens=400, overlap_tokens=80))
    # "备注组" is merged over a hidden column and one in view: it names the column in view, where the sheet shows it
    assert "HEADER: 品类 | 型号 | 内部编码 | 库存/仓库A | 仓库B | 合格率 | 内部备注 | 备注组/对外备注" in text
    assert "电源 | PS-300 | X-002 | 5 | 8 | 12.5% | 仅内部二 | 可公开二" in text.splitlines()
    for hidden in ("SHEET: 旧数据", "隐藏表里的内容", "停产 | PS-100", "X-001", "仅内部三"):      # the hidden sheet, row and columns
        assert hidden in text
    assert "0.125" not in text                                    # still written as shown: 12.5%


def test_an_old_excel_file_whose_formatting_cannot_be_read_is_still_read_by_its_values(tmp_path):
    import xlrd

    xls = tmp_path / "demo.xls"
    xls.write_bytes(XLS)
    opened = xlrd.open_workbook

    def without_formatting(path, **options):
        if options.get("formatting_info"):
            raise AssertionError("an XF record xlrd does not know")
        return opened(path, **options)

    with mock.patch("xlrd.open_workbook", without_formatting):
        sheets = native_table.read_xls_rows(xls)
    assert [name for name, *_ in sheets] == ["库存", "旧数据"]
    assert sheets[0][1][2] == (3, ["电源", "PS-200", "X-001", "12", "30", "0.1", "仅内部一", "可公开一"])


def test_an_old_excel_sheet_with_a_merge_outside_its_cells_and_an_unknown_format_is_read(tmp_path):
    """Stand-ins for what xlrd hands over: a merged range that lies below the last row with a value, and a cell
    whose format record is not in the workbook."""
    import xlrd
    from types import SimpleNamespace

    class Sheet:
        name, visibility, nrows, ncols = "S", 0, 2, 2
        merged_cells = [(5, 7, 0, 2)]
        rowinfo_map, colinfo_map = {}, {}
        grid = [["name", "share"], ["bolt", 0.25]]

        def cell_value(self, row, col):
            return self.grid[row][col]

        def cell_type(self, row, col):
            return xlrd.XL_CELL_NUMBER if isinstance(self.grid[row][col], float) else xlrd.XL_CELL_TEXT

        def cell_xf_index(self, row, col):
            return 99

    book = SimpleNamespace(datemode=0, xf_list=[], format_map={}, sheets=lambda: [Sheet()])
    with mock.patch("xlrd.open_workbook", return_value=book):
        sheets = native_table.read_xls_rows(tmp_path / "any.xls")
    assert sheets == [("S", [(1, ["name", "share"]), (2, ["bolt", "0.25"])], {6: 7})]      # the merge itself is still reported


def test_a_run_of_hidden_columns_is_read_like_the_columns_in_view(tmp_path):
    wb = Workbook()
    ws = wb.active
    for row in (["name", "cost", "margin", "supplier", "price"], ["bolt", 1.5, 0.4, "ACME", 2.1], ["nut", 0.5, 0.5, "ACME", 0.75]):
        ws.append(row)
    ws.column_dimensions.group("B", "D", hidden=True)             # one record for three columns
    path = tmp_path / "prices.xlsx"
    wb.save(path)
    expected = [(1, ["name", "cost", "margin", "supplier", "price"]), (2, ["bolt", "1.5", "0.4", "ACME", "2.1"]),
                (3, ["nut", "0.5", "0.5", "ACME", "0.75"])]
    assert native_table.read_xlsx_rows(path)[0][1] == expected
    with mock.patch.object(native_table, "load_workbook_compat", side_effect=ValueError("bad backend")):
        assert native_table.read_xlsx_rows(path)[0][1] == expected               # the fallback reader agrees
    text = "\n".join(b.text for b in native_table.parse_native_table(path, max_tokens=400, overlap_tokens=80))
    assert "bolt | 1.5 | 0.4 | ACME | 2.1" in text


@pytest.mark.parametrize("gap", ["values", "merged", "empty"])
def test_rows_that_hold_something_only_in_hidden_columns_are_read_and_do_not_end_the_sheet(tmp_path, gap):
    """The reader takes a long run of empty rows for the end of the sheet. Rows that hold something only in a hidden
    column are rows with content: they are read, and what comes after them is read too. Rows that are really empty
    end the sheet as before."""
    wb = Workbook()
    ws = wb.active
    ws.append(["name", "code", "price"])
    ws.append(["bolt", "X-1", 2.1])
    if gap == "values":
        for row_no in range(3, 9):
            ws.cell(row_no, 2, f"X-{row_no}")
    elif gap == "merged":
        ws.cell(3, 2, "spare")
        ws.merge_cells("B3:B8")                   # the cells below B3 hold nothing themselves: the label is filled down
    ws.cell(9, 1, "nut")
    ws.cell(9, 3, 0.75)
    ws.column_dimensions["B"].hidden = True
    path = tmp_path / "gap.xlsx"
    wb.save(path)
    with mock.patch.object(native_table, "_XLSX_EMPTY_RUN_LIMIT", 3):
        rows = native_table.read_xlsx_rows(path)[0][1]
    top = [(1, ["name", "code", "price"]), (2, ["bolt", "X-1", "2.1"])]
    between = {"values": [(n, ["", f"X-{n}", ""]) for n in range(3, 9)], "merged": [(n, ["", "spare", ""]) for n in range(3, 9)],
               "empty": []}[gap]
    assert rows == top + between + ([] if gap == "empty" else [(9, ["nut", "", "0.75"])])


def test_a_workbook_without_hidden_columns_reads_as_before(tmp_path):
    wb = Workbook()
    ws = wb.active
    for row in (["组", "名称", "数量"], ["甲", "螺栓", 3], [None, "螺母", 4]):
        ws.append(row)
    ws.merge_cells("A2:A3")
    ws.column_dimensions["B"].width = 30                          # a column record that hides nothing
    path = tmp_path / "plain.xlsx"
    wb.save(path)
    assert native_table.read_xlsx_rows(path)[0][1] == [(1, ["组", "名称", "数量"]), (2, ["甲", "螺栓", "3"]), (3, ["甲", "螺母", "4"])]


@pytest.mark.parametrize("hide", [False, True])
def test_a_picture_placed_in_a_cell_is_extracted_whether_or_not_its_column_is_hidden(tmp_path, hide):
    picture = tmp_path / "diagram.png"
    Image.new("RGB", (120, 60), "blue").save(picture)
    wb = Workbook()
    ws = wb.active
    ws.append(["Item", "Value"])
    ws.append(["Gate", 17])
    path = tmp_path / "sample.xlsx"
    wb.save(path)
    with zipfile.ZipFile(path) as z:
        parts = {name: z.read(name) for name in z.namelist()}
    sheet = parts["xl/worksheets/sheet1.xml"].decode()
    sheet = sheet.replace("</row>", '<c r="C2"><f>_xlfn.DISPIMG("picture-a",1)</f><v/></c></row>', 1)
    if hide:
        sheet = sheet.replace("<sheetData>", '<cols><col min="2" max="3" width="9" hidden="1" customWidth="1"/></cols><sheetData>', 1)
    parts["xl/worksheets/sheet1.xml"] = sheet.encode()
    parts["xl/cellimages.xml"] = (
        '<etc:cellImages xmlns:etc="http://www.wps.cn/officeDocument/2017/etCustomData" '
        'xmlns:xdr="http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><etc:cellImage><xdr:pic><xdr:nvPicPr>'
        '<xdr:cNvPr id="1" name="picture-a"/></xdr:nvPicPr><xdr:blipFill><a:blip r:embed="rImage"/></xdr:blipFill></xdr:pic>'
        '</etc:cellImage></etc:cellImages>').encode()
    parts["xl/_rels/cellimages.xml.rels"] = (
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rImage" '
        'Target="media/image1.png" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"/>'
        '</Relationships>').encode()
    parts["xl/media/image1.png"] = picture.read_bytes()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in parts.items():
            z.writestr(name, data)
    blocks = extract_xlsx_images(path, tmp_path / "cache")
    assert [b.metadata["source_locator"]["range"] for b in blocks] == ["C2"]
