"""Tier-1 audit fixes in parsing and chunking (2026-11-20).

B2   heading inference sets the level of marked headings only; an unmarked paragraph is never promoted
B13  an over-budget stretch of line records is cut at line ends, not inside a line
B14  pieces joined back together do not repeat their overlap; a heading-only piece joins its body without
     re-splitting it; a multi-line heading is recorded line by line
"""
from __future__ import annotations

import re
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from kb_pipeline.chunking import chunker
from kb_pipeline.chunking.chunker import blocks_to_chunks, chunk_text, tiny_threshold
from kb_pipeline.chunking.packing import pack_prose
from kb_pipeline.headings import HeadingResolver
from kb_pipeline.models import ParsedBlock
from kb_pipeline.parsers.common import SectionTracker
from kb_pipeline.utils import count_tokens



def block(text, bid="b1", kind="text", **metadata):
    return ParsedBlock("fixture", "v1", "pdf", kind, text, bid, metadata=metadata)


def chunks_of(blocks, max_tokens, overlap):
    return blocks_to_chunks(kb_id="fixture", file_key=1, content_version="v1", parser_profile="v1",
                            blocks=blocks, max_tokens=max_tokens, overlap_tokens=overlap)


def walk(seq, **kw):
    """[(text, parser level)] -> [(final level, section path)], the way the parsers walk a document."""
    resolver = HeadingResolver((t for t, lv in seq if lv), **kw)
    sections = SectionTracker()
    out = []
    for text, parser_level in seq:
        level, _ = resolver.resolve(parser_level, text)
        sections.observe(level, text)
        out.append((level, list(sections.path)))
    return out


# ── B2 ────────────────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("trust_parser_levels", [False, True])       # PDF, DOCX
def test_numbered_steps_do_not_become_headings_or_wipe_the_section_path(trust_parser_levels):
    got = walk([("第3章 安装", 1), ("3.2 操作步骤", 1), ("1. 打开设备电源", None), ("按下电源键。", None),
                ("2. 登录系统", None)], trust_parser_levels=trust_parser_levels)
    assert [level for level, _ in got] == [1, 2, None, None, None]
    assert [path for _, path in got][2:] == [["第3章 安装", "3.2 操作步骤"]] * 3


@pytest.mark.parametrize("text", ["1. 打开设备电源", "2.4 GHz 频段", "5.0 版本更新说明", "3-5 年工作经验",
                                  "第六章 落地执行", "Chapter 4: Timing", "1.2.3 电气特性", "IV. Results"])
def test_an_unmarked_paragraph_is_never_promoted(text):
    assert HeadingResolver().resolve(None, text) == (None, False)


def test_marked_headings_still_take_their_level_from_the_numbering():
    resolver = HeadingResolver()
    assert resolver.resolve(1, "5.4.3 函数的凹凸性") == (3, False)       # deeper than the parser's flat level 1
    assert resolver.resolve(1, "1. 打开设备电源") == (1, False)         # the parser says heading: it stays one
    assert resolver.resolve(2, "试验条件")[0] is not None
    # a marked heading that recurs three times is placed by its numbering when it has one (unchanged), and
    # below the nearest heading that does not recur when it has none (it used to become body text)
    labels = HeadingResolver(["1. 概述"] * 3 + ["KEYWORD"] * 3)
    assert labels.resolve(1, "1. 概述") == (1, True)
    assert labels.resolve(1, "KEYWORD") == (2, False)


def test_pdf_blocks_keep_numbered_steps_as_body_text():
    from kb_pipeline.parsers import mineru_pdf

    def item(text, level=None):
        return {"type": "text", "text": text, "page_idx": 0, **({"text_level": level} if level else {})}

    payload = {"content_list": [item("第3章 安装", 1), item("3.2 操作步骤", 1), item("1. 打开设备电源"),
                                item("按下电源键，等待指示灯亮起。"), item("2. 登录系统"), item("输入账号和密码。")]}
    with tempfile.TemporaryDirectory() as tmp, patch.object(mineru_pdf, "call_mineru_sync", return_value=payload):
        blocks = mineru_pdf.mineru_pdf_blocks(mineru_url="http://x", path=Path(tmp) / "a.pdf", cache_dir=Path(tmp) / "c")
    assert [b.block_type for b in blocks] == ["title", "title", "text", "text", "text", "text"]
    assert [b.text for b in blocks][2] == "1. 打开设备电源"                # still there, as body text
    assert all(b.metadata["section_path"] == ["第3章 安装", "3.2 操作步骤"] for b in blocks[1:])


# ── B13 ───────────────────────────────────────────────────────────────────────────────────────────────

LOG = "\n".join(f"2026-03-{d:02d} 12:00:{d:02d} INFO worker-{d % 4} request id={1000 + d} status=200 path=/api/v1/items/{d}"
                for d in range(1, 31))
STOCK = "\n".join(f"{n}号库位 存放 型号A{n:03d} 数量{n * 5}件 负责人张三" for n in range(1, 41))


@pytest.mark.parametrize("text", [LOG, STOCK], ids=["log", "cjk-list"])
def test_over_budget_line_records_are_cut_at_line_ends(text):
    budget = 120
    assert count_tokens(text) > 3 * budget
    parts = chunk_text(text, budget, 0)
    assert len(parts) > 3
    assert "\n".join(parts).split("\n") == text.split("\n")              # whole lines, in order, none lost or repeated
    assert all(count_tokens(part) <= budget for part in parts)


def test_only_a_line_that_alone_exceeds_the_budget_is_split_inside():
    budget = 60
    long_line = "payload=" + "0123456789abcdef" * 60
    lines = LOG.split("\n")[:6] + [long_line] + LOG.split("\n")[6:12]
    parts = chunk_text("\n".join(lines), budget, 0)
    whole = set(lines) - {long_line}
    inside = [part for part in parts if not set(part.split("\n")) <= whole]
    assert len(inside) > 1 and "".join(inside) == long_line              # only the long line was cut, nothing lost
    assert [line for part in parts if part not in inside for line in part.split("\n")] == [l for l in lines if l != long_line]
    assert all(count_tokens(part) <= budget for part in parts)


def test_text_without_line_breaks_and_text_within_budget_are_split_as_before():
    long_line = "0123456789abcdef" * 80
    assert chunk_text(long_line, 50, 0) == chunker._split_long_sentence(long_line, 50)
    short = "\n".join(LOG.split("\n")[:3])
    assert chunk_text(short, 400, 80) == [short]
    prose = "第一句讲安装。第二句讲配置。第三句讲升级。"
    assert chunk_text(prose, 400, 80) == ["第一句讲安装。\n第二句讲配置。\n第三句讲升级。"]


def test_a_log_file_block_becomes_chunks_of_whole_lines():
    out = chunks_of([block(LOG)], 120, 20)
    assert len(out) > 3
    assert [line for c in out for line in c.text.split("\n")] == LOG.split("\n")


# ── B14 ───────────────────────────────────────────────────────────────────────────────────────────────

def sentence(i):
    return f"句{i:03d}号说明了设备在这种工况下的运行参数。"


def numbers(text):
    return re.findall(r"句(\d+)号", text)


def equal_cost_setup(first_chunk_sentences):
    """Sentences of equal token cost and a budget that holds exactly `first_chunk_sentences` of them, with an
    overlap of exactly one sentence. Works with either tokenizer."""
    cost = count_tokens(sentence(1)) + 1
    assert all(count_tokens(sentence(i)) + 1 == cost for i in range(1, 60))
    return cost * first_chunk_sentences, cost


def test_overlaps_reported_by_chunk_text_are_the_repeated_start_of_each_chunk():
    budget, overlap = equal_cost_setup(8)
    leads: list[str] = []
    parts = chunk_text("".join(sentence(i) for i in range(1, 21)), budget, overlap, overlaps=leads)
    assert len(parts) == len(leads) == 3 and leads[0] == ""
    for before, part, lead in zip(parts, parts[1:], leads[1:]):
        assert lead and part.startswith(lead + "\n") and before.endswith("\n" + lead)
    assert chunk_text("".join(sentence(i) for i in range(1, 21)), budget, overlap) == parts      # the list is optional


def test_a_fragment_glued_back_does_not_repeat_the_overlap_sentence():
    budget, overlap = equal_cost_setup(20)
    text = "".join(sentence(i) for i in range(1, 22))                  # 20 fill the first chunk, one is left over
    parts = chunk_text(text, budget, overlap)
    assert [numbers(p)[0] for p in parts] == ["001", "020"]             # the second part: overlap + the last sentence
    assert count_tokens(parts[1]) < tiny_threshold(budget)             # a fragment, so it is glued back
    out = chunks_of([block(text)], budget, overlap)
    assert len(out) == 1
    assert numbers(out[0].text) == [f"{i:03d}" for i in range(1, 22)]   # every sentence once, in order


def test_overlap_between_separate_chunks_is_kept():
    budget, overlap = equal_cost_setup(8)
    out = chunks_of([block("".join(sentence(i) for i in range(1, 14)))], budget, overlap)
    assert [numbers(c.text) for c in out] == [[f"{i:03d}" for i in range(1, 9)], [f"{i:03d}" for i in range(8, 14)]]


def test_glue_and_pack_drop_only_an_overlap_that_really_repeats_the_neighbour():
    body = block("x")

    def piece(text, **extra):
        return {"text": text, "block": body, "tokens": count_tokens(text), **extra}

    glued = chunker._glue_pieces([piece("甲句。\n乙句。\n丙句。" * 1), piece("丙句。\n丁句。", overlap="丙句。")], 4000)
    assert [p["text"] for p in glued] == ["甲句。\n乙句。\n丙句。\n丁句。"]
    # the remainder of a split chunk is shorter than the overlap of the part after it: the whole overlap goes
    packed = pack_prose([piece("小标题\n丙句。"), piece("乙句。\n丙句。\n丁句。", overlap="乙句。\n丙句。")], 4000)
    assert [p["text"] for p in packed] == ["小标题\n丙句。\n丁句。"]
    # no overlap recorded, or the neighbour does not end with it: the text is joined unchanged
    plain = pack_prose([piece("甲句。\n丙句。"), piece("丙句。\n丁句。")], 4000)
    assert [p["text"] for p in plain] == ["甲句。\n丙句。\n丙句。\n丁句。"]
    apart = pack_prose([piece("甲句。\n乙句。"), piece("丙句。\n丁句。", overlap="丙句。")], 4000)
    assert [p["text"] for p in apart] == ["甲句。\n乙句。\n丙句。\n丁句。"]


def test_a_chunk_split_at_a_sub_heading_is_packed_with_the_next_part_without_repeats():
    budget, overlap = equal_cost_setup(14)
    sub = "3.1.1 运行参数"
    body = "".join(sentence(i) for i in range(1, 7)) + "\n" + sub + "\n" + "".join(sentence(i) for i in range(7, 17))
    parts = chunk_text(body, budget, overlap, headings=frozenset([sub]))
    assert len(parts) == 2 and sub in parts[0] and numbers(parts[1])[0] == numbers(parts[0])[-1]   # one sentence of overlap
    section = {"section_path": ["3.1 设备概述"]}
    out = chunks_of([block("3.1 设备概述", "h1", "title", **section), block(body, "b1", heading_lines=[sub], **section)],
                    budget, overlap)
    # the title takes the opening of the first part; what is left of that part starts at the sub-heading and
    # is packed with the second part, whose first sentence repeats its last one
    assert [c.text.split("\n")[0] for c in out] == ["3.1 设备概述", sub]
    assert [numbers(c.text) for c in out] == [[f"{i:03d}" for i in range(1, 7)], [f"{i:03d}" for i in range(7, 17)]]


@pytest.mark.parametrize("long_heading", [False, True])
def test_a_heading_joins_a_full_body_chunk_without_splitting_it_again(long_heading):
    budget, overlap = equal_cost_setup(10)
    text = "".join(sentence(i) for i in range(1, 25))
    section = {"section_path": ["设备概述"]}
    alone = chunks_of([block(text, **section)], budget, overlap)
    heading = "设备概述"
    while long_heading and count_tokens(heading + "\n" + alone[0].text) <= budget:
        heading += "与运行参数"                                          # until heading + body is over the plain budget
    out = chunks_of([block(heading, "h1", "title", **section), block(text, **section)], budget, overlap)
    assert len(out) == len(alone) == 3                                   # no extra piece made of overlap + the last sentence
    assert out[0].text == heading + "\n" + alone[0].text
    assert out[0].token_count <= int(budget * chunker.GLUE_SLACK)        # the limit of every glued fragment
    assert (out[0].token_count > budget) == long_heading
    assert [c.text for c in out[1:]] == [c.text for c in alone[1:]]


def test_a_heading_and_a_short_body_are_one_chunk_as_before():
    section = {"section_path": ["设备概述"]}
    out = chunks_of([block("设备概述", "h1", "title", **section), block("第一句讲安装。第二句讲配置。", **section)], 400, 80)
    assert [c.text for c in out] == ["设备概述\n第一句讲安装。\n第二句讲配置。"]


def test_a_two_line_heading_is_recorded_line_by_line():
    section = {"section_path": ["Guide"]}
    out = chunks_of([block("Installation guide\nPreparation steps", "h1", "title", **section),
                     block("Unpack the device. Place it on a flat surface.", **section)], 400, 80)
    assert len(out) == 1
    assert out[0].block.metadata["heading_lines"] == ["Installation guide", "Preparation steps"]
