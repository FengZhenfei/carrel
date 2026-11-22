"""A chapter heading the parser did not mark is recovered: "第 3 章 安装", "CHAPTER 02", "附录 A" on a line of their
own. Recognisers miss a chapter heading now and then, and a Word file whose chapters were typed without heading
styles has no marked headings at all.

The wording alone decides nothing. The line must also be a chapter heading of this document: not an entry of a list
(contents, enclosures), not a chapter that a marked heading names already, and one of a sequence -- the chapters of
its kind, marked and recovered together, are at least two and count on through the document without a gap. When they
do not (an attachment that starts again at 第一章, a running head, a preface that describes later chapters, a chapter
whose heading cannot be read), none of that kind is recovered. Whatever fails stays body text, as every unmarked line did before; numbered lines are never
recovered."""
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from kb_pipeline import headings
from kb_pipeline.headings import HeadingResolver
from kb_pipeline.parsers.common import SectionTracker

TABLE = (None, None)        # an entry that is content but no line of text: a table, a picture


def walk(seq, contents=(), **kw):
    """[(text, parser level)] -> [(final level, recovered by wording, section path)], the way the parsers walk a
    document."""
    resolver = HeadingResolver((t for t, lv in seq if lv), **kw)
    recovered = headings.recoverable_chapter_lines(((i, t, bool(lv)) for i, (t, lv) in enumerate(seq)), contents=contents)
    sections = SectionTracker()
    out = []
    for i, (text, parser_level) in enumerate(seq):
        if text is None:
            out.append((None, False, list(sections.path)))
            continue
        level, inferred = resolver.recover(recovered[i]) if i in recovered else resolver.resolve(parser_level, text)
        sections.observe(level, text)
        out.append((level, inferred, list(sections.path)))
    return out


def levels(seq, **kw):
    return [level for level, *_ in walk(seq, **kw)]


# ---------------------------------------------------------------- how a chapter title must read

@pytest.mark.parametrize("text, level", [
    ("第六章 落地执行", 1), ("第 3 章 安装", 1), ("第十二章　总则", 1), ("第三章", 1), ("第二部分：方法", 1), ("第一篇 概论", 1),
    ("第二节 范围", 2), ("Chapter 4: Timing", 1), ("CHAPTER 02", 1), ("Part II Methods", 1), ("Section 2 Scope", 2),
    ("附录 A 术语表", 1), ("附录一", 1), ("Appendix B", 1), ("Appendix III Forms", 1),
    ("第 123 章 重逢", 1), ("第一百二十三章 重逢", 1),       # a long book: the number of a 第 N 章 line may have three digits
    ("**第一章 总则**", 1)])                               # emphasis marks around the line do not count
def test_a_line_that_reads_like_a_chapter_title(text, level):
    assert headings.chapter_word_level(text) == level


@pytest.mark.parametrize("text", [
    "第3章的说明请参见附录",                        # a sentence about a chapter, nothing separates the chapter word
    "第三章总则",                                 # written without a separator: left to the parser's own mark
    "第 3 章 介绍了安装方法，见下文",                # reads on as a sentence
    "Chapter 3 describes the installation.", "Section 3 applies only to members", "Chapter 11 bankruptcy protection",
    "Part I think is missing",                   # after an English chapter word a title starts with a capital or a digit
    "第一章 总则 3", "第一章 总则........3", "第一章 总则····3",       # contents entries
    "SECTION 906 OF THE SARBANES-OXLEY ACT OF 2002", "Section 8 of the Housing Act", "Chapter 11 of the Bankruptcy Code",
    "Part 2 of 3",                               # "... N of ...": the line cites a part of something else
    "Section 906 Certification", "Part 4410 Gasket",      # an English chapter word with three digits: a law, a part number
    "Section 906 Certification", "Section ９０６ Certification",          # ... also with a no-break space or full-width digits
    "第 35 卷 第 4 期", "第一卷 总论",              # a volume names the whole publication (head of a journal paper)
    "Section 2.1 Scope", "附录A.1 术语", "第一章 第一节 概述",                 # compound numbers: not one chapter
    "Appendix to the Annual Report", "Appendix is attached separately", "Appendixes", "Section x", "chapter 4 timing",
    "第一节 08:00-09:30 开幕致辞",                  # a time of day: an entry of a timetable
    "第一条 为了规范管理", "第一章 总则\n正文", "第一章 " + "很长的标题" * 13,
    "1. 打开设备电源", "2.4 GHz 频段", "1.2.3 电气特性", "3-5 年工作经验", "IV. Results", "普通正文一句话。", ""])
def test_a_line_that_does_not(text):
    assert headings.chapter_word_level(text) is None


@pytest.mark.parametrize("text", ["第六章 落地执行", "Chapter 4: Timing", "1. 打开设备电源", "2.4 GHz 频段"])
def test_the_resolver_alone_never_promotes_an_unmarked_line(text):
    assert HeadingResolver().resolve(None, text) == (None, False)


@pytest.mark.parametrize("a, b", [("第 3 章 安装", "第三章 安装"), ("第03章", "第3章"), ("CHAPTER 04", "Chapter IV"), ("第十二节", "第 12 节"),
                                  ("第二節 範圍", "第二节 范围"), ("附录 A", "附录A 术语")])
def test_one_chapter_written_in_two_ways_is_one_chapter(a, b):
    assert headings._named_chapter(a) == headings._named_chapter(b) is not None


def test_a_line_about_the_appendices_names_no_appendix():
    assert headings._named_chapter("附录一览表") is None and headings._named_chapter("附录一 术语") == ("附录", "1")


# ---------------------------------------------------------------- which of those lines are chapter headings

@pytest.mark.parametrize("trust_parser_levels", [False, True])       # PDF, DOCX
def test_chapters_typed_without_heading_styles_are_recovered_and_numbered_steps_are_not(trust_parser_levels):
    got = walk([("第一章 总则", None), ("本制度适用于全体员工。", None), ("第二章 流程", None), ("1. 提交申请", None),
                ("2. 部门审批", None), ("第一节 例外", None), ("特殊情况另行处理。", None), ("由部门负责人批准。", None),
                ("第二节 范围", None), ("见附件。", None),
                ("**第三章 附则**", None), ("本制度自发布之日起施行。", None)], trust_parser_levels=trust_parser_levels)
    assert [(level, inferred) for level, inferred, _ in got] == [
        (1, True), (None, False), (1, True), (None, False), (None, False), (2, True), (None, False), (None, False),
        (2, True), (None, False), (1, True), (None, False)]
    assert [path for *_, path in got] == ([["第一章 总则"]] * 2 + [["第二章 流程"]] * 3 + [["第二章 流程", "第一节 例外"]] * 3
                                          + [["第二章 流程", "第二节 范围"]] * 2 + [["第三章 附则"]] * 2)


def test_one_chapter_heading_the_recogniser_missed_is_recovered():
    got = walk([("第一章 总则", 1), ("正文。", None), ("第二章 流程", None), ("正文。", None), ("第三章 附则", 1), ("正文。", None)])
    assert [level for level, *_ in got] == [1, None, 1, None, 1, None]
    assert got[3][2] == ["第二章 流程"]


def test_headings_after_a_recovered_chapter_are_placed_as_after_a_marked_one():
    seq = [("报告", 1), ("CHAPTER 01", 1), ("开场", 1), ("正文。", None), ("CHAPTER 02", None), ("演化路径", 1), ("正文。", None),
           ("CHAPTER 03", 1), ("收尾", 1), ("正文。", None)]
    marked = [(text, 1 if text.startswith("CHAPTER") else level) for text, level in seq]
    got = walk(seq)
    assert [level for level, *_ in got] == [1, 1, 2, None, 1, 2, None, 1, 2, None]     # PDF: an unnumbered heading hangs below the chapter
    assert got[4][1] and got[6][2] == ["CHAPTER 02", "演化路径"]
    assert [(level, path) for level, _, path in got] == [(level, path) for level, _, path in walk(marked)]


def unrecovered(seq, **kw):
    """The same walk with nothing recovered: what every unmarked line was before."""
    with patch.object(headings, "recoverable_chapter_lines", return_value={}):
        return walk(seq, **kw)


@pytest.mark.parametrize("seq", [
    # a tender: every heading is marked, and one clause lists the parts of the bid with a sentence under each
    [("第一章 招标公告", 1), ("公告正文。", None), ("第二章 投标人须知", 1), ("2.1 投标文件的组成", 1), ("投标文件由以下部分组成：", None),
     ("第一部分 商务部分", None), ("包括营业执照和资质证书。", None), ("第二部分 技术部分", None), ("包括技术方案和人员安排。", None),
     ("2.2 投标报价", 1), ("报价应当包含全部费用。", None), ("第三章 评标办法", 1), ("评标正文。", None)],
    # a manual: its introduction lists the parts of the device
    [("1 Introduction", 1), ("This manual covers the device.", None), ("Part 1 Base unit", None), ("The base holds the power supply.", None),
     ("Part 2 Sensor head", None), ("The head holds the sensor.", None), ("Safety notes", 1), ("Read these first.", None), ("2 Installation", 1)],
    # a report: the summary describes its parts, the body has headings of its own
    [("内容提要", 1), ("第一部分 现状分析", None), ("梳理行业现状。", None), ("第二部分 问题诊断", None), ("归纳主要问题。", None),
     ("一、行业现状", 1), ("现状正文。", None), ("二、主要问题", 1), ("问题正文。", None)],
    # two sessions listed inside a chapter
    [("第一章 课程安排", 1), ("第一节 理论", None), ("上午进行。", None), ("第二节 实操", None), ("下午进行。", None), ("考核方式", 1), ("闭卷。", None)],
    # chapter pages the parser did not mark, while it marked the headings inside the chapters
    [("报告", 1), ("CHAPTER 01", None), ("开场", 1), ("正文。", None), ("CHAPTER 02", None), ("演化路径", 1), ("正文。", None)],
    # a marked heading between the lines, none after the last of them
    [("第一章 总则", None), ("正文。", None), ("适用范围", 1), ("正文。", None), ("第二章 流程", None), ("正文。", None)],
    # the parser marked two parts; a list inside the second counts on from them
    [("第一部分 招标公告", 1), ("公告正文。", None), ("第二部分 投标人须知", 1), ("投标文件由四部分组成：", None),
     ("第一部分 资格文件", None), ("包括营业执照。", None), ("第二部分 商务文件", None), ("包括报价表。", None),
     ("第三部分 技术文件", None), ("包括技术方案。", None), ("方案应当逐项响应。", None), ("第四部分 其他材料", None), ("包括承诺函。", None),
     ("一、投标报价", 1), ("报价应当包含全部费用。", None)],
    # three marked chapters, then a preview of the chapters of the next volume
    [("第一章 概述", 1), ("正文。", None), ("第二章 方法", 1), ("正文。", None), ("第三章 结果", 1), ("正文。", None), ("下册预告", None),
     ("第四章 应用", None), ("介绍应用案例。", None), ("案例来自三个行业。", None), ("第五章 索引", None), ("列出全部术语。", None),
     ("参考文献", 1), ("文献正文。", None)],
    # a marked appendix whose text lists the other appendices
    [("1 概述", 1), ("正文。", None), ("附录 A 合同条款", 1), ("本合同另有以下附录：", None), ("附录 B 报价明细", None), ("列明各项报价。", None),
     ("报价含税。", None), ("附录 C 服务水平", None), ("列明响应时间。", None), ("签署页", 1), ("双方签字。", None)],
])
@pytest.mark.parametrize("trust_parser_levels", [False, True])
def test_lines_between_headings_the_parser_marked_are_left_alone(seq, trust_parser_levels):
    """Where the parser marked the headings that follow, an unmarked run of chapter-word lines is a list inside one of
    those sections more often than a chapter the parser missed. As level-1 headings such lines took the sections after
    them away from their chapter ("2.2 投标报价" went below "第二部分 技术部分")."""
    assert walk(seq, trust_parser_levels=trust_parser_levels) == unrecovered(seq, trust_parser_levels=trust_parser_levels)


@pytest.mark.parametrize("seq", [
    # a report typed without styles: the overview names its parts with a sentence under each; the body has headings of its own
    [("内容提要", None), ("第一部分 现状分析", None), ("梳理行业现状。", None), ("第二部分 问题诊断", None), ("归纳主要问题。", None),
     ("第三部分 对策建议", None), ("提出对策。", None), ("一、行业现状", None), ("现状正文。", None), ("现状续。", None), ("二、主要问题", None), ("问题正文。", None)],
    # a notice: two parts with a sentence each, then the closing lines
    [("关于开展培训的通知", 1), ("各部门：", None), ("第一部分 理论培训", None), ("周一上午在会议室进行。", None), ("第二部分 实操培训", None),
     ("周二下午在车间进行。", None), ("请各部门按时参加。", None), ("人力资源部", None), ("2026年3月1日", None)],
    # the last marked section lists two appendices with a sentence each
    [("第一章 总则", 1), ("正文。", None), ("第二章 附则", 1), ("本办法附有：", None), ("附录 A 申请表", None), ("申请时填写。", None),
     ("附录 B 审批表", None), ("审批时填写。", None), ("本办法自发布之日起施行。", None), ("办公室", None)],
    # annotated contents whose first chapter has no heading of its own in the body: the line fills a place among marked headings
    [("目录", 1), ("第1章 入门", None), ("安装和配置。", None), ("第2章 查询", None), ("检索数据。", None), ("第3章 更新", None), ("修改数据。", None),
     ("序", 1), ("序的正文。", None), ("1-1 安装", 1), ("正文一。", None), ("第2章 查询", 1), ("正文二。", None), ("第3章 更新", 1), ("正文三。", None)],
    # two chapters only, the first with a single paragraph: nothing tells them from a list of two entries
    [("第一章 总则", None), ("为规范管理制定本办法。", None), ("第二章 附则", None), ("本办法自发布之日起施行。", None), ("解释权归办公室。", None)],
])
@pytest.mark.parametrize("trust_parser_levels", [False, True])
def test_a_list_with_a_note_under_each_entry_is_left_alone(seq, trust_parser_levels):
    """Each line is followed by one paragraph and then by the next line with the same chapter word. Overviews, the
    parts of a tender and annotated contents are written this way; as chapters, the last entry took everything after
    it (the whole body of the report was filed under "第三部分 对策建议")."""
    assert walk(seq, trust_parser_levels=trust_parser_levels) == unrecovered(seq, trust_parser_levels=trust_parser_levels)


def test_one_chapter_with_more_than_a_paragraph_makes_the_lines_chapters():
    # short chapters of a regulation: only one of them holds two paragraphs
    seq = [("第一章 目的", None), ("为规范管理制定本办法。", None), ("第二章 范围", None), ("适用于全体员工。", None), ("外包人员参照执行。", None),
           ("第三章 附则", None), ("本办法自发布之日起施行。", None)]
    assert levels(seq) == [1, None, 1, None, None, 1, None]
    # and a chapter the parser missed between two marked ones may hold a single paragraph
    assert levels([("第一章 目的", 1), ("正文。", None), ("第二章 范围", None), ("适用于全体员工。", None), ("第三章 附则", 1), ("正文。", None)]) == [1, None, 1, None, 1, None]


@pytest.mark.parametrize("seq", [
    # a training plan typed without styles: the parts are a list inside item 二, and item 三 follows them
    [("培训方案", None), ("一、培训目标", None), ("提高新员工的岗位能力。", None), ("二、培训内容", None), ("第一部分 企业文化", None),
     ("介绍公司的历史和制度。", None), ("由人力资源部讲授。", None), ("第二部分 安全生产", None), ("讲解安全规程。", None), ("第三部分 岗位技能", None), ("分岗位实操。", None),
     ("三、培训时间", None), ("共五个工作日。", None), ("四、培训要求", None), ("不得无故缺席。", None)],
    # a syllabus: the sessions are a list inside item 三
    [("课程大纲", None), ("三、教学内容", None), ("第一节 绪论", None), ("课程的研究对象。", None), ("两个学时。", None), ("第二节 连杆机构", None), ("平面四杆机构。", None),
     ("第三节 齿轮传动", None), ("渐开线齿轮。", None), ("四、考核方式", None), ("闭卷考试。", None), ("五、教材", None), ("机械原理。", None)],
    # the same with dotted numbers
    [("2.1 投标文件的组成", None), ("投标文件由以下部分组成：", None), ("第一部分 商务部分", None), ("包括营业执照。", None), ("复印件加盖公章。", None),
     ("第二部分 技术部分", None), ("包括技术方案。", None), ("2.2 投标报价", None), ("报价应当包含全部费用。", None)],
])
@pytest.mark.parametrize("trust_parser_levels", [False, True])
def test_lines_held_between_two_neighbours_of_a_numbered_list_are_left_alone(seq, trust_parser_levels):
    """ "二、培训内容" stands before the lines and "三、培训时间" after them: the lines divide item 二 of that list. As
    chapters they took items 三 and 四 with them (their text was filed under "第三部分 岗位技能")."""
    assert walk(seq, trust_parser_levels=trust_parser_levels) == unrecovered(seq, trust_parser_levels=trust_parser_levels)


def test_a_numbered_list_that_does_not_run_on_across_the_chapters_does_not_stop_the_recovery():
    # steps numbered anew in every chapter, and a list before the first chapter that ends where no list goes on
    seq = [("1. 目的", None), ("2. 范围", None), ("第一章 总则", None), ("1. 提交申请", None), ("2. 部门审批", None), ("正文。", None),
           ("第二章 流程", None), ("1. 填写表格", None), ("正文。", None)]
    assert levels(seq) == [None, None, 1, None, None, None, 1, None, None]


@pytest.mark.parametrize("seq", [
    # one chapter has sections, and an attachment starts again at chapter one: the chapters are not recovered
    [("第一章 总则", None), ("正文。", None), ("正文续。", None), ("第二章 流程", None), ("第一节 申请", None), ("正文。", None), ("正文续。", None),
     ("第二节 审批", None), ("正文。", None), ("第三章 监督", None), ("正文。", None), ("第四章 附则", None), ("正文。", None), ("附件", None),
     ("第一章 范围", None), ("正文。", None), ("第二章 要求", None), ("正文。", None)],
    # one chapter has sections, and the last chapter cannot be read as a title
    [("第一章 背景", None), ("正文。", None), ("正文续。", None), ("第二章 流程", None), ("第一节 申请", None), ("正文。", None), ("正文续。", None),
     ("第二节 审批", None), ("正文。", None), ("第三章 规划 2030", None), ("正文。", None)],
    # no chapter line reads like a title at all
    [("第一章 背景 3", None), ("正文。", None), ("第一节 申请", None), ("正文。", None), ("正文续。", None), ("第二节 审批", None), ("正文。", None),
     ("第二章 规划 9", None), ("正文。", None)],
])
def test_sections_are_not_recovered_below_chapters_that_are_not(seq):
    """Recovered alone, the sections took the chapters after them: "第三章 监督" and the whole attachment were filed
    under "第二节 审批"."""
    assert levels(seq) == [None] * len(seq)


def test_chapters_taken_for_list_entries_at_the_end_stop_the_recovery():
    """Chapter 4 holds nothing but numbered lines, which are no content for the list rule: chapters 4 and 5 look like
    two entries of a list and are not recovered. With chapters 1 to 3 recovered, all of chapters 4 and 5 would be
    filed under chapter 3."""
    body = [x for n, title in ((1, "总则"), (2, "流程"), (3, "审批")) for x in ((f"第{n}章 {title}", None), ("正文。", None), ("正文续。", None))]
    assert levels(body) == [1, None, None] * 3
    seq = body + [("第4章 所需材料", None), ("1. 报销单", None), ("2. 发票原件", None), ("3. 审批记录", None), ("第5章 附则", None),
                  ("本办法自发布之日起施行。", None)]
    assert levels(seq) == [None] * len(seq)


def test_sections_are_not_recovered_when_the_chapters_are_written_without_a_separator():
    seq = [("第一章总则", None), ("正文。", None), ("第二章流程", None), ("第一节 申请", None), ("正文。", None), ("正文续。", None),
           ("第二节 审批", None), ("正文。", None), ("第三章维护", None), ("正文。", None)]
    assert levels(seq) == [None] * len(seq)


def test_what_counts_as_a_numbered_list_running_on_across_the_lines():
    parts = [("第一部分 企业文化", None), ("介绍公司的历史。", None), ("由人力资源部讲授。", None), ("第二部分 安全生产", None), ("讲解安全规程。", None)]
    held = [None] * 5
    free = [1, None, None, 1, None]
    assert levels([("二、培训内容", None)] + parts + [("三、培训时间", None), ("共五个工作日。", None)]) == [None] + held + [None, None]
    # the number after the lines must be the very next one
    assert levels([("二、培训内容", None)] + parts + [("四、培训要求", None), ("不得缺席。", None)]) == [None] + free + [None, None]
    # a numbered sentence is no item of such a list
    assert levels([("二、培训内容包括以下两个部分，分别说明如下。", None)] + parts + [("三、培训时间", None), ("共五个工作日。", None)]) == [None] + free + [None, None]
    # the item before must stand before the first of the lines
    assert levels(parts[:3] + [("二、其它安排", None), ("另行通知。", None)] + parts[3:] + [("三、培训时间", None), ("共五个工作日。", None)]
                  ) == free[:3] + [None, None] + free[3:] + [None, None]
    # dotted numbers are one list per parent: 2.1 and 3.2 are not neighbours, 2.1 and 2.2 are
    assert levels([("2.1 培训内容", None)] + parts + [("3.2 培训时间", None), ("共五个工作日。", None)]) == [None] + free + [None, None]
    assert levels([("2.1 培训内容", None)] + parts + [("2.2 培训时间", None), ("共五个工作日。", None)]) == [None] + held + [None, None]


def test_a_last_chapter_broken_over_two_lines_stops_the_recovery():
    # the heading of the last chapter is one item of two lines: it reads like no title, and its text would go to chapter two
    two = [("第一章 背景", None), ("背景正文。", None), ("背景续。", None), ("第二章 现状", None), ("现状正文。", None), ("现状续。", None)]
    assert levels(two) == [1, None, None, 1, None, None]
    assert levels(two + [("第三章 展望\n——面向2030", None), ("展望正文。", None)]) == [None] * 8
    # so does a last heading written without a separator, however long
    assert levels(two + [("第三章关于未来五年发展规划与重点任务的安排", None), ("规划正文。", None)]) == [None] * 8
    # a paragraph that opens with the chapter after next, or one before the last chapter, does not
    assert levels(two + [("第四章 的内容较多，\n将在下册介绍。", None), ("正文。", None)]) == [1, None, None, 1, None, None, None, None]
    assert levels(two[:1] + [("第三章 的内容较多，\n将在后文介绍。", None)] + two[1:]) == [1, None, None, None, 1, None, None]


def test_what_the_rules_measure_from():
    recoverable = headings.recoverable_chapter_lines
    plain = [(0, "第一章 总则", False), (1, "正文。", False), (2, "正文续。", False), (3, "第二章 流程", False), (4, "正文。", False)]
    assert recoverable(plain) == {0: 1, 3: 1}
    # a marked line on a contents page is no heading that follows
    assert recoverable(plain + [(5, "索引", True), (6, "第一章 总则 1", True)], contents=[5, 6]) == {0: 1, 3: 1}
    assert recoverable(plain + [(5, "索引", True)]) == {}
    # the chapter that would come next, for lettered chapters too; a line on a contents page is not such a line
    lettered = [(0, "附录 A 术语", False), (1, "术语的说明。", False), (2, "另一段说明。", False), (3, "附录 B 表格", False),
                (4, "表格的说明。", False), (5, "另一段说明。", False)]
    assert recoverable(lettered) == {0: 1, 3: 1}
    assert recoverable(lettered + [(6, "附录 C 样例 12", False), (7, "样例的说明。", False)]) == {}
    # ... also when that line stands on a page taken for contents (a glossary page of short lines is taken for one)
    assert recoverable(lettered + [(6, "附录 C 样例", False), (7, "样例的说明。", False)], contents=[6, 7]) == {}
    # ... but not when it is an entry of a list of these very chapters: an index at the end
    index = [(6, "附录 A 术语 1", False), (7, "附录 B 表格 5", False), (8, "附录 C 样例 9", False)]
    assert recoverable(lettered + index) == {0: 1, 3: 1}
    assert recoverable(lettered + index, contents=[6, 7, 8]) == {0: 1, 3: 1}
    # the rule is about the text after a recovered chapter: after a marked one that text was filed there before
    marked_last = [(0, "第一章 总则", True), (1, "正文。", False), (2, "第二章 流程", False), (3, "正文。", False), (4, "第三章 附则", True),
                   (5, "正文。", False), (6, "第四章 展望 2030", False), (7, "正文。", False)]
    assert recoverable(marked_last) == {2: 1}


def test_chapters_after_the_last_marked_heading_are_recovered():
    # only the title of the document carries a style: from the first chapter on these lines are all the structure there is
    got = walk([("设备管理制度", 1), ("第一章 总则", None), ("正文一。", None), ("正文一续。", None), ("第二章 流程", None), ("正文二。", None)],
               trust_parser_levels=True)
    assert [(level, path) for level, _, path in got] == [(1, ["设备管理制度"]), (1, ["第一章 总则"]), (None, ["第一章 总则"]),
                                                         (None, ["第一章 总则"]), (1, ["第二章 流程"]), (None, ["第二章 流程"])]
    # and appendices typed plain at the very end of a document displace nothing that follows
    got = walk([("1 概述", 1), ("正文。", None), ("附录 A 术语", None), ("术语一。", None), ("术语二。", None), ("附录 B 表格", None), ("表格一。", None)])
    assert [path for *_, path in got][2:] == [["附录 A 术语"]] * 3 + [["附录 B 表格"]] * 2


def test_a_last_chapter_that_cannot_be_read_as_a_title_stops_the_recovery():
    """ "第三章 展望 2030" ends like a contents entry and is not recovered. Recovering the two chapters before it would
    file its text under the second chapter."""
    two = [("第一章 背景", None), ("背景正文。", None), ("背景续。", None), ("第二章 现状", None), ("现状正文。", None), ("现状续。", None)]
    recovered = [1, None, None, 1, None, None]
    assert levels(two) == recovered
    assert levels(two + [("第三章 展望 2030", None), ("展望正文。", None)]) == [None] * 8
    # a sentence about a later chapter does the same: nothing tells it from a heading that cannot be read
    assert levels(two + [("第三章 的内容见下册。", None), ("正文。", None)]) == [None] * 8
    # a line about an earlier chapter, or about one further on, does not
    assert levels(two + [("第一章 的结论，此处不再重复。", None), ("正文。", None)]) == recovered + [None, None]
    assert levels(two + [("第五章 的内容见下册。", None), ("正文。", None)]) == recovered + [None, None]
    # and the entries of an index at the end are a list
    assert levels(two + [("索引", None), ("第一章 背景 1", None), ("第二章 现状 9", None), ("第三章 展望 15", None)]) == recovered + [None] * 4


def test_a_lone_chapter_line_is_not_recovered():
    """The document must have another chapter of the same kind. "第 2 部分：试验方法" on the cover of a standard names
    the document itself; as a heading it would put the foreword and the first chapter one level down."""
    seq = [("第2部分：试验方法", None), ("前言", 1), ("正文。", None), ("1 范围", 1), ("正文。", None), ("2 规范性引用文件", 1), ("正文。", None)]
    assert levels(seq) == [None, 1, None, 1, None, 1, None]
    assert levels([("第六章 落地执行", None), ("正文。", None), ("2 推进计划", 1), ("正文。", None)]) == [None, None, 1, None]
    # the other chapters of the sequence may be headings the parser marked
    assert levels([("PART I", 1), ("x", None), ("PART II", 1), ("x", None), ("PART III", 1), ("x", None), ("PART IV", None), ("x", None)]
                  ) == [1, None, 1, None, 1, None, 1, None]


def test_the_entries_of_a_list_are_not_recovered():
    """A contents list, a list of enclosures, a summary of the chapters: lines with the same chapter word follow one
    another with nothing but headings and page numbers between them."""
    contents = [("目录", None), ("第一章 总则", None), ("第二章 流程", None), ("第三章 附则", None), ("前言正文一段。", None)]
    assert levels(contents) == [None] * 5
    two_levels = [("第一章 总则", None), ("第一节 目的", None), ("第二节 范围", None), ("第二章 流程", None), ("第一节 申请", None),
                  ("第二节 审批", None), ("正文从这里开始。", None)]
    assert levels(two_levels) == [None] * 7
    numbered = [("第一章 总则", None), ("3", None), ("1.1 目的", None), ("4", None), ("第二章 流程", None), ("7", None), ("正文。", None)]
    assert levels(numbered) == [None] * 7                  # page numbers and numbered sub-entries are not content
    enclosures = [("项目建议书", 1), ("附件：", None), ("附录 A 预算表", None), ("附录 B 时间表", None), ("背景", 1), ("正文。", None)]
    assert levels(enclosures) == [1, None, None, None, 1, None]
    paged = [("第一章 总则", None), ("第 1 页", None), ("第二章 流程", None), ("Page 2 of 9", None), ("第三章 附则", None), ("前言的正文。", None)]
    assert levels(paged) == [None] * 6                     # nor are lines that are nothing but a page number
    # the chapters themselves, each with content below it, are recovered even when such a list stands before them;
    # a table is content like a paragraph
    book = contents + [("第一章 总则", None), ("正文一。", None), ("正文一续。", None), ("第二章 流程", None), TABLE, ("第三章 附则", None), ("正文三。", None)]
    assert levels(book) == [None] * 5 + [1, None, None, 1, None, 1, None]
    # ... but a chapter that follows the list at once is taken for its last entry (the text of chapter 1 keeps the
    # path it had; the later chapters are recovered)
    close = contents[:4] + [("第一章 总则", None), ("正文一。", None), ("第二章 流程", None), ("正文二。", None), ("正文二续。", None),
                            ("第三章 附则", None), ("正文三。", None)]
    assert levels(close) == [None] * 6 + [1, None, None, 1, None]


def test_a_chapter_the_parser_marked_is_not_recovered_a_second_time():
    """The contents list and the running text mention chapters too. When the parser marked "第一章 …" as a heading,
    an unmarked line that names the same chapter is one of those mentions, however either of them is written."""
    got = walk([("第一章 总则", 1), ("正文。", None), ("第一章 总则", None), ("续正文。", None), ("第二章 流程", None), ("正文。", None)])
    assert [level for level, *_ in got] == [1, None, None, None, 1, None]
    assert got[3][2] == ["第一章 总则"]
    assert levels([("第一章总则", 1), ("1.1 目的", 1), ("正文。", None), ("第一章 总则", None), ("续。", None), ("第二章流程", 1), ("正文。", None)]
                  ) == [1, 2, None, None, None, 1, None]                  # the marked heading is written without a separator
    assert levels([("第1章 总则", 1), ("正文。", None), ("第一章 总则", None), ("续。", None), ("第2章 流程", 1), ("正文。", None)]
                  ) == [1, None, None, None, 1, None]                     # one writes the number in digits, the other in words
    assert levels([("CHAPTER 01", 1), ("x", None), ("Chapter 1", None), ("x", None), ("CHAPTER 02", 1), ("x", None)]) == [1, None, None, None, 1, None]


@pytest.mark.parametrize("trust_parser_levels", [False, True])       # PDF, DOCX
def test_chapters_that_start_again_are_not_recovered_in_part(trust_parser_levels):
    """A regulation and its attachment, each with a 第一章. Recovering the chapters that happen to be named once
    (第四章, 第五章 of the regulation) would file the whole attachment below 第五章."""
    seq = ([("管理办法", 1)] + [x for n in "一二三四五" for x in ((f"第{n}章 管理办法第{n}章", None), ("正文。", None))]
           + [("附件：实施细则", 1)] + [x for n in "一二三" for x in ((f"第{n}章 细则第{n}章", None), ("正文。", None))])
    got = walk(seq, trust_parser_levels=trust_parser_levels)
    assert [level for level, *_ in got] == [1] + [None] * 10 + [1] + [None] * 6
    assert got[-1][2] == ["附件：实施细则"]
    # the same for sections that start again in every chapter: the chapters are recovered, the sections are not
    seq = [x for n in "一二" for x in ((f"第{n}章 标题", None), ("正文。", None), ("第一节 甲", None), ("正文。", None), ("第二节 乙", None), ("正文。", None))]
    assert levels(seq, trust_parser_levels=trust_parser_levels) == [1, None, None, None, None, None] * 2


def test_a_gap_in_the_chapters_stops_the_recovery():
    """Chapter 3 has a heading that cannot be recovered (it reads on as a sentence; it might as well be a picture).
    With chapters 1, 2, 4 and 5 recovered, all of chapter 3 would be filed under chapter 2."""
    seq = [x for n, title in ((1, "概述"), (2, "安装"), (3, "调试，以及验收"), (4, "维护"), (5, "报废"))
           for x in ((f"第{n}章 {title}", None), ("正文。", None), ("正文续。", None))]
    assert levels(seq) == [None] * 15
    whole = [x for n in (6, 7, 8) for x in ((f"第{n}章 标题", None), ("正文。", None), ("正文续。", None))]     # an excerpt need not start at chapter 1
    assert levels(whole) == [1, None, None] * 3
    assert levels([("附录 A 表格", None), ("x", None), ("y", None), ("附录 C 图", None), ("x", None)]) == [None] * 5


def test_a_line_about_a_chapter_that_stands_out_of_order_stops_the_recovery():
    """A preface that describes later chapters, each on a line of its own with a sentence below it. The heading of
    chapter 5 is a picture, so no marked heading names chapter 5 -- but the line in the preface stands before
    chapter 1, and chapters do not run 5, 1, 2, 3, 4, 6."""
    seq = ([("前言", 1)] + [x for n in (4, 5, 6) for x in ((f"第{n}章 标题{n}", None), (f"这一章讲标题{n}的内容。", None))] + [("致谢", 1), ("感谢。", None)]
           + [x for n in (1, 2, 3, 4, 6) for x in ((f"第{n}章 标题{n}", 1), ("正文。", None))])
    got = walk(seq)
    assert [level for level, *_ in got][:9] == [1, None, None, None, None, None, None, 1, None]
    assert got[8][2] == ["致谢"]


def test_a_list_of_the_parts_of_a_series_is_no_sign_of_other_chapters():
    """The cover of a standard carries "第2部分：试验方法", its foreword lists the parts of the series. The entries of
    that list are no headings, so the cover line is still a lone chapter line."""
    seq = [("电工电子产品环境试验", 1), ("第2部分：试验方法", None), ("试验A：低温", None), ("前言", 1), ("本系列标准分为以下部分：", None),
           ("——第1部分：总则；", None), ("——第2部分：试验方法；", None), ("——第3部分：导则。", None), ("1 范围", 1), ("正文。", None),
           ("2 规范性引用文件", 1), ("正文。", None)]
    assert levels(seq) == [1, None, None, 1, None, None, None, None, 1, None, 1, None]
    assert levels([("Chapter 7 Liquidation", None), ("Text.", None), ("Chapter 11 of the Bankruptcy Code", None), ("Text.", None)]) == [None] * 4


def test_a_document_that_also_writes_its_chapters_without_a_separator_is_left_alone():
    """Chapter 3's own heading is "第三章配置", not marked; "第三章 配置" on a line of its own inside chapter 2 is a
    mention of it."""
    seq = [("第一章 概述", 1), ("正文。", None), ("第二章 流程", 1), ("审批要点", 1), ("第一句。", None), ("第三章 配置", None), ("以上供参考。", None),
           ("第三章配置", None), ("正文。", None), ("第四章 维护", 1), ("正文。", None)]
    got = walk(seq)
    assert [level for level, *_ in got] == [1, None, 1, 2, None, None, None, None, None, 1, None]
    assert got[6][2] == ["第二章 流程", "审批要点"]


@pytest.mark.parametrize("trust_parser_levels", [False, True])       # PDF, DOCX
def test_a_running_head_is_not_recovered(trust_parser_levels):
    """A book prints the chapter, or the part, at the head of every page. When the recogniser hands those lines over
    as text, a line that names the same chapter more than once is such a head: as a heading it would start the
    chapter again on every page and drop the sub-section the text below it is in."""
    chapter = [("第一章 总则", None), ("1.1 目的", 1), ("正文。", None), ("第一章 总则", None), ("续上页的正文。", None), ("1.2 范围", 1),
               ("正文。", None), ("第二章 流程", None), ("正文。", None), ("第二章 流程", None), ("正文。", None)]
    got = walk(chapter, trust_parser_levels=trust_parser_levels)
    assert [level for level, *_ in got] == [None, 2, None, None, None, 2, None, None, None, None, None]
    assert got[4][2] == ["1.1 目的"]                          # as before the change: the text stays in its sub-section
    part = [("第一章 概述", 1), ("1.1 背景", 1), ("正文一。", None), ("第一部分 基础", None), ("续正文一。", None), ("1.2 目标", 1), ("正文二。", None),
            ("第二章 方法", 1), ("第一部分 基础", None), ("正文三。", None), ("第二部分 提高", None), ("正文四。", None), ("第二部分 提高", None), ("正文五。", None)]
    got = walk(part, trust_parser_levels=trust_parser_levels)
    assert [level for level, *_ in got] == [1, 2, None, None, None, 2, None, 1, None, None, None, None, None, None]
    assert got[4][2] == ["第一章 概述", "1.1 背景"] and got[6][2] == ["第一章 概述", "1.2 目标"]


def test_a_line_on_a_contents_page_is_neither_recovered_nor_taken_for_a_heading():
    seq = [("第一篇 基础", None), ("概述一段。", None), ("又一段。", None), ("第二篇 提高", None), ("概述一段。", None), ("又一段。", None),
           ("第三篇 实战", None), ("概述一段。", None)]
    assert levels(seq) == [1, None, None, 1, None, None, 1, None]
    assert levels(seq, contents={0}) == [None, None, None, 1, None, None, 1, None]     # the first line stands on a contents page
    assert levels(seq[:6], contents={0}) == [None] * 6                     # ... and is no second chapter for the one left
    marked = [("第一篇 基础", 1), ("概述一段。", None), ("第二篇 提高", None), ("概述一段。", None)]
    assert levels(marked) == [1, None, 1, None]
    assert levels(marked, contents={0})[2] is None                         # nor is a marked line on a contents page


# ---------------------------------------------------------------- through the two routes

def item(text, level=None, page=0, kind="text"):
    return {"type": kind, "text": text, "page_idx": page, **({"text_level": level} if level else {})}


def pdf_blocks(content):
    from kb_pipeline.parsers import mineru_pdf

    with tempfile.TemporaryDirectory() as tmp, patch.object(mineru_pdf, "call_mineru_sync", return_value={"content_list": content}):
        return mineru_pdf.mineru_pdf_blocks(mineru_url="http://x", path=Path(tmp) / "a.pdf", cache_dir=Path(tmp) / "c")


def test_the_pdf_route_recovers_the_chapter_and_keeps_steps_as_body_text():
    blocks = pdf_blocks([item("第一章 总则"), item("本制度适用于全体员工。"), item("第二章 流程", 1), item("1. 提交申请"), item("填写申请表并提交。")])
    assert [b.block_type for b in blocks] == ["title", "text", "title", "text", "text"]
    assert [(b.metadata["heading_level"], b.metadata["heading_inferred"]) for b in blocks if b.block_type == "title"] == [(1, True), (1, False)]
    assert [b.metadata["section_path"] for b in blocks] == [["第一章 总则"]] * 2 + [["第二章 流程"]] * 3
    assert blocks[3].text == "1. 提交申请"


def test_the_word_route_recovers_chapters_typed_without_heading_styles():
    from kb_pipeline.parsers import mineru_docx

    payload = {"content_list": [item("**第一章 总则**"), item("本制度适用于全体员工。"), item("自发布之日起施行。"), item("第二章 流程"), item("1. 提交申请")]}
    with tempfile.TemporaryDirectory() as tmp, patch.object(mineru_docx, "call_mineru_sync", return_value=payload):
        blocks = mineru_docx.mineru_docx_blocks(mineru_url="http://x", path=Path(tmp) / "a.docx", cache_dir=Path(tmp))
    assert [(b.block_type, b.text) for b in blocks][:4] == [("title", "第一章 总则"), ("text", "本制度适用于全体员工。"), ("text", "自发布之日起施行。"),
                                                           ("title", "第二章 流程")]
    assert [b.metadata["section_path"] for b in blocks] == [["第一章 总则"]] * 3 + [["第二章 流程"]] * 2


def test_the_pdf_route_recovers_nothing_from_a_contents_page():
    """The page is recognised as contents by its title and its dotted entries; the part lines on it carry no page
    number and read like chapter titles. Recovered, they would put the foreword of the book below a part."""
    contents = ([item("目录", 1, 0), item("序........1", page=0), item("前言........3", page=0), item("第一篇 基础篇", page=0)]
                + [item(f"第{n}章 标题{n}........{n * 9}", page=0) for n in range(1, 6)] + [item("第二篇 进阶篇", page=0)]
                + [item(f"第{n}章 标题{n}........{n * 9}", page=0) for n in range(6, 11)])
    body = [item("序", 1, 1), item("序的正文。", page=1), item("前言", 1, 2), item("前言的正文。", page=2), item("第1章 标题1", 1, 3), item("正文。", page=3),
            item("第一篇 基础篇", page=4), item("这一篇讲基础。", page=4), item("共五章。", page=4), item("第二篇 进阶篇", page=9), item("这一篇讲进阶。", page=9)]
    texts = [(c["page_idx"] + 1, c["text"]) for c in contents + body]
    assert headings.explicit_toc_entries(texts) or headings.detect_toc_pages(texts)          # the page is seen as contents
    blocks = pdf_blocks(contents + body)
    assert [(b.text, b.metadata["heading_level"], b.metadata["section_path"]) for b in blocks if b.block_type == "title"] == [
        ("序", 1, ["序"]), ("前言", 1, ["前言"]), ("第1章 标题1", 1, ["第1章 标题1"]),
        ("第一篇 基础篇", 1, ["第一篇 基础篇"]), ("第二篇 进阶篇", 1, ["第二篇 进阶篇"])]       # the part headings in the body, not the contents lines


def test_the_pdf_route_tells_the_lines_of_a_contents_page_apart_even_when_text_stands_between_them():
    """A contents page that is recognised by its many short lines; every entry has a line of description below it, so
    the entries are no list in the sense of the list rule. Being on a contents page keeps them from being recovered."""
    page = [item("目录", page=0), item("以下为各章内容概要。", page=0)] + [x for n in range(1, 13) for x in (item(f"第{n}章 标题{n}", page=0), item(f"简介{n}", page=0))]
    body = [x for n in range(1, 13) for x in (item(f"第{n}章 标题{n}", page=n), item("这一章的正文，写得比目录里的简介长得多，所以这一页不是目录页。" * 3, page=n),
                                              item("这一章的第二段，同样写得比目录里的简介长，所以每一章下面不止一段话。" * 2, page=n))]
    texts = [(c["page_idx"] + 1, c["text"]) for c in page + body]
    assert headings.detect_toc_pages(texts) == {1} and not headings.explicit_toc_entries(texts)
    entries = [(i, c["text"], False) for i, c in enumerate(page + body)]
    assert len(headings.recoverable_chapter_lines(entries)) == 0                       # every chapter is named twice
    assert sorted(headings.recoverable_chapter_lines(entries, contents=range(len(page)))) == list(range(len(page), len(page + body), 3))
    blocks = pdf_blocks(page + body)
    assert [b.metadata["section_path"] for b in blocks if b.block_type == "title"] == [[f"第{n}章 标题{n}"] for n in range(1, 13)]
    assert all(b.page_idx > 1 for b in blocks if b.block_type == "title")


def test_page_headers_are_no_lines_of_the_document_and_a_table_is_content():
    # the page header that stands between two entries of a list (it repeats on every page) does not make them
    # chapters with text between them
    head = "公司管理制度汇编"
    listed = pdf_blocks([item(head, kind="header", page=0), item("第一章 总则", page=0), item(head, kind="header", page=1),
                         item("第二章 流程", page=1), item("正文。", page=1)])
    assert "title" not in [b.block_type for b in listed]
    # a chapter that holds nothing but a table is still a chapter
    table = {"type": "table", "table_body": "<table><tr><td>a</td><td>b</td></tr></table>", "page_idx": 0}
    blocks = pdf_blocks([item("第一章 参数"), table, item("第二章 说明"), item("正文一。"), item("正文二。"), item("第三章 附则"), item("正文三。")])
    assert [b.block_type for b in blocks] == ["title", "table", "title", "text", "text", "title", "text"]
