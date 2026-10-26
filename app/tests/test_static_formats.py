"""Static ingestion, source identity and retained evidence across file formats."""
import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image

from kb_pipeline.chunking.chunker import blocks_to_chunks
from kb_pipeline.localfs.scanner import is_supported_path, list_source_files, list_recent_source_files
from kb_pipeline.models import KBSource
from kb_pipeline.parsers.common import markdown_to_blocks, parser_profile_for_path, strip_markdown_comments
from kb_pipeline.parsers.errors import NonRetryableParseError
from kb_pipeline.parsers.formats import EXTRA_PROFILES
from kb_pipeline.parsers.native_table import parse_native_table, read_csv_rows_numbered
from kb_pipeline.parsers.notebook import parse_notebook
from kb_pipeline.parsers.router import parse_native
from kb_pipeline.parsers.source_guard import git_lfs_pointer, require_source_content
from kb_pipeline.pipeline.parse_job import _parse_blocks


def chunks(blocks):
    return blocks_to_chunks(kb_id="test", file_key=1, content_version="v", parser_profile="test",
                            blocks=blocks, max_tokens=128, overlap_tokens=16)


def text(blocks):
    return "\n".join(c.text for c in chunks(blocks))


@pytest.mark.parametrize("suffix", [".pdf", ".png", ".csv", ".ipynb", ".md", ".jsonl", ".tsv"])
def test_lfs_pointer_stops_before_services_or_parsers(tmp_path, suffix):
    p = tmp_path / ("source" + suffix)
    p.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:" + "a" * 64 + "\nsize 98765\n")
    assert git_lfs_pointer(p) == {"oid": "a" * 64, "size": 98765}
    with pytest.raises(NonRetryableParseError, match="original file missing"):
        _parse_blocks(SimpleNamespace(), SimpleNamespace(), p, tmp_path, "test", {})


def test_lfs_documentation_and_large_files_are_not_pointers(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("Example:\nversion https://git-lfs.github.com/spec/v1\noid sha256:" + "a" * 64 + "\nsize 1\n")
    require_source_content(p)
    p.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:" + "a" * 64 + "\nsize 1\n" + "x" * 2048)
    assert git_lfs_pointer(p) is None


def test_missing_original_is_not_retried_after_cooldown():
    from kb_pipeline.pipeline.scheduler import _failed_parse_blocks_auto_requeue
    row = {"error": "MissingSourceContent('Git LFS original file missing')", "finished_at": 1}
    assert _failed_parse_blocks_auto_requeue(row)
    assert not _failed_parse_blocks_auto_requeue(row, "parser_changed")


def test_scan_and_console_do_not_read_shebang_outside_source(tmp_path):
    from kb_server.service import _dir_file_count, _dir_count_cache
    root = tmp_path / "mirror" / "Example"
    root.mkdir(parents=True)
    outside = tmp_path / "external"
    outside.write_text("#!/usr/bin/env python\nprint('outside')\n")
    (root / "linked-script").symlink_to(outside)
    source = KBSource(kb_id="test", collection="test", source_root="Example",
                      source_type="local_mirror", physical_base=root, max_tokens=128, overlap_tokens=16)
    with patch("kb_pipeline.localfs.scanner.extensionless_script", side_effect=AssertionError("unexpected read")):
        assert list_source_files(source) == []
        assert list_recent_source_files(source, min_age_seconds=30) == []
        _dir_count_cache.clear()
        assert _dir_file_count(root.parent, root.name, ttl=0) == 0


def test_directory_match_count_uses_scanner_rules(tmp_path):
    import sqlite3
    from kb_pipeline.discovery import directory_match_report
    root = tmp_path / "Example"
    root.mkdir()
    (root / "run").write_text("#!/usr/bin/env python\npass\n")
    (root / "flow.nf").write_text("workflow {}")
    (root / ".hidden.py").write_text("pass")
    outside = tmp_path / "external.py"
    outside.write_text("pass")
    (root / "linked.py").symlink_to(outside)
    with sqlite3.connect(":memory:") as con:
        con.row_factory = sqlite3.Row
        con.execute("CREATE TABLE files (kb_id TEXT, rel_path TEXT, size INTEGER, checksum TEXT)")
        con.execute("INSERT INTO files VALUES ('test', 'linked.py', 4, 'hash')")
        report = directory_match_report(con, tmp_path, "test", root.name)
    assert report["new_files"] == 2
    assert report["matched"] == 0


@pytest.mark.parametrize("flag", [True, False])
def test_standalone_image_keeps_visual_chunk_for_empty_caption(tmp_path, flag):
    from kb_pipeline.parsers.image_file import parse_image_file
    p = tmp_path / "symbol.png"
    Image.new("RGB", (64, 64), "green").save(p)
    result = {"image-0001": {"kind": "photo", "decorative": flag, "summary": "", "facts": [], "entities": []}}
    with patch("kb_pipeline.parsers.visual_blocks.caption_images_parallel", return_value=result):
        blocks = parse_image_file(path=p, cache_dir=tmp_path / "cache", vlm_base_url="mock", vlm_api_key="mock",
                                  vlm_model_id="mock", vlm_concurrency=1)
    assert len(chunks(blocks)) == 1
    assert not blocks[0].metadata["decorative"]
    assert blocks[0].visual_ref


def test_hidden_markdown_comments_do_not_become_headings():
    source = "<!--\n# hidden heading\n#\n-->\n# Visible\nActual evidence.\n<!-- hidden tail -->"
    blocks = markdown_to_blocks(source, parser="native", parser_profile="md", doc_type="md")
    assert len(blocks) == 1
    assert blocks[0].title == "Visible"
    assert "Actual evidence." in text(blocks)
    assert "hidden" not in text(blocks)


def test_markdown_comment_syntax_in_examples_survives():
    source = "Inline `<!-- literal -->`.\n```html\n<!-- example -->\n```\n\n    <!-- indented example -->\nA <!-- hidden --> B"
    cleaned = strip_markdown_comments(source)
    assert "<!-- literal -->" in cleaned
    assert "<!-- example -->" in cleaned
    assert "<!-- indented example -->" in cleaned
    assert "hidden" not in cleaned
    assert "A  B" in cleaned


@pytest.mark.parametrize("source", [
    "Use `<!--\nliteral marker\n-->` as a marker.",
    "> ```html\n> <!-- literal marker -->\n> ```",
    "- ```html\n  <!-- literal marker -->\n  ```",
    "> 1. ~~~~html\n>    <!-- literal marker -->\n>    ~~~~",
    "Use ``alpha ``` <!-- literal marker --> ``` omega``.",
    r"Escaped \<!-- literal marker -->",
    '<div note="<!-- literal marker -->">visible</div>',
    '<script>const example = "<!-- literal marker -->";</script>',
    "    <!-- literal marker -->\n    print('example')",
    "```html\n<!-- literal marker -->\n",  # unclosed code fence
])
def test_markdown_retains_comment_syntax_in_literal_contexts(source):
    assert strip_markdown_comments(source) == source
    assert "literal marker" in text(markdown_to_blocks(source, parser="native", parser_profile="md", doc_type="md"))


@pytest.mark.parametrize("prefix", ["", "<!-- hidden preface -->\n\n", "<!-- first -->\n<!-- second -->\n"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_markdown_frontmatter_literals_survive_hidden_preface(prefix, newline):
    source = prefix + '---\nname: "Template <!-- literal marker -->"\ndescription: |\n  Keep <!-- nested literal --> exactly.\n---\n# Body\nVisible <!-- hidden body --> content.\n'
    source = source.replace("\n", newline)
    blocks = markdown_to_blocks(source, parser="native", parser_profile="md", doc_type="md")
    assert blocks[0].metadata["frontmatter"]["name"] == "Template <!-- literal marker -->"
    assert blocks[0].metadata["frontmatter"]["description"] == "Keep <!-- nested literal --> exactly."
    assert "hidden" not in text(blocks)
    assert "Visible" in text(blocks) and "content." in text(blocks)


def test_markdown_html_comments_respect_attributes_and_unicode_offsets():
    source = '中文 <span title="<!-- literal -->">正文</span> <!-- 隐藏 -->\n<div title="<!-- literal -->">\n<!-- 隐藏 -->\n正文\n</div>\n'
    expected = source.replace("<!-- 隐藏 -->", "")
    assert strip_markdown_comments(source) == expected
    assert strip_markdown_comments(expected) == expected


def test_markdown_grammar_failure_keeps_original_source():
    source = "Use `<!--\nliteral marker\n-->` as a marker."
    with patch("tree_sitter_language_pack.get_parser", side_effect=LookupError("grammar unavailable")):
        assert strip_markdown_comments(source) == source


def test_markdown_unclosed_comment_is_hidden_but_keeps_line_boundaries():
    source = "Visible\n\n<!-- hidden\n# hidden heading"
    assert strip_markdown_comments(source) == "Visible\n\n\n"


def test_markdown_trailing_heading_survives():
    blocks = markdown_to_blocks("# First\nEvidence.\n# Last", parser="native", parser_profile="md", doc_type="md")
    assert "Last" in text(blocks)


def notebook(tmp_path, cells):
    p = tmp_path / "analysis.ipynb"
    p.write_text(json.dumps({"nbformat": 4, "metadata": {"language_info": {"name": "python"}}, "cells": cells}))
    return p


def test_notebook_keeps_cell_output_identity_and_does_not_execute(tmp_path):
    p = notebook(tmp_path, [
        {"cell_type": "markdown", "source": ["# Overview\n", "Sample size is 18."]},
        {"cell_type": "code", "source": "raise RuntimeError('never execute')", "execution_count": 7,
         "outputs": [{"output_type": "stream", "name": "stdout", "text": ["result = 42\n"]},
                     {"output_type": "error", "ename": "ValueError", "evalue": "saved failure", "traceback": ["\u001b[31mtrace\u001b[0m"]}]},
    ])
    blocks = parse_notebook(p, tmp_path / "cache")
    result = text(blocks)
    for value in ("Sample size is 18", "never execute", "result = 42", "saved failure", "trace"):
        assert value in result
    assert "\x1b" not in result
    assert {b.metadata["source_locator"]["cell"] for b in blocks} == {1, 2}
    assert len({b.block_id for b in blocks}) == len(blocks)
    assert [b for b in blocks if b.block_type == "code"][0].metadata["execution_count"] == 7


def test_notebook_html_table_and_saved_plot(tmp_path):
    output = io.BytesIO()
    Image.new("RGB", (32, 32), "blue").save(output, format="PNG")
    p = notebook(tmp_path, [{"cell_type": "code", "source": "plot(data)", "outputs": [
        {"output_type": "display_data", "data": {"text/html": "<script>invisible()</script><table><tr><th>Team</th><th>Count</th></tr><tr><td>A</td><td>19</td></tr></table>", "text/plain": "DUPLICATE"}},
        {"output_type": "display_data", "data": {"image/png": base64.b64encode(output.getvalue()).decode()}}]}])
    blocks = parse_notebook(p, tmp_path / "cache")
    content = text(blocks)
    assert "Team" in content and "19" in content
    assert "DUPLICATE" not in content and "invisible" not in content
    images = [b for b in blocks if b.visual_ref]
    assert len(images) == 1 and Path(images[0].visual_ref).is_file()
    assert images[0].metadata["source_locator"]["output"] == 2


def test_notebook_empty_html_uses_saved_text_alternative(tmp_path):
    p = notebook(tmp_path, [{"cell_type": "code", "source": "display(result)", "outputs": [
        {"output_type": "display_data", "data": {"text/html": "<script>display()</script>", "text/plain": "score = 0.92"}},
        {"output_type": "display_data", "data": {"application/json": {"count": 18}}}]}])
    blocks = parse_notebook(p, tmp_path / "cache")
    assert "score = 0.92" in text(blocks)
    assert blocks[-1].metadata["source_locator"]["mime"] == "application/json"


@pytest.mark.parametrize("bad", ["not json", "[]", '{"nbformat":4}', '{"cells":[null]}'])
def test_invalid_notebook_reports_terminal_source_error(tmp_path, bad):
    p = tmp_path / "bad.ipynb"; p.write_text(bad)
    with pytest.raises(NonRetryableParseError):
        parse_notebook(p, tmp_path / "cache")


def test_jsonl_keeps_records_and_line_numbers(tmp_path):
    p = tmp_path / "records.jsonl"
    p.write_text('\n{"id":"A","rate":0.05,"text":"第一段"}\n\n{"id":"B","count":0,"valid":false}\n')
    blocks = parse_native(p)
    assert len(blocks) == 2
    assert [b.metadata["source_locator"]["line"] for b in blocks] == [2, 4]
    assert "0.05" in text(blocks) and "false" in text(blocks) and "第一段" in text(blocks)
    p.write_text('{"id":1}\n{broken}\n')
    with pytest.raises(NonRetryableParseError, match="line 2"):
        parse_native(p)


@pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-16", "utf-32", "gb18030"])
def test_tsv_preserves_columns_zeros_blanks_and_encoding(tmp_path, encoding):
    p = tmp_path / "metrics.tsv"
    p.write_text('编号\t结果\t备注\n001\t0\t"逗号,保留"\n\n002\t0.05\t通过\n', encoding=encoding)
    rows = read_csv_rows_numbered(p)
    assert rows[1] == (2, ["001", "0", "逗号,保留"])
    assert rows[2][0] == 4
    blocks = parse_native_table(p, max_tokens=128, overlap_tokens=0)
    assert "0.05" in text(blocks) and "编号" in text(blocks)
    assert all(b.parser_profile == "tsv-table-v1" for b in blocks)


def test_r_symbols_and_source_are_preserved(tmp_path):
    p = tmp_path / "sample.R"
    p.write_text("library(stats)\n# Scale values\nscale_values <- function(x) {\n  mean(x)\n}\nanswer <- scale_values(1:3)\n")
    blocks = parse_native(p)
    assert any(b.metadata.get("symbol", {}).get("name") == "scale_values" for b in blocks)
    source = "\n".join(b.text for b in blocks)
    for line in p.read_text().splitlines():
        assert line in source


@pytest.mark.parametrize("name,body,needle", [
    ("guide.rst", "Install\n=======\n\n.. code-block:: sh\n\n   run --count 3\n", "--count 3"),
    ("report.qmd", '---\ntitle: Example\nparams:\n  count: 15\n---\n# Result\n```{python}\nx = 15\n```', "x = 15"),
    ("report.Rmd", '# Result\n```{r}\nvalue <- 12\n```', "value <- 12"),
    ("report.tex", r'\section{Method}' + '\n' + r'$x=\frac{1}{2}$', r'\frac{1}{2}'),
    ("sources.bib", '@article{ref1,\n title={A study},\n year={2024}\n}\n', 'year={2024}'),
    ("flow.nf", 'process RUN {\n input: path reads\n output: path "out.txt"\n}', 'out.txt'),
    ("flow.cwl", 'cwlVersion: v1.2\nclass: Workflow\ninputs:\n  data: File\n', 'data: File'),
    ("flow.wdl", 'version 1.0\nworkflow run {\n input { Int n = 4 }\n}', 'Int n = 4'),
    ("test.feature", 'Feature: Export\n Background:\n Given signed in\n Scenario: CSV\n Then count is 7', 'count is 7'),
    ("graph.dot", 'digraph G { A -> B [label="depends"]; }', 'A -> B'),
    ("events.log", '2026-01-01 INFO count=17\nTraceback:\n  ValueError: expected 9\n', 'expected 9'),
    ("Snakefile", 'rule all:\n    input: "result.txt"\n', 'result.txt'),
    ("run", '#!/usr/bin/env nextflow\nprocess RUN { echo true }', 'process RUN'),
])
def test_extended_text_routes_retain_evidence(tmp_path, name, body, needle):
    p = tmp_path / name; p.write_text(body)
    assert is_supported_path(p)
    assert needle in text(parse_native(p))
    assert parser_profile_for_path(p) in set(EXTRA_PROFILES.values()) | {"workflow-text-v1"}


def test_xsd_preserves_types_constraints_namespace_and_does_not_read_entities(tmp_path):
    secret = tmp_path / "secret.txt"; secret.write_text("DO_NOT_READ")
    p = tmp_path / "schema.xsd"
    p.write_text(f'<!DOCTYPE schema [<!ENTITY external SYSTEM "{secret.as_uri()}">]>'
                 '<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema" targetNamespace="urn:sample">'
                 '<xs:element name="count" type="xs:int"><xs:annotation><xs:documentation>&external;</xs:documentation></xs:annotation></xs:element>'
                 '<xs:simpleType name="Code"><xs:restriction base="xs:string"><xs:maxLength value="12"/></xs:restriction></xs:simpleType></xs:schema>')
    result = text(parse_native(p))
    for value in ('urn:sample', 'xs:int', 'maxLength', '12'):
        assert value in result
    assert 'DO_NOT_READ' not in result


def test_scan_and_route_agree_without_admitting_unrelated_binary_files(tmp_path):
    for name, body in [('run', '#!/bin/bash\necho hi'), ('Snakefile', 'rule all: pass'), ('unknown', 'plain unknown'), ('x.pkl', 'blob'), ('x.svg', '<svg/>'), ('x.sqlite', 'db')]:
        (tmp_path / name).write_text(body)
    source = KBSource(kb_id="test", collection="test", source_root="test", source_type="local_mirror", max_tokens=128, overlap_tokens=16, physical_base=tmp_path)
    files = list_source_files(source)
    assert {f.filename for f in files} == {"run", "Snakefile"}


def test_existing_parser_profiles_do_not_trigger_bulk_reparse():
    expected = {"a.md": "md-sections-v3", "a.pdf": "pdf-mineru-table-vlm-v15", "a.png": "image-vlm-v1",
                "a.py": "py-symbols-v2", "a.csv": "table-native-v4", "a.js": "code-symbols-v2"}
    for name, profile in expected.items():
        assert parser_profile_for_path(Path(name)) == profile


def test_missing_original_withdraws_placeholder_index_and_reports_failure(tmp_path):
    import hashlib
    from kb_pipeline.pipeline import worker
    from test_scanner_worker import _parse_harness
    with _parse_harness(str(tmp_path)) as h:
        row = h.con.execute("SELECT * FROM files WHERE file_id=?", (h.file_id,)).fetchone()
        path = Path(row["physical_path"])
        data = ("version https://git-lfs.github.com/spec/v1\noid sha256:" + "b" * 64 + "\nsize 2500\n").encode()
        path.write_bytes(data)
        sha = hashlib.sha256(data).hexdigest()
        h.con.execute("UPDATE files SET size=?,mtime=?,checksum=?,content_version=?,indexed_version=?,status='indexed' WHERE file_id=?",
                      (len(data), int(path.stat().st_mtime), sha, sha, sha, h.file_id))
        h.con.commit()
        h.hooks["parse"] = lambda: require_source_content(path)
        result = worker.run_once(h.con, settings=h.settings)
        assert result.startswith("failed:") and "original file missing" in result
        assert "mark_stale" in h.calls and "fts" in h.calls
        assert "embed" not in h.calls and "upsert" not in h.calls
        updated = h.con.execute("SELECT status,indexed_version FROM files WHERE file_id=?", (h.file_id,)).fetchone()
        assert updated["status"] == "seen" and updated["indexed_version"] is None
        job = h.con.execute("SELECT status,retry_count FROM jobs WHERE job_id=?", (h.job_id,)).fetchone()
        assert job["status"] == "failed" and not job["retry_count"]


def test_restored_source_is_not_withdrawn_after_placeholder_check(tmp_path):
    from kb_pipeline.pipeline import worker
    from kb_pipeline.parsers.source_guard import MissingSourceContent
    from test_scanner_worker import _parse_harness
    with _parse_harness(str(tmp_path)) as h:
        row = h.con.execute("SELECT * FROM files WHERE file_id=?", (h.file_id,)).fetchone()
        def restore():
            Path(row["physical_path"]).write_text("# Restored original\nActual source has arrived.")
            raise MissingSourceContent("original was missing before replacement")
        h.hooks["parse"] = restore
        result = worker.run_once(h.con, settings=h.settings)
        assert result.startswith("parse-changed-skipped:")
        assert "mark_stale" not in h.calls and "fts" not in h.calls


def test_new_source_locations_reach_search_and_keyword_evidence():
    from kb_pipeline.search_fts import doc_source
    from kb_search.evidence import source_row
    from kb_search.text import position
    locator = {"kind": "notebook", "cell": 4, "output": 2}
    payload = {"text": "accuracy = 0.91", "source_locator": locator, "filename": "evaluation.ipynb"}
    assert doc_source(payload)["source_locator"] == locator
    assert source_row(1, {}, payload)["source_locator"] == locator
    assert "cell 4 output 2" in position(payload)
    assert "line 19" in position({"source_locator": {"kind": "jsonl", "line": 19}})
    assert "rows 2–8" in position({"row_start": 2, "row_end": 8})
