"""Tier-1 fixes of the 2026-11 audit in the ingest flow: a knowledge base enabled while a scan is listing files (B9),
CSV cells longer than the csv module's default field limit (B16), xlsx cells in the built-in CJK date formats
(B17b), and the scan following the worker's verdict about a failed parse instead of words in its error text (B17f)."""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from kb_pipeline import cli, db, discovery
from kb_pipeline.embedding.client import EmbeddingDimensionError
from kb_pipeline.models import SourceFile
from kb_pipeline.parsers import native_table
from kb_pipeline.parsers.errors import NonRetryableParseError
from kb_pipeline.parsers.mineru_pdf import MinerUServiceError
from kb_pipeline.pipeline import scheduler, worker


# ── B9: the scan's lifecycle step and a knowledge base that was not in this round's source list ──────────────

def _kb_dir(mirror: Path, name: str) -> None:
    (mirror / name).mkdir(parents=True)
    doc = mirror / name / "a.md"
    doc.write_text(f"# {name}\n\nbody", encoding="utf-8")
    os.utime(doc, (time.time() - 3600, time.time() - 3600))        # older than the settle window


def _mirror(tmp_path: Path, *names: str) -> tuple[Path, Path]:
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    for name in names:
        _kb_dir(mirror, name)
    state = tmp_path / "state.db"
    db.init_db(state)
    return mirror, state


def _console(state: Path, action) -> None:
    """Something the web console does on its own connection."""
    with db.connect(state) as con:
        action(con)
        con.commit()


def _scan(mirror: Path, state: Path, capsys, *, during_listing=None) -> tuple[int, str]:
    """One scan round. The source list is read from the registry first, as load_settings does; during_listing runs
    once, when the scan starts listing its first source."""
    settings = SimpleNamespace(state_db=state, mirror_root=mirror, runtime_dir=mirror.parent / "rt",
                               min_file_age_seconds=30, sources=discovery.enrolled_sources(state, mirror))
    args = argparse.Namespace(env_file=None, source=None, limit=None, verbose=False, dry_run=False, rehash=False,
                              requeue_failed=False, no_detect_deletes=False, force_kb_teardown=False,
                              exit_code_on_recent=False)
    real_listing = cli.list_source_files
    pending = [during_listing] if during_listing else []

    def listing(*a, **k):
        if pending:
            pending.pop()()
        return real_listing(*a, **k)

    capsys.readouterr()
    with mock.patch.object(cli, "load_settings", return_value=settings), mock.patch.object(cli, "list_source_files", listing):
        code = cli.cmd_scan(args)
    return code, capsys.readouterr().out


def _registry(state: Path) -> dict[str, tuple[str, str | None]]:
    with db.connect(state) as con:
        return {str(r["source_root"]): (str(r["status"]), r["inactive_reason"])
                for r in con.execute("SELECT source_root, status, inactive_reason FROM kb_sources")}


def _count(state: Path, sql: str) -> int:
    with db.connect(state) as con:
        return int(con.execute(sql).fetchone()[0])


def test_a_kb_enabled_while_the_scan_lists_files_is_not_taken_for_vanished(tmp_path, capsys):
    """The source list is read when the scan starts. A knowledge base enabled in the console after that is active
    in the registry but absent from this round's list; it used to be marked inactive with directory_missing."""
    mirror, state = _mirror(tmp_path, "base-a", "base-b")
    _console(state, lambda con: discovery.enroll(con, mirror, "base-a"))

    code, out = _scan(mirror, state, capsys,
                      during_listing=lambda: _console(state, lambda con: discovery.enroll(con, mirror, "base-b")))

    assert code == 0
    assert _registry(state) == {"base-a": ("active", None), "base-b": ("active", None)}
    assert "kb vanished" not in out and "kb deferred: kb_002 (base-b)" in out
    assert _count(state, "SELECT COUNT(*) FROM jobs WHERE job_type = 'delete'") == 0
    assert _count(state, "SELECT COUNT(*) FROM files") == 1            # only base-a was listed in this round
    assert _scan(mirror, state, capsys)[0] == 0                         # the next round has it in its list
    assert _count(state, "SELECT COUNT(*) FROM files WHERE status != 'deleted'") == 2


def test_a_kb_reopened_while_the_scan_lists_files_stays_open(tmp_path, capsys):
    mirror, state = _mirror(tmp_path, "base-a", "base-b")
    _console(state, lambda con: (discovery.enroll(con, mirror, "base-a"), discovery.enroll(con, mirror, "base-b")))
    assert _scan(mirror, state, capsys)[0] == 0
    closed = next(k for k, s in discovery.enrolled_sources(state, mirror).items() if s.source_root == "base-b")
    _console(state, lambda con: discovery.mark_inactive(con, closed, reason="unenrolled"))
    assert _scan(mirror, state, capsys)[0] == 0                         # closed: its file is retired
    assert _registry(state)["base-b"] == ("inactive", "unenrolled")

    code, _ = _scan(mirror, state, capsys,
                    during_listing=lambda: _console(state, lambda con: discovery.enroll(con, mirror, "base-b")))

    assert code == 0
    assert _registry(state)["base-b"] == ("active", None)
    assert _scan(mirror, state, capsys)[0] == 0
    assert _count(state, "SELECT COUNT(*) FROM files WHERE status != 'deleted'") == 2     # the file is back


def test_a_kb_whose_directory_is_gone_is_still_marked_vanished(tmp_path, capsys):
    mirror, state = _mirror(tmp_path, "base-a", "base-b")
    _console(state, lambda con: (discovery.enroll(con, mirror, "base-a"), discovery.enroll(con, mirror, "base-b")))
    assert _scan(mirror, state, capsys)[0] == 0
    (mirror / "base-b" / "a.md").unlink()
    (mirror / "base-b").rmdir()

    code, out = _scan(mirror, state, capsys)

    assert code == 0 and "kb vanished: kb_002 (base-b)" in out
    assert _registry(state) == {"base-a": ("active", None), "base-b": ("inactive", "directory_missing")}
    assert _count(state, "SELECT COUNT(*) FROM jobs WHERE job_type = 'delete'") == 1


# ── B16: CSV cells longer than the csv module's default field limit ───────────────────────────────────────────

def test_csv_with_a_cell_longer_than_the_default_field_limit_is_read(tmp_path):
    """csv.field_size_limit() defaults to 131,072 characters; one longer cell used to make the whole file unreadable."""
    path = tmp_path / "long.csv"
    path.write_text("id,note\n1,short\n2," + "长文本" * 50_000 + "\n3,after\n", encoding="utf-8")

    rows = native_table.read_csv_rows_numbered(path)

    assert [row_no for row_no, _ in rows] == [1, 2, 3, 4]
    assert len(rows[2][1][1]) == 150_000 and rows[3][1] == ["3", "after"]


def test_a_csv_error_is_reported_as_itself_and_not_retried_under_other_encodings(tmp_path):
    """Only a decoding failure says the encoding guess was wrong. An error of the csv module is the same under every
    encoding: the file used to be read five times and the message named the last decoding error (gbk)."""
    path = tmp_path / "wide.csv"
    path.write_text("id,note\n1," + "长文本" * 200 + "\n", encoding="utf-8")
    attempts = []
    real = native_table._read_csv_rows_with_encoding

    def counted(p, *, encoding, errors):
        attempts.append((encoding, errors))
        return real(p, encoding=encoding, errors=errors)

    previous = csv.field_size_limit(100)             # provoke an error of the csv module itself
    try:
        with mock.patch.object(native_table, "_read_csv_rows_with_encoding", counted), pytest.raises(RuntimeError) as caught:
            native_table.read_csv_rows_numbered(path)
    finally:
        csv.field_size_limit(previous)

    assert attempts == [("utf-8-sig", "strict")]
    assert "field larger than field limit" in str(caught.value) and "gbk" not in str(caught.value)
    assert isinstance(caught.value.__cause__, csv.Error)


@pytest.mark.parametrize("encoding", ["gb18030", "utf-16", "utf-8-sig"])
def test_csv_in_other_encodings_is_still_found_through_the_encoding_chain(tmp_path, encoding):
    path = tmp_path / "enc.csv"
    path.write_text("编号,说明\n1,第一行\n\n2,第二行\n", encoding=encoding)

    assert native_table.read_csv_rows_numbered(path) == [(1, ["编号", "说明"]), (2, ["1", "第一行"]), (4, ["2", "第二行"])]


# ── B17(b): xlsx cells in the built-in CJK date formats ──────────────────────────────────────────────────────

_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def _workbook(path: Path, cells: list[tuple[int, object]]) -> Path:
    """A one-sheet workbook written by hand: row n holds one number in column A, styled with number format id
    cells[n][0]. Format 176 is defined in styles.xml as a custom date format; the others are built-in ids."""
    ids = [fmt_id for fmt_id, _ in cells]
    xfs = "".join(f'<xf numFmtId="{i}" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>' for i in ids)
    rows = "".join(f'<row r="{n}"><c r="A{n}" s="{n - 1}"><v>{value}</v></c></row>' for n, (_, value) in enumerate(cells, start=1))
    parts = {
        "[Content_Types].xml": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/></Types>',
        "_rels/.rels": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>',
        "xl/workbook.xml": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<workbook xmlns="{_MAIN}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="Dates" sheetId="1" r:id="rId1"/></sheets></workbook>',
        "xl/_rels/workbook.xml.rels": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>',
        "xl/styles.xml": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<styleSheet xmlns="{_MAIN}"><numFmts count="1"><numFmt numFmtId="176" formatCode="yyyy&quot;年&quot;m&quot;月&quot;d&quot;日&quot;"/></numFmts>'
            '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
            '<fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>'
            '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
            '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
            f'<cellXfs count="{len(ids)}">{xfs}</cellXfs>'
            '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>',
        "xl/worksheets/sheet1.xml": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<worksheet xmlns="{_MAIN}"><sheetData>{rows}</sheetData></worksheet>',
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in parts.items():
            archive.writestr(name, data)
    return path


def _column_a(sheets) -> list[str]:
    (_, rows, _), = sheets
    return [cells[0] for _, cells in rows]


# 45123 is the serial number of 2023-07-16. Ids 31, 57, 58 and 27 are built-in formats reserved for CJK date
# formats, 32 for a CJK time format; none of them is written out in the file.
_DATED = [(31, 45123), (57, 45123), (58, 45123), (27, 45123), (32, 0.5)]
_UNCHANGED = [(0, 45123), (14, 45123), (176, 45123), (2, 45123.5)]


@pytest.mark.parametrize("reader", [native_table.read_xlsx_rows, native_table.read_xlsx_rows_via_xml])
def test_xlsx_builtin_cjk_date_formats_are_read_as_dates(tmp_path, reader):
    """Dates used to come out as the serial number (45123) through openpyxl, and the XML fallback refused the
    whole sheet ("Cannot safely recover XLSX number format")."""
    book = _workbook(tmp_path / "dates.xlsx", _DATED)

    assert _column_a(reader(book)) == ["2023-07-16", "2023-07-16", "2023-07-16", "2023-07-16", "12:00:00"]


@pytest.mark.parametrize("reader", [native_table.read_xlsx_rows, native_table.read_xlsx_rows_via_xml])
def test_other_xlsx_number_formats_read_as_before(tmp_path, reader):
    book = _workbook(tmp_path / "plain.xlsx", _UNCHANGED)

    assert _column_a(reader(book)) == ["45123", "2023-07-16", "2023-07-16", "45123.5"]


def test_a_number_outside_the_date_range_keeps_its_value(tmp_path):
    """20230716 typed as a number and then given a date format is not a valid date serial (Excel shows ####):
    the number is kept as it is, and the sheet is still read."""
    book = _workbook(tmp_path / "typed.xlsx", [(31, 20230716), (58, -700000.0)])

    assert _column_a(native_table.read_xlsx_rows(book)) == ["20230716", "-700000"]


# ── B17(f): the scan follows the worker's verdict about a failed parse ───────────────────────────────────────

_DAY = 25 * 3600
_VLM_GATE = "VLM 描述失败 1/200 张,本次解析不入库以免留下没有图片描述的文档"


def _settings(state: Path, *, max_retries: int = 2):
    return SimpleNamespace(state_db=state, parse_enabled=True, parse_job_lease_seconds=600, metadata_job_lease_seconds=600,
                           job_max_retries=max_retries, job_retry_base_seconds=300, job_retry_max_seconds=3600,
                           qdrant_url="http://q.invalid", qdrant_api_key="", opensearch_url="http://o.invalid", sources={})


def _file(name: str = "manual.pdf") -> SourceFile:
    return SourceFile(kb_id="kb", collection="c", source_root="/kb", source_type="local_mirror", file_key=1,
                      source_path=f"/kb/{name}", rel_path=name, filename=name, dir="", physical_path=f"/kb/{name}",
                      mime_type="application/octet-stream", size=1234, mtime=1700000000, checksum="unchanged")


def _fail_through_the_worker(con, settings, error: Exception) -> int:
    """The real worker claims the job; the parse itself is replaced by one that raises `error`. Retries are run
    as soon as they are scheduled. Returns the number of parse attempts."""
    def parse(*args, **kwargs):
        raise error

    attempts = 0
    while True:
        con.execute("UPDATE jobs SET next_attempt_at = 0 WHERE status = 'retry'")
        con.commit()
        with mock.patch.object(worker, "process_parse_job", parse):
            result = worker.run_once(con, settings=settings)
        con.commit()
        if result == "no-job":
            return attempts
        attempts += 1


def _failed_job(con):
    return con.execute("SELECT * FROM jobs WHERE status = 'failed' ORDER BY rowid DESC LIMIT 1").fetchone()


def _a_day_later(con) -> None:
    con.execute("UPDATE jobs SET finished_at = finished_at - ? WHERE status = 'failed'", (_DAY,))
    con.commit()


def _failed_file(tmp_path, error: Exception, *, name: str = "manual.pdf"):
    state = tmp_path / "state.db"
    db.init_db(state)
    con = db.connect(state)
    file = _file(name)
    assert scheduler.schedule_file(con, ingest_run_id="scan-1", file=file)[1] is not None
    con.commit()
    attempts = _fail_through_the_worker(con, _settings(state), error)
    return con, file, attempts


@pytest.mark.parametrize("error", [
    MinerUServiceError("400 Client Error: Bad Request for url: http://mineru/file_parse", status_code=400),
    MinerUServiceError("422 Unprocessable Entity for url: http://mineru/file_parse", status_code=422),
], ids=["mineru-400", "mineru-422"])
def test_a_failure_the_worker_does_not_retry_is_not_queued_again_by_the_scan(tmp_path, error):
    """The worker fails document rejections at once (by exception type). The scan used to look for words in the error text, found
    none, and queued the unchanged file again after every cool-down."""
    con, file, attempts = _failed_file(tmp_path, error)
    with con:
        assert attempts == 1
        _a_day_later(con)
        assert scheduler.schedule_file(con, ingest_run_id="scan-2", file=file) == ("parse_failed", None)
        assert scheduler._failed_parse_blocks_auto_requeue(_failed_job(con), "parser_changed") is False   # a parser change reopens it
        assert scheduler.schedule_file(con, ingest_run_id="scan-3", file=file, requeue_failed=True)[1] is not None   # so does a manual requeue


@pytest.mark.parametrize("error", [
    MinerUServiceError("409 Client Error: Conflict for url: http://mineru/file_parse", status_code=409),
    MinerUServiceError("401 Unauthorized for url: http://mineru/file_parse", status_code=401),
    EmbeddingDimensionError("expected 1024 dimensions, got 768"),
], ids=["mineru-409", "mineru-401", "embedding-dimension"])
def test_a_config_failure_skips_backoff_and_heals_after_cooldown(tmp_path, error):
    """Auth/model/dimension errors skip immediate 5-round backoff (1 attempt), hold inside the cooldown,
    and heal automatically once the cooldown elapses (or immediately on config/LLM save)."""
    con, file, attempts = _failed_file(tmp_path, error)
    with con:
        assert attempts == 1
        assert scheduler.schedule_file(con, ingest_run_id="scan-2", file=file) == ("parse_failed", None)
        _a_day_later(con)
        change, job_id = scheduler.schedule_file(con, ingest_run_id="scan-3", file=file)
        assert change == "needs_parse" and job_id is not None


def test_a_non_retryable_parse_error_stays_held_until_the_parser_changes(tmp_path):
    con, file, attempts = _failed_file(tmp_path, NonRetryableParseError("parser route for .xyz is not enabled in this phase"))
    with con:
        assert attempts == 1
        _a_day_later(con)
        assert scheduler.schedule_file(con, ingest_run_id="scan-2", file=file) == ("parse_failed", None)
        assert scheduler._failed_parse_blocks_auto_requeue(_failed_job(con), "parser_changed") is False


def test_a_retryable_failure_keeps_its_retries_and_one_requeue_per_cooldown(tmp_path):
    """The image-description gate raises an ordinary RuntimeError: retried with back-off, then failed, then queued
    again by the scan once the cool-down has passed. None of that changes."""
    con, file, attempts = _failed_file(tmp_path, RuntimeError(_VLM_GATE), name="figures.pdf")
    with con:
        assert (attempts, _failed_job(con)["retry_count"]) == (3, 2)   # job_max_retries = 2
        assert scheduler.schedule_file(con, ingest_run_id="scan-2", file=file) == ("parse_failed", None)   # inside the cool-down
        _a_day_later(con)
        change, job_id = scheduler.schedule_file(con, ingest_run_id="scan-3", file=file)
        assert change == "needs_parse" and job_id is not None


@pytest.mark.parametrize("name, message", [
    ("非重试策略说明.pdf", "503 Server Error for /kb/非重试策略说明.pdf"),
    ("manual.pdf", "upstream said: NonRetryableParseError is not expected here"),
], ids=["words-in-the-path", "class-name-in-the-text"])
def test_words_in_the_error_text_do_not_turn_a_retryable_failure_into_a_final_one(tmp_path, name, message):
    """The worker retried these to the end; the scan then found the words in the stored text and never queued the
    file again."""
    con, file, attempts = _failed_file(tmp_path, RuntimeError(message), name=name)
    with con:
        assert attempts == 3
        _a_day_later(con)
        assert scheduler.schedule_file(con, ingest_run_id="scan-2", file=file)[1] is not None


def test_a_file_that_was_missing_is_queued_as_soon_as_the_scan_sees_it_again(tmp_path):
    con, file, attempts = _failed_file(tmp_path, NonRetryableParseError("physical file not found: /kb/manual.pdf"))
    with con:
        assert attempts == 1
        assert scheduler.schedule_file(con, ingest_run_id="scan-2", file=file)[1] is not None     # no cool-down


@pytest.mark.parametrize("error, verdict", [
    (MinerUServiceError("400 Client Error", status_code=400), "deterministic"),
    (MinerUServiceError("409 Client Error", status_code=409), "config"),
    (EmbeddingDimensionError("expected 1024, got 768"), "config"),
    (NonRetryableParseError("parser route for .xyz is not enabled in this phase"), "deterministic"),
    (NonRetryableParseError("physical file not found: /kb/manual.pdf"), "file_missing"),
    (RuntimeError(_VLM_GATE), "retryable"),
], ids=["mineru-400", "mineru-409", "embed-dim", "non-retryable", "file-missing", "retryable"])
def test_the_worker_records_its_verdict_on_the_failed_job(tmp_path, error, verdict):
    con, _, _ = _failed_file(tmp_path, error)
    with con:
        payload = json.loads(_failed_job(con)["payload_json"])
    assert payload["failure_verdict"] == verdict
    assert payload["source_path"] == "/kb/manual.pdf"                  # what the scan wrote into the payload is kept


def test_jobs_that_failed_before_the_verdict_was_recorded_keep_the_text_rule(tmp_path):
    held = scheduler._failed_parse_blocks_auto_requeue
    long_ago, now = 1, int(time.time())
    for row in ({"error": "NonRetryableParseError('no parser route')", "finished_at": long_ago},
                {"error": "NonRetryableParseError('no parser route')", "finished_at": long_ago, "payload_json": "{}"},
                {"error": "NonRetryableParseError('no parser route')", "finished_at": long_ago, "payload_json": '{"reason": "new"}'},
                {"error": "NonRetryableParseError('no parser route')", "finished_at": long_ago, "payload_json": "not json"},
                {"error": "NonRetryableParseError('no parser route')", "finished_at": long_ago, "payload_json": '{"failure_verdict": "?"}'}):
        assert held(row) and not held(row, "parser_changed")
    assert not held({"error": "MinerUServiceError('409 Client Error')", "finished_at": long_ago, "payload_json": "{}"})
    assert held({"error": "MinerUServiceError('409 Client Error')", "finished_at": now, "payload_json": "{}"})
    assert not held({"error": "RuntimeError('physical file not found: /kb/a.pdf')", "finished_at": now})


def test_the_verdict_is_recorded_for_parse_jobs_only(tmp_path):
    """Failed delete / metadata jobs are queued again by the scan with a copy of their payload; nothing is added to it."""
    state = tmp_path / "state.db"
    db.init_db(state)
    with db.connect(state) as con:
        file = _file("a.md")
        db.upsert_file(con, file, status="deleted")
        fid = db.file_id_for(file.kb_id, file.file_key)
        job_id = db.enqueue_job(con, ingest_run_id=None, file_id=fid, kb_id=file.kb_id, collection=file.collection,
                                file_key=file.file_key, job_type="delete", payload={"reason": "gone"}, dedupe_key=f"delete:{fid}")
        con.commit()
        with mock.patch.object(worker, "get_storage", side_effect=RuntimeError("index store rejected the request")):
            result = worker.run_once(con, settings=_settings(state, max_retries=0))
        con.commit()
        row = con.execute("SELECT status, payload_json FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    assert result.startswith("failed:") and row["status"] == "failed"
    assert json.loads(row["payload_json"]) == {"reason": "gone"}
