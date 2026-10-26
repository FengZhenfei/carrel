"""Missing originals stay outside ingestion; restored sources enter normally."""
import hashlib
import shutil
import subprocess
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import pytest

from kb_pipeline import db
from kb_pipeline.parsers.source_guard import MissingSourceContent, require_source_content
from kb_pipeline.pipeline import scheduler, worker
from kb_server import service
from _support import _local_file
from test_scanner_worker import _parse_harness, _rescan


POINTER = 'version https://git-lfs.github.com/spec/v1\noid sha256:' + 'a' * 64 + '\nsize 12345\n'


@pytest.mark.parametrize('name', ['source.pdf', 'source.png', 'source.gif', 'source.csv',
                                 'source.ipynb', 'source.md', 'source.jsonl', 'Snakefile'])
def test_scan_excludes_pointers_from_inventory_recent_files_and_directory_counts(tmp_path, name):
    from kb_pipeline import discovery
    from kb_pipeline.localfs.scanner import list_source_files, list_recent_source_files
    from kb_pipeline.models import KBSource

    root = tmp_path / 'mirror' / 'Example'; root.mkdir(parents=True)
    path = root / name; path.write_text(POINTER)
    source = KBSource(kb_id='test', collection='test', source_root='Example',
                      source_type='local_mirror', physical_base=root, max_tokens=128, overlap_tokens=16)
    deferred = set()
    assert list_source_files(source, min_age_seconds=60, too_recent_keys=deferred) == []
    assert deferred == set()
    assert list_recent_source_files(source, min_age_seconds=60) == []
    assert service._dir_file_count(root.parent, root.name, ttl=0) == 0
    state = tmp_path / 'state.db'; db.init_db(state)
    with db.connect(state) as con:
        assert discovery.directory_match_report(con, root.parent, 'test', root.name)['new_files'] == 0
        # No SourceFile exists to register or enqueue. Once restored, it enters exactly once.
        assert not con.execute('SELECT 1 FROM files').fetchone()
        assert not con.execute('SELECT 1 FROM jobs').fetchone()
        path.write_text('Complete source content')
        [file] = list_source_files(source)
        assert scheduler.schedule_file(con, ingest_run_id='restored', file=file)[1]
        assert scheduler.schedule_file(con, ingest_run_id='again', file=file)[1] is None
        assert con.execute('SELECT COUNT(*) FROM files').fetchone()[0] == 1
        assert con.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] == 1
    assert service._dir_file_count(root.parent, root.name, ttl=0) == 1


def test_scan_preserves_pointer_documentation_and_limits_after_filtering(tmp_path):
    from kb_pipeline.localfs.scanner import list_source_files
    from kb_pipeline.models import KBSource
    root = tmp_path / 'Example'; root.mkdir()
    (root / 'a.md').write_text(POINTER)
    (root / 'b.md').write_text('# Git LFS example\n\n```\n' + POINTER + '```')
    source = KBSource(kb_id='test', collection='test', source_root='Example',
                      source_type='local_mirror', physical_base=root, max_tokens=128, overlap_tokens=16)
    assert [f.filename for f in list_source_files(source, limit=1, hash_content=False)] == ['b.md']


def test_scan_command_excludes_new_pointers_and_cleans_previously_registered_ones(tmp_path):
    import argparse
    from kb_pipeline import cli, discovery
    with _parse_harness(str(tmp_path)) as h:
        root = h.settings.mirror_root / '库A'
        (root / 'a.md').write_text(POINTER)
        (root / 'new.png').write_text(POINTER)
        h.settings.runtime_dir = tmp_path / 'runtime'
        h.settings.min_file_age_seconds = 0
        h.settings.sources = discovery.enrolled_sources(h.settings.state_db, h.settings.mirror_root)
        args = argparse.Namespace(env_file=None, source=None, limit=None, verbose=False, dry_run=False,
            rehash=False, requeue_failed=False, no_detect_deletes=False, force_kb_teardown=False,
            exit_code_on_recent=False)
        with patch.object(cli, 'load_settings', return_value=h.settings):
            assert cli.cmd_scan(args) == 0
            assert cli.cmd_scan(args) == 0
        assert db.get_file_by_id(h.con, h.file_id)['status'] == 'deleted'
        assert h.con.execute('SELECT COUNT(*) FROM files').fetchone()[0] == 1
        assert not h.con.execute("SELECT 1 FROM jobs WHERE job_type='parse' AND status IN ('queued','running','retry')").fetchone()
        assert h.con.execute("SELECT COUNT(*) FROM jobs WHERE job_type='delete'").fetchone()[0] == 1
        with patch.object(service, 'settings', return_value=h.settings):
            assert service.kb_files(h.kb_id) == []
        assert service._kb_stats(h.con, h.kb_id)['files_total'] == 0


def test_scan_withdraws_indexed_source_replaced_by_pointer_and_restores_without_duplicates(tmp_path):
    from kb_pipeline import discovery
    from kb_pipeline.localfs.scanner import list_source_files
    with _parse_harness(str(tmp_path)) as h:
        path = Path(db.get_file_by_id(h.con, h.file_id)['physical_path'])
        original = path.read_bytes()
        assert worker.run_once(h.con, settings=h.settings).startswith('parse-done:')
        h.con.commit()
        source = discovery.enrolled_sources(h.settings.state_db, h.settings.mirror_root)[h.kb_id]
        path.write_text(POINTER)
        assert list_source_files(source) == []
        def remove():
            return scheduler.schedule_deletes_for_source(h.con, ingest_run_id='scan', kb_id=h.kb_id,
                collection=source.collection, seen_file_keys=set(), verify_physical=True,
                source_root=source.physical_base)
        assert remove() == (1, 1)
        assert remove() == (0, 0)
        h.con.commit()
        with patch.object(worker, 'qdrant_client'), patch.object(worker, 'mark_file_inactive') as withdraw, \
                patch.object(worker, '_sync_fts_doc') as fts:
            assert worker.run_once(h.con, settings=h.settings).startswith('delete-done:')
            withdraw.assert_called_once(); fts.assert_called_once()
        h.con.commit()
        assert h.con.execute("SELECT COUNT(*) FROM chunks WHERE status='active'").fetchone()[0] == 0
        assert service._kb_stats(h.con, h.kb_id)['files_total'] == 0
        path.write_bytes(original)
        change, job = _rescan(h)
        assert change == 'metadata_changed' and job
        assert _rescan(h)[1] is None
        with patch.object(worker, 'qdrant_client'), patch.object(worker, 'file_version_point_count', return_value=1), \
                patch.object(worker, 'reactivate_file_metadata') as restore, patch.object(worker, '_sync_fts_doc'):
            assert worker.run_once(h.con, settings=h.settings).startswith('metadata-done:')
            restore.assert_called_once()
        h.con.commit()
        assert h.con.execute("SELECT COUNT(*) FROM chunks WHERE status='active'").fetchone()[0] == 1
        assert _rescan(h) == ('unchanged', None)
        assert h.con.execute('SELECT COUNT(*) FROM files').fetchone()[0] == 1
        assert h.con.execute("SELECT COUNT(*) FROM jobs WHERE job_type='parse'").fetchone()[0] == 1


@pytest.mark.parametrize('case', ['restored', 'unreadable', 'outside_link'])
def test_delete_recheck_keeps_unconfirmed_placeholders(tmp_path, case):
    from kb_pipeline import discovery
    with _parse_harness(str(tmp_path)) as h:
        path = Path(db.get_file_by_id(h.con, h.file_id)['physical_path'])
        source = discovery.enrolled_sources(h.settings.state_db, h.settings.mirror_root)[h.kb_id]
        if case == 'outside_link':
            outside = tmp_path / 'external.md'; outside.write_text(POINTER)
            path.unlink(); path.symlink_to(outside)
        elif case == 'unreadable':
            path.write_text(POINTER)
        # A restored original, read error, or external symlink cannot authorize withdrawal.
        with patch('kb_pipeline.parsers.source_guard.git_lfs_pointer',
                   side_effect=PermissionError('unreadable')) if case == 'unreadable' else nullcontext():
            assert scheduler.schedule_deletes_for_source(h.con, ingest_run_id='scan', kb_id=h.kb_id,
                collection=source.collection, seen_file_keys=set(), verify_physical=True,
                source_root=source.physical_base) == (0, 0)
        assert db.get_file_by_id(h.con, h.file_id)['status'] != 'deleted'


def test_skip_survives_rescans_and_restored_source_is_indexed(tmp_path):
    with _parse_harness(str(tmp_path)) as h:
        path = Path(db.get_file_by_id(h.con, h.file_id)['physical_path'])
        path.write_text('version https://git-lfs.github.com/spec/v1\noid sha256:' + 'a' * 64 + '\nsize 12345\n')
        # Simulate an old queued pointer, or a scan-to-parse replacement.
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        h.con.execute('UPDATE files SET size=?,mtime=?,checksum=?,content_version=? WHERE file_id=?',
                      (path.stat().st_size, int(path.stat().st_mtime), sha, sha, h.file_id))
        h.con.commit()
        h.hooks['parse'] = lambda: require_source_content(path)
        assert worker.run_once(h.con, settings=h.settings).startswith('skipped:')
        h.con.commit()
        assert db.get_file_by_id(h.con, h.file_id)['status'] == 'skipped'
        from kb_pipeline import discovery
        from kb_pipeline.localfs.scanner import list_source_files
        source = discovery.enrolled_sources(h.settings.state_db, h.settings.mirror_root)[h.kb_id]
        for _ in range(3):
            assert list_source_files(source) == []
        with patch.object(service, 'settings', return_value=h.settings):
            assert service.kb_files(h.kb_id) == []
        stats = service._kb_stats(h.con, h.kb_id)
        assert (stats['files_total'], stats['files_failed'], stats['files_pending']) == (0, 0, 0)
        assert 'files_skipped' not in stats
        assert scheduler.requeue_kb_files(h.con, ingest_run_id='manual', kb_id=h.kb_id, reason='retry') == 0
        # A complete replacement is a new version: one job, then no duplicate enqueue.
        path.write_text('# Restored source\nThe actual document is available now.')
        change, new_job = _rescan(h)
        assert change == 'content_changed' and new_job
        assert _rescan(h)[1] is None
        assert worker.run_once(h.con, settings=h.settings).startswith('parse-done:')
        h.con.commit()
        assert db.get_file_by_id(h.con, h.file_id)['status'] == 'indexed'
        assert h.con.execute("SELECT count(*) FROM chunks WHERE status='active'").fetchone()[0] == 1
        assert not h.con.execute('SELECT 1 FROM failures').fetchone()
        stats = service._kb_stats(h.con, h.kb_id)
        assert (stats['files_indexed'], stats['files_total'], stats['files_pending']) == (1, 1, 0)
        assert _rescan(h) == ('unchanged', None)


def test_skipped_job_does_not_block_the_next_file(tmp_path):
    with _parse_harness(str(tmp_path)) as h:
        other = _local_file('complete.md')
        _, other_job = scheduler.schedule_file(h.con, ingest_run_id='normal', file=other)
        h.con.commit()
        def parse(con, settings, job):
            if job['job_id'] == h.job_id:
                raise MissingSourceContent('Original file missing (Git LFS pointer)')
            return 'parse-done:' + job['job_id']
        with patch.object(worker, 'process_parse_job', side_effect=parse):
            results = [worker.run_once(h.con, settings=h.settings) for _ in range(2)]
        assert any(r.startswith('skipped:') for r in results)
        assert 'parse-done:' + other_job in results
        assert not h.con.execute('SELECT 1 FROM failures').fetchone()


@pytest.mark.parametrize('case', ['owner_changed', 'cancel_requested', 'cancelled'])
def test_skip_cannot_overwrite_a_cancelled_or_reassigned_job(tmp_path, case):
    path = tmp_path / 'state.db'; db.init_db(path)
    with db.connect(path) as con:
        f = _local_file('source.png')
        _, jid = scheduler.schedule_file(con, ingest_run_id='r', file=f)
        db.claim_job_for_types(con, 'owner', lease_seconds=60)
        if case == 'owner_changed':
            con.execute("UPDATE jobs SET locked_by='other' WHERE job_id=?", (jid,))
        elif case == 'cancel_requested':
            con.execute('UPDATE jobs SET cancel_requested=1 WHERE job_id=?', (jid,))
        else:
            db.cancel_job(con, jid, 'cancelled')
        before = dict(con.execute('SELECT * FROM jobs WHERE job_id=?', (jid,)).fetchone())
        assert not db.mark_job_skipped(con, jid, 'missing', worker_id='owner')
        assert dict(con.execute('SELECT * FROM jobs WHERE job_id=?', (jid,)).fetchone()) == before


def test_migration_is_idempotent_and_keeps_skip_reason_after_history_cleanup(tmp_path):
    path = tmp_path / 'state.db'; db.init_db(path)
    with db.connect(path) as con:
        f = _local_file('source.png')
        _, jid = scheduler.schedule_file(con, ingest_run_id='r', file=f)
        db.mark_job_failed(con, jid, "MissingSourceContent('original missing')", retry=False)
        db.add_failure(con, file_id=db.file_id_for(f.kb_id, f.file_key), job_id=jid,
                       stage='worker', error_type='MissingSourceContent', error_message='original missing')
        assert db.migrate_missing_source_jobs(con) == 1
        assert db.migrate_missing_source_jobs(con) == 0
        assert scheduler.schedule_file(con, ingest_run_id='scan', file=f) == ('skipped', None)
        assert con.execute('SELECT resolved_at FROM failures').fetchone()[0]
        con.execute('UPDATE jobs SET finished_at=1,updated_at=1 WHERE job_id=?', (jid,))
        db.prune_job_history(con, retention_days=1)
        assert con.execute('SELECT status FROM jobs WHERE job_id=?', (jid,)).fetchone()[0] == 'skipped'
        stats = service._kb_stats(con, f.kb_id)
        assert stats['files_failed'] == stats['files_pending'] == stats['files_total'] == 0
        assert scheduler.requeue_single_file(con, ingest_run_id='manual', file_row=db.get_file(con, f.kb_id, f.file_key)) is None



@pytest.mark.parametrize('state', ['indexed', 'deleted', 'new_version', 'new_job'])
def test_migration_does_not_overwrite_newer_file_state(tmp_path, state):
    path = tmp_path / 'state.db'; db.init_db(path)
    with db.connect(path) as con:
        f = _local_file('source.png')
        _, jid = scheduler.schedule_file(con, ingest_run_id='r', file=f)
        db.mark_job_failed(con, jid, "MissingSourceContent('original missing')", retry=False)
        fid = db.file_id_for(f.kb_id, f.file_key)
        if state == 'indexed':
            db.mark_file_indexed(con, fid, f.content_version)
        elif state == 'deleted':
            con.execute("UPDATE files SET status='deleted' WHERE file_id=?", (fid,))
        elif state == 'new_version':
            con.execute("UPDATE files SET content_version='restored' WHERE file_id=?", (fid,))
        else:
            scheduler.requeue_single_file(con, ingest_run_id='manual', file_row=db.get_file_by_id(con, fid))
        before = dict(db.get_file_by_id(con, fid))
        assert db.migrate_missing_source_jobs(con) == 1
        assert dict(db.get_file_by_id(con, fid)) == before


def test_console_has_no_missing_source_badge_filter_or_count():
    static = Path(__file__).parents[1] / 'kb_server/static'
    js = (static / 'app.js').read_text()
    html = (static / 'index.html').read_text()
    assert 'files_skipped' not in js and 'skip_reason' not in js
    assert 'value="skipped"' not in html
    assert 't("已跳过")' not in js
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js is unavailable')
    names = ['kbDot', 'parseProgress']
    functions = '\n'.join(js[js.index('function ' + n + '('):].split('\n}', 1)[0] + '\n}' for n in names)
    script = functions + """
const assert = require('node:assert/strict');
const kb = {state:'active',files_total:4,files_indexed:4,files_pending:0,files_failed:0,jobs_active:0};
assert.equal(kbDot(kb), 'green');
assert.deepEqual(parseProgress(kb, []), {done:4,total:4,pct:100});
assert.equal(kbDot({...kb,files_total:0,files_indexed:0}), 'gray');
"""
    subprocess.run([node, '-e', script], check=True, capture_output=True, text=True)
