"""Visibility of document chunks is committed in the existing SQLite ledger.

Vector writes are staged first. A failed write/cleanup cannot publish candidates
or revive retired evidence just because an external index still says active.
"""
from contextlib import contextmanager
from pathlib import Path
import fcntl
import hashlib

from .. import db
from ..chunking.context import payload_text


class PublishedChunks:
    def __init__(self, raw, state_db):
        self.raw, self.state_db = raw, state_db

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def published(self, collection, ids):
        ids = list(dict.fromkeys(map(str, ids)))
        found = {}
        with db.connect(self.state_db) as con:
            con.execute('BEGIN')  # One publication snapshot across lookup batches.
            for offset in range(0, len(ids), 400):
                batch = ids[offset:offset + 400]
                marks = ','.join('?' for _ in batch)
                rows = con.execute(f'''SELECT c.point_id,c.chunk_uid,c.content_version,c.text_sha
                    FROM chunks c JOIN files f ON f.file_id=c.file_id
                    WHERE c.collection=? AND c.status='active' AND f.status!='deleted'
                    AND c.content_version=f.indexed_version AND c.point_id IN ({marks})''',
                    [collection, *batch]).fetchall()
                found.update({r['point_id']: dict(r) for r in rows})
        return found

    def published_ids(self, collection):
        with db.connect(self.state_db) as con:
            return [r[0] for r in con.execute('''SELECT c.point_id FROM chunks c
                JOIN files f ON f.file_id=c.file_id WHERE c.collection=?
                AND c.status='active' AND f.status!='deleted'
                AND c.content_version=f.indexed_version''', (collection,))]

    def _valid(self, collection, payloads):
        ledger = self.published(collection, payloads)
        return {pid: p for pid, p in payloads.items() if pid in ledger
                and p.get('content_version') == ledger[pid]['content_version']
                and p.get('chunk_uid') == ledger[pid]['chunk_uid']
                and db.chunk_text_sha(payload_text(p)) == ledger[pid]['text_sha']}

    def payloads(self, collection, point_ids, *, timeout=None, active_only=True):
        rows = self.raw.payloads(collection, point_ids, timeout=timeout, active_only=active_only)
        valid = self._valid(collection, rows)
        if active_only:
            return valid
        return {pid: dict(p, is_active=pid in valid) for pid, p in rows.items()}

    def _filter_rows(self, collection, rows):
        valid = self._valid(collection, {str(r['point_id']): r.get('payload', r) for r in rows})
        return [r for r in rows if str(r['point_id']) in valid]

    def vector_candidates(self, collection, vector, **options):
        return self._filter_rows(collection, self.raw.vector_candidates(collection, vector, **options))

    def context(self, collection, **options):
        return self._filter_rows(collection, self.raw.context(collection, **options))

    def neighbors(self, collection, payload, **options):
        rows = self._filter_rows(collection, self.raw.neighbors(collection, payload, **options))
        return [r for r in rows if r.get('chunk_revision') == payload.get('chunk_revision')]

    def table_head(self, collection, payload, **options):
        row = self.raw.table_head(collection, payload, **options)
        if row and row.get('chunk_revision') == payload.get('chunk_revision') and self._filter_rows(collection, [row]):
            return row
        return None

    def image_payload(self, collection, point_id):
        row = self.raw.image_payload(collection, point_id)
        if not self._filter_rows(collection, [row]):
            raise KeyError('image is not part of the published document')
        return row

    def metadata(self, collection, point_ids, *, timeout=None):
        rows = self.payloads(collection, point_ids, timeout=timeout)
        fields = ('doc_id', 'rel_path', 'content_version', 'chunk_revision', 'chunk_index', 'page_idx', 'source_locator')
        return {pid: {'active': True, **{k: rows[pid].get(k) for k in fields}}
                if pid in rows else {'active': False} for pid in point_ids}


class PublishedKeywords:
    def __init__(self, raw, chunks):
        self.raw, self.chunks = raw, chunks

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def candidates(self, question, collection, **options):
        rows = self.raw.candidates(question, collection, **options)
        # OpenSearch candidates carry scores, not the canonical payload. Hydrate
        # only published IDs; the main service subsequently reads their text.
        valid = self.chunks.published(collection, [r['point_id'] for r in rows])
        return [r for r in rows if str(r['point_id']) in valid]

    def grep(self, source, phrases, **options):
        # Apply the publication snapshot before counting, not just to the shown
        # snippets: old revisions must not survive in document/count aggregates.
        return self.raw.grep(source, phrases, published_ids=self.chunks.published_ids(source.collection), **options)


def guard(store):
    if not isinstance(store.chunks, PublishedChunks):
        # Storage construction stays lazy; the ledger is required on first read.
        store.chunks = PublishedChunks(store.chunks, getattr(store.settings, 'state_db', None))
        store.keywords = PublishedKeywords(store.keywords, store.chunks)
    return store


def reconcile_document(store, con, row):
    """Idempotent external cleanup after the SQLite publication commit."""
    chunks = con.execute("SELECT chunk_uid FROM chunks WHERE file_id=? AND status='active'", (row['file_id'],)).fetchall()
    if store.chunks.exists(row['collection']):
        store.chunks.retire_stale(row['collection'], kb_id=row['kb_id'], file_key=row['file_key'],
                                  active_chunk_uids={r['chunk_uid'] for r in chunks})
    return store.keywords.sync_document(collection=row['collection'], doc_id=f"{row['kb_id']}:{row['file_key']}")


def repair_indexes(con, store, kb_id, *, ingest_run_id, dry_run=False):
    """Explicit post-restore reconciliation; connection failures never mean missing.

    Compare every published point to its ledger, then queue ordinary parses for
    incomplete documents. This leaves MinerU cache reuse enabled.
    """
    from ..pipeline.scheduler import requeue_single_file
    rows = con.execute("SELECT * FROM files WHERE kb_id=? AND status NOT IN ('deleted','skipped') AND indexed_version IS NOT NULL", (kb_id,)).fetchall()
    missing = []
    for row in rows:
        if con.execute("SELECT 1 FROM jobs WHERE file_id=? AND status='running'", (row['file_id'],)).fetchone():
            raise RuntimeError('Wait for running document jobs before repairing indexes')
        refs = con.execute("SELECT point_id,chunk_uid,text_sha FROM chunks WHERE file_id=? AND status='active'", (row['file_id'],)).fetchall()
        if not refs:
            continue  # An intentionally empty document has no points to restore.
        valid = True
        if not store.chunks.exists(row['collection']):
            valid = False
        else:
            for start in range(0, len(refs), 200):
                batch = refs[start:start + 200]
                payloads = store.chunks.payloads(row['collection'], [r['point_id'] for r in batch], active_only=False)
                if any(r['point_id'] not in payloads or payloads[r['point_id']].get('chunk_uid') != r['chunk_uid']
                       or db.chunk_text_sha(payload_text(payloads[r['point_id']])) != r['text_sha'] for r in batch):
                    valid = False
                    break
        if not valid:
            missing.append(row)
    for row in missing:
        if not dry_run:
            con.execute("UPDATE files SET indexed_version=NULL,status='pending' WHERE file_id=?", (row['file_id'],))
            db.mark_chunks_inactive(con, row['file_id'])
            requeue_single_file(con, ingest_run_id=ingest_run_id, file_row=row, reason='restore missing document index')
    return len(missing)


@contextmanager
def publication_lock(state_db, collection):
    # Serialize only publication/cleanup, never parsing or model calls. A lock
    # per collection prevents a cleanup task from retiring an in-flight write.
    path = Path(state_db).parent / ("publish-" + hashlib.sha256(collection.encode()).hexdigest()[:16] + ".lock")
    with path.open('a+b') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
