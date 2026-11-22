"""Native Light chunk/vector/keyword storage.

LanceDB keeps payloads and nullable named vectors together; SQLite only catalogs
the tables. No Qdrant/OpenSearch client emulation or database server is used.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import timedelta
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import time
import unicodedata
import uuid


def literal(value):
    """Quote a value for Lance's SQL predicates, never a column or table name."""
    return "'" + str(value).replace("'", "''") + "'"


def one_of(column, values):
    items = list(dict.fromkeys(str(value) for value in values))
    return f"{column} IN ({','.join(map(literal, items))})" if items else "false"


def tokens(text):
    """Same index/query rules: CJK bigrams and whole word/identifier tokens."""
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    cjk = r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]"
    groups = re.findall(cjk + r"+|(?:(?!" + cjk + r")\w)+", text)
    out = []
    for group in groups:
        if re.fullmatch(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]+", group) and len(group) > 1:
            out.extend(group[i:i + 2] for i in range(len(group) - 1))
        else:
            out.append(group)
    return out


FIELDS = ("kb_id", "doc_id", "content_version", "rel_path", "block_id", "chunk_uid")
TEXT_FIELDS = ("filename", "path", "title", "body", "visual")
GRAM_FIELDS = ("filename", "path")


def path_grams(text):
    # Standard indexes 3/4-character substrings on filename/path, not on bodies.
    # Keep that distinction so filenames and partial identifiers still locate
    # their documents without ballooning the prose index.
    groups = str(text).casefold().split()
    return [group[i:i + size] for group in groups for size in (3, 4) for i in range(len(group) - size + 1)]


class LightChunks:
    deadline = None

    def _check_deadline(self):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError('Light query budget exhausted')

    def __init__(self, root: Path):
        # Optional dependencies are loaded only when the adapter is instantiated.
        import lancedb
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.connection = lancedb.connect(self.root / "lance", read_consistency_interval=timedelta(0))
        with self._db() as con:
            con.execute("CREATE TABLE IF NOT EXISTS collections(name TEXT PRIMARY KEY, table_name TEXT UNIQUE NOT NULL, dimensions TEXT NOT NULL)")

    @contextmanager
    def _db(self):
        con = sqlite3.connect(self.root / "catalog.sqlite", timeout=30)
        try:
            con.execute("PRAGMA synchronous=FULL")
            with con:
                yield con
        finally:
            con.close()

    @contextmanager
    def _write_lock(self):
        # Different service processes share this lock; each reader opens a fresh
        # table snapshot and never holds the writer lock during model requests.
        with (self.root / "write.lock").open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _entry(self, collection):
        with self._db() as con:
            return con.execute("SELECT table_name,dimensions FROM collections WHERE name=?", (collection,)).fetchone()

    def _table(self, collection):
        self._check_deadline()
        entry = self._entry(collection)
        if entry is None:
            raise KeyError(f"collection not found: {collection}")
        return self.connection.open_table(entry[0])

    def exists(self, collection):
        return self._entry(collection) is not None

    def check(self):
        # Exercise the store, not merely directory/package existence.
        with self._db() as con:
            if con.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise RuntimeError("Light catalog integrity check failed")
            names = [row[0] for row in con.execute("SELECT name FROM collections")]
        for name in names:
            self._table(name).count_rows()

    def ensure(self, collection, layout, *, dry_run=False):
        import pyarrow as pa
        dims = layout.sizes()
        if any(int(size) <= 0 for size in dims.values()):
            raise ValueError("vector dimensions must be positive")
        if self.exists(collection):
            self.validate(collection, layout)
            return "exists"
        if dry_run:
            return "would_create"
        name = "chunks_" + hashlib.sha256(str(collection).encode()).hexdigest()
        schema = pa.schema([
            ("id", pa.string()), ("payload", pa.string()), ("is_active", pa.bool_()),
            ("file_key", pa.int64()), ("chunk_index", pa.int64()),
            *((key, pa.string()) for key in FIELDS),
            *(("terms_" + key, pa.string()) for key in TEXT_FIELDS),
            *(("grams_" + key, pa.string()) for key in GRAM_FIELDS),
            *((key, pa.list_(pa.float32(), size)) for key, size in dims.items()),
        ])
        with self._write_lock(), self._db() as con:
            existing = con.execute("SELECT dimensions FROM collections WHERE name=?", (collection,)).fetchone()
            if existing:
                if json.loads(existing[0]) != dims:
                    raise ValueError("vector layout mismatch")
                return "exists"
            table = self.connection.create_table(name, schema=schema, exist_ok=True)
            if table.schema != schema:
                raise ValueError("vector layout mismatch for unpublished table")
            con.execute("INSERT INTO collections VALUES(?,?,?)", (collection, name, json.dumps(dims)))
        return "created"

    def validate(self, collection, layout):
        entry = self._entry(collection)
        if entry is None or json.loads(entry[1]) != layout.sizes():
            raise ValueError(f"vector layout mismatch: {collection}")
        return json.loads(entry[1])

    def _row(self, table, point_id, vectors, payload):
        # Reuse the public keyword field composition, not its transport.
        from ..search_fts import doc_source
        fields = doc_source(payload)
        row = {"id": str(point_id), "payload": json.dumps(payload, ensure_ascii=False, allow_nan=False),
               "is_active": payload.get("is_active", True), "file_key": payload.get("file_key"),
               "chunk_index": payload.get("chunk_index")}
        row.update({key: str(payload[key]) if payload.get(key) is not None else None for key in FIELDS})
        row.update({"terms_" + key: " ".join(tokens(fields.get(key, ""))) for key in TEXT_FIELDS})
        row.update({"grams_" + key: " ".join(path_grams(fields.get(key, ""))) for key in GRAM_FIELDS})
        for key in ("text", "visual"):
            vector = vectors.get(key)
            if vector is not None:
                size = table.schema.field(key).type.list_size
                if len(vector) != size or any(not math.isfinite(float(x)) or abs(float(x)) > 3.4028234663852886e38 for x in vector):
                    raise ValueError(f"invalid {key} vector: expected {size} finite values")
                vector = [float(x) for x in vector]
            row[key] = vector  # Missing visual vectors stay null, never zero-filled.
        return row

    def write(self, collection, point_ids, vectors, payloads, *, max_bytes):
        import pyarrow as pa
        if not point_ids:
            return 0
        with self._write_lock():
            table = self._table(collection)
            rows = [self._row(table, pid, vector, payload)
                    for pid, vector, payload in zip(point_ids, vectors, payloads, strict=True)]
            table.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(
                # All provided vectors have already passed dimension/finite
                # checks. Lance classifies intentional null vectors as "bad";
                # preserve those nulls without dropping their source rows.
                pa.Table.from_pylist(rows, schema=table.schema), on_bad_vectors="null")
        return len(rows)

    def _rows(self, collection, where="true", *, vectors=False):
        columns = ["id", "payload"] + (["text", "visual"] if vectors else [])
        return self._table(collection).search().where(where).select(columns).limit(None).to_list()

    def payloads(self, collection, point_ids, *, timeout=None, active_only=True):
        if not point_ids:
            return {}
        where = one_of("id", point_ids)
        if active_only:
            where += " AND is_active = true"
        return {row["id"]: json.loads(row["payload"]) for row in self._rows(collection, where)}

    def vectors(self, collection, point_ids):
        if not point_ids:
            return {}
        return {row["id"]: row["text"] for row in self._rows(collection, one_of("id", point_ids), vectors=True)
                if row["text"] is not None}

    def active_count(self, collection):
        return self._table(collection).count_rows("is_active = true")

    def _filters(self, hints=None, excluded=None):
        hints = hints or {}
        terms = ["is_active = true"]
        for key, field in (("doc_ids", "doc_id"), ("rel_paths", "rel_path")):
            if hints.get(key):
                terms.append(one_of(field, hints[key]))
        if hints.get("content_version"):
            terms.append("content_version = " + literal(hints["content_version"]))
        if excluded:
            terms.append("NOT (" + one_of("rel_path", excluded) + ")")
        return " AND ".join(terms)

    def vector_candidates(self, collection, vector, *, limit, hints=None, excluded=None, visual=False, timeout=None):
        if limit <= 0:
            return []
        column = "visual" if visual else "text"
        table = self._table(collection)
        if len(vector) != table.schema.field(column).type.list_size:
            raise ValueError("query vector dimension mismatch")
        rows = (table.search(vector, vector_column_name=column).distance_type("cosine")
                .bypass_vector_index().where(self._filters(hints, excluded) + f" AND {column} IS NOT NULL", prefilter=True)
                .limit(limit).select(["id", "payload", "_distance"]).to_list())
        self._check_deadline()
        return [{"point_id": row["id"], "score": 1.0 - float(row["_distance"]),
                 "payload": json.loads(row["payload"])} for row in rows]

    def _mutate(self, collection, where, change):
        import pyarrow as pa
        if not self.exists(collection):
            return 0
        with self._write_lock():
            table = self._table(collection)
            records = self._rows(collection, where, vectors=True)
            rows = []
            for record in records:
                payload = json.loads(record["payload"])
                change(payload)
                rows.append(self._row(table, record["id"], record, payload))
            if rows:
                table.merge_insert("id").when_matched_update_all().execute(
                    pa.Table.from_pylist(rows, schema=table.schema), on_bad_vectors="null")
            return len(rows)

    def _file_filter(self, row):
        # The pipeline stores identity in doc_id; file_key is a SQLite ledger
        # field and is not present in ordinary chunk payloads.
        return "doc_id = " + literal(f"{row['kb_id']}:{int(row['file_key'])}")

    @staticmethod
    def _retire(payload):
        payload.update(is_active=False, inactive_at=int(time.time()))

    def retire_previous(self, collection, *, kb_id, file_key, active_content_version):
        where = self._file_filter(dict(kb_id=kb_id, file_key=file_key))
        return self._mutate(collection, where + f" AND content_version != {literal(active_content_version)} AND is_active = true", self._retire)

    def retire_stale(self, collection, *, kb_id, file_key, active_chunk_uids):
        where = self._file_filter(dict(kb_id=kb_id, file_key=file_key))
        return self._mutate(collection, where + " AND is_active = true AND (chunk_uid IS NULL OR NOT (" + one_of("chunk_uid", active_chunk_uids) + "))", self._retire)

    def deactivate(self, row):
        self._mutate(row["collection"], self._file_filter(row) + " AND is_active = true", self._retire)

    def update_metadata(self, row):
        from ..vector.qdrant import metadata_payload_from_file_row
        self._mutate(row["collection"], self._file_filter(row), lambda payload: payload.update(metadata_payload_from_file_row(row)))

    def reactivate(self, row, *, point_ids=None):
        from ..vector.qdrant import metadata_payload_from_file_row
        def change(payload):
            payload.update(metadata_payload_from_file_row(row), is_active=True)
            payload.pop("inactive_at", None)
        if point_ids is None:
            where = self._file_filter(row) + f" AND content_version = {literal(row['content_version'])}"
            self._mutate(row["collection"], where, change)
            return
        ids = list(dict.fromkeys(str(p) for p in point_ids if p))
        for start in range(0, len(ids), 200):
            self._mutate(row["collection"], one_of("id", ids[start : start + 200]), change)

    def version_count(self, row, *, point_ids=None):
        if not self.exists(row["collection"]):
            return 0
        base_where = self._file_filter(row) + f" AND content_version = {literal(row['content_version'])}"
        table = self._table(row["collection"])
        if point_ids is None:
            return table.count_rows(base_where)
        ids = list(dict.fromkeys(str(p) for p in point_ids if p))
        if not ids:
            return 0
        total = 0
        for start in range(0, len(ids), 200):
            total += table.count_rows(base_where + " AND " + one_of("id", ids[start : start + 200]))
        return total

    def context(self, collection, *, doc_id, chunk_from, chunk_to, content_version=None):
        where = f"is_active = true AND doc_id = {literal(doc_id)} AND chunk_index >= {int(chunk_from)} AND chunk_index <= {int(chunk_to)}"
        if content_version:
            where += f" AND content_version = {literal(content_version)}"
        rows = [dict(json.loads(row["payload"]), point_id=row["id"]) for row in self._rows(collection, where)]
        return sorted(rows, key=lambda row: int(row.get("chunk_index") or 0))

    def neighbors(self, collection, payload, *, span, timeout=None):
        index = payload.get("chunk_index")
        if payload.get("doc_id") is None or index is None or span <= 0:
            return []
        try:
            index = int(index)
        except (TypeError, ValueError):
            return []
        return [row for row in self.context(collection, doc_id=payload["doc_id"], chunk_from=max(0, index - span),
                chunk_to=index + span, content_version=payload.get("content_version")) if row.get("chunk_index") != index]

    def metadata(self, collection, point_ids, *, timeout=None):
        payloads = self.payloads(collection, point_ids, active_only=False)
        fields = ("doc_id", "rel_path", "content_version", "chunk_index", "page_idx", "source_locator")
        return {pid: {"active": bool(payloads[pid].get("is_active", True)), **{key: payloads[pid].get(key) for key in fields}}
                if pid in payloads else {"active": False} for pid in point_ids}

    def image_payload(self, collection, point_id):
        point_id = str(uuid.UUID(str(point_id)))
        payload = self.payloads(collection, [point_id]).get(point_id)
        if not payload or not payload.get("visual_ref"):
            raise KeyError(f"active image not found: {point_id}")
        return dict(payload, point_id=point_id)

    def table_head(self, collection, payload, *, timeout=None):
        from kb_search.channels import table_context
        if not payload.get("block_id") or payload.get("doc_id") is None or payload.get("chunk_index") is None:
            return None
        text = str(payload.get("text") or "")
        if "HEADER:" in text[:400]:
            return None
        where = f"is_active = true AND doc_id = {literal(payload['doc_id'])} AND block_id = {literal(payload['block_id'])} AND chunk_index < {int(payload['chunk_index'])}"
        if payload.get("content_version"):
            where += f" AND content_version = {literal(payload['content_version'])}"
        rows = [dict(json.loads(row["payload"]), point_id=row["id"]) for row in self._rows(collection, where)]
        if not rows:
            return None
        head = min(rows, key=lambda row: row["chunk_index"])
        context = table_context(str(head.get("text") or ""))
        return None if context and all(line in {line.strip() for line in text.splitlines()} for line in context) else head


class LightKeywords:
    def __init__(self, chunks: LightChunks, settings=None):
        self.chunks = chunks
        self.settings = settings

    def ensure(self, collections):
        from lancedb.index import FTS
        ready = []
        for collection in collections:
            if self.chunks.exists(collection):
                with self.chunks._write_lock():
                    table = self.chunks._table(collection)
                    names = {index.name for index in table.list_indices()}
                    for field in [*("terms_" + key for key in TEXT_FIELDS), *("grams_" + key for key in GRAM_FIELDS)]:
                        name = "keyword_" + field
                        if name not in names:
                            table.create_index(field, name=name,
                                               config=FTS(base_tokenizer="whitespace", with_position=True,
                                                          stem=False, remove_stop_words=False, max_token_length=None))
                ready.append(collection)
        return {"collections": ready}

    def sync_document(self, collection, doc_id):
        # Keyword fields are in the same row/commit as vectors: replacements and
        # metadata edits cannot leave stale text in a separate search projection.
        if not self.chunks.exists(collection):
            return {"collection": collection, "doc_id": doc_id, "inserted_rows": 0, "deleted_rows": 0}
        self.ensure([collection])
        count = self.chunks._table(collection).count_rows(f"is_active = true AND doc_id = {literal(doc_id)}")
        return {"collection": collection, "doc_id": doc_id, "inserted_rows": count, "deleted_rows": 0}

    def delete(self, collection):
        # Dropping the derived indexes never deletes primary source chunks.
        if self.chunks.exists(collection):
            with self.chunks._write_lock():
                table = self.chunks._table(collection)
                for index in table.list_indices():
                    if index.name.startswith("keyword_"):
                        table.drop_index(index.name)
        return {"collection": collection}

    def candidates(self, question, collection, *, limit, identifiers=None, hints=None, excluded=None, timeout=None):
        from lancedb.query import BooleanQuery, MatchQuery, Occur
        # Identifiers remain searchable as whole tokens. OpenSearch's additional
        # phrase-score boosts are not portable to Lance's BM25 implementation;
        # do not turn them into broad OR-term boosts that displace body evidence.
        query_terms = tokens(question)
        for identifier in identifiers or []:
            query_terms.extend(term for term in tokens(identifier) if term not in query_terms)
        query = " ".join(query_terms)
        if not query or limit <= 0 or not self.chunks.exists(collection):
            return []
        self.ensure([collection])
        # Standard uses most_fields (additive), whereas native MultiMatch uses
        # the strongest matching field. Sum field contributions explicitly.
        parts = [(Occur.SHOULD, MatchQuery(query, "terms_" + field, boost=boost))
                 for field, boost in zip(TEXT_FIELDS, (4., 2., 2., 1., 1.))]
        grams = " ".join(path_grams(question))
        if grams:
            parts.extend((Occur.SHOULD, MatchQuery(grams, "grams_" + field, boost=boost))
                         for field, boost in zip(GRAM_FIELDS, (2., 1.)))
        request = BooleanQuery(parts)
        rows = (self.chunks._table(collection).search(request, query_type="fts")
                .where(self.chunks._filters(hints, excluded), prefilter=True)
                .select(["id", "payload", "_score"]).limit(limit).to_list())
        return [{"point_id": row["id"], "score": float(row["_score"]), "payload": json.loads(row["payload"])} for row in rows]

    def profile(self, question, collections, *, timeout=None):
        terms = list(dict.fromkeys(token for token in tokens(question) if len(token) >= 2))[:16]
        sizes, counts = {}, {term: {} for term in terms}
        for collection in collections:
            if not self.chunks.exists(collection):
                continue
            sizes[collection] = self.chunks.active_count(collection)
            # Counts, not capped search results, drive the existing KB-density rule.
            table = self.chunks._table(collection)
            for term in terms:
                clauses = [f"contains(concat(' ',terms_{field},' '),{literal(' ' + term + ' ')})" for field in TEXT_FIELDS]
                count = table.count_rows("is_active = true AND (" + " OR ".join(clauses) + ")")
                if count:
                    counts[term][collection] = count
        return {"terms": counts, "sizes": sizes}

    def grep(self, source, phrases, *, fields=None, rel_paths=None, doc_ids=None, doc_versions=None,
             limit=30, timeout=None, excluded_paths=None, published_ids=None):
        from collections import Counter
        from kb_search import library
        from ..search_fts import doc_source
        fields = tuple(field for field in library.GREP_FIELDS if field in set(fields or library.GREP_FIELDS))
        phrases = list(dict.fromkeys(str(phrase).strip() for phrase in phrases if str(phrase).strip()))
        paths, docs = list(dict.fromkeys(rel_paths or [])), list(dict.fromkeys(doc_ids or []))
        if set(doc_versions or {}) - set(docs):
            raise ValueError("doc_versions must refer to requested doc_ids")
        if self.settings is None:
            raise RuntimeError("phrase lookup requires the application's version ledger")
        for doc, version in (doc_versions or {}).items():
            library.require_document_version(self.settings, source, doc, version)
        out = {"kb_id": source.kb_id, "kb_name": library.kb_name(source), "fields": list(fields),
               "scope": {"rel_paths": len(paths), "doc_ids": len(docs)}, "results": [],
               "note": "Literal phrase matches in the active document versions."}
        versions = set(library.active_versions(self.settings, source.collection))
        published = set(published_ids) if published_ids is not None else None
        records = []
        if self.chunks.exists(source.collection):
            rows = self.chunks._rows(source.collection, self.chunks._filters({"rel_paths": paths, "doc_ids": docs}, excluded_paths))
            for row in rows:
                payload = json.loads(row["payload"])
                if published is not None and row['id'] not in published:
                    continue
                if payload.get("content_version") not in versions:
                    continue
                if payload.get("doc_id") in (doc_versions or {}) and payload.get("content_version") != doc_versions[payload["doc_id"]]:
                    continue
                records.append(dict(_id=row["id"], _source=doc_source(payload)))
            records.sort(key=lambda row: (row["_source"].get("rel_path", ""), row["_source"].get("chunk_index", 0), row["_id"]))
        else:
            out["degraded"] = [f"{source.kb_id}: keyword index {source.collection} does not exist"]
        for phrase in phrases:
            hits = [library._hit_row(row, phrase, fields) for row in records]
            hits = [hit for hit in hits if hit["literal"]]
            counts = Counter((hit.get("doc_id"), hit.get("rel_path")) for hit in hits)
            documents = [{"doc_id": doc, "rel_path": path, "chunks": count} for (doc, path), count in counts.items()]
            documents.sort(key=lambda row: (-row["chunks"], str(row["rel_path"] or "")))
            shown = hits[:max(0, int(limit))]
            out["results"].append({"phrase": phrase, "total_chunks": len(hits), "docs": documents,
                                   "hits": shown, "literal_checked": len(shown), "literal_true": len(shown)})
        return out
