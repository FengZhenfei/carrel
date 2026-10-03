"""A knowledge base's document list and literal counts: for "deep research" callers that need every document a
knowledge base holds, or how many chunks of which documents mention a phrase.
Both are read-only: the list reads the state database (the same database and the same way of connecting as
catalog.py), the count queries the search process's own OpenSearch client; no generative model is called.
Responses carry only paths relative to the knowledge base, never absolute server paths (physical_path /
source_path / source_root)."""
from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from pathlib import PurePosixPath
from typing import Any, Iterable

from kb_pipeline import db
from kb_pipeline.config import Settings
from kb_pipeline.models import KBSource

from .graphwalk import kb_name

ERROR_CHARS = 300
# The parts of the chunking check carried to callers: how many blocks were parsed, how many headings were
# recognised, how many chunks each block type was cut into, whether the check passed and why not; the other
# statistics (length distribution, fragment thresholds) are for tuning the chunker in the console
DIAG_STATS = ("blocks", "headings", "tokens_total")

GREP_FIELDS = ("body", "title", "visual")
GREP_SOURCE = ["doc_id", "rel_path", "chunk_index", "block_type", "page_idx", "sheet_name", "row_start", "row_end", "content_version",
               *GREP_FIELDS]
GREP_DOC_BUCKETS = 1000
SNIPPET_CHARS = 60
GREP_NOTE = "total_chunks and docs count the analyzer's phrase matches; literal is checked only for the hits returned"
# The keyword index is a per-document projection of the active Qdrant points and has no is_active: for documents
# whose sync failed and whose repair job has not finished, old-version chunks can still be in the index. Old
# content versions are filtered out here by the state database's current versions; old chunks left by re-chunking
# the same version cannot be, and the note says so
GREP_STALE_NOTE = ("only chunks of current content versions are counted (filtered by the state database); for documents whose "
                   "keyword-index sync failed and whose repair job has not finished, the counts may differ from the current state")


def _diag(raw: Any) -> dict[str, Any] | None:
    """A few items of files.chunk_diag_json (the check written by chunking/diagnose.py), carried as they are;
    empty or unreadable ones are left out."""
    if not raw:
        return None
    try:
        diag = json.loads(raw)
        stats = diag.get("stats") or {}
        out: dict[str, Any] = {"ok": bool(diag.get("ok")), "reasons": [str(r.get("key")) for r in diag.get("reasons") or [] if r.get("key")]}
        for key in DIAG_STATS:
            if stats.get(key) is not None:
                out[key] = stats[key]
        by_type = stats.get("by_block_type") or {}
        if by_type:
            out["block_types"] = {str(k): int((v or {}).get("chunks") or 0) for k, v in sorted(by_type.items())}
        return out
    except (ValueError, TypeError, AttributeError):
        return None


# Fallback: any absolute path still left (code files, other directories) keeps only its last segment; it must not
# directly follow a letter, digit or dot, so that "kb/s" or "1/2" are not taken for paths
_ABS_PATH = re.compile(r"(?<![\w.])/(?:[^\s'\"/]+/)+([^\s'\"/]*)")


def _clean_error(message: Any, prefixes: Iterable[str]) -> str:
    """Strip absolute server paths from a failure message before it leaves. The worker writes "exception text +
    the whole traceback": the traceback holds absolute paths of code files (home directory, deployment directory)
    and the exception text often holds the physical_path under the mirror, both within the first 300 characters.
    So the traceback is cut off first, then this file's mirror prefix (physical_path without its rel_path) and the
    mirror root are removed so that only paths relative to the knowledge base remain, and any other absolute path
    keeps only its file name; the text is cut to 300 characters last, so that a half prefix at the cut cannot
    escape the replacement."""
    text = str(message or "")
    cut = text.find("Traceback (most recent call last)")
    if cut >= 0:
        text = text[:cut]
    for prefix in sorted({p.rstrip("/") for p in prefixes if p and p.strip("/")}, key=len, reverse=True):
        text = text.replace(prefix + "/", "")
    text = _ABS_PATH.sub(r"\1", text)
    return text.strip()[:ERROR_CHARS]


def _mirror_prefix(physical: Any, rel: str) -> str:
    """physical_path without its trailing rel_path is this knowledge base's root in the mirror (an absolute path);
    when they do not line up (renamed, older data) nothing is returned."""
    physical = str(physical or "")
    return physical[:-len(rel)] if rel and physical.endswith("/" + rel) else ""


def _file_rows(con: sqlite3.Connection, kb_id: str, *, include_deleted: bool) -> list[sqlite3.Row]:
    # physical_path is used only to recognise the mirror prefix in failure messages and never enters the response
    cols = ("file_key, rel_path, filename, dir, physical_path, mime_type, size, mtime, content_version, status, indexed_version, "
            "indexed_parser_profile, first_seen_at")
    where = "WHERE kb_id = ?" + ("" if include_deleted else " AND status != 'deleted'")
    try:
        return con.execute(f"SELECT {cols}, chunk_diag_json FROM files {where} ORDER BY rel_path, file_key", (kb_id,)).fetchall()
    except sqlite3.OperationalError as exc:
        if "chunk_diag_json" not in str(exc):            # an older database without this column (before migration): no check; other errors still raise
            raise
        return con.execute(f"SELECT {cols}, NULL AS chunk_diag_json FROM files {where} ORDER BY rel_path, file_key", (kb_id,)).fetchall()


def list_docs(settings: Settings, source: KBSource, *, dir: str | None = None, name: str | None = None, include_deleted: bool = False,
              limit: int = 500, offset: int = 0) -> dict[str, Any]:
    """Every document of a knowledge base (sorted by rel_path, with a total and paging): which version was parsed,
    how many chunks it was cut into, the latest unresolved failure.
    Only active chunks of the current content version are counted, matching what search can see; indexed requires
    the parsed version to be the current version and at least one chunk.
    totals covers every document after filtering and is not affected by paging. Filtering happens in process: a
    knowledge base has a few hundred to a few thousand files, and reading them in one go is simpler than building
    LIKE patterns, with no need to escape % and _ in paths."""
    limit = max(1, int(limit))
    offset = max(0, int(offset))
    prefix = str(dir or "").strip().strip("/")
    needle = str(name or "").strip().casefold()
    with db.connect(settings.state_db) as con:
        files = _file_rows(con, source.kb_id, include_deleted=include_deleted)
        counts = {str(r[0]): int(r[1]) for r in con.execute(
            "SELECT c.file_id, COUNT(*) FROM chunks c JOIN files f ON f.file_id = c.file_id "
            "WHERE f.kb_id = ? AND c.collection = ? AND c.status = 'active' AND c.content_version = f.content_version GROUP BY c.file_id",
            (source.kb_id, source.collection)).fetchall()}
        failures: dict[str, sqlite3.Row] = {}
        for r in con.execute(
                "SELECT file_id, stage, error_type, error_message, created_at FROM failures "
                "WHERE resolved_at IS NULL AND file_id IN (SELECT file_id FROM files WHERE kb_id = ?) ORDER BY created_at, failure_id",
                (source.kb_id,)).fetchall():
            failures[str(r["file_id"])] = r            # overwritten in ascending time order: each file keeps its latest
    picked = []
    for f in files:
        rel = str(f["rel_path"])
        if prefix and not rel.startswith(prefix + "/"):
            continue
        if needle and needle not in rel.casefold():
            continue
        picked.append(f)
    rows = []
    totals = {"files": len(picked), "indexed": 0, "not_indexed": 0, "chunks": 0}
    for f in picked:
        doc_id = f"{source.kb_id}:{f['file_key']}"
        chunk_total = counts.get(doc_id, 0)
        indexed = bool(f["indexed_version"]) and f["indexed_version"] == f["content_version"] and chunk_total > 0
        totals["indexed" if indexed else "not_indexed"] += 1
        totals["chunks"] += chunk_total
        rows.append((f, doc_id, chunk_total, indexed))
    mirror = str(getattr(settings, "mirror_root", "") or "")
    docs = []
    for i, (f, doc_id, chunk_total, indexed) in enumerate(rows[offset:offset + limit]):
        filename = str(f["filename"])
        row: dict[str, Any] = {
            "n": offset + i + 1, "doc_id": doc_id, "rel_path": f["rel_path"], "filename": filename, "dir": f["dir"],
            "doc_type": PurePosixPath(filename).suffix[1:].lower() or None, "mime_type": f["mime_type"], "size": f["size"], "mtime": f["mtime"],
            "content_version": f["content_version"], "status": f["status"], "indexed": indexed, "chunk_total": chunk_total,
            "parser_profile": f["indexed_parser_profile"], "first_seen_at": f["first_seen_at"],
        }
        diag = _diag(f["chunk_diag_json"])
        if diag is not None:
            row["diag"] = diag
        fail = failures.get(doc_id)
        if fail is not None:
            row["last_error"] = {"stage": fail["stage"], "error_type": fail["error_type"],
                                 "message": _clean_error(fail["error_message"], (_mirror_prefix(f["physical_path"], str(f["rel_path"])), mirror)),
                                 "at": fail["created_at"]}
        docs.append(row)
    return {"kb_id": source.kb_id, "kb_name": kb_name(source), "total": len(rows), "offset": offset, "limit": limit, "count": len(docs),
            "has_more": offset + len(docs) < len(rows), "totals": totals, "docs": docs}


def doc_paths(settings: Settings, kb_id: str) -> dict[str, str]:
    """doc_id -> rel_path for every file of this knowledge base (deleted ones included: the graph catches up only
    with its next version, and compiled pages still mention them)."""
    with db.connect(settings.state_db) as con:
        rows = con.execute("SELECT file_key, rel_path FROM files WHERE kb_id = ?", (kb_id,)).fetchall()
    return {f"{kb_id}:{r['file_key']}": str(r["rel_path"]) for r in rows}


def resolve_doc(settings: Settings, kb_id: str, *, doc_id: str | None, rel_path: str | None) -> str | None:
    """Turn a document the caller names by rel_path into its doc_id (graph facts and pages record only doc_id).
    When one path has held several files (deleted and put back), the one not deleted and seen most recently is
    taken; not found -> KeyError. Both given and naming different documents -> ValueError."""
    rel = str(rel_path or "").strip().strip("/")
    if not rel:
        return doc_id or None
    with db.connect(settings.state_db) as con:
        row = con.execute("SELECT file_key FROM files WHERE kb_id = ? AND rel_path = ? "
                          "ORDER BY (status = 'deleted'), last_seen_at DESC LIMIT 1", (kb_id, rel)).fetchone()
    if row is None:
        raise KeyError(f"document {rel!r} in {kb_id}")
    found = f"{kb_id}:{row['file_key']}"
    if doc_id and doc_id != found:
        raise ValueError(f"doc_id {doc_id!r} and rel_path {rel!r} name different documents")
    return found


def active_versions(settings: Settings, collection: str) -> list[str]:
    """Every content version of the active chunks in this collection: the literal count uses it to filter out old
    versions the keyword index has not dropped yet."""
    with db.connect(settings.state_db) as con:
        rows = con.execute("SELECT DISTINCT content_version FROM chunks WHERE collection = ? AND status = 'active'", (collection,)).fetchall()
    return sorted(str(r[0]) for r in rows if r[0])


def _squeeze(text: str) -> tuple[str, list[int]]:
    """NFKC-normalise and drop all whitespace, recording where each character sits in the original (a snippet has
    to be cut from the original)."""
    out: list[str] = []
    at: list[int] = []
    for i, ch in enumerate(text):
        for c in unicodedata.normalize("NFKC", ch):
            if not c.isspace():
                out.append(c)
                at.append(i)
    return "".join(out), at


def _norm(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", str(text or "")).split())


def snippet(text: str, phrase: str, *, chars: int = SNIPPET_CHARS) -> tuple[str, bool]:
    """A snippet of about chars characters on each side of the phrase in a field's original text, and whether the
    phrase appears literally in that text (compared after NFKC normalisation and dropping whitespace).
    When there is no literal occurrence a case-insensitive one is looked for (the analyzer ignores case anyway)
    and the snippet is still cut there; failing that, the snippet is the start of the field."""
    raw = str(text or "")
    flat, at = _squeeze(raw)
    needle = _norm(phrase)
    pos = flat.find(needle) if needle else -1
    literal, size = pos >= 0, len(needle)
    if pos < 0 and needle:
        folded, size = flat.casefold(), len(needle.casefold())
        pos = folded.find(needle.casefold()) if len(folded) == len(flat) else -1      # a few characters grow when lowered; then positions do not line up, so no search
    if pos >= 0:
        start, end = at[pos], at[pos + size - 1] + 1
    else:
        start = end = 0
    lo, hi = max(0, start - chars), min(len(raw), end + chars)
    body = " ".join(raw[lo:hi].split())
    return ("…" if lo > 0 else "") + body + ("…" if hi < len(raw) else ""), literal


def grep_query(phrase: str, fields: Iterable[str], filters: list[dict[str, Any]], *, limit: int) -> dict[str, Any]:
    """The query body of one phrase: one match_phrase per field (with _name, so a hit shows which field matched),
    scope restrictions as filters; hits sorted by rel_path and chunk_index ascending (stable and reproducible, not
    by score), aggregated by doc_id to count each document's chunks."""
    should = [{"match_phrase": {f: {"query": phrase, "_name": f}}} for f in fields]
    return {
        "size": int(limit), "track_total_hits": True,
        "query": {"bool": {"filter": filters, "should": should, "minimum_should_match": 1}},
        "sort": [{"rel_path": {"order": "asc"}}, {"chunk_index": {"order": "asc", "missing": "_last"}}],
        "_source": GREP_SOURCE,
        "aggs": {"by_doc": {"terms": {"field": "doc_id", "size": GREP_DOC_BUCKETS},
                            "aggs": {"path": {"top_hits": {"size": 1, "_source": ["rel_path"]}}}}},
    }


def _hit_row(hit: dict[str, Any], phrase: str, fields: tuple[str, ...]) -> dict[str, Any]:
    src = dict(hit.get("_source") or {})
    matched = [str(m) for m in hit.get("matched_queries") or [] if str(m) in fields]
    # the field reported: a field with a literal occurrence first, then a field the analyzer matched; the snippet
    # is cut from that field's original text
    cuts = {f: snippet(str(src.get(f) or ""), phrase) for f in fields if src.get(f)}
    field = next((f for f in fields if f in cuts and cuts[f][1]), None) or next((f for f in matched if f in cuts), None) \
        or next((f for f in fields if f in cuts), fields[0])
    text, _ = cuts.get(field, ("", False))
    row: dict[str, Any] = {"doc_id": src.get("doc_id"), "rel_path": src.get("rel_path"), "chunk_index": src.get("chunk_index"),
                           "block_type": src.get("block_type")}
    for key in ("page_idx", "sheet_name", "row_start", "row_end"):
        if src.get(key) is not None:
            row[key] = src[key]
    row.update({"point_id": str(hit.get("_id")), "field": field, "snippet": text, "literal": any(c[1] for c in cuts.values())})
    return row


def _total(resp: dict[str, Any]) -> int:
    total = (resp.get("hits") or {}).get("total")
    return int(total.get("value") or 0) if isinstance(total, dict) else int(total or 0)


def grep(settings: Settings, source: KBSource, phrases: list[str], *, fields: list[str] | None = None, rel_paths: list[str] | None = None,
         doc_ids: list[str] | None = None, limit: int = 30, client: Any = None, timeout: float | None = None) -> dict[str, Any]:
    """Literal phrase counts: for each phrase, in how many chunks and which documents it appears (the analyzer's
    phrase match), plus the first limit hits (sorted by path and chunk index).
    The cjk analyzer turns CJK text into bigrams, so phrase matching is approximate (a phrase also matches the same
    characters split by whitespace, and even other text whose bigrams happen to sit next to each other); every hit
    returned is therefore checked once more for a literal occurrence (literal). The totals and document counts
    follow the analyzer and are not checked one by one.
    A missing index (a knowledge base just enrolled, nothing written to the keyword index yet) counts as zero hits
    with one degraded entry; backend errors are left to the caller to report as 503."""
    from .channels import os_client

    fields_t = tuple(f for f in GREP_FIELDS if f in set(fields or GREP_FIELDS))
    phrases = list(dict.fromkeys(str(p).strip() for p in phrases if str(p).strip()))
    scope = {"rel_paths": [str(x) for x in dict.fromkeys(rel_paths or []) if str(x).strip()],
             "doc_ids": [str(x) for x in dict.fromkeys(doc_ids or []) if str(x).strip()]}
    filters: list[dict[str, Any]] = [{"terms": {"content_version": active_versions(settings, source.collection)}}]
    if scope["doc_ids"]:
        filters.append({"terms": {"doc_id": scope["doc_ids"]}})
    if scope["rel_paths"]:
        filters.append({"terms": {"rel_path": scope["rel_paths"]}})
    client = client if client is not None else os_client(settings.opensearch_url)
    out: dict[str, Any] = {"kb_id": source.kb_id, "kb_name": kb_name(source), "fields": list(fields_t),
                           "scope": {k: len(v) for k, v in scope.items()}, "results": [], "note": f"{GREP_NOTE}. {GREP_STALE_NOTE}"}
    collection = source.collection
    if not client.indices.exists(index=collection, request_timeout=timeout):
        out["results"] = [{"phrase": p, "total_chunks": 0, "docs": [], "hits": [], "literal_checked": 0, "literal_true": 0} for p in phrases]
        out["degraded"] = [f"{source.kb_id}: keyword index {collection} does not exist"]
        return out
    body: list[dict[str, Any]] = []
    for p in phrases:
        body.append({"index": collection})
        body.append(grep_query(p, fields_t, filters, limit=limit))
    responses = client.msearch(body=body, request_timeout=timeout).get("responses") or []
    if len(responses) != len(phrases):
        raise RuntimeError(f"opensearch msearch returned {len(responses)} responses for {len(phrases)} phrases")
    for p, resp in zip(phrases, responses):
        if resp.get("error"):
            err = resp["error"]
            raise RuntimeError(f"opensearch: {(err.get('type') if isinstance(err, dict) else err) or 'error'}")
        agg = (resp.get("aggregations") or {}).get("by_doc") or {}
        docs = []
        for b in agg.get("buckets") or []:
            top = (((b.get("path") or {}).get("hits") or {}).get("hits") or [{}])[0]
            docs.append({"doc_id": b.get("key"), "rel_path": (top.get("_source") or {}).get("rel_path"), "chunks": int(b.get("doc_count") or 0)})
        docs.sort(key=lambda d: (-d["chunks"], str(d.get("rel_path") or "")))
        hits = [_hit_row(h, p, fields_t) for h in (resp.get("hits") or {}).get("hits") or []]
        result: dict[str, Any] = {"phrase": p, "total_chunks": _total(resp), "docs": docs, "hits": hits,
                                  "literal_checked": len(hits), "literal_true": sum(1 for h in hits if h["literal"])}
        if int(agg.get("sum_other_doc_count") or 0):
            result["docs_truncated"] = True        # more documents matched than GREP_DOC_BUCKETS: docs is incomplete, total_chunks is still the full count
        out["results"].append(result)
    return out
