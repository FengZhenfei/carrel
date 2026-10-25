"""检索时的文档状态与文档日期(破坏性改版,方案附录:文档状态与时间口径)。

· 状态:这个库的更正文件(config/corrections/<kb_id>.yaml)里标了被取代 / 归档 / 有效期的文档,按时间口径(currency.query_scope)
  决定给不给 —— 现行口径不给被取代、归档、过了有效期的;回看某一天按有效期判;history 全给,非现行的标出来。
  筛选在召回时就做(向量路、关键词路的过滤条件里排除这些路径),图路与视觉路的候选在融合时再筛一遍。
  更正文件按修改时间缓存、库的文件清单按 60 秒缓存:改了更正文件,下一次检索就按新的来,不必重启、不必重建。
· 日期:建了图的库用建图时定下的文档日期(文件名 / 正文开头,写在图库的 Document 节点上),没建图的按文件名认。
只读,不调模型。
"""
from __future__ import annotations

import threading
import time
from typing import Any, Iterable

from kb_pipeline import db
from kb_pipeline.config import Settings
from kb_pipeline.graph.corrections import corrections_path, document_in_scope, document_status, load_corrections, status_on
from kb_pipeline.models import KBSource

FILES_TTL = 60.0
EXCLUDED_MAX = 2000         # 一个库最多排除多少条路径(再多说明更正文件写的是整个目录,改按目录写更合适;多出的照样在融合时筛)
_lock = threading.Lock()
_corrections: dict[str, tuple[tuple[int, int], dict[str, Any]]] = {}     # 更正文件路径 → ((mtime_ns, size), 解析结果)
_files: dict[str, tuple[float, list[str]]] = {}                           # kb_id → (读的时刻, 全部未删除文档的 rel_path)


def corrections_of(settings: Settings, source: KBSource) -> dict[str, Any]:
    """这个库的更正文件(解析过的);没有文件就是空的更正。"""
    path = corrections_path(settings, str(getattr(source, "kb_id", "") or ""))
    if path is None:
        return load_corrections(None)
    try:
        st = path.stat()
        key = (int(st.st_mtime_ns), int(st.st_size))
    except OSError:
        return load_corrections(None)
    with _lock:
        hit = _corrections.get(str(path))
        if hit is not None and hit[0] == key:
            return hit[1]
    corr = load_corrections(path)
    with _lock:
        _corrections[str(path)] = (key, corr)
    return corr


def _doc_paths(settings: Settings, kb_id: str) -> list[str]:
    with _lock:
        hit = _files.get(kb_id)
        if hit is not None and time.time() - hit[0] < FILES_TTL:
            return hit[1]
    with db.connect(settings.state_db) as con:
        rows = con.execute("SELECT rel_path FROM files WHERE kb_id = ? AND status != 'deleted'", (kb_id,)).fetchall()
    paths = sorted({str(r[0]) for r in rows if r[0]})
    with _lock:
        _files[kb_id] = (time.time(), paths)
    return paths


def excluded_paths(settings: Settings, source: KBSource, scope: dict[str, Any]) -> list[str]:
    """按口径不给的文档路径(召回时排除);更正文件里没写文档状态就是空。"""
    if scope.get("mode") == "history":
        return []
    corr = corrections_of(settings, source)
    if not corr.get("documents"):
        return []
    out = [rp for rp in _doc_paths(settings, source.kb_id) if not document_in_scope(document_status(rp, corr), scope)]
    return out[:EXCLUDED_MAX]


def allowed(settings: Settings, source: KBSource, rel_path: Any, scope: dict[str, Any]) -> bool:
    """单条候选按口径给不给(图路、视觉路的候选在融合时用)。"""
    if scope.get("mode") == "history":
        return True
    corr = corrections_of(settings, source)
    if corr.get("errors"):
        return False
    if not corr.get("documents"):
        return True
    if not rel_path:
        return False
    return document_in_scope(document_status(str(rel_path), corr), scope)


def annotate(settings: Settings, rows: Iterable[dict[str, Any]], scope: dict[str, Any]) -> int:
    """来源行标文档状态:今天不是现行的(被取代 / 归档 / 过了有效期)写 doc_status(取代它的文档写 superseded_by);
    回看与 history 时才会出现这样的行。返回标了几条。"""
    marked = 0
    today = time.strftime("%Y-%m-%d")
    for row in rows:
        source = settings.sources.get(str(row.get("kb_id") or ""))
        rel = str(row.get("rel_path") or "")
        if source is None or not rel:
            continue
        corr = corrections_of(settings, source)
        if not corr.get("documents"):
            continue
        item = document_status(rel, corr)
        state = status_on(item, today=today)
        if state != "current":
            row["doc_status"] = state
            if item and item.get("superseded_by"):
                row["superseded_by"] = item["superseded_by"]
            marked += 1
    return marked


def document_dates(settings: Settings, rows: Iterable[dict[str, Any]], graph_versions: dict[str, Any], *,
                   driver: Any = None, timeout: float | None = None) -> dict[tuple[str, str], str]:
    """(kb_id, doc_id) → 文档日期:有图的库查图库的 Document 节点(建图时定的),其余按文件名认;认不出就不给。"""
    from kb_pipeline.graph.temporal import name_date

    rows = list(rows)
    out: dict[tuple[str, str], str] = {}
    by_kb: dict[str, set[str]] = {}
    for r in rows:
        if r.get("kb_id") and r.get("doc_id"):
            by_kb.setdefault(str(r["kb_id"]), set()).add(str(r["doc_id"]))
    if driver is not None:
        from kb_pipeline.graph.recall import TimedSession

        with driver.session() as session:
            if timeout:
                session = TimedSession(session, timeout)
            for kb, ids in by_kb.items():
                gv = graph_versions.get(kb)
                if not gv:
                    continue
                for rec in session.run("MATCH (d:Document {kb_id: $kb, graph_version: $gv}) WHERE d.doc_id IN $ids "
                                       "RETURN d.doc_id AS doc_id, d.date AS date", kb=kb, gv=str(gv), ids=sorted(ids)).data():
                    if rec.get("date"):
                        out[(kb, str(rec["doc_id"]))] = str(rec["date"])
    for r in rows:
        key = (str(r.get("kb_id") or ""), str(r.get("doc_id") or ""))
        if all(key) and key not in out:
            found = name_date(r.get("rel_path") or r.get("doc") or "")
            if found:
                out[key] = found
    return out


def graph_refresh_pending(settings: Settings, source: KBSource) -> bool:
    """Fail closed for current derived answers while a correction/date refresh is pending."""
    import json
    from kb_pipeline.graph.corrections import effective_sha, status_fingerprint

    if corrections_path(settings, source.kb_id) is None:
        return False
    corr = corrections_of(settings, source)
    if corr.get("errors"):
        return True
    with db.connect(settings.state_db) as con:
        row = db.latest_successful_graph_build(con, source.kb_id)
        if row is None:
            return False
        try:
            manifest = json.loads(row["manifest_json"] or "{}")
        except ValueError:
            return True
        if manifest.get("temporal_refresh_on") and manifest["temporal_refresh_on"] <= time.strftime("%Y-%m-%d"):
            return True
        if effective_sha(corr) != str(manifest.get("corrections_sha") or ""):
            return True
        if not corr.get("documents"):
            return False
        paths = [c.get("rel_path", "") for c in db.active_chunk_refs(con, source.collection)]
    return status_fingerprint(corr, paths, today=time.strftime("%Y-%m-%d")) != str(manifest.get("document_status_sha") or "")


def page_allowed(settings, source, page, scope, *, paths=None):
    """Compiled text cannot be partially filtered; require all its source paths in scope."""
    if scope.get("mode") == "as_of" and page.get("compiled_day") != scope.get("day"):
        return False
    if scope.get("mode") == "current" and page.get("time_mode") == "history":
        return False
    rels = list(page.get("source_paths") or page.get("docs") or [])
    ids = page.get("doc_ids") or []
    if paths is not None:
        rels.extend(paths.get(str(d), "") for d in ids)
    corr = corrections_of(settings, source)
    if not corr.get("documents") and not corr.get("errors"):
        return True
    if not rels or any(not p for p in rels):
        return scope.get("mode") == "history"
    return all(allowed(settings, source, p, scope) for p in rels)
