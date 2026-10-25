"""模式版本的自动流转(2026-09-08 晚,用户要求管线全自动、不用人看):

- 建图结束把端点账(merge.endpoint_observation)写回现行模式版本 —— 谁提的约束、数据实际怎么用的,记在一起;
- 下一次归并先看账:账里确认过的端点组合直接放行,不必每一版重新凑够阈值;
- 空白图:建图时没有任何模式版本,就用默认参数(本库采样量、tune 槽的模型,没选就是 DEFAULT_GRAPH_LLM)先抽一版
  标签并直接生效,再建图;
- 阈值触发的整库重建:先以现行版本和账为底重抽一版(schema.suggest 的 prior),账里确认过的端点由规则保住
  (schema.apply_observed_guard),自动生效,再建图。

版本条目:id / created_at / model / 采样口径 / domain / persona / linkml(模式本体,ontology.to_linkml 的产物)/
examples(语料示例,不进 LinkML)/ origin(manual / auto_blank / auto_rebuild / upload)/ prior_id(以哪一版为底)/
observed(端点账,建图后写入)。控制台「立即/重新抽取标签」走的也是这里的 suggest_schema_version,只是
adopt=False(仍由人选中并保存);上传 LinkML 走 register_uploaded_schema。
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from typing import Any

from .. import db, discovery
from ..config import bool_value
from ..limits import apply_schema_version, normalize_examples, push_schema_version
from ..models import KBSource
from ..ontology import SchemaError, entry_model, from_linkml, load_linkml, to_linkml

ORIGIN_MANUAL = "manual"
ORIGIN_BLANK = "auto_blank"
ORIGIN_REBUILD = "auto_rebuild"
ORIGIN_UPLOAD = "upload"
SUGGEST_TIMEOUT_SECONDS = 600
# The "label extraction in progress" mark (state database app_config, key schema_suggest:<kb_id>): both the
# console's manual extraction and the automatic one before a build set it, and whichever side sees it waits for
# the other to finish / refuses to extract again, so the two sides do not each produce a version (2026-09-09
# 02:09 kb_001: the timer and the user's click were 20 seconds apart and yielded two identical versions).
# A killed process leaves the mark in the database: older than SUGGEST_MARK_FRESH_SECONDS counts as expired.
SUGGEST_MARK_PREFIX = "schema_suggest:"
SUGGEST_TASK_PREFIX = "schema_task:"  # Latest console task only; no queue/history service.
SUGGEST_MARK_FRESH_SECONDS = SUGGEST_TIMEOUT_SECONDS + 120
SUGGEST_WAIT_POLL_SECONDS = 3.0
SUGGEST_RENEW_SECONDS = 60.0          # renew this often while extracting; slow models / retries can outlast the claim window (Codex re-review N08)
SUGGEST_BUSY_MESSAGE = "A label extraction is already running (the automatic one before the build, or another window); wait for it to finish, then check the label versions"
SUGGEST_SUPERSEDED_MESSAGE = "This label extraction ran past its claim window and another extraction took over; its result was not published. Check the latest label version"


class SuggestBusy(ValueError):
    """Claiming the mark failed: another place is extracting."""


def load_config(settings: Any, kb_id: str) -> dict[str, Any]:
    with db.connect(settings.state_db) as con:
        return discovery.get_config(con, kb_id)


def active_schema_entry(config: dict[str, Any]) -> dict[str, Any] | None:
    """The entry of the active schema version (the one graph_schema_active in config points to); None if there is
    none."""
    active = str(config.get("graph_schema_active") or "")
    if not active:
        return None
    for v in config.get("graph_schema_versions") or []:
        if str(v.get("id")) == active:
            return dict(v)
    return None


def active_schema_entry_for(settings: Any, kb_id: str) -> dict[str, Any] | None:
    """Same as above but read from the state database; unregistered KBs (extra sources, test stubs) count as having
    no version."""
    try:
        return active_schema_entry(load_config(settings, kb_id))
    except Exception:
        return None


def schema_missing(config: dict[str, Any]) -> bool:
    """没有生效的模式版本:建图会退到兜底类型表 —— 这就是「空白图」。"""
    return not config.get("graph_schema_active")


def suggest_in_progress(settings: Any, kb_id: str) -> dict[str, Any] | None:
    """Return the mark when another place is extracting labels for this KB (the console, or the automatic extraction
    before a build); None when there is none or it has expired."""
    with db.connect(settings.state_db) as con:
        mark = db.get_app_config(con, SUGGEST_MARK_PREFIX + kb_id)
    if not isinstance(mark, dict):
        return None
    started = float(mark.get("started_at") or 0)
    if time.time() - started > SUGGEST_MARK_FRESH_SECONDS:
        return None
    return mark


def claim_suggest(settings: Any, kb_id: str, origin: str) -> str | None:
    """Claim the "extracting" mark: one conditional upsert that only succeeds when there is no mark or the mark has
    expired (a single SQLite statement, atomic), so of two simultaneous claimants only one wins. Returns the owner
    token; None when the claim fails. (Codex review F06: read-then-write is not a claim, and whichever side
    finished first would also clear the later claimant's mark.)"""
    token = uuid.uuid4().hex
    now = int(time.time())
    mark = json.dumps({"origin": origin, "started_at": now, "created_at": now, "token": token,
                       "phase": "sampling"}, ensure_ascii=False)
    with db.connect(settings.state_db) as con:
        cur = con.execute(
            "INSERT INTO app_config(key, value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at "
            "WHERE app_config.updated_at < ?",
            (SUGGEST_MARK_PREFIX + kb_id, mark, now, now - SUGGEST_MARK_FRESH_SECONDS),
        )
        claimed = cur.rowcount == 1
        con.commit()
    return token if claimed else None


def release_suggest(settings: Any, kb_id: str, token: str | None) -> bool:
    """Release only the mark we claimed ourselves (deleted only when the token matches); anyone else's mark is left
    alone."""
    if not token:
        return False
    with db.connect(settings.state_db) as con:
        try:
            cur = con.execute("DELETE FROM app_config WHERE key = ? AND json_extract(value, '$.token') = ?",
                              (SUGGEST_MARK_PREFIX + kb_id, token))
            removed = cur.rowcount == 1
        except Exception:
            mark = db.get_app_config(con, SUGGEST_MARK_PREFIX + kb_id)
            removed = isinstance(mark, dict) and mark.get("token") == token
            if removed:
                con.execute("DELETE FROM app_config WHERE key = ?", (SUGGEST_MARK_PREFIX + kb_id,))
        con.commit()
    return removed


def renew_suggest(settings: Any, kb_id: str, token: str | None) -> bool:
    """Renew the mark we claimed (refreshes started_at together with updated_at); does nothing when the token does
    not match."""
    if not token:
        return False
    now = int(time.time())
    key = SUGGEST_MARK_PREFIX + kb_id
    with db.connect(settings.state_db) as con:
        try:
            cur = con.execute("UPDATE app_config SET value = json_set(value, '$.started_at', ?), updated_at = ? "
                              "WHERE key = ? AND json_extract(value, '$.token') = ?", (now, now, key, token))
            renewed = cur.rowcount == 1
        except Exception:
            mark = db.get_app_config(con, key)
            renewed = isinstance(mark, dict) and mark.get("token") == token
            if renewed:
                mark["started_at"] = now
                con.execute("UPDATE app_config SET value = ?, updated_at = ? WHERE key = ?",
                            (json.dumps(mark, ensure_ascii=False), now, key))
        con.commit()
    return renewed


def suggest_owner(settings: Any, kb_id: str) -> str | None:
    """Who owns the current (unexpired) mark: its token; None when there is no mark."""
    mark = suggest_in_progress(settings, kb_id)
    return str(mark.get("token")) if isinstance(mark, dict) and mark.get("token") else None


def wait_for_suggest(settings: Any, kb_id: str, *, timeout: float | None = None,
                     stop: threading.Event | None = None) -> float:
    """If another place is extracting, wait until it finishes (the mark disappears or expires); returns the seconds
    waited."""
    started = time.time()
    # A renewed lease is live work, not a reason to time out after 12 minutes.
    while suggest_in_progress(settings, kb_id) is not None:
        if timeout is not None and time.time() - started >= timeout:
            break
        if stop is not None and stop.wait(SUGGEST_WAIT_POLL_SECONDS):
            from .llm import LLMInterrupted
            raise LLMInterrupted("stop requested")
        if stop is None:
            time.sleep(SUGGEST_WAIT_POLL_SECONDS)
    return round(time.time() - started, 1)


def update_suggest_progress(settings: Any, kb_id: str, token: str, *, phase: str,
                            sampled: int | None = None) -> bool:
    now = int(time.time())
    with db.connect(settings.state_db) as con:
        cur = con.execute("UPDATE app_config SET value=json_set(value,'$.phase',?,'$.sampled',"
                          "coalesce(?,json_extract(value,'$.sampled')),'$.started_at',?),updated_at=? "
                          "WHERE key=? AND json_extract(value,'$.token')=?",
                          (phase, sampled, now, now, SUGGEST_MARK_PREFIX + kb_id, token))
        return cur.rowcount == 1


def discard_suggest(con: Any, kb_id: str) -> None:
    """Invalidate publication before graph deletion; late workers cannot recreate the task."""
    con.execute("DELETE FROM app_config WHERE key IN (?,?)",
                (SUGGEST_MARK_PREFIX + kb_id, SUGGEST_TASK_PREFIX + kb_id))


def adopt_latest_schema_version(settings: Any, source: KBSource) -> dict[str, Any] | None:
    """If the ring holds an extracted but never adopted version (extracted in the console, never saved), adopt the
    latest one instead of extracting another; returns None when there is none."""
    with db.connect(settings.state_db) as con:
        stored = discovery.get_config(con, source.kb_id)
        versions = [dict(v) for v in (stored.get("graph_schema_versions") or [])]
        if not versions or stored.get("graph_schema_active"):
            return None
        entry = versions[0]                       # newest first in the ring
        updates: dict[str, Any] = {"graph_schema_versions": versions, "graph_schema_active": str(entry["id"])}
        apply_schema_version({"graph_schema_versions": versions}, updates)
        discovery.set_config(con, source.kb_id, updates)
        con.commit()
    counts = entry_counts(entry)
    print(f"[schema] {source.kb_id}({source.source_root}) adopt existing: version {entry['id']} ({entry.get('origin')})"
          f" types={counts['types']} predicates={counts['predicates']}", flush=True)
    return {"version_id": str(entry["id"]), "origin": entry.get("origin"), "adopted": True, "adopted_existing": True,
            "entity_types": counts["type_names"], "predicates": counts["predicate_names"]}


def entry_counts(entry: dict[str, Any] | None) -> dict[str, Any]:
    """版本条目里模式的规模(日志、控制台提示用)。"""
    model = entry_model(entry) or {}
    types = [t["name"] for t in model.get("types") or [] if not t.get("abstract")]
    preds = [p["name"] for p in model.get("predicates") or []]
    return {"types": len(types), "predicates": len(preds), "type_names": types, "predicate_names": preds}


def prior_from_entry(entry: dict[str, Any] | None) -> dict[str, Any] | None:
    """现行版本 → 归纳提示词用的 prior(类型表、父类、谓词、定义、画像、端点账)。"""
    model = entry_model(entry)
    if not entry or model is None:
        return None
    types = [t for t in model.get("types") or [] if not t.get("abstract")]
    return {
        "id": entry.get("id"), "domain": entry.get("domain"), "language": model.get("language"),
        "entity_types": [t["name"] for t in types],
        "parent_types": {t["name"]: t["is_a"] for t in types},
        "predicates": [dict(p) for p in model.get("predicates") or [] if not p.get("core")],
        "type_definitions": {t["name"]: t["description"] for t in types if t.get("description")},
        "profile": dict(model.get("profile") or {}),
        "observed": entry.get("observed"),
    }


def reload_source(settings: Any, source: KBSource) -> KBSource:
    """Reassemble the KBSource from the configuration now in the state database (after a new version was adopted
    automatically, the build must see the new labels / predicates)."""
    config = load_config(settings, source.kb_id)
    return discovery.build_source(settings.mirror_root, source.source_root, config,
                                  kb_id=source.kb_id, collection=source.collection)


def protected_schema_ids(con, kb_id):
    protected = set()
    for row in con.execute("SELECT manifest_json FROM graph_builds WHERE kb_id=? AND status IN ('done','running','failed','cancelled')", (kb_id,)):
        try:
            manifest = json.loads(row[0] or "{}")
        except ValueError:
            continue
        schema_id = manifest.get("schema_version_id") or (manifest.get("schema") or {}).get("version_id")
        if schema_id:
            protected.add(str(schema_id))
    return protected


def record_schema_observation(con, kb_id: str, *, version_id: str, observed: dict[str, Any],
                              graph_version: str) -> bool:
    """At the end of a build: write the endpoint ledger into the entry of version version_id (observed). Nothing is
    written when the version is no longer in the ring (evicted, legacy labels)."""
    config = discovery.get_config(con, kb_id)
    versions = [dict(v) for v in (config.get("graph_schema_versions") or [])]
    for v in versions:
        if str(v.get("id")) == str(version_id):
            v["observed"] = {**dict(observed or {}), "graph_version": graph_version, "at": int(time.time())}
            discovery.set_config(con, kb_id, {"graph_schema_versions": versions})
            con.commit()
            return True
    return False


def sample_docs(settings: Any, source: KBSource, *, q: Any = None, spec: Any = None) -> dict[str, Any]:
    """The sample for label extraction: active chunk ledger -> payloads -> stratified + farthest-point sampling
    (schema.sample_chunks). Returns the sample and how it was drawn."""
    from ..vector.layout import TEXT_VECTOR
    from ..vector.qdrant import client as qdrant_client
    from . import schema as graph_schema
    from .units import fetch_chunk_payloads
    from .build import tune_llm_spec
    from .sampling_budget import sampling_budget

    budget = sampling_budget(spec or tune_llm_spec(settings, source))

    with db.connect(settings.state_db) as con:
        ledger = db.active_chunk_refs(con, source.collection)
    if not ledger:
        raise ValueError("This knowledge base has no active chunks yet; finish parsing before extracting labels")
    q = q or qdrant_client(settings.qdrant_url, settings.qdrant_api_key)
    payloads = fetch_chunk_payloads(q, source.collection, ledger)
    chunks = [{"doc_id": c.doc_id, "chunk_index": c.chunk_index, "chunk_uid": c.chunk_uid, "point_id": c.point_id,
               "text": c.text, "section": " > ".join(c.section_path)} for c in payloads]

    def fetch_vectors(pool: list[dict[str, Any]]) -> dict[str, list[float]]:
        ids = [str(c["point_id"]) for c in pool if c.get("point_id")]
        out: dict[str, list[float]] = {}
        for i in range(0, len(ids), 128):
            for p in q.retrieve(collection_name=source.collection, ids=ids[i:i + 128],
                                with_vectors=[TEXT_VECTOR], with_payload=False):
                vec = p.vector.get(TEXT_VECTOR) if isinstance(p.vector, dict) else p.vector
                if vec:
                    out[str(p.id)] = list(vec)
        return out

    sample = graph_schema.sample_chunks(chunks, 0, budget=budget["sample_token_budget"], fetch_vectors=fetch_vectors)
    return {
        "texts": sample["texts"], "truncated": sample["truncated"],
        "chunks_total": len(chunks), "documents": len({c.doc_id for c in payloads}),
        "sample": {**budget, **{k: sample.get(k) for k in ("pool", "documents_covered", "documents_total",
                   "excluded_boilerplate", "method", "algorithm", "estimated_tokens", "stop_reason")},
                   "sampled": len(sample["texts"]),
                   "chunk_ids": [c.get("chunk_uid") for c in sample["picked"]]},
    }


def suggest_schema_version(settings: Any, source: KBSource, *, origin: str, adopt: bool, use_prior: bool = True,
                           client: Any = None, docs: list[str] | None = None, sample: dict[str, Any] | None = None,
                           spec: Any = None, stop: threading.Event | None = None,
                           claimed_token: str | None = None) -> dict[str, Any]:
    """抽一版标签入环;adopt=True 时直接生效(标签 / 语言 / 谓词 / 父类 / 定义 / 示例 / 画像一起展开成生效值)。
    use_prior=True 且有现行版本时,模型在现行版本 + 端点账的基础上修订。
    client / docs 是给测试和调用方注入用的;不给就按本库配置取 tune 模型、采样。
    stop:建图的停止事件,置位后模型调用不再重试、不再发起。"""
    from . import schema as graph_schema

    config = load_config(settings, source.kb_id)
    entry_prior = active_schema_entry(config) if use_prior else None
    prior = prior_from_entry(entry_prior)
    token = claimed_token or claim_suggest(settings, source.kb_id, origin)
    if token is None:
        raise SuggestBusy(SUGGEST_BUSY_MESSAGE)
    from .llm import LLMInterrupted
    stop = stop or threading.Event()
    close = None
    started = time.time()
    stop_renewing = threading.Event()

    def keep_claim() -> None:
        while not stop_renewing.wait(SUGGEST_RENEW_SECONDS):
            try:
                if not renew_suggest(settings, source.kb_id, token):
                    stop.set()
                    return
            except Exception:
                pass  # Publication still checks ownership under the SQLite write lock.

    def progress(phase: str, sampled: int | None = None) -> None:
        if stop.is_set():
            raise LLMInterrupted("stop requested")
        if not update_suggest_progress(settings, source.kb_id, token, phase=phase, sampled=sampled):
            raise LLMInterrupted("label extraction was cancelled or superseded")

    renewer = threading.Thread(target=keep_claim, name=f"schema-claim-{source.kb_id}", daemon=True)
    renewer.start()  # Covers payload loading and sample selection too.
    try:
        progress("sampling")
        if client is None:
            from .build import tune_llm_spec
            spec = spec or tune_llm_spec(settings, source)
        if docs is None:
            sample = sample_docs(settings, source, spec=spec)
            docs = list(sample["texts"])
            if not docs:
                raise ValueError("The active chunks have no text to sample from")
        sample = dict(sample or {})
        progress("domain", len(docs))
        if client is None:
            from .build import graph_paths, tune_llm_spec
            from .llm import ChatClient, LLMCache, env_int
            from .sampling_budget import sampling_budget

            spec = spec or tune_llm_spec(settings, source)
            cache = LLMCache(graph_paths(settings, source).cache_file)
            close = cache.close
            client = ChatClient(spec, cache=cache, timeout=max(1, env_int("KB_SCHEMA_LLM_TIMEOUT", 900)),
                                connect_timeout=10, attempts=3,
                                workers=1, circuit_fails=0, stop=stop,
                                input_token_limit=sampling_budget(spec)["input_token_limit"])
        result = graph_schema.suggest(client, docs, prior=prior, progress=progress)
        progress("saving")
        model = result.get("model") or {}
        if not model.get("types"):
            raise ValueError("The model induced no entity types; this version was not added")
        persona = str(result.pop("persona", "") or "").strip()
        model_name = getattr(spec, "name", None) or str((config.get("graph_llm") or {}).get("tune") or "") or None
        version_id = new_version_id(config)
        domain = str(result.get("domain") or "").strip() or None
        entry = {
            "id": version_id,
            "created_at": int(time.time()),
            "model": model_name,
            "sample_size": len(docs),
            "sampling": sample.get("sample") or {},
            "sampled": len(docs),
            "chunks_total": sample.get("chunks_total"),
            "documents": sample.get("documents"),
            "domain": domain,
            "persona": persona or None,
            "linkml": to_linkml(model, name=source.kb_id, version=version_id, description=domain or ""),
            "examples": normalize_examples(result.get("examples")),
            "example_stats": result.get("example_stats") or {},
            "origin": origin,
            "prior_id": str(entry_prior.get("id")) if entry_prior else None,
            "guard": result.get("guard") or {},
        }
        if entry_prior and entry_prior.get("observed"):
            # the ledger is inherited by the new version: the merge in its first build can already let through the
            # pairs confirmed by the previous version (overwritten by this version's own ledger once the build ends)
            entry["observed"] = {**dict(entry_prior["observed"]), "inherited_from": str(entry_prior.get("id"))}
        if suggest_owner(settings, source.kb_id) != token:
            # the mark was not renewed (hung process, clock jump) and someone else has claimed it: the result is only
            # logged, not published -- the token protects publishing, not just the release (Codex re-review N08)
            print(f"[schema] {source.kb_id}({source.source_root}) {origin}: claim lost before publishing, result discarded "
                  f"types={len(model['types'])} predicates={len(model['predicates'])}", flush=True)
            raise SuggestBusy(SUGGEST_SUPERSEDED_MESSAGE)
        counts = entry_counts(entry)
        result.update({"kb_id": source.kb_id, "sampled": len(docs), "documents": sample.get("documents"),
                       "chunks_total": sample.get("chunks_total"), "truncated": bool(sample.get("truncated")),
                       "sample": sample.get("sample") or {}, "version_id": entry["id"], "adopted": bool(adopt),
                       "origin": origin, "prior_id": entry["prior_id"], "seconds": round(time.time() - started, 1),
                       "entity_types": counts["type_names"], "predicates": counts["predicate_names"],
                       "llm_stats": dict(getattr(client, "stats", {}) or {})})
        result.pop("model", None)
        with db.connect(settings.state_db) as con:
            con.execute("BEGIN IMMEDIATE")
            mark = db.get_app_config(con, SUGGEST_MARK_PREFIX + source.kb_id)
            if not isinstance(mark, dict) or mark.get("token") != token:
                raise SuggestBusy(SUGGEST_SUPERSEDED_MESSAGE)
            stored = discovery.get_config(con, source.kb_id)
            versions = push_schema_version(stored.get("graph_schema_versions") or [], entry,
                                           active_id=stored.get("graph_schema_active"), protected_ids=protected_schema_ids(con, source.kb_id))
            updates: dict[str, Any] = {"graph_schema_versions": versions}
            if adopt:
                updates["graph_schema_active"] = entry["id"]
                apply_schema_version({"graph_schema_versions": versions}, updates)
            discovery.set_config(con, source.kb_id, updates)
            task = db.get_app_config(con, SUGGEST_TASK_PREFIX + source.kb_id)
            if isinstance(task, dict) and task.get("task_id") == token and task.get("status") == "running":
                task.update(status="completed", finished_at=int(time.time()), result={**result,
                            "sample": {k: v for k, v in result["sample"].items() if k != "chunk_ids"}})
                db.set_app_config(con, SUGGEST_TASK_PREFIX + source.kb_id, task)
            con.commit()
        print(f"[schema] {source.kb_id}({source.source_root}) {origin}: version {entry['id']}"
              f"{' adopted' if adopt else ''} types={counts['types']} predicates={counts['predicates']}"
              f" prior={entry['prior_id']} guard={entry['guard']} seconds={result['seconds']}", flush=True)
        return result

    finally:
        stop_renewing.set()
        renewer.join(timeout=5)
        if close is not None:
            close()
        release_suggest(settings, source.kb_id, token)      # 版本已落库(或失败)才释放,不留「看着空闲其实还没发表」的窗口

def new_version_id(config: dict[str, Any]) -> str:
    taken = {str(v.get("id")) for v in (config.get("graph_schema_versions") or [])}
    version_id = f"v{int(time.time())}"
    while version_id in taken:          # 同一秒内入环两版(测试、脚本)不能撞 id:撞了会把上一条顶掉
        version_id += "x"
    return version_id


def register_uploaded_schema(settings: Any, source: KBSource, text: str | bytes, *, adopt: bool = False) -> dict[str, Any]:
    """人工上传的 LinkML(插件审定的本体、手写的模式)入环。校验不过抛 SchemaError(带全部错误);
    读取时丢掉的部分(属性型 slot、enums 之类)作为警告返回。语料示例不在 LinkML 里,上传的版本没有示例,
    抽取用通用示例。端点账沿用现行版本的:同名谓词上一版确认过的端点组合照样放行。"""
    doc = load_linkml(text)
    model, warnings = from_linkml(doc)
    if not model.get("types"):
        raise SchemaError(["模式里没有可抽取的类型"])
    with db.connect(settings.state_db) as con:
        stored = discovery.get_config(con, source.kb_id)
        prior = active_schema_entry(stored)
        version_id = new_version_id(stored)
        entry = {
            "id": version_id,
            "created_at": int(time.time()),
            "model": None,
            "domain": str(doc.get("description") or "").strip()[:200] or (prior or {}).get("domain"),
            "persona": (prior or {}).get("persona"),
            "linkml": to_linkml(model, name=source.kb_id, version=version_id,
                                title=str(doc.get("title") or ""), description=str(doc.get("description") or "")),
            "examples": "",
            "origin": ORIGIN_UPLOAD,
            "prior_id": str(prior.get("id")) if prior else None,
            "upload": {"name": str(doc.get("name") or ""), "id": str(doc.get("id") or ""),
                       "version": str(doc.get("version") or ""), "warnings": warnings[:50]},
        }
        if prior and prior.get("observed"):
            entry["observed"] = {**dict(prior["observed"]), "inherited_from": str(prior.get("id"))}
        versions = push_schema_version(stored.get("graph_schema_versions") or [], entry,
                                       active_id=stored.get("graph_schema_active"), protected_ids=protected_schema_ids(con, source.kb_id))
        updates: dict[str, Any] = {"graph_schema_versions": versions}
        if adopt:
            updates["graph_schema_active"] = version_id
            apply_schema_version({"graph_schema_versions": versions}, updates)
        discovery.set_config(con, source.kb_id, updates)
        con.commit()
    counts = entry_counts(entry)
    print(f"[schema] {source.kb_id}({source.source_root}) upload: version {version_id}{' adopted' if adopt else ''}"
          f" types={counts['types']} predicates={counts['predicates']} warnings={len(warnings)}", flush=True)
    return {"kb_id": source.kb_id, "version_id": version_id, "adopted": bool(adopt), "origin": ORIGIN_UPLOAD,
            "warnings": warnings, "entity_types": counts["type_names"], "predicates": counts["predicate_names"]}


def ensure_schema_before_build(settings: Any, source: KBSource, *,
                               stop: threading.Event | None = None) -> tuple[KBSource, dict[str, Any] | None]:
    """Before a full graph build: with no schema version at all, extract one with the default parameters, adopt it
    and return the source reassembled from the new configuration. With a version, return it unchanged."""
    try:
        config = load_config(settings, source.kb_id)
    except KeyError:
        return source, None
    if not schema_missing(config):
        return source, None
    # the console is extracting (the user just clicked "extract labels"): wait for it and adopt that version
    # rather than extracting another
    mark = suggest_in_progress(settings, source.kb_id)
    if mark is not None:
        print(f"[graph] {source.kb_id}({source.source_root}): 另一处正在抽标签({mark.get('origin')}),等它抽完", flush=True)
        waited = wait_for_suggest(settings, source.kb_id, stop=stop)
        print(f"[graph] {source.kb_id}({source.source_root}): 等了 {waited}s", flush=True)
    # 环里有抽过但没保存的版本(控制台抽了没点保存,或上面刚等到的那版):采用最新那版
    info = adopt_latest_schema_version(settings, source)
    if info is not None:
        return reload_source(settings, source), info
    print(f"[graph] {source.kb_id}({source.source_root}): 没有模式版本,先用默认参数抽一版标签"
          "(按模型输入预算自动采样)", flush=True)
    try:
        info = suggest_schema_version(settings, source, origin=ORIGIN_BLANK, adopt=True, use_prior=False, stop=stop)
    except SuggestBusy:
        # 等待和认领之间有人抢先开抽(控制台):再等它抽完,采用它那版
        waited = wait_for_suggest(settings, source.kb_id, stop=stop)
        info = adopt_latest_schema_version(settings, source)
        if info is None:
            raise
        print(f"[graph] {source.kb_id}({source.source_root}): lost the claim; adopted the other run's version after {waited}s", flush=True)
    return reload_source(settings, source), info


RESUGGEST_MARK_PREFIX = "graph_resuggest_baseline:"


def resuggest_for_rebuild(settings: Any, source: KBSource, *, baseline_version: str | None = None) -> tuple[KBSource, dict[str, Any]]:
    """Before a threshold-triggered full rebuild: re-extract a version on the basis of the active version and its
    ledger, and adopt it. Skipped when graph_rebuild_resuggest is off; a KB without any version is left to the
    build itself (ensure_schema_before_build).
    ``baseline_version``: the full build this rebuild is going to replace. One baseline is re-suggested only
    once: after a failed full build the baseline is unchanged and the rebuild is due again in the next round
    (2 hours later); re-suggesting every round changes the fingerprint every round, invalidates the whole
    extraction cache so the entire corpus goes to the model again, and soon pushes the versions saved by hand
    out of the version ring (2026-09-29 audit)."""
    try:
        config = load_config(settings, source.kb_id)
    except KeyError:
        return source, {"skipped": "unknown_kb"}
    if not bool_value(config.get("graph_rebuild_resuggest"), True):
        return source, {"skipped": "disabled"}
    if schema_missing(config):
        return source, {"skipped": "no_schema"}
    mark_key = RESUGGEST_MARK_PREFIX + source.kb_id
    if baseline_version:
        try:
            with db.connect(settings.state_db) as con:
                mark = db.get_app_config(con, mark_key) or {}
        except Exception:
            mark = {}
        if str(mark.get("baseline") or "") == str(baseline_version):
            return source, {"skipped": "already_resuggested_for_baseline", "baseline": str(baseline_version),
                            "version_id": mark.get("version_id")}
    info = suggest_schema_version(settings, source, origin=ORIGIN_REBUILD, adopt=True, use_prior=True)
    if baseline_version:
        try:
            with db.connect(settings.state_db) as con:
                db.set_app_config(con, mark_key, {"baseline": str(baseline_version), "version_id": info.get("version_id"),
                                                  "at": int(time.time())})
        except Exception as exc:
            print(f"[schema] could not record the re-suggestion mark for {source.kb_id}: {exc!r}", flush=True)
    return reload_source(settings, source), {k: info.get(k) for k in ("version_id", "prior_id", "origin", "guard", "seconds",
                                                                       "entity_types", "predicates")}
