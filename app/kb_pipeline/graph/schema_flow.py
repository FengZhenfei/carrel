"""Automatic flow of schema versions (evening of 2026-09-08; the user wants the pipeline fully automatic, with
nobody watching):

- when a graph build finishes, the endpoint ledger (merge.endpoint_observation) is written back into the active
  schema version -- who declared the constraint and how the data actually uses it are recorded together;
- the next merge consults the ledger first: endpoint pairs confirmed in the ledger pass directly, without
  having to reach the threshold again in every version;
- blank graph: when no schema version exists at build time, a label version is first extracted with the default
  parameters (this KB's sample size, the model in the tune slot, DEFAULT_GRAPH_LLM if none is chosen), adopted
  directly, and then the graph is built;
- threshold-triggered full rebuild: a new version is first extracted on the basis of the active version and its
  ledger (schema.suggest's prior), endpoints confirmed in the ledger are kept by rule
  (schema.apply_observed_guard), it is adopted automatically, and then the graph is built.

Version entries gain three fields: origin (manual / auto_blank / auto_rebuild), prior_id (which version it was
based on) and observed (the endpoint ledger, written after the build). The console's "extract / re-extract
labels now" also goes through suggest_schema_version here, only with adopt=False (a person still selects and
saves it).
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from typing import Any

from .. import db, discovery
from ..config import bool_value
from ..limits import (
    LEGACY_SCHEMA_VERSION_ID, apply_schema_version, normalize_entity_types, normalize_examples,
    normalize_parent_types, normalize_predicates, normalize_profile, normalize_type_definitions,
    push_schema_version,
)
from ..models import KBSource

ORIGIN_MANUAL = "manual"
ORIGIN_BLANK = "auto_blank"
ORIGIN_REBUILD = "auto_rebuild"
SUGGEST_TIMEOUT_SECONDS = 600
# The "label extraction in progress" mark (state database app_config, key schema_suggest:<kb_id>): both the
# console's manual extraction and the automatic one before a build set it, and whichever side sees it waits for
# the other to finish / refuses to extract again, so the two sides do not each produce a version (2026-09-09
# 02:09 kb_001: the timer and the user's click were 20 seconds apart and yielded two identical versions).
# A killed process leaves the mark in the database: older than SUGGEST_MARK_FRESH_SECONDS counts as expired.
SUGGEST_MARK_PREFIX = "schema_suggest:"
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
    """Neither an active version nor a legacy label list: the build would fall back to the global default type
    list -- that is the "blank graph"."""
    return not (config.get("graph_schema_active") or normalize_entity_types(config.get("graph_entity_types")))


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
    mark = json.dumps({"origin": origin, "started_at": now, "token": token}, ensure_ascii=False)
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


def wait_for_suggest(settings: Any, kb_id: str, *, timeout: float | None = None) -> float:
    """If another place is extracting, wait until it finishes (the mark disappears or expires); returns the seconds
    waited."""
    started = time.time()
    limit = SUGGEST_MARK_FRESH_SECONDS if timeout is None else timeout
    while suggest_in_progress(settings, kb_id) is not None and time.time() - started < limit:
        time.sleep(SUGGEST_WAIT_POLL_SECONDS)
    return round(time.time() - started, 1)


def adopt_latest_schema_version(settings: Any, source: KBSource) -> dict[str, Any] | None:
    """If the ring holds an extracted but never adopted version (extracted in the console, never saved), adopt the
    latest one instead of extracting another; returns None when there is none."""
    with db.connect(settings.state_db) as con:
        stored = discovery.get_config(con, source.kb_id)
        versions = ring_with_legacy(stored)
        if not versions or stored.get("graph_schema_active"):
            return None
        entry = versions[0]                       # newest first in the ring
        updates: dict[str, Any] = {"graph_schema_versions": versions, "graph_schema_active": str(entry["id"])}
        apply_schema_version({"graph_schema_versions": versions}, updates)
        discovery.set_config(con, source.kb_id, updates)
        con.commit()
    print(f"[schema] {source.kb_id}({source.source_root}) adopt existing: version {entry['id']} ({entry.get('origin')})"
          f" types={len(entry.get('entity_types') or [])} predicates={len(entry.get('predicates') or [])}", flush=True)
    return {"version_id": str(entry["id"]), "origin": entry.get("origin"), "adopted": True, "adopted_existing": True,
            "entity_types": list(entry.get("entity_types") or []), "predicates": list(entry.get("predicates") or [])}


def ring_with_legacy(stored: dict[str, Any]) -> list[dict[str, Any]]:
    """Before touching the ring, freeze the "labels from before version management" into a real version entry (see
    the 2026-08-24 kb_003 lost-labels incident for why)."""
    ring = [dict(v) for v in (stored.get("graph_schema_versions") or [])]
    if ring:
        return ring
    legacy = list(normalize_entity_types(stored.get("graph_entity_types")))
    if not legacy:
        return ring
    return [{
        "id": LEGACY_SCHEMA_VERSION_ID,
        "created_at": None, "model": None, "sample_size": None,
        "language": stored.get("graph_language"),
        "entity_types": legacy,
        "legacy": True,
    }]


def prior_from_entry(entry: dict[str, Any] | None) -> dict[str, Any] | None:
    if not entry:
        return None
    return {k: entry.get(k) for k in ("id", "domain", "language", "entity_types", "parent_types", "predicates",
                                      "type_definitions", "profile", "observed")}


def reload_source(settings: Any, source: KBSource) -> KBSource:
    """Reassemble the KBSource from the configuration now in the state database (after a new version was adopted
    automatically, the build must see the new labels / predicates)."""
    config = load_config(settings, source.kb_id)
    return discovery.build_source(settings.mirror_root, source.source_root, config,
                                  kb_id=source.kb_id, collection=source.collection)


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


def sample_docs(settings: Any, source: KBSource, *, q: Any = None) -> dict[str, Any]:
    """The sample for label extraction: active chunk ledger -> payloads -> stratified + farthest-point sampling
    (schema.sample_chunks). Returns the sample and how it was drawn."""
    from ..vector.layout import TEXT_VECTOR
    from ..vector.qdrant import client as qdrant_client
    from . import schema as graph_schema
    from .units import fetch_chunk_payloads

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

    sample = graph_schema.sample_chunks(chunks, source.graph_tune_sample_size, fetch_vectors=fetch_vectors)
    return {
        "texts": sample["texts"], "truncated": sample["truncated"],
        "chunks_total": len(chunks), "documents": len({c.doc_id for c in payloads}),
        "sample": {k: sample[k] for k in ("pool", "documents_covered", "documents_total", "excluded_boilerplate", "method")},
    }


def suggest_schema_version(settings: Any, source: KBSource, *, origin: str, adopt: bool, use_prior: bool = True,
                           client: Any = None, docs: list[str] | None = None, sample: dict[str, Any] | None = None,
                           spec: Any = None) -> dict[str, Any]:
    """Extract one label version into the ring; with adopt=True it takes effect immediately (labels / language /
    predicates / parents / definitions / examples / profile are expanded into the active values together).
    With use_prior=True and an active version, the model revises on the basis of that version + its endpoint
    ledger. client / docs are injection points for tests and callers; when absent, the tune model and the sample
    are taken from this KB's configuration."""
    from . import schema as graph_schema

    config = load_config(settings, source.kb_id)
    entry_prior = active_schema_entry(config) if use_prior else None
    prior = prior_from_entry(entry_prior)
    token = claim_suggest(settings, source.kb_id, origin)      # atomic claim, held through sampling, model calls and saving; the other side waits / refuses
    if token is None:
        raise SuggestBusy(SUGGEST_BUSY_MESSAGE)
    try:
        if docs is None:
            sample = sample_docs(settings, source)
            docs = list(sample["texts"])
            if not docs:
                raise ValueError("The active chunks have no text to sample from")
        sample = dict(sample or {})
        close = None
        if client is None:
            from .build import graph_paths, tune_llm_spec
            from .llm import ChatClient, LLMCache

            spec = spec or tune_llm_spec(settings, source)
            cache = LLMCache(graph_paths(settings, source).cache_file)
            close = cache.close
            client = ChatClient(spec, cache=cache, timeout=min(SUGGEST_TIMEOUT_SECONDS, int(settings.graph_llm_timeout_seconds)),
                                workers=1, circuit_fails=0)
        started = time.time()
        stop_renewing = threading.Event()

        def keep_claim() -> None:           # renew the mark during the model calls; stops when extraction ends
            while not stop_renewing.wait(SUGGEST_RENEW_SECONDS):
                try:
                    renew_suggest(settings, source.kb_id, token)
                except Exception:
                    pass

        renewer = threading.Thread(target=keep_claim, name=f"schema-claim-{source.kb_id}", daemon=True)
        renewer.start()
        try:
            result = graph_schema.suggest(client, docs, prior=prior)
        finally:
            stop_renewing.set()
            renewer.join(timeout=5)
            if close is not None:
                close()
        result["entity_types"] = list(normalize_entity_types(result.get("entity_types")))
        result["predicates"] = list(normalize_predicates(result.get("predicates")))
        result["parent_types"] = normalize_parent_types(result.get("parent_types"))
        result["type_definitions"] = normalize_type_definitions(result.get("type_definitions"), result["entity_types"])
        result["examples"] = normalize_examples(result.get("examples"))
        result["profile"] = normalize_profile(result.get("profile"))
        result.pop("persona", None)
        model_name = getattr(spec, "name", None) or str((config.get("graph_llm") or {}).get("tune") or "") or None
        taken = {str(v.get("id")) for v in (config.get("graph_schema_versions") or [])}
        version_id = f"v{int(time.time())}"
        while version_id in taken:          # two versions within one second (tests, scripts) must not collide on id: a collision evicts the earlier entry
            version_id += "x"
        entry = {
            "id": version_id,
            "created_at": int(time.time()),
            "model": model_name,
            "sample_size": int(source.graph_tune_sample_size),
            "sampled": len(docs),
            "chunks_total": sample.get("chunks_total"),
            "documents": sample.get("documents"),
            "language": result.get("language"),
            "entity_types": result["entity_types"],
            "domain": (str(result.get("domain") or "").strip() or None),
            "parent_types": result["parent_types"],
            "predicates": result["predicates"],
            "type_definitions": result["type_definitions"],
            "examples": result["examples"],
            "example_stats": result.get("example_stats") or {},
            "profile": result["profile"],
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
                  f"types={len(entry['entity_types'])} predicates={len(entry['predicates'])}", flush=True)
            raise SuggestBusy(SUGGEST_SUPERSEDED_MESSAGE)
        with db.connect(settings.state_db) as con:
            stored = discovery.get_config(con, source.kb_id)
            versions = push_schema_version(ring_with_legacy(stored), entry, active_id=stored.get("graph_schema_active"))
            updates: dict[str, Any] = {"graph_schema_versions": versions}
            if adopt:
                updates["graph_schema_active"] = entry["id"]
                apply_schema_version({"graph_schema_versions": versions}, updates)
            discovery.set_config(con, source.kb_id, updates)
            con.commit()
        result.update({
            "kb_id": source.kb_id, "sampled": len(docs), "documents": sample.get("documents"),
            "chunks_total": sample.get("chunks_total"), "truncated": bool(sample.get("truncated")),
            "sample": sample.get("sample") or {},
            "version_id": entry["id"], "adopted": bool(adopt), "origin": origin, "prior_id": entry["prior_id"],
            "seconds": round(time.time() - started, 1),
        })
        print(f"[schema] {source.kb_id}({source.source_root}) {origin}: version {entry['id']}"
              f"{' adopted' if adopt else ''} types={len(entry['entity_types'])} predicates={len(entry['predicates'])}"
              f" prior={entry['prior_id']} guard={entry['guard']} seconds={result['seconds']}", flush=True)
        return result

    finally:
        release_suggest(settings, source.kb_id, token)      # released only after the version is saved (or failed): no idle-looking window while still unpublished

def ensure_schema_before_build(settings: Any, source: KBSource) -> tuple[KBSource, dict[str, Any] | None]:
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
        print(f"[graph] {source.kb_id}({source.source_root}): label extraction already running elsewhere ({mark.get('origin')}); waiting for it", flush=True)
        waited = wait_for_suggest(settings, source.kb_id)
        print(f"[graph] {source.kb_id}({source.source_root}): waited {waited}s", flush=True)
    # the ring holds an extracted but unsaved version (extracted in the console without saving, or the one just
    # waited for above): adopt the latest
    info = adopt_latest_schema_version(settings, source)
    if info is not None:
        return reload_source(settings, source), info
    print(f"[graph] {source.kb_id}({source.source_root}): no schema version yet; extracting labels with default settings first"
          f" (sample size {source.graph_tune_sample_size})", flush=True)
    try:
        info = suggest_schema_version(settings, source, origin=ORIGIN_BLANK, adopt=True, use_prior=False)
    except SuggestBusy:
        # someone (the console) started extracting between the wait and the claim: wait again and adopt their version
        waited = wait_for_suggest(settings, source.kb_id)
        info = adopt_latest_schema_version(settings, source)
        if info is None:
            raise
        print(f"[graph] {source.kb_id}({source.source_root}): lost the claim; adopted the other run's version after {waited}s", flush=True)
    return reload_source(settings, source), info


def resuggest_for_rebuild(settings: Any, source: KBSource) -> tuple[KBSource, dict[str, Any]]:
    """Before a threshold-triggered full rebuild: re-extract a version on the basis of the active version and its
    ledger, and adopt it. Skipped when graph_rebuild_resuggest is off; a KB without any version is left to the
    build itself (ensure_schema_before_build)."""
    try:
        config = load_config(settings, source.kb_id)
    except KeyError:
        return source, {"skipped": "unknown_kb"}
    if not bool_value(config.get("graph_rebuild_resuggest"), True):
        return source, {"skipped": "disabled"}
    if schema_missing(config):
        return source, {"skipped": "no_schema"}
    info = suggest_schema_version(settings, source, origin=ORIGIN_REBUILD, adopt=True, use_prior=True)
    return reload_source(settings, source), {k: info.get(k) for k in ("version_id", "prior_id", "origin", "guard", "seconds",
                                                                       "entity_types", "predicates")}
