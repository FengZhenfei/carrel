"""Entity graph build orchestration: prepare corpus → extract → merge → write vectors → graph database
import → switch aliases → GC.

Everything runs inside this process; no external indexing subprocess is spawned. Phase markers
(graph_build_phases), same-version resume, the build lock, signal handling, alias switching with rollback,
GC, the rebuild policy and adopt-current all keep their original mechanisms.

Incremental append (incremental=True, 2026-09-05) goes through the same pipeline and likewise produces a
complete new version; it only saves work in three places by leaning on the previous version: extraction is
cached per unit anyway, so only new units are extracted; entity resolution replays the previous version's
same-entity decisions (see the prior argument of resolution.resolve); the vector write reuses the previous
version's vectors for rows whose title/description did not change (see reuse_from in
vectors.write_graph_vectors). A full rebuild (incremental=False) redoes all three, resetting the drift that
appends accumulate; the rebuild policy only takes full versions as its baseline (evaluate_rebuild).

Workspace layout (settings.graph_work_dir, default runtime/graph):
  work/<short>/<version>/manifest.json   size and configuration of the corpus snapshot
  work/<short>/<version>/units.jsonl     extraction units (with text)
  work/<short>/<version>/graph.json      the merged graph (entities / relations / chunk attribution / stats)
  cache/<short>.sqlite                   LLM response cache (shared across versions)
The extraction results themselves live in the state database's graph_extractions table, keyed by
(kb_id, unit_id, fingerprint).
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Callable

from .. import db
from ..config import Settings
from ..limits import GRAPH_ENTITY_TYPES_DEFAULT
from ..models import KBSource
from ..utils import stable_json_hash
from ..vector.qdrant import (
    ALL_GRAPH_VECTOR_TYPES,
    GRAPH_OPTIONAL_TYPES,
    GRAPH_VECTOR_TYPES,
    LEGACY_GRAPH_VECTOR_TYPES,
    activate_graph_aliases,
    client as qdrant_client,
    collection_exists,
    delete_old_graph_collections,
    drop_graph_aliases,
    graph_alias_targets,
    graph_collection_alias,
    graph_collection_name,
    graph_collection_short_name,
    graph_version_timestamp,
    parse_graph_collection_name,
    restore_graph_aliases,
)
from . import prompts
from .extract import ExtractionSchema, GraphExtractor
from .llm import ChatClient, LLMCache, LLMCircuitOpen, LLMInputRejected, LLMInterrupted, LLMSpec, cache_count


@dataclass(frozen=True)
class GraphPaths:
    source_short: str
    work_dir: Path
    manifest_file: Path
    units_file: Path
    graph_file: Path
    cache_file: Path

    @property
    def output_dir(self) -> Path:
        # Build records' output_dir has always been the artifact directory; artifacts now live in the workspace
        return self.work_dir


from .lock import GraphBuildLock, build_lock_held, build_lock_path  # noqa: E402  build lock (flock, see lock.py)


def graph_paths(settings: Settings, source: KBSource, graph_version: str | None = None) -> GraphPaths:
    short = graph_collection_short_name(source.collection)
    work = settings.graph_work_dir / "work" / short
    if graph_version:
        work = work / graph_version
    return GraphPaths(
        source_short=short,
        work_dir=work,
        manifest_file=work / "manifest.json",
        units_file=work / "units.jsonl",
        graph_file=work / "graph.json",
        cache_file=settings.graph_work_dir / "cache" / f"{short}.sqlite",
    )


def default_graph_version(source: KBSource) -> str:
    suffix = secrets.token_hex(3)
    return f"{graph_collection_short_name(source.collection)}-{time.strftime('%Y%m%d-%H%M%S')}-{suffix}"


def write_graph_file(path: Path, graph: dict[str, Any]) -> None:
    """graph.json is first written to a temporary file in the same directory and renamed once it is on disk. The
    facts and view-page phases keep writing on top of the merge phase's output; overwritten in place, a process
    killed or a power loss halfway would leave a truncated file while the earlier phases' completion marks
    remain, and every resume would fail to read it."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(graph, ensure_ascii=False))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


class GraphBuildInterrupted(RuntimeError):
    """The graph build was interrupted by a signal (e.g. systemctl stop / pause). Inherits RuntimeError so that
    the except Exception in build_graph writes the record as cancelled instead of leaving an orphan row that
    stays running forever."""


class GraphBuildCalledOff(GraphBuildInterrupted):
    """This KB's own switches called the build off (KB turned off, graph turned off, paused); the process did not
    receive a stop signal. The record is written as cancelled all the same; the timer check only skips this KB
    and goes on with the ones after it."""


class GraphInputIncomplete(RuntimeError):
    """The frozen chunk ledger does not match the payloads fetched from the main store (missing points, text
    fingerprint mismatch, content version mismatch): this build does not start, the next round retries
    (final review F08)."""


def kb_still_active(settings: Settings, kb_id: str) -> bool:
    """Re-check before publishing that the knowledge base is still enabled (final review F02). A missing registry
    row (test stubs, extra sources) counts as enabled and is not blocked."""
    try:
        with db.connect(settings.state_db) as con:
            row = con.execute("SELECT status FROM kb_sources WHERE kb_id = ?", (kb_id,)).fetchone()
    except Exception:
        return True
    return row is None or str(row["status"]) == "active"


def graph_switched_off(settings: Settings, source: KBSource, *, paused_at_start: bool = False) -> str | None:
    """The current values of this KB's graph switch and pause mark (read fresh from the registry): when the graph
    is turned off or paused, returns the reason, and the build then neither starts nor publishes.
    The source held by the build process was assembled before the run: the run record is only written once the
    lock is taken, so a "Turn off knowledge graph" clicked before that finds no process and only saves the
    configuration; for KBs late in a timer check round, the configuration may have been read hours ago.
    A pause already set when the run started does not count (a manual build from the command line); a KB missing
    from the registry (test stubs, extra sources) is not blocked."""
    from .schema_flow import reload_source

    try:
        now = reload_source(settings, source)
    except Exception:
        return None
    if not now.graph_enabled:
        return "The knowledge graph has been turned off"
    if now.graph_paused and not paused_at_start:
        return "The graph build has been paused"
    return None


def input_drift(ledger: Iterable[dict[str, Any]], payloads: Iterable[Any]) -> dict[str, int]:
    """Check the ledger (active chunks in SQLite) against the payloads fetched from the main store point by point:
    count the points the store did not return, the text fingerprint mismatches and the content version
    mismatches. Old ledgers without text_sha store an empty string, treated as unknown and not compared."""
    by_point = {str(getattr(p, "point_id", "")): p for p in payloads}
    out = {"missing": 0, "text_mismatch": 0, "version_mismatch": 0}
    for c in ledger:
        pid = str(c.get("point_id") or "")
        if not pid:
            continue
        p = by_point.get(pid)
        if p is None:
            out["missing"] += 1
            continue
        want_ver, got_ver = str(c.get("content_version") or ""), str(getattr(p, "content_version", "") or "")
        if want_ver and got_ver and want_ver != got_ver:
            out["version_mismatch"] += 1
        sha = str(c.get("text_sha") or "")
        if sha and db.chunk_text_sha(str(getattr(p, "text", "") or "")) != sha:
            out["text_mismatch"] += 1
    return out


def attribution_texts(ledger: Iterable[dict[str, Any]], payloads: Iterable[Any]) -> tuple[dict[str, str], dict[str, str], int]:
    """Texts and section paths for chunk attribution in the merge phase, keeping only chunks whose text matches
    the frozen chunk ledger; returns (texts, sections, number of mismatched chunks).
    Extraction used the text frozen when the corpus was prepared, but attribution fetches it again from the main
    store in the merge phase: when a file of the same version was re-parsed in between and its text replaced in
    place, matching the new text against the old extraction results would hang entities on unrelated chunks.
    Mismatched chunks take no part in exact attribution -- an entity that matches no chunk of its unit is still
    attributed to the whole unit. Ledger rows without a fingerprint (old ledgers) are not compared."""
    frozen = {str(c.get("point_id") or ""): str(c.get("text_sha") or "") for c in ledger}
    texts: dict[str, str] = {}
    sections: dict[str, str] = {}
    stale = 0
    for c in payloads:
        sha = frozen.get(str(c.point_id), "")
        if sha and db.chunk_text_sha(str(c.text or "")) != sha:
            stale += 1
            continue
        texts[c.point_id] = c.text
        sections[c.point_id] = " > ".join(x for x in c.section_path if x)
    return texts, sections, stale


class NoGraphCorpus(RuntimeError):
    """This knowledge base has no active chunks yet, so there is no corpus for the graph build."""


# ── Model slots ──────────────────────────────────────────────────────────

GRAPH_LLM_STEPS = ("extract", "summarize")
# The model slot owned by the "extract labels now / re-extract" step. Deliberately kept out of GRAPH_LLM_STEPS:
# those two are hard prerequisites of a graph build, while tune is only needed when that button is clicked.
GRAPH_TUNE_STEP = "tune"
# Default when a model slot is unset (user decision 2026-09-08): use this entry if the registry has it; if not but
# the registry lists exactly one model, use that one (2026-09-12: the user keeps one model and should not have to
# pick it per knowledge base); only with several models and no default name do we report "no model selected"
DEFAULT_GRAPH_LLM = "DeepSeek V4 Flash"


def default_graph_llm(con: Any) -> str | None:
    """Which registry entry to use when a model slot is unset; None = no fallback available (shared by the console
    and resolve_llm_specs)."""
    if db.get_llm(con, DEFAULT_GRAPH_LLM) is not None:
        return DEFAULT_GRAPH_LLM
    rows = db.list_llms(con)
    if len(rows) == 1:
        return str(rows[0]["name"])
    return None
GRAPH_STEP_LABELS = {
    "extract": "Entity extraction model",
    "summarize": "Description summary model",
    "tune": "Label extraction model",
}


def resolve_llm_specs(settings: Settings, source: KBSource, steps: tuple[str, ...]) -> dict[str, LLMSpec]:
    """Map the model chosen for each step in this KB's graph_llm config to its registry row. A missing one raises,
    and the error must be phrased in plain words (the name of that console field), because this is the preflight
    check before a run starts."""
    from .. import discovery

    with db.connect(settings.state_db) as con:
        try:
            kb_llm = dict(discovery.get_config(con, source.kb_id).get("graph_llm") or {})
        except KeyError:
            kb_llm = {}
        specs: dict[str, LLMSpec] = {}
        for step in steps:
            label = GRAPH_STEP_LABELS.get(step, step)
            name = str(kb_llm.get(step) or "").strip()
            if not name:
                name = default_graph_llm(con) or ""
                if not name:
                    raise RuntimeError(
                        f"Knowledge base “{source.source_root}” has no “{label}” selected: "
                        "pick one in the console's Knowledge graph section and click Save"
                    )
            row = db.get_llm(con, name)
            if row is None:
                raise RuntimeError(f"“{label}” references model {name!r}, which is not in the registry")
            specs[step] = LLMSpec.from_row(row)
    return specs


def tune_llm_spec(settings: Settings, source: KBSource) -> LLMSpec:
    return resolve_llm_specs(settings, source, (GRAPH_TUNE_STEP,))[GRAPH_TUNE_STEP]


def deterministic_extractor(settings: Settings, source: KBSource, units: list[Any]):
    """This KB's deterministic extractor (code / structured markdown / config, routed by file type); the original
    files are read from the mirror directory."""
    from .deterministic import DeterministicExtractor

    from ..discovery import directory_admitted
    from ..localfs.scanner import inside_boundary

    base = Path(settings.mirror_root) / source.source_root

    def file_for(rel: str) -> Path | None:
        # Checked at every read, not once at construction: the source object may have been loaded before the
        # directory was swapped for a symbolic link (third review, item 2). Only an admitted root is resolved
        # into the boundary, so a refused link's target never becomes it; a file swapped for a link is read
        # as absent.
        if not directory_admitted(Path(settings.mirror_root), str(source.source_root))[0]:
            return None
        path = base / rel
        return path if inside_boundary(path, base.resolve()) else None

    return DeterministicExtractor(units, file_for=file_for, kb_id=source.kb_id)


def extraction_fingerprint(base_fingerprint: str, *, deterministic_units: int) -> str:
    """Cache fingerprint of the extraction results: when some units use deterministic extraction, mix the rule
    version in (rule changes must trigger a recompute); KBs without such units keep the same fingerprint (so old
    KBs' caches are not invalidated for nothing)."""
    from .deterministic import VERSION

    if not deterministic_units:
        return base_fingerprint
    return hashlib.sha256(f"{base_fingerprint}|{VERSION}".encode("utf-8")).hexdigest()[:24]


def prune_unit_caches(settings: Settings, source: KBSource, units_file: Path) -> dict[str, int]:
    """Prune the two per-unit caches (extraction results, structured facts) against this version's unit table:
    rows of units missing from the table are deleted, rows of units in it are kept whatever their fingerprint."""
    from .units import read_units

    keep_ids = {u.unit_id for u in read_units(units_file)}
    with db.connect(settings.state_db) as con:
        return {
            "extraction_cache_pruned": db.prune_graph_extractions(con, source.kb_id, keep_ids),
            "facts_cache_pruned": db.prune_graph_extractions(con, source.kb_id, keep_ids, table="graph_facts"),
        }


def extraction_schema_for(source: KBSource) -> ExtractionSchema:
    profile = dict(getattr(source, "graph_profile", None) or {})
    return ExtractionSchema(
        entity_types=tuple(source.graph_entity_types) or tuple(GRAPH_ENTITY_TYPES_DEFAULT),
        predicates=tuple(dict(p) for p in (source.graph_predicates or ())),
        language=source.graph_language or "English",
        parent_types=dict(source.graph_parent_types or {}),
        type_definitions=dict(getattr(source, "graph_type_definitions", None) or {}),
        examples=str(getattr(source, "graph_examples", "") or ""),
        subject_types=tuple(str(t) for t in (profile.get("subject_types") or ())),
    )


def scenario_profile(source: KBSource) -> dict[str, Any]:
    """This KB's scenario profile (induced in the schema phase, editable in the console): subject types, axis,
    conclusion headings, extension predicates. Missing items fall back to defaults."""
    raw = dict(getattr(source, "graph_profile", None) or {})
    return {
        "subject_types": [str(t) for t in (raw.get("subject_types") or ())],
        "axis": str(raw.get("axis") or "auto"),
        "conclusion_headings": [str(h) for h in (raw.get("conclusion_headings") or ())],
        "boilerplate_headings": [str(h) for h in (raw.get("boilerplate_headings") or ())],
        "listing_headings": [str(h) for h in (raw.get("listing_headings") or ())],
        "type_words": [str(w) for w in (raw.get("type_words") or ())],
        "extension_predicates": [str(p) for p in (raw.get("extension_predicates") or ())],
    }


def prompts_hash() -> str:
    """Hash of the prompt texts themselves (every upper-case string constant in
    prompts.py), not of the file: comments and docstrings there can change
    without invalidating every knowledge base's extraction cache."""
    parts = [f"{name}={value}" for name, value in sorted(vars(prompts).items())
             if name.isupper() and isinstance(value, str)]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def graph_cache_fingerprint(settings: Settings, source: KBSource) -> str:
    """Compress the configuration items whose change invalidates the whole extraction cache into one fingerprint:
    the per-step models, the type table, the predicate table, parent types, language, unit size, gleaning rounds
    and the prompt texts. The console compares it with the one recorded when the build started and, on mismatch,
    turns "Continue build" back into "Build now / rebuild". The corpus dimension is tracked separately by
    graph_builds.source_content_hash."""
    from .facts import FACTS_MAX_PER_UNIT

    specs = resolve_llm_specs(settings, source, GRAPH_LLM_STEPS)
    return stable_json_hash({
        # Version tag of the fingerprint layout. Any change to the constants below invalidates every recorded
        # fingerprint: resume is judged impossible and append is judged "configuration changed", so every
        # graph-enabled knowledge base has to be rebuilt in full. Bump it only together with such a change.
        "v": "local-graph-3",
        "mode": "entity_graph",
        "models": {step: spec.model_id for step, spec in sorted(specs.items())},
        "entity_types": list(source.graph_entity_types) or list(GRAPH_ENTITY_TYPES_DEFAULT),
        "predicates": [str(p.get("name") or "") for p in (source.graph_predicates or ())],
        # Predicate descriptions and endpoint constraints are part of the fingerprint too: they go into the
        # extraction prompt and decide the later type-violation flags. Comparing names only would let phase
        # artifacts extracted / flagged under the old definitions be reused after a change
        "predicate_defs": [
            {"name": str(p.get("name") or ""), "description": str(p.get("description") or ""),
             "source_parents": sorted(str(x) for x in (p.get("source_parents") or ())),
             "target_parents": sorted(str(x) for x in (p.get("target_parents") or ()))}
            for p in sorted((source.graph_predicates or ()), key=lambda p: str(p.get("name") or ""))
        ],
        "parent_types": dict(source.graph_parent_types or {}),
        "language": source.graph_language or "",
        # 2026-09-07: type definitions, per-KB few-shot examples and the scenario profile all feed the prompt or
        # decide unit classification / axis, so changing them invalidates the cache
        "type_definitions": dict(getattr(source, "graph_type_definitions", None) or {}),
        "examples": hashlib.sha256(str(getattr(source, "graph_examples", "") or "").encode("utf-8")).hexdigest()[:16],
        "profile": scenario_profile(source),
        "unit_chunks": int(source.graph_unit_chunks),
        "max_gleanings": int(source.graph_max_gleanings),
        "facts_max_per_unit": int(FACTS_MAX_PER_UNIT),
        "prompts": prompts_hash(),
        # The embedding model used for graph vectors: switching models at the same dimension means old vectors
        # cannot be reused (the reuse in vectors.py re-checks embed_model on each point as well)
        "embedding": {"model": str(getattr(settings, "embedding_model_id", "") or ""),
                      "dim": int(getattr(settings, "embedding_dim", 0) or 0)},
    })


def _env_flag(name: str) -> bool:
    import os
    return str(os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def facts_phase_gate(failed: list[tuple[Any, BaseException]], *, partial_ok: bool) -> None:
    """Completion condition of the facts phase: it does not count as complete while any unit failed (malformed
    JSON, failed call). Malformed output used to be taken as a "successful empty result", the phase completed
    anyway and nobody knew facts were missing (Codex review F03). Raising lets run_phase retry (successful units
    are cached, only the failed ones are called again); if retries still fail, the operator sets
    KB_GRAPH_FACTS_PARTIAL_OK=1 to explicitly accept a partial release, with the failed units still listed in
    stats.
    Units the provider rejected for their content (LLMInputRejected: content inspection, length) do not count
    as failed, the same rule as in the extraction phase: only a different input would change the outcome, so
    they are a loss inherent to the corpus. Counted, a single table unit stopped by the provider's inspection
    made every later append and rebuild of the base fail in the facts phase and the graph stopped updating
    (2026-09-29 audit)."""
    failed = [(u, err) for u, err in failed if not isinstance(err, LLMInputRejected)]
    if not failed or partial_ok:
        return
    ids = ", ".join(str(getattr(u, "unit_id", u)) for u, _ in failed[:8])
    kinds = "/".join(sorted({type(err).__name__ for _, err in failed}))
    raise RuntimeError(
        f"Structured facts failed for {len(failed)} units ({kinds}): {ids}{' …' if len(failed) > 8 else ''}; "
        "if retries keep failing, set KB_GRAPH_FACTS_PARTIAL_OK=1 to accept a partial release explicitly"
    )


class _StoppableEmbedder:
    """A wrapper around the embedding client: one batch per call, with a stop-signal check after every batch. The
    client's retries swallow an interruption that lands inside an HTTP call as an ordinary error, and the
    resolution phase embeds tens of thousands of titles at once; without the checks the process would be killed
    before the embedding finished."""

    def __init__(self, client: Any, check_stop: Callable[[], None]) -> None:
        self.client = client
        self.check_stop = check_stop

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        step = max(1, int(self.client.batch_size))
        for start in range(0, len(texts), step):
            vectors.extend(self.client.embed(texts[start:start + step]))
            self.check_stop()
        return vectors


def graph_embedder(settings: Settings, check_stop: Callable[[], None] | None = None):
    """The embedding client for graph builds; the pause between calls and the retry count follow the configuration
    (EMBEDDING_SLEEP_SECONDS / EMBEDDING_RETRY). Not passing them left a configured pause without effect on graph
    builds: a fixed 0.1 s after every call, and with the tens of thousands of calls needed to write graph vectors
    the pauses alone took up most of that phase. With check_stop the stop signal is checked batch by batch (see
    _StoppableEmbedder)."""
    from ..embedding.client import EmbeddingClient

    client = EmbeddingClient(
        base_url=settings.embedding_base_url, api_key=settings.embedding_api_key,
        model_id=settings.embedding_model_id, dim=settings.embedding_dim, batch_size=settings.embedding_batch,
        retry=settings.embedding_retry, sleep_seconds=settings.embedding_sleep_seconds,
    )
    return client if check_stop is None else _StoppableEmbedder(client, check_stop)


def graph_cache_entries(settings: Settings, source_collection: str) -> int:
    """LLM response cache entries accumulated by this KB: a direct measure of how much a resume gets for free."""
    short = graph_collection_short_name(source_collection)
    return cache_count(settings.graph_work_dir / "cache" / f"{short}.sqlite")


# ── Corpus ledger ────────────────────────────────────────────────────────

def active_source_chunks(settings: Settings, source: KBSource) -> list[dict[str, str]]:
    with db.connect(settings.state_db) as con:
        return db.active_chunk_refs(con, source.collection)


def source_snapshot_hash(chunks: list[dict[str, str]]) -> str:
    """Corpus snapshot fingerprint: identity of the active chunks + text fingerprints, independent of order.

    This used to be cached by id(chunks): the cache did not hold the object, so once the list was collected and
    its id reused by a new list, the stale value was returned; it also excluded text_sha, so a text change under
    the same point_id / version left the fingerprint unchanged, giving two different definitions of "same
    content" next to the document delta (which already judges changes by text_sha) (Codex review F04). Sorting
    and hashing a few thousand dicts takes tens of milliseconds, not worth caching."""
    values = [
        {
            "point_id": str(chunk.get("point_id") or ""),
            "chunk_uid": str(chunk.get("chunk_uid") or ""),
            "doc_id": str(chunk.get("doc_id") or ""),
            "content_version": str(chunk.get("content_version") or ""),
            "text_sha": str(chunk.get("text_sha") or ""),
        }
        for chunk in chunks
    ]
    values.sort(key=lambda item: (item["point_id"], item["doc_id"], item["chunk_uid"], item["content_version"], item["text_sha"]))
    return stable_json_hash(values)


def filter_chunk_refs(chunks: list[dict[str, str]], doc_ids: Iterable[str] | None) -> list[dict[str, str]]:
    """Keep only the chunks of these documents (trial build: build one version from a few files to see the effect);
    doc_ids=None means everything."""
    if doc_ids is None:
        return chunks
    wanted = {str(d) for d in doc_ids}

    def doc_of(c: dict[str, str]) -> str:
        return str(c.get("doc_id") or (f"{c.get('kb_id')}:{c.get('file_key')}" if c.get("file_key") else ""))

    return [c for c in chunks if doc_of(c) in wanted]


def prepare_graph_input(settings: Settings, source: KBSource, *, paths: GraphPaths, q: Any = None,
                        write_units_file: bool = True, doc_ids: Iterable[str] | None = None) -> tuple[dict[str, Any], list[dict[str, str]], list[Any]]:
    """Graph build input = the active chunk ledger at this moment + extraction units assembled from the chunk
    payloads fetched from the main store. Returns (manifest, chunks, units). With doc_ids, only those documents
    are used (trial build)."""
    from .units import build_units, fetch_chunk_payloads, units_summary, write_units

    with db.connect(settings.state_db) as con:
        chunks = filter_chunk_refs(db.active_chunk_refs(con, source.collection), doc_ids)
    if not chunks:
        raise NoGraphCorpus(
            "This knowledge base has no active chunks (nothing parsed and stored, or everything deactivated / deleted), so there is no corpus to build from. Finish parsing first."
        )
    from .temporal import axis_for_units

    q = q or qdrant_client(settings.qdrant_url, settings.qdrant_api_key)
    payloads = fetch_chunk_payloads(q, source.collection, chunks)
    drift = input_drift(chunks, payloads)
    if drift["missing"] or drift["text_mismatch"] or drift["version_mismatch"]:
        # Missing points and misaligned text were once skipped silently, with the full ledger recorded as baseline,
        # so the next round saw no input change (final review F08): fail explicitly and let the next round retry
        raise GraphInputIncomplete(
            f"The frozen chunk ledger does not match the main store: {drift['missing']} points missing, "
            f"{drift['text_mismatch']} text fingerprints differ, {drift['version_mismatch']} content versions differ. "
            "This build does not start; the next check retries"
        )
    profile = scenario_profile(source)
    units = build_units(payloads, kb_id=source.kb_id, unit_chunks=source.graph_unit_chunks,
                        conclusion_headings=profile["conclusion_headings"],
                        boilerplate_headings=profile["boilerplate_headings"], listing_headings=profile["listing_headings"])
    if not units:
        raise NoGraphCorpus("The active chunks have no text, nothing to extract from")
    # Document axis value (date / version): taken per the profile's axis, auto prefers a date when present;
    # written into the unit (not into unit_id)
    axes = axis_for_units(units, kind=profile["axis"])
    for u in units:
        u.axis = str((axes.get(u.doc_id) or {}).get("value") or "")
    summary = units_summary(units)
    manifest = {
        "source": "qdrant_chunks",
        "collection": source.collection,
        "kb_id": source.kb_id,
        "source_root": source.source_root,
        "work_dir": str(paths.work_dir),
        "doc_filter": sorted({str(d) for d in doc_ids}) if doc_ids is not None else None,
        "documents": summary["documents"],
        "active_chunks": len(chunks),
        "chunks_with_text": len(payloads),
        "units": summary["units"],
        "units_by_kind": summary["units_by_kind"],
        "unit_chunks": int(source.graph_unit_chunks),
        "max_gleanings": int(source.graph_max_gleanings),
        "avg_unit_chunks": summary["avg_unit_chunks"],
        "avg_unit_tokens": summary["avg_unit_tokens"],
        "max_unit_tokens": summary["max_unit_tokens"],
        "profile": profile,
        "documents_axis": {doc: {"rel_path": next((u.rel_path for u in units if u.doc_id == doc), ""), **ax}
                           for doc, ax in sorted(axes.items())},
    }
    if write_units_file:
        write_units(paths.units_file, units)
        paths.manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest, chunks, units


# ── Phases ───────────────────────────────────────────────────────────────

GRAPH_PHASES: tuple[tuple[str, str], ...] = (
    ("prepare_input", "Preparing corpus"),
    ("extract", "Entity extraction"),
    ("merge", "Merge & resolution"),
    ("facts", "Structured facts"),
    ("compile", "Compiling view pages"),
    ("enrich", "Writing vectors"),
    ("neo4j_import", "Graph database import"),
)
GRAPH_PHASE_LABELS = dict(GRAPH_PHASES)

# Retries for network phases (Qdrant write-back, Neo4j import): exponential backoff, first delay 5s.
_PHASE_RETRY_DEFAULTS = {"retries": 3, "backoff": 5.0}
# Unit failure ratio tolerated by the extraction phase (units still failing after retries / all units); beyond it
# the phase fails as a whole. Successful units are in graph_extractions, the next run only redoes the failed ones.
_MAX_FAILED_UNIT_RATIO = 0.05


def _phase_retry_config() -> dict[str, float]:
    cfg: dict[str, float] = dict(_PHASE_RETRY_DEFAULTS)
    for key, env, cast in (("retries", "KB_GRAPH_PHASE_RETRIES", int),
                           ("backoff", "KB_GRAPH_PHASE_BACKOFF_SECONDS", float)):
        raw = os.getenv(env)
        if raw:
            try:
                cfg[key] = cast(raw)
            except ValueError:
                pass
    return cfg


def _run_phase(settings: Settings, *, build_id: str, phase: str, label: str, fn, done: set[str], stage,
               retries: int = 1, backoff: float = 5.0, interrupted: threading.Event | None = None):
    """Run one build phase: skip it if already marked done, otherwise run with backoff retries and write a phase
    marker on completion. Returns fn's return value; None when skipped, in which case the caller rebuilds what it
    needs from the previous artifacts.
    interrupted: the mark set when a stop signal arrives. A signal that lands inside an HTTP call raises an
    interruption that the client's retries swallow or wrap in their own exception type; the mark is not lost, and
    once it is set the phase does not start, is not retried and is not marked done, but ends as interrupted."""
    if phase in done:
        stage(f"{label} (done, skipped)")
        print(f"[graph] phase={phase} skipped: done in a previous run", flush=True)
        return None
    stage(label)
    attempt = 0
    while True:
        attempt += 1
        try:
            if interrupted is not None and interrupted.is_set():
                raise GraphBuildInterrupted("Graph build interrupted by a stop signal")
            out = fn()
            if interrupted is not None and interrupted.is_set():
                # the phase returned normally, so the interruption was swallowed inside: either by a retry (the
                # output is complete) or by a "degrade on error" branch (one route of resolution candidates missing
                # when the vector service is down). There is no telling which, so the phase is not marked done and
                # a resume runs it again
                raise GraphBuildInterrupted("Graph build interrupted by a stop signal")
            break
        except (GraphBuildInterrupted, LLMInterrupted, LLMCircuitOpen):
            raise
        except Exception as exc:
            if interrupted is not None and interrupted.is_set():
                raise GraphBuildInterrupted(f"Graph build interrupted by a stop signal ({exc!r})") from exc
            if attempt >= max(1, int(retries)):
                raise
            delay = float(backoff) * (2 ** (attempt - 1))
            print(f"[graph] phase={phase} attempt {attempt} failed: {exc!r}; retrying in {delay:.0f}s", flush=True)
            stage(f"{label} (attempt {attempt} failed, retrying in {delay:.0f}s)")
            time.sleep(delay)
            stage(label)
    with db.connect(settings.state_db) as con:
        db.mark_graph_phase_done(con, build_id, phase)
    done.add(phase)
    return out


def _resumable_phases(con, previous, *, settings: Settings, source: KBSource) -> set[str]:
    """When resuming the same version, whether the phases marked last time still count: only if both the config
    fingerprint and the corpus fingerprint are unchanged."""
    build_id = str(previous["graph_build_id"])
    same_config = str(previous["cache_fingerprint"] or "") == graph_cache_fingerprint(settings, source)
    stored_hash = str(previous["source_content_hash"] or "")
    same_corpus = bool(stored_hash) and stored_hash == source_snapshot_hash(db.active_chunk_refs(con, source.collection))
    if same_config and same_corpus:
        return set(db.graph_phases_done(con, build_id))
    db.clear_graph_phases(con, build_id)
    return set()


class _Progress:
    """Throttle for in-phase progress updates: write stage at most once every 1.5 seconds, the last one always."""

    def __init__(self, stage: Callable[[str], None], interval: float = 1.5) -> None:
        self.stage = stage
        self.interval = interval
        self.last = 0.0

    def __call__(self, label: str, *, force: bool = False) -> None:
        now = time.time()
        if force or now - self.last >= self.interval:
            self.last = now
            self.stage(label)


# ── Main flow ────────────────────────────────────────────────────────────

def build_graph(
    settings: Settings,
    *,
    source_key: str,
    source: KBSource,
    graph_version: str | None = None,
    dry_run: bool = False,
    activate_aliases: bool = True,
    allow_existing_graph_version: bool = False,
    run_gc: bool = True,
    doc_ids: Iterable[str] | None = None,
    import_neo4j: bool | None = None,
    incremental: bool = False,
) -> dict[str, Any]:
    """doc_ids: build the graph from these documents only (trial build). A trial build usually pairs with
    activate_aliases=False, run_gc=False, import_neo4j=True: the version goes into Neo4j but aliases are not
    switched and GC leaves it alone; inspect it with graph_query(graph_version=...) and delete it by hand afterwards.
    When import_neo4j is not given it follows activate_aliases (builds that do not switch aliases have never
    imported into Neo4j either).
    incremental: incremental append, with the current version as baseline, resolution replay and vector reuse
    (see the module docstring); refused when the config fingerprint changed."""
    if not source.graph_enabled:
        raise ValueError(f"source {source_key!r} has graph_enabled=false")
    from .units import read_units

    # The build lock comes before any preparation (final review F03): it is held during the minutes of automatic
    # label extraction too, so a second "start" is refused right here and the delete-graph / delete-KB probes can
    # see that a build is in progress
    lock = GraphBuildLock(settings) if not dry_run else None
    if lock is not None:
        lock.acquire()
    schema_auto: dict[str, Any] | None = None
    paused_at_start = bool(source.graph_paused)

    graph_version = graph_version or default_graph_version(source)
    print(f"[graph] build start kb={source.kb_id}({source.source_root}) version={graph_version} "
          f"kind={'append' if incremental else 'full'} dry_run={dry_run}", flush=True)
    paths = graph_paths(settings, source, graph_version)
    build_id = "dry-run"
    result: dict[str, Any] = {}
    input_manifest: dict[str, Any] | None = None
    build_chunks: list[dict[str, str]] = []
    stop_event = threading.Event()
    interrupted = threading.Event()
    restore_signals = _install_build_signal_handlers(stop_event, interrupted)

    def check_stop() -> None:
        # the main thread is mostly blocked in HTTP calls (embedding, writing vectors), where the retries of that
        # layer swallow the interruption the signal raises: check again at each checkpoint
        if interrupted.is_set():
            raise GraphBuildInterrupted("Graph build interrupted by a stop signal")

    try:
        q = qdrant_client(settings.qdrant_url, settings.qdrant_api_key)
        graph_types = list(GRAPH_VECTOR_TYPES)
        if not allow_existing_graph_version:
            existing = {item.name for item in q.get_collections().collections}
            targets = [graph_collection_name(source.collection, t, graph_version) for t in graph_types]
            collisions = [name for name in targets if name in existing]
            if collisions:
                raise RuntimeError(f"Graph collection(s) already exist for graph_version={graph_version}: {', '.join(collisions)}")

        # Preflight checks go first (before creating the record and workspace): models all set, Neo4j reachable.
        specs = resolve_llm_specs(settings, source, GRAPH_LLM_STEPS)
        if settings.graph_neo4j_import_after_build and not dry_run:
            from .neo4j_import import neo4j_driver

            try:
                driver = neo4j_driver(settings)
                try:
                    driver.verify_connectivity()
                finally:
                    driver.close()
            except Exception as exc:
                raise RuntimeError(
                    f"Neo4j is unavailable ({exc.__class__.__name__}); the build does not start so LLM calls are not wasted. "
                    "Fix Neo4j or set GRAPH_NEO4J_IMPORT_AFTER_BUILD=0"
                ) from exc

        db.init_db(settings.state_db)
        # Baseline for an incremental append: the current version (latest completed, non-trial build). Its
        # extraction and resolution cannot be reused once the config fingerprint has changed.
        base_row = None
        base_graph: dict[str, Any] | None = None
        base_docs: dict[str, set[tuple[str, str]]] = {}
        if incremental:
            with db.connect(settings.state_db) as con:
                base_row = db.latest_successful_graph_build(con, source_key)
                if base_row is None:
                    raise ValueError("This knowledge base has no completed graph yet, so nothing can be appended; run a full build first")
                if str(base_row["cache_fingerprint"] or "") != graph_cache_fingerprint(settings, source):
                    raise ValueError("Model / labels / predicates / unit settings changed; the previous extraction and merge cannot be reused, run a full rebuild")
                base_docs = db.graph_build_doc_chunks(con, str(base_row["graph_build_id"]))
        phases_done: set[str] = set()
        if not dry_run:
            # the run record is written before the slow preparations (automatic label extraction, reading the
            # previous version's graph): "Turn off knowledge graph", "Pause build" and KB deletion all find the
            # process by its running record, and during the minutes of label extraction the lock used to be held
            # without a record, so a build whose graph had been turned off still ran to the end and published
            with db.connect(settings.state_db) as con:
                previous = (db.graph_build_by_version(con, source.collection, graph_version)
                            if allow_existing_graph_version else None)
                build_id = db.begin_graph_build(
                    con, source_key=source_key, kb_id=source.kb_id, source_collection=source.collection,
                    graph_version=graph_version, allow_existing=allow_existing_graph_version,
                    cache_fingerprint=graph_cache_fingerprint(settings, source),
                    build_kind="append" if incremental else "full",
                )
                # in the same transaction as the fingerprint rewrite: once the record carries this run's fingerprint,
                # the phase marks left on it must be the ones checked against it
                if previous is not None:
                    phases_done = _resumable_phases(con, previous, settings=settings, source=source)
            switched_off = graph_switched_off(settings, source, paused_at_start=paused_at_start)
            if switched_off:
                raise GraphBuildCalledOff(f"{switched_off}; this build does not start")
        if base_row is not None:
            base_paths = graph_paths(settings, source, str(base_row["graph_version"]))
            if base_paths.graph_file.exists():
                try:
                    base_graph = json.loads(base_paths.graph_file.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    base_graph = None
            if base_graph is None:
                print(f"[graph] base workspace {base_paths.graph_file} missing: resolution decisions will be judged afresh", flush=True)
        # Blank graph: with no schema version at all, first extract and activate one label version with default
        # parameters, then build (2026-09-08, the user asked for full automation)
        if not incremental and not dry_run and doc_ids is None:
            from .schema_flow import ensure_schema_before_build

            source, schema_auto = ensure_schema_before_build(settings, source, stop=stop_event)
            check_stop()
            if schema_auto is not None:
                switched_off = graph_switched_off(settings, source, paused_at_start=paused_at_start)
                if switched_off:
                    raise GraphBuildCalledOff(f"{switched_off}; this build does not start")
                # the record and the phase marks were written under the configuration from before the label
                # extraction: the fingerprint becomes the one with the new labels in effect, and phases finished
                # earlier are not reused
                with db.connect(settings.state_db) as con:
                    con.execute("UPDATE graph_builds SET cache_fingerprint = ? WHERE graph_build_id = ?",
                                (graph_cache_fingerprint(settings, source), build_id))
                    db.clear_graph_phases(con, build_id)
                phases_done = set()
        # The response cache is managed per version: entries hit / written by this build are tagged with the build
        # id, and after a full build the unused ones are deleted (see LLMCache)
        LLMCache.build_tag = build_id if not dry_run else None

        schema = extraction_schema_for(source)
        # Endpoint account of the active schema version: endpoint pairs confirmed by the previous version pass the
        # merge directly this time; at the end of the build the new account is written back to the same version
        from .merge import confirmed_pairs
        from .schema_flow import active_schema_entry_for

        schema_entry = active_schema_entry_for(settings, source.kb_id) if not dry_run else None
        prior_pairs = confirmed_pairs((schema_entry or {}).get("observed") or {})
        result = {
            "dry_run": dry_run, "mode": "entity_graph", "source": source_key, "kb_id": source.kb_id,
            "schema_version_id": str(schema_entry.get("id")) if schema_entry else None,
            "schema_auto": ({"version_id": schema_auto.get("version_id"), "origin": schema_auto.get("origin")}
                            if schema_auto else None),
            "source_collection": source.collection, "graph_version": graph_version,
            "graph_build_id": build_id, "resumed_phases": sorted(phases_done),
            "build_kind": "append" if incremental else "full",
            "base_version": str(base_row["graph_version"]) if base_row is not None else None,
            "paths": {"work_dir": str(paths.work_dir), "output_dir": str(paths.output_dir), "cache_file": str(paths.cache_file)},
            "models": {step: spec.name for step, spec in specs.items()},
            "schema": {
                "entity_types": list(schema.entity_types), "predicates": list(schema.predicate_names),
                "parent_types": dict(schema.parent_types), "language": schema.language,
                "unit_chunks": int(source.graph_unit_chunks), "max_gleanings": int(source.graph_max_gleanings),
            },
            "steps": ["preflight"],
        }

        def stage(label: str) -> None:
            if dry_run or build_id == "dry-run":
                return
            with db.connect(settings.state_db) as stage_con:
                db.set_graph_build_stage(stage_con, build_id, label)

        progress = _Progress(stage)
        retry_cfg = _phase_retry_config()

        def run_phase(phase: str, fn, *, retries: int = 1):
            label = GRAPH_PHASE_LABELS[phase]
            if dry_run or build_id == "dry-run":
                return fn()
            return _run_phase(settings, build_id=build_id, phase=phase, label=label, fn=fn,
                              done=phases_done, stage=stage, retries=retries, backoff=retry_cfg["backoff"],
                              interrupted=interrupted)

        # ── 1. Prepare corpus ──
        def prepare():
            paths.work_dir.mkdir(parents=True, exist_ok=True)
            manifest, chunks, _units = prepare_graph_input(settings, source, paths=paths, q=q, write_units_file=not dry_run,
                                                          doc_ids=doc_ids)
            if not dry_run:
                with db.connect(settings.state_db) as con:
                    db.replace_graph_build_chunks(con, build_id, chunks)
                    db.record_graph_build_input(con, build_id, input_rows=int(manifest.get("documents") or 0),
                                                chunks=chunks, source_content_hash=source_snapshot_hash(chunks))
            return manifest, chunks

        prepared = run_phase("prepare_input", prepare)
        if prepared is None:
            if not paths.units_file.exists() or not paths.manifest_file.exists():
                # The workspace was garbage-collected: void the markers and prepare again
                phases_done.discard("prepare_input")
                with db.connect(settings.state_db) as con:
                    db.clear_graph_phases(con, build_id)
                phases_done.clear()
                input_manifest, build_chunks = run_phase("prepare_input", prepare)
            else:
                input_manifest = json.loads(paths.manifest_file.read_text(encoding="utf-8"))
                with db.connect(settings.state_db) as con:
                    build_chunks = db.graph_build_chunk_refs(con, build_id)
        else:
            input_manifest, build_chunks = prepared
        result["input"] = input_manifest
        result["steps"].append("prepare_input")
        if incremental:
            result["delta"] = document_delta(base_docs, build_chunks)
        if dry_run:
            result["steps"].append("dry_run_stop")
            return result

        # ── 2. Extraction ──
        def extract() -> dict[str, Any]:
            units = read_units(paths.units_file)
            cache = LLMCache(paths.cache_file)
            client = ChatClient(specs["extract"], cache=cache, stop=stop_event,
                                timeout=settings.graph_llm_timeout_seconds, workers=settings.graph_llm_concurrency,
                                circuit_fails=settings.graph_circuit_fails)
            try:
                extractor = GraphExtractor(client, schema, max_gleanings=source.graph_max_gleanings)
                det = deterministic_extractor(settings, source, units)
                det_units = [u for u in units if det.wants(u)]
                fingerprint = extraction_fingerprint(extractor.fingerprint, deterministic_units=len(det_units))
                with db.connect(settings.state_db) as con:
                    # Rows where a bad response was recorded as a "zero-entity success" do not count as cached and
                    # are re-extracted (final review F09)
                    have = db.graph_extraction_unit_ids(con, source.kb_id, fingerprint, skip_empty_malformed=True)
                # Rule-based extraction (code / structured markdown / config): no calls, computed and stored first
                det_done = 0
                for unit in det_units:
                    if unit.unit_id in have:
                        continue
                    res = det.extract(unit)
                    with db.connect(settings.state_db) as con:
                        db.save_graph_extraction(
                            con, kb_id=source.kb_id, unit_id=unit.unit_id, fingerprint=fingerprint,
                            entities=res.entities, relations=res.relations, model="deterministic", calls=0, stats=res.stats,
                        )
                    det_done += 1
                det_ids = {u.unit_id for u in det_units}
                todo = [u for u in units if u.unit_id not in have and u.unit_id not in det_ids]
                cached = len(units) - len(todo) - det_done
                stage(f"Entity extraction {cached}/{len(units)} units (cached {cached})")

                def work(unit):
                    res = extractor.extract(unit)
                    with db.connect(settings.state_db) as con:
                        db.save_graph_extraction(
                            con, kb_id=source.kb_id, unit_id=unit.unit_id, fingerprint=fingerprint,
                            entities=res.entities, relations=res.relations, model=specs["extract"].model_id,
                            calls=res.calls, stats=res.stats,
                        )
                    return res

                def on_progress(done: int, total: int) -> None:
                    progress(f"Entity extraction {cached + done}/{len(units)} units (cached {cached})", force=done == total)

                outcomes = client.run_parallel(todo, work, progress=on_progress)
                failed = [(unit, err) for unit, _, err in outcomes if err is not None]
                # Units the provider rejected for content inspection / length: only a different input would change
                # that, so it is an inherent loss of this corpus, not "extraction is broken"
                rejected = [(unit, err) for unit, err in failed if isinstance(err, LLMInputRejected)]
                broken = [(unit, err) for unit, err in failed if not isinstance(err, LLMInputRejected)]
                per_doc: dict[str, int] = {}
                for unit, _ in failed:
                    per_doc[unit.doc_id] = per_doc.get(unit.doc_id, 0) + 1
                failed_docs = sorted(per_doc)      # docs with failed units: shown on the status card (health check R3)
                for unit, err in broken[:5] + rejected[:5]:
                    print(f"[graph] unit {unit.unit_id} ({unit.rel_path}) failed: {err!r}", flush=True)
                if rejected:
                    print(f"[graph] {len(rejected)} units rejected by the provider (content inspection / length), "
                          f"documents: {sorted({u.rel_path for u, _ in rejected})[:8]}", flush=True)
                extracted = [res for _, res, err in outcomes if err is None and res is not None]
                truncated_units = sum(1 for r in extracted if int(r.stats.get("truncated", 0)) > 0)
                if truncated_units:
                    # Still incomplete at twice the budget: finished records are stored, the rest is lost; record it
                    # in stats instead of pretending completeness
                    print(f"[graph] {truncated_units} units still truncated at {extractor.max_tokens * 2} output tokens; "
                          "records after the cut are lost", flush=True)
                with db.connect(settings.state_db) as con:
                    asset_flags = db.graph_extraction_flag_counts(con, source.kb_id, fingerprint, flags=("truncated",))
                stats = {
                    "units": len(units), "cached": cached, "extracted": len(extracted),
                    "deterministic_units": len(det_units), "deterministic_routes": dict(det.route_counts),
                    "failed_units": len(failed), "rejected_units": len(rejected), "truncated_units": truncated_units,
                    # How many new calls this round got truncated vs how many units in the current cache asset ever
                    # were (the former is 0 when everything hits the cache)
                    "truncated_cached": asset_flags.get("truncated", 0),
                    "failed_documents": failed_docs,
                    "fingerprint": fingerprint, "llm": dict(client.stats),
                    "new_entities": sum(len(r.entities) for r in extracted),
                    "new_relations": sum(len(r.relations) for r in extracted),
                    "parse": {
                        k: sum(int(r.stats.get(k, 0)) for r in extracted)
                        for k in ("records", "malformed", "unknown_types", "unknown_predicates", "gleanings", "truncated")
                    },
                }
                if broken and len(broken) / max(1, len(units)) > _MAX_FAILED_UNIT_RATIO:
                    raise RuntimeError(
                        f"Entity extraction failed for {len(broken)}/{len(units)} units (over {_MAX_FAILED_UNIT_RATIO:.0%}); "
                        f"successful units are stored and a rerun only redoes the failed ones. Last error: {broken[-1][1]!r}"
                    )
                with db.connect(settings.state_db) as con:
                    total_entities = db.graph_extraction_entity_total(con, source.kb_id, fingerprint, [u.unit_id for u in units])
                stats["entities_total"] = total_entities
                if total_entities == 0:
                    raise RuntimeError("Entity extraction returned nothing: no unit produced any entity; check the model and the type table")
                return stats
            finally:
                cache.close()

        extract_stats = run_phase("extract", extract)
        if extract_stats is None:
            # Resume skipped extraction: recompute the summary from the current cache asset (unit count, hits, failed,
            # truncated) instead of leaving a bare "skipped" that turns the status card's failed-extraction count
            # into null (Codex 2026-09-14 O01)
            extract_stats = {"skipped": True, "reason": "done in a previous run"}
            try:
                extract_stats.update(_cached_extract_summary(settings, source, schema, specs, paths))
            except Exception as exc:
                extract_stats["summary_error"] = repr(exc)
        result["extract"] = extract_stats
        result["steps"].append("extract")

        # ── 3. Merge / resolution / summaries / weights / attribution ──
        def merge() -> dict[str, Any]:
            from .merge import attribute_mentions, combine_unit_kind, compute_weights, merge_extractions
            from .resolution import resolve as resolve_entities
            from .summarize import summarize_rows
            from .units import fetch_chunk_payloads

            from .deterministic import ALLOWED_ENDS as DET_ENDS, TYPES as DET_TYPES, UPPER as DET_UPPER

            units = read_units(paths.units_file)
            units_by_id = {u.unit_id: u for u in units}
            det = deterministic_extractor(settings, source, units)
            det_count = sum(1 for u in units if det.wants(u))
            fingerprint = extraction_fingerprint(
                GraphExtractor(ChatClient(specs["extract"], cache=LLMCache(None)), schema,
                               max_gleanings=source.graph_max_gleanings).fingerprint,
                deterministic_units=det_count)
            with db.connect(settings.state_db) as con:
                rows = db.load_graph_extractions(con, source.kb_id, fingerprint, [u.unit_id for u in units])
            # Cached extraction results go through the name gate and grounding filter too (2026-09-12): rule changes
            # need no re-extraction, they are cleaned in place before merging
            from .extract import entity_key, entity_name_ok, ground_records, prompt_example_names

            example_names = prompt_example_names(schema)
            gate_dropped = ungrounded_dropped = 0
            for u in units:
                row = rows.get(u.unit_id)
                if not row:
                    continue
                ents = [x for x in (row.get("entities") or []) if x.get("exact") or entity_name_ok(str(x.get("name") or ""))]
                bad = {entity_key(str(x.get("name") or "")) for x in (row.get("entities") or [])} - {entity_key(str(x.get("name") or "")) for x in ents}
                rels = [r for r in (row.get("relations") or [])
                        if entity_key(str(r.get("source") or "")) not in bad and entity_key(str(r.get("target") or "")) not in bad]
                gate_dropped += len(bad)
                context = "\n".join([u.text or "", u.document_label or "", u.section_label or "", u.rel_path or ""])
                ents, rels, n = ground_records(ents, rels, context=context, example_names=example_names)
                ungrounded_dropped += n
                row["entities"], row["relations"] = ents, rels
            # Unit kind = structural rules combined with the model's judgement (stats.unit_kind of the extraction);
            # goes into graph.json for Neo4j and recall
            unit_kinds = {
                u.unit_id: combine_unit_kind(u.kind, ((rows.get(u.unit_id) or {}).get("stats") or {}).get("unit_kind"))
                for u in units
            }
            # Upper ontology: type → one of the six classes. If the schema version's parent types are already the
            # six classes, use them directly; old versions (home-made parent types) get one mapping pass by the
            # summary model (response-cached, one call per type table), falling back to the old names on failure.
            from ..limits import UPPER_PARENTS
            from .schema import map_types_to_upper

            if schema.parent_types and all(str(v) in UPPER_PARENTS for v in schema.parent_types.values()):
                upper_parents = {t: str(schema.parent_types[t]) for t in schema.entity_types if t in schema.parent_types}
            else:
                map_cache = LLMCache(paths.cache_file)
                try:
                    upper_parents = map_types_to_upper(
                        ChatClient(specs["summarize"], cache=map_cache, stop=stop_event,
                                   timeout=settings.graph_llm_timeout_seconds, workers=1, circuit_fails=0),
                        list(schema.entity_types), dict(schema.parent_types or {}))
                finally:
                    map_cache.close()
            if det_count:
                upper_parents = {**upper_parents, **DET_UPPER}
            merge_types = tuple(schema.entity_types) + (DET_TYPES if det_count else ())
            merge_parents = {**schema.parent_types, **(DET_UPPER if det_count else {})}
            merge_ends = {**schema.allowed_ends(), **(DET_ENDS if det_count else {})}
            merged = merge_extractions(units, rows, entity_types=merge_types,
                                       parent_types=merge_parents, allowed_ends=merge_ends,
                                       unit_kinds=unit_kinds, upper_parents=upper_parents, prior_pairs=prior_pairs)
            entities, relations = merged["entities"], merged["relations"]
            merged["stats"]["bad_name_dropped"] = gate_dropped
            merged["stats"]["ungrounded_dropped"] = ungrounded_dropped
            if gate_dropped or ungrounded_dropped:
                print(f"[graph] pre-merge cleanup: {gate_dropped} malformed names, {ungrounded_dropped} prompt-example leaks dropped", flush=True)
            stats: dict[str, Any] = {"merge": merged["stats"], "fingerprint": fingerprint, "upper_parents": upper_parents}
            kind_counts: dict[str, int] = {}
            for kind in unit_kinds.values():
                kind_counts[kind] = kind_counts.get(kind, 0) + 1
            stats["units_by_kind"] = kind_counts
            if not entities:
                raise RuntimeError("No entities left after merging")
            cache = LLMCache(paths.cache_file)
            client = ChatClient(specs["summarize"], cache=cache, stop=stop_event,
                                timeout=settings.graph_llm_timeout_seconds, workers=settings.graph_llm_concurrency,
                                circuit_fails=settings.graph_circuit_fails)
            embedder = graph_embedder(settings, check_stop)
            try:
                stage("Entity resolution")
                # Incremental append: replay the last version's decisions, judge only pairs involving new entities
                prior = None
                if incremental and base_graph is not None:
                    prior = {"map": base_graph.get("resolution_map") or {}, "judged": base_graph.get("resolution_judged") or [],
                             "rejected": base_graph.get("resolution_rejected") or []}
                entities, relations, res_stats = resolve_entities(
                    client, entities, relations,
                    progress=lambda d, t: progress(f"Entity resolution {d}/{t} batches", force=d == t),
                    embed=embedder.embed, prior=prior,
                    type_words=((input_manifest or {}).get("profile") or scenario_profile(source)).get("type_words") or ())
                resolution_map = res_stats.pop("_map", {})
                resolution_judged = res_stats.pop("_judged", [])
                resolution_log = res_stats.pop("_log", [])[:5000]
                resolution_rejected = res_stats.pop("_rejected", [])[:1000]
                stats["resolution"] = res_stats
                stage("Description summaries")
                stats["summaries"] = {
                    "entities": summarize_rows(
                        client, entities, language=schema.language, name_of=lambda e: str(e.get("title") or ""),
                        progress=lambda d, t: progress(f"Description summaries · entities {d}/{t}", force=d == t)),
                    "relations": summarize_rows(
                        client, relations, language=schema.language,
                        name_of=lambda r: f"{r.get('source')} -> {r.get('target')}",
                        progress=lambda d, t: progress(f"Description summaries · relations {d}/{t}", force=d == t)),
                }
                stats["llm"] = dict(client.stats)
            finally:
                cache.close()
            stage("Computing weights and chunk attribution")
            stats["weights"] = compute_weights(entities, relations, n_units=len(units))
            payloads = fetch_chunk_payloads(q, source.collection, build_chunks)
            chunk_texts, chunk_sections, stale_chunks = attribution_texts(build_chunks, payloads)
            if stale_chunks:
                print(f"[graph] {stale_chunks} chunks changed text since the corpus was frozen; "
                      "their entities are attributed at unit level", flush=True)
            stats["attribution_stale_chunks"] = stale_chunks
            mentions = attribute_mentions(entities, units_by_id, chunk_texts, chunk_sections=chunk_sections)
            documents_axis = dict((input_manifest or {}).get("documents_axis") or {})
            graph = {
                "kb_id": source.kb_id, "graph_version": graph_version, "built_at": int(time.time()),
                "schema": result["schema"], "models": result["models"],
                "profile": dict((input_manifest or {}).get("profile") or scenario_profile(source)),
                "documents": documents_axis,
                "unit_kinds": unit_kinds,
                "entities": entities, "relations": relations, "mentions": mentions,
                # Resolution results, replayed by the next incremental append (merged key → representative key;
                # judged key pairs, including the negative ones)
                "resolution_map": resolution_map, "resolution_judged": resolution_judged,
                "resolution_log": resolution_log,     # each merged pair: origin (auto / lexical / embedding / replay) and basis, for audit
                "resolution_rejected": resolution_rejected,     # model said yes but the polarity rule / recheck blocked it (reason: polarity / recheck)
                "stats": {**stats, "units": len(units), "entities": len(entities), "relations": len(relations),
                          "mentions": len(mentions)},
            }
            write_graph_file(paths.graph_file, graph)
            return dict(graph["stats"])

        merge_stats = run_phase("merge", merge)
        if merge_stats is None:
            merge_stats = json.loads(paths.graph_file.read_text(encoding="utf-8")).get("stats") or {}
        result["graph"] = merge_stats
        result["steps"].append("merge")

        # ── 3b. Structured facts (table / list units) ──
        def facts() -> dict[str, Any]:
            from .facts import FactExtractor, document_subjects, link_facts, normalize_measurements, wants_facts
            from .llm import LLMMalformedResponse

            graph = json.loads(paths.graph_file.read_text(encoding="utf-8"))
            units = read_units(paths.units_file)
            units_by_id = {u.unit_id: u for u in units}
            kinds = {str(k): str(v) for k, v in (graph.get("unit_kinds") or {}).items()}
            det = deterministic_extractor(settings, source, units)
            det_fact_units = [u for u in units if det.route(u) in ("config", "structured_md")]
            det_ids = {u.unit_id for u in det_fact_units}
            # Units handled by the rule extractor (code, config, structured markdown) never go to the model for
            # facts: regex alternations, `str | None`, and the tables in test fixtures and prompt examples push the
            # share of pipe lines over the threshold, and fixture data and examples were published as facts
            # (2026-09-29 audit)
            todo_units = [u for u in units if not det.wants(u) and wants_facts(u, kinds.get(u.unit_id))]
            doc_paths = {u.doc_id: u.rel_path for u in units}       # document paths are evidence of subject identity (Li Hua/checkup report/…, ZK7C1049GN/…)
            subjects = document_subjects(graph.get("entities") or [], subject_types=(graph.get("profile") or {}).get("subject_types") or (),
                                         documents=doc_paths)
            cache = LLMCache(paths.cache_file)
            client = ChatClient(specs["extract"], cache=cache, stop=stop_event,
                                timeout=settings.graph_llm_timeout_seconds, workers=settings.graph_llm_concurrency,
                                circuit_fails=settings.graph_circuit_fails)
            try:
                extractor = FactExtractor(client)
                fingerprint = extraction_fingerprint(extractor.fingerprint, deterministic_units=len(det_fact_units))
                with db.connect(settings.state_db) as con:
                    have = db.graph_fact_unit_ids(con, source.kb_id, fingerprint)
                det_facts_written = 0
                for unit in det_fact_units:
                    if unit.unit_id in have:
                        continue
                    rows_det = det.facts(unit)
                    with db.connect(settings.state_db) as con:
                        db.save_graph_facts(con, kb_id=source.kb_id, unit_id=unit.unit_id, fingerprint=fingerprint,
                                            facts=rows_det, model="deterministic", calls=0,
                                            stats={"facts": len(rows_det), "builder": det.route(unit)})
                    det_facts_written += 1
                todo = [u for u in todo_units if u.unit_id not in have]
                cached = len(todo_units) - len(todo)
                stage(f"Structured facts {cached}/{len(todo_units)} table units (cached {cached})")

                def work(unit):
                    res = extractor.extract(unit, document=unit.rel_path, subjects=subjects.get(unit.doc_id, []),
                                            axis=unit.axis)
                    with db.connect(settings.state_db) as con:
                        db.save_graph_facts(con, kb_id=source.kb_id, unit_id=unit.unit_id, fingerprint=fingerprint,
                                            facts=res.facts, model=specs["extract"].model_id, calls=res.calls, stats=res.stats)
                    return res

                outcomes = client.run_parallel(
                    todo, work,
                    progress=lambda d, t: progress(f"Structured facts {cached + d}/{len(todo_units)} table units (cached {cached})", force=d == t))
                failed = [(u, err) for u, _, err in outcomes if err is not None]
                for unit, err in failed[:5]:
                    print(f"[graph] facts unit {unit.unit_id} ({unit.rel_path}) failed: {err!r}", flush=True)
                rejected_units = [u.unit_id for u, err in failed if isinstance(err, LLMInputRejected)]
                if rejected_units:
                    print(f"[graph] facts: {len(rejected_units)} units rejected by the provider (content inspection / length); "
                          "they carry no facts and do not fail the phase", flush=True)
                malformed_units = [u.unit_id for u, err in failed if isinstance(err, LLMMalformedResponse)]
                fact_units = det_fact_units + todo_units
                with db.connect(settings.state_db) as con:
                    rows = db.load_graph_facts(con, source.kb_id, fingerprint, [u.unit_id for u in fact_units])
                all_facts: list[dict[str, Any]] = []
                from .facts import apply_unit_quality
                requalified = {"ambiguous": 0, "evidence_conflicts": 0}
                for u in fact_units:
                    unit_facts = list((rows.get(u.unit_id) or {}).get("facts") or [])
                    for f in unit_facts:
                        f["unit_id"] = u.unit_id
                    # Cached facts get their quality flags recomputed from the unit's current evidence metadata (Codex
                    # re-review N04): the unit id does not cover that metadata
                    for k, v in apply_unit_quality(unit_facts, u).items():
                        requalified[k] += v
                    all_facts.extend(unit_facts)
                link_stats = link_facts(all_facts, graph.get("entities") or [], units_by_id=units_by_id)
                # Measurement facts re-homed (report corpora: measurements under lab sheets / indicators / diagnoses
                # go to the examinee, marker words are split off into flag, period conditions go into the axis)
                measure_stats = normalize_measurements(all_facts, graph.get("entities") or [], units_by_id=units_by_id,
                                                       profile=graph.get("profile") or {}, documents=doc_paths)
                # Fact skeleton (plan 5.A): property concept keys (normalization + vector neighbours + same-concept
                # judgement) and cross-document reconciliation (sequences / conflicts, annotation only)
                from .concepts import build_concepts
                from .reconcile import reconcile_facts

                embedder = graph_embedder(settings, check_stop)
                judge = ChatClient(specs["summarize"], cache=cache, stop=stop_event,
                                   timeout=settings.graph_llm_timeout_seconds, workers=1, circuit_fails=settings.graph_circuit_fails)
                stage("Normalizing property concepts")
                concepts, concept_stats = build_concepts(all_facts, embed=embedder.embed, client=judge)
                reconciled = reconcile_facts(all_facts)
                graph["specs"] = all_facts
                graph["concepts"] = concepts
                graph["conflicts"] = reconciled["conflicts"]
                stats = {
                    "concepts": concept_stats, "reconcile": reconciled["stats"], "measurements": measure_stats,
                    "units": len(todo_units), "deterministic_units": len(det_fact_units),
                    "cached": cached, "extracted": len(todo) - len(failed), "failed_units": len(failed),
                    "facts": len(all_facts), "fingerprint": fingerprint, "llm": dict(client.stats), **link_stats,
                    # Malformed responses are no longer stored, so this counts the units that failed this round (the
                    # next resume calls them again)
                    "malformed_json": len(malformed_units), "malformed_units": malformed_units[:50],
                    "failed_unit_ids": [u.unit_id for u, _ in failed][:50],
                    "rejected_units": len(rejected_units), "rejected_unit_ids": rejected_units[:50],
                    "truncated_units": [u.unit_id for u in fact_units
                                        if int((rows.get(u.unit_id) or {}).get("stats", {}).get("truncated") or 0)][:50],
                    # Units still incomplete after the split-in-half retry: the build completes as usual but is
                    # published as "facts partially complete", visible on the status card (Codex review F10)
                    "partial_units": [u.unit_id for u in fact_units
                                      if int((rows.get(u.unit_id) or {}).get("stats", {}).get("partial") or 0)][:50],
                    # The sample list keeps 50; totals are recorded separately and used by the status card / API
                    # (Codex 2026-09-13 F05: the library KB had 160 but reported 50)
                    "partial_units_total": sum(1 for u in fact_units if int((rows.get(u.unit_id) or {}).get("stats", {}).get("partial") or 0)),
                    "truncated_units_total": sum(1 for u in fact_units if int((rows.get(u.unit_id) or {}).get("stats", {}).get("truncated") or 0)),
                    "split_units": sum(1 for u in fact_units if int((rows.get(u.unit_id) or {}).get("stats", {}).get("split") or 0)),
                    "evidence_conflict_facts": sum(1 for f in all_facts if f.get("evidence_conflict")),
                    "requalified": requalified,
                }
                graph.setdefault("stats", {})["facts"] = stats
                write_graph_file(paths.graph_file, graph)
                facts_phase_gate(failed, partial_ok=_env_flag("KB_GRAPH_FACTS_PARTIAL_OK"))
                return stats
            finally:
                cache.close()

        # Failed units are not stored, so a phase retry only calls them again (on the real box malformed JSON is
        # sporadic, another round usually passes)
        facts_stats = run_phase("facts", facts, retries=max(2, int(retry_cfg["retries"])))
        if facts_stats is None:
            facts_stats = (json.loads(paths.graph_file.read_text(encoding="utf-8")).get("stats") or {}).get("facts") or {"skipped": True}
        result["facts"] = facts_stats
        result["steps"].append("facts")

        # ── 3c. View pages: subject / timeline / source / index (plan 5.C); deterministic projection + subject narration ──
        def compile_views() -> dict[str, Any]:
            from .compile import compile_pages

            graph = json.loads(paths.graph_file.read_text(encoding="utf-8"))
            units = read_units(paths.units_file)
            cache = LLMCache(paths.cache_file)
            client = ChatClient(specs["summarize"], cache=cache, stop=stop_event,
                                timeout=settings.graph_llm_timeout_seconds, workers=settings.graph_llm_concurrency,
                                circuit_fails=settings.graph_circuit_fails)
            try:
                pages, stats = compile_pages(graph, units, out_dir=paths.work_dir / "wiki", language=schema.language,
                                             client=client, stage=lambda label: progress(label),
                                             narrate=not _env_flag("KB_GRAPH_VIEWS_NO_NARRATE"))
            finally:
                cache.close()
            graph["pages"] = [{k: v for k, v in p.items() if k != "narrate_text"} for p in pages]     # the narration input is not persisted
            graph.setdefault("stats", {})["compile"] = stats
            write_graph_file(paths.graph_file, graph)
            return stats

        compile_stats = run_phase("compile", compile_views, retries=int(retry_cfg["retries"]))
        if compile_stats is None:
            compile_stats = (json.loads(paths.graph_file.read_text(encoding="utf-8")).get("stats") or {}).get("compile") or {"skipped": True}
        result["compile"] = compile_stats
        result["steps"].append("compile")

        # ── 4. Write vectors ──
        def enrich() -> dict[str, Any]:
            from .vectors import write_graph_vectors

            graph = json.loads(paths.graph_file.read_text(encoding="utf-8"))
            units_by_id = {u.unit_id: u for u in read_units(paths.units_file)}
            embedder = graph_embedder(settings, check_stop)
            # Incremental append: rows whose title/description is unchanged keep the previous vectors, no re-embedding
            reuse_from = None
            base_version = str(base_row["graph_version"]) if (incremental and base_row is not None) else None
            if base_version:
                reuse_from = {t: graph_collection_name(source.collection, t, base_version) for t in GRAPH_VECTOR_TYPES}
            return write_graph_vectors(
                q, embedder.embed, kb_id=source.kb_id, source_collection=source.collection,
                graph_version=graph_version, bundle=graph, units_by_id=units_by_id,
                vector_size=settings.embedding_dim, stage=lambda label: progress(label),
                reuse_from=reuse_from, base_version=base_version, embed_model=settings.embedding_model_id,
                check_stop=check_stop,
            )

        enrich_result = run_phase("enrich", enrich, retries=int(retry_cfg["retries"]))
        result["enrich"] = enrich_result if enrich_result is not None else {"skipped": True, "reason": "done in a previous run"}
        result["steps"].append("enrich")

        # ── 5. Graph database import ──
        chunks = build_chunks
        should_activate = activate_aliases
        should_import = activate_aliases if import_neo4j is None else bool(import_neo4j)
        active_doc_count = len({c["doc_id"] for c in chunks if c.get("doc_id")})
        input_rows = int((input_manifest or {}).get("documents") or 0)
        if settings.graph_neo4j_import_after_build and should_import:
            from .neo4j_import import import_graph_to_neo4j

            content_hash = source_snapshot_hash(chunks)

            def neo4j_import() -> dict[str, Any]:
                return import_graph_to_neo4j(
                    settings, source_key=source_key, source=source, graph_version=graph_version,
                    replace=True, activate=False, dry_run=False, batch_size=settings.graph_neo4j_import_batch_size,
                    build_record={
                        "graph_build_id": build_id, "source_key": source_key, "kb_id": source.kb_id,
                        "source_collection": source.collection, "graph_version": graph_version,
                        "status": "running", "input_rows": input_rows, "active_chunk_count": len(chunks),
                        "active_doc_count": active_doc_count, "source_content_hash": content_hash,
                        "output_dir": str(paths.output_dir),
                    },
                    require_qdrant_aliases=False,
                )

            neo4j_result = run_phase("neo4j_import", neo4j_import, retries=int(retry_cfg["retries"]))
            result["neo4j_import"] = neo4j_result if neo4j_result is not None else {"skipped": True, "reason": "done in a previous run"}
            result["steps"].append("neo4j_import")
        elif settings.graph_neo4j_import_after_build:
            result["neo4j_import"] = {"skipped": True, "reason": "aliases not activated and import_neo4j not requested"}
        else:
            result["neo4j_import"] = {"skipped": True, "reason": "GRAPH_NEO4J_IMPORT_AFTER_BUILD=false"}

        # ── 6. Switch version ──
        check_stop()                 # a stop signal swallowed in the last phase: stop before publishing
        if should_activate:
            if not dry_run and not kb_still_active(settings, source.kb_id):
                # The KB was turned off during the build (final review F02): this version is not published, the
                # record says cancelled; cache and artifacts stay so a resume can follow once it is re-enabled
                raise GraphBuildCalledOff("The knowledge base was turned off during the build; this version is not published")
            switched_off = graph_switched_off(settings, source, paused_at_start=paused_at_start)
            if switched_off:
                # the same for a graph turned off or paused: normally the process has already been stopped, and
                # getting here means it was not found at the time
                raise GraphBuildCalledOff(f"{switched_off}; this version is not published")
            stage("Switching version aliases")
            alias_result = activate_graph_aliases(q, source_collection=source.collection, graph_version=graph_version)
            result["qdrant_alias_activation"] = alias_result
            # Aliases left by legacy types (the old pipeline's communities, the removed summary tree) no longer
            # represent the current graph
            result["stale_aliases_dropped"] = drop_graph_aliases(q, source.collection, LEGACY_GRAPH_VECTOR_TYPES)
            result["steps"].append("activate_qdrant_aliases")
            if settings.graph_neo4j_import_after_build:
                from .neo4j_import import activate_neo4j_graph_version

                try:
                    result["neo4j_activation"] = activate_neo4j_graph_version(settings, source=source, graph_version=graph_version)
                except Exception:
                    try:
                        restore_graph_aliases(q, alias_result["previous"])
                        result["steps"].append("rollback_qdrant_aliases")
                    except Exception as rollback_exc:
                        result["inconsistent"] = {"qdrant_aliases": "new", "neo4j": "old", "rollback_error": repr(rollback_exc)}
                        result["steps"].append("rollback_failed_INCONSISTENT")
                        print(f"[graph] alias rollback failed; Qdrant and Neo4j versions now disagree: {rollback_exc!r}", flush=True)
                    raise
                result["steps"].append("activate_neo4j")

        # -- 7. Record the success --
        # Once the aliases and Neo4j point at this version it is the live one: record done and commit first, clean
        # up afterwards. The GC (including the grace period for searches in flight) used to run before done was
        # written; a stop during that window recorded the live version as cancelled, the next round redid the
        # work from the old baseline, and with KEEP=1 the old artifacts were already gone, so that append cost
        # about as much as a full rebuild (2026-09-29 audit).
        stage("Done")
        with db.connect(settings.state_db) as con:
            db.finish_graph_build(
                con, build_id, status="done", input_rows=input_rows, active_chunk_count=len(chunks),
                active_doc_count=active_doc_count, source_content_hash=source_snapshot_hash(chunks),
                output_dir=str(paths.output_dir), manifest=result,
            )
        with db.connect(settings.state_db) as con:
            # Write the endpoint account back to the active schema version: the next merge passes them directly
            # and re-extracting labels keeps the rules (schema_flow)
            if not dry_run and doc_ids is None and schema_entry is not None:
                from .schema_flow import record_schema_observation

                observed = ((merge_stats or {}).get("merge") or {}).get("endpoint_observed")
                if observed:
                    try:
                        written = record_schema_observation(con, source.kb_id, version_id=str(schema_entry.get("id")),
                                                            observed=observed, graph_version=graph_version)
                    except Exception as obs_exc:
                        written = False
                        print(f"[graph] schema account write failed: {obs_exc!r}", flush=True)
                    result["schema_observed"] = {
                        "version_id": str(schema_entry.get("id")), "written": bool(written),
                        "predicates": len(observed.get("edges") or {}), "confirmed": len(observed.get("confirmed") or []),
                        "prior_applied": len(((merge_stats or {}).get("merge") or {}).get("endpoint_pairs_prior") or []),
                    }
                    print(f"[graph] schema account: version={schema_entry.get('id')} written={written} "
                          f"confirmed_pairs={len(observed.get('confirmed') or [])}", flush=True)
        if should_activate:
            print(f"[graph] build published kb={source.kb_id}({source.source_root}) version={graph_version}", flush=True)

        # -- 8. Clean up -- the version is recorded as done: a failing clean-up step only goes into the manifest,
        # a stop signal ends the remaining clean-up, and neither changes the terminal state
        stopped: BaseException | None = None
        try:
            if run_gc and should_activate:
                stage("Cleaning up old versions")
                gc = gc_graph_versions(settings, source, q=q, graph_version=graph_version,
                                       grace_seconds=int(getattr(settings, "graph_gc_grace_seconds", 0) or 0))
                result.update(gc["result"])
                result["steps"].extend(gc["steps"])
                if gc["errors"]:
                    result["graph_gc_errors"] = gc["errors"]
                    result["graph_gc_error"] = "; ".join(f"{k}: {v}" for k, v in gc["errors"].items())
            # The extraction and facts caches are content-addressed, so rows left by deleted documents or changed
            # text will never hit again; clean them up after a full-corpus build
            if not dry_run and doc_ids is None and paths.units_file.exists():
                pruned = prune_unit_caches(settings, source, paths.units_file)
                result.update(pruned)
                if any(pruned.values()):
                    print(f"[graph] unit caches pruned extraction_rows={pruned['extraction_cache_pruned']} "
                          f"fact_rows={pruned['facts_cache_pruned']} kb={source.kb_id}", flush=True)
            # The response cache is managed per version: after a full build that skipped no phase (a phase skipped
            # by a resume never tags entries; phases_done holds every phase once the run ends, so use resumed_phases
            # as recorded at the start), delete the responses this version did not use; incremental appends never prune
            if not dry_run and doc_ids is None and not incremental and not result.get("resumed_phases"):
                llm_cache = LLMCache(paths.cache_file)
                try:
                    result["llm_cache_pruned"] = llm_cache.prune_unused(build_id)
                    result["llm_cache_rows"] = llm_cache.count()
                finally:
                    llm_cache.close()
                print(f"[graph] llm cache pruned rows={result['llm_cache_pruned']} kept={result['llm_cache_rows']} kb={source.kb_id}", flush=True)
        except (GraphBuildInterrupted, LLMInterrupted) as stop_exc:
            stopped = stop_exc
            result["interrupted_during_gc"] = repr(stop_exc)
            print(f"[graph] stopped during clean-up; version {graph_version} is already live and recorded as done: {stop_exc!r}",
                  flush=True)
        except Exception as post_exc:
            result["post_publish_error"] = repr(post_exc)
            print(f"[graph] clean-up after publishing failed (the build stays done): {post_exc!r}", flush=True)
        if not dry_run and build_id != "dry-run":
            try:
                with db.connect(settings.state_db) as con:
                    db.update_graph_build_manifest(con, build_id, result)
                stage("Done")                     # the stage read "Cleaning up old versions" during the GC
            except Exception as manifest_exc:
                print(f"[graph] manifest update failed: {manifest_exc!r}", flush=True)
        print(f"[graph] build done kb={source.kb_id}({source.source_root}) version={graph_version} "
              f"steps={','.join(result.get('steps') or [])}", flush=True)
        if stopped is not None:
            raise stopped        # tells the caller (the check-rebuild loop) to stop instead of building the next base
        return result
    except Exception as exc:
        if result:
            result["error"] = repr(exc)
        # A signal interruption is not a failure: record cancelled, the panel shows "stopped", and the cache and
        # extracted units stay for the next resume.
        terminal_status = "cancelled" if isinstance(exc, (GraphBuildInterrupted, LLMInterrupted)) else "failed"
        if not dry_run and build_id != "dry-run":
            with db.connect(settings.state_db) as con:
                record_build_outcome(
                    con, build_id, status=terminal_status,
                    input_rows=int((input_manifest or {}).get("documents") or 0),
                    active_chunk_count=len(build_chunks),
                    active_doc_count=len({c["doc_id"] for c in build_chunks if c.get("doc_id")}),
                    source_content_hash=source_snapshot_hash(build_chunks) if build_chunks else None,
                    output_dir=str(paths.output_dir), manifest=result, error=repr(exc),
                )
        raise
    finally:
        LLMCache.build_tag = None
        stop_event.set()
        restore_signals()
        if lock is not None:
            lock.release()


def record_build_outcome(con, build_id: str, *, status: str, manifest: dict[str, Any] | None, error: str | None,
                         **counts: Any) -> str:
    """Write the terminal state (failed / cancelled) of a build that raised or was interrupted. When the version
    is already live and recorded as done -- it was the clean-up afterwards that went wrong -- only the manifest
    is updated and the terminal state stays: the aliases and Neo4j point at it, and recording it as cancelled
    would make the next round redo the work from the old baseline. Returns the resulting status."""
    row = con.execute("SELECT status, input_rows, active_chunk_count, active_doc_count, source_content_hash "
                      "FROM graph_builds WHERE graph_build_id = ?", (build_id,)).fetchone()
    if row is not None and str(row["status"]) == "done":
        db.update_graph_build_manifest(con, build_id, manifest)
        return "done"
    if row is not None and not counts.get("source_content_hash"):
        # stopped before the corpus was prepared (blocked by the graph switch at the start, stopped during label
        # extraction): the record keeps the corpus figures it already had. Whether a resume of the same version can
        # reuse the phases the previous run finished is checked against the corpus fingerprint; wiped, every
        # finished phase would run again next time
        counts = {**counts, "input_rows": int(row["input_rows"] or 0),
                  "active_chunk_count": int(row["active_chunk_count"] or 0),
                  "active_doc_count": int(row["active_doc_count"] or 0),
                  "source_content_hash": row["source_content_hash"]}
    db.finish_graph_build(con, build_id, status=status, manifest=manifest, error=error, **counts)
    return status


def _cached_extract_summary(settings: Settings, source: KBSource, schema: ExtractionSchema, specs: dict[str, LLMSpec],
                            paths: GraphPaths) -> dict[str, Any]:
    """When a resume skips the extraction phase, compute an extraction summary from the current cache asset: same
    field names as extract() writes (units / cached / failed_units / failed_documents / truncated_cached /
    fingerprint); extracted and truncated_units count "new calls this round", which is 0 when a resume made none."""
    from .units import read_units

    units = read_units(paths.units_file)
    det = deterministic_extractor(settings, source, units)
    det_units = [u for u in units if det.wants(u)]
    fingerprint = extraction_fingerprint(
        GraphExtractor(ChatClient(specs["extract"], cache=LLMCache(None)), schema, max_gleanings=source.graph_max_gleanings).fingerprint,
        deterministic_units=len(det_units))
    with db.connect(settings.state_db) as con:
        have = db.graph_extraction_unit_ids(con, source.kb_id, fingerprint, skip_empty_malformed=True)
        flags = db.graph_extraction_flag_counts(con, source.kb_id, fingerprint, flags=("truncated",))
    failed = [u for u in units if u.unit_id not in have]
    return {
        "units": len(units), "cached": len(units) - len(failed), "extracted": 0,
        "deterministic_units": len(det_units), "failed_units": len(failed), "rejected_units": 0, "truncated_units": 0,
        "truncated_cached": flags.get("truncated", 0), "failed_documents": sorted({u.doc_id for u in failed}),
        "fingerprint": fingerprint,
    }


def _prune_graph_workspaces(settings: Settings, source: KBSource, *, keep_version: str, retention_days: int,
                            keep_latest: int | None = None, discard_versions: Iterable[str] = (),
                            protect_versions: Iterable[str] = (), dry_run: bool = False) -> dict[str, Any]:
    """Delete this base's superseded per-version workspace directories with the same rule as the Qdrant /
    Neo4j GC: with ``keep_latest`` the ``keep_version`` (active) directory plus the ``keep_latest - 1`` newest
    other directories are kept unconditionally and everything older is deleted, age plays no part; without
    ``keep_latest`` directories whose mtime is past the retention are deleted. ``keep_version`` is always kept;
    the LLM cache is shared across versions and untouched. Directories are ranked by the timestamp in the
    version id (mtime when the name is not a version id).
    ``discard_versions``: cancelled / failed / rolled-back versions take no keep slot and are deleted outright.
    ``protect_versions``: the version kept for a resume is neither deleted nor counted against the keep window."""
    short = graph_collection_short_name(source.collection)
    cutoff = time.time() - max(1, retention_days) * 86400
    protect = {str(v) for v in protect_versions if str(v)}
    discard = {str(v) for v in discard_versions if str(v)} - {keep_version} - protect
    removed: list[str] = []
    parent = settings.graph_work_dir / "work" / short
    if parent.is_dir():
        dirs: list[tuple[float, float, Path]] = []
        for version_dir in parent.iterdir():
            if not version_dir.is_dir():
                continue
            try:
                mtime = version_dir.stat().st_mtime
            except OSError:
                continue
            stamp = graph_version_timestamp(version_dir.name)
            dirs.append((float(stamp) if stamp is not None else mtime, mtime, version_dir))
        dirs.sort(key=lambda item: -item[0])
        slots: int | None = None
        if keep_latest is not None:
            slots = max(0, int(keep_latest) - (1 if keep_version and (parent / keep_version).is_dir() else 0))
        rank = 0
        for _stamp, mtime, version_dir in dirs:
            if version_dir.name == keep_version or version_dir.name in protect:
                continue
            if version_dir.name in discard:
                if not dry_run:
                    shutil.rmtree(version_dir, ignore_errors=True)
                removed.append(str(version_dir))
                continue
            if slots is not None:
                keep = rank < slots
                rank += 1
            else:
                keep = mtime >= cutoff
            if keep:
                continue
            if not dry_run:
                shutil.rmtree(version_dir, ignore_errors=True)
            removed.append(str(version_dir))
    return {"removed_dirs": removed, "retention_days": retention_days, "keep_latest": keep_latest,
            "discard_versions": sorted(discard), "protect_versions": sorted(protect), "dry_run": dry_run}


def gc_graph_versions(settings: Settings, source: KBSource, *, q: Any, graph_version: str,
                      keep_latest: int | None = None, discard: set[str] | None = None,
                      protect: set[str] | None = None,
                      grace_seconds: int = 0, dry_run: bool = False) -> dict[str, Any]:
    """One base's version GC, shared by the end of a build, a rollback and the nightly graph-gc: Qdrant
    collections, workspace directories and Neo4j versions are cleaned with the same "graph_version (active)
    plus the keep_latest - 1 newest others" rule, then the build records are pruned. The four steps are
    independent: one failing does not drag the others down (2026-09-13: deleting old Neo4j versions hit the
    transaction memory limit). When ``discard`` is not given it is read from the records (cancelled / failed /
    rolled-back versions).
    ``protect``: the version kept for a resume, neither deleted nor counted against the keep window (given by
    the nightly GC; the end of a build and a rollback do not give it, the newest record is then the successful
    version itself).
    ``grace_seconds``: wait after switching versions so searches still running at the switch finish with the
    old collections before they are deleted.
    A stop signal (GraphBuildInterrupted / LLMInterrupted) is not swallowed as a failed step; it is passed on
    to the caller.
    Returns {"result": per-step results, "steps": step names to record, "errors": failed steps}."""
    keep = max(1, int(keep_latest if keep_latest is not None else getattr(settings, "graph_gc_keep_versions", 2) or 2))
    result: dict[str, Any] = {}
    steps: list[str] = []
    errors: dict[str, str] = {}

    def gc_step(name: str, fn: Callable[[], Any], *, step: bool = True) -> None:
        try:
            result[name] = fn()
            if step:
                steps.append(name)
        except (GraphBuildInterrupted, LLMInterrupted):
            raise
        except Exception as gc_exc:
            errors[name] = repr(gc_exc)
            print(f"[graph] gc step {name} failed: {gc_exc!r}", flush=True)

    if discard is None:
        discard = set()
        try:
            with db.connect(settings.state_db) as con:
                discard = db.unsuccessful_graph_versions(con, source.kb_id) - {graph_version}
        except Exception as gc_exc:
            errors["discard"] = repr(gc_exc)
    protect = {str(v) for v in (protect or ()) if str(v)} - {graph_version}
    discard = set(discard) - protect
    result["graph_gc_discard"] = sorted(discard)
    if protect:
        result["graph_gc_protect"] = sorted(protect)
        print(f"[graph] gc: resumable versions kept outside the quota: {sorted(protect)}", flush=True)
    if discard:
        print(f"[graph] gc: unsuccessful versions discarded outright: {sorted(discard)}", flush=True)
    if grace_seconds > 0 and not dry_run:
        print(f"[graph] gc: waiting {grace_seconds}s for searches in flight before deleting superseded versions", flush=True)
        time.sleep(grace_seconds)
    gc_step("graph_gc", lambda: delete_old_graph_collections(
        q, source_collections=[source.collection], retention_days=settings.graph_gc_retention_days, dry_run=dry_run,
        keep_latest=keep, discard_versions=discard, protect_versions=protect))
    gc_step("workspace_gc", lambda: _prune_graph_workspaces(
        settings, source, keep_version=graph_version, retention_days=settings.graph_gc_retention_days,
        keep_latest=keep, discard_versions=discard, protect_versions=protect, dry_run=dry_run), step=False)
    if settings.graph_neo4j_import_after_build:
        from .neo4j_import import delete_old_neo4j_graph_versions

        gc_step("neo4j_graph_gc", lambda: delete_old_neo4j_graph_versions(
            settings, sources=[source], retention_days=settings.neo4j_graph_retention_days, dry_run=dry_run,
            keep_latest=keep, discard_versions=discard, protect_versions=protect))
    # The kept versions (active, inside the keep window, kept for a resume) keep their records too: the status
    # card still has their sizes and figures after a rollback to them
    kept = {graph_version} | protect | {
        str(r.get("graph_version") or "") for r in (result.get("graph_gc") or {}).get("skipped_collections", [])
        if r.get("reason") in ("aliased", "active_version", "within_keep_latest", "within_retention", "resumable")}

    def prune_records() -> int:
        if dry_run:
            return 0
        with db.connect(settings.state_db) as con:
            return db.prune_graph_builds(con, source.kb_id, keep_versions=kept)

    gc_step("build_records_pruned", prune_records, step=False)
    return {"result": result, "steps": steps, "errors": errors}


def _install_build_signal_handlers(stop_event: threading.Event | None = None,
                                   interrupted: threading.Event | None = None):
    """Turn SIGTERM/SIGINT into an exception handled by build_graph's failure path; also set the stop event so
    worker threads waiting on the LLM exit as soon as possible. Returns a function that restores the original
    handlers.
    interrupted is set only here (stop is also set by a circuit break or a thread-pool error): when the raised
    exception is swallowed on its way, the checkpoints recognise the stop signal by it."""
    def handler(signum, _frame):
        if interrupted is not None:
            interrupted.set()
        if stop_event is not None:
            stop_event.set()
        raise GraphBuildInterrupted(f"Graph build interrupted by signal {signum}")

    previous: dict[int, Any] = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            previous[sig] = signal.signal(sig, handler)
        except (ValueError, OSError):  # cannot register outside the main thread
            pass

    def restore() -> None:
        for sig, old in previous.items():
            try:
                signal.signal(sig, old)
            except (ValueError, OSError):
                pass

    return restore


# ── Rebuild policy, append decision and baseline adoption ────────────────

def _chunk_keys(entries: Iterable[tuple[str, ...]]) -> dict[str, tuple[str, str]]:
    """{point_id: (content_version, text_sha)}; old ledgers / old callers only have 2-tuples, whose fingerprint is
    recorded as an empty string."""
    out: dict[str, tuple[str, str]] = {}
    for entry in entries:
        point = str(entry[0] or "")
        version = str(entry[1] or "") if len(entry) > 1 else ""
        sha = str(entry[2] or "") if len(entry) > 2 else ""
        out[point] = (version, sha)
    return out


def _doc_changed(base: dict[str, tuple[str, str]], cur: dict[str, tuple[str, str]]) -> bool:
    if set(base) != set(cur):
        return True
    for point, (version, sha) in cur.items():
        base_version, base_sha = base[point]
        if version != base_version:
            return True
        if sha and base_sha and sha != base_sha:      # compare only if both have one; old ledgers lack it, not a change
            return True
    return False


def document_delta(base_docs: dict[str, set[tuple[str, ...]]], chunks: list[dict[str, str]]) -> dict[str, Any]:
    """Document-level delta: the chunk ledger frozen by the baseline version vs the active chunks now. A document
    counts as modified when any of its chunk point_id set, content_version or text fingerprint text_sha differs:
    chunks replaced wholesale after a re-parse is one case; profile not upgraded, chunk ids unchanged and only the
    text changed (as in the chunking overhaul of 2026-09-05) is another. Fingerprints are only compared when both
    sides recorded them, so old ledgers without fingerprints do not raise false positives."""
    current: dict[str, set[tuple[str, str, str]]] = {}
    for c in chunks:
        current.setdefault(str(c.get("doc_id") or ""), set()).add(
            (str(c.get("point_id") or ""), str(c.get("content_version") or ""), str(c.get("text_sha") or "")))
    added = sorted(d for d in current if d not in base_docs)
    removed = sorted(d for d in base_docs if d not in current)
    modified = sorted(d for d in current if d in base_docs and _doc_changed(_chunk_keys(base_docs[d]), _chunk_keys(current[d])))
    base_points = {(e[0], e[1] if len(e) > 1 else "") for group in base_docs.values() for e in group}
    cur_points = {(e[0], e[1]) for group in current.values() for e in group}
    return {
        "added_docs": added, "removed_docs": removed, "modified_docs": modified,
        "unchanged_docs": len(current) - len(added) - len(modified),
        "new_chunks": len(cur_points - base_points), "removed_chunks": len(base_points - cur_points),
        "documents": len(current), "base_documents": len(base_docs),
    }


def llm_ready(settings: Settings, source: KBSource) -> str | None:
    """Whether the build's model slots are all set; if not, returns a plain-language reason (the timer check skips
    quietly on it instead of erroring every round)."""
    try:
        resolve_llm_specs(settings, source, GRAPH_LLM_STEPS)
    except RuntimeError as exc:
        return str(exc)
    return None


def evaluate_append(settings: Settings, *, source_key: str, source: KBSource,
                    ignore_auto_flag: bool = False) -> dict[str, Any]:
    """Whether new / changed documents should be appended incrementally to the current graph. Sits beside
    evaluate_rebuild: a full rebuild looks at the policy conditions, an append only asks "has the corpus changed at
    document level relative to the current version" (deleted documents trigger it too and leave the graph on
    append). When due, includes base_version and the delta; otherwise a reason. ignore_auto_flag: a manual trigger
    ignores the auto switch."""
    out: dict[str, Any] = {"source": source_key, "kb_id": source.kb_id,
                           "graph_enabled": bool(source.graph_enabled), "due": False}
    if not source.graph_enabled:
        return {**out, "reason": "disabled"}
    if source.graph_paused:
        return {**out, "reason": "paused_by_operator"}
    if not ignore_auto_flag and not source.graph_auto_append:
        return {**out, "reason": "auto_append_off"}
    chunks = active_source_chunks(settings, source)
    if not chunks:
        return {**out, "reason": "no_active_content"}
    db.init_db(settings.state_db)
    with db.connect(settings.state_db) as con:
        base = db.latest_successful_graph_build(con, source_key)
        if base is None:
            return {**out, "reason": "no_successful_build"}      # the first build belongs to the rebuild policy
        base_docs = db.graph_build_doc_chunks(con, str(base["graph_build_id"]))
    out["base_version"] = str(base["graph_version"])
    delta = document_delta(base_docs, chunks)
    out["delta"] = delta
    out["changed"] = bool(delta["added_docs"] or delta["removed_docs"] or delta["modified_docs"])
    try:
        same_config = str(base["cache_fingerprint"] or "") == graph_cache_fingerprint(settings, source)
    except RuntimeError as exc:
        return {**out, "reason": "llm_not_configured", "detail": str(exc)}
    if not same_config:
        # Config changed: the previous version's extraction and resolution cannot be reused. With new documents the
        # timer check turns this into a full rebuild (cli check-rebuild)
        return {**out, "reason": "config_changed_needs_full_rebuild"}
    if not out["changed"]:
        return {**out, "reason": "source_unchanged"}
    return {**out, "due": True}


def check_kind(decision: dict[str, Any]) -> str:
    """Compress one check-rebuild round's verdict for one KB into a single word: full / append / deferred / skipped /
    none / error. The status card phrases it in plain words from this (health check D7)."""
    if decision.get("error"):
        return "error"
    append = decision.get("append") or {}
    skipped = decision.get("build_skipped") or append.get("build_skipped")
    if skipped:
        return "deferred" if skipped.get("retry") else "skipped"
    if decision.get("build") is not None:
        return "full"
    if append.get("build") is not None:
        return "append"
    return "none"


def stage_weights_from_phases(started_at: int, timestamps: list[tuple[str, int]]) -> dict[str, float]:
    """Weight the progress bar by the actual duration (seconds) of each phase in the last completed build (health
    check D8). Phase markers only carry completion times: each phase = its completion − the previous completion,
    the first counted from the build start. With fewer than three phases of data nothing is returned (phases
    skipped by a resume get no new marker) and the frontend falls back to fixed weights."""
    weights: dict[str, float] = {}
    prev = int(started_at or 0)
    done = dict(timestamps)
    for phase, _label in GRAPH_PHASES:
        ts = done.get(phase)
        if ts is None or ts < prev:
            continue
        weights[phase] = float(max(1, ts - prev))
        prev = ts
    return weights if len(weights) >= 3 else {}


def merge_stage_weights(candidates: list[dict[str, float]]) -> dict[str, float]:
    """Take the per-phase maximum across several builds: the run whose extraction hit the cache took seconds, and
    weighting by it alone would make the next from-scratch build's progress bar jump through "Entity extraction"
    in one step. The slowest run in history is the one that resembles the real pace."""
    merged: dict[str, float] = {}
    for weights in candidates:
        for phase, seconds in (weights or {}).items():
            merged[phase] = max(merged.get(phase, 0.0), float(seconds))
    return merged if len(merged) >= 3 else {}


def evaluate_rebuild(settings: Settings, *, source_key: str, source: KBSource) -> dict[str, Any]:
    if not source.graph_enabled:
        return {"source": source_key, "graph_enabled": False, "due": False, "reason": "disabled"}
    if source.graph_paused:
        # Paused means "do not look": checked before scanning the KB, and no active_chunk_count is reported.
        return {"source": source_key, "graph_enabled": True, "due": False, "reason": "paused_by_operator"}

    chunks = active_source_chunks(settings, source)
    if not chunks:
        return {"source": source_key, "graph_enabled": True, "due": False, "reason": "no_active_content", "active_chunk_count": 0}
    current_ids = {str(chunk["point_id"]) for chunk in chunks if chunk.get("point_id")}
    now = int(time.time())

    with db.connect(settings.state_db) as con:
        latest = db.latest_successful_graph_build(con, source_key)
        if latest is None:
            return {"source": source_key, "graph_enabled": True, "due": True, "reason": "no_successful_build",
                    "active_chunk_count": len(current_ids)}
        # Baseline = the last full version. Appends do not count: otherwise each append would reset the "new
        # content" counter and the clock, and a full rebuild would never come around.
        baseline = db.latest_successful_graph_build(con, source_key, kind="full") or latest
        baseline_ids = db.graph_build_point_ids(con, str(baseline["graph_build_id"]))
        appends_since_full = db.count_graph_builds_since(
            con, source.kb_id, kind="append", since_ts=int(baseline["finished_at"] or baseline["started_at"] or 0))

    policy = source.graph_rebuild_policy
    operator = str(policy.operator or "or").strip().lower()
    if operator not in {"or", "and"}:
        # The console dropdown cannot produce other values, only the CLI / manual DB edits can; one KB's bad value
        # must not stop the whole check-rebuild round for the other KBs (health check B12): treat it as or and log
        print(f"[graph] {source_key}: rebuild operator {policy.operator!r} is neither 'or' nor 'and'; treating it as 'or'", flush=True)
        operator = "or"

    conditions: list[tuple[str, bool, dict[str, Any]]] = []
    if policy.interval_days is not None:
        finished_at = int(baseline["finished_at"] or baseline["started_at"] or 0)
        elapsed_days = (now - finished_at) / 86400
        conditions.append(("interval", elapsed_days >= policy.interval_days,
                           {"interval_days": policy.interval_days, "elapsed_days": round(elapsed_days, 3), "last_finished_at": finished_at}))
    new_ids = current_ids - baseline_ids
    if policy.new_chunk_count is not None:
        conditions.append(("new_chunk_count", len(new_ids) >= policy.new_chunk_count,
                           {"threshold": policy.new_chunk_count, "new_chunks": len(new_ids),
                            "active_chunk_count": len(current_ids), "baseline_chunk_count": len(baseline_ids)}))
    if policy.new_chunk_ratio is not None:
        denominator = len(baseline_ids & current_ids)
        ratio = (len(new_ids) / denominator) if denominator else (1.0 if new_ids else 0.0)
        conditions.append(("new_chunk_ratio", ratio >= policy.new_chunk_ratio,
                           {"threshold": policy.new_chunk_ratio, "ratio": round(ratio, 6), "new_chunks": len(new_ids),
                            "active_baseline_denominator": denominator,
                            "deleted_or_inactive_baseline_chunks": len(baseline_ids - current_ids),
                            "active_chunk_count": len(current_ids), "baseline_chunk_count": len(baseline_ids)}))
    if not conditions:
        return {"source": source_key, "graph_enabled": True, "due": False, "reason": "no_policy",
                "latest_graph_version": latest["graph_version"], "baseline_graph_version": baseline["graph_version"],
                "appends_since_full": appends_since_full, "active_chunk_count": len(current_ids)}

    due = all(item[1] for item in conditions) if operator == "and" else any(item[1] for item in conditions)
    unchanged_reason = None
    if due:
        # Skip only when the corpus has not changed one bit since the last **full** version; unchanged since the
        # latest append does not count, since appends follow the corpus anyway
        previous_hash = str(baseline["source_content_hash"] or "")
        if previous_hash and previous_hash == source_snapshot_hash(chunks):
            due = False
            unchanged_reason = "source_unchanged_since_last_build"
    return {
        "source": source_key, "graph_enabled": True, "due": due,
        **({"skipped_reason": unchanged_reason} if unchanged_reason else {}),
        "operator": operator, "latest_graph_version": latest["graph_version"],
        "latest_graph_build_id": latest["graph_build_id"],
        "baseline_graph_version": baseline["graph_version"], "appends_since_full": appends_since_full,
        "conditions": [{"name": name, "due": cond_due, **details} for name, cond_due, details in conditions],
    }


def adopt_current_graph(settings: Settings, *, source_key: str, source: KBSource, graph_version: str | None = None) -> dict[str, Any]:
    if not source.graph_enabled:
        raise ValueError(f"source {source_key!r} has graph_enabled=false")

    q = qdrant_client(settings.qdrant_url, settings.qdrant_api_key)
    aliases = {item.alias_name: item.collection_name for item in q.get_aliases().aliases}
    inferred_versions: dict[str, str] = {}
    targets: dict[str, str | None] = {}
    for graph_type in GRAPH_VECTOR_TYPES:
        alias = graph_collection_alias(source.collection, graph_type)
        target = aliases.get(alias)
        targets[graph_type] = target
        if target:
            parsed = parse_graph_collection_name(target)
            if parsed:
                inferred_versions[graph_type] = str(parsed["graph_version"])

    selected_version = graph_version
    if selected_version is None:
        versions = set(inferred_versions.values())
        if len(versions) != 1:
            raise RuntimeError(f"cannot infer one graph version from aliases for {source.collection}: {targets}")
        selected_version = versions.pop()

    missing = [t for t in GRAPH_VECTOR_TYPES if t not in GRAPH_OPTIONAL_TYPES
               and targets.get(t) != graph_collection_name(source.collection, t, selected_version)]
    if missing:
        raise RuntimeError(f"graph aliases do not all point to version {selected_version}: missing/mismatched={missing}")

    chunks = active_source_chunks(settings, source)
    paths = graph_paths(settings, source, selected_version)
    manifest: dict[str, Any] = {}
    if paths.manifest_file.exists():
        try:
            manifest = json.loads(paths.manifest_file.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    with db.connect(settings.state_db) as con:
        existing = con.execute(
            "SELECT graph_build_id FROM graph_builds WHERE source_collection = ? AND graph_version = ? LIMIT 1",
            (source.collection, selected_version),
        ).fetchone()
        if existing:
            build_id = str(existing["graph_build_id"])
        else:
            build_id = db.begin_graph_build(con, source_key=source_key, kb_id=source.kb_id,
                                            source_collection=source.collection, graph_version=selected_version)
        db.replace_graph_build_chunks(con, build_id, chunks)
        db.finish_graph_build(
            con, build_id, status="done", input_rows=int(manifest.get("documents") or 0),
            active_chunk_count=len(chunks),
            active_doc_count=len({chunk["doc_id"] for chunk in chunks if chunk.get("doc_id")}),
            source_content_hash=source_snapshot_hash(chunks), output_dir=str(paths.output_dir),
            manifest={"adopted": True, "aliases": targets, "input_manifest": manifest},
        )
    return {
        "source": source_key, "kb_id": source.kb_id, "source_collection": source.collection,
        "graph_version": selected_version, "graph_build_id": build_id,
        "active_chunk_count": len(chunks),
        "active_doc_count": len({chunk["doc_id"] for chunk in chunks if chunk.get("doc_id")}),
        "aliases": targets,
    }


def rollback_graph_version(settings: Settings, *, source_key: str, source: KBSource, graph_version: str,
                           q: Any = None, grace_seconds: int | None = None, force: bool = False) -> dict[str, Any]:
    """Roll the live graph back to an earlier version that is still kept. Switches the Qdrant aliases,
    activates the Neo4j version (re-imported from the workspace when its projection is gone), records it
    (the target becomes the latest successful build; newer successful versions are marked rolled_back) and
    finally deletes the rejected version together with anything beyond the keep window, with the usual GC
    rule. The target's workspace (graph.json) must still exist: incremental appends replay from it.
    The target has to be a version that was built (its record is done or rolled_back): a half-finished version
    left by a paused / failed build, or a version without a record, is refused unless ``force`` is given. Before
    the switch the point counts of its collections and its Neo4j projection are compared with the figures
    recorded when it was built; a projection that does not match is re-imported from the workspace.
    Shares the build lock with builds: refused while a build is running."""
    if not source.graph_enabled:
        raise ValueError(f"source {source_key!r} has graph_enabled=false")
    graph_version = str(graph_version or "").strip()
    if not graph_version:
        raise ValueError("graph_version is required")
    guard = GraphBuildLock(settings)          # same lock as a build: never switch versions under a running build
    guard.acquire()
    try:
        q = q or qdrant_client(settings.qdrant_url, settings.qdrant_api_key)
        current_versions: set[str] = set()
        for target in graph_alias_targets(q, source.collection).values():
            parsed = parse_graph_collection_name(str(target)) if target else None
            if parsed:
                current_versions.add(str(parsed["graph_version"]))
        current = next(iter(current_versions)) if len(current_versions) == 1 else None
        if graph_version in current_versions:
            raise ValueError(f"{graph_version} is already the active graph version of {source.kb_id}")
        required = [t for t in GRAPH_VECTOR_TYPES if t not in GRAPH_OPTIONAL_TYPES]
        missing = [graph_collection_name(source.collection, t, graph_version) for t in required
                   if not collection_exists(q, graph_collection_name(source.collection, t, graph_version))]
        if missing:
            raise ValueError(f"cannot roll back {source.kb_id} to {graph_version}: graph collection(s) missing: {missing}")
        paths = graph_paths(settings, source, graph_version)
        if not paths.graph_file.exists():
            raise ValueError(f"cannot roll back {source.kb_id} to {graph_version}: workspace {paths.graph_file} is gone "
                             "(incremental appends replay from it)")
        target = _rollback_target_check(settings, source, q, graph_version=graph_version, force=force)
        result: dict[str, Any] = {"source": source_key, "kb_id": source.kb_id, "source_collection": source.collection,
                                  "from_graph_version": current, "graph_version": graph_version, "steps": [],
                                  "target": target["summary"]}
        use_neo4j = bool(getattr(settings, "graph_neo4j_import_after_build", False))
        if use_neo4j:
            from .neo4j_import import (activate_neo4j_graph_version, count_version_nodes, counts_match,
                                       import_graph_to_neo4j, neo4j_counts, neo4j_driver)

            driver = neo4j_driver(settings)
            try:
                nodes = count_version_nodes(driver, source.kb_id, graph_version)
                expected = target["neo4j_expected"]
                intact = bool(nodes > 0 and expected and counts_match(expected, neo4j_counts(driver, source.kb_id, graph_version)))
            finally:
                driver.close()
            if not intact:
                # The target's Neo4j projection is gone, or does not match the recorded counts (a half-written
                # projection): re-import it from the workspace, without activating
                result["neo4j_reimport"] = import_graph_to_neo4j(
                    settings, source_key=source_key, source=source, graph_version=graph_version,
                    replace=True, activate=False, require_qdrant_aliases=False)
                result["steps"].append("neo4j_reimport")
        alias_result = activate_graph_aliases(q, source_collection=source.collection, graph_version=graph_version)
        result["qdrant_alias_activation"] = alias_result
        result["steps"].append("activate_qdrant_aliases")
        if use_neo4j:
            try:
                result["neo4j_activation"] = activate_neo4j_graph_version(settings, source=source, graph_version=graph_version)
            except Exception:
                try:
                    restore_graph_aliases(q, alias_result["previous"])
                    result["steps"].append("rollback_qdrant_aliases")
                except Exception as rollback_exc:
                    result["inconsistent"] = {"qdrant_aliases": "new", "neo4j": "old", "rollback_error": repr(rollback_exc)}
                    print(f"[graph] alias restore failed during rollback; Qdrant and Neo4j versions now disagree: {rollback_exc!r}", flush=True)
                raise
            result["steps"].append("activate_neo4j")
        chunks = active_source_chunks(settings, source)
        with db.connect(settings.state_db) as con:
            result["records"] = _record_rollback(con, settings, source, source_key=source_key, graph_version=graph_version,
                                                 rolled_back_from=current, chunks=chunks, output_dir=str(paths.output_dir))
        result["steps"].append("record")
        grace = int(grace_seconds if grace_seconds is not None else getattr(settings, "graph_gc_grace_seconds", 0) or 0)
        gc = gc_graph_versions(settings, source, q=q, graph_version=graph_version, grace_seconds=grace)
        result.update(gc["result"])
        result["steps"].extend(gc["steps"])
        if gc["errors"]:
            result["graph_gc_errors"] = gc["errors"]
        print(f"[graph] rollback {source.kb_id}: {current} -> {graph_version}; rejected={result['records'].get('rejected')}", flush=True)
        return result
    finally:
        guard.release()


def _rollback_target_check(settings: Settings, source: KBSource, q: Any, *, graph_version: str, force: bool) -> dict[str, Any]:
    """Whether the rollback target is a version that was built and is still complete. A record that is not done
    / rolled_back (paused, failed, still running), or no record at all, is refused unless ``force`` is given:
    with KEEP=1 such half-finished versions are the only ones a rollback could reach, and switching to one also
    marks the good live version rolled_back and deletes it (2026-09-29 audit). Collections whose point count
    differs from the figure recorded at build time are always refused, forced or not: they were written only
    in part. Returns {"summary": goes into the result, "neo4j_expected": the Neo4j counts recorded at build
    time, empty when none were recorded}."""
    with db.connect(settings.state_db) as con:
        row = db.graph_build_by_version(con, source.collection, graph_version)
    status = str(row["status"]) if row is not None else ""
    if row is None and not force:
        raise ValueError(f"cannot roll back {source.kb_id} to {graph_version}: no build record for this version; "
                         "pass --force to adopt it as built")
    if row is not None and status not in ("done", "rolled_back") and not force:
        raise ValueError(f"cannot roll back {source.kb_id} to {graph_version}: its build record is {status!r}, "
                         "not a finished build; pass --force only if you are sure the version is complete")
    manifest: dict[str, Any] = {}
    if row is not None:
        try:
            manifest = json.loads(row["manifest_json"] or "{}") or {}
        except (TypeError, ValueError):
            manifest = {}
    recorded = ((manifest.get("enrich") or {}).get("collections") or {}) if isinstance(manifest.get("enrich"), dict) else {}
    counts: dict[str, dict[str, Any]] = {}
    mismatched: list[str] = []
    for graph_type in GRAPH_VECTOR_TYPES:
        collection = graph_collection_name(source.collection, graph_type, graph_version)
        if not collection_exists(q, collection):
            continue                      # required types were checked by the caller; an optional type without a collection had no rows
        actual = int(q.count(collection_name=collection, exact=True).count)
        want = (recorded.get(graph_type) or {}).get("points") if isinstance(recorded.get(graph_type), dict) else None
        counts[graph_type] = {"points": actual, "recorded": want}
        if want is not None and int(want) != actual:
            mismatched.append(f"{collection}: points={actual} recorded={want}")
        elif want is None and graph_type not in GRAPH_OPTIONAL_TYPES and actual <= 0:
            mismatched.append(f"{collection}: empty")
    if mismatched:
        raise ValueError(f"cannot roll back {source.kb_id} to {graph_version}: graph collection(s) incomplete: {mismatched}")
    expected = (manifest.get("neo4j_import") or {}).get("expected_counts") if isinstance(manifest.get("neo4j_import"), dict) else None
    return {"summary": {"record_status": status or None, "forced": bool(force), "collections": counts},
            "neo4j_expected": dict(expected) if isinstance(expected, dict) else {}}


def _record_rollback(con, settings: Settings, source: KBSource, *, source_key: str, graph_version: str,
                     rolled_back_from: str | None, chunks: list[dict[str, str]], output_dir: str) -> dict[str, Any]:
    """Build records after a rollback: the target becomes the latest successful build (finished_at bumped to
    now, so the append baseline follows it); newer successful versions are marked rolled_back, which makes
    them non-resumable and deleted at the next GC. When the target has no record any more (pruned long ago)
    one is created under the current configuration: the fingerprint of the current config, the corpus snapshot
    of the active chunks now."""
    now = int(time.time())
    row = con.execute("SELECT * FROM graph_builds WHERE source_collection = ? AND graph_version = ? LIMIT 1",
                      (source.collection, graph_version)).fetchone()
    target_ts = int(row["finished_at"] or row["started_at"] or 0) if row is not None else None
    if target_ts is not None:
        newer = con.execute(
            "SELECT graph_build_id, graph_version FROM graph_builds WHERE kb_id = ? AND status = 'done' "
            "AND graph_version != ? AND COALESCE(finished_at, started_at) > ?",
            (source.kb_id, graph_version, target_ts)).fetchall()
    else:
        newer = con.execute(
            "SELECT graph_build_id, graph_version FROM graph_builds WHERE kb_id = ? AND status = 'done' AND graph_version != ?",
            (source.kb_id, graph_version)).fetchall()
    rejected = sorted(str(r["graph_version"]) for r in newer)
    for r in newer:
        con.execute("UPDATE graph_builds SET status = 'rolled_back', error = ? WHERE graph_build_id = ?",
                    (f"rolled back to {graph_version}", str(r["graph_build_id"])))
    note = {"at": now, "from": rolled_back_from, "rejected": rejected}
    if row is not None:
        try:
            manifest = json.loads(row["manifest_json"] or "{}")
        except (TypeError, ValueError):
            manifest = {}
        manifest["rollback"] = note
        con.execute("UPDATE graph_builds SET status = 'done', finished_at = ?, error = NULL, manifest_json = ? WHERE graph_build_id = ?",
                    (now, json.dumps(manifest, ensure_ascii=False), str(row["graph_build_id"])))
        return {"graph_build_id": str(row["graph_build_id"]), "record": "updated", "rejected": rejected}
    fingerprint = ""
    try:
        fingerprint = graph_cache_fingerprint(settings, source)
    except Exception:
        pass
    build_id = db.begin_graph_build(con, source_key=source_key, kb_id=source.kb_id, source_collection=source.collection,
                                    graph_version=graph_version, cache_fingerprint=fingerprint, build_kind="full")
    db.replace_graph_build_chunks(con, build_id, chunks)
    db.finish_graph_build(con, build_id, status="done", active_chunk_count=len(chunks),
                          active_doc_count=len({c["doc_id"] for c in chunks if c.get("doc_id")}),
                          source_content_hash=source_snapshot_hash(chunks), output_dir=output_dir,
                          manifest={"adopted": True, "rollback": note})
    return {"graph_build_id": build_id, "record": "created", "rejected": rejected}


def graph_retirable(settings: Settings, source: KBSource) -> list[str]:
    """The versions to retire (the version numbers that were completed) when a KB has been emptied while its graph
    remains; empty when nothing should be retired.
    "Emptied" = no active chunks, and the directory still exists but holds no file that can be ingested. Chunks
    alone are not enough: after a KB is turned off and on again, or deleted files are put back, the chunk count is
    also 0 until the restore jobs finish, but then the directory has files and the graph is usable again once they
    are restored. A directory that no longer exists is left to the KB's deactivation and garbage collection."""
    from ..localfs.scanner import list_source_files

    if source.physical_base is None or not Path(source.physical_base).is_dir():
        return []
    if active_source_chunks(settings, source):
        return []
    with db.connect(settings.state_db) as con:
        rows = con.execute("SELECT graph_version FROM graph_builds WHERE kb_id = ? AND status = 'done'",
                           (source.kb_id,)).fetchall()
    versions = sorted({str(r["graph_version"]) for r in rows if r["graph_version"]})
    if not versions or list_source_files(source, limit=1, hash_content=False):
        return []
    return versions


def retire_graph(settings: Settings, *, source_key: str, source: KBSource, q: Any = None,
                 grace_seconds: int | None = None) -> dict[str, Any] | None:
    """Retire the graph of a KB that has been emptied. Without a corpus no append or rebuild ever happens, so the
    current version would keep its aliases and search would go on returning entities, facts and pages of deleted
    documents. The Qdrant aliases are dropped, the active Neo4j version is deactivated, the completed records are
    marked rolled_back (no longer the current version, and no longer a baseline for appends / rebuilds), and the
    artifacts are deleted by the same rule as the old-version cleanup. The graph switch, labels, extraction and
    response caches are left alone: when files come in again it is handled as a first build, and units extracted
    before hit the cache directly.
    Returns None when there is nothing to retire (see graph_retirable). Shares the build lock with builds: refused
    while a build is running."""
    guard = GraphBuildLock(settings)
    guard.acquire()
    try:
        versions = graph_retirable(settings, source)
        if not versions:
            return None
        q = q or qdrant_client(settings.qdrant_url, settings.qdrant_api_key)
        result: dict[str, Any] = {"source": source_key, "kb_id": source.kb_id, "retired_versions": versions, "steps": []}
        result["aliases_dropped"] = drop_graph_aliases(q, source.collection, ALL_GRAPH_VECTOR_TYPES)
        result["steps"].append("drop_qdrant_aliases")
        if getattr(settings, "graph_neo4j_import_after_build", False):
            from .neo4j_import import deactivate_neo4j_graph_version

            result["neo4j_deactivation"] = deactivate_neo4j_graph_version(settings, source=source)
            result["steps"].append("deactivate_neo4j")
        with db.connect(settings.state_db) as con:
            con.execute("UPDATE graph_builds SET status = 'rolled_back', error = ? WHERE kb_id = ? AND status = 'done'",
                        ("retired: the knowledge base has no active content", source.kb_id))
        result["steps"].append("record")
        grace = int(grace_seconds if grace_seconds is not None else getattr(settings, "graph_gc_grace_seconds", 0) or 0)
        gc = gc_graph_versions(settings, source, q=q, graph_version="", grace_seconds=grace)
        result.update(gc["result"])
        result["steps"].extend(gc["steps"])
        if gc["errors"]:
            result["graph_gc_errors"] = gc["errors"]
        print(f"[graph] retired kb={source.kb_id}({source.source_root}): no active content; versions={versions}", flush=True)
        return result
    finally:
        guard.release()
