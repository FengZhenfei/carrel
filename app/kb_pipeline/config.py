from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass
from pathlib import Path


from .limits import (
    GRAPH_MAX_GLEANINGS_DEFAULT, GRAPH_TUNE_SAMPLE_DEFAULT, GRAPH_UNIT_CHUNKS_DEFAULT,
    normalize_entity_types, normalize_examples, normalize_parent_types, normalize_predicates,
    normalize_profile, normalize_type_definitions,
)
from .models import GraphRebuildPolicy, KBSource
from .utils import load_env_file
from .vector.layout import VectorLayout


BASE_DIR = Path(os.getenv("KB_LOCAL_BASE_DIR", Path(__file__).resolve().parents[2]))
DEFAULT_ENV_FILE = BASE_DIR / "app" / ".env"
DEFAULT_QDRANT_CREDENTIALS = BASE_DIR / "secrets" / "qdrant-credentials.txt"


@dataclass(frozen=True)
class Settings:
    env_file: Path
    qdrant_credentials_file: Path
    state_db: Path
    mirror_root: Path
    opensearch_url: str
    runtime_dir: Path
    cache_dir: Path
    log_dir: Path
    qdrant_url: str
    qdrant_api_key: str | None
    embedding_base_url: str
    embedding_model_id: str
    embedding_dim: int
    embedding_batch: int
    embedding_retry: int
    embedding_sleep_seconds: float
    embedding_api_key: str
    # visual vectors: Qwen3-VL-Embedding on this box, image pixels -> 2048-d
    visual_embedding_enabled: bool
    visual_embedding_base_url: str
    visual_embedding_model_id: str
    visual_embedding_api_key: str
    visual_embedding_dim: int
    visual_embedding_instruction: str
    visual_embedding_concurrency: int
    visual_embedding_retry: int
    visual_embedding_timeout_seconds: float
    # The reranker models (text + cross-modal) take no part in parsing / graph building and serve only the
    # retrieval side; they are here only so the console's "reranker service" row can probe / restart them.
    # Empty = not deployed on this machine, and the row is hidden from the drawer.
    reranker_base_url: str
    visual_reranker_base_url: str
    # pixel budget applied before an image goes to either vision model
    image_max_pixels: int
    vlm_base_url: str
    vlm_model_id: str
    vlm_api_key: str
    vlm_concurrency: int
    vlm_temperature: float
    vlm_top_p: float
    vlm_max_tokens: int
    vlm_structured_output: bool
    vlm_failure_retry_ratio: float
    mineru_url: str
    mineru_timeout_seconds: int
    vlm_timeout_seconds: int
    parse_enabled: bool
    min_file_age_seconds: int
    max_file_bytes: int
    metadata_job_lease_seconds: int
    parse_job_lease_seconds: int
    job_max_retries: int
    job_retry_base_seconds: int
    job_retry_max_seconds: int
    qdrant_upsert_max_bytes: int
    qdrant_inactive_retention_days: int
    graph_work_dir: Path
    graph_llm_concurrency: int
    graph_llm_timeout_seconds: int
    graph_circuit_fails: int
    graph_gc_retention_days: int
    neo4j_graph_retention_days: int
    # Besides expiring by days, keep at most this many latest graph versions (the current one counts):
    # incremental append makes versions arrive more often, each version is a full set of collections, and
    # without a cap the disk fills up.
    graph_gc_keep_versions: int
    graph_gc_grace_seconds: int
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str | None
    graph_neo4j_import_after_build: bool
    graph_neo4j_import_batch_size: int
    # Service rows the console manages (health probe + restart buttons):
    # containers of the compose stack on this host. "database" (Qdrant,
    # OpenSearch, Neo4j) and "mineru" are always displayed; the model rows
    # only appear when listed here, because their endpoints may just as well
    # be hosted APIs that nothing on this box can restart.
    console_services: tuple[str, ...]
    sources: dict[str, KBSource]

    @property
    def vector_layout(self) -> VectorLayout:
        """Named vectors every KB collection is created with. The visual slot
        is declared even when the visual pass is disabled: Qdrant cannot add a
        named vector to an existing collection, so leaving it out would force
        a full rebuild the day the pass is switched on."""
        return VectorLayout(text_size=self.embedding_dim, visual_size=self.visual_embedding_dim)


def _default_env_file() -> Path:
    preferred = BASE_DIR / "config" / "knowledge-base.env"
    return preferred if preferred.exists() else DEFAULT_ENV_FILE


def load_settings(env_file: str | Path | None = None) -> Settings:
    selected_env = Path(env_file or os.getenv("KB_ENV_FILE") or _default_env_file())
    load_env_file(selected_env)

    # Credentials are read from the env file only; secrets/qdrant-credentials.txt is a path from the Mac era,
    # and that directory is long gone.
    qdrant_credentials = Path(os.getenv("QDRANT_CREDENTIALS_FILE", str(DEFAULT_QDRANT_CREDENTIALS)))
    qdrant_key = os.getenv("QDRANT_API_KEY")
    # The password is read from the env file only. The macOS Keychain fallback was a leftover from the Mac
    # era: Linux has no security command, so it only ever spawned a subprocess bound to fail.
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    mirror_root = Path(os.getenv("KB_MIRROR_ROOT", str(BASE_DIR / "runtime" / "mirror"))).expanduser()
    state_db = Path(os.getenv("KB_STATE_DB", str(BASE_DIR / "runtime" / "state" / "kb-pipeline.db")))

    # Knowledge bases are ENROLLED, not auto-discovered: the mirror carries
    # every top-level directory, but only the ones ticked in the web console
    # (rows in kb_sources) enter the pipeline. Per-KB strategy lives in the
    # registry row -- there is no .kb.json any more.
    # KB_EXTRA_SOURCES_JSON is still honoured for sources outside the mirror.
    from .discovery import enrolled_sources

    sources: dict[str, KBSource] = enrolled_sources(state_db, mirror_root)
    sources.update(_load_extra_sources(mirror_root))

    return Settings(
        env_file=selected_env,
        qdrant_credentials_file=qdrant_credentials,
        mirror_root=mirror_root,
        state_db=state_db,
        opensearch_url=os.getenv("OPENSEARCH_URL", "http://127.0.0.1:9200"),
        runtime_dir=Path(os.getenv("KB_RUNTIME_DIR", str(BASE_DIR / "runtime"))),
        cache_dir=Path(os.getenv("KB_CACHE_DIR", str(BASE_DIR / "runtime" / "parse_cache" / "kb-pipeline"))),
        log_dir=Path(os.getenv("KB_LOG_DIR", str(BASE_DIR / "logs"))),
        qdrant_url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"),
        qdrant_api_key=qdrant_key,
        # Defaults point at this host rather than the cloud: with a missing or misspelled env key, requests
        # used to silently go to the cloud endpoint (empty key → 401, but the request had already been sent).
        embedding_base_url=os.getenv("EMBEDDING_BASE_URL", "http://127.0.0.1:8101/v1"),
        embedding_model_id=os.getenv("EMBEDDING_MODEL_ID", "qwen3-embedding-0.6b"),
        embedding_dim=int(os.getenv("EMBEDDING_DIM", "1024")),
        embedding_batch=int(os.getenv("EMBEDDING_BATCH", "10")),
        embedding_retry=int(os.getenv("EMBEDDING_RETRY", "5")),
        embedding_sleep_seconds=float(os.getenv("EMBEDDING_SLEEP_SECONDS", "0.1")),
        embedding_api_key=os.getenv("EMBEDDING_API_KEY", ""),
        visual_embedding_enabled=env_bool("VISUAL_EMBEDDING_ENABLED", True),
        visual_embedding_base_url=os.getenv("VISUAL_EMBEDDING_BASE_URL", "http://127.0.0.1:8103/v1"),
        visual_embedding_model_id=os.getenv("VISUAL_EMBEDDING_MODEL_ID", "qwen3-vl-embedding-2b"),
        visual_embedding_api_key=os.getenv("VISUAL_EMBEDDING_API_KEY", "local"),
        visual_embedding_dim=int(os.getenv("VISUAL_EMBEDDING_DIM", "2048")),
        visual_embedding_instruction=os.getenv("VISUAL_EMBEDDING_INSTRUCTION", "Represent the user's input."),
        visual_embedding_concurrency=int(os.getenv("VISUAL_EMBEDDING_CONCURRENCY", "2")),
        visual_embedding_retry=int(os.getenv("VISUAL_EMBEDDING_RETRY", "5")),
        visual_embedding_timeout_seconds=float(os.getenv("VISUAL_EMBEDDING_TIMEOUT_SECONDS", "300")),
        reranker_base_url=os.getenv("RERANKER_BASE_URL", "http://127.0.0.1:8102/v1").strip(),
        visual_reranker_base_url=os.getenv("VISUAL_RERANKER_BASE_URL", "http://127.0.0.1:8104/v1").strip(),
        image_max_pixels=int(os.getenv("KB_IMAGE_MAX_PIXELS", "3686400")),
        vlm_base_url=os.getenv("VLM_BASE_URL", "http://127.0.0.1:8105/v1"),
        vlm_model_id=os.getenv("VLM_MODEL_ID", "qwen3-vl-8b-instruct-fp8"),
        vlm_api_key=os.getenv("VLM_API_KEY", ""),
        vlm_concurrency=int(os.getenv("VLM_CONCURRENCY") or "1"),
        # Qwen3-VL recommends top_p 0.8 for vision tasks; the temperature is
        # pulled well below its default 0.7 because this is an extraction task,
        # but kept off zero to avoid the repetition loops greedy decoding can
        # fall into on sampling-tuned models.
        vlm_temperature=float(os.getenv("VLM_TEMPERATURE", "0.1")),
        vlm_top_p=float(os.getenv("VLM_TOP_P", "0.8")),
        vlm_max_tokens=int(os.getenv("VLM_MAX_TOKENS", "4096")),
        vlm_structured_output=env_bool("VLM_STRUCTURED_OUTPUT", True),
        # Retry the whole document once image descriptions fail at this ratio (by default any failure
        # retries), so a document with "no image described at all" is not stored permanently as a
        # successful parse.
        vlm_failure_retry_ratio=float(os.getenv("VLM_FAILURE_RETRY_RATIO", "0.001")),
        mineru_url=os.getenv("MINERU_SERVICE_URL", "http://127.0.0.1:8765"),
        # The parse service's read timeout must be independent of the job lease: it used to pass the lease
        # (12-18h) directly, so once MinerU hung (GPU hang) the worker sat silently on its lock for a whole
        # day while the heartbeat kept renewing the lease.
        mineru_timeout_seconds=int(os.getenv("MINERU_TIMEOUT_SECONDS", "3600")),
        vlm_timeout_seconds=int(os.getenv("VLM_TIMEOUT_SECONDS", "180")),
        parse_enabled=os.getenv("KB_PARSE_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"},
        min_file_age_seconds=int(os.getenv("KB_MIN_FILE_AGE_SECONDS", "180")),
        # 0 = unlimited. Default 200MB: single files above this size are almost always logs / data exports
        # dropped in by mistake, and parsing them only drags the worker down.
        max_file_bytes=int(os.getenv("KB_MAX_FILE_BYTES", str(200 * 1024 * 1024))),
        metadata_job_lease_seconds=int(os.getenv("KB_METADATA_JOB_LEASE_SECONDS", "3600")),
        parse_job_lease_seconds=int(os.getenv("KB_PARSE_JOB_LEASE_SECONDS", str(12 * 3600))),
        job_max_retries=int(os.getenv("KB_JOB_MAX_RETRIES", "5")),
        job_retry_base_seconds=int(os.getenv("KB_JOB_RETRY_BASE_SECONDS", "300")),
        job_retry_max_seconds=int(os.getenv("KB_JOB_RETRY_MAX_SECONDS", "3600")),
        qdrant_upsert_max_bytes=int(os.getenv("QDRANT_UPSERT_MAX_BYTES", str(30 * 1024 * 1024))),
        qdrant_inactive_retention_days=int(os.getenv("QDRANT_INACTIVE_RETENTION_DAYS", "7")),
        # Graph build workspace (units / graph / LLM cache); see the layout notes at the top of graph/build.py
        graph_work_dir=Path(os.getenv("KB_GRAPH_WORK_DIR", str(BASE_DIR / "runtime" / "graph"))),
        # Public LLM concurrency / timeout / circuit-breaker threshold; see graph/llm.py for the rationale
        graph_llm_concurrency=int(os.getenv("KB_GRAPH_LLM_CONCURRENCY", "8")),
        graph_llm_timeout_seconds=int(os.getenv("KB_GRAPH_LLM_TIMEOUT", "300")),
        graph_circuit_fails=int(os.getenv("KB_GRAPH_CIRCUIT_FAILS", "20")),
        graph_gc_retention_days=int(os.getenv("QDRANT_GRAPH_COLLECTION_RETENTION_DAYS", "14")),
        neo4j_graph_retention_days=int(
            os.getenv("NEO4J_GRAPH_RETENTION_DAYS", os.getenv("QDRANT_GRAPH_COLLECTION_RETENTION_DAYS", "14"))
        ),
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=neo4j_password,
        graph_neo4j_import_after_build=env_bool("GRAPH_NEO4J_IMPORT_AFTER_BUILD", True),
        # Keep the previous version besides the current one: a quality problem found after release can still
        # be switched back (the rollback on a failed switch only protects the moment of release). It was once
        # 1 version only (morning of 2026-09-06); the Codex review advised keeping at least one accepted
        # version.
        graph_gc_keep_versions=max(1, int(os.getenv("GRAPH_GC_KEEP_VERSIONS", "2"))),
        graph_gc_grace_seconds=max(0, int(os.getenv("GRAPH_GC_GRACE_SECONDS", "60"))),
        graph_neo4j_import_batch_size=int(os.getenv("GRAPH_NEO4J_IMPORT_BATCH_SIZE", "1000")),
        console_services=parse_console_services(os.getenv("KB_CONSOLE_SERVICES")),
        sources=sources,
    )


# Console service rows, in display order. "database" bundles Qdrant, OpenSearch
# and Neo4j; the rest map one row to one container group (see
# kb_server.service.SERVICE_CONTAINERS).
CONSOLE_SERVICE_KEYS: tuple[str, ...] = ("database", "mineru", "embedding", "visual_embedding", "vlm", "reranker")
CONSOLE_SERVICES_DEFAULT: tuple[str, ...] = ("database", "mineru")


def parse_console_services(raw: str | None) -> tuple[str, ...]:
    """KB_CONSOLE_SERVICES: comma-separated row keys the console manages.
    Unknown names are dropped, order follows CONSOLE_SERVICE_KEYS, and an unset
    variable means the two rows every deployment has (database, mineru). An
    explicitly empty value means "manage nothing": rows still show for the
    stores and the parser, but without restart buttons."""
    if raw is None:
        return CONSOLE_SERVICES_DEFAULT
    wanted = {part.strip().lower() for part in raw.split(",") if part.strip()}
    return tuple(key for key in CONSOLE_SERVICE_KEYS if key in wanted)


def _load_extra_sources(mirror_root: Path) -> dict[str, KBSource]:
    raw = os.getenv("KB_LOCAL_EXTRA_SOURCES_JSON") or os.getenv("KB_EXTRA_SOURCES_JSON", "")
    raw = raw.strip()
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("KB_EXTRA_SOURCES_JSON must be a JSON list")

    sources: dict[str, KBSource] = {}
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Each extra source must be a JSON object")
        kb_id = str(item["kb_id"])
        key = str(item.get("key") or kb_id)
        physical_base_raw = item.get("physical_base")
        physical_base = (
            Path(str(physical_base_raw))
            if physical_base_raw
            else mirror_root / str(item["source_root"])
        )

        sources[key] = KBSource(
            kb_id=kb_id,
            collection=str(item.get("collection", f"kb_{kb_id}")),
            source_root=str(item["source_root"]),
            source_type=str(item.get("source_type", "local_mirror")),
            physical_base=physical_base,
            max_tokens=int(item.get("max_tokens", 400)),
            overlap_tokens=int(item.get("overlap_tokens", 80)),
            graph_enabled=bool_value(item.get("graph_enabled"), False),
            graph_auto_append=bool_value(item.get("graph_auto_append"), True),
            graph_unit_chunks=int(item.get("graph_unit_chunks") or GRAPH_UNIT_CHUNKS_DEFAULT),
            graph_max_gleanings=int(item.get("graph_max_gleanings") if item.get("graph_max_gleanings") is not None
                                    else GRAPH_MAX_GLEANINGS_DEFAULT),
            # Keep the same field set as discovery.build_source: one missing, and a dry-run started through
            # this path runs on a configuration different from the real graph build, invisibly so.
            graph_entity_types=normalize_entity_types(item.get("graph_entity_types")),
            graph_language=str(item.get("graph_language") or "").strip() or None,
            graph_predicates=normalize_predicates(item.get("graph_predicates")),
            graph_parent_types=normalize_parent_types(item.get("graph_parent_types")),
            graph_type_definitions=normalize_type_definitions(item.get("graph_type_definitions")),
            graph_examples=normalize_examples(item.get("graph_examples")),
            graph_profile=normalize_profile(item.get("graph_profile")),
            graph_tune_sample_size=int(
                item.get("graph_tune_sample_size") or GRAPH_TUNE_SAMPLE_DEFAULT),
            graph_rebuild_policy=GraphRebuildPolicy(
                interval_days=parse_interval_days(
                    item.get("graph_rebuild_interval") or item.get("graph_rebuild_period_days")
                ),
                new_chunk_ratio=parse_ratio(
                    item.get("graph_rebuild_new_chunk_pct") or item.get("graph_rebuild_new_chunk_ratio")
                ),
                operator=str(item.get("graph_rebuild_operator", "or")).strip().lower(),
            ),
        )
    return sources


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def bool_value(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def parse_interval_days(value: object | None) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip().lower()
    match = re.fullmatch(r"(\d+)\s*([dwm月周日]?)", text)
    if not match:
        raise ValueError(f"invalid graph rebuild interval: {value!r}; use 7d, 2w, 1m, or days")
    amount = int(match.group(1))
    unit = match.group(2)
    if amount <= 0:
        raise ValueError("graph rebuild interval must be positive")
    if unit in {"", "d", "日"}:
        return amount
    if unit in {"w", "周"}:
        return amount * 7
    if unit in {"m", "月"}:
        return amount * 30
    raise ValueError(f"unsupported graph rebuild interval unit: {unit!r}")


def parse_new_chunk_count(value: object | None) -> int | None:
    """Threshold on the number of new chunks. It and parse_ratio are two forms of the same condition, mutually
    exclusive."""
    if value is None or str(value).strip() == "":
        return None
    count = int(str(value).strip())
    if count < 1:
        raise ValueError("graph rebuild new chunk count must be at least 1")
    return count


def parse_ratio(value: object | None) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    if text.endswith("%"):
        ratio = float(text[:-1].strip()) / 100
    else:
        ratio = float(text)
        if ratio > 1:
            ratio = ratio / 100
    if ratio < 0:
        raise ValueError("graph rebuild new chunk ratio must be non-negative")
    return ratio
