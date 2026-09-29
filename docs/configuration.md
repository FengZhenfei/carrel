# Configuration

[Home](../README.md) · English | [中文](configuration.zh-CN.md)

## Configuration files

`deploy.sh` creates two files and preserves them on subsequent runs:

| File | Purpose |
|---|---|
| `config/knowledge-base.env` | Application paths, model endpoints, credentials, and processing settings |
| `deployment/compose/.env` | Container images, parser backend policy, resource limits, and local model services |

Use [knowledge-base.env.example](../config/knowledge-base.env.example) and
[Compose settings](../deployment/compose/README.md) as references. The generated
files hold credentials and are excluded from Git. Application settings prefer
`config/knowledge-base.env`, with `app/.env` as a legacy fallback;
`KB_ENV_FILE` selects a different file.

Restart running application services after changing environment settings. On
Linux, restart `carrel-web.service` and `carrel-search.service` when their
settings change; new scan and worker processes read the updated file. Let
active processing finish before restarting it. See [Operations](operations.md).

## Model endpoints

The default deployment starts the stores and MinerU. Five optional embedding,
reranking, and vision model services are available with `--with-local-models`.

| Capability | Settings | Required when |
|---|---|---|
| Text embeddings | `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL_ID`, `EMBEDDING_API_KEY`, `EMBEDDING_DIM` | Indexing and vector search |
| Image descriptions | `VLM_BASE_URL`, `VLM_MODEL_ID`, `VLM_API_KEY`, `KB_IMAGE_MAX_PIXELS` | Processing images and figures; images are scaled down to the pixel budget before the call |
| Graph extraction and summaries | Register a chat model in the console, then select it per knowledge base | Graph building is enabled |
| Text reranking | `RERANKER_BASE_URL` | Optional; retrieval otherwise uses fusion order |
| Visual embeddings | `VISUAL_EMBEDDING_ENABLED`, `VISUAL_EMBEDDING_BASE_URL`, `VISUAL_EMBEDDING_MODEL_ID`, `VISUAL_EMBEDDING_DIM`, other `VISUAL_EMBEDDING_*` | Optional pixel-level retrieval; the bundled integration uses local vLLM |
| Visual reranking | `VISUAL_RERANKER_BASE_URL` | Optional visual reranking |

Text embeddings use an OpenAI-compatible embeddings API. Image descriptions
and graph models use compatible chat APIs. Rerankers use a `/v1/rerank`
endpoint; visual embedding requests use vLLM's multimodal pooling format.
Configure each endpoint for the corresponding request format.

For the bundled model stack, download the weights and run
`./deploy.sh --with-local-models`, following the
[local model instructions](../deployment/compose/README.md#local-model-servers-profile-local-models).
The graph chat model is configured separately in the console.

Choose embedding models and dimensions before indexing. The template uses
1024 dimensions for text (`EMBEDDING_DIM`) and 2048 for visual vectors
(`VISUAL_EMBEDDING_DIM`). Changing a dimension requires recreating the
affected collections; changing the model requires re-embedding the material
even if its dimension is unchanged.

## Document folders

Place each collection under `runtime/mirror/<folder>/`, or set `KB_MIRROR_ROOT`
to another mirror location. Enable the folder in the console to assign a
stable `kb_NNN` identifier and start processing its files.

Populate the mirror using rsync, Syncthing, a mounted filesystem, or manual
copying.

- Files younger than `KB_MIN_FILE_AGE_SECONDS` (180 seconds in the template)
  are deferred. Use atomic moves or the sync lock for long uploads.
- A sync process can hold `runtime/state/mirror_sync.lock.d` to defer scans
  until copying finishes. Release the lock when the sync completes.
- A removed file is scheduled for index deletion, with inactive data retained
  according to the retention policy. Returning files can be reactivated.
- When every enrolled directory disappears at once (an unmounted or emptied
  mirror), the scanner refuses to deactivate them. Deletions inside a single
  library are applied as seen, so take care with `rsync --delete`.
- Top-level symlink folders require `KB_MIRROR_ALLOW_LINKED_DIRS=1`. Links
  inside a library may not resolve outside its admitted root. A refused root
  remains registered and is shown as blocked in the console.

## Global settings

| Group | Main keys | Notes |
|---|---|---|
| Paths | `KB_LOCAL_BASE_DIR`, `KB_MIRROR_ROOT`, `KB_STATE_DB`, `KB_RUNTIME_DIR`, `KB_CACHE_DIR`, `KB_LOG_DIR`, `KB_GRAPH_WORK_DIR` | Defaults are relative to the installed project |
| Stores | `QDRANT_URL`, `QDRANT_API_KEY`, `OPENSEARCH_URL`, `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` | Match the deployed services |
| Parsing | `MINERU_SERVICE_URL`, `MINERU_BACKEND`, `MINERU_LANG`, `KB_PARSE_ENABLED`, `KB_MIN_FILE_AGE_SECONDS` | `MINERU_BACKEND=auto` reads the parser container's backend decision |
| Jobs | `KB_JOB_MAX_RETRIES`, `KB_JOB_RETRY_BASE_SECONDS`, `KB_JOB_RETRY_MAX_SECONDS`, `KB_PARSE_JOB_LEASE_SECONDS`, `KB_METADATA_JOB_LEASE_SECONDS` | Retry and lease settings |
| Retention | `QDRANT_INACTIVE_RETENTION_DAYS`, `KB_CACHE_ROTATION_KEEP`, `KB_LOG_ROTATION_KEEP_MONTHS`, `KB_VLM_CACHE_MAX_AGE_DAYS` | Index, cache, log, and caption retention |
| Graphs | `GRAPH_GC_KEEP_VERSIONS`, `GRAPH_GC_GRACE_SECONDS`, `QDRANT_GRAPH_COLLECTION_RETENTION_DAYS`, `NEO4J_GRAPH_RETENTION_DAYS`, `GRAPH_NEO4J_IMPORT_*` | Versions kept per base: the active one plus N-1 older ones regardless of age (roll back with `kb graph rollback --source <key> --graph-version <old>`); the two retention-days keys only affect the manual days-only tools; Neo4j import |
| Graph models | `KB_GRAPH_LLM_CONCURRENCY`, `KB_GRAPH_LLM_TIMEOUT`, `KB_GRAPH_CIRCUIT_FAILS` | Concurrency, timeout, and circuit breaker |
| Console | `KB_WEB_HOST`, `KB_WEB_PORT`, `KB_WEB_TOKEN`, `KB_CONSOLE_SERVICES` | Default address is `127.0.0.1:9800`; managed service rows default to `database,mineru` |
| Search | `KB_SEARCH_HOST`, `KB_SEARCH_PORT`, `KB_SEARCH_TOKEN`, `KB_SEARCH_TOP_K`, `KB_SEARCH_RERANK`, `KB_SEARCH_CONTEXT_TOKENS`, `KB_SEARCH_ROUTE_MAX_KBS`, `KB_SEARCH_ROUTE_GAP`, `KB_SEARCH_ROUTE_FLOOR`, other `KB_SEARCH_*` | Bind address and token, result count, reranking, context budget; automatic routing queries the bases within GAP of the best evidence score (at most MAX_KBS) and flags the result weak below FLOOR |

Every key has a default in the code, and the search keys in the template carry
those same defaults, so a key only matters once you change it (a test keeps the
two in step). The full list of search keys, with what each threshold means, is
in [`app/kb_search/config.py`](../app/kb_search/config.py); the search
service's `/health` reports whether reranking and the visual channel are active.

## Per-knowledge-base settings

The console stores these settings with each knowledge-base registration:

| Group | Main keys | Purpose |
|---|---|---|
| Chunking | `max_tokens`, `overlap_tokens` | Chunk size and overlap, within the embedding model's context limit |
| Graph control | `graph_enabled`, `graph_auto_append`, `graph_profile` | Enable graphs, allow incremental updates, and store the inferred scenario profile |
| Schema | `graph_entity_types`, `graph_parent_types`, `graph_type_definitions`, `graph_predicates`, `graph_examples`, `graph_language` | Automatically derived types, relationships, and examples |
| Extraction | `graph_tune_sample_size`, `graph_unit_chunks`, `graph_max_gleanings` | Schema sampling, extraction unit size, and supplementary extraction rounds |
| Models | `graph_llm.extract`, `graph_llm.summarize`, `graph_llm.tune` | Select registered models for each graph stage |
| Rebuild | `graph_rebuild_interval`, `graph_rebuild_new_chunk_pct`, `graph_rebuild_new_chunk_count`, `graph_rebuild_operator` | Time and change thresholds, combined with `or` or `and` |

After changing chunking settings, use the console's full re-parse action to
apply them to existing files. Parser implementation version changes
are handled separately: the next scan can requeue affected files automatically.

## Access and data handling

- **Console:** defaults to localhost. For LAN access, set `KB_WEB_HOST=0.0.0.0`
  and a `KB_WEB_TOKEN`. API calls require that token when configured; browser
  writes also require the console's origin. The console provides access to
  stored content and service controls.
- **Search:** configure `KB_SEARCH_TOKEN` for remote protected endpoints.
  With no token, protected endpoints accept only loopback callers. `/health`
  is unauthenticated and returns operational metadata. Its reachability follows
  the service's listening address and network access rules.
- **Stores and model servers:** the supplied Compose ports bind to loopback.
  Set bind addresses according to the intended network access.
- **Model data:** documents, chunks, and images are sent to the endpoints you
  configure. Hosted endpoints receive that content; local endpoints process it
  within your infrastructure.
- **Credentials:** model registry keys stay in the local state database and
  are not returned by the API. Reusing a stored key with a changed model endpoint
  requires entering it again. Environment files and the database contain
  the deployment's credentials.
- **Optional dependencies:** PDF original-image retrieval requires the
  `pdf-images` extra. Its license is listed in [third-party notices](../NOTICE.md).
