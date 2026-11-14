# Configuration

[Home](../README.md) · English | [中文](configuration.zh-CN.md)

## Configuration files

Container deployment uses two configuration files and preserves them on subsequent runs:

| File | Purpose |
|---|---|
| `data/config/knowledge-base.env` | Application paths, model endpoints, credentials, and processing settings; created by the application at first startup |
| `deployment/compose/.env` | Container images, storage edition, parser backend, resource limits and data mounts; prepared by `deploy.sh` |

Use [knowledge-base.env.example](../config/knowledge-base.env.example) and
[Compose settings](../deployment/compose/README.md) as references. The generated
files hold credentials and are excluded from Git. The container sets `KB_ENV_FILE`
to `/data/config/knowledge-base.env`. Native installations use
`config/knowledge-base.env`, with `app/.env` as a legacy fallback;
`KB_ENV_FILE` selects a different file.

Restart the app container after changing environment settings. For native systemd
installations, restart `carrel-web.service` and `carrel-search.service` when their
settings change; new scan and worker processes read the updated file. Let
active processing finish before restarting it. See [Operations](operations.md).
The search service is the exception for its own `KB_SEARCH_*` keys: it re-reads
the file every minute, so a changed parameter or token applies without a
restart. Its listening address and port still require a restart. Model endpoints saved
in the console also refresh within a minute. Explicit process environment
values take precedence over the environment file.

Deploy embedding and reranking models independently, or use hosted endpoints. Add the vision endpoint to the console LLM section and select it for each knowledge base before saving its parsing configuration.

## Model endpoints

System settings has separate **LLM**, **Embedding**, and **Reranker** sections.
LLM covers image descriptions and graph building. Embedding and Reranker each
configure text and visual endpoints with a Base URL, Model ID and optional
API Key; embedding endpoints also specify dimensions. Text embeddings are
required, while the other three roles can be disabled individually. Deploying
and starting model servers remains separate from configuring their endpoints.

A different endpoint can expose a different vector space even when its model ID and dimensions match. Changing the endpoint, model or dimensions requires reinitializing existing indexes first. Rotating credentials or adjusting batch size and concurrency retains the indexes.

Each LLM has **Multimodal** and **Request concurrency** settings. Mark a model as
multimodal to make it selectable for image descriptions (currently via the
OpenAI-compatible protocol). Saving a multimodal model sends a synthetic image
and requires a nonempty, completed text response; no fixed answer is required,
and test replies are not retained. Other models use the text connectivity check. Concurrency ranges from 1 to 256, defaulting to 1
for new models. Image descriptions and graph jobs each use their selected
model’s limit; dependent steps remain sequential. Changes apply to new tasks,
without changing document-level concurrency or invalidating caches. Upgrades
preserve existing vision selections and adopt the former environment limits.
A model used for both purposes adopts the lower limit. Subsequent changes are
made in the console.

The console initially displays environment settings. Saved values live in the
state database and override the corresponding environment variables for parsing,
graph building and search. Saving checks the endpoint with synthetic text or
images, without sending knowledge-base content. Failed checks preserve the
previous configuration. Keys are hidden by default; the eye button reveals a
saved or newly typed key. Viewing a key does not edit it. A blank field keeps
the current key. Changing the endpoint host requires re-entering the key;
the API also supports explicit credential removal.

Parsing workers read settings after claiming each new task; newly started graph
jobs read them too. Search refreshes within a minute. Saving does not trigger
parsing or graph building. Batch size and request concurrency take effect on the
next task, without rebuilding indexes or invalidating image-vector caches. Other
embedding changes wait for active parsing and graph jobs to finish. The console rejects a different embedding endpoint URL, model ID or dimension
while indexes exist. Changing vector spaces requires reinitializing the indexes
and generating new vectors; changing dimensions also requires recreating the
knowledge-base collections. Reranker changes do not require reindexing.

Both embedding cards expose **Batch size** (items per request) and **Request
concurrency** (simultaneous requests per task). Text settings apply to document
and graph indexing; visual settings apply to image embedding. These correspond
to `EMBEDDING_BATCH`, `EMBEDDING_CONCURRENCY`, `VISUAL_EMBEDDING_BATCH`, and
`VISUAL_EMBEDDING_CONCURRENCY`. They do not change the number of documents parsed
at once. Visual batches use separate conversations, with one vector per image.
Keep visual batch size at 1 for endpoints that only accept single conversations.
Saving a batch size greater than 1 checks the interface with two synthetic inputs;
it does not measure the server's capacity at the chosen batch size or concurrency.

Standard starts database containers and MinerU; Light starts only MinerU and uses embedded storage. Model endpoints are independent in both editions.

| Capability | Settings | Required when |
|---|---|---|
| Text embeddings | `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL_ID`, `EMBEDDING_API_KEY`, `EMBEDDING_DIM` | Indexing and vector search |
| Image descriptions and table review | Select a multimodal model per knowledge base in the console; `KB_IMAGE_MAX_PIXELS` sets the image budget | Processing images and figures; the endpoint must accept image input |
| Graph extraction and summaries | Register a chat model in the console, then select it per knowledge base | Graph building is enabled |
| Text reranking | `RERANKER_BASE_URL`, `RERANKER_MODEL_ID`, `RERANKER_API_KEY` | Optional; retrieval otherwise uses fusion order |
| Visual embeddings | `VISUAL_EMBEDDING_ENABLED`, `VISUAL_EMBEDDING_BASE_URL`, `VISUAL_EMBEDDING_MODEL_ID`, `VISUAL_EMBEDDING_DIM`, other `VISUAL_EMBEDDING_*` | Optional pixel-level retrieval; the bundled integration uses local vLLM |
| Visual reranker service | `VISUAL_RERANKER_BASE_URL`, `VISUAL_RERANKER_MODEL_ID`, `VISUAL_RERANKER_API_KEY` | Optional multimodal reranking; joins retrieval automatically when configured |

Text embeddings use an OpenAI-compatible embeddings API. Image descriptions
and graph models use compatible chat APIs. Rerankers use a `/v1/rerank`
endpoint; visual embedding requests use vLLM's multimodal pooling format.
Configure each endpoint for the corresponding request format.

Model registration keeps free-form URLs and model IDs. Connectivity checks,
graph calls and image descriptions share parameter compatibility rules for
output-token limits and sampling. An explicit unsupported-parameter response
can trigger a bounded retry; successful adaptations are remembered within
the process for that endpoint, model, protocol and request profile.

Custom deployment aliases can use `KB_LLM_COMPAT_OVERRIDES` in the environment
file. Entries match an exact `model_id`, with an optional `base_url` and a
`protocol` (default `openai`). For example:

```sh
KB_LLM_COMPAT_OVERRIDES='[{"model_id":"custom-deployment","base_url":"https://models.example/v1","max_tokens_field":"max_completion_tokens","omit":["temperature","top_p"]}]'
```

`max_tokens_field` selects the output-limit field; `omit` removes optional
generation parameters; `set` supplies values such as `{"reasoning_effort":"low"}`
or `{"thinking":{"type":"disabled"}}`. Matching entries apply in order.
Restart the application processes after changing this setting. Model probes
require a completed response containing `OK`; an HTTP 400 is reported as an error.

For a deployment with a different thinking-control format, use
`"thinking_parameters":{"chat_template_kwargs":{"enable_thinking":false}}`.
These fields replace thinking controls only when the caller requests thinking
to be disabled, such as graph extraction. Image descriptions and probes keep
their existing thinking behavior. Use `set` for parameters intended for every call.

For hosted text reranking, set `RERANKER_BASE_URL` to the API base or the full
`/rerank` endpoint, `RERANKER_MODEL_ID` to the provider's model name, and
`RERANKER_API_KEY` to its Bearer token. Requests contain `model`, `query`, and
`documents`; responses contain `results` with `index` and `relevance_score`.
Leaving the model name empty uses `/models` discovery. Leaving the key empty
keeps unauthenticated local access.

In the console model registry, **Max Input** is the model's input token limit.
Label extraction selects samples automatically for document and section
coverage, keeping the estimated complete request within 30% of this value.
The sample count is read-only and shows **Not calculated** until extraction
starts. Progress remains available after refreshing or reopening the console.

Deployment starts no embedding, reranking or chat model service and downloads none of their weights. See [storage editions](storage.md#deployment).

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
| Jobs | `KB_JOB_MAX_RETRIES`, `KB_JOB_RETRY_BASE_SECONDS`, `KB_JOB_RETRY_MAX_SECONDS`, `KB_PARSE_JOB_LEASE_SECONDS`, `KB_METADATA_JOB_LEASE_SECONDS` | Retry and lease settings. A job that cannot connect to a service it needs goes back to the queue without using up a retry: it waits the base delay, twice as long each following time up to the maximum, and fails after `KB_JOB_MAX_RETRIES` such rounds |
| Retention | `QDRANT_INACTIVE_RETENTION_DAYS`, `KB_CACHE_ROTATION_KEEP`, `KB_LOG_ROTATION_KEEP_MONTHS`, `KB_BACKUP_KEEP` | Undo window for deletions (7 days in the template), cache and log rotations kept, nightly state backups kept (7). Cached picture descriptions and visual vectors have no age limit: the weekly cleanup drops an entry once no picture in the parse cache refers to it |
| Graphs | `GRAPH_GC_KEEP_VERSIONS`, `GRAPH_GC_GRACE_SECONDS`, `QDRANT_GRAPH_COLLECTION_RETENTION_DAYS`, `NEO4J_GRAPH_RETENTION_DAYS`, `GRAPH_NEO4J_IMPORT_*` | Versions kept per base: the active one plus N-1 older ones regardless of age; the newest paused / failed version is kept for resuming and does not count towards N (roll back with `kb graph rollback --source <key> --graph-version <old>`: the target has to be a finished build whose collections still match the recorded point counts, `--force` switches to an unfinished one); the two retention-days keys only affect the manual days-only tools; Neo4j import |
| Graph models | `KB_GRAPH_LLM_TIMEOUT`, `KB_GRAPH_CIRCUIT_FAILS` | Timeout and circuit breaker; concurrency is configured per LLM |
| Console | `KB_WEB_HOST`, `KB_WEB_PORT`, `KB_WEB_TOKEN`, `KB_CONSOLE_SERVICES` | Default address is `127.0.0.1:9800`; managed service rows default to `database,mineru` |
| Search | `KB_SEARCH_HOST`, `KB_SEARCH_PORT`, `KB_SEARCH_TOKEN`, `KB_SEARCH_TOP_K`, `KB_SEARCH_RERANK`, `KB_SEARCH_CONTEXT_TOKENS`, `KB_SEARCH_REQUEST_BUDGET`, `KB_SEARCH_ROUTE_MAX_KBS`, `KB_SEARCH_ROUTE_GAP`, `KB_SEARCH_ROUTE_FLOOR`, other `KB_SEARCH_*` | Bind address and token, result count, reranking, context budget, time budget of a whole request (keep it below the caller's timeout); automatic routing queries the bases within GAP of the best evidence score (at most MAX_KBS) and flags the result weak below FLOOR |

Every key has a default in the code. The template lists every search key with
its code default, commented out except the address and the token, so the
calibrated defaults apply until you uncomment one (a test keeps the two in
step). What each threshold means is explained in
[`app/kb_search/config.py`](../app/kb_search/config.py); the search service's
`/health` reports text reranking, visual recall and visual reranking separately.
`KB_SEARCH_RERANK` controls both rankers. Images share the total
`KB_SEARCH_RERANK_N` budget (64 by default), with up to
`KB_SEARCH_VISUAL_RERANK_N=8` image positions, a separate
`KB_SEARCH_VISUAL_RERANK_THRESHOLD=0.5`, and
`KB_SEARCH_VISUAL_RERANK_TIMEOUT=15` seconds within the request budget.
The image endpoint must support multimodal `content` containing text and
`image_url` data URLs. See [retrieval](retrieval.md#text-and-image-reranking).

## Per-knowledge-base settings

The console stores these settings with each knowledge-base registration:

| Group | Main keys | Purpose |
|---|---|---|
| Chunking | `max_tokens`, `overlap_tokens` | Chunk size and overlap, within the embedding model's context limit |
| Multimodal parsing | `vlm_model`, `vlm_prompt` | Registered model and optional description prompt shared by image, chart and table review stages |
| Graph control | `graph_enabled`, `graph_auto_append`, `graph_profile` | Enable graphs, allow incremental updates, and store the inferred scenario profile |
| Schema | `graph_schema_active`, `graph_language` | Select a versioned LinkML schema and output language; types and relationships are derived from the selected schema |
| Extraction | `graph_unit_chunks`, `graph_max_gleanings` | Extraction unit size and supplementary extraction rounds; schema sampling is automatic |
| Models | `graph_llm.extract`, `graph_llm.summarize`, `graph_llm.tune` | Select registered models for each graph stage |
| Rebuild | `graph_rebuild_interval`, `graph_rebuild_new_chunk_pct`, `graph_rebuild_new_chunk_count`, `graph_rebuild_operator`, `graph_rebuild_resuggest` | Time and change thresholds, combined with `or` or `and`; refresh labels before full rebuilds by default |

Schemas can be extracted from the corpus or uploaded as LinkML YAML. Carrel
uses a LinkML subset for type inheritance, identity scope, and relationships.
Set `graph_rebuild_resuggest=false` to reuse the selected schema on full rebuilds.

`KB_CORRECTIONS_DIR` defaults to `config/corrections`. Each library has its own
`<kb_id>.yaml` for entity corrections and document status. These rules follow
the library through graph deletion and rebuilding, are included in state
backups, and are removed when the library is deleted. New libraries receive
fresh rules even if an identifier is reused.

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
  Set bind addresses according to the intended network access. Loopback keeps
  other machines out, not web pages opened in a browser on the same host:
  Qdrant's CORS is off for that reason, but the bundled Qdrant has no API key,
  so do not browse the web on the host that runs the stack.
- **Model data:** documents, chunks, and images are sent to the endpoints you
  configure. Hosted endpoints receive that content; local endpoints process it
  within your infrastructure.
- **Credentials:** model registry keys stay in the local state database and
  are not returned by the API. Reusing a stored key with a changed model endpoint
  requires entering it again. Environment files and the database contain
  the deployment's credentials.
- **Optional dependencies:** PDF original-image retrieval requires the
  `pdf-images` extra. Its license is listed in [third-party notices](../NOTICE.md).

### Graph and schema history

`GRAPH_GC_KEEP_VERSIONS=3` keeps the current graph and two completed predecessors, including their graph indexes and build artifacts. The newest resumable build is protected separately. Schema history keeps 10 versions plus any older schema referenced by a retained build. The console's **Version comparison** shows the latest 20 lightweight comparison reports, which remain after full graph artifacts are pruned.

Comparisons separate source, schema, model, correction and pipeline changes. An assertion missing while its source is unchanged is marked for review; old assertions are not automatically restored. Increasing retention affects future cleanup and cannot recover versions already removed.

See [scoped corrections](architecture.md#scoped-corrections) for the preview, commit and revoke API.
