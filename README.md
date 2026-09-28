# Carrel

English | [中文](README.zh-CN.md)

**Ontology-Augmented Generation for AI agents.** Carrel turns a folder of
documents into a local knowledge base an agent can consult: files are parsed,
chunked and embedded into Qdrant and OpenSearch, each knowledge base can grow
an ontology-typed entity graph (LLM extraction → merging → entity / relation /
fact collections in Qdrant + a Neo4j projection), and a search service hands
the resulting *evidence* to whichever agent asks. Carrel never writes the
answer itself; the agent does. Everything runs on scheduled tasks behind a
small web console.

A carrel is the private desk in a library where a reader works through the
stacks. This is that desk for your agent.

One command deploys it: `./deploy.sh` detects the machine, starts MinerU,
Qdrant, OpenSearch and Neo4j, installs the app and the console. Embeddings,
figure descriptions and the extraction LLM are plain OpenAI-compatible
endpoints, hosted or local.

- [Design principles](#design-principles)
- [What it does](#what-it-does)
- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Feeding it documents](#feeding-it-documents)
- [Configuration](#configuration)
- [Using it](#using-it)
- [Ingestion pipeline](#ingestion-pipeline)
- [Entity graph](#entity-graph)
- [Scheduled tasks](#scheduled-tasks)
- [Search service and agent skill](#search-service-and-agent-skill)
- [Repository layout](#repository-layout)
- [Development and tests](#development-and-tests)
- [Troubleshooting](#troubleshooting)
- [Security boundary](#security-boundary)
- [License](#license)

## Design principles

1. **Fully automatic, no curation.** There is no manual tagging, no hand-linked
   entities, no graph editing. Anything that needs a person in the loop was
   left out so that the system converges unattended.
2. **General over tuned.** Rules are written for any corpus, not for the
   corpora it was developed on, and a little noise is accepted in exchange.
3. **Evidence, not answers.** The search service returns sources, facts and
   graph neighbourhoods with provenance; generation stays with the calling
   agent, so the server needs no generative model at all.

## What it does

- **Parsing.** PDF, Word and PowerPoint go through MinerU (layout, OCR,
  tables, formulas); Excel, HTML, Markdown and source code are parsed
  natively; figures are described by a vision-language model. Parsers are
  versioned, so a parser change re-processes exactly the affected files.
- **Chunking.** Block-aware: tables, lists and table fragments split across
  pages are merged before splitting; code is split by symbol. Each chunk
  carries its document and section path.
- **Two vectors per chunk.** A 1024-d text vector and, for pictures, a 2048-d
  visual vector stored as named vectors on the same point.
- **Keyword index.** Chunks are mirrored into OpenSearch with the built-in
  `cjk` analyser, so two-character Chinese terms match directly.
- **Entity graph (optional per knowledge base).** An LLM proposes the label
  set (entity types under six upper-level ontology classes, predicates with
  endpoint constraints, examples), extracts entities, relations and
  unit-bearing measurement facts, and Carrel merges, reconciles conflicts
  and compiles subject / timeline pages. New documents are appended
  incrementally; a full rebuild happens only when the configured policy says
  so.
- **Search API.** Vector, keyword, graph and visual channels fused per
  knowledge base with automatic routing, cross-encoder reranking, quota and
  diversity selection, neighbour stitching and a token budget. Every
  degradation is reported, never hidden.
- **Console.** Enrol folders, tune chunking and graph settings, watch
  progress and health, restart services, register LLMs. Chinese and English.

## How it works

```text
your sync tool (rsync / Syncthing / NFS / copy — anything)
   ▼
runtime/mirror/<folder>          ← each enabled folder = one knowledge base kb_NNN
   │  scan.timer (every minute): new / changed / deleted → job queue (SQLite)
   ▼
worker.timer (every 5 min): parse → chunk → embed → write
   ├─ MinerU (8765) + vision-language model (OpenAI-compatible)   parsing, figure descriptions
   ├─ text embeddings (OpenAI-compatible)                          1024-d `text` vector
   ├─ visual embeddings (optional, local vLLM)                     2048-d `visual` vector
   ├─ Qdrant (6333)                                                one collection per knowledge base
   └─ OpenSearch (9200)                                            one keyword index per knowledge base
   │
   ▼  graph-rebuild.timer (every 2 h, graph-enabled bases only)
chunks → extraction units → LLM → merge / concept normalisation / fact reconciliation
   ├─ Qdrant entity / relation / fact collections (versioned, alias switch)
   └─ Neo4j (7687) projection (versioned)
   │
   ▼
web console (9800)          search API (9810) ──▶ agent skill (skills/carrel-search)
```

The infrastructure is a Compose project
([`deployment/compose`](deployment/compose/README.md)): MinerU, Qdrant,
OpenSearch and Neo4j start by default; five vLLM model servers sit behind the
`local-models` profile for hosts that want embeddings, rerankers and the
vision model on the same GPU. The LLM used for graph extraction is registered
in the console; its key lives only in the local state database.

## Requirements

| Platform | What works |
|---|---|
| Linux + NVIDIA GPU | everything local; MinerU runs its `vlm-engine` backend, model servers can run here |
| Linux without GPU | MinerU runs the CPU `pipeline` backend; embeddings, vision and LLM come from hosted APIs |
| macOS (Docker Desktop) | containers run on CPU, the app runs natively; visual channels off; schedule the timers yourself |
| Windows | through WSL2 it is the Linux path; native Windows is not supported |

- Docker 25+ with Compose v2; on GPU hosts the NVIDIA container toolkit (CDI
  or the nvidia runtime).
- Python 3.12+ for the app (pipeline, console and search service live in
  `app/.venv`). `deploy.sh --with-pdf-images` also installs the optional
  `pdf-images` extra (PyMuPDF, AGPL-3.0), which lets search serve
  original-resolution PDF pictures; without it search serves the parser's
  cached images. See `NOTICE.md`.
- One OpenAI-compatible endpoint each for text embeddings, figure
  descriptions and the extraction LLM: a hosted provider, or the
  `local-models` profile. Reranking is optional (`/v1/rerank` as served by
  vLLM, TEI, Jina or Cohere). Visual embeddings and visual reranking exist
  only with local vLLM.
- Memory: the stores are sized by `deploy.sh` from the host's RAM; 16 GB is
  enough for the default stack, the local model servers want a lot more.

## Quick start

```bash
git clone <this repo> carrel && cd carrel
./deploy.sh
```

`deploy.sh` is idempotent; re-running it upgrades. It:

1. Detects the machine: architecture, GPU and container toolkit, memory
   tier, whether Docker Hub, PyPI and HuggingFace answer (falling back to
   mirror registries and ModelScope when they do not).
2. Writes `deployment/compose/.env` (GPU overlay or not, parser image
   flavour, image names, memory tier, a random Neo4j password) and
   `config/knowledge-base.env`. Existing files are kept.
3. Builds the parser image and starts the four containers. On first start
   the MinerU container inspects the GPU it received, picks `vlm-engine` or
   `pipeline`, and downloads the matching weights into `models/mineru/`.
4. Creates `app/.venv`, installs the app, initialises the state database and,
   on Linux, installs and enables the systemd user units.

Open the console URL it prints (`http://127.0.0.1:9800` by default; see
[Security boundary](#security-boundary) for LAN access). Drop a folder under
`runtime/mirror/`, enable it in the console, and scanning and parsing start
on their own. For a graph, register an LLM in the top bar first, then switch
on "graph" in the folder's settings; label extraction, building and
incremental appends are handled by the timers.

Other commands: `./deploy.sh detect` prints the detection without changing
anything; `--with-local-models` also starts the five model servers (weights
go under `models/` first); `--cpu` forces the CPU parser; `down` stops the
containers; `purge [--images] [--data]` uninstalls. Hosts without systemd
(macOS) get the manual commands for the console, the search API and one
ingest round printed at the end.

## Feeding it documents

Carrel only manages what happens after `runtime/mirror/`; how files get
there is yours to choose. The contract:

1. Put content under `runtime/mirror/<folder>/` (`KB_MIRROR_ROOT` to move
   it); one folder is one knowledge base, and only folders enabled in the
   console are processed.
2. Files younger than `KB_MIN_FILE_AGE_SECONDS` (180 s) are left alone, so a
   half-written upload is never picked up.
3. A sync tool that wants the scanner to wait can hold the directory
   `runtime/state/mirror_sync.lock.d` while it pushes (optional).
   Symlinks are not followed outside the folder; to mount a folder from
   elsewhere as a knowledge base, set `KB_MIRROR_ALLOW_LINKED_DIRS=1`.
4. The mirror is the source of truth: a vanished file is soft-deleted and
   kept for a retention period; a folder that suddenly empties is refused by
   `KB_MASS_DELETE_GUARD_*`. Worth knowing if you sync with `rsync --delete`.

## Configuration

### Global: `config/knowledge-base.env`

The template is
[`config/knowledge-base.env.example`](config/knowledge-base.env.example).
Groups:

| Group | Keys | Notes |
|---|---|---|
| roots | `KB_LOCAL_BASE_DIR` `KB_MIRROR_ROOT` `KB_STATE_DB` `KB_RUNTIME_DIR` `KB_CACHE_DIR` `KB_LOG_DIR` `KB_GRAPH_WORK_DIR` | default to the checkout; state is SQLite in WAL mode |
| stores | `QDRANT_URL` `OPENSEARCH_URL` `NEO4J_URI` `NEO4J_USER` `NEO4J_PASSWORD` | all loopback; the Neo4j password lives only in this mode-600 file |
| parsing | `MINERU_SERVICE_URL` `MINERU_BACKEND=auto` `MINERU_LANG` `KB_PARSE_ENABLED` `KB_MIN_FILE_AGE_SECONDS` `KB_JOB_*` | `auto` uses whichever backend the parser container chose; `KB_PARSE_ENABLED=0` pauses parsing |
| retention | `QDRANT_INACTIVE_RETENTION_DAYS` `KB_CACHE_ROTATION_KEEP` `KB_LOG_ROTATION_KEEP_MONTHS` `KB_VLM_CACHE_MAX_AGE_DAYS` | undo window for deletions (7 days), cache / log rotation, caption cache lifetime |
| text embeddings | `EMBEDDING_*` | any OpenAI-compatible `/v1/embeddings`; the dimension is baked into the collections at creation |
| figure descriptions | `VLM_*` `KB_IMAGE_MAX_PIXELS` | any OpenAI-compatible multimodal chat endpoint |
| visual vectors | `VISUAL_EMBEDDING_*` | optional, local vLLM only (`messages`-shaped requests); off by default |
| rerankers | `RERANKER_BASE_URL` `VISUAL_RERANKER_BASE_URL` | search side only; empty means no reranking |
| console | `KB_WEB_HOST` `KB_WEB_PORT` `KB_CONSOLE_SERVICES` | the service rows the console probes and can restart, default `database,mineru` |
| graph | `QDRANT_GRAPH_COLLECTION_RETENTION_DAYS` `NEO4J_GRAPH_RETENTION_DAYS` `GRAPH_GC_KEEP_VERSIONS` `GRAPH_NEO4J_IMPORT_*` `KB_GRAPH_LLM_CONCURRENCY` `KB_GRAPH_LLM_TIMEOUT` `KB_GRAPH_CIRCUIT_FAILS` | old versions kept 14 days and at most 2 per base; LLM concurrency, timeout, circuit breaker |
| search | `KB_SEARCH_*` | host, port, bearer token, channel sizes, routing thresholds (documented in `app/kb_search/config.py`) |

Changing `EMBEDDING_DIM` or `VISUAL_EMBEDDING_DIM` means recreating every
collection.

### Per knowledge base (console → settings)

One `kb_sources` row per base; the strategy lives in its `config_json` and is
edited only in the console:

| Group | Keys | Notes |
|---|---|---|
| chunking | `max_tokens` `overlap_tokens` | bounded by the embedding model's context; the console shows the limit |
| graph switches | `graph_enabled` `graph_auto_append` `graph_profile` | on/off, allow incremental appends, scenario profile |
| labels | `graph_entity_types` `graph_parent_types` `graph_type_definitions` `graph_predicates` `graph_examples` `graph_language` | generated by "extract labels" from `graph_tune_sample_size` samples; a previous version can be re-activated; hand edits discouraged |
| extraction | `graph_unit_chunks` `graph_max_gleanings` | consecutive chunks per extraction unit (3; 1 = none), gleaning rounds per unit |
| models | `graph_llm.extract` `graph_llm.summarize` `graph_llm.tune` | one registered LLM per step |
| rebuild policy | `graph_rebuild_interval` `graph_rebuild_new_chunk_pct` `graph_rebuild_new_chunk_count` `graph_rebuild_operator` | full rebuild after N days or when new chunks exceed a share / count (`or` / `and`); otherwise append |

## Using it

### Console

Three tabs and a sidebar with every knowledge base and a unified progress bar:

- **Settings**: chunking, graph switches and labels, per-step models, rebuild
  policy; buttons for save, parse now, re-parse everything, extract labels,
  build graph, append, pause build, delete graph, disable base.
- **Files**: per-file parse state, chunk count, failure reason, retry; a
  drawer shows the chunk preview and the job timeline.
- **Graph**: entity / relation preview of the current version, filter by
  type, merge records.

The top bar holds the LLM registry (keys are write-only), service health
(restart one / stop all / restart all for the rows in `KB_CONSOLE_SERVICES`,
with the parser's backend shown next to it) and the language switch. "Save"
only stores the strategy: changed chunk settings take effect after "re-parse
everything"; a parser version bump re-queues affected files by itself.

A disabled base enters a retention period (7 days) before GC removes it;
re-enabling restores it. A renamed folder keeps its number automatically;
when the match is ambiguous the console offers a manual "adopt".

### Command line

`app/.venv/bin/kb` (also installed as `carrel`), configured from
`config/knowledge-base.env`:

| Command | Purpose |
|---|---|
| `kb config` / `kb status` / `kb health` | effective settings, queues and bases, service health |
| `kb init-db` | create the SQLite state database |
| `kb scan [--source kb_NNN] [--requeue-failed] [--rehash]` | scan the mirror and queue work (the timer does this every minute) |
| `kb worker --once --max-jobs N --max-seconds S` | drain the queue (timer-driven; never start a second worker next to the systemd one) |
| `kb qdrant ensure-collections` / `ensure-graph-collections` | create missing collections |
| `kb fts init` / `status` / `rebuild` / `sync-doc` / `search` | OpenSearch index maintenance |
| `kb graph build` / `append` / `check-rebuild [--execute] [--force-full]` | full build, incremental append, policy check (what the timer runs) |
| `kb graph adopt-current` / `neo4j-import` / `neo4j-status` / `neo4j-delete` | version baseline, Neo4j projection |
| `kb graph query` / `factcheck` / `status` | graph recall prototype, fact-level evaluation, build status |
| `kb cleanup status` / `weekly` / `monthly` / `qdrant-gc` / `qdrant-graph-gc` / `neo4j-graph-gc` / `parse-assets-gc` | garbage collection (timer-driven) |
| `kb search eval` / `make-set` | search regression evaluation and question-set generation |
| `kb reset --source kb_NNN --yes` | wipe one base's state / cache / index and recreate its collection (without `--yes` it only prints the plan) |

Start over completely (the first command only prints the plan without `--yes`):

```bash
cd app
./.venv/bin/kb reset --all --yes --force
./.venv/bin/kb scan --verbose
systemctl --user start --no-block carrel-worker.service
```

### HTTP API of the console

Everything the console uses is under `/api`: `overview`, `health`, `limits`,
`enroll`, `kbs/{kb_id}/…` (`config`, `files`, `jobs`, `parse_now`,
`reparse`, `graph_build`, `graph_append`, `graph_pause`, `graph_schema`,
`graph_preview`, `graph_merges`, `graph_builds`, `chunk_preview`, `unenroll`,
`adopt`, graph delete), `jobs/{id}/cancel`, `files/retry`,
`services/{key}/restart`, `services/restart_all`, `services/stop_all`,
`llms`. Writes require the same origin.

## Ingestion pipeline

1. **Scan** (every minute). Only enabled folders; new files rest for
   `KB_MIN_FILE_AGE_SECONDS`; changes are detected by checksum and size, a
   touched timestamp alone does not re-parse; deletions are soft (reappearing
   within retention restores); a round is skipped while a sync tool holds the
   mirror lock.
2. **Parse** (every 5 minutes, with a time budget). Each job has a lease, an
   owner and retry back-off; crashed jobs are reclaimed next round; cancel
   beats retry. Parsers are versioned, and a version change re-queues the
   affected files.
3. **Chunk.** Block-structured splitting; tables, lists and cross-page table
   fragments are merged first, then split by `max_tokens` / `overlap_tokens`;
   code by symbol. Chunks are written to the state database.
4. **Embed and write.** The embedded text is "document › section path" plus
   the body (the payload keeps the raw body); pictures are described by the
   VLM and get a `visual` vector (both cached across bases); points go to
   Qdrant (old versions soft-deleted, hard-deleted by GC after 7 days) and to
   OpenSearch.
5. **Artefacts.** MinerU's intermediate output and images stay in
   `runtime/parse_cache/`; nightly GC keeps only what the current versions use.

## Entity graph

Building reads the chunks of the state database directly:

1. **Labels.** From sample chunks the LLM proposes entity types, parents
   (six upper-level classes), predicates and examples, stored as a label
   version; it runs automatically before a build, and a concurrent run is
   marked superseded instead of overwriting.
2. **Extraction.** `graph_unit_chunks` consecutive chunks form a unit; the
   LLM returns entities, relations and measurement facts with units. Results
   are cached per (base, unit, config fingerprint), so unchanged units cost no
   LLM calls next time; malformed replies are retried with a correction hint.
3. **Merging.** Deterministic merging of same-name / alias entities (aliases
   are always double-checked), concept normalisation with the unit alias
   table, `evidence_conflict` instead of picking one of two contradicting
   values, time series attached to the profile's subject.
4. **Publish.** Entity / relation / fact vectors go to versioned Qdrant
   collections, then to Neo4j; only when both succeed does the alias switch,
   and old versions expire by retention.
5. **Append** (every 2 hours). Only units of new or changed documents are
   extracted; the previous version's merge decisions are replayed and
   unchanged vectors reused; deleted documents drop out. When the rebuild
   policy triggers, a full rebuild runs instead, baselined on the last full
   build.

Workspace: `runtime/graph/work/<base>/<version>/` holds units and merged
graphs, `runtime/graph/cache/<base>.sqlite` the LLM response cache. Builds
are mutually exclusive through a flock on
`runtime/state/graph_build.lock.d/lock`; the console's "pause build" waits
for the process to exit.

## Scheduled tasks

| Unit | Cadence | Does |
|---|---|---|
| `carrel-scan.timer` | 30 s after enable, then every minute | scan the mirror and queue changes |
| `carrel-worker.timer` | 1 min, then every 5 min | drain the queue: parse → chunk → embed → write |
| `carrel-graph-rebuild.timer` | 10 min, then every 2 h | append; full rebuild when the policy says so |
| `carrel-qdrant-gc.timer` | 30 min, then every 24 h | point GC, expired bases, old graph versions, job history |
| `carrel-cache-weekly.timer` | 1 h, then every 7 days | cache rotation (host-wide uv/pip/Docker cleanup only with `KB_HOST_HOUSEKEEPING=1`) |
| `carrel-logs-monthly.timer` | 2 h, then every 30 days | log rotation |
| `carrel-web.service` | always on | console |
| `carrel-search.service` | always on | search API |

Timers are relative (`OnActiveSec` / `OnUnitActiveSec`): no wall clock, no
timezone. Maintenance tasks yield while the system is busy (exit 75) and only
fail after several consecutive yields; the policy is in
`scripts/lib/kb-maint-defer.sh`. See
[`deployment/systemd`](deployment/systemd/README.md).

## Search service and agent skill

The retrieval layer is a long-running service, `app/kb_search/` (port 9810,
static bearer `KB_SEARCH_TOKEN`; without a token only loopback callers are
accepted). All strategy lives on the server, the caller passes the question
verbatim, and the response is evidence only. The server's query path uses
scoring models alone: embeddings, a cross-encoder reranker, optionally the
visual embedder. Nothing generates.

The client for agents is
[`skills/carrel-search/`](skills/carrel-search/SKILL.md): a standard-library
Python script plus a `SKILL.md` that tells an agent such as Claude Code or
Codex how to search, when to fetch more context or the original image, and
how to cite. Configure `CARREL_SEARCH_BASE_URL` / `CARREL_SEARCH_TOKEN` or
`~/.config/carrel-search/config.json`.

| Endpoint | Purpose |
|---|---|
| `GET /health` | no auth; bases, auth mode, Qdrant connectivity |
| `GET /catalog` | base catalogue: names, domains, subject types, size, file-name samples, whether a graph exists; `?refresh=1` rebuilds |
| `POST /search` | `{question, kbs?, top_k?, hints?, context?, explain?, image_b64?}` → numbered Sources / Entities / Relationships / Specs / Pages, `doc_aggs`, `retrieval_summary` |
| `POST /context` | `{kb_id, doc_id, content_version?, chunk_from, chunk_to}` → chunks of one document by index range |
| `GET /image/{kb_id}/{point_id}` | the original picture of an image chunk (from the mirrored PDF when its hash matches the chunk's content version, else the parse cache); `X-Image-Source` says which |
| `POST /crop` | `{kb_id, point_id, bbox, pad?}` → deterministic crop of that picture |
| `POST /graph/neighbors` | `{kb_id, entity or entity_id, limit?, types?, direction?}` → one-hop relations of an entity in the current graph with evidence chunks; multi-hop is the agent's job |

One query: vector and BM25 channels probe every base in parallel → bases are
chosen by evidence (unless `kbs` is given) → graph channel (one hop) and
visual channel run on the chosen bases → per-base RRF fusion, interleaved
subject / document buckets → cross-encoder reranking → final score =
rerank × 0.8 + query-token coverage × 0.2, boilerplate and low-confidence
pictures demoted → quota plus MMR selection → neighbour stitching, table
continuation with headers, overlap dedup, token budget → evidence pack. Any
unavailable stage degrades and is listed in `retrieval_summary.degraded`
(reranker down → fusion order; embeddings down → keyword and graph seeds;
Neo4j down → no graph channel; OpenSearch down → no keyword channel).

Routing needs no profile or topic assumption: each base's evidence is the
mean cosine of its top vector hits blended 7:3 with a lexical signal (query
tokens by the `cjk` analyser, weighted by cross-base rarity); bases within
`KB_SEARCH_ROUTE_GAP` of the best are queried together, at most
`KB_SEARCH_ROUTE_MAX_KBS`; when the chosen bases yield nothing usable the
query widens to all bases (`routing.widened`).

Facts come with a clue string ("subject · property = value unit @ condition ·
time"), source numbers, series siblings and conflict flags; compiled pages
(subject, timeline, source) are marked `compiled=true`; picture chunks carry
the visual description, in-image text, facts and conflict notes. Status
fields the caller must read: `retrieval_summary.evidence_state` (accepted /
diagnostic / unranked, with `no_relevant_content` when nothing clears the
rerank floor), `verified` / `sources_active` on facts and pages, stable `id`s
on entities, relations and facts, `graph_versions`, `stitched.pieces`,
`text_truncated` with `/context` for the full text, `hints_used` /
`hints_ignored` / `hints_scope`. `hints` accepts four deterministic keys:
`doc_ids`, `rel_paths` and `content_version` filter hard, `block_types` is a
soft preference. Every parameter is documented next to its value in
`app/kb_search/config.py`.

```bash
curl -s -H "Authorization: Bearer $KB_SEARCH_TOKEN" -X POST http://<host>:9810/search \
  -H "Content-Type: application/json" \
  -d '{"question": "operating temperature range of this part", "kbs": ["kb_002"], "top_k": 8}'
```

Regression evaluation (question sets live in `runtime/eval/`, not in git):

```bash
cd app && .venv/bin/kb search eval --set ../runtime/eval/my-set.json --out ../runtime/eval/result.json --auto
```

`eval` reports hit@3/5/12 on chunk gold, doc_hit@3/5/12 on document gold,
MRR, expect / expect_all, min_docs, negatives, mean latency and error count,
and diffs against the previous result of the same name. `kb search make-set`
drafts a question set from a base.

## Repository layout

```text
app/
  kb_pipeline/        pipeline package (CLI entry kb_pipeline.cli:main)
    pipeline/         scan / worker / parse_job / state migration
    parsers/          router, MinerU client (pdf/docx/pptx), native_table, html_dom, code_symbols, visual_blocks …
    chunking/  embedding/  vector/  vision/  localfs/
    graph/            build, extract, facts, concepts, reconcile, compile, temporal, lock, neo4j_import …
    measure_units.py  unit aliases and normalisation (shared by parsing and graph)
    maintenance.py    GC, pause build, delete graph, service restarts
  kb_server/          FastAPI console (api.py / service.py / static/{index.html,app.js,i18n.js})
  kb_search/          search service (evidence only)
  tests/              one file per module plus _support.py fixtures
deploy.sh             one-command deploy / detect / status / down / purge
config/               knowledge-base.env(.example)
deployment/
  compose/            four default containers + optional local model servers, the parser image and its entrypoint
  systemd/            user-scope timers and services (templates rendered by scripts/install-systemd.sh)
scripts/              wrappers for the timers (scan / worker-once / graph-rebuild-check / cleanup), install-systemd.sh, lib/
skills/carrel-search/ agent skill for the search service
runtime/ (ignored)    mirror/ state/ graph/ parse_cache/ qdrant_data/ neo4j/ opensearch/ mineru-output/
models/ (ignored)     parser weights and local model weights
logs/, backups/ (ignored)
LICENSE, NOTICE.md    MIT and third-party notices
```

## Development and tests

```bash
cd app && .venv/bin/python -m pytest -q        # ~700 tests, no containers needed
```

- Tests use stubs for every service; a full run takes well under a minute
  on a laptop. Tests that talk to a real LLM only use cheap "flash" models.
- Changing a parser's output: bump the archive version in
  `parsers/common.py`; affected files re-parse automatically.
- Every console string needs an entry in `static/i18n.js`;
  `ConsoleI18nTests` fails otherwise. Data (folder names, file names,
  entity names) is never translated.
- Changes under `static/` need no restart. Python changes need
  `carrel-web.service` restarted; check its journal first because
  label extraction and chunk previews run inside the web process.
- After editing `deployment/systemd/`, run `./scripts/install-systemd.sh`
  (`--check` compares) or the installed copies drift silently.
- The state database is `runtime/state/kb-pipeline.db`; a wrong path
  silently creates an empty one.
- `app/tests/test_deployment.py` scans every tracked file for personal or
  machine-specific traces; keep it green.

## Troubleshooting

```bash
./deploy.sh status
systemctl --user list-timers 'knowledge-base*'
journalctl --user -u carrel-worker -f
journalctl --user -u carrel-graph-rebuild --since -3h
curl -s localhost:9800/api/health | python3 -m json.tool
app/.venv/bin/kb status
app/.venv/bin/kb graph status
docker logs carrel-mineru                      # backend decision, weight download
```

- **Files queue but never parse**: `KB_PARSE_ENABLED=1`? Look for "yield"
  or a lock in the worker log; `runtime/state/mirror_sync.lock.d` means a
  sync tool is pushing.
- **Build refused with "lock already exists"**: another build is running.
  The lock is a flock and dies with its process; never delete the directory.
- **A maintenance unit shows failed with exit 75**: it yielded several
  rounds in a row because the system stayed busy; it recovers on its own.
- **All pictures fail**: the vision or visual-embedding endpoint is
  unhealthy; `VISUAL_EMBEDDING_ENABLED=0` ingests text only meanwhile.
- **Parser chose the wrong backend**: `docker logs carrel-mineru` shows the
  decision and why; `MINERU_BACKEND_POLICY` in `deployment/compose/.env`
  forces one.
- **Neo4j data corrupt**: stop the container, delete `runtime/neo4j/data`
  (as root through a throwaway container), start it, `kb graph
  neo4j-import` re-projects the current version.

## Security boundary

- **Console (9800).** Listens on loopback by default. To use it from other
  devices set `KB_WEB_HOST=0.0.0.0` and a `KB_WEB_TOKEN`; every API call
  then needs the token as a bearer header, and the page asks for it once per
  browser. Browser writes are additionally limited to the console's own
  origin (scheme, host and port). Never expose the console to the internet:
  it can read every chunk and restart or stop services.
- **Search API (9810).** Static bearer token; with an empty token only
  loopback callers are accepted.
- **Model keys.** Registered keys are stored in the local state database and
  never returned by the API. Editing a model's address, port or protocol
  requires entering the key again, so a stored key is never sent to a new
  endpoint by accident. Requests that carry keys do not follow redirects.
- **What leaves the machine.** Documents, chunks and pictures are sent to
  whatever endpoints you configure for embeddings, figure descriptions and
  graph extraction. With hosted providers that is your content going to them
  over the network; use HTTPS endpoints and providers you trust.
- **Stores.** Model servers, MinerU, Qdrant, Neo4j and OpenSearch bind to
  loopback. Secrets live only on the host: `config/knowledge-base.env`,
  `deployment/compose/.env` and the state database; none is committed.
- **Mirror boundary.** Symlinks that point outside an enabled folder are
  skipped; symlinked top-level folders are ignored unless
  `KB_MIRROR_ALLOW_LINKED_DIRS=1`.
- For access across networks use a private overlay such as Tailscale rather
  than port forwarding.

## License

MIT, see [`LICENSE`](LICENSE). Third-party components and the origin of the
extraction prompts are listed in [`NOTICE.md`](NOTICE.md).
