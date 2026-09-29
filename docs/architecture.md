# Architecture

[Home](../README.md) · English | [中文](architecture.zh-CN.md)

Carrel separates ingestion, graph construction, and retrieval. A shared
SQLite state database tracks files, configurations, jobs, and build progress.
The console operates these processes; agents consume the retrieval API.

## Ingestion

1. **Scan enabled folders.** Compare files against recorded metadata and
   checksums, defer recent writes, and queue new or changed content. Deletions
   enter the index lifecycle. See [folder rules](configuration.md#document-folders).
2. **Parse by format.** PDF, DOCX, and PPTX use the MinerU integration.
   Spreadsheets use native table readers; HTML and Markdown retain document
   structure; source code is split around symbols. A separate vision-language
   model describes figures and standalone images.
3. **Build chunks.** Merge compatible text, lists, and split tables before
   token-bounded splitting. Preserve document location, section paths, table
   headers, and available page, slide, or row information.
4. **Embed and index.** Text embeddings include document and section context.
   Qdrant stores text vectors and optional visual vectors; OpenSearch indexes
   chunk text with its built-in `cjk` analyser, so two-character Chinese terms
   match directly. Source text is retained in the payload.
5. **Maintain state.** Parser versions determine when reprocessing is needed.
   Leases, retries, cancellation checks, and caches support interrupted work;
   a job that cannot connect to a service it needs goes back to the queue
   without counting as a failed attempt. Old content versions become inactive
   and expire through maintenance tasks.

The parser backend is selected by the deployed MinerU container. GPU and CPU
routes differ; the application can read that decision with
`MINERU_BACKEND=auto`. Backend selection is documented in
[Docker Compose](../deployment/compose/README.md#the-parser-container).

## Knowledge graph construction

Graphs are optional per knowledge base. The system derives entity types and
relationships from document samples.

| Stage | Work |
|---|---|
| Schema preparation | Infer entity types under six upper-level classes, relationship predicates, definitions, and examples; store a schema version |
| Extraction | Group chunks into extraction units; collect entities, relationships, and structured facts with sources |
| Deterministic routes | Extract structure from supported code, structured Markdown, and configuration files without LLM extraction |
| Merge and reconciliation | Resolve entities and property concepts; normalize units; preserve conflicting evidence and time information |
| Derived views | Compile subject, timeline, and source pages from available facts and relationships |
| Publication | Write versioned Qdrant graph collections and, when enabled, the Neo4j projection; coordinate activation with failure and rollback checks |

Extraction responses are cached by input and configuration fingerprint.
Incremental builds reuse unchanged extraction results, merge decisions, and
vectors. Deleted documents are removed from the next graph version. Time or
change thresholds can trigger a full rebuild instead.

Graph work lives under `runtime/graph/`; the build lock coordinates graph
processes. Relationships and compiled pages carry source references that
agents can use to retrieve the underlying passages.

## Retrieval

The query path uses embedding and scoring models; answer generation belongs
to the calling agent.

1. Probe vector and keyword evidence across libraries, unless the caller
   explicitly supplies `kbs`. Image queries can also contribute visual evidence.
2. Select libraries using evidence scores. When selected libraries produce no
   usable results, automatic routing can widen the search.
3. Add graph and visual candidates where those capabilities are available.
4. Fuse candidates with reciprocal rank fusion, balance subjects or documents,
   and optionally rerank with a cross-encoder.
5. Apply coverage and diversity selection, stitch neighboring chunks and table
   continuations, remove supported overlaps, and allocate text token budgets.
6. Return evidence, provenance, and diagnostic status.

Routing combines vector similarity with lexical evidence weighted across
libraries. The ranking and routing parameters are configurable; their actual
values come from the deployment environment, with Python fallbacks in
[`kb_search/config.py`](../app/kb_search/config.py).

The API exposes graph neighborhoods for an agent to follow relations step by
step. Relationship results include source references and query limits.
[Retrieval and API](retrieval.md) explains result status, source verification,
filters, and follow-up requests.

## Service boundaries

| Component | Role |
|---|---|
| SQLite | File registry, configuration, queue, chunks, and graph build state |
| MinerU and native parsers | Document structure and content extraction |
| Model endpoints | Embeddings, image descriptions, graph extraction, and optional reranking |
| Qdrant | Chunk vectors and versioned graph evidence collections |
| OpenSearch | Keyword index |
| Neo4j | Graph projection and relationship queries |
| Web console, port 9800 | Knowledge-base configuration and operational controls |
| Search API, port 9810 | Evidence retrieval for agents |

The two application services run outside the supplied Compose stack. On
Linux, user systemd services and timers run the console, search, ingestion,
graph updates, and maintenance. See [Operations](operations.md).

## Repository layout

```text
app/
  kb_pipeline/        ingestion, graph construction, CLI, and maintenance
    pipeline/         scanning, scheduling, worker, and parse jobs
    parsers/          format routing and document parsers
    chunking/         chunk assembly and diagnostics
    embedding/        text and visual embedding clients
    graph/            schema, extraction, merging, facts, and publication
    vector/           Qdrant layout and writes
    vision/           image handling and descriptions
    localfs/          source-file scanning
  kb_server/          console API, service logic, and static frontend
  kb_search/          routing, retrieval, ranking, and evidence assembly
  tests/              regression tests and shared fixtures
config/               application environment template
deployment/           Docker Compose assets and systemd templates
scripts/              deployment and scheduled-task helpers
skills/carrel-search/ agent skill and standard-library Python client
docs/                 configuration, architecture, API, and operations guides
deploy.sh             deployment entry point
```

Generated material lives in ignored directories. Under `runtime/`: the mirror
(`mirror/`), the state database (`state/kb-pipeline.db`), the parse cache
(`parse_cache/`), graph workspaces and the LLM response cache
(`graph/work/<base>/<version>/`, `graph/cache/<base>.sqlite`), the graph build
lock (`state/graph_build.lock.d/`) and the stores' data (`qdrant_data/`,
`opensearch/`, `neo4j/`). `models/` holds weights; `logs/` and `backups/`
hold operational files.
