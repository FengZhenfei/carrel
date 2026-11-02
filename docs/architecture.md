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
   model describes figures, standalone images, and referenced XLSX pictures,
   including DrawingML anchors and WPS cell images. Spreadsheet pictures retain
   their sheet and cell location.
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

Additional readers handle R, Jupyter Notebook, RST, Quarto/RMarkdown, TeX,
BibTeX, JSONL, TSV, Nextflow/CWL/WDL, Snakefiles, Feature, DOT, XSD, and logs.
Notebooks retain cell numbers, code, saved text/table outputs and raster plots;
JSONL records retain line numbers and TSV tables retain columns and row ranges.
Scripts and document templates are read statically. Markdown HTML comments
are omitted from prose while code examples remain intact. Git LFS pointers
are excluded during scanning, so they do not enter file lists or parsing queues.
If an indexed file becomes a pointer, its old index entries are withdrawn; restoring
the original brings it back through the normal incremental pipeline.

Code files and fenced examples preserve spacing and literal text when split into
chunks, with line endings normalized to LF. Code files include line and column
ranges; embedded examples retain their position within the parsed block, and
document listings keep their captions separately. Parser profiles track changes to these extraction
and chunking rules, so the next scan can reprocess affected documents.

The parser backend is selected by the deployed MinerU container. GPU and CPU
routes differ; the application can read that decision with
`MINERU_BACKEND=auto`. Backend selection is documented in
[Docker Compose](../deployment/compose/README.md#the-parser-container).

Each knowledge base can select a **Multimodal model** from the model registry. The shared selection handles image and chart understanding and screenshot-based table review through an OpenAI-compatible vision endpoint, including models hosted on another machine. A model must be selected before saving the parsing configuration. Saving a selection applies to subsequent parsing; existing documents keep their current results until explicitly reparsed.

PDF extraction also reads visible form fields and associates local text, captions and legends with their source regions. Tables with missing native text receive a targeted visual review; row identity, existing values, units and conditions are checked before a replacement is adopted. Unresolved alignment keeps the original table and separately labelled source text.

Figure descriptions use source context and replace unconfirmed upstream model transcriptions. Complete visual facts are split across text chunks with their source location; one figure still has one visual vector. Truncated model responses retain complete facts and carry a quality flag. Headings and continuation tables are joined using section and layout information.

Decoration is retained as a description hint; image size, filename, and that hint do not remove source images. Deduplication applies to repeated model assertions, while native text, repeated source rows, and code are preserved. Native Word content is located by section when no page layout is available.

## Knowledge graph construction

Graphs are optional per knowledge base. The system derives entity types and
relationships from document samples.

| Stage | Work |
|---|---|
| Schema preparation | Derive a versioned LinkML schema with eight upper classes, inherited types, identity scopes, and relationship definitions |
| Extraction | Group chunks into extraction units; collect entities, relationships, and structured facts with sources |
| Deterministic routes | Extract code and configuration structure directly; infer table column roles, then turn rows into entities and qualified facts |
| Merge and reconciliation | Resolve entities and property concepts; normalize units; preserve conflicting evidence and time information |
| Derived views | Compile subject, timeline, and source pages from available facts and relationships |
| Publication | Write versioned Qdrant graph collections and, when enabled, the Neo4j projection; coordinate activation with failure and rollback checks |

Extraction responses are cached by input and configuration fingerprint.
Incremental builds reuse unchanged extraction results, merge decisions, and
vectors. Deleted documents are removed from the next graph version. Time or
change thresholds can trigger a full rebuild instead, refreshing labels by default.
A global build lock serializes graph publication across libraries.

Schema evolution retains definitions supported by current source evidence while
admitting newly represented subjects. Each induction pass proposes a compact
menu; retained definitions do not consume its proposal budget. Existing meanings,
directions and cardinalities remain stable. Additive relationship endpoints need
named, typed endpoints and a matching source quotation; usage frequency alone
does not expand a constraint. The schema can therefore follow gradual changes in
the corpus while keeping older supported categories interpretable.

Configuration facts retain complete field values and point to the source chunks
that contain each field. Fact normalization preserves identifiers, quantities,
explicit subjects and conditions. Ambiguous abbreviations remain separate;
code imports resolve within their project and package context. Evidence locations
are carried through both graph stores and retrieval responses.
Packed code retains the symbols located in its source ranges. Overlapping native
table rows share source-cell facts and keep their evidence links; dense tables
are partitioned for extraction with column context retained. Name-derived
relationship candidates remain available for review and do not guide traversal.

Graph work lives under `runtime/graph/`; the build lock coordinates graph
processes. Relationships and compiled pages carry source references that
agents can use to retrieve the underlying passages.

The eight upper classes are `agent`, `entity`, `part`, `property`, `process`,
`event`, `concept`, and `document`. Entities distinguish classes from instances,
carry external identifiers when present, and retain links to their source
passages. Per-library correction files supply reusable identity and document
status rules. Current and historical views preserve superseded evidence.

Carrel supplies general knowledge storage, retrieval, and evidence.
**The Agent Loop plugin is in development and not yet implemented.** It is
intended to use these interfaces to learn an enterprise ontology, ask people
for clarification, and maintain correction rules.

## Retrieval

The query path uses embedding and scoring models; answer generation belongs
to the calling agent.

1. Probe vector and keyword evidence across libraries, unless the caller
   explicitly supplies `kbs`. Image queries can also contribute visual evidence.
2. Select libraries using evidence scores. When selected libraries produce no
   usable results, automatic routing can widen the search.
3. Add graph and visual candidates where those capabilities are available.
4. Fuse candidates with reciprocal rank fusion, balance subjects or documents,
   and optionally rerank text and images in parallel. Each branch applies its
   own relevance threshold, then their ranks are fused; unavailable image scores
   fall back to text.
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

## Scoped corrections

Each library has a persistent identity in `config/corrections/<kb_id>.yaml`. A new library gets a new identity even when its numeric ID is reused. Corrections refer to the reviewed graph version, source units and schema fingerprint. Changes to that evidence leave the record visible with a stale status for another review. Records apply during the next graph build and can be revoked.

| Console API | Purpose |
|---|---|
| `GET /api/kbs/{kb_id}/corrections` | Library identity, file revision, records and last build outcomes |
| `POST /api/kbs/{kb_id}/corrections/preview` | Resolve targets and validate evidence without writing |
| `POST /api/kbs/{kb_id}/corrections` | Commit the validated record |
| `POST /api/kbs/{kb_id}/corrections/{id}/revoke` | Revoke with `submitted_by` and `reason` |
| `GET /api/kbs/{kb_id}/graph_comparisons` | List retained comparison reports; `?version=...` returns details |

A record carries `id`, `action`, `targets`, `base_graph_version`, `library_identity`, `revision`, `submitted_by` and `evidence`. Actions are `entity.type`, `entity.aliases`, `entity.merge`, `entity.separate`, `entity.reject`, `relation.reject` and `fact.reject`. Type and alias changes also carry `value`. Targets can use entity keys, external IDs, document/unit IDs, scope, fact properties, values, conditions and times. Every selector must resolve unambiguously. Repeating the same record ID and content is idempotent; concurrent edits are checked against `revision`.

Carrel records and applies the correction. People and the future Agent Loop supply its business interpretation and supporting evidence. Original document chunks remain available.
