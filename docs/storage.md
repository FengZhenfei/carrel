# Storage adapters

[中文](storage.zh-CN.md)

Standard and Light share a storage boundary. Parsing workers, schema sampling, graph builds, the retrieval service and
console chunk previews use `kb_pipeline.storage.get_storage()`.

## Operations

| Interface | Responsibilities | Standard | Light |
|---|---|---|---|
| `ChunkStore` | Collection layout, vector writes and queries, payload/context reads, version retirement, metadata updates and restoration | Qdrant | LanceDB |
| `KeywordStore` | Document projection synchronization, keyword candidates, phrase lookup and lexical routing evidence | OpenSearch | LanceDB full-text search |
| `GraphStore` | Build input, versioned vector/projection writes, publication and rollback, graph queries, facts, dates and source readback | Qdrant and Neo4j | LanceDB and DuckDB |
| `Storage` | Assemble the adapters, library catalog and request-scoped deadlines | Standard adapter factory | Light adapter factory |

SQLite continues to own configuration, queues, chunk ledgers and build records.
Original files and parsing artifacts retain their existing paths.

The Standard adapters delegate to the existing storage implementations. Parsing,
chunking, entity extraction, schema evolution, ranking and evidence assembly retain
their existing algorithms. Document versions, graph versions, time scopes and
source locators pass through the boundary unchanged.

Document publication writes vectors first, then commits the complete chunk ledger,
indexed version and cleanup job in one SQLite transaction. Both backends check
retrieval and evidence reads against that ledger. Partial writes remain unpublished;
old-point retirement and keyword synchronization retry through the existing queue.
Chunk identifiers include the evidence text and document layout, and context
readbacks can pin `chunk_revision` as well as the file's `content_version`.

Standard graph publication retains its original order: write the new vectors, import the
graph projection, switch vector aliases, activate the graph projection, then record
success. An activation failure restores the previous aliases, including removing
an optional page alias when the target version has no page index. Console
previews can read the preceding ledger version while a reparse is being published.

Connections are created on demand. Concurrent calls on one adapter share its vector
client. Search configuration refreshes reuse that connection when its endpoint,
credential and timeout are unchanged, while in-flight requests retain their settings.
Existing request deadlines and channel degradation behavior are preserved.

## Verification

Run the local suite from the repository root:

```sh
app/.venv/bin/python -m pytest -q app/tests
```

`test_storage_contract.py` uses synthetic Markdown, fenced code, plain text and CSV
with real parsers, SQLite and an in-memory Qdrant client. It checks content retention,
source IDs, vector/filter equivalence, graph input, replacement and empty-document
withdrawal. Model responses and the OpenSearch transport are controlled fixtures.
Existing regressions also cover incremental graph builds, deletion and rename,
publication rollback, resume, time/conflict retrieval, image references and endpoint
configuration refresh.

`test_publication_integrity.py` injects vector-write, cleanup and ledger-commit
failures against real LanceDB and in-memory Qdrant. It also verifies missing-index
recovery, revision invalidation and keyword counts over published chunks.

`test_storage_live.py` runs the same file lifecycle against real Qdrant,
OpenSearch and Neo4j servers. It also builds two graph versions, checks source
readback and injects an activation failure to verify rollback and retry. Model
responses remain synthetic so storage behavior is repeatable.

This opt-in suite requires **dedicated, disposable databases** on loopback ports
16333 (Qdrant), 19200 (OpenSearch) and 17687 (Neo4j with authentication disabled).
It checks for empty Qdrant and Neo4j stores and removes its collections and graph
nodes after each test. Do not map these ports to an existing installation.

```sh
CARREL_STORAGE_INTEGRATION=1 app/.venv/bin/python -m pytest -q app/tests/test_storage_live.py
```

## Light storage

`light_data.py` implements native chunk storage, exact cosine vector search and
keyword candidates in LanceDB. Payloads and vector/keyword fields share a row;
missing visual vectors remain null. It supports source filters, context reads,
file replacement, metadata updates, reactivation and literal phrase lookup.
Keyword scoring adds field contributions and indexes filename/path substrings.
Its BM25 rankings can differ from OpenSearch; vector, keyword and combined
candidate results are evaluated separately.

`light_graph.py` stores versioned nodes and edges in immutable LanceDB tables.
SQLite publishes a complete generation with an atomic pointer update. Readers pin
that generation, and retries reuse tables left by an interrupted write. Published
and staged version IDs cannot be reused for different graph content. DuckDB runs
in-process evidence joins with bounded memory and temporary disk use. Source dates,
conflicts and governance use the existing shared scope rules.

Install the optional dependencies to run the native storage regressions:

```sh
app/.venv/bin/python -m pip install -e './app[light]'
app/.venv/bin/python -m pytest -q app/tests/test_light_data.py app/tests/test_light_graph.py app/tests/test_light_integration.py app/tests/test_light_lifecycle.py
```

These tests cover synthetic ordinary text, code and tables, text/visual filtering,
keyword updates, concurrent writes, source evidence, time/conflict scopes, immutable
history, pinned readers and process termination during graph staging. If the optional
dependencies are installed, the live integration suite also compares both graph
versions with Neo4j node properties and evidence paths.

## Queries and lifecycle

Light exposes the same entity listing, one-hop neighborhoods, facts, conflicts,
time series, compiled pages and original-source readback as Standard. Common
modules continue to own ranking, time scopes and evidence formatting. Entity
aliases, merged IDs, document versions and stale-reference checks retain their
existing meaning. Exact vector search is the Light default.

Graph vectors and the projection are staged before a single SQLite pointer
publishes both. Publication also seals that version against further vector writes.
A failed preparation leaves the previous version active. Rollback changes the same
pointer. Readers pin their selected generation; garbage collection waits for those
readers, protects active and resumable versions, and applies the configured history
count and retention period. Interrupted projection and vector writes are reclaimed
after retention. Inactive-chunk and parse-asset cleanup use native storage reads.
The weekly maintenance job also compacts chunk tables and reclaims obsolete Lance
files, retaining a seven-day window for recent internal snapshots and unverified
writes. This storage window is separate from graph history retention.

The Light integration tests run real parsers, the worker, graph build, publication,
HTTP retrieval and graph endpoints against temporary LanceDB/DuckDB data. External
model replies are deterministic fixtures. Further regressions cover expired and
active data, dry-run reclamation, rollback, interrupted publication, stale references,
query deadlines, temporal/conflict pagination and source evidence.

## Deployment

Choose an edition for a fresh deployment directory:

```sh
./deploy.sh --standard --build  # Qdrant, OpenSearch and Neo4j containers
./deploy.sh --light --build     # embedded LanceDB and in-process DuckDB
```

Both include the MinerU parser and use the same application image, with both storage
adapters installed. The edition is recorded in `data/runtime/deployment-mode` and
`KB_STORAGE_BACKEND`. Re-running the script preserves it; changing editions requires
a separate deployment directory. There is no console mode selector or migration.
Light stores its data in `data/runtime/storage` (`KB_STORAGE_ROOT` can set another path),
and its console omits System services. Native storage checks exercise real reads;
package or directory existence alone is not a health check.

Embedding, reranking and chat/vision models are configured as external endpoints.
Neither edition starts their containers or downloads their weights. MinerU's own
parser weights are downloaded on its first startup.
