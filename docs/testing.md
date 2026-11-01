# Test scope

[Home](../README.md) · [Operations](operations.md) · English | [中文](testing.zh-CN.md)

Carrel separates source-level regression from deployment checks and evaluation
against a running knowledge base. Passing one layer establishes the behavior
covered by that layer.

| Layer | Representative tests | Contract |
| --- | --- | --- |
| Parsing and source fidelity | `test_source_fidelity`, `test_pdf_evidence`, `test_short_table_recovery`, `test_section_provenance`, `test_visual_retention` | Preserve original content and locations across parsing, chunking and evidence readback; use synthetic inputs and model stubs. |
| Graph and ontology | `test_graph_integrity`, `test_graph_semantics`, `test_ontology_invariants`, `test_temporal_schema_reuse`, `test_knowledge_foundation` | Preserve identity, predicate constraints, source attribution, temporal scope and graph-version behavior. |
| Governance | `test_correction_api`, `test_enrollment_corrections` | Validate and scope corrections, detect stale evidence, protect library identity and handle concurrent changes. |
| Retrieval and agents | `test_search`, `test_retrieval_quality`, `test_dual_rerank`, `test_search_skill` | Preserve ranking, evidence quality, source scope and agent-facing result structure. |
| Lifecycle and console | `test_scanner_worker`, `test_graph_build`, `test_console_api`, `test_console_stage_language` | Queue ownership, cancellation, resume, publication, bilingual output and progress. |
| Public deployment | `test_deployment`, `test_remote_vlm_deployment`, `test_reranker_endpoints`, `test_maintenance_ops` | Public Compose profiles, endpoint configuration, local service controls and maintenance templates. |
| Platform integration | `test_macos_sync` and installed-unit checks in `test_maintenance_ops` | Host-specific shell behavior and installed service definitions. |

The names above refer to files in `app/tests/`, with a `.py` suffix. Run the
suite from `app/` with `.venv/bin/python -m pytest -q`. JavaScript translation
tests also need Node.js on `PATH`; CI installs Node.js explicitly. A platform
skip means that host-specific behavior was not exercised in that run.

## Comparing a private deployment

Keep shared behavior assertions aligned. Parsing content, source identity,
ranking, correction scope, explicit time filtering, cancellation and graph
publication have the same meaning in a private deployment and the public
package.

Deployment assertions follow the installation they describe:

- The public package uses `carrel-*` containers and systemd units, configurable
  endpoints and loopback defaults. Private installations can use different
  names, paths, bind addresses and model settings.
- The public MinerU container selects a CPU/GPU backend from the host; a private
  installation can select a fixed backend. Check each against its own Compose
  file and service configuration.
- Human-readable backend messages may differ in language. Assert stable fields,
  state changes and error conditions, then verify the Chinese and English UI
  output with the real translation code. File names and entity names retain
  their original content.
- Compatibility checks for hosted model endpoints belong to the public suite.
  A private endpoint's accepted request parameters are checked separately when
  that service is deployed or changed.
- Match tests by behavior as well as file name. A private audit file can map
  to several public feature suites; renamed tests retain their original cases
  and negative assertions.

Keep installation paths, credentials, private question sets and source
documents outside the public tree. Public fixtures use synthetic libraries,
documents, entities and endpoint addresses.

## Live evaluation

`search eval`, source-to-chunk inspection, graph fact checks and fresh-install
smoke checks are separate from the mocked regression suite. Record the exact
source revision, model IDs, vector dimensions, enabled channels and evaluation
set with their results. After a model change, earlier retrieval scores still
describe the earlier model configuration. See [Retrieval evaluation](retrieval.md#evaluate-retrieval).
