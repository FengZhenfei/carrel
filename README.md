# Carrel

English | [中文](README.zh-CN.md)

[![tests](https://github.com/FengZhenfei/carrel/actions/workflows/tests.yml/badge.svg)](https://github.com/FengZhenfei/carrel/actions/workflows/tests.yml) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Ontology-Augmented Generation for AI agents.**

Carrel is a self-hosted knowledge base for AI agents. It automates document
parsing, indexing, and retrieval, with optional knowledge graphs. Agents use
its API to retrieve source passages, images, and related facts, then generate
their own answers.

Use it to give an agent access to a collection of product documents, research,
contracts, meeting notes, or other files that change over time. A web console
manages knowledge bases, models, and processing tasks. A bundled
[search skill](skills/carrel-search/SKILL.md) connects agents such as Claude
Code and Codex to the retrieval service.

[Features](#features) · [Evaluation](#retrieval-evaluation) · [Design](#design) · [Quick start](#quick-start) · [Agent integration](#agent-integration) · [Documentation](#documentation)

## Features

**0.3.1:** [Release notes and upgrade steps](docs/releases/0.3.1.md).

- **Document parsing:** PDF, Word, PowerPoint, spreadsheets, images, HTML,
  Markdown, notebooks, source code, and workflow files, with processing suited
  to each format. [Format details](docs/architecture.md#ingestion).
- **Automatic updates:** scan for file changes, update indexes, and maintain
  enabled knowledge graphs through scheduled tasks.
- **Hybrid retrieval:** combine vector, keyword, graph, and visual search,
  with automatic knowledge-base selection and optional text/image reranking
  with rank fusion.
- **Knowledge graphs:** extract entity types, relationships, and facts from
  your documents; retain sources, time information, and conflict indicators.
- **Agent access:** search through an API, fetch surrounding text or original
  images, and follow graph relationships when more evidence is needed.
- **Web console:** configure processing, inspect chunks and graphs, track
  tasks, and manage services. Available in English and Chinese.

## Retrieval evaluation

The 0.3.0 evaluation compared retrieval strategies on 176 documents and 99
questions. With the graph enabled, dual reranking increased image hit@5 from
4/8 to 7/8. With text reranking, enabling the graph increased complete source
coverage for cross-document questions from 5/10 to 8/10.

The full pipeline achieved 88.46% hit@5 and 0.7327 MRR@12 on 78 ranking
questions. See the [evaluation report](docs/evaluations/retrieval-evaluations-0.3.0.md)
for model configurations, comparison results, and latency.

## How it works

```mermaid
flowchart LR
    D[Document folders] --> P[Parse and chunk]
    P --> I[Vector and keyword indexes]
    P --> G[Optional knowledge graphs]
    I --> R[Retrieval API]
    G --> R
    R --> A[AI agent]
    A -->|Follow-up queries| R
```

Carrel uses MinerU and native parsers for ingestion, Qdrant for vectors,
OpenSearch for keywords, and Neo4j for graph queries. SQLite tracks files and
tasks. Model services can run locally or through compatible hosted endpoints.
See [Architecture](docs/architecture.md) for the pipeline and retrieval design.

## Design

Carrel provides general knowledge storage, retrieval, and evidence interfaces.
Entities, relationships, and facts retain sources, time information, and
conflicts so agents can check and use them alongside the original documents.
**The Agent Loop plugin is in development and not yet implemented.** It is
intended to use these interfaces to continuously learn an enterprise ontology,
seek clarification from relevant people, and maintain correction rules.

Routine graph updates merge new and changed content incrementally. Configured
time or change thresholds trigger full rebuilds, balancing update cost with
overall consistency. Parsing, indexing, and graph construction are maintained
separately: rebuilding a graph can reuse parsed content and chunks while
preserving RAG indexes.

## Requirements

- Python 3.12+ and Docker 25+ with Compose v2.
- A text embedding endpoint; a vision-language endpoint for image descriptions;
  and a chat model for knowledge graph construction, if enabled.
- An NVIDIA GPU and NVIDIA Container Toolkit to run the optional local model
  stack. CPU parsing is also supported. Resource needs depend on the models
  and collection size.

| Platform | Deployment |
|---|---|
| Linux with NVIDIA GPU | GPU parsing and optional local model services; systemd scheduling |
| Linux without GPU | CPU parsing; connect to model services on another host or a provider |
| macOS with Docker Desktop | CPU containers and native Python services; arrange scheduling separately |
| Windows | Use WSL2; native Windows deployment is not supported |

[Example setup](docs/example-setup.md) describes the author's installation
on an NVIDIA DGX Spark and some lighter alternatives.

## Quick start

### 1. Deploy the services

```bash
git clone https://github.com/FengZhenfei/carrel.git
cd carrel
./deploy.sh
```

The script detects the host, starts the infrastructure, installs the Python
app, and creates configuration files. On Linux with user systemd available,
it also enables services and scheduled tasks. Other hosts receive manual
startup commands. First startup downloads the parser models.

### 2. Configure model endpoints

Edit `config/knowledge-base.env` to set the embedding and vision model URLs,
model names, and API keys. The initial URLs point to local model servers;
starting those servers requires the optional local model setup below.

To use the bundled local models, prepare their weights and deploy with
`./deploy.sh --with-local-models`. See [Model configuration](docs/configuration.md#model-endpoints)
and the [local model setup](deployment/compose/README.md#local-model-servers-profile-local-models).
Restart running application services after changing their environment settings.

### 3. Add documents

Open the console at **http://127.0.0.1:9800**. Put documents in
`runtime/mirror/<folder>/`, then enable that folder in the console. Each
enabled folder becomes a knowledge base; scheduled tasks process its files.

For a knowledge graph, register a chat model with its **Max Input** in the console, select it in the
knowledge-base settings, and enable graph building. Check service health and
task progress in the console before making your first query.

## Agent integration

Install the bundled [carrel-search skill](skills/carrel-search/SKILL.md)
in your agent, or call the API directly:

```bash
curl -sS http://127.0.0.1:9810/search \
  -H 'Content-Type: application/json' \
  -d '{"question": "What are the delivery and acceptance requirements?", "top_k": 8}'
```

If you set `KB_SEARCH_TOKEN`, add an `Authorization: Bearer …` header.
For a remote agent,
configure `CARREL_SEARCH_BASE_URL` and `CARREL_SEARCH_TOKEN` in its environment.

Responses include source passages and, when available, graph evidence and
image information. Agents can request more context, inspect images, and
follow relationships before answering. See [Retrieval and API](docs/retrieval.md)
for authentication, response status, and citations.

## Documentation

| Guide | Contents |
|---|---|
| [Configuration](docs/configuration.md) | Model endpoints, folders, per-base settings, and access control |
| [Architecture](docs/architecture.md) | Parsing, graph construction, retrieval, and code layout |
| [Retrieval and API](docs/retrieval.md) | Agent setup, endpoints, evidence status, and evaluation |
| [Retrieval evaluation (0.3.0)](docs/evaluations/retrieval-evaluations-0.3.0.md) | Models, test conditions, graph and reranking comparisons, and latency |
| [Operations and development](docs/operations.md) | Console, CLI, scheduling, troubleshooting, and tests |
| [Example setup](docs/example-setup.md) | The author's installation on a DGX Spark, a Mac that syncs the documents, and lighter alternatives |
| [Docker Compose](deployment/compose/README.md) | Infrastructure, model weights, and container settings |
| [systemd](deployment/systemd/README.md) | Linux services and timers |

## Data and access

The console binds to localhost by default. For LAN access, set both
`KB_WEB_HOST` and `KB_WEB_TOKEN`. Configure `KB_SEARCH_TOKEN` before using
protected search endpoints remotely. The stores and model servers listen on
loopback only, which keeps other machines out but not web pages opened in a
browser on the same host, so avoid browsing the web there.

Documents and images are sent to the model endpoints you configure. Local
endpoints keep model processing within your own infrastructure.
See [Access and data handling](docs/configuration.md#access-and-data-handling).

## License

Carrel is licensed under [MIT](LICENSE). Dependencies and model weights have
their own licenses; see [Third-party notices](NOTICE.md). Original-resolution
PDF image retrieval uses an optional dependency installed with
`./deploy.sh --with-pdf-images`; without it, retrieval uses cached images.
