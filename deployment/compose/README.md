# Container deployment

[Home](../../README.md) · English | [中文](README.zh-CN.md)

Light and Standard use the same application image. The application container runs the
console, search API and background scheduler. MinerU is a separate integrated parser.
Standard also starts Qdrant, OpenSearch and Neo4j; Light stores data in LanceDB,
DuckDB and SQLite inside the persistent application directory.

LLM, Embedding and Reranker services are configured independently in the console.
Their weights and servers are not part of this deployment.

The application image includes grammar libraries for Carrel's supported code languages.
They are prepared during the image build, so code parsing does not need to download them
on first use. Python parsing uses Python's built-in AST.

## Start

Install Docker with Compose v2. From the repository root:

```bash
./deploy.sh --light --build
# Or choose Standard:
# ./deploy.sh --standard --build
```

The script is an optional Compose shortcut. It creates `deployment/compose/.env`,
generates a database password, builds the app and parser images, and starts the
selected services. Python and systemd are not required on the host. CPU parsing is
the default; use `--gpu` with NVIDIA Container Toolkit for GPU parsing. The parser
chooses its backend using the existing MinerU hardware checks.

Open **http://127.0.0.1:9800** and configure model endpoints in **System settings**.
Place source directories under `data/runtime/mirror/`, then enable them in the
console. The initial application config is `data/config/knowledge-base.env`.
The generated `KB_SEARCH_TOKEN` in that file is used by search clients.

First startup downloads MinerU's parser assets. `MINERU_MODELS_PATH` retains them
across restarts. The optional `--with-pdf-images` build includes PyMuPDF for
original-resolution PDF image retrieval; its license is described in [NOTICE](../../NOTICE.md).

## Use Compose directly

```bash
cd deployment/compose
cp .env.example .env
# Edit .env: CARREL_EDITION=light and COMPOSE_PROFILES= for Light.
# For Standard: CARREL_EDITION=standard, COMPOSE_PROFILES=standard,
# and set a nonempty NEO4J_PASSWORD before startup.
docker compose up -d --build
docker compose ps
docker compose logs -f app
```

`CARREL_IMAGE=carrel-app:local` names the locally built image. To use an already
published image, set `CARREL_IMAGE` to its registry reference, then run
`docker compose pull app` and `docker compose up -d --no-build app`.
The image workflow publishes amd64 and arm64 application images. Main-branch pushes
update `ghcr.io/fengzhenfei/carrel:main` and a `sha-<commit>` tag; version-tag pushes
publish the corresponding release image. Main updates leave release tags and `latest`
unchanged. Manual dispatch is also supported.

MinerU is built separately by Compose and defaults to **3.4.5**. For an existing
installation, set `MINERU_VERSION=3.4.5` in `.env`, then run
`docker compose up -d --build mineru`. Pulling the application image alone does not
upgrade the parser. Keep the existing model and output mounts: an upgrade does not
schedule reparsing or graph rebuilding. Cached results retain their producer version;
new results record the version returned by MinerU.

For a registry mirror, set `CARREL_PYTHON_IMAGE`, `MINERU_BASE_IMAGE` and the database
image variables. `PIP_INDEX_URL` controls the Python package index used during build.
GPU parsing uses `COMPOSE_FILE=docker-compose.yml:compose.gpu.yml` with
`MINERU_FLAVOR=gpu` and an appropriate GPU base image.

## Data and networking

| Data | Location |
|---|---|
| App configuration, source documents, indexes, graphs, state, logs and backups | `CARREL_DATA_PATH`, default `data/` at the repository root |
| Standard database files | Project-scoped Docker volumes: `qdrant-data`, `opensearch-data`, `neo4j-data` and Neo4j auxiliary volumes |
| MinerU models | `MINERU_MODELS_PATH`, default `models/mineru/` |
| MinerU output and backend metadata | `MINERU_OUTPUT_PATH`, default `runtime/mineru-output/`, shared with the app |

The edition is recorded in the data directory and cannot change in place. Use a
separate directory and Compose project for another edition. No cross-mode migration
is performed. Keep the Compose project name stable when reusing Standard volumes.

Model URLs resolve from the app container. For a service on the Docker host, use
`host.docker.internal` and make that service reachable from the Docker bridge;
`127.0.0.1` refers to the app container. For another machine, use its reachable address.
The app reaches integrated services by the names `mineru`, `qdrant`, `opensearch`, and `neo4j`.

Published ports bind loopback by default. Set `APP_BIND_ADDRESS` to expose the app
and configure `KB_WEB_TOKEN` before exposing the console. Search uses its generated
Bearer token. The app does not mount the Docker socket; container start/stop and
restart are managed through Compose. Light has no System services section.

## Scheduling, restart and updates

The built-in scheduler runs scanning every minute, parsing every five minutes,
graph maintenance every two hours, daily garbage collection, weekly cache cleanup,
and monthly log rotation. Manual console actions use the same pipeline entry points.
Queue state and next-run timestamps persist in `data/runtime/state/automation.db`.
Repeated pending requests coalesce, maintenance waits for active work, and graph
builds retain the existing global lock. Restarting the web process does not stop a
separately dispatched graph build.

Stopping the container signals active tasks and allows 60 seconds for shutdown.
On restart, interrupted jobs follow the pipeline's existing recovery and cache rules;
operator-paused work remains paused. Docker's `unless-stopped` policy restarts the app
after a host reboot unless it was explicitly stopped.

```bash
./deploy.sh status
./deploy.sh down                   # retains files and database volumes
./deploy.sh --build                # rebuild after source updates; reuses .env and data
cd deployment/compose
docker compose restart app        # reload application configuration
```

App HTTP logs and task logs are under `data/logs/`; scheduler lifecycle logs are also
available through `docker compose logs app`. The container health check covers the
scheduler, console and search process. Dependency readiness is shown separately in
the console and `/health` response; unavailable dependencies defer background work.

Native/source installations retain the [systemd installer](../systemd/README.md).
Converting an existing host installation requires deliberate path and data-volume
mapping; the container helper does not move existing data or stop host services.
