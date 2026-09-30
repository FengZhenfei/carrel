# Operations and development

[Home](../README.md) · English | [中文](operations.zh-CN.md)

Commands below assume the repository root unless a `cd` command says otherwise.
`kb` refers to `app/.venv/bin/kb`; the package also installs a `carrel` alias.

## Console

The console provides three main views:

| View | Controls |
|---|---|
| Settings | Chunking, graph settings, model selection, rebuild policy, and processing actions |
| Files | Parse status, chunk counts, errors, retries, chunk previews, and job timelines |
| Graph | Current entities and relationships, type filters, and merge records |

The top bar provides the model registry, service health and controls, and the
language switch. Saving settings only stores configuration. Use full re-parse
after changing chunking; parser version changes can requeue affected files
automatically on the next scan.

Disabling a knowledge base deactivates its indexes and starts its retention
lifecycle. Re-enabling during retention can restore it. Folder rename detection
can retain the same ID; ambiguous matches can be resolved through the console's
adopt action.

Deleting a knowledge base stops its running parse and graph jobs, then removes
its indexes, graph data, state, and caches at once, without the retention
period. The base is shown as being deleted until that finishes. If a store
could not be cleared, the base is shown as not fully deleted and cannot be
re-enabled or adopted; the next nightly maintenance run finishes the deletion,
or you can delete it again. A deletion holds the graph build lock: no graph
build starts while it runs, and a base cannot be deleted while another one is
building its graph.

The console's administrative API lives under `/api` on port 9800. Writes
must come from the console's own origin, and every call needs the token when
`KB_WEB_TOKEN` is set (see
[Configuration](configuration.md#access-and-data-handling)). Request bodies
are defined in [`kb_server/api.py`](../app/kb_server/api.py).

| Route | Purpose |
|---|---|
| `GET /api/overview`, `/api/health`, `/api/limits` | Sidebar state, service health, chunking limits of the embedding model |
| `POST /api/enroll`; `POST /api/kbs/{kb_id}/adopt`, `unenroll`; `DELETE /api/kbs/{kb_id}` | Enable a folder, adopt a renamed one, disable or delete a base |
| `GET` / `PUT /api/kbs/{kb_id}/config` | Read and save the per-base settings |
| `POST /api/kbs/{kb_id}/parse_now`, `reparse` | Queue new files, re-parse everything |
| `GET /api/kbs/{kb_id}/files`, `files/{file_id}/chunks` | File states, one file's chunks |
| `POST /api/kbs/{kb_id}/graph_schema`, `graph_build`, `graph_append`, `graph_pause`; `DELETE /api/kbs/{kb_id}/graph` | Extract labels, build, append, pause, delete the graph |
| `GET /api/kbs/{kb_id}/graph_preview`, `graph_merges`, `graph_corpus` | Current graph, merge records, corpus size |
| `GET /api/jobs/{job_id}`; `POST /api/jobs/{job_id}/cancel`, `/api/files/retry` | One job and its timeline, cancel, retry |
| `GET` / `POST /api/llms`; `DELETE /api/llms/{name}` | Model registry (keys are write-only) |
| `POST /api/services/{key}/restart`, `restart_all`, `stop_all` | Service controls for the rows in `KB_CONSOLE_SERVICES` |

## Deployment and services

| Command | Effect |
|---|---|
| `./deploy.sh` | Deploy using detected settings; keep existing configuration |
| `./deploy.sh detect` | Report environment and proposed settings |
| `./deploy.sh --cpu` | Request the CPU parser setup |
| `./deploy.sh --with-local-models` | Include the optional model services; weights must be prepared |
| `./deploy.sh --with-pdf-images` | Install optional PDF original-image support |
| `./deploy.sh status` | Inspect containers and console health |
| `./deploy.sh down` | Stop/remove Compose containers while keeping data; application systemd units are separate |
| `./deploy.sh purge` | Remove the deployment's containers, built parser images, units, and Compose configuration |

`purge --images` also removes pulled images; `purge --data` deletes runtime
data and model weights. Back up the data you want to keep before using these
removal options.

For Linux services:

```bash
systemctl --user list-timers 'carrel-*'
journalctl --user -u carrel-worker -f
journalctl --user -u carrel-graph-rebuild --since -3h
journalctl --user -u carrel-web --since -5min
```

The [systemd guide](../deployment/systemd/README.md) covers installation,
lingering, unit templates, and manual scheduling on hosts without systemd.
The [Compose guide](../deployment/compose/README.md) covers backend selection,
model weights, memory settings, and container logs.

## Scheduled tasks

| Unit | Schedule after activation | Purpose |
|---|---|---|
| `carrel-scan.timer` | 30 seconds, then every minute | Scan and queue file changes |
| `carrel-worker.timer` | 1 minute, then every 5 minutes | Process queued ingestion work |
| `carrel-graph-rebuild.timer` | 10 minutes, then every 2 hours | Incremental graph updates or policy-triggered rebuilds |
| `carrel-qdrant-gc.timer` | 30 minutes, then every 24 hours | `kb cleanup backup` (see below), then `kb cleanup parse-assets-gc`, then `cleanup graph-gc`: inactive index points past retention with their chunk rows and parse assets, files deleted longer ago than the undo window, old job rows, disabled libraries past retention and unfinished deletions; graph versions beyond the active one plus N-1 (`GRAPH_GC_KEEP_VERSIONS`, regardless of age, the same rule a finished build applies) |
| `carrel-cache-weekly.timer` | 1 hour, then every 7 days | Rotate project caches; drop cached picture descriptions and visual vectors that no picture in the parse cache refers to any more |
| `carrel-logs-monthly.timer` | 2 hours, then every 30 days | Rotate logs |

The console and search API run as persistent services. Under memory pressure
the batch units are killed before them (`OOMScoreAdjust` 200 against 100).
Timers use relative intervals rather than calendar times. Maintenance can defer
while ingestion, graph building, or synchronization is active. Repeated
deferrals produce exit 75 and a visible failed-unit state; see
`scripts/lib/kb-maint-defer.sh` and the [systemd guide](../deployment/systemd/README.md#busy-yield).

Maintenance covers project data by default. Set `KB_HOST_HOUSEKEEPING=1` to
also clear user-level uv/pip caches and prune Docker caches.

The nightly backup keeps what cannot be recomputed: the state database
(library configuration, model registry, extraction and fact caches), the
pipeline env file, the compose `.env` and the evaluation sets under
`runtime/eval`, as plain copies in `backups/state/<stamp>/` with the newest
`KB_BACKUP_KEEP` (7) kept. Vectors, indexes, graphs and the parse cache are
rebuilt from the mirror. To restore, stop the timers and the two services, put
the env files and `runtime/state/kb-pipeline.db` back from the newest copy,
start the containers with `docker compose up -d`, then the services and the
timers; with the caches in the state database intact, a full rebuild does not
call the remote models again.

## CLI reference

Run `app/.venv/bin/kb --help` and each subcommand's `--help` for arguments.

| Command group | Purpose |
|---|---|
| `config`, `status`, `health` | Effective configuration, queues, and service status |
| `init-db` | Initialize the state database |
| `scan [--source kb_NNN] [--requeue-failed] [--rehash]` | Scan enabled folders and queue work; re-queue failed files; re-hash instead of trusting mtime |
| `worker --once [--max-jobs N] [--max-seconds S]` | Run a bounded ingestion pass (never start one next to the systemd worker) |
| `qdrant ensure-collections`, `ensure-graph-collections` | Create missing collections |
| `fts init/status/rebuild/sync-doc/search` | Keyword index administration |
| `graph build/append/check-rebuild [--execute] [--force-full]` | Full build, incremental append, policy check (what the timer runs) |
| `graph adopt-current/rollback/neo4j-import/neo4j-status/neo4j-delete [--graph-version V]` | Graph version baseline, rollback to a kept earlier version, and Neo4j projection |
| `graph query/factcheck/status` | Graph inspection and evaluation |
| `search eval/make-set` | Retrieval evaluation and question-set preparation |
| `cleanup status/weekly/monthly/parse-assets-gc/graph-gc/backup [--dry-run]` | Maintenance status, the cleanups the timers run and the nightly backup |
| `cleanup qdrant-gc/qdrant-graph-gc/neo4j-graph-gc [--dry-run]` | Manual tools: `qdrant-gc` deletes expired inactive points only; the other two prune graph versions by age (the two graph retention-days keys) |
| `reset --source kb_NNN [--all] --yes` | Wipe one base's state, caches and indexes and recreate its collection |

When systemd manages the worker, trigger its unit to run an ingestion pass. For example, using an actual knowledge-base ID:

```bash
app/.venv/bin/kb scan --source kb_001
systemctl --user start --no-block carrel-worker.service
```

`reset` without `--yes` prints a plan. Review that plan and its library scope
before confirming; use the console's re-parse action when a full reset is not
needed. Normal deployment preserves existing library data.

## Troubleshooting

Start with `./deploy.sh status`, `app/.venv/bin/kb status`, and the relevant
service logs. The console health API is `/api/health`; include the console
token if configured. `app/.venv/bin/kb graph status` reports graph versions
and build state.

| Symptom | Check |
|---|---|
| Files stay queued | `KB_PARSE_ENABLED`, worker logs, active locks, and whether model endpoints are configured and reachable. A job that cannot connect to a service it needs, such as the parser or a model endpoint, returns to the queue without using up a retry and waits twice as long each time; after `KB_JOB_MAX_RETRIES` such rounds it fails like any other error |
| A deleted base shows as not fully deleted | A store could not be cleared: check Qdrant, OpenSearch, and Neo4j, then delete it again or let the nightly maintenance finish it |
| Graph build reports an existing lock | Check for an active build; the flock is released when its process exits |
| Maintenance exits with 75 | Consecutive deferrals because the system is busy; inspect ingestion, graph, sync, and Qdrant status |
| Image processing fails | Check the image-description endpoint and, if enabled, the visual embedding endpoint separately |
| Unexpected parser backend | `docker logs carrel-mineru` and `MINERU_BACKEND_POLICY` in the Compose environment |
| Search returns weak or diagnostic results | Inspect `retrieval_summary`, source status, routing, and available channels before changing thresholds |
| Neo4j projection needs rebuilding | Check logs and backups first; `kb graph neo4j-import` can project the current graph after the service is available |

For store recovery, preserve the state database and relevant runtime data
before rebuilding indexes or projections.

## Development and tests

`deploy.sh` installs the pinned set from `app/requirements.lock`, the versions
the suite was run against (pytest included); by hand that is
`app/.venv/bin/pip install -r app/requirements.lock -e app`. After a dependency
change, regenerate the lock from a virtual environment you have tested with
`app/.venv/bin/python scripts/kb-lock-deps.py app/pyproject.toml > app/requirements.lock`.
After installing the application and test dependencies:

```bash
cd app
.venv/bin/python -m pytest -q
```

Regression tests use service stubs and temporary data; the suite can run
without live containers. Some checks skip when optional dependencies or
installed systemd units are absent. Retrieval evaluation against a real
deployment is a separate step, described in [Retrieval](retrieval.md#evaluate-retrieval).

- When parser output changes, update its profile/version in
  `parsers/common.py` so affected documents can be reprocessed.
- Console strings belong in the `static/i18n.js` language dictionary. File names
  and entity names retain their source language.
- Static frontend edits need no service restart. Python changes require
  restarting the affected application service. Check active work first: a
  label extraction running inside the console is lost, and a knowledge-base
  deletion is interrupted until the nightly maintenance finishes it.
- After editing systemd templates, run `./scripts/install-systemd.sh --check`
  to inspect differences, then reinstall the units as needed.
- Use synthetic regression fixtures and keep runtime data and credentials
  outside the source tree.
  `app/tests/test_deployment.py` includes repository privacy checks, with
  machine-specific patterns stored outside the repository.
