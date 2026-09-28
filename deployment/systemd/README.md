# systemd user units

On Linux the app runs as user-scope systemd units: two long-running services
(console, search API) and six timers that drive ingestion and maintenance.
`deploy.sh` installs and enables them; this page is the reference.

## Units

| Unit | Cadence | Does |
|---|---|---|
| `carrel-web.service` | always on | console (`python -m kb_server`, port 9800) |
| `carrel-search.service` | always on | search API (`python -m kb_search`, port 9810) |
| `carrel-scan.timer` | 30 s after enable, then every minute | scan the mirror, queue new / changed / deleted files |
| `carrel-worker.timer` | 1 min after enable, then every 5 min | drain the queue: parse → chunk → embed → write |
| `carrel-graph-rebuild.timer` | 10 min, then every 2 h | append new documents to the graph; full rebuild when the policy says so |
| `carrel-qdrant-gc.timer` | 30 min, then every 24 h | point GC, expired knowledge bases, old graph versions, job history |
| `carrel-cache-weekly.timer` | 1 h, then every 7 days | parse-cache rotation |
| `carrel-logs-monthly.timer` | 2 h, then every 30 days | log rotation |

Timers use relative time only (`OnActiveSec` / `OnUnitActiveSec`): no wall
clock, no timezone, the same rhythm after a reboot or a re-enable.

## Install

The files here are templates: `__CARREL_HOME__` becomes the checkout path
when `scripts/install-systemd.sh` renders them into
`~/.config/systemd/user/`. The installed copies are files, not symlinks, so
re-run the script after editing a unit.

```bash
./scripts/install-systemd.sh            # render + install + daemon-reload; never restarts a running unit
./scripts/install-systemd.sh --check    # compare only; exit 1 and show a diff on drift
./scripts/install-systemd.sh --enable   # also `enable --now` every timer and service
./scripts/install-systemd.sh --uninstall
```

User units stop when the last session of the user ends unless lingering is
enabled once:

```bash
sudo loginctl enable-linger "$USER"
loginctl show-user "$USER" -p Linger      # must print Linger=yes
```

## Busy-yield

The maintenance timers (graph check, GC, cleanups) step aside while the
system is busy: a running parse or graph build, a sync tool holding
`runtime/state/mirror_sync.lock.d`, or an unreachable Qdrant. Yielding exits
quietly and counts the round in `runtime/state/maintenance/`; only after
several consecutive rounds does the unit go to `failed` (exit 75) so that the
console's "scheduled tasks" list turns red. The policy lives in
`scripts/lib/kb-maint-defer.sh` and is shared by every maintenance script.

The worker has a soft budget (`KB_WORKER_MAX_SECONDS`, default 90 minutes,
checked between files) and a hard `TimeoutStartSec` of 4 hours as the last
resort against a hung parse.

## Logs

```bash
journalctl --user -u carrel-worker -f
journalctl --user -u carrel-graph-rebuild --since -3h
journalctl --user -u carrel-web --since -5min
systemctl --user list-timers 'knowledge-base*'
```

Restarting `carrel-web.service` kills label extraction and chunk
previews that run synchronously inside the web process; check its journal
first. Changes under `app/kb_server/static/` need no restart.

## Without systemd

macOS and containers have no user systemd. `deploy.sh` prints the equivalent
manual commands (console, search API, one ingest round); a cron entry or a
launchd agent calling `scripts/kb-pipeline-scan.sh` and
`scripts/kb-pipeline-worker-once.sh` gives the same behaviour.
