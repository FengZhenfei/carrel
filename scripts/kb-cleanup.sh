#!/usr/bin/env bash
set -euo pipefail

# Default to the repo this script lives in; the old default pointed at the
# deleted Mac path and broke every invocation that had not exported
# KB_LOCAL_BASE_DIR (systemd injects it, interactive shells usually do not).
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${KB_LOCAL_BASE_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
APP_DIR="$BASE_DIR/app"
PY="$APP_DIR/.venv/bin/python"
if [[ -n "${KB_ENV_FILE:-}" ]]; then
  ENV_FILE="$KB_ENV_FILE"
elif [[ -f "$BASE_DIR/config/knowledge-base.env" ]]; then
  ENV_FILE="$BASE_DIR/config/knowledge-base.env"
else
  ENV_FILE="$BASE_DIR/app/.env"
fi

# shellcheck source=lib/kb-maint-defer.sh
source "$SCRIPT_DIR/lib/kb-maint-defer.sh"

# Every parse leaves the MinerU container's intermediate output in /data/output (runtime/mineru-output on the
# host), owned by root; the real copy is long since in parse_cache, so this one is useless and keeps piling up
# (115 directories, 2.4G, in three weeks). Nightly, along with parse-assets-gc, remove those older than a day;
# deletion runs as root inside the container, since an ordinary host user cannot remove root-owned
# directories. Skipped when the container is not running.
MINERU_OUTPUT_KEEP_MINUTES="${KB_MINERU_OUTPUT_KEEP_MINUTES:-1440}"
mineru_output_gc() {
  local name="${KB_MINERU_CONTAINER:-carrel-mineru}"
  if [[ "$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null)" != "true" ]]; then
    echo "=== $(date '+%F %T') mineru-output-gc: container $name not running, skipped ==="
    return 0
  fi
  local removed
  removed=$(docker exec "$name" sh -c "find /data/output -mindepth 1 -maxdepth 1 -mmin +${MINERU_OUTPUT_KEEP_MINUTES} -print -exec rm -rf {} + | wc -l" 2>/dev/null) \
    || { echo "=== $(date '+%F %T') mineru-output-gc: cleanup failed ==="; return 1; }
  echo "=== $(date '+%F %T') mineru-output-gc: removed ${removed:-0} output dir(s) older than ${MINERU_OUTPUT_KEEP_MINUTES} min ==="
}

# Graph-version safety net: the end-of-build GC only runs when that base builds, so idle bases and bases
# whose GC failed rely on this one. Same "active plus N-1" rule (GRAPH_GC_KEEP_VERSIONS); the whole round
# is skipped while a build runs.
graph_versions_gc() {
  local code=0
  "$PY" -m kb_pipeline --env-file "$ENV_FILE" cleanup graph-gc || code=$?
  if [[ "$code" -eq 75 ]]; then
    echo "=== $(date '+%F %T') graph-gc: a graph build is running, yielding this round (its own clean-up covers it) ==="
    return 0
  fi
  if [[ "$code" -ne 0 ]]; then
    echo "=== $(date '+%F %T') graph-gc: cleanup failed (exit ${code}) ==="
    return 1
  fi
}

# Weekly host clutter (2026-09-08, the user asked for all of it to rotate automatically):
# - runtime/scratch: the agreed place for temporary scripts / comparison output; entries older than
#   KB_SCRATCH_KEEP_DAYS (default 14 days) are removed;
# - registry backups in backups/: only the 5 most recent are kept;
# - only with KB_HOST_HOUSEKEEPING=1 is anything outside this project touched: the whole uv / pip caches,
#   Docker's dangling images and build cache unused for 30 days. Other projects on a shared machine use these
#   too, so they are left alone by default.
# A failure in one item does not affect the others and does not count as a maintenance failure.
host_housekeeping() {
  local scratch="$BASE_DIR/runtime/scratch" n
  mkdir -p "$scratch"
  n=$(find "$scratch" -mindepth 1 -maxdepth 1 -mtime +"${KB_SCRATCH_KEEP_DAYS:-14}" -print -exec rm -rf {} + 2>/dev/null | wc -l)
  echo "=== $(date '+%F %T') housekeeping: scratch: removed ${n:-0} item(s) older than ${KB_SCRATCH_KEEP_DAYS:-14} days ==="
  case "${KB_HOST_HOUSEKEEPING:-0}" in
    1|true|yes|on)
      rm -rf "$HOME/.cache/uv" "$HOME/.cache/pip" 2>/dev/null && echo "=== housekeeping: uv / pip caches cleared ==="
      if command -v docker >/dev/null 2>&1; then
        docker image prune -f 2>/dev/null | tail -1 | sed 's/^/=== housekeeping: docker image prune: /'
        docker builder prune -f --filter until=720h 2>/dev/null | tail -1 | sed 's/^/=== housekeeping: docker builder prune: /'
      fi ;;
    *) echo "=== housekeeping: host-level cleanup skipped (KB_HOST_HOUSEKEEPING is not set) ===" ;;
  esac
  if ls "$BASE_DIR"/backups/llm_registry-*.json >/dev/null 2>&1; then
    ls -1t "$BASE_DIR"/backups/llm_registry-*.json | tail -n +6 | xargs -r rm -f
    echo "=== housekeeping: registry backups kept: $(ls -1 "$BASE_DIR"/backups/llm_registry-*.json | wc -l) ==="
  fi
  return 0
}

COMMAND="${1:-status}"
shift || true

# Run on its own: bash scripts/kb-cleanup.sh mineru-output-gc
if [[ "$COMMAND" == "mineru-output-gc" ]]; then
  mineru_output_gc
  exit $?
fi
# Run on its own: bash scripts/kb-cleanup.sh housekeeping
if [[ "$COMMAND" == "housekeeping" ]]; then
  host_housekeeping
  exit $?
fi

# Only the inactive-point / parse-asset GC touches Qdrant: wait for it first (up to 5 minutes); if it never
# comes up, exit quietly under the deferral count
if [[ "$COMMAND" == "parse-assets-gc" ]]; then
  maint_wait_qdrant_or_defer "cleanup-$COMMAND" "cleanup $COMMAND: Qdrant not ready"
fi

# When a maintenance job is blocked by service_busy the CLI exits 75. A oneshot unit does not allow Restart=,
# so the retries happen here; what to do once the retries run out is decided in lib/kb-maint-defer.sh.
ATTEMPTS="${KB_CLEANUP_BUSY_ATTEMPTS:-3}"
DELAY="${KB_CLEANUP_BUSY_RETRY_SECONDS:-900}"
attempt=1
while :; do
  set +e
  "$PY" -m kb_pipeline --env-file "$ENV_FILE" cleanup "$COMMAND" "$@"
  code=$?
  set -e
  if [[ "$code" -eq 0 ]]; then
    maint_defer_clear "cleanup-$COMMAND"
    if [[ "$COMMAND" == "parse-assets-gc" ]]; then
      mineru_output_gc || true    # failing to clear the intermediate output is not a maintenance failure
      graph_versions_gc || true   # graph-version safety net; a failure here is not a maintenance failure either
    fi
    if [[ "$COMMAND" == "weekly" ]]; then
      host_housekeeping || true   # host clutter, same as above
    fi
    exit 0
  fi
  if [[ "$code" -ne 75 ]]; then
    if [[ "$COMMAND" == "parse-assets-gc" ]]; then
      graph_versions_gc || true   # the graph-version safety net does not depend on the previous step succeeding
    fi
    exit "$code"
  fi
  if [[ "$attempt" -ge "$ATTEMPTS" ]]; then
    if [[ "$COMMAND" == "parse-assets-gc" ]]; then
      graph_versions_gc || true   # graph versions are still cleaned on a night the previous step yielded
    fi
    maint_defer_give_up "cleanup-$COMMAND" \
      "cleanup $COMMAND hit service_busy ${attempt} times in a row"
  fi
  echo "=== $(date '+%F %T') cleanup $COMMAND deferred (busy), retrying in ${DELAY}s (attempt ${attempt}/${ATTEMPTS}) ==="
  attempt=$((attempt + 1))
  sleep "$DELAY"
done
