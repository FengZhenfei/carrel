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

# One round every 2 hours (since 2026-09-06; 30 minutes before that): the CLI does a full rebuild when the
# conditions are met, otherwise merges new / changed documents incrementally. On a yield (the CLI exits 75:
# this KB is parsing / the graph build lock is taken) there is no sleeping retry any more; the next round
# comes by itself. How many consecutive yields turn the unit red is defined in lib/kb-maint-defer.sh, relaxed
# to 12 rounds (one day) for the 2-hour cadence.
export KB_MAINT_DEFER_LIMIT="${KB_MAINT_DEFER_LIMIT:-12}"
# Wait for Qdrant first (up to 5 minutes); if it never comes up, exit quietly under the same deferral count
# instead of the unit's ExecStartPre declaring failure outright
maint_wait_qdrant_or_defer "graph-rebuild" "graph rebuild check: Qdrant not ready"
ATTEMPTS="${KB_GRAPH_BUSY_ATTEMPTS:-1}"
DELAY="${KB_GRAPH_BUSY_RETRY_SECONDS:-1800}"
attempt=1
while :; do
  set +e
  "$PY" -m kb_pipeline --env-file "$ENV_FILE" graph check-rebuild --execute "$@"
  code=$?
  set -e
  if [[ "$code" -eq 0 ]]; then
    maint_defer_clear "graph-rebuild"
    exit 0
  fi
  if [[ "$code" -ne 75 ]]; then
    exit "$code"
  fi
  if [[ "$attempt" -ge "$ATTEMPTS" ]]; then
    maint_defer_give_up "graph-rebuild" \
      "graph rebuild check yielded ${attempt} times in a row"
  fi
  echo "=== $(date '+%F %T') graph rebuild deferred (busy), retrying in ${DELAY}s (attempt ${attempt}/${ATTEMPTS}) ==="
  attempt=$((attempt + 1))
  sleep "$DELAY"
done
