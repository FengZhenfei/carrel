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
STATE_DIR="$BASE_DIR/runtime/state"
LOG_DIR="$BASE_DIR/logs"
LOCK_DIR="$STATE_DIR/kb_worker.lock.d"
RUNNING_FLAG="$STATE_DIR/kb_worker_running.flag"
LOCK_STALE_SECONDS="${KB_WORKER_LOCK_STALE_SECONDS:-86400}"

ts() { date "+%F %T"; }

# GNU stat (Linux) first, BSD stat (macOS) as fallback.
mtime_seconds() {
  stat -c %Y "$1" 2>/dev/null || stat -f %m "$1"
}


mkdir -p "$STATE_DIR" "$LOG_DIR"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

QDRANT_HEALTH_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
QDRANT_HEALTH_URL="${QDRANT_HEALTH_URL%/}/healthz"
if ! /usr/bin/curl -fsS --max-time "${KB_QDRANT_HEALTH_TIMEOUT_SECONDS:-5}" "$QDRANT_HEALTH_URL" >/dev/null 2>&1; then
  echo "=== $(ts) KB worker deferred (Qdrant unavailable; queue preserved) ==="
  exit 0
fi

lock_age_seconds() {
  if [[ ! -d "$LOCK_DIR" && ! -f "$RUNNING_FLAG" ]]; then
    echo 0
    return
  fi
  local now mtime
  now="$(date +%s)"
  if [[ -f "$RUNNING_FLAG" ]]; then
    mtime="$(mtime_seconds "$RUNNING_FLAG" 2>/dev/null || echo "$now")"
  else
    mtime="$(mtime_seconds "$LOCK_DIR" 2>/dev/null || echo "$now")"
  fi
  echo $((now - mtime))
}

has_running_worker() {
  pgrep -f "$PY -m kb_pipeline.*worker --once" >/dev/null 2>&1
}

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  age="$(lock_age_seconds)"
  # Liveness beats age: a lock left behind by a killed worker (SIGKILL, power
  # loss) used to block the queue for LOCK_STALE_SECONDS while systemd kept
  # reporting success. If no worker process is alive, the lock is stale now.
  if ! has_running_worker; then
    echo "=== $(ts) KB worker reclaiming stale lock (no worker process, age=${age}s) ==="
    rm -f "$RUNNING_FLAG"
    rmdir "$LOCK_DIR" 2>/dev/null || {
      echo "=== $(ts) KB worker skip (stale lock could not be removed age=${age}s) ==="
      exit 0
    }
    mkdir "$LOCK_DIR"
  else
    echo "=== $(ts) KB worker skip (locked age=${age}s) ==="
    exit 0
  fi
fi

cleanup() {
  rm -f "$RUNNING_FLAG"
  rmdir "$LOCK_DIR" >/dev/null 2>&1 || true
}
trap cleanup EXIT
touch "$RUNNING_FLAG"

echo "=== $(ts) KB worker loop start parse_enabled=${KB_PARSE_ENABLED:-0} ==="
processed=0
crashes=0
lock_waits=0
MAX_CRASHES="${KB_WORKER_MAX_CRASHES:-3}"
# When another process holds the write lock on the state database (an ops script with a long transaction, a
# whole-database prune), python exits with "database is locked": that is not a crash, it is not counted as
# one, and we wait a while and retry; after several consecutive waits the queue is left to the next timer run.
# 2026-09-06 00:00 a backfill script's long transaction made the worker "crash" three times and give up the
# whole round (health check R1 / R2).
MAX_LOCK_WAITS="${KB_WORKER_MAX_LOCK_WAITS:-4}"
LOCK_WAIT_SECONDS="${KB_WORKER_LOCK_WAIT_SECONDS:-30}"
MAX_JOBS_PER_RUN="${KB_WORKER_MAX_JOBS_PER_RUN:-200}"
# How many jobs each interpreter process runs in a row (saves startup cost); a crash loses at most this batch
JOBS_PER_PROCESS="${KB_WORKER_JOBS_PER_PROCESS:-10}"
# Time budget for the whole round. Once it is used up, stop cleanly **between two files** and leave the queue
# to the next timer run.
#
# Why it is needed: TimeoutStartSec on the unit is a hard kill from outside, and where it lands is pure luck.
# On 2026-08-25 it landed in the middle of one file's MinerU parse; the parse cache is only written once the
# whole document is done, so the kill left nothing but an empty directory and those 12 minutes of compute
# were wasted. Whenever a round's total time exceeds the hard limit, some file is inevitably cut down this
# way, and the bigger the KB, the more inevitably.
#
# The budget is **shared by the whole round**, not handed out per python call, so what is passed below is
# the remainder.
START_TS="$(date +%s)"
MAX_SECONDS="${KB_WORKER_MAX_SECONDS:-5400}"
while true; do
  touch "$RUNNING_FLAG"
  if [[ "$MAX_SECONDS" -gt 0 ]]; then
    remaining=$(( MAX_SECONDS - ( $(date +%s) - START_TS ) ))
    if [[ "$remaining" -le 0 ]]; then
      echo "=== $(ts) KB worker time budget exhausted processed=${processed} (timer picks up the rest) ==="
      break
    fi
  else
    remaining=0
  fi
  set +e
  output="$("$PY" -m kb_pipeline --env-file "$ENV_FILE" worker --once \
      --max-jobs "$JOBS_PER_PROCESS" --max-seconds "$remaining" 2>&1)"
  status=$?
  set -e
  printf '%s\n' "$output"
  if [[ "$status" -ne 0 ]]; then
    if printf '%s\n' "$output" | grep -q 'database is locked'; then
      lock_waits=$((lock_waits + 1))
      echo "=== $(ts) KB worker state db locked (wait ${lock_waits}/${MAX_LOCK_WAITS}, ${LOCK_WAIT_SECONDS}s) processed=${processed} ==="
      if [[ "$lock_waits" -ge "$MAX_LOCK_WAITS" ]]; then
        echo "=== $(ts) KB worker state db still locked; leaving the queue to the next timer run ==="
        break
      fi
      sleep "$LOCK_WAIT_SECONDS"
      continue
    fi
    # A crashed interpreter (OOM kill, segfault) must not take the whole run
    # down: the next iteration lets the recovery pass count the attempt and
    # back the job off, so the rest of the queue keeps moving.
    crashes=$((crashes + 1))
    echo "=== $(ts) KB worker iteration failed status=${status} crashes=${crashes} processed=${processed} ==="
    if [[ "$crashes" -ge "$MAX_CRASHES" ]]; then
      echo "=== $(ts) KB worker giving up after ${crashes} crashed iterations ==="
      exit "$status"
    fi
    continue
  fi
  if printf '%s\n' "$output" | grep -qx 'no-job'; then
    echo "=== $(ts) KB worker loop idle processed=${processed} ==="
    break
  fi
  if printf '%s\n' "$output" | grep -q '^time-budget-reached '; then
    echo "=== $(ts) KB worker stopped on time budget processed=${processed} (timer picks up the rest) ==="
    break
  fi
  processed=$((processed + 1))
  lock_waits=0
  if [[ "$processed" -ge "$MAX_JOBS_PER_RUN" ]]; then
    echo "=== $(ts) KB worker reached per-run cap processed=${processed} (timer picks up the rest) ==="
    break
  fi
done
echo "=== $(ts) KB worker loop end processed=${processed} ==="
