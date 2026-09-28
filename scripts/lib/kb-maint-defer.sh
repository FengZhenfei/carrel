# The yield policy for maintenance jobs that run into service_busy. Shared by kb-cleanup.sh and
# kb-graph-rebuild-check.sh: there can be only one policy, and two separate copies would inevitably diverge.
#
# All three approaches have been tried:
#   plain exit 0   -- systemd records success and Persistent does not catch up either; during sustained
#                     ingestion whole rounds of GC/rotation were silently skipped for days, and nobody could tell.
#   plain exit 75  -- the unit shows failed for long stretches. Yet "skip this round to avoid the ingestion
#                     peak" is normal behaviour; painting normal behaviour as a fault means real faults stop
#                     being believed.
#   run counter    -- a yield exits quietly (0) but records "how many rounds in a row" on disk; only when the
#                     run exceeds the limit does it exit 75. An occasional yield is silent, never getting a
#                     turn shows red.
#
# The count is per round, not per time: three yields in a row are 3 days for a daily job and 3 weeks for a
# weekly one, naturally proportional to each job's cadence, so no per-unit window needs tuning. One success
# resets it.

: "${BASE_DIR:?kb-maint-defer.sh: the caller must set BASE_DIR}"

_maint_defer_file() {   # $1 = job name
  local dir="${KB_MAINT_STATE_DIR:-$BASE_DIR/runtime/state/maintenance}"
  mkdir -p "$dir"
  printf '%s/%s.defers\n' "$dir" "$1"
}

# The job really succeeded: clear the consecutive-yield count
maint_defer_clear() {
  rm -f "$(_maint_defer_file "$1")"
}

# This round yielded for good. Whether to exit quietly or turn the unit red depends on how many rounds in a
# row this makes.
# $1 = job name (counter file name), $2 = human-readable description (goes into the log)
maint_defer_give_up() {
  local task="$1" what="$2" file count limit
  file="$(_maint_defer_file "$task")"
  count=$(cat "$file" 2>/dev/null || echo 0)
  [[ "$count" =~ ^[0-9]+$ ]] || count=0
  count=$((count + 1))
  printf '%s\n' "$count" > "$file"
  limit="${KB_MAINT_DEFER_LIMIT:-3}"
  if [[ "$count" -ge "$limit" ]]; then
    echo "=== $(date '+%F %T') ${what}: yielded ${count} rounds in a row (limit ${limit}); exiting 75 so the unit shows it ===" >&2
    exit 75
  fi
  echo "=== $(date '+%F %T') ${what}: yielding this round (${count}/${limit} in a row); the next round retries ==="
  exit 0
}

# Maintenance jobs that depend on Qdrant wait for it before starting (on a catch-up run after boot the
# container may not be up yet): at most KB_QDRANT_WAIT_ROUNDS × KB_QDRANT_WAIT_SECONDS (default 30 × 10 s =
# 5 minutes). If it is still unavailable, treat it as a yield: exit quietly and count, and only turn the unit
# red once the run exceeds the limit. This step used to be an exit 75 straight from the unit's ExecStartPre:
# while the services were deliberately stopped, every round painted the unit failed, although "the services
# are down" is a normal yield just like "hit the ingestion peak", not a fault (2026-09-06).
# $1 = job name (counter file name), $2 = human-readable description (goes into the log)
maint_wait_qdrant_or_defer() {
  local task="$1" what="$2" url rounds delay i
  url="${QDRANT_URL:-}"
  if [[ -z "$url" && -n "${ENV_FILE:-}" && -f "$ENV_FILE" ]]; then
    # not loaded into the environment by systemd (interactive run, the qdrant-gc unit): read it from the env file
    url="$(awk -F= '$1 == "QDRANT_URL" { v = substr($0, length($1) + 2); gsub(/^[[:space:]]+|[[:space:]]+$/, "", v); gsub(/^"|"$/, "", v); print v }' "$ENV_FILE" | tail -n 1)"
  fi
  url="${url:-http://127.0.0.1:6333}"
  rounds="${KB_QDRANT_WAIT_ROUNDS:-30}"
  delay="${KB_QDRANT_WAIT_SECONDS:-10}"
  for ((i = 1; i <= rounds; i++)); do
    if curl -fsS --max-time 3 "${url%/}/healthz" >/dev/null 2>&1; then
      return 0
    fi
    if [[ "$i" -lt "$rounds" ]]; then
      sleep "$delay"
    fi
  done
  echo "=== $(date '+%F %T') Qdrant not ready (${url%/}/healthz): ${what} ===" >&2
  maint_defer_give_up "$task" "$what"
}
