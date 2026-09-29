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
FLAG="$STATE_DIR/mirror_changed.flag"
LOCK_DIR="$STATE_DIR/kb_scan.lock.d"
# Note: there used to be a knowledge_io_gate here as well, for sharing I/O between prefetch / mirror-sync /
# scan on the Mac. On the DGX this script is the only participant left, so the gate only added per-minute
# overhead and a failure mode where a stale lock blocked the scan; it has been removed.
MIRROR_LOCK_DIR="$STATE_DIR/mirror_sync.lock.d"
LOCK_STALE_SECONDS="${KB_SCAN_LOCK_STALE_SECONDS:-3600}"
MIRROR_LOCK_STALE_SECONDS="${KB_MIRROR_LOCK_STALE_SECONDS:-3600}"
SCRIPT_NAME="$(basename "$0")"

ts() { date "+%F %T"; }

# GNU stat (Linux) first, BSD stat (macOS) as fallback.
mtime_seconds() {
  stat -c %Y "$1" 2>/dev/null || stat -f %m "$1"
}

# Nanosecond precision, used only to tell whether the flag was re-armed during the scan. At whole-second
# precision, a user clicking "parse now" in the **same second** the scan started would be judged "not
# re-armed" and that click silently swallowed (the comparison below uses -le). %.9Y with the dot removed is a
# 19-digit fixed-width integer that fits bash's 64-bit arithmetic (until 2262); if unavailable, fall back to
# whole seconds padded with nine zeros, which behaves exactly as before.
mtime_ns() {
  local v
  v="$(stat -c %.9Y "$1" 2>/dev/null || true)"
  if [[ "$v" == *.* ]]; then
    printf '%s' "${v/./}"
    return 0
  fi
  v="$(mtime_seconds "$1" 2>/dev/null || true)"
  [[ "$v" =~ ^[0-9]+$ ]] || return 1
  printf '%s000000000' "$v"
}


mkdir -p "$STATE_DIR" "$LOG_DIR"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

require_flag="${KB_SCAN_REQUIRE_FLAG:-1}"
if [[ "$require_flag" =~ ^(1|true|yes|on)$ && ! -f "$FLAG" ]]; then
  # One line per minute would flood the journal (about 1440 lines/day), and this is the normal state anyway.
  [[ "${KB_SCAN_VERBOSE_SKIP:-0}" == "1" ]] && echo "=== $(ts) KB scan skip (no mirror flag) ==="
  exit 0
fi

lock_age_seconds() {
  if [[ ! -d "$LOCK_DIR" ]]; then
    echo 0
    return
  fi
  local now mtime
  now="$(date +%s)"
  mtime="$(mtime_seconds "$LOCK_DIR" 2>/dev/null || echo "$now")"
  echo $((now - mtime))
}

lock_dir_age_seconds() {
  local lock_dir="$1"
  if [[ ! -d "$lock_dir" ]]; then
    echo 0
    return
  fi
  local now mtime
  now="$(date +%s)"
  mtime="$(mtime_seconds "$lock_dir" 2>/dev/null || echo "$now")"
  echo $((now - mtime))
}

pid_lock_is_live() {
  local lock_dir="$1"
  local expected="$2"
  local pid command
  [[ -f "$lock_dir/pid" ]] || return 1
  pid="$(cat "$lock_dir/pid" 2>/dev/null || true)"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" >/dev/null 2>&1 || return 1
  command="$(ps -p "$pid" -o command= 2>/dev/null || true)"
  [[ "$command" == *"$expected"* ]]
}

if [[ -d "$MIRROR_LOCK_DIR" ]]; then
  mirror_age="$(lock_dir_age_seconds "$MIRROR_LOCK_DIR")"
  # Age only. There used to be a pid_lock_is_live(…, "<the Mac push script>") here too: it looked for a
  # process of that name on **this** machine, while the push runs on the Mac, so it never matched; the remote
  # lock has no pid file at all either (the Mac writes an owner). The decision was always made by this age
  # comparison; that half was dead code. Reclaiming a stale lock is the job of gate 5 on the Mac side, which
  # is the one that needs the lock; reclaiming on both sides would only create races.
  if [[ "$mirror_age" -lt "$MIRROR_LOCK_STALE_SECONDS" ]]; then
    touch "$FLAG"
    echo "=== $(ts) KB scan skip (mirror sync running age=${mirror_age}s; flag kept) ==="
    exit 0
  fi
fi

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  age="$(lock_age_seconds)"
  if pid_lock_is_live "$LOCK_DIR" "$SCRIPT_NAME"; then
    echo "=== $(ts) KB scan skip (process still running age=${age}s) ==="
    exit 0
  fi
  if [[ -f "$LOCK_DIR/pid" || "$age" -ge "$LOCK_STALE_SECONDS" ]]; then
    rm -f "$LOCK_DIR/pid"
    rmdir "$LOCK_DIR" 2>/dev/null || {
      echo "=== $(ts) KB scan skip (locked, stale lock could not be removed age=${age}s) ==="
      exit 0
    }
    # Another instance may grab the lock at exactly this moment: if we lose, yield to it rather than letting
    # set -e take the whole script down with a non-zero exit (systemd would record it as failed).
    mkdir "$LOCK_DIR" 2>/dev/null || {
      echo "=== $(ts) KB scan skip (another run took the lock) ==="
      exit 0
    }
  else
    echo "=== $(ts) KB scan skip (locked age=${age}s) ==="
    exit 0
  fi
fi
printf '%s\n' "$$" > "$LOCK_DIR/pid"
SCAN_OUT=""
cleanup() {
  rm -f "$LOCK_DIR/pid"
  rmdir "$LOCK_DIR" >/dev/null 2>&1 || true
  [[ -n "$SCAN_OUT" ]] && rm -f "$SCAN_OUT"
  # This return 0 is load-bearing; do not delete it. On the normal path SCAN_OUT was already removed and
  # cleared above, so the [[ -n "" ]] on the last line is false, && short-circuits and the compound command's
  # status is 1; being the function's last statement, the EXIT trap returns 1 and set -e makes the whole
  # script exit 1. Symptom: the scan printed "KB scan end" and removed the flag correctly, yet systemd
  # recorded the round as failed and the console's "scan mirror" row stayed red. 450 fake failures in 24 hours.
  # Whether cleanup succeeds must not affect the script's exit code, so it is pinned explicitly here.
  return 0
}
trap cleanup EXIT


FLAG_MTIME_AT_START="$(mtime_ns "$FLAG" 2>/dev/null || echo 0)"
echo "=== $(ts) KB scan start ==="
SCAN_OUT="$(mktemp "${TMPDIR:-/tmp}/kb-scan.XXXXXX")"
set +e
"$PY" -m kb_pipeline --env-file "$ENV_FILE" scan --verbose --exit-code-on-recent 2>&1 | tee "$SCAN_OUT"
scan_code=${PIPESTATUS[0]}
set -e

# Kick the worker once whenever jobs were queued (no pipeline code change: jobs=N is read from the scan
# summary). Previously only the console's polling and the worker's own 5-minute timer kicked it, so with the
# console in the background or closed, "the pipeline has started" merely meant the jobs were queued
# (2026-09-08). KB_SCAN_KICK_WORKER=0 turns this off.
# The trailing || true is load-bearing: when the scan fails its output has no summary line, grep returns 1, and
# pipefail + set -e would kill the script on this line -- the failure branch below (restore the flag, say why,
# pass the exit code through) would never be reached.
queued_jobs="$(grep -oE 'scan summary: .*jobs=[0-9]+' "$SCAN_OUT" | grep -oE 'jobs=[0-9]+' | tail -1 | cut -d= -f2 || true)"
rm -f "$SCAN_OUT"; SCAN_OUT=""
if [[ "${KB_SCAN_KICK_WORKER:-1}" =~ ^(1|true|yes|on)$ && "${queued_jobs:-0}" -gt 0 ]] && command -v systemctl >/dev/null 2>&1; then
  if systemctl --user start --no-block carrel-worker.service >/dev/null 2>&1; then
    echo "=== $(ts) KB scan queued ${queued_jobs} job(s); worker kicked ==="
  else
    echo "=== $(ts) KB scan queued ${queued_jobs} job(s); worker kick failed ==="
  fi
fi

if [[ "$scan_code" -eq 0 ]]; then
  # Remove the flag only if it is still the one we started with. If the web console touched it again during
  # the scan (the user clicked "parse now"), a new request is waiting and this one must not be swallowed.
  if [[ -f "$FLAG" ]]; then
    if [[ "$(mtime_ns "$FLAG" 2>/dev/null || echo 0)" -le "$FLAG_MTIME_AT_START" ]]; then
      rm -f "$FLAG"
      echo "=== $(ts) KB scan end ==="
    else
      echo "=== $(ts) KB scan end (flag re-armed during the run; next run will pick it up) ==="
    fi
  else
    echo "=== $(ts) KB scan end ==="
  fi
elif [[ "$scan_code" -eq 75 ]]; then
  touch "$FLAG"
  echo "=== $(ts) KB scan end (recent files pending; flag kept) ==="
else
  touch "$FLAG"
  echo "=== $(ts) KB scan failed (flag restored) ==="
  exit "$scan_code"
fi
