#!/bin/zsh
# Download iCloud Drive placeholders under the synced folder, so push-mirror.sh finds real files. launchd runs
# it every minute (local.carrel.icloud-prefetch.plist).
#
# With "Optimize Mac Storage" on, iCloud keeps some files as "dataless" placeholders: they list with their
# real size but hold no data until something asks for them. push-mirror.sh refuses to push while any
# placeholder is left, and while this script holds its lock. Download requests use a bounded window,
# refilled as files arrive. The optional helper falls back to brctl and a one-byte read. Completion is
# checked independently of request success; file contents are never changed.
set -euo pipefail

SYNC_HOME="${0:A:h}"
ENV_FILE="${SYNC_ENV_FILE:-$SYNC_HOME/sync.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi
: "${SYNC_SOURCE:?set SYNC_SOURCE in $ENV_FILE}"

# The synced folder, plus any other folders you want kept downloaded (colon-separated).
SCAN_ROOTS=("${SYNC_SOURCE%/}")
if [[ -n "${SYNC_PREFETCH_EXTRA_ROOTS:-}" ]]; then
  SCAN_ROOTS+=("${(@s.:.)SYNC_PREFETCH_EXTRA_ROOTS}")
fi
STATE_DIR="$SYNC_HOME/state"
LOG_DIR="$SYNC_HOME/logs"
LOG_FILE="$LOG_DIR/icloud-prefetch.log"
ERR_FILE="$LOG_DIR/icloud-prefetch.last-errors.log"
TRACE_FILE="$LOG_DIR/icloud-prefetch.trace.log"
FIND_ERR_FILE="$LOG_DIR/icloud-prefetch.find.err"
DOWNLOAD_HELPER="$SYNC_HOME/bin/icloud-download"
LOCK_DIR="$STATE_DIR/icloud_prefetch.lock.d"     # push-mirror.sh skips its round while this exists
LOCK_STALE_SECONDS="${SYNC_PREFETCH_LOCK_STALE_SECONDS:-3600}"
SCRIPT_NAME="${0:t}"
WAIT_SECONDS="${SYNC_DOWNLOAD_WAIT_SECONDS:-900}"      # per file; a file still not there is logged as a timeout
POLL_SECONDS="${SYNC_DOWNLOAD_POLL_SECONDS:-5}"
RETRY_SECONDS="${SYNC_DOWNLOAD_RETRY_SECONDS:-60}"     # ask again this often while waiting
READ_TIMEOUT_SECONDS="${SYNC_READ_TIMEOUT_SECONDS:-30}"
BATCH_SIZE="${SYNC_DOWNLOAD_BATCH_SIZE:-64}"
REQUEST_SECONDS="${SYNC_DOWNLOAD_REQUEST_TIMEOUT_SECONDS:-10}"
MAX_FILES="${SYNC_PREFETCH_LIMIT:-0}"                  # placeholders handled per round; 0 = all

ts() { date "+%F %T"; }

mkdir -p "$STATE_DIR" "$LOG_DIR"

# Size-based rotation of our own log: keep N generations, no external tool needed.
rotate_log() {
  local f="$1" max_kb="${2:-2048}" keep="${3:-3}"
  [[ -f "$f" ]] || return 0
  local kb; kb=$(( $(stat -f %z "$f") / 1024 ))
  [[ "$kb" -ge "$max_kb" ]] || return 0
  local i
  for (( i=keep-1; i>=1; i-- )); do
    [[ -f "$f.$i" ]] && mv -f "$f.$i" "$f.$((i+1))"
  done
  mv -f "$f" "$f.1"
  rm -f "$f.$((keep+1))"
}

VALID_SCAN_ROOTS=()
for scan_root in "${SCAN_ROOTS[@]}"; do
  if [[ -d "$scan_root" ]]; then
    VALID_SCAN_ROOTS+=("$scan_root")
  else
    echo "[$(ts)] skip root: directory not found: $scan_root" >> "$LOG_FILE"
  fi
done
if [[ "${#VALID_SCAN_ROOTS[@]}" -eq 0 ]]; then
  echo "[$(ts)] skip: no folder to check" >> "$LOG_FILE"
  exit 0
fi

lock_age_seconds() {
  if [[ ! -d "$LOCK_DIR" ]]; then
    echo 0
    return
  fi
  local now mtime
  now="$(date +%s)"
  mtime="$(stat -f %m "$LOCK_DIR" 2>/dev/null || echo "$now")"
  echo $((now - mtime))
}

# The pid must still belong to this script: pids are reused, after a reboot especially.
lock_pid_is_live() {
  local pid command
  [[ -f "$LOCK_DIR/pid" ]] || return 1
  pid="$(cat "$LOCK_DIR/pid" 2>/dev/null || true)"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" >/dev/null 2>&1 || return 1
  command="$(ps -p "$pid" -o command= 2>/dev/null || true)"
  [[ "$command" == *"$SCRIPT_NAME"* ]]
}

remove_lock_dir() {
  rm -f "$LOCK_DIR/pid" "$LOCK_DIR/candidates"
  rmdir "$LOCK_DIR" 2>/dev/null
}

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  age="$(lock_age_seconds)"
  if lock_pid_is_live; then
    echo "[$(ts)] skip: prefetch process still running age=${age}s" >> "$LOG_FILE"
    exit 0
  fi
  # A pid file whose process is gone is a crashed run; a lock without a pid file may be a run that has not
  # written it yet, so that one is only reclaimed once it is old.
  if [[ -f "$LOCK_DIR/pid" || "$age" -ge "$LOCK_STALE_SECONDS" ]]; then
    remove_lock_dir || {
      echo "[$(ts)] skip: prefetch locked and stale lock could not be removed age=${age}s" >> "$LOG_FILE"
      exit 0
    }
    mkdir "$LOCK_DIR" 2>/dev/null || {
      echo "[$(ts)] skip: another prefetch took the lock" >> "$LOG_FILE"
      exit 0
    }
  else
    echo "[$(ts)] skip: prefetch locked age=${age}s" >> "$LOG_FILE"
    exit 0
  fi
fi
print -r -- "$$" > "$LOCK_DIR/pid"

cleanup() {
  remove_lock_dir >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

log() { print -r -- "[$(ts)] $*" >> "$LOG_FILE"; }
rotate_log "$LOG_FILE" "${SYNC_LOG_MAX_KB:-2048}" "${SYNC_LOG_KEEP:-3}"
rotate_log "$TRACE_FILE" "${SYNC_LOG_MAX_KB:-2048}" "${SYNC_LOG_KEEP:-3}"

# Validate arithmetic settings before starting requests.
for setting in WAIT_SECONDS POLL_SECONDS RETRY_SECONDS READ_TIMEOUT_SECONDS BATCH_SIZE REQUEST_SECONDS MAX_FILES; do
  value="${(P)setting}"
  if [[ "$value" != <-> ]] || (( value < 0 )) || { [[ "$setting" != MAX_FILES ]] && (( value == 0 )); }; then
    log "abort: $setting must be a positive integer (MAX_FILES may be zero)"
    exit 2
  fi
done
if (( BATCH_SIZE > 256 )); then
  log "abort: BATCH_SIZE must be at most 256"
  exit 2
fi
zmodload zsh/datetime

: >"$ERR_FILE"
: >"$TRACE_FILE"
: >"$FIND_ERR_FILE"
: >"$LOCK_DIR/candidates"
for scan_root in "${VALID_SCAN_ROOTS[@]}"; do
  if ! /usr/bin/find "$scan_root" -type f -flags +dataless -print0 >>"$LOCK_DIR/candidates" 2>>"$FIND_ERR_FILE"; then
    log "abort: placeholder scan failed; see $FIND_ERR_FILE"
    exit 1
  fi
done
if [[ -s "$FIND_ERR_FILE" ]]; then
  log "abort: placeholder scan incomplete; see $FIND_ERR_FILE"
  exit 1
fi

# A failed stat must not be counted as a downloaded file. Inspect flags only, not the filename.
file_state() {
  local flags
  if [[ ! -f "$1" ]] || ! flags="$(/usr/bin/stat -f %Sf "$1" 2>>"$TRACE_FILE")"; then
    FILE_STATE=unavailable
  elif [[ "$flags" == *dataless* ]]; then
    FILE_STATE=pending
  else
    FILE_STATE=local
  fi
}

run_timed() {
  /usr/bin/perl -e 'my $seconds = shift; alarm $seconds; exec @ARGV or exit 127;' \
    "$REQUEST_SECONDS" "$@" >>"$TRACE_FILE" 2>&1
}

read_probe() {
  /usr/bin/perl -e '
    my ($timeout, $file) = @ARGV;
    $SIG{ALRM} = sub { exit 124 };
    alarm int($timeout);
    open(my $fh, "<", $file) or exit 2;
    binmode($fh);
    my $n = read($fh, my $buf, 1);
    exit(defined($n) ? 0 : 3);
  ' "$READ_TIMEOUT_SECONDS" "$1" >>"$TRACE_FILE" 2>&1
}

HELPER_ENABLED=0
[[ -x "$DOWNLOAD_HELPER" ]] && HELPER_ENABLED=1
request_download() {
  local f="$1"
  if (( HELPER_ENABLED )); then
    if run_timed "$DOWNLOAD_HELPER" "$f"; then
      return 0
    fi
    HELPER_ENABLED=0
    log "download helper unavailable; using brctl for this round. Check helper permissions if trace reports Code=257: $DOWNLOAD_HELPER"
  fi
  if run_timed /usr/bin/brctl download "$f"; then
    return 0
  fi
  read_probe "$f"
}

matched=0 requested=0 processed=0 completed=0 failures=0 timeouts=0 request_errors=0
queue_open=1
pending_files=() pending_started=() pending_retry=()
started_at=$EPOCHSECONDS
last_progress=$EPOCHSECONDS
log "started: batch=$BATCH_SIZE wait=${WAIT_SECONDS}s poll=${POLL_SECONDS}s"
exec 3<"$LOCK_DIR/candidates"

while (( queue_open || ${#pending_files} )); do
  # Only the request is synchronous. Downloads proceed together in the file provider.
  while (( queue_open && ${#pending_files} < BATCH_SIZE )); do
    if (( MAX_FILES > 0 && processed >= MAX_FILES )); then
      queue_open=0
      break
    fi
    if ! IFS= read -r -d '' f <&3; then
      queue_open=0
      break
    fi
    matched=$((matched + 1))
    file_state "$f"
    if [[ "$FILE_STATE" == local ]]; then
      completed=$((completed + 1))
      continue
    elif [[ "$FILE_STATE" == unavailable ]]; then
      failures=$((failures + 1))
      print -r -- "unavailable\t$f" >>"$ERR_FILE"
      continue
    fi
    processed=$((processed + 1))
    began=$EPOCHSECONDS
    if request_download "$f"; then
      requested=$((requested + 1))
    else
      request_errors=$((request_errors + 1))
      log "request failed; will retry: $f"
    fi
    pending_files+=("$f")
    pending_started+=("$began")
    pending_retry+=("$((EPOCHSECONDS + RETRY_SECONDS))")
  done

  kept_files=() kept_started=() kept_retry=()
  for (( i=1; i<=${#pending_files}; i++ )); do
    f="${pending_files[$i]}"
    file_state "$f"
    if [[ "$FILE_STATE" == local ]]; then
      completed=$((completed + 1))
      continue
    elif [[ "$FILE_STATE" == unavailable ]]; then
      failures=$((failures + 1))
      print -r -- "unavailable\t$f" >>"$ERR_FILE"
      continue
    elif (( EPOCHSECONDS - pending_started[i] >= WAIT_SECONDS )); then
      timeouts=$((timeouts + 1))
      print -r -- "timeout\t$f" >>"$ERR_FILE"
      continue
    fi
    retry_at=${pending_retry[$i]}
    if (( EPOCHSECONDS >= retry_at )); then
      if ! request_download "$f"; then
        request_errors=$((request_errors + 1))
        log "request failed; will retry: $f"
      fi
      retry_at=$((EPOCHSECONDS + RETRY_SECONDS))
    fi
    kept_files+=("$f")
    kept_started+=("${pending_started[$i]}")
    kept_retry+=("$retry_at")
  done
  pending_files=("${kept_files[@]}")
  pending_started=("${kept_started[@]}")
  pending_retry=("${kept_retry[@]}")
  touch "$LOCK_DIR"
  if (( EPOCHSECONDS - last_progress >= 30 )); then
    log "progress: matched=$matched requested=$requested completed=$completed pending=${#pending_files} timeouts=$timeouts failures=$failures request_errors=$request_errors"
    last_progress=$EPOCHSECONDS
  fi
  # Refill free slots immediately: a slow file does not hold up later files.
  if (( queue_open && ${#pending_files} < BATCH_SIZE )); then
    continue
  fi
  if (( ${#pending_files} )); then
    sleep "$POLL_SECONDS"
  fi
done
exec 3<&-
log "finished: matched=$matched requested=$requested completed=$completed timeouts=$timeouts failures=$failures request_errors=$request_errors elapsed=$((EPOCHSECONDS - started_at))s"
(( timeouts == 0 && failures == 0 ))
