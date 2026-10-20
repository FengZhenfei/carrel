#!/bin/zsh
# Download iCloud Drive placeholders under the synced folder, so push-mirror.sh finds real files. launchd runs
# it every minute (local.carrel.icloud-prefetch.plist).
#
# With "Optimize Mac Storage" on, iCloud keeps some files as "dataless" placeholders: they list with their
# real size but hold no data until something asks for them. push-mirror.sh refuses to push while any
# placeholder is left, and while this script holds its lock. For each placeholder this script asks iCloud to
# download it (bin/icloud-download, built from icloud_download.swift; `brctl download` and a one-byte read
# as fallbacks), then waits until the file is materialized. File contents are never changed.
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
rotate_log "$LOG_FILE" "${SYNC_LOG_MAX_KB:-2048}" "${SYNC_LOG_KEEP:-3}"

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
  rm -f "$LOCK_DIR/pid"
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

is_dataless() {
  /bin/ls -ldO "$1" 2>/dev/null | /usr/bin/grep -q "dataless"
}

# Reading one byte makes the file provider fetch the file; the alarm keeps a stuck read from hanging the round.
read_probe() {
  /usr/bin/perl -e '
    my ($timeout, $path) = @ARGV;
    $SIG{ALRM} = sub { exit 124 };
    alarm int($timeout);
    open(my $fh, "<", $path) or exit 2;
    binmode($fh);
    my $n = read($fh, my $buf, 1);
    exit(($n // 0) >= 0 ? 0 : 3);
  ' "$READ_TIMEOUT_SECONDS" "$1" >>"$TRACE_FILE" 2>&1
}

request_download() {
  local f="$1"
  if [[ -x "$DOWNLOAD_HELPER" ]] && "$DOWNLOAD_HELPER" "$f" >>"$TRACE_FILE" 2>&1; then
    return 0
  fi
  if /usr/bin/brctl download "$f" >>"$TRACE_FILE" 2>&1; then
    return 0
  fi
  read_probe "$f"
}

wait_until_materialized() {
  local f="$1"
  local waited=0
  while is_dataless "$f"; do
    touch "$LOCK_DIR"            # keep the lock fresh, so a long download is not mistaken for a stale lock
    if [[ "$waited" -ge "$WAIT_SECONDS" ]]; then
      return 1
    fi
    if [[ "$waited" -eq 0 || $((waited % RETRY_SECONDS)) -eq 0 ]]; then
      request_download "$f" || true
    fi
    sleep "$POLL_SECONDS"
    waited=$((waited + POLL_SECONDS))
  done
  read_probe "$f" || true
  return 0
}

matched=0
requested=0
processed=0
completed=0
failures=0
timeouts=0
: >"$ERR_FILE"
: >"$TRACE_FILE"
: >"$FIND_ERR_FILE"
while IFS= read -r -d '' f; do
  matched=$((matched + 1))
  if [[ "$MAX_FILES" -gt 0 && "$processed" -ge "$MAX_FILES" ]]; then
    break
  fi
  if ! is_dataless "$f"; then        # downloaded in the meantime
    completed=$((completed + 1))
    continue
  fi

  processed=$((processed + 1))
  asked=0
  if request_download "$f"; then
    requested=$((requested + 1))
    asked=1
  fi
  if wait_until_materialized "$f"; then
    completed=$((completed + 1))
  elif is_dataless "$f"; then
    timeouts=$((timeouts + 1))
    print -r -- "timeout	$f" >>"$ERR_FILE"
  elif [[ "$asked" -eq 0 ]]; then
    failures=$((failures + 1))
    print -r -- "failed	$f" >>"$ERR_FILE"
  fi
done < <(
  for scan_root in "${VALID_SCAN_ROOTS[@]}"; do
    /usr/bin/find "$scan_root" -type f -flags +dataless -print0 2>>"$FIND_ERR_FILE" || true
  done
)

if [[ -s "$FIND_ERR_FILE" ]]; then
  echo "[$(ts)] find warning: $(tr '\n' ';' < "$FIND_ERR_FILE")" >> "$LOG_FILE"
fi

echo "[$(ts)] placeholders: matched=$matched requested=$requested completed=$completed timeouts=$timeouts failures=$failures roots=${#VALID_SCAN_ROOTS[@]} (details: $ERR_FILE)" >> "$LOG_FILE"
