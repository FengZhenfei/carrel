#!/usr/bin/env bash
# Push a folder on this Mac (an iCloud Drive folder, in the author's setup) to the Carrel mirror on the
# server, in one rsync hop over SSH. launchd runs it every minute (local.carrel.mirror-push.plist).
#
# The mirror is Carrel's source of truth: whatever this script puts there gets indexed, and whatever it
# deletes there (rsync --delete) drops out of the index on the next scan (Carrel keeps the data for an undo
# window). So before pushing, every gate below asks one question that only this Mac can answer: "is the tree
# I just read trustworthy?"
#   1. single flight     a lock dir holding our pid; a lock whose process is gone (power cut, kill -9) is
#                        reclaimed at once, one without a pid file once it is old
#   2. launchd only      the Full Disk Access grant that lets a script read iCloud Drive belongs to the shell
#                        launchd runs; from a terminal or an IDE the same folder may read as empty, and
#                        --delete would faithfully replicate "empty" to the server
#   3. readable          the source must list and hold at least one file
#   3b. no placeholders  iCloud, like other cloud drive clients, keeps files it has not downloaded as "dataless"
#                        placeholders, which rsync reads as empty; wait until they are downloaded
#                        (icloud-prefetch.sh does that for iCloud). If the check itself fails, skip the round
#                        (fail closed): it retries a minute later anyway
#   4. reachable         SSH is tested on its own first, so an unreachable server is a logged skip
#   5. server lock       runtime/state/mirror_sync.lock.d on the server is held during the transfer; Carrel's
#                        scan does not walk the mirror while it exists. A lock left by a push that died is
#                        reclaimed once older than SYNC_REMOTE_LOCK_STALE_SECONDS (keep it equal to
#                        KB_MIRROR_LOCK_STALE_SECONDS on the server, which only ignores a stale lock)
# A file in transit is a hidden temp file (.name.XXXXXX), finished ones wait in a hidden .rsync-partial/ dir
# (--delay-updates) and are all renamed into place at the end. The scan skips dot-files and dot-dirs, so even a
# scan that ignores the lock never sees half a batch.
#
# After a push that changed something, runtime/state/mirror_changed.flag is touched on the server. With
# KB_SCAN_REQUIRE_FLAG=1 the scan only runs after such a push; with 0 (the template default) it scans every
# minute and the flag is harmless.
#
# macOS ships openrsync (protocol 29) as /usr/bin/rsync; it accepts every flag below, and the receiving side
# is the server's own rsync. A failed transfer may still have changed the mirror (deletions run before the
# copy), so the flag is set after a failure too (a needless rescan is cheap; a missed one leaves the index
# behind the mirror).
#
# Flags:
#   --dry-run            show what rsync would do; change nothing on the server
#   --no-flag            do not touch the server's mirror_changed.flag
#   --allow-no-launchd   skip gate 2. Only for a shell that really holds Full Disk Access (or a source folder
#                        that needs none); from any other shell gate 3 aborts instead
set -euo pipefail
export LC_ALL=C LANG=C

SYNC_HOME="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SYNC_ENV_FILE:-$SYNC_HOME/sync.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi
: "${SYNC_SOURCE:?set SYNC_SOURCE in $ENV_FILE (the local folder to push; one subfolder per knowledge base)}"
: "${SYNC_HOST:?set SYNC_HOST in $ENV_FILE (the server, as you would pass it to ssh)}"
SRC="${SYNC_SOURCE%/}"
REMOTE_DIR="${SYNC_REMOTE_DIR:-carrel}"            # the Carrel checkout; a relative path is under the remote home
REMOTE_MIRROR="${SYNC_REMOTE_MIRROR:-$REMOTE_DIR/runtime/mirror}"
REMOTE_STATE="$REMOTE_DIR/runtime/state"
REMOTE_FLAG="$REMOTE_STATE/mirror_changed.flag"
REMOTE_LOCK="$REMOTE_STATE/mirror_sync.lock.d"
STATE_DIR="$SYNC_HOME/state"
LOG_DIR="$SYNC_HOME/logs"
LOCK_DIR="$STATE_DIR/mirror_push.lock.d"
LOG_FILE="$LOG_DIR/push-mirror.log"

# BatchMode: a key or host-key problem fails instead of waiting for a prompt nobody sees. ServerAlive: a dead
# peer tears the session down within about a minute instead of wedging the round.
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)

DRY_RUN=0; SET_FLAG=1; ALLOW_NO_LAUNCHD=0
for arg in "$@"; do
  case "$arg" in
    --dry-run)          DRY_RUN=1 ;;
    --no-flag)          SET_FLAG=0 ;;
    --allow-no-launchd) ALLOW_NO_LAUNCHD=1 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

ts() { date "+%F %T"; }
# The log file only; a run from a terminal also prints each line (under launchd, stdout goes to the agent's
# capture file, which then only collects unexpected errors).
log() {
  local line="[$(ts)] $*"
  echo "$line" >> "$LOG_FILE"
  if [[ -t 1 ]]; then echo "$line"; fi
}
mkdir -p "$STATE_DIR" "$LOG_DIR"

# Size-based rotation of our own log: keep N generations, no external tool needed.
rotate_log() {
  local f="$1" max_kb="${2:-2048}" keep="${3:-3}"
  [[ -f "$f" ]] || return 0
  local kb; kb=$(( $(stat -f %z "$f" 2>/dev/null || stat -c %s "$f") / 1024 ))
  [[ "$kb" -ge "$max_kb" ]] || return 0
  local i
  for (( i=keep-1; i>=1; i-- )); do
    [[ -f "$f.$i" ]] && mv -f "$f.$i" "$f.$((i+1))"
  done
  mv -f "$f" "$f.1"
  rm -f "$f.$((keep+1))"
}
rotate_log "$LOG_FILE" "${SYNC_LOG_MAX_KB:-2048}" "${SYNC_LOG_KEEP:-3}"

is_meaningful_change() {
  local line="$1"
  [[ -z "$line" ]] && return 1
  [[ "$line" =~ ^(\.d|cd) ]] && return 1
  [[ "$line" =~ (^|[[:space:]]|/)\.DS_Store$ ]] && return 1
  [[ "$line" =~ (^|[[:space:]]|/)\._[^/]*$ ]] && return 1
  return 0
}

# ── gate 1: single flight, crash-safe ──
lock_pid_alive() {
  local pid command
  pid="$(cat "$LOCK_DIR/pid" 2>/dev/null || true)"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  # pids are reused, after a reboot especially: the live process must be this script, or every round would
  # defer to a stranger and the sync would stop
  command="$(ps -p "$pid" -o command= 2>/dev/null || true)"
  [[ "$command" == *"$(basename "$0")"* ]]
}
lock_age() { echo $(( $(date +%s) - $(stat -f %m "$LOCK_DIR" 2>/dev/null || stat -c %Y "$LOCK_DIR" 2>/dev/null || date +%s) )); }
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  if lock_pid_alive; then
    log "skip: another push is running (pid $(cat "$LOCK_DIR/pid"))"
    exit 0
  fi
  # A pid file whose process is gone (or is no longer this script) is a crashed run; a lock without a pid file
  # may be a run that has not written it yet, so that one is only reclaimed once it is old.
  if [[ -f "$LOCK_DIR/pid" ]]; then
    log "reclaim: lock of pid $(cat "$LOCK_DIR/pid" 2>/dev/null || echo '?'), which is no longer a push (age=$(lock_age)s)"
  elif [[ "$(lock_age)" -ge "${SYNC_PUSH_LOCK_STALE_SECONDS:-1800}" ]]; then
    log "reclaim: stale lock without a pid (age=$(lock_age)s)"
  else
    log "skip: lock without a pid yet (age=$(lock_age)s); another push may be starting"
    exit 0
  fi
  rm -rf "$LOCK_DIR"
  mkdir "$LOCK_DIR" 2>/dev/null || { log "skip: another push took the lock"; exit 0; }
fi
echo $$ > "$LOCK_DIR/pid"
# The server lock is released here as well, so any abnormal exit short of SIGKILL (SIGTERM, set -e, logout)
# does not leave it behind; after a SIGKILL, gate 5 reclaims it once it is stale.
REMOTE_LOCK_HELD=0
release_remote_lock() {
  [[ "$REMOTE_LOCK_HELD" -eq 1 ]] || return 0
  REMOTE_LOCK_HELD=0
  ssh "${SSH_OPTS[@]}" "$SYNC_HOST" "rm -f '$REMOTE_LOCK/owner'; rmdir '$REMOTE_LOCK' 2>/dev/null" || true
}
cleanup() {
  release_remote_lock
  rm -rf "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup EXIT

# If the flag cannot be set, remember it locally and set it next round: by then rsync has nothing left to
# transfer, and without this the batch would wait for the next real change to be scanned.
FLAG_PENDING="$STATE_DIR/mirror_flag_pending"
set_remote_flag() {
  if ssh "${SSH_OPTS[@]}" "$SYNC_HOST" "touch '$REMOTE_FLAG'"; then
    rm -f "$FLAG_PENDING"
    return 0
  fi
  touch "$FLAG_PENDING"
  return 1
}

# ── gate 2: must be running under launchd (that is where the Full Disk Access grant applies) ──
if [[ "$ALLOW_NO_LAUNCHD" -ne 1 && "$(ps -o ppid= -p $$ | tr -d ' ')" != "1" ]]; then
  log "abort: not running under launchd (ppid=$(ps -o ppid= -p $$ | tr -d ' ')); the source may read as empty from here. Use --allow-no-launchd only if this shell has Full Disk Access."
  exit 1
fi

# ── gate 3: the source must be readable and non-empty ──
if ! ls "$SRC" >/dev/null 2>&1; then
  log "abort: source unreadable (privacy settings?): $SRC"
  exit 1
fi
local_count="$(find "$SRC" -type f ! -name '.DS_Store' ! -name '._*' 2>/dev/null | wc -l | tr -d ' ')"
if [[ "$local_count" -eq 0 ]]; then
  log "abort: source lists empty; refusing to sync (it would empty the mirror): $SRC"
  exit 1
fi

# ── gate 3b: no dataless placeholders (for iCloud, icloud-prefetch.sh downloads them) ──
# Fail closed: if find itself fails (a changed flag name after a macOS update, a revoked permission), skip
# the round with a log line instead of pushing placeholders as empty files.
dl_err="$(mktemp)"
set +e
dataless="$(/usr/bin/find "$SRC" -type f -flags +dataless -print -quit 2>"$dl_err")"
dl_rc=$?
set -e
if [[ "$dl_rc" -ne 0 || -s "$dl_err" ]]; then
  log "skip: cannot verify placeholders (find rc=$dl_rc): $(tr '\n' ';' < "$dl_err" | cut -c1-200)"
  rm -f "$dl_err"
  exit 0
fi
rm -f "$dl_err"
if [[ -n "$dataless" ]]; then
  log "skip: placeholder not downloaded yet: ${dataless#"$SRC"/}"
  exit 0
fi
if [[ -d "$STATE_DIR/icloud_prefetch.lock.d" ]]; then
  log "skip: icloud-prefetch mid-flight; retry next round"
  exit 0
fi

# ── gate 4: SSH reachability, checked on its own ──
# Judged by the exit code, not by stderr (a banner or a locale warning on login is not a failure); ssh's last
# stderr line is kept because "unreachable" hides different causes (host down, changed host key, revoked key).
set +e
ssh_err="$(ssh "${SSH_OPTS[@]}" "$SYNC_HOST" true 2>&1 >/dev/null)"
ssh_rc=$?
set -e
if [[ "$ssh_rc" -ne 0 ]]; then
  log "skip: $SYNC_HOST unreachable (ssh rc=$ssh_rc); nothing changed on the server. ssh: ${ssh_err##*$'\n'}"
  exit 0
fi

ssh "${SSH_OPTS[@]}" "$SYNC_HOST" "mkdir -p '$REMOTE_MIRROR' '$REMOTE_STATE'"

if [[ -f "$FLAG_PENDING" && "$SET_FLAG" -eq 1 && "$DRY_RUN" -ne 1 ]]; then
  if set_remote_flag; then
    log "flag: server flag set for an earlier push that could not deliver it"
  else
    log "flag: server flag still cannot be set; will retry next round"
  fi
fi

# ── gate 5: take the server's mirror lock, reclaiming it if it is orphaned ──
# Gate 1 already guarantees no live push on this Mac, so a lock we cannot take belongs either to a manual
# sync on the server or to a push that died; only the second grows old.
REMOTE_LOCK_STALE_SECONDS="${SYNC_REMOTE_LOCK_STALE_SECONDS:-3600}"
take_remote_lock() {
  # mkdir succeeding is what takes the lock; the owner file is only for humans
  ssh "${SSH_OPTS[@]}" "$SYNC_HOST" \
    "mkdir '$REMOTE_LOCK' 2>/dev/null || exit 1; echo 'holder=mac-mirror-push' > '$REMOTE_LOCK/owner' || true"
}
remote_lock_age() {
  local age
  age="$(ssh "${SSH_OPTS[@]}" "$SYNC_HOST" "echo \$(( \$(date +%s) - \$(stat -c %Y '$REMOTE_LOCK' 2>/dev/null || stat -f %m '$REMOTE_LOCK' 2>/dev/null || date +%s) ))" 2>/dev/null || echo "")"
  [[ "$age" =~ ^[0-9]+$ ]] || age=0
  printf '%s' "$age"
}
if [[ "$DRY_RUN" -ne 1 ]]; then
  if ! take_remote_lock; then
    age="$(remote_lock_age)"
    if [[ "$age" -lt "$REMOTE_LOCK_STALE_SECONDS" ]]; then
      log "skip: server mirror lock held (age=${age}s); another push or a manual sync is in flight"
      exit 0
    fi
    log "reclaim: stale server mirror lock (age=${age}s); a previous push died before releasing it"
    ssh "${SSH_OPTS[@]}" "$SYNC_HOST" "rm -f '$REMOTE_LOCK/owner'; rmdir '$REMOTE_LOCK' 2>/dev/null" || true
    if ! take_remote_lock; then
      log "abort: could not take the server mirror lock even after reclaiming it"
      exit 1
    fi
  fi
  REMOTE_LOCK_HELD=1
fi

# Finder metadata, iCloud bookkeeping, Office lock files and sync-conflict copies (English and Chinese
# system languages) never go to the server.
rsync_flags=(-a --delete --itemize-changes
  --partial-dir=.rsync-partial --delay-updates
  -e "ssh ${SSH_OPTS[*]}"
  --exclude='.DS_Store' --exclude='._*' --exclude='.TemporaryItems/'
  --exclude='.Trashes/' --exclude='.fseventsd/' --exclude='.Spotlight-V100/'
  --exclude='~$*' --exclude='*.icloud' --exclude='.rsync-partial/'
  --exclude='*冲突副本*' --exclude='*conflicted copy*')
[[ "$DRY_RUN" -eq 1 ]] && rsync_flags+=(--dry-run)

changes_file="$(mktemp)"
started="$(date +%s)"
set +e
rsync "${rsync_flags[@]}" "$SRC/" "$SYNC_HOST:$REMOTE_MIRROR/" >"$changes_file" 2>>"$LOG_FILE"
status=$?
set -e
elapsed="$(( $(date +%s) - started ))"
release_remote_lock
if [[ "$status" -ne 0 ]]; then
  if [[ "$SET_FLAG" -eq 1 && "$DRY_RUN" -ne 1 ]]; then
    if set_remote_flag; then
      log "failed: rsync status=$status after ${elapsed}s; the mirror may be half-updated, flag set so the server rescans"
    else
      log "failed: rsync status=$status after ${elapsed}s; the mirror may be half-updated, server flag NOT set (ssh failed), will retry next round"
    fi
  else
    log "failed: rsync status=$status after ${elapsed}s; server flag untouched"
  fi
  rm -f "$changes_file"
  exit "$status"
fi

meaningful=0
while IFS= read -r line; do
  if is_meaningful_change "$line"; then meaningful=$((meaningful + 1)); fi
done < "$changes_file"
rm -f "$changes_file"

if [[ "$DRY_RUN" -eq 1 ]]; then
  log "dry-run: local=$local_count files, would change $meaningful entries, ${elapsed}s"
  exit 0
fi

# Residue of an interrupted round: --delay-updates parks finished files in .rsync-partial/ until the batch
# ends, and an interrupted batch leaves them there. The next round normally uses them up, but one whose source
# file was deleted in between would stay forever (the exclude also shields it from --delete). The scan skips
# dot-dirs, so this is only about disk space; anything parked for over two hours is residue by now. (A stray
# temp file .name.XXXXXX outside that dir needs no sweep: --delete removes it on the next round.)
ssh "${SSH_OPTS[@]}" "$SYNC_HOST" "find '$REMOTE_MIRROR' -path '*/.rsync-partial/*' -type f -cmin +120 -delete 2>/dev/null; find '$REMOTE_MIRROR' -type d -name '.rsync-partial' -empty -delete 2>/dev/null" || true

if [[ "$meaningful" -gt 0 ]]; then
  # du only on rounds that changed something, and never fatal: one unreadable subfolder makes du exit 1, and
  # under set -e that would end the script after the push but before the flag is set
  kbytes="$(du -sk "$SRC" 2>/dev/null | cut -f1 || true)"
  kbytes="${kbytes:-?}"
  if [[ "$SET_FLAG" -ne 1 ]]; then
    log "pushed: local=$local_count changed=$meaningful size=${kbytes}KB ${elapsed}s; flag skipped (--no-flag)"
  elif set_remote_flag; then
    log "pushed: local=$local_count changed=$meaningful size=${kbytes}KB ${elapsed}s; server flag set"
  else
    log "pushed: local=$local_count changed=$meaningful size=${kbytes}KB ${elapsed}s; server flag NOT set (ssh failed), will retry next round"
  fi
else
  log "no-op: nothing meaningful changed (${elapsed}s)"
fi
