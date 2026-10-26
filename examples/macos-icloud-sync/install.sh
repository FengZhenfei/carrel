#!/usr/bin/env bash
# Install this example's LaunchAgents for the current user, or remove them.
#   ./install.sh               build bin/icloud-download, write both agents to ~/Library/LaunchAgents, load them
#   ./install.sh --push-only   only the push agent, for a source folder that is not in iCloud Drive
#   ./install.sh --uninstall   unload and remove both agents; sync.env, logs and state stay where they are
# The agents run the scripts from this folder, so leave the folder in place once installed.
set -euo pipefail

SYNC_HOME="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AGENTS_DIR="$HOME/Library/LaunchAgents"
DOMAIN="gui/$(id -u)"
PUSH=local.carrel.mirror-push
PREFETCH=local.carrel.icloud-prefetch

mode=install; with_prefetch=1
for arg in "$@"; do
  case "$arg" in
    --uninstall) mode=uninstall ;;
    --push-only) with_prefetch=0 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

unload() {
  launchctl bootout "$DOMAIN/$1" 2>/dev/null || true
  # bootout returns before a busy job finishes handling SIGTERM.
  local attempt
  for ((attempt=0; attempt<30; attempt++)); do
    if ! launchctl print "$DOMAIN/$1" >/dev/null 2>&1; then return 0; fi
    sleep 1
  done
  echo "$1 is still stopping; retry once it has exited" >&2
  return 1
}

if [[ "$mode" == uninstall ]]; then
  for label in "$PUSH" "$PREFETCH"; do
    unload "$label"
    rm -f "$AGENTS_DIR/$label.plist"
  done
  echo "removed $PUSH and $PREFETCH"
  exit 0
fi

die() { echo "$*" >&2; exit 1; }
[[ -f "$SYNC_HOME/sync.env" ]] || die "create $SYNC_HOME/sync.env first: cp sync.env.example sync.env, then edit it"
case "$SYNC_HOME" in
  *[\&\<\>\|\\]*) die "move this folder to a path without & < > | or \\ (the path goes into the agent files)" ;;
  "$HOME/Library/Mobile Documents/"*) die "move this folder out of iCloud Drive: its logs and lock files change every minute" ;;
esac
# shellcheck disable=SC1091
source "$SYNC_HOME/sync.env"
: "${SYNC_SOURCE:?set SYNC_SOURCE in sync.env}"
: "${SYNC_HOST:?set SYNC_HOST in sync.env}"
if ! ssh -o BatchMode=yes -o ConnectTimeout=8 "$SYNC_HOST" true; then
  die "ssh -o BatchMode=yes $SYNC_HOST true failed: set up key-based login first (the agents cannot answer prompts)"
fi

mkdir -p "$SYNC_HOME/bin" "$SYNC_HOME/logs" "$SYNC_HOME/state" "$AGENTS_DIR"
labels=("$PUSH")
if [[ "$with_prefetch" -eq 1 ]]; then
  labels+=("$PREFETCH")
  if command -v swiftc >/dev/null 2>&1; then
    swiftc -O "$SYNC_HOME/icloud_download.swift" -o "$SYNC_HOME/bin/icloud-download"
  else
    echo "note: swiftc not found (xcode-select --install); the prefetch falls back to brctl download"
  fi
else
  unload "$PREFETCH"
  rm -f "$AGENTS_DIR/$PREFETCH.plist"
fi

for label in "${labels[@]}"; do
  target="$AGENTS_DIR/$label.plist"
  sed "s|__SYNC_HOME__|$SYNC_HOME|g" "$SYNC_HOME/launchd/$label.plist" > "$target"
  plutil -lint -s "$target"
  unload "$label"
  launchctl bootstrap "$DOMAIN" "$target"
  echo "loaded $label"
done

if [[ "$with_prefetch" -eq 1 ]]; then shells="/bin/bash and /bin/zsh"; else shells="/bin/bash"; fi
cat <<EOF

If the source folder is in iCloud Drive or another cloud drive, or in Desktop, Documents or Downloads, open
System Settings > Privacy & Security > Full Disk Access and add $shells (Cmd+Shift+G in the file dialog
takes a path). Without it the agents cannot read the folder, and every round stops at the safety checks.
Then follow the log:
  tail -f "$SYNC_HOME/logs/push-mirror.log"
EOF
if [[ "$with_prefetch" -eq 1 && -x "$SYNC_HOME/bin/icloud-download" ]]; then
  cat <<EOF

Optional download helper: $SYNC_HOME/bin/icloud-download
If the prefetch trace reports Code=257 for this helper, add that executable to Full Disk Access too.
The prefetch uses brctl for the rest of a round if the helper fails, and tries the helper again next round.
Download progress: $SYNC_HOME/logs/icloud-prefetch.log
EOF
fi
