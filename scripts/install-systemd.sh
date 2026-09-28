#!/usr/bin/env bash
# Install the user-scope systemd units from deployment/systemd/.
#
# The installed units are rendered copies, not symlinks: `__CARREL_HOME__` in
# the templates becomes this checkout's path, so several checkouts can coexist
# and moving the repo only needs a re-run. Idempotent: only files whose rendered
# content changed are written, `daemon-reload` runs only when something
# changed, and a running unit is never restarted (a worker mid-file would lose
# its parse progress).
#
#   ./scripts/install-systemd.sh              install + reload, keep enable state
#   ./scripts/install-systemd.sh --check      compare only; exit 1 on drift
#   ./scripts/install-systemd.sh --enable     also `enable --now` the timers and services
#   ./scripts/install-systemd.sh --uninstall  disable, stop and remove the units
#   --force                                    also replace / remove units that belong to another checkout
#
# CARREL_HOME overrides the repo path baked into the units (default: the
# checkout this script lives in).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd -- "$SCRIPT_DIR/../deployment/systemd" && pwd)"
DST_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
CARREL_HOME="${CARREL_HOME:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"

# Timers and long-running services that --enable switches on. The pipeline
# only ingests while KB_PARSE_ENABLED=1 in config/knowledge-base.env, so
# enabling the timers is safe on a fresh install.
ENABLE_UNITS=(
  carrel-scan.timer
  carrel-worker.timer
  carrel-graph-rebuild.timer
  carrel-qdrant-gc.timer
  carrel-cache-weekly.timer
  carrel-logs-monthly.timer
  carrel-web.service
  carrel-search.service
)

MODE="install"
FORCE=0
for arg in "$@"; do
  case "$arg" in
    --check)     MODE="check" ;;
    --enable)    MODE="enable" ;;
    --uninstall) MODE="uninstall" ;;
    --force)     FORCE=1 ;;
    *) echo "usage: $0 [--check|--enable|--uninstall] [--force]" >&2; exit 2 ;;
  esac
done

# A unit "belongs" to a checkout when its rendered paths point into it. Two
# checkouts on one machine must not silently take over each other's units.
# Every template renders the checkout path (ExecStart, or the marker comment on the first line).
owned_by_us() { grep -qF -- "$CARREL_HOME/" "$1"; }

unit_files() {
  local f
  for f in "$SRC_DIR"/*.service "$SRC_DIR"/*.timer; do
    [[ -e "$f" ]] && echo "$f"
  done
}

render() {  # render <template> -> stdout
  sed "s|__CARREL_HOME__|$CARREL_HOME|g" "$1"
}

if [[ "$MODE" == "uninstall" ]]; then
  removed=0
  for src in $(unit_files); do
    name="$(basename "$src")"
    if [[ -f "$DST_DIR/$name" ]]; then
      if ! owned_by_us "$DST_DIR/$name" && [[ "$FORCE" -ne 1 ]]; then
        echo "kept: $name belongs to another checkout ($(grep -m1 -oE 'WorkingDirectory=.*' "$DST_DIR/$name")); use --force to remove it" >&2
        continue
      fi
      systemctl --user disable --now "$name" >/dev/null 2>&1 || true
      rm -f "$DST_DIR/$name"
      echo "removed: $name"
      removed=1
    fi
  done
  if [[ "$removed" -eq 1 ]]; then
    systemctl --user daemon-reload
    systemctl --user reset-failed >/dev/null 2>&1 || true
  else
    echo "nothing installed"
  fi
  exit 0
fi

[[ "$MODE" == "check" ]] || mkdir -p "$DST_DIR"

changed=0
drift=0
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
for src in $(unit_files); do
  name="$(basename "$src")"
  dst="$DST_DIR/$name"
  render "$src" > "$tmp"
  if [[ -f "$dst" ]] && cmp -s "$tmp" "$dst"; then
    continue
  fi
  drift=1
  if [[ "$MODE" == "check" ]]; then
    if [[ -f "$dst" ]]; then
      echo "drift: $name"
      diff -u "$dst" "$tmp" | sed 's/^/    /' || true
    else
      echo "not installed: $name"
    fi
    continue
  fi
  if [[ -f "$dst" ]] && ! owned_by_us "$dst" && [[ "$FORCE" -ne 1 ]]; then
    echo "refusing to overwrite $name: it belongs to another checkout ($(grep -m1 -oE 'WorkingDirectory=.*' "$dst")); use --force to replace it" >&2
    exit 1
  fi
  install -m 0644 "$tmp" "$dst"
  echo "updated: $name"
  changed=1
done

if [[ "$MODE" == "check" ]]; then
  if [[ "$drift" -eq 0 ]]; then
    echo "all units match ($(unit_files | wc -l | tr -d ' ') files)"
    exit 0
  fi
  echo
  echo "drift found; run ./scripts/install-systemd.sh to sync." >&2
  exit 1
fi

if [[ "$changed" -eq 1 ]]; then
  systemctl --user daemon-reload
  echo "daemon-reload done"
else
  echo "no changes, daemon-reload skipped"
fi

if [[ "$MODE" == "enable" ]]; then
  for unit in "${ENABLE_UNITS[@]}"; do
    systemctl --user enable --now "$unit"
  done
  echo "enabled --now: ${ENABLE_UNITS[*]}"
fi

echo
echo "status:"
for src in $(unit_files); do
  name="$(basename "$src")"
  printf '  %-40s %-10s %s\n' "$name" \
    "$(systemctl --user is-enabled "$name" 2>&1 || true)" \
    "$(systemctl --user is-active "$name" 2>&1 || true)"
done
