# Example: sync an iCloud Drive folder from a Mac

English | [中文](README.zh-CN.md)

The Mac side of the [author's setup](../../docs/example-setup.md), in
which Carrel runs on an NVIDIA DGX Spark. The documents live in an iCloud
Drive folder, one subfolder per knowledge base,
and two LaunchAgents on the Mac keep the server's mirror
(`runtime/mirror`) in step with it:

| Agent | Script | Every minute |
|---|---|---|
| `local.carrel.icloud-prefetch` | `icloud-prefetch.sh` | downloads files that iCloud keeps only in the cloud |
| `local.carrel.mirror-push` | `push-mirror.sh` | copies the folder into the server's mirror with rsync over SSH |

Carrel accepts any way of filling the mirror (see
[Document folders](../../docs/configuration.md#document-folders)). iCloud
Drive needs extra care for two reasons. First, files that are not
downloaded yet are placeholders, which rsync would copy as empty files.
Second, a background job without permission sees the folder as empty, and
`rsync --delete` would then empty the mirror. The push script checks for
both before every round.

## What the push checks

- **Placeholders:** if any file in the folder is still a placeholder, or the
  prefetch is busy, the round is skipped.
- **An empty or unreadable folder:** the push refuses to run, so a missing
  permission cannot empty the mirror.
- **Where it runs:** only under launchd, where the permission below applies.
  From a terminal it refuses unless you pass `--allow-no-launchd`.
- **The server:** if SSH cannot reach it, the round is skipped; nothing on
  the server changes.
- **Locks:** a local lock keeps pushes from overlapping. During the copy the
  push holds `runtime/state/mirror_sync.lock.d` on the server, and Carrel's
  scan waits until it is released. A lock left by a crashed run is taken
  over once it is stale.
- **Half-copied files:** with `--delay-updates`, new and changed files stay
  hidden until the whole batch has arrived: rsync writes them as dot-files
  and parks finished ones in `.rsync-partial/` folders. The scan skips
  hidden files and folders, so it never indexes half a batch.

Deletions are mirrored as well: a file removed from the folder disappears
from the mirror on the next push, and Carrel then removes it from the
index. Carrel keeps the inactive data for an undo window
(`QDRANT_INACTIVE_RETENTION_DAYS`, 7 days in the template), and a file that
comes back within that window is restored.

## Requirements

- Key-based SSH login to the server that works without a prompt:
  `ssh -o BatchMode=yes <host> true` must succeed.
- rsync on the server (most Linux distributions include it).
- Optional: the Xcode Command Line Tools (`xcode-select --install`). With
  them, `install.sh` builds a small download helper from
  `icloud_download.swift`; without them the prefetch uses `brctl download`.

## Install

From a Carrel checkout (or the files of this folder downloaded from GitHub):

```bash
cp -R examples/macos-icloud-sync ~/carrel-sync   # any folder outside iCloud Drive
cd ~/carrel-sync
cp sync.env.example sync.env                     # set SYNC_SOURCE, SYNC_HOST and SYNC_REMOTE_DIR
./install.sh
```

`install.sh` checks the SSH login, builds the helper, writes both agents to
`~/Library/LaunchAgents` and loads them. The agents run the scripts from this
folder, so leave it in place.

Then open **System Settings → Privacy & Security → Full Disk Access** and
add `/bin/bash` and `/bin/zsh` (in the file dialog, Cmd+Shift+G takes a
path). launchd runs the push with bash and the prefetch with zsh. macOS asks
for permission before a program reads iCloud Drive, Desktop or Documents,
and a background job cannot answer that prompt. Without the permission every
round stops at the checks above. The permission also applies to other bash
and zsh scripts that launchd starts, not only to these two, so weigh that
before you grant it.

Follow the log to see the first rounds:

```bash
tail -f ~/carrel-sync/logs/push-mirror.log
```

| Log line | Meaning |
|---|---|
| `pushed: … server flag set` | changes were copied |
| `no-op` | nothing changed |
| `skip: …` | this round waited for something (placeholders, a lock, the network); the next round tries again |
| `abort: …` | a check failed and needs attention (permission, empty folder) |

To preview a push by hand, run
`./push-mirror.sh --dry-run --allow-no-launchd` from a terminal that has
access to the folder. It lists what would change and changes nothing on the
server.

## Server settings

- After a push that changed something, the script touches
  `runtime/state/mirror_changed.flag` on the server. With
  `KB_SCAN_REQUIRE_FLAG=1`, Carrel's scan runs only after such a push. The
  template's `0` scans every minute, and the flag does no harm.
- Keep `SYNC_REMOTE_LOCK_STALE_SECONDS` in `sync.env` equal to
  `KB_MIRROR_LOCK_STALE_SECONDS` on the server (both default to an hour).
- If `KB_MIRROR_ROOT` moves the mirror, set `SYNC_REMOTE_MIRROR` to the same
  place.

## Other folders and drives

The push works for any folder on the Mac: point `SYNC_SOURCE` at it and
install with `./install.sh --push-only`. In an ordinary folder the
placeholder check finds nothing and costs nothing.

A folder kept by the sync client of a NAS or another cloud drive works the
same way, as long as the client keeps its files downloaded on the Mac:
online-only files are placeholders there too, and the push waits until
they are downloaded. macOS may protect such folders as it does iCloud
Drive; if the log shows `abort: source unreadable`, grant Full Disk Access
as described above. Ordinary folders outside Desktop, Documents and
Downloads need no Full Disk Access.

## Files

| File | Purpose |
|---|---|
| `push-mirror.sh` | the push; flags `--dry-run`, `--no-flag` (leave the server flag alone) and `--allow-no-launchd` |
| `icloud-prefetch.sh` | asks iCloud to download placeholders and waits until they are real files |
| `icloud_download.swift` | source of the download helper `bin/icloud-download` |
| `sync.env.example` | settings; optional keys are listed with their defaults |
| `launchd/*.plist` | the two agents; `install.sh` fills in this folder's path |
| `install.sh` | installs the agents, `--push-only`, `--uninstall` |

Each script rotates its own log in `logs/` at 2 MB and keeps three old
copies. The `*.launchd.log` files only collect unexpected errors.
`logs/icloud-prefetch.last-errors.log` lists the files that failed or timed
out in the last prefetch round.

## Uninstall

```bash
~/carrel-sync/install.sh --uninstall
```

This unloads and removes both agents. Delete the folder afterwards, and
remove `/bin/bash` and `/bin/zsh` from Full Disk Access unless something
else needs them.
