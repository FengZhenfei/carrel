"""Graph build lock: only one graph build / Neo4j import at a time.

The kernel ties the lock to the process (flock): when the holder exits, crashes or is SIGKILLed the lock is
released automatically; to find out whether a build is running, probe the lock without blocking.
No pid and no command line are consulted, so a reused pid cannot mislead it -- before this it was "directory +
pid file", where the lock itself did not know whether its holder was dead and the outside had to guess ("pid
alive + command line looks like one of ours"); missing one way of launching was enough to break it (2026-09-09:
a manually started `kb graph build` was cleared as a stale lock and two builds ran side by side).
The lock directory still holds pid / started_at, used only for display and for signalling "pause build"; they
play no part in deciding whether the lock is valid.
The lock file is never deleted: deleting and recreating it makes another inode, and two processes would each
lock a different file.
"""
from __future__ import annotations

import fcntl
import os
import time
from pathlib import Path
from typing import Any

LOCK_DIR_NAME = "graph_build.lock.d"
LOCK_FILE_NAME = "lock"


def build_lock_path(settings: Any) -> Path:
    return Path(settings.runtime_dir) / "state" / LOCK_DIR_NAME


def build_lock_held(path: Path) -> bool:
    """Whether a live process holds this lock: try a non-blocking acquire and release it at once if it succeeds.
    No lock file = no build ever ran, which does not count as held."""
    try:
        fd = os.open(Path(path) / LOCK_FILE_NAME, os.O_RDWR | os.O_CLOEXEC)
    except OSError:
        return False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def build_lock_holder(path: Path) -> dict[str, int]:
    """The holder information recorded in the lock directory (pid, start time), for display and signalling only;
    empty when unavailable."""
    out: dict[str, int] = {}
    for name in ("pid", "started_at"):
        try:
            out[name] = int((Path(path) / name).read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            pass
    return out


def clear_lock_leftovers(path: Path) -> bool:
    """When nobody holds the lock, clear the pid / started_at left by the previous holder (a SIGKILLed process had no
    chance to delete them). Returns False and touches nothing when someone holds the lock."""
    path = Path(path)
    if not path.exists() or build_lock_held(path):
        return False
    removed = False
    for name in ("pid", "started_at"):
        try:
            (path / name).unlink()
            removed = True
        except OSError:
            pass
    return removed


class GraphBuildLock:
    def __init__(self, settings: Any) -> None:
        self.path = build_lock_path(settings)
        self.acquired = False
        self._fd: int | None = None

    def acquire(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path / LOCK_FILE_NAME, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            raise RuntimeError(f"Graph build lock already exists: {self.path}") from exc
        self._fd = fd
        self.acquired = True
        (self.path / "pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
        (self.path / "started_at").write_text(f"{int(time.time())}\n", encoding="utf-8")

    def release(self) -> None:
        if not self.acquired:
            return
        for name in ("pid", "started_at"):
            try:
                (self.path / name).unlink()
            except OSError:
                pass
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(self._fd)
            self._fd = None
        self.acquired = False
