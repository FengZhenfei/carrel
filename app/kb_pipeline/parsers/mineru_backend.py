"""Which MinerU backend to request.

The parser container (deployment/compose/mineru) decides at start whether it
runs `vlm-engine` (capable GPU) or `pipeline` (CPU or small GPU) and writes
that decision to `<runtime>/mineru-output/carrel-mineru.json`, a directory it
shares with the host. With MINERU_BACKEND=auto (the default) the pipeline
reads that file, so one host never asks for a backend its parser cannot
serve. An explicit MINERU_BACKEND wins, which is the setting for a MinerU that
runs somewhere else.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

KNOWN_BACKENDS: tuple[str, ...] = ("pipeline", "vlm-engine", "hybrid-engine", "vlm-http-client", "hybrid-http-client")
INFO_FILE_NAME = "carrel-mineru.json"
DEFAULT_BACKEND = "pipeline"

_cache: dict[str, Any] = {"path": None, "mtime": None, "value": None}


def info_file(runtime_dir: str | Path | None = None) -> Path:
    """Where the container publishes its decision. MINERU_INFO_FILE overrides;
    otherwise it sits in the runtime dir the compose file bind-mounts."""
    raw = os.getenv("MINERU_INFO_FILE", "").strip()
    if raw:
        return Path(raw).expanduser()
    if runtime_dir is None:
        from ..config import BASE_DIR

        runtime_dir = os.getenv("KB_RUNTIME_DIR") or (BASE_DIR / "runtime")
    return Path(runtime_dir) / "mineru-output" / INFO_FILE_NAME


def read_info(path: Path) -> dict[str, Any] | None:
    """The container's JSON, re-read only when the file changes."""
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        return None
    if _cache["path"] == path and _cache["mtime"] == mtime:
        return _cache["value"]
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(value, dict):
        return None
    _cache.update(path=path, mtime=mtime, value=value)
    return value


def resolve_backend(runtime_dir: str | Path | None = None) -> tuple[str, str]:
    """(backend, source): source is `env` for an explicit MINERU_BACKEND,
    `container` when the parser's published decision was used, `default`
    when neither exists (pipeline: the only backend every image serves)."""
    raw = (os.getenv("MINERU_BACKEND") or "auto").strip().lower()
    if raw and raw != "auto":
        return raw, "env"
    info = read_info(info_file(runtime_dir))
    backend = str((info or {}).get("backend") or "").strip().lower()
    if backend in KNOWN_BACKENDS:
        return backend, "container"
    return DEFAULT_BACKEND, "default"
