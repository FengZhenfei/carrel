"""A manual parse refreshes model output once; retries reuse this run's progress."""
from __future__ import annotations

import json
import os
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_REFRESH_RUN = ContextVar("parse_cache_refresh_run", default=None)
_STAMP = "_kb_refresh_run"


@contextmanager
def refresh_scope(run_id: str | None):
    token = _REFRESH_RUN.set(run_id)
    try:
        yield
    finally:
        _REFRESH_RUN.reset(token)


def reusable_in_run(payload: dict) -> bool:
    run_id = _REFRESH_RUN.get()
    return run_id is None or payload.get(_STAMP) == run_id


def write_cache(path: Path, payload: dict) -> None:
    """Replace one result atomically, without a second cache tree or deleting good assets first."""
    data = dict(payload)
    data.pop(_STAMP, None)
    if _REFRESH_RUN.get() is not None:
        data[_STAMP] = _REFRESH_RUN.get()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".parse-cache-{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as out:
            if path.exists():
                temporary.chmod(path.stat().st_mode & 0o777)
            json.dump(data, out, ensure_ascii=False, separators=(",", ":"))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
