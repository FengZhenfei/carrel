from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

import requests

from .mineru_backend import resolve_backend


def _local_service_session() -> requests.Session:
    session = requests.Session()
    # Local MinerU calls must never be routed through system/VPN proxies.
    session.trust_env = False
    return session


def _load_cached_mineru(output_json: Path, input_path: Path) -> dict[str, Any] | None:
    """Reuse the previous MinerU result. Accepted only when the source-file size recorded in the cache
    matches the current one (the cache directory is already isolated by content_version; this is a
    second safeguard)."""
    if not output_json.exists():
        return None
    try:
        payload = json.loads(output_json.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    stamp = payload.get("_kb_source")
    if isinstance(stamp, dict):
        try:
            if int(stamp.get("size", -1)) != int(input_path.stat().st_size):
                return None
        except (OSError, TypeError, ValueError):
            return None
    return payload


def call_mineru_sync(
    mineru_url: str,
    input_path: Path,
    output_json: Path,
    output_md: Path,
    *,
    table_enable: bool = False,
    return_middle_json: bool = True,
    timeout: int = 43200,
) -> dict[str, Any]:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    # Result cache: cache_dir is already split by kb/file/content_version, so MinerU output for the
    # same version can be reused directly. It used to be written but never read, so any later-stage
    # failure (embedding hiccup, Qdrant timeout, OpenSearch outage) sent the whole document back to
    # the GPU on retry -- minutes to tens of minutes for a large PDF, up to 5 times.
    if os.getenv("MINERU_REUSE_CACHE", "1").strip().lower() in {"1", "true", "yes", "on"}:
        cached = _load_cached_mineru(output_json, input_path)
        if cached is not None:
            print(f"[mineru] reusing cached result for {input_path.name}", flush=True)
            return cached
    backend, _backend_source = resolve_backend()
    request_data = [
        ("lang_list", os.getenv("MINERU_LANG", "ch")),
        ("backend", backend),
        ("parse_method", "auto"),
        ("formula_enable", "true"),
        ("table_enable", "true" if table_enable else "false"),
        ("return_md", "true"),
        ("return_content_list", "true"),
        ("return_middle_json", "true" if return_middle_json else "false"),
        ("return_model_output", "false"),
        ("return_images", "true"),
        ("response_format_zip", "false"),
    ]
    response = None
    for attempt in range(1, 4):
        try:
            # (connect, read): the flat 12-hour timeout doubled as the connect
            # timeout, so an unreachable MinerU blocked the worker for the
            # whole lease. Connection errors (including connect timeouts) are
            # retried; a read timeout is not -- the parse may be mid-flight.
            with input_path.open("rb") as f, _local_service_session() as session:
                response = session.post(
                    f"{mineru_url.rstrip('/')}/file_parse",
                    files=[("files", (input_path.name, f, "application/octet-stream"))],
                    data=request_data,
                    timeout=(10, timeout),
                )
            break
        except requests.ConnectionError as exc:
            if attempt == 3:
                raise
            print(f"[mineru] connection failed attempt={attempt}/3 error={exc!r}", flush=True)
            time.sleep(5 * attempt)
    assert response is not None
    response.raise_for_status()
    payload = response.json()
    try:   # source stamp: used to confirm the file has not changed before reusing the cache
        payload["_kb_source"] = {"name": input_path.name, "size": int(input_path.stat().st_size)}
    except (OSError, TypeError):
        pass
    output_json.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    md = extract_markdown(payload)
    if md:
        output_md.write_text(md, encoding="utf-8")
    return payload


def extract_markdown(payload: Any) -> str:
    # The generic "text" key is no longer a candidate: when a MinerU response lacked md_content
    # (scanned file / error), the recursive search descended into content_list and returned the first
    # block's text as the whole markdown, leaving only the first paragraph of the document. Better to
    # return empty and let the caller take the proper content_list path.
    value = _find_first_string(payload, ("md_content", "markdown", "md"))
    return value.strip() if value else ""


def extract_content_list(payload: Any) -> list[dict[str, Any]]:
    value = _find_first_list(payload, ("content_list", "content", "pages"))
    if not value:
        raw = _find_first_string(payload, ("content_list",))
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    value = parsed
            except json.JSONDecodeError:
                value = []
    if not value:
        return []
    return [item for item in value if isinstance(item, dict)]


def extract_images(payload: Any) -> dict[str, str]:
    value = _find_first_dict(payload, ("images",))
    return {str(k): str(v) for k, v in value.items() if isinstance(v, str)}


def _find_first_string(value: Any, keys: Sequence[str]) -> str:
    # Ordered: iterating a set made the winning key hash-random per process
    # when a payload carried more than one candidate at the same level.
    if isinstance(value, dict):
        for key in keys:
            found = value.get(key)
            if isinstance(found, str) and found.strip():
                return found
        for found in value.values():
            result = _find_first_string(found, keys)
            if result:
                return result
    elif isinstance(value, list):
        for item in value:
            result = _find_first_string(item, keys)
            if result:
                return result
    return ""


def _find_first_list(value: Any, keys: Sequence[str]) -> list[Any]:
    if isinstance(value, dict):
        for key in keys:
            found = value.get(key)
            if isinstance(found, list):
                return found
        for found in value.values():
            result = _find_first_list(found, keys)
            if result:
                return result
    elif isinstance(value, list):
        for item in value:
            result = _find_first_list(item, keys)
            if result:
                return result
    return []


def _find_first_dict(value: Any, keys: Sequence[str]) -> dict[str, Any]:
    if isinstance(value, dict):
        for key in keys:
            found = value.get(key)
            if isinstance(found, dict):
                return found
        for found in value.values():
            result = _find_first_dict(found, keys)
            if result:
                return result
    elif isinstance(value, list):
        for item in value:
            result = _find_first_dict(item, keys)
            if result:
                return result
    return {}


def mineru_health(mineru_url: str, timeout: int = 10) -> dict:
    with _local_service_session() as session:
        response = session.get(f"{mineru_url.rstrip('/')}/health", timeout=timeout)
    response.raise_for_status()
    return response.json()

