"""Console configuration for embedding and reranking endpoints."""
from __future__ import annotations

import hashlib
import json
import math
from urllib.parse import urlsplit

import requests

from kb_pipeline import db
from kb_pipeline.config import load_settings
from kb_pipeline.model_endpoints import CONFIG_KEY, ROLES, endpoint_values, read_overrides


def _revision(row: dict) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()


def _connection_signature(role: str, row: dict) -> str:
    fields = {k: row.get(k, "") for k in ("base_url", "model_id", "api_key")}
    if role == "embedding":
        fields["dim"] = row.get("dim")
    return hashlib.sha256(json.dumps([role, fields], sort_keys=True).encode()).hexdigest()


def _public(role: str, row: dict) -> dict:
    result = {"role": role, **{k: v for k, v in row.items() if k not in {"api_key", "_validated_signature"}},
              "has_api_key": bool(row.get("api_key")), "revision": _revision(row)}
    if role == "embedding" or "reranker" in role:
        result["validated"] = row.get("_validated_signature") == _connection_signature(role, row)
    return result


def get_endpoints() -> dict:
    values = endpoint_values(load_settings())
    return {"endpoints": [_public(role, values[role]) for role in ROLES]}


def reveal_endpoint_key(role: str, payload: dict) -> dict:
    values = endpoint_values(load_settings())
    if role not in values:
        raise KeyError(role)
    if payload.get("revision") != _revision(values[role]):
        raise ValueError("Model settings changed; refresh and try again")
    return {"api_key": values[role]["api_key"]}


def _candidate(role: str, payload: dict, current: dict, *, require_connection: bool = False) -> dict:
    if role not in ROLES:
        raise KeyError(role)
    allowed = {"base_url", "model_id", "api_key", "clear_api_key", "enabled", "revision"}
    if "embedding" in role:
        allowed.update(("dim", "batch_size", "concurrency"))
    if set(payload) - allowed:
        raise ValueError("Unknown model endpoint field")
    if payload.get("revision") != _revision(current):
        raise ValueError("Model settings changed; refresh and try again")
    row = dict(current)
    for field in ("base_url", "model_id", "api_key"):
        if field in payload and not isinstance(payload[field], str):
            raise ValueError("Model endpoint fields must be text")
    for field in ("base_url", "model_id"):
        if field in payload:
            row[field] = payload[field].strip()
    row["base_url"] = row["base_url"].rstrip("/")
    for field in ("enabled", "clear_api_key"):
        if field in payload and not isinstance(payload[field], bool):
            raise ValueError("Model endpoint switches must be boolean")
    row["enabled"] = payload.get("enabled", row["enabled"])
    if role == "embedding" and not row["enabled"]:
        raise ValueError("Text embedding is required")
    if "embedding" in role:
        dim = payload.get("dim", row["dim"])
        if isinstance(dim, bool) or not isinstance(dim, int) or not 1 <= dim <= 65536:
            raise ValueError("Embedding dimensions must be an integer from 1 to 65536")
        row["dim"] = dim
        for field, maximum, message in (
            ("batch_size", 1024, "Batch size must be an integer from 1 to 1024"),
            ("concurrency", 32, "Concurrency must be an integer from 1 to 32"),
        ):
            value = payload.get(field, row[field])
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
                raise ValueError(message)
            row[field] = value
    if row["base_url"]:
        try:
            url = urlsplit(row["base_url"])
            valid = url.scheme in {"http", "https"} and url.hostname and url.port != 0
            valid = valid and not (url.username or url.password or url.query or url.fragment)
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("Use an HTTP or HTTPS API address without credentials or query parameters")
    if (row["enabled"] or require_connection) and not row["base_url"]:
        raise ValueError("API address is required")
    if (row["enabled"] or require_connection) and "embedding" in role and not row["model_id"]:
        raise ValueError("Model ID is required")
    key = payload.get("api_key", "").strip()
    clear = payload.get("clear_api_key", False)
    if clear and key:
        raise ValueError("Enter a key or clear it, not both")
    if (urlsplit(row["base_url"]).netloc != urlsplit(current["base_url"]).netloc
            or urlsplit(row["base_url"]).scheme != urlsplit(current["base_url"]).scheme):
        if current["api_key"] and not key and not clear:
            raise ValueError("Re-enter or clear the API key when changing the endpoint host")
    row["api_key"] = "" if clear else key or current["api_key"]
    return row


def _probe(role: str, row: dict) -> dict:
    if not row["enabled"]:
        return row
    base = row["base_url"].rstrip("/")
    headers = {"Authorization": f"Bearer {row['api_key']}"} if row["api_key"] else {}
    # Synthetic input only: never send knowledge-base content during configuration.
    text = "Connection test."
    try:
        if "reranker" in role:
            from kb_search.rerank import Reranker
            client = Reranker(base, model_id=row["model_id"], api_key=row["api_key"], timeout=15)
            query = text
            docs = [text, "Another example."]
            if role == "visual_reranker":
                image = {"type": "image_url", "image_url": {"url": _test_image()}}
                query = {"content": [{"type": "text", "text": text}, image]}
                # Every candidate is an actual image, with no text-only fallback.
                other_image = {"type": "image_url", "image_url": {"url": _test_image((180, 130, 80))}}
                docs = [{"content": [image]}, {"content": [other_image]}]
            try:
                model = client.resolve_model()
                response = requests.post(client.url,
                                         json={"model": model, "query": query, "documents": docs},
                                         headers=headers, timeout=15, allow_redirects=False)
                response.raise_for_status()
                result = response.json()
            except ValueError:
                raise RuntimeError("Invalid reranking response") from None
            # Configuration checks connectivity only. Runtime retrieval owns
            # score/index handling; no expected count, ordering or score here.
            if not isinstance(result, dict) or not result.get("results"):
                raise RuntimeError("Empty reranking response")
            checked = {**row, "model_id": model}
            # Keep only proof of which configuration passed, never test scores.
            return {**checked, "_validated_signature": _connection_signature(role, checked)}
        body = {"model": row["model_id"], "dimensions": row["dim"], "encoding_format": "float"}
        # Verify the batch contract with two synthetic inputs, without a load test.
        count = min(row["batch_size"], 2)
        if role == "embedding":
            body["input"] = [text] * count
        else:
            # Same messages contract as the visual pipeline, using a generated image.
            body["messages"] = [{"role": "system", "content": "Represent the user's input."},
                                {"role": "user", "content": [{"type": "image_url", "image_url": {
                                    "url": _test_image()}}]}]
            if count > 1:
                body["messages"] = [body["messages"] for _ in range(count)]
        response = requests.post(base + "/embeddings", json=body, headers=headers,
                                 timeout=15, allow_redirects=False)
        response.raise_for_status()
        try:
            items = response.json().get("data") or []
        except (ValueError, AttributeError):
            raise RuntimeError("Invalid embedding response") from None
        if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
            raise RuntimeError("Invalid embedding response")
        if len(items) != count:
            raise ValueError("Embedding response count does not match the batch size")
        indices = [item.get("index", 0 if count == 1 else None) for item in items]
        if (any(isinstance(i, bool) or not isinstance(i, int) for i in indices)
                or sorted(indices) != list(range(count))):
            raise ValueError("Model connection test failed: invalid response")
        for item in items:
            vector = item.get("embedding")
            if (not isinstance(vector, list) or len(vector) != row["dim"] or
                    any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in vector)):
                raise ValueError("Embedding response does not match the configured dimensions")
    except requests.HTTPError as exc:
        # Provider bodies/URLs may contain credentials; return a bounded status only.
        raise ValueError(f"Model connection test failed: HTTP {exc.response.status_code}") from None
    except requests.RequestException:
        raise ValueError("Model connection test failed: endpoint unavailable") from None
    except (KeyError, TypeError, IndexError, RuntimeError):
        raise ValueError("Model connection test failed: invalid response") from None
    if role == "embedding":
        # Record validation for this vector space, never the returned vectors.
        return {**row, "_validated_signature": _connection_signature(role, row)}
    return row


def _test_image(color: tuple[int, int, int] = (80, 130, 180)) -> str:
    from base64 import b64encode
    from io import BytesIO
    from PIL import Image
    buf = BytesIO()
    Image.new("RGB", (32, 32), color).save(buf, format="PNG")
    return "data:image/png;base64," + b64encode(buf.getvalue()).decode()


def probe_endpoint(role: str, payload: dict) -> dict:
    values = endpoint_values(load_settings())
    if role not in values:
        raise KeyError(role)
    check_off = "reranker" in role
    candidate = _candidate(role, payload, values[role], require_connection=check_off)
    row = _probe(role, {**candidate, "enabled": candidate["enabled"] or check_off})
    return {"ok": True, "model_id": row["model_id"]}


def _check_embedding_change(con, cfg, role: str, current: dict, row: dict) -> None:
    if "embedding" not in role or all(row[k] == current[k] for k in
                                     ("base_url", "model_id", "api_key", "dim", "enabled")):
        return
    from kb_pipeline.graph.lock import build_lock_held, build_lock_path
    busy = con.execute("SELECT 1 FROM jobs WHERE job_type='parse' AND status='running' LIMIT 1").fetchone()
    busy = busy or con.execute("SELECT 1 FROM graph_builds WHERE status='running' LIMIT 1").fetchone()
    if busy or build_lock_held(build_lock_path(cfg)):
        raise ValueError("Wait for parsing and graph tasks to finish before changing embedding settings")
    changed_space = any(row[k] != current[k] for k in ("base_url", "model_id", "dim"))
    if changed_space:
        indexed = con.execute("SELECT 1 FROM chunks LIMIT 1").fetchone()
        indexed = indexed or con.execute("SELECT 1 FROM graph_builds WHERE status='done' LIMIT 1").fetchone()
        if indexed:
            raise ValueError("Embedding endpoint, model or dimensions cannot change while indexes exist; reinitialize indexes first")
        if row["dim"] != current["dim"] and con.execute("SELECT 1 FROM kb_sources LIMIT 1").fetchone():
            raise ValueError("Embedding dimensions cannot change while knowledge-base collections exist")


def save_endpoint(role: str, payload: dict) -> dict:
    cfg = load_settings()
    saved = read_overrides(cfg.state_db)
    current = endpoint_values(cfg, saved)
    if role not in current:
        raise KeyError(role)
    # Saving endpoint fields tests the connection even if the role stays off.
    # A switch-only request can still disable an unavailable endpoint.
    configuring = bool(set(payload) - {"enabled", "revision"})
    candidate = _candidate(role, payload, current[role], require_connection=configuring)
    db.init_db(cfg.state_db)
    with db.connect(cfg.state_db) as con:
        _check_embedding_change(con, cfg, role, current[role], candidate)
    enabled = candidate["enabled"]
    candidate = {**_probe(role, {**candidate, "enabled": enabled or configuring}), "enabled": enabled}
    with db.connect(cfg.state_db) as con:
        con.execute("BEGIN IMMEDIATE")
        if db.get_app_config(con, CONFIG_KEY, {}) != saved:
            raise ValueError("Model settings changed; refresh and try again")
        _check_embedding_change(con, cfg, role, current[role], candidate)
        db.set_app_config(con, CONFIG_KEY, {**saved, role: candidate})
    return _public(role, candidate)
