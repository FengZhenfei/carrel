from __future__ import annotations

import json
import os
from pathlib import Path
import secrets
import shlex
import time
from urllib.request import urlopen

from .queue import state_dir


def initialize():
    base = Path(os.environ.setdefault("KB_LOCAL_BASE_DIR", "/data"))
    os.environ["KB_RUNTIME"] = "container"
    env_file = Path(
        os.environ.setdefault("KB_ENV_FILE", str(base / "config/knowledge-base.env"))
    )
    edition = os.environ.get("KB_STORAGE_BACKEND", "standard")
    if edition not in ("light", "standard"):
        raise ValueError("Unknown storage edition")
    for directory in (
        "config",
        "runtime/state",
        "runtime/mirror",
        "runtime/mineru-output",
        "logs",
        "backups",
    ):
        (base / directory).mkdir(parents=True, exist_ok=True)
    mode = base / "runtime/deployment-mode"
    if mode.exists() and mode.read_text().strip() != edition:
        raise ValueError("Storage edition is fixed for this data directory")
    if env_file.exists():
        saved = dict(
            line.split("=", 1)
            for line in env_file.read_text().splitlines()
            if "=" in line and not line.lstrip().startswith("#")
        )
        saved_edition = saved.get("KB_STORAGE_BACKEND", "standard").strip().strip("\"'")
        if saved_edition != edition:
            raise ValueError("Saved and selected storage editions differ")
    if not env_file.exists():
        if edition == "standard" and os.environ.get("NEO4J_PASSWORD", "") in (
            "",
            "CHANGE_ME",
        ):
            raise ValueError("Set NEO4J_PASSWORD before starting Standard")
        template = Path(
            os.getenv(
                "KB_CONFIG_TEMPLATE", "/opt/carrel/config/knowledge-base.env.example"
            )
        ).read_text()
        defaults = {
            "KB_STORAGE_BACKEND": edition,
            "QDRANT_URL": "http://qdrant:6333",
            "OPENSEARCH_URL": "http://opensearch:9200",
            "NEO4J_URI": "bolt://neo4j:7687",
            "NEO4J_PASSWORD": os.environ.get("NEO4J_PASSWORD", ""),
            "MINERU_SERVICE_URL": "http://mineru:8000",
            "KB_WEB_HOST": "0.0.0.0",
            "KB_SEARCH_HOST": "0.0.0.0",
            "KB_CONSOLE_SERVICES": "",
            "KB_HOST_HOUSEKEEPING": "0",
            "KB_SEARCH_TOKEN": secrets.token_urlsafe(32),
            # Endpoints are supplied by the user; no loopback model is assumed inside the container.
            "EMBEDDING_BASE_URL": "",
            "EMBEDDING_MODEL_ID": "",
            "EMBEDDING_API_KEY": "",
            "VISUAL_EMBEDDING_BASE_URL": "",
            "VISUAL_EMBEDDING_MODEL_ID": "",
            "VISUAL_EMBEDDING_API_KEY": "",
            "VISUAL_EMBEDDING_ENABLED": "0",
            "RERANKER_BASE_URL": "",
            "RERANKER_MODEL_ID": "",
            "RERANKER_API_KEY": "",
            "VISUAL_RERANKER_BASE_URL": "",
            "VISUAL_RERANKER_MODEL_ID": "",
            "VISUAL_RERANKER_API_KEY": "",
        }
        lines = []
        for line in template.splitlines():
            key, sep, _ = line.partition("=")
            if sep and key.strip() in defaults:
                continue
            lines.append(line)
        lines.extend(f"{k}={shlex.quote(v) if v else ''}" for k, v in defaults.items())
        env_file.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(env_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as out:
            out.write("\n".join(lines) + "\n")
    from kb_pipeline.config import load_settings
    from kb_pipeline import db

    cfg = load_settings(env_file)
    if cfg.storage_backend != edition:
        raise ValueError("Saved and selected storage editions differ")
    db.init_db(cfg.state_db)
    mode.write_text(edition + "\n")


def health():
    status = json.loads((state_dir() / "runtime.json").read_text())
    if time.time() - status["at"] > 20 or status.get("stopping"):
        raise RuntimeError("Background scheduler is not ready")
    for name in ("web", "search"):
        os.kill(status["services"][name], 0)
    for port, path in (
        (os.getenv("KB_WEB_PORT", "9800"), "/"),
        (os.getenv("KB_SEARCH_PORT", "9810"), "/openapi.json"),
    ):
        with urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as response:
            if response.status != 200:
                raise RuntimeError("HTTP service is not ready")


def cleanup_mineru_output():
    import shutil

    root = Path(os.environ["KB_LOCAL_BASE_DIR"]) / "runtime/mineru-output"
    cutoff = (
        time.time()
        - max(1, int(os.getenv("KB_MINERU_OUTPUT_KEEP_MINUTES", "1440"))) * 60
    )
    removed = 0
    for path in root.iterdir() if root.exists() else ():
        if path.is_dir() and not path.is_symlink() and path.stat().st_mtime < cutoff:
            shutil.rmtree(path)
            removed += 1
    print(f"[runtime] MinerU output directories removed: {removed}", flush=True)
