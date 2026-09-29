from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


def load_env_file(path: str | Path) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        os.environ.setdefault(key, value)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_json_hash(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def normalize_rel_path(path: str) -> str:
    return re.sub(r"/+", "/", path.replace("\\", "/")).strip("/")


def dir_ancestors_from_rel_path(path: str) -> list[str]:
    parts = [part for part in normalize_rel_path(path).split("/") if part]
    return ["/".join(parts[:idx]) for idx in range(1, len(parts))]


def infer_doc_type(filename: str, mime_type: str = "") -> str:
    suffix = Path(filename).suffix.lower().lstrip(".")
    if suffix:
        return suffix
    if "pdf" in mime_type:
        return "pdf"
    if "word" in mime_type:
        return "docx"
    if "presentation" in mime_type:
        return "pptx"
    if "spreadsheet" in mime_type:
        return "xlsx"
    if "markdown" in mime_type:
        return "md"
    if "text" in mime_type:
        return "txt"
    return "unknown"


def split_sentences(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # CJK sentence enders are never followed by a space, so requiring \s+
    # after them meant long Chinese paragraphs never split at all and fell
    # through to arbitrary character cuts. ASCII .!? keep the whitespace
    # requirement -- "3.14" and file names must not split.
    parts = re.split(r"(?<=[。！？；])\s*|(?<=[.!?])\s+|\n{2,}", text)
    return [p.strip() for p in parts if p.strip()]


def approx_tokens(text: str) -> int:
    # Conservative enough for mixed Chinese/English chunk sizing without pulling
    # in a tokenizer at MVP stage.
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    other = len(text) - cjk
    return cjk + max(1, other // 4)


def detect_lang(text: str) -> str:
    """Coarse label for mixed Chinese/English corpora."""
    sample = text[:4000]
    cjk = len(re.findall(r"[\u4e00-\u9fff]", sample))
    latin = len(re.findall(r"[A-Za-z]", sample))
    if not cjk and not latin:
        return "und"
    if cjk and latin and min(cjk, latin) / max(cjk, latin) >= 0.25:
        return "mixed"
    return "zh" if cjk >= latin else "en"


_ENCODER = None


def pin_tokenizer_cache(runtime_dir: str | Path) -> None:
    """By default tiktoken caches its encoding files in the system temp directory, which is cleared at boot; the
    next count then has to download them again, and when that fails the whole process falls back to the
    estimate. Point the cache at tiktoken/ under the runtime directory; a cache directory already set in the
    environment is left alone."""
    if "TIKTOKEN_CACHE_DIR" not in os.environ and "DATA_GYM_CACHE_DIR" not in os.environ:
        os.environ["TIKTOKEN_CACHE_DIR"] = str(Path(runtime_dir) / "tiktoken")


def count_tokens(text: str, encoding_model: str = "o200k_base") -> int:
    """Real token count for context budgeting. Falls back to the heuristic."""
    global _ENCODER
    if _ENCODER is None:
        try:
            import tiktoken

            _ENCODER = tiktoken.get_encoding(encoding_model)
        except Exception as exc:
            _ENCODER = False
            # The estimate can be 20-30% off the real tokenizer: chunk boundaries and the evidence budget of
            # search change their measure along with it, so this must not happen silently
            cache_dir = os.getenv("TIKTOKEN_CACHE_DIR") or os.getenv("DATA_GYM_CACHE_DIR") or "the system temp directory"
            print(f"[tokenizer] {encoding_model} is unavailable ({exc!r}); this process counts tokens with the heuristic "
                  f"until it restarts. The encoding file is cached in {cache_dir}", flush=True)
    if _ENCODER is False:
        return approx_tokens(text)
    return len(_ENCODER.encode(text, disallowed_special=()))


def looks_like_kb_process(command: str) -> bool:
    """Whether a command line is one of this project's own processes: `python -m kb_pipeline …`, or the `kb`
    command registered in pyproject (`.venv/bin/kb …`). The worker's orphan detection and the maintenance
    module's stop / delete-KB rely on it to recognize our processes; it used to recognize only the former,
    so kb processes started by hand / by an ad-hoc unit were taken for dead (a parse job was wrongly
    reclaimed on 2026-09-09; Codex review N10)."""
    tokens = str(command or "").split()
    return any("kb_pipeline" in t for t in tokens) or any(t == "kb" or t.endswith("/kb") for t in tokens)


def looks_like_worker_command(command: str) -> bool:
    tokens = str(command or "").split()
    return "worker" in tokens and "--once" in tokens and looks_like_kb_process(command)
