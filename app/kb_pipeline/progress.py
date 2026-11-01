"""Small, language-independent progress records shared by workers and the console."""
from __future__ import annotations

import json
import re
from typing import Any


# Existing stage text remains available in logs and timelines. Codes are the API contract.
PHASES = (
    ("parse", "Parsing document"), ("vision", "Describing images"),
    ("chunk", "Chunking"), ("embed_text", "Text embedding"),
    ("embed_visual", "Visual embedding"), ("write_vectors", "Writing vectors"),
    ("keyword", "Keyword indexing"), ("schema", "Preparing labels"),
    ("prepare", "Preparing corpus"), ("extract", "Entity extraction"),
    ("resolve", "Entity resolution"), ("summarize", "Description summaries"),
    ("resolve", "Merge & resolution"),
    ("attribute", "Computing weights and chunk attribution"),
    ("facts", "Structured facts"), ("concepts", "Normalizing property concepts"),
    ("compile", "Compiling view pages"), ("import", "Graph database import"),
    ("publish", "Switching version aliases"), ("cleanup", "Cleaning up old versions"),
    ("done", "Done"),
)


def read_progress(value: Any) -> dict[str, Any]:
    try:
        out = json.loads(value or "{}") if isinstance(value, (str, bytes)) else value
    except (ValueError, TypeError):
        return {}
    return out if isinstance(out, dict) else {}


def phase_code(label: str) -> str:
    return next((code for code, prefix in PHASES if label.startswith(prefix)), "other")


def advance(previous: Any, label: str, now: int, *, models: dict[str, str] | None = None) -> dict[str, Any]:
    old = read_progress(previous)
    phase = phase_code(label)
    # Substeps have separate counters (entities / relations, subjects / timelines, etc.).
    part = re.search(r"·\s*([^\d(]+)", label)
    subphase = part.group(1).strip() if part else ""
    same = (old.get("phase"), old.get("subphase")) == (phase, subphase)
    out: dict[str, Any] = {
        "phase": phase, "subphase": subphase,
        "started_at": old.get("started_at", now) if same else now,
        "updated_at": now,
        "models": {**old.get("models", {}), **(models or {})},
    }
    count = re.search(r"(\d+)\s*/\s*(\d+)", label)
    if count:
        out.update(completed=int(count[1]), total=int(count[2]))
    cached = re.search(r"cached (\d+)", label)
    if cached:
        out["cached"] = int(cached[1])
    if "done, skipped" in label:
        out["reused"] = True
    retry = re.search(r"attempt (\d+) failed, retrying in (\d+)s", label)
    if retry:
        out.update(wait_reason="retry", attempt=int(retry[1]), next_attempt_at=now + int(retry[2]))
    return out


def job_view(row: Any, now: int) -> dict[str, Any]:
    """Only known scheduler state determines waiting; elapsed time never implies failure."""
    data = dict(row)
    out = read_progress(data.get("progress_json")).copy()
    status = data.get("job_status", data.get("status"))
    if status == "retry":
        error = str(data.get("error") or "")
        out["wait_reason"] = "service" if error.startswith("Service not ready,") else "retry"
        out["next_attempt_at"] = data.get("next_attempt_at")
    elif status == "queued":
        out["wait_reason"] = "queue"
    if data.get("cancel_requested") and status in {"running", "queued", "retry"}:
        out["wait_reason"] = "stopping"
    if status == "running" and out.get("started_at"):
        out["elapsed_seconds"] = max(0, now - int(out["started_at"]))
    return out
