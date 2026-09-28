"""Description summaries: the descriptions accumulated for one entity / relation get one LLM call only once
they reach a threshold; below it they are simply concatenated.

2026-09-07, thresholds changed to LightRAG's terms: summarise only with >= SUMMARY_MIN_DESCRIPTIONS distinct
descriptions or a total above SUMMARY_MIN_TOKENS (it used to summarise from 2 descriptions on; one kb_001
build made 960 calls, the most expensive part of the build). Descriptions go into the prompt as JSON lines
with their source document and axis value: statements from different sources / times must be attributed side
by side and never flattened (age 30 and 31 are two reports two years apart, not a contradiction). Batched by
token budget, with chained summaries above the budget; on failure it falls back to concatenating the raw
descriptions (the semantics of P3)."""
from __future__ import annotations

import json
from typing import Any, Callable

from ..utils import count_tokens
from . import prompts
from .llm import ChatClient, LLMCallError

DEFAULT_MAX_LENGTH_WORDS = 300
DEFAULT_MAX_INPUT_TOKENS = 4000
SUMMARY_MIN_DESCRIPTIONS = 4
SUMMARY_MIN_TOKENS = 800
CONCAT_MAX_CHARS = 4000


def _batches(descriptions: list[dict[str, str]], budget: int) -> list[list[dict[str, str]]]:
    batches: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []
    tokens = 0
    for d in descriptions:
        n = count_tokens(json.dumps(d, ensure_ascii=False))
        if current and tokens + n > budget:
            batches.append(current)
            current, tokens = [], 0
        current.append(d)
        tokens += n
    if current:
        batches.append(current)
    return batches


def description_rows(row: dict[str, Any]) -> list[dict[str, str]]:
    """The descriptions of one row (entity / relation) -> [{"text", "source"?, "when"?}], deduplicated, order kept.
    merge records which document each description came from (description_sources, aligned with descriptions);
    an old graph.json without it yields text only."""
    texts = [str(d).strip() for d in (row.get("descriptions") or [])]
    sources = list(row.get("description_sources") or [])
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for i, text in enumerate(texts):
        if not text or text in seen:
            continue
        seen.add(text)
        item: dict[str, str] = {"text": text}
        src = sources[i] if i < len(sources) and isinstance(sources[i], dict) else {}
        if src.get("source"):
            item["source"] = str(src["source"])
        if src.get("when"):
            item["when"] = str(src["when"])
        out.append(item)
    return out


def needs_summary(rows: list[dict[str, str]], *, min_descriptions: int = SUMMARY_MIN_DESCRIPTIONS,
                  min_tokens: int = SUMMARY_MIN_TOKENS) -> bool:
    if len(rows) < 2:
        return False
    if len(rows) >= min_descriptions:
        return True
    return sum(count_tokens(r["text"]) for r in rows) >= min_tokens


def concatenate(rows: list[dict[str, str]], *, limit: int = CONCAT_MAX_CHARS) -> str:
    """Rows below the threshold: descriptions are labelled with their source and concatenated. The same source is
    not labelled twice."""
    parts: list[str] = []
    for r in rows:
        label = r.get("source") or ""
        if r.get("when"):
            label = f"{label} ({r['when']})" if label else r["when"]
        parts.append(f"[{label}] {r['text']}" if label and len(rows) > 1 else r["text"])
    return "\n".join(parts)[:limit]


def summarize_one(client: ChatClient, name: str, descriptions: list[Any], *, language: str,
                  max_length: int = DEFAULT_MAX_LENGTH_WORDS, max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS) -> str:
    """descriptions: a list of strings or of {"text", "source", "when"} dicts."""
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for d in descriptions:
        item = dict(d) if isinstance(d, dict) else {"text": str(d)}
        text = str(item.get("text") or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        item["text"] = text
        rows.append({k: str(v) for k, v in item.items() if v})
    if not rows:
        return ""
    if len(rows) == 1:
        return rows[0]["text"]
    current = rows
    while True:
        batches = _batches(current, max_input_tokens)
        outputs: list[dict[str, str]] = []
        for batch in batches:
            prompt = prompts.SUMMARIZE_PROMPT.format(
                language=language or "English", max_length=int(max_length), entity_name=name,
                description_list="\n".join(json.dumps(d, ensure_ascii=False) for d in batch),
            )
            outputs.append({"text": client.chat(prompt, max_tokens=1024).strip()})
        if len(outputs) == 1:
            return outputs[0]["text"]
        current = outputs


def summarize_rows(client: ChatClient, rows: list[dict[str, Any]], *, language: str, name_of: Callable[[dict[str, Any]], str],
                   progress: Callable[[int, int], None] | None = None) -> dict[str, int]:
    """Rows that reach the threshold are summarised in parallel and the result written into row["description"];
    the rest are concatenated."""
    prepared = {id(row): description_rows(row) for row in rows}
    todo = [row for row in rows if needs_summary(prepared[id(row)])]
    stats = {"rows": len(rows), "summarized": 0, "failed": 0, "skipped": len(rows) - len(todo), "concatenated": 0}

    def work(row: dict[str, Any]) -> str:
        return summarize_one(client, name_of(row), prepared[id(row)], language=language)

    for row, summary, error in client.run_parallel(todo, work, progress=progress):
        if error is None and summary:
            row["description"] = summary
            stats["summarized"] += 1
        else:
            stats["failed"] += 1
            if error is not None and not isinstance(error, LLMCallError):
                print(f"[graph] summarize failed for {name_of(row)!r}: {error!r}", flush=True)
    for row in rows:
        if not row.get("description"):
            row["description"] = concatenate(prepared[id(row)])
            stats["concatenated"] += 1
    return stats
