"""Tables flagged as ambiguous: give the VLM the table screenshot MinerU left behind and have it
transcribe only those rows; the result is adopted only when the concatenation matches (Codex review
F02).

Safety boundary: the split the model gives must, with whitespace removed, concatenate back to the
merged raw string (75 + 75 + 57 = 757557, mA x 3 = mAmAmA), and every merged cell in a row must have
the same number of segments; condition cells (t_RC = 20 ns ...) are not compared by concatenation and
are split along only when the segment count matches and every segment can be found in the raw
string. On a mismatch the original text and the flags are kept; nothing is guessed. Results are
cached by screenshot fingerprint + prompt.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable

from ..models import ParsedBlock
from ..vision.vlm import DEFAULT_MAX_PIXELS, _client_for, image_data_url
from .common import grid_to_markdown, html_table_to_grid
from .table_check import GLUED_KINDS, expand_glued_rows, high_confidence

TABLE_ROWS_PROMPT = """This image is a table cropped from a document. Some cells contain several lines stacked vertically (one line per test condition), and an OCR pass merged those lines into one string.
The table has {width} columns; its header row reads: {header}.
Transcribe ONLY the table rows whose first cell is one of: {keys}.
For every one of the {width} cells of each such row, in column order, give its lines top-to-bottom as a JSON array of strings: a single-line cell is a one-element array, an empty cell is an empty array. Copy numbers and units exactly as printed; do not compute, convert or normalise anything.
Return JSON only, in this exact shape:
{{"rows": [{{"key": "<first cell text>", "cells": [["line 1", "line 2"], ["only line"], []]}}]}}
"""
REPAIR_CONTRACT_VERSION = 2
_TAG_RE = re.compile(r"<[^>]{1,20}>")
_NORM_RE = re.compile(r"[\s$\\{}_^]+")


_SUB_RE = re.compile(r"<sub>(.*?)</sub>", re.IGNORECASE | re.DOTALL)
_SUP_RE = re.compile(r"<sup>(.*?)</sup>", re.IGNORECASE | re.DOTALL)


def clean_part(text: str) -> str:
    """HTML markup from the VLM transcription stays out of the body: <sub>RC</sub> -> _RC (the manual's
    own notation), <sup>2</sup> -> ^2, other tags are removed."""
    out = _SUB_RE.sub(lambda m: "_" + m.group(1).strip(), str(text or ""))
    out = _SUP_RE.sub(lambda m: "^" + m.group(1).strip(), out)
    return re.sub(r"\s+", " ", _TAG_RE.sub("", out)).strip()


def _norm(text: str) -> str:
    """Normalised form for comparison: strip HTML tags (the VLM likes to write <sub>), LaTeX markup
    ($ { } \\ _ ^) and whitespace, case-insensitive."""
    return _NORM_RE.sub("", _TAG_RE.sub("", str(text or ""))).casefold()


def verify_split(raw: str, parts: list[str]) -> bool:
    """Trust a split only when the segments, with whitespace removed, concatenate back to the raw string."""
    parts = [str(p) for p in parts]
    return len(parts) >= 2 and all(p.strip() for p in parts) and _norm("".join(parts)) == _norm(raw)


def _parse_rows(content: str) -> list[dict[str, Any]]:
    body = str(content or "")
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        return []
    try:
        data = json.loads(body[start:end + 1])
    except ValueError:
        try:
            data = json.loads(re.sub(r",\s*([}\]])", r"\1", body[start:end + 1]))
        except ValueError:
            return []
    rows = data.get("rows") if isinstance(data, dict) else None
    out: list[dict[str, Any]] = []
    for r in rows or []:
        if not isinstance(r, dict) or not isinstance(r.get("cells"), list):
            continue
        cells = [[str(x) for x in c] if isinstance(c, list) else [str(c)] for c in r["cells"]]
        out.append({"key": str(r.get("key") or ""), "cells": cells})
    return out


def transcribe_rows(image_path: Path, *, base_url: str, api_key: str, model_id: str, row_keys: list[str],
                    header: list[str] | None = None, width: int = 0,
                    cache_json: Path | None = None, temperature: float = 0.1, top_p: float = 0.8,
                    max_tokens: int = 1500, max_pixels: int | None = DEFAULT_MAX_PIXELS) -> list[dict[str, Any]]:
    prompt = TABLE_ROWS_PROMPT.format(keys="; ".join(f'"{k}"' for k in row_keys), width=width or len(header or []) or "?",
                                      header=" | ".join(header or []) or "(unknown)")
    identity = {"contract_version": REPAIR_CONTRACT_VERSION, "model_id": model_id,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "temperature": temperature, "top_p": top_p, "max_pixels": max_pixels}
    if cache_json and cache_json.exists():
        try:
            cached = json.loads(cache_json.read_text(encoding="utf-8"))
        except Exception:
            cached = None
        if isinstance(cached, dict) and cached.get("_cache") == identity:
            return _parse_rows(json.dumps(cached.get("data") or {}))
    client = _client_for(base_url, api_key)
    resp = client.chat.completions.create(
        model=model_id, temperature=temperature, top_p=top_p, max_tokens=max_tokens,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_data_url(image_path, max_pixels)}},
        ]}],
    )
    content = resp.choices[0].message.content or ""
    rows = _parse_rows(content)
    if cache_json and rows:
        cache_json.parent.mkdir(parents=True, exist_ok=True)
        cache_json.write_text(json.dumps({"data": {"rows": rows}, "_cache": identity}, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    return rows


def _find_rebuild(cells: list[list[str]], raw: str) -> list[str] | None:
    """Find, in any cell of the transcription, a run of consecutive lines (>= 2) that concatenate, with
    whitespace removed, to exactly the merged raw string. Does not rely on column alignment: the VLM
    often merges blanks / "-" into a neighbouring cell, or skips the first cell."""
    target = _norm(raw)
    if not target:
        return None
    for parts in cells:
        parts = [str(p) for p in parts]
        for i in range(len(parts)):
            acc = ""
            for j in range(i, len(parts)):
                acc += _norm(parts[j])
                if acc == target and j - i + 1 >= 2 and all(p.strip() for p in parts[i:j + 1]):
                    return [clean_part(p) for p in parts[i:j + 1]]
                if len(acc) >= len(target):
                    break
    return None


def _find_lines(cells: list[list[str]], raw: str, n: int) -> list[str] | None:
    """Condition cells: find a run of n consecutive lines, each of which after normalisation occurs in
    the raw string and all distinct from each other -- the condition text is wrapped in LaTeX markup
    and cannot be concatenated back to the raw string, so this is the only way to match it."""
    target = _norm(raw)
    for parts in cells:
        parts = [str(p) for p in parts]
        for i in range(0, len(parts) - n + 1):
            seg = [p.strip() for p in parts[i:i + n]]
            norms = [_norm(p) for p in seg]
            if all(norms) and all(x in target for x in norms) and len(set(norms)) == n:
                return [clean_part(p) for p in seg]
    return None


def plan_repairs(grid: list[list[str]], flags: list[dict[str, Any]],
                 transcribed: list[dict[str, Any]]) -> tuple[dict[int, dict[int, list[str]]], list[str]]:
    """Verification: for every flagged row, every merged cell must have a run of lines in the
    transcription that rebuilds the raw string (with a consistent segment count); a multi-condition
    cell is split along only when n lines all present in the raw string are found. Returns
    ({row index: {column index: lines}}, reasons for failures)."""
    rows_flags: dict[int, list[dict[str, Any]]] = {}
    for f in high_confidence(flags):
        rows_flags.setdefault(int(f["row"]), []).append(f)
    repairs: dict[int, dict[int, list[str]]] = {}
    reasons: list[str] = []
    for i, row_flags in sorted(rows_flags.items()):
        row = grid[i] if i < len(grid) else []
        key = _norm(row[0] if row else "")
        matched = [r for r in transcribed if r.get("key") and (_norm(r["key"]) == key or (key and (key in _norm(r["key"]) or _norm(r["key"]) in key)))]
        if not matched and len(rows_flags) == 1 and len(transcribed) == 1:
            matched = list(transcribed)                 # asked for one row, got one row back: match by position
        if not matched:
            matched = list(transcribed)                 # keys do not match: search all results (content matching is strict enough by itself)
        if not matched:
            reasons.append(f"row {i}: nothing transcribed")
            continue
        cells = [c for r in matched for c in r.get("cells") or []]
        plan: dict[int, list[str]] = {}
        n = 0
        ok = True
        for f in row_flags:
            if f.get("kind") not in GLUED_KINDS:
                continue
            j = int(f["col"])
            parts = _find_rebuild(cells, row[j] if j < len(row) else "")
            if parts is None:
                reasons.append(f"row {i} col {j}: no transcribed lines rebuild {row[j]!r}")
                ok = False
                break
            if n and len(parts) != n:
                reasons.append(f"row {i}: inconsistent line counts ({n} vs {len(parts)})")
                ok = False
                break
            n = len(parts)
            plan[j] = parts
        if not ok or not plan:
            continue
        for j, cell in enumerate(row):
            if j in plan or not cell.strip():
                continue
            lines = _find_lines(cells, cell, n)
            if lines is not None:
                plan[j] = lines
        repairs[i] = plan
    return repairs, reasons


def repair_ambiguous_tables(
    blocks: list[ParsedBlock],
    *,
    base_url: str,
    api_key: str,
    model_id: str,
    cache_dir: Path,
    vlm_options: dict[str, Any] | None = None,
    caption_cache_root: Path | None = None,
    transcribe: Callable[..., list[dict[str, Any]]] = transcribe_rows,
) -> dict[str, int]:
    """Run screenshot verification on table blocks with high-confidence table_flags; on success the
    merged rows are expanded and text / table_markdown rewritten, with the result recorded in
    metadata.table_repair. A failure on any table never affects the parse itself."""
    stats = {"flagged": 0, "verified": 0, "unverified": 0, "failed": 0}
    opts = dict(vlm_options or {})
    kwargs = {k: opts[k] for k in ("temperature", "top_p", "max_pixels") if k in opts}
    for block in blocks:
        flags = high_confidence(block.metadata.get("table_flags"))
        if block.block_type != "table" or not flags:
            continue
        stats["flagged"] += 1
        html = str((block.metadata.get("source_item") or {}).get("table_body") or "")
        grid = html_table_to_grid(html)
        image = Path(block.visual_ref) if block.visual_ref else None
        if not grid:
            block.metadata["table_repair"] = {"status": "unverified", "reason": "no table grid"}
            stats["unverified"] += 1
            continue
        if image is None or not image.exists():
            block.metadata["table_repair"] = {"status": "unverified", "reason": "no table crop"}
            stats["unverified"] += 1
            continue
        if not api_key:
            block.metadata["table_repair"] = {"status": "unverified", "reason": "missing_api_key"}
            stats["unverified"] += 1
            continue
        keys = list(dict.fromkeys(str(f.get("key") or "") for f in flags if str(f.get("key") or "").strip()))
        image_hash = hashlib.sha256(image.read_bytes()).hexdigest()
        block.metadata["visual_sha256"] = block.metadata.get("visual_sha256") or image_hash
        keys_sha = hashlib.sha256("|".join(keys).encode("utf-8")).hexdigest()[:8]
        root = caption_cache_root if caption_cache_root is not None else cache_dir / "visual"
        cache_json = root / "table-repair" / image_hash[:2] / f"{image_hash}-{keys_sha}.json"
        try:
            transcribed = transcribe(image, base_url=base_url, api_key=api_key, model_id=model_id,
                                     row_keys=keys, header=[c.strip() for c in grid[0]], width=len(grid[0]),
                                     cache_json=cache_json, **kwargs)
        except Exception as exc:      # noqa: BLE001 - a verification failure must not fail the whole parse
            print(f"[table-repair] {block.block_id} transcription failed: {exc!r}", flush=True)
            block.metadata["table_repair"] = {"status": "failed", "reason": repr(exc)[:300]}
            stats["failed"] += 1
            continue
        repairs, reasons = plan_repairs(grid, flags, transcribed)
        if not repairs:
            block.metadata["table_repair"] = {"status": "unverified", "reason": "; ".join(reasons)[:500] or "no verified split"}
            stats["unverified"] += 1
            continue
        old_md = grid_to_markdown(grid) or ""
        new_md = grid_to_markdown(expand_glued_rows(grid, repairs)) or old_md
        if old_md and old_md in (block.text or ""):
            block.text = block.text.replace(old_md, new_md, 1)
        else:
            block.text = new_md
        block.table_markdown = block.text
        rows = sorted(repairs)
        n = max(len(p) for plan in repairs.values() for p in plan.values())
        block.metadata["table_repair"] = {"status": "verified", "rows": rows, "n": n,
                                          **({"partial": "; ".join(reasons)[:300]} if reasons else {})}
        for f in block.metadata.get("table_flags") or []:
            if int(f.get("row", -1)) in repairs:
                f["repaired"] = True
        stats["verified"] += 1
        print(f"[table-repair] {block.block_id} rows={rows} expanded x{n}", flush=True)
    return stats
