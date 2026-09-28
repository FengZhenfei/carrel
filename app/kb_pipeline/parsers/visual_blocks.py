from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import math
import re
import unicodedata

from ..measure_units import unit_key
from ..models import ParsedBlock
from ..utils import count_tokens
from ..vision.filter import conservative_decorative_filter, degenerate_image
from ..vision.images import file_hash, image_size  # re-exported for callers/tests
from ..vision.vlm import caption_images_parallel, compose_prompt


# vlm_status values that mean "the VLM actually looked at this picture". The
# visual-vector pass keys off the same set: an image earns a visual vector iff
# it went down the description path (skipped/decorative ones do not).
VLM_SEEN_STATUSES = {"success", "empty", "failed"}


def went_through_vlm(block: ParsedBlock) -> bool:
    return bool(block.visual_ref) and str(block.metadata.get("vlm_status") or "") in VLM_SEEN_STATUSES


def enrich_blocks_with_vlm(
    blocks: list[ParsedBlock],
    *,
    base_url: str,
    api_key: str,
    model_id: str,
    cache_dir: Path,
    concurrency: int = 1,
    prompt: str | None = None,
    filter_decorative: bool = True,
    vlm_options: dict[str, Any] | None = None,
    caption_cache_root: Path | None = None,
    progress_cb: Any = None,
) -> list[ParsedBlock]:
    if not api_key:
        for block in blocks:
            if block.visual_ref:
                block.metadata["vlm_status"] = "skipped"
                block.metadata["vlm_skip_reason"] = "missing_api_key"
        return blocks

    jobs: list[tuple[str, Path, str | None, Path | None]] = []
    job_blocks: dict[str, ParsedBlock] = {}
    for block in blocks:
        if block.block_type not in {"image", "chart", "vision", "slide"}:
            continue
        if not block.visual_ref:
            continue
        image_path = Path(block.visual_ref)
        if not image_path.exists():
            block.metadata["vlm_status"] = "skipped"
            block.metadata["vlm_skip_reason"] = "missing_image"
            continue
        width, height = image_size(image_path)
        if width is None and height is None:
            # Pillow could not open it (garbage bytes, truncated header, or a
            # decompression bomb). Sending the raw bytes anyway used to poison
            # the job: the VLM call fails, the visual-embed pass then fails
            # the same way, and the whole file loops through retries.
            block.metadata["vlm_status"] = "skipped"
            block.metadata["vlm_skip_reason"] = "unreadable_image"
            continue
        degenerate = degenerate_image(width, height)
        if degenerate:
            # A 2-pixel-high rule or a hair-thin divider: the model's processor rejects it outright, and
            # the whole file used to retry repeatedly until it failed. Not counted as a failure and no
            # visual vector; in the body route it is folded into the previous chunk as a decorative
            # image, a standalone image file merely skips the description
            block.metadata["vlm_status"] = "skipped"
            block.metadata["vlm_skip_reason"] = degenerate
            if filter_decorative:
                block.metadata["decorative"] = True
            continue
        if filter_decorative:
            decision = conservative_decorative_filter(
                width=width,
                height=height,
                object_name=image_path.name,
            )
            if decision.decorative_skip:
                block.metadata["vlm_status"] = "skipped"
                block.metadata["vlm_skip_reason"] = decision.decorative_reason
                block.metadata["decorative"] = True      # icon / repeated logo: folded into the previous chunk when chunking, no chunk of its own
                continue
        image_hash = file_hash(image_path)
        block.metadata["visual_sha256"] = image_hash
        if caption_cache_root is not None:
            # Keyed by picture hash + prompt, not by parse location: an edited
            # document (new content_version) or the same figure in another
            # document reuses the caption instead of re-running the VLM. The
            # prompt goes into the name because the effective prompt still
            # varies (per-KB custom instruction; filename hint for standalone
            # image files) and any default-wording change must miss the cache.
            prompt_sha = hashlib.sha256((prompt or compose_prompt("figure")).encode("utf-8")).hexdigest()[:8]
            cache_json = caption_cache_root / image_hash[:2] / f"{image_hash}-{prompt_sha}.caption.json"
        else:
            cache_json = cache_dir / "visual" / f"{block.block_id}-{image_hash[:16]}.json"
        jobs.append((block.block_id, image_path, prompt, cache_json))
        job_blocks[block.block_id] = block

    if not jobs:
        return blocks

    print(f"[vlm] start images={len(jobs)} concurrency={max(1, concurrency)}", flush=True)
    results = caption_images_parallel(
        jobs,
        base_url=base_url,
        api_key=api_key,
        model_id=model_id,
        concurrency=concurrency,
        progress_cb=progress_cb,
        **(vlm_options or {}),
    )
    print(f"[vlm] done images={len(jobs)}", flush=True)
    for block_id, data in results.items():
        block = job_blocks.get(block_id)
        if block is None:
            continue
        summary = str(data.get("summary") or "").strip()
        title = str(data.get("title") or "").strip()
        verbatim = str(data.get("text_verbatim") or "").strip()
        kind = str(data.get("kind") or "").strip()
        confidence = str(data.get("confidence") or "").strip() or "low"
        facts = data.get("facts") if isinstance(data.get("facts"), list) else []
        keywords = data.get("keywords") if isinstance(data.get("keywords"), list) else []
        entities = data.get("entities") if isinstance(data.get("entities"), list) else []
        # In-image text (caption, transcription, chart / formula reading) is evidence, the model's FACTS
        # are interpretation: when the same metric's values disagree the text wins, and the conflict
        # leaves a trace
        own_text = (block.text or "").strip()
        trusted = "\n".join(x for x in (str(block.caption or ""), verbatim, own_text if kind in KEEP_OWN_TEXT_KINDS else "") if x)
        facts, summary, value_conflicts = reconcile_visual_facts([str(x) for x in facts], trusted, summary)
        if value_conflicts:
            print(f"[vlm] {block.block_id}: {len(value_conflicts)} estimated value(s) disagree with the in-image text; using the text", flush=True)
        # The VLM title becomes the block caption when the parser found none;
        # then it is emitted once as "CAPTION:" by the chunker and must not be
        # repeated as a TITLE line here. A block that already has a caption
        # (MinerU figure caption) keeps the VLM title in the text as new info.
        title_becomes_caption = bool(title) and not block.caption
        if title_becomes_caption:
            block.caption = title
        # text_verbatim is deliberately kept out of the embedded text: exact
        # on-image strings are what the FTS index is for, and pasting a whole
        # UI's worth of labels into the chunk would dilute its vector.
        parts = [
            f"TITLE: {title}" if title and not title_becomes_caption else "",
            f"VISUAL SUMMARY: {summary}" if summary else "",
            "FACTS: " + "；".join(str(x) for x in facts) if facts else "",
            "ENTITIES: " + "，".join(str(x) for x in entities) if entities else "",
            "KEYWORDS: " + "，".join(str(x) for x in keywords) if keywords else "",
        ]
        visual_text = "\n".join(part for part in parts if part).strip()
        own = (block.text or "").strip()
        got_content = bool(summary or verbatim or facts or entities)
        # The image's own text (MinerU's reading of the image) is trustworthy only for charts /
        # architecture diagrams / formulas; on screenshots and photos it is mostly hallucinated OCR
        # (repetition, fabrication), so it is dropped whenever the VLM returned something. The VLM
        # paragraphs go first: when an over-long block is truncated, the read text is what falls off
        keep_own = bool(own) and (not got_content or kind in KEEP_OWN_TEXT_KINDS)
        pieces = ([visual_text] if visual_text else []) + ([own] if keep_own else [])
        if pieces:
            block.text = "\n".join(pieces)
        if own and not keep_own:
            block.metadata["own_text_dropped"] = True
        block.visual_summary = summary or visual_text or block.visual_summary
        if "+vlm" not in block.parser:
            block.parser = f"{block.parser}+vlm"
        got_content = bool(summary or verbatim or facts or entities)
        # Decorative image: the model says it is decorative; or a photo with no text, no facts, no
        # entities and only a one-line stock description; or a description that opens with a QR code /
        # barcode / icon / signature and the like. No chunk of its own when chunking, only a one-line
        # description left in the previous body chunk (see chunker.blocks_to_chunks). Photos with a
        # substantive description (a full-page ultrasound image in a check-up report) used to be folded
        # away as decorative too, losing both the page number and the visual vector (2026-09-07 health
        # knowledge base audit)
        decorative = is_decorative_result(kind, summary, facts=facts, entities=entities, verbatim=verbatim,
                                          flagged=bool(data.get("decorative")))
        block.metadata.update(
            {
                "visual_entities": entities,
                "visual_facts": facts,
                "visual_keywords": keywords,
                "visual_text": verbatim,
                "visual_kind": kind,
                "visual_confidence": confidence,
                "visual_value_conflicts": value_conflicts,
                "decorative": decorative,
                "vlm_status": "success" if got_content else "empty",
                "vlm_model": model_id,
            }
        )
        if data.get("error"):
            block.metadata["vlm_status"] = "failed"
            block.metadata["vlm_error"] = data.get("error")
    return blocks


# Only these subjects, named right at the start of the description, count; English words like logo /
# icon are not included -- a "customer logo wall" is a proper photo with entities
_DECOR_SUMMARY_RE = re.compile(r"二维码|条形码|条码|barcode|qr\s?code|图标|签名|印章|水印", re.IGNORECASE)
_PLAIN_PHOTO_MAX_CHARS = 30      # "A hand holding a test tube." at 17 chars is a stock phrase; "Six ultrasound images showing ... morphological changes." at 37 chars is content


# "label + value" pairs: in a segment of text, the run of characters before the first number is the
# label (at least two CJK characters or three letters) and the number is its value. Extracted once
# from the in-image text (caption, transcription, MinerU reading) and once from the model's FACTS /
# summary; when the two sides give different numbers for the same metric, the estimate has overridden
# the text
_LABEL_NUM_RE = re.compile(r"(?P<label>[^\d\n；;。.:：,，|]*?[\u4e00-\u9fffA-Za-z][^\d\n；;。:：,，|]*?)\s*[:：=]?\s*(?P<num>[-+]?\d+(?:\.\d+)?)")
_NUM_ONLY_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_SEG_SPLIT_RE = re.compile(r"[\n；;。|,，]+")
_LABEL_CLEAN_RE = re.compile(r"[\s_\-·:：,，()（）\[\]【】\"'“”]+")
# Reference values / ranges / means are not "the reading of this object in this image" and take no
# part in the comparison (generic cross-domain words, not a rule for one particular knowledge base)
_REF_RE = re.compile(r"参考|范围|正常值|标准|平均|均值|上限|下限|阈值|ref|range|avg|mean|limit|threshold|normal", re.I)
# Generic words in labels: what remains after stripping them is "which metric"; the model often calls
# a "composite risk index" the "current risk value"
_GENERIC_RE = re.compile(r"当前|风险|指数|数值|数量|水平|比例|含量|结果|读数|等级|评分|分数|显示|大约|约|为|值|"
                         r"score|value|index|level|reading|current|risk|rate|count|number|total", re.I)


# The unit directly after the number (mV, %, °C, mmol/L ...): when both sides carry units and they
# differ, the numbers cannot be compared directly -- 0.1 V and 100 mV are two spellings of one value
_UNIT_AFTER_RE = re.compile(r"[ \t]*(?P<unit>[%°℃℉µμΩ℧A-Za-z][A-Za-z0-9%°℃℉µμΩ℧/·^]*)")


def _unit_norm(unit: str) -> str:
    # The same normalised key as the graph side, case preserved (final review F04): "1 MW" and
    # "1000000000 mW" have different units, so neither compare nor rewrite
    return unit_key(unicodedata.normalize("NFKC", str(unit or "")).replace("μ", "µ"))


_ID_NUM_RE = re.compile(r"^[-+]?0\d")       # 0001 / 007: leading zeros mean an id / code, not a reading
TRUSTED_SEGMENT_MAX_NUMBERS = 3            # more numbers than this in one in-image text segment means a table row / a whole flattened table; which number belongs to which label cannot be told


def _is_flat_header(label: str) -> bool:
    """Three or more whitespace-separated Chinese words: a header row of a flattened table / several
    cells run together, not the label of one metric. Chinese phrases use no internal spaces, and
    English labels (short circuit current) are unaffected. 2026-09-13 Codex F06: "product id, product
    name, product category, sale price, purchase price, registration date 0001" was taken as one
    "label-value" pair, and containment matching then rewrote both sale price / purchase price to 1."""
    cjk_words = [w for w in re.split(r"\s+", label) if any("\u4e00" <= ch <= "\u9fff" for ch in w)]
    return len(cjk_words) >= 3


# When the label ends with one of these, the number after it is part of a name / citation / id, not a
# reading: MACD(8, reference[3, Top 8, Figure 3, version 2, No. 1 (Codex 2026-09-14 R01)
_NAME_NUMBER_TAIL_RE = re.compile(r"(?:[(（\[【]|\b(?:top|no\.?|#|fig\.?|figure|table|ref\.?|version|ver\.?|rev\.?|page|p\.)|"
                                  r"[第图表注版期章节条款]|型号|编号|序号|排名)\s*$", re.IGNORECASE)
_CLOSERS = "]】)）"


def _is_name_number(label: str, seg: str, num_end: int) -> bool:
    """The "label + number" is really a name / citation / id: the label ends with an opening
    parenthesis, a square bracket, Top or a figure/table numbering word, or the number is directly
    followed by a closing parenthesis / bracket (the 1 in "five-year colorectal cancer survival
    rate[1-2]" is a reference number, and it rewrote all four percentages to 1%)."""
    if _NAME_NUMBER_TAIL_RE.search(label):
        return True
    after = seg[num_end:num_end + 1]
    return bool(after) and after in _CLOSERS


def _fully_paired(seg: str) -> bool:
    """A segment of in-image text is trusted only when it splits completely into "label -> number"
    pairs: every number has its own label in front of it (heart rate 65 blood pressure 120). A segment
    that does not split fully is a table row / header plus data (Product Price Stock 1 20 50) or a
    citation range ([1-2]); which number belongs to which label cannot be told, so the whole segment
    is rejected."""
    numbers = len(_NUM_ONLY_RE.findall(seg))
    pairs = sum(1 for _ in _LABEL_NUM_RE.finditer(seg))
    return numbers > 0 and pairs == numbers


def _collect_pair(body: str, start: int, end: int, single_only: bool, out: list[dict[str, Any]],
                  max_numbers: int | None = None) -> None:
    seg = body[start:end]
    if single_only and len(_NUM_ONLY_RE.findall(seg)) != 1:
        return
    if max_numbers is not None and (len(_NUM_ONLY_RE.findall(seg)) > max_numbers or not _fully_paired(seg)):
        return          # too many numbers in this in-image text segment, or no one-to-one "label -> number" split: cannot tell which number belongs to which label, so not a trusted reading
    m = _LABEL_NUM_RE.search(seg)
    if not m:
        return
    label = m.group("label").strip(" ,，:：=-")
    if _is_flat_header(label) or _ID_NUM_RE.match(m.group("num")) or _is_name_number(label, seg, m.end("num")):
        return
    key = _LABEL_CLEAN_RE.sub("", label).casefold()
    cjk = sum(1 for ch in key if "\u4e00" <= ch <= "\u9fff")
    if cjk < 2 and len(key) < 3:
        return
    try:
        num = float(m.group("num"))
    except ValueError:
        return
    unit_m = _UNIT_AFTER_RE.match(seg, m.end("num"))
    out.append({"key": key, "label": label, "num": num, "unit": (unit_m.group("unit") if unit_m else "") or "",
                "start": start + m.start("num"), "end": start + m.end("num")})


def _label_num_spans(text: str, *, single_only: bool = False, max_numbers: int | None = None) -> list[dict[str, Any]]:
    """Take the first "label value" pair of every segment (split on newline / semicolon / full stop /
    comma): key is the normalised label, label the original label, num, unit (the unit directly after
    the number, empty if none), start / end (the position of this number in the whole string -- a
    rewrite touches only this number in this segment, it does not look for the first equal number in
    the whole sentence, Codex re-review N01). With single_only=True only segments containing exactly
    one number are collected (model-side sentences: two numbers as in "reference range 3.1-5.2" are
    not a reading); when max_numbers is given, segments with more numbers are skipped (in-image text
    side: a table row is not one reading). Flattened headers taken as labels and leading-zero ids
    taken as readings are never collected."""
    out: list[dict[str, Any]] = []
    body = text or ""
    pos = 0
    for gap in _SEG_SPLIT_RE.finditer(body):
        _collect_pair(body, pos, gap.start(), single_only, out, max_numbers)
        pos = gap.end()
    _collect_pair(body, pos, len(body), single_only, out, max_numbers)
    return out


def _label_num_pairs(text: str, *, single_only: bool = False) -> list[tuple[str, str, float]]:
    return [(p["key"], p["label"], p["num"]) for p in _label_num_spans(text, single_only=single_only)]


def _cjk_bigrams(key: str) -> set[str]:
    return {key[i:i + 2] for i in range(len(key) - 1) if all("\u4e00" <= ch <= "\u9fff" for ch in key[i:i + 2])}


def _match_trusted(key: str, trusted: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Which "label value" pair in the image a model sentence's label corresponds to. When both sides
    still have content words after stripping generic ones: equal content words win; otherwise one side
    contains the other, but only a unique match counts ("current" is contained in both "input current"
    and "output current", so it cannot be resolved and is left alone; "leakage current" with an exact
    match is not stolen by "current" -- Codex re-review N01). A model label made entirely of generic
    words ("the current risk value is") is matched by shared CJK bigrams, again only on a unique match,
    or when the image holds a single value."""
    ks = _GENERIC_RE.sub("", key)
    if ks:
        exact = [t for t in trusted if _GENERIC_RE.sub("", t["key"]) == ks]
        if exact:
            # Same label, several values ("voltage 3.3 V", "voltage 5 V" are readings of two objects):
            # cannot tell which one, so leave it alone; repeated equal values are safe to use (final
            # review F05)
            return exact[0] if len({round(t["num"], 9) for t in exact}) == 1 else None
        contained: list[dict[str, Any]] = []
        for t in trusted:
            ts = _GENERIC_RE.sub("", t["key"])
            if not ts:
                continue
            shorter, longer = sorted((ts, ks), key=len)
            if len(shorter) >= 2 and shorter in longer:
                contained.append(t)
        return contained[0] if len(contained) == 1 else None
    grams = _cjk_bigrams(key)
    cands = [t for t in trusted if grams & _cjk_bigrams(t["key"])]
    if len(cands) == 1:
        return cands[0]
    return trusted[0] if len(trusted) == 1 else None


def reconcile_visual_facts(facts: list[str], trusted_text: str, summary: str = "") -> tuple[list[str], str, list[dict[str, Any]]]:
    """When a value in the model's FACTS / summary disagrees with the value of the same metric in the
    in-image text (caption / transcription / reading), rewrite it to the text's value and record the
    conflict (in-image label, text value, estimated value, the model's wording). The caption says 28,
    the summary and FACTS say "the current risk value is 25", and the final fact took 25 -- that is
    the path handled here. Only sentences with exactly one number that is not a reference value /
    range / mean are compared; what counts as the same metric is in _match_trusted. Anything
    uncertain is left untouched: several candidate matches or differing units on the two sides stay
    as they are (Codex re-review N01: deterministic post-processing must not introduce errors of its
    own). Returns (facts, summary, conflicts)."""
    trusted = [t for t in _label_num_spans(trusted_text, max_numbers=TRUSTED_SEGMENT_MAX_NUMBERS) if not _REF_RE.search(t["label"])]
    if not trusted or not (facts or summary):
        return list(facts), summary, []
    conflicts: list[dict[str, Any]] = []

    def fix(text: str) -> str:
        edits: list[tuple[int, int, str]] = []
        for pair in _label_num_spans(text, single_only=True):
            if _REF_RE.search(pair["label"]):
                continue
            hit = _match_trusted(pair["key"], trusted)
            if hit is None or math.isclose(hit["num"], pair["num"], rel_tol=1e-9, abs_tol=1e-9):
                continue
            if pair["unit"] and hit["unit"] and _unit_norm(pair["unit"]) != _unit_norm(hit["unit"]):
                continue                      # numbers with different units cannot be compared directly, let alone rewriting the number while keeping the unit
            edits.append((pair["start"], pair["end"], f"{hit['num']:g}"))
            conflicts.append({"label": hit["label"], "text_value": f"{hit['num']:g}", "model_value": f"{pair['num']:g}",
                              "model_label": pair["label"]})
        fixed = text
        for start, end, new in sorted(edits, reverse=True):
            fixed = fixed[:start] + new + fixed[end:]
        return fixed

    out_facts = [fix(f) for f in facts]
    out_summary = fix(summary) if summary else summary
    return out_facts, out_summary, conflicts


def is_decorative_result(kind: str, summary: str, *, facts: list | None, entities: list | None, verbatim: str,
                         flagged: bool) -> bool:
    """VLM result -> whether to fold the image into the previous body chunk as a decorative image."""
    if flagged:
        return True
    head = str(summary or "").strip()[:24]
    if _DECOR_SUMMARY_RE.search(head):
        return True
    return (str(kind) == "photo" and not (facts or entities or (verbatim or "").strip())
            and len(str(summary or "").strip()) < _PLAIN_PHOTO_MAX_CHARS)


def vlm_failure_summary(blocks: list[ParsedBlock]) -> dict[str, int]:
    """Count how many of the blocks that went through the VLM failed. The parse flow uses this to decide
    whether to retry the whole task: a document processed while the VLM service was down used to be
    stored as "parsed successfully" with no description on any image, and since content_version did
    not change the scan judged it unchanged, so it would never be filled in automatically."""
    total = failed = 0
    for block in blocks:
        if not went_through_vlm(block):
            continue
        total += 1
        if str((block.metadata or {}).get("vlm_status") or "") == "failed":
            failed += 1
    return {"visual_blocks": total, "vlm_failed": failed}


# For these image kinds the text MinerU read out (nodes, data points, formulas) is worth keeping
# alongside the VLM result
KEEP_OWN_TEXT_KINDS = frozenset({"diagram", "chart", "table", "formula"})
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
VISUAL_TEXT_MAX_TOKENS = 400
VISUAL_LINE_REPEAT_LIMIT = 2


_LINE_CUT_MIN_TOKENS = 40       # do not cut a line when the remaining budget is below this many tokens: half a sentence is worse than none
_FACT_SEP_RE = re.compile(r"[；;]")


def _cut_line_to_tokens(line: str, max_tokens: int) -> str:
    """Cut a line down to at most max_tokens tokens: shorten proportionally and re-check, which
    converges in a few rounds; a line separated by ";" such as FACTS is cut at a separator where
    possible so no half fact is left."""
    cut = line
    for _ in range(8):
        n = count_tokens(cut)
        if n <= max_tokens:
            break
        cut = cut[: max(1, int(len(cut) * max_tokens / n * 0.95))]
    if cut != line:
        seps = [m.end() for m in _FACT_SEP_RE.finditer(cut)]
        if seps and seps[-1] >= len(cut) // 2:
            cut = cut[: seps[-1]]
    return cut.rstrip()


def tidy_visual_text(text: str, *, kind: str = "", max_tokens: int = VISUAL_TEXT_MAX_TOKENS) -> tuple[str, dict[str, int]]:
    """Tidy the text an image block carries on its own (MinerU's reading of the image):
      - unless it is an architecture diagram / flowchart, drop mermaid and other code fences -- a
        screenshot or results table drawn as a graph LR chain is fabricated;
      - drop a line that repeats more than VISUAL_LINE_REPEAT_LIMIT times (repetition: twenty lines
        of "Home", dozens of lines of "Password");
      - truncate line by line past max_tokens.
    Returns (tidied text, stats)."""
    stats = {"fences_dropped": 0, "lines_dropped": 0, "tokens_cut": 0}
    if not (text or "").strip():
        return text or "", stats
    out = text
    if kind not in {"diagram", "chart"}:
        out, n = _FENCE_RE.subn("", out)
        stats["fences_dropped"] = n
    seen: dict[str, int] = {}
    kept: list[str] = []
    for line in out.splitlines():
        key = line.strip()
        if key:
            seen[key] = seen.get(key, 0) + 1
            if seen[key] > VISUAL_LINE_REPEAT_LIMIT:
                stats["lines_dropped"] += 1
                continue
        kept.append(line)
    out = "\n".join(kept).strip()
    if count_tokens(out) > max_tokens:
        lines = out.splitlines()
        budget: list[str] = []
        used = 0
        for line in lines:
            t = count_tokens(line) + 1
            if used + t > max_tokens:
                # The line that does not fit is cut to the remaining budget rather than dropped whole:
                # a FACTS line that overflowed used to vanish entirely, leaving a dense screenshot with
                # only a one-line summary; and a first line already over budget emptied the whole block,
                # after which chunking put the uncapped visual_summary back verbatim (2026-09-10 product
                # material, one screenshot of 3,157 tokens). If the remaining budget is too small, do
                # not cut, to avoid leaving a tail with neither head nor end.
                remaining = max_tokens - used
                if not budget or remaining >= _LINE_CUT_MIN_TOKENS:
                    budget.append(_cut_line_to_tokens(line, max(remaining, 1) if budget else max_tokens))
                break
            budget.append(line)
            used += t
        cut = "\n".join(budget).strip()
        stats["tokens_cut"] = count_tokens(out) - count_tokens(cut)
        out = cut
    return out, stats


def tidy_visual_blocks(blocks: list[ParsedBlock], *, max_tokens: int = VISUAL_TEXT_MAX_TOKENS) -> int:
    """Apply tidy_visual_text to the text of image / chart blocks. For a block that went through the
    VLM, text already holds "VISUAL SUMMARY / FACTS / ..." plus its own reading (see
    enrich_blocks_with_vlm), so the budget governs the whole passage; the caption lives in caption and
    is unaffected. Returns the number of blocks changed."""
    changed = 0
    for block in blocks:
        if block.block_type not in {"image", "chart", "vision"} or not (block.text or "").strip():
            continue
        kind = str(block.metadata.get("visual_kind") or "")
        cleaned, stats = tidy_visual_text(block.text, kind=kind, max_tokens=max_tokens)
        if cleaned != block.text:
            block.text = cleaned
            block.metadata["visual_text_tidy"] = stats
            changed += 1
    if changed:
        print(f"[vlm] tidied image text on {changed} block(s)", flush=True)
    return changed


def ensure_visual_blocks_have_text(blocks: list[ParsedBlock], *, doc_name: str) -> int:
    """A picture the VLM looked at must still yield a chunk even when the
    model came back empty or failed: text_for_block() of a bare block is
    empty, so the block would silently vanish from the index -- taking the
    visual vector with it. The filename-based title is weak embedding text,
    but it keeps the point alive; the pixels carry the real signal through
    the visual vector. (parsers/image_file.py guarantees the same for
    standalone images via title=stem.)"""
    patched = 0
    for block in blocks:
        if not went_through_vlm(block):
            continue
        if (block.text or "").strip() or (block.caption or "").strip():
            continue
        if (block.visual_summary or "").strip() or (block.title or "").strip():
            continue
        location = f" p{block.page_idx}" if block.page_idx is not None else ""
        block.title = f"{doc_name}{location} {block.block_type}"
        patched += 1
    if patched:
        print(f"[vlm] fallback title added for {patched} caption-less visual block(s)", flush=True)
    return patched


__all__ = [
    "enrich_blocks_with_vlm",
    "ensure_visual_blocks_have_text",
    "tidy_visual_blocks",
    "tidy_visual_text",
    "vlm_failure_summary",
    "went_through_vlm",
    "VLM_SEEN_STATUSES",
    "file_hash",
    "image_size",
]
