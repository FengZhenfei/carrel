"""Body text cleaning, windowing, position strings and overlap detection for the reranker and the
evidence. All pure functions; no external service is touched."""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable

from kb_pipeline.utils import count_tokens

# Structural prefixes at the start of chunk text: the rerank looks at the body only, and these labels
# carry no meaning (Q06); the content of HEADER / SHEET / ROWS stays, only the label is removed
_PREFIX_RE = re.compile(r"^(?:TITLE|CAPTION|EQUATION|VISUAL SUMMARY|FACTS|VISUAL TEXT|SHEET|ROWS|HEADER|Title|Caption|Note)\s*[:：]\s*", re.MULTILINE)
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}[-:| ]*$", re.MULTILINE)
_MD_MARK_RE = re.compile(r"(\*\*|__|^#{1,6}\s+|^>\s?|`)", re.MULTILINE)
_WS_RE = re.compile(r"[ \t]+")
_NORM_RE = re.compile(r"[\s\W_]+", re.UNICODE)


def clean_for_rerank(text: str) -> str:
    """Strip structural labels, turn pipe tables into comma-separated lines, drop separator lines and
    markdown marks; no field is repeated and nothing is truncated."""
    body = str(text or "")
    body = _TABLE_SEP_RE.sub("", body)
    body = _PREFIX_RE.sub("", body)
    lines = []
    for line in body.splitlines():
        s = line.strip()
        if not s:
            continue
        if "|" in s:
            cells = [c.strip() for c in s.strip("|").split("|")]
            s = ", ".join(c for c in cells if c)
        s = _MD_MARK_RE.sub("", s)
        lines.append(_WS_RE.sub(" ", s).strip())
    return "\n".join(l for l in lines if l)


def head_line(payload: dict[str, Any]) -> str:
    """At most one head line for the rerank input: "filename › section path"."""
    name = str(payload.get("filename") or payload.get("rel_path") or "").strip()
    section = " / ".join(str(s) for s in (payload.get("section_path") or []) if str(s).strip())
    if name and section:
        return f"{name} › {section}"
    return name or section


def _split_long_piece(piece: str, window_tokens: int) -> list[str]:
    """Fallback split of an over-long piece without sentence breaks by token (Codex S07: 1400 tokens of
    unpunctuated text used to be a single window). Binary search over characters, each part <=
    window_tokens."""
    if count_tokens(piece) <= window_tokens:
        return [piece]
    out: list[str] = []
    rest = piece
    while rest and count_tokens(rest) > window_tokens:
        lo, hi = 1, len(rest)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if count_tokens(rest[:mid]) <= window_tokens:
                lo = mid
            else:
                hi = mid - 1
        cut = max(1, lo)
        out.append(rest[:cut])
        rest = rest[cut:]
    if rest:
        out.append(rest)
    return out


def truncate_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to at most max_tokens (binary search over characters), for the hard budget
    boundary."""
    body = str(text or "")
    if max_tokens <= 0:
        return ""
    if count_tokens(body) <= max_tokens:
        return body
    lo, hi = 1, len(body)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(body[:mid]) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return body[:max(1, lo)]


def windows(text: str, *, window_tokens: int = 480, overlap_tokens: int = 32) -> list[str]:
    """Window a long chunk by sentence / line, each window about window_tokens with adjacent windows
    overlapping by about overlap_tokens; a short chunk is a single window."""
    body = str(text or "").strip()
    if not body:
        return [""]
    if count_tokens(body) <= window_tokens:
        return [body]
    pieces = [p for p in re.split(r"(?<=[。!?！？；;\n])", body) if p.strip()]
    pieces = [sub for p in pieces for sub in _split_long_piece(p, max(16, window_tokens - overlap_tokens))]
    out: list[str] = []
    cur: list[str] = []
    cur_tokens = 0
    for piece in pieces:
        t = count_tokens(piece)
        if cur and cur_tokens + t > window_tokens:
            out.append("".join(cur).strip())
            tail: list[str] = []
            tail_tokens = 0
            for item in reversed(cur):
                it = count_tokens(item)
                if tail_tokens + it > overlap_tokens:
                    break                       # the overlap carries only whole pieces that fit; if not even one fits, none is carried (otherwise one long piece would double the next window)
                tail.insert(0, item)
                tail_tokens += it
            cur, cur_tokens = tail, tail_tokens
        cur.append(piece)
        cur_tokens += t
    if cur:
        out.append("".join(cur).strip())
    return [w for w in out if w] or [body]


def position(payload: dict[str, Any]) -> str:
    """Position string: "page 46 · 3. AC Characteristics / 3.2 Timing"; slides and sheets are written
    by their type. The payload's page_idx / slide_idx are already 1-based page numbers
    (parsers.common.page_idx adds 1 to MinerU's 0-based index) and are shown as-is here."""
    parts: list[str] = []
    page = payload.get("page_idx")
    slide = payload.get("slide_idx")
    sheet = payload.get("sheet_name")
    if slide is not None and str(slide) != "":
        parts.append(f"slide {int(slide)}")
    elif sheet:
        rs, re_ = payload.get("row_start"), payload.get("row_end")
        parts.append(f"sheet {sheet}" + (f" rows {rs}–{re_}" if rs is not None and re_ is not None else ""))
    elif page is not None and str(page) != "":
        try:
            parts.append(f"page {int(page)}")
        except (TypeError, ValueError):
            pass
    section = " / ".join(str(s) for s in (payload.get("section_path") or []) if str(s).strip())
    if section:
        parts.append(section)
    return " · ".join(parts)


_NUM_SEP_RE = re.compile(r"(?<=\d)[.,](?=\d)")
_NUM_SEP_MARK = "qqdotqq"          # letters as placeholder: \W would strip private-use characters as well
_NUM_SIGN_RE = re.compile(r"(?<![A-Za-z0-9.])[-−+](?=\d)")     # the hyphen in a model number (ZK7C-4021) is not a sign; a minus after CJK text is
_NUM_SIGN_MARK = {"-": "qqnegqq", "−": "qqnegqq", "+": "qqposqq"}
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _norm(text: str) -> str:
    """Normalisation for dedupe: NFKC, casefold, whitespace and punctuation removed, but the decimal
    point / thousands separator inside a number and the sign before it are kept ("3.3 V" vs "33 V"
    and "40 degrees" vs "-40 degrees" are each two different facts; reproduced once each in Codex S01
    / R2)."""
    body = unicodedata.normalize("NFKC", str(text or ""))
    body = _NUM_SEP_RE.sub(_NUM_SEP_MARK, body)
    body = _NUM_SIGN_RE.sub(lambda m: _NUM_SIGN_MARK[m.group(0)], body)
    body = _NORM_RE.sub("", body).casefold()
    return body.replace(_NUM_SEP_MARK, ".").replace("qqnegqq", "-").replace("qqposqq", "+")


def numbers_of(text: str) -> list[str]:
    """Numbers in the normalised text (with sign and decimal point), in order of appearance; two chunks
    whose numbers disagree are not the same passage."""
    return _NUMBER_RE.findall(_norm(text))


def same_facts(a: str, b: str) -> bool:
    """An approximate overlap must also pass this check to count as a duplicate: the number multisets
    agree (in a containment case, every number of the shorter side occurs in the longer one)."""
    na, nb = numbers_of(a), numbers_of(b)
    if not na and not nb:
        return True
    short, long_ = (na, nb) if len(na) <= len(nb) else (nb, na)
    pool = list(long_)
    for x in short:
        if x in pool:
            pool.remove(x)
        else:
            return False
    return True


def adjacent(a: dict[str, Any], b: dict[str, Any], *, span: int = 2) -> bool:
    """Real overlap only comes from adjacent chunks (chunking overlap of 80 tokens): when both have an
    index they must differ by <= span; without indexes there is no restriction."""
    ia, ib = a.get("chunk_index"), b.get("chunk_index")
    if ia is None or ib is None:
        return True
    try:
        return abs(int(ia) - int(ib)) <= span
    except (TypeError, ValueError):
        return True


def _shingles(text: str, n: int = 3) -> set[str]:
    s = _norm(text)
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))} if len(s) >= n else ({s} if s else set())


def overlap_ratio(a: str, b: str) -> float:
    """Overlap coefficient of two texts: trigram intersection / trigram count of the shorter side; a
    substring containment after normalisation counts as 1."""
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return 0.0
    if na in nb or nb in na:
        return 1.0
    sa, sb = _shingles(na), _shingles(nb)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, min(len(sa), len(sb)))


def scope_of(row: dict[str, Any]) -> tuple[str, str, str]:
    """Dedupe scope: same knowledge base, same document, same version. Similar passages in different
    documents are two pieces of evidence and must not cancel each other out."""
    return (str(row.get("kb_id") or ""), str(row.get("doc_id") or ""), str(row.get("content_version") or ""))


def dedupe_overlaps(rows: Iterable[dict[str, Any]], *, threshold: float = 0.85, key: str = "text",
                    scoped: bool = True) -> tuple[list[dict[str, Any]], int]:
    """Of two chunks overlapping at or above threshold, drop the shorter and keep the longer (with a
    chunking overlap of 80 tokens, double hits are the norm, Q07). Compared only between adjacent
    chunks of the same document and version (scoped); chunks whose numbers (with sign and decimal
    point) disagree are not duplicates; cross-document and non-adjacent chunks are always kept. Input
    order is preserved; returns (remaining, number dropped)."""
    kept: list[dict[str, Any]] = []
    dropped = 0
    for row in rows:
        text = str(row.get(key) or "")
        replaced = False
        for i, other in enumerate(kept):
            if scoped and (scope_of(row) != scope_of(other) or not adjacent(row, other)):
                continue
            if overlap_ratio(text, str(other.get(key) or "")) >= threshold and same_facts(text, str(other.get(key) or "")):
                if len(text) > len(str(other.get(key) or "")):
                    kept[i] = row          # keep the longer one
                dropped += 1
                replaced = True
                break
        if not replaced:
            kept.append(row)
    return kept, dropped


_LATIN_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]*")
_CJK_RE = re.compile(r"[㐀-鿿]+")


def token_set(text: str) -> set[str]:
    """Token set: Latin words (casefolded, >= 2 characters) + Chinese bigrams; used for the MMR Jaccard
    (Q16)."""
    body = str(text or "")
    out = {w.casefold() for w in _LATIN_WORD_RE.findall(body) if len(w) >= 2}
    for run in _CJK_RE.findall(body):
        if len(run) == 1:
            out.add(run)
        out.update(run[i:i + 2] for i in range(len(run) - 1))
    return out


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


# Generic signals of boilerplate chunks (Q18): the section name / title is a table of contents,
# revision history, copyright page, disclaimer and the like; no knowledge-base-specific words here
_BOILERPLATE_PATTERNS = (
    re.compile(r"^(目\s*录|contents|table of contents|toc)\b", re.IGNORECASE),
    re.compile(r"(修订记录|修订历史|版本历史|版本记录|变更记录|revision history|change ?log|document history)", re.IGNORECASE),
    re.compile(r"(版权所有|版权声明|copyright|all rights reserved|免责声明|法律声明|disclaimer|legal notice)", re.IGNORECASE),
)


def is_boilerplate(payload: dict[str, Any]) -> bool:
    segments = [str(payload.get("title") or "")] + [str(s) for s in (payload.get("section_path") or [])]
    for seg in segments:
        s = unicodedata.normalize("NFKC", seg).strip()
        if not s:
            continue
        if any(p.search(s) for p in _BOILERPLATE_PATTERNS):
            return True
    return False


_STRUCT_LINE_RE = re.compile(r"^\s*(SHEET|ROWS|HEADER|TITLE|CAPTION)\s*[:：]", re.IGNORECASE)
_CJK_STOP = {"的", "了", "是", "在", "和", "与", "或", "及", "等", "中", "有", "为", "对", "把", "被", "从", "到", "这", "那", "什么", "怎么", "哪些", "如何", "多少", "分别", "请问", "一下"}


def body_token_set(text: str) -> set[str]:
    """Token set for MMR: the SHEET / ROWS / HEADER / TITLE / CAPTION structural lines at the start of
    the chunk are removed before tokenising -- every chunk of one table carries the same header line,
    and without removing it their mutual Jaccard is high and MMR would suppress the correct row as a
    duplicate."""
    lines = [ln for ln in str(text or "").splitlines() if not _STRUCT_LINE_RE.match(ln)]
    return token_set("\n".join(lines) if lines else str(text or ""))


def question_coverage(question: str, text: str) -> float:
    """What fraction of the question's tokens land in this chunk (0-1): Latin words and Chinese bigrams
    with function words removed; a token-level tie-breaker for the final score."""
    q = {t for t in token_set(question) if t not in _CJK_STOP and len(t) >= 2}
    if not q:
        return 0.0
    body = token_set(text)
    return round(len(q & body) / len(q), 4)
