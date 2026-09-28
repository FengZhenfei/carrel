"""Extraction units: assembled directly from our own chunks, instead of letting the upstream re-split the
text with a 1200-token sliding window.

A unit = graph_unit_chunks consecutive raw chunks of the same document (default 3; 1 = no merging).
The chunks come from the main collection's Qdrant payloads, selected by the chunk ledger frozen at the
moment the graph build started (graph_build_chunks). The whole-sentence overlap between adjacent chunks
(the chunker's overlap) is removed while joining, so entities in the overlap region are not extracted
twice with an inflated frequency. On a section change the current unit is only cut once it is at least
half full: a datasheet has one tiny section after another, each a few dozen tokens, and giving every
section its own unit makes the extractor "glean" a pile of fragment entities out of a hundred-odd tokens.

Table blocks are kept whole (as long as they fit within twice the budget): table rows mean nothing
away from their header, and since the chunker repeats the header in every table chunk, splitting them
would extract the same row twice.

unit_id is content-addressed (kb_id + doc_id + unit text): a unit whose text has not changed hits
graph_extractions in the next graph build without a single LLM call -- that is the basis of
"incremental" builds.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..parsers.table_check import unrepaired_glued_values
from ..utils import count_tokens

DEFAULT_UNIT_CHUNKS = 3
MIN_OVERLAP_CHARS = 12
TABLE_BLOCK_TYPES = {"table"}

# ── Unit kinds: body / listing / boilerplate ──────────────────────────────
# The "device -> document / template / parameter name" relations extracted from boilerplate pages (table of
# contents, revision history, legal notices) and listing pages (ordering-code tables, ball maps) pushed the
# document-history page to the top of almost every question on kb_003. Here we only label by structural
# signals; the model gives a second verdict at extraction time (extract.py), and merge.combine_unit_kind holds
# the combination rule. Labelling deletes no data: relations and entities carry a boilerplate flag and recall
# skips them by default.
# 2026-09-07 added conclusion: the document's own conclusion / summary sections (abnormal-result summaries,
# conclusions, recommendations, Summary). It is a kind of body text (goes into recall and fact extraction);
# it is labelled separately so the view layer can use it as the source of "conclusion facts". The boilerplate
# structural rules also gained the layouts of Chinese reports (TOC pages without dot leaders, reading guides,
# cover letters, QR-code / app promotion pages) -- the health KB's 215 units had only 3 recognised as boilerplate.
UNIT_KINDS = ("body", "conclusion", "listing", "boilerplate")
_TOC_RE = re.compile(r"(?:\.{3,}|…{2,}|(?:\s\.){3,})\s*\d{1,4}\b")
# TOC lines without dot leaders: "1 Key abnormal findings and follow-up advice 04", "3 Checkup results 07 Details"
_TOC_LINE_RE = re.compile(r"^\s*\d{1,2}(?:\.\d{1,2})?\s+\S.{2,80}?\s\d{1,3}(?:\s+\S.{0,60})?\s*$")
_GREETING_RE = re.compile(r"尊敬的[^\n]{1,30}(?:先生|女士|客户|用户)[,,:：]?\s*您好|^\s*dear\s+[a-z .]{2,40},", re.IGNORECASE | re.MULTILINE)
_PROMO_RE = re.compile(r"二维码|扫码|扫一扫|下载[^\n]{0,8}app|app\s*store|公众号|小程序|qr\s*code|scan\s+the\s+code", re.IGNORECASE)
_READING_GUIDE_RE = re.compile(r"阅读说明|阅读须知|使用须知|报告说明书|免责声明|disclaimer|how\s+to\s+read\s+this\s+report|reading\s+guide", re.IGNORECASE)
# The heading words for conclusion / boilerplate / listing sections hold only language-level generic words; domain
# phrasings (a checkup report's "positive results and abnormalities", a datasheet's "Ordering Information" or
# "Worldwide Sales") are induced per KB by the scenario profile (conclusion_headings / boilerplate_headings /
# listing_headings) and never written into code -- 2026-09-08 generality review.
# The Chinese word for "advice" stays out of the generic list: it appears in every section of a Chinese report
# (it classed 72 units of the health KB as conclusions); the specific "medication advice / prevention advice"
# headings come from the profile instead.
_CONCLUSION_HINT_RE = re.compile(
    r"结论|小结|总结|摘要|汇总|"
    r"\bsummary\b|\bconclusions?\b|key\s+findings|\bfindings\b|recommendations?|executive\s+summary|\babstract\b|\bverdict\b",
    re.IGNORECASE)
_DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b|\d{4}\s*年\s*\d{1,2}\s*月")
_REVISION_RE = re.compile(r"\brev(?:ision)?\b|\becn\b|\bchange\s+(?:log|history|description)|修订|版本|变更|更改", re.IGNORECASE)
_BOILER_STRONG_RE = re.compile(
    r"revision\s+history|document\s+history|change\s+history|change\s+log|revision\s+record|版本历史|修订(?:历史|记录|说明)|"
    r"变更记录|更改记录|legal\b|disclaimer|trademark|copyright|免责|商标|版权|important\s+notice|"
    r"acknowledg|致谢|document\s+conventions|units\s+of\s+measure|测量单位|法律信息|法律声明|contact\s+(?:us|information)|联系方式",
    re.IGNORECASE)
_TOC_HINT_RE = re.compile(r"table\s+of\s+contents|\bcontents\b|^\s*目录|\b目录\b", re.IGNORECASE)
_LISTING_HINT_RE = re.compile(
    r"(?:code|parts?|item|part\s+number)\s+list|list\s+of\s+(?:codes|parts|items)|\bindex\b|物料清单|清单|索引", re.IGNORECASE)
_IDENT_TOKEN_RE = re.compile(r"[A-Z0-9][A-Za-z0-9_#<>\[\]./+-]*")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


# The image parser's marker words (CAPTION / VISUAL SUMMARY / FACTS ...) and mermaid code blocks are layout, not
# content: the "summary" in "VISUAL SUMMARY" once classed all 33 image units of kb_001 as conclusions, and mermaid
# arrows were once counted as TOC dot leaders
_VISUAL_MARKER_RE = re.compile(r"\b(?:VISUAL\s+SUMMARY|CAPTION|TITLE|FACTS|ENTITIES|KEYWORDS|FOOTNOTE|EQUATION)\s*:")
_MERMAID_RE = re.compile(r"```\s*mermaid.*?(?:```|$)", re.DOTALL)


def clean_for_signals(text: str) -> str:
    return _VISUAL_MARKER_RE.sub(" ", _MERMAID_RE.sub(" ", str(text or "")))


def _heading_hit(head: str, headings: Iterable[str]) -> bool:
    words = [str(h).strip() for h in headings if str(h).strip()]
    return bool(words) and re.search("|".join(re.escape(w) for w in words), head, re.IGNORECASE) is not None


def unit_signals(text: str, sections: Iterable[str] = (), *, boilerplate_headings: Iterable[str] = (),
                 listing_headings: Iterable[str] = ()) -> dict[str, Any]:
    """Structural signals: TOC dot leaders, date lines, share of table rows, mean cell length, identifier density,
    prose share, heading hints. boilerplate_headings / listing_headings: the boilerplate / listing page heading
    words induced by this KB's profile, used as hints just like the generic word lists."""
    text = clean_for_signals(unicodedata.normalize("NFKC", str(text or "")))
    head = " ".join(str(s) for s in sections if s) + " " + text[:160]
    lines = [ln for ln in text.splitlines() if ln.strip()]
    table_lines = [ln for ln in lines if ln.count("|") >= 2 or "<td" in ln]
    cells = [c.strip() for ln in table_lines for c in ln.split("|") if c.strip() and set(c.strip()) - set("-: ")]
    cell_len = (sum(len(c) for c in cells) / len(cells)) if cells else 0.0
    tokens = [t for t in re.split(r"[\s|]+", text) if t and set(t) - set("-:")]
    ident = sum(1 for t in tokens if _IDENT_TOKEN_RE.fullmatch(t) and len(t) >= 2 and (any(ch.isdigit() for ch in t) or t.isupper()))
    segments = [seg.strip() for seg in re.split(r"[\n|]", text) if seg.strip()]
    prose_chars = sum(len(seg) for seg in segments if len(seg) >= 40 and (" " in seg or _CJK_RE.search(seg)))
    total_chars = max(1, sum(len(seg) for seg in segments))
    hint = "boilerplate" if (_BOILER_STRONG_RE.search(head) or _READING_GUIDE_RE.search(head) or _heading_hit(head, boilerplate_headings)) else (
        "toc" if _TOC_HINT_RE.search(head) else ("listing" if (_LISTING_HINT_RE.search(head) or _heading_hit(head, listing_headings)) else ""))
    toc_lines = sum(1 for ln in lines if _TOC_LINE_RE.match(ln))
    return {
        "toc": len(_TOC_RE.findall(text)) + (toc_lines if hint == "toc" else 0),
        "toc_lines": toc_lines,
        "dates": len(_DATE_RE.findall(text)),
        "revision": bool(_REVISION_RE.search(text[:400])),
        "table_ratio": round(len(table_lines) / len(lines), 3) if lines else 0.0,
        "cell_len": round(cell_len, 1),
        "ident_ratio": round(ident / len(tokens), 3) if tokens else 0.0,
        "prose_ratio": round(prose_chars / total_chars, 3),
        "hint": hint,
        "greeting": bool(_GREETING_RE.search(text[:600])),
        "promo": len(_PROMO_RE.findall(text)),
        "chars": len(text),
        "conclusion": bool(_CONCLUSION_HINT_RE.search(head)),
    }


def classify_unit_text(text: str, sections: Iterable[str] = (), *, conclusion_headings: Iterable[str] = (),
                       boilerplate_headings: Iterable[str] = (), listing_headings: Iterable[str] = ()) -> str:
    """Structural rules (no model call):
    · TOC: >= 5 "dot leader + page number" hits, or a TOC-like heading with almost no prose in the body, or a
      TOC-like heading with >= 3 "number ... page" lines;
    · revision history / legal notice / reading guide / disclaimer: the heading or first line hits a strong hint,
      or >= 3 dates plus revision wording;
    · cover letter ("Dear Mr. X, hello"), QR-code / app promotion page (>= 2 promotion words and hardly any body);
    · listing page: more than half the lines are table rows, cells are very short and identifiers are dense
      (ball maps, ordering-code tables), or a listing hint in the heading plus a table;
    · conclusion section: the heading hits a conclusion / summary / recommendation word (generic list + heading
      words induced by this KB's scenario profile);
    · everything else is body. Parameter tables (AC / DC characteristics) have descriptive text in their cells
      and land in body."""
    sig = unit_signals(text, sections, boilerplate_headings=boilerplate_headings, listing_headings=listing_headings)
    if sig["toc"] >= 5:
        return "boilerplate"
    if sig["hint"] == "boilerplate":
        return "boilerplate"
    if sig["dates"] >= 3 and sig["revision"]:
        return "boilerplate"
    if sig["hint"] == "toc" and (sig["toc"] >= 2 or sig["prose_ratio"] < 0.3 or sig["toc_lines"] >= 3):
        return "boilerplate"
    if sig["greeting"]:
        return "boilerplate"
    if sig["table_ratio"] < 0.3 and (sig["promo"] >= 3 or (sig["promo"] >= 2 and sig["chars"] <= 600)):
        return "boilerplate"      # QR-code / app promotion page: a short page dense with promotion words
    if sig["table_ratio"] >= 0.5 and sig["cell_len"] <= 8 and sig["ident_ratio"] >= 0.4:
        return "listing"
    if sig["hint"] == "listing" and sig["table_ratio"] >= 0.3 and sig["prose_ratio"] < 0.4:
        return "listing"
    head = " ".join(str(s) for s in sections if s) + " " + clean_for_signals(unicodedata.normalize("NFKC", str(text or "")))[:160]
    extra = [str(h).strip() for h in conclusion_headings if str(h).strip()]
    if sig["conclusion"] or (extra and re.search("|".join(re.escape(h) for h in extra), head, re.IGNORECASE)):
        return "conclusion"
    return "body"


@dataclass
class ChunkRef:
    point_id: str
    chunk_uid: str
    doc_id: str
    content_version: str
    chunk_index: int
    block_id: str
    block_type: str
    section_path: list[str]
    text: str
    n_tokens: int
    rel_path: str = ""
    filename: str = ""
    page_idx: int | None = None
    page_end: int | None = None          # last page of a table / paragraph merged across pages (Codex review F09); equals page_idx for one page
    # glued raw strings in tables that screenshot verification did not split (757557): facts that take these values
    # are not trusted parameters (F02)
    ambiguous_values: list[str] = field(default_factory=list)
    # metrics where the model's estimated reading in an image description conflicts with the text in the image
    # (Codex review F01): {label, text_value, model_value}; the fact layer downgrades such facts
    value_conflicts: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Unit:
    unit_id: str
    doc_id: str
    rel_path: str
    section_path: list[str]
    block_ids: list[str]
    chunk_uids: list[str]
    point_ids: list[str]
    n_tokens: int
    text: str
    order: int = 0
    block_types: list[str] = field(default_factory=list)
    # other sections the unit spans (those not cut at the section change); written into the prompt's Section line too
    extra_sections: list[str] = field(default_factory=list)
    # unit kind given by the structural rules (body / listing / boilerplate); the model's verdict lives in the
    # extraction result, and the combined kind is written into graph.json
    kind: str = "body"
    # unverified glued raw table strings in the unit; fact extraction marks facts holding these values untrusted (F02)
    ambiguous_values: list[str] = field(default_factory=list)
    # metrics whose estimated reading in an image description conflicted with the in-image text (the chunks'
    # visual_value_conflicts): the corresponding facts are downgraded (Codex review F01)
    value_conflicts: list[dict[str, Any]] = field(default_factory=list)
    # the document's axis value (date or version, computed by graph/temporal.document_axis); goes into the prompt's
    # Document line and is the default validity period of facts. Not part of unit_id (content addressing looks only
    # at text and position), so a changed axis value needs no re-extraction.
    axis: str = ""

    @property
    def document_label(self) -> str:
        """The prompt's Document line: the file name (without directory) plus the axis value."""
        name = self.rel_path.rsplit("/", 1)[-1] if self.rel_path else (self.doc_id or "")
        return f"{name} ({self.axis})" if self.axis else name

    @property
    def section_label(self) -> str:
        first = " > ".join(s for s in self.section_path if s) or "(no section)"
        if self.extra_sections:
            return " | ".join([first, *self.extra_sections])
        return first

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Unit":
        return cls(
            unit_id=str(data["unit_id"]), doc_id=str(data["doc_id"]), rel_path=str(data.get("rel_path") or ""),
            section_path=list(data.get("section_path") or []), block_ids=list(data.get("block_ids") or []),
            chunk_uids=list(data.get("chunk_uids") or []), point_ids=list(data.get("point_ids") or []),
            n_tokens=int(data.get("n_tokens") or 0), text=str(data.get("text") or ""),
            order=int(data.get("order") or 0), block_types=list(data.get("block_types") or []),
            extra_sections=list(data.get("extra_sections") or []),
            kind=str(data.get("kind") or "body"),
            ambiguous_values=[str(x) for x in (data.get("ambiguous_values") or [])],
            value_conflicts=[dict(x) for x in (data.get("value_conflicts") or []) if isinstance(x, dict)],
            axis=str(data.get("axis") or ""),
        )


def unit_id_for(kb_id: str, doc_id: str, text: str, chunk_uids: Iterable[str] = (), *,
                positions: Iterable[int] | None = None) -> str:
    """Content-addressed unit id. Besides the text it also takes the unit's position in the document (chunk
    indexes): one document may contain two identical passages (a repeated table in a datasheet, the same note at
    the end of every chapter), and addressing by text alone would collapse the two units into one -- Neo4j would
    MERGE them into a single TextUnit and the counts would not add up.
    It used to take the chunk uids, which carry content_version: after a file was re-parsed the unit id changed
    even when not a single character of the text had, and every graph_extractions lookup missed (health check
    R10). With position-based addressing, units whose text and chunking are unchanged stay stable across parse
    versions; when positions is not given the uids are still used (legacy callers)."""
    key = ",".join(str(int(p)) for p in positions) if positions is not None else ",".join(chunk_uids)
    raw = f"unit|{kb_id}|{doc_id}|{key}|{text}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:32]


def fetch_chunk_payloads(q: Any, collection: str, ledger: Iterable[dict[str, Any]], *, batch: int = 256) -> list[ChunkRef]:
    """Fetch payloads (no vectors) from the main collection according to the frozen chunk ledger. Chunks that are
    in the ledger but no longer in the collection are skipped (a benign race between a delete job and the build)."""
    rows = [c for c in ledger if c.get("point_id")]
    by_point = {str(c["point_id"]): c for c in rows}
    out: list[ChunkRef] = []
    ids = list(by_point)
    for i in range(0, len(ids), batch):
        points = q.retrieve(collection_name=collection, ids=ids[i:i + batch], with_vectors=False, with_payload=True)
        for p in points:
            payload = p.payload or {}
            text = str(payload.get("text") or "")
            if not text.strip():
                continue
            ref = by_point.get(str(p.id), {})
            section = [str(s) for s in (payload.get("section_path") or []) if str(s).strip()]
            out.append(ChunkRef(
                point_id=str(p.id),
                chunk_uid=str(payload.get("chunk_uid") or ref.get("chunk_uid") or ""),
                doc_id=str(payload.get("doc_id") or ref.get("doc_id") or ""),
                content_version=str(payload.get("content_version") or ref.get("content_version") or ""),
                chunk_index=int(payload.get("chunk_index") or 0),
                block_id=str(payload.get("block_id") or ""),
                block_type=str(payload.get("block_type") or ""),
                section_path=section,
                text=text,
                n_tokens=int(payload.get("token_count") or 0) or count_tokens(text),
                rel_path=str(payload.get("rel_path") or ""),
                filename=str(payload.get("filename") or ""),
                page_idx=payload.get("page_idx"),
                page_end=payload.get("page_end", payload.get("page_idx")),
                ambiguous_values=unrepaired_glued_values(payload.get("table_flags"), payload.get("table_repair")),
                value_conflicts=[dict(x) for x in (payload.get("visual_value_conflicts") or []) if isinstance(x, dict)],
            ))
    return out


def dedupe_overlap(previous: str, text: str, *, min_chars: int = MIN_OVERLAP_CHARS) -> str:
    """Remove the part at the start of text that repeats the end of previous (the chunker's whole-sentence overlap).
    An overlap must be at least min_chars long, so one or two coincidentally identical characters are not cut off."""
    if not previous or not text:
        return text
    longest = min(len(previous), len(text))
    for k in range(longest, min_chars - 1, -1):
        if previous.endswith(text[:k]):
            return text[k:].lstrip()
    return text


def _flush(unit_chunks: list[ChunkRef], texts: list[str], *, kb_id: str, order: int,
           conclusion_headings: Iterable[str] = (), boilerplate_headings: Iterable[str] = (),
           listing_headings: Iterable[str] = ()) -> Unit | None:
    if not unit_chunks:
        return None
    text = "\n\n".join(t for t in texts if t.strip()).strip()
    if not text:
        return None
    first = unit_chunks[0]
    block_ids: list[str] = []
    for c in unit_chunks:
        if c.block_id and c.block_id not in block_ids:
            block_ids.append(c.block_id)
    chunk_uids = [c.chunk_uid for c in unit_chunks]
    extra: list[str] = []
    for c in unit_chunks:
        if tuple(c.section_path) != tuple(first.section_path):
            label = " > ".join(x for x in c.section_path if x) or "(no section)"
            if label not in extra:
                extra.append(label)
    sections = [" > ".join(x for x in first.section_path if x), *extra]
    return Unit(
        kind="body" if any(str(c.block_type) == "code" for c in unit_chunks)
        else classify_unit_text(text, sections, conclusion_headings=conclusion_headings,
                                boilerplate_headings=boilerplate_headings, listing_headings=listing_headings),
        unit_id=unit_id_for(kb_id, first.doc_id, text, positions=[c.chunk_index for c in unit_chunks]),
        doc_id=first.doc_id,
        rel_path=first.rel_path,
        section_path=list(first.section_path),
        block_ids=block_ids,
        chunk_uids=chunk_uids,
        point_ids=[c.point_id for c in unit_chunks],
        n_tokens=count_tokens(text),
        text=text,
        order=order,
        block_types=sorted({c.block_type for c in unit_chunks if c.block_type}),
        extra_sections=extra,
        ambiguous_values=list(dict.fromkeys(v for c in unit_chunks for v in (c.ambiguous_values or []))),
        value_conflicts=[c for ch in unit_chunks for c in (ch.value_conflicts or [])],
    )


def build_units(chunks: list[ChunkRef], *, kb_id: str, unit_chunks: int = DEFAULT_UNIT_CHUNKS,
                conclusion_headings: Iterable[str] = (), axes: dict[str, str] | None = None,
                boilerplate_headings: Iterable[str] = (), listing_headings: Iterable[str] = ()) -> list[Unit]:
    """Walk the chunks in document -> chunk_index order and merge every unit_chunks consecutive chunks into one
    unit: on a section change the current unit is only cut once it is at least half full, table blocks are kept
    whole (when they fit within twice the limit), and unit_chunks=1 means one chunk per unit with no merging.
    conclusion_headings: conclusion-section heading words induced by this KB's scenario profile;
    axes: {doc_id: axis value}."""
    limit = max(1, int(unit_chunks))
    headings = tuple(str(h) for h in conclusion_headings if str(h).strip())
    boiler = tuple(str(h) for h in boilerplate_headings if str(h).strip())
    listing = tuple(str(h) for h in listing_headings if str(h).strip())
    axes = axes or {}
    fold_min = max(1, (limit + 1) // 2)
    by_doc: dict[str, list[ChunkRef]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)
    units: list[Unit] = []
    order = 0
    for doc_id in sorted(by_doc):
        doc_chunks = sorted(by_doc[doc_id], key=lambda c: (c.chunk_index, c.chunk_uid))
        block_sizes: dict[str, int] = {}
        for c in doc_chunks:
            block_sizes[c.block_id] = block_sizes.get(c.block_id, 0) + 1
        current: list[ChunkRef] = []
        texts: list[str] = []
        pending_table: str | None = None      # the table block currently being absorbed as a whole

        def flush() -> None:
            nonlocal current, texts, order
            unit = _flush(current, texts, kb_id=kb_id, order=order, conclusion_headings=headings, boilerplate_headings=boiler, listing_headings=listing)
            if unit is not None:
                unit.axis = str(axes.get(unit.doc_id) or "")
                units.append(unit)
                order += 1
            current, texts = [], []

        for c in doc_chunks:
            section = tuple(c.section_path)
            same_table = pending_table is not None and c.block_id == pending_table
            if current and not same_table:
                if limit == 1:
                    flush()                                   # no merging: one chunk per unit
                elif section != tuple(current[0].section_path) and len(current) >= fold_min:
                    flush()
                elif c.block_type in TABLE_BLOCK_TYPES and c.block_id != current[-1].block_id:
                    whole = block_sizes.get(c.block_id, 1)
                    # a table that fits joins the current unit; otherwise start a new unit that holds the whole table
                    if len(current) + whole > limit and whole <= 2 * limit:
                        flush()
                elif len(current) >= limit:
                    flush()
            piece = c.text
            if current and current[-1].block_id == c.block_id and current[-1].chunk_index + 1 == c.chunk_index:
                piece = dedupe_overlap(texts[-1], piece)
                if piece:
                    texts[-1] = (texts[-1].rstrip() + ("\n" if texts[-1].rstrip().endswith("|") else " ") + piece)
                current.append(c)
            else:
                current.append(c)
                texts.append(piece)
            if c.block_type in TABLE_BLOCK_TYPES and limit > 1:
                pending_table = c.block_id if block_sizes.get(c.block_id, 1) <= 2 * limit else None
            else:
                pending_table = None
        flush()
    seen: set[str] = set()
    for unit in units:
        if unit.unit_id in seen:
            raise RuntimeError(f"duplicate unit_id {unit.unit_id} in {unit.doc_id}: chunk_uids={unit.chunk_uids}")
        seen.add(unit.unit_id)
    return units


def write_units(path: Path, units: Iterable[Unit]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for unit in units:
            fh.write(json.dumps(unit.to_json(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_units(path: Path) -> list[Unit]:
    units: list[Unit] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                units.append(Unit.from_json(json.loads(line)))
    return units


def units_summary(units: list[Unit]) -> dict[str, Any]:
    docs = {u.doc_id for u in units}
    tokens = [u.n_tokens for u in units]
    sizes = [len(u.chunk_uids) for u in units]
    kinds: dict[str, int] = {}
    for u in units:
        kinds[u.kind or "body"] = kinds.get(u.kind or "body", 0) + 1
    return {
        "units": len(units),
        "units_by_kind": kinds,
        "documents": len(docs),
        "chunks": sum(sizes),
        "tokens": sum(tokens),
        "max_unit_tokens": max(tokens) if tokens else 0,
        "avg_unit_tokens": (sum(tokens) // len(tokens)) if tokens else 0,
        "avg_unit_chunks": round(sum(sizes) / len(sizes), 2) if sizes else 0,
    }
