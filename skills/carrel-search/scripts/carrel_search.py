#!/usr/bin/env python3
"""Small read-only client for the Carrel search service. Python 3.9+, standard library only.

Every JSON command prints a compact view of the response (ids, scores and debug fields left out, document text kept)
and keeps the complete response in a file under the work directory. A view that does not fit one output is paged:
`show <call id> --page 2` prints the next page. Later commands point at one entry of an earlier response with
--ref "<call id>:<label>" (S3, F2, E5, H1.2, N4 ...) instead of retyping ids or writing request files; `show` prints
an entry in full."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

KEEP_SECONDS = 86400              # responses and pictures in the work directory older than a day are removed
REF_RE = re.compile(r"^([0-9a-f]{6}):([SFPERHN])(\d+)(?:\.(\d+))?$")
STORED_RE = re.compile(r"^[0-9a-f]{6}(-[A-Za-z0-9.\-]+)?\.(json|png|jpg|jpeg|webp|gif|bmp|img)$")
# One printed output stays below what agent harnesses show inline: Claude Code moves a tool output above 30,000 bytes
# into a file and shows a 2 KB preview, which costs the agent further steps. Nothing is dropped to get there: what does
# not fit goes to the next page (`show <call id> --page 2`).
PAGE_BYTES = 28000
PAGE_RESERVE = 400                # room kept on a page for the lines that say which page it is
PAGE_BREAK = "\f"                 # a view puts it where the next page must start
TOP_SOURCES = 4                   # the best hits: when one of them does not fit the first page, the next page must be read
CONTEXT_STEP = 60                 # chunks /context is asked for per request; a longer range is read in several
CONTEXT_MAX = 300                 # chunks one context command reads at most
OPENING_CHARS = 60
GRAPH_LIST_MAX = 12               # entities / relations matching the question that the view lists
INDEX_LINE_RE = re.compile(r"^(ENTITIES|KEYWORDS):", re.I)      # index fodder of picture chunks, dropped from the view

# Wording of the compact view.
TXT = {
    "call": "call {id} · {op}",
    "kb": "knowledge base: {names}",
    "ref_hint": "Point at an entry as {id}:{label} after --ref: context (read the original), image / crop (see the picture), neighbors (walk the graph), facts (facts under it); show {id}:{label} prints its full text or every field",
    "full": "complete response: {path}",
    "next": "▶ Answer directly from the result above, in this order: the answer, the reference files, numbered follow-ups. No walking along the graph this turn; pick the follow-ups from the graph leads, each one hop from one entity.",
    "next_hop": "▶ Answer only the follow-up the user picked and go no further; then list the reference files as before and offer new numbered follow-ups.",
    "state": "evidence: {state}",
    "hits": "{hits} hits",
    "neighbors_n": "{n} neighbouring chunks",
    "none": "⚠ nothing relevant was found, or only diagnostic candidates: not enough to support an answer; this does not mean the knowledge base holds nothing",
    "degraded": "gaps in this retrieval: {items}",
    "routing_weak": "routing: weak vector evidence (weak)",
    "routing_widened": "routing: widened to every knowledge base (widened)",
    "sec_sources": "━━ Sources ━━",
    "sec_specs": "━━ Facts ━━",
    "sec_pages": "━━ Pages (compiled second-hand summaries; leads only) ━━",
    "sec_graph": "━━ Graph leads (model-extracted, not conclusions; for composing follow-ups: when the user picks one, walk one hop from its entry) ━━",
    "sec_files": "━━ Reference files (list the ones the answer uses, each path as printed here; the entries after ← come from that file) ━━",
    "neighbor_of": "neighbouring chunk of {label}",
    "neighbor_at": "neighbouring chunk of {label}",
    "neighbor_refs": "neighbouring chunks (context; full text on the next page): {items}",
    "opening_only": "This page ran out of room: {labels} shown by the opening words only; their full text, and that of the neighbouring chunks not printed, is on the next page (show {id} --page 2), to be read when useful",
    "opening_top": "This page ran out of room: {labels} shown by the opening words only, some of the best sources among them. Their full text is on the next page; read it before answering: show {id} --page 2",
    "rest_only": "This page ran out of room: the full text of the neighbouring chunks not printed is on the next page (show {id} --page 2), to be read when useful",
    "sec_rest": "━━ Full text of the other sources ━━",
    "sec_rest_must": "━━ Full text of the sources (continued from the previous page; to be read) ━━",
    "continued": "(continued from the previous page)",
    "maybe_more": "⚠ The length of this document is not known and one command reads at most {n} chunks; there may be more: read on with context --ref {ref} --before 0 --after {m}",
    "position": "chunk {k} of {n}",
    "same_header": "HEADER: same as {label}",
    "page_head": "call {id} · {op} · page {page} of {total}",
    "page_more": "(page {page} of {total}; more follows, read on: show {what} --page {next})",
    "page_more_optional": "(page {page} of {total}; the next page holds the full text of the other sources, to be read when useful: show {what} --page {next})",
    "page_last": "(page {total} of {total}, the end)",
    "page_text": "has body text, show {ref}",
    "not_accepted": "below the relevance threshold; a lead only",
    "unranked": "relevance not confirmed",
    "truncated": "text truncated; context --ref {ref} when needed",
    "stitched": "stitched with adjacent chunks",
    "boilerplate": "boilerplate page",
    "parse_degraded": "parsing degraded: {why}",
    "picture": "picture · confidence {conf}",
    "picture_conflicts": "{n} conflicts between the text in the picture and estimated readings; trust the text in the picture",
    "picture_view": "see the picture: image --ref {ref}",
    "same_cite": "same citation as {label}",
    "same_file": "same file as {label}",
    "conflict": "conflict",
    "stale": "sources no longer active",
    "series": "series: {text}",
    "sources_active": "sources active {x}",
    "more_docs": "(one of {n} documents)",
    "hoods": "One-hop neighbourhood of the subjects (→ points at the object, — is undirected; the number in brackets is the far end's own relation count):",
    "named": "named in the question",
    "relations_n": "{n} relations",
    "facts_n": "{n} facts",
    "docs_n": "{n} documents",
    "entities": "Entities:",
    "relationships": "Relations (— joins two ends whose direction is not given: do not read subject and object from the order; an R entry cannot follow --ref, walk on from the entity entry of one of its ends):",
    "center": "{title} ({type}) · {total} relations in all, {count} here (by weight) · {facts} facts under it",
    "kinds": "relations by kind: {items} (one kind only: --type <predicate>)",
    "matches": "other entities with this name: {items}",
    "not_found": "No entity with this name. Candidates (look one up again by id: --entity-id):",
    "picture_text": "picture description",
    "unlocated": "the excerpt did not locate the far end; context --ref {ref} when needed",
    "inactive": "evidence no longer active",
    "violation": "the end types do not fit this kind of relation; judge it by the excerpt",
    "same_excerpt": "same excerpt as {label}",
    "no_evidence": "no evidence text retrieved",
    "total_page": "{total} in all, this page {first}–{last}",
    "has_more": "more: --offset {offset}",
    "types": "entities per type (--type takes the type name; the upper class in brackets goes with --parent-type): {items}",
    "properties": "properties: {items}",
    "subject": "subject {title} ({type})",
    "property": "property \"{query}\" matched by {matched}, {concepts} concepts",
    "context": "document {doc} · chunks {a}–{b} of {total} · {tokens} tokens",
    "context_plain": "document {doc} · {n} chunks · {tokens} tokens",
    "chunk_at": "chunk {k}",
    "no_rows": "nothing matches",
    "catalog_kb": "{kb} {name} · {docs} documents · {chunks} chunks · {graph}",
    "graph_yes": "has a graph",
    "graph_no": "no graph",
    "saved": "picture saved: {path} ({mime}, {w}×{h}, source {source})",
    "shown": "{ref}:",
    "lp": " (", "rp": ")", "list": ", ", "semi": "; ", "colon": ": ", "ql": "\"", "qr": "\"", "dot": " · ",
}

LABEL_LISTS = {"S": ("sources",), "F": ("specs", "facts"), "P": ("pages",), "E": ("entities",), "R": ("relationships",),
               "H": ("neighborhoods",), "N": ("neighbors",)}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Do not forward bearer credentials to redirected destinations.


def read_object(path):
    raw = sys.stdin.read() if path == "-" else Path(path).expanduser().read_text(encoding="utf-8")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("JSON input must be an object")
    return value


def config(args):
    explicit = args.config or os.environ.get("CARREL_SEARCH_CONFIG")
    path = Path(explicit).expanduser() if explicit else Path.home() / ".config/carrel-search/config.json"
    return read_object(str(path)) if explicit or path.exists() else {}


def settings(args, cfg):
    url = str(args.base_url or os.environ.get("CARREL_SEARCH_BASE_URL") or cfg.get("base_url")
              or "http://127.0.0.1:9810").rstrip("/")
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("base_url must be an HTTP(S) service URL without credentials, query or fragment")
    timeout = float(args.timeout if args.timeout is not None else cfg.get("timeout_seconds", 60))
    if not math.isfinite(timeout) or not 0 < timeout <= 300:
        raise ValueError("timeout must be greater than 0 and at most 300 seconds")
    token = os.environ.get("CARREL_SEARCH_TOKEN") or os.environ.get(str(cfg.get("token_env") or "KB_SEARCH_TOKEN"), "")
    if not token and cfg.get("token_file"):
        token = Path(cfg["token_file"]).expanduser().read_text(encoding="utf-8").strip()
    if "\n" in token or "\r" in token:
        raise ValueError("token must be a single line")
    return url, token.strip(), timeout


def work_dir(cfg):
    """Where complete responses and fetched pictures are kept so later commands can refer to them: the configured
    directory, else a folder under the system temporary directory, created for this user only (responses hold
    document text). Entries older than a day are removed."""
    root = os.environ.get("CARREL_SEARCH_WORK_DIR") or cfg.get("work_dir")
    path = Path(str(root)).expanduser() if root else Path(tempfile.gettempdir()) / "carrel-search"
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    limit = time.time() - KEEP_SECONDS
    try:
        for item in path.iterdir():
            if STORED_RE.match(item.name) and item.is_file() and item.stat().st_mtime < limit:
                item.unlink()
    except OSError:
        pass
    return path


def cite_for(name, rel_path, where):
    """Where an entry comes from, as plain text for reading: "<KB folder>/<rel_path>" from the knowledge base's
    top-level folder, then where in the file (the service's short locator: page / slide / sheet rows, or the deepest
    heading)."""
    return (str(name) + "/" if name else "") + str(rel_path) + (TXT["dot"] + where if where else "")


def file_for(name, rel_path):
    """The line a file takes in the answer's reference list: "<KB folder>/<rel_path>", a path that starts at the
    knowledge base's top-level folder whatever directory the files were synced from. No locator: the list names
    files, not places in them."""
    return (str(name) + "/" if name else "") + str(rel_path)


def add_cites(result):
    """Attach `cite` (where it comes from, plain text) and `file` (its line for the reference list) to every object
    that names a document (rel_path), and `docs_cite` / `docs_file` next to every `docs` list, so the agent copies
    reference lines instead of assembling paths."""
    if not isinstance(result, dict):
        return
    names = dict(result.get("kb_names") or {})
    if result.get("kb_id") and result.get("kb_name"):
        names[str(result["kb_id"])] = result["kb_name"]
    by_n = {s.get("n"): s for s in result.get("sources") or [] if isinstance(s, dict)}

    def where_of(obj):
        if obj.get("place"):
            return str(obj["place"])
        # A fact comes from a unit of several chunks: it takes the locator of the chunk the service found its value
        # in, else a locator the chunks it points at in this document agree on, and stays at document level otherwise.
        located = next((e for e in obj.get("evidence") or [] if isinstance(e, dict) and e.get("located") and e.get("place")), None)
        if located is not None and located.get("rel_path") == obj.get("rel_path"):
            return str(located["place"])
        linked = [by_n.get(n) for n in obj.get("sources") or [] if not isinstance(n, dict)] + list(obj.get("evidence") or [])
        places = {str(c["place"]) for c in linked if isinstance(c, dict) and c.get("rel_path") == obj.get("rel_path") and c.get("place")}
        return places.pop() if len(places) == 1 else ""

    def walk(obj, kb):
        if isinstance(obj, list):
            for item in obj:
                walk(item, kb)
            return
        if not isinstance(obj, dict):
            return
        kb = str(obj.get("kb_id") or kb or "")
        if isinstance(obj.get("rel_path"), str) and obj["rel_path"]:
            obj["cite_doc"], obj["cite_at"] = cite_for(names.get(kb), obj["rel_path"], ""), where_of(obj)
            obj["cite"] = cite_for(names.get(kb), obj["rel_path"], obj["cite_at"])
            obj["file"] = file_for(names.get(kb), obj["rel_path"])
        docs = obj.get("docs")
        if isinstance(docs, list) and docs and all(isinstance(d, str) for d in docs):
            obj["docs_cite"] = [cite_for(names.get(kb), d, "") for d in docs]
            obj["docs_file"] = [file_for(names.get(kb), d) for d in docs]
        for value in obj.values():
            if isinstance(value, (dict, list)):
                walk(value, kb)

    walk(result, str(result.get("kb_id") or ""))


def number_rows(result):
    """Rows the service does not number get an `n` so every entry of a stored response has a label: relations of a
    neighbourhood lookup count from 1, listed entities continue across pages."""
    if not isinstance(result, dict):
        return
    for i, row in enumerate(result.get("neighbors") or [], 1):
        if isinstance(row, dict):
            row.setdefault("n", i)
    offset = int(result.get("offset") or 0)
    for i, row in enumerate(result.get("entities") or [], 1):
        if isinstance(row, dict):
            row.setdefault("n", offset + i)


# ── stored responses and references ──

def stored(work, call_id):
    path = work / (call_id + ".json")
    if not path.exists():
        raise ValueError("no stored response " + call_id + " in " + str(work) + " (responses are kept for a day)")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(work, ref):
    """"<call id>:<label>" → (envelope, letter, entry, sub entry). Labels: S source, F fact, P page, E entity,
    R relationship, H subject neighbourhood (H1.2 = its second relation), N relation of a neighbourhood lookup."""
    m = REF_RE.match(str(ref or "").strip())
    if not m:
        raise ValueError("a reference looks like 3fa2c1:S3 (also F2, P1, E5, R4, H1, H1.2, N7): " + str(ref))
    call_id, letter, num, sub = m.group(1), m.group(2), int(m.group(3)), m.group(4)
    envelope = stored(work, call_id)
    result = envelope.get("result") or {}
    rows = next((result[k] for k in LABEL_LISTS[letter] if isinstance(result.get(k), list)), None)
    entry = next((r for r in rows or [] if isinstance(r, dict) and r.get("n") == num), None)
    if entry is None:
        raise ValueError("response " + call_id + " has no entry " + letter + str(num))
    child = None
    if sub is not None:
        items = entry.get("neighbors") or []
        if letter != "H" or not 1 <= int(sub) <= len(items):
            raise ValueError("response " + call_id + " has no entry " + letter + str(num) + "." + sub)
        child = items[int(sub) - 1]
    return envelope, letter, entry, child


def within(named, kb, what):
    """A reference may not lead outside the knowledge bases the caller named: the conflict is reported instead of
    widening the scope or quietly looking the entry up somewhere else."""
    if named and kb not in named:
        raise ValueError(what + " is in " + kb + ", outside the knowledge base named for this call (" + ", ".join(named)
                         + "); drop --kb to follow the reference, or name " + kb)


def entity_of(envelope, letter, entry, child):
    """The knowledge base and entity id a reference stands for when walking the graph from it."""
    result = envelope.get("result") or {}
    kb = entry.get("kb_id") or result.get("kb_id")
    if child is not None or letter == "N":
        eid = ((child or entry).get("other") or {}).get("id")
    elif letter in ("E", "H"):
        eid = entry.get("id")
    elif letter == "R":
        # A matched relation names its two ends by title only, and which end is meant is the caller's choice.
        raise ValueError("a relation is not a starting point; walk from one of its ends, by that entity's E / H entry or by name: --kb "
                         + str(kb) + ' --entity "' + str(entry.get("source")) + '" (or "' + str(entry.get("target")) + '")')
    else:
        raise ValueError("this entry is not an entity; use an E, H, H1.2 or N reference, or --entity with a name")
    if not kb or not eid:
        raise ValueError("the referenced entry carries no knowledge base or entity id")
    return str(kb), str(eid)


def chunk_of(envelope, letter, entry):
    """The chunk a reference stands for (to read the original around it or fetch its picture): a source itself, the
    first evidence chunk of a relation or of a listed fact, or the first source a fact of /search points at."""
    result = envelope.get("result") or {}
    row = entry
    if letter in ("N", "F") and entry.get("evidence"):
        row = entry["evidence"][0]
    elif letter == "F":
        by_n = {s.get("n"): s for s in result.get("sources") or []}
        row = next((by_n[n] for n in entry.get("sources") or [] if n in by_n), None)
        if row is None:
            raise ValueError("this fact names no chunk in the response; look inside its document instead: search --question ... --rel-path <path in the knowledge base>")
    elif letter != "S":
        raise ValueError("this entry is not a chunk; use an S, N or F reference")
    kb = row.get("kb_id") or entry.get("kb_id") or result.get("kb_id")
    if not kb or not row.get("point_id"):
        raise ValueError("the referenced entry carries no chunk")
    return str(kb), row


# ── compact view ──

def flat(text, limit=None):
    s = " ".join(str(text or "").split())
    return s if not limit or len(s) <= limit else s[:limit - 1] + "…"


def par(*parts):
    """Bracketed remark: the non-empty parts joined and wrapped in the view's brackets."""
    body = TXT["dot"].join(str(x) for x in parts if x not in (None, ""))
    return TXT["lp"] + body + TXT["rp"] if body else ""


def notes_of(items):
    return "  " + TXT["lp"] + TXT["semi"].join(items) + TXT["rp"] if items else ""


class Cites:
    """A citation is printed in full the first time. A later entry with the same citation points at that entry; one
    from the same file but another place in it names the place and points at the entry for the file."""

    def __init__(self):
        self.first, self.docs = {}, {}

    def put(self, label, cite, row=None):
        if not cite:
            return ""
        first = self.first.setdefault(cite, label)
        if first != label:
            return TXT["same_cite"].format(label=first)
        doc, at = (row or {}).get("cite_doc"), (row or {}).get("cite_at")
        if doc:
            known = self.docs.setdefault(doc, label)
            if known != label and at:
                return TXT["same_file"].format(label=known) + TXT["dot"] + str(at)
        return cite


class Files:
    """The files the entries of a view come from, in order of first appearance, each with the labels of its entries:
    printed at the end of the view as the lines the answer's reference list is copied from."""

    def __init__(self):
        self.rows = {}

    def add(self, label, file):
        if file:
            self.rows.setdefault(file, []).append(label)

    def lines(self):
        if not self.rows:
            return []
        return ["", TXT["sec_files"]] + [file + "  ← " + " ".join(labels) for file, labels in self.rows.items()]


def kb_line(result):
    names = dict(result.get("kb_names") or {})
    if result.get("kb_id"):
        names.setdefault(str(result["kb_id"]), result.get("kb_name") or "")
    kbs = [kb for kb in (result.get("kbs") or list(names)) if isinstance(kb, str)]
    return TXT["kb"].format(names=TXT["list"].join(str(names[kb]) + par(kb) if names.get(kb) else str(kb) for kb in kbs)) if kbs else ""


def head(envelope, *parts):
    bits = [TXT["call"].format(id=envelope["call_id"], op=envelope["operation"]), kb_line(envelope.get("result") or {})]
    return TXT["dot"].join(b for b in bits + list(parts) if b)


def arrow(row):
    if not row.get("directed"):
        return "—" + str(row.get("type")) + "—"
    return ("→" if row.get("direction") == "out" else "←") + str(row.get("type"))


def link(row):
    """A relation matched by the question, source first. The arrow is drawn only when the service says the relation is
    directed; one it marks undirected, or gives no direction for, is joined neutrally."""
    return str(row.get("source")) + " —" + str(row.get("type")) + ("→ " if row.get("directed") is True else "— ") + str(row.get("target"))


def fact_body(row):
    """"subject · property = value unit @ conditions · time", the form of the service's hint, also for the rows the
    service gave no hint (it composes one for the first few facts only); the stored text when the value is not a
    single one."""
    if row.get("hint"):
        return row["hint"]
    value = str(row.get("value") or "").strip()
    if not value or not row.get("subject"):
        return row.get("text")
    unit = str(row.get("unit") or "").strip()
    head = TXT["dot"].join(str(x).strip() for x in (row.get("subject"), row.get("property") or row.get("concept")) if x)
    body = head + " = " + value + (" " + unit if unit and unit.casefold() not in value.casefold() else "")
    if row.get("conditions_text"):
        body += " @ " + str(row["conditions_text"]).strip()
    when = str(row.get("when") or row.get("valid_from") or "").strip()
    return body + (TXT["dot"] + when if when else "")


def fact_line(row):
    flags = [TXT["conflict"]] if row.get("conflict") else []
    if row.get("verified") is False:
        flags.append(TXT["stale"])
    return flat(fact_body(row)) + "".join(" [" + f + "]" for f in flags)


def cited(body, cite):
    return body + (" — " + cite if cite else "")


def also_named(rows):
    return TXT["matches"].format(items=TXT["semi"].join(
        str(m.get("title")) + par(m.get("type"), TXT["relations_n"].format(n=m.get("degree")), "id=" + str(m.get("id"))) for m in rows))


def candidates(envelope):
    out = [head(envelope), TXT["not_found"]]
    for c in (envelope["result"].get("candidates") or []):
        out.append("  " + str(c.get("title")) + par(c.get("type")) + " id=" + str(c.get("id")) + " — " + flat(c.get("description"), 80))
    return out


def nbytes(lines):
    return sum(len(str(x).encode("utf-8")) + 1 for x in lines)


def source_text(s, headers=None, label=None):
    """The text of a source as a view prints it. Table chunks of one sheet repeat the same header line: given `headers`
    (header line → label of the first entry printed with it), a repeated one is replaced by a pointer to that entry."""
    lines = [line for line in str(s.get("text") or "").strip().splitlines() if not INDEX_LINE_RE.match(line)]
    if headers is not None:
        for i, line in enumerate(lines):
            if line.startswith("HEADER:") and len(line) > 30:
                first = headers.setdefault(line, label)
                if first != label:
                    lines[i] = TXT["same_header"].format(label=first)
    return "\n".join(lines)


def forget(headers, label):
    """An entry that is not printed in full after all gives up the header lines registered under its label."""
    for line in [line for line, first in headers.items() if first == label]:
        del headers[line]


def source_notes(s, ref, parent=None, position=False):
    notes = []
    if parent is not None:
        notes.append(TXT["neighbor_of"].format(label=parent))
    elif s.get("accepted") is False:
        notes.append(TXT["not_accepted"])
    elif s.get("accepted") is None and s.get("role") == "hit":
        notes.append(TXT["unranked"])
    if s.get("text_truncated"):
        notes.append(TXT["truncated"].format(ref=ref))
    if s.get("stitched"):
        notes.append(TXT["stitched"])
    if s.get("boilerplate"):
        notes.append(TXT["boilerplate"])
    if s.get("degraded"):
        notes.append(TXT["parse_degraded"].format(why=s["degraded"]))
    visual = s.get("visual") or {}
    if visual:
        notes.append(TXT["picture"].format(conf=visual.get("confidence")))
        if visual.get("value_conflicts"):
            notes.append(TXT["picture_conflicts"].format(n=visual["value_conflicts"]))
        notes.append(TXT["picture_view"].format(ref=ref))
    if position and s.get("chunk_total") and s.get("chunk_index") is not None:
        # where the chunk sits in its document and how long the document is (a stitched hit covers a range)
        k, st = str(int(s["chunk_index"]) + 1), s.get("stitched") or {}
        if st.get("chunk_from") is not None and st.get("chunk_to") is not None and st["chunk_to"] != st["chunk_from"]:
            k = str(int(st["chunk_from"]) + 1) + "–" + str(int(st["chunk_to"]) + 1)
        notes.append(TXT["position"].format(k=k, n=s["chunk_total"]))
    return notes


def view_search(envelope, limit=PAGE_BYTES):
    r = envelope["result"]
    cid = envelope["call_id"]
    summary = r.get("retrieval_summary") or {}
    src = [s for s in r.get("sources") or [] if isinstance(s, dict)]
    hits = [s for s in src if s.get("role") == "hit"]
    out = [head(envelope, TXT["state"].format(state=summary.get("evidence_state")),
                TXT["hits"].format(hits=len(hits)) + (" + " + TXT["neighbors_n"].format(n=len(src) - len(hits)) if len(src) > len(hits) else ""))]
    if summary.get("no_relevant_content"):
        out.append(TXT["none"])
    if summary.get("degraded"):
        out.append(TXT["degraded"].format(items="; ".join(str(x) for x in summary["degraded"])))
    routing = summary.get("routing") or {}
    if routing.get("weak"):
        out.append(TXT["routing_weak"])
    if routing.get("widened"):
        out.append(TXT["routing_widened"])
    out.append(TXT["ref_hint"].format(id=cid, label="S3"))
    cites = Cites()
    # Sources come first in the view and their citations are the ones later entries point back at, so they are
    # registered before the other sections are built; which of them are printed in full is decided last, from what
    # the other sections leave of the budget.
    source_cites = {s.get("n"): cites.put("S" + str(s.get("n")), s.get("cite"), s) for s in hits}
    tail = []
    files = Files()
    for s in hits:
        files.add("S" + str(s.get("n")), s.get("file"))
    specs = r.get("specs") or []
    if specs:
        tail += ["", TXT["sec_specs"]]
        for row in specs:
            label = "F" + str(row.get("n"))
            tail.append(cited("[" + label + "] " + fact_line(row), cites.put(label, row.get("cite"), row)))
            files.add(label, row.get("file"))
            if row.get("series_text"):
                tail.append("     " + TXT["series"].format(text=flat(row["series_text"])))
    pages = r.get("pages") or []
    if any(row.get("kind") != "source" or row.get("text") for row in pages):
        tail += ["", TXT["sec_pages"]]
        for row in pages:
            if row.get("kind") == "source" and not row.get("text"):
                continue                # only a file name and the entities it mentions; the document line says as much
            label = "P" + str(row.get("n"))
            docs = row.get("docs_cite") or []
            cite = cites.put(label, docs[0]) if docs else ""
            more = " " + TXT["more_docs"].format(n=len(docs)) if len(docs) > 1 else ""
            bits = [row.get("kind"), row.get("title"), TXT["sources_active"].format(x=row["sources_active"]) if row.get("sources_active") else ""]
            line = "[" + label + "] " + TXT["dot"].join(str(b) for b in bits if b) + TXT["colon"] + flat(row.get("summary"))
            if row.get("text"):
                line += par(TXT["page_text"].format(ref=cid + ":" + label))
            tail.append(cited(line, cite + more if cite else ""))
    hoods, entities, relations = r.get("neighborhoods") or [], r.get("entities") or [], r.get("relationships") or []
    if hoods or entities or relations:
        tail += ["", TXT["sec_graph"]]
    if hoods:
        tail.append(TXT["hoods"])
        for h in hoods:
            label = "H" + str(h.get("n"))
            bits = [str(h.get("title")) + par(h.get("type")), TXT["named"] if h.get("named") else "", TXT["relations_n"].format(n=h.get("relations")),
                    TXT["facts_n"].format(n=h.get("facts")),
                    TXT["list"].join(str(k.get("type")) + " " + str(k.get("count")) for k in h.get("predicates") or [])]
            tail.append("[" + label + "] " + TXT["dot"].join(b for b in bits if b))
            items = [label + "." + str(i) + " " + arrow(x) + " " + str((x.get("other") or {}).get("title")) + par((x.get("other") or {}).get("degree"))
                     for i, x in enumerate(h.get("neighbors") or [], 1)]
            if items:
                tail.append("     " + TXT["semi"].join(items))
    # Entities and relations that read the same are listed once (same-named entities of different scopes are often
    # matched together), under the label of the first
    seen, lines = set(), []
    for e in entities:
        text = str(e.get("title")) + par(e.get("type"))
        if text not in seen and len(lines) < GRAPH_LIST_MAX:
            seen.add(text)
            lines.append("[E" + str(e.get("n")) + "] " + text)
    if lines:
        tail.append(TXT["entities"] + " " + TXT["dot"].join(lines))
    seen, lines = set(), []
    for x in relations:
        text = link(x)
        if text not in seen and len(lines) < GRAPH_LIST_MAX:
            seen.add(text)
            lines.append("[R" + str(x.get("n")) + "] " + text)
    if lines:
        tail.append(TXT["relationships"] + " " + TXT["dot"].join(lines))
    tail += files.lines()
    if hits and not summary.get("no_relevant_content"):
        tail += ["", TXT["next"]]       # said last, where the agent decides what to do with what it has just read

    # Sources. The page is first costed with every hit cut to its opening and every neighbour chunk only named; the
    # room left after that goes to full texts: the hits in rank order (the best ones whatever the room), then the
    # neighbour chunks of the hits printed in full. What stays cut is printed in full on the next page.
    body, rest = [], []
    if src:
        title = ["", TXT["sec_sources"]]
        near = {}
        for nb in src:
            if nb.get("role") != "hit":
                near.setdefault(nb.get("of"), []).append(nb)

        def named(rows):
            return "     " + TXT["neighbor_refs"].format(items=TXT["list"].join("S" + str(nb.get("n")) + par(nb.get("place")) for nb in rows))

        firsts, cut = {}, {}
        for s in hits:
            label = "S" + str(s.get("n"))
            firsts[s.get("n")] = "[" + label + "] " + source_cites[s.get("n")] + notes_of(source_notes(s, cid + ":" + label, position=True))
            cut[s.get("n")] = [firsts[s.get("n")], "     " + flat(source_text(s), OPENING_CHARS)]
        note = TXT["opening_only"].format(id=cid, labels=TXT["list"].join("S" + str(s.get("n")) for s in hits))
        left = (limit - PAGE_RESERVE - 200 - nbytes(out) - nbytes(tail) - nbytes(title) - nbytes([note])
                - sum(nbytes(cut[s.get("n")]) for s in hits) - sum(nbytes([named(rows)]) for rows in near.values()))
        headers, blocks, later, opened = {}, {}, {}, []
        for i, s in enumerate(hits):
            label = "S" + str(s.get("n"))
            full = [firsts[s.get("n")], source_text(s, headers, label)]
            extra = nbytes(full) - nbytes(cut[s.get("n")])
            if extra <= left:
                left -= extra
                blocks[s.get("n")] = full
            else:
                forget(headers, label)
                blocks[s.get("n")] = cut[s.get("n")]
                later[s.get("n")] = [firsts[s.get("n")], source_text(s)]
                opened.append(label)
        for s in hits:
            parent, unprinted = "S" + str(s.get("n")), []
            for nb in near.get(s.get("n")) or []:
                label = "S" + str(nb.get("n"))
                first = ("[" + label + "] " + TXT["dot"].join(str(x) for x in (TXT["neighbor_at"].format(label=parent), nb.get("place")) if x)
                         + notes_of(source_notes(nb, cid + ":" + label)))
                block = [first, source_text(nb, headers, label)]
                if parent not in opened and nbytes(block) <= left:
                    left -= nbytes(block)
                    blocks[s.get("n")] += block
                else:
                    forget(headers, label)
                    later.setdefault(s.get("n"), []).extend([first, source_text(nb)])
                    unprinted.append(nb)
            if unprinted:
                blocks[s.get("n")].append(named(unprinted))
        body = title + [line for s in hits for line in blocks[s.get("n")]]
        rest = [line for s in hits for line in later.get(s.get("n")) or []]
        if rest:
            # One of the best hits cut to its opening: the answer needs the next page. Otherwise it is supplementary.
            must = any("S" + str(s.get("n")) in opened for s in hits[:TOP_SOURCES])
            key = "opening_top" if must else "opening_only" if opened else "rest_only"
            body.append(TXT[key].format(id=cid, labels=TXT["list"].join(opened)))
            rest = [PAGE_BREAK, TXT["sec_rest_must" if must else "sec_rest"]] + rest
    return out + body + tail + rest


def view_context(envelope, limit=PAGE_BYTES):
    r = envelope["result"]
    src = r.get("sources") or []
    first = src[0] if src else {}
    doc = first.get("rel_path") or first.get("doc") or r.get("doc_id")
    at = [int(s["chunk_index"]) for s in src if s.get("chunk_index") is not None]
    if at and first.get("chunk_total"):
        what = TXT["context"].format(doc=doc, a=min(at) + 1, b=max(at) + 1, total=first["chunk_total"], tokens=r.get("tokens_total"))
    else:
        what = TXT["context_plain"].format(doc=doc, n=len(src), tokens=r.get("tokens_total"))
    out = [head(envelope, what)]
    # All chunks are of one document: the first line names it, each chunk carries only its place in it.
    headers = {}
    for s in src:
        label = "S" + str(s.get("n"))
        where = s.get("place") or (TXT["chunk_at"].format(k=int(s["chunk_index"]) + 1) if s.get("chunk_index") is not None else "")
        out.append("[" + label + "] " + str(where) + notes_of(source_notes(s, envelope["call_id"] + ":" + label)))
        out.append(source_text(s, headers, label))
    if r.get("maybe_more") and src:
        out.append(TXT["maybe_more"].format(n=len(src), m=CONTEXT_MAX - 1, ref=envelope["call_id"] + ":S" + str(src[-1].get("n"))))
    files = Files()
    if src:
        files.add("S1" if len(src) == 1 else "S1–S" + str(len(src)), first.get("file"))
    return out + files.lines()


def view_neighbors(envelope, limit=PAGE_BYTES):
    r = envelope["result"]
    cid = envelope["call_id"]
    if not r.get("found"):
        return candidates(envelope)
    center = r.get("entity") or {}
    out = [head(envelope, TXT["center"].format(title=center.get("title"), type=center.get("type"), total=r.get("total"), count=r.get("count"),
                                               facts=center.get("facts")))]
    if r.get("predicates"):
        out.append(TXT["kinds"].format(items=TXT["dot"].join(str(k.get("type")) + " " + str(k.get("count")) for k in r["predicates"])))
    if r.get("matches"):
        out.append(also_named(r["matches"]))
    if r.get("neighbors"):
        out.append(TXT["ref_hint"].format(id=cid, label="N3"))
    cites, excerpts, files = Cites(), {}, Files()
    for x in r.get("neighbors") or []:
        label = "N" + str(x.get("n"))
        other = x.get("other") or {}
        out.append("[" + label + "] " + arrow(x) + " " + str(other.get("title")) + par(other.get("type"), TXT["relations_n"].format(n=other.get("degree")))
                   + notes_of([TXT["violation"]] if x.get("type_violation") else []))
        ev = next((e for e in x.get("evidence") or [] if e.get("excerpt")), None) or next(iter(x.get("evidence") or []), None)
        if ev is None:
            out.append("     " + TXT["no_evidence"])
            continue
        marks = []
        if ev.get("visual"):
            marks.append(TXT["picture_text"])
        if ev.get("active") is False:
            marks.append(TXT["inactive"])
        if ev.get("excerpt") and ev.get("excerpt_match") not in ("both", "other"):
            marks.append(TXT["unlocated"].format(ref=cid + ":" + label))
        body = TXT["ql"] + ev["excerpt"] + TXT["qr"] if ev.get("excerpt") else TXT["no_evidence"]
        if ev.get("excerpt"):
            first = excerpts.setdefault((ev["excerpt"], ev.get("cite")), label)
            if first != label:
                body = TXT["same_excerpt"].format(label=first)
        out.append("     " + cited(body + (TXT["lp"] + TXT["semi"].join(marks) + TXT["rp"] if marks else ""), cites.put(label, ev.get("cite"), ev)))
        files.add(label, ev.get("file"))
    return out + files.lines() + (["", TXT["next_hop"]] if r.get("neighbors") else [])


def page_bits(r, count):
    offset = int(r.get("offset") or 0)
    bits = [TXT["total_page"].format(total=r.get("total"), first=offset + 1 if count else 0, last=offset + count)]
    if r.get("has_more"):
        bits.append(TXT["has_more"].format(offset=offset + count))
    return bits


def view_entities(envelope, limit=PAGE_BYTES):
    r = envelope["result"]
    rows = r.get("entities") or []
    out = [head(envelope, *page_bits(r, len(rows)))]
    if r.get("types"):
        out.append(TXT["types"].format(items=TXT["dot"].join(str(t.get("type")) + " " + str(t.get("count")) + par(t.get("parent_type")) for t in r["types"])))
    out.append(TXT["ref_hint"].format(id=envelope["call_id"], label="E3") if rows else TXT["no_rows"])
    cites, files = Cites(), Files()
    for e in rows:
        label = "E" + str(e.get("n"))
        docs = e.get("docs_cite") or []
        out.append(cited("[" + label + "] " + str(e.get("title")) + par(e.get("type"), TXT["relations_n"].format(n=e.get("degree")),
                                                                        TXT["docs_n"].format(n=e.get("doc_count"))),
                         cites.put(label, docs[0]) if docs else ""))
        files.add(label, next(iter(e.get("docs_file") or []), None))
    return out + files.lines()


def view_facts(envelope, limit=PAGE_BYTES):
    r = envelope["result"]
    if not r.get("found", True):
        return candidates(envelope)
    rows = r.get("facts") or []
    parts = []
    if r.get("subject"):
        parts.append(TXT["subject"].format(title=r["subject"].get("title"), type=r["subject"].get("type")))
    if r.get("property"):
        parts.append(TXT["property"].format(query=r["property"].get("query"), matched=r["property"].get("matched"), concepts=r["property"].get("concepts")))
    out = [head(envelope, *(parts + page_bits(r, len(rows))))]
    if r.get("degraded"):
        out.append(TXT["degraded"].format(items="; ".join(str(x) for x in r["degraded"])))
    if r.get("matches"):
        out.append(also_named(r["matches"]))
    if r.get("properties"):
        out.append(TXT["properties"].format(items=TXT["dot"].join(str(p.get("concept")) + " " + str(p.get("count")) for p in r["properties"])))
    out.append(TXT["ref_hint"].format(id=envelope["call_id"], label="F3") if rows else TXT["no_rows"])
    cites, files = Cites(), Files()
    for row in rows:
        label = "F" + str(row.get("n"))
        out.append(cited("[" + label + "] " + fact_line(row), cites.put(label, row.get("cite"), row)))
        files.add(label, row.get("file"))
        if row.get("series_text"):
            out.append("     " + TXT["series"].format(text=flat(row["series_text"])))
    return out + files.lines() + (["", TXT["next_hop"]] if rows and r.get("subject") else [])


def view_catalog(envelope, limit=PAGE_BYTES):
    out = [head(envelope)]
    for kb in (envelope["result"].get("kbs") or []):
        out.append(TXT["catalog_kb"].format(kb=kb.get("kb_id"), name=kb.get("name"), docs=kb.get("docs"), chunks=kb.get("chunks"),
                                            graph=TXT["graph_yes"] if kb.get("has_graph") else TXT["graph_no"]))
        for key in ("domain", "subject_types", "entity_types", "docs_sample"):
            if kb.get(key):
                out.append("  " + key + ": " + flat(kb[key] if isinstance(kb[key], str) else json.dumps(kb[key], ensure_ascii=False), 400))
    return out


VIEWS = {"search": view_search, "context": view_context, "neighbors": view_neighbors, "entities": view_entities, "facts": view_facts,
         "catalog": view_catalog}


BLOCK_RE = re.compile(r"^(\[(?:[0-9a-f]{6}:)?[A-Z]\d+(?:\.\d+)?\] |▶ )")


def split_block(block, room, first):
    """An entry longer than a page, cut into pieces that fit: the first into `first` bytes (what is left of the
    current page), the others into `room`. It is cut at line ends, and inside a line only when the line alone is
    longer than a page. Every piece after the first opens with the entry's label and a note that it continues."""
    lines = "\n".join(block).split("\n")
    opening = next((m.group(1) for m in map(BLOCK_RE.match, lines) if m), "")
    more = (opening if opening.startswith("[") else "") + TXT["continued"]
    room -= nbytes([more])
    pieces, cur, used, cap = [], [], 0, first
    for line in lines:
        raw = line.encode("utf-8")
        while used + len(raw) + 1 > cap:
            if len(raw) + 1 > room and cap - used > 200:
                cut = raw[:cap - used - 1].decode("utf-8", errors="ignore")     # never in the middle of a character
                cur.append(cut)
                raw = raw[len(cut.encode("utf-8")):]
            pieces.append(cur)
            cur, used, cap = [more], 0, room
        cur.append(raw.decode("utf-8"))
        used += len(raw) + 1
    return pieces + [cur]


def pages_of(lines, limit):
    """Split a view into pages below `limit` bytes. An entry starts at a line that opens with its label and is kept
    whole unless it is longer than a page by itself; a blank line or a section title stays with what follows it;
    PAGE_BREAK starts a new page."""
    blocks, cur, titles = [], [], True
    for line in lines:
        text = str(line)
        if text == PAGE_BREAK:
            blocks += [cur, None] if cur else [None]
            cur, titles = [], True
            continue
        title = text == "" or text.startswith("━━ ")
        if cur and not titles and (title or BLOCK_RE.match(text)):
            blocks.append(cur)
            cur, titles = [], True
        cur.append(text)
        titles = titles and title
    if cur:
        blocks.append(cur)
    room = limit - PAGE_RESERVE
    pages, page, used = [], [], 0
    for block in blocks:
        pieces = [block]
        if block is not None and nbytes(block) > room:
            # an entry too long for any page starts on this one when a fair part of it is still free
            pieces = split_block(block, room, room - used if page and room - used >= 3000 else room)
        for piece in pieces:
            size = nbytes(piece) if piece is not None else 0
            if page and (piece is None or used + size > room):
                pages.append(page)
                page, used = [], 0
            if piece is not None:
                page, used = page + piece, used + size
    return pages + [page] if page or not pages else pages


def one_page(pages, page, what, head=None):
    """Page `page` of a paged view with the lines that say where it stands: a first line on the later pages, and at
    the end what the next page is — more of the same, to be read on, or only the full texts a search left out."""
    total = len(pages)
    if not 1 <= page <= total:
        raise ValueError("show " + what + " has " + str(total) + (" pages" if total > 1 else " page"))
    out = list(pages[page - 1])
    while out and out[0] == "":
        out.pop(0)
    if page > 1 and head:
        out = [head.format(page=page, total=total), ""] + out
    if page < total:
        optional = next((i for i, lines in enumerate(pages, 1) if next((x for x in lines if x != ""), None) == TXT["sec_rest"]), None)
        key = "page_more_optional" if optional is not None and page + 1 >= optional else "page_more"
        out += ["", TXT[key].format(page=page, total=total, what=what, next=page + 1)]
    elif total > 1:
        out += ["", TXT["page_last"].format(total=total)]
    return out


def page_limit(cfg):
    """Bytes one printed output may take: PAGE_BYTES unless the configuration says otherwise (a harness whose inline
    limit was raised can take larger pages)."""
    value = int(os.environ.get("CARREL_SEARCH_PAGE_BYTES") or cfg.get("page_bytes") or PAGE_BYTES)
    if not 4000 <= value <= 400000:
        raise ValueError("page_bytes must be between 4000 and 400000")
    return value


def compact(envelope, path, limit=PAGE_BYTES, page=1):
    view = VIEWS.get(envelope["operation"])
    if view is None:
        return json.dumps(envelope, ensure_ascii=False, indent=2, allow_nan=False)
    cid = envelope["call_id"]
    head = TXT["page_head"].format(id=cid, op=envelope["operation"], page="{page}", total="{total}")
    out = one_page(pages_of(view(envelope, limit), limit), page, cid, head)
    return "\n".join(out + ["", TXT["full"].format(path=path)])


# ── requests ──

def payload(args, work=None):
    data = read_object(args.request) if getattr(args, "request", None) else {}
    ref = resolve(work, args.ref) if getattr(args, "ref", None) else None
    if args.command == "search":
        if args.question is not None:
            data["question"] = args.question
        if args.kb:
            data["kbs"] = args.kb
        hints = dict(data.get("hints") or {})
        if args.block_type:
            hints["block_types"] = args.block_type
        if args.rel_path:
            hints["rel_paths"] = args.rel_path
        named = list(data.get("kbs") or [])         # named by the caller (--kb or the request file): never widened
        for item in args.in_doc or []:
            kb, row = chunk_of(*resolve(work, item)[:3])
            if not row.get("doc_id"):
                raise ValueError("the referenced entry carries no document id: " + item)
            within(named, kb, "--in-doc " + item)
            hints.setdefault("doc_ids", []).append(row["doc_id"])
            if not named and kb not in data.setdefault("kbs", []):
                data["kbs"].append(kb)      # no knowledge base named: the documents pointed at decide
        if hints:
            data["hints"] = hints
        if args.top_k is not None:
            data["top_k"] = args.top_k
        if args.no_context:
            data["context"] = False
        if args.explain:
            data["explain"] = True
        if args.image:
            image = Path(args.image).expanduser()
            if image.stat().st_size > 9_000_000:
                raise ValueError("image is too large for the API's base64 limit")
            data["image_b64"] = base64.b64encode(image.read_bytes()).decode("ascii")
        if not isinstance(data.get("question"), str) or not data["question"].strip():
            raise ValueError("search requires --question or a question in --request")
    if args.command == "context":
        if ref is not None:
            kb, row = chunk_of(*ref[:3])
            if row.get("doc_id") is None or row.get("chunk_index") is None:
                raise ValueError("the referenced chunk carries no document position")
            first = last = int(row["chunk_index"])
            stitched = row.get("stitched") or {}
            if stitched.get("chunk_from") is not None and stitched.get("chunk_to") is not None:
                first, last = int(stitched["chunk_from"]), int(stitched["chunk_to"])
            lo, hi = max(0, first - args.before), last + args.after
            if args.whole:
                # the whole document; a row that does not say how long it is gets read until the chunks run out
                total = int(row.get("chunk_total") or 0)
                if total > CONTEXT_MAX:
                    raise ValueError(too_long(total))
                lo, hi = 0, (total or CONTEXT_MAX) - 1
            if hi - lo + 1 > CONTEXT_MAX:
                raise ValueError("one context command reads up to " + str(CONTEXT_MAX) + " chunks")
            data.update({"kb_id": kb, "doc_id": row["doc_id"], "chunk_from": lo, "chunk_to": hi})
            if row.get("content_version"):
                data["content_version"] = row["content_version"]
        if not data.get("kb_id") or not data.get("doc_id"):
            raise ValueError("context requires --ref or --request")
    if args.command == "crop":
        if ref is not None:
            kb, row = chunk_of(*ref[:3])
            data.update({"kb_id": kb, "point_id": row["point_id"]})
        if args.bbox:
            data["bbox"] = [float(x) for x in args.bbox.split(",")]
        if args.pad is not None:
            data["pad"] = args.pad
        if not data.get("kb_id") or not data.get("point_id") or not data.get("bbox"):
            raise ValueError("crop requires --ref with --bbox, or --request")
    if args.command == "neighbors":
        if ref is not None:
            data["kb_id"], data["entity_id"] = entity_of(*ref)
            within([args.kb] if args.kb else [], data["kb_id"], "--ref " + args.ref)
        if args.kb:
            data["kb_id"] = args.kb
        if args.entity:
            data["entity"] = args.entity
        if args.entity_id:
            data["entity_id"] = args.entity_id
        if args.limit is not None:
            data["limit"] = args.limit
        if args.type:
            data["types"] = args.type
        if args.direction:
            data["direction"] = args.direction
        if not data.get("kb_id") or not (data.get("entity") or data.get("entity_id")):
            raise ValueError("neighbors requires --ref, or --kb with one of --entity / --entity-id")
    if args.command in ("entities", "facts"):
        if args.command == "facts" and ref is not None:
            data["kb_id"], data["subject_id"] = entity_of(*ref)
            within([args.kb] if args.kb else [], data["kb_id"], "--ref " + args.ref)
        if args.kb:
            data["kb_id"] = args.kb
        for key in ("limit", "offset"):
            if getattr(args, key) is not None:
                data[key] = getattr(args, key)
        if not data.get("kb_id"):
            raise ValueError(args.command + " requires --kb" + (" or --ref" if args.command == "facts" else ""))
    if args.command == "entities":
        if args.type:
            data["types"] = args.type
        if args.parent_type:
            data["parent_types"] = args.parent_type
        if args.name:
            data["name"] = args.name
    if args.command == "facts":
        for key, field in (("subject", "subject"), ("subject_id", "subject_id"), ("property", "property"), ("match", "match")):
            if getattr(args, key):
                data[field] = getattr(args, key)
        if not (data.get("subject") or data.get("subject_id") or data.get("property")):
            raise ValueError("facts requires at least one of --ref / --subject / --subject-id / --property")
    return data


def request(url, token, timeout, method, route, data=None):
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    body = None
    if data is not None:
        body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    req = urllib.request.Request(url + route, data=body, headers=headers, method=method)
    opener = urllib.request.build_opener(NoRedirect())
    with opener.open(req, timeout=timeout) as response:
        return response.read(), response.headers


def too_long(total):
    return ("this document has " + str(total) + " chunks, --whole reads up to " + str(CONTEXT_MAX)
            + "; read a part of it with --before / --after, or search inside it with search --in-doc")


def read_context(url, token, timeout, data, whole=False):
    """/context serves a limited number of chunks per request: a longer range is read in consecutive requests and
    merged into one response, its sources numbered through. A whole-document read that started from a row without
    the document's length learns it from the first chunks: a document over the limit is refused then, not returned
    in part; when the length never shows, the response is marked as possibly incomplete."""
    lo, hi = data.get("chunk_from"), data.get("chunk_to")
    if not isinstance(lo, int) or not isinstance(hi, int) or (hi - lo < CONTEXT_STEP and not whole):
        return json.loads(request(url, token, timeout, "POST", "/context", data)[0])
    merged, total, start, ended = None, 0, lo, False
    while start <= hi:
        part = dict(data, chunk_from=start, chunk_to=min(start + CONTEXT_STEP - 1, hi))
        result = json.loads(request(url, token, timeout, "POST", "/context", part)[0])
        rows = result.get("sources") or []
        if merged is None:
            merged = result
            total = next((int(row["chunk_total"]) for row in rows if isinstance(row, dict) and row.get("chunk_total")), 0)
            if whole and total:
                if total > CONTEXT_MAX:
                    raise ValueError(too_long(total))
                hi = total - 1
        else:
            merged["sources"] = (merged.get("sources") or []) + rows
            merged["tokens_total"] = (merged.get("tokens_total") or 0) + (result.get("tokens_total") or 0)
        if len(rows) < part["chunk_to"] - part["chunk_from"] + 1:
            ended = True                # the document ends here
            break
        start += CONTEXT_STEP
    if whole and not total and not ended:
        merged["maybe_more"] = True
    for i, row in enumerate(merged.get("sources") or [], 1):
        row["n"] = i
    return merged


def save_new(path, content):
    destination = Path(path).expanduser().absolute()
    # Exclusive creation: a mistyped output path must not replace user data.
    with destination.open("xb") as f:
        f.write(content)
    return str(destination)


def new_call(work):
    for _ in range(20):
        call_id = uuid.uuid4().hex[:6]
        if not (work / (call_id + ".json")).exists():
            return call_id
    raise OSError("could not allocate a call id in " + str(work))


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="JSON config path; otherwise CARREL_SEARCH_CONFIG or ~/.config/carrel-search/config.json")
    p.add_argument("--base-url", help="Override service URL; default http://127.0.0.1:9810")
    p.add_argument("--timeout", type=float, help="HTTP timeout seconds (default 60); no automatic retries")
    sub = p.add_subparsers(dest="command", required=True)
    ref_help = "An entry of an earlier response: <call id>:<label>, e.g. 3fa2c1:S3"
    for name in ("health", "catalog", "search", "context", "image", "crop", "neighbors", "entities", "facts", "show"):
        cmd = sub.add_parser(name)
        if name == "show":
            cmd.add_argument("refs", nargs="+", help="A call id (prints its compact view again) or references such as 3fa2c1:S3")
            cmd.add_argument("--json", action="store_true", help="Print every field of a source or page as JSON")
            cmd.add_argument("--page", type=int, help="With a call id: the page of its compact view to print (default 1)")
            continue
        cmd.add_argument("--output", help="Also save to this new file (pictures: save here instead of the work directory)")
        cmd.add_argument("--json", action="store_true", help="Print the complete response instead of the compact view")
        if name in ("search", "context", "crop", "neighbors", "entities", "facts"):
            cmd.add_argument("--request", help="UTF-8 JSON request file, or - for stdin")
        if name in ("context", "image", "crop", "neighbors", "facts"):
            cmd.add_argument("--ref", help=ref_help)
        if name == "context":
            cmd.add_argument("--before", type=int, default=1, help="Chunks to read before the referenced one (default 1)")
            cmd.add_argument("--after", type=int, default=1, help="Chunks to read after the referenced one (default 1)")
            cmd.add_argument("--whole", action="store_true", help="Read the whole document the referenced entry is in")
        if name == "crop":
            cmd.add_argument("--bbox", help="x0,y0,x1,y1 as 0-1 ratios or 0-1000 per mille")
            cmd.add_argument("--pad", type=int)
        if name in ("entities", "facts"):
            cmd.add_argument("--kb", help="KB ID that has a graph")
            cmd.add_argument("--limit", type=int, choices=range(1, 201), metavar="1..200")
            cmd.add_argument("--offset", type=int, help="Skip this many rows (paging)")
        if name == "entities":
            cmd.add_argument("--type", action="append", help="Only these entity types; repeat to allow several")
            cmd.add_argument("--parent-type", action="append", help="Only these upper classes (entity / part / property / process / standard / document)")
            cmd.add_argument("--name", help="Text contained in the title or an alias")
        if name == "facts":
            cmd.add_argument("--subject", help="Subject title or alias (case and spaces ignored)")
            cmd.add_argument("--subject-id", help="Entity id from a previous response")
            cmd.add_argument("--property", help="Property name, symbol or concept name")
            cmd.add_argument("--match", choices=("auto", "exact", "contains"))
        if name == "neighbors":
            cmd.add_argument("--kb", help="KB ID that has a graph")
            cmd.add_argument("--entity", help="Entity title or alias (case and spaces ignored)")
            cmd.add_argument("--entity-id", help="Entity id from a previous response")
            cmd.add_argument("--limit", type=int, choices=range(1, 101), metavar="1..100")
            cmd.add_argument("--type", action="append", help="Only these predicates; repeat to allow several")
            cmd.add_argument("--direction", choices=("both", "out", "in"))
        if name == "search":
            cmd.add_argument("--question")
            cmd.add_argument("--kb", action="append", help="Explicit KB ID; repeat to select multiple")
            cmd.add_argument("--top-k", type=int, choices=range(1, 51), metavar="1..50")
            cmd.add_argument("--image", help="Local query image file; sent to /search only")
            cmd.add_argument("--no-context", action="store_true")
            cmd.add_argument("--explain", action="store_true")
            cmd.add_argument("--block-type", action="append", help="Prefer chunks of this kind (table, image, ...); a soft preference, repeatable")
            cmd.add_argument("--in-doc", action="append", metavar="REF", help="Search only inside the document of this entry (e.g. 3fa2c1:S3); repeatable")
            cmd.add_argument("--rel-path", action="append", help="Search only this document path inside the knowledge base; repeatable")
        if name == "image":
            cmd.add_argument("--kb")
            cmd.add_argument("--point-id")
    return p


def show(work, refs, raw=False, page=None, limit=PAGE_BYTES):
    calls = [ref for ref in refs if re.fullmatch(r"[0-9a-f]{6}", ref)]
    if page is not None and calls and len(refs) > 1:
        raise ValueError("--page goes with one call id (show 3fa2c1 --page 2) or with entry references only")
    lines = []
    for ref in refs:
        if ref in calls:
            print(compact(stored(work, ref), work / (ref + ".json"), limit, page or 1))
            continue
        envelope, letter, entry, child = resolve(work, ref)
        if letter == "S" and not raw:
            parent = "S" + str(entry.get("of")) if entry.get("role") != "hit" and entry.get("of") else None
            lines.append("[" + ref + "] " + str(entry.get("cite") or "") + notes_of(source_notes(entry, ref, parent)))
            lines.append(str(entry.get("text") or "").strip())
        elif letter == "P" and not raw:
            docs = entry.get("docs_cite") or []
            lines.append(cited("[" + ref + "] " + TXT["dot"].join(str(b) for b in (entry.get("kind"), entry.get("title")) if b) + TXT["colon"]
                               + flat(entry.get("summary")), TXT["semi"].join(docs)))
            lines.append(str(entry.get("text") or "").strip())
        else:
            lines.append(TXT["shown"].format(ref=ref))
            lines.append(json.dumps(child if child is not None else entry, ensure_ascii=False, indent=1, allow_nan=False))
    if lines:
        # Entries shown together are paged like a view: the same references with --page print the next page.
        print("\n".join(one_page(pages_of(lines, limit), page or 1, " ".join(ref for ref in refs if ref not in calls))))
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    token = ""
    call_id = "-"
    try:
        cfg = config(args)
        work = work_dir(cfg)
        limit = page_limit(cfg)
        if args.command == "show":
            return show(work, args.refs, args.json, args.page, limit)
        call_id = new_call(work)
        if args.output and Path(args.output).expanduser().exists():
            raise FileExistsError("output already exists; choose a new filename")
        url, token, timeout = settings(args, cfg)
        binary = args.command in ("image", "crop")
        tag = None
        if args.command == "image":
            kb, point = args.kb, args.point_id
            if args.ref:
                kb, row = chunk_of(*resolve(work, args.ref)[:3])
                within([args.kb] if args.kb else [], kb, "--ref " + args.ref)
                point, tag = row["point_id"], args.ref.replace(":", "-")
            if not kb or not point:
                raise ValueError("image requires --ref, or --kb with --point-id")
            quote = lambda s: urllib.parse.quote(s, safe="")
            route = "/image/" + quote(kb) + "/" + quote(point)
            method, data = "GET", None
        elif args.command in ("neighbors", "entities", "facts"):
            route, method, data = "/graph/" + args.command, "POST", payload(args, work)
        elif args.command in ("search", "context", "crop"):
            route, method, data = "/" + args.command, "POST", payload(args, work)
            if args.command == "crop" and args.ref:
                tag = args.ref.replace(":", "-") + "-crop"
        else:
            route, method, data = "/" + args.command, "GET", None
        envelope = {"call_id": call_id, "operation": args.command}
        if args.command == "context":
            raw, headers = None, None
            result = read_context(url, token, timeout, data, bool(args.whole and args.ref))
        else:
            raw, headers = request(url, token, timeout, method, route, data)
            result = None if binary else json.loads(raw)
        if binary:
            mime = headers.get_content_type()
            if not mime.startswith("image/"):
                raise ValueError("server did not return an image; no file saved")
            ext = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/gif": "gif", "image/bmp": "bmp"}.get(mime, "img")
            target = args.output or str(work / (call_id + "-" + (tag or args.command) + "." + ext))
            envelope["result"] = {
                "path": save_new(target, raw), "mime_type": mime,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "source": headers.get("X-Image-Source"),
                "width": headers.get("X-Image-Width"), "height": headers.get("X-Image-Height"),
                "crop_box": headers.get("X-Crop-Box"),
            }
            if args.json:
                print(json.dumps(envelope, ensure_ascii=False, indent=2, allow_nan=False))
            else:
                res = envelope["result"]
                print(TXT["saved"].format(path=res["path"], mime=mime, w=res["width"], h=res["height"], source=res["source"]))
            return 0
        envelope["request"] = {k: v for k, v in (data or {}).items() if k != "image_b64"}
        envelope["result"] = result
        add_cites(envelope["result"])
        number_rows(envelope["result"])
        encoded = (json.dumps(envelope, ensure_ascii=False, indent=1, allow_nan=False) + "\n").encode("utf-8")
        path = save_new(work / (call_id + ".json"), encoded)
        if args.output:
            save_new(args.output, encoded)
        print(json.dumps(envelope, ensure_ascii=False, indent=2, allow_nan=False) if args.json else compact(envelope, path, limit))
        return 0
    except (OSError, ValueError, TypeError, KeyError, urllib.error.URLError) as exc:
        if isinstance(exc, urllib.error.HTTPError):
            message = f"HTTP {exc.code}"
            if exc.code in (401, 403):
                message += ": check the configured bearer token; do not print it"
            else:
                detail = exc.read(2000).decode("utf-8", errors="replace")
                message += ": " + detail[:600]
        else:
            message = f"{type(exc).__name__}: {exc}"
        if token:
            message = message.replace(token, "[redacted]")
        print(json.dumps({"call_id": call_id, "operation": args.command, "error": message}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
