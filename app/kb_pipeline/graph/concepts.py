"""Property concepts: the property / symbol of facts within one KB is mapped to a canonical concept key (plan
5.A, the source of OG-RAG's "facts hang off fixed properties" effect).

Three steps, in the same shape as entity resolution: normalisation (NFKC, casefold, strip whitespace and
underscores, restore LaTeX, symbol first) merges directly -> vector nearest neighbours propose candidate
pairs (only between concepts with compatible units) -> the model judges sameness in batches -> union-find.
The concept key is derived from the representative name, so building the same corpus twice yields the same
keys; the concept table is written into graph.json, facts carry concept / concept_key, and the timeline pages
and reconciliation group by it. There are no domain words in the pipeline: the health KB folds "total
cholesterol / TC / total cholesterol measurement result" into one concept and a datasheet folds
"V_CC / VCC / Supply voltage" into one, using the same rules and the same judging.
"""
from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter
from typing import Any, Callable, Iterable

from . import prompts
from .extract import strip_latex
from .llm import ChatClient, LLMCallError

CONCEPT_EMBED_AUTO = 0.95          # cosine >= this value: merge directly
CONCEPT_EMBED_ASK = 0.82           # cosine in [ASK, AUTO): ask the model
CONCEPT_MAX_NEIGHBOURS = 3
CONCEPT_JUDGE_BATCH = 60
CONCEPT_MAX_CANDIDATES = 600       # above this many concepts only normalise and auto-merge, never ask the model (cost fallback)
_STRIP_RE = re.compile(r"[\s_\-·•.。,,;;:：()()\[\]【】/\\|\"'“”‘’]+")
_TRAILING_WORDS = ("测量结果", "检查结果", "检测结果", "结果", "值", "数值", "参数", "指标", "measurement", "measured value", "value", "result", "parameter")
_ANSWER_RE = re.compile(r"^\s*(\d+)\s*[\.:：\)]\s*(yes|no|是|否|y|n)\b", re.IGNORECASE | re.MULTILINE)


def concept_norm(prop: Any, symbol: Any = "") -> str:
    """Canonical key: the symbol takes precedence when present (VCC / V_CC / $V_{CC}$ are all vcc); otherwise the
    property name, casefolded after stripping whitespace, underscores, punctuation and trailing words such as
    "measurement result / value"."""
    sym = strip_latex(unicodedata.normalize("NFKC", str(symbol or ""))).strip()
    if sym:
        key = _STRIP_RE.sub("", sym).casefold()
        if key:
            return key
    text = strip_latex(unicodedata.normalize("NFKC", str(prop or ""))).strip().casefold()
    for w in sorted(_TRAILING_WORDS, key=len, reverse=True):
        w = w.casefold()
        if text.endswith(w) and len(text) > len(w) + 1:
            text = text[: -len(w)].strip()
    return _STRIP_RE.sub("", text)


def concept_key(norm: str) -> str:
    return "c" + hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


# Identity tokens: codes mixing digits and letters (rs0000133, ZK7C1049GN, TP53, COVID-19). Two candidates that
# each carry tokens with none in common are not the same metric, however alike their names and vectors are
# (Codex review F02: rs0000133 and rs0000131 were merged into one concept, creating a false conflict). Pure
# numbers (years, section numbers) do not count as tokens.
_IDENT_RE = re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z]{1,8}[-_]?\d[A-Za-z0-9_.\-]*|\d+[A-Za-z][A-Za-z0-9_\-]*)(?![A-Za-z0-9])")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
_LATIN_WORD_RE = re.compile(r"[A-Za-z]{3,}")


def _identity_tokens(label: str, symbols: str = "") -> set[str]:
    return {t.casefold() for t in _IDENT_RE.findall(f"{label} {symbols}")}


_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z]+|\d+|[^\sA-Za-z0-9\u4e00-\u9fff]")


def _contrast(a: str, b: str) -> bool:
    """Two labels that differ by a single short token: left/right, I/II, +/-, mild/severe, vein/artery, v7 2407 /
    v7 2407b. The names are alike enough, but the difference is exactly the token that tells them apart -- never
    auto-merge, only the model may decide (2026-09-12: the health KB auto-merged corrected vision (right)/(left)
    and pepsinogen I/II by cosine, the library KB the CG coefficients j=j1+1/2 and j=j1-1/2). Domain-independent:
    looks at token form only."""
    ta, tb = _TOKEN_RE.findall(str(a or "")), _TOKEN_RE.findall(str(b or ""))
    if not ta or not tb or ta == tb:
        return False
    if len(ta) == len(tb):
        diff = [(x, y) for x, y in zip(ta, tb) if x != y]
        if len(diff) != 1:
            return False
        x, y = diff[0]
        if len(x) <= 3 and len(y) <= 3:
            return True                                              # different short tokens (left/right, I/II, +/-, 2407/2407b)
        return len(x) == len(y) and sum(1 for p, q in zip(x, y) if p != q) == 1    # same length, one character apart (vein/artery, mild/severe)
    if abs(len(ta) - len(tb)) == 1:
        longer, shorter = (ta, tb) if len(ta) > len(tb) else (tb, ta)
        for k in range(len(longer)):
            if longer[:k] + longer[k + 1:] == shorter:
                extra = longer[k]
                if not re.search(r"[\w\u4e00-\u9fff]", extra):
                    return False                                     # only an extra hyphen / punctuation (F distribution / F-distribution): spelling, not a token
                return len(extra) <= 2                               # one extra short token of <= 2 characters (version suffix, side)
    return False


def _measure_word(label: str) -> str:
    """The last word of a label after removing identity tokens (genotype / population share / risk status; voltage /
    frequency): different quantities of the same object."""
    rest = _IDENT_RE.sub(" ", str(label or ""))
    cjk = _CJK_RUN_RE.findall(rest)
    if cjk:
        return cjk[-1]
    words = _LATIN_WORD_RE.findall(rest)
    return words[-1].casefold() if words else ""


def compatible(a: dict[str, Any], b: dict[str, Any]) -> str:
    """Whether two concept candidates may merge: conflicting identity tokens -> "identity" (neither merged nor
    asked); different measure words -> "measure" (no auto-merge, the model decides); otherwise "ok"."""
    ia, ib = _identity_tokens(a["label"], a.get("symbols_text", "")), _identity_tokens(b["label"], b.get("symbols_text", ""))
    if ia and ib and ia.isdisjoint(ib):
        return "identity"
    if bool(ia) != bool(ib):
        return "identity"                 # identity token on one side only (GENEX rs0000133 genotype / genotype): merging would drop the identity
    la, lb = re.sub(r"\s+", "", str(a["label"] or "")).casefold(), re.sub(r"\s+", "", str(b["label"] or "")).casefold()
    if la != lb and _contrast(la, lb):
        return "contrast"                 # differ by one short token: no auto-merge, the model decides
    if la != lb and re.findall(r"\d+", la) != re.findall(r"\d+", lb):
        return "contrast"                 # different digit sequences (edition 2 / 3, CG coefficients j=j1+1/2 / j=j1,m2=1): no auto-merge, model decides
    if la and lb and (la in lb or lb in la):
        return "ok"                       # one label fully contains the other (Address setup / Address setup time): short and long forms of one quantity
    ma, mb = _measure_word(a["label"]), _measure_word(b["label"])
    if ma and mb and ma != mb and ma not in mb and mb not in ma:
        return "measure"
    return "ok"


def _unit_family(unit: str) -> str:
    """Coarse unit compatibility: same canonical unit, or one side has no unit. The canonical key keeps case
    (final review F04)."""
    from ..measure_units import unit_key

    return unit_key(unit)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5 or 1.0
    nb = sum(x * x for x in b) ** 0.5 or 1.0
    return dot / (na * nb)


def _neighbour_pairs_python(vectors: list[list[float]], cands: list[dict[str, Any]], fams: list[str],
                            k: int) -> list[tuple[int, int, float]]:
    """The original pairwise-cosine implementation: the fallback when vectors have uneven lengths (test stubs), and
    the semantic definition of _neighbour_pairs."""
    by_unit: dict[str, list[int]] = {}
    for i, fam in enumerate(fams):
        by_unit.setdefault(fam, []).append(i)
    out: list[tuple[int, int, float]] = []
    for i, fam in enumerate(fams):
        pool = [j for j in by_unit.get(fam, []) if j != i]
        if fam:
            pool += by_unit.get("", [])
        else:
            pool = [j for j in range(len(cands)) if j != i]
        scored = sorted(((_cosine(vectors[i], vectors[j]), j) for j in pool), reverse=True)[:k]
        out.extend((i, j, sim) for sim, j in scored)
    return out


def _neighbour_pairs(vectors: list[list[float]], cands: list[dict[str, Any]], k: int) -> list[tuple[int, int, float]]:
    """For every candidate take the top-k neighbours by cosine within its comparable pool (same unit family plus
    the unit-less ones; everything when it has no unit itself); returns (i, j, sim) with i ascending and, within
    each i, (sim, j) descending -- the same order as the pairwise implementation.
    2026-09-12, product-documentation KB: 48,201 facts and 12,933 concept candidates; pure-Python pairwise meant
    167 million cosines, about 3 hours, with the build stuck motionless at "property concept normalisation"; this
    is now a blocked numpy matrix product (256 rows per block, 26 MB of memory) and takes seconds."""
    n = len(cands)
    fams = [_unit_family(c["unit"]) for c in cands]
    try:
        import numpy as np
        mat = np.asarray(vectors, dtype=np.float64)
        if mat.ndim != 2 or mat.shape[0] != n:
            raise ValueError("ragged vectors")
    except (ImportError, ValueError, TypeError):
        return _neighbour_pairs_python(vectors, cands, fams, k)
    norms = np.linalg.norm(mat, axis=1)
    norms[norms == 0] = 1.0
    unit = mat / norms[:, None]
    fam_ids = {f: idx for idx, f in enumerate(dict.fromkeys(fams))}
    fam_of = np.array([fam_ids[f] for f in fams])
    no_unit = fam_of == fam_ids.get("", -1)
    allowed = np.zeros((len(fam_ids), n), dtype=bool)          # the columns each family may compare against
    for fam, idx in fam_ids.items():
        allowed[idx] = True if fam == "" else ((fam_of == idx) | no_unit)
    kk = max(1, min(int(k), n - 1))
    out: list[tuple[int, int, float]] = []
    block = 256
    for start in range(0, n, block):
        stop = min(n, start + block)
        sims = unit[start:stop] @ unit.T
        sims[~allowed[fam_of[start:stop]]] = -np.inf
        sims[np.arange(stop - start), np.arange(start, stop)] = -np.inf        # exclude self
        cols = np.broadcast_to(np.arange(n), sims.shape)
        top = np.lexsort((-cols, -sims), axis=1)[:, :kk]           # similarity descending, ties by j descending: same order as the pairwise implementation
        for r in range(stop - start):
            scored = sorted(((float(sims[r, j]), int(j)) for j in top[r]), reverse=True)
            out.extend((start + r, j, sim) for sim, j in scored if math.isfinite(sim))
    return out


def render_judge_batch(pairs: list[tuple[int, int]], cands: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for i, (a, b) in enumerate(pairs, 1):
        ca, cb = cands[a], cands[b]
        lines.append(f"{i}. \"{ca['label']}\" ({ca.get('symbols_text') or '-'}; unit {ca.get('unit') or '-'}; e.g. {ca.get('example') or '-'})"
                     f"  vs  \"{cb['label']}\" ({cb.get('symbols_text') or '-'}; unit {cb.get('unit') or '-'}; e.g. {cb.get('example') or '-'})")
    return prompts.CONCEPT_JUDGE_PROMPT.format(pairs="\n".join(lines))


def parse_answers(text: str, count: int) -> list[bool]:
    out = [False] * count
    for m in _ANSWER_RE.finditer(str(text or "")):
        idx = int(m.group(1)) - 1
        if 0 <= idx < count:
            out[idx] = m.group(2).lower() in ("yes", "是", "y")
    return out


SHORT_SYMBOL_MAX_CHARS = 2         # symbols this short after normalisation (n / N / t / F / p) recur across disciplines; alone they cannot decide identity


def _bucket_norm(f: dict[str, Any]) -> str:
    """Which candidate bucket a fact goes into: the canonical key (symbol first); identity tokens in the property
    (loci, part numbers) that are not in the symbol are added to the bucket key -- the same symbol with different
    loci must not be merged already at bucketing (Codex re-review N05).
    Short symbols (<= SHORT_SYMBOL_MAX_CHARS characters) also take the property name: principal quantum number n
    vs period count N, node time t vs t value share a symbol but mean entirely different things; they used to
    become one concept right at bucketing, and neither the compatibility check nor the judging could stop it
    afterwards (Codex 2026-09-13 F02).
    The same quantity with slightly different property spellings first lands in several buckets and is merged
    back by vector similarity + judging later; when in doubt, keep one concept too many."""
    norm = concept_norm(f.get("property"), f.get("symbol"))
    if norm and f.get("symbol"):
        sym = str(f.get("symbol") or "").strip()
        sym_key = _STRIP_RE.sub("", strip_latex(unicodedata.normalize("NFKC", sym))).casefold()
        if len(sym_key) <= SHORT_SYMBOL_MAX_CHARS:
            prop = str(f.get("property") or "")
            # the property name ends with the symbol itself ("principal quantum number n", "node time t (years)"):
            # strip it before normalising so it shares the bucket with "principal quantum number"
            prop = re.sub(r"[\s(（]*" + re.escape(sym) + r"[)）]?\s*$", "", prop, flags=re.IGNORECASE).strip() or prop
            prop_norm = concept_norm(prop)
            if prop_norm and prop_norm != sym_key:
                norm = f"{norm}|{prop_norm}"
        ident = _identity_tokens(str(f.get("property") or ""))
        if ident and ident.isdisjoint(_identity_tokens(str(f.get("symbol") or ""))):
            norm = f"{norm}|{'+'.join(sorted(ident))}"       # identity tokens last: the short-symbol bucket-merge rule reads only the "symbol|property" parts
    return norm


def _short_symbol_parts(norm: str) -> tuple[str, str] | None:
    """Split a short-symbol bucket key "symbol|property"; returns None for buckets that are not short-symbol ones
    or that carry an identity token (three parts)."""
    parts = norm.split("|")
    if len(parts) != 2 or len(parts[0]) > SHORT_SYMBOL_MAX_CHARS or not parts[1]:
        return None
    return parts[0], parts[1]


def build_concepts(
    facts: list[dict[str, Any]],
    *,
    embed: Callable[[list[str]], list[list[float]]] | None = None,
    client: ChatClient | None = None,
    auto_threshold: float = CONCEPT_EMBED_AUTO,
    ask_threshold: float = CONCEPT_EMBED_ASK,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Write concept / concept_key into every fact; returns (concept table, stats). facts is modified in place."""
    stats: dict[str, Any] = {"facts": len(facts), "norms": 0, "concepts": 0, "auto_merged": 0, "asked_pairs": 0,
                             "judged_yes": 0, "judge_batches": 0, "judge_failed": 0, "skipped_llm": False,
                             "blocked_identity": 0, "measure_to_judge": 0, "contrast_to_judge": 0, "blocked_bridge": 0,
                             "symbol_contained": 0, "symbol_to_judge": 0}
    merges: list[dict[str, Any]] = []        # merge log: which two, which route, similarity (entity resolution had an audit; concept merging did not)
    blocked: list[dict[str, Any]] = []
    groups: dict[str, dict[str, Any]] = {}
    for f in facts:
        norm = _bucket_norm(f)
        if not norm:
            continue
        slot = groups.get(norm)
        if slot is None:
            slot = groups[norm] = {"norm": norm, "labels": Counter(), "symbols": Counter(), "units": Counter(),
                                   "count": 0, "docs": set(), "example": ""}
        slot["labels"][str(f.get("property") or "")] += 1
        if f.get("symbol"):
            slot["symbols"][str(f["symbol"])] += 1
        unit = str(f.get("unit_canonical") or f.get("unit") or "")
        if unit:
            slot["units"][unit] += 1
        slot["count"] += 1
        if f.get("doc_id"):
            slot["docs"].add(str(f["doc_id"]))
        if not slot["example"]:
            from .facts import values_text
            slot["example"] = f"{f.get('subject')}: {values_text(f)}"[:80]
    cands = list(groups.values())
    for c in cands:
        c["label"] = c["labels"].most_common(1)[0][0]
        c["symbols_text"] = ", ".join(s for s, _ in c["symbols"].most_common(3))
        c["unit"] = c["units"].most_common(1)[0][0] if c["units"] else ""
    stats["norms"] = len(cands)
    uf = _UnionFind(len(cands))
    root_ident: dict[int, set[str]] = {i: _identity_tokens(c["label"], c.get("symbols_text", "")) for i, c in enumerate(cands)}

    def merge(a: int, b: int, via: str, sim: float | None = None) -> bool:
        """Identity exclusion at the set level (the bridging path of Codex re-review N05): A (rs0000133) and C
        (rs0000131) can each merge with the generic concept B, but landing in the same set fuses two loci into
        one. Two sets that each carry tokens with none in common are not merged and go into the block log; better
        to keep one near-synonymous concept too many."""
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb:
            return True
        ia, ib = root_ident.get(ra, set()), root_ident.get(rb, set())
        extra = {"sim": round(sim, 3)} if sim is not None else {}
        if ia and ib and ia.isdisjoint(ib):
            stats["blocked_bridge"] += 1
            if len(blocked) < 200:
                blocked.append({"a": cands[a]["label"], "b": cands[b]["label"], "reason": "bridge", **extra})
            return False
        uf.union(a, b)
        root_ident[uf.find(a)] = ia | ib
        if len(merges) < 500:
            merges.append({"a": cands[a]["label"], "b": cands[b]["label"], "via": via, **extra})
        return True

    # Re-merging buckets after the short-symbol split: same short symbol, both sides have a unit in the same unit
    # family, and one property name fully contains the other. This route must also pass the compatible() guard
    # (Codex 2026-09-14 R03: "current / leakage current", same symbol I and same unit A, were merged outright):
    # identity-token conflicts (price / price at point P2) are always blocked; pairs the guard calls ok merge
    # directly; so do pairs whose extra part is only a prefix / suffix of <= 3 Latin letters (HR uric acid / uric
    # acid: the lab sheet's instrument / method code, same symbol UA and same unit); the rest (leakage current,
    # annual interest rate) go to the judging model, with "same symbol and unit" added as evidence for the
    # candidate. Buckets whose property name is just the symbol itself ("t") and buckets with identity tokens do
    # not take part.
    forced_ask: list[tuple[int, int]] = []
    by_symbol: dict[tuple[str, str], list[int]] = {}
    for i, c in enumerate(cands):
        parts = _short_symbol_parts(c["norm"])
        if parts is None or not c["unit"]:
            continue
        by_symbol.setdefault((parts[0], _unit_family(c["unit"])), []).append(i)
    for members in by_symbol.values():
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                a, b = members[x], members[y]
                pa, pb = _short_symbol_parts(cands[a]["norm"])[1], _short_symbol_parts(cands[b]["norm"])[1]
                if pa == pb or not (pa in pb or pb in pa):
                    continue
                verdict = compatible(cands[a], cands[b])
                if verdict == "identity":
                    stats["blocked_identity"] += 1
                    if len(blocked) < 200:
                        blocked.append({"a": cands[a]["label"], "b": cands[b]["label"], "reason": "identity", "via": "symbol_contain"})
                    continue
                extra = (pb.replace(pa, "", 1) if pa in pb else pa.replace(pb, "", 1))
                if verdict == "ok" or re.fullmatch(r"[a-z]{1,3}", extra):
                    if merge(a, b, "symbol_contain"):
                        stats["symbol_contained"] += 1
                else:
                    stats["symbol_to_judge"] += 1
                    forced_ask.append((min(a, b), max(a, b)))

    if embed is not None and len(cands) >= 2:
        texts = [f"{c['label']} ({c['symbols_text']})" if c["symbols_text"] else c["label"] for c in cands]
        try:
            vectors = embed(texts)
        except Exception as exc:
            print(f"[graph] concept embeddings unavailable, keeping normalized keys only: {exc!r}", flush=True)
            vectors = []
        ask: list[tuple[int, int]] = list(forced_ask)
        if len(vectors) == len(cands):
            seen_pairs: set[tuple[int, int]] = set(forced_ask)
            for i, j, sim in _neighbour_pairs(vectors, cands, CONCEPT_MAX_NEIGHBOURS):
                pair = (min(i, j), max(i, j))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                if sim < ask_threshold:
                    continue
                verdict = compatible(cands[i], cands[j])
                if verdict == "identity":
                    stats["blocked_identity"] += 1
                    if len(blocked) < 200:
                        blocked.append({"a": cands[i]["label"], "b": cands[j]["label"], "reason": "identity", "sim": round(sim, 3)})
                    continue
                if sim >= auto_threshold and verdict == "ok":
                    if merge(i, j, "auto", sim):
                        stats["auto_merged"] += 1
                else:
                    if sim >= auto_threshold:
                        stats["contrast_to_judge" if verdict == "contrast" else "measure_to_judge"] += 1   # alike enough but measure word / token differs: model decides
                    ask.append(pair)
        if ask and client is not None and len(cands) <= CONCEPT_MAX_CANDIDATES:
            stats["asked_pairs"] = len(ask)
            for start in range(0, len(ask), CONCEPT_JUDGE_BATCH):
                batch = ask[start:start + CONCEPT_JUDGE_BATCH]
                stats["judge_batches"] += 1
                try:
                    answers = parse_answers(client.chat(render_judge_batch(batch, cands), max_tokens=1024), len(batch))
                except LLMCallError as exc:
                    stats["judge_failed"] += 1
                    print(f"[graph] concept judge batch failed: {exc!r}", flush=True)
                    continue
                for (a, b), same in zip(batch, answers):
                    if same:
                        stats["judged_yes"] += 1
                        merge(a, b, "judge")
        elif ask:
            stats["skipped_llm"] = True
    stats["merges"] = merges
    stats["blocked"] = blocked
    members: dict[int, list[int]] = {}
    for i in range(len(cands)):
        members.setdefault(uf.find(i), []).append(i)
    concepts: list[dict[str, Any]] = []
    norm_to_key: dict[str, str] = {}
    norm_to_label: dict[str, str] = {}
    for root, idxs in members.items():
        rows = [cands[i] for i in idxs]
        rep = max(rows, key=lambda c: (c["count"], -len(c["label"]), c["norm"]))
        key = concept_key(rep["norm"])
        labels = Counter()
        symbols = Counter()
        units = Counter()
        docs: set[str] = set()
        count = 0
        for c in rows:
            labels.update(c["labels"])
            symbols.update(c["symbols"])
            units.update(c["units"])
            docs |= c["docs"]
            count += c["count"]
            norm_to_key[c["norm"]] = key
            norm_to_label[c["norm"]] = rep["label"]
        concepts.append({
            "key": key, "label": rep["label"], "norms": sorted(c["norm"] for c in rows),
            "aliases": [l for l, _ in labels.most_common(12) if l != rep["label"]],
            "symbols": [s for s, _ in symbols.most_common(6)], "units": [u for u, _ in units.most_common(4)],
            "facts": count, "docs": sorted(docs),
        })
    concepts.sort(key=lambda c: (-c["facts"], c["label"]))
    for f in facts:
        norm = _bucket_norm(f)
        if norm and norm in norm_to_key:
            f["concept_key"] = norm_to_key[norm]
            f["concept"] = norm_to_label[norm]
    stats["concepts"] = len(concepts)
    stats["cross_doc_concepts"] = sum(1 for c in concepts if len(c["docs"]) >= 2)
    return concepts, stats
