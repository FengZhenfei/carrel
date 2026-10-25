"""Entity resolution: coarse candidate screening → one LLM same-entity call per batch of 100 pairs → pairs
judged yes are merged as whole connected components.

Moved in from the old pipeline's resolution workflow; the
rules come from an actual reading of RAGFlow's entity_resolution.py: differing digit-bearing 2-grams → no
outright (ZK7C4021KV13 and ZK7C4041KV13 are two devices); English by edit distance ≤ min(len)//2; Chinese
changed to the Dice coefficient of **ordered** character bigrams ≥ 0.7 (the original bag of characters judged
"power management" and its reversed word order as the same). Scale is handled by coarse screening with "same
type + sliding window over normalized sorted titles".

The same-entity prompt carries the first sentence of both descriptions: giving only names and types is one of
the sources of RAGFlow's misjudgements.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable, Iterable

from .llm import ChatClient, LLMCallError

WINDOW = 40
BATCH_SIZE = 100
# Same-entity batch concurrency used to have its own cap of 5 (set conservatively for rate limits); user decision
# 2026-09-12: batches are independent, so follow the global concurrency (KB_GRAPH_LLM_CONCURRENCY, i.e.
# client.workers) and this step is no longer the slowest stretch on large KBs
# Vector candidates: same-type title pairs with cosine ≥ threshold go into the judgement batches (catches the same
# name across Chinese / English); each entity takes at most a few neighbours
EMBED_THRESHOLD = 0.9
EMBED_NEIGHBOURS = 3
# Type words (names equal after removing them → the same thing: tASH parameter = tASH, "reset signal" = "reset")
# are no longer hard-coded: English ones come from the ontology type names (type_words_of), corpus-language ones
# from the profile's type_words (induced in the suggestion phase, editable in the console). The old in-code list
# (parameter / signal / pin / register …) was electronics / software vocabulary and carried only 3 merges across
# two KBs (2026-09-08 generality review).
# Suffix words of organization names: Northwind / Northwind Technologies / Northwind AG are one company; only
# equal after removal counts as the same-name qualification
SUFFIX_WORDS = {
    "inc", "inc.", "ltd", "ltd.", "llc", "corp", "corp.", "corporation", "co", "co.", "company", "ag", "gmbh", "plc", "sa",
    "technologies", "technology", "holdings", "group", "limited", "公司", "有限公司", "股份有限公司", "集团", "股份", "科技",
}
# Polarity word pairs (general language words, not domain words): one side containing X and the other its
# antonym Y means not the same thing; "data input / Data Output" was once judged one entity (2026-09-08 kb_001).
# English matches on word boundaries, Chinese by substring
POLARITY_PAIRS = [
    ("input", "output"), ("in", "out"), ("输入", "输出"), ("high", "low"), ("高", "低"), ("min", "max"), ("minimum", "maximum"),
    ("最小", "最大"), ("rising", "falling"), ("rise", "fall"), ("上升", "下降"), ("read", "write"), ("读", "写"),
    ("enable", "disable"), ("使能", "禁用"), ("source", "sink"), ("positive", "negative"), ("阳性", "阴性"), ("正", "负"),
    ("left", "right"), ("左", "右"), ("before", "after"), ("前", "后"), ("increase", "decrease"), ("增高", "降低"), ("升高", "降低"),
    ("increased", "decreased"), ("up", "down"), ("on", "off"), ("open", "close"), ("start", "stop"), ("start", "end"), ("开始", "结束"),
    ("male", "female"), ("男", "女"), ("upper", "lower"), ("top", "bottom"), ("internal", "external"), ("内", "外"),
    ("primary", "secondary"), ("active", "standby"), ("set", "reset"), ("first", "last"), ("acute", "chronic"), ("急性", "慢性"),
]
_IDENT_STRIP_RE = re.compile(r"[\s_\-\.·/:]+")
_WORD_RE = re.compile(r"[a-z0-9]+")
_PAREN_RE = re.compile(r"\s*[(（]([^()（）]{1,48})[)）]\s*")
_TYPE_STOP = {"of", "and", "or", "the", "a", "an", "in", "for", "to"}


def type_words_of(etype: Any) -> set[str]:
    """Words in an ontology type name are type words too: with an entity type named package, "361-ball FCBGA
    Package" and "361-ball FCBGA" are the same; with laboratory test, the test in "blood routine test" can go. A
    general rule, no domain word list."""
    return {w for w in re.split(r"[\s_\-/]+", str(etype or "").casefold()) if len(w) >= 2 and w not in _TYPE_STOP}


_LATIN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")


def _expands(base: str, inner: str) -> bool:
    """Whether the bracketed text expands the symbol: the initials of the English words in the expansion (Forward
    Deployed Engineer → DSE), or its upper-case letters joined (AI Platform → AIP), equal the letters of the
    symbol. Condition notes (ΘJA (With Still Air), HSB (STORE only)) do not match and do not count."""
    letters = "".join(ch for ch in base if ch.isascii() and ch.isalpha()).upper()
    words = _LATIN_WORD_RE.findall(inner)
    if len(letters) < 2 or not words:
        return False
    initials = "".join(w[0] for w in words).upper()
    uppers = "".join(ch for ch in inner if ch.isascii() and ch.isupper())
    return letters in (initials, uppers)


def paren_alias(title: str) -> tuple[str, str]:
    """Two spellings of X(Y) count as aliases: "body mass index (BMI)" where Y is a symbol / abbreviation; "DSE
    (Data-Systems Engineer)" where X is the symbol and Y its expansion (English initials or upper-case letters
    joined equal X). Returns (X, Y), otherwise ("", ""). Brackets holding an example, an explanation or a condition
    (dark vegetables (red cabbage), stroke (ischemic stroke), ΘJA (With Still Air)) are not aliases."""
    text = unicodedata.normalize("NFKC", str(title or "")).strip()
    m = _PAREN_RE.search(text)
    if not m:
        return "", ""
    inner = m.group(1).strip()
    base = (text[: m.start()] + text[m.end():]).strip()
    if not base or not inner:
        return "", ""
    if not re.search(r"[A-Za-z\u4e00-\u9fff]", inner):
        return "", ""        # a year / bare number in brackets (Moore and McCabe (1998)): not an alias, must not key-merge with other (1998)s
    if is_identifier_like(inner):
        return base, inner
    if is_identifier_like(base) and _expands(base, inner):
        return base, inner
    return "", ""


def _strip_cjk_type_suffix(text: str, words: set[str]) -> str:
    """Strip one trailing CJK type word (reset signal → reset); left alone when the whole name is a type word."""
    for w in sorted((w for w in words if not w.isascii()), key=len, reverse=True):
        if len(text) - len(w) >= 2 and text.endswith(w):
            return text[:-len(w)].strip()
    return text


# Dice floor for ordered CJK character bigrams: a 7- vs 6-character spelling of "non-volatile memory" scores 0.73
# (passes); "power management" vs its reversed word order scores 0.67 (fails)
DICE_THRESHOLD = 0.7
_CJK_RE = re.compile(r"[一-鿿]")
_ANSWER_RE = re.compile(r"^\s*(\d+)\s*[\.:：\)]\s*(yes|no|是|否|y|n)\b", re.IGNORECASE | re.MULTILINE)
_SENTENCE_END_RE = re.compile(r"(?<=[。！？；.!?;])\s*")


def normalize(title: str) -> str:
    return re.sub(r"\s+", " ", str(title or "")).strip().lower()


def _digit_bigrams(text: str) -> set[str]:
    return {text[i:i + 2] for i in range(len(text) - 1) if any(ch.isdigit() for ch in text[i:i + 2])}


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


_LATIN_RUN_RE = re.compile(r"[a-z0-9][a-z0-9+#./\-]*")


def _bigrams(text: str) -> list[str]:
    """Similarity features: CJK text as ordered character bigrams; a Latin / digit run embedded in CJK counts as one
    feature as a whole (a CJK word followed by "Skill" → its CJK bigrams plus the token skill), otherwise the
    internal bigrams of one short English word would carry the whole score and "reusable Skill" would look like
    "Skill" (2026-09-08 papers KB)."""
    compact = text.replace(" ", "")
    if len(compact) < 2:
        return [compact] if compact else []
    feats: list[str] = []
    pos = 0
    for m in _LATIN_RUN_RE.finditer(compact):
        seg = compact[pos:m.start()]
        feats.extend(seg[i:i + 2] for i in range(len(seg) - 1))
        feats.append(m.group(0))
        pos = m.end()
    seg = compact[pos:]
    feats.extend(seg[i:i + 2] for i in range(len(seg) - 1))
    return feats or [compact]


def dice(a: str, b: str) -> float:
    ga, gb = _bigrams(a), _bigrams(b)
    if not ga or not gb:
        return 0.0
    counts: dict[str, int] = {}
    for g in ga:
        counts[g] = counts.get(g, 0) + 1
    overlap = 0
    for g in gb:
        if counts.get(g, 0) > 0:
            counts[g] -= 1
            overlap += 1
    return 2.0 * overlap / (len(ga) + len(gb))


def is_identifier_like(title: str) -> bool:
    """Symbol / code-like names: no spaces, at most 24 characters, containing digits or all upper-case letters and
    symbols (t_DBE, VIH, ZK7C1049GN, rs0000133, ADRB2, ZK14B108L-ZS25XIT). One character's difference makes
    another thing here, so only canonical_identifier equality counts: no similarity, no model call."""
    text = unicodedata.normalize("NFKC", str(title or "")).strip()
    if not text or " " in text or len(text) > 24 or _CJK_RE.search(text):
        return False
    letters = [ch for ch in text if ch.isalpha()]
    has_digit = any(ch.isdigit() for ch in text)
    mostly_upper = bool(letters) and sum(1 for ch in letters if ch.isupper()) >= max(1, len(letters) - 1)
    return has_digit or mostly_upper


def _polarity_terms(title: str) -> set[str]:
    text = normalize(unicodedata.normalize("NFKC", str(title or "")))
    words = set(_WORD_RE.findall(text))
    found: set[str] = set()
    for x, y in POLARITY_PAIRS:
        for term in (x, y):
            if term.isascii():
                if term in words:
                    found.add(term)
            elif term in text:
                found.add(term)
    return found


def polarity_conflict(a: str, b: str) -> bool:
    """One side contains X and the other contains X's antonym (and not X): opposite direction / polarity, not the
    same thing."""
    ta, tb = _polarity_terms(a), _polarity_terms(b)
    if not ta or not tb:
        return False
    for x, y in POLARITY_PAIRS:
        if (x in ta and y in tb and x not in tb) or (y in ta and x in tb and y not in tb):
            return True
    return False


def is_similar(a: str, b: str) -> bool:
    """A pair worth asking the LLM about. Not the final verdict, which is the model's. Symbol-like names and names
    of opposite polarity never become candidates."""
    ident = is_identifier_like(a) or is_identifier_like(b)       # case is checked before normalization: VIH / t_DBE are symbols
    a, b = normalize(a), normalize(b)
    if not a or not b or a == b:
        return a == b and bool(a)
    if _digit_bigrams(a) != _digit_bigrams(b):
        return False
    if ident:
        return False
    if polarity_conflict(a, b):
        return False
    if _CJK_RE.search(a) or _CJK_RE.search(b):
        if min(len(a), len(b)) < 3:
            return False
        return dice(a, b) >= DICE_THRESHOLD
    return levenshtein(a, b) <= min(len(a), len(b)) // 2


def _identifier_separator(m: re.Match) -> str:
    text = m.string
    between_digits = 0 < m.start() and m.end() < len(text) and text[m.start() - 1].isdigit() and text[m.end()].isdigit()
    return "-" if between_digits else ""


def canonical_identifier(title: str) -> str:
    """Drop spaces, underscores, hyphens, dots, slashes and case: t_AS / tAS / t AS are one symbol.
    A separator between two digits leaves one marker: it decides how the digits are grouped, so 3-12 and 31-2, or
    v1.10 and v11.0, are two different identifiers, and dropping it altogether would give them the same key."""
    text = unicodedata.normalize("NFKC", str(title or "")).casefold()
    return _IDENT_STRIP_RE.sub(_identifier_separator, text)


def strip_type_words(title: str, extra: Iterable[str] = ()) -> str:
    """Strip leading / trailing type words (extra: words of the ontology type names + profile type_words) and
    organization suffix words (Technologies / AG / Co.), returning the normalized (lower-case, single-space)
    remainder. Without extra only the organization suffixes are stripped."""
    words = {str(w).casefold() for w in extra if str(w).strip()}
    text = normalize(unicodedata.normalize("NFKC", str(title or "")))
    tokens = text.split(" ")
    while tokens and tokens[0] in words:
        tokens = tokens[1:]
    while tokens and (tokens[-1] in words or tokens[-1] in SUFFIX_WORDS):
        tokens = tokens[:-1]
    rest = " ".join(tokens)
    return _strip_cjk_type_suffix(rest, words)


def _type_word_set(type_words: Iterable[str]) -> set[str]:
    return {str(w).casefold().strip() for w in type_words if str(w).strip()}


AUTO_RULES = ("identifier", "type_words", "paren", "declared")     # rule-merge bases, most reliable first: symbol > type words > paren alias > declared alias

_DECLARED_ALIAS_RE = re.compile(r"^\s*(?P<title>[^()（）\s]{1,40})\s*[(（](?P<inner>[^()（）]{1,40})[)）]")
# "abbreviation of WorkWidget" / "short for WorkWidget": the description opens by declaring whose short form /
# abbreviation / alias this name is
_DECLARED_SHORT_CJK_RE = re.compile(r"^\s*(?:即|是|为)?\s*(?P<inner>[^,，。;；:：()（）\s]{2,40}?)\s*的(?:简称|缩写|简写|别称|昵称|全称)")
_DECLARED_SHORT_EN_RE = re.compile(r"^\s*(?:short for|abbreviation (?:of|for)|abbreviated from|also known as|aka|alias of)\s+(?P<inner>[^,.;:()]{2,40})", re.IGNORECASE)


def declared_alias(title: str, descriptions: Iterable[str]) -> str:
    """An alias declared at the start of the description, such as "WW (WorkWidget) ..."
    or "DSE (Data Systems Engineer) ...", is direct evidence of identity. Only brackets
    immediately following this entity name count; a pure number or year does not.
    Returns the bracketed name, or an empty string."""
    want = normalize(title)
    if not want:
        return ""
    for d in descriptions:
        text = unicodedata.normalize("NFKC", str(d or ""))
        m = _DECLARED_ALIAS_RE.match(text)
        inner = ""
        if m and normalize(m.group("title")) == want:
            inner = m.group("inner").strip()
        else:
            m = _DECLARED_SHORT_CJK_RE.match(text) or _DECLARED_SHORT_EN_RE.match(text)
            if m:
                inner = m.group("inner").strip()
        if inner and re.search(r"[A-Za-z\u4e00-\u9fff]", inner) and normalize(inner) != want:
            return inner
    return ""


def auto_merge_pairs(entities: list[dict[str, Any]], reasons: dict[tuple[int, int], str] | None = None) -> list[tuple[int, int]]:
    """Pairs mergeable without asking the model: same type and equal canonical_identifier, or equal after removing
    the words of the ontology type name, or the two spellings of a paren alias. Profile-induced type words
    (type_words) do not go through here: "permission model / permission", "Demo phase / Demo" are equal after
    removal but not necessarily one thing, so they only become candidates for the model to judge by description
    (candidate_pairs). Pass a dict as reasons to record which rule merged each pair (the basis category of the
    resolution log)."""
    groups: dict[tuple[str, str, str], list[int]] = {}
    key_rule: dict[tuple[str, str, str], str] = {}
    # A declared alias only becomes a key when that name really is another entity of the same type: two entities
    # each claiming to be short for the same non-existent name must not merge with each other (2026-09-12 product
    # KB: the descriptions of 8 SKUs in a price list all began with "customer-facing full name / copyright full
    # name" and were merged into one)
    titled: dict[tuple[str, str], set[str]] = {}
    for row in entities:
        if not row.get("exact"):
            titled.setdefault((str(row.get("type") or ""), str(row.get("scope") or "")), set()).add(canonical_identifier(str(row.get("title") or "")))
    for idx, row in enumerate(entities):
        if row.get("exact"):
            continue           # deterministic extraction identities are exact (qualified name + file scope): no merging
        etype = str(row.get("type") or "")
        scope = str(row.get("scope") or "")
        title = str(row.get("title") or "")
        keys = {canonical_identifier(title): "identifier"}
        stripped = strip_type_words(title, type_words_of(etype))
        if stripped and len(stripped) >= 2:
            keys.setdefault(canonical_identifier(stripped), "type_words")
        base, inner = paren_alias(title)          # body mass index (BMI): merges with "body mass index" and with "BMI"
        if base:
            keys.setdefault(canonical_identifier(base), "paren")
            keys.setdefault(canonical_identifier(inner), "paren")
        stated = declared_alias(title, row.get("descriptions") or ([row["description"]] if row.get("description") else []))
        if stated and canonical_identifier(stated) in titled.get((etype, scope), set()):
            keys.setdefault(canonical_identifier(stated), "declared")     # WW's description says WW (WorkWidget): merge with WorkWidget
        for key, rule in keys.items():
            if len(key) >= 2:
                gk = (etype, scope, key)
                groups.setdefault(gk, []).append(idx)
                # The two sides of one key may each arrive by a different rule (one the original name, the other
                # with brackets removed): record the weaker one
                prev = key_rule.get(gk)
                key_rule[gk] = rule if prev is None or AUTO_RULES.index(rule) > AUTO_RULES.index(prev) else prev
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for gk, members in groups.items():
        for j in members[1:]:
            pair = (members[0], j)
            if pair in seen or members[0] == j:
                continue
            if polarity_conflict(str(entities[members[0]].get("title") or ""), str(entities[j].get("title") or "")):
                continue          # equal after type words but opposite polarity (input data / output data): no merge
            seen.add(pair)
            pairs.append(pair)
            if reasons is not None:
                reasons[pair] = key_rule.get(gk, "identifier")
    return pairs


def is_containment(a: str, b: str, extra: Iterable[str] = ()) -> bool:
    """After stripping type words and organization suffixes, one side is a word-boundary prefix / suffix of the
    other with at most 2 extra words: tASH and tASH setup time. The extra words must not be a meaningful qualifier
    (diabetic complications ≠ complications, binocular vision loss ≠ vision loss): English may only add type /
    suffix words, CJK may only add ≤ 2 characters that are all in the type word list. "One is a subtype of the
    other" is not the same thing; merging needs other grounds. When the short side is a symbol (tASH, CYP2C9) and
    the long side is "symbol + descriptive words" (tASH setup time, CYP2C9 enzyme), let it through.
    extra: type words allowed as the extra (words of the ontology type names + profile type_words)."""
    words = _type_word_set(extra)
    sa, sb = strip_type_words(a, words), strip_type_words(b, words)
    if not sa or not sb:
        return False
    if sa == sb:
        # Differ only by type words: the ontology-type-name route was already merged directly by auto_merge_pairs
        # (duplicate pairs are removed in resolve); what remains are profile type words (permission model /
        # permission), which become candidates for the model to judge by description
        return normalize(a) != normalize(b)
    if min(len(sa), len(sb)) < 3:
        return False
    if polarity_conflict(sa, sb):
        return False
    raw_short = a if len(sa) <= len(sb) else b
    symbol_plus_words = is_identifier_like(raw_short)               # short side is a symbol (tASH, VIH): long side is "symbol + descriptive words"
    short, long_ = (sa, sb) if len(sa) <= len(sb) else (sb, sa)
    if not symbol_plus_words and _digit_bigrams(short) != _digit_bigrams(long_) and not _digit_bigrams(long_) <= _digit_bigrams(short):
        return False              # a whole symbol contained entirely has matching digits anyway; skip the bigram check
    if _CJK_RE.search(long_):
        if not (long_.startswith(short) or long_.endswith(short)):
            return False
        extra_text = long_[len(short):] if long_.startswith(short) else long_[:-len(short)]
        if symbol_plus_words:
            return 0 < len(extra_text.strip()) <= 4
        return bool(extra_text) and extra_text.strip() in words
    if long_.startswith(short + " "):
        extra, prefix = long_[len(short):].split(), False
    elif long_.endswith(" " + short):
        extra, prefix = long_[:-len(short)].split(), True
    else:
        return False
    if symbol_plus_words:
        # Only symbol first, descriptive words after (tASH setup time) counts; words before the symbol are modifiers
        # (I VSS is a current, not VSS; 48 FBGA is a specific package)
        return not prefix and 1 <= len(extra) <= 3
    return 1 <= len(extra) <= 2 and all(w in words or w in SUFFIX_WORDS for w in extra)


def candidate_pairs(entities: list[dict[str, Any]], *, window: int = WINDOW, type_words: Iterable[str] = ()) -> list[tuple[int, int]]:
    """Within each type, sort by normalized title and run is_similar / is_containment over pairs inside a sliding
    window. Returns index pairs."""
    learned = _type_word_set(type_words)
    by_type: dict[tuple[str, str], list[int]] = {}
    for idx, row in enumerate(entities):
        if row.get("exact"):
            continue
        by_type.setdefault((str(row.get("type") or ""), str(row.get("scope") or "")), []).append(idx)
    pairs: list[tuple[int, int]] = []
    for indexes in by_type.values():
        ordered = sorted(indexes, key=lambda i: normalize(entities[i].get("title")))
        extra = (type_words_of(entities[indexes[0]].get("type")) | learned) if indexes else set()
        for pos, i in enumerate(ordered):
            for j in ordered[pos + 1:pos + 1 + window]:
                a, b = str(entities[i].get("title")), str(entities[j].get("title"))
                if is_similar(a, b) or is_containment(a, b, extra):
                    pairs.append((i, j))
    known = set(pairs)
    pairs.extend(p for p in prefix_and_initial_candidates(entities, by_type) if p not in known)
    return pairs


_INITIALS_LATIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")


def _initials(title: str) -> str:
    """WorkWidget → WW (the upper-case letters of camel case), Data Systems Engineer → DSE (word initials). A
    name that is itself a short all-caps word does not count."""
    text = str(title or "").strip()
    if not text or not text.isascii():
        return ""
    words = _INITIALS_LATIN_RE.findall(text)
    if not words:
        return ""
    if len(words) == 1:
        if text.isupper():
            return ""
        caps = "".join(ch for ch in text if ch.isupper())
        return caps if len(caps) >= 2 else ""
    return "".join(w[0] for w in words).upper()


def prefix_and_initial_candidates(entities: list[dict[str, Any]], by_type: dict[tuple[str, str], list[int]]) -> list[tuple[int, int]]:
    """Two kinds of alias that literal similarity cannot reach, looked for among global entities only and only as
    candidates for the model (2026-09-12 spot check over five KBs):
    · removing a prefix that is itself a name in this graph leaves exactly another entity of the same type: EXO
      Gadget / Gadget, Example Notes / Notes;
    · one is the initialism of the other: WorkWidget / WW, Data Systems Engineer / DSE."""
    all_titles = {normalize(e.get("title")) for e in entities if not e.get("scope") and not e.get("exact")}
    out: set[tuple[int, int]] = set()
    for (_etype, scope), indexes in by_type.items():
        if scope or len(indexes) < 2:
            continue
        by_norm: dict[str, int] = {}
        for i in indexes:
            by_norm.setdefault(normalize(entities[i].get("title")), i)
        initials_of: dict[str, list[int]] = {}
        for i in indexes:
            ini = _initials(str(entities[i].get("title") or ""))
            if ini:
                initials_of.setdefault(ini, []).append(i)
        for i in indexes:
            t = normalize(entities[i].get("title"))
            for cut in range(2, min(len(t) - 1, 16)):
                prefix, rest = t[:cut].strip(" ·-_/"), t[cut:].strip(" ·-_/")
                if len(prefix) >= 2 and len(rest) >= 2 and prefix in all_titles and rest in by_norm and by_norm[rest] != i:
                    j = by_norm[rest]
                    out.add((min(i, j), max(i, j)))
            raw = str(entities[i].get("title") or "").strip()
            if re.fullmatch(r"[A-Z]{2,5}", raw):
                for j in initials_of.get(raw, []):
                    if j != i:
                        out.add((min(i, j), max(i, j)))
    return sorted(out)


def embedding_candidates(entities: list[dict[str, Any]], embed: Callable[[list[str]], list[list[float]]], *,
                         threshold: float = EMBED_THRESHOLD, neighbours: int = EMBED_NEIGHBOURS,
                         batch: int = 64) -> list[tuple[int, int]]:
    """Vector neighbours among same-type titles: pairs with cosine ≥ threshold (at most neighbours per entity).
    Same names that look nothing alike literally (one Chinese, one English) can only be caught this way."""
    import numpy as np

    titles = [str(row.get("title") or "") for row in entities]
    if not titles:
        return []
    vectors: list[list[float]] = []
    for start in range(0, len(titles), batch):
        vectors.extend(embed(titles[start:start + batch]))
    matrix = np.asarray(vectors, dtype="float32")
    if matrix.ndim != 2 or matrix.shape[0] != len(titles):
        return []
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms
    pairs: list[tuple[int, int]] = []
    by_type: dict[tuple[str, str], list[int]] = {}
    for idx, row in enumerate(entities):
        if row.get("exact"):
            continue
        by_type.setdefault((str(row.get("type") or ""), str(row.get("scope") or "")), []).append(idx)
    for indexes in by_type.values():
        if len(indexes) < 2:
            continue
        sub = matrix[indexes]
        sims = sub @ sub.T
        for a_pos, i in enumerate(indexes):
            row = sims[a_pos]
            order = np.argsort(-row)
            taken = 0
            for b_pos in order:
                j = indexes[int(b_pos)]
                if j == i:
                    continue
                if row[int(b_pos)] < threshold or taken >= neighbours:
                    break
                ta, tb = titles[i], titles[j]
                if is_identifier_like(ta) or is_identifier_like(tb) or polarity_conflict(ta, tb) or _digit_bigrams(normalize(ta)) != _digit_bigrams(normalize(tb)):
                    taken += 1
                    continue          # symbol / code, opposite polarity, different digits: not the same however close the vectors
                if i < j:
                    pairs.append((i, j))
                taken += 1
    return sorted(set(pairs))


def _first_sentence(row: dict[str, Any], limit: int = 160, sentences: int = 2) -> str:
    """Evidence shown to the model for the judgement: the first two sentences of the description (≤160 chars).
    Names alone cause misjudgements, and one sentence is not enough either. With dozens of descriptions the first
    is no longer taken (it is merely the first unit in extraction order, not necessarily representative): prefer
    a definition-style description starting with this entity's name ("WW (WorkWidget) is a sample tool …"), else the
    longest among the first few."""
    descriptions = [str(d).strip() for d in (row.get("descriptions") or ([row["description"]] if row.get("description") else [])) if str(d).strip()]
    if not descriptions:
        return ""
    want = normalize(row.get("title"))
    pool = descriptions[:8]
    text = next((d for d in pool if want and normalize(d).startswith(want)), None) or max(pool, key=len)
    parts = [x.strip() for x in _SENTENCE_END_RE.split(text) if x.strip()]
    lead = " ".join(parts[:sentences]) if parts else text
    return (lead[:limit] + "…") if len(lead) > limit else lead


def render_batch(pairs: list[tuple[int, int]], entities: list[dict[str, Any]]) -> str:
    entity_type = str(entities[pairs[0][0]].get("type") or "unknown") if pairs else "unknown"
    lines = [
        "You are deduplicating entities extracted from documents.",
        f"Entity type: {entity_type}",
        "For each numbered pair of entity names below (each followed by a short description in brackets), "
        "decide whether both names refer to the SAME thing, and say WHY with one of these evidence categories: "
        "abbreviation (one is an abbreviation or acronym of the other), spelling (spelling, case, spacing, punctuation, "
        "singular/plural, word order), translation (the same name in another language or script), alias (a documented "
        "alternative name for the same thing; the descriptions must support this).",
        "Treat as DIFFERENT (answer no): a specific thing and its general category (a subtype and its type, a product and its "
        "product line, a symptom and a disease, a finding and a diagnosis); a part and its whole (a function and its module, "
        "a clause and its contract); a cause and its effect; or names whose descriptions describe different things. "
        "When unsure, answer no.",
        "Answer with exactly one line per pair, in the form `<number>: yes <category>` or `<number>: no`. No other text.",
        "",
    ]
    for n, (i, j) in enumerate(pairs, start=1):
        a, b = entities[i], entities[j]
        lines.append(f'{n}. "{a.get("title")}" [{_first_sentence(a)}] | "{b.get("title")}" [{_first_sentence(b)}]')
    return "\n".join(lines)


YES_CATEGORIES = ("abbreviation", "spelling", "translation", "alias")
_ANSWER_CAT_RE = re.compile(r"^\s*(\d+)\s*[\.:：\)]\s*(yes|no|是|否|y|n)\b[\s:：\-–(]*([a-z_]+)?", re.IGNORECASE | re.MULTILINE)


def parse_answer_categories(text: str, count: int) -> list[str]:
    """Read verdict and basis category by number: returns each pair's category ("" means no; yes without a
    category is recorded as "unspecified")."""
    out = [""] * count
    for m in _ANSWER_CAT_RE.finditer(str(text or "")):
        idx = int(m.group(1)) - 1
        if not 0 <= idx < count:
            continue
        if m.group(2).lower() in ("yes", "是", "y"):
            cat = (m.group(3) or "").lower()
            out[idx] = cat if cat in YES_CATEGORIES else "unspecified"
        else:
            out[idx] = ""
    return out


_SAME_RE = re.compile(r"^\s*(\d+)\s*[\.:：\)]\s*(same|broader|narrower|different|相同|不同)\b", re.IGNORECASE | re.MULTILINE)


def render_hyponym_batch(pairs: list[tuple[int, int]], entities: list[dict[str, Any]]) -> str:
    """Second-pass judgement (pairs the model answered alias): only the same thing counts as same; one being a kind
    / part / manifestation / consequence / risk of the other does not. The "… are never same" sentence is task
    definition, not redundancy: after 88d3e91 removed it, the papers KB re-check went from withdrawing 13 pairs to
    4 (pairs like Demo / Demo phase merged again); restored 2026-09-08 by user decision. Symbol / polarity pairs
    are already blocked in code and are not repeated in the first-pass prompt."""
    lines = [
        "Two names below were judged to be aliases of the same thing. Check that judgement strictly.",
        "For each numbered pair answer with one word: `same` only if both names denote exactly the same thing (a synonym, an "
        "abbreviation, a translation, or a documented alternative name); `broader` if the FIRST is a general category that includes the "
        "second; `narrower` if the first is a kind, a part, a manifestation, a finding, a risk or a consequence of the second; "
        "`different` otherwise. A specific thing and its category (a product and its product line, a subtype and its type, a symptom "
        "and a disease, a finding and a diagnosis), a part and its whole (a function and its module, a clause and its contract), "
        "a risk and its outcome, a cause and its effect are never `same`. When unsure, answer `different`.",
        "Answer with exactly one line per pair, in the form `<number>: same|broader|narrower|different`. No other text.",
        "",
    ]
    for n, (i, j) in enumerate(pairs, start=1):
        a, b = entities[i], entities[j]
        lines.append(f'{n}. "{a.get("title")}" [{_first_sentence(a)}] | "{b.get("title")}" [{_first_sentence(b)}]')
    return "\n".join(lines)


_VERDICT_WORDS = {"same": "same", "相同": "same", "broader": "broader", "narrower": "narrower", "different": "different", "不同": "different"}


def parse_verdicts(text: str, count: int) -> list[str]:
    """Read the second-pass answer word by number: same / broader / narrower / different; unanswered ones are
    empty strings."""
    out = [""] * count
    for m in _SAME_RE.finditer(str(text or "")):
        idx = int(m.group(1)) - 1
        if 0 <= idx < count:
            out[idx] = _VERDICT_WORDS.get(m.group(2).lower(), "")
    return out


def parse_same(text: str, count: int) -> list[bool]:
    return [v == "same" for v in parse_verdicts(text, count)]


def parse_answers(text: str, count: int) -> list[bool]:
    """Read yes/no by number; unanswered ones count as no."""
    answers = [False] * count
    for m in _ANSWER_RE.finditer(str(text or "")):
        n = int(m.group(1))
        if 1 <= n <= count:
            answers[n - 1] = m.group(2).lower() in ("yes", "是", "y")
    return answers


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
            self.parent[rb] = ra


def merged_period(a: dict[str, Any], b: dict[str, Any]) -> tuple[str, str]:
    """两条关系并成一条时的起止:两段时间并成从最早的起到最晚的止(一边只写了起、一边只写了止也能拼上)。
    有一边是一个时间点(起止相同:某天做的一次检查)就不并 —— 两次检查不是一段从第一次到第二次的状态,
    并出来的假区间过了止期会被当成已结束(currency.lapses);留起得早的那边(一样早留原来的)。"""
    sides = [(str(x.get("valid_from") or ""), str(x.get("valid_until") or "")) for x in (a, b)]
    sides = [p for p in sides if p[0] or p[1]]
    if not sides:
        return ("", "")
    if any(lo and lo == hi for lo, hi in sides):
        return min(sides, key=lambda p: p[0] or p[1])
    starts = sorted(lo for lo, _ in sides if lo)
    ends = sorted(hi for _, hi in sides if hi)
    return (starts[0] if starts else ""), (ends[-1] if ends else "")


def _ordered_union(groups: list[list[Any]]) -> list[Any]:
    seen: set[Any] = set()
    out: list[Any] = []
    for group in groups:
        for item in group:
            if item not in seen:
                seen.add(item)
                out.append(item)
    return out


def _absorb_descriptions(slot: dict[str, Any], row: dict[str, Any]) -> None:
    """Merge row's descriptions, each with its source label, into slot (deduplicated, order kept). The two lists
    are aligned by index and the summary takes the source by index: a description without a label (attribute
    rows, inferred edges) gets an empty dict as a placeholder so the labels after it do not shift."""
    descriptions, sources = slot["descriptions"], slot["description_sources"]
    sources.extend({} for _ in range(len(descriptions) - len(sources)))
    seen = set(descriptions)
    own = list(row.get("description_sources") or [])
    for i, d in enumerate(row.get("descriptions") or ([row["description"]] if row.get("description") else [])):
        d = str(d).strip()
        if d and d not in seen:
            seen.add(d)
            descriptions.append(d)
            sources.append(dict(own[i]) if i < len(own) and isinstance(own[i], dict) else {})


def merge_entities(
    entities: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    yes_pairs: list[tuple[int, int]],
    *,
    canonical_out: dict[str, str] | None = None,
    forbidden: set[frozenset[str]] | None = None,
    require_compatible_identity: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """判 yes 的对连成图,按连通分量整体合并(不是两两)。canonical_out 给了就填上
    {被并掉的键: 代表键}(增量并入回放用)。forbidden:更正文件里禁止合并的键对,两个分量里分别有这样一对就不连。
    require_compatible_identity:自动判同/回放必须核对类型与作用域;显式的唯一编号关联和人工强制合并不走此猜测约束。

    外部编号(表格编号列、主数据):同一体系下编号不同的两个实体绝不合并 —— 连通时逐对检查两个分量已有的编号,
    冲突的那一对不连(否则 A-B、B-C 会把两个编号不同的 A、C 经由 B 并到一起)。
    代表 = 带外部编号的优先,再按 frequency 最高(平手取标题最短、再按键);描述去重保序、带着各自的来源标签,一条不截
    (没归并的实体也不截,长度由摘要按 token 预算分批),unit_ids / doc_ids / aliases / 外部编号并集,frequency 求和。
    member_keys 记下分量里所有成员的键(含代表):代表换了、名字改了,旧编号照样能查到(方案附录第 4 条)。
    关系两端改写成代表;自环丢掉;改写后重合的边合并(strength_sum 相加、evidence 相加、描述与 unit_ids 并集)。
    """
    uf = _UnionFind(len(entities))
    ext_of: dict[int, dict[str, str]] = {}
    for idx, e in enumerate(entities):
        ext_of[idx] = {str(x.get("scheme") or ""): str(x.get("value") or "") for x in (e.get("external_ids") or [])
                       if isinstance(x, dict) and x.get("value")}
    blocked = 0
    forbid_partners: dict[str, set[str]] = {}
    for pair in forbidden or ():
        a_key, b_key = tuple(pair)
        forbid_partners.setdefault(a_key, set()).add(b_key)
        forbid_partners.setdefault(b_key, set()).add(a_key)
    keys_of: dict[int, set[str]] = {i: {str(e.get("key"))} for i, e in enumerate(entities)} if forbid_partners else {}
    identity_of = {i: {(str(e.get("type") or "").casefold(), str(e.get("scope") or ""), str(e.get("upper") or ""))}
                   for i, e in enumerate(entities)}
    for i, j in yes_pairs:
        ri, rj = uf.find(i), uf.find(j)
        if ri == rj:
            continue
        if require_compatible_identity and identity_of[ri] != identity_of[rj]:
            blocked += 1
            continue
        a, b = ext_of.get(ri) or {}, ext_of.get(rj) or {}
        if any(scheme in b and b[scheme] != value for scheme, value in a.items()):
            blocked += 1
            continue
        if forbid_partners and any(forbid_partners.get(k, set()) & keys_of[rj] for k in keys_of[ri]):
            blocked += 1
            continue
        uf.union(i, j)
        root = uf.find(i)
        ext_of[root] = {**b, **a}
        if forbid_partners:
            keys_of[root] = keys_of[ri] | keys_of[rj]
    groups: dict[int, list[int]] = {}
    for idx in range(len(entities)):
        groups.setdefault(uf.find(idx), []).append(idx)

    canonical_of: dict[str, str] = {}
    merged: list[dict[str, Any]] = []
    merged_away = 0
    for members in groups.values():
        if len(members) == 1:
            merged.append(dict(entities[members[0]]))
            continue
        members.sort(key=lambda i: (not entities[i].get("external_ids"), -int(entities[i].get("frequency") or 0),
                                    len(str(entities[i].get("title") or "")), str(entities[i].get("key") or "")))
        head = dict(entities[members[0]])
        head["descriptions"], head["description_sources"] = [], []
        for i in members:
            _absorb_descriptions(head, entities[i])
        head["description"] = ""
        head["unit_ids"] = _ordered_union([list(entities[i].get("unit_ids") or []) for i in members])
        head["doc_ids"] = _ordered_union([list(entities[i].get("doc_ids") or []) for i in members])
        head["aliases"] = [a for a in _ordered_union(
            [list(entities[i].get("aliases") or []) + [str(entities[i].get("title") or "")] for i in members]
        ) if a and a != head.get("title")]
        head["frequency"] = sum(int(entities[i].get("frequency") or 0) for i in members)
        head["attributes"] = _ordered_union([list(entities[i].get("attributes") or []) for i in members])
        head["member_keys"] = _ordered_union([[str(entities[i].get("key"))] + list(entities[i].get("member_keys") or [])
                                              for i in members])
        ext: list[dict[str, str]] = []
        for i in members:
            for x in entities[i].get("external_ids") or []:
                if isinstance(x, dict) and x not in ext:
                    ext.append(dict(x))
        head["external_ids"] = ext
        kinds_by_freq: dict[str, int] = {}
        for i in members:
            k = str(entities[i].get("kind") or "instance")
            kinds_by_freq[k] = kinds_by_freq.get(k, 0) + int(entities[i].get("frequency") or 0)
        head["kind"] = "class" if kinds_by_freq.get("class", 0) > kinds_by_freq.get("instance", 0) else "instance"
        head["boilerplate"] = all(bool(entities[i].get("boilerplate")) for i in members)
        head["reference"] = all(bool(entities[i].get("reference")) for i in members)
        kinds = {str(entities[i].get("evidence_kind") or "body") for i in members}
        head["evidence_kind"] = "body" if "body" in kinds else ("listing" if "listing" in kinds else "boilerplate")
        for i in members[1:]:
            canonical_of[str(entities[i].get("key"))] = str(head.get("key"))
            merged_away += 1
        merged.append(head)

    title_of = {str(e["key"]): str(e.get("title") or "") for e in merged}
    edges: dict[tuple[str, str, str], dict[str, Any]] = {}
    self_loops = 0
    for rel in relations:
        src = canonical_of.get(str(rel.get("source_key")), str(rel.get("source_key")))
        dst = canonical_of.get(str(rel.get("target_key")), str(rel.get("target_key")))
        if src == dst:
            self_loops += 1
            continue
        predicate = str(rel.get("predicate") or "related_to")
        if not rel.get("directed") and dst < src:
            src, dst = dst, src
        key = (src, dst, predicate)
        if key in edges:
            slot = edges[key]
            slot["strength_sum"] = round(float(slot.get("strength_sum") or 0) + float(rel.get("strength_sum") or 0), 3)
            slot["evidence"] = int(slot.get("evidence") or 0) + int(rel.get("evidence") or 0)
            _absorb_descriptions(slot, rel)
            slot["unit_ids"] = _ordered_union([list(slot.get("unit_ids") or []), list(rel.get("unit_ids") or [])])
            slot["doc_ids"] = _ordered_union([list(slot.get("doc_ids") or []), list(rel.get("doc_ids") or [])])
            slot["valid_from"], slot["valid_until"] = merged_period(slot, rel)
            slot["type_violation"] = bool(slot.get("type_violation")) and bool(rel.get("type_violation"))
            slot["boilerplate"] = bool(slot.get("boilerplate")) and bool(rel.get("boilerplate"))
            slot["reference"] = bool(slot.get("reference")) and bool(rel.get("reference"))
            kinds = {str(slot.get("evidence_kind") or "body"), str(rel.get("evidence_kind") or "body")}
            slot["evidence_kind"] = "body" if "body" in kinds else ("listing" if "listing" in kinds else "boilerplate")
        else:
            row = dict(rel)
            row["source_key"], row["target_key"] = src, dst
            row["source"], row["target"] = title_of.get(src, row.get("source")), title_of.get(dst, row.get("target"))
            row["descriptions"] = list(rel.get("descriptions") or [])
            row["description_sources"] = [dict(s) if isinstance(s, dict) else {} for s in (rel.get("description_sources") or [])]
            edges[key] = row
    if canonical_out is not None:
        canonical_out.update(canonical_of)
    stats = {
        "entities_before": len(entities), "entities_after": len(merged), "merged_away": merged_away,
        "external_id_conflicts_blocked": blocked,
        "relationships_before": len(relations), "relationships_after": len(edges),
        "self_loops_dropped": self_loops, "groups": sum(1 for m in groups.values() if len(m) > 1),
    }
    return merged, list(edges.values()), stats


def _merge_log(entities: list[dict[str, Any]], key_of: list[str], canonical_of: dict[str, str],
               evidence: dict[tuple[int, int], tuple[str, str]]) -> list[dict[str, Any]]:
    """Resolution log: one row per merged-away entity, recording the representative it went into and the pair
    whose judgement merged it (source auto / lexical / embedding / replay, and the basis category).
    Merging works on whole connected components, so neither end of a matched pair need be the representative
    (A-B and B-C linked, representative A): walk out from the representative along the matched pairs, record every
    entity reached, and when the entity it was reached from is not the representative, note that one in
    via_title. Recorded per pair, such a pair's kept key and kept_title would name two different entities and the
    merged-away one could not be found in the log."""
    index = {k: i for i, k in enumerate(key_of) if k}
    links: dict[int, list[tuple[int, str, str]]] = {}
    for (i, j), (source, cat) in evidence.items():
        links.setdefault(i, []).append((j, source, cat))
        links.setdefault(j, []).append((i, source, cat))

    def title(i: int) -> str:
        return str(entities[i].get("title") or "")

    log: list[dict[str, Any]] = []
    for head_key in dict.fromkeys(canonical_of.values()):
        head = index.get(head_key)
        if head is None:
            continue
        reached, queue = {head}, [head]
        while queue:
            cur = queue.pop(0)
            for other, source, cat in links.get(cur, []):
                if other in reached or canonical_of.get(key_of[other]) != head_key:
                    continue
                reached.add(other)
                queue.append(other)
                row = {"kept": head_key, "merged": key_of[other], "kept_title": title(head), "merged_title": title(other),
                       "source": source, "category": cat}
                if cur != head:
                    row["via_title"] = title(cur)
                log.append(row)
    return log


def resolve(client: ChatClient, entities: list[dict[str, Any]], relations: list[dict[str, Any]], *,
            progress: Callable[[int, int], None] | None = None,
            embed: Callable[[list[str]], list[list[float]]] | None = None,
            prior: dict[str, Any] | None = None,
            type_words: Iterable[str] = (),
            forbidden: set[frozenset[str]] | None = None,
            ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """确定性直接合并 → 粗筛(字面 + 包含 + 向量近邻)→ 批量判同 → 合并。
    forbidden:更正文件里禁止合并的键对(合并时守住,判成同一个也不并)。
    type_words:画像归纳的类型词(语料语言的 信号 / 寄存器 / 指标 …),与本体类型名里的词一起当可去掉的类型词。
    单批调用失败只跳过并计数,不中止整次建图。

    prior:上一版图的归并结论({"map": {被并掉的键: 代表键}, "judged": [[键, 键], …]}),增量并入时传。
    判过的对照上一版回放(并过的照并、否过的照否),只把没判过的候选对(牵涉新实体的)送去判同;
    上一版并掉的对即使这一版没再进候选也照旧并 —— 归并结论不随并入次数漂移,整库重建时才全部重判。
    返回的 stats 里带 _map / _judged(这一版的结论,供下一版回放),调用方取走后再落 manifest。"""
    auto_reasons: dict[tuple[int, int], str] = {}
    auto = auto_merge_pairs(entities, auto_reasons)
    auto_set = set(auto)
    lexical = [p for p in candidate_pairs(entities, type_words=type_words) if p not in auto_set]
    pairs = list(lexical)
    embedded: list[tuple[int, int]] = []
    if embed is not None:
        try:
            known = set(pairs) | auto_set
            embedded = [p for p in embedding_candidates(entities, embed) if p not in known]
        except Exception as exc:   # an unavailable embedding service only loses this candidate route, not the build
            print(f"[graph] embedding candidates skipped: {exc!r}", flush=True)
            embedded = []
    pairs = pairs + embedded
    source_of: dict[tuple[int, int], str] = {p: "auto" for p in auto}
    source_of.update({p: "lexical" for p in lexical})
    source_of.update({p: "embedding" for p in embedded})
    key_of = [str(e.get("key") or "") for e in entities]
    stats: dict[str, Any] = {"candidates": len(pairs), "auto_pairs": len(auto), "embedding_candidates": len(embedded),
                             "batches": 0, "failed_batches": 0, "yes": 0, "replayed_pairs": 0, "replayed_yes": 0,
                             "vetoed": 0, "yes_by_category": {}, "rechecked": 0, "recheck_dropped": 0}
    rejected: list[dict[str, Any]] = []     # model said yes but hard rules / second pass blocked: for audit (not merged, not in the log)
    yes_pairs: list[tuple[int, int]] = list(auto)
    evidence: dict[tuple[int, int], tuple[str, str]] = {p: ("auto", auto_reasons.get(p, "identifier")) for p in auto}     # pair → (source, basis category)
    judged_keys: set[frozenset[str]] = {frozenset((key_of[i], key_of[j])) for i, j in auto}
    if prior is not None:
        prior_map = {str(k): str(v) for k, v in (prior.get("map") or {}).items()}
        prior_judged = {frozenset((str(a), str(b))) for a, b in (prior.get("judged") or []) if str(a) != str(b)}
        index = {k: i for i, k in enumerate(key_of) if k}

        def canon(k: str) -> str:
            return prior_map.get(k, k)

        replayed_yes: list[tuple[int, int]] = []
        for merged_key, canonical in prior_map.items():
            i, j = index.get(canonical), index.get(merged_key)
            if i is not None and j is not None and i != j:
                replayed_yes.append((i, j))
        fresh: list[tuple[int, int]] = []
        for i, j in pairs:
            pair_keys = frozenset((key_of[i], key_of[j]))
            if pair_keys in prior_judged:
                stats["replayed_pairs"] += 1
                judged_keys.add(pair_keys)
                if canon(key_of[i]) == canon(key_of[j]) and (i, j) not in replayed_yes:
                    replayed_yes.append((i, j))
            else:
                fresh.append((i, j))
        pairs = fresh
        for i, j in replayed_yes:
            judged_keys.add(frozenset((key_of[i], key_of[j])))
            evidence.setdefault((i, j), ("replay", "prior"))
        yes_pairs.extend(replayed_yes)
        stats["replayed_yes"] = len(replayed_yes)
        stats["judged_new"] = len(pairs)
    if pairs:
        by_type: dict[str, list[tuple[int, int]]] = {}
        for i, j in pairs:
            by_type.setdefault(str(entities[i].get("type") or ""), []).append((i, j))   # batches per type; scopes were already separated at the candidate stage
        batches = [group[k:k + BATCH_SIZE] for group in by_type.values() for k in range(0, len(group), BATCH_SIZE)]
        stats["batches"] = len(batches)

        def judge(batch: list[tuple[int, int]]) -> list[tuple[tuple[int, int], str]]:
            response = client.chat(render_batch(batch, entities), max_tokens=2048)
            cats = parse_answer_categories(response, len(batch))
            return [(pair, cat) for pair, cat in zip(batch, cats) if cat]

        llm_yes = 0
        for batch, result, error in client.run_parallel(batches, judge, workers=client.workers, progress=progress):
            if error is not None:
                stats["failed_batches"] += 1     # unjudged pairs stay out of judged: asked again next version
                if not isinstance(error, LLMCallError):
                    print(f"[graph] resolution batch failed ({len(batch)} pairs): {error!r}", flush=True)
                continue
            for i, j in batch:
                judged_keys.add(frozenset((key_of[i], key_of[j])))
            for pair, cat in result or []:
                i, j = pair
                # A model yes still passes the hard rules: opposite polarity or differing symbols do not merge (the
                # no is recorded in judged, not asked again next version)
                if polarity_conflict(str(entities[i].get("title")), str(entities[j].get("title"))):
                    stats["vetoed"] += 1
                    rejected.append({"a": key_of[i], "b": key_of[j], "a_title": str(entities[i].get("title") or ""),
                                     "b_title": str(entities[j].get("title") or ""), "source": source_of.get(pair, "lexical"),
                                     "category": cat, "reason": "polarity"})
                    continue
                yes_pairs.append(pair)
                evidence[pair] = (source_of.get(pair, "lexical"), cat)
                stats["yes_by_category"][cat] = stats["yes_by_category"].get(cat, 0) + 1
                llm_yes += 1
        stats["yes"] = llm_yes
        # Second pass: alias is the weakest of the four bases (spelling / abbreviation / translation can be checked
        # against the names, an alias can only be trusted from the descriptions), so every pair the model answered
        # alias (or no category) is strictly asked once more whether it is "the same thing": vector-route pairs do
        # not look alike, and lexical-route pairs like "output buffers / Output drivers", "reusable Skill / Skill"
        # differ by exactly the meaningful word. Only "symbol + descriptive words" (CLK clock input / CLK) is
        # exempt: the containment itself is the basis.
        def _symbol_plus_words(p: tuple[int, int]) -> bool:
            ta, tb = str(entities[p[0]].get("title") or ""), str(entities[p[1]].get("title") or "")
            x, y = normalize(ta), normalize(tb)
            if x == y or not (x in y or y in x):
                return False
            return is_identifier_like(ta if len(x) <= len(y) else tb)
        recheck = [p for p in yes_pairs if evidence.get(p, ("", ""))[1] in ("alias", "unspecified") and not _symbol_plus_words(p)]
        stats["rechecked"] = len(recheck)
        stats["recheck_dropped"] = 0
        if recheck:
            keep: set[tuple[int, int]] = set()
            verdicts: dict[tuple[int, int], str] = {}     # withdrawn pairs keep the verdict (broader / narrower / different) for audits
            batches2 = [recheck[k:k + BATCH_SIZE] for k in range(0, len(recheck), BATCH_SIZE)]

            def judge2(batch: list[tuple[int, int]]) -> list[tuple[tuple[int, int], str]]:
                response = client.chat(render_hyponym_batch(batch, entities), max_tokens=2048)
                return list(zip(batch, parse_verdicts(response, len(batch))))

            for batch, result, error in client.run_parallel(batches2, judge2, workers=client.workers):
                if error is not None:
                    stats["failed_batches"] += 1
                    keep.update(batch)            # not judged: keep the first-pass verdict rather than lose it to service jitter
                    continue
                for pair, verdict in result or []:
                    verdicts[pair] = verdict
                    if verdict == "same":
                        keep.add(pair)
            dropped = [p for p in recheck if p not in keep]
            stats["recheck_dropped"] = len(dropped)
            if dropped:
                gone = set(dropped)
                yes_pairs = [p for p in yes_pairs if p not in gone]
                for p in dropped:
                    source, cat = evidence.pop(p, ("embedding", "alias"))
                    stats["yes_by_category"][cat] = max(0, stats["yes_by_category"].get(cat, 0) - 1)
                    i, j = p
                    rejected.append({"a": key_of[i], "b": key_of[j], "a_title": str(entities[i].get("title") or ""),
                                     "b_title": str(entities[j].get("title") or ""), "source": source, "category": cat,
                                     "reason": "recheck", "verdict": verdicts.get(p, "")})
    if prior and prior.get("rejected"):
        # Pairs blocked by rules / the second pass last version: not asked again this version (replayed as no from
        # judged), but the audit record must keep showing them, with source replay
        index_now = {k: i for i, k in enumerate(key_of)}
        merged_now = set()
        for p in yes_pairs:
            merged_now.add(frozenset((key_of[p[0]], key_of[p[1]])))
        for r in prior["rejected"]:
            a, b = str(r.get("a") or ""), str(r.get("b") or "")
            if a in index_now and b in index_now and frozenset((a, b)) not in merged_now:
                rejected.append({**r, "a_title": str(entities[index_now[a]].get("title") or r.get("a_title") or ""),
                                 "b_title": str(entities[index_now[b]].get("title") or r.get("b_title") or ""), "source": "replay"})
        stats["replayed_rejected"] = len(rejected)
    canonical_of: dict[str, str] = {}
    merged_entities, merged_relations, merge_stats = merge_entities(entities, relations, yes_pairs, canonical_out=canonical_of,
                                                                    forbidden=forbidden, require_compatible_identity=True)
    stats.update(merge_stats)
    stats["_map"] = canonical_of
    stats["_judged"] = sorted(sorted(p) for p in judged_keys if len(p) == 2)
    stats["_log"] = _merge_log(entities, key_of, canonical_of, evidence)
    stats["_rejected"] = rejected
    return merged_entities, merged_relations, stats
