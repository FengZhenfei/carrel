"""Fact-level check (2026-09-08): take a set of manually verified "gold standard" facts and look them up in the
graph. It measures the result of the extraction -> resolution -> attribution -> reconciliation chain without
retrieval or generation, so changes invisible to the keyword question sets (a fact filed under the wrong
subject, a broken series, a false conflict) show up here.

The gold file is a JSON list; each entry looks like:
  {"subject": "Jane Doe", "property": "body mass index", "series": [["2024-03-15", "23.4"], ["2025-09-20", "24.1"]], "unit": ""}
  {"subject": "ZK7C1049GN", "property": "operating temperature range", "min": "-40", "max": "85", "unit": "°C", "conditions": {"range": "industrial"}}
  {"subject": "ZK14B108L/ZK14B108N", "subject_any": ["ZK14B108L"], "property": "address access time", "symbol": "t_AA", "max": "20", "conditions": {"speed grade": "20 ns"}}
Matching: the subject by the canonical form of the entity title / aliases (after resolution every name of an
entity counts); the property by the normalised property name / concept name / symbol, or any spelling in
property_any; the time by prefix (2025 matches 2025-10-11); values by type (scalars compare the number and the
comparator, ranges both ends, different types are never equal), and by canonical unit when the gold entry
gives one; conditions are compared only when given.
Verdict per entry: found / wrong_subject (same property, time and value, but filed under another subject) /
wrong_value (subject, property and time match, the value differs) / missing.
For a series it also checks how many points were found, whether they belong to one series (same series_key),
and whether any was recorded as a conflict (conflict_group)."""
from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any

from .facts import canonical_unit, classify_value

_WS_RE = re.compile(r"[\s_\-\.·/:：()（）,，;；]+")
_CMP_ALIAS = {"<=": "≤", "=<": "≤", ">=": "≥", "=>": "≥", "≦": "≤", "≧": "≥"}


def canon(text: Any) -> str:
    return _WS_RE.sub("", unicodedata.normalize("NFKC", str(text or ""))).casefold()


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)


def value_equal(gold: Any, got: Any) -> bool:
    """Compare by value type (Codex review F05): it used to take the first number in the string, so "20" matched
    "20/25/45" and "1.3" matched "<1.3". Now scalar vs scalar needs an equal number and the same comparator,
    range vs range needs both ends equal, different types are never equal, and the rest compares text without
    whitespace."""
    if gold in (None, "") and got in (None, ""):
        return True
    g, v = classify_value(gold), classify_value(got)
    if g["kind"] == "scalar" and v["kind"] == "scalar":
        gc, vc = _CMP_ALIAS.get(g.get("cmp") or "", g.get("cmp") or ""), _CMP_ALIAS.get(v.get("cmp") or "", v.get("cmp") or "")
        return gc == vc and _close(g["num"], v["num"])
    if g["kind"] == "range" and v["kind"] == "range":
        return _close(g["lo"], v["lo"]) and _close(g["hi"], v["hi"])
    if g["kind"] != v["kind"]:
        return False
    return canon(gold) == canon(got) and bool(canon(gold))


def unit_equal(item: dict[str, Any], fact: dict[str, Any]) -> bool:
    """When the gold entry gives a unit it must match (compared by canonical unit); when it does not, units are not
    compared."""
    want = canonical_unit(item.get("unit"))
    if not want:
        return True
    return canonical_unit(fact.get("unit")) == want


def when_matches(gold_when: str, fact: dict[str, Any]) -> bool:
    if not gold_when:
        return True
    got = str(fact.get("valid_from") or fact.get("axis") or "")
    return got.startswith(str(gold_when)) or str(gold_when).startswith(got) and bool(got)


def _entity_names(graph: dict[str, Any]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for e in graph.get("entities") or []:
        names = {canon(e.get("title"))} | {canon(a) for a in (e.get("aliases") or [])}
        out[str(e.get("key") or "")] = {n for n in names if n}
    return out


def subject_matches(item: dict[str, Any], fact: dict[str, Any], names: dict[str, set[str]]) -> bool:
    wanted = {canon(item.get("subject"))} | {canon(s) for s in (item.get("subject_any") or [])}
    wanted = {w for w in wanted if w}
    if canon(fact.get("subject")) in wanted:
        return True
    for key in [fact.get("subject_key")] + list(fact.get("subject_keys") or []):
        if key and names.get(str(key), set()) & wanted:
            return True
    return False


def property_matches(item: dict[str, Any], fact: dict[str, Any]) -> bool:
    wanted = {canon(item.get("property"))} | {canon(p) for p in (item.get("property_any") or [])}
    wanted = {w for w in wanted if w}
    if canon(fact.get("property")) in wanted or canon(fact.get("concept")) in wanted:
        return True
    sym = canon(item.get("symbol"))
    return bool(sym) and canon(fact.get("symbol")) == sym


def conditions_match(item: dict[str, Any], fact: dict[str, Any]) -> bool:
    want = item.get("conditions") or {}
    if not want:
        return True
    have = {canon(k): canon(v) for k, v in (fact.get("conditions") or {}).items()}
    for k, v in want.items():
        got = have.get(canon(k))
        if got is None:
            # when the condition name does not match, fall back: the value appearing in any condition value counts
            # too ("speed grade: 20 ns" vs "grade: 20ns")
            if not any(canon(v) == x or canon(v) in x for x in have.values()):
                return False
        elif not (got == canon(v) or canon(v) in got or got in canon(v)):
            return False
    return True


def fields_equal(item: dict[str, Any], fact: dict[str, Any]) -> bool:
    checked = False
    for f in ("value", "min", "typ", "max"):
        if f in item and str(item.get(f) or "") != "":
            checked = True
            if not value_equal(item[f], fact.get(f)):
                # a single-value gold entry also accepts the value in any of the min / typ / max columns
                if f == "value" and any(value_equal(item[f], fact.get(x)) for x in ("min", "typ", "max")):
                    continue
                return False
    return checked


def check_point(item: dict[str, Any], when: str, expect: dict[str, Any], facts: list[dict[str, Any]], names: dict[str, set[str]]) -> dict[str, Any]:
    """Where one gold entry (or one point of a series) lands in the fact table."""
    point = {**item, **expect}
    same_prop = [f for f in facts if property_matches(item, f) and when_matches(when, f) and conditions_match(item, f)]
    on_subject = [f for f in same_prop if subject_matches(item, f, names)]
    hit = [f for f in on_subject if fields_equal(point, f) and unit_equal(item, f)]
    if hit:
        return {"status": "found", "fact_id": hit[0].get("id"), "series_key": hit[0].get("series_key"), "conflict": bool(hit[0].get("conflict_group")),
                "subject": hit[0].get("subject"), "rel_path": hit[0].get("rel_path")}
    elsewhere = [f for f in same_prop if fields_equal(point, f) and unit_equal(item, f)]
    if elsewhere:
        return {"status": "wrong_subject", "fact_id": elsewhere[0].get("id"), "subject": elsewhere[0].get("subject"), "rel_path": elsewhere[0].get("rel_path")}
    if on_subject:
        f = on_subject[0]
        return {"status": "wrong_value", "fact_id": f.get("id"), "got": {k: f.get(k) for k in ("value", "min", "typ", "max", "unit")}, "rel_path": f.get("rel_path")}
    return {"status": "missing"}


def factcheck(graph: dict[str, Any], gold: list[dict[str, Any]]) -> dict[str, Any]:
    facts = list(graph.get("specs") or [])
    names = _entity_names(graph)
    rows: list[dict[str, Any]] = []
    counts = {"found": 0, "wrong_subject": 0, "wrong_value": 0, "missing": 0}
    series_total = series_found = series_ok = false_conflicts = 0
    for item in gold:
        label = f"{item.get('subject')} · {item.get('property') or item.get('symbol')}"
        if item.get("series"):
            points = []
            for when, value in item["series"]:
                res = check_point(item, str(when), {"value": value}, facts, names)
                res["when"] = when
                points.append(res)
                counts[res["status"]] += 1
            found = [p for p in points if p["status"] == "found"]
            keys = {p.get("series_key") for p in found if p.get("series_key")}
            one_series = len(found) == len(points) and len(keys) == 1
            conflicts = sum(1 for p in found if p.get("conflict"))
            series_total += 1
            series_found += len(found) == len(points)
            series_ok += one_series and conflicts == 0
            false_conflicts += conflicts
            rows.append({"label": label, "kind": "series", "points": points, "points_found": len(found), "points_total": len(points),
                         "one_series": one_series, "false_conflicts": conflicts})
        else:
            res = check_point(item, str(item.get("when") or ""), {}, facts, names)
            counts[res["status"]] += 1
            rows.append({"label": label, "kind": "fact", **res})
    total = sum(counts.values())
    return {"graph_version": graph.get("graph_version"), "rows": rows,
            "summary": {"points": total, **counts, "found_rate": round(counts["found"] / total, 4) if total else None,
                        "series": series_total, "series_complete": series_found, "series_ok": series_ok, "false_conflicts": false_conflicts}}


def factcheck_markdown(report: dict[str, Any], kb_id: str = "") -> str:
    s = report["summary"]
    lines = [f"# Fact-level check · {kb_id}", "", f"Version {report.get('graph_version') or '(current)'}", "",
             "| Metric | Value |", "|---|---|",
             f"| Gold points | {s['points']} |", f"| Found | {s['found']} ({s['found_rate']:.1%}) |" if s['found_rate'] is not None else "| Found | 0 |",
             f"| Wrong subject | {s['wrong_subject']} |", f"| Wrong value | {s['wrong_value']} |", f"| Missing | {s['missing']} |",
             f"| Series (complete / single run without conflicts / total) | {s['series_complete']} / {s['series_ok']} / {s['series']} |",
             f"| False conflicts | {s['false_conflicts']} |", "",
             "| Gold | Verdict | Note |", "|---|---|---|"]
    for r in report["rows"]:
        if r["kind"] == "series":
            bad = [f"{p['when']}:{p['status']}" for p in r["points"] if p["status"] != "found"]
            note = ("single series" if r["one_series"] else "series broken") + (f", conflicts {r['false_conflicts']}" if r["false_conflicts"] else "") + ("; " + ", ".join(bad) if bad else "")
            lines.append(f"| {r['label']} | {r['points_found']}/{r['points_total']} | {note} |")
        else:
            note = ""
            if r["status"] == "wrong_subject":
                note = f"filed under “{r.get('subject')}”"
            elif r["status"] == "wrong_value":
                note = f"the graph has {r.get('got')}"
            lines.append(f"| {r['label']} | {r['status']} | {note} |")
    return "\n".join(lines) + "\n"


def load_gold(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [r for r in rows if isinstance(r, dict)]
