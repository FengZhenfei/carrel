"""Cache identity follows each unit's extraction route, not unrelated table profiles."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict

from .. import db


def fingerprints(base, units, deterministic_ids, tables):
    from .deterministic import VERSION
    from .tables import VERSION as TABLE_VERSION

    result = {}
    ambiguous = tables.ambiguous_names()
    for unit in units:
        if unit.unit_id in deterministic_ids:
            material = [base, "deterministic", VERSION]
        elif tables.wants(unit):
            profiles = {key: tables.profiles.get(key) for key, _ in tables.unit_tables.get(unit.unit_id, [])}
            names = {tables._name(profile, values)[0].casefold() for profile, values in tables._rows(unit)[0]}
            material = [base, "table-identity-v1", TABLE_VERSION, profiles, sorted(names & ambiguous)]
        else:
            result[unit.unit_id] = base
            continue
        result[unit.unit_id] = hashlib.sha256(json.dumps(material, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:24]
    return result


def load(con, kb_id, keys, *, facts=False):
    groups = defaultdict(list)
    for uid, fingerprint in keys.items():
        groups[fingerprint].append(uid)
    result = {}
    reader = db.load_graph_facts if facts else db.load_graph_extractions
    for fingerprint, uids in groups.items():
        result.update(reader(con, kb_id, fingerprint, uids))
    return result


def valid_ids(rows, *, facts=False):
    if facts:
        return set(rows)
    return {uid for uid, row in rows.items() if row.get("entities") or row.get("relations")
            or not int(row.get("stats", {}).get("malformed", 0))}


def migrate_legacy(con, source, config_fingerprint, keys, *, facts=False):
    """Reuse only proven same-config LLM caches. Deterministic/table rows are cheap to regenerate.

    No arbitrary old fingerprint is accepted. The last published build's manifest identifies
    the cache and the graph configuration must still match; new routing never borrows it.
    """
    previous = db.latest_successful_graph_build(con, source.kb_id)
    if previous is None or str(previous["cache_fingerprint"] or "") != config_fingerprint:
        return 0
    try:
        manifest = json.loads(previous["manifest_json"] or "{}")
    except ValueError:
        return 0
    phase = manifest.get("facts" if facts else "extract") or {}
    old = str(phase.get("fingerprint") or "")
    if not old or phase.get("unit_fingerprints"):
        return 0
    table = "graph_facts" if facts else "graph_extractions"
    columns = [str(r["name"]) for r in con.execute(f"PRAGMA table_info({table})")]
    select = ["?" if c == "fingerprint" else c for c in columns]
    moved = 0
    for uid, new in keys.items():
        if new == old:
            continue
        moved += con.execute(
            f"INSERT OR IGNORE INTO {table} ({','.join(columns)}) SELECT {','.join(select)} FROM {table} "
            "WHERE kb_id=? AND unit_id=? AND fingerprint=? AND model NOT IN ('table','deterministic')",
            (new, source.kb_id, uid, old)).rowcount
    return moved
