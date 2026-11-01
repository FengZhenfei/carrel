"""Synthetic temporal and schema-evolution regression cases."""
from copy import deepcopy

import pytest

from kb_pipeline.graph.currency import in_scope, mark_fact_lapses, query_scope
from kb_pipeline.graph.facts import fact_id, link_facts, normalize_fact, recover_observation_date
from kb_pipeline.graph.reconcile import reconcile_facts
from kb_pipeline.graph.schema_reuse import reuse_supported_model, supported_definitions
from kb_pipeline.ontology import normalize_model
from _support import _unit


def fact(fid, start, value, **extra):
    return {"id": fid, "subject": "Device A", "subject_key": "a", "property": "price",
            "concept_key": "price", "value": str(value), "unit": "USD", "doc_id": fid,
            "fact_kind": "state", "valid_from": start, "time_basis": "explicit", **extra}


@pytest.mark.parametrize("end", ["2026", "2026-06", "2026-06-30"])
def test_overlapping_different_states_retain_both_sources(end):
    rows = [fact("a", "2025-01-01", 70, valid_until=end), fact("b", "2026-06-01", 80)]
    out = reconcile_facts(rows)
    assert out["stats"]["conflicts"] == 1
    assert all("superseded_by" not in r for r in rows)
    assert {r["fact_id"] for r in out["conflicts"][0]["values"]} == {"a", "b"}
    mark_fact_lapses(rows, [])
    assert all(in_scope(r, query_scope(as_of="2026-06-10")) for r in rows)
    assert not in_scope(rows[0], query_scope(as_of="2027-01-01"))


def test_same_date_disagreement_does_not_choose_an_arbitrary_successor():
    rows = [fact("old", "2024", 70), fact("a", "2025", 80), fact("b", "2025", 90)]
    assert reconcile_facts(rows)["stats"]["conflicts"] == 1
    assert "superseded_by" not in rows[0]


def test_nonoverlapping_states_and_history_queries():
    rows = [fact("a", "2024", 70, valid_until="2024"), fact("b", "2025", 80)]
    assert reconcile_facts(rows)["stats"]["conflicts"] == 0
    mark_fact_lapses(rows, [])
    assert all("superseded_by" not in r for r in rows)
    assert all(in_scope(r, query_scope()) for r in rows)
    assert in_scope(rows[0], query_scope(as_of="2024-06-01"))
    assert not in_scope(rows[1], query_scope(as_of="2024-06-01"))
    assert all(in_scope(r, query_scope(history=True)) for r in rows)


def test_document_snapshot_is_not_an_effective_date():
    unit = _unit("u", doc="kb_x:1"); unit.doc_date = "2026-03-01"; unit.axis = unit.doc_date
    row = normalize_fact({"subject": "Policy A", "property": "status", "value": "active", "fact_kind": "state"})
    row.update(id="a", unit_id="u")
    link_facts([row], [], units_by_id={"u": unit})
    assert row["observed"] == unit.doc_date and row["time_basis"] == "document"
    assert not row.get("valid_from")
    rows = [row, {**row, "id": "b", "value": "inactive", "axis": "2026-04-01", "observed": "2026-04-01"}]
    assert reconcile_facts(rows)["stats"]["series"] == 1
    assert all("superseded_by" not in r for r in rows)


def test_explicit_fact_time_overrides_later_publication_date():
    unit = _unit("u", doc="kb_x:1"); unit.doc_date = "2026-03-01"; unit.axis = unit.doc_date
    row = normalize_fact({"subject": "Device A", "property": "price", "value": "70", "valid_from": "2025", "period_text": "in 2025"})
    row["unit_id"] = "u"
    link_facts([row], [], units_by_id={"u": unit})
    assert (row["valid_from"], row["observed"], row["time_basis"]) == ("2025", "2026-03-01", "explicit")


def test_interval_end_is_part_of_assertion_identity():
    a = fact("a", "2024", 70, valid_until="2025")
    b = {**a, "valid_until": "2026"}
    assert fact_id("u", a) != fact_id("u", b)


def test_dated_reports_remain_two_observations_not_an_effective_interval():
    rows = []
    for when in ("2025-02-11", "2025-03-12"):
        row = normalize_fact({"subject": "Batch A", "property": "status", "value": "awaiting review",
                              "fact_kind": "state", "observed_at": when, "period_text": when})
        row.update(id=fact_id("u", row), doc_id="d", observed="2026-03-01", axis="2026-03-01")
        rows.append(row)
    assert rows[0]["id"] != rows[1]["id"]
    out = reconcile_facts(rows)
    assert out["stats"]["series"] == 1
    assert all(r["time_basis"] == "reported" and not r.get("valid_from") and not r.get("superseded_by") for r in rows)
    assert in_scope(rows[0], query_scope(as_of="2025-02-28"))
    assert not in_scope(rows[1], query_scope(as_of="2025-02-28"))


@pytest.mark.parametrize("period,expected", [("In 2025", "2025"), ("2024年6月", "2024-06"),
    ("2025-01-15", "2025-01-15"), ("before 2025", None), ("2024 to 2025", None),
    ("next year", None), ("2025-13", None)])
def test_only_literal_single_dates_can_be_recovered(period, expected):
    row = normalize_fact({"subject": "Line A", "property": "output", "value": "20", "period_text": period})
    recover_observation_date(row, period + ": Line A output is 20 kg.")
    assert row.get("observed_at") == expected
    assert not row.get("valid_from")
    row = normalize_fact({"subject": "Line A", "property": "output", "value": "20", "period_text": period})
    recover_observation_date(row, "Line A output is 20 kg; date unknown.")
    assert not row.get("observed_at")


@pytest.mark.parametrize("kind,expires", [("state", True), ("observation", False), ("reference", False)])
def test_one_day_policy_state_expires_but_observation_does_not(kind, expires):
    rows = [fact("a", "2025-01-01", 70, valid_until="2025-01-01", fact_kind=kind)]
    mark_fact_lapses(rows, [{"key": "a", "upper": "document", "kind": "instance"}])
    assert rows[0]["lapses"] == expires
    assert in_scope(rows[0], query_scope(as_of="2025-01-02")) != expires


@pytest.mark.parametrize("qualifier", [{"region": "west"}, {"tax": "excluded"}, {"quantity": "bulk"}])
def test_different_applicability_is_not_a_conflict(qualifier):
    rows = [fact("a", "2025", 70, valid_until="2027", conditions={}), fact("b", "2026", 80, conditions=qualifier)]
    out = reconcile_facts(rows)
    assert out["stats"]["series"] == out["stats"]["conflicts"] == 0


def model():
    return normalize_model({"types": [{"name": "Machine", "is_a": "entity", "description": "A named machine"},
                                       {"name": "Component", "is_a": "part", "description": "A physical component"},
                                       {"name": "Unused", "is_a": "concept"}],
        "predicates": [{"name": "contains", "source_parents": ["entity"], "target_parents": ["part"],
                        "description": "A machine contains a physical component", "multivalued": True}], "profile": {}})[0]


def graph():
    return {"entities": [{"key": "a", "type": "Machine", "unit_ids": ["live"]},
                          {"key": "b", "type": "Component", "unit_ids": ["live"]},
                          {"key": "gone", "type": "Unused", "unit_ids": ["removed"]}],
            "relations": [{"source_key": "a", "target_key": "b", "predicate": "contains", "unit_ids": ["live"]}]}


def test_resampling_preserves_supported_contract_even_when_sample_omits_it():
    old = model(); g = graph()
    proposed = normalize_model({"types": [{"name": "NewCategory", "is_a": "concept"}], "predicates": []})[0]
    for _ in range(3):
        support = supported_definitions(g, old, {"live"})
        out, guard = reuse_supported_model(proposed, {"schema_model": old, "source_support": support})
        assert {"Machine", "Component"} <= {t["name"] for t in out["types"]}
        assert "Unused" not in {t["name"] for t in out["types"]}
        assert next(p for p in out["predicates"] if p["name"] == "contains") == next(p for p in old["predicates"] if p["name"] == "contains")
        assert guard["kept_predicates"] == ["contains"]
        old = out


def test_repeated_violations_do_not_relax_or_preserve_a_bad_endpoint():
    g = graph(); g["relations"] = [{**g["relations"][0], "target_key": "a"} for _ in range(100)]
    support = supported_definitions(g, model(), {"live"})
    assert support["predicates"] == []


def test_removed_sources_do_not_keep_old_schema_definitions():
    assert supported_definitions(graph(), model(), set()) == {"types": [], "predicates": []}


def test_supported_predicate_cannot_silently_change_direction_or_meaning():
    old = model(); changed = deepcopy(old)
    p = next(p for p in changed["predicates"] if p["name"] == "contains")
    p.update(source_parents=["part"], target_parents=["entity"], multivalued=False)
    out, guard = reuse_supported_model(changed, {"schema_model": old, "source_support": supported_definitions(graph(), old, {"live"})})
    assert next(p for p in out["predicates"] if p["name"] == "contains")["source_parents"] == ["entity"]
    assert {"kind": "predicate", "name": "contains"} in guard["deferred_changes"]


def test_price_change_scenario_preserves_all_claims_by_default():
    rows = [fact("september", "2026-09-01", 100),
            fact("october_email", "2026-11-15", 120, observed="2026-10-01"),
            fact("november_minutes", "", 100, observed_at="2026-11-18", time_basis="reported")]
    out = reconcile_facts(rows)
    assert out["stats"]["series"] == 1
    assert all(not r.get("superseded_by") for r in rows)
    for day in ("2026-10-20", "2026-11-20"):
        assert all(in_scope(r, query_scope(today=day)) for r in rows)
    assert not in_scope(rows[1], query_scope(as_of="2026-11-01"))
    rows[0]["doc_status"] = "archived"
    assert not in_scope(rows[0], query_scope())
    assert in_scope(rows[0], query_scope(history=True))


@pytest.mark.parametrize("dates", ["2025-02-11 and 2025-03-12", "before 2025", "2024–2026"])
def test_date_conditions_do_not_collapse_or_invent_effective_dates(dates):
    from kb_pipeline.graph.facts import normalize_measurements
    row = fact("a", "", 70, conditions={"record date": dates})
    normalize_measurements([row], [], units_by_id={})
    assert not row.get("observed_at") and not row.get("valid_from")
    assert row["conditions"]["record date"] == dates


def test_reported_date_survives_measurement_normalization_and_retrieval_projection():
    from kb_pipeline.graph.facts import normalize_measurements
    from kb_pipeline.graph.recall import spec_result_row
    row = fact("a", "", 70, observed_at="2025-02-11", time_basis="reported", conditions={"report date":"2025-02-11"})
    normalize_measurements([row], [], units_by_id={})
    assert not row.get("valid_from") and row["time_basis"] == "reported"
    # Verify the shared vector -> API projection retains explicit observation provenance.
    result = spec_result_row({**row, "gr_id":"a"})
    assert (result["observed_at"], result["time_basis"]) == ("2025-02-11", "reported")


def test_document_fallback_is_not_an_effective_start():
    from kb_pipeline.graph.facts import retain_source_time
    u = _unit("u"); u.doc_date = "2026-03-01"; u.text = "Device A costs 25 USD."
    row = {"valid_from": u.doc_date, "value":"25"}
    retain_source_time(row, u)
    assert not row.get("valid_from")
    u.text = "The price applies from 2026-03-01."
    row = {"valid_from":u.doc_date,"period_text":"from 2026-03-01"}
    retain_source_time(row,u)
    assert row["valid_from"] == u.doc_date


def test_cached_extraction_keeps_subject_and_value_while_requalifying_time():
    from kb_pipeline.graph.facts import retain_source_time
    u = _unit("u"); u.doc_date = "2026-03-01"; u.axis = u.doc_date
    u.text = "Operator Alice records Device B mass as 25 kg."
    row = {"subject": "Device B", "property": "mass", "value": "25", "unit": "kg",
           "valid_from": u.doc_date, "unit_id": u.unit_id}
    retain_source_time(row, u)
    recover_observation_date(row, u.text)
    row["id"] = fact_id(u.unit_id, row)
    link_facts([row], [], units_by_id={u.unit_id: u})
    assert (row["subject"], row["value"], row["unit"]) == ("Device B", "25", "kg")
    assert row["observed"] == u.doc_date and row["time_basis"] == "document"
    assert not row.get("valid_from")


def test_report_date_does_not_override_a_literal_historical_observation():
    from kb_pipeline.graph.facts import retain_source_time
    u = _unit("u"); u.doc_date = "2026-03-01"
    u.text = "2026-03-01 report. In 2013, output was 25 kg."
    row = {"valid_from": u.doc_date, "period_text": "2013", "value": "25"}
    retain_source_time(row, u)
    assert row["observed_at"] == "2013" and not row.get("valid_from")
    assert row["value"] == "25"
    # An interval is not reduced to one observation, and an unsupported quote is not used.
    for quote, end in [("2013 to 2026", "2026"), ("2011", "")]:
        row = {"valid_from": u.doc_date, "period_text": quote, "valid_until": end}
        retain_source_time(row, u)
        assert not row.get("observed_at")
    # A policy's explicitly stated start can differ from the data period it covers.
    u.text = "Policy effective 2026-03-01 covers records from 2013."
    row = {"valid_from": u.doc_date, "period_text": "2013", "fact_kind": "state"}
    retain_source_time(row, u)
    assert row["valid_from"] == u.doc_date and not row.get("observed_at")


def test_default_evidence_preserves_future_claims_and_explicit_time_filters():
    from kb_pipeline.graph import currency

    old = {"superseded_by": "new", "superseded_on": "2027-01-01"}
    new = {"valid_from": "2027-01-01"}
    ended = {"valid_until": "2026-11", "lapses": True}
    for day in ("2026-10-25", "2027-01-01"):
        for row in (old, new, ended):
            assert currency.row_status(row, day=day) == "current"
        assert currency.next_change_day([old, new, ended], today=day) == ""
    assert currency.row_status(new, day="2026-10-25", as_of=True) == "not_started"
    assert currency.row_status(new, day="2027-01-01", as_of=True) == "current"
    assert currency.row_status(ended, day="2026-11-30", as_of=True) == "current"
    assert currency.row_status(ended, day="2026-12-01", as_of=True) == "ended"
    # Explicit document governance still excludes a source from default recall.
    assert not in_scope({"doc_status": "archived"}, query_scope(today="2026-10-25"))


def test_summary_sources_respect_governance_without_automatic_future_hiding():
    from unittest.mock import patch
    from kb_pipeline.graph.corrections import parse_corrections
    from kb_pipeline.graph.current_view import filter_descriptions, graph_for_pages
    from kb_search import docstatus

    corr = parse_corrections({"documents": [{"path": "old.pdf", "status": "archived"}]})
    rows = [{"description": "mixed summary", "descriptions": ["old value 100", "current value 200"],
             "description_sources": [{"source": "old.pdf"}, {"source": "current.pdf"}]}]
    filter_descriptions(rows, corr, today="2026-10-25")
    assert rows[0]["descriptions"] == ["current value 200"] and not rows[0]["description"]
    graph = {"documents": {"old": {"status": "archived"}, "new": {}},
             "entities": [{"key": "old", "doc_ids": ["old"]}, {"key": "new", "doc_ids": ["new"]}],
             "relations": [{"doc_ids": ["new"], "valid_from": "2027-01-01"}], "specs": []}
    projected, _ = graph_for_pages(graph, [], today="2026-10-25")
    assert [e["key"] for e in projected["entities"]] == ["new"]
    assert projected["relations"] == graph["relations"]
    assert not in_scope(projected["relations"][0], query_scope(as_of="2026-10-25"))
    assert in_scope(projected["relations"][0], query_scope(as_of="2027-01-01"))
    with patch.object(docstatus, "corrections_of", return_value=corr):
        assert not docstatus.page_allowed(None, None, {"source_paths": ["current.pdf", "old.pdf"]}, {"mode": "current"})
        assert docstatus.page_allowed(None, None, {"source_paths": ["current.pdf"], "time_mode": "history"}, {"mode": "current"})
        assert not docstatus.page_allowed(None, None, {"compiled_day": "2026-10-24"}, {"mode": "as_of", "day": "2025-01-01"})
