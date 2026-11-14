from types import SimpleNamespace as NS
import pytest
from kb_pipeline.graph.time_context import chunk_contexts, apply_time_context, period_end
from kb_pipeline.graph.currency import row_status, mark_fact_lapses
from kb_pipeline.graph.quality import relation_qualifier
from kb_pipeline.graph.extract import consolidate
from kb_pipeline.graph.units import build_units, ChunkRef


def chunk(i, text, path=("Report",), kind="paragraph"):
    return ChunkRef(point_id=f"p{i}", chunk_uid=f"c{i}", doc_id="doc", content_version="v1",
                    chunk_index=i, block_id=f"b{i}", block_type=kind, section_path=list(path),
                    text=text, n_tokens=100, rel_path="report.md", filename="report.md")


def test_scope_and_sibling_reset():
    cs = [chunk(0, "本报告统计时点为2024-12-31。"),
          chunk(1, "本节实验期间为2023年5月至2023年7月。", ("Report", "Study")),
          chunk(2, "测量结果为42。", ("Report", "Study", "Result")),
          chunk(3, "在岗人员有81人。", ("Report", "Other"))]
    hints = chunk_contexts(cs)
    assert hints["c2"][0]["dates"] == ["2023-05", "2023-07"]
    assert hints["c3"][0]["dates"] == ["2024-12-31"]
    units = build_units(cs, kb_id="kb", unit_chunks=3)
    assert len({u.time_contexts[0]["quote"] for u in units}) == 2
    assert all(len({hints[c][0]["quote"] for c in u.chunk_uids}) == 1 for u in units)


def test_quoted_and_code_dates_are_not_context():
    cs = [chunk(0, '引用：“本报告统计时点为2024-12-31。”'),
          chunk(1, '本报告统计时点为2024-12-31。', kind="code"),
          chunk(2, '元件MX2048A和v2.4均为标识符。')]
    assert not any(chunk_contexts(cs).values())


def apply(quote, row=None, text="实验记录", **raw_extra):
    c = chunk(0, quote)
    contexts = chunk_contexts([c])["c0"]
    assert contexts
    unit = NS(time_contexts=contexts, text=text)
    result = dict(row or {"fact_kind": "observation", "value": "0.9"})
    raw = dict(result, time_source_id=contexts[0]["id"], time_source_quote=quote, **raw_extra)
    apply_time_context(result, raw, unit)
    return result


def test_reporting_window_ends_but_observation_does_not_expire():
    f = apply("本报告实验期间为2024-10-01至2024-12-31。",
              {"fact_kind": "state", "valid_from": "2024-10-01", "valid_until": "2024-12-31"})
    mark_fact_lapses([f], [])
    assert f["observed_at"] == "2024-12-31"
    assert f["time_basis"] == "reported"
    assert not f.get("valid_until") and not f.get("lapses")
    assert row_status(f, as_of=True, day="2024-11-15") == "not_started"
    assert row_status(f, as_of=True, day="2024-12-31") == "current"
    assert row_status(f, as_of=True, day="2026-01-01") == "current"
    assert f["time_source"]["content_version"] == "v1"


def test_validity_is_still_expiring():
    f = apply("本节有效期为2025-03-01至2025-08-31。", {"fact_kind": "state"})
    mark_fact_lapses([f], [])
    assert f["valid_from"] == "2025-03-01" and f["lapses"]
    assert row_status(f, as_of=True, day="2025-09-01") == "ended"


def test_local_event_overrides_inherited_window():
    f = apply("本报告实验期间为2024-10-01至2024-12-31。",
              {"fact_kind": "observation", "valid_from": "2024-10-12"},
              text="2024-10-12执行单次测量。", period_text="2024-10-12")
    assert f["valid_from"] == "2024-10-12" and not f.get("time_source")


def test_local_relative_time_is_not_replaced_by_snapshot():
    f = apply("本报告统计截至2025-12-31。",
              {"fact_kind": "observation", "observed_at": "2024-11"},
              text="去年11月完成测量。", period_text="去年11月")
    assert f["observed_at"] == "2024-11" and not f.get("time_source")


def test_reference_is_not_dated_by_context():
    contexts = chunk_contexts([chunk(0, "本报告统计时点为2024-12-31。")])["c0"]
    f = {"fact_kind": "reference", "value": "8192"}
    apply_time_context(f, dict(f), NS(time_contexts=contexts, text="缓冲区上限定义为8192。"))
    assert not f.get("valid_from") and not f.get("observed_at")


def test_ambiguous_or_fabricated_source_cannot_supply_time():
    cs = [chunk(0, "本报告统计时点为2023-12-31。\n本报告统计时点为2024-12-31。")]
    hints = chunk_contexts(cs)["c0"]
    row = {"observed_at": "2024-12-31"}
    apply_time_context(row, {**row, "time_source_id": hints[0]["id"], "time_source_quote": hints[0]["quote"]},
                       NS(time_contexts=hints, text="容量618。"))
    assert not row.get("observed_at")
    row = {"observed_at": "2024-12-31"}
    apply_time_context(row, {**row, "time_source_id": "made-up"}, NS(time_contexts=[], text="容量618。"))
    assert not row.get("observed_at")


def test_observation_windows_are_not_merged():
    r = {"source": "A", "target": "B", "predicate": "uses", "observed_at": "2024-12-31", "period_text": "2024"}
    earlier = dict(r, observed_at="2023-12-31")
    assert relation_qualifier(r) != relation_qualifier(earlier)
    _, rows = consolidate([], [r, earlier])
    assert len(rows) == 2 and rows[0]["period_text"] == "2024"


def test_reporting_precision():
    assert period_end("2024") == "2024-12-31"
    assert period_end("2024-02") == "2024-02-29"


def test_year_in_names_does_not_create_a_reporting_period():
    for heading in ("The 0.000% 2025 Notes", "SECTION 906 OF AN ACT OF 2002", "Model 2027 edition"):
        hints = chunk_contexts([chunk(0, "Issued on 2019-11-15.", ("Report", heading))])
        assert hints["c0"] == []
    assert chunk_contexts([chunk(0, "统计值为8。", ("Report", "2024"))])["c0"]


def fiscal_unit(text, extra=()):
    cs = [chunk(0, "The Company's fiscal year ends in September."),
          chunk(1, "| Fiscal year | 2024 | 2023 |\n| Years ended | September 28, 2024 | September 30, 2023 |"), *extra,
          chunk(9, text)]
    return NS(time_contexts=chunk_contexts(cs)["c9"], text=text)


def test_fiscal_end_uses_document_anchors_not_december():
    unit = fiscal_unit("Revenue in 2024 was 12.")
    f = {"fact_kind": "observation", "value": "12", "observed_at": "2024-12-31", "period_text": "2024"}
    apply_time_context(f, dict(f), unit)
    assert f["observed_at"] == "2024-09-28"
    assert f["time_source"]["anchor"]["point_id"] == "p1"
    assert row_status(f, day="2024-10-01", as_of=True) == "current"
    assert row_status(f, day="2024-09-01", as_of=True) == "not_started"


def test_fiscal_quarter_without_boundary_is_retained_but_not_confirmed_as_of():
    from kb_pipeline.graph.currency import in_scope, query_scope, cypher_in_scope
    from kb_pipeline.graph.facts import link_facts, when_text
    unit = fiscal_unit("An extra week occurred in the first quarter of 2023.")
    f = {"fact_kind": "observation", "value": "extra week", "period_text": "first quarter of 2023",
         "observed_at": "2023-12-30", "observed": "2024-09-28"}
    apply_time_context(f, dict(f), unit)
    assert f["value"] == "extra week" and f["time_basis"] == "unknown"
    assert not f.get("observed_at")
    assert row_status(f, day="2025-01-01", as_of=True) == "time_unknown"
    assert not in_scope(f, query_scope(as_of="2025-01-01"))
    assert in_scope(f, query_scope(history=True)) and in_scope(f, query_scope())
    assert "time_basis" in cypher_in_scope("f", observed="observed", scope=query_scope(as_of="2025-01-01"))
    f["unit_id"] = "u"
    unit.doc_id, unit.rel_path, unit.section_label = "doc", "report.pdf", "Summary"
    unit.point_ids, unit.axis, unit.doc_date = ["p9"], "2024-09-28", "2024-09-28"
    link_facts([f], [], units_by_id={"u": unit})
    assert f["time_basis"] == "unknown" and not in_scope(f, query_scope(as_of="2025-01-01"))
    assert when_text(f) == "first quarter of 2023"


def test_conflicting_fiscal_anchors_do_not_pick_one():
    unit = fiscal_unit("Revenue in 2024 was 12.", [chunk(2, "Fiscal 2024 ended September 30, 2024.")])
    f = {"fact_kind": "observation", "value": "12", "observed_at": "2024-12-31", "period_text": "2024"}
    apply_time_context(f, dict(f), unit)
    assert not f.get("observed_at") and f["period_text"] == "2024"


def test_fiscal_context_does_not_change_local_dates_or_year_only_events():
    unit = fiscal_unit("On October 18, 2024 there were 30 shares. A launch occurred in 2023.")
    for date, quote in (("2024-10-18", "October 18, 2024"), ("2023", "2023")):
        f = {"fact_kind": "observation", "observed_at": date, "period_text": quote}
        apply_time_context(f, dict(f), unit)
        assert f["observed_at"] == date


def test_local_bound_is_preserved_when_model_omits_period_quote():
    f = apply("本报告统计期间为2024年。", {"valid_from": "2023-07-02"}, text="自2023-07-02生效。")
    assert f["valid_from"] == "2023-07-02" and not f.get("observed_at")
    row = {"valid_from": "2019-11-15", "valid_until": "2025-11-15"}
    apply_time_context(row, {**row, "time_source_id": "rejected-title"},
                       NS(time_contexts=[], text="Issued November 15, 2019, due November 15, 2025."))
    assert row["valid_from"] == "2019-11-15" and row["valid_until"] == "2025-11-15"


def test_calendar_mappings_do_not_conflict_with_section_context():
    from kb_pipeline.graph.time_context import conflicted
    unit = fiscal_unit("本节统计截至2024-04-01。")
    assert not conflicted(unit.time_contexts)


def test_annual_anchor_does_not_capture_audit_or_signature_dates():
    extra = [chunk(2, "Our report dated November 1, 2024 for the fiscal year ended September 28, 2024, signed November 2, 2024.\nDate: November 1, 2024")]
    unit = fiscal_unit("2024 revenue", extra)
    calendar = next(r for r in unit.time_contexts if r["kind"] == "calendar")
    assert [e["date"] for e in calendar["period_ends"]["2024"]] == ["2024-09-28"]
    assert not {"2024-11-01", "2024-11-02"} & {e["date"] for e in calendar["end_dates"]}


def test_markdown_emphasis_keeps_the_original_quote():
    quote = "- **本报告**统计截至2024年12月31日。"
    hints = chunk_contexts([chunk(0, quote)])["c0"]
    assert hints[0]["scope"] == "document" and hints[0]["quote"] == quote


def test_dotted_dates_and_ambiguous_snapshot():
    from kb_pipeline.graph.time_context import conflicted
    hints = chunk_contexts([chunk(0, "本报告统计截至2024.05.31。")])["c0"]
    assert hints[0]["dates"] == ["2024-05-31"]
    hints = chunk_contexts([chunk(0, "本报告统计截至2023年或2024年，记录不明。")])["c0"]
    assert conflicted(hints)


def test_overlap_does_not_change_the_time_reference():
    quote = "本节统计截至2024-12-31。"
    hints = chunk_contexts([chunk(0, quote), chunk(1, quote + "\n新增内容")])
    assert hints["c0"] == hints["c1"]


def test_context_change_invalidates_an_unchanged_body_cache():
    from kb_pipeline.graph.unit_cache import fingerprints
    tables = NS(ambiguous_names=lambda: set(), wants=lambda u: False)
    unit = NS(unit_id="same-body", time_contexts=[{"quote": "2023", "content_version": "v1"}])
    before = fingerprints("base", [unit], set(), tables)
    unit.time_contexts = [{"quote": "2024", "content_version": "v2"}]
    assert fingerprints("base", [unit], set(), tables) != before


def test_temporal_fields_survive_retrieval_projection():
    from kb_pipeline.graph.vectors import spec_payload, relation_payload
    from kb_pipeline.graph.recall import spec_result_row, RELATION_TIME_KEYS
    from kb_search.service import SPEC_FIELDS, RELATION_FIELDS
    from kb_search.graphwalk import RELATION_TIME_FIELDS
    fact = apply("本报告实验期间为2024年10月至2024年12月。")
    fact.update(id="f", subject="Method", property="Score", value="0.9")
    payload = spec_payload(fact, kb_id="kb", source_collection="kb", graph_version="v")
    result = spec_result_row(payload)
    for key in ("observed_at", "period_text", "time_source"):
        assert result[key] == fact[key] and key in SPEC_FIELDS
    relation = dict(fact, source="A", target="B", source_key="a", target_key="b", predicate="uses")
    payload = relation_payload(relation, kb_id="kb", source_collection="kb", graph_version="v", point_ids=[])
    for key in ("observed_at", "period_text", "time_source"):
        assert payload[key] == fact[key]
        assert key in RELATION_TIME_KEYS and key in RELATION_FIELDS and key in RELATION_TIME_FIELDS


@pytest.mark.parametrize("source,year,end", [
    ("| Fiscal year | 2031 | 2030 |\n| Years ended | March 31, 2032 | March 31, 2031 |", "2031", "2032-03-31"),
    ("| Fiscal year | 2030 | 2031 |\n| --- | --- | --- |\n| Years ended | November 30, 2030 | November 30, 2031 |", "2031", "2031-11-30"),
    ("Fiscal 2031 runs from April 1, 2031 to March 31, 2032.", "2031", "2032-03-31"),
    ("FY2031 ended November 30, 2031.", "2031", "2031-11-30"),
    ("2031财年自2031年4月1日至2032年3月31日。", "2031", "2032-03-31"),
    ("| 财年 | 2031 | 2030 |\n| 年度截至 | 2032-03-31 | 2031-03-31 |", "2030", "2031-03-31"),
])
def test_explicit_fiscal_labels_do_not_assume_start_or_end_year(source, year, end):
    cs = [chunk(0, "The fiscal year is defined as follows.\n" + source), chunk(1, f"Total processed in {year}: 12.")]
    unit = NS(time_contexts=chunk_contexts(cs)["c1"], text=cs[1].text)
    row = {"fact_kind": "observation", "period_text": year, "observed_at": year + "-12-31"}
    apply_time_context(row, dict(row), unit)
    assert row["observed_at"] == end
    assert row["time_source"]["content_version"] == "v1"
    assert row["time_source"]["anchor"]["point_id"] == "p0"


def unlabelled_fiscal_unit():
    cs = [chunk(0, "The fiscal year ends in March."),
          chunk(1, "Years ended\n| March 31, 2032 | March 31, 2031 |"),
          chunk(2, "Total processed in 2031: 12.")]
    return NS(time_contexts=chunk_contexts(cs)["c2"], text=cs[2].text)


def test_unlabelled_annual_dates_validate_but_never_infer_fiscal_labels():
    unit = unlabelled_fiscal_unit()
    calendar = unit.time_contexts[0]
    assert calendar["period_ends"] == {}
    for date, expected in (("2032-03-31", "2032-03-31"), ("2031-12-31", None), ("", None)):
        row = {"fact_kind": "observation", "observed_at": date, "period_text": "2031"}
        apply_time_context(row, dict(row, time_source_id=calendar["id"], time_source_quote=calendar["quote"]), unit)
        assert row.get("observed_at") == expected


def test_source_supported_header_quote_does_not_have_to_equal_convention_quote():
    unit = unlabelled_fiscal_unit()
    row = {"fact_kind": "observation", "observed_at": "2032-03-31", "period_text": "fiscal year ended March 31, 2032"}
    apply_time_context(row, dict(row, time_source_id=unit.time_contexts[0]["id"],
                                 time_source_quote="fiscal year ended March 31, 2032"), unit)
    assert row["observed_at"] == "2032-03-31"
    assert row["time_source"]["anchor"]["quote"].startswith("Years ended")
    assert row_status(row, day="2032-03-30", as_of=True) == "not_started"
    assert row_status(row, day="2032-03-31", as_of=True) == "current"


@pytest.mark.parametrize("sid,date", [("fabricated", "2032-03-31"), (None, "2032-04-01")])
def test_header_validation_requires_real_source_and_supported_date(sid, date):
    unit = unlabelled_fiscal_unit()
    row = {"fact_kind": "observation", "observed_at": date, "period_text": f"fiscal year ended {date}"}
    apply_time_context(row, dict(row, time_source_id=sid or unit.time_contexts[0]["id"],
                                 time_source_quote=row["period_text"]), unit)
    assert not row.get("observed_at")


@pytest.mark.parametrize("source", [
    "| Fiscal year | 2031 | 2030 |\n| Years ended | March 31, 2032 |",
    "| Fiscal year | 2031 | 2030 |\n| Years ended | | March 31, 2031 |",
    "Fiscal 2031 begins April 1, 2031.",
    "Fiscal 2031 review signed March 31, 2032.",
    "Fiscal 2031 quarter ended March 31, 2032.",
    "Fiscal 2031 and fiscal 2030 ended March 31, 2032 and March 31, 2031.",
])
def test_incomplete_or_nonannual_labels_do_not_supply_mapping(source):
    cs = [chunk(0, "The fiscal year ends in March."), chunk(1, source)]
    calendar = chunk_contexts(cs)["c1"][0]
    assert not calendar["period_ends"].get("2031")


def test_relative_date_omissions_are_not_filled_from_notes():
    quote = "本节记录时间为2030年6月20日，以下相对时间均以该记录时间为准。"
    f = apply(quote, {"fact_kind": "state", "period_text": "次日", "note": "即2030-06-21"},
              text="名额从次日起调整。")
    assert not f.get("observed_at") and not f.get("valid_from")
    assert f["time_basis"] == "unknown"


@pytest.mark.parametrize("kind", ["state", "observation", "reference"])
def test_omitted_time_stays_unknown_through_linking_projection_and_scope(kind):
    from kb_pipeline.graph.facts import normalize_fact, link_facts, normalize_measurements, when_text
    from kb_pipeline.graph.currency import in_scope, query_scope, cypher_in_scope
    from kb_pipeline.graph.vectors import spec_payload
    from kb_search.evidence import spec_hint
    quote = "本节有效期为2030-02-01至2030-07-31。"
    contexts = chunk_contexts([chunk(0, quote)])["c0"]
    unit = NS(time_contexts=contexts, text=quote + "\n额度为63。", doc_id="doc", rel_path="report.md",
              section_label="Terms", point_ids=["p0"], axis="2032-01-05", doc_date="2032-01-05")
    raw = {"subject": "Account", "property": "limit", "value": "63", "fact_kind": kind}
    row = normalize_fact(raw)
    apply_time_context(row, raw, unit)
    row.update(unit_id="u", id="f")
    link_facts([row], [], units_by_id={"u": unit})
    normalize_measurements([row], [], units_by_id={"u": unit})
    assert row["time_basis"] == "unknown" and row["observed"] == unit.doc_date
    assert not any(row.get(k) for k in ("observed_at", "valid_from", "valid_until", "time_source", "period_text"))
    assert when_text(row) == ""
    payload = spec_payload(row, kb_id="kb", graph_version="g", source_collection="kb")
    assert payload["time_basis"] == "unknown"
    assert "time unresolved" in spec_hint(payload) and unit.doc_date not in spec_hint(payload)
    for day in ("2030-05-01", "2033-01-01"):
        assert row_status(payload, day=day, as_of=True) == "time_unknown"
        assert not in_scope(payload, query_scope(as_of=day))
        assert in_scope(payload, query_scope(history=True, as_of=day))
    assert in_scope(payload, query_scope())
    clause = cypher_in_scope("f", observed="observed", scope=query_scope(as_of="2033"))
    assert "coalesce(f.time_basis, '') <> 'unknown'" in clause
    assert "period_text" not in clause


def test_missing_relation_time_is_not_replaced_by_document_observation():
    quote = "本报告统计截至2030-06-30。"
    contexts = chunk_contexts([chunk(0, quote)])["c0"]
    row = {"source": "A", "target": "B", "predicate": "uses"}
    apply_time_context(row, dict(row), NS(time_contexts=contexts, text="A uses B."))
    row["observed_first"] = "2032-01-05"
    assert row["time_basis"] == "unknown"
    assert row_status(row, day="2033-01-01", as_of=True) == "time_unknown"


def test_literal_time_condition_remains_available_without_document_date_fallback():
    from kb_pipeline.graph.facts import normalize_fact, link_facts, normalize_measurements
    quote = "本报告统计截至2032-06-30。"
    contexts = chunk_contexts([chunk(0, quote)])["c0"]
    unit = NS(time_contexts=contexts, text="| Year | 2030 |\n| Rate | 21% |", doc_id="doc",
              rel_path="report.md", section_label="Rates", point_ids=["p0"], axis="2032-06-30", doc_date="2032-06-30")
    raw = {"subject": "Account", "property": "rate", "value": "21", "conditions": {"year": "2030"}}
    row = normalize_fact(raw)
    apply_time_context(row, raw, unit)
    row["unit_id"] = "u"
    link_facts([row], [], units_by_id={"u": unit})
    normalize_measurements([row], [], units_by_id={"u": unit})
    assert row["observed_at"] == "2030" and row["time_basis"] == "reported"
    assert row["observed"] == "2032-06-30" and not row.get("time_source")


@pytest.mark.parametrize("quote,fields", [
    ("本研究实验期间为2030-02-01至2030-07-31。", {"observed_at": "2030-07-31"}),
    ("In this section, reporting period: February 1, 2030 to July 31, 2030.", {"observed_at": "2030-07-31"}),
    ("本节统计截至2030-07-31。", {"observed_at": "2030-07-31"}),
    ("本节有效期为2030-02-01至2030-07-31。", {"valid_from": "2030-02-01", "valid_until": "2030-07-31"}),
])
def test_local_matching_time_keeps_verified_provenance_without_rewriting(quote, fields):
    row = apply(quote, {"fact_kind": "observation", **fields}, text=quote + "\nMeasured value: 63.")
    assert all(row[k] == v for k, v in fields.items())
    assert {k: row[k] for k in ("valid_from", "valid_until", "observed_at") if row.get(k)} == fields
    assert row["time_source"]["quote"] == quote
    assert row["time_source"]["content_version"] == "v1"
    assert row["time_source"]["point_id"] == "p0"


def test_same_unit_date_does_not_attach_unrelated_period_provenance():
    quote = "本研究实验期间为2030-02-01至2030-07-31。"
    row = apply(quote, {"fact_kind": "observation", "observed_at": "2030-05-12"},
                text=quote + "\nOn 2030-05-12 a separate measurement was made.")
    assert row["observed_at"] == "2030-05-12" and row["time_basis"] == "reported"
    assert not row.get("time_source")


@pytest.mark.parametrize("bad", [{"time_source_id": "invented"}, {"time_source_quote": "not the source"}])
def test_local_date_does_not_let_invalid_provenance_through(bad):
    quote = "本研究实验期间为2030-02-01至2030-07-31。"
    contexts = chunk_contexts([chunk(0, quote)])["c0"]
    row = {"fact_kind": "observation", "observed_at": "2030-07-31"}
    raw = {**row, "time_source_id": contexts[0]["id"], "time_source_quote": quote, **bad}
    apply_time_context(row, raw, NS(time_contexts=contexts, text=quote))
    assert row["observed_at"] == "2030-07-31" and not row.get("time_source")
