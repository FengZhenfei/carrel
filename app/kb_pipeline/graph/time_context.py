"""Source-scoped time hints, shared by fact and relationship extraction.

Scope comes from chunk structure. The existing extractor decides whether a hint
applies to an assertion; quoted context is never another assertion to extract.
Reporting windows describe observations, not expiring states.
"""
from __future__ import annotations

import calendar
import hashlib
import json
import re
import unicodedata

from .temporal import find_dates, parse_date

VERSION = "source-time-v6"
_DOC = re.compile(r"^(?:本报告|本文件|全文|All .{0,90}in this report|Throughout this report)", re.I)
_SECTION = re.compile(r"^(?:本节|本章|本研究|本次实验|All .{0,90}in this section|In this section|For this study)", re.I)
_DECLARATION = re.compile(r"截至|(?:统计|报告|实验|测量|观测|研究|记录)(?:时间|期间|时点)|有效期|(?:自|从).{0,40}(?:生效|终止|适用)|as of|(?:reporting|experiment|measurement|observation) (?:period|window)|effective|valid from", re.I)
_VALIDITY = re.compile(r"生效|终止|有效期|effective|valid (?:from|until)", re.I)
_RELATIVE = re.compile(r"相对时间|记录时间|relative.*(?:based|anchor)", re.I)
_RELATIVE_EXPR = re.compile(r"去年|今年|明年|上月|本月|下月|次日|翌日|此前|之后|\b(?:last|next|previous|following)\s+(?:day|week|month|year)\b", re.I)
_FISCAL_DEFINITION = re.compile(r"fiscal year (?:is|ends|begins)|references to .{0,100}(?:fiscal|financial) periods|(?:财年|会计年度).{0,60}(?:起|止|结束|截止|截至)|(?:年度|季度).{0,40}均指.{0,20}财年", re.I)
_YEAR_END = re.compile(r"(?:fiscal )?years? ended|(?:财年|年度).{0,15}(?:截至|截止|结束)|(?:截至|截止).{0,45}(?:财年|年度)", re.I)
_QUARTER = re.compile(r"\b(?:quarter|Q[1-4])\b|季度", re.I)
_FISCAL_LABEL = re.compile(r"\b(?:fiscal(?:\s+year)?|FY)\s*(\d{4})\b|(\d{4})年?\s*(?:财年|会计年度)", re.I)
_FISCAL_COLUMN = re.compile(r"(?:fiscal year|FY|财年|会计年度)", re.I)


def _time_heading(text):
    """A date in a name is not a declaration of the section's time scope."""
    if _DECLARATION.search(text):
        return True
    return bool(re.fullmatch(r"\s*(?:\d{4}(?:[-/.]\d{1,2}(?:[-/.]\d{1,2})?)?|\d{4}年(?:\d{1,2}月(?:\d{1,2}日)?)?)\s*", text))


def _calendar_context(chunks):
    """Separate explicit year labels from dates that merely end an annual period.

    Fiscal years can be named for either their start or end year. Only a source
    correspondence supplies a mapping; an unlabelled date can validate an
    existing model date but cannot supply a missing date or rename a year.
    """
    declaration = None
    ends, end_dates = {}, {}

    def anchor(c, day, quote):
        return {"date": day, "quote": quote, "point_id": c.point_id, "chunk_uid": c.chunk_uid}

    def labelled(c, year, day, quote):
        entry = anchor(c, day, quote)
        ends.setdefault(year, {}).setdefault(day, entry)
        end_dates.setdefault(day, entry)

    for c in chunks:
        if c.block_type == "code":
            continue
        lines = c.text.splitlines()
        for i, line in enumerate(lines):
            if declaration is None and len(line) <= 500 and _FISCAL_DEFINITION.search(line):
                declaration = _record(c, line.strip(), "document")
                declaration.update(kind="calendar", dates=[])
            # A single, explicit label followed by an end date or a date range.
            # Stop at sentence boundaries so unrelated dates cannot form a pair.
            for sentence in re.split(r"[。;；]|(?<=[.!?])\s+", line):
                labels = list(_FISCAL_LABEL.finditer(sentence))
                if len(labels) != 1 or "|" in sentence or _QUARTER.search(sentence):
                    continue
                label = labels[0]
                tail = sentence[label.end():].strip()
                dates = find_dates(tail)
                if (re.match(r"(?:(?:runs |extends )?from\b|(?:ends?|ended)\b|[:：]|自|从|截至|截止)", tail, re.I)
                        and 1 <= len(dates) <= 2 and all(len(d) == 10 for d in dates)
                        and dates == sorted(dates)
                        and (len(dates) == 1 and re.match(r"(?:ends?|ended)\b|截至|截止", tail, re.I)
                             or len(dates) == 2 and re.search(r"\b(?:to|through)\b|至|到|[–—]", tail, re.I))):
                    labelled(c, label.group(1) or label.group(2), dates[-1], sentence.strip())
            match = _YEAR_END.search(line)
            if match and not _QUARTER.search(line):
                # Read the date governed by "year ended", not an auditor's
                # signature before it or another dated event on following lines.
                quote = line[match.start():][:600]
                dated_line = line[match.end():]
                dates = find_dates(dated_line)
                if not any(len(day) == 10 for day in dates):
                    for following in lines[i + 1:i + 4]:
                        if not following.strip(" |:-\t"):
                            continue
                        quote += "\n" + following
                        dated_line = following
                        dates = find_dates(dated_line)
                        break
                # Prose can continue with a signature/publication date. Only
                # tabular date columns support several annual ends here.
                dates = [d for d in dates if len(d) == 10]
                if "|" not in dated_line:
                    dates = dates[:1]
                for day in dates:
                    if len(day) == 10:
                        end_dates.setdefault(day, anchor(c, day, quote))
                # Adjacent table rows must explicitly pair each label and date.
                # Never zip dates by their calendar year or across missing cells.
                cells = [v.strip() for v in line.strip().strip("|").split("|")]
                previous = next((v for v in reversed(lines[max(0, i - 3):i]) if v.strip(" |:-\t")), "")
                labels = [v.strip() for v in previous.strip().strip("|").split("|")]
                if (len(cells) == len(labels) > 1 and _FISCAL_COLUMN.fullmatch(labels[0])
                        and _YEAR_END.fullmatch(cells[0])):
                    for year, date in zip(labels[1:], cells[1:]):
                        days = find_dates(date)
                        if re.fullmatch(r"\d{4}", year) and len(days) == 1 and len(days[0]) == 10:
                            labelled(c, year, days[0], previous + "\n" + line)
    if declaration:
        declaration["period_ends"] = {y: list(values.values()) for y, values in sorted(ends.items())}
        declaration["end_dates"] = list(end_dates.values())
        return [declaration]
    return []


def _norm(value):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value or ""))).casefold()


def _kind(text, *, heading=False):
    if _RELATIVE.search(text):
        return "relative_anchor"
    if _VALIDITY.search(text):
        return "validity"
    if re.search(r"截至|时点|as of", text, re.I):
        return "snapshot"
    if heading and len(find_dates(text)) == 1 and len(find_dates(text)[0]) == 10:
        return "snapshot"
    return "reporting_period"


def period_end(value):
    """The end of an explicitly named reporting year/month, at its precision."""
    value = parse_date(value) or ""
    if len(value) == 4:
        return value + "-12-31"
    if len(value) == 7:
        y, m = map(int, value.split("-"))
        return f"{value}-{calendar.monthrange(y, m)[1]:02d}"
    return value


def chunk_contexts(chunks):
    """Return contexts by chunk ID. No file-name dates or nearby arbitrary dates."""
    records = {}
    calendar_context = _calendar_context(chunks)
    document = []
    for c in chunks:
        values = []
        if c.block_type not in {"code", "table"}:
            # A compact, complete sentence is sufficient. Never consume the body
            # of a fenced example or let a lengthy paragraph become extra input.
            for line in c.text.splitlines():
                for quote in re.split(r"(?<=[。!?])\s*|(?<!\d)\.(?!\d)\s*", line.strip()):
                    if not quote or len(quote) > 500 or not _DECLARATION.search(quote):
                        continue
                    lead = re.sub(r"^(?:[-*+]\s+|\d+[.)]\s+)?(?:\*\*|__)?", "", quote)
                    scope = "document" if _DOC.match(lead) else "section" if _SECTION.match(lead) else ""
                    if not scope or not find_dates(quote):
                        continue
                    record = _record(c, quote, scope)
                    if not any(_norm(v["quote"]) == _norm(quote) for v in values):
                        values.append(record)
                    if scope == "document" and not any(_norm(v["quote"]) == _norm(quote) for v in document):
                        document.append(record)
        records[c.chunk_uid] = values
    result, active, previous = {}, {}, ()
    for c in chunks:
        path = tuple(c.section_path)
        if path != previous:
            common = 0
            for a, b in zip(path, previous):
                if a != b:
                    break
                common += 1
            active = {level: values for level, values in active.items() if level <= common}
        local = [r for r in records[c.chunk_uid] if r["scope"] == "section"]
        if local:
            # Overlap copies of the same declaration do not start a new scope.
            old = active.get(len(path), []) if path == previous else []
            active[len(path)] = {(_norm(r["quote"])): r for r in local + old}.values()
            active[len(path)] = list(active[len(path)])
        elif path != previous and path and _time_heading(path[-1]) and len(find_dates(path[-1])) == 1:
            active[len(path)] = [_record(c, path[-1], "section", heading=True)]
        selected = active[max(active)] if active else document
        result[c.chunk_uid] = list(selected) + calendar_context
        previous = path
    return result


def _record(chunk, quote, scope, *, heading=False):
    return {"id": "t-" + hashlib.sha256((chunk.doc_id + chunk.block_id + quote).encode()).hexdigest()[:16],
            "quote": quote, "scope": scope, "kind": _kind(quote, heading=heading), "dates": find_dates(quote),
            "doc_id": chunk.doc_id, "content_version": chunk.content_version,
            "section": list(chunk.section_path), "point_id": chunk.point_id, "chunk_uid": chunk.chunk_uid}


def context_key(contexts):
    return json.dumps(contexts, ensure_ascii=False, sort_keys=True)


def conflicted(contexts):
    # Repeated equivalent declarations agree; incompatible declarations remain
    # unresolved rather than selecting the last date in a document.
    contexts = [r for r in contexts if r["kind"] != "calendar"]
    for record in contexts:
        dates = record["dates"]
        if (len(dates) > 2 or (record["kind"] in {"snapshot", "relative_anchor"} and len(dates) != 1)
                or (len(dates) == 2 and dates[0] > dates[1])):
            return True
    return len({tuple(r["dates"]) for r in contexts}) > 1


INSTRUCTIONS = """
Time semantics:
- Publication date is provenance, not the occurrence/effective date of every assertion.
- A local date or table column overrides section context; section context overrides document context.
- Years in names, identifiers, instrument maturities or law titles do not date other assertions.
- Calendar records explain fiscal conventions, not the time of every statement. For annual aggregates
  use a uniquely supported period_ends mapping for that year. These mappings require explicit labels
  in the source. end_dates are source date anchors, not year-label mappings; do not infer a fiscal
  label from a date's calendar year. Never assume a fiscal year ends on
  December 31. Do not calculate fiscal quarter boundaries from an annual end date alone. If the end
  is unresolved, keep period_text and leave structured time empty. Ordinary events dated only to a
  year/month retain that precision. Local explicit dates still take precedence.
- Keep timeless definitions/examples undated. Do not infer time merely from a nearby date.
- A reporting/experimental period is NOT a validity period. Aggregate results and recorded experimental
  settings describe observations over that window. Put the complete window in period_text and its end
  in observed_at; leave valid_from/valid_until empty. A full-period aggregate is not completed before
  the window ends. A local single-event date still wins over the window.
- Use valid_from/valid_until only for explicit event timing or genuinely effective/expiring states
  (appointments, policies, contracts), never to make historical experiment settings expire.
The context records below are same-document source quotes, not additional assertions to extract.
Use a record only when its stated scope applies to the assertion. When used, copy its id into
time_source_id and its quote into time_source_quote. For relationships, these fields plus observed_at
and period_text go in the final JSON metadata object. For facts they go in the fact JSON object.
For relative expressions use a relative_anchor if present, otherwise the document date.
If contexts conflict, leave inherited time unresolved; local explicit time can still be used.
"""


def prompt_context(unit):
    records = getattr(unit, "time_contexts", [])
    # Keep operational source IDs/version hashes out of the prompt; the program
    # attaches those from the validated record, not from model-generated strings.
    hints = [{k: r[k] for k in ("id", "scope", "kind", "quote", "dates")} for r in records]
    for hint, record in zip(hints, records):
        if record["kind"] == "calendar":
            hint["period_ends"] = {y: [{"date": e["date"], "quote": e["quote"]} for e in ends]
                                   for y, ends in record["period_ends"].items()}
            hint["end_dates"] = [{"date": e["date"], "quote": e["quote"]} for e in record.get("end_dates", [])]
    return INSTRUCTIONS + "\nTIME CONTEXT: " + json.dumps({"conflict": conflicted(records), "records": hints}, ensure_ascii=False)


def _time_basis(row):
    if "fact_kind" in row or "time_basis" in row or row.get("period_text") or row.get("observed_at"):
        row["time_basis"] = ("explicit" if row.get("valid_from") or row.get("valid_until")
                             else "reported" if row.get("observed_at") else "unknown")


def _unresolved(row, quote):
    for key in ("observed_at", "valid_from", "valid_until"):
        row.pop(key, None)
    row.pop("time_source", None)
    row["period_text"] = quote
    row["time_basis"] = "unknown"


def _fiscal_time(row, raw, unit, calendars):
    """Check fiscal precision using source anchors; preserve uncertainty explicitly."""
    quote = str(raw.get("period_text") or row.get("period_text") or "").strip()
    if not quote or not calendars or _RELATIVE_EXPR.search(quote) or row.get("valid_from") or row.get("valid_until"):
        return False
    sid = str(raw.get("time_source_id") or "")
    calendar = next((c for c in calendars if c["id"] == sid), None)
    if sid and calendar is None:
        return False                         # do not rescue an invalid source ID
    dates = find_dates(quote)
    requested = str(row.get("observed_at") or "")
    full_dates = [d for d in dates if len(d) == 10]
    if requested in full_dates and requested in find_dates(unit.text):
        return False                         # local complete dates are authoritative
    labelled_year = len(dates) == 1 and len(dates[0]) == 4
    if not labelled_year and not (calendar and full_dates):
        return False
    if labelled_year and _norm(quote) not in _norm(unit.text):
        _unresolved(row, quote)
        return True
    # A fiscal quarter cannot be placed by substituting the calendar quarter.
    if _QUARTER.search(quote):
        local = find_dates(str(raw.get("evidence") or ""))
        if not any(d == row.get("observed_at") and len(d) == 10 and d in find_dates(unit.text) for d in local):
            _unresolved(row, quote)
        else:
            _time_basis(row)
        return True
    # Only resolve a full-period observation. A year-only event stays year-only.
    if len(requested) <= 4 and calendar is None:
        return False
    calendar = calendar or calendars[0]
    ends = {e["date"]: e for c in calendars for e in c["period_ends"].get(dates[0], [])} if labelled_year else {}
    # A model's full date can be supported by an annual header even when its
    # quotation is not a verbatim copy of the separate fiscal convention.
    if not ends and len(requested) == 10 and (not full_dates or requested in full_dates):
        ends = {e["date"]: e for c in calendars for e in c.get("end_dates", []) if e["date"] == requested}
    if len(ends) != 1:
        _unresolved(row, quote)
    else:
        day, anchor = next(iter(ends.items()))
        row.pop("valid_from", None)
        row.pop("valid_until", None)
        row["observed_at"] = day
        row["period_text"] = quote
        row["time_source"] = {**{k: v for k, v in calendar.items() if k not in {"period_ends", "end_dates"}}, "anchor": anchor}
        _time_basis(row)
    return True


def apply_time_context(row, raw, unit):
    """Attach verified context; normalize reporting windows without guessing scope."""
    records = getattr(unit, "time_contexts", [])
    calendars = [r for r in records if r["kind"] == "calendar"]
    for key in ("observed_at", "valid_from", "valid_until"):
        if key in row:
            parsed = parse_date(str(row[key] or ""))
            if parsed:
                row[key] = parsed
            else:
                row.pop(key, None)
    sid = str(raw.get("time_source_id") or "")
    chosen = next((r for r in records if r["id"] == sid), None)
    if records or sid:
        # Available context is not proof that it applies to this assertion.
        # Keep omissions unresolved instead of falling back to a source date.
        row["time_basis"] = "unknown"
    quote = str(raw.get("period_text") or "")
    # Explicit local evidence can never be erased by a bad inherited reference.
    local_quote = (quote and _norm(quote) in _norm(unit.text)
                   and not any(_norm(quote) in _norm(r["quote"]) for r in records))
    local_dates = find_dates(quote) if local_quote else []
    local_relative = bool(local_quote and _RELATIVE_EXPR.search(quote))
    if _fiscal_time(row, raw, unit, calendars):
        return
    if sid and (not chosen or conflicted(records) or _norm(raw.get("time_source_quote")) != _norm(chosen["quote"])):
        if not local_dates and not local_relative:
            for k in ("valid_from", "valid_until", "observed_at"):
                if len(str(row.get(k) or "")) != 10 or row.get(k) not in find_dates(unit.text):
                    row.pop(k, None)
        _time_basis(row)
        return
    if chosen and ((local_dates and tuple(local_dates) != tuple(chosen["dates"]))
                   or (local_relative and chosen["kind"] != "relative_anchor")):
        _time_basis(row)
        return
    if chosen and chosen["kind"] == "calendar":
        _time_basis(row)
        return
    # An inherited period may not overwrite a date/bound explicitly supported
    # by local text, even when the model omitted period_text.
    if chosen and any(row.get(k) in find_dates(unit.text) for k in ("valid_from", "valid_until", "observed_at")
                      if row.get(k)) and not local_quote:
        dates = chosen["dates"]
        # Do not overwrite a local date. A matching, already validated context
        # can still supply provenance for that same observation or interval.
        observation = (chosen["kind"] in {"reporting_period", "snapshot"}
                       and not row.get("valid_from") and not row.get("valid_until")
                       and row.get("observed_at") == (period_end(dates[-1]) if chosen["kind"] == "reporting_period" else dates[-1]))
        interval = (chosen["kind"] == "validity" and not row.get("observed_at")
                    and row.get("valid_from") == dates[0]
                    and (row.get("valid_until") or "") == (dates[1] if len(dates) == 2 else ""))
        if observation or interval:
            row["time_source"] = dict(chosen)
        _time_basis(row)
        return
    if chosen:
        dates, kind = chosen["dates"], chosen["kind"]
        if kind == "reporting_period":
            row.pop("valid_from", None)
            row.pop("valid_until", None)
            row["observed_at"] = period_end(dates[-1])
            row["period_text"] = chosen["quote"]
            if "fact_kind" in row:
                row["fact_kind"] = "observation"
        elif kind == "snapshot":
            row.pop("valid_from", None)
            row.pop("valid_until", None)
            row["observed_at"] = dates[-1]
            row["period_text"] = chosen["quote"]
        elif kind == "validity":
            row["valid_from"] = dates[0]
            if len(dates) == 2:
                row["valid_until"] = dates[1]
        row["time_source"] = dict(chosen)
    # Explicit local statistical periods follow the same semantics even when
    # their date declaration fits inside the current unit.
    elif local_quote and _DECLARATION.search(quote) and _kind(quote) == "reporting_period" and local_dates:
        row.pop("valid_from", None)
        row.pop("valid_until", None)
        row["observed_at"] = period_end(local_dates[-1])
        if "fact_kind" in row:
            row["fact_kind"] = "observation"
    _time_basis(row)


def time_fields(row):
    return {k: row[k] for k in ("observed_at", "period_text", "time_source", "time_basis") if row.get(k)}
