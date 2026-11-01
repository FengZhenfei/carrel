"""A single bounded retry for an empty short-table extraction."""
from dataclasses import replace

import pytest

from kb_pipeline.graph.extract import ExtractionSchema, GraphExtractor
from kb_pipeline.graph.units import Unit
from kb_pipeline.graph.tabletext import contains_table
from _support import _client

SCHEMA = ExtractionSchema(entity_types=("machine", "sensor"), predicates=({"name": "contains"},))
EMPTY = '("unit"<|>listing<|>unexplained codes)<|COMPLETE|>'
RECORDS = ('("unit"<|>body<|>stated relationship)##'
           '("entity"<|>Machine M1<|>machine<|>A named machine<|>instance)##'
           '("entity"<|>Sensor S1<|>sensor<|>A named sensor<|>instance)##'
           '("relationship"<|>Machine M1<|>Sensor S1<|>contains<|>Machine M1 contains Sensor S1<|>8)'
           '<|COMPLETE|>')
TABLE = '| Machine | Component |\n|---|---|\n| Machine M1 | Sensor S1 |'


def unit(text=TABLE):
    return Unit("u", "d", "register.md", [], [], [], [], 80, text, block_types=["text"])


def test_empty_markdown_table_recovers_without_forcing_more_than_one_retry():
    r = GraphExtractor(_client([EMPTY, RECORDS]), SCHEMA, max_gleanings=1).extract(unit())
    assert r.calls == 2 and r.stats["table_recovery"] == 1 and r.unit_kind == "body"
    assert len(r.entities) == 2 and r.relations[0]["predicate"] == "contains"
    empty = GraphExtractor(_client([EMPTY, EMPTY]), SCHEMA, max_gleanings=1).extract(unit())
    assert empty.calls == 2 and not empty.entities and not empty.relations
    assert empty.unit_kind == "listing"


@pytest.mark.parametrize("case", ["nonempty", "prose", "boilerplate", "disabled"])
def test_existing_results_and_call_budget_are_preserved(case):
    response = RECORDS if case == "nonempty" else EMPTY
    u = unit("Machine M1 contains Sensor S1." if case == "prose" else TABLE)
    if case == "boilerplate": response = '("unit"<|>boilerplate<|>contents)<|COMPLETE|>'
    r = GraphExtractor(_client([response]), SCHEMA, max_gleanings=0 if case == "disabled" else 1).extract(u)
    assert r.calls == 1 and not r.stats.get("table_recovery")


def test_long_table_keeps_ordinary_gleaning():
    r = GraphExtractor(_client([EMPTY, RECORDS]), SCHEMA, max_gleanings=1).extract(replace(unit(), n_tokens=500))
    assert r.calls == 2 and r.stats["gleanings"] == 1 and not r.stats.get("table_recovery")


@pytest.mark.parametrize("text,expected", [(TABLE, True), ("HEADER: A | B\nx | y", True),
                                             ("a | b is a prose alternative", False), ("---", False)])
def test_table_detection_uses_structure(text, expected):
    assert bool(contains_table(text)) is expected
