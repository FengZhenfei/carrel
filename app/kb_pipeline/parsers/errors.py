"""Exception types of the parsing layer. Kept in the parsers package rather than parse_job: the router
(parsers.router) also needs to raise a "deterministic failure", and parse_job imports the router in
turn, so placing it there would create a circular import. parse_job still re-exports
NonRetryableParseError, so the import paths of the worker and the tests are unchanged."""
from __future__ import annotations


class NonRetryableParseError(RuntimeError):
    """Deterministic failure: the result is the same however many times it is retried (unsupported
    format, missing file, dimension mismatch, single file over the limit). Previously the only rule
    was the string "physical file not found"; every other deterministic failure needlessly went
    through all 5 rounds of exponential backoff (over 2.5 hours in total), recomputing the embedding
    each time."""
