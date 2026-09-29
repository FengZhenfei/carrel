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


class JobCancelled(RuntimeError):
    """The user cancelled the job mid-parse, or closed / deleted this knowledge base. Not a failure: no retry
    is counted, no failure is recorded, the worker marks the job cancelled directly. The checkpoint is stage()
    in parse_job -- every phase boundary and every per-image VLM callback passes through it, so the worst-case
    wait is "the current step", not "the current knowledge base". Cancellation only takes effect before
    writing to the vector store begins: once points have been written the job counts as committed and goes
    on to finish the state database. Defined here because the image-description layer (vision.vlm) has to
    recognize it for a cancellation raised in its callback to pass through."""


def service_unreachable(exc: BaseException | None) -> bool:
    """Whether this failure means "the service cannot be reached": it is not up yet, or it is restarting. Walks
    the exception chain looking for connection-level errors (connection refused, connection reset by the peer,
    connect timeout); read timeouts and HTTP status codes do not count -- then the service is there and only
    this request did not succeed."""
    import httpx
    import requests

    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, (ConnectionError, requests.ConnectionError, httpx.NetworkError,
                            httpx.ConnectTimeout, httpx.RemoteProtocolError)):
            return True
        exc = exc.__cause__ or exc.__context__
    return False
