"""Exception types of the parsing layer. Kept in the parsers package rather than parse_job: the router
(parsers.router) also needs to raise a "deterministic failure", and parse_job imports the router in
turn, so placing it there would create a circular import. parse_job still re-exports
NonRetryableParseError, so the import paths of the worker and the tests are unchanged."""
from __future__ import annotations


class NonRetryableParseError(RuntimeError):
    """Deterministic failure: the result is the same however many times it is retried (unsupported
    format, missing file, single file over the limit). Previously the only rule
    was the string "physical file not found"; every other deterministic failure needlessly went
    through all 5 rounds of exponential backoff (over 2.5 hours in total), recomputing the embedding
    each time."""


class ConfigParseError(NonRetryableParseError):
    """Configuration/auth/model error (missing/unregistered VLM, wrong protocol, unreadable KB config).
    Skips immediate exponential backoff like NonRetryableParseError, but is recoverable via cooldown
    auto-heal or immediately when the matching model/KB settings are updated in the console."""

    def __init__(self, message: str, *, scope: str | None = None) -> None:
        super().__init__(message)
        self.config_scope = scope


class JobCancelled(RuntimeError):
    """The user cancelled the job mid-parse, or closed / deleted this knowledge base. Not a failure: no retry
    is counted, no failure is recorded, the worker marks the job cancelled directly. The checkpoint is stage()
    in parse_job -- every phase boundary and every per-image VLM callback passes through it, so the worst-case
    wait is "the current step", not "the current knowledge base". Cancellation only takes effect before
    writing to the vector store begins: once points have been written the job counts as committed and goes
    on to finish the state database. Defined here because the image-description layer (vision.vlm) has to
    recognize it for a cancellation raised in its callback to pass through."""


class JobPaused(JobCancelled):
    """Yield at a safe parse checkpoint, retaining this job and its completed caches."""


# The verdict on a failed parse job. The worker gives it from the exception (pipeline.worker._failure_verdict)
# and stores it in the job payload under FAILURE_VERDICT_KEY when the job fails for good; the scan uses it to
# decide whether the file is queued again (pipeline.scheduler._failed_parse_blocks_auto_requeue) instead of
# looking for words in the error text. The two used to judge separately: failures the worker knew a retry
# could not change (MinerU 4xx, an embedding dimension mismatch) were not recognised by the scan from the text
# and were queued again after every cooldown; and a file whose path happened to contain a marker word was
# never queued again after one transient failure.
FAILURE_VERDICT_KEY = "failure_verdict"
FAILURE_SCOPE_KEY = "failure_scope"
FAILURE_CONFIG = "config"                  # auth/model/KB config error: no immediate backoff; auto-heals after cooldown or config save
FAILURE_DETERMINISTIC = "deterministic"    # the same input fails the same way: queued again only when the parser / router changes
FAILURE_FILE_MISSING = "file_missing"      # the file was not on disk: queued as soon as a scan sees it again
FAILURE_RETRYABLE = "retryable"            # everything else: once the retries are used up, queued again after the cooldown

# Subsystem scopes for FAILURE_CONFIG jobs so saving an unrelated endpoint (e.g. a search-only reranker)
# never wakes jobs waiting on a different service's credentials or config.
FAILURE_SCOPE_MINERU = "mineru"
FAILURE_SCOPE_EMBEDDING = "embedding"
FAILURE_SCOPE_VISUAL_EMBEDDING = "visual_embedding"
FAILURE_SCOPE_VLM = "vlm"
FAILURE_SCOPE_KB_CONFIG = "kb_config"


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
