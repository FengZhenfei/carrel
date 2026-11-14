"""Storage boundary for the pipeline, retrieval service and console.

Both backends provide the same domain operations; unknown modes never fall back
to Standard stores. Deployment mode is not a per-library UI setting.
"""
from __future__ import annotations

from typing import Any

from .contracts import Storage


def get_storage(settings: Any, *, vector_client: Any = None, query_timeout: float | None = None,
                previous: Storage | None = None) -> Storage:
    from .publication import guard
    backend = getattr(settings, "storage_backend", "standard")
    if backend == "light":
        from .light import LightStorage
        return guard(LightStorage(settings))
    if backend != "standard":
        raise ValueError(f"Storage backend is not implemented: {backend!r}")
    from .standard import StandardStorage
    store = StandardStorage(settings, vector_client=vector_client, query_timeout=query_timeout)
    # Reuse the connection without mutating the settings held by in-flight requests.
    if isinstance(previous, StandardStorage) and store._client_key == previous._client_key:
        store._client = previous._client
    return guard(store)
