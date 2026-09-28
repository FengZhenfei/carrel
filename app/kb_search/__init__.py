"""Retrieval service (recall layer): multi-channel recall over vectors / BM25 / graph -> RRF fusion ->
cross-encoder rerank -> evidence assembly. Returns evidence only and never generates answers; all
policy lives on the DGX, and the client hands over the question verbatim (2026-09-09 to-do list Q01)."""
