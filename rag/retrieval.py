"""
RAG Retrieval Layer
-------------------
Provides a high-level ``retrieve_context`` helper that combines telemetry
ingestion with the vector store so agents can pull relevant log context for
any free-text query.
"""

from __future__ import annotations

import logging
import os

from pipelines.telemetry_ingestion import ingest_logs
from rag.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

_TOP_K = int(os.getenv("RAG_TOP_K", "5"))

# Module-level singleton so the index is built once per process.
_manager: VectorStoreManager | None = None


def _get_manager() -> VectorStoreManager:
    """Return (and lazily initialise) the shared VectorStoreManager."""
    global _manager
    if _manager is None:
        _manager = VectorStoreManager()
        chunks = ingest_logs()
        _manager.build_from_texts(chunks)
        logger.info("RAG retriever initialised with %d chunks.", len(chunks))
    return _manager


def retrieve_context(query: str, k: int | None = None) -> list[str]:
    """Retrieve the most relevant log snippets for *query*.

    Args:
        query: The search / question string (e.g. "packet drop on R17").
        k:     Number of results to return.  Defaults to ``RAG_TOP_K`` env var.

    Returns:
        Ordered list of relevant log text chunks.
    """
    top_k = k if k is not None else _TOP_K
    manager = _get_manager()
    results = manager.similarity_search(query, k=top_k)
    logger.debug("Retrieved %d context chunks for query=%r", len(results), query)
    return results


def reset_retriever() -> None:
    """Reset the module-level retriever (useful for testing)."""
    global _manager
    _manager = None
