"""
Vector Store Abstraction
------------------------
Wraps FAISS (default) with a clean interface.  A Pinecone back-end can be
swapped in by setting VECTOR_STORE_BACKEND=pinecone in the environment.

Usage
-----
    from rag.vector_store import VectorStoreManager

    manager = VectorStoreManager()
    manager.build_from_texts(["log line 1", "log line 2"])
    results = manager.similarity_search("packet drop", k=3)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

_BACKEND = os.getenv("VECTOR_STORE_BACKEND", "faiss").lower()
_FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index")


class VectorStoreManager:
    """Unified interface for vector store operations.

    Supports FAISS (default) with an optional Pinecone backend.
    The backend is selected via the ``VECTOR_STORE_BACKEND`` env var.
    """

    def __init__(self) -> None:
        self._embeddings = OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        )
        self._store: Any = None
        self._backend = _BACKEND
        logger.info("VectorStoreManager initialised with backend=%s", self._backend)

    # ------------------------------------------------------------------
    # Build / load
    # ------------------------------------------------------------------

    def build_from_texts(self, texts: list[str], metadatas: list[dict] | None = None) -> None:
        """Embed *texts* and build an in-memory vector store.

        Args:
            texts:     List of text chunks to embed.
            metadatas: Optional list of metadata dicts aligned with *texts*.
        """
        if not texts:
            raise ValueError("Cannot build vector store from empty text list.")

        if self._backend == "faiss":
            self._store = FAISS.from_texts(texts, self._embeddings, metadatas=metadatas)
            logger.info("FAISS index built with %d vectors.", len(texts))
        elif self._backend == "pinecone":
            self._store = self._build_pinecone(texts, metadatas)
        else:
            raise ValueError(f"Unsupported vector store backend: {self._backend!r}")

    def save(self, path: str | Path | None = None) -> None:
        """Persist the FAISS index to disk (no-op for Pinecone).

        Args:
            path: Directory to save the FAISS index.  Defaults to
                  the ``FAISS_INDEX_PATH`` env var.
        """
        if self._backend != "faiss":
            logger.info("save() is a no-op for backend=%s", self._backend)
            return
        save_path = str(path or _FAISS_INDEX_PATH)
        self._store.save_local(save_path)
        logger.info("FAISS index saved to %s", save_path)

    def load(self, path: str | Path | None = None) -> None:
        """Load a previously saved FAISS index from disk.

        Args:
            path: Directory containing the saved FAISS index.
        """
        if self._backend != "faiss":
            raise RuntimeError("load() is only supported for the FAISS backend.")
        load_path = str(path or _FAISS_INDEX_PATH)
        self._store = FAISS.load_local(
            load_path,
            self._embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("FAISS index loaded from %s", load_path)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def similarity_search(self, query: str, k: int = 5) -> list[str]:
        """Return the top-*k* most relevant text chunks for *query*.

        Args:
            query: Natural-language search string.
            k:     Number of results to return.

        Returns:
            List of text chunks ordered by descending relevance.
        """
        if self._store is None:
            raise RuntimeError("Vector store is not initialised. Call build_from_texts() first.")

        docs = self._store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

    # ------------------------------------------------------------------
    # Pinecone helper (optional back-end)
    # ------------------------------------------------------------------

    def _build_pinecone(self, texts: list[str], metadatas: list[dict] | None) -> Any:
        """Build a Pinecone vector store (requires pinecone-client installed).

        Reads ``PINECONE_API_KEY`` and ``PINECONE_INDEX_NAME`` from env.
        """
        try:
            from langchain_community.vectorstores import Pinecone as LCPinecone  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Install pinecone-client to use the Pinecone backend: "
                "pip install pinecone-client"
            ) from exc

        api_key = os.environ["PINECONE_API_KEY"]
        index_name = os.environ["PINECONE_INDEX_NAME"]

        try:
            from pinecone import Pinecone  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "Install pinecone-client>=3.0 to use the Pinecone backend."
            ) from exc

        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)

        store = LCPinecone.from_texts(
            texts,
            self._embeddings,
            index_name=index_name,
            metadatas=metadatas,
        )
        logger.info("Pinecone index '%s' populated with %d vectors.", index_name, len(texts))
        return store
