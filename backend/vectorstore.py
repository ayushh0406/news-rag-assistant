"""
Vector Store Module
===================
Manages a persistent ChromaDB collection for storing and retrieving
news article embeddings.

Responsibilities:
  - Collection lifecycle (create / open / reset)
  - Adding documents with metadata
  - Semantic similarity search
  - Source-filtered retrieval
  - Statistics reporting
"""

from __future__ import annotations

import time
from typing import Any, Optional

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from loguru import logger

from backend.config import settings
from backend.embeddings import get_embeddings
from backend.utils import timed


# ---------------------------------------------------------------------------
# ChromaDB client factory
# ---------------------------------------------------------------------------

def _get_chroma_client() -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client rooted at the configured path."""
    persist_path = settings.chroma_persist_path
    persist_path.mkdir(parents=True, exist_ok=True)
    logger.debug("ChromaDB persist path: {}", persist_path)
    return chromadb.PersistentClient(path=str(persist_path))


# ---------------------------------------------------------------------------
# VectorStore wrapper
# ---------------------------------------------------------------------------

class NewsVectorStore:
    """
    Thin wrapper around LangChain's :class:`Chroma` vector store.

    Provides a domain-specific interface for storing news article chunks
    and performing semantic search with optional source filtering.
    """

    def __init__(self) -> None:
        self._embeddings = get_embeddings()
        self._client = _get_chroma_client()
        self._store: Optional[Chroma] = None
        self._collection_name = settings.chroma_collection_name
        logger.info("NewsVectorStore initialised | collection='{}'", self._collection_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_store(self) -> Chroma:
        """Lazily initialise (or reuse) the Chroma store."""
        if self._store is None:
            self._store = Chroma(
                client=self._client,
                collection_name=self._collection_name,
                embedding_function=self._embeddings,
            )
            count = self._store._collection.count()
            logger.info(
                "Chroma collection '{}' opened ({} existing vectors).",
                self._collection_name,
                count,
            )
        return self._store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @timed
    def add_documents(self, chunks: list[Document]) -> int:
        """
        Embed and persist a list of Document chunks.

        Args:
            chunks: Pre-split LangChain Documents.

        Returns:
            Number of chunks successfully added.

        Raises:
            RuntimeError: On ChromaDB or embedding failures.
        """
        if not chunks:
            logger.warning("add_documents called with empty list — nothing to store.")
            return 0

        store = self._get_or_create_store()

        try:
            store.add_documents(chunks)
            logger.success("Added {} chunks to ChromaDB.", len(chunks))
            return len(chunks)
        except Exception as exc:
            logger.exception("Failed to add documents to ChromaDB: {}", exc)
            raise RuntimeError(f"Vector store insertion failed: {exc}") from exc


    @timed
    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        filter_urls: Optional[list[str]] = None,
    ) -> list[Document]:
        """
        Retrieve the *k* most semantically similar chunks for *query*.

        Args:
            query:       The user's question / search string.
            k:           Number of results to return (defaults to TOP_K_RESULTS).
            filter_urls: Optional list of source URLs to restrict results to.

        Returns:
            List of Document chunks ranked by relevance (most relevant first).
        """
        k = k or settings.top_k_results
        store = self._get_or_create_store()

        chroma_filter: Optional[dict[str, Any]] = None
        if filter_urls:
            if len(filter_urls) == 1:
                chroma_filter = {"url": {"$eq": filter_urls[0]}}
            else:
                chroma_filter = {"url": {"$in": filter_urls}}

        try:
            results = store.similarity_search(query, k=k, filter=chroma_filter)
            logger.debug(
                "Similarity search for '{}' returned {} results.",
                query[:80],
                len(results),
            )
            return results
        except Exception as exc:
            logger.exception("Similarity search failed: {}", exc)
            return []

    @timed
    def similarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> list[tuple[Document, float]]:
        """
        Like :meth:`similarity_search` but also returns relevance scores.

        Returns:
            List of (Document, score) tuples (lower score = more similar for
            cosine-distance-based Chroma).
        """
        k = k or settings.top_k_results
        store = self._get_or_create_store()

        try:
            return store.similarity_search_with_score(query, k=k)
        except Exception as exc:
            logger.exception("Scored similarity search failed: {}", exc)
            return []

    def as_retriever(self, k: Optional[int] = None) -> Any:
        """Return a LangChain-compatible retriever interface."""
        store = self._get_or_create_store()
        return store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k or settings.top_k_results},
        )

    def reset_collection(self) -> None:
        """
        Delete and recreate the ChromaDB collection.

        Useful for a "process new URLs" flow where the user wants to
        start fresh rather than accumulate documents across sessions.
        """
        try:
            self._client.delete_collection(self._collection_name)
            logger.info("Deleted ChromaDB collection '{}'.", self._collection_name)
        except Exception:
            pass  # Collection may not exist yet

        self._store = None
        self._get_or_create_store()  # Recreate
        logger.success("ChromaDB collection '{}' reset successfully.", self._collection_name)

    def get_stats(self) -> dict[str, Any]:
        """Return basic statistics about the current collection."""
        try:
            store = self._get_or_create_store()
            count = store._collection.count()
            return {
                "collection_name": self._collection_name,
                "total_vectors": count,
                "persist_path": str(settings.chroma_persist_path),
            }
        except Exception as exc:
            logger.error("Failed to retrieve vector store stats: {}", exc)
            return {"error": str(exc)}

    def is_empty(self) -> bool:
        """Return True if the collection contains no documents."""
        stats = self.get_stats()
        return stats.get("total_vectors", 0) == 0


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_vector_store_instance: Optional[NewsVectorStore] = None


def get_vector_store() -> NewsVectorStore:
    """Return the application-level :class:`NewsVectorStore` singleton."""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = NewsVectorStore()
    return _vector_store_instance
