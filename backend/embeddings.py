"""
Embeddings Module
=================
Provides a thin, cached wrapper using sentence-transformers directly.

Uses the ``all-MiniLM-L6-v2`` model which:
  - Runs fully locally (no API calls, no rate limits, no quota)
  - Produces 384-dimensional embeddings
  - Is fast and high quality for semantic search / RAG

This wraps sentence-transformers in a LangChain-compatible interface
without importing langchain-huggingface (which would force a langchain-core
version conflict).
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from langchain_core.embeddings import Embeddings
from loguru import logger
from sentence_transformers import SentenceTransformer


# The model to use — small, fast, and great for semantic search
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class LocalEmbeddings(Embeddings):
    """
    LangChain-compatible embeddings using a local sentence-transformers model.
    No API calls, no rate limits, no quota.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        logger.info("Loading local embedding model: {}", model_name)
        self._model = SentenceTransformer(model_name)
        logger.success("Local embedding model loaded successfully.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        embedding = self._model.encode([text], normalize_embeddings=True)
        return embedding[0].tolist()


@lru_cache(maxsize=1)
def get_embeddings() -> LocalEmbeddings:
    """
    Return a cached :class:`LocalEmbeddings` instance.

    The model is downloaded once on first use (~90MB) and cached locally.
    Subsequent calls return the same in-memory instance.

    Returns:
        Configured ``LocalEmbeddings`` ready to use with LangChain.
    """
    return LocalEmbeddings(EMBEDDING_MODEL)


def get_query_embeddings() -> LocalEmbeddings:
    """
    Return embeddings for query encoding.

    With sentence-transformers, the same model handles both
    document and query encoding, so this is an alias for get_embeddings().
    """
    return get_embeddings()
