"""
Document Chunker Module
=======================
Splits LangChain Documents into smaller, overlapping chunks suitable for
embedding and semantic search.

Uses LangChain's RecursiveCharacterTextSplitter, which tries to split on
paragraph boundaries, then sentences, then words — preserving semantic
coherence as much as possible.

Each chunk inherits the parent document's metadata and receives additional
chunk-level metadata (index, total count, character offsets).
"""

from __future__ import annotations

from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from backend.config import settings
from backend.utils import timed


# ---------------------------------------------------------------------------
# Splitter factory
# ---------------------------------------------------------------------------

def _build_splitter(
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> RecursiveCharacterTextSplitter:
    """
    Create a :class:`RecursiveCharacterTextSplitter`.

    Falls back to configured defaults if explicit sizes are not provided.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
        is_separator_regex=False,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@timed
def chunk_documents(
    documents: list[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> list[Document]:
    """
    Split a list of Documents into smaller chunks.

    Each produced chunk carries the original document's metadata plus:
      - ``chunk_index``:   zero-based position within the parent document
      - ``chunk_total``:   total number of chunks from that document
      - ``parent_url``:    URL of the source article

    Args:
        documents:     LangChain Documents to split.
        chunk_size:    Override for chunk character size (uses config default).
        chunk_overlap: Override for chunk overlap (uses config default).

    Returns:
        Flat list of chunk Documents, ordered by source document then position.
    """
    if not documents:
        logger.warning("chunk_documents called with empty document list.")
        return []

    splitter = _build_splitter(chunk_size, chunk_overlap)
    all_chunks: list[Document] = []

    for doc_idx, doc in enumerate(documents):
        raw_chunks = splitter.split_documents([doc])

        # Enrich metadata
        total = len(raw_chunks)
        for chunk_idx, chunk in enumerate(raw_chunks):
            chunk.metadata = {
                **chunk.metadata,
                "chunk_index": chunk_idx,
                "chunk_total": total,
                "parent_url": doc.metadata.get("url", ""),
                "doc_index": doc_idx,
            }

        logger.debug(
            "Document '{}' → {} chunks",
            doc.metadata.get("title", f"doc_{doc_idx}"),
            total,
        )
        all_chunks.extend(raw_chunks)

    logger.info(
        "Chunking complete: {} documents → {} chunks (size={}, overlap={})",
        len(documents),
        len(all_chunks),
        chunk_size or settings.chunk_size,
        chunk_overlap or settings.chunk_overlap,
    )
    return all_chunks


def get_chunk_stats(chunks: list[Document]) -> dict[str, float | int]:
    """
    Return basic statistics about a list of chunk Documents.

    Useful for debugging and logging.
    """
    if not chunks:
        return {"count": 0, "total_chars": 0, "avg_chars": 0, "min_chars": 0, "max_chars": 0}

    lengths = [len(c.page_content) for c in chunks]
    return {
        "count": len(chunks),
        "total_chars": sum(lengths),
        "avg_chars": round(sum(lengths) / len(lengths), 1),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
    }
