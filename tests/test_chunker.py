"""
Unit Tests: backend.chunker
============================
Tests for document chunking behaviour.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from backend.chunker import chunk_documents, get_chunk_stats


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_documents() -> list[Document]:
    """Realistic news article documents for testing."""
    return [
        Document(
            page_content=(
                "Breaking News: Tech Giant Reports Record Profits\n\n"
                "In a stunning announcement, the company reported quarterly earnings "
                "that exceeded analyst expectations by a wide margin. The CEO attributed "
                "the success to aggressive expansion into new markets and strong consumer "
                "demand for their flagship products.\n\n"
                "Industry analysts say this could signal a broader recovery across the "
                "technology sector. Several competitors are expected to report similarly "
                "positive results in the coming weeks.\n\n"
                "Shares rose more than 8% in after-hours trading, pushing the company's "
                "market capitalisation past $2 trillion for the first time."
            ),
            metadata={
                "url": "https://example.com/tech-earnings",
                "title": "Tech Giant Reports Record Profits",
                "domain": "example.com",
                "source": "https://example.com/tech-earnings",
            },
        ),
        Document(
            page_content=(
                "Climate Summit Reaches Historic Agreement\n\n"
                "World leaders gathered in Geneva have signed a landmark accord "
                "committing 150 nations to net-zero emissions by 2050. The deal, "
                "brokered after three weeks of intensive negotiations, was hailed as "
                "a 'turning point for humanity' by the UN Secretary-General.\n\n"
                "Key provisions include a $500 billion annual fund for developing "
                "nations to transition to clean energy, and binding emission reduction "
                "targets reviewed every five years."
            ),
            metadata={
                "url": "https://example.com/climate-summit",
                "title": "Climate Summit Reaches Historic Agreement",
                "domain": "example.com",
                "source": "https://example.com/climate-summit",
            },
        ),
    ]


# ---------------------------------------------------------------------------
# chunk_documents
# ---------------------------------------------------------------------------

class TestChunkDocuments:
    def test_returns_chunks_from_documents(self, sample_documents) -> None:
        chunks = chunk_documents(sample_documents, chunk_size=300, chunk_overlap=50)
        assert len(chunks) > 0

    def test_empty_documents_returns_empty_list(self) -> None:
        result = chunk_documents([])
        assert result == []

    def test_chunk_metadata_includes_parent_info(self, sample_documents) -> None:
        chunks = chunk_documents(sample_documents[:1], chunk_size=200, chunk_overlap=20)
        for chunk in chunks:
            assert "chunk_index" in chunk.metadata
            assert "chunk_total" in chunk.metadata
            assert "parent_url" in chunk.metadata
            assert chunk.metadata["parent_url"] == sample_documents[0].metadata["url"]

    def test_chunk_indices_are_sequential(self, sample_documents) -> None:
        chunks = chunk_documents(sample_documents[:1], chunk_size=200, chunk_overlap=20)
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(indices)))

    def test_chunk_content_within_size_limit(self, sample_documents) -> None:
        chunk_size = 250
        chunks = chunk_documents(sample_documents, chunk_size=chunk_size, chunk_overlap=25)
        for chunk in chunks:
            # Allow some tolerance due to splitter behaviour
            assert len(chunk.page_content) <= chunk_size * 1.5

    def test_inherits_parent_metadata(self, sample_documents) -> None:
        chunks = chunk_documents(sample_documents[:1], chunk_size=300, chunk_overlap=50)
        for chunk in chunks:
            assert chunk.metadata.get("url") == sample_documents[0].metadata["url"]
            assert chunk.metadata.get("title") == sample_documents[0].metadata["title"]

    def test_multiple_documents_produces_more_chunks(self, sample_documents) -> None:
        single_chunks = chunk_documents(sample_documents[:1], chunk_size=300, chunk_overlap=50)
        multi_chunks = chunk_documents(sample_documents, chunk_size=300, chunk_overlap=50)
        assert len(multi_chunks) >= len(single_chunks)


# ---------------------------------------------------------------------------
# get_chunk_stats
# ---------------------------------------------------------------------------

class TestGetChunkStats:
    def test_empty_list_returns_zeros(self) -> None:
        stats = get_chunk_stats([])
        assert stats["count"] == 0
        assert stats["total_chars"] == 0

    def test_stats_for_known_chunks(self) -> None:
        chunks = [
            Document(page_content="a" * 100),
            Document(page_content="b" * 200),
            Document(page_content="c" * 300),
        ]
        stats = get_chunk_stats(chunks)
        assert stats["count"] == 3
        assert stats["total_chars"] == 600
        assert stats["min_chars"] == 100
        assert stats["max_chars"] == 300
        assert stats["avg_chars"] == 200.0
