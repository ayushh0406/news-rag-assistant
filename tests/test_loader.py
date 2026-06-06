"""
Unit Tests: backend.loader
===========================
Tests article loading, URL validation, and error handling.
Mocks HTTP calls to avoid network dependencies.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from backend.loader import (
    BulkLoadResult,
    LoadResult,
    _extract_domain,
    _extract_title,
    load_article,
    load_articles,
)


# ---------------------------------------------------------------------------
# _extract_domain
# ---------------------------------------------------------------------------

class TestExtractDomain:
    def test_strips_www(self) -> None:
        assert _extract_domain("https://www.bbc.com/news") == "bbc.com"

    def test_keeps_subdomain(self) -> None:
        assert _extract_domain("https://techcrunch.com/news") == "techcrunch.com"

    def test_handles_invalid_url_gracefully(self) -> None:
        result = _extract_domain("not-a-url")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# load_article
# ---------------------------------------------------------------------------

SAMPLE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Test Article Title</title>
    <meta property="og:title" content="OG Test Article">
    <meta name="description" content="A test article description.">
</head>
<body>
    <nav>Navigation Menu</nav>
    <article>
        <h1>Breaking: Scientists Discover New Planet</h1>
        <p>Astronomers announced the discovery of a new exoplanet orbiting a distant star.
        The planet, dubbed Kepler-999b, is located 340 light-years from Earth and shows
        signs of having liquid water on its surface. This makes it a prime candidate for
        further study in the search for extraterrestrial life.</p>
        <p>The research team used data from the James Webb Space Telescope combined with
        ground-based observations to confirm the discovery. Results were published in
        the peer-reviewed journal Nature Astronomy.</p>
    </article>
    <footer>Footer content</footer>
    <script>var x = 1;</script>
</body>
</html>
"""


class TestLoadArticle:
    def test_invalid_url_returns_failure(self) -> None:
        result = load_article("not-a-valid-url")
        assert result.success is False
        assert result.document is None
        assert "Invalid URL" in result.error

    def test_successful_load_produces_document(self) -> None:
        mock_response = MagicMock()
        mock_response.text = SAMPLE_HTML
        mock_response.raise_for_status = MagicMock()

        with patch("backend.loader.requests.get", return_value=mock_response):
            result = load_article("https://example.com/news/planet-discovery")

        assert result.success is True
        assert result.document is not None
        assert "Scientists" in result.document.page_content or "planet" in result.document.page_content.lower()

    def test_document_has_required_metadata(self) -> None:
        mock_response = MagicMock()
        mock_response.text = SAMPLE_HTML
        mock_response.raise_for_status = MagicMock()

        with patch("backend.loader.requests.get", return_value=mock_response):
            result = load_article("https://example.com/article")

        assert result.document is not None
        meta = result.document.metadata
        assert "url" in meta
        assert "title" in meta
        assert "domain" in meta
        assert meta["url"] == "https://example.com/article"
        assert meta["domain"] == "example.com"

    def test_boilerplate_is_removed(self) -> None:
        mock_response = MagicMock()
        mock_response.text = SAMPLE_HTML
        mock_response.raise_for_status = MagicMock()

        with patch("backend.loader.requests.get", return_value=mock_response):
            result = load_article("https://example.com/article")

        assert result.document is not None
        # Nav and footer content should be removed
        assert "Navigation Menu" not in result.document.page_content
        assert "Footer content" not in result.document.page_content

    def test_http_error_returns_failure(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.reason = "Not Found"
        http_error = requests.HTTPError(response=mock_response)
        mock_response.raise_for_status.side_effect = http_error

        with patch("backend.loader.requests.get", return_value=mock_response):
            result = load_article("https://example.com/missing")

        assert result.success is False
        assert "404" in result.error

    def test_timeout_returns_failure(self) -> None:
        with patch("backend.loader.requests.get", side_effect=requests.Timeout()):
            result = load_article("https://example.com/slow")

        assert result.success is False
        assert "timed out" in result.error.lower()


# ---------------------------------------------------------------------------
# load_articles
# ---------------------------------------------------------------------------

class TestLoadArticles:
    def test_returns_bulk_result(self) -> None:
        with patch("backend.loader.load_article") as mock_load:
            mock_doc = MagicMock()
            mock_load.return_value = LoadResult(
                url="https://example.com/article",
                success=True,
                document=mock_doc,
            )
            result = load_articles(["https://example.com/article"])

        assert isinstance(result, BulkLoadResult)
        assert len(result.documents) == 1
        assert len(result.successful_urls) == 1

    def test_deduplicates_urls(self) -> None:
        with patch("backend.loader.load_article") as mock_load:
            mock_doc = MagicMock()
            mock_load.return_value = LoadResult(
                url="https://example.com/article",
                success=True,
                document=mock_doc,
            )
            result = load_articles([
                "https://example.com/article",
                "https://example.com/article",
            ])

        assert mock_load.call_count == 1

    def test_failed_urls_tracked(self) -> None:
        with patch("backend.loader.load_article") as mock_load:
            mock_load.return_value = LoadResult(
                url="https://example.com/broken",
                success=False,
                error="HTTP 404: Not Found",
            )
            result = load_articles(["https://example.com/broken"])

        assert len(result.failed_urls) == 1
        assert len(result.documents) == 0
        assert result.success_rate == 0.0
