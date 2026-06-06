"""
Unit Tests: backend.utils
=========================
Tests for URL validation, text cleaning, and helper functions.
"""

from __future__ import annotations

import pytest

from backend.utils import (
    clean_text,
    format_sources,
    is_valid_url,
    truncate_text,
    validate_urls,
)


# ---------------------------------------------------------------------------
# is_valid_url
# ---------------------------------------------------------------------------

class TestIsValidUrl:
    @pytest.mark.parametrize(
        "url",
        [
            "https://www.bbc.com/news/article",
            "http://example.com",
            "https://techcrunch.com/2024/01/01/some-article/",
            "https://news.ycombinator.com/item?id=12345",
        ],
    )
    def test_valid_http_https_urls(self, url: str) -> None:
        assert is_valid_url(url) is True

    @pytest.mark.parametrize(
        "url",
        [
            "",
            "   ",
            "not-a-url",
            "ftp://example.com/file.txt",
            "javascript:void(0)",
            "//missing-scheme.com",
            "http://",
            None,
        ],
    )
    def test_invalid_urls(self, url) -> None:
        assert is_valid_url(url) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# validate_urls
# ---------------------------------------------------------------------------

class TestValidateUrls:
    def test_separates_valid_and_invalid(self) -> None:
        urls = [
            "https://valid.com/article",
            "not-valid",
            "https://also-valid.org/news",
            "",
        ]
        valid, invalid = validate_urls(urls)
        assert valid == ["https://valid.com/article", "https://also-valid.org/news"]
        assert invalid == ["not-valid"]

    def test_empty_list(self) -> None:
        valid, invalid = validate_urls([])
        assert valid == []
        assert invalid == []

    def test_strips_whitespace(self) -> None:
        urls = ["  https://example.com/news  "]
        valid, _ = validate_urls(urls)
        assert valid == ["https://example.com/news"]


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_removes_extra_blank_lines(self) -> None:
        dirty = "Line 1\n\n\n\nLine 2\n\n\nLine 3"
        cleaned = clean_text(dirty)
        assert "\n\n\n" not in cleaned
        assert "Line 1" in cleaned
        assert "Line 3" in cleaned

    def test_strips_leading_trailing_whitespace(self) -> None:
        assert clean_text("  hello world  ") == "hello world"

    def test_empty_string_returns_empty(self) -> None:
        assert clean_text("") == ""

    def test_normalises_line_endings(self) -> None:
        text = "Windows\r\nline\rending"
        result = clean_text(text)
        assert "\r" not in result

    def test_preserves_content(self) -> None:
        text = "Breaking News: Market rallies on strong earnings report."
        assert clean_text(text) == text


# ---------------------------------------------------------------------------
# truncate_text
# ---------------------------------------------------------------------------

class TestTruncateText:
    def test_short_text_unchanged(self) -> None:
        text = "Short text"
        assert truncate_text(text, max_chars=100) == text

    def test_long_text_is_truncated(self) -> None:
        text = "a" * 600
        result = truncate_text(text, max_chars=500)
        assert len(result) <= 502  # 500 + ellipsis char
        assert result.endswith("…")

    def test_exact_length_unchanged(self) -> None:
        text = "a" * 100
        assert truncate_text(text, max_chars=100) == text


# ---------------------------------------------------------------------------
# format_sources
# ---------------------------------------------------------------------------

class TestFormatSources:
    def test_empty_sources_returns_empty_string(self) -> None:
        assert format_sources([]) == ""

    def test_formats_sources_with_title(self) -> None:
        sources = [
            {"url": "https://example.com/article", "title": "Example Article"},
            {"url": "https://other.com/news", "title": "Other News"},
        ]
        result = format_sources(sources)
        assert "**Sources:**" in result
        assert "Example Article" in result
        assert "Other News" in result

    def test_deduplicates_urls(self) -> None:
        sources = [
            {"url": "https://example.com/article", "title": "Title"},
            {"url": "https://example.com/article", "title": "Title Duplicate"},
        ]
        result = format_sources(sources)
        assert result.count("example.com") == 1

    def test_falls_back_to_url_when_no_title(self) -> None:
        sources = [{"url": "https://example.com/article"}]
        result = format_sources(sources)
        assert "https://example.com/article" in result
