"""
Article Loader Module
=====================
Fetches news article content from URLs using Requests + BeautifulSoup.

Responsibilities:
  - HTTP fetching with retry logic and timeout handling
  - HTML parsing and boilerplate removal
  - LangChain Document creation with rich metadata
  - Graceful error handling per URL
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, Comment
from langchain_core.documents import Document
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from backend.config import settings
from backend.utils import clean_text, is_valid_url, timed

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUEST_TIMEOUT = 15  # seconds
MAX_RETRIES = 3
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

# Tags whose entire subtree we discard (navigation, ads, footers, etc.)
BOILERPLATE_TAGS = {
    "script", "style", "noscript", "nav", "header", "footer",
    "aside", "form", "button", "iframe", "figure", "figcaption",
    "svg", "canvas", "meta", "link", "head",
}

# CSS selectors for likely article content containers (tried in order)
ARTICLE_SELECTORS = [
    "article",
    "[role='main']",
    "main",
    ".article-body",
    ".article-content",
    ".post-content",
    ".entry-content",
    ".story-body",
    ".content-body",
    "#article-body",
    "#content",
    "#main-content",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LoadResult:
    """Result of loading a single URL."""
    url: str
    success: bool
    document: Optional[Document] = None
    error: Optional[str] = None
    load_time_ms: float = 0.0


@dataclass
class BulkLoadResult:
    """Aggregated result of loading multiple URLs."""
    documents: list[Document] = field(default_factory=list)
    successful_urls: list[str] = field(default_factory=list)
    failed_urls: list[dict[str, str]] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.successful_urls) + len(self.failed_urls)

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return len(self.successful_urls) / self.total


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _remove_boilerplate(soup: BeautifulSoup) -> None:
    """Remove boilerplate tags and HTML comments in-place."""
    for tag in soup(BOILERPLATE_TAGS):
        tag.decompose()
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()


def _extract_article_text(soup: BeautifulSoup) -> str:
    """
    Try known article-content selectors in order; fall back to <body>.
    Returns the cleaned plain-text content.
    """
    for selector in ARTICLE_SELECTORS:
        container = soup.select_one(selector)
        if container:
            return container.get_text(separator="\n", strip=True)

    # Fallback: whole body
    body = soup.find("body")
    if body:
        return body.get_text(separator="\n", strip=True)

    return soup.get_text(separator="\n", strip=True)


def _extract_title(soup: BeautifulSoup) -> str:
    """Best-effort extraction of the article title."""
    # Try OG title first
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        return og_title["content"].strip()

    # Twitter card title
    tw_title = soup.find("meta", attrs={"name": "twitter:title"})
    if tw_title and tw_title.get("content"):
        return tw_title["content"].strip()

    # <title> tag
    title_tag = soup.find("title")
    if title_tag and title_tag.string:
        return title_tag.string.strip()

    # First <h1>
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)

    return "Untitled Article"


def _extract_description(soup: BeautifulSoup) -> str:
    """Extract meta description."""
    og_desc = soup.find("meta", property="og:description")
    if og_desc and og_desc.get("content"):
        return og_desc["content"].strip()
    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc and meta_desc.get("content"):
        return meta_desc["content"].strip()
    return ""


def _extract_domain(url: str) -> str:
    """Extract domain name from a URL."""
    try:
        return urlparse(url).netloc.removeprefix("www.")
    except Exception:
        return url


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

@retry(
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def _fetch_html(url: str) -> str:
    """Fetch raw HTML from *url* with retry on transient network errors."""
    response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.text


def load_article(url: str) -> LoadResult:
    """
    Load, parse, and clean a single news article URL.

    Args:
        url: The HTTP/HTTPS URL to fetch.

    Returns:
        A :class:`LoadResult` with either a populated Document or an error.
    """
    start = time.perf_counter()

    if not is_valid_url(url):
        return LoadResult(
            url=url,
            success=False,
            error="Invalid URL format. Must be a valid HTTP/HTTPS URL.",
            load_time_ms=0.0,
        )

    try:
        logger.info("Fetching article: {}", url)
        html = _fetch_html(url)

        soup = BeautifulSoup(html, "lxml")
        _remove_boilerplate(soup)

        title = _extract_title(soup)
        description = _extract_description(soup)
        raw_text = _extract_article_text(soup)
        cleaned = clean_text(raw_text)

        if len(cleaned) < 100:
            return LoadResult(
                url=url,
                success=False,
                error=(
                    "Extracted content is too short (< 100 chars). "
                    "The page may require JavaScript or is behind a paywall."
                ),
                load_time_ms=(time.perf_counter() - start) * 1000,
            )

        doc = Document(
            page_content=cleaned,
            metadata={
                "url": url,
                "title": title,
                "description": description,
                "domain": _extract_domain(url),
                "source": url,
                "char_count": len(cleaned),
            },
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.success(
            "Loaded '{}' from {} ({} chars, {:.0f}ms)",
            title,
            url,
            len(cleaned),
            elapsed_ms,
        )
        return LoadResult(url=url, success=True, document=doc, load_time_ms=elapsed_ms)

    except requests.HTTPError as exc:
        err = f"HTTP {exc.response.status_code}: {exc.response.reason}"
        logger.warning("HTTP error for {}: {}", url, err)
        return LoadResult(
            url=url,
            success=False,
            error=err,
            load_time_ms=(time.perf_counter() - start) * 1000,
        )
    except requests.Timeout:
        err = f"Request timed out after {REQUEST_TIMEOUT}s"
        logger.warning("{} for {}", err, url)
        return LoadResult(
            url=url,
            success=False,
            error=err,
            load_time_ms=(time.perf_counter() - start) * 1000,
        )
    except Exception as exc:
        err = f"Unexpected error: {exc}"
        logger.exception("Failed to load {}: {}", url, exc)
        return LoadResult(
            url=url,
            success=False,
            error=err,
            load_time_ms=(time.perf_counter() - start) * 1000,
        )


@timed
def load_articles(urls: list[str]) -> BulkLoadResult:
    """
    Load multiple news articles from a list of URLs.

    Enforces the configured MAX_URLS limit. Each URL is processed
    independently so one failure does not block the rest.

    Args:
        urls: List of HTTP/HTTPS URLs (duplicates are deduplicated).

    Returns:
        A :class:`BulkLoadResult` with all documents and per-URL status.
    """
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_urls: list[str] = []
    for url in urls:
        url = url.strip()
        if url and url not in seen:
            seen.add(url)
            unique_urls.append(url)

    # Enforce limit
    if len(unique_urls) > settings.max_urls:
        logger.warning(
            "Received {} URLs; truncating to configured maximum of {}.",
            len(unique_urls),
            settings.max_urls,
        )
        unique_urls = unique_urls[: settings.max_urls]

    result = BulkLoadResult()

    for url in unique_urls:
        load_result = load_article(url)
        if load_result.success and load_result.document:
            result.documents.append(load_result.document)
            result.successful_urls.append(url)
        else:
            result.failed_urls.append({"url": url, "error": load_result.error or "Unknown error"})

    logger.info(
        "Bulk load complete: {}/{} URLs succeeded.",
        len(result.successful_urls),
        len(unique_urls),
    )
    return result
