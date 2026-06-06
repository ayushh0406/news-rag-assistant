"""
Utilities Module
================
Shared helpers used across the backend:
  - Logging setup (Loguru)
  - URL validation
  - Text cleaning
  - Timing decorators
  - Retry helpers
"""

from __future__ import annotations

import re
import sys
import time
import unicodedata
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar
from urllib.parse import urlparse

from loguru import logger

from backend.config import settings

# ---------------------------------------------------------------------------
# Type helpers
# ---------------------------------------------------------------------------
F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    """
    Configure Loguru logger.

    Outputs to stderr (with colours) and, optionally, to a rotating file.
    Should be called once at application startup.
    """
    logger.remove()  # Remove default handler

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    # Console handler
    logger.add(
        sys.stderr,
        format=log_format,
        level=settings.log_level,
        colorize=True,
        backtrace=True,
        diagnose=settings.debug,
    )

    # File handler (optional)
    if settings.log_file_path:
        settings.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(settings.log_file_path),
            format=log_format,
            level=settings.log_level,
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            backtrace=True,
            diagnose=settings.debug,
        )

    logger.info(
        "Logging initialised | level={} | file={}",
        settings.log_level,
        settings.log_file_path or "disabled",
    )


# ---------------------------------------------------------------------------
# URL Validation
# ---------------------------------------------------------------------------

_URL_RE = re.compile(
    r"^(?:http|https)://"
    r"(?:\S+(?::\S*)?@)?"
    r"(?:"
    r"(?!(?:10|127)(?:\.\d{1,3}){3})"
    r"(?!(?:169\.254|192\.168)(?:\.\d{1,3}){2})"
    r"(?!172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})"
    r"(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])"
    r"(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}"
    r"(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))"
    r"|"
    r"(?:(?:[a-z0-9\u00a1-\uffff][a-z0-9\u00a1-\uffff_-]{0,62})?"
    r"[a-z0-9\u00a1-\uffff]\.)+"
    r"(?:[a-z\u00a1-\uffff]{2,}\.?)"
    r")"
    r"(?::\d{2,5})?"
    r"(?:[/?#]\S*)?$",
    re.IGNORECASE,
)


def is_valid_url(url: str) -> bool:
    """Return True if *url* is a syntactically valid HTTP/HTTPS URL."""
    if not isinstance(url, str) or not url.strip():
        return False
    try:
        result = urlparse(url.strip())
        return result.scheme in ("http", "https") and bool(result.netloc)
    except Exception:
        return False


def validate_urls(urls: list[str]) -> tuple[list[str], list[str]]:
    """
    Split a list of URLs into valid and invalid buckets.

    Returns:
        (valid_urls, invalid_urls)
    """
    valid, invalid = [], []
    for url in urls:
        url = url.strip()
        if url and is_valid_url(url):
            valid.append(url)
        elif url:
            invalid.append(url)
    return valid, invalid


# ---------------------------------------------------------------------------
# Text Cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Normalise and clean raw text scraped from web pages.

    Steps:
      1. Unicode normalisation (NFC)
      2. Remove control characters (except newline / tab)
      3. Collapse multiple blank lines → single blank line
      4. Strip leading/trailing whitespace per line
      5. Strip overall leading/trailing whitespace
    """
    if not text:
        return ""

    # 1. Unicode normalisation
    text = unicodedata.normalize("NFC", text)

    # 2. Remove control characters (keep \n, \t, \r)
    text = "".join(
        ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t\r"
    )

    # 3. Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 4. Strip each line
    lines = [line.strip() for line in text.split("\n")]

    # 5. Collapse runs of blank lines
    cleaned_lines: list[str] = []
    blank_count = 0
    for line in lines:
        if line == "":
            blank_count += 1
            if blank_count <= 1:
                cleaned_lines.append(line)
        else:
            blank_count = 0
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def truncate_text(text: str, max_chars: int = 500) -> str:
    """Truncate *text* to *max_chars* and append ellipsis if needed."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


# ---------------------------------------------------------------------------
# Timing Decorator
# ---------------------------------------------------------------------------

def timed(func: F) -> F:
    """Decorator that logs the execution time of a function."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.debug("{} completed in {:.3f}s", func.__qualname__, elapsed)
            return result
        except Exception as exc:
            elapsed = time.perf_counter() - start
            logger.error(
                "{} failed after {:.3f}s: {}",
                func.__qualname__,
                elapsed,
                exc,
            )
            raise

    return wrapper  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Source formatting helper
# ---------------------------------------------------------------------------

def format_sources(sources: list[dict[str, Any]]) -> str:
    """
    Format a list of source metadata dicts into a readable string for display.

    Each dict is expected to have at least a 'url' key.
    """
    if not sources:
        return ""
    lines = ["**Sources:**"]
    seen: set[str] = set()
    for i, src in enumerate(sources, 1):
        url = src.get("url", "Unknown")
        if url in seen:
            continue
        seen.add(url)
        title = src.get("title", url)
        lines.append(f"{i}. [{title}]({url})")
    return "\n".join(lines)
