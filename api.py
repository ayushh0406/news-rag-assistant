"""
FastAPI Backend Server
======================
Provides RESTful endpoints that expose the RAG pipeline over HTTP.

Endpoints:
  GET  /health         → liveness / readiness probe
  POST /process-urls   → ingest news articles
  POST /ask            → answer a question using RAG

Run with:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from backend.config import settings
from backend.rag_pipeline import ProcessURLsResult, get_pipeline
from backend.utils import setup_logging, validate_urls


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # noqa: ARG001
    """Startup / shutdown lifecycle for the FastAPI app."""
    setup_logging()
    logger.info(
        "Starting {} v{} | debug={}",
        settings.app_name,
        settings.app_version,
        settings.debug,
    )
    yield
    logger.info("Shutting down {}.", settings.app_name)


# ---------------------------------------------------------------------------
# App instantiation
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Production-ready RAG API for news article research. "
        "Uses Gemini 2.5 Flash + ChromaDB for semantic search."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class ProcessURLsRequest(BaseModel):
    """Request payload for the /process-urls endpoint."""

    urls: list[str] = Field(
        ...,
        min_length=1,
        description=f"List of news article URLs (max {settings.max_urls}).",
        examples=[["https://example.com/news/article-1"]],
    )
    reset: bool = Field(
        default=False,
        description="If true, wipe existing ChromaDB data before ingesting.",
    )

    @field_validator("urls")
    @classmethod
    def validate_url_count(cls, v: list[str]) -> list[str]:
        if len(v) > settings.max_urls:
            raise ValueError(
                f"Too many URLs. Maximum allowed is {settings.max_urls}, "
                f"but {len(v)} were provided."
            )
        return v


class ProcessURLsResponse(BaseModel):
    """Response from /process-urls."""
    success: bool
    message: str
    successful_urls: list[str]
    failed_urls: list[dict[str, str]]
    total_chunks: int
    processing_time_ms: float


class AskRequest(BaseModel):
    """Request payload for the /ask endpoint."""
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The question to ask about the loaded news articles.",
        examples=["What are the main topics covered in these articles?"],
    )


class SourceModel(BaseModel):
    """A single source citation."""
    url: str
    title: str
    domain: str
    description: str = ""


class AskResponse(BaseModel):
    """Response from /ask."""
    success: bool
    question: str
    answer: str
    sources: list[SourceModel]
    sources_text: str
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Response from /health."""
    status: str
    app_name: str
    version: str
    vector_store_stats: dict[str, Any]
    uptime_seconds: float


# ---------------------------------------------------------------------------
# Startup timestamp
# ---------------------------------------------------------------------------

_START_TIME = time.time()


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on {}: {}", request.url, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Internal server error: {exc}"},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["System"],
)
async def health_check() -> HealthResponse:
    """
    Liveness and readiness probe.

    Returns application metadata and vector store statistics.
    """
    pipeline = get_pipeline()
    return HealthResponse(
        status="healthy",
        app_name=settings.app_name,
        version=settings.app_version,
        vector_store_stats=pipeline.get_stats(),
        uptime_seconds=round(time.time() - _START_TIME, 2),
    )


@app.post(
    "/process-urls",
    response_model=ProcessURLsResponse,
    summary="Ingest news articles from URLs",
    tags=["RAG"],
)
async def process_urls(request: ProcessURLsRequest) -> ProcessURLsResponse:
    """
    Fetch, parse, embed, and store news articles from the provided URLs.

    - Validates URL format before processing.
    - Returns per-URL success/failure details.
    - Optionally resets the ChromaDB collection (``reset=true``).
    """
    # Pre-validate URLs
    valid_urls, invalid_urls = validate_urls(request.urls)

    if not valid_urls:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="None of the provided URLs are valid HTTP/HTTPS URLs.",
        )

    # Pre-populate failed list with invalid URLs
    pre_failed = [
        {"url": u, "error": "Invalid URL format"} for u in invalid_urls
    ]

    start = time.perf_counter()
    pipeline = get_pipeline()

    try:
        result: ProcessURLsResult = pipeline.process_urls(valid_urls, reset=request.reset)
    except Exception as exc:
        logger.exception("process_urls endpoint error: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    elapsed_ms = (time.perf_counter() - start) * 1000

    return ProcessURLsResponse(
        success=result.success,
        message=result.message,
        successful_urls=result.successful_urls,
        failed_urls=result.failed_urls + pre_failed,
        total_chunks=result.total_chunks,
        processing_time_ms=round(elapsed_ms, 2),
    )


@app.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question about loaded articles",
    tags=["RAG"],
)
async def ask_question(request: AskRequest) -> AskResponse:
    """
    Answer a natural-language question using the RAG pipeline.

    Requires at least one article to have been loaded via ``/process-urls``.
    Maintains conversation history within the current server session.
    """
    start = time.perf_counter()
    pipeline = get_pipeline()

    if not pipeline.has_documents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "No articles have been loaded yet. "
                "Please call /process-urls first."
            ),
        )

    try:
        result = pipeline.answer_question(request.question)
    except Exception as exc:
        logger.exception("ask endpoint error: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    elapsed_ms = (time.perf_counter() - start) * 1000

    sources = [
        SourceModel(
            url=s.get("url", ""),
            title=s.get("title", ""),
            domain=s.get("domain", ""),
            description=s.get("description", ""),
        )
        for s in result.sources
    ]

    return AskResponse(
        success=result.success,
        question=result.question,
        answer=result.answer,
        sources=sources,
        sources_text=result.sources_text,
        processing_time_ms=round(elapsed_ms, 2),
    )


@app.delete(
    "/clear-history",
    summary="Clear conversation history",
    tags=["RAG"],
)
async def clear_history() -> dict[str, str]:
    """Clear the in-memory conversation history for the current session."""
    pipeline = get_pipeline()
    pipeline.clear_history()
    return {"message": "Conversation history cleared."}


# ---------------------------------------------------------------------------
# Entry point (for direct execution)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower(),
    )
