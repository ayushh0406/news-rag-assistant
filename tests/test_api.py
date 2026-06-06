"""
Integration Tests: FastAPI Endpoints
======================================
Uses FastAPI's TestClient to verify API contracts without
a live Gemini API key (pipeline methods are mocked).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api import app
from backend.rag_pipeline import AnswerResult, ProcessURLsResult


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient) -> None:
        with patch("api.get_pipeline") as mock_get:
            mock_pipeline = MagicMock()
            mock_pipeline.get_stats.return_value = {
                "collection_name": "test",
                "total_vectors": 0,
                "persist_path": "/tmp/chroma",
            }
            mock_get.return_value = mock_pipeline

            resp = client.get("/health")
            assert resp.status_code == 200

    def test_health_response_structure(self, client: TestClient) -> None:
        with patch("api.get_pipeline") as mock_get:
            mock_pipeline = MagicMock()
            mock_pipeline.get_stats.return_value = {"total_vectors": 42}
            mock_get.return_value = mock_pipeline

            resp = client.get("/health")
            data = resp.json()
            assert "status" in data
            assert "app_name" in data
            assert "version" in data


# ---------------------------------------------------------------------------
# /process-urls
# ---------------------------------------------------------------------------

class TestProcessUrlsEndpoint:
    def test_valid_urls_returns_200(self, client: TestClient) -> None:
        with patch("api.get_pipeline") as mock_get:
            mock_pipeline = MagicMock()
            mock_pipeline.process_urls.return_value = ProcessURLsResult(
                success=True,
                message="✅ Processed 1 article.",
                successful_urls=["https://example.com/news/1"],
                failed_urls=[],
                total_chunks=5,
            )
            mock_get.return_value = mock_pipeline

            resp = client.post(
                "/process-urls",
                json={"urls": ["https://example.com/news/1"], "reset": False},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert data["total_chunks"] == 5

    def test_invalid_url_format_returns_422(self, client: TestClient) -> None:
        resp = client.post(
            "/process-urls",
            json={"urls": ["not-a-url", "also-not-valid"]},
        )
        assert resp.status_code == 422

    def test_empty_urls_list_returns_422(self, client: TestClient) -> None:
        resp = client.post("/process-urls", json={"urls": []})
        assert resp.status_code == 422

    def test_too_many_urls_returns_422(self, client: TestClient) -> None:
        urls = [f"https://example.com/article-{i}" for i in range(25)]
        resp = client.post("/process-urls", json={"urls": urls})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /ask
# ---------------------------------------------------------------------------

class TestAskEndpoint:
    def test_ask_without_articles_returns_400(self, client: TestClient) -> None:
        with patch("api.get_pipeline") as mock_get:
            mock_pipeline = MagicMock()
            mock_pipeline.has_documents = False
            mock_get.return_value = mock_pipeline

            resp = client.post("/ask", json={"question": "What happened?"})
            assert resp.status_code == 400

    def test_ask_with_articles_returns_200(self, client: TestClient) -> None:
        with patch("api.get_pipeline") as mock_get:
            mock_pipeline = MagicMock()
            mock_pipeline.has_documents = True
            mock_pipeline.answer_question.return_value = AnswerResult(
                success=True,
                answer="The articles discuss a major tech earnings report.",
                sources=[
                    {
                        "url": "https://example.com/article",
                        "title": "Tech Earnings",
                        "domain": "example.com",
                        "description": "",
                    }
                ],
                sources_text="**Sources:**\n1. [Tech Earnings](https://example.com/article)",
                question="What happened?",
            )
            mock_get.return_value = mock_pipeline

            resp = client.post("/ask", json={"question": "What happened?"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert "answer" in data
            assert isinstance(data["sources"], list)
            assert data["processing_time_ms"] >= 0

    def test_empty_question_returns_422(self, client: TestClient) -> None:
        resp = client.post("/ask", json={"question": ""})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /clear-history
# ---------------------------------------------------------------------------

class TestClearHistoryEndpoint:
    def test_clear_history_returns_200(self, client: TestClient) -> None:
        with patch("api.get_pipeline") as mock_get:
            mock_pipeline = MagicMock()
            mock_get.return_value = mock_pipeline

            resp = client.delete("/clear-history")
            assert resp.status_code == 200
            assert "message" in resp.json()
