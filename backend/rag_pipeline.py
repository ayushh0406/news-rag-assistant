"""
RAG Pipeline Module
===================
Orchestrates the full Retrieval-Augmented Generation flow:

  1. URL loading       → ``loader.load_articles``
  2. Document chunking → ``chunker.chunk_documents``
  3. Vector storage    → ``vectorstore.NewsVectorStore.add_documents``
  4. Query rewriting   → standalone question condensation (multi-turn)
  5. Retrieval         → semantic similarity search
  6. Answer generation → Gemini 2.5 Flash via LangChain

Exposes two primary public functions:
  - ``process_urls``  : ingest articles and populate the vector store
  - ``answer_question``: retrieve context and generate an answer

The pipeline keeps an in-memory conversation history for session memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from loguru import logger

from backend.chunker import chunk_documents
from backend.config import settings
from backend.loader import BulkLoadResult, load_articles
from backend.prompts import (
    get_condense_question_prompt,
    get_empty_results_message,
    get_no_context_message,
    get_rag_prompt,
)
from backend.utils import format_sources, timed
from backend.vectorstore import NewsVectorStore, get_vector_store


# ---------------------------------------------------------------------------
# Data classes / response models
# ---------------------------------------------------------------------------

@dataclass
class ProcessURLsResult:
    """Result returned after ingesting article URLs."""
    success: bool
    message: str
    successful_urls: list[str] = field(default_factory=list)
    failed_urls: list[dict[str, str]] = field(default_factory=list)
    total_chunks: int = 0
    bulk_result: Optional[BulkLoadResult] = None


@dataclass
class AnswerResult:
    """Result returned after answering a user question."""
    success: bool
    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    sources_text: str = ""
    context_docs: list[Any] = field(default_factory=list)
    question: str = ""


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _get_llm() -> ChatGoogleGenerativeAI:
    """Instantiate the Gemini 2.5 Flash chat model."""
    if not settings.google_api_key:
        raise ValueError(
            "GOOGLE_API_KEY is not set. Please configure it in your .env file."
        )
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=settings.temperature,
        max_output_tokens=settings.max_output_tokens,
        convert_system_message_to_human=False,
    )


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    Stateful RAG pipeline managing:
      - Article ingestion
      - Persistent ChromaDB vector store
      - In-session conversation history
      - LangChain retrieval chain
    """

    def __init__(self) -> None:
        self._vector_store: NewsVectorStore = get_vector_store()
        self._llm: Optional[ChatGoogleGenerativeAI] = None
        self._chat_history: list[BaseMessage] = []
        self._processed_urls: list[str] = []
        logger.info("RAGPipeline initialised.")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def llm(self) -> ChatGoogleGenerativeAI:
        """Lazy-initialise and cache the LLM."""
        if self._llm is None:
            self._llm = _get_llm()
            logger.info("LLM '{}' initialised.", settings.gemini_model)
        return self._llm

    @property
    def has_documents(self) -> bool:
        """Return True if the vector store contains at least one chunk."""
        return not self._vector_store.is_empty()

    @property
    def chat_history(self) -> list[BaseMessage]:
        return self._chat_history

    @property
    def processed_urls(self) -> list[str]:
        return self._processed_urls.copy()

    # ------------------------------------------------------------------
    # Article ingestion
    # ------------------------------------------------------------------

    @timed
    def process_urls(
        self,
        urls: list[str],
        reset: bool = False,
    ) -> ProcessURLsResult:
        """
        Load articles from *urls*, chunk them, and store in ChromaDB.

        Args:
            urls:  List of HTTP/HTTPS news article URLs.
            reset: If True, wipe existing ChromaDB data before ingesting.

        Returns:
            :class:`ProcessURLsResult` with ingestion statistics.
        """
        if not urls:
            return ProcessURLsResult(
                success=False,
                message="No URLs provided.",
            )

        # Optionally reset
        if reset:
            logger.info("Resetting ChromaDB collection before ingestion.")
            self._vector_store.reset_collection()
            self._processed_urls.clear()

        # 1. Load articles
        logger.info("Loading {} URL(s)…", len(urls))
        bulk = load_articles(urls)

        if not bulk.documents:
            return ProcessURLsResult(
                success=False,
                message=(
                    "❌ No articles could be loaded. "
                    "Please check the URLs and try again."
                ),
                failed_urls=bulk.failed_urls,
                bulk_result=bulk,
            )

        # 2. Chunk
        logger.info("Chunking {} document(s)…", len(bulk.documents))
        chunks = chunk_documents(bulk.documents)

        if not chunks:
            return ProcessURLsResult(
                success=False,
                message="Documents were loaded but could not be split into chunks.",
                successful_urls=bulk.successful_urls,
                failed_urls=bulk.failed_urls,
                bulk_result=bulk,
            )

        # 3. Store
        logger.info("Storing {} chunk(s) in ChromaDB…", len(chunks))
        stored = self._vector_store.add_documents(chunks)

        self._processed_urls.extend(bulk.successful_urls)

        # Build summary message
        parts = [f"✅ Successfully processed **{len(bulk.successful_urls)}** article(s)."]
        parts.append(f"📄 Created **{stored}** searchable chunks.")
        if bulk.failed_urls:
            parts.append(
                f"⚠️ **{len(bulk.failed_urls)}** URL(s) failed — see details below."
            )

        return ProcessURLsResult(
            success=True,
            message="\n".join(parts),
            successful_urls=bulk.successful_urls,
            failed_urls=bulk.failed_urls,
            total_chunks=stored,
            bulk_result=bulk,
        )

    # ------------------------------------------------------------------
    # Question answering
    # ------------------------------------------------------------------

    def _build_retrieval_chain(self) -> Any:
        """
        Build a LangChain history-aware retrieval chain.

        Chain structure:
          condense_question_chain  → history-aware retriever
          stuff_documents_chain    → answer generation
          retrieval_chain          → combines both
        """
        retriever = self._vector_store.as_retriever()

        # Step 1: Condense the follow-up question into a standalone query
        history_aware_retriever = create_history_aware_retriever(
            llm=self.llm,
            retriever=retriever,
            prompt=get_condense_question_prompt(),
        )

        # Step 2: Generate answer from retrieved context
        document_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=get_rag_prompt(),
        )

        # Step 3: Wire them together
        return create_retrieval_chain(history_aware_retriever, document_chain)

    @timed
    def answer_question(self, question: str) -> AnswerResult:
        """
        Answer a user question using the RAG pipeline.

        Args:
            question: The user's natural-language question.

        Returns:
            :class:`AnswerResult` with the answer, sources, and context docs.
        """
        if not question.strip():
            return AnswerResult(
                success=False,
                answer="Please enter a question.",
                question=question,
            )

        # Guard: no documents loaded
        if not self.has_documents:
            return AnswerResult(
                success=False,
                answer=get_no_context_message(),
                question=question,
            )

        try:
            chain = self._build_retrieval_chain()

            logger.info("Answering: '{}'", question[:100])
            response = chain.invoke(
                {
                    "input": question,
                    "chat_history": self._chat_history,
                }
            )

            answer: str = response.get("answer", "")
            context_docs: list = response.get("context", [])

            if not answer.strip():
                return AnswerResult(
                    success=False,
                    answer=get_empty_results_message(question),
                    question=question,
                )

            # Deduplicate and extract sources from retrieved chunks
            sources = _extract_sources(context_docs)
            sources_text = format_sources(sources)

            # Update conversation history
            self._chat_history.append(HumanMessage(content=question))
            self._chat_history.append(AIMessage(content=answer))

            logger.success("Answer generated ({} chars).", len(answer))

            return AnswerResult(
                success=True,
                answer=answer,
                sources=sources,
                sources_text=sources_text,
                context_docs=context_docs,
                question=question,
            )

        except Exception as exc:
            logger.exception("Error generating answer: {}", exc)
            return AnswerResult(
                success=False,
                answer=f"❌ An error occurred while generating the answer: {exc}",
                question=question,
            )

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def clear_history(self) -> None:
        """Clear the in-memory conversation history."""
        self._chat_history.clear()
        logger.info("Conversation history cleared.")

    def get_stats(self) -> dict[str, Any]:
        """Return pipeline and vector store statistics."""
        vs_stats = self._vector_store.get_stats()
        return {
            **vs_stats,
            "processed_urls": self._processed_urls,
            "conversation_turns": len(self._chat_history) // 2,
            "gemini_model": settings.gemini_model,
            "embedding_model": settings.gemini_embedding_model,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_sources(context_docs: list) -> list[dict[str, Any]]:
    """Deduplicate and format source metadata from retrieved documents."""
    seen_urls: set[str] = set()
    sources: list[dict[str, Any]] = []

    for doc in context_docs:
        meta = getattr(doc, "metadata", {})
        url = meta.get("url") or meta.get("source", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            sources.append(
                {
                    "url": url,
                    "title": meta.get("title", url),
                    "domain": meta.get("domain", ""),
                    "description": meta.get("description", ""),
                }
            )

    return sources


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_pipeline_instance: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    """Return the application-level :class:`RAGPipeline` singleton."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline()
    return _pipeline_instance
