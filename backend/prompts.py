"""
Prompts Module
==============
Centralised prompt templates for the RAG pipeline.

Keeping all prompts in one place makes experimentation and A/B testing easy
without touching business logic.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# ---------------------------------------------------------------------------
# System messages
# ---------------------------------------------------------------------------

_RAG_SYSTEM_PROMPT = """\
You are an expert AI News Research Assistant with deep analytical capabilities.
Your role is to answer questions based **exclusively** on the news articles provided as context.

## Core Guidelines

1. **Ground every answer in the provided context.**  
   If the context does not contain enough information, say so honestly.  
   Never fabricate facts, statistics, or quotes.

2. **Be comprehensive yet concise.**  
   Provide thorough, well-structured answers. Use bullet points or numbered lists  
   for multi-part answers. Avoid unnecessary filler.

3. **Cite your sources.**  
   When referencing specific facts or quotes, mention the article title or URL  
   where that information appeared. Use inline citations like [Source: <title>].

4. **Maintain journalistic objectivity.**  
   Present multiple perspectives when they exist in the source material.  
   Do not insert personal opinions or biases.

5. **Handle uncertainty gracefully.**  
   If asked about something not covered in the articles, clearly state:  
   "The provided articles do not contain information about [topic]."

6. **Preserve context across the conversation.**  
   Use the chat history to answer follow-up questions coherently.

---

## Context from News Articles

{context}

---

Remember: Only answer based on the context above. If the answer is not in the context,  
say "I couldn't find relevant information in the provided articles."
"""

_CONDENSE_QUESTION_SYSTEM_PROMPT = """\
You are a conversation context analyser.

Given the chat history and a new follow-up question from the user, rewrite the  
follow-up question into a **standalone question** that fully captures the user's intent  
without requiring any prior context.

Rules:
- Keep the rewritten question concise and specific.
- If the follow-up is already standalone, return it unchanged.
- Do NOT answer the question — only rephrase it.
- Preserve the language of the original question.
"""


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

def get_rag_prompt() -> ChatPromptTemplate:
    """
    Return the main RAG answer-generation prompt.

    Slots:
      - ``context``:       Retrieved document chunks (filled by the pipeline).
      - ``chat_history``:  List of previous (human, ai) message pairs.
      - ``input``:         The user's current question.
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", _RAG_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )


def get_condense_question_prompt() -> ChatPromptTemplate:
    """
    Return the question-condensation prompt used for multi-turn conversations.

    Slots:
      - ``chat_history``:  Conversation history.
      - ``input``:         The raw follow-up question.
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", _CONDENSE_QUESTION_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )


def get_no_context_message() -> str:
    """Return a friendly message when no articles have been loaded yet."""
    return (
        "⚠️ **No articles loaded yet.**\n\n"
        "Please add news article URLs in the sidebar and click **Process URLs** "
        "before asking questions. The assistant needs article content to answer accurately."
    )


def get_empty_results_message(query: str) -> str:
    """Return a message when similarity search returns no results."""
    return (
        f"🔍 I searched the loaded articles for *\"{query}\"* but couldn't find "
        "relevant information.\n\n"
        "**Suggestions:**\n"
        "- Try rephrasing your question.\n"
        "- Make sure the topic is covered in the loaded articles.\n"
        "- Load more articles related to your question."
    )
