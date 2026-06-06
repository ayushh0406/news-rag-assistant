"""
AI News Research Assistant — Streamlit Frontend
================================================
Premium, modern chat interface for RAG-powered news research.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import time
from typing import Any

import streamlit as st
from loguru import logger

from backend.config import settings
from backend.rag_pipeline import get_pipeline
from backend.utils import setup_logging, validate_urls

# ---------------------------------------------------------------------------
# Page Config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI News Research Assistant",
    page_icon="🗞️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/your-repo/news-research-assistant",
        "Report a bug": "https://github.com/your-repo/news-research-assistant/issues",
        "About": "AI News Research Assistant powered by Gemini 2.5 Flash & ChromaDB",
    },
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
setup_logging()


# ---------------------------------------------------------------------------
# Custom CSS — Premium Dark Theme
# ---------------------------------------------------------------------------

def _inject_css() -> None:
    st.markdown(
        """
        <style>
        /* ── Google Fonts ── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        /* ── Root Variables ── */
        :root {
            --bg-primary:    #0d1117;
            --bg-secondary:  #161b22;
            --bg-tertiary:   #21262d;
            --bg-glass:      rgba(22, 27, 34, 0.85);
            --accent-blue:   #58a6ff;
            --accent-purple: #bc8cff;
            --accent-green:  #3fb950;
            --accent-orange: #f0883e;
            --accent-red:    #f85149;
            --text-primary:  #e6edf3;
            --text-secondary:#8b949e;
            --text-muted:    #6e7681;
            --border:        rgba(48, 54, 61, 0.8);
            --shadow-lg:     0 16px 40px rgba(0,0,0,0.5);
            --shadow-glow:   0 0 20px rgba(88, 166, 255, 0.15);
            --radius-sm:     8px;
            --radius-md:     12px;
            --radius-lg:     16px;
            --radius-xl:     24px;
        }

        /* ── Global Reset ── */
        * { box-sizing: border-box; }

        .stApp {
            background: linear-gradient(135deg, #0d1117 0%, #0f1923 50%, #0d1117 100%);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: var(--text-primary);
        }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {
            background: var(--bg-secondary) !important;
            border-right: 1px solid var(--border) !important;
        }
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p {
            color: var(--text-primary) !important;
        }

        /* ── Text Inputs ── */
        .stTextArea textarea,
        .stTextInput input {
            background: var(--bg-tertiary) !important;
            border: 1px solid var(--border) !important;
            color: var(--text-primary) !important;
            border-radius: var(--radius-sm) !important;
            font-family: 'Inter', sans-serif !important;
            transition: border-color 0.2s ease !important;
        }
        .stTextArea textarea:focus,
        .stTextInput input:focus {
            border-color: var(--accent-blue) !important;
            box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.12) !important;
        }

        /* ── Buttons ── */
        .stButton > button {
            background: linear-gradient(135deg, var(--accent-blue), #1f6feb) !important;
            color: #fff !important;
            border: none !important;
            border-radius: var(--radius-sm) !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 600 !important;
            letter-spacing: 0.3px !important;
            padding: 0.5rem 1.25rem !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 2px 8px rgba(88, 166, 255, 0.3) !important;
        }
        .stButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 16px rgba(88, 166, 255, 0.45) !important;
        }
        .stButton > button:active {
            transform: translateY(0) !important;
        }

        /* ── Chat Messages ── */
        .chat-message {
            display: flex;
            gap: 12px;
            margin: 12px 0;
            animation: fadeInUp 0.3s ease;
        }
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(8px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        .chat-avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            flex-shrink: 0;
        }
        .chat-avatar.user   { background: linear-gradient(135deg, #1f6feb, var(--accent-blue)); }
        .chat-avatar.ai     { background: linear-gradient(135deg, #6e40c9, var(--accent-purple)); }
        .chat-bubble {
            max-width: 82%;
            padding: 14px 18px;
            border-radius: var(--radius-md);
            line-height: 1.65;
            font-size: 0.925rem;
        }
        .chat-bubble.user {
            background: linear-gradient(135deg, rgba(31,111,235,0.2), rgba(88,166,255,0.1));
            border: 1px solid rgba(88,166,255,0.25);
            color: var(--text-primary);
            margin-left: auto;
        }
        .chat-bubble.ai {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            color: var(--text-primary);
        }
        .chat-timestamp {
            font-size: 0.72rem;
            color: var(--text-muted);
            margin-top: 5px;
            opacity: 0.7;
        }

        /* ── Source Card ── */
        .source-card {
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
            padding: 10px 14px;
            margin: 4px 0;
            transition: border-color 0.2s ease;
        }
        .source-card:hover { border-color: var(--accent-blue); }
        .source-card a {
            color: var(--accent-blue) !important;
            text-decoration: none;
            font-weight: 500;
            font-size: 0.875rem;
        }
        .source-card .domain {
            color: var(--text-muted);
            font-size: 0.75rem;
            margin-top: 2px;
        }

        /* ── Metric Cards ── */
        .metric-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 16px;
            text-align: center;
        }
        .metric-card .value {
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .metric-card .label {
            font-size: 0.78rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin-top: 4px;
        }

        /* ── Section Headers ── */
        .section-header {
            font-size: 0.72rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1.2px;
            margin: 18px 0 8px;
            padding-bottom: 6px;
            border-bottom: 1px solid var(--border);
        }

        /* ── Status Badge ── */
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.78rem;
            font-weight: 600;
        }
        .status-badge.success {
            background: rgba(63,185,80,0.12);
            color: var(--accent-green);
            border: 1px solid rgba(63,185,80,0.3);
        }
        .status-badge.warning {
            background: rgba(240,136,62,0.12);
            color: var(--accent-orange);
            border: 1px solid rgba(240,136,62,0.3);
        }
        .status-badge.error {
            background: rgba(248,81,73,0.12);
            color: var(--accent-red);
            border: 1px solid rgba(248,81,73,0.3);
        }
        .status-badge.info {
            background: rgba(88,166,255,0.12);
            color: var(--accent-blue);
            border: 1px solid rgba(88,166,255,0.3);
        }

        /* ── Loading Spinner ── */
        .thinking-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            color: var(--text-secondary);
            font-size: 0.875rem;
            padding: 12px 16px;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            animation: pulse 1.5s ease infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50%       { opacity: 0.6; }
        }
        .dot {
            width: 8px; height: 8px;
            border-radius: 50%;
            background: var(--accent-blue);
            animation: bounce 1.4s infinite ease-in-out both;
        }
        .dot:nth-child(1) { animation-delay: -0.32s; }
        .dot:nth-child(2) { animation-delay: -0.16s; }
        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40%           { transform: scale(1); }
        }

        /* ── Hero Banner ── */
        .hero-banner {
            background: linear-gradient(135deg, rgba(31,111,235,0.08), rgba(110,64,201,0.08));
            border: 1px solid rgba(88,166,255,0.15);
            border-radius: var(--radius-xl);
            padding: 32px 40px;
            margin-bottom: 28px;
            text-align: center;
        }
        .hero-title {
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 8px;
        }
        .hero-subtitle {
            color: var(--text-secondary);
            font-size: 1rem;
            font-weight: 400;
        }

        /* ── Scrollbar ── */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb {
            background: var(--bg-tertiary);
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

        /* ── Streamlit Overrides ── */
        .stAlert { border-radius: var(--radius-sm) !important; }
        [data-testid="stMarkdownContainer"] p { color: var(--text-primary); }
        .stExpander { border: 1px solid var(--border) !important; border-radius: var(--radius-sm) !important; }
        .stExpander summary { color: var(--text-primary) !important; }
        .streamlit-expanderHeader { background: var(--bg-tertiary) !important; }
        footer { visibility: hidden; }
        #MainMenu { visibility: hidden; }
        header { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Session State Initialisation
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    """Initialise all Streamlit session state keys."""
    defaults: dict[str, Any] = {
        "messages": [],          # Chat history: [{"role": "user"|"ai", "content": ..., "sources": [...], "ts": ...}]
        "pipeline": None,        # RAGPipeline singleton
        "articles_loaded": False,
        "processed_urls": [],
        "total_chunks": 0,
        "failed_urls": [],
        "url_inputs": [""] * 3,  # Start with 3 URL fields
        "api_key_confirmed": False,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ---------------------------------------------------------------------------
# Pipeline accessor
# ---------------------------------------------------------------------------

def _get_pipeline():
    """Return (and cache in session state) the RAGPipeline singleton."""
    if st.session_state.pipeline is None:
        st.session_state.pipeline = get_pipeline()
    return st.session_state.pipeline


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    """Render the left sidebar with URL management and settings."""
    with st.sidebar:
        # Logo / branding
        st.markdown(
            """
            <div style="text-align:center; padding: 16px 0 24px;">
                <div style="font-size:2.8rem;">🗞️</div>
                <div style="font-weight:700; font-size:1.05rem; color:#e6edf3; margin-top:6px;">
                    News Research Assistant
                </div>
                <div style="font-size:0.75rem; color:#6e7681; margin-top:2px;">
                    Powered by Gemini 2.5 Flash
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Status ──
        st.markdown('<div class="section-header">📊 Status</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            articles_count = len(st.session_state.processed_urls)
            # Sync with actual DB state on startup
            pipeline = _get_pipeline()
            db_stats = pipeline.get_stats()
            db_chunks = db_stats.get("total_vectors", 0)
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="value">{articles_count}</div>
                    <div class="label">Articles</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="value">{db_chunks}</div>
                    <div class="label">Chunks</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── URL Input ──
        st.markdown('<div class="section-header">🔗 News Article URLs</div>', unsafe_allow_html=True)
        st.caption(f"Add up to {settings.max_urls} news article URLs")

        # Dynamic URL fields
        if "url_count" not in st.session_state:
            st.session_state.url_count = 3

        urls_entered: list[str] = []
        for i in range(st.session_state.url_count):
            url = st.text_input(
                label=f"URL {i + 1}",
                key=f"url_input_{i}",
                placeholder="https://example.com/news/article",
                label_visibility="collapsed",
            )
            if url.strip():
                urls_entered.append(url.strip())

        # Add / Remove URL field buttons
        col_add, col_rem = st.columns(2)
        with col_add:
            if st.session_state.url_count < settings.max_urls:
                if st.button("＋ Add URL", use_container_width=True):
                    st.session_state.url_count = min(
                        st.session_state.url_count + 1, settings.max_urls
                    )
                    st.rerun()
        with col_rem:
            if st.session_state.url_count > 1:
                if st.button("－ Remove", use_container_width=True):
                    st.session_state.url_count = max(st.session_state.url_count - 1, 1)
                    st.rerun()

        st.markdown("---")

        # ── Process Options ──
        reset_collection = st.toggle(
            "🔄 Reset existing data",
            value=False,
            help="Delete all previously loaded articles before processing new URLs.",
        )

        # Process button
        if st.button(
            "⚡ Process URLs",
            use_container_width=True,
            type="primary",
            disabled=not urls_entered,
        ):
            _handle_process_urls(urls_entered, reset=reset_collection)

        # ── Processed Articles ──
        if st.session_state.processed_urls:
            st.markdown(
                '<div class="section-header">✅ Loaded Articles</div>',
                unsafe_allow_html=True,
            )
            for url in st.session_state.processed_urls:
                domain = url.split("//")[-1].split("/")[0].replace("www.", "")
                st.markdown(
                    f"""
                    <div class="source-card">
                        <a href="{url}" target="_blank" title="{url}">{domain}</a>
                        <div class="domain">{url[:55]}{"…" if len(url) > 55 else ""}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ── Failed URLs ──
        if st.session_state.failed_urls:
            with st.expander(f"⚠️ {len(st.session_state.failed_urls)} Failed URL(s)"):
                for item in st.session_state.failed_urls:
                    st.error(f"**{item['url'][:50]}…**\n{item['error']}")

        st.markdown("---")

        # ── Chat Controls ──
        st.markdown('<div class="section-header">💬 Chat Controls</div>', unsafe_allow_html=True)
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            pipeline = _get_pipeline()
            pipeline.clear_history()
            st.rerun()

        # ── Settings info ──
        st.markdown("---")
        with st.expander("⚙️ Configuration"):
            st.markdown(
                f"""
                | Setting | Value |
                |---------|-------|
                | **Model** | `{settings.gemini_model}` |
                | **Embeddings** | `{settings.gemini_embedding_model}` |
                | **Temperature** | `{settings.temperature}` |
                | **Chunk Size** | `{settings.chunk_size}` |
                | **Chunk Overlap** | `{settings.chunk_overlap}` |
                | **Top-K** | `{settings.top_k_results}` |
                """
            )


# ---------------------------------------------------------------------------
# URL Processing Handler
# ---------------------------------------------------------------------------

def _handle_process_urls(urls: list[str], reset: bool = False) -> None:
    """Validate and process URLs, updating session state."""
    # Validate
    valid_urls, invalid_urls = validate_urls(urls)

    if invalid_urls:
        for bad_url in invalid_urls:
            st.sidebar.warning(f"⚠️ Invalid URL skipped: `{bad_url}`")

    if not valid_urls:
        st.sidebar.error("❌ No valid URLs to process.")
        return

    pipeline = _get_pipeline()

    with st.sidebar:
        with st.spinner("🔄 Loading and processing articles…"):
            progress_bar = st.progress(0, text="Initialising…")

            try:
                progress_bar.progress(10, text="Fetching articles…")
                result = pipeline.process_urls(valid_urls, reset=reset)
                progress_bar.progress(70, text="Embedding and storing…")
                time.sleep(0.3)
                progress_bar.progress(100, text="Done!")
                time.sleep(0.3)
                progress_bar.empty()

                if result.success:
                    st.session_state.processed_urls = pipeline.processed_urls
                    # Get true total from DB rather than just newly added
                    st.session_state.total_chunks = pipeline.get_stats().get("total_vectors", result.total_chunks)
                    st.session_state.articles_loaded = True
                    st.session_state.failed_urls = result.failed_urls

                    st.success(result.message)
                else:
                    st.error(result.message)
                    st.session_state.failed_urls = result.failed_urls

            except Exception as exc:
                progress_bar.empty()
                st.error(f"❌ Processing failed: {exc}")
                logger.exception("URL processing error: {}", exc)

    st.rerun()


# ---------------------------------------------------------------------------
# Chat Interface
# ---------------------------------------------------------------------------

def render_chat_message(msg: dict[str, Any]) -> None:
    """Render a single chat message bubble."""
    role = msg["role"]
    content = msg["content"]
    sources = msg.get("sources", [])
    ts = msg.get("ts", "")

    if role == "user":
        st.markdown(
            f"""
            <div class="chat-message">
                <div style="flex:1"></div>
                <div>
                    <div class="chat-bubble user">{content}</div>
                    <div class="chat-timestamp" style="text-align:right;">{ts}</div>
                </div>
                <div class="chat-avatar user">👤</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # AI message — render avatar + bubble wrapper, then markdown content, then sources
        st.markdown(
            """
            <div class="chat-message">
                <div class="chat-avatar ai">🤖</div>
                <div style="flex:1; min-width:0;">
                    <div class="chat-bubble ai">
            """,
            unsafe_allow_html=True,
        )
        # Render markdown content properly (bold, bullets, etc.)
        st.markdown(content)

        # Render source cards as HTML
        if sources:
            source_items = "".join(
                f"""
                <div class="source-card">
                    <a href="{s['url']}" target="_blank">{s['title'] or s['url']}</a>
                    <div class="domain">🌐 {s.get('domain', s['url'][:40])}</div>
                </div>
                """
                for s in sources
            )
            st.markdown(
                f"""
                <div style="margin-top:12px;">
                    <div class="section-header" style="margin-top:0;">📎 Sources</div>
                    {source_items}
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
                    </div>
                    <div class="chat-timestamp">{ts}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_main_chat() -> None:
    """Render the main chat area."""
    # Hero banner
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">🗞️ AI News Research Assistant</div>
            <div class="hero-subtitle">
                Ask questions about any news articles — get AI-powered answers with source citations
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Status row
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.session_state.articles_loaded:
            st.markdown(
                '<span class="status-badge success">✅ Articles Loaded</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-badge warning">⚠️ No Articles Loaded</span>',
                unsafe_allow_html=True,
            )
    with col2:
        turns = len(st.session_state.messages) // 2
        st.markdown(
            f'<span class="status-badge info">💬 {turns} Conversation Turn{"s" if turns != 1 else ""}</span>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<span class="status-badge info">🧠 {settings.gemini_model}</span>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Onboarding hint
    if not st.session_state.articles_loaded:
        st.info(
            "👈 **Get started:** Add news article URLs in the sidebar and click **Process URLs**. "
            "Then come back here to ask questions!",
            icon="💡",
        )

    # Chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            render_chat_message(msg)

    # Thinking indicator placeholder
    thinking_placeholder = st.empty()

    # ── Input ──
    st.markdown("---")
    with st.form(key="chat_form", clear_on_submit=True):
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            user_input = st.text_input(
                label="Ask a question",
                placeholder="e.g. What are the key findings in these articles?",
                label_visibility="collapsed",
                key="user_question",
            )
        with col_btn:
            submit = st.form_submit_button(
                "Send 🚀",
                use_container_width=True,
                type="primary",
            )

    # Example questions
    if not st.session_state.messages:
        st.markdown(
            '<div class="section-header">💡 Example Questions</div>',
            unsafe_allow_html=True,
        )
        example_cols = st.columns(3)
        examples = [
            "What are the main topics covered?",
            "Summarise the key findings",
            "What opinions are expressed?",
            "Who are the key people mentioned?",
            "What events are described?",
            "Compare the perspectives in these articles",
        ]
        for i, example in enumerate(examples):
            with example_cols[i % 3]:
                if st.button(f'"{example}"', use_container_width=True, key=f"ex_{i}"):
                    _process_question(example, thinking_placeholder)
                    st.rerun()

    # Handle form submission
    if submit and user_input.strip():
        _process_question(user_input.strip(), thinking_placeholder)
        st.rerun()


def _process_question(question: str, thinking_placeholder: Any) -> None:
    """Process a user question through the RAG pipeline."""
    import time as _time

    ts = _time.strftime("%H:%M")

    # Add user message
    st.session_state.messages.append(
        {"role": "user", "content": question, "ts": ts}
    )

    # Show thinking indicator
    thinking_placeholder.markdown(
        """
        <div class="thinking-indicator">
            <div class="dot"></div>
            <div class="dot"></div>
            <div class="dot"></div>
            <span style="margin-left:4px;">AI is researching the articles…</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        pipeline = _get_pipeline()
        start = _time.perf_counter()
        result = pipeline.answer_question(question)
        elapsed = _time.perf_counter() - start

        thinking_placeholder.empty()

        # Add AI message
        st.session_state.messages.append(
            {
                "role": "ai",
                "content": result.answer,
                "sources": result.sources,
                "ts": f"{ts} · {elapsed:.1f}s",
            }
        )

        if not result.success:
            logger.warning("Question answered with failure flag: {}", question)

    except Exception as exc:
        thinking_placeholder.empty()
        error_msg = f"❌ Sorry, I encountered an error: {exc}"
        logger.exception("Question processing error: {}", exc)
        st.session_state.messages.append(
            {"role": "ai", "content": error_msg, "sources": [], "ts": ts}
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Application entry point."""
    _init_session_state()
    _inject_css()
    render_sidebar()
    render_main_chat()


if __name__ == "__main__":
    main()
