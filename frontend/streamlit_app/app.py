from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.domain.chat import ChatMessage
from app.infrastructure.file_storage.local import LocalFileStorage
from app.infrastructure.llm.base import MockLLMProvider
from app.services.answer_service import AnswerService
from app.services.chunking_service import ChunkingConfig, ChunkingService
from app.services.llm_service import LLMService
from app.services.parsing_service import ParsingService
from app.services.preprocessing_service import PreprocessingService
from app.services.prompt_service import PromptService
from app.services.upload_service import UploadService


class StreamlitUploadFile:
    def __init__(self, uploaded_file) -> None:
        self.filename = uploaded_file.name
        self.content_type = uploaded_file.type or "application/pdf"
        self._content = uploaded_file.getvalue()

    async def read(self) -> bytes:
        return self._content


def configure_page() -> None:
    st.set_page_config(
        page_title="DocMind AI",
        page_icon="DM",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        :root {
            --dm-bg: #f7f7f8;
            --dm-panel: #ffffff;
            --dm-border: #d9d9e3;
            --dm-text: #202123;
            --dm-muted: #6e6e80;
            --dm-accent: #10a37f;
            --dm-accent-dark: #0e8064;
            --dm-code: #f4f4f5;
        }
        .stApp {
            background: var(--dm-bg);
            color: var(--dm-text);
        }
        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--dm-border);
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            letter-spacing: 0;
        }
        .dm-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 14px 0 8px;
            border-bottom: 1px solid var(--dm-border);
            margin-bottom: 18px;
        }
        .dm-title {
            font-size: 24px;
            font-weight: 700;
            letter-spacing: 0;
        }
        .dm-subtitle {
            color: var(--dm-muted);
            font-size: 14px;
            margin-top: 2px;
        }
        .dm-badge {
            border: 1px solid var(--dm-border);
            border-radius: 6px;
            padding: 6px 10px;
            color: var(--dm-muted);
            background: #fff;
            font-size: 13px;
        }
        .dm-panel {
            background: var(--dm-panel);
            border: 1px solid var(--dm-border);
            border-radius: 8px;
            padding: 16px;
        }
        .dm-metric-label {
            color: var(--dm-muted);
            font-size: 12px;
            margin-bottom: 4px;
        }
        .dm-metric-value {
            font-size: 20px;
            font-weight: 700;
        }
        .dm-citation {
            border-left: 3px solid var(--dm-accent);
            background: #ffffff;
            padding: 10px 12px;
            margin: 8px 0;
            border-radius: 6px;
            border-top: 1px solid var(--dm-border);
            border-right: 1px solid var(--dm-border);
            border-bottom: 1px solid var(--dm-border);
        }
        .dm-small {
            color: var(--dm-muted);
            font-size: 13px;
        }
        .stButton > button {
            border-radius: 6px;
            border: 1px solid var(--dm-accent);
            background: var(--dm-accent);
            color: white;
            font-weight: 600;
        }
        .stButton > button:hover {
            border-color: var(--dm-accent-dark);
            background: var(--dm-accent-dark);
            color: white;
        }
        [data-testid="stChatMessage"] {
            background: transparent;
            border-radius: 8px;
        }
        textarea, input {
            border-radius: 8px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    st.session_state.setdefault("documents", [])
    st.session_state.setdefault("parsed_document", None)
    st.session_state.setdefault("cleaned_document", None)
    st.session_state.setdefault("chunks", [])
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_answer", None)


def render_header() -> None:
    st.markdown(
        """
        <div class="dm-header">
            <div>
                <div class="dm-title">DocMind AI</div>
                <div class="dm-subtitle">Professional RAG validation workspace for document intelligence</div>
            </div>
            <div class="dm-badge">Local validation UI</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def save_and_process_upload(uploaded_file, chunk_size: int, overlap: int) -> None:
    upload_service = UploadService(LocalFileStorage("uploads"))
    document = asyncio.run(upload_service.save_pdf(StreamlitUploadFile(uploaded_file)))
    parsed = ParsingService().parse_document(document)
    cleaned = PreprocessingService().clean_document(parsed)
    chunks = ChunkingService(ChunkingConfig(max_words=chunk_size, overlap_words=overlap)).chunk_document(cleaned)

    st.session_state.documents.append(document)
    st.session_state.parsed_document = parsed
    st.session_state.cleaned_document = cleaned
    st.session_state.chunks = chunks


def render_sidebar() -> tuple[int, int]:
    with st.sidebar:
        st.markdown("## DocMind AI")
        st.caption("Validation console")
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

        st.markdown("### Chunking")
        chunk_size = st.slider("Chunk size", min_value=80, max_value=400, value=180, step=20)
        overlap = st.slider("Overlap", min_value=0, max_value=120, value=40, step=10)

        if uploaded_file and st.button("Process document", use_container_width=True):
            try:
                save_and_process_upload(uploaded_file, chunk_size, overlap)
                st.success("Document processed.")
            except Exception as exc:
                st.error(str(exc))

        st.markdown("### Provider")
        st.selectbox("LLM mode", ["Mock validation", "Gemini later", "OpenAI later"], index=0)
        st.caption("Keys belong in backend/.env, not in the UI.")

        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_answer = None

    return chunk_size, overlap


def render_metrics() -> None:
    parsed = st.session_state.parsed_document
    chunks = st.session_state.chunks
    pages = len(parsed.pages) if parsed else 0
    words = sum(page.word_count for page in parsed.pages) if parsed else 0

    cols = st.columns(4)
    values = [
        ("Documents", len(st.session_state.documents)),
        ("Pages", pages),
        ("Words", words),
        ("Chunks", len(chunks)),
    ]
    for col, (label, value) in zip(cols, values):
        with col:
            st.markdown(
                f"""
                <div class="dm-panel">
                    <div class="dm-metric-label">{label}</div>
                    <div class="dm-metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_pipeline_inspector() -> None:
    with st.expander("Pipeline inspector", expanded=False):
        parsed = st.session_state.parsed_document
        cleaned = st.session_state.cleaned_document
        chunks = st.session_state.chunks

        if not parsed:
            st.info("Upload and process a PDF to inspect parsing, cleaning, and chunking.")
            return

        left, right = st.columns(2)
        with left:
            st.markdown("#### Parsed pages")
            for page in parsed.pages[:3]:
                st.text_area(
                    f"Page {page.page_number} raw text",
                    page.text[:1200],
                    height=140,
                    key=f"raw-{page.page_number}",
                )
        with right:
            st.markdown("#### Cleaned pages")
            for page in cleaned.pages[:3]:
                st.text_area(
                    f"Page {page.page_number} cleaned text",
                    page.text[:1200],
                    height=140,
                    key=f"clean-{page.page_number}",
                )

        st.markdown("#### First chunks")
        for chunk in chunks[:5]:
            st.markdown(
                f"""
                <div class="dm-citation">
                    <strong>{chunk.chunk_id}</strong><br>
                    <span class="dm-small">{chunk.source}</span><br>
                    {chunk.text[:500]}
                </div>
                """,
                unsafe_allow_html=True,
            )


def choose_mock_context(question: str):
    chunks = st.session_state.chunks
    if not chunks:
        return []

    terms = {term.lower() for term in question.split() if len(term) > 2}
    scored = []
    for chunk in chunks:
        text = chunk.text.lower()
        score = sum(1 for term in terms if term in text)
        scored.append((score, chunk))

    ranked = [chunk for score, chunk in sorted(scored, key=lambda item: item[0], reverse=True) if score > 0]
    return ranked[:3] or chunks[:3]


def render_chat() -> None:
    st.markdown("### Chat")
    if not st.session_state.chunks:
        st.info("Upload and process a PDF first. The chat will use the processed chunks as document context.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about your uploaded document")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    retrieved_chunks = choose_mock_context(question)
    answer_service = AnswerService(
        PromptService(),
        LLMService(
            MockLLMProvider(
                "This validation response is generated from the local mock LLM. "
                "The production answer will use the configured provider and retrieved document context."
            )
        ),
    )
    history = [
        ChatMessage(role=message["role"], content=message["content"])
        for message in st.session_state.messages[-6:]
        if message["role"] in {"user", "assistant"}
    ]
    answer = answer_service.generate_answer(question, retrieved_chunks, history)
    st.session_state.last_answer = answer
    st.session_state.messages.append({"role": "assistant", "content": answer.answer})

    with st.chat_message("assistant"):
        st.markdown(answer.answer)
        if answer.citations:
            st.markdown("**Sources**")
            for citation in answer.citations:
                st.markdown(f'<div class="dm-citation">{citation}</div>', unsafe_allow_html=True)


def main() -> None:
    configure_page()
    init_state()
    render_sidebar()
    render_header()
    render_metrics()
    render_chat()
    render_pipeline_inspector()


if __name__ == "__main__":
    main()
