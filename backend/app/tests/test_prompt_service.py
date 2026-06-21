import pytest

from app.domain.chat import ChatMessage
from app.domain.retrieval import RetrievedChunk
from app.services.prompt_service import PromptService


def make_chunk(
    chunk_id: str = "doc:p1:c0",
    text: str = "RAG retrieves document chunks before generating an answer.",
    heading: str | None = "RAG Basics",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc",
        filename="rag.pdf",
        text=text,
        score=0.87654,
        page_number=4,
        source="rag.pdf#page=4",
        heading=heading,
    )


def test_build_grounded_prompt_contains_strict_grounding_rules():
    prompt = PromptService().build_grounded_prompt("What is RAG?", [make_chunk()])

    assert "Answer strictly using the provided document context" in prompt.system_prompt
    assert "Do not use outside knowledge" in prompt.system_prompt
    assert "Use only the document context as factual evidence" in prompt.user_prompt
    assert "The provided documents do not contain enough information" in prompt.user_prompt


def test_build_grounded_prompt_formats_context_with_citation():
    prompt = PromptService().build_grounded_prompt("What is RAG?", [make_chunk()])

    assert "[Source 1]" in prompt.user_prompt
    assert "Citation: rag.pdf, page 4, RAG Basics" in prompt.user_prompt
    assert "Score: 0.8765" in prompt.user_prompt
    assert "RAG retrieves document chunks" in prompt.user_prompt


def test_build_grounded_prompt_includes_history_as_secondary_context():
    history = [
        ChatMessage(role="user", content="Summarize chapter 1."),
        ChatMessage(role="assistant", content="It discusses retrieval."),
    ]

    prompt = PromptService().build_grounded_prompt(
        "What does it say next?",
        [make_chunk()],
        conversation_history=history,
    )

    assert "Conversation History:" in prompt.user_prompt
    assert "User: Summarize chapter 1." in prompt.user_prompt
    assert "Assistant: It discusses retrieval." in prompt.user_prompt
    assert "Conversation history is only for understanding follow-up questions" in prompt.user_prompt


def test_build_grounded_prompt_handles_no_retrieved_context():
    prompt = PromptService().build_grounded_prompt("What is missing?", [])

    assert "No document context was retrieved." in prompt.user_prompt


def test_build_grounded_prompt_rejects_empty_question():
    with pytest.raises(ValueError):
        PromptService().build_grounded_prompt("   ", [make_chunk()])
