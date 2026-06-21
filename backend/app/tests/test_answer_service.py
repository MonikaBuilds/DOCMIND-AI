from app.domain.chat import ChatMessage, GroundedPrompt
from app.domain.retrieval import RetrievedChunk
from app.infrastructure.llm.base import MockLLMProvider
from app.services.answer_service import AnswerService
from app.services.llm_service import LLMService
from app.services.prompt_service import PromptService


def make_chunk(
    chunk_id: str,
    page_number: int,
    heading: str | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc",
        filename="answers.pdf",
        text="RAG answers should be grounded in retrieved context.",
        score=0.9,
        page_number=page_number,
        source=f"answers.pdf#page={page_number}",
        heading=heading,
    )


def test_generate_answer_uses_prompt_and_llm_services():
    provider = MockLLMProvider("RAG uses retrieved context. [answers.pdf, page 2]")
    service = AnswerService(PromptService(), LLMService(provider))
    chunks = [make_chunk("doc:p2:c0", page_number=2)]

    answer = service.generate_answer("How does RAG answer?", chunks)

    assert answer.answer == "RAG uses retrieved context. [answers.pdf, page 2]"
    assert answer.citations == ["answers.pdf, page 2"]
    assert len(provider.prompts) == 1
    assert isinstance(provider.prompts[0], GroundedPrompt)
    assert "Document Context:" in provider.prompts[0].user_prompt


def test_generate_answer_passes_conversation_history_into_prompt():
    provider = MockLLMProvider("Follow-up answer.")
    service = AnswerService(PromptService(), LLMService(provider))
    history = [ChatMessage(role="user", content="Earlier question")]

    service.generate_answer("Follow-up?", [make_chunk("doc:p1:c0", 1)], history)

    assert "User: Earlier question" in provider.prompts[0].user_prompt


def test_generate_answer_deduplicates_citations():
    provider = MockLLMProvider("Answer.")
    service = AnswerService(PromptService(), LLMService(provider))
    chunks = [
        make_chunk("doc:p1:c0", page_number=1, heading="Intro"),
        make_chunk("doc:p1:c1", page_number=1, heading="Intro"),
        make_chunk("doc:p2:c2", page_number=2, heading=None),
    ]

    answer = service.generate_answer("Question?", chunks)

    assert answer.citations == [
        "answers.pdf, page 1, Intro",
        "answers.pdf, page 2",
    ]


def test_generate_answer_allows_no_retrieved_chunks():
    provider = MockLLMProvider("The provided documents do not contain enough information to answer this.")
    service = AnswerService(PromptService(), LLMService(provider))

    answer = service.generate_answer("Unknown?", [])

    assert answer.answer.startswith("The provided documents do not contain enough information")
    assert answer.citations == []
