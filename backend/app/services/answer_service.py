from app.domain.chat import ChatMessage, CitedAnswer
from app.domain.metadata import DocumentMetadataSummary
from app.domain.retrieval import RetrievedChunk
from app.services.citation_service import CitationService
from app.services.llm_service import LLMService
from app.services.prompt_service import PromptService


class AnswerService:
    """Generates grounded answers from retrieved context."""

    def __init__(
        self,
        prompt_service: PromptService,
        llm_service: LLMService,
        citation_service: CitationService | None = None,
    ) -> None:
        self.prompt_service = prompt_service
        self.llm_service = llm_service
        self.citation_service = citation_service or CitationService()

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: list[RetrievedChunk],
        summaries: list[DocumentMetadataSummary] | None = None,
        conversation_history: list[ChatMessage] | None = None,
    ) -> CitedAnswer:
        prompt = self.prompt_service.build_grounded_prompt(
            question=question,
            retrieved_chunks=retrieved_chunks,
            summaries=summaries,
            conversation_history=conversation_history,
        )
        answer = self.llm_service.generate(prompt)
        citations = self.citation_service.citation_labels(retrieved_chunks)
        return CitedAnswer(answer=answer, citations=citations)
