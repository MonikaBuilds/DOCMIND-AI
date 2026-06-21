from fastapi import APIRouter, HTTPException, status

from app.core.exceptions import DocMindError
from app.core.providers import build_embedding_provider, build_llm_provider, build_vector_store
from app.core.runtime_store import runtime_store
from app.schemas.chat import ChatRequest, ChatResponse, CitationResponse
from app.services.answer_service import AnswerService
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.memory_service import ConversationMemoryService
from app.services.prompt_service import PromptService
from app.services.reranking_service import RerankingService
from app.services.retrieval_service import RetrievalService
from app.services.vector_service import VectorService

router = APIRouter(prefix="/chat", tags=["chat"])
memory_service = ConversationMemoryService()


@router.get("/status")
def chat_module_status() -> dict[str, str]:
    return {
        "module": "chat",
        "status": "mock_api_ready",
        "next_step": "Configure a real LLM provider for production answers.",
    }


@router.post("", response_model=ChatResponse)
def chat_with_documents(request: ChatRequest) -> ChatResponse:
    chunks = runtime_store.get_chunks(request.document_ids)
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No processed chunks available. Upload and process a document first.",
        )

    try:
        embedding_service = EmbeddingService(build_embedding_provider())
        vector_service = VectorService(build_vector_store())
        retrieval_service = RetrievalService(embedding_service, vector_service)
        retrieved = retrieval_service.semantic_search(request.question, top_k=request.top_k)
        reranked = RerankingService().rerank(request.question, retrieved, top_k=request.top_k)

        memory_service.append_message(request.session_id, "user", request.question)
        history = memory_service.get_history(request.session_id)
        answer_service = AnswerService(
            PromptService(),
            LLMService(build_llm_provider()),
        )
        answer = answer_service.generate_answer(request.question, reranked, history)
    except DocMindError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    memory_service.append_message(request.session_id, "assistant", answer.answer)

    return ChatResponse(
        answer=answer.answer,
        citations=[
            CitationResponse(label=citation)
            for citation in answer.citations
        ],
        retrieved_count=len(reranked),
    )
