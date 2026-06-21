from fastapi import APIRouter, HTTPException, status

from app.core.exceptions import DocMindError
from app.core.providers import build_embedding_provider, build_llm_provider, build_vector_store
from app.core.runtime_store import runtime_store
from app.schemas.chat import ChatRequest, ChatResponse, CitationResponse
from app.services.answer_service import AnswerService
from app.services.document_query_service import DocumentQueryService
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.memory_service import ConversationMemoryService
from app.services.page_query_service import PageQueryService
from app.services.prompt_service import PromptService
from app.services.metadata_service import MetadataService
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
        structured_retrieval = False
        document_query_service = DocumentQueryService()
        page_query_service = PageQueryService()
        metadata_service = MetadataService()
        active_document_ids = request.document_ids or [doc.document.document_id for doc in runtime_store.list_records()]
        page_number = page_query_service.extract_page_number(request.question)
        summaries = []
        for doc_id in active_document_ids:
            record = runtime_store.get_record(doc_id)
            if record and record.parsed_document:
                summaries.append(metadata_service.document_summary(record.parsed_document, record.chunks))

        if page_number is not None:
            structured_retrieval = True
            retrieved = page_query_service.chunks_for_page(chunks, page_number, top_k=request.top_k)
            if not retrieved:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"No prepared content found for page {page_number} in the selected document.",
                )
        elif document_query_service.is_meta_question(request.question):
            structured_retrieval = True
            # For meta questions, we don't necessarily need chunks, but we can provide overview chunks to be safe
            retrieved = document_query_service.overview_chunks(chunks, top_k=5)
        elif document_query_service.is_overview_question(request.question):
            structured_retrieval = True
            retrieved = document_query_service.overview_chunks(chunks, top_k=max(request.top_k, 8))
        else:
            embedding_service = EmbeddingService(build_embedding_provider())
            vector_service = VectorService(build_vector_store())
            retrieval_service = RetrievalService(embedding_service, vector_service)
            retrieved = retrieval_service.hybrid_search(
                request.question,
                chunks,
                top_k=request.top_k,
                document_ids=request.document_ids,
            )
        reranked = retrieved if structured_retrieval else RerankingService().rerank(request.question, retrieved, top_k=request.top_k)

        memory_service.append_message(request.session_id, "user", request.question)
        history = memory_service.get_history(request.session_id)
        answer_service = AnswerService(
            PromptService(),
            LLMService(build_llm_provider()),
        )
        answer = answer_service.generate_answer(request.question, reranked, summaries, history)
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
