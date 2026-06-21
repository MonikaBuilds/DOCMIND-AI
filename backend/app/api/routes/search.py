from fastapi import APIRouter, HTTPException, status

from app.core.exceptions import DocMindError
from app.core.providers import build_embedding_provider, build_vector_store
from app.core.runtime_store import runtime_store
from app.schemas.search import SearchRequest, SearchResponse, SearchResultResponse
from app.services.embedding_service import EmbeddingService
from app.services.page_query_service import PageQueryService
from app.services.retrieval_service import RetrievalService
from app.services.vector_service import VectorService

router = APIRouter(prefix="/search", tags=["search"])


@router.get("/status")
def search_module_status() -> dict[str, str]:
    return {
        "module": "search",
        "status": "keyword_api_ready",
        "next_step": "Vector-backed semantic search is enabled after embeddings and Chroma are configured.",
    }


@router.post("", response_model=SearchResponse)
def search_documents(request: SearchRequest) -> SearchResponse:
    chunks = runtime_store.get_chunks(request.document_ids)
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No processed chunks available. Upload and process a document first.",
        )

    try:
        page_query_service = PageQueryService()
        page_number = page_query_service.extract_page_number(request.query)
        if page_number is not None:
            results = page_query_service.chunks_for_page(chunks, page_number, top_k=request.top_k)
        else:
            results = RetrievalService(
                EmbeddingService(build_embedding_provider()),
                VectorService(build_vector_store()),
            ).semantic_search(
                request.query,
                top_k=request.top_k,
                document_ids=request.document_ids,
            )
    except DocMindError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    return SearchResponse(
        results=[
            SearchResultResponse(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                filename=result.filename,
                text=result.text,
                score=result.score,
                page_number=result.page_number,
                source=result.source,
                heading=result.heading,
            )
            for result in results
        ],
        count=len(results),
    )
