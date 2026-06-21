from fastapi import APIRouter

from app.domain.retrieval import RetrievedChunk
from app.schemas.evaluation import RetrievalEvaluationRequest, RetrievalEvaluationResponse
from app.services.evaluation_service import EvaluationService

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.get("/status")
def evaluation_module_status() -> dict[str, str]:
    return {
        "module": "rag_evaluation",
        "status": "implemented",
        "metrics": "Recall@K, Precision@K, hit rate, citation accuracy, groundedness, faithfulness",
    }


@router.post("/retrieval", response_model=RetrievalEvaluationResponse)
def evaluate_retrieval(request: RetrievalEvaluationRequest) -> RetrievalEvaluationResponse:
    retrieved_chunks = [
        RetrievedChunk(
            chunk_id=chunk_id,
            document_id="evaluation",
            filename="evaluation",
            text="",
            score=1.0,
            page_number=index + 1,
            source="evaluation",
            heading=None,
        )
        for index, chunk_id in enumerate(request.retrieved_chunk_ids)
    ]
    result = EvaluationService().evaluate_retrieval(
        expected_chunk_ids=set(request.expected_chunk_ids),
        retrieved_chunks=retrieved_chunks,
        k=request.k,
    )
    return RetrievalEvaluationResponse(
        recall_at_k=result.recall_at_k,
        precision_at_k=result.precision_at_k,
        hit_rate=result.hit_rate,
        expected_count=result.expected_count,
        retrieved_count=result.retrieved_count,
        matched_count=result.matched_count,
    )
