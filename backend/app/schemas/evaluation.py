from pydantic import BaseModel, Field


class RetrievalEvaluationRequest(BaseModel):
    expected_chunk_ids: list[str]
    retrieved_chunk_ids: list[str]
    k: int = Field(default=5, ge=1)


class RetrievalEvaluationResponse(BaseModel):
    recall_at_k: float
    precision_at_k: float
    hit_rate: float
    expected_count: int
    retrieved_count: int
    matched_count: int
