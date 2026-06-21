from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    session_id: str = Field(default="default", min_length=1)
    document_ids: list[str] | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class CitationResponse(BaseModel):
    label: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]
    retrieved_count: int
