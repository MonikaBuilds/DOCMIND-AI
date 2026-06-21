from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    document_ids: list[str] | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class SearchResultResponse(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    text: str
    score: float
    page_number: int
    source: str
    heading: str | None = None


class SearchResponse(BaseModel):
    results: list[SearchResultResponse]
    count: int
