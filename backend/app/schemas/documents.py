from pydantic import BaseModel, Field


class ProcessDocumentRequest(BaseModel):
    chunk_size: int = Field(default=180, ge=20, le=1000)
    overlap: int = Field(default=40, ge=0, le=300)


class DocumentRecordResponse(BaseModel):
    document_id: str
    filename: str
    content_type: str
    size_bytes: int
    storage_path: str
    status: str
    page_count: int | None = None
    chunk_count: int = 0


class ProcessDocumentResponse(BaseModel):
    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    status: str
