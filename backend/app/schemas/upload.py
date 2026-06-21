from pydantic import BaseModel


class UploadedDocumentResponse(BaseModel):
    document_id: str
    original_filename: str
    stored_filename: str
    content_type: str
    size_bytes: int
    storage_path: str
    status: str


class MultiUploadResponse(BaseModel):
    documents: list[UploadedDocumentResponse]
    count: int
