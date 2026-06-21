from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentMetadataSummary:
    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    title: str | None = None
    author: str | None = None
