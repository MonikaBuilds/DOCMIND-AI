from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    document_id: str
    filename: str
    text: str
    page_number: int
    chunk_index: int
    source: str
    heading: str | None = None
