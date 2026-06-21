from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    filename: str
    text: str
    score: float
    page_number: int
    source: str
    heading: str | None = None
