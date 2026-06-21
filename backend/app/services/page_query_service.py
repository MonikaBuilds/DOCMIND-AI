import re

from app.domain.chunks import DocumentChunk
from app.domain.retrieval import RetrievedChunk


class PageQueryService:
    """Routes page-specific questions to page metadata instead of semantic search."""

    page_pattern = re.compile(r"\b(?:page|pg|p)\.?\s*(?:number|no\.?)?\s*(\d{1,4})\b", re.IGNORECASE)

    def extract_page_number(self, query: str) -> int | None:
        match = self.page_pattern.search(query)
        if not match:
            return None

        page_number = int(match.group(1))
        return page_number if page_number > 0 else None

    def chunks_for_page(
        self,
        chunks: list[DocumentChunk],
        page_number: int,
        top_k: int = 8,
    ) -> list[RetrievedChunk]:
        page_chunks = [chunk for chunk in chunks if chunk.page_number == page_number]
        return [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                filename=chunk.filename,
                text=chunk.text,
                score=1.0,
                page_number=chunk.page_number,
                source=chunk.source,
                heading=chunk.heading,
            )
            for chunk in page_chunks[:top_k]
        ]
