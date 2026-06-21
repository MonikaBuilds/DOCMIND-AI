from app.domain.chunks import DocumentChunk
from app.domain.retrieval import RetrievedChunk


class DocumentQueryService:
    """Routes document-level questions to representative document context."""

    def is_overview_question(self, query: str) -> bool:
        normalized = " ".join(query.lower().strip().split())
        overview_patterns = [
            "what is this document about",
            "what's this document about",
            "what this document is about",
            "this document is about",
            "document is about",
            "document about",
            "what is this pdf about",
            "what's this pdf about",
            "what this pdf is about",
            "this pdf is about",
            "pdf is about",
            "pdf about",
            "file is about",
            "file about",
            "what is the document about",
            "what's the document about",
            "what is the pdf about",
            "what's the pdf about",
            "tell me about this document",
            "tell me about this pdf",
            "tell about this document",
            "tell about this pdf",
            "about this document",
            "about this pdf",
            "summarize this document",
            "summarize this pdf",
            "summarize the document",
            "summarize the pdf",
            "summary of this document",
            "summary of this pdf",
            "document summary",
            "pdf summary",
            "main idea",
            "main topic",
            "what is the topic",
            "what topic",
            "overview",
        ]
        return any(pattern in normalized for pattern in overview_patterns)

    def overview_chunks(
        self,
        chunks: list[DocumentChunk],
        top_k: int = 8,
    ) -> list[RetrievedChunk]:
        ordered_chunks = sorted(chunks, key=lambda chunk: (chunk.page_number, chunk.chunk_index))
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
            for chunk in ordered_chunks[:top_k]
        ]
